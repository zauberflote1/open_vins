/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "TrackOCL.h"

#include <algorithm>
#include <unordered_set>
#include "../Grider_FAST.h"
#include "../Grider_GRID.h"
#include "Grider_OCL.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"

using namespace ov_core;

int test_feed_all = 1;

void TrackOCL::feed_new_camera(const CameraData &message)
{

    // Error check that we have all the data
    if (message.sensor_ids.empty() ||
        (message.sensor_ids.size() != message.images.size()) ||
        (message.images.size() != message.masks.size()))
    {
        PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
        PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
        PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
        PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
        std::exit(EXIT_FAILURE);
    }

    // Preprocessing steps that we do not parallelize
    // NOTE: DO NOT PARALLELIZE THESE!
    // NOTE: These seem to be much slower if you parallelize them...
    rT1 = boost::posix_time::microsec_clock::local_time();

    size_t num_images = message.images.size();

    for (size_t msg_id = 0; msg_id < num_images; msg_id++)
    {
        // Lock this data feed for this camera
        size_t cam_id = message.sensor_ids.at(msg_id);
        std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

        // Histogram equalize
        cv::Mat img;
        img = message.images.at(msg_id);

        // Extract image pyramid
        std::vector<cv::Mat> imgpyr;
        imgpyr.push_back(img);

        // Save!
        img_curr[cam_id] = img;
        img_pyramid_curr[cam_id] = imgpyr;
    }

    if (test_feed_all)
    {
        feed_batch_monocular(message);
    }
    else
    {
        // Either call our stereo or monocular version
        // If we are doing binocular tracking, then we should parallize our tracking
        if (num_images == 1)
        {
            feed_monocular(message, 0);
        }
        else if (!use_stereo)
        {
            // NOTE: opencv::parallel_for() seems to be less efficient than direct loop
            for (int i = 0; i < (int)num_images; i++)
            {
                feed_monocular(message, i);
            }
        }
        else
        {
            PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, we only support mono or stereo tracking", num_images);
            std::exit(EXIT_FAILURE);
        }
    }
}

void TrackOCL::feed_batch_monocular(const CameraData &message)
{
    // initial check
    size_t N = message.images.size();

    // always build the next pyramids
    modal_flow::PyramidBatch pyr_batch;
    std::vector<cv::Mat> float_img(N);

    for (size_t msg_id = 0; msg_id < N; msg_id++)
    {
        size_t cam_id = message.sensor_ids[msg_id];
        std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);

        message.images[msg_id].convertTo(float_img[msg_id], CV_32F);

        modal_flow::ImageView iv{{std::get<0>(dims), std::get<1>(dims),
                                  modal_flow::PixelFormat::R32F, float_img[msg_id].step},
                                 float_img[msg_id].data,
                                 modal_flow::ExternalType::None,
                                 0};
        modal_flow::Frame f({cam_id, 0, iv});
        pyr_batch.frames.push_back(f);
    }

    mgr_.build_next_img_pyramids(pyr_batch);

    std::vector<int> to_track(N, 0);
    std::vector<int> to_detect(N, 0); // holds 1 if action is to be taken, 0 otherwise

    modal_flow::DetectOptions dopt;
    dopt.max_features = -1;
    dopt.threshold = 20.f;

    modal_flow::DetectionBatch detect_batch;
    detect_batch.options = dopt;

    std::vector<cv::Mat> grid_2d_close(N);
    std::vector<cv::Mat> detection_mask(N);                      // holds msg_id's detection mask
    std::vector<std::vector<std::pair<int, int>>> valid_locs(N); // holds msg_id's detection grid cells

    for (size_t msg_id = 0; msg_id < N; msg_id++)
    {
        size_t cam_id = message.sensor_ids[msg_id];
        std::lock_guard<std::mutex> lck(mtx_feeds[cam_id]);

        cv::Mat mask = message.masks.at(msg_id);
        cv::Mat img = message.images[msg_id];

        // If has prior points, do tracking
        to_track[msg_id] = !pts_last[cam_id].empty() ? 1 : 0;

        std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);

        auto pts_left_old = pts_last[cam_id];
        auto ids_left_old = ids_last[cam_id];

        // check if any grid cells need to be refilled
        valid_locs[msg_id] = get_grids_to_fill(std::get<0>(dims), std::get<1>(dims), mask, detection_mask[msg_id], grid_2d_close[msg_id], pts_left_old, ids_left_old);

        // if we need to find points, add the frame to the detection batch
        printf("cam_id: %d, pts: %d, valid_locs: %d\n", cam_id, pts_left_old.size(), valid_locs[msg_id].size());
        if (!valid_locs[msg_id].empty())
        {
            modal_flow::ImageView iv{{std::get<0>(dims), std::get<1>(dims), modal_flow::PixelFormat::R8, img.step}, img.data, modal_flow::ExternalType::None, 0};
            modal_flow::Frame f({cam_id, 0, iv});
            detect_batch.frames.push_back(f);
            to_detect[msg_id] = 1;
        }
        else
        {
            to_detect[msg_id] = 0;
        }
    }

    // run the detection
    auto results = mgr_.run_detection(detect_batch);

    // process the detection results, and setup the tracking batch
    int job_idx = 0;
    for (size_t msg_id = 0; msg_id < N; msg_id++)
    {
        if (!to_detect[msg_id])
            continue;

        size_t cam_id = message.sensor_ids[msg_id];
        std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);

        auto pts_left_old = pts_last[cam_id];
        auto ids_left_old = ids_last[cam_id];

        int pts_before = pts_left_old.size();

        process_batch_detection_result(std::get<0>(dims), std::get<1>(dims), cam_id, detection_mask[msg_id], grid_2d_close[msg_id], valid_locs[msg_id], results[job_idx], pts_left_old, ids_left_old);
        job_idx++;
        printf("cam_id: %d, pts_before: %d, pts after: %d\n", cam_id, pts_before, pts_left_old.size());

        // if (pts_last[cam_id].empty())
        // {
        pts_last[cam_id] = pts_left_old;
        ids_last[cam_id] = ids_left_old;
        // printf("cam: %d, pts_last adding: %d\n", cam_id, pts_left_old.size());
        // }
    }
    printf("processed detection results\n");

    std::vector<std::vector<uchar>> mask_ll(N);
    std::vector<std::vector<cv::KeyPoint>> pts_left_new(N);
    std::vector<int> too_few_pts(N);
    for (size_t msg_id = 0; msg_id < N; msg_id++)
    {
        if (to_track[msg_id])
        {
            size_t cam_id = message.sensor_ids[msg_id];
            pts_left_new[msg_id] = pts_last[cam_id];
        }
    }

    modal_flow::TrackingBatch track_batch;
    modal_flow::TrackOptions topt;

    printf("making tracking batch\n");

    job_idx = 0;
    for (size_t msg_id = 0; msg_id < N; msg_id++)
    {
        printf("msg_id: %d, to_track: %d\n", msg_id, to_track[msg_id]);
        if (!to_track[msg_id])
            continue;

        size_t cam_id = message.sensor_ids[msg_id];

        std::pair<int, int> dims = mgr_.get_cam_dim(cam_id);

        // account for not enough points for RANSAC
        if (pts_last[cam_id].size() < 10)
        {
            for (size_t i = 0; i < pts_last[cam_id].size(); i++)
            {
                mask_ll[msg_id].push_back((uchar)0);
            }
            too_few_pts[msg_id] = 1;
            to_track[msg_id] = 0;
            printf("cam_id: %d had too few pts: %d\n", cam_id, pts_last[cam_id].size());
            continue;
        }

        modal_flow::TrackInput track_in;
        for (const auto &point : pts_last[cam_id])
        {
            track_in.prev_points.push_back({point.pt.x, point.pt.y, 0.f});
        }
        modal_flow::ImageView iv{{std::get<0>(dims), std::get<1>(dims),
                                  modal_flow::PixelFormat::R32F, float_img[msg_id].step},
                                 float_img[msg_id].data,
                                 modal_flow::ExternalType::None,
                                 0};
        modal_flow::Frame f({cam_id, 0, iv});
        track_in.next_frame = f;
        track_batch.jobs.push_back(track_in);
        printf("pushing back batch, for msg: %d\n", msg_id);
    }
    printf("track_batch size: %d\n", track_batch.jobs.size());

    printf("running tracking\n");

    std::vector<modal_flow::TrackResult> track_results = mgr_.run_tracking(track_batch);

    printf("tracking ran\n");

    job_idx = 0;
    for (int i = 0; i < N; i++)
    {
        printf("msg_id: %d, to_track: %d\n", i, to_track[i]);
        if (to_track[i])
            printf("msg_id: %d, tracked count: %d\n", i, track_results[job_idx++].next_points.size());
    }

    job_idx = 0;
    for (size_t msg_id = 0; msg_id < N; ++msg_id)
    {
        if (!to_track[msg_id])
            continue;

        printf("should see a good points left print\n");

        size_t cam_id = message.sensor_ids[msg_id];
        cv::Mat img_mask = message.masks.at(msg_id);
        auto ids_left_old = ids_last[cam_id];

        auto [width, height] = mgr_.get_cam_dim(cam_id);

        auto &mask = mask_ll[msg_id];

        if (!too_few_pts[msg_id])
        {
            const auto &trk = track_results[job_idx];
            assert(trk.next_points.size() == trk.status.size());
            size_t n_pts = trk.next_points.size();
            // printf("cam_id: %d, track result n_pts: %d\n", cam_id, n_pts);

            // Build pts0/pts1 for RANSAC, undistort, etc...
            std::vector<cv::Point2f> pts0(n_pts), pts1(n_pts);
            for (int p = 0; p < n_pts; p++)
            {
                printf("p: %d, (%.3f, %.3f) -> (%.3f, %.3f)\n", p, track_batch.jobs[job_idx].prev_points[p].x,
                       track_batch.jobs[job_idx].prev_points[p].y,
                       trk.next_points[p].x,
                       trk.next_points[p].y);
                    cv::Point2f pt0 = {track_batch.jobs[job_idx].prev_points[p].x,
                                       track_batch.jobs[job_idx].prev_points[p].y};
                cv::Point2f pt1 = {trk.next_points[p].x,
                                   trk.next_points[p].y};
                pts0[p] = pt0;
                pts1[p] = pt1;
            }

            // undistort
            std::vector<cv::Point2f> u0, u1;
            for (size_t p = 0; p < n_pts; ++p)
            {
                u0.push_back(camera_calib.at(cam_id)->undistort_cv(pts0[p]));
                u1.push_back(camera_calib.at(cam_id)->undistort_cv(pts1[p]));
            }

            std::vector<uchar> mask_ransac;
            double max_focallength_img0 = std::max(camera_calib.at(cam_id)->get_K()(0, 0), camera_calib.at(cam_id)->get_K()(1, 1));
            double max_focallength_img1 = std::max(camera_calib.at(cam_id)->get_K()(0, 0), camera_calib.at(cam_id)->get_K()(1, 1));
            double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
            cv::findFundamentalMat(u0, u1, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_ransac);

            mask.resize(n_pts);
            for (size_t p = 0; p < n_pts; ++p)
            {
                // printf("p; %d, (%.4f, %.4f) status: %d\n", p, trk.next_points[p].x, trk.next_points[p].y, trk.status[p]);
                bool good = trk.status[p] && (p < mask_ransac.size() && mask_ransac[p]);
                mask[p] = good ? 1u : 0u;
            }

            pts_left_new[msg_id].resize(n_pts);
            for (size_t p = 0; p < n_pts; ++p)
            {
                pts_left_new[msg_id][p].pt = pts1[p];
            }
            job_idx++;
        }

        if (mask.empty())
        {
            std::lock_guard<std::mutex> lckv(mtx_last_vars);
            // img_last[cam_id] = img;
            // img_pyramid_last[cam_id] = imgpyr;
            img_mask_last[cam_id] = img_mask;
            pts_last[cam_id].clear();
            ids_last[cam_id].clear();
            continue;
        }

        // Get our "good tracks"
        std::vector<cv::KeyPoint> good_left;
        std::vector<size_t> good_ids_left;

        // Loop through all left points
        for (size_t p = 0; p < pts_left_new[msg_id].size(); p++)
        {
            // Ensure we do not have any bad KLT tracks (i.e., points are negative)
            if (pts_left_new[msg_id].at(p).pt.x < 0 || pts_left_new[msg_id].at(p).pt.y < 0 || (int)pts_left_new[msg_id].at(p).pt.x >= width ||
                (int)pts_left_new[msg_id].at(p).pt.y >= height)
                continue;
            // Check if it is in the mask
            // NOTE: mask has max value of 255 (white) if it should be
            if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new[msg_id].at(p).pt.y, (int)pts_left_new[msg_id].at(p).pt.x) > 127)
                continue;
            // If it is a good track, and also tracked from left to right
            if (mask[p])
            {
                good_left.push_back(pts_left_new[msg_id][p]);
                good_ids_left.push_back(ids_left_old[p]);
            }
        }

        // Update our feature database, with theses new observations
        for (size_t p = 0; p < good_left.size(); p++)
        {
            cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(p).pt);
            database->update_feature(good_ids_left.at(p), message.timestamp, cam_id, good_left.at(p).pt.x, good_left.at(p).pt.y, npt_l.x,
                                     npt_l.y);
        }

        // Move forward in time
        {
            std::lock_guard<std::mutex> lckv(mtx_last_vars);
            // img_last[cam_id] = img;
            // img_pyramid_last[cam_id] = imgpyr;
            img_mask_last[cam_id] = img_mask;
            printf("good points left; %d\n", good_left.size());
            pts_last[cam_id] = good_left;
            ids_last[cam_id] = good_ids_left;
        }
    }
}

void TrackOCL::feed_monocular(const CameraData &message, size_t msg_id)
{
    // Lock this data feed for this camera
    size_t cam_id = message.sensor_ids.at(msg_id);
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Get our image objects for this image
    cv::Mat img = img_curr.at(cam_id);
    std::vector<cv::Mat> imgpyr = img_pyramid_curr.at(cam_id);
    cv::Mat mask = message.masks.at(msg_id);
    rT2 = boost::posix_time::microsec_clock::local_time();

    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    if (pts_last[cam_id].empty())
    {
        // Detect new features
        std::vector<cv::KeyPoint> good_left;
        std::vector<size_t> good_ids_left;
        perform_detection_monocular(imgpyr, mask, good_left, good_ids_left, cam_id);

        // Save the current image and pyramid
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_pyramid_last[cam_id] = imgpyr;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;

        cv::Mat img_float;
        message.images[msg_id].convertTo(img_float, CV_32F);
        mgr.getTracker(cam_id)->buildNextPyramid(img_float.data);
        return;
    }

    cv::Mat img_float;
    message.images[msg_id].convertTo(img_float, CV_32F);

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    auto pts_left_old = pts_last[cam_id];
    auto ids_left_old = ids_last[cam_id];
    perform_detection_monocular(img_pyramid_last[cam_id], img_mask_last[cam_id], pts_left_old, ids_left_old, cam_id);
    rT3 = boost::posix_time::microsec_clock::local_time();

    mgr.getTracker(cam_id)->swapPyramids();
    mgr.getTracker(cam_id)->buildNextPyramid(img_float.data);

    // Our return success masks, and predicted new features
    std::vector<uchar> mask_ll;
    std::vector<cv::KeyPoint> pts_left_new = pts_left_old;

    // Lets track temporally
    perform_matching(img_pyramid_last[cam_id], imgpyr, pts_left_old, pts_left_new, cam_id, cam_id, mask_ll);
    assert(pts_left_new.size() == ids_left_old.size());
    rT4 = boost::posix_time::microsec_clock::local_time();

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_ll.empty())
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_pyramid_last[cam_id] = imgpyr;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id].clear();
        ids_last[cam_id].clear();
        PRINT_ERROR(RED "[KLT-EXTRACTOR]: Failed to get enough points to do RANSAC, resetting.....\n" RESET);
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;

    // Loop through all left points
    for (size_t i = 0; i < pts_left_new.size(); i++)
    {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_left_new.at(i).pt.x < 0 || pts_left_new.at(i).pt.y < 0 || (int)pts_left_new.at(i).pt.x >= img.cols ||
            (int)pts_left_new.at(i).pt.y >= img.rows)
            continue;
        // Check if it is in the mask
        // NOTE: mask has max value of 255 (white) if it should be
        if ((int)message.masks.at(msg_id).at<uint8_t>((int)pts_left_new.at(i).pt.y, (int)pts_left_new.at(i).pt.x) > 127)
            continue;
        // If it is a good track, and also tracked from left to right
        if (mask_ll[i])
        {
            good_left.push_back(pts_left_new[i]);
            good_ids_left.push_back(ids_left_old[i]);
        }
    }

    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_left.size(); i++)
    {
        cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
        database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                                 npt_l.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lckv(mtx_last_vars);
        img_last[cam_id] = img;
        img_pyramid_last[cam_id] = imgpyr;
        img_mask_last[cam_id] = mask;
        pts_last[cam_id] = good_left;
        ids_last[cam_id] = good_ids_left;
    }
    rT5 = boost::posix_time::microsec_clock::local_time();

    //  // Timing information
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for pyramid\n", (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for detection\n", (rT3 - rT2).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for temporal klt\n", (rT4 - rT3).total_microseconds() * 1e-6);
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
                (int)good_left.size());
    PRINT_DEBUG("[TIME-KLT]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

std::vector<std::pair<int, int>> TrackOCL::get_grids_to_fill(int width, int height,
                                                             const cv::Mat &mask0,
                                                             cv::Mat &mask0_updated, // used for detection
                                                             cv::Mat &grid_2d_close,
                                                             std::vector<cv::KeyPoint> &pts0,
                                                             std::vector<size_t> &ids0)
{
    cv::Size size_close((int)((float)width / (float)min_px_dist),
                        (int)((float)height / (float)min_px_dist)); // width x height
    grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
    float size_x = (float)width / (float)grid_x;
    float size_y = (float)height / (float)grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
    mask0_updated = mask0.clone();
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end())
    {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= width - edge || y < edge || y >= height - edge)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x);
        int y_grid = std::floor(kpt.pt.y / size_y);
        if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask0.at<uint8_t>(y, x) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255)
        {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < width && y - min_px_dist >= 0 && y + min_px_dist < height)
        {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255));
        }
        it0++;
        it1++;
    }

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    std::vector<std::pair<int, int>> valid_locs;

    double min_feat_percent = 0.50;
    int num_featsneeded = num_features - (int)pts0.size();
    if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
        return valid_locs;

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    for (int x = 0; x < grid_2d_grid.cols; x++)
    {
        for (int y = 0; y < grid_2d_grid.rows; y++)
        {
            if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255)
            {
                valid_locs.emplace_back(x, y);
            }
        }
    }

    return valid_locs;
}

void TrackOCL::process_batch_detection_result(int width, int height, int cam_id,
                                              const cv::Mat &detection_mask,
                                              cv::Mat grid_2d_close,
                                              std::vector<std::pair<int, int>> valid_locs,
                                              modal_flow::DetectResult detect_res,
                                              std::vector<cv::KeyPoint> &pts0,
                                              std::vector<size_t> &ids0)
{
    auto pack_cell = [&](int cx, int cy)
    {
        return size_t(cx) * size_t(grid_y) + size_t(cy);
    };
    // compute cell sizes
    float size_x = float(width) / float(grid_x);
    float size_y = float(height) / float(grid_y);

    int num_features_grid = int(double(num_features) / double(grid_x * grid_y)) + 1;

    // build a set of valid cells
    std::unordered_set<size_t> valid_set;
    valid_set.reserve(valid_locs.size());
    for (auto &c : valid_locs)
    {
        valid_set.insert(pack_cell(c.first, c.second));
    }

    // sort detections by score descending
    std::sort(detect_res.keypoints.begin(),
              detect_res.keypoints.end(),
              [](auto &a, auto &b)
              { return a.score > b.score; });

    // count how many we've picked per cell
    std::unordered_map<size_t, int> picked_count;
    picked_count.reserve(valid_locs.size());

    std::vector<cv::KeyPoint> new_pts;
    new_pts.reserve(detect_res.keypoints.size());

    // printf("cam_id: %d, keypoints: %d\n", cam_id, detect_res.keypoints.size());

    for (size_t i = 0; i < detect_res.keypoints.size(); ++i)
    {
        cv::KeyPoint kp;
        kp.pt.x = detect_res.keypoints[i].x;
        kp.pt.y = detect_res.keypoints[i].y;

        int x = int(kp.pt.x), y = int(kp.pt.y);

        // skip masked-out pixels
        if (detection_mask.at<uint8_t>(y, x) > 127)
            continue;

        // which cell?
        int cx = int(std::floor(kp.pt.x / size_x));
        int cy = int(std::floor(kp.pt.y / size_y));
        if (cx < 0 || cx >= grid_x || cy < 0 || cy >= grid_y)
            continue;

        size_t cell_key = pack_cell(cx, cy);

        // only cells that are valid and not yet full
        if (!valid_set.count(cell_key))
            continue;
        if (picked_count[cell_key] >= num_features_grid)
            continue;

        // accept it
        new_pts.push_back(kp);
        picked_count[cell_key]++;

        // optional early exit
        if (new_pts.size() >= valid_locs.size() * num_features_grid)
            break;
    }

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    cv::Size size_close((int)((float)width / (float)min_px_dist),
                        (int)((float)height / (float)min_px_dist)); // width x height
    for (auto &kpt : new_pts)
    {
        // Check that it is in bounds
        int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
        int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
            continue;
        // See if there is a point at this location
        if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
            continue;
        // Else lets add it!
        kpts0_new.push_back(kpt);
        pts0_new.push_back(kpt.pt);
        grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
    }

    // Loop through and record only ones that are valid
    // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
    // NOTE: this is due to the fact that we select update features based on feat id
    // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
    // NOTE: not sure how to remove... maybe a better way?
    for (size_t i = 0; i < pts0_new.size(); i++)
    {
        // update the uv coordinates
        kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts0.push_back(kpts0_new.at(i));
        // move id foward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);
    };
}

void TrackOCL::perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0,
                                           std::vector<cv::KeyPoint> &pts0,
                                           std::vector<size_t> &ids0, int cam_id)
{

    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less than grid_px_size points away then existing features

    auto rT1 = boost::posix_time::microsec_clock::local_time();

    cv::Size size_close((int)((float)img0pyr.at(0).cols / (float)min_px_dist),
                        (int)((float)img0pyr.at(0).rows / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
    float size_x = (float)img0pyr.at(0).cols / (float)grid_x;
    float size_y = (float)img0pyr.at(0).rows / (float)grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
    cv::Mat mask0_updated = mask0.clone();
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end())
    {
        // Get current left keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img0pyr.at(0).cols - edge || y < edge || y >= img0pyr.at(0).rows - edge)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate mask coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x);
        int y_grid = std::floor(kpt.pt.y / size_y);
        if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Now check if it is in a mask area or not
        // NOTE: mask has max value of 255 (white) if it should be
        if (mask0.at<uint8_t>(y, x) > 127)
        {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255)
        {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        // Append this to the local mask of the image
        if (x - min_px_dist >= 0 && x + min_px_dist < img0pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img0pyr.at(0).rows)
        {
            cv::Point pt1(x - min_px_dist, y - min_px_dist);
            cv::Point pt2(x + min_px_dist, y + min_px_dist);
            cv::rectangle(mask0_updated, pt1, pt2, cv::Scalar(255));
        }
        it0++;
        it1++;
    }

    auto rT2 = boost::posix_time::microsec_clock::local_time();
    PRINT_DEBUG("[TIME-DTCT]: %.4f seconds for grid creation\n", (rT2 - rT1).total_microseconds() * 1e-6);

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    double min_feat_percent = 0.50;
    int num_featsneeded = num_features - (int)pts0.size();
    if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
        return;

    // We also check a downsampled mask such that we don't extract in areas where it is all masked!
    cv::Mat mask0_grid;
    cv::resize(mask0, mask0_grid, size_grid, 0.0, 0.0, cv::INTER_NEAREST);

    // Create grids we need to extract from and then extract our features (use fast with griding)
    int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;
    int num_features_grid_req = std::max(1, (int)(min_feat_percent * num_features_grid));
    std::vector<std::pair<int, int>> valid_locs;
    for (int x = 0; x < grid_2d_grid.cols; x++)
    {
        for (int y = 0; y < grid_2d_grid.rows; y++)
        {
            if ((int)grid_2d_grid.at<uint8_t>(y, x) < num_features_grid_req && (int)mask0_grid.at<uint8_t>(y, x) != 255)
            {
                valid_locs.emplace_back(x, y);
            }
        }
    }
    std::vector<cv::KeyPoint> pts0_ext;
    Grider_OCL::perform_griding(mgr.getTracker(cam_id), img0pyr.at(0), mask0_updated, valid_locs, pts0_ext, num_features, grid_x, grid_y, threshold, true);

    auto rT3 = boost::posix_time::microsec_clock::local_time();
    PRINT_DEBUG("[TIME-DTCT]: %.4f seconds for grid detection\n", (rT3 - rT2).total_microseconds() * 1e-6);

    // Now, reject features that are close a current feature
    std::vector<cv::KeyPoint> kpts0_new;
    std::vector<cv::Point2f> pts0_new;
    for (auto &kpt : pts0_ext)
    {
        // Check that it is in bounds
        int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
        int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
            continue;
        // See if there is a point at this location
        if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
            continue;
        // Else lets add it!
        kpts0_new.push_back(kpt);
        pts0_new.push_back(kpt.pt);
        grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
    }

    // Loop through and record only ones that are valid
    // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
    // NOTE: this is due to the fact that we select update features based on feat id
    // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
    // NOTE: not sure how to remove... maybe a better way?
    for (size_t i = 0; i < pts0_new.size(); i++)
    {
        // update the uv coordinates
        kpts0_new.at(i).pt = pts0_new.at(i);
        // append the new uv coordinate
        pts0.push_back(kpts0_new.at(i));
        // move id foward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);
    }
    auto rT4 = boost::posix_time::microsec_clock::local_time();
    PRINT_DEBUG("[TIME-DTCT]: %.4f seconds for feature rejection\n", (rT4 - rT3).total_microseconds() * 1e-6);
}

void TrackOCL::perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, size_t id0, size_t id1, std::vector<uchar> &mask_out)
{

    // We must have equal vectors
    assert(kpts0.size() == kpts1.size());

    // Return if we don't have any points
    if (kpts0.empty() || kpts1.empty())
        return;

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::Point2f> pts0, pts1;
    std::vector<float> pts_out;
    for (size_t i = 0; i < kpts0.size(); i++)
    {
        pts0.push_back(kpts0.at(i).pt);
        pts1.push_back(kpts1.at(i).pt);

        // for gpu run
        pts_out.push_back(kpts0.at(i).pt.x);
        pts_out.push_back(kpts0.at(i).pt.y);
    }

    // If we don't have enough points for ransac just return empty
    // We set the mask to be all zeros since all points failed RANSAC
    if (pts0.size() < 10)
    {
        for (size_t i = 0; i < pts0.size(); i++)
            mask_out.push_back((uchar)0);
        return;
    }

    // Now do KLT tracking to get the valid new points
    std::vector<uchar> mask_klt;
    std::vector<float> error;

    int n_points = pts0.size();
    mgr.getTracker(id0)->runTrackingStep(n_points, (float *)pts_out.data());

    size_t buffer_size = n_points * sizeof(float) * 2;

    std::vector<float> tracked_pts;
    mask_klt.resize(n_points);
    error.resize(n_points);
    tracked_pts.resize(n_points * 2);

    mgr.getTracker(id0)->readResults(n_points, tracked_pts.data(), mask_klt.data(), error.data());

    for (int i = 0; i < n_points; i++)
    {
        cv::Point2f pt = (cv::Point2f){tracked_pts[i * 2], tracked_pts[i * 2 + 1]};
        pts1[i] = pt;
    }

    // Normalize these points, so we can then do ransac
    // We don't want to do ransac on distorted image uvs since the mapping is nonlinear
    std::vector<cv::Point2f> pts0_n, pts1_n;
    for (size_t i = 0; i < pts0.size(); i++)
    {
        pts0_n.push_back(camera_calib.at(id0)->undistort_cv(pts0.at(i)));
        pts1_n.push_back(camera_calib.at(id1)->undistort_cv(pts1.at(i)));
    }

    // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    double max_focallength_img0 = std::max(camera_calib.at(id0)->get_K()(0, 0), camera_calib.at(id0)->get_K()(1, 1));
    double max_focallength_img1 = std::max(camera_calib.at(id1)->get_K()(0, 0), camera_calib.at(id1)->get_K()(1, 1));
    double max_focallength = std::max(max_focallength_img0, max_focallength_img1);
    cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 2.0 / max_focallength, 0.999, mask_rsc);

    // Loop through and record only ones that are valid
    for (size_t i = 0; i < mask_klt.size(); i++)
    {
        auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
        mask_out.push_back(mask);
    }

    // Copy back the updated positions
    for (size_t i = 0; i < pts0.size(); i++)
    {
        kpts0.at(i).pt = pts0.at(i);
        kpts1.at(i).pt = pts1.at(i);
    }
}
