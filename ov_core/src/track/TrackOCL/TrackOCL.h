#ifndef OV_CORE_TRACK_OCL_H
#define OV_CORE_TRACK_OCL_H

#include "../TrackBase.h"
#include "cam/CamBase.h"
#include <modal_flow_ocl_manager.h>
#include <ocl/OclDevice.hpp>
#include <ocl/ManagerCL.hpp>
#include <modal_flow/Types.hpp>

namespace ov_core
{

  /**
   * @brief Leveraging OpenCL + GPU to perform KLT tracking of features.
   */
  class TrackOCL : public TrackBase
  {

  public:
    /**
     * @brief Public constructor with configuration variables
     * @param cameras camera calibration object which has all camera intrinsics in it
     * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
     * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
     * @param stereo if we should do stereo feature tracking or binocular
     * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
     * @param fast_threshold FAST detection threshold
     * @param gridx size of grid in the x-direction / u-direction
     * @param gridy size of grid in the y-direction / v-direction
     * @param minpxdist features need to be at least this number pixels away from each other
     */
    explicit TrackOCL(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco,
                      int fast_threshold, int gridx, int gridy, int minpxdist)
        : TrackBase(cameras, numfeats, numaruco, false, NONE),
          threshold(fast_threshold),
          grid_x(gridx),
          grid_y(gridy),
          min_px_dist(minpxdist),
          mgr(VoxlOCLManager::instance()),
          dev_(modal_flow::ocl::OclDevice::Instance()),
          mgr_(dev_)
    {
      if (cameras.empty() || !cameras.at(0))
      {
        throw std::runtime_error("Invalid camera data");
      }

      // Create and set detector
      auto det = std::make_unique<modal_flow::ocl::DetectorCL>(dev_, 3);
      mgr_.set_detector(std::move(det));

      // create and set tracker
      auto trk = std::make_unique<modal_flow::ocl::TrackerCL>(dev_);
      mgr_.set_tracker(std::move(trk));

      for (auto const &[camId, camPtr] : cameras)
      {
        //  assumes track input frames will be uint8_t grayscale
        modal_flow::Camera cam{.id = camId, .width = camPtr->w(), .height = camPtr->h(), .format = modal_flow::PixelFormat::R8};
        mgr_.add_camera(cam);
      }

      // Retrieve width and height
      int width = cameras.at(0)->w();
      int height = cameras.at(0)->h();

      // Initialize OpenCL manager
      cl_image_format fmt{};
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_FLOAT;
      for (int i = 0; i < cameras.size(); i++)
      {
        mgr.createTracker(i, width, height, pyr_levels, fmt);
      }
    }

    /**
     * @brief Process a new image
     * @param message Contains our timestamp, images, and camera ids
     */
    void feed_new_camera(const CameraData &message) override;

    /**
     * @brief set pyramid levels
     */
    void set_pyramid_levels(int levels) { pyr_levels = levels; };

  protected:
    /**
     * @brief Process a new monocular image
     * @param message Contains our timestamp, images, and camera ids
     * @param msg_id the camera index in message data vector
     */
    void feed_monocular(const CameraData &message, size_t msg_id);

    /**
     * @brief Process a new monocular image
     * @param message Contains our timestamp, images, and camera ids
     * @param msg_id the camera index in message data vector
     */
    void feed_batch_monocular(const CameraData &message);

    /**
     * @brief Detects new features in the current image
     * @param img0pyr image we will detect features on (first level of pyramid)
     * @param mask0 mask which has what ROI we do not want features in
     * @param pts0 vector of currently extracted keypoints in this image
     * @param ids0 vector of feature ids for each currently extracted keypoint
     *
     * Given an image and its currently extracted features, this will try to add new features if needed.
     * Will try to always have the "max_features" being tracked through KLT at each timestep.
     * Passed images should already be grayscaled.
     */

    std::vector<std::pair<int, int>> get_grids_to_fill(int width, int height,
                                                       const cv::Mat &mask0,
                                                       cv::Mat &mask0_updated, // used for detection
                                                       cv::Mat &grid_2d_close,
                                                       std::vector<cv::KeyPoint> &pts0,
                                                       std::vector<size_t> &ids0);

    void process_batch_detection_result(int width, int height, int cam_id,
                                        const cv::Mat &detection_mask,
                                        cv::Mat grid_2d_close,
                                        std::vector<std::pair<int, int>> valid_locs,
                                        modal_flow::DetectResult detect_res,
                                        std::vector<cv::KeyPoint> &pts0,
                                        std::vector<size_t> &ids0);

    void perform_detection_monocular(const std::vector<cv::Mat> &img0pyr, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                     std::vector<size_t> &ids0, int id);

    /**
     * @brief KLT track between two images, and do RANSAC afterwards
     * @param img0pyr starting image pyramid
     * @param img1pyr image pyramid we want to track too
     * @param pts0 starting points
     * @param pts1 points we have tracked
     * @param id0 id of the first camera
     * @param id1 id of the second camera
     * @param mask_out what points had valid tracks
     *
     * This will track features from the first image into the second image.
     * The two point vectors will be of equal size, but the mask_out variable will specify which points are good or bad.
     * If the second vector is non-empty, it will be used as an initial guess of where the keypoints are in the second image.
     */
    void perform_matching(const std::vector<cv::Mat> &img0pyr, const std::vector<cv::Mat> &img1pyr, std::vector<cv::KeyPoint> &pts0,
                          std::vector<cv::KeyPoint> &pts1, size_t id0, size_t id1, std::vector<uchar> &mask_out);

    void perform_batch_matching();

    // Parameters for our FAST grid detector
    int threshold;
    int grid_x;
    int grid_y;

    // Minimum pixel distance to be "far away enough" to be a different extracted feature
    int min_px_dist;

    // How many pyramid levels to track
    int pyr_levels = 5;
    cv::Size win_size = cv::Size(15, 15);

    // Last set of image pyramids
    std::map<size_t, std::vector<cv::Mat>> img_pyramid_last;
    std::map<size_t, cv::Mat> img_curr;
    std::map<size_t, std::vector<cv::Mat>> img_pyramid_curr;

  private:
    VoxlOCLManager &mgr;

    modal_flow::ocl::OclDevice &dev_;
    modal_flow::ocl::ManagerCL mgr_;
  };

} // namespace ov_core

#endif /* OV_CORE_TRACK_KLT_H */
