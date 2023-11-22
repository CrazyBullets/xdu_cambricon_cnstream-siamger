#ifndef EASYTRACK_NASKALMANFILTER_H
#define EASYTRACK_NASKALMANFILTER_H

#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include "easytrack/easy_track.h"
#include "matrix.h"
#include "kalmanfilter.h"

namespace edk {

/**
 * @brief Implementation of NAS Kalman filter
 */
class NasKalmanFilter : public KalmanFilter {
 public:
  /**
   * @brief Initialize the state transition matrix and measurement matrix
   */
  NasKalmanFilter();

  /**
   * @brief Initialize the initial state X(k-1|k-1), MMSE P(k-1|k-1), and score
   */
  void Initiate(const BoundingBox& measurement, const float& score);

  /**
   * @brief Calculate measurement noise R
   */
  void Project(const Matrix& mean, const Matrix& covariance);

  /**
   * @brief Apply camera motion compensation
   */
  void ApplyCMC_1(const cv::Mat& warp);
  void ApplyCMC_2(const cv::Mat& warp);
 private:
  float score_;

};  // class NAS KalmanFilter

}  // namespace edk

#endif  // EASYTRACK_NASKALMANFILTER_H
