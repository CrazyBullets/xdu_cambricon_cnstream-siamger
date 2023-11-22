#ifndef EASYTRACK_BOTKALMANFILTER_H
#define EASYTRACK_BOTKALMANFILTER_H

#include <cmath>
#include <utility>
#include <vector>

#include "easytrack/easy_track.h"
#include "kalmanfilter.h"
#include "matrix.h"

namespace edk {

/**
 * @brief Implementation of BoTSORT Kalman filter
 */
class BotKalmanFilter : public KalmanFilter {
 public:
  /**
   * @brief Initialize the state transition matrix and measurement matrix
   */
  BotKalmanFilter();

  /**
   * @brief Initialize the initial state X(k-1|k-1) and MMSE P(k-1|k-1)
   */
  void Initiate(const BoundingBox& measurement);

  /**
   * @brief Predict the x(k|k-1) and P(k|k-1)
   */
  void Predict();

  /**
   * @brief Calculate measurement noise R
   */
  void Project(const Matrix& mean, const Matrix& covariance);

  /**
   * @brief Calculate the Kalman gain and update the state and MMSE
   */
  void Update(const BoundingBox& measurement);
  
  void ApplyCMC_2(const cv::Mat& warp);
};  // class BotKalmanFilter

}  // namespace edk

#endif  // EASYTRACK_BOTKALMANFILTER_H
