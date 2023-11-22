#ifndef EASYTRACK_OCKALMANFILTER_H
#define EASYTRACK_OCKALMANFILTER_H

#include <utility>
#include <vector>

#include "easytrack/easy_track.h"
#include "matrix.h"

namespace edk {

/**
 * @brief Implementation of Kalman filter
 */
class OcKalmanFilter {
 public:
  /**
   * @brief Initialize the state transition matrix and measurement matrix
   */
  OcKalmanFilter();

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

  /**
   * @brief Observation-centric Online Smoothing Update
   */
  void Update();

  /**
   * @brief Save the parameters before non-observation forward
   */
  void Freeze();

  /**
   * @brief Performance Observation-centric Online Smoothing before Update
   */
  void Unfreeze(const BoundingBox &bbox);

  /**
   * @brief Calculate the mahalanobis distance
   */
  Matrix GatingDistance(const std::vector<BoundingBox>& measurements);

  BoundingBox GetCurPos();

 protected:
  static const Matrix motion_mat_;
  static const Matrix update_mat_;
  static const Matrix motion_mat_trans_;
  static const Matrix update_mat_trans_;
  Matrix mean_;
  Matrix covariance_;

  Matrix project_mean_;
  Matrix project_covariance_;
  
  // observation noise R
  Matrix innovation_cov_;
  
  // noise covariance Q
  Matrix motion_cov_;

  float std_weight_position_;
  float std_weight_velocity_;
  
  int time_gap_;

  bool need_recalc_project_;
  bool observed_;
  bool unfrozen_;

  // last observation before being untracked
  BoundingBox history_obs_;
};  // class OcKalmanFilter

}  // namespace edk

#endif  // EASYTRACK_OCKALMANFILTER_H
