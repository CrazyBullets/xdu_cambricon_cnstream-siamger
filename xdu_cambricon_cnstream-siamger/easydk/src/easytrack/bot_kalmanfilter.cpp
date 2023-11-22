#include "bot_kalmanfilter.h"
#include "util.hpp"
namespace edk {

BotKalmanFilter::BotKalmanFilter() : KalmanFilter() {}

void BotKalmanFilter::Initiate(const BoundingBox &measurement) {
  // initial state X(k-1|k-1)
  mean_(0, 0) = measurement.x;
  mean_(0, 1) = measurement.y;
  mean_(0, 2) = measurement.width;
  mean_(0, 3) = measurement.height;
  for (int i = 4; i < 8; ++i) {
    mean_(0, i) = 0;
  }

  vector<float> std{2 * std_weight_position_ * measurement.width,  2 * std_weight_position_ * measurement.height,
                    2 * std_weight_position_ * measurement.width,  2 * std_weight_position_ * measurement.height,
                    10 * std_weight_velocity_ * measurement.width, 10 * std_weight_velocity_ * measurement.height,
                    10 * std_weight_velocity_ * measurement.width, 10 * std_weight_velocity_ * measurement.height};

  // init MMSE P(k-1|k-1)
  for (int i = 0; i < 8; ++i) covariance_(i, i) = std[i] * std[i];
}

void BotKalmanFilter::Predict() {
  vector<float> std{std_weight_position_ * mean_(0, 2), std_weight_position_ * mean_(0, 3),
                    std_weight_position_ * mean_(0, 2), std_weight_position_ * mean_(0, 3),
                    std_weight_velocity_ * mean_(0, 2), std_weight_velocity_ * mean_(0, 3),
                    std_weight_velocity_ * mean_(0, 2), std_weight_velocity_ * mean_(0, 3)};

  // process noise covariance Q
  Matrix motion_cov(8, 8);
  for (int i = 0; i < 8; ++i) {
    motion_cov(i, i) = std[i] * std[i];
  }

  // formula 1：x(k|k-1)=A*x(k-1|k-1)
  // mean_ = (motion_mat_ * mean_.Trans()).Trans();
  Matrix mean1 = mean_ * motion_mat_trans_;
  // formula 2：P(k|k-1)=A*P(k-1|k-1)A^T +Q
  Matrix covariance1 = motion_mat_ * covariance_ * motion_mat_trans_ + motion_cov;

  mean_ = std::move(mean1);
  covariance_ = std::move(covariance1);
  need_recalc_project_ = true;
}

void BotKalmanFilter::Project(const Matrix &mean, const Matrix &covariance) {
  if (!need_recalc_project_) return;
  vector<float> std{std_weight_position_ * mean_(0, 2), std_weight_position_ * mean_(0, 3),
                    std_weight_position_ * mean_(0, 2), std_weight_position_ * mean_(0, 3)};

  // measurement noise R
  Matrix innovation_cov(4, 4);

  for (int i = 0; i < 4; ++i) {
    innovation_cov(i, i) = std[i] * std[i];
  }

  // project_mean_ = (update_mat_ * mean_.Trans()).Trans();
  project_mean_ = mean_ * update_mat_trans_;

  // part of formula 3：(H*P(k|k-1)*H^T + R)
  project_covariance_ = update_mat_ * covariance * update_mat_trans_ + innovation_cov;
  need_recalc_project_ = false;
}

void BotKalmanFilter::Update(const BoundingBox &bbox) {
  Project(mean_, covariance_);

  Matrix measurement(1, 4);
  measurement(0, 0) = bbox.x;
  measurement(0, 1) = bbox.y;
  measurement(0, 2) = bbox.width;
  measurement(0, 3) = bbox.height;

  // formula 3: Kg = P(k|k-1) * H^T * (H*P(k|k-1)*H^T + R)^(-1)
  Matrix kalman_gain = covariance_ * update_mat_trans_ * project_covariance_.Inv();
  // formula 4: x(k|k) = x(k|k-1) + Kg * (m - H * x(k|k-1))
  mean_ += (measurement - project_mean_) * kalman_gain.Trans();
  // formula 5: P(k|k) = P(k|k-1) - Kg * H * P(k|k-1)
  covariance_ = covariance_ - kalman_gain * project_covariance_ * kalman_gain.Trans();

  need_recalc_project_ = true;
}
void BotKalmanFilter::ApplyCMC_2(const cv::Mat& warp){
  
  //BoTSort Method
  cv::Mat R = warp(cv::Rect(0,0,2,2));
  cv::Mat cv_R8x8;
  kron(cv::Mat::eye(4,4,CV_32F), R, cv_R8x8);
  
  Matrix R8x8 = mat2edk(cv_R8x8);
  mean_ = mean_ * R8x8.Trans();

  mean_(0,0) +=  warp.at<float>(0,2);
  mean_(0,1) +=  warp.at<float>(1,2);

  covariance_ = R8x8 * covariance_ * R8x8.Trans();
}
}  // namespace edk
