#include "oc_kalmanfilter.h"
#include <cmath>
#include <vector>
#include "cxxutil/log.h"

namespace edk {

const Matrix OcKalmanFilter::motion_mat_{std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0,
                                                          0, 1, 0, 0, 0, 1, 0, 0,
                                                          0, 0, 1, 0, 0, 0, 1, 0,
                                                          0, 0, 0, 1, 0, 0, 0, 1,
                                                          0, 0, 0, 0, 1, 0, 0, 0,
                                                          0, 0, 0, 0, 0, 1, 0, 0,
                                                          0, 0, 0, 0, 0, 0, 1, 0,
                                                          0, 0, 0, 0, 0, 0, 0, 1}, 8u, 8u};
const Matrix OcKalmanFilter::update_mat_{std::vector<float>{1, 0, 0, 0, 0, 0, 0, 0,
                                                          0, 1, 0, 0, 0, 0, 0, 0,
                                                          0, 0, 1, 0, 0, 0, 0, 0,
                                                          0, 0, 0, 1, 0, 0, 0, 0}, 4u, 8u};

const Matrix OcKalmanFilter::update_mat_trans_{OcKalmanFilter::update_mat_.Trans()};
const Matrix OcKalmanFilter::motion_mat_trans_{OcKalmanFilter::motion_mat_.Trans()};

OcKalmanFilter::OcKalmanFilter() : mean_(1, 8), covariance_(8, 8), innovation_cov_(4, 4), motion_cov_(8, 8),
                               std_weight_position_(1. / 20), std_weight_velocity_(1. / 160), time_gap_(0), need_recalc_project_(true), observed_(false), unfrozen_(true) {}

void OcKalmanFilter::Initiate(const BoundingBox &observation) {
  // initial state X(k-1|k-1)
  mean_(0, 0) = observation.x;
  mean_(0, 1) = observation.y;
  mean_(0, 2) = observation.width;
  mean_(0, 3) = observation.height;
  for (int i = 4; i < 8; ++i) {
    mean_(0, i) = 0;
  }

  vector<float> std(8, 0);
  std[2] = 1e-2;
  std[0] = std[1] = std[3] = 2 * std_weight_position_ * observation.height;

  std[6] = 1e-5;
  std[4] = std[5] = std[7] = 10 * std_weight_velocity_ * observation.height;

  // init P
  for (int i = 0; i < 8; ++i) covariance_(i, i) = std[i] * std[i];
  // // give high uncertainty to the unobservable initial velocities
  // for (int i = 0; i < 4; ++i) covariance_(i, i) = 10;
  // for (int i = 4; i < 8; ++i) covariance_(i, i) = 10000;

  // init Q
  for (int i = 4; i < 8; ++i) motion_cov_(i, i) = 0.01f;
  motion_cov_(7, 7) *= 0.01f;

  // init R
  // for (int i = 0; i < 2; ++i) innovation_cov_(i, i) = 1;
  // for (int i = 2; i < 4; ++i) innovation_cov_(i, i) = 10;
  
}

void OcKalmanFilter::Predict() {
  LOGD(OCSORT) << "Object Predict";
  if(!unfrozen_)
  {
    ++time_gap_;
    return;
  }
    
  // formula 1：x(k|k-1)=A*x(k-1|k-1)
  // mean_ = (motion_mat_ * mean_.Trans()).Trans();
  Matrix mean1 = mean_ * motion_mat_trans_;
  // formula 2：P(k|k-1)=A*P(k-1|k-1)A^T +Q
  Matrix covariance1 = motion_mat_ * covariance_ * motion_mat_trans_ + motion_cov_;

  mean_ = std::move(mean1);
  covariance_ = std::move(covariance1);
  need_recalc_project_ = true;
}

void OcKalmanFilter::Project(const Matrix &mean, const Matrix &covariance) {
  if (!need_recalc_project_) return;

  float cov_val1 = 1e-1 * 1e-1;
  float cov_val2 = std_weight_position_ * mean(0, 3);
  cov_val2 *= cov_val2;

  // measurement noise R
  innovation_cov_(0, 0) = cov_val2;
  innovation_cov_(1, 1) = cov_val2;
  innovation_cov_(3, 3) = cov_val2;

  innovation_cov_(2, 2) = cov_val1;

  // project_mean_ = (update_mat_ * mean_.Trans()).Trans();
  project_mean_ = mean_ * update_mat_trans_;

  // part of formula 3：(H*P(k|k-1)*H^T + R)
  project_covariance_ = update_mat_ * covariance * update_mat_trans_ + innovation_cov_;
  need_recalc_project_ = false;
}

void OcKalmanFilter::Update(const BoundingBox &bbox) {
  LOGD(OCSORT) << "Object Update";
  if (!observed_ && !unfrozen_)
  {
    Unfreeze(bbox);
  }
  observed_ = true;
  history_obs_ = bbox;  // save observation

  Project(mean_, covariance_);

  Matrix observation(1, 4);
  observation(0, 0) = bbox.x;
  observation(0, 1) = bbox.y;
  observation(0, 2) = bbox.width;
  observation(0, 3) = bbox.height;

  // formula 3: Kg = P(k|k-1) * H^T * (H*P(k|k-1)*H^T + R)^(-1)
  Matrix kalman_gain = covariance_ * update_mat_trans_ * project_covariance_.Inv();
  // formula 4: x(k|k) = x(k|k-1) + Kg * (m - H * x(k|k-1))
  mean_ += (observation - project_mean_) * kalman_gain.Trans();
  // formula 5: P(k|k) = P(k|k-1) - Kg * H * P(k|k-1)
  covariance_ = covariance_ - kalman_gain * update_mat_ * covariance_;
  // covariance_ = covariance_ - kalman_gain * projected_cov * kalman_gain.Trans();

  need_recalc_project_ = true;
}

void OcKalmanFilter::Update() {
  if (observed_)
  {
    Freeze();
  }
  observed_ = false;
}

void OcKalmanFilter::Freeze() {
  LOGD(OCSORT) << "Object freeze";
  unfrozen_ = false;
  time_gap_ = 0;
}

void OcKalmanFilter::Unfreeze(const BoundingBox &bbox) {
  LOGD(OCSORT) << "Object Unfreeze";
  unfrozen_ = true;

  float x1 = bbox.x;
  float y1 = bbox.y;
  float s1 = bbox.width;
  float r1 = bbox.height;
  float h1 = std::sqrt(std::max(0.0f, s1 * r1));
  float w1 = s1 / h1;

  float x2 = history_obs_.x;
  float y2 = history_obs_.y;
  float s2 = history_obs_.width;
  float r2 = history_obs_.height;
  float h2 = std::sqrt(std::max(0.0f, s2 * r2));
  float w2 = s2 / h2;

  float dx = (x2 - x1) / static_cast<float>(time_gap_);
  float dy = (y2 - y1) / static_cast<float>(time_gap_);
  float dw = (w2 - w1) / static_cast<float>(time_gap_);
  float dh = (h2 - h1) / static_cast<float>(time_gap_);

  float h,w;
  for (int i = 0; i < time_gap_; i++)
  {
    BoundingBox virtual_bbox;
    virtual_bbox.x = x1 + (i + 1) * dx;
    virtual_bbox.y = y1 + (i + 1) * dy;
    w = w1 + (i + 1) * dw;
    h = h1 + (i + 1) * dh;
    virtual_bbox.width = w * h;
    virtual_bbox.height = h / w;
    LOGD(OCSORT) << "Virtual Object Update";
    Update(virtual_bbox);
    if (i != time_gap_ - 1)
    {
      LOGD(OCSORT) << "Virtual Object Predict";
      Predict();
    }
      
  }
  
}

Matrix OcKalmanFilter::GatingDistance(const std::vector<BoundingBox> &observations) {
  Project(mean_, covariance_);
  Matrix const& mean1 = project_mean_;
  Matrix covariance1_inv = project_covariance_.Inv();

  int num = observations.size();
  Matrix d(1, 4);
  Matrix square_maha(1, num);
  for (int i = 0; i < num; i++) {
    d(0, 0) = observations[i].x - mean1(0, 0);
    d(0, 1) = observations[i].y - mean1(0, 1);
    d(0, 2) = observations[i].width - mean1(0, 2);
    d(0, 3) = observations[i].height - mean1(0, 3);

    square_maha(0, i) = (d * covariance1_inv * d.Trans())(0, 0);
  }
  return square_maha;
}

BoundingBox OcKalmanFilter::GetCurPos() {
  return {mean_(0, 0), mean_(0, 1), mean_(0, 2), mean_(0, 3)};
}

}  // namespace edk
