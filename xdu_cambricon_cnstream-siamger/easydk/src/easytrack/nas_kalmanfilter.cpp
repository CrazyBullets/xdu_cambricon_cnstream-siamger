#include "nas_kalmanfilter.h"
#include <cmath>
#include <vector>
#include "util.hpp"
#include "track_data_type.h"

namespace edk {

NasKalmanFilter::NasKalmanFilter() : KalmanFilter() {}

void NasKalmanFilter::Initiate(const BoundingBox &measurement, const float& score) {
  // initial state X(k-1|k-1) and MMSE P(k-1|k-1)
  KalmanFilter::Initiate(measurement);
  // init score
  score_ = score;
}

void NasKalmanFilter::Project(const Matrix &mean, const Matrix &covariance) {
  if (!need_recalc_project_) return;
  vector<float> std{std_weight_position_ * mean(0, 3),
                    std_weight_position_ * mean(0, 3),
                    0.1f,
                    std_weight_position_ * mean(0, 3)};

  // measurement noise R
  Matrix innovation_cov(4, 4);
  for (int i = 0; i < 4; ++i) innovation_cov(i, i) = std[i] * std[i];
  

  // project_mean_ = (update_mat_ * mean_.Trans()).Trans();
  project_mean_ = mean_ * update_mat_trans_;

  // part of formula 3：(H*P(k|k-1)*H^T + R(1-c))
  project_covariance_ = update_mat_ * covariance * update_mat_trans_ + innovation_cov * (1.0f - score_);
  need_recalc_project_ = false;
}

void NasKalmanFilter::ApplyCMC_1(const cv::Mat& warp){

  //StrongSort Method
  Matrix warp_ = mat2edk(warp);
  BoundingBox xyah, tlbr;
  Matrix tl(1,3);
  Matrix br(1,3);
  xyah.x = mean_(0,0);
  xyah.y = mean_(0,1);
  xyah.width = mean_(0,2);
  xyah.height = mean_(0,3);
  tlbr = xyah2tlbr(xyah);

  tl(0, 0) = tlbr.x;
  tl(0, 1) = tlbr.y;
  tl(0, 2) = 1; 

  br(0, 0) = tlbr.width; 
  br(0, 1) = tlbr.height; 
  br(0, 2) = 1; 

  tl = tl*warp_.Trans();
  br = br*warp_.Trans();

  float w = br(0,0) - tl(0,0);
  float h = br(0,1) - tl(0,1);

  mean_(0,3) = h;
  mean_(0,2) = w/h;
  mean_(0,1) = tl(0,1) + h/2;
  mean_(0,0) = tl(0,0) + w/2;

}
void NasKalmanFilter::ApplyCMC_2(const cv::Mat& warp){
  
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
