/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "cxxutil/log.h"
#include "easytrack/easy_track.h"
#include "nas_kalmanfilter.h"
#include "match.h"
#include "matrix.h"
#include "track_data_type.h"
#include "util.hpp"


#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

// chi2inv95 at 4 degree of freedom
constexpr const float gating_threshold = 9.4877;

namespace edk {

struct StrongSORTTrackObject {
  NasKalmanFilter kf;
  std::vector<Feature> features;
  Rect pos;
  int class_id;
  int track_id = -1;
  float score;
  TrackState state;
  int age = 1;
  int time_since_last_update = 0;
  bool has_feature = false;

  StrongSORTTrackObject() = default;
  StrongSORTTrackObject(StrongSORTTrackObject&&) = default;
  StrongSORTTrackObject& operator=(StrongSORTTrackObject&&) = default;
  StrongSORTTrackObject(const StrongSORTTrackObject&) = delete;
  StrongSORTTrackObject& operator=(const StrongSORTTrackObject&) = delete;
};

class StrongSORTPrivate {
 private:
  explicit StrongSORTPrivate(StrongSORTTrack *fm) {
    fm_ = fm;
    first_mat_ = true;
    match_algo_ = MatchAlgorithm::Instance();
  }
  void MatchCascade();
  void MatchIou(const std::vector<int>& detect_matrices, const std::vector<int>& track_matrices);
  void InitNewTrack(const DetectObject &obj);
  void MarkMiss(StrongSORTTrackObject *track);
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks);

  StrongSORTTrack *fm_;

  MatchAlgorithm *match_algo_;
  std::vector<StrongSORTTrackObject> tracks_;
  std::vector<int> unconfirmed_track_;
  std::vector<int> confirmed_track_;
  std::vector<int> assignments_;
  MatchResult res_feature_;
  MatchResult res_iou_;
  const Objects *detects_ = nullptr;

  bool first_mat_;
  cv::Mat warp_matrix_;
  cv::Mat pre_gray_mat_;
  cv::Mat cur_gray_mat_;
  cv::TermCriteria criteria_;

  uint64_t next_id_ = 0;
  friend class StrongSORTTrack;
};  // class StrongSORTPrivate

StrongSORTTrack::StrongSORTTrack() { fm_p_ = new StrongSORTPrivate(this); }

StrongSORTTrack::~StrongSORTTrack() {
  delete fm_p_;
}

void StrongSORTTrack::SetParams(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age,
                                  int n_init, int coordinate, bool ema, int max_count, double epsilon, float scale, int sz, int CMC_method) {
  // clang-format off
  LOGD(STRONGSORT) << "StrongSORTTrack Params -----\n"
              << "\n\t max cosine distance: " << max_cosine_distance
              << "\n\t max IoU distance: " << max_iou_distance
              << "\n\t max age: " << max_age
              << "\n\t nn budget: " << nn_budget
              << "\n\t n_init: " << n_init;
  // clang-format on
  max_cosine_distance_ = max_cosine_distance;
  max_iou_distance_ = max_iou_distance;
  nn_budget_ = nn_budget;
  max_age_ = max_age;
  n_init_ = n_init;
  if(coordinate == 0)
  {
    transformer_func_ = tlwh2xyah;
    inv_transformer_func_ = xyah2tlwh;
  }else if(coordinate == 1)
  {
    transformer_func_ = tlwh2xywh;
    inv_transformer_func_ = xywh2tlwh;
  }else if(coordinate == 2)
  {
    transformer_func_ = tlwh2xyar;
    inv_transformer_func_ = xyar2tlwh;
  }
  ema_ = ema;
  max_count_ = max_count;
  epsilon_ = epsilon;
  scale_ = scale; 
  sz_ = sz;
  CMC_method_ = CMC_method;
}


void StrongSORTPrivate::MatchCascade() {
  const Objects &det_objs = *detects_;
  Matrix cost_matrix;
  MatchResult &res = res_feature_;

  // refresh feature match result
  res.Clean();
  res.unmatched_detections.resize(det_objs.size());
  std::iota(res.unmatched_detections.begin(), res.unmatched_detections.end(), 0);

  if (confirmed_track_.empty() || det_objs.empty()) return;

  for (auto& det_obj : det_objs) {
    det_obj.feat_mold = L2Norm(det_obj.feature);
  }

  std::set<int> remained_detections;
  remained_detections.insert(res.unmatched_detections.begin(), res.unmatched_detections.end());
  LOGT(STRONGSORT) << "MatchCascade) Match scale, detects " << det_objs.size() << " tracks " << confirmed_track_.size();

  std::map<int, std::vector<int>> age_track_indices;
  for (size_t t = 0; t < confirmed_track_.size(); ++t) {
    int age = tracks_[confirmed_track_[t]].time_since_last_update - 1;
    age_track_indices[age].push_back(confirmed_track_[t]);
  }

  for (int age = 0; age < fm_->max_age_; ++age) {
    LOGA(STRONGSORT) << "Cascade: Number of remained detections ----- " << remained_detections.size();
    // no remained detections or no confirmed tracks, end match
    if (remained_detections.empty() || confirmed_track_.empty()) break;

    // get all confirmed tracks with same age
    auto track_indices_iter = age_track_indices.find(age);
    if (track_indices_iter == age_track_indices.end()) {
      LOGA(STRONGSORT) << "Cascade: No tracks for age " << age << " round, continue";
      continue;
    }

    std::vector<int>& track_indices = track_indices_iter->second;
    size_t det_num = res.unmatched_detections.size();
    size_t tra_num = track_indices.size();
    cost_matrix.Resize(tra_num, det_num);

    // calculate cost matrix
    std::vector<BoundingBox> measurements;
    measurements.reserve(det_num);
    for (size_t i = 0; i < det_num; ++i) {
      measurements.emplace_back(tlwh2xyah(det_objs[res.unmatched_detections[i]].bbox));
    }
    for (size_t i = 0; i < tra_num; ++i) {
      Matrix gating_dist = tracks_[track_indices[i]].kf.GatingDistance(measurements);
      for (size_t j = 0; j < det_num; ++j) {
        auto& det = det_objs[res.unmatched_detections[j]];
        cost_matrix(i, j) = match_algo_->Distance(tracks_[track_indices[i]].features,
                                                  Feature(det.feature, det.feat_mold));
        if (cost_matrix(i, j) > fm_->max_cosine_distance_ || gating_dist(0, j) > gating_threshold) {
          LOGA(STRONGSORT) << "object " << i << " - " << j << " feature distance is larger than max_cosine_distance";
          cost_matrix(i, j) = fm_->max_cosine_distance_ + 1e-5;
        }
      }
    }
    
    std::vector<int>& detect_indices = res.unmatched_detections;

    if (detect_indices.empty()) {
      LOGD(STRONGSORT) << "No remained detections to process IoU match";
      res.unmatched_tracks = track_indices;
      return;
    } else if (track_indices.empty()) {
      LOGD(STRONGSORT) << "No remained track objects to process IoU match";
      res.unmatched_detections = detect_indices;
      return;
    }

    MatchResult &iou_res = res_iou_;
    // clean iou match result
    iou_res.Clean();

    if (detect_indices.empty()) {
      LOGD(STRONGSORT) << "No remained detections to process IoU match";
      iou_res.unmatched_tracks = track_indices;
      return;
    } else if (track_indices.empty()) {
      LOGD(STRONGSORT) << "No remained track objects to process IoU match";
      iou_res.unmatched_detections = detect_indices;
      return;
    }
    // calculate iou cost matrix
    uint32_t detect_num = detect_indices.size();
    uint32_t track_num = track_indices.size();
    LOGT(STRONGSORT) << "MatchIoU) Match scale, detects " << detect_num << " tracks " << track_num;
    std::vector<Rect> det_rects, tra_rects;
    const Objects &det_objs = *detects_;
    std::set<int> remained_detections;
    det_rects.reserve(detect_num);
    tra_rects.reserve(track_num);
    for (auto &idx : detect_indices) {
      det_rects.emplace_back(BoundingBox2Rect(det_objs[idx].bbox));
      remained_detections.insert(idx);
    }
    for (auto &idx : track_indices) {
      tra_rects.emplace_back(tracks_[idx].pos);
    }
    Matrix iou_cost_matrix = match_algo_->IoUCost(tra_rects, det_rects);
    float gamma = 0.1;
    cost_matrix = cost_matrix*(1 - gamma) + iou_cost_matrix*gamma;

    // min cost match
    match_algo_->HungarianMatch(cost_matrix, &assignments_);

    // arrange match result
    for (size_t i = 0; i < assignments_.size(); ++i) {
      if (assignments_[i] < 0 || cost_matrix(i, assignments_[i]) > fm_->max_cosine_distance_) {
        res.unmatched_tracks.push_back(track_indices[i]);
      } else {
        res.matches.emplace_back(std::make_pair(res.unmatched_detections[assignments_[i]], track_indices[i]));
        remained_detections.erase(res.unmatched_detections[assignments_[i]]);
      }
    }
    age_track_indices.erase(track_indices_iter);
    res.unmatched_detections.clear();
    res.unmatched_detections.insert(res.unmatched_detections.end(), remained_detections.begin(),
                                    remained_detections.end());
  }
}

void StrongSORTPrivate::MatchIou(const std::vector<int>& detect_indices,
                                   const std::vector<int>& track_indices) {
  MatchResult &res = res_iou_;
  // clean iou match result
  res.Clean();

  if (detect_indices.empty()) {
    LOGD(STRONGSORT) << "No remained detections to process IoU match";
    res.unmatched_tracks = track_indices;
    return;
  } else if (track_indices.empty()) {
    LOGD(STRONGSORT) << "No remained track objects to process IoU match";
    res.unmatched_detections = detect_indices;
    return;
  }
  uint32_t detect_num = detect_indices.size();
  uint32_t track_num = track_indices.size();
  LOGT(STRONGSORT) << "MatchIoU) Match scale, detects " << detect_num << " tracks " << track_num;
  std::vector<Rect> det_rects, tra_rects;
  const Objects &det_objs = *detects_;
  std::set<int> remained_detections;
  det_rects.reserve(detect_num);
  tra_rects.reserve(track_num);

  // calculate iou cost matrix
  for (auto &idx : detect_indices) {
    det_rects.emplace_back(BoundingBox2Rect(det_objs[idx].bbox));
    remained_detections.insert(idx);
  }
  for (auto &idx : track_indices) {
    tra_rects.emplace_back(tracks_[idx].pos);
  }
  Matrix cost_matrix = match_algo_->IoUCost(tra_rects, det_rects);
  match_algo_->HungarianMatch(cost_matrix, &assignments_);

  for (size_t i = 0; i < assignments_.size(); ++i) {
    if (assignments_[i] < 0 || cost_matrix(i, assignments_[i]) > fm_->max_iou_distance_) {
      res.unmatched_tracks.push_back(track_indices[i]);
    } else {
      res.matches.emplace_back(std::make_pair(detect_indices[assignments_[i]], track_indices[i]));
      remained_detections.erase(detect_indices[assignments_[i]]);
    }
  }

  res.unmatched_detections.insert(res.unmatched_detections.end(), remained_detections.begin(),
                                  remained_detections.end());
}

void StrongSORTPrivate::InitNewTrack(const DetectObject &det) {
  StrongSORTTrackObject obj;
  obj.age = 1;
  obj.class_id = det.label;
  obj.score = det.score;
  obj.pos = BoundingBox2Rect(det.bbox);
  obj.state = TrackState::TENTATIVE;
  if (!det.feature.empty()) {
    for (auto& val : det.feature) {
      if (val != 0) {
        obj.has_feature = true;
        obj.features.emplace_back(det.feature, det.feat_mold);
        break;
      }
    }
  }
  obj.kf.Initiate(fm_->transformer_func_(det.bbox), obj.score);
  tracks_.emplace_back(std::move(obj));
}

void StrongSORTPrivate::MarkMiss(StrongSORTTrackObject *track) {
  if (track->state == TrackState::TENTATIVE || track->time_since_last_update > fm_->max_age_) {
    track->state = TrackState::DELETED;
  }
}

void StrongSORTPrivate::UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) {
  // Store frame in cv::Mat
  size_t sz = fm_->sz_;
  float scale = fm_->scale_;
  cv::Size raw;
  cur_gray_mat_ = frame.data.clone();
  cvtColor(cur_gray_mat_, cur_gray_mat_, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(cur_gray_mat_, cur_gray_mat_, cv::Size(3, 3), 1.5);
  if(fm_->scale_ > 0.01){
    cv::resize(cur_gray_mat_, cur_gray_mat_, cv::Size(static_cast<int>(cur_gray_mat_.cols*scale), static_cast<int>(cur_gray_mat_.rows*scale)));
  }
  else if(fm_->sz_ > 0){
    cv::resize(cur_gray_mat_, cur_gray_mat_, cv::Size(sz, sz));
  }
  else{
    LOGW(STRONGSORT)<<"Both variables sz_ and scale_ are 0! set sz_ to 640!";
    cv::resize(cur_gray_mat_, cur_gray_mat_, cv::Size(640, 640));

  }
  raw = cv::Size(cur_gray_mat_.cols, cur_gray_mat_.rows);
  
  // Initialization
  if(first_mat_)
  {
    pre_gray_mat_ = cur_gray_mat_.clone();
    warp_matrix_ = cv::Mat::eye(2, 3, CV_32F);  // for MOTION_EUCLIDEAN 
    criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, fm_->max_count_, fm_->epsilon_);
    first_mat_ = false;
  }else if(!cur_gray_mat_.empty()&&!cur_gray_mat_.empty())
  {
    // Run the ECC algorithm. The results are stored in warp_matrix.
    warp_matrix_ = cv::Mat::eye(2, 3, CV_32F);  // for MOTION_EUCLIDEAN 
    _findTransformECC(
      pre_gray_mat_,
      cur_gray_mat_,
      warp_matrix_,
      cv::MOTION_EUCLIDEAN,
      criteria_
      );
     
    warp_matrix_.at<float>(0,2) = warp_matrix_.at<float>(0,2)/(static_cast<float>(raw.width));
    warp_matrix_.at<float>(1,2) = warp_matrix_.at<float>(1,2)/(static_cast<float>(raw.height));
    
    cv::Mat eye = cv::Mat::eye(3,3,CV_32F);
    float a[1][3] ={0.,0.,1.};
    cv::Mat row = cv::Mat(1,3,CV_32F,a);
    warp_matrix_.push_back(row);
    double dst = cv::norm(eye - warp_matrix_);
    if( dst < 100){
      warp_matrix_.pop_back();
    }
    else{
      warp_matrix_ = eye;
    }

    pre_gray_mat_ = cur_gray_mat_.clone();
  }
  else{
    LOGE(STRONGSORT)<<"empty input!";
  }
  
  // set feat_mold to -1 means mold has not been computed
  for (auto& obj : detects) {
    obj.feat_mold = -1;
  }

  uint32_t detect_num = detects.size();
  uint32_t track_num = tracks_.size();
  LOGD(STRONGSORT) << "FeatureMatch) Track scale, detects " << detect_num << " tracks " << track_num;
  // no tracks, first enter
  if (tracks_.empty()) {
    tracks_.reserve(detect_num);
    for (size_t i = 0; i < detect_num; ++i) {
      InitNewTrack(detects[i]);
      tracks->emplace_back(detects[i]);
      tracks->rbegin()->track_id = -1;
      tracks->rbegin()->detect_id = i;
    }
  } else {
    detects_ = &detects;
    unconfirmed_track_.clear();
    confirmed_track_.clear();
    for (size_t i = 0; i < track_num; ++i) {
      // update track indices
      if (tracks_[i].state == TrackState::CONFIRMED && tracks_[i].has_feature) {
        confirmed_track_.push_back(i);
      } else {
        unconfirmed_track_.push_back(i);
      }
      tracks_[i].time_since_last_update++;
      // tracks_[i].pos;
      if(fm_->CMC_method_ == 1){
        tracks_[i].kf.ApplyCMC_1(warp_matrix_);
      }
      else if(fm_->CMC_method_ == 2){
        tracks_[i].kf.ApplyCMC_2(warp_matrix_);
      }
      else{
        LOGE(STRONGSORT)<<"Unsupported CMC method:"<<fm_->CMC_method_;
      }
      tracks_[i].kf.Predict();
      tracks_[i].pos = BoundingBox2Rect(fm_->inv_transformer_func_(tracks_[i].kf.GetCurPos()));
    }
    // LOGE(Track)<<warp_matrix_;

    MatchCascade();
    LOGT(STRONGSORT) << "FeatureMatch) Cascade result, matched " << res_feature_.matches.size()
                << " unmatched detects " << res_feature_.unmatched_detections.size()
                << " unmatched tracks " << res_feature_.unmatched_tracks.size();

    // give first missed object a chance
    std::vector<int> match_iou_track = unconfirmed_track_;
    for (auto &idx : res_feature_.unmatched_tracks) {
      if (tracks_[idx].time_since_last_update == 1) {
        match_iou_track.push_back(idx);
      } else {
        LOGT(STRONGSORT) << "Object " << idx << " missed";
        MarkMiss(&(tracks_[idx]));
      }
    }

    // Matrix iou_cost_matrix = GetIouMat(res_feature_.unmatched_detections, match_iou_track);
    // match with features


    // match with iou
    MatchIou(res_feature_.unmatched_detections, match_iou_track);
    LOGT(STRONGSORT) << "FeatureMatch) IoU result, matched " << res_iou_.matches.size()
                << " unmatched detects " << res_iou_.unmatched_detections.size()
                << " unmatched tracks " << res_iou_.unmatched_tracks.size();

    // update matched
    DetectObject tmp_obj;
    StrongSORTTrackObject *ptrack_obj;
    const DetectObject *pdetect_obj;
    res_feature_.matches.insert(res_feature_.matches.end(), res_iou_.matches.begin(), res_iou_.matches.end());

    tracks->reserve(detect_num);
    for (auto &pair : res_feature_.matches) {
      ptrack_obj = &(tracks_[pair.second]);
      pdetect_obj = &detects[pair.first];
      ptrack_obj->kf.Update(fm_->transformer_func_(pdetect_obj->bbox));

      if (ptrack_obj->has_feature) {
        if(fm_->ema_)
        {
          // use ema to update feature
          for(size_t i = 0; i < pdetect_obj->feature.size(); ++i)
          {
            ptrack_obj->features[0].vec[i] = ptrack_obj->features[0].vec[i] * 0.9 + pdetect_obj->feature[i] * 0.1;
          }
          ptrack_obj->features[0].mold = -1;
        }else
        {
          ptrack_obj->features.emplace_back(pdetect_obj->feature, pdetect_obj->feat_mold);
          if (ptrack_obj->features.size() > fm_->nn_budget_) {
            ptrack_obj->features.erase(ptrack_obj->features.begin());
          }
        }
      }

      ptrack_obj->time_since_last_update = 0;
      ptrack_obj->age++;
      if (ptrack_obj->state == TrackState::TENTATIVE && ptrack_obj->age > fm_->n_init_) {
        LOGD(STRONGSORT) << "new track: " << next_id_;
        ptrack_obj->state = TrackState::CONFIRMED;
        ptrack_obj->track_id = next_id_++;
      }

      // fill the output
      tracks->emplace_back(*pdetect_obj);
      tracks->rbegin()->track_id = ptrack_obj->track_id;
      tracks->rbegin()->detect_id = pair.first;
    }

    // unmatched detections: init new track
    for (auto &idx : res_iou_.unmatched_detections) {
      InitNewTrack(detects[idx]);
      tracks->emplace_back(detects[idx]);
      tracks->rbegin()->track_id = tracks_.rbegin()->track_id;
      tracks->rbegin()->detect_id = idx;
    }

    // unmatched tracks: mark missed
    for (auto idx : res_iou_.unmatched_tracks) {
      LOGT(STRONGSORT) << "Object " << idx << " missed";
      MarkMiss(&(tracks_[idx]));
    }

    // erase dead track object
    for (auto iter = tracks_.begin(); iter != tracks_.end();) {
      if (iter->state == TrackState::DELETED || iter->time_since_last_update > fm_->max_age_) {
        LOGD(STRONGSORT) << "delete track: " << iter->track_id;
        iter = tracks_.erase(iter);
      } else {
        iter++;
      }
    }
  }
}

void StrongSORTTrack::UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) {
  if (!tracks) {
    THROW_EXCEPTION(Exception::INVALID_ARG, "parameter 'tracks' is nullptr");
  }
  fm_p_->UpdateFrame(frame, detects, tracks);
}

}  // namespace edk
