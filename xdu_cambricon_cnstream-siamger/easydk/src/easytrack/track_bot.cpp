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
#include "bot_kalmanfilter.h"
#include "match.h"
#include "matrix.h"
#include "track_data_type.h"
#include "util.hpp"

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

// chi2inv95 at 4 degree of freedom
constexpr const float gating_threshold = 9.4877;

namespace edk {

struct BotTrackTrackObject {
  // KalmanFilter kf;
  BotKalmanFilter bkf;
  std::vector<Feature> features;
  Rect pos;
  int class_id;
  int track_id = -1;
  float score;
  TrackState state;
  int age = 1;
  int time_since_last_update = 0;
  bool has_feature = false;

  BotTrackTrackObject() = default;
  BotTrackTrackObject(BotTrackTrackObject&&) = default;
  BotTrackTrackObject& operator=(BotTrackTrackObject&&) = default;
  BotTrackTrackObject(const BotTrackTrackObject&) = delete;
  BotTrackTrackObject& operator=(const BotTrackTrackObject&) = delete;
};

class BotTrackPrivate {
 private:
  explicit BotTrackPrivate(BotTrackTrack *fm) {
    fm_ = fm;
    match_algo_ = MatchAlgorithm::Instance();
  }
  void MatchCascade(const std::vector<int>& detect_indices);
  void MatchIou(const std::vector<int>& detect_matrices, const std::vector<int>& track_matrices, const float& max_iou_distance);
  void InitNewTrack(const DetectObject &obj);
  void MarkMiss(BotTrackTrackObject *track);
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks);

  BotTrackTrack *fm_;

  MatchAlgorithm *match_algo_;
  std::vector<BotTrackTrackObject> tracks_;
  std::vector<int> confirmed_unconfirmed_track_;
  std::vector<int> unconfirmed_track_;
  std::vector<int> confirmed_track_;

  std::vector<int> assignments_;
  MatchResult res_feature_;
  MatchResult res_iou_;
  const Objects *detects_ = nullptr;


  bool first_mat_=true;
  cv::Mat warp_matrix_;
  cv::Mat pre_gray_mat_;
  cv::Mat cur_gray_mat_;
  cv::TermCriteria criteria_;

  uint64_t next_id_ = 0;
  friend class BotTrackTrack;
};  // class BotTrackPrivate

BotTrackTrack::BotTrackTrack() { fm_p_ = new BotTrackPrivate(this); }

BotTrackTrack::~BotTrackTrack() {
  delete fm_p_;
} 

void BotTrackTrack::SetParams(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age,
                                  int n_init, float track_high_threshold, float track_low_threshold, int coordinate,
                                  int max_count, double epsilon, float scale, int sz, int CMC_method) {
  // clang-format off
  LOGD(BotTRACK) << "BotTrackTrack Params -----\n"
              << "\n\t max cosine distance: " << max_cosine_distance
              << "\n\t max IoU distance: " << max_iou_distance
              << "\n\t max age: " << max_age
              << "\n\t nn budget: " << nn_budget
              << "\n\t n_init: " << n_init
              << "\n\t track_high_threshold: " << track_high_threshold
              << "\n\t track_low_threshold: " << track_low_threshold
              << "\n\t coordinate: " << coordinate;
  // clang-format on
  max_cosine_distance_ = max_cosine_distance;
  max_iou_distance_ = max_iou_distance;
  nn_budget_ = nn_budget;
  max_age_ = max_age;
  n_init_ = n_init;
  track_high_threshold_ = track_high_threshold;
  track_low_threshold_ = track_low_threshold;
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
  
  max_count_ = max_count;
  epsilon_ = epsilon;
  scale_ = scale; 
  sz_ = sz;
  CMC_method_ = CMC_method;

}

void BotTrackPrivate::MatchCascade(const std::vector<int>& detect_indices) {
  const Objects &det_objs = *detects_;
  Matrix cost_matrix;
  MatchResult &res = res_feature_;

  // refresh feature match result
  res.Clean();
  res.unmatched_detections = detect_indices;

  if (confirmed_unconfirmed_track_.empty() || detect_indices.empty()) return;

  for (auto &idx : detect_indices) {
    det_objs[idx].feat_mold = L2Norm(det_objs[idx].feature);
  }

  std::set<int> remained_detections;
  remained_detections.insert(res.unmatched_detections.begin(), res.unmatched_detections.end());
  LOGT(BotTRACK) << "MatchCascade) Match scale, detects " << detect_indices.size() << " tracks " << confirmed_unconfirmed_track_.size();

  std::map<int, std::vector<int>> age_track_indices;
  for (size_t t = 0; t < confirmed_unconfirmed_track_.size(); ++t) {
    int age = tracks_[confirmed_unconfirmed_track_[t]].time_since_last_update - 1;
    age_track_indices[age].push_back(confirmed_unconfirmed_track_[t]);
  }

  for (int age = 0; age < fm_->max_age_; ++age) {
    LOGA(BotTRACK) << "Cascade: Number of remained detections ----- " << remained_detections.size();
    // no remained detections or no confirmed tracks, end match
    if (remained_detections.empty() || confirmed_unconfirmed_track_.empty()) break;

    // get all confirmed tracks with same age
    auto track_indices_iter = age_track_indices.find(age);
    if (track_indices_iter == age_track_indices.end()) {
      LOGA(BotTRACK) << "Cascade: No tracks for age " << age << " round, continue";
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
      measurements.emplace_back(fm_->transformer_func_(det_objs[res.unmatched_detections[i]].bbox));
    }
    for (size_t i = 0; i < tra_num; ++i) {
      Matrix gating_dist = tracks_[track_indices[i]].bkf.GatingDistance(measurements);
      for (size_t j = 0; j < det_num; ++j) {
        auto& det = det_objs[res.unmatched_detections[j]];
        cost_matrix(i, j) = match_algo_->Distance(tracks_[track_indices[i]].features,
                                                  Feature(det.feature, det.feat_mold));
        if (cost_matrix(i, j) > fm_->max_cosine_distance_ || gating_dist(0, j) > gating_threshold) {
          LOGA(BotTRACK) << "object " << i << " - " << j << " feature distance is larger than max_cosine_distance";
          cost_matrix(i, j) = fm_->max_cosine_distance_ + 1e-5;
        }
      }
    }

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

void BotTrackPrivate::MatchIou(const std::vector<int>& detect_indices,
                                   const std::vector<int>& track_indices, const float& max_iou_distance) {
  MatchResult &res = res_iou_;
  // clean iou match result
  res.Clean();

  if (detect_indices.empty()) {
    LOGD(BotTRACK) << "No remained detections to process IoU match";
    res.unmatched_tracks = track_indices;
    return;
  } else if (track_indices.empty()) {
    LOGD(BotTRACK) << "No remained track objects to process IoU match";
    res.unmatched_detections = detect_indices;
    return;
  }
  uint32_t detect_num = detect_indices.size();
  uint32_t track_num = track_indices.size();
  LOGT(BotTRACK) << "MatchIoU) Match scale, detects " << detect_num << " tracks " << track_num;
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
    if (assignments_[i] < 0 || cost_matrix(i, assignments_[i]) > max_iou_distance) {
      res.unmatched_tracks.push_back(track_indices[i]);
    } else {
      res.matches.emplace_back(std::make_pair(detect_indices[assignments_[i]], track_indices[i]));
      remained_detections.erase(detect_indices[assignments_[i]]);
    }
  }

  res.unmatched_detections.insert(res.unmatched_detections.end(), remained_detections.begin(),
                                  remained_detections.end());
}

void BotTrackPrivate::InitNewTrack(const DetectObject &det) {
  BotTrackTrackObject obj;
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
  obj.bkf.Initiate(fm_->transformer_func_(det.bbox));
  tracks_.emplace_back(std::move(obj));
}

void BotTrackPrivate::MarkMiss(BotTrackTrackObject *track) {
  if (track->state == TrackState::TENTATIVE || track->time_since_last_update > fm_->max_age_) {
    track->state = TrackState::DELETED;
  }
}

void BotTrackPrivate::UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) {
  
  size_t sz = fm_->sz_;
  float scale = fm_->scale_;
  cv::Size raw;
  cur_gray_mat_ = frame.data.clone();
  // LOG(BotTrack) << frame.data;
  
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
    LOGD(BOTTrack) << cur_gray_mat_;
    LOGD(BOTTrack) << pre_gray_mat_;
    warp_matrix_ = cv::Mat::eye(2, 3, CV_32F);  // for MOTION_EUCLIDEAN 
    criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, fm_->max_count_, fm_->epsilon_);
    first_mat_ = false;
  }else if(!cur_gray_mat_.empty()&&!pre_gray_mat_.empty())
  {
    // Run the ECC algorithm. The results are stored in warp_matrix.
    warp_matrix_ = cv::Mat::eye(2, 3, CV_32F);  // for MOTION_EUCLIDEAN
    LOGD(BotTrack) << pre_gray_mat_;
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
    if (cur_gray_mat_.empty() && pre_gray_mat_.empty() )
      LOGE(BOTTRACK)<<"both empty!";
    else if (cur_gray_mat_.empty())
      LOGE(BOTTRACK)<<"curr empty!";
    else if (pre_gray_mat_.empty())
      LOGE(BOTTRACK)<<"pre empty!";
  }
  
  // set feat_mold to -1 means mold has not been computed
  for (auto& obj : detects) {
    obj.feat_mold = -1;
  }



  uint32_t detect_num = detects.size();
  uint32_t track_num = tracks_.size();
  LOGD(BotTRACK) << "BotTrack) Track scale, detects " << detect_num << " tracks " << track_num;

  // seperate detects into high conf detects and low conf detects
  std::vector<int> high_conf_dets;
  std::vector<int> low_conf_dets;
  high_conf_dets.reserve(detect_num);
  low_conf_dets.reserve(detect_num);

  // set feat_mold to -1 means mold has not been computed
  for(size_t i = 0; i < detect_num; ++i)
  {
    detects[i].feat_mold = -1;
    if(detects[i].score >= fm_->track_high_threshold_)
      high_conf_dets.emplace_back(i);
    else if(detects[i].score >= fm_->track_low_threshold_)
      low_conf_dets.emplace_back(i);
  }
  LOGD(BotTRACK) << " high conf dets: " << high_conf_dets.size()
              << " low conf dets: " << low_conf_dets.size();

  // no tracks, first enter
  if (tracks_.empty()) {
    tracks_.reserve(detect_num);
    for (size_t i = 0; i < high_conf_dets.size(); ++i) {
      InitNewTrack(detects[high_conf_dets[i]]);
      tracks->emplace_back(detects[high_conf_dets[i]]);
      tracks->rbegin()->track_id = -1;
      tracks->rbegin()->detect_id = high_conf_dets[i];
    }
  } else {
    detects_ = &detects;
    confirmed_unconfirmed_track_.clear();
    confirmed_unconfirmed_track_.resize(track_num);
    std::iota(confirmed_unconfirmed_track_.begin(), confirmed_unconfirmed_track_.end(), 0);

    for (size_t i = 0; i < track_num; ++i) {
      if (tracks_[i].state == TrackState::CONFIRMED && tracks_[i].has_feature) {
        confirmed_track_.push_back(i);
      } else {
        unconfirmed_track_.push_back(i);
      }
      tracks_[i].time_since_last_update++;
      // tracks_[i].pos;
      
    
      tracks_[i].bkf.ApplyCMC_2(warp_matrix_);

      
      tracks_[i].bkf.Predict();
      tracks_[i].pos = BoundingBox2Rect(fm_->inv_transformer_func_(tracks_[i].bkf.GetCurPos()));
    }

    if(!tracks_[0].has_feature)
    {
      // match high conf detects with iou
      LOGD(BotTRACK) << "match high conf detects with iou";
      MatchIou(high_conf_dets, confirmed_unconfirmed_track_, 0.8f);
      res_feature_ = res_iou_;
      LOGD(BotTRACK) << "BotTrack) IOU 1 result, matched " << res_feature_.matches.size()
                  << " unmatched detects " << res_feature_.unmatched_detections.size()
                  << " unmatched tracks " << res_feature_.unmatched_tracks.size();
                  
    }else
    {
      // match high conf detects with features
      LOGD(BotTRACK) << "match high conf detects with features";
      MatchCascade(high_conf_dets);
      LOGD(BotTRACK) << "BotTrack) Cascade result, matched " << res_feature_.matches.size()
                  << " unmatched detects " << res_feature_.unmatched_detections.size()
                  << " unmatched tracks " << res_feature_.unmatched_tracks.size();
    }
    
    // match low conf detects with iou
    MatchIou(low_conf_dets, res_feature_.unmatched_tracks, 0.5f);
    LOGD(BotTRACK) << "BotTrack) IoU 2 result, matched " << res_iou_.matches.size()
                << " unmatched detects " << res_iou_.unmatched_detections.size()
                << " unmatched tracks " << res_iou_.unmatched_tracks.size();
    res_feature_.matches.insert(res_feature_.matches.end(), res_iou_.matches.begin(), res_iou_.matches.end());
    
    // match unmatched high conf detects with iou
    std::vector<int> match_iou_track;
    for (auto &idx : res_iou_.unmatched_tracks) {
      if (tracks_[idx].time_since_last_update == 1 ) {
        match_iou_track.push_back(idx);
      }else {
        LOGD(TRACK) << "Object " << idx << " missed";
        MarkMiss(&(tracks_[idx]));
      }
    }
    MatchIou(res_feature_.unmatched_detections, match_iou_track, 0.7f);
    LOGD(BotTRACK) << "BotTrack) IoU 3 result, matched " << res_iou_.matches.size()
                << " unmatched detects " << res_iou_.unmatched_detections.size()
                << " unmatched tracks " << res_iou_.unmatched_tracks.size();

    // update matched
    DetectObject tmp_obj;
    BotTrackTrackObject *ptrack_obj;
    const DetectObject *pdetect_obj;
    res_feature_.matches.insert(res_feature_.matches.end(), res_iou_.matches.begin(), res_iou_.matches.end());
    tracks->reserve(detect_num);
    for (auto &pair : res_feature_.matches) {
      ptrack_obj = &(tracks_[pair.second]);
      pdetect_obj = &detects[pair.first];
      ptrack_obj->bkf.Update(fm_->transformer_func_(pdetect_obj->bbox));

      if (ptrack_obj->has_feature) {
        ptrack_obj->features.emplace_back(pdetect_obj->feature, pdetect_obj->feat_mold);
        if (ptrack_obj->features.size() > fm_->nn_budget_) {
          ptrack_obj->features.erase(ptrack_obj->features.begin());
        }
      }

      ptrack_obj->time_since_last_update = 0;
      ptrack_obj->age++;
      if (ptrack_obj->state == TrackState::TENTATIVE && ptrack_obj->age > fm_->n_init_) {
        LOGD(BotTRACK) << "new track: " << next_id_;
        ptrack_obj->state = TrackState::CONFIRMED;
        ptrack_obj->track_id = next_id_++;
      }

      // fill the output
      tracks->emplace_back(*pdetect_obj);
      tracks->rbegin()->track_id = ptrack_obj->track_id;
      tracks->rbegin()->detect_id = pair.first;

    }


    // unmatched high conf detections: init new track
    for (auto &idx : res_iou_.unmatched_detections) {
      if(detects[idx].score < fm_->track_high_threshold_) continue;
      InitNewTrack(detects[idx]);
      tracks->emplace_back(detects[idx]);
      tracks->rbegin()->track_id = tracks_.rbegin()->track_id;
      tracks->rbegin()->detect_id = idx;
    }

    // unmatched tracks: mark missed
    for (auto idx : res_iou_.unmatched_tracks) {
      LOGT(BotTRACK) << "Object " << idx << " missed";
      MarkMiss(&(tracks_[idx]));
    }

    // erase dead track object
    for (auto iter = tracks_.begin(); iter != tracks_.end();) {
      if (iter->state == TrackState::DELETED || iter->time_since_last_update > fm_->max_age_) {
        LOGD(BotTRACK) << "delete track: " << iter->track_id;
        iter = tracks_.erase(iter);
      } else {
        iter++;
      }
    }
  }
}

void BotTrackTrack::UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) {
  if (!tracks) {
    THROW_EXCEPTION(Exception::INVALID_ARG, "parameter 'tracks' is nullptr");
  }
  fm_p_->UpdateFrame(frame, detects, tracks);
}

}  // namespace edk
