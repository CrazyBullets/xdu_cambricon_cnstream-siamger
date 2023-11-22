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

/**
 * @file easy_track.h
 * This file contains FeatureMatchTrack class and KcfTrack class.
 * Its purpose is to achieve object tracking.
 */

#ifndef EASYTRACK_EASY_TRACK_H_
#define EASYTRACK_EASY_TRACK_H_

#include <memory>
#include <vector>
#include "opencv2/opencv.hpp"
#include "cxxutil/exception.h"
#include "easyinfer/model_loader.h"

namespace edk {

/**
 * @brief Struct of BoundingBox
 */
struct BoundingBox {
  float x;       ///< Topleft coordinates x
  float y;       ///< Topleft coordinates y
  float width;   ///< BoundingBox width
  float height;  ///< BoundingBox height
};

/**
 * @brief Struct of detection objects.
 */
struct DetectObject {
  /// Object detection label
  int label;
  /// Object detection confidence rate
  float score;
  /// Struct BoundingBox
  BoundingBox bbox;
  /// Object track identification
  int track_id;
  /// Object index in input vector
  int detect_id;
  /**
   * @brief Features of object extraction.
   * @attention The dimension of the feature vector is 128.
   */
  std::vector<float> feature;
  /// internal
  mutable float feat_mold;
};

/// Alias of vector stored DetectObject
using Objects = std::vector<DetectObject>;

/**
 * @brief Track frame stored frame data and information needed in track
 */
struct TrackFrame {
  /**
   * @brief The data of frame.
   * @attention This parameter is used for KcfTrack only.
   */
  cv::Mat data;
  /// The width of trackframe
  uint32_t width;
  /// The height of trackframe
  uint32_t height;

  /// The identification of trackframe
  int64_t frame_id;
  /// The identification of device
  int device_id;

  /**
   * @brief Color space enumeration.
   */
  enum class ColorSpace { GRAY, NV21, NV12, RGB24, BGR24 } format;

  /**
   * @brief Device type enumeration.
   */
  enum class DevType {
    CPU = 0,
    MLU,
  } dev_type;
};

/**
 * @brief EasyTrack class, help for tracking objects.
 */
class EasyTrack {
 public:
  /**
   * @brief Destroy the EasyTrack object.
   */
  virtual ~EasyTrack() {}

  /**
   * @brief Update object status and do track
   *
   * @param frame Track frame
   * @param detects Detected objects
   * @param tracks Tracked objects
   */
  virtual void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) noexcept(false) = 0;
};  // class EasyTrack

class FeatureMatchPrivate;

/**
 * @brief Track objects based on match feature.
 *
 * @note Match tentative and featureless objects using IOU,
 *       and cascade-match confirmed objects using feature cosine distance
 */
class FeatureMatchTrack : public EasyTrack {
 public:
  /**
   * @brief Constructor of the FeatureMatchTrack class.
   */
  FeatureMatchTrack();

  /**
   * @brief Destroy the FeatureMatchTrack object.
   */
  ~FeatureMatchTrack();

  /**
   * @brief Set params related to Tracking algorithm.
   *
   * @param max_cosine_distance Threshold of cosine distance
   * @param nn_budget Tracker only saves the latest [nn_budget] samples of feature for each object
   * @param max_iou_distance Threshold of iou distance
   * @param max_age Object stay alive for [max_age] after disappeared
   * @param n_init After matched [n_init] times in a row, object is turned from TENTATIVE to CONFIRMED
   */
  void SetParams(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init, int coordinate);

  /**
   * @brief Update object status and do tracking using cascade matching and IOU matching.
   *
   * @param frame Track frame
   * @param detects Detected objects
   * @param tracks Tracked objects
   */
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) override;

 private:
  FeatureMatchPrivate *fm_p_;
  friend class FeatureMatchPrivate;
  float max_cosine_distance_ = 0.2;
  float max_iou_distance_ = 0.7;
  int max_age_ = 30;
  int n_init_ = 3;
  uint32_t nn_budget_ = 100;

  BoundingBox (*transformer_func_)(const BoundingBox&);
  BoundingBox (*inv_transformer_func_)(const BoundingBox&);
};  // class FeatureMatchTrack

class OCSORTPrivate;

/**
 * @brief Track objects based on match feature.
 *
 * @note Match tentative and featureless objects using IOU,
 *       and cascade-match confirmed objects using feature cosine distance
 */
class OCSORTTrack : public EasyTrack {
 public:
  /**
   * @brief Constructor of the OCSORTTrack class.
   */
  OCSORTTrack();

  /**
   * @brief Destroy the OCSORTTrack object.
   */
  ~OCSORTTrack();

  /**
   * @brief Set params related to Tracking algorithm.
   *
   * @param max_cosine_distance Threshold of cosine distance
   * @param nn_budget Tracker only saves the latest [nn_budget] samples of feature for each object
   * @param max_iou_distance Threshold of iou distance
   * @param max_age Object stay alive for [max_age] after disappeared
   * @param n_init After matched [n_init] times in a row, object is turned from TENTATIVE to CONFIRMED
   */
  void SetParams(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init, float track_high_threshold, float track_low_threshold, int coordinate);

  /**
   * @brief Update object status and do tracking using cascade matching and IOU matching.
   *
   * @param frame Track frame
   * @param detects Detected objects
   * @param tracks Tracked objects
   */
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) override;

 private:
  OCSORTPrivate *fm_p_;
  friend class OCSORTPrivate;
  float max_cosine_distance_ = 0.2;
  float max_iou_distance_ = 0.7;
  float track_high_threshold_ = 0.6;
  float track_low_threshold_ = 0.3;
  int max_age_ = 30;
  int n_init_ = 3;
  uint32_t nn_budget_ = 100;
  
  BoundingBox (*transformer_func_)(const BoundingBox&);
  BoundingBox (*inv_transformer_func_)(const BoundingBox&);
};  // class OCSORTTrack

class ByteTrackPrivate;

/**
 * @brief Track objects based on match feature.
 *
 * @note Match tentative and featureless objects using IOU,
 *       and cascade-match confirmed objects using feature cosine distance
 */
class ByteTrackTrack : public EasyTrack {
 public:
  /**
   * @brief Constructor of the ByteTrackTrack class.
   */
  ByteTrackTrack();

  /**
   * @brief Destroy the ByteTrackTrack object.
   */
  ~ByteTrackTrack();

  /**
   * @brief Set params related to Tracking algorithm.
   *
   * @param max_cosine_distance Threshold of cosine distance
   * @param nn_budget Tracker only saves the latest [nn_budget] samples of feature for each object
   * @param max_iou_distance Threshold of iou distance
   * @param max_age Object stay alive for [max_age] after disappeared
   * @param n_init After matched [n_init] times in a row, object is turned from TENTATIVE to CONFIRMED
   */
  void SetParams(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init, float track_high_threshold, float track_low_threshold, int coordinate);

  /**
   * @brief Update object status and do tracking using cascade matching and IOU matching.
   *
   * @param frame Track frame
   * @param detects Detected objects
   * @param tracks Tracked objects
   */
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) override;

 private:
  ByteTrackPrivate *fm_p_;
  friend class ByteTrackPrivate;
  float max_cosine_distance_ = 0.2;
  float max_iou_distance_ = 0.7;
  float track_high_threshold_ = 0.6;
  float track_low_threshold_ = 0.3;
  int max_age_ = 30;
  int n_init_ = 3;
  uint32_t nn_budget_ = 100;
  
  BoundingBox (*transformer_func_)(const BoundingBox&);
  BoundingBox (*inv_transformer_func_)(const BoundingBox&);
};  // class ByteTrackTrack

class BotTrackPrivate;

class BotTrackTrack : public EasyTrack{
 public:
  /**
   * @brief Constructor of the BotTrackTrack class.
   */
  BotTrackTrack();

  /**
   * @brief Destroy the BotTrackTrack object.
   */
  ~BotTrackTrack();

  /**
   * @brief Set params related to Tracking algorithm.
   *
   * @param max_cosine_distance Threshold of cosine distance
   * @param nn_budget Tracker only saves the latest [nn_budget] samples of feature for each object
   * @param max_iou_distance Threshold of iou distance
   * @param max_age Object stay alive for [max_age] after disappeared
   * @param n_init After matched [n_init] times in a row, object is turned from TENTATIVE to CONFIRMED
   */
  void SetParams(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init, float track_high_threshold, float track_low_threshold, int coordinate,int max_count, double epsilon, float scale, int sz, int CMC_method);

  /**
   * @brief Update object status and do tracking using cascade matching and IOU matching.
   *
   * @param frame Track frame
   * @param detects Detected objects
   * @param tracks Tracked objects
   */
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) override;
  
private:
  BotTrackPrivate *fm_p_;
  friend class BotTrackPrivate;
  float max_cosine_distance_ = 0.2;
  float max_iou_distance_ = 0.7;
  float track_high_threshold_ = 0.6;
  float track_low_threshold_ = 0.3;
  int max_age_ = 30;
  int n_init_ = 3;
  uint32_t nn_budget_ = 100;

  float scale_ = 0.1;
  int sz_ = 640;
  int max_count_ = 100;
  double epsilon_ = 1e-5;
  float CMC_method_ = 1;
  BoundingBox (*transformer_func_)(const BoundingBox&);
  BoundingBox (*inv_transformer_func_)(const BoundingBox&);
};// class BotTrackTrack

class StrongSORTPrivate;

/**
 * @brief Track objects based on match feature.
 *
 * @note Match tentative and featureless objects using IOU,
 *       and cascade-match confirmed objects using feature cosine distance
 */
class StrongSORTTrack : public EasyTrack {
 public:
  /**
   * @brief Constructor of the StrongSORTTrack class.
   */
  StrongSORTTrack();

  /**
   * @brief Destroy the StrongSORTTrack object.
   */
  ~StrongSORTTrack();

  /**
   * @brief Set params related to Tracking algorithm.
   *
   * @param max_cosine_distance Threshold of cosine distance
   * @param nn_budget Tracker only saves the latest [nn_budget] samples of feature for each object
   * @param max_iou_distance Threshold of iou distance
   * @param max_age Object stay alive for [max_age] after disappeared
   * @param n_init After matched [n_init] times in a row, object is turned from TENTATIVE to CONFIRMED
   */
  void SetParams(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init, int coordinate,
                 bool ema, int max_count, double epsilon, float scale, int sz, int CMC_method);

  /**
   * @brief Update object status and do tracking using cascade matching and IOU matching.
   *
   * @param frame Track frame
   * @param detects Detected objects
   * @param tracks Tracked objects
   */
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) override;

 private:
  StrongSORTPrivate *fm_p_;
  friend class StrongSORTPrivate;
  float max_cosine_distance_ = 0.2;
  float max_iou_distance_ = 0.7;
  int max_age_ = 30;
  int n_init_ = 3;
  uint32_t nn_budget_ = 100;

  bool ema_ = true;
  int max_count_ = 100;
  double epsilon_ = 1e-5;
  float scale_ = 0.1;
  int sz_ = 640;
  float CMC_method_ = 1;
  BoundingBox (*transformer_func_)(const BoundingBox&);
  BoundingBox (*inv_transformer_func_)(const BoundingBox&);
};  // class StrongSORTTrack

class KcfTrackPrivate;

/**
 * @brief Track objects based on KCF
 *
 * @note Track objects using KCF, and match them using IOU
 */
class KcfTrack : public EasyTrack {
 public:
  /**
   * @brief Constructor of the KcfTrack class.
   */
  KcfTrack();

  /**
   * @brief Destroy the KcfTrack object.
   */
  ~KcfTrack();

  /**
   * @brief Set params related to offline model.
   *
   * @param model ModelLoader
   * @param dev_id the id of device
   * @param batch_size Batch size
   */
  void SetModel(std::shared_ptr<ModelLoader> model, int dev_id = 0, uint32_t batch_size = 1);

  /**
   * @brief Set params related to KcfTrack.
   * @param max_iou_distance Threshold of iou distance
   */
  void SetParams(float max_iou_distance);

  /**
   * @brief Update result of objects tracking after kcf and IOU matching.
   * @see edk::EasyTrack::UpdateFrame
   */
  void UpdateFrame(const TrackFrame &frame, const Objects &detects, Objects *tracks) override;

 private:
  KcfTrackPrivate *kcf_p_;
  friend class KcfTrackPrivate;
  float max_iou_distance_ = 0.7;
};  // class KcfTrack

/**
 * @brief Insert DetectObject into the ostream
 *
 * @param os output stream to insert data to
 * @param obj reference to an DetectObject to insert
 *
 * @return reference to output stream
 */
inline std::ostream &operator<<(std::ostream &os, const DetectObject &obj) {
  os << "[Object] label: " << obj.label << " score: " << obj.score << " track_id: " << obj.track_id << '\t'
     << "bbox: " << obj.bbox.x << "  " << obj.bbox.y << "  " << obj.bbox.width << "  " << obj.bbox.height;
  return os;
}

}  // namespace edk

#endif  // EASYTRACK_EASY_TRACK_H_
