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

#include <memory>
#include <string>
#include <vector>

#include "cnis/processor.h"
#include "cnstream_frame_va.hpp"
#include "device/mlu_context.h"
#include "feature_extractor.hpp"
#include "profiler/module_profiler.hpp"
#include "track.hpp"

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

namespace cnstream {

static bool STR2BOOL(const std::string &value, bool *ret) {
  if (!ret) return false;
  static const std::set<std::string> true_value_list = {
    "1", "true", "True", "TRUE"
  };
  static const std::set<std::string> false_value_list = {
    "0", "false", "False", "FALSE"
  };

  if (true_value_list.find(value) != true_value_list.end()) {
    *ret = true;
    return true;
  }
  if (false_value_list.find(value) != false_value_list.end()) {
    *ret = false;
    return true;
  }
  return false;
}

struct TrackerContext {
  std::unique_ptr<edk::EasyTrack> processer_ = nullptr;
  TrackerContext() = default;
  ~TrackerContext() = default;
  TrackerContext(const TrackerContext &) = delete;
  TrackerContext &operator=(const TrackerContext &) = delete;
};

thread_local std::unique_ptr<FeatureExtractor> g_feature_extractor;

Tracker::Tracker(const std::string &name) : Module(name) {
  hasTransmit_.store(true);
  param_register_.SetModuleDesc("Tracker is a module for realtime tracking.");
  param_register_.Register("model_path",
                           "The offline model path. Normally offline model is a file"
                           " with cambricon or model extension.");
  param_register_.Register("func_name",
                           "The offline model function name, usually is 'subnet0'."
                           "Works only if backend is CNRT.");
  param_register_.Register("engine_num", "Infer server engine number.");
  param_register_.Register("track_name", "Track algorithm name. Choose from FeatureMatch, and IoUMatch.");
  param_register_.Register("device_id", "Which device will be used. If there is only one device, it might be 0.");
  param_register_.Register("max_cosine_distance", "Threshold of cosine distance.");
  param_register_.Register("max_iou_distance", "Threshold of iou distance.");
  param_register_.Register("max_age", "Object stay alive for [max_age] after disappeared.");
  param_register_.Register("n_init", "After matched [n_init] times in a row, object is turned from TENTATIVE to CONFIRMED.");
  param_register_.Register("n_budget", "Tracker only saves the latest [n_budget] samples of feature for each object.");
  param_register_.Register("bytetrack_threshold", "High threshold of ByteTrack.");
  param_register_.Register("coordinate", "Coordinate kinds of BoundingBox.");
  param_register_.Register("scale", "scale of ECC input.");
  param_register_.Register("sz", "sz of ECC input.");
  param_register_.Register("CMC_method", "method of ECC.");
  param_register_.Register("ema", "use EMA. true/false");
  param_register_.Register("ecc_count", "iter counts of ECC.");
  param_register_.Register("ecc_eps", "ECC eps.");




}

Tracker::~Tracker() { Close(); }

bool Tracker::InitFeatureExtractor(const CNFrameInfoPtr &data) {
  if (!g_feature_extractor) {
    if (!model_) {
      LOGI(TRACK) << "[Track] FeatureExtract model not set, extract feature on CPU";
      g_feature_extractor.reset(new FeatureExtractor(match_func_));
    } else {
      if (!infer_server::SetCurrentDevice(device_id_)) return false;
      g_feature_extractor.reset(new FeatureExtractor(model_, match_func_, device_id_));
      if (!g_feature_extractor->Init(engine_num_)) {
        LOGE(TRACK) << "[Track] Extract feature on MLU. Init extractor failed.";
        g_feature_extractor.reset();
        return false;
      }
    }
  }
  return true;
}

TrackerContext *Tracker::GetContext(const CNFrameInfoPtr &data, const std::string &track_name) {
  TrackerContext *ctx = nullptr;
  std::unique_lock<std::mutex> guard(mutex_);
  auto search = contexts_.find(data->GetStreamIndex());
  if (search != contexts_.end()) {
    // context exists
    ctx = search->second;
  } else {
    ctx = new TrackerContext;
    if(track_name == "ByteTrack")
    {
      edk::ByteTrackTrack *track = new edk::ByteTrackTrack;
      track->SetParams(max_cosine_distance_, n_budget_, max_iou_distance_, max_age_, n_init_, 
                      bytetrack_threshold_, std::max(0.15f, bytetrack_threshold_ - 0.3f), coordinate_);
      ctx->processer_.reset(track);
      contexts_[data->GetStreamIndex()] = ctx;
    }
    else if(track_name == "BotTrack")
    {
      edk::BotTrackTrack *track = new edk::BotTrackTrack;
      track->SetParams(max_cosine_distance_, n_budget_, max_iou_distance_, max_age_, n_init_, 
                      bytetrack_threshold_, std::max(0.15f, bytetrack_threshold_ - 0.3f), coordinate_,
                      100, 1e-5, scale_, sz_,CMC_method_);
      ctx->processer_.reset(track);
      contexts_[data->GetStreamIndex()] = ctx;
    }
    else if(track_name == "StrongSORT"){
      edk::StrongSORTTrack *track = new edk::StrongSORTTrack;
      // TODO: add parameters in configs
      track->SetParams(max_cosine_distance_, n_budget_, max_iou_distance_, max_age_, n_init_, coordinate_, ema_, ecc_count_, ecc_eps_, scale_,sz_,CMC_method_);
      ctx->processer_.reset(track);
      contexts_[data->GetStreamIndex()] = ctx;
    }
    else if(track_name == "OCSORT")
    {
      edk::OCSORTTrack *track = new edk::OCSORTTrack;
      track->SetParams(max_cosine_distance_, n_budget_, max_iou_distance_, max_age_, n_init_, 
                      bytetrack_threshold_, std::max(0.15f, bytetrack_threshold_ - 0.3f), coordinate_);
      ctx->processer_.reset(track);
      contexts_[data->GetStreamIndex()] = ctx;
    }
    else{
      // default
      // FeatureMatch and IOUMatch
      edk::FeatureMatchTrack *track = new edk::FeatureMatchTrack;
      track->SetParams(max_cosine_distance_, n_budget_, max_iou_distance_, max_age_, n_init_, coordinate_);
      ctx->processer_.reset(track);
      contexts_[data->GetStreamIndex()] = ctx;
    }

  }
  return ctx;
}

bool Tracker::Open(ModuleParamSet paramSet) {
  ModuleProfiler* profiler = this->GetProfiler();
  if (profiler) {
    if (!profiler->RegisterProcessName("Tracker")) {
      LOGE(TRACK) << "Register [" << "Tracker" << "] failed.";
      return false;
    }
  }
  bool use_magicmind = infer_server::Predictor::Backend() == "magicmind";
  if (use_magicmind) {
    if (paramSet.find("model_path") != paramSet.end()) {
      model_pattern1_ = paramSet["model_path"];
      model_pattern1_ = GetPathRelativeToTheJSONFile(model_pattern1_, paramSet);
    }
    if (!model_pattern1_.empty())
      model_ = infer_server::InferServer::LoadModel(model_pattern1_);
  } else {
    if (paramSet.find("model_path") != paramSet.end()) {
      model_pattern1_ = paramSet["model_path"];
      model_pattern1_ = GetPathRelativeToTheJSONFile(model_pattern1_, paramSet);
    }

    std::string model_pattern2_ = "subnet0";
    if (paramSet.find("func_name") != paramSet.end()) {
      model_pattern2_ = paramSet["func_name"];
    }
    if (!model_pattern1_.empty() && !model_pattern2_.empty())
      model_ = infer_server::InferServer::LoadModel(model_pattern1_, model_pattern2_);
  }

  if (paramSet.find("max_cosine_distance") != paramSet.end()) {
    max_cosine_distance_ = std::stof(paramSet["max_cosine_distance"]);
  }

  if (paramSet.find("max_iou_distance") != paramSet.end()) {
    max_iou_distance_ = std::stof(paramSet["max_iou_distance"]);
  }
  
  if (paramSet.find("bytetrack_threshold") != paramSet.end()) {
    bytetrack_threshold_ = std::stof(paramSet["bytetrack_threshold"]);
  }

  if (paramSet.find("max_age") != paramSet.end()) {
    max_age_ = std::stoi(paramSet["max_age"]);
  }

  if (paramSet.find("n_init") != paramSet.end()) {
    n_init_ = std::stoi(paramSet["n_init"]);
  }

  if (paramSet.find("n_budget") != paramSet.end()) {
    n_budget_ = std::stoi(paramSet["n_budget"]);
  }

  if (paramSet.find("engine_num") != paramSet.end()) {
    engine_num_ = std::stoi(paramSet["engine_num"]);
  }

  if (paramSet.find("device_id") != paramSet.end()) {
    device_id_ = std::stoi(paramSet["device_id"]);
  }

  if(paramSet.find("coordinate") != paramSet.end()){
    std::string coordinate = paramSet["coordinate"];
    if(coordinate == "xyah"){
      LOGI(TRACK) <<"Select xyah coordinate";
      coordinate_ = 0;
    }
    else if(coordinate == "xywh"){
      LOGI(TRACK) <<"Select xywh coordinate";
      coordinate_ = 1;
    }
    else if(coordinate == "xyar"){
      LOGI(TRACK) <<"Select xyar coordinate";
      coordinate_ = 2;
    }else
    {
      LOGW(TRACK) << "Unsupported coordinate type: " << coordinate << ". Set to default: xyah";
      coordinate_ = 0;
    }
   if(paramSet.find("scale") != paramSet.end()){
    scale_ = std::stof(paramSet["scale"]);
   }
   if(paramSet.find("sz") != paramSet.end()){
    sz_ = std::stoi(paramSet["sz"]);
   }
   if(paramSet.find("CMC_method") != paramSet.end()){
    CMC_method_ = std::stoi(paramSet["CMC_method"]);
   }
   if(paramSet.find("ema") != paramSet.end()){
    if(paramSet["ema"] == "false") ema_ = false;
   }
   if(paramSet.find("ecc_count") != paramSet.end()){
    ecc_count_ = std::stoi(paramSet["ecc_count"]);
   } 
   if(paramSet.find("ecc_eps") != paramSet.end()){
    ecc_eps_ = std::stof(paramSet["ecc_eps"]);
   }
  }

  track_name_ = "FeatureMatch";   // default track algorithm
  if (paramSet.find("track_name") != paramSet.end()) {
    track_name_ = paramSet["track_name"];
  }

  if (track_name_ == "FeatureMatch" || track_name_ == "StrongSORT")
  {
    need_feature_ = true;
    LOGI(TRACK) << "Select track type: " << track_name_;
  }else if(track_name_ == "IoUMatch")
  {
    need_feature_ = false;
    LOGI(TRACK) << "Select track type: " << track_name_;
  }else if(track_name_ == "ByteTrack" || track_name_ == "OCSORT" || track_name_ == "BotTrack" )
  {
    need_feature_ = false;
    LOGI(TRACK) << "Select track type: " << track_name_;
    if (paramSet.find("need_feature") != paramSet.end()) {
      STR2BOOL(paramSet["need_feature"], &need_feature_);
    }
  }else{
    LOGE(TRACK) << "Unsupported track type: " << track_name_;
    return -1;
  }
  

  match_func_ = [this](const CNFrameInfoPtr data, bool valid) {
    if (!valid) {
      PostEvent(EventType::EVENT_ERROR, "Extract feature failed");
      return;
    }
    CNInferObjsPtr objs_holder = data->collection.Get<CNInferObjsPtr>(kCNInferObjsTag);

    std::vector<edk::DetectObject> in, out;
    in.reserve(objs_holder->objs_.size());
    for (size_t i = 0; i < objs_holder->objs_.size(); i++) {
      edk::DetectObject obj;
      obj.label = std::stoi(objs_holder->objs_[i]->id);
      obj.score = objs_holder->objs_[i]->score;
      obj.bbox.x = objs_holder->objs_[i]->bbox.x;
      obj.bbox.y = objs_holder->objs_[i]->bbox.y;
      obj.bbox.width = objs_holder->objs_[i]->bbox.w;
      obj.bbox.height = objs_holder->objs_[i]->bbox.h;
      obj.feature = objs_holder->objs_[i]->GetFeature("track");
      in.emplace_back(obj);
    }
    cnstream::RecordKey key = std::make_pair(data->stream_id, data->timestamp);

    if (this->GetProfiler()) {
      this->GetProfiler()->RecordProcessStart("Tracker", key);
    }

    edk::TrackFrame frame;
    CNDataFramePtr framedata = data->collection.Get<CNDataFramePtr>(kCNDataFrameTag);
    if(track_name_ == "StrongSORT" || track_name_ == "BotTrack"){
      frame.data = data->collection.Get<CNDataFramePtr>(kCNDataFrameTag)->ImageBGR();
      frame.width = framedata->width;
      frame.height = framedata->height;
    }
    GetContext(data, track_name_)->processer_->UpdateFrame(frame, in, &out);

    if (this->GetProfiler()) {
      this->GetProfiler()->RecordProcessEnd("Tracker", key);
    }
    
    for (size_t i = 0; i < out.size(); i++) {
      objs_holder->objs_[out[i].detect_id]->track_id = std::to_string(out[i].track_id);
    }
    TransmitData(data);
  };

  return true;
}

void Tracker::Close() {
  for (auto &pair : contexts_) {
    delete pair.second;
  }
  contexts_.clear();
  g_feature_extractor.reset();
}

int Tracker::Process(std::shared_ptr<CNFrameInfo> data) {
  if (data->GetStreamIndex() >= GetMaxStreamNumber()) {
    return -1;
  }
  if (need_feature_ && !InitFeatureExtractor(data)) {
    LOGE(TRACK) << "Init Feature Extractor Failed.";
    return -1;
  }

  if (!data->IsEos()) {
    CNDataFramePtr frame = data->collection.Get<CNDataFramePtr>(kCNDataFrameTag);
    if (frame->width <= 0 || frame->height <= 0) {
      LOGE(TRACK) << "Frame width and height can not be lower than 0.";
      return -1;
    }
    bool have_obj = data->collection.HasValue(kCNInferObjsTag);
    if (have_obj) {
      CNInferObjsPtr objs_holder = data->collection.Get<CNInferObjsPtr>(kCNInferObjsTag);
      for (size_t idx = 0; idx < objs_holder->objs_.size(); ++idx) {
        auto &obj = objs_holder->objs_[idx];
        cnstream::CNInferBoundingBox &bbox = obj->bbox;
        bbox.x = CLIP(bbox.x);
        bbox.w = CLIP(bbox.w);
        bbox.y = CLIP(bbox.y);
        bbox.h = CLIP(bbox.h);
        bbox.w = (bbox.x + bbox.w > 1.0) ? (1.0 - bbox.x) : bbox.w;
        bbox.h = (bbox.y + bbox.h > 1.0) ? (1.0 - bbox.y) : bbox.h;
        
      }
    }

    if (need_feature_) {
      // async extract feature
      if (!g_feature_extractor->ExtractFeature(data)) {
        LOGE(TRACK) << "Extract Feature failed";
        return -1;
      }
    } else {
      match_func_(data, true);
    }
  } else {
    if (need_feature_) {
      g_feature_extractor->WaitTaskDone(data->stream_id);
    }
    TransmitData(data);
  }
  return 0;
}

bool Tracker::CheckParamSet(const ModuleParamSet &paramSet) const {
  bool ret = true;
  ParametersChecker checker;
  for (auto &it : paramSet) {
    if (!param_register_.IsRegisted(it.first)) {
      LOGW(TRACK) << "[Tracker] Unknown param: " << it.first;
    }
  }

  if (paramSet.find("model_path") != paramSet.end()) {
    if (!checker.CheckPath(paramSet.at("model_path"), paramSet)) {
      LOGE(TRACK) << "[Tracker] [model_path] : " << paramSet.at("model_path") << " non-existence.";
      ret = false;
    }
  }

  if (paramSet.find("track_name") != paramSet.end()) {
    std::string track_name = paramSet.at("track_name");
    if (track_name != "StrongSORT" && track_name != "StrongFeatureMatch" && track_name != "FeatureMatch" && track_name != "BotTrack" && track_name != "IoUMatch") {
      LOGE(TRACK) << "[Tracker] [track_name] : Unsupported tracker type " << track_name;
      ret = false;
    }
  }

  std::string err_msg;
  if (paramSet.find("device_id") != paramSet.end()) {
    if (!checker.IsNum({"device_id"}, paramSet, err_msg)) {
      LOGE(TRACK) << "[Tracker] " << err_msg;
      ret = false;
    }
  }

  if (paramSet.find("engine_num") != paramSet.end()) {
    if (!checker.IsNum({"engine_num"}, paramSet, err_msg)) {
      LOGE(TRACK) << "[Tracker] " << err_msg;
      ret = false;
    }
  }

  if (paramSet.find("max_cosine_distance") != paramSet.end()) {
    if (!checker.IsNum({"max_cosine_distance"}, paramSet, err_msg)) {
      LOGE(TRACK) << "[Tracker] " << err_msg;
      ret = false;
    }
  }
  return ret;
}

}  // namespace cnstream
