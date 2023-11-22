/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights resized
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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"
#include "siamger_postproc.hpp"

#include <stdio.h>

extern std::vector<float> update_size;
extern std::vector<float> update_center;
// #define DEBUG_DETECTION
/**
 * @brief Video postprocessing for Siamger neural network
 * The input frame of the model should keep aspect ratio.
 */
class VideoPostprocSiamger : public cnstream::SiamgerPostproc {
 public:
  /**
   * @brief User process. Postprocess on outputs of YOLOv6 neural network and fill data to frame.
   *
   * @param output_data: the raw output data from neural network
   * @param model_output: the raw neural network output data
   * @param model_info: model information, e.g., input/output number, shape and etc.
   *
   * @return return true if succeed
   */
  bool Execute(infer_server::InferData* output_data, const infer_server::ModelIO& model_output,
               const infer_server::ModelInfo* model_info) override;

 private:
  DECLARE_REFLEX_OBJECT_EX(VideoPostprocSiamger, cnstream::SiamgerPostproc);

  
  int STRIDE_ = 16;
  int INSTANCE_SIZE_ = 255;
  int EXEMPLAR_SIZE_ = 127;
  int BASE_SIZE_ = 7;
  int SCORE_SIZE_ = 16; 
  float CONTEXT_AMOUNT_ = 0.5;
  float PENALTY_K_ = 0.15;
  float WINDOW_INFLUENCE_ = 0.455;
  float TRACK_LR_ = 0.37;

  std::vector<float> tracker_posttrack(const float* net0_output, const float* net1_output);
   

  float change(float r)
  {
    if (r >= 1/r) return r;
    else return 1/r;
  }

  float sz(float w, float h)
  {
    return sqrt((1.5 * w + 0.5 * h) * (1.5 * h + 0.5 * w));
  }

  std::vector<float> change_sc(std::vector<std::vector<float>> pred_bbox, float scale_z)
  {
    std::vector<float> sz_result(256);
    for (int i = 0;i < 256;++i)
    {
      sz_result[i] = change(sz(pred_bbox[i][2], pred_bbox[i][3]) / sz(size_[0] * scale_z, size_[1] * scale_z));
    }
    return sz_result;
  }

  std::vector<float> change_rc(std::vector<std::vector<float>> pred_bbox)
  {
    std::vector<float> zc_result(256);
    for (int i = 0;i < 256;++i)
    {
      zc_result[i] = change((size_[0]/size_[1]) /(pred_bbox[i][2]/pred_bbox[i][3]));
    }
    return zc_result;
  }

  std::vector<std::vector<float>> _convert_bbox(const float* delta, 
                                   std::vector<std::vector<int>> &points)
  {
    std::vector<std::vector<float>> bbox(256, std::vector<float>(4));
    for (int i = 0;i < STRIDE_;++i)
    {
      for (int j = 0;j < STRIDE_;++j)
      {
        for (int k = 0;k < 4;++k)
        {
          if (k / 2 == 0)
          {
            bbox[i * STRIDE_ + j][k] = points[i * STRIDE_ + j][k%2] - delta[i * STRIDE_ * 4 + j * 4 + k];
          }
          else
          {
            bbox[i * STRIDE_ + j][k] = points[i * STRIDE_ + j][k%2] + delta[i * STRIDE_ * 4 + j * 4 + k];
          }
        }
      }
    }

    std::vector<std::vector<float>> res(256, std::vector<float>(4));
    // conrner2Center
    for (int i = 0;i < 256;++i)
    {
      res[i][0] = (bbox[i][0] + bbox[i][2]) * 0.5;
      res[i][1] = (bbox[i][1] + bbox[i][3]) * 0.5;
      res[i][2] = bbox[i][2] - bbox[i][0];
      res[i][3] = bbox[i][3] - bbox[i][1];
    }
    return res;
  }

  std::vector<float> _bbox_clip(float cx, 
                                float cy,
                                float width,
                                float height,
                                float bound_h,
                                float bound_w)
  {
    cx = std::max(0.0f, std::min(cx, bound_w));
    cy = std::max(0.0f, std::min(cy, bound_h));
    width = std::max(10.0f, std::min(width, bound_w));
    height = std::max(10.0f, std::min(height, bound_h));
    std::vector<float> bbox{cx, cy, width, height};
    return bbox;
  }

  std::vector<float> _convert_score(const float* score)
  {
    // 后一半score 的softmax放入vector
    std::vector<float> res;
    res.reserve(256);
    
    for (int i = 0;i < 16;++i)
    {
      for (int j = 0;j < 16;++j)
      {
        res.push_back( exp(*(score + i * 16 * 2 + j * 2 + 1)) / 
          ( exp(*(score + i * 16 * 2 + j * 2 + 0)) + exp(*(score + i * 16 * 2 + j * 2 + 1)) ) );
      }
    }
    
    return res;
  }


};  // class VideoPostprocSiamger

IMPLEMENT_REFLEX_OBJECT_EX(VideoPostprocSiamger, cnstream::SiamgerPostproc);

bool VideoPostprocSiamger::Execute(infer_server::InferData* output_data, const infer_server::ModelIO& model_output,
                                  const infer_server::ModelInfo* model_info) {
  LOGF_IF(VideoPostprocSiamger, model_info->InputNum() != 2) << "VideoPostprocSiamger: model input number is not equal to 2";
  LOGF_IF(VideoPostprocSiamger, model_info->OutputNum() != 2) << "VideoPostprocSiamger: model output number is not equal to 2";
  LOGF_IF(VideoPostprocSiamger, model_output.buffers.size() != 2) << "VideoPostprocSiamger: model result size is not equal to 2";

  cnstream::CNFrameInfoPtr frame = output_data->GetUserData<cnstream::CNFrameInfoPtr>();
  cnstream::CNInferObjsPtr objs_holder = frame->collection.Get<cnstream::CNInferObjsPtr>(cnstream::kCNInferObjsTag);
  cnstream::CNObjsVec &objs = objs_holder->objs_;

  const auto output0_sp = model_info->OutputShape(0); // cls
  const auto output1_sp = model_info->OutputShape(1); // loc

  int w_idx = 2;
  int h_idx = 1;
  if (model_info->InputLayout(0).order == infer_server::DimOrder::NCHW) {
    w_idx = 3;
    h_idx = 2;
  }
  
  const float* net0_output = reinterpret_cast<const float*>(model_output.buffers[0].Data()); // cls
  const float* net1_output = reinterpret_cast<const float*>(model_output.buffers[1].Data()); // loc
  
  // debug 
  // for (int i = 0;i < 256;i++) {
  //   LOGE(VideoPostprocSiamger_test) << net1_output[i] ;
  // }
  
  const int model_input_h = static_cast<int>(output0_sp[h_idx]);  
  const int model_input_w = static_cast<int>(output0_sp[w_idx]);
  std::vector<int> use_unused{model_input_h, model_input_w};

  const int img_w = frame->collection.Get<cnstream::CNDataFramePtr>(cnstream::kCNDataFrameTag)->width;
  const int img_h = frame->collection.Get<cnstream::CNDataFramePtr>(cnstream::kCNDataFrameTag)->height;


  std::vector<float> bbox_res;
  bbox_res = tracker_posttrack(net0_output, net1_output);

  auto obj = std::make_shared<cnstream::CNInferObject>();



  auto range_0_1 = [](float num) { return std::max(.0f, std::min(1.0f, num)); };

  float left = (bbox_res[0] - static_cast<float>(bbox_res[2] / 2)) / img_w;
  float right = (bbox_res[0] + static_cast<float>(bbox_res[2] / 2)) / img_w;
  float top = (bbox_res[1] - static_cast<float>(bbox_res[3] / 2)) / img_h;
  float bottom = (bbox_res[1] + static_cast<float>(bbox_res[3] / 2)) / img_h;

  left = range_0_1(left);
  right = range_0_1(right);
  top = range_0_1(top);
  bottom = range_0_1(bottom);
  
  obj->id = std::to_string(static_cast<int>(1));
  obj->score = bbox_res[4];
  
  obj->bbox.x = left;
  obj->bbox.y = top;
  obj->bbox.w = std::min(1.0f - obj->bbox.x, right - left);
  obj->bbox.h = std::min(1.0f - obj->bbox.y, bottom - top);

  std::lock_guard<std::mutex> objs_mutex(objs_holder->mutex_);
  objs.push_back(obj);
  
  return true;
}

std::vector<float> VideoPostprocSiamger::tracker_posttrack(
                    const float* net0_output, const float* net1_output)
{
  std::vector<float> score;
  score = _convert_score(net0_output);

  std::vector<std::vector<float>> pred_bbox;
  pred_bbox = _convert_bbox(net1_output, points_);

  float w_z = size_[0] + CONTEXT_AMOUNT_ * (size_[0] + size_[1]);
  float h_z = size_[1] + CONTEXT_AMOUNT_ * (size_[0] + size_[1]);
  int s_z = static_cast<int>(round(sqrt(w_z * h_z)));
  float scale_z = float(EXEMPLAR_SIZE_) / float(s_z);

  std::vector<float> s_c;
  std::vector<float> r_c;
  s_c = change_sc(pred_bbox, scale_z);
  r_c = change_rc(pred_bbox);

  std::vector<float> penalty;
  penalty.reserve(256);

  for (int i = 0;i < 256;++i)
  {
    penalty.push_back(exp(-(r_c[i] * s_c[i] - 1)* PENALTY_K_));
  }

  std::vector<float> pscore;
  pscore.reserve(256);
  for (int i = 0;i < 256;++i)
  {
    pscore.push_back(penalty[i] * score[i]);
  }

  for (int i = 0;i < 256;++i)
  {
    pscore[i] = pscore[i] * (1 - WINDOW_INFLUENCE_) + window_[i] * WINDOW_INFLUENCE_;
  }

  float best_score = 0.f;
  int best_idx = 0;
  for (int i = 0;i < 256;++i)
  {
    if (pscore[i] > best_score)
    {
      best_score = pscore[i];
      best_idx = i;
    }
  }

  std::vector<float> bbox;
  bbox.reserve(4);

  for(int i = 0;i < 4;++i)
  {
    bbox.emplace_back(pred_bbox[best_idx][i] / scale_z);
  }

  float lr;
  lr = penalty[best_idx] * score[best_idx] * TRACK_LR_;

  float cx = bbox[0] + center_pos_[0];
  float cy = bbox[1] + center_pos_[1];

  float width = size_[0] * (1 - lr) + bbox[2] * lr;
  float height = size_[1] * (1 - lr) + bbox[3] * lr;

      
  std::vector<float> bbox_res;
  bbox_res = _bbox_clip(cx, cy, width, height, 1080.0f, 1440.0f);
  
  center_pos_[0] = bbox_res[0];
  center_pos_[1] = bbox_res[1];
  size_[0] = bbox_res[2];
  size_[1] = bbox_res[3];
  
  update_center = center_pos_;
  update_size = size_;

  bbox_res.push_back(score[best_idx]); // cx, cy, width, height, score
  return bbox_res;
}






