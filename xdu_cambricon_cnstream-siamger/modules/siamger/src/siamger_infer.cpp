/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
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
#include "siamger_infer.hpp"

#include <iostream>
#include <memory>
#include <string>

#include "device/mlu_context.h"
#include "siamger_handler.hpp"
#include "siamger_params.hpp"
#include "postproc.hpp"

namespace cnstream {

Siginfer::Siginfer(const std::string& name) : Module(name) {
  hasTransmit_.store(true);
  param_register_.SetModuleDesc("Siginfer is a module for running offline model inference, preprocessing and "
                                "postprocessing based on infer_server.");
  param_manager_ = std::make_shared<SiamgerParamManager>();
  LOGF_IF(SIGINFER_INFER, !param_manager_) << "Inferencer::Inferencer(const std::string& name) new InferParams failed.";
  param_manager_->RegisterAll(&param_register_);
}

bool Siginfer::Open(ModuleParamSet raw_params) {
  SiamgerParam params;

  if (!param_manager_->ParseBy(raw_params, &params)) {
    LOGE(SIGINFER_INFER) << "[" << GetName() << "] parse parameters failed.";
    return false;
  }

  infer_params_ = params;

  // fix paths
  if (!infer_params_.model_path.empty())
    infer_params_.model_path = GetPathRelativeToTheJSONFile(infer_params_.model_path, raw_params);

  // check preprocess
  if (!infer_server::SetCurrentDevice(params.device_id)) return false;

  if (params.preproc_name.empty()) {
    LOGE(SIGINFER_INFER) << "Preproc name can't be empty string. Please set preproc_name.";
    return false;
  }
  std::shared_ptr<SiamgerPreproc> pre_processor = nullptr;

  if (params.preproc_name == "VideoPreprocSiamger") {

    pre_processor.reset(SiamgerPreproc::Create(params.preproc_name));
    if (!pre_processor) {
      LOGE(SIGINFER_INFER) << "Can not find SiamgerPreproc implemention by name: " << params.preproc_name;
      return false;
    }
    if (!pre_processor->Init(params.custom_preproc_params)) {
      LOGE(SIGINFER_INFER) << "SiamgerPreprocessor init failed.";
      return false;
    }
    pre_processor->SetModelInputPixelFormat(params.model_input_pixel_format);
    pre_processor->SetSizeCenter(params.init_loc_);
  }

  std::shared_ptr<SiamgerPostproc> post_processor = nullptr;
  if (params.postproc_name.empty()) {
    LOGE(SIGINFER_INFER) << "Postproc name can't be empty string. Please set postproc_name.";
    return false;
  }
  post_processor.reset(SiamgerPostproc::Create(params.postproc_name));
  if (!post_processor) {
    LOGE(SIGINFER_INFER) << "Can not find SiagmerPostproc implemention by name: " << params.postproc_name;
    return false;
  }
  if (!post_processor->Init(params.custom_postproc_params)) {
    LOGE(SIGINFER_INFER) << "Postprocessor init failed.";
    return false;
  }
  post_processor->SetSizeCenter(params.init_loc_);
  // stride_ , SCORE_SIZR_ = (INSTANCE_SIZE - EXEMPLAR_SIZE) //  STRIDE + 1 + BASE_SIZE
  post_processor->generate_points(16, 16);
  // SCORE_SIZR_
  post_processor->hamming(16);
  
  std::shared_ptr<FrameFilter> frame_filter = nullptr;
  if (!params.frame_filter_name.empty()) {
    frame_filter.reset(FrameFilter::Create(params.frame_filter_name));
    if (!frame_filter) {
      LOGE(SIGINFER_INFER) << "Can not find FrameFilter implemention by name: " << params.frame_filter_name;
      return false;
    }
  }
  std::shared_ptr<ObjFilter> obj_filter = nullptr;
  if (!params.obj_filter_name.empty()) {
    obj_filter.reset(ObjFilter::Create(params.obj_filter_name));
    if (!obj_filter) {
      LOGE(SIGINFER_INFER) << "Can not find ObjFilter implemention by name: " << params.obj_filter_name;
      return false;
    }
  }

  infer_handler_ =
      std::make_shared<SiamgerhandlerImpl>(this, infer_params_, post_processor, pre_processor, frame_filter, obj_filter);
  return infer_handler_->Open();
}

void Siginfer::Close() { infer_handler_.reset(); }

int Siginfer::Process(std::shared_ptr<CNFrameInfo> data) {
  if (!data) {
    LOGE(SIAMGER) << "Process inputdata is nulltpr!";
    return -1;
  }

  if (!data->IsEos()) {
    CNDataFramePtr frame = data->collection.Get<CNDataFramePtr>(kCNDataFrameTag);
    if (frame->dst_device_id < 0) {
      /* CNSyncedMemory data is on CPU */
      for (int i = 0; i < frame->GetPlanes(); i++) {
        frame->data[i]->SetMluDevContext(infer_params_.device_id);
      }
      frame->dst_device_id = infer_params_.device_id;
    } else if (static_cast<uint32_t>(frame->dst_device_id) != infer_params_.device_id &&
               frame->ctx.dev_type == DevContext::DevType::MLU) {
      /* CNSyncedMemory data is on different MLU from the data this module needed, and SOURCE data is on MLU*/
      frame->CopyToSyncMemOnDevice(infer_params_.device_id);
      frame->dst_device_id = infer_params_.device_id;
    } else if (static_cast<uint32_t>(frame->dst_device_id) != infer_params_.device_id &&
               frame->ctx.dev_type == DevContext::DevType::CPU) {
      /* CNSyncedMemory data is on different MLU from the data this module needed, and SOURCE data is on CPU*/
      void *dst = frame->cpu_data.get();
      for (int i = 0; i < frame->GetPlanes(); i++) {
        size_t plane_size = frame->GetPlaneBytes(i);
        frame->data[i].reset(new CNSyncedMemory(plane_size));
        frame->data[i]->SetCpuData(dst);
        dst = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(dst) + plane_size);
        frame->data[i]->SetMluDevContext(infer_params_.device_id);
      }
      frame->dst_device_id = infer_params_.device_id;  // set dst_device_id to param_.device_id
    }
    if (infer_handler_->Process(data, infer_params_.object_infer) != 0) {
      return -1;
    }
  } else {
    infer_handler_->WaitTaskDone(data->stream_id);
    TransmitData(data);
  }

  return 0;
}

Siginfer::~Siginfer() {
  Close();
}

bool Siginfer::CheckParamSet(const ModuleParamSet& param_set) const {
  SiamgerParam params;
  return param_manager_->ParseBy(param_set, &params);
}

}  // namespace cnstream
