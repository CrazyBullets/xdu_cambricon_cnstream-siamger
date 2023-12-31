/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
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

#include "detection_runner.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cxxutil/log.h"
#include "cnis/contrib/video_helper.h"
#include "cnis/infer_server.h"
#include "cnis/processor.h"

#include "postprocess/postproc.h"
#include "preprocess/preproc.h"

#if CV_VERSION_EPOCH == 2
#define OPENCV_MAJOR_VERSION 2
#elif CV_VERSION_MAJOR >= 3
#define OPENCV_MAJOR_VERSION CV_VERSION_MAJOR
#endif

static const cv::Size g_out_video_size = cv::Size(1280, 720);

DetectionRunner::DetectionRunner(const VideoDecoder::DecoderType& decode_type, int device_id,
                                 const std::string& model_path, const std::string& func_name,
                                 const std::string& label_path, const std::string& track_model_path,
                                 const std::string& track_func_name, const std::string& data_path,
                                 const std::string& net_type, bool show, bool save_video)
    : StreamRunner(data_path, decode_type, device_id), show_(show), save_video_(save_video), net_type_(net_type) {
  // set mlu environment
  env_.SetDeviceId(device_id);
  env_.BindDevice();

  infer_server_.reset(new infer_server::InferServer(device_id));
  infer_server::SessionDesc desc;
  desc.strategy = infer_server::BatchStrategy::STATIC;
  desc.engine_num = 1;
  desc.priority = 0;
  desc.show_perf = true;
  desc.name = "detection session";

  // load offline model
#ifdef CNIS_USE_MAGICMIND
  desc.model = infer_server::InferServer::LoadModel(model_path);
#else
  desc.model = infer_server::InferServer::LoadModel(model_path, func_name);
#endif
  // set preproc and postproc
  desc.preproc = infer_server::video::PreprocessorMLU::Create();
  desc.postproc = infer_server::Postprocessor::Create();

#ifdef CNIS_USE_MAGICMIND
  if (net_type == "YOLOv3") {
    desc.preproc->SetParams("preprocess_type", infer_server::video::PreprocessType::CNCV_PREPROC,
                            "src_format", infer_server::video::PixelFmt::NV12,
                            "dst_format", infer_server::video::PixelFmt::RGB24,
                            "keep_aspect_ratio", true);
    desc.postproc->SetParams("process_function", infer_server::Postprocessor::ProcessFunction(PostprocYolov3MM(0.5)));
  } else {
    THROW_EXCEPTION(edk::Exception::INVALID_ARG, "unsupported magicmind net type: " + net_type);
  }
#else
  if (net_type == "SSD") {
    // custom preprocessor example
    desc.preproc = infer_server::Preprocessor::Create();
    desc.preproc->SetParams("process_function", infer_server::Preprocessor::ProcessFunction(
        PreprocSSD(desc.model, device_id, edk::PixelFmt::BGRA)));
    desc.postproc->SetParams("process_function", infer_server::Postprocessor::ProcessFunction(PostprocSSD(0.5)));
  } else if (net_type == "YOLOv3") {
    desc.preproc->SetParams("preprocess_type", infer_server::video::PreprocessType::CNCV_PREPROC,
                            "src_format", infer_server::video::PixelFmt::NV12,
                            "dst_format", infer_server::video::PixelFmt::ARGB,
                            "keep_aspect_ratio", true);
    desc.postproc->SetParams("process_function", infer_server::Postprocessor::ProcessFunction(PostprocYolov3(0.5)));
  } else if (net_type == "YOLOv5") {
    desc.preproc->SetParams("preprocess_type", infer_server::video::PreprocessType::RESIZE_CONVERT,
                            "src_format", infer_server::video::PixelFmt::NV12,
                            "dst_format", infer_server::video::PixelFmt::RGB24,
                            "normalize", true,
                            "keep_aspect_ratio", true);
    desc.postproc->SetParams("process_function", infer_server::Postprocessor::ProcessFunction(PostprocYolov5(0.5)));
  } else {
    THROW_EXCEPTION(edk::Exception::INVALID_ARG, "unsupported net type: " + net_type);
  }
#endif
  session_ = infer_server_->CreateSyncSession(desc);

  // init tracker
  tracker_.reset(new edk::FeatureMatchTrack);
  feature_extractor_.reset(new FeatureExtractor);
  if (track_model_path != "" && track_model_path != "cpu") {
    feature_extractor_->Init(track_model_path.c_str(), track_func_name.c_str(), device_id);
  }

  // init osd
  osd_.LoadLabels(label_path);

  // display or video writer
  if (save_video_) {
#if OPENCV_MAJOR_VERSION > 2
    video_writer_.reset(
        new cv::VideoWriter("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, g_out_video_size));
#else
    video_writer_.reset(new cv::VideoWriter("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, g_out_video_size));
#endif
    if (!video_writer_->isOpened()) {
      THROW_EXCEPTION(edk::Exception::Code::INIT_FAILED, "create output video file failed");
    }
  }

  Start();
}

DetectionRunner::~DetectionRunner() {
  Stop();
  if (infer_server_ && session_) {
    infer_server_->DestroySession(session_);
  }
}

cv::Mat DetectionRunner::ConvertToMatAndReleaseBuf(edk::CnFrame* frame) {
  // alloc memory to store image
  auto img_data = new uint8_t[frame->strides[0] * frame->height * 3 / 2];
  // copy out frame
  decoder_->CopyFrameD2H(img_data, *frame);
  uint32_t frame_h = frame->height;
  uint32_t frame_stride = frame->strides[0];
  // release codec buffer
  decoder_->ReleaseFrame(std::move(*frame));

  // yuv to bgr
  cv::Mat yuv(frame_h * 3 / 2, frame_stride, CV_8UC1, img_data);
  cv::Mat img;
  if (frame->pformat == edk::PixelFmt::NV12) {
    cv::cvtColor(yuv, img, cv::COLOR_YUV2BGR_NV12);
  } else if (frame->pformat == edk::PixelFmt::NV21) {
    cv::cvtColor(yuv, img, cv::COLOR_YUV2BGR_NV12);
  } else {
    LOGE(SAMPLE) << "unsupported pixel format";
  }
  delete[] img_data;

  // resize to show
  cv::resize(img, img, cv::Size(1280, 720));
  return img;
}

void DetectionRunner::Process(edk::CnFrame frame) {
  // prepare input
  infer_server::PackagePtr in = infer_server::Package::Create(1);
  if (net_type_ == "YOLOv3" || net_type_ == "YOLOv5") {
    infer_server::video::VideoFrame vframe;
    vframe.plane_num = frame.n_planes;
    vframe.format = infer_server::video::PixelFmt::NV12;
    vframe.width = frame.width;
    vframe.height = frame.height;

    for (size_t plane_idx = 0; plane_idx < vframe.plane_num; ++plane_idx) {
      vframe.stride[plane_idx] = frame.strides[plane_idx];
      uint32_t plane_bytes = vframe.height * vframe.stride[plane_idx];
      if (plane_idx == 1) plane_bytes = std::ceil(1.0 * plane_bytes / 2);
      infer_server::Buffer mlu_buffer(frame.ptrs[plane_idx], plane_bytes, nullptr, GetDeviceId());
      vframe.plane[plane_idx] = mlu_buffer;
    }
    in->data[0]->Set(std::move(vframe));
    in->data[0]->SetUserData(FrameSize{static_cast<int>(vframe.width), static_cast<int>(vframe.height)});
  } else if (net_type_ == "SSD") {
    in->data[0]->Set(frame);
  } else {
    THROW_EXCEPTION(edk::Exception::INVALID_ARG, "unsupported net type: " + net_type_);
  }

  infer_server::PackagePtr out = infer_server::Package::Create(1);
  infer_server::Status status = infer_server::Status::SUCCESS;
  bool ret = infer_server_->RequestSync(session_, std::move(in), &status, out);
  if (!ret || status != infer_server::Status::SUCCESS) {
    decoder_->ReleaseFrame(std::move(frame));
    THROW_EXCEPTION(edk::Exception::INTERNAL,
        "Request sending data to infer server failed. Status: " + std::to_string(static_cast<int>(status)));
  }
  const std::vector<DetectObject>& postproc_results = out->data[0]->GetLref<std::vector<DetectObject>>();
  std::vector<edk::DetectObject> detect_objs;
  detect_objs.reserve(postproc_results.size());
  for (auto &obj : postproc_results) {
    edk::DetectObject detect_obj;
    detect_obj.label = obj.label;
    detect_obj.score = obj.score;
    detect_obj.bbox.x = obj.bbox.x;
    detect_obj.bbox.y = obj.bbox.y;
    detect_obj.bbox.width = obj.bbox.w;
    detect_obj.bbox.height = obj.bbox.h;
    detect_objs.emplace_back(std::move(detect_obj));
  }

  cv::Mat img;
  if (feature_extractor_->OnMlu()) {
    // extract feature on mlu
    if (!feature_extractor_->ExtractFeatureOnMlu(frame, &detect_objs)) {
      THROW_EXCEPTION(edk::Exception::INTERNAL, "Extract feature on Mlu failed");
    }
    img = ConvertToMatAndReleaseBuf(&frame);
  } else {
    // extract feature on cpu
    img = ConvertToMatAndReleaseBuf(&frame);
    if (!feature_extractor_->ExtractFeatureOnCpu(img, &detect_objs)) {
      THROW_EXCEPTION(edk::Exception::INTERNAL, "Extract feature on Cpu failed");
    }
  }

  // feature match
  std::vector<edk::DetectObject> track_result;
  track_result.clear();
  tracker_->UpdateFrame(edk::TrackFrame(), detect_objs, &track_result);

  std::cout << "----- Object detected in one frame:\n";
  for (auto& obj : track_result) {
    std::cout << obj << "\n";
  }
  std::cout << "-----------------------------------\n" << std::endl;

  osd_.DrawLabel(img, track_result);

  if (show_) {
    auto window_name = "stream app";
    cv::imshow(window_name, img);
    cv::waitKey(5);
    // std::string fn = std::to_string(frame.frame_id) + ".jpg";
    // cv::imwrite(fn.c_str(), img);
  }
  if (save_video_) {
    video_writer_->write(img);
  }
}
