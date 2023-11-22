/*************************************************************************
 * Copyright (C) [2021] by xidian114, Inc. All rights reserved
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
 * THE AUTHORS OR COPYRIGHT HOLDERS BELIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#if (CV_MAJOR_VERSION >= 3)
#include "opencv2/imgcodecs/imgcodecs.hpp"
#endif

#include "cnis/contrib/video_helper.h"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"
#include "siamger_preproc.hpp"
#include "video_preprocess_common.hpp"


std::vector<float> update_size{0,0};  
std::vector<float> update_center{0,0};
cv::Mat in_Mat1;

/**
 * @brief Video preprocessing for Siamger neural network
 */
class VideoPreprocSiamger : public cnstream::SiamgerPreproc {
 public:
  /**
   * @brief Execute Siamger neural network preprocessing
   *
   * @param model_input: the input of neural network. The preproc result should be set to it.
   * @param input_data: the raw input data. The user could get infer_server::video::VideoFrame object from it.
   * @param model_info: model information, e.g., input/output number, shape and etc.
   *
   * @return return true if succeed
   */
  bool Execute(infer_server::ModelIO* model_input, const infer_server::InferData& input_data,
               const infer_server::ModelInfo* model_info) {

    if (update_size[0] && update_size[1])
    {
      size_ = update_size;
      center_pos_ = update_center;
    }

    // check model input number and shape
    uint32_t input_num = model_info->InputNum();
    if (input_num != 2) {
      LOGE(VideoPreprocSiamger) << "[Siamger] model input number not supported. It should be 2, but " << input_num;
      return false;
    }

    infer_server::Shape input0_shape;
    infer_server::Shape input1_shape;
    input0_shape = model_info->InputShape(0);
    input1_shape = model_info->InputShape(1);

    int c_idx = 3;
    int w_idx = 2;
    int h_idx = 1;
    if (model_info->InputLayout(0).order == infer_server::DimOrder::NCHW) {
      c_idx = 1;
      w_idx = 3;
      h_idx = 2;
    }

    if (input0_shape[c_idx] != 3) {
      LOGE(VideoPreprocSiamger) << "[Siamger] model input0 shape not supported, `c` should be 3, but " << input0_shape[c_idx];
      return false;
    }


    if (input1_shape[c_idx] != 3) {
      LOGE(VideoPreprocSiamger) << "[Siamger] model input1 shape not supported, `c` should be 3, but " << input1_shape[c_idx];
      return false;
    }
    // do preproc
    const infer_server::video::VideoFrame& frame = input_data.GetLref<infer_server::video::VideoFrame>();

    size_t src_w = frame.width; // 1440
    size_t src_h = frame.height; // 1080
    size_t src_stride = frame.stride[0];

    uint32_t dst0_w = input0_shape[w_idx];
    uint32_t dst0_h = input0_shape[h_idx];

    uint32_t dst1_w = input1_shape[w_idx];
    uint32_t dst1_h = input1_shape[h_idx];

    uint8_t* img_data = new (std::nothrow) uint8_t[frame.GetTotalSize()];

    if (!img_data) {
      LOGE(VideoPreprocSiamger) << "[Siamger] Failed to alloc memory, size: " << frame.GetTotalSize();
      return false;
    }

    uint8_t* img_data_tmp = img_data;
    for (auto plane_idx = 0u; plane_idx < frame.plane_num; ++plane_idx) {
      memcpy(img_data_tmp, frame.plane[plane_idx].Data(), frame.GetPlaneSize(plane_idx));
      img_data_tmp += frame.GetPlaneSize(plane_idx);
    }

    // convert color space from src to dst

    cv::Mat dst_cvt_color_img;
    if (!ConvertColorSpace(src_w, src_h, src_stride, frame.format, model_input_pixel_format_, img_data,
                           &dst_cvt_color_img)) {
      LOGW(VideoPreprocSiamger) << "[Siamger] Unsupport pixel format. src: " << static_cast<int>(frame.format)
                 << " dst: " << static_cast<int>(model_input_pixel_format_);
      delete[] img_data;
      return false;
    }

    if (in_Mat1.empty()) 
    {
      in_Mat1 = tracker_preinit(dst_cvt_color_img);  
    }
    
    cv::Mat in_Mat0;
    in_Mat0 = tracker_pretrack(dst_cvt_color_img); 
    
    cv::Mat dst0(dst0_h, dst0_w, CV_32FC3, model_input->buffers[0].MutableData());
    in_Mat0.convertTo(dst0, CV_32FC3);
    dst0 /= 255.0;
 
    cv::Mat dst1(dst1_h, dst1_w, CV_32FC3, model_input->buffers[1].MutableData());
    in_Mat1.convertTo(dst1, CV_32FC3);
    dst1 /= 255.0;

    delete[] img_data;
    return true;
  }


  

 private:
  DECLARE_REFLEX_OBJECT_EX(VideoPreprocSiamger, cnstream::SiamgerPreproc);
  std::vector<float> channel_average_{0.0f, 0.0f, 0.0f};
  float CONTEXT_AMOUNT_ = 0.5;
  uint32_t EXEMPLAR_SIZE_ = 127;
  uint32_t INSTANCE_SIZE_ = 255;
  // get template Mat
  cv::Mat get_subwindow(cv:: Mat im,  
                     uint32_t model_sz,
                     uint32_t original_sz,
                     std::vector<float> avg_chans) 
  {
    int sz = original_sz;
    std::vector<int> im_sz = {im.rows, im.cols, 3};
    float c = (static_cast<float>(original_sz) + 1) / 2;
    
    int context_xmin = floor(center_pos_[0] - c + 0.5);
    int context_xmax = context_xmin + sz - 1;

    int context_ymin = floor(center_pos_[1] - c + 0.5);
    int context_ymax = context_ymin + sz - 1;



    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - im_sz[1] + 1);
    int bottom_pad = std::max(0, context_ymax - im_sz[0] + 1);

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

    int im_r = im.rows;
    int im_c = im.cols;
    // int im_k = im.channels();
    cv::Mat im_patch;

    if (top_pad || right_pad || left_pad || bottom_pad)
    {

      cv::Mat te_im(im_r + top_pad + bottom_pad, im_c + left_pad + right_pad, 
                        CV_8UC3, cv::Scalar(0, 0, 0));

      for(int i = 0;i < im_r;++i)
      {
        for (int j = 0;j < im_c;++j)
        {
            te_im.at<cv::Vec3b>(i + top_pad, j + left_pad) = im.at<cv::Vec3b>(i, j) ;
        }
      }

      if (top_pad)
      {
        for (int i = 0; i < top_pad; i++) //矩阵行数循环
        {
          for (int j = left_pad; j < left_pad + im_c; j++) //矩阵列数循环
          {
            for (int cc = 0;cc < 3;++cc)
            {
              te_im.at<cv::Vec3b>(i,j)[cc] = int(avg_chans[cc]); 
            }
          }
        }
      }
      if (bottom_pad)
      {
        for (int i = im_r + top_pad; i < te_im.rows; i++) //矩阵行数循环
        {
          for (int j = left_pad; j < left_pad + im_c; j++) //矩阵列数循环
          {
            for (int cc = 0;cc < 3;++cc)
            {
              te_im.at<cv::Vec3b>(i,j)[cc] = int(avg_chans[cc]); 
            }
          }
        }
      }
      if (left_pad)
      {
        for (int i = 0; i < te_im.rows; i++) //矩阵行数循环
        {
          for (int j = 0; j < left_pad; j++) //矩阵列数循环
          {
            for (int cc = 0;cc < 3;++cc)
            {
              te_im.at<cv::Vec3b>(i,j)[cc] = int(avg_chans[cc]); 
            }
          }
        }
      }
      if (right_pad)
      {
        for (int i = 0; i < te_im.rows; i++) //矩阵行数循环
        {
          for (int j = im_c + left_pad; j < te_im.cols; j++) //矩阵列数循环
          {
            for (int cc = 0;cc < 3;++cc)
            {
              te_im.at<cv::Vec3b>(i,j)[cc] = int(avg_chans[cc]); 
            }
          }
        }
      }
      im_patch = te_im(cv::Rect(context_xmin , context_ymin,  
       context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else 
    {
      im_patch = im(cv::Rect(context_xmin , context_ymin, 
       context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }

    if (model_sz != original_sz)
    {
      cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
    }
    
    return im_patch;
  }

  cv::Mat tracker_preinit(cv::Mat dst_cvt_color_img)
  {
    // calculate channel average
    cv::Scalar meanValue=cv::mean(dst_cvt_color_img);
    for (int c = 0;c < 3;++c)
    {
      channel_average_[c] = meanValue.val[c];
    }

    // calculate z crop size
    float w_z = size_[0] + CONTEXT_AMOUNT_ * (size_[0] + size_[1]);
    float h_z = size_[1] + CONTEXT_AMOUNT_ * (size_[0] + size_[1]);
    uint32_t s_z = static_cast<uint32_t>(round(sqrt(w_z * h_z)));
    // get_subwindow

    cv::Mat res;
    res = get_subwindow(dst_cvt_color_img, static_cast<float>(EXEMPLAR_SIZE_), s_z, channel_average_);

    // uptate template 
    return res;
  }

  cv::Mat tracker_pretrack (cv::Mat img)
  {
    float w_z = size_[0] + CONTEXT_AMOUNT_ * (size_[0] + size_[1]);
    float h_z = size_[1] + CONTEXT_AMOUNT_ * (size_[0] + size_[1]);
    float s_z = sqrt(w_z * h_z);
    uint32_t s_x = static_cast<uint32_t>(s_z * (static_cast<float>(INSTANCE_SIZE_) / EXEMPLAR_SIZE_));
    cv::Mat res;
    res = get_subwindow(img, INSTANCE_SIZE_, s_x, channel_average_);
    return res;
  }
};  // class VideoPreprocSiamger

IMPLEMENT_REFLEX_OBJECT_EX(VideoPreprocSiamger, cnstream::SiamgerPreproc);
