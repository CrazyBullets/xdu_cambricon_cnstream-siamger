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

#ifndef MODULES_SIAMGER_INCLUDE_POSTPROC_HPP_
#define MODULES_SIAMGER_INCLUDE_POSTPROC_HPP_

/**
 *  \file siamger_postproc.hpp
 *
 *  This file contains a declaration of class SiamgerPostproc
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "reflex_object.h"



namespace cnstream {
/*!
 * @class SiamgerPostproc
 *
 * @brief SiamgerPostproc is the base class of post processing classes for Inference2.
 */
class SiamgerPostproc : virtual public ReflexObjectEx<SiamgerPostproc> {
 public:
  /**
   * @brief Destructs an object.
   *
   * @return No return value.
   */
  virtual ~SiamgerPostproc() = 0;
  /**
   * @brief Creates a postprocess object with the given postprocess's class name.
   *
   * @param[in] proc_name The postprocess class name.
   *
   * @return Returns the pointer to postprocess object.
   */
  static SiamgerPostproc* Create(const std::string& proc_name);
  /**
   * @brief Initializes postprocessing parameters.
   *
   * @param[in] params The postprocessing parameters.
   *
   * @return Returns ture for success, otherwise returns false.
   **/
  virtual bool Init(const std::unordered_map<std::string, std::string> &params) { return true; }
  /**
   * @brief Sets threshold.
   *
   * @param[in] threshold The value between 0 and 1.
   *
   * @return No return value.
   */

  void SetThreshold(const float threshold);
  
  void SetSizeCenter(const std::vector<uint32_t> init_loc);

  void generate_points(int stride, int size);

  //hamming : a = 0.54;   hann : a = 0.5
  void hamming(int len);
  

  /**
   * @brief Executes postprocessing on the model's output data.
   *
   * @param[out] output_data The postprocessing result. The result of postprocessing should be set to it.
   *                     You could set any type of data to this parameter and get it in UserProcess function.
   * @param[in] model_output The neural network origin output data.
   * @param[in] model_info The model information, such as input/output number and shape.
   *
   * @return Returns true if successful, otherwise returns false.
   *
   * @note This function is executed by infer server postproc processor. You could override it to develop custom
   *       postprocessing.
   *       To set any type of data to output_data, use this statement,
   *       e.g., `int example_var = 1; output_data->Set(example_var);`
   *
   */
  virtual bool Execute(infer_server::InferData* output_data, const infer_server::ModelIO& model_output,
                       const infer_server::ModelInfo* model_info) = 0;

 protected:
  float threshold_ = 0;
  std::vector<std::vector<int>> points_;
  std::vector<float> window_ = std::vector<float>(256,0);
  std::vector<float> size_{0,0};
  std::vector<float> center_pos_{0,0};

};  // class SiamgerPostproc
}  // namespace cnstream

#endif  // MODULES_SIAMGER_INCLUDE_POSTPROC_HPP_
