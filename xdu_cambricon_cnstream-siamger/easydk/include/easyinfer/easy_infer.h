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
 * @file easy_infer.h
 *
 * This file contains a declaration of the EasyInfer class.
 */

#ifndef EASYINFER_EASY_INFER_H_
#define EASYINFER_EASY_INFER_H_

#include <memory>
#include "cxxutil/edk_attribute.h"
#include "cxxutil/exception.h"
#include "easyinfer/model_loader.h"

namespace edk {

struct MluTaskQueue;
using MluTaskQueue_t = std::shared_ptr<MluTaskQueue>;

class EasyInferPrivate;

/**
 * @brief Inference helper class
 */
class EasyInfer {
 public:
  /**
   * @brief Construct a new Easy Infer object
   */
  EasyInfer();

  /**
   * @brief Destroy the Easy Infer object
   */
  ~EasyInfer();

  /**
   * @brief Initialize the inference helper class
   *
   * @param model Model loader which contain neural network offline model and informations
   * @param dev_id init EasyInfer in device with origin id dev_id. only supported on MLU270
   */
  void Init(std::shared_ptr<ModelLoader> model, int dev_id);

  /**
   * @brief Invoke inference function
   *
   * @param input Input data in MLU
   * @param output Output data in MLU
   * @param hw_time Hardware time of inference
   */
  void Run(void** input, void** output, float* hw_time = nullptr) const;

  /**
   * @brief  Async invoke inference function
   *
   * @param input Input data in MLU
   * @param output Output data in MLU
   * @param task_queue 
   */
  void RunAsync(void** input, void** output, MluTaskQueue_t task_queue) const;

  /**
   * @brief Get the model loader
   *
   * @see edk::ModelLoader
   * @return Model loader
   */
  std::shared_ptr<ModelLoader> Model() const;

  /**
   * @brief Get the MLU task queue, used to share MLU queue with Bang kernel
   *
   * @return MluTaskQueue
   */
  MluTaskQueue_t GetMluQueue() const;

 private:
  EasyInferPrivate* d_ptr_;

  EasyInfer(const EasyInfer&) = delete;
  EasyInfer& operator=(const EasyInfer&) = delete;
};  // class EasyInfer

}  // namespace edk

#endif  // CNINFER_H_
