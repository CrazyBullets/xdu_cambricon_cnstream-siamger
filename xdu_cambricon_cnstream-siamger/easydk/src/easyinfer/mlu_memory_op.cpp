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

#include "easyinfer/mlu_memory_op.h"

#include <cstring>
#include <memory>
#include <string>

#include "cxxutil/log.h"
#include "easyinfer/model_loader.h"
#include "model_loader_internal.h"

namespace edk {

#define CHECK_MODEL_LOADER                                          \
  if (!model_) {                                                  \
    THROW_EXCEPTION(Exception::UNAVAILABLE, "ModelLoader Not Set"); \
  }

#define CHECK_CNRT_RET(err_code, str)                                                                           \
  if (CNRT_RET_SUCCESS != err_code) {                                                                           \
    THROW_EXCEPTION(Exception::MEMORY, std::string(str) + " cnrt error code: " + std::to_string(error_code)); \
  }

#define ONLY_SUPPORT_FLOAT32_ON_CPU                        \
  do {                                                     \
    int num = model_->InputNum();                        \
    for (int i = 0; i < num; ++i) {                        \
      DataLayout layout = model_->GetCpuInputLayout(i);  \
      if (layout.dtype != DataType::FLOAT32) {             \
        THROW_EXCEPTION(Exception::INVALID_ARG,            \
                        "Only support cpu"                 \
                        " layout with data type FLOAT32"); \
      }                                                    \
    }                                                      \
    num = model_->OutputNum();                           \
    for (int i = 0; i < num; ++i) {                        \
      DataLayout layout = model_->GetCpuOutputLayout(i); \
      if (layout.dtype != DataType::FLOAT32) {             \
        THROW_EXCEPTION(Exception::INVALID_ARG,            \
                        "Only support cpu"                 \
                        " layout with data type FLOAT32"); \
      }                                                    \
    }                                                      \
  } while (0)

extern cnrtDataType CastDataType(const DataType &type);
extern DataType CastDataType(const cnrtDataType &type);

static size_t TypeSize(const DataType &type) {
  switch (type) {
    case DataType::UINT8:
      return sizeof(uint8_t);
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::FLOAT16:
      return sizeof(int16_t);
    case DataType::INT16:
      return sizeof(int16_t);
    case DataType::INT32:
      return sizeof(int32_t);
    default:
      THROW_EXCEPTION(Exception::UNSUPPORTED, "Unsupported data type");
  }
}

static void TransLayout(const DataLayout &src_layout, const DataLayout &dst_layout, void *src_data, void *dst_data,
                        const ShapeEx &shape) {
  if (src_layout.order != DimOrder::NHWC && src_layout.order != DimOrder::NCHW) {
    THROW_EXCEPTION(Exception::INVALID_ARG, "TransLayout: Unsupport data order(src).");
  }
  if (dst_layout.order != DimOrder::NHWC && dst_layout.order != DimOrder::NCHW) {
    THROW_EXCEPTION(Exception::INVALID_ARG, "TransLayout: Unsupport data order(dst).");
  }

  char bits = 0;
  if (src_layout.dtype != dst_layout.dtype) bits |= 1 << 0;
  if (src_layout.order != dst_layout.order) bits |= 1 << 1;
  cnrtRet_t error_code = CNRT_RET_SUCCESS;
  int size = shape.BatchDataCount();
  int dim_values[4] = {shape.N(), shape.H(), shape.W(), shape.C()};
  int dim_order[4];
  if (dst_layout.order == DimOrder::NHWC) {
    dim_order[0] = 0, dim_order[1] = 2, dim_order[2] = 3, dim_order[3] = 1;
  } else if (dst_layout.order == DimOrder::NCHW) {
    dim_order[0] = 0, dim_order[1] = 3, dim_order[2] = 1, dim_order[3] = 2;
  } else {
    THROW_EXCEPTION(Exception::INVALID_ARG, "TransLayout: Unsupport data order(dst).");
  }
  switch (bits) {
    case 1 << 0:
      error_code = cnrtCastDataType(src_data, CastDataType(src_layout.dtype), dst_data, CastDataType(dst_layout.dtype),
                                    size, nullptr);
      CHECK_CNRT_RET(error_code, "Cast data type failed.");
      break;
    case 1 << 1:
      error_code = cnrtTransDataOrder(src_data, CastDataType(src_layout.dtype), dst_data, 4, dim_values, dim_order);
      CHECK_CNRT_RET(error_code, "Trans data order failed.");
      break;
    case 1 << 0 | 1 << 1:
      error_code = cnrtTransOrderAndCast(src_data, CastDataType(src_layout.dtype), dst_data,
                                         CastDataType(dst_layout.dtype), nullptr, 4, dim_values, dim_order);
      CHECK_CNRT_RET(error_code, "Trans data order and cast data type failed.");
      break;
    default:
      size_t mem_size = size * TypeSize(src_layout.dtype);
      memcpy(dst_data, src_data, mem_size);
      break;
  }
}

MluMemoryOp::MluMemoryOp() : model_(nullptr) {}

void MluMemoryOp::SetModel(std::shared_ptr<ModelLoader> model) { model_ = model; }

std::shared_ptr<ModelLoader> MluMemoryOp::Model() const { return model_; }

void **MluMemoryOp::AllocCpuInput() const {
  CHECK_MODEL_LOADER;
  ONLY_SUPPORT_FLOAT32_ON_CPU;
  uint32_t num = model_->InputNum();

  LOGT(MEMORY) << "Alloc memory on CPU for model input";

  void **ret = new void *[num];
  for (uint32_t i = 0; i < num; ++i) {
    auto &shape = model_->InputShape(i);
    uint64_t data_size = shape.BatchDataCount();
    LOGT(MEMORY) << "Alloc CPU input memory (" << i << ") on CPU in " << data_size * sizeof(float) << " bytes";
    ret[i] = reinterpret_cast<void *>(new float[data_size]);
  }
  return ret;
}

void **MluMemoryOp::AllocCpuOutput() const {
  CHECK_MODEL_LOADER;
  ONLY_SUPPORT_FLOAT32_ON_CPU;
  uint32_t num = model_->OutputNum();

  LOGT(MEMORY) << "Alloc memory on CPU for model output";

  void **ret = new void *[num];
  for (uint32_t i = 0; i < num; ++i) {
    auto &shape = model_->OutputShape(i);
    uint64_t data_size = shape.BatchDataCount();
    LOGT(MEMORY) << "Alloc output memory (" << i << ") on CPU in " << data_size * sizeof(float) << " bytes";
    ret[i] = reinterpret_cast<void *>(new float[data_size]);
  }
  return ret;
}

void **MluMemoryOp::AllocMluInput() const {
  CHECK_MODEL_LOADER;
  void **ret = nullptr;
  cnrtRet_t error_code;
  uint32_t num = model_->InputNum();
  ModelLoaderInternalInterface interface(model_.get());

  LOGT(MEMORY) << "Alloc memory on MLU for model input";

  ret = new void *[num];
  for (uint32_t i = 0; i < num; ++i) {
    void *t = nullptr;
    int64_t size = interface.InputDataSize(i);
    LOGT(MEMORY) << "Alloc input memory (" << i << ") on MLU in " << size << " bytes";
    error_code = cnrtMalloc(&t, size);
    CHECK_CNRT_RET(error_code, "Mlu malloc failed.");
    ret[i] = t;
  }
  return ret;
}

void **MluMemoryOp::AllocMluOutput() const {
  CHECK_MODEL_LOADER;
  void **ret = nullptr;
  cnrtRet_t error_code;
  uint32_t num = model_->OutputNum();
  ModelLoaderInternalInterface interface(model_.get());

  LOGT(MEMORY) << "Alloc memory on MLU for model output";

  ret = new void *[num];
  for (uint32_t i = 0; i < num; ++i) {
    void *t = nullptr;
    int64_t size = interface.OutputDataSize(i);
    LOGT(MEMORY) << "Alloc output memory (" << i << ") on MLU in " << size << " bytes";
    error_code = cnrtMalloc(&t, size);
    CHECK_CNRT_RET(error_code, "Mlu malloc failed.");
    ret[i] = t;
  }
  return ret;
}

void *MluMemoryOp::AllocMlu(size_t nBytes) {
  void *ret = nullptr;
  cnrtRet_t error_code;
  LOGT(MEMORY) << "Alloc memory on MLU in " << nBytes << " bytes";
  error_code = cnrtMalloc(&ret, nBytes);
  CHECK_CNRT_RET(error_code, "Mlu malloc failed.");
  return ret;
}

void MluMemoryOp::FreeCpuInput(void **ptr) const {
  CHECK_MODEL_LOADER;
  LOGT(MEMORY) << "Free input memory on CPU";
  uint32_t num = model_->InputNum();
  for (uint32_t i = 0; i < num; ++i) {
    delete[] reinterpret_cast<float *>(ptr[i]);
  }
  delete[] ptr;
}

void MluMemoryOp::FreeCpuOutput(void **ptr) const {
  CHECK_MODEL_LOADER;
  LOGT(MEMORY) << "Free output memory on CPU";
  uint32_t num = model_->OutputNum();
  for (uint32_t i = 0; i < num; ++i) {
    delete[] reinterpret_cast<float *>(ptr[i]);
  }
  delete[] ptr;
}

void MluMemoryOp::FreeMluInput(void **ptr) const {
  CHECK_MODEL_LOADER;
  LOGT(MEMORY) << "Free input memory on MLU";
  uint32_t mem_num = model_->InputNum();
  for (uint32_t i = 0; i < mem_num; ++i) {
    cnrtRet_t ret = cnrtFree(ptr[i]);
    if (ret != CNRT_RET_SUCCESS) {
      LOGE(MEMORY) << "free MLU input memory failed";
    }
  }
  delete[] ptr;
}

void MluMemoryOp::FreeMluOutput(void **ptr) const {
  CHECK_MODEL_LOADER;
  LOGT(MEMORY) << "Free input memory on MLU";
  uint32_t mem_num = model_->OutputNum();
  for (uint32_t i = 0; i < mem_num; ++i) {
    cnrtRet_t ret = cnrtFree(ptr[i]);
    if (ret != CNRT_RET_SUCCESS) {
      LOGE(MEMORY) << "free MLU output memory failed";
    }
  }
  delete[] ptr;
}

void MluMemoryOp::FreeMlu(void *ptr) {
  LOGT(MEMORY) << "Free memory on MLU";
  cnrtRet_t ret = cnrtFree(ptr);
  if (ret != CNRT_RET_SUCCESS) {
    LOGE(MEMORY) << "free MLU memory failed";
  }
}

void MluMemoryOp::MemcpyInputH2D(void **mlu_dst, void **cpu_src) const {
  CHECK_MODEL_LOADER;
  ONLY_SUPPORT_FLOAT32_ON_CPU;
  ModelLoaderInternalInterface interface(model_.get());
  cnrtRet_t error_code;
  LOGA(MEMORY) << "copy input memory from host to device";

  int64_t num = model_->InputNum();
  for (int i = 0; i < num; ++i) {
    void *src = cpu_src[i];
    void *dst = mlu_dst[i];
    size_t size = interface.InputDataSize(i);

    // format data
    DataLayout cpu_layout = model_->GetCpuInputLayout(i);
    DataLayout mlu_layout = interface.GetMluInputLayout(i);
    const ShapeEx& sp = model_->InputShape(i);
    void *temp_data = malloc(size);
    CHECK(MEMORY, temp_data) << "Malloc temp data on cpu failed.";
    TransLayout(cpu_layout, mlu_layout, src, temp_data, sp);
    LOGA(MEMORY) << "MemcpyInputH2D in size " << size << ", dst: " << dst << ", src: " << src << ", tmp: " << temp_data;
    error_code = cnrtMemcpy(dst, temp_data, size, CNRT_MEM_TRANS_DIR_HOST2DEV);
    CHECK_CNRT_RET(error_code, "Memcpy host to device failed.");
    free(temp_data);
  }
}

void MluMemoryOp::MemcpyOutputD2H(void **cpu_dst, void **mlu_src) const {
  CHECK_MODEL_LOADER;
  ONLY_SUPPORT_FLOAT32_ON_CPU;
  ModelLoaderInternalInterface interface(model_.get());
  LOGA(MEMORY) << "copy output memory from device to host";

  int64_t num = model_->OutputNum();
  for (int i = 0; i < num; ++i) {
    void *src = mlu_src[i];
    void *dst = cpu_dst[i];
    size_t size = interface.OutputDataSize(i);
    void *temp_data = malloc(size);
    CHECK(MEMORY, temp_data) << "Malloc temp data on cpu failed.";
    LOGA(MEMORY) << "MemcpyOutputD2H in size " << size << ", dst: " << dst << ", src: " << src
                 << ", tmp: " << temp_data;
    auto error_code = cnrtMemcpy(temp_data, src, size, CNRT_MEM_TRANS_DIR_DEV2HOST);
    CHECK_CNRT_RET(error_code, "Memcpy device to host failed.");
    // format data
    DataLayout cpu_layout = model_->GetCpuOutputLayout(i);
    DataLayout mlu_layout = interface.GetMluOutputLayout(i);
    const ShapeEx& sp = model_->OutputShape(i);
    TransLayout(mlu_layout, cpu_layout, temp_data, dst, sp);
    free(temp_data);
  }
}

void MluMemoryOp::MemcpyH2D(void *mlu_dst, void *cpu_src, size_t nBytes) {
  cnrtRet_t error_code;
  LOGA(MEMORY) << "copy memory from host to device in size " << nBytes << ", dst: " << mlu_dst << ", src: " << cpu_src;
  error_code = cnrtMemcpy(mlu_dst, cpu_src, nBytes, CNRT_MEM_TRANS_DIR_HOST2DEV);
  CHECK_CNRT_RET(error_code, "Memcpy host to device failed.");
}

void MluMemoryOp::MemcpyD2H(void *cpu_dst, void *mlu_src, size_t nBytes) {
  cnrtRet_t error_code;
  LOGA(MEMORY) << "copy memory from device to host in size " << nBytes << ", dst: " << cpu_dst << ", src: " << mlu_src;
  error_code = cnrtMemcpy(cpu_dst, mlu_src, nBytes, CNRT_MEM_TRANS_DIR_DEV2HOST);
  CHECK_CNRT_RET(error_code, "Memcpy host to device failed.");
}

void MluMemoryOp::MemcpyD2D(void *mlu_dst, void *mlu_src, size_t nBytes) {
  cnrtRet_t error_code;
  LOGA(MEMORY) << "copy memory from device to device in size " << nBytes << ", dst: " << mlu_dst
               << ", src: " << mlu_src;
  error_code = cnrtMemcpy(mlu_dst, mlu_src, nBytes, CNRT_MEM_TRANS_DIR_DEV2DEV);
  CHECK_CNRT_RET(error_code, "Memcpy device to device failed.");
}

}  // namespace edk
