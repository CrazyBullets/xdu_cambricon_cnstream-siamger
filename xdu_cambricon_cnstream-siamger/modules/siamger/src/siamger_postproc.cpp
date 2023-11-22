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

#include "siamger_postproc.hpp"

#include <string>

namespace cnstream {

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

SiamgerPostproc::~SiamgerPostproc() {}

SiamgerPostproc* SiamgerPostproc::Create(const std::string& proc_name) {
  return ReflexObjectEx<SiamgerPostproc>::CreateObject(proc_name);
}

void SiamgerPostproc::SetThreshold(const float threshold) { threshold_ = threshold; }

void SiamgerPostproc::SetSizeCenter(const std::vector<uint32_t> init_loc)
{
  size_[0] = init_loc[2];
  size_[1] = init_loc[3];
  center_pos_[0] = init_loc[0] + static_cast<float>(init_loc[2] - 1) / 2;
  center_pos_[1] = init_loc[1] + static_cast<float>(init_loc[3] - 1) / 2;
}

void SiamgerPostproc::generate_points(int stride, int size)
{
  std::vector<std::vector<int>> points(size * size, std::vector<int>(2));

  int ori = - (size / 2) * stride;

  for(int i = 0;i < size;++i)
  {
    for (int j = 0;j < size;++j)
    {
      points[i * size + j][0] = ori + stride * j;
      points[i * size + j][1] = ori + stride * i;
    }
  }

  points_ = points;
}

void SiamgerPostproc::hamming(int len)
{
    std::vector<float> win;
    win.resize(len);
    float a = 0.5; 
    float PI = 3.1415926;
    for (int i = 0; i < len; i++)
    {
      win[i] = a - (1.0f - a) * cos(2 * PI * i / (len - 1));
    }
 
    for (int i = 0;i < len;++i)
    {
      for (int j = 0;j < len;++j)
      {
        window_[i * len + j] = win[i] * win[j];
      }
    }
}



}  // namespace cnstream
