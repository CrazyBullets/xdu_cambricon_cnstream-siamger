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

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cnstream_frame_va.hpp"
#include "cnstream_module.hpp"

namespace cnstream {

class TestOsd : public Module, public ModuleCreator<TestOsd> {
    public:
    explicit TestOsd(const std::string& name): Module(name) {}
    ~TestOsd() { Close(); }
    bool Open(cnstream::ModuleParamSet paramSet) override {
        return true;
    }
    void Close() override {}

    int Process(std::shared_ptr<CNFrameInfo> data) override {
        return 0;
    }
};  // class TestOsd

}   // namespace cnstream