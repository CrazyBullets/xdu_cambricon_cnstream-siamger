/*************************************************************************
 *  Copyright (C) [2019] by Lerenhua, Inc. All rights reserved
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

#include "osd.hpp"


#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <cstdio>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#if (CV_MAJOR_VERSION >= 3)
#include "opencv2/imgcodecs/imgcodecs.hpp"
#endif

// #include "cnfont.hpp"
// #include "cnosd.hpp"
#include "cnstream_frame_va.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

#define CLIP(x) x < 0 ? 0 : (x > 1 ? 1 : x)
namespace cnstream {
class CnFont {
#ifdef HAVE_FREETYPE

 public:
  /**
   * @brief Constructor of CnFont
   */
  CnFont() { }
  /**
   * @brief Release font resource
   */
  ~CnFont();
  /**
   * @brief Initialize the display font
   * @param
   *   font_path: the font of path
   */
  bool Init(const std::string &font_path, float font_pixel = 30, float space = 0.4, float step = 0.15);

  /**
   * @brief Configure font Settings
   */
  void restoreFont(float font_pixel = 30, float space = 0.4, float step = 0.15);
  /**
   * @brief Displays the string on the image
   * @param
   *   img: source image
   *   text: the show of message
   *   pos: the show of position
   *   color: the color of font
   * @return Size of the string
   */
  int putText(cv::Mat& img, char* text, cv::Point pos, cv::Scalar color);  // NOLINT
  bool GetTextSize(char* text, uint32_t* width, uint32_t* height);
  uint32_t GetFontPixel();

 private:
  void GetWCharSize(wchar_t wc, uint32_t* width, uint32_t* height);
  /**
   * @brief Converts character to wide character
   * @param
   *   src: The original string
   *   dst: The Destination wide string
   *   locale: Coded form
   * @return
   *   -1: Conversion failure
   *    0: Conversion success
   */
  int ToWchar(char*& src, wchar_t*& dest, const char* locale = "C.UTF-8");  // NOLINT

  /**
   * @brief Print single wide character in the image
   * @param
   *   img: source image
   *   wc: single wide character
   *   pos: the show of position
   *   color: the color of font
   */
  void putWChar(cv::Mat& img, wchar_t wc, cv::Point& pos, cv::Scalar color);  // NOLINT
  CnFont& operator=(const CnFont&);

  FT_Library m_library;
  FT_Face m_face;
  bool is_initialized_ = false;

  // Default font output parameters
  int m_fontType;
  cv::Scalar m_fontSize;
  bool m_fontUnderline;
  float m_fontDiaphaneity;
#else

 public:
  explicit CnFont(const char* font_path) {}
  ~CnFont() {}
  int putText(cv::Mat& img, char* text, cv::Point pos, cv::Scalar color) { return 0; }  // NOLINT
  bool GetTextSize(char* text, uint32_t* width, uint32_t* height) { return true; }
  uint32_t GetFontPixel() { return 0; }
#endif
};  // class CnFont

class CnOsd {
 public:
  CnOsd() = delete;
  explicit CnOsd(const std::vector<std::string>& labels);

  inline void SetTextScale(float scale)  { text_scale_ = scale; }
  inline void SetTextThickness(float thickness)  { text_thickness_ = thickness; }
  inline void SetBoxThickness(float thickness)  { box_thickness_ = thickness; }
  inline void SetSecondaryLabels(std::vector<std::string> labels) { secondary_labels_ = labels; }
  inline void SetCnFont(std::shared_ptr<CnFont> cn_font) { cn_font_ = cn_font; }

  void DrawLabel(cv::Mat image, const CNInferObjsPtr &objects, std::vector<std::string> attr_keys = {}) const;
  void DrawLogo(cv::Mat image, std::string logo) const;
  void FormatResult(std::stringstream& ss, const CNInferObjsPtr& objs_holder, const int& cols, const int& rows);
  void TrackResult(std::stringstream& ss, const CNInferObjsPtr& objs_holder, const int& frame_id, const int& cols, const int& rows);
  void TrackResult(std::stringstream& ss, const CNInferObjsPtr& objs_holder, const int& frame_id, const int& cols, const int& rows, std::string track_target);
  
 private:
  std::pair<cv::Point, cv::Point> GetBboxCorner(const cnstream::CNInferObject &object,
                                                int img_width, int img_height) const;
  bool LabelIsFound(const int &label_id) const;
  int GetLabelId(const std::string &label_id_str) const;
  void DrawBox(cv::Mat image, const cv::Point &top_left, const cv::Point &bottom_right, const cv::Scalar &color) const;
  void DrawText(cv::Mat image, const cv::Point &bottom_left, const std::string &text, const cv::Scalar &color,
                float scale = 1, int *text_height = nullptr) const;
  int CalcThickness(int image_width, float thickness) const;
  double CalcScale(int image_width, float scale) const;

  float text_scale_ = 1;
  float text_thickness_ = 1;
  float box_thickness_ = 1;
  std::vector<std::string> labels_;
  std::vector<std::string> secondary_labels_;
  std::vector<cv::Scalar> colors_;
  int font_ = cv::FONT_HERSHEY_SIMPLEX;
  std::shared_ptr<CnFont> cn_font_;
};  // class CnOsd

static std::vector<std::string> LoadLabels(const std::string& label_path) {
  std::vector<std::string> labels;
  std::ifstream ifs(label_path);
  if (!ifs.is_open()) return labels;

  while (!ifs.eof()) {
    std::string label_name;
    std::getline(ifs, label_name);
    labels.push_back(label_name);
  }

  ifs.close();
  return labels;
}





class MotTcpOsd : public Module, public ModuleCreator<MotTcpOsd> {
    public:
    /**
     *  @brief  Generate osd and dump the detections result
     *
     *  @param  Name : Module name
     *
     *  @return None
     */
    explicit MotTcpOsd(const std::string& name): Module(name) {
        param_register_.SetModuleDesc("MotTcpOsd is a module for drawing objects on image and dumping the detection results.");
        param_register_.Register("label_path", "The path of the label file.");
        param_register_.Register("font_path", "The path of font.");
        param_register_.Register("label_size", " The size of the label, support value: "
                                "normal, large, larger, small, smaller and number. The default value is normal");
        param_register_.Register("text_scale", "The scale of the text, which can change the size of text put on image. "
                                "The default value is 1. scale = label_size * text_scale");
        param_register_.Register("text_thickness", "The thickness of the text, which can change the thickness of text put on "
                                "image. The default value is 1. thickness = label_size * text_thickness");
        param_register_.Register("box_thickness", "The thickness of the box drawn on the image. "
                                "thickness = label_size * box_thickness");
        param_register_.Register("secondary_label_path", "The path of the secondary label file");
        param_register_.Register("attr_keys", "The keys of attribute which you want to draw on image");
        param_register_.Register("logo", "draw 'logo' on each frame");
        param_register_.Register("mot_tcp_ip", "The tcp ip of the track result.");
        param_register_.Register("mot_tcp_port", "The port of the tcp ip.");
        param_register_.Register("track_target", "track_target.");


    }
    /**
     * @brief Release osd
     * @param None
     * @return None
     */
    ~MotTcpOsd() { Close(); }

    /**
     * @brief Called by pipeline when pipeline start.
     *
     * @param paramSet :
     * @verbatim
     *   label_path: label path
     * @endverbatim
     *
     * @return if module open succeed
     */
    bool Open(cnstream::ModuleParamSet paramSet) override {
        std::string label_path = "";
        if (paramSet.find("label_path") == paramSet.end()) {
            LOGW(OSD) << "Can not find label_path from module parameters.";
        } else {
            label_path = paramSet["label_path"];
            label_path = GetPathRelativeToTheJSONFile(label_path, paramSet);
            labels_ = LoadLabels(label_path);
            if (labels_.empty()) {
                LOGW(OSD) << "Empty label file or wrong file path.";
            } else {
            #ifdef HAVE_FREETYPE
                if (paramSet.find("font_path") != paramSet.end()) {
                std::string font_path = paramSet["font_path"];
                font_path_ = GetPathRelativeToTheJSONFile(font_path, paramSet);
                }
            #endif
            }
        }
        
        if (paramSet.find("label_size") != paramSet.end()) {
            std::string label_size = paramSet["label_size"];
            if (label_size == "large") {
                label_size_ = 1.5;
            } else if (label_size == "larger") {
                label_size_ = 2;
            } else if (label_size == "small") {
                label_size_ = 0.75;
            } else if (label_size == "smaller") {
                label_size_ = 0.5;
            } else if (label_size != "normal") {
                float size = std::stof(paramSet["label_size"]);
                label_size_ = size;
            }
        }

        if (paramSet.find("text_scale") != paramSet.end()) {
            text_scale_ = std::stof(paramSet["text_scale"]);
        }

        if (paramSet.find("text_thickness") != paramSet.end()) {
            text_thickness_ = std::stof(paramSet["text_thickness"]);
        }

        if (paramSet.find("box_thickness") != paramSet.end()) {
            box_thickness_ = std::stof(paramSet["box_thickness"]);
        }

        if (paramSet.find("secondary_label_path") != paramSet.end()) {
            label_path = paramSet["secondary_label_path"];
            label_path = GetPathRelativeToTheJSONFile(label_path, paramSet);
            secondary_labels_ = LoadLabels(label_path);
        }

        if (paramSet.find("attr_keys") != paramSet.end()) {
            std::string attr_key = paramSet["attr_keys"];
            attr_key.erase(std::remove_if(attr_key.begin(), attr_key.end(), ::isspace), attr_key.end());
            attr_keys_ = StringSplit(attr_key, ',');
        }

        if (paramSet.find("logo") != paramSet.end()) {
            logo_ = paramSet["logo"];
        }

        if (paramSet.find("mot_tcp_ip") != paramSet.end()) {
            mot_tcp_ip = paramSet["mot_tcp_ip"];
            mot_tcp_port = paramSet["mot_tcp_port"];
            // mot_dump_path_ = GetPathRelativeToTheJSONFile(mot_dump_path_, paramSet);
            bzero(&ser_addr, sizeof(ser_addr));
            ser_addr.sin_family = AF_INET;
            if(mot_tcp_port==""){
                ser_addr.sin_port = htons(std::stoi(mot_tcp_port));
            }
            else{
                ser_addr.sin_port = htons(8000);   
            }
            ser_addr.sin_addr.s_addr = inet_addr(mot_tcp_ip.c_str());
            sockfd = socket(AF_INET, SOCK_STREAM, 0);
            ret = connect(sockfd, (struct sockaddr *)&ser_addr, sizeof(ser_addr));
            if(ret == 0){
                LOGE(OSD)<<"start sending data.";
            }
            else{
                LOGE(OSD)<<"can't connect to server.";
            }
        }

        if(paramSet.find("track_target") != paramSet.end()){
            track_target = paramSet["track_target"];
        }

        return true;
    }

    /**
     * @brief  Called by pipeline when pipeline stop
     *
     * @param  None
     *
     * @return  None
     */
    void Close() override {
        close(sockfd);
        osd_ctxs_.clear();
        
    }

    std::pair<cv::Point, cv::Point> GetBboxCorner(const CNInferObject& object, int img_width, int img_height) const {
        float x = CLIP(object.bbox.x);
        float y = CLIP(object.bbox.y);
        float w = CLIP(object.bbox.w);
        float h = CLIP(object.bbox.h);
        w = (x + w > 1) ? (1 - x) : w;
        h = (y + h > 1) ? (1 - y) : h;
        cv::Point top_left(x * img_width, y * img_height);
        cv::Point bottom_right((x + w) * img_width, (y + h) * img_height);
        return std::make_pair(top_left, bottom_right);
    }
    
    void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c) {
        std::string::size_type pos1, pos2;
        pos2 = s.find(c);
        pos1 = 0;
        while(std::string::npos != pos2)
        {
            v.push_back(s.substr(pos1, pos2-pos1));
        
            pos1 = pos2 + c.size();
            pos2 = s.find(c, pos1);
        }
        if(pos1 != s.length())
            v.push_back(s.substr(pos1));
    }

    /**
     * @brief Do for each frame
     *
     * @param data : Pointer to the frame info
     *
     * @return whether process succeed
     * @retval 0: succeed and do no intercept data
     * @retval <0: failed
     *
     */
    int Process(std::shared_ptr<CNFrameInfo> data) override {
        std::shared_ptr<CnOsd> ctx = GetOsdContext();
        // std::cout << "call the custiom osd!" << std::endl;
        if (ctx == nullptr) {
            LOGE(OSD) << "Get Osd Context Failed.";
            return -1;
        }

        CNDataFramePtr frame = data->collection.Get<CNDataFramePtr>(kCNDataFrameTag);
        if (frame->width < 0 || frame->height < 0) {
            LOGE(OSD) << "OSD module processed illegal frame: width or height may < 0.";
            return -1;
        }

        CNInferObjsPtr objs_holder = nullptr;
        if (data->collection.HasValue(kCNInferObjsTag)) {
            objs_holder = data->collection.Get<CNInferObjsPtr>(kCNInferObjsTag);
        }

        // if (!logo_.empty()) {
        //     ctx->DrawLogo(frame->ImageBGR(), logo_);
        // }
        // ctx->DrawLabel(frame->ImageBGR(), objs_holder, attr_keys_);


        // tcp通信

        if (ret == 0) {

            std::stringstream ss;
            char buf[38];
            // std::ofstream file(name,std::ios::app);
            ctx->TrackResult(ss, objs_holder, frame->frame_id, frame->ImageBGR().cols, frame->ImageBGR().rows, track_target);
            
            while(ss >> buf){
                // LOGE(OSD)<<buf;
                // send(sockfd, buf, strlen(buf), 0 );
                sendData(sockfd,buf,sizeof(buf));
            }
            // ss.clear();
            // ss.str("");
            // file << ss.str();
            // file.close();
        } else {
            LOGE(OSD) << "sever not found";
        }

        return 0;
    }

    int sendData(int socketfd, char* msg, int len)
    {
        if(msg == NULL || len <= 0 || socketfd <=0)
        {
            return -1;
        }
        // 申请内存空间: 数据长度 + 包头4字节(存储数据长度)
        char* data = (char*)malloc(len+4);
        int bigLen = htonl(len);
        memcpy(data, &bigLen, 4);
        memcpy(data+4, msg, len);
        // 发送数据
        int ret = writen(socketfd, data, len+4);
        // 释放内存
        free(data);
        return ret;
    }

    int writen(int socketfd, const char* msg, int size)
    {
        const char* buf = msg;
        int count = size;
        while (count > 0)
        {
            int len = send(socketfd, buf, count, 0);
            if (len == -1)
            {
                close(socketfd);
                return -1;
            }
            else if (len == 0)
            {
                continue;
            }
            buf += len;
            count -= len;
        }
        return size;
    }



    /**
     * @brief Check ParamSet for a module.
     *
     * @param paramSet Parameters for this module.
     *
     * @return Returns true if this API run successfully. Otherwise, returns false.
     */
    bool CheckParamSet(const ModuleParamSet& paramSet) const override {
        bool ret = true;
        ParametersChecker checker;
        for (auto& it : paramSet) {
            if (!param_register_.IsRegisted(it.first)) {
            LOGW(OSD) << "[Osd] Unknown param: " << it.first;
            }
        }
        if (paramSet.find("label_path") != paramSet.end()) {
            if (!checker.CheckPath(paramSet.at("label_path"), paramSet)) {
            LOGE(OSD) << "[Osd] [label_path] : " << paramSet.at("label_path") << " non-existence.";
            ret = false;
            }
        }
        if (paramSet.find("font_path") != paramSet.end()) {
            if (!checker.CheckPath(paramSet.at("font_path"), paramSet)) {
            LOGE(OSD) << "[Osd] [font_path] : " << paramSet.at("font_path") << " non-existence.";
            ret = false;
            }
        }
        if (paramSet.find("secondary_label_path") != paramSet.end()) {
            if (!checker.CheckPath(paramSet.at("secondary_label_path"), paramSet)) {
            LOGE(OSD) << "[Osd] [secondary_label_path] : " << paramSet.at("secondary_label_path") << " non-existence.";
            ret = false;
            }
        }
        std::string err_msg;
        if (!checker.IsNum({"text_scale", "text_thickness", "box_thickness"}, paramSet, err_msg)) {
            LOGE(OSD) << "[Osd] " << err_msg;
            ret = false;
        }
        if (paramSet.find("label_size") != paramSet.end()) {
            std::string label_size = paramSet.at("label_size");
            if (label_size != "normal" && label_size != "large" && label_size != "larger" &&
                label_size != "small" && label_size != "smaller") {
            if (!checker.IsNum({"label_size"}, paramSet, err_msg)) {
                LOGE(OSD) << "[Osd] " << err_msg << " Please choose from 'normal', 'large', 'larger', 'small', 'smaller'."
                        << " Or set a number to it.";
                ret = false;
            }
            }
        }
        return ret;
    }

private:
    std::shared_ptr<CnOsd> GetOsdContext() {
        std::shared_ptr<CnOsd> ctx = nullptr;
        std::thread::id thread_id = std::this_thread::get_id();
        {
            RwLockReadGuard lg(ctx_lock_);
            if (osd_ctxs_.find(thread_id) != osd_ctxs_.end()) {
            ctx = osd_ctxs_[thread_id];
            }
        }
        if (!ctx) {
            ctx = std::make_shared<CnOsd>(labels_);
            if (!ctx) {
            LOGE(OSD) << "Osd::GetOsdContext() create Osd Context Failed";
            return nullptr;
            }
            ctx->SetTextScale(label_size_ * text_scale_);
            ctx->SetTextThickness(label_size_ * text_thickness_);
            ctx->SetBoxThickness(label_size_ * box_thickness_);

            ctx->SetSecondaryLabels(secondary_labels_);

        #ifdef HAVE_FREETYPE
            if (!font_path_.empty()) {
            std::shared_ptr<CnFont> font = std::make_shared<CnFont>();
            float font_size = label_size_ * text_scale_ * 30;
            float space = font_size / 75;
            float step = font_size / 200;
            if (font && font->Init(font_path_, font_size, space, step)) {
                ctx->SetCnFont(font);
            } else {
                LOGE(OSD) << "Create and initialize CnFont failed.";
            }
            }
        #endif
            RwLockWriteGuard lg(ctx_lock_);
            osd_ctxs_[thread_id] = ctx;
        }
        return ctx;
    }

    std::unordered_map<std::thread::id, std::shared_ptr<CnOsd>> osd_ctxs_;
    RwLock ctx_lock_;
    std::vector<std::string> labels_;
    std::vector<std::string> secondary_labels_;
    std::vector<std::string> attr_keys_;
    std::string font_path_ = "";
    std::string logo_ = "";

    // std::string mot_dump_path_ = "";        // The path of object detections result json file
    // std::string name;
    std::string mot_tcp_ip = "";
    std::string mot_tcp_port = "";
    std::string track_target;
    float text_scale_ = 1;
    float text_thickness_ = 1;
    float box_thickness_ = 1;
    float label_size_ = 1;
    int init = 0;

    int sockfd ;
	int ret;
    struct sockaddr_in ser_addr;
};
}