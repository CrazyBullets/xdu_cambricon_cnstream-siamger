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

#include <gflags/gflags.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if (CV_MAJOR_VERSION >= 3)
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <set>
#include <vector>

#include <sys/socket.h>
#include <netinet/in.h>

#include "cnstream_version.hpp"
#include "data_source.hpp"
#include "udp-piece.hpp"
#ifdef HAVE_DISPLAY
#include "displayer.hpp"
#endif
#include "cnstream_logging.hpp"
#include "util.hpp"

#include "profiler/pipeline_profiler.hpp"
#include "profiler/profile.hpp"
#include "profiler/trace_serialize_helper.hpp"

DEFINE_string(data_path, "", "video file list.");
DEFINE_string(data_name, "", "video file name.");
DEFINE_int32(src_frame_rate, 25, "frame rate for send data");
DEFINE_int32(maximum_video_width, -1, "maximum video width, for variable video resolutions, "
    "not supported on MLU220/MLU270");
DEFINE_int32(maximum_video_height, -1, "maximum video height, for variable video resolutions, "
    "not supported on MLU220/MLU270");
DEFINE_int32(maximum_image_width, 7680, "maximum image width, valid when jpeg_from_mem is true");
DEFINE_int32(maximum_image_height, 4320, "maxmum image width, valid when jpeg_from_mem is true");
DEFINE_int32(wait_time, 0, "time of one test case");
DEFINE_int32(udp_port, 6868, "port of udp, default is 6868");
DEFINE_bool(loop, false, "display repeat");
DEFINE_string(config_fname, "", "pipeline config filename");
DEFINE_bool(jpeg_from_mem, true, "Jpeg bitstream from mem.");
DEFINE_bool(raw_img_input, false, "feed decompressed image to source");
DEFINE_bool(use_cv_mat, true, "feed cv mat to source. It is valid only if ``raw_img_input`` is set to true");
DEFINE_string(trace_data_dir, "", "dump trace data to specified dir. An empty string means that no data is stored");

#ifdef HAVE_DISPLAY
cnstream::Displayer *gdisplayer = nullptr;
#endif

std::atomic<bool> thread_running{true};

std::atomic<bool> gstop_perf_print{false};
std::mutex gmutex_for_add_source;

class UserLogSink : public cnstream::LogSink {
 public:
  void Send(cnstream::LogSeverity severity, const char *category, const char *filename, int line,
            const struct ::tm *tm_time, int32_t usecs, const char *message, size_t message_len) override {
    std::cout << "UserLogSink: " << ToString(severity, category, filename, line, tm_time, usecs, message, message_len)
              << std::endl;
  }
};

class MsgObserver : cnstream::StreamMsgObserver {
 public:
  MsgObserver(cnstream::Pipeline *pipeline, std::string source_name)
      : pipeline_(pipeline), source_name_(source_name) {}

  void Update(const cnstream::StreamMsg &smsg) override {
    std::lock_guard<std::mutex> add_src_lg(gmutex_for_add_source);
    std::lock_guard<std::mutex> lg(mutex_);
    if (stop_) return;
    cnstream::DataSource *source = nullptr;
    source = dynamic_cast<cnstream::DataSource *>(pipeline_->GetModule(source_name_));
    switch (smsg.type) {
      case cnstream::StreamMsgType::EOS_MSG:
        LOGI(DEMO) << "[" << pipeline_->GetName() << "] received EOS message from stream: [" << smsg.stream_id << "]";
        if (stream_set_.find(smsg.stream_id) != stream_set_.end()) {
          if (source) source->RemoveSource(smsg.stream_id);
          stream_set_.erase(smsg.stream_id);
        }
        if (stream_set_.empty()) {
          LOGI(DEMO) << "[" << pipeline_->GetName() << "] received all EOS";
          stop_ = true;
        }
        break;

      case cnstream::StreamMsgType::STREAM_ERR_MSG:
        LOGW(DEMO) << "[" << pipeline_->GetName() << "] received stream error from stream: " << smsg.stream_id
                   << ", remove it from pipeline.";
        if (stream_set_.find(smsg.stream_id) != stream_set_.end()) {
          if (source) source->RemoveSource(smsg.stream_id, true);
          stream_set_.erase(smsg.stream_id);
        }
        if (stream_set_.empty()) {
          LOGI(DEMO) << "[" << pipeline_->GetName() << "] all streams is removed from pipeline, pipeline will stop.";
          stop_ = true;
        }
        break;

      case cnstream::StreamMsgType::ERROR_MSG:
        LOGW(DEMO) << "[" << pipeline_->GetName() << "] received error from stream: " << smsg.stream_id
                   << ", remove sources.";
        if (source) source->RemoveSources(true);
        stream_set_.clear();
        stop_ = true;
        break;

      case cnstream::StreamMsgType::FRAME_ERR_MSG:
        LOGW(DEMO) << "[" << pipeline_->GetName() << "] received frame error from stream: " << smsg.stream_id
                   << ", pts: " << smsg.pts << ".";
        break;

      default:
        LOGE(DEMO) << "[" << pipeline_->GetName() << "] unknown message type.";
        break;
    }
    if (stop_) {
      wakener_.notify_one();
    }
  }

  void WaitForStop() {
    std::unique_lock<std::mutex> lk(mutex_);
    if (stream_set_.empty()) {
     stop_ = true;
    }
    wakener_.wait(lk, [this]() { return stop_.load(); });
    lk.unlock();
    pipeline_->Stop();
  }

  void IncreaseStream(std::string stream_id) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (stream_set_.find(stream_id) != stream_set_.end()) {
      LOGF(DEMO) << "IncreaseStream() The stream is ongoing []" << stream_id;
    }
    stream_set_.insert(stream_id);
    if (stop_) stop_ = false;
  }

 private:
  cnstream::Pipeline *pipeline_ = nullptr;
  std::string source_name_;
  std::atomic<bool> stop_{false};
  std::set<std::string> stream_set_;
  std::condition_variable wakener_;
  mutable std::mutex mutex_;
};

int AddSourceForUdpStream(cnstream::DataSource *source, const std::string &stream_id, const std::string &filename,
                           const cnstream::MaximumVideoResolution& maximum_resolution) {
  auto handler = cnstream::UdpHandler::Create(source, stream_id, filename, 10, maximum_resolution);
  return source->AddSource(handler);
}

int AddSourceForRtspStream(cnstream::DataSource *source, const std::string &stream_id, const std::string &filename,
                           const cnstream::MaximumVideoResolution& maximum_resolution) {
  auto handler = cnstream::RtspHandler::Create(source, stream_id, filename, false, 10, maximum_resolution);
  return source->AddSource(handler);
}

/**
 * @brief Adds source for usb camera.
 *
 * @note Supports Logitech C505e 720P USB CAMERA on Ubuntu 18.04.5.
 *       Required steps:
 *       1. Compiles x264
 *          git clone https://code.videolan.org/videolan/x264.git
 *          cd x264
 *          ./configure --enable-shared  --prefix=/usr/local/x264 --disable-asm
 *          make
 *          sudo make install
 *       2. Compiles ffmpeg
 *          wget http://www.ffmpeg.org/releases/ffmpeg-3.4.8.tar.xz
 *          tar xvf ffmpeg-3.4.8.tar.xz
 *          cd ffmpeg-3.4.8
 *          export PKG_CONFIG_PATH=/usr/local/x264/lib/pkgconfig
 *          ./configure \
 *            --prefix=/usr/local/ \
 *            --enable-shared \
 *            --enable-static \
 *            --enable-gpl \
 *            --enable-nonfree \
 *            --enable-ffmpeg \
 *            --disable-ffplay \
 *            --enable-swscale \
 *            --pkg-config="pkg-config --static" \
 *            --enable-pthreads \
 *            --disable-armv5te \
 *            --disable-armv6 \
 *            --disable-armv6t2 \
 *            --disable-yasm \
 *            --disable-stripping \
 *            --enable-libx264 \
 *            --enable-libv4l2 \
 *            --extra-cflags=-I/usr/local/x264/include \
 *            --extra-ldflags=-L/usr/local/x264/lib
 *          make -j4
 *          sudo make install
 *        3. Modifies modules/CMakeLists.txt
 *           turns WITH_FFMPEG_AVDEVICE into ON
 *        4. Compiles cnstream
 *        5. Runs the demo of usb camera
 *           cd samples/cns_launcher/object_detection
 *           export LD_LIBRARY_PATH=/usr/local/x264/lib:$LD_LIBRARY_PATH
 *           run.sh [mlu220/mlu270/mlu370] [encode_jpeg/encode_video/display/rtsp] usb
 *
 *       And the support of other types of USB CAMERA and operate system is not tested.
 *       Above steps may be a reference.
 */
int AddSourceForUsbCam(cnstream::DataSource *source, const std::string &stream_id, const std::string &filename,
                       const int &frame_rate, const bool &loop,
                       const cnstream::MaximumVideoResolution& maximum_resolution) {
  int ret = -1;
  auto handler = cnstream::FileHandler::Create(source, stream_id, filename, frame_rate, loop, maximum_resolution);
  ret = source->AddSource(handler);
  return ret;
}
std::vector<std::future<int>> gFeedMemFutures;
int AddSourceForVideoInMem(cnstream::DataSource *source, const std::string &stream_id, const std::string &filename,
                           const bool &loop, const cnstream::MaximumVideoResolution& maximum_resolution) {
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    LOGE(DEMO) << "Open file failed. file name : " << filename;
    return -1;
  }

  auto handler = cnstream::ESMemHandler::Create(source, stream_id, maximum_resolution);
  if (source->AddSource(handler)) {
    LOGE(DEMO) << "failed to add  " << stream_id;
    return -1;
  }

  // Start another thread to read file's binary data into memory and feed it to pipeline.
  gFeedMemFutures.emplace_back(std::async(std::launch::async, [=]() {
    auto memHandler = std::dynamic_pointer_cast<cnstream::ESMemHandler>(handler);
    memHandler->SetDataType(cnstream::ESMemHandler::DataType::H264);

    unsigned char buf[4096];
    while (thread_running.load()) {
      if (!feof(fp)) {
        int size = fread(buf, 1, 4096, fp);
        if (memHandler->Write(buf, size) != 0) {
          break;
        }
      } else if (loop) {
        fseek(fp, 0, SEEK_SET);
      } else {
        break;
      }
    }

    if (!feof(fp)) {
      memHandler->WriteEos();
    } else {
      memHandler->Write(nullptr, 0);
    }

    fclose(fp);

    return 0;
  }));

  return 0;
}

int AddSourceForImageInMem(cnstream::DataSource *source, const std::string &stream_id, const std::string &filename,
                           const bool &loop, std::vector<std::string> & filename_vecs) {
  int index = filename.find_last_of("/");
  std::string dir_path = filename.substr(0, index);
  std::list<std::string> files = GetFileNameFromDir(dir_path, "*.jpg");
  if (files.empty()) {
    LOGE(DEMO) << "there is no jpg files";
    return -1;
  }

  auto handler = cnstream::ESJpegMemHandler::Create(source, stream_id,
      FLAGS_maximum_image_width, FLAGS_maximum_image_height);
  if (source->AddSource(handler)) {
    LOGE(DEMO) << "failed to add  " << stream_id;
    return -1;
  }

  // Start another thread to read file's binary data into memory and feed it to pipeline.
  gFeedMemFutures.emplace_back(std::async(std::launch::async, [files, handler, loop, &filename_vecs]() {
    auto memHandler = std::dynamic_pointer_cast<cnstream::ESJpegMemHandler>(handler);
    memHandler->SetFilenameList(&filename_vecs);
    auto iter = files.begin();
    size_t jpeg_buffer_size = GetFileSize(*iter);

    std::unique_ptr<unsigned char[]> buf(new unsigned char[jpeg_buffer_size]);

    cnstream::ESPacket pkt;
    uint64_t pts = 0;
    while (thread_running.load() && iter != files.end()) {
      size_t file_size = GetFileSize(*iter);
      if (file_size > jpeg_buffer_size) {
        buf.reset(new unsigned char[file_size]);
        jpeg_buffer_size = file_size;
      }
      filename_vecs.push_back(*iter);
      std::ifstream file_stream(*iter, std::ios::binary);
      if (!file_stream.is_open()) {
        LOGW(DEMO) << "failed to open " << (*iter);
      } else {
        file_stream.read(reinterpret_cast<char *>(buf.get()), file_size);
        pkt.data = buf.get();
        pkt.size = file_size;
        pkt.pts = pts++;
        if (memHandler->Write(&pkt) != 0) {
          break;
        }
      }

      if (++iter == files.end() && loop) {
        iter = files.begin();
      }
    }
    pkt.data = nullptr;
    pkt.size = 0;
    pkt.flags = static_cast<size_t>(cnstream::ESPacket::FLAG::FLAG_EOS);
    memHandler->Write(&pkt);
    return 0;
  }));

  return 0;
}

int AddSourceForImageInUdp(cnstream::DataSource *source, const std::string &stream_id, const std::string &filename,
                           const bool &loop) {
  auto handler = cnstream::UdpJpegMemHandler::Create(source, stream_id,
      FLAGS_maximum_image_width, FLAGS_maximum_image_height);
  if (source->AddSource(handler)) {
    LOGE(UDP_PIECE) << "failed to add  " << stream_id;
    return -1;
  }

  /* sock_fd --- socket文件描述符 创建udp套接字*/
  int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if(sock_fd < 0)
  {
    LOGE(UDP_PIECE) << "Failed to create socket";
    return false;
  }
  /* 将套接字和IP、端口绑定 */
  struct sockaddr_in addr_serv; 
  struct sockaddr_in addr_client;

  memset(&addr_serv, 0, sizeof(struct sockaddr_in));    //每个字节都用0填充
  addr_serv.sin_family = AF_INET;                       //使用IPV4地址
  addr_serv.sin_port = htons(FLAGS_udp_port);                //端口
  /* INADDR_ANY表示不管是哪个网卡接收到数据，只要目的端口是SERV_PORT，就会被该应用程序接收到 */
  addr_serv.sin_addr.s_addr = htonl(INADDR_ANY);  //自动获取IP地址
  int len = sizeof(addr_serv);

  /* 绑定socket */
  if(bind(sock_fd, (struct sockaddr *)&addr_serv, sizeof(addr_serv)) < 0)
  {
    LOGE(UDP_PIECE) << "Failed to bind socket, port:" << FLAGS_udp_port;
    return false;
  }
  LOGI(UDP_PIECE) << "Successfully bind socket, port:" << FLAGS_udp_port;

  // Start another thread to read file's binary data into memory and feed it to pipeline.
  gFeedMemFutures.emplace_back(std::async(std::launch::async, [sock_fd, addr_serv, addr_client, len, handler]() {
    auto memHandler = std::dynamic_pointer_cast<cnstream::UdpJpegMemHandler>(handler);
    bool udp_stop_flag = false;
    int ret;                        // return code of udp pieces merging
    int recv_num;         
    int send_num;
    char send_buf[20] = "Save img";
    unsigned char recv_buf[65535];  // buffer number
    // udp piece
    udp_piece_t *udp_piece = udp_piece_init(64*1024);

    cnstream::ESPacket pkt;
    uint64_t pts = 0;
    while (thread_running.load() && !udp_stop_flag) {
      recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *)&addr_client, (socklen_t *)&len);

      if(recv_num < 0)
      {
        LOGW(UDP_PIECE) << "recvfrom error:";
      }else
      {
        // LOGI(UDP_PIECE) << "server received " << recv_num << " bytes: \n";
        ret = udp_piece_merge(udp_piece, recv_buf, recv_num);

        if (ret == 1)   
        {
          // Processing completed data
          send_num = sendto(sock_fd, send_buf, 20, 0, (struct sockaddr *)&addr_client, len);

          if(send_num < 0)
          {
            LOGW(UDP_PIECE) << "Sendto error";
          }
          pkt.data = udp_piece->recv_buf;
          pkt.size = udp_piece->recv_len;
          pkt.pts = pts++;
          if (memHandler->Write(&pkt) != 0) {
            break;
          }

          udp_piece_reset(udp_piece);
        }else if(ret == 0)
        {
          LOGD(UDP_PIECE) << "Continue merging udp pieces ";
        }else if(ret == -1)
        {
          LOGW(UDP_PIECE) << "Unexpected udp pieces";
          udp_piece_reset(udp_piece);
        }else
        {
          LOGW(UDP_PIECE) << "Unkonwed error";
          udp_piece_reset(udp_piece);
        }
        
        
      }
    }
    pkt.data = nullptr;
    pkt.size = 0;
    pkt.flags = static_cast<size_t>(cnstream::ESPacket::FLAG::FLAG_EOS);
    memHandler->Write(&pkt);
    return 0;
  }));

  return 0;
}

int AddSourceForUdpDecompressedImage(cnstream::DataSource *source, const std::string &stream_id,
                                  const std::string &filename, const bool &loop, const bool &use_cv_mat) {
  // The following code is only for image input. For video input, you could use OpenCV VideoCapture.
  
  auto handler = cnstream::UdpRawImgMemHandler::Create(source, stream_id, filename);
  if (source->AddSource(handler)) {
    LOGE(DEMO) << "failed to add  " << stream_id;
    return -1;
  }

  /* sock_fd --- socket文件描述符 创建udp套接字*/
  int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if(sock_fd < 0)
  {
    LOGE(DEMO) << "Failed to create socket";
    return false;
  }
  /* 将套接字和IP、端口绑定 */
  struct sockaddr_in addr_serv; 
  struct sockaddr_in addr_client;

  memset(&addr_serv, 0, sizeof(struct sockaddr_in));    //每个字节都用0填充
  addr_serv.sin_family = AF_INET;                       //使用IPV4地址
  addr_serv.sin_port = htons(FLAGS_udp_port);                //端口
  /* INADDR_ANY表示不管是哪个网卡接收到数据，只要目的端口是SERV_PORT，就会被该应用程序接收到 */
  addr_serv.sin_addr.s_addr = htonl(INADDR_ANY);  //自动获取IP地址
  int len = sizeof(addr_serv);

  /* 绑定socket */
  if(bind(sock_fd, (struct sockaddr *)&addr_serv, sizeof(addr_serv)) < 0)
  {
    LOGE(DEMO) << "Failed to bind socket, port:" << FLAGS_udp_port;
    return false;
  }
  LOGI(DEMO) << "Successfully bind socket, port:" << FLAGS_udp_port;


  // Start another thread to read data from udp to cv mat and feed data to pipeline.
  gFeedMemFutures.emplace_back(std::async(std::launch::async, [sock_fd, addr_serv, addr_client, len, handler, use_cv_mat]() {
    auto memHandler = std::dynamic_pointer_cast<cnstream::UdpRawImgMemHandler>(handler);
    bool udp_stop_flag = false;
    int recv_num;
    int send_num;
    char send_buf[20] = "Save img";
    unsigned char recv_buf[65535];  // buffer number
    int ret_code = -1;
    uint64_t pts = 0;

    while (thread_running.load() && !udp_stop_flag) {
      recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *)&addr_client, (socklen_t *)&len);
      if(recv_num < 0)
      {
        LOGW(DEMO) << "recvfrom error:";
      }else
      {
        LOGD(DEMO) << "server received " << recv_num << " bytes: \n";

        send_num = sendto(sock_fd, send_buf, recv_num, 0, (struct sockaddr *)&addr_client, len);

        if(send_num < 0)
        {
          LOGW(DEMO) << "sendto error:";
        }

        std::vector<unsigned char> src_code(recv_num);
        for(int i=0;i<recv_num;i++)
        {
          src_code[i] = recv_buf[i];
        }
        cv::Mat bgr_frame = cv::imdecode(src_code, CV_LOAD_IMAGE_COLOR);
        if (!bgr_frame.empty()) {
          if (use_cv_mat) {
            // feed bgr24 image mat, with api-Write(cv::Mat)
            ret_code = memHandler->Write(&bgr_frame, pts++);
          } else {
            // feed rgb24 image data, with api-Write(unsigned char* data, int size, int w, int h, cnstream::CNDataFormat)
            cv::Mat rgb_frame(bgr_frame.rows, bgr_frame.cols, CV_8UC3);
            cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);

            ret_code = memHandler->Write(rgb_frame.data, rgb_frame.cols * rgb_frame.rows * 3, pts++, rgb_frame.cols,
                                        rgb_frame.rows, cnstream::CNDataFormat::CN_PIXEL_FORMAT_RGB24);
          }

          if (-2 == ret_code) {
            LOGW(DEMO) << "write image failed(invalid data).";
          }
        }
      }
    }
    memHandler->Write(nullptr, 0);
    return 0;
  }));
  return 0;
}

int AddSourceForDecompressedImage(cnstream::DataSource *source, const std::string &stream_id,
                                  const std::string &filename, const bool &loop, const bool &use_cv_mat) {
  // The following code is only for image input. For video input, you could use OpenCV VideoCapture.
  int index = filename.find_last_of("/");
  std::string dir_path = filename.substr(0, index);
  std::list<std::string> files = GetFileNameFromDir(dir_path, "*.jpg");
  if (files.empty()) {
    LOGE(DEMO) << "there is no jpg files";
    return -1;
  }

  auto handler = cnstream::RawImgMemHandler::Create(source, stream_id);
  if (source->AddSource(handler)) {
    LOGE(DEMO) << "failed to add  " << stream_id;
    return -1;
  }

  // Start another thread to read data from files to cv mat and feed data to pipeline.
  gFeedMemFutures.emplace_back(std::async(std::launch::async, [files, handler, loop, use_cv_mat]() {
    auto memHandler = std::dynamic_pointer_cast<cnstream::RawImgMemHandler>(handler);
    auto iter = files.begin();
    int ret_code = -1;
    uint64_t pts = 0;

    while (thread_running.load() && iter != files.end()) {
      cv::Mat bgr_frame = cv::imread(*iter);
      if (!bgr_frame.empty()) {
        if (use_cv_mat) {
          // feed bgr24 image mat, with api-Write(cv::Mat)
          ret_code = memHandler->Write(&bgr_frame, pts++);
        } else {
          // feed rgb24 image data, with api-Write(unsigned char* data, int size, int w, int h, cnstream::CNDataFormat)
          cv::Mat rgb_frame(bgr_frame.rows, bgr_frame.cols, CV_8UC3);
          cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);

          ret_code = memHandler->Write(rgb_frame.data, rgb_frame.cols * rgb_frame.rows * 3, pts++, rgb_frame.cols,
                                       rgb_frame.rows, cnstream::CNDataFormat::CN_PIXEL_FORMAT_RGB24);
        }

        if (-2 == ret_code) {
          LOGW(DEMO) << "write image failed(invalid data).";
        }
      }

      if (++iter == files.end() && loop) {
        iter = files.begin();
      }
    }
    memHandler->Write(nullptr, 0);
    return 0;
  }));
  return 0;
}

int AddSourceForFile(cnstream::DataSource *source, const std::string &stream_id, const std::string &filename,
                     const int &frame_rate, const bool &loop,
                     const cnstream::MaximumVideoResolution& maximum_resolution) {
  auto handler = cnstream::FileHandler::Create(source, stream_id, filename, frame_rate, loop, maximum_resolution);
  return source->AddSource(handler);
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  cnstream::InitCNStreamLogging(nullptr);
#if 0
  UserLogSink log_listener;
  cnstream::AddLogSink(&log_listener);
#endif

  LOGI(DEMO) << "CNSTREAM VERSION:" << cnstream::VersionString();

  /*
    flags to variables
  */
  std::list<std::string> video_urls;
  if (FLAGS_data_name != "") {
    video_urls = {FLAGS_data_name};
  } else {
    video_urls = ::ReadFileList(FLAGS_data_path);
  }

  std::string source_name = "source";  // source module name, which is defined in pipeline json config

  /*
    build pipeline
  */
  cnstream::Pipeline pipeline("MyPipeline");

  if (!pipeline.BuildPipelineByJSONFile(FLAGS_config_fname)) {
    LOGE(DEMO) << "Build pipeline failed.";
    return EXIT_FAILURE;
  }

  /*
    message observer
   */
  MsgObserver msg_observer(&pipeline, source_name);
  pipeline.SetStreamMsgObserver(reinterpret_cast<cnstream::StreamMsgObserver *>(&msg_observer));

  /*
    find data source
   */
  cnstream::DataSource *source = dynamic_cast<cnstream::DataSource *>(pipeline.GetModule(source_name));
  if (nullptr == source) {
    LOGE(DEMO) << "DataSource module not found.";
    return EXIT_FAILURE;
  }

  /*
    start pipeline
  */
  if (!pipeline.Start()) {
    LOGE(DEMO) << "Pipeline start failed.";
    return EXIT_FAILURE;
  }

  /*
    start print performance informations
   */
  std::future<void> perf_print_th_ret;
  int trace_data_file_cnt = 0;
  if (pipeline.IsProfilingEnabled()) {
    perf_print_th_ret = std::async(std::launch::async, [&pipeline, &trace_data_file_cnt] {
      cnstream::Time last_time = cnstream::Clock::now();
      int trace_data_dump_times = 0;
      cnstream::TraceSerializeHelper trace_dumper;
      while (!gstop_perf_print) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        ::PrintPipelinePerformance("Whole", pipeline.GetProfiler()->GetProfile());
        if (pipeline.IsTracingEnabled()) {
          cnstream::Duration duration(2000);
          ::PrintPipelinePerformance("Last two seconds",
                                     pipeline.GetProfiler()->GetProfileBefore(cnstream::Clock::now(), duration));
          if (!FLAGS_trace_data_dir.empty()) {
            cnstream::Time now_time = cnstream::Clock::now();
            trace_dumper.Serialize(pipeline.GetTracer()->GetTrace(last_time, now_time));
            last_time = now_time;
            if (++trace_data_dump_times == 10) {
              trace_dumper.ToFile(FLAGS_trace_data_dir + "/cnstream_trace_data_" +
                                  std::to_string(trace_data_file_cnt++));
              trace_dumper.Reset();
              trace_data_dump_times = 0;
            }
          }
        }
      }
      if (pipeline.IsTracingEnabled() && !FLAGS_trace_data_dir.empty() && trace_data_dump_times) {
        trace_dumper.ToFile(FLAGS_trace_data_dir + "/cnstream_trace_data_" + std::to_string(trace_data_file_cnt++));
        trace_dumper.Reset();
      }
    });
  }

  /*
    add stream sources...
  */
  std::vector<std::string> filename_vecs;
  cnstream::MaximumVideoResolution maximum_video_resolution;
  maximum_video_resolution.maximum_width = FLAGS_maximum_video_width;
  maximum_video_resolution.maximum_height = FLAGS_maximum_video_height;
  maximum_video_resolution.enable_variable_resolutions =
      FLAGS_maximum_video_width != -1 && FLAGS_maximum_video_height != -1;
  int streams = static_cast<int>(video_urls.size());
  auto url_iter = video_urls.begin();
  for (int i = 0; i < streams; i++, url_iter++) {
    const std::string &filename = *url_iter;
    std::string stream_id = "stream_" + std::to_string(i);

    int ret = 0;
    std::unique_lock<std::mutex> lk(gmutex_for_add_source);
    if (nullptr != source) {
      if (filename.find("rtsp://") != std::string::npos) {
        ret = AddSourceForRtspStream(source, stream_id, filename, maximum_video_resolution);
      } else if (filename.find("udp://") != std::string::npos) {
        ret = AddSourceForUdpStream(source, stream_id, filename, maximum_video_resolution);
      } else if (filename.find("udp2://") != std::string::npos) {
        ret = AddSourceForUdpDecompressedImage(source, stream_id, filename, FLAGS_loop, FLAGS_use_cv_mat);
      } else if (filename.find("udp3://") != std::string::npos) {
        ret = AddSourceForImageInUdp(source, stream_id, filename, FLAGS_loop);
      }else if (filename.find("/dev/video") != std::string::npos) {  // only support linux
        ret = AddSourceForUsbCam(source, stream_id, filename, FLAGS_src_frame_rate,
            FLAGS_loop, maximum_video_resolution);
      } else if (filename.find(".jpg") != std::string::npos && FLAGS_jpeg_from_mem) {
        ret = AddSourceForImageInMem(source, stream_id, filename, FLAGS_loop, filename_vecs);
      } else if (filename.find(".jpg") != std::string::npos && FLAGS_raw_img_input) {
        ret = AddSourceForDecompressedImage(source, stream_id, filename, FLAGS_loop, FLAGS_use_cv_mat);
      } else if (filename.find(".h264") != std::string::npos) {
        ret = AddSourceForVideoInMem(source, stream_id, filename, FLAGS_loop, maximum_video_resolution);
      } else {
        ret = AddSourceForFile(source, stream_id, filename, FLAGS_src_frame_rate, FLAGS_loop, maximum_video_resolution);
      }
    }

    if (ret == 0) {
      msg_observer.IncreaseStream(stream_id);
    }
  }

#ifdef HAVE_DISPLAY
  auto quit_callback = [&pipeline, &source, &msg_observer]() {
    // stop feed-data threads before remove-sources...
    thread_running.store(false);
    for (unsigned int i = 0; i < gFeedMemFutures.size(); i++) {
      gFeedMemFutures[i].wait();
    }
    msg_observer.WaitForStop();
  };

  gdisplayer = dynamic_cast<cnstream::Displayer *>(pipeline.GetModule("displayer"));

  if (gdisplayer && gdisplayer->Show()) {
    gdisplayer->GUILoop(quit_callback);
#else
  if (false) {
#endif
  } else {
    /*
     * close pipeline
     */
    if (FLAGS_loop) {
      // stop by hand or by FLAGS_wait_time
      if (FLAGS_wait_time) {
        std::this_thread::sleep_for(std::chrono::seconds(FLAGS_wait_time));
        LOGI(DEMO) << "run out time and quit...";
      } else {
        getchar();
        LOGI(DEMO) << "receive a character from stdin and quit...";
      }

      thread_running.store(false);
      if (nullptr != source) {
        source->RemoveSources();
      }
      for (unsigned int i = 0; i < gFeedMemFutures.size(); i++) {
        gFeedMemFutures[i].wait();
      }

      msg_observer.WaitForStop();
    } else {
      // stop automatically
      msg_observer.WaitForStop();
      thread_running.store(false);
      for (unsigned int i = 0; i < gFeedMemFutures.size(); i++) {
        gFeedMemFutures[i].wait();
      }
    }
  }
  gFeedMemFutures.clear();

  cnstream::ShutdownCNStreamLogging();
  if (pipeline.IsProfilingEnabled()) {
    gstop_perf_print = true;
    perf_print_th_ret.get();
    ::PrintPipelinePerformance("Whole", pipeline.GetProfiler()->GetProfile());
  }

  if (pipeline.IsTracingEnabled() && !FLAGS_trace_data_dir.empty()) {
    LOGI(DEMO) << "Wait for trace data merge ...";
    cnstream::TraceSerializeHelper helper;
    for (int file_index = 0; file_index < trace_data_file_cnt; ++file_index) {
      std::string filename = FLAGS_trace_data_dir + "/cnstream_trace_data_" + std::to_string(file_index);
      cnstream::TraceSerializeHelper t;
      cnstream::TraceSerializeHelper::DeserializeFromJSONFile(filename, &t);
      helper.Merge(t);
      remove(filename.c_str());
    }
    if (!helper.ToFile(FLAGS_trace_data_dir + "/cnstream_trace_data.json")) {
      LOGE(DEMO) << "Dump trace data failed.";
    }
  }
  return EXIT_SUCCESS;
}
