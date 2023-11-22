#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <vector>
#include <list>
#include <dirent.h>
#include "udp-piece.hpp"
#include "circular_buffer.hpp"
#include <opencv2/opencv.hpp>

#ifdef DUMP
#include <fstream>
#endif

#define DEST_PORT 6868
#define DSET_IP_ADDRESS  "127.0.0.1"
#define LOOP true

std::list<std::string> GetFileNameFromDir(const std::string &dir, const char *filter) {
  std::list<std::string> files;
#if defined(_WIN32) || defined(_WIN64)
  int64_t hFile = 0;
  struct _finddata_t fileinfo;
  std::string path;
  if ((hFile = _findfirst(path.assign(dir).append("/" + std::string(filter)).c_str(), &fileinfo)) != -1) {
    do {
      if (!(fileinfo.attrib & _A_SUBDIR)) {  // not directory
        std::string file_path = dir + "/" + fileinfo.name;
        files.push_back(file_path);
      }
    } while (_findnext(hFile, &fileinfo) == 0);
    _findclose(hFile);
  }
#elif defined(__linux) || defined(__unix)
  DIR *pDir = nullptr;
  struct dirent *pEntry;
  pDir = opendir(dir.c_str());
  if (pDir != nullptr) {
    while ((pEntry = readdir(pDir)) != nullptr) {
      if (strcmp(pEntry->d_name, ".") == 0 || strcmp(pEntry->d_name, "..") == 0
          || strstr(pEntry->d_name, strstr(filter, "*") + 1) == nullptr || pEntry->d_type != DT_REG) {  // regular file
        continue;
      }
      std::string file_path = dir + "/" + pEntry->d_name;
      files.push_back(file_path);
    }
    closedir(pDir);
  }
#endif
  return files;
}

int main()
{
  /* socket文件描述符 */
  int sock_fd;

  /* 建立udp socket */
  sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if(sock_fd < 0)
  {
    perror("socket");
    exit(1);
  }

  /* 设置address */
  struct sockaddr_in addr_serv;
  int len;
  memset(&addr_serv, 0, sizeof(addr_serv));
  addr_serv.sin_family = AF_INET;
  addr_serv.sin_addr.s_addr = inet_addr(DSET_IP_ADDRESS);
  addr_serv.sin_port = htons(DEST_PORT);
  len = sizeof(addr_serv);


  // int send_num;
  // int recv_num;
  // char recv_buf[20];
  std::cout << "Enter the directory where the images in. For example, /workspace/volume/private/CNStream/data/images" << std::endl;
  std::string image_dir = "/workspace/volume/private/CNStream/samples/cns_launcher/udp_test";
  // std::cin >> image_dir;
  std::list<std::string> files = GetFileNameFromDir(image_dir, "*.jpg");
  if (files.empty()) {
    std::cout << "Error: there is no jpg files in " << image_dir << std::endl;
    return -1;
  }
  
  // udp piece
  udp_piece_t *udp_piece = udp_piece_init(64*1024*1024);
  int pieces;

  auto iter = files.begin();
  while(iter != files.end())
  {
    cv::Mat src_img = cv::imread(*iter);
    std::vector<unsigned char> encode_img;
    cv::imencode(".jpg", src_img, encode_img);
    int datalen = encode_img.size();
    unsigned char *send_buf = new unsigned char[datalen];
    for(int i=0;i<datalen;i++)
    {
      send_buf[i] = encode_img[i];
    }
    printf("send_buf send: %s\n", send_buf);
    printf("datalen : %d\n", datalen);

    pieces = udp_piece_cut(udp_piece, send_buf, datalen);
    for (int i = 0; i < pieces; i++)
    {
      // printf("recv_pieces = %d\t",udp_piece->recv_pieces);
      // printf("total_size = %d\t",udp_piece->total_size);
      // printf("total_pieces = %d\t",udp_piece->total_pieces);
      // printf("left = %d\t",udp_piece->left);
      // printf("piece_size = %d\t",udp_piece->piece_size);
      // printf("recv_len = %d\t\n",udp_piece->recv_len);
      uint8_t *buf;
			int size;
			buf = udp_piece_get(udp_piece, i, &size);
#ifdef DUMP
      std::string dump_file = "dump_" + std::to_string(i) + ".bin";
      std::ofstream outFile(dump_file, std::ios::out);
      outFile.write(reinterpret_cast<char*>(buf),size);
      outFile.close();
#endif
      // printf("向服务器发送分片[%d]长度：%d, buf = %p\n", i, size, buf);  
      sendto(sock_fd, buf, size, 0, (struct sockaddr *)&addr_serv, len); 

      // printf("向服务器发送分片[%d]长度：%d, buf = %p\n", i*2, 12, buf);  
			// sendto(sock_fd, buf, 12, 0, (struct sockaddr *)&addr_serv, len); 
			// printf("向服务器发送分片[%d]长度：%d, buf = %p\n", i*2+1, size - 12, buf);
      // sendto(sock_fd, buf + 12, size - 12, 0, (struct sockaddr *)&addr_serv, len); 
			// if(send_len != size)
			// {
			// 	printf("An error occurred in this call of  sendto, need send len = %d, but actual len = %d\n",
			// 		size, send_len);
			// }
      // recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *)&addr_serv, (socklen_t *)&len);

      // if(recv_num < 0)
      // {
      //   perror("recvfrom error:");
      //   exit(1);
      // }

      // recv_buf[recv_num] = '\0';
      // printf("client receive %d bytes: %s\n", recv_num, recv_buf);
    }
    sleep(0.03);
    
    // send_num = sendto(sock_fd, send_buf, datalen, 0, (struct sockaddr *)&addr_serv, len);

    // if(send_num < 0)
    // {
    //   perror("sendto error:");
    //   exit(1);
    // }

    

    ++iter;
    if (iter == files.end() && LOOP) {
      iter = files.begin();
    }
    // delete []send_buf;
  }

  
  udp_piece_deinit(udp_piece);
  close(sock_fd);

  return 0;
}