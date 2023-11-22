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
#include <opencv2/opencv.hpp>

#define DEST_PORT 8000
#define DSET_IP_ADDRESS  "127.0.0.1"
#define FILE "/workspace/volume/private/CNStream/data/images/33.jpg"

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


  int send_num;
  int recv_num;
  char recv_buf[20];
  while(1)
  {
    std::string file;
    std::cout << "Enter the absolute path of image" << std::endl;
    std::cin >> file;
    cv::Mat src_img = cv::imread(file);
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

    send_num = sendto(sock_fd, send_buf, datalen, 0, (struct sockaddr *)&addr_serv, len);

    if(send_num < 0)
    {
      perror("sendto error:");
      exit(1);
    }

    recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *)&addr_serv, (socklen_t *)&len);

    if(recv_num < 0)
    {
      perror("recvfrom error:");
      exit(1);
    }

    recv_buf[recv_num] = '\0';
    printf("client receive %d bytes: %s\n", recv_num, recv_buf);
  }

  

  close(sock_fd);

  return 0;
}