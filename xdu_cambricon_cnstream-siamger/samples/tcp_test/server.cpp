#include <iostream>
#include <fstream>
#include <sys/socket.h> //socket
#include <netinet/in.h> //struct sockaddr_in
#include <string.h>     //memset
#include <arpa/inet.h>  //htos
#include <unistd.h>     //close
#include <vector>


int readn(int fd, char* buf, int size)
{
    char* pt = buf;
    int count = size;
    while (count > 0)
    {
        int len = recv(fd, pt, count, 0);
        if (len == -1)
        {
            return -1;
        }
        else if (len == 0)
        {
            return size - count;
        }
        pt += len;
        count -= len;
    }
    return size;
}

/*
函数描述: 接收带数据头的数据包
函数参数:
    - cfd: 通信的文件描述符(套接字)
    - msg: 一级指针的地址，函数内部会给这个指针分配内存，用于存储待接收的数据，这块内存需要使用者释放
函数返回值: 函数调用成功返回接收的字节数, 发送失败返回-1
*/
int recvData(int cfd, char** msg)
{
    // 接收数据
    // 1. 读数据头
    int len = 0;
    readn(cfd, (char*)&len, 4);
    len = ntohl(len);
    // printf("数据块大小: %d\n", len);

    // 根据读出的长度分配内存，+1 -> 这个字节存储\0
    char *buf = (char*)malloc(len+1);
    int ret = readn(cfd, buf, len);
    if(ret != len)
    {
        close(cfd);
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    std::cout<<buf<<std::endl;
    std::ofstream file("test.txt",std::ios::app);
    file<<buf<<std::endl;
    file.close();
    *msg = buf;

    return ret;
}



int main()
{
    //创建套接字
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0)
	{
		perror("cockt");
		return 0;
	}
    //bind绑定固定的port、ip地址信息
    struct sockaddr_in my_addr;
    bzero(&my_addr, sizeof(my_addr));
    my_addr.sin_family = AF_INET;
    my_addr.sin_port = htons(8000);
    my_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    int ret = bind(sockfd, (struct sockaddr *)&my_addr, sizeof(my_addr));
    std::vector<char*> vecRecvData;
	if(ret == -1)
	{
		perror("bind");
		return 0;
	}
	//监听套接字 创捷连接队列
	ret = listen(sockfd, 10);
	if(ret == -1)
	{
		perror("listen");
		return 0;
	}
	//提取客户端的连接
    std::ofstream file("test.txt",std::ios::trunc);
    file.close();
	while(1)
	{
		//一次只能提取一个客户端
		struct sockaddr_in cli_addr;
		socklen_t cli_len = sizeof(cli_addr);
		int cfd = accept(sockfd, (struct sockaddr *)&cli_addr, &cli_len);
		if(cfd < 0 ) //提取失败
		{
			perror("accept\n");
			break;
		}
		else
		{
			char ip[16] = "";
			unsigned short port = 0;
			inet_ntop(AF_INET, &cli_addr.sin_addr.s_addr, ip, 16);
			port = ntohs(cli_addr.sin_port);
			//打印客户端的信息
			printf("客户端：%s %d connected\n", ip, port);
			while(1)
			{
				//获取客户端的请求

				char* buf[64];
                // memset(buf,NULL,sizeof(buf));
				int len = recvData(cfd, buf);
				if(len == 0) //客户端已经关闭
				{
					//关闭与客户端连接的套接字
                    printf("客户端已下线");
					close(cfd);
					break;
				}
                else if(len < 0){
                    perror("len < 0");
                    break;
                }
                else{
                    //应答客户端
                    // vecRecvData.push_back(buf);
                    // if(strlen(buf)>40) 
                    // printf("%s\n",vecRecvData.back());
                    if(vecRecvData.size() > 1000){
                        std::cout<<("p")<<std::endl;
                        break;
                    }
                    // send(cfd, buf, len, 0);
                }
				
			}
            if(vecRecvData.size() > 4000){
                break;
            }
		}	
	}
    std::cout<<"p"<<std::endl;
	//关闭监听套接字
	close(sockfd);
	return 0;
}
