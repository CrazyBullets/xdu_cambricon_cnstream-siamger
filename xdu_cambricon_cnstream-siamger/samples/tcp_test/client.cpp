#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

int main(int argc, char const *argv[])
{
	//创建tcp套接字 SOCK_STREAN
	int sockfd = socket(AF_INET, SOCK_STREAM, 0);
	//connect连接服务器（知道服务器地址信息）
	struct sockaddr_in ser_addr;
	bzero(&ser_addr, sizeof(ser_addr));
	ser_addr.sin_family = AF_INET;
	ser_addr.sin_port = htons(8000);
	ser_addr.sin_addr.s_addr = inet_addr("172.17.0.3");
	connect(sockfd, (struct sockaddr *)&ser_addr, sizeof(ser_addr));
	//客户端发送请求
    char buff[] = "0,-1,491,480,54,124,0.87793,0,-1,-1";
	send(sockfd, buff, strlen(buff), 0 );
	//客户端接收服务器的应答
	unsigned char buf[1500] = "";
	int len = recv(sockfd, buf, sizeof(buf), 0);
	printf("服务器的应答：%s\n", buf);
	//关闭套接字
	close(sockfd);
	return 0;
}
