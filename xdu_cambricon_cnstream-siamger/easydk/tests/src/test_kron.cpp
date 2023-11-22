#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
// using namespace cv;
void kron(cv::InputArray src1, cv::InputArray src2,cv::OutputArray dst){
    CV_Assert(src1.type() == src2.type() && src1.type() == CV_32F);
    cv::Mat m1 = src1.getMat();
    cv::Mat m2 = src2.getMat();
    cv::Mat k = cv::Mat(m1.rows*m2.rows, m1.cols*m2.cols, m1.type());

    // if(m1.type() == CV_32S)
    float *data = (float *)m1.data;
    cv::Rect roi;
    for(int i = 0; i < m1.rows; i++){
        for(int j = 0; j < m1.cols; j++){
            roi.x = j*m2.cols;
            roi.y = i*m2.rows;
            roi.width = m2.cols;
            roi.height = m2.rows;
            m2.convertTo(k(roi), m1.type(), data[i*m1.cols + j]);
        }
    }
    k.copyTo(dst);
}

int main(int argc, char const *argv[])
{
    float a1[2][2] = {{1.,2.},{3.,4.}};
    float a2[3][3] = {{1.,3.,5.},{2.,4.,6.},{3.,6.,9.}};

    cv::Mat m1  = cv::Mat(2,2,CV_32F,a1);
    cv::Mat m2  = cv::Mat(3,3,CV_32F,a2);
    cv::Mat k;
    kron(m1,m2,k);
    cout<<m1<<endl;
    cout<<m2<<endl;

    cout<<k<<endl;
    cv::waitKey();
    return 0;
}
