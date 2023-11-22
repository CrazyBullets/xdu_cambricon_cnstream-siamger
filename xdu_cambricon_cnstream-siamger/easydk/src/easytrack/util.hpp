#include "opencv2/opencv.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core.hpp"

#include <utility>
#include <vector>

#include "easytrack/easy_track.h"
#include "matrix.h"
#include "cxxutil/log.h"

using namespace cv;


edk::Matrix mat2edk(const Mat& in);

void kron(InputArray src1, InputArray src2,OutputArray dst);

double _findTransformECC(InputArray templateImage,
                            InputArray inputImage,
                            InputOutputArray warpMatrix,
                            int motionType,
                            TermCriteria criteria,
                            InputArray inputMask,
                            int gaussFiltSize);

double _findTransformECC( InputArray templateImage, InputArray inputImage,
                                      InputOutputArray warpMatrix, int motionType = MOTION_AFFINE,
                                      TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001),
                                      InputArray inputMask = noArray());