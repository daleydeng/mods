#ifndef __EXTREMA_H__
#define __EXTREMA_H__
#undef __STRICT_ANSI__
#include "../../common.hpp"
#include <opencv2/core/core.hpp>

namespace mods {

int DetectMSERs(cv::Mat &input, std::vector<AffineKeypoint> &out1, const ExtremaParams &params, const double tilt = 1.0, const double zoom = 1.0);
//Entry point

int DetectMSERs(cv::Mat &input, std::vector<AffineKeypoint> &out1, const ExtremaParams &params, ScalePyramid &scale_pyramid, const double tilt = 1.0, const double zoom = 1.0);

}
#endif //__EXTREMA_H__
