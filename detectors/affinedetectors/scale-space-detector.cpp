/*------------------------------------------------------*/
/* Copyright 2013-2015, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#include "scale-space-detector.hpp"
using cv::Mat;
using namespace std;

namespace mods {

int DetectAffineKeypoints(cv::Mat &input, vector<AffineKeypoint> &out1,
                          const ScaleSpaceDetectorParams &params,
                          ScalePyramid &scale_pyramid,
                          double tilt, double zoom)
{
  PyramidParams p1 = params.PyramidPars;
  AffineShapeParams ap = params.AffineShapePars;
  if ((tilt > 2.0) || (zoom < 0.5))
    p1.reg_number = (int)floor(zoom*(double)p1.reg_number/tilt);

  // Detect keypoints on normal image
  if (params.PyramidPars.doOnNormal)
    {
      p1.doOnWLD = 0;
      p1.doOnNormal = params.PyramidPars.doOnNormal;
      AffineDetector detector(input, p1, ap);
      detector.detectPyramidKeypoints(input);
      detector.exportKeypoints(out1);
      detector.exportScaleSpace(scale_pyramid);
    }

  return out1.size();
}

}
