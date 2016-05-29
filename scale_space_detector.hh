#ifndef SCALESPACEDETECTOR_HPP
#define SCALESPACEDETECTOR_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "common.hpp"
#include "pyramid.h"
#include "affine.h"
#include <iterator>
#include <iostream>

using cv::Mat;
using std::vector;
using std::min;
using std::max;

namespace mods {

struct AffineDetector : public ScaleSpaceDetector, AffineShape, KeypointCallback, AffineShapeCallback
{
  vector<AffineKeypoint> keys;

public:
  AffineDetector(const PyramidParams &par, const AffineShapeParams &ap) :
    ScaleSpaceDetector(par),
    AffineShape(ap)
  {
    this->setKeypointCallback(this);
    this->setAffineShapeCallback(this);
  }

  void onKeypointDetected(const Mat &blur, AffineKeypoint &key)
  {
    findAffineShape(blur, key);
  }

  void onAffineShapeFound(
      const Mat &blur, AffineKeypoint &key, int iters)
  {
    // convert shape into a up is up frame
    // rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);
    // now sample the patch
    keys.push_back(AffineKeypoint());
    AffineKeypoint &k = keys.back();
    key.x *= key.pyramid_scale;
    key.y *= key.pyramid_scale;
    key.s *= key.pyramid_scale;
    k = key;
  }

  void exportKeypoints(vector<AffineKeypoint>& out1);

  void exportScaleSpace(ScalePyramid& exp_scale_pyramid)
  {
    exp_scale_pyramid = scale_pyramid;
  }

private:
  void sortKeys()
  {
    std::sort (keys.begin(), keys.end(), responseCompareInvOrder);
  }
};

template<class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v)
{
  copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  return os;
}

int DetectAffineKeypoints(const cv::Mat &input, vector<AffineKeypoint> &out1, const ScaleSpaceDetectorParams &params, ScalePyramid &scale_pyramid, double tilt = 1.0, double zoom = 1.0);

} //namespace mods
#endif // SCALESPACEDETECTOR_HPP
