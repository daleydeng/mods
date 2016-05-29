#ifndef SCALESPACEDETECTOR_HPP
#define SCALESPACEDETECTOR_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "common.hh"
#include "scale_space_detector.hh"
#include <iterator>
#include <iostream>

using cv::Mat;
using std::vector;
using std::min;
using std::max;

namespace mods {

bool responseCompare(AffineKeypoint k1,AffineKeypoint k2);
bool responseCompareInvOrder(AffineKeypoint k1,AffineKeypoint k2);

class KeypointCallback
{
public:
  virtual void onKeypointDetected(const Mat &blur, AffineKeypoint &key) = 0;
};

struct ScaleSpaceDetector
{
  enum
  {
    HESSIAN_DARK   = 0,
    HESSIAN_BRIGHT = 1,
    HESSIAN_SADDLE = 2,
    DOG_DARK   = 10,
    DOG_BRIGHT = 11,
    HARRIS_DARK   = 30,
    HARRIS_BRIGHT = 31,
    CAFFE_GRAD = 40
  };
public:
  KeypointCallback *keypointCallback;
  PyramidParams Pyrpar;
  ScalePyramid scale_pyramid;
  ScaleSpaceDetector(const PyramidParams &Pyrpar) :
    edgeScoreThreshold((Pyrpar.edgeEigenValueRatio + 1.0f)*(Pyrpar.edgeEigenValueRatio + 1.0f)/Pyrpar.edgeEigenValueRatio),
    finalThreshold(Pyrpar.threshold),
    positiveThreshold(0.8 * finalThreshold),
    negativeThreshold(-positiveThreshold)
  {
    extrema_points = 0;
    localized_points = 0;
    this->Pyrpar = Pyrpar;
    if (Pyrpar.DetectorType == DET_HESSIAN)
      finalThreshold = Pyrpar.threshold*Pyrpar.threshold;

    if (Pyrpar.DetectorMode !=FIXED_TH)
      finalThreshold = positiveThreshold = negativeThreshold = effectiveThreshold = 0.0;
    else effectiveThreshold = Pyrpar.threshold;

    if (Pyrpar.DetectorType == DET_HESSIAN)
      effectiveThreshold = effectiveThreshold*effectiveThreshold;

    keypointCallback = 0;
  }
  void setKeypointCallback(KeypointCallback *callback)
  {
    keypointCallback = callback;
  }
  void detectPyramidKeypoints(const Mat &image);
  int extrema_points;
  int localized_points;
  float effectiveThreshold;

protected:
  void detectOctaveKeypoints(const Mat &firstLevel, float pixelDistance, Mat &nextOctaveFirstLevel);
  void localizeKeypoint(int r, int c, float curScale, float pixelDistance);
  void findLevelKeypoints(float curScale, float pixelDistance);
  Mat Response(const Mat &inputImage, float norm);
  Mat iidogResponse(const Mat &inputImage, float norm);
  Mat dogResponse(const Mat &inputImage, float norm);
  Mat HessianResponse(const Mat &inputImage, float norm);
  Mat HarrisResponse(const Mat &inputImage, float norm);
  const Mat* originalImg;
  std::vector<Mat> levels;

private:
  // some constants derived from parameters
  const double edgeScoreThreshold;
  float finalThreshold;
  float positiveThreshold;
  float negativeThreshold;

  // temporary arrays used by protected functions
  Mat octaveMap;
  Mat prevBlur, blur;
  Mat low, cur, high;
};

struct AffineShape
{
public:
  AffineShape(const AffineShapeParams &par_) :
    par(par_),
    mask(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
    img(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
    imgHes(3, 3, CV_32FC1),
    fx(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
    fy(par.smmWindowSize, par.smmWindowSize, CV_32FC1)
  {
    computeGaussMask(mask);
  }

  // computes affine shape
  bool findAffineShape(const cv::Mat &blur, AffineKeypoint &key);

  // fills patch with affine normalized neighbourhood around point in the img, enlarged mrSize times, optionally a dominant orientation is estimated
  // the result is returned via NormalizedPatchCallback (called multiple times, once per each dominant orientation discovered)

public:
  AffineShapeParams par;

private:
  cv::Mat mask, img, imgHes, fx, fy;
};

struct AffineDetector : public ScaleSpaceDetector, AffineShape, KeypointCallback
{
  vector<AffineKeypoint> keys;

public:
  AffineDetector(const PyramidParams &par, const AffineShapeParams &ap) :
    ScaleSpaceDetector(par),
    AffineShape(ap)
  {
    this->setKeypointCallback(this);
  }

  void onKeypointDetected(const Mat &blur, AffineKeypoint &key)
  {
    if (findAffineShape(blur, key)) {
      keys.push_back(AffineKeypoint());
      AffineKeypoint &k = keys.back();
      key.x *= key.pyramid_scale;
      key.y *= key.pyramid_scale;
      key.s *= key.pyramid_scale;
      k = key;
    }
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
