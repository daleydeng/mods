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
  }

  void detectPyramidKeypoints(const Mat &image, vector<AffineKeypoint> *keys);
  int extrema_points;
  int localized_points;
  float effectiveThreshold;
  std::vector<Mat> levels;
  std::map<int, int> level_idx_map;

protected:
  void detectOctaveKeypoints(const Mat &firstLevel, float pixelDistance, Mat &nextOctaveFirstLevel, vector<AffineKeypoint> *keys);
  void localizeKeypoint(int r, int c, float curScale, float pixelDistance, vector<AffineKeypoint> *keys);
  void findLevelKeypoints(float curScale, float pixelDistance, vector<AffineKeypoint> *keys);
  Mat Response(const Mat &inputImage, float norm);
  Mat iidogResponse(const Mat &inputImage, float norm);
  Mat dogResponse(const Mat &inputImage, float norm);
  Mat HessianResponse(const Mat &inputImage, float norm);
  Mat HarrisResponse(const Mat &inputImage, float norm);
  const Mat* originalImg;

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

public:
  AffineShapeParams par;

private:
  cv::Mat mask, img, imgHes, fx, fy;
};

struct AffineDetector : public ScaleSpaceDetector, AffineShape
{
public:
  AffineDetector(const PyramidParams &par, const AffineShapeParams &ap) :
    ScaleSpaceDetector(par), AffineShape(ap) {}

  void find_affine_shapes(vector<AffineKeypoint> &keys, vector<AffineKeypoint>  *out) {
    out->clear();
    for (auto &key: keys) {
      if (findAffineShape(levels[key.octave_number], key)) {
        out->push_back(AffineKeypoint());
        auto &k = out->back();
        k = key;
        k.x *= k.pyramid_scale;
        k.y *= k.pyramid_scale;
        k.s *= k.pyramid_scale;
        out->push_back(k);
      }
    }
  }

  void exportKeypoints(vector<AffineKeypoint>& keys, vector<AffineKeypoint> *out);

private:
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
