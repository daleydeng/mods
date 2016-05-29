/*------------------------------------------------------*/
/* Copyright 2013-2015, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#include "scale_space_detector.hh"
using cv::Mat;
using namespace std;

namespace mods {

static inline int keys_nr_by_threshold(vector<AffineKeypoint> &keys, float th) {
  AffineKeypoint tempKey = keys[0];
  tempKey.response = th;
  std::vector<AffineKeypoint>::iterator low;
  low = std::lower_bound(keys.begin(), keys.end(), tempKey, responseCompareInvOrder);
  return low - keys.begin();
}

void AffineDetector::exportKeypoints(vector<AffineKeypoint>& out1) {
  if (keys.size() <= 0)
    return;

  int key_nr = keys.size(), new_nr = 0;

  if (Pyrpar.DetectorMode == FIXED_TH) {
    effectiveThreshold = Pyrpar.threshold;
    goto out;
  }

  sortKeys();
  //    std::cerr << "Keys sorted" << std::endl;

  switch (Pyrpar.DetectorMode)
  {
    case RELATIVE_TH:
      new_nr = keys_nr_by_threshold(keys, fabs(keys[0].response) * Pyrpar.rel_threshold);
      break;

    case FIXED_REG_NUMBER:
        new_nr = floor(Pyrpar.reg_number);
        break;

    case RELATIVE_REG_NUMBER:
      new_nr = floor(Pyrpar.rel_reg_number * key_nr);
      break;

    case NOT_LESS_THAN_REGIONS:
        new_nr = keys_nr_by_threshold(keys, Pyrpar.threshold);
        new_nr = max(new_nr, Pyrpar.reg_number);
        break;

    default:
      break;
  }
  effectiveThreshold = keys[keys.size() - 1].response;

  if (new_nr < key_nr and new_nr > 0)
    keys.resize(new_nr);

out:
  out1.reserve(out1.size() + keys.size());
  out1.insert(out1.end(), keys.begin(), keys.end());
}

int DetectAffineKeypoints(const cv::Mat &input, vector<AffineKeypoint> &out1,
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
  }

  AffineDetector detector(p1, ap);
  detector.detectPyramidKeypoints(input);
  detector.exportKeypoints(out1);
  detector.exportScaleSpace(scale_pyramid);
  for (auto &i: out1)
    rectifyTransformation(i);
  return out1.size();
}

}
