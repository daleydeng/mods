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
  const Mat image;
  vector<AffineKeypoint> keys;
  int g_numberOfPoints;
  int g_numberOfAffinePoints;
  int g_numberOfDescribedPoints;

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
    g_numberOfPoints++;
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

    g_numberOfDescribedPoints++;
    g_numberOfAffinePoints++;
  }

  void exportKeypoints(vector<AffineKeypoint>& out1)
  {
  //  std::cerr << "Hessian points detected " << g_numberOfPoints << std::endl;
  //  std::cerr << "AffineShapes points detected " << g_numberOfAffinePoints << std::endl;

    prepareKeysForExport();
    unsigned int keys_size = keys.size();
    out1.reserve(out1.size() + keys_size);
    for (size_t i=0; i < keys_size; i++)
      {
        AffineKeypoint &k = keys[i];
        AffineKeypoint tmpRegion;
        tmpRegion.x=k.x;
        tmpRegion.y=k.y;
        tmpRegion.a11=k.a11;
        tmpRegion.a12=k.a12;
        tmpRegion.a21=k.a21;
        tmpRegion.a22=k.a22;
        tmpRegion.s=k.s;
        tmpRegion.response = k.response;
        tmpRegion.sub_type = k.sub_type;
        out1.push_back(tmpRegion);
      };
  }
  void exportScaleSpace(ScalePyramid& exp_scale_pyramid)
  {
    exp_scale_pyramid = scale_pyramid;
  }

private:
  void sortKeys()
  {
    std::sort (keys.begin(), keys.end(), responseCompareInvOrder);
  }
  int prepareKeysForExport()
  {
    if (keys.size() <= 0) return 0;
    if (Pyrpar.DetectorMode == FIXED_TH)
      {
        effectiveThreshold = Pyrpar.threshold;
      }
    else
      {
        sortKeys();
    //    std::cerr << "Keys sorted" << std::endl;
        double maxResponse = fabs(keys[0].response);
        int regNumber = (int) keys.size();

        switch (Pyrpar.DetectorMode)
          {
          case RELATIVE_TH:
            {
              effectiveThreshold = maxResponse * Pyrpar.rel_threshold;
              AffineKeypoint tempKey = keys[0];
              tempKey.response = effectiveThreshold;
              std::vector<AffineKeypoint>::iterator low;
              low = std::lower_bound(keys.begin(), keys.end(), tempKey,responseCompareInvOrder);
              keys.resize(low - keys.begin());
              break;
            }
          case FIXED_REG_NUMBER:
            {
              int newRegNumber = Pyrpar.reg_number;
              if (par.doBaumberg)
                newRegNumber =(int) floor(3.0*(double)newRegNumber);

              if ((newRegNumber < regNumber) && (newRegNumber >=0))
                keys.resize(newRegNumber);

              break;
            }
          case RELATIVE_REG_NUMBER:
            {
              int newRegNumber = (int)floor(Pyrpar.rel_reg_number * (double)keys.size());
              keys.resize(newRegNumber);
              break;
            }
          case NOT_LESS_THAN_REGIONS:
            {
              AffineKeypoint tempKey = keys[0];
              tempKey.response = Pyrpar.threshold;
              std::vector<AffineKeypoint>::iterator low;
              low = std::lower_bound(keys.begin(), keys.end(), tempKey,responseCompareInvOrder);

              int RegsFixThNumber = std::distance( keys.begin(), low);

              if (RegsFixThNumber < Pyrpar.reg_number)
                keys.resize(min(Pyrpar.reg_number,regNumber)); //use reg_number
              else
                keys.resize(min(RegsFixThNumber,regNumber)); //use threshold
              //enough keys, use fixed threshold
              break;
            }

          default:
            break;
          }
        effectiveThreshold = keys[keys.size() - 1].response;

      }

    if ((Pyrpar.DetectorMode == FIXED_REG_NUMBER) && ((int)keys.size() > Pyrpar.reg_number))
      keys.resize(Pyrpar.reg_number);
    return keys.size();
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
