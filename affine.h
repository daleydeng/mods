/*
 * Copyright (C) 2008-12 Michal Perdoch, 2013-15 Dmytro Mishkin
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#ifndef __AFFINE_H__
#define __AFFINE_H__
#undef __STRICT_ANSI__
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "common.hpp"

namespace mods {

struct AffineShapeCallback
{
  virtual void onAffineShapeFound(
      const cv::Mat &blur,     // corresponding scale level
      AffineKeypoint &key, int iters) = 0;
};

struct AffineShape
{
public:
  AffineShape(const AffineShapeParams &par) :
    patch(par.patchSize, par.patchSize, CV_32FC1),
    gmag(par.patchSize, par.patchSize, CV_32FC1),
    gori(par.patchSize, par.patchSize, CV_32FC1),
    orimask(par.patchSize, par.patchSize, CV_32FC1),
    mask(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
    img(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
    imgHes(3, 3, CV_32FC1),
    fx(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
    fy(par.smmWindowSize, par.smmWindowSize, CV_32FC1)
  {
    this->par = par;
    computeGaussMask(mask);
    computeCircularGaussMask(orimask, par.patchSize/3.0f);
    affineShapeCallback = 0;
    fx = cv::Scalar(0);
    fy = cv::Scalar(0);
  }

  // computes affine shape
  bool findAffineShape(const cv::Mat &blur, AffineKeypoint &key);

  // fills patch with affine normalized neighbourhood around point in the img, enlarged mrSize times, optionally a dominant orientation is estimated
  // the result is returned via NormalizedPatchCallback (called multiple times, once per each dominant orientation discovered)
  void setAffineShapeCallback(AffineShapeCallback *callback)
  {
    affineShapeCallback = callback;
  }

public:
  cv::Mat patch;
  AffineShapeParams par;
protected:

//  void estimateDominantAngles(const cv::Mat &img, std::vector<float> &angles);
  AffineShapeCallback *affineShapeCallback;

private:
  cv::Mat gmag, gori, orimask;
  std::vector<unsigned char> workspace;
  cv::Mat mask, img, imgHes, fx, fy;
};

} //namespace mods

#endif // __AFFINE_H__
