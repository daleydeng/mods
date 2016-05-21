#ifndef MODS_SIFT_DESC_HPP
#define MODS_SIFT_DESC_HPP
#include "common.hpp"

namespace mods {
struct SIFTDescriptor: DescriptorFunctor
{
public:
  // top level interface
  SIFTDescriptor(const SIFTDescriptorParams &par) :
    mask(par.PEParam.patchSize, par.PEParam.patchSize, CV_32FC1),
    grad(par.PEParam.patchSize, par.PEParam.patchSize, CV_32FC1),
    ori(par.PEParam.patchSize, par.PEParam.patchSize, CV_32FC1)
  {
    this->par = par;
    if (par.useRootSIFT) type = DESC_ROOT_SIFT;
    else
      type = DESC_SIFT;
    vec.resize(par.spatialBins * par.spatialBins * par.orientationBins);
    computeCircularGaussMask(mask);
    precomputeBinsAndWeights();
  }

  void computeSiftDescriptor(cv::Mat &patch);
  void computeRootSiftDescriptor(cv::Mat &patch);
  void operator()(cv::Mat &patch, std::vector<float>& desc);

public:
  std::vector<double> vec;
  void SIFTnorm(std::vector<float> &in_vect);
  void RootSIFTnorm(std::vector<float> &in_vect);
  void SIFTnorm(std::vector<double> &in_vect);
  void RootSIFTnorm(std::vector<double> &in_vect);

private:
  // helper functions

  double normalize(std::vector<double>& vec1);
  float normalize(std::vector<float>& vec1);

  void sample(bool do_norm);
  void rootsample(bool do_norm);
  void samplePatch();
  void precomputeBinsAndWeights();

private:
  SIFTDescriptorParams par;
  cv::Mat mask, grad, ori;
  std::vector<int> precomp_bins;
  std::vector<double> precomp_weights;
  int *bin0, *bin1;
  double *w0, *w1;
};

} //namespace mods
#endif
