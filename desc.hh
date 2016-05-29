#ifndef MODS_DESC_HPP
#define MODS_DESC_HPP

#include "common.hh"

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

template <typename Container>
struct compare_indirect_index_ascend  {
  const Container& container;
  explicit compare_indirect_index_ascend(const Container& container):
    container(container) {
  }
  bool operator()(size_t lindex, size_t rindex) const {
    return container[lindex] < container[rindex];
  }
};

template <typename Container>
struct compare_indirect_index_descend {
  const Container& container;
  explicit compare_indirect_index_descend(const Container& container):
    container(container)  {
  }
  bool operator()(size_t lindex, size_t rindex) const {
    return container[lindex] > container[rindex];
  }
};

template <typename Dtype>
void sort_idxs(const std::vector<Dtype> &v,
               std::vector<size_t> &idx, const int ascend) {  // NOLINT(runtime/references)
  if (ascend) {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_ascend <std::vector<Dtype> > (v));
    } else {
      std::sort(idx.begin(), idx.end(),
                compare_indirect_index_descend <std::vector<Dtype> > (v));
    }
  return;
}

struct PIXELSDescriptor: DescriptorFunctor
{
public:
  PIXELSDescriptor(const PIXELSDescriptorParams &par)
  {
    this->par = par;
    type = DESC_PIXELS;
  }
  void operator()(cv::Mat &patch, std::vector<float>& desc)
  {
    const int desc_size = patch.cols * patch.rows * patch.channels();
    desc.resize(desc_size);
    float *patchPtr = patch.ptr<float>(0);

    if (par.normType == "L2"){
        double norm2=0;
        for (int jj = 0; jj < desc_size; jj++) {
            norm2 +=patchPtr[jj];
          }
        norm2 = 1.0/sqrt(norm2);
        for (int jj = 0; jj < desc_size; jj++) {
            desc[jj] = norm2 * patchPtr[jj];
          }
        return ;
      }

    if (par.normType == "LUCID"){
        std::vector<size_t> idxs(desc_size,0);
        for (int ii=0; ii < desc_size; ++ii ){
            idxs[ii] = ii;
            desc[ii] = patchPtr[ii];
          }
        sort_idxs(desc,idxs,1);
        for (int ii=0; ii < desc_size; ++ii ){
            desc[ii] = (float) idxs[ii];
          }
        return ;
      }

    if (par.normType == "None"){
        for (int jj = 0; jj < desc_size; jj++) {
            desc[jj] = patchPtr[jj];
            return;
          }
      }
  }
private:
  PIXELSDescriptorParams par;
};

} //namespace mods
#endif
