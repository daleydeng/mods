#ifndef MODS_PIXEL_DESC_HPP
#define MODS_PIXEL_DESC_HPP

#include "common.hpp"

namespace mods {
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
