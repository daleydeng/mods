#ifndef MODS_DESCRIPTORS_PARAMETERS_HPP
#define MODS_DESCRIPTORS_PARAMETERS_HPP

#include "structures.hpp"
#include "siftdesc.h"
#include "pixelsdesc.hpp"

namespace mods {

struct DominantOrientationParams {

  int maxAngles;
  float threshold;
  bool addUpRight;
  bool halfSIFTMode;
  PatchExtractionParams PEParam;
  DominantOrientationParams() {
    maxAngles = -1;
    threshold = 0.8;
    addUpRight = false;
    halfSIFTMode = false;
  }
};

struct DescriptorsParameters {
  SIFTDescriptorParams SIFTParam;
  SIFTDescriptorParams RootSIFTParam;
  SIFTDescriptorParams HalfSIFTParam;
  SIFTDescriptorParams HalfRootSIFTParam;
  PIXELSDescriptorParams PixelsParam;

};

} //namespace mods
#endif // DESCRIPTORS_PARAMETERS_HPP
