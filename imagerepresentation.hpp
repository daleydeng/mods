#ifndef MODS_IMAGEREPRESENTATION_H
#define MODS_IMAGEREPRESENTATION_H

#include <vector>
#include <string>
#include <map>

#include "structures.hpp"
#include "detectors_parameters.hpp"
#include "descriptors_parameters.hpp"

namespace mods {

class ImageRepresentation
{
public:
  ImageRepresentation();
  ImageRepresentation(cv::Mat _in_img, std::string _name);
  ~ImageRepresentation();
  std::vector< std::map<std::string, SynthImage> > SynthViews;

  descriptor_type GetDescriptorType(std::string desc_name);
  detector_type GetDetectorType(std::string det_name);
  TimeLog GetTimeSpent();
  int GetDescriptorDimension(std::string desc_name);
  int GetRegionsNumber(std::string det_name = "All");
  int GetDescriptorsNumber(std::string desc_name = "All", std::string det_name = "All");
  cv::Mat GetDescriptorsMatByDetDesc(const std::string desc_name,const std::string det_name = "All");
  cv::Mat GetDescriptorsMatByDetDesc(std::vector<cv::Point2f> &coordinates,
                                     const std::string desc_name,const std::string det_name = "All");
  AffineRegionVector GetAffineRegionVector(std::string desc_name, std::string det_name, std::vector<int> idxs);
  AffineRegionVector GetAffineRegionVector(std::string desc_name, std::string det_name);
  AffineRegion GetAffineRegion(std::string desc_name, std::string det_name, int idx);
  void SynthDetectDescribeKeypoints (IterationViewsynthesisParam &synth_par,
                                     DetectorsParameters &det_par,
                                     DescriptorsParameters &desc_par,
                                     DominantOrientationParams &dom_ori_par);
   cv::Mat OriginalImg;
  void SaveRegions(std::string fname, int mode);
  void LoadRegions(std::string fname);

protected:
  TimeLog TimeSpent;
  void AddRegions(AffineRegionVector& RegionsToAdd,std::string det_name, std::string desc_name);
  void AddRegions(AffineRegionVectorMap& RegionsMapToAdd,std::string det_name);
  void AddRegionsToList(AffineRegionVector &kp_list, AffineRegionVector &new_kps);

  std::map<std::string, AffineRegionVectorMap> RegionVectorMap;
  std::string Name;
};

} //namespace mods
#endif // IMAGEREPRESENTATION_H
