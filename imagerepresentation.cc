#include "desc.hh"
#include "imagerepresentation.hh"
#include "synth_detection.hh"
#include "mser/extrema/extrema.h"
#include "scale_space_detector.hh"
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <fstream>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#define VERBOSE 1
using std::endl;

namespace mods {

void saveKP(AffineKeypoint &ak, std::ostream &s) {
  s << ak.x << " " << ak.y << " " << ak.a11 << " " << ak.a12 << " " << ak.a21 << " " << ak.a22 << " ";
  s << ak.pyramid_scale << " " << ak.octave_number << " " << ak.s << " " << ak.sub_type << " ";
}
void saveKPBench(AffineKeypoint &ak, std::ostream &s) {
  s << ak.x << " " << ak.y << " "  << ak.s << " " << ak.a11 << " " << ak.a12 << " " << ak.a21 << " " << ak.a22;
}

void saveAR(AffineRegion &ar, std::ostream &s) {
  s << ar.id << " " << ar.img_id << " "  << " ";
  s << ar.parent_id <<  " ";
  saveKP(ar.det_kp,s);
  saveKP(ar.reproj_kp,s);
  // s << ar.desc.type <<
  s << " " << ar.desc.size() << " ";
  for (unsigned int i = 0; i < ar.desc.size(); ++i) {
      s << ar.desc[i] << " ";
    }
}
void saveARBench(AffineRegion &ar, std::ostream &s, std::ostream &s2) {
  saveKPBench(ar.det_kp,s2);
  saveKPBench(ar.reproj_kp,s);
}
void loadKP(AffineKeypoint &ak, std::istream &s) {
  s >> ak.x >> ak.y >> ak.a11 >> ak.a12 >>ak.a21 >> ak.a22 >> ak.pyramid_scale >> ak.octave_number >> ak.s >> ak.sub_type;
}

void loadAR(AffineRegion &ar, std::istream &s) {
  s >> ar.id >> ar.img_id;
  s >> ar.parent_id;
  loadKP(ar.det_kp,s);
  loadKP(ar.reproj_kp,s);
  //  s >> ar.desc.type;
  int size1;
  s >> size1;
  ar.desc.resize(size1);
  for (unsigned int i = 0; i < ar.desc.size(); ++i) {
      s >> ar.desc[i];
    }
}
void L2normalize(const float* input_arr, int size, std::vector<float> &output_vect)
{
  double norm = 0.0;
  for (int i = 0; i < size; ++i) {
      norm+=input_arr[i] * input_arr[i];
    }
  double norm_coef = 1.0/sqrt(norm);
  for (int i = 0; i < size; ++i) {
    float v1 = floor(512.0*norm_coef*input_arr[i]);
      output_vect[i] = v1;
    }
}
void L1normalize(const float* input_arr, int size, std::vector<float> &output_vect)
{
  double norm=0.0;
  for (int i = 0; i < size; ++i) {
      norm+=input_arr[i];
    }
  double norm_coef = 1.0/norm;
  for (int i = 0; i < size; ++i) {
      float v1 = floor(512.0*norm_coef*input_arr[i]);
      output_vect[i] = v1;
    }
}
void RootNormalize(const float* input_arr, int size, std::vector<float> &output_vect)
{
  L2normalize(input_arr,size,output_vect);
  double norm=0.0;
  for (int i = 0; i < size; ++i) {
      norm+=input_arr[i];
    }
  double norm_coef = 1.0/norm;
  for (int i = 0; i < size; ++i) {
      float v1 = sqrt(512.0*norm_coef*input_arr[i]);
      output_vect[i] = v1;
    }
}

ImageRepresentation::ImageRepresentation(cv::Mat _in_img, std::string _name)
{
  if (_in_img.channels() ==3) {
      _in_img.convertTo(OriginalImg,CV_32FC3);

    } else {
      _in_img.convertTo(OriginalImg,CV_32F);
    }
  Name = _name;
}
ImageRepresentation::ImageRepresentation()
{

}
ImageRepresentation::~ImageRepresentation()
{
  RegionVectorMap.clear();
}
descriptor_type ImageRepresentation::GetDescriptorType(std::string desc_name)
{
  for (unsigned int i=0; i< DescriptorNames.size(); i++)
    if (DescriptorNames[i].compare(desc_name)==0)
      return static_cast<descriptor_type>(i);
  return DESC_UNKNOWN;
}

detector_type ImageRepresentation::GetDetectorType(std::string det_name)
{
  for (unsigned int i=0; i< DetectorNames.size(); i++)
    if (DetectorNames[i].compare(det_name)==0)
      return static_cast<detector_type>(i);
  return DET_UNKNOWN;
}

TimeLog ImageRepresentation::GetTimeSpent()
{
  return TimeSpent;
}

int ImageRepresentation::GetRegionsNumber(std::string det_name)
{
  int reg_number = 0;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  if (det_name.compare("All") == 0)
    {
      for (regions_it = RegionVectorMap.begin();
           regions_it != RegionVectorMap.end(); regions_it++)
        {
          AffineRegionVectorMap::iterator desc_it;
          if ( (desc_it = regions_it->second.find("None")) != regions_it->second.end() )
            reg_number +=  desc_it->second.size();
        }
    }
  else
    {
      regions_it = RegionVectorMap.find(det_name);
      if ( regions_it != RegionVectorMap.end())
        {
          AffineRegionVectorMap::iterator desc_it;
          if ( (desc_it = regions_it->second.find("None")) != regions_it->second.end() )
            reg_number +=  desc_it->second.size();
        }
    }
  return reg_number;
}
int ImageRepresentation::GetDescriptorsNumber(std::string desc_name, std::string det_name)
{
  int reg_number = 0;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  if (det_name.compare("All") == 0)
    {
      for (regions_it = RegionVectorMap.begin();
           regions_it != RegionVectorMap.end(); regions_it++)
        if (desc_name.compare("All") == 0)
          {
            for (desc_it = regions_it->second.begin();
                 desc_it != regions_it->second.end(); desc_it++)
              reg_number +=  desc_it->second.size();
          }
        else
          {
            desc_it = regions_it->second.find(desc_name);
            if (desc_it != regions_it->second.end() )
              reg_number +=  desc_it->second.size();

          }
    }
  else
    {
      regions_it = RegionVectorMap.find(det_name);
      if ( regions_it != RegionVectorMap.end())
        {
          if (desc_name.compare("All") == 0)
            {
              for (desc_it = regions_it->second.begin();
                   desc_it != regions_it->second.end(); desc_it++)
                reg_number +=  desc_it->second.size();
            }
          else
            {
              desc_it = regions_it->second.find(desc_name);
              if (desc_it != regions_it->second.end() )
                reg_number +=  desc_it->second.size();

            }
        }
    }
  return reg_number;
}
int ImageRepresentation::GetDescriptorDimension(std::string desc_name)
{
  int dim = 0;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  for (regions_it = RegionVectorMap.begin();regions_it != RegionVectorMap.end(); regions_it++)
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        if (desc_it->second.size() > 0)
          {
            dim = desc_it->second[0].desc.size();
            break;
          }
    }
  return dim;
}

AffineRegion ImageRepresentation::GetAffineRegion(std::string desc_name, std::string det_name, int idx)
{
  AffineRegion curr_region;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  regions_it = RegionVectorMap.find(det_name);
  if ( regions_it != RegionVectorMap.end())
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        {
          AffineRegionVector *currentDescVector = &(desc_it->second);
          curr_region = (*currentDescVector)[idx];
          return curr_region;
        }
    }
  return curr_region;
}
AffineRegionVector ImageRepresentation::GetAffineRegionVector(std::string desc_name, std::string det_name, std::vector<int> idxs)
{
  unsigned int n_regs = idxs.size();
  AffineRegionVector regions;
  regions.reserve(n_regs);

  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  regions_it = RegionVectorMap.find(det_name);
  if ( regions_it != RegionVectorMap.end())
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        {
          AffineRegionVector *currentDescVector = &(desc_it->second);
          for (unsigned int i = 0; i < n_regs; i++)
            regions.push_back((*currentDescVector)[idxs[i]]);
        }
    }
  return regions;
}
AffineRegionVector ImageRepresentation::GetAffineRegionVector(std::string desc_name, std::string det_name)
{
  unsigned int n_regs = GetDescriptorsNumber(desc_name,det_name);
  AffineRegionVector regions;
  regions.reserve(n_regs);

  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  regions_it = RegionVectorMap.find(det_name);
  if ( regions_it != RegionVectorMap.end())
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        {
          AffineRegionVector *currentDescVector = &(desc_it->second);
          for (unsigned int i = 0; i < n_regs; i++)
            regions.push_back((*currentDescVector)[i]);
        }
    }
  return regions;
}

void ImageRepresentation::AddRegions(AffineRegionVector &RegionsToAdd, std::string det_name, std::string desc_name)
{
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  regions_it = RegionVectorMap.find(det_name);
  if ( regions_it != RegionVectorMap.end())
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        {
          AffineRegionVector *currentDescVector = &(desc_it->second);
          ImageRepresentation::AddRegionsToList(*currentDescVector,RegionsToAdd);
        }
      else
        {
          regions_it->second[desc_name] = RegionsToAdd;
        }
    }
  else
    {
      std::map<std::string, AffineRegionVector> new_desc;
      new_desc[desc_name] = RegionsToAdd;
      RegionVectorMap[det_name] = new_desc;
    }
}
void ImageRepresentation::AddRegions(AffineRegionVectorMap &RegionsMapToAdd, std::string det_name)
{
  AffineRegionVectorMap::iterator desc_it;

  for (desc_it = RegionsMapToAdd.begin();
       desc_it != RegionsMapToAdd.end(); desc_it++)
    AddRegions(desc_it->second,det_name,desc_it->first);
}

void ImageRepresentation::AddRegionsToList(AffineRegionVector &kp_list, AffineRegionVector &new_kps)
{
  int size = (int)kp_list.size();
  unsigned int new_size = size + new_kps.size();
  AffineRegionVector::iterator ptr = new_kps.begin();
  for (unsigned int i=size; i< new_size; i++, ptr++)
    {
      AffineRegion temp_reg = *ptr;
      temp_reg.id += size;
      temp_reg.parent_id +=size;
      kp_list.push_back(temp_reg);
    }
}

static const double mrSizeORB = 3.0;

int detect_orbs(cv::Mat &input, std::vector<AffineKeypoint> &out1, const ORBParams &params, ScalePyramid &scale_pyramid, double tilt = 1.0, double zoom = 1.0) {
  std::vector<cv::KeyPoint> keys;
  auto det = cv::ORB::create(
      params.nfeatures,
      params.scaleFactor,
      params.nlevels,
      params.edgeThreshold,
      params.firstLevel,
      params.WTA_K,
      cv::ORB::HARRIS_SCORE,
      params.PEParam.patchSize);

  cv::Mat img;
  input.convertTo(img, CV_8U);
  det->detect(img, keys);

  int kp_size = keys.size();
  out1.resize(kp_size);
  for (int kp_num=0; kp_num<kp_size; kp_num++)
  {
    auto &tmp_kp = out1[kp_num];
    auto &key = keys[kp_num];

    tmp_kp.x = key.pt.x;
    tmp_kp.y = key.pt.y;
    tmp_kp.a11 = cos(key.angle*M_PI/180.0);
    tmp_kp.a12 = sin(key.angle*M_PI/180.0);
    tmp_kp.a21 = -sin(key.angle*M_PI/180.0);
    tmp_kp.a22 = cos(key.angle*M_PI/180.0);
    tmp_kp.s = key.size / mrSizeORB;
    tmp_kp.response = key.response;
  }
  return kp_size;
}

void describe_ORBs(vector<AffineKeypoint> &aff_keys, vector<descriptor_t> descs, cv::Mat &input, ORBParams &param) {
  auto det = cv::ORB::create(
      param.nfeatures,
      param.scaleFactor,
      param.nlevels,
      param.edgeThreshold,
      param.firstLevel,
      param.WTA_K,
      cv::ORB::HARRIS_SCORE,
      param.PEParam.patchSize);
  cv::Mat descriptors;
  std::vector<cv::KeyPoint> keys;
  cv::Mat img;
  cv::Mat mask;
  input.convertTo(img, CV_8U);
  det->detectAndCompute(img, mask, keys, descriptors);
  int kp_size = keys.size();
  descs.resize(kp_size);
  int desc_dim = descriptors.cols;

  aff_keys.resize(kp_size);
  for (int i = 0; i < kp_size; i++) {
    auto &aff_key = aff_keys[i];
    auto &key = keys[i];
    auto &desc = descs[i];
    aff_key.x = key.pt.x;
    aff_key.y = key.pt.y;
    aff_key.a11 = cos(key.angle * M_PI / 180.0);
    aff_key.a12 = sin(key.angle * M_PI / 180.0);
    aff_key.a21 = -sin(key.angle * M_PI / 180.0);
    aff_key.a22 = cos(key.angle * M_PI / 180.0);
    aff_key.s = key.size / mrSizeORB;
    aff_key.response = key.response;

    desc.resize(desc_dim);

    unsigned char *descPtr = descriptors.ptr<unsigned char>(i);
    for (int jj = 0; jj < desc_dim; jj++, descPtr++)
      desc[jj] = (float) *descPtr;
  }
}

vector<AffineRegion> reproject_clean_to_affine_regions(vector<AffineKeypoint> &keys, vector<descriptor_t> &descs, detector_type det_type, descriptor_type desc_type, int img_id, double *H, int w, int h, double mrSize) {
  vector<AffineKeypoint> reproj_kps;
  reproject_and_remove_boundary(keys, reproj_kps, H, w, h, mrSize);
  return convert_affine_regions(keys, reproj_kps, descs, det_type, desc_type, img_id);
}

void ImageRepresentation::SynthDetectDescribeKeypoints (IterationViewsynthesisParam &synth_par,
                                                        DetectorsParameters &det_par,
                                                        DescriptorsParameters &desc_par,
                                                        DominantOrientationParams &dom_ori_par)
{
  double time1 = 0;
  //Create grayscale image

  cv::Mat gray_in_img;

  if (OriginalImg.channels() == 3)
    {
      //cv::cvtColor(in_img, gray_in_img, CV_BGR2GRAY);
      std::vector<cv::Mat> RGB_planes(3);
      cv::Mat in_32f;
      OriginalImg.convertTo(in_32f,CV_32FC3);
      cv::split(in_32f, RGB_planes);
      // gray_in_img = cv::Mat::zeros(in_img.cols, in_img.rows,CV_32FC1);
      gray_in_img = (RGB_planes[0] + RGB_planes[1] + RGB_planes[2]) / 3.0 ;
    } else
    {
      gray_in_img = OriginalImg;
      std::cerr << "Grayscale input" << std::endl;
    }

#ifdef _OPENMP
  omp_set_nested(1);
#endif
#pragma omp parallel for schedule (dynamic,1)
  for (unsigned int det=0; det < DetectorNames.size(); det++)
    {
      std::string curr_det = DetectorNames[det];
      unsigned int n_synths = synth_par[curr_det].size();

      std::vector<AffineRegionVectorMap> OneDetectorKeypointsMapVector;
      OneDetectorKeypointsMapVector.resize(n_synths);

#pragma omp parallel for schedule (dynamic,1)
      for (unsigned int synth=0; synth<n_synths; synth++)
        {
          ///Synthesis
          long s_time = getMilliSecs1();
          SynthImage temp_img1;
          GenerateSynthImageCorr(gray_in_img, temp_img1, Name.c_str(),
                                 synth_par[curr_det][synth].tilt,
                                 synth_par[curr_det][synth].phi,
                                 synth_par[curr_det][synth].zoom,
                                 synth_par[curr_det][synth].InitSigma,
                                 synth_par[curr_det][synth].doBlur, synth);
          time1 = ((double)(getMilliSecs1() - s_time))/1000;
          TimeSpent.SynthTime += time1;

          bool present_SIFT_like_desc = false;
          bool present_HalfSIFT_like_desc = false;

          for (unsigned int i_desc=0; i_desc < synth_par[curr_det][synth].descriptors.size();i_desc++) {
              std::string curr_desc = synth_par[curr_det][synth].descriptors[i_desc];
              if (curr_desc.find("Half") != std::string::npos) {
                  present_HalfSIFT_like_desc = true;
                } else {
                  if (curr_desc.find("SIFT") != std::string::npos) {
                      present_SIFT_like_desc = true;
                    }
                }
            }
          /// Detection
          s_time = getMilliSecs1();
          std::vector<AffineKeypoint> aff_keys, reproj_aff_keys;
          detector_type det_type;
          if (curr_det == "HessianAffine" || curr_det == "DoG" || curr_det == "HarrisAffine") {
            ScaleSpaceDetectorParams *par;
            if (curr_det == "HessianAffine")
              par = &det_par.HessParam;
            else if (curr_det == "DoG")
              par = &det_par.DoGParam;
            else if (curr_det == "HarrisAffine")
              par = &det_par.HarrParam;

            DetectAffineKeypoints(temp_img1.pixels, aff_keys, *par, temp_img1.pyramid, temp_img1.tilt, temp_img1.zoom);
            det_type = par->PyramidPars.DetectorType;
          }
          else if (curr_det == "MSER")
          {
            DetectMSERs(temp_img1.pixels, aff_keys, det_par.MSERParam, temp_img1.pyramid, temp_img1.tilt, temp_img1.zoom);
            det_type = DET_MSER;
          }
          else if (curr_det == "ORB")
          {
            detect_orbs(temp_img1.pixels, aff_keys, det_par.ORBParam, temp_img1.pyramid, temp_img1.tilt, temp_img1.zoom);

            if (det_par.ORBParam.doBaumberg) {
              vector<AffineKeypoint> temp_aff_keys;
              detect_affine_shapes(aff_keys, temp_aff_keys, temp_img1.pixels, det_par.BaumbergParam);
              aff_keys = temp_aff_keys;
            }
            det_type = DET_ORB;
          }

          vector<descriptor_t> empty_descs;
          AffineRegionVectorMap temp_kp_map;
          temp_kp_map["None"] = reproject_clean_to_affine_regions(aff_keys, empty_descs, det_type, DESC_UNKNOWN, temp_img1.id, temp_img1.H, OriginalImg.cols, OriginalImg.rows, 3.);

          time1 = ((double)(getMilliSecs1() - s_time))/1000;
          TimeSpent.DetectTime += time1;

          //
          /// Orientation estimation
          vector<AffineKeypoint> temp_kp1_SIFT_like_desc;
          vector<AffineKeypoint> temp_kp1_HalfSIFT_like_desc;
          vector<AffineKeypoint> temp_kp1_upright;

          if (present_SIFT_like_desc) {
              detect_orientations(aff_keys, temp_kp1_SIFT_like_desc, temp_img1.pixels,
                                dom_ori_par.PEParam.mrSize, dom_ori_par.PEParam.patchSize,
                                false, dom_ori_par.maxAngles,
                                dom_ori_par.threshold, false);
            }
          if (present_HalfSIFT_like_desc) {
              detect_orientations(aff_keys, temp_kp1_HalfSIFT_like_desc, temp_img1.pixels, dom_ori_par.PEParam.mrSize, dom_ori_par.PEParam.patchSize,
                                true, dom_ori_par.maxAngles,
                                dom_ori_par.threshold, false);
            }
          if (dom_ori_par.addUpRight) {
              detect_orientations(aff_keys, temp_kp1_upright, temp_img1.pixels,
                                dom_ori_par.PEParam.mrSize, dom_ori_par.PEParam.patchSize,
                                false, 0, 1.0, true);
            }

          // Description

          time1 = ((double) (getMilliSecs1() - s_time)) / 1000;
          TimeSpent.OrientTime += time1;
          s_time = getMilliSecs1();
          //          std::cerr << "Desc" << std::endl;

          for (unsigned int i_desc=0; i_desc < synth_par[curr_det][synth].descriptors.size();i_desc++) {
              std::string curr_desc = synth_par[curr_det][synth].descriptors[i_desc];
              vector<AffineKeypoint> temp_kp1_desc, temp_reproj_kp1_desc;
              vector<descriptor_t> temp_kp1_desc_desc;

              //Add oriented and upright keypoints if any
              if (curr_desc.find("Half") != std::string::npos) {
                  temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1_HalfSIFT_like_desc.begin(), temp_kp1_HalfSIFT_like_desc.end());
              } else {
                temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1_SIFT_like_desc.begin(), temp_kp1_SIFT_like_desc.end());
              }
              // Add upright if detected
              temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1_upright.begin(), temp_kp1_upright.end());

              reproject_and_remove_boundary(temp_kp1_desc, temp_reproj_kp1_desc, temp_img1.H, OriginalImg.cols, OriginalImg.rows);

              ///Description

              descriptor_type desc_type;
              if (curr_desc.compare("RootSIFT") == 0) //RootSIFT
                {
                  SIFTDescriptor RootSIFTdesc(desc_par.RootSIFTParam);
                  describe_regions(temp_kp1_desc, temp_kp1_desc_desc,
                                  temp_img1.pixels, &RootSIFTdesc,
                                  desc_par.RootSIFTParam.PEParam.mrSize,
                                  desc_par.RootSIFTParam.PEParam.patchSize,
                                  desc_par.RootSIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.RootSIFTParam.PEParam.photoNorm);
                  desc_type = DESC_ROOT_SIFT;
                }
              else if (curr_desc.compare("HalfRootSIFT") == 0) //HalfRootSIFT
                {
                  SIFTDescriptor HalfRootSIFTdesc(desc_par.HalfRootSIFTParam);
                  describe_regions(temp_kp1_desc, temp_kp1_desc_desc,
                                  temp_img1.pixels, &HalfRootSIFTdesc,
                                  desc_par.HalfRootSIFTParam.PEParam.mrSize,
                                  desc_par.HalfRootSIFTParam.PEParam.patchSize,
                                  desc_par.HalfRootSIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.HalfRootSIFTParam.PEParam.photoNorm);
                  desc_type = DESC_HALF_ROOT_SIFT;
                }
              else if (curr_desc.compare("HalfSIFT") == 0) //HalfSIFT
                {
                  ///Description
                  SIFTDescriptor HalfSIFTdesc(desc_par.HalfSIFTParam);
                  describe_regions(temp_kp1_desc, temp_kp1_desc_desc,
                                  temp_img1.pixels, &HalfSIFTdesc,
                                  desc_par.HalfSIFTParam.PEParam.mrSize,
                                  desc_par.HalfSIFTParam.PEParam.patchSize,
                                  desc_par.HalfSIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.HalfSIFTParam.PEParam.photoNorm);
                  desc_type = DESC_HALF_SIFT;
                }

              else if (curr_desc.compare("SIFT") == 0) //SIFT
                {
                  SIFTDescriptor SIFTdesc(desc_par.SIFTParam);
                  describe_regions(temp_kp1_desc, temp_kp1_desc_desc,
                                  temp_img1.pixels, &SIFTdesc,
                                  desc_par.SIFTParam.PEParam.mrSize,
                                  desc_par.SIFTParam.PEParam.patchSize,
                                  desc_par.SIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.SIFTParam.PEParam.photoNorm);
                  desc_type = DESC_SIFT;
                }
              else if (curr_desc.compare("Pixels") == 0) //Raw Pixels
                {
                  PIXELSDescriptor PixelDesc(desc_par.PixelsParam);
                  describe_regions(temp_kp1_desc, temp_kp1_desc_desc,
                                  temp_img1.pixels, &PixelDesc,
                                  desc_par.PixelsParam.PEParam.mrSize,
                                  desc_par.PixelsParam.PEParam.patchSize,
                                  desc_par.PixelsParam.PEParam.FastPatchExtraction,
                                  desc_par.PixelsParam.PEParam.photoNorm);
                  desc_type = DESC_PIXELS;
                }
              else if (curr_desc.compare("ORB") == 0) //ORB (not uses orientation estimated points)
              {
                describe_ORBs(temp_kp1_desc, temp_kp1_desc_desc, temp_img1.pixels, det_par.ORBParam);
                reproject_and_remove_boundary(temp_kp1_desc, temp_reproj_kp1_desc, temp_img1.H, OriginalImg.cols, OriginalImg.rows, mrSizeORB);
                desc_type = DESC_ORB;
              }

              reproject_and_remove_boundary(temp_kp1_desc, temp_reproj_kp1_desc, temp_img1.H, OriginalImg.cols, OriginalImg.rows, 3.0);
              temp_kp_map[curr_desc] = convert_affine_regions(temp_kp1_desc, temp_reproj_kp1_desc, temp_kp1_desc_desc, det_type, desc_type, temp_img1.id);

              time1 = ((double)(getMilliSecs1() - s_time)) / 1000;
              TimeSpent.DescTime += time1;
              s_time = getMilliSecs1();
            }
          OneDetectorKeypointsMapVector[synth] = temp_kp_map;
        }
      for (unsigned int synth=0; synth < n_synths; synth++)
        AddRegions(OneDetectorKeypointsMapVector[synth], curr_det);
    }
}

void ImageRepresentation::SaveRegions(std::string fname, int mode) {
  std::ofstream kpfile(fname);
  if (mode == std::ios::binary) {

    } else {
      if (kpfile.is_open()) {
          kpfile << RegionVectorMap.size() << std::endl;
          for (std::map<std::string, AffineRegionVectorMap>::const_iterator
               reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
              kpfile << reg_it->first << " " << reg_it->second.size() << std::endl;
              std::cerr << reg_it->first << " " << reg_it->second.size() << std::endl;

              for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
                   desc_it != reg_it->second.end(); ++desc_it) {
                  kpfile << desc_it->first << " " << desc_it->second.size() << std::endl;
                  int n_desc = desc_it->second.size();
                  if (n_desc > 0) {
                      kpfile << (desc_it->second)[0].desc.size() << std::endl;
                    } else {
                      std::cerr << "No descriptor " << desc_it->first << std::endl;
                    }
                  for (int i = 0; i < n_desc ; i++ ) {
                      AffineRegion ar = desc_it->second[i];
                      saveAR(ar, kpfile);
                      kpfile << std::endl;
                    }
                }
            }
        }
      else {
          std::cerr << "Cannot open file " << fname << " to save keypoints" << endl;
        }
      kpfile.close();
    }
}

void ImageRepresentation::LoadRegions(std::string fname) {

  std::ifstream kpfile(fname);
  if (kpfile.is_open()) {
      int numberOfDetectors = 0;
      kpfile >> numberOfDetectors;
      //    std::cerr << "numberOfDetectors=" <<numberOfDetectors << std::endl;
      for (int det = 0; det < numberOfDetectors; det++) {
          std::string det_name;
          int num_of_descs = 0;
          kpfile >> det_name;
          kpfile >> num_of_descs;
          //      std::cerr << det_name << " " << num_of_descs << std::endl;

          //reg_it->first << " " << reg_it->second.size() << std::endl;
          for (int desc = 0; desc < num_of_descs; desc++)  {
              AffineRegionVector desc_regions;
              std::string desc_name;
              kpfile >> desc_name;

              int num_of_kp = 0;
              kpfile >> num_of_kp;
              int desc_size;
              kpfile >> desc_size;
              //        std::cerr << desc_name << " " << num_of_kp << " " << desc_size << std::endl;
              for (int kp = 0; kp < num_of_kp; kp++)  {
                  AffineRegion ar;
                  loadAR(ar, kpfile);
                  desc_regions.push_back(ar);
                }
              AddRegions(desc_regions,det_name,desc_name);
            }
        }
    }
  else {
      std::cerr << "Cannot open file " << fname << " to save keypoints" << endl;
    }
  kpfile.close();
}

} //namespace mods
