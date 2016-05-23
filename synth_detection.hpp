#ifndef MODS_SYNTHDETECTION_HPP
#define MODS_SYNTHDETECTION_HPP
/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/
#undef __STRICT_ANSI__

#include <sys/time.h>
#include "common.hpp"

namespace mods {

inline long getMilliSecs1()
{
  timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec*1000 + t.tv_usec/1000;
}
/// Functions

//Rotates ellipse vertically(not the shape, just orientation) and normalizes matrix determinant to one

int SetVSPars (const std::vector <double> &scale_set,
               const std::vector <double> &tilt_set,
               double phi_base,
               const std::vector <double> &FGINNThreshold,
               const std::vector <double> &DistanceThreshold,
               const std::vector <std::string> descriptors,
               std::vector<ViewSynthParameters> &par,
               std::vector<ViewSynthParameters> &prev_par,
               double InitSigma=0.5,
               int doBlur=1, int dsplevels = 0,
               double mixSigma=1.0, double maxSigma=1.0);
//Function generates parameters for view synthesis based on gived scale, tilt and rotation sets, avoiding duplicates with previous synthesis.
void GenerateSynthImageCorr(const cv::Mat &in_img,
                            SynthImage &out_img,
                            const std::string in_img_name,
                            double tilt,
                            double phi,
                            double zoom,
                            double InitSigma=0.5,
                            int doBlur=1,
                            int img_id = 0);
//Function generates scaled, rotated and tilted image with homography from original to generated image and places all this into SynthImage structure
//Phi is rotation angle in radians
//Tilt - is scale factor in horizontal direction (to simulate real tilt)
//Zoom - scale factor
//InitSigma (= 0.5 by default). Bluring is done with sigma_aa = InitSigma * tilt / 2 for tilting and sigma_aa = InitSigma / (4*zoom) for downscaling.
//doBlur - to make gaussian convolution before scaling or no

//void GenerateSynthImageByH(const cv::Mat &in_img, SynthImage &out_img,double* H,double InitSigma = 0.5,int doBlur =1,int img_id = 0);
//Function generates scaled, rotated and tilted image from image and homography matrix from original to generated image and places all this into SynthImage structure


vector<AffineRegion> convert_affine_regions(vector<AffineKeypoint> &aff_keys, vector<AffineKeypoint> &reproj_keys, vector<descriptor_t> &descs, detector_type det_type, descriptor_type desc_type, int img_id);
//Function detects affine regions using detector function and writes them into AffineRegionVector structure

int reproject_and_remove_boundary(vector<AffineKeypoint> &keypoints, vector<AffineKeypoint> &reproj_keys, double *H, int orig_w, int orig_h, double mrSize = 3.0*sqrt(3.0));
//Function reprojects detected regions to other image ("original") using H matrix (H is from original to tilted).
//Then all regions that are outside original image (fully or partially) are deleted.


int detect_orientation(const vector<AffineKeypoint> &in_kp_list,
                      vector<AffineKeypoint> &out_kp_list1,
                       cv::Mat &img,
                      double mrSize = 3.0*sqrt(3.0),
                      int patchSize = 41,
                      bool doHalfSIFT = false,
                      int maxAngNum= 0,
                      double th = 0.8,
                      bool addUpRight = false);

int detect_affine_shape(const vector<AffineKeypoint> &in_kp_list,
                        vector<AffineKeypoint> &out_kp_list1,
                        cv::Mat &img,
                        const AffineShapeParams par);

//Detects orientation of the affine region and adds regions with detected orientation to the list.
//All points that derived from one have the same parent_id

void describe_regions(vector<AffineKeypoint> &in_kp_list, vector<descriptor_t> &descs, cv::Mat &img, DescriptorFunctor *descriptor, double mrSize = 3.0*sqrt(3.0), int patchSize = 41, bool fast_extraction = false, bool photoNorm = false);

void AddRegionsToList(AffineRegionVector &kp_list, AffineRegionVector& new_kps);
//Function for getting new regions ID right (original IDs are changed to new ones to ensure no collisions in kp_list)

void AddRegionsToListByType(AffineRegionVector &kp_list, AffineRegionVector& new_kps, int type);
//Function for getting new regions ID right AND only given type

void WriteKPs(AffineRegionVector &keys, std::ostream &out1);
//Function writes keypoints to stream in format:
//descriptor_size(default = 128) keys_number
//x y scale a11 a12 a21 a22 desc[descriptor_size]

void ReadKPs(AffineRegionVector &keys, std::istream &in1);
//Function reads keypoints from stream in format:
//descriptor_size(default = 128) keys_number
//x y scale a11 a12 a21 a22 desc[descriptor_size]

// void ReadKPsMik(AffineRegionVector &keys, std::istream &in1);
//Function reads keypoints from stream in Mikolajczuk format:
//descriptor_size(default = 128) keys_number
//x y scale a b c desc[descriptor_size]

void linH(double x, double y, double *H, double *linearH);
//Function linearizes homography matrix to affine

} // namespace mods

#endif // SYNTHDETECTION_HPP
