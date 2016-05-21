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

void rectifyTransformation(double &a11, double &a12, double &a21, double &a22);
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


int DetectAffineRegions(vector<AffineKeypoint> &aff_keys, AffineRegionVector &keypoints, detector_type det_type, int img_id);
//Function detects affine regions using detector function and writes them into AffineRegionVector structure

int ReprojectRegionsAndRemoveTouchBoundary(AffineRegionVector &keypoints, double *H, int orig_w, int orig_h, double mrSize = 3.0*sqrt(3.0));
//Function reprojects detected regions to other image ("original") using H matrix (H is from original to tilted).
//Then all regions that are outside original image (fully or partially) are deleted.


int DetectOrientation(AffineRegionVector &in_kp_list,
                      AffineRegionVector &out_kp_list1,
                      SynthImage &img,
                      const  double mrSize = 3.0*sqrt(3.0),
                      int patchSize = 41,
                      int doHalfSIFT = 0,
                      int maxAngNum= 0,
                      double th = 0.8,
                      const bool addUpRight = false);

int DetectAffineShape(AffineRegionVector &in_kp_list,
                      AffineRegionVector &out_kp_list1,
                      SynthImage &img,
                      const AffineShapeParams par);

//Detects orientation of the affine region and adds regions with detected orientation to the list.
//All points that derived from one have the same parent_id

template <typename FuncType>
void DescribeRegions(AffineRegionVector &in_kp_list,
                     SynthImage &img, FuncType descriptor,
                     double mrSize = 3.0*sqrt(3.0), int patchSize = 41, bool fast_extraction = false, bool photoNorm = false)
//Describes region with SIFT or other descriptor
{
 // std::cerr << "photonorm=" << photoNorm << std::endl;
  std::vector<unsigned char> workspace;
  unsigned int i;
  // patch size in the image / patch size -> amount of down/up sampling
  cv::Mat patch(patchSize, patchSize, CV_32FC1);
  unsigned int n_descs = in_kp_list.size();
  cv::Mat mask(patchSize,patchSize,CV_32F);
  computeCircularGaussMask(mask);
  if ( !fast_extraction) {
    for (i = 0; i < n_descs; i++) {
      float mrScale = ceil(in_kp_list[i].det_kp.s * mrSize); // half patch size in pixels of image

      int patchImageSize = 2 * int(mrScale) + 1; // odd size
      float imageToPatchScale = float(patchImageSize) / float(patchSize);  // patch size in the image / patch size -> amount of down/up sampling
      // is patch touching boundary? if yes, ignore this feature
      if (imageToPatchScale > 0.4) {
        // the pixels in the image are 0.4 apart + the affine deformation
        // leave +1 border for the bilinear interpolation
        patchImageSize += 2;
        size_t wss = patchImageSize * patchImageSize * sizeof(float);
        if (wss >= workspace.size())
          workspace.resize(wss);

        Mat smoothed(patchImageSize, patchImageSize, CV_32FC1, (void *) &workspace.front());
        // interpolate with det == 1
        interpolate(img.pixels,
                    (float) in_kp_list[i].det_kp.x,
                    (float) in_kp_list[i].det_kp.y,
                    (float) in_kp_list[i].det_kp.a11,
                    (float) in_kp_list[i].det_kp.a12,
                    (float) in_kp_list[i].det_kp.a21,
                    (float) in_kp_list[i].det_kp.a22,
                    smoothed);

        gaussianBlurInplace(smoothed, 1.5f * imageToPatchScale);
        // subsample with corresponding scale
        interpolate(smoothed, (float) (patchImageSize >> 1), (float) (patchImageSize >> 1),
                    imageToPatchScale, 0, 0, imageToPatchScale, patch);
      } else {
        // if imageToPatchScale is small (i.e. lot of oversampling), affine normalize without smoothing
        interpolate(img.pixels,
                    (float) in_kp_list[i].det_kp.x,
                    (float) in_kp_list[i].det_kp.y,
                    (float) in_kp_list[i].det_kp.a11 * imageToPatchScale,
                    (float) in_kp_list[i].det_kp.a12 * imageToPatchScale,
                    (float) in_kp_list[i].det_kp.a21 * imageToPatchScale,
                    (float) in_kp_list[i].det_kp.a22 * imageToPatchScale,
                    patch);

      }
      if (photoNorm) {
          float mean, var;
          photometricallyNormalize(patch, mask, mean, var);
        }
      descriptor(patch, in_kp_list[i].desc.vec);
      in_kp_list[i].desc.type = descriptor.type;
    }
  } else {
    for (i = 0; i < n_descs; i++) {
      double mrScale = (double) mrSize * in_kp_list[i].det_kp.s; // half patch size in pixels of image
      int patchImageSize = 2 * int(mrScale) + 1; // odd size
      double imageToPatchScale = double(patchImageSize) / (double) patchSize;
      float curr_sc = imageToPatchScale;

      interpolate(img.pixels,
                  (float) in_kp_list[i].det_kp.x,
                  (float) in_kp_list[i].det_kp.y,
                  (float) in_kp_list[i].det_kp.a11 * curr_sc,
                  (float) in_kp_list[i].det_kp.a12 * curr_sc,
                  (float) in_kp_list[i].det_kp.a21 * curr_sc,
                  (float) in_kp_list[i].det_kp.a22 * curr_sc,
                  patch);
      if (photoNorm) {
          float mean, var;
          photometricallyNormalize(patch, mask, mean, var);
        }
      descriptor(patch, in_kp_list[i].desc.vec);
      in_kp_list[i].desc.type = descriptor.type;
    }
  }
}
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

void ReadKPsMik(AffineRegionVector &keys, std::istream &in1);
//Function reads keypoints from stream in Mikolajczuk format:
//descriptor_size(default = 128) keys_number
//x y scale a b c desc[descriptor_size]

void linH(double x, double y, double *H, double *linearH);
//Function linearizes homography matrix to affine

} // namespace mods

#endif // SYNTHDETECTION_HPP
