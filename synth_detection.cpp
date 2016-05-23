/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#undef __STRICT_ANSI__

#include "synth_detection.hpp"
#include <assert.h>
#include <opencv2/opencv.hpp>
#include "detectors/mser/utls/matrix.h"
#include "detectors/mser/extrema/extrema.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mods {

//const double k_sigma = /*2 **/ 3.0 * sqrt(3.0);//to compare ellipses in 3*sigma size
const double eps1 = 0.01;
using namespace std;
using namespace cv;

#define VERBOSE 0
#define VERTICAL 1
#define HORIZONTAL 0

const int MAX_HEIGHT = 10000;
const int MAX_WIDTH = 10000;
const double COS_PI_2 = cos(M_PI/2);
const double SIN_PI_2 = sin(M_PI/2);

int SetVSPars(const std::vector <double> &scale_set,
              const std::vector <double> &tilt_set,
              double phi_base,
              const std::vector <double> &FGINNThreshold,
              const std::vector <double> &DistanceThreshold,
              const std::vector <std::string> descriptors,
              std::vector<ViewSynthParameters> &par,
              std::vector<ViewSynthParameters> &prev_par,
              double InitSigma,
              int doBlur,
              int dsplevels,
              double minSigma, double maxSigma)
{
  par.clear();
  std::vector<ViewSynthParameters> prev_par_tmp(prev_par);
  std::vector<ViewSynthParameters> pars_tmp;

  if ((scale_set.size() ==0) || (tilt_set.size() == 0))
    {
      ViewSynthParameters temp_par;
      temp_par.phi = 0;
      temp_par.tilt = 0;
      temp_par.zoom = 0;
      temp_par.InitSigma = InitSigma;
      temp_par.doBlur = 0;
      temp_par.DSPlevels = dsplevels;
      temp_par.descriptors = descriptors;
      for (unsigned int desc=0; desc<descriptors.size(); desc++)
        {
          temp_par.DistanceThreshold[descriptors[desc]]=DistanceThreshold[desc];
          temp_par.FGINNThreshold[descriptors[desc]]=FGINNThreshold[desc];
        }
      pars_tmp.push_back(temp_par);
    }
  for (unsigned int sc=0; sc < scale_set.size(); sc++)
    for (unsigned int t=0; t < tilt_set.size(); t++)
      {
        if (fabs(tilt_set[t] - 1) > eps1)
          {
            int n_rot1 = floor(180.0*tilt_set[t]/phi_base);
            double delta_phi = M_PI/n_rot1;
            if (n_rot1 < 0) { //no rotation mode if negative, add vertical tilt
                n_rot1 = 1;
                delta_phi = 0;
                double phi = 0;
                assert (phi >= 0);
                ViewSynthParameters temp_par;
                temp_par.phi = phi;
                temp_par.tilt = -tilt_set[t];
                temp_par.zoom = scale_set[sc];
                temp_par.InitSigma = InitSigma;
                temp_par.doBlur = doBlur;
                temp_par.DSPlevels = dsplevels;
                temp_par.minSigma = minSigma;
                temp_par.maxSigma = maxSigma;
                temp_par.descriptors = descriptors;

                for (unsigned int desc=0; desc<descriptors.size(); desc++)
                  {
                    temp_par.DistanceThreshold[descriptors[desc]]=DistanceThreshold[desc];
                    temp_par.FGINNThreshold[descriptors[desc]]=FGINNThreshold[desc];
                  }
                pars_tmp.push_back(temp_par);

              }
            for (int r=0 ; r<n_rot1; r++)
              {
                double phi = delta_phi * r;
                assert (phi >= 0);
                ViewSynthParameters temp_par;
                temp_par.phi = phi;
                temp_par.tilt = tilt_set[t];
                temp_par.zoom = scale_set[sc];
                temp_par.InitSigma = InitSigma;
                temp_par.doBlur = doBlur;
                temp_par.DSPlevels = dsplevels;
                temp_par.minSigma = minSigma;
                temp_par.maxSigma = maxSigma;
                temp_par.descriptors = descriptors;

                for (unsigned int desc=0; desc<descriptors.size(); desc++)
                  {
                    temp_par.DistanceThreshold[descriptors[desc]]=DistanceThreshold[desc];
                    temp_par.FGINNThreshold[descriptors[desc]]=FGINNThreshold[desc];
                  }
                pars_tmp.push_back(temp_par);
              }
          }
        else
          {
            ViewSynthParameters temp_par;
            temp_par.phi = 0;
            temp_par.tilt = tilt_set[t];
            temp_par.zoom = scale_set[sc];
            temp_par.InitSigma = InitSigma;
            temp_par.doBlur = doBlur;
            temp_par.DSPlevels = dsplevels;
            temp_par.minSigma = minSigma;
            temp_par.maxSigma = maxSigma;
            temp_par.descriptors = descriptors;
            for (unsigned int desc=0; desc<descriptors.size(); desc++)
              {
                temp_par.DistanceThreshold[descriptors[desc]]=DistanceThreshold[desc];
                temp_par.FGINNThreshold[descriptors[desc]]=FGINNThreshold[desc];
              }
            pars_tmp.push_back(temp_par);
            continue;
          }
      }
  std::vector<char> isUnique(pars_tmp.size());
  for (unsigned int i=0; i<pars_tmp.size(); i++)
    isUnique[i]=1;

  for (unsigned int i=0; i<pars_tmp.size(); i++)
    for (unsigned int j=0; j<prev_par_tmp.size(); j++)
      if ((fabs(pars_tmp[i].zoom - prev_par_tmp[j].zoom) <= eps1) &&
          (fabs(pars_tmp[i].tilt - prev_par_tmp[j].tilt) <= eps1) &&
          (fabs(pars_tmp[i].phi - prev_par_tmp[j].phi) <= eps1))
        {
          isUnique[i]=0;
          break;
        }

  std::vector<ViewSynthParameters>::iterator ptr = pars_tmp.begin();
  for (unsigned int i=0; i<pars_tmp.size(); i++, ptr++)
    if (isUnique[i]) par.push_back(*ptr);

  for (unsigned int i=0; i<par.size(); i++)
    prev_par_tmp.push_back(par[i]);
  prev_par = prev_par_tmp;
  return (int)par.size();
}

template<typename T>
static inline T round(T v) {
  return floor(0.5+v);
}

void GenerateSynthImageCorr(const cv::Mat &in_img,
                            SynthImage &out_img,
                            const std::string in_img_name,
                            double tilt,
                            double phi,
                            double zoom,
                            double init_sigma,
                            int doBlur,
                            int img_id)
{
  while (phi >= 2.0*M_PI)
      phi -= 2.0*M_PI;
  while (phi < 0 )
      phi += 2.0*M_PI;

  bool vertical_tilt = false;
  if (tilt < 0) { // vertical tilt
      tilt = -tilt;
      vertical_tilt = true;
    }
  bool zoomed = fabs(zoom-1.0f)>=0.05;
  cv::Mat temp_img, gray_in_img = in_img;

  double sigma_aa, sigma_aa_2, sigma_x,sigma_y;
  int w =in_img.cols;
  int h = in_img.rows;
  auto out_H = out_img.H;
  out_img.OrigImgName= in_img_name;

  if ((fabs(tilt - 1.) <=0.1) && (abs(phi) <= 0.2) && (fabs(zoom - 1.) <=0.1)) //original image
    { out_img.rotation= 0.0;
      out_img.tilt= 1.0;
      out_img.zoom= 1.0;
      out_img.id = 0;
      out_H[0]=1.0; out_H[1]=0;   out_H[2]=0;
      out_H[3]=0;   out_H[4]=1.0; out_H[5]=0;
      out_H[6]=0;   out_H[7]=0;   out_H[8]=1.0;
      out_img.pixels = gray_in_img;
      return;
    }

  out_img.id = img_id;
  out_H[6]= 0;
  out_H[7]=0;
  out_H[8]=1;
  double phi_deg = phi*180/M_PI;
  out_img.rotation= phi_deg;
  out_img.tilt=tilt;
  out_img.zoom = zoom;

  double nw = (abs(sin(phi)*h) + abs(cos(phi)*w)) * zoom;
  double nh = (abs(cos(phi)*h) + abs(sin(phi)*w)) * zoom;
  auto R = cv::getRotationMatrix2D(Point2f(nw/2, nh/2), phi_deg, zoom);
  auto R_off = (Mat)(R * (cv::Mat_<double>(3,1) << (nw-w)/2, (nh-h)/2, 0));
  R.at<double>(0,2) += R_off.at<double>(0, 0);
  R.at<double>(1,2) += R_off.at<double>(1, 0);
  cv::warpAffine(gray_in_img, temp_img, R,
                 cv::Size(nw, nh),cv::INTER_LINEAR, cv::BORDER_CONSTANT,cv::Scalar(128,128,128));

  /// Anti-aliasing filtering
  if (zoomed)
    sigma_aa_2 = init_sigma / (4.0*zoom);
  else
    sigma_aa_2 = init_sigma / 2.0;
  sigma_aa = init_sigma * tilt / (2.0*zoom);

  if (vertical_tilt) {
    sigma_x = sigma_aa_2;
    sigma_y = sigma_aa;
  } else {
    sigma_x = sigma_aa;
    sigma_y = sigma_aa_2;
  }
  if (doBlur)
  {
    int k_size_x = round(2.0 * 3.0 * sigma_x + 1.0);
    if (k_size_x % 2 == 0)
      k_size_x++;
    if (k_size_x < 3)
      k_size_x = 3;

    int k_size_y = round(2.0 * 3.0 * sigma_y + 1.0);
    if (k_size_y % 2 == 0)
      k_size_y++;
    if (k_size_y < 3)
      k_size_y = 3;
    cv::GaussianBlur(temp_img,temp_img,cv::Size(k_size_x, k_size_y),sigma_x,sigma_y);
  }

  double a, b;
  if (vertical_tilt) {
    a = 1;
    b = 1/tilt;
  } else {
    a = 1/tilt;
    b = 1;
  }
  Mat H1 = (cv::Mat_<double>(2,3) << a, 0, 0, 0, b, 0);
  cv::warpAffine(temp_img, out_img.pixels, H1,
                 cv::Size(nw*a,nh*b),cv::INTER_LINEAR, cv::BORDER_CONSTANT,cv::Scalar(128,128,128));
  Mat H = H1(Range::all(), Range(0, 2)) * R;
  out_H[0] = H.at<double>(0,0);   out_H[1] = H.at<double>(0,1);   out_H[2] = H.at<double>(0,2);
  out_H[3] = H.at<double>(1,0);   out_H[4] = H.at<double>(1,1);   out_H[5] = H.at<double>(1,2);
}
void ReprojectByH(AffineKeypoint in_kp, AffineKeypoint &out_kp, double* H) //For H=[h11 h12 h13; h21 h22 h23; 0 0 1];
{
  out_kp.x=(H[0]*in_kp.x+H[1]*in_kp.y+H[2]);// /(H[6]*in_kp.x+H[7]*in_kp.y+H[8]);
  out_kp.y=(H[3]*in_kp.x+H[4]*in_kp.y+H[5]);// /(H[6]*in_kp.x+H[7]*in_kp.y+H[8]);
  out_kp.a11=(H[0]*in_kp.a11+H[1]*in_kp.a21);
  out_kp.a12=(H[0]*in_kp.a12+H[1]*in_kp.a22);
  out_kp.a21=(H[3]*in_kp.a11+H[4]*in_kp.a21);
  out_kp.a22=(H[3]*in_kp.a12+H[4]*in_kp.a22);
}

bool HIsEye(double* H) {
  return (fabs(H[0] - 1.0) + fabs(H[1]) + fabs(H[2])
      + fabs(H[3]) + fabs(H[4] - 1.0) + fabs(H[5]) +
      fabs(H[6]) + fabs(H[7]) + fabs(H[8] - 1.0) < eps1);

}
int reproject_and_remove_boundary(vector<AffineKeypoint> &keypoints, vector<AffineKeypoint> &reproj_kps, double *H, int orig_w, int orig_h, double mrSize) {

  cv::Mat H1(3, 3, CV_64F, H);
  cv::Mat Hinv(3, 3, CV_64F);
  cv::invert(H1, Hinv, cv::DECOMP_LU);
  double *HinvPtr = (double *) Hinv.data;

  reproj_kps.resize(keypoints.size());
  if (HIsEye(H)) {
    for (unsigned i = 0; i < keypoints.size(); i++)
      reproj_kps[i] = keypoints[i];
  } else {
    for (unsigned i = 0; i < keypoints.size(); i++) {
      reproj_kps[i] = keypoints[i];
      ReprojectByH(keypoints[i], reproj_kps[i], HinvPtr);
    }
  }

  vector<AffineKeypoint> temp_keypoints, temp_reproj_kps;
  temp_keypoints.reserve(keypoints.size());
  for (unsigned int i=0; i < keypoints.size(); i++)
    {
      auto &reproj_kp = reproj_kps[i];
      if ( (reproj_kp.x < orig_w) && (reproj_kp.y < orig_h)
           && (reproj_kp.x > 0) && (reproj_kp.y > 0)) {  //center is inside
          if ( !interpolateCheckBorders(orig_w, orig_h,
                                        reproj_kp.x, reproj_kp.y,
                                        reproj_kp.a11, reproj_kp.a12,
                                        reproj_kp.a21, reproj_kp.a22,
                                        mrSize * reproj_kp.s,
                                        mrSize * reproj_kp.s)) {
              temp_keypoints.push_back(keypoints[i]);
              temp_reproj_kps.push_back(reproj_kps[i]);
            }
        }
    }
  keypoints = temp_keypoints;
  reproj_kps = temp_reproj_kps;
  return (int)keypoints.size();
}

void AddRegionsToList(AffineRegionVector &kp_list, AffineRegionVector &new_kps)
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
void AddRegionsToListByType(AffineRegionVector &kp_list, AffineRegionVector &new_kps,int type)
{
  int size = (int)kp_list.size();
  AffineRegionVector::iterator ptr =new_kps.begin();
  unsigned int new_size = size + new_kps.size();
  for (unsigned int i=size; i< new_size; i++, ptr++)
    {
      if (ptr->type == type)
        {
          AffineRegion temp_reg = *ptr;
          temp_reg.id += size;
          temp_reg.parent_id +=size;
          kp_list.push_back(temp_reg);
        }
    }
}
template <int bins>
void smoothCircularBuffer(float *hist)
{
  float first = hist[0], prev = hist[bins-1];
  for (int i = 0; i < bins - 1; i++)
    {
      float cur = hist[i];
      hist[i] = prev + cur + hist[i+1];
      prev = cur;
    }
  hist[bins-1] = prev + hist[bins-1] + first;
}

template <int bins>
inline void addPeakAngle(float *hist, vector<float> &angles, int a, int b, int c,
                         float threshold, vector<float> &peak_values)
{
  if (hist[b] >= threshold && hist[b] > hist[a] && hist[b] > hist[c])
    {
      float pp = (hist[a] - hist[c]) / (hist[a] - 2.0f * hist[b] + hist[c]) / 2.0f;
      angles.push_back(2.0f * float(M_PI) * (b + 0.5f + pp) / bins - float(M_PI));
      peak_values.push_back(hist[b]);
    }
}

vector<AffineRegion> convert_affine_regions(vector<AffineKeypoint> &aff_keys, vector<AffineKeypoint> &reproj_keys, vector<descriptor_t> &descs, detector_type det_type, descriptor_type desc_type, int img_id)
//Function detects affine regions using detector function and writes them into AffineRegionVector structure
{

  int region_nr=aff_keys.size();
  vector<AffineRegion> keypoints(region_nr);
  AffineRegion ar;
  ar.img_id=img_id;
  ar.type = det_type;

  for (int i = 0; i < region_nr; i++)
  {
    ar.id = i;
    // ar.det_kp.s=ptr->s * sqrt(fabs(ptr->a11 * ptr->a22 - ptr->a12 * ptr->a21));
    // rectifyTransformation(ptr->a11,ptr->a12,ptr->a21,ptr->a22);
    ar.det_kp = aff_keys[i];
    ar.reproj_kp = reproj_keys[i];
    keypoints[i] = ar;
  }
  if (descs.size() > 0)
  {
    assert(descs.size() == aff_keys.size());
    for (int i = 0; i < descs.size(); i++) {
      keypoints[i].desc_type = desc_type;
      keypoints[i].desc = descs[i];
    }
  }
  return keypoints;
}

struct EstimateDominantAnglesFunctor
{
private:
  cv::Mat gmag;
  cv::Mat gori;
  cv::Mat orimask;
  int pS;
  double magThresh;
  int doHalfSIFT;
  static const int bins = 36;
public:
  EstimateDominantAnglesFunctor(int patchSize,int doHalfSIFT1 = 0) : pS(patchSize),doHalfSIFT(doHalfSIFT1)
  {
    gmag = cv::Mat (pS, pS, CV_32FC1, cv::Scalar(0));
    gori = cv::Mat (pS, pS, CV_32FC1, cv::Scalar(0));
    orimask = cv::Mat(pS, pS, CV_32FC1);
    computeCircularGaussMask(orimask, pS/3.0f);
  }
  void operator()(const cv::Mat &img, vector<float> &angles1,
                  double max_th=0.8, int maxAngles= -1)
  {
    if (maxAngles == 0) {
        angles1.clear();
        return;
      }
    float hist[bins+1];
    vector<float> peak_values;
    for (int i = 0; i<bins; i++) hist[i] = 0.0f;

    computeGradientMagnitudeAndOrientation(img, gmag, gori);

    float *maskptr = orimask.ptr<float>(1);
    float *pmag = gmag.ptr<float>(1), *pori = gori.ptr<float>(1);
    int maskPixels = orimask.cols * (orimask.rows-2);

    for (int i = 0; i < maskPixels; ++i)
      {
        if (*maskptr > 0 && *pmag > 1.0)
          {
            int bin = (int) (bins * (*pori/float(M_PI) + 1.0f) / 2.0f);
            assert(bin >= 0 && bin <= bins);
            hist[bin] += (*pmag) * (*maskptr);
          }
        pmag++;
        pori++;
        maskptr++;
      }

    for (int i = 0; i < 6; i++)
      smoothCircularBuffer<bins>(hist);
    float thresh = 0.0;
    for (int i = 0; i < bins; i++)
      if (hist[i] > thresh) thresh = hist[i];
    thresh *= max_th;

    if (doHalfSIFT) {
        int halfbins = bins / 2;
        for (int i = 0; i < halfbins; i++)
          {
            hist[i] += hist[i+halfbins];
            hist[i+halfbins] = 0;
          }
      }

    // output all local maxima above threshold
    angles1.clear();
    addPeakAngle<bins>(hist, angles1, bins-1, 0, 1, thresh,peak_values);
    for (int i = 1; i < bins-1; i++)
      addPeakAngle<bins>(hist, angles1, i-1, i, i+1, thresh,peak_values);

    addPeakAngle<bins>(hist, angles1, bins-2, bins-1, 0, thresh,peak_values);
    if (maxAngles == -1) {
        maxAngles = 100000000;
      }
    maxAngles = min(maxAngles, (int)peak_values.size());
    if (maxAngles > 0) {
        vector<float> peak_values_sorted = peak_values;
        std::sort(peak_values_sorted.begin(), peak_values_sorted.end());
        vector<float> ang_tmp;
        for (int ang = 0; ang < maxAngles; ang++)
          {
            if (peak_values[ang] >= thresh) {
                ang_tmp.push_back(angles1[ang]);
              } else {
                break;
              }
          }
        angles1 = ang_tmp;
      } else {
        angles1.clear();
      }
  }
};
int detect_orientation(const vector<AffineKeypoint> &in_kp_list,
                      vector<AffineKeypoint> &out_kp_list,
                       cv::Mat &img,
                      double mrSize,
                      int patchSize,
                      bool doHalfSIFT,
                      int maxAngNum,
                      double th,
                      bool addUpRight) {
  vector<AffineKeypoint> temp_kp_list;
  temp_kp_list.reserve(in_kp_list.size());

  //unsigned int i;
  double mrScale = (double)mrSize; // half patch size in pixels of image
  int patchImageSize = 2*int(mrScale)+1; // odd size
  vector<float> angles1;//, angles2;
  //  angles1.reserve(5);
  // angles2.reserve(5);
  double imageToPatchScale = double(patchImageSize) / (double)patchSize;
  // patch size in the image / patch size -> amount of down/up sampling

  cv::Mat patch(patchSize,patchSize,CV_32FC1);

  EstimateDominantAnglesFunctor EstDomOri(patchSize,doHalfSIFT);
  for (int i=0; i < in_kp_list.size(); i++)
    {
      auto &r=in_kp_list[i];
      angles1.clear();
      float curr_sc = imageToPatchScale*r.s;

      if (interpolateCheckBorders(img.cols,img.rows,
                                  (float) r.x,
                                  (float) r.y,
                                  (float) r.a11,
                                  (float) r.a12,
                                  (float) r.a21,
                                  (float) r.a22,
                                  mrSize * r.s,
                                  mrSize * r.s) ) {
          continue;
        }
      if (maxAngNum > 0) {
          //we have to renumerate next regions.

          interpolate(img,(float)r.x,
                      (float)r.y,
                      (float)r.a11*curr_sc,
                      (float)r.a12*curr_sc,
                      (float)r.a21*curr_sc,
                      (float)r.a22*curr_sc,
                      patch);
          EstDomOri(patch,angles1,th,maxAngNum);
          for (size_t j = 0; j < angles1.size(); j++)
            {
              double ci = cos(-angles1[j]);
              double si = sin(-angles1[j]);

              auto temp_region = r;
              temp_region.a11 = r.a11*ci-r.a12*si;
              temp_region.a12 = r.a11*si+r.a12*ci;
              temp_region.a21 = r.a21*ci-r.a22*si;
              temp_region.a22 = r.a21*si+r.a22*ci;
              temp_kp_list.push_back(temp_region);
            }
        }
      if (addUpRight) {
          temp_kp_list.push_back(r);
        }
    }
  out_kp_list=temp_kp_list;
  return (int)temp_kp_list.size();
}

int detect_affine_shape(const vector<AffineKeypoint> &in_kp_list,
                        vector<AffineKeypoint> &out_kp_list,
                        cv::Mat &img,
                        const AffineShapeParams par) {

  out_kp_list.clear();
  int kp_size = in_kp_list.size();
  const float initialSigma = 1.6;
  cv::Mat gmag, gori, orimask;
  //  std::vector<unsigned char> workspace;
  cv::Mat mask, patch, imgHes, fx, fy;

  gmag = cv::Mat(par.patchSize, par.patchSize, CV_32FC1);
  gori = cv::Mat(par.patchSize, par.patchSize, CV_32FC1);
  orimask = cv::Mat(par.patchSize, par.patchSize, CV_32FC1);
  mask = cv::Mat(par.smmWindowSize, par.smmWindowSize, CV_32FC1);
  patch = cv::Mat(par.smmWindowSize, par.smmWindowSize, CV_32FC1);
  fx = cv::Mat(par.smmWindowSize, par.smmWindowSize, CV_32FC1);
  fy = cv::Mat(par.smmWindowSize, par.smmWindowSize, CV_32FC1);

  computeGaussMask(mask);
  computeCircularGaussMask(orimask, par.smmWindowSize);
  for (int kp_num=0; kp_num < kp_size; kp_num++)
    {
      auto r = in_kp_list[kp_num];
      float eigen_ratio_act = 0.0f, eigen_ratio_bef = 0.0f;
      float u11 = 1.0f, u12 = 0.0f, u21 = 0.0f, u22 = 1.0f, l1 = 1.0f, l2 = 1.0f;
      float lx = r.x, ly = r.y;
      float ratio =  r.s / (initialSigma);
      cv::Mat U, V, d, Au, Ap, D;

      int maskPixels = par.smmWindowSize * par.smmWindowSize;
      if (interpolateCheckBorders(img.cols,img.rows,
                                  (float) r.x,
                                  (float) r.y,
                                  (float) r.a11,
                                  (float) r.a12,
                                  (float) r.a21,
                                  (float) r.a22,
                                  2*5.0*ratio,
                                  2*5.0*ratio) ) {
          continue;
        }
      for (int l = 0; l < par.maxIterations; l++)
        {
          float a = 0, b = 0, c = 0;
          if (par.affBmbrgMethod == AFF_BMBRG_SMM) {
              // warp input according to current shape matrix
              interpolate(img, lx, ly, u11*ratio, u12*ratio, u21*ratio, u22*ratio, patch);
              //            std::cerr << "after interp ok" << std::endl;
              // compute SMM on the warped patch
              float *maskptr = mask.ptr<float>(0);
              float *pfx = fx.ptr<float>(0), *pfy = fy.ptr<float>(0);
              computeGradient(patch, fx, fy);

              // estimate SMM
              for (int i = 0; i < maskPixels; ++i)
                {
                  float v = (*maskptr);
                  float gxx = *pfx;
                  float gyy = *pfy;
                  float gxy = gxx * gyy;

                  a += gxx * gxx * v;
                  b += gxy * v;
                  c += gyy * gyy * v;
                  pfx++;
                  pfy++;
                  maskptr++;
                }
              a /= maskPixels;
              b /= maskPixels;
              c /= maskPixels;

              // compute inverse sqrt of the SMM
              invSqrt(a, b, c, l1, l2);

              if ((a != a) || (b != b) || (c !=c)){ //check for nan
                  break;
                }

              // update e igen ratios
              eigen_ratio_bef = eigen_ratio_act;
              eigen_ratio_act = 1.0 - l2 / l1;

              // accumulate the affine shape matrix
              float u11t = u11, u12t = u12;

              u11 = a*u11t+b*u21;
              u12 = a*u12t+b*u22;
              u21 = b*u11t+c*u21;
              u22 = b*u12t+c*u22;

            } else {
              if (par.affBmbrgMethod == AFF_BMBRG_HESSIAN) {
                  float Dxx, Dxy, Dyy;
                  float affRatio = r.s * 0.5;
                  Ap = (cv::Mat_<float>(2,2) << u11, u12, u21, u22);
                  interpolate(img, lx, ly, u11*affRatio, u12*affRatio, u21*affRatio, u22*affRatio, imgHes);

                  Dxx = (      imgHes.at<float>(0,0) - 2.f*imgHes.at<float>(0,1) +     imgHes.at<float>(0,2)
                               + 2.f*imgHes.at<float>(1,0) - 4.f*imgHes.at<float>(1,1) + 2.f*imgHes.at<float>(1,2)
                               +     imgHes.at<float>(2,0) - 2.f*imgHes.at<float>(2,1) +     imgHes.at<float>(2,2));

                  Dyy = (      imgHes.at<float>(0,0) + 2.f*imgHes.at<float>(0,1) +     imgHes.at<float>(0,2)
                               - 2.f*imgHes.at<float>(1,0) - 4.f*imgHes.at<float>(1,1) - 2.f*imgHes.at<float>(1,2)
                               +     imgHes.at<float>(2,0) + 2.f*imgHes.at<float>(2,1) +     imgHes.at<float>(2,2));

                  Dxy = (      imgHes.at<float>(0,0)           -     imgHes.at<float>(0,2)
                               - imgHes.at<float>(2,0)           +     imgHes.at<float>(2,2));

                  // Inv. square root using SVD method, somehow the SMM method does not work
                  Au = (cv::Mat_<float>(2,2) << Dxx, Dxy, Dxy, Dyy);
                  cv::SVD::compute(Au,d,U,V);

                  l1 = d.at<float>(0,0);
                  l2 = d.at<float>(0,1);

                  eigen_ratio_bef=eigen_ratio_act;
                  eigen_ratio_act=1.0-abs(l2)/abs(l1);

                  float det = sqrt(abs(l1*l2));
                  l2 = sqrt(sqrt(abs(l1)/det));
                  l1 = 1./l2;

                  D = (cv::Mat_<float>(2,2) << l1, 0, 0, l2);
                  Au = U * D * V;
                  Ap = Au * Ap * Au;

                  u11 = Ap.at<float>(0,0); u12 = Ap.at<float>(0,1);
                  u21 = Ap.at<float>(1,0); u22 = Ap.at<float>(1,1);
                }
            }
          // compute the eigen values of the shape matrix
          if (!getEigenvalues(u11, u12, u21, u22, l1, l2))
            break;

          // leave on too high anisotropy
          if ((l1/l2>6) || (l2/l1>6))
            break;

          if (eigen_ratio_act < par.convergenceThreshold
              && eigen_ratio_bef < par.convergenceThreshold) {
              r.a11 = u11;
              r.a12 = u12;
              r.a21 = u21;
              r.a22 = u22;
              out_kp_list.push_back(r);
              break;
            }
        }
    }
}

void linH(double x, double y, double *H, double *linearH)
{
  double den, den_sq, num1_densq, num2_densq, a11,a12,a21,a22;

  den =(H[6]*x + H[7]*y +H[8]);
  den_sq=den*den;

  num1_densq = (H[0]*x + H[1]*y +H[2])/den_sq;
  num2_densq = (H[3]*x + H[4]*y +H[5])/den_sq;
  a11 = H[0]/den - num1_densq*H[6];
  a12 = H[1]/den - num1_densq*H[7];

  a21 = H[3]/den - num2_densq*H[6];
  a22 = H[4]/den - num2_densq*H[7];

  linearH[0]=a11;
  linearH[1]=a12;
  linearH[2]=a21;
  linearH[3]=a22;
}

//Detects orientation of the affine region and adds regions with detected orientation to the list.
//All points that derived from one have the same parent_id

void describe_regions(vector<AffineKeypoint> &in_kp_list,
                      vector<descriptor_t> &descs,
                      cv::Mat &img, DescriptorFunctor *descriptor,
                      double mrSize, int patchSize, bool fast_extraction, bool photoNorm)
//Describes region with SIFT or other descriptor
{
 // std::cerr << "photonorm=" << photoNorm << std::endl;
  std::vector<unsigned char> workspace;
  unsigned int i;
  // patch size in the image / patch size -> amount of down/up sampling
  cv::Mat patch(patchSize, patchSize, CV_32FC1);
  unsigned int n_descs = in_kp_list.size();
  cv::Mat mask(patchSize,patchSize,CV_32F);
  descs.resize(n_descs);
  computeCircularGaussMask(mask);
  if ( !fast_extraction) {
    for (i = 0; i < n_descs; i++) {
      auto &key = in_kp_list[i];
      float mrScale = ceil(key.s * mrSize); // half patch size in pixels of image

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
        interpolate(img,
                    (float) key.x,
                    (float) key.y,
                    (float) key.a11,
                    (float) key.a12,
                    (float) key.a21,
                    (float) key.a22,
                    smoothed);

        gaussianBlurInplace(smoothed, 1.5f * imageToPatchScale);
        // subsample with corresponding scale
        interpolate(smoothed, (float) (patchImageSize >> 1), (float) (patchImageSize >> 1),
                    imageToPatchScale, 0, 0, imageToPatchScale, patch);
      } else {
        // if imageToPatchScale is small (i.e. lot of oversampling), affine normalize without smoothing
        interpolate(img,
                    (float) key.x,
                    (float) key.y,
                    (float) key.a11 * imageToPatchScale,
                    (float) key.a12 * imageToPatchScale,
                    (float) key.a21 * imageToPatchScale,
                    (float) key.a22 * imageToPatchScale,
                    patch);

      }
      if (photoNorm) {
          float mean, var;
          photometricallyNormalize(patch, mask, mean, var);
        }
      (*descriptor)(patch, descs[i]);
    }
  } else {
    for (i = 0; i < n_descs; i++) {
      auto &key = in_kp_list[i];
      double mrScale = (double) mrSize * key.s; // half patch size in pixels of image
      int patchImageSize = 2 * int(mrScale) + 1; // odd size
      double imageToPatchScale = double(patchImageSize) / (double) patchSize;
      float curr_sc = imageToPatchScale;

      interpolate(img,
                  (float) key.x,
                  (float) key.y,
                  (float) key.a11 * curr_sc,
                  (float) key.a12 * curr_sc,
                  (float) key.a21 * curr_sc,
                  (float) key.a22 * curr_sc,
                  patch);
      if (photoNorm) {
          float mean, var;
          photometricallyNormalize(patch, mask, mean, var);
        }
      (*descriptor)(patch, descs[i]);
    }
  }
}

} //namespace mods
