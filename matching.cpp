/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#undef __STRICT_ANSI__

#include "matching.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <fstream>

#include "USAC/src/estimators/USAC.h"
#include "USAC/src/estimators/HomogEstimator.h"
#include "USAC/src/estimators/FundmatrixEstimator.h"


#define DO_TRANSFER_H_CHECK

#ifdef _OPENMP
#include <omp.h>
#endif

#define WRITE_H 1
#define VERB 0
using namespace std;

namespace mods {

bool CompareCorrespondenceByRatio(TentativeCorrespExt corr1, TentativeCorrespExt corr2) {return (fabs(corr1.ratio) < fabs(corr2.ratio));}
bool CompareCorrespondenceByDistance(TentativeCorrespExt corr1, TentativeCorrespExt corr2) {return (fabs(corr1.d1) < fabs(corr2.d1));}
bool CompareCorrespondenceByScale(TentativeCorrespExt corr1, TentativeCorrespExt corr2) {return (fabs(corr1.first.reproj_kp.s) < fabs(corr2.first.reproj_kp.s));}

cv::flann::Index GenFLANNIndex(cv::Mat keys, cvflann::flann_algorithm_t indexType, cvflann::flann_distance_t dist_type, const int nTrees)
{
  switch (indexType)
    {
    case cvflann::FLANN_INDEX_KDTREE:
      {
        return  cv::flann::Index(keys,cv::flann::KDTreeIndexParams(nTrees),dist_type);
        break;
      }
    case cvflann::FLANN_INDEX_COMPOSITE:
      {
        return  cv::flann::Index(keys,cv::flann::CompositeIndexParams(nTrees),dist_type);
        break;
      }
    case cvflann::FLANN_INDEX_AUTOTUNED:
      {
        return cv::flann::Index(keys,cv::flann::AutotunedIndexParams(0.8,0.9),dist_type);
        break;
      }
    case cvflann::FLANN_INDEX_KMEANS:
      {
        return cv::flann::Index(keys,cv::flann::KMeansIndexParams(),dist_type);
        break;
      }
    case cvflann::FLANN_INDEX_LSH:
      {
        return cv::flann::Index(keys,cv::flann::LshIndexParams(30, 8, 2),dist_type);
        break;
      }
    case cvflann::FLANN_INDEX_LINEAR:
      {
        return cv::flann::Index(keys,cv::flann::LinearIndexParams(),dist_type);
        break;
      }
    case cvflann::FLANN_INDEX_HIERARCHICAL:
      {
        return cv::flann::Index(keys,cv::flann::HierarchicalClusteringIndexParams(),dist_type);
        break;
      }
    default:
      {
        return cv::flann::Index(keys,cv::flann::KDTreeIndexParams(nTrees),dist_type);
        break;
      }
    }

}

void  GetEpipoles (double *F, double *e1, double *e2)
{
  cv::Mat Fmat (3,3,CV_64F,F);
  cv::Mat U,D,V;
  cv::SVDecomp(Fmat,D,U,V,4);


  e2[0] = U.at<double>(0,2) / U.at<double>(2,2);
  e2[1] = U.at<double>(1,2) / U.at<double>(2,2);
  e2[2] = 1.0;

  e1[0] = V.at<double>(0,2) / V.at<double>(2,2);
  e1[1] = V.at<double>(1,2) / V.at<double>(2,2);
  e1[2] = 1.0;

}
void GetEpipolarLine(double *e, double *pt, double *l, double &k, double &b)
{
  l[0] = e[1]*pt[2] - e[2]*pt[1];
  l[1] = e[2]*pt[0] - e[0]*pt[2];
  l[2] = e[0]*pt[1] - e[1]*pt[0];

  double x_crossx = - l[2] / l[0];
  double x_crossy = 0;
  double y_crossx = 0;
  double y_crossy = -l[2] / l[1];
  k = (y_crossx - y_crossy)/(x_crossx - x_crossy);
  b = y_crossy;
}


void GetEpipolarLineF(double *F, double *pt, double *l, double &k, double &b)
{

  l[0] = pt[0]*F[0] + pt[1]*F[3] + pt[2]*F[6];
  l[1] = pt[0]*F[1] + pt[1]*F[4] + pt[2]*F[7];
  l[2] = pt[0]*F[2] + pt[1]*F[5] + pt[2]*F[8];

  double x_crossx = - l[2] / l[0];
  double x_crossy = 0;
  double y_crossx = 0;
  double y_crossy = -l[2] / l[1];
  k = (y_crossx - y_crossy)/(x_crossx - x_crossy);
  b = y_crossy;
}
const double k_sigma = 3.0;

inline double distanceSq (const AffineKeypoint &kp1,const AffineKeypoint &kp2)
{
  double dx = kp1.x - kp2.x;
  double dy = kp1.y - kp2.y;
  return dx*dx + dy*dy;
}
inline void oppositeDirection (AffineRegion &kp1)
{
  kp1.reproj_kp.a11 = - kp1.reproj_kp.a11;
  kp1.reproj_kp.a12 = - kp1.reproj_kp.a12;
  kp1.reproj_kp.a21 = - kp1.reproj_kp.a21;
  kp1.reproj_kp.a22 = - kp1.reproj_kp.a22;

  kp1.det_kp.a11 = - kp1.det_kp.a11;
  kp1.det_kp.a12 = - kp1.det_kp.a12;
  kp1.det_kp.a21 = - kp1.det_kp.a21;
  kp1.det_kp.a22 = - kp1.det_kp.a22;
}
int F_LAF_check_USAC(std::vector<TentativeCorrespExt> &in_matches, double *F, std::vector<TentativeCorrespExt> &res,
                     const double affineFerror, USACConfig::GeometricErrorFunction geom_error_func)
{
  int n_tents = (int)in_matches.size();
  int bad_pts=0;
  std::vector<TentativeCorrespExt> good_matches;
  std::vector<int> good_pts(n_tents);
  for (int a=0; a<n_tents; a++)
    good_pts[a]=1; //initialization


  if (affineFerror > 0)
    {

      std::vector<TentativeCorrespExt>::iterator ptr =  in_matches.begin();
      for (int l=0; l<n_tents; l++,ptr++)
        {
          double u[18],err[3];
          u[0] = ptr->first.reproj_kp.x;
          u[1] = ptr->first.reproj_kp.y;
          u[2] = 1.0;

          u[3] = ptr->second.reproj_kp.x;
          u[4] = ptr->second.reproj_kp.y;
          u[5] = 1.0;

          if (geom_error_func == USACConfig::ERR_SYMMETRIC_TRANSFER_SUM) {
              err[0] = calcFundSampsonErr(u,u+3,F);

            } else if (geom_error_func == USACConfig::ERR_SAMPSON) {
              err[0] = calcFundSymmetricEpipolarErr(u,u+3,F);
            }

          u[6] = u[0]+k_sigma*ptr->first.reproj_kp.a12*ptr->first.reproj_kp.s;
          u[7] = u[1]+k_sigma*ptr->first.reproj_kp.a22*ptr->first.reproj_kp.s;
          u[8] = 1.0;

          u[9]  = u[3]+k_sigma*ptr->second.reproj_kp.a12*ptr->second.reproj_kp.s;
          u[10] = u[4]+k_sigma*ptr->second.reproj_kp.a22*ptr->second.reproj_kp.s;
          u[11] = 1.0;

          if (geom_error_func == USACConfig::ERR_SYMMETRIC_TRANSFER_SUM) {
              err[1] = calcFundSampsonErr(u+6,u+9,F);

            } else if (geom_error_func == USACConfig::ERR_SAMPSON) {
              err[1] = calcFundSymmetricEpipolarErr(u+6,u+9,F);
            }
          u[12] = u[0]+k_sigma*ptr->first.reproj_kp.a11*ptr->first.reproj_kp.s;
          u[13] = u[1]+k_sigma*ptr->first.reproj_kp.a21*ptr->first.reproj_kp.s;
          u[14] = 1.0;

          u[15] = u[3]+k_sigma*ptr->second.reproj_kp.a11*ptr->second.reproj_kp.s;
          u[16] = u[4]+k_sigma*ptr->second.reproj_kp.a21*ptr->second.reproj_kp.s;
          u[17] = 1.0;

          if (geom_error_func == USACConfig::ERR_SYMMETRIC_TRANSFER_SUM) {
              err[2] = calcFundSampsonErr(u+12,u+15,F);

            } else if (geom_error_func == USACConfig::ERR_SAMPSON) {
              err[2] = calcFundSymmetricEpipolarErr(u+12,u+15,F);
            }

          double sumErr=sqrt(err[0])+sqrt(err[1])+sqrt(err[2]);
          if (sumErr > affineFerror)
            {
              good_pts[l]=0;
              bad_pts++;
            }
        }
      good_matches.reserve(n_tents - bad_pts);
      for (int l=0; l<n_tents; l++)
        if (good_pts[l]) good_matches.push_back(in_matches[l]);
      res = good_matches;
    }
  else res = in_matches;
  return res.size();
}

int H_LAF_check_USAC(std::vector<TentativeCorrespExt> &in_matches, double *H, double *H_inv, std::vector<TentativeCorrespExt> &res,
                const double affineFerror, USACConfig::GeometricErrorFunction geom_error_func)
{
  int n_tents = (int)in_matches.size();
  int bad_pts=0;
  std::vector<TentativeCorrespExt> good_matches;
  std::vector<int> good_pts(n_tents);
  for (int a=0;  a < n_tents; a++)
    good_pts[a]=1; //initialization
  if (affineFerror > 0)
    {
      std::vector<TentativeCorrespExt>::iterator ptr =  in_matches.begin();
      for (int l=0; l<n_tents; l++,ptr++)
        {
          double u[18],err[3];
          u[0] = ptr->first.reproj_kp.x;
          u[1] = ptr->first.reproj_kp.y;
          u[2] = 1.0;

          u[3] = ptr->second.reproj_kp.x;
          u[4] = ptr->second.reproj_kp.y;
          u[5] = 1.0;

          if (geom_error_func == USACConfig::ERR_SYMMETRIC_TRANSFER_SUM) {
              err[0] = calcSymmetricTransferErrorSum(u,u+3,H, H_inv);

            } else if (geom_error_func == USACConfig::ERR_SAMPSON) {
              err[0] = calc2DHomogSampsonErr(u,u+3,H);
            }

          u[6] = u[0]+k_sigma*ptr->first.reproj_kp.a12*ptr->first.reproj_kp.s;
          u[7] = u[1]+k_sigma*ptr->first.reproj_kp.a22*ptr->first.reproj_kp.s;
          u[8] = 1.0;

          u[9]  = u[3]+k_sigma*ptr->second.reproj_kp.a12*ptr->second.reproj_kp.s;
          u[10] = u[4]+k_sigma*ptr->second.reproj_kp.a22*ptr->second.reproj_kp.s;
          u[11] = 1.0;

          if (geom_error_func == USACConfig::ERR_SYMMETRIC_TRANSFER_SUM) {
              err[1] = calcSymmetricTransferErrorSum(u+6,u+9,H, H_inv);

            } else if (geom_error_func == USACConfig::ERR_SAMPSON) {
              err[1] = calc2DHomogSampsonErr(u+6,u+9,H);
            }

          u[12] = u[0]+k_sigma*ptr->first.reproj_kp.a11*ptr->first.reproj_kp.s;
          u[13] = u[1]+k_sigma*ptr->first.reproj_kp.a21*ptr->first.reproj_kp.s;
          u[14] = 1.0;

          u[15] = u[3]+k_sigma*ptr->second.reproj_kp.a11*ptr->second.reproj_kp.s;
          u[16] = u[4]+k_sigma*ptr->second.reproj_kp.a21*ptr->second.reproj_kp.s;
          u[17] = 1.0;

          if (geom_error_func == USACConfig::ERR_SYMMETRIC_TRANSFER_SUM) {
              err[2] = calcSymmetricTransferErrorSum(u+12,u+15,H, H_inv);

            } else if (geom_error_func == USACConfig::ERR_SAMPSON) {
              err[2] = calc2DHomogSampsonErr(u+12,u+15,H);
            }

          double sumErr=sqrt(err[0]) +sqrt(err[1]) + sqrt(err[2]);
          if (sumErr > affineFerror)
            {
              good_pts[l]=0;
              bad_pts++;
            }
        }
      good_matches.reserve(n_tents - bad_pts);
      for (int l=0; l<n_tents; l++)
        if (good_pts[l]) good_matches.push_back(in_matches[l]);
      res = good_matches;
    }
  else res = in_matches;
  return res.size();
}

void AddMatchingsToList(TentativeCorrespListExt &tent_list, TentativeCorrespListExt &new_tents)
{
  int size = (int)tent_list.TCList.size();
  unsigned int new_size = size + (int)new_tents.TCList.size();
  std::vector<TentativeCorrespExt>::iterator ptr =new_tents.TCList.begin();
  for (unsigned int i=size; i< new_size; i++, ptr++)
    tent_list.TCList.push_back(*ptr);
}

int MatchFlannFGINN(const AffineRegionList &list1, const AffineRegionList &list2, TentativeCorrespListExt &corresp,const MatchPars &par, const int nn)
{
  double sqminratio = par.currMatchRatio* par.currMatchRatio;
  double contrDistSq = par.contradDist *par.contradDist;
  unsigned int i,j;
  int matches = 0;
  if (list1.size() == 0) return 0;
  if (list2.size() == 0) return 0;

  unsigned int desc_size = list1[0].desc.vec.size();

  corresp.TCList.reserve((int)(list1.size()/10));

  cv::Mat keys1,keys2;
  keys1 = cv::Mat(list1.size(), desc_size, CV_32F);
  keys2 = cv::Mat(list2.size(), desc_size, CV_32F);

  for (i=0; i <list1.size(); i++)
    {
      float* Row = keys1.ptr<float>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = list1[i].desc.vec[j];
    }

  for (i=0; i <list2.size(); i++)
    {
      float* Row = keys2.ptr<float>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = list2[i].desc.vec[j];
    }

  cv::flann::Index tree = GenFLANNIndex(keys2,par.vector_matcher,par.vector_dist,par.kd_trees);

  cv::Mat indices;
  cv::Mat dists;


  cv::flann::SearchParams SearchParams1(par.knn_checks);
  tree.knnSearch(keys1, indices, dists, nn, SearchParams1);

  if (sqminratio >= 1.0) //to get all points (for example, for calculating PDF)
    {
      for (i=0; i< list1.size(); i++)
        {
          int* indicesRow=indices.ptr<int>(i);
          float* distsRow=dists.ptr<float>(i);
          for (int j=1; j<nn; j++)
            {
              double ratio = distsRow[0]/distsRow[j];
              double dist1 = distanceSq(list2[indicesRow[0]].reproj_kp,list2[indicesRow[j]].reproj_kp);
              if ((j == nn-1) || (dist1 > contrDistSq) /*|| (ratio <= sqminratio) */)
                {
                  TentativeCorrespExt tmp_corr;
                  tmp_corr.first = list1[i];
                  tmp_corr.second = list2[indicesRow[0]];
#ifdef USE_SECOND_BAD
                  tmp_corr.secondbad = list2[indicesRow[j]];
                  tmp_corr.secondbadby2ndcl = list2[indicesRow[1]];
                  tmp_corr.d2by2ndcl = distsRow[1];

#endif
                  tmp_corr.d1 = distsRow[0];
                  tmp_corr.d2 = distsRow[j];
                  tmp_corr.ratio = sqrt(ratio);
                  corresp.TCList.push_back(tmp_corr);
                  matches++;
                  break;
                };
            }
        }

    }
  else
    {
      for (i=0; i< list1.size(); i++)
        {
          int* indicesRow=indices.ptr<int>(i);
          float* distsRow=dists.ptr<float>(i);
          for (int j=1; j<nn; j++)
            {
              double ratio = distsRow[0]/distsRow[j];
              if ((ratio <= sqminratio ))// || (distsRow[0] <= (float)par.matchDistanceThreshold))
                {
                  TentativeCorrespExt tmp_corr;
                  tmp_corr.first = list1[i];
                  tmp_corr.second = list2[indicesRow[0]];
#ifdef USE_SECOND_BAD
                  tmp_corr.secondbad = list2[indicesRow[j]];
                  tmp_corr.secondbadby2ndcl = list2[indicesRow[1]];
                  tmp_corr.d2by2ndcl = distsRow[1];
#endif
                  tmp_corr.d1 = distsRow[0];
                  tmp_corr.d2 = distsRow[j];
                  tmp_corr.ratio = sqrt(ratio);
                  corresp.TCList.push_back(tmp_corr);
                  matches++;
                  break;
                };
              double dist1 = distanceSq(list2[indicesRow[0]].reproj_kp,list2[indicesRow[j]].reproj_kp);
              if (dist1 > contrDistSq) break; //first contradictive
            }
        }
    }
  return matches;
}

int MatchFLANNDistance(const AffineRegionList &list1, const AffineRegionList &list2, TentativeCorrespListExt &corresp,const MatchPars &par, const int nn)
{

  int max_distance = (int)float(par.matchDistanceThreshold);

  unsigned int i,j;
  int matches = 0;
  if (list1.size() == 0) return 0;
  if (list2.size() == 0) return 0;

  unsigned int desc_size = list1[0].desc.vec.size();

  corresp.TCList.clear();
  corresp.TCList.reserve((int)(list1.size()/10));

  cv::Mat keys1,keys2;
  keys1 = cv::Mat(list1.size(), desc_size, CV_8U);
  keys2 = cv::Mat(list2.size(), desc_size, CV_8U);

  for (i=0; i <list1.size(); i++)
    {
      unsigned char* Row = keys1.ptr<unsigned char>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = floor(list1[i].desc.vec[j]);
    }

  for (i=0; i <list2.size(); i++)
    {
      unsigned char* Row = keys2.ptr<unsigned char>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = floor(list2[i].desc.vec[j]);
    }
  cv::flann::SearchParams SearchParams1(par.knn_checks);
  cv::flann::Index tree = GenFLANNIndex(keys2,par.binary_matcher,par.binary_dist,par.kd_trees);

  //  cv::flann::Index tree(keys2,setFlannIndexParams(par.binary_matcher,par.kd_trees),par.binary_dist);
  cv::Mat indices, dists;

  tree.knnSearch(keys1, indices, dists, 2, SearchParams1);

  for (i=0; i<list1.size(); i++)
    {
      int* indicesRow=indices.ptr<int>(i);
      int* distsRow=dists.ptr<int>(i);
      if (distsRow[0] <= max_distance)
        {
          TentativeCorrespExt tmp_corr;
          tmp_corr.first = list1[i];
          tmp_corr.second = list2[indicesRow[0]];
          tmp_corr.d1 = distsRow[0];
          tmp_corr.d2 = distsRow[1];
          tmp_corr.ratio = (double)tmp_corr.d1 / (double)tmp_corr.d2;
          corresp.TCList.push_back(tmp_corr);
          matches++;
        }
    }

  tree.release();
  return matches;
}

template<typename T>
void USACParamsFromCMPParams(const RANSACPars pars, T &cfg)
{
  cfg.common.geomErrFunc = USACConfig::ERR_SYMMETRIC_TRANSFER_SUM;

  if (pars.errorType == RANSAC_error_t::SAMPSON) {
      cfg.common.geomErrFunc = USACConfig::ERR_SAMPSON;
    } else if (pars.errorType == RANSAC_error_t::SYMM_MAX) {
      cfg.common.geomErrFunc = USACConfig::ERR_SYMMETRIC_TRANSFER_MAX;
    } else if (pars.errorType == RANSAC_error_t::SYMM_SUM) {
      cfg.common.geomErrFunc = USACConfig::ERR_SYMMETRIC_TRANSFER_SUM;
    }
  cfg.common.confThreshold = pars.confidence;
  cfg.common.inlierThreshold = pars.err_threshold;
  cfg.common.maxHypotheses = pars.max_samples;
  cfg.common.check_with_Symmetric = pars.doSymmCheck;
  if (pars.useF)
    {
      cfg.common.minSampleSize = 7;
      cfg.common.maxSolutionsPerSample = 3;
    } else {
      cfg.common.minSampleSize = 4;
      cfg.common.maxSolutionsPerSample = 1;
    }

  cfg.common.prevalidateSample = pars.prevalidateSample;
  cfg.common.prevalidateModel = pars.prevalidateModel;
  cfg.common.testDegeneracy = pars.testDegeneracy;
  if (pars.verifMethod == "SPRT") {
      cfg.common.verifMethod = USACConfig::VERIF_SPRT; //Add to ini-file
    } else {
      cfg.common.verifMethod = USACConfig::VERIF_STANDARD;
    }

  if (pars.localOptimization > 0) {
      cfg.common.localOptMethod = USACConfig::LO_LOSAC;

    } else {
      cfg.common.localOptMethod = USACConfig::LO_NONE;
    }
  if (pars.randomSamplingMethod == "PROSAC") {
      cfg.common.randomSamplingMethod = USACConfig::SAMP_PROSAC; //Add to ini file

    } else if (pars.randomSamplingMethod == "UniformMM") {
      cfg.common.randomSamplingMethod = USACConfig::SAMP_UNIFORM_MM; //Add to ini file
    }
  else { //Uniform
      cfg.common.randomSamplingMethod = USACConfig::SAMP_UNIFORM; //Add to ini file
    }

  //LOSAC

  cfg.losac.innerRansacRepetitions = pars.innerRansacRepetitions;
  cfg.losac.innerSampleSize = pars.innerSampleSize; //add allto ini
  cfg.losac.numStepsIterative = pars.numStepsIterative;
  cfg.losac.thresholdMultiplier = pars.thresholdMultiplier;

  //PROSAC //Add all to ini
  cfg.prosac.beta = pars.prosacBeta;
  cfg.prosac.maxSamples= pars.prosacSamples;
  cfg.prosac.minStopLen = pars.prosacMinStopLen;
  cfg.prosac.nonRandConf = pars.prosacNonRandConf;
  //cfg.prosac.sortedPointIndices = some_ptr;

  //SPRT //Add all to ini
  cfg.sprt.tM = pars.SPRT_tM;
  cfg.sprt.mS = pars.SPRT_mS;
  cfg.sprt.delta = pars.SPRT_delta;
  cfg.sprt.epsilon = pars.SPRT_eps;

}

int USACFiltering(TentativeCorrespListExt &in_corresp, TentativeCorrespListExt &ransac_corresp,double *H, const RANSACPars pars)
{

  unsigned int i;
  unsigned int tent_size = in_corresp.TCList.size();

  USACConfig::GeometricErrorFunction err_function;
  ransac_corresp.TCList.clear();
  if (tent_size >= MIN_POINTS)
    {
      double Hloran[3*3];
      double *u2Ptr = new double[tent_size*6], *u2;
      u2=u2Ptr;
      typedef unsigned char uchar;
      std::vector<TentativeCorrespExt>::iterator ptr1 = in_corresp.TCList.begin();
      for(i=0; i < tent_size; i++, ptr1++)
        {
          *u2Ptr =  ptr1->first.reproj_kp.x;
          u2Ptr++;

          *u2Ptr =  ptr1->first.reproj_kp.y;
          u2Ptr++;
          *u2Ptr =  1.;
          u2Ptr++;

          *u2Ptr =  ptr1->second.reproj_kp.x;
          u2Ptr++;

          *u2Ptr =  ptr1->second.reproj_kp.y;
          u2Ptr++;
          *u2Ptr =  1.;
          u2Ptr++;
        };
      if (pars.useF)
        {
          FundMatrixEstimator* fund = new FundMatrixEstimator;
          ConfigParamsFund cfg;
          USACParamsFromCMPParams(pars,cfg);

          if (tent_size <=20) cfg.common.maxHypotheses = 1000;

          err_function = cfg.common.geomErrFunc;
          cfg.common.numDataPoints = tent_size;

          // initialize the USAC parameters, either from a config file, or from your application
          fund->initParamsUSAC(cfg);

          // get input data points/prosac ordering data (if required)
          // set up point_data, cfg.common.numDataPoints, cfg.prosac.sortedPointIndices

          // set up the estimation problem
          fund->initDataUSAC(cfg);
          fund->initProblem(cfg, u2);

          // solve
          if (!fund->solve())
            {
              std::cerr << "Cannot solve problem!" << std::endl;
            }
          for (unsigned int i = 0; i < 3; ++i)
            {
              for (unsigned int j = 0; j < 3; ++j)
                {
                  Hloran[3*i+j] = fund->final_model_params_[3*i+j];
                }
            }

          // writing ransac matchings list
          std::vector<TentativeCorrespExt>::iterator ptr2 = in_corresp.TCList.begin();
          if (!pars.justMarkOutliers)
            {
              for(i=0; i < tent_size; i++, ptr2++)
                {
                  ptr2->isTrue=fund->usac_results_.inlier_flags_[i] > 0;
                  if (ptr2->isTrue)
                    ransac_corresp.TCList.push_back(*ptr2);
                };
            }
          else  {
              for(i=0; i < tent_size; i++, ptr2++)
                {
                  ptr2->isTrue=fund->usac_results_.inlier_flags_[i] > 0;
                  ransac_corresp.TCList.push_back(*ptr2);
                };
            }

          // cleanup

          fund->cleanupProblem();
          delete fund;

        }  else {

          HomogEstimator* homog = new HomogEstimator;
          ConfigParamsHomog cfg;
          USACParamsFromCMPParams(pars,cfg);
          if (tent_size <=20) cfg.common.maxHypotheses = 1000;
          err_function = cfg.common.geomErrFunc;
          cfg.common.numDataPoints = tent_size;
          // initialize the USAC parameters, either from a config file, or from your application
          homog->initParamsUSAC(cfg);

          // get input data points/prosac ordering data (if required)
          // set up point_data, cfg.common.numDataPoints, cfg.prosac.sortedPointIndices

          // set up the estimation problem
          homog->initDataUSAC(cfg);
          homog->initProblem(cfg, u2);

          // solve
          if (!homog->solve())
            {
              std::cerr << "Cannot solve problem!" << std::endl;
            }
          for (unsigned int i = 0; i < 3; ++i) {
              for (unsigned int j = 0; j < 3; ++j) {
                  Hloran[3*i+j] = homog->final_model_params_[3*i+j];
                }
            }

          // writing ransac matchings list
          std::vector<TentativeCorrespExt>::iterator ptr2 = in_corresp.TCList.begin();
          if (!pars.justMarkOutliers) {
              for(i=0; i < tent_size; i++, ptr2++) {
                  //   std::cerr << homog->usac_results_.inlier_flags_[i] << std::endl;
                  ptr2->isTrue=homog->usac_results_.inlier_flags_[i] > 0;
                  if (ptr2->isTrue) {
                      ransac_corresp.TCList.push_back(*ptr2);

                    }
                };
            }
          else  {
              for(i=0; i < tent_size; i++, ptr2++) {
                  ptr2->isTrue=homog->usac_results_.inlier_flags_[i] > 0;
                  ransac_corresp.TCList.push_back(*ptr2);
                };
            }
          // cleanup
          homog->cleanupProblem();
          delete homog;
        }

      delete [] u2;

      //Empirical checks
      if (!(pars.useF)) //H
        {
          cv::Mat Hlor(3,3,CV_64F, Hloran);
          cv::Mat Hinv(3,3,CV_64F);
          cv::invert(Hlor.t(),Hinv, cv::DECOMP_LU);
          double* HinvPtr = (double*)Hinv.data;

          int HIsNotZeros = 0;
          for (i=0; i<9; i++)
            HIsNotZeros = (HIsNotZeros || (HinvPtr[i] != 0.0));
          if (!HIsNotZeros)
            {
              ransac_corresp.TCList.clear();
              return 0;
            }
          for (i=0; i<9; i++)
            {
              ransac_corresp.H[i]=Hloran[i];
              H[i] = Hloran[i];
              //              ransac_corresp.H[i]=HinvPtr[i];
              //              H[i] = HinvPtr[i];
            }
          ///
          TentativeCorrespListExt checked_corresp;
          Hinv=Hinv.t();
          H_LAF_check_USAC(ransac_corresp.TCList,Hloran,HinvPtr,checked_corresp.TCList,
                           3.0*pars.HLAFCoef*pars.err_threshold, err_function);

          if (checked_corresp.TCList.size() < MIN_POINTS)
            checked_corresp.TCList.clear();

          std::cerr << checked_corresp.TCList.size() << " out of " << ransac_corresp.TCList.size() << " left after H-LAF-check" << std::endl;
          ransac_corresp.TCList = checked_corresp.TCList;

        }
      else   //F
        {
          TentativeCorrespListExt checked_corresp;
          F_LAF_check_USAC(ransac_corresp.TCList,Hloran,
                           checked_corresp.TCList,
                           pars.LAFCoef*pars.err_threshold, err_function);
          if (checked_corresp.TCList.size() < MIN_POINTS)
            checked_corresp.TCList.clear();

          std::cerr << checked_corresp.TCList.size() << " out of " << ransac_corresp.TCList.size() << " left after LAF-check" << std::endl;
          ransac_corresp.TCList = checked_corresp.TCList;
          for (i=0; i<9; i++)
            ransac_corresp.H[i]=Hloran[i];
        }
    }
  else
    {
      if (VERB)  cout << tent_size << " points is not enought points to do RANSAC" << endl;
      ransac_corresp.TCList.clear();
      return 0;
    }
  return ransac_corresp.TCList.size();
}

void DrawMatches(const cv::Mat &in_img1,const cv::Mat &in_img2, cv::Mat &out_img1, cv::Mat &out_img2,const cv::Mat &H1,
                 TentativeCorrespListExt matchings,
                 const int DrawCentersOnly,
                 const int ReprojectToOneImage,
                 const int r1,
                 const int r2,
                 const int drawEpipolarLines,
                 const int useSCV,
                 const double LAFcoef,
                 const cv::Scalar color1,
                 const cv::Scalar color2)
{
  cv::Mat out_tmp1, out_tmp2;
  double k_scale = 3.0;//3 sigma
  double *H = (double*)H1.data;
  double ransac_th = 2*2.0;
  double affineFerror = LAFcoef * ransac_th;
  double Ht[9];
  Ht[0] = H[0];
  Ht[1] = H[3];
  Ht[2] = H[6];
  Ht[3] = H[1];
  Ht[4] = H[4];
  Ht[5] = H[7];
  Ht[6] = H[2];
  Ht[7] = H[5];
  Ht[8] = H[8];

  /////
  double e1[3],e2[3];
  std::vector< std::vector<double> > Ferrors(matchings.TCList.size());
  for (unsigned int i=0; i<Ferrors.size(); i++)
    Ferrors[i].resize(3);

  if (affineFerror > 0)
    GetEpipoles(H,e1,e2);
  int bad_count = 0;
  if (ReprojectToOneImage)
    {
      //  double *H = (double*)H1.data;
      cv::Mat h1inv(3,3,CV_64F);
      cv::invert(H1,h1inv,cv::DECOMP_LU);
      double *Hinv = (double*)h1inv.data;

      if (in_img1.channels() != 3)
        cv::cvtColor(in_img1,out_tmp1,CV_GRAY2RGB);
      else
        out_tmp1=in_img1.clone();
      if (in_img2.channels() != 3)
        cv::cvtColor(in_img2,out_tmp2,CV_GRAY2RGB);
      else
        out_tmp2=in_img2.clone();

      std::vector<TentativeCorrespExt>::iterator ptrOut = matchings.TCList.begin();
      if(!DrawCentersOnly)
        {

          double cosine_sine_table[44];
          double cosine_sine_table3d[66];
          cosine_sine_table[21]=0;
          cosine_sine_table[43]=0;
          for (int l=0; l<21; l++)
            {
              cosine_sine_table[l]=cos(l*M_PI/10);
              cosine_sine_table[22+l]=sin(l*M_PI/10);
            }
          for (int l=0; l<44; l++)
            cosine_sine_table3d[l]=cosine_sine_table[l];
          for (int l=44; l<66; l++)
            cosine_sine_table3d[l]=1.0;

          cv::Mat cs_table(2,22,CV_64F, cosine_sine_table);
          cv::Mat cs_table3d(3,22,CV_64F, cosine_sine_table3d);

          /// Image 1
          ptrOut = matchings.TCList.begin();
          for(unsigned int i=0; i < matchings.TCList.size(); i++, ptrOut++)
            {

              double A[4]= {k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a11, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a12,
                            k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a21, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a22
                           };
              cv::Mat A1(2,2,CV_64F, A);
              cv::Mat X;
              cv::gemm(A1,cs_table,1,A1,0,X);
              vector<cv::Point> contour;
              for (int k=0; k<22; k++)
                contour.push_back(cv::Point(floor(X.at<double>(0,k)+ptrOut->first.reproj_kp.x),floor(X.at<double>(1,k)+ptrOut->first.reproj_kp.y)));

              const cv::Point *pts = (const cv::Point*) cv::Mat(contour).data;
              int npts = cv::Mat(contour).rows;
              polylines(out_tmp1, &pts,&npts, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color1,// colour RGB ordering (here = green)
                        r1,                     // line thickness
                        CV_AA, 0);
              double B[9]= {k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a11, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a12, ptrOut->second.reproj_kp.x,
                            k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a21, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a22, ptrOut->second.reproj_kp.y,
                            0, 0, 1
                           };
              cv::Mat B1(3,3,CV_64F, B);
              cv::gemm(h1inv,B1,1,B1,0,B1);
              cv::Mat X2;
              cv::gemm(B1,cs_table3d,1,B1,0,X2);
              vector<cv::Point> contour2;
              for (int k=0; k<22; k++)
                contour2.push_back(cv::Point(floor(X2.at<double>(0,k) / X2.at<double>(2,k)),floor(X2.at<double>(1,k) / X2.at<double>(2,k))));

              const cv::Point *pts2 = (const cv::Point*) cv::Mat(contour2).data;
              int npts2 = cv::Mat(contour2).rows;
              polylines(out_tmp1, &pts2,&npts2, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color2,
                        r2,                     // line thickness
                        CV_AA, 0);

            }
          /// Image 2
          ptrOut = matchings.TCList.begin();
          for(unsigned int i=0; i < matchings.TCList.size(); i++, ptrOut++)
            {
              double A[4]= {k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a11, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a12,
                            k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a21, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a22
                           };
              cv::Mat A1(2,2,CV_64F, A);
              cv::Mat X;
              cv::gemm(A1,cs_table,1,A1,0,X);
              vector<cv::Point> contour;
              for (int k=0; k<22; k++)
                contour.push_back(cv::Point(floor(X.at<double>(0,k)+ptrOut->second.reproj_kp.x),floor(X.at<double>(1,k)+ptrOut->second.reproj_kp.y)));

              const cv::Point *pts = (const cv::Point*) cv::Mat(contour).data;
              int npts = cv::Mat(contour).rows;
              polylines(out_tmp2, &pts,&npts, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color1,// colour RGB ordering (here = green)
                        r1,                     // line thickness
                        CV_AA, 0);
              double B[9]= {k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a11, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a12, ptrOut->first.reproj_kp.x,
                            k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a21, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a22, ptrOut->first.reproj_kp.y,
                            0, 0, 1
                           };
              cv::Mat B1(3,3,CV_64F, B);
              cv::gemm(H1,B1,1,B1,0,B1);
              cv::Mat X2;
              cv::gemm(B1,cs_table3d,1,B1,0,X2);

              vector<cv::Point> contour2;
              for (int k=0; k<22; k++)
                contour2.push_back(cv::Point(floor(X2.at<double>(0,k) / X2.at<double>(2,k)),floor(X2.at<double>(1,k) / X2.at<double>(2,k))));

              const cv::Point *pts2 = (const cv::Point*) cv::Mat(contour2).data;
              int npts2 = cv::Mat(contour2).rows;
              polylines(out_tmp2, &pts2,&npts2, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color2,
                        r2,                     // line thickness
                        CV_AA, 0);
            }
        }
      /// Draw centers
      ptrOut = matchings.TCList.begin();
      //Image1
      for(unsigned int i=0; i < matchings.TCList.size(); i++, ptrOut++)
        {
          cv::circle(out_tmp1, cv::Point(int(ptrOut->first.reproj_kp.x),int(ptrOut->first.reproj_kp.y)),r1+2,color1,-1); //draw original points
          double xa,ya;
          xa = (Hinv[0]*ptrOut->second.reproj_kp.x+Hinv[1]*ptrOut->second.reproj_kp.y+Hinv[2])/(Hinv[6]*ptrOut->second.reproj_kp.x+Hinv[7]*ptrOut->second.reproj_kp.y+Hinv[8]);
          ya = (Hinv[3]*ptrOut->second.reproj_kp.x+Hinv[4]*ptrOut->second.reproj_kp.y+Hinv[5])/(Hinv[6]*ptrOut->second.reproj_kp.x+Hinv[7]*ptrOut->second.reproj_kp.y+Hinv[8]);
          cv::circle(out_tmp1, cv::Point(int(xa),int(ya)),r2,color2,-1); //draw correpspondent point
          cv::line(out_tmp1,cv::Point(int(xa),int(ya)),cv::Point(int(ptrOut->first.reproj_kp.x),int(ptrOut->first.reproj_kp.y)), color2);
        }
      //Image2
      ptrOut = matchings.TCList.begin();
      for(unsigned int i=0; i < matchings.TCList.size(); i++, ptrOut++)
        {
          cv::circle(out_tmp2, cv::Point(int(ptrOut->second.reproj_kp.x),int(ptrOut->second.reproj_kp.y)),r1+2,color1,-1); //draw original points
          double xa,ya;
          xa = (H[0]*ptrOut->first.reproj_kp.x+H[1]*ptrOut->first.reproj_kp.y+H[2])/(H[6]*ptrOut->first.reproj_kp.x+H[7]*ptrOut->first.reproj_kp.y+H[8]);
          ya = (H[3]*ptrOut->first.reproj_kp.x+H[4]*ptrOut->first.reproj_kp.y+H[5])/(H[6]*ptrOut->first.reproj_kp.x+H[7]*ptrOut->first.reproj_kp.y+H[8]);
          cv::circle(out_tmp2, cv::Point(int(xa),int(ya)),r2,color2,-1); //draw correpspondent point
          cv::line(out_tmp2,cv::Point(int(xa),int(ya)),cv::Point(int(ptrOut->second.reproj_kp.x),int(ptrOut->second.reproj_kp.y)), color2);
        }
    }
  else
    {
      int n_tents = matchings.TCList.size();
      std::vector<int> good_pts(n_tents);
      for (int a=0; a<n_tents; a++)
        good_pts[a]=1; //initialization

      int w1 = in_img1.cols;
      int h1 = in_img1.rows;

      int w2 = in_img2.cols;
      int h2 = in_img2.rows;



      unsigned int i;
      cv::Scalar color_corr = color2;
      int sep=20;
      cv::Mat roiImg1 = in_img1(cv::Rect(0,0,in_img1.cols,in_img1.rows));
      cv::Mat roiImg2 = in_img2(cv::Rect(0,0,in_img2.cols,in_img2.rows));

      out_tmp1 = cv::Mat (max(in_img1.rows,in_img2.rows),in_img1.cols+in_img2.cols+sep,in_img1.type(), cv::Scalar(255,255,255));

      cv::Mat roiImgResult_Left = out_tmp1(cv::Rect(0,0,in_img1.cols,in_img1.rows));
      cv::Mat roiImgResult_Right = out_tmp1(cv::Rect(in_img1.cols+sep,0,in_img2.cols,in_img2.rows));
      roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
      roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult

      out_tmp2 = cv::Mat(in_img1.rows+in_img2.rows+sep, max(in_img1.cols,in_img2.cols),in_img2.type(),cv::Scalar(255,255,255));

      cv::Mat roiImgResult_Up = out_tmp2(cv::Rect(0,0,in_img1.cols,in_img1.rows));
      cv::Mat roiImgResult_Down = out_tmp2(cv::Rect(0,in_img1.rows+sep, in_img2.cols,in_img2.rows));
      roiImg1.copyTo(roiImgResult_Up); //Img1 will be on the left of imgResult
      roiImg2.copyTo(roiImgResult_Down); //Img2 will be on the right of imgResult

      if(!DrawCentersOnly)
        {
          double cosine_sine_table[44];
          cosine_sine_table[21]=0;
          cosine_sine_table[43]=0;

          for (int l=0; l<21; l++)
            {
              cosine_sine_table[l]=cos(l*M_PI/10);
              cosine_sine_table[22+l]=sin(l*M_PI/10);
            }
          cv::Mat cs_table(2,22,CV_64F, cosine_sine_table);

          /// Image 1 Regions
          std::vector<TentativeCorrespExt>::iterator ptrOut = matchings.TCList.begin();
          for(i=0; i < matchings.TCList.size(); i++, ptrOut++)
            {
              if (!good_pts[i])
                {
                  color_corr = cv::Scalar(0,0,255);
                }
              else color_corr = color2;

              double A[4]= {k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a11, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a12,
                            k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a21, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a22
                           };
              cv::Mat A1(2,2,CV_64F, A);
              cv::Mat X;
              cv::gemm(A1,cs_table,1,A1,0,X);
              vector<cv::Point> contour;
              for (int l=0; l<22; l++)
                contour.push_back(cv::Point(floor(X.at<double>(0,l)+ptrOut->first.reproj_kp.x),floor(X.at<double>(1,l)+ptrOut->first.reproj_kp.y)));
              const cv::Point *pts = (const cv::Point*) cv::Mat(contour).data;
              int npts = cv::Mat(contour).rows;

              polylines(out_tmp1, &pts,&npts, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color_corr,// colour RGB ordering (here = green)
                        r1,                     // line thickness
                        CV_AA, 0);

              double A2[4]= {k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a11, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a12,
                             k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a21, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a22
                            };
              A1 = cv::Mat (2,2,CV_64F, A2);
              cv::gemm(A1,cs_table,1,A1,0,X);
              contour.clear();
              for (int l=0; l<22; l++)
                contour.push_back(cv::Point(floor(X.at<double>(0,l)+ptrOut->second.reproj_kp.x+in_img1.cols+sep),floor(X.at<double>(1,l)+ptrOut->second.reproj_kp.y)));

              pts = (const cv::Point*) cv::Mat(contour).data;

              npts = cv::Mat(contour).rows;
              polylines(out_tmp1, &pts,&npts, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color_corr,
                        r2,                     // line thickness
                        CV_AA, 0);


            }
          /// Image 2
          ptrOut = matchings.TCList.begin();
          for(i=0; i < matchings.TCList.size(); i++, ptrOut++)
            {
              if (!good_pts[i]) color_corr = cv::Scalar(0,0,255);
              else color_corr = color2;

              double A[4]= {k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a11, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a12,
                            k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a21, k_scale*ptrOut->first.reproj_kp.s*ptrOut->first.reproj_kp.a22
                           };
              cv::Mat A1(2,2,CV_64F, A);
              cv::Mat cs_table(2,22,CV_64F, cosine_sine_table);
              cv::Mat X;
              cv::gemm(A1,cs_table,1,A1,0,X);
              vector<cv::Point> contour;
              for (int l=0; l<22; l++)
                contour.push_back(cv::Point(floor(X.at<double>(0,l)+ptrOut->first.reproj_kp.x),floor(X.at<double>(1,l)+ptrOut->first.reproj_kp.y)));
              const cv::Point *pts = (const cv::Point*) cv::Mat(contour).data;
              int npts = cv::Mat(contour).rows;
              polylines(out_tmp2, &pts,&npts, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color_corr,// colour RGB ordering (here = green)
                        r1,                     // line thickness
                        CV_AA, 0);
              double A2[4]= {k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a11, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a12,
                             k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a21, k_scale*ptrOut->second.reproj_kp.s*ptrOut->second.reproj_kp.a22
                            };
              A1 = cv::Mat (2,2,CV_64F, A2);
              cs_table =  cv::Mat (2,22,CV_64F, cosine_sine_table);
              cv::gemm(A1,cs_table,1,A1,0,X);
              contour.clear();
              for (int l=0; l<22; l++)
                contour.push_back(cv::Point(floor(X.at<double>(0,l)+ptrOut->second.reproj_kp.x),floor(X.at<double>(1,l)+ptrOut->second.reproj_kp.y+in_img1.rows+sep)));

              pts = (const cv::Point*) cv::Mat(contour).data;
              npts = cv::Mat(contour).rows;
              polylines(out_tmp2, &pts,&npts, 1,
                        false,                  // draw closed contour (i.e. joint end to start)
                        color_corr,
                        r2,                     // line thickness
                        CV_AA, 0);

            }
        }

      std::vector<TentativeCorrespExt>::iterator ptrOut = matchings.TCList.begin();

      for(i=0; i < matchings.TCList.size(); i++, ptrOut++)
        {
          double xa,ya;
          xa = in_img1.cols+sep + ptrOut->second.reproj_kp.x;
          ya = ptrOut->second.reproj_kp.y;

          cv::circle(out_tmp1, cv::Point(int(ptrOut->first.reproj_kp.x),int(ptrOut->first.reproj_kp.y)),r1+2,color1,-1); //draw original points
          cv::circle(out_tmp1, cv::Point(int(xa),int(ya)),r1+2,color1,-1); //draw correpspondent point
          if (good_pts[i]) color_corr = color2;
          else color_corr = cv::Scalar(0,0,255); //red color for non-scv matches
          if (drawEpipolarLines)

            {
              double l[3], l2[3], k,b,k2,b2;
              double pt[3], pt2[3];

              pt[0] = ptrOut->first.reproj_kp.x;
              pt[1] = ptrOut->first.reproj_kp.y;
              pt[2] = 1.0;

              pt2[0] = ptrOut->second.reproj_kp.x;
              pt2[1] = ptrOut->second.reproj_kp.y;
              pt2[2] = 1.0;

              GetEpipolarLineF(H,pt2,l,k,b);
              GetEpipolarLineF(Ht,pt,l2,k2,b2);

              cv::Point sp,ep;
              cv::Scalar EpLineColor = cv::Scalar(255,255,0);
              cv::Rect img1rect1 = cv::Rect(0, 0, w1, h1);
              cv::Rect img1rect2 = cv::Rect(w1+sep, 0, w2, h2);

              cv::Rect img2rect1 = cv::Rect(0, 0, w1, h1);
              cv::Rect img2rect2 = cv::Rect(0, h1+sep, w2, h2);


              sp = cv::Point(0,int(b));
              ep = cv::Point(w1,int(k*w1+b));
              cv::clipLine(img1rect1,sp,ep);
              cv::line(out_tmp1,sp,ep,EpLineColor);

              sp = cv::Point(w1+sep,int(b2));
              ep = cv::Point(w2+w1+sep,int(k2*w2+b2));
              cv::clipLine(img1rect2,sp,ep);
              cv::line(out_tmp1,sp,ep,EpLineColor);

              sp = cv::Point(0,int(b));
              ep = cv::Point(w1,int(k*w1+b));
              cv::clipLine(img2rect1,sp,ep);
              cv::line(out_tmp2,sp,ep,EpLineColor);

              sp = cv::Point(0,int(b2)+h1+sep);
              ep = cv::Point(w2,int(k2*w2+b2)+h1+sep);
              cv::clipLine(img2rect2,sp,ep);
              cv::line(out_tmp2,sp,ep,EpLineColor);

            }

          cv::line(out_tmp1,cv::Point(int(xa),int(ya)),cv::Point(int(ptrOut->first.reproj_kp.x),int(ptrOut->first.reproj_kp.y)), color_corr);


          xa = ptrOut->second.reproj_kp.x;
          ya = in_img1.rows+sep +ptrOut->second.reproj_kp.y;
          cv::circle(out_tmp2, cv::Point(int(ptrOut->first.reproj_kp.x),int(ptrOut->first.reproj_kp.y)),r1+2,color1,-1); //draw original points
          cv::circle(out_tmp2, cv::Point(int(xa),int(ya)),r1+2,color1,-1); //draw correpspondent point
          cv::line(out_tmp2,cv::Point(int(xa),int(ya)),cv::Point(int(ptrOut->first.reproj_kp.x),int(ptrOut->first.reproj_kp.y)), color_corr);
        }
    }
  out_img1 = out_tmp1.clone();
  out_img2 = out_tmp2.clone();
}


void WriteMatchings(TentativeCorrespListExt &match, std::ostream &out1, int writeWithRatios)
{
  out1 << (int) match.TCList.size() << std::endl;
  std::vector<TentativeCorrespExt>::iterator ptr = match.TCList.begin();
  if (writeWithRatios)
    {
      for(int i=0; i < (int) match.TCList.size(); i++, ptr++)
        out1 << ptr->first.reproj_kp.x << " " << ptr->first.reproj_kp.y << " " << ptr->second.reproj_kp.x << " " << ptr->second.reproj_kp.y << " "
             << sqrt(ptr->d1 / ptr->d2) << " " << sqrt(ptr->d1 / ptr->d2by2ndcl) << " " << ptr->isTrue << std::endl;
    }
  else
    {
      for(int i=0; i < (int) match.TCList.size(); i++, ptr++)
        out1 << ptr->first.reproj_kp.x << " " << ptr->first.reproj_kp.y << " " << ptr->second.reproj_kp.x << " " << ptr->second.reproj_kp.y  << std::endl;
    }
}

void DuplicateFiltering(TentativeCorrespListExt &in_corresp, const double r, const int mode)
{
  if (r <= 0) return; //no filtering
  unsigned int i,j;
  unsigned int tent_size = in_corresp.TCList.size();
  double r_sq = r*r;
  double d1_sq, d2_sq;
  vector <char> flag_unique;
  flag_unique = vector <char> (tent_size);
  for (i=0; i<tent_size; i++)
    flag_unique[i] = 1;

  switch (mode) {
    case MODE_RANDOM:
      break;
    case MODE_FGINN:
      {
        std::sort(in_corresp.TCList.begin(),in_corresp.TCList.end(),CompareCorrespondenceByRatio);
        break;
      }
    case MODE_DISTANCE:
      {
        std::sort(in_corresp.TCList.begin(),in_corresp.TCList.end(),CompareCorrespondenceByDistance);
        break;
      }
    case MODE_BIGGER_REGION:
      {
        std::sort(in_corresp.TCList.begin(),in_corresp.TCList.end(),CompareCorrespondenceByScale);
        break;
      }
    default:
      break;
    }

  std::vector<TentativeCorrespExt>::iterator ptr1 = in_corresp.TCList.begin();
  for(i=0; i < tent_size; i++, ptr1++)
    {
      if (flag_unique[i] == 0) continue;
      std::vector<TentativeCorrespExt>::iterator ptr2 = ptr1+1;
      for(j=i+1; j < tent_size; j++, ptr2++)
        {
          if (flag_unique[j] == 0) continue;
          double dx = (ptr1->first.reproj_kp.x - ptr2->first.reproj_kp.x);
          double dy = (ptr1->first.reproj_kp.y - ptr2->first.reproj_kp.y);
          d1_sq = dx*dx+dy*dy;
          if (d1_sq > r_sq)
            continue;
          dx = (ptr1->second.reproj_kp.x - ptr2->second.reproj_kp.x);
          dy = (ptr1->second.reproj_kp.y - ptr2->second.reproj_kp.y);
          d2_sq = dx*dx+dy*dy;
          if (d2_sq <= r_sq)
            flag_unique[j] = 0;
        }
    }
  TentativeCorrespListExt unique_list;
  unique_list.TCList.reserve(0.8*in_corresp.TCList.size());
  for (i=0; i<9; i++)
    unique_list.H[i] = in_corresp.H[i];

  for (i=0; i<tent_size; i++)
    if (flag_unique[i] == 1)
      unique_list.TCList.push_back(in_corresp.TCList[i]);

  in_corresp.TCList = unique_list.TCList;
}

void WriteH(double* H, std::ostream &out1)
{
  out1  << H[0] << " " << H[1] << " " << H[2] << endl
                << H[3] << " " << H[4] << " " << H[5] << endl
                << H[6] << " " << H[7] << " " << H[8] << endl;
}

} // ns mods
