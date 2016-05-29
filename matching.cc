/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#include "matching.hh"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

int MatchFlannFGINN(const AffineRegionVector &list1, const AffineRegionVector &list2, TentativeCorrespListExt &corresp,const MatchPars &par, const int nn)
{
  double sqminratio = par.currMatchRatio* par.currMatchRatio;
  double contrDistSq = par.contradDist *par.contradDist;
  unsigned int i,j;
  int matches = 0;
  if (list1.size() == 0) return 0;
  if (list2.size() == 0) return 0;

  unsigned int desc_size = list1[0].desc.size();

  corresp.TCList.reserve((int)(list1.size()/10));

  cv::Mat keys1,keys2;
  keys1 = cv::Mat(list1.size(), desc_size, CV_32F);
  keys2 = cv::Mat(list2.size(), desc_size, CV_32F);

  for (i=0; i <list1.size(); i++)
    {
      float* Row = keys1.ptr<float>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = list1[i].desc[j];
    }

  for (i=0; i <list2.size(); i++)
    {
      float* Row = keys2.ptr<float>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = list2[i].desc[j];
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

int MatchFLANNDistance(const AffineRegionVector &list1, const AffineRegionVector &list2, TentativeCorrespListExt &corresp,const MatchPars &par, const int nn)
{

  int max_distance = (int)float(par.matchDistanceThreshold);

  unsigned int i,j;
  int matches = 0;
  if (list1.size() == 0) return 0;
  if (list2.size() == 0) return 0;

  unsigned int desc_size = list1[0].desc.size();

  corresp.TCList.clear();
  corresp.TCList.reserve((int)(list1.size()/10));

  cv::Mat keys1,keys2;
  keys1 = cv::Mat(list1.size(), desc_size, CV_8U);
  keys2 = cv::Mat(list2.size(), desc_size, CV_8U);

  for (i=0; i <list1.size(); i++)
    {
      unsigned char* Row = keys1.ptr<unsigned char>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = floor(list1[i].desc[j]);
    }

  for (i=0; i <list2.size(); i++)
    {
      unsigned char* Row = keys2.ptr<unsigned char>(i);
      for (j=0; j < desc_size; j++)
        Row[j] = floor(list2[i].desc[j]);
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


} // ns mods
