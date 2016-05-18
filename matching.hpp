/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#ifndef MODS_MATCHING_HPP
#define MODS_MATCHING_HPP

#include <opencv2/flann/flann.hpp>
#include "structures.hpp"
#include "descriptors_parameters.hpp"
#include "siftdesc.h"

#define MIN_POINTS 8 //threshold for symmetrical error check
#define USE_SECOND_BAD 1//uncomment if you need to use/output 1st geom.inconsistent region
#include "configuration.hpp"

namespace mods {

struct TentativeCorresp
{
  AffineRegion first;
  AffineRegion second;
};

struct TentativeCorrespExt : TentativeCorresp
{
#ifdef USE_SECOND_BAD
  AffineRegion secondbadby2ndcl;
  AffineRegion secondbad;
#endif
  double d1;
  double d2;
  double d2by2ndcl;
  double d2byDB;
  double ratio;
  int isTrue;
};

//struct Keypoint4Match     //structure for the bruteforce matching only
//{
//  float x,y;
//  int parent_id;           //parent region id (when detect orientation). = -1 when parent is undefined;
//  int group_id;            //group id
//  unsigned char desc[128]; //SIFT descriptor

//};
//struct Keypoint4OverlapMatch
//{
//  double x,y, a11,a12,a21,a22,s;
//};

//typedef std::vector<Keypoint4Match> Keypoint4MatchList;
//typedef std::vector<Keypoint4OverlapMatch> Keypoint4OverlapMatchList;


struct TentativeCorrespList
{
  std::vector<TentativeCorresp> TCList;
  double H[3*3]; // by default H[i] = -1, if no H-estimation done
  TentativeCorrespList()
  {
    for (int i=0; i<9; i++)
      H[i] = -1;
  }

};


struct TentativeCorrespListExt : TentativeCorrespList
{
  std::vector<TentativeCorrespExt> TCList;
  double H[3*3]; // by default H[i] = -1, if no H-estimation done
  TentativeCorrespListExt()
  {
    for (int i=0; i<9; i++)
      H[i] = -1;
  }

};

enum RANSAC_error_t {SAMPSON,SYMM_MAX,SYMM_SUM};

struct MatchPars
{
  std::vector <WhatToMatch> IterWhatToMatch;
  std::map <std::string, double> FGINNThreshold;
  std::map <std::string, double> DistanceThreshold;
  double currMatchRatio;
  double matchDistanceThreshold;

  double contradDist;
  int standard_2nd_closest;
  int kd_trees;
  int knn_checks;
  int RANSACforStopping;
  int minMatches;
  int maxSteps;
  int doBothRANSACgroundTruth;
  int doOverlapMatching;
  double overlapError;
  cvflann::flann_algorithm_t binary_matcher;
  cvflann::flann_algorithm_t vector_matcher;
  cvflann::flann_distance_t binary_dist;
  cvflann::flann_distance_t vector_dist;
  int doDensification;
  double FPRate;
  int useDBforFGINN;
  std::string SIFTDBfile;
  MatchPars()
  {
    SIFTDBfile="100_db.txt";
    useDBforFGINN=0;
    currMatchRatio = -1.0;
    contradDist = 10.0;
    standard_2nd_closest = 0;
    kd_trees = 4;
    knn_checks = 128;
    RANSACforStopping=1;
    minMatches = 15;
    maxSteps = 4;
    doBothRANSACgroundTruth = 1;
    doOverlapMatching = 0;
    overlapError = 0.09;
    binary_matcher = cvflann::FLANN_INDEX_HIERARCHICAL;
    vector_matcher = cvflann::FLANN_INDEX_KDTREE;
    doDensification=0;
    FPRate = 0.8;
    matchDistanceThreshold = 64.0;
  }
};

struct RANSACPars
{
  RANSAC_error_t errorType;
  int useF;
  double err_threshold;
  double confidence;
  int max_samples;
  int localOptimization;
  double LAFCoef;
  double HLAFCoef;
  int doSymmCheck;
  int justMarkOutliers;
  bool prevalidateSample;
  bool prevalidateModel;
  bool testDegeneracy;
  std::string verifMethod;
  std::string randomSamplingMethod;
  int innerRansacRepetitions;
  int innerSampleSize;
  int numStepsIterative;
  double thresholdMultiplier;
  double prosacBeta;
  int prosacSamples;
  int prosacMinStopLen;
  double prosacNonRandConf;
  double SPRT_tM;
  double SPRT_mS;
  double SPRT_delta;
  double SPRT_eps;
  RANSACPars()
  {
    useF=0;
    err_threshold = 2.0;
    confidence = 0.99;
    max_samples = 1e5;
    localOptimization = 1;
    LAFCoef = 3.0;
    HLAFCoef = 10.0;
    errorType = SYMM_SUM;
    doSymmCheck = 0;
    justMarkOutliers=0;

    prevalidateSample = true;
    prevalidateModel = true;
    testDegeneracy = true;
    verifMethod = "Standard";
    randomSamplingMethod = "Uniform";

    //LOSAC
    innerRansacRepetitions = 3;
    innerSampleSize = 12; //add allto ini
    numStepsIterative = 4;
    thresholdMultiplier = 2.0;

    //PROSAC
    prosacBeta = 0.99;
    prosacSamples= 500000;
    prosacMinStopLen = 20;
    prosacNonRandConf = 0.99;

    //SPRT
    SPRT_tM = 100.0;
    SPRT_mS = 1.0;
    SPRT_delta = 0.01;
    SPRT_eps = 0.2;
  }
};
/* Correspondence for drawing: */
typedef std::pair<cv::Point2f,cv::Point2f> corresp;

void AddMatchingsToList(TentativeCorrespListExt &tent_list, TentativeCorrespListExt &new_tents);
cv::flann::Index GenFLANNIndex(cv::Mat keys, cvflann::flann_algorithm_t indexType, cvflann::flann_distance_t dist_type, const int nTrees = 4);

int MatchFlannFGINN(const AffineRegionList &list1, const AffineRegionList &list2,
                  TentativeCorrespListExt &corresp,const MatchPars &par, const int nn=50);

int MatchFLANNDistance(const AffineRegionList &list1, const AffineRegionList &list2,
                  TentativeCorrespListExt &corresp,const MatchPars &par, const int nn=50);

int USACFiltering(TentativeCorrespListExt &in_corresp,
                      TentativeCorrespListExt &out_corresp, double *H,
                      const RANSACPars pars);

//void DuplicateFiltering(TentativeCorrespList &in_corresp, const double r = 3.0);
void DuplicateFiltering(TentativeCorrespListExt &in_corresp, const double r = 3.0, const int mode = MODE_RANDOM);
//Function does pairwise computing of the distance between ellipse centers in 1st and 2nd images.
//If distance^2 < r_sq in both images, correspondences are considered as duplicates and
//second point is deleted.


void DrawMatches(const cv::Mat &in_img1,const cv::Mat &in_img2, cv::Mat &out_img1,cv::Mat &out_img2,const cv::Mat &H,
                 TentativeCorrespListExt matchings,
                 const int DrawCentersOnly = 1,
                 const int ReprojectToOneImage = 1,
                 const int r1=2,
                 const int r2=2,
                 const int drawEpipolarLines =0,
                 const int useSCV=0,
                 const double LAFcoef = 0,
                 const cv::Scalar color1= cv::Scalar(255,0,0),
                 const cv::Scalar color2= cv::Scalar(0,255,0));
void WriteMatchings(TentativeCorrespListExt &match, std::ostream &out1, int writeWithRatios = 0);
//Writes matchings in format: number x1 y1 x2 y2

void WriteH(double* H, std::ostream &out1);
//Writes homography matrix 3*3 into stream or file

} //namespace mods

#endif // MATCHING_HPP
