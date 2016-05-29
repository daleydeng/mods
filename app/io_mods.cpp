/*-----------------------------------------------------------*/
/* Copyright 2013-2015, Dmytro Mishkin  ducha.aiki@gmail.com */
/*----------------------------------------------------------*/
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "io_mods.h"
#include "synth_detection.hh"

#ifdef _OPENMP
#include <omp.h>
#endif
void WriteLog(logs log, ostream& out)
{
  out << std::setprecision(3) << log.FinalTime << " ";
  out << log.TrueMatch1st << " ";
  out << log.Tentatives1st << " ";
  out << log.InlierRatio1st*100 << " ";
  out << log.OrientReg1 << " ";
  out << log.OrientReg2 << " ";
  out << log.FinalStep << " ";
  out << std::endl;

}
void WriteTimeLog(TimeLog log, ostream &out,
                  const int writeRelValues,
                  const int writeAbsValues,
                  const int writeDescription)
{
  if(writeDescription)
    {
      out << "Timings: (sec/%) "<< std::endl << "Synth|Detect|Orient|Desc|Match|RANSAC|MISC|Total " << std::endl;
    }
  if (writeAbsValues)
    {
      out << log.SynthTime << " "
          << log.DetectTime << " "
          << log.OrientTime << " "
          << log.DescTime<< " "
          << log.MatchingTime << " "
          << log.RANSACTime << " "
          << log.MiscTime << " "
          << log.TotalTime << std::endl;
    }
  if (writeRelValues)
    {
      out << log.SynthTime/log.TotalTime*100 << " "
          << log.DetectTime/log.TotalTime*100 << " "
          << log.OrientTime/log.TotalTime*100 << " "
          << log.DescTime/log.TotalTime*100 << " "
          << log.MatchingTime/log.TotalTime*100 << " "
          << log.RANSACTime/log.TotalTime*100 << " "
          << log.MiscTime/log.TotalTime*100 << " "
          << log.TotalTime/log.TotalTime*100 << std::endl;
    }
}

void GetMSERPars(ExtremaParams &MSERPars, INIReader &reader,const char* section)
{
  MSERPars.rel_threshold = reader.GetDouble(section, "relativeThreshold", MSERPars.rel_threshold);
  MSERPars.rel_reg_number = reader.GetDouble(section, "relativeRegionsNumber", MSERPars.rel_reg_number);
  MSERPars.reg_number = reader.GetInteger(section, "regionsNumber", MSERPars.reg_number);
  MSERPars.max_area = reader.GetDouble(section, "max_area", MSERPars.max_area);
  MSERPars.min_size = reader.GetInteger(section, "min_size", MSERPars.min_size);
  MSERPars.min_margin = reader.GetInteger(section, "min_margin", MSERPars.min_margin);
  MSERPars.doOnWLD = reader.GetInteger(section, "doOnWLD", MSERPars.doOnWLD);
  MSERPars.doOnNormal = reader.GetInteger(section, "doOnNormal", MSERPars.doOnNormal);
  MSERPars.WLDPar.a = reader.GetDouble(section, "a", MSERPars.WLDPar.a);
  MSERPars.WLDPar.b = reader.GetDouble(section, "b", MSERPars.WLDPar.b);
  MSERPars.WLDPar.g = reader.GetDouble(section, "g", MSERPars.WLDPar.g);

  std::vector< std::string> temp_str;
  reader.GetStringVector(section, "mode",temp_str);
  if (temp_str[0].compare("RelativeTh")==0)
    MSERPars.DetectorMode = RELATIVE_TH;
  else if (temp_str[0].compare("FixedRegNumber")==0)
    MSERPars.DetectorMode = FIXED_REG_NUMBER;
  else if (temp_str[0].compare("NotLessThanRegions")==0)
    MSERPars.DetectorMode = NOT_LESS_THAN_REGIONS;
  else if (temp_str[0].compare("RelativeRegNumber")==0)
    MSERPars.DetectorMode = RELATIVE_REG_NUMBER;
  else //if (temp_str[0].compare("FixedTh")==0)
    MSERPars.DetectorMode = FIXED_TH;
}
void GetBaumbergPars(AffineShapeParams &par, INIReader &reader,const char* section) {
  par.maxIterations = reader.GetInteger(section, "max_iter", par.maxIterations);
  par.patchSize = reader.GetInteger(section, "patch_size", par.patchSize);
  par.smmWindowSize = reader.GetInteger(section, "smmWindowSize", par.smmWindowSize);
  par.convergenceThreshold = reader.GetDouble(section, "convergenceThreshold", par.convergenceThreshold);
  par.doBaumberg = reader.GetInteger(section, "doBaumberg", par.doBaumberg);
  par.initialSigma = reader.GetDouble(section, "initialSigma", par.initialSigma);
  std::string method = reader.GetString(section, "method", "SMM");
  par.affBmbrgMethod = AFF_BMBRG_SMM;
  if (method == "SMM") {
      par.affBmbrgMethod = AFF_BMBRG_SMM;
    }
  if (method == "Hessian") {
      par.affBmbrgMethod = AFF_BMBRG_HESSIAN;
    }
}
void GetPatchExtractionPars(PatchExtractionParams &pars, INIReader &reader,const char* section)
{
  pars.patchSize = reader.GetInteger(section, "patchSize", pars.patchSize);
  pars.mrSize = reader.GetDouble(section, "mrSize", pars.mrSize);
  pars.FastPatchExtraction = reader.GetBoolean(section, "FastPatchExtraction", pars.FastPatchExtraction);
  pars.photoNorm =reader.GetBoolean(section, "photoNorm", pars.photoNorm);
}
void GetPixelPars(PIXELSDescriptorParams &pars, INIReader &reader,const char* section)
{
  GetPatchExtractionPars(pars.PEParam,reader,section);
  pars.normType = reader.GetString(section, "normType", pars.normType);
}

void GetORBPars(ORBParams &pars, INIReader &reader,const char* section)
{
  pars.edgeThreshold = reader.GetInteger(section, "edgeThreshold", pars.edgeThreshold);
  pars.firstLevel= reader.GetInteger(section, "firstLevel", pars.firstLevel);
  pars.nfeatures = reader.GetInteger(section, "nfeatures", pars.nfeatures);
  pars.nlevels = reader.GetInteger(section, "nlevels", pars.nlevels);
  pars.scaleFactor = reader.GetDouble(section, "scaleFactor", pars.scaleFactor);
  pars.WTA_K = reader.GetInteger(section, "WTA_K", pars.WTA_K);
  GetPatchExtractionPars(pars.PEParam,reader,section);
  pars.doBaumberg = reader.GetBoolean(section,"doBaumberg",pars.doBaumberg);
}
void GetHessPars(ScaleSpaceDetectorParams &HessPars, INIReader &reader,const char* section)
{
  HessPars.PyramidPars.DetectorType = DET_HESSIAN;

  HessPars.PyramidPars.threshold = reader.GetDouble(section, "threshold", HessPars.PyramidPars.threshold);
  HessPars.PyramidPars.rel_threshold = reader.GetDouble(section, "relativeThreshold", HessPars.PyramidPars.rel_threshold);
  HessPars.PyramidPars.rel_reg_number = reader.GetDouble(section, "relativeRegionsNumber", HessPars.PyramidPars.rel_reg_number);
  HessPars.PyramidPars.reg_number = reader.GetInteger(section, "regionsNumber", HessPars.PyramidPars.reg_number);

  HessPars.PyramidPars.border = reader.GetInteger(section, "border", HessPars.PyramidPars.border);
  HessPars.PyramidPars.numberOfScales =reader.GetInteger(section, "numberOfScales", HessPars.PyramidPars.numberOfScales);
  HessPars.PyramidPars.doOnWLD = reader.GetInteger(section, "doOnWLD", HessPars.PyramidPars.doOnWLD);
  HessPars.PyramidPars.doOnNormal = reader.GetInteger(section, "doOnNormal", HessPars.PyramidPars.doOnNormal);
  HessPars.PyramidPars.initialSigma = reader.GetDouble(section, "initialSigma", HessPars.PyramidPars.initialSigma);
  HessPars.PyramidPars.edgeEigenValueRatio = reader.GetDouble(section, "edgeEigenValueRatio", HessPars.PyramidPars.edgeEigenValueRatio);
  HessPars.PyramidPars.iiDoGMode = reader.GetBoolean(section, "iiDoGMode", HessPars.PyramidPars.iiDoGMode);

  HessPars.AffineShapePars.maxIterations = reader.GetInteger(section, "max_iter", HessPars.AffineShapePars.maxIterations);
  HessPars.AffineShapePars.patchSize = reader.GetInteger(section, "patch_size", HessPars.AffineShapePars.patchSize);
  HessPars.AffineShapePars.smmWindowSize = reader.GetInteger(section, "smmWindowSize", HessPars.AffineShapePars.smmWindowSize);
  HessPars.AffineShapePars.convergenceThreshold = reader.GetDouble(section, "convergenceThreshold", HessPars.AffineShapePars.convergenceThreshold);
  HessPars.AffineShapePars.doBaumberg = reader.GetInteger(section, "doBaumberg", HessPars.AffineShapePars.doBaumberg);
  HessPars.PyramidPars.WLDPar.a = reader.GetDouble(section, "a", HessPars.PyramidPars.WLDPar.a);
  HessPars.PyramidPars.WLDPar.b = reader.GetDouble(section, "b", HessPars.PyramidPars.WLDPar.b);
  HessPars.PyramidPars.WLDPar.g = reader.GetDouble(section, "g", HessPars.PyramidPars.WLDPar.g);

  std::vector< std::string> temp_str;
  reader.GetStringVector(section, "mode",temp_str);
  if (temp_str[0].compare("RelativeTh")==0)
    HessPars.PyramidPars.DetectorMode = RELATIVE_TH;
  else if (temp_str[0].compare("FixedRegNumber")==0)
    HessPars.PyramidPars.DetectorMode = FIXED_REG_NUMBER;
  else if (temp_str[0].compare("NotLessThanRegions")==0)
    HessPars.PyramidPars.DetectorMode = NOT_LESS_THAN_REGIONS;
  else if (temp_str[0].compare("RelativeRegNumber")==0)
    HessPars.PyramidPars.DetectorMode = RELATIVE_REG_NUMBER;
  else //if (temp_str[0].compare("FixedTh")==0)
    HessPars.PyramidPars.DetectorMode = FIXED_TH;

}
void GetHarrPars(ScaleSpaceDetectorParams &HarrPars, INIReader &reader,const char* section)
{
  HarrPars.PyramidPars.DetectorType = DET_HARRIS;
  HarrPars.PyramidPars.threshold = reader.GetDouble(section, "threshold", HarrPars.PyramidPars.threshold);
  HarrPars.PyramidPars.rel_threshold = reader.GetDouble(section, "relativeThreshold", HarrPars.PyramidPars.rel_threshold);
  HarrPars.PyramidPars.rel_reg_number = reader.GetDouble(section, "relativeRegionsNumber", HarrPars.PyramidPars.rel_reg_number);
  HarrPars.PyramidPars.reg_number = reader.GetInteger(section, "regionsNumber", HarrPars.PyramidPars.reg_number);

  HarrPars.PyramidPars.border = reader.GetInteger(section, "border", HarrPars.PyramidPars.border);
  HarrPars.PyramidPars.numberOfScales =reader.GetInteger(section, "numberOfScales", HarrPars.PyramidPars.numberOfScales);
  HarrPars.PyramidPars.doOnWLD = reader.GetInteger(section, "doOnWLD", HarrPars.PyramidPars.doOnWLD);
  HarrPars.PyramidPars.doOnNormal = reader.GetInteger(section, "doOnNormal", HarrPars.PyramidPars.doOnNormal);
  HarrPars.PyramidPars.initialSigma = reader.GetDouble(section, "initialSigma", HarrPars.PyramidPars.initialSigma);
  HarrPars.PyramidPars.edgeEigenValueRatio = reader.GetDouble(section, "edgeEigenValueRatio", HarrPars.PyramidPars.edgeEigenValueRatio);
  HarrPars.PyramidPars.iiDoGMode = reader.GetBoolean(section, "iiDoGMode", HarrPars.PyramidPars.iiDoGMode);

  HarrPars.AffineShapePars.maxIterations = reader.GetInteger(section, "max_iter", HarrPars.AffineShapePars.maxIterations);
  HarrPars.AffineShapePars.patchSize = reader.GetInteger(section, "patch_size", HarrPars.AffineShapePars.patchSize);
  HarrPars.AffineShapePars.smmWindowSize = reader.GetInteger(section, "smmWindowSize", HarrPars.AffineShapePars.smmWindowSize);
  HarrPars.AffineShapePars.convergenceThreshold = reader.GetDouble(section, "convergenceThreshold", HarrPars.AffineShapePars.convergenceThreshold);
  HarrPars.AffineShapePars.doBaumberg = reader.GetInteger(section, "doBaumberg", HarrPars.AffineShapePars.doBaumberg);

  HarrPars.PyramidPars.WLDPar.a = reader.GetDouble(section, "a", HarrPars.PyramidPars.WLDPar.a);
  HarrPars.PyramidPars.WLDPar.b = reader.GetDouble(section, "b", HarrPars.PyramidPars.WLDPar.b);
  HarrPars.PyramidPars.WLDPar.g = reader.GetDouble(section, "g", HarrPars.PyramidPars.WLDPar.g);

  std::vector< std::string> temp_str;
  reader.GetStringVector(section, "mode",temp_str);
  if (temp_str[0].compare("RelativeTh")==0)
    HarrPars.PyramidPars.DetectorMode = RELATIVE_TH;
  else if (temp_str[0].compare("FixedRegNumber")==0)
    HarrPars.PyramidPars.DetectorMode = FIXED_REG_NUMBER;
  else if (temp_str[0].compare("NotLessThanRegions")==0)
    HarrPars.PyramidPars.DetectorMode = NOT_LESS_THAN_REGIONS;
  else if (temp_str[0].compare("RelativeRegNumber")==0)
    HarrPars.PyramidPars.DetectorMode = RELATIVE_REG_NUMBER;
  else //if (temp_str[0].compare("FixedTh")==0)
    HarrPars.PyramidPars.DetectorMode = FIXED_TH;

}
void GetDoGPars(ScaleSpaceDetectorParams &DoGPars, INIReader &reader,const char* section)
{
  DoGPars.PyramidPars.DetectorType = DET_DOG;

  DoGPars.PyramidPars.threshold = reader.GetDouble(section, "threshold", DoGPars.PyramidPars.threshold);
  DoGPars.PyramidPars.rel_threshold = reader.GetDouble(section, "relativeThreshold", DoGPars.PyramidPars.rel_threshold);
  DoGPars.PyramidPars.rel_reg_number = reader.GetDouble(section, "relativeRegionsNumber", DoGPars.PyramidPars.rel_reg_number);
  DoGPars.PyramidPars.reg_number = reader.GetInteger(section, "regionsNumber", DoGPars.PyramidPars.reg_number);

  DoGPars.PyramidPars.border = reader.GetInteger(section, "border", DoGPars.PyramidPars.border);
  DoGPars.PyramidPars.numberOfScales =reader.GetInteger(section, "numberOfScales", DoGPars.PyramidPars.numberOfScales);
  DoGPars.PyramidPars.doOnWLD = reader.GetInteger(section, "doOnWLD", DoGPars.PyramidPars.doOnWLD);
  DoGPars.PyramidPars.doOnNormal = reader.GetInteger(section, "doOnNormal", DoGPars.PyramidPars.doOnNormal);
  DoGPars.PyramidPars.initialSigma = reader.GetDouble(section, "initialSigma", DoGPars.PyramidPars.initialSigma);
  DoGPars.PyramidPars.edgeEigenValueRatio = reader.GetDouble(section, "edgeEigenValueRatio", DoGPars.PyramidPars.edgeEigenValueRatio);
  DoGPars.PyramidPars.iiDoGMode = reader.GetBoolean(section, "iiDoGMode", DoGPars.PyramidPars.iiDoGMode);

  DoGPars.AffineShapePars.maxIterations = reader.GetInteger(section, "max_iter", DoGPars.AffineShapePars.maxIterations);
  DoGPars.AffineShapePars.patchSize = reader.GetInteger(section, "patch_size", DoGPars.AffineShapePars.patchSize);
  DoGPars.AffineShapePars.smmWindowSize = reader.GetInteger(section, "smmWindowSize", DoGPars.AffineShapePars.smmWindowSize);
  DoGPars.AffineShapePars.convergenceThreshold = reader.GetDouble(section, "convergenceThreshold", DoGPars.AffineShapePars.convergenceThreshold);
  DoGPars.AffineShapePars.doBaumberg = reader.GetInteger(section, "doBaumberg", DoGPars.AffineShapePars.doBaumberg);

  DoGPars.PyramidPars.WLDPar.a = reader.GetDouble(section, "a", DoGPars.PyramidPars.WLDPar.a);
  DoGPars.PyramidPars.WLDPar.b = reader.GetDouble(section, "b", DoGPars.PyramidPars.WLDPar.b);
  DoGPars.PyramidPars.WLDPar.g = reader.GetDouble(section, "g", DoGPars.PyramidPars.WLDPar.g);


  std::vector< std::string> temp_str;
  reader.GetStringVector(section, "mode",temp_str);
  if (temp_str[0].compare("RelativeTh")==0)
    DoGPars.PyramidPars.DetectorMode = RELATIVE_TH;
  else if (temp_str[0].compare("FixedRegNumber")==0)
    DoGPars.PyramidPars.DetectorMode = FIXED_REG_NUMBER;
  else if (temp_str[0].compare("NotLessThanRegions")==0)
    DoGPars.PyramidPars.DetectorMode = NOT_LESS_THAN_REGIONS;
  else if (temp_str[0].compare("RelativeRegNumber")==0)
    DoGPars.PyramidPars.DetectorMode = RELATIVE_REG_NUMBER;
  else //if (temp_str[0].compare("FixedTh")==0)
    DoGPars.PyramidPars.DetectorMode = FIXED_TH;

}
void GetMatchPars(MatchPars &pars, INIReader &reader, INIReader &iter_reader, const char* section)
{
  int Steps = iter_reader.GetInteger("Iterations", "Steps", 1);
  pars.IterWhatToMatch.clear();
  pars.IterWhatToMatch.reserve(Steps);
  for (int i=0; i<Steps; i++) //Reading parameters
    {
      WhatToMatch currentWhatToMatch;

      iter_reader.GetStringVector("Matching"+IntToStr(i), "GroupDescriptors", currentWhatToMatch.group_descriptors);
      iter_reader.GetStringVector("Matching"+IntToStr(i), "SeparateDescriptors", currentWhatToMatch.separate_descriptors);
      iter_reader.GetStringVector("Matching"+IntToStr(i), "GroupDetectors", currentWhatToMatch.group_detectors);
      iter_reader.GetStringVector("Matching"+IntToStr(i), "SeparateDetectors", currentWhatToMatch.separate_detectors);
      pars.IterWhatToMatch.push_back(currentWhatToMatch);
    }
  //std::vector<std::vector<ViewSynthParameters> > acc_par(DetectorNames.size());

  for (int i=0; i<Steps; i++) //Reading parameters


    pars.contradDist = reader.GetDouble(section, "contradDist", pars.contradDist);

  for (unsigned int desc=0; desc< DescriptorNames.size(); desc++) //Reading parameters
    {
      pars.FGINNThreshold[DescriptorNames[desc]] = reader.GetDouble(section, "matchRatio"+DescriptorNames[desc], 0);
      pars.DistanceThreshold[DescriptorNames[desc]] = reader.GetDouble(section, "matchDistance"+DescriptorNames[desc], 0);
    }

  pars.standard_2nd_closest = reader.GetInteger(section, "standard_2nd_closest", pars.standard_2nd_closest);
  pars.kd_trees = reader.GetInteger(section, "kd_trees", pars.kd_trees);
  pars.knn_checks = reader.GetInteger(section, "knn_checks", pars.knn_checks);
  pars.standard_2nd_closest = reader.GetInteger(section, "doStandard_2nd_closestToo", 0);
  pars.RANSACforStopping = reader.GetInteger(section, "RANSACforStopping",1);
  pars.doBothRANSACgroundTruth = reader.GetInteger(section,"doBothRANSACgroundTruth",1);
  pars.doDensification =  reader.GetInteger(section, "doDensification", pars.doDensification);
  pars.useDBforFGINN = reader.GetInteger(section, "useDBforFGINN", pars.useDBforFGINN);
  pars.SIFTDBfile = reader.GetString(section, "SIFTDBfile", "100_db.txt");

  pars.FPRate = reader.GetDouble(section, "FPRate", pars.FPRate);


  std::string vector_dist, binary_dist,vector_index,binary_index;
  vector_dist = reader.GetString(section, "vector_dist", "L2");

  if (vector_dist.compare("L2")==0)
    pars.vector_dist = cvflann::FLANN_DIST_L2;
  else if (vector_dist.compare("L1")==0)
    pars.vector_dist = cvflann::FLANN_DIST_L1;
  else if (vector_dist.compare("Hamming")==0)
    pars.vector_dist = cvflann::FLANN_DIST_HAMMING;
  else if (vector_dist.compare("Mink")==0)
    pars.vector_dist = cvflann::FLANN_DIST_MINKOWSKI;
  else if (vector_dist.compare("Hellinger")==0)
    pars.vector_dist = cvflann::FLANN_DIST_HELLINGER;
  else if (vector_dist.compare("Chi_square")==0)
    pars.vector_dist = cvflann::FLANN_DIST_CHI_SQUARE;
  else if (vector_dist.compare("KL")==0)
    pars.vector_dist = cvflann::FLANN_DIST_KULLBACK_LEIBLER;
  else if (vector_dist.compare("Max")==0)
    pars.vector_dist = cvflann::FLANN_DIST_MAX;
  else //L2 = default
    pars.vector_dist = cvflann::FLANN_DIST_L2;

  binary_dist = reader.GetString(section, "binary_dist", "Hamming");
  if (binary_dist.compare("L2")==0)
    pars.binary_dist = cvflann::FLANN_DIST_L2;
  else if (binary_dist.compare("L1")==0)
    pars.binary_dist = cvflann::FLANN_DIST_L1;
  else if (binary_dist.compare("Hamming")==0)
    pars.binary_dist = cvflann::FLANN_DIST_HAMMING;
  else if (binary_dist.compare("Mink")==0)
    pars.binary_dist = cvflann::FLANN_DIST_MINKOWSKI;
  else if (binary_dist.compare("Hellinger")==0)
    pars.binary_dist = cvflann::FLANN_DIST_HELLINGER;
  else if (binary_dist.compare("Chi_square")==0)
    pars.binary_dist = cvflann::FLANN_DIST_CHI_SQUARE;
  else if (binary_dist.compare("KL")==0)
    pars.binary_dist = cvflann::FLANN_DIST_KULLBACK_LEIBLER;
  else if (binary_dist.compare("Max")==0)
    pars.binary_dist = cvflann::FLANN_DIST_MAX;
  else //L2 = default
    pars.binary_dist = cvflann::FLANN_DIST_HAMMING;

  vector_index = reader.GetString(section, "vector_matcher", "kdtree");

  if (vector_index.compare("kdtree")==0)
    pars.vector_matcher = cvflann::FLANN_INDEX_KDTREE;
  else if (vector_index.compare("linear")==0)
    pars.vector_matcher = cvflann::FLANN_INDEX_LINEAR;
  else if (vector_index.compare("composite")==0)
    pars.vector_matcher = cvflann::FLANN_INDEX_COMPOSITE;
  else if (vector_index.compare("autotuned")==0)
    pars.vector_matcher = cvflann::FLANN_INDEX_AUTOTUNED;
  else if (vector_index.compare("kmeans")==0)
    pars.vector_matcher = cvflann::FLANN_INDEX_KMEANS;
  else if (vector_index.compare("lsh")==0)
    pars.vector_matcher = cvflann::FLANN_INDEX_LSH;
  else if (vector_index.compare("hierarchical")==0)
    pars.vector_matcher = cvflann::FLANN_INDEX_HIERARCHICAL;
  else //kdtree = default
    pars.vector_matcher = cvflann::FLANN_INDEX_KDTREE;

  binary_index = reader.GetString(section, "binary_matcher", "LSH");
  if (binary_index.compare("lsh")==0)
    pars.binary_matcher = cvflann::FLANN_INDEX_LSH;
  else if (binary_index.compare("kdtree")==0)
    pars.binary_matcher = cvflann::FLANN_INDEX_KDTREE;
  else if (binary_index.compare("linear")==0)
    pars.binary_matcher = cvflann::FLANN_INDEX_LINEAR;
  else if (binary_index.compare("composite")==0)
    pars.binary_matcher = cvflann::FLANN_INDEX_COMPOSITE;
  else if (binary_index.compare("autotuned")==0)
    pars.binary_matcher = cvflann::FLANN_INDEX_AUTOTUNED;
  else if (binary_index.compare("kmeans")==0)
    pars.binary_matcher = cvflann::FLANN_INDEX_KMEANS;
  else if (binary_index.compare("hierarchical")==0)
    pars.binary_matcher = cvflann::FLANN_INDEX_HIERARCHICAL;
  else //kdtree = default
    pars.binary_matcher = cvflann::FLANN_INDEX_LSH;

}
void GetSIFTDescPars(SIFTDescriptorParams &pars, INIReader &reader,const char* section)
{
  pars.spatialBins = reader.GetInteger(section, "spatialBins", pars.spatialBins);
  pars.orientationBins = reader.GetInteger(section, "orientationBins", pars.orientationBins);
  pars.maxBinValue = reader.GetDouble(section, "maxBinValue", pars.maxBinValue);
  pars.orientTh = reader.GetDouble(section,"orientationThreshold", pars.orientTh);
  pars.maxOrientations = reader.GetInteger(section, "maxOrientations", pars.maxOrientations);
  pars.estimateOrientation = reader.GetBoolean(section, "estimateOrientation", pars.estimateOrientation);

  GetPatchExtractionPars(pars.PEParam,reader,section);
}
void GetRANSACPars(RANSACPars &pars, INIReader &reader,const char* section)
{
  pars.err_threshold = reader.GetDouble(section, "err_threshold", pars.err_threshold);
  pars.confidence = reader.GetDouble(section, "confidence", pars.confidence);
  pars.max_samples = reader.GetInteger(section, "max_samples", pars.max_samples);
  pars.localOptimization = reader.GetInteger(section, "localOptimization", pars.localOptimization);
  pars.LAFCoef = reader.GetInteger(section, "LAFcoef", pars.LAFCoef);
  pars.HLAFCoef = reader.GetInteger(section, "HLAFcoef", pars.HLAFCoef);
  pars.doSymmCheck = reader.GetInteger(section, "doSymmCheck", pars.doSymmCheck);
  std::vector< std::string> temp_str;
  reader.GetStringVector(section, "ErrorType",temp_str);
  if (temp_str[0].compare("Sampson")==0)
    pars.errorType = SAMPSON;
  else if (temp_str[0].compare("SymmMax")==0)
    pars.errorType = SYMM_MAX;
  else //if (temp_str[0].compare("SymmSum")==0)
    pars.errorType = SYMM_SUM;

  pars.prevalidateSample = reader.GetBoolean(section,"prevalidateSample",pars.prevalidateSample);
  pars.prevalidateModel = reader.GetBoolean(section,"prevalidateModel",pars.prevalidateModel);
  pars.testDegeneracy  = reader.GetBoolean(section, "testDegeneracy",pars.testDegeneracy);
  pars.verifMethod = reader.GetString(section, "verifMethod",pars.verifMethod);
  pars.randomSamplingMethod =  reader.GetString(section, "randomSamplingMethod ",pars.randomSamplingMethod);
  pars.innerRansacRepetitions = reader.GetInteger(section, "innerRansacRepetitions",pars.innerRansacRepetitions);
  pars.innerSampleSize = reader.GetInteger(section, "innerSampleSize",pars.innerSampleSize);
  pars.numStepsIterative = reader.GetInteger(section, "numStepsIterative",pars.numStepsIterative);
  pars.thresholdMultiplier = reader.GetDouble(section, "thresholdMultiplier",pars.thresholdMultiplier);
  pars.prosacBeta = reader.GetDouble(section, "prosacBeta",pars.prosacBeta);
  pars.prosacSamples = reader.GetInteger(section, "prosacSamples",pars.prosacSamples);
  pars.prosacMinStopLen = reader.GetInteger(section, "prosacMinStopLen",pars.prosacMinStopLen);
  pars.prosacNonRandConf = reader.GetDouble(section, "prosacNonRandConf",pars.prosacNonRandConf);
  pars.SPRT_tM = reader.GetDouble(section, "SPRT_tM",pars.SPRT_tM);
  pars.SPRT_mS = reader.GetDouble(section, "SPRT_mS",pars.SPRT_mS);
  pars.SPRT_delta = reader.GetDouble(section, "SPRT_delta",pars.SPRT_delta);
  pars.SPRT_eps = reader.GetDouble(section, "SPRT_eps",pars.SPRT_eps);


}
void GetIterPars(std::vector<IterationViewsynthesisParam> &pars, INIReader &reader)
{
  int Steps = reader.GetInteger("Iterations", "Steps", 1);

  pars.clear();
  pars.resize(Steps);

  std::vector<std::vector<ViewSynthParameters> > acc_par(DetectorNames.size());

  for (int i=0; i<Steps; i++) //Reading parameters
    for (unsigned int j=0; j< DetectorNames.size(); j++) //Reading parameters
      {
        std::vector <double> tilt_set;
        std::vector <double> scale_set;
        double phi, initSigma;
        int dsplevels;
        double minSigma, maxSigma;
        std::vector<std::string> descriptors;
        std::vector<double> FGINNThreshold, DistanceThreshold;
        int doBlur = 1;
        reader.GetDoubleVector(DetectorNames[j]+IntToStr(i), "TiltSet",tilt_set);
        reader.GetDoubleVector(DetectorNames[j]+IntToStr(i), "ScaleSet",scale_set);
        phi = reader.GetDouble(DetectorNames[j]+IntToStr(i), "Phi",360);
        initSigma =  reader.GetDouble(DetectorNames[j]+IntToStr(i), "initSigma", 0.5);
        dsplevels = reader.GetInteger(DetectorNames[j]+IntToStr(i), "DSPLevels",0);
        minSigma = reader.GetDouble(DetectorNames[j]+IntToStr(i), "minSigma",1.0);
        maxSigma = reader.GetDouble(DetectorNames[j]+IntToStr(i), "maxSigma",1.0);

        reader.GetStringVector(DetectorNames[j]+IntToStr(i), "Descriptors", descriptors);
        reader.GetDoubleVector(DetectorNames[j]+IntToStr(i), "FGINNThreshold",FGINNThreshold);
        reader.GetDoubleVector(DetectorNames[j]+IntToStr(i), "DistanceThreshold",DistanceThreshold);

        SetVSPars(scale_set,tilt_set,phi,FGINNThreshold,DistanceThreshold,descriptors,
                  pars[i][DetectorNames[j]],acc_par[j],initSigma,doBlur,dsplevels, minSigma,maxSigma);
      }
}

int getCLIparam(configs &conf1,int argc, char **argv)
{

  conf1.CLIparams.img1_fname = argv[1];
  conf1.CLIparams.img2_fname = argv[2];
  conf1.CLIparams.out1_fname = argv[3];
  conf1.CLIparams.out2_fname = argv[4];
  conf1.CLIparams.k1_fname = argv[5];
  conf1.CLIparams.k2_fname = argv[6];
  conf1.CLIparams.matchings_fname = argv[7];
  conf1.CLIparams.log_fname = argv[8];
  if (argc >= (Tmin +1))
    conf1.CLIparams.logOnly = atoi(argv[Tmin]);

  conf1.CLIparams.ver_type = Homog;

  if (argc >= (Tmin +2))
    {
      std::string ver_type = argv[Tmin+1];
      if (ver_type == "H") {
          conf1.CLIparams.ver_type = RANSAC_mode_t::Homog;
          conf1.verification_type="Homography";

        } else if (ver_type == "F") {
          conf1.CLIparams.ver_type = RANSAC_mode_t::Fundamental;
          conf1.verification_type="Epipolar";

        } else {
          std::cerr << conf1.CLIparams.ver_type << " is wrong correspondence verification type." << std::endl;
          std::cerr << "Try H for homography or F for epipolar"<< std::endl;
          return 1;
        }
      if (argc >= Tmin +3) conf1.CLIparams.config_fname = argv[Tmin+2];
      if (argc >= Tmin +4) conf1.CLIparams.iters_fname = argv[Tmin+3];
    }
  INIReader ConfigIni(conf1.CLIparams.config_fname);
  if (ConfigIni.ParseError() < 0)
    {
      std::cerr << "Can't load " << conf1.CLIparams.config_fname << std::endl;
      return 1;
    }
  INIReader ItersIni(conf1.CLIparams.iters_fname);
  if (ItersIni.ParseError() < 0)
    {
      std::cerr << "Can't load  "<< conf1.CLIparams.iters_fname << std::endl;
      return 1;
    }
  GetDoGPars(conf1.DetectorsPars.DoGParam,ConfigIni);
  GetHessPars(conf1.DetectorsPars.HessParam,ConfigIni);
  GetDomOriPars(conf1.DomOriPars,ConfigIni);
  GetHarrPars(conf1.DetectorsPars.HarrParam,ConfigIni);
  GetMSERPars(conf1.DetectorsPars.MSERParam, ConfigIni);
  GetORBPars(conf1.DetectorsPars.ORBParam, ConfigIni);
  GetPixelPars(conf1.DescriptorPars.PixelsParam, ConfigIni);
  GetBaumbergPars(conf1.DetectorsPars.BaumbergParam, ConfigIni);
  GetMatchPars(conf1.Matchparam,ConfigIni,ItersIni);
  conf1.LoadColor = ConfigIni.GetInteger("Computing", "LoadColor", conf1.LoadColor);

  ///SIFTs
  GetSIFTDescPars(conf1.DescriptorPars.SIFTParam, ConfigIni);
  conf1.DescriptorPars.RootSIFTParam = conf1.DescriptorPars.SIFTParam;
  conf1.DescriptorPars.RootSIFTParam.useRootSIFT = 1;
  conf1.DescriptorPars.HalfRootSIFTParam =  conf1.DescriptorPars.RootSIFTParam;
  conf1.DescriptorPars.HalfRootSIFTParam.doHalfSIFT = 1;

  conf1.DescriptorPars.HalfSIFTParam = conf1.DescriptorPars.HalfRootSIFTParam;
  conf1.DescriptorPars.HalfSIFTParam.useRootSIFT = 0;

  GetIterPars(conf1.ItersParam,ItersIni);

  conf1.DrawParam.drawEpipolarLines = ConfigIni.GetInteger("ImageOutput", "drawEpipolarLines", 0);
  conf1.DrawParam.drawOnlyCenters = ConfigIni.GetInteger("ImageOutput", "drawOnlyCenters", 1);
  conf1.DrawParam.drawReprojected = ConfigIni.GetInteger("ImageOutput", "drawReprojected", 1);
  conf1.DrawParam.writeImages = ConfigIni.GetInteger("ImageOutput", "writeImages", 1);
  conf1.OutputParam.writeKeypoints = ConfigIni.GetInteger("TextOutput", "writeKeypoints", 1);
  conf1.OutputParam.writeMatches = ConfigIni.GetInteger("TextOutput", "writeMatches", 1);
  conf1.OutputParam.timeLog = ConfigIni.GetInteger("TextOutput", "timeLog", 0);
  conf1.OutputParam.featureComplemetaryLog = ConfigIni.GetInteger("TextOutput", "featureComplemetaryLog", 0);
  conf1.OutputParam.verbose = ConfigIni.GetInteger("TextOutput", "verbose", 0);
  conf1.OutputParam.outputAllTentatives = ConfigIni.GetInteger("TextOutput", "outputAllTentatives", 0);
  conf1.OutputParam.outputEstimatedHorF = ConfigIni.GetInteger("TextOutput", "outputEstimatedHorF", 0);
  conf1.RANSACParam.LAFCoef = ConfigIni.GetInteger("Matching", "LAFcoef", 0);
  conf1.FilterParam.duplicateDist = ConfigIni.GetDouble("DuplicateFiltering", "duplicateDist", 3.0);
  conf1.FilterParam.doBeforeRANSAC = ConfigIni.GetDouble("DuplicateFiltering", "doBeforeRANSAC", 1);
  conf1.CLIparams.doCLAHE = ConfigIni.GetInteger("Matching", "doCLAHE", conf1.CLIparams.doCLAHE);


  std::string filter_mode = ConfigIni.GetString("DuplicateFiltering", "whichCorrespondenceRemains", "random");
  if (filter_mode.compare("bestFGINN")==0)
    conf1.FilterParam.mode = MODE_FGINN;
  else if (filter_mode.compare("bestDistance")==0)
    conf1.FilterParam.mode = MODE_DISTANCE;
  else if (filter_mode.compare("biggerRegion")==0)
    conf1.FilterParam.mode = MODE_BIGGER_REGION;
  else
    conf1.FilterParam.mode = MODE_RANDOM;

  conf1.Matchparam.maxSteps = ItersIni.GetInteger("Iterations", "Steps", 4);
  conf1.Matchparam.minMatches =  ItersIni.GetInteger("Iterations", "minMatches", 15);

  GetRANSACPars(conf1.RANSACParam,ConfigIni);
  if ((conf1.CLIparams.ver_type == Fundamental))
    conf1.RANSACParam.useF=1;
  else conf1.RANSACParam.useF=0;

  conf1.RANSACParam.justMarkOutliers = conf1.OutputParam.outputAllTentatives;
#ifdef _OPENMP
  conf1.n_threads = ConfigIni.GetInteger("Computing", "numberOfCores", -1);
  if (conf1.n_threads >= 0) omp_set_num_threads(conf1.n_threads);
  if (conf1.OutputParam.verbose) std::cerr << "Maximum threads can be used: " << omp_get_max_threads() << std::endl;
#endif

  return 0;
}


void GetDomOriPars(DominantOrientationParams &DomOriPars, INIReader &reader, char const *section) {
  DomOriPars.addUpRight = reader.GetBoolean(section, "addUpRight", DomOriPars.addUpRight);
  DomOriPars.halfSIFTMode = reader.GetBoolean(section, "halfSIFTMode", DomOriPars.halfSIFTMode);
  DomOriPars.maxAngles = (int)reader.GetInteger(section, "maxAngles", DomOriPars.maxAngles);
  DomOriPars.threshold = (float) reader.GetDouble(section, "threshold", DomOriPars.threshold);
  GetPatchExtractionPars(DomOriPars.PEParam,reader, section);
}

void WriteH(double* H, std::ostream &out1)
{
  out1  << H[0] << " " << H[1] << " " << H[2] << std::endl
        << H[3] << " " << H[4] << " " << H[5] << std::endl
        << H[6] << " " << H[7] << " " << H[8] << std::endl;
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

void DrawMatches(const cv::Mat &in_img1,const cv::Mat &in_img2, cv::Mat &out_img1, cv::Mat &out_img2,const cv::Mat &H1,
                 TentativeCorrespListExt matchings,
                 const int DrawCentersOnly,
                 const int ReprojectToOneImage,
                 const int r1,
                 const int r2,
                 const int drawEpipolarLines,
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

      out_tmp1 = cv::Mat (cv::max(in_img1.rows,in_img2.rows),in_img1.cols+in_img2.cols+sep,in_img1.type(), cv::Scalar(255,255,255));

      cv::Mat roiImgResult_Left = out_tmp1(cv::Rect(0,0,in_img1.cols,in_img1.rows));
      cv::Mat roiImgResult_Right = out_tmp1(cv::Rect(in_img1.cols+sep,0,in_img2.cols,in_img2.rows));
      roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
      roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult

      out_tmp2 = cv::Mat(in_img1.rows+in_img2.rows+sep, cv::max(in_img1.cols,in_img2.cols),in_img2.type(),cv::Scalar(255,255,255));

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
