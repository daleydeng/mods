/*-----------------------------------------------------------*/
/* Copyright 2013-2015, Dmytro Mishkin  ducha.aiki@gmail.com */
/*----------------------------------------------------------*/
#ifndef MODS_NEW_IO_MODS_H
#define MODS_NEW_IO_MODS_H

#include "configuration.hpp"
#include "common.hh"
#include "matching.hh"
#include "inih/cpp/INIReader.h"

const int Tmin = 9;//minimum number of command-line parameters
using std::string;
using std::ostream;
using namespace mods;
struct configs
{
    int n_threads;
    DescriptorsParameters DescriptorPars;
    DetectorsParameters DetectorsPars;
    DominantOrientationParams DomOriPars;
    int LoadColor;
    MatchPars Matchparam;
    RANSACPars RANSACParam;
    std::vector<IterationViewsynthesisParam> ItersParam;
    parameters CLIparams;
    filteringParams FilterParam;
    drawingParams DrawParam;
    outputParams OutputParam;
    bool read_pre_extracted;
    bool match_one_to_many;
    string descriptor, matching_lib, verification_type;
    configs()
    {
        n_threads = 1;
        LoadColor = 1;
        read_pre_extracted = false;
        match_one_to_many = false;
    }
};

void WriteLog(logs log, ostream& out);
void WriteTimeLog(TimeLog log, ostream &out,
                  const int writeRelValues = 1,
                  const int writeAbsValues = 0,
                  const int writeDescription = 0);
void GetMSERPars(ExtremaParams &MSERPars, INIReader &reader,const char* section="MSER");
void GetORBPars(ORBParams &pars, INIReader &reader,const char* section="ORB");
void GetReadPars(ReadAffsFromFileParams &pars, INIReader &reader,const char* section="ReadAffs");
void GetPixelPars(PIXELSDescriptorParams &pars, INIReader &reader,const char* section="PixelDescriptor");
void GetHessPars(ScaleSpaceDetectorParams &HessPars, INIReader &reader,const char* section="HessianAffine");
void GetPatchExtractionPars(PatchExtractionParams &PEPars, INIReader &reader,const char* section);
void GetHarrPars(ScaleSpaceDetectorParams &HarrPars, INIReader &reader,const char* section="HarrisAffine");
void GetDoGPars(ScaleSpaceDetectorParams &DoGPars, INIReader &reader,const char* section="DoG");
void GetDomOriPars(DominantOrientationParams &DomOriPars, INIReader &reader,const char* section="DominantOrientation");
void GetBaumbergPars(AffineShapeParams &pars, INIReader &reader,const char* section="AffineAdaptation");

void GetMatchPars(MatchPars &pars, INIReader &reader, INIReader &iter_reader, const char* section="Matching");
void GetSIFTDescPars(SIFTDescriptorParams &pars, INIReader &reader,const char* section="SIFTDescriptor");
void GetRANSACPars(RANSACPars &pars, INIReader &reader,const char* section="RANSAC");
void GetIterPars(std::vector<IterationViewsynthesisParam> &pars, INIReader &reader);
int getCLIparam(configs &conf1,int argc, char **argv);
int getCLIparamExtractFeatures(configs &conf1,int argc, char **argv);
int getCLIparamExtractFeaturesBenchmark(configs &conf1,int argc, char **argv);
int getCLIparamExportDescriptorsBenchmark(configs &conf1,int argc, char **argv);

void WriteMatchings(TentativeCorrespListExt &match, std::ostream &out1, int writeWithRatios = 0);
void WriteH(double* H, std::ostream &out1);

void DrawMatches(const cv::Mat &in_img1,const cv::Mat &in_img2, cv::Mat &out_img1,cv::Mat &out_img2,const cv::Mat &H,
                 TentativeCorrespListExt matchings,
                 const int DrawCentersOnly = 1,
                 const int ReprojectToOneImage = 1,
                 const int r1=2,
                 const int r2=2,
                 const int drawEpipolarLines =0,
                 const double LAFcoef = 0,
                 const cv::Scalar color1= cv::Scalar(255,0,0),
                 const cv::Scalar color2= cv::Scalar(0,255,0));

#endif //MODS_NEW_IO_MODS_H
