/*-----------------------------------------------------------*/
/* Copyright 2013-2015, Dmytro Mishkin  ducha.aiki@gmail.com */
/*----------------------------------------------------------*/

#undef __STRICT_ANSI__
#include <fstream>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include "io_mods.h"
#include "matching.hh"
#include "configuration.hpp"
#include "imagerepresentation.hh"
#include "correspondencebank.hh"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace mods;

const int nn_n = 50; //number of nearest neighbours retrieved to get 1st inconsistent

inline long getMilliSecs()
{
  timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec*1000 + t.tv_usec/1000;
}

int main(int argc, char **argv)
{
  srand(1);
  if ((argc < Tmin))
    {
      std::cerr << " ************************************************************************** " << std::endl
                << " ******** Two-view Matching with On-Demand Synthesis ********************** " << std::endl
                << " ************************************************************************** " << std::endl
                << "Usage: " << argv[0] << " imgIn1.png imgIn2.png imgOut1.png imgOut2.png " << std::endl
                << "           keys-1.txt keys2.txt matchings.txt log.txt 0 H config_iter.ini iters.ini" << std::endl
                << "- imgIn1.png, imgIn2.png: input images " << std::endl
                << "- imgOut1.png, imgOut2.png: output images " << std::endl
                << "  The detected matchings are represented by green and blue dots" << std::endl
                << "- keys1.txt keys2.txt: affine regions and their SIFTs of the two images." << std::endl
                << "- matchings.txt: coordinates of matched points (col1, row1, col2, row2). " << std::endl
                << "- log.txt - log-file for graphs" << std::endl
                << "- write log file only [0/1]" << std::endl
                << "- geometric model type [H/F] " << std::endl
                << "- config_iter.ini: input file with detectors and descriptors paramaters [optional, default = 'config_iter.ini'] " << std::endl
                << "- iters.ini: input file with parameters of iterative view synthesis [optional, default= 'iters.ini']" << std::endl
                << " ******************************************************************************* " << std::endl;
      return 1;
    }
  long c_start = getMilliSecs();
  double time1;
  TimeLog TimingLog;
  logs log1;
  /// Parameters reading
  configs Config1;
  if (getCLIparam(Config1,argc,argv)) return 1;
  int VERB = Config1.OutputParam.verbose;
  /// Ground truth homography reading
  log1.VerifMode =  Config1.CLIparams.ver_type;
  double Hready[3*3];

  /// Input images reading
  cv::Mat img1, img2;
  SynthImage tilt_img1,tilt_img2;
  tilt_img1.id=0;
  tilt_img2.id=1000;
#pragma omp parallel
  {
#pragma omp single nowait
    {
#pragma omp task

      img1 = cv::imread(Config1.CLIparams.img1_fname, 1); //
      cv::Mat gray_in_img;
      if (img1.channels() == 3)
        {
          //cv::cvtColor(in_img, gray_in_img, CV_BGR2GRAY);
          std::vector<cv::Mat> RGB_planes(3);
          cv::Mat in_32f;
          img1.convertTo(in_32f,CV_32FC3);
          cv::split(in_32f, RGB_planes);
          // gray_in_img = cv::Mat::zeros(in_img.cols, in_img.rows,CV_32FC1);
          gray_in_img = (RGB_planes[0] + RGB_planes[1] + RGB_planes[2]) / 3.0 ;
        } else
        {
          gray_in_img = img1;
        }
      // img1 = gray_in_img;
#pragma omp task
      img2 = cv::imread(Config1.CLIparams.img2_fname, 1); //?
      cv::Mat gray_in_img2;
      if (img2.channels() == 3)
        {
          //cv::cvtColor(in_img, gray_in_img, CV_BGR2GRAY);
          std::vector<cv::Mat> RGB_planes2(3);
          cv::Mat in_32f2;
          img2.convertTo(in_32f2,CV_32FC3);
          cv::split(in_32f2, RGB_planes2);
          // gray_in_img = cv::Mat::zeros(in_img.cols, in_img.rows,CV_32FC1);
          gray_in_img2 = (RGB_planes2[0] + RGB_planes2[1] + RGB_planes2[2]) / 3.0 ;
        } else
        {
          gray_in_img2 = img2;
        }
      // img2 = gray_in_img2;
    }
#pragma omp taskwait
  }
  if(!img1.data) {
      std::cerr <<  "Could not open or find the image1 " << Config1.CLIparams.img1_fname << std::endl;
      return 1;
    }
  if(!img2.data) {
      std::cerr <<  "Could not open or find the image2 " << Config1.CLIparams.img2_fname << std::endl;
      return 1;
    }

  /// Data structures preparation
  ImageRepresentation ImgRep1,ImgRep2;

  ImgRep1 = ImageRepresentation(img1,Config1.CLIparams.img1_fname);
  ImgRep2 = ImageRepresentation(img2,Config1.CLIparams.img2_fname);

  CorrespondenceBank Tentatives;
  std::map<std::string, TentativeCorrespListExt> tentatives, verified_coors;

  int final_step = 0;
  int curr_matches = 0;

  /// Affine regions detection
  std::cerr << "View synthesis, detection and description..." << endl;
  if (Config1.read_pre_extracted)
    {
      Config1.Matchparam.maxSteps = 1;
    }
  /// Main program loop
  for (int step=0; (step < Config1.Matchparam.maxSteps)
       && (curr_matches < Config1.Matchparam.minMatches); step++, final_step++)
    {
      double parallel_curr_start = getMilliSecs();
      if (VERB)
        {
          std::cerr << "Iteration " << step << std::endl;
          for (unsigned int det=0; det < DetectorNames.size(); det++)
            {
              unsigned int n_synths = Config1.ItersParam[step][DetectorNames[det]].size();
              if (n_synths > 0)
                std::cerr << DetectorNames[det] << ": " << n_synths << " synthesis will be done." << std::endl;
            }
        }
      if (Config1.read_pre_extracted) {
          ImgRep1.LoadRegions(Config1.CLIparams.k1_fname);
          ImgRep2.LoadRegions(Config1.CLIparams.k2_fname);
        } else {
#ifdef _OPENMP
          omp_set_nested(1);
#endif
#pragma omp parallel
          {
#pragma omp single nowait
            {
#pragma omp task
              ImgRep1.SynthDetectDescribeKeypoints(Config1.ItersParam[step],
                                                   Config1.DetectorsPars,
                                                   Config1.DescriptorPars,
                                                   Config1.DomOriPars);
#pragma omp task
              ImgRep2.SynthDetectDescribeKeypoints(Config1.ItersParam[step],
                                                   Config1.DetectorsPars,
                                                   Config1.DescriptorPars,
                                                   Config1.DomOriPars);
            }
#pragma omp taskwait
          }
        }
      TimeLog img1time = ImgRep1.GetTimeSpent();
      TimeLog img2time = ImgRep2.GetTimeSpent();
      double parallel_curr_end = ((double)(getMilliSecs() - parallel_curr_start))/1000;
      double sum1 = img1time.DescTime+img1time.DetectTime+img1time.OrientTime+img1time.SynthTime;
      double sum2 = img2time.DescTime+img2time.DetectTime+img2time.OrientTime+img2time.SynthTime;
      double sum_time = sum1+sum2;
      if (sum_time > 0)
        {
          TimingLog.DescTime += (img1time.DescTime+img2time.DescTime)*parallel_curr_end /sum_time;
          TimingLog.DetectTime += (img1time.DetectTime+img2time.DetectTime)*parallel_curr_end /sum_time;
          TimingLog.OrientTime += (img1time.OrientTime+img2time.OrientTime)*parallel_curr_end /sum_time;
          TimingLog.SynthTime += (img1time.SynthTime+img2time.SynthTime)*parallel_curr_end /sum_time;
        }
      /// Preparation for matching
      double curr_start = getMilliSecs();
      if (step == 2) {
        Tentatives.ClearCorrespondences("ORB", "ORB");
        std::cerr << "Discarding ORB features" << std::endl;
        }

      Tentatives.MatchImgReps(ImgRep1,ImgRep2,Config1.ItersParam[step],Config1.Matchparam.IterWhatToMatch[step],
                              Config1.Matchparam,Config1.DescriptorPars);

      time1 = ((double)(getMilliSecs() - curr_start))/1000;
      TimingLog.MatchingTime +=time1;

      /// Geometric verification
      //Change afterwards
      tentatives["All"] = Tentatives.GetCorresponcesVector();
      if (VERB) std::cerr << tentatives["All"].TCList.size() << " tentatives found." << endl;
      if (Config1.FilterParam.doBeforeRANSAC) //duplicate before RANSAC
        {
          if (VERB) std::cerr << "Duplicate filtering before RANSAC with threshold = "
                              << Config1.FilterParam.duplicateDist << " pixels." << endl;
          DuplicateFiltering(tentatives["All"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);
          if (VERB) std::cerr << tentatives["All"].TCList.size() << " unique tentatives left" << endl;
        }
      curr_matches=log1.TrueMatch1st;

      log1.Tentatives1st = tentatives["All"].TCList.size();
      curr_start = getMilliSecs();
      log1.TrueMatch1st =  USACFiltering(tentatives["All"],
          verified_coors["All"],
          verified_coors["All"].H,
          Config1.RANSACParam);

      log1.InlierRatio1st = (double) log1.TrueMatch1st / (double) log1.Tentatives1st;
      if (VERB) std::cerr << log1.TrueMatch1st  << " RANSAC correspondences got" << endl;

      time1 = ((double)(getMilliSecs() - curr_start))/1000;

      if (!Config1.FilterParam.doBeforeRANSAC) //duplicate after RANSAC
        {
          if (VERB) std::cerr << "Duplicate filtering after RANSAC with threshold = "
                              << Config1.FilterParam.duplicateDist << " pixels." << endl;

          DuplicateFiltering(verified_coors["All"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);
          DuplicateFiltering(verified_coors["AllRANSAC"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);
          DuplicateFiltering(verified_coors["AllRANSACverified"], Config1.FilterParam.duplicateDist,Config1.FilterParam.mode);
          log1.TrueMatch1stRANSAC = verified_coors["AllRANSACverified"].TCList.size();
          log1.TrueMatch1st = verified_coors["All"].TCList.size();
          log1.Tentatives1stRANSAC = verified_coors["AllRANSAC"].TCList.size();
          if (VERB) std::cerr << verified_coors["All"].TCList.size() << " unique matches left" << endl;
        }
      curr_matches=log1.TrueMatch1st;

      time1 = ((double)(getMilliSecs() - c_start))/1000;
      double time2 = ((double)(getMilliSecs() - curr_start))/1000;

      TimingLog.RANSACTime +=time2;
      log1.FinalTime = time1;

      curr_matches=log1.TrueMatch1st;

    }
  log1.UnorientedReg1 = ImgRep1.GetRegionsNumber();
  log1.UnorientedReg2 = ImgRep2.GetRegionsNumber();

  log1.OrientReg1 = ImgRep1.GetDescriptorsNumber() - ImgRep1.GetDescriptorsNumber("None");
  log1.OrientReg2 = ImgRep2.GetDescriptorsNumber() - ImgRep2.GetDescriptorsNumber("None");

  log1.FinalStep = final_step;
  std::cerr << "Done in " << final_step << " iterations" << endl;
  std::cerr << "*********************" << endl;

  /// Writing images and logs
  std::cerr << "Writing files... " << endl;

  ofstream file_log(Config1.CLIparams.log_fname);
  if (file_log.is_open())
    WriteLog(log1, file_log);
  file_log.close();

  if (Config1.OutputParam.outputEstimatedHorF) {
      if (!Config1.RANSACParam.useF)
        {
          ofstream fileH(argv[Tmin+2]);
          //ofstream fileH("Hmatrix.txt");
          if (fileH.is_open())
            WriteH(verified_coors["All"].H,fileH);
          fileH.close();
        }
      else
        {
          ofstream fileH(argv[Tmin+2]);
          //ofstream fileH("Fmatrix.txt");
          if (fileH.is_open())
            WriteH(verified_coors["All"].H,fileH);
          fileH.close();
        }
    }
  if (!Config1.CLIparams.logOnly)
    {
      if (Config1.OutputParam.outputAllTentatives)
        std::cerr << "Warning! Matchings file contains all tentative correspondences! (not all are correct)" << std::endl;
      //match 2nd cl
      if (Config1.OutputParam.writeMatches)
        {
          ofstream file_match(Config1.CLIparams.matchings_fname);
          if (file_match.is_open())
            WriteMatchings(verified_coors["All"],file_match, Config1.OutputParam.outputAllTentatives);
          file_match.close();
        }
      if (Config1.OutputParam.writeKeypoints && !Config1.read_pre_extracted)
        {
          ImgRep1.SaveRegions(Config1.CLIparams.k1_fname,0);
          ImgRep2.SaveRegions(Config1.CLIparams.k2_fname,0);
        }
      if (Config1.DrawParam.writeImages)
        {
          cv::Mat img_out1s, img_out2s;

          cv::Mat h1cv(3,3,CV_64F,verified_coors["All"].H);
          cv::Mat h1inv(3,3,CV_64F);
          cv::invert(h1cv,h1inv,cv::DECOMP_LU);

          DrawMatches(ImgRep1.OriginalImg,ImgRep2.OriginalImg,img_out1s,img_out2s,h1cv,verified_coors["All"],
              Config1.DrawParam.drawOnlyCenters,
              (!Config1.RANSACParam.useF && Config1.DrawParam.drawReprojected),5,4,
              (Config1.RANSACParam.useF && Config1.DrawParam.drawEpipolarLines),0,
              0);
          cv::imwrite(Config1.CLIparams.out1_fname,img_out1s);
          cv::imwrite(Config1.CLIparams.out2_fname,img_out2s);
        }
    }
  /// Console output, quite ugly :(
  std::cerr << "Image1: regions descriptors | Image2: regions descriptors " << endl;
  std::cerr << log1.UnorientedReg1 << " " << log1.OrientReg1 << " | " << log1.UnorientedReg2 << " " << log1.OrientReg2 << std::endl;
  std::cerr << std::endl;
  std::cerr << "True matches | unique tentatives" << endl;
  if (log1.InlierRatio1st == log1.InlierRatio1st)
    std::cerr << log1.TrueMatch1st << " | " << log1.Tentatives1st << " | " << std::setprecision(3) << log1.InlierRatio1st*100 <<"% " << Config1.descriptor << " 1st geom inc" << std::endl;
  else
    std::cerr << log1.TrueMatch1st << " | " << log1.Tentatives1st << " | " << " - " << Config1.descriptor << " 1st geom inc" << std::endl;
  std::cerr << std::endl;

  long c_end = getMilliSecs();
  std::cerr << "Main matching | All Time: " << endl;
  std::cerr << log1.FinalTime << " | " << ((double)(c_end - c_start))/1000 << " seconds" << std::endl;

  TimingLog.TotalTime = double(c_end - c_start)/1000;
  TimingLog.MiscTime = TimingLog.TotalTime -
      (TimingLog.SynthTime +TimingLog.DescTime + TimingLog.DetectTime +
       TimingLog.MatchingTime + TimingLog.OrientTime + TimingLog.RANSACTime);

  if (Config1.OutputParam.timeLog)
    {
      WriteTimeLog(TimingLog, std::cerr,1,1,1);
      ofstream file_log1("time.log");
      if (file_log1.is_open())
        WriteTimeLog(TimingLog, file_log1,1,0,0);
      file_log1.close();
    }

  return 0;
}
