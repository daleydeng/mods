/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <vector>
#include "matching.hpp"

#define WITH_ORSA

namespace mods {

struct drawingParams
{
    int writeImages;
    int drawEpipolarLines;
    int drawOnlyCenters;
    int drawReprojected;
    drawingParams()
    {
        writeImages = 1;
        drawOnlyCenters = 1;
        drawEpipolarLines = 0;
        drawReprojected = 1;
    }
};

struct outputParams
{
    int verbose;
    int timeLog;
    int writeKeypoints;
    int writeMatches;
    int featureComplemetaryLog;
    int outputAllTentatives;
    int outputEstimatedHorF;
    outputParams()
    {
        verbose = 0;
        timeLog = 1;
        writeKeypoints = 1;
        writeMatches = 1;
        outputAllTentatives = 0;
        featureComplemetaryLog = 0;
        outputEstimatedHorF = 0;
    }
};

struct filteringParams
{
    int useSCV;
    int doBeforeRANSAC;
    double duplicateDist;
    int mode;
    filteringParams()
    {
        useSCV = 0;
        doBeforeRANSAC = 1;
        duplicateDist = 3.0;
        mode = MODE_RANDOM;
    }
};

struct parameters
{
    char* img1_fname;
    char* img2_fname;
    char* out1_fname;
    char* out2_fname;
    char* k1_fname;
    char* k2_fname;
    char* matchings_fname;
    char* log_fname;
    char* ground_truth_fname;
    char* config_fname;
    char* iters_fname;
    int doCLAHE;
    int det_type;
    RANSAC_mode_t ver_type;
    int tilt_numb;
    int rot_numb;
    double phi;
    double zoom;
    double initSigma;
    char doBlur;
    std::vector <double> tilt_set;
    std::vector <double> scale_set;
    int logOnly;
    parameters()
    {
        config_fname="config_iter.ini";
        iters_fname="iters.ini";
        det_type = HESAFF;
        ver_type = Homog;
        tilt_numb = 2;
        phi = 72.;
        rot_numb = 1;
        zoom = 1.0;
        initSigma = 0.5;
        doBlur = 1;
        logOnly = 1;
        doCLAHE = 0;
      //  overlap_error = 0.04;
        tilt_set.push_back(1.0);
        scale_set.push_back(1.0);
    }
};
struct logs
{
    int TrueMatch;
    int TrueMatch1st;
    int TrueMatch1stRANSAC;

    int Tentatives;
    int Tentatives1st;
    int Tentatives1stRANSAC;

    double InlierRatio1st;
    double InlierRatio1stRANSAC;

    int OtherTrueMatch;
    int OtherTrueMatch1st;
    int OtherTrueMatch1stRANSAC;

    double OtherInlierRatio1st;
    double OtherInlierRatio1stRANSAC;

    int OtherTentatives;
    int OtherTentatives1st;
    int OtherTentatives1stRANSAC;

    int OrientReg1;
    int OrientReg2;

    int UnorientedReg1;
    int UnorientedReg2;
    double TotalArea;
    int Syms;
    double FinalTime;
    int OverlapMatches;
    int FinalStep;
    RANSAC_mode_t VerifMode;

    double densificationCoef;
    logs()
    {
        TrueMatch = 0;
        TrueMatch1st = 0;
        TrueMatch1stRANSAC = 0;

        Tentatives = 0;
        Tentatives1st = 0;
        Tentatives1stRANSAC = 0;

        OtherTrueMatch = 0;
        OtherTrueMatch1st = 0;
        OtherTrueMatch1stRANSAC = 0;

        OtherTentatives = 0;
        OtherTentatives1st = 0;
        OtherTentatives1stRANSAC = 0;

        OrientReg1 = 0;
        OrientReg2 = 0;
        UnorientedReg1 = 0;
        UnorientedReg2 = 0;
        Syms = 0;
        FinalTime = 0;
        OverlapMatches = 0;
        TotalArea = 1;
        FinalStep=1;
        densificationCoef = 1.0;
    }
};

} //namespace mods
#endif // CONFIGURATION_HPP
