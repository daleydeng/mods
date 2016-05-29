#ifndef MODS_COMMON_HPP
#define MODS_COMMON_HPP

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>

using std::vector;
using cv::Mat;

namespace mods {

enum detection_mode_t {FIXED_TH,
                       RELATIVE_TH,
                       FIXED_REG_NUMBER,
                       RELATIVE_REG_NUMBER,
                       NOT_LESS_THAN_REGIONS};

enum detector_type {DET_HESSIAN = 0,
                    DET_DOG = 1,
                    DET_HARRIS = 2,
                    DET_MSER = 3,
                    DET_ORB = 4,
                    DET_UNKNOWN = 1000};


const std::string _DetectorNames [] = {"HessianAffine", "DoG",
                                       "HarrisAffine", "MSER",
                                       "ORB"};


const std::vector<std::string> DetectorNames (_DetectorNames,_DetectorNames +
                                              sizeof(_DetectorNames)/sizeof(*_DetectorNames));

struct TimeLog
{
  double SynthTime;
  double DetectTime;
  double OrientTime;
  double DescTime;
  double MatchingTime;
  double RANSACTime;
  double MiscTime;
  double TotalTime;
  double SCVTime;
  TimeLog()
  {
    SynthTime=0.0;
    DetectTime=0.0;
    OrientTime=0.0;
    DescTime = 0.0;
    MatchingTime=0.0;
    RANSACTime=0.0;
    MiscTime=0.0;
    TotalTime = 0.0;
    SCVTime = 0.0;
  }
};

enum descriptor_type {DESC_SIFT = 0,
                      DESC_ROOT_SIFT = 1,
                      DESC_HALF_SIFT = 2,
                      DESC_HALF_ROOT_SIFT = 3,
                      DESC_INV_SIFT = 4,
                      DESC_ORB = 5,
                      DESC_PIXELS = 6,
                      DESC_UNKNOWN = 1000};


const std::string _DescriptorNames [] = {"SIFT", "RootSIFT",
                                     "HalfSIFT", "HalfRootSIFT",
                                     "InvSIFT", "ORB","Pixels"};

const std::vector<std::string> DescriptorNames (_DescriptorNames,_DescriptorNames +
                                              sizeof(_DescriptorNames)/sizeof(*_DescriptorNames));


/// Basic structures:

struct WLDParams
{
  double a; // WLD = a*DoG(px) / (I(px)/g + b) ;
  double b;
  double g;
  WLDParams()
  {
    a = 3.0;
    b = 5.0;
    g = 5.0;
  }
};


struct PyramidParams
{
  // shall input image be upscaled ( > 0)
  int upscaleInputImage;
  // number of scale per octave
  int  numberOfScales;
  // amount of smoothing applied to the initial level of first octave
  float initialSigma;
  // noise dependent threshold on the response (sensitivity)
  float threshold;
  float rel_threshold;
  int reg_number;
  float rel_reg_number;
  // ratio of the eigenvalues
  double edgeEigenValueRatio;
  // number of pixels ignored at the border of image
  int  border;
  int   doOnWLD; // detect Hessian points on WLD-transformed image
  int   doOnNormal; // detect Hessian points on normal image
  WLDParams WLDPar; //Parameters for WLD-transformation
  detection_mode_t DetectorMode;
  detector_type DetectorType;
  bool iiDoGMode;
  PyramidParams()
  {
    upscaleInputImage = 0;
    numberOfScales = 3;
    initialSigma = 1.6f;
    threshold = 16.0f/3.0f; //0.04f * 256 / 3;
    edgeEigenValueRatio = 10.0f;
    border = 5;
    doOnWLD = 0;
    doOnNormal = 1;
    DetectorMode = FIXED_TH;
    rel_threshold = -1;
    reg_number = -1;
    rel_reg_number = -1;
    DetectorType = DET_HESSIAN;
    iiDoGMode = false;
  }
};
struct Octave
{
  int    id;
  float  pixelDistance;
  float  initScale;

  std::vector<float> scales;
  std::vector<cv::Mat> blurs;
};

struct ScalePyramid
{
  PyramidParams par;
  ScalePyramid()
  {
  }
  std::vector<Octave> octaves;
};

struct SynthImage           // SynthImage: synthesised image from unwarped one
{
  int id;                 // image identifier
  std::string OrigImgName;   // filename of original image
  double tilt;            // tilt - scale factor in vertical direction. (y_synth=y_original / tilt)
  double rotation;        // angle of rotation, befote tilting. Counterclockwise, around top-left pixel, radians
  double zoom;            // scale factor, (x_synth,y_synth) = zoom*(x,y), before tilting and rotating
  double H[3*3];          // homography matrix from original image to synthesised
  cv::Mat pixels;         // image data
  ScalePyramid pyramid;
};

struct AffineKeypoint
{
  double x,y;            // subpixel, image coordinates
  double a11, a12, a21, a22;  // affine shape matrix
  double s;                   // scale
  double response;
  int octave_number;
  double pyramid_scale;
  int sub_type; //i.e. dark/bright for DoG
};

struct ViewSynthParameters
{
  double zoom;
  double tilt;
  double phi; //in radians
  double InitSigma;
  int doBlur;
  int DSPlevels;
  double minSigma;
  double maxSigma;
  std::vector<std::string> descriptors;
  std::map <std::string, double> FGINNThreshold;
  std::map <std::string, double> DistanceThreshold;
};

typedef std::map<std::string, std::vector<ViewSynthParameters> > IterationViewsynthesisParam;

typedef std::vector<float> descriptor_t;

struct AffineRegion{

  int img_id;              //image id, where shape detected
  int id;                  //region id
  int parent_id;
  detector_type type;
  AffineKeypoint det_kp;   //affine region in detected image
  AffineKeypoint reproj_kp;//reprojected affine region to the original image
  descriptor_type desc_type;
  descriptor_t desc;
};

struct PatchExtractionParams {

  int patchSize;
  double mrSize;
  bool FastPatchExtraction;
  bool photoNorm;
  PatchExtractionParams() {
    mrSize = 5.1962;
    patchSize = 41;
    FastPatchExtraction = false;
    photoNorm = true;
  }
};


typedef std::vector<AffineRegion> AffineRegionVector;
typedef std::map <std::string, AffineRegionVector> AffineRegionVectorMap;

struct WhatToMatch
{
  std::vector<std::string> group_detectors;
  std::vector<std::string> group_descriptors;
  std::vector<std::string> separate_detectors;
  std::vector<std::string> separate_descriptors;
};

struct WAVEParams{
  float b_wave;
  float r;
  bool pyramid;
  int s;
  int nms;
  int t;
  float k;
  bool doBaumberg;
  WAVEParams() {
    b_wave=0.166666;
    r=0.05;
    pyramid=true;
    s=12;
    nms=3;
    t=200;
    k = 0.16;
    doBaumberg = false;
  }
};


struct WASHParams{
  int threshold;
  bool doBaumberg;
  WASHParams() {
    threshold=100;
    doBaumberg = false;
  }
};

struct SFOPParams{
  float noise;
  int pThresh;
  float lWeight;
  int nOctaves;
  int nLayers;
  bool doBaumberg;
  SFOPParams() {
    noise=0.02;
    pThresh=0;
    lWeight=2;
    nOctaves=3;
    nLayers=4;
    doBaumberg = false;
  }
};



struct FOCIParams{
  int numberKPs;
  bool computeOrientation;
  bool secondOrientation;
  bool doBaumberg;
  FOCIParams() {
    numberKPs = 0;
    computeOrientation =true;
    secondOrientation = false;
    doBaumberg = false;
  }
};
struct SURFParams
{
  int octaves;
  int intervals;
  int init_sample;
  float thresh;
  bool doBaumberg;

  //  int patchSize;
  //  double mrSize;
  //  bool FastPatchExtraction;
  PatchExtractionParams PEParam;
  SURFParams()
  {
    octaves = 4;
    intervals = 4;
    init_sample=2;
    thresh =0.0004;
    doBaumberg = false;
    //   patchSize = 41;
    //    mrSize =  3.0*sqrt(3.0);
    //    FastPatchExtraction = false;
  }
};
struct FASTParams
{
  float threshold;
  bool nonmaxSuppression;
  int type;
  bool doBaumberg;
  FASTParams()
  {
    doBaumberg = false;
    threshold=10.0;
    nonmaxSuppression=true;
    type=0;
  }
};
struct STARParams
{
  int maxSize;
  int responseThreshold;
  int lineThresholdProjected;
  int lineThresholdBinarized;
  int suppressNonmaxSize;
  bool doBaumberg;
  STARParams()
  {
    doBaumberg = false;
    maxSize=45;
    responseThreshold=30;
    lineThresholdProjected=10;
    lineThresholdBinarized=8;
    suppressNonmaxSize=5;
  }
};
struct BRISKParams
{
  int thresh;
  int octaves;
  float patternScale;
  PatchExtractionParams PEParam;
  bool doBaumberg;
  //  int patchSize;
  //  double mrSize;
  //  bool FastPatchExtraction;
  BRISKParams()
  {
    doBaumberg = false;
    thresh=30;
    octaves=3;
    patternScale=1.0f;
    //   patchSize=41;
    //    mrSize = 3.0*sqrt(3.0);
    //    FastPatchExtraction = false;
  }
};
struct ReadAffsFromFileParams {
  std::string fname;
  ReadAffsFromFileParams() {
    fname="";
  }
};
struct ORBParams
{
  int nfeatures;
  float scaleFactor;
  int nlevels;
  int edgeThreshold;
  int firstLevel;
  int WTA_K;
  PatchExtractionParams PEParam;
  bool doBaumberg;
  //  int patchSize;
  //  double mrSize;
  //  bool FastPatchExtraction;
  //  bool photoNorm;
  ORBParams()
  {
    doBaumberg = false;
    nfeatures = 500;
    scaleFactor = 1.2;
    nlevels = 8;
    edgeThreshold = 31;
    firstLevel = 0;
    WTA_K=2;
    //    patchSize=31;
    //    mrSize = 3.0*sqrt(3.0);
    //    FastPatchExtraction = false;
    //    photoNorm =false;
  }
};

#define  GENERATE_MSER_PLUS    1
#define  GENERATE_MSER_MINUS   2

#define TIME_STATS                   0

enum EXTREMA_PREPROCESS
{
    PREPROCESS_CHANNEL_none          = 0x00000000,
    PREPROCESS_CHANNEL_intensity     = 0x00000001,
    PREPROCESS_CHANNEL_saturation    = 0x00000002,
    PREPROCESS_CHANNEL_hue           = 0x00000003,
    PREPROCESS_CHANNEL_redblue       = 0x00000004,
    PREPROCESS_CHANNEL_red           = 0x00000005,
    PREPROCESS_CHANNEL_green         = 0x00000006,
    PREPROCESS_CHANNEL_blue          = 0x00000007,
    PREPROCESS_CHANNEL_greenmagenta  = 0x00000008,
    PREPROCESS_CHANNEL_intensity_half= 0x00000009,

    PREPROCESS_CHANNEL_MASK          = 0x0000ffff,

    PREPROCESS_INTENSITY_none        = 0x00000000,
    PREPROCESS_INTENSITY_MASK        = 0xffff0000
};

//! A structure with MSER detector parameters.
struct ExtremaParams
{
    bool   relative;
    int    preprocess; /* see EXT_PREPROCESS enum */
    int    min_size;
    double max_area;
    double min_margin;
    bool   verbose;
    int    debug;
    bool   replace_with_ext;
    int    doOnWLD;
    int    doOnNormal;

    detection_mode_t DetectorMode;
    float rel_threshold;
    int reg_number;
    float rel_reg_number;

    WLDParams WLDPar; //Parameters for WLD-transformation

    ExtremaParams()
    {
        relative=false;
        preprocess=PREPROCESS_CHANNEL_none;
        max_area=0.01;
        min_size=30;
        min_margin=10;
        replace_with_ext=false;
        verbose=0;
        debug=0;
        doOnWLD=0;
        doOnNormal = 1;
        DetectorMode = FIXED_TH;
        rel_threshold = -1;
        reg_number = -1;
        rel_reg_number = -1;
    }
};

/**
 * @brief Possible invariants for Baumberg iteration
 */
enum AffineBaumbergMethod {
    AFF_BMBRG_SMM = 0, // Use Second Moment Matrix (original baumberg)
    AFF_BMBRG_HESSIAN = 1  // Use Hessian matrix
};

struct AffineShapeParams
{
  // number of affine shape interations
  int maxIterations;

  // convergence threshold, i.e. maximum deviation from isotropic shape at convergence
  float convergenceThreshold;

  // width and height of the SMM mask
  int smmWindowSize;

  // width and height of the patch
  int patchSize;

  // amount of smoothing applied to the initial level of first octave
  float initialSigma;

  // size of the measurement region (as multiple of the feature scale)
  float mrSize;

  int   doBaumberg;

  // Invariant used for Baumberg iteration
  AffineBaumbergMethod affBmbrgMethod;
  float affMeasRegion;

  AffineShapeParams()
  {
    maxIterations = 16;
    initialSigma = 1.6f;
    convergenceThreshold = 0.05;
    patchSize = 41;
    smmWindowSize = 19;
    mrSize = 3.0f*sqrt(3.0f);
    doBaumberg = 1;
    affBmbrgMethod = AFF_BMBRG_SMM;
    affMeasRegion = 0.5;
  }
};

struct ScaleSpaceDetectorParams
{
  AffineShapeParams AffineShapePars;
  PyramidParams PyramidPars;
};

struct DetectorsParameters
{
  ExtremaParams MSERParam;
  ScaleSpaceDetectorParams HessParam;
  ScaleSpaceDetectorParams HarrParam;
  ScaleSpaceDetectorParams DoGParam;
  SURFParams SURFParam;
  FASTParams FASTParam;
  STARParams STARParam;
  BRISKParams BRISKParam;
  ORBParams ORBParam;
  FOCIParams FOCIParam;
  ReadAffsFromFileParams ReadAffsFromFileParam;
  SFOPParams SFOPParam;
  WASHParams WASHParam;
  WAVEParams WAVEParam;
  AffineShapeParams BaumbergParam;
};

struct DominantOrientationParams {

  int maxAngles;
  float threshold;
  bool addUpRight;
  bool halfSIFTMode;
  PatchExtractionParams PEParam;
  DominantOrientationParams() {
    maxAngles = -1;
    threshold = 0.8;
    addUpRight = false;
    halfSIFTMode = false;
  }
};

struct SIFTDescriptorParams
{
  int spatialBins;
  int orientationBins;
  double maxBinValue;
  int patchSize;
  char useRootSIFT;
  bool FastPatchExtraction;
  int doHalfSIFT;
  int dims;
  int maxOrientations;
  bool estimateOrientation;
  double orientTh;
  bool doNorm;
  PatchExtractionParams PEParam;
  SIFTDescriptorParams()
  {
    spatialBins = 4;
    orientationBins = 8;
    maxBinValue = 0.2f;
    patchSize = 41;
    useRootSIFT=0;
    doHalfSIFT = 0;
    dims = spatialBins*spatialBins*orientationBins;
    maxOrientations = 0;
    estimateOrientation= true;
    doNorm=true;
    orientTh = 0.8;
  }
};

struct PIXELSDescriptorParams
{
    PatchExtractionParams PEParam;
//  int patchSize;
//  double mrSize;
//  bool FastPatchExtraction;
  std::string normType;
//  bool photoNorm;
  PIXELSDescriptorParams()
  {
//    patchSize = 41;
//    mrSize =  3.0*sqrt(3.0);
//    FastPatchExtraction = false;
    normType = "L2";
//    photoNorm =true;
  }
};

struct DescriptorsParameters {
  SIFTDescriptorParams SIFTParam;
  SIFTDescriptorParams RootSIFTParam;
  SIFTDescriptorParams HalfSIFTParam;
  SIFTDescriptorParams HalfRootSIFTParam;
  PIXELSDescriptorParams PixelsParam;
};

void solveLinear3x3(float *A, float *b);
bool getEigenvalues(float a, float b, float c, float d, float &l1, float &l2);
void invSqrt(float &a, float &b, float &c, float &l1, float &l2);
void computeGaussMask(cv::Mat &mask);
void computeCircularGaussMask(cv::Mat &mask, float sigma = 0);
void rectifyAffineTransformationUpIsUp(float &a11, float &a12, float &a21, float &a22);
void rectifyAffineTransformationUpIsUp(double *U);
void rectifyAffineTransformationUpIsUp(double &a11, double &a12, double &a21, double &a22);

bool interpolate(const cv::Mat &im,const float ofsx,const float ofsy,
                 const float a11,const float a12,const float a21,const float a22, cv::Mat &res);

bool interpolateCheckBorders(const cv::Mat &im, const float ofsx, const float ofsy,
                             const float a11,const float a12,const float a21,const float a22, const cv::Mat &res);

bool interpolateCheckBorders(const int orig_img_w, const int orig_img_h, const float ofsx, const float ofsy,
                             const float a11, const float a12,const float a21,const float a22, const int res_w, const int res_h);

void photometricallyNormalize(cv::Mat &image, const cv::Mat &weight_mask, float &sum, float &var);

cv::Mat gaussianBlur(const cv::Mat input, float sigma);
void gaussianBlurInplace(cv::Mat &inplace, float sigma);
cv::Mat doubleImage(const cv::Mat &input);
cv::Mat halfImage(const cv::Mat &input);
float atan2LUTff(float y,float x);

void computeGradient(const cv::Mat &img, cv::Mat &gradx, cv::Mat &grady);
void computeGradientMagnitudeAndOrientation(const cv::Mat &img, cv::Mat &mag, cv::Mat &ori);
double getTime();

struct DescriptorFunctor {
  virtual void operator()(cv::Mat &patch, std::vector<float>& desc) = 0;
  descriptor_type type;
};

void rectifyTransformation(double &a11, double &a12, double &a21, double &a22);
void rectifyTransformation(AffineKeypoint &k);

} //namespace mods

#endif // MODS_COMMON_HPP
