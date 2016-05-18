#ifndef HOMOGESTIMATOR_H
#define HOMOGESTIMATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include "../config/ConfigParamsHomog.h"
#include "../utils/MathFunctions.h"
#include "../utils/FundmatrixFunctions.h"
#include "../utils/HomographyFunctions.h"
#include "USAC.h"

//Taken from homest
#include <math.h>

const double ratio_degen_test = 0.001;
double determinant3x3(double M[9]) {
  return M[0]*M[4]*M[8] + M[1]*M[5]*M[6] + M[2]*M[3]*M[7] -
      M[2]*M[4]*M[6] -M[1]*M[3]*M[8] - M[0]*M[5]*M[7] ;
}

double calcSymmetricTransferErrorSum(double m1[3],double m2[3],double h[9], double h_inv[9])
{

  double h_x[3], h_inv_xp[3];
  MathTools::vmul(h_x, h, m1, 3);
  MathTools::vmul(h_inv_xp, h_inv, m2, 3);

  double err1 = 0.0, err2 = 0.0;
  for (unsigned int j = 0; j < 2; ++j)
    {
      err1 += (h_x[j]/h_x[2] - m2[j]) * (h_x[j]/h_x[2] - m2[j]);
      err2 += (h_inv_xp[j]/h_inv_xp[2] - m1[j]) * (h_inv_xp[j]/h_inv_xp[2] - m1[j]);
    }
  return err1 + err2;
}
double calcSymmetricTransferErrorMax(double m1[3],double m2[3],double h[9], double h_inv[9])
{
  double h_x[3], h_inv_xp[3];
  MathTools::vmul(h_x, h, m1, 3);
  MathTools::vmul(h_inv_xp, h_inv, m2, 3);

  double err1 = 0.0, err2 = 0.0;
  for (unsigned int j = 0; j < 2; ++j)
    {
      err1 += (h_x[j]/h_x[2] - m2[j]) * (h_x[j]/h_x[2] - m2[j]);
      err2 += (h_inv_xp[j]/h_inv_xp[2] - m1[j]) * (h_inv_xp[j]/h_inv_xp[2] - m1[j]);
    }
  return std::max(err1, err2);
}

double  calc2DHomogSampsonErr(double m1[2],double m2[2],double h[9])
{
  double t1;
  double t10;
  double t100;
  double t104;
  double t108;
  double t112;
  double t118;
  double t12;
  double t122;
  double t125;
  double t126;
  double t129;
  double t13;
  double t139;
  double t14;
  double t141;
  double t144;
  double t15;
  double t150;
  double t153;
  double t161;
  double t167;
  double t17;
  double t174;
  double t18;
  double t19;
  double t193;
  double t199;
  double t2;
  double t20;
  double t201;
  double t202;
  double t21;
  double t213;
  double t219;
  double t22;
  double t220;
  double t222;
  double t225;
  double t23;
  double t236;
  double t24;
  double t243;
  double t250;
  double t253;
  double t26;
  double t260;
  double t27;
  double t271;
  double t273;
  double t28;
  double t29;
  double t296;
  double t3;
  double t30;
  double t303;
  double t31;
  double t317;
  double t33;
  double t331;
  double t335;
  double t339;
  double t34;
  double t342;
  double t345;
  double t35;
  double t350;
  double t354;
  double t36;
  double t361;
  double t365;
  double t37;
  double t374;
  double t39;
  double t4;
  double t40;
  double t41;
  double t42;
  double t43;
  double t44;
  double t45;
  double t46;
  double t47;
  double t49;
  double t51;
  double t57;
  double t6;
  double t65;
  double t66;
  double t68;
  double t69;
  double t7;
  double t72;
  double t78;
  double t8;
  double t86;
  double t87;
  double t90;
  double t95;
  {
    t1 = m2[0];
    t2 = h[6];
    t3 = t2*t1;
    t4 = m1[0];
    t6 = h[7];
    t7 = t1*t6;
    t8 = m1[1];
    t10 = h[8];
    t12 = h[0];
    t13 = t12*t4;
    t14 = h[1];
    t15 = t14*t8;
    t17 = t3*t4+t7*t8+t1*t10-t13-t15-h[2];
    t18 = m2[1];
    t19 = t18*t18;
    t20 = t2*t2;
    t21 = t19*t20;
    t22 = t18*t2;
    t23 = h[3];
    t24 = t23*t22;
    t26 = t23*t23;
    t27 = t6*t6;
    t28 = t19*t27;
    t29 = t18*t6;
    t30 = h[4];
    t31 = t29*t30;
    t33 = t30*t30;
    t34 = t4*t4;
    t35 = t20*t34;
    t36 = t2*t4;
    t37 = t6*t8;
    t39 = 2.0*t36*t37;
    t40 = t36*t10;
    t41 = 2.0*t40;
    t42 = t8*t8;
    t43 = t42*t27;
    t44 = t37*t10;
    t45 = 2.0*t44;
    t46 = t10*t10;
    t47 = t21-2.0*t24+t26+t28-2.0*t31+t33+t35+t39+t41+t43+t45+t46;
    t49 = t12*t12;
    t51 = t6*t30;
    t57 = t20*t2;
    t65 = t1*t1;
    t66 = t65*t20;
    t68 = t65*t57;
    t69 = t4*t10;
    t72 = t2*t49;
    t78 = t27*t6;
    t86 = t65*t78;
    t87 = t8*t10;
    t90 = t65*t27;
    t95 = -2.0*t49*t18*t51-2.0*t3*t12*t46-2.0*t1*t57*t12*t34-2.0*t3*t12*t33+t66
        *t43+2.0*t68*t69+2.0*t72*t69-2.0*t7*t14*t46-2.0*t1*t78*t14*t42-2.0*t7*t14*t26+
        2.0*t86*t87+t90*t35+2.0*t49*t6*t87;
    t100 = t14*t14;
    t104 = t100*t2;
    t108 = t2*t23;
    t112 = t78*t42*t8;
    t118 = t57*t34*t4;
    t122 = t10*t26;
    t125 = t57*t4;
    t126 = t10*t19;
    t129 = t78*t8;
    t139 = -2.0*t57*t34*t18*t23+2.0*t100*t6*t87+2.0*t104*t69-2.0*t100*t18*t108+
        4.0*t36*t112+6.0*t43*t35+4.0*t118*t37+t35*t28+2.0*t36*t122+2.0*t125*t126+2.0*
        t129*t126+2.0*t37*t122-2.0*t78*t42*t18*t30+t43*t21;
    t141 = t10*t33;
    t144 = t46*t18;
    t150 = t46*t19;
    t153 = t46*t10;
    t161 = t27*t27;
    t167 = 2.0*t36*t141-2.0*t144*t108+2.0*t37*t141+t66*t33+t150*t27+t150*t20+
        4.0*t37*t153+6.0*t43*t46+4.0*t112*t10+t43*t33+t161*t42*t19+t43*t26+4.0*t36*t153
        ;
    t174 = t20*t20;
    t193 = 6.0*t35*t46+4.0*t10*t118+t35*t33+t35*t26+t174*t34*t19+t100*t27*t42+
        t100*t20*t34+t100*t19*t20+t90*t46+t65*t161*t42+t90*t26+t49*t27*t42+t49*t20*t34+
        t49*t19*t27;
    t199 = t34*t34;
    t201 = t12*t23;
    t202 = t14*t30;
    t213 = t42*t42;
    t219 = t66*t46+t100*t26+t46*t100+t174*t199-2.0*t201*t202-2.0*t144*t51+t46*
        t26+t65*t174*t34+t49*t33+t49*t46+t46*t33+t161*t213-2.0*t7*t14*t20*t34;
    t220 = t1*t27;
    t222 = t36*t8;
    t225 = t7*t14;
    t236 = t4*t6*t8;
    t243 = t3*t12;
    t250 = t46*t46;
    t253 = t1*t20;
    t260 = -4.0*t220*t14*t222-4.0*t225*t40-4.0*t220*t15*t10+2.0*t90*t40+2.0*
        t225*t24+2.0*t72*t236-2.0*t3*t12*t27*t42-4.0*t243*t44+2.0*t66*t44+2.0*t243*t31+
        t250+2.0*t68*t236-4.0*t253*t12*t236-4.0*t253*t13*t10;
    t271 = t4*t20;
    t273 = t8*t18;
    t296 = t10*t18;
    t303 = 2.0*t104*t236-2.0*t35*t31+12.0*t35*t44+2.0*t125*t37*t19-4.0*t271*t6*
        t273*t23+2.0*t36*t37*t26+2.0*t36*t129*t19-4.0*t36*t27*t273*t30+2.0*t36*t37*t33+
        12.0*t36*t43*t10+12.0*t36*t37*t46-4.0*t271*t296*t23+2.0*t36*t126*t27;
    t317 = t18*t14;
    t331 = t14*t2;
    t335 = t12*t18;
    t339 = t220*t18;
    t342 = t7*t30;
    t345 = t317*t6;
    t350 = -4.0*t31*t40-2.0*t43*t24+2.0*t37*t126*t20-4.0*t44*t24-4.0*t27*t8*
        t296*t30-2.0*t253*t317*t30-2.0*t65*t2*t23*t6*t30+2.0*t3*t23*t14*t30-2.0*t12*t19
        *t331*t6+2.0*t335*t331*t30-2.0*t201*t339+2.0*t201*t342+2.0*t201*t345+2.0*t86*
        t222;
    t354 = 1/(t95+t139+t167+t193+t219+t260+t303+t350);
    t361 = t22*t4+t29*t8+t296-t23*t4-t30*t8-h[5];
    t365 = t253*t18-t3*t23-t335*t2+t201+t339-t342-t345+t202;
    t374 = t66-2.0*t243+t49+t90-2.0*t225+t100+t35+t39+t41+t43+t45+t46;
    return sqrt((t17*t47*t354-t361*t365*t354)*t17+(-t17*t365*t354+t361*t374*
                                                   t354)*t361);

  }
}



//

class HomogEstimator: public USAC<HomogEstimator>
{
public:
  inline bool		 initProblem(const ConfigParamsHomog& cfg, double* pointData);
  // ------------------------------------------------------------------------
  // storage for the final result
  std::vector<double> final_model_params_;

public:
  HomogEstimator()
  {
    input_points_ = NULL;
    data_matrix_  = NULL;
    models_.clear();
    models_denorm_.clear();
  };
  ~HomogEstimator()
  {
    if (input_points_) { delete[] input_points_; input_points_ = NULL; }
    if (data_matrix_) { delete[] data_matrix_; data_matrix_ = NULL; }
    for (size_t i = 0; i < models_.size(); ++i)
      {
        if (models_[i]) { delete[] models_[i]; }
      }
    models_.clear();
    for (size_t i = 0; i < models_denorm_.size(); ++i)
      {
        if (models_denorm_[i]) { delete[] models_denorm_[i]; }
      }
    models_denorm_.clear();
  };

public:
  // ------------------------------------------------------------------------
  // problem specific functions
  inline void		 cleanupProblem();
  inline unsigned int generateMinimalSampleModels();
  inline bool		 generateRefinedModel(std::vector<unsigned int>& sample, const unsigned int numPoints,
                                              bool weighted = false, double* weights = NULL);
  inline bool		 validateSample();
  inline bool		 validateModel(unsigned int modelIndex);
  inline bool		 evaluateModel(unsigned int modelIndex, unsigned int* numInliers, unsigned int* numPointsTested);
  inline void		 testSolutionDegeneracy(bool* degenerateModel, bool* upgradeModel);
  inline unsigned int upgradeDegenerateModel();
  inline void		 findWeights(unsigned int modelIndex, const std::vector<unsigned int>& inliers,
                                     unsigned int numInliers, double* weights);
  inline void		 storeModel(unsigned int modelIndex, unsigned int numInliers);

private:
  double*      input_points_denorm_;					// stores pointer to original input points

  // ------------------------------------------------------------------------
  // temporary storage
  double* input_points_;							// stores normalized data points
  double* data_matrix_;							// linearized input data
  double  m_T1_[9], m_T2_[9], m_T2inv_[9];			// normalization matrices
  std::vector<double*> models_;				    // stores vector of models
  std::vector<double*> models_denorm_;			// stores vector of (denormalized) models
};


// ============================================================================================
// initProblem: initializes problem specific data and parameters
// this function is called once per run on new data
// ============================================================================================
bool HomogEstimator::initProblem(const ConfigParamsHomog& cfg, double* pointData)
{
  // copy pointer to input data
  input_points_denorm_ = pointData;
  input_points_       = new double[6*cfg.common.numDataPoints];
  if (input_points_denorm_ == NULL)
    {
      std::cerr << "Input point data not properly initialized" << std::endl;
      return false;
    }
  if (input_points_ == NULL)
    {
      std::cerr << "Could not allocate storage for normalized data points" << std::endl;
      return false;
    }

  // normalize input data
  // following this, input_points_ has the normalized points and input_points_denorm_ has
  // the original input points
  FTools::normalizePoints(input_points_denorm_, input_points_, cfg.common.numDataPoints, m_T1_, m_T2_);
  for (unsigned int i = 0; i < 9; ++i)
    {
      m_T2inv_[i] = m_T2_[i];
    }
  MathTools::minv(m_T2inv_, 3);

  // allocate storage for models
  final_model_params_.clear(); final_model_params_.resize(9);
  models_.clear(); models_.resize(usac_max_solns_per_sample_);
  models_denorm_.clear(); models_denorm_.resize(usac_max_solns_per_sample_);
  for (unsigned int i = 0; i < usac_max_solns_per_sample_; ++i)
    {
      models_[i] = new double[9];
      models_denorm_[i] = new double[9];
    }

  // precompute the data matrix
  data_matrix_ = new double[18*usac_num_data_points_];	// 2 equations per correspondence
  HTools::computeDataMatrix(data_matrix_, usac_num_data_points_, input_points_);

  return true;
}


// ============================================================================================
// cleanupProblem: release any temporary problem specific data storage 
// this function is called at the end of each run on new data
// ============================================================================================
void HomogEstimator::cleanupProblem()
{
  if (input_points_) { delete[] input_points_; input_points_ = NULL; }
  if (data_matrix_) { delete[] data_matrix_; data_matrix_ = NULL; }
  for (size_t i = 0; i < models_.size(); ++i)
    {
      if (models_[i]) { delete[] models_[i]; }
    }
  models_.clear();
  for (size_t i = 0; i < models_denorm_.size(); ++i)
    {
      if (models_denorm_[i]) { delete[] models_denorm_[i]; }
    }
  models_denorm_.clear();
}


// ============================================================================================
// generateMinimalSampleModels: generates minimum sample model from the data points whose  
// indices are currently stored in min_sample_. 
// in this case, only one model per minimum sample
// ============================================================================================
unsigned int HomogEstimator::generateMinimalSampleModels()
{
  double A[8*9];
  double At[9*8];

  // form the matrix of equations for this minimal sample
  double *src_ptr;
  double *dst_ptr = A;
  for (unsigned int i = 0; i < usac_min_sample_size_; ++i)
    {
      for (unsigned int j = 0; j < 2; ++j)
        {
          src_ptr = data_matrix_ + 2*min_sample_[i] + j;
          for (unsigned int k = 0; k < 9; ++k)
            {
              *dst_ptr = *src_ptr;
              ++dst_ptr;
              src_ptr += 2*usac_num_data_points_;
            }
        }
    }

  MathTools::mattr(At, A, 8, 9);

  double D[9], U[9*9], V[8*8], *p;
  MathTools::svduv(D, At, U, 9, V, 8);
  p = U + 8;

  double T2_H[9];
  for (unsigned int i = 0; i < 9; ++i)
    {
      *(models_[0]+i) = *p;
      p += 9;
    }
  MathTools::mmul(T2_H, m_T2inv_, models_[0], 3);
  MathTools::mmul(models_denorm_[0], T2_H, m_T1_, 3);

  return 1;
}


// ============================================================================================
// generateRefinedModel: compute model using non-minimal set of samples
// default operation is to use a weight of 1 for every data point
// ============================================================================================
bool HomogEstimator::generateRefinedModel(std::vector<unsigned int>& sample,
					  unsigned int numPoints,
					  bool weighted,
					  double* weights)
{
  // form the matrix of equations for this non-minimal sample
  double *A = new double[numPoints*2*9];
  double *src_ptr;
  double *dst_ptr = A;
  for (unsigned int i = 0; i < numPoints; ++i)
    {
      for (unsigned int j = 0; j < 2; ++j)
        {
          src_ptr = data_matrix_ + 2*sample[i] + j;
          for (unsigned int k = 0; k < 9; ++k)
            {
              if (!weighted)
                {
                  *dst_ptr = *src_ptr;
                }
              else
                {
                  *dst_ptr = (*src_ptr)*weights[i];
                }
              ++dst_ptr;
              src_ptr += 2*usac_num_data_points_;
            }
        }
    }

  // decompose
  double V[9*9], D[9], *p;
  MathTools::svdu1v(D, A, 2*numPoints, V, 9);

  unsigned int j = 0;
  for (unsigned int i = 1; i < 9; ++i)
    {
      if (D[i] < D[j])
        j = i;
    }
  p = V + j;

  for (unsigned int i = 0; i < 9; ++i)
    {
      *(models_[0]+i) = *p;
      p += 9;
    }
  double T2_H[9];
  MathTools::mmul(T2_H, m_T2inv_, models_[0], 3);
  MathTools::mmul(models_denorm_[0], T2_H, m_T1_, 3);

  delete[] A;

  return true;
}


// ============================================================================================
// validateSample: check if minimal sample is valid
// ============================================================================================
bool HomogEstimator::validateSample()
{
  // check oriented constraints
  double p[3], q[3];
  double *a, *b, *c, *d;

  a = input_points_ + 6*min_sample_[0];
  b = input_points_ + 6*min_sample_[1];
  c = input_points_ + 6*min_sample_[2];
  d = input_points_ + 6*min_sample_[3];

  HTools::crossprod(p, a, b, 1);
  HTools::crossprod(q, a+3, b+3, 1);

  if ((p[0]*c[0]+p[1]*c[1]+p[2]*c[2])*(q[0]*c[3]+q[1]*c[4]+q[2]*c[5])<0)
    return false;
  if ((p[0]*d[0]+p[1]*d[1]+p[2]*d[2])*(q[0]*d[3]+q[1]*d[4]+q[2]*d[5])<0)
    return false;

  HTools::crossprod(p, c, d, 1);
  HTools::crossprod(q, c+3, d+3, 1);

  if ((p[0]*a[0]+p[1]*a[1]+p[2]*a[2])*(q[0]*a[3]+q[1]*a[4]+q[2]*a[5])<0)
    return false;
  if ((p[0]*b[0]+p[1]*b[1]+p[2]*b[2])*(q[0]*b[3]+q[1]*b[4]+q[2]*b[5])<0)
    return false;

  return true;
}


// ============================================================================================
// validateModel: check if model computed from minimal sample is valid
// ============================================================================================
bool HomogEstimator::validateModel(const unsigned int modelIndex)
{
  bool isValid = true;
  double* curr_model = models_denorm_[modelIndex];
  const double det = determinant3x3(curr_model);
  double tolerance = curr_model[8];
  if (tolerance == 0) {
      for (unsigned int i = 0; i < 9; ++i) {
          tolerance += curr_model[i] * curr_model[i];
        }
      tolerance = sqrt(tolerance);
      tolerance *= ratio_degen_test;
    }
  if (fabs(det/(tolerance * tolerance * tolerance)) < 10e-2)
    isValid = false;

  return isValid;
}


// ============================================================================================
// evaluateModel: test model against all/subset of the data points
// ============================================================================================
bool HomogEstimator::evaluateModel(unsigned int modelIndex, unsigned int* numInliers, unsigned int* numPointsTested)
{
  double* model = models_denorm_[modelIndex];
  double inv_model[9];
  double temp_err;
  double* pt;
  std::vector<double>::iterator current_err_array = err_ptr_[0];
  bool good_flag = true;
  double lambdaj, lambdaj_1 = 1.0;
  *numInliers = 0;
  *numPointsTested = 0;
  unsigned int pt_index;

  for (unsigned int i = 0; i < 9; ++i)
    {
      inv_model[i] = model[i];
    }
  MathTools::minv(inv_model, 3);
  for (unsigned int i = 0; i < usac_num_data_points_; ++i)
    {
      // get index of point to be verified
      if (eval_pool_index_ > usac_num_data_points_-1)
        {
          eval_pool_index_ = 0;
        }
      pt_index = evaluation_pool_[eval_pool_index_];
      ++eval_pool_index_;
      pt = input_points_denorm_ + 6*pt_index;
      double m1[3];
      m1[0] = pt[0];
      m1[1] = pt[1];
      m1[2] = pt[2];

      double m2[3];
      m2[0] = pt[3];
      m2[1] = pt[4];
      m2[2] = pt[5];

      if (usac_error_function_ == USACConfig::ERR_SYMMETRIC_TRANSFER_SUM) {
          // compute symmetric transfer error

          temp_err = calcSymmetricTransferErrorSum(m1,m2,model,inv_model);

        } else if (usac_error_function_ == USACConfig::ERR_SAMPSON) {
          // compute Sampson transfer error

          temp_err = calc2DHomogSampsonErr(m1,m2,model);

        } else if (usac_error_function_ == USACConfig::ERR_SYMMETRIC_TRANSFER_MAX) {
          // compute symmetric transfer error max

          temp_err = calcSymmetricTransferErrorSum(m1,m2,model,inv_model);

        }
      *(current_err_array+pt_index) = temp_err;

      if (temp_err < usac_inlier_threshold_)
        {
          ++(*numInliers);
        }

      if (usac_verif_method_ == USACConfig::VERIF_SPRT)
        {
          if (temp_err < usac_inlier_threshold_)
            {
              lambdaj = lambdaj_1 * (sprt_delta_/sprt_epsilon_);
            }
          else
            {
              lambdaj = lambdaj_1 * ( (1 - sprt_delta_)/(1 - sprt_epsilon_) );
            }

          if (lambdaj > decision_threshold_sprt_)
            {
              good_flag = false;
              *numPointsTested = i+1;
              return good_flag;
            }
          else
            {
              lambdaj_1 = lambdaj;
            }
        }
    }
  *numPointsTested = usac_num_data_points_;

  return good_flag;
}

// ============================================================================================
// testSolutionDegeneracy: check if model is degenerate
// ============================================================================================
void HomogEstimator::testSolutionDegeneracy(bool* degenerateModel, bool* upgradeModel)
{
  *degenerateModel = !validateModel(0);
  *upgradeModel = false;
}

// ============================================================================================
// upgradeDegenerateModel: try to upgrade degenerate model to non-degenerate by sampling from
// the set of outliers to the degenerate model
// ============================================================================================
unsigned int HomogEstimator::upgradeDegenerateModel()
{
  return 0;
}


// ============================================================================================
// findWeights: given model and points, compute weights to be used in local optimization
// ============================================================================================
void HomogEstimator::findWeights(unsigned int modelIndex, const std::vector<unsigned int>& inliers, 
                                 unsigned int numInliers, double* weights)
{
  for (unsigned int i = 0; i < numInliers; ++i)
    {
      weights[i] = 1.0;
    }
}


// ============================================================================================
// storeModel: stores current best model
// this function is called  (by USAC) every time a new best model is found
// ============================================================================================
void HomogEstimator::storeModel(const unsigned int modelIndex, unsigned int numInliers)
{
  // save the current model as the best solution so far
  for (unsigned int i = 0; i < 9; ++i)
    {
      final_model_params_[i] = *(models_denorm_[modelIndex]+i);
    }
}

#endif

