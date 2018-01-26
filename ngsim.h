#include <fftw3.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <kms_fftw.hpp>

#ifndef _NGSIM_H
#define _NGSIM_H


// this data structure collects the input parameters for the boundary value problem
struct boundary_value_data {
  int N;
  double pixsize;
  int T;
  double tauI, tauF;
  double *tau;
  double dlntau;
  double Lambda;
  double boxsize;
  double *kmax;
  double f; double f2;
  double c; double c2;
  int shells;

  double *pvariance;
  double *pmean;

  double convergence_threshold;

  int loop_max;
  int loop_diff;
  double bool_pvar;
  double bool_ensemble;
  int flag;
  int toggle; // an extra parameter, if needed
  // 0 = frozen source arrays, 
  // 1 = frozen source arrays, memory optimized
  // 2 = up-to-date source arrays
  // 3 = up-to-date source arrays, memory optimized

  virtual boost::shared_ptr<kms::fftw_fourier_grid3d_base> allocate_fourier_grid3d();
  virtual boost::shared_ptr<kms::fftw_grid3d_base> allocate_grid3d();

  boost::shared_ptr<kms::fftw_fourier_grid3d_base> clone(const kms::fftw_fourier_grid3d_base &x);
};

// collects the output values for the algorithm; called from ABsolution()
struct relax_output{
  boost::shared_ptr<kms::fftw_fourier_grid3d_base> forceint;
  boost::shared_ptr<kms::fftw_fourier_grid3d_base> forcefree;
  double actionint, actionfree, actionint_error;
};

extern void allocate_arrays(boundary_value_data &d);
extern void compute_taukmax(boundary_value_data &d);

// the next several functions are defined in fourier_field_products.cpp

extern boost::shared_ptr<kms::fftw_fourier_grid3d_base> Square(boundary_value_data &d, double kmax, kms::fftw_fourier_grid3d_base &P);
extern boost::shared_ptr<kms::fftw_fourier_grid3d_base> cube(boundary_value_data &d, double kmax, kms::fftw_fourier_grid3d_base &P);
extern boost::shared_ptr<kms::fftw_fourier_grid3d_base> product(boundary_value_data &d, double kmax, kms::fftw_fourier_grid3d_base &p1, kms::fftw_fourier_grid3d_base &p2);

extern boost::shared_ptr<kms::fftw_fourier_grid3d_base> threeproduct(boundary_value_data &d, double kmax, kms::fftw_fourier_grid3d_base &p1,kms::fftw_fourier_grid3d_base &p2, kms::fftw_fourier_grid3d_base &p3);

extern boost::shared_ptr<kms::fftw_fourier_grid3d_base> gradprod(boundary_value_data &d, double kmax, kms::fftw_fourier_grid3d_base &p1, kms::fftw_fourier_grid3d_base &p2);

extern void gradsquare(boundary_value_data &d, double kmax, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &input, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &gradsquareinput);

extern void pgradsquare(boundary_value_data &d, double kmax, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &input, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &pgradsquareinput);

extern void fill_sigma_p(boundary_value_data &d, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A3d, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B3d, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &sigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p);

extern void L_derivs(boundary_value_data &d, double kmax, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &sigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdsigma);

extern void L_deriv_f1(boundary_value_data &d, double kmax, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp);

extern void fill_source(boundary_value_data &d, double tau, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdsigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_a, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_b);

extern void fill_source_aorb(boundary_value_data &d, bool AB_switch, double tau, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdsigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s);

extern void source(boundary_value_data &d, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_a, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_b);

extern void source_aorb(boundary_value_data &d, int t, bool AB_switch, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s);

extern void source_mem(boundary_value_data &d, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B);

extern void source_mem_aorb(boundary_value_data &d, int t, bool AB_switch, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B);

extern void Bsweep(boundary_value_data &d, 
		   std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B, 
		   std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &s);

extern void Asweep(boundary_value_data &d, 
		   std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A, 
		   std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &s);

extern void B_sweep_mem(boundary_value_data &d, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B);

extern void A_sweep_mem(boundary_value_data &d, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A);

extern void B_sweep_adv(boundary_value_data &d, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B);

extern void A_sweep_adv(boundary_value_data &d, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B);

extern void B_sweep_step(boundary_value_data &d, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &AB);

extern void A_sweep_step(boundary_value_data &d, std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &AB);

extern void Afinal(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f,
		   boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A_f, 
		   boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B_f);

extern relax_output ABsolution(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f);

extern double action(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f, 
		     std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A,
		     std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B);

extern double action_boundary(boundary_value_data &d, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A_f, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B_f);

extern double freeaction(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f,
			 std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A,
			 std::vector< boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B,
			 kms::fftw_fourier_grid3d_base &p, kms::fftw_fourier_grid3d_base &pp,
			 kms::fftw_fourier_grid3d_base &s, double actionint);

extern boost::shared_ptr<kms::fftw_fourier_grid3d_base> force_free(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f);

extern boost::shared_ptr<kms::fftw_fourier_grid3d_base> force_int(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A_f, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B_f);

extern double forcetest(boundary_value_data &d, double eps, kms::fftw_fourier_grid3d_base &sigma_f, kms::fftw_fourier_grid3d_base &sigma_kick);

extern void *bispectrum(boundary_value_data &d, fftw_complex *sigma_f, fftw_complex *A, fftw_complex *B);

extern double action_int_check(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f);

extern double p_var(boundary_value_data &d, double tau);

extern double p_var_realization(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f, double tau);

extern double p_mean_realization(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f, double tau);

extern void p_variance(boundary_value_data &d, std::vector<boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A, std::vector<boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B);

#endif  // _NGSIM_H

