#include "ngsim.h"
#include <kms_fftw.hpp>

using namespace std;
using namespace kms;
using namespace boost;

// these functions compute the nonlinear source terms in the equation of motion, and are called from ABsolutio()
// there are several different source functions, which are used in slight variations of the iterative algorithm in ABsolution(); these different options correspond to different values of d.flag as used in ABsolution()

void source(boundary_value_data &d, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_a, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_b)
{
  double kmax = d.kmax[t];
  double tau = d.tau[t];

  boost::shared_ptr<fftw_fourier_grid3d_base> sigma = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  fill_sigma_p(d, t, A, B, sigma, p);

  boost::shared_ptr<fftw_fourier_grid3d_base> dLdp = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> dLdsigma = d.allocate_fourier_grid3d();
  if ((d.f2==0.) & (d.c2==0.)) {L_deriv_f1(d, kmax, t, p, dLdp);} else {L_derivs(d, kmax, t, sigma, p, dLdp, dLdsigma);}

  fill_source(d, tau, dLdp, dLdsigma, s_a, s_b);
}

void source_aorb(boundary_value_data &d, int t, bool AB_switch, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s)
{
  double kmax = d.kmax[t];
  double tau = d.tau[t];

  boost::shared_ptr<fftw_fourier_grid3d_base> sigma = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  fill_sigma_p(d, t, A, B, sigma, p);

  boost::shared_ptr<fftw_fourier_grid3d_base> dLdp = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> dLdsigma = d.allocate_fourier_grid3d();
  if ((d.f2==0.) & (d.c2==0.)) {L_deriv_f1(d, kmax, t, p, dLdp);} else {L_derivs(d, kmax, t, sigma, p, dLdp, dLdsigma);}

  fill_source_aorb(d, AB_switch, tau, dLdp, dLdsigma, s);

}

void source_mem(boundary_value_data &d, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B)
{
  double kmax = d.kmax[t];
  double tau = d.tau[t];

  boost::shared_ptr<fftw_fourier_grid3d_base> sigma = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  fill_sigma_p(d, t, A, B, sigma, p);

  boost::shared_ptr<fftw_fourier_grid3d_base> dLdp = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> dLdsigma = d.allocate_fourier_grid3d();
  if ((d.f2==0.) & (d.c2==0.)) {L_deriv_f1(d, kmax, t, p, dLdp);} else {L_derivs(d, kmax, t, sigma, p, dLdp, dLdsigma);}

  fill_source(d, tau, dLdp, dLdsigma, A, B); // sources replace A and B in "A" and "B" arrays
}

void source_mem_aorb(boundary_value_data &d, int t, bool AB_switch, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B)
{
  double kmax = d.kmax[t];
  double tau = d.tau[t];

  boost::shared_ptr<fftw_fourier_grid3d_base> sigma = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  fill_sigma_p(d, t, A, B, sigma, p);

  boost::shared_ptr<fftw_fourier_grid3d_base> dLdp = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> dLdsigma = d.allocate_fourier_grid3d();
  if ((d.f2==0.) & (d.c2==0.)) {L_deriv_f1(d, kmax, t, p, dLdp);} else {L_derivs(d, kmax, t, sigma, p, dLdp, dLdsigma);}
OA
  boost::shared_ptr<fftw_fourier_grid3d_base> a_or_b = d.allocate_fourier_grid3d();

  if (AB_switch == true) {*a_or_b = *B;} else {*a_or_b = *A;}
  fill_source_aorb(d, AB_switch, tau, dLdp, dLdsigma, a_or_b);
}

void fill_source(boundary_value_data &d, double tau, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdsigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_a, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s_b)
{
  int i, j, k;
  int N = d.N;
  int nx_local = (*s_a).local_nx;

    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      double kk = (*s_a).k(i,j,k); 
	  double ekt = exp(kk*tau);
	  if (kk<-d.Lambda/tau) {
	  (*s_a)(i,j,k)  = 1/(ekt*2*kk)/tau*(*dLdp)(i,j,k);
          (*s_a)(i,j,k) += 1/(ekt*2*kk*kk*kk)*(1+kk*tau)*(*dLdsigma)(i,j,k);
	  (*s_b)(i,j,k)  = ekt/(2*kk)/tau*(*dLdp)(i,j,k);
          (*s_b)(i,j,k) += ekt/(2*kk*kk*kk)*(1-kk*tau)*(*dLdsigma)(i,j,k); }
    }}}
    (*s_a).set_dc_mode(0.0);
    (*s_b).set_dc_mode(0.0);

}

void fill_source_aorb(boundary_value_data &d, bool AB_switch, double tau, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdsigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &s)
{
  int i, j, k;
  int N = d.N;
  int nx_local = (*s).local_nx;

  if (AB_switch == false) {
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      double kk = (*s).k(i,j,k); 
      double ekt = exp(kk*tau);
          if (kk<-d.Lambda/tau) {
	  (*s)(i,j,k)  = 1/(ekt*2*kk)/tau*(*dLdp)(i,j,k);
          (*s)(i,j,k) += 1/(ekt*2*kk*kk*kk)*(1+kk*tau)*(*dLdsigma)(i,j,k); }
	  else {(*s)(i,j,k) = 0.;}
    }}} }
  else {
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      double kk = (*s).k(i,j,k); 
      double ekt = exp(kk*tau);
	  if (kk<-d.Lambda/tau) {
	  (*s)(i,j,k)  = ekt/(2*kk)/tau*(*dLdp)(i,j,k);
          (*s)(i,j,k) += ekt/(2*kk*kk*kk)*(1-kk*tau)*(*dLdsigma)(i,j,k); }
	  else {(*s)(i,j,k) = 0.;}
    }}} }
  (*s).set_dc_mode(0.0);

}

void L_derivs(boundary_value_data &d, double kmax, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &sigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdsigma)
{
  int i, j, k;
  int N = d.N;
  int nx_local = (*p).local_nx;

  double tau = d.tau[t];

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if ((*p).k(i,j,k) > kmax) {(*p)(i,j,k) = 0.;}  }}}
  boost::shared_ptr<fftw_grid3d_base> p_real = d.allocate_grid3d();
  p_real->fill_with_normalized_fft(*p);

  boost::shared_ptr<fftw_fourier_grid3d_base> sigmax = d.clone(*sigma);
  boost::shared_ptr<fftw_fourier_grid3d_base> sigmay = d.clone(*sigma);
  boost::shared_ptr<fftw_fourier_grid3d_base> sigmaz = d.clone(*sigma);
  (*sigmax).apply_x_derivative();
  (*sigmay).apply_y_derivative();
  (*sigmaz).apply_z_derivative();

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if ((*p).k(i,j,k) > kmax) {
      (*sigmax)(i,j,k) = (*sigmay)(i,j,k) = (*sigmaz)(i,j,k) = 0.0; }  }}}

  boost::shared_ptr<fftw_grid3d_base> sigmax_real = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> sigmay_real = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> sigmaz_real = d.allocate_grid3d();
  sigmax_real->fill_with_normalized_fft(*sigmax);
  sigmay_real->fill_with_normalized_fft(*sigmay);
  sigmaz_real->fill_with_normalized_fft(*sigmaz);

  boost::shared_ptr<fftw_grid3d_base> gradsigmasquare = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> pgradsigmasquare = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> pgradsigmax = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> pgradsigmay = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> pgradsigmaz = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> ppgradsigmax = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> ppgradsigmay = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> ppgradsigmaz = d.allocate_grid3d();

  boost::shared_ptr<fftw_grid3d_base> pp = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> ppp = d.allocate_grid3d();

  *pp += *p_real; *pp *= *p_real;
  *ppp += *pp; *ppp *= *p_real;
  *pgradsigmax += *p_real; *pgradsigmax *= *sigmax_real;
  *pgradsigmay += *p_real; *pgradsigmay *= *sigmay_real;
  *pgradsigmaz += *p_real; *pgradsigmaz *= *sigmaz_real;
  *ppgradsigmax += *pp;
  *ppgradsigmay += *pp;
  *ppgradsigmaz += *pp;

  if (d.bool_pvar == true) {
    double p_variance;
    if (d.bool_ensemble == false) {p_variance = d.pvariance[t]; }
    if (d.bool_ensemble == true) {p_variance = p_var(d, tau); }
  for (i = 0; i < nx_local; i++) {
  for (j = 0; j < N; j++) {
  for (k = 0; k < N; k++) {
    (*ppgradsigmax)(i,j,k) -= p_variance;
    (*ppgradsigmay)(i,j,k) -= p_variance;
    (*ppgradsigmaz)(i,j,k) -= p_variance; }}}
  }

  *ppgradsigmax *= *sigmax_real;
  *ppgradsigmay *= *sigmay_real;
  *ppgradsigmaz *= *sigmaz_real;

  *sigmax_real *= *sigmax_real;   *sigmay_real *= *sigmay_real;   *sigmaz_real *= *sigmaz_real;
  *gradsigmasquare += *sigmax_real;
  *gradsigmasquare += *sigmay_real;
  *gradsigmasquare += *sigmaz_real;

  *pgradsigmasquare += *p_real;
  if (d.bool_pvar == true){
    if (d.bool_ensemble == false) {
      boost::shared_ptr<fftw_grid3d_base> p_mean = d.allocate_grid3d();
      for (i=0; i<nx_local; i++) {
      for (j=0; j<N; j++) {
      for (k=0; k<N/2+1; k++) {
	(*p_mean)(i,j,k) = d.pmean[t]; }}}
        *pgradsigmasquare -= *p_mean;
    }}
  *pgradsigmasquare *= *gradsigmasquare;

  double c2_prefactor = d.c2*tau*tau*tau*tau*tau;

  (*pp) *= -3*d.f*tau*tau*tau*tau*tau*tau;
  (*ppp) *= 2*d.c*tau*tau*tau*tau*tau*tau*tau*tau*tau;;
  (*gradsigmasquare) *= d.f2*tau*tau;
  (*pgradsigmasquare) *= -c2_prefactor;
  *pp += *ppp; *pp += *gradsigmasquare; *pp += *pgradsigmasquare;
  dLdp->fill_with_normalized_fft(*pp);

  (*pgradsigmax) *= -2*d.f2*tau*tau;   (*pgradsigmay) *= -2*d.f2*tau*tau;   (*pgradsigmaz) *= -2*d.f2*tau*tau;
  (*ppgradsigmax) *= c2_prefactor;   (*ppgradsigmay) *= c2_prefactor;   (*ppgradsigmaz) *= c2_prefactor;
  *pgradsigmax += *ppgradsigmax;   *pgradsigmay += *ppgradsigmay;   *pgradsigmaz += *ppgradsigmaz;
  boost::shared_ptr<fftw_fourier_grid3d_base> x_dLdsigma = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> y_dLdsigma = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> z_dLdsigma = d.allocate_fourier_grid3d();
  x_dLdsigma->fill_with_normalized_fft(*pgradsigmax);
  y_dLdsigma->fill_with_normalized_fft(*pgradsigmay);
  z_dLdsigma->fill_with_normalized_fft(*pgradsigmaz);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if ((*p).k(i,j,k) > kmax) {
      (*dLdp)(i,j,k) = (*x_dLdsigma)(i,j,k) = (*y_dLdsigma)(i,j,k) = (*z_dLdsigma)(i,j,k) = 0.0; } }}}

  (*x_dLdsigma).apply_x_derivative();
  (*y_dLdsigma).apply_y_derivative();
  (*z_dLdsigma).apply_z_derivative();
  *dLdsigma += *x_dLdsigma;
  *dLdsigma += *y_dLdsigma;
  *dLdsigma += *z_dLdsigma;
}

void L_deriv_f1(boundary_value_data &d, double kmax, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &dLdp)
{
  int i, j, k;
  int N = d.N;
  int nx_local = (*p).local_nx;

  double tau = d.tau[t];

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if ((*p).k(i,j,k) > kmax) {(*p)(i,j,k) = 0.;}  }}}

  boost::shared_ptr<fftw_grid3d_base> p_real = d.allocate_grid3d();
  p_real->fill_with_normalized_fft(*p);

  boost::shared_ptr<fftw_grid3d_base> pp = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> ppp = d.allocate_grid3d();

  *pp += *p_real; *pp *= *p_real;
  *ppp += *pp; *ppp *= *p_real;

  (*pp) *= -3*d.f*tau*tau*tau*tau*tau*tau;
  (*ppp) *= 2*d.c*tau*tau*tau*tau*tau*tau*tau*tau*tau;;
  *pp += *ppp;

  dLdp->fill_with_normalized_fft(*pp);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if ((*p).k(i,j,k) > kmax) {(*dLdp)(i,j,k) = 0;}  }}}
}
