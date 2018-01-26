#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <kms_fftw.hpp>
#include <vector>
#include <iomanip>

#include "ngsim.h"

using namespace std;
using namespace kms;
using namespace boost;

// this file contains additional variations of the functions Asweep() and Bsweep(), which are called from ABsolution() for different variations of the iterative convergence algorithm, as indexed by d.flag there

void Afinal(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f, boost::shared_ptr<fftw_fourier_grid3d_base> &A_f, boost::shared_ptr<fftw_fourier_grid3d_base> &B_f)
{
  int i, j, k;
  int N=d.N; int nx_local = sigma_f.local_nx;

  double tau = d.tau[d.T-1]; 
  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    double kk = sigma_f.k(i,j,k); double ekt = exp(kk*tau);
    (*A_f)(i,j,k) = (sigma_f(i,j,k) - (1+kk*tau)/ekt*(*B_f)(i,j,k))/(ekt*(1-kk*tau)); }}}

  A_f->set_dc_mode(0.0);
}

void Bsweep(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &B, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &s)
{
  int i, j, k, t;
  int N=d.N; int T=d.T;
  int nx_local = (*s[0]).local_nx;

  for (t = 1; t<T; t++) {
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
	(*B[t])(i,j,k) = (*B[t-1])(i,j,k) + d.dlntau*(*s[t-1])(i,j,k); 
    }}}
  }

  for (t = 0; t<T; t++) { B[t]->set_dc_mode(0.0); }
}

void Asweep(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &A, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &s)
{
  int i, j, k, t;
  int N=d.N; int T=d.T;
  int nx_local = (*s[0]).local_nx;
  
  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    (*A[T-2])(i,j,k) = (*A[T-1])(i,j,k); }}}

  for (t = T-3; t>=0; t--) {
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      (*A[t])(i,j,k) = (*A[t+1])(i,j,k) + d.dlntau * (*s[t+1])(i,j,k); 
    }}}
  }
  for (t = 0; t<T; t++) { A[t]->set_dc_mode(0.0); }
}

void B_sweep_mem(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &B)
{
  int i, j, k, t;
  int N = d.N;  int nx_local = (*B[0]).local_nx;
  int T = d.T;

  boost::shared_ptr<fftw_fourier_grid3d_base> extra1 = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> extra2 = d.allocate_fourier_grid3d();

  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    (*extra1)(i,j,k) = (*B[0])(i,j,k); // stores s_b[0] in extra1
    (*B[0])(i,j,k) = 0.; // sets B[0] to correct initial condition for B sweep
  }}}

  for (t = 1; t<T; t++) {
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      (*extra2)(i,j,k) = (*B[t])(i,j,k); // stores s_b[t] for next timestep
      (*B[t])(i,j,k) = (*B[t-1])(i,j,k) + d.dlntau*(*extra1)(i,j,k);
      (*extra1)(i,j,k) = (*extra2)(i,j,k); // shifts s_b[t] to "on deck" slot after s_b[t-1] has been used to update B[t]
    }}}
  (*B[t-1]).set_dc_mode(0.0); }

  (*B[T-1]).set_dc_mode(0.0);
}

void A_sweep_mem(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &A)
{
  int i, j, k, t;
  int N = d.N;  int nx_local = (*A[0]).local_nx;
  int T = d.T;

  boost::shared_ptr<fftw_fourier_grid3d_base> extra1 = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> extra2 = d.allocate_fourier_grid3d();

  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    (*extra1)(i,j,k) = (*A[T-2])(i,j,k);  // stores s_a[T-2] in extra1 (instead of s_a[T-1] we already have A_final)
    (*A[T-2])(i,j,k) = (*A[T-1])(i,j,k);  // sets A[T-2] to correct initial condition for A sweep
}}}

  for (t = T-3; t>=0; t--) {
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      (*extra2)(i,j,k) = (*A[t])(i,j,k); // stores s_a[t] for next timestep
      (*A[t])(i,j,k) = (*A[t+1])(i,j,k) + d.dlntau*(*extra1)(i,j,k);
      (*extra1)(i,j,k) = (*extra2)(i,j,k); // shifts s_a[t] to "on deck" slot after s_a[t+1] has been used to update B[t]
    }}}
    (*A[t+1]).set_dc_mode(0.0); }

  (*A[0]).set_dc_mode(0.0);
}

void B_sweep_adv(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &A, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &B)
{
  int i, j, k, t;
  int N = d.N;  int nx_local = (*B[0]).local_nx;
  int T = d.T;

  boost::shared_ptr<fftw_fourier_grid3d_base> s = d.allocate_fourier_grid3d();

   for (t = 1; t<T; t++) {
     source_aorb(d, t-1, true, A[t-1], B[t-1], s);
   for (i = 0; i<nx_local; i++) {
   for (j = 0; j<N; j++) {
   for (k = 0; k<N/2+1; k++) {
     (*B[t])(i,j,k) = (*B[t-1])(i,j,k) + d.dlntau*(*s)(i,j,k);
   }}}

    (*B[t-1]).set_dc_mode(0.0); }

  (*B[T-1]).set_dc_mode(0.0);
}

void A_sweep_adv(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &A, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &B)
{
  int i, j, k, t;
  int N = d.N;  int nx_local = (*A[0]).local_nx;
  int T = d.T;

  boost::shared_ptr<fftw_fourier_grid3d_base> s = d.allocate_fourier_grid3d();

  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    (*A[T-2])(i,j,k) = (*A[T-1])(i,j,k);  }}}

  for (t = T-3; t>=0; t--) {
    source_aorb(d, t+1, false, A[t+1], B[t+1], s);
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      (*A[t])(i,j,k) = (*A[t+1])(i,j,k) + d.dlntau*(*s)(i,j,k);
    }}}
    (*A[t+1]).set_dc_mode(0.0); }

  (*A[0]).set_dc_mode(0.0);
}

void B_sweep_step(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &AB)
{
  int i, j, k, t;
  int N = d.N;  int nx_local = (*AB[0]).local_nx;
  int T = d.T;

  boost::shared_ptr<fftw_fourier_grid3d_base> s_b = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> extra_A = d.allocate_fourier_grid3d();

  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    (*extra_A)(i,j,k) = (*AB[0])(i,j,k);
    (*AB[0])(i,j,k) = 0.; }}}

   for (t = 1; t<T; t++) {
     source_aorb(d, t-1, true, extra_A, AB[t-1], s_b);
     for (i = 0; i<nx_local; i++) {
     for (j = 0; j<N; j++) {
     for (k = 0; k<N/2+1; k++) {
      (*extra_A)(i,j,k) = (*AB[t])(i,j,k);
      (*AB[t])(i,j,k) = (*AB[t-1])(i,j,k) + d.dlntau*(*s_b)(i,j,k);  }}}
     (*AB[t-1]).set_dc_mode(0.0);}

  (*AB[T-1]).set_dc_mode(0.0);
}

void A_sweep_step(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &AB)
{
  int i, j, k, t;
  int N = d.N;  int nx_local = (*AB[0]).local_nx;
  int T = d.T;

  boost::shared_ptr<fftw_fourier_grid3d_base> s_a = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> extra_B = d.allocate_fourier_grid3d();

  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    (*extra_B)(i,j,k) = (*AB[T-2])(i,j,k);
    (*AB[T-2])(i,j,k) = (*AB[T-1])(i,j,k);  }}}

  for (t = T-3; t>=0; t--) {
    source_aorb(d, t+1, false, AB[t+1], extra_B, s_a);
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      (*extra_B)(i,j,k) = (*AB[t])(i,j,k);
      (*AB[t])(i,j,k) = (*AB[t+1])(i,j,k) + d.dlntau*(*s_a)(i,j,k);  }}}
    (*AB[t+1]).set_dc_mode(0.0); }

  (*AB[0]).set_dc_mode(0.0);
}
