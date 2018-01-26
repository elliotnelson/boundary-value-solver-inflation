#include "ngsim.h"
#include <kms_fftw.hpp>

using namespace std;
using namespace kms;
using namespace boost;

// these functions take as input two or more fourier-space fields, and compute the fourier-transform of the real-space product (at each point in space) of those input fields; some functions apply various gradients as well; these functions are needed to compute the particular action for the underlying theory that we are interested in

boost::shared_ptr<fftw_fourier_grid3d_base> product(boundary_value_data &d, double kmax, fftw_fourier_grid3d_base &p1, fftw_fourier_grid3d_base &p2)
{
  int i, j, k;
  int N = d.N; int nx_local = p1.local_nx;

  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes1 = d.allocate_fourier_grid3d();
  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*Kmodes1)(i,j,k) = p1(i,j,k);
  }}}

  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes2 = d.allocate_fourier_grid3d();
  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*Kmodes2)(i,j,k) = p2(i,j,k); 
  }}}

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if( p1.k(i,j,k) > kmax) {
	(*Kmodes1)(i,j,k) = complex<double>(0,0);
	(*Kmodes2)(i,j,k) = complex<double>(0,0);
    }
  }}}

  boost::shared_ptr<fftw_grid3d_base> Xpoints = d.allocate_grid3d();
  Xpoints->fill_with_normalized_fft(*Kmodes1);
  boost::shared_ptr<fftw_grid3d_base> Xpoints2 = d.allocate_grid3d();
  Xpoints2->fill_with_normalized_fft(*Kmodes2);
  (*Xpoints) *= (*Xpoints2);

  boost::shared_ptr<fftw_fourier_grid3d_base> product = d.allocate_fourier_grid3d();
  product->fill_with_normalized_fft(*Xpoints);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (p1.k(i,j,k) > kmax)
	  (*product)(i,j,k) = 0.0;
  }}}
  
  if (abs((*product)(0,0,0)) > 1.0e100)
      throw runtime_error("blowup behavior happened!");

  return product;
}

boost::shared_ptr<fftw_fourier_grid3d_base> dotprod(boundary_value_data &d, double kmax, 
					     fftw_fourier_grid3d_base &v1, fftw_fourier_grid3d_base &v2, fftw_fourier_grid3d_base &v3, 
					     fftw_fourier_grid3d_base &w1, fftw_fourier_grid3d_base &w2, fftw_fourier_grid3d_base &w3)
{
  int i, j, k;
  int N = d.N; int nx_local = v1.local_nx;

  boost::shared_ptr<fftw_fourier_grid3d_base> vx = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> vy = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> vz = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> wx = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> wy = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> wz = d.allocate_fourier_grid3d();

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*vx)(i,j,k) = v1(i,j,k); 
      (*vy)(i,j,k) = v2(i,j,k); 
      (*vz)(i,j,k) = v3(i,j,k);
      (*wx)(i,j,k) = w1(i,j,k); 
      (*wy)(i,j,k) = w2(i,j,k);
      (*wz)(i,j,k) = w3(i,j,k); 
  }}}

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (v1.k(i,j,k) > kmax) {
	  (*vx)(i,j,k) = (*vy)(i,j,k) = (*vz)(i,j,k) = 0.0;
	  (*wx)(i,j,k) = (*wy)(i,j,k) = (*wz)(i,j,k) = 0.0; 
      }
  }}}

  boost::shared_ptr<fftw_grid3d_base> realvx = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realvy = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realvz = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realwx = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realwy = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realwz = d.allocate_grid3d();

  realvx->fill_with_normalized_fft(*vx);
  realvy->fill_with_normalized_fft(*vy);
  realvz->fill_with_normalized_fft(*vz);
  realwx->fill_with_normalized_fft(*wx);
  realwy->fill_with_normalized_fft(*wy);
  realwz->fill_with_normalized_fft(*wz);

  // Overwrite vx with dot product (vx*wx + vy*wy + vz*wz)
  (*realvx) *= (*realwx);
  (*realvy) *= (*realwy);
  (*realvz) *= (*realwz);
  (*realvx) += (*realvy);
  (*realvx) += (*realvz);

  boost::shared_ptr<fftw_fourier_grid3d_base> vw = d.allocate_fourier_grid3d();
  vw->fill_with_normalized_fft(*realvx);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (v1.k(i,j,k) > kmax)
	  (*vw)(i,j,k) = complex<double>(0,0);
  }}}
  
  if (abs((*vw)(0,0,0)) > 1.0e100)
    throw runtime_error("blowup behavior happened!");

  return vw;
}

boost::shared_ptr<fftw_fourier_grid3d_base> gradprod(boundary_value_data &d, double kmax, fftw_fourier_grid3d_base &p1, fftw_fourier_grid3d_base &p2)
{
  int i, j, k;
  int N = d.N; int nx_local = p1.local_nx;

  boost::shared_ptr<fftw_fourier_grid3d_base> p1p2 = d.allocate_fourier_grid3d();

  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes1 = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes2 = d.allocate_fourier_grid3d();

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*Kmodes1)(i,j,k) = p1(i,j,k);
      (*Kmodes2)(i,j,k) = p2(i,j,k); }}}

  (*Kmodes1).apply_x_derivative();
  (*Kmodes2).apply_x_derivative();
  Kmodes1 = product(d, kmax, *Kmodes1, *Kmodes2);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (p1.k(i,j,k) < kmax)
 	(*p1p2)(i,j,k) += (*Kmodes1)(i,j,k); }}}

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*Kmodes1)(i,j,k) = p1(i,j,k);
      (*Kmodes2)(i,j,k) = p2(i,j,k); }}}

  (*Kmodes1).apply_y_derivative();
  (*Kmodes2).apply_y_derivative();
  Kmodes1 = product(d, kmax, *Kmodes1, *Kmodes2);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (p1.k(i,j,k) < kmax)
	(*p1p2)(i,j,k) += (*Kmodes1)(i,j,k); }}}

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*Kmodes1)(i,j,k) = p1(i,j,k);
      (*Kmodes2)(i,j,k) = p2(i,j,k); }}}

  (*Kmodes1).apply_z_derivative();
  (*Kmodes2).apply_z_derivative();
  Kmodes1 = product(d, kmax, *Kmodes1, *Kmodes2);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (p1.k(i,j,k) < kmax)
	(*p1p2)(i,j,k) += (*Kmodes1)(i,j,k); }}}

  return p1p2;
}

boost::shared_ptr<fftw_fourier_grid3d_base> cube(boundary_value_data &d, double kmax, fftw_fourier_grid3d_base &P)
{
  int i, j, k;
  int N = d.N; int nx_local = P.local_nx;

  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes = d.allocate_fourier_grid3d();
  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
   (*Kmodes)(i,j,k) = P(i,j,k); }}}

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if( P.k(i,j,k) > kmax) { (*Kmodes)(i,j,k) = 0.0;} else {}; }}}

  boost::shared_ptr<fftw_grid3d_base> Xpoints = d.allocate_grid3d();
  Xpoints->fill_with_normalized_fft(*Kmodes);

  // real-space cube
  for (i = 0; i < Xpoints->local_nx; i++) {
      for (j = 0; j < N; j++) {
	  for (k = 0; k < N; k++) {
	      double &x = (*Xpoints)(i,j,k);
	      x = x*x*x;
	  }
      }
  }

  boost::shared_ptr<fftw_fourier_grid3d_base> ppp = d.allocate_fourier_grid3d();
  ppp->fill_with_normalized_fft(*Xpoints);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if (P.k(i,j,k) > kmax) {(*ppp)(i,j,k) = 0.0;} else{}; }}}
  
  if (abs((*ppp)(0,0,0)) > 1.0e100)
    throw runtime_error("blowup behavior happened!");

  return ppp;
}

boost::shared_ptr<fftw_fourier_grid3d_base> dotcube(boundary_value_data &d, double kmax, fftw_fourier_grid3d_base &uu, 
					     fftw_fourier_grid3d_base &v1, fftw_fourier_grid3d_base &v2, fftw_fourier_grid3d_base &v3, 
					     fftw_fourier_grid3d_base &w1, fftw_fourier_grid3d_base &w2, fftw_fourier_grid3d_base &w3)
{
  int i, j, k;
  int N = d.N; int nx_local = uu.local_nx;

  boost::shared_ptr<fftw_fourier_grid3d_base> u = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> vx = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> vy = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> vz = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> wx = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> wy = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> wz = d.allocate_fourier_grid3d();

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*u)(i,j,k) = uu(i,j,k);
      (*vx)(i,j,k) = v1(i,j,k); 
      (*vy)(i,j,k) = v2(i,j,k); 
      (*vz)(i,j,k) = v3(i,j,k);
      (*wx)(i,j,k) = w1(i,j,k); 
      (*wy)(i,j,k) = w2(i,j,k); 
      (*wz)(i,j,k) = w3(i,j,k); 
  }}}

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (uu.k(i,j,k) > kmax) {
	  (*u)(i,j,k) = 0.0;
	  (*vx)(i,j,k) = (*vy)(i,j,k) = (*vz)(i,j,k) = 0.0;
	  (*wx)(i,j,k) = (*wy)(i,j,k) = (*wz)(i,j,k) = 0.0; 
      }
  }}}

  boost::shared_ptr<fftw_grid3d_base> realu = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realvx = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realvy = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realvz = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realwx = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realwy = d.allocate_grid3d();
  boost::shared_ptr<fftw_grid3d_base> realwz = d.allocate_grid3d();

  realu->fill_with_normalized_fft(*u);
  realvx->fill_with_normalized_fft(*vx);
  realvy->fill_with_normalized_fft(*vy);
  realvz->fill_with_normalized_fft(*vz);
  realwx->fill_with_normalized_fft(*wx);
  realwy->fill_with_normalized_fft(*wy);
  realwz->fill_with_normalized_fft(*wz);

  // Overwrite u with quantity u * (vx*wx + vy*wy + vz*wz)
  (*realvx) *= (*realwx);
  (*realvy) *= (*realwy);
  (*realvz) *= (*realwz);
  (*realvx) += (*realvy);
  (*realvx) += (*realvz);
  (*realu) *= (*realvx);

  boost::shared_ptr<fftw_fourier_grid3d_base> uvw = d.allocate_fourier_grid3d();
  uvw->fill_with_normalized_fft(*realu);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (uu.k(i,j,k) > kmax)
	  (*uvw)(i,j,k) = complex<double>(0,0);
  }}}
  
  if (abs((*uvw)(0,0,0)) > 1.0e100)
    throw runtime_error("blowup behavior happened!");

  return uvw;
}

// <p^2> ensemble average at one time
double p_var(boundary_value_data &d, double tau)
{
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> p2 = d.allocate_fourier_grid3d();

  int i, j, k;
  int N = d.N; int nx_local = (*p).local_nx;

  for (i = 0; i < nx_local; i++) {
  for (j = 0; j < N; j++) {
  for (k = 0; k < N/2+1; k++) {
    double kk = (*p).k(i,j,k); 
      double ekt = exp(kk*tau);
      (*p)(i,j,k) += kk*kk/tau*ekt;
      (*p2)(i,j,k) += kk*kk/tau*ekt/(2*kk*kk*kk); }}}

  (*p).set_dc_mode(0.0);
  (*p2).set_dc_mode(0.0);

  // cout << "{" << t << "," << fftw_normalized_dot_product(*p, *p2) << "},";

  return fftw_normalized_dot_product(*p, *p2);
}

// <p^2> realization average in **free** theory, at time tau
double p_var_realization(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f, double tau)
{
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> one = d.allocate_fourier_grid3d();
  p = d.clone(sigma_f);

  int i, j, k;
  int N = d.N; int nx_local = (*p).local_nx;

  for (i = 0; i < nx_local; i++) {
  for (j = 0; j < N; j++) {
  for (k = 0; k < N/2+1; k++) {
    double kk = (*p).k(i,j,k); 
      double ekt = exp(kk*tau);
      (*one)(i,j,k) = 1.;
      (*p)(i,j,k) *= kk*kk/tau*ekt; }}}

  (*p).set_dc_mode(0.0);

  return fftw_normalized_dot_product(*p, *p)/fftw_normalized_dot_product(*one, *one);
}

// same as p_var_realization, except just p not p^2
double p_mean_realization(boundary_value_data &d, kms::fftw_fourier_grid3d_base &sigma_f, double tau)
{
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> one = d.allocate_fourier_grid3d();
  p = d.clone(sigma_f);

  int i, j, k;
  int N = d.N; int nx_local = (*p).local_nx;

  for (i = 0; i < nx_local; i++) {
  for (j = 0; j < N; j++) {
  for (k = 0; k < N/2+1; k++) {
    double kk = (*p).k(i,j,k); 
      double ekt = exp(kk*tau);
      (*one)(i,j,k) = 1.;
      (*p)(i,j,k) *= kk*kk/tau*ekt; }}}

  (*p).set_dc_mode(0.0);

  return fftw_normalized_dot_product(*p, *one)/fftw_normalized_dot_product(*one, *one);
}

// <p^2> at all times, for the specific realization of \sigma_f, from A and B; **interacting** theory average
void p_variance(boundary_value_data &d, std::vector<boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &A, std::vector<boost::shared_ptr<kms::fftw_fourier_grid3d_base> > &B)
{
  int i, j, k;
  int N = d.N; int nx_local = (*A[0]).local_nx;
  int T = d.T;

  //  cout << "(t,<p^2>) OUTPUT:" << endl;

  for (int t=0; t<T-1; t++) {
  double tau = d.tau[t];

  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();

  for (i = 0; i < nx_local; i++) {
  for (j = 0; j < N; j++) {
  for (k = 0; k < N/2+1; k++) {
    double kk = (*A[t]).k(i,j,k); 
    if (kk <= d.kmax[t]) {
      double ekt = exp(kk*tau);
      (*p)(i,j,k) = kk*kk/tau*(ekt*(*A[t])(i,j,k) + 1/ekt*(*B[t])(i,j,k)); }
    else
      (*p)(i,j,k) = complex<double>(0.0); }}}

  cout << "{" << tau << "," << std::fixed << fftw_normalized_dot_product(*p, *p) << "},";
  }

  cout << endl << endl;
}

boost::shared_ptr<fftw_fourier_grid3d_base> threeproduct(boundary_value_data &d, double kmax, fftw_fourier_grid3d_base &p1, fftw_fourier_grid3d_base &p2, fftw_fourier_grid3d_base &p3)
{
  int i, j, k;
  int N = d.N; int nx_local = p1.local_nx;

  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes1 = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes2 = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> Kmodes3 = d.allocate_fourier_grid3d();

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      (*Kmodes1)(i,j,k) = p1(i,j,k);
      (*Kmodes2)(i,j,k) = p2(i,j,k);
      (*Kmodes3)(i,j,k) = p3(i,j,k); 
  }}}

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
    if (p1.k(i,j,k) > kmax)
	(*Kmodes1)(i,j,k) = (*Kmodes2)(i,j,k) = (*Kmodes3)(i,j,k) = complex<double>(0,0);
  }}}

  boost::shared_ptr<fftw_grid3d_base> Xpoints1 = d.allocate_grid3d(); 
  boost::shared_ptr<fftw_grid3d_base> Xpoints2 = d.allocate_grid3d(); 
  boost::shared_ptr<fftw_grid3d_base> Xpoints3 = d.allocate_grid3d(); 

  Xpoints1->fill_with_normalized_fft(*Kmodes1);
  Xpoints2->fill_with_normalized_fft(*Kmodes2);
  Xpoints3->fill_with_normalized_fft(*Kmodes3);

  (*Xpoints1) *= (*Xpoints2);
  (*Xpoints1) *= (*Xpoints3);

  boost::shared_ptr<fftw_fourier_grid3d_base> product = d.allocate_fourier_grid3d();
  product->fill_with_normalized_fft(*Xpoints1);

  for (i=0; i<nx_local; i++) {
  for (j=0; j<N; j++) {
  for (k=0; k<N/2+1; k++) {
      if (p1.k(i,j,k) > kmax) 
	  (*product)(i,j,k) = 0.0;
  }}}
  
  if (abs((*product)(0,0,0)) > 1.0e100)
      throw runtime_error("blowup behavior happened!");

  return product;
}

void gradsquare(boundary_value_data &d, double kmax, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &input, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &gradsquareinput)
{
    boost::shared_ptr<fftw_fourier_grid3d_base> inputx = d.clone(*input);
    boost::shared_ptr<fftw_fourier_grid3d_base> inputy = d.clone(*input);
    boost::shared_ptr<fftw_fourier_grid3d_base> inputz = d.clone(*input);
    (*inputx).apply_x_derivative();
    (*inputy).apply_y_derivative();
    (*inputz).apply_z_derivative();

    (*gradsquareinput) += *Square(d, kmax, *inputx);
    (*gradsquareinput) += *Square(d, kmax, *inputy);
    (*gradsquareinput) += *Square(d, kmax, *inputz);
}

void pgradsquare(boundary_value_data &d, double kmax, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &input, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &pgradsquareinput)
{
    boost::shared_ptr<fftw_fourier_grid3d_base> inputx = d.clone(*input);
    boost::shared_ptr<fftw_fourier_grid3d_base> inputy = d.clone(*input);
    boost::shared_ptr<fftw_fourier_grid3d_base> inputz = d.clone(*input);
    (*inputx).apply_x_derivative();
    (*inputy).apply_y_derivative();
    (*inputz).apply_z_derivative();

    (*pgradsquareinput) = *threeproduct(d, kmax, *p, *inputx, *inputx);
    (*pgradsquareinput) += *threeproduct(d, kmax, *p, *inputy, *inputy);
    (*pgradsquareinput) += *threeproduct(d, kmax, *p, *inputz, *inputz);
}
