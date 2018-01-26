// This file is a collection of functions used to evaluate the probability P of a given three-dimensional scalar field configuration sigma_f(x1,x2,x3), along with its functional gradient in configuration space, dP/d(sigma_f). The probability is related to an action functional, P=exp(-action), which is computed by a function "action". In the underlying theory, the action is determined as an integral of scalar field values over both space and time, and evaluating it requires solving the equations of motion of the theory which determine the time-evolution of the scalar field, subject to a spatial boundary condition which fixes the final field configuration to be the particular configuration sigma_f(x1,x2,x3) whose probability we are evaluating. For this reason, we need "action" to call additional functions which iteratively implement the equations of motion.

// This file makes use of several libraries (in particular for three-dimensional Fourier arrays) which are not included on GitHub.

// The boundary value problem is solved by calling ABsolution() and supplying it with various input parameters; ABsolution() calls various other functions in order to iteratively find a solution to the equations of motion, and to compute the action with that solution.

#include <kms_fftw.hpp> // not included in github repository
#include <string>
#include <iostream>
#include <fstream>

#include "ngsim.h"
#include "sources.cpp"
#include "sweeps.cpp"
#include "force_test.cpp"
#include "bispectrum_test.cpp"
#include "fourier_field_products.cpp"
#include "change_variables.cpp"

using namespace std;
using namespace kms;
using namespace boost;

// This string will be prepended to each output line, for cosmetic convenience
static const char *spacing = "        ";

// boundary_value_data is a data structure defined in the companion file ngsim.h

// virtual
boost::shared_ptr<fftw_fourier_grid3d_base> boundary_value_data::allocate_fourier_grid3d()
{
    return boost::make_shared<fftw_fourier_grid3d>(this->N, this->N, this->N, this->pixsize);
}

// virtual
boost::shared_ptr<fftw_grid3d_base> boundary_value_data::allocate_grid3d()
{
    return boost::make_shared<fftw_grid3d>(this->N, this->N, this->N, this->pixsize);
}

// clones a 3d array of the Fourier-transformed values of the scalar field
boost::shared_ptr<fftw_fourier_grid3d_base> boundary_value_data::clone(const fftw_fourier_grid3d_base &x)
{
    boost::shared_ptr<fftw_fourier_grid3d_base> ret = this->allocate_fourier_grid3d();
    ret->deep_copy(x);
    return ret;
}

// allocates memory for the time-evolving field as needed, depending on the number of timesteps and number of Fourier modes (input parameters)
void allocate_arrays(boundary_value_data &d)
{
    d.tau = (double*) fftw_malloc(sizeof(double)*d.T);
    d.kmax = (double*) fftw_malloc(sizeof(double)*d.T);
}

// updates time and Fourier momentum values in the data structure "d" which contains all input parameters for the boundary value problem
void compute_taukmax(boundary_value_data &d)
{
  int t;  
  d.dlntau = log((-d.tauI)/(-d.tauF))/(d.T-1);
  d.tau[0] = d.tauI;
  for (t=1; t<d.T; t++)
    d.tau[t] = d.tau[t-1]*exp(-d.dlntau);
  for (t=0; t<d.T; t++)
    if(d.Lambda/(-d.tau[t]) > sqrt(3)*d.N/2) {d.kmax[t] = sqrt(3)*d.N/2;} else {d.kmax[t] = d.Lambda/(-d.tau[t]);};
  // d.Lambda is a parameter in d which fixes a maximum wavelength, removing higher-momentum Fourier modes
}

// this function, called from ABsolution, "sweeps" forward in time and updates half of the canonical field variables (denoted as a vector "B" of three-dimensional arrays which specify field configurations at different times) to satisfy the equations of motion of the theory
// the array s acts as a source in the equation of motion, and includes the nonlinear part of the evolution
void Bsweep(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &A, 
	    vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &B, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &s)
{
  int i, j, k, t;
  int N=d.N; int T=d.T;
  int nx_local = (*s[0]).local_nx;

  for (t = 1; t<T; t++) {
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      double kk = (*s[0]).k(i,j,k); 
      if (kk <= d.kmax[t-1])
	(*B[t])(i,j,k) = (*B[t-1])(i,j,k) + d.dlntau*(*s[t-1])(i,j,k); 
      else
	(*B[t])(i,j,k) = (*B[t-1])(i,j,k);
    }}}
  }

  for (t = 0; t<T; t++) { B[t]->set_dc_mode(0.0); }
}

// this function, called from ABsolution, "sweeps" backward in time and updates the other half of the canonical field variables to satisfy the equations of motion
void Asweep(boundary_value_data &d, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &A, 
	    vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &B, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &s)
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
      double kk = (*s[0]).k(i,j,k); 
      if (kk <= d.kmax[t+1])
	(*A[t])(i,j,k) = (*A[t+1])(i,j,k) + d.dlntau * (*s[t+1])(i,j,k); 
      else
	(*A[t])(i,j,k) = (*A[t+1])(i,j,k);
    }}}
  }
    for (t = 0; t<T; t++) { A[t]->set_dc_mode(0.0); }
}

// action() takes as input a three-dimensional field configuration, the array "sigma_f", as well as the vectors A and B which together contain arrays of the field values and field velocities at different times, and integrates over these arrays to evaluate the action of the underlying field theory. The various Fourier grid manipulations, called from external functions, evalute various terms in the action which are products of the field and its time-derivative and/or spatial gradients of the field. Additional time- and parameter-dependence in the integration is fixed by the underlying theory.
double action(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &A, vector< boost::shared_ptr<fftw_fourier_grid3d_base> > &B)
{
  int i, j, k, t;
  int N=d.N; int nx_local = sigma_f.local_nx; int T=d.T; double f = d.f; double f2 = d.f2;
  double Action = 0.0;
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();
  boost::shared_ptr<fftw_fourier_grid3d_base> sigma = d.allocate_fourier_grid3d();
  double tau_i = d.tau[0];
  
  for (t=0; t<T-1; t++) {
      p->deep_copy(*B[t+1]);
      (*p) -= (*B[t]);
      for (i = 0; i< nx_local; i++) {
      for (j = 0; j< N; j++) {
      for (k = 0; k < N/2+1; k++) {
	  double kk = sigma_f.k(i,j,k);
	  (*p)(i,j,k) *= kk*kk*kk;  }}}

      Action += 2 * fftw_normalized_dot_product(*A[t],*p);
  }

  // note to self:
  // Interacting bulk term S = (Delta/V) sum_{tk} f tau^6 p^3 - c/2 f^2 tau^9 p^4 - f2 tau^2 p (d_i sigma)^2 - c2/2 f2^2 tau^5 p^2 (d_i sigma)^2
  //
  for (t=0; t<T-1; t++) {

    double tau = d.tau[t];
    double tau5 = pow(tau,5);    double tau6 = pow(tau,6);    double tau9 = pow(tau,9);
  
    // Initialize p, sigma
    for (i = 0; i < nx_local; i++) {
	for (j = 0; j < N; j++) {
	    for (k = 0; k < N/2+1; k++) {
		double kk = sigma_f.k(i,j,k); 
		if (kk <= d.kmax[t]) {
		    double ekt = exp(kk*tau);
		    (*p)(i,j,k) = kk*kk/tau*(ekt*(*A[t])(i,j,k) + 1/ekt*(*B[t])(i,j,k)); 
		    (*sigma)(i,j,k) = (1-kk*tau)*ekt*(*A[t])(i,j,k) + (1+kk*tau)/ekt*(*B[t])(i,j,k);
		}
		else
		    (*p)(i,j,k) = (*sigma)(i,j,k) = complex<double>(0.0);
	    }
	}
    }

    boost::shared_ptr<fftw_fourier_grid3d_base> pp = Square(d, d.kmax[t], *p);
    boost::shared_ptr<fftw_fourier_grid3d_base> ppp = cube(d, d.kmax[t], *p);

    Action += d.dlntau * f * tau6 * fftw_normalized_dot_product(*p,*pp);
    Action -= d.dlntau * d.c/2 * tau9 * fftw_normalized_dot_product(*p,*ppp);

    if ((d.f2!=0) || (d.c2!=0)) {

    boost::shared_ptr<fftw_fourier_grid3d_base> gradsigmasquare = d.allocate_fourier_grid3d();
    boost::shared_ptr<fftw_fourier_grid3d_base> pgradsigmasquare = d.allocate_fourier_grid3d();
    gradsquare(d, d.kmax[t], sigma, gradsigmasquare);
    pgradsquare(d, d.kmax[t], sigma, p, pgradsigmasquare);

    Action -= d.dlntau * f2 * tau * tau * fftw_normalized_dot_product(*p, *gradsigmasquare);
    Action += d.dlntau * d.c2/2 * tau5 * fftw_normalized_dot_product(*p, *pgradsigmasquare);

    if(d.bool_pvar==true) {
      double p_variance;
      if (d.bool_ensemble == false) {p_variance = d.pvariance[t]; }
      if (d.bool_ensemble == true) {p_variance = p_var(d, tau); }
    boost::shared_ptr<fftw_fourier_grid3d_base> grad_sigma = d.allocate_fourier_grid3d();
      for (i=0; i<nx_local; i++) {
      for (j=0; j<N; j++) {
      for (k=0; k<N/2+1; k++) {
	(*grad_sigma)(i,j,k) = (*sigma)(i,j,k); }}}
      (*grad_sigma).apply_x_derivative();
      Action -= d.dlntau * d.c2/2 * tau5 * p_variance * fftw_normalized_dot_product(*grad_sigma,*grad_sigma);
      for (i=0; i<nx_local; i++) {
      for (j=0; j<N; j++) {
      for (k=0; k<N/2+1; k++) {
	(*grad_sigma)(i,j,k) = (*sigma)(i,j,k); }}}
      (*grad_sigma).apply_y_derivative();
      Action -= d.dlntau * d.c2/2 * tau5 * p_variance * fftw_normalized_dot_product(*grad_sigma,*grad_sigma);
      for (i=0; i<nx_local; i++) {
      for (j=0; j<N; j++) {
      for (k=0; k<N/2+1; k++) {
	(*grad_sigma)(i,j,k) = (*sigma)(i,j,k); }}}
      (*grad_sigma).apply_z_derivative();
      Action -= d.dlntau * d.c2/2 * tau5 * p_variance * fftw_normalized_dot_product(*grad_sigma,*grad_sigma);
    } }

  }
  // note to self:
  //  double ActionIntBulk = Action - ActionFreeBulk;
  //  cout << spacing << "Interacting bulk action: " << ActionIntBulk << endl;

  // note to self:
  // Initial time boundary term
  for (i = 0; i< nx_local; i++) {
      for (j = 0; j< N; j++) {
	  for (k = 0; k < N/2+1; k++) {
	      double kk = sigma_f.k(i,j,k); 
	      (*p)(i,j,k) = (kk*kk*kk/2.) * (1-kk*tau_i) * exp(2*kk*tau_i) * (*A[0])(i,j,k);
	  }
      }
  }
  Action -= fftw_normalized_dot_product(*p, *A[0]);

  return Action;
}

// evaluates the action for a given field configuration sigma_f, in the case where nonlinearities are set to zero and the field evolves exactly linearly in time
double freeaction(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f)
{
  int i, j, k;
  int N=d.N; int nx_local = sigma_f.local_nx; int T=d.T;
  double tau = d.tau[T-1];

  boost::shared_ptr<fftw_fourier_grid3d_base> tmp = d.clone(sigma_f);
  for (i = 0; i < nx_local; i++) {
      for (j = 0; j < N; j++) {
	  for (k = 0; k < N/2+1; k++) {
	      double kk = sigma_f.k(i,j,k);
	      (*tmp)(i,j,k) *= kk*kk*kk/(1-kk*tau);
	  }
      }
  }

  return 0.5 * fftw_normalized_dot_product(sigma_f, *tmp);
}

// evaluates the "force" which is defined to be the gradient of the action with respect to the input field configuration sigma_f, a particular direction in configuration-space; this function also turns off all nonlinearities in the theory
boost::shared_ptr<fftw_fourier_grid3d_base> force_free(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f)
{
  int i, j, k;
  int N = d.N; int nx_local = sigma_f.local_nx;

  boost::shared_ptr<fftw_fourier_grid3d_base> forcefree = d.allocate_fourier_grid3d();

  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    double kk = sigma_f.k(i,j,k);
    (*forcefree)(i,j,k) = kk*kk*kk/(1-kk*d.tau[d.T-1])*sigma_f(i,j,k);
  }}}

  return forcefree;
}

// evaluates the contributions to the "force" or gradient of the action with respect to the field configuration, arising ONLY from the nonlinear part of the theory
boost::shared_ptr<fftw_fourier_grid3d_base> force_int(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f, boost::shared_ptr<fftw_fourier_grid3d_base> &A_f, boost::shared_ptr<fftw_fourier_grid3d_base> &B_f)
{
  int i, j, k;
  int N = d.N; int nx_local = sigma_f.local_nx;

  int T = d.T;
  double tau = d.tau[T-1]; 

  boost::shared_ptr<fftw_fourier_grid3d_base> forceint = d.allocate_fourier_grid3d();

  for (i = 0; i<nx_local; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    double kk = sigma_f.k(i,j,k); double ekt = exp(kk*tau);
    (*forceint)(i,j,k) = kk*kk*kk*(ekt * (*A_f)(i,j,k) - 1/ekt * (*B_f)(i,j,k) - 1/(1-kk*tau)*sigma_f(i,j,k));
  }}}

  return forceint;
}

// ABsolution() takes as input a field configuration (3d array sigma_f) and solves for the time-evolution of the field (determined by A and B, which are vectors of spatial arrays of field configurations and field-velocity configurations at different times) which evolves according to the equations of motion of the theory, with final boundary condition that the field evolve to the input configuration sigma_f at the final timestep, t=T
// ABsolution() implements the equations of motion iteratively, by (i) updating "B" arrays sweeping forward in time, (ii) updating final field values to satisfy the final boundary condition, (iii) updating "A" arrays sweeping backward in time, (iv) updating field values to satisfy an initial boundayr condition, and repeating (i)-(iv) iteratively until the field evolution converges to the correct solution to the equations of motion
// lastly, ABsolution() calls action(), force_free(), and force_int() to evaluate the action and its gradient for the correct time-dependent field values, and outputs these values collectively in a data structure defined in ngsim.h
relax_output ABsolution(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f)
{
  int t; int T=d.T;
  double converge = 1.0;
  int loop = 0;

  vector< boost::shared_ptr<fftw_fourier_grid3d_base> > A(T);
  vector< boost::shared_ptr<fftw_fourier_grid3d_base> > B(T);
  for (t = 0; t<T; t++) {
      A[t] = d.clone(sigma_f);
      B[t] = d.allocate_fourier_grid3d(); }

  double actionint_0 = 0.;

  // notes to self:
  // if we choose to compute <p^2> for specific realization, use function p_var_realization to compute it
  // note: this option (bool_ensemble==false) doesn't work; using the ensemble <p^2> does work though (it passes the force test)
  if ((d.bool_pvar==true) & (d.bool_ensemble==false)) {
    d.pvariance = (double*) fftw_malloc(sizeof(double)*d.T);
    d.pmean = (double*) fftw_malloc(sizeof(double)*d.T);
    for (t=0; t<T; t++) {
      d.pvariance[t] = p_var_realization(d, sigma_f, d.tau[t]); 
      d.pmean[t] = p_mean_realization(d, sigma_f, d.tau[t]); } }

  // note to self: frozen source arrays
  if (d.flag==0) {
  vector< boost::shared_ptr<fftw_fourier_grid3d_base> > s_a(T);
  vector< boost::shared_ptr<fftw_fourier_grid3d_base> > s_b(T);
  int s_update;
  s_update = 0;
  for (t = 0; t<T; t++) {
      s_a[t] = d.allocate_fourier_grid3d();
      s_b[t] = d.allocate_fourier_grid3d();  }

  while ((loop<d.loop_max) && (converge>d.convergence_threshold)) {
    boost::shared_ptr<fftw_fourier_grid3d_base> A1 = d.clone(*A[T-1]);
    // updates the source arrays s_a and s_b, calling source() from sources.cpp
    for (t=0; t<T-1; t++) {source(d, t, A[t], B[t], s_a[t], s_b[t]);}
    Bsweep(d, B, s_b);
    Afinal(d, sigma_f, A[T-1], B[T-1]);
    boost::shared_ptr<fftw_fourier_grid3d_base> A2 = d.clone(*A[T-1]);
    if (loop==d.loop_max-d.loop_diff) {actionint_0 = action(d, sigma_f, A, B) - freeaction(d, sigma_f);}
    if (s_update==1) {for (t = T-2; t>0; t--) {source_aorb(d, t, false, A[t], B[t], s_a[t]); } }
    Asweep(d, A, s_a);
    converge = fftw_compare(*A1, *A2); // this function is defined in a library not included on github
    cout << "   iteration=" << loop << ": converge=" << converge << endl;
    loop += 1; }
  for (t=0; t<T-1; t++) {source(d, t, A[t], B[t], s_a[t], s_b[t]); }
  Bsweep(d, B, s_b);
  Afinal(d, sigma_f, A[T-1], B[T-1]);
  cout << spacing << "Relaxation Convergence: " << converge << "   Number of Loops: " << loop << endl << endl;
  }

  // note to self: frozen source arrays, with memory optimization
  if (d.flag==1) {
  while ((loop<d.loop_max) && (converge>d.convergence_threshold)) {
    boost::shared_ptr<fftw_fourier_grid3d_base> A1 = d.clone(*A[T-1]);
    for (t=0; t<T-1; t++) {source_mem(d, t, A[t], B[t]);}
    B_sweep_mem(d, B);
    Afinal(d, sigma_f, A[T-1], B[T-1]);
    boost::shared_ptr<fftw_fourier_grid3d_base> A2 = d.clone(*A[T-1]);
    if (loop==d.loop_max-d.loop_diff) {actionint_0 = action(d, sigma_f, A, B) - freeaction(d, sigma_f);}
    A_sweep_mem(d, A);
    converge = fftw_compare(*A1, *A2);
    cout << "   iteration=" << loop << ": converge=" << converge << endl;
    loop += 1; }
  for (t=0; t<T-1; t++) {source_mem_aorb(d, t, true, A[t], B[t]);} // fill B sources, leave A intact
  B_sweep_mem(d, B);
  Afinal(d, sigma_f, A[T-1], B[T-1]);
  cout << spacing << "Relaxation Convergence: " << converge << "   Number of Loops: " << loop << endl << endl;
  }

  // up-to-date source arrays
  if (d.flag==2) {
  while ((loop<d.loop_max) && (converge>d.convergence_threshold)) {
    boost::shared_ptr<fftw_fourier_grid3d_base> A1 = d.clone(*A[T-1]);
    B_sweep_adv(d, A, B);
    //    cout << "B[3]246  " << (*B[3])(2,4,6) << endl; ///
    //    cout << "B[4]246  " << (*B[4])(2,4,6) << endl; ///
    Afinal(d, sigma_f, A[T-1], B[T-1]);
    boost::shared_ptr<fftw_fourier_grid3d_base> A2 = d.clone(*A[T-1]);
    if (loop==d.loop_max-d.loop_diff) {actionint_0 = action(d, sigma_f, A, B) - freeaction(d, sigma_f);}
    A_sweep_adv(d, A, B);
    //    cout << "A[T-1]246  " << (*A[T-1])(2,4,6) << endl; ///
    converge = fftw_compare(*A1, *A2);
    cout << "   iteration=" << loop << ": converge=" << converge << endl;
    loop += 1; }
  B_sweep_adv(d, A, B);
  Afinal(d, sigma_f, A[T-1], B[T-1]);
  cout << spacing << "Relaxation Convergence: " << converge << "   Number of Loops: " << loop << endl << endl;
  }

  // up-to-date source arrays, with memory optimization
  if (d.flag==3) {
    cout << "This option does not (yet) compute the action correctly." << endl; //
  while ((loop<d.loop_max) && (converge>d.convergence_threshold)) {
    boost::shared_ptr<fftw_fourier_grid3d_base> A1 = d.clone(*A[T-1]);
    B_sweep_step(d, A); // A array used for B update
    Afinal(d, sigma_f, A[T-1], A[T-1]); // replace B_f with A_f in A[T-1]
    boost::shared_ptr<fftw_fourier_grid3d_base> A2 = d.clone(*A[T-1]);
    if (loop==d.loop_max-d.loop_diff) {actionint_0 = action(d, sigma_f, A, B) - freeaction(d, sigma_f);}
    A_sweep_step(d, A);
    converge = fftw_compare(*A1, *A2);
    cout << "   iteration=" << loop << ": converge=" << converge << endl;
    loop += 1; }
  B_sweep_step(d, A); // define action here!!
  Afinal(d, sigma_f, A[T-1], A[T-1]);
  cout << spacing << "Relaxation Convergence: " << converge << "   Number of Loops: " << loop << endl << endl;
  }

  //incremental increasing of f
  if (d.flag==4) {
    double f_spacing; cout << "f_spacing = "; cin >> f_spacing;
    double f_final = d.f; double f2_final = d.f2;
    d.f = f_spacing; d.c = d.f*d.f;
    double f2_spacing = f_spacing*f2_final/f_final;
    d.f2 = f2_spacing; d.c2 = d.f2*d.f2;
  while (d.f <= f_final) {
    while ((loop<d.loop_max) && (converge>d.convergence_threshold)) {
      boost::shared_ptr<fftw_fourier_grid3d_base> A1 = d.clone(*A[T-1]);
      for (t=0; t<T-1; t++) {source_mem(d, t, A[t], B[t]);}
      B_sweep_mem(d, B);
      Afinal(d, sigma_f, A[T-1], B[T-1]);
      boost::shared_ptr<fftw_fourier_grid3d_base> A2 = d.clone(*A[T-1]);
      if (loop==d.loop_max-d.loop_diff) {actionint_0 = action(d, sigma_f, A, B) - freeaction(d, sigma_f);}
      A_sweep_mem(d, A);
      converge = fftw_compare(*A1, *A2);
      loop += 1; }
    for (t=0; t<T-1; t++) {source_mem_aorb(d, t, true, A[t], B[t]);}
    B_sweep_mem(d, B);
    Afinal(d, sigma_f, A[T-1], B[T-1]);
    cout << "Convergence for f = " << d.f << endl;
    d.f += f_spacing; d.c = d.f*d.f;
    d.f2 += f2_spacing; d.c2 = d.f2*d.f2;
  }
  d.f = f_final; d.f2 = f2_final;
  }

  boost::shared_ptr<fftw_fourier_grid3d_base> forcefree = force_free(d, sigma_f);
  boost::shared_ptr<fftw_fourier_grid3d_base> forceint = force_int(d, sigma_f, A[T-1], B[T-1]);

  double actionfree = freeaction(d, sigma_f);
  double actionint = action(d, sigma_f, A, B) - actionfree;

  relax_output r;
  r.forceint = forceint;
  r.forcefree = forcefree;
  r.actionint = actionint;
  r.actionfree = actionfree;
  r.actionint_error = abs((actionint - actionint_0)/actionint);

  return r;
}

// this is a variation of ABsolution() which implements the equations of motion with less memory allocation
relax_output ABsolution_mem(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f)
{
  int t; int T=d.T;
  double converge = 1.0;
  int loop = 0;

  vector< boost::shared_ptr<fftw_fourier_grid3d_base> > A(T);
  vector< boost::shared_ptr<fftw_fourier_grid3d_base> > B(T);
  for (t = 0; t<T; t++) {
      A[t] = d.clone(sigma_f);
      B[t] = d.allocate_fourier_grid3d(); }

  double actionint_0 = 0.;

  while ((loop<d.loop_max) && (converge>d.convergence_threshold)) {
    boost::shared_ptr<fftw_fourier_grid3d_base> A1 = d.clone(*A[T-1]);
    cout << "A[T-2] " << (*A[T-2])(2,4,6) << endl; ///
    for (t=0; t<T-1; t++) {source_mem(d, t, A[t], B[t]);}
    B_sweep_mem(d, B);
    Afinal(d, sigma_f, A[T-1], B[T-1]);
    boost::shared_ptr<fftw_fourier_grid3d_base> A2 = d.clone(*A[T-1]);
    if (loop==d.loop_max-d.loop_diff) {actionint_0 = action(d, sigma_f, A, B) - freeaction(d, sigma_f);}
    A_sweep_mem(d, A);
    converge = fftw_compare(*A1, *A2);
    cout << "   iteration=" << loop << ": converge=" << converge << endl;
    loop += 1; }
  for (t=0; t<T-1; t++) {source_mem_aorb(d, t, false, A[t], B[t]);} // fill B sources, leave A intact
  B_sweep_mem(d, B);
  Afinal(d, sigma_f, A[T-1], B[T-1]);

  cout << spacing << "Relaxation Convergence: " << converge << "   Number of Loops: " << loop << endl << endl;

  boost::shared_ptr<fftw_fourier_grid3d_base> forcefree = force_free(d, sigma_f);
  boost::shared_ptr<fftw_fourier_grid3d_base> forceint = force_int(d, sigma_f, A[T-1], B[T-1]);

  double actionfree = freeaction(d, sigma_f);
  double actionint = action(d, sigma_f, A, B) - actionfree;

  relax_output r;
  r.forceint = forceint;
  r.forcefree = forcefree;
  r.actionint = actionint;
  r.actionfree = actionfree;
  r.actionint_error = abs((actionint - actionint_0)/actionint);

  return r;
}

// ignore this function
double action_int_check(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f)
{
  int i, j, k, t;
  int N=d.N; int nx_local = sigma_f.local_nx; double f = d.f; // double f2 = d.f2;
  double action_int = 0.0; 
  boost::shared_ptr<fftw_fourier_grid3d_base> p = d.allocate_fourier_grid3d();

  allocate_arrays(d);
  compute_taukmax(d);

  for (t=0; t<d.T; t++) {
    double tau=d.tau[t];
      for (i = 0; i< nx_local; i++) {
      for (j = 0; j< N; j++) {
      for (k = 0; k < N/2+1; k++) {
	 double kk = sigma_f.k(i,j,k);
	 (*p)(i,j,k) = kk*kk*exp(kk*tau)*sigma_f(i,j,k);
      }}}
  boost::shared_ptr<fftw_fourier_grid3d_base> pp = Square(d, d.kmax[t], *p);
  action_int += d.dlntau * f * tau*tau*tau * fftw_normalized_dot_product(*p,*pp);
  }

  return action_int;
}
