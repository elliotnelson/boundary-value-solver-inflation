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

// this is a function which evaluates the action and its configuration-space gradient for a given input configuration sigma_f and small displacement in configuration space, sigma_kick; it also evaluates the "force" which is an independent measure of the configuration-space gradient, which should yield the same answer of the algorithm in ABsolution() has converged to a correct solution of the equations of motion (this is a nontrivial fact which can be shown analytically)

double forcetest(boundary_value_data &d, double eps, fftw_fourier_grid3d_base &sigma_f, fftw_fourier_grid3d_base &sigma_kick)
{
  int i, j, k;
  int N=d.N; int nx_local = sigma_f.local_nx; int T=d.T;

  boost::shared_ptr<fftw_fourier_grid3d_base> sigma_f1 = d.clone(sigma_kick);
  *sigma_f1 *= -eps;
  *sigma_f1 += sigma_f;
  boost::shared_ptr<fftw_fourier_grid3d_base> sigma_f3 = d.clone(sigma_kick);
  *sigma_f3 *= eps;
  *sigma_f3 += sigma_f;

  int prec = 8;

  cout << "Force Test output: " << endl;
  cout << "epsilon = " << eps << endl;

  relax_output r1 = ABsolution(d, *sigma_f1);
  cout << spacing << "Action(-eps): " << std::setprecision(prec) << r1.actionint + r1.actionfree << "   FreeAction(-eps): " << std::setprecision(prec) << r1.actionfree << endl;

  relax_output r3 = ABsolution(d, *sigma_f3);
  cout << spacing << "Action(+eps): " << std::setprecision(prec) << r3.actionint + r3.actionfree << "   FreeAction(+eps): " << std::setprecision(prec) << r3.actionfree << endl;

  double actionderiv = (r3.actionint + r3.actionfree - r1.actionint - r1.actionfree)/(2.0*eps);
  double freeactionderiv = (r3.actionfree - r1.actionfree)/(2.0*eps);

  double integratedforce = 0.0;
  double integratedforcefree = 0.0;

  double tau = d.tau[T-1];
  for (i = 0; i < nx_local; i++) {
  for (j = 0; j < N; j++) {
  for (k = 0; k < N/2+1; k++) {
      double kk = sigma_f.k(i,j,k);
      (*sigma_f1)(i,j,k) = kk*kk*kk/(1-kk*tau) * sigma_f(i,j,k);  }}}

  relax_output r = ABsolution(d, sigma_f);
  integratedforcefree = fftw_normalized_dot_product(sigma_kick, *sigma_f1);
  integratedforce = integratedforcefree + fftw_normalized_dot_product(sigma_kick, *r.forceint);

  double error = (actionderiv/integratedforce-1)/(integratedforce/integratedforcefree-1);

  cout << spacing <<  "Integrated Force: " << std::setprecision(prec) << integratedforce << "   " << "Integrated Force (f=0):" << std::setprecision(prec) << integratedforcefree << "   Difference: " << std::setprecision(prec) << integratedforce - integratedforcefree << endl;
  cout << spacing << "dS/dsigma: " << std::setprecision(prec) << actionderiv << "   (f=0): " << std::setprecision(prec) << freeactionderiv << "   Difference: " << std::setprecision(prec) << actionderiv - freeactionderiv << endl;
  cout << spacing << "Fractional error (f=0): " << integratedforcefree/freeactionderiv - 1 << endl;
  cout << spacing << "(dS/dsigma - force)/(force - (f=0)force): " << std::setprecision(prec) << error << endl;
  cout << endl;

  return error;
}

void forcetestfree(boundary_value_data &d, fftw_fourier_grid3d_base &sigma_f, fftw_fourier_grid3d_base &sigma_kick, double eps)
{
  int i, j, k; int T=d.T; int N=d.N;

  allocate_arrays(d);
  compute_taukmax(d);
  
  boost::shared_ptr<fftw_fourier_grid3d_base> forcefree = d.allocate_fourier_grid3d();
  double tau = d.tau[T-1];
  for (i = 0; i<N; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    double kk = sigma_f.k(i,j,k);
    (*forcefree)(i,j,k) = kk*kk*kk/(1-kk*tau)*sigma_f(i,j,k); }}}

  double integratedforce = fftw_normalized_dot_product(*forcefree,sigma_kick);

  boost::shared_ptr<fftw_fourier_grid3d_base> sigma_f1 = d.clone(sigma_kick);
  (*sigma_f1) *= (-eps);
  (*sigma_f1) += sigma_f;

  boost::shared_ptr<fftw_fourier_grid3d_base> sigma_f3 = d.clone(sigma_kick);
  (*sigma_f3) *= eps;
  (*sigma_f3) += sigma_f;

  double dSdeps = (freeaction(d,*sigma_f3) - freeaction(d,*sigma_f1))/(2.*eps);

  cout << "Integrated force: " << integratedforce << endl;
  cout << "dS/deps: " << dSdeps << endl;
  cout << "Fractional Difference: " << integratedforce/dSdeps - 1 << endl;
}
