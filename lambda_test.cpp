#include <iostream>
#include <kms_fftw.hpp>
#include <vector>
#include <iomanip>

#include "ngsim.h"

using namespace std;
using namespace kms;
using namespace boost;

void lambda_test(boundary_value_data &d, double lambda_max, double lambda_mult, double N_lambda, fftw_fourier_grid3d &sigma_f)
{
  cout << "Test S_E for various Lambda:" << endl << endl;
  cout << "Lambda_min = " << lambda_max/pow(lambda_mult,N_lambda-1) << endl;
  cout << "Lambda_max = " << lambda_max << endl;
  cout << "Ratio of subsequent Lambda_i = " << lambda_mult << endl;

  double pi = 3.14159265358979323846;

  d.Lambda=lambda_max*lambda_mult;
  d.tauI = -d.Lambda*d.boxsize/(2.0*pi);
  d.tauF = d.tauI/ exp(d.dlntau*(d.T-1)); // just be sure we're avoiding tau_f error
  allocate_arrays(d);
  compute_taukmax(d);

  relax_output out = ABsolution(d, sigma_f);
  double S_int_limit = out.actionint;
  //DELETE: relaxation_write(d, sigma_f, sigma_kick, false); // modified function; in relax_main_delete.cpp; OK I guess it's not there

  double lambda_min = lambda_max/pow(lambda_mult,N_lambda);
  for (d.Lambda=lambda_max; d.Lambda > lambda_min; d.Lambda *= 1/lambda_mult)
    {
      compute_taukmax(d); // fix kmax(Lambda)
      cout << "{" << d.Lambda << ",";
      int prec = 12;
      cout << std::fixed << std::setprecision(prec) << (ABsolution(d,sigma_f)).actionint / S_int_limit - 1.;
      cout << "},";
    }
}
