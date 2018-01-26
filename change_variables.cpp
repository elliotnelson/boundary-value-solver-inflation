#include <kms_fftw.hpp>
#include <vector>
#include "ngsim.h"

using namespace std;
using namespace kms;
using namespace boost;

// this function converts the (A,B) arrays to arrays for the field sigma and its rescaled time-derivative "p", with a particular time-dependence that comes from the underlying theory and definitions of (A,B)

void fill_sigma_p(boundary_value_data &d, int t, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &A3d, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &B3d, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &sigma, boost::shared_ptr<kms::fftw_fourier_grid3d_base> &p)
{
  int i, j, k;
  int N = d.N;
  int nx_local = (*A3d).local_nx;

    double tau = d.tau[t]; 
    for (i = 0; i<nx_local; i++) {
    for (j = 0; j<N; j++) {
    for (k = 0; k<N/2+1; k++) {
      double kk = (*p).k(i,j,k); 
	if (kk <= d.kmax[t]) {
	    double ekt = exp(kk*tau);
	    (*p)(i,j,k) = kk*kk/tau*(ekt*(*A3d)(i,j,k) + 1/ekt*(*B3d)(i,j,k));
	    (*sigma)(i,j,k) = (1-kk*tau)*ekt*(*A3d)(i,j,k) + (1+kk*tau)/ekt*(*B3d)(i,j,k); 
	}
	else
	    (*p)(i,j,k) = (*sigma)(i,j,k) = complex<double>(0,0);
    }}}
}
