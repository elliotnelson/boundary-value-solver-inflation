#include <kms_fftw.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include "ngsim.h"
#include "write_array.cpp"
#include "change_variables.cpp"
#include "sources.cpp"
#include "sweeps.cpp"
//#include "comparearrays.cpp"
#include "relax.cpp" //_07052015.cpp" //temp change to check original timing
#include "force_test.cpp"
#include "bispectrum_test.cpp"
#include "f_incremental.cpp"
#include "f_convergence.cpp" // need to fix up before using
#include "lambda_test.cpp"
#include "p_variance.cpp"

//temporary:
#include "Square.cpp"
#include "product.cpp"
#include "cube.cpp"
#include "threeproduct.cpp"
#include "gradsquare.cpp"
#include "pgradsquare.cpp"

using namespace std;
using namespace kms;
using namespace boost;


// main() generates a Gaussian random field (in 3 dimensions), ...

int main(int argc, char **argv)
{
  int i, j, k;
  double pi = 3.14159265358979323846;
  boundary_value_data d;
  // d.N = 512;
  cout << "N = "; cin >> d.N;
  const int N = d.N;
  d.boxsize = 2.0*pi; // calibrate this in Mpc
  d.pixsize = d.boxsize/N; // = 2*M_PI/N;

  //TEMP;  cout << "d.const = "; cin >> d.con; ///

  //  d.f = 0.;
  cout << "f1 = "; cin >> d.f;
  cout << "alpha1 = "; cin >> d.c;
  //  d.c = 1.;
  d.c *= d.f*d.f;
  //  d.f2 = 0.;
  cout << "f2 = "; cin >> d.f2;
  cout << "alpha2 = "; cin >> d.c2;
  //  d.c2 = 0.;
  d.c2 *= d.f2*d.f2;

  d.T = 25; //  cout << "T = "; cin >> d.T;
  d.dlntau = 0.6; //  cout << "Delta = "; cin >> d.dlntau;
  d.Lambda = 10.0;
  // cout << "Lambda = "; cin >> d.Lambda;
  d.tauI = -d.Lambda*d.boxsize/(2.0*pi); // cout << "tau_i = "; cin >> d.tauI;
  d.tauF = d.tauI/ exp(d.dlntau*(d.T-1));
  // d.dlntau = log((-d.tauI)/(-d.tauF))/(d.T-1);
  // d.tauF = -d.pixsize/(d.Lambda*2*pi); ///

  d.loop_max = 30; ///
  d.loop_diff = 1;
  d.convergence_threshold = 1.0e-06;

  d.bool_pvar = true; // subtracts off <p^2> in c_2 term
  int boolpvar, bool2;
  if (d.c2!=0) {
  cout << "Subtract off <p^2> in quartic term? (1 or 0): "; cin >> boolpvar;
  if (boolpvar==1) {d.bool_pvar = true;}
  if (boolpvar==0) {d.bool_pvar = false;}
  cout << d.bool_pvar << endl;
  if (d.bool_pvar==true) {
    cout << "Subtract off ensemble <p^2> (1) or realization <p^2> (0): "; cin >> bool2;
  if (bool2==1) {d.bool_ensemble = true;}
  if (bool2==0) {d.bool_ensemble = false;} } }

  // d.shells = 1; // number of shells for StableSquare

  allocate_arrays(d);
  compute_taukmax(d);

  /*  cout << spacing << "PARAMETERS: " << endl;
  cout << spacing << "Box size: " << d.boxsize << endl;
  cout << spacing << "Pixel size: " << d.pixsize << endl;
  cout << spacing << "N = " << d.N << endl;
  cout << spacing << "f = " << d.f << "     " << "f2 = " << d.f2 << endl;
  cout << spacing << "c*f^2 = " << d.c << "     " << "c2*f2^2 = " << d.c2 << endl;
  cout << spacing << "time steps = " << d.T << endl;
  cout << spacing << "time step in ln(tau) = " << d.dlntau << endl;
  cout << spacing << "initial time momentum cutoff = " << d.Lambda << endl;
  cout << spacing << "initial time = " << d.tauI << endl;
  cout << spacing << "final time = " << d.tauF << endl;
  cout << spacing << "maximum number of loops = "; cin >> d.loop_max;
  cout << spacing << "convergence threshold = "; cin >> d.convergence_threshold; */

  /*  fftw_fourier_grid3d sigma_f(N, N, N, d.pixsize);
  rng r;
  sigma_f.simulate_gaussian_normalized(r);
  for (i = 0; i<N; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    sigma_f(i,j,k) = sigma_f(i,j,k)/sqrt(pow(sigma_f.k(i,j,k),3)); }}}
  sigma_f.set_dc_mode(0.0);
  sigma_f.write("sigma_f_32_saved_2"); */

  //  UNNECESSARY:
  //  const std::string sigma_f_filename = "sigma_f_16_saved";
  //  write_array(N, d.pixsize, sigma_f_filename);

  fftw_fourier_grid3d sigma_f("sigma_f_32_saved"); // this is the original array saved to disk, with boxsize=2pi

  fftw_fourier_grid3d sigma_kick("sigma_f_32_saved_2");

  d.flag = 0;
  //  cout << "flag = "; cin >> d.flag;

  //  f_convergence(d, sigma_f);
  //  cout << relaxation_incremental(d, sigma_f, sigma_kick, 100, .05, false);

  //  lambda_test(d,20.,pow(10,0.1),20.,sigma_f);

  /*  fftw_fourier_grid3d sigma_kick(N, N, N, d.pixsize);
  rng r2;
  sigma_kick.simulate_gaussian_normalized(r2);
  for (i = 0; i<N; i++) {
  for (j = 0; j<N; j++) {
  for (k = 0; k<N/2+1; k++) {
    sigma_kick(i,j,k) = sigma_kick(i,j,k)/sqrt(pow(sigma_kick.k(i,j,k),3)); }}}
    sigma_kick.set_dc_mode(0.0); */

      //      if (random->local_nx != random->global_nx)
      //	  throw runtime_error("Fatal: do_forceactioncheck is only supported in serial code");
  forcetest(d, pow(10,-3), sigma_f, sigma_kick); //action is linearly related to eps for eps<.0001 or so; unstable somewhere between 10^-7 and 10^-12; f=0 force test passes most closely for 10^-3 or 10^-4

  //  cout << spacing << "BOUNDARY VALUE SOLVER OUTPUT:" << endl << endl;
  //  relax_output out = ABsolution(d, sigma_f);  
  ///  relax_output out = ABsolution_mem(d, sigma_f);

  //just define directly in terms of out.forceint
  /*  fftw_fourier_grid3d forceint(N, N, N, d.pixsize);
  for (i = 0; i< N; i++) {
  for (j = 0; j< N; j++) {
  for (k = 0; k < N/2+1; k++) {
  forceint(i,j,k) = (*out.forceint)(i,j,k); }}} */
  //  forceint.write("forceint_original"); //forceint from sigma_f_original is in file forceint_original
 
  //  int prec = 10;

  /*  fftw_fourier_grid3d forceint_original("forceint_original");
  cout << "Compare unmodified and modified output: " << endl;
  cout << "Difference between forceint from unoptimized and optimized codes: " << std::setprecision(prec) << fftw_compare(forceint, forceint_original) << endl; */

  /*  cout << "S_int: " << std::setprecision(prec) << out.actionint << endl;
  cout << "S_free: " << std::setprecision(prec) << out.actionfree << endl;
  cout << "Uncertainty in S_int: " << std::setprecision(prec) << out.actionint_error << endl; */
  //note: for unmodified action(), actionfree+actionint from sigma_f_original = 1608.54465568241 + 0.337161502714707

  //  double df = 0.01;
  //  bispectrumtest_f2(d, sigma_f, df);

  //  f_inc_spacing(d, sigma_f); // checked that this uses equivalent algorithm to flag=0,1 options in ABsolution

  // FOR MAKING MOVIE: ================>
  /*  <- COMMENTING OUT FOR NOW
  
  //  int Nk=0; // CURRENTLY SET FOR 2D ARRAY
  // 2D array
  double ns = 1.;
  cout << "spectral index (1=flat) = "; cin >> ns;
  fftw_fourier_grid2d sigma_f(N, N, d.pixsize); ///
  rng r;
  sigma_f.simulate_gaussian_normalized(r);
  for (i = 0; i<N; i++) {
  for (k = 0; k<N/2+1; k++) {
    sigma_f(i,k) *= pow(sigma_f.k(i,k),0.5*ns-1.5);
    // different from 3D case
    // this ensures that the power spectrum ~ k^(ns-3)=1/k^2 for flat 2D power
    if (sigma_f.k(i,k)==0.) {sigma_f(i,k) = 0.; } }}

  //  std::string sig = "sigma";

  // USE TO PUT ALL ARRAYS INTO SAME FILE:
  //  string stringname = "sigma_256.txt";
  //  const char* filename = stringname.c_str();
  //ofstream sigma_t(filename);
  //streambuf *coutbuf = cout.rdbuf(); // save old buf
  //cout.rdbuf(sigma_t.rdbuf());
  //cout << "{";

  int t; int T = d.T;
  int tmin = 0;
  cout << "t_max = " << d.T-1 << endl;
  cout << "t_min = "; cin >> tmin;
  for (t = tmin; t < T; t++) {
    double tau = d.tau[t];

    cout << "Start of t = " << t << endl; ///

    fftw_fourier_grid2d sigma(N, N, d.pixsize); // 2D case
    // 3D:  fftw_fourier_grid3d sigma("sigma_f_512_saved");
    //  boost::shared_ptr<fftw_fourier_grid3d_base> sigma = d.allocate_fourier_grid2d();

    //  int kfix = N/2;
  for (i = 0; i < N; i++) {
    //  for (j = 0; j < N; j++) {
  for (k = 0; k < N/2+1; k++) {
    double kk = sigma.k(i,k);  double ekt = exp(kk*tau);
    sigma(i,k) = (1-kk*tau)*ekt*sigma_f(i,k); }}

  fftw_grid2d field(N, N, d.pixsize);
  field = sigma.normalized_fft(); // KMS library: fftw_grid2d normalized_fft()
  // 3D:  boost::shared_ptr<fftw_grid3d_base> field = d.allocate_grid3d();
  // 3D:  field -> fill_with_normalized_fft(sigma);

  // NONZERO f_1; 2D
  // I'd want to store the contribution so far, and accumulate into it the real-space product of the capital "Sigma" field (dS/df discrete)
  // (*sigma2)(i,j,k) = kk*kk/(1-kk*tau_f)*(ekt/ektf)* sigma_f(i,j,k);

  // USE TO PUT EACH TIME SLICE IN DIFFERENT FILE
  string stringname = "sigma_" + boost::lexical_cast<std::string>(t) + ".txt";
  const char* filename = stringname.c_str();
  ofstream sigma_t(filename);
  streambuf *coutbuf = cout.rdbuf(); // save old buf
  cout.rdbuf(sigma_t.rdbuf());

  cout << "{";
  //  k = N/2; // k = 0;
  //  int jmax = N-1;
  int kmax = N/2;
  for (i = 0; i<N; i++) {
    if (i==N-1) {kmax = N/2-1; } // {jmax = N-2; }
  for (k = 0; k<kmax+1; k++) {
    cout << "{" << i << "," << k << "," << std::fixed << field(i,k) << "},"; }}
  i = N-1; k = N/2; // j = N-1;
  // in 3d case, was using pointer notation (*field)(i,j,k);
  cout << "{" << i << "," << k << "," << std::fixed << field(i,k) << "}}";

  cout.rdbuf(coutbuf); // reset to standard output
  cout << "Done with t = " << t << endl;

  //  if (t<T-1) {cout << ","; }

  }

  //  cout << "}";
  //  cout.rdbuf(coutbuf); // reset to standard output

  // <========== FOR MAKING MOVIE
  */ // <- COMMENTING OUT FOR NOW
 
  return 0;

}

  //  double abs_error_average = 0.0; // average force test error over L trials
  ///  double error_squares_sum = 0.0; // to compute variance in force test error

  //  cout << endl << "Number of trials: " << Lmax << endl;
  //  int Lmax = 1;
  //  for (int L = 0; L<Lmax; L++) { ... }

  //  double epsilon = 0.001;
  //  forcetestfree(d,sigma_f,sigma_kick,epsilon);

  //  abs_error_average += sqrt((relax_out.force_test_error)*(relax_out.force_test_error));
  ///  error_squares_sum += (relax_out.force_test_error)*(relax_out.force_test_error);

  /*  double action_int = action_int_check(d, sigma_f); // the on-shell interacting part of S_E at O(f) for f2=0
  cout << endl;
  cout << spacing << "On-shell interacting part of S_E = " << action_int << endl; */

  //  abs_error_average *= 1./Lmax;
  //  cout << endl << "Average force test error: " << abs_error_average << endl;
  ///  cout << endl << "RMS variation of force test error: " << sqrt(error_squares_sum/Lmax + (1/Lmax-2)*error_average*error_average) << endl;
  //  relaxation(d, sigma_f, sigma_kick, false);

  // check free action: this is same as output from relaxation, up to tau_f difference
  //  sigma_f.set_dc_mode(0.0);
  //  cout << spacing << "freeaction: " <<  freeaction(d, sigma_f) << endl;
