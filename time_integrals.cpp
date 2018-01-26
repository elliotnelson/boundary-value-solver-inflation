#include <iostream>
#include <kms_fftw.hpp>
#include <vector>
#include <iomanip>

using namespace std;
using namespace kms;

// this file contains several functions which are used to compute the time-integrated three-point correlation function of the underlying theory, which is directly related to the nonlinear part of the action

struct boundary_value_data {
  int T;
  double tauI, tauF;
  double *tau;
  double dlntau;
};

void bisp_equil_time_int(boundary_value_data &d, double kt)
{
  int t;

  double discrete_int = 0.0;

  for (t=0; t<d.T; t++) {
    double tau = d.tau[t];
    discrete_int += d.dlntau*tau*tau*tau*exp(kt*tau); }

  int prec = 12;
  cout << "{" << d.T << "," << std::setprecision(prec) << discrete_int / (-2.0/(kt*kt*kt)) << "},";
  //  cout << "Number of time steps: " << d.T << endl;
  //  cout << "Discrete equilateral bispectrum time integral: " << discrete_int << endl;
  //  cout << "Ratio to continuous integral: " << discrete_int / (-2.0/(kt*kt*kt)) << endl;
}

void bisp_orth_time_int(boundary_value_data &d, double k1, double k2, double k3)
{
  int t;

  double kt = k1+k2+k3;

  double discrete_int = 0.0;

  for (t=0; t<d.T; t++) {
    double tau = d.tau[t];
    discrete_int += d.dlntau*tau*(1-k1*tau)*(1-k2*tau)*exp(kt*tau); }

  double continuous_int = (2*k1*k2/(kt*kt) + k1/kt + k2/kt + 1)/kt;

  int prec = 15;
  cout << "{" << d.T << "," << /*d.tau[d.T-1] << "," <<*/ std::setprecision(prec) << discrete_int / continuous_int << "},";
  //  cout << "Number of time steps: " << d.T << endl;
  //  cout << "Discrete orthogonal bispectrum time integral: " << discrete_int << endl;
  //  cout << "Ratio to continuous integral: " << discrete_int / continuous_int << endl;
 
  //  double ratio = discrete_int / continuous_int - 1;
  //  return ratio;
}

double bisp_equil_error(boundary_value_data &d, double kt)
{
  int t;

  double discrete_int = 0.0;
  for (t=0; t<d.T; t++) {
    double tau = d.tau[t];
    discrete_int += d.dlntau*tau*tau*tau*exp(kt*tau); }

  double error = discrete_int / (-2.0/(kt*kt*kt)) - 1.0;
  return error;
}

double bisp_orth_error(boundary_value_data &d, double k1, double k2, double k3)
{
  int t;

  double kt = k1+k2+k3;

  double discrete_int = 0.0;
  for (t=0; t<d.T; t++) {
    double tau = d.tau[t];
    discrete_int -= d.dlntau*tau*(1-k1*tau)*(1-k2*tau)*exp(kt*tau); }

  double continuous_int = (2*k1*k2/(kt*kt) + k1/kt + k2/kt + 1)/kt;

  double error = discrete_int / continuous_int - 1.0;
  return error;
}

double trispectrum_time_int(boundary_value_data &d, double k1, double k12, double k3, double mu1, double mu2, int m, int n, int flag)
{
  int t, tt;

  double k2 = sqrt(k1*k1+k12*k12-2*k1*k12*mu1);
  double k4 = sqrt(k3*k3+k12*k12-2*k3*k12*mu2);

  double kt = k1+k2+k3+k4;
  //following notation from p. 20 NB3
  double K1 = k1+k2+k12;
  double K2 = k3+k4+k12;
  double K3 = k1+k2-k12;
  double K4 = k3+k4-k12;

  double continuous_int;
  if ((flag == 0) & (m==0) & (n==0)) {
    continuous_int = 1/(k1+k2+k12)/(k3+k4+k12); }
  if ((flag == 0) & (m==1) & (n==1)) {
    continuous_int = 1/(k1+k2+k12)/(k3+k4+k12)/(k1+k2+k12)/(k3+k4+k12); }
  if ((flag == 0) & (m==2) & (n==2)) {
    continuous_int = 4/(k1+k2+k12)/(k3+k4+k12)/(k1+k2+k12)/(k3+k4+k12)/(k1+k2+k12)/(k3+k4+k12); }
  if ((flag == 0) & (m==0) & (n==1)) {
    continuous_int = -1/(k1+k2+k12)/(k3+k4+k12)/(k3+k4+k12); }
  if ((flag == 0) & (m==0) & (n==2)) {
    continuous_int = 2/(k1+k2+k12)/(k3+k4+k12)/(k3+k4+k12)/(k3+k4+k12); }
  if ((flag == 0) & (m==1) & (n==2)) {
    continuous_int = -2/(k1+k2+k12)/(k1+k2+k12)/(k3+k4+k12)/(k3+k4+k12)/(k3+k4+k12); }
  if ((flag == 1) & (m==0) & (n==0)) {
    continuous_int = (kt+2*k12)/(k1+k2+k12)/(k3+k4+k12)/kt; }
  if ((flag == 1) & (m==1) & (n==1)) {
    continuous_int = ((3*k1+3*k2+k3+k4+2*k12)/(k1+k2+k12)/(k1+k2+k12) + (3*k3+3*k4+2*k12+k1+k2)/(k3+k4+k12)/(k3+k4+k12))/kt/kt/kt; }
  if ((flag == 1) & (m==2) & (n==2)) {
    continuous_int = 4*(1/(k3+k4+k12)/(k3+k4+k12)/(k3+k4+k12) + (2*k12*(-6*(k1+k2-k12)*(k1+k2-k12)*(k1+k2+k12)*(k1+k2+k12) - 6*(k1+k2)*(k1+k2-k12)*(k1+k2+k12)*kt - (k12*k12+3*(k1+k2)*(k1+k2))*kt*kt))/(k1+k2+k12)/(k1+k2+k12)/(k1+k2+k12)/kt/kt/kt/kt/kt)/(k1+k2-k12)/(k1+k2-k12)/(k1+k2-k12); }
  if ((flag == 1) & (m==0) & (n==1)) {
    continuous_int = -(1/(k1+k2+k12) + (k1+k2+k12+2*(k3+k4))/(k3+k4+k12)/(k3+k4+k12))/kt/kt; }
  if ((flag == 1) & (m==0) & (n==2)) {
    continuous_int = 2*(1/(k3+k4+k12)/(k3+k4+k12)/(k3+k4+k12) - 2*k12/(k1+k2+k12)/kt/kt/kt)/(k1+k2-k12); }
  if ((flag == 1) & (m==1) & (n==2)) {
    continuous_int = 2*(-1/(k3+k4+k12)/(k3+k4+k12)/(k3+k4+k12) + 2*k12*(-3*k12*k12+(k1+k2)*(5*k1+5*k2+2*(k3+k4)))/(k1+k2+k12)/(k1+k2+k12)/kt/kt/kt/kt)/(k1+k2-k12)/(k1+k2-k12); }

  double discrete_int = 0.0;

  for (t=0; t<d.T; t++) {
    for (tt=0; tt<d.T; tt++) {
      double tau = d.tau[t];
      double tau2 = d.tau[tt];
      //      cout << pow(tau,m+1)*pow(tau2,n+1) << endl; ////
      if (flag == 1) {
	if (tt>t) {
	  discrete_int += d.dlntau*d.dlntau*pow(tau,m+1)*pow(tau2,n+1) * exp((k1+k2)*tau + (k3+k4)*tau2 - k12*(tau2-tau)); }
	else {
	  discrete_int += d.dlntau*d.dlntau*pow(tau,m+1)*pow(tau2,n+1) * exp((k1+k2)*tau + (k3+k4)*tau2 - k12*(tau-tau2)); } }
      //      cout << k12*(tau2-tau) << "int: " << discrete_int << endl; }}
      if (flag == 0) { // & (m==0) & (n==0)) {
	discrete_int += d.dlntau*d.dlntau*pow(tau,m+1)*pow(tau2,n+1) * (exp((k1+k2)*tau + (k3+k4)*tau2 + k12*(tau+tau2))); }
    }}

  //  cout << endl;
  //  cout << "Number of time steps: " << d.T << endl;
  //  cout << "Continuous trispectrum time integral: " << continuous_int << endl;
  //  cout << "Discrete trispectrum time integral: " << discrete_int << endl;
  //  cout << "Ratio: " << discrete_int / continuous_int << endl;

  return discrete_int / continuous_int - 1.;
}

// use when initially fixing tau_initial, dlntau, and number of timesteps T
void fill_tau(boundary_value_data &d)
{
  d.tau = (double*) fftw_malloc(sizeof(double)*d.T);
  int t;  
  d.tau[0] = d.tauI;
  for (t=1; t<d.T; t++)
    d.tau[t] = d.tau[t-1]*exp(-d.dlntau);
}

void fill_tau_2(boundary_value_data &d)
{
  d.tau = (double*) fftw_malloc(sizeof(double)*d.T);
  d.tau[0] = d.tauI;
  int t;
  for (t=1; t<d.T; t++)
    d.tau[t] = d.tau[t-1]*exp(-d.dlntau);
}
