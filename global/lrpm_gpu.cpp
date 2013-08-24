  /*
 lrpm_gpu.cpp (global memory access)
 This file is part of  LRPM-CUDA

 Copyright (C)  2013 -  A. Boer boera@unitbv.ro

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.

 */

#include <alps/alea.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <cuda.h>
#include <curand.h>
#include <gsl/gsl_sf_zeta.h>
#include <cutil.h>
#include "input.h"

using namespace std;

#define CUDA_ERROR_CHECK

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif
	return;
}

extern float JJ(int L, int rij, float sigma);

extern void therm_steps_w(int *s, float *JJJ, float *ran, float *T);

extern void prod_steps_w(int *s, float *JJJ, float *ran, float *T, float *E, int *out, float *M);

extern void pt_w(int *s, float *E, float *M, float *T, float *ran);

float JJ(int L, int rij, float sigma)
{
	float res1, res2, coupling;
	res1 = gsl_sf_hzeta (1 + sigma, 1 + (float)rij / (float)L);
	res2 = gsl_sf_hzeta (1 + sigma, 1 - (float)rij / (float)L);
	coupling = 1 / (pow((float)rij, (1 + sigma))) + 1 / (pow((float)L, (1 + sigma))) * \
	           (res1 + res2);
	return coupling;
}

void measurements(float *E, float *M, alps::RealObservable *energy, alps::RealObservable *mag, \
                  alps::RealObservable *energy2, alps::RealObservable *energy4, \
                  alps::RealObservable *mag2, alps::RealObservable *mag4)
{
	for (int i=0; i<N_R; i++) {
		double e_per_spin = E[i]/(double)N;
		double m_per_spin = M[i];
		double e2_per_spin = e_per_spin * e_per_spin;
		double e4_per_spin = e2_per_spin * e2_per_spin;
		double m2_per_spin = m_per_spin * m_per_spin;
		double m4_per_spin = m2_per_spin * m2_per_spin;
		energy[i] << e_per_spin;
		mag[i] << m_per_spin;
		energy2[i] << e2_per_spin;
		energy4[i] << e4_per_spin;
		mag2[i] << m2_per_spin;
		mag4[i] << m4_per_spin;
	}
}

int main()
{
	int *s_h, *s_d;
	float *JJJ_h, *JJJ_d;
	float *E_h, *E_d;
	float *M_h, *M_d;
	int *out_h, *out_d;
	float *ran_d, *ran2_d;
	float *T_h, *T_d;
	alps::RealObservable energy[N_R];
	alps::RealObservable mag[N_R];
	alps::RealObservable energy2[N_R];
	alps::RealObservable energy4[N_R];
	alps::RealObservable mag2[N_R];
	alps::RealObservable mag4[N_R];
	float sigma = SIGMA;
	curandGenerator_t gen;
	curandGenerator_t gen2;
	float T_min = Tmin;
	float T_max = Tmax;
	float dT = (T_max-T_min)/((float)(N_R-1));

	unsigned int timer=0;
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	s_h = new int[N*N_R];
	JJJ_h = new float[N];
	E_h = new float[N_R];
	M_h = new float[N_R];
	out_h = new int[R*N_R];
	T_h = new float[N_R];

	cudaMalloc((void **) &s_d, N * N_R * sizeof(int));
	cudaMalloc((void **) &JJJ_d, N * sizeof(float));
	cudaMalloc((void **) &E_d, N_R * sizeof(float));
	cudaMalloc((void **) &M_d, N_R * sizeof(float));
	cudaMalloc((void **) &out_d, R*N_R*sizeof(int));
	cudaMalloc((void **) &T_d, N_R*sizeof(float));
	cudaMalloc((void **) &ran_d, 3*N*N_R*sizeof(float));
	cudaMalloc((void **) &ran2_d, N_R*sizeof(float));

	JJJ_h[0] = 0;

	for (int i = 1; i < N; i++)
		JJJ_h[i] = JJ(N, i, sigma);

	for (int i=0; i<N_R; i++) {
		T_h[i]=T_min + (float)i * dT;
	}

	for (int i = 0; i < N * N_R; i++)
		s_h[i] = 1;

	cudaMemcpy(JJJ_d, JJJ_h, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(s_d, s_h, N * N_R * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(T_d, T_h, N_R * sizeof(float), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen2, 4567ULL);

	for (int i=0; i<MC_therm; i++) {
		curandGenerateUniform(gen, ran_d, 3*N*N_R);
		therm_steps_w(s_d, JJJ_d, ran_d, T_d);
		CudaCheckError();
	}

	for (int i=0; i<MC_prod; i++) {
		curandGenerateUniform(gen, ran_d, 3*N*N_R);
		prod_steps_w(s_d, JJJ_d, ran_d, T_d, E_d, out_d, M_d);
		curandGenerateUniform(gen2, ran2_d, N_R);
		pt_w(s_d, E_d, M_d, T_d, ran2_d);
		CudaCheckError();
		cudaThreadSynchronize();
		cudaMemcpy(E_h, E_d, N_R * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(M_h, M_d, N_R * sizeof(float), cudaMemcpyDeviceToHost);
		measurements(E_h, M_h, energy, mag, energy2, energy4, mag2, mag4);
	}

	double Umax = 0;
	double Umaxerr = 0;
	
	cout << "# BLOCK_SIZE = " << BLOCK_SIZE << endl;
	cout << "# R = " << R << endl;
	cout << "# N = " << N << endl;
	cout << "# N_R = " << N_R << endl;
	cout << "# SIGMA = " << SIGMA << endl;
	cout << "# Tmin = " << Tmin << endl;
	cout << "# Tmax = " << Tmax << endl;
	cout << "# dT = " << dT << endl;
	cout << "# MCtherm = " << MC_therm << endl;
	cout << "# MCprod = " << MC_prod << endl;

	for (int i=0; i<N_R; i++) {
		alps::RealObsevaluator energy2ev(energy2[i]);
		alps::RealObsevaluator energy4ev(energy4[i]);
		alps::RealObsevaluator mag2ev(mag2[i]);
		alps::RealObsevaluator mag4ev(mag4[i]);
		alps::RealObsevaluator U = energy4ev/(energy2ev*energy2ev);
		alps::RealObsevaluator Binder = 1.0 - mag4ev/(3.0*mag2ev*mag2ev);
		if (Umax < U.mean()) {
			Umax = U.mean();
			Umaxerr = U.error();
		}
		cout.setf(ios::fixed, ios::floatfield);
		cout.precision(3);
		cout << i << '\t' << T_h[i] << '\t';
		cout.precision(6);
		cout << energy[i].mean() << '\t' << energy[i].error() << '\t';
		cout << mag[i].mean() << '\t' << mag[i].error() << '\t';
		cout << U.mean() << '\t' << U.error() << '\t';
		cout << Binder.mean() << '\t' << Binder.error() << endl;

	}

	cout << "# Umax = " << Umax << " +/- " << Umaxerr << endl;

	cutStopTimer(timer);
	float sim_time=(float)cutGetTimerValue(timer)/1000;

	cout.precision(3);
	cout << "# Simulation time: " << sim_time << " seconds" << endl;

	cutDeleteTimer(timer);

	free(s_h);
	free(JJJ_h);
	free(E_h);
	free(M_h);
	cudaFree(s_d);
	cudaFree(JJJ_d);
	cudaFree(E_d);
	cudaFree(M_d);

	return 0;

}
