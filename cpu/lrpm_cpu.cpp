 /*
 lrpm_cpu.cpp
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
#include <gsl/gsl_sf_zeta.h>
#include <time.h>
#include "input.h"

using namespace std;

double JJ(int L, int rij, double sigma)
{
	double res1, res2, coupling;
	res1 = gsl_sf_hzeta (1+sigma, 1+(double)rij/(double)L);
	res2 = gsl_sf_hzeta (1+sigma, 1-(double)rij/(double)L);
	coupling = 1 / (pow((double)rij,(1+sigma))) + 1 / (pow((double)L,(1+sigma))) * \
	           (res1+res2);
	return coupling;
}

double calc_energy(int *s, double *JJJ, int r)
{
	double E=0;
	int i,j;
	for (i=0; i<N-1; i++) {
		for (j=i+1; j<N; j++) {
			if (s[i+r*N]==s[j+r*N]) E-=JJJ[j-i];
		}
	}
	return E;
}

double calc_mag(int *s, int r)
{
	double M;
	int ns[R];
	int i, j;
	for (i=0; i<R; i++)
		ns[i]=0;

	for (i=0; i<N; i++) {
		for (j=0; j<R; j++) {
			if (s[i+r*N]==j+1) ns[j]++;
		}
	}

	int max=ns[0];
	for (i=1; i<R; i++)
		if (ns[i]>max) max=ns[i];

	M=((float)R*max/N-1)/((float)R-1);

	return M;
}

void metropolis(int *s, double *JJJ, double *ran, double *T)
{
	int r, i, pos, sold, snew;
	double Eold, Enew, deltaE;

	for (r=0; r<N_R; r++) {
		pos=r*N+N*ran[r];
		sold=s[pos];
		snew=1 + ((int)(sold+(R-1)*ran[r+N_R]))%R;
		Eold=0;
		Enew=0;

		for (i=r*N; i<r*N+N; i++) {
			int rij=i-pos;
			if (rij<0) rij=-rij;
			if (rij) {
				if (s[i]==sold) Eold-=JJJ[rij];
				if (s[i]==snew) Enew-=JJJ[rij];
			}
		}

		deltaE=Enew-Eold;

		if (deltaE<0) {
			s[pos]=snew;
		}
		else {
			if (exp(-deltaE/T[r])>ran[r+2*N_R]) {
				s[pos]=snew;
			}
			else {
				s[pos]=sold;
			}
		}
	}
}

void pt(int *s, double *E, double *M, double *T, double *ran)
{
	int i, j;
	for (i=0; i<N_R-1; i++) {
		double delta = (1/T[i]-1/T[i+1])*(E[i]-E[i+1]);
		if (exp(delta)>ran[i]) {
			for (j=i*N; j<i*N+N; j++) {
				int s_temp=s[j];
				s[j]=s[j+N];
				s[j+N]=s_temp;
			}
			double E_temp=E[i];
			E[i]=E[i+1];
			E[i+1]=E_temp;
			double M_temp=M[i];
			M[i]=M[i+1];
			M[i+1]=M_temp;
		}
	}
}

void measurements(double *E, double *M, alps::RealObservable *energy, alps::RealObservable *mag, \
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
	int *s;
	double *JJJ, *E, *M, *ran, *T;
	int i, step, step2;
	double sigma=SIGMA;
	double T_min = Tmin;
	double T_max = Tmax;
	double dT = (T_max-T_min)/((double)(N_R-1));

	clock_t start, end;
	double sim_time;
	start = clock();

	s = new int[N*N_R];
	JJJ = new double[N];
	E = new double[N_R];
	M = new double[N_R];
	ran = new double[3*N_R];
	T = new double[N_R];

	alps::RealObservable energy[N_R];
	alps::RealObservable mag[N_R];
	alps::RealObservable energy2[N_R];
	alps::RealObservable energy4[N_R];
	alps::RealObservable mag2[N_R];
	alps::RealObservable mag4[N_R];

	for (i=0; i<N*N_R; i++)
		s[i]=1;

	JJJ[0]=0;

	for (i=1; i<N; i++)
		JJJ[i]=JJ(N,i,sigma);

	for (i=0; i<N_R; i++)
		T[i]=T_min + (double)i * dT;

	for (step=0; step<MC_therm; step++) {
		for (step2=0; step2<N; step2++) {
			for (i=0; i<3*N_R; i++)
				ran[i]=drand48();
			metropolis(s,JJJ, ran, T);
		}
	}

	for (step=0; step<MC_prod; step++) {
		for (step2=0; step2<N; step2++) {
			for (i=0; i<3*N_R; i++)
				ran[i]=drand48();
			metropolis(s, JJJ, ran, T);
		}
		for (i=0; i<N_R; i++) {
			E[i]=calc_energy(s,JJJ,i);
			M[i]=calc_mag(s,i);
		}
		for (i=0; i<N_R; i++)
			ran[i]=drand48();
		pt(s,E,M,T,ran);
		measurements(E, M, energy, mag, energy2, energy4, mag2, mag4);
	}

	double Umax = 0;
	double Umaxerr = 0;
	
	cout << "# R = " << R << endl;
	cout << "# N = " << N << endl;
	cout << "# N_R = " << N_R << endl;
	cout << "# SIGMA = " << SIGMA << endl;
	cout << "# Tmin = " << Tmin << endl;
	cout << "# Tmax = " << Tmax << endl;
	cout << "# dT = " << dT << endl;
	cout << "# MCtherm = " << MC_therm << endl;
	cout << "# MCprod = " << MC_prod << endl;

	for (i=0; i<N_R; i++) {
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
		cout << i << '\t' << T[i] << '\t';
		cout.precision(6);
		cout << energy[i].mean() << '\t' << energy[i].error() << '\t';
		cout << mag[i].mean() << '\t' << mag[i].error() << '\t';
		cout << U.mean() << '\t' << U.error() << '\t';
		cout << Binder.mean() << '\t' << Binder.error() << endl;
	}

	cout << "# Umax = " << Umax << " +/- " << Umaxerr << endl;

	end = clock();
	sim_time = ((double) (end - start)) / CLOCKS_PER_SEC;

	cout.precision(3);
	cout << "# Simulation time: " << sim_time << " seconds" << endl;


	free(s);
	free(JJJ);
	free(E);
	free(M);
	free(ran);
	free(T);

	return 0;
}
