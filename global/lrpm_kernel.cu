  /*
 lrpm_kernel.cu (global memory access)
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

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include "input.h"

__device__ float energy(int *s, float *JJJ)
{
	int tid = threadIdx.x;
	int rid = blockIdx.x;
	int spt = (N % BLOCK_SIZE == 0 ? N / BLOCK_SIZE : N / BLOCK_SIZE + 1);
	__shared__ float ee[BLOCK_SIZE];
	float E;
	int ind1, ind2, j;

	ee[tid] = 0;
	if (tid == 0) E = 0;

	__syncthreads();

	j = 0;
	while (j < spt) {
		int spin = tid * spt + j;
		if (spin < N - 1) {
			for (int i = 0; i < N / 2; i++) {
				if (spin + 1 > i) {
					ind1 = i;
					ind2 = spin + 1;
				}
				else {
					ind1 = N - i - 1;
					ind2 = N - spin - 1;
				}
				int s_ind1 = rid*N+ind1;
				int s_ind2 = rid*N+ind2;
				int rij = ind2 - ind1;
				if (s[s_ind1] == s[s_ind2]) ee[tid] -= JJJ[rij];
			}
		}
		j++;
	}

	__syncthreads();

	for (unsigned int i = 1; i < BLOCK_SIZE; i *= 2) {
		int index = 2 * i * threadIdx.x;
		if (index < BLOCK_SIZE)
			ee[index] += ee[index + i];
		__syncthreads();
	}

	if (tid == 0) E = ee[0];
	return E;
}

__global__ void calc_energy(int *s, float *JJJ, float *E)
{
	int tid = threadIdx.x;
	int rid = blockIdx.x;
	float ee;

	__syncthreads();

	ee = energy(s, JJJ);

	if (tid == 0) {
		E[rid] = ee;
	}
}

__global__ void calc_mag(int *s, int *out, float *M)
{
	int tid = threadIdx.x;
	int rid = blockIdx.x;
	int spt = (N % BLOCK_SIZE == 0 ? N / BLOCK_SIZE : N / BLOCK_SIZE + 1);
	int max;

	if (tid==0) {
		for (int i=0; i<R; i++)
			out[rid*R+i]=0;
	}

	__syncthreads();

	int j=0;
	while (j<spt) {
		for (int i=0; i<R; i++) {
			int spin = tid*spt+j;
			if (spin < N) {
				if (s[rid*N+spin]==i+1) atomicAdd(&out[rid*R+i], 1);
			}
		}
		j++;
	}

	__syncthreads();

	if (tid==0) {
		max=out[rid*R];
		for (int i=1; i<R; i++)
			if (max<out[rid*R+i]) max=out[rid*R+i];
		M[rid]=((float)R*max/N-1)/((float)R-1);
	}
}

__global__ void metropolis(int *s, float *JJJ, float *ran, float *T)
{
	int tid = threadIdx.x;
	int rid = blockIdx.x;
	int spt = (N % BLOCK_SIZE == 0 ? N / BLOCK_SIZE : N / BLOCK_SIZE + 1);
	float eold, enew, deltaE;
	__shared__ int pos, sold, snew;
	__shared__ float eeold[BLOCK_SIZE];
	__shared__ float eenew[BLOCK_SIZE];

	__syncthreads();

	int step = 0;
	while (step < N) {
		eeold[tid]=0;
		eenew[tid]=0;
		if (tid==0) {
			pos = N * ran[3*N_R*step+rid];
			sold = s[pos+rid*N];
			snew = 1 + ((int)(sold+(R-1)*ran[3*N_R*step+rid+N_R]))%R;
		}

		__syncthreads();

		int j=0;
		while (j<spt) {
			if (tid*spt+j<N) {
				int rij=tid*spt+j-pos;
				if (rij<0) rij=-rij;
				if (rij) {
					if (sold==s[tid*spt+j+rid*N]) eeold[tid]-=JJJ[rij];
					if (snew==s[tid*spt+j+rid*N]) eenew[tid]-=JJJ[rij];
				}
			}
			j++;
		}

		__syncthreads();

		for (unsigned int i = 1; i < BLOCK_SIZE; i *= 2) {
			int index = 2 * i * threadIdx.x;
			if (index < BLOCK_SIZE) {
				eeold[index] += eeold[index + i];
				eenew[index] += eenew[index + i];
			}
			__syncthreads();
		}

		if (tid==0) {
			eold = eeold[0];
			enew = eenew[0];
			deltaE=enew-eold;
			if (deltaE < 0) {
				s[pos+rid*N]=snew;
			}
			else {
				if (__expf(-deltaE/T[rid])>ran[3*N_R*step+rid+2*N_R]) {
					s[pos+rid*N]=snew;
				}
				else {
					s[pos+rid*N]=sold;
				}
			}
		}

		__syncthreads();

		step++;
	}
}

__global__ void pt(int *s, float *E, float *M, float *T, float *ran, int flag)
{

	int tid = threadIdx.x;
	int rid;
	int spt = (N%BLOCK_SIZE==0 ? N/BLOCK_SIZE : N/BLOCK_SIZE+1);
	__shared__ float ratio;
	if (flag==0) rid = 2 * blockIdx.x;
	else rid = 2 * blockIdx.x + 1;

	if (rid<N_R-1) {
		if (tid==0) {
			float delta = (1/T[rid]-1/T[rid+1])*(E[rid]-E[rid+1]);
			ratio = exp(delta);
		}

		__syncthreads();

		if (ratio > ran[rid]) {
			int j=0;
			while (j<spt) {
				int spin = tid * spt + j;
				if (spin<N) {
					int temp = s[rid*N+spin];
					s[rid*N+spin]=s[rid*N+spin+N];
					s[rid*N+spin+N]=temp;
				}
				j++;
			}

			if (tid==0) {
				float e_temp = E[rid];
				E[rid]=E[rid+1];
				E[rid+1]=e_temp;
				float m_temp = M[rid];
				M[rid]=M[rid+1];
				M[rid+1]=m_temp;
			}
		}
	}
}

void therm_steps_w(int *s, float *JJJ, float *ran, float *T)
{
	metropolis <<< N_R, BLOCK_SIZE >>> (s, JJJ, ran, T);
}

void prod_steps_w(int *s, float *JJJ, float *ran, float *T, float *E, int *out, float *M)
{
	metropolis <<< N_R, BLOCK_SIZE >>> (s, JJJ, ran, T);
	calc_energy <<< N_R, BLOCK_SIZE >>> (s, JJJ, E);
	calc_mag <<< N_R, BLOCK_SIZE >>> (s, out, M);
}

void pt_w(int *s, float *E, float *M, float *T, float *ran)
{
	pt <<< N_R/2, BLOCK_SIZE >>> (s, E, M, T, ran, 0);
	pt <<< N_R/2, BLOCK_SIZE >>> (s, E, M, T, ran, 1);
}
