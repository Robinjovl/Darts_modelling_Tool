/*
 ============================================================================
 Name        : gpu_benchmark.cu
 Author      : Mark Khait
 Version     :
 Copyright   : © 2016 MARK KHAIT ALL RIGHTS RESERVED
 Description : GPU Benchmark for interpolation kernel
 ============================================================================
 */


#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

#ifdef _MSC_VER 
#include <time.h>
#endif

#include "gpu_benchmark.cu.h"
#include "2d_interpolation.cu.h"
#include "interp_table.h"
#include "globals.h"


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString (err) << "(" << err << ") at " << file << ":" << line << std::endl;
  exit (1);
}



/**
 * Host function that copies the data and launches the work on GPU
 */
interp_value_t* gpuInterpolation(interp_value_t *val1, interp_value_t *val2, unsigned size, interp_table *tbl)
{

  int nIter = 100;
  float msecTotal = 0.0f, msecPerMatrixMul;
  double flopsPerMatrixMul, gigaFlops;

  interp_value_t *gpuInterp, *gpuVal1 , *gpuVal2;
  interp_value_t *gpuRes , *gpuResDer1, *gpuResDer2;
  interp_value_t *cpuRes = new interp_value_t[4 * size];
  interp_value_t *cpuResDer1 = new interp_value_t[size];
  interp_value_t *cpuResDer2 = new interp_value_t[size];

  int ax1_points = tbl->ax1_npoints;
  interp_value_t ax1_min = tbl->ax1_min;
  interp_value_t ax1_max = tbl->ax1_max;
  interp_value_t ax1_step_inv = (ax1_points - 1) / (ax1_max - ax1_min);

  int ax2_points = tbl->ax2_npoints;
  interp_value_t ax2_min = tbl->ax2_min;
  interp_value_t ax2_max = tbl->ax2_max;
  interp_value_t ax2_step_inv = (ax2_points - 1) / (ax2_max - ax2_min);

  int interp_size = (ax1_points - 1)*(ax2_points - 1) * 4;
  interp_value_t *cpuInterp = tbl->data;

// CUDA
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuInterp, sizeof(interp_value_t)*interp_size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuInterp, cpuInterp, sizeof(interp_value_t)*interp_size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuVal1, sizeof(interp_value_t)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuVal1, val1, sizeof(interp_value_t)*size, cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuVal2, sizeof(interp_value_t)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuVal2, val2, sizeof(interp_value_t)*size, cudaMemcpyHostToDevice));
	
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuRes, sizeof(interp_value_t)*4*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuResDer1, sizeof(interp_value_t)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuResDer2, sizeof(interp_value_t)*size));


	const int blockCount = (size+INTERP_BLOCK_SIZE-1)/INTERP_BLOCK_SIZE;


	// Warmup


	bilinear_interpolation_kernel<int, interp_value_t, INTERP_BLOCK_SIZE><<<blockCount, INTERP_BLOCK_SIZE>>>
				(size, ax1_points, ax1_min, ax1_step_inv,
				 ax2_points, ax2_min, ax2_step_inv,
				 gpuInterp, gpuVal1,
				 gpuRes);


	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	cudaError_t error;

	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	

	for (int j = 0; j < nIter; j++)
	{
		bilinear_interpolation_kernel<int, interp_value_t, INTERP_BLOCK_SIZE><<<blockCount, INTERP_BLOCK_SIZE>>>
				(size, ax1_points, ax1_min, ax1_step_inv,
				 ax2_points, ax2_min, ax2_step_inv,
				 gpuInterp, gpuVal1, gpuRes);
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Compute and print the performance
	msecPerMatrixMul = msecTotal / nIter;
	flopsPerMatrixMul = 28 * size;
	gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec, Precision= %u bits, BlockSize= %u threads/block\n",
		gigaFlops,
		msecPerMatrixMul,
		sizeof(interp_value_t) * 8,
		INTERP_BLOCK_SIZE);


	CUDA_CHECK_RETURN(cudaMemcpy(cpuRes, gpuRes, sizeof(interp_value_t)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuInterp));
	CUDA_CHECK_RETURN(cudaFree(gpuVal1));
	CUDA_CHECK_RETURN(cudaFree(gpuVal2));
	CUDA_CHECK_RETURN(cudaFree(gpuRes));
	CUDA_CHECK_RETURN(cudaFree(gpuResDer1));
	CUDA_CHECK_RETURN(cudaFree(gpuResDer2));

	return cpuRes;
}
