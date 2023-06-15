//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#ifndef E47AA730_C763_46D6_8FE1_579846DFE543
#define E47AA730_C763_46D6_8FE1_579846DFE543

#include <vector>
#include <string>
#include <iomanip>
#include <thrust/device_vector.h>

#include "globals.h"

#define KERNEL_1D_THREAD(n_threads_total, n_threads_block) \
<<<(n_threads_total + n_threads_block - 1) / n_threads_block, n_threads_block>>>

#define KERNEL_1D_THREAD_STREAM(n_threads_total, n_threads_block, stream) \
<<<(n_threads_total + n_threads_block - 1) / n_threads_block, n_threads_block, 0, stream>>>

#define KERNEL_1D_WARP(n_rows_total, n_threads_block) \
<<<(n_rows_total * 32) / n_threads_block, n_threads_block>>>

#define KERNEL_1D(n_rows_total, n_threads_per_row, n_threads_block) \
<<<(n_rows_total * n_threads_per_row + n_threads_block - 1) / n_threads_block, n_threads_block>>>

// workaround for vscode grammar checker
#ifdef __INTELLISENSE__
#define __global__
#define __constant__
#endif

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
  exit(1);
}

template <typename T>
void allocate_device_data(std::vector<T> &host_data, T **device_data)
{
  CUDA_CHECK_RETURN(cudaMalloc((void **)device_data, sizeof(T) * host_data.size()));
};

template <typename T>
void allocate_device_data(T **device_data, int data_size)
{
  CUDA_CHECK_RETURN(cudaMalloc((void **)device_data, sizeof(T) * data_size));
};

template <typename T>
void free_device_data(T *device_data)
{
  CUDA_CHECK_RETURN(cudaFree(device_data));
};

static void print_gpu_mem_usage()
{
  double free_gb, total_gb;

  size_t free_t, total_t;

  cudaMemGetInfo(&free_t, &total_t);

  free_gb = (double)free_t / (1024 * 1024 * 1024);
  total_gb = (double)total_t / (1024 * 1024 * 1024);

  printf("GPU MEM INFO: used: %lf: Gb, total %lf\n", total_gb - free_gb, total_gb);
};

template <typename T>
void copy_data_to_device(std::vector<T> &host_data, T *device_data)
{
  CUDA_CHECK_RETURN(cudaMemcpy(device_data, &host_data[0], sizeof(T) * host_data.size(), cudaMemcpyHostToDevice));
};

template <typename T>
void copy_data_to_host(std::vector<T> &host_data, T *device_data, int data_size = 0)
{
  if (data_size)
    host_data.resize(data_size);
  CUDA_CHECK_RETURN(cudaMemcpy(&host_data[0], device_data, sizeof(T) * host_data.size(), cudaMemcpyDeviceToHost));
};

template <typename T>
void copy_data_to_host(T *host_data, T *device_data, int data_size)
{
  CUDA_CHECK_RETURN(cudaMemcpy(host_data, device_data, sizeof(T) * data_size, cudaMemcpyDeviceToHost));
};

template <typename T>
void write_device_vector_to_file(std::string filename, std::vector<T> &host_data, T *device_data, int data_size = 0)
{
  copy_data_to_host(host_data, device_data, data_size);
  write_vector_to_file(filename, host_data);
};

template <typename T>
void copy_data_within_device(T *dest, T *src, int data_size)
{
  CUDA_CHECK_RETURN(cudaMemcpy(dest, src, sizeof(T) * data_size, cudaMemcpyDeviceToDevice));
};

template <typename T>
__host__ __device__ void print_device_vector_kernel(const T *vector_d, int n_items, char *vector_name)
{
  printf("%s:", vector_name);
  for (int i = 0; i < n_items; i++)
    printf(" [%d] %7.2f", i, vector_d[i]);
  printf("\n");
  //   std::cout << " [" << i << "] " << std::setw(7) << std::left << vector_h[i];
  // std::vector<T> vector_h(n_items);
  // thrust::copy_n(thrust::device_ptr<T>(vector_d), n_items, vector_h.begin());
  // std::cout << vector_name << ":";
  // for (int i = 0; i < n_items; i++)
  //   std::cout << " [" << i << "] " << std::setw(7) << std::left << vector_h[i];
  // std::cout << std::endl;
};

static __host__ __device__ void print_device_int_vector_kernel(const int *vector_d, int n_items, char *vector_name)
{
  printf("%s:", vector_name);
  for (int i = 0; i < n_items; i++)
    printf(" [%d] %7.2d", i, vector_d[i]);
  printf("\n");
};

template <typename T>
__host__ void print_device_vector(const T *vector_d, int n_items, std::string vector_name)
{
  std::vector<T> vector_h(n_items);
  thrust::copy_n(thrust::device_ptr<T>(vector_d), n_items, vector_h.begin());
  std::cout << vector_name << ":";
  for (int i = 0; i < n_items; i++)
    std::cout << " [" << i << "] " << std::setw(7) << std::left << vector_h[i];
  std::cout << std::endl;
};

template <class T>
__host__ int get_kernel_thread_block_size(T kernel, int &min_job_size, size_t dynamic_shared_memory_size = 0, int block_size_limit = 0)
{
  int min_grid_size;
  int kernel_block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, kernel, dynamic_shared_memory_size, block_size_limit);
  min_job_size = min_grid_size * kernel_block_size - kernel_block_size + 1;
  return kernel_block_size;
}

#endif /* E47AA730_C763_46D6_8FE1_579846DFE543 */
