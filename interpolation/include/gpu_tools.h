#ifndef INTERPOLATION_GPU_TOOLS_H
#define INTERPOLATION_GPU_TOOLS_H

#include <iostream>
#include <cuda_runtime.h>

// ── Kernel launch helpers ────────────────────────────────────────────────────

#define KERNEL_1D_THREAD(n_threads_total, n_threads_block) \
<<<(n_threads_total + n_threads_block - 1) / n_threads_block, n_threads_block>>>

#define KERNEL_1D_THREAD_STREAM(n_threads_total, n_threads_block, stream) \
<<<(n_threads_total + n_threads_block - 1) / n_threads_block, n_threads_block, 0, stream>>>

// ── CUDA error checking ─────────────────────────────────────────────────────

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
  exit(1);
}

// ── Occupancy-based block size selection ─────────────────────────────────────

template <class T>
__host__ int get_kernel_thread_block_size(T kernel, int &min_job_size, size_t dynamic_shared_memory_size = 0, int block_size_limit = 0)
{
  int min_grid_size;
  int kernel_block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, kernel, dynamic_shared_memory_size, block_size_limit);
  min_job_size = min_grid_size * kernel_block_size - kernel_block_size + 1;
  return kernel_block_size;
}

// ── VSCode IntelliSense workaround ──────────────────────────────────────────

#ifdef __INTELLISENSE__
#define __global__
#define __constant__
#endif

#endif /* INTERPOLATION_GPU_TOOLS_H */
