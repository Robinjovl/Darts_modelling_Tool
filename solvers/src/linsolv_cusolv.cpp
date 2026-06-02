//*************************************************************************
//    Copyright (c) 2022
//    Delft University of Technology, the Netherlands
//    Netherlands eScience Center
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// *************************************************************************

#ifdef WITH_GPU

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "linsolv_cusolv.hpp"
#include "block_csr_matrix.hpp"
#include "csr_matrix.hpp"

// CUDA 12+ deprecates cusolverSp dense/sparse QR entry points in favour of
// cuDSS. The legacy routine remains functional and cuDSS is not a build
// dependency here, so the deprecation diagnostic is silenced for this wrapper.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      // Pull the scalar-CSR device triple out of either matrix subclass.
      // For block_csr_matrix this requires that build_scalar_csr_device()
      // was called this Newton iteration (init/setup do that).
      template <uint8_t N_BLOCK_SIZE>
      bool fetch_scalar_csr_device(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t &n_scalar_rows,
        opendarts::config::index_t &n_scalar_nnz,
        const double *&vals_d,
        const opendarts::config::index_t *&row_ptr_d,
        const opendarts::config::index_t *&col_ind_d)
      {
        if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A))
        {
          n_scalar_rows = A_typed->n_rows * N_BLOCK_SIZE;
          n_scalar_nnz = A_typed->get_n_non_zeros() * N_BLOCK_SIZE * N_BLOCK_SIZE;
          if (N_BLOCK_SIZE > 1)
          {
            vals_d = A_typed->csrValC;
            row_ptr_d = A_typed->csrRowPtrC;
            col_ind_d = A_typed->csrColIndC;
          }
          else
          {
            vals_d = A_typed->values_d;
            row_ptr_d = A_typed->rows_ptr_d;
            col_ind_d = A_typed->cols_ind_d;
          }
          return true;
        }
        if (auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A))
        {
          n_scalar_rows = A_block->n_rows * A_block->block_size();
          n_scalar_nnz = A_block->scalar_csr_nnz();
          vals_d = A_block->scalar_csr_values_device();
          row_ptr_d = A_block->scalar_csr_row_ptr_device();
          col_ind_d = A_block->scalar_csr_col_ind_device();
          return true;
        }
        return false;
      }
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cusolv<N_BLOCK_SIZE>::linsolv_cusolv()
    {
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::solver = this;
      tol = 1.e-12;
      device_num = 0;
      n_rows = nnz = 0;
      single_precision = 0;
      d_B = d_X = d_Z = nullptr;
      h_Q = d_Q = nullptr;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cusolv<N_BLOCK_SIZE>::~linsolv_cusolv()
    {
      delete[] h_Q;

      cudaFree(d_B);
      cudaFree(d_X);
      cudaFree(d_Q);
      cudaFree(d_Z);

      cusolverSpDestroy(handle);
      cusparseDestroyMatDescr(descr);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusolv<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
      opendarts::config::index_t /*max_iters*/,
      opendarts::config::mat_float /*tolerance*/)
    {
      A_matrix = A_input;
      n_rows = A_matrix->n_rows;
      nnz = A_matrix->n_non_zeros;

      // Mirror the matrix on the device and, for block matrices, build the
      // scalar-CSR device view that cuSOLVER QR consumes. Dispatches between
      // the legacy csr_matrix<N> path (init_device + convert_to_ELL) and the
      // unified block_csr_matrix path (build_scalar_csr_device, via
      // gpu_bsr_spmv's cusparseDbsr2csr).
      if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input))
      {
        A_typed->init_device(n_rows, nnz);
        A_typed->copy_struct_to_device();
        if (N_BLOCK_SIZE > 1)
          A_typed->convert_to_ELL();
      }
      else if (auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input))
      {
        if (A_block->build_scalar_csr_device() != 0)
          return -1;
      }
      else
      {
        fprintf(stderr, "linsolv_cusolv: unsupported csr_matrix_base subclass\n");
        return -1;
      }

      cudaError_t cudaStat;
      cudaStat = cudaMalloc((void **)&d_B, sizeof(opendarts::config::mat_float) * n_rows * N_BLOCK_SIZE);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory (linsolv_cusolv)\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&d_Z, sizeof(opendarts::config::mat_float) * n_rows * N_BLOCK_SIZE);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory (linsolv_cusolv)\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&d_X, sizeof(opendarts::config::mat_float) * n_rows * N_BLOCK_SIZE);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory (linsolv_cusolv)\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&d_Q, sizeof(int) * n_rows * N_BLOCK_SIZE);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory (linsolv_cusolv)\n");
        return -1;
      }

      // Identity permutation (sparse -> dense vector mapping).
      h_Q = (int *)malloc(sizeof(int) * n_rows * N_BLOCK_SIZE);
      for (int i = 0; i < n_rows * N_BLOCK_SIZE; i++)
        h_Q[i] = i;

      cudaStat = cudaMemcpy(d_Q, h_Q, sizeof(int) * n_rows * N_BLOCK_SIZE, cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy memory to device (linsolv_cusolv)\n");
        return -1;
      }

      // step 1: cuSOLVER / cuSPARSE handles and the matrix descriptor.
      cusolverStatus_t cusolverStat = cusolverSpCreate(&handle);
      if (cusolverStat != CUSOLVER_STATUS_SUCCESS)
      {
        printf("Error! Can't create cusolver handle (linsolv_cusolv)\n");
        return -1;
      }
      // For the legacy csr_matrix<N> share its handle; for block_csr_matrix
      // each call to gpu_bsr_spmv owns its own handle, so a per-solver
      // descriptor is sufficient (no shared cuSPARSE handle needed).
      if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input))
        cusparseHandle = A_typed->cus_handle;
      else
        cusparseHandle = nullptr;

      cusparseCreateMatDescr(&descr);
      cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseStatus_t cusparseStat = cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
      if (cusparseStat != CUSPARSE_STATUS_SUCCESS)
      {
        printf("Error! Can't create cusparse descriptor (linsolv_cusolv)\n");
        return -1;
      }

      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusolv<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A_input)
    {
      this->timer_setup->node["CUSOLVER"].start();

      A_matrix = A_input;

      // Refresh the scalar-CSR device view from the (re-assembled) matrix.
      if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input))
      {
        if (N_BLOCK_SIZE > 1)
          A_typed->convert_to_ELL();
      }
      else if (auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input))
      {
        if (A_block->build_scalar_csr_device() != 0)
        {
          this->timer_setup->node["CUSOLVER"].stop();
          return -1;
        }
      }
      else
      {
        fprintf(stderr, "linsolv_cusolv: unsupported csr_matrix_base subclass\n");
        this->timer_setup->node["CUSOLVER"].stop();
        return -1;
      }

      cudaDeviceSynchronize();

      this->timer_setup->node["CUSOLVER"].stop();
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusolv<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      this->timer_solve->node["CUSOLVER"].start();

      n_rows = N_BLOCK_SIZE * A_matrix->n_rows;
      nnz = N_BLOCK_SIZE * N_BLOCK_SIZE * A_matrix->n_non_zeros;

      cudaError_t cudaStat;
      cusolverStatus_t cusolvStat;

      cudaStat = cudaMemcpy(d_B, B, sizeof(double) * n_rows, cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy memory to device\n");
        return -1;
      }

      // Pull the scalar-CSR device pointers from whichever backend is bound.
      opendarts::config::index_t scalar_rows = 0, scalar_nnz = 0;
      const double *vals_d = nullptr;
      const opendarts::config::index_t *row_ptr_d = nullptr;
      const opendarts::config::index_t *col_ind_d = nullptr;
      if (!fetch_scalar_csr_device<N_BLOCK_SIZE>(A_matrix, scalar_rows, scalar_nnz,
            vals_d, row_ptr_d, col_ind_d))
      {
        fprintf(stderr, "linsolv_cusolv: scalar-CSR device view unavailable\n");
        return -1;
      }

      cusolvStat = cusolverSpDcsrlsvqr(handle, scalar_rows, scalar_nnz, descr,
        vals_d, row_ptr_d, col_ind_d,
        d_B, tol, reorder, d_Z, &singularity);
      if (cusolvStat != CUSOLVER_STATUS_SUCCESS)
      {
        printf("Error! cusolverSpDcsrlsvqr failed\n");
        return -1;
      }

      if (0 <= singularity)
        printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);

      cudaDeviceSynchronize();
      cudaStat = cudaMemcpy(X, d_Z, sizeof(opendarts::config::mat_float) * n_rows, cudaMemcpyDeviceToHost);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy memory to host\n");
        return -1;
      }

      this->timer_solve->node["CUSOLVER"].stop();
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusolv<N_BLOCK_SIZE>::get_n_iters()
    {
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_cusolv<N_BLOCK_SIZE>::get_residual()
    {
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusolv<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface * /*prec_input*/)
    {
      return 0;
    }

    // Explicit instantiation for the supported block sizes.
    template class linsolv_cusolv<1>;
    template class linsolv_cusolv<2>;
    template class linsolv_cusolv<3>;
    template class linsolv_cusolv<4>;
    template class linsolv_cusolv<5>;
    template class linsolv_cusolv<6>;
    template class linsolv_cusolv<7>;
    template class linsolv_cusolv<8>;
    template class linsolv_cusolv<9>;
    template class linsolv_cusolv<10>;
    template class linsolv_cusolv<11>;
    template class linsolv_cusolv<12>;
    template class linsolv_cusolv<13>;

  } // namespace linear_solvers
} // namespace opendarts

#pragma GCC diagnostic pop

#endif // WITH_GPU
