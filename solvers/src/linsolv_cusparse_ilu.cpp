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

#include <cuda_runtime.h>

#include "linsolv_cusparse_ilu.hpp"
#include "csr_matrix.hpp"
#include "gpu_tools.hpp"

// CUDA 12+ deprecates the legacy block-CSR cuSPARSE routines (bsrilu02,
// bsrsv2, ...). They remain functional and have no drop-in generic-API
// replacement, so the deprecation diagnostic is silenced for this wrapper;
// migrating to the generic cuSPARSE API is tracked separately.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    linsolv_cusparse_ilu<N_BLOCK_SIZE>::linsolv_cusparse_ilu(int factorize_in_place_input, int single_precision_input)
      : factorize_in_place(factorize_in_place_input), single_precision(single_precision_input)
    {
      // Register as the linear_solver_base behind the BOS interface.
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::solver = this;
      values_d_ilu = nullptr;
      values_d_ilu_sfp = ilu_rhs = ilu_sol = d_z_sfp = nullptr;
      d_x = d_y = d_z = nullptr;
      d_bsrRowPtr = d_bsrColInd = nullptr;
      mb = nnzb = 0;
      pBufferSize_M = pBufferSize_L = pBufferSize_U = pBufferSize = 0;
      structural_zero = numerical_zero = 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cusparse_ilu<N_BLOCK_SIZE>::~linsolv_cusparse_ilu()
    {
      if (single_precision)
      {
        cudaFree(ilu_rhs);
        cudaFree(ilu_sol);
        cudaFree(d_z_sfp);
        cudaFree(values_d_ilu_sfp);
      }
      else
      {
        cudaFree(d_z);
        if (!factorize_in_place)
          cudaFree(values_d_ilu);
      }

      cudaFree(pBuffer);
      cusparseDestroyMatDescr(descr_M);
      cusparseDestroyMatDescr(descr_L);
      cusparseDestroyMatDescr(descr_U);
      cusparseDestroyBsrilu02Info(info_M);
      cusparseDestroyBsrsv2Info(info_L);
      cusparseDestroyBsrsv2Info(info_U);
      if (owns_handle_ && handle)
        cusparseDestroy(handle);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusparse_ilu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
      opendarts::config::index_t /*max_iters*/,
      opendarts::config::mat_float /*tolerance*/)
    {
      cudaError_t cudaStat;

      A_matrix = A_input;
      // Try to share the cuSPARSE handle the matrix already owns (legacy
      // csr_matrix<N> creates one per matrix); if the matrix is a
      // block_csr_matrix (or any csr_matrix_base subclass without a
      // matrix-owned handle), create our own. The owned handle is destroyed
      // in the destructor.
      cusparseHandle_t matrix_handle = nullptr;
      if (auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input))
        matrix_handle = A_typed->cus_handle;
      if (matrix_handle != nullptr)
      {
        handle = matrix_handle;
        owns_handle_ = false;
      }
      else
      {
        if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS)
        {
          printf("Error! cusparseCreate failed (linsolv_cusparse_ilu)\n");
          return -1;
        }
        owns_handle_ = true;
      }

      mb = A_matrix->n_rows;
      nnzb = A_matrix->n_non_zeros;
      d_bsrRowPtr = A_matrix->get_rows_ptr_d();
      d_bsrColInd = A_matrix->get_cols_ind_d();

      if (single_precision)
      {
        // Single-precision ILU storage on device.
        cudaStat = cudaMalloc((void **)&values_d_ilu_sfp, sizeof(float) * nnzb * N_BLOCK_SIZE * N_BLOCK_SIZE);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory (linsolv_cusparse_ilu)\n");
          return -1;
        }
        cudaStat = cudaMalloc((void **)&ilu_rhs, sizeof(float) * A_matrix->n_rows * N_BLOCK_SIZE);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory (linsolv_cusparse_ilu)\n");
          return -1;
        }
        cudaStat = cudaMalloc((void **)&ilu_sol, sizeof(float) * A_matrix->n_rows * N_BLOCK_SIZE);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory (linsolv_cusparse_ilu)\n");
          return -1;
        }
        cudaStat = cudaMalloc((void **)&d_z_sfp, sizeof(float) * A_matrix->n_rows * N_BLOCK_SIZE);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory (linsolv_cusparse_ilu)\n");
          return -1;
        }
      }
      else
      {
        if (factorize_in_place)
        {
          // Matrix-free mode: factorise into the matrix values themselves.
          values_d_ilu = A_matrix->get_values_d();
          printf("ILU(0): matrix-free mode\n");
        }
        else
        {
          // Double-precision ILU storage on device.
          cudaStat = cudaMalloc((void **)&values_d_ilu, sizeof(double) * nnzb * N_BLOCK_SIZE * N_BLOCK_SIZE);
          if (cudaStat != cudaSuccess)
          {
            printf("Error! Can't allocate device memory (linsolv_cusparse_ilu)\n");
            return -1;
          }
        }
        cudaStat = cudaMalloc((void **)&d_z, sizeof(double) * A_matrix->n_rows * N_BLOCK_SIZE);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory (linsolv_cusparse_ilu)\n");
          return -1;
        }
      }

      // step 1: matrix descriptors for M (= L*U), the unit-lower L and the
      // non-unit-upper U, all base-0.
      cusparseCreateMatDescr(&descr_M);
      cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

      cusparseCreateMatDescr(&descr_L);
      cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
      cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

      cusparseCreateMatDescr(&descr_U);
      cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
      cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

      // step 2: one info structure for bsrilu02 and two for bsrsv2.
      cusparseCreateBsrilu02Info(&info_M);
      cusparseCreateBsrsv2Info(&info_L);
      cusparseCreateBsrsv2Info(&info_U);

      // step 3: query and allocate the work buffer.
      if (single_precision)
      {
        cusparseSbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_M, &pBufferSize_M);
        cusparseSbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_L, &pBufferSize_L);
        cusparseSbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_U, &pBufferSize_U);
      }
      else
      {
        cusparseDbsrilu02_bufferSize(handle, dir, mb, nnzb, descr_M, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_M, &pBufferSize_M);
        cusparseDbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb, descr_L, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_L, &pBufferSize_L);
        cusparseDbsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb, descr_U, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_U, &pBufferSize_U);
      }

      pBufferSize = pBufferSize_M + pBufferSize_L + pBufferSize_U;

      cudaStat = cudaMalloc((void **)&pBuffer, pBufferSize);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory (linsolv_cusparse_ilu)\n");
        return -1;
      }

      pBufferM = pBuffer;
      pBufferL = (char *)pBufferM + pBufferSize_M;
      pBufferU = (char *)pBufferL + pBufferSize_L;

      // step 4: analyse the factorisation and both triangular solves.
      if (single_precision)
      {
        cusparseSbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_M, policy_M, pBufferM);
        cusparseSbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_L, policy_L, pBufferL);
        cusparseSbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_U, policy_U, pBufferU);
      }
      else
      {
        cusparseDbsrilu02_analysis(handle, dir, mb, nnzb, descr_M, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_M, policy_M, pBufferM);
        cusparseDbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_L, policy_L, pBufferL);
        cusparseDbsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_U, policy_U, pBufferU);
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusparse_ilu<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *matrix)
    {
      cudaError_t cudaStat;

      this->timer_setup->node["ILU(0)"].start();

      // May still be uninitialised when invoked through the BOS interface.
      if (!A_matrix)
        init(matrix, 0, 1.0);

      if (single_precision)
      {
        opendarts::linear_solvers::copy_device_data(values_d_ilu_sfp, A_matrix->get_values_d(),
          nnzb * N_BLOCK_SIZE * N_BLOCK_SIZE);
      }
      else
      {
        if (!factorize_in_place)
        {
          // Not matrix-free: copy the matrix values into the ILU storage first.
          cudaStat = cudaMemcpy(values_d_ilu, A_matrix->get_values_d(),
            sizeof(double) * nnzb * N_BLOCK_SIZE * N_BLOCK_SIZE, cudaMemcpyDeviceToDevice);
          if (cudaStat != cudaSuccess)
          {
            printf("Error! Can't copy memory within device\n");
            return -1;
          }
        }
      }

      // step 5: M = L * U.
      if (single_precision)
        cusparseSbsrilu02(handle, dir, mb, nnzb, descr_M, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_M, policy_M, pBufferM);
      else
        cusparseDbsrilu02(handle, dir, mb, nnzb, descr_M, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_M, policy_M, pBufferM);

      this->timer_setup->node["ILU(0)"].stop();
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusparse_ilu<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *cur_rhs,
      opendarts::config::mat_float *sol)
    {
      this->timer_solve->node["ILU(0)"].start();

      if (single_precision)
      {
        this->timer_solve->node["ILU(0)"].node["double <-> float"].start();
        opendarts::linear_solvers::copy_device_data(ilu_rhs, cur_rhs, mb * N_BLOCK_SIZE);
        this->timer_solve->node["ILU(0)"].node["double <-> float"].stop();

        // step 6: solve L*z = x.
        cusparseSbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha_sfp, descr_L, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_L, ilu_rhs, d_z_sfp, policy_L, pBufferL);

        // step 7: solve U*y = z.
        cusparseSbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha_sfp, descr_U, values_d_ilu_sfp,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_U, d_z_sfp, ilu_sol, policy_U, pBufferU);

        this->timer_solve->node["ILU(0)"].node["double <-> float"].start();
        opendarts::linear_solvers::copy_device_data(sol, ilu_sol, mb * N_BLOCK_SIZE);
        this->timer_solve->node["ILU(0)"].node["double <-> float"].stop();
      }
      else
      {
        // step 6: solve L*z = x.
        cusparseDbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_L, cur_rhs, d_z, policy_L, pBufferL);

        // step 7: solve U*y = z.
        cusparseDbsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U, values_d_ilu,
          d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_U, d_z, sol, policy_U, pBufferU);
      }

      this->timer_solve->node["ILU(0)"].stop();
      return 0;
    }

    // Explicit instantiation for the supported block sizes.
    template class linsolv_cusparse_ilu<1>;
    template class linsolv_cusparse_ilu<2>;
    template class linsolv_cusparse_ilu<3>;
    template class linsolv_cusparse_ilu<4>;
    template class linsolv_cusparse_ilu<5>;
    template class linsolv_cusparse_ilu<6>;
    template class linsolv_cusparse_ilu<7>;
    template class linsolv_cusparse_ilu<8>;
    template class linsolv_cusparse_ilu<9>;
    template class linsolv_cusparse_ilu<10>;
    template class linsolv_cusparse_ilu<11>;
    template class linsolv_cusparse_ilu<12>;
    template class linsolv_cusparse_ilu<13>;
  } // namespace linear_solvers
} // namespace opendarts

#pragma GCC diagnostic pop

#endif // WITH_GPU
