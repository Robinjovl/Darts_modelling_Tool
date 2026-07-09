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
    void linsolv_cusparse_ilu<N_BLOCK_SIZE>::free_device_resources() noexcept
    {
      if (!initialized_)
        return;
      if (single_precision)
      {
        cudaFree(ilu_rhs);
        cudaFree(ilu_sol);
        cudaFree(d_z_sfp);
        cudaFree(values_d_ilu_sfp);
        ilu_rhs = ilu_sol = d_z_sfp = values_d_ilu_sfp = nullptr;
      }
      else
      {
        cudaFree(d_z);
        d_z = nullptr;
        if (!factorize_in_place)
          cudaFree(values_d_ilu);
        values_d_ilu = nullptr;
      }

      cudaFree(pBuffer);
      pBuffer = nullptr;
      if (pBuffer_t)
        cudaFree(pBuffer_t);
      pBuffer_t = nullptr;
      cusparseDestroyMatDescr(descr_M);
      cusparseDestroyMatDescr(descr_L);
      cusparseDestroyMatDescr(descr_U);
      descr_M = descr_L = descr_U = 0;
      cusparseDestroyBsrilu02Info(info_M);
      cusparseDestroyBsrsv2Info(info_L);
      cusparseDestroyBsrsv2Info(info_U);
      info_M = 0;
      info_L = info_U = 0;
      if (info_Lt)
        cusparseDestroyBsrsv2Info(info_Lt);
      if (info_Ut)
        cusparseDestroyBsrsv2Info(info_Ut);
      info_Lt = info_Ut = 0;
      transposed_analysis_done_ = false;
      if (owns_handle_ && handle)
        cusparseDestroy(handle);
      handle = nullptr;
      owns_handle_ = false;
      initialized_ = false;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cusparse_ilu<N_BLOCK_SIZE>::~linsolv_cusparse_ilu()
    {
      free_device_resources();
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusparse_ilu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
      opendarts::config::index_t /*max_iters*/,
      opendarts::config::mat_float /*tolerance*/)
    {
      cudaError_t cudaStat;

      // Re-init guard: the adjoint backward driver re-inits its solver chain
      // once per gradient evaluation, so a reused cusparse_ilu object must not
      // leak the previous device state. Free the old resources and do a FULL
      // fresh init (destroy-then-recreate, the same pattern as linsolv_amgx).
      // NB: a "skip if the structure is unchanged" shortcut is NOT safe here --
      // the cached device row/col pointers and the bsrsv2 analyses go stale
      // across the forward run and re-fetching them is exactly what the fresh
      // init does. These buffers are solver-internal (nothing external holds a
      // reference), so freeing them dangles nothing.
      if (initialized_)
        free_device_resources();

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
      initialized_ = true;
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

      // Query the zero-pivot report: a structural/numerical zero pivot means
      // the triangular factors contain garbage (previously never checked --
      // a singular diagonal block produced a silent NaN preconditioner).
      // Report failure so the caller (CPR-GPU / BiCGStab -> engine) can cut
      // the timestep.
      {
        int pivot_row = -1;
        cusparseStatus_t piv_stat =
            cusparseXbsrilu02_zeroPivot(handle, info_M, &pivot_row);
        if (piv_stat == CUSPARSE_STATUS_ZERO_PIVOT)
        {
          numerical_zero = pivot_row;
          printf("cusparse ILU(0): zero pivot at block row %d -- "
                 "factorisation unusable\n", pivot_row);
          this->timer_setup->node["ILU(0)"].stop();
          return -1;
        }
      }

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

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cusparse_ilu<N_BLOCK_SIZE>::solve_transposed(opendarts::config::mat_float *cur_rhs,
      opendarts::config::mat_float *sol)
    {
      // (L U)^T = U^T L^T: apply U^{-T} first, then L^{-T}, with transposed
      // bsrsv2 solves on the SAME factor values as the forward apply. The
      // analysis is sparsity-only, so it is created once and reused across
      // setups (the solves read the current values_d_ilu).
      if (single_precision)
      {
        printf("linsolv_cusparse_ilu::solve_transposed: single-precision mode "
               "is not supported on the adjoint path\n");
        return -1;
      }

      this->timer_solve->node["ILU(0)"].start();

      if (!transposed_analysis_done_)
      {
        cusparseCreateBsrsv2Info(&info_Lt);
        cusparseCreateBsrsv2Info(&info_Ut);

        int buf_Lt = 0, buf_Ut = 0;
        cusparseDbsrsv2_bufferSize(handle, dir, CUSPARSE_OPERATION_TRANSPOSE, mb, nnzb,
          descr_L, values_d_ilu, d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_Lt, &buf_Lt);
        cusparseDbsrsv2_bufferSize(handle, dir, CUSPARSE_OPERATION_TRANSPOSE, mb, nnzb,
          descr_U, values_d_ilu, d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_Ut, &buf_Ut);

        if (cudaMalloc(&pBuffer_t, static_cast<std::size_t>(buf_Lt) + buf_Ut) != cudaSuccess)
        {
          printf("Error! Can't allocate device memory (linsolv_cusparse_ilu, transposed)\n");
          this->timer_solve->node["ILU(0)"].stop();
          return -1;
        }
        pBufferLt = pBuffer_t;
        pBufferUt = (char *)pBufferLt + buf_Lt;

        cusparseDbsrsv2_analysis(handle, dir, CUSPARSE_OPERATION_TRANSPOSE, mb, nnzb,
          descr_L, values_d_ilu, d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_Lt, policy_L, pBufferLt);
        cusparseDbsrsv2_analysis(handle, dir, CUSPARSE_OPERATION_TRANSPOSE, mb, nnzb,
          descr_U, values_d_ilu, d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_Ut, policy_U, pBufferUt);

        transposed_analysis_done_ = true;
      }

      // step 1: solve U^T * z = rhs.
      cusparseDbsrsv2_solve(handle, dir, CUSPARSE_OPERATION_TRANSPOSE, mb, nnzb, &alpha,
        descr_U, values_d_ilu, d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_Ut,
        cur_rhs, d_z, policy_U, pBufferUt);

      // step 2: solve L^T * x = z.
      cusparseDbsrsv2_solve(handle, dir, CUSPARSE_OPERATION_TRANSPOSE, mb, nnzb, &alpha,
        descr_L, values_d_ilu, d_bsrRowPtr, d_bsrColInd, N_BLOCK_SIZE, info_Lt,
        d_z, sol, policy_L, pBufferLt);

      this->timer_solve->node["ILU(0)"].stop();
      return 0;
    }

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see solvers/src/CMakeLists.txt.
    // Edit the block-size range there, not here.
#include "linsolv_cusparse_ilu_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts

#pragma GCC diagnostic pop

#endif // WITH_GPU
