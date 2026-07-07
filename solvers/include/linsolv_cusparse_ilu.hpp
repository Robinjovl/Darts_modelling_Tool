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

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUSPARSE_ILU_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUSPARSE_ILU_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

#include <cusparse.h>

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief cuSPARSE block-ILU(0) preconditioner on the GPU.

        Wraps the cuSPARSE legacy block-CSR ILU(0) factorisation
        (cusparseDbsrilu02) and the two triangular solves
        (cusparseDbsrsv2_solve). Intended as the preconditioner for the GPU
        BiCGStab solver, but it also implements linear_solver_base so it can
        serve as a stage preconditioner inside the GPU CPR wrapper.

        Ported from the proprietary darts-linear-solvers linsolv_cusparse_ilu.
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_cusparse_ilu : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                                 public opendarts::linear_solvers::linear_solver_base
    {
    public:
      /** @param factorize_in_place - if set, factorise directly into the
              matrix values (matrix-free); otherwise factorise into a copy.
          @param single_precision - if set, run the ILU in single precision. */
      linsolv_cusparse_ilu(int factorize_in_place = 0, int single_precision = 0);

      ~linsolv_cusparse_ilu();

      // Keep the csr_matrix_base init()/setup() overloads visible: declaring
      // the csr_matrix<N>* overloads below otherwise hides them by name.
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      //////////////////////
      // linear_solver_base
      //////////////////////

      int solve(opendarts::linear_solvers::csr_matrix_base * /*matrix*/,
        opendarts::config::mat_float *v,
        opendarts::config::mat_float *r) override
      {
        return solve(v, r);
      }

      // csr_matrix_base override -- bypasses the linsolv_iface_bos<N>
      // static_cast which is UB when A is a block_csr_matrix.
      // The linsolv_iface_bos<N> init/setup csr_matrix<N>* overrides below
      // remain for callers that still pass the typed pointer; they forward
      // here so all paths share the polymorphic implementation.
      int init(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;

      //////////////////////
      // linsolv_iface
      //////////////////////

      int set_prec(opendarts::linear_solvers::linsolv_iface * /*prec_input*/) override
      {
        return 0;
      }

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        int max_iters,
        double tolerance) override
      {
        return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input),
            static_cast<opendarts::config::index_t>(max_iters), static_cast<opendarts::config::mat_float>(tolerance));
      }

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A_input));
      }

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      /** Apply the transpose of the ILU(0) preconditioner: X = (LU)^{-T} B =
       *  L^{-T} U^{-T} B, on the SAME factors as the forward solve (transposed
       *  bsrsv2 pair, U^T first then L^T). Device pointers, double precision
       *  only (the adjoint stack runs in double); single_precision mode
       *  returns -1. The transposed triangular-solve analyses are created
       *  lazily on the first call (sparsity-fixed, reused across setups). */
      int solve_transposed(opendarts::config::mat_float *B,
        opendarts::config::mat_float *X) override;

      int get_n_iters() override
      {
        return 1;
      }

      opendarts::config::mat_float get_residual() override
      {
        return 0;
      }

    private:
      // Release all device buffers + cuSPARSE descriptor/info handles. Shared
      // by the destructor and by init() (free-before-realloc), so a solver
      // reused across repeated init() -- as the adjoint backward driver does,
      // once per gradient evaluation -- does not leak the previous allocation.
      void free_device_resources() noexcept;

      // True once init() has allocated device state; gates the re-init guard.
      bool initialized_ = false;

      // Factorise into the matrix values themselves rather than a copy.
      int factorize_in_place;

      // Run the ILU factorisation and solves in single precision.
      int single_precision;

      // Polymorphic matrix pointer -- accepts both the legacy csr_matrix<N>
      // and the unified block_csr_matrix Jacobian. All access goes through
      // csr_matrix_base accessors (get_values_d, get_rows_ptr_d, ...).
      opendarts::linear_solvers::csr_matrix_base *A_matrix = nullptr;

      double *values_d_ilu;

      // Single-precision ILU storage.
      float *values_d_ilu_sfp, *ilu_rhs, *ilu_sol, *d_z_sfp;

      // cuSPARSE state. Owned by this solver so the lifetime is independent
      // of the matrix (csr_matrix<N> historically created its own per-matrix
      // handle; block_csr_matrix routes through gpu_bsr_spmv for SpMV but
      // does not expose a shareable handle for ILU/triangular solves).
      cusparseHandle_t handle = nullptr;
      bool owns_handle_ = false;
      cusparseMatDescr_t descr_M = 0;
      cusparseMatDescr_t descr_L = 0;
      cusparseMatDescr_t descr_U = 0;
      // CUDA 12+ deprecates the legacy block-CSR ILU/triangular-solve info
      // handles and solve-policy enum. They remain functional and have no
      // drop-in generic-API replacement, so the deprecation diagnostic is
      // silenced for these member declarations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      bsrilu02Info_t info_M = 0;
      bsrsv2Info_t info_L = 0;
      bsrsv2Info_t info_U = 0;
      // Transposed triangular-solve state (adjoint path); created lazily on
      // the first solve_transposed() -- forward-only runs never pay for it.
      bsrsv2Info_t info_Lt = 0;
      bsrsv2Info_t info_Ut = 0;
      void *pBuffer_t = 0;      // work buffer for both transposed solves
      void *pBufferLt = 0, *pBufferUt = 0;
      bool transposed_analysis_done_ = false;
      int mb;
      int nnzb;
      int *d_bsrRowPtr;
      int *d_bsrColInd;

      int pBufferSize_M;
      int pBufferSize_L;
      int pBufferSize_U;
      int pBufferSize;
      void *pBuffer = 0;
      void *pBufferM, *pBufferL, *pBufferU = 0;
      double *d_x, *d_y, *d_z;
      int structural_zero;
      int numerical_zero;
      const double alpha = 1.;
      const float alpha_sfp = 1.;
      const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
      const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
      const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
#pragma GCC diagnostic pop
      const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
      const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;
      const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUSPARSE_ILU_HPP
//--------------------------------------------------------------------------
