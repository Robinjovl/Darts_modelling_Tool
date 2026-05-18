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

      //////////////////////
      // linear_solver_base
      //////////////////////

      int solve(opendarts::linear_solvers::csr_matrix_base * /*matrix*/,
        opendarts::config::mat_float *v,
        opendarts::config::mat_float *r) override
      {
        return solve(v, r);
      }

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override
      {
        return this->setup(static_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A));
      }

      //////////////////////
      // linsolv_iface
      //////////////////////

      int set_prec(opendarts::linear_solvers::linsolv_iface * /*prec_input*/) override
      {
        return 0;
      }

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        int max_iters,
        double tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override;

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override
      {
        return 1;
      }

      opendarts::config::mat_float get_residual() override
      {
        return 0;
      }

    private:
      // Factorise into the matrix values themselves rather than a copy.
      int factorize_in_place;

      // Run the ILU factorisation and solves in single precision.
      int single_precision;

      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_matrix = nullptr;

      double *values_d_ilu;

      // Single-precision ILU storage.
      float *values_d_ilu_sfp, *ilu_rhs, *ilu_sol, *d_z_sfp;

      // cuSPARSE state.
      cusparseHandle_t handle;
      cusparseMatDescr_t descr_M = 0;
      cusparseMatDescr_t descr_L = 0;
      cusparseMatDescr_t descr_U = 0;
      bsrilu02Info_t info_M = 0;
      bsrsv2Info_t info_L = 0;
      bsrsv2Info_t info_U = 0;
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
