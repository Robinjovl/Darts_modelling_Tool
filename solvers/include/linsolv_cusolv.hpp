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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUSOLV_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUSOLV_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

#include <cusolverSp.h>
#include <cusparse.h>

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Direct sparse linear solver on the GPU via cuSOLVER QR.

        Wraps cusolverSpDcsrlsvqr, the cuSOLVER sparse QR direct solve. For
        block matrices the device matrix is first expanded to scalar CSR by
        csr_matrix::convert_to_ELL. Useful as a robust fallback when iterative
        GPU solvers fail to converge.

        Ported from the proprietary darts-linear-solvers linsolv_cusolv.
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_cusolv : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                           public opendarts::linear_solvers::linear_solver_base
    {
    public:
      linsolv_cusolv();

      ~linsolv_cusolv();

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

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input,
        int max_iters,
        double tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_input) override;

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override;

      opendarts::config::mat_float get_residual() override;

      int device_num;
      int n_rows, nnz;
      int single_precision;
      opendarts::config::mat_float tol;
      const int reorder = 0;   // no reordering
      int singularity = 0;     // -1 if A is invertible under tol

    private:
      cusolverSpHandle_t handle;
      cusparseHandle_t cusparseHandle;
      cudaStream_t stream;
      cusparseMatDescr_t descr;
      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_matrix = nullptr;
      opendarts::config::mat_float *d_B, *d_X, *d_Z;
      int *h_Q, *d_Q;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_CUSOLV_HPP
//--------------------------------------------------------------------------
