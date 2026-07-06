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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_BICGSTAB_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_BICGSTAB_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

#include <cublas_v2.h>

#include "data_types.hpp"
#include "csr_matrix_base.hpp"
#include "linsolv_iface.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Preconditioned BiCGStab Krylov solver running on the GPU.

        A right-preconditioned BiCGStab iteration; all vector algebra runs on
        the device through cuBLAS and the block sparse matrix-vector products
        go through the csr_matrix GPU device layer (see csr_matrix.hpp). The
        preconditioner is any linsolv_iface whose solve() operates on device
        pointers (e.g. cuSPARSE ILU(0)).

        Ported from the proprietary darts-linear-solvers linsolv_bicgstab.
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_bicgstab : public opendarts::linear_solvers::linsolv_iface
    {
    public:
      linsolv_bicgstab();

      ~linsolv_bicgstab() override;

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      int init(opendarts::linear_solvers::csr_matrix_base *A_input,
        opendarts::config::index_t max_iters_input,
        opendarts::config::mat_float tolerance_input) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A_input) override;

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override;

      opendarts::config::mat_float get_residual() override;

      /** Outcome of the last solve; converged distinguishes genuine
       *  convergence from a max_iters exit (solve() returns 0 in both cases,
       *  legacy parity; hard failures -- breakdown / NaN -- return nonzero). */
      opendarts::linear_solvers::solver_stats stats() const override
      {
        opendarts::linear_solvers::solver_stats st;
        st.iterations = n_iters;
        st.residual = final_resid;
        st.converged = last_converged;
        return st;
      }

      // Set by solve(): whether the last solve met the tolerance.
      bool last_converged = false;

      // Preconditioner applied at every iteration (not owned until set_prec).
      opendarts::linear_solvers::linsolv_iface *prec;

      // System matrix; kept as a base pointer and driven via the GPU device layer.
      opendarts::linear_solvers::csr_matrix_base *A;

      // Iterations done in the last solve and the maximum allowed.
      int n_iters, max_iters;

      // Final relative residual of the last solve and the target tolerance.
      double final_resid, tolerance;

      cublasHandle_t cub_handle;

      // Single device workspace block, sliced into the work vectors below.
      double *wksp_d;
      double *r, *rw, *p, *pw, *s, *t, *v;

      // Length of the (scalar) solution vector, n_rows * N_BLOCK_SIZE.
      int n;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_BICGSTAB_HPP
//--------------------------------------------------------------------------
