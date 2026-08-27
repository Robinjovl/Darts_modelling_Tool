//*************************************************************************
//    Copyright (c) 2026
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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_GMRES_GPU_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_GMRES_GPU_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

#include <vector>

#include <cublas_v2.h>

#include "data_types.hpp"
#include "csr_matrix_base.hpp"
#include "linsolv_iface.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Restarted right-preconditioned GMRES running on the GPU.

        The in-tree GPU counterpart of the proprietary ``linsolv_bos_gmres``
        (gpu mode): the Krylov basis lives on the device, block SpMV goes
        through the csr_matrix / block_csr_matrix GPU device layer, and the
        vector algebra runs through cuBLAS. The small Hessenberg
        least-squares problem (Givens rotations) is solved on the host.

        Semantics follow the proprietary solver: effective restart length
        m = max(1, min(max_iters, restart_requested)), with a default requested
        restart of 50; zero-vector-tolerant relative convergence
        (||r|| <= tol * ||b||, falling back to ||r0|| when ||b|| ~ 0), one
        preconditioner application + one SpMV per iteration, and a single
        preconditioner application on the combined basis update at each
        restart (standard, non-flexible right preconditioning).

        Orthogonalisation is classical Gram-Schmidt with one
        re-orthogonalisation pass (CGS2) implemented as cuBLAS GEMVs -- two
        fused matrix-vector products instead of i round-trip dot products
        per iteration, with stability at least on par with modified
        Gram-Schmidt.

        The preconditioner is any linsolv_iface whose solve() operates on
        device pointers (AMGX-CPR, cuSPARSE ILU(0), cuDSS, ...).
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_gmres_gpu : public opendarts::linear_solvers::linsolv_iface
    {
    public:
      linsolv_gmres_gpu();

      ~linsolv_gmres_gpu() override;

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      int init(opendarts::linear_solvers::csr_matrix_base *A_input,
        opendarts::config::index_t max_iters_input,
        opendarts::config::mat_float tolerance_input) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A_input) override;

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      /** Adjoint solve of A^T x = B: the forward algorithm with the
       *  transposed SpMV (A->calc_lin_comb_t_d / matrix_vector_product_t_d0)
       *  and prec->solve_transposed(). Unlike solve(), B and X are HOST
       *  pointers -- the caller is the host-side adjoint backward driver
       *  (engine_base::calc_adjoint_gradient_dirac_all); the vectors are
       *  staged to a device scratch here. X is used as the initial guess. */
      int solve_transposed(opendarts::config::mat_float *B,
        opendarts::config::mat_float *X) override;

      int get_n_iters() override;

      opendarts::config::mat_float get_residual() override;

      /** Requested restart length; clamped to [1, max_iters] at init(). */
      void set_restart(int m) { restart_requested = m; }

      // Preconditioner applied at every iteration (owned once set).
      opendarts::linear_solvers::linsolv_iface *prec;

      // System matrix; kept as a base pointer and driven via the GPU device layer.
      opendarts::linear_solvers::csr_matrix_base *A;

      // Iterations done in the last solve and the maximum allowed.
      int n_iters, max_iters;

      // Final relative residual of the last solve and the target tolerance.
      double final_resid, tolerance;

      cublasHandle_t cub_handle;

      // Device workspace: Krylov basis V ((m+1) columns of length n, column
      // major) plus w (SpMV result), z (preconditioner output) and h_d (the
      // per-iteration Hessenberg column for the GEMV-based Gram-Schmidt).
      double *V_d;
      double *w_d, *z_d, *h_d;

      // Host<->device staging for solve_transposed (B | X, 2n doubles);
      // allocated lazily on the first transposed solve.
      double *bx_t_d = nullptr;
      int bx_t_n = 0;

      // Host-side Hessenberg / Givens data ((m+1) x m, column-wise).
      std::vector<double> hh, cs, sn, rs, y;

      int restart_requested; // requested restart length (default 50)
      int m;                 // effective restart length; workspace capacity follows it

      // Length of the (scalar) solution vector, n_rows * N_BLOCK_SIZE.
      int n;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_GMRES_GPU_HPP
//--------------------------------------------------------------------------
