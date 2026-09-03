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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_MCSGS_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_MCSGS_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

#include <vector>

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Multicolor symmetric block-Gauss-Seidel preconditioner on the GPU.

        Latency-friendly replacement for the exact block-ILU(0) second stage of
        the GPU CPR: the cuSPARSE level-scheduled triangular solves are
        dependency-bound (~363 wavefront levels on an SPE10-class grid), while a
        multicolor sweep needs only 2 x n_colors kernel launches per apply.

        Setup inverts the diagonal blocks (Gauss-Jordan with partial pivoting);
        rows whose diagonal block is numerically singular -- well-control rows
        can carry a zero dR/dz entry -- fall back to the identity. (That raw
        singularity is why the AMGX relaxation smoothers cannot be used here:
        they invert the raw diagonal and one Inf corrupts the Krylov basis.)
        The row coloring is computed once from the host sparsity (greedy,
        structurally-symmetric pattern assumed) and reused across setups.

        The apply is a fixed linear operator (forward color sweep + backward
        color sweep from a zero initial guess), so it is safe under standard
        (non-flexible) GMRES.
    */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_mcsgs : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>,
                          public opendarts::linear_solvers::linear_solver_base
    {
    public:
      linsolv_mcsgs();

      ~linsolv_mcsgs();

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

      int get_n_iters() override
      {
        return 1;
      }

      opendarts::config::mat_float get_residual() override
      {
        return 0;
      }

    private:
      void free_all();

      opendarts::linear_solvers::csr_matrix_base *A_saved = nullptr;
      opendarts::config::index_t n_rows = 0;
      int n_colors = 0;
      int n_sweeps = 1; // symmetric sweeps per apply (DARTS_MCSGS_SWEEPS)
      std::vector<opendarts::config::index_t> color_offsets; // host, [n_colors + 1]
      opendarts::config::index_t *color_rows_d = nullptr;    // [n_rows], rows grouped by color
      opendarts::config::mat_float *inv_diag_d = nullptr;    // [n_rows * N^2] inverted diagonal blocks
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_MCSGS_HPP
//--------------------------------------------------------------------------
