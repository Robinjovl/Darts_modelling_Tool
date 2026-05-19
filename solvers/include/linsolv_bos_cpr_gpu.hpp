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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_CPR_GPU_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_CPR_GPU_HPP
//--------------------------------------------------------------------------

#ifdef WITH_GPU

#include <cstdio>

#include "data_types.hpp"
#include "csr_matrix.hpp"
#include "linear_solver_base.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Two-stage Constrained-Pressure-Residual preconditioner on the GPU.

        CPR splits the block reservoir system into a scalar pressure system
        and the full system. The pressure restriction/prolongation runs in
        custom CUDA kernels; the pressure system is solved by an injected
        pressure preconditioner (p_system_preconditioner) and the full system
        by a second injected preconditioner (full_system_preconditioner). The
        block matrix-vector product comes from the csr_matrix GPU device layer.

        Ported from the proprietary darts-linear-solvers linsolv_bos_cpr_gpu.
        Note: the diagonal-first reordering path (p_solver_requires_diag_first)
        depends on csr_matrix::set_diag_first, which is not part of the
        open-source csr_matrix; that flag is therefore forced off with a
        warning and the non-diagonal-first kernel path is always used.
    */
    template <uint8_t n_block_size>
    class linsolv_bos_cpr_gpu : public opendarts::linear_solvers::linsolv_iface_bos<n_block_size>,
                                public opendarts::linear_solvers::linear_solver_base
    {
    public:
      linsolv_bos_cpr_gpu();

      ~linsolv_bos_cpr_gpu();

      // Keep the csr_matrix_base init()/setup() overloads visible: declaring
      // the csr_matrix<N>* overloads below otherwise hides them by name.
      using opendarts::linear_solvers::linsolv_iface_bos<n_block_size>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<n_block_size>::setup;

      //////////////////////
      // linear_solver_base
      //////////////////////

      int solve(opendarts::linear_solvers::csr_matrix_base * /*matrix*/,
        opendarts::config::mat_float *v,
        opendarts::config::mat_float *r) override
      {
        return solve(v, r);
      }

      int setup(opendarts::linear_solvers::csr_matrix_base *matrix) override;

      // Real CPR setup happens through setup(csr_matrix_base*); the typed
      // overload is never the entry point.
      int setup(opendarts::linear_solvers::csr_matrix<n_block_size> * /*A*/) override
      {
        printf("CPR wrong method call\n");
        return 0;
      }

      //////////////////////
      // linsolv_iface
      //////////////////////

      int set_p_system_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override
      {
        p_system_preconditioner = prec_input;
        return 0;
      }

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override
      {
        full_system_preconditioner = prec_input;
        return 0;
      }

      int init(opendarts::linear_solvers::csr_matrix<n_block_size> *A,
        int max_iters,
        double tolerance) override;

      void setup_kernel(void);

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      int get_n_iters() override
      {
        return 0;
      }

      opendarts::config::mat_float get_residual() override
      {
        return 0;
      }

      int p_solver_setup_gpu = 0;
      int p_solver_solve_gpu = 0;
      int p_solver_requires_diag_first = 0;

    private:
      // Full block matrix.
      opendarts::linear_solvers::csr_matrix_base *A_base;

      opendarts::linear_solvers::linsolv_iface *p_system_preconditioner;
      opendarts::linear_solvers::linsolv_iface *full_system_preconditioner;

      // Reduced scalar pressure matrix.
      opendarts::linear_solvers::csr_matrix<1> *P;

      opendarts::config::mat_float *D_ps_ss;   // [n_rows * (nvar - 1)] D_ps * inv(D_ss)
      opendarts::config::index_t *block_p_jac_idx;     // per-connection Jacobian off-diagonal block index
      opendarts::config::index_t *inverse_fail_counter; // count of failed block inversions

      opendarts::config::mat_float *P_B, *P_X;     // device pressure-system RHS and solution
      opendarts::config::mat_float *P_B_h, *P_X_h; // host pressure-system RHS and solution
      opendarts::config::mat_float *full_B;        // device second-stage RHS
      opendarts::config::mat_float *rhs_mults;     // device RHS multipliers
      opendarts::config::mat_float *p_vals;        // device pressure-matrix values
      opendarts::config::mat_float *diag_acc;      // device accumulative diagonal (debug)

      int setup_block_size;
      int solve_reduce_block_size;
      int solve_prolongate_block_size;
      int solve_sum_up_block_size;
      int min_grid_size;
      int grid_size;
    };
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_CPR_GPU_HPP
//--------------------------------------------------------------------------
