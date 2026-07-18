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

      // Polymorphic init() entry -- bypasses the linsolv_iface_bos<N>
      // static_cast which is UB when A is a block_csr_matrix. Matches the
      // setup() shape above.
      int init(opendarts::linear_solvers::csr_matrix_base *A,
        opendarts::config::index_t max_iters,
        opendarts::config::mat_float tolerance) override;

      //////////////////////
      // linsolv_iface
      //////////////////////

      int set_p_system_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override
      {
        p_system_preconditioner = prec_input;
        return 0;
      }

      /** Forward the outer Krylov iteration count to the pressure stage so its
       *  hierarchy-reuse policy (AMGX structure reuse) can gate the next setup. */
      void set_last_outer_iters(int n_iters) override
      {
        if (p_system_preconditioner)
          p_system_preconditioner->set_last_outer_iters(n_iters);
      }

      /** Pressure preconditioner for the TRANSPOSED apply (CPRA). AMGX has no
       *  transpose-solve API, so the adjoint path binds a second instance to
       *  the explicitly transposed pressure matrix P^T. Lazily initialised on
       *  the first solve_transposed(); forward-only runs never pay for it. */
      int set_p_system_prec_t(opendarts::linear_solvers::linsolv_iface *prec_input)
      {
        p_system_preconditioner_t = prec_input;
        return 0;
      }

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override
      {
        full_system_preconditioner = prec_input;
        return 0;
      }

      int init(opendarts::linear_solvers::csr_matrix<n_block_size> *A,
        int max_iters,
        double tolerance) override
      {
        return this->init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A),
            static_cast<opendarts::config::index_t>(max_iters),
            static_cast<opendarts::config::mat_float>(tolerance));
      }

      void setup_kernel(void);

      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      /** Transposed CPR apply (CPRA, Han et al. 2013): each stage is the
       *  transpose of the matching forward stage, in reverse order:
       *    1. X   = ILU^{-T} B          (transposed full-system stage)
       *    2. r_m = B - A^T X           (transposed block SpMV)
       *    3. r_p = C^T r_m             (pressure-slot extraction)
       *    4. y   = (P^T)^{-1} r_p      (second AMGX instance on P^T)
       *    5. X  += W^T (D_m y)         (weights + sign mults on the
       *                                  PROLONGATION side -- exact transpose
       *                                  of the forward weighted reduction)
       *  Device pointers, like solve(). The P^T chain (transpose value map,
       *  P^T shell, the second pressure preconditioner) is built lazily on
       *  the first call and refreshed after every setup(). */
      int solve_transposed(opendarts::config::mat_float *B,
        opendarts::config::mat_float *X) override;

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
      /// (Re)builds the transposed pressure chain after a setup(): gathers
      /// P^T values through the transpose map and re-sets-up the second
      /// pressure preconditioner. First call also builds the map/shell.
      int refresh_transpose_chain();

      // Full block matrix.
      opendarts::linear_solvers::csr_matrix_base *A_base;

      opendarts::linear_solvers::linsolv_iface *p_system_preconditioner;
      opendarts::linear_solvers::linsolv_iface *full_system_preconditioner;
      // Second pressure preconditioner bound to P^T (adjoint path; see
      // set_p_system_prec_t). Not owned.
      opendarts::linear_solvers::linsolv_iface *p_system_preconditioner_t = nullptr;

      // Reduced scalar pressure matrix.
      opendarts::linear_solvers::csr_matrix<1> *P;

      // Transposed pressure matrix (adjoint path): same (structurally
      // symmetric) pattern as P; values gathered per refresh through the
      // device transpose map t_map (t_map[j] = position of the mirrored
      // entry). Built lazily on the first solve_transposed().
      opendarts::linear_solvers::csr_matrix<1> *P_T = nullptr; // owns the device P^T values
      opendarts::config::index_t *t_map_d = nullptr;
      long setup_generation_ = 0;     // bumped by every setup()
      long transpose_generation_ = -1; // generation the P^T chain matches
      bool p_prec_t_initialized_ = false;

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
