//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_CPR_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_CPR_HPP
//--------------------------------------------------------------------------

#include <memory>

#include "csr_matrix.hpp"
#include "csr_matrix_base.hpp"
#include "data_types.hpp"
#include "linsolv_hypre_amg.hpp"
#include "linsolv_hypre_ilu.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Open-source two-stage Constrained Pressure Residual (CPR) preconditioner.
     *
     * The in-tree replacement for the proprietary ``linsolv_bos_cpr``: paired
     * with an outer Krylov solver (typically :class:`linsolv_gmres`) it forms
     * the open-source equivalent of the legacy ``bos_gmres + bos_cpr`` stack.
     *
     * Algorithm (forward, quasi-IMPES variant; Wallis 1983):
     *   1. Extract the scalar pressure subsystem ``A_p`` from the block-CSR
     *      Jacobian -- the (0,0) entry of each block, same sparsity pattern.
     *   2. Apply HYPRE BoomerAMG to ``A_p`` for the pressure correction
     *      ``x_p``; prolong to ``x_g`` (pressure component, zero elsewhere).
     *   3. Apply HYPRE ILU(0) to the full-system residual ``r_m = b - A x_g``
     *      for the global update ``x_f``.
     *   4. Return ``x = x_g + x_f``.
     *
     * Transpose (CPRA, Han et al. 2013) -- needed for the adjoint Newton step:
     * the same two-stage structure with the order reversed and each stage
     * applied as its transpose (``M̃^T`` then ``(A_p)^T``). Hooked up via the
     * ``linsolv_iface::solve_transposed`` entry point. The forward CPR is
     * implemented; the transposed solve is currently a placeholder until the
     * transpose ILU path is in place (see :meth:`solve_transposed`).
     *
     * The matrix is consumed through :class:`csr_matrix_base` accessors, so
     * this works against both the legacy ``csr_matrix<N>`` and the unified
     * ``block_csr_matrix`` Jacobian.
     */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_cpr : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {
    public:
      linsolv_cpr();
      ~linsolv_cpr() override;

      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      int set_prec(opendarts::linear_solvers::linsolv_iface * /*prec_input*/) override
      {
        return 0;  // CPR composes its own inner stages internally
      }

      int init(opendarts::linear_solvers::csr_matrix_base *A,
          int max_iters,
          opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A_input) override;

      // iface_bos pure virtuals -- the csr_matrix_base* variants above are used.
      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> * /*A*/,
          int /*max_iters*/,
          double /*tolerance*/) override
      {
        return 0;
      }
      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> * /*A*/) override
      {
        return 0;
      }

      int solve(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X) override;

      int solve_transposed(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X) override;

      int get_n_iters() override { return n_iters_; }
      opendarts::config::mat_float get_residual() override { return 0.0; }

      /** Maximum HYPRE-AMG iterations on the pressure subsystem per CPR apply.
       *  AMG is used as a preconditioner stage -- a few sweeps usually suffice. */
      void set_amg_max_iters(int n) { amg_max_iters_ = n; }
      void set_amg_tolerance(opendarts::config::mat_float t) { amg_tolerance_ = t; }
      void set_ilu_fill_level(int k) { ilu_fill_level_ = k; }

    private:
      // Extract the scalar pressure subsystem A_p (block (0,0) of each block)
      // into a csr_matrix<1> sharing the block-CSR structure.
      void build_pressure_subsystem(opendarts::linear_solvers::csr_matrix_base *A);

      opendarts::linear_solvers::csr_matrix_base *A_;  // full-system matrix (kept by pointer)

      // Pressure subsystem A_p as a scalar matrix + its AMG solver.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> Ap_;
      std::unique_ptr<opendarts::linear_solvers::linsolv_hypre_amg<1>> amg_;

      // Scalar-expanded full-system A_s (size n_block_rows * N_BLOCK_SIZE) and
      // its HYPRE-ILU(0). HYPRE-ILU is specialised for scalar (block size 1)
      // matrices, so the block-CSR Jacobian is expanded once via the
      // polymorphic ``csr_matrix<1>::to_nb_1(csr_matrix_base*)`` helper.
      std::unique_ptr<opendarts::linear_solvers::csr_matrix<1>> As_;
      std::unique_ptr<opendarts::linear_solvers::linsolv_hypre_ilu<1>> ilu_;

      // Pressure-component index inside each block (always 0 in DARTS).
      static constexpr int P_VAR = 0;

      // Configuration.
      int amg_max_iters_;
      opendarts::config::mat_float amg_tolerance_;
      int ilu_fill_level_;
      int max_iters_;
      opendarts::config::mat_float tolerance_;
      int n_iters_;

      // Workspace (scalar arrays of total size n_block_rows * N).
      std::vector<opendarts::config::mat_float> wksp_;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_CPR_HPP
//--------------------------------------------------------------------------
