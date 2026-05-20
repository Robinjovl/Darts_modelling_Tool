//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_GMRES_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_GMRES_HPP
//--------------------------------------------------------------------------

#include <vector>

#include "csr_matrix.hpp"
#include "csr_matrix_base.hpp"
#include "data_types.hpp"
#include "linsolv_iface_bos.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** @brief Open-source restarted GMRES with right preconditioning.
     *
     * The CPU outer Krylov solver of the open-DARTS solver stack: paired with
     * an inner preconditioner (HYPRE AMG, HYPRE ILU, MGR, SuperLU, or any
     * registered solver) it is the natural FGMRES-style default that replaces
     * the proprietary ``linsolv_bos_gmres`` in the open-source build.
     *
     * The algorithm is the standard restarted GMRES of Saad with modified
     * Gram-Schmidt orthogonalisation and Givens rotations for the Hessenberg
     * least-squares step. Right-preconditioning: the preconditioner is applied
     * inside the Arnoldi loop (``z = M^{-1} v_{i-1}``, ``v_i = A z``) and once
     * more to the linear combination at the end of each restart cycle, so the
     * inner ``prec->solve(matrix, in, out)`` call is the only matrix-aware
     * preconditioner contract. Modelled on the proven design of the
     * proprietary ``gmres_solver2`` in ``bos_linear_solver_lib``.
     *
     * The matrix is consumed through :class:`csr_matrix_base` accessors
     * (``get_rows_ptr`` / ``get_cols_ind`` / ``get_values`` plus the block size
     * ``n_row_size``), so the solver works against both the legacy
     * ``csr_matrix<N>`` and the unified ``block_csr_matrix`` Jacobian.
     */
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_gmres : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {
    public:
      linsolv_gmres();
      ~linsolv_gmres() override;

      // Keep the csr_matrix_base init()/setup() overloads visible.
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::init;
      using opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::setup;

      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      int init(opendarts::linear_solvers::csr_matrix_base *A,
          int max_iters,
          opendarts::config::mat_float tolerance) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A_input) override;

      // iface_bos pure virtuals -- not used here; the csr_matrix_base*
      // overloads above are what get called.
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

      int get_n_iters() override { return n_iters_; }
      opendarts::config::mat_float get_residual() override { return final_resid_; }

      /** Restart (Krylov subspace) dimension; default 30. */
      void set_restart(int m) { restart_m_ = m; }
      int get_restart() const { return restart_m_; }

    private:
      opendarts::linear_solvers::csr_matrix_base *A_;
      opendarts::linear_solvers::linsolv_iface *prec_;
      int max_iters_;
      double tolerance_;
      int restart_m_;
      int n_iters_;
      double final_resid_;
      std::vector<opendarts::config::mat_float> wksp_;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_GMRES_HPP
//--------------------------------------------------------------------------
