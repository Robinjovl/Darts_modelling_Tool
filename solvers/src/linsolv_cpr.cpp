//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

// Open-source CPR (Constrained Pressure Residual) two-stage preconditioner.
// See linsolv_cpr.hpp for the design rationale; algorithm follows the
// classical quasi-IMPES CPR (Wallis 1983) -- a pressure-subsystem AMG
// correction followed by a full-system ILU smoothing pass.

#include <cstring>
#include <vector>

#include <HYPRE_utilities.h>

#include "linsolv_cpr.hpp"

extern "C" {
HYPRE_Int HYPRE_Initialize(void);
HYPRE_Int HYPRE_Initialized(void);
}

namespace opendarts
{
  namespace linear_solvers
  {
    using opendarts::config::index_t;
    using opendarts::config::mat_float;

    namespace
    {
      // r += A * v  (block-CSR, host, polymorphic via csr_matrix_base).
      template <uint8_t N>
      inline void block_csr_spmv_add(csr_matrix_base *A,
          const mat_float *v,
          mat_float *r)
      {
        const index_t *rows = A->get_rows_ptr();
        const index_t *cols = A->get_cols_ind();
        const mat_float *vals = A->get_values();
        const index_t n_block_rows = A->n_rows;
        constexpr int Ni = static_cast<int>(N);
        const std::size_t b2 = static_cast<std::size_t>(Ni) * Ni;
        for (index_t i = 0; i < n_block_rows; ++i)
        {
          mat_float *ri = r + static_cast<std::size_t>(i) * Ni;
          for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
          {
            const mat_float *blk = vals + static_cast<std::size_t>(jb) * b2;
            const mat_float *vj = v + static_cast<std::size_t>(cols[jb]) * Ni;
            for (int e = 0; e < Ni; ++e)
            {
              mat_float acc = 0;
              for (int w = 0; w < Ni; ++w)
                acc += blk[e * Ni + w] * vj[w];
              ri[e] += acc;
            }
          }
        }
      }
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cpr<N_BLOCK_SIZE>::linsolv_cpr()
      : A_(nullptr),
        amg_max_iters_(2),       // AMG used as a prec -- a couple of V-cycles
        amg_tolerance_(1.0e-2),
        ilu_fill_level_(0),
        max_iters_(50),
        tolerance_(1.0e-5),
        n_iters_(0)
    {
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cpr<N_BLOCK_SIZE>::~linsolv_cpr() = default;

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::init(csr_matrix_base *A,
        int max_iters,
        mat_float tolerance)
    {
      A_ = A;
      max_iters_ = max_iters;
      tolerance_ = tolerance;
      n_iters_ = 0;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::build_pressure_subsystem(csr_matrix_base *A)
    {
      const index_t *rows = A->get_rows_ptr();
      const index_t *cols = A->get_cols_ind();
      const mat_float *vals = A->get_values();
      const index_t n_block_rows = A->n_rows;
      const index_t n_block_cols = A->n_cols;
      const std::size_t b2 =
          static_cast<std::size_t>(N_BLOCK_SIZE) * N_BLOCK_SIZE;
      const index_t nnz = rows[n_block_rows];

      if (!Ap_)
        Ap_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      Ap_->init(n_block_rows, n_block_cols, nnz);

      // Copy the (block) sparsity pattern verbatim -- A_p has the same
      // sparsity as the block-CSR (one scalar entry per block).
      std::copy(rows, rows + n_block_rows + 1, Ap_->rows_ptr.data());
      std::copy(cols, cols + nnz, Ap_->cols_ind.data());

      // Extract the (0, 0) entry of every block as the A_p value.
      mat_float *ap_vals = Ap_->values.data();
      for (index_t jb = 0; jb < nnz; ++jb)
        ap_vals[jb] = vals[static_cast<std::size_t>(jb) * b2
            + static_cast<std::size_t>(P_VAR) * N_BLOCK_SIZE + P_VAR];

      // Diagonal indices (csr_matrix_base contract).
      Ap_->n_non_zeros = nnz;
      Ap_->n_row_size = 1;
      Ap_->is_square = (n_block_rows == n_block_cols) ? 1 : 0;
      Ap_->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::setup(csr_matrix_base *A_input)
    {
      // Ensure HYPRE is initialised; linsolv_hypre_amg / linsolv_hypre_ilu
      // assume it (MGR's compositionalFlowStrategy already does this -- when
      // CPR is used without MGR in the process the wrappers would otherwise
      // hit "[Generic error]" out of HYPRE's diagnostic layer).
      if (!HYPRE_Initialized())
        HYPRE_Initialize();

      A_ = A_input;
      build_pressure_subsystem(A_input);

      // Build the scalar expansion A_s of the block-CSR Jacobian once -- the
      // pattern is fixed for the run -- and hand it to HYPRE-ILU (specialised
      // for block size 1). The expansion is the polymorphic to_nb_1 helper
      // (phase C), which produces a csr_matrix<1> sharing the engine's value
      // ordering, so the scalar RHS / dX vectors do not need permutation.
      if (!As_)
        As_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      As_->to_nb_1(A_input);

      // TODO: re-enable the full-system ILU stage once linsolv_hypre_ilu's
      // setup is stable on the open-darts scalar expansion. HYPRE_ILUSetup
      // currently aborts with "[Generic error]" -- linsolv_hypre_ilu appears
      // underused (no tests exercise it) and likely needs the same kind of
      // tightening linsolv_hypre_amg got. Until then CPR runs as a one-stage
      // pressure-only preconditioner (Wallis "single-stage" CPR), which is
      // weaker than the full two-stage CPR but a valid place-holder.
      // if (!ilu_)
      //   ilu_ = std::make_unique<opendarts::linear_solvers::linsolv_hypre_ilu<1>>();
      // ilu_->init(As_.get(), 1, 0.0);
      // ilu_->setup(As_.get());

      // AMG on the scalar pressure subsystem A_p. AMG is run as a
      // preconditioner stage, so a relaxed tolerance and a couple of cycles
      // are sufficient.
      if (!amg_)
        amg_ = std::make_unique<opendarts::linear_solvers::linsolv_hypre_amg<1>>();
      amg_->init(Ap_.get(), amg_max_iters_, amg_tolerance_);
      amg_->setup(Ap_.get());

      // Scratch buffers for the per-apply CPR stages.
      const std::size_t n_scalar =
          static_cast<std::size_t>(A_input->n_rows) * N_BLOCK_SIZE;
      const std::size_t n_pressure = static_cast<std::size_t>(A_input->n_rows);
      // Layout: x_g[n_scalar], r_m[n_scalar], x_f[n_scalar],
      //         r_p[n_pressure], x_p[n_pressure].
      wksp_.assign(3 * n_scalar + 2 * n_pressure, 0.0);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::solve(mat_float *B, mat_float *X)
    {
      if (!A_ || !amg_)
        return -1;

      const index_t n_block_rows = A_->n_rows;
      const std::size_t n_scalar =
          static_cast<std::size_t>(n_block_rows) * N_BLOCK_SIZE;
      const std::size_t n_pressure = static_cast<std::size_t>(n_block_rows);

      mat_float *x_g = wksp_.data();
      mat_float *r_m = x_g + n_scalar;
      mat_float *x_f = r_m + n_scalar;
      mat_float *r_p = x_f + n_scalar;
      mat_float *x_p = r_p + n_pressure;

      // Stage 1: pressure correction. Restrict B to the pressure subsystem,
      // solve A_p x_p = r_p with AMG, prolong to x_g (pressure component;
      // zero elsewhere).
      for (index_t i = 0; i < n_block_rows; ++i)
        r_p[i] = B[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR];
      std::memset(x_p, 0, n_pressure * sizeof(mat_float));
      amg_->solve(r_p, x_p);

      std::memset(x_g, 0, n_scalar * sizeof(mat_float));
      for (index_t i = 0; i < n_block_rows; ++i)
        x_g[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR] = x_p[i];

      // Stage 2: full-system smoothing on r_m = B - A x_g.
      std::memcpy(r_m, B, n_scalar * sizeof(mat_float));
      // r_m -= A x_g
      // Use a scratch (negated): compute A x_g into x_f, subtract from r_m.
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      block_csr_spmv_add<N_BLOCK_SIZE>(A_, x_g, x_f);
      for (std::size_t k = 0; k < n_scalar; ++k)
        r_m[k] -= x_f[k];

      // Apply the full ILU(0) to r_m for x_f.
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      if (ilu_)
        ilu_->solve(r_m, x_f);
      // else: skip the full-system smoothing stage (TODO -- see setup()).

      // X = x_g + x_f.
      for (std::size_t k = 0; k < n_scalar; ++k)
        X[k] = x_g[k] + x_f[k];

      n_iters_ = 1;  // CPR as a preconditioner: one application
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::solve_transposed(mat_float * /*B*/,
        mat_float * /*X*/)
    {
      // CPRA (Han et al. 2013, eq 13):
      //   1. Solve M̃^T x_f = r          -- transpose ILU
      //   2. r_m = r - (Ã)^T x_f         -- transpose SpMV
      //   3. r_p = C^T r_m                -- pressure restriction
      //   4. Solve (A_p)^T x_p = r_p     -- HYPRE_BoomerAMGSolveT
      //   5. X = C x_p + x_f             -- pressure prolongation + ILU update
      //
      // Pending: transpose-ILU solve is not exposed by HYPRE_ILU; needs either
      // building a separate ILU on A^T or hooking HYPRE's L^T/U^T sweeps.
      // The transpose-AMG step (HYPRE_BoomerAMGSolveT) is straightforward and
      // will land together with the ILU transpose in the next iteration.
      return -1;
    }

    // Explicit instantiations for the block sizes the engine uses.
    template class linsolv_cpr<1>;
    template class linsolv_cpr<2>;
    template class linsolv_cpr<3>;
    template class linsolv_cpr<4>;
    template class linsolv_cpr<5>;
    template class linsolv_cpr<6>;
    template class linsolv_cpr<7>;
    template class linsolv_cpr<8>;
    template class linsolv_cpr<9>;
    template class linsolv_cpr<10>;
    template class linsolv_cpr<11>;
    template class linsolv_cpr<12>;
    template class linsolv_cpr<13>;
  } // namespace linear_solvers
} // namespace opendarts
