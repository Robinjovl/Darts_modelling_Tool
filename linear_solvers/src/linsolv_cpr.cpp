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
//
// HYPRE BoomerAMG / HYPRE_ILU are driven directly here (not through
// linsolv_hypre_amg / linsolv_hypre_ilu), with the same configuration the
// in-tree MGR uses for the pressure level of compositional flow
// (CompositionalFlowStrategy::setupPressureAMG). The wrappers are tuned for
// the elasticity engines (mechanical systems) and would be the wrong AMG
// setup for flow.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_utilities.h"
#include "_hypre_utilities.h"

#include "hypre_ij_builder.hpp"
#include "linsolv_cpr.hpp"
#include "solver_configs.hpp"

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
      // Error helper for HYPRE calls. A HYPRE failure here is not always a
      // programmer bug -- a diverging Newton step can hand the preconditioner
      // a degenerate pressure subsystem (NaN/Inf entries, singular
      // decoupling), which is runtime data. Throwing (instead of the former
      // std::exit(-1)) lets setup()/solve() translate the failure into a
      // nonzero return, which engine_base::solve_linear_equation() converts
      // into a timestep cut -- instead of killing the host Python process.
      inline void check_hypre(int res, const char *where)
      {
        if (res)
        {
          char msg[256];
          HYPRE_DescribeError(res, msg);
          std::string err = std::string("linsolv_cpr: HYPRE error in ") +
                            where + " -- " + msg;
          std::cerr << err << std::endl;
          // Clear HYPRE's sticky global error flag so a recovered solver
          // (after the timestep cut) does not keep tripping on it.
          HYPRE_ClearAllErrors();
          throw std::runtime_error(err);
        }
      }

      // Solve the small dense system M x = b (row-major, n x n, n <= block
      // size - 1) by Gaussian elimination with partial pivoting. Used to build
      // the true-IMPES pressure-decoupling weights from a cell's diagonal
      // block. Returns false if the f-block is (near-)singular -- the caller
      // then falls back to quasi-IMPES for that row. M and b are overwritten.
      inline bool solve_dense_gepp(mat_float *M, mat_float *b, int n,
          mat_float *x)
      {
        mat_float scale = 1.0;
        for (int k = 0; k < n * n; ++k)
          scale = std::max(scale, std::abs(M[k]));
        const mat_float piv_tol =
            std::numeric_limits<mat_float>::epsilon() * scale * 100.0;
        for (int col = 0; col < n; ++col)
        {
          int piv = col;
          mat_float best = std::abs(M[col * n + col]);
          for (int r = col + 1; r < n; ++r)
          {
            const mat_float v = std::abs(M[r * n + col]);
            if (v > best) { best = v; piv = r; }
          }
          if (best <= piv_tol)
            return false;
          if (piv != col)
          {
            for (int c = 0; c < n; ++c)
              std::swap(M[col * n + c], M[piv * n + c]);
            std::swap(b[col], b[piv]);
          }
          const mat_float d = M[col * n + col];
          for (int r = col + 1; r < n; ++r)
          {
            const mat_float f = M[r * n + col] / d;
            for (int c = col; c < n; ++c)
              M[r * n + c] -= f * M[col * n + c];
            b[r] -= f * b[col];
          }
        }
        for (int r = n - 1; r >= 0; --r)
        {
          mat_float s = b[r];
          for (int c = r + 1; c < n; ++c)
            s -= M[r * n + c] * x[c];
          x[r] = s / M[r * n + r];
        }
        return true;
      }

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

      // r += A^T * v  (block-CSR, host). Walks rows of A and scatters each
      // block contribution to the corresponding row of A^T. Mirrors the
      // helper in linsolv_gmres.cpp.
      template <uint8_t N>
      inline void block_csr_spmv_t_add(csr_matrix_base *A,
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
          const mat_float *vi = v + static_cast<std::size_t>(i) * Ni;
          for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
          {
            const mat_float *blk = vals + static_cast<std::size_t>(jb) * b2;
            mat_float *rj = r + static_cast<std::size_t>(cols[jb]) * Ni;
            for (int w = 0; w < Ni; ++w)
            {
              mat_float acc = 0;
              for (int e = 0; e < Ni; ++e)
                acc += blk[e * Ni + w] * vi[e];
              rj[w] += acc;
            }
          }
        }
      }

      // CSR transpose for csr_matrix<1>: builds B = A^T with the same data
      // layout. Output is initialised in place into dst.
      inline void csr_transpose_scalar(
          opendarts::linear_solvers::csr_matrix<1> &src,
          opendarts::linear_solvers::csr_matrix<1> &dst)
      {
        const index_t n_rows = src.n_rows;
        const index_t n_cols = src.n_cols;
        const index_t nnz = src.rows_ptr[n_rows];

        dst.init(n_cols, n_rows, nnz);

        // Step 1: column counts (= row counts of A^T).
        std::fill(dst.rows_ptr.begin(), dst.rows_ptr.end(), 0);
        for (index_t k = 0; k < nnz; ++k)
          dst.rows_ptr[src.cols_ind[k] + 1]++;
        // Cumulative sum to build row pointers.
        for (index_t i = 0; i < n_cols; ++i)
          dst.rows_ptr[i + 1] += dst.rows_ptr[i];

        // Step 2: scatter values, advancing a temporary write head per row.
        std::vector<index_t> head(dst.rows_ptr.begin(), dst.rows_ptr.begin() + n_cols);
        for (index_t i = 0; i < n_rows; ++i)
        {
          for (index_t jb = src.rows_ptr[i]; jb < src.rows_ptr[i + 1]; ++jb)
          {
            const index_t j = src.cols_ind[jb];
            const index_t w = head[j]++;
            dst.cols_ind[w] = i;
            dst.values[w] = src.values[jb];
          }
        }

        dst.n_non_zeros = nnz;
        dst.n_row_size = 1;
        dst.is_square = (n_rows == n_cols) ? 1 : 0;
        dst.type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
      }

      // Child timer node under an (optional) parent; null-safe so the solver
      // works identically when the engine never wired timer nodes.
      inline ::timer_node *cpr_sub_timer(::timer_node *parent, const char *name)
      {
        return parent ? &parent->node[name] : nullptr;
      }

      struct cpr_scoped_timer
      {
        ::timer_node *t;
        explicit cpr_scoped_timer(::timer_node *tn) : t(tn)
        {
          if (t)
            t->start();
        }
        ~cpr_scoped_timer()
        {
          if (t)
            t->stop();
        }
        cpr_scoped_timer(const cpr_scoped_timer &) = delete;
        cpr_scoped_timer &operator=(const cpr_scoped_timer &) = delete;
      };
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cpr<N_BLOCK_SIZE>::linsolv_cpr()
      : A_(nullptr),
        Ap_ij_(nullptr),
        Ap_parcsr_(nullptr),
        amg_b_ij_(nullptr),
        amg_x_ij_(nullptr),
        amg_b_par_(nullptr),
        amg_x_par_(nullptr),
        amg_(nullptr),
        amg_setup_done_(false),
        Ap_T_ij_(nullptr),
        Ap_T_parcsr_(nullptr),
        amg_T_b_ij_(nullptr),
        amg_T_x_ij_(nullptr),
        amg_T_b_par_(nullptr),
        amg_T_x_par_(nullptr),
        amg_T_(nullptr),
        amg_T_setup_done_(false),
        As_ij_(nullptr),
        As_parcsr_(nullptr),
        ilu_b_ij_(nullptr),
        ilu_x_ij_(nullptr),
        ilu_b_par_(nullptr),
        ilu_x_par_(nullptr),
        ilu_(nullptr),
        ilu_setup_done_(false),
        As_T_ij_(nullptr),
        As_T_parcsr_(nullptr),
        ilu_T_b_ij_(nullptr),
        ilu_T_x_ij_(nullptr),
        ilu_T_b_par_(nullptr),
        ilu_T_x_par_(nullptr),
        ilu_T_(nullptr),
        ilu_T_setup_done_(false),
        amg_max_iters_(2),       // AMG used as a prec -- a couple of V-cycles
        ilu_fill_level_(0),
        max_iters_(50),
        tolerance_(1.0e-5),
        n_iters_(0),
        reuse_amg_hierarchy_(false),
        adaptive_amg_rebuild_(false),
        adaptive_iter_threshold_(15),
        adaptive_consecutive_bad_(2),
        last_outer_iters_(0),
        consecutive_bad_streak_(0),
        force_amg_rebuild_(false),
        first_setup_(true)
    {
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::ensure_row_indices(index_t n)
    {
      if (static_cast<index_t>(row_indices_.size()) < n)
      {
        const index_t old_size = static_cast<index_t>(row_indices_.size());
        row_indices_.resize(n);
        std::iota(row_indices_.begin() + old_size, row_indices_.end(), old_size);
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_cpr<N_BLOCK_SIZE>::~linsolv_cpr()
    {
      destroy_hypre();
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::destroy_hypre()
    {
      if (amg_) { HYPRE_BoomerAMGDestroy(amg_); amg_ = nullptr; }
      if (Ap_ij_) { HYPRE_IJMatrixDestroy(Ap_ij_); Ap_ij_ = nullptr; }
      if (amg_b_ij_) { HYPRE_IJVectorDestroy(amg_b_ij_); amg_b_ij_ = nullptr; }
      if (amg_x_ij_) { HYPRE_IJVectorDestroy(amg_x_ij_); amg_x_ij_ = nullptr; }

      if (amg_T_) { HYPRE_BoomerAMGDestroy(amg_T_); amg_T_ = nullptr; }
      if (Ap_T_ij_) { HYPRE_IJMatrixDestroy(Ap_T_ij_); Ap_T_ij_ = nullptr; }
      if (amg_T_b_ij_) { HYPRE_IJVectorDestroy(amg_T_b_ij_); amg_T_b_ij_ = nullptr; }
      if (amg_T_x_ij_) { HYPRE_IJVectorDestroy(amg_T_x_ij_); amg_T_x_ij_ = nullptr; }

      if (ilu_) { HYPRE_ILUDestroy(ilu_); ilu_ = nullptr; }
      if (As_ij_) { HYPRE_IJMatrixDestroy(As_ij_); As_ij_ = nullptr; }
      if (ilu_b_ij_) { HYPRE_IJVectorDestroy(ilu_b_ij_); ilu_b_ij_ = nullptr; }
      if (ilu_x_ij_) { HYPRE_IJVectorDestroy(ilu_x_ij_); ilu_x_ij_ = nullptr; }

      if (ilu_T_) { HYPRE_ILUDestroy(ilu_T_); ilu_T_ = nullptr; }
      if (As_T_ij_) { HYPRE_IJMatrixDestroy(As_T_ij_); As_T_ij_ = nullptr; }
      if (ilu_T_b_ij_) { HYPRE_IJVectorDestroy(ilu_T_b_ij_); ilu_T_b_ij_ = nullptr; }
      if (ilu_T_x_ij_) { HYPRE_IJVectorDestroy(ilu_T_x_ij_); ilu_T_x_ij_ = nullptr; }

      amg_setup_done_ = false;
      amg_T_setup_done_ = false;
      ilu_setup_done_ = false;
      ilu_T_setup_done_ = false;
    }

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

      // True-IMPES pressure decoupling (Wallis 1983). For each block-row i,
      // build a weight row w_i (1 x N) that eliminates the non-pressure ("f")
      // unknowns from the diagonal block D = A[i,i]:
      //   w_i = [1 (at P_VAR), -D_pf D_ff^{-1}],
      // obtained by solving (D_ff)^T s = -(D_pf)^T for s = the f-weights. The
      // scalar pressure system is then A_p = R A C with R_i = w_i^T (weighted
      // row restriction) and C = e_P (pressure prolongation):
      //   A_p[i,j] = sum_v w_i[v] * A[i,j][v, P_VAR].
      // This is the open-source standalone equivalent of the in-tree
      // mgr_linear_solver BCSR-CPR true-IMPES reduction, and a strict upgrade
      // over the previous quasi-IMPES extraction (just the (P_VAR, P_VAR)
      // entry), which is too weak a pressure operator for the wider, strongly
      // coupled MPFA Jacobian -- there FlexGMRES stalls and Newton diverges.
      // Quasi-IMPES (w_i = e_P) is recovered per-row as a fallback whenever the
      // local f-block is missing / singular / yields non-finite or oversized
      // weights, so the result is never worse than before.
      constexpr int Ni = static_cast<int>(N_BLOCK_SIZE);
      constexpr int n_f = Ni - 1;             // non-pressure variables per cell
      constexpr int NF = (n_f > 0) ? n_f : 1; // array sizing (avoid zero-length)
      const mat_float weight_max = 1e6;       // reject pathological f-weights

      cpr_weights_.assign(static_cast<std::size_t>(n_block_rows) * Ni, 0.0);

      int f_vars[NF];
      for (int v = 0, a = 0; v < Ni; ++v)
        if (v != P_VAR)
          f_vars[a++] = v;

      // weight_scheme_ == 1 (default): column-sum True-IMPES (Wallis 1983;
      // parity with the reference linsolv_bos_cpr). The decoupling block
      // for cell i is the sum of A(m,i) over ALL block-rows m holding a
      // block in column i (diagonal included) -- the IMPES mass-balance
      // lumping. weight_scheme_ == 0 keeps the previous diagonal-block-only
      // variant (quasi-IMPES with local f-elimination).
      if (n_f > 0 && weight_scheme_ == 1)
      {
        colsum_.assign(static_cast<std::size_t>(n_block_rows) * b2, 0.0);
        for (index_t m = 0; m < n_block_rows; ++m)
          for (index_t jb = rows[m]; jb < rows[m + 1]; ++jb)
          {
            const index_t ci = cols[jb];
            const mat_float *blk = vals + static_cast<std::size_t>(jb) * b2;
            mat_float *cs = colsum_.data() + static_cast<std::size_t>(ci) * b2;
            for (std::size_t e = 0; e < b2; ++e)
              cs[e] += blk[e];
          }
      }

      for (index_t i = 0; i < n_block_rows; ++i)
      {
        mat_float *w = &cpr_weights_[static_cast<std::size_t>(i) * Ni];
        w[P_VAR] = 1.0;  // f-weights stay 0 == quasi-IMPES unless replaced below

        if (n_f > 0)
        {
          const mat_float *D = nullptr;
          if (weight_scheme_ == 1)
          {
            D = colsum_.data() + static_cast<std::size_t>(i) * b2;
          }
          else
          {
            // Locate the diagonal block (col == i) of block-row i.
            for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
              if (cols[jb] == i)
              {
                D = vals + static_cast<std::size_t>(jb) * b2;
                break;
              }
          }

          if (D != nullptr)
          {
            mat_float M[NF * NF];
            mat_float rhs[NF];
            for (int a = 0; a < n_f; ++a)
            {
              rhs[a] = -D[static_cast<std::size_t>(P_VAR) * Ni + f_vars[a]];
              for (int bb = 0; bb < n_f; ++bb)
                M[a * n_f + bb] =
                    D[static_cast<std::size_t>(f_vars[bb]) * Ni + f_vars[a]];
            }
            mat_float sol[NF];
            if (solve_dense_gepp(M, rhs, n_f, sol))
            {
              bool ok = true;
              for (int a = 0; a < n_f; ++a)
                if (!std::isfinite(sol[a]) || std::abs(sol[a]) > weight_max)
                {
                  ok = false;
                  break;
                }
              if (ok)
                for (int a = 0; a < n_f; ++a)
                  w[f_vars[a]] = sol[a];
            }
          }
        }
      }

      // Weighted pressure column of every block: A_p[jb] = sum_v w_i[v]
      // * block[v, P_VAR], where w_i is the weight row of jb's block-row.
      mat_float *ap_vals = Ap_->values.data();
      for (index_t i = 0; i < n_block_rows; ++i)
      {
        const mat_float *w = &cpr_weights_[static_cast<std::size_t>(i) * Ni];
        for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
        {
          const mat_float *blk = vals + static_cast<std::size_t>(jb) * b2;
          mat_float acc = 0.0;
          for (int v = 0; v < Ni; ++v)
            acc += w[v] * blk[static_cast<std::size_t>(v) * Ni + P_VAR];
          ap_vals[jb] = acc;
        }
      }

      // Row sign normalisation (default rhs_mults parity): flip rows whose
      // pressure diagonal is negative so the AMG's M-matrix-oriented
      // coarsening/smoothing heuristics see a positive diagonal. The same
      // +-1 multiplier is applied to the restricted residual on the forward
      // path and to the prolonged solution on the transposed path.
      rhs_mults_.assign(static_cast<std::size_t>(n_block_rows), 1.0);
      for (index_t i = 0; i < n_block_rows; ++i)
      {
        mat_float diag_val = 0.0;
        for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
          if (cols[jb] == i)
          {
            diag_val = ap_vals[jb];
            break;
          }
        if (diag_val < 0.0)
        {
          for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
            ap_vals[jb] = -ap_vals[jb];
          rhs_mults_[i] = -1.0;
        }
      }

      Ap_->n_non_zeros = nnz;
      Ap_->n_row_size = 1;
      Ap_->is_square = (n_block_rows == n_block_cols) ? 1 : 0;
      Ap_->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
    }

    // Pick the per-handle row-degree cache (Ap / Ap_T / As / As_T); each of the
    // four IJ matrices keeps its own so refresh() can skip recomputing.
    template <uint8_t N_BLOCK_SIZE>
    std::vector<opendarts::config::index_t> *
    linsolv_cpr<N_BLOCK_SIZE>::pick_n_cols_cache(HYPRE_IJMatrix &A_ij)
    {
      if (&A_ij == &Ap_ij_)             return &n_cols_Ap_;
      else if (&A_ij == &Ap_T_ij_)      return &n_cols_Ap_T_;
      else if (&A_ij == &As_ij_)        return &n_cols_As_;
      else if (&A_ij == &As_T_ij_)      return &n_cols_As_T_;
      return &n_cols_As_;  // fallback
    }

    // Raw scalar-CSR triple overload -- lets the forward stage feed the
    // scalar_csr_adapter's borrowed (row_ptr, col_ind, values) directly, with no
    // intermediate As_ shell copy. The scalar-CSR -> HYPRE-IJ create/set/assemble
    // sequence lives in the shared hypre_ij::build helper (also used by
    // linsolv_hypre_amg/_ilu).
    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::build_hypre_ij(index_t n_rows,
        const index_t *row_ptr, const index_t *col_ind,
        const opendarts::config::mat_float *values, HYPRE_IJMatrix &A_ij,
        HYPRE_ParCSRMatrix &A_parcsr)
    {
      opendarts::linear_solvers::hypre_ij::build(n_rows, row_ptr, col_ind, values,
          row_indices_, *pick_n_cols_cache(A_ij), A_ij, &A_parcsr);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::build_hypre_ij(
        opendarts::linear_solvers::csr_matrix<1> &A,
        HYPRE_IJMatrix &A_ij,
        HYPRE_ParCSRMatrix &A_parcsr)
    {
      build_hypre_ij(A.n_rows, A.rows_ptr.data(), A.get_cols_ind(),
          A.get_values(), A_ij, A_parcsr);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::refresh_hypre_ij(index_t n_rows,
        const index_t *row_ptr, const index_t *col_ind,
        const opendarts::config::mat_float *values, HYPRE_IJMatrix &A_ij,
        HYPRE_ParCSRMatrix &A_parcsr)
    {
      opendarts::linear_solvers::hypre_ij::refresh(n_rows, row_ptr, col_ind,
          values, row_indices_, *pick_n_cols_cache(A_ij), A_ij, &A_parcsr);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::refresh_hypre_ij(
        opendarts::linear_solvers::csr_matrix<1> &A,
        HYPRE_IJMatrix &A_ij,
        HYPRE_ParCSRMatrix &A_parcsr)
    {
      refresh_hypre_ij(A.n_rows, A.rows_ptr.data(), A.get_cols_ind(),
          A.get_values(), A_ij, A_parcsr);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::create_hypre_vectors(index_t n_rows,
        HYPRE_IJVector &b_ij,
        HYPRE_IJVector &x_ij)
    {
      const index_t ilower = 0;
      const index_t iupper = n_rows - 1;

      check_hypre(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &b_ij),
          "IJVectorCreate(b)");
      check_hypre(HYPRE_IJVectorSetPrintLevel(b_ij, 0), "IJVectorSetPrintLevel(b)");
      check_hypre(HYPRE_IJVectorSetObjectType(b_ij, HYPRE_PARCSR),
          "IJVectorSetObjectType(b)");

      check_hypre(HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, ilower, iupper, &x_ij),
          "IJVectorCreate(x)");
      check_hypre(HYPRE_IJVectorSetPrintLevel(x_ij, 0), "IJVectorSetPrintLevel(x)");
      check_hypre(HYPRE_IJVectorSetObjectType(x_ij, HYPRE_PARCSR),
          "IJVectorSetObjectType(x)");
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::set_hypre_vector(HYPRE_IJVector &v_ij,
        index_t n_rows,
        mat_float *vals,
        HYPRE_ParVector &v_par)
    {
      ensure_row_indices(n_rows);

      check_hypre(HYPRE_IJVectorInitialize(v_ij), "IJVectorInitialize");
      check_hypre(HYPRE_IJVectorSetValues(v_ij, n_rows, row_indices_.data(), vals),
          "IJVectorSetValues");
      check_hypre(HYPRE_IJVectorAssemble(v_ij), "IJVectorAssemble");
      check_hypre(HYPRE_IJVectorGetObject(v_ij, (void **) &v_par),
          "IJVectorGetObject");
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::refresh_scalar_expansion(
        csr_matrix_base *A_input)
    {
      if (!As_)
        As_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();

      // Preferred path -- bind a scalar_csr_adapter to the block Jacobian and
      // reuse the cached scalar structure across Newton iterations. The adapter's
      // value buffer is the source of truth: the forward HYPRE-ILU stage reads it
      // directly (SuperLU-style), so no per-Newton copy into the As_ shell is
      // paid. The As_ csr_matrix<1> shell is materialised only when the
      // transpose/adjoint chain needs it -- by ensure_As_shell().
      auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input);
      if (A_block != nullptr)
      {
        if (!scalar_adapter_
            || scalar_adapter_->n_rows() != A_block->scalar_n_rows()
            || scalar_adapter_->nnz()
                   != A_block->n_blocks() * A_block->block_size() * A_block->block_size())
        {
          // First setup or a structure change (e.g. AMR -- not used today
          // but the adapter binds to a specific (matrix, expansion) pair).
          scalar_adapter_ = std::make_unique<
              opendarts::linear_solvers::scalar_csr_adapter>(*A_block);
        }
        else
        {
          scalar_adapter_->refresh();
        }
      }
      else
      {
        // Legacy path -- engine is still feeding a csr_matrix<N> (tests,
        // GPU, reference build). Use the polymorphic to_nb_1, which
        // performs both the structural rebuild and the value gather. There is
        // no adapter, so the forward stage and ensure_As_shell() both read As_.
        scalar_adapter_.reset();
        As_->to_nb_1(A_input);
      }
    }

    // Materialise the HYPRE-facing As_ csr_matrix<1> shell (structure once +
    // values) from the scalar_csr_adapter. Needed only by the transpose/adjoint
    // chain: csr_transpose_scalar reads As_, whereas the forward stage reads the
    // adapter directly. No-op on the legacy path, where refresh_scalar_expansion's
    // to_nb_1 already filled As_. (The body is the As_ population that used to run
    // unconditionally inside refresh_scalar_expansion, so As_ is byte-identical to
    // before -- the adjoint chain is unchanged.)
    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::ensure_As_shell()
    {
      if (!scalar_adapter_)
        return;  // legacy path: As_ already populated by to_nb_1
      if (!As_)
        As_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      const auto n = scalar_adapter_->n_rows();
      const auto nnz = scalar_adapter_->nnz();
      if (As_->n_rows != n
          || static_cast<opendarts::config::index_t>(As_->n_non_zeros) != nnz)
      {
        As_->init(n, scalar_adapter_->n_cols(), nnz);
        std::copy(scalar_adapter_->row_ptr(),
            scalar_adapter_->row_ptr() + n + 1, As_->rows_ptr.data());
        std::copy(scalar_adapter_->col_ind(),
            scalar_adapter_->col_ind() + nnz, As_->cols_ind.data());
        As_->n_non_zeros = nnz;
        As_->n_row_size = 1;
        As_->is_square = (n == scalar_adapter_->n_cols()) ? 1 : 0;
        As_->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
      }
      std::copy(scalar_adapter_->values(),
          scalar_adapter_->values() + nnz, As_->values.data());
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::reconfigure(const solver_config &config)
    {
      const auto *cfg = dynamic_cast<const cpr_solver_config *>(&config);
      if (cfg == nullptr)
        return 1;

      // Structural on a live solver: the HYPRE scalar-ILU chain and the
      // block-ILU0 factor state are disjoint; swapping the stage identity
      // needs a fresh solver (still no Jacobian impact -- the caller rebuilds
      // and engine_base::set_linear_solver re-inits on the existing matrix).
      if (!first_setup_ && cfg->stage2_type != stage2_type_)
        return 1;

      // Detect warm changes BEFORE overwriting the stored options.
      const bool amg_profile_changed =
          cfg->amg_coarsen_type != amg_coarsen_type_
          || cfg->amg_interp_type != amg_interp_type_
          || cfg->amg_relax_type != amg_relax_type_
          || cfg->amg_relax_order != amg_relax_order_
          || cfg->amg_num_sweeps != amg_num_sweeps_
          || cfg->amg_strong_threshold != amg_strong_threshold_
          || cfg->amg_agg_num_levels != amg_agg_num_levels_
          || cfg->amg_agg_interp_type != amg_agg_interp_type_
          || cfg->amg_agg_pmax_elmts != amg_agg_pmax_elmts_
          || cfg->amg_pmax_elmts != amg_pmax_elmts_
          || cfg->amg_trunc_factor != amg_trunc_factor_
          || cfg->amg_max_levels != amg_max_levels_
          || cfg->amg_cycle_type != amg_cycle_type_
          || cfg->amg_max_coarse_size != amg_max_coarse_size_
          || cfg->amg_coarse_relax_type != amg_coarse_relax_type_
          || cfg->amg_relax_wt != amg_relax_wt_;
      const bool ilu_fill_changed = cfg->ilu_fill_level != ilu_fill_level_;
      const bool amg_budget_changed = cfg->amg_max_iters != amg_max_iters_;
      const bool weights_changed = cfg->weight_scheme != weight_scheme_;

      // Store everything (hot fields become effective at the next
      // setup()/solve() through the stored members).
      amg_max_iters_ = cfg->amg_max_iters;
      ilu_fill_level_ = cfg->ilu_fill_level;
      weight_scheme_ = cfg->weight_scheme;
      stage2_type_ = cfg->stage2_type;
      eager_adjoint_ = cfg->eager_adjoint;
      set_pressure_amg_options(cfg->amg_coarsen_type, cfg->amg_interp_type,
          cfg->amg_relax_type, cfg->amg_relax_order, cfg->amg_num_sweeps,
          cfg->amg_strong_threshold, cfg->amg_agg_num_levels,
          cfg->amg_agg_interp_type, cfg->amg_agg_pmax_elmts,
          cfg->amg_pmax_elmts, cfg->amg_trunc_factor, cfg->amg_max_levels,
          cfg->amg_cycle_type, cfg->amg_max_coarse_size,
          cfg->amg_coarse_relax_type, cfg->amg_relax_wt);
      set_reuse_amg_hierarchy(cfg->reuse_amg_hierarchy);
      set_adaptive_amg_rebuild(cfg->adaptive_amg_rebuild,
          cfg->adaptive_iter_threshold, cfg->adaptive_consecutive_bad);

      if (first_setup_)
        return 0; // not built yet -- everything applies at the first setup

      // Warm path on a live solver: re-apply the options to the existing
      // HYPRE handles (BoomerAMG reads them at Setup time) and force a
      // hierarchy rebuild on the next setup(). The bound matrices, the
      // scalar expansion and the workspace are untouched. A weights change
      // reshapes A_p's VALUES only (same sparsity) -- the next setup()
      // recomputes them anyway, but the AMG hierarchy should be rebuilt to
      // match, hence the same force flag.
      if (amg_profile_changed)
      {
        apply_pressure_amg_options(amg_, "");
        if (amg_T_ != nullptr)
          apply_pressure_amg_options(amg_T_, "(T)");
        force_amg_rebuild_ = true;
      }
      else if (amg_budget_changed)
      {
        // Hot: the V-cycle budget is read per solve.
        check_hypre(HYPRE_BoomerAMGSetMaxIter(amg_, amg_max_iters_),
            "BoomerAMGSetMaxIter(reconfigure)");
        if (amg_T_ != nullptr)
          check_hypre(HYPRE_BoomerAMGSetMaxIter(amg_T_, amg_max_iters_),
              "BoomerAMGSetMaxIter(T,reconfigure)");
      }
      if (weights_changed)
        force_amg_rebuild_ = true;
      if (ilu_fill_changed)
      {
        if (stage2_type_ == 0 && ilu_ != nullptr)
        {
          check_hypre(HYPRE_ILUSetLevelOfFill(ilu_, ilu_fill_level_),
              "ILUSetLevelOfFill(reconfigure)");
          force_amg_rebuild_ = true;
        }
        if (ilu_T_ != nullptr)
        {
          check_hypre(HYPRE_ILUSetLevelOfFill(ilu_T_, ilu_fill_level_),
              "ILUSetLevelOfFill(T,reconfigure)");
          force_amg_rebuild_ = true;
        }
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::create_pressure_amg(HYPRE_Solver &amg,
        const char *tag)
    {
      const std::string t(tag);
      check_hypre(HYPRE_BoomerAMGCreate(&amg),
          ("BoomerAMGCreate" + t).c_str());
      // NOTE: systems / unknown-based AMG (HYPRE_BoomerAMGSetNumFunctions) is
      // deliberately NOT enabled. The operator handed to BoomerAMG here is the
      // scalar (1 DOF/cell) pressure system produced by the CPR True-IMPES
      // reduction (build_pressure_subsystem), not the coupled block system --
      // so it is already a single-function scalar problem. Do not "fix" this by
      // calling SetNumFunctions; the block coupling is handled outside HYPRE.
      apply_pressure_amg_options(amg, tag);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::apply_pressure_amg_options(HYPRE_Solver &amg,
        const char *tag)
    {
      const std::string t(tag);
      check_hypre(HYPRE_BoomerAMGSetPrintLevel(amg, 0),
          ("BoomerAMGSetPrintLevel" + t).c_str());
      check_hypre(HYPRE_BoomerAMGSetLogging(amg, 0),
          ("BoomerAMGSetLogging" + t).c_str());
      check_hypre(HYPRE_BoomerAMGSetMaxIter(amg, amg_max_iters_),
          ("BoomerAMGSetMaxIter" + t).c_str());
      check_hypre(HYPRE_BoomerAMGSetTol(amg, 0.0),
          ("BoomerAMGSetTol" + t).c_str());
      // Negative option values leave the HYPRE built-in default untouched
      // (see cpr_solver_config).
      if (amg_coarsen_type_ >= 0)
        check_hypre(HYPRE_BoomerAMGSetCoarsenType(amg, amg_coarsen_type_),
            ("BoomerAMGSetCoarsenType" + t).c_str());
      if (amg_interp_type_ >= 0)
        check_hypre(HYPRE_BoomerAMGSetInterpType(amg, amg_interp_type_),
            ("BoomerAMGSetInterpType" + t).c_str());
      if (amg_relax_type_ >= 0)
        check_hypre(HYPRE_BoomerAMGSetRelaxType(amg, amg_relax_type_),
            ("BoomerAMGSetRelaxType" + t).c_str());
      if (amg_relax_order_ >= 0)
        check_hypre(HYPRE_BoomerAMGSetRelaxOrder(amg, amg_relax_order_),
            ("BoomerAMGSetRelaxOrder" + t).c_str());
      if (amg_num_sweeps_ > 0)
        check_hypre(HYPRE_BoomerAMGSetNumSweeps(amg, amg_num_sweeps_),
            ("BoomerAMGSetNumSweeps" + t).c_str());
      if (amg_strong_threshold_ >= 0.0)
        check_hypre(
            HYPRE_BoomerAMGSetStrongThreshold(amg, amg_strong_threshold_),
            ("BoomerAMGSetStrongThreshold" + t).c_str());
      if (amg_agg_num_levels_ >= 0)
        check_hypre(HYPRE_BoomerAMGSetAggNumLevels(amg, amg_agg_num_levels_),
            ("BoomerAMGSetAggNumLevels" + t).c_str());
      if (amg_agg_num_levels_ > 0)
      {
        check_hypre(HYPRE_BoomerAMGSetAggPMaxElmts(amg, amg_agg_pmax_elmts_),
            ("BoomerAMGSetAggPMaxElmts" + t).c_str());
        check_hypre(HYPRE_BoomerAMGSetAggInterpType(amg, amg_agg_interp_type_),
            ("BoomerAMGSetAggInterpType" + t).c_str());
      }
      if (amg_pmax_elmts_ >= 0)
        check_hypre(HYPRE_BoomerAMGSetPMaxElmts(amg, amg_pmax_elmts_),
            ("BoomerAMGSetPMaxElmts" + t).c_str());
      if (amg_trunc_factor_ >= 0.0)
        check_hypre(HYPRE_BoomerAMGSetTruncFactor(amg, amg_trunc_factor_),
            ("BoomerAMGSetTruncFactor" + t).c_str());
      if (amg_max_levels_ > 0)
        check_hypre(HYPRE_BoomerAMGSetMaxLevels(amg, amg_max_levels_),
            ("BoomerAMGSetMaxLevels" + t).c_str());
      if (amg_cycle_type_ > 0)
        check_hypre(HYPRE_BoomerAMGSetCycleType(amg, amg_cycle_type_),
            ("BoomerAMGSetCycleType" + t).c_str());
      if (amg_max_coarse_size_ > 0)
        check_hypre(HYPRE_BoomerAMGSetMaxCoarseSize(amg, amg_max_coarse_size_),
            ("BoomerAMGSetMaxCoarseSize" + t).c_str());
      if (amg_coarse_relax_type_ >= 0)
        check_hypre(
            HYPRE_BoomerAMGSetCycleRelaxType(amg, amg_coarse_relax_type_, 3),
            ("BoomerAMGSetCycleRelaxType" + t).c_str());
      if (amg_relax_wt_ >= 0.0)
        check_hypre(HYPRE_BoomerAMGSetRelaxWt(amg, amg_relax_wt_),
            ("BoomerAMGSetRelaxWt" + t).c_str());
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::create_fullsystem_ilu(HYPRE_Solver &ilu,
        const char *tag)
    {
      const std::string t(tag);
      check_hypre(HYPRE_ILUCreate(&ilu), ("ILUCreate" + t).c_str());
      check_hypre(HYPRE_ILUSetPrintLevel(ilu, 0),
          ("ILUSetPrintLevel" + t).c_str());
      check_hypre(HYPRE_ILUSetLogging(ilu, 0), ("ILUSetLogging" + t).c_str());
      check_hypre(HYPRE_ILUSetMaxIter(ilu, 1), ("ILUSetMaxIter" + t).c_str());
      check_hypre(HYPRE_ILUSetTol(ilu, 0.0), ("ILUSetTol" + t).c_str());
      check_hypre(HYPRE_ILUSetType(ilu, 0), ("ILUSetType" + t).c_str());
      check_hypre(HYPRE_ILUSetLevelOfFill(ilu, ilu_fill_level_),
          ("ILUSetLevelOfFill" + t).c_str());
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_cpr<N_BLOCK_SIZE>::activate_adjoint_chain()
    {
      if (adjoint_active_)
        return;
      cpr_scoped_timer timer(cpr_sub_timer(this->timer_setup, "CPR adjoint chain"));

      // The block-ILU0 forward stage skips the scalar expansion; the adjoint
      // ILU factorisation of A_s^T needs it, so materialise it now from the
      // matrix of the last setup().
      if (stage2_type_ == 1)
        refresh_scalar_expansion(A_);
      // The forward stage reads scalar values from the adapter (not As_), so
      // materialise the As_ shell now for the transpose below (no-op on the
      // legacy path, where refresh_scalar_expansion already filled As_).
      ensure_As_shell();

      // Transpose twins of the current pressure / full-system matrices. The
      // forward Ap_ / As_ always hold the values of the last setup(), so the
      // chain built here matches the matrix the engine just assembled.
      if (!Ap_T_)
        Ap_T_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      csr_transpose_scalar(*Ap_, *Ap_T_);
      if (!As_T_)
        As_T_ = std::make_unique<opendarts::linear_solvers::csr_matrix<1>>();
      csr_transpose_scalar(*As_, *As_T_);

      // Transpose AMG on A_p^T -- a separate hierarchy because
      // HYPRE_BoomerAMGSolveT supports only relax types 7 / 9 and interacts
      // poorly with the flow-tuned configuration.
      if (!amg_T_)
      {
        build_hypre_ij(*Ap_T_, Ap_T_ij_, Ap_T_parcsr_);
        create_hypre_vectors(Ap_T_->n_rows, amg_T_b_ij_, amg_T_x_ij_);
        create_pressure_amg(amg_T_, "(T)");
      }
      else
      {
        refresh_hypre_ij(*Ap_T_, Ap_T_ij_, Ap_T_parcsr_);
      }
      check_hypre(
          HYPRE_BoomerAMGSetup(amg_T_, Ap_T_parcsr_, amg_T_b_par_, amg_T_x_par_),
          "BoomerAMGSetup(T)");
      amg_T_setup_done_ = true;

      // Transpose HYPRE_ILU on A_s^T (HYPRE_ILU has no transpose-solve entry
      // point, so the adjoint path uses a second factorisation).
      if (!ilu_T_)
      {
        build_hypre_ij(*As_T_, As_T_ij_, As_T_parcsr_);
        create_hypre_vectors(As_T_->n_rows, ilu_T_b_ij_, ilu_T_x_ij_);
        create_fullsystem_ilu(ilu_T_, "(T)");
      }
      else
      {
        refresh_hypre_ij(*As_T_, As_T_ij_, As_T_parcsr_);
      }
      check_hypre(
          HYPRE_ILUSetup(ilu_T_, As_T_parcsr_, ilu_T_b_par_, ilu_T_x_par_),
          "ILUSetup(T)");
      ilu_T_setup_done_ = true;

      adjoint_active_ = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::setup_unguarded(csr_matrix_base *A_input)
    {
      // Ensure HYPRE is initialised; when CPR is used without MGR alive in
      // the process the wrappers would otherwise hit "[Generic error]" out of
      // HYPRE's diagnostic layer.
      if (!HYPRE_Initialized())
        HYPRE_Initialize();
      HYPRE_ClearAllErrors();

      A_ = A_input;

      // Rebuild the (block-extracted) pressure subsystem and the scalar
      // expansion of the full system. Both are CSR<1> objects owned here;
      // their sparsity pattern is reused across nonlinear iterations. The
      // transposed twins (A_p^T / A_s^T, CPRA adjoint path) are handled by
      // activate_adjoint_chain() / the refresh branch below -- they are not
      // touched until the adjoint chain is active.
      {
        cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR pressure system"));
        build_pressure_subsystem(A_input);
      }

      // The scalar expansion feeds the HYPRE scalar-ILU stage and the
      // adjoint chain; with the block-ILU0 stage and no active adjoint it is
      // skipped entirely (the block stage factors the block matrix in
      // place).
      if (stage2_type_ == 0 || eager_adjoint_ || adjoint_active_)
      {
        cpr_scoped_timer expand_timer(
            cpr_sub_timer(this->timer_setup, "CPR scalar expand"));
        refresh_scalar_expansion(A_input);
      }

      if (first_setup_)
      {
        // First setup -- create the forward HYPRE handles fresh. Subsequent
        // setup() calls reuse them; a destroy/recreate cycle on every Newton
        // iteration crashes BoomerAMGSetup on the second call (HYPRE
        // accumulates state that BoomerAMGDestroy doesn't fully release).
        build_hypre_ij(*Ap_, Ap_ij_, Ap_parcsr_);
        create_hypre_vectors(Ap_->n_rows, amg_b_ij_, amg_x_ij_);

        create_pressure_amg(amg_, "");
        {
          cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR AMG setup"));
          check_hypre(
              HYPRE_BoomerAMGSetup(amg_, Ap_parcsr_, amg_b_par_, amg_x_par_),
              "BoomerAMGSetup");
        }
        amg_setup_done_ = true;

        if (stage2_type_ == 1)
        {
          // Full-system stage: in-tree block ILU(0) on the block-CSR matrix
          // (default csr_ilu_prec equivalent) -- no scalar expansion involved.
          cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR BILU0 setup"));
          if (!bilu0_)
            bilu0_ = std::make_unique<
                opendarts::linear_solvers::cpr_block_ilu0<N_BLOCK_SIZE>>();
          bilu0_ready_ = (bilu0_->factor(A_input) == 0);
          if (!bilu0_ready_)
          {
            // factor() returns -1 when a block row has no diagonal block. The
            // solve dispatch would then match neither stage-2 branch and return
            // the pressure correction alone as a valid result. Report failure so
            // the Newton loop cuts the timestep instead.
            std::cerr << "linsolv_cpr: block ILU(0) factorisation failed (a block row "
                         "has no diagonal block); CPR cannot form its second stage"
                      << std::endl;
            return -1;
          }
        }
        else
        {
          // Forward HYPRE_ILU on A_s (full-system scalar expansion). Feed the
          // scalar_csr_adapter's borrowed arrays directly (block path) -- no As_
          // shell copy -- falling back to the As_ shell on the legacy path.
          const index_t as_n = scalar_adapter_ ? scalar_adapter_->n_rows() : As_->n_rows;
          const index_t *as_rp = scalar_adapter_ ? scalar_adapter_->row_ptr() : As_->rows_ptr.data();
          const index_t *as_ci = scalar_adapter_ ? scalar_adapter_->col_ind() : As_->cols_ind.data();
          const opendarts::config::mat_float *as_v = scalar_adapter_ ? scalar_adapter_->values() : As_->values.data();
          build_hypre_ij(as_n, as_rp, as_ci, as_v, As_ij_, As_parcsr_);
          create_hypre_vectors(as_n, ilu_b_ij_, ilu_x_ij_);

          create_fullsystem_ilu(ilu_, "");
          {
            cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR ILU(0) setup"));
            check_hypre(HYPRE_ILUSetup(ilu_, As_parcsr_, ilu_b_par_, ilu_x_par_),
                "ILUSetup");
          }
          ilu_setup_done_ = true;
        }

        // Transpose (CPRA / adjoint) chain: built here only on the eager
        // path. The default lazy path defers it to the first
        // solve_transposed() call -- forward-only simulations then never pay
        // for the A_p^T / A_s^T transposes and their hierarchies.
        if (eager_adjoint_)
          activate_adjoint_chain();

        first_setup_ = false;
      }
      else
      {
        // Subsequent setup -- refresh IJ matrix values (always: values change
        // every Newton iteration) and then optionally re-run *Setup to rebuild
        // the AMG hierarchy / ILU factorisation in place.
        //
        // Hierarchy reuse policy mirrors mgr::SolverParameters' BCSR-CPR knobs:
        //   - default (reuse_amg_hierarchy_ == false): rebuild every Newton.
        //     This preserves the conservative pre-policy behaviour.
        //   - reuse_amg_hierarchy_ == true: skip *Setup; the existing
        //     hierarchy is reused with the refreshed values. Coefficients have
        //     changed (the IJ values have been refreshed) but the algebraic
        //     hierarchy from the previous Newton iteration is reused -- this
        //     is the common "AMG hierarchy reuse" trick.
        //   - adaptive_amg_rebuild_: even with reuse on, force a rebuild when
        //     the last solve took more than adaptive_iter_threshold_ iters
        //     for adaptive_consecutive_bad_ solves in a row.
        {
          cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR IJ refresh"));
          refresh_hypre_ij(*Ap_, Ap_ij_, Ap_parcsr_);
          if (stage2_type_ == 0)
          {
            // Forward As stage: refresh from the adapter directly (block path),
            // no As_ shell copy; fall back to As_ on the legacy path.
            const index_t as_n = scalar_adapter_ ? scalar_adapter_->n_rows() : As_->n_rows;
            const index_t *as_rp = scalar_adapter_ ? scalar_adapter_->row_ptr() : As_->rows_ptr.data();
            const index_t *as_ci = scalar_adapter_ ? scalar_adapter_->col_ind() : As_->cols_ind.data();
            const opendarts::config::mat_float *as_v = scalar_adapter_ ? scalar_adapter_->values() : As_->values.data();
            refresh_hypre_ij(as_n, as_rp, as_ci, as_v, As_ij_, As_parcsr_);
          }
          if (adjoint_active_)
          {
            // Keep the transpose twins in sync with the refreshed values. The
            // forward stage no longer fills As_, so materialise it for the
            // transpose below (no-op on the legacy path).
            ensure_As_shell();
            csr_transpose_scalar(*Ap_, *Ap_T_);
            csr_transpose_scalar(*As_, *As_T_);
            refresh_hypre_ij(*Ap_T_, Ap_T_ij_, Ap_T_parcsr_);
            refresh_hypre_ij(*As_T_, As_T_ij_, As_T_parcsr_);
          }
        }

        bool rebuild = !reuse_amg_hierarchy_ || force_amg_rebuild_;
        if (reuse_amg_hierarchy_ && adaptive_amg_rebuild_ && last_outer_iters_ > 0)
        {
          if (last_outer_iters_ > adaptive_iter_threshold_)
          {
            consecutive_bad_streak_++;
            if (consecutive_bad_streak_ >= adaptive_consecutive_bad_)
            {
              rebuild = true;
              consecutive_bad_streak_ = 0;
            }
          }
          else
          {
            consecutive_bad_streak_ = 0;
          }
        }
        force_amg_rebuild_ = false;

        if (rebuild)
        {
          {
            cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR AMG setup"));
            check_hypre(
                HYPRE_BoomerAMGSetup(amg_, Ap_parcsr_, amg_b_par_, amg_x_par_),
                "BoomerAMGSetup(re)");
            if (adjoint_active_)
              check_hypre(
                  HYPRE_BoomerAMGSetup(amg_T_, Ap_T_parcsr_, amg_T_b_par_,
                      amg_T_x_par_),
                  "BoomerAMGSetup(T,re)");
          }
          if (stage2_type_ == 1)
          {
            cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR BILU0 setup"));
            bilu0_ready_ = bilu0_ && (bilu0_->factor(A_input) == 0);
            if (!bilu0_ready_)
            {
              std::cerr << "linsolv_cpr: block ILU(0) refactorisation failed; CPR "
                           "cannot form its second stage" << std::endl;
              return -1;
            }
          }
          else
          {
            cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR ILU(0) setup"));
            check_hypre(
                HYPRE_ILUSetup(ilu_, As_parcsr_, ilu_b_par_, ilu_x_par_),
                "ILUSetup(re)");
          }
          if (adjoint_active_)
          {
            cpr_scoped_timer t(cpr_sub_timer(this->timer_setup, "CPR ILU(0) setup"));
            check_hypre(
                HYPRE_ILUSetup(ilu_T_, As_T_parcsr_, ilu_T_b_par_, ilu_T_x_par_),
                "ILUSetup(T,re)");
          }
        }
      }

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
    int linsolv_cpr<N_BLOCK_SIZE>::solve_unguarded(mat_float *B, mat_float *X)
    {
      if (!A_ || !amg_setup_done_)
        return -1;

      const index_t n_block_rows = A_->n_rows;
      // A valid matrix never has a negative row count; asserting the invariant
      // keeps the signed-to-size_t cast below from looking unbounded to the
      // optimizer (otherwise n_scalar * sizeof(mat_float) trips
      // -Wstringop-overflow on the memsets).
      if (n_block_rows < 0)
        return -1;
      const std::size_t n_scalar =
          static_cast<std::size_t>(n_block_rows) * N_BLOCK_SIZE;
      const std::size_t n_pressure = static_cast<std::size_t>(n_block_rows);

      mat_float *x_g = wksp_.data();
      mat_float *r_m = x_g + n_scalar;
      mat_float *x_f = r_m + n_scalar;
      mat_float *r_p = x_f + n_scalar;
      mat_float *x_p = r_p + n_pressure;

      // Stage 1: pressure correction. Restrict B to the pressure subsystem
      // with the true-IMPES weights (R_i = w_i^T): r_p[i] = sum_v w_i[v] B[i,v]
      // (consistent with A_p = R A C built in build_pressure_subsystem), solve
      // A_p x_p = r_p with AMG, prolong to x_g (pressure component; zero
      // elsewhere).
      for (index_t i = 0; i < n_block_rows; ++i)
      {
        const mat_float *w = &cpr_weights_[static_cast<std::size_t>(i) * N_BLOCK_SIZE];
        mat_float acc = 0.0;
        for (int v = 0; v < static_cast<int>(N_BLOCK_SIZE); ++v)
          acc += w[v] * B[static_cast<std::size_t>(i) * N_BLOCK_SIZE + v];
        // rhs_mults_ carries the A_p row sign normalisation (see
        // build_pressure_subsystem): scaling row i of A_p and entry i of the
        // restricted residual identically leaves the pressure solution
        // unchanged.
        r_p[i] = rhs_mults_[i] * acc;
      }
      std::memset(x_p, 0, n_pressure * sizeof(mat_float));

      {
        cpr_scoped_timer t(cpr_sub_timer(this->timer_solve, "CPR AMG"));
        set_hypre_vector(amg_b_ij_, n_block_rows, r_p, amg_b_par_);
        set_hypre_vector(amg_x_ij_, n_block_rows, x_p, amg_x_par_);
        check_hypre(HYPRE_BoomerAMGSolve(amg_, Ap_parcsr_, amg_b_par_, amg_x_par_),
            "BoomerAMGSolve");
        // row_indices_ was already grown to at least n_block_rows by
        // set_hypre_vector above; reuse it for HYPRE_IJVectorGetValues.
        check_hypre(HYPRE_IJVectorGetValues(amg_x_ij_, n_block_rows,
                        row_indices_.data(), x_p),
            "IJVectorGetValues(amg_x)");
      }

      std::memset(x_g, 0, n_scalar * sizeof(mat_float));
      for (index_t i = 0; i < n_block_rows; ++i)
        x_g[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR] = x_p[i];

      // Stage 2: full-system smoothing on r_m = B - A x_g.
      std::memcpy(r_m, B, n_scalar * sizeof(mat_float));
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      block_csr_spmv_add<N_BLOCK_SIZE>(A_, x_g, x_f);
      for (std::size_t k = 0; k < n_scalar; ++k)
        r_m[k] -= x_f[k];

      // Apply the full-system smoother to r_m for x_f. Stage 2 must be
      // available: without it x_f stays zero and X would silently degrade to the
      // pressure-only correction.
      if (stage2_type_ == 1 ? !bilu0_ready_ : !ilu_setup_done_)
        return -1;
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      if (stage2_type_ == 1 && bilu0_ready_)
      {
        cpr_scoped_timer t(cpr_sub_timer(this->timer_solve, "CPR BILU0"));
        bilu0_->apply(r_m, x_f);
      }
      else if (ilu_setup_done_)
      {
        cpr_scoped_timer t(cpr_sub_timer(this->timer_solve, "CPR ILU(0)"));
        const index_t n_s = static_cast<index_t>(n_scalar);
        set_hypre_vector(ilu_b_ij_, n_s, r_m, ilu_b_par_);
        set_hypre_vector(ilu_x_ij_, n_s, x_f, ilu_x_par_);
        check_hypre(HYPRE_ILUSolve(ilu_, As_parcsr_, ilu_b_par_, ilu_x_par_),
            "ILUSolve");
        check_hypre(HYPRE_IJVectorGetValues(ilu_x_ij_, n_s,
                        row_indices_.data(), x_f),
            "IJVectorGetValues(ilu_x)");
      }

      // X = x_g + x_f.
      for (std::size_t k = 0; k < n_scalar; ++k)
        X[k] = x_g[k] + x_f[k];

      n_iters_ = 1;  // CPR as a preconditioner: one application
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::solve_transposed_unguarded(mat_float *B, mat_float *X)
    {
      if (!A_)
        return -1;
      // Lazy CPRA: the transpose hierarchies are built on the first
      // transposed solve (forward-only simulations never pay for them). The
      // chain is refreshed by every subsequent setup().
      if (!adjoint_active_)
        activate_adjoint_chain();
      if (!amg_T_setup_done_ || !ilu_T_setup_done_)
        return -1;

      // CPRA (Han et al. 2013, eq 13). Each stage is the transpose of the
      // matching forward stage; the order is reversed:
      //   1. x_f = M̃^{-T} r              -- HYPRE_ILUSolve on A_s^T
      //   2. r_m = r - (Ã)^T x_f         -- block-CSR transpose SpMV
      //   3. r_p = C^T r_m                -- restrict to pressure
      //   4. x_p = (A_p^T)^{-1} r_p      -- HYPRE_BoomerAMGSolve on A_p^T
      //   5. X = C x_p + x_f             -- prolong + add
      const index_t n_block_rows = A_->n_rows;
      // See solve(): the non-negativity guard bounds the signed-to-size_t cast
      // so the per-stage memsets don't trip -Wstringop-overflow.
      if (n_block_rows < 0)
        return -1;
      const std::size_t n_scalar =
          static_cast<std::size_t>(n_block_rows) * N_BLOCK_SIZE;
      const std::size_t n_pressure = static_cast<std::size_t>(n_block_rows);

      mat_float *x_g = wksp_.data();       // unused on transpose path
      mat_float *r_m = x_g + n_scalar;
      mat_float *x_f = r_m + n_scalar;
      mat_float *r_p = x_f + n_scalar;
      mat_float *x_p = r_p + n_pressure;

      const index_t n_s = static_cast<index_t>(n_scalar);
      ensure_row_indices(n_s);

      // Step 1: x_f = (A_s^T)^{-1} r
      std::memset(x_f, 0, n_scalar * sizeof(mat_float));
      set_hypre_vector(ilu_T_b_ij_, n_s, B, ilu_T_b_par_);
      set_hypre_vector(ilu_T_x_ij_, n_s, x_f, ilu_T_x_par_);
      check_hypre(HYPRE_ILUSolve(ilu_T_, As_T_parcsr_, ilu_T_b_par_, ilu_T_x_par_),
          "ILUSolve(T)");
      check_hypre(HYPRE_IJVectorGetValues(ilu_T_x_ij_, n_s,
                      row_indices_.data(), x_f),
          "IJVectorGetValues(ilu_T_x)");

      // Step 2: r_m = r - A^T x_f. x_g is unused on the transpose path --
      // reuse it as the SpMV accumulator instead of a per-apply heap
      // allocation.
      std::memcpy(r_m, B, n_scalar * sizeof(mat_float));
      std::memset(x_g, 0, n_scalar * sizeof(mat_float));
      block_csr_spmv_t_add<N_BLOCK_SIZE>(A_, x_f, x_g);
      for (std::size_t k = 0; k < n_scalar; ++k)
        r_m[k] -= x_g[k];

      // Step 3: r_p = C^T r_m (pressure restriction).
      for (index_t i = 0; i < n_block_rows; ++i)
        r_p[i] = r_m[static_cast<std::size_t>(i) * N_BLOCK_SIZE + P_VAR];
      std::memset(x_p, 0, n_pressure * sizeof(mat_float));

      // Step 4: x_p = (A_p^T)^{-1} r_p via the AMG hierarchy on A_p^T.
      set_hypre_vector(amg_T_b_ij_, n_block_rows, r_p, amg_T_b_par_);
      set_hypre_vector(amg_T_x_ij_, n_block_rows, x_p, amg_T_x_par_);
      check_hypre(
          HYPRE_BoomerAMGSolve(amg_T_, Ap_T_parcsr_, amg_T_b_par_, amg_T_x_par_),
          "BoomerAMGSolve(T)");
      check_hypre(HYPRE_IJVectorGetValues(amg_T_x_ij_, n_block_rows,
                      row_indices_.data(), x_p),
          "IJVectorGetValues(amg_T_x)");
      // The stored A_p carries the row sign normalisation D_m (see
      // build_pressure_subsystem): A_p = D_m A_p_orig, so A_p^T = A_p_orig^T
      // D_m and the true transpose solution is z = D_m y for the y computed
      // above. Apply the +-1 multipliers to recover it.
      for (index_t i = 0; i < n_block_rows; ++i)
        x_p[i] *= rhs_mults_[i];

      // Step 5: X = R^T x_p + x_f. The transpose of the forward weighted
      // restriction R_i = w_i^T is a weighted prolongation: X[i,v] += w_i[v]
      // x_p[i] (consistent with Ap_T_ = transpose of the weighted A_p).
      std::memcpy(X, x_f, n_scalar * sizeof(mat_float));
      for (index_t i = 0; i < n_block_rows; ++i)
      {
        const mat_float *w = &cpr_weights_[static_cast<std::size_t>(i) * N_BLOCK_SIZE];
        for (int v = 0; v < static_cast<int>(N_BLOCK_SIZE); ++v)
          X[static_cast<std::size_t>(i) * N_BLOCK_SIZE + v] += w[v] * x_p[i];
      }

      n_iters_ = 1;
      return 0;
    }

    // Explicit instantiations for the block sizes the engine uses.
    // ---- Guarded public entry points --------------------------------------
    // check_hypre() throws on a HYPRE failure (degenerate pressure subsystem,
    // NaNs from a diverging Newton step). These wrappers translate that into
    // the nonzero return engine_base::solve_linear_equation() expects, so the
    // Newton loop cuts the timestep instead of the process dying. try/catch
    // is zero-cost on the non-throwing hot path.
    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::setup(csr_matrix_base *A_input)
    {
      try
      {
        return setup_unguarded(A_input);
      }
      catch (const std::exception &e)
      {
        std::cerr << "linsolv_cpr::setup failed: " << e.what()
                  << " -- reporting failure to the Newton loop" << std::endl;
        return -1;
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::solve(mat_float *B, mat_float *X)
    {
      try
      {
        return solve_unguarded(B, X);
      }
      catch (const std::exception &e)
      {
        std::cerr << "linsolv_cpr::solve failed: " << e.what()
                  << " -- reporting failure to the Newton loop" << std::endl;
        return -1;
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_cpr<N_BLOCK_SIZE>::solve_transposed(mat_float *B, mat_float *X)
    {
      try
      {
        return solve_transposed_unguarded(B, X);
      }
      catch (const std::exception &e)
      {
        std::cerr << "linsolv_cpr::solve_transposed failed: " << e.what()
                  << " -- reporting failure to the Newton loop" << std::endl;
        return -1;
      }
    }

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see solvers/src/CMakeLists.txt.
    // Edit the block-size range there, not here.
#include "linsolv_cpr_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts
