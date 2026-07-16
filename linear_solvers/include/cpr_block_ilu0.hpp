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
#ifndef OPENDARTS_LINEAR_SOLVERS_CPR_BLOCK_ILU0_HPP
#define OPENDARTS_LINEAR_SOLVERS_CPR_BLOCK_ILU0_HPP
//--------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "csr_matrix_base.hpp"
#include "data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Block ILU(0) factorisation of a block-CSR system.
     *
     *  The in-tree equivalent of the proprietary ``csr_ilu_prec<N>`` used as
     *  the full-system stage of the legacy BOS CPR: an incomplete block LU
     *  with no fill, dense N x N diagonal-block inverses, sequential
     *  forward/backward block substitution. Compared with a scalar ILU(0) on
     *  the expanded system, the block variant treats each cell's inter-phase
     *  coupling exactly (dense block inverse), which is what the reservoir
     *  CPR literature (and the proprietary implementation) relies on.
     *
     *  Structure (column ordering, diagonal positions) is prepared once and
     *  reused as long as the bound matrix keeps its sparsity; factor()
     *  re-runs the numerical factorisation from the current values. A
     *  near-singular (updated) diagonal block falls back to an identity
     *  block inverse -- mirroring linsolv_mgr's default BILU0 fallback -- so
     *  a handful of degenerate well blocks degrade smoothing quality locally
     *  instead of failing the whole preconditioner.
     */
    template <uint8_t N_BLOCK_SIZE>
    class cpr_block_ilu0
    {
      static constexpr int Ni = static_cast<int>(N_BLOCK_SIZE);
      static constexpr std::size_t B2 =
          static_cast<std::size_t>(Ni) * static_cast<std::size_t>(Ni);

      using index_t = opendarts::config::index_t;
      using mat_float = opendarts::config::mat_float;

      index_t n_rows_ = 0;
      const index_t *rows_ = nullptr;
      const index_t *cols_ = nullptr;
      std::vector<index_t> diag_pos_;   // per-row position of the diagonal block
      std::vector<index_t> sorted_pos_; // per-row block positions in ascending column order
      std::vector<mat_float> f_;        // factored block values (copy of A's values)
      std::vector<mat_float> dinv_;     // explicit inverses of factored diagonal blocks
      std::size_t n_fallback_ = 0;      // diagonal blocks replaced by identity

      // C = A * B (dense N x N, row-major).
      static void block_mul(const mat_float *A, const mat_float *B, mat_float *C)
      {
        for (int r = 0; r < Ni; ++r)
          for (int c = 0; c < Ni; ++c)
          {
            mat_float acc = 0.0;
            for (int k = 0; k < Ni; ++k)
              acc += A[r * Ni + k] * B[k * Ni + c];
            C[r * Ni + c] = acc;
          }
      }

      // y -= A * x (dense N x N block times N vector).
      static void block_matvec_sub(const mat_float *A, const mat_float *x,
          mat_float *y)
      {
        for (int r = 0; r < Ni; ++r)
        {
          mat_float acc = 0.0;
          for (int k = 0; k < Ni; ++k)
            acc += A[r * Ni + k] * x[k];
          y[r] -= acc;
        }
      }

      // Minv = M^{-1} via Gauss-Jordan with partial pivoting. Returns false
      // when a pivot underflows (near-singular block).
      static bool invert_dense(const mat_float *M_in, mat_float *Minv)
      {
        mat_float M[B2];
        std::copy(M_in, M_in + B2, M);
        for (int r = 0; r < Ni; ++r)
          for (int c = 0; c < Ni; ++c)
            Minv[r * Ni + c] = (r == c) ? 1.0 : 0.0;

        mat_float scale = 1.0;
        for (std::size_t k = 0; k < B2; ++k)
          scale = std::max(scale, std::abs(M[k]));
        const mat_float piv_tol =
            std::numeric_limits<mat_float>::epsilon() * scale * 100.0;

        for (int col = 0; col < Ni; ++col)
        {
          int piv = col;
          mat_float best = std::abs(M[col * Ni + col]);
          for (int r = col + 1; r < Ni; ++r)
          {
            const mat_float v = std::abs(M[r * Ni + col]);
            if (v > best)
            {
              best = v;
              piv = r;
            }
          }
          if (best <= piv_tol)
            return false;
          if (piv != col)
            for (int c = 0; c < Ni; ++c)
            {
              std::swap(M[col * Ni + c], M[piv * Ni + c]);
              std::swap(Minv[col * Ni + c], Minv[piv * Ni + c]);
            }
          const mat_float d = 1.0 / M[col * Ni + col];
          for (int c = 0; c < Ni; ++c)
          {
            M[col * Ni + c] *= d;
            Minv[col * Ni + c] *= d;
          }
          for (int r = 0; r < Ni; ++r)
          {
            if (r == col)
              continue;
            const mat_float f = M[r * Ni + col];
            if (f == 0.0)
              continue;
            for (int c = 0; c < Ni; ++c)
            {
              M[r * Ni + c] -= f * M[col * Ni + c];
              Minv[r * Ni + c] -= f * Minv[col * Ni + c];
            }
          }
        }
        return true;
      }

      // Position of block (k, c) in row k, or -1 when absent (linear scan --
      // reservoir rows hold a handful of blocks).
      index_t find_in_row(index_t k, index_t c) const
      {
        for (index_t jb = rows_[k]; jb < rows_[k + 1]; ++jb)
          if (cols_[jb] == c)
            return jb;
        return -1;
      }

    public:
      /** (Re)factor from the matrix's current values. Structure is captured
       *  on the first call (or when the bound matrix changes) and reused
       *  afterwards. Returns 0 on success, -1 when a row has no diagonal
       *  block. */
      int factor(opendarts::linear_solvers::csr_matrix_base *A)
      {
        const index_t *rows = A->get_rows_ptr();
        const index_t *cols = A->get_cols_ind();
        const mat_float *vals = A->get_values();
        const index_t n = A->n_rows;
        const index_t nnz = rows[n];

        if (n_rows_ != n || rows_ != rows || cols_ != cols)
        {
          n_rows_ = n;
          rows_ = rows;
          cols_ = cols;
          diag_pos_.assign(static_cast<std::size_t>(n), -1);
          sorted_pos_.resize(static_cast<std::size_t>(nnz));
          for (index_t i = 0; i < n; ++i)
          {
            for (index_t jb = rows[i]; jb < rows[i + 1]; ++jb)
            {
              sorted_pos_[jb] = jb;
              if (cols[jb] == i)
                diag_pos_[i] = jb;
            }
            std::sort(sorted_pos_.begin() + rows[i],
                sorted_pos_.begin() + rows[i + 1],
                [cols](index_t a, index_t b) { return cols[a] < cols[b]; });
            if (diag_pos_[i] < 0)
              return -1;
          }
          dinv_.resize(static_cast<std::size_t>(n) * B2);
        }

        f_.assign(vals, vals + static_cast<std::size_t>(nnz) * B2);
        n_fallback_ = 0;

        mat_float tmp[B2];
        for (index_t i = 0; i < n; ++i)
        {
          const index_t rb = rows[i];
          const index_t re = rows[i + 1];
          // Eliminate the strictly-lower blocks in ascending column order.
          for (index_t s = rb; s < re; ++s)
          {
            const index_t jb = sorted_pos_[s];
            const index_t k = cols_[jb];
            if (k >= i)
              break; // sorted: strictly-lower part exhausted
            // L_ik = A_ik * D_k^{-1}
            mat_float *Lik = f_.data() + static_cast<std::size_t>(jb) * B2;
            block_mul(Lik, dinv_.data() + static_cast<std::size_t>(k) * B2, tmp);
            std::copy(tmp, tmp + B2, Lik);
            // Row-i update: A_ic -= L_ik * U_kc for every c > k present in
            // both row i and row k (ILU(0): only existing positions).
            for (index_t s2 = s + 1; s2 < re; ++s2)
            {
              const index_t jb2 = sorted_pos_[s2];
              const index_t c = cols_[jb2];
              const index_t kb = find_in_row(k, c);
              if (kb < 0)
                continue;
              block_mul(Lik, f_.data() + static_cast<std::size_t>(kb) * B2, tmp);
              mat_float *dst = f_.data() + static_cast<std::size_t>(jb2) * B2;
              for (std::size_t e = 0; e < B2; ++e)
                dst[e] -= tmp[e];
            }
          }
          // Invert the (updated) diagonal block; identity fallback keeps the
          // preconditioner usable when a well block degenerates.
          const mat_float *D =
              f_.data() + static_cast<std::size_t>(diag_pos_[i]) * B2;
          mat_float *Dinv = dinv_.data() + static_cast<std::size_t>(i) * B2;
          if (!invert_dense(D, Dinv))
          {
            for (int r = 0; r < Ni; ++r)
              for (int c = 0; c < Ni; ++c)
                Dinv[r * Ni + c] = (r == c) ? 1.0 : 0.0;
            ++n_fallback_;
          }
        }
        return 0;
      }

      /** x = (L U)^{-1} b: unit-block-lower forward substitution followed by
       *  backward substitution with the explicit diagonal-block inverses. */
      void apply(const mat_float *b, mat_float *x) const
      {
        // Forward: y_i = b_i - sum_{k<i} L_ik y_k (y stored in x).
        for (index_t i = 0; i < n_rows_; ++i)
        {
          mat_float acc[Ni];
          const mat_float *bi = b + static_cast<std::size_t>(i) * Ni;
          for (int e = 0; e < Ni; ++e)
            acc[e] = bi[e];
          for (index_t s = rows_[i]; s < rows_[i + 1]; ++s)
          {
            const index_t jb = sorted_pos_[s];
            const index_t k = cols_[jb];
            if (k >= i)
              break;
            block_matvec_sub(f_.data() + static_cast<std::size_t>(jb) * B2,
                x + static_cast<std::size_t>(k) * Ni, acc);
          }
          mat_float *xi = x + static_cast<std::size_t>(i) * Ni;
          for (int e = 0; e < Ni; ++e)
            xi[e] = acc[e];
        }
        // Backward: x_i = D_i^{-1} (y_i - sum_{c>i} U_ic x_c).
        for (index_t i = n_rows_ - 1; i >= 0; --i)
        {
          mat_float acc[Ni];
          mat_float *xi = x + static_cast<std::size_t>(i) * Ni;
          for (int e = 0; e < Ni; ++e)
            acc[e] = xi[e];
          for (index_t s = rows_[i + 1] - 1; s >= rows_[i]; --s)
          {
            const index_t jb = sorted_pos_[s];
            const index_t c = cols_[jb];
            if (c <= i)
              break;
            block_matvec_sub(f_.data() + static_cast<std::size_t>(jb) * B2,
                x + static_cast<std::size_t>(c) * Ni, acc);
          }
          const mat_float *Dinv = dinv_.data() + static_cast<std::size_t>(i) * B2;
          for (int r = 0; r < Ni; ++r)
          {
            mat_float v = 0.0;
            for (int k = 0; k < Ni; ++k)
              v += Dinv[r * Ni + k] * acc[k];
            xi[r] = v;
          }
        }
      }

      /** Diagonal blocks replaced by the identity fallback in the last
       *  factor() (diagnostic). */
      std::size_t fallback_count() const { return n_fallback_; }
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_CPR_BLOCK_ILU0_HPP
//--------------------------------------------------------------------------
