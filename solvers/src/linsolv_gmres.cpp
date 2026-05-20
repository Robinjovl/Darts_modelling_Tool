//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator
//    (open-DARTS). It is distributed under the Apache License.
// *************************************************************************

// Open-source restarted GMRES with right preconditioning.
// See linsolv_gmres.hpp for the design rationale; algorithm follows the
// proven layout of the proprietary bos gmres_solver2.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include "linsolv_gmres.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    using opendarts::config::index_t;
    using opendarts::config::mat_float;

    namespace
    {
      // Block-CSR sparse mat-vec: r += A * v.
      // The matrix is consumed through csr_matrix_base accessors so this
      // works for both csr_matrix<N> and the unified block_csr_matrix.
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

      // r = alpha * A * u + beta * v  (matches bos mv_calc_lin_comb<N>).
      template <uint8_t N>
      inline void block_csr_lin_comb(csr_matrix_base *A,
          mat_float alpha,
          mat_float beta,
          const mat_float *u,
          const mat_float *v,
          mat_float *r,
          std::size_t n_scalar)
      {
        const mat_float eps = 1.0e-12;
        if (std::fabs(beta) > eps)
        {
          mat_float d = beta;
          if (std::fabs(alpha) > eps)
            d /= alpha;
          for (std::size_t i = 0; i < n_scalar; ++i)
            r[i] = v[i] * d;
        }
        else
        {
          std::memset(r, 0, n_scalar * sizeof(mat_float));
        }
        if (std::fabs(alpha) > eps)
        {
          block_csr_spmv_add<N>(A, u, r);
          if (alpha != 1.0)
            for (std::size_t i = 0; i < n_scalar; ++i)
              r[i] *= alpha;
        }
      }

      inline mat_float dot(const mat_float *a, const mat_float *b, std::size_t n)
      {
        mat_float s = 0;
        for (std::size_t i = 0; i < n; ++i)
          s += a[i] * b[i];
        return s;
      }

      inline void axpy(mat_float *y, mat_float a, const mat_float *x, std::size_t n)
      {
        for (std::size_t i = 0; i < n; ++i)
          y[i] += a * x[i];
      }

      inline void scale(mat_float *y, mat_float a, std::size_t n)
      {
        for (std::size_t i = 0; i < n; ++i)
          y[i] *= a;
      }
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_gmres<N_BLOCK_SIZE>::linsolv_gmres()
      : A_(nullptr), prec_(nullptr), max_iters_(50), tolerance_(1.0e-5),
        restart_m_(30), n_iters_(0), final_resid_(0.0)
    {
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_gmres<N_BLOCK_SIZE>::~linsolv_gmres()
    {
      // The preconditioner is supplied externally; do not own it.
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres<N_BLOCK_SIZE>::set_prec(linsolv_iface *prec_input)
    {
      prec_ = prec_input;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres<N_BLOCK_SIZE>::init(csr_matrix_base *A,
        int max_iters,
        mat_float tolerance)
    {
      A_ = A;
      max_iters_ = max_iters;
      tolerance_ = tolerance;
      n_iters_ = 0;
      final_resid_ = 0.0;
      if (prec_)
        prec_->init(A, max_iters, tolerance);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres<N_BLOCK_SIZE>::setup(csr_matrix_base *A_input)
    {
      A_ = A_input;
      if (prec_)
        return prec_->setup(A_input);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres<N_BLOCK_SIZE>::solve(mat_float *rhs, mat_float *sol)
    {
      if (!A_)
        return -1;
      const int N = static_cast<int>(N_BLOCK_SIZE);
      const index_t n_block_rows = A_->n_rows;
      const std::size_t n = static_cast<std::size_t>(n_block_rows) * N;
      const int m = restart_m_;
      const int max_iter = max_iters_;
      const mat_float tol_in = static_cast<mat_float>(tolerance_);
      const mat_float epsmac = 1.0e-16;

      // Workspace layout (single contiguous block):
      //   w[n], p[(m+1) * n], r[n], s[m], c[m], rs[m+1], hh[(m+2)*(m+1)]
      // Required total: n*(m+3) + 2m + (m+1) + (m+2)*(m+1)
      //              = n*(m+3) + (m+2)*(m+1) + 3m + 1.
      const std::size_t nw = n * (m + 3) + (m + 2) * (m + 1) + 3 * m + 1;
      if (wksp_.size() < nw)
        wksp_.assign(nw, 0.0);

      mat_float *w = wksp_.data();
      mat_float *p = w + n;
      mat_float *r_buf = p + static_cast<std::size_t>(m + 1) * n;
      mat_float *s = r_buf + n;
      mat_float *c = s + m;
      mat_float *rs = c + m;
      mat_float *hh = rs + (m + 1);

      // sol = 0; p_0 = rhs - A * sol = rhs (since sol = 0)
      std::memset(sol, 0, n * sizeof(mat_float));
      block_csr_lin_comb<N_BLOCK_SIZE>(A_, -1.0, 1.0, sol, rhs, p, n);

      const mat_float b_norm = std::sqrt(dot(rhs, rhs, n));
      mat_float r_norm = std::sqrt(dot(p, p, n));

      mat_float tol_scaled, den_norm;
      if (b_norm > epsmac)
      {
        tol_scaled = tol_in * b_norm;
        den_norm = b_norm;
      }
      else
      {
        tol_scaled = tol_in * r_norm;
        den_norm = r_norm;
      }

      int iter = 0;
      int i = 0;
      while (iter < max_iter)
      {
        rs[0] = r_norm;
        if (r_norm < epsmac || r_norm <= tol_scaled)
          break;

        mat_float t = 1.0 / r_norm;
        scale(p, t, n);

        for (i = 1; i < m && iter < max_iter; ++i, ++iter)
        {
          mat_float *cur_p_i = p + static_cast<std::size_t>(i) * n;
          // r_buf = M^{-1} p_{i-1}; if no prec, r_buf = p_{i-1}.
          if (prec_)
          {
            if (prec_->solve(p + static_cast<std::size_t>(i - 1) * n, r_buf))
              return -3;
          }
          else
          {
            std::memcpy(r_buf, p + static_cast<std::size_t>(i - 1) * n,
                n * sizeof(mat_float));
          }

          // p_i = A * r_buf  (block CSR SpMV, zero-out then accumulate).
          std::memset(cur_p_i, 0, n * sizeof(mat_float));
          block_csr_spmv_add<N_BLOCK_SIZE>(A_, r_buf, cur_p_i);

          // Modified Gram-Schmidt: orthogonalise p_i against p_0..p_{i-1}.
          mat_float *cur_h = hh + static_cast<std::size_t>(i - 1) * (m + 1);
          for (int j = 0; j < i; ++j)
          {
            mat_float *cur_p_j = p + static_cast<std::size_t>(j) * n;
            cur_h[j] = dot(cur_p_j, cur_p_i, n);
            axpy(cur_p_i, -cur_h[j], cur_p_j, n);
          }
          t = std::sqrt(dot(cur_p_i, cur_p_i, n));
          cur_h[i] = t;
          if (t > epsmac)
            scale(cur_p_i, 1.0 / t, n);

          // Apply the previous Givens rotations to the new Hessenberg column.
          for (int j = 1; j < i; ++j)
          {
            t = cur_h[j - 1];
            cur_h[j - 1] = c[j - 1] * t + s[j - 1] * cur_h[j];
            cur_h[j] = -s[j - 1] * t + c[j - 1] * cur_h[j];
          }
          mat_float gamma = std::sqrt(cur_h[i - 1] * cur_h[i - 1]
              + cur_h[i] * cur_h[i]);
          if (gamma < epsmac)
            gamma = epsmac;
          c[i - 1] = cur_h[i - 1] / gamma;
          s[i - 1] = cur_h[i] / gamma;
          rs[i] = -s[i - 1] * rs[i - 1];
          rs[i - 1] = c[i - 1] * rs[i - 1];

          cur_h[i - 1] = c[i - 1] * cur_h[i - 1] + s[i - 1] * cur_h[i];
          r_norm = std::fabs(rs[i]);
          if (r_norm <= tol_scaled)
            break;  // i = number of completed Arnoldi steps; do not over-increment.
        }
        if (i == m || iter == max_iter)
          i = i - 1;

        // Back-solve the upper-triangular Hessenberg system.
        rs[i - 1] = rs[i - 1] / hh[(i - 1) + static_cast<std::size_t>(i - 1) * (m + 1)];
        for (int k = i - 2; k >= 0; --k)
        {
          mat_float t2 = rs[k];
          for (int j = k + 1; j < i; ++j)
            t2 -= hh[k + static_cast<std::size_t>(j) * (m + 1)] * rs[j];
          rs[k] = t2 / hh[k + static_cast<std::size_t>(k) * (m + 1)];
        }

        // w = sum_{j=0..i-1} rs[j] * p_j
        std::memcpy(w, p, n * sizeof(mat_float));
        scale(w, rs[0], n);
        for (int j = 1; j < i; ++j)
          axpy(w, rs[j], p + static_cast<std::size_t>(j) * n, n);

        // Apply the preconditioner once more to the linear combination.
        if (prec_)
        {
          if (prec_->solve(w, r_buf))
            return -5;
        }
        else
        {
          std::memcpy(r_buf, w, n * sizeof(mat_float));
        }
        axpy(sol, 1.0, r_buf, n);

        // If predicted convergence reached, verify on the actual residual.
        if (r_norm <= tol_scaled)
        {
          block_csr_lin_comb<N_BLOCK_SIZE>(A_, -1.0, 1.0, sol, rhs, r_buf, n);
          r_norm = std::sqrt(dot(r_buf, r_buf, n));
          if (r_norm <= tol_scaled)
            break;
          // Otherwise restart with the actual residual.
          std::memcpy(p, r_buf, n * sizeof(mat_float));
          i = 0;
          ++iter;
          continue;
        }

        // Otherwise compute the residual vector for the next restart cycle.
        for (int j = i; j > 0; --j)
        {
          rs[j - 1] = -s[j - 1] * rs[j];
          rs[j] = c[j - 1] * rs[j];
        }
        if (i)
          scale(p, rs[0], n);
        for (int j = 1; j < i + 1; ++j)
          axpy(p, rs[j], p + static_cast<std::size_t>(j) * n, n);
      }

      n_iters_ = iter + 1;
      final_resid_ = (den_norm > 1.0e-12) ? (r_norm / den_norm) : r_norm;
      return 0;
    }

    // Explicit instantiations for the block sizes the engine uses.
    template class linsolv_gmres<1>;
    template class linsolv_gmres<2>;
    template class linsolv_gmres<3>;
    template class linsolv_gmres<4>;
    template class linsolv_gmres<5>;
    template class linsolv_gmres<6>;
    template class linsolv_gmres<7>;
    template class linsolv_gmres<8>;
    template class linsolv_gmres<9>;
    template class linsolv_gmres<10>;
    template class linsolv_gmres<11>;
    template class linsolv_gmres<12>;
    template class linsolv_gmres<13>;
  } // namespace linear_solvers
} // namespace opendarts
