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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef WITH_GPU
#include <cuda_runtime.h>
#endif

#include "linsolv_schur_elim.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      using opendarts::config::index_t;
      using opendarts::config::mat_float;

#ifdef WITH_GPU
      constexpr int SE_BLOCK = 256;

      // Per block row: pivot + g factors from the (device) diagonal block.
      template <uint8_t N>
      __global__ void se_factors_kernel(index_t n_rows, const index_t *diag_ind,
        const mat_float *values, const uint8_t *r_sel, const uint8_t *c_sel,
        const uint8_t *cmap, mat_float *pivot, mat_float *g, double pivot_eps,
        int *pivot_bad)
      {
        constexpr uint8_t M = N - 1;
        const index_t i = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (i >= n_rows)
          return;
        const mat_float *db = values + (size_t)diag_ind[i] * N * N;
        const uint8_t r = r_sel[i];
        const mat_float p = db[r * N + c_sel[i]];
        pivot[i] = p;
        // detection validated the pivot only on the first setup's values; flag
        // a pivot that has since degenerated so setup can fail loudly instead of
        // producing inf/NaN in the reduced matrix.
        if (!(fabs(p) > pivot_eps) || !isfinite(p))
          atomicExch(pivot_bad, 1);
        const mat_float pinv = mat_float(1) / p;
        const uint8_t *cm = cmap + (size_t)i * M;
        for (int cc = 0; cc < M; cc++)
          g[(size_t)i * M + cc] = db[r * N + cm[cc]] * pinv;
      }

      // Per block row: condense every block of the row.
      // S_kj[rr, cc] = A_kj[rmap(k)[rr], cmap(j)[cc]] - A_kj[rmap(k)[rr], c_sel(j)] * g_j[cc]
      template <uint8_t N>
      __global__ void se_condense_kernel(index_t n_rows, const index_t *rows_ptr,
        const index_t *cols_ind, const mat_float *values, const uint8_t *rmap,
        const uint8_t *cmap, const uint8_t *c_sel, const mat_float *g, mat_float *red_values)
      {
        constexpr uint8_t M = N - 1;
        const index_t k = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (k >= n_rows)
          return;
        const uint8_t *rm = rmap + (size_t)k * M;
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const index_t j = cols_ind[b];
          const mat_float *A = values + (size_t)b * N * N;
          mat_float *S = red_values + (size_t)b * M * M;
          const mat_float *gj = g + (size_t)j * M;
          const uint8_t *cm = cmap + (size_t)j * M;
          const uint8_t cs = c_sel[j];
          for (int rr = 0; rr < M; rr++)
          {
            const mat_float *Ar = A + rm[rr] * N;
            const mat_float am = Ar[cs];
            for (int cc = 0; cc < M; cc++)
              S[rr * M + cc] = Ar[cm[cc]] - am * gj[cc];
          }
        }
      }

      // Per block row: h_eff[i] = B[i][r_sel]/pivot (local rows; the O(wells)
      // chain fixups run on the host afterwards).
      template <uint8_t N>
      __global__ void se_h_kernel(index_t n_rows, const mat_float *B,
        const uint8_t *r_sel, const mat_float *pivot, mat_float *h_eff)
      {
        const index_t i = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (i >= n_rows)
          return;
        h_eff[i] = B[(size_t)i * N + r_sel[i]] / pivot[i];
      }

      // Per block row: b_red[k] = B[k][kept rows] - sum_j A_kj[kept, c_sel(j)]*h_eff[j]
      template <uint8_t N>
      __global__ void se_reduce_rhs_kernel(index_t n_rows, const index_t *rows_ptr,
        const index_t *cols_ind, const mat_float *values, const uint8_t *rmap,
        const uint8_t *c_sel, const mat_float *h_eff, const mat_float *B, mat_float *b_red)
      {
        constexpr uint8_t M = N - 1;
        const index_t k = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (k >= n_rows)
          return;
        const uint8_t *rm = rmap + (size_t)k * M;
        mat_float acc[M];
        for (int rr = 0; rr < M; rr++)
          acc[rr] = B[(size_t)k * N + rm[rr]];
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const mat_float he = h_eff[cols_ind[b]];
          const mat_float *A = values + (size_t)b * N * N;
          const uint8_t cs = c_sel[cols_ind[b]];
          for (int rr = 0; rr < M; rr++)
            acc[rr] -= A[rm[rr] * N + cs] * he;
        }
        for (int rr = 0; rr < M; rr++)
          b_red[(size_t)k * M + rr] = acc[rr];
      }

      // Per block row: scatter x_red into X (kept cols) and reconstruct the
      // eliminated unknown for LOCAL rows: x_m = h_eff - g . x_f.
      template <uint8_t N>
      __global__ void se_backsub_kernel(index_t n_rows, const mat_float *x_red,
        const uint8_t *c_sel, const uint8_t *cmap, const mat_float *g,
        const mat_float *h_eff, mat_float *X)
      {
        constexpr uint8_t M = N - 1;
        const index_t i = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (i >= n_rows)
          return;
        const mat_float *xr = x_red + (size_t)i * M;
        const mat_float *gi = g + (size_t)i * M;
        const uint8_t *cm = cmap + (size_t)i * M;
        mat_float xm = h_eff[i];
        for (int cc = 0; cc < M; cc++)
        {
          X[(size_t)i * N + cm[cc]] = xr[cc];
          xm -= gi[cc] * xr[cc];
        }
        X[(size_t)i * N + c_sel[i]] = xm;
      }
#endif // WITH_GPU
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_schur_elim<N_BLOCK_SIZE>::linsolv_schur_elim(bool on_device_, int elim_col_,
      uint8_t elim_row_, double pivot_eps_)
      : on_device(on_device_), elim_col(elim_col_), elim_row(elim_row_), pivot_eps(pivot_eps_)
    {
      static_assert(N_BLOCK_SIZE >= 2, "schur elimination needs at least a 2x2 block");
      if (elim_col == 0)
      {
        printf("linsolv_schur_elim: the pressure column (0) cannot be eliminated; "
               "using per-row auto pivot selection instead\n");
        elim_col = -1;
      }
#ifndef WITH_GPU
      if (on_device)
      {
        printf("linsolv_schur_elim: device mode requested in a CPU-only build; using host path\n");
        on_device = false;
      }
#endif
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_schur_elim<N_BLOCK_SIZE>::~linsolv_schur_elim()
    {
      free_device();
      delete reduced;
      // `inner` is intentionally NOT freed here: registry-built inners are
      // shared_ptr-owned on the Python side (the spec keeps them alive), and
      // engine-built chains follow the engine convention of never freeing
      // solver graph nodes.
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_schur_elim<N_BLOCK_SIZE>::free_device()
    {
#ifdef WITH_GPU
      for (auto **p : {&r_sel_d, &c_sel_d, &rmap_d, &cmap_d})
        if (*p)
        {
          cudaFree(*p);
          *p = nullptr;
        }
      for (auto **p : {&pivot_d, &g_d, &h_eff_d, &b_red_d, &x_red_d})
        if (*p)
        {
          cudaFree(*p);
          *p = nullptr;
        }
      if (pivot_bad_d)
      {
        cudaFree(pivot_bad_d);
        pivot_bad_d = nullptr;
      }
#endif
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_schur_elim<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A,
      opendarts::config::index_t max_iters, opendarts::config::mat_float tolerance)
    {
      constexpr uint8_t M = M_BLOCK_SIZE;

      if (!inner)
      {
        printf("linsolv_schur_elim: no inner solver attached (set_prec) before init\n");
        return 1;
      }

      A_saved = A;
      n_rows = A->n_rows;
      const index_t *rows_ptr = A->get_rows_ptr();
      n_nnz = rows_ptr[n_rows];

      // Reduced matrix: identical pattern, block size M.
      reduced = new opendarts::linear_solvers::csr_matrix<M>;
      reduced->init(n_rows, n_rows, n_nnz);
      std::memcpy(reduced->rows_ptr.data(), rows_ptr, sizeof(index_t) * (n_rows + 1));
      std::memcpy(reduced->cols_ind.data(), A->get_cols_ind(), sizeof(index_t) * n_nnz);
      reduced->diag_ind.resize(n_rows);
      std::memcpy(reduced->diag_ind.data(), A->get_diag_ind(), sizeof(index_t) * n_rows);

      r_sel.assign(n_rows, elim_row);
      c_sel.assign(n_rows, elim_col > 0 ? (uint8_t)elim_col : (uint8_t)1);
      rmap.assign((size_t)n_rows * M, 0);
      cmap.assign((size_t)n_rows * M, 0);
      pivot.assign(n_rows, 0);
      g.assign((size_t)n_rows * M, 0);
      h_eff.assign(n_rows, 0);
      b_red.assign((size_t)n_rows * M, 0);
      x_red.assign((size_t)n_rows * M, 0);

#ifdef WITH_GPU
      if (on_device)
      {
        reduced->init_device(n_rows, n_nnz);
        reduced->copy_struct_to_device();
        cudaMalloc((void **)&r_sel_d, n_rows);
        cudaMalloc((void **)&c_sel_d, n_rows);
        cudaMalloc((void **)&rmap_d, (size_t)n_rows * M);
        cudaMalloc((void **)&cmap_d, (size_t)n_rows * M);
        cudaMalloc((void **)&pivot_d, sizeof(mat_float) * n_rows);
        cudaMalloc((void **)&g_d, sizeof(mat_float) * n_rows * M);
        cudaMalloc((void **)&h_eff_d, sizeof(mat_float) * n_rows);
        cudaMalloc((void **)&b_red_d, sizeof(mat_float) * n_rows * M);
        cudaMalloc((void **)&x_red_d, sizeof(mat_float) * n_rows * M);
        cudaMalloc((void **)&pivot_bad_d, sizeof(int));
        values_h_staging.resize((size_t)n_nnz * N_BLOCK_SIZE * N_BLOCK_SIZE);
      }
#endif

      inner->init_timer_nodes(this->timer_setup, this->timer_solve);
      return inner->init(reduced, max_iters, tolerance);
    }

    // Detection: per block row, pick the eliminated (equation row, unknown
    // column) pair. Preference: the flux-free ``elim_row`` with a per-row
    // max-magnitude pivot column (or the forced ``elim_col``); rows without a
    // usable pivot fall back to other rows, preferring chain-free ones. All
    // off-diagonal entries of a selected row become one-level chains.
    template <uint8_t N_BLOCK_SIZE>
    int linsolv_schur_elim<N_BLOCK_SIZE>::detect_rows(const mat_float *values_h)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      const index_t *rows_ptr = A_saved->get_rows_ptr();
      const index_t *cols_ind = A_saved->get_cols_ind();
      const index_t *diag_ind = A_saved->get_diag_ind();

      chains.clear();
      chain_fixes.clear();
      int n_fallback = 0;

      // pass 1: per-row (row, column) selection
      for (index_t i = 0; i < n_rows; i++)
      {
        const mat_float *db = values_h + (size_t)diag_ind[i] * N * N;

        // candidate rows in preference order: elim_row first, then the rest
        int best_r = -1, best_c = -1;
        bool best_has_offdiag = true;
        for (int t = 0; t < N; t++)
        {
          const int r = (t == 0) ? (int)elim_row : (t - 1 < (int)elim_row ? t - 1 : t);
          // column choice for this row: forced or per-row max pivot over
          // the non-pressure columns
          int c = -1;
          if (elim_col > 0)
          {
            if (std::fabs(db[r * N + elim_col]) > pivot_eps)
              c = elim_col;
          }
          else
          {
            mat_float best_p = (mat_float)pivot_eps;
            for (int cand = 1; cand < N; cand++)
              if (std::fabs(db[r * N + cand]) > best_p)
              {
                best_p = std::fabs(db[r * N + cand]);
                c = cand;
              }
          }
          if (c < 0)
            continue;
          // off-diagonal footprint of row r (any entry forms a chain)
          bool has_offdiag = false;
          for (index_t b = rows_ptr[i]; b < rows_ptr[i + 1] && !has_offdiag; b++)
          {
            if (cols_ind[b] == i)
              continue;
            const mat_float *Ab = values_h + (size_t)b * N * N;
            for (int cc = 0; cc < N; cc++)
              if (Ab[r * N + cc] != 0.0)
              {
                has_offdiag = true;
                break;
              }
          }
          if (!has_offdiag)
          {
            best_r = r;
            best_c = c;
            best_has_offdiag = false;
            break; // chain-free candidate in preference order: take it
          }
          if (best_r < 0)
          {
            best_r = r; // remember the first pivot-usable row as fallback
            best_c = c;
          }
        }
        if (best_r < 0)
        {
          printf("linsolv_schur_elim: block row %d has no eliminable equation "
                 "(no row with usable pivot); disable the mineral elimination "
                 "option for this model\n", (int)i);
          return 1;
        }
        r_sel[i] = (uint8_t)best_r;
        c_sel[i] = (uint8_t)best_c;
        if (best_r != (int)elim_row)
          n_fallback++;
        int rr = 0;
        for (int r = 0; r < N; r++)
          if (r != best_r)
            rmap[(size_t)i * M + rr++] = (uint8_t)r;
        int cc = 0;
        for (int c = 0; c < N; c++)
          if (c != best_c)
            cmap[(size_t)i * M + cc++] = (uint8_t)c;
        (void)best_has_offdiag;
      }

      // pass 2: chain records (needs every dep's c_sel, hence a second pass)
      for (index_t i = 0; i < n_rows; i++)
      {
        for (index_t b = rows_ptr[i]; b < rows_ptr[i + 1]; b++)
        {
          if (cols_ind[b] == i)
            continue;
          const mat_float *Ab = values_h + (size_t)b * N * N;
          const index_t dep = cols_ind[b];
          for (int c = 0; c < N; c++)
          {
            const mat_float o = Ab[r_sel[i] * N + c];
            if (o == 0.0)
              continue;
            chain_t ch;
            ch.row = i;
            ch.dep = dep;
            ch.blk = b;
            ch.coeff = o;
            ch.dep_col = (uint8_t)c;
            ch.dep_eliminated = ((uint8_t)c == c_sel[dep]);
            ch.dep_pos = 0;
            if (!ch.dep_eliminated)
              for (int cc = 0; cc < M; cc++)
                if (cmap[(size_t)dep * M + cc] == (uint8_t)c)
                {
                  ch.dep_pos = (uint8_t)cc;
                  break;
                }
            chains.push_back(ch);
          }
        }
      }

      // chains referencing an eliminated dep unknown must be one-level:
      // the dep row itself must be chain-free
      for (const auto &ch : chains)
      {
        if (!ch.dep_eliminated)
          continue;
        for (const auto &ch2 : chains)
          if (ch2.row == ch.dep)
          {
            printf("linsolv_schur_elim: multi-level elimination chain at block "
                   "rows %d -> %d; not supported\n", (int)ch.row, (int)ch.dep);
            return 1;
          }
      }

      // correction records: every pattern block (k, chained row) needs a
      // correction into block (k, dep) — which must exist in the pattern
      for (size_t ci = 0; ci < chains.size(); ci++)
      {
        const auto &ch = chains[ci];
        for (index_t k = 0; k < n_rows; k++)
        {
          index_t src = (index_t)-1;
          for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
            if (cols_ind[b] == ch.row)
            {
              src = b;
              break;
            }
          if (src == (index_t)-1)
            continue;
          index_t dst = (index_t)-1;
          for (index_t b2 = rows_ptr[k]; b2 < rows_ptr[k + 1]; b2++)
            if (cols_ind[b2] == ch.dep)
            {
              dst = b2;
              break;
            }
          if (dst == (index_t)-1)
          {
            printf("linsolv_schur_elim: chain correction target block (%d,%d) "
                   "is not in the sparsity pattern; not supported\n", (int)k, (int)ch.dep);
            return 1;
          }
          chain_fixes.push_back({src, dst, k, ci});
        }
      }

      if (!detected)
      {
        int n_col1 = 0;
        for (index_t i = 0; i < n_rows; i++)
          if (c_sel[i] == 1)
            n_col1++;
        printf("linsolv_schur_elim<%d>: eliminating row %d per block "
               "(pivot column: %s; %d of %d rows use column 1); "
               "%d fallback rows, %d chain(s)\n",
               (int)N, (int)elim_row, elim_col > 0 ? "fixed" : "auto",
               n_col1, (int)n_rows, n_fallback, (int)chains.size());
      }
      detected = true;
      return 0;
    }

    // Re-read the chain coefficients from the CURRENT values (detection froze
    // the topology, not the magnitudes). O(chains) -- one scalar per chain.
    template <uint8_t N_BLOCK_SIZE>
    void linsolv_schur_elim<N_BLOCK_SIZE>::refresh_chain_coeffs(const mat_float *values_at,
      bool values_on_device)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      for (auto &ch : chains)
      {
        const size_t off = (size_t)ch.blk * N * N + (size_t)r_sel[ch.row] * N + ch.dep_col;
#ifdef WITH_GPU
        if (values_on_device)
        {
          cudaMemcpy(&ch.coeff, values_at + off, sizeof(mat_float), cudaMemcpyDeviceToHost);
          continue;
        }
#endif
        (void)values_on_device;
        ch.coeff = values_at[off];
      }
    }

    // Validate every eliminated pivot against the current values (host path).
    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_schur_elim<N_BLOCK_SIZE>::check_pivots_host()
    {
      for (index_t i = 0; i < n_rows; i++)
        if (!(std::fabs(pivot[i]) > pivot_eps) || !std::isfinite(pivot[i]))
        {
          printf("linsolv_schur_elim: block row %d pivot |%.3e| <= eps on the "
                 "current Jacobian (the eliminated row/column selected at the "
                 "first setup is no longer usable, e.g. a well control switched); "
                 "disable mineral elimination for this run\n", (int)i, (double)pivot[i]);
          return false;
        }
      return true;
    }

    // Apply the O(wells) chain corrections to the reduced values. values_at /
    // red_values_at point at host arrays (host path) or are used as the base
    // for tiny device round-trips (device path).
    template <uint8_t N_BLOCK_SIZE>
    void linsolv_schur_elim<N_BLOCK_SIZE>::apply_chain_fixes(const mat_float *values_at,
      mat_float *red_values_at, bool values_on_device)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;

      for (const auto &fx : chain_fixes)
      {
        const auto &ch = chains[fx.chain_idx];
        const mat_float f = ch.coeff / pivot[ch.row];
        mat_float Ablk[N * N], Sblk[M * M];
        const mat_float *A;
        mat_float *S;
#ifdef WITH_GPU
        if (values_on_device)
        {
          cudaMemcpy(Ablk, values_at + (size_t)fx.src_blk * N * N,
            sizeof(mat_float) * N * N, cudaMemcpyDeviceToHost);
          cudaMemcpy(Sblk, red_values_at + (size_t)fx.dst_blk * M * M,
            sizeof(mat_float) * M * M, cudaMemcpyDeviceToHost);
          A = Ablk;
          S = Sblk;
        }
        else
#endif
        {
          (void)values_on_device;
          A = values_at + (size_t)fx.src_blk * N * N;
          S = red_values_at + (size_t)fx.dst_blk * M * M;
        }

        const uint8_t *rm = &rmap[(size_t)fx.row_k * M];
        if (ch.dep_eliminated)
        {
          // x_m(row) references x_m(dep): S(k,dep)[:,cc] += A[kept, c_sel(row)] * f * g_dep[cc]
          const mat_float *gd = &g[(size_t)ch.dep * M];
          for (int rr = 0; rr < M; rr++)
          {
            const mat_float am = A[rm[rr] * N + c_sel[ch.row]];
            for (int cc = 0; cc < M; cc++)
              S[rr * M + cc] += am * f * gd[cc];
          }
        }
        else
        {
          // x_m(row) references a KEPT dep unknown: S(k,dep)[:,dep_pos] -= A[kept, c_sel(row)] * f
          for (int rr = 0; rr < M; rr++)
            S[rr * M + ch.dep_pos] -= A[rm[rr] * N + c_sel[ch.row]] * f;
        }

#ifdef WITH_GPU
        if (values_on_device)
          cudaMemcpy(red_values_at + (size_t)fx.dst_blk * M * M, Sblk,
            sizeof(mat_float) * M * M, cudaMemcpyHostToDevice);
#endif
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_schur_elim<N_BLOCK_SIZE>::condense_host(const mat_float *values_h)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      const index_t *rows_ptr = A_saved->get_rows_ptr();
      const index_t *cols_ind = A_saved->get_cols_ind();
      const index_t *diag_ind = A_saved->get_diag_ind();
      mat_float *red_values = reduced->values.data();

      // per-row factors
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t i = 0; i < n_rows; i++)
      {
        const mat_float *db = values_h + (size_t)diag_ind[i] * N * N;
        const uint8_t r = r_sel[i];
        const mat_float p = db[r * N + c_sel[i]];
        pivot[i] = p;
        const mat_float pinv = mat_float(1) / p;
        const uint8_t *cm = &cmap[(size_t)i * M];
        for (int cc = 0; cc < M; cc++)
          g[(size_t)i * M + cc] = db[r * N + cm[cc]] * pinv;
      }

      // condensation over rows/blocks
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t k = 0; k < n_rows; k++)
      {
        const uint8_t *rm = &rmap[(size_t)k * M];
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const index_t j = cols_ind[b];
          const mat_float *A = values_h + (size_t)b * N * N;
          mat_float *S = red_values + (size_t)b * M * M;
          const mat_float *gj = &g[(size_t)j * M];
          const uint8_t *cm = &cmap[(size_t)j * M];
          const uint8_t cs = c_sel[j];
          for (int rr = 0; rr < M; rr++)
          {
            const mat_float *Ar = A + rm[rr] * N;
            const mat_float am = Ar[cs];
            for (int cc = 0; cc < M; cc++)
              S[rr * M + cc] = Ar[cm[cc]] - am * gj[cc];
          }
        }
      }

      if (!check_pivots_host())
        return 1;
      refresh_chain_coeffs(values_h, /*values_on_device=*/false);
      apply_chain_fixes(values_h, red_values, /*values_on_device=*/false);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_schur_elim<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      A_saved = A;

#ifdef WITH_GPU
      if (on_device)
      {
        // values live on the device; download for detection only (first setup)
        if (!detected)
        {
          cudaMemcpy(values_h_staging.data(), A->get_values_d(),
            sizeof(mat_float) * (size_t)n_nnz * N * N, cudaMemcpyDeviceToHost);
          if (detect_rows(values_h_staging.data()))
            return 1;
          cudaMemcpy(r_sel_d, r_sel.data(), n_rows, cudaMemcpyHostToDevice);
          cudaMemcpy(c_sel_d, c_sel.data(), n_rows, cudaMemcpyHostToDevice);
          cudaMemcpy(rmap_d, rmap.data(), (size_t)n_rows * M, cudaMemcpyHostToDevice);
          cudaMemcpy(cmap_d, cmap.data(), (size_t)n_rows * M, cudaMemcpyHostToDevice);
          // detection is done; release the full-values host staging buffer
          std::vector<mat_float>().swap(values_h_staging);
        }

        const index_t *rows_ptr_d = A->get_rows_ptr_d();
        const index_t *cols_ind_d = A->get_cols_ind_d();
        const index_t *diag_ind_d = A->get_diag_ind_d();
        const mat_float *values_d = A->get_values_d();

        const int grid_rows = (int)((n_rows + SE_BLOCK - 1) / SE_BLOCK);
        cudaMemset(pivot_bad_d, 0, sizeof(int));
        se_factors_kernel<N><<<grid_rows, SE_BLOCK>>>(n_rows, diag_ind_d, values_d,
          r_sel_d, c_sel_d, cmap_d, pivot_d, g_d, pivot_eps, pivot_bad_d);
        int pivot_bad = 0;
        cudaMemcpy(&pivot_bad, pivot_bad_d, sizeof(int), cudaMemcpyDeviceToHost);
        if (pivot_bad)
        {
          printf("linsolv_schur_elim: an eliminated pivot degenerated on the "
                 "current Jacobian (the row/column selected at first setup is no "
                 "longer usable, e.g. a well control switched); disable mineral "
                 "elimination for this run\n");
          return 1;
        }
        se_condense_kernel<N><<<grid_rows, SE_BLOCK>>>(n_rows, rows_ptr_d, cols_ind_d,
          values_d, rmap_d, cmap_d, c_sel_d, g_d, reduced->values_d);

        if (!chain_fixes.empty())
        {
          // chain corrections need pivot/g on the host (few records)
          cudaMemcpy(pivot.data(), pivot_d, sizeof(mat_float) * n_rows, cudaMemcpyDeviceToHost);
          cudaMemcpy(g.data(), g_d, sizeof(mat_float) * (size_t)n_rows * M, cudaMemcpyDeviceToHost);
          refresh_chain_coeffs(values_d, /*values_on_device=*/true);
          apply_chain_fixes(values_d, reduced->values_d, /*values_on_device=*/true);
        }
        return inner->setup(reduced);
      }
#endif

      const mat_float *values_h = A->get_values();
      if (!detected && detect_rows(values_h))
        return 1;
      if (condense_host(values_h))
        return 1;
      return inner->setup(reduced);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_schur_elim<N_BLOCK_SIZE>::reduce_rhs_host(const mat_float *B)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      const index_t *rows_ptr = A_saved->get_rows_ptr();
      const index_t *cols_ind = A_saved->get_cols_ind();
      const mat_float *values_h = A_saved->get_values();

      // affine constants: locals first, then the one-level chains (only
      // eliminated-dep chains carry an h correction)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t i = 0; i < n_rows; i++)
        h_eff[i] = B[(size_t)i * N + r_sel[i]] / pivot[i];
      for (const auto &ch : chains)
        if (ch.dep_eliminated)
          h_eff[ch.row] = (B[(size_t)ch.row * N + r_sel[ch.row]] - ch.coeff * h_eff[ch.dep]) / pivot[ch.row];

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t k = 0; k < n_rows; k++)
      {
        const uint8_t *rm = &rmap[(size_t)k * M];
        mat_float acc[M];
        for (int rr = 0; rr < M; rr++)
          acc[rr] = B[(size_t)k * N + rm[rr]];
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const mat_float he = h_eff[cols_ind[b]];
          const mat_float *A = values_h + (size_t)b * N * N;
          const uint8_t cs = c_sel[cols_ind[b]];
          for (int rr = 0; rr < M; rr++)
            acc[rr] -= A[rm[rr] * N + cs] * he;
        }
        for (int rr = 0; rr < M; rr++)
          b_red[(size_t)k * M + rr] = acc[rr];
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_schur_elim<N_BLOCK_SIZE>::backsub_host(const mat_float *B, mat_float *X)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t i = 0; i < n_rows; i++)
      {
        const mat_float *xr = &x_red[(size_t)i * M];
        const mat_float *gi = &g[(size_t)i * M];
        const uint8_t *cm = &cmap[(size_t)i * M];
        mat_float xm = h_eff[i];
        for (int cc = 0; cc < M; cc++)
        {
          X[(size_t)i * N + cm[cc]] = xr[cc];
          xm -= gi[cc] * xr[cc];
        }
        X[(size_t)i * N + c_sel[i]] = xm;
      }
      // chained rows: recompute the eliminated unknown from the exact
      // recurrence (dependency values are final in X / x_red)
      for (const auto &ch : chains)
      {
        const index_t i = ch.row;
        const mat_float *xr = &x_red[(size_t)i * M];
        const mat_float *gi = &g[(size_t)i * M];
        const mat_float dep_val = ch.dep_eliminated
            ? X[(size_t)ch.dep * N + c_sel[ch.dep]]
            : x_red[(size_t)ch.dep * M + ch.dep_pos];
        mat_float xm = (B[(size_t)i * N + r_sel[i]] - ch.coeff * dep_val) / pivot[i];
        for (int cc = 0; cc < M; cc++)
          xm -= gi[cc] * xr[cc];
        X[(size_t)i * N + c_sel[i]] = xm;
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_schur_elim<N_BLOCK_SIZE>::solve(mat_float *B, mat_float *X)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;

#ifdef WITH_GPU
      if (on_device)
      {
        const index_t *rows_ptr_d = A_saved->get_rows_ptr_d();
        const index_t *cols_ind_d = A_saved->get_cols_ind_d();
        const mat_float *values_d = A_saved->get_values_d();
        const int grid_rows = (int)((n_rows + SE_BLOCK - 1) / SE_BLOCK);

        se_h_kernel<N><<<grid_rows, SE_BLOCK>>>(n_rows, B, r_sel_d, pivot_d, h_eff_d);
        // chain fixups on h_eff (O(wells)): tiny host round trips
        for (const auto &ch : chains)
        {
          if (!ch.dep_eliminated)
            continue;
          mat_float bm, hdep;
          cudaMemcpy(&bm, B + (size_t)ch.row * N + r_sel[ch.row], sizeof(mat_float), cudaMemcpyDeviceToHost);
          cudaMemcpy(&hdep, h_eff_d + ch.dep, sizeof(mat_float), cudaMemcpyDeviceToHost);
          const mat_float he = (bm - ch.coeff * hdep) / pivot[ch.row];
          cudaMemcpy(h_eff_d + ch.row, &he, sizeof(mat_float), cudaMemcpyHostToDevice);
        }
        se_reduce_rhs_kernel<N><<<grid_rows, SE_BLOCK>>>(n_rows, rows_ptr_d, cols_ind_d,
          values_d, rmap_d, c_sel_d, h_eff_d, B, b_red_d);

        const int rc = inner->solve(b_red_d, x_red_d);
        if (rc)
          return rc;

        se_backsub_kernel<N><<<grid_rows, SE_BLOCK>>>(n_rows, x_red_d, c_sel_d, cmap_d,
          g_d, h_eff_d, X);
        // chained rows: recompute the eliminated unknown with final dep values
        for (const auto &ch : chains)
        {
          mat_float bm, dep_val, xr[M], gi[M];
          cudaMemcpy(&bm, B + (size_t)ch.row * N + r_sel[ch.row], sizeof(mat_float), cudaMemcpyDeviceToHost);
          if (ch.dep_eliminated)
            cudaMemcpy(&dep_val, X + (size_t)ch.dep * N + c_sel[ch.dep], sizeof(mat_float), cudaMemcpyDeviceToHost);
          else
            cudaMemcpy(&dep_val, x_red_d + (size_t)ch.dep * M + ch.dep_pos, sizeof(mat_float), cudaMemcpyDeviceToHost);
          cudaMemcpy(xr, x_red_d + (size_t)ch.row * M, sizeof(mat_float) * M, cudaMemcpyDeviceToHost);
          cudaMemcpy(gi, g_d + (size_t)ch.row * M, sizeof(mat_float) * M, cudaMemcpyDeviceToHost);
          mat_float xm = (bm - ch.coeff * dep_val) / pivot[ch.row];
          for (int cc = 0; cc < M; cc++)
            xm -= gi[cc] * xr[cc];
          cudaMemcpy(X + (size_t)ch.row * N + c_sel[ch.row], &xm, sizeof(mat_float), cudaMemcpyHostToDevice);
        }
        return 0;
      }
#endif

      if (reduce_rhs_host(B))
        return 1;
      const int rc = inner->solve(b_red.data(), x_red.data());
      if (rc)
        return rc;
      return backsub_host(B, X);
    }

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see linear_solvers/src/CMakeLists.txt.
#include "linsolv_schur_elim_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts
