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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
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

#if defined(__CUDACC__)
#define OD_SE_HD __host__ __device__
#else
#define OD_SE_HD
#endif

      // Dense K x K inverse with partial pivoting (host + device, K small).
      // Returns false if a pivot falls at/below `eps` (singular). Uses
      // unqualified fabs/isfinite so it resolves to device intrinsics under
      // nvcc and to the C library on the host.
      template <uint8_t K>
      OD_SE_HD inline bool invert_kxk(const mat_float *a_in, mat_float *inv, double eps)
      {
        mat_float a[K * K], I[K * K];
        for (int r = 0; r < K; r++)
          for (int c = 0; c < K; c++)
          {
            a[r * K + c] = a_in[r * K + c];
            I[r * K + c] = (r == c) ? mat_float(1) : mat_float(0);
          }
        for (int col = 0; col < K; col++)
        {
          int piv = col;
          mat_float pv = fabs(a[col * K + col]);
          for (int r = col + 1; r < K; r++)
          {
            mat_float v = fabs(a[r * K + col]);
            if (v > pv) { pv = v; piv = r; }
          }
          if (!(pv > eps) || !isfinite(pv))
            return false;
          if (piv != col)
            for (int c = 0; c < K; c++)
            {
              mat_float t = a[col * K + c]; a[col * K + c] = a[piv * K + c]; a[piv * K + c] = t;
              t = I[col * K + c]; I[col * K + c] = I[piv * K + c]; I[piv * K + c] = t;
            }
          mat_float d = mat_float(1) / a[col * K + col];
          for (int c = 0; c < K; c++) { a[col * K + c] *= d; I[col * K + c] *= d; }
          for (int r = 0; r < K; r++)
          {
            if (r == col) continue;
            mat_float f = a[r * K + col];
            if (f == mat_float(0)) continue;
            for (int c = 0; c < K; c++) { a[r * K + c] -= f * a[col * K + c]; I[r * K + c] -= f * I[col * K + c]; }
          }
        }
        for (int r = 0; r < K * K; r++) inv[r] = I[r];
        return true;
      }

#ifdef WITH_GPU
      constexpr int SE_BLOCK = 256;

      // Pinv (K x K) and Gm = Pinv . D[e_rows, keep_cols] (K x M), per cell.
      template <uint8_t N, uint8_t K, uint8_t M>
      __global__ void se_factors_kernel(index_t n_rows, const index_t *diag_ind, const mat_float *values,
        const uint8_t *e_rows, const uint8_t *elim_cols, const uint8_t *keep_cols,
        mat_float *Pinv, mat_float *Gm, double eps, int *pivot_bad)
      {
        const index_t i = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (i >= n_rows) return;
        const mat_float *db = values + (size_t)diag_ind[i] * N * N;
        const uint8_t *er = e_rows + (size_t)i * K;
        mat_float P[K * K], Pi[K * K];
        for (int a = 0; a < K; a++)
          for (int b = 0; b < K; b++)
            P[a * K + b] = db[er[a] * N + elim_cols[b]];
        if (!invert_kxk<K>(P, Pi, eps))
        {
          atomicExch(pivot_bad, 1);
          for (int r = 0; r < K * K; r++) Pi[r] = 0;
        }
        for (int r = 0; r < K * K; r++) Pinv[(size_t)i * K * K + r] = Pi[r];
        // Gm[a, c] = sum_b Pi[a,b] * D[er[b], keep_cols[c]]
        for (int a = 0; a < K; a++)
          for (int c = 0; c < M; c++)
          {
            mat_float s = 0;
            for (int b = 0; b < K; b++) s += Pi[a * K + b] * db[er[b] * N + keep_cols[c]];
            Gm[((size_t)i * K + a) * M + c] = s;
          }
      }

      // S_kj[rr, cc] = A_kj[k_rows(k)[rr], keep_cols[cc]] - sum_e A_kj[k_rows(k)[rr], elim_cols[e]] * Gm_j[e, cc]
      // Also re-validates chain topology: any nonzero in an ELIMINATED row of an
      // off-diagonal block that is not among the recorded chain entries
      // (sorted encoded offsets) sets flags[1] -- the condensation would silently
      // drop that coupling otherwise.
      template <uint8_t N, uint8_t K, uint8_t M>
      __global__ void se_condense_kernel(index_t n_rows, const index_t *rows_ptr, const index_t *cols_ind,
        const mat_float *values, const uint8_t *k_rows, const uint8_t *e_rows,
        const uint8_t *elim_cols, const uint8_t *keep_cols,
        const mat_float *Gm, mat_float *red_values,
        const unsigned long long *chain_offs, int n_chain_offs, int *flags)
      {
        const index_t k = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (k >= n_rows) return;
        const uint8_t *kr = k_rows + (size_t)k * M;
        const uint8_t *er = e_rows + (size_t)k * K;
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const index_t j = cols_ind[b];
          const mat_float *A = values + (size_t)b * N * N;
          mat_float *S = red_values + (size_t)b * M * M;
          const mat_float *Gj = Gm + (size_t)j * K * M;
          for (int rr = 0; rr < M; rr++)
          {
            const mat_float *Ar = A + kr[rr] * N;
            for (int cc = 0; cc < M; cc++)
            {
              mat_float s = Ar[keep_cols[cc]];
              for (int e = 0; e < K; e++) s -= Ar[elim_cols[e]] * Gj[e * M + cc];
              S[rr * M + cc] = s;
            }
          }
          if (j != k)
          {
            // topology check on the eliminated rows of this off-diagonal block
            for (int e = 0; e < K; e++)
            {
              const mat_float *Ae = A + er[e] * N;
              for (int c = 0; c < N; c++)
              {
                if (Ae[c] == mat_float(0)) continue;
                const unsigned long long off =
                    ((unsigned long long)b * N + er[e]) * N + (unsigned long long)c;
                // binary search in the (small) sorted recorded set
                int lo = 0, hi = n_chain_offs - 1;
                bool found = false;
                while (lo <= hi)
                {
                  const int mid = (lo + hi) >> 1;
                  if (chain_offs[mid] == off) { found = true; break; }
                  if (chain_offs[mid] < off) lo = mid + 1; else hi = mid - 1;
                }
                if (!found)
                  atomicExch(flags + 1, 1);
              }
            }
          }
        }
      }

      // h_i = Pinv_i . b_i[e_rows]   (local part; chain recurrence fixed on host)
      template <uint8_t N, uint8_t K>
      __global__ void se_h_kernel(index_t n_rows, const mat_float *B, const uint8_t *e_rows,
        const mat_float *Pinv, mat_float *h_eff)
      {
        const index_t i = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (i >= n_rows) return;
        const uint8_t *er = e_rows + (size_t)i * K;
        const mat_float *Pi = Pinv + (size_t)i * K * K;
        for (int a = 0; a < K; a++)
        {
          mat_float s = 0;
          for (int b = 0; b < K; b++) s += Pi[a * K + b] * B[(size_t)i * N + er[b]];
          h_eff[(size_t)i * K + a] = s;
        }
      }

      // b_red[k][cc] = B[k][k_rows[cc]] - sum_j sum_e A_kj[k_rows[cc], elim_cols[e]] * h_j[e]
      template <uint8_t N, uint8_t K, uint8_t M>
      __global__ void se_reduce_rhs_kernel(index_t n_rows, const index_t *rows_ptr, const index_t *cols_ind,
        const mat_float *values, const uint8_t *k_rows, const uint8_t *elim_cols,
        const mat_float *h_eff, const mat_float *B, mat_float *b_red)
      {
        const index_t k = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (k >= n_rows) return;
        const uint8_t *kr = k_rows + (size_t)k * M;
        mat_float acc[M];
        for (int rr = 0; rr < M; rr++) acc[rr] = B[(size_t)k * N + kr[rr]];
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const mat_float *A = values + (size_t)b * N * N;
          const mat_float *hj = h_eff + (size_t)cols_ind[b] * K;
          for (int rr = 0; rr < M; rr++)
          {
            const mat_float *Ar = A + kr[rr] * N;
            mat_float s = 0;
            for (int e = 0; e < K; e++) s += Ar[elim_cols[e]] * hj[e];
            acc[rr] -= s;
          }
        }
        for (int rr = 0; rr < M; rr++) b_red[(size_t)k * M + rr] = acc[rr];
      }

      // x[keep_cols] = x_red; x[elim_cols] = h_i - Gm_i . x_red  (local part)
      template <uint8_t N, uint8_t K, uint8_t M>
      __global__ void se_backsub_kernel(index_t n_rows, const mat_float *x_red, const uint8_t *elim_cols,
        const uint8_t *keep_cols, const mat_float *Gm, const mat_float *h_eff, mat_float *X)
      {
        const index_t i = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (i >= n_rows) return;
        const mat_float *xr = x_red + (size_t)i * M;
        const mat_float *Gi = Gm + (size_t)i * K * M;
        const mat_float *hi = h_eff + (size_t)i * K;
        for (int cc = 0; cc < M; cc++) X[(size_t)i * N + keep_cols[cc]] = xr[cc];
        for (int a = 0; a < K; a++)
        {
          mat_float xm = hi[a];
          for (int cc = 0; cc < M; cc++) xm -= Gi[a * M + cc] * xr[cc];
          X[(size_t)i * N + elim_cols[a]] = xm;
        }
      }
#endif // WITH_GPU
#undef OD_SE_HD
    } // namespace

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::linsolv_schur_elim(bool on_device_,
      const std::vector<int> &elim_rows_in, const std::vector<int> &elim_cols_in, double pivot_eps_)
      : on_device(on_device_), pivot_eps(pivot_eps_)
    {
      // Validate the caller-supplied selection hard (memory safety + a clear
      // contract): exactly K entries each, all indices in range, distinct rows,
      // distinct columns, and pressure (column 0) kept so a CPR-type inner
      // preconditioner retains its pressure subsystem. Throwing here fails the
      // construction cleanly through the factory / engine / pybind layers.
      auto fail = [&](const std::string &msg) {
        throw std::runtime_error("linsolv_schur_elim<" + std::to_string((int)N_BLOCK_SIZE) + "," +
            std::to_string((int)N_ELIM) + ">: " + msg);
      };
      if (elim_rows_in.size() != N_ELIM || elim_cols_in.size() != N_ELIM)
        fail("elim_rows/elim_cols must each have exactly " + std::to_string((int)N_ELIM) +
             " entries (got " + std::to_string(elim_rows_in.size()) + "/" +
             std::to_string(elim_cols_in.size()) + ")");
      bool row_seen[N_BLOCK_SIZE] = {false}, col_seen[N_BLOCK_SIZE] = {false};
      for (int k = 0; k < N_ELIM; k++)
      {
        const int r = elim_rows_in[k], c = elim_cols_in[k];
        if (r < 0 || r >= N_BLOCK_SIZE)
          fail("elim_rows[" + std::to_string(k) + "]=" + std::to_string(r) + " out of range [0," +
               std::to_string((int)N_BLOCK_SIZE) + ")");
        if (c < 0 || c >= N_BLOCK_SIZE)
          fail("elim_cols[" + std::to_string(k) + "]=" + std::to_string(c) + " out of range [0," +
               std::to_string((int)N_BLOCK_SIZE) + ")");
        if (c == 0)
          fail("column 0 (pressure) cannot be eliminated -- a CPR-type inner needs the "
               "pressure subsystem preserved");
        if (row_seen[r]) fail("duplicate eliminated row " + std::to_string(r));
        if (col_seen[c]) fail("duplicate eliminated column " + std::to_string(c));
        row_seen[r] = col_seen[c] = true;
      }
      elim_rows_pref.resize(N_ELIM);
      elim_cols.resize(N_ELIM);
      for (int k = 0; k < N_ELIM; k++)
      {
        elim_rows_pref[k] = (uint8_t)elim_rows_in[k];
        elim_cols[k] = (uint8_t)elim_cols_in[k];
      }
      // global kept-column map = complement of elim_cols
      keep_cols.clear();
      for (int c = 0; c < N_BLOCK_SIZE; c++)
        if (!col_seen[c])
          keep_cols.push_back((uint8_t)c);
#ifndef WITH_GPU
      if (on_device)
      {
        printf("linsolv_schur_elim: device mode requested in a CPU-only build; using host path\n");
        on_device = false;
      }
#endif
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::~linsolv_schur_elim()
    {
      free_device();
      delete reduced;
      // Explicit ownership: the engine-built GPU chain marks the inner solver as
      // owned (set_inner_owned) so the whole raw-pointer chain is released here;
      // registry-built inners stay shared_ptr-owned on the Python side.
      if (inner_owned)
        delete inner;
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    void linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::free_device()
    {
#ifdef WITH_GPU
      for (auto **p : {&e_rows_d, &k_rows_d, &elim_cols_d, &keep_cols_d})
        if (*p) { cudaFree(*p); *p = nullptr; }
      for (auto **p : {&Pinv_d, &Gm_d, &h_eff_d, &b_red_d, &x_red_d})
        if (*p) { cudaFree(*p); *p = nullptr; }
      if (flags_d) { cudaFree(flags_d); flags_d = nullptr; }
      if (chain_offsets_d) { cudaFree(chain_offsets_d); chain_offsets_d = nullptr; }
#endif
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    int linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::init(opendarts::linear_solvers::csr_matrix_base *A,
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

      reduced = new opendarts::linear_solvers::csr_matrix<M>;
      reduced->init(n_rows, n_rows, n_nnz);
      std::memcpy(reduced->rows_ptr.data(), rows_ptr, sizeof(index_t) * (n_rows + 1));
      std::memcpy(reduced->cols_ind.data(), A->get_cols_ind(), sizeof(index_t) * n_nnz);
      reduced->diag_ind.resize(n_rows);
      std::memcpy(reduced->diag_ind.data(), A->get_diag_ind(), sizeof(index_t) * n_rows);

      e_rows.assign((size_t)n_rows * K, 0);
      k_rows.assign((size_t)n_rows * M, 0);
      Pinv.assign((size_t)n_rows * K * K, 0);
      Gm.assign((size_t)n_rows * K * M, 0);
      h_eff.assign((size_t)n_rows * K, 0);
      b_red.assign((size_t)n_rows * M, 0);
      x_red.assign((size_t)n_rows * M, 0);

#ifdef WITH_GPU
      if (on_device)
      {
        reduced->init_device(n_rows, n_nnz);
        reduced->copy_struct_to_device();
        cudaMalloc((void **)&e_rows_d, (size_t)n_rows * K);
        cudaMalloc((void **)&k_rows_d, (size_t)n_rows * M);
        cudaMalloc((void **)&elim_cols_d, K);
        cudaMalloc((void **)&keep_cols_d, M);
        cudaMalloc((void **)&Pinv_d, sizeof(mat_float) * n_rows * K * K);
        cudaMalloc((void **)&Gm_d, sizeof(mat_float) * n_rows * K * M);
        cudaMalloc((void **)&h_eff_d, sizeof(mat_float) * n_rows * K);
        cudaMalloc((void **)&b_red_d, sizeof(mat_float) * n_rows * M);
        cudaMalloc((void **)&x_red_d, sizeof(mat_float) * n_rows * M);
        cudaMalloc((void **)&flags_d, 2 * sizeof(int));
        cudaMemcpy(elim_cols_d, elim_cols.data(), K, cudaMemcpyHostToDevice);
        cudaMemcpy(keep_cols_d, keep_cols.data(), M, cudaMemcpyHostToDevice);
        values_h_staging.resize((size_t)n_nnz * N_BLOCK_SIZE * N_BLOCK_SIZE);
      }
#endif
      inner->init_timer_nodes(this->timer_setup, this->timer_solve);
      return inner->init(reduced, max_iters, tolerance);
    }

    // Structural detection (once): per cell pick K rows to pair with the K fixed
    // eliminated columns via greedy pivoting (prefer elim_rows_pref), and record
    // the one-level chains from selected rows' off-diagonal entries.
    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    int linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::detect_rows(const mat_float *values_h)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      const index_t *rows_ptr = A_saved->get_rows_ptr();
      const index_t *cols_ind = A_saved->get_cols_ind();
      const index_t *diag_ind = A_saved->get_diag_ind();
      chains.clear();
      chain_fixes.clear();
      int n_fallback = 0;

      for (index_t i = 0; i < n_rows; i++)
      {
        const mat_float *db = values_h + (size_t)diag_ind[i] * N * N;
        // Row selection: first try the preferred rows; then a greedy per-column
        // assignment; if the assembled K x K block is still singular, fall back
        // to an EXHAUSTIVE search over all K-subsets of the N rows (N <= 16,
        // K <= 4 => at most a few hundred candidates, and only at the rare
        // cells -- well heads -- where greedy fails). Invertibility of the
        // pivot block depends only on the row SET, so subsets suffice.
        uint8_t chosen[N_ELIM];
        auto try_rows = [&](const uint8_t *rows) {
          mat_float P[N_ELIM * N_ELIM], Pi[N_ELIM * N_ELIM];
          for (int a = 0; a < N_ELIM; a++)
            for (int b = 0; b < N_ELIM; b++)
              P[a * N_ELIM + b] = db[rows[a] * N + elim_cols[b]];
          return invert_kxk<N_ELIM>(P, Pi, pivot_eps);
        };
        // 1) preferred rows verbatim
        bool ok = try_rows(elim_rows_pref.data());
        if (ok)
        {
          for (int k = 0; k < N_ELIM; k++)
            chosen[k] = elim_rows_pref[k];
        }
        else
        {
          // 2) greedy per-column assignment (prefer the preferred row per slot)
          bool used[N] = {false};
          bool greedy_ok = true;
          for (int k = 0; k < N_ELIM; k++)
          {
            int r = -1;
            const uint8_t pref = elim_rows_pref[k];
            if (!used[pref] && std::fabs(db[pref * N + elim_cols[k]]) > pivot_eps)
              r = pref;
            else
            {
              mat_float best = (mat_float)pivot_eps;
              for (int cand = 0; cand < N; cand++)
                if (!used[cand] && std::fabs(db[cand * N + elim_cols[k]]) > best)
                {
                  best = std::fabs(db[cand * N + elim_cols[k]]);
                  r = cand;
                }
            }
            if (r < 0) { greedy_ok = false; break; }
            chosen[k] = (uint8_t)r;
            used[r] = true;
          }
          ok = greedy_ok && try_rows(chosen);
          if (!ok)
          {
            // 3) exhaustive K-subset search (lexicographic; first invertible wins)
            uint8_t idx[N_ELIM];
            for (int k = 0; k < N_ELIM; k++) idx[k] = (uint8_t)k;
            while (true)
            {
              if (try_rows(idx))
              {
                for (int k = 0; k < N_ELIM; k++) chosen[k] = idx[k];
                ok = true;
                break;
              }
              // next combination
              int pos = N_ELIM - 1;
              while (pos >= 0 && idx[pos] == (uint8_t)(N - N_ELIM + pos)) pos--;
              if (pos < 0) break;
              idx[pos]++;
              for (int k = pos + 1; k < N_ELIM; k++) idx[k] = (uint8_t)(idx[k - 1] + 1);
            }
          }
          if (ok) n_fallback++;
        }
        if (!ok)
        {
          printf("linsolv_schur_elim<%d,%d>: block row %d has no invertible K x K "
                 "pivot for the requested columns (exhaustive row search); disable "
                 "local (Schur) elimination for this model\n", (int)N, (int)N_ELIM, (int)i);
          return 1;
        }
        for (int k = 0; k < N_ELIM; k++)
          e_rows[(size_t)i * K + k] = chosen[k];
        int cc = 0;
        for (int r = 0; r < N; r++)
        {
          bool is_e = false;
          for (int k = 0; k < N_ELIM; k++) if (chosen[k] == r) { is_e = true; break; }
          if (!is_e) k_rows[(size_t)i * M + cc++] = (uint8_t)r;
        }
      }

      // chains: off-diagonal entries in the selected eliminated rows
      auto col_class = [&](index_t cell, int c, bool &elim, uint8_t &epos, uint8_t &kpos) {
        elim = false;
        for (int k = 0; k < N_ELIM; k++) if (elim_cols[k] == c) { elim = true; epos = (uint8_t)k; }
        if (!elim)
          for (int k = 0; k < M; k++) if (keep_cols[k] == c) { kpos = (uint8_t)k; }
        (void)cell;
      };
      for (index_t i = 0; i < n_rows; i++)
      {
        for (index_t b = rows_ptr[i]; b < rows_ptr[i + 1]; b++)
        {
          const index_t dep = cols_ind[b];
          if (dep == i) continue;
          const mat_float *Ab = values_h + (size_t)b * N * N;
          for (int e = 0; e < N_ELIM; e++)
          {
            const uint8_t r = e_rows[(size_t)i * K + e];
            for (int c = 0; c < N; c++)
            {
              const mat_float o = Ab[r * N + c];
              if (o == 0.0) continue;
              chain_t ch;
              ch.row = i; ch.dep = dep; ch.blk = b; ch.coeff = o;
              ch.eidx = (uint8_t)e; ch.dep_col = (uint8_t)c;
              ch.dep_eliminated = false; ch.dep_epos = 0; ch.dep_kpos = 0;
              col_class(dep, c, ch.dep_eliminated, ch.dep_epos, ch.dep_kpos);
              chains.push_back(ch);
            }
          }
        }
      }
      // one-level: an eliminated-dep chain's dependency row must be chain-free
      for (const auto &ch : chains)
      {
        if (!ch.dep_eliminated) continue;
        for (const auto &ch2 : chains)
          if (ch2.row == ch.dep)
          {
            printf("linsolv_schur_elim: multi-level elimination chain at block rows "
                   "%d -> %d; not supported\n", (int)ch.row, (int)ch.dep);
            return 1;
          }
      }
      // reduced-matrix correction records: for every pattern block (k, chained row)
      // there is a correction into block (k, dep) — which must exist in the pattern.
      for (size_t ci = 0; ci < chains.size(); ci++)
      {
        const auto &ch = chains[ci];
        for (index_t k = 0; k < n_rows; k++)
        {
          index_t src = (index_t)-1;
          for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
            if (cols_ind[b] == ch.row) { src = b; break; }
          if (src == (index_t)-1) continue;
          index_t dst = (index_t)-1;
          for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
            if (cols_ind[b] == ch.dep) { dst = b; break; }
          if (dst == (index_t)-1)
          {
            printf("linsolv_schur_elim: chain correction target block (%d,%d) is not "
                   "in the sparsity pattern; not supported\n", (int)k, (int)ch.dep);
            return 1;
          }
          chain_fixes.push_back({src, dst, k, ci});
        }
      }

      // encoded offsets of every recorded chain entry, sorted, for the
      // per-setup topology re-validation (host scan / device binary search)
      chain_offsets.clear();
      chain_offsets.reserve(chains.size());
      for (const auto &ch : chains)
        chain_offsets.push_back(((unsigned long long)ch.blk * N +
            e_rows[(size_t)ch.row * K + ch.eidx]) * N + (unsigned long long)ch.dep_col);
      std::sort(chain_offsets.begin(), chain_offsets.end());
      chain_touched.clear();
      for (const auto &ch : chains)
      {
        chain_touched.push_back(ch.row);
        chain_touched.push_back(ch.dep);
      }
      for (const auto &fx : chain_fixes)
        chain_touched.push_back(fx.row_k);
      std::sort(chain_touched.begin(), chain_touched.end());
      chain_touched.erase(std::unique(chain_touched.begin(), chain_touched.end()),
                          chain_touched.end());

      if (!detected)
        printf("linsolv_schur_elim<%d,%d>: eliminating %d pair(s) per block "
               "(rows[0]=%d..., cols[0]=%d...); %d fallback row-slots, %d chain(s)\n",
               (int)N, (int)N_ELIM, (int)N_ELIM, (int)elim_rows_pref[0], (int)elim_cols[0],
               n_fallback, (int)chains.size());
      detected = true;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    void linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::refresh_chain_coeffs(const mat_float *values_at,
      bool values_on_device)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      for (auto &ch : chains)
      {
        const size_t off = (size_t)ch.blk * N * N + (size_t)e_rows[(size_t)ch.row * K + ch.eidx] * N + ch.dep_col;
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

    // Diagnostic for a degenerate pivot: identify the first few offending cells,
    // their pivot blocks, and whether the block is non-finite (a NaN/Inf Jacobian,
    // e.g. from a zero-viscosity kr/mu evaluation upstream) or has an exactly
    // singular pivot pairing (state-dependent zero derivative).
    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    void linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::report_degenerate_cells(const mat_float *values_h)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      const index_t *diag_ind = A_saved->get_diag_ind();
      int reported = 0, n_bad = 0, n_nonfinite = 0;
      for (index_t i = 0; i < n_rows && reported < 3; i++)
      {
        const mat_float *db = values_h + (size_t)diag_ind[i] * N * N;
        const uint8_t *er = &e_rows[(size_t)i * K];
        mat_float P[N_ELIM * N_ELIM], Pi[N_ELIM * N_ELIM];
        bool finite = true;
        for (int a = 0; a < N_ELIM; a++)
          for (int b = 0; b < N_ELIM; b++)
          {
            P[a * N_ELIM + b] = db[er[a] * N + elim_cols[b]];
            if (!std::isfinite(P[a * N_ELIM + b]))
              finite = false;
          }
        if (invert_kxk<N_ELIM>(P, Pi, pivot_eps))
          continue;
        n_bad++;
        if (!finite)
          n_nonfinite++;
        if (reported < 3)
        {
          printf("linsolv_schur_elim: degenerate pivot at block row %d (rows:", (int)i);
          for (int a = 0; a < N_ELIM; a++) printf(" %d", (int)er[a]);
          printf(", cols:");
          for (int a = 0; a < N_ELIM; a++) printf(" %d", (int)elim_cols[a]);
          printf("): P =");
          for (int a = 0; a < N_ELIM * N_ELIM; a++) printf(" %.3e", (double)P[a]);
          printf("%s\n", finite ? " (exactly singular)" : " (NON-FINITE: NaN/Inf Jacobian)");
          reported++;
        }
      }
      printf("linsolv_schur_elim: %d degenerate cell(s) (%d non-finite) on the current "
             "Jacobian -- %s\n", n_bad, n_nonfinite,
             n_nonfinite
               ? "the Jacobian itself is corrupted (NaN/Inf from upstream property "
                 "evaluation); elimination is not the root cause"
               : "re-detection could not find a usable row pairing; disable local "
                 "(Schur) elimination for this model/regime");
    }

    // Host condensation: Pinv, Gm, reduced blocks, chain corrections; validates pivots.
    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    int linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::condense_host(const mat_float *values_h)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      const index_t *rows_ptr = A_saved->get_rows_ptr();
      const index_t *cols_ind = A_saved->get_cols_ind();
      const index_t *diag_ind = A_saved->get_diag_ind();
      mat_float *red_values = reduced->values.data();
      int bad = 0;

      // per-cell factors
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(| : bad)
#endif
      for (index_t i = 0; i < n_rows; i++)
      {
        const mat_float *db = values_h + (size_t)diag_ind[i] * N * N;
        const uint8_t *er = &e_rows[(size_t)i * K];
        mat_float P[K * K];
        for (int a = 0; a < K; a++)
          for (int b = 0; b < K; b++)
            P[a * K + b] = db[er[a] * N + elim_cols[b]];
        mat_float *Pi = &Pinv[(size_t)i * K * K];
        if (!invert_kxk<K>(P, Pi, pivot_eps))
        {
          bad |= 1;
          for (int r = 0; r < K * K; r++) Pi[r] = 0;
        }
        mat_float *Gi = &Gm[(size_t)i * K * M];
        for (int a = 0; a < K; a++)
          for (int c = 0; c < M; c++)
          {
            mat_float s = 0;
            for (int b = 0; b < K; b++) s += Pi[a * K + b] * db[er[b] * N + keep_cols[c]];
            Gi[a * M + c] = s;
          }
      }
      if (bad)
      {
        report_degenerate_cells(values_h);
        return 1;
      }

      // condensation + chain-topology re-validation (an eliminated-row entry
      // that was 0.0 at detection and is nonzero now would be silently dropped)
      int topo_bad = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(| : topo_bad)
#endif
      for (index_t k = 0; k < n_rows; k++)
      {
        const uint8_t *kr = &k_rows[(size_t)k * M];
        const uint8_t *er = &e_rows[(size_t)k * K];
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const index_t j = cols_ind[b];
          const mat_float *A = values_h + (size_t)b * N * N;
          mat_float *S = red_values + (size_t)b * M * M;
          const mat_float *Gj = &Gm[(size_t)j * K * M];
          for (int rr = 0; rr < M; rr++)
          {
            const mat_float *Ar = A + kr[rr] * N;
            for (int cc = 0; cc < M; cc++)
            {
              mat_float s = Ar[keep_cols[cc]];
              for (int e = 0; e < K; e++) s -= Ar[elim_cols[e]] * Gj[e * M + cc];
              S[rr * M + cc] = s;
            }
          }
          if (j != k)
            for (int e = 0; e < K; e++)
            {
              const mat_float *Ae = A + er[e] * N;
              for (int c = 0; c < N; c++)
              {
                if (Ae[c] == 0.0) continue;
                const unsigned long long off =
                    ((unsigned long long)b * N + er[e]) * N + (unsigned long long)c;
                if (!std::binary_search(chain_offsets.begin(), chain_offsets.end(), off))
                  topo_bad |= 1;
              }
            }
        }
      }
      if (topo_bad)
      {
        printf("linsolv_schur_elim: an eliminated-row off-diagonal coupling appeared "
               "that was zero at detection (well control change?); the recorded chain "
               "topology is stale -- disable local (Schur) elimination or re-init the solver\n");
        return 1;
      }

      // chain corrections (O(wells))
      refresh_chain_coeffs(values_h, /*values_on_device=*/false);
      for (const auto &fx : chain_fixes)
      {
        const auto &ch = chains[fx.chain_idx];
        const mat_float *A = values_h + (size_t)fx.src_blk * N * N;
        mat_float *S = red_values + (size_t)fx.dst_blk * M * M;
        const uint8_t *kr = &k_rows[(size_t)fx.row_k * M];
        const mat_float *Pi = &Pinv[(size_t)ch.row * K * K];
        // w[rr] = sum_e A_ki[k_rows[rr], elim_cols[e]] * Pinv_i[e, eidx]
        for (int rr = 0; rr < M; rr++)
        {
          const mat_float *Ar = A + kr[rr] * N;
          mat_float w = 0;
          for (int e = 0; e < K; e++) w += Ar[elim_cols[e]] * Pi[e * K + ch.eidx];
          const mat_float wc = w * ch.coeff;
          if (ch.dep_eliminated)
          {
            const mat_float *Gd = &Gm[(size_t)ch.dep * K * M];
            for (int cc = 0; cc < M; cc++) S[rr * M + cc] += wc * Gd[ch.dep_epos * M + cc];
          }
          else
            S[rr * M + ch.dep_kpos] -= wc;
        }
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    int linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::setup(opendarts::linear_solvers::csr_matrix_base *A)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      A_saved = A;

#ifdef WITH_GPU
      if (on_device)
      {
        if (!detected)
        {
          cudaMemcpy(values_h_staging.data(), A->get_values_d(),
            sizeof(mat_float) * (size_t)n_nnz * N * N, cudaMemcpyDeviceToHost);
          if (detect_rows(values_h_staging.data())) return 1;
          cudaMemcpy(e_rows_d, e_rows.data(), (size_t)n_rows * K, cudaMemcpyHostToDevice);
          cudaMemcpy(k_rows_d, k_rows.data(), (size_t)n_rows * M, cudaMemcpyHostToDevice);
          if (!chain_offsets.empty())
          {
            cudaMalloc((void **)&chain_offsets_d, sizeof(unsigned long long) * chain_offsets.size());
            cudaMemcpy(chain_offsets_d, chain_offsets.data(),
              sizeof(unsigned long long) * chain_offsets.size(), cudaMemcpyHostToDevice);
          }
          std::vector<mat_float>().swap(values_h_staging);
        }
        const index_t *rows_ptr_d = A->get_rows_ptr_d();
        const index_t *cols_ind_d = A->get_cols_ind_d();
        const index_t *diag_ind_d = A->get_diag_ind_d();
        const mat_float *values_d = A->get_values_d();
        const int grid = (int)((n_rows + SE_BLOCK - 1) / SE_BLOCK);
        cudaMemset(flags_d, 0, 2 * sizeof(int));
        se_factors_kernel<N, K, M><<<grid, SE_BLOCK>>>(n_rows, diag_ind_d, values_d,
          e_rows_d, elim_cols_d, keep_cols_d, Pinv_d, Gm_d, pivot_eps, flags_d);
        se_condense_kernel<N, K, M><<<grid, SE_BLOCK>>>(n_rows, rows_ptr_d, cols_ind_d,
          values_d, k_rows_d, e_rows_d, elim_cols_d, keep_cols_d, Gm_d, reduced->values_d,
          chain_offsets_d, (int)chain_offsets.size(), flags_d);
        int flags[2] = {0, 0};
        cudaMemcpy(flags, flags_d, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        if (flags[0])
        {
          printf("linsolv_schur_elim: a K x K pivot degenerated on the current "
                 "Jacobian (row/col selected at first setup no longer usable); "
                 "disable local (Schur) elimination\n");
          return 1;
        }
        if (flags[1])
        {
          printf("linsolv_schur_elim: an eliminated-row off-diagonal coupling "
                 "appeared that was zero at detection (well control change?); the "
                 "recorded chain topology is stale -- disable local (Schur) elimination "
                 "or re-init the solver\n");
          return 1;
        }

        if (!chain_fixes.empty())
        {
          // copy ONLY the touched rows' factors (O(wells)), not the full arrays
          for (const index_t r : chain_touched)
          {
            cudaMemcpy(&Pinv[(size_t)r * K * K], Pinv_d + (size_t)r * K * K,
              sizeof(mat_float) * K * K, cudaMemcpyDeviceToHost);
            cudaMemcpy(&Gm[(size_t)r * K * M], Gm_d + (size_t)r * K * M,
              sizeof(mat_float) * K * M, cudaMemcpyDeviceToHost);
          }
          refresh_chain_coeffs(values_d, /*values_on_device=*/true);
          for (const auto &fx : chain_fixes)
          {
            const auto &ch = chains[fx.chain_idx];
            mat_float Ablk[N * N], Sblk[M * M];
            cudaMemcpy(Ablk, values_d + (size_t)fx.src_blk * N * N, sizeof(mat_float) * N * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(Sblk, reduced->values_d + (size_t)fx.dst_blk * M * M, sizeof(mat_float) * M * M, cudaMemcpyDeviceToHost);
            const uint8_t *kr = &k_rows[(size_t)fx.row_k * M];
            const mat_float *Pi = &Pinv[(size_t)ch.row * K * K];
            for (int rr = 0; rr < M; rr++)
            {
              const mat_float *Ar = Ablk + kr[rr] * N;
              mat_float w = 0;
              for (int e = 0; e < K; e++) w += Ar[elim_cols[e]] * Pi[e * K + ch.eidx];
              const mat_float wc = w * ch.coeff;
              if (ch.dep_eliminated)
              {
                const mat_float *Gd = &Gm[(size_t)ch.dep * K * M];
                for (int cc = 0; cc < M; cc++) Sblk[rr * M + cc] += wc * Gd[ch.dep_epos * M + cc];
              }
              else
                Sblk[rr * M + ch.dep_kpos] -= wc;
            }
            cudaMemcpy(reduced->values_d + (size_t)fx.dst_blk * M * M, Sblk, sizeof(mat_float) * M * M, cudaMemcpyHostToDevice);
          }
        }
        return inner->setup(reduced);
      }
#endif
      const mat_float *values_h = A->get_values();
      if (!detected && detect_rows(values_h)) return 1;
      if (condense_host(values_h)) return 1;
      return inner->setup(reduced);
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    int linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::reduce_rhs_host(const mat_float *B)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      const index_t *rows_ptr = A_saved->get_rows_ptr();
      const index_t *cols_ind = A_saved->get_cols_ind();
      const mat_float *values_h = A_saved->get_values();

      // h_i = Pinv_i . b_i[e_rows]  (local)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t i = 0; i < n_rows; i++)
      {
        const uint8_t *er = &e_rows[(size_t)i * K];
        const mat_float *Pi = &Pinv[(size_t)i * K * K];
        mat_float *hi = &h_eff[(size_t)i * K];
        for (int a = 0; a < K; a++)
        {
          mat_float s = 0;
          for (int b = 0; b < K; b++) s += Pi[a * K + b] * B[(size_t)i * N + er[b]];
          hi[a] = s;
        }
      }
      // eliminated-dep chain recurrence: h_i -= Pinv_i[:,eidx] * coeff * h_dep[dep_epos]
      for (const auto &ch : chains)
      {
        if (!ch.dep_eliminated) continue;
        const mat_float *Pi = &Pinv[(size_t)ch.row * K * K];
        const mat_float hd = h_eff[(size_t)ch.dep * K + ch.dep_epos];
        mat_float *hi = &h_eff[(size_t)ch.row * K];
        for (int a = 0; a < K; a++) hi[a] -= Pi[a * K + ch.eidx] * ch.coeff * hd;
      }

      // b_red[k][cc] = B[k][k_rows[cc]] - sum_j sum_e A_kj[k_rows[cc], elim_cols[e]] * h_j[e]
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t k = 0; k < n_rows; k++)
      {
        const uint8_t *kr = &k_rows[(size_t)k * M];
        mat_float acc[M];
        for (int rr = 0; rr < M; rr++) acc[rr] = B[(size_t)k * N + kr[rr]];
        for (index_t b = rows_ptr[k]; b < rows_ptr[k + 1]; b++)
        {
          const mat_float *A = values_h + (size_t)b * N * N;
          const mat_float *hj = &h_eff[(size_t)cols_ind[b] * K];
          for (int rr = 0; rr < M; rr++)
          {
            const mat_float *Ar = A + kr[rr] * N;
            mat_float s = 0;
            for (int e = 0; e < K; e++) s += Ar[elim_cols[e]] * hj[e];
            acc[rr] -= s;
          }
        }
        for (int rr = 0; rr < M; rr++) b_red[(size_t)k * M + rr] = acc[rr];
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    int linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::backsub_host(const mat_float *B, mat_float *X)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
      // local: x[keep_cols] = x_red; x[elim_cols] = h - Gm x_red
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (index_t i = 0; i < n_rows; i++)
      {
        const mat_float *xr = &x_red[(size_t)i * M];
        const mat_float *Gi = &Gm[(size_t)i * K * M];
        const mat_float *hi = &h_eff[(size_t)i * K];
        for (int cc = 0; cc < M; cc++) X[(size_t)i * N + keep_cols[cc]] = xr[cc];
        for (int a = 0; a < K; a++)
        {
          mat_float xm = hi[a];
          for (int cc = 0; cc < M; cc++) xm -= Gi[a * M + cc] * xr[cc];
          X[(size_t)i * N + elim_cols[a]] = xm;
        }
      }
      // chained rows: recompute x_i[elim_cols] with the FINAL dependency values
      // x_i[elim] = Pinv_i(b_i[e_rows] - sum_chains e_eidx coeff x_dep[dep_col]) - Gm_i x_red_i
      for (index_t i = 0; i < n_rows; i++)
      {
        // does cell i carry any chain?
        bool has = false;
        for (const auto &ch : chains) if (ch.row == i) { has = true; break; }
        if (!has) continue;
        const mat_float *er_b = B + (size_t)i * N;
        const uint8_t *er = &e_rows[(size_t)i * K];
        mat_float rhs_e[K];
        for (int a = 0; a < K; a++) rhs_e[a] = er_b[er[a]];
        for (const auto &ch : chains)
        {
          if (ch.row != i) continue;
          const mat_float xdep = ch.dep_eliminated
              ? X[(size_t)ch.dep * N + elim_cols[ch.dep_epos]]
              : x_red[(size_t)ch.dep * M + ch.dep_kpos];
          rhs_e[ch.eidx] -= ch.coeff * xdep;
        }
        const mat_float *Pi = &Pinv[(size_t)i * K * K];
        const mat_float *Gi = &Gm[(size_t)i * K * M];
        const mat_float *xr = &x_red[(size_t)i * M];
        for (int a = 0; a < K; a++)
        {
          mat_float xm = 0;
          for (int b = 0; b < K; b++) xm += Pi[a * K + b] * rhs_e[b];
          for (int cc = 0; cc < M; cc++) xm -= Gi[a * M + cc] * xr[cc];
          X[(size_t)i * N + elim_cols[a]] = xm;
        }
      }
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE, uint8_t N_ELIM>
    int linsolv_schur_elim<N_BLOCK_SIZE, N_ELIM>::solve(mat_float *B, mat_float *X)
    {
      constexpr uint8_t N = N_BLOCK_SIZE;
      constexpr uint8_t M = M_BLOCK_SIZE;
#ifdef WITH_GPU
      if (on_device)
      {
        const index_t *rows_ptr_d = A_saved->get_rows_ptr_d();
        const index_t *cols_ind_d = A_saved->get_cols_ind_d();
        const mat_float *values_d = A_saved->get_values_d();
        const int grid = (int)((n_rows + SE_BLOCK - 1) / SE_BLOCK);
        se_h_kernel<N, K><<<grid, SE_BLOCK>>>(n_rows, B, e_rows_d, Pinv_d, h_eff_d);
        // eliminated-dep chain recurrence on host (O(wells))
        bool any_elim_chain = false;
        for (const auto &ch : chains) if (ch.dep_eliminated) { any_elim_chain = true; break; }
        if (any_elim_chain)
        {
          cudaMemcpy(h_eff.data(), h_eff_d, sizeof(mat_float) * (size_t)n_rows * K, cudaMemcpyDeviceToHost);
          for (const auto &ch : chains)
          {
            if (!ch.dep_eliminated) continue;
            const mat_float *Pi = &Pinv[(size_t)ch.row * K * K];
            const mat_float hd = h_eff[(size_t)ch.dep * K + ch.dep_epos];
            for (int a = 0; a < K; a++) h_eff[(size_t)ch.row * K + a] -= Pi[a * K + ch.eidx] * ch.coeff * hd;
            cudaMemcpy(h_eff_d + (size_t)ch.row * K, &h_eff[(size_t)ch.row * K], sizeof(mat_float) * K, cudaMemcpyHostToDevice);
          }
        }
        se_reduce_rhs_kernel<N, K, M><<<grid, SE_BLOCK>>>(n_rows, rows_ptr_d, cols_ind_d,
          values_d, k_rows_d, elim_cols_d, h_eff_d, B, b_red_d);
        const int rc = inner->solve(b_red_d, x_red_d);
        if (rc) return rc;
        se_backsub_kernel<N, K, M><<<grid, SE_BLOCK>>>(n_rows, x_red_d, elim_cols_d, keep_cols_d, Gm_d, h_eff_d, X);
        // chained rows: recompute x[elim] with final dep values (host, O(wells)).
        // A cell can carry several chains (multiple eliminated rows / neighbours);
        // recompute each such cell ONCE -- the inner loop below already folds in
        // all of its chains -- so de-duplicate the row list first.
        std::vector<index_t> chain_rows;
        for (const auto &ch : chains)
          if (chain_rows.empty() || std::find(chain_rows.begin(), chain_rows.end(), ch.row) == chain_rows.end())
            chain_rows.push_back(ch.row);
        if (!chain_rows.empty())
        {
          // copy ONLY the rows the chain recurrence touches (O(wells)), not the
          // whole reduced solution
          for (const index_t r : chain_touched)
            cudaMemcpy(&x_red[(size_t)r * M], x_red_d + (size_t)r * M,
              sizeof(mat_float) * M, cudaMemcpyDeviceToHost);
          for (index_t i : chain_rows)
          {
            mat_float rhs_e[K], bb[N];
            cudaMemcpy(bb, B + (size_t)i * N, sizeof(mat_float) * N, cudaMemcpyDeviceToHost);
            const uint8_t *er = &e_rows[(size_t)i * K];
            for (int a = 0; a < K; a++) rhs_e[a] = bb[er[a]];
            for (const auto &ch : chains)
            {
              if (ch.row != i) continue;
              mat_float xdep;
              if (ch.dep_eliminated)
                cudaMemcpy(&xdep, X + (size_t)ch.dep * N + elim_cols[ch.dep_epos], sizeof(mat_float), cudaMemcpyDeviceToHost);
              else
                xdep = x_red[(size_t)ch.dep * M + ch.dep_kpos];
              rhs_e[ch.eidx] -= ch.coeff * xdep;
            }
            const mat_float *Pi = &Pinv[(size_t)i * K * K];
            const mat_float *Gi = &Gm[(size_t)i * K * M];
            const mat_float *xr = &x_red[(size_t)i * M];
            mat_float xelim[K];
            for (int a = 0; a < K; a++)
            {
              mat_float xm = 0;
              for (int b = 0; b < K; b++) xm += Pi[a * K + b] * rhs_e[b];
              for (int cc = 0; cc < M; cc++) xm -= Gi[a * M + cc] * xr[cc];
              xelim[a] = xm;
            }
            for (int a = 0; a < K; a++)
              cudaMemcpy(X + (size_t)i * N + elim_cols[a], &xelim[a], sizeof(mat_float), cudaMemcpyHostToDevice);
          }
        }
        return 0;
      }
#endif
      if (reduce_rhs_host(B)) return 1;
      const int rc = inner->solve(b_red.data(), x_red.data());
      if (rc) return rc;
      return backsub_host(B, X);
    }

    // Explicit instantiations for (N, K): N in 2..OD_SE_NMAX, K in 1..min(4, N-1).
    //
    // OD_SE_NMAX must cover the largest full block size the engine can request:
    // for the super engine that is MAX_NC + 1 (thermal), and the CPU factory
    // dispatches up to 13. The linear_solvers CMake passes -DOD_SE_NMAX =
    // max(OPENDARTS_MAX_DIMS+1, 13) so this stays in lock-step with the single-
    // parameter solvers (default build: 13). Blocks for N > OD_SE_NMAX are
    // preprocessed out to avoid compiling unused instantiations.
#ifndef OD_SE_NMAX
#define OD_SE_NMAX 13
#endif
#if OD_SE_NMAX > 16
#error "OD_SE_NMAX exceeds the written instantiation ceiling (16); extend the OD_SE_INST list below."
#endif
#define OD_SE_INST(N, K) template class linsolv_schur_elim<N, K>;
    OD_SE_INST(2, 1)
    OD_SE_INST(3, 1) OD_SE_INST(3, 2)
    OD_SE_INST(4, 1) OD_SE_INST(4, 2) OD_SE_INST(4, 3)
    OD_SE_INST(5, 1) OD_SE_INST(5, 2) OD_SE_INST(5, 3) OD_SE_INST(5, 4)
    OD_SE_INST(6, 1) OD_SE_INST(6, 2) OD_SE_INST(6, 3) OD_SE_INST(6, 4)
    OD_SE_INST(7, 1) OD_SE_INST(7, 2) OD_SE_INST(7, 3) OD_SE_INST(7, 4)
    OD_SE_INST(8, 1) OD_SE_INST(8, 2) OD_SE_INST(8, 3) OD_SE_INST(8, 4)
    OD_SE_INST(9, 1) OD_SE_INST(9, 2) OD_SE_INST(9, 3) OD_SE_INST(9, 4)
    OD_SE_INST(10, 1) OD_SE_INST(10, 2) OD_SE_INST(10, 3) OD_SE_INST(10, 4)
    OD_SE_INST(11, 1) OD_SE_INST(11, 2) OD_SE_INST(11, 3) OD_SE_INST(11, 4)
    OD_SE_INST(12, 1) OD_SE_INST(12, 2) OD_SE_INST(12, 3) OD_SE_INST(12, 4)
    OD_SE_INST(13, 1) OD_SE_INST(13, 2) OD_SE_INST(13, 3) OD_SE_INST(13, 4)
#if OD_SE_NMAX >= 14
    OD_SE_INST(14, 1) OD_SE_INST(14, 2) OD_SE_INST(14, 3) OD_SE_INST(14, 4)
#endif
#if OD_SE_NMAX >= 15
    OD_SE_INST(15, 1) OD_SE_INST(15, 2) OD_SE_INST(15, 3) OD_SE_INST(15, 4)
#endif
#if OD_SE_NMAX >= 16
    OD_SE_INST(16, 1) OD_SE_INST(16, 2) OD_SE_INST(16, 3) OD_SE_INST(16, 4)
#endif
#undef OD_SE_INST
  } // namespace linear_solvers
} // namespace opendarts
