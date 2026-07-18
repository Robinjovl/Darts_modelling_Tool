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

#ifdef WITH_GPU

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "linsolv_mcsgs.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      using opendarts::config::index_t;
      using opendarts::config::mat_float;

      constexpr int MCSGS_BLOCK = 256;

      // One thread per block row: Gauss-Jordan inversion of the NxN diagonal
      // block with partial pivoting; numerically singular blocks (well-control
      // rows can carry an exactly-zero column) fall back to the identity.
      template <uint8_t N>
      __global__ void mcsgs_invert_diag_kernel(index_t n_rows, const index_t *diag_ind,
        const mat_float *values, mat_float *inv_diag)
      {
        const index_t i = blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (i >= n_rows)
          return;

        mat_float a[N * N], inv[N * N];
        const mat_float *db = values + (size_t)diag_ind[i] * N * N;
        for (int k = 0; k < N * N; k++)
        {
          a[k] = db[k];
          inv[k] = 0;
        }
        for (int k = 0; k < N; k++)
          inv[k * N + k] = 1;

        bool ok = true;
        for (int col = 0; col < N && ok; col++)
        {
          int piv = col;
          mat_float pv = fabs(a[col * N + col]);
          for (int r = col + 1; r < N; r++)
          {
            const mat_float v = fabs(a[r * N + col]);
            if (v > pv)
            {
              pv = v;
              piv = r;
            }
          }
          if (!(pv > 0) || !isfinite(pv))
          {
            ok = false;
            break;
          }
          if (piv != col)
            for (int c = 0; c < N; c++)
            {
              mat_float t = a[col * N + c];
              a[col * N + c] = a[piv * N + c];
              a[piv * N + c] = t;
              t = inv[col * N + c];
              inv[col * N + c] = inv[piv * N + c];
              inv[piv * N + c] = t;
            }
          const mat_float d = mat_float(1) / a[col * N + col];
          for (int c = 0; c < N; c++)
          {
            a[col * N + c] *= d;
            inv[col * N + c] *= d;
          }
          for (int r = 0; r < N; r++)
          {
            if (r == col)
              continue;
            const mat_float f = a[r * N + col];
            if (f != 0)
              for (int c = 0; c < N; c++)
              {
                a[r * N + c] -= f * a[col * N + c];
                inv[r * N + c] -= f * inv[col * N + c];
              }
          }
        }

        mat_float *out = inv_diag + (size_t)i * N * N;
        if (!ok)
        {
          for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
              out[r * N + c] = (r == c) ? mat_float(1) : mat_float(0);
        }
        else
        {
          for (int k = 0; k < N * N; k++)
            out[k] = inv[k];
        }
      }

      // Gauss-Seidel update of one color class: for row i of the class,
      // x_i = D_i^{-1} (b_i - sum_{j != i} A_ij x_j). Rows of other classes
      // contribute their latest value (0 for classes not yet visited in this
      // sweep, since x starts from zero).
      template <uint8_t N>
      __global__ void mcsgs_color_sweep_kernel(index_t color_begin, index_t color_end,
        const index_t *color_rows, const index_t *rows_ptr, const index_t *cols_ind,
        const index_t *diag_ind, const mat_float *values, const mat_float *inv_diag,
        const mat_float *B, mat_float *X)
      {
        const index_t idx = color_begin + blockIdx.x * (index_t)blockDim.x + threadIdx.x;
        if (idx >= color_end)
          return;
        const index_t i = color_rows[idx];

        mat_float r[N];
        for (int c = 0; c < N; c++)
          r[c] = B[(size_t)i * N + c];

        const index_t row_begin = rows_ptr[i], row_end = rows_ptr[i + 1], dj = diag_ind[i];
        for (index_t j = row_begin; j < row_end; j++)
        {
          if (j == dj)
            continue;
          const index_t col = cols_ind[j];
          const mat_float *blk = values + (size_t)j * N * N;
          for (int c = 0; c < N; c++)
          {
            mat_float s = 0;
            for (int v = 0; v < N; v++)
              s += blk[c * N + v] * X[(size_t)col * N + v];
            r[c] -= s;
          }
        }

        const mat_float *Di = inv_diag + (size_t)i * N * N;
        for (int c = 0; c < N; c++)
        {
          mat_float s = 0;
          for (int v = 0; v < N; v++)
            s += Di[c * N + v] * r[v];
          X[(size_t)i * N + c] = s;
        }
      }
    } // namespace

    template <uint8_t N_BLOCK_SIZE>
    linsolv_mcsgs<N_BLOCK_SIZE>::linsolv_mcsgs()
    {
      // Register as the linear_solver_base behind the block interface.
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::solver = this;
      const char *sw = std::getenv("DARTS_MCSGS_SWEEPS");
      n_sweeps = sw ? atoi(sw) : 1;
      if (n_sweeps < 1)
        n_sweeps = 1;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_mcsgs<N_BLOCK_SIZE>::~linsolv_mcsgs()
    {
      free_all();
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mcsgs<N_BLOCK_SIZE>::free_all()
    {
      if (color_rows_d)
        cudaFree(color_rows_d);
      if (inv_diag_d)
        cudaFree(inv_diag_d);
      color_rows_d = nullptr;
      inv_diag_d = nullptr;
      color_offsets.clear();
      n_colors = 0;
      n_rows = 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mcsgs<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A,
      opendarts::config::index_t /*max_iters*/,
      opendarts::config::mat_float /*tolerance*/)
    {
      free_all(); // repeated init (adjoint re-init) must not leak / keep stale coloring

      A_saved = A;
      n_rows = A->n_rows;

      // Greedy coloring on the host sparsity. The Jacobian pattern is
      // structurally symmetric (flow terms and well head<->body couplings both
      // appear in both rows), so out-neighbors are sufficient.
      const index_t *rows_ptr = A->get_rows_ptr();
      const index_t *cols_ind = A->get_cols_ind();
      std::vector<int> color(n_rows, -1);
      int max_color = -1;
      {
        std::vector<unsigned char> used(256, 0);
        for (index_t i = 0; i < n_rows; i++)
        {
          for (index_t j = rows_ptr[i]; j < rows_ptr[i + 1]; j++)
          {
            const index_t nb = cols_ind[j];
            if (nb != i && color[nb] >= 0 && color[nb] < 256)
              used[color[nb]] = 1;
          }
          int c = 0;
          while (c < 256 && used[c])
            c++;
          color[i] = c;
          if (c > max_color)
            max_color = c;
          for (index_t j = rows_ptr[i]; j < rows_ptr[i + 1]; j++)
          {
            const index_t nb = cols_ind[j];
            if (nb != i && color[nb] >= 0 && color[nb] < 256)
              used[color[nb]] = 0;
          }
        }
      }
      n_colors = max_color + 1;

      // Group rows by color (stable order within a color).
      color_offsets.assign(n_colors + 1, 0);
      for (index_t i = 0; i < n_rows; i++)
        color_offsets[color[i] + 1]++;
      for (int c = 0; c < n_colors; c++)
        color_offsets[c + 1] += color_offsets[c];
      std::vector<index_t> color_rows(n_rows);
      {
        std::vector<index_t> cursor(color_offsets.begin(), color_offsets.end() - 1);
        for (index_t i = 0; i < n_rows; i++)
          color_rows[cursor[color[i]]++] = i;
      }

      cudaMalloc(&color_rows_d, sizeof(index_t) * n_rows);
      cudaMemcpy(color_rows_d, color_rows.data(), sizeof(index_t) * n_rows, cudaMemcpyHostToDevice);
      cudaMalloc(&inv_diag_d, sizeof(mat_float) * (size_t)n_rows * N_BLOCK_SIZE * N_BLOCK_SIZE);

      printf("MCSGS<%d>: %d colors over %d rows\n", (int)N_BLOCK_SIZE, n_colors, (int)n_rows);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mcsgs<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A)
    {
      this->timer_setup->node["MCSGS"].start();
      A_saved = A;
      const index_t grid = (n_rows + MCSGS_BLOCK - 1) / MCSGS_BLOCK;
      mcsgs_invert_diag_kernel<N_BLOCK_SIZE><<<grid, MCSGS_BLOCK>>>(
        n_rows, A->get_diag_ind_d(), A->get_values_d(), inv_diag_d);
      this->timer_setup->node["MCSGS"].stop();
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mcsgs<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      this->timer_solve->node["MCSGS"].start();
      opendarts::linear_solvers::csr_matrix_base *A = A_saved;
      const index_t *rows_ptr_d = A->get_rows_ptr_d();
      const index_t *cols_ind_d = A->get_cols_ind_d();
      const index_t *diag_ind_d = A->get_diag_ind_d();
      const mat_float *values_d = A->get_values_d();

      cudaMemset(X, 0, sizeof(mat_float) * (size_t)n_rows * N_BLOCK_SIZE);

      auto sweep_color = [&](int c)
      {
        const index_t begin = color_offsets[c], end = color_offsets[c + 1];
        if (end == begin)
          return;
        const index_t grid = (end - begin + MCSGS_BLOCK - 1) / MCSGS_BLOCK;
        mcsgs_color_sweep_kernel<N_BLOCK_SIZE><<<grid, MCSGS_BLOCK>>>(
          begin, end, color_rows_d, rows_ptr_d, cols_ind_d, diag_ind_d, values_d, inv_diag_d, B, X);
      };

      for (int s = 0; s < n_sweeps; s++)
      {
        for (int c = 0; c < n_colors; c++)
          sweep_color(c);
        for (int c = n_colors - 1; c >= 0; c--)
          sweep_color(c);
      }

      this->timer_solve->node["MCSGS"].stop();
      return 0;
    }

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see linear_solvers/src/CMakeLists.txt.
#include "linsolv_mcsgs_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU
