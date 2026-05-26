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

#ifdef WITH_GPU

#include <cmath>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "linsolv_bos_cpr_gpu.hpp"
#include "csr_matrix.hpp"
#include "linear_solvers_data_types.hpp"

// Active code path: accurate block inversion. The PRESSURE_EQUATION_SUM and
// SATURATION_EQUATIONS_START_FIRST variants are kept (via #ifdef) for parity
// with the proprietary source but are not enabled.
#define ACCURATE_INVERSION

namespace opendarts
{
  namespace linear_solvers
  {
    using opendarts::config::index_t;
    typedef opendarts::config::mat_float value_t;

    // Linear-solve counter, kept for parity with the proprietary source.
    static int lin_it = 0;

    // In-place block inversion (ADGPRS routine): inverts the n_block_size x
    // n_block_size dense block D with full pivoting. Returns false on a
    // singular block. is/js point at a 2*n_block_size scratch array.
    template <uint8_t n_block_size>
    __device__ bool calcDinv_Opt_obls(double *D, int ldd, int *is)
    {
      int i, j, k;
      double d, p;
      double *D_l, *D_u, *D_v;
      double *D_kldd = D;

      int *const js = is + n_block_size; // is[n]: first scratch half; js[n]: second half

      for (k = 0; k < n_block_size; k++)
      {
        // --- choose the diagonal term ---
        d = 0.0;
        for (i = k; i < n_block_size; i++)
        {
          for (j = k, D_l = D + (i * ldd + k); j < n_block_size; j++, ++D_l)
          {
            p = fabs(*D_l);
            if (p > d)
            {
              d = p;
              is[k] = i;
              js[k] = j;
            }
          }
        }

        if (d + 1.0 == 1.0)
        { // singularity is encountered
          printf("CPR error: Matrix D can't be inversed\n");
          return false;
        }

        // --- start forward elimination ---
        if (is[k] != k)
        {
          for (j = 0, D_u = D_kldd, D_v = D + is[k] * ldd; j < n_block_size; j++, ++D_u, ++D_v)
          {
            p = *D_u;
            *D_u = *D_v;
            *D_v = p;
          }
        }

        if (js[k] != k)
        {
          for (i = 0, D_u = D + k, D_v = D + js[k]; i < n_block_size; i++, D_u += ldd, D_v += ldd)
          {
            p = *D_u;
            *D_u = *D_v;
            *D_v = p;
          }
        }

        D_l = D_kldd + k;
        *D_l = 1.0 / *D_l;

        for (j = 0; j < n_block_size; j++)
          if (j != k)
            D_kldd[j] *= *D_l;

        for (i = 0, D_u = D; i < n_block_size; i++, D_u += ldd)
        {
          if (i != k)
          {
            for (j = 0; j < n_block_size; j++)
            {
              if (j != k)
                D_u[j] -= D_u[k] * D_kldd[j];
            }
          }
        }

        for (i = 0, D_u = D + k; i < n_block_size; i++, D_u += ldd)
          if (i != k)
            *D_u *= -(*D_l);

        D_kldd += ldd; // equivalent to k*ldd when k++
      }

      // --- do backward elimination ---
      for (k = n_block_size - 1; k >= 0; k--)
      {
        if (js[k] != k)
        {
          for (j = 0, D_u = D + k * ldd, D_v = D + js[k] * ldd; j < n_block_size; j++, ++D_u, ++D_v)
          {
            p = *D_u;
            *D_u = *D_v;
            *D_v = p;
          }
        }

        if (is[k] != k)
        {
          for (i = 0, D_u = D + k, D_v = D + is[k]; i < n_block_size; i++, D_u += ldd, D_v += ldd)
          {
            p = *D_u;
            *D_u = *D_v;
            *D_v = p;
          }
        }
      }
      return true; // inversion succeeds
    }

    // CPR setup kernel: builds the reduced pressure matrix values (p_vals) and
    // the D_ps * inv(D_ss) constraint. This variant assumes the diagonal block
    // is stored first in each row.
    template <uint8_t n_block_size>
    __global__ void cpr_setup_kernel(index_t n_rows, index_t *rows, index_t *diags, index_t *block_p_jac_idx,
      value_t *vals, value_t *p_vals,
      value_t *D_ps_ss, value_t *rhs_mults, index_t *inverse_fail_counter, value_t *diag_acc = 0)
    {
      index_t i = threadIdx.x + blockIdx.x * blockDim.x;

      if (i >= n_rows)
        return;

      value_t colsum_block[(n_block_size) * (n_block_size - 1)];
      // effectively the number of off-diagonal nonzeros for all previous blocks
      index_t conn = rows[i] - i;

#ifdef ACCURATE_INVERSION
      int inv_wrkspc[2 * (n_block_size - 1)];
      value_t tmp_val;
#endif

      // 1. Work out diagonal: colsum + D_sp * inv(D_ss) -> D_spss
      // Make the last equation the "pressure" so D_ss has a non-zero diagonal.
#ifdef SATURATION_EQUATIONS_START_FIRST
      value_t *D_ss = colsum_block;
      value_t *D_ps = colsum_block + (n_block_size - 1) * (n_block_size - 1);
#else
      value_t *D_ps = colsum_block;
      value_t *D_ss = colsum_block + (n_block_size - 1);
#endif

      index_t j1 = rows[i];
      index_t j2 = rows[i + 1];
      index_t jd = diags[i];

      // colsum into the work block, initialised with the diagonal values
      for (int v = 1; v < n_block_size; ++v)
        for (int e = 0; e < n_block_size; ++e)
        {
          colsum_block[e * (n_block_size - 1) + v - 1] = vals[jd * n_block_size * n_block_size + e * n_block_size + v];
        }

      // add up all column off-diagonal non-pressure derivatives
      for (index_t j = j1; j < j2; ++j)
      {
        if (j != jd)
        {
          index_t jo = block_p_jac_idx[conn++];
          for (int v = 1; v < n_block_size; ++v)
            for (int e = 0; e < n_block_size; ++e)
            {
              colsum_block[e * (n_block_size - 1) + v - 1] += vals[jo + e * n_block_size + v];
            }
        }
      }

      if (diag_acc)
      {
        index_t jda = i * n_block_size * n_block_size;
        for (int e = 0; e < n_block_size; ++e)
        {
          diag_acc[jda + e * n_block_size] = vals[jd * n_block_size * n_block_size + e * n_block_size];
        }
        for (int v = 1; v < n_block_size; ++v)
        {
          for (int e = 0; e < n_block_size; ++e)
          {
            diag_acc[jda + e * n_block_size + v] = colsum_block[e * (n_block_size - 1) + v - 1];
          }
        }
      }

#ifdef PRESSURE_EQUATION_SUM
      for (int e = 0; e < n_block_size - 1; ++e)
      {
        for (int v = 0; v < n_block_size - 1; ++v)
        {
          D_ps[v] += colsum_block[e * (n_block_size - 1) + v];
        }
      }
#endif

      // Now calculate D_ps = D_ps * inv(D_ss).
#ifdef ACCURATE_INVERSION
      // transpose D_ss to make it column-major
      for (int e = 0; e < (n_block_size - 1); ++e)
      {
        for (int v = e; v < (n_block_size - 1); ++v)
        {
          if (e != v)
          {
            tmp_val = D_ss[v + e * (n_block_size - 1)];
            D_ss[v + e * (n_block_size - 1)] = D_ss[e + v * (n_block_size - 1)];
            D_ss[e + v * (n_block_size - 1)] = tmp_val;
          }
        }
      }
      if (!calcDinv_Opt_obls<n_block_size - 1>(D_ss, n_block_size - 1, inv_wrkspc))
      {
        atomicAdd(inverse_fail_counter, 1);
        return;
      }

      // finally calc D_ps = D_ps * inv(D_ss)
      for (int e = 0; e < (n_block_size - 1); ++e)
        for (int v = 0; v < (n_block_size - 1); ++v)
          D_ps_ss[i * (n_block_size - 1) + e] += D_ps[v] * D_ss[e * (n_block_size - 1) + v];
#endif

      // 2. Calculate pressure values A_pp -= A_sp * (D_ps * inv(D_ss)).
      // The diagonal block must be placed first for the BOS AMG solver.
      j1 = rows[i];
      j2 = rows[i + 1];
      jd = diags[i];

      // lower triangle
      for (index_t j = j1; j < jd; ++j)
      {
#if defined(SATURATION_EQUATIONS_START_FIRST) || defined(PRESSURE_EQUATION_SUM)
        p_vals[j + 1] = vals[j * n_block_size * n_block_size + n_block_size * (n_block_size - 1)];
#else
        p_vals[j + 1] = vals[j * n_block_size * n_block_size];
#endif
#ifdef PRESSURE_EQUATION_SUM
        for (int e = 0; e < n_block_size - 1; ++e)
        {
          p_vals[j + 1] += vals[j * n_block_size * n_block_size + n_block_size * e];
        }
#endif
        for (int e = 0; e < (n_block_size - 1); ++e)
        {
#ifdef SATURATION_EQUATIONS_START_FIRST
          p_vals[j + 1] -= vals[j * n_block_size * n_block_size + n_block_size * e] * D_ps_ss[i * (n_block_size - 1) + e];
#else
          p_vals[j + 1] -= vals[j * n_block_size * n_block_size + n_block_size * (e + 1)] * D_ps_ss[i * (n_block_size - 1) + e];
#endif
        }
      }
      // diagonal
#if defined(SATURATION_EQUATIONS_START_FIRST) || defined(PRESSURE_EQUATION_SUM)
      p_vals[j1] = vals[jd * n_block_size * n_block_size + n_block_size * (n_block_size - 1)];
#else
      p_vals[j1] = vals[jd * n_block_size * n_block_size];
#endif
#ifdef PRESSURE_EQUATION_SUM
      for (int e = 0; e < n_block_size - 1; ++e)
      {
        p_vals[j1] += vals[jd * n_block_size * n_block_size + n_block_size * e];
      }
#endif
      for (int e = 0; e < (n_block_size - 1); ++e)
      {
#ifdef SATURATION_EQUATIONS_START_FIRST
        p_vals[j1] -= vals[jd * n_block_size * n_block_size + n_block_size * e] * D_ps_ss[i * (n_block_size - 1) + e];
#else
        p_vals[j1] -= vals[jd * n_block_size * n_block_size + n_block_size * (e + 1)] * D_ps_ss[i * (n_block_size - 1) + e];
#endif
      }

      // upper triangle
      for (index_t j = jd + 1; j < j2; ++j)
      {
#if defined(SATURATION_EQUATIONS_START_FIRST) || defined(PRESSURE_EQUATION_SUM)
        p_vals[j] = vals[j * n_block_size * n_block_size + n_block_size * (n_block_size - 1)];
#else
        p_vals[j] = vals[j * n_block_size * n_block_size];
#endif
#ifdef PRESSURE_EQUATION_SUM
        for (int e = 0; e < n_block_size - 1; ++e)
        {
          p_vals[j] += vals[j * n_block_size * n_block_size + n_block_size * e];
        }
#endif
        for (int e = 0; e < (n_block_size - 1); ++e)
        {
#ifdef SATURATION_EQUATIONS_START_FIRST
          p_vals[j] -= vals[j * n_block_size * n_block_size + n_block_size * e] * D_ps_ss[i * (n_block_size - 1) + e];
#else
          p_vals[j] -= vals[j * n_block_size * n_block_size + n_block_size * (e + 1)] * D_ps_ss[i * (n_block_size - 1) + e];
#endif
        }
      }
      if (p_vals[j1] < 0)
      {
        rhs_mults[i] = -1;
        for (index_t j = j1; j < j2; ++j)
        {
          p_vals[j] *= -1;
        }
      }
      else
        rhs_mults[i] = 1;
    }

    // CPR setup kernel variant that does not assume a diagonal-first layout.
    template <uint8_t n_block_size>
    __global__ void cpr_setup_kernel_no_first_diag(index_t n_rows, index_t *rows, index_t *diags,
      index_t *block_p_jac_idx, value_t *vals, value_t *p_vals,
      value_t *D_ps_ss, value_t *rhs_mults, index_t *inverse_fail_counter, value_t *diag_acc = 0)
    {
      index_t i = threadIdx.x + blockIdx.x * blockDim.x;

      if (i >= n_rows)
        return;

      value_t colsum_block[(n_block_size) * (n_block_size - 1)];
      index_t conn = rows[i] - i;

#ifdef ACCURATE_INVERSION
      int inv_wrkspc[2 * (n_block_size - 1)];
      value_t tmp_val;
#endif

#ifdef SATURATION_EQUATIONS_START_FIRST
      value_t *D_ss = colsum_block;
      value_t *D_ps = colsum_block + (n_block_size - 1) * (n_block_size - 1);
#else
      value_t *D_ps = colsum_block;
      value_t *D_ss = colsum_block + (n_block_size - 1);
#endif

      index_t j1 = rows[i];
      index_t j2 = rows[i + 1];
      index_t jd = diags[i];

      for (int v = 1; v < n_block_size; ++v)
        for (int e = 0; e < n_block_size; ++e)
        {
          colsum_block[e * (n_block_size - 1) + v - 1] = vals[jd * n_block_size * n_block_size + e * n_block_size + v];
        }

      for (index_t j = j1; j < j2; ++j)
      {
        if (j != jd)
        {
          index_t jo = block_p_jac_idx[conn++];
          for (int v = 1; v < n_block_size; ++v)
            for (int e = 0; e < n_block_size; ++e)
            {
              colsum_block[e * (n_block_size - 1) + v - 1] += vals[jo + e * n_block_size + v];
            }
        }
      }

      if (diag_acc)
      {
        index_t jda = i * n_block_size * n_block_size;
        for (int e = 0; e < n_block_size; ++e)
        {
          diag_acc[jda + e * n_block_size] = vals[jd * n_block_size * n_block_size + e * n_block_size];
        }
        for (int v = 1; v < n_block_size; ++v)
        {
          for (int e = 0; e < n_block_size; ++e)
          {
            diag_acc[jda + e * n_block_size + v] = colsum_block[e * (n_block_size - 1) + v - 1];
          }
        }
      }

#ifdef PRESSURE_EQUATION_SUM
      for (int e = 0; e < n_block_size - 1; ++e)
      {
        for (int v = 0; v < n_block_size - 1; ++v)
        {
          D_ps[v] += colsum_block[e * (n_block_size - 1) + v];
        }
      }
#endif

#ifdef ACCURATE_INVERSION
      for (int e = 0; e < (n_block_size - 1); ++e)
      {
        for (int v = e; v < (n_block_size - 1); ++v)
        {
          if (e != v)
          {
            tmp_val = D_ss[v + e * (n_block_size - 1)];
            D_ss[v + e * (n_block_size - 1)] = D_ss[e + v * (n_block_size - 1)];
            D_ss[e + v * (n_block_size - 1)] = tmp_val;
          }
        }
      }
      if (!calcDinv_Opt_obls<n_block_size - 1>(D_ss, n_block_size - 1, inv_wrkspc))
      {
        atomicAdd(inverse_fail_counter, 1);
        return;
      }

      for (int e = 0; e < (n_block_size - 1); ++e)
        for (int v = 0; v < (n_block_size - 1); ++v)
          D_ps_ss[i * (n_block_size - 1) + e] += D_ps[v] * D_ss[e * (n_block_size - 1) + v];
#endif

      j1 = rows[i];
      j2 = rows[i + 1];
      jd = diags[i];

      // lower triangle
      for (index_t j = j1; j < jd; ++j)
      {
#if defined(SATURATION_EQUATIONS_START_FIRST) || defined(PRESSURE_EQUATION_SUM)
        p_vals[j] = vals[j * n_block_size * n_block_size + n_block_size * (n_block_size - 1)];
#else
        p_vals[j] = vals[j * n_block_size * n_block_size];
#endif
#ifdef PRESSURE_EQUATION_SUM
        for (int e = 0; e < n_block_size - 1; ++e)
        {
          p_vals[j] += vals[j * n_block_size * n_block_size + n_block_size * e];
        }
#endif
        for (int e = 0; e < (n_block_size - 1); ++e)
        {
#ifdef SATURATION_EQUATIONS_START_FIRST
          p_vals[j] -= vals[j * n_block_size * n_block_size + n_block_size * e] * D_ps_ss[i * (n_block_size - 1) + e];
#else
          p_vals[j] -= vals[j * n_block_size * n_block_size + n_block_size * (e + 1)] * D_ps_ss[i * (n_block_size - 1) + e];
#endif
        }
      }
      // diagonal
#if defined(SATURATION_EQUATIONS_START_FIRST) || defined(PRESSURE_EQUATION_SUM)
      p_vals[jd] = vals[jd * n_block_size * n_block_size + n_block_size * (n_block_size - 1)];
#else
      p_vals[jd] = vals[jd * n_block_size * n_block_size];
#endif
#ifdef PRESSURE_EQUATION_SUM
      for (int e = 0; e < n_block_size - 1; ++e)
      {
        p_vals[jd] += vals[jd * n_block_size * n_block_size + n_block_size * e];
      }
#endif
      for (int e = 0; e < (n_block_size - 1); ++e)
      {
#ifdef SATURATION_EQUATIONS_START_FIRST
        p_vals[jd] -= vals[jd * n_block_size * n_block_size + n_block_size * e] * D_ps_ss[i * (n_block_size - 1) + e];
#else
        p_vals[jd] -= vals[jd * n_block_size * n_block_size + n_block_size * (e + 1)] * D_ps_ss[i * (n_block_size - 1) + e];
#endif
      }

      // upper triangle
      for (index_t j = jd + 1; j < j2; ++j)
      {
#if defined(SATURATION_EQUATIONS_START_FIRST) || defined(PRESSURE_EQUATION_SUM)
        p_vals[j] = vals[j * n_block_size * n_block_size + n_block_size * (n_block_size - 1)];
#else
        p_vals[j] = vals[j * n_block_size * n_block_size];
#endif
#ifdef PRESSURE_EQUATION_SUM
        for (int e = 0; e < n_block_size - 1; ++e)
        {
          p_vals[j] += vals[j * n_block_size * n_block_size + n_block_size * e];
        }
#endif
        for (int e = 0; e < (n_block_size - 1); ++e)
        {
#ifdef SATURATION_EQUATIONS_START_FIRST
          p_vals[j] -= vals[j * n_block_size * n_block_size + n_block_size * e] * D_ps_ss[i * (n_block_size - 1) + e];
#else
          p_vals[j] -= vals[j * n_block_size * n_block_size + n_block_size * (e + 1)] * D_ps_ss[i * (n_block_size - 1) + e];
#endif
        }
      }
      if (p_vals[jd] < 0)
      {
        rhs_mults[i] = -1;
        for (index_t j = j1; j < j2; ++j)
        {
          p_vals[j] *= -1;
        }
      }
      else
        rhs_mults[i] = 1;
    }

    // Restrict the full RHS B to the pressure RHS P_B.
    template <uint8_t n_block_size>
    __global__ void cpr_solve_reduce_kernel(index_t n_rows, value_t *B, value_t *P_B,
      value_t *D_ps_ss, value_t *rhs_mults)
    {
      index_t i = threadIdx.x + blockIdx.x * blockDim.x;
      value_t P_B_local;

      if (i >= n_rows)
        return;

#if defined(SATURATION_EQUATIONS_START_FIRST) || defined(PRESSURE_EQUATION_SUM)
      P_B_local = B[i * n_block_size + (n_block_size - 1)];
#else
      P_B_local = B[i * n_block_size];
#endif

#ifdef PRESSURE_EQUATION_SUM
      for (int e = 0; e < n_block_size - 1; ++e)
      {
        P_B_local += B[i * n_block_size + e];
      }
#endif

      for (int e = 0; e < (n_block_size - 1); ++e)
      {
#ifdef SATURATION_EQUATIONS_START_FIRST
        P_B_local -= B[i * n_block_size + e] * D_ps_ss[i * (n_block_size - 1) + e];
#else
        P_B_local -= B[i * n_block_size + e + 1] * D_ps_ss[i * (n_block_size - 1) + e];
#endif
      }
      P_B_local *= rhs_mults[i];

      P_B[i] = P_B_local;
    }

    // Prolongate the pressure solution P_X into the pressure slot of X.
    template <uint8_t n_block_size>
    __global__ void cpr_solve_prolongate_kernel(index_t n_rows, value_t *X, value_t *P_X)
    {
      index_t i = threadIdx.x + blockIdx.x * blockDim.x;

      if (i >= n_rows)
        return;

      X[i * n_block_size] = P_X[i];
    }

    // Add the pressure solution P_X back into the pressure slot of X.
    template <uint8_t n_block_size>
    __global__ void cpr_solve_sum_up_kernel(index_t n_rows, value_t *X, value_t *P_X)
    {
      index_t i = threadIdx.x + blockIdx.x * blockDim.x;

      if (i >= n_rows)
        return;

      X[i * n_block_size] += P_X[i];
    }

    template <uint8_t n_block_size>
    linsolv_bos_cpr_gpu<n_block_size>::linsolv_bos_cpr_gpu()
    {
      opendarts::linear_solvers::linsolv_iface_bos<n_block_size>::solver = this;
      P = new opendarts::linear_solvers::csr_matrix<1>;

      A_base = nullptr;
      D_ps_ss = nullptr;
      block_p_jac_idx = nullptr;
      inverse_fail_counter = nullptr;
      P_B = nullptr;
      P_X = nullptr;
      P_B_h = nullptr;
      P_X_h = nullptr;
      full_B = nullptr;
      rhs_mults = nullptr;
      p_vals = nullptr;
      diag_acc = nullptr;
      p_system_preconditioner = nullptr;
      full_system_preconditioner = nullptr;
      setup_block_size = solve_reduce_block_size = 0;
      solve_prolongate_block_size = solve_sum_up_block_size = 0;
      min_grid_size = grid_size = 0;
    }

    template <uint8_t n_block_size>
    linsolv_bos_cpr_gpu<n_block_size>::~linsolv_bos_cpr_gpu()
    {
      delete (P);
      delete (full_system_preconditioner);
      if (D_ps_ss)
      {
        cudaFree(D_ps_ss);
        cudaFree(P_B);
        cudaFree(P_X);
        cudaFree(full_B);
        cudaFree(rhs_mults);
        cudaFree(block_p_jac_idx);
        cudaFree(inverse_fail_counter);
        if (!p_solver_setup_gpu)
        {
          cudaFree(p_vals);
        }
      }
      if (p_system_preconditioner)
        delete (p_system_preconditioner);
    }

    template <uint8_t n_block_size>
    void linsolv_bos_cpr_gpu<n_block_size>::setup_kernel()
    {
    }

    template <uint8_t n_block_size>
    int linsolv_bos_cpr_gpu<n_block_size>::init(opendarts::linear_solvers::csr_matrix<n_block_size> *A,
      int /*max_iters*/,
      double /*tolerance*/)
    {
      index_t n_rows = A->n_rows;

      // The pressure matrix repeats the reservoir matrix structure, block size 1.
      P->init(n_rows, n_rows, 1, A->get_n_non_zeros());
      P->type = opendarts::linear_solvers::MATRIX_TYPE_CSR;
      P->is_square = 1;

      std::memcpy(P->get_rows_ptr(), A->get_rows_ptr(), (n_rows + 1) * sizeof(index_t));
      std::memcpy(P->get_cols_ind(), A->get_cols_ind(), P->get_n_non_zeros() * sizeof(index_t));

      // The diagonal-first reordering (csr_matrix::set_diag_first) is not part
      // of the open-source csr_matrix; force the flag off and use the
      // non-diagonal-first kernel path.
      if (p_solver_requires_diag_first)
      {
        printf("CPR GPU: diagonal-first pressure solvers are not supported in the "
               "open-source build; falling back to the non-diagonal-first path\n");
        p_solver_requires_diag_first = 0;
      }
      std::memcpy(P->get_diag_ind(), A->get_diag_ind(), n_rows * sizeof(index_t));

      if (p_solver_setup_gpu)
      {
        P->init_device(n_rows, A->get_n_non_zeros());
        P->copy_struct_to_device();
      }

      if (D_ps_ss)
      {
        cudaFree(D_ps_ss);
        cudaFree(P_B);
        cudaFree(P_X);
        cudaFree(full_B);
        cudaFree(rhs_mults);
        cudaFree(block_p_jac_idx);
      }

      cudaError_t cudaStat;

      cudaStat = cudaMalloc((void **)&D_ps_ss, sizeof(value_t) * A->n_rows * (n_block_size - 1));
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory in CPR\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&P_B, sizeof(value_t) * A->n_rows);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory in CPR\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&P_X, sizeof(value_t) * A->n_rows);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory in CPR\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&full_B, sizeof(value_t) * A->n_rows * n_block_size);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory in CPR\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&rhs_mults, sizeof(value_t) * A->n_rows);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory in CPR\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&block_p_jac_idx, sizeof(index_t) * (A->get_n_non_zeros() - n_rows));
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory in CPR\n");
        return -1;
      }
      cudaStat = cudaMalloc((void **)&inverse_fail_counter, sizeof(index_t));
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't allocate device memory in CPR\n");
        return -1;
      }

      // initialise the inverse-fail counter with 0
      int inverse_fail_counter_host = 0;
      cudaMemcpy(inverse_fail_counter, &inverse_fail_counter_host, sizeof(index_t), cudaMemcpyHostToDevice);

      diag_acc = 0;

      if (p_solver_setup_gpu)
      {
        p_vals = P->values_d;
      }
      else
      {
        cudaStat = cudaMalloc((void **)&p_vals, sizeof(value_t) * P->get_n_non_zeros());
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory in CPR\n");
          return -1;
        }
      }

      if (!p_solver_solve_gpu)
      {
        P_B_h = new value_t[A->n_rows];
        P_X_h = new value_t[A->n_rows];
      }

      // index vector for the single-loop colsum
      index_t *block_p_jac_idx_h = new index_t[A->get_n_non_zeros() - n_rows];
      index_t *rows = A->get_rows_ptr();
      index_t *cols = A->get_cols_ind();
      index_t conn = 0;

      for (index_t i = 0; i < n_rows; ++i)
      {
        index_t j1 = rows[i];
        index_t j2 = rows[i + 1];

        for (index_t j = j1; j < j2; ++j)
        {
          // for all off-diagonal values
          if (cols[j] != i)
          {
            index_t cl_j1 = rows[cols[j]];
            index_t cl_j2 = rows[cols[j] + 1];

            for (index_t cl_j = cl_j1; cl_j < cl_j2; ++cl_j)
            {
              if (cols[cl_j] == i)
                block_p_jac_idx_h[conn++] = cl_j * n_block_size * n_block_size;
            }
          }
        }
      }

      cudaStat = cudaMemcpy(block_p_jac_idx, block_p_jac_idx_h,
        sizeof(index_t) * (A->get_n_non_zeros() - n_rows), cudaMemcpyHostToDevice);
      if (cudaStat != cudaSuccess)
      {
        printf("Error! Can't copy memory to device\n");
        return -1;
      }

      delete[] block_p_jac_idx_h;

      // run a single cycle only -- tolerance is irrelevant for a preconditioner
      p_system_preconditioner->init(P, 1, 0);
      full_system_preconditioner->init(A, 1, 0);

      p_system_preconditioner->init_timer_nodes(&this->timer_setup->node["CPR"],
        &this->timer_solve->node["CPR"]);
      full_system_preconditioner->init_timer_nodes(&this->timer_setup->node["CPR"],
        &this->timer_solve->node["CPR"]);

      setup_block_size = 1024;
      solve_reduce_block_size = 1024;
      solve_prolongate_block_size = 1024;
      solve_sum_up_block_size = 1024;
      cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &setup_block_size,
        cpr_setup_kernel<n_block_size>, 0, A->n_rows);
      printf("cpr_setup_kernel %d %d\n", min_grid_size, setup_block_size);
      cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &solve_reduce_block_size,
        cpr_solve_reduce_kernel<n_block_size>, 0, A->n_rows);
      printf("cpr_solve_reduce_kernel %d %d\n", min_grid_size, solve_reduce_block_size);
      cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &solve_prolongate_block_size,
        cpr_solve_prolongate_kernel<n_block_size>, 0, A->n_rows);
      printf("solve_prolongate_kernel %d %d\n", min_grid_size, solve_prolongate_block_size);
      cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &solve_sum_up_block_size,
        cpr_solve_sum_up_kernel<n_block_size>, 0, A->n_rows);
      printf("solve_sum_up_kernel %d %d\n", min_grid_size, solve_sum_up_block_size);

      return 0;
    }

    template <uint8_t n_block_size>
    int linsolv_bos_cpr_gpu<n_block_size>::setup(opendarts::linear_solvers::csr_matrix_base *A_)
    {
      cudaError_t cudaStat;

      this->timer_setup->node["CPR"].start();
      A_base = A_;

      opendarts::linear_solvers::csr_matrix<n_block_size> *A =
        static_cast<opendarts::linear_solvers::csr_matrix<n_block_size> *>(A_);

      index_t n_rows = A->n_rows;
      index_t *rows = A->rows_ptr_d;
      index_t *diags = A->diag_ind_d;
      value_t *vals = A->values_d;

#ifdef ACCURATE_INVERSION
      cudaMemset(D_ps_ss, 0, n_rows * (n_block_size - 1) * sizeof(value_t));
#endif

      int grid_size = (n_rows + setup_block_size - 1) / setup_block_size;

      if (p_solver_requires_diag_first)
      {
        cpr_setup_kernel<n_block_size><<<grid_size, setup_block_size>>>(n_rows, rows, diags,
          block_p_jac_idx, vals, p_vals, D_ps_ss, rhs_mults, inverse_fail_counter);
      }
      else
      {
        cpr_setup_kernel_no_first_diag<n_block_size><<<grid_size, setup_block_size>>>(n_rows, rows,
          diags, block_p_jac_idx, vals, p_vals, D_ps_ss, rhs_mults, inverse_fail_counter, diag_acc);
      }
      int inverse_fail_counter_host;
      cudaMemcpy(&inverse_fail_counter_host, inverse_fail_counter, sizeof(index_t), cudaMemcpyDeviceToHost);

      if (inverse_fail_counter_host)
      {
        printf("CPR inverse failed for %d blocks\n", inverse_fail_counter_host);
        // re-initialise the inverse-fail counter with 0
        inverse_fail_counter_host = 0;
        cudaMemcpy(inverse_fail_counter, &inverse_fail_counter_host, sizeof(index_t), cudaMemcpyHostToDevice);
        this->timer_setup->node["CPR"].stop();
        return -1;
      }

      if (!p_solver_setup_gpu)
      {
        value_t *p_vals_h = P->get_values();
        this->timer_setup->node["CPR"].node["P_comm"].start();
        cudaStat = cudaMemcpy(p_vals_h, p_vals, sizeof(value_t) * A->get_n_non_zeros(), cudaMemcpyDeviceToHost);
        this->timer_setup->node["CPR"].node["P_comm"].stop();
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't copy memory to host\n");
          return -1;
        }
      }

      // set up the second stage first to benefit from asynchronous execution
      full_system_preconditioner->setup(A);

      // Synchronise before the first-stage setup. Removing this allows the two
      // setups to overlap when they run on different devices (CPU and GPU).
      cudaDeviceSynchronize();

      p_system_preconditioner->setup(P);

      lin_it = 0;
      this->timer_setup->node["CPR"].stop();
      return 0;
    }

    template <uint8_t n_block_size>
    int linsolv_bos_cpr_gpu<n_block_size>::solve(value_t *B, value_t *X)
    {
      this->timer_solve->node["CPR"].start();

      cudaError_t cudaStat;
      index_t n_rows = A_base->n_rows;

      // Reduce B to the pressure RHS P_B.
      int grid_size = (n_rows + solve_reduce_block_size - 1) / solve_reduce_block_size;
      cpr_solve_reduce_kernel<n_block_size><<<grid_size, solve_reduce_block_size>>>(n_rows, B, P_B,
        D_ps_ss, rhs_mults);

      // Solve the pressure system.
      if (!p_solver_solve_gpu)
      {
        this->timer_solve->node["CPR"].node["P_comm"].start();
        cudaStat = cudaMemcpy(P_B_h, P_B, sizeof(value_t) * n_rows, cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't copy memory to host\n");
          return -1;
        }
        this->timer_solve->node["CPR"].node["P_comm"].stop();
        std::memset(P_X_h, 0, n_rows * sizeof(value_t));
        p_system_preconditioner->solve(P_B_h, P_X_h);

        this->timer_solve->node["CPR"].node["P_comm"].start();
        cudaStat = cudaMemcpy(P_X, P_X_h, sizeof(value_t) * n_rows, cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't copy memory to device\n");
          return -1;
        }
        this->timer_solve->node["CPR"].node["P_comm"].stop();
      }
      else
      {
        cudaMemset(P_X, 0, sizeof(value_t) * n_rows);
        p_system_preconditioner->solve(P_B, P_X);
        cudaDeviceSynchronize();
      }

      // Prolongate P_X into X.
      cudaMemset(X, 0, sizeof(value_t) * n_rows * n_block_size);

      grid_size = (n_rows + solve_prolongate_block_size - 1) / solve_prolongate_block_size;
      cpr_solve_prolongate_kernel<n_block_size><<<grid_size, solve_prolongate_block_size>>>(n_rows, X, P_X);

      // Correct the RHS for the full system: full_B = B - A * X.
      A_base->calc_lin_comb_d(-1.0, 1.0, X, B, full_B);
      cudaDeviceSynchronize();

      // Solve the full system.
      full_system_preconditioner->solve(full_B, X);

      // Add up the two-stage solutions.
      grid_size = (n_rows + solve_sum_up_block_size - 1) / solve_sum_up_block_size;
      cpr_solve_sum_up_kernel<n_block_size><<<grid_size, solve_sum_up_block_size>>>(n_rows, X, P_X);

      this->timer_solve->node["CPR"].stop();
      lin_it++;

      return 0;
    }

    // Explicit instantiation; CPR requires block size >= 2 (pressure +
    // saturation split).
    template class linsolv_bos_cpr_gpu<2>;
    template class linsolv_bos_cpr_gpu<3>;
    template class linsolv_bos_cpr_gpu<4>;
    template class linsolv_bos_cpr_gpu<5>;
    template class linsolv_bos_cpr_gpu<6>;
    template class linsolv_bos_cpr_gpu<7>;
    template class linsolv_bos_cpr_gpu<8>;
    template class linsolv_bos_cpr_gpu<9>;
    template class linsolv_bos_cpr_gpu<10>;
    template class linsolv_bos_cpr_gpu<11>;
    template class linsolv_bos_cpr_gpu<12>;
    template class linsolv_bos_cpr_gpu<13>;
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU
