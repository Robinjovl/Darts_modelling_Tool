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

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "linsolv_gmres_gpu.hpp"
#include "csr_matrix_base.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    linsolv_gmres_gpu<N_BLOCK_SIZE>::linsolv_gmres_gpu()
    {
      prec = nullptr;
      A = nullptr;
      n_iters = 0;
      max_iters = 0;
      final_resid = 0.0;
      tolerance = 0.0;
      n = 0;
      restart_requested = 50;
      m = 0;
      V_d = nullptr;
      w_d = z_d = h_d = nullptr;

      cublasStatus_t stat = cublasCreate(&cub_handle);
      if (stat != CUBLAS_STATUS_SUCCESS)
        printf("CUBLAS initialization failed\n");
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_gmres_gpu<N_BLOCK_SIZE>::~linsolv_gmres_gpu()
    {
      if (prec)
        delete prec;
      if (V_d)
        cudaFree(V_d);
      if (bx_t_d)
        cudaFree(bx_t_d);
      cublasDestroy(cub_handle);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres_gpu<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_input)
    {
      prec = prec_input;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres_gpu<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
      opendarts::config::index_t max_iters_input,
      opendarts::config::mat_float tolerance_input)
    {
      max_iters = max_iters_input;
      tolerance = tolerance_input;

      if (prec && this->timer_setup && this->timer_solve)
        prec->init_timer_nodes(&this->timer_setup->node["GMRES"],
          &this->timer_solve->node["GMRES"]);

      if (!A_input->is_square)
      {
        printf("Matrix is not square!\n");
        return -1;
      }

      // Proprietary parity: restart length capped by the iteration budget.
      m = restart_requested;
      if (m > max_iters)
        m = max_iters;
      if (m < 1)
        m = 1;

      const int n_new = A_input->n_rows * N_BLOCK_SIZE;
      if (n_new != n || !V_d)
      {
        if (V_d)
          cudaFree(V_d);
        n = n_new;
        // One contiguous block: V ((m+1)*n) | w (n) | z (n) | h (m+1).
        const std::size_t total =
            static_cast<std::size_t>(m + 1) * n + 2 * static_cast<std::size_t>(n)
            + (m + 1);
        cudaError_t cudaStat = cudaMalloc((void **)&V_d, sizeof(double) * total);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory: %s\n", cudaGetErrorString(cudaStat));
          V_d = nullptr;
          return -2;
        }
        w_d = V_d + static_cast<std::size_t>(m + 1) * n;
        z_d = w_d + n;
        h_d = z_d + n;
      }

      hh.assign(static_cast<std::size_t>(m + 1) * m, 0.0);
      cs.assign(m, 0.0);
      sn.assign(m, 0.0);
      rs.assign(m + 1, 0.0);
      y.assign(m, 0.0);

      if (prec)
        return prec->init(A_input, max_iters, tolerance);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres_gpu<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A_input)
    {
      A = A_input;
      if (!prec)
        return 0;
      if (this->timer_setup)
        this->timer_setup->node["GMRES"].start();
      int res = prec->setup(A);
      if (this->timer_setup)
        this->timer_setup->node["GMRES"].stop();
      return res;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres_gpu<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      const double one = 1.0, mone = -1.0, zero = 0.0;
      const double epsmac = 1.0e-16;

      if (!A || !V_d)
        return -1;

      ::timer_node *tsolve = this->timer_solve ? &this->timer_solve->node["GMRES"] : nullptr;
      if (tsolve)
        tsolve->start();

      double b_norm = 0.0;
      cublasDnrm2(cub_handle, n, B, 1, &b_norm);

      int total_iters = 0;
      double res_norm = 0.0;
      int rc = 0;

      for (;;)
      {
        // r = B - A X into V column 0 (X holds the current iterate; the
        // engine hands in a zero initial guess).
        {
          if (tsolve)
            tsolve->node["SPMV_bsr"].start();
          A->calc_lin_comb_d(mone, one, X, B, V_d);
          if (tsolve)
            tsolve->node["SPMV_bsr"].stop();
        }
        double beta = 0.0;
        cublasDnrm2(cub_handle, n, V_d, 1, &beta);

        if (total_iters == 0)
        {
          // Proprietary parity: converge relative to ||b|| (falling back to
          // ||r0|| for a vanishing RHS).
          const double den = (b_norm > 1e-16) ? b_norm : (beta > 1e-16 ? beta : 1.0);
          res_norm = beta;
          final_resid = beta / den;
          if (beta <= tolerance * den)
            break;
        }

        if (beta <= epsmac)
          break;

        const double inv_beta = 1.0 / beta;
        cublasDscal(cub_handle, n, &inv_beta, V_d, 1);
        rs[0] = beta;

        const double den = (b_norm > 1e-16) ? b_norm : beta;
        const double tol_abs = tolerance * den;

        int i = 0; // Krylov vectors built in this cycle
        for (; i < m && total_iters < max_iters;)
        {
          double *v_i = V_d + static_cast<std::size_t>(i) * n;
          double *v_next = V_d + static_cast<std::size_t>(i + 1) * n;

          // z = M^{-1} v_i ; w = A z.
          if (prec)
          {
            if (prec->solve(v_i, z_d))
            {
              rc = -3;
              break;
            }
          }
          else
          {
            cublasDcopy(cub_handle, n, v_i, 1, z_d, 1);
          }
          {
            if (tsolve)
              tsolve->node["SPMV_bsr"].start();
            A->matrix_vector_product_d0(z_d, w_d);
            if (tsolve)
              tsolve->node["SPMV_bsr"].stop();
          }

          // Classical Gram-Schmidt with one re-orthogonalisation pass
          // (CGS2), as two GEMVs per pass against the first i+1 basis
          // columns.
          double *h_col = &hh[static_cast<std::size_t>(i) * (m + 1)];
          const int ncols = i + 1;
          // h = V^T w ; w -= V h
          cublasDgemv(cub_handle, CUBLAS_OP_T, n, ncols, &one, V_d, n, w_d, 1,
            &zero, h_d, 1);
          cublasDgemv(cub_handle, CUBLAS_OP_N, n, ncols, &mone, V_d, n, h_d, 1,
            &one, w_d, 1);
          cublasGetVector(ncols, sizeof(double), h_d, 1, h_col, 1);
          // Re-orthogonalisation pass: h += correction.
          cublasDgemv(cub_handle, CUBLAS_OP_T, n, ncols, &one, V_d, n, w_d, 1,
            &zero, h_d, 1);
          cublasDgemv(cub_handle, CUBLAS_OP_N, n, ncols, &mone, V_d, n, h_d, 1,
            &one, w_d, 1);
          {
            std::vector<double> corr(ncols);
            cublasGetVector(ncols, sizeof(double), h_d, 1, corr.data(), 1);
            for (int k = 0; k < ncols; ++k)
              h_col[k] += corr[k];
          }

          double h_next = 0.0;
          cublasDnrm2(cub_handle, n, w_d, 1, &h_next);
          h_col[i + 1] = h_next;
          if (h_next > epsmac)
          {
            const double inv_h = 1.0 / h_next;
            cublasDscal(cub_handle, n, &inv_h, w_d, 1);
            cublasDcopy(cub_handle, n, w_d, 1, v_next, 1);
          }

          // Apply previous Givens rotations to the new Hessenberg column.
          for (int k = 1; k <= i; ++k)
          {
            const double t = h_col[k - 1];
            h_col[k - 1] = cs[k - 1] * t + sn[k - 1] * h_col[k];
            h_col[k] = -sn[k - 1] * t + cs[k - 1] * h_col[k];
          }
          double gamma = std::sqrt(h_col[i] * h_col[i] + h_col[i + 1] * h_col[i + 1]);
          if (gamma < epsmac)
            gamma = epsmac;
          cs[i] = h_col[i] / gamma;
          sn[i] = h_col[i + 1] / gamma;
          rs[i + 1] = -sn[i] * rs[i];
          rs[i] = cs[i] * rs[i];
          h_col[i] = gamma;

          ++i;
          ++total_iters;
          res_norm = std::fabs(rs[i]);
          if (res_norm <= tol_abs || h_next <= epsmac)
            break;
        }

        if (rc)
          break;

        // Back-substitution: y = H^{-1} rs (upper triangular, i x i).
        for (int r = i - 1; r >= 0; --r)
        {
          double acc = rs[r];
          for (int k = r + 1; k < i; ++k)
            acc -= hh[static_cast<std::size_t>(k) * (m + 1) + r] * y[k];
          y[r] = acc / hh[static_cast<std::size_t>(r) * (m + 1) + r];
        }

        if (i > 0)
        {
          // u = V(:,0:i) y  (into w_d), then X += M^{-1} u -- the single
          // preconditioner application on the combined update (standard
          // right preconditioning; proprietary parity).
          cublasSetVector(i, sizeof(double), y.data(), 1, h_d, 1);
          cublasDgemv(cub_handle, CUBLAS_OP_N, n, i, &one, V_d, n, h_d, 1,
            &zero, w_d, 1);
          if (prec)
          {
            if (prec->solve(w_d, z_d))
            {
              rc = -3;
              break;
            }
            cublasDaxpy(cub_handle, n, &one, z_d, 1, X, 1);
          }
          else
          {
            cublasDaxpy(cub_handle, n, &one, w_d, 1, X, 1);
          }
        }

        const double den2 = (b_norm > 1e-16) ? b_norm : 1.0;
        final_resid = res_norm / den2;
        if (res_norm <= tol_abs || total_iters >= max_iters || i == 0)
          break;
      }

      n_iters = total_iters;
      {
        const double den2 = (b_norm > 1e-16) ? b_norm : 1.0;
        final_resid = res_norm / den2;
      }

      if (tsolve)
        tsolve->stop();
      return rc;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres_gpu<N_BLOCK_SIZE>::solve_transposed(opendarts::config::mat_float *B,
      opendarts::config::mat_float *X)
    {
      // Adjoint solve of A^T x = B. Same restarted right-preconditioned GMRES
      // as solve(), with the transposed SpMV and prec->solve_transposed().
      // B and X are HOST pointers (the adjoint backward driver is host code);
      // they are staged through a lazily allocated device scratch.
      const double one = 1.0, mone = -1.0, zero = 0.0;
      const double epsmac = 1.0e-16;

      if (!A || !V_d)
        return -1;

      // The transposed products run over a cached transpose/scalar view of A;
      // re-anchor it to the current matrix values once per outer solve.
      if (A->refresh_transpose_spmv_d())
      {
        printf("linsolv_gmres_gpu::solve_transposed: the system matrix does "
               "not support transposed device SpMV\n");
        return -1;
      }

      if (bx_t_n != n || !bx_t_d)
      {
        if (bx_t_d)
          cudaFree(bx_t_d);
        bx_t_n = 0;
        if (cudaMalloc((void **)&bx_t_d, sizeof(double) * 2 * n) != cudaSuccess)
        {
          printf("linsolv_gmres_gpu::solve_transposed: can't allocate the "
                 "host-staging device buffer\n");
          bx_t_d = nullptr;
          return -2;
        }
        bx_t_n = n;
      }
      double *B_d = bx_t_d;
      double *X_d = bx_t_d + n;
      cudaMemcpy(B_d, B, sizeof(double) * n, cudaMemcpyHostToDevice);
      cudaMemcpy(X_d, X, sizeof(double) * n, cudaMemcpyHostToDevice); // initial guess

      ::timer_node *tsolve = this->timer_solve ? &this->timer_solve->node["GMRES"] : nullptr;
      if (tsolve)
        tsolve->start();

      double b_norm = 0.0;
      cublasDnrm2(cub_handle, n, B_d, 1, &b_norm);

      int total_iters = 0;
      double res_norm = 0.0;
      int rc = 0;

      for (;;)
      {
        // r = B - A^T X into V column 0.
        {
          if (tsolve)
            tsolve->node["SPMV_bsr"].start();
          if (A->calc_lin_comb_t_d(mone, one, X_d, B_d, V_d))
          {
            rc = -4;
            if (tsolve)
              tsolve->node["SPMV_bsr"].stop();
            break;
          }
          if (tsolve)
            tsolve->node["SPMV_bsr"].stop();
        }
        double beta = 0.0;
        cublasDnrm2(cub_handle, n, V_d, 1, &beta);

        if (total_iters == 0)
        {
          const double den = (b_norm > 1e-16) ? b_norm : (beta > 1e-16 ? beta : 1.0);
          res_norm = beta;
          final_resid = beta / den;
          if (beta <= tolerance * den)
            break;
        }

        if (beta <= epsmac)
          break;

        const double inv_beta = 1.0 / beta;
        cublasDscal(cub_handle, n, &inv_beta, V_d, 1);
        rs[0] = beta;

        const double den = (b_norm > 1e-16) ? b_norm : beta;
        const double tol_abs = tolerance * den;

        int i = 0;
        for (; i < m && total_iters < max_iters;)
        {
          double *v_i = V_d + static_cast<std::size_t>(i) * n;
          double *v_next = V_d + static_cast<std::size_t>(i + 1) * n;

          // z = M^{-T} v_i ; w = A^T z.
          if (prec)
          {
            if (prec->solve_transposed(v_i, z_d))
            {
              rc = -3;
              break;
            }
          }
          else
          {
            cublasDcopy(cub_handle, n, v_i, 1, z_d, 1);
          }
          {
            if (tsolve)
              tsolve->node["SPMV_bsr"].start();
            if (A->matrix_vector_product_t_d0(z_d, w_d))
            {
              rc = -4;
              if (tsolve)
                tsolve->node["SPMV_bsr"].stop();
              break;
            }
            if (tsolve)
              tsolve->node["SPMV_bsr"].stop();
          }

          // CGS2 orthogonalisation, as in solve().
          double *h_col = &hh[static_cast<std::size_t>(i) * (m + 1)];
          const int ncols = i + 1;
          cublasDgemv(cub_handle, CUBLAS_OP_T, n, ncols, &one, V_d, n, w_d, 1,
            &zero, h_d, 1);
          cublasDgemv(cub_handle, CUBLAS_OP_N, n, ncols, &mone, V_d, n, h_d, 1,
            &one, w_d, 1);
          cublasGetVector(ncols, sizeof(double), h_d, 1, h_col, 1);
          cublasDgemv(cub_handle, CUBLAS_OP_T, n, ncols, &one, V_d, n, w_d, 1,
            &zero, h_d, 1);
          cublasDgemv(cub_handle, CUBLAS_OP_N, n, ncols, &mone, V_d, n, h_d, 1,
            &one, w_d, 1);
          {
            std::vector<double> corr(ncols);
            cublasGetVector(ncols, sizeof(double), h_d, 1, corr.data(), 1);
            for (int k = 0; k < ncols; ++k)
              h_col[k] += corr[k];
          }

          double h_next = 0.0;
          cublasDnrm2(cub_handle, n, w_d, 1, &h_next);
          h_col[i + 1] = h_next;
          if (h_next > epsmac)
          {
            const double inv_h = 1.0 / h_next;
            cublasDscal(cub_handle, n, &inv_h, w_d, 1);
            cublasDcopy(cub_handle, n, w_d, 1, v_next, 1);
          }

          for (int k = 1; k <= i; ++k)
          {
            const double t = h_col[k - 1];
            h_col[k - 1] = cs[k - 1] * t + sn[k - 1] * h_col[k];
            h_col[k] = -sn[k - 1] * t + cs[k - 1] * h_col[k];
          }
          double gamma = std::sqrt(h_col[i] * h_col[i] + h_col[i + 1] * h_col[i + 1]);
          if (gamma < epsmac)
            gamma = epsmac;
          cs[i] = h_col[i] / gamma;
          sn[i] = h_col[i + 1] / gamma;
          rs[i + 1] = -sn[i] * rs[i];
          rs[i] = cs[i] * rs[i];
          h_col[i] = gamma;

          ++i;
          ++total_iters;
          res_norm = std::fabs(rs[i]);
          if (res_norm <= tol_abs || h_next <= epsmac)
            break;
        }

        if (rc)
          break;

        for (int r = i - 1; r >= 0; --r)
        {
          double acc = rs[r];
          for (int k = r + 1; k < i; ++k)
            acc -= hh[static_cast<std::size_t>(k) * (m + 1) + r] * y[k];
          y[r] = acc / hh[static_cast<std::size_t>(r) * (m + 1) + r];
        }

        if (i > 0)
        {
          cublasSetVector(i, sizeof(double), y.data(), 1, h_d, 1);
          cublasDgemv(cub_handle, CUBLAS_OP_N, n, i, &one, V_d, n, h_d, 1,
            &zero, w_d, 1);
          if (prec)
          {
            if (prec->solve_transposed(w_d, z_d))
            {
              rc = -3;
              break;
            }
            cublasDaxpy(cub_handle, n, &one, z_d, 1, X_d, 1);
          }
          else
          {
            cublasDaxpy(cub_handle, n, &one, w_d, 1, X_d, 1);
          }
        }

        const double den2 = (b_norm > 1e-16) ? b_norm : 1.0;
        final_resid = res_norm / den2;
        if (res_norm <= tol_abs || total_iters >= max_iters || i == 0)
          break;
      }

      n_iters = total_iters;
      {
        const double den2 = (b_norm > 1e-16) ? b_norm : 1.0;
        final_resid = res_norm / den2;
      }

      cudaMemcpy(X, X_d, sizeof(double) * n, cudaMemcpyDeviceToHost);

      if (tsolve)
        tsolve->stop();
      return rc;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_gmres_gpu<N_BLOCK_SIZE>::get_n_iters()
    {
      return n_iters;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_gmres_gpu<N_BLOCK_SIZE>::get_residual()
    {
      return final_resid;
    }

    // Explicit instantiation for the supported block sizes.
    template class linsolv_gmres_gpu<1>;
    template class linsolv_gmres_gpu<2>;
    template class linsolv_gmres_gpu<3>;
    template class linsolv_gmres_gpu<4>;
    template class linsolv_gmres_gpu<5>;
    template class linsolv_gmres_gpu<6>;
    template class linsolv_gmres_gpu<7>;
    template class linsolv_gmres_gpu<8>;
    template class linsolv_gmres_gpu<9>;
    template class linsolv_gmres_gpu<10>;
    template class linsolv_gmres_gpu<11>;
    template class linsolv_gmres_gpu<12>;
    template class linsolv_gmres_gpu<13>;
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU
