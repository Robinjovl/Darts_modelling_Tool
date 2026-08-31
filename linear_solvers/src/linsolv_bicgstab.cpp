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
#include <limits>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "linsolv_bicgstab.hpp"
#include "csr_matrix_base.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    linsolv_bicgstab<N_BLOCK_SIZE>::linsolv_bicgstab()
    {
      prec = nullptr;
      A = nullptr;
      n_iters = 0;
      max_iters = 0;
      final_resid = 0.0;
      tolerance = 0.0;
      n = 0;
      r = rw = p = pw = s = t = v = nullptr;

      cublasStatus_t stat = cublasCreate(&cub_handle);
      if (stat != CUBLAS_STATUS_SUCCESS)
        printf("CUBLAS initialization failed\n");
      wksp_d = nullptr;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_bicgstab<N_BLOCK_SIZE>::~linsolv_bicgstab()
    {
      if (prec)
        delete prec;

      if (wksp_d)
        cudaFree(wksp_d);

      cublasDestroy(cub_handle);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_bicgstab<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_input)
    {
      prec = prec_input;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_bicgstab<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
      opendarts::config::index_t max_iters_input,
      opendarts::config::mat_float tolerance_input)
    {
      max_iters = max_iters_input;
      tolerance = tolerance_input;

      prec->init_timer_nodes(&this->timer_setup->node["BiCGStab"], &this->timer_solve->node["BiCGStab"]);

      if (!A_input->is_square)
      {
        printf("Matrix is not square!\n");
        return -1;
      }
      const int n_new = A_input->n_rows * N_BLOCK_SIZE;

      // The r/rw/p/pw/s/t/v slices below are spaced by n, so a re-init against a
      // differently sized system must re-allocate and re-slice: keeping the old
      // buffer while solve() operates on the new n makes the slices alias and
      // writes past the end of the allocation (cf. linsolv_gmres_gpu::init).
      if (n_new != n || !wksp_d)
      {
        if (wksp_d)
          cudaFree(wksp_d);
        wksp_d = nullptr;
        r = rw = p = pw = s = t = v = nullptr;
        n = n_new;

        cudaError_t cudaStat = cudaMalloc((void **)&wksp_d, sizeof(double) * n * 7);
        if (cudaStat != cudaSuccess)
        {
          printf("Error! Can't allocate device memory: %s\n", cudaGetErrorString(cudaStat));
          wksp_d = nullptr;
          return -2;
        }
        r = wksp_d;
        rw = r + n;
        p = rw + n;
        pw = p + n;
        s = pw + n;
        t = s + n;
        v = t + n;
      }

      return prec->init(A_input, max_iters, tolerance);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_bicgstab<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A_input)
    {
      A = A_input;
      this->timer_setup->node["BiCGStab"].start();
      int res = prec->setup(A);
      this->timer_setup->node["BiCGStab"].stop();
      return res;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_bicgstab<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
      double nrmr = 0.0, nrmr0, b_norm;
      rho = 0.0;
      alpha = 0.0;
      omega = 0.0;
      double one = 1.0;
      double mone = -1.0;
      int i = 0;

      this->timer_solve->node["BiCGStab"].start();

      // Initial residual r0 = B - A*X (initial guess in X).
      this->timer_solve->node["BiCGStab"].node["SPMV_bsr"].start();
      A->calc_lin_comb_d(mone, one, X, B, r);
      this->timer_solve->node["BiCGStab"].node["SPMV_bsr"].stop();

      // Copy residual r into r^hat and p.
      cublasDcopy(cub_handle, n, r, 1, rw, 1);
      cublasDcopy(cub_handle, n, r, 1, p, 1);
      cublasDnrm2(cub_handle, n, r, 1, &nrmr0);
      cublasDnrm2(cub_handle, n, B, 1, &b_norm);

      // Convergence criterion |r_i|/|b| <= tol when |b| > 0.
      if (b_norm > 1e-16)
        nrmr0 = b_norm;

      for (i = 0; i < max_iters;)
      {
        rhop = rho;
        cublasDdot(cub_handle, n, rw, 1, r, 1, &rho);

        if (i > 0)
        {
          beta = (rho / rhop) * (alpha / omega);
          negomega = -omega;
          cublasDaxpy(cub_handle, n, &negomega, v, 1, p, 1);
          cublasDscal(cub_handle, n, &beta, p, 1);
          cublasDaxpy(cub_handle, n, &one, r, 1, p, 1);
        }

        // Preconditioning step.
        if (prec->solve(p, pw))
        {
          this->timer_solve->node["BiCGStab"].stop();
          return -3;
        }

        // Matrix-vector multiplication.
        this->timer_solve->node["BiCGStab"].node["SPMV_bsr"].start();
        A->matrix_vector_product_d0(pw, v);
        this->timer_solve->node["BiCGStab"].node["SPMV_bsr"].stop();

        cublasDdot(cub_handle, n, rw, 1, v, 1, &temp);
        alpha = rho / temp;
        // Krylov breakdown (rho or (rw,v) vanished) or NaN propagation from
        // the preconditioner/matrix: the update would be garbage. Fail hard
        // so the engine cuts the timestep instead of applying it.
        if (!std::isfinite(alpha))
        {
          n_iters = i;
          final_resid = std::numeric_limits<double>::infinity();
          last_converged = false;
          this->timer_solve->node["BiCGStab"].stop();
          return -4;
        }
        negalpha = -(alpha);
        cublasDaxpy(cub_handle, n, &negalpha, v, 1, r, 1);
        cublasDaxpy(cub_handle, n, &alpha, pw, 1, X, 1);
        cublasDnrm2(cub_handle, n, r, 1, &nrmr);

        if (nrmr < tolerance * nrmr0)
          break;

        // Preconditioning step.
        if (prec->solve(r, s))
        {
          this->timer_solve->node["BiCGStab"].stop();
          return -3;
        }

        // Matrix-vector multiplication.
        this->timer_solve->node["BiCGStab"].node["SPMV_bsr"].start();
        A->matrix_vector_product_d0(s, t);
        this->timer_solve->node["BiCGStab"].node["SPMV_bsr"].stop();

        cublasDdot(cub_handle, n, t, 1, r, 1, &temp);
        cublasDdot(cub_handle, n, t, 1, t, 1, &temp2);
        omega = temp / temp2;
        if (!std::isfinite(omega))
        {
          n_iters = i;
          final_resid = std::numeric_limits<double>::infinity();
          last_converged = false;
          this->timer_solve->node["BiCGStab"].stop();
          return -4;
        }
        negomega = -(omega);
        cublasDaxpy(cub_handle, n, &omega, s, 1, X, 1);
        cublasDaxpy(cub_handle, n, &negomega, t, 1, r, 1);
        cublasDnrm2(cub_handle, n, r, 1, &nrmr);

        if (nrmr < tolerance * nrmr0)
        {
          i++;
          break;
        }
        i++;
      }
      n_iters = i;
      final_resid = nrmr / nrmr0;
      last_converged = (nrmr < tolerance * nrmr0);

      this->timer_solve->node["BiCGStab"].stop();
      // A non-finite residual means the iterate is garbage -- hard failure.
      // Plain non-convergence at max_iters keeps the legacy 0 return; it is
      // visible through stats().converged.
      if (!std::isfinite(nrmr))
        return -4;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_bicgstab<N_BLOCK_SIZE>::get_n_iters()
    {
      return n_iters;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_bicgstab<N_BLOCK_SIZE>::get_residual()
    {
      return final_resid;
    }

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see solvers/src/CMakeLists.txt.
    // Edit the block-size range there, not here.
#include "linsolv_bicgstab_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU
