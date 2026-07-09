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

// AMGX support is opt-in (CMake option WITH_AMGX).
#if defined(WITH_GPU) && defined(WITH_AMGX)

#include <cstdio>
#include <fstream>
#include <string>

#include <cuda_runtime.h>
#include "amgx_c.h"

#include "linsolv_amgx.hpp"
#include "block_csr_matrix.hpp"
#include "csr_matrix.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    // AMGX resources are shared across all solver instances: a single
    // resource handle, created with the first solver and destroyed with the
    // last (see NVIDIA/AMGX issue #109).
    static int amgx_instances = 0;
    static void *rsrc = nullptr;

    template <uint8_t N_BLOCK_SIZE>
    linsolv_amgx<N_BLOCK_SIZE>::linsolv_amgx(int device_num_input, int convert_to_bs1_input)
      : device_num(device_num_input), convert_to_bs1(convert_to_bs1_input)
    {
      // Register as the linear_solver_base behind the BOS interface.
      opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>::solver = this;
      n_rows = 0;
      A = x = b = nullptr;

      if (amgx_instances == 0)
      {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_initialize_plugins());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
      }

      std::string config_filename = "amgx_config_bs" + std::to_string((int)N_BLOCK_SIZE) + ".json";

      std::ifstream f(config_filename.c_str());
      if (f.good())
      {
        AMGX_config_create_from_file((AMGX_config_handle_struct **)&config, config_filename.c_str());
        printf("AMGX: configuration picked up from %s\n", config_filename.c_str());
      }
      else
      {
        const char *default_config =
          "{\
          \"config_version\": 2,\
          \"solver\": {\
              \"solver\": \"AMG\",\
              \"convergence\": \"RELATIVE_MAX_CORE\",\
              \"norm\": \"L2\",\
              \"use_scalar_norm\": 1,\
              \"tolerance\": 1e-3,\
              \"max_iters\": 1,\
              \"algorithm\": \"CLASSICAL\",\
              \"interpolator\": \"D2\",\
              \"max_levels\": 10,\
              \"smoother\": \"BLOCK_JACOBI\",\
              \"strength\": \"AHAT\",\
              \"strength_threshold\": 0.75,\
              \"presweeps\": 1,\
              \"postsweeps\": 1,\
              \"coarse_solver\": \"DENSE_LU_SOLVER\",\
              \"min_coarse_rows\": 256,\
              \"print_solve_stats\": 0,\
              \"monitor_residual\": 0,\
              \"obtain_timings\": 0\
              }\
        }";

        AMGX_config_create((AMGX_config_handle_struct **)&config, default_config);
        printf("AMGX: %s not found; default configuration applied\n", config_filename.c_str());
      }

      if (amgx_instances == 0)
      {
        AMGX_resources_create((AMGX_resources_handle_struct **)&rsrc, (AMGX_config_handle)config,
          NULL, 1, &device_num);
        printf("AMGX: running on device %d\n", device_num);
      }

      amgx_instances++;
      AMGX_mode = AMGX_mode_dDDI;

      AMGX_solver_create((AMGX_solver_handle_struct **)&solver, (AMGX_resources_handle)rsrc,
        (AMGX_Mode)AMGX_mode, (AMGX_config_handle)config);
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_amgx<N_BLOCK_SIZE>::~linsolv_amgx()
    {
      // A/x/b are created lazily in init(); an instance that was constructed
      // but never initialised (e.g. a transposed pressure preconditioner on a
      // forward-only run) has null handles.
      if (A)
        AMGX_matrix_destroy((AMGX_matrix_handle)A);
      if (x)
        AMGX_vector_destroy((AMGX_vector_handle)x);
      if (b)
        AMGX_vector_destroy((AMGX_vector_handle)b);
      AMGX_solver_destroy((AMGX_solver_handle)solver);

      AMGX_SAFE_CALL(AMGX_config_destroy((AMGX_config_handle)config));
      amgx_instances--;

      // Tear down the shared resources only with the last solver instance.
      if (amgx_instances == 0)
      {
        AMGX_resources_destroy((AMGX_resources_handle)rsrc);
        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        AMGX_SAFE_CALL(AMGX_finalize());
      }
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_amgx<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface * /*prec_input*/)
    {
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_amgx<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A_input,
      opendarts::config::index_t /*max_iters*/,
      opendarts::config::mat_float /*tolerance*/)
    {
      // Repeated init() (the adjoint backward driver re-inits its solver
      // chain on every gradient evaluation) must not leak the previous
      // handles: leaked AMGX objects outlive AMGX_finalize and are torn down
      // by AMGX's atexit MemManager after the CUDA context is gone, which
      // aborts the process at exit.
      if (A)
        AMGX_matrix_destroy((AMGX_matrix_handle)A);
      if (x)
        AMGX_vector_destroy((AMGX_vector_handle)x);
      if (b)
        AMGX_vector_destroy((AMGX_vector_handle)b);
      AMGX_matrix_create((AMGX_matrix_handle_struct **)&A, (AMGX_resources_handle)rsrc, (AMGX_Mode)AMGX_mode);
      AMGX_vector_create((AMGX_vector_handle_struct **)&x, (AMGX_resources_handle)rsrc, (AMGX_Mode)AMGX_mode);
      AMGX_vector_create((AMGX_vector_handle_struct **)&b, (AMGX_resources_handle)rsrc, (AMGX_Mode)AMGX_mode);

      // bs1 expansion path: AMGX consumes a scalar-CSR device view. For the
      // legacy csr_matrix<N> Jacobian this is the in-place convert_to_ELL
      // pathway; for the unified block_csr_matrix it is the
      // gpu_bsr_spmv-backed cusparseDbsr2csr buffer
      // (block_csr_matrix::build_scalar_csr_device).
      if (N_BLOCK_SIZE > 1 && convert_to_bs1)
      {
        auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input);
        if (A_typed != nullptr)
        {
          A_typed->convert_to_ELL();
          AMGX_matrix_upload_all((AMGX_matrix_handle)A, A_typed->n_rows * N_BLOCK_SIZE,
            A_typed->rows_ptr[A_typed->n_rows] * N_BLOCK_SIZE * N_BLOCK_SIZE, 1, 1,
            A_typed->csrRowPtrC, A_typed->csrColIndC, A_typed->csrValC, 0);
          return 0;
        }
        auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input);
        if (A_block != nullptr)
        {
          if (A_block->build_scalar_csr_device() != 0)
            return -1;
          AMGX_matrix_upload_all((AMGX_matrix_handle)A,
            A_block->n_rows * N_BLOCK_SIZE,
            static_cast<int>(A_block->scalar_csr_nnz()), 1, 1,
            A_block->scalar_csr_row_ptr_device(),
            A_block->scalar_csr_col_ind_device(),
            A_block->scalar_csr_values_device(), 0);
          return 0;
        }
        // Unknown subclass -- fall through to the native block path.
      }

      AMGX_matrix_upload_all((AMGX_matrix_handle)A, A_input->n_rows,
        A_input->get_rows_ptr()[A_input->n_rows], N_BLOCK_SIZE, N_BLOCK_SIZE,
        A_input->get_rows_ptr_d(), A_input->get_cols_ind_d(), A_input->get_values_d(), 0);
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_amgx<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A_input)
    {
      const std::string timer_key = "AMGX<" + std::to_string((int)N_BLOCK_SIZE) + ">";
      this->timer_setup->node[timer_key].start();

      bool used_bs1 = false;
      if (N_BLOCK_SIZE > 1 && convert_to_bs1)
      {
        auto *A_typed = dynamic_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A_input);
        if (A_typed != nullptr)
        {
          A_typed->convert_to_ELL();
          AMGX_matrix_replace_coefficients((AMGX_matrix_handle)A, A_typed->n_rows * N_BLOCK_SIZE,
            A_typed->rows_ptr[A_typed->n_rows] * N_BLOCK_SIZE * N_BLOCK_SIZE, A_typed->csrValC, 0);
          used_bs1 = true;
        }
        else
        {
          auto *A_block = dynamic_cast<opendarts::linear_solvers::block_csr_matrix *>(A_input);
          if (A_block != nullptr)
          {
            if (A_block->build_scalar_csr_device() != 0)
              return -1;
            AMGX_matrix_replace_coefficients((AMGX_matrix_handle)A,
              A_block->n_rows * N_BLOCK_SIZE,
              static_cast<int>(A_block->scalar_csr_nnz()),
              A_block->scalar_csr_values_device(), 0);
            used_bs1 = true;
          }
        }
      }
      if (!used_bs1)
      {
        AMGX_matrix_replace_coefficients((AMGX_matrix_handle)A, A_input->n_rows,
          A_input->get_rows_ptr()[A_input->n_rows], A_input->get_values_d(), 0);
      }
      AMGX_solver_setup((AMGX_solver_handle)solver, (AMGX_matrix_handle)A);
      n_rows = A_input->n_rows;

      this->timer_setup->node[timer_key].stop();
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_amgx<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      const std::string timer_key = "AMGX<" + std::to_string((int)N_BLOCK_SIZE) + ">";
      this->timer_solve->node[timer_key].start();

      if (!convert_to_bs1)
      {
        AMGX_vector_upload((AMGX_vector_handle)b, n_rows, N_BLOCK_SIZE, B);
        AMGX_vector_upload((AMGX_vector_handle)x, n_rows, N_BLOCK_SIZE, X);
      }
      else
      {
        AMGX_vector_upload((AMGX_vector_handle)b, n_rows * N_BLOCK_SIZE, 1, B);
        AMGX_vector_upload((AMGX_vector_handle)x, n_rows * N_BLOCK_SIZE, 1, X);
      }

      AMGX_solver_solve_with_0_initial_guess((AMGX_solver_handle)solver, (AMGX_vector_handle)b,
        (AMGX_vector_handle)x);

      // Previously the solve status was never inspected, so a failed or
      // diverged AMGX solve (NaNs and all) was reported as success.
      // NOT_CONVERGED is fine here -- as a preconditioner stage AMGX runs a
      // fixed cycle budget and the outer Krylov drives convergence.
      AMGX_SOLVE_STATUS st = AMGX_SOLVE_SUCCESS;
      AMGX_solver_get_status((AMGX_solver_handle)solver, &st);
      if (st == AMGX_SOLVE_FAILED || st == AMGX_SOLVE_DIVERGED)
      {
        printf("AMGX: solve %s\n", st == AMGX_SOLVE_FAILED ? "failed" : "diverged");
        this->timer_solve->node[timer_key].stop();
        return -1;
      }

      AMGX_vector_download((AMGX_vector_handle)x, X);

      this->timer_solve->node[timer_key].stop();
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_amgx<N_BLOCK_SIZE>::get_n_iters()
    {
      int n;
      AMGX_solver_get_iterations_number((AMGX_solver_handle)solver, &n);
      return n - 1;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_amgx<N_BLOCK_SIZE>::get_residual()
    {
      double final_residual = 0;
      if (convert_to_bs1)
      {
        AMGX_solver_get_iteration_residual((AMGX_solver_handle)solver, get_n_iters(), 0, &final_residual);
      }
      else
      {
        for (int idx = 0; idx < N_BLOCK_SIZE; idx++)
        {
          double comp_residual;
          AMGX_solver_get_iteration_residual((AMGX_solver_handle)solver, get_n_iters(), idx, &comp_residual);
          if (comp_residual > final_residual)
            final_residual = comp_residual;
        }
      }
      return final_residual;
    }

    // Explicit template instantiations — generated by CMake's
    // od_emit_template_instantiations(); see solvers/src/CMakeLists.txt.
    // Edit the block-size range there, not here.
#include "linsolv_amgx_instantiations.inc"
  } // namespace linear_solvers
} // namespace opendarts

#endif // WITH_GPU && WITH_AMGX
