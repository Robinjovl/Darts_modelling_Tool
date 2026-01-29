//*************************************************************************
//    Copyright (c) 2026
//    MGR Linear Solver Integration
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
//
//*************************************************************************

//--------------------------------------------------------------------------
#include "linsolv_mgr.hpp"
#include "csr_matrix_base.hpp"
#include "LinearSolver.hpp"
#include "CompositionalFlowStrategy.hpp"
#include "Types.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
//--------------------------------------------------------------------------

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    linsolv_mgr<N_BLOCK_SIZE>::linsolv_mgr()
      : linsolv_iface_bos<N_BLOCK_SIZE>()
      , initialized(false)
      , first_solve(true)
      , global_num_rows(0)
      , matrix_ptr(nullptr)
      , max_iters_cached(50)
      , tolerance_cached(1e-3)
      , kdim_cached(30)
      , use_mgr_cached(true)
      , log_level_cached(1)
      , use_physics_scaling_cached(true)
      , use_flex_gmres_cached(true)
    {
      std::cout << "[MGR] linsolv_mgr created with N_BLOCK_SIZE = " << (int)N_BLOCK_SIZE << std::endl;
    }

    template <uint8_t N_BLOCK_SIZE>
    linsolv_mgr<N_BLOCK_SIZE>::~linsolv_mgr()
    {
      std::cout << "[MGR] linsolv_mgr destroyed" << std::endl;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_input)
    {
      // MGR has built-in preconditioner, no external preconditioner needed
      (void)prec_input;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_max_iterations(opendarts::config::index_t max_iters)
    {
      max_iters_cached = max_iters;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.maxIter = max_iters;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_tolerance(opendarts::config::mat_float tolerance)
    {
      tolerance_cached = tolerance;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.tolerance = tolerance;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_kdim(int kdim)
    {
      kdim_cached = kdim;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.kdim = kdim;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_use_mgr(bool use_mgr)
    {
      use_mgr_cached = use_mgr;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.useMGR = use_mgr;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_log_level(int log_level)
    {
      log_level_cached = log_level;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.logLevel = log_level;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_use_physics_scaling(bool use_scaling)
    {
      use_physics_scaling_cached = use_scaling;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.usePhysicsScaling = use_scaling;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_use_flex_gmres(bool use_flex_gmres)
    {
      use_flex_gmres_cached = use_flex_gmres;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.krylovType = use_flex_gmres ? mgr::KrylovType::flexgmres
                                         : mgr::KrylovType::gmres;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::index_t linsolv_mgr<N_BLOCK_SIZE>::get_max_iterations() const
    {
      return max_iters_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_tolerance() const
    {
      return tolerance_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_kdim() const
    {
      return kdim_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_use_mgr() const
    {
      return use_mgr_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_log_level() const
    {
      return log_level_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_use_physics_scaling() const
    {
      return use_physics_scaling_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_use_flex_gmres() const
    {
      return use_flex_gmres_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A,
                                        opendarts::config::index_t max_iters,
                                        opendarts::config::mat_float tolerance)
    {
      // Store parameters and matrix pointer (like SuperLU)
      matrix_ptr = A;
      max_iters_cached = max_iters;
      tolerance_cached = tolerance;

      opendarts::config::index_t n_blocks = A->n_rows;
      opendarts::config::index_t block_size = N_BLOCK_SIZE;
      global_num_rows = n_blocks * block_size;

      // Keep block size for MGR reduction (dofs per cell)
      mgr_solver.setMGRBlockSize(block_size);

      // Update mgr_solver parameters with current cached values
      mgr::SolverParameters params;
      params.maxIter = max_iters_cached;
      params.tolerance = tolerance_cached;
      params.kdim = kdim_cached;
      params.useMGR = use_mgr_cached;
      params.logLevel = log_level_cached;
      params.usePhysicsScaling = use_physics_scaling_cached;
      params.krylovType = use_flex_gmres_cached ? mgr::KrylovType::flexgmres
                                               : mgr::KrylovType::gmres;
      mgr_solver.setParameters(params);

      initialized = true;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A)
    {
      // Just update matrix pointer (like SuperLU)
      matrix_ptr = A;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      if (!initialized)
      {
        std::cerr << "[MGR] Error: Solver not initialized. Call init() first." << std::endl;
        return -1;
      }

      if (matrix_ptr == nullptr)
      {
        std::cerr << "[MGR] Error: Matrix pointer is null." << std::endl;
        return -1;
      }

      // Get block CSR parameters
      const opendarts::config::index_t n_blocks = matrix_ptr->n_rows;
      const opendarts::config::index_t block_size = N_BLOCK_SIZE;
      const opendarts::config::index_t nnz_blocks_declared = matrix_ptr->n_non_zeros;
      opendarts::config::index_t *row_ptr = matrix_ptr->get_rows_ptr();
      opendarts::config::index_t *col_ind = matrix_ptr->get_cols_ind();
      opendarts::config::mat_float *values = matrix_ptr->get_values();
      opendarts::config::index_t *diag_ind = matrix_ptr->get_diag_ind();

      if (row_ptr == nullptr || col_ind == nullptr || values == nullptr)
      {
        std::cerr << "[MGR] Error: Matrix data pointers are null." << std::endl;
        return -1;
      }

      // Validate row_ptr monotonicity and bounds vs declared nnz_blocks
      const opendarts::config::index_t nnz_blocks_from_rows = row_ptr[n_blocks];
      if (nnz_blocks_from_rows > nnz_blocks_declared)
      {
        std::cerr << "[MGR] Error: row_ptr last entry (" << nnz_blocks_from_rows
                  << ") exceeds declared nnz_blocks (" << nnz_blocks_declared << ")."
                  << std::endl;
        return -1;
      }

      for (opendarts::config::index_t i = 0; i < n_blocks; ++i)
      {
        if (row_ptr[i] > row_ptr[i + 1] || row_ptr[i + 1] > nnz_blocks_declared)
        {
          std::cerr << "[MGR] Error: Invalid row_ptr at row " << i
                    << " (row_ptr[i]=" << row_ptr[i]
                    << ", row_ptr[i+1]=" << row_ptr[i + 1]
                    << ", nnz_blocks=" << nnz_blocks_declared << ")."
                    << std::endl;
          return -1;
        }
      }

      if (nnz_blocks_from_rows != nnz_blocks_declared)
      {
        std::cerr << "[MGR] Warning: row_ptr last entry (" << nnz_blocks_from_rows
                  << ") does not match declared nnz_blocks (" << nnz_blocks_declared << ")."
                  << std::endl;
      }

      const opendarts::config::index_t nnz_blocks = nnz_blocks_declared;

      // Pass block CSR directly to MGR solver
      if (!mgr_solver.setMatrixFromCSR(n_blocks, n_blocks, block_size, nnz_blocks,
                                       row_ptr,
                                       col_ind,
                                       values,
                                       diag_ind))
      {
        std::cerr << "[MGR] Error: Failed to set matrix from CSR data" << std::endl;
        return -1;
      }

      // Only create strategy on first solve
      if (first_solve)
      {
        auto strategy = std::make_unique<mgr::strategies::CompositionalFlowStrategy>(
            block_size, n_blocks * block_size, n_blocks);
        strategy->setup();
        mgr_solver.setStrategy(std::move(strategy));

        // Use cached parameters for setup
        mgr::SolverParameters setup_params = mgr_solver.getParameters();
        if (mgr_solver.setup(setup_params.maxIter, setup_params.tolerance) != 0)
        {
          std::cerr << "[MGR] Error: Failed to setup solver" << std::endl;
          return -1;
        }

        first_solve = false;
      }

      if (log_level_cached >= 2)
      {
        std::cout << "[MGR] Solving with tolerance=" << tolerance_cached
                  << ", max_iter=" << max_iters_cached << std::endl;
      }

      // Solve
      mgr::int_t iters = mgr_solver.solve(B, X);

      if (log_level_cached >= 2)
      {
        const bool converged = (iters >= 0);
        const mgr::int_t iters_reported = converged ? iters : -iters;
        const auto final_res = mgr_solver.get_residual();
        std::cout << "[MGR] Solve complete: iterations=" << iters_reported
                  << ", final_res=" << final_res
                  << ", converged=" << (converged ? "YES" : "NO") << std::endl;
      }

      if (iters < 0)
      {
        std::cerr << "[MGR] Warning: Solve did not converge (iters = " << -iters << ")" << std::endl;
        return iters;  // Return negative iteration count for failure
      }

      // Return 0 for success (open-darts convention)
      // Iteration count is available via get_n_iters()
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_n_iters()
    {
      return mgr_solver.get_n_iters();
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_residual()
    {
      return mgr_solver.get_residual();
    }

    // Explicit template instantiations
    // Based on MAX_NC = 8 and THERMAL = 1, max N_VARS = 9
    // Instantiate up to 13 to match other solvers (linsolv_superlu)
    template class linsolv_mgr<1>;
    template class linsolv_mgr<2>;
    template class linsolv_mgr<3>;
    template class linsolv_mgr<4>;
    template class linsolv_mgr<5>;
    template class linsolv_mgr<6>;
    template class linsolv_mgr<7>;
    template class linsolv_mgr<8>;
    template class linsolv_mgr<9>;
    template class linsolv_mgr<10>;
    template class linsolv_mgr<11>;
    template class linsolv_mgr<12>;
    template class linsolv_mgr<13>;

  } // namespace linear_solvers
} // namespace opendarts
