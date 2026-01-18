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
#include "openDARTS/linear_solvers/linsolv_mgr.hpp"
#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "mgr-linear-solver/LinearSolver.hpp"
#include "mgr-linear-solver/strategies/CompositionalFlowStrategy.hpp"
#include "mgr-linear-solver/Types.hpp"
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
    int linsolv_mgr<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A,
                                        opendarts::config::index_t max_iters,
                                        opendarts::config::mat_float tolerance)
    {

      // Just store parameters and matrix pointer (like SuperLU)
      matrix_ptr = A;
      max_iters_cached = max_iters;
      tolerance_cached = tolerance;

      opendarts::config::index_t n_blocks = A->n_rows;
      opendarts::config::index_t block_size = N_BLOCK_SIZE;
      global_num_rows = n_blocks * block_size;


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

      // Get block CSR parameters
      opendarts::config::index_t n_blocks = matrix_ptr->n_rows;
      opendarts::config::index_t block_size = N_BLOCK_SIZE;
      opendarts::config::index_t nnz_blocks = matrix_ptr->n_non_zeros;

      opendarts::config::index_t *row_ptr = matrix_ptr->get_rows_ptr();
      opendarts::config::index_t *col_ind = matrix_ptr->get_cols_ind();
      opendarts::config::mat_float *values = matrix_ptr->get_values();


      // Compute expanded dimensions
      opendarts::config::index_t num_rows = n_blocks * block_size;
      opendarts::config::index_t num_cols = n_blocks * block_size;
      opendarts::config::index_t nnz_expanded = nnz_blocks * block_size * block_size;

      // Allocate expanded CSR arrays
      std::vector<opendarts::config::index_t> row_ptr_expanded(num_rows + 1);
      std::vector<opendarts::config::index_t> col_ind_expanded;
      std::vector<opendarts::config::mat_float> values_expanded;

      col_ind_expanded.reserve(nnz_expanded);
      values_expanded.reserve(nnz_expanded);

      // Expand block CSR to regular CSR
      // This works for both block_size=1 and block_size>1
      opendarts::config::index_t value_idx = 0;

      for (opendarts::config::index_t block_row = 0; block_row < n_blocks; ++block_row)
      {
        // For each DOF within this block row
        for (opendarts::config::index_t i = 0; i < block_size; ++i)
        {
          opendarts::config::index_t global_row = block_row * block_size + i;
          row_ptr_expanded[global_row] = value_idx;

          // Iterate over all blocks in this row
          for (opendarts::config::index_t block_idx = row_ptr[block_row];
               block_idx < row_ptr[block_row + 1];
               ++block_idx)
          {
            opendarts::config::index_t block_col = col_ind[block_idx];
            opendarts::config::index_t block_start = block_idx * block_size * block_size;

            // For each DOF within this block column
            for (opendarts::config::index_t j = 0; j < block_size; ++j)
            {
              opendarts::config::index_t global_col = block_col * block_size + j;
              opendarts::config::mat_float val = values[block_start + i * block_size + j];

              col_ind_expanded.push_back(global_col);
              values_expanded.push_back(val);
              value_idx++;
            }
          }
        }
      }

      // Set the last row pointer
      row_ptr_expanded[num_rows] = value_idx;


      // Pass expanded CSR to MGR solver as block_size=1 (regular CSR)
      if (!mgr_solver.setMatrixFromCSR(num_rows, num_cols, 1, value_idx,
                                       row_ptr_expanded.data(),
                                       col_ind_expanded.data(),
                                       values_expanded.data(),
                                       nullptr))
      {
        std::cerr << "[MGR] Error: Failed to set matrix from expanded CSR data" << std::endl;
        return -1;
      }

      // Only create strategy on first solve
      if (first_solve)
      {

        auto strategy = std::make_unique<mgr::strategies::CompositionalFlowStrategy>(
            block_size, n_blocks * block_size, n_blocks);
        strategy->setup();
        mgr_solver.setStrategy(std::move(strategy));

        if (mgr_solver.setup(max_iters_cached, tolerance_cached) != 0)
        {
          std::cerr << "[MGR] Error: Failed to setup solver" << std::endl;
          return -1;
        }

        first_solve = false;
      }

      // Solve
      mgr::int_t iters = mgr_solver.solve(B, X);

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
