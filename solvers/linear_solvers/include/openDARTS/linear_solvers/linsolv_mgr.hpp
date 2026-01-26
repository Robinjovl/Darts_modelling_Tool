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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_MGR_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_MGR_HPP
//--------------------------------------------------------------------------

#include "openDARTS/linear_solvers/linsolv_iface_bos.hpp"
#include "mgr-linear-solver/LinearSolver.hpp"
#include <memory>

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_mgr : public linsolv_iface_bos<N_BLOCK_SIZE>
    {
    public:
      linsolv_mgr();
      virtual ~linsolv_mgr();

      // Set preconditioner (not used by MGR, but required by interface)
      int set_prec(opendarts::linear_solvers::linsolv_iface *prec_input) override;

      // Implement the template-specific init from linsolv_iface_bos
      int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A,
               opendarts::config::index_t max_iters,
               opendarts::config::mat_float tolerance) override;

      // Implement the template-specific setup from linsolv_iface_bos
      int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A) override;

      // Solve linear system
      int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;

      // Set verbosity for HYPRE-backed MGR solver
      void set_log_level(int log_level);

      // Get number of iterations from last solve
      int get_n_iters() override;

      // Get final residual from last solve
      opendarts::config::mat_float get_residual() override;

    private:
      mgr::LinearSolver mgr_solver;
      bool initialized;
      bool first_solve;
      opendarts::config::index_t global_num_rows;  // Cached matrix size
      opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *matrix_ptr;  // Pointer to open-darts matrix
      opendarts::config::index_t max_iters_cached;
      opendarts::config::mat_float tolerance_cached;
    };

  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_MGR_HPP
//--------------------------------------------------------------------------
