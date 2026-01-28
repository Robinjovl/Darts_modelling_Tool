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

#include "linsolv_iface_bos.hpp"
#include "LinearSolver.hpp"
#include <memory>
#include <string>

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

      // Configuration methods for MGR solver
      void set_max_iterations(opendarts::config::index_t max_iters);
      void set_tolerance(opendarts::config::mat_float tolerance);
      void set_kdim(int kdim);
      void set_use_mgr(bool use_mgr);
      void set_log_level(int log_level);
      void set_dump_ij_matrix(bool enabled);
      void set_ij_dump_file(const std::string& filename);

      // Get current configuration
      opendarts::config::index_t get_max_iterations() const;
      opendarts::config::mat_float get_tolerance() const;
      int get_kdim() const;
      bool get_use_mgr() const;
      int get_log_level() const;
      bool get_dump_ij_matrix() const;
      std::string get_ij_dump_file() const;

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

      // Cached configuration parameters (to avoid overwriting during solve)
      int kdim_cached;
      bool use_mgr_cached;
      int log_level_cached;
      bool dump_ij_matrix_cached;
      std::string ij_dump_file_cached;
    };

  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_MGR_HPP
//--------------------------------------------------------------------------
