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
#include "CompositionalFlowStrategy.hpp"
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
      void set_use_physics_scaling(bool use_scaling);
      void set_use_flex_gmres(bool use_flex_gmres);
      void set_mgr_pressure_amg_options(int coarsen_type,
                                        int interp_type,
                                        int relax_type,
                                        int agg_num_levels,
                                        int agg_interp_type,
                                        int agg_pmax_elmts,
                                        int relax_order);
      void set_n_reservoir_blocks(opendarts::config::index_t n_reservoir_blocks);
      void set_mgr_enable_well_level(bool enable_well_level);
      void set_mgr_enable_composition_level(bool enable_composition_level);
      void set_mgr_reservoir_variable_roles(const std::vector<int> & variable_roles);
      void set_mgr_well_variable_roles(const std::vector<int> & variable_roles);
      void clear_mgr_custom_levels();
      void set_mgr_num_custom_levels(int num_custom_levels);
      void set_mgr_custom_level_options(int custom_level_index,
                                        const std::vector<int> & keep_labels,
                                        int frelax_type,
                                        int frelax_iters,
                                        int interp_type,
                                        int restrict_type,
                                        int coarse_method,
                                        int smoother_type,
                                        int smoother_iters);
      void set_mgr_well_strategy(int well_strategy);
      void set_mgr_well_frelax_type(int frelax_type);
      void set_mgr_well_frelax_iters(int frelax_iters);
      void set_mgr_well_level_options(int frelax_type,
                                      int frelax_iters,
                                      int interp_type,
                                      int restrict_type,
                                      int coarse_method,
                                      int smoother_type,
                                      int smoother_iters);
      void set_mgr_composition_level_options(int frelax_type,
                                             int frelax_iters,
                                             int interp_type,
                                             int restrict_type,
                                             int coarse_method,
                                             int smoother_type,
                                             int smoother_iters);
      void set_mgr_pressure_level_options(int frelax_type,
                                          int frelax_iters,
                                          int interp_type,
                                          int restrict_type,
                                          int coarse_method,
                                          int smoother_type,
                                          int smoother_iters);

      // Get current configuration
      opendarts::config::index_t get_max_iterations() const;
      opendarts::config::mat_float get_tolerance() const;
      int get_kdim() const;
      bool get_use_mgr() const;
      int get_log_level() const;
      bool get_use_physics_scaling() const;
      bool get_use_flex_gmres() const;
      opendarts::config::index_t get_n_reservoir_blocks() const;
      bool get_mgr_enable_well_level() const;
      bool get_mgr_enable_composition_level() const;
      int get_mgr_num_custom_levels() const;
      std::vector<int> get_mgr_reservoir_variable_roles() const;
      std::vector<int> get_mgr_well_variable_roles() const;
      int get_mgr_well_strategy() const;
      int get_mgr_well_frelax_type() const;
      int get_mgr_well_frelax_iters() const;

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
      bool use_physics_scaling_cached;
      bool use_flex_gmres_cached;
      opendarts::config::index_t n_reservoir_blocks_cached;
      mgr::strategies::CompositionalFlowStrategyConfig mgr_strategy_config_cached;

      static std::vector<mgr::strategies::VariableRole> to_variable_roles(const std::vector<int> & variable_roles);
      static std::vector<int> to_int_roles(const std::vector<mgr::strategies::VariableRole> & variable_roles);
      static std::vector<mgr::int_t> to_labels(const std::vector<int> & labels);
      static void set_level_options(mgr::MGRLevelParameters & level,
                                    int frelax_type,
                                    int frelax_iters,
                                    int interp_type,
                                    int restrict_type,
                                    int coarse_method,
                                    int smoother_type,
                                    int smoother_iters);
    };

  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_MGR_HPP
//--------------------------------------------------------------------------
