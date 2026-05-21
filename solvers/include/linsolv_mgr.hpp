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
      void set_mgr_scaling_type(int scaling_type);
      void set_use_flex_gmres(bool use_flex_gmres);
      void set_mgr_composite_mode(int composite_mode);
      void set_mgr_local_solver(int local_solver);
      void set_mgr_bilu0_pivot_shift(opendarts::config::mat_float pivot_shift);
      void set_mgr_bilu0_fallback_options(int fallback_strategy,
                                          opendarts::config::mat_float diagonal_tolerance,
                                          opendarts::config::mat_float shifted_max,
                                          opendarts::config::mat_float shifted_growth);
      void set_mgr_local_correction_options(opendarts::config::mat_float alpha,
                                            opendarts::config::mat_float adaptive_fallback_threshold,
                                            opendarts::config::mat_float adaptive_alpha,
                                            opendarts::config::mat_float adaptive_fallback_threshold_high,
                                            opendarts::config::mat_float adaptive_alpha_high);
      void set_mgr_local_correction_quality_options(bool enabled,
                                                    opendarts::config::mat_float min_alpha);
      void set_use_bcsr_cpr(bool use_bcsr_cpr);
      void set_bcsr_cpr_options(int reduction_type,
                                int pressure_variable,
                                opendarts::config::mat_float weight_max);
      void set_bcsr_cpr_reuse_options(bool reuse_amg_hierarchy,
                                      opendarts::config::index_t amg_rebuild_interval);
      void set_bcsr_cpr_adaptive_rebuild_options(
          bool adaptive_amg_rebuild,
          opendarts::config::index_t li_threshold,
          opendarts::config::mat_float li_growth_factor,
          opendarts::config::index_t min_reuse_setups,
          opendarts::config::index_t max_reuse_setups);
      void set_bcsr_cpr_adaptive_quality_options(
          opendarts::config::mat_float pressure_overshoot_threshold,
          opendarts::config::mat_float final_proxy_threshold,
          opendarts::config::mat_float fallback_threshold);
      void set_bcsr_cpr_diagnostics_options(
          bool diagnostics,
          opendarts::config::index_t apply_interval,
          opendarts::config::index_t matrix_interval);
      void set_bcsr_cpr_pressure_correction_options(
          opendarts::config::mat_float alpha,
          opendarts::config::mat_float guard_threshold,
          opendarts::config::mat_float guard_min_alpha);
      void set_mgr_pressure_amg_options(int coarsen_type,
                                        int interp_type,
                                        int relax_type,
                                        int agg_num_levels,
                                        int agg_interp_type,
                                        int agg_pmax_elmts,
                                        int relax_order);
      void set_mgr_pressure_amg_solve_options(opendarts::config::index_t max_iter,
                                              opendarts::config::mat_float tolerance);
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
      int get_mgr_scaling_type() const;
      bool get_use_flex_gmres() const;
      int get_mgr_composite_mode() const;
      int get_mgr_local_solver() const;
      opendarts::config::mat_float get_mgr_bilu0_pivot_shift() const;
      int get_mgr_bilu0_fallback_strategy() const;
      opendarts::config::mat_float get_mgr_bilu0_fallback_diagonal_tolerance() const;
      opendarts::config::mat_float get_mgr_bilu0_fallback_shifted_max() const;
      opendarts::config::mat_float get_mgr_bilu0_fallback_shifted_growth() const;
      opendarts::config::mat_float get_mgr_local_correction_alpha() const;
      opendarts::config::mat_float get_mgr_local_correction_adaptive_fallback_threshold() const;
      opendarts::config::mat_float get_mgr_local_correction_adaptive_alpha() const;
      opendarts::config::mat_float get_mgr_local_correction_adaptive_fallback_threshold_high() const;
      opendarts::config::mat_float get_mgr_local_correction_adaptive_alpha_high() const;
      bool get_use_bcsr_cpr() const;
      int get_bcsr_cpr_reduction_type() const;
      int get_bcsr_cpr_pressure_variable() const;
      opendarts::config::mat_float get_bcsr_cpr_weight_max() const;
      bool get_bcsr_cpr_reuse_amg_hierarchy() const;
      opendarts::config::index_t get_bcsr_cpr_amg_rebuild_interval() const;
      bool get_bcsr_cpr_adaptive_amg_rebuild() const;
      opendarts::config::index_t get_bcsr_cpr_adaptive_li_threshold() const;
      opendarts::config::mat_float get_bcsr_cpr_adaptive_li_growth_factor() const;
      opendarts::config::index_t get_bcsr_cpr_adaptive_min_reuse_setups() const;
      opendarts::config::index_t get_bcsr_cpr_adaptive_max_reuse_setups() const;
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
      int scaling_type_cached;
      bool use_flex_gmres_cached;
      int composite_mode_cached;
      int local_solver_cached;
      opendarts::config::mat_float bilu0_pivot_shift_cached;
      int bilu0_fallback_strategy_cached;
      opendarts::config::mat_float bilu0_fallback_diagonal_tolerance_cached;
      opendarts::config::mat_float bilu0_fallback_shifted_max_cached;
      opendarts::config::mat_float bilu0_fallback_shifted_growth_cached;
      opendarts::config::mat_float local_correction_alpha_cached;
      opendarts::config::mat_float local_correction_adaptive_fallback_threshold_cached;
      opendarts::config::mat_float local_correction_adaptive_alpha_cached;
      opendarts::config::mat_float local_correction_adaptive_fallback_threshold_high_cached;
      opendarts::config::mat_float local_correction_adaptive_alpha_high_cached;
      bool local_correction_quality_gate_cached;
      opendarts::config::mat_float local_correction_quality_min_alpha_cached;
      bool use_bcsr_cpr_cached;
      int bcsr_cpr_reduction_type_cached;
      int bcsr_cpr_pressure_variable_cached;
      opendarts::config::mat_float bcsr_cpr_weight_max_cached;
      bool bcsr_cpr_reuse_amg_hierarchy_cached;
      opendarts::config::index_t bcsr_cpr_amg_rebuild_interval_cached;
      bool bcsr_cpr_adaptive_amg_rebuild_cached;
      opendarts::config::index_t bcsr_cpr_adaptive_li_threshold_cached;
      opendarts::config::mat_float bcsr_cpr_adaptive_li_growth_factor_cached;
      opendarts::config::index_t bcsr_cpr_adaptive_min_reuse_setups_cached;
      opendarts::config::index_t bcsr_cpr_adaptive_max_reuse_setups_cached;
      opendarts::config::mat_float bcsr_cpr_adaptive_pressure_overshoot_threshold_cached;
      opendarts::config::mat_float bcsr_cpr_adaptive_final_proxy_threshold_cached;
      opendarts::config::mat_float bcsr_cpr_adaptive_fallback_threshold_cached;
      bool bcsr_cpr_diagnostics_cached;
      opendarts::config::index_t bcsr_cpr_diagnostic_apply_interval_cached;
      opendarts::config::index_t bcsr_cpr_diagnostic_matrix_interval_cached;
      opendarts::config::mat_float bcsr_cpr_pressure_correction_alpha_cached;
      opendarts::config::mat_float bcsr_cpr_pressure_correction_guard_threshold_cached;
      opendarts::config::mat_float bcsr_cpr_pressure_correction_guard_min_alpha_cached;
      opendarts::config::index_t pressure_amg_max_iter_cached;
      opendarts::config::mat_float pressure_amg_tolerance_cached;
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
