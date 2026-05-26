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
    namespace
    {
      class ScopedTimer
      {
      public:
        explicit ScopedTimer(::timer_node *timer)
          : timer_(timer)
        {
          if (timer_)
          {
            timer_->start();
          }
        }

        ~ScopedTimer()
        {
          if (timer_)
          {
            timer_->stop();
          }
        }

        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;

      private:
        ::timer_node *timer_;
      };

      template <uint8_t N_BLOCK_SIZE>
      void build_transpose_from_base(csr_matrix_base *src,
                                     csr_matrix<N_BLOCK_SIZE> &dst)
      {
        const opendarts::config::index_t n_rows = src->n_rows;
        const opendarts::config::index_t n_cols = src->n_cols;
        const opendarts::config::index_t block_size = src->n_row_size;
        const opendarts::config::index_t block_size_sq = block_size * block_size;
        const opendarts::config::index_t *src_rows = src->get_rows_ptr();
        const opendarts::config::index_t *src_cols = src->get_cols_ind();
        const opendarts::config::mat_float *src_vals = src->get_values();
        const opendarts::config::index_t nnz = src_rows[n_rows];

        dst.init(n_cols, n_rows, nnz);
        dst.type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
        dst.diag_ind.assign(n_cols, -1);

        std::fill(dst.rows_ptr.begin(), dst.rows_ptr.end(), 0);
        for (opendarts::config::index_t k = 0; k < nnz; ++k)
        {
          dst.rows_ptr[src_cols[k] + 1]++;
        }
        for (opendarts::config::index_t i = 0; i < n_cols; ++i)
        {
          dst.rows_ptr[i + 1] += dst.rows_ptr[i];
        }

        std::vector<opendarts::config::index_t> head(
            dst.rows_ptr.begin(), dst.rows_ptr.begin() + n_cols);
        for (opendarts::config::index_t row = 0; row < n_rows; ++row)
        {
          for (opendarts::config::index_t jb = src_rows[row]; jb < src_rows[row + 1]; ++jb)
          {
            const opendarts::config::index_t col = src_cols[jb];
            const opendarts::config::index_t dst_block = head[col]++;
            dst.cols_ind[dst_block] = row;
            if (row == col)
            {
              dst.diag_ind[col] = dst_block;
            }

            const opendarts::config::mat_float *src_block = src_vals + jb * block_size_sq;
            opendarts::config::mat_float *dst_block_vals = dst.values.data() + dst_block * block_size_sq;
            for (opendarts::config::index_t i = 0; i < block_size; ++i)
            {
              for (opendarts::config::index_t j = 0; j < block_size; ++j)
              {
                dst_block_vals[j * block_size + i] = src_block[i * block_size + j];
              }
            }
          }
        }

        dst.n_row_size = N_BLOCK_SIZE;
        dst.is_square = (n_rows == n_cols) ? 1 : 0;
      }
    }

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
      , scaling_type_cached(static_cast<int>(mgr::ScalingType::physics))
      , use_flex_gmres_cached(true)
      , composite_mode_cached(static_cast<int>(mgr::CompositePreconditionerMode::mgrOnly))
      , local_solver_cached(static_cast<int>(mgr::LocalPreconditionerType::none))
      , bilu0_pivot_shift_cached(1.0e-12)
      , bilu0_fallback_strategy_cached(static_cast<int>(mgr::LocalFallbackStrategy::identity))
      , bilu0_fallback_diagonal_tolerance_cached(1.0e-4)
      , bilu0_fallback_shifted_max_cached(1.0e-4)
      , bilu0_fallback_shifted_growth_cached(100.0)
      , local_correction_alpha_cached(1.0)
      , local_correction_adaptive_fallback_threshold_cached(-1.0)
      , local_correction_adaptive_alpha_cached(0.0)
      , local_correction_adaptive_fallback_threshold_high_cached(-1.0)
      , local_correction_adaptive_alpha_high_cached(0.0)
      , local_correction_quality_gate_cached(false)
      , local_correction_quality_min_alpha_cached(0.0)
      , use_bcsr_cpr_cached(false)
      , bcsr_cpr_reduction_type_cached(static_cast<int>(mgr::BCSRCPRReductionType::trueIMPES))
      , bcsr_cpr_pressure_variable_cached(0)
      , bcsr_cpr_weight_max_cached(1.0e6)
      , bcsr_cpr_reuse_amg_hierarchy_cached(false)
      , bcsr_cpr_amg_rebuild_interval_cached(1)
      , bcsr_cpr_adaptive_amg_rebuild_cached(false)
      , bcsr_cpr_adaptive_li_threshold_cached(80)
      , bcsr_cpr_adaptive_li_growth_factor_cached(2.0)
      , bcsr_cpr_adaptive_min_reuse_setups_cached(1)
      , bcsr_cpr_adaptive_max_reuse_setups_cached(0)
      , bcsr_cpr_adaptive_pressure_overshoot_threshold_cached(-1.0)
      , bcsr_cpr_adaptive_final_proxy_threshold_cached(-1.0)
      , bcsr_cpr_adaptive_fallback_threshold_cached(-1.0)
      , bcsr_cpr_diagnostics_cached(false)
      , bcsr_cpr_diagnostic_apply_interval_cached(0)
      , bcsr_cpr_diagnostic_matrix_interval_cached(0)
      , bcsr_cpr_pressure_correction_alpha_cached(1.0)
      , bcsr_cpr_pressure_correction_guard_threshold_cached(-1.0)
      , bcsr_cpr_pressure_correction_guard_min_alpha_cached(0.0)
      , pressure_amg_max_iter_cached(1)
      , pressure_amg_tolerance_cached(0.0)
      , n_reservoir_blocks_cached(0)
      , mgr_strategy_config_cached()
    {
      this->timer_setup = nullptr;
      this->timer_solve = nullptr;
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
      scaling_type_cached = use_scaling ? static_cast<int>(mgr::ScalingType::physics)
                                        : static_cast<int>(mgr::ScalingType::none);
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.usePhysicsScaling = use_scaling;
      params.scalingType = static_cast<mgr::ScalingType>(scaling_type_cached);
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_scaling_type(int scaling_type)
    {
      scaling_type_cached = scaling_type;
      use_physics_scaling_cached =
          scaling_type != static_cast<int>(mgr::ScalingType::none);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.usePhysicsScaling = use_physics_scaling_cached;
      params.scalingType = static_cast<mgr::ScalingType>(scaling_type_cached);
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
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_composite_mode(int composite_mode)
    {
      composite_mode_cached = composite_mode;
      if (composite_mode != static_cast<int>(mgr::CompositePreconditionerMode::mgrOnly) &&
          local_solver_cached == static_cast<int>(mgr::LocalPreconditionerType::none))
      {
        local_solver_cached = static_cast<int>(mgr::LocalPreconditionerType::blockILU0);
      }

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.compositeMode = static_cast<mgr::CompositePreconditionerMode>(composite_mode_cached);
      params.localPreconditioner = static_cast<mgr::LocalPreconditionerType>(local_solver_cached);
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_local_solver(int local_solver)
    {
      local_solver_cached = local_solver;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.localPreconditioner = static_cast<mgr::LocalPreconditionerType>(local_solver_cached);
      params.compositeMode = static_cast<mgr::CompositePreconditionerMode>(composite_mode_cached);
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_bilu0_pivot_shift(opendarts::config::mat_float pivot_shift)
    {
      bilu0_pivot_shift_cached = pivot_shift;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.localPivotShift = pivot_shift;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_bilu0_fallback_options(
        int fallback_strategy,
        opendarts::config::mat_float diagonal_tolerance,
        opendarts::config::mat_float shifted_max,
        opendarts::config::mat_float shifted_growth)
    {
      bilu0_fallback_strategy_cached = fallback_strategy;
      bilu0_fallback_diagonal_tolerance_cached = std::max<opendarts::config::mat_float>(diagonal_tolerance, 0.0);
      bilu0_fallback_shifted_max_cached = std::max<opendarts::config::mat_float>(shifted_max, 0.0);
      bilu0_fallback_shifted_growth_cached = std::max<opendarts::config::mat_float>(shifted_growth, 1.0);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.localFallbackStrategy = static_cast<mgr::LocalFallbackStrategy>(bilu0_fallback_strategy_cached);
      params.localFallbackDiagonalTolerance = bilu0_fallback_diagonal_tolerance_cached;
      params.localFallbackShiftMax = bilu0_fallback_shifted_max_cached;
      params.localFallbackShiftGrowth = bilu0_fallback_shifted_growth_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_local_correction_options(
        opendarts::config::mat_float alpha,
        opendarts::config::mat_float adaptive_fallback_threshold,
        opendarts::config::mat_float adaptive_alpha,
        opendarts::config::mat_float adaptive_fallback_threshold_high,
        opendarts::config::mat_float adaptive_alpha_high)
    {
      local_correction_alpha_cached = std::max<opendarts::config::mat_float>(alpha, 0.0);
      local_correction_adaptive_fallback_threshold_cached = adaptive_fallback_threshold;
      local_correction_adaptive_alpha_cached =
          std::max<opendarts::config::mat_float>(adaptive_alpha, 0.0);
      local_correction_adaptive_fallback_threshold_high_cached =
          adaptive_fallback_threshold_high;
      local_correction_adaptive_alpha_high_cached =
          std::max<opendarts::config::mat_float>(adaptive_alpha_high, 0.0);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.localCorrectionAlpha = local_correction_alpha_cached;
      params.localCorrectionAdaptiveFallbackThreshold =
          local_correction_adaptive_fallback_threshold_cached;
      params.localCorrectionAdaptiveAlpha = local_correction_adaptive_alpha_cached;
      params.localCorrectionAdaptiveFallbackThresholdHigh =
          local_correction_adaptive_fallback_threshold_high_cached;
      params.localCorrectionAdaptiveAlphaHigh = local_correction_adaptive_alpha_high_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_local_correction_quality_options(
        bool enabled,
        opendarts::config::mat_float min_alpha)
    {
      local_correction_quality_gate_cached = enabled;
      local_correction_quality_min_alpha_cached =
          std::clamp<opendarts::config::mat_float>(min_alpha, 0.0, 1.0);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.localCorrectionQualityGate = local_correction_quality_gate_cached;
      params.localCorrectionQualityMinAlpha =
          local_correction_quality_min_alpha_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_use_bcsr_cpr(bool use_bcsr_cpr)
    {
      use_bcsr_cpr_cached = use_bcsr_cpr;
      if (use_bcsr_cpr_cached &&
          local_solver_cached == static_cast<int>(mgr::LocalPreconditionerType::none))
      {
        local_solver_cached = static_cast<int>(mgr::LocalPreconditionerType::blockILU0);
      }

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.useBCSRCPR = use_bcsr_cpr_cached;
      params.localPreconditioner =
          static_cast<mgr::LocalPreconditionerType>(local_solver_cached);
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_bcsr_cpr_options(
        int reduction_type,
        int pressure_variable,
        opendarts::config::mat_float weight_max)
    {
      bcsr_cpr_reduction_type_cached = reduction_type;
      bcsr_cpr_pressure_variable_cached = std::max(pressure_variable, 0);
      bcsr_cpr_weight_max_cached =
          std::max<opendarts::config::mat_float>(weight_max, 1.0);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.bcsrCPRReduction =
          static_cast<mgr::BCSRCPRReductionType>(bcsr_cpr_reduction_type_cached);
      params.bcsrCPRPressureVariable = bcsr_cpr_pressure_variable_cached;
      params.bcsrCPRWeightMax = bcsr_cpr_weight_max_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_bcsr_cpr_reuse_options(
        bool reuse_amg_hierarchy,
        opendarts::config::index_t amg_rebuild_interval)
    {
      bcsr_cpr_reuse_amg_hierarchy_cached = reuse_amg_hierarchy;
      bcsr_cpr_amg_rebuild_interval_cached = amg_rebuild_interval;

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.bcsrCPRReuseAMGHierarchy = bcsr_cpr_reuse_amg_hierarchy_cached;
      params.bcsrCPRAMGRebuildInterval = bcsr_cpr_amg_rebuild_interval_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_bcsr_cpr_adaptive_rebuild_options(
        bool adaptive_amg_rebuild,
        opendarts::config::index_t li_threshold,
        opendarts::config::mat_float li_growth_factor,
        opendarts::config::index_t min_reuse_setups,
        opendarts::config::index_t max_reuse_setups)
    {
      bcsr_cpr_adaptive_amg_rebuild_cached = adaptive_amg_rebuild;
      bcsr_cpr_adaptive_li_threshold_cached =
          std::max<opendarts::config::index_t>(li_threshold, 0);
      bcsr_cpr_adaptive_li_growth_factor_cached =
          std::max<opendarts::config::mat_float>(li_growth_factor, 1.0);
      bcsr_cpr_adaptive_min_reuse_setups_cached =
          std::max<opendarts::config::index_t>(min_reuse_setups, 0);
      bcsr_cpr_adaptive_max_reuse_setups_cached =
          std::max<opendarts::config::index_t>(max_reuse_setups, 0);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.bcsrCPRAdaptiveAMGRebuild = bcsr_cpr_adaptive_amg_rebuild_cached;
      params.bcsrCPRAdaptiveLIThreshold = bcsr_cpr_adaptive_li_threshold_cached;
      params.bcsrCPRAdaptiveLIGrowthFactor =
          bcsr_cpr_adaptive_li_growth_factor_cached;
      params.bcsrCPRAdaptiveMinReuseSetups =
          bcsr_cpr_adaptive_min_reuse_setups_cached;
      params.bcsrCPRAdaptiveMaxReuseSetups =
          bcsr_cpr_adaptive_max_reuse_setups_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_bcsr_cpr_adaptive_quality_options(
        opendarts::config::mat_float pressure_overshoot_threshold,
        opendarts::config::mat_float final_proxy_threshold,
        opendarts::config::mat_float fallback_threshold)
    {
      bcsr_cpr_adaptive_pressure_overshoot_threshold_cached =
          pressure_overshoot_threshold;
      bcsr_cpr_adaptive_final_proxy_threshold_cached = final_proxy_threshold;
      bcsr_cpr_adaptive_fallback_threshold_cached = fallback_threshold;

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.bcsrCPRAdaptivePressureOvershootThreshold =
          bcsr_cpr_adaptive_pressure_overshoot_threshold_cached;
      params.bcsrCPRAdaptiveFinalProxyThreshold =
          bcsr_cpr_adaptive_final_proxy_threshold_cached;
      params.bcsrCPRAdaptiveFallbackThreshold =
          bcsr_cpr_adaptive_fallback_threshold_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_bcsr_cpr_diagnostics_options(
        bool diagnostics,
        opendarts::config::index_t apply_interval,
        opendarts::config::index_t matrix_interval)
    {
      bcsr_cpr_diagnostics_cached = diagnostics;
      bcsr_cpr_diagnostic_apply_interval_cached =
          std::max<opendarts::config::index_t>(apply_interval, 0);
      bcsr_cpr_diagnostic_matrix_interval_cached =
          std::max<opendarts::config::index_t>(matrix_interval, 0);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.bcsrCPRDiagnostics = bcsr_cpr_diagnostics_cached;
      params.bcsrCPRDiagnosticApplyInterval =
          bcsr_cpr_diagnostic_apply_interval_cached;
      params.bcsrCPRDiagnosticMatrixInterval =
          bcsr_cpr_diagnostic_matrix_interval_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_bcsr_cpr_pressure_correction_options(
        opendarts::config::mat_float alpha,
        opendarts::config::mat_float guard_threshold,
        opendarts::config::mat_float guard_min_alpha)
    {
      bcsr_cpr_pressure_correction_alpha_cached =
          std::clamp<opendarts::config::mat_float>(alpha, 0.0, 1.0);
      bcsr_cpr_pressure_correction_guard_threshold_cached = guard_threshold;
      bcsr_cpr_pressure_correction_guard_min_alpha_cached =
          std::clamp<opendarts::config::mat_float>(guard_min_alpha, 0.0, 1.0);

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.bcsrCPRPressureCorrectionAlpha =
          bcsr_cpr_pressure_correction_alpha_cached;
      params.bcsrCPRPressureCorrectionGuardThreshold =
          bcsr_cpr_pressure_correction_guard_threshold_cached;
      params.bcsrCPRPressureCorrectionGuardMinAlpha =
          bcsr_cpr_pressure_correction_guard_min_alpha_cached;
      mgr_solver.setParameters(params);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_pressure_amg_options(int coarsen_type,
                                                                 int interp_type,
                                                                 int relax_type,
                                                                 int agg_num_levels,
                                                                 int agg_interp_type,
                                                                 int agg_pmax_elmts,
                                                                 int relax_order)
    {
      mgr_strategy_config_cached.pressureAmgCoarsenType = coarsen_type;
      mgr_strategy_config_cached.pressureAmgInterpType = interp_type;
      mgr_strategy_config_cached.pressureAmgRelaxType = relax_type;
      mgr_strategy_config_cached.pressureAmgAggNumLevels = std::max(0, agg_num_levels);
      mgr_strategy_config_cached.pressureAmgAggInterpType = agg_interp_type;
      mgr_strategy_config_cached.pressureAmgAggPMaxElmts = std::max(0, agg_pmax_elmts);
      mgr_strategy_config_cached.pressureAmgRelaxOrder = relax_order;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.pressureAMGCoarsenType = mgr_strategy_config_cached.pressureAmgCoarsenType;
      params.pressureAMGInterpType = mgr_strategy_config_cached.pressureAmgInterpType;
      params.pressureAMGRelaxType = mgr_strategy_config_cached.pressureAmgRelaxType;
      params.pressureAMGAggNumLevels = mgr_strategy_config_cached.pressureAmgAggNumLevels;
      params.pressureAMGAggInterpType = mgr_strategy_config_cached.pressureAmgAggInterpType;
      params.pressureAMGAggPMaxElmts = mgr_strategy_config_cached.pressureAmgAggPMaxElmts;
      params.pressureAMGRelaxOrder = mgr_strategy_config_cached.pressureAmgRelaxOrder;
      mgr_solver.setParameters(params);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_pressure_amg_advanced_options(
        opendarts::config::mat_float strong_threshold,
        opendarts::config::mat_float trunc_factor,
        int pmax_elmts,
        int max_levels)
    {
      mgr_strategy_config_cached.pressureAmgStrongThreshold = strong_threshold;
      mgr_strategy_config_cached.pressureAmgTruncFactor = trunc_factor;
      mgr_strategy_config_cached.pressureAmgPMaxElmts = pmax_elmts;
      mgr_strategy_config_cached.pressureAmgMaxLevels = max_levels;
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.pressureAMGStrongThreshold =
          mgr_strategy_config_cached.pressureAmgStrongThreshold;
      params.pressureAMGTruncFactor = mgr_strategy_config_cached.pressureAmgTruncFactor;
      params.pressureAMGPMaxElmts = mgr_strategy_config_cached.pressureAmgPMaxElmts;
      params.pressureAMGMaxLevels = mgr_strategy_config_cached.pressureAmgMaxLevels;
      mgr_solver.setParameters(params);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_pressure_amg_solve_options(
        opendarts::config::index_t max_iter,
        opendarts::config::mat_float tolerance)
    {
      pressure_amg_max_iter_cached = std::max<opendarts::config::index_t>(max_iter, 1);
      pressure_amg_tolerance_cached = std::max<opendarts::config::mat_float>(tolerance, 0.0);
      mgr::SolverParameters params = mgr_solver.getParameters();
      params.pressureAMGMaxIter = pressure_amg_max_iter_cached;
      params.pressureAMGTolerance = pressure_amg_tolerance_cached;
      mgr_solver.setParameters(params);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_n_reservoir_blocks(opendarts::config::index_t n_reservoir_blocks)
    {
      n_reservoir_blocks_cached = n_reservoir_blocks;
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_enable_well_level(bool enable_well_level)
    {
      mgr_strategy_config_cached.enableWellLevel = enable_well_level;
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_enable_composition_level(bool enable_composition_level)
    {
      mgr_strategy_config_cached.enableCompositionLevel = enable_composition_level;
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    std::vector<mgr::strategies::VariableRole>
    linsolv_mgr<N_BLOCK_SIZE>::to_variable_roles(const std::vector<int> & variable_roles)
    {
      std::vector<mgr::strategies::VariableRole> roles;
      roles.reserve(variable_roles.size());
      for (const auto role : variable_roles)
      {
        roles.push_back(static_cast<mgr::strategies::VariableRole>(role));
      }
      return roles;
    }

    template <uint8_t N_BLOCK_SIZE>
    std::vector<int>
    linsolv_mgr<N_BLOCK_SIZE>::to_int_roles(const std::vector<mgr::strategies::VariableRole> & variable_roles)
    {
      std::vector<int> roles;
      roles.reserve(variable_roles.size());
      for (const auto role : variable_roles)
      {
        roles.push_back(static_cast<int>(role));
      }
      return roles;
    }

    template <uint8_t N_BLOCK_SIZE>
    std::vector<mgr::int_t>
    linsolv_mgr<N_BLOCK_SIZE>::to_labels(const std::vector<int> & labels)
    {
      std::vector<mgr::int_t> converted_labels;
      converted_labels.reserve(labels.size());
      for (const auto label : labels)
      {
        converted_labels.push_back(static_cast<mgr::int_t>(label));
      }
      return converted_labels;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_reservoir_variable_roles(const std::vector<int> & variable_roles)
    {
      mgr_strategy_config_cached.reservoirVariableRoles = to_variable_roles(variable_roles);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_well_variable_roles(const std::vector<int> & variable_roles)
    {
      mgr_strategy_config_cached.wellVariableRoles = to_variable_roles(variable_roles);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::clear_mgr_custom_levels()
    {
      mgr_strategy_config_cached.customLevels.clear();
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_num_custom_levels(int num_custom_levels)
    {
      mgr_strategy_config_cached.customLevels.resize(std::max(0, num_custom_levels));
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_custom_level_options(int custom_level_index,
                                                                 const std::vector<int> & keep_labels,
                                                                 int frelax_type,
                                                                 int frelax_iters,
                                                                 int interp_type,
                                                                 int restrict_type,
                                                                 int coarse_method,
                                                                 int smoother_type,
                                                                 int smoother_iters)
    {
      if (custom_level_index < 0)
      {
        std::cerr << "[MGR] Warning: ignoring custom level with negative index "
                  << custom_level_index << "." << std::endl;
        return;
      }

      const auto level_index = static_cast<size_t>(custom_level_index);
      if (mgr_strategy_config_cached.customLevels.size() <= level_index)
      {
        mgr_strategy_config_cached.customLevels.resize(level_index + 1);
      }

      auto & level = mgr_strategy_config_cached.customLevels[level_index];
      set_level_options(level,
                        frelax_type,
                        frelax_iters,
                        interp_type,
                        restrict_type,
                        coarse_method,
                        smoother_type,
                        smoother_iters);
      level.labels = to_labels(keep_labels);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_well_strategy(int well_strategy)
    {
      mgr_strategy_config_cached.wellStrategy =
          static_cast<mgr::strategies::WellStrategy>(well_strategy);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_well_frelax_type(int frelax_type)
    {
      mgr_strategy_config_cached.wellLevel.fRelaxType =
          static_cast<mgr::FRelaxationType>(frelax_type);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_well_frelax_iters(int frelax_iters)
    {
      mgr_strategy_config_cached.wellLevel.fRelaxIters = std::max(0, frelax_iters);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_level_options(mgr::MGRLevelParameters & level,
                                                      int frelax_type,
                                                      int frelax_iters,
                                                      int interp_type,
                                                      int restrict_type,
                                                      int coarse_method,
                                                      int smoother_type,
                                                      int smoother_iters)
    {
      level.fRelaxType = static_cast<mgr::FRelaxationType>(frelax_type);
      level.fRelaxIters = std::max(0, frelax_iters);
      level.interpType = static_cast<mgr::InterpolationType>(interp_type);
      level.restrictType = static_cast<mgr::RestrictionType>(restrict_type);
      level.coarseGridMethod = static_cast<mgr::CoarseGridMethod>(coarse_method);
      level.globalSmootherType = static_cast<mgr::GlobalSmootherType>(smoother_type);
      level.globalSmootherIters = std::max(0, smoother_iters);
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_well_level_options(int frelax_type,
                                                               int frelax_iters,
                                                               int interp_type,
                                                               int restrict_type,
                                                               int coarse_method,
                                                               int smoother_type,
                                                               int smoother_iters)
    {
      set_level_options(mgr_strategy_config_cached.wellLevel,
                        frelax_type,
                        frelax_iters,
                        interp_type,
                        restrict_type,
                        coarse_method,
                        smoother_type,
                        smoother_iters);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_composition_level_options(int frelax_type,
                                                                      int frelax_iters,
                                                                      int interp_type,
                                                                      int restrict_type,
                                                                      int coarse_method,
                                                                      int smoother_type,
                                                                      int smoother_iters)
    {
      set_level_options(mgr_strategy_config_cached.compositionLevel,
                        frelax_type,
                        frelax_iters,
                        interp_type,
                        restrict_type,
                        coarse_method,
                        smoother_type,
                        smoother_iters);
      first_solve = true;
    }

    template <uint8_t N_BLOCK_SIZE>
    void linsolv_mgr<N_BLOCK_SIZE>::set_mgr_pressure_level_options(int frelax_type,
                                                                   int frelax_iters,
                                                                   int interp_type,
                                                                   int restrict_type,
                                                                   int coarse_method,
                                                                   int smoother_type,
                                                                   int smoother_iters)
    {
      set_level_options(mgr_strategy_config_cached.pressureLevel,
                        frelax_type,
                        frelax_iters,
                        interp_type,
                        restrict_type,
                        coarse_method,
                        smoother_type,
                        smoother_iters);
      first_solve = true;
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
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_scaling_type() const
    {
      return scaling_type_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_use_flex_gmres() const
    {
      return use_flex_gmres_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_composite_mode() const
    {
      return composite_mode_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_local_solver() const
    {
      return local_solver_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_mgr_bilu0_pivot_shift() const
    {
      return bilu0_pivot_shift_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_bilu0_fallback_strategy() const
    {
      return bilu0_fallback_strategy_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_mgr_bilu0_fallback_diagonal_tolerance() const
    {
      return bilu0_fallback_diagonal_tolerance_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_mgr_bilu0_fallback_shifted_max() const
    {
      return bilu0_fallback_shifted_max_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_mgr_bilu0_fallback_shifted_growth() const
    {
      return bilu0_fallback_shifted_growth_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_mgr_local_correction_alpha() const
    {
      return local_correction_alpha_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_mgr_local_correction_adaptive_fallback_threshold() const
    {
      return local_correction_adaptive_fallback_threshold_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_mgr_local_correction_adaptive_alpha() const
    {
      return local_correction_adaptive_alpha_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float
    linsolv_mgr<N_BLOCK_SIZE>::get_mgr_local_correction_adaptive_fallback_threshold_high() const
    {
      return local_correction_adaptive_fallback_threshold_high_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float
    linsolv_mgr<N_BLOCK_SIZE>::get_mgr_local_correction_adaptive_alpha_high() const
    {
      return local_correction_adaptive_alpha_high_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_use_bcsr_cpr() const
    {
      return use_bcsr_cpr_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_reduction_type() const
    {
      return bcsr_cpr_reduction_type_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_pressure_variable() const
    {
      return bcsr_cpr_pressure_variable_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_weight_max() const
    {
      return bcsr_cpr_weight_max_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_reuse_amg_hierarchy() const
    {
      return bcsr_cpr_reuse_amg_hierarchy_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::index_t linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_amg_rebuild_interval() const
    {
      return bcsr_cpr_amg_rebuild_interval_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_adaptive_amg_rebuild() const
    {
      return bcsr_cpr_adaptive_amg_rebuild_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::index_t linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_adaptive_li_threshold() const
    {
      return bcsr_cpr_adaptive_li_threshold_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::mat_float linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_adaptive_li_growth_factor() const
    {
      return bcsr_cpr_adaptive_li_growth_factor_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::index_t linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_adaptive_min_reuse_setups() const
    {
      return bcsr_cpr_adaptive_min_reuse_setups_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::index_t linsolv_mgr<N_BLOCK_SIZE>::get_bcsr_cpr_adaptive_max_reuse_setups() const
    {
      return bcsr_cpr_adaptive_max_reuse_setups_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    opendarts::config::index_t linsolv_mgr<N_BLOCK_SIZE>::get_n_reservoir_blocks() const
    {
      return n_reservoir_blocks_cached;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_mgr_enable_well_level() const
    {
      return mgr_strategy_config_cached.enableWellLevel;
    }

    template <uint8_t N_BLOCK_SIZE>
    bool linsolv_mgr<N_BLOCK_SIZE>::get_mgr_enable_composition_level() const
    {
      return mgr_strategy_config_cached.enableCompositionLevel;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_num_custom_levels() const
    {
      return static_cast<int>(mgr_strategy_config_cached.customLevels.size());
    }

    template <uint8_t N_BLOCK_SIZE>
    std::vector<int> linsolv_mgr<N_BLOCK_SIZE>::get_mgr_reservoir_variable_roles() const
    {
      return to_int_roles(mgr_strategy_config_cached.reservoirVariableRoles);
    }

    template <uint8_t N_BLOCK_SIZE>
    std::vector<int> linsolv_mgr<N_BLOCK_SIZE>::get_mgr_well_variable_roles() const
    {
      return to_int_roles(mgr_strategy_config_cached.wellVariableRoles);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_well_strategy() const
    {
      return static_cast<int>(mgr_strategy_config_cached.wellStrategy);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_well_frelax_type() const
    {
      return static_cast<int>(mgr_strategy_config_cached.wellLevel.fRelaxType);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::get_mgr_well_frelax_iters() const
    {
      return mgr_strategy_config_cached.wellLevel.fRelaxIters;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A,
                                        opendarts::config::index_t max_iters,
                                        opendarts::config::mat_float tolerance)
    {
      return init(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A),
                  max_iters,
                  tolerance);
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix_base *A,
                                        opendarts::config::index_t max_iters,
                                        opendarts::config::mat_float tolerance)
    {
      // Store parameters and matrix pointer (like SuperLU)
      matrix_ptr = A;
      max_iters_cached = max_iters;
      tolerance_cached = tolerance;

      if (A == nullptr)
      {
        std::cerr << "[MGR] Error: Matrix pointer is null." << std::endl;
        return -1;
      }
      if (A->n_row_size != N_BLOCK_SIZE)
      {
        std::cerr << "[MGR] Error: Matrix block size (" << A->n_row_size
                  << ") does not match solver block size ("
                  << static_cast<int>(N_BLOCK_SIZE) << ")." << std::endl;
        return -1;
      }

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
      params.scalingType = static_cast<mgr::ScalingType>(scaling_type_cached);
      params.compositeMode = static_cast<mgr::CompositePreconditionerMode>(composite_mode_cached);
      params.localPreconditioner = static_cast<mgr::LocalPreconditionerType>(local_solver_cached);
      params.localPivotShift = bilu0_pivot_shift_cached;
      params.localFallbackStrategy = static_cast<mgr::LocalFallbackStrategy>(bilu0_fallback_strategy_cached);
      params.localFallbackDiagonalTolerance = bilu0_fallback_diagonal_tolerance_cached;
      params.localFallbackShiftMax = bilu0_fallback_shifted_max_cached;
      params.localFallbackShiftGrowth = bilu0_fallback_shifted_growth_cached;
      params.localCorrectionAlpha = local_correction_alpha_cached;
      params.localCorrectionAdaptiveFallbackThreshold =
          local_correction_adaptive_fallback_threshold_cached;
      params.localCorrectionAdaptiveAlpha = local_correction_adaptive_alpha_cached;
      params.localCorrectionAdaptiveFallbackThresholdHigh =
          local_correction_adaptive_fallback_threshold_high_cached;
      params.localCorrectionAdaptiveAlphaHigh = local_correction_adaptive_alpha_high_cached;
      params.localCorrectionQualityGate =
          local_correction_quality_gate_cached;
      params.localCorrectionQualityMinAlpha =
          local_correction_quality_min_alpha_cached;
      params.useBCSRCPR = use_bcsr_cpr_cached;
      params.bcsrCPRReduction =
          static_cast<mgr::BCSRCPRReductionType>(bcsr_cpr_reduction_type_cached);
      params.bcsrCPRPressureVariable = bcsr_cpr_pressure_variable_cached;
      params.bcsrCPRWeightMax = bcsr_cpr_weight_max_cached;
      params.bcsrCPRReuseAMGHierarchy = bcsr_cpr_reuse_amg_hierarchy_cached;
      params.bcsrCPRAMGRebuildInterval = bcsr_cpr_amg_rebuild_interval_cached;
      params.bcsrCPRAdaptiveAMGRebuild = bcsr_cpr_adaptive_amg_rebuild_cached;
      params.bcsrCPRAdaptiveLIThreshold = bcsr_cpr_adaptive_li_threshold_cached;
      params.bcsrCPRAdaptiveLIGrowthFactor =
          bcsr_cpr_adaptive_li_growth_factor_cached;
      params.bcsrCPRAdaptiveMinReuseSetups =
          bcsr_cpr_adaptive_min_reuse_setups_cached;
      params.bcsrCPRAdaptiveMaxReuseSetups =
          bcsr_cpr_adaptive_max_reuse_setups_cached;
      params.bcsrCPRAdaptivePressureOvershootThreshold =
          bcsr_cpr_adaptive_pressure_overshoot_threshold_cached;
      params.bcsrCPRAdaptiveFinalProxyThreshold =
          bcsr_cpr_adaptive_final_proxy_threshold_cached;
      params.bcsrCPRAdaptiveFallbackThreshold =
          bcsr_cpr_adaptive_fallback_threshold_cached;
      params.bcsrCPRDiagnostics = bcsr_cpr_diagnostics_cached;
      params.bcsrCPRDiagnosticApplyInterval =
          bcsr_cpr_diagnostic_apply_interval_cached;
      params.bcsrCPRDiagnosticMatrixInterval =
          bcsr_cpr_diagnostic_matrix_interval_cached;
      params.bcsrCPRPressureCorrectionAlpha =
          bcsr_cpr_pressure_correction_alpha_cached;
      params.bcsrCPRPressureCorrectionGuardThreshold =
          bcsr_cpr_pressure_correction_guard_threshold_cached;
      params.bcsrCPRPressureCorrectionGuardMinAlpha =
          bcsr_cpr_pressure_correction_guard_min_alpha_cached;
      params.pressureAMGMaxIter = pressure_amg_max_iter_cached;
      params.pressureAMGTolerance = pressure_amg_tolerance_cached;
      params.pressureAMGCoarsenType = mgr_strategy_config_cached.pressureAmgCoarsenType;
      params.pressureAMGInterpType = mgr_strategy_config_cached.pressureAmgInterpType;
      params.pressureAMGRelaxType = mgr_strategy_config_cached.pressureAmgRelaxType;
      params.pressureAMGAggNumLevels = mgr_strategy_config_cached.pressureAmgAggNumLevels;
      params.pressureAMGAggInterpType = mgr_strategy_config_cached.pressureAmgAggInterpType;
      params.pressureAMGAggPMaxElmts = mgr_strategy_config_cached.pressureAmgAggPMaxElmts;
      params.pressureAMGRelaxOrder = mgr_strategy_config_cached.pressureAmgRelaxOrder;
      params.pressureAMGStrongThreshold =
          mgr_strategy_config_cached.pressureAmgStrongThreshold;
      params.pressureAMGTruncFactor = mgr_strategy_config_cached.pressureAmgTruncFactor;
      params.pressureAMGPMaxElmts = mgr_strategy_config_cached.pressureAmgPMaxElmts;
      params.pressureAMGMaxLevels = mgr_strategy_config_cached.pressureAmgMaxLevels;
      params.localReservoirBlockCount = n_reservoir_blocks_cached;
      params.krylovType = use_flex_gmres_cached ? mgr::KrylovType::flexgmres
                                               : mgr::KrylovType::gmres;
      mgr_solver.setParameters(params);
      mgr_solver.init_timer_nodes(this->timer_setup, this->timer_solve);

      initialized = true;
      first_solve = true;
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A)
    {
      return setup(static_cast<opendarts::linear_solvers::csr_matrix_base *>(A));
    }

    template <uint8_t N_BLOCK_SIZE>
    int linsolv_mgr<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix_base *A)
    {
      if (!initialized)
      {
        std::cerr << "[MGR] Error: Solver not initialized. Call init() first." << std::endl;
        return -1;
      }

      if (A == nullptr)
      {
        std::cerr << "[MGR] Error: Matrix pointer is null." << std::endl;
        return -1;
      }
      if (A->n_row_size != N_BLOCK_SIZE)
      {
        std::cerr << "[MGR] Error: Matrix block size (" << A->n_row_size
                  << ") does not match solver block size ("
                  << static_cast<int>(N_BLOCK_SIZE) << ")." << std::endl;
        return -1;
      }

      matrix_ptr = A;
      mgr_solver.init_timer_nodes(this->timer_setup, this->timer_solve);
      ::timer_node *mgr_timer = this->timer_setup ? &this->timer_setup->node["MGR"] : nullptr;
      ScopedTimer mgr_total(mgr_timer);

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

      bool matrix_imported = false;
      {
        ScopedTimer timer(mgr_timer ? &mgr_timer->node["import CSR"] : nullptr);
        matrix_imported = mgr_solver.setMatrixFromCSR(n_blocks, n_blocks, block_size, nnz_blocks_declared,
                                                      row_ptr,
                                                      col_ind,
                                                      values,
                                                      diag_ind);
      }
      if (!matrix_imported)
      {
        std::cerr << "[MGR] Error: Failed to set matrix from CSR data" << std::endl;
        return -1;
      }

      opendarts::config::index_t n_reservoir_blocks = n_blocks;
      if (n_reservoir_blocks_cached > 0)
      {
        if (n_reservoir_blocks_cached <= n_blocks)
        {
          n_reservoir_blocks = n_reservoir_blocks_cached;
        }
        else
        {
          std::cerr << "[MGR] Warning: configured reservoir block count ("
                    << n_reservoir_blocks_cached
                    << ") exceeds matrix block count (" << n_blocks
                    << "); treating all blocks as reservoir blocks." << std::endl;
        }
      }

      mgr::SolverParameters params = mgr_solver.getParameters();
      params.localReservoirBlockCount = n_reservoir_blocks;
      mgr_solver.setParameters(params);

      if (first_solve)
      {
        ScopedTimer timer(mgr_timer ? &mgr_timer->node["strategy setup"] : nullptr);
        auto strategy = std::make_unique<mgr::strategies::CompositionalFlowStrategy>(
            block_size,
            n_blocks * block_size,
            n_blocks,
            n_reservoir_blocks,
            mgr_strategy_config_cached);
        strategy->setup();
        mgr_solver.setStrategy(std::move(strategy));
      }

      mgr::SolverParameters setup_params = mgr_solver.getParameters();
      if (mgr_solver.setup(setup_params.maxIter, setup_params.tolerance) != 0)
      {
        std::cerr << "[MGR] Error: Failed to setup solver" << std::endl;
        return -1;
      }

      first_solve = false;
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

      if (first_solve)
      {
        if (setup(matrix_ptr) != 0)
        {
          return -1;
        }
      }

      if (log_level_cached >= 2)
      {
        std::cout << "[MGR] Solving with tolerance=" << tolerance_cached
                  << ", max_iter=" << max_iters_cached << std::endl;
      }

      // Solve
      mgr_solver.init_timer_nodes(this->timer_setup, this->timer_solve);
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
    int linsolv_mgr<N_BLOCK_SIZE>::solve_transposed(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
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

      if (matrix_ptr->n_row_size != N_BLOCK_SIZE)
      {
        std::cerr << "[MGR] Error: Matrix block size (" << matrix_ptr->n_row_size
                  << ") does not match solver block size ("
                  << static_cast<int>(N_BLOCK_SIZE) << ")." << std::endl;
        return -1;
      }

      build_transpose_from_base<N_BLOCK_SIZE>(matrix_ptr, transpose_matrix);

      csr_matrix_base *original_matrix = matrix_ptr;
      const bool original_first_solve = first_solve;

      matrix_ptr = &transpose_matrix;
      first_solve = true;
      const int setup_rc = setup(&transpose_matrix);
      if (setup_rc != 0)
      {
        matrix_ptr = original_matrix;
        first_solve = original_first_solve;
        return setup_rc;
      }

      const int solve_rc = solve(B, X);

      matrix_ptr = original_matrix;
      first_solve = true;
      return solve_rc;
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
