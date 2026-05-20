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

#include "solver_factories.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "linsolv_cpr.hpp"
#include "linsolv_gmres.hpp"
#include "linsolv_iface.hpp"
#include "linsolv_mgr.hpp"
#include "linsolv_superlu.hpp"
#include "solver_config.hpp"
#include "solver_configs.hpp"
#include "solver_registry.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    namespace
    {
      // open-DARTS instantiates the block-templated solvers for N = 1..13.
      constexpr int MIN_BLOCK_SIZE = 1;
      constexpr int MAX_BLOCK_SIZE = 13;

      using solver_handle = std::shared_ptr<opendarts::linear_solvers::linsolv_iface>;

      // ---- MGR (HYPRE Multigrid Reduction) ---------------------------------

      // Construct and configure an MGR solver for a compile-time block size.
      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_mgr(const opendarts::linear_solvers::mgr_solver_config &config)
      {
        auto solver = std::make_shared<opendarts::linear_solvers::linsolv_mgr<N_BLOCK_SIZE>>();

        // Scalar parameters: the config defaults match linsolv_mgr's own
        // defaults, so applying them unconditionally is safe.
        solver->set_max_iterations(config.max_iterations);
        solver->set_tolerance(config.tolerance);
        solver->set_kdim(config.kdim);
        solver->set_use_mgr(config.use_mgr);
        solver->set_log_level(config.log_level);
        solver->set_use_physics_scaling(config.use_physics_scaling);
        solver->set_use_flex_gmres(config.use_flex_gmres);
        solver->set_n_reservoir_blocks(config.n_reservoir_blocks);
        solver->set_mgr_enable_well_level(config.enable_well_level);
        solver->set_mgr_enable_composition_level(config.enable_composition_level);

        if (!config.reservoir_variable_roles.empty())
          solver->set_mgr_reservoir_variable_roles(config.reservoir_variable_roles);
        if (!config.well_variable_roles.empty())
          solver->set_mgr_well_variable_roles(config.well_variable_roles);
        if (config.well_strategy.has_value())
          solver->set_mgr_well_strategy(config.well_strategy.value());

        // Level overrides: applied only when set, so linsolv_mgr's built-in
        // level defaults are preserved otherwise.
        if (config.well_level.has_value())
        {
          const opendarts::linear_solvers::mgr_level_config &level = config.well_level.value();
          solver->set_mgr_well_level_options(level.frelax_type, level.frelax_iters,
              level.interp_type, level.restrict_type, level.coarse_method,
              level.smoother_type, level.smoother_iters);
        }
        if (config.composition_level.has_value())
        {
          const opendarts::linear_solvers::mgr_level_config &level =
              config.composition_level.value();
          solver->set_mgr_composition_level_options(level.frelax_type, level.frelax_iters,
              level.interp_type, level.restrict_type, level.coarse_method,
              level.smoother_type, level.smoother_iters);
        }
        if (config.pressure_level.has_value())
        {
          const opendarts::linear_solvers::mgr_level_config &level =
              config.pressure_level.value();
          solver->set_mgr_pressure_level_options(level.frelax_type, level.frelax_iters,
              level.interp_type, level.restrict_type, level.coarse_method,
              level.smoother_type, level.smoother_iters);
        }

        if (!config.custom_levels.empty())
        {
          solver->set_mgr_num_custom_levels(static_cast<int>(config.custom_levels.size()));
          for (std::size_t level_index = 0; level_index < config.custom_levels.size();
               ++level_index)
          {
            const opendarts::linear_solvers::mgr_level_config &level =
                config.custom_levels[level_index];
            solver->set_mgr_custom_level_options(static_cast<int>(level_index),
                level.keep_labels, level.frelax_type, level.frelax_iters, level.interp_type,
                level.restrict_type, level.coarse_method, level.smoother_type,
                level.smoother_iters);
          }
        }

        return solver;  // shared_ptr<linsolv_mgr<N>> -> shared_ptr<linsolv_iface>
      }

      // Runtime block-size dispatch for MGR.
      solver_handle build_mgr_for_block_size(int block_size,
          const opendarts::linear_solvers::mgr_solver_config &config)
      {
        switch (block_size)
        {
          case 1:  return build_mgr<1>(config);
          case 2:  return build_mgr<2>(config);
          case 3:  return build_mgr<3>(config);
          case 4:  return build_mgr<4>(config);
          case 5:  return build_mgr<5>(config);
          case 6:  return build_mgr<6>(config);
          case 7:  return build_mgr<7>(config);
          case 8:  return build_mgr<8>(config);
          case 9:  return build_mgr<9>(config);
          case 10: return build_mgr<10>(config);
          case 11: return build_mgr<11>(config);
          case 12: return build_mgr<12>(config);
          case 13: return build_mgr<13>(config);
          default:
            throw std::runtime_error("MGR solver: unsupported block size " +
                std::to_string(block_size) + " (supported: " +
                std::to_string(MIN_BLOCK_SIZE) + ".." + std::to_string(MAX_BLOCK_SIZE) + ").");
        }
      }

      // Factory registered under the name "mgr".
      solver_handle make_mgr_solver(const opendarts::linear_solvers::solver_config &config,
          int block_size)
      {
        // Use the MGR-specific configuration if one was supplied; a plain
        // solver_config falls back to the MGR defaults.
        const opendarts::linear_solvers::mgr_solver_config default_config;
        const opendarts::linear_solvers::mgr_solver_config *mgr_config =
            dynamic_cast<const opendarts::linear_solvers::mgr_solver_config *>(&config);
        if (mgr_config == nullptr)
          mgr_config = &default_config;

        return build_mgr_for_block_size(block_size, *mgr_config);
      }

      // ---- SuperLU (direct solver) -----------------------------------------

      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_superlu()
      {
        return std::make_shared<opendarts::linear_solvers::linsolv_superlu<N_BLOCK_SIZE>>();
      }

      solver_handle build_superlu_for_block_size(int block_size)
      {
        switch (block_size)
        {
          case 1:  return build_superlu<1>();
          case 2:  return build_superlu<2>();
          case 3:  return build_superlu<3>();
          case 4:  return build_superlu<4>();
          case 5:  return build_superlu<5>();
          case 6:  return build_superlu<6>();
          case 7:  return build_superlu<7>();
          case 8:  return build_superlu<8>();
          case 9:  return build_superlu<9>();
          case 10: return build_superlu<10>();
          case 11: return build_superlu<11>();
          case 12: return build_superlu<12>();
          case 13: return build_superlu<13>();
          default:
            throw std::runtime_error("SuperLU solver: unsupported block size " +
                std::to_string(block_size) + " (supported: " +
                std::to_string(MIN_BLOCK_SIZE) + ".." + std::to_string(MAX_BLOCK_SIZE) + ").");
        }
      }

      // Factory registered under the name "superlu". SuperLU is a direct solver
      // and takes no parameters beyond the matrix, so the configuration is unused.
      solver_handle make_superlu_solver(
          const opendarts::linear_solvers::solver_config & /*config*/, int block_size)
      {
        return build_superlu_for_block_size(block_size);
      }
      // ---- GMRES (open-source restarted Krylov, see linsolv_gmres) ---------

      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_gmres(const opendarts::linear_solvers::gmres_solver_config &config)
      {
        auto solver = std::make_shared<opendarts::linear_solvers::linsolv_gmres<N_BLOCK_SIZE>>();
        solver->set_restart(config.restart);
        // tolerance / max_iterations are honoured through init(); the engine
        // calls linear_solver->init(matrix, max_iters, tol) which forwards
        // them. We store nothing else here.
        return solver;
      }

      solver_handle build_gmres_for_block_size(int block_size,
          const opendarts::linear_solvers::gmres_solver_config &config)
      {
        switch (block_size)
        {
          case 1:  return build_gmres<1>(config);
          case 2:  return build_gmres<2>(config);
          case 3:  return build_gmres<3>(config);
          case 4:  return build_gmres<4>(config);
          case 5:  return build_gmres<5>(config);
          case 6:  return build_gmres<6>(config);
          case 7:  return build_gmres<7>(config);
          case 8:  return build_gmres<8>(config);
          case 9:  return build_gmres<9>(config);
          case 10: return build_gmres<10>(config);
          case 11: return build_gmres<11>(config);
          case 12: return build_gmres<12>(config);
          case 13: return build_gmres<13>(config);
          default:
            throw std::runtime_error("GMRES solver: unsupported block size " +
                std::to_string(block_size) + " (supported: " +
                std::to_string(MIN_BLOCK_SIZE) + ".." + std::to_string(MAX_BLOCK_SIZE) + ").");
        }
      }

      // Factory registered under the name "gmres".
      solver_handle make_gmres_solver(
          const opendarts::linear_solvers::solver_config &config, int block_size)
      {
        const opendarts::linear_solvers::gmres_solver_config default_config;
        const opendarts::linear_solvers::gmres_solver_config *gmres_config =
            dynamic_cast<const opendarts::linear_solvers::gmres_solver_config *>(&config);
        if (gmres_config == nullptr)
          gmres_config = &default_config;
        return build_gmres_for_block_size(block_size, *gmres_config);
      }

      // ---- CPR (open-source two-stage CPR preconditioner) ------------------

      template <uint8_t N_BLOCK_SIZE>
      solver_handle build_cpr(const opendarts::linear_solvers::cpr_solver_config &config)
      {
        auto solver = std::make_shared<opendarts::linear_solvers::linsolv_cpr<N_BLOCK_SIZE>>();
        solver->set_amg_max_iters(config.amg_max_iters);
        solver->set_amg_tolerance(config.amg_tolerance);
        solver->set_ilu_fill_level(config.ilu_fill_level);
        return solver;
      }

      solver_handle build_cpr_for_block_size(int block_size,
          const opendarts::linear_solvers::cpr_solver_config &config)
      {
        switch (block_size)
        {
          case 1:  return build_cpr<1>(config);
          case 2:  return build_cpr<2>(config);
          case 3:  return build_cpr<3>(config);
          case 4:  return build_cpr<4>(config);
          case 5:  return build_cpr<5>(config);
          case 6:  return build_cpr<6>(config);
          case 7:  return build_cpr<7>(config);
          case 8:  return build_cpr<8>(config);
          case 9:  return build_cpr<9>(config);
          case 10: return build_cpr<10>(config);
          case 11: return build_cpr<11>(config);
          case 12: return build_cpr<12>(config);
          case 13: return build_cpr<13>(config);
          default:
            throw std::runtime_error("CPR solver: unsupported block size " +
                std::to_string(block_size));
        }
      }

      // Factory registered under the name "cpr".
      solver_handle make_cpr_solver(
          const opendarts::linear_solvers::solver_config &config, int block_size)
      {
        const opendarts::linear_solvers::cpr_solver_config default_config;
        const opendarts::linear_solvers::cpr_solver_config *cpr_config =
            dynamic_cast<const opendarts::linear_solvers::cpr_solver_config *>(&config);
        if (cpr_config == nullptr)
          cpr_config = &default_config;
        return build_cpr_for_block_size(block_size, *cpr_config);
      }
    } // anonymous namespace

    void register_builtin_solvers()
    {
      // register_solver() returns false if the name is already registered,
      // which makes repeated calls to register_builtin_solvers() harmless.
      register_solver("mgr", make_mgr_solver);
      register_solver("superlu", make_superlu_solver);
      register_solver("gmres", make_gmres_solver);
      register_solver("cpr", make_cpr_solver);
    }
  } // namespace linear_solvers
} // namespace opendarts
