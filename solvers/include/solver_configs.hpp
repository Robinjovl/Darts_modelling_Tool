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

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_SOLVER_CONFIGS_HPP
#define OPENDARTS_LINEAR_SOLVERS_SOLVER_CONFIGS_HPP
//--------------------------------------------------------------------------

// Per-solver configuration structs. Each derives from solver_config and is the
// C++ counterpart of a Python LinearSolverSpec class. The registered factory
// for a solver down-casts solver_config to its own type here.
//
// Solvers with no parameters beyond max_iterations / tolerance / print_level
// (e.g. SuperLU) just use the base solver_config directly -- no struct here.

#include <optional>
#include <vector>

#include "data_types.hpp"
#include "solver_config.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Configuration of one HYPRE MGR reduction level.
     *
     *  The integer codes match HYPRE's MGR API -- see the FRelaxationType,
     *  InterpolationType, RestrictionType, CoarseGridMethod and
     *  GlobalSmootherType enums in MGRStrategy.hpp. The Python-side
     *  MGRLevelSpec mirrors this struct field-for-field.
     */
    struct mgr_level_config
    {
      std::vector<int> keep_labels;  // DOF labels kept on this level (custom levels only)
      int frelax_type = -1;          // F-relaxation type    (-1 = none)
      int frelax_iters = 0;          // F-relaxation sweeps
      int interp_type = 0;           // interpolation type
      int restrict_type = 0;         // restriction type
      int coarse_method = 0;         // coarse-grid method
      int smoother_type = -1;        // global smoother type (-1 = none)
      int smoother_iters = 0;        // global smoother iterations
    };

    /** Configuration for the HYPRE MGR solver (linsolv_mgr).
     *
     *  Inherits max_iterations / tolerance / print_level from solver_config.
     *  The scalar defaults below match linsolv_mgr's own internal defaults, so
     *  a default-constructed mgr_solver_config reproduces the out-of-the-box
     *  MGR behaviour. The optional level members are applied by the factory
     *  only when they are set, leaving linsolv_mgr's built-in level defaults
     *  in place otherwise.
     *
     *  C++ counterpart of the Python MGRSolverSpec.
     */
    struct mgr_solver_config : opendarts::linear_solvers::solver_config
    {
      int kdim = 30;                                      // Krylov subspace dimension
      bool use_mgr = true;                                // MGR preconditioner (false = BoomerAMG)
      int log_level = 1;                                  // 0 silent / 1 basic / 2 detailed
      bool use_physics_scaling = true;                    // physics-based matrix scaling
      bool use_flex_gmres = true;                         // FlexGMRES (false = plain GMRES)
      opendarts::config::index_t n_reservoir_blocks = 0;  // 0 = inferred from the matrix

      bool enable_well_level = false;                     // dedicated well-elimination level
      bool enable_composition_level = false;              // dedicated composition-reduction level

      std::vector<int> reservoir_variable_roles;          // per-reservoir-DOF physical roles
      std::vector<int> well_variable_roles;               // per-well-DOF physical roles

      std::optional<int> well_strategy;                   // MGR well strategy
      std::optional<mgr_level_config> well_level;         // overrides the well level
      std::optional<mgr_level_config> composition_level;  // overrides the composition level
      std::optional<mgr_level_config> pressure_level;     // overrides the pressure level
      std::vector<mgr_level_config> custom_levels;        // user-defined extra reduction levels
    };

    /** Configuration for the open-source GMRES outer Krylov solver (linsolv_gmres).
     *
     *  Inherits max_iterations / tolerance / print_level from solver_config; the
     *  preconditioner is composed on the Python side (see GMRESSolverSpec).
     *  C++ counterpart of the Python GMRESSolverSpec.
     */
    struct gmres_solver_config : opendarts::linear_solvers::solver_config
    {
      int restart = 30;  // Krylov subspace dimension (restart length)
    };

    /** Configuration for the open-source CPR two-stage preconditioner (linsolv_cpr).
     *
     *  CPR is the in-tree replacement for the proprietary ``linsolv_bos_cpr``:
     *  pressure-subsystem AMG correction followed by full-system ILU(0). It is
     *  intended as the inner preconditioner of an outer Krylov solver (typically
     *  ``GMRESSolverSpec``). The transposed apply (CPRA, Han et al. 2013) is
     *  used by the adjoint Newton step.
     */
    struct cpr_solver_config : opendarts::linear_solvers::solver_config
    {
      int amg_max_iters = 2;          // AMG sweeps on the pressure subsystem per CPR apply
      int ilu_fill_level = 0;         // full-system ILU(k)
      // No amg_tolerance: BoomerAMG is configured with tol=0 since it is used
      // as a preconditioner stage of CPR; convergence is driven by the outer
      // Krylov, the AMG sweep budget is set by amg_max_iters.
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_SOLVER_CONFIGS_HPP
//--------------------------------------------------------------------------
