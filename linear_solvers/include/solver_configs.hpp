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

    /** Block-ILU(0) local-solver options for the MGR F-relaxation stage.
     *
     *  Mirrors linsolv_mgr::set_mgr_bilu0_pivot_shift /
     *  set_mgr_bilu0_fallback_options. Defaults match linsolv_mgr's
     *  constructor (pivot 1e-12, identity fallback).
     */
    struct mgr_bilu0_config
    {
      opendarts::config::mat_float pivot_shift = 1.0e-12;
      int fallback_strategy = 0;                                  // mgr::LocalFallbackStrategy::identity
      opendarts::config::mat_float fallback_diagonal_tolerance = 1.0e-4;
      opendarts::config::mat_float fallback_shifted_max = 1.0e-4;
      opendarts::config::mat_float fallback_shifted_growth = 100.0;
    };

    /** MGR local-correction (pressure-block damping) options.
     *
     *  Mirrors linsolv_mgr::set_mgr_local_correction_options /
     *  set_mgr_local_correction_quality_options. Defaults match the
     *  linsolv_mgr constructor.
     */
    struct mgr_local_correction_config
    {
      opendarts::config::mat_float alpha = 1.0;
      opendarts::config::mat_float adaptive_fallback_threshold = -1.0;
      opendarts::config::mat_float adaptive_alpha = 0.0;
      opendarts::config::mat_float adaptive_fallback_threshold_high = -1.0;
      opendarts::config::mat_float adaptive_alpha_high = 0.0;
      bool quality_enabled = false;
      opendarts::config::mat_float quality_min_alpha = 0.0;
    };

    /** BCSR-CPR (block-CSR Constrained Pressure Residual) options.
     *
     *  Mirrors the linsolv_mgr set_bcsr_cpr_* setter family. Presence of this
     *  struct on mgr_solver_config enables BCSR-CPR (set_use_bcsr_cpr(true)).
     *  Scalar defaults match the linsolv_mgr constructor. transpose_apply /
     *  forward_source are optional: the factory calls their setters only when
     *  set, so the linsolv_mgr auto-derivation is preserved when unset.
     */
    struct mgr_bcsr_cpr_config
    {
      int reduction_type = 1;                                     // mgr::BCSRCPRReductionType::trueIMPES
      int pressure_variable = 0;
      opendarts::config::mat_float weight_max = 1.0e6;
      bool reuse_amg_hierarchy = false;
      opendarts::config::index_t amg_rebuild_interval = 1;
      bool adaptive_amg_rebuild = false;
      opendarts::config::index_t adaptive_li_threshold = 80;
      opendarts::config::mat_float adaptive_li_growth_factor = 2.0;
      opendarts::config::index_t adaptive_min_reuse_setups = 1;
      opendarts::config::index_t adaptive_max_reuse_setups = 0;
      opendarts::config::mat_float adaptive_pressure_overshoot_threshold = -1.0;
      opendarts::config::mat_float adaptive_final_proxy_threshold = -1.0;
      opendarts::config::mat_float adaptive_fallback_threshold = -1.0;
      bool diagnostics = false;
      opendarts::config::index_t diagnostic_apply_interval = 0;
      opendarts::config::index_t diagnostic_matrix_interval = 0;
      opendarts::config::mat_float pressure_correction_alpha = 1.0;
      opendarts::config::mat_float pressure_correction_guard_threshold = -1.0;
      opendarts::config::mat_float pressure_correction_guard_min_alpha = 0.0;
      std::optional<bool> transpose_apply;                        // unset = keep linsolv_mgr auto-derivation
      std::optional<bool> forward_source;                         // unset = keep linsolv_mgr auto-derivation
    };

    /** MGR pressure-subsystem BoomerAMG options.
     *
     *  Mirrors set_mgr_pressure_amg_options / _advanced_options / _solve_options.
     *  Defaults match the typical HYPRE settings used by the composite models;
     *  applied only when this struct is set on mgr_solver_config.
     */
    struct mgr_pressure_amg_config
    {
      // set_mgr_pressure_amg_options
      int coarsen_type = 6;
      int interp_type = 6;
      int relax_type = 6;
      int agg_num_levels = 1;
      int agg_interp_type = 6;
      int agg_pmax_elmts = 20;
      int relax_order = 1;
      // set_mgr_pressure_amg_advanced_options
      opendarts::config::mat_float strong_threshold = 0.5;
      opendarts::config::mat_float trunc_factor = -1.0;
      int pmax_elmts = -1;
      int max_levels = 0;
      // set_mgr_pressure_amg_solve_options
      opendarts::config::index_t solve_max_iter = 1;
      opendarts::config::mat_float solve_tolerance = 0.0;
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

      // Composite-preconditioner / local-solver knobs (applied only when set, so
      // a default mgr_solver_config keeps linsolv_mgr's out-of-the-box behaviour).
      std::optional<int> scaling_type;                          // set_mgr_scaling_type
      std::optional<int> composite_mode;                        // set_mgr_composite_mode
      std::optional<int> local_solver;                          // set_mgr_local_solver
      std::optional<bool> use_bcsr_cpr;                         // set_use_bcsr_cpr (standalone toggle)
      std::optional<mgr_bilu0_config> bilu0;                    // block-ILU0 local solver options
      std::optional<mgr_local_correction_config> local_correction;  // pressure-block damping
      std::optional<mgr_bcsr_cpr_config> bcsr_cpr;              // BCSR-CPR options (presence enables it)
      std::optional<mgr_pressure_amg_config> pressure_amg;      // pressure-subsystem BoomerAMG options
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
      int amg_max_iters = 1;          // AMG V-cycles on the pressure subsystem per CPR apply
      int ilu_fill_level = 0;         // full-system ILU(k) (stage2_type == 0)
      // No amg_tolerance: BoomerAMG is configured with tol=0 since it is used
      // as a preconditioner stage of CPR; convergence is driven by the outer
      // Krylov, the AMG sweep budget is set by amg_max_iters.

      // Pressure-stage BoomerAMG configuration. Negative values leave the
      // HYPRE built-in default untouched. The defaults below mirror the
      // proprietary BOS AMG as configured by the legacy cpu_gmres_cpr_amg
      // stack (PMIS coarsening, theta = 0.75, standard interpolation with no
      // truncation, C/F-ordered hybrid Gauss-Seidel with 1 sweep, plain
      // V-cycle, dense direct coarse solve at <= 100 rows, no aggressive
      // coarsening). See the HYPRE BoomerAMG reference manual for the codes.
      int amg_coarsen_type = 8;       // HYPRE_BoomerAMGSetCoarsenType (8 = PMIS; BOS: PMIS_2)
      int amg_interp_type = 8;        // HYPRE_BoomerAMGSetInterpType (8 = standard; BOS: Stuben standard)
      int amg_relax_type = 3;         // HYPRE_BoomerAMGSetRelaxType (3 = hybrid forward GS)
      int amg_relax_order = 1;        // HYPRE_BoomerAMGSetRelaxOrder (1 = C/F ordering: C->F down, F->C up)
      int amg_num_sweeps = 1;         // HYPRE_BoomerAMGSetNumSweeps
      double amg_strong_threshold = 0.75; // HYPRE_BoomerAMGSetStrongThreshold (BOS: 0.75)
      int amg_agg_num_levels = 0;     // HYPRE_BoomerAMGSetAggNumLevels (BOS: no aggressive coarsening)
      int amg_agg_interp_type = 6;    // HYPRE_BoomerAMGSetAggInterpType (used only when agg levels > 0)
      int amg_agg_pmax_elmts = 20;    // HYPRE_BoomerAMGSetAggPMaxElmts (used only when agg levels > 0)
      int amg_pmax_elmts = 0;         // HYPRE_BoomerAMGSetPMaxElmts (0 = no interp truncation; BOS: none)
      double amg_trunc_factor = 0.0;  // HYPRE_BoomerAMGSetTruncFactor (BOS: 0)
      int amg_max_levels = -1;        // HYPRE_BoomerAMGSetMaxLevels
      int amg_cycle_type = -1;        // HYPRE_BoomerAMGSetCycleType (HYPRE default 1 = V; BOS: V)
      int amg_max_coarse_size = 100;  // HYPRE_BoomerAMGSetMaxCoarseSize (BOS: dense LU at <= 100 rows)
      int amg_coarse_relax_type = 9;  // HYPRE_BoomerAMGSetCycleRelaxType(.., 3) (9 = Gaussian elim.)
      double amg_relax_wt = -1.0;     // HYPRE_BoomerAMGSetRelaxWt (HYPRE default 1.0 = BOS plain GS)

      // Pressure-decoupling weight scheme: 1 = column-sum True-IMPES
      // (Wallis 1983; parity with the proprietary linsolv_bos_cpr), 0 =
      // diagonal-block-only (the previous behaviour).
      int weight_scheme = 1;
      // Full-system smoothing stage: 1 = in-tree block ILU(0) on the
      // block-CSR system (dense NxN block inverses; parity with the
      // proprietary csr_ilu_prec), 0 = HYPRE scalar ILU(k) on the expanded
      // scalar system (the previous behaviour).
      int stage2_type = 1;

      // Adjoint (CPRA) transpose-chain policy. The transpose hierarchies
      // (BoomerAMG on A_p^T + HYPRE_ILU on A_s^T) are only needed by
      // solve_transposed(); by default they are built lazily on the first
      // transposed solve and kept refreshed afterwards. eager_adjoint = true
      // restores the historical behaviour of building them on every setup()
      // from the start (paying ~2x preconditioner-setup cost in forward-only
      // simulations).
      bool eager_adjoint = false;

      // Hierarchy-reuse policy (mirrors mgr::SolverParameters' BCSR-CPR knobs).
      // OFF by default -- rebuild AMG/ILU every Newton iteration, the proven
      // baseline behaviour. With reuse on, *Setup is skipped while the outer
      // Krylov converges within adaptive_iter_threshold iterations (the outer
      // linsolv_gmres feeds the count back after every solve); the adaptive
      // rebuild forces a fresh hierarchy after adaptive_consecutive_bad
      // over-threshold solves in a row.
      bool reuse_amg_hierarchy = false;
      bool adaptive_amg_rebuild = false;
      int adaptive_iter_threshold = 15;
      int adaptive_consecutive_bad = 2;
    };

    /** Configuration for the open-source FS-CPR (Full-System CPR) 4-block
     *  poromechanics preconditioner (linsolv_fs_cpr).
     *
     *  FS-CPR is a two-stage poromechanics preconditioner: an HYPRE-BoomerAMG
     *  correction on the displacement (U) subsystem followed by a second
     *  correction on the flow / pressure (P or PPSS) subsystem driven via a
     *  Schur-complement approximation. It is the in-tree replacement for the
     *  proprietary ``linsolv_bos_fs_cpr``.
     *
     *  The sub-preconditioners (HYPRE BoomerAMG for both U and PPSS stages)
     *  are created internally by the factory using these knobs; nested
     *  preconditioner spec injection is not yet supported. Block sizes 4..8
     *  are supported (ND = 3 hard-coded; NE = N_BLOCK_SIZE - 3 in 1..5).
     *  ``n_fracs > 0`` is NOT YET SUPPORTED (FS_UPG path pending).
     *
     *  Variable-layout overrides (``p_var`` / ``z_var`` / ``u_var`` / ``nc``):
     *  the engine's variable index convention is normally inferred at factory
     *  time from the assumed ``engine_super_elastic_cpu`` layout
     *  (``P_VAR = 0, Z_VAR = 1, U_VAR = NE, NC = NE``). Models that drive
     *  ``engine_pm_cpu`` use a different layout (``U_VAR = 0, P_VAR = 3,
     *  Z_VAR = 255 sentinel, NC = 1``) and need to override these fields. Any
     *  field left at the default of ``-1`` falls back to the convention
     *  default; only fields the model explicitly sets get used.
     *
     *  Inherits max_iterations / tolerance / print_level from solver_config.
     */
    struct fs_cpr_solver_config : opendarts::linear_solvers::solver_config
    {
      bool force_amg_asymmetric = true;          // BoomerAMG symmetric-detection workaround
      opendarts::config::index_t n_res = 0;      // engine-set partition (can stay 0 here)
      opendarts::config::index_t n_fracs = 0;    // n_fracs > 0 is currently rejected
      opendarts::config::index_t n_wells = 0;
      int u_amg_max_iters = 1;                   // U-block BoomerAMG V-cycle budget (1 = single sweep)
      int p_amg_max_iters = 1;                   // PPSS BoomerAMG V-cycle budget

      // Variable-index overrides; -1 = "use the engine_super_elastic_cpu
      // convention default". Set explicitly for engine_pm_cpu (which uses
      // p_var=3, u_var=0, z_var=255, nc=1). Stored as int because the
      // engine_pm_cpu Z_VAR sentinel (255) collides with std::uint8_t's
      // useful unset values.
      int p_var = -1;
      int z_var = -1;
      int u_var = -1;
      int nc    = -1;
    };

    /** Configuration for the mineral-equation Schur elimination wrapper
     *  (linsolv_schur_elim).
     *
     *  Exact per-cell static condensation of one flux-free (mineral balance)
     *  equation/unknown pair per block before the attached inner solver runs on
     *  the reduced (N-1)-sized system. The inner solver is attached by the
     *  caller via set_prec() and must be built for block size N-1.
     *  The Python counterpart is ``SchurEliminationSpec``.
     */
    struct schur_elim_solver_config : opendarts::linear_solvers::solver_config
    {
      int elim_col = 1;        // eliminated unknown column: the mineral z (natural pairing:
                               // the mineral balance determines the mineral unknown; its small
                               // pivot is balanced by the equally small mineral column).
                               // -1 = experimental per-row max-magnitude auto pivot
      int elim_row = 0;        // preferred eliminated equation row (mineral balance)
      double pivot_eps = 0.0;  // pivots <= eps disqualify a candidate during detection
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_SOLVER_CONFIGS_HPP
//--------------------------------------------------------------------------
