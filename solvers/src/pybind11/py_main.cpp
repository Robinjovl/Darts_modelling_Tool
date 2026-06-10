#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <memory>

namespace py = pybind11;

#include "linsolv_iface.hpp"
#include "linsolv_iface_bos.hpp"
#include "linsolv_mgr.hpp"
#include "linear_solver.hpp"
#include "solver_config.hpp"
#include "solver_configs.hpp"
#include "solver_factories.hpp"
#include "solver_registry.hpp"

using namespace opendarts::linear_solvers;
using namespace opendarts::config;

// Factory functions for creating MGR solvers with different block sizes
template<uint8_t N>
std::shared_ptr<linsolv_mgr<N>> create_mgr_solver()
{
    return std::make_shared<linsolv_mgr<N>>();
}

// Helper function to bind linsolv_mgr for a specific block size
template<uint8_t N>
void bind_linsolv_mgr_specialization(py::module &m, const char* name)
{
    py::class_<linsolv_mgr<N>, linsolv_iface_bos<N>, std::shared_ptr<linsolv_mgr<N>>>(
        m, name,
        "MGR linear solver for block size N")
        .def(py::init<>())

        // Configuration methods
        .def("set_max_iterations", &linsolv_mgr<N>::set_max_iterations,
             "Set maximum number of iterations", py::arg("max_iters"))
        .def("set_tolerance", &linsolv_mgr<N>::set_tolerance,
             "Set convergence tolerance", py::arg("tolerance"))
        .def("set_kdim", &linsolv_mgr<N>::set_kdim,
             "Set Krylov subspace dimension", py::arg("kdim"))
        .def("set_use_mgr", &linsolv_mgr<N>::set_use_mgr,
             "Enable/disable MGR preconditioner", py::arg("use_mgr"))
        .def("set_log_level", &linsolv_mgr<N>::set_log_level,
             "Set logging verbosity level (0=none, 1=basic, 2=detailed)", py::arg("log_level"))
        .def("set_use_physics_scaling", &linsolv_mgr<N>::set_use_physics_scaling,
             "Enable/disable physics-based row/column scaling",
             py::arg("use_scaling"))
        .def("set_mgr_scaling_type", &linsolv_mgr<N>::set_mgr_scaling_type,
             "Set scaling mode (0=none, 1=physics, 2=row/column one-norm, 3=diagonal)",
             py::arg("scaling_type"))
        .def("set_use_flex_gmres", &linsolv_mgr<N>::set_use_flex_gmres,
             "Select FlexGMRES (true) or GMRES (false) for the outer Krylov solver",
             py::arg("use_flex_gmres"))
        .def("set_mgr_composite_mode", &linsolv_mgr<N>::set_mgr_composite_mode,
             "Set composite preconditioner mode (0=MGR only, 1=MGR then local, 2=local only)",
             py::arg("composite_mode"))
        .def("set_mgr_local_solver", &linsolv_mgr<N>::set_mgr_local_solver,
             "Set full-system BCSR local solver (0=none, 1=block Jacobi, 2=block ILU(0), 3=block ILU(1))",
             py::arg("local_solver"))
        .def("set_mgr_bilu0_pivot_shift", &linsolv_mgr<N>::set_mgr_bilu0_pivot_shift,
             "Set relative diagonal shift used when inverting BILU0 dense diagonal blocks",
             py::arg("pivot_shift"))
        .def("set_mgr_bilu0_fallback_options", &linsolv_mgr<N>::set_mgr_bilu0_fallback_options,
             "Set BILU0 fallback behavior (0=identity, 1=shifted dense, 2=bounded diagonal, 3=shifted dense then bounded diagonal)",
             py::arg("fallback_strategy"), py::arg("diagonal_tolerance") = 1.0e-4,
             py::arg("shifted_max") = 1.0e-4, py::arg("shifted_growth") = 100.0)
        .def("set_mgr_local_correction_options", &linsolv_mgr<N>::set_mgr_local_correction_options,
             "Set BCSR local correction damping and fallback-ratio adaptive damping",
             py::arg("alpha") = 1.0, py::arg("adaptive_fallback_threshold") = -1.0,
             py::arg("adaptive_alpha") = 0.0,
             py::arg("adaptive_fallback_threshold_high") = -1.0,
             py::arg("adaptive_alpha_high") = 0.0)
        .def("set_mgr_local_correction_quality_options",
             &linsolv_mgr<N>::set_mgr_local_correction_quality_options,
             "Enable residual-minimizing scalar damping for the BCSR local correction",
             py::arg("enabled") = false, py::arg("min_alpha") = 0.0)
        .def("set_use_bcsr_cpr", &linsolv_mgr<N>::set_use_bcsr_cpr,
             "Enable/disable experimental BCSR-native CPR preconditioner",
             py::arg("use_bcsr_cpr"))
        .def("set_bcsr_cpr_options", &linsolv_mgr<N>::set_bcsr_cpr_options,
             "Set experimental BCSR CPR options (reduction type, pressure variable, max row weight)",
             py::arg("reduction_type"), py::arg("pressure_variable") = 0,
             py::arg("weight_max") = 1.0e6)
        .def("set_bcsr_cpr_reuse_options", &linsolv_mgr<N>::set_bcsr_cpr_reuse_options,
             "Set experimental BCSR CPR reuse options (reuse AMG hierarchy, rebuild interval; <=0 means first setup only)",
             py::arg("reuse_amg_hierarchy") = false,
             py::arg("amg_rebuild_interval") = 1)
        .def("set_bcsr_cpr_adaptive_rebuild_options",
             &linsolv_mgr<N>::set_bcsr_cpr_adaptive_rebuild_options,
             "Set adaptive BCSR CPR pressure AMG rebuild options driven by previous LI",
             py::arg("adaptive_amg_rebuild") = false,
             py::arg("li_threshold") = 80,
             py::arg("li_growth_factor") = 2.0,
             py::arg("min_reuse_setups") = 1,
             py::arg("max_reuse_setups") = 0)
        .def("set_bcsr_cpr_adaptive_quality_options",
             &linsolv_mgr<N>::set_bcsr_cpr_adaptive_quality_options,
             "Set adaptive BCSR CPR pressure AMG rebuild quality signals",
             py::arg("pressure_overshoot_threshold") = -1.0,
             py::arg("final_proxy_threshold") = -1.0,
             py::arg("fallback_threshold") = -1.0)
        .def("set_bcsr_cpr_diagnostics_options",
             &linsolv_mgr<N>::set_bcsr_cpr_diagnostics_options,
             "Set BCSR CPR diagnostic logging options",
             py::arg("diagnostics") = false,
             py::arg("apply_interval") = 0,
             py::arg("matrix_interval") = 0)
        .def("set_bcsr_cpr_pressure_correction_options",
             &linsolv_mgr<N>::set_bcsr_cpr_pressure_correction_options,
             "Set BCSR CPR pressure correction damping and overshoot guard options",
             py::arg("alpha") = 1.0,
             py::arg("guard_threshold") = -1.0,
             py::arg("guard_min_alpha") = 0.0)
        .def("set_bcsr_cpr_transpose_apply",
             &linsolv_mgr<N>::set_bcsr_cpr_transpose_apply,
             "Apply BCSR CPR in adjoint transpose order (local stage before pressure stage)",
             py::arg("transpose_apply"))
        .def("set_bcsr_cpr_forward_source",
             &linsolv_mgr<N>::set_bcsr_cpr_forward_source,
             "Build adjoint BCSR CPR pressure data from the forward matrix and use its transpose",
             py::arg("forward_source"))
        .def("set_mgr_pressure_amg_options", &linsolv_mgr<N>::set_mgr_pressure_amg_options,
             "Set key BoomerAMG options for the pressure coarse solver",
             py::arg("coarsen_type"), py::arg("interp_type"), py::arg("relax_type"),
             py::arg("agg_num_levels"), py::arg("agg_interp_type"),
             py::arg("agg_pmax_elmts"), py::arg("relax_order"))
        .def("set_mgr_pressure_amg_advanced_options",
             &linsolv_mgr<N>::set_mgr_pressure_amg_advanced_options,
             "Set advanced BoomerAMG options for the pressure coarse solver; negative values keep HYPRE defaults",
             py::arg("strong_threshold") = -1.0, py::arg("trunc_factor") = -1.0,
             py::arg("pmax_elmts") = -1, py::arg("max_levels") = 0)
        .def("set_mgr_pressure_amg_solve_options",
             &linsolv_mgr<N>::set_mgr_pressure_amg_solve_options,
             "Set BoomerAMG solve options for the pressure coarse solver",
             py::arg("max_iter") = 1, py::arg("tolerance") = 0.0)
        .def("set_n_reservoir_blocks", &linsolv_mgr<N>::set_n_reservoir_blocks,
             "Set number of reservoir blocks before appended well blocks", py::arg("n_reservoir_blocks"))
        .def("set_mgr_enable_well_level", &linsolv_mgr<N>::set_mgr_enable_well_level,
             "Enable/disable the dedicated MGR well-elimination level",
             py::arg("enable_well_level"))
        .def("set_mgr_enable_composition_level", &linsolv_mgr<N>::set_mgr_enable_composition_level,
             "Enable/disable the optional reservoir composition reduction level",
             py::arg("enable_composition_level"))
        .def("set_mgr_reservoir_variable_roles", &linsolv_mgr<N>::set_mgr_reservoir_variable_roles,
             "Set physical roles for reservoir local variables, ordered by local variable index",
             py::arg("variable_roles"))
        .def("set_mgr_well_variable_roles", &linsolv_mgr<N>::set_mgr_well_variable_roles,
             "Set physical roles for well local variables, ordered by local variable index",
             py::arg("variable_roles"))
        .def("clear_mgr_custom_levels", &linsolv_mgr<N>::clear_mgr_custom_levels,
             "Remove all user-defined MGR custom reduction levels")
        .def("set_mgr_num_custom_levels", &linsolv_mgr<N>::set_mgr_num_custom_levels,
             "Resize the user-defined MGR custom reduction level list",
             py::arg("num_custom_levels"))
        .def("set_mgr_custom_level_options", &linsolv_mgr<N>::set_mgr_custom_level_options,
             "Set keep labels and HYPRE options for a user-defined MGR custom reduction level",
             py::arg("custom_level_index"), py::arg("keep_labels"), py::arg("frelax_type"),
             py::arg("frelax_iters"), py::arg("interp_type"), py::arg("restrict_type"),
             py::arg("coarse_method"), py::arg("smoother_type"), py::arg("smoother_iters"))
        .def("set_mgr_well_strategy", &linsolv_mgr<N>::set_mgr_well_strategy,
             "Set well strategy (0=eliminate well block, 1=keep well primary on coarse grid)",
             py::arg("well_strategy"))
        .def("set_mgr_well_frelax_type", &linsolv_mgr<N>::set_mgr_well_frelax_type,
             "Set HYPRE MGR F-relaxation type for the well reduction level (e.g. 7=Jacobi, 18=L1-Jacobi, 199=direct inverse)",
             py::arg("frelax_type"))
        .def("set_mgr_well_frelax_iters", &linsolv_mgr<N>::set_mgr_well_frelax_iters,
             "Set number of F-relaxation sweeps for the well reduction level", py::arg("frelax_iters"))
        .def("set_mgr_well_level_options", &linsolv_mgr<N>::set_mgr_well_level_options,
             "Set all options for the well reduction level",
             py::arg("frelax_type"), py::arg("frelax_iters"), py::arg("interp_type"),
             py::arg("restrict_type"), py::arg("coarse_method"), py::arg("smoother_type"),
             py::arg("smoother_iters"))
        .def("set_mgr_composition_level_options", &linsolv_mgr<N>::set_mgr_composition_level_options,
             "Set all options for the optional reservoir composition reduction level",
             py::arg("frelax_type"), py::arg("frelax_iters"), py::arg("interp_type"),
             py::arg("restrict_type"), py::arg("coarse_method"), py::arg("smoother_type"),
             py::arg("smoother_iters"))
        .def("set_mgr_pressure_level_options", &linsolv_mgr<N>::set_mgr_pressure_level_options,
             "Set all options for the reservoir pressure reduction level",
             py::arg("frelax_type"), py::arg("frelax_iters"), py::arg("interp_type"),
             py::arg("restrict_type"), py::arg("coarse_method"), py::arg("smoother_type"),
             py::arg("smoother_iters"))
        // Getter methods
        .def("get_max_iterations", &linsolv_mgr<N>::get_max_iterations,
             "Get maximum number of iterations")
        .def("get_tolerance", &linsolv_mgr<N>::get_tolerance,
             "Get convergence tolerance")
        .def("get_kdim", &linsolv_mgr<N>::get_kdim,
             "Get Krylov subspace dimension")
        .def("get_use_mgr", &linsolv_mgr<N>::get_use_mgr,
             "Get whether MGR preconditioner is enabled")
        .def("get_log_level", &linsolv_mgr<N>::get_log_level,
             "Get logging verbosity level")
        .def("get_use_physics_scaling", &linsolv_mgr<N>::get_use_physics_scaling,
             "Get whether physics-based row/column scaling is enabled")
        .def("get_mgr_scaling_type", &linsolv_mgr<N>::get_mgr_scaling_type,
             "Get scaling mode (0=none, 1=physics, 2=row/column one-norm, 3=diagonal)")
        .def("get_use_flex_gmres", &linsolv_mgr<N>::get_use_flex_gmres,
             "Get whether FlexGMRES is enabled")
        .def("get_mgr_composite_mode", &linsolv_mgr<N>::get_mgr_composite_mode,
             "Get composite preconditioner mode")
        .def("get_mgr_local_solver", &linsolv_mgr<N>::get_mgr_local_solver,
             "Get full-system BCSR local solver type")
        .def("get_mgr_bilu0_pivot_shift", &linsolv_mgr<N>::get_mgr_bilu0_pivot_shift,
             "Get BILU0 relative diagonal pivot shift")
        .def("get_mgr_bilu0_fallback_strategy", &linsolv_mgr<N>::get_mgr_bilu0_fallback_strategy,
             "Get BILU0 fallback strategy")
        .def("get_mgr_bilu0_fallback_diagonal_tolerance",
             &linsolv_mgr<N>::get_mgr_bilu0_fallback_diagonal_tolerance,
             "Get BILU0 fallback diagonal tolerance")
        .def("get_mgr_bilu0_fallback_shifted_max",
             &linsolv_mgr<N>::get_mgr_bilu0_fallback_shifted_max,
             "Get BILU0 shifted dense fallback maximum relative shift")
        .def("get_mgr_bilu0_fallback_shifted_growth",
             &linsolv_mgr<N>::get_mgr_bilu0_fallback_shifted_growth,
             "Get BILU0 shifted dense fallback shift growth factor")
        .def("get_mgr_local_correction_alpha",
             &linsolv_mgr<N>::get_mgr_local_correction_alpha,
             "Get BCSR local correction damping factor")
        .def("get_mgr_local_correction_adaptive_fallback_threshold",
             &linsolv_mgr<N>::get_mgr_local_correction_adaptive_fallback_threshold,
             "Get fallback-ratio threshold for adaptive BCSR local correction damping")
        .def("get_mgr_local_correction_adaptive_alpha",
             &linsolv_mgr<N>::get_mgr_local_correction_adaptive_alpha,
             "Get adaptive BCSR local correction damping factor")
        .def("get_mgr_local_correction_adaptive_fallback_threshold_high",
             &linsolv_mgr<N>::get_mgr_local_correction_adaptive_fallback_threshold_high,
             "Get high fallback-ratio threshold for adaptive BCSR local correction damping")
        .def("get_mgr_local_correction_adaptive_alpha_high",
             &linsolv_mgr<N>::get_mgr_local_correction_adaptive_alpha_high,
             "Get high-ratio adaptive BCSR local correction damping factor")
        .def("get_use_bcsr_cpr", &linsolv_mgr<N>::get_use_bcsr_cpr,
             "Get whether experimental BCSR-native CPR is enabled")
        .def("get_bcsr_cpr_reduction_type", &linsolv_mgr<N>::get_bcsr_cpr_reduction_type,
             "Get BCSR CPR reduction type")
        .def("get_bcsr_cpr_pressure_variable", &linsolv_mgr<N>::get_bcsr_cpr_pressure_variable,
             "Get BCSR CPR pressure variable index")
        .def("get_bcsr_cpr_weight_max", &linsolv_mgr<N>::get_bcsr_cpr_weight_max,
             "Get BCSR CPR maximum accepted True-IMPES row weight")
        .def("get_bcsr_cpr_transpose_apply",
             &linsolv_mgr<N>::get_bcsr_cpr_transpose_apply,
             "Get whether BCSR CPR transpose apply order is enabled")
        .def("get_bcsr_cpr_forward_source",
             &linsolv_mgr<N>::get_bcsr_cpr_forward_source,
             "Get whether adjoint BCSR CPR uses the forward matrix as CPR source")
        .def("get_bcsr_cpr_reuse_amg_hierarchy",
             &linsolv_mgr<N>::get_bcsr_cpr_reuse_amg_hierarchy,
             "Get whether BCSR CPR reuses the pressure AMG hierarchy")
        .def("get_bcsr_cpr_amg_rebuild_interval",
             &linsolv_mgr<N>::get_bcsr_cpr_amg_rebuild_interval,
             "Get BCSR CPR pressure AMG rebuild interval")
        .def("get_bcsr_cpr_adaptive_amg_rebuild",
             &linsolv_mgr<N>::get_bcsr_cpr_adaptive_amg_rebuild,
             "Get whether adaptive BCSR CPR pressure AMG rebuild is enabled")
        .def("get_bcsr_cpr_adaptive_li_threshold",
             &linsolv_mgr<N>::get_bcsr_cpr_adaptive_li_threshold,
             "Get adaptive BCSR CPR LI threshold")
        .def("get_bcsr_cpr_adaptive_li_growth_factor",
             &linsolv_mgr<N>::get_bcsr_cpr_adaptive_li_growth_factor,
             "Get adaptive BCSR CPR LI growth factor")
        .def("get_bcsr_cpr_adaptive_min_reuse_setups",
             &linsolv_mgr<N>::get_bcsr_cpr_adaptive_min_reuse_setups,
             "Get adaptive BCSR CPR minimum reuse setups")
        .def("get_bcsr_cpr_adaptive_max_reuse_setups",
             &linsolv_mgr<N>::get_bcsr_cpr_adaptive_max_reuse_setups,
             "Get adaptive BCSR CPR maximum reuse setups")
        .def("get_n_reservoir_blocks", &linsolv_mgr<N>::get_n_reservoir_blocks,
             "Get configured number of reservoir blocks")
        .def("get_mgr_enable_well_level", &linsolv_mgr<N>::get_mgr_enable_well_level,
             "Get whether the dedicated MGR well-elimination level is enabled")
        .def("get_mgr_enable_composition_level", &linsolv_mgr<N>::get_mgr_enable_composition_level,
             "Get whether the optional reservoir composition reduction level is enabled")
        .def("get_mgr_num_custom_levels", &linsolv_mgr<N>::get_mgr_num_custom_levels,
             "Get number of user-defined MGR custom reduction levels")
        .def("get_mgr_reservoir_variable_roles", &linsolv_mgr<N>::get_mgr_reservoir_variable_roles,
             "Get configured reservoir variable roles")
        .def("get_mgr_well_variable_roles", &linsolv_mgr<N>::get_mgr_well_variable_roles,
             "Get configured well variable roles")
        .def("get_mgr_well_strategy", &linsolv_mgr<N>::get_mgr_well_strategy,
             "Get configured well strategy")
        .def("get_mgr_well_frelax_type", &linsolv_mgr<N>::get_mgr_well_frelax_type,
             "Get configured HYPRE MGR F-relaxation type for the well reduction level")
        .def("get_mgr_well_frelax_iters", &linsolv_mgr<N>::get_mgr_well_frelax_iters,
             "Get configured F-relaxation sweeps for the well reduction level")
        // Interface methods (inherited from linsolv_iface)
        .def("get_n_iters", &linsolv_mgr<N>::get_n_iters,
             "Get number of iterations from last solve")
        .def("get_residual", &linsolv_mgr<N>::get_residual,
             "Get final residual from last solve");
}

// Helper function to bind linsolv_iface_bos for a specific block size
template<uint8_t N>
void bind_linsolv_iface_bos_specialization(py::module &m, const char* name)
{
    py::class_<linsolv_iface_bos<N>, linsolv_iface, std::shared_ptr<linsolv_iface_bos<N>>>(
        m, name,
        "Base BOS solver interface for block size N");
}

// Bind the unified linear-solver API: the configuration structs, the solver
// handle, and the registry. This is the modern, enum-free API -- a solver is
// selected by registered name (e.g. "mgr", "superlu") plus a config object.
void bind_unified_solver_api(py::module &m)
{
    // Outcome of the last solve.
    py::class_<solver_stats>(m, "SolverStats", "Outcome of the last linear solve.")
        .def(py::init<>())
        .def_readwrite("iterations", &solver_stats::iterations)
        .def_readwrite("residual", &solver_stats::residual)
        .def_readwrite("converged", &solver_stats::converged)
        .def_readwrite("setup_time", &solver_stats::setup_time)
        .def_readwrite("solve_time", &solver_stats::solve_time);

    // Base configuration shared by every solver.
    py::class_<solver_config>(m, "SolverConfig",
        "Base linear-solver configuration (max_iterations, tolerance). For "
        "engine-resident solvers these are subsequently overridden by the "
        "engine's init() call from data_ts.linear_tol / linear_max_iter. "
        "Per-solver verbosity is exposed by the solver-specific config, e.g. "
        "MGRSolverConfig.log_level.")
        .def(py::init<>())
        .def_readwrite("max_iterations", &solver_config::max_iterations)
        .def_readwrite("tolerance", &solver_config::tolerance);

    // One HYPRE MGR reduction level.
    py::class_<mgr_level_config>(m, "MGRLevelConfig",
        "HYPRE MGR reduction-level options.")
        .def(py::init<>())
        .def_readwrite("keep_labels", &mgr_level_config::keep_labels)
        .def_readwrite("frelax_type", &mgr_level_config::frelax_type)
        .def_readwrite("frelax_iters", &mgr_level_config::frelax_iters)
        .def_readwrite("interp_type", &mgr_level_config::interp_type)
        .def_readwrite("restrict_type", &mgr_level_config::restrict_type)
        .def_readwrite("coarse_method", &mgr_level_config::coarse_method)
        .def_readwrite("smoother_type", &mgr_level_config::smoother_type)
        .def_readwrite("smoother_iters", &mgr_level_config::smoother_iters);

    // Block-ILU(0) local-solver options for the MGR F-relaxation stage.
    py::class_<mgr_bilu0_config>(m, "MGRBILU0Config",
        "MGR block-ILU(0) local-solver options.")
        .def(py::init<>())
        .def_readwrite("pivot_shift", &mgr_bilu0_config::pivot_shift)
        .def_readwrite("fallback_strategy", &mgr_bilu0_config::fallback_strategy)
        .def_readwrite("fallback_diagonal_tolerance", &mgr_bilu0_config::fallback_diagonal_tolerance)
        .def_readwrite("fallback_shifted_max", &mgr_bilu0_config::fallback_shifted_max)
        .def_readwrite("fallback_shifted_growth", &mgr_bilu0_config::fallback_shifted_growth);

    // MGR local-correction (pressure-block damping) options.
    py::class_<mgr_local_correction_config>(m, "MGRLocalCorrectionConfig",
        "MGR local-correction (pressure-block damping) options.")
        .def(py::init<>())
        .def_readwrite("alpha", &mgr_local_correction_config::alpha)
        .def_readwrite("adaptive_fallback_threshold", &mgr_local_correction_config::adaptive_fallback_threshold)
        .def_readwrite("adaptive_alpha", &mgr_local_correction_config::adaptive_alpha)
        .def_readwrite("adaptive_fallback_threshold_high", &mgr_local_correction_config::adaptive_fallback_threshold_high)
        .def_readwrite("adaptive_alpha_high", &mgr_local_correction_config::adaptive_alpha_high)
        .def_readwrite("quality_enabled", &mgr_local_correction_config::quality_enabled)
        .def_readwrite("quality_min_alpha", &mgr_local_correction_config::quality_min_alpha);

    // BCSR-CPR (block-CSR Constrained Pressure Residual) options.
    py::class_<mgr_bcsr_cpr_config>(m, "MGRBCSRCPRConfig",
        "MGR BCSR-CPR options; presence on MGRSolverConfig enables BCSR-CPR.")
        .def(py::init<>())
        .def_readwrite("reduction_type", &mgr_bcsr_cpr_config::reduction_type)
        .def_readwrite("pressure_variable", &mgr_bcsr_cpr_config::pressure_variable)
        .def_readwrite("weight_max", &mgr_bcsr_cpr_config::weight_max)
        .def_readwrite("reuse_amg_hierarchy", &mgr_bcsr_cpr_config::reuse_amg_hierarchy)
        .def_readwrite("amg_rebuild_interval", &mgr_bcsr_cpr_config::amg_rebuild_interval)
        .def_readwrite("adaptive_amg_rebuild", &mgr_bcsr_cpr_config::adaptive_amg_rebuild)
        .def_readwrite("adaptive_li_threshold", &mgr_bcsr_cpr_config::adaptive_li_threshold)
        .def_readwrite("adaptive_li_growth_factor", &mgr_bcsr_cpr_config::adaptive_li_growth_factor)
        .def_readwrite("adaptive_min_reuse_setups", &mgr_bcsr_cpr_config::adaptive_min_reuse_setups)
        .def_readwrite("adaptive_max_reuse_setups", &mgr_bcsr_cpr_config::adaptive_max_reuse_setups)
        .def_readwrite("adaptive_pressure_overshoot_threshold", &mgr_bcsr_cpr_config::adaptive_pressure_overshoot_threshold)
        .def_readwrite("adaptive_final_proxy_threshold", &mgr_bcsr_cpr_config::adaptive_final_proxy_threshold)
        .def_readwrite("adaptive_fallback_threshold", &mgr_bcsr_cpr_config::adaptive_fallback_threshold)
        .def_readwrite("diagnostics", &mgr_bcsr_cpr_config::diagnostics)
        .def_readwrite("diagnostic_apply_interval", &mgr_bcsr_cpr_config::diagnostic_apply_interval)
        .def_readwrite("diagnostic_matrix_interval", &mgr_bcsr_cpr_config::diagnostic_matrix_interval)
        .def_readwrite("pressure_correction_alpha", &mgr_bcsr_cpr_config::pressure_correction_alpha)
        .def_readwrite("pressure_correction_guard_threshold", &mgr_bcsr_cpr_config::pressure_correction_guard_threshold)
        .def_readwrite("pressure_correction_guard_min_alpha", &mgr_bcsr_cpr_config::pressure_correction_guard_min_alpha)
        .def_readwrite("transpose_apply", &mgr_bcsr_cpr_config::transpose_apply)
        .def_readwrite("forward_source", &mgr_bcsr_cpr_config::forward_source);

    // MGR pressure-subsystem BoomerAMG options.
    py::class_<mgr_pressure_amg_config>(m, "MGRPressureAMGConfig",
        "MGR pressure-subsystem BoomerAMG options.")
        .def(py::init<>())
        .def_readwrite("coarsen_type", &mgr_pressure_amg_config::coarsen_type)
        .def_readwrite("interp_type", &mgr_pressure_amg_config::interp_type)
        .def_readwrite("relax_type", &mgr_pressure_amg_config::relax_type)
        .def_readwrite("agg_num_levels", &mgr_pressure_amg_config::agg_num_levels)
        .def_readwrite("agg_interp_type", &mgr_pressure_amg_config::agg_interp_type)
        .def_readwrite("agg_pmax_elmts", &mgr_pressure_amg_config::agg_pmax_elmts)
        .def_readwrite("relax_order", &mgr_pressure_amg_config::relax_order)
        .def_readwrite("strong_threshold", &mgr_pressure_amg_config::strong_threshold)
        .def_readwrite("trunc_factor", &mgr_pressure_amg_config::trunc_factor)
        .def_readwrite("pmax_elmts", &mgr_pressure_amg_config::pmax_elmts)
        .def_readwrite("max_levels", &mgr_pressure_amg_config::max_levels)
        .def_readwrite("solve_max_iter", &mgr_pressure_amg_config::solve_max_iter)
        .def_readwrite("solve_tolerance", &mgr_pressure_amg_config::solve_tolerance);

    // MGR solver configuration (derives SolverConfig).
    py::class_<mgr_solver_config, solver_config>(m, "MGRSolverConfig",
        "Configuration for the HYPRE MGR solver.")
        .def(py::init<>())
        .def_readwrite("kdim", &mgr_solver_config::kdim)
        .def_readwrite("use_mgr", &mgr_solver_config::use_mgr)
        .def_readwrite("log_level", &mgr_solver_config::log_level)
        .def_readwrite("use_physics_scaling", &mgr_solver_config::use_physics_scaling)
        .def_readwrite("use_flex_gmres", &mgr_solver_config::use_flex_gmres)
        .def_readwrite("n_reservoir_blocks", &mgr_solver_config::n_reservoir_blocks)
        .def_readwrite("enable_well_level", &mgr_solver_config::enable_well_level)
        .def_readwrite("enable_composition_level", &mgr_solver_config::enable_composition_level)
        .def_readwrite("reservoir_variable_roles", &mgr_solver_config::reservoir_variable_roles)
        .def_readwrite("well_variable_roles", &mgr_solver_config::well_variable_roles)
        .def_readwrite("well_strategy", &mgr_solver_config::well_strategy)
        .def_readwrite("well_level", &mgr_solver_config::well_level)
        .def_readwrite("composition_level", &mgr_solver_config::composition_level)
        .def_readwrite("pressure_level", &mgr_solver_config::pressure_level)
        .def_readwrite("custom_levels", &mgr_solver_config::custom_levels)
        .def_readwrite("scaling_type", &mgr_solver_config::scaling_type)
        .def_readwrite("composite_mode", &mgr_solver_config::composite_mode)
        .def_readwrite("local_solver", &mgr_solver_config::local_solver)
        .def_readwrite("use_bcsr_cpr", &mgr_solver_config::use_bcsr_cpr)
        .def_readwrite("bilu0", &mgr_solver_config::bilu0)
        .def_readwrite("local_correction", &mgr_solver_config::local_correction)
        .def_readwrite("bcsr_cpr", &mgr_solver_config::bcsr_cpr)
        .def_readwrite("pressure_amg", &mgr_solver_config::pressure_amg);

    // Open-source GMRES outer Krylov solver configuration.
    py::class_<gmres_solver_config, solver_config>(m, "GMRESSolverConfig",
        "Configuration for the open-source GMRES outer Krylov solver.")
        .def(py::init<>())
        .def_readwrite("restart", &gmres_solver_config::restart);

    // Open-source CPR two-stage preconditioner configuration.
    py::class_<cpr_solver_config, solver_config>(m, "CPRSolverConfig",
        "Configuration for the open-source CPR two-stage preconditioner. "
        "BoomerAMG is run with tol=0 as a preconditioner stage, so there is "
        "no inner AMG tolerance knob -- amg_max_iters sets the sweep budget.")
        .def(py::init<>())
        .def_readwrite("amg_max_iters", &cpr_solver_config::amg_max_iters)
        .def_readwrite("ilu_fill_level", &cpr_solver_config::ilu_fill_level);

    // Open-source FS-CPR (poromechanics) preconditioner configuration.
    py::class_<fs_cpr_solver_config, solver_config>(m, "FSCPRSolverConfig",
        "Configuration for the open-source FS-CPR (Full-System CPR) 4-block "
        "poromechanics preconditioner. Sub-preconditioners (HYPRE BoomerAMG "
        "for both U and PPSS) are created by the factory using these knobs; "
        "nested spec injection is not yet supported.")
        .def(py::init<>())
        .def_readwrite("force_amg_asymmetric", &fs_cpr_solver_config::force_amg_asymmetric)
        .def_readwrite("n_res",   &fs_cpr_solver_config::n_res)
        .def_readwrite("n_fracs", &fs_cpr_solver_config::n_fracs)
        .def_readwrite("n_wells", &fs_cpr_solver_config::n_wells)
        .def_readwrite("u_amg_max_iters", &fs_cpr_solver_config::u_amg_max_iters)
        .def_readwrite("p_amg_max_iters", &fs_cpr_solver_config::p_amg_max_iters)
        // Variable-layout overrides (-1 = use engine_super_elastic_cpu
        // convention default). engine_pm_cpu wires through p_var/u_var/
        // z_var/nc explicitly.
        .def_readwrite("p_var", &fs_cpr_solver_config::p_var)
        .def_readwrite("z_var", &fs_cpr_solver_config::z_var)
        .def_readwrite("u_var", &fs_cpr_solver_config::u_var)
        .def_readwrite("nc",    &fs_cpr_solver_config::nc);

    // Unified solver handle returned by create_linear_solver(). Bound once,
    // exposed under two names: "LinearSolver" (new, preferred) and
    // "LinearSolverInterface" (back-compat alias). After the linear_solver /
    // linsolv_iface merger these point to the same C++ class; pybind11
    // rejects registering the same C++ type twice, so we share the binding
    // and only add a Python-side attribute alias for the legacy name. The
    // legacy class_<linsolv_iface, ...> block in PYBIND11_MODULE has been
    // removed for the same reason.
    py::class_<linear_solver, std::shared_ptr<linear_solver>>(m, "LinearSolver",
        "Unified linear-solver handle produced by create_linear_solver().")
        .def("get_n_iters", &linear_solver::get_n_iters)
        .def("get_residual", &linear_solver::get_residual)
        // set_prec stores a raw pointer to the preconditioner; tell pybind11
        // to keep the prec alive as long as the outer solver lives.
        .def("set_prec", &linear_solver::set_prec,
             "Attach a preconditioner (kept alive by the outer solver).",
             py::arg("prec"), py::keep_alive<1, 2>())
        .def("stats", &linear_solver::stats, "Outcome of the last solve.");
    // Legacy Python name for the same handle.
    m.attr("LinearSolverInterface") = m.attr("LinearSolver");

    // Registry API -- this replaces sim_params.linear_solver_t.
    m.def("register_builtin_solvers", &register_builtin_solvers,
          "Register all built-in open-source solvers with the registry (idempotent).");
    m.def("registered_solvers", &registered_solvers,
          "Names of all linear solvers registered in this build.");
    m.def("is_solver_registered", &is_solver_registered,
          "Whether a solver name is available in this build.", py::arg("name"));
    m.def("create_linear_solver",
          [](const std::string &name, const solver_config &config, int block_size)
              -> std::shared_ptr<linsolv_iface> {
              return create_linear_solver(name, config, block_size);
          },
          "Create a linear solver by registered name, configuration and block size. "
          "Returns a LinearSolverInterface that engine_base.set_linear_solver() accepts.",
          py::arg("name"), py::arg("config"), py::arg("block_size"));
}

PYBIND11_MODULE(solvers, m)
{
    m.doc() = "openDARTS linear solvers module";

    // Unified solver-handle binding MUST come first: the linsolv_iface_bos
    // template specialisations below name it as their base class via the
    // ``linsolv_iface`` alias, and pybind11 requires the base C++ type to
    // already be registered. After the linear_solver / linsolv_iface merger
    // the two refer to the same C++ class, so a single class_<> binding is
    // exposed under both Python names ("LinearSolver" preferred,
    // "LinearSolverInterface" kept as a back-compat alias inside
    // bind_unified_solver_api()).
    bind_unified_solver_api(m);

    // Bind linsolv_iface_bos specializations for block sizes 1-13
    bind_linsolv_iface_bos_specialization<1>(m, "LinearSolverBOS_1");
    bind_linsolv_iface_bos_specialization<2>(m, "LinearSolverBOS_2");
    bind_linsolv_iface_bos_specialization<3>(m, "LinearSolverBOS_3");
    bind_linsolv_iface_bos_specialization<4>(m, "LinearSolverBOS_4");
    bind_linsolv_iface_bos_specialization<5>(m, "LinearSolverBOS_5");
    bind_linsolv_iface_bos_specialization<6>(m, "LinearSolverBOS_6");
    bind_linsolv_iface_bos_specialization<7>(m, "LinearSolverBOS_7");
    bind_linsolv_iface_bos_specialization<8>(m, "LinearSolverBOS_8");
    bind_linsolv_iface_bos_specialization<9>(m, "LinearSolverBOS_9");
    bind_linsolv_iface_bos_specialization<10>(m, "LinearSolverBOS_10");
    bind_linsolv_iface_bos_specialization<11>(m, "LinearSolverBOS_11");
    bind_linsolv_iface_bos_specialization<12>(m, "LinearSolverBOS_12");
    bind_linsolv_iface_bos_specialization<13>(m, "LinearSolverBOS_13");

    // Bind MGR solver specializations for block sizes 1-13
    bind_linsolv_mgr_specialization<1>(m, "MGRSolver_1");
    bind_linsolv_mgr_specialization<2>(m, "MGRSolver_2");
    bind_linsolv_mgr_specialization<3>(m, "MGRSolver_3");
    bind_linsolv_mgr_specialization<4>(m, "MGRSolver_4");
    bind_linsolv_mgr_specialization<5>(m, "MGRSolver_5");
    bind_linsolv_mgr_specialization<6>(m, "MGRSolver_6");
    bind_linsolv_mgr_specialization<7>(m, "MGRSolver_7");
    bind_linsolv_mgr_specialization<8>(m, "MGRSolver_8");
    bind_linsolv_mgr_specialization<9>(m, "MGRSolver_9");
    bind_linsolv_mgr_specialization<10>(m, "MGRSolver_10");
    bind_linsolv_mgr_specialization<11>(m, "MGRSolver_11");
    bind_linsolv_mgr_specialization<12>(m, "MGRSolver_12");
    bind_linsolv_mgr_specialization<13>(m, "MGRSolver_13");

    // Factory functions
    m.def("create_mgr_solver", &create_mgr_solver<1>, "Create MGR solver with block size 1");
    m.def("create_mgr_solver", &create_mgr_solver<2>, "Create MGR solver with block size 2");
    m.def("create_mgr_solver", &create_mgr_solver<3>, "Create MGR solver with block size 3");
    m.def("create_mgr_solver", &create_mgr_solver<4>, "Create MGR solver with block size 4");
    m.def("create_mgr_solver", &create_mgr_solver<5>, "Create MGR solver with block size 5");
    m.def("create_mgr_solver", &create_mgr_solver<6>, "Create MGR solver with block size 6");
    m.def("create_mgr_solver", &create_mgr_solver<7>, "Create MGR solver with block size 7");
    m.def("create_mgr_solver", &create_mgr_solver<8>, "Create MGR solver with block size 8");
    m.def("create_mgr_solver", &create_mgr_solver<9>, "Create MGR solver with block size 9");
    m.def("create_mgr_solver", &create_mgr_solver<10>, "Create MGR solver with block size 10");
    m.def("create_mgr_solver", &create_mgr_solver<11>, "Create MGR solver with block size 11");
    m.def("create_mgr_solver", &create_mgr_solver<12>, "Create MGR solver with block size 12");
    m.def("create_mgr_solver", &create_mgr_solver<13>, "Create MGR solver with block size 13");

    // Convenience function that selects the right solver based on block size
    m.def("create_mgr_solver_for_block_size",
          [](int block_size) -> std::shared_ptr<linsolv_iface> {
              if (block_size == 1) return create_mgr_solver<1>();
              else if (block_size == 2) return create_mgr_solver<2>();
              else if (block_size == 3) return create_mgr_solver<3>();
              else if (block_size == 4) return create_mgr_solver<4>();
              else if (block_size == 5) return create_mgr_solver<5>();
              else if (block_size == 6) return create_mgr_solver<6>();
              else if (block_size == 7) return create_mgr_solver<7>();
              else if (block_size == 8) return create_mgr_solver<8>();
              else if (block_size == 9) return create_mgr_solver<9>();
              else if (block_size == 10) return create_mgr_solver<10>();
              else if (block_size == 11) return create_mgr_solver<11>();
              else if (block_size == 12) return create_mgr_solver<12>();
              else if (block_size == 13) return create_mgr_solver<13>();
              else throw std::runtime_error("Unsupported block size: " + std::to_string(block_size));
          },
          "Create MGR solver for given block size (1-13)", py::arg("block_size"));

    // Populate the registry on module import so create_linear_solver() works.
    register_builtin_solvers();
}

#endif // PYBIND11_ENABLED
