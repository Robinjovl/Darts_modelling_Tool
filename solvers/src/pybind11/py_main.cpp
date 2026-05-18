#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <memory>

namespace py = pybind11;

#include "linsolv_iface.hpp"
#include "linsolv_iface_bos.hpp"
#include "linsolv_mgr.hpp"

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
        .def("set_use_flex_gmres", &linsolv_mgr<N>::set_use_flex_gmres,
             "Select FlexGMRES (true) or GMRES (false) for the outer Krylov solver",
             py::arg("use_flex_gmres"))
        .def("set_mgr_pressure_amg_options", &linsolv_mgr<N>::set_mgr_pressure_amg_options,
             "Set key BoomerAMG options for the pressure coarse solver",
             py::arg("coarsen_type"), py::arg("interp_type"), py::arg("relax_type"),
             py::arg("agg_num_levels"), py::arg("agg_interp_type"),
             py::arg("agg_pmax_elmts"), py::arg("relax_order"))
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
        .def("get_use_flex_gmres", &linsolv_mgr<N>::get_use_flex_gmres,
             "Get whether FlexGMRES is enabled")
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

PYBIND11_MODULE(solvers, m)
{
    m.doc() = "openDARTS linear solvers module";

    // Bind abstract base interface
    py::class_<linsolv_iface, std::shared_ptr<linsolv_iface>>(
        m, "LinearSolverInterface",
        "Abstract interface for linear solvers")
        .def("get_n_iters", &linsolv_iface::get_n_iters)
        .def("get_residual", &linsolv_iface::get_residual);

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
}

#endif // PYBIND11_ENABLED
