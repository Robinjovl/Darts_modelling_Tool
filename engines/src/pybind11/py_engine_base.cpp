//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as 
//    published by the Free Software Foundation, either version 3 of the 
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public 
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************


#ifdef PYBIND11_ENABLED
#include <pybind11.h>
#include "py_globals.h"
#include <stl.h>

namespace py = pybind11;
#include "engine_base.h"

#include "conn_mesh.h"

void pybind_engine_base (py::module &m)
{
	py::class_<engine_base>(m, "engine_base", "Base simulator engine class")  \
		.def("run", &engine_base::run, py::call_guard<py::gil_scoped_release>())  \
		.def("run_timestep", &engine_base::run_timestep, py::call_guard<py::gil_scoped_release>())  \
		.def("report", &engine_base::report)  \
		.def("print_stat", &engine_base::print_stat)  \
		.def("test_assembly", &engine_base::test_assembly)  \
		.def("test_spmv", &engine_base::test_spmv)  \
		.def("run_single_newton_iteration", &engine_base::run_single_newton_iteration, py::call_guard<py::gil_scoped_release>())  \
		.def("calc_newton_residual", &engine_base::calc_newton_residual, py::call_guard<py::gil_scoped_release>())  \
		.def("calc_well_residual", &engine_base::calc_well_residual, py::call_guard<py::gil_scoped_release>())  \
		.def("apply_newton_update", &engine_base::apply_newton_update, py::call_guard<py::gil_scoped_release>())  \
		.def("post_newtonloop", &engine_base::post_newtonloop, py::call_guard<py::gil_scoped_release>())  \
		.def("solve_linear_equation", &engine_base::solve_linear_equation, py::call_guard<py::gil_scoped_release>())  \
		.def_readwrite("X", &engine_base::X) \
		.def_readwrite("Xn", &engine_base::Xn) \
		.def_readwrite("t", &engine_base::t) \
		.def_readwrite("op_vals_arr", &engine_base::op_vals_arr) \
		.def_readwrite("timer", &engine_base::timer) \
		.def_readwrite("CFL_max", &engine_base::CFL_max) \
		.def_readwrite("n_newton_last_dt", &engine_base::n_newton_last_dt) \
		.def_readwrite("newton_residual_last_dt", &engine_base::newton_residual_last_dt) \
		.def_readwrite("stat", &engine_base::stat) \
		.def_readwrite("n_linear_last_dt", &engine_base::n_linear_last_dt) \
		.def_readwrite("op_vals_arr_n", &engine_base::op_vals_arr_n) \
		.def_readwrite("time_data", &engine_base::time_data) \
		.def_readwrite("time_data_report", &engine_base::time_data_report) \
		.def_readwrite("engine_name", &engine_base::engine_name) \
		.def_readwrite("params", &engine_base::params) \
		.def_readwrite("newton_residual_last_dt", &engine_base::newton_residual_last_dt) \
		.def_readwrite("well_residual_last_dt", &engine_base::well_residual_last_dt) \
		.def("add_value_to_Q", &engine_base::add_value_to_Q)  \
		.def("clear_Q", &engine_base::clear_Q)  \
		.def("calc_adjoint_gradient_dirac_all", &engine_base::calc_adjoint_gradient_dirac_all, py::call_guard<py::gil_scoped_release>())  \

		.def("clear_Q_p", &engine_base::clear_Q_p)  \
		.def("add_value_to_Q_p", &engine_base::add_value_to_Q_p)  \
		.def("push_back_to_Q_all", &engine_base::push_back_to_Q_all)  \
		.def("clear_cov_prod_p", &engine_base::clear_cov_prod_p)  \
		.def("add_value_to_cov_prod_p", &engine_base::add_value_to_cov_prod_p)  \
		.def("push_back_to_cov_prod_all", &engine_base::push_back_to_cov_prod_all)  \
		.def("clear_prod_wei_p", &engine_base::clear_prod_wei_p)  \
		.def("add_value_to_prod_wei_p", &engine_base::add_value_to_prod_wei_p)  \
		.def("push_back_to_prod_wei_all", &engine_base::push_back_to_prod_wei_all)  \

		.def("clear_Q_inj_p", &engine_base::clear_Q_inj_p)  \
		.def("add_value_to_Q_inj_p", &engine_base::add_value_to_Q_inj_p)  \
		.def("push_back_to_Q_inj_all", &engine_base::push_back_to_Q_inj_all)  \
		.def("clear_cov_inj_p", &engine_base::clear_cov_inj_p)  \
		.def("add_value_to_cov_inj_p", &engine_base::add_value_to_cov_inj_p)  \
		.def("push_back_to_cov_inj_all", &engine_base::push_back_to_cov_inj_all)  \
		.def("clear_inj_wei_p", &engine_base::clear_inj_wei_p)  \
		.def("add_value_to_inj_wei_p", &engine_base::add_value_to_inj_wei_p)  \
		.def("push_back_to_inj_wei_all", &engine_base::push_back_to_inj_wei_all)  \

		.def("push_back_to_BHP_all", &engine_base::push_back_to_BHP_all)  \
		.def("push_back_to_cov_BHP_all", &engine_base::push_back_to_cov_BHP_all)  \
		.def("push_back_to_BHP_wei_all", &engine_base::push_back_to_BHP_wei_all)  \

		.def("push_back_to_well_tempr_all", &engine_base::push_back_to_well_tempr_all)  \
		.def("push_back_to_cov_well_tempr_all", &engine_base::push_back_to_cov_well_tempr_all)  \
		.def("push_back_to_well_tempr_wei_all", &engine_base::push_back_to_well_tempr_wei_all)  \

		.def("push_back_to_temperature_all", &engine_base::push_back_to_temperature_all)  \
		.def("push_back_to_cov_temperature_all", &engine_base::push_back_to_cov_temperature_all)  \
		.def("push_back_to_temperature_wei_all", &engine_base::push_back_to_temperature_wei_all)  \

		.def("push_back_to_customized_op_all", &engine_base::push_back_to_customized_op_all)  \
		.def("push_back_to_cov_customized_op_all", &engine_base::push_back_to_cov_customized_op_all)  \
		.def("push_back_to_customized_op_wei_all", &engine_base::push_back_to_customized_op_wei_all)  \
		.def_readwrite("threshold", &engine_base::threshold) \
		.def("push_back_to_binary_all", &engine_base::push_back_to_binary_all)  \

		.def("clear_previous_adjoint_assembly", &engine_base::clear_previous_adjoint_assembly)  \
		.def_readwrite("prod_well_name", &engine_base::prod_well_name) \
		.def_readwrite("inj_well_name", &engine_base::inj_well_name) \
		.def_readwrite("BHP_well_name", &engine_base::BHP_well_name) \
		.def_readwrite("well_tempr_name", &engine_base::well_tempr_name) \
		.def_readwrite("component_index", &engine_base::component_index) \
		.def_readwrite("phase_index", &engine_base::phase_index) \
		.def_readwrite("prod_phase_name", &engine_base::prod_phase_name) \
		.def_readwrite("inj_phase_name", &engine_base::inj_phase_name) \
		.def_readwrite("unit", &engine_base::unit) \
		.def_readwrite("Temp_dj_dx", &engine_base::Temp_dj_dx) \
		.def_readwrite("Temp_dj_du", &engine_base::Temp_dj_du) \
		.def_readwrite("col_dT_du", &engine_base::col_dT_du) \
		.def_readwrite("n_control_vars", &engine_base::n_control_vars) \
		.def_readwrite("derivatives", &engine_base::derivatives) \
		.def_readwrite("scale_function_value", &engine_base::scale_function_value) \
		.def_readwrite("X_t", &engine_base::X_t) \
		.def_readwrite("dt", &engine_base::dt) \
		.def_readwrite("dt_t", &engine_base::dt_t) \
		.def_readwrite("t_t", &engine_base::t_t) \
		.def_readwrite("dt_t_report", &engine_base::dt_t_report) \
		.def_readwrite("t_t_report", &engine_base::t_t_report) \
		.def_readwrite("dirac_vec", &engine_base::dirac_vec) \
		.def_readwrite("Q", &engine_base::Q) \
		.def_readwrite("customize_operator", &engine_base::customize_operator) \
		.def_readwrite("customize_op_num", &engine_base::customize_op_num) \
		.def_readwrite("idx_customized_operator", &engine_base::idx_customized_operator) \
		.def_readwrite("op_vals_arr_customized", &engine_base::op_vals_arr_customized) \
		.def_readwrite("time_data_report_customized", &engine_base::time_data_report_customized) \
		.def_readwrite("time_data_customized", &engine_base::time_data_customized) \
		.def_readwrite("cov_mat_inv", &engine_base::cov_mat_inv) \
		.def_readwrite("phase_relative_density", &engine_base::phase_relative_density) \
		.def_readwrite("opt_history_matching", &engine_base::opt_history_matching) \
		.def_readwrite("optimize_component_rate", &engine_base::optimize_component_rate) \
		.def_readwrite("objfun_prod_phase_rate", &engine_base::objfun_prod_phase_rate) \
		.def_readwrite("objfun_inj_phase_rate", &engine_base::objfun_inj_phase_rate) \
		.def_readwrite("objfun_BHP", &engine_base::objfun_BHP) \
		.def_readwrite("objfun_well_tempr", &engine_base::objfun_well_tempr) \
		.def_readwrite("objfun_temperature", &engine_base::objfun_temperature) \
		.def_readwrite("objfun_customized_op", &engine_base::objfun_customized_op) \
		.def_readwrite("objfun_saturation", &engine_base::objfun_saturation) \
		
		;

}

#endif //PYBIND11_ENABLED