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
#include "mech/engine_pm_cpu.hpp"
#include "conn_mesh.h"

PYBIND11_MAKE_OPAQUE(std::vector<pm::contact>);

void pybind_engine_pm_cpu(py::module& m)
{
	py::class_<engine_pm_cpu, engine_base>(m, "engine_pm_cpu", "Isothermal poromechanics CPU simulator engine for sigle phase single component flow")  \
		.def(py::init<>()) \
		.def("init", (int (engine_pm_cpu::*)(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_pm_cpu::init, "Initialize simulator by mesh and params", py::keep_alive<1, 5>()) \
		.def("calc_newton_dev", &engine_pm_cpu::calc_newton_dev) \
		.def("apply_newton_update", &engine_pm_cpu::apply_newton_update) \
		.def("post_newtonloop", &engine_pm_cpu::post_newtonloop) \
		.def("update_uu_jacobian", &engine_pm_cpu::update_uu_jacobian) \
		.def_readwrite("find_equilibrium", &engine_pm_cpu::FIND_EQUILIBRIUM) \
		.def_readwrite("print_linear_system", &engine_pm_cpu::PRINT_LINEAR_SYSTEM) \
		.def_readwrite("time_dependent_discretization", &engine_pm_cpu::TIME_DEPENDENT_DISCRETIZATION) \
		.def_readwrite("scale_rows", &engine_pm_cpu::SCALE_ROWS) \
		.def_readwrite("geomechanics_mode", &engine_pm_cpu::geomechanics_mode) \
		.def_readwrite("newton_update_coefficient", &engine_pm_cpu::newton_update_coefficient) \
		.def_readwrite("dev_u", &engine_pm_cpu::dev_u) \
		.def_readwrite("dev_p", &engine_pm_cpu::dev_p) \
		.def_readwrite("dev_g", &engine_pm_cpu::dev_g) \
		.def_readwrite("dev_u_prev", &engine_pm_cpu::dev_u_prev) \
		.def_readwrite("dev_p_prev", &engine_pm_cpu::dev_p_prev) \
		.def_readwrite("dev_g_prev", &engine_pm_cpu::dev_g_prev) \
		.def_readwrite("well_residual_prev_dt", &engine_pm_cpu::well_residual_prev_dt) \
		.def_readwrite("fluxes", &engine_pm_cpu::fluxes) \
		.def_readwrite("fluxes_n", &engine_pm_cpu::fluxes_n) \
		.def_readwrite("fluxes_biot", &engine_pm_cpu::fluxes_biot) \
		.def_readwrite("fluxes_ref", &engine_pm_cpu::fluxes_ref) \
		.def_readwrite("fluxes_biot_ref", &engine_pm_cpu::fluxes_biot_ref) \
		.def_readwrite("fluxes_ref_n", &engine_pm_cpu::fluxes_ref_n) \
		.def_readwrite("fluxes_biot_ref_n", &engine_pm_cpu::fluxes_biot_ref_n) \
		.def_readwrite("eps_vol", &engine_pm_cpu::eps_vol) \
		.def_readwrite("contacts", &engine_pm_cpu::contacts) \
		.def_readwrite("X", &engine_pm_cpu::X) \
		.def_readwrite("Xn", &engine_pm_cpu::Xn) \
		.def_readwrite("Xn1", &engine_pm_cpu::Xn1) \
		.def_readwrite("dt", &engine_pm_cpu::dt) \
		.def_readwrite("dt1", &engine_pm_cpu::dt1) \
		.def_readwrite("momentum_inertia", &engine_pm_cpu::momentum_inertia) \
		.def_readwrite("Xref", &engine_pm_cpu::Xref) \
		.def_readwrite("Xn_ref", &engine_pm_cpu::Xn_ref) \
		.def_readwrite("dX", &engine_pm_cpu::dX) \
		.def_readwrite("RHS", &engine_pm_cpu::RHS) \
		.def_readwrite("contact_solver", &engine_pm_cpu::contact_solver) \
		.def_property_readonly_static("P_VAR", [](py::object) {return engine_pm_cpu::P_VAR; }) \
		.def_property_readonly_static("Z_VAR", [](py::object) {return engine_pm_cpu::Z_VAR; }) \
		.def_property_readonly_static("U_VAR", [](py::object) {return engine_pm_cpu::U_VAR; }) \
		.def_property_readonly_static("N_VARS", [](py::object) {return engine_pm_cpu::N_VARS; }) \
		.def_property_readonly_static("N_OPS", [](py::object) {return engine_pm_cpu::N_OPS; }) \
		.def_property_readonly_static("NC", [](py::object) {return engine_pm_cpu::NC_; }) \
		.def_property_readonly_static("ACC_OP", [](py::object) {return engine_pm_cpu::ACC_OP; }) \
		.def_property_readonly_static("FLUX_OP", [](py::object) {return engine_pm_cpu::FLUX_OP; }) \
		.def_property_readonly_static("GRAV_OP", [](py::object) {return engine_pm_cpu::GRAV_OP; });

	py::bind_vector<std::vector<pm::contact>>(m, "contact_vector")
		.def(py::pickle(
			[](const std::vector<pm::contact>& p) { // __getstate__
				py::tuple t(p.size());
				for (int i = 0; i < p.size(); i++)
					t[i] = p[i];

				return t;
			},
			[](py::tuple t) { // __setstate__
				std::vector<pm::contact> p(t.size());

				for (int i = 0; i < p.size(); i++)
					p[i] = t[i].cast<pm::contact>();

				return p;
			}));
};

#endif //PYBIND11_ENABLED