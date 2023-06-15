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
#include "mech/engine_super_elastic_cpu.hpp"
#include "conn_mesh.h"

PYBIND11_MAKE_OPAQUE(std::vector<pm::contact>);

template <uint8_t NC, uint8_t NP, bool THERMAL>
struct engine_super_elastic_exposer
{
	static void expose(py::module &m)
	{
		std::string short_name, long_name;
		short_name = "engine_super_elastic_cpu" + std::to_string(NC) + "_" + std::to_string(NP);
		if (THERMAL)
		{
			long_name = "Non-isothermal ";
			short_name += "_t";
		}
		else
		{
		    long_name = "Isothermal ";
		}
		long_name += "CPU simulator engine for " + std::to_string(NC) + " components and " + std::to_string(NP) + " phases with momentum balance, diffusion and kinetic reaction";
		py::class_<engine_super_elastic_cpu<NC, NP, THERMAL>, engine_base>(m, short_name.c_str(), long_name.c_str())   \
			.def(py::init<>()) \
			.def("init", (int (engine_super_elastic_cpu<NC, NP, THERMAL>::*)
			(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_super_elastic_cpu<NC, NP, THERMAL>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>())
			.def("calc_newton_residual", &engine_super_elastic_cpu<NC, NP, THERMAL>::calc_newton_dev) \
			.def("apply_newton_update", &engine_super_elastic_cpu<NC, NP, THERMAL>::apply_newton_update) \
			.def("post_newtonloop", &engine_super_elastic_cpu<NC, NP, THERMAL>::post_newtonloop) \
			.def_readwrite("find_equilibrium", &engine_super_elastic_cpu<NC, NP, THERMAL>::FIND_EQUILIBRIUM) \
			.def_readwrite("geomechanics_mode", &engine_super_elastic_cpu<NC, NP, THERMAL>::geomechanics_mode) \
			.def_readwrite("newton_update_coefficient", &engine_super_elastic_cpu<NC, NP, THERMAL>::newton_update_coefficient) \
			.def_readwrite("dev_u", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_u) \
			.def_readwrite("dev_p", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_p) \
			.def_readwrite("dev_e", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_e) \
			.def_readwrite("dev_g", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_g) \
			//.def_readwrite("dev_z", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_z) 
			.def_readwrite("dev_u_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_u_prev) \
			.def_readwrite("dev_p_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_p_prev) \
			.def_readwrite("dev_e_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_e_prev) \
			.def_readwrite("dev_g_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_g_prev) \
			//.def_readwrite("dev_z_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_z_prev) 
			.def_readwrite("well_residual_prev_dt", &engine_super_elastic_cpu<NC, NP, THERMAL>::well_residual_prev_dt) \
			.def_readwrite("fluxes", &engine_super_elastic_cpu<NC, NP, THERMAL>::fluxes) \
			.def_readwrite("fluxes_n", &engine_super_elastic_cpu<NC, NP, THERMAL>::fluxes_n) \
			.def_readwrite("fluxes_biot", &engine_super_elastic_cpu<NC, NP, THERMAL>::fluxes_biot) \
			.def_readwrite("dX", &engine_super_elastic_cpu<NC, NP, THERMAL>::dX) \
			.def_readwrite("RHS", &engine_super_elastic_cpu<NC, NP, THERMAL>::RHS) \
			.def_readwrite("contacts", &engine_super_elastic_cpu<NC, NP, THERMAL>::contacts) \
			.def_readwrite("contact_solver", &engine_super_elastic_cpu<NC, NP, THERMAL>::contact_solver) \
			.def_readwrite("eps_vol", &engine_super_elastic_cpu<NC, NP, THERMAL>::eps_vol) \
			.def_property_readonly_static("P_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::P_VAR; }) \
			.def_property_readonly_static("Z_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::Z_VAR; }) \
			.def_property_readonly_static("P_VAR_T", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::P_VAR_T; }) \
			.def_property_readonly_static("U_VAR_T", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::U_VAR_T; }) \
			.def_property_readonly_static("U_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::U_VAR; }) \
			.def_property_readonly_static("T_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::T_VAR; }) \
			.def_property_readonly_static("N_VARS", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::N_VARS; }) \
			.def_property_readonly_static("NT", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::NT; }) \
			.def_property_readonly_static("N_OPS", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::N_OPS; }) \
			.def_property_readonly_static("NC", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::NC_; }) \
			.def_property_readonly_static("ACC_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::ACC_OP; }) \
			.def_property_readonly_static("FLUX_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::FLUX_OP; }) \
			.def_property_readonly_static("GRAV_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::GRAV_OP; })
			.def_property_readonly_static("SAT_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::SAT_OP; });
	}
};

void pybind_engine_super_elastic_cpu(py::module &m)
{
  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 1, false> re;
  re.expose(m);

  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 2, false> re1;
  re1.expose(m);

  //recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 2, MAX_NC, 3, false> re2;
  //re2.expose(m);

  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 1, true> re3;
  re3.expose(m);

  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 2, true> re4;
  re4.expose(m);

  //recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 2, MAX_NC, 3, true> re5;
  //re5.expose(m);
}

#endif //PYBIND11_ENABLED