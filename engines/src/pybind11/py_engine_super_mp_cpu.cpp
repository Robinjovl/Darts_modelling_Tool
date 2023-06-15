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
#include "engine_super_mp_cpu.hpp"
#include "conn_mesh.h"


template <uint8_t NC, uint8_t NP, bool THERMAL>
struct engine_super_mp_exposer
{
	static void expose(py::module &m)
	{
    std::string short_name, long_name;
    short_name = "engine_super_mp_cpu" + std::to_string(NC) + "_" + std::to_string(NP);
    if (THERMAL)
    {
      long_name = "Isothermal ";
      short_name += "_t";
    }
    else
    {
      long_name = "Non-isothermal ";
    }
    long_name += "CPU simulator engine for " + std::to_string(NC) + " components and " + std::to_string(NP) + " phases with diffusion and kinetic reaction";
	py::class_<engine_super_mp_cpu<NC, NP, THERMAL>, engine_base>(m, short_name.c_str(), long_name.c_str())   \
		.def(py::init<>()) \
		.def("init", (int (engine_super_mp_cpu<NC, NP, THERMAL>::*)(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_super_mp_cpu<NC, NP, THERMAL>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>())
		.def("run_single_newton_iteration", &engine_super_mp_cpu<NC, NP, THERMAL>::run_single_newton_iteration) \
		.def_readwrite("fluxes", &engine_super_mp_cpu<NC, NP, THERMAL>::fluxes) \
		.def_readwrite("dX", &engine_super_mp_cpu<NC, NP, THERMAL>::dX) \
		.def_readwrite("RHS", &engine_super_mp_cpu<NC, NP, THERMAL>::RHS) \
		.def_property_readonly_static("P_VAR", [](py::object) {return engine_super_mp_cpu<NC, NP, THERMAL>::P_VAR; }) \
		.def_property_readonly_static("Z_VAR", [](py::object) {return engine_super_mp_cpu<NC, NP, THERMAL>::Z_VAR; }) \
		.def_property_readonly_static("T_VAR", [](py::object) {return engine_super_mp_cpu<NC, NP, THERMAL>::T_VAR; }) \
		.def_property_readonly_static("NC", [](py::object) {return engine_super_mp_cpu<NC, NP, THERMAL>::NC_; });
	}
};

void pybind_engine_super_mp_cpu(py::module &m)
{
  recursive_exposer_nc_np_t<engine_super_mp_exposer, py::module, 2, MAX_NC, 1, false> re;
  re.expose(m);

  recursive_exposer_nc_np_t<engine_super_mp_exposer, py::module, 2, MAX_NC, 2, false> re1;
  re1.expose(m);

  recursive_exposer_nc_np_t<engine_super_mp_exposer, py::module, 2, MAX_NC, 3, false> re2;
  re2.expose(m);

  recursive_exposer_nc_np_t<engine_super_mp_exposer, py::module, 1, MAX_NC, 1, true> re3;
  re3.expose(m);

  recursive_exposer_nc_np_t<engine_super_mp_exposer, py::module, 1, MAX_NC, 2, true> re4;
  re4.expose(m);

  recursive_exposer_nc_np_t<engine_super_mp_exposer, py::module, 2, MAX_NC, 3, true> re5;
  re5.expose(m);
}

#endif //PYBIND11_ENABLED