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
#include "engine_nc_mp_cpu.hpp"
#include "conn_mesh.h"


template <uint8_t NC> 
struct engine_nc_mp_exposer
{
  static void expose(py::module &m)
  {
	  py::class_<engine_nc_mp_cpu<NC>, engine_base>(m, ("engine_nc_mp_cpu" + std::to_string(NC)).c_str(), ("Isothermal CPU multipoint simulator engine for " + std::to_string(NC) + " components").c_str())  \
		  .def(py::init<>()) \
		  .def("init", (int (engine_nc_mp_cpu<NC>::*)(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_nc_mp_cpu<NC>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>()) \
		  .def_property_readonly_static("P_VAR",[](py::object){return engine_nc_mp_cpu<NC>::P_VAR; }) \
		  .def_readwrite("two_point_assembly", &engine_nc_mp_cpu<NC>::TWO_POINT_RES_ASSEMBLY) \
		  .def_readwrite("use_calculated_flux", &engine_nc_mp_cpu<NC>::USE_CALCULATED_FLUX);
  }
};

void pybind_engine_nc_mp_cpu (py::module &m)
{
  recursive_exposer_nc<engine_nc_mp_exposer, py::module, 2, MAX_NC> re;
  re.expose(m);
}

#endif //PYBIND11_ENABLED