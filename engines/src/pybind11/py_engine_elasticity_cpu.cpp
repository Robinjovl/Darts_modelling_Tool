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
#include "mech/engine_elasticity_cpu.hpp"
#include "conn_mesh.h"

template <uint8_t ND> 
struct engine_elasticity_exposer
{
  static void expose(py::module &m)
  {
    py::class_<engine_elasticity_cpu<ND>, engine_base>(m, ("engine_elasticity_cpu" + std::to_string(ND)).c_str(), (std::to_string(ND) + "D elastic mechanics CPU engine").c_str())  \
      .def(py::init<>()) \
      .def("init", (int (engine_elasticity_cpu<ND>::*)(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_elasticity_cpu<ND>::init, "Initialize simulator by mesh and params", py::keep_alive<1, 5>()) \
	  .def("write_matrix", &engine_elasticity_cpu<ND>::write_matrix) \
	  .def_readwrite("RHS", &engine_elasticity_cpu<ND>::RHS) \
	  .def_readwrite("use_calculated_flux", &engine_elasticity_cpu<ND>::USE_CALCULATED_FLUX) \
	  .def_readwrite("fluxes", &engine_elasticity_cpu<ND>::fluxes) \
	  .def_readwrite("newton_update_coefficient", &engine_elasticity_cpu<ND>::newton_update_coefficient);
  }
};

void pybind_engine_elasticity_cpu(py::module& m)
{
    engine_elasticity_exposer<3> e3;
    e3.expose(m);
};

#endif //PYBIND11_ENABLED