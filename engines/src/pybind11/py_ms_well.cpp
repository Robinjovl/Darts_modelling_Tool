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

#include "py_globals.h"
#include "ms_well.h"
#include "stl.h"

namespace py = pybind11;

void pybind_ms_well(py::module &m)

{
  using namespace pybind11::literals;

  py::class_<ms_well>(m, "ms_well", "Multisegment well, modeled as an extension of the reservoir")
    .def(py::init<>())
    //methods
    .def("init_rate_parameters", &ms_well::init_rate_parameters, "Init by NC and rate operators", 
         "n_vars"_a, "phase_names"_a, "rate_ev"_a, "thermal"_a = 0, py::keep_alive<1, 4>())
	.def("init_mech_rate_parameters", &ms_well::init_mech_rate_parameters, "Init by NC and rate operators for poromechanics",
		"N_VARS"_a, "P_VAR"_a, "n_vars"_a, "phase_names"_a, "rate_ev"_a, "thermal"_a = 0, py::keep_alive<1, 6>())
    //properties
    .def_readwrite("name", &ms_well::name)
    .def_readwrite("perforations", &ms_well::perforations)
    .def_readwrite("segment_volume", &ms_well::segment_volume)
    .def_readwrite("segment_transmissibility", &ms_well::segment_transmissibility)
    .def_readwrite("well_head_depth", &ms_well::well_head_depth)
    .def_readwrite("well_body_depth", &ms_well::well_body_depth)
    .def_readwrite("segment_depth_increment", &ms_well::segment_depth_increment)
    .def_readwrite("segment_diameter", &ms_well::segment_diameter)
    .def_readwrite("segment_roughness", &ms_well::segment_roughness)
    .def_property("control",
                  [](ms_well &self) { return self.control; },
                  py::cpp_function([](ms_well &self, well_control_iface *control_) { self.control = control_; }, py::keep_alive<1, 2>()))
    .def_property("constraint",
                [](ms_well &self) { return self.constraint; },
                py::cpp_function([](ms_well &self, well_control_iface *constraint_) { self.constraint = constraint_; }, py::keep_alive<1, 2>()));
}
#endif //PYBIND11_ENABLED