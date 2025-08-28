#ifdef PYBIND11_ENABLED

#include "py_globals.h"
#include "ms_well.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void pybind_ms_well(py::module &m)

{
  using namespace pybind11::literals;

  py::class_<ms_well>(m, "ms_well", "Multisegment well, modeled as an extension of the reservoir")
    .def(py::init<>())
    //methods
    .def("init_rate_parameters", &ms_well::init_rate_parameters, "Init by NC and rate operators", 
        "n_vars"_a, "n_ops"_a, "phase_names"_a, "rate_ev"_a, "well_init_ev"_a, "thermal"_a = 0, py::keep_alive<1, 6>())
	  .def("init_mech_rate_parameters", &ms_well::init_mech_rate_parameters, "Init by NC and rate operators for poromechanics",
		    "N_VARS"_a, "P_VAR"_a, "n_vars"_a, "n_ops"_a, "phase_names"_a, "rate_ev"_a, "well_init_ev"_a, "thermal"_a = 0, py::keep_alive<1, 8>())
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
    .def_readwrite("control", &ms_well::control)
    .def_readwrite("constraint", &ms_well::constraint)
    .def_readonly("well_body_idx", &ms_well::well_body_idx)
    .def_readonly("well_head_idx", &ms_well::well_head_idx)
    .def("set_bhp_control", &ms_well::set_bhp_control)
    .def("set_bhp_constraint", &ms_well::set_bhp_constraint)
    .def("set_rate_control", &ms_well::set_rate_control)
    .def("set_rate_constraint", &ms_well::set_rate_constraint);
}
#endif //PYBIND11_ENABLED