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
        "n_vars"_a, "n_ops"_a, "phase_names"_a, "rate_ev"_a, "thermal"_a = 0, py::keep_alive<1, 5>())
	.def("init_mech_rate_parameters", &ms_well::init_mech_rate_parameters, "Init by NC and rate operators for poromechanics",
		"N_VARS"_a, "P_VAR"_a, "n_vars"_a, "n_ops"_a, "phase_names"_a, "rate_ev"_a, "thermal"_a = 0, py::keep_alive<1, 7>())
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
    .def_readonly("well_body_idx", &ms_well::well_body_idx)
    .def_readonly("well_head_idx", &ms_well::well_head_idx)
    .def_property("control",
                  [](ms_well &self) { return self.control; },
                  py::cpp_function([](ms_well &self, well_control_iface *control_) { self.control = control_; }, py::keep_alive<1, 2>()))
    .def_property("constraint",
                [](ms_well &self) { return self.constraint; },
                py::cpp_function([](ms_well &self, well_control_iface *constraint_) { self.constraint = constraint_; }, py::keep_alive<1, 2>()));
}
#endif //PYBIND11_ENABLED