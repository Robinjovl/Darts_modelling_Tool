#ifdef PYBIND11_ENABLED

#include "py_globals.h"
#include "ms_well.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void pybind_ms_well(py::module& m)
{
    using namespace pybind11::literals;

    py::class_<ms_well> ms_well_class(
        m, "ms_well",
        "Multisegment well, modeled as an extension of the reservoir");

    ms_well_class
        .def(py::init<>())
        // methods
        .def("init_rate_parameters", &ms_well::init_rate_parameters,
            "Init by NC and rate operators", "n_vars"_a, "n_ops"_a,
            "phase_names"_a, "well_controls_etor"_a, "thermal_var_etor"_a, "thermal"_a = 0,
            py::keep_alive<1, 6>())
        .def("init_mech_rate_parameters", &ms_well::init_mech_rate_parameters,
            "Init by NC and rate operators for poromechanics", "N_VARS"_a,
            "P_VAR"_a, "n_vars"_a, "n_ops"_a, "phase_names"_a, "well_controls_etor"_a,
            "thermal_var_etor"_a, "thermal"_a = 0, py::keep_alive<1, 8>())
        // properties
        .def_readwrite("name", &ms_well::name)

        .def_readwrite("perforations", &ms_well::perforations)
        .def_readwrite("segment_volume", &ms_well::segment_volume)
        .def_readwrite("well_transmissibility", &ms_well::well_transmissibility)
        .def_readwrite("well_head_depth", &ms_well::well_head_depth)
        .def_readwrite("well_body_depth", &ms_well::well_body_depth)
        .def_readwrite("segment_depth_increment",
            &ms_well::segment_depth_increment)
        .def_readwrite("segment_diameter", &ms_well::segment_diameter)
        .def_readwrite("segment_roughness", &ms_well::segment_roughness)
        .def_readwrite("control", &ms_well::control)
        .def_readwrite("constraint", &ms_well::constraint)
        .def_readonly("well_head_idx", &ms_well::well_head_idx)
        .def_readonly("well_body_idx", &ms_well::well_body_idx)
        .def_readonly("well_bottom_idx", &ms_well::well_bottom_idx)
        .def("set_bhp_control", &ms_well::set_bhp_control)
        .def("set_bhp_constraint", &ms_well::set_bhp_constraint)
        .def("set_rate_control", &ms_well::set_rate_control)
        .def("set_rate_constraint", &ms_well::set_rate_constraint)

        // only used for DFM wells
        .def_readwrite("ms_type", &ms_well::ms_type)
        .def_readwrite("segment_volumes", &ms_well::segment_volumes)
        .def_readwrite("segment_depths", &ms_well::segment_depths)
        .def_readwrite("num_segments", &ms_well::num_segments)
        .def_readwrite("init_state", &ms_well::init_state)
        .def_readwrite("phases_vels", &ms_well::phases_vels)
        .def_readwrite("phases_vels_ders", &ms_well::phases_vels_ders)
        // For lateral heat transfer in DFM wells
        .def_readwrite("with_lateral_heat_transfer", &ms_well::with_lateral_heat_transfer)
        .def_readwrite("connections_for_lateral_heat_transfer", &ms_well::connections_for_lateral_heat_transfer)
        .def_readwrite("Xhis_well_default", &ms_well::Xhis_well_default);


    py::enum_<ms_well::MS_Type>(ms_well_class, "MS_Type")
        .value("EPM", ms_well::MS_Type::EPM)
        .value("DFM", ms_well::MS_Type::DFM)
        .export_values();
}
#endif // PYBIND11_ENABLED
