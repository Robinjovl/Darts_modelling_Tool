#ifdef PYBIND11_ENABLED
#include "py_globals.h"
#include "pybind11/stl.h"
#include <memory>
#include <stdexcept>
#include "ms_well.h"

namespace py = pybind11;

// Function to convert the `void` pointer to `py::object`
py::object &get_py_object(std::shared_ptr<void> ptr) 
{
  if (!ptr) 
  {
    throw std::runtime_error("Tried to use an unset py object.");
  }

  return *static_pointer_cast<py::object>(ptr);
}

tuple<vector<value_t>, vector<value_t>>
ms_well::evaluate_phase_velocities_and_derivatives(vector<value_t> Xn_ms_well, vector<value_t> X_ms_well, value_t dt)
{
  py::gil_scoped_acquire gil;  // Acquire the GIL

  // method evaluate_phase_velocities_and_derivatives of the Python object returns the velocities of the two phases and derivatives of velocities of  the two phases in the wellbore
  py::object result = get_py_object(velocity_evaluator).attr("evaluate_phase_velocities_and_derivatives")(Xn_ms_well, X_ms_well, dt);
  //// convert the py::object into a C++ tuple
  auto result_tuple = result.cast<std::tuple<std::vector<value_t>, std::vector<value_t>>>();

  return result_tuple;
}

void pybind_ms_well(py::module &m)

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
           "phase_names"_a, "rate_ev"_a, "thermal"_a = 0,
           py::keep_alive<1, 5>())
      .def("init_mech_rate_parameters", &ms_well::init_mech_rate_parameters,
           "Init by NC and rate operators for poromechanics", "N_VARS"_a,
           "P_VAR"_a, "n_vars"_a, "n_ops"_a, "phase_names"_a, "rate_ev"_a,
           "thermal"_a = 0, py::keep_alive<1, 7>())
      // properties
      .def_readwrite("name", &ms_well::name)
      .def_readwrite("ms_type", &ms_well::ms_type)
      .def_readwrite("segments_volumes", &ms_well::segments_volumes)
      .def_readwrite("segments_depths", &ms_well::segments_depths)
      .def("set_velocity_evaluator", [](ms_well &w, py::object evaluator) {w.velocity_evaluator = make_shared<py::object>(evaluator);})
      //.def_readwrite("velocity_evaluator", &ms_well::velocity_evaluator)
      .def_readwrite("num_segments", &ms_well::num_segments)
      .def_readwrite("perforations", &ms_well::perforations)
      .def_readwrite("segment_volume", &ms_well::segment_volume)
      .def_readwrite("well_transmissibility", &ms_well::well_transmissibility)
      .def_readwrite("well_head_depth", &ms_well::well_head_depth)
      .def_readwrite("well_body_depth", &ms_well::well_body_depth)
      .def_readwrite("segment_depth_increment",
                     &ms_well::segment_depth_increment)
      .def_readwrite("segment_diameter", &ms_well::segment_diameter)
      .def_readwrite("segment_roughness", &ms_well::segment_roughness)
      .def_readonly("well_body_idx", &ms_well::well_body_idx)
      .def_readonly("well_head_idx", &ms_well::well_head_idx)
      .def_property(
          "control", [](ms_well &self) { return self.control; },
          py::cpp_function(
              [](ms_well &self, well_control_iface *control_) {
                self.control = control_;
              },
              py::keep_alive<1, 2>()))
      .def_property(
          "constraint", [](ms_well &self) { return self.constraint; },
          py::cpp_function(
              [](ms_well &self, well_control_iface *constraint_) {
                self.constraint = constraint_;
              },
              py::keep_alive<1, 2>()));

  py::enum_<ms_well::MS_Type>(ms_well_class, "MS_Type")
      .value("EPM", ms_well::MS_Type::EPM)
      .value("DFM", ms_well::MS_Type::DFM)
      .export_values();
}
#endif // PYBIND11_ENABLED
