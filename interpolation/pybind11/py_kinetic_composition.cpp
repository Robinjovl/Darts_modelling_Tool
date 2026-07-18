#include "py_globals_interpolation.h"
#include "kinetic_composition_cpu_interpolator.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * Expose the nested-kinetics composition wrapper. Non-template (runtime n_dims/n_ops),
 * so a single class serves every dimensionality — no 2^d instantiation growth.
 *
 * List arguments are converted manually because std::vector<value_t>/<index_t> are
 * opaque in this module (bind_vector) which disables automatic list conversion.
 */
void pybind_kinetic_composition(py::module &m)
{
  using namespace pybind11::literals;

  py::class_<kinetic_composition_cpu_interpolator, operator_set_gradient_evaluator_iface>(
      m, "kinetic_composition_cpu_interpolator",
      "Nested-kinetics wrapper: composes sharp KIN operators analytically (exact "
      "chain-rule derivatives) from smooth fields tabulated in the inner interpolator")
      .def(py::init([](operator_set_gradient_evaluator_iface *inner, int n_dims, int n_ops,
                       int kin_op_start, int ne, py::sequence mineral_axes, py::sequence c_coeffs,
                       py::sequence p_aff, py::sequence q_aff, py::sequence stoich_flat) {
             std::vector<int> ax;
             for (auto h : mineral_axes)
               ax.push_back(h.cast<int>());
             auto to_vd = [](py::sequence s) {
               std::vector<double> v;
               for (auto h : s)
                 v.push_back(h.cast<double>());
               return v;
             };
             return new kinetic_composition_cpu_interpolator(
                 inner, n_dims, n_ops, kin_op_start, ne, ax, to_vd(c_coeffs), to_vd(p_aff),
                 to_vd(q_aff), to_vd(stoich_flat));
           }),
           "inner"_a, "n_dims"_a, "n_ops"_a, "kin_op_start"_a, "ne"_a, "mineral_axes"_a,
           "c_coeffs"_a, "p_aff"_a, "q_aff"_a, "stoich_flat"_a,
           py::keep_alive<1, 2>() /* wrapper keeps inner alive */)
      .def("evaluate", &kinetic_composition_cpu_interpolator::evaluate,
           "Evaluate composed operator values for a single state", "state"_a, "values"_a)
      .def("evaluate_with_derivatives",
           &kinetic_composition_cpu_interpolator::evaluate_with_derivatives,
           "Interpolate fields and compose KIN values + exact gradients",
           "states"_a, "states_idxs"_a, "values"_a, "derivatives"_a)
      .def_property_readonly(
          "inner",
          [](kinetic_composition_cpu_interpolator &self) { return self.inner; },
          py::return_value_policy::reference_internal);
}
