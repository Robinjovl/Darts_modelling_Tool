#include "py_globals.h"
#include "py_evaluator_iface.h"

namespace py = pybind11;

void pybind_evaluator_iface(py::module &m)
{
  using namespace pybind11::literals;

  py::class_<property_evaluator_iface, py_property_evaluator_iface /* <--- trampoline*/> property_evaluator_iface(m, "property_evaluator_iface");
  property_evaluator_iface
      .def(py::init<>())
      //.def("evaluate", &property_evaluator_iface::evaluate, "Evaluate property value", "state"_a);
      .def("evaluate", (value_t(property_evaluator_iface::*)(const std::vector<value_t> &)) & property_evaluator_iface::evaluate, "Evaluate property value", "state"_a)
      .def("evaluate", (int (property_evaluator_iface::*)(const std::vector<value_t> &, index_t, std::vector<value_t> &)) & property_evaluator_iface::evaluate, "Evaluate property values", "states"_a, "n_blocks"_a, "values"_a);

  py::class_<operator_set_evaluator_iface, py_operator_set_evaluator_iface /* <--- trampoline*/> operator_set_evaluator_iface(m, "operator_set_evaluator_iface");
  operator_set_evaluator_iface
      .def(py::init<>())
      .def("evaluate", &operator_set_evaluator_iface::evaluate, "Evaluate operator values", "states"_a, "values"_a)
      .def("evaluate_batch", &operator_set_evaluator_iface::evaluate_batch,
           "Batch-evaluate operator values for multiple points",
           "states"_a, "n_points"_a, "values"_a, "n_ops"_a);

  py::class_<operator_set_gradient_evaluator_iface>(m, "operator_set_gradient_evaluator_iface", operator_set_evaluator_iface);
}
