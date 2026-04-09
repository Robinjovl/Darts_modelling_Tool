#include "py_globals_interpolation.h"
#include "py_evaluator_iface.h"
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename T>
py::array to_numpy(std::vector<T>& vec)
{
  return py::array(
    py::buffer_info(
      vec.data(),
      sizeof(T),
      py::format_descriptor<T>::format(),
      1,
      { vec.size() },
      { sizeof(T) }
    ),
    py::cast(&vec)
  );
}

void pybind_operator_set_interpolator_all(py::module &);
void pybind_operator_set_interpolator_super(py::module &);
void pybind_operator_set_interpolator_rates(py::module &);
void pybind_operator_set_interpolator_pz_cap_gra(py::module &);
void pybind_operator_set_interpolator_pze_gra(py::module &);
void pybind_evaluator_iface(py::module&);
void pybind_globals(py::module&);

class operator_set_gradient_evaluator_iface;

PYBIND11_MODULE(interpolators, m)
{
  m.doc() = "open-DARTS interpolation library";

  py::bind_vector<std::vector<index_t>>(m, "index_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<index_t>& p) { // __getstate__
              py::tuple t(p.size());
              for (size_t i = 0; i < p.size(); i++)
                  t[i] = p[i];
              return t;
          },
          [](py::tuple t) { // __setstate__
              std::vector<index_t> p(t.size());
              for (size_t i = 0; i < p.size(); i++)
                  p[i] = t[i].cast<index_t>();
              return p;
          }))
      .def("resize",
          (void (std::vector<index_t>::*) (size_t count)) & std::vector<index_t>::resize,
          "changes the number of elements stored")
      .def("to_numpy", [](std::vector<index_t>& vec) {
          return to_numpy(vec);
      });

  py::bind_vector<std::vector<value_t>>(m, "value_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<value_t> &p) { // __getstate__
            py::tuple t(p.size());
            for (size_t i = 0; i < p.size(); i++)
              t[i] = p[i];
            return t;
          },
          [](py::tuple t) { // __setstate__
            std::vector<value_t> p(t.size());
            for (size_t i = 0; i < p.size(); i++)
              p[i] = t[i].cast<value_t>();
            return p;
          }))
      .def("resize",
          (void (std::vector<value_t>::*) (size_t count)) &std::vector<value_t>::resize,
          "changes the number of elements stored")
      .def("to_numpy", [](std::vector<value_t>& vec) {
          return to_numpy(vec);
      });

  py::bind_vector<std::vector<operator_set_gradient_evaluator_iface *>>(m, "op_vector");
  py::bind_map<std::map<std::string, timer_node>>(m, "timer_map");

  pybind_evaluator_iface(m);
  pybind_globals(m);
  pybind_operator_set_interpolator_all(m);
  pybind_operator_set_interpolator_super(m);
  pybind_operator_set_interpolator_rates(m);
  pybind_operator_set_interpolator_pz_cap_gra(m);
  pybind_operator_set_interpolator_pze_gra(m);
}

void pybind_globals(py::module& m)
{
    using namespace pybind11::literals;

    py::class_<timer_node>(m, "timer_node", "Timers tree structure")
        .def(py::init<>())
        .def("start", &timer_node::start)
        .def("stop", &timer_node::stop)
        .def("get_timer", &timer_node::get_timer)
        .def("print", &timer_node::print)
        .def("reset_recursive", &timer_node::reset_recursive)
        //properties
        .def_readwrite("node", &timer_node::node);
}
