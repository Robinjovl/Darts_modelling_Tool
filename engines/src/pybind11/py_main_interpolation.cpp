#include "py_globals_interpolation.h"
#include "py_evaluator_iface.h"
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <cstdint>
using namespace std;

typedef int index_t;
typedef double value_t;
typedef int interp_index_t;
typedef double interp_value_t;
 

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/auxiliary/timer_node.hpp"
#else
#include "timer_node.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#include <fstream>
#include <vector>

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::auxiliary;
#endif // OPENDARTS_LINEAR_SOLVERS

void pybind_operator_set_interpolator_all(py::module &);
void pybind_evaluator_iface(py::module&);
void pybind_globals(py::module&);

class operator_set_gradient_evaluator_iface;

PYBIND11_MODULE(engines_interpolator, m)
{
  m.doc() = "Delft Advanced Research Terra Simulator";
  //auto m1 = m.def_submodule("engines", "Collection of DARTS simulators based on OBL approach");
  py::bind_vector<std::vector<index_t>>(m, "index_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<index_t>& p) { // __getstate__
              py::tuple t(p.size());
              for (int i = 0; i < p.size(); i++)
                  t[i] = p[i];

              return t;
          },
          [](py::tuple t) { // __setstate__
              std::vector<index_t> p(t.size());

              for (int i = 0; i < p.size(); i++)
                  p[i] = t[i].cast<index_t>();

              //p.setExtra(t[1].cast<int>());

              return p;
          })) \
      .def("resize",
          (void (std::vector<index_t>::*) (size_t count)) & std::vector<index_t>::resize,
          "changes the number of elements stored");
  py::bind_vector<std::vector<value_t>>(m, "value_vector", py::module_local(true), py::buffer_protocol())
      .def(py::pickle(
          [](const std::vector<value_t> &p) { // __getstate__
            py::tuple t(p.size());
            for (int i = 0; i < p.size(); i++)
              t[i] = p[i];

            return t;
          },
          [](py::tuple t) { // __setstate__
            std::vector<value_t> p(t.size());

            for (int i = 0; i < p.size(); i++)
              p[i] = t[i].cast<value_t>();

            //p.setExtra(t[1].cast<int>());

            return p;
          })) \
      .def("resize",
          (void (std::vector<value_t>::*) (size_t count)) &std::vector<value_t>::resize,
          "changes the number of elements stored");
  
  py::bind_vector<std::vector<operator_set_gradient_evaluator_iface *>>(m, "op_vector");
  py::bind_map<std::map<std::string, timer_node>>(m, "timer_map");

  pybind_evaluator_iface(m);
  pybind_globals(m);
  pybind_operator_set_interpolator_all(m);


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