#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "engine_nc_nl_cpu.hpp"
#include "conn_mesh.h"


template <uint8_t NC> 
struct engine_nc_nl_exposer
{
  static void expose(py::module &m)
  {
	  py::class_<engine_nc_nl_cpu<NC>, engine_base>(m, ("engine_nc_nl_cpu" + std::to_string(NC)).c_str(), ("Isothermal CPU multipoint simulator engine for " + std::to_string(NC) + " components with non-linear discretization").c_str())  \
		  .def(py::init<>()) \
		  .def("init", (int (engine_nc_nl_cpu<NC>::*)(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_nc_nl_cpu<NC>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>()) \
		  .def_readwrite("appr_mode", &engine_nc_nl_cpu<NC>::appr_mode) \
		  .def_property_readonly_static("P_VAR", [](py::object) { return engine_nc_nl_cpu<NC>::P_VAR; });
  }
};

void pybind_engine_nc_nl_cpu (py::module &m)
{
  recursive_exposer_nc<engine_nc_nl_exposer, py::module, 2, MAX_NC> re;
  re.expose(m);
}

#endif //PYBIND11_ENABLED