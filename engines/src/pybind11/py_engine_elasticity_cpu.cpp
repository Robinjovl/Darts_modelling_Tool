#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "mech/engine_elasticity_cpu.hpp"
#include "conn_mesh.h"

template <uint8_t ND> 
struct engine_elasticity_exposer
{
  static void expose(py::module &m)
  {
    py::class_<engine_elasticity_cpu<ND>, engine_base>(m, ("engine_elasticity_cpu" + std::to_string(ND)).c_str(), (std::to_string(ND) + "D elastic mechanics CPU engine").c_str())  \
      .def(py::init<>()) \
      .def("init", (int (engine_elasticity_cpu<ND>::*)(conn_mesh*, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) & engine_elasticity_cpu<ND>::init, "Initialize simulator by mesh and params", py::keep_alive<1, 5>()) \
      .def("write_matrix", &engine_elasticity_cpu<ND>::write_matrix) \
      .def_readwrite("RHS", &engine_elasticity_cpu<ND>::RHS) \
      .def_readwrite("use_calculated_flux", &engine_elasticity_cpu<ND>::USE_CALCULATED_FLUX) \
      .def_readwrite("fluxes", &engine_elasticity_cpu<ND>::fluxes);
  }
};

void pybind_engine_elasticity_cpu(py::module& m)
{
    engine_elasticity_exposer<3> e3;
    e3.expose(m);
};

#endif //PYBIND11_ENABLED