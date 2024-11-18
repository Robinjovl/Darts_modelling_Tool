#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "engine_nc_nonnewton_cpu.hpp"
#include "conn_mesh.h"

template <uint8_t NC, uint8_t NP>
struct engine_nc_nonnewton_exposer
{
  static void expose(py::module &m)
  {
    py::class_<engine_nc_nonnewton_cpu<NC, NP>, engine_base>(m, ("engine_nc_nonnewton_cpu" + std::to_string(NC) + "_" + std::to_string(NP)).c_str(),
                                                          ("Isothermal CPU simulator engine for " + std::to_string(NC) + " components and " + std::to_string(NP) + " phases with nonnewtonian property").c_str())  \
      .def(py::init<>())  \
      .def("init", (int (engine_nc_nonnewton_cpu<NC, NP>::*)(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_nc_nonnewton_cpu<NC, NP>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>());
  }
};

void pybind_engine_nc_nonnewton_cpu(py::module &m)
{
  // 2 phase compositional
  recursive_exposer_nc_np<engine_nc_nonnewton_exposer, py::module, 2, MAX_NC, 2> re;
  // 3 phase black oil
  recursive_exposer_nc_np<engine_nc_nonnewton_exposer, py::module, 3, 3, 3> re1;

  re.expose(m);
  re1.expose(m);
}


#endif //PYBIND11_ENABLED