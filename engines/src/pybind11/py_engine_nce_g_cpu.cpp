#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "conn_mesh.h"
#include "engine_nce_g_cpu.hpp"
#ifdef WITH_GPU
#include "engine_nce_g_gpu.hpp"
#endif

template <uint8_t NC, uint8_t NP>
struct engine_nce_g_exposer
{
  static void expose(py::module &m)
  {
    py::class_<engine_nce_g_cpu<NC, NP>, engine_base>(m, ("engine_nce_g_cpu" + std::to_string(NC) + "_" + std::to_string(NP)).c_str(),
      ("Thermal enthalpy-based CPU simulator engine class for " + std::to_string(NC) + " components " + std::to_string(NP) + " phases with gravity").c_str())
      .def(py::init<>())
      .def("init", (int (engine_nce_g_cpu<NC, NP>::*)(conn_mesh*, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) & engine_nce_g_cpu<NC, NP>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>());

#ifdef WITH_GPU
    py::class_<engine_nce_g_gpu<NC, NP>, engine_base>(m, ("engine_nce_g_gpu" + std::to_string(NC) + "_" + std::to_string(NP)).c_str(),
                                                      ("Thermal enthalpy-based GPU simulator engine class for " + std::to_string(NC) + " components " + std::to_string(NP) + " phases with gravity").c_str())
        .def(py::init<>())
        .def("init", (int (engine_nce_g_gpu<NC, NP>::*)(conn_mesh *, std::vector<ms_well *> &, std::vector<operator_set_gradient_evaluator_iface *> &, sim_params *, timer_node *)) & engine_nce_g_gpu<NC, NP>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>())
        .def("get_X_d", [](const engine_nce_g_gpu<NC, NP>& self) -> py::capsule {
            return py::capsule(self.X_d, "double_ptr");
        }) \
        .def("get_RHS_d", [](const engine_nce_g_gpu<NC, NP>& self) -> py::capsule {
            return py::capsule(self.RHS_d, "double_ptr");
        });
#endif
  }
};

void pybind_engine_nce_g_cpu(py::module &m)
{
  engine_nce_g_exposer<1, 2> exposer;
  exposer.expose(m);
}

#endif //PYBIND11_ENABLED