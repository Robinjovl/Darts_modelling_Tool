#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "engine_super_cpu.hpp"
#include "conn_mesh.h"


template <uint8_t NC, uint8_t NP, bool THERMAL>
struct engine_super_exposer
{
	static void expose(py::module &m)
	{
    std::string short_name, long_name;
    short_name = "engine_super_cpu" + std::to_string(NC) + "_" + std::to_string(NP);
    if (THERMAL)
    {
      long_name = "Isothermal ";
      short_name += "_t";
    }
    else
    {
      long_name = "Non-isothermal ";
    }
    long_name += "CPU simulator engine for " + std::to_string(NC) + " components and " + std::to_string(NP) + " phases with diffusion and kinetic reaction";
    py::class_<engine_super_cpu<NC, NP, THERMAL>, engine_base>(m, short_name.c_str(), long_name.c_str())   \
      .def(py::init<>()) \
      .def("init", (int (engine_super_cpu<NC, NP, THERMAL>::*)(conn_mesh*, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) & engine_super_cpu<NC, NP, THERMAL>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>());
	}
};


void pybind_engine_super_cpu(py::module &m)
{
  recursive_exposer_nc_np_t<engine_super_exposer, py::module, 1, MAX_NC, 1, false> re;
  re.expose(m);

  recursive_exposer_nc_np_t<engine_super_exposer, py::module, 2, MAX_NC, 2, false> re1;
  re1.expose(m);

  recursive_exposer_nc_np_t<engine_super_exposer, py::module, 2, MAX_NC, 3, false> re2;
  re2.expose(m);

  recursive_exposer_nc_np_t<engine_super_exposer, py::module, 1, MAX_NC, 1, true> re3;
  re3.expose(m);

  recursive_exposer_nc_np_t<engine_super_exposer, py::module, 1, MAX_NC, 2, true> re4;
  re4.expose(m);

  recursive_exposer_nc_np_t<engine_super_exposer, py::module, 2, MAX_NC, 3, true> re5;
  re5.expose(m);
}

#endif //PYBIND11_ENABLED