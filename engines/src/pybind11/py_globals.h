#ifndef PY_GLOBALS_H
#define PY_GLOBALS_H

#include <stl_bind.h>
#include "globals.h"
#include "ms_well.h"

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<index_t>);
PYBIND11_MAKE_OPAQUE(std::vector<value_t>);
PYBIND11_MAKE_OPAQUE(std::vector<ms_well*>);
PYBIND11_MAKE_OPAQUE(std::vector<operator_set_gradient_evaluator_iface*>);
PYBIND11_MAKE_OPAQUE(std::vector<linear_solver_params>);
//PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string,timer_node>);


#endif





