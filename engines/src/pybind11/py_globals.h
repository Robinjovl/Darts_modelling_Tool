#ifndef PY_GLOBALS_H
#define PY_GLOBALS_H

#include <pybind11/stl_bind.h>
#include "globals.h"
#include "ms_well.h"
#include "py_globals_interpolation.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<ms_well*>);

PYBIND11_MAKE_OPAQUE(std::vector<linear_solver_params>);
//PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string,timer_node>);


#endif





