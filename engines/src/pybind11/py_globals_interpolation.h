#ifndef PY_GLOBALS_INTERPOLATION_H
#define PY_GLOBALS_INTERPOLATION_H

#include <pybind11/stl_bind.h>
#include "globals.h"

#include <vector>
#include <tuple>
#include <unordered_map>

#include "evaluator_iface.h"

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<index_t>);
PYBIND11_MAKE_OPAQUE(std::vector<value_t>);

PYBIND11_MAKE_OPAQUE(std::vector<operator_set_gradient_evaluator_iface*>);


#endif









