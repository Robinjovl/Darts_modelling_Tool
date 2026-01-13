#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 1: Single component and single phase

void pybind_operator_set_interpolator_rates_part1(py::module &m)
{
  const int N_DIMS_MAX = MAX_NC;

  // single component: N_OPS = 1
  const int A0 = 0;
  const int B0 = 1;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A0, B0> e0;

  // single phase: N_OPS = 6
  const int A1 = 0;
  const int B1 = 4 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;

  e0.expose(m);
  e1.expose(m);
}

#endif //PYBIND11_ENABLED
