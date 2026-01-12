#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 3: Four phase

void pybind_operator_set_interpolator_rates_part3(py::module &m)
{
  const int N_DIMS_MAX = MAX_NC;

  // four phase: N_OPS = 18
  const int A4 = 0;
  const int B4 = 16 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A4, B4> e4;

  e4.expose(m);
}

#endif //PYBIND11_ENABLED
