#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 2: Two phase and three phase

void pybind_operator_set_interpolator_rates_part2(py::module &m)
{
  const int N_DIMS_MAX = MAX_DIMS;

  // two phase: N_OPS = 10
  const int A2 = 0;
  const int B2 = 8 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A2, B2> e2;

  // three phase: N_OPS = 14
  const int A3 = 0;
  const int B3 = 12 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A3, B3> e3;

  e2.expose(m);
  e3.expose(m);
}

#endif //PYBIND11_ENABLED
