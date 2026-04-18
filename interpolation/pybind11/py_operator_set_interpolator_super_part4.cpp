#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 4: engine_super_elastic_*
// N_OPS = (2 * NP + 2) * NC + 7 * NP + 4

void pybind_operator_set_interpolator_super_part4(py::module &m)
{
  const int N_DIMS_MAX = MAX_DIMS;

  // NP = 1: A = 4, B = 11 (poroelasticity, single-phase)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 4, 11> e1;

  // NP = 2: A = 6, B = 18 (poroelasticity, two-phase)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 6, 18> e2;

  e1.expose(m);
  e2.expose(m);
}

#endif //PYBIND11_ENABLED
