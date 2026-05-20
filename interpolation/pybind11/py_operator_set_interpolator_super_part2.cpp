#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 2: Thermal three-phase and four-phase
// N_OPS = (2 * NP + 2) * NC + 7 * NP + 3

void pybind_operator_set_interpolator_super_part2(py::module &m)
{
  const int N_DIMS_MAX = MAX_DIMS;

  // NP = 3: A = 8, B = 24 (thermal three-phase)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 8, 24> e1;

  // NP = 4: A = 10, B = 31 (isothermal four-phase)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 10, 31> e2;

  e1.expose(m);
  e2.expose(m);
}

#endif //PYBIND11_ENABLED
