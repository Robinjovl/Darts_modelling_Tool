#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 3: Geothermal and poroelasticity PM engine

void pybind_operator_set_interpolator_super_part3(py::module &m)
{
  const int N_DIMS_MAX = MAX_NC;

  // A = 4, B = 4 (geothermal problem, three phases)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 4, 4> e1;

  // A = 2, B = 0 (poroelasticity, pm engine)
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 2, 0> e2;

  e1.expose(m);
  e2.expose(m);
}

#endif //PYBIND11_ENABLED
