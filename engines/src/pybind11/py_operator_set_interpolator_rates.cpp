#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_rates(py::module &m)
{
  // rates, 2 phases: N_OPS = 8

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;
  
  // N_OPS = N_DIMS + 1P * 4 ops
  const int B = 1 * 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 1, B> e;

  // N_OPS = N_DIMS + 2P * 4 ops
  const int B1 = 2 * 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 1, B1> e1;
 

  //// N_OPS = N_DIMS + 3P * 4 ops
  const int B2 = 3 * 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 1, B2> e2;

  //// N_OPS = N_DIMS + 4P * 4 ops
  const int B3 = 4 * 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 1, B3> e3;

  e.expose(m);
  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
}

#endif //PYBIND11_ENABLED