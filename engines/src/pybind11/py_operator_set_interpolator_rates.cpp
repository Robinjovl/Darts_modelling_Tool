#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_rates(py::module &m)
{
  // rates, 2 phases: N_OPS = 2

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;
  
  // N_OPS = A * N_DIMS + B
  const int A = 0;
  const int B = 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e;

  // rates, 2 phases: N_OPS = 2

  // N_OPS = A * N_DIMS + B
  const int A1 = 0;
  const int B1 = 3;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;
 

  //// N_OPS = A * N_DIMS + B
  const int A2 = 0;
  const int B2 = 1;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A2, B2> e2;

  //// N_OPS = A * N_DIMS + B
  const int A3 = 0;
  const int B3 = 3;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A3, B3> e3;

  //// N_OPS = A * N_DIMS + B
  const int A4 = 0;
  const int B4 = 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A4, B4> e4;

  e.expose(m);
  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
  e4.expose(m);
}

#endif //PYBIND11_ENABLED