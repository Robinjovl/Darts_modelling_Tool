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

  // single comp
  const int A0 = 0;
  const int B0 = 1;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A0, B0> e0;
  
  // N_OPS = A * N_DIMS + B

  // single phase
  const int A1 = 0;
  const int B1 = 4 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;

  // two phase
  const int A2 = 0;
  const int B2 = 8 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A2, B2> e2;

  // three phase
  const int A3 = 0;
  const int B3 = 12 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A3, B3> e3;

  // four phase
  const int A4 = 0;
  const int B4 = 16 + 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A4, B4> e4;

  e0.expose(m);
  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
  e4.expose(m);
}

#endif //PYBIND11_ENABLED