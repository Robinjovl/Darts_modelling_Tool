#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_pz(py::module &m)
{
  // nc, no grav : N_OPS = 2 * N_DIMS + 0

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;

  // N_OPS = A * N_DIMS + B
  const int A = 2;
  const int B = 0;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e;
  // nc, no grav : n_ops = 2 * N_DIMS + 1  totalvelocty
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 2, 1> e8;
  e8.expose(m);

  e.expose(m);
}

#endif //PYBIND11_ENABLED