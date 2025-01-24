#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_pze_gra(py::module &m)
{
  // nce, grav : n_ops = (1 + 2) * (N_DIMS - 1) + 6 + 2 + 1 = 3 * N_DIMS + 4

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = 2;
  
  // N_OPS = A * N_DIMS + B
  const int A = 3;
  const int B = 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e;
 
  e.expose(m);
}

#endif //PYBIND11_ENABLED