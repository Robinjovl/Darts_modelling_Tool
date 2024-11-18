#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_pz_cap_gra(py::module &m)
{
  // nc, grav + pc, 2 phases  : n_ops = (1 + 2) * N_DIMS + 2 + 2

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;
  
  // N_OPS = A * N_DIMS + B
  const int A = 3;
  const int B = 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e;

  // nc, grav + pc, 3 phases  : n_ops = (1 + 3) * N_DIMS + 3 + 3

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX1 = 3; // used only for black-oil

  // N_OPS = A * N_DIMS + B
  const int A1 = 4;
  const int B1 = 6;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX1, A1, B1> e1;

 
  e.expose(m);
  e1.expose(m);
}

#endif //PYBIND11_ENABLED