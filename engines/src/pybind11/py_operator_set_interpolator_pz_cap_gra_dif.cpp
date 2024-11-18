#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_pz_cap_gra_dif(py::module &m)
{
	// nc, convect (nc*np), diff (nc*np), densat (nc*np), grav + pc; 2 phases  : n_ops = (1 + 2 + 2 + 2) * N_DIMS + 2 + 2
  
  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;
  
  // N_OPS = A * N_DIMS + B
  const int A = 7;
  const int B = 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e;

  // nc, grav + pc, diff, 3 phases  : n_ops = (1 + 3 + 1) * N_DIMS + 3 + 3

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX1 = 3; // used only for black-oil

  // N_OPS = A * N_DIMS + B
  const int A1 = 5;
  const int B1 = 6;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX1, A1, B1> e1;

  // nc, convect (nc*np), diff (nc*np), densat (nc*np), grav + pc; 3 phases  : n_ops = (1 + 3 + 3 + 3) * N_DIMS + 3 + 3
  
  // N_OPS = A * N_DIMS + B
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, 10, 6> e2;

 
  e.expose(m);
  e1.expose(m);
  e2.expose(m);
}

#endif //PYBIND11_ENABLED