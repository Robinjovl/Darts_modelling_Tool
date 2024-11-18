#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_pz_kin_dif_np(py::module &m)
{
  // Kinetics + Diffusion = (3 + np) (acc, flux, kin rate, and diff per comp per phase) * N_DIMS + np (diff upstream sat_dens) + 1 (porosity)
  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;

  // N_OPS = A * N_DIMS + B (for np = 1)
  const int A1 = 4;
  const int B1 = 2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;
  e1.expose(m);

#if N_DIMS_MAX > 1
  // N_OPS = A * N_DIMS + B (for np = 2)
  const int A2 = 5;
  const int B2 = 3;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX - 1, A2, B2> e2;
  e2.expose(m);

#if N_DIMS_MAX > 2
  // N_OPS = A * N_DIMS + B (for np = 3)
  const int A3 = 6;
  const int B3 = 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX - 2, A3, B3> e3;
  e3.expose(m);

#endif
#endif
}

#endif //PYBIND11_ENABLED