#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Unified well ctrl operator layouts:
// N_OPS = 2 * N_RATE_CTRL_TYPES * NP + N_STATE_CTRL_OPS = 8 * NP + 2.

void pybind_operator_set_interpolator_well_ctrl(py::module &m)
{
  const int N_DIMS_MAX = MAX_DIMS;

  // single-phase well ctrl: N_OPS = 10
  const int A1 = 0;
  const int B1 = 10;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;

  // two-phase well ctrl: N_OPS = 18
  const int A2 = 0;
  const int B2 = 18;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A2, B2> e2;

  // three-phase well ctrl: N_OPS = 26
  const int A3 = 0;
  const int B3 = 26;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A3, B3> e3;

  // four-phase well ctrl: N_OPS = 34
  const int A4 = 0;
  const int B4 = 34;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A4, B4> e4;

  // five-phase well ctrl: N_OPS = 42
  const int A5 = 0;
  const int B5 = 42;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A5, B5> e5;

  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
  e4.expose(m);
  e5.expose(m);
}

#endif //PYBIND11_ENABLED
