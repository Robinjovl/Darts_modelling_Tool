#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_pzt_kin(py::module &m)
{
  // kinetics: 3 * NC (acc, flux, and kin) + 1 (porosity) + 5 (thermal) + 1 (thermal_reaction):
  // kinetics: 3 * (N_DIMS-1) + 1 (porosity) + 5 (thermal) + 1 (thermal_reaction)

  // N_DIMS = 1, 2, ..., N_DIMS_MAX
  const int N_DIMS_MAX = MAX_NC;
  
  // N_OPS = A * N_DIMS + B
  const int A = 3;
  const int B = 4;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e;
 
  e.expose(m);
}

#endif //PYBIND11_ENABLED