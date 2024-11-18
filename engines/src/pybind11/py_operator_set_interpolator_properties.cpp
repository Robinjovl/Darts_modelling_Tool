#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_properties(py::module &m)
{

	// N_DIMS = 1, 2, ..., N_DIMS_MAX
	const int N_DIMS_MAX = MAX_NC;

	// thermal problem: N_OPS = A * N_DIMS + B
	const int A = 1;
	const int B = 7;

  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A, B> e1;
 
  e1.expose(m);
}

#endif //PYBIND11_ENABLED