#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_super(py::module &m)
{
	// N_OPT = N_DIMS * (2 * NP + 2) + 3 * NP + 3

	// N_DIMS = 1, 2, ..., N_DIMS_MAX
	const int N_DIMS_MAX = MAX_NC;

	// single phase
	// NP = 1: A =  4, B =  6
	const int A1 = 4;
	const int B1 = 6;

	// two phase
	// NP = 2: A =  6, B =  9
	const int A2 = 6;
	const int B2 = 9;

	// three phase
	// NP = 3: A =  8, B = 12
	const int A3 = 8;
	const int B3 = 12;

	// four phase
	// NP = 4: A = 10, B = 15
	const int A4 = 10;
	const int B4 = 15;

  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A1, B1> e1;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A2, B2> e2;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A3, B3> e3;
  recursive_exposer_ndims_nops<interpolator_exposer, py::module, N_DIMS_MAX, A4, B4> e4;

  e1.expose(m);
  e2.expose(m);
  e3.expose(m);
  e4.expose(m);
}

#endif //PYBIND11_ENABLED