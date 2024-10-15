#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_all(py::module& m)
{
	const int N_DIMS_MAX = 10;
	const int N_OPS_MAX = 10;
	recursive_exposer_ndims_nops2<interpolator_exposer, py::module, N_DIMS_MAX, N_OPS_MAX> e;
	e.expose(m);
}

#endif //PYBIND11_ENABLED