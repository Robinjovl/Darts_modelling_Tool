#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

void pybind_operator_set_interpolator_d7_d8(py::module& m)
{
	const int N_OPS_MAX = 10;
	recursive_exposer_nops<interpolator_exposer, py::module, 7, N_OPS_MAX>::expose(m);
	recursive_exposer_nops<interpolator_exposer, py::module, 8, N_OPS_MAX>::expose(m);
}

#endif //PYBIND11_ENABLED
