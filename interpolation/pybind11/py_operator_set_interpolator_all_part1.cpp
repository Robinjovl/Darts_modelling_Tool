#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 1: N_DIMS = 1, 2  (all N_OPS from 1 to MAX_DIMS)

void pybind_operator_set_interpolator_all_part1(py::module& m)
{
	if constexpr (MAX_DIMS >= 1)
		recursive_exposer_nops<interpolator_exposer, py::module, 1, MAX_DIMS>::expose(m);
	if constexpr (MAX_DIMS >= 2)
		recursive_exposer_nops<interpolator_exposer, py::module, 2, MAX_DIMS>::expose(m);
}

#endif //PYBIND11_ENABLED
