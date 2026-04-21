#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 2: N_DIMS = 3, 4  (all N_OPS from 1 to MAX_DIMS)

void pybind_operator_set_interpolator_all_part2(py::module& m)
{
	if constexpr (MAX_DIMS >= 3)
		recursive_exposer_nops<interpolator_exposer, py::module, 3, MAX_DIMS>::expose(m);
	if constexpr (MAX_DIMS >= 4)
		recursive_exposer_nops<interpolator_exposer, py::module, 4, MAX_DIMS>::expose(m);
}

#endif //PYBIND11_ENABLED
