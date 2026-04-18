#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 3: N_DIMS = 5, 6  (all N_OPS from 1 to MAX_DIMS)

void pybind_operator_set_interpolator_all_part3(py::module& m)
{
	if constexpr (MAX_DIMS >= 5)
		recursive_exposer_nops<interpolator_exposer, py::module, 5, MAX_DIMS>::expose(m);
	if constexpr (MAX_DIMS >= 6)
		recursive_exposer_nops<interpolator_exposer, py::module, 6, MAX_DIMS>::expose(m);
}

#endif //PYBIND11_ENABLED
