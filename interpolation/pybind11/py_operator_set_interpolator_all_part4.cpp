#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals_interpolation.h"
#include <pybind11/stl.h>

#include "py_interpolator_exposer.hpp"

namespace py = pybind11;

// Part 4: N_DIMS = 7 through MAX_DIMS  (all N_OPS from 1 to MAX_DIMS)
// Automatically adapts when MAX_DIMS is increased or decreased.

template <uint8_t D>
void expose_dims_from(py::module& m)
{
	if constexpr (D <= MAX_DIMS)
	{
		recursive_exposer_nops<interpolator_exposer, py::module, D, MAX_DIMS>::expose(m);
		expose_dims_from<D + 1>(m);
	}
}

void pybind_operator_set_interpolator_all_part4(py::module& m)
{
	expose_dims_from<7>(m);
}

#endif //PYBIND11_ENABLED
