#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Split into 4 parts by N_DIMS range (2 dims × MAX_DIMS ops each)
// to enable parallel compilation and reduce per-TU memory usage.
void pybind_operator_set_interpolator_all_part1(py::module& m);  // N_DIMS 1-2
void pybind_operator_set_interpolator_all_part2(py::module& m);  // N_DIMS 3-4
void pybind_operator_set_interpolator_all_part3(py::module& m);  // N_DIMS 5-6
void pybind_operator_set_interpolator_all_part4(py::module& m);  // N_DIMS 7-8

void pybind_operator_set_interpolator_all(py::module& m)
{
	pybind_operator_set_interpolator_all_part1(m);
	pybind_operator_set_interpolator_all_part2(m);
	pybind_operator_set_interpolator_all_part3(m);
	pybind_operator_set_interpolator_all_part4(m);
}

#endif //PYBIND11_ENABLED
