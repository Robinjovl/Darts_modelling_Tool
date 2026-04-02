#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Split across multiple TUs to reduce peak compiler memory usage
void pybind_operator_set_interpolator_d1_d2(py::module&);
void pybind_operator_set_interpolator_d3_d4(py::module&);
void pybind_operator_set_interpolator_d5_d6(py::module&);
void pybind_operator_set_interpolator_d7_d8(py::module&);
void pybind_operator_set_interpolator_d9_d10(py::module&);

void pybind_operator_set_interpolator_all(py::module& m)
{
	pybind_operator_set_interpolator_d1_d2(m);
	pybind_operator_set_interpolator_d3_d4(m);
	pybind_operator_set_interpolator_d5_d6(m);
	pybind_operator_set_interpolator_d7_d8(m);
	pybind_operator_set_interpolator_d9_d10(m);
}

#endif //PYBIND11_ENABLED
