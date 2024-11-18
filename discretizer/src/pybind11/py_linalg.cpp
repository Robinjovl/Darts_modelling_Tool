#include "py_linalg.h"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include <valarray>
#include <array>

namespace py = pybind11;
using linalg::value_t;
using linalg::index_t;
using linalg::Vector3;
using linalg::ND;

PYBIND11_MAKE_OPAQUE(std::valarray<Vector3>);

void pybind_linalg(py::module &m)
{
	py::class_<Vector3>(m, "Vector3", py::module_local())
		.def(py::init<>())
		.def(py::init<value_t, value_t, value_t>())
		.def_readwrite("values", &Vector3::values);
	py::bind_vector<std::vector<Vector3>>(m, "vector_vector3", py::module_local());
}
