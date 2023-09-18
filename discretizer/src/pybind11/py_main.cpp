#include "py_global.h"
#include "discretizer_build_info.h"
#include "utils.h"

namespace py = pybind11;

void pybind_elem(py::module &m);
void pybind_mesh(py::module &m);
void pybind_discretizer(py::module &m);
void pybind_mech_discretizer(py::module &m);
void pybind_linalg(py::module &m);
void pybind_approximation(py::module& m);


void print_build_info()
{
	std::cout << "darts-discretizer built on " << DISCRETIZER_BUILD_DATE << " by " << DISCRETIZER_BUILD_MACHINE << " from " << DISCRETIZER_BUILD_GIT_HASH << std::endl;
}

PYBIND11_MODULE(discretizer, m)
{
	py::bind_vector<std::vector<index_t>>(m, "index_vector", py::module_local(), py::buffer_protocol());
	py::bind_vector<std::vector<value_t>>(m, "value_vector", py::module_local(), py::buffer_protocol())
		.def(py::pickle(
			[](const std::vector<value_t> &p) { // __getstate__
		py::tuple t(p.size());
		for (int i = 0; i < p.size(); i++)
			t[i] = p[i];

		return t;
	},
			[](py::tuple t) { // __setstate__
		std::vector<value_t> p(t.size());

		for (int i = 0; i < p.size(); i++)
			p[i] = t[i].cast<value_t>();

		return p;
	}));
	
	m.def("load_single_float_keyword", utils::load_single_keyword<value_t>);
	m.def("load_single_int_keyword", utils::load_single_keyword<index_t>);
	m.def("print_build_info", &print_build_info, "Print build information: date, user, machine, git hash");

	pybind_elem(m);
	pybind_linalg(m);
	pybind_mesh(m);
	pybind_discretizer(m);
	pybind_mech_discretizer(m);
	pybind_approximation(m);
}
