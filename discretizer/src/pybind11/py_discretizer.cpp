#include "py_discretizer.h"

namespace py = pybind11;
using dis::Discretizer;
using dis::BoundaryCondition;
using dis::Matrix33; 
using dis::Matrix;

PYBIND11_MAKE_OPAQUE(std::vector<Matrix>);
PYBIND11_MAKE_OPAQUE(std::vector<Matrix33>);

void pybind_discretizer(py::module &m)
{
	py::class_<Matrix>(m, "matrix", py::module_local()) \
		.def(py::init<>())
		.def(py::init<std::valarray<value_t> &, index_t, index_t>())
		.def_readwrite("values", &Matrix::values);
	py::bind_vector<std::vector<Matrix>>(m, "vector_matrix", py::module_local());

	py::class_<Matrix33, Matrix>(m, "matrix33", py::module_local()) \
		.def(py::init<>())
		.def(py::init<value_t>())
		.def(py::init<value_t, value_t, value_t>())
		.def(py::init<std::valarray<value_t> &>())
		.def_readwrite("values", &Matrix33::values);
	py::bind_vector<std::vector<Matrix33>>(m, "vector_matrix33", py::module_local());

	py::class_<Discretizer>(m, "Discretizer", py::module_local())
		.def(py::init<>())

		// expose params
		.def_readwrite("perms", &Discretizer::perms)
		.def_readwrite("permx", &Discretizer::permx)
		.def_readwrite("permy", &Discretizer::permy)
		.def_readwrite("permz", &Discretizer::permz)
		.def_readwrite("poro", &Discretizer::poro)
		.def_readwrite("cell_m", &Discretizer::cell_m)
		.def_readwrite("cell_p", &Discretizer::cell_p)
		.def_readwrite("grad_vals", &Discretizer::grad_vals)
		.def_readwrite("grad_offsets", &Discretizer::grad_offset)
		.def_readwrite("grad_stencil", &Discretizer::grad_stencil)
		.def_readwrite("flux_vals", &Discretizer::flux_vals)
		.def_readwrite("flux_rhs", &Discretizer::flux_rhs)
		.def_readwrite("fluxes_matrix", &Discretizer::fluxes_matrix)
		.def_readwrite("flux_offset", &Discretizer::flux_offset)
		.def_readwrite("flux_stencil", &Discretizer::flux_stencil)
		.def_readwrite("grav_vec", &Discretizer::grav_vec)

		.def("init", &Discretizer::init)
		.def("write_tran_cube", &Discretizer::write_tran_cube)
		.def("write_tran_list", &Discretizer::write_tran_list)
		.def("set_mesh", &Discretizer::set_mesh)
		.def("calc_tpfa_transmissibilities", &Discretizer::calc_tpfa_transmissibilities)
		.def("reconstruct_pressure_gradients_per_face", &Discretizer::reconstruct_pressure_gradients_per_face)
		.def("reconstruct_pressure_gradients_per_cell", &Discretizer::reconstruct_pressure_gradients_per_cell)
		.def("calc_mpfa_transmissibilities", &Discretizer::calc_mpfa_transmissibilities)
		.def("mergeMatrices", &Discretizer::mergeMatrices)
		.def("set_permeability", &Discretizer::set_permeability)
		.def("set_porosity", &Discretizer::set_porosity)
		.def("nbContributors", &Discretizer::nbContributors)
		.def("get_one_way_tpfa_transmissibilities", &Discretizer::get_one_way_tpfa_transmissibilities)
		.def("get_fault_xyz", &Discretizer::get_fault_xyz)
		;

	py::class_<BoundaryCondition>(m, "BoundaryCondition", py::module_local())
		.def(py::init<>())
		.def_readwrite("a", &BoundaryCondition::a)
		.def_readwrite("b", &BoundaryCondition::b)
		.def_readwrite("r", &BoundaryCondition::r)
	  ;
}