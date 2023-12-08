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
		.def_readwrite("values", &Matrix::values)
		.def(py::pickle(
		  [](const Matrix& p) { // __getstate__
			const size_t size = p.M * p.N;
			py::tuple t(size + 2);
			for (int i = 0; i < size; i++)
			  t[i] = p.values[i];

			t[size] = p.M;
			t[size + 1] = p.N;

			return t;
		  },
		  [](py::tuple t) { // __setstate__
			index_t M = t[t.size() - 2].cast<index_t>();
			index_t N = t[t.size() - 1].cast<index_t>();

			Matrix p(M, N);

			for (int i = 0; i < t.size() - 2; i++)
			  p.values[i] = t[i].cast<value_t>();

			return p;
		  }));
	py::bind_vector<std::vector<Matrix>>(m, "vector_matrix")
	  .def(py::pickle(
		[](const std::vector<Matrix>& p) { // __getstate__
		  py::tuple t(p.size());
		  for (int i = 0; i < p.size(); i++)
			t[i] = p[i];

		  return t;
		},
		[](py::tuple t) { // __setstate__
		  std::vector<Matrix> p(t.size());

		  for (int i = 0; i < p.size(); i++)
			p[i] = t[i].cast<Matrix>();

		  return p;
		}));

	py::class_<Matrix33, Matrix>(m, "matrix33") \
	  .def(py::init<>())
	  .def(py::init<value_t>())
	  .def(py::init<value_t, value_t, value_t>())
	  .def(py::init<std::valarray<value_t> &>())
	  .def_readwrite("values", &Matrix33::values)
	  .def(py::pickle(
		[](const Matrix33& p) { // __getstate__
		  py::tuple t(p.values.size());
		  for (int i = 0; i < p.values.size(); i++)
			t[i] = p.values[i];

		  return t;
		},
		[](py::tuple t) { // __setstate__
		  Matrix33 p;

		  for (int i = 0; i < t.size(); i++)
			p.values[i] = t[i].cast<value_t>();

		  return p;
		}));
	py::bind_vector<std::vector<Matrix33>>(m, "vector_matrix33")
	  .def(py::pickle(
		[](const std::vector<Matrix33>& p) { // __getstate__
		  py::tuple t(p.size());
		  for (int i = 0; i < p.size(); i++)
			t[i] = p[i];

		  return t;
		},
		[](py::tuple t) { // __setstate__
		  std::vector<Matrix33> p(t.size());

		  for (int i = 0; i < p.size(); i++)
			p[i] = t[i].cast<Matrix33>();

		  return p;
		}));

	py::class_<Discretizer>(m, "Discretizer", py::module_local())
		.def(py::init<>())

		// expose params
		.def_readwrite("perms", &Discretizer::perms)
		.def_readwrite("heat_conductions", &Discretizer::heat_conductions)
		.def_readwrite("permx", &Discretizer::permx)
		.def_readwrite("permy", &Discretizer::permy)
		.def_readwrite("permz", &Discretizer::permz)
		.def_readwrite("poro", &Discretizer::poro)
		.def_readwrite("cell_m", &Discretizer::cell_m)
		.def_readwrite("cell_p", &Discretizer::cell_p)
		.def_readwrite("flux_vals", &Discretizer::flux_vals)
		.def_readwrite("flux_vals_homo", &Discretizer::flux_vals_homo)
		.def_readwrite("flux_vals_thermal", &Discretizer::flux_vals_thermal)
		.def_readwrite("flux_rhs", &Discretizer::flux_rhs)
		.def_readwrite("fluxes_matrix", &Discretizer::fluxes_matrix)
		.def_readwrite("flux_offset", &Discretizer::flux_offset)
		.def_readwrite("flux_stencil", &Discretizer::flux_stencil)
		.def_readwrite("grav_vec", &Discretizer::grav_vec)
		.def_readwrite("p_grads", &Discretizer::p_grads)
		.def_readwrite("t_grads", &Discretizer::t_grads)

		.def("init", &Discretizer::init)
		.def("write_tran_cube", &Discretizer::write_tran_cube)
		.def("write_tran_list", &Discretizer::write_tran_list)
		.def("set_mesh", &Discretizer::set_mesh)
		.def("calc_tpfa_transmissibilities", &Discretizer::calc_tpfa_transmissibilities)
		.def("reconstruct_pressure_gradients_per_cell", &Discretizer::reconstruct_pressure_gradients_per_cell)
		.def("reconstruct_pressure_temperature_gradients_per_cell", &Discretizer::reconstruct_pressure_temperature_gradients_per_cell)
		.def("calc_mpfa_transmissibilities", &Discretizer::calc_mpfa_transmissibilities)
		.def("set_permeability", &Discretizer::set_permeability)
		.def("set_porosity", &Discretizer::set_porosity)
		.def("get_one_way_tpfa_transmissibilities", &Discretizer::get_one_way_tpfa_transmissibilities)
		.def("get_fault_xyz", &Discretizer::get_fault_xyz)
		;

	py::class_<BoundaryCondition>(m, "BoundaryCondition", py::module_local())
		.def(py::init<>())
		.def_readwrite("a", &BoundaryCondition::a)
		.def_readwrite("b", &BoundaryCondition::b)
	  ;
}