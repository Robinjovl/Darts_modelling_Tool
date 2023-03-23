//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as 
//    published by the Free Software Foundation, either version 3 of the 
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public 
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************


#ifdef PYBIND11_ENABLED
#include <pybind11.h>
#include "py_globals.h"
#include <stl_bind.h>
#include <stl.h>

namespace py = pybind11;
#include "mech/pm_discretizer.hpp"
using namespace pm;

//PYBIND11_MAKE_OPAQUE(std::vector<value_t>);
PYBIND11_MAKE_OPAQUE(std::vector<Matrix>);
PYBIND11_MAKE_OPAQUE(std::vector<Face>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<Face>>);
PYBIND11_MAKE_OPAQUE(std::vector<Matrix33>);
PYBIND11_MAKE_OPAQUE(std::vector<Stiffness>);

void pybind_pm_discretizer (py::module &m)
{

	py::class_<Matrix>(m, "matrix") \
		.def(py::init<>())
		.def(py::init<std::valarray<value_t> &, index_t, index_t>())
		.def_readwrite("values", &Matrix::values);
	py::bind_vector<std::vector<Matrix>>(m, "vector_matrix");

	py::class_<Matrix33, Matrix>(m, "matrix33") \
		.def(py::init<>())
		.def(py::init<value_t>())
		.def(py::init<value_t,value_t,value_t>())
		.def(py::init<std::valarray<value_t> &>())
		.def_readwrite("values", &Matrix33::values);
	py::bind_vector<std::vector<Matrix33>>(m, "vector_matrix33");

	py::class_<Face>(m, "Face") \
		.def(py::init<>())
		.def(py::init<index_t, index_t, index_t, index_t, index_t, value_t, std::valarray<value_t> &, std::valarray<value_t> &>())
		.def(py::init<index_t, index_t, index_t, index_t, index_t, value_t, std::valarray<value_t> &, std::valarray<value_t> &, uint8_t>())
		.def(py::init<index_t, index_t, index_t, index_t, index_t, value_t, std::valarray<value_t> &, std::valarray<value_t> &, std::vector<index_t> &>())
		.def(py::init<index_t, index_t, index_t, index_t, index_t, value_t, std::valarray<value_t> &, std::valarray<value_t> &, std::vector<index_t> &, uint8_t>())
		.def_readwrite("type", &Face::type)
		.def_readwrite("cell_id1", &Face::cell_id1)
		.def_readwrite("cell_id2", &Face::cell_id2)
		.def_readwrite("face_id1", &Face::face_id1)
		.def_readwrite("face_id2", &Face::face_id2)
		.def_readwrite("area", &Face::area)
		.def_readwrite("n", &Face::n)
		.def_readwrite("c", &Face::c)
		.def_readwrite("pts", &Face::pts)
		.def_readwrite("is_impermeable", &Face::is_impermeable);
	py::bind_vector<std::vector<Face>>(m, "face_vector");
	py::bind_vector<std::vector<std::vector<Face>>>(m, "vector_face_vector");

	py::class_<Stiffness, Matrix>(m, "Stiffness") \
		.def(py::init<>())
		.def(py::init<value_t, value_t>())
		.def(py::init<std::valarray<value_t> &>())
		.def_readwrite("values", &Stiffness::values);
	py::bind_vector<std::vector<Stiffness>>(m, "stf_vector");

	enum Scheme { DEFAULT, APPLY_EIGEN_SPLITTING, APPLY_EIGEN_SPLITTING_NEW, AVERAGE };
	py::enum_<pm::Scheme>(m, "scheme_type")											\
		.value("default", pm::Scheme::DEFAULT)										\
		.value("apply_eigen_splitting", pm::Scheme::APPLY_EIGEN_SPLITTING)			\
		.value("apply_eigen_splitting_new", pm::Scheme::APPLY_EIGEN_SPLITTING_NEW)	\
		.value("average", pm::Scheme::AVERAGE)
		.export_values();

	py::class_<pm_discretizer>(m, "pm_discretizer", "Multipoint discretizer for poromechanics") \
		.def(py::init<>())
		.def("init", (void (pm_discretizer::*)(const index_t, const index_t, std::vector<index_t>&)) &pm_discretizer::init)
		.def("reconstruct_gradients_per_cell", &pm_discretizer::reconstruct_gradients_per_cell)
		.def("reconstruct_gradients_per_node", &pm_discretizer::reconstruct_gradients_per_node)
		.def("reconstruct_gradients_thermal_per_cell", &pm_discretizer::reconstruct_gradients_thermal_per_cell)
		//.def("calc_all_fluxes", &pm_discretizer::calc_all_fluxes)
		.def("calc_all_fluxes_once", &pm_discretizer::calc_all_fluxes_once)
		.def("get_gradient", &pm_discretizer::get_gradient)
		.def("get_thermal_gradient", &pm_discretizer::get_thermal_gradient)
		.def_readwrite("faces", &pm_discretizer::faces)
		.def_readwrite("ref_contact_ids", &pm_discretizer::ref_contact_ids)
		.def_readwrite("perms", &pm_discretizer::perms)
		.def_readwrite("diffs", &pm_discretizer::diffs)
		.def_readwrite("biots", &pm_discretizer::biots)
		.def_readwrite("th_expns", &pm_discretizer::th_expns)
		.def_readwrite("th_conds", &pm_discretizer::th_conds)
		.def_readwrite("stfs", &pm_discretizer::stfs)
		.def_readwrite("cell_centers", &pm_discretizer::cell_centers)
		.def_readwrite("frac_apers", &pm_discretizer::frac_apers)
		.def_readwrite("bc", &pm_discretizer::bc)
		.def_readwrite("bc_prev", &pm_discretizer::bc_prev)
		.def_readwrite("x_prev", &pm_discretizer::x_prev)
		.def_readwrite("cell_m", &pm_discretizer::cell_m)
		.def_readwrite("cell_p", &pm_discretizer::cell_p)
		.def_readwrite("stencil", &pm_discretizer::stencil)
		.def_readwrite("offset", &pm_discretizer::offset)
		.def_readwrite("tran", &pm_discretizer::tran)
		.def_readwrite("rhs", &pm_discretizer::rhs)
		.def_readwrite("tran_biot", &pm_discretizer::tran_biot)
		.def_readwrite("tran_th_expn", &pm_discretizer::tran_th_expn)
		.def_readwrite("tran_th_cond", &pm_discretizer::tran_th_cond)
		.def_readwrite("rhs_biot", &pm_discretizer::rhs_biot)
		.def_readwrite("tran_face_unknown", &pm_discretizer::tran_face_unknown)
		.def_readwrite("rhs_face_unknown", &pm_discretizer::rhs_face_unknown)
		.def_readwrite("visc", &pm_discretizer::visc)
		.def_readwrite("grav", &pm_discretizer::grav_vec)
		.def_readwrite("scheme", &pm_discretizer::scheme)
		.def_readwrite("assemble_heat_conduction", &pm_discretizer::ASSEMBLE_HEAT_CONDUCTION)
		.def_readwrite("neumann_boundaries_grad_reconstruction", &pm_discretizer::NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION)
		.def_readwrite("min_alpha_stabilization", &pm_discretizer::min_alpha_stabilization)
		.def_readwrite("max_alpha_in_domain", &pm_discretizer::max_alpha_in_domain)
		.def_readwrite("dt_max_alpha_in_domain", &pm_discretizer::dt_max_alpha_in_domain)
		.def_readwrite("cells_to_node", &pm_discretizer::cells_to_node)
		.def_readwrite("nodes_to_face", &pm_discretizer::nodes_to_face)
		.def_readwrite("cells_to_node_offset", &pm_discretizer::cells_to_node_offset)
		.def_readwrite("nodes_to_face_offset", &pm_discretizer::nodes_to_face_offset)
		.def_readwrite("nodes_to_face_cell_offset", &pm_discretizer::nodes_to_face_cell_offset);
}

#endif //PYBIND11_ENABLED