//*************************************************************************
//    Copyright (c) 2022
//            Ilshat Saifullin         I.S.Saifullin@tudelft.nl
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

#include "py_elem.h"

namespace py = pybind11;
using mesh::Elem;

void pybind_elem(py::module &m)
{
	py::class_<Elem>(m, "Elem", "Mesh element")
		.def(py::init<>())

		// expose params
		.def_readwrite("loc", &mesh::Elem::loc)
		.def_readwrite("type", &mesh::Elem::type)
		.def_readwrite("n_pts", &mesh::Elem::n_pts)
		.def_readwrite("pts_offset", &mesh::Elem::pts_offset)
		.def_readwrite("elem_id", &mesh::Elem::elem_id)

		// expose functions
		.def("calculate_centroid", &Elem::calculate_centroid)
		.def("calculate_volume_and_centroid", &Elem::calculate_volume_and_centroid)
		;

	// expose enums

	py::enum_<mesh::ElemType>(m, "elem_type")		\
		.value("LINE",		mesh::ElemType::LINE)		\
		.value("TRI",			mesh::ElemType::TRI)		\
		.value("QUAD",		mesh::ElemType::QUAD)		\
		.value("TETRA",		mesh::ElemType::TETRA)	\
		.value("HEX",			mesh::ElemType::HEX)		\
		.value("PRISM",		mesh::ElemType::PRISM)	\
		.value("PYRAMID", mesh::ElemType::PYRAMID) 
		.export_values();
	py::bind_vector<std::vector<mesh::ElemType>>(m, "elem_type_vector");

	py::enum_<mesh::ElemLoc>(m, "elem_loc")  \
		.value("FRACTURE_BOUNDARY", mesh::ElemLoc::FRACTURE_BOUNDARY)		 \
		.value("BOUNDARY", mesh::ElemLoc::BOUNDARY)		 \
		.value("FRACTURE", mesh::ElemLoc::FRACTURE)		 \
		.value("MATRIX", mesh::ElemLoc::MATRIX)	 \
		.value("WELL", mesh::ElemLoc::WELL)
		.export_values();
	py::bind_vector<std::vector<mesh::ElemLoc>>(m, "elem_loc_vector");

	py::enum_<mesh::ConnType>(m, "conn_type")  \
		.value("MAT_MAT", mesh::ConnType::MAT_MAT)		 \
		.value("MAT_BOUND", mesh::ConnType::MAT_BOUND)		 \
		.value("MAT_FRAC", mesh::ConnType::MAT_FRAC)		 \
		.value("FRAC_MAT", mesh::ConnType::FRAC_MAT)	 \
		.value("FRAC_FRAC", mesh::ConnType::FRAC_FRAC) \
		.value("FRAC_BOUND", mesh::ConnType::FRAC_BOUND)
		.export_values();
	py::bind_vector<std::vector<mesh::ConnType>>(m, "conn_type_vector");


	py::class_<mesh::Connection>(m, "Connection", "Connection between higher-dimensional elements")
		.def(py::init<>())

		// expose params
		.def_readwrite("type", &mesh::Connection::type)
		.def_readwrite("n_pts", &mesh::Connection::n_pts)
		.def_readwrite("conn_id", &mesh::Connection::conn_id)
		.def_readwrite("elem_id1", &mesh::Connection::elem_id1)
		.def_readwrite("elem_id2", &mesh::Connection::elem_id2)
		.def_readwrite("pts_offset", &mesh::Connection::pts_offset)
		.def_readwrite("n", &mesh::Connection::n)
		.def_readwrite("c", &mesh::Connection::c)
		.def_readwrite("area", &mesh::Connection::area);
	py::bind_vector<std::vector<mesh::Connection>>(m, "conn_vector");
  
}
