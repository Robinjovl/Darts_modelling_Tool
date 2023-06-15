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

#include "globals.h"
#include "conn_mesh.h"

PYBIND11_MAKE_OPAQUE(std::vector<std::vector<index_t>>);

namespace py = pybind11;

void pybind_mesh_conn(py::module &m)
{
  using namespace pybind11::literals;
  
  py::class_<conn_mesh>(m, "conn_mesh", "Class for connection-based mesh and it`s properties")
	  .def(py::init<>())
	  //methods
	  .def("init", (int (conn_mesh::*)(std::string)) &conn_mesh::init, "Initialize from TPFACONNS (TPFACONNSN) keyword file")
	  .def("init", (int (conn_mesh::*)(std::vector<index_t> &, std::vector<index_t> &,
		  std::vector<value_t> &, std::vector<value_t> &)) &conn_mesh::init,
		  "Initialize by connection list defined by block_m, block_p, tran and tranD arrays ",
		  py::arg("block_m"), py::arg("block_p"), py::arg("tran"), py::arg("tranD") = std::vector<value_t>(0))
	  .def("init_mpfa", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t)) & conn_mesh::init_mpfa)
	  .def("init_mpfa_e", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_mpfa_e)
	  .def("init_mpfa", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_mpfa)
	  .def("init_mpsa", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<index_t>&, std::vector<value_t>&, uint8_t, index_t, index_t, index_t)) & conn_mesh::init_mpsa)
	  .def("init_mpsa", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, uint8_t, index_t, index_t, index_t)) & conn_mesh::init_mpsa)
	  .def("init_pm", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_pm)
	  .def("init_pm", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_pm)
	  .def("init_pm", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_pm)
	  .def("init_pme", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_pme)
	  .def("init_const_1d", &conn_mesh::init_const_1d)
	  .def("add_conn", &conn_mesh::add_conn)
	  //.def("add_conn_mpfa", &conn_mesh::add_conn_mpfa)
	  .def("reverse_and_sort", &conn_mesh::reverse_and_sort)
	  .def("reverse_and_sort_dvel", &conn_mesh::reverse_and_sort_dvel)
	  .def("reverse_and_sort_mpfa", &conn_mesh::reverse_and_sort_mpfa)
	  .def("reverse_and_sort_mpsa", &conn_mesh::reverse_and_sort_mpsa)
	  .def("reverse_and_sort_pm", &conn_mesh::reverse_and_sort_pm)
	  .def("reverse_and_sort_pme", &conn_mesh::reverse_and_sort_pme)
	  .def("init_poro", &conn_mesh::init_poro)
	  .def("add_wells", &conn_mesh::add_wells)
	  .def("add_wells_mpfa", &conn_mesh::add_wells_mpfa)
	  .def("init_grav_coef", &conn_mesh::init_grav_coef, "Initialize gravity coefficients for every connection", "grav_const"_a = 9.80665e-5)
	  .def("get_res_tran", &conn_mesh::get_res_tran, "Get reservoir transmissibilities", "tran"_a, "tranD"_a)
	  .def("set_res_tran", &conn_mesh::set_res_tran, "Set reservoir transmissibilities", "tran"_a, "tranD"_a)
	  .def("get_wells_tran", &conn_mesh::get_wells_tran, "Get well indexes", "tran"_a)
	  .def("set_wells_tran", &conn_mesh::set_wells_tran, "Set well indexes", "tran"_a)
	  .def("set_wells_tran", &conn_mesh::set_wells_tran, "Set well indexes", "tran"_a)
	  .def("save_poro", &conn_mesh::save_poro)
	  .def("save_volume", &conn_mesh::save_volume)
	  .def("save_pressure", &conn_mesh::save_pressure)
	  .def("save_zmf", &conn_mesh::save_zmf)
	  .def("save_temperature", &conn_mesh::save_temperature)
	  .def("save_enthalpy", &conn_mesh::save_enthalpy)
		.def("connect_segments", &conn_mesh::connect_segments)
	  //properties
	  .def_readwrite("n_blocks", &conn_mesh::n_blocks)
	  .def_readwrite("n_res_blocks", &conn_mesh::n_res_blocks)
	  .def_readwrite("poro", &conn_mesh::poro)
	  .def_readwrite("volume", &conn_mesh::volume)
	  .def_readwrite("pressure", &conn_mesh::pressure)
	  .def_readwrite("ref_pressure", &conn_mesh::ref_pressure)
	  .def_readwrite("ref_temperature", &conn_mesh::ref_temperature)
	  .def_readwrite("ref_eps_vol", &conn_mesh::ref_eps_vol)
	  .def_readwrite("composition", &conn_mesh::composition)
	  .def_readwrite("temperature", &conn_mesh::temperature)
	  .def_readwrite("enthalpy", &conn_mesh::enthalpy)
	  .def_readwrite("velocity", &conn_mesh::velocity)
	  .def_readwrite("op_num", &conn_mesh::op_num)
	  .def_readwrite("depth", &conn_mesh::depth)
	  .def_readwrite("heat_capacity", &conn_mesh::heat_capacity)
	  .def_readwrite("rock_cond", &conn_mesh::rock_cond)
	  .def_readwrite("kin_factor", &conn_mesh::kin_factor)
	  .def_readwrite("mob_multiplier", &conn_mesh::mob_multiplier)
	  .def_readwrite("block_m", &conn_mesh::block_m)
	  .def_readwrite("block_p", &conn_mesh::block_p)
	  .def_readwrite("tran", &conn_mesh::tran)
	  .def_readwrite("tran_ref", &conn_mesh::tran_ref)
	  .def_readwrite("tranD", &conn_mesh::tranD)
	  .def_readwrite("displacement", &conn_mesh::displacement)
	  .def_readwrite("bc", &conn_mesh::bc)
	  .def_readwrite("bc_prev", &conn_mesh::bc_n)
	  .def_readwrite("bc_ref", &conn_mesh::bc_ref)
	  .def_readwrite("pz_bounds", &conn_mesh::pz_bounds)
	  .def_readwrite("biot", &conn_mesh::biot)
	  .def_readwrite("kd", &conn_mesh::drained_compressibility)
	  .def_readwrite("f", &conn_mesh::f)
	  .def_readwrite("stencil", &conn_mesh::stencil)
	  .def_readwrite("offset", &conn_mesh::offset)
	  .def_readwrite("rhs", &conn_mesh::rhs)
	  .def_readwrite("rhs_ref", &conn_mesh::rhs_ref)
	  .def_readwrite("flux", &conn_mesh::flux)
	  .def_readwrite("rhs_biot", &conn_mesh::rhs_biot)
	  .def_readwrite("rhs_biot_n", &conn_mesh::rhs_biot_n)
	  .def_readwrite("rhs_biot_ref", &conn_mesh::rhs_biot_ref)
	  .def_readwrite("tran_biot", &conn_mesh::tran_biot)
	  .def_readwrite("tran_biot_n", &conn_mesh::tran_biot_n)
	  .def_readwrite("tran_biot_ref", &conn_mesh::tran_biot_ref)
	  .def_readwrite("th_poro", &conn_mesh::th_poro)
	  .def_readwrite("fault_conn_id", &conn_mesh::fault_conn_id)
	  .def_readwrite("sorted_conn_ids", &conn_mesh::sorted_conn_ids)
	  .def_readwrite("sorted_stencil_ids", &conn_mesh::sorted_stencil_ids)
	  .def_readwrite("fault_normals", &conn_mesh::fault_normals)
	  .def_readwrite("unsorted_gravity_fluxes", &conn_mesh::one_way_gravity_flux);

	py::bind_vector<std::vector<std::vector<index_t>>>(m, "vector_index_vector");
}

/*
  py::class_ <interp_multitable<int, double, 2, 4>>(m, "interp_multitable")
    .def(py::init<>())
    .def("init_op", &interp_multitable<int, double, 2, 4>::init_op);
}
*/

#endif //PYBIND11_ENABLED