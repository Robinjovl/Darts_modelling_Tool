#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

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
	  .def("init", (int (conn_mesh::*)(std::vector<index_t> &, std::vector<index_t> &,
		  std::vector<value_t> &, std::vector<value_t> &)) &conn_mesh::init,
		  "Initialize by connection list defined by block_m, block_p, tran and tranD arrays ",
		  py::arg("block_m"), py::arg("block_p"), py::arg("tran"), py::arg("tranD") = std::vector<value_t>(0))
	  //.def("init_mpfa", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		//  std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t)) & conn_mesh::init_mpfa)
	  .def("init_mpfa", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t, index_t)) & conn_mesh::init_mpfa)
	  //.def("init_mpfa", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		//  std::vector<index_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_mpfa)
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
	  .def("init_pm_mech_discretizer", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		  std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, 
		  std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, index_t, index_t, index_t)) & conn_mesh::init_pm_mech_discretizer)
	  .def("init_pme_mech_discretizer", (int (conn_mesh::*)(std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&, std::vector<index_t>&,
		std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&,
		std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, std::vector<value_t>&, 
		index_t, index_t, index_t)) & conn_mesh::init_pme_mech_discretizer)
	  .def("add_conn", &conn_mesh::add_conn)
	  //.def("add_conn_mpfa", &conn_mesh::add_conn_mpfa)
	  .def("reverse_and_sort", &conn_mesh::reverse_and_sort)
	  .def("reverse_and_sort_dvel", &conn_mesh::reverse_and_sort_dvel)
	  .def("reverse_and_sort_mpfa", &conn_mesh::reverse_and_sort_mpfa)
	  .def("reverse_and_sort_mpsa", &conn_mesh::reverse_and_sort_mpsa)
	  .def("reverse_and_sort_pm", &conn_mesh::reverse_and_sort_pm)
	  .def("reverse_and_sort_pm_mech_discretizer", &conn_mesh::reverse_and_sort_pm_mech_discretizer)
	  .def("reverse_and_sort_pme_mech_discretizer", &conn_mesh::reverse_and_sort_pme_mech_discretizer)
	  .def("add_wells", &conn_mesh::add_wells)
	  .def("add_wells_mpfa", &conn_mesh::add_wells_mpfa)
	  .def("init_grav_coef", &conn_mesh::init_grav_coef, "Initialize gravity coefficients for every connection", "grav_const"_a = 9.80665e-5)
	  .def("get_res_tran", &conn_mesh::get_res_tran, "Get reservoir transmissibilities", "tran"_a, "tranD"_a)
	  .def("set_res_tran", &conn_mesh::set_res_tran, "Set reservoir transmissibilities", "tran"_a, "tranD"_a)
	  .def("get_wells_tran", &conn_mesh::get_wells_tran, "Get well indexes", "tran"_a)
	  .def("set_wells_tran", &conn_mesh::set_wells_tran, "Set well indexes", "tran"_a)
	  .def("set_wells_tran", &conn_mesh::set_wells_tran, "Set well indexes", "tran"_a)
	  .def("connect_segments", &conn_mesh::connect_segments)
	  //properties
	  .def_readwrite("n_blocks", &conn_mesh::n_blocks)
	  .def_readwrite("n_res_blocks", &conn_mesh::n_res_blocks)
	  .def_readwrite("poro", &conn_mesh::poro)
	  .def_readwrite("volume", &conn_mesh::volume)
	  .def_readwrite("initial_state", &conn_mesh::initial_state)
	  .def_readwrite("ref_pressure", &conn_mesh::ref_pressure)
	  .def_readwrite("ref_temperature", &conn_mesh::ref_temperature)
	  .def_readwrite("ref_eps_vol", &conn_mesh::ref_eps_vol)
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
	  .def_readwrite("rock_compressibility", &conn_mesh::rock_compressibility)
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
	  .def_readwrite("unsorted_gravity_fluxes", &conn_mesh::one_way_gravity_flux)
	  .def_readwrite("hooke_tran", &conn_mesh::hooke_tran)
	  .def_readwrite("biot_tran", &conn_mesh::biot_tran)
	  .def_readwrite("darcy_tran", &conn_mesh::darcy_tran)
	  .def_readwrite("vol_strain_tran", &conn_mesh::vol_strain_tran)
	  .def_readwrite("fourier_tran", &conn_mesh::fourier_tran)
	  .def_readwrite("hooke_rhs", &conn_mesh::hooke_rhs)
	  .def_readwrite("biot_rhs", &conn_mesh::biot_rhs)
	  .def_readwrite("darcy_rhs", &conn_mesh::darcy_rhs)
	  .def_readwrite("vol_strain_rhs", &conn_mesh::vol_strain_rhs)
	  .def_readwrite("velocity_appr", &conn_mesh::velocity_appr)
	  .def_readwrite("velocity_offset", &conn_mesh::velocity_offset);

	py::bind_vector<std::vector<std::vector<index_t>>>(m, "vector_index_vector");
}

/*
  py::class_ <interp_multitable<int, double, 2, 4>>(m, "interp_multitable")
    .def(py::init<>())
    .def("init_op", &interp_multitable<int, double, 2, 4>::init_op);
}
*/

#endif //PYBIND11_ENABLED