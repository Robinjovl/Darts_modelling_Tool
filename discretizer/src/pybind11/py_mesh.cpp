#include "py_mesh.h"

namespace py = pybind11;

void pybind_mesh(py::module &m)
{
	using mesh::Mesh;
	using linalg::index_t;

	py::class_<Mesh>(m, "Mesh", py::module_local())
		.def(py::init<>())

		.def_readwrite("poro", &Mesh::poro)
		.def_readwrite("nodes", &Mesh::nodes)
		.def_readwrite("num_of_nodes", &Mesh::num_of_nodes)
		.def_readwrite("num_of_elements", &Mesh::num_of_elements)
		.def_readwrite("n_cells", &Mesh::n_cells)
		.def_readwrite("nx", &Mesh::nx)
		.def_readwrite("ny", &Mesh::ny)
		.def_readwrite("nz", &Mesh::nz)
		.def_readwrite("local_to_global", &Mesh::local_to_global)
		.def_readwrite("global_to_local", &Mesh::global_to_local)
		.def_readwrite("num_of_nodes", &Mesh::num_of_nodes)
		.def_readwrite("num_of_elements", &Mesh::num_of_elements)
		.def_readwrite("elems", &Mesh::elems)
		.def_readwrite("volumes", &Mesh::volumes)
		.def_readwrite("depths", &Mesh::depths)
		.def_readwrite("centroids", &Mesh::centroids)
		.def_readwrite("actnum", &Mesh::actnum)
		.def_readwrite("zcorn", &Mesh::zcorn)
		.def_readwrite("coord", &Mesh::coord)
		.def_readwrite("elem_nodes", &Mesh::elem_nodes)
		.def_readwrite("elem_nodes_sorted", &Mesh::elem_nodes_sorted)
		.def_readwrite("region_ranges", &Mesh::region_ranges)
		.def_readwrite("region_elems_num", &Mesh::region_elems_num)
		.def_readwrite("elem_type_map", &Mesh::elem_type_map)
		.def_readwrite("init_apertures", &Mesh::init_apertures)
		.def_readwrite("tags", &Mesh::element_tags)
		.def_readwrite("conns", &Mesh::conns)
		.def_readwrite("conn_type_map", &Mesh::conn_type_map)

		.def_readwrite("adj_matrix", &Mesh::adj_matrix)
		.def_readwrite("adj_matrix_cols", &Mesh::adj_matrix_cols)
		.def_readwrite("adj_matrix_offset", &Mesh::adj_matrix_offset)
		.def("construct_local_global", &Mesh::construct_local_global)
		.def("generate_adjacency_matrix", &Mesh::generate_adjacency_matrix)
		.def("gmsh_mesh_processing", &Mesh::gmsh_mesh_processing)
		.def("get_global_index", &Mesh::get_global_index)
		.def("get_ijk", &Mesh::get_ijk)
		.def("get_ijk_as_str", &Mesh::get_ijk_as_str)
		.def("calc_cell_sizes", &Mesh::calc_cell_sizes)
		.def("write_int_array_to_file", &Mesh::write_array_to_file<index_t>)
		.def("write_float_array_to_file", &Mesh::write_array_to_file<value_t>)
		.def("calc_cell_nodes", &Mesh::calc_cell_nodes)
		.def("get_prisms", &Mesh::get_prisms)
		.def("get_centers", &Mesh::get_centers)
		.def("get_nodes_array", &Mesh::get_nodes_array)
		.def("cpg_elems_nodes", &Mesh::cpg_elems_nodes)
		.def("cpg_cell_props", &Mesh::cpg_cell_props)
		.def("cpg_connections", &Mesh::cpg_connections)
		;
}