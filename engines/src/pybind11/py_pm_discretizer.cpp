#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

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
		.def(py::init<value_t,value_t,value_t>())
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
		.def_readwrite("is_impermeable", &Face::is_impermeable)
		.def(py::pickle(
			[](const Face& f) { // __getstate__
				py::tuple t(10);
				
				t[0] = f.type;
				t[1] = f.cell_id1;
				t[2] = f.cell_id2;
				t[3] = f.face_id1;
				t[4] = f.face_id2;
				t[5] = f.n;
				t[6] = f.c;
				t[7] = f.area;
				t[8] = f.pts;
				t[9] = f.is_impermeable;

				return t;
			},
			[](py::tuple t) { // __setstate__
				Face f;

				f.type = t[0].cast<index_t>();
				f.cell_id1 = t[1].cast<index_t>();
				f.cell_id2 = t[2].cast<index_t>();
				f.face_id1 = t[3].cast<index_t>();
				f.face_id2 = t[4].cast<index_t>();
				f.n = t[5].cast<Matrix>();
				f.c = t[6].cast<Matrix>();
				f.area = t[7].cast<value_t>();
				f.pts = t[8].cast<std::vector<index_t>>();
				f.is_impermeable = t[9].cast<uint8_t>();

				return f;
			}));
	py::bind_vector<std::vector<Face>>(m, "face_vector")
		.def(py::pickle(
			[](const std::vector<Face>& p) { // __getstate__
				py::tuple t(p.size());
				for (int i = 0; i < p.size(); i++)
					t[i] = p[i];

				return t;
			},
			[](py::tuple t) { // __setstate__
				std::vector<Face> p(t.size());

				for (int i = 0; i < p.size(); i++)
					p[i] = t[i].cast<Face>();

				return p;
			}));
	py::bind_vector<std::vector<std::vector<Face>>>(m, "vector_face_vector")
		.def(py::pickle(
			[](const std::vector<std::vector<Face>>& p) { // __getstate__
				py::tuple t(p.size());
				for (int i = 0; i < p.size(); i++)
					t[i] = p[i];

				return t;
			},
			[](py::tuple t) { // __setstate__
				std::vector<std::vector<Face>> p(t.size());

				for (int i = 0; i < p.size(); i++)
					p[i] = t[i].cast<std::vector<Face>>();

				return p;
			}));

	py::class_<Stiffness, Matrix>(m, "Stiffness") \
		.def(py::init<>())
		.def(py::init<value_t, value_t>())
		.def(py::init<std::valarray<value_t> &>())
		.def_readwrite("values", &Stiffness::values)
		.def(py::pickle(
			[](const Stiffness& p) { // __getstate__
				py::tuple t(p.values.size());
				for (int i = 0; i < p.values.size(); i++)
					t[i] = p.values[i];

				return t;
			},
			[](py::tuple t) { // __setstate__
				Stiffness p;

				for (int i = 0; i < t.size(); i++)
					p.values[i] = t[i].cast<value_t>();

				return p;
			}));
	py::bind_vector<std::vector<Stiffness>>(m, "stf_vector")
		.def(py::pickle(
			[](const std::vector<Stiffness>& p) { // __getstate__
				py::tuple t(p.size());
				for (int i = 0; i < p.size(); i++)
					t[i] = p[i];

				return t;
			},
			[](py::tuple t) { // __setstate__
				std::vector<Stiffness> p(t.size());

				for (int i = 0; i < p.size(); i++)
					p[i] = t[i].cast<Stiffness>();

				return p;
			}));

	enum Scheme { DEFAULT, APPLY_EIGEN_SPLITTING, APPLY_EIGEN_SPLITTING_NEW, AVERAGE };
	py::enum_<pm::Scheme>(m, "scheme_type")											\
		.value("default", pm::Scheme::DEFAULT)										\
		.value("apply_eigen_splitting", pm::Scheme::APPLY_EIGEN_SPLITTING)			\
		.value("apply_eigen_splitting_new", pm::Scheme::APPLY_EIGEN_SPLITTING_NEW)	\
		.value("average", pm::Scheme::AVERAGE)
		/*.def(py::pickle(
			[](const Scheme& p) { // __getstate__
				py::tuple t(1);
				t[0] = p;

				return t;
			},
			[](py::tuple t) { // __setstate__
				Scheme p(t[0].cast<Scheme>());
				return p;
			}))*/
		.export_values();

	py::class_<pm_discretizer::Gradients>(m, "Gradients", "Approximation for gradients") \
		.def_readwrite("stencil", &pm_discretizer::Gradients::stencil)
		.def_readwrite("tran", &pm_discretizer::Gradients::mat)
		.def_readwrite("rhs", &pm_discretizer::Gradients::rhs)
		.def(py::pickle(
			[](const pm_discretizer::Gradients& p) { // __getstate__
				py::tuple t(3);
				t[0] = p.stencil;
				t[1] = p.mat;
				t[2] = p.rhs;

				return t;
			},
			[](py::tuple t) { // __setstate__
				pm_discretizer::Gradients p;

				p.stencil = t[0].cast<std::vector<index_t>>();
				p.mat = t[1].cast<Matrix>();
				p.rhs = t[2].cast<Matrix>();
				
				return p;
			}));
	py::bind_vector<std::vector<pm_discretizer::Gradients>>(m, "grad_vector")
		.def(py::pickle(
			[](const std::vector<pm_discretizer::Gradients>& p) { // __getstate__
				py::tuple t(p.size());
				for (int i = 0; i < p.size(); i++)
					t[i] = p[i];

				return t;
			},
			[](py::tuple t) { // __setstate__
				std::vector<pm_discretizer::Gradients> p(t.size());

				for (int i = 0; i < p.size(); i++)
					p[i] = t[i].cast<pm_discretizer::Gradients>();

				return p;
			}));

	py::class_<pm_discretizer>(m, "pm_discretizer", "Multipoint discretizer for poromechanics") \
		.def(py::init<>())
		.def("init", (void (pm_discretizer::*)(const index_t, const index_t, std::vector<index_t>&)) &pm_discretizer::init)
		.def("reconstruct_gradients_per_cell", &pm_discretizer::reconstruct_gradients_per_cell)
		//.def("reconstruct_gradients_per_node", &pm_discretizer::reconstruct_gradients_per_node)
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
		.def_readwrite("nodes_to_face_cell_offset", &pm_discretizer::nodes_to_face_cell_offset)
		.def(py::pickle(
			[](const pm_discretizer& p) { // __getstate__
				py::tuple t(36);
				t[0] = p.n_matrix;
				t[1] = p.n_cells;
				t[2] = p.n_fracs;
				t[3] = p.n_faces;
				t[4] = p.nb_faces;
				t[5] = p.faces;
				t[6] = p.perms;
				t[7] = p.diffs;
				t[8] = p.biots;
				t[9] = p.th_expns;
				t[10] = p.stfs;
				t[11] = p.cell_centers;
				t[12] = p.u0;
				t[13] = p.bc;
				t[14] = p.bc_prev;
				t[15] = p.x_prev;
				t[16] = p.frac_apers;
				t[17] = p.visc;
				t[18] = p.grav;
				t[19] = p.density;
				t[20] = p.grav_vec;
				t[21] = p.cell_m;
				t[22] = p.cell_p;
				t[23] = p.stencil;
				t[24] = p.offset;
				t[25] = p.tran;
				t[26] = p.rhs;
				t[27] = p.tran_biot;
				t[28] = p.rhs_biot;
				t[29] = p.tran_th_expn;
				t[30] = p.tran_th_cond;
				t[31] = p.tran_face_unknown;
				t[32] = p.rhs_face_unknown;
				t[33] = p.grad;
				t[34] = p.grad_d;
				t[35] = p.grad_prev;

				return t;
			},
			[](py::tuple t) { // __setstate__
				pm_discretizer p;
				p.n_matrix = t[0].cast<int>();
				p.n_cells = t[1].cast<int>();
				p.n_fracs = t[2].cast<int>();
				p.n_faces = t[3].cast<int>();
				p.nb_faces = t[4].cast<int>();
				p.faces = t[5].cast<std::vector<std::vector<Face>>>();
				p.perms = t[6].cast<std::vector<Matrix33>>();
				p.diffs = t[7].cast<std::vector<Matrix33>>();
				p.biots = t[8].cast<std::vector<Matrix33>>();
				p.th_expns = t[9].cast<std::vector<Matrix>>();
				p.stfs = t[10].cast<std::vector<Stiffness>>();
				p.cell_centers = t[11].cast<std::vector<Matrix>>();
				p.u0 = t[12].cast<std::vector<Matrix>>();
				p.bc = t[13].cast<std::vector<Matrix>>();
				p.bc_prev = t[14].cast<std::vector<Matrix>>();
				p.x_prev = t[15].cast<std::vector<value_t>>();
				p.frac_apers = t[16].cast<std::vector<value_t>>();
				p.visc = t[17].cast<value_t>();
				p.grav = t[18].cast<value_t>();
				p.density = t[19].cast<value_t>();
				p.grav_vec = t[20].cast<Matrix>();
				p.cell_m = t[21].cast<std::vector<index_t>>();
				p.cell_p = t[22].cast<std::vector<index_t>>();
				p.stencil = t[23].cast<std::vector<index_t>>();
				p.offset = t[24].cast<std::vector<index_t>>();
				p.tran = t[25].cast<std::vector<value_t>>();
				p.rhs = t[26].cast<std::vector<value_t>>();
				p.tran_biot = t[27].cast<std::vector<value_t>>();
				p.rhs_biot = t[28].cast<std::vector<value_t>>();
				p.tran_th_expn = t[29].cast<std::vector<value_t>>();
				p.tran_th_cond = t[30].cast<std::vector<value_t>>();
				p.tran_face_unknown = t[31].cast<std::vector<value_t>>();
				p.rhs_face_unknown = t[32].cast<std::vector<value_t>>();
				p.grad = t[33].cast<std::vector<pm_discretizer::Gradients>>();
				p.grad_d = t[34].cast<std::vector<pm_discretizer::Gradients>>();
				p.grad_prev = t[35].cast<std::vector<pm_discretizer::Gradients>>();

				return p;
			}));
}

#endif //PYBIND11_ENABLED