#include "mech/pm_discretizer.hpp"
#include "mech/matrix.h"
#include <iostream>
#include <unordered_set>
#include <assert.h>
#include "contact.h"

#define _USE_MATH_DEFINES
#include <math.h>

using namespace pm;
//using namespace linalg;
using std::cout;
using std::endl;
using std::abs;
using std::vector;
using std::unordered_set;
using pm::Matrix;
using std::fill_n;

const pm::Matrix pm_discretizer::I3 = pm::Matrix({ 1,0,0, 0,1,0, 0,0,1 }, ND, ND);
const pm::Matrix pm_discretizer::I4 = pm::Matrix({ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }, ND + 1, ND + 1);
const value_t pm_discretizer::darcy_constant = 0.0085267146719160104986876640419948;
const value_t pm_discretizer::heat_cond_constant = 1.0;
const index_t pm_discretizer::BLOCK_SIZE = 4;

pm_discretizer::pm_discretizer() : W(9, 6)
{
	W(0, 0) = 1.0;
	W(1, 5) = 1.0;
	W(2, 4) = 1.0;
	W(3, 5) = 1.0;
	W(4, 1) = 1.0;
	W(5, 3) = 1.0;
	W(6, 4) = 1.0;
	W(7, 3) = 1.0;
	W(8, 2) = 1.0;
	visc = 1.0;
	grav = 0.01;
	density = 800.0;
	//grav_vec = density * grav * Matrix({ 0, 0, 1 }, 1, ND);
	grav_vec = density * grav * Matrix({ 0, 0, 0 }, 1, ND);
	scheme = DEFAULT;
	ASSEMBLE_HEAT_CONDUCTION = false;
	NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION = true;
	min_alpha_stabilization = 1.e-2;
}
pm_discretizer::~pm_discretizer() 
{
}
pm_discretizer::Gradients pm_discretizer::merge_stencils(const vector<index_t>& st1, const Matrix& m1, const vector<index_t>& st2, const Matrix& m2)
{
	assert(m1.M == m2.M);
	pre_merged_stencil = st1;
	MERGE_BLOCK_SIZE = m1.M / ND;
	auto& pre_grad = pre_merged_grad[MERGE_BLOCK_SIZE];
	std::fill_n(&pre_grad.values[0], pre_grad.values.size(), 0.0);
	pre_grad(0, { (size_t)pre_grad.M, (size_t)(MERGE_BLOCK_SIZE * st1.size()) }, { (size_t)pre_grad.N, 1 }) = 
		m1(0, { (size_t)m1.M, (size_t)(MERGE_BLOCK_SIZE * st1.size()) }, { (size_t)m1.N, 1 });

	for (counter = 0; counter < st2.size(); counter++)
	{
		res1 = findInVector(pre_merged_stencil, st2[counter]);
		if (res1.first) { id = res1.second; }
		else { id = pre_merged_stencil.size(); pre_merged_stencil.push_back(st2[counter]); }
		pre_grad(MERGE_BLOCK_SIZE * id, { (size_t)pre_grad.M, (size_t)MERGE_BLOCK_SIZE }, { (size_t)pre_grad.N, 1 }) += m2(MERGE_BLOCK_SIZE * counter, { (size_t)pre_grad.M, (size_t)MERGE_BLOCK_SIZE }, { (size_t)m2.N, 1 });
	}
	return { pre_merged_stencil,
			Matrix(pre_grad(0, { (size_t)pre_grad.M, (size_t)(MERGE_BLOCK_SIZE * pre_merged_stencil.size()) }, { (size_t)pre_grad.N, 1 }), pre_grad.M, MERGE_BLOCK_SIZE * (size_t)pre_merged_stencil.size()),
			Matrix(ND * MERGE_BLOCK_SIZE, 1) };
}
Approximation& pm_discretizer::merge_approximations(const Approximation& flux1, const Approximation& flux2, const index_t ws_id)
{
	auto& res = pre_merged_flux[ws_id];
	res.stencil = flux1.stencil;
	std::fill_n(&res.a.values[0], res.a.values.size(), 0.0);
	std::fill_n(&res.a_biot.values[0], res.a_biot.values.size(), 0.0);
	res.a(0, { BLOCK_SIZE, (size_t)(BLOCK_SIZE * flux1.stencil.size()) }, { (size_t)res.a.N, 1 }) =
		flux1.a(0, { BLOCK_SIZE, (size_t)(BLOCK_SIZE * flux1.stencil.size()) }, { (size_t)res.a.N, 1 });
	res.a_biot(0, { BLOCK_SIZE, (size_t)(BLOCK_SIZE * flux1.stencil.size()) }, { (size_t)res.a_biot.N, 1 }) =
		flux1.a_biot(0, { BLOCK_SIZE, (size_t)(BLOCK_SIZE * flux1.stencil.size()) }, { (size_t)res.a_biot.N, 1 });

	for (counter = 0; counter < flux2.stencil.size(); counter++)
	{
		res1 = findInVector(res.stencil, flux2.stencil[counter]);
		if (res1.first) { id = res1.second; }
		else { id = res.stencil.size(); res.stencil.push_back(flux2.stencil[counter]); }
		res.a(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)res.a.N, 1 }) +=
			flux2.a(BLOCK_SIZE * counter, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux2.a.N, 1 });
		res.a_biot(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)res.a_biot.N, 1 }) +=
			flux2.a_biot(BLOCK_SIZE * counter, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux2.a.N, 1 });
	}
	res.f.values = flux1.f.values + flux2.f.values;
	res.f_biot.values = flux1.f_biot.values + flux2.f_biot.values;
	
	return res;
}
Matrix pm_discretizer::calc_grad_prev(const index_t cell_id) const
{
	if (grad_prev.size() > 0)
	{
		const auto& g = grad_prev[cell_id];
		Matrix cur_grad(4 * ND, 1);
		index_t id;
		for (int i = 0; i < g.stencil.size(); i++)
		{
			id = g.stencil[i];
			Matrix cur_u({ x_prev[4 * id], x_prev[4 * id + 1], x_prev[4 * id + 2], x_prev[4 * id + 3] }, 4, 1);
			cur_grad += Matrix(g.mat(4 * i, { 12, 4 }, { (size_t)g.mat.N, 1 }), 12, 4) * cur_u;
		}
		cur_grad += g.rhs;
		return cur_grad;
	}
	else
		return Matrix(4 * ND, 1);
}
Matrix pm_discretizer::calc_grad_cur(const index_t cell_id) const
{
	if (grad.size() > 0)
	{
		const auto& g = grad[cell_id];
		Matrix cur_grad(4 * ND, 1);
		index_t id;
		for (int i = 0; i < g.stencil.size(); i++)
		{
			id = g.stencil[i];
			Matrix cur_u({ x_prev[4 * id], x_prev[4 * id + 1], x_prev[4 * id + 2], x_prev[4 * id + 3] }, 4, 1);
			cur_grad += Matrix(g.mat(4 * i, { 12, 4 }, { (size_t)g.mat.N, 1 }), 12, 4) * cur_u;
		}
		cur_grad += g.rhs;
		return cur_grad;
	}
	else
		return Matrix(4 * ND, 1);
}
Matrix pm_discretizer::calc_vector(const Matrix& a, const Matrix& rhs, const vector<index_t>& st) const
{
	Matrix cur_grad(BLOCK_SIZE, 1);
	index_t id;
	for (int i = 0; i < st.size(); i++)
	{
		id = st[i];
		Matrix cur_u({ x_prev[4 * id], x_prev[4 * id + 1], x_prev[4 * id + 2], x_prev[4 * id + 3] }, 4, 1);
		cur_grad += Matrix(a(4 * i, { 12, 4 }, { (size_t)a.N, 1 }), 12, 4) * cur_u;
	}
	cur_grad += rhs;
	return cur_grad;
}
Matrix pm_discretizer::get_u_face_prev(const Matrix dr, const index_t cell_id) const
{
	Matrix u1 ({ x_prev[4 * cell_id], x_prev[4 * cell_id + 1], x_prev[4 * cell_id + 2], x_prev[4 * cell_id + 3] }, 4, 1);
	const auto u_face = u1 + make_block_diagonal(dr.transpose(), 4) * calc_grad_prev(cell_id);
	return Matrix(u_face(0, { 3 }, { 1 }), 3, 1);
};
Matrix pm_discretizer::get_ub_prev(const Face& face) const
{
	const auto& b = bc_prev[face.face_id2];
	const auto& an = b(0, 0);			const auto& bn = b(1, 0);
	const auto& at = b(2, 0);			const auto& bt = b(3, 0);
	const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
	index_t bface_id = n_cells + face.face_id2;
	Matrix ru({ x_prev[4 * bface_id], x_prev[4 * bface_id + 1], x_prev[4 * bface_id + 2]}, 3, 1);
	const value_t rp = x_prev[4 * bface_id + 3];
	// Geometry
	const auto& c1 = cell_centers[face.cell_id1];
	const auto n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
	const auto P = I3 - outer_product(n, n.transpose());
	const value_t rn = (n.transpose() * ru).values[0];
	const auto rt = P * ru;
	double r1 = (n.transpose() * (face.c - c1))(0, 0);		assert(r1 > 0.0);
	const auto y1 = c1 + r1 * n;
	const auto& B1 = biots[face.cell_id1];
	// Stiffness decomposition
	const auto C1 = W * stfs[face.cell_id1] * W.transpose();
	const auto nblock = make_block_diagonal(n, ND);
	const auto nblock_t = make_block_diagonal(n.transpose(), ND);
	const auto tblock = make_block_diagonal(P, ND);
	const auto T1 = nblock_t * C1 * nblock;
	const auto G1 = nblock_t * C1 * tblock;
	// Permeability decomposition
	const auto& perm1 = pm_discretizer::darcy_constant * perms[face.cell_id1];
	const auto K1n = perm1 * n;
	const value_t lam1 = (n.transpose() * K1n)(0, 0);
	const auto gam1 = K1n - lam1 * n;
	// Extra 'boundary' stuff
	auto An = (an * I3 + bn / r1 * T1);
	auto At = (at * I3 + bt / r1 * T1);
	const value_t Ap = 1.0 / (ap + bp / r1 / visc * lam1);
	auto res = At.inv();
	if (!res) 
	{ 
		cout << "Inversion failed!\n";	exit(-1); 
	}
	const auto L = An * At;
	const value_t gamma = 1.0 / (n.transpose() * L * n).values[0];
	const auto gamma_nnt = gamma * outer_product(n, n.transpose());
	const auto gamma_nnt_mult = gamma_nnt * (bn * I3 - bt * L);

	const Matrix u_prev({ x_prev[4 * face.cell_id1], x_prev[4 * face.cell_id1 + 1],
				x_prev[4 * face.cell_id1 + 2] }, 3, 1);
	const value_t p_prev = x_prev[BLOCK_SIZE * face.cell_id1 + 3];
	const auto g_prev = calc_grad_prev(face.cell_id1);
	const auto gu_prev = make_block_diagonal(P, 3) * Matrix(g_prev(0, { ND * ND }, { 1 }), ND * ND, 1);
	const auto gp_prev = P * Matrix(g_prev(ND * ND, { ND }, { 1 }), ND, 1);
	const value_t pb_prev = Ap * (rp + bp / visc * lam1 / r1 * p_prev -
		bp / visc * ((lam1 / r1 * (y1 - face.c) + gam1).transpose() * gp_prev).values[0] +
		bp / visc * (grav_vec * K1n).values[0]);
	const auto mult_p = B1 * n;
	const auto mult_u = At * (bt * I3 + gamma_nnt * (bn * I3 - bt * L));
	const auto gu_coef = T1 / r1 * make_block_diagonal((y1 - face.c).transpose(), ND) + G1;
	Matrix ub_prev(ND, 1);
	if (std::abs(gu_prev.values).max() != 0.0)
	{
		ub_prev = At * (gamma * rn * n + (I3 - gamma_nnt * L) * rt) +
			mult_u * (T1 * u_prev / r1 - gu_coef * gu_prev + pb_prev * mult_p);
	}
	return ub_prev;
}
bool pm_discretizer::check_trans_sum(const vector<index_t>& st, const Matrix& a) const
{
	Matrix sum(BLOCK_SIZE, BLOCK_SIZE);

	for (int i = 0; i < st.size(); i++)
	{
		if (st[i] < cell_centers.size())
		{
			sum.values += a(i * BLOCK_SIZE, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)a.N, 1 });
		}
	}

	return sum == Matrix(BLOCK_SIZE, BLOCK_SIZE);
}
void pm_discretizer::write_trans(const vector<index_t>& st, const Matrix& from)
{
	for (index_t i = 0; i < st.size(); i++)
	{
		auto block = from(BLOCK_SIZE * i, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from.N, 1 });
		block[abs(block) < EQUALITY_TOLERANCE] = 0.0;
		if (abs(block).max() > EQUALITY_TOLERANCE)
		{
			stencil.push_back(st[i]);
			tran.insert(std::end(tran), std::begin(block), std::end(block));
		}
	}
}
void pm_discretizer::write_trans_biot(const vector<index_t>& st, const Matrix& from, const Matrix& from_biot)
{
	value_t m_value, m_value_biot;
	for (st_id = 0; st_id < st.size(); st_id++)
	{
		auto block = from(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from.N, 1 });
		auto block_biot = from_biot(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from_biot.N, 1 });
		block[abs(block) < EQUALITY_TOLERANCE] = 0.0;
		block_biot[abs(block_biot) < EQUALITY_TOLERANCE] = 0.0;
		m_value = abs(block).max();
		m_value_biot = abs(block_biot).max();
		if (m_value > EQUALITY_TOLERANCE || m_value_biot > EQUALITY_TOLERANCE)
		{
			stencil.push_back(st[st_id]);
			tran.insert(std::end(tran), std::begin(block), std::end(block));
			tran_biot.insert(std::end(tran_biot), std::begin(block_biot), std::end(block_biot));
		}
	}
}
void pm_discretizer::write_trans_biot(const vector<index_t>& st, const Matrix& from, const Matrix& from_biot, const Matrix& from_face_unknonws)
{
	value_t m_value, m_value_biot, m_value_face;
	for (st_id = 0; st_id < st.size(); st_id++)
	{
		auto block = from(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from.N, 1 });
		auto block_biot = from_biot(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from_biot.N, 1 });
		auto block_face = from_face_unknonws(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from_biot.N, 1 });
		block[abs(block) < EQUALITY_TOLERANCE] = 0.0;
		block_biot[abs(block_biot) < EQUALITY_TOLERANCE] = 0.0;
		block_face[abs(block_face) < EQUALITY_TOLERANCE] = 0.0;
		m_value = abs(block).max();
		m_value_biot = abs(block_biot).max();
		m_value_face = abs(block_face).max();
		if (m_value > EQUALITY_TOLERANCE || m_value_biot > EQUALITY_TOLERANCE || m_value_face > EQUALITY_TOLERANCE)
		{
			stencil.push_back(st[st_id]);
			tran.insert(std::end(tran), std::begin(block), std::end(block));
			tran_biot.insert(std::end(tran_biot), std::begin(block_biot), std::end(block_biot));
			tran_face_unknown.insert(std::end(tran_face_unknown), std::begin(block_face), std::end(block_face));
		}
	}
}
void pm_discretizer::write_trans_biot_therm_cond(const vector<index_t>& st, const Matrix& from, const Matrix& from_biot, const Matrix& from_th_cond)
{
	for (st_id = 0; st_id < st.size(); st_id++)
	{
		auto block = from(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from.N, 1 });
		auto block_biot = from_biot(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)from_biot.N, 1 });
		block[abs(block) < EQUALITY_TOLERANCE] = 0.0;
		block_biot[abs(block_biot) < EQUALITY_TOLERANCE] = 0.0;
		if (abs(block).max() > EQUALITY_TOLERANCE || abs(block_biot).max() > EQUALITY_TOLERANCE)
		{
			stencil.push_back(st[st_id]);
			tran.insert(std::end(tran), std::begin(block), std::end(block));
			tran_biot.insert(std::end(tran_biot), std::begin(block_biot), std::end(block_biot));
			tran_th_cond.push_back( from_th_cond(0, st_id) );
		}
	}
}
void pm_discretizer::init(const index_t _n_matrix, const index_t _n_fracs, vector<index_t>& _ref_contact_ids)
{
	n_matrix = _n_matrix;
	n_fracs = _n_fracs;
	n_cells = n_matrix + n_fracs;
	ref_contact_ids = _ref_contact_ids;

	inner.resize(n_cells);
	grad.resize(n_cells);
	if (diffs.size()) grad_d.resize(n_cells);
		
	size_t n_cur_faces;
	n_faces = 0;
	nb_faces = bc.size();
	for (int cell_id = 0; cell_id < n_cells; cell_id++)
	{
		const auto& vec_faces = faces[cell_id];
		n_cur_faces = vec_faces.size();
		for (int face_id = 0; face_id < n_cur_faces; face_id++)
		{
			const Face& face = vec_faces[face_id];
			if (face.type == MAT)
			{
				inner[cell_id][face_id] = InnerMatrices();
				auto& cur = inner[cell_id][face_id];
				cur.A1 = Matrix(4, 4);			cur.A2 = Matrix(4, 4);
				cur.Q1 = Matrix(4, 4);			cur.Q2 = Matrix(4, 4);
				cur.Th1 = Matrix(4, 12);		cur.Th2 = Matrix(4, 12);
				cur.R1 = Matrix(4, 1);			cur.R2 = Matrix(4, 1);
				cur.y1 = Matrix(3, 1);			cur.y2 = Matrix(3, 1);
				cur.S1 = Matrix(4, 4);			cur.S2 = Matrix(4, 4);
				cur.T1 = Matrix(ND, ND);		cur.T2 = Matrix(ND, ND);
				cur.G1 = Matrix(ND, ND * ND);	cur.G2 = Matrix(ND, ND * ND);
			}
			n_faces++;
		}

		auto& cur_grad = grad[cell_id];
		cur_grad.stencil.reserve(MAX_STENCIL);
		cur_grad.mat.values.resize(BLOCK_SIZE * ND * BLOCK_SIZE * MAX_STENCIL);
		cur_grad.rhs.values.resize(BLOCK_SIZE * ND);

		if (diffs.size()) 
		{
			auto& cur_grad = grad_d[cell_id];
			//cur_grad.stencil.reserve(MAX_STENCIL);
			cur_grad.mat.values.resize(ND * MAX_STENCIL);
			//cur_grad.rhs.values.resize(ND);
		}
	}

	// Reserving memory for output arrays
	cell_m.reserve(n_faces);
	cell_p.reserve(n_faces);
	offset.reserve(n_faces + 1);
	stencil.reserve(n_faces * MAX_STENCIL);
	tran.reserve(n_faces * MAX_STENCIL * BLOCK_SIZE * BLOCK_SIZE);
	tran_biot.reserve(n_faces * MAX_STENCIL * BLOCK_SIZE * BLOCK_SIZE);
	rhs.reserve(BLOCK_SIZE * n_faces);
	rhs_biot.reserve(BLOCK_SIZE * n_faces);

	tran_th_cond.reserve(n_faces * MAX_STENCIL);
	tran_th_expn.reserve(n_faces * MAX_STENCIL * ND);

	tran_face_unknown.reserve(n_faces * MAX_STENCIL * BLOCK_SIZE * BLOCK_SIZE);
	rhs_face_unknown.reserve(BLOCK_SIZE * n_faces);

	// Preallocations for different number of cell faces
	for (index_t fn = MIN_FACE_NUM; fn <= MAX_FACE_NUM; fn++)
	{
		pre_A[fn] = Matrix(BLOCK_SIZE * fn, BLOCK_SIZE * ND);
		pre_rest[fn] = Matrix(BLOCK_SIZE * fn, 1);
		pre_rhs_mult[fn] = Matrix(BLOCK_SIZE * fn, BLOCK_SIZE * MAX_STENCIL);
		// Heat conduction
		pre_Ad[fn] = Matrix(fn, ND);
		pre_restd[fn] = Matrix(fn, 1);
		pre_rhs_multd[fn] = Matrix(fn, MAX_STENCIL);
		
		pre_frac_grad_mult[fn] = Matrix(BLOCK_SIZE, fn * BLOCK_SIZE);
		pre_Wsvd[fn] = Matrix(fn * BLOCK_SIZE, fn * BLOCK_SIZE);
		pre_w_svd[fn] = Matrix(fn * BLOCK_SIZE, 1);
		pre_Zsvd[fn] = Matrix(fn * BLOCK_SIZE, fn * BLOCK_SIZE);
		for (index_t st_size = 1; st_size < MAX_STENCIL; st_size++)
		{
			pre_cur_rhs[fn][st_size] = Matrix(BLOCK_SIZE * fn, BLOCK_SIZE * st_size);
			if (diffs.size())
				pre_cur_rhsd[fn][st_size] = Matrix(fn, st_size);
		}
	}

	// Preallocations for different size of stencil
	// merge of gradients
	for (index_t i = 0; i <= BLOCK_SIZE; i++)
		pre_merged_grad[i] = Matrix(i * ND, MAX_STENCIL * i);
	pre_merged_stencil.reserve(MAX_STENCIL);

	// allocation of variables used in calculation of fluxes
	fluxes.resize(MAX_FLUXES_NUM);
	fluxes_th_cond.resize(MAX_FLUXES_NUM);
	face_unknowns.resize(MAX_FLUXES_NUM);
	pre_merged_flux.resize(MAX_FLUXES_NUM);
	for (index_t k = 0; k < MAX_FLUXES_NUM; k++)
	{
		// Darcy's, elastic fluxes and Biot's fluxes 
		auto& flux = fluxes[k];
		flux.a = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
		flux.f = Matrix(BLOCK_SIZE, 1);
		flux.a_biot = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
		flux.f_biot = Matrix(BLOCK_SIZE, 1);
		flux.stencil.reserve(MAX_STENCIL);
		// Heat conduction fluxes
		auto& flux_th_cond = fluxes_th_cond[k];
		flux_th_cond.a = Matrix(1, MAX_STENCIL);
		// Approximation of unknowns at interfaces
		auto& face_unknown = face_unknowns[k];
		face_unknown.a = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
		face_unknown.f = Matrix(BLOCK_SIZE, 1);
		// Premerged fluxes
		pre_merged_flux[k].a = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
		pre_merged_flux[k].a_biot = Matrix(BLOCK_SIZE, MAX_STENCIL * BLOCK_SIZE);
		pre_merged_flux[k].f = Matrix(BLOCK_SIZE, 1);
		pre_merged_flux[k].f_biot = Matrix(BLOCK_SIZE, 1);
		pre_merged_flux[k].stencil.reserve(MAX_STENCIL);
	}
}
std::tuple<std::vector<index_t>, std::valarray<value_t>> pm_discretizer::get_gradient(const index_t cell_id)
{
	auto& cur = grad[cell_id];
	return std::make_tuple(cur.stencil, cur.mat.values);
}
std::tuple<std::vector<index_t>, std::valarray<value_t>> pm_discretizer::get_thermal_gradient(const index_t cell_id)
{
	auto& cur = grad_d[cell_id];
	return std::make_tuple(cur.stencil, cur.mat.values);
}

void pm_discretizer::reconstruct_gradients_per_cell(value_t dt)
{
	bool isStationary = dt == 0.0 ? true : false;
	if (isStationary) dt = 1.0;

	n_cells = perms.size();
	grad_prev = grad;

	// Variables
	size_t n_cur_faces;
	std::vector<index_t> st;
	std::vector<index_t> admissible_connections(4, 0);
	st.reserve(MAX_STENCIL);
	Matrix n(ND, 1), K1n(ND, 1), K2n(ND, 1), gam1(ND, 1), gam2(ND, 1), B1n(ND, 1), B2n(ND, 1), T1(ND, ND), G1(ND, ND * ND);
	value_t lam1, lam2;
	Matrix C1(9, 9), C2(9, 9);
	Matrix nblock(9, 3), nblock_t(3, 9), tblock(9, 9);
	bool res, grad_p_depends_on_displacements;
	value_t r1, Ap, gamma, sign, angle;
	Matrix y1(ND, 1), An(ND, ND), At(ND, ND), L(ND, ND), P(ND, ND), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), mult_p(ND, 1);
	Matrix sq_mat(BLOCK_SIZE * ND, BLOCK_SIZE * ND), Wsvd(BLOCK_SIZE * ND, BLOCK_SIZE * ND), frac_grad_mult(BLOCK_SIZE, BLOCK_SIZE * ND);
	value_t k_stab1, k_stab2, c_stab1, c_stab2;
	Matrix B1nn(ND * ND, 1), B2nn(ND * ND, 1);
	value_t max_dgrad_p_du = 0.0;

	int face_id, face_id1, face_count_id, cell_id, bface_id, st_id, loop_face_id;
	value_t r2_frac;

	// Gradient reconstruction in fracture cells
	for (cell_id = 0; cell_id < n_fracs; cell_id++)
	{
		st.clear();
		admissible_connections.clear();
		const auto& vec_faces = faces[n_matrix + cell_id];
		const auto& cur_face = vec_faces[4];

		// Build the system from the continuity at the interfaces
		n_cur_faces = 0;// vec_faces.size();
		for (face_id = 0; face_id < vec_faces.size(); face_id++)
		{
			const Face& face = vec_faces[face_id];
			if (!face.is_impermeable)
			{
				// try to avoid taking into account other crossing faults in gradient reconstruction
				if (face.type == FRAC)
				{
					angle = acos(fabs((cur_face.n.transpose() * face.n).values[0]));
					if ((M_PI / 2 - angle) <= M_PI / 10) admissible_connections.push_back(face_id);
				}
				else if (face.type == FRAC_BOUND) admissible_connections.push_back(face_id);
			}
		}

		n_cur_faces = admissible_connections.size();
		auto& A = pre_A[n_cur_faces];
		auto& rhs_mult = pre_rhs_mult[n_cur_faces];
		auto& rest = pre_rest[n_cur_faces];
		// Cleaning
		std::fill_n(&A.values[0], A.values.size(), 0.0);
		std::fill_n(&rhs_mult.values[0], rhs_mult.values.size(), 0.0);
		std::fill_n(&rest.values[0], rest.values.size(), 0.0);

		face_count_id = 0;
		for (const auto& face_id : admissible_connections)
		{
			const Face& face = vec_faces[face_id];
			if (face.type == FRAC)
			{
				const int& cell_id1 = face.cell_id1;
				const int& cell_id2 = face.cell_id2;
				const auto& c1 = cell_centers[cell_id1];
				const auto& c2 = cell_centers[cell_id2];
				// pressure condition
				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
				r2_frac = ((c2 - face.c).transpose() * (c2 - face.c)).values[0];
				lam2 = (n.transpose() * K2n)(0, 0);
				// gap
				A(face_count_id * BLOCK_SIZE * A.N,
					{ ND, ND * ND },
					{ (size_t)A.N, 1 }) = make_block_diagonal((c2 - c1).transpose(), ND).values;
				// pressure
				A((face_count_id * BLOCK_SIZE + ND) * A.N + ND * ND, { ND }, { 1 }) = (c2 - c1 + r2_frac / lam2 * (K1n - K2n)).values;

				res1 = findInVector(st, face.cell_id1);
				if (res1.first) { id = res1.second; }
				else { id = st.size(); st.push_back(face.cell_id1); }
				rhs_mult(BLOCK_SIZE * (face_count_id * rhs_mult.N + id),
					{ ND, ND },
					{ (size_t)rhs_mult.N, 1 }) += -I3.values;
				rhs_mult(BLOCK_SIZE * face_count_id + ND, BLOCK_SIZE * id + ND) = -1.0;

				res2 = findInVector(st, face.cell_id2);
				if (res2.first) { id = res2.second; }
				else { id = st.size(); st.push_back(face.cell_id2); }
				rhs_mult(BLOCK_SIZE * (face_count_id * rhs_mult.N + id),
					{ ND, ND },
					{ (size_t)rhs_mult.N, 1 }) += I3.values;
				rhs_mult(BLOCK_SIZE * face_count_id + ND, BLOCK_SIZE * id + ND) = 1.0;

				rest(BLOCK_SIZE * face_count_id + ND, 0) = r2_frac / lam2 * dt / visc * ((grav_vec * K1n).values[0] - (grav_vec * K2n).values[0]);

				face_count_id++;
			}
			else if (face.type == FRAC_BOUND)
			{
				const auto& b = bc[face.face_id2];
				const auto& an = b(0, 0);			const auto& bn = b(1, 0);
				const auto& at = b(2, 0);			const auto& bt = b(3, 0);
				const auto& ap = b(4, 0);			const auto& bp = b(5, 0);

				const auto& c1 = cell_centers[face.cell_id1];
				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
				nblock_t = make_block_diagonal(face.n.transpose(), ND);
				K1n = pm_discretizer::darcy_constant * perms[face.cell_id1] * n;

				A(face_count_id * BLOCK_SIZE * A.N,
					{ ND, ND * ND },
					{ (size_t)A.N, 1 }) = (at * make_block_diagonal((face.c - c1).transpose(), ND) + bt * nblock_t).values;

				A((face_count_id * BLOCK_SIZE + ND) * A.N + ND * ND, { ND }, { 1 }) = K1n.values;

				res1 = findInVector(st, face.cell_id1);
				if (res1.first) { id = res1.second; }
				else { id = st.size(); st.push_back(face.cell_id1); }

				rhs_mult(BLOCK_SIZE * (face_count_id * rhs_mult.N + id),
					{ ND, ND },
					{ (size_t)rhs_mult.N, 1 }) += -at * I3.values;
				rest(BLOCK_SIZE * face_count_id + ND, 0) = (grav_vec * K1n).values[0];

				face_count_id++;
			}
		}

		auto& cur_grad = grad[n_matrix + cell_id];
		cur_grad.stencil = st;
		auto& cur_rhs = pre_cur_rhs[n_cur_faces][st.size()];
		cur_rhs.values = rhs_mult(0,
			{ BLOCK_SIZE * (n_cur_faces), BLOCK_SIZE * st.size() },
			{ (size_t)rhs_mult.N, 1 });

		// SVD is produced for M x N matrix where M >= N, decompose transposed otherwise
		face_count_id = (face_count_id > ND) ? ND : face_count_id;
		//assert(face_count_id == n_cur_faces);
		auto& Wsvd = pre_Wsvd[face_count_id];
		auto& Zsvd = pre_Zsvd[face_count_id];
		auto& w_svd = pre_w_svd[face_count_id].values;
		std::fill_n(&Wsvd.values[0], Wsvd.values.size(), 0.0);
		std::fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
		std::fill_n(&w_svd[0], w_svd.size(), 0.0);
		if (A.M >= A.N)
		{
			// SVD decomposition A = M W Z*
			if (Zsvd.M != A.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
			res = A.svd(Zsvd, w_svd);
			assert(Zsvd.M == face_count_id * BLOCK_SIZE && Zsvd.N == face_count_id * BLOCK_SIZE);
			if (!res) { cout << "SVD failed!\n"; /*sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");*/ exit(-1); }
			// check SVD
			//Wsvd.set_diagonal(w_svd);
			//assert(A == M * Wsvd * Zsvd.transpose());
			for (index_t i = 0; i < w_svd.size(); i++)
				w_svd[i] = (fabs(w_svd[i]) < 1000 * EQUALITY_TOLERANCE) ? 0.0 /*w_svd[i]*/ : 1.0 / w_svd[i];
			Wsvd.set_diagonal(w_svd);
			// check pseudo-inverse
			//Matrix Ainv = Zsvd * Wsvd * M.transpose();
			//assert(A * Ainv * A == A && Ainv * A * Ainv == Ainv);
			cur_grad.mat = Zsvd * Wsvd * A.transpose() * cur_rhs;
			cur_grad.rhs = Zsvd * Wsvd * A.transpose() * rest;
		}
		else
		{
			// SVD decomposition A* = M W Z*
			A.transposeInplace();
			if (Zsvd.M != A.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
			res = A.svd(Zsvd, w_svd);
			assert(Zsvd.M == face_count_id * BLOCK_SIZE && Zsvd.N == face_count_id * BLOCK_SIZE);
			if (!res) { cout << "SVD failed!\n"; /*sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");*/ exit(-1); }
			// check SVD
			//Wsvd.set_diagonal(w_svd);
			//assert(A.transpose() == M * Wsvd * Zsvd.transpose());
			for (index_t i = 0; i < w_svd.size(); i++)
				w_svd[i] = (fabs(w_svd[i]) < 1000 * EQUALITY_TOLERANCE) ? 0.0 /*w_svd[i]*/ : 1.0 / w_svd[i];
			Wsvd.set_diagonal(w_svd);
			// check pseudo-inverse
			//Matrix Ainv = M * Wsvd * Zsvd.transpose();
			//assert(A * Ainv * A == A && Ainv * A * Ainv == Ainv);
			cur_grad.mat = A * Wsvd * Zsvd.transpose() * cur_rhs;
			cur_grad.rhs = A * Wsvd * Zsvd.transpose() * rest;
			A.transposeInplace();
		}
	}

	// Gradient reconstruction in matrix cells
	for (cell_id = 0; cell_id < n_matrix; cell_id++)
	{
		st.clear();
		const auto& vec_faces = faces[cell_id];

		// Build the system from the continuity at the interfaces
		n_cur_faces = 0;// vec_faces.size();
		for (face_id = 0; face_id < vec_faces.size(); face_id++)
		{
			const Face& face = vec_faces[face_id];
			if (face.type == BORDER)
			{
				const auto& b = bc[face.face_id2];
				const auto& an = b(0, 0);			const auto& bn = b(1, 0);
				const auto& at = b(2, 0);			const auto& bt = b(3, 0);
				const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
				if (NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION || an != 0.0 || at != 0.0)	n_cur_faces++;
			}
			else if (face.type != MAT_TO_FRAC) n_cur_faces++;
		}

		auto& A = pre_A[n_cur_faces];
		auto& rest = pre_rest[n_cur_faces];
		auto& rhs_mult = pre_rhs_mult[n_cur_faces];
		// Cleaning
		std::fill_n(&A.values[0], A.values.size(), 0.0);
		std::fill_n(&rest.values[0], rest.values.size(), 0.0);
		std::fill_n(&rhs_mult.values[0], rhs_mult.values.size(), 0.0);

		face_id = 0;
		for (loop_face_id = 0; loop_face_id < vec_faces.size(); loop_face_id++)
		{
			const Face& face = vec_faces[loop_face_id];
			if (face.type == MAT)
			{
				// Clean matrices
				auto& cur = inner[cell_id][loop_face_id];
				std::fill_n(&cur.A1.values[0],	cur.A1.values.size(), 0.0);
				std::fill_n(&cur.A2.values[0],	cur.A2.values.size(), 0.0);
				std::fill_n(&cur.Q1.values[0],	cur.Q1.values.size(), 0.0);
				std::fill_n(&cur.Q2.values[0],	cur.Q2.values.size(), 0.0);
				std::fill_n(&cur.Th1.values[0], cur.Th1.values.size(), 0.0);
				std::fill_n(&cur.Th2.values[0], cur.Th2.values.size(), 0.0);
				std::fill_n(&cur.R1.values[0],	cur.R1.values.size(), 0.0);
				std::fill_n(&cur.R2.values[0],	cur.R2.values.size(), 0.0);

				const int& cell_id1 = face.cell_id1;
				const int& cell_id2 = face.cell_id2;
				const auto& c1 = cell_centers[cell_id1];	
				const auto& c2 = cell_centers[cell_id2];
				n = (face.n.transpose() * (c2 - c1)).values[0] > 0 ? face.n : -face.n;
				P = I3 - outer_product(n, n.transpose());
				// Permeability decomposition
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
				lam1 = (n.transpose() * K1n)(0, 0);
				lam2 = (n.transpose() * K2n)(0, 0);
				gam1 = K1n - lam1 * n;
				gam2 = K2n - lam2 * n;
				// Stiffness decomposition
				C1 = W * stfs[cell_id1] * W.transpose();
				C2 = W * stfs[cell_id2] * W.transpose();
				nblock = make_block_diagonal(n, ND);
				nblock_t = make_block_diagonal(n.transpose(), ND);
				tblock = make_block_diagonal(P, ND);
				cur.T1 = nblock_t * C1 * nblock;
				cur.T2 = nblock_t * C2 * nblock;
				cur.G1 = nblock_t * C1 * tblock;
				cur.G2 = nblock_t * C2 * tblock;
				// Process geometry
				auto& r1 = cur.r1;
				auto& r2 = cur.r2;
				r1 = (n.transpose() * (face.c - c1))(0, 0);
				r2 = (n.transpose() * (c2 - face.c))(0, 0);
				assert(r1 > 0.0);		assert(r2 > 0.0);
				auto& y1 = cur.y1;
				auto& y2 = cur.y2;
				y1 = c1 + r1 * n;	 y2 = c2 - r2 * n;
				// Assemble matrices
				auto& A1 = cur.A1;						auto& A2 = cur.A2;
				auto& Q1 = cur.Q1;						auto& Q2 = cur.Q2;
				auto& R1 = cur.R1;						auto& R2 = cur.R2;
				B1n = biots[cell_id1] * n;
				B2n = biots[cell_id2] * n;

				A1(3, { 3, 1 }, { 4, 1 }) = B1n.values;				A2(3, { 3, 1 }, { 4, 1 }) = B2n.values;
				if (!isStationary)
				{
					A1(4 * 3, { 3 }, { 1 }) = B1n.transpose().values;	A2(4 * 3, { 3 }, { 1 }) = B2n.transpose().values;
					R1(3, 0) = -(B1n.transpose() * get_u_face_prev(face.c - c1, cell_id1)).values[0];
					R2(3, 0) = -(B2n.transpose() * get_u_face_prev(face.c - c2, cell_id2)).values[0];
				}
				Q1(0, { 3, 3 }, { 4, 1 }) = -cur.T1.values;			
				Q1(3, 3) = -lam1 * dt / visc;		
				Q1 += r1 * A1;
				Q2(0, { 3, 3 }, { 4, 1 }) = -cur.T2.values;			
				Q2(3, 3) = -lam2 * dt / visc;		
				Q2 -= r2 * A2;
				auto& Th1 = cur.Th1;	
				auto& Th2 = cur.Th2;	
				Th1(0, { 3, 9 }, { 12, 1 }) = -cur.G1.values;
				Th2(0, { 3, 9 }, { 12, 1 }) = -cur.G2.values;
				Th1(45, { 3 }, { 1 }) = -dt / visc * gam1.transpose().values;
				Th2(45, { 3 }, { 1 }) = -dt / visc * gam2.transpose().values;
				Th1 += A1 * make_block_diagonal((face.c - y1).transpose(), 4);
				Th2 += A2 * make_block_diagonal((face.c - y2).transpose(), 4);
				R1(3, 0) += (dt / visc * grav_vec * K1n).values[0];
				R2(3, 0) += (dt / visc * grav_vec * K2n).values[0];



				// to consider impermeable faces inside domain
				if (face.is_impermeable)
				{
					// mechanics
					// Matrix
					A(4 * face_id * 4 * ND, {ND, ND * ND}, {4 * ND, 1}) = -(r2 * (cur.G1 - cur.G2) * make_block_diagonal(I3 - outer_product(n, n.transpose()), ND) +
						(r2 * cur.T1 + r1 * cur.T2) * make_block_diagonal(n.transpose(), ND)).values;
					A(4 * face_id * 4 * ND, { ND, ND * ND }, { 4 * ND, 1 }) += (-cur.T2 * make_block_diagonal((y2 - y1).transpose(), ND)).values;
					A(4 * face_id * 4 * ND + ND * ND, { ND, ND }, { 4 * ND, 1 }) += r2 * ((outer_product(B1n, ((face.c - y1) - r1 / lam1 * gam1).transpose()) - 
																							outer_product(B2n, ((face.c - y2) - r2 / lam2 * gam2).transpose())) *
																								(I3 - outer_product(n, n.transpose()))).values;

					res1 = findInVector(st, cell_id1);
					if (res1.first) { id = res1.second; }
					else { id = st.size(); st.push_back(cell_id1); }
					rhs_mult(BLOCK_SIZE * face_id * rhs_mult.N + BLOCK_SIZE * id, { ND, ND }, { (size_t)rhs_mult.N, 1 }) += cur.T2.values;
					rhs_mult(BLOCK_SIZE * face_id * rhs_mult.N + BLOCK_SIZE * id + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += -r2 * B1n.values;

					res2 = findInVector(st, cell_id2);
					if (res2.first) { id = res2.second; }
					else { id = st.size(); st.push_back(cell_id2); }
					rhs_mult(BLOCK_SIZE * face_id * rhs_mult.N + BLOCK_SIZE * id, { ND, ND }, { (size_t)rhs_mult.N, 1 }) += -cur.T2.values;
					rhs_mult(BLOCK_SIZE * face_id * rhs_mult.N + BLOCK_SIZE * id, { ND, ND }, { (size_t)rhs_mult.N, 1 }) += r2 * B2n.values;
					
					// flow
					//rhs_mult((BLOCK_SIZE * face_id + ND) * rhs_mult.N, { (size_t)rhs_mult.N }, { 1 }) = 0.0;
					A((BLOCK_SIZE * face_id + ND) * A.N, { BLOCK_SIZE * ND }, { 1 }) = 0.0;
					A((BLOCK_SIZE * face_id + ND) * A.N + ND * ND, { ND }, { 1 }) = dt * K1n.values / visc;
					rest(BLOCK_SIZE * face_id + ND, 0) = R1(3, 0);
					if (face.is_impermeable == 1)
					{
						Q1(ND, ND) = 0.0;
						Th1(45, { ND }, { 1 }) = 0.0;
					}
					else if (face.is_impermeable == 2)
					{
						Q2(ND, ND) = 0.0;
						Th2(45, { ND }, { 1 }) = 0.0;
					}
				}
				else
				{
					// Matrix
					A(4 * face_id * 4 * ND, { 4, 4 * ND }, { 4 * ND, 1 }) = (r2 * (Th1 - Th2) * make_block_diagonal(I3 - outer_product(n, n.transpose()), 4) +
						(r2 * Q1 + r1 * Q2) * make_block_diagonal(n.transpose(), 4)).values;
					A(4 * face_id * 4 * ND, { 4, 4 * ND }, { 4 * ND, 1 }) += (Q2 * make_block_diagonal((y2 - y1).transpose(), BLOCK_SIZE)).values;
					/*(Q2 * make_block_diagonal((c2 - c1).transpose(), 4) +
					r2 * (Q1 - Q2) * make_block_diagonal(n.transpose(), 4) + r2 * (Th1 - Th2)).values;*/
					// RHS
					res1 = findInVector(st, cell_id1);
					if (res1.first) { id = res1.second; }
					else { id = st.size(); st.push_back(cell_id1); }
					rhs_mult(4 * face_id * rhs_mult.N + 4 * id, { 4, 4 }, { (size_t)rhs_mult.N, 1 }) += -(Q2 + r2 * A1).values;

					res2 = findInVector(st, cell_id2);
					if (res2.first) { id = res2.second; }
					else { id = st.size(); st.push_back(cell_id2); }
					rhs_mult(4 * face_id * rhs_mult.N + 4 * id, { 4, 4 }, { (size_t)rhs_mult.N, 1 }) += (Q2 + r2 * A2).values;

					rest(BLOCK_SIZE * face_id, { BLOCK_SIZE }, { 1 }) = r2 * (R2 - R1).values;
				}

				face_id++;

				// stabilization parameters
				if (face.is_impermeable)
				{
					if (face.is_impermeable == 1) { cur.k_stab1 = 0.0; cur.k_stab2 = lam2 / r2; }
					else if (face.is_impermeable == 2) { cur.k_stab1 = lam1 / r1; cur.k_stab2 = 0.0; }
				}
				else
				{
					cur.k_stab1 = lam1 / r1;
					cur.k_stab2 = lam2 / r2;
				}
				cur.beta_stab1 = sqrt((B1n.transpose() * B1n).values[0]);
				cur.beta_stab2 = sqrt((B2n.transpose() * B2n).values[0]);
				B1nn.values = (make_block_diagonal(n, ND) * B1n).values;
				B2nn.values = (make_block_diagonal(n, ND) * B2n).values;
				cur.c_stab1 = (B1nn.transpose() * C1 * B1nn).values[0] / r1 / cur.beta_stab1 / cur.beta_stab1;
				cur.c_stab2 = (B2nn.transpose() * C2 * B2nn).values[0] / r2 / cur.beta_stab2 / cur.beta_stab2;
				//cur.alpha_min_stab1 = (sqrt((k_stab1 - c_stab1) * (k_stab1 - c_stab1) + 4 * cur.beta_stab1 * cur.beta_stab1 / dt) - (k_stab1 + c_stab1)) / (2 * cur.beta_stab1);
				//cur.alpha_min_stab2 = (sqrt((k_stab2 - c_stab2) * (k_stab2 - c_stab2) + 4 * cur.beta_stab2 * cur.beta_stab2 / dt) - (k_stab2 + c_stab2)) / (2 * cur.beta_stab2);
				
				cur.S1(0, { ND, ND }, { 4, 1 }) = outer_product(B1n, B1n.transpose()).values / cur.beta_stab1;
				cur.S1(ND, ND) = cur.beta_stab1;
				// cur.S1.values *= std::max(cur.alpha_min_stab1, 1.0);

				cur.S2(0, { ND, ND }, { 4, 1 }) = outer_product(B2n, B2n.transpose()).values / cur.beta_stab2;
				cur.S2(ND, ND) = cur.beta_stab2;
				// cur.S2.values *= std::max(cur.alpha_min_stab2, 1.0);
			}
			else if (face.type == BORDER)
			{
				const auto& b = bc[face.face_id2];
				const auto& an = b(0, 0);			const auto& bn = b(1, 0);			 
				const auto& at = b(2, 0);			const auto& bt = b(3, 0);			
				const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
				// Skip if pure neumann
				if (!NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION && an == 0.0 && at == 0.0)	continue;

				const int& cell_id1 = face.cell_id1;
				// Geometry
				const auto& c1 = cell_centers[cell_id1];
				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
				P = I3 - outer_product(n, n.transpose());
				r1 = (n.transpose() * (face.c - c1))(0, 0);		assert(r1 > 0.0);
				y1 = c1 + r1 * n;
				B1n = biots[cell_id1] * n;
				// Stiffness decomposition
				C1 = W * stfs[cell_id1] * W.transpose();
				nblock = make_block_diagonal(n, ND);
				nblock_t = make_block_diagonal(n.transpose(), ND);
				tblock = make_block_diagonal(P, ND);
				T1 = nblock_t * C1 * nblock;
				G1 = nblock_t * C1 * tblock;
				// Permeability decomposition
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				lam1 = (n.transpose() * K1n)(0, 0);
				gam1 = K1n - lam1 * n;
				// Extra 'boundary' stuff
				An = (an * I3 + bn / r1 * T1);
				At = (at * I3 + bt / r1 * T1);
				Ap = 1.0 / (ap + bp / r1 / visc * lam1);
				res = At.inv();
				if (!res) 
				{ 
					cout << "Inversion failed!\n";	exit(-1); 
				}
				L = An * At;
				gamma = 1.0 / (n.transpose() * L * n).values[0];
				gamma_nnt = gamma * outer_product(n, n.transpose());
				gamma_nnt_mult = gamma_nnt * (bn * I3 - bt * L);
				mult_p = (bt * I3 + gamma_nnt_mult) * B1n;
				// Filling mechanics equations
				A(4 * face_id * 4 * ND, 
					{ ND, 3 * ND }, 
					{ 4 * ND, 1 }) = (at * make_block_diagonal((face.c - c1).transpose(), ND) + 
										bt * nblock_t * C1 + 
										gamma_nnt_mult * (G1 + T1 / r1 *
											make_block_diagonal( (y1 - face.c).transpose(), ND ))).values;
				A(4 * face_id * 4 * ND + 3 * ND,
					{ ND, ND },
					{ 4 * ND, 1 }) = outer_product(mult_p, Ap * bp / visc * ((lam1 / r1 * (y1 - face.c) + gam1).transpose() *
										P)).values;

				res1 = findInVector(st, face.cell_id1);
				if (res1.first) { id = res1.second; }
				else { id = st.size(); st.push_back(face.cell_id1); }

				rhs_mult(4 * face_id * rhs_mult.N + 4 * id, { 3, 3 }, { (size_t)rhs_mult.N, 1 }) += (gamma_nnt_mult * T1 / r1 - at * I3).values;
				rhs_mult(4 * face_id * rhs_mult.N + 4 * id + 3, { 3, 1 }, { (size_t)rhs_mult.N, 1 }) += (Ap * bp / visc * lam1 / r1 * mult_p).values;
				rest(4 * face_id, { 3 }, { 1 }) = (mult_p * Ap * bp / visc * (grav_vec * K1n).values[0]).values;
				// Filling flow equation
				A((4 * face_id + 3) * 4 * ND + 3 * ND,
					{ ND },
					{ 1 }) = (ap * (face.c - c1) + bp / visc * K1n).values;
				rhs_mult(4 * face_id + 3, 4 * id + 3) -= ap;
				rest(4 * face_id + 3, 0) = bp / visc * (grav_vec * K1n).values[0];
				// BC RHS coefficients
				bface_id = n_cells + face.face_id2;
				res2 = findInVector(st, bface_id);
				if (res2.first) { id = res2.second; }
				else { id = st.size(); st.push_back(bface_id); }
				// Mech
				rhs_mult(4 * face_id * rhs_mult.N + 4 * id, { 3, 3 }, { (size_t)rhs_mult.N, 1 }) = (gamma_nnt + (I3 - gamma_nnt * L) * P).values;
				rhs_mult(4 * face_id * rhs_mult.N + 4 * id + 3, { 3, 1 }, { (size_t)rhs_mult.N, 1 }) = (mult_p * Ap).values;
				// Flow
				rhs_mult(4 * face_id + 3, 4 * id + 3) = 1.0;

				face_id++;
 			}
			else if (face.type == MAT_TO_FRAC)
			{
				const Face& face0 = vec_faces[face.face_id1];
				const auto& cur = inner[cell_id][face.face_id1];
				// recount original face id
				face_id1 = 0;
				for (index_t h = 0; h < face.face_id1; h++)
				{
					const Face& face = vec_faces[h];
					if (face.type == BORDER)
					{
						const auto& b = bc[face.face_id2];
						const auto& an = b(0, 0);	const auto& at = b(2, 0);
						if (an != 0.0 || at != 0.0)	face_id1++;
					}
					else if (face.type != MAT_TO_FRAC) face_id1++;
				}

				const int& cell_id1 = face0.cell_id1;
				const int& frac_id = face.cell_id2;
				assert(frac_id >= n_matrix);
				const auto& c1 = cell_centers[cell_id1];
				const auto& c2 = cell_centers[frac_id];
				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
				sign = get_fault_sign(n, ref_contact_ids[frac_id - n_matrix]);
				// add gap
				res1 = findInVector(st, frac_id);
				if (res1.first) { id1 = res1.second; }
				else { id1 = st.size(); st.push_back(frac_id); }
				rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id1,
						{ BLOCK_SIZE, BLOCK_SIZE }, 
						{ (size_t)rhs_mult.N, 1 }) += -sign * (cur.Q2 + cur.r2 * cur.A1).values;
				// add gap gradient
				const auto& frac_grad = grad[frac_id];
				auto& frac_grad_mult = pre_frac_grad_mult[frac_grad.stencil.size()];
				std::fill_n(&frac_grad_mult.values[0], frac_grad_mult.values.size(), 0.0);
				frac_grad_mult = sign * (cur.Q2 * make_block_diagonal((cur.y2 - face.c).transpose(), BLOCK_SIZE) - cur.r2 * cur.Th2) * frac_grad.mat;
				for (st_id = 0; st_id < frac_grad.stencil.size(); st_id++)
				{
					assert(frac_grad.stencil[st_id] >= n_matrix);
					res1 = findInVector(st, frac_grad.stencil[st_id]);
					if (res1.first) { id = res1.second; }
					else { id = st.size(); st.push_back(frac_grad.stencil[st_id]); }
					rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id,
						{ BLOCK_SIZE, BLOCK_SIZE },
						{ (size_t)rhs_mult.N, 1 }) -= frac_grad_mult(	st_id * BLOCK_SIZE,
																		{ BLOCK_SIZE, BLOCK_SIZE }, 
																		{ (size_t)frac_grad_mult.N, 1});
				}
				// no discontinuity in pressure
				rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = 0.0;
				
				// pressure condition
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				K2n = pm_discretizer::darcy_constant * perms[frac_id] * n;
				r2_frac = frac_apers[frac_id - n_matrix] / 2;
				lam2 = (n.transpose() * K2n)(0, 0);
				
				// remove previous numbers
				A((BLOCK_SIZE * face_id1 + ND) * A.N, { BLOCK_SIZE * ND}, { 1 }) = 0.0;
				rhs_mult((BLOCK_SIZE * face_id1 + ND) * rhs_mult.N, { (size_t)rhs_mult.N }, { 1 }) = 0.0;
				rest(BLOCK_SIZE * face_id1 + ND, 0) = 0.0;
				// fill newer
				if (face.is_impermeable)
				{
					A((BLOCK_SIZE* face_id1 + ND) * A.N + ND * ND, { ND }, { 1 }) = K1n.values;
					rest(BLOCK_SIZE* face_id1 + ND, 0) = (grav_vec * K1n).values[0];
				} 
				else
				{
					A((BLOCK_SIZE * face_id1 + ND) * A.N + ND * ND, { ND }, { 1 }) = (c2 - c1 + r2_frac / lam2 * (K1n - K2n)).values;
					res1 = findInVector(st, cell_id1);
					if (res1.first) { id = res1.second; }
					else { id = st.size(); st.push_back(cell_id1); }
					rhs_mult(BLOCK_SIZE * face_id1 + ND, BLOCK_SIZE * id + ND) = -1.0;

					res2 = findInVector(st, frac_id);
					if (res2.first) { id = res2.second; }
					else { id = st.size(); st.push_back(frac_id); }
					rhs_mult(BLOCK_SIZE * face_id1 + ND, BLOCK_SIZE * id + ND) = 1.0;

					rest(BLOCK_SIZE * face_id1 + ND, 0) = r2_frac / lam2 * ((grav_vec * K1n).values[0] - (grav_vec * K2n).values[0]);
				}
				//face_id++;
			}
		}
		
		// scaling of equations for pressure gradients for better condition number
		const value_t PRESSURE_CONDITION_MULTIPLIER = (stfs[cell_id].values[0] + stfs[cell_id].values[7] + stfs[cell_id].values[14]) / 3 / 
					(pm_discretizer::darcy_constant * (perms[cell_id].values[0] + perms[cell_id].values[4] + perms[cell_id].values[8]) / 3);
		for (index_t ii = ND; ii < A.M; ii += BLOCK_SIZE)
		{
			for (index_t jj = 0; jj < A.N; jj++)
			{
				A(ii, jj) *= PRESSURE_CONDITION_MULTIPLIER;
			}
			for (index_t jj = 0; jj < BLOCK_SIZE * st.size(); jj++)
			{
				rhs_mult(ii, jj) *= PRESSURE_CONDITION_MULTIPLIER;
			}
			rest(ii, 0) *= PRESSURE_CONDITION_MULTIPLIER;
		}

		auto& cur_rhs = pre_cur_rhs[n_cur_faces][st.size()];
		cur_rhs.values = rhs_mult(0, { 4 * n_cur_faces, 4 * st.size() }, { (size_t)rhs_mult.N, 1 });
		auto& cur_grad = grad[cell_id];
		cur_grad.stencil = st;

		sq_mat = A.transpose() * A;

		//// check if pressure gradients are well defined, i.e. there are at least 3 conditions
		// calculate some reference
		value_t ref_value = 0.0, tmp_value;
		for (index_t ii = 0; ii < sq_mat.M; ii++)
		{
			tmp_value = sqrt(fabs(sq_mat(ii, ii)));
			if (ref_value < tmp_value)
				ref_value = tmp_value;
		}
		// check number of pressure conditions with above-precision coefficients
		index_t num_conditions = 0;
		for (index_t ii = ND; ii < A.M; ii += BLOCK_SIZE)
		{
			tmp_value = std::max(fabs(A(ii, ND* ND)), std::max(fabs(A(ii, ND* ND + 1)), fabs(A(ii, ND* ND + 2))));
			if (tmp_value > EQUALITY_TOLERANCE * ref_value)
				num_conditions++;
		}

		if (num_conditions < ND)
		{
			printf("Pressure gradients in cell %d are not well-defined, regularization applied\n", cell_id);
			for (index_t dd = 0; dd < sq_mat.M; dd++)
				sq_mat(dd, dd) += 100.0 * EQUALITY_TOLERANCE;
		}

		res = sq_mat.inv();
		if (!res) 
		{ 
			cout << "Inversion failed!\n";	
			//sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");
			exit(-1); 
		}
		if (sq_mat.is_nan())
		{
			face_count_id = (face_id > ND) ? ND : face_id;
			auto& Wsvd = pre_Wsvd[face_count_id];
			auto& Zsvd = pre_Zsvd[face_count_id];
			auto& w_svd = pre_w_svd[face_count_id].values;
			std::fill_n(&Wsvd.values[0], Wsvd.values.size(), 0.0);
			std::fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
			std::fill_n(&w_svd[0], w_svd.size(), 0.0);

			// SVD decomposition A = M W Z*
			if (Zsvd.M != A.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
			res = A.svd(Zsvd, w_svd);
			assert(Zsvd.M == face_count_id * BLOCK_SIZE && Zsvd.N == face_count_id * BLOCK_SIZE);
			if (!res) { cout << "SVD failed!\n"; /*sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");*/ exit(-1); }
			// check SVD
			// Wsvd.set_diagonal(w_svd);
			// assert(A == M * Wsvd * Zsvd.transpose());
			for (index_t i = 0; i < w_svd.size(); i++)
				w_svd[i] = (fabs(w_svd[i]) < 1000 * EQUALITY_TOLERANCE) ? 0.0 /*w_svd[i]*/ : 1.0 / w_svd[i];
			Wsvd.set_diagonal(w_svd);
			// check pseudo-inverse
			// Matrix Ainv = Zsvd * Wsvd * M.transpose();
			// assert(A * Ainv * A == A && Ainv * A * Ainv == Ainv);
			cur_grad.mat = Zsvd * Wsvd * A.transpose() * cur_rhs;
			cur_grad.rhs = Zsvd * Wsvd * A.transpose() * rest;
		}
		else
		{
			cur_grad.mat = sq_mat * A.transpose() * cur_rhs;
			cur_grad.rhs = sq_mat * A.transpose() * rest;
		}

		//  check if pressure gradients depend on displacements (they should not)
		grad_p_depends_on_displacements = false;
		for (index_t ii = ND * ND; ii < cur_grad.mat.M; ii++)
		{
			for (index_t jj = 0; jj < st.size(); jj++)
			{
				for (index_t kk = 0; kk < ND; kk++)
				{
					value_t tmp = fabs(cur_grad.mat(ii, jj * BLOCK_SIZE + kk));
					max_dgrad_p_du = std::max(max_dgrad_p_du, tmp);
					if (tmp > EQUALITY_TOLERANCE)
						grad_p_depends_on_displacements = true;
				}
			}
		}
		//if (grad_p_depends_on_displacements)
		//	printf("Pressure gradients in cell %d depend on displacements!\n", cell_id);
	}
	
	printf("Pressure gradients depend on displacements with a max coefficient: %.3e\n", max_dgrad_p_du);

	printf("Gradient reconstruction was done!\n");
}
//void pm_discretizer::reconstruct_gradients_per_node(value_t dt, index_t n_nodes)
//{
//	bool isStationary = dt == 0.0 ? true : false;
//	if (isStationary) dt = 1.0;
//
//	n_cells = perms.size();
//	grad_prev = grad;
//
//	// Variables
//	size_t n_cur_faces;
//	vector<index_t> admissible_connections(4, 0);
//	index_t id1, id2;
//	Matrix n(ND, 1), K1n(ND, 1), K2n(ND, 1), gam1(ND, 1), gam2(ND, 1), B1n(ND, 1), B2n(ND, 1);
//	value_t lam1, lam2;
//	Matrix C1(9, 9), C2(9, 9), T1(3, 3), T2(3, 3), G1(3, 9), G2(3, 9);
//	Matrix nblock(9, 3), nblock_t(3, 9), tblock(9, 9);
//	bool res;
//	value_t r1, Ap, gamma, sign, angle;
//	Matrix y1(ND, 1), An(ND, ND), At(ND, ND), L(ND, ND), P(ND, ND), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), mult_p(ND, 1);
//	Matrix sq_mat(BLOCK_SIZE * ND, BLOCK_SIZE * ND), Wsvd(BLOCK_SIZE * ND, BLOCK_SIZE * ND), frac_grad_mult(BLOCK_SIZE, BLOCK_SIZE * ND);
//	value_t k_stab1, k_stab2, c_stab1, c_stab2;
//	Matrix B1nn(ND * ND, 1), B2nn(ND * ND, 1);
//
//	int face_id, face_id1, face_count_id, cell_id, node_id, bface_id, st_id, loop_face_id;
//	value_t r2_frac;
//
//	// initialize gradients for each node
//	vector<std::vector<index_t>> node_stencil(n_nodes, std::vector<index_t>(MAX_STENCIL));
//	vector<Matrix> node_A(n_nodes);
//	vector<Matrix> node_rhs_mult(n_nodes);
//	vector<Matrix> node_rest(n_nodes);
//
//	// allocate memory for gradients
//	vector<index_t> faces_per_node(n_nodes, 0);
//	vector<unordered_set<index_t>> nodes_per_cell(n_matrix, std::unordered_set<index_t>(MAX_POINTS_PER_CELL));
//	unordered_set<index_t>::const_iterator it;
//	for (cell_id = 0; cell_id < n_matrix; cell_id++)
//	{
//		const auto& vec_faces = faces[cell_id];
//		for (face_id = 0; face_id < vec_faces.size(); face_id++)
//		{
//			const Face& face = vec_faces[face_id];
//			for (const auto& pt : face.pts)
//				faces_per_node[pt]++;
//		}
//	}
//	for (node_id = 0; node_id < n_nodes; node_id++)
//	{
//		auto& eq_num = faces_per_node[node_id];
//		node_stencil[node_id].clear();
//		node_A[node_id] = Matrix(BLOCK_SIZE * eq_num, BLOCK_SIZE * ND);
//		node_rhs_mult[node_id] = Matrix(BLOCK_SIZE * eq_num, MAX_STENCIL);
//		node_rest[node_id] = Matrix(BLOCK_SIZE * eq_num, 1);
//		eq_num = 0;
//	}
//
//	// Gradient reconstruction in matrix cells
//	for (cell_id = 0; cell_id < n_matrix; cell_id++)
//	{
//		const auto& vec_faces = faces[cell_id];
//		auto& cell_pts = nodes_per_cell[cell_id];
//		cell_pts.clear();
//
//		// Build the system from the continuity at the interfaces
//		/*n_cur_faces = 0;// vec_faces.size();
//		for (face_id = 0; face_id < vec_faces.size(); face_id++)
//		{
//			const Face& face = vec_faces[face_id];
//			if (face.type == BORDER)
//			{
//				const auto& b = bc[face.face_id2];
//				const auto& an = b(0, 0);			const auto& bn = b(1, 0);
//				const auto& at = b(2, 0);			const auto& bt = b(3, 0);
//				const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
//				if (NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION || an != 0.0 || at != 0.0)	n_cur_faces++;
//			}
//			else if (face.type != MAT_TO_FRAC) n_cur_faces++;
//		}*/
//
//		for (loop_face_id = 0; loop_face_id < vec_faces.size(); loop_face_id++)
//		{
//			const Face& face = vec_faces[loop_face_id];
//			node_id = face.pts[0];
//			it = cell_pts.find(node_id);	if (it == cell_pts.end()) { cell_pts.insert(node_id); }
//			auto& A = node_A[node_id];
//			auto& rest = node_rest[node_id];
//			auto& rhs_mult = node_rhs_mult[node_id];
//			auto& node_face_id = faces_per_node[node_id];
//			auto& st_node = node_stencil[node_id];
//
//			// fill equation for first node that belongs to the interface
//			if (face.type == MAT)
//			{
//				// Clean matrices
//				auto& cur = inner[cell_id][loop_face_id];
//				std::fill_n(&cur.A1.values[0], cur.A1.values.size(), 0.0);
//				std::fill_n(&cur.A2.values[0], cur.A2.values.size(), 0.0);
//				std::fill_n(&cur.Q1.values[0], cur.Q1.values.size(), 0.0);
//				std::fill_n(&cur.Q2.values[0], cur.Q2.values.size(), 0.0);
//				std::fill_n(&cur.Th1.values[0], cur.Th1.values.size(), 0.0);
//				std::fill_n(&cur.Th2.values[0], cur.Th2.values.size(), 0.0);
//				std::fill_n(&cur.R1.values[0], cur.R1.values.size(), 0.0);
//				std::fill_n(&cur.R2.values[0], cur.R2.values.size(), 0.0);
//
//				const int& cell_id1 = face.cell_id1;
//				const int& cell_id2 = face.cell_id2;
//				const auto& c1 = cell_centers[cell_id1];
//				const auto& c2 = cell_centers[cell_id2];
//				n = (face.n.transpose() * (c2 - c1)).values[0] > 0 ? face.n : -face.n;
//				P = I3 - outer_product(n, n.transpose());
//				// Permeability decomposition
//				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
//				K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
//				lam1 = (n.transpose() * K1n)(0, 0);
//				lam2 = (n.transpose() * K2n)(0, 0);
//				gam1 = K1n - lam1 * n;
//				gam2 = K2n - lam2 * n;
//				// Stiffness decomposition
//				C1 = W * stfs[cell_id1] * W.transpose();
//				C2 = W * stfs[cell_id2] * W.transpose();
//				nblock = make_block_diagonal(n, ND);
//				nblock_t = make_block_diagonal(n.transpose(), ND);
//				tblock = make_block_diagonal(P, ND);
//				T1 = nblock_t * C1 * nblock;
//				T2 = nblock_t * C2 * nblock;
//				G1 = nblock_t * C1 * tblock;
//				G2 = nblock_t * C2 * tblock;
//				// Process geometry
//				auto& r1 = cur.r1;
//				auto& r2 = cur.r2;
//				r1 = (n.transpose() * (face.c - c1))(0, 0);
//				r2 = (n.transpose() * (c2 - face.c))(0, 0);
//				assert(r1 > 0.0);		assert(r2 > 0.0);
//				auto& y1 = cur.y1;
//				auto& y2 = cur.y2;
//				y1 = c1 + r1 * n;	 y2 = c2 - r2 * n;
//				// Assemble matrices
//				auto& A1 = cur.A1;						auto& A2 = cur.A2;
//				auto& Q1 = cur.Q1;						auto& Q2 = cur.Q2;
//				auto& R1 = cur.R1;						auto& R2 = cur.R2;
//				B1n = biots[cell_id1] * n;
//				B2n = biots[cell_id2] * n;
//				if (!isStationary)
//				{
//					A1(3, { 3, 1 }, { 4, 1 }) = B1n.values;				A2(3, { 3, 1 }, { 4, 1 }) = B2n.values;
//					A1(4 * 3, { 3 }, { 1 }) = B1n.transpose().values;	A2(4 * 3, { 3 }, { 1 }) = B2n.transpose().values;
//					R1(3, 0) = -(B1n.transpose() * get_u_face_prev(face.c - c1, cell_id1)).values[0];
//					R2(3, 0) = -(B2n.transpose() * get_u_face_prev(face.c - c2, cell_id2)).values[0];
//				}
//				Q1(0, { 3, 3 }, { 4, 1 }) = -T1.values;
//				Q1(3, 3) = -lam1 * dt / visc;
//				Q1 += r1 * A1;
//				Q2(0, { 3, 3 }, { 4, 1 }) = -T2.values;
//				Q2(3, 3) = -lam2 * dt / visc;
//				Q2 -= r2 * A2;
//				auto& Th1 = cur.Th1;
//				auto& Th2 = cur.Th2;
//				Th1(0, { 3, 9 }, { 12, 1 }) = -G1.values;
//				Th2(0, { 3, 9 }, { 12, 1 }) = -G2.values;
//				Th1(45, { 3 }, { 1 }) = -dt / visc * gam1.transpose().values;
//				Th2(45, { 3 }, { 1 }) = -dt / visc * gam2.transpose().values;
//				Th1 += A1 * make_block_diagonal((face.c - y1).transpose(), 4);
//				Th2 += A2 * make_block_diagonal((face.c - y2).transpose(), 4);
//				R1(3, 0) += (dt / visc * grav_vec * K1n).values[0];
//				R2(3, 0) += (dt / visc * grav_vec * K2n).values[0];
//
//				// Matrix
//				A(4 * node_face_id * 4 * ND, { 4, 4 * ND }, { 4 * ND, 1 }) = ((Q2 * make_block_diagonal((y2 - y1).transpose(), 4) + r2 * (Th1 - Th2)) * make_block_diagonal(I3 - outer_product(n, n.transpose()), 4) +
//					(r2 * Q1 + r1 * Q2) * make_block_diagonal(n.transpose(), 4)).values;
//				/*(Q2 * make_block_diagonal((c2 - c1).transpose(), 4) +
//				r2 * (Q1 - Q2) * make_block_diagonal(n.transpose(), 4) + r2 * (Th1 - Th2)).values;*/
//				// RHS
//				res1 = findInVector(st_node, cell_id1);
//				if (res1.first) { id = res1.second; }
//				else { id = st_node.size(); st_node.push_back(cell_id1); }
//				id1 = id;
//				rhs_mult(4 * node_face_id * rhs_mult.N + 4 * id, { 4, 4 }, { (size_t)rhs_mult.N, 1 }) += -(Q2 + r2 * A1).values;
//
//				res2 = findInVector(st_node, cell_id2);
//				if (res2.first) { id = res2.second; }
//				else { id = st_node.size(); st_node.push_back(cell_id2); }
//				id2 = id;
//				rhs_mult(4 * node_face_id * rhs_mult.N + 4 * id, { 4, 4 }, { (size_t)rhs_mult.N, 1 }) += (Q2 + r2 * A2).values;
//
//				rest(4 * node_face_id, { 4 }, { 1 }) = r2 * (R2 - R1).values;
//				node_face_id++;
//
//				// stabilization parameters
//				cur.k_stab1 = lam1 / r1;
//				cur.k_stab2 = lam2 / r2;
//				cur.beta_stab1 = sqrt((B1n.transpose() * B1n).values[0]);
//				cur.beta_stab2 = sqrt((B2n.transpose() * B2n).values[0]);
//				B1nn.values = (make_block_diagonal(n, ND) * B1n).values;
//				B2nn.values = (make_block_diagonal(n, ND) * B2n).values;
//				cur.c_stab1 = (B1nn.transpose() * C1 * B1nn).values[0] / r1 / cur.beta_stab1 / cur.beta_stab1;
//				cur.c_stab2 = (B2nn.transpose() * C2 * B2nn).values[0] / r2 / cur.beta_stab2 / cur.beta_stab2;
//				//cur.alpha_min_stab1 = (sqrt((k_stab1 - c_stab1) * (k_stab1 - c_stab1) + 4 * cur.beta_stab1 * cur.beta_stab1 / dt) - (k_stab1 + c_stab1)) / (2 * cur.beta_stab1);
//				//cur.alpha_min_stab2 = (sqrt((k_stab2 - c_stab2) * (k_stab2 - c_stab2) + 4 * cur.beta_stab2 * cur.beta_stab2 / dt) - (k_stab2 + c_stab2)) / (2 * cur.beta_stab2);
//
//				cur.S1(0, { ND, ND }, { 4, 1 }) = outer_product(B1n, B1n.transpose()).values / cur.beta_stab1;
//				cur.S1(ND, ND) = cur.beta_stab1;
//				// cur.S1.values *= std::max(cur.alpha_min_stab1, 1.0);
//
//				cur.S2(0, { ND, ND }, { 4, 1 }) = outer_product(B2n, B2n.transpose()).values / cur.beta_stab2;
//				cur.S2(ND, ND) = cur.beta_stab2;
//				// cur.S2.values *= std::max(cur.alpha_min_stab2, 1.0);
//			}
//			else if (face.type == BORDER)
//			{
//				const auto& b = bc[face.face_id2];
//				const auto& an = b(0, 0);			const auto& bn = b(1, 0);
//				const auto& at = b(2, 0);			const auto& bt = b(3, 0);
//				const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
//				// Skip if pure neumann
//				// if (!NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION && an == 0.0 && at == 0.0)	continue;
//
//				const int& cell_id1 = face.cell_id1;
//				// Geometry
//				const auto& c1 = cell_centers[cell_id1];
//				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
//				P = I3 - outer_product(n, n.transpose());
//				r1 = (n.transpose() * (face.c - c1))(0, 0);		assert(r1 > 0.0);
//				y1 = c1 + r1 * n;
//				B1n = biots[cell_id1] * n;
//				// Stiffness decomposition
//				C1 = W * stfs[cell_id1] * W.transpose();
//				nblock = make_block_diagonal(n, ND);
//				nblock_t = make_block_diagonal(n.transpose(), ND);
//				tblock = make_block_diagonal(P, ND);
//				T1 = nblock_t * C1 * nblock;
//				G1 = nblock_t * C1 * tblock;
//				// Permeability decomposition
//				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
//				lam1 = (n.transpose() * K1n)(0, 0);
//				gam1 = K1n - lam1 * n;
//				// Extra 'boundary' stuff
//				An = (an * I3 + bn / r1 * T1);
//				At = (at * I3 + bt / r1 * T1);
//				Ap = 1.0 / (ap + bp / r1 / visc * lam1);
//				res = At.inv();
//				if (!res)
//				{
//					cout << "Inversion failed!\n";	exit(-1);
//				}
//				L = An * At;
//				gamma = 1.0 / (n.transpose() * L * n).values[0];
//				gamma_nnt = gamma * outer_product(n, n.transpose());
//				gamma_nnt_mult = gamma_nnt * (bn * I3 - bt * L);
//				mult_p = (bt * I3 + gamma_nnt_mult) * B1n;
//				// Filling mechanics equations
//				A(4 * node_face_id * 4 * ND,
//					{ ND, 3 * ND },
//					{ 4 * ND, 1 }) = (at * make_block_diagonal((face.c - c1).transpose(), ND) +
//						bt * nblock_t * C1 +
//						gamma_nnt_mult * (G1 + T1 / r1 *
//							make_block_diagonal((y1 - face.c).transpose(), ND))).values;
//				A(4 * node_face_id * 4 * ND + 3 * ND,
//					{ ND, ND },
//					{ 4 * ND, 1 }) = outer_product(mult_p, Ap * bp / visc * ((lam1 / r1 * (y1 - face.c) + gam1).transpose() *
//						P)).values;
//
//				res1 = findInVector(st_node, face.cell_id1);
//				if (res1.first) { id = res1.second; }
//				else { id = st_node.size(); st_node.push_back(face.cell_id1); }
//				id1 = id;
//
//				rhs_mult(4 * node_face_id * rhs_mult.N + 4 * id, { 3, 3 }, { (size_t)rhs_mult.N, 1 }) += (gamma_nnt_mult * T1 / r1 - at * I3).values;
//				rhs_mult(4 * node_face_id * rhs_mult.N + 4 * id + 3, { 3, 1 }, { (size_t)rhs_mult.N, 1 }) += (Ap * bp / visc * lam1 / r1 * mult_p).values;
//				rest(4 * node_face_id, { 3 }, { 1 }) = (mult_p * Ap * bp / visc * (grav_vec * K1n).values[0]).values;
//				// Filling flow equation
//				A((4 * node_face_id + 3) * 4 * ND + 3 * ND,
//					{ ND },
//					{ 1 }) = (ap * (face.c - c1) + bp / visc * K1n).values;
//				rhs_mult(4 * node_face_id + 3, 4 * id + 3) -= ap;
//				rest(4 * node_face_id + 3, 0) = bp / visc * (grav_vec * K1n).values[0];
//				// BC RHS coefficients
//				bface_id = n_cells + face.face_id2;
//				res2 = findInVector(st_node, bface_id);
//				if (res2.first) { id = res2.second; }
//				else { id = st_node.size(); st_node.push_back(bface_id); }
//				id2 = id;
//				// Mech
//				rhs_mult(4 * node_face_id * rhs_mult.N + 4 * id, { 3, 3 }, { (size_t)rhs_mult.N, 1 }) = (gamma_nnt + (I3 - gamma_nnt * L) * P).values;
//				rhs_mult(4 * node_face_id * rhs_mult.N + 4 * id + 3, { 3, 1 }, { (size_t)rhs_mult.N, 1 }) = (mult_p * Ap).values;
//				// Flow
//				rhs_mult(4 * node_face_id + 3, 4 * id + 3) = 1.0;
//
//				node_face_id++;
//			}
//			/*else if (face.type == MAT_TO_FRAC)
//			{
//				const Face& face0 = vec_faces[face.face_id1];
//				const auto& cur = inner[cell_id][face.face_id1];
//				// recount original face id
//				face_id1 = 0;
//				for (index_t h = 0; h < face.face_id1; h++)
//				{
//					const Face& face = vec_faces[h];
//					if (face.type == BORDER)
//					{
//						const auto& b = bc[face.face_id2];
//						const auto& an = b(0, 0);	const auto& at = b(2, 0);
//						if (an != 0.0 || at != 0.0)	face_id1++;
//					}
//					else if (face.type != MAT_TO_FRAC) face_id1++;
//				}
//
//				const int& cell_id1 = face0.cell_id1;
//				const int& frac_id = face.cell_id2;
//				assert(frac_id >= n_matrix);
//				const auto& c1 = cell_centers[cell_id1];
//				const auto& c2 = cell_centers[frac_id];
//				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
//				sign = get_fault_sign(n, ref_contact_ids[frac_id - n_matrix]);
//				// add gap
//				res1 = findInVector(st, frac_id);
//				if (res1.first) { id1 = res1.second; }
//				else { id1 = st.size(); st.push_back(frac_id); }
//				rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id1,
//					{ BLOCK_SIZE, BLOCK_SIZE },
//					{ (size_t)rhs_mult.N, 1 }) += -sign * (cur.Q2 + cur.r2 * cur.A1).values;
//				// add gap gradient
//				const auto& frac_grad = grad[frac_id];
//				auto& frac_grad_mult = pre_frac_grad_mult[frac_grad.stencil.size()];
//				std::fill_n(&frac_grad_mult.values[0], frac_grad_mult.values.size(), 0.0);
//				frac_grad_mult = sign * (cur.Q2 * make_block_diagonal((cur.y2 - face.c).transpose(), BLOCK_SIZE) - cur.r2 * cur.Th2) * frac_grad.mat;
//				for (st_id = 0; st_id < frac_grad.stencil.size(); st_id++)
//				{
//					assert(frac_grad.stencil[st_id] >= n_matrix);
//					res1 = findInVector(st, frac_grad.stencil[st_id]);
//					if (res1.first) { id = res1.second; }
//					else { id = st.size(); st.push_back(frac_grad.stencil[st_id]); }
//					rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id,
//						{ BLOCK_SIZE, BLOCK_SIZE },
//						{ (size_t)rhs_mult.N, 1 }) -= frac_grad_mult(st_id * BLOCK_SIZE,
//							{ BLOCK_SIZE, BLOCK_SIZE },
//							{ (size_t)frac_grad_mult.N, 1 });
//				}
//				// no discontinuity in pressure
//				rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = 0.0;
//
//				// pressure condition
//				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
//				K2n = pm_discretizer::darcy_constant * perms[frac_id] * n;
//				r2_frac = frac_apers[frac_id - n_matrix] / 2;
//				lam2 = (n.transpose() * K2n)(0, 0);
//
//				// remove previous numbers
//				A((BLOCK_SIZE * face_id1 + ND) * A.N, { BLOCK_SIZE * ND }, { 1 }) = 0.0;
//				rhs_mult((BLOCK_SIZE * face_id1 + ND) * rhs_mult.N, { (size_t)rhs_mult.N }, { 1 }) = 0.0;
//				rest(BLOCK_SIZE * face_id1 + ND, 0) = 0.0;
//				// fill newer
//				A((BLOCK_SIZE * face_id1 + ND) * A.N + ND * ND, { ND }, { 1 }) = (c2 - c1 + r2_frac / lam2 * (K1n - K2n)).values;
//				res1 = findInVector(st, cell_id1);
//				if (res1.first) { id = res1.second; }
//				else { id = st.size(); st.push_back(cell_id1); }
//				rhs_mult(BLOCK_SIZE * face_id1 + ND, BLOCK_SIZE * id + ND) = -1.0;
//
//				res2 = findInVector(st, frac_id);
//				if (res2.first) { id = res2.second; }
//				else { id = st.size(); st.push_back(frac_id); }
//				rhs_mult(BLOCK_SIZE * face_id1 + ND, BLOCK_SIZE * id + ND) = 1.0;
//
//				rest(BLOCK_SIZE * face_id1 + ND, 0) = r2_frac / lam2 * ((grav_vec * K1n).values[0] - (grav_vec * K2n).values[0]);
//				//face_id++;
//			}*/
//
//			// copy to other nodes that belong to the interface
//			for (index_t k = 1; k < face.pts.size(); k++)
//			{
//				node_id = face.pts[k];
//				it = cell_pts.find(node_id);	if (it == cell_pts.end()) { cell_pts.insert(node_id); }
//				auto& other_A = node_A[node_id];
//				auto& other_rest = node_rest[node_id];
//				auto& other_rhs_mult = node_rhs_mult[node_id];
//				auto& other_node_face_id = faces_per_node[node_id];
//				auto& st_node = node_stencil[node_id];
//
//				//// stencil & rhs_mult
//				// 1
//				res1 = findInVector(st_node, face.cell_id1);
//				if (res1.first) { id = res1.second; }
//				else { id = st_node.size(); st_node.push_back(face.cell_id1); }
//				other_rhs_mult(BLOCK_SIZE * other_node_face_id * rhs_mult.N + BLOCK_SIZE * id, { BLOCK_SIZE , BLOCK_SIZE }, { (size_t)other_rhs_mult.N, 1 }) =
//					rhs_mult(BLOCK_SIZE * (node_face_id-1) * rhs_mult.N + BLOCK_SIZE * id1, { BLOCK_SIZE , BLOCK_SIZE }, { (size_t)rhs_mult.N, 1 });
//				// 2
//				if (face.type == MAT)
//				{
//					res2 = findInVector(st_node, face.cell_id2);
//					if (res2.first) { id = res2.second; }
//					else { id = st_node.size(); st_node.push_back(face.cell_id2); }
//				}
//				else
//				{
//					bface_id = n_cells + face.face_id2;
//					res2 = findInVector(st_node, bface_id);
//					if (res2.first) { id = res2.second; }
//					else { id = st_node.size(); st_node.push_back(bface_id); }
//				}
//				other_rhs_mult(BLOCK_SIZE * other_node_face_id * rhs_mult.N + BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)other_rhs_mult.N, 1 }) =
//					rhs_mult(BLOCK_SIZE * (node_face_id - 1) * rhs_mult.N + BLOCK_SIZE * id2, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)rhs_mult.N, 1 });
//				//// main matrix
//				other_A(BLOCK_SIZE * other_node_face_id * BLOCK_SIZE * ND, { BLOCK_SIZE, BLOCK_SIZE * ND }, { (size_t)other_A.N, 1 }) =
//					A(BLOCK_SIZE * (node_face_id - 1) * BLOCK_SIZE * ND, { BLOCK_SIZE, BLOCK_SIZE * ND }, { (size_t)A.N, 1 });
//				//// free term
//				other_rest(BLOCK_SIZE * other_node_face_id, { BLOCK_SIZE }, { 1 }) = rest(BLOCK_SIZE * (node_face_id - 1), { BLOCK_SIZE }, { 1 });
//
//				other_node_face_id++;
//			}
//		}
//	}
//
//	// Calculate node-based gradients
//	vector<Gradients> node_grad(n_nodes);
//	for (node_id = 0; node_id < n_nodes; node_id++)
//	{
//		const auto& st_node = node_stencil[node_id];
//		const auto& eq_num = faces_per_node[node_id];
//		auto& A = node_A[node_id];
//		auto& rhs_mult = node_rhs_mult[node_id];
//		const auto& rest = node_rest[node_id];
//		auto& cur_rhs = pre_cur_rhs[eq_num][st_node.size()];
//		cur_rhs.values = rhs_mult(0, { (size_t)BLOCK_SIZE * eq_num, BLOCK_SIZE * st_node.size() }, { (size_t)rhs_mult.N, 1 });
//		auto& cur_grad = node_grad[node_id];
//
//		cur_grad.stencil = st_node;
//
//		sq_mat = A.transpose() * A;
//		res = sq_mat.inv();
//		if (!res)
//		{
//			cout << "Inversion failed!\n";
//			//sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");
//			exit(-1);
//		}
//		if (sq_mat.is_nan())
//		{
//			face_count_id = (face_id > ND) ? ND : face_id;
//			auto& Wsvd = pre_Wsvd[face_count_id];
//			auto& Zsvd = pre_Zsvd[face_count_id];
//			auto& w_svd = pre_w_svd[face_count_id].values;
//			std::fill_n(&Wsvd.values[0], Wsvd.values.size(), 0.0);
//			std::fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
//			std::fill_n(&w_svd[0], w_svd.size(), 0.0);
//
//			// SVD decomposition A = M W Z*
//			if (Zsvd.M != A.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
//			res = A.svd(Zsvd, w_svd);
//			assert(Zsvd.M == face_count_id * BLOCK_SIZE && Zsvd.N == face_count_id * BLOCK_SIZE);
//			if (!res) { cout << "SVD failed!\n"; /*sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");*/ exit(-1); }
//			// check SVD
//			// Wsvd.set_diagonal(w_svd);
//			// assert(A == M * Wsvd * Zsvd.transpose());
//			for (index_t i = 0; i < w_svd.size(); i++)
//				w_svd[i] = (fabs(w_svd[i]) < 1000 * EQUALITY_TOLERANCE) ? 0.0 /*w_svd[i]*/ : 1.0 / w_svd[i];
//			Wsvd.set_diagonal(w_svd);
//			// check pseudo-inverse
//			// Matrix Ainv = Zsvd * Wsvd * M.transpose();
//			// assert(A * Ainv * A == A && Ainv * A * Ainv == Ainv);
//			cur_grad.mat = Zsvd * Wsvd * A.transpose() * cur_rhs;
//			cur_grad.rhs = Zsvd * Wsvd * A.transpose() * rest;
//		}
//		else
//		{
//			cur_grad.mat = sq_mat * A.transpose() * cur_rhs;
//			cur_grad.rhs = sq_mat * A.transpose() * rest;
//		}
//	
//	}
//
//	// Calculate cell-based gradients
//	value_t n_nodes_per_cell;
//	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat = Matrix(BLOCK_SIZE * ND, MAX_STENCIL * BLOCK_SIZE);	
//
//	for (cell_id = 0; cell_id < n_matrix; cell_id++)
//	{
//		const auto& cell_pts = nodes_per_cell[cell_id];
//		// number of nodes for current cell
//		n_nodes_per_cell = cell_pts.size();
//		// cell gradient
//		auto& cur_cell_grad = grad[cell_id];
//		// clean up gradients
//		g.stencil.clear();					g.mat.values = 0.0;					g.rhs.values = 0.0;
//		cur_cell_grad.stencil.clear();		cur_cell_grad.mat.values = 0.0;		cur_cell_grad.rhs = Matrix(BLOCK_SIZE * ND, 1);
//		for (const auto& node_id : cell_pts)
//		{
//			const auto& cur_node_grad = node_grad[node_id];
//			g = merge_stencils(g.stencil, g.mat, cur_node_grad.stencil, cur_node_grad.mat / n_nodes_per_cell);
//			cur_cell_grad.rhs += cur_node_grad.rhs / n_nodes_per_cell;
//		}
//
//		cur_cell_grad.stencil = g.stencil;
//		cur_cell_grad.mat = Matrix(BLOCK_SIZE * ND, BLOCK_SIZE * g.stencil.size());
//		cur_cell_grad.mat.values = g.mat(0, { BLOCK_SIZE * ND, BLOCK_SIZE * g.stencil.size() }, { (size_t)g.mat.N, 1 });
//	}
//
//	printf("Gradient reconstruction was done!\n");
//}
void pm_discretizer::reconstruct_gradients_thermal_per_cell(value_t dt)
{
	bool isStationary = dt == 0.0 ? true : false;
	if (isStationary) dt = 1.0;

	assert(diffs.size());

	n_cells = perms.size();
	grad_prev = grad;

	// Variables
	size_t n_cur_faces;
	std::vector<index_t> st;
	st.reserve(MAX_STENCIL);
	Matrix n(ND, 1), K1n(ND, 1), K2n(ND, 1), D1n(ND, 1), D2n(ND, 1), B1n(ND, 1), B2n(ND, 1),
		gam1(ND, 1), gam2(ND, 1);
	value_t lam1, lam2, dlam1, dlam2;
	Matrix C1(9, 9), C2(9, 9), T1(3, 3), T2(3, 3), G1(3, 9), G2(3, 9);
	Matrix nblock(9, 3), nblock_t(3, 9), tblock(9, 9);
	bool res;
	value_t r1, Ap, gamma, sign;
	Matrix y1(ND, 1), An(ND, ND), At(ND, ND), L(ND, ND), P(ND, ND), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), mult_p(ND, 1);
	Matrix sq_mat(BLOCK_SIZE * ND, BLOCK_SIZE * ND), sq_matd(ND, ND), Wsvd(BLOCK_SIZE * ND, BLOCK_SIZE * ND), frac_grad_mult(BLOCK_SIZE, BLOCK_SIZE * ND);

	int face_id, face_id1, face_count_id, cell_id, bface_id, st_id, loop_face_id;
	value_t r2_frac;

	// Gradient reconstruction in fracture cells
	for (cell_id = 0; cell_id < n_fracs; cell_id++)
	{
		st.clear();
		const auto& vec_faces = faces[n_matrix + cell_id];
		n_cur_faces = vec_faces.size();
		auto& A = pre_A[n_cur_faces - 2];
		auto& rhs_mult = pre_rhs_mult[n_cur_faces - 2];
		// Cleaning
		std::fill_n(&A.values[0], A.values.size(), 0.0);
		std::fill_n(&rhs_mult.values[0], rhs_mult.values.size(), 0.0);
		auto& rest = pre_rest[ND];
		std::fill_n(&rest.values[0], rest.values.size(), 0.0);

		face_count_id = 0;
		for (face_id = 0; face_id < n_cur_faces; face_id++)
		{
			const Face& face = vec_faces[face_id];
			if (face.type == FRAC)
			{
				const int& cell_id1 = face.cell_id1;
				const int& cell_id2 = face.cell_id2;
				const auto& c1 = cell_centers[cell_id1];
				const auto& c2 = cell_centers[cell_id2];
				A(face_count_id * BLOCK_SIZE * A.N,
					{ BLOCK_SIZE, BLOCK_SIZE * ND },
					{ (size_t)A.N, 1 }) = make_block_diagonal((c2 - c1).transpose(), BLOCK_SIZE).values;

				res1 = findInVector(st, face.cell_id1);
				if (res1.first) { id = res1.second; }
				else { id = st.size(); st.push_back(face.cell_id1); }
				rhs_mult(BLOCK_SIZE * (face_count_id * rhs_mult.N + id),
					{ BLOCK_SIZE, BLOCK_SIZE },
					{ (size_t)rhs_mult.N, 1 }) += -I4.values;

				res2 = findInVector(st, face.cell_id2);
				if (res2.first) { id = res2.second; }
				else { id = st.size(); st.push_back(face.cell_id2); }
				rhs_mult(BLOCK_SIZE * (face_count_id * rhs_mult.N + id),
					{ BLOCK_SIZE, BLOCK_SIZE },
					{ (size_t)rhs_mult.N, 1 }) += I4.values;

				face_count_id++;
			}
		}

		auto& cur_grad = grad[n_matrix + cell_id];
		cur_grad.stencil = st;
		cur_grad.rhs = rest;
		auto& cur_rhs = pre_cur_rhs[n_cur_faces - 2][st.size()];
		cur_rhs.values = rhs_mult(0,
			{ BLOCK_SIZE * (n_cur_faces - 2), BLOCK_SIZE * st.size() },
			{ (size_t)rhs_mult.N, 1 });

		// SVD is produced for M x N matrix where M >= N, decompose transposed otherwise
		face_count_id = (face_count_id > ND) ? ND : face_count_id;
		auto& Wsvd = pre_Wsvd[face_count_id];
		auto& Zsvd = pre_Zsvd[face_count_id];
		auto& w_svd = pre_w_svd[face_count_id].values;
		std::fill_n(&Wsvd.values[0], Wsvd.values.size(), 0.0);
		std::fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
		std::fill_n(&w_svd[0], w_svd.size(), 0.0);
		if (A.M >= A.N)
		{
			// SVD decomposition A = M W Z*
			res = A.svd(Zsvd, w_svd);
			if (!res) { cout << "SVD failed!\n"; /*sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");*/ exit(-1); }
			// check SVD
			//Wsvd.set_diagonal(w_svd);
			//assert(A == M * Wsvd * Zsvd.transpose());
			for (index_t i = 0; i < w_svd.size(); i++)
				w_svd[i] = (fabs(w_svd[i]) < 1000 * EQUALITY_TOLERANCE) ? 0.0 /*w_svd[i]*/ : 1.0 / w_svd[i];
			Wsvd.set_diagonal(w_svd);
			// check pseudo-inverse
			//Matrix Ainv = Zsvd * Wsvd * M.transpose();
			//assert(A * Ainv * A == A && Ainv * A * Ainv == Ainv);
			cur_grad.mat = Zsvd * Wsvd * A.transpose() * cur_rhs;
		}
		else
		{
			// SVD decomposition A* = M W Z*
			A.transposeInplace();
			res = A.svd(Zsvd, w_svd);
			if (!res) { cout << "SVD failed!\n"; /*sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");*/ exit(-1); }
			// check SVD
			//Wsvd.set_diagonal(w_svd);
			//assert(A.transpose() == M * Wsvd * Zsvd.transpose());
			for (index_t i = 0; i < w_svd.size(); i++)
				w_svd[i] = (fabs(w_svd[i]) < 1000 * EQUALITY_TOLERANCE) ? 0.0 /*w_svd[i]*/ : 1.0 / w_svd[i];
			Wsvd.set_diagonal(w_svd);
			// check pseudo-inverse
			//Matrix Ainv = M * Wsvd * Zsvd.transpose();
			//assert(A * Ainv * A == A && Ainv * A * Ainv == Ainv);
			cur_grad.mat = A * Wsvd * Zsvd.transpose() * cur_rhs;
			A.transposeInplace();
		}
	}

	// Gradient reconstruction in matrix cells
	for (cell_id = 0; cell_id < n_matrix; cell_id++)
	{
		st.clear();
		const auto& vec_faces = faces[cell_id];

		// Build the system from the continuity at the interfaces
		n_cur_faces = 0;// vec_faces.size();
		for (face_id = 0; face_id < vec_faces.size(); face_id++)
		{
			const Face& face = vec_faces[face_id];
			if (face.type == BORDER)
			{
				const auto& b = bc[face.face_id2];
				const auto& an = b(0, 0);			const auto& bn = b(1, 0);
				const auto& at = b(2, 0);			const auto& bt = b(3, 0);
				if (NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION || an != 0.0 || at != 0.0)	n_cur_faces++;
			}
			else if (face.type != MAT_TO_FRAC) n_cur_faces++;
		}

		auto& A = pre_A[n_cur_faces];
		auto& rest = pre_rest[n_cur_faces];
		auto& rhs_mult = pre_rhs_mult[n_cur_faces];
		// Heat conduction
		auto& Ad = pre_Ad[n_cur_faces];
		auto& restd = pre_restd[n_cur_faces];
		auto& rhs_multd = pre_rhs_multd[n_cur_faces];

		// Cleaning
		std::fill_n(&A.values[0], A.values.size(), 0.0);
		std::fill_n(&rest.values[0], rest.values.size(), 0.0);
		std::fill_n(&rhs_mult.values[0], rhs_mult.values.size(), 0.0);

		std::fill_n(&Ad.values[0], Ad.values.size(), 0.0);
		std::fill_n(&restd.values[0], restd.values.size(), 0.0);
		std::fill_n(&rhs_multd.values[0], rhs_multd.values.size(), 0.0);

		face_id = 0;
		for (loop_face_id = 0; loop_face_id < vec_faces.size(); loop_face_id++)
		{
			const Face& face = vec_faces[loop_face_id];
			if (face.type == MAT)
			{
				// Clean matrices
				auto& cur = inner[cell_id][loop_face_id];
				std::fill_n(&cur.A1.values[0], cur.A1.values.size(), 0.0);
				std::fill_n(&cur.A2.values[0], cur.A2.values.size(), 0.0);
				std::fill_n(&cur.Q1.values[0], cur.Q1.values.size(), 0.0);
				std::fill_n(&cur.Q2.values[0], cur.Q2.values.size(), 0.0);
				std::fill_n(&cur.Th1.values[0], cur.Th1.values.size(), 0.0);
				std::fill_n(&cur.Th2.values[0], cur.Th2.values.size(), 0.0);
				std::fill_n(&cur.R1.values[0], cur.R1.values.size(), 0.0);
				std::fill_n(&cur.R2.values[0], cur.R2.values.size(), 0.0);

				const int& cell_id1 = face.cell_id1;
				const int& cell_id2 = face.cell_id2;
				const auto& c1 = cell_centers[cell_id1];
				const auto& c2 = cell_centers[cell_id2];
				n = (face.n.transpose() * (c2 - c1)).values[0] > 0 ? face.n : -face.n;
				P = I3 - outer_product(n, n.transpose());
				// Permeability decomposition
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
				D1n = pm_discretizer::heat_cond_constant * diffs[cell_id1] * n;
				D2n = pm_discretizer::heat_cond_constant * diffs[cell_id2] * n;
				lam1 = (n.transpose() * K1n)(0, 0);		dlam1 = (n.transpose() * D1n)(0, 0);
				lam2 = (n.transpose() * K2n)(0, 0);		dlam2 = (n.transpose() * D2n)(0, 0);
				gam1 = K1n - lam1 * n;
				gam2 = K2n - lam2 * n;
				// Stiffness decomposition
				C1 = W * stfs[cell_id1] * W.transpose();
				C2 = W * stfs[cell_id2] * W.transpose();
				nblock = make_block_diagonal(n, ND);
				nblock_t = make_block_diagonal(n.transpose(), ND);
				tblock = make_block_diagonal(P, ND);
				T1 = nblock_t * C1 * nblock;
				T2 = nblock_t * C2 * nblock;
				G1 = nblock_t * C1 * tblock;
				G2 = nblock_t * C2 * tblock;
				// Process geometry
				auto& r1 = cur.r1;
				auto& r2 = cur.r2;
				r1 = (n.transpose() * (face.c - c1))(0, 0);
				r2 = (n.transpose() * (c2 - face.c))(0, 0);
				assert(r1 > 0.0);		assert(r2 > 0.0);
				auto& y1 = cur.y1;
				auto& y2 = cur.y2;
				y1 = c1 + r1 * n;	 y2 = c2 - r2 * n;
				// Assemble matrices
				auto& A1 = cur.A1;						auto& A2 = cur.A2;
				auto& Q1 = cur.Q1;						auto& Q2 = cur.Q2;
				auto& R1 = cur.R1;						auto& R2 = cur.R2;
				B1n = biots[cell_id1] * n;
				B2n = biots[cell_id2] * n;

				A1(3, { 3, 1 }, { 4, 1 }) = B1n.values;				A2(3, { 3, 1 }, { 4, 1 }) = B2n.values;
				if (!isStationary)
				{
					A1(4 * 3, { 3 }, { 1 }) = B1n.transpose().values;	A2(4 * 3, { 3 }, { 1 }) = B2n.transpose().values;
					R1(3, 0) = -(B1n.transpose() * get_u_face_prev(face.c - c1, cell_id1)).values[0];
					R2(3, 0) = -(B2n.transpose() * get_u_face_prev(face.c - c2, cell_id2)).values[0];
				}
				Q1(0, { 3, 3 }, { 4, 1 }) = -T1.values;
				Q1(3, 3) = -lam1 * dt / visc;
				Q1 += r1 * A1;
				Q2(0, { 3, 3 }, { 4, 1 }) = -T2.values;
				Q2(3, 3) = -lam2 * dt / visc;
				Q2 -= r2 * A2;
				auto& Th1 = cur.Th1;
				auto& Th2 = cur.Th2;
				Th1(0, { 3, 9 }, { 12, 1 }) = -G1.values;
				Th2(0, { 3, 9 }, { 12, 1 }) = -G2.values;
				Th1(45, { 3 }, { 1 }) = -dt / visc * gam1.transpose().values;
				Th2(45, { 3 }, { 1 }) = -dt / visc * gam2.transpose().values;
				Th1 += A1 * make_block_diagonal((face.c - y1).transpose(), 4);
				Th2 += A2 * make_block_diagonal((face.c - y2).transpose(), 4);
				R1(3, 0) += (dt / visc * grav_vec * K1n).values[0];
				R2(3, 0) += (dt / visc * grav_vec * K2n).values[0];

				// Matrix
				A(BLOCK_SIZE * face_id * BLOCK_SIZE * ND, { BLOCK_SIZE, BLOCK_SIZE * ND }, { BLOCK_SIZE * ND, 1 }) = ((Q2 * make_block_diagonal((y2 - y1).transpose(), 4) + r2 * (Th1 - Th2)) * make_block_diagonal(I3 - outer_product(n, n.transpose()), 4) +
					(r2 * Q1 + r1 * Q2) * make_block_diagonal(n.transpose(), BLOCK_SIZE)).values;
				/*(Q2 * make_block_diagonal((c2 - c1).transpose(), 4) +
				r2 * (Q1 - Q2) * make_block_diagonal(n.transpose(), 4) + r2 * (Th1 - Th2)).values;*/
				Ad(face_id * ND, { ND }, { 1 }) = (dlam2 * (c2 - c1) + r2 * (D1n - D2n)).values;

				// RHS
				res1 = findInVector(st, cell_id1);
				if (res1.first) { id = res1.second; }
				else { id = st.size(); st.push_back(cell_id1); }
				rhs_mult(BLOCK_SIZE * face_id * rhs_mult.N + BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)rhs_mult.N, 1 }) += -(Q2 + r2 * A1).values;
				rhs_multd(face_id, id) += -dlam2;

				res2 = findInVector(st, cell_id2);
				if (res2.first) { id = res2.second; }
				else { id = st.size(); st.push_back(cell_id2); }
				rhs_mult(BLOCK_SIZE * face_id * rhs_mult.N + BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)rhs_mult.N, 1 }) += (Q2 + r2 * A2).values;
				rhs_multd(face_id, id) += dlam2;

				rest(BLOCK_SIZE * face_id, { BLOCK_SIZE }, { 1 }) = r2 * (R2 - R1).values;
				face_id++;
			}
			else if (face.type == BORDER)
			{
				const auto& b = bc[face.face_id2];
				const auto& an = b(0, 0);			const auto& bn = b(1, 0);
				const auto& at = b(2, 0);			const auto& bt = b(3, 0);
				const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
				const auto& ath = b(6, 0);			const auto& bth = b(7, 0);
				// Skip if pure neumann
				if (an == 0.0 && at == 0.0)	continue;

				const int& cell_id1 = face.cell_id1;
				// Geometry
				const auto& c1 = cell_centers[cell_id1];
				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
				P = I3 - outer_product(n, n.transpose());
				r1 = (n.transpose() * (face.c - c1))(0, 0);		assert(r1 > 0.0);
				y1 = c1 + r1 * n;
				B1n = biots[cell_id1] * n;
				// Stiffness decomposition
				C1 = W * stfs[cell_id1] * W.transpose();
				nblock = make_block_diagonal(n, ND);
				nblock_t = make_block_diagonal(n.transpose(), ND);
				tblock = make_block_diagonal(P, ND);
				T1 = nblock_t * C1 * nblock;
				G1 = nblock_t * C1 * tblock;
				// Permeability decomposition
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				D1n = pm_discretizer::heat_cond_constant * diffs[cell_id1] * n;		
				lam1 = (n.transpose() * K1n)(0, 0);
				gam1 = K1n - lam1 * n;
				// Extra 'boundary' stuff
				An = (an * I3 + bn / r1 * T1);
				At = (at * I3 + bt / r1 * T1);
				Ap = 1.0 / (ap + bp / r1 / visc * lam1);
				res = At.inv();
				if (!res)
				{
					cout << "Inversion failed!\n";	exit(-1);
				}
				L = An * At;
				gamma = 1.0 / (n.transpose() * L * n).values[0];
				gamma_nnt = gamma * outer_product(n, n.transpose());
				gamma_nnt_mult = gamma_nnt * (bn * I3 - bt * L);
				mult_p = (bt * I3 + gamma_nnt_mult) * B1n;
				// Filling mechanics equations
				A(4 * face_id * 4 * ND,
					{ ND, 3 * ND },
					{ 4 * ND, 1 }) = (at * make_block_diagonal((face.c - c1).transpose(), ND) +
						bt * nblock_t * C1 +
						gamma_nnt_mult * (G1 + T1 / r1 *
							make_block_diagonal((y1 - face.c).transpose(), ND))).values;
				A(4 * face_id * 4 * ND + 3 * ND,
					{ ND, ND },
					{ 4 * ND, 1 }) = outer_product(mult_p, Ap * bp / visc * ((lam1 / r1 * (y1 - face.c) + gam1).transpose() *
						P)).values;

				res1 = findInVector(st, face.cell_id1);
				if (res1.first) { id = res1.second; }
				else { id = st.size(); st.push_back(face.cell_id1); }

				rhs_mult(4 * face_id * rhs_mult.N + 4 * id, { 3, 3 }, { (size_t)rhs_mult.N, 1 }) += (gamma_nnt_mult * T1 / r1 - at * I3).values;
				rhs_mult(4 * face_id * rhs_mult.N + 4 * id + 3, { 3, 1 }, { (size_t)rhs_mult.N, 1 }) += (Ap * bp / visc * lam1 / r1 * mult_p).values;
				rest(4 * face_id, { 3 }, { 1 }) = (mult_p * Ap * bp / visc * (grav_vec * K1n).values[0]).values;
				// Filling flow equation
				A((4 * face_id + 3) * 4 * ND + 3 * ND,
					{ ND },
					{ 1 }) = (ap * (face.c - c1) + bp / visc * K1n).values;
				Ad(face_id * ND, { ND }, { 1 }) = (ath * (face.c - c1) + bth * D1n).values;
				rhs_mult(4 * face_id + 3, 4 * id + 3) -= ap;
				rhs_multd(face_id, id) -= ath;
				rest(4 * face_id + 3, 0) = bp / visc * (grav_vec * K1n).values[0];
				// BC RHS coefficients
				bface_id = n_cells + face.face_id2;
				res2 = findInVector(st, bface_id);
				if (res2.first) { id = res2.second; }
				else { id = st.size(); st.push_back(bface_id); }
				// Mech
				rhs_mult(4 * face_id * rhs_mult.N + 4 * id, { 3, 3 }, { (size_t)rhs_mult.N, 1 }) = (gamma_nnt + (I3 - gamma_nnt * L) * P).values;
				rhs_mult(4 * face_id * rhs_mult.N + 4 * id + 3, { 3, 1 }, { (size_t)rhs_mult.N, 1 }) = (mult_p * Ap).values;
				// Flow
				rhs_mult(4 * face_id + 3, 4 * id + 3) = 1.0;
				rhs_multd(face_id, id) = 1.0;

				face_id++;
			}
			else if (face.type == MAT_TO_FRAC)
			{
				const Face& face0 = vec_faces[face.face_id1];
				const auto& cur = inner[cell_id][face.face_id1];
				// recount original face id
				face_id1 = 0;
				for (index_t h = 0; h < face.face_id1; h++)
				{
					const Face& face = vec_faces[h];
					if (face.type == BORDER)
					{
						const auto& b = bc[face.face_id2];
						const auto& an = b(0, 0);	const auto& at = b(2, 0);
						if (an != 0.0 || at != 0.0)	face_id1++;
					}
					else if (face.type != MAT_TO_FRAC) face_id1++;
				}

				const int& cell_id1 = face0.cell_id1;
				const int& frac_id = face.cell_id2;
				assert(frac_id >= n_matrix);
				const auto& c1 = cell_centers[cell_id1];
				const auto& c2 = cell_centers[frac_id];
				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
				sign = get_fault_sign(n, ref_contact_ids[frac_id - n_matrix]);
				// add gap
				res1 = findInVector(st, frac_id);
				if (res1.first) { id1 = res1.second; }
				else { id1 = st.size(); st.push_back(frac_id); }
				rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id1,
					{ BLOCK_SIZE, BLOCK_SIZE },
					{ (size_t)rhs_mult.N, 1 }) += -sign * (cur.Q2 + cur.r2 * cur.A1).values;
				// add gap gradient
				const auto& frac_grad = grad[frac_id];
				auto& frac_grad_mult = pre_frac_grad_mult[frac_grad.stencil.size()];
				std::fill_n(&frac_grad_mult.values[0], frac_grad_mult.values.size(), 0.0);
				frac_grad_mult = sign * (cur.Q2 * make_block_diagonal((cur.y2 - face.c).transpose(), BLOCK_SIZE) - cur.r2 * cur.Th2) * frac_grad.mat;
				for (st_id = 0; st_id < frac_grad.stencil.size(); st_id++)
				{
					assert(frac_grad.stencil[st_id] >= n_matrix);
					res1 = findInVector(st, frac_grad.stencil[st_id]);
					if (res1.first) { id = res1.second; }
					else { id = st.size(); st.push_back(frac_grad.stencil[st_id]); }
					rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id,
						{ BLOCK_SIZE, BLOCK_SIZE },
						{ (size_t)rhs_mult.N, 1 }) -= frac_grad_mult(st_id * BLOCK_SIZE,
							{ BLOCK_SIZE, BLOCK_SIZE },
							{ (size_t)frac_grad_mult.N, 1 });
				}
				// no discontinuity in pressure
				rhs_mult(BLOCK_SIZE * face_id1 * rhs_mult.N + BLOCK_SIZE * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = 0.0;

				// pressure condition
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				K2n = pm_discretizer::darcy_constant * perms[frac_id] * n;
				r2_frac = frac_apers[frac_id - n_matrix] / 2;
				lam2 = (n.transpose() * K2n)(0, 0);

				// remove previous numbers
				A((BLOCK_SIZE * face_id1 + ND) * A.N, { BLOCK_SIZE * ND }, { 1 }) = 0.0;
				rhs_mult((BLOCK_SIZE * face_id1 + ND) * rhs_mult.N, { (size_t)rhs_mult.N }, { 1 }) = 0.0;
				rest(BLOCK_SIZE * face_id1 + ND, 0) = 0.0;
				// fill newer
				A((BLOCK_SIZE * face_id1 + ND) * A.N + ND * ND, { ND }, { 1 }) = (c2 - c1 + r2_frac / lam2 * (K1n - K2n)).values;
				res1 = findInVector(st, cell_id1);
				if (res1.first) { id = res1.second; }
				else { id = st.size(); st.push_back(cell_id1); }
				rhs_mult(BLOCK_SIZE * face_id1 + ND, BLOCK_SIZE * id + ND) = -1.0;

				res2 = findInVector(st, frac_id);
				if (res2.first) { id = res2.second; }
				else { id = st.size(); st.push_back(frac_id); }
				rhs_mult(BLOCK_SIZE * face_id1 + ND, BLOCK_SIZE * id + ND) = 1.0;
				//face_id++;
			}
		}
		sq_mat = A.transpose() * A;

		res = sq_mat.inv();
		if (!res)
		{
			cout << "Inversion failed!\n";
			//sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");
			exit(-1);
		}
		auto& cur_rhs = pre_cur_rhs[n_cur_faces][st.size()];
		cur_rhs.values = rhs_mult(0, { 4 * n_cur_faces, 4 * st.size() }, { (size_t)rhs_mult.N, 1 });

		auto& cur_grad = grad[cell_id];
		cur_grad.stencil = st;
		cur_grad.mat = sq_mat * A.transpose() * cur_rhs;
		cur_grad.rhs = sq_mat * A.transpose() * rest;
		
		// Heat conduction
		sq_matd = Ad.transpose() * Ad;
		res = sq_matd.inv();
		if (!res)
		{
			cout << "Inversion of heat conduction failed!\n";
			//sq_mat.write_in_file("sq_mat_" + std::to_string(cell_id) + ".txt");
			exit(-1);
		}

		auto& cur_rhsd = pre_cur_rhsd[n_cur_faces][st.size()];
		cur_rhsd.values = rhs_multd(0, { n_cur_faces, st.size() }, { (size_t)rhs_multd.N, 1 });

		auto& cur_grad_d = grad_d[cell_id];
		//cur_grad_d.stencil = st;
		cur_grad_d.mat = sq_matd * Ad.transpose() * cur_rhsd;
		//cur_grad_d.rhs = sq_matd * Ad.transpose() * restd;
	}
	printf("Gradient reconstruction was done!\n");
}
/*void pm_discretizer::calc_all_fluxes(value_t dt)
{
	bool isStationary = dt == 0.0 ? true : false;
	if (isStationary) dt = 1.0;
	assert(grad.size() > 0);
	cell_m.clear();		cell_p.clear();
	stencil.clear();	offset.clear();
	tran.clear();		rhs.clear();

	// Variables
	Matrix det(BLOCK_SIZE, BLOCK_SIZE), Q(BLOCK_SIZE, BLOCK_SIZE), coef1(BLOCK_SIZE, BLOCK_SIZE), coef2(BLOCK_SIZE, BLOCK_SIZE), grad_coef(ND * BLOCK_SIZE, ND * BLOCK_SIZE);
	value_t r1, Ap, gamma, lam1;
	bool res;
	Gradients g;
	Matrix C1(ND*ND, ND*ND), T1(ND, ND), G1(ND, ND*ND), T1inv(ND, ND);
	Matrix P(ND, ND), nblock(ND*ND, ND), nblock_t(ND, ND*ND), tblock(ND*ND, ND*ND);
	Matrix n(ND, 1), K1n(ND, 1), gam1(ND, 1), B1n(ND, 1);
	Matrix y1(ND, 1), An(ND, ND), At(ND, ND), 
		L(ND, ND), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), coef(ND, ND), mult_p(ND, 1), mult_u(ND, ND), gu_coef(ND, ND * ND);
	size_t n_cur_faces;
	Matrix tmp(BLOCK_SIZE, ND * BLOCK_SIZE);
	int cell_id, face_id, bface_id;

	offset.push_back(0);
	for (cell_id = 0; cell_id < n_cells; cell_id++)
	{
		const auto& vec_faces = faces[cell_id];
		n_cur_faces = vec_faces.size();
		for (face_id = 0; face_id < n_cur_faces; face_id++)
		{
			const Face& face = vec_faces[face_id];
			cell_m.push_back(face.cell_id1);
			if (face.type == MAT)
			{
				cell_p.push_back(face.cell_id2);
				n = (face.n.transpose() * (cell_centers[face.cell_id2] - cell_centers[face.cell_id1])).values[0] > 0 ? face.n : -face.n;
				P = I3 - outer_product(n, n.transpose());
				const auto& cur = inner[cell_id][face_id];
				det = cur.r1 * cur.Q2 + cur.r2 * cur.Q1;
				res = det.inv();
				if (!res) { cout << "Inversion failed!\n";	exit(-1); }
				Q = cur.Q1 * det * cur.Q2;
				coef1 = cur.r1 * cur.Q2 * det;
				coef2 = cur.r2 * cur.Q1 * det;
				grad_coef = (Q * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE) +
									coef1 * cur.Th1 + coef2 * cur.Th2) * 
									make_block_diagonal(P, BLOCK_SIZE);
				//grad_coef.values[std::abs(grad_coef.values) < EQUALITY_TOLERANCE] = 0.0;
				const auto& g1 = grad[face.cell_id1];
				const auto& g2 = grad[face.cell_id2];
				g = merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat); 
				a = grad_coef * g.mat;
				f = grad_coef * (g1.rhs + g2.rhs) / 2.0 + coef1 * cur.R1 + coef2 * cur.R2;

				res1 = findInVector(g.stencil, face.cell_id1);
				if (res1.first) { id = res1.second; }
				else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
				a(BLOCK_SIZE * id, { (size_t)a.M, BLOCK_SIZE }, { (size_t)a.N, 1 }) += (coef1 * cur.A1 - Q).values;

				res2 = findInVector(g.stencil, face.cell_id2);
				if (res2.first) { id = res2.second; }
				else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id2);	exit(-1); }
				a(BLOCK_SIZE * id, { (size_t)a.M, BLOCK_SIZE }, { (size_t)a.N, 1 }) += (coef2 * cur.A2 + Q).values;

				a.values *= face.area;
				f.values *= face.area;
			}
			else if (face.type == BORDER)
			{
				std::fill_n(&tmp.values[0], tmp.values.size(), 0.0);
				const auto& b = bc[face.face_id2];
				const auto& an = b(0, 0);			const auto& bn = b(1, 0);
				const auto& at = b(2, 0);			const auto& bt = b(3, 0);
				const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
				const int& cell_id1 = face.cell_id1;
				// Geometry
				const auto& c1 = cell_centers[cell_id1];
				n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
				P = I3 - outer_product(n, n.transpose());
				r1 = (n.transpose() * (face.c - c1))(0, 0);		assert(r1 > 0.0);
				y1 = c1 + r1 * n;
				// Stiffness decomposition
				C1 = W * stfs[cell_id1] * W.transpose();
				nblock = make_block_diagonal(n, ND);
				nblock_t = make_block_diagonal(n.transpose(), ND);
				tblock = make_block_diagonal(P, ND);
				T1 = nblock_t * C1 * nblock;
				G1 = nblock_t * C1 * tblock;
				T1inv = T1;
				res = T1inv.inv();
				if (!res) { cout << "Inversion failed!\n";	exit(-1); }
				// Permeability decomposition
				K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
				lam1 = (n.transpose() * K1n)(0, 0);
				gam1 = K1n - lam1 * n;
				// Extra 'boundary' stuff
				An = (an * I3 + bn / r1 * T1);
				At = (at * I3 + bt / r1 * T1);
				Ap = 1.0 / (ap + bp / r1 / visc * lam1);
				res = At.inv();
				if (!res) { cout << "Inversion failed!\n";	exit(-1); }
				L = An * At;
				gamma = 1.0 / (n.transpose() * L * n).values[0];
				gamma_nnt = gamma * outer_product(n, n.transpose());
				gamma_nnt_mult = gamma_nnt * (bn * I3 - bt * L);
				coef = -T1 / r1 * At * (gamma_nnt_mult - r1 * at * T1inv);
				mult_p = biots[cell_id1] * n;
				mult_u = At * (bt * I3 + gamma_nnt * (bn * I3 - bt * L));
				gu_coef = T1 / r1 * make_block_diagonal((y1 - face.c).transpose(), ND) + G1;
				// Gradient
				g = grad[face.cell_id1]; 

				// Filling mechanics equations
				tmp(0, { ND, ND * ND }, { (size_t)tmp.N, 1 }) = -(coef * gu_coef).values;
				tmp(ND * ND, { ND, ND }, { (size_t)tmp.N, 1 }) = (coef * outer_product(mult_p, -Ap * bp / visc * (lam1 / r1 * (y1 - face.c) + gam1).transpose())).values;
				f(0, { ND }, { 1 }) = (Ap * bp / visc * (grav_vec * K1n).values[0] * coef * mult_p).values;

				// Filling flow equation
				tmp(ND * tmp.N, { ND * ND }, { 1 }) = -(mult_p.transpose() * mult_u * gu_coef).values;
				tmp(ND * tmp.N + ND * ND, { ND }, { 1 }) = (-dt / visc * Ap * ap * (lam1 / r1 * (y1 - face.c) + gam1) -
															(mult_p.transpose() * (mult_u * mult_p)).values[0] * Ap * bp / visc * 
															(lam1 / r1 * (y1 - face.c) + gam1) ).values;
				const auto ub_prev = get_ub_prev(face);
				f(ND, 0) = -dt / visc * Ap * (- ap * (grav_vec * K1n).values[0]) - 
								(mult_p.transpose() * ub_prev).values[0] +
							Ap * (bp / visc * (grav_vec * K1n).values[0]) * (mult_p.transpose() * (mult_u * mult_p)).values[0];

				// Assembling fluxes
				a = tmp * make_block_diagonal(P, BLOCK_SIZE) * g.mat;
				f += tmp * make_block_diagonal(P, BLOCK_SIZE) * g.rhs;
				// Add extra 'cell_id' contribution
				res1 = findInVector(g.stencil, face.cell_id1);
				if (res1.first) { id = res1.second; }
				else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
				a(BLOCK_SIZE * id, { ND, ND }, { (size_t)a.N, 1 }) += (coef * T1 / r1).values;
				a(BLOCK_SIZE * id + ND, { ND, 1 }, { (size_t)a.N, 1 }) += (coef * mult_p * Ap * bp / visc * lam1 / r1).values;
				a(BLOCK_SIZE * id + ND * a.N, { ND }, { 1 }) += (mult_u * T1 / r1 * mult_p).values;
				a(ND, BLOCK_SIZE * id + ND) += dt / visc * Ap * ap * lam1 / r1 +
					(mult_p.transpose() * (mult_u * mult_p)).values[0] * Ap * bp / visc * lam1 / r1;
				// BC RHS coefficients
				bface_id = n_cells + face.face_id2;
				res2 = findInVector(g.stencil, bface_id);
				if (res2.first) { id = res2.second; }
				else { printf("Gradient within %d cell does not depend on boundary value!\n", face.cell_id1);	exit(-1); }
				a(BLOCK_SIZE * id, { ND, ND }, { (size_t)a.N, 1 }) += (-T1 / r1 * At * (gamma_nnt + (I3 - gamma_nnt * L) * P)).values;
				a(BLOCK_SIZE * id + ND, { ND, 1 }, { (size_t)a.N, 1 }) += (Ap * coef * mult_p).values;
				a(BLOCK_SIZE * id + ND * a.N, { ND }, { 1 }) += (mult_p.transpose() * At * (gamma_nnt + (I3 - gamma_nnt * L) * P)).values;
				a(ND, BLOCK_SIZE * id + ND) += -dt / visc * Ap * lam1 / r1 + Ap * (mult_p.transpose() * (mult_u * mult_p)).values[0];
				// area multiplier
				a.values *= face.area;
				f.values *= face.area;
				cell_p.push_back(bface_id);
			}
			else
			{
				printf("Fractures are not supported yet\n");
				exit(-1);
			}

			//assert(check_trans_sum(g.stencil, a));
			//stencil.insert(stencil.end(), g.stencil.begin(), g.stencil.end());
			write_trans(g.stencil, a);
			offset.push_back(stencil.size());
			//tran.insert(std::end(tran), std::begin(a.values), std::end(a.values));
			rhs.insert(std::end(rhs), std::begin(f.values), std::end(f.values));
		}
	}

	printf("Calculation of fluxes was done!\n");
}*/
void pm_discretizer::calc_border_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_th_cond, Approximation& face_unknown)
{
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Matrix grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), face_unknowns_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE);
	Matrix C1(ND*ND, ND*ND), T1(ND, ND), G1(ND, ND*ND), T1inv(ND, ND);
	Matrix P(ND, ND), nblock(ND*ND, ND), nblock_t(ND, ND*ND), tblock(ND*ND, ND*ND);
	Matrix n(ND, 1), K1n(ND, 1), gam1(ND, 1), K2n(ND, 1), gam2(ND, 1), B1n(ND, 1);
	Matrix y1(ND, 1), An(ND, ND), At(ND, ND), L(ND, ND), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), coef(ND, ND), mult_p(ND, 1), mult_u(ND, ND), gu_coef(ND, ND * ND);
	value_t r1, Ap, gamma, lam1;
	index_t bface_id;
	bool res;

	const auto& b = bc[face.face_id2];
	const auto& an = b(0, 0);			const auto& bn = b(1, 0);
	const auto& at = b(2, 0);			const auto& bt = b(3, 0);
	const auto& ap = b(4, 0);			const auto& bp = b(5, 0);
	const int& cell_id1 = face.cell_id1;
	// Geometry
	const auto& c1 = cell_centers[cell_id1];
	n = (face.n.transpose() * (face.c - c1)).values[0] > 0 ? face.n : -face.n;
	P = I3 - outer_product(n, n.transpose());
	r1 = (n.transpose() * (face.c - c1))(0, 0);		assert(r1 > 0.0);
	y1 = c1 + r1 * n;
	// Stiffness decomposition
	C1 = W * stfs[cell_id1] * W.transpose();
	nblock = make_block_diagonal(n, ND);
	nblock_t = make_block_diagonal(n.transpose(), ND);
	tblock = make_block_diagonal(P, ND);
	T1 = nblock_t * C1 * nblock;
	G1 = nblock_t * C1 * tblock;
	T1inv = T1;
	res = T1inv.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}
	// Permeability decomposition
	K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
	lam1 = (n.transpose() * K1n)(0, 0);
	gam1 = K1n - lam1 * n;
	// Extra 'boundary' stuff
	An = (an * I3 + bn / r1 * T1);
	At = (at * I3 + bt / r1 * T1);
	Ap = 1.0 / (ap + bp / r1 / visc * lam1);
	res = At.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}
	L = An * At;
	gamma = 1.0 / (n.transpose() * L * n).values[0];
	gamma_nnt = gamma * outer_product(n, n.transpose());
	gamma_nnt_mult = gamma_nnt * (bn * I3 - bt * L);
	coef = -T1 / r1 * At * (gamma_nnt_mult - r1 * at * T1inv);
	mult_p = biots[cell_id1] * n;
	mult_u = At * (bt * I3 + gamma_nnt * (bn * I3 - bt * L));
	gu_coef = T1 / r1 * make_block_diagonal((y1 - face.c).transpose(), ND) + G1;
	// Gradient
	g = grad[face.cell_id1];

	//// Filling mechanics equations
	grad_coef(0, { ND, ND * ND }, { (size_t)grad_coef.N, 1 }) = -(coef * gu_coef).values;
	biot_grad_coef(ND * ND, { ND, ND }, { (size_t)grad_coef.N, 1 }) = (coef * outer_product(mult_p, -Ap * bp / visc * (lam1 / r1 * (y1 - face.c) + gam1).transpose())).values;
	flux.f_biot(0, { ND }, { 1 }) = (Ap * bp / visc * (grav_vec * K1n).values[0] * coef * mult_p).values;
	// face unknowns (pressure)
	face_unknowns_grad_coef(ND * biot_grad_coef.N + ND * ND, { ND }, { 1 }) = -(Ap * bp / visc * (lam1 / r1 * (y1 - face.c) + gam1)).values;
	face_unknown.f(ND, 0) = Ap * bp / visc * (grav_vec * K1n).values[0];

	//// Filling flow equation
	biot_grad_coef(ND * biot_grad_coef.N, { ND * ND }, { 1 }) = -(mult_p.transpose() * mult_u * gu_coef).values;
	grad_coef(ND * grad_coef.N + ND * ND, { ND }, { 1 }) = (-1.0 / visc * Ap * ap * (lam1 / r1 * (y1 - face.c) + gam1)).values;
	biot_grad_coef(ND * biot_grad_coef.N + ND * ND, { ND }, { 1 }) = -((mult_p.transpose() * (mult_u * mult_p)).values[0] * Ap * bp / visc *
		(lam1 / r1 * (y1 - face.c) + gam1)).values;
	flux.f(ND, 0) = -1.0 / visc * Ap * (-ap * (grav_vec * K1n).values[0]);
	flux.f_biot(ND, 0) = Ap * (bp / visc * (grav_vec * K1n).values[0]) * (mult_p.transpose() * (mult_u * mult_p)).values[0];
	// face unknowns (displacements)
	face_unknowns_grad_coef(0, { ND, ND * ND }, { (size_t)face_unknowns_grad_coef.N, 1 }) = -(mult_u * gu_coef).values;
	face_unknowns_grad_coef(ND * ND, { ND, ND }, { (size_t)face_unknowns_grad_coef.N, 1 }) = outer_product(mult_u * mult_p, -Ap * bp / visc * (lam1 / r1 * (y1 - face.c) + gam1).transpose()).values;
	face_unknown.f(0, { ND }, { 1 }) = Ap * (bp / visc * (grav_vec * K1n).values[0]) * (mult_u * mult_p).values;

	//// Assembling fluxes
	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * make_block_diagonal(P, BLOCK_SIZE) * g.mat).values;
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) = (biot_grad_coef * make_block_diagonal(P, BLOCK_SIZE) * g.mat).values;
	flux.f += grad_coef * make_block_diagonal(P, BLOCK_SIZE) * g.rhs;
	flux.f_biot += biot_grad_coef * make_block_diagonal(P, BLOCK_SIZE) * g.rhs;
	// face unknowns
	fill_n(std::begin(face_unknown.a.values), face_unknown.a.values.size(), 0.0);
	face_unknown.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)face_unknown.a.N, 1 }) = (face_unknowns_grad_coef * make_block_diagonal(P, BLOCK_SIZE) * g.mat).values;
	face_unknown.f += face_unknowns_grad_coef * make_block_diagonal(P, BLOCK_SIZE) * g.rhs;

	//// Add extra 'cell_id' contributions
	res1 = findInVector(g.stencil, face.cell_id1);
	if (res1.first) { id1 = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
	flux.a(BLOCK_SIZE * id1, { ND, ND }, { (size_t)flux.a.N, 1 }) += (coef * T1 / r1).values;
	flux.a_biot(BLOCK_SIZE * id1 + ND, { ND, 1 }, { (size_t)flux.a_biot.N, 1 }) += (coef * mult_p * Ap * bp / visc * lam1 / r1).values;
	flux.a_biot(BLOCK_SIZE * id1 + ND * flux.a_biot.N, { ND }, { 1 }) += (mult_u * T1 / r1 * mult_p).values;
	flux.a(ND, BLOCK_SIZE * id1 + ND) += 1.0 / visc * Ap * ap * lam1 / r1;
	flux.a_biot(ND, BLOCK_SIZE * id1 + ND) += (mult_p.transpose() * (mult_u * mult_p)).values[0] * Ap * bp / visc * lam1 / r1;
	// face unknowns
	face_unknown.a(BLOCK_SIZE * id1, { ND, ND }, { (size_t)face_unknown.a.N, 1 }) += (mult_u * T1 / r1).values;
	face_unknown.a(BLOCK_SIZE * id1 + ND, { ND, 1 }, { (size_t)face_unknown.a.N, 1 }) += Ap * bp / visc * lam1 / r1 * (mult_u * mult_p).values;
	face_unknown.a(ND, BLOCK_SIZE * id1 + ND) += Ap * bp / visc * lam1 / r1;

	// BC RHS coefficients
	bface_id = n_cells + face.face_id2;
	res2 = findInVector(g.stencil, bface_id);
	if (res2.first) { id2 = res2.second; }
	else { id2 = g.stencil.size(); g.stencil.push_back(bface_id); }
	flux.a(BLOCK_SIZE * id2, { ND, ND }, { (size_t)flux.a.N, 1 }) += (-T1 / r1 * At * (gamma_nnt + (I3 - gamma_nnt * L) * P)).values;
	flux.a_biot(BLOCK_SIZE * id2 + ND, { ND, 1 }, { (size_t)flux.a_biot.N, 1 }) += (Ap * coef * mult_p).values;
	flux.a_biot(BLOCK_SIZE * id2 + ND * flux.a_biot.N, { ND }, { 1 }) += (mult_p.transpose() * At * (gamma_nnt + (I3 - gamma_nnt * L) * P)).values;
	flux.a(ND, BLOCK_SIZE * id2 + ND) += -1.0 / visc * Ap * lam1 / r1;
	flux.a_biot(ND, BLOCK_SIZE * id2 + ND) += Ap * (mult_p.transpose() * (mult_u * mult_p)).values[0];
	// face unknowns
	face_unknown.a(BLOCK_SIZE * id2, { ND, ND }, { (size_t)face_unknown.a.N, 1 }) += (At * (gamma_nnt + (I3 - gamma_nnt * L) * P)).values;
	face_unknown.a(BLOCK_SIZE * id2 + ND, { ND, 1 }, { (size_t)face_unknown.a.N, 1 }) += (Ap * (mult_u * mult_p)).values;
	face_unknown.a(ND, BLOCK_SIZE * id2 + ND) += Ap;

	flux.stencil = g.stencil;
	// face unknowns (stencil)
	face_unknown.stencil = g.stencil;

	if (ASSEMBLE_HEAT_CONDUCTION)
	{
		const auto& ath = b(6, 0);			const auto& bth = b(7, 0);
		Matrix D1n(ND, 1), dgam1(ND, 1);
		value_t dlam1, Ath;
		D1n = diffs[face.cell_id1] * n;
		dlam1 = (n.transpose() * D1n).values[0];
		dgam1 = D1n - dlam1 * n;
		Ath = 1.0 / (ath + bth * dlam1 / r1);
		const auto& cur_grad = grad_d[face.cell_id1];

		fill_n(std::begin(flux_th_cond.a.values), flux_th_cond.a.values.size(), 0.0);
		flux_th_cond.a(0, { (size_t)cur_grad.mat.N }, { 1 }) = -Ath * ath * ((dlam1 / r1  * (y1 - face.c) + dgam1).transpose() * cur_grad.mat).values;
		flux_th_cond.a(0, id1) += Ath * ath * dlam1 / r1;
		flux_th_cond.a(0, id2) = -Ath * dlam1 / r1;
	}
}
void pm_discretizer::calc_contact_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_flow_biot, index_t fault_id)
{
	Matrix det(BLOCK_SIZE, BLOCK_SIZE), Q(BLOCK_SIZE, BLOCK_SIZE), coef1(BLOCK_SIZE, BLOCK_SIZE), coef2(BLOCK_SIZE, BLOCK_SIZE), grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE),
		biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), frac_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), biot_frac_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), A1_tilde(BLOCK_SIZE, BLOCK_SIZE),
		biot_flow_buf(BLOCK_SIZE, BLOCK_SIZE), block(BLOCK_SIZE * ND, BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Matrix n(ND, 1), P(ND, ND);
	bool res;
	value_t sign;

	n = (face.n.transpose() * (cell_centers[face.cell_id2] - cell_centers[face.cell_id1])).values[0] > 0 ? face.n : -face.n;
	P = I3 - outer_product(n, n.transpose());
	const auto& cur = inner[face.cell_id1][face.face_id1];
	det = cur.r1 * cur.Q2 + cur.r2 * cur.Q1;
	res = det.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}

	coef1 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A1);
	coef2 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A2);
	grad_coef = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.r2 * (cur.Th2 - cur.Th1) +
		cur.Q2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE)) + cur.Th1 -
		cur.A1 * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);
	const auto& g1 = grad[face.cell_id1];
	const auto& g2 = grad[face.cell_id2];
	// Biot term for flow
	A1_tilde(ND * BLOCK_SIZE, { ND }, { 1 }) = (biots[face.cell_id1] * n).values;
	A1_tilde(ND, { ND, 1 }, { BLOCK_SIZE, 1 }) = (biots[face.cell_id1] * n).values;
	biot_flow_buf = cur.r1 * A1_tilde * det;
	biot_grad_coef = biot_flow_buf * (cur.r2 * (cur.Th2 - cur.Th1) +
		cur.Q2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE)) +
		A1_tilde * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);

	fill_n(std::begin(flux.f.values), flux.f.values.size(), 0.0);
	fill_n(std::begin(flux.f_biot.values), flux.f_biot.values.size(), 0.0);
	fill_n(std::begin(flux_flow_biot.f.values), flux_flow_biot.f.values.size(), 0.0);
	fill_n(std::begin(flux_flow_biot.f_biot.values), flux_flow_biot.f_biot.values.size(), 0.0);

	g = merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat);
	flux.f = grad_coef * (g1.rhs + g2.rhs) / 2.0 + cur.R1 + cur.r2 * (cur.Q1 - cur.r1 * cur.A1) * det * (cur.R2 - cur.R1);
	flux.f_biot = biot_grad_coef * (g1.rhs + g2.rhs) / 2.0 + biot_flow_buf * cur.r2 * (cur.R2 - cur.R1);
	flux_flow_biot.f_biot = biot_grad_coef * (g1.rhs + g2.rhs) / 2.0 + biot_flow_buf * cur.r2 * (cur.R2 - cur.R1);

	//g = g1;
	//flux.f = grad_coef * g1.rhs + coef1 * cur.R1 + coef2 * cur.R2;
	//flux.f_biot = biot_grad_coef * g1.rhs + biot_flow_buf * cur.r2 * (cur.R2 - cur.R1);
	//flux_flow_biot.f_biot = biot_grad_coef * g1.rhs + biot_flow_buf * cur.r2 * (cur.R2 - cur.R1);

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	fill_n(std::begin(flux_flow_biot.a.values), flux_flow_biot.a.values.size(), 0.0);
	fill_n(std::begin(flux_flow_biot.a_biot.values), flux_flow_biot.a_biot.values.size(), 0.0);

	flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * g.mat).values;
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) = (biot_grad_coef * g.mat).values;
	flux_flow_biot.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux_flow_biot.a_biot.N, 1 }) = (biot_grad_coef * g.mat).values;

	// fault extras
	const auto& fault = faces[face.cell_id1][fault_id];
	assert(fault.cell_id2 >= n_matrix);
	sign = get_fault_sign(n, ref_contact_ids[fault.cell_id2 - n_matrix]);
	res1 = findInVector(g.stencil, fault.cell_id2);
	if (res1.first) { id = res1.second; }
	else { id = g.stencil.size(); g.stencil.push_back(fault.cell_id2); }
	// add gap
	flux.a(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) -= sign * coef1.values;
	flux_flow_biot.a_biot(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux_flow_biot.a_biot.N, 1 }) += -sign * (biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;
	// add gap gradient
	const auto& frac_grad = grad[fault.cell_id2];
	frac_grad_coef = sign * (Q * make_block_diagonal((face.c - cur.y2).transpose(), BLOCK_SIZE) + coef2 * cur.Th2);
	biot_frac_grad_coef = sign * biot_flow_buf * (cur.r2 * cur.Th2 + cur.Q2 * make_block_diagonal((face.c - cur.y2).transpose(), BLOCK_SIZE));
	for (st_id = 0; st_id < frac_grad.stencil.size(); st_id++)
	{
		res1 = findInVector(g.stencil, frac_grad.stencil[st_id]);
		if (res1.first) { id = res1.second; }
		else { id = g.stencil.size(); g.stencil.push_back(frac_grad.stencil[st_id]); }
		block.values = frac_grad.mat(BLOCK_SIZE * st_id, { (size_t)frac_grad.mat.M, BLOCK_SIZE }, { (size_t)frac_grad.mat.N, 1 });
		block(ND * ND * block.N, { ND, (size_t)block.N }, { (size_t)block.N, 1 }) = 0.0;			// no discotinuity in pressure
		flux.a(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += (frac_grad_coef * block).values;
		flux_flow_biot.a_biot(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux_flow_biot.a_biot.N, 1 }) += (biot_frac_grad_coef * block).values;
	}
	flux.f += frac_grad_coef * frac_grad.rhs;
	flux.f_biot += biot_frac_grad_coef * frac_grad.rhs;

	res1 = findInVector(g.stencil, face.cell_id1);
	if (res1.first) { id = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
	flux.a(BLOCK_SIZE * id, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += -coef1.values;
	flux.a_biot(BLOCK_SIZE * id, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += (A1_tilde - biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;
	flux_flow_biot.a_biot(BLOCK_SIZE * id, { (size_t)flux_flow_biot.a_biot.M, BLOCK_SIZE }, { (size_t)flux_flow_biot.a_biot.N, 1 }) += (A1_tilde - biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;

	res2 = findInVector(g.stencil, face.cell_id2);
	if (res2.first) { id = res2.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id2);	exit(-1); }
	flux.a(BLOCK_SIZE * id, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += coef2.values;
	flux.a_biot(BLOCK_SIZE * id, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += (biot_flow_buf * (cur.Q2 + cur.r2 * cur.A2)).values;
	flux_flow_biot.a_biot(BLOCK_SIZE * id, { (size_t)flux_flow_biot.a_biot.M, BLOCK_SIZE }, { (size_t)flux_flow_biot.a_biot.N, 1 }) += (biot_flow_buf * (cur.Q2 + cur.r2 * cur.A2)).values;

	// no direct fluid flow to another side of the fault
	flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
	//flux.a_biot(ND * flux.a_biot.N, { (size_t)flux.a_biot.N }, { 1 }) = 0.0;
	flux.f(ND, 0) = 0.0;
	//flux.f_biot(ND, 0) = 0.0;

	flux.a_biot.values = 0.0;
	flux.f_biot.values = 0.0;

	flux.stencil = g.stencil;
	// only fluid mass flux to the fault
	flux_flow_biot.a_biot(0, { ND, (size_t)flux_flow_biot.a_biot.N }, { (size_t)flux_flow_biot.a_biot.N, 1 }) = 0.0;
	flux_flow_biot.f_biot(0, { ND, 1 }, { (size_t)flux_flow_biot.f_biot.N, 1 }) = 0.0;
	flux_flow_biot.stencil = g.stencil;
}
void pm_discretizer::calc_contact_flux_new(value_t dt, const Face& face, Approximation& flux, Approximation& flux_flow_biot, index_t fault_id)
{
	Matrix det(ND, ND), coef1(ND, ND), coef2(ND, ND), coef_biot(ND, ND), T(ND, ND), block(BLOCK_SIZE * ND, BLOCK_SIZE),
				frac_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE),
				biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), frac_biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Matrix n(ND, 1), P(ND, ND);
	index_t biot_id;
	bool res;

	n = (face.n.transpose() * (cell_centers[face.cell_id2] - cell_centers[face.cell_id1])).values[0] > 0 ? face.n : -face.n;
	P = I3 - outer_product(n, n.transpose());
	const auto& fault = faces[face.cell_id1][fault_id];
	const auto& cur = inner[face.cell_id1][face.face_id1];
	det = cur.r1 * cur.T2 + cur.r2 * cur.T1;
	res = det.inv();
	if (!res) { cout << "Inversion failed!\n";	exit(-1); }
	T = cur.T1 * det * cur.T2;

	coef1 = cur.r1 * cur.T2 * det;
	coef2 = cur.r2 * cur.T1 * det;
	coef_biot = (biots[face.cell_id1] * n).transpose() * det;
	value_t r11 = cur.r1 - frac_apers[fault.cell_id2 - n_matrix] / 2.0;
	grad_coef(0, { ND, ND * ND }, { (size_t)grad_coef.N, 1 }) = (T * make_block_diagonal((cur.y2 - cur.y1).transpose(), ND) - 
																	 coef1 * cur.G1 - coef2 * cur.G2).values;
	biot_grad_coef(ND * ND * BLOCK_SIZE, { ND * ND }, { 1 }) = (coef_biot * (
			r11 * (cur.T2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), ND) + cur.r2 * cur.G2) + 
			(cur.r1 * cur.T2 + cur.r2 * cur.T1) * make_block_diagonal((face.c - cur.y1).transpose(), ND))).values;
	const auto& g1 = grad[face.cell_id1];
	//const auto& g2 = grad[face.cell_id2];
	g = g1;
	flux.f = grad_coef * g1.rhs;
	flux.f_biot.values = 0.0;
	flux_flow_biot.f.values = 0.0;
	flux_flow_biot.f_biot = biot_grad_coef * g1.rhs;

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	fill_n(std::begin(flux_flow_biot.a.values), flux_flow_biot.a.values.size(), 0.0);
	fill_n(std::begin(flux_flow_biot.a_biot.values), flux_flow_biot.a_biot.values.size(), 0.0);

	flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * g.mat).values;
	flux_flow_biot.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (biot_grad_coef * g.mat).values;

	// fault extras
	assert(fault.cell_id2 >= n_matrix);
	value_t sign = get_fault_sign(n, ref_contact_ids[fault.cell_id2 - n_matrix]);
	res1 = findInVector(g.stencil, fault.cell_id2);
	if (res1.first) { id = res1.second; }
	else { id = g.stencil.size(); g.stencil.push_back(fault.cell_id2); }
	// add gap
	flux.a(BLOCK_SIZE * id, { ND, ND }, { (size_t)flux.a.N, 1 }) += sign * T.values;
	biot_id = BLOCK_SIZE * id + flux_flow_biot.a_biot.N * ND;
	flux_flow_biot.a_biot(biot_id, { ND }, { 1 }) += -(sign * r11 * coef_biot * cur.T2).values;
	// fault pressure
	flux.a(BLOCK_SIZE * id + ND, { ND, 1 }, { (size_t)flux.a.N, 1 }) += (coef2 * (biots[face.cell_id1] - biots[face.cell_id2]) * n).values;
	flux_flow_biot.a_biot(biot_id + ND, { 1 }, { 1 }) += (r11 * cur.r2 * coef_biot * (biots[face.cell_id1] - biots[face.cell_id2]) * n).values;
	// add gap gradient
	const auto& frac_grad = grad[fault.cell_id2];
	frac_grad_coef(0, { ND, ND * ND }, { (size_t)grad_coef.N, 1 }) = -sign * (T * make_block_diagonal((face.c - cur.y2).transpose(), ND) + coef2 * cur.G2).values;
	frac_biot_grad_coef(ND * ND * BLOCK_SIZE, { ND * ND }, { 1 }) = (sign * r11 * coef_biot *
		(cur.T2 * make_block_diagonal((face.c - cur.y2).transpose(), ND) + cur.r2 * cur.G2)).values;
	for (st_id = 0; st_id < frac_grad.stencil.size(); st_id++)
	{
		res1 = findInVector(g.stencil, frac_grad.stencil[st_id]);
		if (res1.first) { id = res1.second; }
		else { id = g.stencil.size(); g.stencil.push_back(frac_grad.stencil[st_id]); }
		block.values = frac_grad.mat(BLOCK_SIZE * st_id, { (size_t)frac_grad.mat.M, BLOCK_SIZE }, { (size_t)frac_grad.mat.N, 1 });
		block(ND * ND * block.N, { ND, (size_t)block.N }, { (size_t)block.N, 1 }) = 0.0;			// no discotinuity in pressure
		flux.a(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += (frac_grad_coef * block).values;
		flux_flow_biot.a_biot(BLOCK_SIZE * id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux_flow_biot.a_biot.N, 1 }) += (frac_biot_grad_coef * block).values;
	}
	flux.f += frac_grad_coef * frac_grad.rhs;
	flux_flow_biot.f_biot += frac_biot_grad_coef * frac_grad.rhs;

	res1 = findInVector(g.stencil, face.cell_id1);
	if (res1.first) { id = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
	flux.a(BLOCK_SIZE * id, { ND, ND }, { (size_t)flux.a.N, 1 }) += T.values;
	biot_id = BLOCK_SIZE * id + flux_flow_biot.a_biot.N * ND;
	flux_flow_biot.a_biot(biot_id, { ND }, { 1 }) -= (r11 * coef_biot * cur.T2).values;

	res2 = findInVector(g.stencil, face.cell_id2);
	if (res2.first) { id = res2.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id2);	exit(-1); }
	flux.a(BLOCK_SIZE * id, { ND, ND }, { (size_t)flux.a.N, 1 }) -= T.values;
	biot_id = BLOCK_SIZE * id + flux_flow_biot.a_biot.N * ND;
	flux_flow_biot.a_biot(biot_id, { ND }, { 1 }) += (r11 * coef_biot * cur.T2).values;

	// no direct fluid flow to another side of the fault
	flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
	//flux.a_biot(ND * flux.a_biot.N, { (size_t)flux.a_biot.N }, { 1 }) = 0.0;
	flux.f(ND, 0) = 0.0;
	//flux.f_biot(ND, 0) = 0.0;

	flux.a_biot.values = 0.0;
	flux.f_biot.values = 0.0;

	flux.stencil = g.stencil;
	// only fluid mass flux to the fault
	flux_flow_biot.a_biot(0, { ND, (size_t)flux_flow_biot.a_biot.N }, { (size_t)flux_flow_biot.a_biot.N, 1 }) = 0.0;
	flux_flow_biot.f_biot(0, { ND, 1 }, { (size_t)flux_flow_biot.f_biot.N, 1 }) = 0.0;
	flux_flow_biot.stencil = g.stencil;
}
void pm_discretizer::calc_matrix_fault_flow_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_biot, index_t fault_id)
{
	Matrix grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), biot_flow_buf(BLOCK_SIZE, BLOCK_SIZE), biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), det(BLOCK_SIZE, BLOCK_SIZE), A1_tilde(BLOCK_SIZE, BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Matrix n(ND, 1), K1n(ND, 1), gam1(ND, 1), K2n(ND, 1), gam2(ND, 1);
	Matrix y1(ND, 1), y2(ND, 1);
	value_t r1, r2, Ap, lam1, lam2, sign, R1, R2;

	// fault connection for fluid flow
	const auto& fault = faces[face.cell_id1][fault_id];
	const int& cell_id1 = fault.cell_id1;
	const int& cell_id2 = fault.cell_id2;
	const auto& c1 = cell_centers[cell_id1];
	const auto& c2 = cell_centers[cell_id2];
	n = (face.n.transpose() * (c2 - c1)).values[0] > 0 ? face.n : -face.n;
	// Permeability decomposition
	K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
	K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
	lam1 = (n.transpose() * K1n)(0, 0);
	lam2 = (n.transpose() * K2n)(0, 0);
	gam1 = K1n - lam1 * n;
	gam2 = K2n - lam2 * n;
	r1 = (n.transpose() * (face.c - c1))(0, 0) - frac_apers[cell_id2 - n_matrix] / 2.0;
	r2 = frac_apers[cell_id2 - n_matrix] / 2.0;//(n.transpose() * (c2 - face.c))(0, 0);
	y1 = c1 + r1 * n;	 y2 = c2 - r2 * n;
	A1_tilde(ND * BLOCK_SIZE, { ND }, { 1 }) = (biots[face.cell_id1] * n).values;
	// transversal term
	const auto& g1 = grad[face.cell_id1];
	//const auto& g2 = grad[face.cell_id2];
	R1 = (1.0 / visc * grav_vec * K1n).values[0];
	R2 = (1.0 / visc * grav_vec * K2n).values[0];
	g = g1;// merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat);
	std::fill_n(&grad_coef.values[0], grad_coef.values.size(), 0.0);
	std::fill_n(&biot_grad_coef.values[0], biot_grad_coef.values.size(), 0.0);
	grad_coef(ND * grad_coef.N + ND * ND, { ND }, { 1 }) = -((lam1 * lam2 * (y1 - y2) +
		lam1 * r2 * gam2 + lam2 * r1 * gam1) / (r2 * lam1 + r1 * lam2)).values;

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.f.values), flux.f.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	fill_n(std::begin(flux.f_biot.values), flux.f_biot.values.size(), 0.0);

	fill_n(std::begin(flux_biot.a.values), flux_biot.a.values.size(), 0.0);
	fill_n(std::begin(flux_biot.f.values), flux_biot.f.values.size(), 0.0);
	fill_n(std::begin(flux_biot.a_biot.values), flux_biot.a_biot.values.size(), 0.0);
	fill_n(std::begin(flux_biot.f_biot.values), flux_biot.f_biot.values.size(), 0.0);

	// biot for stress
	const auto& cur = inner[face.cell_id1][face.face_id1];
	det = cur.r1 * cur.Q2 + cur.r2 * cur.Q1;
	bool res = det.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}
	biot_flow_buf = cur.r1 * A1_tilde * det;
	biot_grad_coef = biot_flow_buf * (cur.r2 * (cur.Th2 - cur.Th1) +
		cur.Q2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE)) +
		A1_tilde * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);

	//flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * g.mat).values;
	//flux_biot.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux_biot.a_biot.N, 1 }) = (biot_grad_coef * g.mat).values;
	//flux_biot.f_biot = biot_grad_coef * g.rhs + biot_flow_buf * cur.r2 * (cur.R2 - cur.R1);
	//flux_biot.f_biot(ND, 0) = 0.0;

	// harmonic term
	res1 = findInVector(g.stencil, cell_id1);
	if (res1.first) { id = res1.second; }
	else { id = g.stencil.size(); g.stencil.push_back(cell_id1); }
	flux.a(ND, BLOCK_SIZE * id + ND) += lam1 * lam2 / (r1 * lam2 + r2 * lam1);
	//flux_biot.a_biot(BLOCK_SIZE * id, { (size_t)flux_biot.a_biot.M, BLOCK_SIZE }, { (size_t)flux_biot.a_biot.N, 1 }) += (A1_tilde - biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;

	res2 = findInVector(g.stencil, cell_id2);
	if (res2.first) { id = res2.second; }
	else { id = g.stencil.size(); g.stencil.push_back(cell_id2); }
	flux.a(ND, BLOCK_SIZE * id + ND) += -lam1 * lam2 / (r1 * lam2 + r2 * lam1);
	//flux_biot.a_biot(BLOCK_SIZE * id, { (size_t)flux_biot.a_biot.M, BLOCK_SIZE }, { (size_t)flux_biot.a_biot.N, 1 }) += (biot_flow_buf * (cur.Q2 + cur.r2 * cur.A2)).values;
	flux_biot.a_biot(BLOCK_SIZE * id + ND, { ND, 1 }, { (size_t)flux_biot.a_biot.N, 1 }) += (biots[cell_id2] * n).values;

	flux.f(ND, 0) += r2 * lam1 / (r2 * lam1 + r1 * lam2) * R2 + r1 * lam2 / (r2 * lam1 + r1 * lam2) * R1;

	flux_biot.a_biot(ND * (size_t)g.mat.N, { (size_t)g.mat.N }, { 1 }) = 0.0;
	flux.stencil = g.stencil;
	flux_biot.stencil = g.stencil;

	if (fault.is_impermeable)
	{
		flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
		flux.f(ND, 0) = 0.0;
	}
}
void pm_discretizer::calc_matrix_flux(value_t dt, const Face& face, Approximation& flux, Approximation& flux_th_cond, Approximation& face_unknown)
{
	Matrix det(BLOCK_SIZE, BLOCK_SIZE), coef1(BLOCK_SIZE, BLOCK_SIZE), coef2(BLOCK_SIZE, BLOCK_SIZE), grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE),
		biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), face_unknown_coef(BLOCK_SIZE, ND * BLOCK_SIZE), A1_tilde(BLOCK_SIZE, BLOCK_SIZE), biot_flow_buf(BLOCK_SIZE, BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Matrix n(ND, 1), P(ND, ND);
	bool res;

	n = (face.n.transpose() * (cell_centers[face.cell_id2] - cell_centers[face.cell_id1])).values[0] > 0 ? face.n : -face.n;
	P = I3 - outer_product(n, n.transpose());
	const auto& cur = inner[face.cell_id1][face.face_id1];
	det = cur.r1 * cur.Q2 + cur.r2 * cur.Q1;
	res = det.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}
	coef1 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A1);
	coef2 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A2);
	grad_coef = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.r2 * (cur.Th2 - cur.Th1) +
		cur.Q2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE)) + cur.Th1 -
		cur.A1 * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);

	const auto& g1 = grad[face.cell_id1];
	const auto& g2 = grad[face.cell_id2];
	// Biot term for flow
	A1_tilde(ND * BLOCK_SIZE, { ND }, { 1 }) = (biots[face.cell_id1] * n).values;
	A1_tilde(ND, { ND, 1 }, { BLOCK_SIZE, 1 }) = (biots[face.cell_id1] * n).values;

	biot_flow_buf = cur.r1 * A1_tilde * det;
	face_unknown_coef = cur.r1 * det * (cur.r2 * (cur.Th2 - cur.Th1) +
		cur.Q2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE)) +
		make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);
	biot_grad_coef = A1_tilde * face_unknown_coef;

	g = merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat);
	flux.f = grad_coef * (g1.rhs + g2.rhs) / 2.0 + cur.R1 + cur.r2 * (cur.Q1 - cur.r1 * cur.A1) * det * (cur.R2 - cur.R1);
	flux.f_biot = biot_grad_coef * (g1.rhs + g2.rhs) / 2.0 + biot_flow_buf * cur.r2 * (cur.R2 - cur.R1);
	// face unknowns
	face_unknown.f = face_unknown_coef * (g1.rhs + g2.rhs) / 2.0 + cur.r1 * det * cur.r2 * (cur.R2 - cur.R1);

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * g.mat).values;
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) = (biot_grad_coef * g.mat).values;
	// face unknowns
	fill_n(std::begin(face_unknown.a.values), face_unknown.a.values.size(), 0.0);
	face_unknown.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)face_unknown.a.N, 1 }) = (face_unknown_coef * g.mat).values;

	res1 = findInVector(g.stencil, face.cell_id1);
	if (res1.first) { id1 = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
	flux.a(BLOCK_SIZE * id1, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += -coef1.values;
	flux.a_biot(BLOCK_SIZE * id1, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += (A1_tilde - biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;
	// face unknowns
	face_unknown.a(BLOCK_SIZE * id1, { (size_t)face_unknown.a.M, BLOCK_SIZE }, { (size_t)face_unknown.a.N, 1 }) += (I4 - cur.r1 * det * (cur.Q2 + cur.r2 * cur.A1)).values;

	res2 = findInVector(g.stencil, face.cell_id2);
	if (res2.first) { id2 = res2.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id2);	exit(-1); }
	flux.a(BLOCK_SIZE * id2, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += coef2.values;
	flux.a_biot(BLOCK_SIZE * id2, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += (biot_flow_buf * (cur.Q2 + cur.r2 * cur.A2)).values;
	// face unknowns
	face_unknown.a(BLOCK_SIZE * id2, { (size_t)face_unknown.a.M, BLOCK_SIZE }, { (size_t)face_unknown.a.N, 1 }) += (cur.r1 * det * (cur.Q2 + cur.r2 * cur.A2)).values;

	flux.stencil = g.stencil;
	// face unknowns
	face_unknown.stencil = flux.stencil;

	if (face.is_impermeable)
	{
		flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
		flux.f(ND, 0) = 0.0;
	}

	if (ASSEMBLE_HEAT_CONDUCTION)
	{
		value_t lam1, lam2, lam_av;
		Matrix gam1(ND, 1), gam2(ND, 1);
		fill_n(std::begin(flux_th_cond.a.values), flux_th_cond.a.values.size(), 0.0);

		lam1 = (n.transpose() * diffs[face.cell_id1] * n).values[0];		gam1 = diffs[face.cell_id1] * n - lam1 * n;
		lam2 = (n.transpose() * diffs[face.cell_id2] * n).values[0];		gam2 = diffs[face.cell_id2] * n - lam2 * n;
		lam_av = lam1 * lam2 / (cur.r1 * lam2 + cur.r2 * lam1);
		
		const auto& g1d = grad_d[face.cell_id1];
		const auto& g2d = grad_d[face.cell_id2];
		g = merge_stencils(g1.stencil, 0.5 * g1d.mat, g2.stencil, 0.5 * g2d.mat);

		flux_th_cond.a(0, { 1, (size_t)g.mat.N }, { (size_t)flux_th_cond.a.N, 1 }) = 
		(((lam1 * lam2 * (cur.y1 - cur.y2) + lam1 * cur.r2 * gam2 + lam2 * cur.r1 * gam1) / (cur.r1 * lam2 + cur.r2 * lam1)).transpose() * g.mat).values;
		flux_th_cond.a(0, id1) -= lam_av;
		flux_th_cond.a(0, id2) += lam_av;
	}
}
void pm_discretizer::calc_matrix_flux_stabilized(value_t dt, const Face& face, Approximation& flux)
{
	Matrix det(BLOCK_SIZE, BLOCK_SIZE), det_ad(BLOCK_SIZE, BLOCK_SIZE), coef1(BLOCK_SIZE, BLOCK_SIZE), coef2(BLOCK_SIZE, BLOCK_SIZE), 
		coef1_ad(BLOCK_SIZE, BLOCK_SIZE), coef2_ad(BLOCK_SIZE, BLOCK_SIZE),
		grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE),	biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE),
		biot_flow_buf(BLOCK_SIZE, BLOCK_SIZE), M1p(BLOCK_SIZE, BLOCK_SIZE), M1m(BLOCK_SIZE, BLOCK_SIZE), M2p(BLOCK_SIZE, BLOCK_SIZE), M2m(BLOCK_SIZE, BLOCK_SIZE),
		b1n(ND, 1), b2n(ND, 1), biot_flow_advective(BLOCK_SIZE, BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Matrix n(ND, 1), P(ND, ND);
	value_t xi1, xi2;
	bool res;

	n = (face.n.transpose() * (cell_centers[face.cell_id2] - cell_centers[face.cell_id1])).values[0] > 0 ? face.n : -face.n;
	P = I3 - outer_product(n, n.transpose());
	const auto& cur = inner[face.cell_id1][face.face_id1];

	// Biot term for flow
	b1n = biots[face.cell_id1] * n;
	xi1 = sqrt((b1n.transpose() * b1n).values[0]);
	M1p(ND * BLOCK_SIZE, { ND }, { 1 }) = b1n.values / 2.0;
	M1p(ND, { ND, 1 }, { BLOCK_SIZE, 1 }) = b1n.values / 2.0;
	M1p(0, { ND, ND }, { BLOCK_SIZE, 1 }) = outer_product(b1n, b1n.transpose()).values / xi1 / 2.0;
	M1p(ND, ND) = xi1 / 2.0;
	M1m.values = M1p.values;
	M1m(0, { ND, ND }, { BLOCK_SIZE, 1 }) = -outer_product(b1n, b1n.transpose()).values / xi1 / 2.0;
	M1m(ND, ND) = -xi1 / 2.0;
	
	//M1m.values *= 2.0;
	//fill_n(std::begin(M1p.values), M1p.values.size(), 0.0);

	b2n = biots[face.cell_id2] * n;
	xi2 = sqrt((b2n.transpose() * b2n).values[0]);
	M2p(ND * BLOCK_SIZE, { ND }, { 1 }) = b2n.values / 2.0;
	M2p(ND, { ND, 1 }, { BLOCK_SIZE, 1 }) = b2n.values / 2.0;
	M2p(0, { ND, ND }, { BLOCK_SIZE, 1 }) = outer_product(b2n, b2n.transpose()).values / xi2 / 2.0;
	M2p(ND, ND) = xi2 / 2.0;
	M2m.values = M2p.values;
	M2m(0, { ND, ND }, { BLOCK_SIZE, 1 }) = -outer_product(b2n, b2n.transpose()).values / xi2 / 2.0;
	M2m(ND, ND) = -xi2 / 2.0;
	
	//M2p.values *= 2.0;
	//fill_n(std::begin(M2m.values), M2m.values.size(), 0.0);

	std::swap(M1m.values, M1p.values);
	std::swap(M2m.values, M2p.values);

	det = cur.r1 * cur.Q2 + cur.r2 * cur.Q1;
	det_ad = det + cur.r1 * cur.r2 * (M1p - M2m);
	res = det.inv();
	if (!res) { cout << "Inversion failed!\n";	exit(-1); }
	res = det_ad.inv();
	if (!res) { cout << "Inversion failed!\n";	exit(-1); }

	coef1 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A1);
	coef2 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A2);
	coef1_ad = cur.r2 * (cur.Q1 - cur.r1 * M1m);
	coef2_ad = cur.r1 * (cur.Q2 + cur.r2 * M2p);
	grad_coef = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.r2 * (cur.Th2 - cur.Th1) +
		cur.Q2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE)) + cur.Th1 -
		cur.A1 * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);

	const auto& g1 = grad[face.cell_id1];
	const auto& g2 = grad[face.cell_id2];

	biot_flow_buf = M1p * det_ad;
	biot_grad_coef = biot_flow_buf * (cur.r1 * cur.r2 * (cur.Th2 - cur.Th1) +
		coef1_ad * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE) +
		coef2_ad * make_block_diagonal((face.c - cur.y2).transpose(), BLOCK_SIZE)) + 
		M1m * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);

	g = merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat);
	flux.f = grad_coef * (g1.rhs + g2.rhs) / 2.0 + cur.R1 + cur.r2 * (cur.Q1 - cur.r1 * cur.A1) * det * (cur.R2 - cur.R1);
	flux.f_biot = biot_grad_coef * (g1.rhs + g2.rhs) / 2.0 + biot_flow_buf * cur.r1 * cur.r2 * (cur.R2 - cur.R1);

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * g.mat).values;
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) = (biot_grad_coef * g.mat).values;

	g = merge_stencils(g.stencil, pre_merged_grad[BLOCK_SIZE], g1.stencil, g1.mat);
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) += ((I4 - cur.r1 * cur.r2 * biot_flow_buf) * cur.r1 * M1m *
																make_block_diagonal(n.transpose(), BLOCK_SIZE) * g.mat).values;
	flux.f_biot += (I4 - cur.r1 * cur.r2 * biot_flow_buf) * cur.r1 * M1m * make_block_diagonal(n.transpose(), BLOCK_SIZE) * g.rhs;
	
	g = merge_stencils(g.stencil, pre_merged_grad[BLOCK_SIZE], g2.stencil, g2.mat);
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) += (-cur.r1 * cur.r2 * biot_flow_buf * cur.r2 * M2p *
																make_block_diagonal(n.transpose(), BLOCK_SIZE) * g.mat).values;
	flux.f_biot += -cur.r1 * cur.r2 * biot_flow_buf * cur.r2 * M2p * make_block_diagonal(n.transpose(), BLOCK_SIZE) * g.rhs;

	res1 = findInVector(g.stencil, face.cell_id1);
	if (res1.first) { id = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
	flux.a(BLOCK_SIZE * id, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += -coef1.values;
	flux.a_biot(BLOCK_SIZE * id, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += (biot_flow_buf * coef1_ad + M1m).values;

	res2 = findInVector(g.stencil, face.cell_id2);
	if (res2.first) { id = res2.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id2);	exit(-1); }
	flux.a(BLOCK_SIZE * id, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += coef2.values;
	flux.a_biot(BLOCK_SIZE * id, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += (biot_flow_buf * coef2_ad).values;

	flux.stencil = g.stencil;

	if (face.is_impermeable)
	{
		flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
		flux.f(ND, 0) = 0.0;
	}
}
void pm_discretizer::calc_matrix_flux_stabilized_new(value_t dt, const Face& face, Approximation& flux)
{
	Matrix det(BLOCK_SIZE, BLOCK_SIZE), det_stab(BLOCK_SIZE, BLOCK_SIZE), coef1(BLOCK_SIZE, BLOCK_SIZE), coef2(BLOCK_SIZE, BLOCK_SIZE), 
		grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), grad_coef1(BLOCK_SIZE, ND * BLOCK_SIZE), grad_coef2(BLOCK_SIZE, ND * BLOCK_SIZE),
		coef_stab1(BLOCK_SIZE, BLOCK_SIZE), coef_stab2(BLOCK_SIZE, BLOCK_SIZE),
		M1(BLOCK_SIZE, BLOCK_SIZE), M2(BLOCK_SIZE, BLOCK_SIZE), biot_flow_buf(BLOCK_SIZE, BLOCK_SIZE), nblock_t(BLOCK_SIZE, BLOCK_SIZE * ND),
		biot_grad_coef1(BLOCK_SIZE, BLOCK_SIZE * ND), biot_grad_coef2(BLOCK_SIZE, BLOCK_SIZE * ND), R1(BLOCK_SIZE, BLOCK_SIZE), R2(BLOCK_SIZE, BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Gradients g1g;	g1g.stencil.reserve(MAX_STENCIL);	g1g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g1g.rhs.values.resize(BLOCK_SIZE * ND);
	Gradients g2g;	g2g.stencil.reserve(MAX_STENCIL);	g2g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g2g.rhs.values.resize(BLOCK_SIZE * ND);
	Matrix n(ND, 1), P(ND, ND);
	std::valarray<value_t> eigv1(BLOCK_SIZE), eigvec1(BLOCK_SIZE), eigv2(BLOCK_SIZE), eigvec2(BLOCK_SIZE);
	bool res;
	Matrix B1n(ND, 1), B2n(ND, 1), S1(BLOCK_SIZE, BLOCK_SIZE), S2(BLOCK_SIZE, BLOCK_SIZE);
	value_t alpha_min_stab1, alpha_min_stab2;

	n = (face.n.transpose() * (cell_centers[face.cell_id2] - cell_centers[face.cell_id1])).values[0] > 0 ? face.n : -face.n;
	nblock_t = make_block_diagonal(face.n.transpose(), BLOCK_SIZE);
	P = I3 - outer_product(n, n.transpose());
	// Biot terms
	B1n = biots[face.cell_id1] * n;
	B2n = biots[face.cell_id2] * n;
	M1(ND * BLOCK_SIZE, { ND }, { 1 }) = B1n.values;
	M1(ND, { ND, 1 }, { BLOCK_SIZE, 1 }) = B1n.values;
	M2(ND * BLOCK_SIZE, { ND }, { 1 }) = B2n.values;
	M2(ND, { ND, 1 }, { BLOCK_SIZE, 1 }) = B2n.values;

	const auto& cur = inner[face.cell_id1][face.face_id1];

	// stabilization parameters
	alpha_min_stab1 = (sqrt((cur.k_stab1 / visc - cur.c_stab1) * (cur.k_stab1 / visc - cur.c_stab1) + 4 * cur.beta_stab1 * cur.beta_stab1 / dt) - (cur.k_stab1 / visc + cur.c_stab1)) / (2 * cur.beta_stab1);
	alpha_min_stab2 = (sqrt((cur.k_stab2 / visc - cur.c_stab2) * (cur.k_stab2 / visc - cur.c_stab2) + 4 * cur.beta_stab2 * cur.beta_stab2 / dt) - (cur.k_stab2 / visc + cur.c_stab2)) / (2 * cur.beta_stab2);
	S1.values = std::max(alpha_min_stab1, min_alpha_stabilization) * cur.S1.values;
	S2.values = std::max(alpha_min_stab2, min_alpha_stabilization) * cur.S2.values;

	max_alpha = std::max(max_alpha, std::max(alpha_min_stab1, min_alpha_stabilization));
	//if (face.cell_id1 == 0 && face.cell_id2 == 1)
	//	printf("dt=%f\t%f\t%f\n", dt, std::max(alpha_min_stab1, min_alpha_stabilization), std::max(alpha_min_stab2, min_alpha_stabilization));

	det = cur.r1 * cur.Q2 + cur.r2 * cur.Q1;
	//det = cur.Q1 / cur.r1 + cur.Q2 / cur.r2 - S1 - S2;
	res = det.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}
	R1 = S1 - M1 - cur.Q1 / cur.r1;
	R2 = S2 + M2 - cur.Q2 / cur.r2;
	det_stab = R1 + R2;
	res = det_stab.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}

	coef1 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A1);
	coef2 = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.Q2 + cur.r2 * cur.A2);
	coef_stab1 = R2 * det_stab;
	coef_stab2 = R1 * det_stab;
	/*(S1 - cur.Q1 / cur.r1 - M1).eigen(eigv1, eigvec1);
	(S2 - cur.Q2 / cur.r2 + M2).eigen(eigv2, eigvec2);
	for (index_t i = 0; i < BLOCK_SIZE; i++)
	{
		assert(eigv1[i] >= 0.0);
		assert(eigv2[i] >= 0.0);
	}*/

	grad_coef = (cur.Q1 - cur.r1 * cur.A1) * det * (cur.r2 * (cur.Th2 - cur.Th1) +
		cur.Q2 * make_block_diagonal((cur.y1 - cur.y2).transpose(), BLOCK_SIZE)) + cur.Th1 -
		cur.A1 * make_block_diagonal((face.c - cur.y1).transpose(), BLOCK_SIZE);
	//grad_coef1 = coef_stab1 * (cur.Q1 * nblock_t + cur.Th1 - (cur.Q1 / cur.r1 - S1) * make_block_diagonal((face.c - cell_centers[face.cell_id1]).transpose(), BLOCK_SIZE));
	//grad_coef2 = coef_stab2 * (cur.Q2 * nblock_t + cur.Th2 - (cur.Q2 / cur.r2 - S2) * make_block_diagonal((cell_centers[face.cell_id2] - face.c).transpose(), BLOCK_SIZE));
	const auto& g1 = grad[face.cell_id1];
	const auto& g2 = grad[face.cell_id2];

	// biot_flow_buf = M1 * det_stab;
	biot_grad_coef1 = -R1 * det_stab * (cur.Q1 * nblock_t + cur.Th1) - R2 * det_stab * (cur.Q1 / cur.r1 - S1) * make_block_diagonal((face.c - cell_centers[face.cell_id1]).transpose(), BLOCK_SIZE);
	biot_grad_coef2 = R1 * det_stab * (cur.Q2 * nblock_t + cur.Th2 -
		(cur.Q2 / cur.r2 - S2) * make_block_diagonal((cell_centers[face.cell_id2] - face.c).transpose(), BLOCK_SIZE));

	g = merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat);
	flux.f = grad_coef * (g1.rhs + g2.rhs) / 2.0 + cur.R1 + cur.r2 * (cur.Q1 - cur.r1 * cur.A1) * det * (cur.R2 - cur.R1);
	//flux.f = (cur.Q1 * nblock_t + cur.Th1) * g1.rhs;//grad_coef1 * g1.rhs + grad_coef1 * g2.rhs + coef_stab1 * cur.R1 + coef_stab2 * cur.R2;
	flux.f_biot = biot_grad_coef2 * g2.rhs + biot_grad_coef1 * g1.rhs + R1 * det_stab * (cur.R2 - cur.R1);

	g1g = merge_stencils(g.stencil, pre_merged_grad[BLOCK_SIZE], g1.stencil, g1.mat);
	g2g = merge_stencils(g.stencil, pre_merged_grad[BLOCK_SIZE], g2.stencil, g2.mat);

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * g.mat).values;
	//flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = (grad_coef1 * g1g.mat + grad_coef2 * g2g.mat).values;
	//flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a.N, 1 }) = ((cur.Q1 * nblock_t + cur.Th1) * g1g.mat).values;
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) = (biot_grad_coef2 * g2g.mat + biot_grad_coef1 * g1g.mat).values;

	res1 = findInVector(g.stencil, face.cell_id1);
	if (res1.first) { id1 = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id1);	exit(-1); }
	flux.a(BLOCK_SIZE * id1, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += -coef1.values;
	//flux.a(BLOCK_SIZE * id1, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += (coef_stab1 * (S1 - cur.Q1 / cur.r1)).values;
	flux.a_biot(BLOCK_SIZE * id1, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += (coef_stab1 * (S1 - cur.Q1 / cur.r1)).values;

	res2 = findInVector(g.stencil, face.cell_id2);
	if (res2.first) { id2 = res2.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", face.cell_id2);	exit(-1); }
	flux.a(BLOCK_SIZE * id2, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += coef2.values;
	//flux.a(BLOCK_SIZE * id2, { (size_t)flux.a.M, BLOCK_SIZE }, { (size_t)flux.a.N, 1 }) += -(coef_stab2 * (S2 - cur.Q2 / cur.r2)).values;
	flux.a_biot(BLOCK_SIZE * id2, { (size_t)flux.a_biot.M, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 }) += -(coef_stab2 * (S2 - cur.Q2 / cur.r2)).values;

	flux.stencil = g.stencil;

	if (face.is_impermeable)
	{
		flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
		flux.f(ND, 0) = 0.0;
	}
}
void pm_discretizer::calc_fault_matrix(value_t dt, const Face& face, Approximation& flux)
{
	Matrix n(ND, 1), K1n(ND, 1), K2n(ND, 1), gam1(ND, 1), gam2(ND, 1), y1(ND, 1), y2(ND, 1);
	Matrix grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	value_t lam1, lam2, r1, r2, R1, R2;

	const int& cell_id1 = face.cell_id1;
	const int& cell_id2 = face.cell_id2;
	const auto& c1 = cell_centers[cell_id1];
	const auto& c2 = cell_centers[cell_id2];
	n = (face.n.transpose() * (c2 - c1)).values[0] > 0 ? face.n : -face.n;
	// Permeability decomposition
	K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
	K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
	lam1 = (n.transpose() * K1n)(0, 0);
	lam2 = (n.transpose() * K2n)(0, 0);
	gam1 = K1n - lam1 * n;
	gam2 = K2n - lam2 * n;
	r1 = frac_apers[cell_id1 - n_matrix] / 2.0;
	r2 = (n.transpose() * (c2 - face.c))(0, 0) - frac_apers[cell_id1 - n_matrix] / 2.0;
	y1 = c1 + r1 * n;	 y2 = c2 - r2 * n;
	// transversal term
	const auto& g1 = grad[face.cell_id1];
	const auto& g2 = grad[face.cell_id2];
	R1 = (1.0 / visc * grav_vec * K1n).values[0];
	R2 = (1.0 / visc * grav_vec * K2n).values[0];
	g = g2;// merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat);
	std::fill_n(&grad_coef.values[0], grad_coef.values.size(), 0.0);
	std::fill_n(&biot_grad_coef.values[0], biot_grad_coef.values.size(), 0.0);
	grad_coef(ND * grad_coef.N + ND * ND, { ND }, { 1 }) = -((lam1 * lam2 * (y1 - y2) +
		lam1 * r2 * gam2 + lam2 * r1 * gam1) / (r2 * lam1 + r1 * lam2)).values;

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	fill_n(std::begin(flux.f_biot.values), flux.f_biot.values.size(), 0.0);
	fill_n(std::begin(flux.f.values), flux.f.values.size(), 0.0);
	//a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)a.N, 1 }) = (grad_coef * g.mat).values;
	//a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)a_biot.N, 1 }) = (biot_grad_coef * g.mat).values;
	//f = grad_coef * g2.rhs;//(g1.rhs + g2.rhs) / 2.0;
	flux.f(ND, 0) += r2 * lam1 / (r2 * lam1 + r1 * lam2) * R2 + r1 * lam2 / (r2 * lam1 + r1 * lam2) * R1;
	// harmonic term
	res1 = findInVector(g.stencil, cell_id1);
	if (res1.first) { id = res1.second; }
	else { id = g.stencil.size(); g.stencil.push_back(cell_id1); }
	flux.a(ND, BLOCK_SIZE * id + ND) += lam1 * lam2 / (r1 * lam2 + r2 * lam1);

	res2 = findInVector(g.stencil, cell_id2);
	if (res2.first) { id = res2.second; }
	else { id = g.stencil.size(); g.stencil.push_back(cell_id2); }
	flux.a(ND, BLOCK_SIZE * id + ND) += -lam1 * lam2 / (r1 * lam2 + r2 * lam1);

	flux.stencil = g.stencil;

	if (face.is_impermeable)
	{
		flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
		flux.f(ND, 0) = 0.0;
	}
}
void pm_discretizer::calc_fault_fault(value_t dt, const Face& face, Approximation& flux)
{
	Matrix n(ND, 1), K1n(ND, 1), K2n(ND, 1), gam1(ND, 1), gam2(ND, 1), y1(ND, 1), y2(ND, 1);
	Matrix grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE), biot_grad_coef(BLOCK_SIZE, ND * BLOCK_SIZE);
	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	value_t lam1, lam2, r1, r2, R1, R2;

	const int& cell_id1 = face.cell_id1;
	const int& cell_id2 = face.cell_id2;
	const auto& c1 = cell_centers[cell_id1];
	const auto& c2 = cell_centers[cell_id2];
	n = (face.n.transpose() * (c2 - c1)).values[0] > 0 ? face.n : -face.n;
	// Permeability decomposition
	K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
	K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
	lam1 = (n.transpose() * K1n)(0, 0);
	lam2 = (n.transpose() * K2n)(0, 0);
	gam1 = K1n - lam1 * n;
	gam2 = K2n - lam2 * n;
	r1 = (n.transpose() * (face.c - c1))(0, 0);
	r2 = (n.transpose() * (c2 - face.c))(0, 0);
	y1 = c1 + r1 * n;	 y2 = c2 - r2 * n;
	// transversal term
	const auto& g1 = grad[face.cell_id1];
	const auto& g2 = grad[face.cell_id2];
	R1 = (1.0 / visc * grav_vec * K1n).values[0];
	R2 = (1.0 / visc * grav_vec * K2n).values[0];
	g = merge_stencils(g1.stencil, 0.5 * g1.mat, g2.stencil, 0.5 * g2.mat);
	std::fill_n(&grad_coef.values[0], grad_coef.values.size(), 0.0);
	std::fill_n(&biot_grad_coef.values[0], biot_grad_coef.values.size(), 0.0);
	grad_coef(ND * grad_coef.N + ND * ND, { ND }, { 1 }) = -((lam1 * lam2 * (y1 - y2) +
		lam1 * r2 * gam2 + lam2 * r1 * gam1) / (r2 * lam1 + r1 * lam2)).values;

	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	std::fill_n(&flux.f.values[0], flux.f.values.size(), 0.0);
	std::fill_n(&flux.f_biot.values[0], flux.f_biot.values.size(), 0.0);
	//a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)a.N, 1 }) = (grad_coef * g.mat).values;
	//a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)a_biot.N, 1 }) = (biot_grad_coef * g.mat).values;
	//f = grad_coef * (g1.rhs + g2.rhs) / 2.0;
	//f(ND, 0) += r2 * lam1 / (r2 * lam1 + r1 * lam2) * R2 + r1 * lam2 / (r2 * lam1 + r1 * lam2) * R1;
	// harmonic term
	res1 = findInVector(g.stencil, cell_id1);
	if (res1.first) { id = res1.second; }
	else { id = g.stencil.size(); g.stencil.push_back(cell_id1); }
	flux.a(ND, BLOCK_SIZE * id + ND) += lam1 * lam2 / (r1 * lam2 + r2 * lam1);

	res2 = findInVector(g.stencil, cell_id2);
	if (res2.first) { id = res2.second; }
	else { id = g.stencil.size(); g.stencil.push_back(cell_id2); }
	flux.a(ND, BLOCK_SIZE * id + ND) += -lam1 * lam2 / (r1 * lam2 + r2 * lam1);

	flux.f(ND, 0) += r2 * lam1 / (r2 * lam1 + r1 * lam2) * R2 + r1 * lam2 / (r2 * lam1 + r1 * lam2) * R1;

	flux.a.values *= (r2 * frac_apers[cell_id1 - n_matrix] + r1 * frac_apers[cell_id2 - n_matrix]) / (r1 + r2) * face.area;
	flux.f.values *= (r2 * frac_apers[cell_id1 - n_matrix] + r1 * frac_apers[cell_id2 - n_matrix]) / (r1 + r2) * face.area;
	
	flux.stencil = g.stencil;

	if (face.is_impermeable)
	{
		flux.a(ND * flux.a.N, { (size_t)flux.a.N }, { 1 }) = 0.0;
		flux.f(ND, 0) = 0.0;
	}
}
void pm_discretizer::calc_all_fluxes_once(value_t dt)
{
	bool isStationary = dt == 0.0 ? true : false;
	if (isStationary) dt = 1.0;
	assert(grad.size() > 0);
	cell_m.clear();					cell_p.clear();
	stencil.clear();				offset.clear();
	tran.clear();					rhs.clear();
	tran_biot.clear();				rhs_biot.clear();
	tran_th_cond.clear();			tran_th_expn.clear();
	tran_face_unknown.clear();		rhs_face_unknown.clear();

	// Variables
	index_t fault_id, fault_id2;
	Matrix ths(ND, 1);
	max_alpha = -std::numeric_limits<value_t>::infinity();
	std::array<value_t,2> max_flow, mult_flow;

	// Fluxes for matrix cells
	offset.push_back(0);
	for (index_t cell_id = 0; cell_id < n_matrix; cell_id++)
	{
		const auto& vec_faces = faces[cell_id];
		size_t n_cur_faces = vec_faces.size();
		for (index_t face_id = 0; face_id < n_cur_faces; face_id++)
		{
			const Face& face = vec_faces[face_id];
			if (face.type == MAT)
			{
				// check if exists a fault connetion for this face or not
				fault_id = check_face_is_fault(face);
				if (fault_id < 0)
				{
					// matrix -> matrix
					auto& flux = fluxes[0];
					auto& flux_th_cond = fluxes_th_cond[0];
					auto& face_unknown = face_unknowns[0];
					if (scheme == APPLY_EIGEN_SPLITTING)			calc_matrix_flux_stabilized(dt, face, flux);
					else if (scheme == APPLY_EIGEN_SPLITTING_NEW) 	calc_matrix_flux_stabilized_new(dt, face, flux);
					else if (scheme == AVERAGE)						calc_avg_matrix_flux(dt, face, flux);
					else											calc_matrix_flux(dt, face, flux, flux_th_cond, face_unknown);
					flux.a.values *= face.area;
					flux.f.values *= face.area;
					flux.a_biot.values *= face.area;
					flux.f_biot.values *= face.area;
					flux_th_cond.a.values *= face.area;
					cell_m.push_back(face.cell_id1);
					cell_p.push_back(face.cell_id2);
					if (ASSEMBLE_HEAT_CONDUCTION)	write_trans_biot_therm_cond(flux.stencil, flux.a, flux.a_biot, flux_th_cond.a);
					else							write_trans_biot(flux.stencil, flux.a, flux.a_biot, face_unknown.a);
					offset.push_back(stencil.size());
					rhs.insert(std::end(rhs), std::begin(flux.f.values), std::end(flux.f.values));
					rhs_biot.insert(std::end(rhs_biot), std::begin(flux.f_biot.values), std::end(flux.f_biot.values));
					rhs_face_unknown.insert(std::end(rhs_face_unknown), std::begin(face_unknown.f.values), std::end(face_unknown.f.values));
				}
				else
				{
					//contact_mixing(dt, cell_id, fault_id, face);
					contact_mixing_new(dt, cell_id, fault_id, face);
				}
			}
			else if (face.type == BORDER)
			{
				// flux assembly
				auto& flux = fluxes[0];
				auto& flux_th_cond = fluxes_th_cond[0];
				auto& face_unknown = face_unknowns[0];
				calc_border_flux(dt, face, flux, flux_th_cond, face_unknown);
				// area multiplier
				flux.a.values *= face.area;
				flux.f.values *= face.area;
				flux.a_biot.values *= face.area;
				flux.f_biot.values *= face.area;
				flux_th_cond.a.values *= face.area;
				cell_m.push_back(face.cell_id1);
				cell_p.push_back(n_cells + face.face_id2);
				if (ASSEMBLE_HEAT_CONDUCTION)	write_trans_biot_therm_cond(flux.stencil, flux.a, flux.a_biot, flux_th_cond.a);
				else							write_trans_biot(flux.stencil, flux.a, flux.a_biot, face_unknown.a);
				offset.push_back(stencil.size());
				rhs.insert(std::end(rhs), std::begin(flux.f.values), std::end(flux.f.values));
				rhs_biot.insert(std::end(rhs_biot), std::begin(flux.f_biot.values), std::end(flux.f_biot.values));
				rhs_face_unknown.insert(std::end(rhs_face_unknown), std::begin(face_unknown.f.values), std::end(face_unknown.f.values));
			}

			if (th_expns.size())
			{
				ths = stfs[cell_id] * th_expns[cell_id];
				tran_th_expn.push_back(ths(0, 0) * face.n(0, 0) + ths(5, 0) * face.n(1, 0) + ths(4, 0) * face.n(2, 0));
				tran_th_expn.push_back(ths(5, 0) * face.n(0, 0) + ths(1, 0) * face.n(1, 0) + ths(3, 0) * face.n(2, 0));
				tran_th_expn.push_back(ths(4, 0) * face.n(0, 0) + ths(3, 0) * face.n(1, 0) + ths(2, 0) * face.n(2, 0));
			}
		}
	}

	// Fluid fluxes for fault cells
	for (index_t cell_id = 0; cell_id < n_fracs; cell_id++)
	{
		const auto& vec_faces = faces[n_matrix + cell_id];
		size_t n_cur_faces = vec_faces.size();
		for (index_t face_id = 0; face_id < n_cur_faces; face_id++)
		{
			const Face& face = vec_faces[face_id];
			if (face.type == FRAC_TO_MAT)
			{
				//contact_mixing(dt, cell_id, -1, face);
				contact_mixing_new(dt, cell_id, -1, face);
			}
			else if (face.type == FRAC)
			{
				// fracture -> fracture
				auto& flux = fluxes[0];
				calc_fault_fault(dt, face, flux);
				cell_m.push_back(face.cell_id1);
				cell_p.push_back(face.cell_id2);
				write_trans_biot(flux.stencil, flux.a, flux.a_biot);
				offset.push_back(stencil.size());
				rhs.insert(std::end(rhs), std::begin(flux.f.values), std::end(flux.f.values));
				rhs_biot.insert(std::end(rhs_biot), std::begin(flux.f_biot.values), std::end(flux.f_biot.values));
			}

			if (th_expns.size())
			{
				ths = stfs[cell_id] * th_expns[cell_id];
				tran_th_expn.push_back(ths(0, 0) * face.n(0, 0) + ths(5, 0) * face.n(1, 0) + ths(4, 0) * face.n(2, 0));
				tran_th_expn.push_back(ths(5, 0) * face.n(0, 0) + ths(1, 0) * face.n(1, 0) + ths(3, 0) * face.n(2, 0));
				tran_th_expn.push_back(ths(4, 0) * face.n(0, 0) + ths(3, 0) * face.n(1, 0) + ths(2, 0) * face.n(2, 0));
			}
		}
	}

	if (scheme == APPLY_EIGEN_SPLITTING_NEW)
	{
		max_alpha_in_domain.push_back(max_alpha);
		dt_max_alpha_in_domain.push_back(dt);
	}

	printf("Calculation of fluxes was done!\n");
}
void pm_discretizer::contact_mixing(value_t dt, index_t cell_id, index_t fault_id, const Face& face)
{
	std::array<value_t, 2> max_flow, mult_flow;
	index_t fault_id2;

	if (face.type == MAT)
	{
		const auto& vec_faces = faces[cell_id];

		// mixing for biot terms in flow
		mult_flow[0] = mult_flow[1] = 0.5;

		// original side
		auto& traction1 = fluxes[0];
		auto& flow_biot1 = fluxes[1];
		calc_contact_flux(dt, face, traction1, flow_biot1, fault_id);	// discontinuous displacements
		traction1.a.values *= face.area / 2.0;
		traction1.f.values *= face.area / 2.0;
		traction1.a_biot.values *= face.area / 2.0;
		traction1.f_biot.values *= face.area / 2.0;

		if (face.is_impermeable == 1)
		{
			flow_biot1.a_biot.values = 0.0;
			flow_biot1.f_biot.values = 0.0;
		}

		max_flow[0] = abs(flow_biot1.a_biot.values).max();
		if (max_flow[0] < EQUALITY_TOLERANCE) mult_flow[1] = 1.0; // if one equal to zero than choose another approximation

		auto& flow_main = fluxes[2];
		auto& traction_biot1 = fluxes[3];
		calc_matrix_fault_flow_flux(dt, face, flow_main, traction_biot1, fault_id);	// continuous pressure
		flow_main.a.values *= face.area;
		flow_main.f.values *= face.area;
		traction_biot1.a_biot.values *= face.area;
		traction_biot1.f_biot.values *= face.area;

		// opposite side
		const auto& face2 = faces[face.cell_id2][face.face_id2];
		fault_id2 = check_face_is_fault(face2);
		auto& traction2 = fluxes[4];
		auto& flow_biot2 = fluxes[5];
		calc_contact_flux(dt, face2, traction2, flow_biot2, fault_id2);	// discontinuous displacements
		traction2.a.values *= -face2.area / 2.0;
		traction2.f.values *= -face2.area / 2.0;
		traction2.a_biot.values *= -face2.area / 2.0;
		traction2.f_biot.values *= -face2.area / 2.0;

		if (face2.is_impermeable == 2)
		{
			flow_biot2.a_biot.values = 0.0;
			flow_biot2.f_biot.values = 0.0;
		}

		max_flow[1] = abs(flow_biot2.a_biot.values).max();
		if (max_flow[1] < EQUALITY_TOLERANCE) mult_flow[0] = 1.0; // if one equal to zero than choose another approximation

		flow_biot1.a_biot.values *= mult_flow[0] * face.area;
		flow_biot1.f_biot.values *= mult_flow[0] * face.area;
		flow_biot2.a_biot.values *= -mult_flow[1] * face.area;
		flow_biot2.f_biot.values *= -mult_flow[1] * face.area;

		// auto& traction_biot2 = fluxes[6];
		// auto& dummy_flux = fluxes[7];
		// calc_matrix_fault_flow_flux(dt, face2, dummy_flux, traction_biot2, fault_id2);	// continuous pressure
		// traction_biot2.a_biot.values *= -face2.area / 2.0;
		// traction_biot2.f_biot.values *= -face2.area / 2.0;

		// merge
		auto& mixed_traction_main = merge_approximations(traction1, traction2, 0);
		auto& mixed_traction_biot = traction_biot1;// merge_approximations(traction_biot1, traction_biot2, 1);
		auto& mixed_traction = merge_approximations(mixed_traction_main, mixed_traction_biot, 2);

		auto& mixed_flow_biot = merge_approximations(flow_biot1, flow_biot2, 3); //flow_biot1;
		auto& mixed_flow = merge_approximations(flow_main, mixed_flow_biot, 4);

		// matrix -> matrix
		cell_m.push_back(face.cell_id1);
		cell_p.push_back(face.cell_id2);
		write_trans_biot(mixed_traction.stencil, mixed_traction.a, mixed_traction.a_biot);
		offset.push_back(stencil.size());
		rhs.insert(std::end(rhs), std::begin(mixed_traction.f.values), std::end(mixed_traction.f.values));
		rhs_biot.insert(std::end(rhs_biot), std::begin(mixed_traction.f_biot.values), std::end(mixed_traction.f_biot.values));

		// matrix -> fault
		const auto& fault = vec_faces[fault_id];
		cell_m.push_back(fault.cell_id1);
		cell_p.push_back(fault.cell_id2);
		write_trans_biot(mixed_flow.stencil, mixed_flow.a, mixed_flow.a_biot);
		offset.push_back(stencil.size());
		rhs.insert(std::end(rhs), std::begin(mixed_flow.f.values), std::end(mixed_flow.f.values));
		rhs_biot.insert(std::end(rhs_biot), std::begin(mixed_flow.f_biot.values), std::end(mixed_flow.f_biot.values));
	}
	else if (face.type == FRAC_TO_MAT)
	{
		const auto& vec_faces = faces[n_matrix + cell_id];

		// fracture -> matrix
		auto& flux = fluxes[0];
		calc_fault_matrix(dt, face, flux);
		flux.a.values *= face.area;
		flux.f.values *= face.area;

		// mass fluid flux due to biot
		auto& dummy_flux = fluxes[1];
		auto& flux_flow_biot = fluxes[2];
		const auto& op_face = faces[face.cell_id2][face.face_id2];
		fault_id = check_face_is_fault(op_face);
		assert(fault_id >= 0);
		calc_contact_flux(dt, op_face, dummy_flux, flux_flow_biot, fault_id);
		flux_flow_biot.a_biot.values *= -face.area;
		flux_flow_biot.f_biot.values *= -face.area;

		const auto& merged_flux = merge_approximations(flux, flux_flow_biot, 0);

		cell_m.push_back(face.cell_id1);
		cell_p.push_back(face.cell_id2);
		write_trans_biot(merged_flux.stencil, merged_flux.a, merged_flux.a_biot);
		offset.push_back(stencil.size());
		rhs.insert(std::end(rhs), std::begin(merged_flux.f.values), std::end(merged_flux.f.values));
		rhs_biot.insert(std::end(rhs_biot), std::begin(merged_flux.f_biot.values), std::end(merged_flux.f_biot.values));
	}
}
void pm_discretizer::contact_mixing_new(value_t dt, index_t cell_id, index_t fault_id, const Face& face)
{
	std::array<value_t, 2> max_flow, mult_flow;
	index_t fault_id2;

	if (face.type == MAT)
	{
		const auto& vec_faces = faces[cell_id];

		// mixing for biot terms in flow
		mult_flow[0] = mult_flow[1] = 0.5;

		// original side
		auto& traction1 = fluxes[0];
		auto& flow_biot1 = fluxes[1];
		calc_contact_flux_new(dt, face, traction1, flow_biot1, fault_id);	// discontinuous displacements
		traction1.a.values *= face.area / 2.0;
		traction1.f.values *= face.area / 2.0;
		traction1.a_biot.values *= face.area / 2.0;
		traction1.f_biot.values *= face.area / 2.0;

		if (face.is_impermeable == 1)
		{
			flow_biot1.a_biot.values = 0.0;
			flow_biot1.f_biot.values = 0.0;
		}

		max_flow[0] = abs(flow_biot1.a_biot.values).max();
		if (max_flow[0] < EQUALITY_TOLERANCE) mult_flow[1] = 1.0; // if one equal to zero than choose another approximation

		auto& flow_main = fluxes[2];
		auto& traction_biot1 = fluxes[3];
		calc_matrix_fault_flow_flux(dt, face, flow_main, traction_biot1, fault_id);	// continuous pressure
		flow_main.a.values *= face.area;
		flow_main.f.values *= face.area;
		traction_biot1.a_biot.values *= face.area;
		traction_biot1.f_biot.values *= face.area;

		// opposite side
		const auto& face2 = faces[face.cell_id2][face.face_id2];
		fault_id2 = check_face_is_fault(face2);
		auto& traction2 = fluxes[4];
		auto& flow_biot2 = fluxes[5];
		calc_contact_flux_new(dt, face2, traction2, flow_biot2, fault_id2);	// discontinuous displacements
		traction2.a.values *= -face2.area / 2.0;
		traction2.f.values *= -face2.area / 2.0;
		traction2.a_biot.values *= -face2.area / 2.0;
		traction2.f_biot.values *= -face2.area / 2.0;

		if (face2.is_impermeable == 2)
		{
			flow_biot2.a_biot.values = 0.0;
			flow_biot2.f_biot.values = 0.0;
		}

		max_flow[1] = abs(flow_biot2.a_biot.values).max();
		if (max_flow[1] < EQUALITY_TOLERANCE) mult_flow[0] = 1.0; // if one equal to zero than choose another approximation

		flow_biot1.a_biot.values *= face.area;
		//flow_biot1.f_biot.values *= mult_flow[0] * face.area;
		flow_biot2.a_biot.values *= -mult_flow[1] * face.area;
		flow_biot2.f_biot.values *= -mult_flow[1] * face.area;

		// auto& traction_biot2 = fluxes[6];
		// auto& dummy_flux = fluxes[7];
		// calc_matrix_fault_flow_flux(dt, face2, dummy_flux, traction_biot2, fault_id2);	// continuous pressure
		// traction_biot2.a_biot.values *= -face2.area / 2.0;
		// traction_biot2.f_biot.values *= -face2.area / 2.0;

		// merge
		auto& mixed_traction_main = merge_approximations(traction1, traction2, 0);
		auto& mixed_traction_biot = traction_biot1;// merge_approximations(traction_biot1, traction_biot2, 1);
		auto& mixed_traction = merge_approximations(mixed_traction_main, mixed_traction_biot, 2);

		auto& mixed_flow_biot = flow_biot1;// merge_approximations(flow_biot1, flow_biot2, 3); // flow_biot1;
		auto& mixed_flow = merge_approximations(flow_main, mixed_flow_biot, 4);

		// matrix -> matrix
		cell_m.push_back(face.cell_id1);
		cell_p.push_back(face.cell_id2);
		write_trans_biot(mixed_traction.stencil, mixed_traction.a, mixed_traction.a_biot);
		offset.push_back(stencil.size());
		rhs.insert(std::end(rhs), std::begin(mixed_traction.f.values), std::end(mixed_traction.f.values));
		rhs_biot.insert(std::end(rhs_biot), std::begin(mixed_traction.f_biot.values), std::end(mixed_traction.f_biot.values));

		// matrix -> fault
		const auto& fault = vec_faces[fault_id];
		cell_m.push_back(fault.cell_id1);
		cell_p.push_back(fault.cell_id2);
		write_trans_biot(mixed_flow.stencil, mixed_flow.a, mixed_flow.a_biot);
		offset.push_back(stencil.size());
		rhs.insert(std::end(rhs), std::begin(mixed_flow.f.values), std::end(mixed_flow.f.values));
		rhs_biot.insert(std::end(rhs_biot), std::begin(mixed_flow.f_biot.values), std::end(mixed_flow.f_biot.values));
	}
	else if (face.type == FRAC_TO_MAT)
	{
		const auto& vec_faces = faces[n_matrix + cell_id];

		// fracture -> matrix
		auto& flux = fluxes[0];
		calc_fault_matrix(dt, face, flux);
		flux.a.values *= face.area;
		flux.f.values *= face.area;

		// mass fluid flux due to biot
		auto& dummy_flux = fluxes[1];
		auto& flux_flow_biot = fluxes[2];
		const auto& op_face = faces[face.cell_id2][face.face_id2];
		fault_id = check_face_is_fault(op_face);
		assert(fault_id >= 0);
		calc_contact_flux_new(dt, op_face, dummy_flux, flux_flow_biot, fault_id);
		flux_flow_biot.a_biot.values *= -face.area;
		flux_flow_biot.f_biot.values *= -face.area;

		const auto& merged_flux = merge_approximations(flux, flux_flow_biot, 0);

		cell_m.push_back(face.cell_id1);
		cell_p.push_back(face.cell_id2);
		write_trans_biot(merged_flux.stencil, merged_flux.a, merged_flux.a_biot);
		offset.push_back(stencil.size());
		rhs.insert(std::end(rhs), std::begin(merged_flux.f.values), std::end(merged_flux.f.values));
		rhs_biot.insert(std::end(rhs_biot), std::begin(merged_flux.f_biot.values), std::end(merged_flux.f_biot.values));
	}
}

void pm_discretizer::calc_avg_matrix_flux(value_t dt, const Face& face, Approximation& flux)
{
	Matrix n(ND, 1), P(ND, ND), K1n(ND, 1), K2n(ND, 1), B1n(ND, 1), B2n(ND, 1);
	Matrix C1(ND * ND, ND * ND), C2(ND * ND, ND * ND), nblock(ND * ND, ND), nblock_t(ND, ND * ND), tblock(ND * ND, ND * ND);
	Matrix w1(BLOCK_SIZE, BLOCK_SIZE), w2(BLOCK_SIZE, BLOCK_SIZE);
	Matrix grad_coef1(BLOCK_SIZE, ND * BLOCK_SIZE), biot_grad_coef1(BLOCK_SIZE, ND * BLOCK_SIZE),
			grad_coef2(BLOCK_SIZE, ND * BLOCK_SIZE), biot_grad_coef2(BLOCK_SIZE, ND * BLOCK_SIZE);

	Gradients g;	g.stencil.reserve(MAX_STENCIL);	g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g.rhs.values.resize(BLOCK_SIZE * ND);
	Gradients g1g;	g1g.stencil.reserve(MAX_STENCIL);	g1g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g1g.rhs.values.resize(BLOCK_SIZE * ND);
	Gradients g2g;	g2g.stencil.reserve(MAX_STENCIL);	g2g.mat.values.resize(BLOCK_SIZE * ND * MAX_STENCIL * BLOCK_SIZE);	g2g.rhs.values.resize(BLOCK_SIZE * ND);

	const int& cell_id1 = face.cell_id1;
	const int& cell_id2 = face.cell_id2;
	const auto& c1 = cell_centers[cell_id1];
	const auto& c2 = cell_centers[cell_id2];
	n = (face.n.transpose() * (c2 - c1)).values[0] > 0 ? face.n : -face.n;
	P = I3 - outer_product(n, n.transpose());
	// Permeability
	K1n = pm_discretizer::darcy_constant * perms[cell_id1] * n;
	K2n = pm_discretizer::darcy_constant * perms[cell_id2] * n;
	// Stiffness
	C1 = W * stfs[cell_id1] * W.transpose();
	C2 = W * stfs[cell_id2] * W.transpose();
	nblock = make_block_diagonal(n, ND);
	nblock_t = make_block_diagonal(n.transpose(), ND);
	tblock = make_block_diagonal(P, ND);
	// Biot
	B1n = biots[cell_id1] * n;
	B2n = biots[cell_id2] * n;

	// Coefficients
	w1 = I4 / 2.0;
	grad_coef1(0, { ND, ND * ND }, { (size_t)grad_coef1.N, 1 }) = -(nblock_t * C1).values;
	grad_coef1(ND * ND * (BLOCK_SIZE + 1), { ND }, { 1 }) = -K1n.values / visc;
	biot_grad_coef1(ND * ND, { ND, ND }, { (size_t)grad_coef1.N, 1 }) = outer_product(B1n, (face.c - c1).transpose()).values;
	biot_grad_coef1(ND * ND, { ND, ND }, { (size_t)grad_coef1.N, 1 }) = (B1n.transpose() * make_block_diagonal((face.c - c1).transpose(), ND)).values;

	w2 = I4 / 2.0;
	grad_coef2(0, { ND, ND * ND }, { (size_t)grad_coef2.N, 1 }) = -(nblock_t * C2).values;
	grad_coef2(ND * ND * (BLOCK_SIZE + 1), { ND }, { 1 }) = -K2n.values / visc;
	biot_grad_coef2(ND * ND, { ND, ND }, { (size_t)grad_coef2.N, 1 }) = outer_product(B2n, (face.c - c2).transpose()).values;
	biot_grad_coef2(ND * ND, { ND, ND }, { (size_t)grad_coef2.N, 1 }) = (B2n.transpose() * make_block_diagonal((face.c - c2).transpose(), ND)).values;

	// Gradients
	const auto& g1 = grad[cell_id1];
	const auto& g2 = grad[cell_id2];
	g = merge_stencils(g1.stencil, g1.mat, g2.stencil, g2.mat);
	g1g = merge_stencils(g.stencil, pre_merged_grad[BLOCK_SIZE], g1.stencil, g1.mat); // alignment according the same g.stencil
	g2g = merge_stencils(g.stencil, pre_merged_grad[BLOCK_SIZE], g2.stencil, g2.mat);

	// Stencil
	flux.stencil = g.stencil;

	// Terms proportional to gradients
	fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
	fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
	flux.a(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) = (w1 * grad_coef1 * g1g.mat + w2 * grad_coef2 * g2g.mat).values;
	flux.a_biot(0, { BLOCK_SIZE, (size_t)g.mat.N }, { (size_t)flux.a_biot.N, 1 }) = (w1 * biot_grad_coef1 * g1g.mat + w2 * biot_grad_coef2 * g2g.mat).values;
	
	// Rest of advection terms
	res1 = findInVector(g.stencil, cell_id1);
	if (res1.first) { id1 = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", cell_id1);	exit(-1); }
	flux.a_biot(BLOCK_SIZE * id1 + ND, { ND, 1 }, { (size_t)flux.a_biot.N, 1 }) += (Matrix(w1(0, { ND, ND }, { (size_t)w1.N, 1 }), ND, ND) * B1n).values;
	flux.a_biot(ND * flux.a_biot.N + BLOCK_SIZE * id1, { ND }, { 1 }) += (Matrix(w1(0, { ND, ND }, { (size_t)w1.N, 1 }), ND, ND) * B1n).values;

	res2 = findInVector(g.stencil, cell_id2);
	if (res2.first) { id2 = res2.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", cell_id2);	exit(-1); }
	flux.a_biot(BLOCK_SIZE * id1 + ND, { ND, 1 }, { (size_t)flux.a_biot.N, 1 }) += (Matrix(w2(0, { ND, ND }, { (size_t)w2.N, 1 }), ND, ND) * B2n).values;
	flux.a_biot(ND * flux.a_biot.N + BLOCK_SIZE * id1, { ND }, { 1 }) += (Matrix(w2(0, { ND, ND }, { (size_t)w2.N, 1 }), ND, ND) * B2n).values;

	// Free terms 
	flux.f = w1 * grad_coef1 * g1.rhs + w2 * grad_coef2 * g2.rhs;
	flux.f(ND, 0) += w1(ND, ND) * dt / visc * (grav_vec * K1n).values[0] + w2(ND, ND) * dt / visc * (grav_vec * K2n).values[0];
	flux.f_biot = w1 * biot_grad_coef1 * g1.rhs + w2 * biot_grad_coef2 * g2.rhs;
}


