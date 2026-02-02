#include <algorithm>
#include <valarray>
#include <numeric>
#include <stdio.h>
#include <cstdint>
#include <cstring>
#include "contact.h"

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"
#else
#include "csr_matrix.h"
#include "linsolv_superlu.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::auxiliary;
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS

using namespace pm;
using std::fill_n;
using std::cout;
using std::pair;
using std::iota;
using std::vector;
using std::stable_sort;

contact::contact()
{
	f_scale = 10.0;
	friction_model = STATIC;
	friction_criterion = TERZAGHI;
	file_id = 0;
	local_jacobian = nullptr;
	local_solver = nullptr;
	timer = nullptr;
	output_counter = 0;
	normal_condition = ZERO_GAP_CHANGE;
	implicit_scheme_multiplier = 1.0;
}
contact::~contact()
{
	if (local_solver != nullptr) delete local_solver;
	if (timer != nullptr) 
	{
		timer->stop();  
		delete timer;
	}
}
int contact::init_friction(pm_discretizer* _discr, conn_mesh* _mesh)
{
	discr = _discr;
	mesh = _mesh;

	assert(mu0.size() == cell_ids.size());
	assert(mu.size() == cell_ids.size());

	if (friction_model == RSF || friction_model == RSF_STAB)
	{
		index_t cell_id;
		value_t flux_t_norm;
		Matrix n(ND, 1);

		rsf.theta.resize( cell_ids.size() );
		rsf.theta_n.resize( cell_ids.size() );
		rsf.mu_rate.resize(cell_ids.size());
		rsf.mu_state.resize(cell_ids.size());

		for (index_t i = 0; i < cell_ids.size(); i++)
		{
			cell_id = cell_ids[i];
			std::copy_n(fault_stress.begin() + (size_t)(ND * i), ND, std::begin(flux.values));

			// local basis
			const auto& n_ref = discr->faces[cell_id].back().n;
			n = discr->get_fault_sign(n_ref, cell_ids[0]) * n_ref;
			const auto& S_cur = S[i];
			flux.values = (S_cur * flux).values;
			flux_t_norm = sqrt(flux(1, 0) * flux(1, 0) + flux(2, 0) * flux(2, 0));
			mu[i] = fabs(flux_t_norm / flux(0, 0));
			if (friction_model == RSF)
				rsf.theta[i] = rsf.Dc / rsf.vel0 * exp((mu[i] - mu0[i]) / rsf.b);
			else if (friction_model == RSF_STAB)
				rsf.theta[i] = rsf.theta_n[i] = rsf.Dc / rsf.vel0 * exp(rsf.a / rsf.b * log(2 * rsf.vel0 / rsf.min_vel * sinh( mu[i] / rsf.a )) - mu0[i] / rsf.b);
		}
	}

	return 0;
}
int contact::init_fault()
{
	n_blocks = mesh->n_blocks;
	n_matrix = mesh->n_matrix;
	n_res_blocks = mesh->n_res_blocks;
	block_m = mesh->block_m.data();
	block_p = mesh->block_p.data();
	stencil = mesh->stencil.data();
	offset = mesh->offset.data();
	tran = mesh->tran.data();
	tran_biot = mesh->tran_biot.data();
	rhs = mesh->rhs.data();
	rhs_biot = mesh->rhs_biot.data();
	N_VARS_SQ = N_VARS * N_VARS;
	NT_SQ = NT * NT;

	// preallocations
	const uint8_t MAX_STENCIL = discr->MAX_STENCIL;
	Fcoef = Matrix(ND, MAX_STENCIL * ND);
	Fpres_coef = Matrix(ND, MAX_STENCIL);
	Frhs = Matrix(ND, 1);
	flux = Matrix(ND, 1);
	flux_n = Matrix(ND, 1);
	dg = Matrix(ND, 1);
	dg_iter = Matrix(ND, 1);
	F_trial = Matrix(ND, 1);
	I3 = pm::pm_discretizer::I3;
	st.reserve(MAX_STENCIL);
	ind.reserve(MAX_STENCIL);
	for (uint8_t st_size = 1; st_size < MAX_STENCIL; st_size++)
	{
		// full
		pre_F[st_size] = Matrix(ND, st_size * ND);
		pre_Fpres[st_size] = Matrix(ND, st_size);
		// normal 
		pre_Fn[st_size] = Matrix(1, st_size * ND);
		pre_Fn_pres[st_size] = Matrix(1, st_size);
	}

	// transition to local basis
	value_t avg_vol, avg_mu, avg_lam, avg_young;
	index_t cell_id, res;
	Matrix Asvd(ND, ND), Zsvd(ND, ND), w_svd(ND, 1), buf(ND, 1), n(ND, 1);
	for (index_t i = 0; i < cell_ids.size(); i++)
	{
		cell_id = cell_ids[i];
		const auto& faces = discr->faces[cell_id];
		const auto face1 = faces[faces.size() - 1];
		const auto face2 = faces[faces.size() - 2];
		const auto& n_ref = face1.n;
		const auto& conn_ids = mesh->fault_conn_id[cell_id - n_matrix];
		//sign = ((discr->cell_centers[mesh->block_p[conn_ids[0]]] - discr->cell_centers[mesh->block_m[conn_ids[0]]]).transpose() * n_ref).values[0] >= 0.0 ? 1.0 : -1.0;//discr->get_fault_sign(n_ref);
		
		n = discr->get_fault_sign(n_ref, cell_ids[0]) * n_ref;
		// null space
		fill_n(&Asvd.values[0], Asvd.values.size(), 0.0);
		fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
		fill_n(&w_svd.values[0], w_svd.values.size(), 0.0);
		Asvd(0, { ND,  1 }, { ND, 1 }) = n.values;
		res = Asvd.svd(Zsvd, w_svd.values);
		if (!res) { cout << "SVD failed!\n"; exit(-1); }
		//Asvd.transposeInplace();
		Asvd(0, { ND }, { 1 }) = n.values;
		buf = Asvd * n;
		assert(fabs(buf.values[0] - 1.0) < 1.E-10 &&
			fabs(buf.values[1]) < 1.E-10 &&
			fabs(buf.values[2]) < 1.E-10);
		S.push_back(Asvd);
		res = Asvd.inv();
		if (!res) { cout << "Inversion failed!\n"; exit(-1); }
		Sinv.push_back(Asvd);
		states.push_back(TRUE_STUCK);

		/*n = discr->get_fault_sign(n_ref) * n_ref;
		// null space
		fill_n(&Asvd.values[0], Asvd.values.size(), 0.0);
		fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
		fill_n(&w_svd.values[0], w_svd.values.size(), 0.0);
		Asvd(0, { ND,  1 }, { ND, 1 }) = n.values;
		res = Asvd.svd(Zsvd, w_svd.values);
		if (!res) { cout << "SVD failed!\n"; exit(-1); }
		//Asvd.transposeInplace();
		Asvd(0, { ND }, { 1 }) = n.values;
		buf = Asvd * n;
		assert(fabs(buf.values[0] - 1.0) < 1.E-10 &&
			fabs(buf.values[1]) < 1.E-10 &&
			fabs(buf.values[2]) < 1.E-10);
		S_fault.push_back(Asvd);*/

		// see DOI: 10.1002/nme.5345
		avg_vol = (mesh->volume[face1.cell_id2] + mesh->volume[face2.cell_id2]) / 2.0;
		avg_mu = (discr->stfs[face1.cell_id2](SUM_N(ND) - 1, SUM_N(ND) - 1) + discr->stfs[face1.cell_id2](SUM_N(ND) - 1, SUM_N(ND) - 1) ) / 2.0;
		avg_lam = (discr->stfs[face1.cell_id2](0, 1) + discr->stfs[face1.cell_id2](0, 1) ) / 2.0;
		avg_young = avg_mu * (3 * avg_lam + 2 * avg_mu) / (avg_lam + avg_mu);
		eps_t.push_back(f_scale * avg_mu);// *face1.area* face1.area / avg_vol );
		eps_n.push_back(f_scale * avg_young);// *face1.area* face1.area / avg_vol );
		value_t density = 2500.0;
		value_t s_velocity = sqrt(avg_mu * 1e+5 / density) * 86400.0;
		eta.push_back(eps_t.back()); // avg_mu / s_velocity / 2.0);
	}
	phi.resize(cell_ids.size());
	fault_stress.resize(ND * cell_ids.size());
	jacobian_explicit_scheme.resize(cell_ids.size());

	return 0;
}
int contact::init_local_iterations()
{
	// allocate memory
	dg_local.resize(ND * cell_ids.size());
	rhs_local.resize(ND * cell_ids.size());
	// create jacobian matrix
	if (!local_jacobian)
	{
		local_jacobian = new csr_matrix<ND>;
		local_jacobian->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
	}
	// allocate memory
	const uint8_t MAX_FAULT_CELL_NEBRS_NUM = 4;
	(static_cast<csr_matrix<ND>*>(local_jacobian))->init(cell_ids.size(), cell_ids.size(), ND, (MAX_FAULT_CELL_NEBRS_NUM + 1) * cell_ids.size());
	// create linear solver
	if (!local_solver) 
		local_solver = new linsolv_superlu<ND>;
	// init matrix pattern
	if (cell_ids.size())
		init_local_jacobian_structure();
	// init linear solver
	const uint8_t MAX_LOCAL_ITERS = 10;
	const value_t LINEAR_TOLERANCE = 1.E-6;

	if (!timer) timer = new timer_node();
	timer->start();
	timer->node["linear solver setup"] = timer_node();
	timer->node["linear solver solve"] = timer_node();

	local_solver->init_timer_nodes(&timer->node["linear solver setup"], &timer->node["linear solver solve"]);
	local_solver->init(local_jacobian, MAX_LOCAL_ITERS, LINEAR_TOLERANCE);

	return 0;
}

void contact::set_state(const ContactState& state)
{
	fill_n(states.begin(), states.size(), state);
	if (!states_n.size())
		fill_n(states_n.begin(), states_n.size(), state);
}
void contact::merge_tractions_biot(const index_t i, const vector<value_t>& fluxes, const vector<value_t>& fluxes_biot, const vector<value_t>& X,
													const vector<value_t>& fluxes_n, const vector<value_t>& fluxes_biot_n, const vector<value_t>& Xn,
													const vector<value_t>& fluxes_ref, const vector<value_t>& fluxes_biot_ref, const vector<value_t>& Xref,
													const vector<value_t>& fluxes_ref_n, const vector<value_t>& fluxes_biot_ref_n, const vector<value_t>& Xn_ref)
{
	std::pair<bool, size_t> res;
	index_t conn_st_id;
	size_t id;
	Matrix n (ND, 1);
	value_t sign;
	uint8_t d, v;
	st.resize(0);
	st.reserve(pm_discretizer::MAX_STENCIL);
	fill_n(&Fcoef.values[0], Fcoef.values.size(), 0.0);
	fill_n(&Fpres_coef.values[0], Fpres_coef.values.size(), 0.0);
	fill_n(&Frhs.values[0], Frhs.values.size(), 0.0);
	fill_n(&flux.values[0], flux.values.size(), 0.0);
	fill_n(&flux_n.values[0], flux_n.values.size(), 0.0);

	// merge tractions approximated from two sides
	const auto& cell_id = cell_ids[i];
	const auto& conn_ids = mesh->fault_conn_id[cell_id - n_matrix];
	n = discr->faces[cell_id].back().n * discr->faces[cell_id].back().area;
	if ((n.transpose() * (discr->cell_centers[mesh->block_p[conn_ids[0]]] - discr->cell_centers[mesh->block_m[conn_ids[0]]])).values[0] < 0.0) n = -n;
	sign = discr->get_fault_sign(n, cell_ids[0]);// / 2.0;

	for (uint8_t k = 0; k < 1/*conn_ids.size()*/; k++)
	{
		const auto& conn_id = conn_ids[k];
		if (k == 1) sign = -sign;
		for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
		{
			res = discr->findInVector(st, stencil[conn_st_id]);
			if (res.first) { id = res.second; }
			else { id = st.size(); st.push_back(stencil[conn_st_id]); }
			for (d = 0; d < ND; d++)
			{
				for (v = 0; v < ND; v++)
				{
					Fcoef.values[d * Fcoef.N + id * ND + v] += sign *
						(tran[conn_st_id * NT_SQ + (U_VAR_T + d) * NT + (U_VAR_T + v)]);
				}
				Fpres_coef.values[d * Fpres_coef.N + id] += sign * 
					(tran[conn_st_id * NT_SQ + (U_VAR_T + d) * NT + P_VAR_T]);
			}
		}

		res = discr->findInVector(st, cell_id);
		if (res.first) { id = res.second; }
		else { printf("Traction does not depend on the gap!\n");	exit(-1); }

		// pressure contribution & free term of flux
		for (d = 0; d < ND; d++)
		{
			Frhs.values[d] += sign * (rhs[NT * conn_id + U_VAR_T + d]);
			flux.values[d] += sign * (fluxes_ref[N_VARS * conn_id + U_VAR + d] + fluxes[N_VARS * conn_id + U_VAR + d]);
			flux_n.values[d] += sign * (fluxes_ref_n[N_VARS * conn_id + U_VAR + d] + fluxes_n[N_VARS * conn_id + U_VAR + d]);
		}

		//printf("%d, #%d, %d: %f\t%f\t%f\n", cell_id, k, conn_id, fluxes[N_VARS * conn_id + U_VAR + 0], 
		//														 fluxes[N_VARS * conn_id + U_VAR + 1], 
		//														 fluxes[N_VARS * conn_id + U_VAR + 2]);
	}

	std::copy(std::begin(flux.values), std::end(flux.values), fault_stress.begin() + (size_t)(ND * i));

	// add local gap change
	/*res = discr->findInVector(st, stencil[conn_st_id]);
	if (res.first) { id = res.second; }
	else { printf("No gap d.o.f.\n"); exit(-1); }
	for (d = 0; d < ND; d++)
	{
		for (v = 0; v < ND; v++)
		{
			flux.values[d] += Fcoef.values[d * Fcoef.N + id * ND + v] * dg_local.values[v];
		}
	}*/

	// sort merged stencil
	ind.resize(st.size());
	iota(ind.begin(), ind.begin() + st.size(), 0);
	stable_sort(ind.begin(), ind.end(), [this](index_t i1, index_t i2) { return st[i1] < st[i2]; });
}
void contact::merge_tractions_terzaghi(const index_t i, const vector<value_t>& fluxes, const vector<value_t>& fluxes_biot, const vector<value_t>& X,
													const vector<value_t>& fluxes_n, const vector<value_t>& fluxes_biot_n, const vector<value_t>& Xn,
													const vector<value_t>& fluxes_ref, const vector<value_t>& fluxes_biot_ref, const vector<value_t>& Xref,
													const vector<value_t>& fluxes_ref_n, const vector<value_t>& fluxes_biot_ref_n, const vector<value_t>& Xn_ref)
{
	std::pair<bool, size_t> res;
	index_t conn_st_id;
	size_t id;
	Matrix n(ND, 1);
	value_t sign;
	uint8_t d, v;
	st.resize(0);
	st.reserve(pm_discretizer::MAX_STENCIL);
	fill_n(&Fcoef.values[0], Fcoef.values.size(), 0.0);
	fill_n(&Fpres_coef.values[0], Fpres_coef.values.size(), 0.0);
	fill_n(&Frhs.values[0], Frhs.values.size(), 0.0);
	fill_n(&flux.values[0], flux.values.size(), 0.0);
	fill_n(&flux_n.values[0], flux_n.values.size(), 0.0);

	// merge tractions approximated from two sides
	const auto& cell_id = cell_ids[i];
	const auto& conn_ids = mesh->fault_conn_id[cell_id - n_matrix];
	n = discr->faces[cell_id].back().n * discr->faces[cell_id].back().area;
	if ((n.transpose() * (discr->cell_centers[mesh->block_p[conn_ids[0]]] - discr->cell_centers[mesh->block_m[conn_ids[0]]])).values[0] < 0.0) n = -n;
	sign = discr->get_fault_sign(n, cell_ids[0]);// / 2.0;

	for (uint8_t k = 0; k < 1/*conn_ids.size()*/; k++)
	{
		const auto& conn_id = conn_ids[k];
		if (k == 1) sign = -sign;
		for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
		{
			res = discr->findInVector(st, stencil[conn_st_id]);
			if (res.first) { id = res.second; }
			else { id = st.size(); st.push_back(stencil[conn_st_id]); }
			for (d = 0; d < ND; d++)
			{
				for (v = 0; v < ND; v++)
				{
					Fcoef.values[d * Fcoef.N + id * ND + v] += sign *
						(tran[conn_st_id * NT_SQ + (U_VAR_T + d) * NT + (U_VAR_T + v)] + tran_biot[conn_st_id * NT + (U_VAR_T + d) * NT + (U_VAR_T + v)]);
				}
				Fpres_coef.values[d * Fpres_coef.N + id] += sign *
					(tran[conn_st_id * NT_SQ + (U_VAR_T + d) * NT + P_VAR_T] + tran_biot[conn_st_id * NT + (U_VAR_T + d) * NT + P_VAR_T]);
			}
		}

		res = discr->findInVector(st, cell_id);
		if (res.first) { id = res.second; }
		else { printf("Traction does not depend on the gap!\n");	exit(-1); }
		Fpres_coef(id, { ND, 1 }, { (uint8_t)Fpres_coef.N, 1 }) -= sign * n.values;

		// pressure contribution & free term of flux
		for (d = 0; d < ND; d++)
		{
			Frhs.values[d] += sign * (rhs[NT * conn_id + U_VAR_T + d] + rhs_biot[NT * conn_id + U_VAR_T + d]);
			flux.values[d] += sign * (fluxes_ref[N_VARS * conn_id + U_VAR + d] + fluxes_biot_ref[N_VARS * conn_id + U_VAR + d] + 
				fluxes[N_VARS * conn_id + U_VAR + d] + fluxes_biot[N_VARS * conn_id + U_VAR + d] - (X[N_VARS * cell_id + P_VAR]) * n(d, 0));
			flux_n.values[d] += sign * (fluxes_ref_n[N_VARS * conn_id + U_VAR + d] + fluxes_biot_ref_n[N_VARS * conn_id + U_VAR + d] + 
				fluxes_n[N_VARS * conn_id + U_VAR + d] + fluxes_biot_n[N_VARS * conn_id + U_VAR + d] - (Xn[N_VARS * cell_id + P_VAR]) * n(d, 0));
		}

		//printf("%d, #%d, %d: %f\t%f\t%f\n", cell_id, k, conn_id, fluxes[N_VARS * conn_id + U_VAR + 0], 
		//														 fluxes[N_VARS * conn_id + U_VAR + 1], 
		//														 fluxes[N_VARS * conn_id + U_VAR + 2]);
	}

	std::copy(std::begin(flux.values), std::end(flux.values), fault_stress.begin() + (size_t)(ND * i));

	// add local gap change
	/*res = discr->findInVector(st, stencil[conn_st_id]);
	if (res.first) { id = res.second; }
	else { printf("No gap d.o.f.\n"); exit(-1); }
	for (d = 0; d < ND; d++)
	{
		for (v = 0; v < ND; v++)
		{
			flux.values[d] += Fcoef.values[d * Fcoef.N + id * ND + v] * dg_local.values[v];
		}
	}*/

	// sort merged stencil
	ind.resize(st.size());
	iota(ind.begin(), ind.begin() + st.size(), 0);
	stable_sort(ind.begin(), ind.end(), [this](index_t i1, index_t i2) { return st[i1] < st[i2]; });
}
int contact::add_to_jacobian_return_mapping(value_t dt, csr_matrix_base* jacobian, vector<value_t>& RHS, const vector<value_t>& X, const vector<value_t>& fluxes, const vector<value_t>& fluxes_biot,
																								const vector<value_t>& Xn, const vector<value_t>& fluxes_n, const vector<value_t>& fluxes_biot_n,
																									vector<value_t>& Xref, vector<value_t>& fluxes_ref, vector<value_t>& fluxes_biot_ref,
																										const vector<value_t>& Xn_ref, const vector<value_t>& fluxes_ref_n, const vector<value_t>& fluxes_biot_ref_n)
{
	std::pair<bool, size_t> res;
	uint8_t d, v;
	size_t id;
	index_t cell_id;
	value_t slip_vel_norm, alpha;
	Matrix n(ND, 1);
	Matrix buf(ND, ND);
	value_t sign, proj, sign_trial, flux_t_norm;
	Matrix g(ND, 1), gn(ND, 1), dmu(ND, 1), slip_vel(ND, 1), drad_dump(ND, 1);
	const value_t *p_ref = mesh->ref_pressure.data();
	num_of_change_sign = 0;

	Jac = jacobian->get_values();
	diag_ind = jacobian->get_diag_ind();
	rows = jacobian->get_rows_ptr();
	cols = jacobian->get_cols_ind();
	max_allowed_gap_change = 0.0;

	/*FILE* pFile;
	std::string fname = "sol_poromechanics/friction_output_" + std::to_string(file_id++) + ".txt";
	pFile = fopen(fname.c_str(), "w");*/
	
	for (index_t i = 0; i < cell_ids.size(); i++)
	{
		jacobian_explicit_scheme[i].values = 0.0;

		cell_id = cell_ids[i];

		// set gap change
		for (d = 0; d < ND; d++)
		{
			dg.values[d] = X[N_VARS * cell_id + U_VAR + d] - Xn[N_VARS * cell_id + U_VAR + d];
			//dg_iter.values[d] = -dX[N_VARS * cell_id + U_VAR + d];

			g.values[d] = X[N_VARS * cell_id + U_VAR + d];
			gn.values[d] = Xn[N_VARS * cell_id + U_VAR + d];
		}

		if (friction_criterion == BIOT)
			merge_tractions_biot(i, fluxes, fluxes_biot, X, fluxes_n, fluxes_biot_n, Xn, fluxes_ref, fluxes_biot_ref, Xref, fluxes_ref_n, fluxes_biot_ref_n, Xn_ref);
		else if (friction_criterion == TERZAGHI)
			merge_tractions_terzaghi(i, fluxes, fluxes_biot, X, fluxes_n, fluxes_biot_n, Xn, fluxes_ref, fluxes_biot_ref, Xref, fluxes_ref_n, fluxes_biot_ref_n, Xn_ref);


		auto& state = states[i];
		if (state == TRUE_STUCK)
		{
			add_to_jacobian_stuck(i, dt, RHS);
		}
		else if (state == PEN_STUCK || state == SLIP || state == FREE)
		{
			// set tangential condition
			auto& F = pre_F[st.size()];
			F.values = Fcoef(0, { ND, st.size() * ND }, { (uint8_t)Fcoef.N, 1 });
			auto& Fpres = pre_Fpres[st.size()];
			Fpres.values = Fpres_coef(0, { ND, st.size() }, { (uint8_t)Fpres_coef.N, 1 });

			// local basis
			const auto& n_ref = discr->faces[cell_id].back().n;
			const auto& conn_ids = mesh->fault_conn_id[cell_id - n_matrix];
			//sign = ((discr->cell_centers[mesh->block_p[conn_ids[0]]] - discr->cell_centers[mesh->block_m[conn_ids[0]]]).transpose() * n_ref).values[0] >= 0.0 ? 1.0 : -1.0;//discr->get_fault_sign(n_ref);
			n = discr->get_fault_sign(n_ref, cell_ids[0]) * n_ref;
			const auto& S_cur = S[i];
			//const auto& Sf = S_fault[i];
			const auto& Sinv_cur = Sinv[i];
			F.values = (S_cur * F * make_block_diagonal(Sinv_cur, st.size())).values;
			Fpres.values = (S_cur * Fpres).values;
			Frhs.values = (S_cur * Frhs).values;
			flux.values = (S_cur * flux).values;
			flux_n.values = (S_cur * flux_n).values;
			g.values = (S_cur * g).values;
			dg.values = (S_cur * dg).values;
			dg_iter.values = (S_cur * dg_iter).values;
			dgt_norm = sqrt(dg(1, 0) * dg(1, 0) + dg(2, 0) * dg(2, 0));
			dgt_iter_norm = sqrt(dg_iter(1, 0) * dg_iter(1, 0) + dg_iter(2, 0) * dg_iter(2, 0));
			//assert(flux(0, 0) >= 0.0);

			// gap id
			res = discr->findInVector(st, cell_id);
			if (res.first) { id = res.second; }
			else { printf("Traction does not depend on the gap!\n");	exit(-1); }

			slip_vel.values = dg.values / dt;
			slip_vel(0, 0) = 0.0;
			slip_vel_norm = dgt_norm / dt;
			if (eta[i] != 0.0 || friction_model == RSF || friction_model == RSF_STAB) slip_vel_norm += rsf.min_vel;
			// friction and its derivative
			if (friction_model == RSF)
			{
				auto res = getFrictionCoef(i, dt, slip_vel, g);
				mu[i] = res[0];
				dmu.values[0] = res[1];		dmu.values[1] = res[2];		dmu.values[2] = res[3];
			}
			else
			{
				auto res = getStabilizedFrictionCoef(i, dt, slip_vel, g);
				mu[i] = res[0];
				dmu.values[0] = res[1];		dmu.values[1] = res[2];		dmu.values[2] = res[3];
			}

			// trial traction 
			//max_allowed_gap_change = std::max(max_allowed_gap_change, fabs(sqrt(flux.values[1] * flux.values[1] + flux.values[2] * flux.values[2]) -
			//	sqrt(flux_n.values[1] * flux_n.values[1] + flux_n.values[2] * flux_n.values[2])) / (fabs(F(1, ND * id + 1)) + fabs(F(2, ND * id + 2))));
			sign_trial = -discr->get_fault_sign(n, cell_ids[0]);
			assert(sign_trial < 0);
			//assert(dg.values[1] <= 0);
			F_trial = flux_n + sign_trial * eps_t[i] * dg;
			F_trial(0, 0) = 0.0;
			flux_t_norm = sqrt(flux(1, 0) * flux(1, 0) + flux(2, 0) * flux(2, 0));
			Ft_trial_norm = sqrt(F_trial(1, 0) * F_trial(1, 0) + F_trial(2, 0) * F_trial(2, 0));
			phi[i] = Ft_trial_norm - mu[i] * flux(0, 0);

			// radiation dumping
			if (eta[i] != 0.0)
			  phi[i] -= dt * eta[i] * slip_vel_norm;

			// normal 
			auto& Fn = pre_Fn[st.size()];
			auto& Fn_pres = pre_Fn_pres[st.size()];
			Fn.values = F(0, { (uint8_t)F.N }, { 1 });
			Fn_pres.values = Fpres(0, { (uint8_t)Fpres.N }, { 1 });
			 
			// update state
			if (flux(0, 0) < -100 * EQUALITY_TOLERANCE && false)
				state = FREE;
			else if (phi[i] >= 0.0 || (state == SLIP && dgt_iter_norm > 100 * EQUALITY_TOLERANCE))
				state = SLIP;
			else
				state = PEN_STUCK;

			if (state == FREE)
			{
			}
			else if (state == SLIP && Ft_trial_norm > EQUALITY_TOLERANCE / 100)
			{
				// static (or zero) friction by default
				drad_dump.values = 0.0;
				drad_dump.values = dt * eta[i] * slip_vel.values / slip_vel_norm / dt;

				////// friction
				//// if use gap derivatives as direction
				// F -= mu0[i] / dgt_norm * outer_product(dg, Fn);
				// Fpres -= mu0[i] / dgt_norm * outer_product(dg, Fn_pres);
				// F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= mu0[i] * flux.values[0] / dgt_norm * (I3 - outer_product(dg / dgt_norm, dg.transpose() / dgt_norm)).values;
				// flux -= mu0[i] * flux.values[0] * dg / dgt_norm;

				//// trial traction as direction (V. Yastrebov, 2013; Simo et al., 1992)
				alpha = 1 - phi[i] / Ft_trial_norm;

				//// dphi 
				// dFn
				F -= mu[i] / Ft_trial_norm * outer_product(F_trial, Fn);
				Fpres -= mu[i] / Ft_trial_norm * outer_product(F_trial, Fn_pres);
				// dmu
				F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= flux(0, 0) / Ft_trial_norm * outer_product(F_trial, dmu.transpose()).values;
				jacobian_explicit_scheme[i].values = -flux(0, 0) / Ft_trial_norm * outer_product(F_trial, Matrix(jacobian_explicit_scheme[i](0, {ND}, {1}), 1, ND)).values;

				buf.values = outer_product(F_trial / Ft_trial_norm, F_trial.transpose() / Ft_trial_norm).values;
				buf(0, { ND }, { 1 }) = 0.0;
				// radiation dumping
				if (eta[i] != 0.0) {
				  F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= 1.0 / Ft_trial_norm * outer_product(F_trial, drad_dump.transpose()).values;
				}
				// dFt_trial_norm
				F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= -sign_trial * eps_t[i] * buf.values;
				jacobian_explicit_scheme[i].values += sign_trial * eps_t[i] * buf.values;
				//// d(1/Ft_trial_norm)
				F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= phi[i] / Ft_trial_norm * sign_trial * eps_t[i] * buf.values;
				jacobian_explicit_scheme[i].values -= phi[i] / Ft_trial_norm * sign_trial * eps_t[i] * buf.values;
				//// dF_trial
				F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= alpha * sign_trial * eps_t[i] * I3.values;
				jacobian_explicit_scheme[i].values -= alpha * sign_trial * eps_t[i] * I3.values;

				// trial traction as direction (Garipov)
				//F -= mu_cur / Ft_trial_norm * outer_product(F_trial, Fn);
				//Fpres -= mu_cur / Ft_trial_norm * outer_product(F_trial, Fn_pres);
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= mu_cur * flux(0, 0) / Ft_trial_norm * sign_trial * eps_t[i] * I3.values;
				//// add contribution due to friction coefficient change
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= flux(0, 0) * outer_product(F_trial, dmu.transpose()).values / Ft_trial_norm;
				//buf.values = outer_product(F_trial / Ft_trial_norm, F_trial.transpose() / Ft_trial_norm).values;
				//buf(0, { ND }, { 1 }) = 0.0;
				// denominator
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= -mu_cur * flux(0, 0) / Ft_trial_norm * sign_trial * eps_t[i] * buf.values;

				// residual
				//flux -= mu_cur * flux(0, 0) * F_trial / Ft_trial_norm;
				flux -= alpha * F_trial;

				// radiation damping
				if (eta[i] != 0.0)
				{
				  F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= dt * eta[i] * I3.values / dt;
				  flux -= dt * eta[i] * slip_vel;
				}
					//fprintf(pFile, "%d\t%d\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", i, 1, flux.values[1], F(1, ND * id + 1), mu_cur, dmu.values[1], slip_vel.values[1], g.values[1], dg.values[1]);
				//}
			}
			else
			{
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= eps_t[i] * I3.values;
				//sign = dg(1, 0) * flux_n(1, 0) + dg(2, 0) * flux_n(2, 0) < 0.0 ? -1.0 : 1.0;
				//fill_n(&F.values[0], F.values.size(), 0.0);
				//fill_n(&Fpres.values[0], Fpres.values.size(), 0.0);
				//fill_n(&Frhs.values[0], Frhs.values.size(), 0.0);
				F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= sign_trial * eps_t[i] * I3.values;
				jacobian_explicit_scheme[i].values -= sign_trial * eps_t[i] * I3.values;
				flux -= F_trial;
				//flux -= flux + sign_trial * eps_t[i] * dg;

				//fprintf(pFile, "%d\t%d\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", i, 0, 0.0, 0.0, 0.0, 0.0, 0.0, g.values[1], dg.values[1]);
			}
			//printf("%d\tft=%.10e\tft_trial=%.10e\tgt=%.10e\tdgt=%.10e\n", i, flux.values[1], F_trial.values[1], g.values[1], dg.values[1]);
			
			//// set normal condition (first row)
			if (normal_condition == PENALIZED)
			{
				//printf("%d:\t#%d\t%f\tn:%f %f %f\n", fault_tag, cell_id, sign_trial, n.values[0], n.values[1], n.values[2]);
				//// f_n - eps_n * <g_n> = 0
				F(0, { (uint8_t)F.N }, { 1 }) = Fn.values;
				Fpres(0, { (uint8_t)Fpres.N }, { 1 }) = Fn_pres.values;
				F(0, ND * id) += sign_trial * (g(0, 0) >= 0.0 ? 1.0 : 0.0) * eps_n[i];
				flux(0, 0) += sign_trial * eps_n[i] * ( g(0,0) + fabs(g(0,0)) ) / 2;
				jacobian_explicit_scheme[i](0, 0) = (g(0, 0) >= 0.0 ? 1.0 : 0.0) * sign_trial * eps_n[i];
			}
			else if (normal_condition == ZERO_GAP_CHANGE)
			{
				//// dg_n = 0
				F(0, { (uint8_t)F.N }, { 1 }) = 0.0;
				Fpres(0, { (uint8_t)Fpres.N }, { 1 }) = 0.0;
				F(0, ND * id) = eps_n[i];
				flux(0, 0) = eps_n[i] * dg(0, 0);
				jacobian_explicit_scheme[i](0, 0) = eps_n[i];
			}
			jacobian_explicit_scheme[i](0, 1) = 0.0;
			jacobian_explicit_scheme[i](0, 2) = 0.0;

			// scale the equations
			F.values /= sqrt(eps_n[i]);
			Fpres.values /= sqrt(eps_n[i]);
			flux.values /= sqrt(eps_n[i]);
			jacobian_explicit_scheme[i].values /= sqrt(eps_n[i]);

			// stay with 'x' vector in Cartesian coordinates
			F.values = (F * make_block_diagonal(S_cur, st.size())).values;
			jacobian_explicit_scheme[i].values = (jacobian_explicit_scheme[i] * S_cur).values;

			// permutation of equations
			permut[0] = 0;
			permut[1] = 1;
			permut[2] = 2;
			//iota(permut, permut + ND, 0);
			stable_sort(permut, permut + ND, [&n](index_t i1, index_t i2) { return fabs(n.values[i1]) > fabs(n.values[i2]); });

			add_to_jacobian_slip(i, dt, RHS);
		}
	}
	//fclose(pFile);

	return 0;
}
int contact::add_to_jacobian_linear(value_t dt, csr_matrix_base* jacobian, vector<value_t>& RHS, const vector<value_t>& X, const vector<value_t>& fluxes, const vector<value_t>& fluxes_biot,
	const vector<value_t>& Xn, const vector<value_t>& fluxes_n, const vector<value_t>& fluxes_biot_n,
	vector<value_t>& Xref, vector<value_t>& fluxes_ref, vector<value_t>& fluxes_biot_ref,
	const vector<value_t>& Xn_ref, const vector<value_t>& fluxes_ref_n, const vector<value_t>& fluxes_biot_ref_n)
{
	std::pair<bool, size_t> res;
	uint8_t d, v;
	size_t id;
	index_t cell_id;
	value_t slip_vel_norm, alpha;
	Matrix n(ND, 1);
	Matrix buf(ND, ND);
	value_t sign, proj, sign_trial, flux_t_norm;
	Matrix g(ND, 1), gn(ND, 1), dmu(ND, 1), slip_vel(ND, 1);
	const value_t *p_ref = mesh->ref_pressure.data();
	num_of_change_sign = 0;

	Jac = jacobian->get_values();
	diag_ind = jacobian->get_diag_ind();
	rows = jacobian->get_rows_ptr();
	cols = jacobian->get_cols_ind();
	max_allowed_gap_change = 0.0;

	/*FILE* pFile;
	std::string fname = "sol_poromechanics/friction_output_" + std::to_string(file_id++) + ".txt";
	pFile = fopen(fname.c_str(), "w");*/

	for (index_t i = 0; i < cell_ids.size(); i++)
	{
		cell_id = cell_ids[i];
		// set gap change
		for (d = 0; d < ND; d++)
		{
			dg.values[d] = X[N_VARS * cell_id + U_VAR + d] - Xn[N_VARS * cell_id + U_VAR + d];
			//dg_iter.values[d] = -dX[N_VARS * cell_id + U_VAR + d];

			g.values[d] = X[N_VARS * cell_id + U_VAR + d];
			gn.values[d] = Xn[N_VARS * cell_id + U_VAR + d];
		}

		if (friction_criterion == BIOT)
			merge_tractions_biot(i, fluxes, fluxes_biot, X, fluxes_n, fluxes_biot_n, Xn, fluxes_ref, fluxes_biot_ref, Xref, fluxes_ref_n, fluxes_biot_ref_n, Xn_ref);
		else if (friction_criterion == TERZAGHI)
			merge_tractions_terzaghi(i, fluxes, fluxes_biot, X, fluxes_n, fluxes_biot_n, Xn, fluxes_ref, fluxes_biot_ref, Xref, fluxes_ref_n, fluxes_biot_ref_n, Xn_ref);

		auto& state = states[i];
		if (state == TRUE_STUCK)
		{
			add_to_jacobian_stuck(i, dt, RHS);
		}
		else if (state == PEN_STUCK || state == SLIP)
		{
			// set tangential condition
			auto& F = pre_F[st.size()];
			F.values = Fcoef(0, { ND, st.size() * ND }, { (uint8_t)Fcoef.N, 1 });
			auto& Fpres = pre_Fpres[st.size()];
			Fpres.values = Fpres_coef(0, { ND, st.size() }, { (uint8_t)Fpres_coef.N, 1 });

			// local basis
			const auto& n_ref = discr->faces[cell_id].back().n;
			const auto& conn_ids = mesh->fault_conn_id[cell_id - n_matrix];
			//sign = ((discr->cell_centers[mesh->block_p[conn_ids[0]]] - discr->cell_centers[mesh->block_m[conn_ids[0]]]).transpose() * n_ref).values[0] >= 0.0 ? 1.0 : -1.0;//discr->get_fault_sign(n_ref);
			n = discr->get_fault_sign(n_ref, cell_ids[0]) * n_ref;
			const auto& S_cur = S[i];
			//const auto& Sf = S_fault[i];
			const auto& Sinv_cur = Sinv[i];
			F.values = (S_cur * F * make_block_diagonal(Sinv_cur, st.size())).values;
			Fpres.values = (S_cur * Fpres).values;
			Frhs.values = (S_cur * Frhs).values;
			flux.values = (S_cur * flux).values;
			flux_n.values = (S_cur * flux_n).values;
			g.values = (S_cur * g).values;
			dg.values = (S_cur * dg).values;
			dg_iter.values = (S_cur * dg_iter).values;
			dgt_norm = sqrt(dg(1, 0) * dg(1, 0) + dg(2, 0) * dg(2, 0));
			dgt_iter_norm = sqrt(dg_iter(1, 0) * dg_iter(1, 0) + dg_iter(2, 0) * dg_iter(2, 0));
			//assert(flux(0, 0) >= 0.0);

			// gap id
			res = discr->findInVector(st, cell_id);
			if (res.first) { id = res.second; }
			else { printf("Traction does not depend on the gap!\n");	exit(-1); }

			slip_vel.values = dg.values / dt;
			slip_vel_norm = dgt_norm / dt;
			if (friction_model == RSF || friction_model == RSF_STAB) slip_vel_norm += rsf.min_vel;
			// friction and its derivative
			auto res = getStabilizedFrictionCoef(i, dt, slip_vel, g);
			mu[i] = res[0];
			dmu.values[0] = res[1];		dmu.values[1] = res[2];		dmu.values[2] = res[3];

			// trial traction 
			//max_allowed_gap_change = std::max(max_allowed_gap_change, fabs(sqrt(flux.values[1] * flux.values[1] + flux.values[2] * flux.values[2]) -
			//	sqrt(flux_n.values[1] * flux_n.values[1] + flux_n.values[2] * flux_n.values[2])) / (fabs(F(1, ND * id + 1)) + fabs(F(2, ND * id + 2))));
			sign_trial = -discr->get_fault_sign(n, cell_ids[0]);
			assert(sign_trial < 0);
			//assert(dg.values[1] <= 0);
			F_trial = flux_n + sign_trial * eps_t[i] * dg;
			F_trial(0, 0) = 0.0;
			flux_t_norm = sqrt(flux(1, 0) * flux(1, 0) + flux(2, 0) * flux(2, 0));
			Ft_trial_norm = sqrt(F_trial(1, 0) * F_trial(1, 0) + F_trial(2, 0) * F_trial(2, 0));
			phi[i] = Ft_trial_norm - mu[i] * flux(0, 0);

			// normal 
			auto& Fn = pre_Fn[st.size()];
			auto& Fn_pres = pre_Fn_pres[st.size()];
			Fn.values = F(0, { (uint8_t)F.N }, { 1 });
			Fn_pres.values = Fpres(0, { (uint8_t)Fpres.N }, { 1 });

			// update state
			if (phi[i] >= 0.0 || (state == SLIP && dgt_iter_norm > 100 * EQUALITY_TOLERANCE))
				state = SLIP;
			else
				state = PEN_STUCK;

			if (state == SLIP && flux_t_norm > 0.0)
			{
				////// friction
				//// if use gap derivatives as direction
				// F -= mu0[i] / dgt_norm * outer_product(dg, Fn);
				// Fpres -= mu0[i] / dgt_norm * outer_product(dg, Fn_pres);
				// F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= mu0[i] * flux.values[0] / dgt_norm * (I3 - outer_product(dg / dgt_norm, dg.transpose() / dgt_norm)).values;
				// flux -= mu0[i] * flux.values[0] * dg / dgt_norm;

				//// trial traction as direction (V. Yastrebov, 2013; Simo et al., 1992)
				// alpha = 1 - phi[i] / Ft_trial_norm;

				//// dphi 
				// dFn
				// F -= mu_cur / Ft_trial_norm * outer_product(F_trial, Fn);
				// Fpres -= mu_cur / Ft_trial_norm * outer_product(F_trial, Fn_pres);
				// dmu
				// F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= flux(0, 0) / Ft_trial_norm * outer_product(F_trial, dmu.transpose()).values;
				// buf.values = outer_product(F_trial / Ft_trial_norm, F_trial.transpose() / Ft_trial_norm).values;
				// buf(0, { ND }, { 1 }) = 0.0;
				// dFt_trial_norm
				// F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= -sign_trial * eps_t[i] * buf.values;
				//// d(1/Ft_trial_norm)
				// F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= phi[i] / Ft_trial_norm * sign_trial * eps_t[i] * buf.values;
				//// dF_trial
				// F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= alpha * sign_trial * eps_t[i] * I3.values;

				// trial traction as direction (Garipov)
				//F -= mu_cur / Ft_trial_norm * outer_product(F_trial, Fn);
				//Fpres -= mu_cur / Ft_trial_norm * outer_product(F_trial, Fn_pres);
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= mu_cur * flux(0, 0) / Ft_trial_norm * sign_trial * eps_t[i] * I3.values;
				//// add contribution due to friction coefficient change
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= flux(0, 0) * outer_product(F_trial, dmu.transpose()).values / Ft_trial_norm;
				//buf.values = outer_product(F_trial / Ft_trial_norm, F_trial.transpose() / Ft_trial_norm).values;
				//buf(0, { ND }, { 1 }) = 0.0;
				// denominator
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= -mu_cur * flux(0, 0) / Ft_trial_norm * sign_trial * eps_t[i] * buf.values;

				// residual
				//flux -= mu_cur * flux(0, 0) * F_trial / Ft_trial_norm;
				// flux -= alpha * F_trial;

				//// current traction as direction
				F -= mu[i] / flux_t_norm * outer_product(flux, Fn);
				Fpres -= mu[i] / flux_t_norm * outer_product(flux, Fn_pres);
				//// add contribution due to friction coefficient change
				F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= flux(0, 0) * outer_product(flux, dmu.transpose()).values / flux_t_norm;
				// residual
				flux -= mu[i] * flux(0, 0) * flux / flux_t_norm;

				//fprintf(pFile, "%d\t%d\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", i, 1, flux.values[1], F(1, ND * id + 1), mu_cur, dmu.values[1], slip_vel.values[1], g.values[1], dg.values[1]);
			}
			else
			{
				//F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= eps_t[i] * I3.values;
				//sign = dg(1, 0) * flux_n(1, 0) + dg(2, 0) * flux_n(2, 0) < 0.0 ? -1.0 : 1.0;
				//fill_n(&F.values[0], F.values.size(), 0.0);
				//fill_n(&Fpres.values[0], Fpres.values.size(), 0.0);
				//fill_n(&Frhs.values[0], Frhs.values.size(), 0.0);
				F(ND * id, { ND, ND }, { (uint8_t)F.N, 1 }) -= sign_trial * eps_t[i] * I3.values;
				flux -= F_trial;
				//flux -= flux + sign_trial * eps_t[i] * dg;

				//fprintf(pFile, "%d\t%d\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", i, 0, 0.0, 0.0, 0.0, 0.0, 0.0, g.values[1], dg.values[1]);
			}

			// set normal condition (first row)
			// f_n - eps_n * <g_n> = 0
			F(0, { (uint8_t)F.N }, { 1 }) = Fn.values;
			Fpres(0, { (uint8_t)Fpres.N }, { 1 }) = Fn_pres.values;
			F(0, ND * id) -= (g(0, 0) >= 0.0 ? 1.0 : 0.0) * eps_n[i];
			flux(0, 0) -= eps_n[i] * (g(0, 0) + fabs(g(0, 0))) / 2;

			// scale the equations
			F.values /= sqrt(eps_n[i]);
			Fpres.values /= sqrt(eps_n[i]);
			flux.values /= sqrt(eps_n[i]);

			// stay with 'x' vector in Cartesian coordinates
			F.values = (F * make_block_diagonal(S_cur, st.size())).values;

			// permutation of equations
			permut[0] = 0;
			permut[1] = 1;
			permut[2] = 2;
			//iota(permut, permut + ND, 0);
			stable_sort(permut, permut + ND, [&n](index_t i1, index_t i2) { return fabs(n.values[i1]) > fabs(n.values[i2]); });

			add_to_jacobian_slip(i, dt, RHS);
		}
	}
	//fclose(pFile);

	return 0;
}
int contact::add_to_jacobian_local_iters(value_t dt, csr_matrix_base* jacobian, vector<value_t>& RHS, vector<value_t>& X, vector<value_t>& fluxes, vector<value_t>& fluxes_biot,
																									const vector<value_t>& Xn, const vector<value_t>& fluxes_n, const vector<value_t>& fluxes_biot_n,
																										vector<value_t>& Xref, vector<value_t>& fluxes_ref, vector<value_t>& fluxes_biot_ref,
																											const vector<value_t>& Xn_ref, const vector<value_t>& fluxes_ref_n, const vector<value_t>& fluxes_biot_ref_n)
{
	// update jacobian
	add_to_jacobian_return_mapping(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
	
	index_t is_not_true_stuck = 0;
	for (const auto& state : states)
		if (state != TRUE_STUCK) { is_not_true_stuck++; break; }

	// calculate residual
	const value_t init_res = calc_gap_L2_residual(RHS);
	value_t res = init_res;

	if (is_not_true_stuck && res > 10000 * EQUALITY_TOLERANCE)
	{
		// local iterations
		uint8_t v, d;
		const uint8_t ND_SQ = ND * ND;
		const uint8_t MAX_LOCAL_ITER_NUM = 5;
		const index_t *rows_loc = local_jacobian->get_rows_ptr();
		const index_t *cols_loc = local_jacobian->get_cols_ind();
		value_t *Jac_loc = local_jacobian->get_values();
		index_t cell_id, st_id_loc, st_cell_id, id, r_code, iter = 0, conn_st_id, csr_idx_start, csr_idx_end;

		printf("contact iter #%d:\t res = %.10e\n", iter, res);
		while (res > 0.0000001 * init_res && iter < MAX_LOCAL_ITER_NUM)
		{
			// copy
			for (index_t i = 0; i < cell_ids.size(); i++)
			{
				cell_id = cell_ids[i];
				csr_idx_start = rows[cell_id];
				csr_idx_end = rows[cell_id + 1];
				st_id_loc = rows_loc[i];

				for (index_t st_id = csr_idx_start; st_id < csr_idx_end; st_id++)
				{
					st_cell_id = cell_ids[cols_loc[st_id_loc]];
					if (cols[st_id] == st_cell_id)
					{
						for (uint8_t d = 0; d < ND; d++)
							for (uint8_t v = 0; v < ND; v++)
								Jac_loc[st_id_loc * ND_SQ + d * ND + v] = Jac[st_id * N_VARS_SQ + (U_VAR + d) * N_VARS + U_VAR + v];

						st_id_loc++;
					}
				}
				assert(st_id_loc == rows_loc[i + 1]);
				std::copy_n(RHS.begin() + N_VARS * cell_id + U_VAR, ND, rhs_local.begin() + ND * i);
			}
			// solve
			fill_n(dg_local.begin(), dg_local.size(), 0.0);
			r_code = local_solver->setup(local_jacobian);
			if (r_code) { printf("ERROR: Local linear solver setup returned %d \n", r_code); }
			r_code = local_solver->solve(&rhs_local[0], &dg_local[0]);
			if (r_code) { printf("ERROR: Local linear solver solve returned %d \n", r_code); }

			/*if (1) //changed this to write jacobian to file!
			{
				//static_cast<csr_matrix<4>*>(Jacobian)->write_matrix_to_file_mm(("jac_nc_dar_" + std::to_string(output_counter) + ".csr").c_str());
				local_jacobian->write_matrix_to_file(("loc_jac_nc_dar_" + std::to_string(output_counter) + ".csr").c_str());
				write_vector_to_file("loc_jac_nc_dar_" + std::to_string(output_counter) + ".rhs", rhs_local);
				write_vector_to_file("loc_jac_nc_dar_" + std::to_string(output_counter) + ".sol", dg_local);
				//apply_newton_update(deltat);
				//write_vector_to_file("X_nc_dar", X);
				//write_vector_to_file("Xn_nc_dar", Xn);
				vector<value_t> buf(rhs_local.size(), 0.0);
				local_jacobian->matrix_vector_product(dg_local.data(), buf.data());
				std::transform(buf.begin(), buf.end(), rhs_local.begin(), buf.begin(), std::minus<double>());
				write_vector_to_file("diff_" + std::to_string(output_counter) + ".txt", buf);
				//exit(0);
				//return 0;
				output_counter++;
			}*/

			// update dX
			for (index_t i = 0; i < cell_ids.size(); i++)
			{
				cell_id = cell_ids[i];
				for (uint8_t d = 0; d < ND; d++)
					X[N_VARS * cell_id + U_VAR + d] -= dg_local[ND * i + d];

				// update fluxes
				const auto& conn_ids = mesh->fault_conn_id[i];
				for (uint8_t k = 0; k < conn_ids.size(); k++)
				{
					const auto& conn_id = conn_ids[k];
					for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
					{
						auto it = std::find(cell_ids.begin(), cell_ids.end(), stencil[conn_st_id]);
						if (it != cell_ids.end())
						{
							id = std::distance(cell_ids.begin(), it);
							for (d = 0; d < ND; d++)
							{
								for (v = 0; v < ND; v++)
								{
									fluxes[N_VARS * conn_id + U_VAR + d] -= tran[conn_st_id * N_VARS_SQ + (U_VAR + d) * N_VARS + (U_VAR + v)] * dg_local[ND * id + v];
									fluxes_biot[N_VARS * conn_id + U_VAR + d] -= tran_biot[conn_st_id * N_VARS_SQ + (U_VAR + d) * N_VARS + (U_VAR + v)] * dg_local[ND * id + v];
								}
							}
						}
					}
				}
			}
			// update jacobian
			add_to_jacobian_return_mapping(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
			// calculate residual
			res = calc_gap_L2_residual(RHS);
			iter++;
			printf("contact iter #%d:\t res = %.10e\n", iter, res);
		}
	}
	return 0;
}
int contact::solve_explicit_scheme(std::vector<value_t>& RHS, std::vector<value_t>& dX)
{
	bool res;
	uint8_t d, c;
	index_t l_id;

	for (index_t i = 0; i < cell_ids.size(); i++)
	{
		auto& jac = jacobian_explicit_scheme[i];
		res = jac.inv();
		if (!res) { cout << "Inversion failed!\n"; exit(-1); }

		const auto& cell_id = cell_ids[i];
		l_id = cell_id * N_VARS + U_VAR;
		for (d = 0; d < ND; d++)
		{
			dX[l_id + d] = 0.0;
			for (c = 0; c < ND; c++)
				dX[l_id + d] += jac(d, c) * RHS[l_id + c];
		}
	}
	return 0;
}
int contact::add_to_jacobian_slip(index_t id, value_t dt, vector<value_t>& RHS)
{
	const auto& F = pre_F[st.size()];
	const auto& Fpres = pre_Fpres[st.size()];
	const index_t cell_id = cell_ids[id];

	// assemble jacobian
	const index_t csr_idx_start = rows[cell_id];
	const index_t csr_idx_end = rows[cell_id + 1];
	index_t conn_st_id = 0, st_id;
	uint8_t d, v;
	fill_n(RHS.begin() + N_VARS * cell_id + U_VAR, ND, 0.0);

	for (st_id = csr_idx_start; conn_st_id < ind.size() && st_id < csr_idx_end; st_id++)
	{
		if (st[ind[conn_st_id]] == cols[st_id])
		{
			for (d = 0; d < ND; d++)
			{
				fill_n(Jac + N_VARS_SQ * st_id + (U_VAR + permut[d]) * N_VARS, N_VARS, 0.0);
				for (v = 0; v < ND; v++)
				{
					Jac[st_id * N_VARS_SQ + (U_VAR + permut[d]) * N_VARS + U_VAR + v] += implicit_scheme_multiplier * F.values[d * F.N + ind[conn_st_id] * ND + v];
				}
				Jac[st_id * N_VARS_SQ + (U_VAR + permut[d]) * N_VARS + ND] += implicit_scheme_multiplier * Fpres.values[d * Fpres.N + ind[conn_st_id]];
			}
			conn_st_id++;
		}
	}
	// assemble RHS
	for (d = 0; d < ND; d++)
	{
		RHS[cell_id * N_VARS + U_VAR + permut[d]] += (flux.values[d] - Frhs.values[d]);
	}

	return 0;
}
int contact::add_to_jacobian_stuck(index_t id, value_t dt, vector<value_t>& RHS)
{
	index_t conn_st_id = 0, st_id;
	uint8_t d;
	const index_t cell_id = cell_ids[id];
	const index_t csr_idx_start = rows[cell_id];
	const index_t csr_idx_end = rows[cell_id + 1];
	fill_n(RHS.begin() + N_VARS * cell_id + U_VAR, ND, 0.0);

	for (st_id = csr_idx_start; conn_st_id < ind.size() && st_id < csr_idx_end; st_id++)
	{
		for (d = 0; d < ND; d++)
		{
			fill_n(Jac + N_VARS_SQ * st_id + (U_VAR + d) * N_VARS, N_VARS, 0.0);
		}
	}

	diag_idx = N_VARS_SQ * diag_ind[cell_id];
	for (d = 0; d < ND; d++)
	{
		Jac[diag_idx + (U_VAR + d) * N_VARS + (U_VAR + d)] = implicit_scheme_multiplier * eps_t[id];
		jacobian_explicit_scheme[id](d, d) = eps_t[id];
		RHS[N_VARS * cell_id + U_VAR + d] = eps_t[id] * dg.values[d];
	}
	return 0;
}
int contact::apply_direction_chop(const std::vector<value_t>& X, const std::vector<value_t>& Xn, std::vector<value_t>& dX)
{
	size_t id;
	index_t cell_id;
	std::pair<bool, size_t> res;
	bool do_chop = false;
	value_t sign, sign_trial, terzaghi_coef, dgt_iter_norm, max_wrong_slip;
	Matrix dg(ND, 1), dg_new(ND, 1), dg_iter(ND, 1), dg_iter_chop(ND, 1), n(ND, 1), new_flux(ND, 1);

	if (friction_criterion == TERZAGHI)
		terzaghi_coef = 1.0;
	else
		terzaghi_coef = 0.0;

	max_wrong_slip = 0.0;
	for (index_t i = 0; i < cell_ids.size(); i++)
	{
		cell_id = cell_ids[i];
		// set gap change
		for (uint8_t d = 0; d < ND; d++)
		{
			dg.values[d] = X[N_VARS * cell_id + U_VAR + d] - Xn[N_VARS * cell_id + U_VAR + d];
			dg_new.values[d] = X[N_VARS * cell_id + U_VAR + d] - Xn[N_VARS * cell_id + U_VAR + d] - dX[N_VARS * cell_id + U_VAR + d];
			dg_iter.values[d] = -dX[N_VARS * cell_id + U_VAR + d];
		}

		const auto& S_cur = S[i];
		const auto& Sinv_cur = Sinv[i];
		dg_iter.values = (S_cur * dg_iter).values;
		dgt_iter_norm = sqrt(dg_iter(1, 0) * dg_iter(1, 0) + dg_iter(2, 0) * dg_iter(2, 0));

		if (dgt_iter_norm > 10000.0 * EQUALITY_TOLERANCE)
		{
			// copy current fault stress
			std::copy_n(fault_stress.begin() + (size_t)(ND * i), (size_t)ND, std::begin(flux.values));
			new_flux = flux;

			// calculate new fault stress
			const auto& conn_ids = mesh->fault_conn_id[cell_id - n_matrix];
			n = discr->faces[cell_id].back().n * discr->faces[cell_id].back().area;
			if ((n.transpose() * (discr->cell_centers[mesh->block_p[conn_ids[0]]] - discr->cell_centers[mesh->block_m[conn_ids[0]]])).values[0] < 0.0) n = -n;
			sign = discr->get_fault_sign(n, cell_ids[0]);// / 2.0;
			for (uint8_t k = 0; k < 1/*conn_ids.size()*/; k++)
			{
				const auto& conn_id = conn_ids[k];
				if (k == 1) sign = -sign;
				for (index_t conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
				{
					if (stencil[conn_st_id] < n_blocks)
					{
						for (uint8_t d = 0; d < ND; d++)
						{
							for (uint8_t v = 0; v < ND; v++)
							{
								new_flux.values[d] += -sign * dX[N_VARS * stencil[conn_st_id] + U_VAR + v] *
									(tran[conn_st_id * NT_SQ + (U_VAR_T + d) * NT + (U_VAR_T + v)] + terzaghi_coef * tran_biot[conn_st_id * NT + (U_VAR_T + d) * NT + (U_VAR_T + v)]);
							}
							new_flux.values[d] += -sign * dX[N_VARS * stencil[conn_st_id] + P_VAR] *
								(tran[conn_st_id * NT_SQ + (U_VAR_T + d) * NT + P_VAR_T] + terzaghi_coef * tran_biot[conn_st_id * NT + (U_VAR_T + d) * NT + P_VAR_T]);
						}
					}
				}
				new_flux -= -terzaghi_coef * sign * n * dX[N_VARS * cell_id + P_VAR];
			}

			// check & chop slip
			const auto& n_ref = discr->faces[cell_id].back().n;
			n = discr->get_fault_sign(n_ref, cell_ids[0]) * n_ref;
			sign_trial = -discr->get_fault_sign(n, cell_ids[0]);
			assert(sign_trial < 0);
			flux.values = (S_cur * flux).values;
			new_flux.values = (S_cur * new_flux).values;
			dg.values = (S_cur * dg).values;
			dg_new.values = (S_cur * dg_new).values;
			dg.values[0] = dg_new.values[0] = flux.values[0] = new_flux.values[0] = 0.0;
			assert(sign_trial * (flux.transpose() * dg).values[0] >= 0.0);
			// if new gap vector is opposite to new traction than chop slip
			if (sign_trial * (new_flux.transpose() * dg_new).values[0] < 0.0)
			{
				do_chop = true;
				if (dgt_iter_norm > max_wrong_slip) max_wrong_slip = dgt_iter_norm;

				// dg_iter.values[1] = dg_iter.values[2] = 0.0;
				// dg_iter.values[1] *= -1.0;
				// dg_iter.values[2] *= -1.0;
				// dg_iter_chop = Sinv_cur * dg_iter;
				// chop slip
				//for (uint8_t d = 0; d < ND; d++)
				// {
				//	dX[N_VARS * cell_id + U_VAR + d] = -dg_iter_chop.values[d];
				// }
				// printf("Chop slip in %d fault cell\n", cell_id);
			}
		}
	}

	// apply global chop
	if (do_chop)
	{
		/*for (index_t i = 0; i < cell_ids.size(); i++)
		{
			cell_id = cell_ids[i];
			// set gap change
			for (uint8_t d = 0; d < ND; d++)
				dg_iter.values[d] = -dX[N_VARS * cell_id + U_VAR + d];

			const auto& S_cur = S[i];
			const auto& Sinv_cur = Sinv[i];
			dg_iter.values = (S_cur * dg_iter).values;
			dg_iter.values[1] *= 0.01;//1.0 / max_wrong_slip / eps_t[i];
			dg_iter.values[2] *= 0.01;//1.0 / max_wrong_slip / eps_t[i];
			dg_iter_chop = Sinv_cur * dg_iter;		
			for (uint8_t d = 0; d < ND; d++)
				dX[N_VARS * cell_id + U_VAR + d] = -dg_iter_chop.values[d];
		}*/
		//for (index_t i = 0; i < dX.size(); i++)
		//	dX[i] *= 0.01;

		//printf("Global chop for slip \n");
	}

	return 0;
}

vector<value_t> contact::getFrictionCoef(const index_t i, const value_t dt, Matrix slip_vel, const Matrix& slip)
{
	value_t mu_cur, numer, denom;
	Matrix dmu(ND, 1), dnumer(ND, 1), dinvdenom(ND, 1);
	const index_t cell_id = cell_ids[i];
	const value_t slip_vel_norm = sqrt(slip_vel(1, 0) * slip_vel(1, 0) + slip_vel(2, 0) * slip_vel(2, 0));
	
	dmu.values = 0.0;
	if (friction_model == FRICTIONLESS)	
		mu_cur = 0.0;
	else if (friction_model == STATIC)
		mu_cur = mu0[i];
	else if (friction_model == SLIP_DEPENDENT)
	{
		const value_t slip_norm = sqrt(slip(1, 0) * slip(1, 0) + slip(2, 0) * slip(2, 0));
		mu_cur = mu0[i];
		if (slip_norm > EQUALITY_TOLERANCE && slip_norm <= sd_props.Dc)
		{
			mu_cur = mu0[i] + (sd_props.mu_d - mu0[i]) * slip_norm / sd_props.Dc;
			dmu.values += (sd_props.mu_d - mu0[i]) / sd_props.Dc * slip.values / slip_norm;
		}
		else if (slip_norm > sd_props.Dc)
		{
			mu_cur = sd_props.mu_d;
		}
	}
	else if (friction_model == RSF)
	{
		mu_cur = mu0[i];
		if (slip_vel_norm > EQUALITY_TOLERANCE || slip_vel_norm > 0.0001 * rsf.vel0)
		{
			rsf.mu_rate[i] = rsf.a * log(slip_vel_norm / rsf.vel0);
			mu_cur += rsf.mu_rate[i];
			dmu.values += rsf.a * slip_vel.values / dt / slip_vel_norm / slip_vel_norm;
			jacobian_explicit_scheme[i](0, { ND }, { 1 }) += rsf.a * slip_vel.values / dt / slip_vel_norm / slip_vel_norm;

			if ( (rsf.law == MIXED && slip_vel_norm > 0.01 * rsf.vel0) || rsf.law == SLIP_LAW )	// slip law
			{
				numer = rsf.Dc / slip_vel_norm * log(rsf.vel0 / rsf.Dc * rsf.theta_n[i]) + dt * log(rsf.vel0 / slip_vel_norm);
				denom = rsf.Dc / slip_vel_norm + dt;
				rsf.mu_state[i] = rsf.b * numer / denom;
				mu_cur += rsf.mu_state[i];
				// new state
				rsf.theta[i] = exp((rsf.Dc / slip_vel_norm * log(rsf.theta_n[i]) + dt * log(rsf.Dc / slip_vel_norm)) / denom);
				// derivative
				dnumer = -slip_vel / slip_vel_norm / slip_vel_norm * (1.0 + rsf.Dc / dt / slip_vel_norm * log(rsf.vel0 * rsf.theta_n[i] / rsf.Dc));
				dinvdenom = rsf.Dc * slip_vel / dt / slip_vel_norm / (rsf.Dc + dt * slip_vel_norm);
				dmu.values += rsf.b * (dnumer.values / denom + numer * dinvdenom.values);
			}
			else																				// ageing law
			{
				rsf.mu_state[i] = rsf.b * log(rsf.vel0 * (dt + rsf.theta_n[i]) / (rsf.Dc + slip_vel_norm * dt));
				mu_cur += rsf.mu_state[i];
				// new state
				rsf.theta[i] = rsf.Dc * (dt + rsf.theta_n[i]) / (rsf.Dc + dt * slip_vel_norm);
				// derivative
				dmu.values += -rsf.b * slip_vel.values / slip_vel_norm / (rsf.Dc + slip_vel_norm * dt);
			}
		}
	}

	return { mu_cur, 0.0, dmu.values[1], dmu.values[2] };
}
vector<value_t> contact::getStabilizedFrictionCoef(const index_t i, const value_t dt, Matrix slip_vel, const Matrix& slip)
{
	value_t mu_cur, numer, denom, tmp;
	Matrix dmu(ND, 1), dnumer(ND, 1), dinvdenom(ND, 1);
	const index_t cell_id = cell_ids[i];
	bool min_vel_limit = false;
	value_t slip_vel_norm = sqrt(slip_vel(1, 0) * slip_vel(1, 0) + slip_vel(2, 0) * slip_vel(2, 0));
	if (slip_vel_norm < rsf.min_vel)
	{
		//slip_vel.values[1] = rsf.min_vel;
		//min_vel_limit = true;
		slip_vel_norm = rsf.min_vel;// sqrt(slip_vel(1, 0) * slip_vel(1, 0) + slip_vel(2, 0) * slip_vel(2, 0));
	}

	dmu.values = 0.0;
	if (friction_model == FRICTIONLESS)
		mu_cur = 0.0;
	else if (friction_model == STATIC)
		mu_cur = mu0[i];
	else if (friction_model == SLIP_DEPENDENT)
	{
		const value_t slip_norm = sqrt(slip(1, 0) * slip(1, 0) + slip(2, 0) * slip(2, 0));
		mu_cur = mu0[i];
		if (slip_norm > EQUALITY_TOLERANCE && slip_norm <= sd_props.Dc)
		{
			mu_cur = mu0[i] + (sd_props.mu_d - mu0[i]) * slip_norm / sd_props.Dc;
			dmu.values += (sd_props.mu_d - mu0[i]) / sd_props.Dc * slip.values / slip_norm;
		}
		else if (slip_norm > sd_props.Dc)
		{
			mu_cur = sd_props.mu_d;
		}
	}
	else if (friction_model == RSF_STAB)
	{
		if ((rsf.law == MIXED && slip_vel_norm > 0.01 * rsf.vel0) || rsf.law == SLIP_LAW)	// slip law
		{
			numer = rsf.Dc / slip_vel_norm * log(rsf.vel0 / rsf.Dc * rsf.theta_n[i]) + dt * log(rsf.vel0 / slip_vel_norm);
			denom = rsf.Dc / slip_vel_norm + dt;
			// new state
			rsf.theta[i] = exp((rsf.Dc / slip_vel_norm * log(rsf.theta_n[i]) + dt * log(rsf.Dc / slip_vel_norm)) / denom);
			// derivative
			dnumer = -slip_vel / slip_vel_norm / slip_vel_norm * (1.0 + rsf.Dc / dt / slip_vel_norm * log(rsf.vel0 * rsf.theta_n[i] / rsf.Dc));
			dinvdenom = rsf.Dc * slip_vel / dt / slip_vel_norm / (rsf.Dc + dt * slip_vel_norm);
		}
		else																				// ageing law
		{
			numer = log(rsf.vel0 * (dt + rsf.theta_n[i]) / (rsf.Dc + slip_vel_norm * dt));
			denom = 1.0;
			// new state
			rsf.theta[i] = rsf.Dc * (dt + rsf.theta_n[i]) / (rsf.Dc + dt * slip_vel_norm);
			// derivative
			if (min_vel_limit)
				dnumer.values = 0.0;
			else
				dnumer.values = -slip_vel.values / slip_vel_norm / (rsf.Dc + slip_vel_norm * dt);
			dinvdenom.values = 0.0;
		}
		tmp = exp((mu0[i] + rsf.b * numer / denom) / rsf.a) / rsf.vel0 / 2.0;
		mu_cur = rsf.a * asinh(slip_vel_norm * tmp);
		dmu = slip_vel_norm * rsf.b * (dnumer / denom + numer * dinvdenom);
		if (!min_vel_limit)
			dmu += slip_vel / slip_vel_norm / dt;
		dmu.values *= tmp / sqrt(1.0 + (slip_vel_norm * tmp) * (slip_vel_norm * tmp));
	}

	return { mu_cur, 0.0, dmu.values[1], dmu.values[2] };
}

// for local iterations
value_t contact::calc_gap_L2_residual(const vector<value_t>& RHS) const
{
	value_t res = 0.0, gap_vol = 0.0;
	index_t cell_id;
	for (index_t i = 0; i < cell_ids.size(); i++)
	{
		cell_id = cell_ids[i];
		for (uint8_t d = 0; d < ND; d++)
			res += RHS[N_VARS * cell_id + U_VAR + d] * RHS[N_VARS * cell_id + U_VAR + d];
		gap_vol += 1.0;// mesh->volume[cell_id] * mesh->volume[cell_id];
	}
	return sqrt(res / gap_vol);
}
int contact::init_local_jacobian_structure()
{
	// init Jacobian structure
	index_t *rows_ptr = local_jacobian->get_rows_ptr();
	index_t *diag_ind = local_jacobian->get_diag_ind();
	index_t *cols_ind = local_jacobian->get_cols_ind();
	index_t *row_thread_starts = local_jacobian->get_row_thread_starts();

	index_t cell_id, id;
	vector<index_t> &block_m = mesh->block_m;
	vector<index_t> &block_p = mesh->block_p;

	rows_ptr[0] = 0;
	memset(diag_ind, -1, cell_ids.size() * sizeof(index_t)); // t_long <-----> index_t
	for (index_t i = 0; i < cell_ids.size(); i++)
	{
		cell_id = cell_ids[i];
		const auto &cur = mesh->cell_stencil[cell_id];
		rows_ptr[i + 1] = rows_ptr[i];
		for (const auto& st : cur)
		{
			auto it = std::find(cell_ids.begin(), cell_ids.end(), st);
			if (it != cell_ids.end())
			{
				id = std::distance(cell_ids.begin(), it);
				cols_ind[rows_ptr[i + 1]++] = id;
			}
		}
		diag_ind[i] = index_t(intptr_t(std::find(cols_ind + rows_ptr[i], cols_ind + rows_ptr[i + 1], cell_id) - cols_ind - rows_ptr[i]));
	}

	return 0;
}
