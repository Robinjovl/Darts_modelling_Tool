#include "mech/mech_operators.hpp"

using namespace pm;
using std::vector;
using std::array;
using std::fill_n;
using std::begin;
using std::end;
using std::copy_n;

mech_operators::mech_operators() 
{
}
mech_operators::~mech_operators()
{}
void mech_operators::init(conn_mesh* _mesh, pm_discretizer* _discr, uint8_t _P_VAR, uint8_t _Z_VAR, uint8_t _U_VAR,
							uint8_t _N_VARS, uint8_t _N_OPS, uint8_t _NC, uint8_t _ACC_OP, uint8_t _FLUX_OP, uint8_t _GRAV_OP)
{
	mesh = _mesh;
	discr = _discr;
	P_VAR = _P_VAR;	P_VAR_T = _P_VAR;
	Z_VAR = _Z_VAR;
	U_VAR = _U_VAR; U_VAR_T = _U_VAR;
	N_VARS = _N_VARS;
	N_VARS_SQ = _N_VARS * _N_VARS;
	N_OPS = _N_OPS;
	NC = _NC;
	ACC_OP = _ACC_OP;
	FLUX_OP = _FLUX_OP;
	GRAV_OP = _GRAV_OP;

	// Preallocations for different number of cell faces
	for (uint8_t fn = pm_discretizer::MIN_FACE_NUM; fn <= pm_discretizer::MAX_FACE_NUM; fn++)
	{
		pre_N[fn] = Matrix(fn * ND, SUM_N(ND));
		pre_Nflux[fn] = Matrix(fn, ND);
		pre_R[fn] = Matrix(fn * ND, SUM_N(ND));
		pre_Ft[fn] = Matrix(fn * ND, 1);
		pre_F[fn] = Matrix(fn * ND, 1);
		pre_Q[fn] = Matrix(fn, 1);
	}

	pressures.resize(mesh->n_matrix);
	porosities.resize(mesh->n_matrix);
	eps_vol.resize(mesh->n_matrix);
	stresses.resize(SUM_N(ND) * mesh->n_matrix, 0.0);
	total_stresses.resize(SUM_N(ND) * mesh->n_matrix, 0.0);
	velocity.resize(ND * mesh->n_matrix, 0.0);
	face_unknowns.resize(N_VARS * mesh->n_conns, 0.0);
	for (index_t i = 0; i < mesh->n_matrix; i++)
	{
		const auto& cur_faces = discr->faces[i];
		pressures[i].resize(cur_faces.size(), 0.0);
	}
}
void mech_operators::init(conn_mesh* _mesh, pm_discretizer* _discr, uint8_t _P_VAR, uint8_t _Z_VAR, uint8_t _U_VAR, uint8_t _P_VAR_T, uint8_t _U_VAR_T, 
	uint8_t _N_VARS, uint8_t _N_OPS, uint8_t _NC, uint8_t _ACC_OP, uint8_t _FLUX_OP, uint8_t _GRAV_OP)
{
	mesh = _mesh;
	discr = _discr;
	P_VAR = _P_VAR;	P_VAR_T = _P_VAR_T;
	Z_VAR = _Z_VAR;
	U_VAR = _U_VAR; U_VAR_T = _U_VAR_T;
	N_VARS = _N_VARS;
	N_VARS_SQ = _N_VARS * _N_VARS;
	N_OPS = _N_OPS;
	NC = _NC;
	ACC_OP = _ACC_OP;
	FLUX_OP = _FLUX_OP;
	GRAV_OP = _GRAV_OP;

	// Preallocations for different number of cell faces
	for (uint8_t fn = pm_discretizer::MIN_FACE_NUM; fn <= pm_discretizer::MAX_FACE_NUM; fn++)
	{
		pre_N[fn] = Matrix(fn * ND, SUM_N(ND));
		pre_Nflux[fn] = Matrix(fn, ND);
		pre_R[fn] = Matrix(fn * ND, SUM_N(ND));
		pre_Ft[fn] = Matrix(fn * ND, 1);
		pre_F[fn] = Matrix(fn * ND, 1);
		pre_Q[fn] = Matrix(fn, 1);
	}

	pressures.resize(mesh->n_matrix);
	porosities.resize(mesh->n_matrix);
	eps_vol.resize(mesh->n_matrix);
	stresses.resize(SUM_N(ND) * mesh->n_matrix, 0.0);
	total_stresses.resize(SUM_N(ND) * mesh->n_matrix, 0.0);
	velocity.resize(ND * mesh->n_matrix, 0.0);
	face_unknowns.resize(N_VARS * mesh->n_conns, 0.0);
	for (index_t i = 0; i < mesh->n_matrix; i++)
	{
		const auto& cur_faces = discr->faces[i];
		pressures[i].resize(cur_faces.size(), 0.0);
	}
}

void mech_operators::prepare()
{
	const index_t n_res_blocks = mesh->n_res_blocks;
	const index_t n_matrix = mesh->n_matrix;
	auto& faces = discr->faces;
	index_t face_id, counter, n_faces;
	bool res;

	Matrix t_face(ND, 1), n(ND, 1), sq_mat_flux(ND, ND), sq_mat_stress(SUM_N(ND), SUM_N(ND));
	Matrix Ndelta(ND, SUM_N(ND)), Rdelta(ND, SUM_N(ND));

	for (index_t i = 0; i < n_matrix; i++)
	{
		const auto& cur_faces = faces[i];
		n_faces = 0;
		for (const auto& face : cur_faces) { if (face.type != MAT_TO_FRAC) n_faces++; }
		auto& N = pre_N[n_faces];
		auto& Nflux = pre_Nflux[n_faces];
		auto& R = pre_R[n_faces];
		fill_n(begin(N.values), N.values.size(), 0.0);
		fill_n(begin(Nflux.values), Nflux.values.size(), 0.0);
		fill_n(begin(R.values), R.values.size(), 0.0);

		counter = 0;
		for (face_id = 0; face_id < cur_faces.size(); face_id++)
		{
			const auto& face = cur_faces[face_id];
			if (face.type == MAT_TO_FRAC) { face_id++; continue; }
			t_face = face.c - discr->cell_centers[i];
			n = (face.n.transpose() * t_face).values[0] > 0 ? face.n : -face.n;

			// N-matrix
			Ndelta(0, 0) = n(0, 0);				Ndelta(1, 1) = n(1, 0);			Ndelta(2, 2) = n(2, 0);
			Ndelta(1, ND + 2) = n(0, 0);		Ndelta(2, ND + 1) = n(0, 0);
			Ndelta(0, ND + 2) = n(1, 0);		Ndelta(2, ND) = n(1, 0);
			Ndelta(1, ND) = n(2, 0);		Ndelta(0, ND + 1) = n(2, 0);
			N(counter * Ndelta.values.size(), { ND, SUM_N(ND) }, { SUM_N(ND), 1 }) = face.area * Ndelta.values;
			// N-matrix for fluxes
			Nflux(counter * ND, { ND }, { 1 }) = face.area * n.values;
			// R-matrix
			Rdelta(0, 0) = t_face(0, 0);		Rdelta(1, 1) = t_face(1, 0);	Rdelta(2, 2) = t_face(2, 0);
			Rdelta(1, ND + 2) = t_face(0, 0) / 2;	Rdelta(2, ND + 1) = t_face(0, 0) / 2;
			Rdelta(0, ND + 2) = t_face(1, 0) / 2;	Rdelta(2, ND) = t_face(1, 0) / 2;
			Rdelta(1, ND) = t_face(2, 0) / 2;	Rdelta(0, ND + 1) = t_face(2, 0) / 2;
			R(counter * Rdelta.values.size(), { ND, SUM_N(ND) }, { SUM_N(ND), 1 }) = Rdelta.values;

			counter++;
		}
		// stress reconstruction
		sq_mat_stress = R.transpose() * N;
		res = sq_mat_stress.inv();
		if (!res) { std::cout << "Inversion failed!\n";	exit(-1); };
		mat_stress.push_back(sq_mat_stress * R.transpose());
		// flux reconstruction
		sq_mat_flux = Nflux.transpose() * Nflux;
		res = sq_mat_flux.inv();
		if (!res) { std::cout << "Inversion failed!\n";	exit(-1); };
		mat_flux.push_back(sq_mat_flux * Nflux.transpose());
	}
}
void mech_operators::eval_stresses(const vector<value_t>& fluxes, const vector<value_t>& fluxes_biot, vector<value_t>& X, vector<value_t>& bc_rhs, const vector<value_t>& op_vals_arr)
{
	const index_t* block_m = mesh->block_m.data();
	const index_t* block_p = mesh->block_p.data();
	value_t* bc = bc_rhs.data();
	value_t* bc_ref = mesh->bc_ref.data();
	const value_t* p_ref = mesh->ref_pressure.data();
	const value_t* pz_bounds = mesh->pz_bounds.data();
	const index_t n_matrix = mesh->n_matrix;
	const index_t n_res_blocks = mesh->n_res_blocks;
	const index_t n_blocks = mesh->n_blocks;
	const index_t n_conns = mesh->n_conns;
	const index_t n_bounds = mesh->n_bounds;

	Matrix w(SUM_N(ND), 1), Ndelta(ND, SUM_N(ND));
	auto& faces = discr->faces;
	uint8_t face_id, counter, n_faces;

	vector<value_t> flux(ND + NC);
	index_t j, d, conn_id = 0, conn_st_id = 0, st_id = 0;
	value_t p_face, p_face_ref;

	Matrix dx(ND, 1), grad_vec(ND, 1), ref_grad_vec(ND, 1), Xmat(NT, 1), Xref(NT, 1), n(ND, 1), t_face(ND, 1),
		traction(ND, 1), stress(SUM_N(ND), 1), total_stress(SUM_N(ND), 1), vel(ND, 1);
	for (index_t i = 0; i < n_matrix; i++)
	{
		const auto& cur_faces = faces[i];
		n_faces = 0;
		for (const auto& face : cur_faces) { if (face.type != MAT_TO_FRAC) n_faces++; }
		const auto& b = discr->biots[i];
		auto& F = pre_F[n_faces];
		auto& Ft = pre_Ft[n_faces];
		auto& Q = pre_Q[n_faces];
		fill_n(begin(F.values), F.values.size(), 0.0);
		fill_n(begin(Ft.values), Ft.values.size(), 0.0);
		fill_n(begin(Q.values), Q.values.size(), 0.0);

		counter = 0;
		for (face_id = 0; block_m[conn_id] == i && conn_id < n_conns;)
		{
			j = block_p[conn_id];
			// skip connection
			if (j >= n_matrix && j < n_blocks) { conn_id++;  continue; }
			const auto& face = cur_faces[face_id];
			assert(j == face.cell_id2 || (face.type == BORDER && j == n_blocks + face.face_id2));
			// skip face
			if (face.type == MAT_TO_FRAC) { face_id++; continue; }

			fill_n(flux.begin(), NT, 0.0);

			t_face = face.c - discr->cell_centers[i];
			n = (face.n.transpose() * t_face).values[0] > 0 ? face.n : -face.n;

			const auto& grad = discr->grad[i];
			dx = face.c - discr->cell_centers[i];
			if (Z_VAR < 255)
			{
				grad_vec.values = (op_vals_arr[i * N_OPS + GRAV_OP] * X[i * N_VARS + Z_VAR]
					+ op_vals_arr[i * N_OPS + GRAV_OP + 1] * (1.0 - X[i * N_VARS + Z_VAR])) * grad.rhs(ND * ND, { 3 }, { 1 });
				ref_grad_vec.values = (op_vals_arr[i * N_OPS + GRAV_OP] * X[i * N_VARS + Z_VAR]
					+ op_vals_arr[i * N_OPS + GRAV_OP + 1] * (1.0 - X[i * N_VARS + Z_VAR])) * grad.rhs(ND * ND, { 3 }, { 1 });
			}
			else
			{
				grad_vec.values = op_vals_arr[i * N_OPS + GRAV_OP] * grad.rhs(ND * ND, { 3 }, { 1 });
				ref_grad_vec.values = op_vals_arr[i * N_OPS + GRAV_OP] * grad.rhs(ND * ND, { 3 }, { 1 });
			}
			p_face = X[i * N_VARS + P_VAR];
			p_face_ref = p_ref[i];
			for (d = 0; d < grad.stencil.size(); d++)
			{
				if (grad.stencil[d] < n_res_blocks)
				{
					Xmat.values = { X[grad.stencil[d] * N_VARS + U_VAR + 0],
									X[grad.stencil[d] * N_VARS + U_VAR + 1],
									X[grad.stencil[d] * N_VARS + U_VAR + 2],
									X[grad.stencil[d] * N_VARS + P_VAR] };
					Xref.values = { 0.0,
									0.0,
									0.0,
									p_ref[grad.stencil[d]] };
				}
				else
				{
					const value_t *cur_bc = &bc[NT * (grad.stencil[d] - n_res_blocks)];
					Xmat.values = { cur_bc[0], cur_bc[1], cur_bc[2], cur_bc[3] };
					const value_t *ref_bc = &bc_ref[NT * (grad.stencil[d] - n_res_blocks)];
					Xref.values = { ref_bc[0], ref_bc[1], ref_bc[2], ref_bc[3] };
				}
				grad_vec += Matrix(grad.mat(ND * ND * (uint8_t)(grad.mat.N) + d * NT, { ND, NT }, { (uint8_t)(grad.mat.N) , 1 }), ND, NT) * Xmat;
				ref_grad_vec += Matrix(grad.mat(ND * ND * (uint8_t)(grad.mat.N) + d * NT, { ND, NT }, { (uint8_t)(grad.mat.N) , 1 }), ND, NT) * Xref;
			}
			p_face += (grad_vec.transpose() * dx).values[0];
			pressures[i][counter] = p_face;
			p_face_ref += (ref_grad_vec.transpose() * dx).values[0];
			traction.values = { -fluxes[conn_id * N_VARS + U_VAR] - fluxes_biot[conn_id * N_VARS + U_VAR],
								-fluxes[conn_id * N_VARS + U_VAR + 1] - fluxes_biot[conn_id * N_VARS + U_VAR + 1],
								-fluxes[conn_id * N_VARS + U_VAR + 2] - fluxes_biot[conn_id * N_VARS + U_VAR + 2] };
			w.values = { b(0, 0), b(1, 1), b(2, 2), b(1, 2), b(0, 2), b(0, 1) };
			// N-matrix
			Ndelta(0, 0) = n(0, 0);				Ndelta(1, 1) = n(1, 0);			Ndelta(2, 2) = n(2, 0);
			Ndelta(1, ND + 2) = n(0, 0);		Ndelta(2, ND + 1) = n(0, 0);
			Ndelta(0, ND + 2) = n(1, 0);		Ndelta(2, ND) = n(1, 0);
			Ndelta(1, ND) = n(2, 0);			Ndelta(0, ND + 1) = n(2, 0);

			Ft(counter * ND, { ND }, { 1 }) = traction.values;
			F(counter * ND, { ND }, { 1 }) = (traction + face.area * (p_face - p_face_ref) * Ndelta * w).values;
			Q.values[counter] = fluxes[conn_id * N_VARS + P_VAR];

			counter++;
			face_id++;
			conn_id++;
		}
		stress = mat_stress[i] * F;
		total_stress = mat_stress[i] * Ft;
		vel = mat_flux[i] * Q;
		copy_n(begin(stress.values), SUM_N(ND), begin(stresses) + SUM_N(ND) * i);
		copy_n(begin(total_stress.values), SUM_N(ND), begin(total_stresses) + SUM_N(ND) * i);
		copy_n(begin(vel.values), ND, begin(velocity) + ND * i);
	}

	// facet pressures are not continuous
	/*for (index_t i = 0; i < n_blocks; i++)
	{
		const auto& cur_faces = faces[i];
		for (face_id = 0; face_id < cur_faces.size(); face_id++)
		{
			const auto& face = cur_faces[face_id];
			if (face.type == MAT)
			{
				assert(fabs(pressures[i][face_id] - pressures[face.cell_id2][face.face_id2]) < 1.E-6);
			}
			else if (face.type == BORDER)
			{
				assert(face.cell_id1 == face.cell_id2);
				assert(fabs(pressures[i][face_id] - pz_bounds[NC * face.face_id2]) < 1.E-6);
			}
		}
	}*/
}
void mech_operators::eval_porosities(vector<value_t>& X, vector<value_t>& bc_rhs)
{
	const index_t* block_m = mesh->block_m.data();
	const index_t* block_p = mesh->block_p.data();
	const index_t* stencil = mesh->stencil.data();
	const index_t* offset = mesh->offset.data();
	const value_t* tran_biot = mesh->tran_biot.data();
	const value_t* rhs_biot = mesh->rhs_biot.data();
	const value_t* poro = mesh->poro.data();
	const value_t* cs = mesh->rock_compressibility.data();
	const value_t* p_ref = mesh->ref_pressure.data();
	const value_t* eps_vol_ref = mesh->ref_eps_vol.data();
	const value_t* V = mesh->volume.data();
	const value_t* bc = bc_rhs.data();
	const value_t* bc_ref = mesh->bc_ref.data();
	const value_t* pz_bounds = mesh->pz_bounds.data();
	const index_t n_res_blocks = mesh->n_res_blocks;
	const index_t n_matrix = mesh->n_matrix;
	const index_t n_blocks = mesh->n_blocks;
	const index_t n_conns = mesh->n_conns;
	const index_t n_bounds = mesh->n_bounds;
	const uint8_t N_BRHS = (n_bounds * NT == bc_rhs.size() ? NT : N_VARS);

	auto& faces = discr->faces;
	index_t i, j, v, conn_id = 0, conn_st_id = 0, st_id = 0, idx;
	value_t biot_mult, b, comp_mult;
	//vector<pm::Face>::const_iterator it;
	for (index_t i = 0; i < n_matrix; i++)
	{
		biot_mult = 0.0;
		const auto& cur_faces = faces[i];
		for (; block_m[conn_id] == i && conn_id < n_conns; conn_id++)
		{
			j = block_p[conn_id];
			if (j >= n_res_blocks && j < n_blocks) continue;
			conn_st_id = offset[conn_id];
			// Inner contribution
			for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1] && stencil[conn_st_id] < n_res_blocks; conn_st_id++)
			{
				// Biot term for accumulation
				for (v = 0; v < ND; v++)
				{
					biot_mult += tran_biot[conn_st_id * N_TRANS_SQ + NT * P_VAR_T + U_VAR_T + v] * X[stencil[conn_st_id] * N_VARS + U_VAR + v];
				}
				biot_mult += tran_biot[conn_st_id * N_TRANS_SQ + NT * P_VAR_T + P_VAR_T] * (X[stencil[conn_st_id] * N_VARS + P_VAR] - p_ref[stencil[conn_st_id]]);
			}
			// Boundary contribution
			for (; conn_st_id < offset[conn_id + 1]; conn_st_id++)
			{
				idx = N_BRHS * (stencil[conn_st_id] - n_blocks);
				const value_t* cur_bc = &bc[idx];
				const value_t* ref_bc = &bc_ref[idx];
				for (v = 0; v < ND; v++)
				{
					biot_mult += tran_biot[conn_st_id * N_TRANS_SQ + NT * P_VAR_T + U_VAR_T + v] * (cur_bc[U_VAR + v] - ref_bc[U_VAR + v]);
				}
				biot_mult += tran_biot[conn_st_id * N_TRANS_SQ + NT * P_VAR_T + P_VAR_T] * (cur_bc[P_VAR] - ref_bc[P_VAR]);
			}

			// Biot term for accumulation
			biot_mult += rhs_biot[conn_id * NT + P_VAR_T];
		}

		eps_vol[i] = biot_mult / V[i];
		porosities[i] = poro[i] + cs[i] * (X[i * N_VARS + P_VAR] - p_ref[i]) + (biot_mult / V[i] - eps_vol_ref[i]);
		if (porosities[i] < 0.0) porosities[i] = poro[i];
	}
}
void mech_operators::eval_unknowns_on_faces(vector<value_t>& X, vector<value_t>& bc_rhs, vector<value_t>& Xref)
{
	const index_t* block_m = mesh->block_m.data();
	const index_t* block_p = mesh->block_p.data();
	const index_t* stencil = mesh->stencil.data();
	const index_t* offset = mesh->offset.data();
	const value_t* tran_face = mesh->tran_face.data();
	const value_t* rhs_face = mesh->rhs_face.data();

	const value_t* bc = bc_rhs.data();
	const value_t* bc_ref = mesh->bc_ref.data();

	const index_t n_matrix = mesh->n_matrix;
	const index_t n_res_blocks = mesh->n_res_blocks;
	const index_t n_blocks = mesh->n_blocks;
	const index_t n_conns = mesh->n_conns;
	const index_t n_bounds = mesh->n_bounds;

	// assembly function
	auto assemble = [&](const index_t conn_id, const index_t conn_st_id)
	{
		index_t l_ind, r_ind1, r_ind2, idx = 0;
		value_t *data;

		if (stencil[conn_st_id] < n_res_blocks)
			data = X.data();
		else
		{
			data = bc_rhs.data();
			idx = n_blocks;
		}

		// displacements
		for (uint8_t i = 0; i < ND; i++)
		{
			l_ind = conn_id * N_VARS + U_VAR + i;
			r_ind1 = conn_st_id * N_TRANS_SQ + NT * (U_VAR_T + i) + U_VAR_T;
			r_ind2 = (stencil[conn_st_id] - idx) * N_VARS + U_VAR;
			for (uint8_t j = 0; j < ND; j++)
			{
				face_unknowns[l_ind] += tran_face[r_ind1 + j] * data[r_ind2 + j];
			}
			r_ind1 = conn_st_id * N_TRANS_SQ + NT * (U_VAR_T + i) + P_VAR_T;
			r_ind2 = (stencil[conn_st_id] - idx) * N_VARS + P_VAR;
			face_unknowns[l_ind] += tran_face[r_ind1] * data[r_ind2];
		}

		// pressure
		l_ind = conn_id * N_VARS + P_VAR;
		r_ind1 = conn_st_id * N_TRANS_SQ + NT * P_VAR_T + U_VAR_T;
		r_ind2 = (stencil[conn_st_id] - idx) * N_VARS + U_VAR;
		for (uint8_t j = 0; j < ND; j++)
		{
			face_unknowns[l_ind] += tran_face[r_ind1 + j] * data[r_ind2 + j];
		}
		r_ind1 = conn_st_id * N_TRANS_SQ + NT * P_VAR_T + P_VAR_T;
		r_ind2 = (stencil[conn_st_id] - idx) * N_VARS + P_VAR;
		face_unknowns[l_ind] += tran_face[r_ind1] * data[r_ind2];
	};

	for (index_t conn_id = 0; conn_id < n_conns; conn_id++)
	{
		fill_n(face_unknowns.data() + conn_id * NT, NT, 0.0);

		// contributions from cells and boundaries
		for (index_t conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
		{
			assemble(conn_id, conn_st_id);
		}

		// free terms
		for (uint8_t i = 0; i < NT; i++)
		{
			face_unknowns[conn_id * NT + i] += rhs_face[conn_id * NT + i];
		}
	}
}