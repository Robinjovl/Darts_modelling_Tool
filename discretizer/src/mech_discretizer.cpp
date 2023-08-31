#include "mech_discretizer.h"
#include <chrono>

using namespace dis;
using std::vector;
using std::cout;
using std::endl;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::fill_n;

template <MechDiscretizerMode MODE>
const uint8_t MechDiscretizer<MODE>::n_unknowns = N_UNKNOWNS.at(MODE);

template <MechDiscretizerMode MODE>
MechDiscretizer<MODE>::MechDiscretizer() : W(9, 6)
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

  NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION = true;
}

template <MechDiscretizerMode MODE>
MechDiscretizer<MODE>::~MechDiscretizer()
{
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::init()
{
  Discretizer::init();
  index_t face_id;

  inner.resize(mesh->region_ranges.at(mesh::MATRIX).second);

  for (index_t i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
  {
	face_id = 0;
	for (index_t j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++, face_id++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[j]];
	  if (conn.type == mesh::MAT_MAT)
	  {
		inner[i][face_id] = InnerMatrices();
		auto& cur = inner[i][face_id];

		cur.Q1 = Matrix(ND, ND);		  cur.Q2 = Matrix(ND, ND);
		cur.Th1 = Matrix(ND, ND * ND);	  cur.Th2 = Matrix(ND, ND * ND);
		cur.R1 = Matrix(ND, 1);			  cur.R2 = Matrix(ND, 1);
		cur.y1 = Matrix(ND, 1);			  cur.y2 = Matrix(ND, 1);
		cur.T1 = Matrix(ND, ND);		  cur.T2 = Matrix(ND, ND);
		cur.G1 = Matrix(ND, ND * ND);	  cur.G2 = Matrix(ND, ND * ND);
	  }
	}
  }

  for (index_t i = mesh::MIN_CONNS_PER_ELEM; i < mesh::MAX_CONNS_PER_ELEM; i++)
  {
	pre_grad_A_u[i] = Matrix(ND * i, ND * ND);
	pre_grad_R_u[i] = Matrix(ND * i, n_unknowns * MAX_STENCIL);
	pre_grad_rhs_u[i] = Matrix(ND * i, 1);

	for (index_t st_size = 1; st_size < MAX_STENCIL; st_size++)
	{
	  pre_cur_rhs[i][st_size] = Matrix(ND * i, n_unknowns * st_size);
	}
  }

  mech_fluxes.resize(MAX_FLUXES_NUM, MechApproximation(4, n_unknowns * MAX_STENCIL));
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::reconstruct_displacement_gradients_per_cell(const MechBoundaryCondition& _bc_mech)
{
  // Variables
  std::vector<index_t> st;		st.reserve(MAX_STENCIL);
  std::vector<index_t> admissible_connections(4, 0);
  Matrix n(ND, 1), conn_c(ND, 1), P(ND, ND), B1n(ND, 1), B2n(ND, 1), A1n(ND, 1), A2n(ND, 1), K1n(ND, 1), gam1(ND, 1), tmp(ND, 1);
  Vector3 n_vec, diff1, diff2;
  Matrix C1(ND * ND, ND * ND), C2(ND * ND, ND * ND), T1(ND, ND), T2(ND, ND), G1(ND, ND * ND), G2(ND, ND * ND);
  Matrix nblock(ND * ND, ND), nblock_t(ND, ND * ND), tblock(ND * ND, ND * ND);
  Matrix mult_p(ND, 1), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), An(ND, ND), At(ND, ND), L(ND, ND), y1(ND, 1), c1_mat(ND, 1);
  Matrix to_invert(ND * ND, ND * ND);
  value_t buf1, buf2, Ap, gamma, r1, lam1;
  index_t n_cur_faces, loop_face_id, face_id, conn_id, id1, id2, cur_cell_id;
  bool res;

  // allocate memory for arrays
  u_grad_stencil.reserve(mesh->num_of_elements * MAX_STENCIL);
  u_grad_offset.reserve(mesh->num_of_elements + 1);
  u_grad_vals.reserve(ND * ND * mesh->num_of_elements * MAX_STENCIL);
  u_grad_rhs.reserve(ND * ND * mesh->num_of_elements);

  bc_mech = _bc_mech;

  steady_clock::time_point t1, t2;
  t1 = steady_clock::now();

  // loop through the adjacency matrix (matrix cells)
  for (index_t i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
  {
	// Build the system from the continuity at the interfaces
	n_cur_faces = 0;
	for (index_t j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[j]];
	  if (conn.type == mesh::MAT_BOUND)
	  {
		// Coefficients that define boundary condition
		const auto& an = bc_mech.a_n[conn.elem_id2 - mesh->n_cells];
		const auto& bn = bc_mech.b_n[conn.elem_id2 - mesh->n_cells];
		const auto& at = bc_mech.a_t[conn.elem_id2 - mesh->n_cells];
		const auto& bt = bc_mech.b_t[conn.elem_id2 - mesh->n_cells];

		if (NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION || an != 0.0 || at != 0.0)	n_cur_faces++;
	  }
	  else if (conn.type != mesh::MAT_FRAC) n_cur_faces++;
	}

	auto& A = pre_grad_A_u[n_cur_faces];
	auto& rhs_mult = pre_grad_R_u[n_cur_faces];
	auto& rest = pre_grad_rhs_u[n_cur_faces];
	std::fill_n(&A.values[0], A.values.size(), 0.0);
	std::fill_n(&rest.values[0], rest.values.size(), 0.0);
	std::fill_n(&rhs_mult.values[0], rhs_mult.values.size(), 0.0);

	face_id = conn_id = 0;
	for (loop_face_id = mesh->adj_matrix_offset[i]; loop_face_id < mesh->adj_matrix_offset[i + 1]; loop_face_id++, conn_id++)
	{
	  st.clear();
	  const auto& conn = mesh->conns[mesh->adj_matrix[loop_face_id]];

	  if (conn.type == mesh::MAT_MAT)
	  {
		// Clean matrices
		auto& cur = inner[i][conn_id];
		std::fill_n(&cur.Q1.values[0], cur.Q1.values.size(), 0.0);
		std::fill_n(&cur.Q2.values[0], cur.Q2.values.size(), 0.0);
		std::fill_n(&cur.Th1.values[0], cur.Th1.values.size(), 0.0);
		std::fill_n(&cur.Th2.values[0], cur.Th2.values.size(), 0.0);
		std::fill_n(&cur.R1.values[0], cur.R1.values.size(), 0.0);
		std::fill_n(&cur.R2.values[0], cur.R2.values.size(), 0.0);

		const index_t& cell_id1 = conn.elem_id1;
		const index_t& cell_id2 = conn.elem_id2;
		const auto& c1 = mesh->centroids[cell_id1];
		const auto& c2 = mesh->centroids[cell_id2];

		if (dot((conn.c - c1), conn.n) < 0)
		{
		  n.values = -std::valarray<value_t>(conn.n.values.data(), conn.n.values.size());
		  n_vec = -conn.n;
		}
		else
		{
		  n.values = std::valarray<value_t>(conn.n.values.data(), conn.n.values.size());
		  n_vec = conn.n;
		}
		P = I3 - linalg::outer_product(n, n.transpose());

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
		conn_c.values = std::valarray<value_t>(conn.c.values.data(), ND);
		auto& r1 = cur.r1;
		auto& r2 = cur.r2;
		r1 = dot(n_vec, conn.c - c1);
		r2 = dot(n_vec, c2 - conn.c);
		assert(r1 > 0.0);		assert(r2 > 0.0);
		auto& y1 = cur.y1;
		auto& y2 = cur.y2;
		y1.values = std::valarray<value_t>((c1 + r1 * n_vec).values.data(), ND);	 
		y2.values = std::valarray<value_t>((c2 - r2 * n_vec).values.data(), ND);
		// Assemble matrices
		auto& Q1 = cur.Q1;						auto& Q2 = cur.Q2;
		auto& Th1 = cur.Th1;					auto& Th2 = cur.Th2;
		auto& R1 = cur.R1;						auto& R2 = cur.R2;

		Q1(0, { ND, ND }, { (uint8_t)Q1.N, 1 }) = -T1.values;
		Q2(0, { ND, ND }, { (uint8_t)Q1.N, 1 }) = -T2.values;
		Th1(0, { ND, ND * ND }, { (uint8_t)Th1.N, 1 }) = -G1.values;
		Th2(0, { ND, ND * ND }, { (uint8_t)Th1.N, 1 }) = -G2.values;
		

		B1n = biots[cell_id1] * n;				B2n = biots[cell_id2] * n;
		if /* constexpr */ (MODE == THERMOPOROELASTIC)
		{
		  A1n = th_exps[cell_id1] * n;			A2n = th_exps[cell_id2] * n;
		}
		
		// main matrix
		A(ND * face_id * A.N, { ND, (uint8_t)A.N }, { (uint8_t)A.N, 1 }) = ((Q2 * make_block_diagonal((y2 - y1).transpose(), ND) + r2 * (Th1 - Th2)) * make_block_diagonal(P, ND) +
					(r2 * Q1 + r1 * Q2) * make_block_diagonal(n.transpose(), ND)).values;
		
		// RHS
		res1 = findInVector(st, cell_id1);
		if (res1.first) { id1 = res1.second; }
		else { id1 = st.size(); st.push_back(cell_id1); }
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1, { ND, ND }, { (size_t)rhs_mult.N, 1 }) += -Q2.values;

		res2 = findInVector(st, cell_id2);
		if (res2.first) { id2 = res2.second; }
		else { id2 = st.size(); st.push_back(cell_id2); }
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2, { ND, ND }, { (size_t)rhs_mult.N, 1 }) += Q2.values;
		
		// left Biot term: B_1 * n * (p_1 + (x_c - x_1)^T * \nabla p_1)
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) -= r2 * B1n.values;

		diff1 = conn.c - c1;
		for (index_t j = grad_offset[cell_id1]; j < grad_offset[cell_id1 + 1]; j++)
		{
		  cur_cell_id = grad_stencil[j];
		  res1 = findInVector(st, cur_cell_id);
		  if (res1.first) { id1 = res1.second; }
		  else { id1 = st.size(); st.push_back(cur_cell_id); }
		  buf1 = diff1.x * p_grad_vals[ND * j] +
				  diff1.y * p_grad_vals[ND * j + 1] +
					diff1.z * p_grad_vals[ND * j + 2];
		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) -= r2 * buf1 * B1n.values;
		}

		buf1 = diff1.x * p_grad_rhs[ND * cell_id1] +
				diff1.y * p_grad_rhs[ND * cell_id1 + 1] +
				  diff1.z * p_grad_rhs[ND * cell_id1 + 2];
		rest(ND * face_id, { ND }, { 1 }) -= r2 * buf1 * B1n.values;

		// right Biot term: B_2 * n * (p_2 + (x_c - x_2)^T * \nabla p_2)
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * B2n.values;

		diff2 = conn.c - c2;
		for (index_t j = grad_offset[cell_id2]; j < grad_offset[cell_id2 + 1]; j++)
		{
		  cur_cell_id = grad_stencil[j];
		  res2 = findInVector(st, cur_cell_id);
		  if (res1.first) { id2 = res2.second; }
		  else { id2 = st.size(); st.push_back(cur_cell_id); }
		  buf2 = diff2.x * p_grad_vals[ND * j] +
				  diff2.y * p_grad_vals[ND * j + 1] +
					diff2.z * p_grad_vals[ND * j + 2];
		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * buf2 * B2n.values;
		}

		buf2 = diff2.x * p_grad_rhs[ND * cell_id2] +
				diff2.y * p_grad_rhs[ND * cell_id2 + 1] +
				  diff2.z * p_grad_rhs[ND * cell_id2 + 2];
		rest(ND * face_id, { ND }, { 1 }) += r2 * buf2 * B2n.values;

		face_id++;
	  }
	  else if (conn.type == mesh::MAT_BOUND)
	  {
		const auto& an = bc_mech.a_n[conn.elem_id2 - mesh->n_cells];
		const auto& bn = bc_mech.b_n[conn.elem_id2 - mesh->n_cells];
		const auto& at = bc_mech.a_t[conn.elem_id2 - mesh->n_cells];
		const auto& bt = bc_mech.b_t[conn.elem_id2 - mesh->n_cells];
		const auto& ap = bc_flow.a_p[conn.elem_id2 - mesh->n_cells];
		const auto& bp = bc_flow.b_p[conn.elem_id2 - mesh->n_cells];

		
		// Skip if pure neumann
		if (!NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION && an == 0.0 && at == 0.0)	continue;

		const index_t& cell_id1 = conn.elem_id1;
		const index_t& cell_id2 = conn.elem_id2;
		const auto& c1 = mesh->centroids[cell_id1];
		c1_mat.values = std::valarray<value_t>(c1.values.data(), c1.values.size());

		if (dot((conn.c - c1), conn.n) < 0)
		{
		  n.values = -std::valarray<value_t>(conn.n.values.data(), conn.n.values.size());
		  n_vec = -conn.n;
		}
		else
		{
		  n.values = std::valarray<value_t>(conn.n.values.data(), conn.n.values.size());
		  n_vec = conn.n;
		}
		P = I3 - linalg::outer_product(n, n.transpose());
		conn_c.values = std::valarray<value_t>(conn.c.values.data(), ND);
		r1 = dot(n_vec, conn.c - c1);		assert(r1 > 0.0);
		y1.values = std::valarray<value_t>((c1 + r1 * n_vec).values.data(), ND);

		B1n = biots[cell_id1] * n;
		K1n = DARCY_CONSTANT * perms[cell_id1] * n;
		lam1 = (n.transpose() * K1n).values[0];
		gam1 = K1n - lam1 * n;

		// Stiffness decomposition
		C1 = W * stfs[cell_id1] * W.transpose();
		nblock = make_block_diagonal(n, ND);
		nblock_t = make_block_diagonal(n.transpose(), ND);
		tblock = make_block_diagonal(P, ND);
		T1 = nblock_t * C1 * nblock;
		G1 = nblock_t * C1 * tblock;

		// Extra 'boundary' stuff
		An = (an * I3 + bn / r1 * T1);
		At = (at * I3 + bt / r1 * T1);
		Ap = 1.0 / (ap + bp / r1 * lam1);
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

		// filling matrix
		A(ND * face_id * A.N, { ND, (uint8_t)A.N }, { (uint8_t)A.N, 1 }) = (at * make_block_diagonal((conn_c - c1_mat).transpose(), ND) +
			bt * nblock_t * C1 + gamma_nnt_mult * (G1 + T1 / r1 * make_block_diagonal((y1 - conn_c).transpose(), ND))).values;

		// filling right-hand side
		res1 = findInVector(st, cell_id1);
		if (res1.first) { id1 = res1.second; }
		else { id1 = st.size(); st.push_back(cell_id1); }

		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1, { ND, ND }, { (size_t)rhs_mult.N, 1 }) += (gamma_nnt_mult * T1 / r1 - at * I3).values;
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += (Ap * bp * lam1 / r1 * mult_p).values;

		res2 = findInVector(st, cell_id2);
		if (res2.first) { id2 = res2.second; }
		else { id2 = st.size(); st.push_back(cell_id2); }

		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2, { ND, ND }, { (size_t)rhs_mult.N, 1 }) = (gamma_nnt + (I3 - gamma_nnt * L) * P).values;
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = (mult_p * Ap).values;

		// pressure gradient
		tmp.values = ((- Ap * bp) * (lam1 / r1 * (y1 - conn_c) + gam1).transpose() * P).values;
		for (index_t j = grad_offset[cell_id1]; j < grad_offset[cell_id1 + 1]; j++)
		{
		  cur_cell_id = grad_stencil[j];
		  res1 = findInVector(st, cur_cell_id);
		  if (res1.first) { id1 = res1.second; }
		  else { id1 = st.size(); st.push_back(cur_cell_id); }
		  buf1 = tmp.values[0] * p_grad_vals[ND * j] +
				  tmp.values[1] * p_grad_vals[ND * j + 1] +
					tmp.values[2] * p_grad_vals[ND * j + 2];
		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += (buf1 * mult_p).values;
		}
		buf1 = tmp.values[0] * p_grad_rhs[ND * cell_id1] +
				tmp.values[1] * p_grad_rhs[ND * cell_id1 + 1] +
				  tmp.values[2] * p_grad_rhs[ND * cell_id1 + 2];
		rest(ND * face_id, { ND }, { 1 }) += (buf1 * mult_p).values;

		rest(ND * face_id, { ND }, { 1 }) = (mult_p * Ap * bp * (grav_vec * K1n).values[0]).values;

		face_id++;
	  }
	}

	auto& cur_rhs = pre_cur_rhs[n_cur_faces][st.size()];
	cur_rhs.values = rhs_mult(0, { (size_t)cur_rhs.M, (size_t)cur_rhs.N }, { (size_t)rhs_mult.N, 1 });

	to_invert = A.transpose() * A;
	try
	{
	  to_invert.inv();

	  // check inversion
	  for (const auto& val : to_invert.values)
		assert(val == val && std::isfinite(val));

	  Matrix tempGrad = to_invert * A.transpose() * cur_rhs;
	  Matrix rhsGrad = to_invert * A.transpose() * rest;

	  std::vector<std::pair<index_t, index_t>> sort_vec(st.size());
	  for (index_t row = 0; row < st.size(); row++)
	  {
		sort_vec[row] = std::make_pair(st[row], row);
	  }
	  std::sort(sort_vec.begin(), sort_vec.end(), [](auto& left, auto& right) { return left.first < right.first; });

	  u_grad_offset.push_back(static_cast<index_t>(u_grad_stencil.size()));

	  // push sorted stencil
	  for (const auto& st : sort_vec)
		u_grad_stencil.push_back(st.first);

	  // push sorted coefficients & rhs
	  for (int row = 0; row < tempGrad.M; row++)
	  {
		u_grad_rhs.push_back(rhsGrad(row, 0));
		for (int col = 0; col < st.size(); col++)
		{
		  u_grad_vals.push_back(tempGrad(row, sort_vec[col].second));
		}
	  }
	}
	catch (const std::exception&)
	{
	  throw "Matrix is not invertible";
	}
  }

  u_grad_offset.push_back(static_cast<index_t>(u_grad_stencil.size()));

  t2 = steady_clock::now();
  cout << "Reconstruction of displacements gradients:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_mpfa_mpsa_transmissibilities()
{
  // clear previous approximations
  mech_cell_m.clear();			mech_cell_p.clear();
  mech_stencil.clear();			mech_offset.clear();
  mech_tran.clear();			mech_rhs.clear();
  mech_tran_biot.clear();		mech_rhs_biot.clear();

  // reserve memory
  mech_cell_m.reserve(mesh->adj_matrix.size());
  mech_cell_p.reserve(mesh->adj_matrix.size());
  mech_stencil.reserve(mesh->adj_matrix.size() * MAX_STENCIL);
  mech_offset.reserve(mesh->adj_matrix.size() + 1);

  // 4 means number of approximations: momentum + fluid flow
  const size_t APPR_SIZE = 4 * n_unknowns * MAX_STENCIL; 
  mech_tran.reserve(mesh->adj_matrix.size() * APPR_SIZE);
  mech_tran_biot.reserve(mesh->adj_matrix.size() * APPR_SIZE);
  mech_rhs.reserve(mesh->adj_matrix.size() * n_unknowns);
  mech_rhs_biot.reserve(mesh->adj_matrix.size());

  index_t cell_id1, cell_id2;
  value_t sign;
  steady_clock::time_point t1, t2;
  t1 = steady_clock::now();

  mech_offset.push_back(0);
  for (index_t i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
  {
	cell_id1 = i;

	// loop through connections of particular element
	for (index_t j = mesh->adj_matrix_offset[i], conn_id = 0; j < mesh->adj_matrix_offset[i + 1]; j++, conn_id++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[j]];
	  cell_id2 = mesh->adj_matrix_cols[j];
	  sign = (conn.elem_id1 == cell_id1) ? 1.0 : -1.0;

	  if (conn.type == mesh::MAT_MAT)
	  {
		auto& flux = mech_fluxes[0];
		calc_matrix_matrix_mech(conn, flux, conn_id);

		flux.a.values *= sign * conn.area;
		flux.a_biot.values *= sign * conn.area;
		flux.rhs.values *= sign * conn.area;
		flux.rhs_biot.values *= sign * conn.area;

		mech_cell_m.push_back(cell_id1);
		mech_cell_p.push_back(cell_id2);
		write_trans_mech(flux);
	  }
	  else if (conn.type == mesh::MAT_BOUND)
	  {
		auto& flux = mech_fluxes[0];
		calc_matrix_boundary_mech(conn, flux, conn_id);

		flux.a.values *= sign * conn.area;
		flux.a_biot.values *= sign * conn.area;
		flux.rhs.values *= sign * conn.area;
		flux.rhs_biot.values *= sign * conn.area;

		mech_cell_m.push_back(cell_id1);
		mech_cell_p.push_back(cell_id2);
		write_trans_mech(flux);
	  }
	}
  }

  t2 = steady_clock::now();
  cout << "Find MPFA-MPSA trans: \t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_matrix_matrix_mech(const mesh::Connection& conn, MechApproximation& flux, index_t conn_id)
{
  Matrix n(ND, 1), det(ND, ND), coef1(ND, ND), coef2(ND, ND), bcoef1(ND, 1), bcoef2(ND, 1), Q(ND, ND), grad_coef(ND, ND * ND);
  Vector3 diff1, diff2;
  size_t id1, id2;
  value_t buf1, buf2;
  index_t cur_cell_id;
  std::pair<bool, size_t> res1, res2;

  const auto& x1 = mesh->centroids[conn.elem_id1];
  const auto& x2 = mesh->centroids[conn.elem_id2];
  bool res;

  // normal vector
  copy_n(std::begin(conn.n.values), ND, std::begin(n.values));
  if (dot(conn.c - x1, conn.n) < 0.0) n.values *= -1.0;

  // allocate arrays for merging gradients
  const uint8_t n_st1 = grad_offset[conn.elem_id1 + 1] - grad_offset[conn.elem_id1];
  const uint8_t n_st2 = grad_offset[conn.elem_id2 + 1] - grad_offset[conn.elem_id2];
  MechApproximation g1(ND * ND, n_unknowns * n_st1), g2(ND * ND, n_unknowns * n_st2);

  // merging gradients
  const index_t grad_coef_size = ND * ND * n_unknowns;
  std::copy_n(u_grad_stencil.begin() + u_grad_offset[conn.elem_id1], n_st1, g1.stencil.begin());
  std::copy_n(u_grad_vals.data() + grad_coef_size * u_grad_offset[conn.elem_id1], grad_coef_size * n_st1, std::begin(g1.a.values));
  std::copy_n(u_grad_rhs.data() + ND * ND * conn.elem_id1, ND * ND, std::begin(g1.rhs.values));

  std::copy_n(u_grad_stencil.begin() + grad_offset[conn.elem_id2], n_st2, g2.stencil.begin());
  std::copy_n(u_grad_vals.data() + grad_coef_size * u_grad_offset[conn.elem_id2], grad_coef_size * n_st2, std::begin(g2.a.values));
  std::copy_n(u_grad_rhs.data() + ND * ND * conn.elem_id2, ND * ND, std::begin(g2.rhs.values));

  flux.stencil.clear();
  Matrix nabla_u = mergeMatrices(g1.a, g2.a, g1.stencil, g2.stencil, flux.stencil);

  const auto& cur = inner[conn.elem_id1][conn_id];
  det = cur.r1 * cur.Q2 + cur.r2 * cur.Q1;
  res = det.inv();
  if (!res)
  {
	cout << "Inversion failed!\n";	exit(-1);
  }
  Q = cur.Q1 * det * cur.Q2;
  coef1 = cur.r1 * cur.Q2 * det;
  coef2 = cur.r2 * cur.Q1 * det;
  grad_coef = coef1 * cur.Th1 + coef2 * cur.Th2 + Q * make_block_diagonal((cur.y1 - cur.y2).transpose(), ND);

  fill_n(std::begin(flux.a.values), flux.a.values.size(), 0.0);
  fill_n(std::begin(flux.a_biot.values), flux.a_biot.values.size(), 0.0);
  
  flux.a(0, { ND, (size_t)nabla_u.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * nabla_u).values;
  flux.rhs = grad_coef * (g1.rhs + g2.rhs) / 2.0;

  res1 = findInVector(flux.stencil, conn.elem_id1);
  if (res1.first) { id1 = res1.second; }
  else { printf("Gradient within %d cell does not depend on its value!\n", conn.elem_id1);	exit(-1); }
  flux.a(n_unknowns * id1, { (size_t)flux.a.M, (size_t)n_unknowns }, { (size_t)flux.a.N, 1 }) -= Q.values;
  //flux.a_biot(n_unknowns * id1, { (size_t)flux.a_biot.M, n_unknowns }, { (size_t)flux.a_biot.N, 1 }) += (A1_tilde - biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;

  // left Biot term: B_1 * n * (p_1 + (x_c - x_1)^T * \nabla p_1)
  bcoef1.values = (coef1 * biots[conn.elem_id1] * n).values;
  flux.a(n_unknowns * id1 + ND, { (size_t)flux.a.M, 1 }, { (size_t)flux.a.N, 1 }) += bcoef1.values;

  diff1 = conn.c - x1;
  for (index_t j = grad_offset[conn.elem_id1]; j < grad_offset[conn.elem_id1 + 1]; j++)
  {
	cur_cell_id = grad_stencil[j];
	res1 = findInVector(flux.stencil, cur_cell_id);
	if (res1.first) { id1 = res1.second; }
	else { id1 = flux.stencil.size(); flux.stencil.push_back(cur_cell_id); }
	buf1 = diff1.x * p_grad_vals[ND * j] +
			diff1.y * p_grad_vals[ND * j + 1] +
			  diff1.z * p_grad_vals[ND * j + 2];
	flux.a(n_unknowns * id1 + ND, { ND, 1 }, { (size_t)flux.a.N, 1 }) += buf1 * bcoef1.values;
  }

  buf1 = diff1.x * p_grad_rhs[ND * conn.elem_id1] +
		  diff1.y * p_grad_rhs[ND * conn.elem_id1 + 1] +
			diff1.z * p_grad_rhs[ND * conn.elem_id1 + 2];
  flux.rhs(0, { ND }, { 1 }) += buf1 * bcoef1.values;

  res2 = findInVector(flux.stencil, conn.elem_id2);
  if (res2.first) { id2 = res2.second; }
  else { printf("Gradient within %d cell does not depend on its value!\n", conn.elem_id2);	exit(-1); }
  flux.a(n_unknowns * id2, { (size_t)flux.a.M, (size_t)n_unknowns }, { (size_t)flux.a.N, 1 }) += Q.values;
  //flux.a_biot(n_unknowns * id1, { (size_t)flux.a_biot.M, n_unknowns }, { (size_t)flux.a_biot.N, 1 }) += (A1_tilde - biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;

  // right Biot term: B_2 * n * (p_2 + (x_c - x_2)^T * \nabla p_2)
  bcoef2.values = (coef2 * biots[conn.elem_id2] * n).values;
  flux.a(n_unknowns * id2 + ND, { (size_t)flux.a.M, 1 }, { (size_t)flux.a.N, 1 }) += bcoef2.values;

  diff2 = conn.c - x2;
  for (index_t j = grad_offset[conn.elem_id2]; j < grad_offset[conn.elem_id2 + 1]; j++)
  {
	cur_cell_id = grad_stencil[j];
	res2 = findInVector(flux.stencil, cur_cell_id);
	if (res2.first) { id2 = res2.second; }
	else { id2 = flux.stencil.size(); flux.stencil.push_back(cur_cell_id); }
	buf2 = diff2.x * p_grad_vals[ND * j] +
			diff2.y * p_grad_vals[ND * j + 1] +
			  diff2.z * p_grad_vals[ND * j + 2];
	flux.a(n_unknowns * id2 + ND, { ND, 1 }, { (size_t)flux.a.N, 1 }) += buf2 * bcoef2.values;
  }

  buf2 = diff2.x * p_grad_rhs[ND * conn.elem_id2] +
		  diff2.y * p_grad_rhs[ND * conn.elem_id2 + 1] +
			diff2.z * p_grad_rhs[ND * conn.elem_id2 + 2];
  flux.rhs(0, { ND }, { 1 }) += buf2 * bcoef2.values;
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_matrix_boundary_mech(const mesh::Connection& conn, MechApproximation& flux, index_t conn_id)
{

}

template class MechDiscretizer<POROELASTIC>;
template class MechDiscretizer<THERMOPOROELASTIC>;