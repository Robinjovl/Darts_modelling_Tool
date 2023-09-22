#include "mech_discretizer.h"
#include <chrono>

using namespace dis;
using std::vector;
using std::cout;
using std::endl;
using std::begin;
using std::end;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::fill_n;
using std::copy_n;

template <MechDiscretizerMode MODE>
const uint8_t MechDiscretizer<MODE>::n_unknowns = N_UNKNOWNS.at(MODE);

// this matrix W helps to translate elasticity operator to simpler form
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
  GRADIENTS_EXTENDED_STENCIL = false;
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

		cur.R1 = Matrix(ND, 1);			  cur.R2 = Matrix(ND, 1);
		cur.y1 = Matrix(ND, 1);			  cur.y2 = Matrix(ND, 1);
		cur.T1 = Matrix(ND, ND);		  cur.T2 = Matrix(ND, ND);
		cur.G1 = Matrix(ND, ND * ND);	  cur.G2 = Matrix(ND, ND * ND);
	  }
	}
  }

  for (index_t i = mesh::MIN_CONNS_PER_ELEM; i < 10; i++) // mesh::MAX_CONNS_PER_ELEM is too big
  {
	pre_grad_A_u[i] = Matrix(ND * i, ND * ND);
	pre_grad_R_u[i] = Matrix(ND * i, n_unknowns * MAX_STENCIL);
	pre_grad_rhs_u[i] = Matrix(ND * i, 1);

	for (index_t st_size = 1; st_size < MAX_STENCIL; st_size++)
	{
	  pre_cur_rhs[i][st_size] = Matrix(ND * i, n_unknowns * st_size);
	}
  }

  mech_fluxes.resize(MAX_FLUXES_NUM, MechApproximation<MODE>(MAX_STENCIL));
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::reconstruct_displacement_gradients_per_cell(const THMBoundaryCondition& bc_thm_new)
{
  // Variables
  std::vector<index_t> st;		st.reserve(MAX_STENCIL);
  std::vector<index_t> admissible_connections(4, 0);
  Matrix n(ND, 1), conn_c(ND, 1), P(ND, ND), B1n(ND, 1), B2n(ND, 1), A1n(ND, 1), A2n(ND, 1), K1n(ND, 1), gam1(ND, 1), K2n(ND, 1), gam2(ND, 1), tmp(ND, 1);
  Vector3 n_vec, diff1, diff2;
  Matrix C1(ND * ND, ND * ND), C2(ND * ND, ND * ND), T1(ND, ND), G1(ND, ND * ND), mat_diff1(1, ND), mat_diff2(1, ND);
  Matrix nblock(ND * ND, ND), nblock_t(ND, ND * ND), tblock(ND * ND, ND * ND);
  Matrix mult_p(ND, 1), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), An(ND, ND), At(ND, ND), L(ND, ND), y1(ND, 1), c1_mat(ND, 1);
  Matrix to_invert(ND * ND, ND * ND);
  value_t buf1, buf2, Ap, gamma, r1, lam1, lam2;
  index_t n_cur_faces, loop_face_id, face_id, conn_id, id1, id2, cur_cell_id;
  bool res;

  // allocate memory for arrays
  u_grads.resize(mesh->n_cells, ApproximationType<MODE>(ND * ND, MAX_STENCIL));

  bc_thm = bc_thm_new;

  steady_clock::time_point t1, t2;
  t1 = steady_clock::now();

  // loop through the adjacency matrix (matrix cells)
  for (index_t i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
  {
	st.clear();

	// Build the system from the continuity at the interfaces
	n_cur_faces = 0;
	for (index_t j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[j]];
	  if (conn.type == mesh::MAT_BOUND)
	  {
		// Coefficients that define boundary condition
		const auto& an = bc_thm.mech_normal.a[conn.elem_id2 - mesh->n_cells];
		const auto& bn = bc_thm.mech_normal.b[conn.elem_id2 - mesh->n_cells];
		const auto& at = bc_thm.mech_tangen.b[conn.elem_id2 - mesh->n_cells];
		const auto& bt = bc_thm.mech_tangen.b[conn.elem_id2 - mesh->n_cells];

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
	  const auto& conn = mesh->conns[mesh->adj_matrix[loop_face_id]];

	  if (conn.type == mesh::MAT_MAT)
	  {
		// Clean matrices
		auto& cur = inner[i][conn_id];

		const index_t& cell_id1 = i;
		const index_t& cell_id2 = mesh->adj_matrix_cols[loop_face_id];
		const auto& c1 = mesh->centroids[cell_id1];
		const auto& c2 = mesh->centroids[cell_id2];
		
		//TODO: as connection's normal is always outside for the cell_id1 (see sign = (conn.elem_id1 == cell_id1) ? 1.0 : -1.0;)
		// here the second condition should always be true
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
		// Stiffness : (div(u) + div(u)^T)/2 = [I*n^T]S[div*u], where '*' is tensor multiplication and ':' is tensor reduction, where S = WCW^T
		C1 = W * stfs[cell_id1] * W.transpose();
		C2 = W * stfs[cell_id2] * W.transpose();
		nblock = make_block_diagonal(n, ND);
		nblock_t = make_block_diagonal(n.transpose(), ND);
		tblock = make_block_diagonal(P, ND);
		auto& T1 = cur.T1;	  		auto& G1 = cur.G1;
		auto& T2 = cur.T2;			auto& G2 = cur.G2;
		T1.values = (nblock_t * C1 * nblock).values;
		T2.values = (nblock_t * C2 * nblock).values;
		G1.values = (nblock_t * C1 * tblock).values;
		G2.values = (nblock_t * C2 * tblock).values;

		// Process geometry
		conn_c.values = std::valarray<value_t>(conn.c.values.data(), ND);
		auto& r1 = cur.r1;			auto& y1 = cur.y1;
		auto& r2 = cur.r2;			auto& y2 = cur.y2;
		r1 = dot(n_vec, conn.c - c1);
		r2 = dot(n_vec, c2 - conn.c);
		assert(r1 > 0.0);		assert(r2 > 0.0);
		y1.values = std::valarray<value_t>((c1 + r1 * n_vec).values.data(), ND);	 
		y2.values = std::valarray<value_t>((c2 - r2 * n_vec).values.data(), ND);
		
		// projection to normal
		B1n = biots[cell_id1] * n;				B2n = biots[cell_id2] * n;
		if constexpr  (MODE == THERMOPOROELASTIC)
		{
		  A1n = th_exps[cell_id1] * n;			A2n = th_exps[cell_id2] * n;
		}
		
		// main matrix
		A(ND * face_id * A.N, { ND, (uint8_t)A.N }, { (uint8_t)A.N, 1 }) = (T2 * make_block_diagonal((y2 - y1).transpose(), ND) + r2 * (G1 - G2) +
					(r2 * T1 + r1 * T2) * make_block_diagonal(n.transpose(), ND)).values;
		
		// RHS
		res1 = findInVector(st, cell_id1);
		if (res1.first) { id1 = res1.second; }
		else { id1 = st.size(); st.push_back(cell_id1); }
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1, { ND, ND }, { (size_t)rhs_mult.N, 1 }) -= T2.values;

		res2 = findInVector(st, cell_id2);
		if (res2.first) { id2 = res2.second; }
		else { id2 = st.size(); st.push_back(cell_id2); }
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2, { ND, ND }, { (size_t)rhs_mult.N, 1 }) += T2.values;
		
		if (GRADIENTS_EXTENDED_STENCIL) // use of \nabla p_2
		{
		  // left Biot term: B_1 * n * (p_1 + (x_c - x_1)^T * \nabla p_1)
		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * B1n.values;

		  diff1 = conn.c - c1;
		  const auto& g1 = p_grads[cell_id1];
		  for (index_t k = 0; k < g1.stencil.size(); k++)
		  {
			cur_cell_id = g1.stencil[k];
			res1 = findInVector(st, cur_cell_id);
			if (res1.first) { id1 = res1.second; }
			else { id1 = st.size(); st.push_back(cur_cell_id); }
			buf1 = diff1.x * g1.a(0, k) +
			  diff1.y * g1.a(1, k) +
			  diff1.z * g1.a(2, k);
			rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * buf1 * B1n.values;
		  }

		  buf1 = diff1.x * g1.rhs(0, 0) +
			diff1.y * g1.rhs(1, 0) +
			diff1.z * g1.rhs(2, 0);
		  rest(ND * face_id, { ND }, { 1 }) += r2 * buf1 * B1n.values;

		  // right Biot term: B_2 * n * (p_2 + (x_c - x_2)^T * \nabla p_2)
		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) -= r2 * B2n.values;

		  diff2 = conn.c - c2;
		  const auto& g2 = p_grads[cell_id2];
		  for (index_t k = 0; k < g2.stencil.size(); k++)
		  {
			cur_cell_id = g2.stencil[k];
			res2 = findInVector(st, cur_cell_id);
			if (res2.first) { id2 = res2.second; }
			else { id2 = st.size(); st.push_back(cur_cell_id); }
			buf2 = diff2.x * g2.a(0, k) +
			  diff2.y * g2.a(1, k) +
			  diff2.z * g2.a(2, k);
			rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) -= r2 * buf2 * B2n.values;
		  }

		  buf2 = diff2.x * g2.rhs(0, 0) +
			diff2.y * g2.rhs(1, 0) +
			diff2.z * g2.rhs(2, 0);
		  rest(ND * face_id, { ND }, { 1 }) -= r2 * buf2 * B2n.values;
		}
		else // no use of \nabla p_2 (default)
		{
		  // r_2 * (p_{\beta1} * B_1 * n - p_{\beta2} * B_2 * n )
		  // p_{\beta1} remains the same, p_{\beta2} uses the following approximation
		  // p_{\beta 2} = p_2 + (x_\beta - y_2 - r_2 / \lambda_2 * (K_1 * n - \gamma_2) )^T * \nabla p_1 + 
		  // + r_2 / \lambda_2 * \rho * g * \nabla z * (K_1 - K_2) * n  
		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * B1n.values;

		  K1n = DARCY_CONSTANT * perms[cell_id1] * n;
		  K2n = DARCY_CONSTANT * perms[cell_id2] * n;
		  lam2 = (n.transpose() * K2n).values[0];
		  gam2 = K2n - lam2 * n;

		  mat_diff1.values = std::valarray<value_t>((conn.c - c1).values.data(), ND);
		  mat_diff2.values = std::valarray<value_t>(conn.c.values.data(), ND) - y2.values;
		  
		  const auto& g1 = p_grads[cell_id1];
		  Matrix grad_mult(ND, ND);
		  Matrix grad_term(ND, g1.stencil.size());
		  grad_mult = outer_product(B1n, mat_diff1) - outer_product(B2n, mat_diff2 + r2 / lam2 * (gam2 - K1n).transpose());
		  grad_term = grad_mult * g1.a;
		  for (index_t k = 0; k < g1.stencil.size(); k++)
		  {
			cur_cell_id = g1.stencil[k];
			res1 = findInVector(st, cur_cell_id);
			if (res1.first) { id1 = res1.second; }
			else { id1 = st.size(); st.push_back(cur_cell_id); }
			Matrix block(grad_term(k, { ND, 1 }, { (size_t)grad_term.N, 1 }), ND, 1);
			rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * block.values;
		  }
		  rest(ND * face_id, { ND }, { 1 }) += r2 * (grad_mult * g1.rhs + r2 / lam2 * (grav_vec * (K2n - K1n)).values[0] * B2n).values;

		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) -= r2 * B2n.values;
		}

		face_id++;
	  }
	  else if (conn.type == mesh::MAT_BOUND)
	  {
		const auto& an = bc_thm.mech_normal.a[conn.elem_id2 - mesh->n_cells];
		const auto& bn = bc_thm.mech_normal.b[conn.elem_id2 - mesh->n_cells];
		const auto& at = bc_thm.mech_tangen.a[conn.elem_id2 - mesh->n_cells];
		const auto& bt = bc_thm.mech_tangen.b[conn.elem_id2 - mesh->n_cells];
		const auto& ap = bc_thm.flow.a[conn.elem_id2 - mesh->n_cells];
		const auto& bp = bc_thm.flow.b[conn.elem_id2 - mesh->n_cells];

		
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
		tmp.values = ((-Ap * bp) * (lam1 / r1 * (y1 - conn_c) + gam1).transpose() * P).values;
		const auto& g1 = p_grads[cell_id1];
		for (index_t k = 0; k < g1.stencil.size(); k++)
		{
		  cur_cell_id = g1.stencil[k];
		  res1 = findInVector(st, cur_cell_id);
		  if (res1.first) { id1 = res1.second; }
		  else { id1 = st.size(); st.push_back(cur_cell_id); }
		  buf1 = tmp.values[0] * g1.a(0, k) +
				  tmp.values[1] * g1.a(1, k) +
					tmp.values[2] * g1.a(2, k);
		  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += (buf1 * mult_p).values;
		}
		buf1 = tmp.values[0] * g1.rhs(0, 0) +
				tmp.values[1] * g1.rhs(1, 0) +
				  tmp.values[2] * g1.rhs(2, 0);
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

	  auto& cur_grad = u_grads[i];
      // 9 x n_unknowns * stencil matrix for the each cell
	  cur_grad.a = to_invert * A.transpose() * cur_rhs;
      // 5 x 1
	  cur_grad.rhs = to_invert * A.transpose() * rest;
	  cur_grad.stencil = st;
      // sorted array needed for the fast merge of two arrays
	  cur_grad.sort();
	}
	catch (const std::exception&)
	{
	  throw "Matrix is not invertible";
	}
  }

  keep_same_stencil_gradients();

  t2 = steady_clock::now();
  cout << "Reconstruction of displacements gradients:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::keep_same_stencil_gradients()
{
  index_t i, j;
  std::vector<index_t> new_stencil;
  new_stencil.reserve(MAX_STENCIL);
  
  for (index_t cell_id = 0; cell_id < mesh->region_ranges.at(mesh::FRACTURE).second; cell_id++)
  {
	auto& p_grad = p_grads[cell_id];
	auto& u_grad = u_grads[cell_id];

	if (p_grad.stencil != u_grad.stencil)
	{
	  new_stencil.clear();
	  merge_stencils(u_grad.stencil, p_grad.stencil, new_stencil);
	  Matrix new_a(ND, new_stencil.size());

	  for (i = 0, j = 0; i < p_grad.stencil.size(); i++)
	  {
		while (new_stencil[j] != p_grad.stencil[i]) { j++; }

		for (uint8_t row = 0; row < ND; row++)
		  new_a(row, j) = p_grad.a(row, i);
	  }

	  p_grad.a = new_a;
	  p_grad.stencil = new_stencil;
	}

	assert(p_grad.stencil == u_grad.stencil);
  }
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_mpfa_mpsa_transmissibilities()
{
  // clear previous approximations
  /*mech_cell_m.clear();			mech_cell_p.clear();
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
		// the connection stores only one-side normal, so need to use a sign
		// the normal for the elem_id1 points outside the cell
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
  cout << "Find MPFA-MPSA trans: \t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;*/
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_matrix_matrix_mech(const mesh::Connection& conn, MechApproximation<MODE>& flux, index_t conn_id)
{
  /*Matrix n(ND, 1), det(ND, ND), coef1(ND, ND), coef2(ND, ND), bcoef1(ND, 1), bcoef2(ND, 1), Q(ND, ND);
  Matrix grad_coef(ND, ND * ND), K1n(ND, 1), K2n(ND, 1), B1n(ND, 1), gam1(ND, 1), gam2(ND, 1);
  Matrix conn_c(ND, 1), face_unknown_coef(1, ND), biot_grad_coef(ND, 1);
  Vector3 diff1, diff2;
  size_t id1, id2;
  value_t buf1, buf2, lam1, lam2, det_lam;
  index_t cur_cell_id;
  std::pair<bool, size_t> res1, res2;

  const auto& x1 = mesh->centroids[conn.elem_id1];
  const auto& x2 = mesh->centroids[conn.elem_id2];
  conn_c.values = std::valarray<value_t>(conn.c.values.data(), ND);
  bool res;

  // normal vector
  copy_n(begin(conn.n.values), ND, begin(n.values));
  if (dot(conn.c - x1, conn.n) < 0.0) n.values *= -1.0;

  const auto& g1 = u_grads[conn.elem_id1];
  const auto& g2 = u_grads[conn.elem_id2];

  flux.stencil.clear();
  flux.darcy = g1 / 2.0 + g2 / 2.0;

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
  
  B1n = biots[conn.elem_id1] * n;
  K1n = DARCY_CONSTANT * perms[conn.elem_id1] * n;
  K2n = DARCY_CONSTANT * perms[conn.elem_id2] * n;
  lam1 = (n.transpose() * K1n).values[0];
  lam2 = (n.transpose() * K2n).values[0];
  gam1 = K1n - lam1 * n;
  gam2 = K2n - lam2 * n;
  det_lam = 1.0 / (cur.r1 * lam2 + cur.r2 * lam1);
  
  face_unknown_coef = det_lam * (cur.r2 * lam1 * (conn_c - cur.y1).transpose() + 
								  cur.r1 * lam2 * (conn_c - cur.y2).transpose() + 
									cur.r1 * cur.r2 * (gam2 - gam1).transpose() );
  biot_grad_coef = outer_product(B1n, face_unknown_coef);

  fill_n(begin(flux.hooke.a.values), flux.hooke.a.values.size(), 0.0);
  fill_n(begin(flux.biot_traction.a.values), flux.biot_traction.a.values.size(), 0.0);
  
  //flux.hooke.a(0, { ND, (size_t)nabla_u.N }, { (size_t)flux.a.N, 1 }) = (grad_coef * nabla_u).values;
  //flux.hooke.rhs = grad_coef * flux.hooke.rhs;

  res1 = findInVector(flux.stencil, conn.elem_id1);
  if (res1.first) { id1 = res1.second; }
  else { printf("Gradient within %d cell does not depend on its value!\n", conn.elem_id1);	exit(-1); }
  flux.a(n_unknowns * id1, { (size_t)flux.a.M, (size_t)n_unknowns }, { (size_t)flux.a.N, 1 }) -= Q.values;
  //flux.a_biot(n_unknowns * id1, { (size_t)flux.a_biot.M, n_unknowns }, { (size_t)flux.a_biot.N, 1 }) += (A1_tilde - biot_flow_buf * (cur.Q2 + cur.r2 * cur.A1)).values;

  // left Biot term: B_1 * n * (p_1 + (x_c - x_1)^T * \nabla p_1)
  bcoef1.values = (coef1 * biots[conn.elem_id1] * n).values;
  flux.a(n_unknowns * id1 + ND, { (size_t)flux.a.M, 1 }, { (size_t)flux.a.N, 1 }) += bcoef1.values;

  diff1 = conn.c - x1;
  auto g1p = get_pressure_gradient(conn.elem_id1);
  for (index_t k = 0; k < g1p.stencil.size(); k++)
  {
	cur_cell_id = grad_offset[conn.elem_id1];
	res1 = findInVector(flux.stencil, cur_cell_id);
	if (res1.first) { id1 = res1.second; }
	else { id1 = flux.stencil.size(); flux.stencil.push_back(cur_cell_id); }
	buf1 = diff1.x * g1p.a(0, k) + diff1.y * g1p.a(1, k) + diff1.z * g1p.a(2, k);
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
  auto g2p = get_pressure_gradient(conn.elem_id2);
  for (index_t k = 0; k < g2p.stencil.size(); k++)
  {
	cur_cell_id = grad_stencil[grad_offset[conn.elem_id2] + k];
	res2 = findInVector(flux.stencil, cur_cell_id);
	if (res2.first) { id2 = res2.second; }
	else { id2 = flux.stencil.size(); flux.stencil.push_back(cur_cell_id); }
	buf2 = diff2.x * g2p.a(0, k) + diff2.y * g2p.a(1, k) + diff2.z * g2p.a(2, k);
	flux.a(n_unknowns * id2 + ND, { ND, 1 }, { (size_t)flux.a.N, 1 }) += buf2 * bcoef2.values;
  }

  buf2 = diff2.x * p_grad_rhs[ND * conn.elem_id2] +
		  diff2.y * p_grad_rhs[ND * conn.elem_id2 + 1] +
			diff2.z * p_grad_rhs[ND * conn.elem_id2 + 2];
  flux.rhs(0, { ND }, { 1 }) += buf2 * bcoef2.values;*/
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_matrix_boundary_mech(const mesh::Connection& conn, MechApproximation<MODE>& flux, index_t conn_id)
{

}

template class MechDiscretizer<POROELASTIC>;
template class MechDiscretizer<THERMOPOROELASTIC>;