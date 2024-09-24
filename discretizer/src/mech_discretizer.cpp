#include "mech_discretizer.h"
#include "utils.h"
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
using utils::get_valarray_from_array;

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

  for (index_t i = mesh::MIN_CONNS_PER_ELEM; i <= mesh::MAX_CONNS_PER_ELEM; i++) 
  {
	pre_grad_A_u[i] = Matrix(ND * i, ND * ND);
	pre_grad_R_u[i] = Matrix(ND * i, n_unknowns * MAX_STENCIL);
	pre_grad_rhs_u[i] = Matrix(ND * i, 1);

	for (index_t st_size = 1; st_size <= MAX_STENCIL; st_size++)
	{
	  pre_cur_rhs[i][st_size] = Matrix(ND * i, n_unknowns * st_size);
	}

	pre_N[i] = Matrix(i * ND, SUM_N(ND));
	pre_Nflux[i] = Matrix(i, ND);
	pre_R[i] = Matrix(i * ND, SUM_N(ND));
	pre_stress_approx[i] = Matrix(SUM_N(ND), ND * i);
	pre_vel_approx[i] = Matrix(ND, i);
  }

  mech_fluxes.resize(MAX_FLUXES_NUM, MechApproximation<MODE>(MAX_STENCIL));
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::reconstruct_displacement_gradients_per_cell(const THMBoundaryCondition& bc_thm_new)
{
  // Variables
  std::vector<index_t> st;		st.reserve(MAX_STENCIL);
  std::vector<index_t> admissible_connections(4, 0);
  Matrix n(ND, 1);
  Matrix conn_c(ND, 1);
  Matrix P(ND, ND);
  Matrix B1n(ND, 1), B2n(ND, 1); // projection of Biot tensor to the normal vector
  Matrix A1n(ND, 1), A2n(ND, 1); // projection of thermal expansion coefficients tensor to the normal vector
  Matrix K1n(ND, 1), K2n(ND, 1); // projection of permeability tensor to the normal vector
  Matrix C1n(ND, 1), C2n(ND, 1); // projection of thermal conductivity tensor to the normal vector
  Matrix gam1(ND, 1), gam2(ND, 1);
  Matrix gam1_thermal(ND, 1), gam2_thermal(ND, 1);
  Matrix tmp(ND, 1);
  Vector3 n_vec, diff1, diff2;
  Matrix C1(ND * ND, ND * ND), C2(ND * ND, ND * ND), T1(ND, ND), G1(ND, ND * ND), mat_diff1(1, ND), mat_diff2(1, ND);
  Matrix nblock(ND * ND, ND), nblock_t(ND, ND * ND), tblock(ND * ND, ND * ND);
  Matrix mult_p(ND, 1), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND), An(ND, ND), At(ND, ND), L(ND, ND), y1(ND, 1), c1_mat(ND, 1);
  Matrix mult_thermal(ND, 1);
  value_t A_thermal;
  Matrix tmp_thermal(ND, 1);
  Matrix to_invert(ND * ND, ND * ND);
  value_t buf1, buf2, Ap, gamma, r1, lam1, lam2;
  value_t lam1_thermal, lam2_thermal;
  index_t n_cur_faces, loop_face_id, face_id, conn_id, id1, id2, cur_cell_id;
  LinearApproximation<Tvar>* g1_thermal;
  value_t a_thermal, b_thermal;
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
	A.values = 0.;
	rest.values = 0.;
	rhs_mult.values = 0.;

	face_id = conn_id = 0;
	for (loop_face_id = mesh->adj_matrix_offset[i]; loop_face_id < mesh->adj_matrix_offset[i + 1]; loop_face_id++, conn_id++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[loop_face_id]];

	  if (conn.type == mesh::MAT_MAT)
	  {
		auto& cur = inner[i][conn_id];

		const index_t& cell_id1 = i;
		const index_t& cell_id2 = mesh->adj_matrix_cols[loop_face_id];
		const auto& c1 = mesh->centroids[cell_id1];
		const auto& c2 = mesh->centroids[cell_id2];

		if (dot((conn.c - c1), conn.n) < 0)
		{
		  n.values = -get_valarray_from_array(conn.n.values);
		  n_vec = -conn.n;
		}
		else
		{
		  n.values = get_valarray_from_array(conn.n.values);
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
			if constexpr (MODE == THERMOPOROELASTIC)
			{
				rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND + 1, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * A1n.values;
			}
			K1n = DARCY_CONSTANT * perms[cell_id1] * n;
			K2n = DARCY_CONSTANT * perms[cell_id2] * n;
			lam2 = (n.transpose() * K2n).values[0]; // scalar kappa
			gam2 = K2n - lam2 * n; // bold kappa

			if constexpr (MODE == THERMOPOROELASTIC) {
				C1n = heat_conductions[cell_id1] * n; //TODO check units
				C2n = heat_conductions[cell_id2] * n;
				lam2_thermal = (n.transpose() * C2n).values[0]; // scalar lambda
				gam2_thermal = C2n - lam2_thermal * n; // bold lambda
			}

			mat_diff1.values = std::valarray<value_t>((conn.c - c1).values.data(), ND);      // x_beta - x_1
			mat_diff2.values = std::valarray<value_t>(conn.c.values.data(), ND) - y2.values; // x_beta - y_2 (this vector lies on the interface plane)

			const auto& g1 = p_grads[cell_id1];
			Matrix grad_mult_p(ND, ND);
			Matrix grad_term_p(ND, g1.stencil.size());// grad_term for all neighbours, pressure part
			// B1n <*> (x_beta - x_1)^T - B2n <*> ( x_beta - y_2 - d2*(K1n-kappa2)^T/kappa2 )
			grad_mult_p = outer_product(B1n, mat_diff1) - outer_product(B2n, mat_diff2 + r2 / lam2 * (gam2 - K1n).transpose());
			grad_term_p = grad_mult_p * g1.a; // g1.a is grad(p)

			// pressure and thermal have the same stencil
			Matrix grad_mult_t(ND, ND);
			Matrix grad_term_t(ND, g1.stencil.size());// grad_term for all neighbours, thermal part
			if constexpr (MODE == THERMOPOROELASTIC) {
				const auto& g1_thermal = t_grads[cell_id1];
				grad_mult_t = outer_product(A1n, mat_diff1) - outer_product(A2n, mat_diff2 + r2 / lam2_thermal * (gam2_thermal - C1n).transpose());
				grad_term_t = grad_mult_t * g1_thermal.a;
			}

			for (index_t k = 0; k < g1.stencil.size(); k++) // grad(p) = sum_i(a_i * p_i) + b
			{
				// add grad_term matrix to rhs_mult matrix. These matrices have different stencils.
				// Find a column index in rhs_mult where to add. If there is no such index, add 
				cur_cell_id = g1.stencil[k];
				res1 = findInVector(st, cur_cell_id);
				if (res1.first) { id1 = res1.second; }
				else { id1 = st.size(); st.push_back(cur_cell_id); }

				// extract a column from grad_term
				Matrix grad_term_p_block(grad_term_p(k, { ND, 1 }, { (size_t)grad_term_p.N, 1 }), ND, 1);
				Matrix grad_term_t_block(grad_term_t(k, { ND, 1 }, { (size_t)grad_term_t.N, 1 }), ND, 1);

				rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * grad_term_p_block.values;

				if constexpr (MODE == THERMOPOROELASTIC)
				{
					// each 'column' actually contains n_unknowns columns (ux, uy, uz, p, ..)
					//TODO: define T_VAR = ND + 1;
					rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND + 1, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += r2 * grad_term_t_block.values;
				}
			}
			rest(ND * face_id, { ND }, { 1 }) += r2 * (grad_mult_p * g1.rhs + r2 / lam2 * (grav_vec * (K2n - K1n)).values[0] * B2n).values;

			rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) -= r2 * B2n.values;

			if constexpr (MODE == THERMOPOROELASTIC)
			{
				rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND + 1, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) -= r2 * A2n.values;
			}
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
		if constexpr (MODE == THERMOPOROELASTIC)
		{
		  a_thermal = bc_thm.thermal.a[conn.elem_id2 - mesh->n_cells];
		  b_thermal = bc_thm.thermal.b[conn.elem_id2 - mesh->n_cells];
		}

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

		if constexpr (MODE == THERMOPOROELASTIC)
		{
			A1n = th_exps[cell_id1] * n;
			C1n = heat_conductions[cell_id1] * n; //TODO check units
			lam1_thermal = (n.transpose() * C1n).values[0]; // scalar lambda
			gam1_thermal = C1n - lam1_thermal * n; // bold lambda
		}

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

		if constexpr (MODE == THERMOPOROELASTIC)
		{
			mult_thermal = (bt * I3 + gamma_nnt_mult) * A1n;
			A_thermal = 1.0 / (a_thermal + b_thermal / r1 * lam1_thermal);
		}

		// filling matrix
		A(ND * face_id * A.N, { ND, (uint8_t)A.N }, { (uint8_t)A.N, 1 }) = (at * make_block_diagonal((conn_c - c1_mat).transpose(), ND) +
			bt * nblock_t * C1 + gamma_nnt_mult * (G1 + T1 / r1 * make_block_diagonal((y1 - conn_c).transpose(), ND))).values;

		// filling right-hand side
		res1 = findInVector(st, cell_id1);
		if (res1.first) { id1 = res1.second; }
		else { id1 = st.size(); st.push_back(cell_id1); }

		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1, { ND, ND }, { (size_t)rhs_mult.N, 1 }) = (gamma_nnt_mult * T1 / r1 - at * I3).values;
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = (Ap * bp * lam1 / r1 * mult_p).values;
		if constexpr (MODE == THERMOPOROELASTIC)
			rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND + 1, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = (A_thermal * b_thermal * lam1_thermal / r1 * mult_thermal).values;

		res2 = findInVector(st, cell_id2);
		if (res2.first) { id2 = res2.second; }
		else { id2 = st.size(); st.push_back(cell_id2); }

		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2, { ND, ND }, { (size_t)rhs_mult.N, 1 }) = (gamma_nnt + (I3 - gamma_nnt * L) * P).values;
		rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id2 + ND, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = (mult_p * Ap).values;
		if constexpr (MODE == THERMOPOROELASTIC)
		  rhs_mult(ND* face_id* rhs_mult.N + n_unknowns * id2 + ND + 1, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) = (mult_thermal * A_thermal).values;
		// pressure gradient
		tmp.values = ((-Ap * bp) * (lam1 / r1 * (y1 - conn_c) + gam1).transpose() * P).values;
		const auto& g1 = p_grads[cell_id1];

		if constexpr (MODE == THERMOPOROELASTIC) {
			tmp_thermal.values = ((-A_thermal * b_thermal) * (lam1_thermal / r1 * (y1 - conn_c) + gam1_thermal).transpose() * P).values;
			g1_thermal = &t_grads[cell_id1];
		}
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
		  if constexpr (MODE == THERMOPOROELASTIC) {
			  buf2 = tmp_thermal.values[0] * g1_thermal->a(0, k) +
				  tmp_thermal.values[1] * g1_thermal->a(1, k) +
				  tmp_thermal.values[2] * g1_thermal->a(2, k);
			  rhs_mult(ND * face_id * rhs_mult.N + n_unknowns * id1 + ND + 1, { ND, 1 }, { (size_t)rhs_mult.N, 1 }) += (buf2 * mult_thermal).values;
		  }
		}
		buf1 = tmp.values[0] * g1.rhs(0, 0) +
				tmp.values[1] * g1.rhs(1, 0) +
				  tmp.values[2] * g1.rhs(2, 0);
		rest(ND * face_id, { ND }, { 1 }) += (buf1 * mult_p).values;

		if constexpr (MODE == THERMOPOROELASTIC) {
			buf2 = tmp_thermal.values[0] * g1_thermal->rhs(0, 0) +
				tmp_thermal.values[1] * g1_thermal->rhs(1, 0) +
				tmp_thermal.values[2] * g1_thermal->rhs(2, 0);
			rest(ND * face_id, { ND }, { 1 }) += (buf2 * mult_thermal).values;
		}

		rest(ND * face_id, { ND }, { 1 }) = (mult_p * Ap * bp * (grav_vec * K1n).values[0]).values;

		face_id++;
	  }
	}

	auto& cur_rhs = pre_cur_rhs[n_cur_faces][st.size()]; // take a slice from the pre-allocated matrix with actual size we have
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
void MechDiscretizer<MODE>::calc_interface_approximations()
{
  bool with_thermal = false;
  if constexpr (MODE == THERMOPOROELASTIC)
	with_thermal = true;

  // clear previous approximations
  cell_m.clear();				cell_p.clear();
  flux_stencil.clear();			flux_offset.clear();
  hooke.clear();				hooke_rhs.clear();
  biot_traction.clear();		biot_traction_rhs.clear();
  biot_vol_strain.clear();		biot_vol_strain_rhs.clear();
  darcy.clear();				darcy_rhs.clear();
  fick.clear();					fick_rhs.clear();
  fourier.clear();
  thermal_traction.clear();

  // reserve memory
  cell_m.reserve(mesh->adj_matrix.size());
  cell_p.reserve(mesh->adj_matrix.size());
  flux_stencil.reserve(mesh->adj_matrix.size() * MAX_STENCIL);
  flux_offset.reserve(mesh->adj_matrix.size() + 1);

  hooke.reserve(mesh->adj_matrix.size() * ND * n_unknowns * MAX_STENCIL);
  hooke_rhs.reserve(mesh->adj_matrix.size() * ND);

  biot_traction.reserve(mesh->adj_matrix.size() * ND * MAX_STENCIL);
  biot_traction_rhs.reserve(mesh->adj_matrix.size() * ND);

  biot_vol_strain.reserve(mesh->adj_matrix.size() * n_unknowns * MAX_STENCIL);
  biot_vol_strain.reserve(mesh->adj_matrix.size());

  darcy.reserve(mesh->adj_matrix.size() * MAX_STENCIL);
  darcy_rhs.reserve(mesh->adj_matrix.size());

  fick.reserve(mesh->adj_matrix.size() * MAX_STENCIL);
  fick_rhs.reserve(mesh->adj_matrix.size());

  fourier.reserve(mesh->adj_matrix.size() * MAX_STENCIL);

  thermal_traction.reserve(mesh->adj_matrix.size() * ND * MAX_STENCIL);

  value_t sign;
  index_t cell_id1, cell_id2;
  steady_clock::time_point t1, t2;
  t1 = steady_clock::now();

  flux_offset.push_back(0);
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
		// assemble approximations
		auto& flux = mech_fluxes[0];
		calc_matrix_matrix_mech(conn, flux, cell_id1, conn_id); //!
		calc_matrix_matrix(conn, flux.flow, with_thermal);

		// multiply matrix by area
		flux.hooke.a.values *= conn.area;		  
		flux.biot_traction.a.values *= conn.area;
		flux.vol_strain.a.values *= conn.area;
		flux.flow.darcy.a.values *= sign * conn.area;
		flux.flow.fick.a.values *= sign * conn.area;
		flux.flow.fourier.a.values *= sign * conn.area;
		flux.thermal_traction.a.values *= conn.area;
		// multiply rhs by area
		flux.hooke.rhs.values *= conn.area;
		flux.biot_traction.rhs.values *= conn.area;
		flux.vol_strain.rhs.values *= conn.area;
		flux.flow.darcy.rhs.values *= sign * conn.area;
		flux.flow.fick.rhs.values *= sign * conn.area;
		flux.flow.fourier.rhs.values *= sign * conn.area;
		flux.thermal_traction.rhs.values *= conn.area;

		cell_m.push_back(cell_id1);
		cell_p.push_back(cell_id2);
		write_trans_mech(flux);

		// offset
		flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));
	  }
	  else if (conn.type == mesh::MAT_BOUND)
	  {
		// assemble approximations
		auto& flux = mech_fluxes[0];
		calc_matrix_boundary_mech(conn, flux, conn_id);
		calc_matrix_boundary(conn, flux.flow, with_thermal);

		// multiply matrix by area
		flux.hooke.a.values *= conn.area;
		flux.biot_traction.a.values *= conn.area;
		flux.vol_strain.a.values *= conn.area;
		flux.flow.darcy.a.values *= conn.area;
		flux.flow.fick.a.values *= conn.area;
		flux.flow.fourier.a.values *= conn.area;
		flux.thermal_traction.a.values *= conn.area;
		// multiply rhs by area
		flux.hooke.rhs.values *= conn.area;
		flux.biot_traction.rhs.values *= conn.area;
		flux.vol_strain.rhs.values *= conn.area;
		flux.flow.darcy.rhs.values *= conn.area;
		flux.flow.fick.rhs.values *= conn.area;
		flux.flow.fourier.rhs.values *= conn.area;
		flux.thermal_traction.rhs.values *= conn.area;

		cell_m.push_back(cell_id1);
		cell_p.push_back(cell_id2);
		write_trans_mech(flux);

		// offset
		flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));
	  }
	}
  }

  t2 = steady_clock::now();
  cout << "Find MPFA-MPSA trans: \t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_matrix_matrix_mech(const mesh::Connection& conn,
	MechApproximation<MODE>& flux,
	index_t cell_id,
	index_t conn_id)
{
	Matrix n(ND, 1), det(ND, ND), coef1(ND, ND), coef2(ND, ND), bcoef1(ND, 1), bcoef2(ND, 1), B1n(ND, 1), B2n(ND, 1), T(ND, ND);
	Matrix grad_coef(ND, ND * ND), u_beta_grad_coef(ND, ND * ND), K1n(ND, 1), K2n(ND, 1), gam1(ND, 1), gam2(ND, 1);
	Matrix conn_c(ND, 1), face_unknown_coef(1, ND), biot_grad_coef(ND, ND), mat_diff1(1, ND), mat_diff2(1, ND);
	Vector3 diff1, diff2;
	size_t id1, id2;
	value_t lam1, lam2, det_lam;
	index_t cur_cell_id;
	std::pair<bool, size_t> res1, res2;
	Matrix A1n(ND, 1), A2n(ND, 1), C1n(ND, 1), C2n(ND, 1), gam1_thermal(ND, 1), gam2_thermal(ND, 1), face_unknown_coef_thermal(1, ND);
	value_t lam1_thermal, lam2_thermal, det_lam_thermal;

	const index_t cell_id1 = cell_id;
	const index_t cell_id2 = cell_id1 == conn.elem_id1 ? conn.elem_id2 : conn.elem_id1;

	const auto& x1 = mesh->centroids[cell_id1]; // cell center of cell_id1
	const auto& x2 = mesh->centroids[cell_id2]; // cell center of cell_id2
	conn_c.values = std::valarray<value_t>(conn.c.values.data(), ND);
	bool res;

	// normal vector
	copy_n(std::begin(conn.n.values), ND, std::begin(n.values));
	if (dot(conn.c - x1, conn.n) < 0.0) n.values *= -1.0;

	// gradient in the elastic traction term
	const auto& cur = inner[cell_id1][conn_id];
	det = cur.r1 * cur.T2 + cur.r2 * cur.T1;
	res = det.inv();
	if (!res)
	{
		cout << "Inversion failed!\n";	exit(-1);
	}
	T = cur.T1 * det * cur.T2;
	coef1 = cur.r1 * cur.T2 * det;
	coef2 = cur.r2 * cur.T1 * det;
	const auto& u_grad1 = u_grads[cell_id1];
	const auto& u_grad2 = u_grads[cell_id2];
	flux.hooke = (T * make_block_diagonal((cur.y2 - cur.y1).transpose(), ND) -
		coef1 * cur.G1 - coef2 * cur.G2) * (u_grad1 + u_grad2) / 2.0;

	// gradient for the biot contribution to traction
	B1n = biots[cell_id1] * n;
	B2n = biots[cell_id2] * n;
	K1n = DARCY_CONSTANT * perms[cell_id1] * n;
	K2n = DARCY_CONSTANT * perms[cell_id2] * n;
	lam1 = (n.transpose() * K1n).values[0];
	lam2 = (n.transpose() * K2n).values[0];
	gam1 = K1n - lam1 * n;
	gam2 = K2n - lam2 * n;
	det_lam = 1.0 / (cur.r1 * lam2 + cur.r2 * lam1);
	face_unknown_coef = det_lam * (cur.r2 * lam1 * (conn_c - cur.y1).transpose() +
		cur.r1 * lam2 * (conn_c - cur.y2).transpose() +
		cur.r1 * cur.r2 * (gam2 - gam1).transpose());
	const auto& p_grad1 = p_grads[cell_id1];
	const auto& p_grad2 = p_grads[cell_id2];
	LinearApproximation<Pvar> p_beta = face_unknown_coef * (p_grad1 + p_grad2) / 2.0;

	// gradient for the biot contribution to fluid flow
	mat_diff1.values = (conn_c - cur.y1).values;
	mat_diff2.values = (conn_c - cur.y2).values;
	u_beta_grad_coef = det * (cur.r1 * cur.r2 * (cur.G2 - cur.G1) +
		cur.r2 * cur.T1 * make_block_diagonal(mat_diff1, ND) +
		cur.r1 * cur.T2 * make_block_diagonal(mat_diff2, ND));
	ApproximationType<MODE> u_beta = u_beta_grad_coef * (u_grad1 + u_grad2) / 2.0;

	res1 = findInVector(flux.hooke.stencil, cell_id1);
	if (res1.first) { id1 = res1.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", cell_id1);	exit(-1); }
	flux.hooke.a(n_unknowns * id1, { (size_t)flux.hooke.a.M, ND }, { (size_t)flux.hooke.a.N, 1 }) += T.values;
	p_beta.a(0, id1) += det_lam * cur.r2 * lam1; // need same stencil for pressure and displacement gradients
	u_beta.a(id1 * n_unknowns, { ND, ND }, { (size_t)u_beta.a.N, 1 }) += cur.r2 * (det * cur.T1).values;

	res2 = findInVector(flux.hooke.stencil, cell_id2);
	if (res2.first) { id2 = res2.second; }
	else { printf("Gradient within %d cell does not depend on its value!\n", cell_id2);	exit(-1); }
	flux.hooke.a(n_unknowns * id2, { (size_t)flux.hooke.a.M, ND }, { (size_t)flux.hooke.a.N, 1 }) -= T.values;
	p_beta.a(0, id2) += det_lam * cur.r1 * lam2; // need same stencil for pressure and displacement gradients
	u_beta.a(id2 * n_unknowns, { ND, ND }, { (size_t)u_beta.a.N, 1 }) += cur.r1 * (det * cur.T2).values;

	p_beta.rhs += cur.r1 * cur.r2 * grav_vec * (K1n - K2n); // p_beta assembled

	flux.hooke += coef2 * (B2n - B1n) * p_beta; // Hooke's term assembled

	flux.biot_traction = B1n * p_beta; // Biot's term in traction assembled

	mat_diff1.values = std::valarray<value_t>((conn.c - x1).values.data(), ND);
	mat_diff2.values = std::valarray<value_t>(conn.c.values.data(), ND) - cur.y2.values;
	const auto d_pressure_terms = (outer_product(B2n, mat_diff2 + cur.r2 / lam2 * (gam2 - K1n).transpose()) - outer_product(B1n, mat_diff1)) * p_grad1;

	u_beta -= cur.r1 * cur.r2 * d_pressure_terms; // u_beta assembled
	flux.vol_strain = B1n.transpose() * u_beta;
	// flux.vol_strain.a(id1 * n_unknowns, { ND }, { 1 }) -= B1n.transpose().values; // Biot's term for fluid flow assembled

	if constexpr (MODE == THERMOPOROELASTIC) {
		//compute t_beta see (A.11)
		A1n = th_exps[cell_id1] * n;
		A2n = th_exps[cell_id2] * n;
		C1n = heat_conductions[cell_id1] * n;// TODO units
		C2n = heat_conductions[cell_id2] * n;
		lam1_thermal = (n.transpose() * C1n).values[0];
		lam2_thermal = (n.transpose() * C2n).values[0];
		gam1_thermal = C1n - lam1_thermal * n;
		gam2_thermal = C2n - lam2_thermal * n;
		det_lam_thermal = 1.0 / (cur.r1 * lam2_thermal + cur.r2 * lam1_thermal);
		face_unknown_coef_thermal = det_lam_thermal * (cur.r2 * lam1_thermal * (conn_c - cur.y1).transpose() +
			cur.r1 * lam2_thermal * (conn_c - cur.y2).transpose() +
			cur.r1 * cur.r2 * (gam2_thermal - gam1_thermal).transpose());
		const auto& t_grad1 = t_grads[cell_id1];
		const auto& t_grad2 = t_grads[cell_id2];
		LinearApproximation<Tvar> t_beta = face_unknown_coef_thermal * (t_grad1 + t_grad2) / 2.0;
		t_beta.a(0, id1) += det_lam_thermal * cur.r2 * lam1_thermal; // need same stencil for thermal and displacement gradients
		t_beta.a(0, id2) += det_lam_thermal * cur.r1 * lam2_thermal; // need same stencil for thermal and displacement gradients

		flux.thermal_traction = A1n * t_beta; // thermal term in traction assembled
	}
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_matrix_boundary_mech(const mesh::Connection& conn, 
													  MechApproximation<MODE>& flux, 
													  index_t conn_id)
{
  Matrix c1_mat(ND, 1), conn_mat(ND, 1), n(ND, 1), P(ND, ND), y1(ND, 1);
  Matrix C1(ND * ND, ND * ND), T1(ND, ND), G1(ND, ND * ND), T1inv(ND, ND);
  Matrix K1n(ND, 1), gam1(ND, 1);
  Matrix An(ND, ND), At(ND, ND), L(ND, ND), gamma_nnt(ND, ND), gamma_nnt_mult(ND, ND);
  Matrix coef(ND, ND), coef_u(ND, ND), mult_p(ND, 1), mult_u(ND, ND), gu_coef(ND, ND * ND);
  Matrix nblock(ND * ND, ND), nblock_t(ND, ND * ND), tblock(ND * ND, ND * ND);
  Matrix grad_coef(ND, ND * ND), biot_grad_coef(ND, ND), hooke_p_grad_coef(ND, ND), hooke_t_grad_coef(ND, ND);
  Matrix vol_strain_u_grad_coef(1, ND * ND), vol_strain_p_grad_coef(1, ND);
  Matrix A1n(ND, 1), C1n(ND, 1), gam1_thermal(ND, 1), mult_thermal(ND, 1), thermal_grad_coef(ND, ND);
  value_t lam1_thermal, A_thermal;
  value_t r1, lam1, Ap, gamma;
  value_t a_thermal, b_thermal;
  index_t id1, id2;
  bool res;
  std::pair<bool, size_t> res1, res2;
  const auto& an = bc_thm.mech_normal.a[conn.elem_id2 - mesh->n_cells];
  const auto& bn = bc_thm.mech_normal.b[conn.elem_id2 - mesh->n_cells];
  const auto& at = bc_thm.mech_tangen.a[conn.elem_id2 - mesh->n_cells];
  const auto& bt = bc_thm.mech_tangen.b[conn.elem_id2 - mesh->n_cells];
  const auto& ap = bc_thm.flow.a[conn.elem_id2 - mesh->n_cells];
  const auto& bp = bc_thm.flow.b[conn.elem_id2 - mesh->n_cells];
  if constexpr (MODE == THERMOPOROELASTIC)
  {
	a_thermal = bc_thm.thermal.a[conn.elem_id2 - mesh->n_cells];
	b_thermal = bc_thm.thermal.b[conn.elem_id2 - mesh->n_cells];
  }

  const index_t& cell_id1 = conn.elem_id1;
  const index_t& cell_id2 = conn.elem_id2;
  const auto& c1 = mesh->centroids[cell_id1];
  std::copy_n(c1.values.begin(), ND, std::begin(c1_mat.values));
  std::copy_n(conn.c.values.begin(), ND, std::begin(conn_mat.values));
  std::copy_n(conn.n.values.begin(), ND, std::begin(n.values));

  // Geometry
  if (dot((conn.c - c1), conn.n) < 0)
  {
	n.values *= -1.0;
  }
  P = I3 - outer_product(n, n.transpose());
  r1 = (n.transpose() * (conn_mat - c1_mat))(0, 0);		assert(r1 > 0.0);
  y1 = c1_mat + r1 * n;
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
  K1n = DARCY_CONSTANT * perms[cell_id1] * n;
  lam1 = (n.transpose() * K1n)(0, 0);
  gam1 = K1n - lam1 * n;
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
  mult_u = At * (bt * I3 + gamma_nnt_mult);
  coef = -T1 / r1 * At * (gamma_nnt_mult - r1 * at * T1inv);
  coef_u = -T1 / r1 * mult_u;
  mult_p = biots[cell_id1] * n;
  gu_coef = T1 / r1 * make_block_diagonal((y1 - conn_mat).transpose(), ND) + G1;

  //// Filling mechanics equations
  grad_coef.values = -(coef * gu_coef).values;
  biot_grad_coef.values = outer_product(mult_p, -Ap * bp * (lam1 / r1 * (y1 - conn_mat) + gam1).transpose()).values;
  hooke_p_grad_coef.values = (coef_u * biot_grad_coef).values;

  //// Filling flow equation
  vol_strain_u_grad_coef.values = -(mult_p.transpose() * mult_u * gu_coef).values;
  vol_strain_p_grad_coef.values = -((mult_p.transpose() * (mult_u * mult_p)).values[0] * Ap * bp *
			(lam1 / r1 * (y1 - conn_mat) + gam1)).values;

  //// Assembling fluxes
  flux.hooke = grad_coef * u_grads[cell_id1];
  flux.hooke += hooke_p_grad_coef * p_grads[cell_id1];
  flux.hooke.rhs.values += (coef_u * Ap * bp * (grav_vec * K1n).values[0] * mult_p).values;
  flux.biot_traction = biot_grad_coef * p_grads[cell_id1];
  flux.biot_traction.rhs.values += (Ap * bp * (grav_vec * K1n).values[0] * mult_p).values;
  flux.vol_strain = vol_strain_u_grad_coef * u_grads[cell_id1];
  flux.vol_strain += vol_strain_p_grad_coef * p_grads[cell_id1];
  flux.vol_strain.rhs.values += Ap * (bp * (grav_vec * K1n).values[0]) * (mult_p.transpose() * (mult_u * mult_p)).values[0];

  //thermal_traction and fourier
  if constexpr (MODE == THERMOPOROELASTIC)
  {
	  // Heat conduction decomposition
	  C1n = heat_conductions[cell_id1] * n;
	  lam1_thermal = (n.transpose() * C1n)(0, 0);
	  gam1_thermal = C1n - lam1_thermal * n;
	  // Extra 'boundary' stuff
	  A_thermal = 1.0 / (a_thermal + b_thermal / r1 * lam1_thermal);
	  mult_thermal = th_exps[cell_id1] * n;
	  thermal_grad_coef.values = outer_product(mult_thermal, -A_thermal * b_thermal * (lam1_thermal / r1 * (y1 - conn_mat) + gam1_thermal).transpose()).values;
	  hooke_t_grad_coef.values = (coef_u * thermal_grad_coef).values;
	  flux.hooke += hooke_t_grad_coef * t_grads[cell_id1];
      flux.thermal_traction = thermal_grad_coef * t_grads[cell_id1];
  }

  // matrix cell contribution
  res1 = findInVector(flux.hooke.stencil, cell_id1);
  if (res1.first) { id1 = res1.second; }
  else { printf("Gradient within %d cell does not depend on its value!\n", cell_id1);	exit(-1); }
  flux.hooke.a(n_unknowns * id1, { ND, ND }, { (size_t)flux.hooke.a.N, 1 }) += (coef * T1 / r1).values;
  flux.hooke.a(n_unknowns * id1 + ND, { ND, 1 }, { (size_t)flux.hooke.a.N, 1 }) += (coef_u * mult_p * Ap * bp * lam1 / r1).values;
  flux.biot_traction.a(id1, { ND, 1 }, { (size_t)flux.biot_traction.a.N, 1 }) += (mult_p * Ap * bp * lam1 / r1).values;
  flux.vol_strain.a(n_unknowns * id1, { ND }, { 1 }) += (mult_u * T1 / r1 * mult_p).values;
  flux.vol_strain.a(0, n_unknowns * id1 + ND) += (mult_p.transpose() * (mult_u * mult_p)).values[0] * Ap * bp * lam1 / r1;
  if constexpr (MODE == THERMOPOROELASTIC)
  {
	  flux.hooke.a(n_unknowns * id1 + ND + 1, { ND, 1 }, { (size_t)flux.hooke.a.N, 1 }) += (coef_u * mult_thermal * A_thermal * b_thermal * lam1_thermal / r1).values;
	  flux.thermal_traction.a(id1, { ND, 1 }, { (size_t)flux.thermal_traction.a.N, 1 }) += (mult_thermal * A_thermal * b_thermal * lam1_thermal / r1).values;
  }

  // boundary condition contribution
  res2 = findInVector(flux.hooke.stencil, cell_id2);
  if (res2.first) { id2 = res2.second; }
  else { printf("Gradient within %d cell does not depend on its value!\n", cell_id2);	exit(-1); }
  flux.hooke.a(n_unknowns * id2, { ND, ND }, { (size_t)flux.hooke.a.N, 1 }) += (-T1 / r1 * At * (gamma_nnt + (I3 - gamma_nnt * L) * P)).values;
  flux.hooke.a(n_unknowns * id2 + ND, { ND, 1 }, { (size_t)flux.hooke.a.N, 1 }) += (coef_u * Ap * mult_p).values;
  flux.biot_traction.a(id2, { ND, 1 }, { (size_t)flux.biot_traction.a.N, 1 }) += (Ap * mult_p).values;
  flux.vol_strain.a(n_unknowns * id2, { ND }, { 1 }) += (mult_p.transpose() * At * (gamma_nnt + (I3 - gamma_nnt * L) * P)).values;
  flux.vol_strain.a(0, n_unknowns * id2 + ND) += Ap * (mult_p.transpose() * (mult_u * mult_p)).values[0];
  if constexpr (MODE == THERMOPOROELASTIC)
  {
	  flux.hooke.a(n_unknowns * id2 + ND + 1, { ND, 1 }, { (size_t)flux.hooke.a.N, 1 }) += (coef_u * A_thermal * mult_thermal).values;
	  flux.thermal_traction.a(id2, { ND, 1 }, { (size_t)flux.thermal_traction.a.N, 1 }) += (A_thermal * mult_thermal).values;
  }
}

template <MechDiscretizerMode MODE>
void MechDiscretizer<MODE>::calc_cell_centered_stress_velocity_approximations()
{
  index_t loop_face_id, face_id, n_faces;
  Matrix Ndelta(ND, SUM_N(ND)), Rdelta(ND, SUM_N(ND));
  Matrix sq_mat_flux(ND, ND), sq_mat_stress(SUM_N(ND), SUM_N(ND));
  Vector3 t_face, n;
  bool res;

  // loop through the adjacency matrix (matrix cells)
  for (index_t i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
  {
	// count contributing connections
	n_faces = 0;
	for (loop_face_id = mesh->adj_matrix_offset[i]; loop_face_id < mesh->adj_matrix_offset[i + 1]; loop_face_id++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[loop_face_id]];
	  if (conn.type != mesh::MAT_FRAC) { n_faces++; }
	}

	// choose suitable working matrices
	auto& N = pre_N[n_faces];
	auto& Nflux = pre_Nflux[n_faces];
	auto& R = pre_R[n_faces];
	auto& st_approx = pre_stress_approx[n_faces];
	auto& vel_approx = pre_vel_approx[n_faces];
	
	// assemble matrices for approximation
	for (loop_face_id = mesh->adj_matrix_offset[i], face_id = 0; loop_face_id < mesh->adj_matrix_offset[i + 1]; loop_face_id++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[loop_face_id]];
	  if (conn.type == mesh::MAT_FRAC) { continue; }

	  // vector connecting cell center to the center of interface
	  t_face = conn.c - mesh->centroids[i];
	  n = (dot(conn.n, t_face) > 0 ? conn.n : -conn.n);

	  // N-matrix
	  Ndelta(0, 0) = n.x;					Ndelta(1, 1) = n.y;					  Ndelta(2, 2) = n.z;
	  Ndelta(1, ND + 2) = n.x;				Ndelta(2, ND + 1) = n.x;
	  Ndelta(0, ND + 2) = n.y;				Ndelta(2, ND) = n.y;
	  Ndelta(1, ND) = n.z;					Ndelta(0, ND + 1) = n.z;
	  N(face_id * Ndelta.values.size(), { ND, SUM_N(ND) }, { SUM_N(ND), 1 }) = (conn.area * Ndelta).values;
	  // N-matrix for fluxes
	  Nflux(face_id, 0) = conn.area * n.x;	Nflux(face_id, 1) = conn.area * n.y;  Nflux(face_id, 2) = conn.area * n.z;
	  // R-matrix
	  Rdelta(0, 0) = t_face.x;			  Rdelta(1, 1) = t_face.y;			  Rdelta(2, 2) = t_face.z;
	  Rdelta(1, ND + 2) = t_face.x / 2;	  Rdelta(2, ND + 1) = t_face.x / 2;
	  Rdelta(0, ND + 2) = t_face.y / 2;	  Rdelta(2, ND) = t_face.y / 2;
	  Rdelta(1, ND) = t_face.z / 2;		  Rdelta(0, ND + 1) = t_face.z / 2;
	  R(face_id * Rdelta.values.size(), { ND, SUM_N(ND) }, { SUM_N(ND), 1 }) = Rdelta.values;

	  face_id++;
	}

	// stress reconstruction
	sq_mat_stress = R.transpose() * N;
	res = sq_mat_stress.inv();
	if (!res) { std::cout << "Inversion failed!\n";	exit(-1); };
	st_approx.values = (sq_mat_stress * R.transpose()).values;
	stress_approx.insert(std::end(stress_approx), std::begin(st_approx.values), std::end(st_approx.values));
	// fluid flux reconstruction
	sq_mat_flux = Nflux.transpose() * Nflux;
	res = sq_mat_flux.inv();
	if (!res) { std::cout << "Inversion failed!\n";	exit(-1); };
	vel_approx.values = (sq_mat_flux * Nflux.transpose()).values;
	velocity_approx.insert(std::end(velocity_approx), std::begin(vel_approx.values), std::end(vel_approx.values));
  }
}


template class MechDiscretizer<POROELASTIC>;
template class MechDiscretizer<THERMOPOROELASTIC>;