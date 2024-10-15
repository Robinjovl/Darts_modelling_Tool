#include <vector>
#include <fstream>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <functional>
#include <unordered_set>
#include "discretizer.h"
#include "linalg/matrix.h"

# define M_PI 3.14159265358979323846

using std::vector;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::cout;
using std::endl;
using std::fill;
using std::pair;
using std::fill_n;
using std::copy_n;
using std::unordered_set;
using std::array;
using namespace dis;

const Matrix Discretizer::I3 = Matrix({ 1,0,0, 0,1,0, 0,0,1 }, ND, ND);
const Matrix Discretizer::I4 = Matrix({ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }, ND + 1, ND + 1);

Discretizer::Discretizer()
{
	grav_vec = Matrix({ 0, 0, 0 },  1, ND);
}

Discretizer::~Discretizer()
{
}

void Discretizer::init()
{
	for (index_t i = mesh::MIN_CONNS_PER_ELEM; i <= mesh::MAX_CONNS_PER_ELEM; i++)
	{
		pre_grad_A_p[i] = Matrix(i, ND);
		pre_grad_R_p[i] = Matrix(i, i + 1);
		pre_grad_rhs_p[i] = Matrix(i, 1);
		pre_Wsvd[i] = Matrix(i, i);
		pre_w_svd[i] = Matrix(i, 1);
		pre_Zsvd[i] = Matrix(i, i);
		pre_grad_A_th[i] = Matrix(i, ND);
		pre_grad_R_th[i] = Matrix(i, i + 1);
	}

	const uint8_t BLOCK_SIZE = 1;
	fluxes.resize(MAX_FLUXES_NUM);
	pre_merged_flux.resize(MAX_FLUXES_NUM);
	for (uint8_t k = 0; k < MAX_FLUXES_NUM; k++)
	{
		// Darcy's, Fick's and Fourier's fluxes 
		fluxes[k] = FlowHeatApproximation(MAX_STENCIL);
		// Premerged fluxes
		pre_merged_flux[k] = FlowHeatApproximation(MAX_STENCIL);
	}
}

void Discretizer::set_mesh(Mesh* _mesh)
{
	mesh = _mesh;
}

void Discretizer::calc_tpfa_transmissibilities(const PhysicalTags& tags)
{
	steady_clock::time_point t1, t2;
	t1 = steady_clock::now();

	// allocate memory
	flux_vals.reserve(2 * mesh->adj_matrix.size());
	flux_vals_homo.reserve(2 * mesh->adj_matrix.size());
	flux_vals_thermal.reserve(2 * mesh->adj_matrix.size());
	flux_offset.reserve(mesh->adj_matrix.size() + 1);
	flux_stencil.reserve(2 * mesh->adj_matrix.size());
	flux_rhs.reserve(mesh->adj_matrix.size());
	cell_m.reserve(mesh->adj_matrix.size());
	cell_p.reserve(mesh->adj_matrix.size());

	vector<vector<value_t>> half_trans;
	half_trans.resize(mesh->conns.size());
	vector<vector<value_t>> half_trans_thermal;
	half_trans_thermal.resize(mesh->conns.size());

	// mesh->region_ranges.at(mesh::FRACTURE).second
	for (index_t i = 0; i < mesh->elems.size(); i++)
	{
		index_t el1_tag = mesh->element_tags[i];
		index_t el_id1 = mesh->elems[i].elem_id;

		for (index_t j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
		{
			// ignore boundary -> matrix connections (not matrix -> boundary!)
			if (!tags.at(mesh::BOUNDARY).count(el1_tag))
			{
				// calculate semi-transmissibilites
				value_t T, Td;
				Matrix K = perms[i];
				//Matrix C = cond[i];//TODO use conductivity for half-trans
				Matrix C(3,3); // identical matrix
				for (int ii = 0; ii < 3; ii++)
					for (int jj = 0; jj < 3; jj++)
					  C(ii, jj) = double(ii == jj);


				index_t el_id2;
				const auto& conn = mesh->conns[mesh->adj_matrix[j]];
				if (conn.elem_id1 != el_id1)
					el_id2 = conn.elem_id1;
				else
					el_id2 = conn.elem_id2;

				Vector3 n = mesh->conns[mesh->adj_matrix[j]].n;
				value_t A = mesh->conns[mesh->adj_matrix[j]].area;

				//// TODO: Why have we used that before?
				// Find vector d as the distance from cell centre to projection on interface
				// Vector3 p = mesh->centroids[el_id2] - mesh->centroids[el_id1];
				// double t = dot((conn.c - mesh->centroids[el_id1]), n) / dot(p, n);
				// Vector3 projection = mesh->centroids[i] + p * t;
				// Vector3 temp4 = mesh->centroids[i] - projection;
				// Vector3 d = temp4 / temp4.norm();
				Vector3 conn_center;
				auto &cn = mesh->conns[mesh->adj_matrix[j]];
				conn_center = cn.c;
 
				// use c for first cell half-trans, c_2 for second cell half-trans
				if (mesh->mesh_type == mesh::MESH_TYPE::CPG) {
						if (el_id2 < el_id1)
							conn_center = cn.c_2;
				}
 
				Vector3 d = conn_center - mesh->centroids[el_id1];
				if (dot(d, n) < 0.0) n = -n;
				Vector3 Kn = matrix_vector_product(K, n);
				T = dot(d, Kn) * A / dot(d, d);

				Vector3 Cn = matrix_vector_product(C, n);
				Td = dot(d, Cn) * A / dot(d, d);

				half_trans[mesh->conns[mesh->adj_matrix[j]].conn_id].push_back(T);
				half_trans_thermal[mesh->conns[mesh->adj_matrix[j]].conn_id].push_back(Td);

#ifdef DEBUG_TRANS
				std::cout << "----- TPFA CPP -----" << std::endl;
				std::cout << "Connection cells (local): " << conn.elem_id1 << " ";
				std::cout << conn.elem_id2 << " " << std::endl;
				//std::cout << "Connection cells (global): " << mesh->local_to_global[conn.elem_id1] << " ";
				//std::cout << mesh->local_to_global[conn.elem_id2] << " " << std::endl;
				//std::cout << "Connection cells: " << mesh->get_ijk_as_str(conn.elem_id1, false) << " ";
				//std::cout << mesh->get_ijk_as_str(conn.elem_id2, false) << " " << std::endl;
				std::cout << "Connection center: " << conn_center.x << " " << conn_center.y << " ";
				std::cout << conn_center.z << std::endl;
				std::cout << "d vector: " << d.x << " " << d.y << " " << d.z << std::endl;
				std::cout << "n vector: " << n.x << " " << n.y << " " << n.z << std::endl;
				std::cout << "T for connection =" << T << std::endl;
				std::cout << "Area for connection =" << A << std::endl;
				std::cout << "dot(d, Kn) for connection =" << dot(d, Kn) << std::endl;
				std::cout << "std::abs(dot(n, d)) for connection =" << std::abs(dot(n, d)) << std::endl;
				//std::cout << "temp4.norm() for connection =" << temp4.norm() << std::endl;
				std::cout << "Between elements: " << conn.elem_id1 << " " << conn.elem_id2 << std::endl;
				std::cout << "Centroid of elem1: " << mesh->centroids[el_id1].x << " " << mesh->centroids[el_id1].y << " " << mesh->centroids[el_id1].z << std::endl;
				std::cout << "Centroid of elem2: " << mesh->centroids[el_id2].x << " " << mesh->centroids[el_id2].y << " " << mesh->centroids[el_id2].z << std::endl;
#endif // DEBUG_TRANS
			}
		}
	}

#ifdef DEBUG_TRANS
	vector<index_t> cell_i_idx, cell_j_idx;;
	vector<value_t> trans_mat_mat, trans_mat_mat_d;
#endif //DEBUG_TRANS

	index_t counter = 0;
	for (int i = 0; i < half_trans.size(); i++)
	{
		// take harmonic average of the 2 semi-transmissibilities and append to trans
		if (half_trans[i].size() == 2)
		{
			value_t det_t = half_trans[i][0] + half_trans[i][1];
			value_t Transmissibility;
			if (det_t < 1e-12)
				Transmissibility = 0.;
			else
				Transmissibility = (half_trans[i][0] * half_trans[i][1]) / det_t;

			value_t det_t_thermal = half_trans_thermal[i][0] + half_trans_thermal[i][1];
			value_t Transmissibility_thermal;
			if (det_t_thermal < 1e-12)
				Transmissibility_thermal = 0.;
			else
				Transmissibility_thermal = (half_trans_thermal[i][0] * half_trans_thermal[i][1]) / det_t_thermal;

			// index of trans2d is the conn id of that interface
			cell_m.push_back(mesh->conns[i].elem_id1);
			cell_p.push_back(mesh->conns[i].elem_id2);
			flux_stencil.push_back(mesh->conns[i].elem_id1);
			flux_stencil.push_back(mesh->conns[i].elem_id2);

			flux_vals.push_back(Transmissibility * DARCY_CONSTANT);
			flux_vals.push_back(-1 * Transmissibility * DARCY_CONSTANT);
			flux_rhs.push_back(Transmissibility* DARCY_CONSTANT* dot(grav_vec, (mesh->centroids[mesh->conns[i].elem_id2] - mesh->centroids[mesh->conns[i].elem_id1])));
			flux_vals_thermal.push_back(Transmissibility_thermal);
			flux_vals_thermal.push_back(-Transmissibility_thermal);
			flux_offset.push_back(counter);
			counter += static_cast<index_t>(half_trans[i].size());

			cell_m.push_back(mesh->conns[i].elem_id2);
			cell_p.push_back(mesh->conns[i].elem_id1);
			flux_stencil.push_back(mesh->conns[i].elem_id1);
			flux_stencil.push_back(mesh->conns[i].elem_id2);

			flux_vals.push_back(-1 * Transmissibility * DARCY_CONSTANT);
			flux_vals.push_back(Transmissibility* DARCY_CONSTANT);
			flux_rhs.push_back(-Transmissibility * DARCY_CONSTANT * dot(grav_vec, (mesh->centroids[mesh->conns[i].elem_id2] - mesh->centroids[mesh->conns[i].elem_id1])));
			flux_vals_thermal.push_back(-Transmissibility_thermal);
			flux_vals_thermal.push_back(Transmissibility_thermal);
			flux_offset.push_back(counter);
			counter += static_cast<index_t>(half_trans[i].size());
#ifdef DEBUG_TRANS
			cell_i_idx.push_back(mesh->conns[i].elem_id1);
			cell_j_idx.push_back(mesh->conns[i].elem_id2);
			trans_mat_mat.push_back(Transmissibility);
			trans_mat_mat_d.push_back(Transmissibility * DARCY_CONSTANT);
#endif //DEBUG_TRANS
#ifdef DEBUG_TRANS
			std::cout << "CPP Transmissibilty for MAT connection  (" << mesh->conns[i].elem_id1 << ", " << mesh->conns[i].elem_id2 << ") = " << Transmissibility << "  * darcy =" << Transmissibility * DARCY_CONSTANT << std::endl;
#endif // DEBUG_TRANS
			//myfile << std::fixed << std::setprecision(6) << centroid[i][0].x << " " << centroid[i][0].y << " " << centroid[i][0].z << " " << centroid[i][1].x << " " << centroid[i][1].y << " " << centroid[i][1].z << " " << Transmissibility << " " << trans2d[i][0] << " " << trans2d[i][1] << "\n";
			// myfile << std::setprecision(6) << centroid[i][0].x << " " << centroid[i][0].y << " " << centroid[i][0].z << " " << centroid[i][1].x << " " << centroid[i][1].y << " " << centroid[i][1].z << " " << Transmissibility * DARCY_CONSTANT << " " << dvecs[i][0].x << " " << dvecs[i][0].y << " " << dvecs[i][0].z << " " << dvecs[i][1].x << " " << dvecs[i][1].y << " " << dvecs[i][1].z << " " << nvecs[i][0].x << " " << nvecs[i][0].y << " " << nvecs[i][0].z << " " << nvecs[i][1].x << " " << nvecs[i][1].y << " " << nvecs[i][1].z << "\n";
			//myfile << std::setprecision(6) << centroid[i][0].x << " " << centroid[i][0].y << " " << centroid[i][0].z << " " << centroid[i][1].x << " " << centroid[i][1].y << " " << centroid[i][1].z << " " << Transmissibility * DARCY_CONSTANT << " " << mesh->conns[mesh->adj_matrix[i]].area << " " << dvecs[i][0].x << " " << dvecs[i][0].y << " " << dvecs[i][0].z << " " << dvecs[i][1].x << " " << dvecs[i][1].y << " " << dvecs[i][1].z << " " << nvecs[i][0].x << " " << nvecs[i][0].y << " " << nvecs[i][0].z << "\n";
			//myfile << std::setprecision(6) << centroid[i][0].x << " " << centroid[i][0].y << " " << centroid[i][0].z << " " << centroid[i][1].x << " " << centroid[i][1].y << " " << centroid[i][1].z << " " << Transmissibility * DARCY_CONSTANT << "\n";

		}
		// for boundary transmissibilities
		if (half_trans[i].size() == 1)
		{
			flux_vals.push_back(half_trans[i][0] * DARCY_CONSTANT);
			flux_vals_thermal.push_back(half_trans_thermal[i][0]);
			flux_offset.push_back(counter);
			counter += static_cast<index_t>(half_trans[i].size());
			cell_m.push_back(mesh->conns[i].elem_id1);
			cell_p.push_back(mesh->conns[i].elem_id2);

			if (mesh->elems[mesh->conns[i].elem_id1].loc == mesh::BOUNDARY) {
				flux_stencil.push_back(mesh->conns[i].elem_id1);
			}
			else {
				flux_stencil.push_back(mesh->conns[i].elem_id2);
			}

			flux_rhs.push_back(0.0);
#ifdef DEBUG_TRANS
			std::cout << "CPP Transmissibilty for BND connection (" << mesh->conns[i].elem_id1 <<", "<< mesh->conns[i].elem_id2 << ") = "<< 
				half_trans[i][0] << "\t *darcy = " << half_trans[i][0] * DARCY_CONSTANT << std::endl;
#endif // DEBUG_TRANS

			//trans.push_back(-1 * trans2d[i][0] * DARCY_CONSTANT);
			//trans_offset.push_back(counter);
			//counter += trans2d[i].size();
			//stencil.push_back(mesh->conns[i].elem_id2);
			// myfile << centroid[i][0].x << " " << centroid[i][0].y << " " << centroid[i][0].z << " " << trans2d[i][0] << "\n";
		}
	}

	t2 = steady_clock::now();
	cout << "Find TPFA trans:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;

	flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));

	//myfile.close();

#ifdef DEBUG_TRANS
	std::ofstream file_trans;
	file_trans.open("cppTPFA_mat_mat.dat");
	for (int i = 0; i < cell_i_idx.size(); i++) {
		file_trans << cell_i_idx[i] << "\t";
		file_trans << cell_j_idx[i] << "\t";
		file_trans << trans_mat_mat[i] << "\t";
		file_trans << trans_mat_mat_d[i] << "\n";
	}
	file_trans.close();
#endif //DEBUG_TRANS
}

void Discretizer::reconstruct_pressure_gradients_per_cell(const BoundaryCondition& _bc)
{
	// allocate memory for arrays
	p_grads.resize(mesh->n_cells, LinearApproximation<Pvar>(ND, MAX_STENCIL));

	steady_clock::time_point t1, t2;
	t1 = steady_clock::now();

	bc_flow = _bc;

	// viscosity should be provided from outside
	const value_t mu = 1.0;
	index_t el_id1, el_id2;
	uint8_t conns_num, stencil_size;
	value_t d2, lambda2, scale_boundary;
	Vector3 n, x1, x2, temp;
	Matrix to_invert(ND, ND);
	bool no_neumann_conns, res;

	// loop through the adjacency matrix (matrix cells)
	for (int i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
	{
		// Resize the dims of A and R that depend on the amount of connections
		conns_num = 0;// mesh->adj_matrix_offset[i + 1] - mesh->adj_matrix_offset[i];
		for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
		{
			const auto& conn = mesh->conns[mesh->adj_matrix[j]];
			if (conn.type == mesh::MAT_MAT || 
				conn.type == mesh::MAT_FRAC || 
				conn.type == mesh::FRAC_MAT || 
				conn.type == mesh::MAT_BOUND)
				conns_num++;
		}
		stencil_size = conns_num + 1;
		if (conns_num != 0)
		{
			// matrix of coefficients in front of gradients in equations
			auto& A = pre_grad_A_p[conns_num];
			A.values = 0.0;
			// matrix of coefficients in front of pressures and boundary conditions in equations
			auto& R = pre_grad_R_p[conns_num];
			R.values = 0.0;
			// free term (gravity) in equations
			auto& rhs = pre_grad_rhs_p[conns_num];
			rhs.values = 0.0;

			const auto& el1 = mesh->elems[i];
			x1 = mesh->centroids[i];
			el_id1 = el1.elem_id;
			std::vector<index_t> temp_stencil(conns_num + 1);

			no_neumann_conns = true;
			index_t counter = 0;
			for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
			{
				const auto& conn = mesh->conns[mesh->adj_matrix[j]];
				if (conn.elem_id1 != el_id1)
					el_id2 = conn.elem_id1;
				else
					el_id2 = conn.elem_id2;

				n = conn.n;
				if (dot((conn.c - mesh->centroids[i]), n) < 0) n = -n;

				// check the connection type (whether is a matrix matrix or boundary matrix)
				if (conn.type == mesh::MAT_MAT || conn.type == mesh::MAT_FRAC || conn.type == mesh::FRAC_MAT)
				{
					// Location of connecting element centroid
					x2 = mesh->centroids[el_id2];
					// Projection from centroid of connecting element to interface surface
					d2 = abs(dot(x2 - conn.c, n));
					// Coefficient lambda for each connection
					lambda2 = dot(n, matrix_vector_product(perms[el_id2], n));

					if (lambda2 > EQUALITY_TOLERANCE)
					{
						// pseudo: A[row] = x2 - x1 + d2 / lambda2 * (Perm[cell_1] - Perm[cell_2]) * n
						// equation 17 Terekhov's paper
						temp = (d2 / lambda2) * matrix_vector_product(perms[el_id1] - perms[el_id2], n);
						A(counter, 0) = x2.x - x1.x + temp.x;
						A(counter, 1) = x2.y - x1.y + temp.y;
						A(counter, 2) = x2.z - x1.z + temp.z;

						R(counter, R.N - 1) = -1.0;
						R(counter, counter) = 1.0;

						rhs(counter, 0) = dot(grav_vec, temp);
					}
					else
					{
						temp = matrix_vector_product(perms[i], n);
						A(counter, 0) = temp.x;
						A(counter, 1) = temp.y;
						A(counter, 2) = temp.z;

						rhs(counter, 0) = dot(grav_vec, temp);
					}
					temp_stencil[counter++] = el_id2;
				}
				else if (conn.type == mesh::MAT_BOUND)
				{
					//TODO Implement other BC

					// Coefficients that define boundary condition
					const auto& alpha = bc_flow.a[conn.elem_id2 - mesh->n_cells];
					const auto& beta = bc_flow.b[conn.elem_id2 - mesh->n_cells];

					temp = beta / mu * matrix_vector_product(perms[i], n);

					// scaling factor
					if (alpha != 1.0 || beta != 0.0)
					{
						no_neumann_conns = false;
						scale_boundary = sqrt((conn.c.x - x1.x) * (conn.c.x - x1.x) + (conn.c.y - x1.y) * (conn.c.y - x1.y) + (conn.c.z - x1.z) * (conn.c.z - x1.z))
							/ sqrt(temp.x * temp.x + temp.y * temp.y + temp.z * temp.z);
						if (scale_boundary != scale_boundary || std::isinf(scale_boundary))
							scale_boundary = 1.0;
					}
					else
					{
						scale_boundary = 1.0;
					}

					A(counter, 0) = scale_boundary * (alpha * (conn.c.x - x1.x) + temp.x);
					A(counter, 1) = scale_boundary * (alpha * (conn.c.y - x1.y) + temp.y);
					A(counter, 2) = scale_boundary * (alpha * (conn.c.z - x1.z) + temp.z);

					// update the row on matrix R
					// the p1 term has to be -alpha, all other elements have to be 0 except for the last
					// element which is going to be rp
					R(counter, R.N - 1) = -scale_boundary * alpha;
					R(counter, counter) = scale_boundary;

					rhs(counter, 0) = scale_boundary * dot(grav_vec, temp);
					temp_stencil[counter++] = el_id2;
				}
			}
			temp_stencil[counter] = el_id1;

			// computing the least squares solution for nabla p = (A^T * A) ^(-1) * (A^T) * R
			to_invert = A.transpose() * A;
			try {
				to_invert.inv();

				// check inversion
				for (const auto& val : to_invert.values)
					assert(val == val && std::isfinite(val));

				auto& cur_grad = p_grads[i];
				cur_grad.a = to_invert * A.transpose() * R;
				cur_grad.rhs = to_invert * A.transpose() * rhs;
				cur_grad.stencil = temp_stencil;
				cur_grad.sort();
			}
			catch (const std::exception&)
			{
				throw "Matrix is not invertible";
			}

#ifdef DEBUG_TRANS
			// check sum of coefficients
			if (no_neumann_conns)
			{
                // recover gradient matrix from array
                Matrix cur(ND, conns_num+1);

                int grad_value_idx = ND * grad_offset.back();
                for (int row = 0; row < cur.M; row++) 
				{
                    for (int col = 0; col < cur.N; col++) 
					{
                        cur(row, col) = grad_vals[grad_value_idx++];
                    }
                }

				for (uint8_t c = 0; c < ND; c++)
				{
					lambda2 = fabs(std::accumulate(&cur.values[c * cur.N], &cur.values[c * cur.N] + cur.N, 0.0));
					assert(lambda2 < EQUALITY_TOLERANCE);
				}
			}
#endif /* DEBUG_TRANS */
		}
	}

	// loop through the adjacency matrix (fracture cells)
	for (int i = mesh->region_ranges.at(mesh::FRACTURE).first; i < mesh->region_ranges.at(mesh::FRACTURE).second; i++)
	{
		// Resize the dims of A and R that depend on the amount of connections
		conns_num = 0;// mesh->adj_matrix_offset[i + 1] - mesh->adj_matrix_offset[i];
		for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
		{
			const auto& conn = mesh->conns[mesh->adj_matrix[j]];
			if (conn.type == mesh::FRAC_FRAC || conn.type == mesh::FRAC_FRAC)
				conns_num++;
		}

		stencil_size = conns_num + 1;
		if (conns_num != 0)
		{
			// matrix of coefficients in front of gradients in equations
			auto& A = pre_grad_A_p[conns_num];
			A.values = 0.0;
			// matrix of coefficients in front of pressures and boundary conditions in equations
			auto& R = pre_grad_R_p[conns_num];
			R.values = 0.0;
			// free term (gravity) in equations
			auto& rhs = pre_grad_rhs_p[conns_num];
			rhs.values = 0.0;

			const auto& el1 = mesh->elems[i];
			x1 = mesh->centroids[i];
			el_id1 = el1.elem_id;
			std::vector<index_t> temp_stencil(conns_num + 1);

			no_neumann_conns = true;
			index_t counter = 0;
			for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
			{
				const auto& conn = mesh->conns[mesh->adj_matrix[j]];
				if (conn.elem_id1 != el_id1)
					el_id2 = conn.elem_id1;
				else
					el_id2 = conn.elem_id2;

				n = conn.n;
				if (dot((conn.c - mesh->centroids[i]), n) < 0) n = -n;

				// check the connection type (whether is a matrix matrix or boundary matrix)
				if (conn.type == mesh::FRAC_FRAC)
				{
					// Location of connecting element centroid
					x2 = mesh->centroids[el_id2];
					// Projection from centroid of connecting element to interface surface
					d2 = abs(dot(x2 - conn.c, n));
					// Coefficient lambda for each connection
					lambda2 = dot(n, matrix_vector_product(perms[el_id2], n));

					if (lambda2 > EQUALITY_TOLERANCE)
					{
						// pseudo: A[row] = x2 - x1 + d2 / lambda2 * (Perm[cell_1] - Perm[cell_2]) * n
						// equation 17 Terekhov's paper
						temp = (d2 / lambda2) * matrix_vector_product(perms[el_id1] - perms[el_id2], n);
						A(counter, 0) = x2.x - x1.x + temp.x;
						A(counter, 1) = x2.y - x1.y + temp.y;
						A(counter, 2) = x2.z - x1.z + temp.z;

						R(counter, R.N - 1) = -1.0;
						R(counter, counter) = 1.0;

						rhs(counter, 0) = dot(grav_vec, temp);
					}
					else
					{
						temp = matrix_vector_product(perms[i], n);
						A(counter, 0) = temp.x;
						A(counter, 1) = temp.y;
						A(counter, 2) = temp.z;

						rhs(counter, 0) = dot(grav_vec, temp);
					}
					temp_stencil[counter++] = el_id2;
				}
				else if (conn.type == mesh::FRACTURE_BOUNDARY)
				{
					temp_stencil[counter++] = el_id2;
				}

			}
			temp_stencil[counter] = el_id1;

			// inversion
			// SVD is produced for M x N matrix where M >= N, decompose transposed otherwise
			counter = (counter > ND) ? ND : counter;
			//assert(face_count_id == n_cur_faces);
			auto& Wsvd = pre_Wsvd[counter];
			auto& Zsvd = pre_Zsvd[counter];
			auto& w_svd = pre_w_svd[counter].values;
			std::fill_n(&Wsvd.values[0], Wsvd.values.size(), 0.0);
			std::fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
			std::fill_n(&w_svd[0], w_svd.size(), 0.0);
			try {
				Matrix tempGrad, rhsGrad;
				if (A.M >= A.N)
				{
					// SVD decomposition A = M W Z*
					if (Zsvd.M != A.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
					res = A.svd(Zsvd, w_svd);
					assert(Zsvd.M == counter && Zsvd.N == counter);
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
					tempGrad = Zsvd * Wsvd * A.transpose() * R;
					rhsGrad = Zsvd * Wsvd * A.transpose() * rhs;
				}
				else
				{
					// SVD decomposition A* = M W Z*
					A.transposeInplace();
					if (Zsvd.M != A.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
					res = A.svd(Zsvd, w_svd);
					assert(Zsvd.M == counter && Zsvd.N == counter);
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
					tempGrad = A * Wsvd * Zsvd.transpose() * R;
					rhsGrad = A * Wsvd * Zsvd.transpose() * rhs;
					A.transposeInplace();
				}

				auto& cur_grad = p_grads[i];
				cur_grad.a = tempGrad;
				cur_grad.rhs = rhsGrad;
				cur_grad.stencil = temp_stencil;
				cur_grad.sort();
			}
			catch (const std::exception&)
			{
				throw "Matrix is not invertible";
			}

#ifdef DEBUG_TRANS
			// check sum of coefficients
			if (no_neumann_conns)
			{
				// recover gradient matrix from array
				Matrix cur(ND, conns_num + 1);

				int grad_value_idx = ND * grad_offset.back();
				for (int row = 0; row < cur.M; row++)
				{
					for (int col = 0; col < cur.N; col++)
					{
						cur(row, col) = grad_vals[grad_value_idx++];
					}
				}

				for (uint8_t c = 0; c < ND; c++)
				{
					lambda2 = fabs(std::accumulate(&cur.values[c * cur.N], &cur.values[c * cur.N] + cur.N, 0.0));
					//assert(lambda2 < EQUALITY_TOLERANCE);
				}
			}
#endif /* DEBUG_TRANS */
		}
	}

	t2 = steady_clock::now();
	cout << "Reconstruction of gradients:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

void Discretizer::reconstruct_pressure_temperature_gradients_per_cell(const BoundaryCondition& _bc_flow, const BoundaryCondition& _bc_heat)
{
  // allocate memory for arrays
  p_grads.resize(mesh->n_cells, LinearApproximation<Pvar>(ND, MAX_STENCIL));
  t_grads.resize(mesh->n_cells, LinearApproximation<Tvar>(ND, MAX_STENCIL));

  steady_clock::time_point t1, t2;
  t1 = steady_clock::now();

  bc_flow = _bc_flow;
  bc_heat = _bc_heat;

  // viscosity should be provided from outside
  const value_t mu = 1.0;
  index_t el_id1, el_id2;
  uint8_t conns_num, stencil_size;
  value_t d2, lambda2, kappa2, scale_boundary;
  Vector3 n, x1, x2, temp;
  Matrix to_invert(ND, ND);
  bool no_neumann_conns, res;
  std::vector<std::pair<index_t, index_t>> sort_vec(MAX_STENCIL);

  // loop through the adjacency matrix (matrix cells)
  for (int i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
  {
	// Resize the dims of A and R that depend on the amount of connections
	conns_num = 0;// mesh->adj_matrix_offset[i + 1] - mesh->adj_matrix_offset[i];
	for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[j]];
	  if (conn.type == mesh::MAT_MAT ||
		conn.type == mesh::MAT_FRAC ||
		conn.type == mesh::FRAC_MAT ||
		conn.type == mesh::MAT_BOUND)
		conns_num++;
	}
	stencil_size = conns_num + 1;
	if (conns_num != 0)
	{
	  // matrix of coefficients in front of pressure gradients in equations
	  auto& A_p = pre_grad_A_p[conns_num];
	  A_p.values = 0.0;
	  // matrix of coefficients in front of pressures and boundary conditions in equations
	  auto& R_p = pre_grad_R_p[conns_num];
	  R_p.values = 0.0;
	  // free term (gravity) in equations
	  auto& rhs_p = pre_grad_rhs_p[conns_num];
	  rhs_p.values = 0.0;
	  // matrix of coefficients in front of temperature gradients in equations
	  auto& A_th = pre_grad_A_th[conns_num];
	  A_th.values = 0.0;
	  // matrix of coefficients in front of temperature and boundary conditions in equations
	  auto& R_th = pre_grad_R_th[conns_num];
	  R_th.values = 0.0;


	  const auto& el1 = mesh->elems[i];
	  x1 = mesh->centroids[i];
	  el_id1 = el1.elem_id;
	  std::vector<index_t> temp_stencil(conns_num + 1);

	  no_neumann_conns = true;
	  index_t counter = 0;
	  for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
	  {
		const auto& conn = mesh->conns[mesh->adj_matrix[j]];
		if (conn.elem_id1 != el_id1)
		  el_id2 = conn.elem_id1;
		else
		  el_id2 = conn.elem_id2;

		n = conn.n;
		if (dot((conn.c - mesh->centroids[i]), n) < 0) n = -n;

		// check the connection type (whether is a matrix matrix or boundary matrix)
		if (conn.type == mesh::MAT_MAT || conn.type == mesh::MAT_FRAC || conn.type == mesh::FRAC_MAT)
		{
		  // Location of connecting element centroid
		  x2 = mesh->centroids[el_id2];
		  // Projection from centroid of connecting element to interface surface
		  d2 = abs(dot(x2 - conn.c, n));
		  // Co-normal hydraulic conductivity for each connection
		  lambda2 = dot(n, matrix_vector_product(perms[el_id2], n));
		  // Co-normal heat conductivity for each connection
		  kappa2 = dot(n, matrix_vector_product(heat_conductions[el_id2], n));

		  // fluid flux balance
		  if (lambda2 > EQUALITY_TOLERANCE)
		  {
			// pseudo: A[row] = x2 - x1 + d2 / lambda2 * (Perm[cell_1] - Perm[cell_2]) * n
			// equation 17 Terekhov's paper
			temp = (d2 / lambda2) * matrix_vector_product(perms[el_id1] - perms[el_id2], n);
			A_p(counter, 0) = x2.x - x1.x + temp.x;
			A_p(counter, 1) = x2.y - x1.y + temp.y;
			A_p(counter, 2) = x2.z - x1.z + temp.z;

			R_p(counter, R_p.N - 1) = -1.0;
			R_p(counter, counter) = 1.0;

			rhs_p(counter, 0) = dot(grav_vec, temp);
		  }
		  else
		  {
			temp = matrix_vector_product(perms[i], n);
			A_p(counter, 0) = temp.x;
			A_p(counter, 1) = temp.y;
			A_p(counter, 2) = temp.z;

			rhs_p(counter, 0) = dot(grav_vec, temp);
		  }
		  // heat conduction flux balance
		  if (kappa2 > EQUALITY_TOLERANCE)
		  {
			temp = (d2 / kappa2) * matrix_vector_product(heat_conductions[el_id1] - heat_conductions[el_id2], n);
			A_th(counter, 0) = x2.x - x1.x + temp.x;
			A_th(counter, 1) = x2.y - x1.y + temp.y;
			A_th(counter, 2) = x2.z - x1.z + temp.z;

			R_th(counter, R_p.N - 1) = -1.0;
			R_th(counter, counter) = 1.0;
		  }
		  else
		  {
			temp = matrix_vector_product(heat_conductions[i], n);
			A_th(counter, 0) = temp.x;
			A_th(counter, 1) = temp.y;
			A_th(counter, 2) = temp.z;
		  }

		  temp_stencil[counter++] = el_id2;
		}
		else if (conn.type == mesh::MAT_BOUND)
		{
		  //TODO Implement other BC

		  //// fluid flux constraint
		  // Coefficients that define pressure boundary condition
		  const auto& alpha_p = bc_flow.a[conn.elem_id2 - mesh->n_cells];
		  const auto& beta_p = bc_flow.b[conn.elem_id2 - mesh->n_cells];

		  temp = beta_p / mu * matrix_vector_product(perms[i], n);

		  // scaling factor
		  if (alpha_p != 1.0 || beta_p != 0.0)
		  {
			no_neumann_conns = false;
			scale_boundary = sqrt((conn.c.x - x1.x) * (conn.c.x - x1.x) + (conn.c.y - x1.y) * (conn.c.y - x1.y) + (conn.c.z - x1.z) * (conn.c.z - x1.z))
			  / sqrt(temp.x * temp.x + temp.y * temp.y + temp.z * temp.z);
			if (scale_boundary != scale_boundary || std::isinf(scale_boundary))
			  scale_boundary = 1.0;
		  }
		  else
		  {
			scale_boundary = 1.0;
		  }

		  A_p(counter, 0) = scale_boundary * (alpha_p * (conn.c.x - x1.x) + temp.x);
		  A_p(counter, 1) = scale_boundary * (alpha_p * (conn.c.y - x1.y) + temp.y);
		  A_p(counter, 2) = scale_boundary * (alpha_p * (conn.c.z - x1.z) + temp.z);

		  // update the row on matrix R
		  // the p1 term has to be -alpha, all other elements have to be 0 except for the last
		  // element which is going to be rp
		  R_p(counter, R_p.N - 1) = -scale_boundary * alpha_p;
		  R_p(counter, counter) = scale_boundary;

		  rhs_p(counter, 0) = scale_boundary * dot(grav_vec, temp);

		  //// heat conduction flux constraint
		  // Coefficients that define pressure boundary condition
		  const auto& alpha_th = bc_heat.a[conn.elem_id2 - mesh->n_cells];
		  const auto& beta_th = bc_heat.b[conn.elem_id2 - mesh->n_cells];

		  temp = beta_th * matrix_vector_product(heat_conductions[i], n);

		  // scaling factor
		  if (alpha_th != 1.0 || beta_th != 0.0)
		  {
			no_neumann_conns = false;
			scale_boundary = sqrt((conn.c.x - x1.x) * (conn.c.x - x1.x) + (conn.c.y - x1.y) * (conn.c.y - x1.y) + (conn.c.z - x1.z) * (conn.c.z - x1.z))
			  / sqrt(temp.x * temp.x + temp.y * temp.y + temp.z * temp.z);
			if (scale_boundary != scale_boundary || std::isinf(scale_boundary))
			  scale_boundary = 1.0;
		  }
		  else
		  {
			scale_boundary = 1.0;
		  }

		  A_th(counter, 0) = scale_boundary * (alpha_th * (conn.c.x - x1.x) + temp.x);
		  A_th(counter, 1) = scale_boundary * (alpha_th * (conn.c.y - x1.y) + temp.y);
		  A_th(counter, 2) = scale_boundary * (alpha_th * (conn.c.z - x1.z) + temp.z);

		  // update the row on matrix R
		  R_th(counter, R_th.N - 1) = -scale_boundary * alpha_th;
		  R_th(counter, counter) = scale_boundary;

		  temp_stencil[counter++] = el_id2;
		}
	  }
	  temp_stencil[counter] = el_id1;

	  // computing the least squares solution for nabla p = (A^T * A) ^(-1) * (A^T) * R
	  to_invert = A_p.transpose() * A_p;
	  try {
		to_invert.inv();

		// check inversion
		for (const auto& val : to_invert.values)
		  assert(val == val && std::isfinite(val));

		auto& cur_grad = p_grads[i];
		cur_grad.a = to_invert * A_p.transpose() * R_p;
		cur_grad.rhs = to_invert * A_p.transpose() * rhs_p;
		cur_grad.stencil = temp_stencil;
		cur_grad.sort();
	  }
	  catch (const std::exception&)
	  {
		throw "Pressure gradient matrix is not invertible";
	  }

	  // computing the least squares solution for nabla \theta = (A^T * A) ^(-1) * (A^T) * R
	  to_invert = A_th.transpose() * A_th;
	  try {
		to_invert.inv();

		// check inversion
		for (const auto& val : to_invert.values)
		  assert(val == val && std::isfinite(val));

		Matrix tempGrad = to_invert * A_th.transpose() * R_th;

		auto& cur_grad = t_grads[i];
		cur_grad.a = to_invert * A_th.transpose() * R_th;
		cur_grad.stencil = temp_stencil;
		cur_grad.sort();
	  }
	  catch (const std::exception&)
	  {
		throw "Temperature gradient matrix is not invertible";
	  }

#ifdef DEBUG_TRANS
	  // check sum of coefficients
	  if (no_neumann_conns)
	  {
		// recover gradient matrix from array
		Matrix cur(ND, conns_num + 1);

		int grad_value_idx = ND * grad_offset.back();
		for (int row = 0; row < cur.M; row++)
		{
		  for (int col = 0; col < cur.N; col++)
		  {
			cur(row, col) = grad_vals[grad_value_idx++];
		  }
		}

		for (uint8_t c = 0; c < ND; c++)
		{
		  lambda2 = fabs(std::accumulate(&cur.values[c * cur.N], &cur.values[c * cur.N] + cur.N, 0.0));
		  assert(lambda2 < EQUALITY_TOLERANCE);
		}
	  }
#endif /* DEBUG_TRANS */
	}
  }

  // loop through the adjacency matrix (fracture cells)
  for (int i = mesh->region_ranges.at(mesh::FRACTURE).first; i < mesh->region_ranges.at(mesh::FRACTURE).second; i++)
  {
	// Resize the dims of A and R that depend on the amount of connections
	conns_num = 0;// mesh->adj_matrix_offset[i + 1] - mesh->adj_matrix_offset[i];
	for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
	{
	  const auto& conn = mesh->conns[mesh->adj_matrix[j]];
	  if (conn.type == mesh::FRAC_FRAC || conn.type == mesh::FRAC_FRAC)
		conns_num++;
	}

	stencil_size = conns_num + 1;
	if (conns_num != 0)
	{
	  // matrix of coefficients in front of gradients in equations
	  auto& A_p = pre_grad_A_p[conns_num];
	  A_p.values = 0.0;
	  // matrix of coefficients in front of pressures and boundary conditions in equations
	  auto& R_p = pre_grad_R_p[conns_num];
	  R_p.values = 0.0;
	  // free term (gravity) in equations
	  auto& rhs_p = pre_grad_rhs_p[conns_num];
	  rhs_p.values = 0.0;

	  const auto& el1 = mesh->elems[i];
	  x1 = mesh->centroids[i];
	  el_id1 = el1.elem_id;
	  std::vector<index_t> temp_stencil(conns_num + 1);

	  no_neumann_conns = true;
	  index_t counter = 0;
	  for (int j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
	  {
		const auto& conn = mesh->conns[mesh->adj_matrix[j]];
		if (conn.elem_id1 != el_id1)
		  el_id2 = conn.elem_id1;
		else
		  el_id2 = conn.elem_id2;

		n = conn.n;
		if (dot((conn.c - mesh->centroids[i]), n) < 0) n = -n;

		// check the connection type (whether is a matrix matrix or boundary matrix)
		if (conn.type == mesh::FRAC_FRAC)
		{
		  // Location of connecting element centroid
		  x2 = mesh->centroids[el_id2];
		  // Projection from centroid of connecting element to interface surface
		  d2 = abs(dot(x2 - conn.c, n));
		  // Coefficient lambda for each connection
		  lambda2 = dot(n, matrix_vector_product(perms[el_id2], n));

		  if (lambda2 > EQUALITY_TOLERANCE)
		  {
			// pseudo: A[row] = x2 - x1 + d2 / lambda2 * (Perm[cell_1] - Perm[cell_2]) * n
			// equation 17 Terekhov's paper
			temp = (d2 / lambda2) * matrix_vector_product(perms[el_id1] - perms[el_id2], n);
			A_p(counter, 0) = x2.x - x1.x + temp.x;
			A_p(counter, 1) = x2.y - x1.y + temp.y;
			A_p(counter, 2) = x2.z - x1.z + temp.z;

			R_p(counter, R_p.N - 1) = -1.0;
			R_p(counter, counter) = 1.0;

			rhs_p(counter, 0) = dot(grav_vec, temp);
		  }
		  else
		  {
			temp = matrix_vector_product(perms[i], n);
			A_p(counter, 0) = temp.x;
			A_p(counter, 1) = temp.y;
			A_p(counter, 2) = temp.z;

			rhs_p(counter, 0) = dot(grav_vec, temp);
		  }
		  temp_stencil[counter++] = el_id2;
		}
		else if (conn.type == mesh::FRACTURE_BOUNDARY)
		{
		  temp_stencil[counter++] = el_id2;
		}

	  }
	  temp_stencil[counter] = el_id1;

	  // inversion
	  // SVD is produced for M x N matrix where M >= N, decompose transposed otherwise
	  counter = (counter > ND) ? ND : counter;
	  //assert(face_count_id == n_cur_faces);
	  auto& Wsvd = pre_Wsvd[counter];
	  auto& Zsvd = pre_Zsvd[counter];
	  auto& w_svd = pre_w_svd[counter].values;
	  std::fill_n(&Wsvd.values[0], Wsvd.values.size(), 0.0);
	  std::fill_n(&Zsvd.values[0], Zsvd.values.size(), 0.0);
	  std::fill_n(&w_svd[0], w_svd.size(), 0.0);
	  try {
		Matrix tempGrad, rhsGrad;
		if (A_p.M >= A_p.N)
		{
		  // SVD decomposition A = M W Z*
		  if (Zsvd.M != A_p.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
		  res = A_p.svd(Zsvd, w_svd);
		  assert(Zsvd.M == counter && Zsvd.N == counter);
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
		  tempGrad = Zsvd * Wsvd * A_p.transpose() * R_p;
		  rhsGrad = Zsvd * Wsvd * A_p.transpose() * rhs_p;
		}
		else
		{
		  // SVD decomposition A* = M W Z*
		  A_p.transposeInplace();
		  if (Zsvd.M != A_p.N) { printf("Wrong matrix dimension!\n"); exit(-1); }
		  res = A_p.svd(Zsvd, w_svd);
		  assert(Zsvd.M == counter && Zsvd.N == counter);
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
		  tempGrad = A_p * Wsvd * Zsvd.transpose() * R_p;
		  rhsGrad = A_p * Wsvd * Zsvd.transpose() * rhs_p;
		  A_p.transposeInplace();
		}

		auto& cur_grad = p_grads[i];
		cur_grad.a = tempGrad;
		cur_grad.rhs = rhsGrad;
		cur_grad.stencil = temp_stencil;
		cur_grad.sort();
	  }
	  catch (const std::exception&)
	  {
		throw "Matrix is not invertible";
	  }

#ifdef DEBUG_TRANS
	  // check sum of coefficients
	  if (no_neumann_conns)
	  {
		// recover gradient matrix from array
		Matrix cur(ND, conns_num + 1);

		int grad_value_idx = ND * grad_offset.back();
		for (int row = 0; row < cur.M; row++)
		{
		  for (int col = 0; col < cur.N; col++)
		  {
			cur(row, col) = grad_vals[grad_value_idx++];
		  }
		}

		for (uint8_t c = 0; c < ND; c++)
		{
		  lambda2 = fabs(std::accumulate(&cur.values[c * cur.N], &cur.values[c * cur.N] + cur.N, 0.0));
		  //assert(lambda2 < EQUALITY_TOLERANCE);
		}
	  }
#endif /* DEBUG_TRANS */
	}
  }

  t2 = steady_clock::now();
  cout << "Reconstruction of gradients:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

vector<index_t> Discretizer::find_connections_to_reconstruct_gradient(const index_t cell_id, const index_t cur_conn_id)
{
	vector<index_t> conns, buf;
	vector<pair<array<index_t, ND>, value_t>> triplets;
	Vector3 n;
	value_t proj1, proj2, det;
	Matrix basis(ND, ND);

	// reference co-normal
	const auto& cur_conn = mesh->conns[mesh->adj_matrix[cur_conn_id]];
	n = cur_conn.n;
	if (dot((cur_conn.c - mesh->centroids[cell_id]), n) < 0) n = -n;
	const Vector3 d = matrix_vector_product(perms[cell_id], n);
	basis(0, 0) = (cur_conn.c - mesh->centroids[cell_id]).x;
	basis(0, 1) = (cur_conn.c - mesh->centroids[cell_id]).y;
	basis(0, 2) = (cur_conn.c - mesh->centroids[cell_id]).z;

	auto det3 = [](Matrix& a)
	{
		return a(0, 0) * a(1, 1) * a(2, 2) + a(0, 2) * a(1, 0) * a(2, 1) + a(0, 1) * a(1, 2) * a(2, 0) -
			a(0, 2) * a(1, 1) * a(2, 0) - a(0, 0) * a(1, 2) * a(2, 1) - a(0, 1) * a(1, 0) * a(2, 2);
	};

	// produce all triplets and calculate their values of objective function
	std::function<void(const vector<index_t>&, index_t, index_t, index_t, vector<index_t>&)> subset;
	subset = [&](const vector<index_t>& arr, index_t size, index_t left, index_t index, vector<index_t>& l) {
		if (left == 0) 
		{
			const auto& conn1 = mesh->conns[mesh->adj_matrix[l[0]]];
			const auto& conn2 = mesh->conns[mesh->adj_matrix[l[1]]];
			// projection
			proj1 = dot(conn1.c - mesh->centroids[cell_id], d);
			proj2 = dot(conn2.c - mesh->centroids[cell_id], d);
			// complanarity
			basis(1, 0) = (conn1.c - mesh->centroids[cell_id]).x;
			basis(1, 1) = (conn1.c - mesh->centroids[cell_id]).y;
			basis(1, 2) = (conn1.c - mesh->centroids[cell_id]).z;
			basis(2, 0) = (conn2.c - mesh->centroids[cell_id]).x;
			basis(2, 1) = (conn2.c - mesh->centroids[cell_id]).y;
			basis(2, 2) = (conn2.c - mesh->centroids[cell_id]).z;
			det = det3(basis);
			det = det == det ? det : 0.0;
			triplets.push_back({ {cur_conn_id, l[0], l[1]}, proj1 + proj2 - 1 / (fabs(det) + EQUALITY_TOLERANCE) });
			return;
		}
		for (int i = index; i < size; i++) 
		{
			l.push_back(arr[i]);
			subset(arr, size, left - 1, i + 1, l);
			l.pop_back();
		}
	};

	for (index_t conn_id = mesh->adj_matrix_offset[cell_id]; conn_id < mesh->adj_matrix_offset[cell_id + 1]; conn_id++)
		if (conn_id != cur_conn_id)
			conns.push_back(conn_id);

	subset(conns, (index_t)conns.size(), 2, 0, buf);


	std::sort(triplets.begin(), triplets.end(),
		[](const pair<array<index_t, ND>, value_t>& a, const pair<array<index_t, ND>, value_t>& b) -> bool
		{
			return a.second > b.second;
		});

	return { cur_conn_id, triplets[0].first[1], triplets[0].first[2]};
}

/*void Discretizer::reconstruct_pressure_gradients_per_face(const BoundaryCondition& bc)
{
	USE_CONNECTION_BASED_GRADIENTS = true;

	// allocate memory for arrays
	grad_stencil.reserve(2 * mesh->conns.size() * MAX_STENCIL);
	grad_offset.reserve(2 * mesh->conns.size() + 1);
	p_grad_vals.reserve(2 * mesh->conns.size() * MAX_STENCIL);
	p_grad_rhs.reserve(2 * mesh->conns.size());

	steady_clock::time_point t1, t2;
	t1 = steady_clock::now();

	bc_flow = bc;

	// viscosity should be provided from outside
	const value_t mu = 1.0;
	index_t el_id1, el_id2, row_id;
	uint8_t conns_num, stencil_size;
	value_t d2, lambda2;
	Vector3 n, x1, x2, temp;
	Matrix to_invert(ND, ND);
	bool no_neumann_conns;

	// loop through the adjacency matrix (enough to iterate up to FRACTUREs)
	for (index_t i = 0; i < mesh->region_ranges.at(mesh::FRACTURE).second; i++)
	{
		for (index_t conn_id = mesh->adj_matrix_offset[i]; conn_id < mesh->adj_matrix_offset[i + 1]; conn_id++)
		{
			auto conns_for_reconstruction = find_connections_to_reconstruct_gradient(i, conn_id);

			// Resize the dims of A and R that depend on the amount of connections
			conns_num = (uint8_t)conns_for_reconstruction.size();
			assert(conns_num >= ND);
			stencil_size = conns_num + 1;
			// matrix of coefficients in front of gradients in equations
			auto& A = pre_grad_A_p[conns_num];
			A.values = 0.0;
			// matrix of coefficients in front of pressures and boundary conditions in equations
			auto& R = pre_grad_R_p[conns_num];
			R.values = 0.0;
			// free term (gravity) in equations
			auto& rhs = pre_grad_rhs_p[conns_num];
			rhs.values = 0.0;

			x1 = mesh->centroids[i];
			el_id1 = mesh->elems[i].elem_id;
			std::vector<index_t> temp_stencil(conns_num + 1);

			no_neumann_conns = true;
			index_t counter = 0;
			for (auto& j: conns_for_reconstruction)
			{
				const auto& conn = mesh->conns[mesh->adj_matrix[j]];
				if (conn.elem_id1 != el_id1)
					el_id2 = conn.elem_id1;
				else
					el_id2 = conn.elem_id2;

				// the row index of the current cells in A and R matrices
				row_id = counter;

				n = conn.n;
				if (dot((conn.c - mesh->centroids[i]), n) < 0) n = -n;

				// check the connection type (whether is a matrix matrix or boundary matrix)
				switch (conn.type)
				{
				case mesh::MAT_MAT:
					// Location of connecting element centroid
					x2 = mesh->centroids[el_id2];
					// Projection from centroid of connecting element to interface surface
					d2 = abs(dot(x2 - conn.c, n));
					// Coefficient lambda for each connection
					lambda2 = dot(n, matrix_vector_product(perms[el_id2], n));

					if (lambda2 > EQUALITY_TOLERANCE)
					{
						// pseudo: A[row] = x2 - x1 + d2 / lambda2 * (Perm[cell_1] - Perm[cell_2]) * n
						// equation 17 Terekhov's paper
						temp = (d2 / lambda2) * matrix_vector_product(perms[el_id1] - perms[el_id2], n);
						A(row_id, 0) = x2.x - x1.x + temp.x;
						A(row_id, 1) = x2.y - x1.y + temp.y;
						A(row_id, 2) = x2.z - x1.z + temp.z;

						R(row_id, R.N - 1) = -1.0;
						R(row_id, row_id) = 1.0;

						rhs(row_id, 0) = dot(grav_vec, temp);
					}
					else
					{
						temp = matrix_vector_product(perms[i], n);
						A(row_id, 0) = temp.x;
						A(row_id, 1) = temp.y;
						A(row_id, 2) = temp.z;

						rhs(row_id, 0) = dot(grav_vec, temp);
					}

					break;
				case mesh::MAT_BOUND:
					//TODO Implement other BC

					// Coefficients that define boundary condition
					const auto& alpha = bc.a_p[conn.elem_id2 - mesh->n_cells];
					const auto& beta = bc.b_p[conn.elem_id2 - mesh->n_cells];

					// for debuging purposes
					if (alpha != 1.0 || beta != 0.0) no_neumann_conns = false;

					temp = beta / mu * matrix_vector_product(perms[i], n);
					A(row_id, 0) = alpha * (conn.c.x - x1.x) + temp.x;
					A(row_id, 1) = alpha * (conn.c.y - x1.y) + temp.y;
					A(row_id, 2) = alpha * (conn.c.z - x1.z) + temp.z;

					// update the row on matrix R
					// the p1 term has to be -alpha, all other elements have to be 0 except for the last
					// element which is going to be rp
					R(row_id, R.N - 1) = -alpha;
					R(row_id, row_id) = 1.0;

					rhs(row_id, 0) = dot(grav_vec, temp);

					break;
				}
				temp_stencil[counter] = el_id2;
				counter++;
			}
			temp_stencil[counter] = el_id1;

			// computing the least squares solution for nabla p = (A^T * A) ^(-1) * (A^T) * R
			to_invert = A;// A.transpose()* A;
			try {
				to_invert.inv();
				Matrix tempGrad = to_invert * R;// *A.transpose()* R;
				Matrix rhsGrad = to_invert * rhs;// *A.transpose()* rhs;

				std::vector<std::pair<index_t, index_t>> sort_vec(tempGrad.N);
				for (index_t row = 0; row < tempGrad.N; row++)
				{
					sort_vec[row] = std::make_pair(temp_stencil[row], row);
				}
				std::sort(sort_vec.begin(), sort_vec.end(), [](auto& left, auto& right) { return left.first < right.first; });

				grad_offset.push_back(static_cast<index_t>(grad_stencil.size()));

				// push sorted stencil
				for (const auto& st : sort_vec)
					grad_stencil.push_back(st.first);

				// push sorted coefficients & rhs
				for (int row = 0; row < tempGrad.M; row++)
				{
					p_grad_rhs.push_back(rhsGrad(row, 0));
					for (int col = 0; col < tempGrad.N; col++)
					{
						p_grad_vals.push_back(tempGrad(row, sort_vec[col].second));
					}
				}
			}
			catch (const std::exception&)
			{
				throw "Matrix not invertible";
			}

#ifdef DEBUG_TRANS
			// check sum of coefficients
			if (no_neumann_conns)
			{
				// recover gradient matrix from array
				Matrix cur(ND, conns_num + 1);

				int grad_value_idx = ND * grad_offset.back();
				for (int row = 0; row < cur.M; row++)
				{
					for (int col = 0; col < cur.N; col++)
					{
						cur(row, col) = grad_vals[grad_value_idx++];
					}
				}

				for (uint8_t c = 0; c < ND; c++)
				{
					lambda2 = fabs(std::accumulate(&cur.values[c * cur.N], &cur.values[c * cur.N] + cur.N, 0.0));
					tmp = std::inner_product(&cur.values[c * cur.N], &cur.values[c * cur.N] + cur.N, &cur.values[c * cur.N], 0.0);
					assert(lambda2 < EQUALITY_TOLERANCE * tmp);
				}
			}
	#endif /* DEBUG_TRANS 
		}
	}

	grad_offset.push_back(static_cast<index_t>(grad_stencil.size()));

	t2 = steady_clock::now();
	cout << "Reconstruction of gradients:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}*/

void Discretizer::calc_mpfa_transmissibilities(const bool with_thermal) 
{
	steady_clock::time_point t1, t2;
	t1 = steady_clock::now();
	value_t sign;
	index_t cell_id1, cell_id2, adj_nebr_id;

	cell_m.clear();	cell_p.clear();
	cell_m.reserve(mesh->adj_matrix.size());
	cell_p.reserve(mesh->adj_matrix.size());
	flux_vals.reserve(mesh->adj_matrix.size() * MAX_STENCIL);
	flux_vals_homo.reserve(mesh->adj_matrix.size() * MAX_STENCIL);
	flux_rhs.reserve(mesh->adj_matrix.size());
	flux_stencil.reserve(mesh->adj_matrix.size() * MAX_STENCIL);
	flux_offset.reserve(mesh->adj_matrix.size() + 1);
	if (with_thermal)
	  flux_vals_thermal.reserve(mesh->adj_matrix.size() * MAX_STENCIL);

	flux_offset.push_back(0);
	// loop through matrix elements
	for (index_t i = 0; i < mesh->region_ranges.at(mesh::MATRIX).second; i++)
	{
		cell_id1 = i;

		// loop through connections of particular element
		for (index_t j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
		{
			const auto& conn = mesh->conns[mesh->adj_matrix[j]];
			cell_id2 = mesh->adj_matrix_cols[j];
			sign = (conn.elem_id1 == cell_id1) ? 1.0 : -1.0;

			if (conn.type == mesh::MAT_MAT)
			{
				auto& flux = fluxes[0];
				for (index_t k = mesh->adj_matrix_offset[cell_id2]; k < mesh->adj_matrix_offset[cell_id2 + 1]; k++) { if (mesh->adj_matrix_cols[k] == cell_id1) { adj_nebr_id = k; break; } }
				calc_matrix_matrix(conn, flux, with_thermal);
				
				flux.darcy.a.values *= sign * conn.area;
				flux.fick.a.values *= sign * conn.area;
				flux.fourier.a.values *= sign * conn.area;
				flux.darcy.rhs.values *= sign * conn.area;

				cell_m.push_back(cell_id1);
				cell_p.push_back(cell_id2);

				if (with_thermal)
				  write_trans_thermal(flux);
				else
				  write_trans(flux);

#ifdef DEBUG_TRANS
				sum = 0.0;
				for (uint8_t k = 0; k < flux.stencil.size(); k++)
				{
					if (flux.stencil[k] < mesh->region_ranges.at(mesh::FRACTURE).second)
						sum += flux.a.values[k];
				}

				//sum = std::accumulate(std::begin(flux.a.values), std::end(flux.a.values), 0.0);
				/*if (fabs(sum) > 100.0 * EQUALITY_TOLERANCE)
				{
					std::cout << "Sum of all transmissibilities: " << sum << std::endl;
					exit(-1);
				}*/
#endif /* DEBUG_TRANS */
			}
			else if (conn.type == mesh::MAT_BOUND)
			{
				auto& flux = fluxes[0];
				calc_matrix_boundary(conn, flux, with_thermal);

				flux.darcy.a.values *= conn.area;
				flux.fourier.a.values *= conn.area;
				flux.fick.a.values *= conn.area;
				flux.darcy.rhs.values *= conn.area;

				cell_m.push_back(cell_id1);
				cell_p.push_back(cell_id2);

				if (with_thermal)
				  write_trans_thermal(flux);
				else
				  write_trans(flux);
			}
			else if (conn.type == mesh::MAT_FRAC || conn.type == mesh::FRAC_MAT)
			{
				auto& flux = fluxes[0];
				for (index_t k = mesh->adj_matrix_offset[cell_id2]; k < mesh->adj_matrix_offset[cell_id2 + 1]; k++) { if (mesh->adj_matrix_cols[k] == cell_id1) { adj_nebr_id = k; break; } }
				calc_matrix_matrix(conn, flux);

				flux.darcy.a.values *= sign * conn.area;
				flux.darcy.rhs.values *= sign * conn.area;

				cell_m.push_back(cell_id1);
				cell_p.push_back(cell_id2);
				write_trans(flux);
			}
		}
	}
	// loop through fracture elements
	for (index_t i = mesh->region_ranges.at(mesh::FRACTURE).first; i < mesh->region_ranges.at(mesh::FRACTURE).second; i++)
	{
		cell_id1 = i;

		// loop through connections of particular element
		for (index_t j = mesh->adj_matrix_offset[i]; j < mesh->adj_matrix_offset[i + 1]; j++)
		{
			const auto& conn = mesh->conns[mesh->adj_matrix[j]];
			cell_id2 = mesh->adj_matrix_cols[j];
			sign = (conn.elem_id1 == cell_id1) ? 1.0 : -1.0;

			if (conn.type == mesh::MAT_FRAC || conn.type == mesh::FRAC_MAT)
			{
				auto& flux = fluxes[0];
				for (index_t k = mesh->adj_matrix_offset[cell_id2]; k < mesh->adj_matrix_offset[cell_id2 + 1]; k++) { if (mesh->adj_matrix_cols[k] == cell_id1) { adj_nebr_id = k; break; } }
				calc_matrix_matrix(conn, flux);

				flux.darcy.a.values *= sign * conn.area;
				flux.darcy.rhs.values *= sign * conn.area;

				cell_m.push_back(cell_id1);
				cell_p.push_back(cell_id2);
				write_trans(flux);

#ifdef DEBUG_TRANS
				sum = 0.0;
				for (uint8_t k = 0; k < flux.stencil.size(); k++)
				{
					if (flux.stencil[k] < mesh->region_ranges.at(mesh::FRACTURE).second)
						sum += flux.a.values[k];
				}

				//sum = std::accumulate(std::begin(flux.a.values), std::end(flux.a.values), 0.0);
				/*if (fabs(sum) > 100.0 * EQUALITY_TOLERANCE)
				{
					std::cout << "Sum of all transmissibilities: " << sum << std::endl;
					exit(-1);
				}*/
#endif /* DEBUG_TRANS */
			}
			else if (conn.type == mesh::FRAC_FRAC)
			{
				auto& flux = fluxes[0];
				for (index_t k = mesh->adj_matrix_offset[cell_id2]; k < mesh->adj_matrix_offset[cell_id2 + 1]; k++) { if (mesh->adj_matrix_cols[k] == cell_id1) { adj_nebr_id = k; break; } }
				calc_fault_fault(conn, flux);

				flux.darcy.a.values *= sign * conn.area;
				flux.darcy.rhs.values *= sign * conn.area;

				cell_m.push_back(cell_id1);
				cell_p.push_back(cell_id2);
				write_trans(flux);
			}
		}
	}
	
	t2 = steady_clock::now();
	cout << "Find MPFA trans: \t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

void Discretizer::calc_matrix_matrix(const mesh::Connection& conn, FlowHeatApproximation& flux, const bool with_thermal)
{
	uint8_t id1, id2;
	value_t lam1, lam2, d1, d2, Fh, T1, T2, T;
	Matrix gam1(ND, 1), gam2(ND, 1), y1(ND, 1), y2(ND, 1), grad_coef, n(ND, 1), K1n(ND, 1), K2n(ND, 1);
	const auto& x1 = mesh->centroids[conn.elem_id1];
	const auto& x2 = mesh->centroids[conn.elem_id2];
	Vector3 vec1, vec2;
	std::vector<index_t> th_stencil;
	
	// normal vector
	copy_n(std::begin(conn.n.values), ND, std::begin(n.values));
	if (dot(conn.c - x1, conn.n) < 0.0) n.values *= -1.0;

	// co-normal decomposition
	K1n = DARCY_CONSTANT * perms[conn.elem_id1] * n;
	K2n = DARCY_CONSTANT * perms[conn.elem_id2] * n;
	lam1 = (n.transpose() * K1n).values[0];
	lam2 = (n.transpose() * K2n).values[0];
	gam1 = K1n - lam1 * n;
	gam2 = K2n - lam2 * n;
	
	d1 = fabs(dot(conn.c - x1, n));
	d2 = fabs(dot(x2 - conn.c, n));
	y1.values = { x1.values[0], x1.values[1], x1.values[2] };	y1 += d1 * n;
	y2.values = { x2.values[0], x2.values[1], x2.values[2] };	y2 -= d2 * n;

	// allocate arrays for merging gradients
	const auto& g1 = p_grads[conn.elem_id1];
	const auto& g2 = p_grads[conn.elem_id2];

	flux.darcy = g1 / 2.0 + g2 / 2.0;
	
	// flux approximation
	grad_coef = -(lam1 * lam2 * (y1 - y2).transpose() + lam1 * d2 * gam2.transpose() + lam2 * d1 * gam1.transpose()) / (lam1 * d2 + lam2 * d1);
	flux.darcy.a = grad_coef * flux.darcy.a;
	flux.darcy.rhs = grad_coef * flux.darcy.rhs;
	flux.darcy.rhs += DARCY_CONSTANT * grav_vec * (lam2 * d1 * perms[conn.elem_id1] + lam1 * d2 * perms[conn.elem_id2]) * n / (lam1 * d2 + lam2 * d1);
	const auto it1 = std::find(flux.darcy.stencil.begin(), flux.darcy.stencil.end(), conn.elem_id1);
	const auto it2 = std::find(flux.darcy.stencil.begin(), flux.darcy.stencil.end(), conn.elem_id2);
	assert(it1 != flux.darcy.stencil.end() && it2 != flux.darcy.stencil.end());
	Fh = -lam1 * lam2 / (lam1 * d2 + lam2 * d1);
	id1 = static_cast<uint8_t>(std::distance(flux.darcy.stencil.begin(), it1));
	id2 = static_cast<uint8_t>(std::distance(flux.darcy.stencil.begin(), it2));
	flux.darcy.a(0, id1) -= Fh;
	flux.darcy.a(0, id2) += Fh;

	vec1 = conn.c - x1; T1 = dot(vec1, n) / dot(vec1, vec1);
	vec2 = conn.c - x2;	T2 = -dot(vec2, n) / dot(vec2, vec2);
	assert(T1 >= 0.0 && T2 >= 0.0);
	T = (T1 + T2) > EQUALITY_TOLERANCE ? T1 * T2 / (T1 + T2) : 0.0;
	flux.fick.a.values = 0.0;
	flux.fick.a(0, id1) = T;
	flux.fick.a(0, id2) = -T;

	if (with_thermal)
	{
	  // co-normal decomposition
	  K1n = heat_conductions[conn.elem_id1] * n;
	  K2n = heat_conductions[conn.elem_id2] * n;
	  lam1 = (n.transpose() * K1n).values[0];
	  lam2 = (n.transpose() * K2n).values[0];
	  gam1 = K1n - lam1 * n;
	  gam2 = K2n - lam2 * n;
	  // temperature gradients
	  const auto& g1 = t_grads[conn.elem_id1];
	  const auto& g2 = t_grads[conn.elem_id2];
	  flux.fourier = g1 / 2.0 + g2 / 2.0;
	  // flux approximation
	  grad_coef = -(lam1 * lam2 * (y1 - y2).transpose() + lam1 * d2 * gam2.transpose() + lam2 * d1 * gam1.transpose()) / (lam1 * d2 + lam2 * d1);
	  flux.fourier.a = grad_coef * flux.fourier.a;
	  Fh = -lam1 * lam2 / (lam1 * d2 + lam2 * d1);
	  flux.fourier.a(0, id1) -= Fh;
	  flux.fourier.a(0, id2) += Fh;
	}
}

void Discretizer::calc_fault_fault(const mesh::Connection& conn, FlowHeatApproximation& flux)
{
	uint8_t id1, id2;
	value_t lam1, lam2, d1, d2, Fh;
	Matrix n(ND, 1), K1n(ND, 1), K2n(ND, 1);
	const auto& x1 = mesh->centroids[conn.elem_id1];
	const auto& x2 = mesh->centroids[conn.elem_id2];

	// normal vector
	copy_n(std::begin(conn.n.values), ND, std::begin(n.values));
	if (dot(conn.c - x1, conn.n) < 0.0) n.values *= -1.0;

	// co-normal decomposition
	K1n = DARCY_CONSTANT * perms[conn.elem_id1] * n;
	K2n = DARCY_CONSTANT * perms[conn.elem_id2] * n;
	lam1 = (n.transpose() * K1n).values[0];
	lam2 = (n.transpose() * K2n).values[0];
	d1 = fabs(dot(conn.c - x1, n));
	d2 = fabs(dot(x2 - conn.c, n));

	flux.darcy.stencil.clear();
	flux.darcy.stencil.push_back(std::min(conn.elem_id1, conn.elem_id2));
	flux.darcy.stencil.push_back(std::max(conn.elem_id1, conn.elem_id2));

	// flux approximation
	//grad_coef = -(lam1 * lam2 * (y1 - y2).transpose() + lam1 * d2 * gam2.transpose() + lam2 * d1 * gam1.transpose()) / (lam1 * d2 + lam2 * d1);
	//flux.a = grad_coef * nabla_p;
	//flux.rhs = grad_coef * (g1.rhs + g2.rhs) / 2.0;
	//flux.rhs += DARCY_CONSTANT * grav_vec * (lam2 * d1 * perms[conn.elem_id1] + lam1 * d2 * perms[conn.elem_id2]) * n / (lam1 * d2 + lam2 * d1);
	const auto it1 = std::find(flux.darcy.stencil.begin(), flux.darcy.stencil.end(), conn.elem_id1);
	const auto it2 = std::find(flux.darcy.stencil.begin(), flux.darcy.stencil.end(), conn.elem_id2);
	assert(it1 != flux.darcy.stencil.end() && it2 != flux.darcy.stencil.end());
	
	Fh = -lam1 * lam2 / (lam1 * d2 + lam2 * d1);
	id1 = static_cast<uint8_t>(std::distance(flux.darcy.stencil.begin(), it1));
	id2 = static_cast<uint8_t>(std::distance(flux.darcy.stencil.begin(), it2));
	flux.darcy.a(0, id1) -= Fh;
	flux.darcy.a(0, id2) += Fh;
}

void Discretizer::calc_matrix_boundary(const mesh::Connection& conn, FlowHeatApproximation& flux, const bool with_thermal)
{
	uint8_t id1, id2;
	value_t lam1, d1, T1, T;
	Matrix K1n(ND, 1), gam1(ND, 1), y1(ND, 1), grad_coef, n(ND, 1), c2(ND, 1);
	const auto& x1 = mesh->centroids[conn.elem_id1];
	const auto& x2 = mesh->centroids[conn.elem_id2];
	const value_t mu = 1.0;
	Vector3 vec1;

	// normal vector
	copy_n(std::begin(conn.n.values), ND, std::begin(n.values));
	if (dot(conn.c - x1, conn.n) < 0.0) n.values *= -1.0;

	// boundary conditions: a*p + b*f = r 
	const auto& a = bc_flow.a[conn.elem_id2 - mesh->n_cells];
	const auto& b = bc_flow.b[conn.elem_id2 - mesh->n_cells];

	// co-normal decomposition
	K1n.values = DARCY_CONSTANT * (perms[conn.elem_id1] * n).values;
	lam1 = (n.transpose() * K1n).values[0];
	gam1 = K1n - lam1 * n;

	d1 = fabs(dot(conn.c - x1, n));
	y1.values = { x1.values[0], x1.values[1], x1.values[2] };	y1 += d1 * n;
	c2.values = { x2.values[0], x2.values[1], x2.values[2] };

	const auto& g1 = p_grads[conn.elem_id1];

	// flux approximation
	value_t mult = 1.0 / (a + b * lam1 / mu / d1);
	grad_coef = -mult / mu * a * (lam1 / d1 * (y1 - c2).transpose() + gam1.transpose());
	flux.darcy.a = grad_coef * g1.a;
	flux.darcy.rhs = grad_coef * g1.rhs;
	flux.darcy.rhs += mult / mu * a * grav_vec * K1n;
	flux.darcy.stencil = g1.stencil;

	const auto it1 = std::find(flux.darcy.stencil.begin(), flux.darcy.stencil.end(), conn.elem_id1);
	const auto it2 = std::find(flux.darcy.stencil.begin(), flux.darcy.stencil.end(), conn.elem_id2);
	assert(it1 != flux.darcy.stencil.end() && it2 != flux.darcy.stencil.end());
	id1 = static_cast<uint8_t>(std::distance(flux.darcy.stencil.begin(), it1));
	id2 = static_cast<uint8_t>(std::distance(flux.darcy.stencil.begin(), it2));
	flux.darcy.a(0, id1) += lam1 / d1 / mu * mult * a;
	flux.darcy.a(0, id2) += -lam1 / d1 / mu * mult;

	vec1 = conn.c - x1; 
	T1 = dot(vec1, n) / dot(vec1, vec1);
	T = T1 / (a + b * T1);
	assert(T >= 0.0);
	flux.fick.a.values = 0.0;
	flux.fick.a(0, id1) = T;
	flux.fick.a(0, id2) = -T;

	if (with_thermal)
	{
	  Matrix C1n(ND, 1);
	  // boundary conditions: a*p + b*f = r 
	  const auto& a = bc_heat.a[conn.elem_id2 - mesh->n_cells];
	  const auto& b = bc_heat.b[conn.elem_id2 - mesh->n_cells];
	  
	  // co-normal decomposition
	  C1n.values = (heat_conductions[conn.elem_id1] * n).values;
	  lam1 = (n.transpose() * C1n).values[0];
	  gam1 = C1n - lam1 * n;

	  const auto& g1 = t_grads[conn.elem_id1]; 

	  // flux approximation
	  mult = 1.0 / (a + b * lam1 / d1);
	  grad_coef = -mult * a * (lam1 / d1 * (y1 - c2).transpose() + gam1.transpose());
	  flux.fourier.a = grad_coef * g1.a;
	  flux.fourier.stencil = g1.stencil;

	  flux.fourier.a(0, id1) += lam1 / d1 * mult * a;
	  flux.fourier.a(0, id2) += -lam1 / d1 * mult;
	}
}

// fill the permeability tensor as diagonal with permx, permy, permz on diagonal
void Discretizer::calcPermeabilitySimple(const double permx, const double permy, const double permz)
{
	perms.resize(mesh->elems.size(), Matrix33());

	for (int i = 0; i < mesh->n_cells; i++)
	{
		auto& perm = perms[i];
		perm(0, 0) = permx;
		perm(1, 1) = permy;
		perm(2, 2) = permz;
	}
}

// fill the permeability tensor as diagonal with arrays of permx, permy, permz on diagonal
// assuming the order in input arrays (permx) is as in input GRDECL file (ZYX-loop order)
void Discretizer::set_permeability(std::vector<value_t> &permx, std::vector<value_t> &permy, std::vector<value_t> &permz)
{
	perms.resize(mesh->n_cells, Matrix33());

	if (permx.size() == 0 || permy.size() == 0 || permz.size() == 0) {
		cout << "Error in set_permeability: " << permx.size() << permy.size() << permz.size() << "\n";
		return;
	}

	for (index_t i = 0, counter = 0; i < mesh->get_n_cells_total(); i++)
	{
		if (mesh->global_to_local[i] >= 0) // cell is active
		{
			auto& perm = perms[counter++];
			perm(0, 0) = permx[i];
			perm(1, 1) = permy[i];
			perm(2, 2) = permz[i];
		}
		else
		{
			continue;
		}

		//DEBUG
		//int i1, j1, k1;
		//mesh->get_ijk_from_global_idx(i, i1, j1, k1);
		//std::array<value_t, 3> d = mesh->calc_cell_sizes(i1, j1, k1);
		//cout << "cell (" << i1 << ", " << j1 << ", " << k1 << ") dx=" << d[0] << " dy=" << d[1] <<  " dz=" << d[2] << "\n";

	}//loop by cells
}

// assuming the order in input arrays (poro) is as in input GRDECL file (ZYX-loop order)
void Discretizer::set_porosity(std::vector<value_t> &new_poro)
{
	poro.resize(mesh->n_cells);

	index_t counter = 0;
	for (index_t i = 0; i < mesh->get_n_cells_total(); i++)
	{
		if (mesh->global_to_local[i] >= 0) // cell is active
		{
			poro[counter++] = new_poro[i];
		}
	}//loop by cells
}

vector<index_t> Discretizer::get_one_way_tpfa_transmissibilities() const
{
	assert(cell_m.size());

	unordered_set<pair<index_t, index_t>, mesh::pair_cantor_hash, mesh::one_way_connection_comparator> conn_set;
	unordered_set<pair<index_t, index_t>>::const_iterator it;
	pair<index_t, index_t> ids;
	vector<index_t> res;
	conn_set.reserve(mesh->conns.size());
	res.reserve(mesh->conns.size());

	for (index_t i = 0; i < cell_m.size(); i++)
	{
		ids.first = cell_m[i];
		ids.second = cell_p[i];
		
		it = conn_set.find(ids);
		if (it == conn_set.end())
		{
			res.push_back(i);
			conn_set.insert(ids);
		}
	}

	return res;
}

// 
void Discretizer::write_tran_cube(std::string fname, std::string fname_nnc) const
{
	index_t nnc_counter = 0;
	index_t n_cells_all = mesh->get_n_cells_total();
	std::vector<value_t> tranx;
	std::vector<value_t> trany;
	std::vector<value_t> tranz;
	tranx.resize(n_cells_all);
	trany.resize(n_cells_all);
	tranz.resize(n_cells_all);
	std::fill(tranx.begin(), tranx.end(), 0);
	std::fill(trany.begin(), trany.end(), 0);
	std::fill(tranz.begin(), tranz.end(), 0);

	unordered_set<pair<index_t, index_t>, mesh::pair_xor_hash, mesh::one_way_connection_comparator> conn_set;
	unordered_set<pair<index_t, index_t>>::const_iterator it;
	pair<index_t, index_t> ids;
	conn_set.reserve(mesh->conns.size());

	std::ofstream f;
	f.open(fname_nnc);
	f << "NNC\n M\tP\tTRANS\tM_IJK\tP_IJK\n";

	for (int i = 0; i < cell_m.size(); i++) {
		// filter duplicates (i<->j, j<->i)
		ids.first  = cell_m[i];
		ids.second = cell_p[i];
		it = conn_set.find(ids);
		if (it == conn_set.end()){
				conn_set.insert(ids);
			}
		else
			continue;

		double t = fabs(flux_vals[i*2]);
	  int m = cell_m[i];
		int p = cell_p[i];

		int i1, j1, k1;
		mesh->get_ijk(m, i1, j1, k1, false);

		int i2, j2, k2;
		mesh->get_ijk(p, i2, j2, k2, false);

		index_t idx = 0;
		if (i1 != i2 && j1 == j2 && k1 == k2) {
			idx = mesh->get_global_index(std::min(i1, i2), j1, k1);
			tranx[idx] = t;
		}
		else if (i1 == i2 && j1 != j2 && k1 == k2) {
			idx = mesh->get_global_index(i1, std::min(j1, j2), k1);
			trany[idx] = t;
		}
		else if (i1 == i2 && j1 == j2 && k1 != k2) {
			idx = mesh->get_global_index(i1, j1, std::min(k1, k2));
			tranz[idx] = t;
		}
		else {
			f << mesh->local_to_global[m] + 1 << "\t" << mesh->local_to_global[p] + 1<< "\t" << t << "\t";
			f << mesh->get_ijk_as_str(m, false) << "\t" << mesh->get_ijk_as_str(p, false) << "\n";

			nnc_counter++;
		}
	}
	f.close();

	std::vector<int> dummy_actnum;
	mesh->write_array_to_file(fname, "ACTNUM", mesh->actnum, dummy_actnum, n_cells_all, 1.0, false);
	// tran in metric units
	mesh->write_array_to_file(fname, "TRANX", tranx, dummy_actnum, n_cells_all, 1.0, true);
	mesh->write_array_to_file(fname, "TRANY", trany, dummy_actnum, n_cells_all, 1.0, true);
	mesh->write_array_to_file(fname, "TRANZ", tranz, dummy_actnum, n_cells_all, 1.0, true);

	cout << "number of NNC = " << nnc_counter << "\n";
}

// get fault location (only NNC layers)
std::vector<value_t> Discretizer::get_fault_xyz() const
{
	index_t nnc_counter = 0;
	index_t n_cells_all = mesh->get_n_cells_total();

	std::vector<index_t> nnc_cells;
	std::vector<value_t> fault_xyz;

	unordered_set<pair<index_t, index_t>, mesh::pair_xor_hash, mesh::one_way_connection_comparator> conn_set;
	unordered_set<pair<index_t, index_t>>::const_iterator it;
	pair<index_t, index_t> ids;
	conn_set.reserve(mesh->conns.size());

	for (int i = 0; i < cell_m.size(); i++) {
		// filter duplicates (i<->j, j<->i)
		ids.first = cell_m[i];
		ids.second = cell_p[i];
		it = conn_set.find(ids);
		if (it == conn_set.end()) {
			conn_set.insert(ids);
		}
		else
			continue;

		int m = cell_m[i];
		int p = cell_p[i];

		int i1, j1, k1;
		mesh->get_ijk(m, i1, j1, k1, false);

		int i2, j2, k2;
		mesh->get_ijk(p, i2, j2, k2, false);

		if (i1 != i2 && j1 == j2 && k1 == k2) {
		}
		else if (i1 == i2 && j1 != j2 && k1 == k2) {
		}
		else if (i1 == i2 && j1 == j2 && k1 != k2) {
		}
		else {
			//f << mesh->local_to_global[m] + 1 << "\t" << mesh->local_to_global[p] + 1 << "\t" << t << "\t";
			//f << mesh->get_ijk_as_str(m, false) << "\t" << mesh->get_ijk_as_str(p, false) << "\n";
			nnc_cells.push_back(m);
			nnc_cells.push_back(p);

			fault_xyz.push_back(mesh->centroids[m].y);
			fault_xyz.push_back(mesh->centroids[m].x);
			fault_xyz.push_back(mesh->centroids[m].z);

			fault_xyz.push_back(mesh->centroids[p].y);
			fault_xyz.push_back(mesh->centroids[p].x);
			fault_xyz.push_back(mesh->centroids[p].z);
			nnc_counter++;
		}
	}

	cout << "number of NNC = " << nnc_counter << "\n";
	return fault_xyz;
}


void Discretizer::write_tran_list(std::string fname) const
{
	std::ofstream f;
	f.open(fname);

	for (int i = 0; i < cell_m.size(); i++) {
		double t = fabs(flux_vals[i * 2]);
		int m = cell_m[i];
		int p = cell_p[i];

		f << m << "\t" << p << "\t";
		f << mesh->centroids[m].x << "\t" << mesh->centroids[m].y << "\t" << mesh->centroids[m].z << "\t";
		f << mesh->centroids[p].x << "\t" << mesh->centroids[p].y << "\t" << mesh->centroids[p].z << "\t";
		f << t << "\n";
	}
	
	f.close();
}
