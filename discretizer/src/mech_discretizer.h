#ifndef MECH_DISCRETIZER_H_
#define MECH_DISCRETIZER_H_

#include "discretizer.h"

namespace dis
{
	/* Boundary condition */
	class MechBoundaryCondition
	{
	public:
	  std::vector<value_t> a_n, b_n;
	  std::vector<value_t> a_t, b_t;
	  MechBoundaryCondition() {};
	  ~MechBoundaryCondition() {};
	};
	/* 6x6 stiffness matrix */
	class Stiffness : public Matrix
	{
	public:
	  static const index_t N = 6;
	  typedef Matrix Base;

	  Stiffness() : Base(6, 6) {};
	  Stiffness(value_t la, value_t mu) : Base(6, 6)
	  {
		(*this)(0, 0) = (*this)(1, 1) = (*this)(2, 2) = la + 2 * mu;
		(*this)(3, 3) = (*this)(4, 4) = (*this)(5, 5) = mu;
		(*this)(0, 1) = (*this)(0, 2) = (*this)(1, 2) = la;
		(*this)(1, 0) = (*this)(2, 0) = (*this)(2, 1) = la;
	  };
	  Stiffness(std::valarray<value_t> _c) : Base(_c, 6, 6) {}
	};

	/* Here is what we call linear approximation */
	struct MechApproximation 
	{
	  MechApproximation() {};
	  MechApproximation(uint8_t M, uint8_t N)
	  {
		a = Matrix(M, N);
		rhs = Matrix(M, 1);
		a_biot = Matrix(M, N);
		rhs_biot = Matrix(M, 1);
		stencil.reserve(N);
	  };
	  Matrix a, rhs, a_biot, rhs_biot;
	  std::vector<index_t> stencil;
	};

	enum MechDiscretizerMode { POROELASTIC, THERMOPOROELASTIC };

	const std::unordered_map<MechDiscretizerMode, uint8_t> N_UNKNOWNS = { { POROELASTIC, ND + 1 }, { THERMOPOROELASTIC, ND + 2 } };

	/* Discretiser */
	template <MechDiscretizerMode MODE>
	class MechDiscretizer : public Discretizer
	{
	protected:
	  struct InnerMatrices
	  {
		Matrix T1, T2, G1, G2, Q1, Q2, Th1, Th2, R1, R2, y1, y2, S1, S2;
		value_t r1, r2, beta_stab1, beta_stab2, k_stab1, k_stab2, c_stab1, c_stab2;
	  };
	  
	  std::unordered_map<index_t, Matrix> pre_grad_A_u, pre_grad_R_u, pre_grad_rhs_u;
	  std::map<index_t, std::map<index_t, Matrix>> pre_cur_rhs;
	  std::vector<MechApproximation> mech_fluxes;
	  Matrix W;
	  std::vector<std::map<index_t, InnerMatrices>> inner;

	  static const uint8_t n_unknowns;

	  std::vector<index_t>::const_iterator it_find;
	  std::pair<bool, size_t> res1, res2;
	  inline std::pair<bool, size_t> findInVector(const std::vector<index_t>& vec, const index_t& element)
	  {
		// Find given element in vector
		it_find = std::find(vec.begin(), vec.end(), element);
		if (it_find != vec.end())
		{
		  return { true, std::distance(vec.begin(), it_find) };
		}
		else
		{
		  return { false, -1 };
		}
	  };

	  void calc_matrix_matrix_mech(const mesh::Connection& conn, MechApproximation& flux, index_t conn_id);

	  void calc_matrix_boundary_mech(const mesh::Connection& conn, MechApproximation& flux, index_t conn_id);

	  //void calc_fault_fault(const mesh::Connection& conn, Approximation& flux);
	  //void calc_matrix_boundary(const mesh::Connection& conn, Approximation& flux, const index_t adj_mat_id1, const bool with_thermal = false);

	  inline void write_trans_mech(const MechApproximation& flux)
	  {
		const uint8_t BLOCK_SIZE = 4;
		// stencil & transmissibilities
		for (uint8_t st_id = 0; st_id < flux.stencil.size(); st_id++)
		{
		  auto block = flux.a(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux.a.N, 1 });
		  auto block_biot = flux.a_biot(BLOCK_SIZE * st_id, { BLOCK_SIZE, BLOCK_SIZE }, { (size_t)flux.a_biot.N, 1 });
		  block[abs(block) < EQUALITY_TOLERANCE] = 0.0;
		  block_biot[abs(block_biot) < EQUALITY_TOLERANCE] = 0.0;
		  if (abs(block).max() > EQUALITY_TOLERANCE || abs(block_biot).max() > EQUALITY_TOLERANCE)
		  {
			mech_stencil.push_back(flux.stencil[st_id]);
			mech_tran.insert(std::end(mech_tran), std::begin(block), std::end(block));
			mech_tran_biot.insert(std::end(mech_tran_biot), std::begin(block_biot), std::end(block_biot));
		  }
		}
		// offset
		mech_offset.push_back(static_cast<index_t>(mech_stencil.size()));
		// free terms
		mech_rhs.insert(std::end(mech_rhs), std::begin(flux.rhs.values), std::end(flux.rhs.values));
		mech_rhs_biot.insert(std::end(mech_rhs_biot), std::begin(flux.rhs_biot.values), std::end(flux.rhs_biot.values));
	  };

	  void keep_same_stencil_gradients();

	  inline MechApproximation get_displacement_gradient(index_t elem_id)
	  {
		const index_t n_st = u_grad_offset[elem_id + 1] - u_grad_offset[elem_id];
		const index_t grad_coef_size = ND * ND * n_unknowns;
		MechApproximation g(ND * ND, n_unknowns * n_st);
		std::copy_n(u_grad_stencil.begin() + u_grad_offset[elem_id], n_st, g.stencil.begin());
		std::copy_n(u_grad_vals.data() + grad_coef_size * u_grad_offset[elem_id], grad_coef_size * n_st, begin(g.a.values));
		std::copy_n(u_grad_rhs.data() + ND * ND * elem_id, ND * ND, begin(g.rhs.values));

		return g;
	  }
	  inline MechApproximation get_pressure_gradient(index_t elem_id)
	  {
		const index_t n_st = grad_offset[elem_id + 1] - grad_offset[elem_id];
		const index_t grad_coef_size = ND;
		MechApproximation g(ND, n_st);
		std::copy_n(grad_stencil.begin() + grad_offset[elem_id], n_st, g.stencil.begin());
		std::copy_n(p_grad_vals.data() + grad_coef_size * grad_offset[elem_id], grad_coef_size * n_st, begin(g.a.values));
		std::copy_n(p_grad_rhs.data() + ND * ND * elem_id, ND * ND, begin(g.rhs.values));

		return g;
	  }

	public:
	  void init() override;

	  MechDiscretizer();
	  ~MechDiscretizer();

	  /* 3x3 matrices of Biot coefficients */
	  std::vector<Matrix33> biots;
	  /* 6x6 stiffness matrices */
	  std::vector<Stiffness> stfs;
	  /* 3x3 matrices of thermal expansion coefficients */
	  std::vector<Matrix33> th_exps;

	  /* MPFA */

	  // gradient offsets
	  std::vector<index_t> u_grad_offset;
	  // gradient stencil
	  std::vector<index_t> u_grad_stencil;
	  // pressure gradient transmissibilities
	  std::vector<value_t> u_grad_vals;
	  // pressure gradient free-term (gravity)
	  std::vector<value_t> u_grad_rhs;

	  // approximations 
	  std::vector<index_t> mech_cell_m, mech_cell_p, mech_stencil, mech_offset;
	  std::vector<index_t> mech_tran, mech_rhs, mech_tran_biot, mech_rhs_biot;

	  bool USE_CONNECTION_BASED_GRADIENTS;
	  bool NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION;

	  void reconstruct_displacement_gradients_per_cell(const MechBoundaryCondition& bc_mech);

	  void calc_mpfa_mpsa_transmissibilities();

	  MechBoundaryCondition bc_mech;
    };
}

#endif /* MECH_DISCRETIZER_H_ */
