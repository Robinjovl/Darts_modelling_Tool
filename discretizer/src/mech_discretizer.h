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
		a_thermal = Matrix(M, N);
		rhs_thermal = Matrix(M, 1);
		stencil.resize(N);
	  };
	  Matrix a, rhs, a_biot, rhs_biot, a_thermal, rhs_thermal;
	  std::vector<index_t> stencil;
	};

	enum MechDiscretizerMode { POROELASTIC, THERMOPOROELASTIC };

	const std::unordered_map<MechDiscretizerMode, std::pair<uint8_t, uint8_t>> BLOCK_DIM = { { POROELASTIC, {ND, ND + 1} }, { THERMOPOROELASTIC, {ND, ND + 2} } };

	/* Discretiser */
	template <MechDiscretizerMode MODE>
	class MechDiscretizer : public Discretizer
	{
	protected:
	  std::unordered_map<index_t, Matrix> pre_grad_A_u, pre_grad_R_u, pre_grad_rhs_u;
	  std::vector<MechApproximation> pre_merged_mom_flux, mom_fluxes;

	  static const index_t M;
	  static const index_t N;

	  //void calc_matrix_matrix(const mesh::Connection& conn, Approximation& flux, const index_t adj_mat_id1, const index_t adj_mat_id2, const bool with_thermal = false);
	  //void calc_fault_fault(const mesh::Connection& conn, Approximation& flux);
	  //void calc_matrix_boundary(const mesh::Connection& conn, Approximation& flux, const index_t adj_mat_id1, const bool with_thermal = false);
	  /*
	  inline void write_trans(const Approximation& flux)
	  {
		  value_t buf, buf_homo;
		  // free term (gravity)
		  flux_rhs.push_back(flux.rhs.values[0]);
		  // stencil & transmissibilities
		  for (uint8_t st_id = 0; st_id < flux.stencil.size(); st_id++)
		  {
			  buf = flux.a.values[st_id];
			  buf_homo = flux.a_homo.values[st_id];
			  if (fabs(buf) > EQUALITY_TOLERANCE)
			  {
				  flux_vals.push_back(buf);
				  flux_vals_homo.push_back(buf_homo);
				  flux_stencil.push_back(flux.stencil[st_id]);
			  }
		  }
		  // offset
		  flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));
	  };
	  inline void write_trans_thermal(const Approximation& flux)
	  {
		value_t buf, buf_homo, buf_t;
		// free term (gravity)
		flux_rhs.push_back(flux.rhs.values[0]);
		// stencil & transmissibilities
		for (uint8_t st_id = 0; st_id < flux.stencil.size(); st_id++)
		{
		  buf = flux.a.values[st_id];
		  buf_homo = flux.a_homo.values[st_id];
		  buf_t = flux.a_thermal.values[st_id];
		  if (fabs(buf) + fabs(buf_t) > EQUALITY_TOLERANCE)
		  {
			flux_vals.push_back(buf);
			flux_vals_homo.push_back(buf_homo);
			flux_vals_thermal.push_back(buf_t);
			flux_stencil.push_back(flux.stencil[st_id]);
		  }
		}
		// offset
		flux_offset.push_back(static_cast<index_t>(flux_stencil.size()));
	  };
	  */
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

	  bool USE_CONNECTION_BASED_GRADIENTS;
	  bool NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION;

	  void reconstruct_displacement_gradients_per_cell(const MechBoundaryCondition& bc);

	  MechBoundaryCondition bc;
    };
}

#endif /* MECH_DISCRETIZER_H_ */
