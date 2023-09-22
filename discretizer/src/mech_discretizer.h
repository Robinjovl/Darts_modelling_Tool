#ifndef MECH_DISCRETIZER_H_
#define MECH_DISCRETIZER_H_

#include "discretizer.h"

namespace dis
{
	// Boundary condition 
	class GenericBoundaryCondition
	{
	public:
		// length = number of boundary elements
		// Dirichle type
	  std::vector<value_t> a; 
		// Neumann type
		std::vector<value_t> b; 
		GenericBoundaryCondition() {};
	  ~GenericBoundaryCondition() {};
	};

	/* Boundary condition for Thermo-Hydro-Mechanical coupled problem */
	class THMBoundaryCondition
	{

	public:
		GenericBoundaryCondition flow;
		GenericBoundaryCondition thermal;
		GenericBoundaryCondition mech_normal;
		GenericBoundaryCondition mech_tangen;
		THMBoundaryCondition() {};
		~THMBoundaryCondition() {};
	};

	/* 6x6 stiffness matrix */
	class Stiffness : public Matrix
	{
	public:
	  static const index_t N = 6;
	  typedef Matrix Base;

	  Stiffness() : Base(6, 6) {};
		// Stiffness matrix from Lame coefficients 
	  Stiffness(value_t lambda, value_t mu) : Base(6, 6)
	  {
		(*this)(0, 0) = (*this)(1, 1) = (*this)(2, 2) = lambda + 2 * mu;
		(*this)(3, 3) = (*this)(4, 4) = (*this)(5, 5) = mu;
		(*this)(0, 1) = (*this)(0, 2) = (*this)(1, 2) = lambda;
		(*this)(1, 0) = (*this)(2, 0) = (*this)(2, 1) = lambda;
	  };
	  Stiffness(std::valarray<value_t> _c) : Base(_c, 6, 6) {}
	};

	enum MechDiscretizerMode { POROELASTIC, THERMOPOROELASTIC };

	const std::unordered_map<MechDiscretizerMode, uint8_t> N_UNKNOWNS = { { POROELASTIC, ND + 1 }, { THERMOPOROELASTIC, ND + 2 } };

	template <MechDiscretizerMode MODE>
	using ApproximationType = typename std::conditional<MODE == THERMOPOROELASTIC,
	  LinearApproximation<Uvar, Pvar, Tvar>,
	  LinearApproximation<Uvar, Pvar>>::type;

	template <MechDiscretizerMode MODE>
	struct MechApproximation
	{
	public:
	  MechApproximation() {};
	  MechApproximation(index_t stencil_size)
	  {
		hooke = ApproximationType<MODE>(ND, stencil_size);
		biot_traction = LinearApproximation<Pvar>(ND, stencil_size);
		biot_vol_strain = ApproximationType<MODE>(1, stencil_size);
	  };

	  ApproximationType<MODE> hooke;
	  LinearApproximation<Pvar> biot_traction;
	  ApproximationType<MODE> biot_vol_strain;

	  bool is_same_stencil = true;
	};

	/* Discretiser */
	template <MechDiscretizerMode MODE>
	class MechDiscretizer : public Discretizer
	{
	protected:
	  struct InnerMatrices
	  {
		Matrix T1, T2;	// 3x3 matrices, conormal stiffness,
		Matrix G1, G2;	// 3x9 matrices, transversal stiffness
		Matrix R1, R2;	// 3x1 vectors, free terms in traction balance
		Matrix y1, y2;	// 3x1 vectors, tangential components of vectors between cell and inteface centers
		value_t r1, r2; // distances from cell centers to the interface
	  };
	  
	  std::unordered_map<index_t, Matrix> pre_grad_A_u, pre_grad_R_u, pre_grad_rhs_u;
	  std::map<index_t, std::map<index_t, Matrix>> pre_cur_rhs;
	  std::vector<MechApproximation<MODE>> mech_fluxes;
	  Matrix W;
	  // cache for the "inner" (matrix-matrix) connections to reduce computations, size n_cells
	  std::vector<std::map<index_t, InnerMatrices>> inner;
	  // the number of variables per cell
	  static const uint8_t n_unknowns;

	  std::pair<bool, size_t> res1, res2;
	  inline std::pair<bool, size_t> findInVector(const std::vector<index_t>& vec, const index_t& element)
	  {
		std::vector<index_t>::const_iterator it_find;
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

	  void calc_matrix_matrix_mech(const mesh::Connection& conn, MechApproximation<MODE>& flux, index_t conn_id);

	  void calc_matrix_boundary_mech(const mesh::Connection& conn, MechApproximation<MODE>& flux, index_t conn_id);

	  //void calc_fault_fault(const mesh::Connection& conn, Approximation& flux);
	  //void calc_matrix_boundary(const mesh::Connection& conn, Approximation& flux, const index_t adj_mat_id1, const bool with_thermal = false);

	  inline void write_trans_mech(const MechApproximation<MODE>& flux)
	  {
		/*const uint8_t BLOCK_SIZE = 4;
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
		mech_rhs_biot.insert(std::end(mech_rhs_biot), std::begin(flux.rhs_biot.values), std::end(flux.rhs_biot.values));*/
	  };

	  void keep_same_stencil_gradients();

	public:
	  void init() override;

	  MechDiscretizer();
	  ~MechDiscretizer();

	  // all geomechanical properties such as  Biot, Stifffness, etc are for the drained conditions

	  // 3x3 matrices of Biot coefficients for the each cell
	  std::vector<Matrix33> biots;
	  // 6x6 stiffness matrices for the each cell (tensor of rank 4)
	  std::vector<Stiffness> stfs;
	  // 3x3 matrices of thermal expansion coefficients for the each cell
	  std::vector<Matrix33> th_exps;

	  /* MPFA */
		// gradinents stored in 1-dimensional arrays with the stride=9
		// 9 values for the each cell
		// (u_x)'x   (u_x)'y   (u_x)'z
		// (u_y)'x   (u_y)'y   (u_y)'z
		// (u_z)'x   (u_z)'y   (u_z)'z


	  // pressure/heat gradient transmissibilities, flattened vector with 9*n_unknowns for the each cell 
	  std::vector<ApproximationType<MODE>> u_grads;

		// grad = sum(i=1..stencil) A_i * (u, p, temperature)_i + b
		// fot THM: A - 9x5, b - 5x1, len{u_x, u_y, u_z, p, temperature}

	  // approximations 
	  std::vector<index_t> mech_cell_m, mech_cell_p, mech_stencil, mech_offset;
	  std::vector<index_t> mech_tran, mech_rhs, mech_tran_biot, mech_rhs_biot;

	  bool USE_CONNECTION_BASED_GRADIENTS;
	  bool NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION;
	  bool GRADIENTS_EXTENDED_STENCIL;

	  void reconstruct_displacement_gradients_per_cell(const THMBoundaryCondition& bc_mech);

	  void calc_mpfa_mpsa_transmissibilities();

	  THMBoundaryCondition bc_thm;
    };
}

#endif /* MECH_DISCRETIZER_H_ */
