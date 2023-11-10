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
		vol_strain = ApproximationType<MODE>(1, stencil_size);
		flow = FlowHeatApproximation(stencil_size);
	  };

	  ApproximationType<MODE> hooke;
	  LinearApproximation<Pvar> biot_traction;
	  ApproximationType<MODE> vol_strain;
	  FlowHeatApproximation flow;

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

	  void calc_matrix_matrix_mech(const mesh::Connection& conn, MechApproximation<MODE>& flux, index_t cell_id, index_t conn_id);

	  void calc_matrix_boundary_mech(const mesh::Connection& conn, MechApproximation<MODE>& flux, index_t conn_id);

	  //void calc_fault_fault(const mesh::Connection& conn, Approximation& flux);
	  //void calc_matrix_boundary(const mesh::Connection& conn, Approximation& flux, const index_t adj_mat_id1, const bool with_thermal = false);

	  inline void write_trans_mech(const MechApproximation<MODE>& flux)
	  {
		assert(flux.is_same_stencil);
		value_t coef_darcy, coef_fick, coef_fourier;
		assert(flux.hooke.stencil == flux.flow.darcy.stencil);

		// stencil & transmissibilities
		for (index_t st_id = 0; st_id < flux.hooke.stencil.size(); st_id++)
		{
		  auto block_hooke = flux.hooke.a(flux.hooke.n_block * st_id, { (size_t)flux.hooke.a.M, (size_t)flux.hooke.n_block }, { (size_t)flux.hooke.a.N, 1 });
		  auto block_biot = flux.biot_traction.a(flux.biot_traction.n_block * st_id, { (size_t)flux.biot_traction.a.M, (size_t)flux.biot_traction.n_block }, { (size_t)flux.biot_traction.a.N, 1 });
		  auto block_vol_strain = flux.vol_strain.a(flux.vol_strain.n_block * st_id, { (size_t)flux.vol_strain.n_block }, { 1 });
		  coef_darcy = flux.flow.darcy.a.values[st_id];
		  coef_fick = flux.flow.fick.a.values[st_id];
		  // eliminate numerical noise: TODO: formalize
		  // block_hooke[abs(block_hooke) < EQUALITY_TOLERANCE] = 0.0;
		  // block_biot[abs(block_biot) < EQUALITY_TOLERANCE] = 0.0;
		  // block_vol_strain[abs(block_vol_strain) < EQUALITY_TOLERANCE] = 0.0;
		  // add transmissibilities
		  if (abs(block_hooke).max() > EQUALITY_TOLERANCE || 
			  abs(block_biot).max() > EQUALITY_TOLERANCE ||
			  abs(block_vol_strain).max() > EQUALITY_TOLERANCE ||
			  abs(coef_darcy) > EQUALITY_TOLERANCE)
		  {
			// stencil
			flux_stencil.push_back(flux.hooke.stencil[st_id]);
			// Hooke's law
			hooke.insert(std::end(hooke), std::begin(block_hooke), std::end(block_hooke));
			// Biot's term in traction
			biot_traction.insert(std::end(biot_traction), std::begin(block_biot), std::end(block_biot));
			// Biot's term in fluid flow
			biot_vol_strain.insert(std::end(biot_vol_strain), std::begin(block_vol_strain), std::end(block_vol_strain));
			// Darcy's flow
			darcy.push_back(coef_darcy);
			fick.push_back(coef_fick);
		  }
		}
		// free terms
		hooke_rhs.insert(std::end(hooke_rhs), std::begin(flux.hooke.rhs.values), std::end(flux.hooke.rhs.values));
		biot_traction_rhs.insert(std::end(biot_traction_rhs), std::begin(flux.biot_traction.rhs.values), std::end(flux.biot_traction.rhs.values));
		biot_vol_strain_rhs.push_back(flux.vol_strain.rhs.values[0]);
		darcy_rhs.push_back(flux.flow.darcy.rhs.values[0]);
		fick_rhs.push_back(flux.flow.fick.rhs.values[0]);
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
	  std::vector<value_t> hooke, hooke_rhs;
	  std::vector<value_t> biot_traction, biot_traction_rhs;
	  std::vector<value_t> darcy, darcy_rhs;
	  std::vector<value_t> biot_vol_strain, biot_vol_strain_rhs;
	  std::vector<value_t> fick, fick_rhs;
	  std::vector<value_t> fourier, fourier_rhs;

	  bool USE_CONNECTION_BASED_GRADIENTS;
	  bool NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION;
	  bool GRADIENTS_EXTENDED_STENCIL;

	  void reconstruct_displacement_gradients_per_cell(const THMBoundaryCondition& bc_mech);

	  void calc_mpfa_mpsa_transmissibilities();

	  THMBoundaryCondition bc_thm;
    };
}

#endif /* MECH_DISCRETIZER_H_ */
