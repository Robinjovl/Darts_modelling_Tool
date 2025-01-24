#ifndef MECH_DISCRETIZER_H_
#define MECH_DISCRETIZER_H_

#include "discretizer.h"

/**
 * @def SUM_N(N)
 * @brief Computes the sum of the first N natural numbers using the arithmetic series formula.
 *
 * @param N The last number in the series to sum up to.
 * @return The sum of the arithmetic series from 1 to N.
 */
#define SUM_N(N) ((N + 1) / 2 * N)

namespace dis
{
  /**
	* @brief Boundary condition structure for Thermo-Hydro-Mechanical (THM) coupled problems.
	*
	* This structure aggregates various types of boundary conditions relevant for THM analysis,
	* including flow, thermal, and mechanical (both normal and tangential).
	*/
  struct THMBoundaryCondition
	{
	BoundaryCondition flow;        ///< Flow boundary conditions.
	BoundaryCondition thermal;     ///< Thermal boundary conditions.
	BoundaryCondition mech_normal; ///< Mechanical normal boundary conditions.
	BoundaryCondition mech_tangen; ///< Mechanical tangential boundary conditions.
	};

  /**
	* @brief Represents a 6x6 stiffness matrix, typically used in mechanical simulations.
	*/
	class Stiffness : public Matrix
	{
	public:
	static const index_t N = 6; ///< Dimension of the stiffness matrix.
	typedef Matrix Base; ///< Base class alias.

	/**
	  * @brief Default constructor for Stiffness, initializes a 6x6 matrix.
	  */
	  Stiffness() : Base(6, 6) {};

	/**
	  * @brief Constructs a stiffness matrix from Lame coefficients.
	  *
	  * @param lambda First Lame coefficient.
	  * @param mu Second Lame coefficient.
	  */
	  Stiffness(value_t lambda, value_t mu) : Base(6, 6)
	  {
		(*this)(0, 0) = (*this)(1, 1) = (*this)(2, 2) = lambda + 2 * mu;
		(*this)(3, 3) = (*this)(4, 4) = (*this)(5, 5) = mu;
		(*this)(0, 1) = (*this)(0, 2) = (*this)(1, 2) = lambda;
		(*this)(1, 0) = (*this)(2, 0) = (*this)(2, 1) = lambda;
	  };

	/**
	  * @brief Constructs a stiffness matrix from a given array of values.
	  *
	  * @param _c Array of values to initialize the matrix.
	  */
	  Stiffness(std::valarray<value_t> _c) : Base(_c, 6, 6) {}
	};

  /**
	* @brief Enumeration of modes for the mechanical discretizer.
	*/
  enum MechDiscretizerMode {
	POROELASTIC,         ///< Represents the poroelastic discretization mode.
	THERMOPOROELASTIC    ///< Represents the thermoporoelastic discretization mode.
  };

  /**
	* @brief Maps discretizer modes to the number of unknowns in each mode.
	*
	* This map associates each MechDiscretizerMode with the count of unknowns
	* needed for that particular mode. ND represents the spatial dimensions.
	*/
  const std::unordered_map<MechDiscretizerMode, uint8_t> N_UNKNOWNS = {
	  { POROELASTIC, ND + 1 },         ///< ND + 1 unknowns for poroelastic mode.
	  { THERMOPOROELASTIC, ND + 2 }    ///< ND + 2 unknowns for thermoporoelastic mode.
  };

  /**
	* @brief Template alias for defining approximation types based on the discretizer mode.
	*
	* For THERMOPOROELASTIC mode, it resolves to LinearApproximation with Uvar, Pvar, and Tvar variables.
	* For POROELASTIC mode, it resolves to LinearApproximation with Uvar and Pvar variables.
	*
	* @tparam MODE The discretizer mode to determine the approximation type.
	*/
	template <MechDiscretizerMode MODE>
	using ApproximationType = typename std::conditional<MODE == THERMOPOROELASTIC,
	LinearApproximation<Uvar, Pvar, Tvar>, // Approximation for thermoporoelastic mode.
	LinearApproximation<Uvar, Pvar>>::type; // Approximation for poroelastic mode.

  /**
	* @brief Represents a structure for the approximation of fluxes in THM modeling.
	*
	* @tparam MODE The discretization mode.
	*/
	template <MechDiscretizerMode MODE>
	struct MechApproximation
	{
	/**
	  * @brief Default constructor for MechApproximation.
	  */
	  MechApproximation() {};

	/**
	  * @brief Constructs a MechApproximation with a specified stencil size.
	  *
	  * @param stencil_size The size of the stencil to be used in approximations.
	  */
	  MechApproximation(index_t stencil_size)
	  {
		hooke = ApproximationType<MODE>(ND, stencil_size);
		biot_traction = LinearApproximation<Pvar>(ND, stencil_size);
		vol_strain = ApproximationType<MODE>(1, stencil_size);
		flow = FlowHeatApproximation(stencil_size);
		thermal_traction = LinearApproximation<Tvar>(ND, stencil_size);
	  };

	ApproximationType<MODE> hooke; ///< Approximation for Hooke's law.
	LinearApproximation<Pvar> biot_traction; ///< Approximation for Biot's traction.
	ApproximationType<MODE> vol_strain; ///< Approximation for Biot's * volumetric strain.
	FlowHeatApproximation flow; ///< Approximation for combined fluid flow and heat condution fluxes.
	  LinearApproximation<Tvar> thermal_traction;

	bool is_same_stencil = true; ///< Flag indicating if the same stencil is used for all approximations.
	};

  /**
	* @brief A mechanical discretizer.
	*
	* @tparam MODE The discretization mode.
	*/
	template <MechDiscretizerMode MODE>
	class MechDiscretizer : public Discretizer
	{
	protected:

	static const uint8_t n_unknowns; ///< The number of variables per cell.

	/**
	  * @brief Cached matrices structure holding terms for co-normal decomposition
	  */
	  struct InnerMatrices
	  {
	  Matrix T1, T2;  ///< 3x3 matrices, conormal stiffness.
	  Matrix G1, G2;  ///< 3x9 matrices, transversal stiffness.
	  Matrix R1, R2;  ///< 3x1 vectors, free terms in traction balance.
	  Matrix y1, y2;  ///< 3x1 vectors, tangential components of vectors between cell and interface centers.
	  value_t r1, r2; ///< Distances from cell centers to the interface.
	  };
	  
	std::vector<std::map<index_t, InnerMatrices>> inner; ///< Cached matrices for matrix-matrix connections.

	std::unordered_map<index_t, Matrix> pre_grad_A_u; ///< Pre-allocated matrices for gradient reconstruction.
	std::unordered_map<index_t, Matrix> pre_grad_R_u; ///< Pre-allocated matrices for gradient reconstruction.
	std::unordered_map<index_t, Matrix> pre_grad_rhs_u; ///< Pre-allocated matrices for gradient reconstruction.
	std::map<index_t, std::map<index_t, Matrix>> pre_cur_rhs; ///< Pre-allocated matrices for gradient reconstruction.
	std::unordered_map<index_t, Matrix> pre_N, pre_R, pre_Nflux; ///< Pre-allocated matrices for reconstruction of cell-centered stresses / velocities.
	std::unordered_map<index_t, Matrix> pre_stress_approx, pre_vel_approx; ///< Pre-allocated matrices for reconstruction of cell-centered stresses / velocities.

	std::vector<MechApproximation<MODE>> mech_fluxes; ///< Vector of mechanical flux approximations.

	Matrix W; ///< W matrix.

	std::pair<bool, size_t> res1, res2; ///< Result containers for internal use.

	/**
	  * @brief Finds an element in a given vector and returns its position.
	  *
	  * @param vec The vector to search in.
	  * @param element The element to find.
	  * @return std::pair<bool, size_t> A pair where the first element is true if found,
	  *         false otherwise, and the second element is the position in the vector.
	  */
	  inline std::pair<bool, size_t> findInVector(const std::vector<index_t>& vec, 
												  const index_t& element)
	  {
	  for (size_t i = 0; i < vec.size(); ++i)
		{
		if (vec[i] == element)
		{
		  return { true, i };
		}
	  }
	  return { false, static_cast<size_t>(-1) };
	  };

	/**
	  * @brief Calculates approximation of tractions for matrix-matrix connections.
	  *
	  * @param conn The mesh connection to consider.
	  * @param flux The flux to store approximation.
	  * @param cell_id The ID of the cell.
	  * @param conn_id The ID of the connection.
	  */
	  void calc_matrix_matrix_mech(const mesh::Connection& conn, 
									MechApproximation<MODE>& flux, 
									index_t cell_id, 
									index_t conn_id);

	/**
	  * @brief Calculates approximation of tractions for matrix-boundary connections.
	  *
	  * @param conn The mesh connection to consider.
	  * @param flux The flux to store approximation.
	  * @param conn_id The ID of the connection.
	  */
	  void calc_matrix_boundary_mech(const mesh::Connection& conn, 
									  MechApproximation<MODE>& flux, 
									  index_t conn_id);

	/**
	  * @brief Writes mechanical transmissibilities in plain arrays
	  *
	  * @param flux The approximation of the fluxes to be written in plain arrays.
	  */
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
		  std::valarray<value_t> block_thermal;
		  if constexpr (MODE == THERMOPOROELASTIC) {
			  block_thermal = flux.thermal_traction.a(flux.thermal_traction.n_block * st_id, { (size_t)flux.thermal_traction.a.M, (size_t)flux.thermal_traction.n_block }, { (size_t)flux.thermal_traction.a.N, 1 });
			  coef_fourier = flux.flow.fourier.a.values[st_id];
		  }
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
			if constexpr (MODE == THERMOPOROELASTIC) {
				// Fourier's law
				fourier.push_back(coef_fourier);
				// Thermal term in traction
				thermal_traction.insert(std::end(thermal_traction), std::begin(block_thermal), std::end(block_thermal));
			}
		  }
		}
		// free terms
		hooke_rhs.insert(std::end(hooke_rhs), std::begin(flux.hooke.rhs.values), std::end(flux.hooke.rhs.values));
		biot_traction_rhs.insert(std::end(biot_traction_rhs), std::begin(flux.biot_traction.rhs.values), std::end(flux.biot_traction.rhs.values));
		biot_vol_strain_rhs.push_back(flux.vol_strain.rhs.values[0]);
		darcy_rhs.push_back(flux.flow.darcy.rhs.values[0]);
		fick_rhs.push_back(flux.flow.fick.rhs.values[0]);
	  };

	/**
	  * @brief Maintains the same stencil between pressure/temperature and displacement gradients.
	  */
	  void keep_same_stencil_gradients();
	public:

	/**
	  * @brief Constructor for MechDiscretizer.
	  */
	  MechDiscretizer();

	/**
	  * @brief Destructor for MechDiscretizer.
	  */
	  ~MechDiscretizer();

	/**
	  * @brief Initializes the discretizer.
	  */
	void init() override;

	std::vector<Matrix33> biots; ///< 3x3 matrices of Biot coefficients for each cell.
	std::vector<Stiffness> stfs; ///< 6x6 stiffness matrices for each cell.
	std::vector<Matrix33> th_exps; ///< 3x3 matrices of thermal expansion coefficients for each cell.

	std::vector<ApproximationType<MODE>> u_grads; ///< The approximation of cell-wise displacement gradients.

	std::vector<value_t> hooke, hooke_rhs; ///< Hooke's law approximations and their RHS
	std::vector<value_t> biot_traction, biot_traction_rhs; ///< Biot's term in traction and its RHS
	std::vector<value_t> darcy, darcy_rhs; ///< Darcy's flow approximations and their RHS
	std::vector<value_t> biot_vol_strain, biot_vol_strain_rhs; ///< Biot's term in fluid flow and its RHS
	std::vector<value_t> fick, fick_rhs; ///< Fick's law approximations and their RHS
	std::vector<value_t> fourier; ///< Fourier's law approximations
	std::vector<value_t> thermal_traction; ///< Thermal term in traction

	std::vector<value_t> stress_approx; ///< Approximation of cell-centered stress tensor over tractions
	std::vector<value_t> velocity_approx; ///< Approximation of cell-centered velocity over fluid fluxed

	bool USE_CONNECTION_BASED_GRADIENTS; ///< Flag for using connection-based gradients
	bool NEUMANN_BOUNDARIES_GRAD_RECONSTRUCTION; ///< Flag for using Neumann boundaries in gradient reconstruction
	bool GRADIENTS_EXTENDED_STENCIL; ///< Flag for using extended stencil in gradient reconstruction

	THMBoundaryCondition bc_thm; ///< THM boundary condition

	/**
	  * @brief Reconstructs displacement gradients for each cell based on boundary conditions.
	  *
	  * @param bc_mech The array of THM boundary conditions.
	  */
	  void reconstruct_displacement_gradients_per_cell(const THMBoundaryCondition& bc_mech);

	/**
	  * @brief Calculates the approximations of fluxes at all interfaces in computational grid
	  */
	void calc_interface_approximations();

	/**
	  * @brief Calculates the approximations of stress tensor and Darcy velocities at cells' centers
	*/
	void calc_cell_centered_stress_velocity_approximations();
	};
}

#endif /* MECH_DISCRETIZER_H_ */
