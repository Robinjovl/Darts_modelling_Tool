#ifndef ENGINE_SUPER_ELASTIC_CPU_HPP
#define ENGINE_SUPER_ELASTIC_CPU_HPP

#include <vector>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <type_traits>

#include "globals.h"
#include "ms_well.h"
#include "engine_base.h"
#include "evaluator_iface.h"
#include "mech/contact.h"
#include "../../discretizer/src/mech_discretizer.h"

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#else
#include "csr_matrix.h"
#include "linsolv_iface.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::auxiliary;
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS

template <uint8_t NC, uint8_t NP, bool THERMAL>
class engine_super_elastic_cpu : public engine_base
{
public:
  /// @brief Compile-time evaluation of discretizer type.
  using DiscretizerType = typename std::conditional<THERMAL,
    dis::MechDiscretizer<dis::MechDiscretizerMode::THERMOPOROELASTIC>,
    dis::MechDiscretizer<dis::MechDiscretizerMode::POROELASTIC>>::type;
protected:
  /// @brief Pointer to discretizer required for the evaluation of stresses and velocities.
  DiscretizerType* discr;
public:
  // number of components
  const static uint8_t NC_ = NC;
  // number of phases
  const static uint8_t NP_ = NP;
  // number of primary variables : [P, Z_1, ... Z_(NC-1), T]
  const static uint8_t N_VARS = NC_ + THERMAL + ND;
  // number of equations
  const static uint8_t NE = NC_ + THERMAL;
  // order of primary variables:
  const static uint8_t P_VAR = 0;
  const static uint8_t Z_VAR = 1;
  const static uint8_t T_VAR = NC;
  const static uint8_t U_VAR = NE;
  // size of transmissibility block
  const static uint8_t NT = ND + 1 + THERMAL;
  // order of variables in transmissibility block
  const static uint8_t U_VAR_T = 0;
  const static uint8_t P_VAR_T = 3;
  // number of boundary conditions
  const static uint8_t N_BC_VARS = 1 + THERMAL + ND;
  // order of boundary conditions
  const static uint8_t P_BC_VAR = 0;
  const static uint8_t T_BC_VAR = THERMAL;
  const static uint8_t U_BC_VAR = 1 + THERMAL;

  // dimension of state space
  const static uint8_t N_STATE = NC_ + THERMAL;

  // number of operators: NE accumulation operators, NE*NP flux operators, NP up_constant, NE*NP gradient, NE kinetic rate operators, 2 rock internal energy and conduction, 2*NP gravity and capillarity, 1 porosity
  const static uint8_t N_OPS = NE /*acc*/ + NE * NP /*flux*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/ + 2 * NP /*gravpc*/ + 1 /*poro*/ + NP /*enthalpy*/ + 2 /*temperature and pressure*/ + 1 /*weight*/;
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t FLUX_OP = NE;
  // diffusion
  const static uint8_t UPSAT_OP = FLUX_OP + NE * NP;
  const static uint8_t GRAD_OP = UPSAT_OP + NP;
  // kinetic reaction
  const static uint8_t KIN_OP = GRAD_OP + NE * NP;

  // extra operators
  const static uint8_t GRAV_OP = KIN_OP + NE;
  const static uint8_t PC_OP = GRAV_OP + NP;
  const static uint8_t PORO_OP = PC_OP + NP;
  const static uint8_t ENTH_OP = PORO_OP + 1;
  const static uint8_t TEMP_OP = ENTH_OP + NP;
  const static uint8_t PRES_OP = TEMP_OP + 1;
  const static uint8_t ROCK_DENS = PRES_OP + 1;
  
  const static uint8_t SAT_OP = UPSAT_OP;
  // mapping 
  // from transmissibility order of unknowns 
  // to the order of unknowns in simulation
  const static uint8_t T2U[5];
  const static uint8_t BC2U[5];

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // Define extra class property stoichiometric coefficient of the reaction (now hard coded, but has to become input?, maybe required to be placed somewhere else?):
  std::vector<index_t> stoich_coef;

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  uint8_t get_n_vars() const { return N_VARS; };
  uint8_t get_n_ops() const  { return N_OPS; };
  uint8_t get_n_comps() const  { return NC; };
  uint8_t get_z_var() const  { return Z_VAR; };
  uint8_t get_n_state() const { return N_STATE; };

  engine_super_elastic_cpu()
  {
    if (THERMAL)
    {
      engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component non-isothermal poroelasticity with kinetic reaction and diffusion CPU engine";
    }
    else
    {
      engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component isothermal poroelasticity with kinetic reaction and diffusion CPU engine";
    }
  };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
	  std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
	  sim_params *params_, timer_node *timer_);

  int init_jacobian_structure_pme(csr_matrix_base *jacobian);
  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);
  int eval_stresses_and_velocities();

  int solve_linear_equation();
  //void apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
  int assemble_linear_system(value_t deltat);
  int post_newtonloop(value_t deltat, value_t time, index_t converged);

  /// @brief vector of variables in the current timestep provided for operator evaluation
  std::vector<value_t> Xop;
  void extract_Xop();

  std::vector<value_t> calc_newton_dev_L2();
  std::vector<value_t> calc_newton_dev();
  double calc_well_residual_L2();
  int apply_newton_update(value_t dt);

  value_t dev_u,		dev_p,		dev_e,		dev_z[NC], dev_g;
  value_t dev_u_prev,	dev_p_prev, dev_e_prev, dev_z_prev[NC], dev_g_prev, well_residual_prev_dt;
  
  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

  void set_discretizer(DiscretizerType* _discr);

public:

  std::vector<value_t> eps_vol;

  /// @brief Vector storing Darcy fluxes.
  std::vector<value_t> darcy_fluxes;

  /// @brief Vector storing fluid fluxes caused by matrix movement.
  std::vector<value_t> structural_movement_fluxes;

  /// @brief Vector storing heat conduction fluxes.
  std::vector<value_t> fourier_fluxes;

  /// @brief Vector storing molecular diffusion fluxes.
  std::vector<value_t> fick_fluxes;

  /// @brief Vectors storing pure elastic forces at this and previous time steps.
  std::vector<value_t> hooke_forces, hooke_forces_n;

  /// @brief Vectors storing pore pressure-induced forces at this and previous time steps.
  std::vector<value_t> biot_forces, biot_forces_n;

  /// @brief Vectors storing thermally-induced forces at this and previous time steps.
  std::vector<value_t> thermal_forces, thermal_forces_n;

  /// @brief Vector storing cell-centered total stresses in Voigt notation.
  std::vector<value_t> total_stresses;

  /// @brief Vector storing cell-centered effective Biot stresses in Voigt notation.
  std::vector<value_t> effective_stresses;

  /// @brief Vector storing cell-centered Darcy velocities.
  std::vector<value_t> darcy_velocities;

  std::vector<value_t> Xref, Xn_ref;
  bool FIND_EQUILIBRIUM;
  std::vector<pm::contact> contacts;
  pm::ContactSolver contact_solver;
  std::vector<index_t> geomechanics_mode;
  std::array<value_t, ND> gravity;

  void apply_composition_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);
  void apply_global_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);
  void apply_composition_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
  void apply_global_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX);

  void apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
};

#include "engine_super_elastic_cpu.tpp"

#endif /* ENGINE_SUPER_ELASTIC_CPU_HPP */
