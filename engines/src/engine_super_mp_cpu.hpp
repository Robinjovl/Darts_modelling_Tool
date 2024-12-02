#ifndef CPU_SIMULATOR_SUPER_MP_HPP
#define CPU_SIMULATOR_SUPER_MP_HPP

#include <vector>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "globals.h"
#include "ms_well.h"
#include "engine_base.h"
#include "evaluator_iface.h"

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
class engine_super_mp_cpu : public engine_base
{

public:
  // number of components
  const static uint8_t NC_ = NC;
  // number of phases
  const static uint8_t NP_ = NP;
  // number of primary variables : [P, Z_1, ... Z_(NC-1), T]
  const static uint8_t N_VARS = NC + THERMAL;
  // number of equations
  const static uint8_t NE = N_VARS;
  // order of primary variables:
  const static uint8_t P_VAR = 0;
  const static uint8_t Z_VAR = 1;
  const static uint8_t T_VAR = NC;

  const static uint8_t N_STATE = NC_ + THERMAL;

  // number of operators: NE accumulation operators, NE*NP flux operators, NP up_constant, NE*NP gradient, NE kinetic rate operators, 2*NP gravity and capillarity, 1 porosity, NP enthalpy, 2 temperature and pressure
  const static uint8_t N_OPS = NE /*acc*/ + NE * NP /*flux*/ + NP /*UPSAT*/ + NE * NP /*gradient*/ + NE /*kinetic*/ + 2 * NP /*gravpc*/ + 1 /*poro*/ + NP /*enthalpy*/ + 2 /*temperature and pressure*/;
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t FLUX_OP = NE;
  // diffusion
  const static uint8_t UPSAT_OP = NE + NE * NP;
  const static uint8_t GRAD_OP = NE + NE * NP + NP;
  // kinetic reaction
  const static uint8_t KIN_OP = NE + NE * NP + NP + NE * NP;

  // extra operators
  const static uint8_t GRAV_OP = NE + NE * NP + NP + NE * NP + NE;
  const static uint8_t PC_OP = NE + NE * NP + NP + NE * NP + NE + NP;
  const static uint8_t PORO_OP = NE + NE * NP + NP + NE * NP + NE + 2 * NP;
  const static uint8_t ENTH_OP = NE + NE * NP + NP + NE * NP + NE + 2 * NP + 1;
  const static uint8_t TEMP_OP = NE + NE * NP + NP + NE * NP + NE + 2 * NP + 1 + NP;
  const static uint8_t PRES_OP = NE + NE * NP + NP + NE * NP + NE + 2 * NP + 1 + NP + 1;

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // Define extra class property stoichiometric coefficient of the reaction (now hard coded, but has to become input?, maybe required to be placed somewhere else?):
  std::vector<index_t> stoich_coef;

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  uint8_t get_n_vars() const override { return N_VARS; };
  uint8_t get_n_ops() const override { return N_OPS; };
  uint8_t get_n_comps() const override { return NC; };
  uint8_t get_z_var() const override { return Z_VAR; };
  uint8_t get_n_state() const { return N_STATE; };

  engine_super_mp_cpu()
  {
    if (THERMAL)
    {
      engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component non-isothermal flow with kinetic reaction and diffusion CPU engine with multi-point approximation";
    }
    else
    {
      engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component isothermal flow with kinetic reaction and diffusion CPU engine with multi-point approximation";
    }
  };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
	  std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
	  sim_params *params_, timer_node *timer_);

  int init_jacobian_structure_mpfa(csr_matrix_base *jacobian);

  /// @brief vector of variables in the current timestep provided for operator evaluation
  std::vector<value_t> Xop;
  void extract_Xop();

  // vector of fluxes for every unknown per connection, assembled in jacobian assembly
  std::vector<value_t> fluxes;

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);


  int assemble_linear_system(value_t deltat);
  //double calc_newton_residual();


  // adjoint method--------------------------------------------------------------------------------------

  // initialize dg_dT_general, which is similar to the jacobian initialization
  int init_adjoint_structure_mpfa(csr_matrix_base* init_adjoint);

  // assemble dg_dx_n, dg_dT, dj_dx. This is similar to "init_jacobian_structure" in the forward simulation
  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);



public:
};

#include "engine_super_mp_cpu.tpp"

#endif /* CPU_SIMULATOR_SUPER_MP_HPP */
