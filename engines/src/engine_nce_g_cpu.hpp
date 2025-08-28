#ifndef CPU_SIMULATOR_NCE_G_HPP
#define CPU_SIMULATOR_NCE_G_HPP

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


template <uint8_t NC, uint8_t NP>
class engine_nce_g_cpu : public engine_base
{
private:
  // utilities for scaling and better conditionining of Jacobian
  void make_dimensionless();
  void dimensionalize_unknowns();

public:
  // number of components
  const static uint8_t NC_ = NC;
  // number of phases
  const static uint8_t NP_ = NP;
  // number of primary variables : [P, Z_1, ... Z_(NC-1), E]
  const static uint8_t N_VARS = NC + 1;
  // order of primary variables:
  const static uint8_t P_VAR = 0;
  const static uint8_t Z_VAR = 1;
  const static uint8_t E_VAR = NC; // FLUID ENTHALPY
  // number of operators:
  // mass: NC accumulation operators, NC*NP flux operators
  // energy: 1    fluid energy accumulation,
  //         1    rock energy accumulation,
  //         NP   fluid energy flux,
  //         1    fluid conduction,
  //         1    rock conduction,
  //         1    temperature,
  //         1    water density,
  //         1    steam density
  const static uint8_t N_OPS = NC /*acc*/ + NC * NP /*flux*/ + 2 + NP /*energy acc, flux, cond*/ + NP /*density*/ + 1 /*temperature*/;
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t FLUX_OP = NC;
  const static uint8_t FE_ACC_OP = NC + NC * NP;
  const static uint8_t FE_FLUX_OP = NC + NC * NP + 1;
  const static uint8_t FE_COND_OP = NC + NC * NP + NP + 1;
  const static uint8_t DENS_OP = NC + NC * NP + NP + 2;
  const static uint8_t TEMP_OP = NC + NC * NP + NP + 2 + NP;

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  uint8_t get_n_vars() const override { return N_VARS; };
  uint8_t get_n_ops() const override { return N_OPS; };
  uint8_t get_n_comps() const override { return NC; };
  uint8_t get_z_var() const override { return Z_VAR; };

  engine_nce_g_cpu() { engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component enthalpy-based thermal flow with gravity CPU engine"; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);
  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

  double calc_newton_residual_L2();
  double calc_newton_residual_Linf();
  double calc_well_residual_L2();
  double calc_well_residual_Linf();

  double H2O_MW = 18.01528;

  int solve_linear_equation();

  void enable_flux_output();
};
#endif
