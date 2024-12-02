#ifndef CPU_SIMULATOR_NC_CG_HPP
#define CPU_SIMULATOR_NC_CG_HPP

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
class engine_nc_cg_cpu : public engine_base
{

public:
  // number of components
  const static uint8_t NC_ = NC;
  // number of phases
  const static uint8_t NP_ = NP;
  // number of primary variables : [P, Z_1, ... Z_(NC-1)]
  const static uint8_t N_VARS = NC;
  // order of primary variables:
  const static uint8_t P_VAR = 0;
  const static uint8_t Z_VAR = 1;
  // number of operators: NC accumulation operators, NC*NP flux operators, NP density operators, NP capillary pressure operator
  const static uint8_t N_OPS = NC + NC * NP + NP + NP;
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t N_PHASE_OPS = 1 /*dens*/ + 1 /*pc*/ + NC /*flux*/;
  // for each phase:
  const static uint8_t DENS_OP = NC + 0;
  const static uint8_t PC_OP = NC + 1;
  const static uint8_t FLUX_OP = NC + 2;

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  uint8_t get_n_vars() const override { return N_VARS; };
  uint8_t get_n_ops() const override { return N_OPS; };
  uint8_t get_n_comps() const override { return NC; };
  uint8_t get_z_var() const override { return Z_VAR; };

  engine_nc_cg_cpu() { engine_name = "Multiphase " + std::to_string(NC) + "-component isothermal flow with gravity and capillarity CPU engine"; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);
  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

public:
};
#endif
