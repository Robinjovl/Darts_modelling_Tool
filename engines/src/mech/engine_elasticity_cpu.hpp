#ifndef CPU_SIMULATOR_ELASTICITY_HPP
#define CPU_SIMULATOR_ELASTICITY_HPP

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

template <uint8_t ND>
class engine_elasticity_cpu : public engine_base
{
public:
  // space dimension
  const static uint8_t ND_ = ND;
  // number of primary variables : [P, Z_1, ... Z_(NC-1)]
  const static uint8_t N_VARS = ND;
  // order of primary variables:
  const static uint8_t U_VAR = 0;
  // number of operators: NC accumulation operators, NC flux operators
  const static uint8_t N_OPS = ND_;
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t FLUX_OP = 0;

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  uint8_t get_n_vars() const { return N_VARS; };
  uint8_t get_n_ops() const { return N_OPS; };
  uint8_t get_n_dim() const { return ND_; };
  uint8_t get_n_comps() const { return 0; };
  uint8_t get_z_var() const { return -1; };
  bool USE_CALCULATED_FLUX;

  engine_elasticity_cpu() { engine_name = std::to_string(ND) + "D elastic mechanics CPU engine"; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);
  int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                sim_params *params_, timer_node *timer_);

  int assemble_linear_system(value_t deltat);
  int init_jacobian_structure_mpsa(csr_matrix_base *jacobian);
  int solve_linear_equation();
  double calc_newton_residual_L2();
  void write_matrix();
  int apply_newton_update(value_t dt);
  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);
  index_t output_counter;
  std::vector<value_t> fluxes;

  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

public:
};
#endif /* CPU_SIMULATOR_ELASTICITY_HPP */
