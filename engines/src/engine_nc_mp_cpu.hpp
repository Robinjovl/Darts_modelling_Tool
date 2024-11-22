#ifndef CPU_SIMULATOR_NC_MP_HPP
#define CPU_SIMULATOR_NC_MP_HPP

#include <vector>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "globals.h"
#include "ms_well.h"
#include "engine_base.h"
#include "csr_matrix.h"
#include "linsolv_iface.h"
#include "evaluator_iface.h"

template <uint8_t NC>
class engine_nc_mp_cpu : public engine_base
{

public:
  // number of components
  const static uint8_t NC_ = NC;
  // number of primary variables : [P, Z_1, ... Z_(NC-1)]
  const static uint8_t N_VARS = NC;
  // order of primary variables:
  const static uint8_t P_VAR = 0;
  const static uint8_t Z_VAR = 1;
  // number of operators: NC accumulation operators, NC flux operators
  const static uint8_t N_OPS = 2 * NC;
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t FLUX_OP = NC;
  // flag used to support two-point residual assembly for fluxes
  bool TWO_POINT_RES_ASSEMBLY;
  // flag used to use already calculated flux in residual
  bool USE_CALCULATED_FLUX;

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  const uint8_t get_n_vars() override { return N_VARS; };
  const uint8_t get_n_ops() { return N_OPS; };
  const uint8_t get_n_comps() { return NC; };
  const uint8_t get_z_var() { return Z_VAR; };

  engine_nc_mp_cpu() { engine_name = "Multiphase " + std::to_string(NC) + "-component multipoint isothermal flow CPU engine"; };

  //inline index_t get_z_idx(char c, index_t block_idx) { return block_idx * N_VARS + c + 1; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);
  int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                sim_params *params_, timer_node *timer_);
  int init_jacobian_structure_mpfa(csr_matrix_base *jacobian);

  int run_single_newton_iteration(value_t deltat);
  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);
  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

  /// @brief vector of variables in the current timestep provided for operator evaluation
  std::vector<value_t> Xop;
  void extract_Xop();

public:
};
#endif
