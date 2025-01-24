#ifndef ENGINE_NC_GPU_HPP
#define ENGINE_NC_GPU_HPP

#include <vector>
#include <array>
#include <fstream>
#include <iostream>

#include "globals.h"
#include "ms_well.h"
#include "engine_base_gpu.h"
#include "linsolv_iface.h"
#include "evaluator_iface.h"

template <uint8_t NC>
class engine_nc_gpu : public engine_base_gpu
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

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  const uint8_t get_n_vars() override { return N_VARS; };
  const uint8_t get_n_ops() { return N_OPS; };
  const uint8_t get_n_comps() { return NC; };
  const uint8_t get_z_var() { return Z_VAR; };

  engine_nc_gpu() { engine_name = "Multiphase " + std::to_string(NC) + "-component isothermal flow GPU engine"; };

  //inline index_t get_z_idx(char c, index_t block_idx) { return block_idx * N_VARS + c + 1; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);

  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

  int test_spmv(int n_times, int kernel_number = 0, int dump_result = 0);

  // matrix-free methods

  // calc r_d = Jacobian * v_d
  virtual int matrix_vector_product_d0(const value_t *v_d, value_t *r_d);

  // calc r_d = alpha * Jacobian * u_d + beta * v_d
  virtual int calc_lin_comb_d(value_t alpha, value_t beta, value_t *u_d, value_t *v_d, value_t *r_d);

public:
};
#endif
