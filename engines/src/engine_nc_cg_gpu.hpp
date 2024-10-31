#ifndef ENGINE_NC_CG_GPU_HPP
#define ENGINE_NC_CG_GPU_HPP

#include <vector>
#include <array>
#include <fstream>
#include <iostream>

#include "globals.h"
#include "ms_well.h"
#include "engine_base_gpu.h"
#include "csr_matrix.h"
#include "linsolv_iface.h"
#include "evaluator_iface.h"

template <uint8_t NC, uint8_t NP>
class engine_nc_cg_gpu : public engine_base_gpu
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

  // for some reason destructor is not picked up by recursive instantiator when defined in cu file, so put it here
  ~engine_nc_cg_gpu()
  {
    free_device_data(mesh_grav_coef_d);
  }

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  uint8_t get_n_vars() const { return N_VARS; };
  uint8_t get_n_ops() const { return N_OPS; };
  uint8_t get_n_comps() const { return NC; };
  uint8_t get_z_var() const { return Z_VAR; };

  engine_nc_cg_gpu() { engine_name = "Multiphase " + std::to_string(NC) + "-component isothermal flow with gravity and capillarity GPU engine"; };

  //inline index_t get_z_idx(char c, index_t block_idx) { return block_idx * N_VARS + c + 1; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);

  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

  // calc r_d = Jacobian * v_d
  virtual int matrix_vector_product_d0(const value_t *v_d, value_t *r_d);

  // calc r_d = alpha * Jacobian * u_d + beta * v_d
  virtual int calc_lin_comb_d(value_t alpha, value_t beta, value_t *u_d, value_t *v_d, value_t *r_d);

public:
  value_t *mesh_grav_coef_d; // [n_conns] gravity coefficient for each block
};
#endif
