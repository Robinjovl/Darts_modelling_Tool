#ifndef F16383E3_34B5_44CE_A7BF_FB812B8C820C
#define F16383E3_34B5_44CE_A7BF_FB812B8C820C

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
class engine_nce_g_gpu : public engine_base_gpu
{

public:
  // number of components
  const static uint8_t NC_ = NC;
  // number of primary variables : [P, Z_1, ... Z_(NC-1), E]
  const static uint8_t N_VARS = NC + 1;
  // order of primary variables:
  const static uint8_t P_VAR = 0;
  const static uint8_t Z_VAR = 1;
  const static uint8_t E_VAR = NC; // FLUID ENTHALPY
  // number of operators:
  // mass: NC accumulation operators, NC*NP flux operators
  // energy: 1    fluid energy accumulation,
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

  // for some reason destructor is not picked up by recursive instantiator when defined in cu file, so put it here
  ~engine_nce_g_gpu()
  {
    free_device_data(RV_d);
    free_device_data(mesh_hcap_d);
    free_device_data(mesh_poro_d);
    free_device_data(mesh_rcond_d);
    free_device_data(mesh_tranD_d);
    free_device_data(mesh_grav_coef_d);
  }

  uint8_t get_n_vars() const { return N_VARS; };
  uint8_t get_n_ops() const { return N_OPS; };
  uint8_t get_n_comps() const { return NC; };
  uint8_t get_z_var() const { return Z_VAR; };

  engine_nce_g_gpu() { engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component enthalpy-based thermal flow with gravity GPU engine"; };

  //inline index_t get_z_idx(char c, index_t block_idx) { return block_idx * N_VARS + c + 1; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);

  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);

  double calc_newton_residual_L2();
  double calc_newton_residual_Linf();
  double calc_well_residual_L2();
  double calc_well_residual_Linf();

private:
  value_t *RV_d;             // [n_blocks] rock volumes for each block
  value_t *mesh_tranD_d;     // [n_conns] transmissibility and diffusive transmissibility for each (duplicated) connection
  value_t *mesh_hcap_d;      // [n_blocks] rock heat capacity for each block
  value_t *mesh_rcond_d;     // [n_blocks] rock heat conduction for each block
  value_t *mesh_poro_d;      // [n_blocks] porosity for each block
  value_t *mesh_grav_coef_d; // [n_conns] porosity for each block
};
#endif /* F16383E3_34B5_44CE_A7BF_FB812B8C820C */
