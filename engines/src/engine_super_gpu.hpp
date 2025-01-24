#ifndef D240CA8D_F200_4D4A_9312_42994D1DFBC7
#define D240CA8D_F200_4D4A_9312_42994D1DFBC7

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

template <uint8_t NC, uint8_t NP, bool THERMAL>
class engine_super_gpu : public engine_base_gpu
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

  // number of variables per jacobian matrix block
  const static uint16_t N_VARS_SQ = N_VARS * N_VARS;

  // for some reason destructor is not picked up by recursive instantiator when defined in cu file, so put it here
  ~engine_super_gpu()
  {
    free_device_data(mesh_grav_coef_d);
  }

  uint8_t get_n_vars() const override { return N_VARS; };
  uint8_t get_n_ops() const override { return N_OPS; };
  uint8_t get_n_comps() const override { return NC; };
  uint8_t get_z_var() const override { return Z_VAR; };

  engine_super_gpu()
  {
    if (THERMAL)
    {
      engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component non-isothermal flow with kinetic reaction and diffusion GPU engine";
    }
    else
    {
      engine_name = std::to_string(NP) + "-phase " + std::to_string(NC) + "-component isothermal flow with kinetic reaction and diffusion GPU engine";
    }
  };

  //inline index_t get_z_idx(char c, index_t block_idx) { return block_idx * N_VARS + c + 1; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_) override;

  int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS) override;
  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS) override;

public:
  value_t *RV_d;              // [n_blocks] rock volumes for each block
  value_t *mesh_tranD_d;      // [n_conns] transmissibility and diffusive transmissibility for each (duplicated) connection
  value_t *mesh_hcap_d;       // [n_blocks] rock heat capacity for each block
  value_t *mesh_rcond_d;      // [n_blocks] rock heat conduction for each block
  value_t *mesh_poro_d;       // [n_blocks] porosity for each block
  value_t *mesh_kin_factor_d; // [n_blocks] kin factor for each block
  value_t *mesh_grav_coef_d;  // [n_conns] porosity for each block
};

#include "engine_super_gpu.tpp"
#endif /* D240CA8D_F200_4D4A_9312_42994D1DFBC7 */
