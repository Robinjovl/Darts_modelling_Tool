//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

#ifndef CPU_SIMULATOR_PM_HPP
#define CPU_SIMULATOR_PM_HPP

#include <vector>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>

#include "globals.h"
#include "ms_well.h"
#include "engine_base.h"
#include "evaluator_iface.h"
#include "mech/contact.h"

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

class engine_pm_cpu : public engine_base
{

public:
  // space dimension
  const static uint8_t ND_ = 3;
  // number of components
  const static uint8_t NC_ = 1;
  const static uint8_t NT_ = 4;
  // number of primary variables : [P, Z_1, ... Z_(NC-1)]
  const static uint8_t N_VARS = NC_ + ND_;
  // order of primary variables:
  const static uint8_t U_VAR = 0;
  const static uint8_t P_VAR = 3;
  const static uint8_t Z_VAR = 255;
  // number of operators: NC accumulation operators, NC flux operators
  const static uint8_t N_OPS = 2 * NC_;
  // order of operators:
  const static uint8_t ACC_OP = 0;
  const static uint8_t FLUX_OP = NC_;
  const static uint8_t GRAV_OP = 0;
  // coefficient to fit units
  const static value_t BAR_DAY2_TO_PA_S2;

  // IMPORTANT: all constants above have to be in agreement with acc_flux_op_set

  // number of variables per jacobian matrix block
  const static uint8_t N_VARS_SQ = N_VARS * N_VARS;

  const uint8_t get_n_vars() override { return N_VARS; };
  const uint8_t get_n_ops() { return N_OPS; };
  const uint8_t get_n_dim() { return ND_; };
  const uint8_t get_n_comps() { return NC_; };
  const uint8_t get_z_var() { return Z_VAR; };

  engine_pm_cpu() { engine_name = "Single phase " + std::to_string(NC_) + "-component isothermal poromechanics CPU engine"; };

  //inline index_t get_z_idx(char c, index_t block_idx) { return block_idx * N_VARS + c + 1; };

  int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
           std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
           sim_params *params_, timer_node *timer_);

  int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                sim_params *params_, timer_node *timer_);

  int init_jacobian_structure_pm(csr_matrix_base *jacobian);

  int assemble_jacobian_array(value_t _dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);
  int assemble_jacobian_array_time_dependent_discr(value_t _dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS);
  void update_uu_jacobian();

  int solve_linear_equation();
  void apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
  int run_single_newton_iteration(value_t deltat);
  int post_newtonloop(value_t deltat, value_t time);
  // fluxes at current and previous time steps, fluxes for reference state at current and previous time steps
  std::vector<value_t> fluxes, fluxes_n, fluxes_biot, fluxes_biot_n, fluxes_ref, fluxes_biot_ref, fluxes_ref_n, fluxes_biot_ref_n;
  std::vector<value_t> Xref, Xn_ref;
  std::vector<value_t> eps_vol;
  // inertia for momentum
  std::vector<value_t> Xn1;
  value_t dt1;
  value_t momentum_inertia;

  // maximum absolute values in rows of jacobian
  std::vector<value_t> max_row_values;

  /// @brief vector of variables in the current timestep provided for operator evaluation
  std::vector<value_t> Xop;
  void extract_Xop();
  std::vector<value_t> calc_newton_dev_L2();
  double calc_well_residual_L2();
  std::vector<value_t> calc_newton_dev();
  int apply_newton_update(value_t dt);
  value_t dev_u, dev_p, dev_g, dev_u_prev, dev_p_prev, dev_g_prev, well_residual_prev_dt;
  int output_counter;
  value_t newton_update_coefficient;

  std::vector<pm::contact> contacts;
  std::vector<index_t> geomechanics_mode;
  
  int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS);


public:
  bool FIND_EQUILIBRIUM, PRINT_LINEAR_SYSTEM, TIME_DEPENDENT_DISCRETIZATION, SCALE_ROWS;
  pm::ContactSolver contact_solver;
};
#endif /* CPU_SIMULATOR_PM_HPP */
