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

#ifndef GPU_SIMULATOR_CU_H
#define GPU_SIMULATOR_CU_H

#include <vector>

#include "globals.h"
#include "csr_matrix.h"
// bos linear solvers
#include "linear_solvers.h"
// ugatu linear solvers
#include "decl.h"
#include "Solver.h"
#include "mmio.h"
#include "engine_base.h"

class conn_mesh;
class interp_table;
class sim_params;


class gpu_simulator : public engine_base
{

  public:
  gpu_simulator () {};

  int init (conn_mesh *_mesh, std::string table_base_name);

  int run (sim_params *params);

  int assemble_jacobian (value_t dt, int is_first);

  int gpu_test (int argc, char** argv);

  public:
  //////////////////////////////////
  // CPU properties
  //////////////////////////////////

  conn_mesh *mesh;
  sim_params params;
  linear_solver_base *cpu_solver;
  linear_solver_base *cpu_preconditioner;

  LinearSolver* gpu_solver;
  Preconditioner* gpu_preconditioner;


  interp_table *acc1, *acc2;
  interp_table *flu1, *flu2;

  csr_matrix Jacobian;        // [n_blocks * n_blocks] Jacobian matrix in 2*2 Block CSR format
  std::vector<value_t> RHS;   // [2 * n_blocks] right hand side for linear system

  std::vector<value_t> X;     // [2 * n_blocks] array of current timestep solution 
  std::vector<value_t> Xn;    // [2 * n_blocks] array of previous timestep solution 
  std::vector<value_t> dX;    // [2 * n_blocks] array of linear solution 

  float gpu_assemble_timer;


  //////////////////////////////////
  // GPU properties
  //////////////////////////////////



  value_t *gpu_acc1_data;
  interp_value_t *gpu_acc1_res;

  value_t *gpu_acc2_data;
  interp_value_t *gpu_acc2_res;

  value_t *gpu_flu1_data;
  interp_value_t *gpu_flu1_res;

  value_t *gpu_flu2_data;
  interp_value_t *gpu_flu2_res;

  index_t* gpu_block_m;
  index_t* gpu_block_p;
  value_t* gpu_tran;

  value_t* gpu_PV;

  value_t* gpu_jac_values;
  index_t* gpu_jac_rows_ptr;
  index_t* gpu_jac_cols_ind;

  value_t* gpu_jac_values_ilu;

  value_t* gpu_rhs;
  value_t* gpu_x;
  value_t* gpu_xn;
  value_t* gpu_dx;


};
#endif

