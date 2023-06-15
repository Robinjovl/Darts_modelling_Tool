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

#include <algorithm>
#include <time.h>
#include <functional>
#include <string>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iomanip>

#include "engine_nc_cg_cpu.hpp"

template <uint8_t NC, uint8_t NP>
int engine_nc_cg_cpu<NC, NP>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                   std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                   sim_params *params_, timer_node *timer_)
{
  engine_base::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);

  return 0;
}

template <uint8_t NC, uint8_t NP>
int engine_nc_cg_cpu<NC, NP>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  std::vector<value_t> &tran = mesh->tran;
  std::vector<value_t> &grav_coef = mesh->grav_coef;

  value_t *Jac = jacobian->get_values();
  index_t *diag_ind = jacobian->get_diag_ind();
  index_t *rows = jacobian->get_rows_ptr();
  index_t *cols = jacobian->get_cols_ind();
  index_t *row_thread_starts = jacobian->get_row_thread_starts();

  CFL_max = 0;

#ifdef _OPENMP
  //#pragma omp parallel reduction (max: CFL_max)
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    index_t start = row_thread_starts[id];
    index_t end = row_thread_starts[id + 1];

    numa_set(Jac, 0, rows[start] * N_VARS_SQ, rows[end] * N_VARS_SQ);
#else
  index_t start = 0;
  index_t end = n_blocks;
  memset(Jac, 0, rows[end] * N_VARS_SQ * sizeof(value_t));
#endif //_OPENMP

    index_t j, diag_idx, jac_idx;
    value_t p_diff;
    value_t CFL_in[NC], CFL_out[NC];
    value_t CFL_max_local = 0;

    int connected_with_well;

    for (index_t i = start; i < end; ++i)
    {
      // index of diagonal block entry for block i in CSR values array
      diag_idx = N_VARS_SQ * diag_ind[i];

      // fill diagonal part
      for (uint8_t c = 0; c < NC; c++)
      {
        RHS[i * N_VARS + c] = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]); // acc operators only
        CFL_out[c] = 0;
        CFL_in[c] = 0;
        connected_with_well = 0;
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          Jac[diag_idx + c * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];
        }
      }

      // index of first entry for block i in CSR cols array
      index_t csr_idx_start = rows[i];
      // index of last entry for block i in CSR cols array
      index_t csr_idx_end = rows[i + 1];
      // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
      index_t conn_idx = csr_idx_start - i;

      jac_idx = N_VARS_SQ * csr_idx_start;

      // fill offdiagonal part + contribute to diagonal
      for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS_SQ)
      {
        j = cols[csr_idx];
        // skip diagonal
        if (i == j)
        {
          continue;
        }

        p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];

        if (j >= mesh->n_res_blocks)
          connected_with_well = 1;

        for (uint8_t p = 0; p < NP; p++)
        {
          // calculate gravity term for phase p
          value_t avg_density = (op_vals_arr[i * N_OPS + p * N_PHASE_OPS + DENS_OP] +
                                 op_vals_arr[j * N_OPS + p * N_PHASE_OPS + DENS_OP]) /
                                2;

          // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
          value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + p * N_PHASE_OPS + PC_OP] + op_vals_arr[i * N_OPS + p * N_PHASE_OPS + PC_OP];

          // calculate partial derivatives for gravity and capillary terms
          value_t grav_pc_der_i[N_VARS];
          value_t grav_pc_der_j[N_VARS];
          for (uint8_t v = 0; v < N_VARS; v++)
          {
            grav_pc_der_i[v] = -(op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];
            grav_pc_der_j[v] = -(op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];
          }

          double phase_gamma_p_diff = tran[conn_idx] * dt * phase_p_diff;
          if (phase_p_diff < 0)
          {
            //outflow
            for (uint8_t c = 0; c < NC; c++)
            {
              value_t c_flux = tran[conn_idx] * dt * op_vals_arr[i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];
              CFL_out[c] -= phase_p_diff * c_flux;          // subtract negative value of flux
              RHS[i * N_VARS + c] -= phase_p_diff * c_flux; // flux operators only
              for (uint8_t v = 0; v < N_VARS; v++)
              {
                Jac[diag_idx + c * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
                Jac[diag_idx + c * N_VARS + v] += c_flux * grav_pc_der_i[v];
                Jac[jac_idx + c * N_VARS + v] += c_flux * grav_pc_der_j[v];

                if (v == 0)
                {
                  Jac[jac_idx + c * N_VARS + v] -= c_flux;
                  Jac[diag_idx + c * N_VARS + v] += c_flux;
                }
              }
            }
          }
          else
          {
            //inflow

            for (uint8_t c = 0; c < NC; c++)
            {
              value_t c_flux = tran[conn_idx] * dt * op_vals_arr[j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];
              CFL_in[c] += phase_p_diff * c_flux;
              RHS[i * N_VARS + c] -= phase_p_diff * c_flux; // flux operators only
              for (uint8_t v = 0; v < N_VARS; v++)
              {
                Jac[jac_idx + c * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
                Jac[diag_idx + c * N_VARS + v] += c_flux * grav_pc_der_i[v];
                Jac[jac_idx + c * N_VARS + v] += c_flux * grav_pc_der_j[v];
                if (v == 0)
                {
                  Jac[diag_idx + c * N_VARS + v] += c_flux; //-= Jac[jac_idx + c * N_VARS];
                  Jac[jac_idx + c * N_VARS + v] -= c_flux;  // -tran[conn_idx] * dt * op_vals[NC + c];
                }
              }
            }
          }
        }
        conn_idx++;
      }

      // calc CFL for reservoir cells, not connected with wells
      if (i < mesh->n_res_blocks && !connected_with_well)
      {
        for (uint8_t c = 0; c < NC; c++)
        {
          if ((PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]) > 1e-4)
          {
            CFL_max_local = std::max(CFL_max_local, CFL_in[c] / (PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]));
            CFL_max_local = std::max(CFL_max_local, CFL_out[c] / (PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]));
          }
        }
      }
    }
#ifdef _OPENMP
#pragma omp critical
    {
      if (CFL_max < CFL_max_local)
        CFL_max = CFL_max_local;
    }
  } // end of omp parallel
#else
  CFL_max = CFL_max_local;
#endif

  for (ms_well *w : wells)
  {
    value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
    w->add_to_jacobian(dt, X, jac_well_head, RHS);
  }

  return 0;
};


template<uint8_t NC, uint8_t NP>
int
engine_nc_cg_cpu<NC, NP>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};

// 2 phase: from 2 to MAX_NC components
template struct recursive_instantiator_nc_np<engine_nc_cg_cpu, 2, MAX_NC, 2>;

// 3 phase: only for 3 components
template struct recursive_instantiator_nc_np<engine_nc_cg_cpu, 3, 3, 3>;
