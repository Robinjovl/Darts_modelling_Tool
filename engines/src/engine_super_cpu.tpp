#include <algorithm>
#include <time.h>
#include <functional>
#include <string>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "engine_super_cpu.hpp"
#include "conn_mesh.h"

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/linsolv_bos_gmres.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_bilu0.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_cpr.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_amg.hpp"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"
#else
#include "linsolv_bos_gmres.h"
#include "linsolv_bos_bilu0.h"
#include "linsolv_bos_cpr.h"
#include "linsolv_bos_amg.h"
#include "linsolv_amg1r5.h" // Not available in opendarts_linear_solvers
#include "linsolv_superlu.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::auxiliary;
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS    

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_cpu<NC, NP, THERMAL>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                            std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                            sim_params *params_, timer_node *timer_)
{
  // prepare dg_dx_n_temp for adjoint method
  if (opt_history_matching)
  {

    if (!dg_dx_n_temp)
    {
      dg_dx_n_temp = new csr_matrix<N_VARS>;
      dg_dx_n_temp->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
    }

      // allocate Adjoint matrices
      (static_cast<csr_matrix<N_VARS>*>(dg_dx_n_temp))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);
  }

  engine_base::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);
  this->expose_jacobian();

  return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_cpu<NC, NP, THERMAL>::enable_flux_output()
{
  enabled_flux_output = true;

  if (darcy_fluxes.empty())
  {
    // mass fluxes
    darcy_fluxes.resize(NC * NP * mesh->n_conns);
    diffusion_fluxes.resize(NP * NC * mesh->n_conns);

    // energy fluxes
    if (THERMAL)
    {
      heat_darcy_advection_fluxes.resize(NP * mesh->n_conns);
      heat_diffusion_advection_fluxes.resize(NP * NC * mesh->n_conns);
      fourier_fluxes.resize((NP + 1) * mesh->n_conns);
    }
  }
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_cpu<NC, NP, THERMAL>::assemble_jacobian_array(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
    index_t n_blocks = mesh->n_blocks;
    index_t n_res_blocks = mesh->n_res_blocks;
    index_t n_conns = mesh->n_conns;
    const std::vector<value_t>& tran = mesh->tran;
    const std::vector<value_t>& tranD = mesh->tranD;
    const std::vector<value_t>& hcap = mesh->heat_capacity;
    const std::vector<value_t>& kin_fac = mesh->kin_factor; // default value of 1
    const std::vector<value_t>& grav_coef = mesh->grav_coef;
    const std::vector<value_t>& velocity_appr = mesh->velocity_appr;
    const std::vector<index_t>& velocity_offset = mesh->velocity_offset;
    const std::vector<index_t>& op_num = mesh->op_num;

    value_t* Jac = jacobian->get_values();
    index_t* diag_ind = jacobian->get_diag_ind();
    index_t* rows = jacobian->get_rows_ptr();
    index_t* cols = jacobian->get_cols_ind();
    index_t* row_thread_starts = jacobian->get_row_thread_starts();

    // for reconstruction of phase velocities
    if (!mesh->velocity_appr.empty() && darcy_velocities.empty())
        darcy_velocities.resize(n_res_blocks * NP * ND);

    if (!mesh->velocity_appr.empty() && !dispersivity.empty())
    {
      if (dispersion_fluxes.empty())
        dispersion_fluxes.resize(NP * NC * mesh->n_conns);

      if (THERMAL && heat_dispersion_advection_fluxes.empty())
        heat_dispersion_advection_fluxes.resize(NP * NC * mesh->n_conns);
    }

    std::fill(darcy_velocities.begin(), darcy_velocities.end(), 0.0);

    // if velocity reconstruction enabled, also provide molar weights
    if (!mesh->velocity_appr.empty())
    {
      if (molar_weights.empty())
      {
        printf("Velocity reconstruction is enabled. Provide molar weights!");
        exit(-1);
      }
    }

    // fill fourier_fluxes with zeros
    std::fill(fourier_fluxes.begin(), fourier_fluxes.end(), 0.0);

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
    value_t p_diff, gamma_p_diff, t_diff, gamma_t_i, gamma_t_j, phi_i, phi_j, phi_avg, phi_0_avg;
    value_t CFL_in[NC], CFL_out[NC];
    value_t CFL_max_local = 0;
    value_t phase_presence_mult;
    index_t cell_conn_idx, cell_conn_num;
    std::array<value_t, NP> phase_fluxes;
    
    // fluxes for output
    value_t *cur_darcy_fluxes, *cur_diffusion_fluxes, *cur_dispersion_fluxes;
    value_t *cur_heat_darcy_advection_fluxes, *cur_heat_diffusion_advection_fluxes, 
            *cur_heat_dispersion_advection_fluxes, *cur_fourier_fluxes;

    int connected_with_well;

    for (index_t i = start; i < end; ++i)
    { // loop over grid blocks

      // initialize the CFL_in and CFL_out
        for (uint8_t c = 0; c < NC; c++)
        {
            CFL_out[c] = 0;
            CFL_in[c] = 0;
            connected_with_well = 0;
        }

        // index of diagonal block entry for block i in CSR values array
        diag_idx = N_VARS_SQ * diag_ind[i];

        // [1] fill diagonal part for both mass (and energy equations if needed, only fluid energy is involved here)
        for (uint8_t c = 0; c < NE; c++)
        {
            RHS[i * N_VARS + c] = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]); // acc operators only

            // Add reaction term to diagonal of reservoir cells (here the volume is pore volume or block volume):
            if (i < n_res_blocks)
                RHS[i * N_VARS + c] += (PV[i] + RV[i]) * dt * op_vals_arr[i * N_OPS + KIN_OP + c] * kin_fac[i]; // kinetics

            for (uint8_t v = 0; v < N_VARS; v++)
            {
                Jac[diag_idx + c * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v]; // der of accumulation term

                // Include derivatives for reaction term if part of reservoir cells:
                if (i < n_res_blocks)
                {
                    Jac[diag_idx + c * N_VARS + v] += (PV[i] + RV[i]) * dt * op_ders_arr[(i * N_OPS + KIN_OP + c) * N_VARS + v] * kin_fac[i]; // derivative kinetics
                }
            }
        }

        // index of first entry for block i in CSR cols array
        index_t csr_idx_start = rows[i];
        // index of last entry for block i in CSR cols array
        index_t csr_idx_end = rows[i + 1];
        // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
        index_t conn_idx = csr_idx_start - i;

        jac_idx = N_VARS_SQ * csr_idx_start;

        // for velocity reconstruction
        if (!velocity_offset.empty() && i < n_res_blocks)
            cell_conn_num = velocity_offset[i + 1] - velocity_offset[i];

        cell_conn_idx = 0;
        for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS_SQ)
        { // fill offdiagonal part + contribute to diagonal

            j = cols[csr_idx];
            // skip diagonal
            if (i == j)
                continue;

            // fluxes for current connection
            if (enabled_flux_output)
            {
              cur_darcy_fluxes = &darcy_fluxes[NP * NC * conn_idx];
              cur_diffusion_fluxes = &diffusion_fluxes[NP * NC * conn_idx];
              if constexpr (THERMAL)
              {
                cur_heat_darcy_advection_fluxes = &heat_darcy_advection_fluxes[NP * conn_idx];
                cur_heat_diffusion_advection_fluxes = &heat_diffusion_advection_fluxes[NP * NC * conn_idx];
                cur_fourier_fluxes = &fourier_fluxes[(NP + 1) * conn_idx];
              }
            }

            value_t trans_mult = 1;
            value_t trans_mult_der_i[N_VARS];
            value_t trans_mult_der_j[N_VARS];
            if (params->trans_mult_exp > 0 && i < n_res_blocks && j < n_res_blocks)
            {
                // Calculate transmissibility multiplier:
                phi_i = op_vals_arr[i * N_OPS + PORO_OP];
                phi_j = op_vals_arr[j * N_OPS + PORO_OP];

                // Take average interface porosity:
                phi_avg = (phi_i + phi_j) * 0.5;
                phi_0_avg = (mesh->poro[i] + mesh->poro[j]) * 0.5;

                trans_mult = params->trans_mult_exp * pow(phi_avg, params->trans_mult_exp - 1) * 0.5;
                for (uint8_t v = 0; v < N_VARS; v++)
                {
                    trans_mult_der_i[v] = trans_mult * op_ders_arr[(i * N_OPS + PORO_OP) * N_VARS + v];
                    trans_mult_der_j[v] = trans_mult * op_ders_arr[(j * N_OPS + PORO_OP) * N_VARS + v];
                }
                trans_mult = pow(phi_avg, params->trans_mult_exp);
            }
            else
            {
                for (uint8_t v = 0; v < N_VARS; v++)
                {
                    trans_mult_der_i[v] = 0;
                    trans_mult_der_j[v] = 0;
                }
            }

            p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];

            if (j >= n_res_blocks)
                connected_with_well = 1;

            // [2] fill offdiagonal part + contribute to diagonal, only fluid part is considered in energy equation
            for (uint8_t p = 0; p < NP; p++)
            { // loop over number of phases for convective operator

              // calculate gravity term for phase p
                value_t avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] +
                    op_vals_arr[j * N_OPS + GRAV_OP + p]) /
                    2;

                // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
                value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

                // calculate partial derivatives for gravity and capillary terms
                value_t grav_pc_der_i[N_VARS];
                value_t grav_pc_der_j[N_VARS];
                for (uint8_t v = 0; v < N_VARS; v++)
                {
                    grav_pc_der_i[v] = -(op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + PC_OP + p) * N_VARS + v];
                    grav_pc_der_j[v] = -(op_ders_arr[(j * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + PC_OP + p) * N_VARS + v];
                }

                phase_fluxes[p] = 0.0;

                double phase_gamma_p_diff = trans_mult * tran[conn_idx] * dt * phase_p_diff;

                if (phase_p_diff < 0)
                {
                    // mass and energy outflow with effect of gravity and capillarity
                    for (uint8_t c = 0; c < NE; c++)
                    {
                        value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c];

                        if (c < NC)
                        {
                            CFL_out[c] -= phase_p_diff * c_flux; // subtract negative value of flux
                            if (!molar_weights.empty())
                            phase_fluxes[p] += op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c] * molar_weights[NC * op_num[i] + c];
                            if (enabled_flux_output) cur_darcy_fluxes[p * NC + c] = -phase_p_diff * c_flux / dt;
                        }
                        else
                          if (enabled_flux_output) cur_heat_darcy_advection_fluxes[p] = -phase_p_diff * c_flux / dt;

                        RHS[i * N_VARS + c] -= phase_p_diff * c_flux; // flux operators 
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            Jac[diag_idx + c * N_VARS + v] -= (phase_gamma_p_diff * op_ders_arr[(i * N_OPS + FLUX_OP + p * NE + c) * N_VARS + v] +
                                tran[conn_idx] * dt * phase_p_diff * trans_mult_der_i[v] * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c]);
                            Jac[diag_idx + c * N_VARS + v] += c_flux * grav_pc_der_i[v];
                            Jac[jac_idx + c * N_VARS + v] += c_flux * grav_pc_der_j[v];

                            if (v == 0)
                            {
                                Jac[jac_idx + c * N_VARS + v] -= c_flux;
                                Jac[diag_idx + c * N_VARS + v] += c_flux;
                            }
                        }
                    }
                    if (phase_fluxes[p] != 0.0)
                        phase_fluxes[p] *= -trans_mult * tran[conn_idx] * phase_p_diff / op_vals_arr[i * N_OPS + GRAV_OP + p];
                }
                else
                {
                    // mass and energy inflow with effect of gravity and capillarity
                    for (uint8_t c = 0; c < NE; c++)
                    {
                        value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c];

                        if (c < NC)
                        {
                          CFL_in[c] += phase_p_diff * c_flux;
                          if (!molar_weights.empty())
                            phase_fluxes[p] += op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c] * molar_weights[NC * op_num[j] + c];
                          if (enabled_flux_output) cur_darcy_fluxes[p * NC + c] = -phase_p_diff * c_flux / dt;
                        }
                        else
                          if (enabled_flux_output) cur_heat_darcy_advection_fluxes[p] = -phase_p_diff * c_flux / dt;

                        RHS[i * N_VARS + c] -= phase_p_diff * c_flux; // flux operators only
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            Jac[jac_idx + c * N_VARS + v] -= (phase_gamma_p_diff * op_ders_arr[(j * N_OPS + FLUX_OP + p * NE + c) * N_VARS + v] +
                                tran[conn_idx] * dt * phase_p_diff * trans_mult_der_j[v] * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c]);
                            Jac[diag_idx + c * N_VARS + v] += c_flux * grav_pc_der_i[v];
                            Jac[jac_idx + c * N_VARS + v] += c_flux * grav_pc_der_j[v];
                            if (v == 0)
                            {
                                Jac[diag_idx + c * N_VARS + v] += c_flux; //-= Jac[jac_idx + c * N_VARS];
                                Jac[jac_idx + c * N_VARS + v] -= c_flux;  // -tran[conn_idx] * dt * op_vals[NC + c];
                            }
                        }
                    }
                    if (phase_fluxes[p] != 0.0)
                        phase_fluxes[p] *= -trans_mult * tran[conn_idx] * phase_p_diff / op_vals_arr[j * N_OPS + GRAV_OP + p];
                }
            } // end of loop over number of phases for convective operator with gravity and capillarity

            // [3] Additional diffusion code here:   (phi_p * S_p) * (rho_p * D_cp * Delta_x_cp)  or (phi_p * S_p) * (kappa_p * Delta_T)
            // Only if block connection is between reservoir and reservoir cells!
            if (i < n_res_blocks && j < n_res_blocks)
            {
                // Add diffusion term to the residual:
                for (uint8_t c = 0; c < NE; c++)
                {
                    for (uint8_t p = 0; p < NP; p++)
                    {
                        value_t grad_con = op_vals_arr[j * N_OPS + GRAD_OP + p * NE + c] - op_vals_arr[i * N_OPS + GRAD_OP + p * NE + c];

                        if (op_vals_arr[i * N_OPS + UPSAT_OP + p] * op_vals_arr[j * N_OPS + UPSAT_OP + p] > params->phase_existence_tolerance)
                          phase_presence_mult = 1.0;
                        else
                          phase_presence_mult = 0.0;

                        value_t diff_mob_ups_m = dt * phase_presence_mult * mesh->tranD[conn_idx] * (mesh->poro[i] * op_vals_arr[i * N_OPS + UPSAT_OP + p] +
                                                                                                    mesh->poro[j] * op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;

                        RHS[i * N_VARS + c] -= diff_mob_ups_m * grad_con; // diffusion term
                        if (enabled_flux_output)
                        {
                          if (c < NC)
                            cur_diffusion_fluxes[p * NC + c] = -diff_mob_ups_m * grad_con / dt;
                          else
                            cur_fourier_fluxes[p] = -diff_mob_ups_m * grad_con / dt;
                        }

                        // Add diffusion terms to Jacobian:
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            Jac[diag_idx + c * N_VARS + v] += diff_mob_ups_m * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                            Jac[jac_idx + c * N_VARS + v] -= diff_mob_ups_m * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

                            Jac[diag_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[i] * op_ders_arr[(i * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;
                            Jac[jac_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[j] * op_ders_arr[(j * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;
                        }
                        if (is_fickian_energy_transport_on)
                        {
                          // respective heat flux
                          if constexpr (THERMAL)
                          {
                            if (c < NC)
                            {
                              value_t avg_enthalpy = (op_vals_arr[i * N_OPS + ENTH_OP + p] + op_vals_arr[j * N_OPS + ENTH_OP + p]) / 2.;
                              RHS[i * N_VARS + NC] -= avg_enthalpy * diff_mob_ups_m * grad_con;
                              if (enabled_flux_output) cur_heat_diffusion_advection_fluxes[p * NC + c] = -avg_enthalpy * diff_mob_ups_m * grad_con / dt;

                              for (uint8_t v = 0; v < N_VARS; v++)
                              {
                                Jac[diag_idx + NC * N_VARS + v] += avg_enthalpy * diff_mob_ups_m * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                                Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * diff_mob_ups_m * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

                                Jac[diag_idx + NC * N_VARS + v] -= avg_enthalpy * grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[i] * op_ders_arr[(i * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;
                                Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[j] * op_ders_arr[(j * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;

                                Jac[diag_idx + NC * N_VARS + v] -= op_ders_arr[(i * N_OPS + ENTH_OP + p) * N_VARS + v] * diff_mob_ups_m * grad_con / 2;
                                Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + ENTH_OP + p) * N_VARS + v] * diff_mob_ups_m * grad_con / 2;
                              }
                            }
                          }
                        }
                    }
                }

                // assemble velocities, dispersion is assembled in a separate loop as it requires multiple velocities
                if (!velocity_appr.empty())
                {
                    index_t vel_idx = ND * velocity_offset[i];
                    for (uint8_t p = 0; p < NP; p++)
                    {
                        for (uint8_t d = 0; d < ND; d++)
                            darcy_velocities[NP * ND * i + p * ND + d] += velocity_appr[vel_idx + d * cell_conn_num + cell_conn_idx] * phase_fluxes[p];
                    }
                }
            }

        // [4] add rock conduction
        if (THERMAL)
        {
            t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
            gamma_t_i = tranD[conn_idx] * dt * (1 - mesh->poro[i]) * mesh->rock_cond[i];
            gamma_t_j = tranD[conn_idx] * dt * (1 - mesh->poro[j]) * mesh->rock_cond[j];

            // rock heat transfers flows from cell i to j
            RHS[i * N_VARS + NC] -= t_diff * (gamma_t_i + gamma_t_j) / 2;
            if (enabled_flux_output) cur_fourier_fluxes[NP] = -t_diff * (gamma_t_i + gamma_t_j) / 2 / dt;
            for (uint8_t v = 0; v < N_VARS; v++)
            {
              Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + TEMP_OP) * N_VARS + v] * (gamma_t_i + gamma_t_j) / 2;
              Jac[diag_idx + NC * N_VARS + v] += op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * (gamma_t_i + gamma_t_j) / 2;
            }
        }
        conn_idx++;
        if (j < n_res_blocks)
            cell_conn_idx++;

      }

      // [5] finally add rock energy
      // + rock energy (no rock compressibility included in these computations)
      if (THERMAL)
      {
        RHS[i * N_VARS + NC] += RV[i] * (op_vals_arr[i * N_OPS + TEMP_OP] - op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];

        for (uint8_t v = 0; v < N_VARS; v++)
        {
          Jac[diag_idx + NC * N_VARS + v] += RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i];
        } // end of fill offdiagonal part + contribute to diagonal
      }

        // calc CFL for reservoir cells, not connected with wells
        if (i < n_res_blocks && !connected_with_well)
        {
            for (uint8_t c = 0; c < NC; c++)
            {
              double denominator = PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c];
              if (denominator != 0.0)
              {
                CFL_max_local = std::max(CFL_max_local, CFL_in[c] / denominator);
                CFL_max_local = std::max(CFL_max_local, CFL_out[c] / denominator);
              }
            }
        }

    } // end of loop over grid blocks
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

    // dispersion
    if (!velocity_appr.empty() && !dispersivity.empty())
    {
        value_t avg_dispersivity;
        value_t *cur_dispersion_fluxes, *cur_heat_dispersion_advection_fluxes;
        std::array<value_t, ND> avg_velocity;

        for (index_t i = 0; i < n_res_blocks; ++i)
        { // loop over grid blocks
            if (i > n_res_blocks) // skip wells
                continue;
            // index of diagonal block entry for block i in CSR values array
            index_t diag_idx = N_VARS_SQ * diag_ind[i];
            // index of first entry for block i in CSR cols array
            index_t csr_idx_start = rows[i];
            // index of last entry for block i in CSR cols array
            index_t csr_idx_end = rows[i + 1];
            // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
            index_t conn_idx = csr_idx_start - i;

            index_t jac_idx = N_VARS_SQ * csr_idx_start;

            for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS_SQ)
            {
                index_t j = cols[csr_idx];

                if (i == j)
                  continue;

                if (enabled_flux_output)
                {
                  cur_dispersion_fluxes = &dispersion_fluxes[NP * NC * conn_idx];
                  if constexpr (THERMAL)
                    cur_heat_dispersion_advection_fluxes = &heat_dispersion_advection_fluxes[NP * NC * conn_idx];
                }

                if (j < n_res_blocks)
                {
                    for (uint8_t p = 0; p < NP; p++)
                    {
                        value_t avg_enthalpy = (op_vals_arr[i * N_OPS + ENTH_OP + p] + op_vals_arr[j * N_OPS + ENTH_OP + p]) / 2.;

                        // approximate facial velocity
                        index_t vel_idx_i = ND * NP * i;
                        index_t vel_idx_j = ND * NP * j;
                        for (uint8_t d = 0; d < ND; d++)
                            avg_velocity[d] = (darcy_velocities[vel_idx_i + p * ND + d] + darcy_velocities[vel_idx_j + p * ND + d]) / 2.0;

                        for (uint8_t c = 0; c < NC; c++)
                        {
                            value_t grad_con = op_vals_arr[j * N_OPS + GRAD_OP + p * NE + c] - op_vals_arr[i * N_OPS + GRAD_OP + p * NE + c];

                            // Diffusion flows from cell i to j (high to low), use upstream quantity from cell i for compressibility and saturation (mass or energy):
                            value_t vel_norm = sqrt(avg_velocity[0] * avg_velocity[0] +
                                avg_velocity[1] * avg_velocity[1] +
                                avg_velocity[2] * avg_velocity[2]);

                            value_t arith_mean_dispersivity = (dispersivity[NP * NC * op_num[i] + p * NC + c] + dispersivity[NP * NC * op_num[j] + p * NC + c]) / 2.;
                            if (arith_mean_dispersivity > 0.)
                                avg_dispersivity = dispersivity[NP * NC * op_num[i] + p * NC + c] * dispersivity[NP * NC * op_num[j] + p * NC + c] / arith_mean_dispersivity;
                            else
                                avg_dispersivity = 0.0;

                            value_t disp = dt * avg_dispersivity * mesh->tranD[conn_idx] * vel_norm;

                            RHS[i * N_VARS + c] -= disp * grad_con; // diffusion term
                            if (enabled_flux_output) cur_dispersion_fluxes[p * NC + c] = -disp * grad_con;

                            // Add diffusion terms to Jacobian:
                            for (uint8_t v = 0; v < N_VARS; v++)
                            {
                                Jac[diag_idx + c * N_VARS + v] += disp * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                                Jac[jac_idx + c * N_VARS + v] -= disp * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                            }

                            // respective heat fluxes
                            if (is_fickian_energy_transport_on)
                            {
                              if constexpr (THERMAL)
                              {
                                RHS[i * N_VARS + NC] -= avg_enthalpy * disp * grad_con;
                                if (enabled_flux_output) cur_heat_dispersion_advection_fluxes[p * NC + c] = -avg_enthalpy * disp * grad_con;

                                for (uint8_t v = 0; v < N_VARS; v++)
                                {
                                  Jac[diag_idx + NC * N_VARS + v] += avg_enthalpy * disp * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                                  Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * disp * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

                                  Jac[diag_idx + NC * N_VARS + v] -= op_ders_arr[(i * N_OPS + ENTH_OP + p) * N_VARS + v] * disp * grad_con / 2;
                                  Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + ENTH_OP + p) * N_VARS + v] * disp * grad_con / 2;
                                }
                              }
                            }
                        }
                    }
                }
                conn_idx++;
            }
        }
    }

  for (ms_well *w : wells)
  {
    value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
    w->add_to_jacobian(dt, X, jac_well_head, RHS);
  }

  return 0;
};

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_cpu<NC, NP, THERMAL>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  std::vector<value_t> &tran = mesh->tran;
  std::vector<value_t> &tranD = mesh->tranD;
  std::vector<value_t> &hcap = mesh->heat_capacity;
  std::vector<value_t> &kin_fac = mesh->kin_factor; // default value of 1
  std::vector<value_t> &grav_coef = mesh->grav_coef;
  std::vector <index_t>& conn_index_to_one_way = mesh->conn_index_to_one_way;


  value_t* Jac = Jacobian->get_values();
  index_t* diag_ind = Jacobian->get_diag_ind();
  index_t* rows = Jacobian->get_rows_ptr();
  index_t* cols = Jacobian->get_cols_ind();
  index_t* row_thread_starts = Jacobian->get_row_thread_starts();

  value_t* ad_values = dg_dx_T->get_values();
  index_t* ad_rows = dg_dx_T->get_rows_ptr();
  index_t* ad_cols = dg_dx_T->get_cols_ind();
  index_t* ad_diag = dg_dx_T->get_diag_ind();
  index_t* row_T_thread_starts = dg_dx_T->get_row_thread_starts();

  value_t* Jac_n = dg_dx_n_temp->get_values();
  //value_t* v_g_T = dg_dT->get_values();
  value_t* value_dg_dT = dg_dT_general->get_values();
  well_head_tran_idx_collection.clear();

  CFL_max = 0;

  //#ifdef _OPENMP
  //  //#pragma omp parallel reduction (max: CFL_max)
  //#pragma omp parallel
  //  {
  //    int id = omp_get_thread_num();
  //
  //    //index_t start = row_thread_starts[id];
  //    //index_t end = row_thread_starts[id + 1];
  //
  //    //index_t start = row_T_thread_starts[id];
  //    //index_t end = row_T_thread_starts[id + 1];
  //#else
  //  index_t start = 0;
  //  index_t end = n_blocks;
  //
  //#endif //_OPENMP

  index_t start = 0;
  index_t end = n_blocks;

  index_t j, diag_idx, jac_idx;
  value_t p_diff, gamma_p_diff, t_diff, gamma_t_diff, phi_i, phi_j, phi_avg, phi_0_avg;

  memset(Jac_n, 0, (n_conns + n_blocks) * N_VARS_SQ * sizeof(value_t));
  memset(value_dg_dT, 0, n_conns * N_VARS * sizeof(value_t));


  double value_g_u = 0.0;
  index_t N_element = 0;
  index_t count = 0;

  index_t k_count = 0;
  index_t idx;
  std::vector<index_t> temp_conn_one_way;
  std::vector<index_t> temp_num;


  for (index_t i = start; i < end; ++i)
  { // loop over grid blocks

    // index of diagonal block entry for block i in CSR values array
    diag_idx = N_VARS_SQ * diag_ind[i];

    // [1] fill diagonal part for both mass (and energy equations if needed, only fluid energy is involved here)
    for (uint8_t c = 0; c < NE; c++)
    {
      for (uint8_t v = 0; v < N_VARS; v++)
      {
        Jac_n[diag_idx + c * N_VARS + v] = -(PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v]); // der of accumulation term

        // Include derivatives for reaction term if part of reservoir cells:
        if (i < mesh->n_res_blocks)
        {
          Jac_n[diag_idx + c * N_VARS + v] -= ((PV[i] + RV[i]) * dt * op_ders_arr[(i * N_OPS + KIN_OP + c) * N_VARS + v] * kin_fac[i]); // derivative kinetics
        }
      }
    }

    // index of first entry for block i in CSR cols array
    index_t csr_idx_start = rows[i];
    // index of last entry for block i in CSR cols array
    index_t csr_idx_end = rows[i + 1];
    // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
    index_t conn_idx = csr_idx_start - i;

    jac_idx = N_VARS_SQ * csr_idx_start;

    // number of blocks between the last diagonal block and the current diagonal block in CSR array
    N_element = rows[i + 1] - rows[i] - 1;
    temp_conn_one_way.clear();
    temp_num.clear();
    for (index_t m = 0; m < N_element; m++)
    {
      temp_conn_one_way.push_back(conn_index_to_one_way[conn_idx + m]);
      temp_num.push_back(0);
    }

    for (index_t m = 0; m < N_element; m++)
    {
      for (index_t com : temp_conn_one_way)
      {
        if (com < temp_conn_one_way[m])
          temp_num[m] += 1;
      }
    }

    k_count = 0;

    for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS_SQ)
    { // fill offdiagonal part + contribute to diagonal

      j = cols[csr_idx];
      // skip diagonal
      if (i == j)
        continue;

      value_t trans_mult = 1;
      value_t trans_mult_der_i[N_VARS];
      value_t trans_mult_der_j[N_VARS];
      if (params->trans_mult_exp > 0 && i < mesh->n_res_blocks && j < mesh->n_res_blocks)
      {
        // Calculate transmissibility multiplier:
        phi_i = op_vals_arr[i * N_OPS + PORO_OP];
        phi_j = op_vals_arr[j * N_OPS + PORO_OP];

        // Take average interface porosity:
        phi_avg = (phi_i + phi_j) * 0.5;
        phi_0_avg = (mesh->poro[i] + mesh->poro[j]) * 0.5;

        trans_mult = params->trans_mult_exp * pow(phi_avg, params->trans_mult_exp - 1) * 0.5;
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          trans_mult_der_i[v] = trans_mult * op_ders_arr[(i * N_OPS + PORO_OP) * N_VARS + v];
          trans_mult_der_j[v] = trans_mult * op_ders_arr[(j * N_OPS + PORO_OP) * N_VARS + v];
        }
        trans_mult = pow(phi_avg, params->trans_mult_exp);
      }
      else
      {
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          trans_mult_der_i[v] = 0;
          trans_mult_der_j[v] = 0;
        }
      }

      p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];




      for (index_t wh : well_head_idx_collection)
      {
        if (i == wh)
        {
          well_head_tran_idx_collection.push_back(conn_index_to_one_way[conn_idx]);
        }
      }


      // [2] fill offdiagonal part + contribute to diagonal, only fluid part is considered in energy equation
      for (uint8_t p = 0; p < NP; p++)
      { // loop over number of phases for convective operator

        // calculate gravity term for phase p
        value_t avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] +
          op_vals_arr[j * N_OPS + GRAV_OP + p]) /
          2;

        // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
        value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

        // calculate partial derivatives for gravity and capillary terms
        value_t grav_pc_der_i[N_VARS];
        value_t grav_pc_der_j[N_VARS];
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          grav_pc_der_i[v] = -(op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + PC_OP + p) * N_VARS + v];
          grav_pc_der_j[v] = -(op_ders_arr[(j * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + PC_OP + p) * N_VARS + v];
        }

        double phase_gamma_p_diff = trans_mult * tran[conn_idx] * dt * phase_p_diff;

        if (phase_p_diff < 0)
        {
          // mass and energy outflow with effect of gravity and capillarity
          for (uint8_t c = 0; c < NE; c++)
          {
            //value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c];

            //RHS[i * N_VARS + c] -= phase_p_diff * c_flux; // flux operators only

            value_g_u = phase_p_diff * trans_mult * dt * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c];
            idx = count + c * N_element + temp_num[k_count];
            value_dg_dT[idx] -= value_g_u;
          }
        }
        else
        {
          // mass and energy inflow with effect of gravity and capillarity
          for (uint8_t c = 0; c < NE; c++)
          {
            //value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c];

            //RHS[i * N_VARS + c] -= phase_p_diff * c_flux; // flux operators only

            value_g_u = phase_p_diff * trans_mult * dt * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c];
            idx = count + c * N_element + temp_num[k_count];
            value_dg_dT[idx] -= value_g_u;
          }
        }

      } // end of loop over number of phases for convective operator with gravity and capillarity

      // [3] Additional diffusion code here:   (phi_p * S_p) * (rho_p * D_cp * Delta_x_cp)  or (phi_p * S_p) * (kappa_p * Delta_T)
      phi_avg = (mesh->poro[i] + mesh->poro[j]) * 0.5; // diffusion term depends on total porosity!

      // Only if block connection is between reservoir and reservoir cells!
      if (i < mesh->n_res_blocks && j < mesh->n_res_blocks)
      {
        // Add diffusion term to the residual:
        for (uint8_t c = 0; c < NE; c++)
        {
          for (uint8_t p = 0; p < NP; p++)
          {
            value_t grad_con = op_vals_arr[j * N_OPS + GRAD_OP + c * NP + p] - op_vals_arr[i * N_OPS + GRAD_OP + c * NP + p];

            // Diffusion flows, use arithmetic mean for compressibility and saturation (mass or energy):
            //value_t diff_mob_ups_m = dt * mesh->tranD[conn_idx] * phi_avg * (op_vals_arr[i * N_OPS + UPSAT_OP + p] + op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;
            //RHS[i * N_VARS + c] -= diff_mob_ups_m * grad_con; // diffusion term

            value_g_u = grad_con * dt * phi_avg * (op_vals_arr[i * N_OPS + UPSAT_OP + p] + op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;
            idx = count + c * N_element + temp_num[k_count];
            value_dg_dT[idx] -= value_g_u;
          }
        }
      }

      // [4] add rock conduction
      if (THERMAL)
      {
        t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
        gamma_t_diff = tranD[conn_idx] * dt * t_diff;

          // rock heat transfers flows from cell i to j
          //RHS[i * N_VARS + NC] -= gamma_t_diff * ((1 - mesh->poro[i]) * mesh->rock_cond[i] +
          //                                        (1 - mesh->poro[j]) * mesh->rock_cond[j]) / 2;

        value_g_u = dt * t_diff * ((1 - mesh->poro[i]) * mesh->rock_cond[i] +
                                   (1 - mesh->poro[j]) * mesh->rock_cond[j]) / 2;
        idx = count + NC * N_element + temp_num[k_count];
        value_dg_dT[idx] -= value_g_u;
      }

      k_count++;

      conn_idx++;

      //set the values of non-diagonal elements to zero
      /*for (uint8_t c = 0; c < N_VARS; c++)
      {
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          Jac_n[jac_idx + c * N_VARS + v] = 0;
        }
      }*/
      memset(&Jac_n[jac_idx], 0, N_VARS * N_VARS);
    }


    //// [5] finally add rock energy
    //// + rock energy (no rock compressibility included in these computations)
    //if (THERMAL)
    //{
    //  RHS[i * N_VARS + NC] += RV[i] * (op_vals_arr[i * N_OPS + TEMP_OP] - op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];

    //  for (uint8_t v = 0; v < N_VARS; v++)
    //  {
    //    Jac[diag_idx + NC * N_VARS + v] += RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i];
    //  } // end of fill offdiagonal part + contribute to diagonal
    //}


    if (jac_idx == diag_idx)
      jac_idx += N_VARS_SQ;

    count += N_VARS * N_element;


  } // end of loop over grid blocks
  


//  value_t CFL_max_local = 0;
//#ifdef _OPENMP
//#pragma omp critical 
//  {
//    if (CFL_max < CFL_max_local)
//      CFL_max = CFL_max_local;
//  }
//  }
//#else
//  CFL_max = CFL_max_local;
//#endif

  for (ms_well* w : wells)
  {
    //w->add_to_jacobian(dt, X, dg_dx, RHS);

    value_t *jac_n_well_head = &(dg_dx_n_temp->get_values()[dg_dx_n_temp->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
    memset(jac_n_well_head, 0, 2 * N_VARS_SQ * sizeof(value_t));
    for (uint8_t idx = 0; idx < N_VARS; idx++)
    {
      jac_n_well_head[idx + idx * N_VARS] = 0;
    }
  }

  // we have to convert the csr matrix to the csr matrix with block size of 1
  // because the function "build_transpose" is only applicable for the csr matrix with the block size of 1
  // this is also required by the linear solver "linsolv_superlu<1>", as the preconditioner is not applicable to adjoint so far
  // so this might be improved in the future
  csr_matrix<1> Temp, T1, T2;
  Temp.to_nb_1(static_cast<csr_matrix<N_VARS>*>(Jacobian));
  T1.build_transpose(&Temp);

  value_t* T1_values = T1.get_values();
  index_t* T1_rows = T1.get_rows_ptr();
  index_t* T1_cols = T1.get_cols_ind();
  index_t* T1_diag = T1.get_diag_ind();


  for (index_t i = 0; i <= n_blocks * N_VARS; i++)
  {
    //ad_diag[i] = i;  //so far using superlu, it may need to be fixed if using other linear solver
    ad_rows[i] = T1_rows[i];
  }
  //ad_rows[n_blocks * N_VARS] = T1_rows[n_blocks * N_VARS];

  index_t n_value = (mesh->n_conns + mesh->n_blocks) * N_VARS * N_VARS;
  for (index_t i = 0; i < n_value; i++)
  {
    ad_values[i] = T1_values[i];
    ad_cols[i] = T1_cols[i];
  }


  T2.to_nb_1(static_cast<csr_matrix<N_VARS>*>(dg_dx_n_temp));
  //T2.build_transpose(&Temp);

  value_t* T2_values = T2.get_values();
  index_t* T2_rows = T2.get_rows_ptr();
  index_t* T2_cols = T2.get_cols_ind();
  index_t* T2_diag = T2.get_diag_ind();

  value_t* ad_values_n = dg_dx_n->get_values();
  index_t* ad_rows_n = dg_dx_n->get_rows_ptr();
  index_t* ad_cols_n = dg_dx_n->get_cols_ind();
  index_t* ad_diag_n = dg_dx_n->get_diag_ind();



  for (index_t i = 0; i <= n_blocks * N_VARS; i++)
  {
    //ad_diag_n[i] = i;  //so far using superlu, it may need to be fixed if using other linear solver
    ad_rows_n[i] = T2_rows[i];
  }
  //ad_rows_n[n_blocks * N_VARS] = T2_rows[n_blocks * N_VARS];

  n_value = (mesh->n_conns + mesh->n_blocks) * N_VARS * N_VARS;
  for (index_t i = 0; i < n_value; i++)
  {
    ad_values_n[i] = T2_values[i];
    ad_cols_n[i] = T2_cols[i];
  }


    return 0;
};


//template<uint8_t NC, uint8_t NP, , bool THERMAL>
//double
//engine_super_cpu<NC, NP, THERMAL>::calc_newton_residual()
//{
//  double residual = 0, res;
//
//  std::vector <value_t> &hcap = mesh->heat_capacity;
//
//  residual = 0;
//  for (int i = 0; i < mesh->n_blocks; i++)
//  {
//    for (int c = 0; c < NC; c++)
//    {
//      res = fabs(RHS[i * N_VARS + c] / (PV[i] * op_vals_arr[i * N_OPS + c]));
//      if (res > residual)
//        residual = res;
//    }
//
//    if (THERMAL)
//    {
//      res = fabs(RHS[i * N_VARS + T_VAR] / (PV[i] * op_vals_arr[i * N_OPS + NC] + RV[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i]));
//      if (res > residual)
//        residual = res;
//    }
//  }
//  return residual;
//}

// compositional, kinetic (H2O, CO2, Ca+2, CO3-2, CaCO3):
//template class engine_super_cpu<2, 2, 1>;
//template struct recursive_instantiator_nc_np<engine_super_cpu, 2, MAX_NC, 1>;
//template struct recursive_instantiator_nc_np<engine_super_cpu, 2, MAX_NC, 2>;
//template struct recursive_instantiator_nc_np<engine_super_cpu, 2, MAX_NC, 3>;
