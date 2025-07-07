#include <algorithm>
#include <cmath>
#include <cstring>
#include <time.h>
#include <functional>
#include <string>
#include <iomanip>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "engine_nce_g_gpu.hpp"
#define H2O_MW 18.01528

template <uint8_t NC, uint8_t NP, uint8_t N_VARS, uint8_t P_VAR, uint8_t E_VAR, uint8_t N_OPS, uint8_t ACC_OP, uint8_t FLUX_OP, uint8_t FE_ACC_OP, uint8_t FE_FLUX_OP, uint8_t FE_COND_OP, uint8_t DENS_OP, uint8_t TEMP_OP>
__global__ void
assemble_jacobian_array_kernel(const unsigned int n_blocks, value_t dt,
                               value_t *X, value_t *RHS,
                               index_t *rows, index_t *cols, value_t *Jac, index_t *diag_ind,
                               value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                               value_t *tran, value_t *PV,
                               value_t *tranD, value_t *RV,
                               value_t *hcap, value_t *rock_cond, value_t *poro,
                               value_t *grav_coef)
{
  // Each thread processes single matrix block row
  // Memory access is not coalesced

  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  if (i > n_blocks - 1)
    return;

  index_t j, jac_idx = 0;
  value_t p_diff, t_diff, gamma_p_diff, gamma_t_diff, tran_dt;
  value_t RHS_l[N_VARS];
  value_t jac_diag_l[N_VARS * N_VARS];
  value_t jac_offd_l[N_VARS * N_VARS];
  //value_t op_vals_flux_arr_l[NC];

  // for now, leave CFL computation out of this
  // value_t CFL_in[NC], CFL_out[NC];
  // value_t CFL_max_local = 0;
  //int connected_with_well;

  // fill diagonal part
  // [NC] mass eqns
  for (uint8_t c = 0; c < NC; c++)
  {
    RHS_l[c] = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]); // acc operators only
    // CFL_out[c] = 0;
    // CFL_in[c] = 0;
    // connected_with_well = 0;
    for (uint8_t v = 0; v < N_VARS; v++)
    {
      jac_diag_l[c * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];
    }
  }
  // [1] energy eqn
  // fluid energy
  RHS_l[NC] = PV[i] * (op_vals_arr[i * N_OPS + FE_ACC_OP] - op_vals_arr_n[i * N_OPS + FE_ACC_OP]);
  // + rock energy (no rock compressibility included in these computations)
  RHS_l[NC] += RV[i] * (op_vals_arr[i * N_OPS + TEMP_OP] - op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];

  for (uint8_t v = 0; v < N_VARS; v++)
  {
    jac_diag_l[NC * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + FE_ACC_OP) * N_VARS + v];
    jac_diag_l[NC * N_VARS + v] += RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i];
  }

  // index of first entry for block i in CSR cols array
  index_t csr_idx_start = rows[i];
  // index of last entry for block i in CSR cols array
  index_t csr_idx_end = rows[i + 1];
  // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
  index_t conn_idx = csr_idx_start - i;

  jac_idx = N_VARS * N_VARS * csr_idx_start;

  // fill offdiagonal part + contribute to diagonal
  for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS * N_VARS)
  {
    j = cols[csr_idx];
    // skip diagonal
    if (i == j)
    {
      continue;
    }

    p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];
    t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
    gamma_t_diff = tranD[conn_idx] * dt * t_diff;

    // set offdiagonal values to 0 since they are added up in the loop over phases
    for (uint8_t e = 0; e < N_VARS; e++)
    {
      for (uint8_t v = 0; v < N_VARS; v++)
      {
        jac_offd_l[e * N_VARS + v] = 0;
      }
    }

    for (uint8_t p = 0; p < NP; p++)
    {
      // calculate gravity term for phase p
      value_t avg_density = (op_vals_arr[i * N_OPS + DENS_OP + p] + op_vals_arr[j * N_OPS + DENS_OP + p]) / 2;

      value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] * H2O_MW;
      double phase_gamma_p_diff = tran[conn_idx] * dt * phase_p_diff;

      if (phase_p_diff < 0)
      {
        // mass outflow
        for (uint8_t c = 0; c < NC; c++)
        {
          value_t c_flux = tran[conn_idx] * dt * op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c];
          RHS_l[c] -= phase_gamma_p_diff * op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c]; // flux operators only

          for (uint8_t v = 0; v < N_VARS; v++)
          {
            jac_diag_l[c * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + FLUX_OP + p * NC + c) * N_VARS + v];
            jac_diag_l[c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
            jac_offd_l[c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
            if (v == P_VAR)
            {
              jac_offd_l[c * N_VARS] -= c_flux;
              jac_diag_l[c * N_VARS] += c_flux;
            }
          }
        }

        // energy outflow
        RHS_l[E_VAR] -= phase_gamma_p_diff * op_vals_arr[i * N_OPS + FE_FLUX_OP + p]; // energy flux
        value_t phase_e_flux = tran[conn_idx] * dt * op_vals_arr[i * N_OPS + FE_FLUX_OP + p];

        for (uint8_t v = 0; v < N_VARS; v++)
        {
          jac_diag_l[NC * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + FE_FLUX_OP + p) * N_VARS + v];
          jac_diag_l[NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
          jac_offd_l[NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
          if (v == P_VAR)
          {
            jac_offd_l[NC * N_VARS] -= phase_e_flux;
            jac_diag_l[NC * N_VARS] += phase_e_flux;
          }
        }
      }
      else
      {
        //inflow

        // mass
        for (uint8_t c = 0; c < NC; c++)
        {
          value_t c_flux = tran[conn_idx] * dt * op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c];
          RHS_l[c] -= phase_gamma_p_diff * op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c]; // flux operators only

          for (uint8_t v = 0; v < N_VARS; v++)
          {
            jac_offd_l[c * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + FLUX_OP + p * NC + c) * N_VARS + v];
            jac_offd_l[c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
            jac_diag_l[c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
            if (v == P_VAR)
            {
              jac_diag_l[c * N_VARS] += c_flux;
              jac_offd_l[c * N_VARS] -= c_flux;
            }
          }
        }

        // energy flux
        RHS_l[E_VAR] -= phase_gamma_p_diff * op_vals_arr[j * N_OPS + FE_FLUX_OP + p]; // energy flux operator
        value_t phase_e_flux = tran[conn_idx] * dt * op_vals_arr[j * N_OPS + FE_FLUX_OP + p];
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          jac_offd_l[NC * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + FE_FLUX_OP + p) * N_VARS + v];
          jac_offd_l[NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
          jac_diag_l[NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] / 2 * H2O_MW;
          if (v == P_VAR)
          {
            jac_diag_l[NC * N_VARS] += phase_e_flux;
            jac_offd_l[NC * N_VARS] -= phase_e_flux;
          }
        }
      }
    }

    if (t_diff < 0)
    {
      // energy outflow

      // conduction
      value_t local_cond_dt = tranD[conn_idx] * dt * (op_vals_arr[i * N_OPS + FE_COND_OP] * poro[i] + (1 - poro[i]) * rock_cond[i]);

      RHS_l[NC] -= local_cond_dt * t_diff;
      for (uint8_t v = 0; v < N_VARS; v++)
      {
        // conduction part derivative
        jac_diag_l[NC * N_VARS + v] -= gamma_t_diff * op_ders_arr[(i * N_OPS + FE_COND_OP) * N_VARS + v] * poro[i];
        // t_diff derivatives
        jac_offd_l[NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
        jac_diag_l[NC * N_VARS + v] += op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
      }
    }
    else
    {
      //energy inflow
      // conduction
      value_t local_cond_dt = tranD[conn_idx] * dt * (op_vals_arr[j * N_OPS + FE_COND_OP] * poro[j] + (1 - poro[j]) * rock_cond[j]);

      RHS_l[NC] -= local_cond_dt * t_diff;
      for (uint8_t v = 0; v < N_VARS; v++)
      {
        // conduction part derivative
        jac_offd_l[NC * N_VARS + v] -= gamma_t_diff * op_ders_arr[(j * N_OPS + FE_COND_OP) * N_VARS + v] * poro[j];
        // t_diff derivatives
        jac_offd_l[NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
        jac_diag_l[NC * N_VARS + v] += op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
      }
    }
    conn_idx++;
    for (uint8_t e = 0; e < N_VARS; e++)
    {
      for (uint8_t v = 0; v < N_VARS; v++)
      {
        Jac[jac_idx + e * N_VARS + v] = jac_offd_l[e * N_VARS + v];
      }
    }
  }

  // index of diagonal block entry for block i in CSR values array
  jac_idx = N_VARS * N_VARS * diag_ind[i];

  for (uint8_t e = 0; e < N_VARS; e++)
  {
    RHS[i * N_VARS + e] = RHS_l[e];
    for (uint8_t v = 0; v < N_VARS; v++)
    {
      Jac[jac_idx + e * N_VARS + v] = jac_diag_l[e * N_VARS + v];
    }
  }
};

template <uint8_t NC, uint8_t NP>
int engine_nce_g_gpu<NC, NP>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                   std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                   sim_params *params_, timer_node *timer_)
{
  engine_base_gpu::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);

  allocate_device_data(RV, &RV_d);
  allocate_device_data(mesh->heat_capacity, &mesh_hcap_d);
  allocate_device_data(mesh->tranD, &mesh_tranD_d);
  allocate_device_data(mesh->rock_cond, &mesh_rcond_d);
  allocate_device_data(mesh->poro, &mesh_poro_d);
  allocate_device_data(mesh->grav_coef, &mesh_grav_coef_d);

  copy_data_to_device(RV, RV_d);
  copy_data_to_device(mesh->heat_capacity, mesh_hcap_d);
  copy_data_to_device(mesh->tranD, mesh_tranD_d);
  copy_data_to_device(mesh->rock_cond, mesh_rcond_d);
  copy_data_to_device(mesh->poro, mesh_poro_d);
  copy_data_to_device(mesh->grav_coef, mesh_grav_coef_d);

  return 0;
}

template <uint8_t NC, uint8_t NP>
int engine_nce_g_gpu<NC, NP>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
  timer->node["jacobian assembly"].node["kernel"].start_gpu();
  assemble_jacobian_array_kernel<NC, NP, N_VARS, P_VAR, E_VAR, N_OPS, ACC_OP, FLUX_OP, FE_ACC_OP, FE_FLUX_OP, FE_COND_OP, DENS_OP, TEMP_OP>
      KERNEL_1D_THREAD(mesh->n_blocks, KERNEL_BLOCK_SIZE)(mesh->n_blocks, dt,
                                                          X_d, RHS_d,
                                                          jacobian->rows_ptr_d, jacobian->cols_ind_d, jacobian->values_d, jacobian->diag_ind_d,
                                                          op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                          mesh_tran_d, PV_d,
                                                          mesh_tranD_d, RV_d,
                                                          mesh_hcap_d, mesh_rcond_d, mesh_poro_d, mesh_grav_coef_d);

  timer->node["jacobian assembly"].node["kernel"].stop_gpu();
  timer->node["jacobian assembly"].node["wells"].start_gpu();
  int i_w = 0;
  for (ms_well *w : wells)
  {
    value_t *jac_well_head = &(jac_wells[2 * N_VARS * N_VARS * i_w]);
    w->add_to_jacobian(dt, X, jac_well_head, RHS);
    i_w++;
  }
  copy_data_to_device(jac_wells, jac_wells_d);
  copy_data_to_device(RHS, RHS_wells_d);

  i_w = 0;
  for (ms_well *w : wells)
  {
    copy_data_within_device(RHS_d + N_VARS * w->well_head_idx, RHS_wells_d + N_VARS * w->well_head_idx, N_VARS);
    copy_data_within_device(jacobian->values_d + jacobian->rows_ptr[w->well_head_idx] * N_VARS * N_VARS, jac_wells_d + 2 * N_VARS * N_VARS * i_w, 2 * N_VARS * N_VARS);
    i_w++;
  }
  timer->node["jacobian assembly"].node["wells"].stop_gpu();

  // copy_data_to_host(jacobian->values, jacobian->values_d, N_VARS * N_VARS * jacobian->rows_ptr[mesh->n_blocks]);
  // jacobian->write_matrix_to_file("jac_nc_dar_gpu.csr");
  // exit(0);

  return 0;
};

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_gpu<NC, NP>::calc_newton_residual_L2()
{
  double residual = 0, res = 0;
  double res_m = 0, res_e = 0;
  std::vector<value_t> &hcap = mesh->heat_capacity;

  for (int i = 0; i < mesh->n_res_blocks; i++)
  {
    for (int c = 0; c < NC; c++)
    {
      res = fabs(RHS[i * N_VARS + c] / (PV[i] * op_vals_arr[i * N_OPS + c]));
      res_m += res * res;
    }

    res = fabs(RHS[i * N_VARS + E_VAR] / (PV[i] * op_vals_arr[i * N_OPS + FE_ACC_OP] + RV[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i]));
    res_e += res * res;
  }
  residual = sqrt(res_m + res_e);
  return residual;
}

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_gpu<NC, NP>::calc_newton_residual_Linf()
{
  double residual = 0, res = 0;
  std::vector<value_t> &hcap = mesh->heat_capacity;

  for (int i = 0; i < mesh->n_res_blocks; i++)
  {
    for (int c = 0; c < NC; c++)
    {
      res = fabs(RHS[i * N_VARS + c] / (PV[i] * op_vals_arr[i * N_OPS + c]));
      if (res > residual)
        residual = res;
    }

    res = fabs(RHS[i * N_VARS + E_VAR] / (PV[i] * op_vals_arr[i * N_OPS + FE_ACC_OP] + RV[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i]));
    if (res > residual)
      residual = res;
  }
  return residual;
}

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_gpu<NC, NP>::calc_well_residual_L2()
{
  double residual = 0;
  std::vector<value_t> res(n_vars, 0);
  std::vector<value_t> norm(n_vars, 0);

  std::vector<value_t> &hcap = mesh->heat_capacity;

  for (ms_well *w : wells)
  {
    int nperf = w->perforations.size();
    for (int ip = 0; ip < nperf; ip++)
    {
      index_t i_w, i_r;
      value_t wi, wid;
      std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

      for (int c = 0; c < nc; c++)
      {
        res[c] += RHS[(w->well_body_idx + i_w) * n_vars + c] * RHS[(w->well_body_idx + i_w) * n_vars + c];
        norm[c] += PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + c] * PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + c];
      }
      res[E_VAR] += RHS[(w->well_body_idx + i_w) * n_vars + E_VAR] * RHS[(w->well_body_idx + i_w) * n_vars + E_VAR];
      norm[E_VAR] += PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP] * PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP];
    }
    // and then add RHS for well control equations
    for (int c = 0; c < nc; c++)
    {
      // well constraints should not be normalized, so pre-multiply by norm
      res[c] += RHS[w->well_head_idx * n_vars + c] * RHS[w->well_head_idx * n_vars + c] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + c] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + c];
    }
    res[E_VAR] += RHS[(w->well_head_idx) * n_vars + E_VAR] * RHS[(w->well_head_idx) * n_vars + E_VAR] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP];
  }

  for (int v = 0; v < n_vars; v++)
  {
    residual = std::max(residual, sqrt(res[v] / norm[v]));
  }
  return residual;
}

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_gpu<NC, NP>::calc_well_residual_Linf()
{
  double residual = 0, res = 0;
  std::vector<value_t> &hcap = mesh->heat_capacity;

  for (ms_well *w : wells)
  {
    int nperf = w->perforations.size();
    for (int ip = 0; ip < nperf; ip++)
    {
      index_t i_w, i_r;
      value_t wi, wid;
      std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

      for (int c = 0; c < nc; c++)
      {
        res = fabs(RHS[(w->well_body_idx + i_w) * n_vars + c] / (PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + c]));
        residual = std::max(residual, res);
      }
      res = fabs(RHS[(w->well_body_idx + i_w) * n_vars + E_VAR] / (PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP]));
      residual = std::max(residual, res);
    }
    // and then add RHS for well control equations
    for (int c = 0; c < nc; c++)
    {
      // well constraints should not be normalized, so pre-multiply by norm
      res = fabs(RHS[w->well_head_idx * n_vars + c]);
      residual = std::max(residual, res);
    }
    res = fabs(RHS[(w->well_head_idx) * n_vars + E_VAR]);
    residual = std::max(residual, res);
  }

  return residual;
}

template<uint8_t NC, uint8_t NP>
int
engine_nce_g_gpu<NC, NP>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};

template class engine_nce_g_gpu<1, 2>;

