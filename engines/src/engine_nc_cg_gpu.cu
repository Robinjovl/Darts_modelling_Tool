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

#include "engine_nc_cg_gpu.hpp"

template <uint8_t NC, uint8_t NP, uint8_t N_VARS, uint8_t P_VAR, uint8_t N_OPS, uint8_t N_PHASE_OPS, uint8_t ACC_OP, uint8_t FLUX_OP, uint8_t DENS_OP, uint8_t PC_OP>
__global__ void
assemble_jacobian_array_kernel(const unsigned int n_blocks, value_t dt,
                               value_t *X, value_t *RHS,
                               index_t *rows, index_t *cols, value_t *Jac, index_t *diag_ind,
                               value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                               value_t *tran, value_t *PV, value_t *grav_coef)
{
  // Each matrix block row is processed by N_VARS * N_VARS threads
  // Memory access is coalesced for most data, while communications minimized

  const int N_VARS_SQ = N_VARS * N_VARS;

  // mesh grid block number
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / N_VARS_SQ;
  // Each thread is pinned to specific position in matrix block
  // and evaluates diagonal and all offdiagonal Jacobian entries at that position for the block row i
  // if the block position corresponds to the first column of the block (v==0, see below),
  // that thread also evaluates RHS value for equation c of the the block row i.
  const int block_pos = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS_SQ;
  // v (variable) is the column of that position
  const int v = block_pos % N_VARS;
  // c (component, but it is outdated notation- better to use e, equation) is the row of that position
  const int c = block_pos / N_VARS;

  if (i > n_blocks - 1)
    return;

  // local value of jacobian diagonal block value at block_pos
  // gets contributions in the loop over connections and is written to global memory at the end
  value_t jac_diag = 0;
  // local value of RHS according to c
  // gets contributions in the loop over connections and is written to global memory at the end
  value_t rhs = 0;
  // local value of jacobian offdiagonal block value at block_pos
  // is evaluated and written to global memory during each iteration of the loop over connections
  value_t jac_offd;

  index_t j, jac_idx;
  value_t p_diff, tran_dt;

  // initialize rhs and diagonal part with accumulation
  if (v == 0)
  {
    rhs = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]); // acc operators only
  }
  jac_diag = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];

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
    tran_dt = tran[conn_idx] * dt;
    jac_offd = 0;

    for (uint8_t p = 0; p < NP; p++)
    {
      // calculate gravity term for phase p
      value_t avg_density = (op_vals_arr[i * N_OPS + p * N_PHASE_OPS + DENS_OP] +
                             op_vals_arr[j * N_OPS + p * N_PHASE_OPS + DENS_OP]) /
                            2;

      // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
      value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + p * N_PHASE_OPS + PC_OP] + op_vals_arr[i * N_OPS + p * N_PHASE_OPS + PC_OP];

      // calculate partial derivatives for gravity and capillary terms
      value_t grav_pc_der_i, grav_pc_der_j;
      grav_pc_der_i = -(op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];
      grav_pc_der_j = -(op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];

      double phase_gamma_p_diff = tran_dt * phase_p_diff;
      if (phase_p_diff < 0)
      {
        //outflow
        value_t c_flux = tran_dt * op_vals_arr[i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];

        jac_diag -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
        jac_diag += c_flux * grav_pc_der_i;
        jac_offd += c_flux * grav_pc_der_j;

        if (v == 0)
        {
          rhs -= phase_p_diff * c_flux; // flux operators only
          jac_offd -= c_flux;
          jac_diag += c_flux;
        }
      }
      else
      {
        //inflow

        value_t c_flux = tran_dt * op_vals_arr[j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];
        jac_offd -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
        jac_diag += c_flux * grav_pc_der_i;
        jac_offd += c_flux * grav_pc_der_j;
        if (v == 0)
        {
          rhs -= phase_p_diff * c_flux; // flux operators only
          jac_diag += c_flux;           //-= Jac[jac_idx + c * N_VARS];
          jac_offd -= c_flux;           // -tran_dt * op_vals[NC + c];
        }
      }
    }
    // write down offdiagonal value
    Jac[jac_idx + c * N_VARS + v] = jac_offd;
    conn_idx++;
  }

  // index of diagonal block entry for block i
  index_t diag_idx = N_VARS_SQ * diag_ind[i];
  // write down diag value and RHS
  Jac[diag_idx + block_pos] = jac_diag;

  if (v == 0)
  {
    RHS[i * N_VARS + c] = rhs;
  }
};

// Assembly jacobian and compute spmv product R = Jacobian * V
template <uint8_t NC, uint8_t NP, uint8_t N_VARS, uint8_t P_VAR, uint8_t N_OPS, uint8_t N_PHASE_OPS, uint8_t ACC_OP, uint8_t FLUX_OP, uint8_t DENS_OP, uint8_t PC_OP>
__global__ void
assemble_jacobian_array_spmv0(const unsigned int n_blocks, value_t dt,
                              value_t *X, value_t *RHS,
                              index_t *rows, index_t *cols, value_t *Jac, index_t *diag_ind,
                              value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                              value_t *tran, value_t *PV, value_t *grav_coef,
                              const value_t *V, value_t *R)
{
  // Each matrix block row is processed by N_VARS threads, block by block, and each block is processed column by column
  // Each thread is pinned to specific row in matrix block - c (component)

  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / (N_VARS); // global thread index
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;

  if (i > n_blocks - 1)
    return;

  // local value of jacobian offdiagonal
  // is computed for every column in every offdiagonal block and immediately used
  value_t jac_offd;
  // local values of all columns of jacobian diagonal block at row c
  // is contributed by all ofdiagonal entries and used at the end
  value_t jac_diag[N_VARS];

  // local value of R and V
  value_t r_val = 0, v_val;

  index_t j, jac_idx;
  value_t p_diff, tran_dt;

  // initialize diagonal part with accumulation
  for (index_t v = 0; v < N_VARS; v++)
  {
    jac_diag[v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];
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
    tran_dt = tran[conn_idx] * dt;
    for (uint8_t p = 0; p < NP; p++)
    {
      // calculate gravity term for phase p
      value_t avg_density = (op_vals_arr[i * N_OPS + p * N_PHASE_OPS + DENS_OP] +
                             op_vals_arr[j * N_OPS + p * N_PHASE_OPS + DENS_OP]) /
                            2;

      // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
      value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + p * N_PHASE_OPS + PC_OP] + op_vals_arr[i * N_OPS + p * N_PHASE_OPS + PC_OP];

      for (index_t v = 0; v < N_VARS; v++)
      {
        jac_offd = 0;
        v_val = V[j * N_VARS + v];

        // calculate partial derivatives for gravity and capillary terms
        value_t grav_pc_der_i, grav_pc_der_j;
        grav_pc_der_i = -(op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];
        grav_pc_der_j = -(op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];

        double phase_gamma_p_diff = tran_dt * phase_p_diff;
        if (phase_p_diff < 0)
        {
          //outflow
          value_t c_flux = tran_dt * op_vals_arr[i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];

          jac_diag[v] -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
          jac_diag[v] += c_flux * grav_pc_der_i;
          jac_offd += c_flux * grav_pc_der_j;

          if (v == 0)
          {
            jac_offd -= c_flux;
            jac_diag[v] += c_flux;
          }
        }
        else
        {
          //inflow

          value_t c_flux = tran_dt * op_vals_arr[j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];
          jac_offd -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
          jac_diag[v] += c_flux * grav_pc_der_i;
          jac_offd += c_flux * grav_pc_der_j;
          if (v == 0)
          {
            jac_diag[v] += c_flux; //-= Jac[jac_idx + c * N_VARS];
            jac_offd -= c_flux;    // -tran_dt * op_vals[NC + c];
          }
        }
        // compute intermediate result
        r_val += jac_offd * v_val;
      }
    }
    conn_idx++;
  }

  // diag is now complete - take it into account
  for (index_t v = 0; v < N_VARS; v++)
  {
    r_val += jac_diag[v] * V[i * N_VARS + v];
  }
  R[i * N_VARS + c] = r_val;
};

// Assembly jacobian and compute R = alpha * Jacobian * V + beta * U
// Note: different notation from csr_matrix, V corresponds to u and U == v
// This way lincomb kernel is similar to spmv, and V always gets multiplied by Jacobian
// however the order of arguments is matching
template <uint8_t NC, uint8_t NP, uint8_t N_VARS, uint8_t P_VAR, uint8_t N_OPS, uint8_t N_PHASE_OPS, uint8_t ACC_OP, uint8_t FLUX_OP, uint8_t DENS_OP, uint8_t PC_OP>
__global__ void
assemble_jacobian_array_lincomb(const unsigned int n_blocks, value_t dt,
                                value_t *X, value_t *RHS,
                                index_t *rows, index_t *cols, value_t *Jac, index_t *diag_ind,
                                value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                                value_t *tran, value_t *PV, value_t *grav_coef,
                                value_t alpha, value_t beta,
                                const value_t *V, value_t *U, value_t *R)
{
  // Each matrix block row is processed by N_VARS threads, block by block, and each block is processed column by column
  // Each thread is pinned to specific row in matrix block - c (component)

  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / (N_VARS); // global thread index
  const int c = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;

  if (i > n_blocks - 1)
    return;

  // local value of jacobian offdiagonal
  // is computed for every column in every offdiagonal block and immediately used
  value_t jac_offd;
  // local values of all columns of jacobian diagonal block at row c
  // is contributed by all ofdiagonal entries and used at the end
  value_t jac_diag[N_VARS];

  // local value of R and V
  value_t r_val = 0, v_val;

  index_t j, jac_idx;
  value_t p_diff, tran_dt;

  // initialize diagonal part with accumulation
  for (index_t v = 0; v < N_VARS; v++)
  {
    jac_diag[v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];
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
    tran_dt = tran[conn_idx] * dt;
    for (uint8_t p = 0; p < NP; p++)
    {
      // calculate gravity term for phase p
      value_t avg_density = (op_vals_arr[i * N_OPS + p * N_PHASE_OPS + DENS_OP] +
                             op_vals_arr[j * N_OPS + p * N_PHASE_OPS + DENS_OP]) /
                            2;

      // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
      value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + p * N_PHASE_OPS + PC_OP] + op_vals_arr[i * N_OPS + p * N_PHASE_OPS + PC_OP];

      for (index_t v = 0; v < N_VARS; v++)
      {
        jac_offd = 0;
        v_val = V[j * N_VARS + v];

        // calculate partial derivatives for gravity and capillary terms
        value_t grav_pc_der_i, grav_pc_der_j;
        grav_pc_der_i = -(op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];
        grav_pc_der_j = -(op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + DENS_OP) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + PC_OP) * N_VARS + v];

        double phase_gamma_p_diff = tran_dt * phase_p_diff;
        if (phase_p_diff < 0)
        {
          //outflow
          value_t c_flux = tran_dt * op_vals_arr[i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];

          jac_diag[v] -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
          jac_diag[v] += c_flux * grav_pc_der_i;
          jac_offd += c_flux * grav_pc_der_j;

          if (v == 0)
          {
            jac_offd -= c_flux;
            jac_diag[v] += c_flux;
          }
        }
        else
        {
          //inflow

          value_t c_flux = tran_dt * op_vals_arr[j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c];
          jac_offd -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + p * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
          jac_diag[v] += c_flux * grav_pc_der_i;
          jac_offd += c_flux * grav_pc_der_j;
          if (v == 0)
          {
            jac_diag[v] += c_flux; //-= Jac[jac_idx + c * N_VARS];
            jac_offd -= c_flux;    // -tran_dt * op_vals[NC + c];
          }
        }
        // compute intermediate result
        r_val += jac_offd * v_val;
      }
    }
    conn_idx++;
  }

  // diag is now complete - take it into account
  for (index_t v = 0; v < N_VARS; v++)
  {
    r_val += jac_diag[v] * V[i * N_VARS + v];
  }

  R[i * N_VARS + c] = r_val * alpha + U[i * N_VARS + c] * beta;
};

template <uint8_t N_VARS>
__global__ void
jacobian_wells_spmv0(index_t n_wells, index_t *jac_well_idxs,
                     index_t *rows, index_t *cols,
                     value_t *jac_wells,
                     const value_t *V, value_t *R)
{
  // Each matrix block row is processed by N_VARS threads, block by block, and each block is processed column by column
  // Each thread is pinned to specific row in matrix block - r

  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / (N_VARS); // global thread index
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;

  // local value of R and V
  value_t r_val = 0;

  if (w > n_wells - 1)
    return;

  // index of first entry for block i in CSR cols array
  index_t i = jac_well_idxs[w];
  value_t *jac_well = jac_wells + 2 * N_VARS * N_VARS * w;
  index_t csr_idx_start = rows[i];

  // rely on the fact that each Jacobian row, corresponding to pre-computed well constraints (well heads, wh), has exactly 2 blocks
  for (index_t b = 0; b < 2; b++)
  {
    index_t j = cols[csr_idx_start + b];
    for (index_t c = 0; c < N_VARS; c++)
    {
      r_val += jac_well[b * N_VARS * N_VARS + r * N_VARS + c] * V[j * N_VARS + c];
    }
  }

  R[i * N_VARS + r] = r_val;

  return;
};

// Contribute to lincomb for Jacobian rows corresponding to pre-computed well constraints (well heads, wh)
// R[wh] = alpha * Jacobian[wh] * V + beta * U[wh]
template <uint8_t N_VARS>
__global__ void
jacobian_wells_lincomb(index_t n_wells, index_t *jac_well_idxs,
                       index_t *rows, index_t *cols,
                       value_t *jac_wells,
                       value_t alpha, value_t beta,
                       const value_t *V, value_t *U, value_t *R)
{

  // Each matrix block row is processed by N_VARS threads, block by block, and each block is processed column by column
  // Each thread is pinned to specific row in matrix block - r

  const int w = (blockIdx.x * blockDim.x + threadIdx.x) / (N_VARS); // global thread index
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;

  // local value of R and V
  value_t r_val = 0, u_val;

  if (w > n_wells - 1)
    return;

  // index of first entry for block i in CSR cols array
  index_t i = jac_well_idxs[w];
  value_t *jac_well = jac_wells + 2 * N_VARS * N_VARS * w;
  index_t csr_idx_start = rows[i];
  u_val = U[i * N_VARS + r] * beta;

  // rely on the fact that each Jacobian row, corresponding to pre-computed well constraints (well heads, wh), has exactly 2 blocks
  for (index_t b = 0; b < 2; b++)
  {
    index_t j = cols[csr_idx_start + b];
    for (index_t c = 0; c < N_VARS; c++)
    {
      r_val += jac_well[b * N_VARS * N_VARS + r * N_VARS + c] * V[j * N_VARS + c];
    }
  }

  R[i * N_VARS + r] = r_val * alpha + u_val;

  return;
};

template <uint8_t NC, uint8_t NP>
int engine_nc_cg_gpu<NC, NP>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                   std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                   sim_params *params_, timer_node *timer_)
{
  engine_base_gpu::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);

  allocate_device_data(mesh->grav_coef, &mesh_grav_coef_d);
  copy_data_to_device(mesh->grav_coef, mesh_grav_coef_d);

  return 0;
}

template <uint8_t NC, uint8_t NP>
int engine_nc_cg_gpu<NC, NP>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
  timer->node["jacobian assembly"].node["kernel"].start_gpu();
  //cudaMemset(jacobian->values_d, 0, jacobian->rows_ptr[mesh->n_blocks] * N_VARS_SQ * sizeof(double));

  assemble_jacobian_array_kernel<NC, NP, N_VARS, P_VAR, N_OPS, N_PHASE_OPS, ACC_OP, FLUX_OP, DENS_OP, PC_OP>
      KERNEL_1D(mesh->n_blocks, N_VARS * N_VARS, ASSEMBLY_N_VARS_N_VARS_BLOCK_SIZE)(mesh->n_blocks, dt,
                                                                                    X_d, RHS_d,
                                                                                    jacobian->rows_ptr_d, jacobian->cols_ind_d, jacobian->values_d, jacobian->diag_ind_d,
                                                                                    op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                                                    mesh_tran_d, PV_d, mesh_grav_coef_d);

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
  //exit(0);

  return 0;
}

// calc r_d = Jacobian * v_d
template <uint8_t NC, uint8_t NP>
int engine_nc_cg_gpu<NC, NP>::matrix_vector_product_d0(const value_t *v_d, value_t *r_d)
{
  assemble_jacobian_array_spmv0<NC, NP, N_VARS, P_VAR, N_OPS, N_PHASE_OPS, ACC_OP, FLUX_OP, DENS_OP, PC_OP>
      KERNEL_1D(mesh->n_blocks, N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(mesh->n_blocks, dt,
                                                                    X_d, RHS_d,
                                                                    Jacobian->rows_ptr_d, Jacobian->cols_ind_d, Jacobian->values_d, Jacobian->diag_ind_d,
                                                                    op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                                    mesh_tran_d, PV_d, mesh_grav_coef_d, v_d, r_d);

  // lay SPMV result from correct well constraint jacobian over r_d
  jacobian_wells_spmv0<N_VARS>
      KERNEL_1D(wells.size(), N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(wells.size(), jac_well_head_idxs_d,
                                                                  Jacobian->rows_ptr_d, Jacobian->cols_ind_d, jac_wells_d, v_d, r_d);

  return 0;
}

// calc r_d = alpha * Jacobian * v_d + beta * r_d
template <uint8_t NC, uint8_t NP>
int engine_nc_cg_gpu<NC, NP>::calc_lin_comb_d(value_t alpha, value_t beta, value_t *u_d, value_t *v_d, value_t *r_d)
{
  assemble_jacobian_array_lincomb<NC, NP, N_VARS, P_VAR, N_OPS, N_PHASE_OPS, ACC_OP, FLUX_OP, DENS_OP, PC_OP>
      KERNEL_1D(mesh->n_blocks, N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(mesh->n_blocks, dt,
                                                                    X_d, RHS_d,
                                                                    Jacobian->rows_ptr_d, Jacobian->cols_ind_d, Jacobian->values_d, Jacobian->diag_ind_d,
                                                                    op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                                    mesh_tran_d, PV_d, mesh_grav_coef_d,
                                                                    alpha, beta, u_d, v_d, r_d);
  // lay SPMV result from correct well constraint jacobian over r_d
  jacobian_wells_lincomb<N_VARS>
      KERNEL_1D(wells.size(), N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(wells.size(), jac_well_head_idxs_d,
                                                                  Jacobian->rows_ptr_d, Jacobian->cols_ind_d, jac_wells_d, alpha, beta, u_d, v_d, r_d);
  return 0;
}

template<uint8_t NC, uint8_t NP>
int
engine_nc_cg_gpu<NC, NP>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};

// 2 phase: from 2 to MAX_NC components
template struct recursive_instantiator_nc_np<engine_nc_cg_gpu, 2, MAX_NC, 2>;

// 3 phase: only for 3 components
template struct recursive_instantiator_nc_np<engine_nc_cg_gpu, 3, 3, 3>;

