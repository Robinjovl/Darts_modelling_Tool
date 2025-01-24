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
#ifdef __GNUC__
#include <cxxabi.h>
#endif

#include "engine_nc_gpu.hpp"

template <uint8_t NC, uint8_t N_VARS, uint8_t P_VAR, uint8_t N_OPS, uint8_t ACC_OP, uint8_t FLUX_OP>
__global__ void
assemble_jacobian_array_kernel3(const unsigned int n_blocks, value_t dt,
                                value_t *X, value_t *RHS,
                                index_t *rows, index_t *cols, value_t *Jac, index_t *diag_ind,
                                value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                                value_t *tran, value_t *PV)
{
  // Each matrix block rows is processed by N_VARS * N_VARS threads
  // Memory access is coalesced for most data, while communications minimized
  // Each thread is pinned to specific position in matrix block
  // Therefore matrix row blocks are 'scanned' through, block by block

  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / (N_VARS * N_VARS); // global thread index
  const int rowblockvalid = (blockIdx.x * blockDim.x + threadIdx.x) % (N_VARS * N_VARS);
  const int c = rowblockvalid % N_VARS;
  const int r = rowblockvalid / N_VARS;

  // local value of jacobian block according to rowblockvalid
  value_t jac = 0;
  value_t jac_offd_flux = 0;
  value_t jac_diag = 0;
  // local value of RHS to r
  value_t rhs = 0;

  // value_t jac_offd_flux = 0;
  // value_t jac_diag_flux = 0;

  if (i > n_blocks - 1)
    return;

  index_t jac_diag_idx = N_VARS * N_VARS * diag_ind[i];

  // index of first entry for block i in CSR cols array
  index_t csr_idx_start = rows[i];
  // index of last entry for block i in CSR cols array
  index_t csr_idx_end = rows[i + 1];
  index_t conn_idx = csr_idx_start - i;

  for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++)
  {
    jac_offd_flux = 0;
    index_t j = cols[csr_idx];
    if (i != j)
    {
      // process offdiag

      value_t p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];
      value_t tran_dt = tran[conn_idx] * dt;
      value_t gamma_p_diff = tran_dt * p_diff;

      if (p_diff < 0)
      {
        jac_diag += -gamma_p_diff * op_ders_arr[(i * N_OPS + FLUX_OP) * N_VARS + rowblockvalid];

        if (c == 0)
        {
          // pressure
          value_t flux = op_vals_arr[i * N_OPS + FLUX_OP + r];
          jac_offd_flux = -tran_dt * flux;
          jac_diag -= jac_offd_flux;
          rhs -= gamma_p_diff * flux;
        }
      }
      else
      {
        jac_offd_flux = -gamma_p_diff * op_ders_arr[(j * N_OPS + FLUX_OP) * N_VARS + rowblockvalid];

        if (c == 0)
        {
          // pressure
          value_t val = -tran_dt * op_vals_arr[j * N_OPS + FLUX_OP + r];
          jac_offd_flux += val;
          jac_diag -= val;
          rhs -= gamma_p_diff * op_vals_arr[j * N_OPS + FLUX_OP + r];
        }
      }
      // write down offdiag value
      Jac[csr_idx * N_VARS * N_VARS + rowblockvalid] = jac_offd_flux;
      conn_idx++;
    }
    else
    {
      // process diag
      jac_diag += PV[i] * op_ders_arr[(i * N_OPS + ACC_OP) * N_VARS + rowblockvalid];
      if (c == 0)
      {
        rhs += PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + r] - op_vals_arr_n[i * N_OPS + ACC_OP + r]); // acc operators only
      }
    }
  }

  Jac[jac_diag_idx + rowblockvalid] = jac_diag;

  if (c == 0)
  {
    RHS[i * N_VARS + r] = rhs;
  }

  return;
};

// Assembly jacobian and compute spmv product R = Jacobian * V
template <uint8_t NC, uint8_t N_VARS, uint8_t P_VAR, uint8_t N_OPS, uint8_t ACC_OP, uint8_t FLUX_OP>
__global__ void
assemble_jacobian_array_kernel4_spmv0(const unsigned int n_blocks, value_t dt,
                                      value_t *X,
                                      index_t *rows, index_t *cols,
                                      value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                                      value_t *tran, value_t *PV,
                                      const value_t *V, value_t *R)
{
  // Each matrix block row is processed by N_VARS threads, block by block, and each block is processed column by column
  // Each thread is pinned to specific row in matrix block - r

  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / (N_VARS); // global thread index
  //const int rowblockvalid = (blockIdx.x * blockDim.x + threadIdx.x) % (N_VARS * N_VARS);
  //const int c = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;
  //printf("Row %d, r %d\n", i, r);
  // local value of jacobian block according to rowblockvalid
  value_t jac = 0;
  value_t jac_offd_flux = 0;
  value_t jac_diag[N_VARS];
  // local value of R and V
  value_t r_val = 0, v_val;

  if (i > n_blocks - 1)
    return;

  for (index_t c = 0; c < N_VARS; c++)
  {
    //printf("Row %d, r %d, c%d\n", i, r, c);
    jac_diag[c] = 0;
  }

  // index of first entry for block i in CSR cols array
  index_t csr_idx_start = rows[i];
  // index of last entry for block i in CSR cols array
  index_t csr_idx_end = rows[i + 1];
  index_t conn_idx = csr_idx_start - i;

  for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++)
  {

    jac_offd_flux = 0;
    index_t j = cols[csr_idx];
    if (i != j)
    {
      // process offdiag

      value_t p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];
      value_t tran_dt = tran[conn_idx] * dt;
      value_t gamma_p_diff = tran_dt * p_diff;

      if (p_diff < 0)
      {
        for (index_t c = 0; c < N_VARS; c++)
        {
          v_val = V[j * N_VARS + c];
          jac_diag[c] += -gamma_p_diff * op_ders_arr[(i * N_OPS + FLUX_OP) * N_VARS + r * N_VARS + c];
          jac_offd_flux = 0;
          if (c == 0)
          {
            // pressure
            value_t flux = op_vals_arr[i * N_OPS + FLUX_OP + r];
            jac_offd_flux = -tran_dt * flux;
            jac_diag[c] -= jac_offd_flux;
          }
          // compute intermediate result
          r_val += jac_offd_flux * v_val;
        }
      }
      else
      {
        for (index_t c = 0; c < N_VARS; c++)
        {
          v_val = V[j * N_VARS + c];
          jac_offd_flux = -gamma_p_diff * op_ders_arr[(j * N_OPS + FLUX_OP) * N_VARS + r * N_VARS + c];

          if (c == 0)
          {
            // pressure
            value_t val = -tran_dt * op_vals_arr[j * N_OPS + FLUX_OP + r];
            jac_offd_flux += val;
            jac_diag[c] -= val;
          }

          // compute intermediate result
          r_val += jac_offd_flux * v_val;
        }
      }

      conn_idx++;
    }
    else
    {
      // process diag
      for (index_t c = 0; c < N_VARS; c++)
      {
        jac_diag[c] += PV[i] * op_ders_arr[(i * N_OPS + ACC_OP) * N_VARS + r * N_VARS + c];
      }
    }
  }

  // diag is now complete - take it into account
  for (index_t c = 0; c < N_VARS; c++)
  {
    r_val += jac_diag[c] * V[i * N_VARS + c];
  }
  R[i * N_VARS + r] = r_val;

  return;
};

// Assembly jacobian and compute R = alpha * Jacobian * V + beta * U
// Note: different notation from csr_matrix, V corresponds to u and U == v
// This way lincomb kernel is similar to spmv, and V always gets multiplied by Jacobian
// however the order of arguments is matching
template <uint8_t NC, uint8_t N_VARS, uint8_t P_VAR, uint8_t N_OPS, uint8_t ACC_OP, uint8_t FLUX_OP>
__global__ void
assemble_jacobian_array_kernel4_lincomb(const unsigned int n_blocks, value_t dt,
                                        value_t *X,
                                        index_t *rows, index_t *cols,
                                        value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                                        value_t *tran, value_t *PV,
                                        value_t alpha, value_t beta,
                                        const value_t *V, value_t *U, value_t *R)
{
  // Each matrix block row is processed by N_VARS threads, block by block, and each block is processed column by column
  // Each thread is pinned to specific row in matrix block - r

  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / (N_VARS); // global thread index
  //const int rowblockvalid = (blockIdx.x * blockDim.x + threadIdx.x) % (N_VARS * N_VARS);
  //const int c = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS;
  //printf("Row %d, r %d\n", i, r);
  // local value of jacobian block according to rowblockvalid
  value_t jac = 0;
  value_t jac_offd_flux = 0;
  value_t jac_diag[N_VARS];
  // local value of R, V and U
  value_t r_val = 0, v_val, u_val;

  if (i > n_blocks - 1)
    return;

  for (index_t c = 0; c < N_VARS; c++)
  {
    //printf("Row %d, r %d, c%d\n", i, r, c);
    jac_diag[c] = 0;
  }

  // index of first entry for block i in CSR cols array
  index_t csr_idx_start = rows[i];
  // index of last entry for block i in CSR cols array
  index_t csr_idx_end = rows[i + 1];
  index_t conn_idx = csr_idx_start - i;
  u_val = U[i * N_VARS + r] * beta;

  for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++)
  {

    jac_offd_flux = 0;
    index_t j = cols[csr_idx];
    if (i != j)
    {
      // process offdiag

      value_t p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];
      value_t tran_dt = tran[conn_idx] * dt;
      value_t gamma_p_diff = tran_dt * p_diff;

      if (p_diff < 0)
      {
        for (index_t c = 0; c < N_VARS; c++)
        {
          v_val = V[j * N_VARS + c];
          jac_diag[c] += -gamma_p_diff * op_ders_arr[(i * N_OPS + FLUX_OP) * N_VARS + r * N_VARS + c];
          jac_offd_flux = 0;
          if (c == 0)
          {
            // pressure
            value_t flux = op_vals_arr[i * N_OPS + FLUX_OP + r];
            jac_offd_flux = -tran_dt * flux;
            jac_diag[c] -= jac_offd_flux;
          }
          // compute intermediate result
          r_val += jac_offd_flux * v_val;
        }
      }
      else
      {
        for (index_t c = 0; c < N_VARS; c++)
        {
          v_val = V[j * N_VARS + c];
          jac_offd_flux = -gamma_p_diff * op_ders_arr[(j * N_OPS + FLUX_OP) * N_VARS + r * N_VARS + c];

          if (c == 0)
          {
            // pressure
            value_t val = -tran_dt * op_vals_arr[j * N_OPS + FLUX_OP + r];
            jac_offd_flux += val;
            jac_diag[c] -= val;
          }

          // compute intermediate result
          r_val += jac_offd_flux * v_val;
        }
      }

      conn_idx++;
    }
    else
    {
      // process diag
      for (index_t c = 0; c < N_VARS; c++)
      {
        jac_diag[c] += PV[i] * op_ders_arr[(i * N_OPS + ACC_OP) * N_VARS + r * N_VARS + c];
      }
    }
  }

  // diag is now complete - take it into account
  for (index_t c = 0; c < N_VARS; c++)
  {
    r_val += jac_diag[c] * V[i * N_VARS + c];
  }
  R[i * N_VARS + r] = r_val * alpha + u_val;

  return;
};

// Contribute to SPMV for Jacobian rows corresponding to pre-computed well constraints (well heads, wh)
// R[wh] = Jacobian[wh] * V

template <uint8_t NC>
int engine_nc_gpu<NC>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                            std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                            sim_params *params_, timer_node *timer_)
{
  engine_base_gpu::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);

    return 0;
}

template <uint8_t NC>
int engine_nc_gpu<NC>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
  timer->node["jacobian assembly"].node["kernel"].start_gpu();
  assemble_jacobian_array_kernel3<NC, N_VARS, P_VAR, N_OPS, ACC_OP, FLUX_OP>
      KERNEL_1D(mesh->n_blocks, N_VARS * N_VARS, ASSEMBLY_N_VARS_N_VARS_BLOCK_SIZE)(mesh->n_blocks, dt,
                                                                                    X_d, RHS_d,
                                                                                    jacobian->rows_ptr_d, jacobian->cols_ind_d, jacobian->values_d, jacobian->diag_ind_d,
                                                                                    op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                                                    mesh_tran_d, PV_d);
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
  value_t r_val = 0, v_val;

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
  value_t r_val = 0, v_val, u_val;

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

// calc r_d = Jacobian * v_d
template <uint8_t NC>
int engine_nc_gpu<NC>::matrix_vector_product_d0(const value_t *v_d, value_t *r_d)
{
  assemble_jacobian_array_kernel4_spmv0<NC, N_VARS, P_VAR, N_OPS, ACC_OP, FLUX_OP>
      KERNEL_1D(mesh->n_blocks, N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(mesh->n_blocks, dt,
                                                                    X_d,
                                                                    Jacobian->rows_ptr_d, Jacobian->cols_ind_d,
                                                                    op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                                    mesh_tran_d, PV_d, v_d, r_d);

  // lay SPMV result from correct well constraint jacobian over r_d
  jacobian_wells_spmv0<N_VARS>
      KERNEL_1D(wells.size(), N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(wells.size(), jac_well_head_idxs_d,
                                                                  Jacobian->rows_ptr_d, Jacobian->cols_ind_d, jac_wells_d, v_d, r_d);

  return 0;
}

// calc r_d = alpha * Jacobian * v_d + beta * r_d
template <uint8_t NC>
int engine_nc_gpu<NC>::calc_lin_comb_d(value_t alpha, value_t beta, value_t *u_d, value_t *v_d, value_t *r_d)
{
  assemble_jacobian_array_kernel4_lincomb<NC, N_VARS, P_VAR, N_OPS, ACC_OP, FLUX_OP>
      KERNEL_1D(mesh->n_blocks, N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(mesh->n_blocks, dt,
                                                                    X_d,
                                                                    Jacobian->rows_ptr_d, Jacobian->cols_ind_d,
                                                                    op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                                    mesh_tran_d, PV_d,
                                                                    alpha, beta, u_d, v_d, r_d);
  // lay SPMV result from correct well constraint jacobian over r_d
  jacobian_wells_lincomb<N_VARS>
      KERNEL_1D(wells.size(), N_VARS, ASSEMBLY_N_VARS_BLOCK_SIZE)(wells.size(), jac_well_head_idxs_d,
                                                                  Jacobian->rows_ptr_d, Jacobian->cols_ind_d, jac_wells_d, alpha, beta, u_d, v_d, r_d);
  return 0;
}

template <uint8_t NC>
int engine_nc_gpu<NC>::test_spmv(int n_times, int kernel_number, int dump_result)
{
  cudaMemset(RHS_d, 0, sizeof(double) * Jacobian->n_row_size * mesh->n_blocks);

  timer->node["test_spmv"].timer = 0;

  if (kernel_number == 0)
  {
    timer->node["test_spmv"].start_gpu();
    for (int i = 0; i < n_times; i++)
    {
      Jacobian->matrix_vector_product_d(Xn_d, RHS_d);
    }
    timer->node["test_spmv"].stop_gpu();
  }
  else if (kernel_number == 1)
  {
    Jacobian->convert_to_ELL();
    timer->node["test_spmv"].start_gpu();
    for (int i = 0; i < n_times; i++)
    {
      Jacobian->matrix_vector_product_d_ell(Xn_d, RHS_d);
    }
    timer->node["test_spmv"].stop_gpu();
  }
  else if (kernel_number == 2)
  {
    // for other kernels, SPMV is done based on Jac obtained from test_assembly
    // in tets_assembly, dt is strictly 1
    // so set it also to 1 here to match SMPV results
    dt = 1;
    timer->node["test_spmv"].start_gpu();
    for (int i = 0; i < n_times; i++)
    {
      matrix_vector_product_d0(Xn_d, RHS_d);
    }
    timer->node["test_spmv"].stop_gpu();
  }
  else if (kernel_number == 3)
  {
    // for other kernels, SPMV is done based on Jac obtained from test_assembly
    // in tets_assembly, dt is strictly 1
    // so set it also to 1 here to match SMPV results
    dt = 1;
    timer->node["test_spmv"].start_gpu();
    for (int i = 0; i < n_times; i++)
    {
      matrix_vector_product_d(Xn_d, RHS_d);
    }
    timer->node["test_spmv"].stop_gpu();
  }
  else if (kernel_number == 4)
  {
    // for other kernels, SPMV is done based on Jac obtained from test_assembly
    // in tets_assembly, dt is strictly 1
    // so set it also to 1 here to match SMPV results
    dt = 1;
    timer->node["test_spmv"].start_gpu();
    for (int i = 0; i < n_times; i++)
    {
      calc_lin_comb_d(1.3, -2.3, Xn_d, Xn_d, RHS_d);
    }
    timer->node["test_spmv"].stop_gpu();
  }
  else if (kernel_number == 5)
  {
    // for other kernels, SPMV is done based on Jac obtained from test_assembly
    // in tets_assembly, dt is strictly 1
    // so set it also to 1 here to match SMPV results
    dt = 1;
    timer->node["test_spmv"].start_gpu();
    for (int i = 0; i < n_times; i++)
    {
      Jacobian->calc_lin_comb_d(1.3, -2.3, Xn_d, Xn_d, RHS_d);
    }
    timer->node["test_spmv"].stop_gpu();
  }
  printf("Average SPMV kernel %d: %e sec\n", kernel_number, timer->node["test_spmv"].get_timer_gpu() / n_times);

  if (dump_result)
  {
    char filename[1024];
    int status;
#ifdef __GNUC__
    char *res = abi::__cxa_demangle(typeid(*this).name(), NULL, NULL, &status);
#else
    char *res = "test";
#endif
    sprintf(filename, "%s_SPMV_%d.vec", res, kernel_number);
    copy_data_to_host(RHS, RHS_d, Jacobian->n_row_size * mesh->n_blocks);
    write_vector_to_file(filename, RHS);
  }
  return 0;
}

template<uint8_t NC>
int
engine_nc_gpu<NC>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};

template struct recursive_instantiator_nc<engine_nc_gpu, 2, 10>;

