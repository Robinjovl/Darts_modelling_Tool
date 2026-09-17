#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#ifdef __GNUC__
#include <cxxabi.h>
#endif

#include "engine_base_gpu.h"
#ifdef OPENDARTS_LINEAR_SOLVERS
#include "csr_matrix.hpp"
#include "linsolv_iface.hpp"
#else
#include "csr_matrix.h"
#include "linsolv_iface.h"
#endif

// use efficien reduction routine for future norm calculation
// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, unsigned int tid) {
// if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
// if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
// if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
// if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
// if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
// if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
// }

// template <unsigned int blockSize>
// __global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
// extern __shared__ int sdata[];
// unsigned int tid = threadIdx.x;
// unsigned int i = blockIdx.x*(blockSize*2) + tid;
// unsigned int gridSize = blockSize*2*gridDim.x;
// sdata[tid] = 0;
// while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
// __syncthreads();
// if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
// if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
// if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
// if (tid < 32) warpReduce(sdata, tid);
// if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }

namespace
{
// mirrors the file-local helper in engine_base.cpp
inline value_t safe_denominator_gpu(value_t denom)
{
  value_t abs_denom = std::fabs(denom);
  return abs_denom > value_t(0) ? abs_denom : std::numeric_limits<value_t>::min();
}

constexpr int NORM_MAX_VARS = 16;
constexpr int NORM_BLOCK = 256;

// per-variable sums over reservoir blocks: acc[c] = sum RHS^2, acc[n_vars+c] = sum (PV*op_vals)^2
__global__ void newton_residual_l2_kernel(const value_t *RHS, const value_t *PV, const value_t *op_vals,
                                          const index_t n_res_blocks, const int n_vars, const int n_ops,
                                          value_t *acc)
{
  value_t res2[NORM_MAX_VARS], norm2[NORM_MAX_VARS];
  for (int c = 0; c < n_vars; c++)
    res2[c] = norm2[c] = 0;

  for (index_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_res_blocks;
       i += (index_t)gridDim.x * blockDim.x)
  {
    for (int c = 0; c < n_vars; c++)
    {
      const value_t r = RHS[i * n_vars + c];
      const value_t nm = PV[i] * op_vals[i * n_ops + c];
      res2[c] += r * r;
      norm2[c] += nm * nm;
    }
  }

  __shared__ value_t sh[NORM_BLOCK];
  for (int c = 0; c < 2 * n_vars; c++)
  {
    sh[threadIdx.x] = (c < n_vars) ? res2[c] : norm2[c - n_vars];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
        sh[threadIdx.x] += sh[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      atomicAdd(&acc[c], sh[0]);
    __syncthreads();
  }
}

__global__ void average_operator_kernel(const value_t *op_vals, const index_t n_res_blocks, const int n_vars,
                                        const int n_ops, value_t *acc)
{
  value_t sum[NORM_MAX_VARS];
  for (int c = 0; c < n_vars; c++)
    sum[c] = 0;
  for (index_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_res_blocks;
       i += (index_t)gridDim.x * blockDim.x)
    for (int c = 0; c < n_vars; c++)
      sum[c] += op_vals[i * n_ops + c];

  __shared__ value_t sh[NORM_BLOCK];
  for (int c = 0; c < n_vars; c++)
  {
    sh[threadIdx.x] = sum[c];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
        sh[threadIdx.x] += sh[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      atomicAdd(&acc[c], sh[0]);
    __syncthreads();
  }
}
} // namespace

engine_base_gpu::~engine_base_gpu()
{
  for (void *p : pinned_host_ptrs)
  {
    const cudaError_t unpin_status = cudaHostUnregister(p);
    if (unpin_status != cudaSuccess)
    {
      std::cerr << "WARNING: cudaHostUnregister failed: "
                << cudaGetErrorString(unpin_status) << " (" << unpin_status
                << ")" << std::endl;
      // Do not let a teardown-only failure poison AMGX's subsequent CUDA
      // last-error check in the engine_base destructor.
      (void)cudaGetLastError();
    }
  }
  pinned_host_ptrs.clear();
  free_device_data(residual_scratch_d);
  free_device_data(X_d);
  free_device_data(Xn_d);
  free_device_data(dX_d);
  free_device_data(RHS_d);
  free_device_data(Xop_d);
  free_device_data(PV_d);
  free_device_data(mesh_tran_d);
  free_device_data(jac_wells_d);
  free_device_data(op_vals_arr_d);
  free_device_data(op_vals_arr_n_d);
  free_device_data(op_ders_arr_d);
  free_device_data(op_ders_arr_ext_d);
  for (int op_region = 0; op_region < block_idxs.size(); op_region++)
  {
    free_device_data(block_idxs_d[op_region]);
  }
}

int engine_base_gpu::evaluate_operators_d()
{
  if (get_n_history() > 0)
  {
    build_Xop();
    copy_data_to_device(Xop, Xop_d);
    for (int r = 0; r < acc_flux_op_set_list.size(); r++)
    {
      // Zero-work op sets are skipped: an operator set with no assigned
      // blocks (e.g. a host-only customized operator, evaluated through
      // customize_block_idxs in engine_base) may be a CPU evaluator whose
      // device entry point is unimplemented. The CPU engine's host loop is
      // a natural no-op for an empty index list -- mirror that here.
      if (block_idxs[r].empty())
        continue;
      int result = acc_flux_op_set_list[r]->evaluate_with_derivatives_d(
          block_idxs[r].size(), Xop_d, block_idxs_d[r], op_vals_arr_d, op_ders_arr_ext_d);
      if (result < 0)
        return result;
    }

    copy_data_to_host(op_ders_arr_ext, op_ders_arr_ext_d);
    project_xop_ders();
    copy_data_to_device(op_ders_arr, op_ders_arr_d);
    return 0;
  }

  for (int r = 0; r < acc_flux_op_set_list.size(); r++)
  {
    if (block_idxs[r].empty())
      continue; // see the history-loop note: host-only op sets have no blocks
    int result = acc_flux_op_set_list[r]->evaluate_with_derivatives_d(
        block_idxs[r].size(), X_d, block_idxs_d[r], op_vals_arr_d, op_ders_arr_d);
    if (result < 0)
      return result;
  }
  return 0;
}

void engine_base_gpu::sync_host_data_for_accepted_step()
{
	// The accepted-step path consumes host op_vals_arr (ms_well::calc_rates,
	// FIPS); the per-assembly host mirror is gone, so refresh it here -- the
	// base calls this hook from inside the converged branch, so the refresh
	// follows the convergence verdict wherever that logic lives.
	sync_op_vals_to_host();
}

int engine_base_gpu::post_newtonloop(value_t deltat, value_t time, index_t converged_in)
{
	int converged = engine_base::post_newtonloop(deltat, time, converged_in);
	if (!converged)
	{
		copy_data_to_device(X, X_d);
	}
	else
	{
		copy_data_within_device(Xn_d, X_d, X.size());
		copy_data_within_device(op_vals_arr_n_d, op_vals_arr_d, op_vals_arr.size());
	}
	return converged;
}

int engine_base_gpu::assemble_linear_system(value_t deltat)
{
	// switch constraints if needed
	timer->node["jacobian assembly"].start_gpu();

	for (ms_well *w : wells)
	{
		w->check_constraints(deltat, X);
	}

	// evaluate all operators and their derivatives
	timer->node["jacobian assembly"].node["interpolation"].start_gpu();

	if (evaluate_operators_d() < 0)
		return 0;

	timer->node["jacobian assembly"].node["interpolation"].stop_gpu();

	// assemble jacobian
	assemble_jacobian_array(deltat, X, Jacobian, RHS);

	timer->node["jacobian assembly"].stop_gpu();

	timer->node["host<->device_overhead"].start_gpu();
	// Host RHS mirror stays (well assembly and apply_rhs_flux contract); the
	// 260MB-class op_vals_arr mirror is refreshed lazily instead: residual
	// norms reduce on the device, converged-step consumers trigger
	// sync_op_vals_to_host() from post_newtonloop.
	copy_data_to_host(RHS, RHS_d);
	timer->node["host<->device_overhead"].stop_gpu();

	return 0;
}

void engine_base_gpu::sync_op_vals_to_host()
{
	timer->node["host<->device_overhead"].start_gpu();
	copy_data_to_host(op_vals_arr, op_vals_arr_d);
	timer->node["host<->device_overhead"].stop_gpu();
}

double engine_base_gpu::calc_newton_residual_L2()
{
	if (n_vars > NORM_MAX_VARS)
	{
		sync_op_vals_to_host();
		return engine_base::calc_newton_residual_L2();
	}
	if (!residual_scratch_d)
		allocate_device_data(&residual_scratch_d, 2 * NORM_MAX_VARS);
	cudaMemset(residual_scratch_d, 0, 2 * n_vars * sizeof(value_t));
	const int n_blocks_launch =
		std::min<index_t>((mesh->n_res_blocks + NORM_BLOCK - 1) / NORM_BLOCK, 4096);
	newton_residual_l2_kernel<<<n_blocks_launch, NORM_BLOCK>>>(
		RHS_d, PV_d, op_vals_arr_d, mesh->n_res_blocks, n_vars, n_ops, residual_scratch_d);
	std::vector<value_t> acc(2 * n_vars);
	copy_data_to_host(acc.data(), residual_scratch_d, 2 * n_vars);

	double residual = 0;
	for (int c = 0; c < n_vars; c++)
		residual = std::max(residual, sqrt(acc[c] / safe_denominator_gpu(acc[n_vars + c])));
	return residual;
}

double engine_base_gpu::calc_newton_residual_L1()
{
	sync_op_vals_to_host();
	return engine_base::calc_newton_residual_L1();
}

double engine_base_gpu::calc_newton_residual_Linf()
{
	sync_op_vals_to_host();
	return engine_base::calc_newton_residual_Linf();
}

void engine_base_gpu::average_operator(std::vector<value_t> &av_op)
{
	if (n_vars > NORM_MAX_VARS)
	{
		sync_op_vals_to_host();
		engine_base::average_operator(av_op);
		return;
	}
	if (!residual_scratch_d)
		allocate_device_data(&residual_scratch_d, 2 * NORM_MAX_VARS);
	cudaMemset(residual_scratch_d, 0, n_vars * sizeof(value_t));
	const int n_blocks_launch =
		std::min<index_t>((mesh->n_res_blocks + NORM_BLOCK - 1) / NORM_BLOCK, 4096);
	average_operator_kernel<<<n_blocks_launch, NORM_BLOCK>>>(
		op_vals_arr_d, mesh->n_res_blocks, n_vars, n_ops, residual_scratch_d);
	std::vector<value_t> acc(n_vars);
	copy_data_to_host(acc.data(), residual_scratch_d, n_vars);
	for (int c = 0; c < n_vars; c++)
		av_op[c] = acc[c] / mesh->n_res_blocks;
}

int engine_base_gpu::solve_linear_equation()
{
	int r_code;
	char buffer[1024];
	last_linear_iters = 0;

	timer->node["linear solver setup"].start_gpu();
	if (params->assembly_kernel == 13)
	{
		r_code = linear_solver->setup(this);
	}
	else
	{
		r_code = linear_solver->setup(Jacobian);
	}
	timer->node["linear solver setup"].stop_gpu();

	if (r_code)
	{
		sprintf(buffer, "ERROR: Linear solver setup returned %d \n", r_code);
		std::cout << buffer << std::flush;
		return 1;
	}

	timer->node["linear solver solve"].start_gpu();
	r_code = linear_solver->solve(RHS_d, dX_d);
	timer->node["linear solver solve"].stop_gpu();

	timer->node["host<->device_overhead"].start_gpu();
	copy_data_to_host(dX, dX_d);
	timer->node["host<->device_overhead"].stop_gpu();

  if (print_linear_system) //changed this to write jacobian to file!
  {
    const std::string matrix_filename = "jac_nc_dar_" + std::to_string(output_counter) + ".csr";
#ifdef OPENDARTS_LINEAR_SOLVERS
    copy_data_to_host(Jacobian->get_values(), Jacobian->get_values_d(),
      Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->get_rows_ptr()[mesh->n_blocks]);
#else
    copy_data_to_host(Jacobian->values, Jacobian->values_d, Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
#endif
#ifdef OPENDARTS_LINEAR_SOLVERS
    Jacobian->export_matrix_to_file(matrix_filename, opendarts::linear_solvers::sparse_matrix_export_format::csr);
#else
    Jacobian->write_matrix_to_file_mm(matrix_filename.c_str());
#endif
    //Jacobian->write_matrix_to_file(("jac_nc_dar_" + std::to_string(output_counter) + ".csr").c_str());
    copy_data_to_host(RHS, RHS_d);
    write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".rhs", RHS);
    write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".sol", dX);
    output_counter++;
  }

	// Unified solve() convention: a POSITIVE code is "budget exhausted, iterate
	// usable" -- reported as engine status 3 for the nonlinear policy to act on.
	if (const int nc = classify_linear_solve_status(r_code); nc == 3)
		return 3;
	if (r_code)
	{
		sprintf(buffer, "ERROR: Linear solver solve returned %d \n", r_code);
		std::cout << buffer << std::flush;
		return 2;
	}
	else
	{
		last_linear_iters = linear_solver->get_n_iters();
		last_linear_residual = linear_solver->get_residual();
	}

	return 0;
}

int engine_base_gpu::apply_update(value_t dt)
{
	engine_base::apply_update(dt);

	timer->node["host<->device_overhead"].start_gpu();
	copy_data_to_device(X, X_d);
	timer->node["host<->device_overhead"].stop_gpu();

  return 0;
}

void engine_base_gpu::apply_global_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
  double max_ratio = 0;
  index_t n_vars_total = X.size();

  for (index_t i = 0; i < n_vars_total; i++)
  {
    if (fabs(X[i]) > 1e-4)
    {
      double ratio = fabs(dX[i]) / fabs(X[i]);
      max_ratio = (max_ratio < ratio) ? ratio : max_ratio;
    }
  }

  if (max_ratio > newton_chop_factor)
  {
    std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
    for (size_t i = 0; i < n_vars_total; i++)
      dX[i] *= newton_chop_factor / max_ratio;
  }
}

void engine_base_gpu::apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
  value_t max_dx = newton_chop_factor;
  value_t ratio, dx;
  index_t n_corrected = 0;

  for (int i = 0; i < mesh->n_blocks; i++)
  {
    ratio = 1.0;
    old_z[dependent_comp_idx] = 1.0;
    new_z[dependent_comp_idx] = 1.0;
    for (int j = 0; j < nc - 1; j++)
    {
      const uint8_t p = explicit_comp_idxs[j];
      old_z[p] = X[i * n_vars + j + z_var_idx];
      old_z[dependent_comp_idx] -= old_z[p];
      new_z[p] = old_z[p] - dX[i * n_vars + j + z_var_idx];
      new_z[dependent_comp_idx] -= new_z[p];
    }

    for (int j = 0; j < nc; j++)
    {
      dx = fabs(new_z[j] - old_z[j]);
      if (dx > 0.0001) // if update is not too small
      {
        ratio = std::min<value_t>(ratio, max_dx / dx); // update the ratio
      }
    }

    if (ratio < 1.0) // perform chopping if ratio is below 1.0
    {
      n_corrected++;
      for (int j = z_var_idx; j < z_var_idx + nc - 1; j++)
      {
        dX[i * n_vars + j] *= ratio;
      }
    }
  }
  if (n_corrected)
    std::cout << "Local chop applied in " << n_corrected << " block(s)" << std::endl;
}

int engine_base_gpu::test_assembly(int n_times, int kernel_number, int dump_jacobian_rhs)
{
  // timestep does not matter
  double deltat = 1;
  timer->node["jacobian assembly"].timer = 0;
  timer->node["jacobian assembly"].node["kernel"].timer = 0;
  timer->node["jacobian assembly"].node["interpolation"].timer = 0;

  // switch constraints if needed

  for (ms_well *w : wells)
  {
    w->check_constraints(deltat, X);
  }
  // reset Jacobian and RHS values for correct dump result
#ifdef OPENDARTS_LINEAR_SOLVERS
  cudaMemset(Jacobian->get_values_d(), 0, sizeof(double) * Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->get_rows_ptr()[mesh->n_blocks]);
#else
  cudaMemset(Jacobian->values_d, 0, sizeof(double) * Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
#endif
  cudaMemset(RHS_d, 0, sizeof(double) * Jacobian->n_row_size * mesh->n_blocks);
  //copy_data_to_host(Jacobian->values, Jacobian->values_d, Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
  timer->node["jacobian assembly"].start_gpu();

  // evaluate all operators and their derivatives
  timer->node["jacobian assembly"].node["interpolation"].start_gpu();
  for (int i = 0; i < n_times; i++)
  {
    if (evaluate_operators_d() < 0)
      return 0;
  }
  timer->node["jacobian assembly"].node["interpolation"].stop_gpu();
  for (int i = 0; i < n_times; i++)
  {
    // assemble jacobian

    assemble_jacobian_array(deltat, X, Jacobian, RHS);
  }
  timer->node["jacobian assembly"].stop_gpu();

  if (dump_jacobian_rhs)
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    copy_data_to_host(Jacobian->get_values(), Jacobian->get_values_d(),
      Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->get_rows_ptr()[mesh->n_blocks]);
#else
    copy_data_to_host(Jacobian->values, Jacobian->values_d, Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
#endif
    copy_data_to_host(RHS, RHS_d, Jacobian->n_row_size * mesh->n_blocks);
    char filename[1024];
    int status;
#ifdef __GNUC__
    char *res = abi::__cxa_demangle(typeid(*this).name(), NULL, NULL, &status);
#else
    char *res = "test";
#endif

    sprintf(filename, "%s_%d_jac.csr", res, kernel_number);
    Jacobian->write_matrix_to_file(filename);
    sprintf(filename, "%s_%d_rhs.vec", res, kernel_number);
    write_vector_to_file(filename, RHS);
  }

  printf("Average assembly %d: %e sec, interpolation %e sec, kernel %e\n", kernel_number, timer->node["jacobian assembly"].get_timer_gpu() / n_times,
         timer->node["jacobian assembly"].node["interpolation"].get_timer_gpu() / n_times,
         timer->node["jacobian assembly"].node["kernel"].get_timer_gpu() / n_times);
  //printf ("Average assembly kernel: %e sec\n", timer->node["test_assembly"].get_timer_gpu() / n_times);
  return 0;
}

int engine_base_gpu::test_spmv(int n_times, int kernel_number, int dump_result)
{
  // BCSR
  cudaMemset(RHS_d, 0, sizeof(double) * Jacobian->n_row_size * mesh->n_blocks);

  timer->node["test_spmv"].timer = 0;
  timer->node["test_spmv"].start_gpu();
  for (int i = 0; i < n_times; i++)
  {
    Jacobian->matrix_vector_product_d(Xn_d, RHS_d);
  }
  timer->node["test_spmv"].stop_gpu();
  printf("Average SPMV kernel: %e sec\n", timer->node["test_spmv"].get_timer_gpu() / n_times);
  if (dump_result)
  {
    char filename[1024];
    int status;
#ifdef __GNUC__
    char *res = abi::__cxa_demangle(typeid(*this).name(), NULL, NULL, &status);
#else
    char *res = "test";
#endif
    sprintf(filename, "%s_%d_bcsr.vec", res, kernel_number);
    copy_data_to_host(RHS, RHS_d, Jacobian->n_row_size * mesh->n_blocks);
    write_vector_to_file(filename, RHS);
  }

  // CSR / ELL benchmark path -- cuSPARSE removed the HYB/ELL format in
  // CUDA 11 and the open-source block_csr_matrix does not expose an ELL
  // variant either (matrix_vector_product_d_ell is a no-op fallback to the
  // block SpMV; convert_to_ELL is not on csr_matrix_base). Compile the
  // benchmark only for the legacy csr_matrix<N> path where the ELL hooks
  // exist; under OPENDARTS_LINEAR_SOLVERS the block SpMV above is the only
  // measurable kernel anyway.
#ifndef OPENDARTS_LINEAR_SOLVERS
  Jacobian->convert_to_ELL();
  cudaMemset(RHS_d, 0, sizeof(double) * Jacobian->n_row_size * mesh->n_blocks);
  timer->node["test_spmv"].timer = 0;
  timer->node["test_spmv"].start_gpu();
  for (int i = 0; i < n_times; i++)
  {
    Jacobian->matrix_vector_product_d_ell(Xn_d, RHS_d);
  }
  timer->node["test_spmv"].stop_gpu();
  printf("Average SPMV ELL kernel: %e sec\n", timer->node["test_spmv"].get_timer_gpu() / n_times);
  if (dump_result)
  {
    char filename[1024];
    int status;
#ifdef __GNUC__
    char *res = abi::__cxa_demangle(typeid(*this).name(), NULL, NULL, &status);
#else
    char *res = "test";
#endif
    sprintf(filename, "%s_ell.vec", res);
    copy_data_to_host(RHS, RHS_d, Jacobian->n_row_size * mesh->n_blocks);
    write_vector_to_file(filename, RHS);
  }
#endif // OPENDARTS_LINEAR_SOLVERS

  return 0;
}

// calc r_d += Jacobian * v_d
int engine_base_gpu::matrix_vector_product_d(const value_t *v_d, value_t *r_d)
{
  // TODO: Too difficult now to take into account correct well constraints (like in spmv0 and lincomb),
  // because r_d already has incorrect contributions
  // When well heads will all be gathered at the bottom of Jacobian, it`ll be easy to implement
  printf("matrix_vector_product_d is not implemented for matrix-free\n");

  return -1;
}

// calc r_d = Jacobian * v_d
int engine_base_gpu::matrix_vector_product_d0(const value_t *v_d, value_t *r_d)
{
  // TODO: Too difficult now to take into account correct well constraints (like in spmv0 and lincomb),
  // because r_d already has incorrect contributions
  // When well heads will all be gathered at the bottom of Jacobian, it`ll be easy to implement
  printf("matrix_vector_product_d0 is not implemented");

  return -1;
}

// calc r_d = Jacobian * v_d
int engine_base_gpu::calc_lin_comb_d(value_t alpha, value_t beta, value_t *u_d, value_t *v_d, value_t *r_d)
{
  // TODO: Too difficult now to take into account correct well constraints (like in spmv0 and lincomb),
  // because r_d already has incorrect contributions
  // When well heads will all be gathered at the bottom of Jacobian, it`ll be easy to implement
  printf("matrix_vector_product_d0 is not implemented");

  return -1;
}
