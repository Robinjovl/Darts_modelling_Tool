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
#include "csr_matrix.h"
#include "linsolv_iface.h"

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

engine_base_gpu::~engine_base_gpu()
{
  free_device_data(X_d);
  free_device_data(Xn_d);
  free_device_data(dX_d);
  free_device_data(RHS_d);
  free_device_data(RHS_wells_d);
  free_device_data(PV_d);
  free_device_data(mesh_tran_d);
  free_device_data(jac_wells_d);
  free_device_data(op_vals_arr_d);
  free_device_data(op_vals_arr_n_d);
  free_device_data(op_ders_arr_d);
  for (int op_region = 0; op_region < block_idxs.size(); op_region++)
  {
    free_device_data(block_idxs_d[op_region]);
  }
}

int engine_base_gpu::post_newtonloop(value_t deltat, value_t time)
{
	int converged = engine_base::post_newtonloop(deltat, time);
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

	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		int result = acc_flux_op_set_list[r]->evaluate_with_derivatives_d(block_idxs[r].size(), X_d, block_idxs_d[r], op_vals_arr_d, op_ders_arr_d);
		if (result < 0)
			return 0;
	}

	timer->node["jacobian assembly"].node["interpolation"].stop_gpu();

	// assemble jacobian
	assemble_jacobian_array(deltat, X, Jacobian, RHS);

	timer->node["jacobian assembly"].stop_gpu();

	timer->node["host<->device_overhead"].start_gpu();
	copy_data_to_host(RHS, RHS_d);
	copy_data_to_host(op_vals_arr, op_vals_arr_d);
	timer->node["host<->device_overhead"].stop_gpu();

	return 0;
}

int engine_base_gpu::solve_linear_equation()
{
	int r_code;
	char buffer[1024];
	linear_solver_error_last_dt = 0;

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
		// use class property to save error state from linear solver
		// this way it will work for both C++ and python newton loop
		//Jacobian->write_matrix_to_file("jac_linear_setup_fail.csr");
		linear_solver_error_last_dt = 1;
		return linear_solver_error_last_dt;
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
    copy_data_to_host(Jacobian->values, Jacobian->values_d, Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
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

	if (r_code)
	{
		sprintf(buffer, "ERROR: Linear solver solve returned %d \n", r_code);
		std::cout << buffer << std::flush;
		// use class property to save error state from linear solver
		// this way it will work for both C++ and python newton loop
		linear_solver_error_last_dt = 2;
		return linear_solver_error_last_dt;
	}
	else
	{
		sprintf(buffer, "\t #%d (%.4e, %.4e): lin %d (%.1e)\n", n_newton_last_dt + 1, newton_residual_last_dt,
			well_residual_last_dt, linear_solver->get_n_iters(), linear_solver->get_residual());
		std::cout << buffer << std::flush;
		n_linear_last_dt += linear_solver->get_n_iters();
	}

	return 0;
}

int engine_base_gpu::apply_newton_update(value_t dt)
{
	engine_base::apply_newton_update(dt);

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

  if (max_ratio > params->newton_params[0])
  {
    std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
    for (size_t i = 0; i < n_vars_total; i++)
      dX[i] *= params->newton_params[0] / max_ratio;
  }
}

void engine_base_gpu::apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
  value_t max_dx = params->newton_params[0];
  value_t ratio, dx;
  index_t n_corrected = 0;

  for (int i = 0; i < mesh->n_blocks; i++)
  {
    ratio = 1.0;
    old_z[nc - 1] = 1.0;
    new_z[nc - 1] = 1.0;
    for (int j = 0; j < nc - 1; j++)
    {
      old_z[j] = X[i * n_vars + j + z_var];
      old_z[nc - 1] -= old_z[j];
      new_z[j] = old_z[j] - dX[i * n_vars + j + z_var];
      new_z[nc - 1] -= new_z[j];
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
      for (int j = z_var; j < z_var + nc - 1; j++)
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
  cudaMemset(Jacobian->values_d, 0, sizeof(double) * Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
  cudaMemset(RHS_d, 0, sizeof(double) * Jacobian->n_row_size * mesh->n_blocks);
  //copy_data_to_host(Jacobian->values, Jacobian->values_d, Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
  timer->node["jacobian assembly"].start_gpu();

  // evaluate all operators and their derivatives
  timer->node["jacobian assembly"].node["interpolation"].start_gpu();
  for (int i = 0; i < n_times; i++)
  {
    for (int r = 0; r < acc_flux_op_set_list.size(); r++)
    {
      int result = acc_flux_op_set_list[r]->evaluate_with_derivatives_d(block_idxs[r].size(), X_d, block_idxs_d[r], op_vals_arr_d, op_ders_arr_d);
      if (result < 0)
        return 0;
    }
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
    copy_data_to_host(Jacobian->values, Jacobian->values_d, Jacobian->n_row_size * Jacobian->n_row_size * Jacobian->rows_ptr[mesh->n_blocks]);
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

  // CSR
  Jacobian->convert_to_ELL();
  // Now ELL
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
