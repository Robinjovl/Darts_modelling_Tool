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

#include "mech/engine_elasticity_cpu.hpp"
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

template <uint8_t ND>
int engine_elasticity_cpu<ND>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
									std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
									sim_params *params_, timer_node *timer_)
{
	output_counter = 0;
	newton_update_coefficient = 1.0;
	USE_CALCULATED_FLUX = false;
	init_base(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);
	this->expose_jacobian();
	return 0;
}
template <uint8_t ND>
int engine_elasticity_cpu<ND>::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
										 std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
										 sim_params *params_, timer_node *timer_)
{
	time_t rawtime;
	struct tm *timeinfo;
	char buffer[1024];

	mesh = mesh_;
	wells = well_list_;
	acc_flux_op_set_list = acc_flux_op_set_list_;
	params = params_;
	timer = timer_;

	// Instantiate Jacobian
	if (!Jacobian)
	{
		Jacobian = new csr_matrix<N_VARS>;
		Jacobian->type = MATRIX_TYPE_CSR;
	}

	// figure out if this is GPU engine from its name.
	int is_gpu_engine = engine_name.find(" GPU ") != std::string::npos;

	// allocate Jacobian
	// if (!is_gpu_engine)
	{
		// for CPU engines we need full init
		(static_cast<csr_matrix<N_VARS> *>(Jacobian))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_links);
	}
	// else
	// {
	//   // for GPU engines we need only structure - rows_ptr and cols_ind
	//   // they are filled on CPU and later copied to GPU
	//   (static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_struct(mesh_->n_blocks, mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);
	// }
#ifdef WITH_GPU
	if (params->linear_type >= params->GPU_GMRES_CPR_AMGX_ILU)
	{
		(static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_device(mesh_->n_blocks, mesh_->n_links);
	}
#endif

	// create linear solver
	if (!linear_solver)
	{
		switch (params->linear_type)
		{
		case sim_params::CPU_GMRES_CPR_AMG:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linsolv_iface *cpr = new linsolv_bos_cpr<N_VARS>;
			cpr->set_prec(new linsolv_bos_amg<1>);
			linear_solver->set_prec(cpr);
			break;
		}
#ifdef _WIN32
#if 0 // can be enabled if amgdll.dll is available \
	  // since we compile PIC code, we cannot link existing static library, which was compiled withouf fPIC flag.
		case sim_params::CPU_GMRES_CPR_AMG1R5:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linsolv_iface *cpr = new linsolv_bos_cpr<N_VARS>;
			cpr->set_prec(new linsolv_amg1r5<1>);
			linear_solver->set_prec(cpr);
			break;
		}
#endif
#endif //_WIN32
		case sim_params::CPU_GMRES_ILU0:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linear_solver->set_prec(new linsolv_bos_bilu0<N_VARS>);
			break;
		}
		case sim_params::CPU_SUPERLU:
		{
			linear_solver = new linsolv_superlu<N_VARS>;
			break;
		}

#ifdef WITH_GPU
		case sim_params::GPU_GMRES_CPR_AMG:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			linsolv_iface *cpr = new linsolv_bos_cpr_gpu<N_VARS>;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 0;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 0;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 1;
			cpr->set_prec(new linsolv_bos_amg<1>);
			linear_solver->set_prec(cpr);
			break;
		}
#ifdef WITH_AIPS
		case sim_params::GPU_GMRES_CPR_AIPS:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			linsolv_iface *cpr = new linsolv_bos_cpr_gpu<N_VARS>;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 1;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 1;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 0;

			int n_terms = 10;
			bool print_radius = false;
			int aips_type = 2; // thomas_structure
			bool print_structure = false;
			if (params->linear_params.size() > 0)
			{
				n_terms = params->linear_params[0];
				if (params->linear_params.size() > 1)
				{
					print_radius = params->linear_params[1];
					if (params->linear_params.size() > 2)
					{
						aips_type = params->linear_params[2];
						if (params->linear_params.size() > 3)
						{
							print_structure = params->linear_params[3];
						}
					}
				}
			}
			cpr->set_prec(new linsolv_aips<1>(n_terms, print_radius, aips_type, print_structure));
			linear_solver->set_prec(cpr);
			break;
		}
#endif //WITH_AIPS
		case sim_params::GPU_GMRES_CPR_AMGX_ILU:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			linsolv_iface *cpr = new linsolv_bos_cpr_gpu<N_VARS>;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 1;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 1;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 0;

			cpr->set_p_system_prec(new linsolv_amgx<1>(device_num));
			// set full system prec
			cpr->set_prec(new linsolv_cusparse_ilu<N_VARS>);
			linear_solver->set_prec(cpr);
			break;
		}
#ifdef WITH_ADGPRS_NF
		case sim_params::GPU_GMRES_CPR_NF:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			linsolv_iface *cpr = new linsolv_bos_cpr_gpu<N_VARS>;
			// NF was initially created for CPU-based solver, so keeping unnesessary GPU->CPU->GPU copies so far for simplicity
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 0;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 0;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 1;

			int nx, ny, nz;
			int n_colors = 4;
			int coloring_scheme = 3;
			bool is_ordering_reversed = true;
			bool is_factorization_twisted = true;
			if (params->linear_params.size() < 3)
			{
				printf("Error: Missing nx, ny, nz parameters, required for NF solver\n");
				exit(-3);
			}

			nx = params->linear_params[0];
			ny = params->linear_params[1];
			nz = params->linear_params[2];
			if (params->linear_params.size() > 3)
			{
				n_colors = params->linear_params[3];
				if (params->linear_params.size() > 4)
				{
					coloring_scheme = params->linear_params[4];
					if (params->linear_params.size() > 5)
					{
						is_ordering_reversed = params->linear_params[5];
						if (params->linear_params.size() > 6)
						{
							is_factorization_twisted = params->linear_params[6];
						}
					}
				}
			}

			cpr->set_prec(new linsolv_adgprs_nf<1>(nx, ny, nz, params->global_actnum, n_colors, coloring_scheme, is_ordering_reversed, is_factorization_twisted));
			linear_solver->set_prec(cpr);
			break;
		}
#endif //WITH_ADGPRS_NF
		case sim_params::GPU_GMRES_ILU0:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);

			break;
		}
#endif
		}
	}

	n_vars = get_n_vars();
	n_ops = get_n_ops();
	nc = get_n_comps();
	z_var = get_z_var();

	X_init.resize(n_vars * mesh->n_blocks);
	fluxes.resize(N_VARS * mesh->n_conns);
	PV.resize(mesh->n_blocks);
	RV.resize(mesh->n_blocks);

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		for (uint8_t j = 0; j < ND_; j++)
		{
			X_init[n_vars * i + j] = mesh->displacement[n_vars * i + j];
		}
		PV[i] = mesh->volume[i] * mesh->poro[i];
		RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
	}

	op_vals_arr.resize(n_ops * mesh->n_blocks);
	op_ders_arr.resize(n_ops * n_vars * mesh->n_blocks);

	t = 0;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	stat = sim_stat();

	print_header();

	//acc_flux_op_set->init_timer_node(&timer->node["jacobian assembly"].node["interpolation"]);

	// initialize jacobian structure
	init_jacobian_structure_mpsa(Jacobian);

#ifdef WITH_GPU
	if (params->linear_type >= sim_params::GPU_GMRES_CPR_AMG)
	{
		timer->node["jacobian assembly"].node["send_to_device"].start();
		Jacobian->copy_struct_to_device();
		timer->node["jacobian assembly"].node["send_to_device"].stop();
	}
#endif

	linear_solver->init_timer_nodes(&timer->node["linear solver setup"], &timer->node["linear solver solve"]);
	// initialize linear solver
	linear_solver->init(Jacobian, params->max_i_linear, params->tolerance_linear);

	//Xn.resize (n_vars * mesh->n_blocks);
	RHS.resize(n_vars * mesh->n_blocks);
	dX.resize(n_vars * mesh->n_blocks);

	sprintf(buffer, "\nSTART SIMULATION\n-------------------------------------------------------------------------------------------------------------\n");
	std::cout << buffer << std::flush;

	// let wells initialize their state
	for (ms_well *w : wells)
	{
		w->initialize_control(X_init);
	}

	Xn = X = X_init;
	dt = params->first_ts;
	prev_usual_dt = dt;

	// initialize arrays for every operator set
	block_idxs.resize(acc_flux_op_set_list.size());
	op_axis_min.resize(acc_flux_op_set_list.size());
	op_axis_max.resize(acc_flux_op_set_list.size());

	// initialize arrays for every operator set

	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		block_idxs[r].clear();
		op_axis_min[r].resize(n_vars);
		op_axis_max[r].resize(n_vars);
		for (int j = 0; j < n_vars; j++)
		{
			op_axis_min[r][j] = acc_flux_op_set_list[r]->get_axis_min(j);
			op_axis_max[r][j] = acc_flux_op_set_list[r]->get_axis_max(j);
		}
	}

	// create a block list for every operator set
	index_t idx = 0;
	for (auto op_region : mesh->op_num)
	{
		block_idxs[op_region].emplace_back(idx++);
	}

	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
		acc_flux_op_set_list[r]->evaluate_with_derivatives(X, block_idxs[r], op_vals_arr, op_ders_arr);
	op_vals_arr_n = op_vals_arr;

	time_data.clear();
	time_data_report.clear();

	return 0;
}

template <uint8_t ND>
int engine_elasticity_cpu<ND>::init_jacobian_structure_mpsa(csr_matrix_base *jacobian)
{
	const char n_vars = get_n_vars();

	// init Jacobian structure
	index_t *rows_ptr = jacobian->get_rows_ptr();
	index_t *diag_ind = jacobian->get_diag_ind();
	index_t *cols_ind = jacobian->get_cols_ind();
	index_t *row_thread_starts = jacobian->get_row_thread_starts();

	index_t n_blocks = mesh->n_blocks;
	std::vector<index_t> &block_m = mesh->block_m;
	std::vector<index_t> &block_p = mesh->block_p;

	rows_ptr[0] = 0;
	memset(diag_ind, -1, n_blocks * sizeof(index_t)); // t_long <-----> index_t
	for (index_t i = 0; i < n_blocks; i++)
	{
		const auto &cur = mesh->cell_stencil[i];
		rows_ptr[i + 1] = rows_ptr[i] + cur.size();
		std::copy_n(cur.data(), cur.size(), cols_ind + rows_ptr[i]);
		diag_ind[i] = rows_ptr[i] + index_t(find(cur.begin(), cur.end(), i) - cur.begin());
	}

	return 0;
}

template <uint8_t ND>
int engine_elasticity_cpu<ND>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	// We need extended connection list for that with all connections for each block

	index_t n_blocks = mesh->n_blocks;
	index_t n_conns = mesh->n_conns;
	//std::vector <value_t> &tran = mesh->tran;
	value_t *Jac = jacobian->get_values();
	index_t *diag_ind = jacobian->get_diag_ind();
	index_t *rows = jacobian->get_rows_ptr();
	index_t *cols = jacobian->get_cols_ind();
	index_t *row_thread_starts = jacobian->get_row_thread_starts();

	const index_t *block_m = mesh->block_m.data();
	const index_t *stencil = mesh->stencil.data();
	const index_t *offset = mesh->offset.data();
	const value_t *tran = mesh->tran.data();
	const value_t *bc = mesh->bc.data();
	const value_t *f = mesh->f.data();
	const value_t *predifined_fluxes = mesh->flux.data();

#ifdef _OPENMP
//#pragma omp parallel reduction (max: CFL_max)
#pragma omp parallel
	{
		int id = omp_get_thread_num();
		index_t start = row_thread_starts[id];
		index_t end = row_thread_starts[id + 1];
#else
	index_t start = 0;
	index_t end = n_blocks;
#endif //_OPENMP

		std::fill_n(Jac, ND_ * N_VARS * mesh->n_links, 0.0);
		std::fill(RHS.begin(), RHS.end(), 0.0);
		std::fill(fluxes.begin(), fluxes.end(), 0.0);

		index_t j, diag_idx, jac_idx = 0, conn_id = 0, st_id = 0, conn_st_id = 0;
		bool isSet = false;
		value_t p_diff;

		for (index_t i = start; i < end; ++i)
		{
			// index of diagonal block entry for block i in CSR values array
			diag_idx = N_VARS_SQ * diag_ind[i];
			// index of first entry for block i in CSR cols array
			index_t csr_idx_start = rows[i];
			// index of last entry for block i in CSR cols array
			index_t csr_idx_end = rows[i + 1];
			// index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
			//index_t conn_idx = csr_idx_start - i;

			//jac_idx = N_VARS_SQ * csr_idx_start;
			for (; block_m[conn_id] == i && conn_id < n_conns; conn_id++)
			{
				// st_id -- index of cell in full (jacobian) stencil
				// conn_st_id -- index of cell in this particular connection
				conn_st_id = offset[conn_id];
				// Inner contribution
				for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + 1]; st_id++)
				{
					if (stencil[conn_st_id] == cols[st_id])
					{
						// N_VARS_SQ = N_VARS * N_VARS per link
						for (uint8_t d = 0; d < N_VARS; d++)
						{
							for (uint8_t v = 0; v < N_VARS; v++)
							{
								fluxes[conn_id * N_VARS + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + v] * X[stencil[conn_st_id] * N_VARS + v];
								Jac[st_id * N_VARS_SQ + d * N_VARS + v] += op_vals_arr[i * N_OPS + d] * tran[conn_st_id * N_VARS_SQ + d * N_VARS + v];
							}
						}
						conn_st_id++;
					}
				}
				// Boundary contribution
				for (; conn_st_id < offset[conn_id + 1]; conn_st_id++)
				{
					if (stencil[conn_st_id] >= n_blocks)
					{
						int idx = (ND_ + 3) * (stencil[conn_st_id] - n_blocks) + 3;
						const value_t *cur_bc = &bc[idx];
						for (uint8_t d = 0; d < N_VARS; d++)
						{
							for (uint8_t v = 0; v < N_VARS; v++)
							{
								fluxes[conn_id * N_VARS + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + v] * cur_bc[v];
							}
						}
					}
				}

				if (USE_CALCULATED_FLUX)
				{
					for (uint8_t d = 0; d < N_VARS; d++)
					{
						fluxes[conn_id * N_VARS + d] = predifined_fluxes[N_VARS * conn_id + d];
					}
				}

				for (uint8_t d = 0; d < N_VARS; d++)
				{
					RHS[i * N_VARS + d] += op_vals_arr[i * N_OPS + d] * fluxes[conn_id * N_VARS + d];
				}
			}

			for (uint8_t d = 0; d < N_VARS; d++)
			{
				RHS[i * N_VARS + d] += op_vals_arr[i * N_OPS + d] * f[i * N_VARS + d];
			}
		}

#ifdef _OPENMP
	}
#endif

	for (ms_well *w : wells)
	{
		value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
		w->add_to_jacobian(dt, X, jac_well_head, RHS);
	}
	return 0;
};

template <uint8_t ND>
int engine_elasticity_cpu<ND>::assemble_linear_system(value_t deltat)
{
	newton_update_coefficient = 1.0;

	// switch constraints if needed
	timer->node["jacobian assembly"].start();
	for (ms_well *w : wells)
	{
		w->check_constraints(deltat, X);
	}

	// evaluate all operators and their derivatives
	timer->node["jacobian assembly"].node["interpolation"].start();

	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		int result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X, block_idxs[r], op_vals_arr, op_ders_arr);
		if (result < 0)
			return 0;
	}

	timer->node["jacobian assembly"].node["interpolation"].stop();

	// assemble jacobian
	assemble_jacobian_array(deltat, X, Jacobian, RHS);

#ifdef WITH_GPU
	if (params->linear_type >= sim_params::GPU_GMRES_CPR_AMG)
	{
		timer->node["jacobian assembly"].node["send_to_device"].start();
		Jacobian->copy_values_to_device();
		timer->node["jacobian assembly"].node["send_to_device"].stop();
	}
#endif

	timer->node["jacobian assembly"].stop();
	return 0;
}

template <uint8_t ND>
int engine_elasticity_cpu<ND>::solve_linear_equation()
{
	int r_code;
	char buffer[1024];
	linear_solver_error_last_dt = 0;
	timer->node["linear solver setup"].start();
	r_code = linear_solver->setup(Jacobian);
	timer->node["linear solver setup"].stop();

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

	if (1) //changed this to write jacobian to file!
	{
		//static_cast<csr_matrix<ND_>*>(Jacobian)->write_matrix_to_file_mm(("jac_nc_dar_" + std::to_string(output_counter++) + ".csr").c_str());
		//write_vector_to_file("jac_nc_dar.rhs", RHS);
		//write_vector_to_file("jac_nc_dar.sol", dX);
		//apply_newton_update(deltat);
		//write_vector_to_file("X_nc_dar", X);
		//write_vector_to_file("Xn_nc_dar", Xn);
		//std::vector<value_t> buf(RHS.size(), 0.0);
		//Jacobian->matrix_vector_product(dX.data(), buf.data());
		//std::transform(buf.begin(), buf.end(), RHS.begin(), buf.begin(), std::minus<double>());
		//write_vector_to_file("diff", buf);
		//exit(0);
		//return 0;
	}

	timer->node["linear solver solve"].start();
	r_code = linear_solver->solve(&RHS[0], &dX[0]);
	timer->node["linear solver solve"].stop();

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

template <uint8_t ND>
void engine_elasticity_cpu<ND>::write_matrix()
{
        #ifndef OPENDARTS_LINEAR_SOLVERS
        static_cast<csr_matrix<ND_> *>(Jacobian)->write_matrix_to_file_mm("jac.csr");
        #endif  // OPENDARTS_LINEAR_SOLVERS
	write_vector_to_file("jac_nc_dar.rhs", RHS);
	write_vector_to_file("jac_nc_dar.sol", dX);
}

template <uint8_t ND>
double engine_elasticity_cpu<ND>::calc_newton_residual_L2()
{
	double residual = 0;
	std::vector<value_t> res(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);

	for (int i = 0; i < mesh->n_res_blocks; i++)
	{
		for (int c = 0; c < n_vars; c++)
		{
			res[c] += RHS[i * n_vars + c] * RHS[i * n_vars + c];
			norm[c] += op_vals_arr[i * N_OPS + c] * op_vals_arr[i * N_OPS + c];
		}
	}
	for (int c = 0; c < n_vars; c++)
	{
		residual = std::max(residual, sqrt(res[c] / norm[c]));
	}

	return residual;
}

template <uint8_t ND>
int engine_elasticity_cpu<ND>::apply_newton_update(value_t dt)
{
	// apply update
	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		for (uint8_t d = 0; d < ND_; d++)
			X[N_VARS * i + U_VAR + d] -= newton_update_coefficient * dX[N_VARS * i + U_VAR + d];
	}

	return 0;
}

template <uint8_t ND>
int engine_elasticity_cpu<ND>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};

template class engine_elasticity_cpu<3>;
