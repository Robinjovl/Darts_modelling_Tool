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

#include "engine_nc_nl_cpu.hpp"
#include "conn_mesh.h"
#include "mech/matrix.h"

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

template <uint8_t NC>
const std::string engine_nc_nl_cpu<NC>::AVG_MPFA = "AVG_MPFA";
template <uint8_t NC>
const std::string engine_nc_nl_cpu<NC>::NLTPFA = "NLTPFA";
template <uint8_t NC>
const std::string engine_nc_nl_cpu<NC>::NLMPFA = "NLMPFA";

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
							   std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
							   sim_params *params_, timer_node *timer_)
{
	init_base(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);
	appr_mode = NLTPFA;
	return 0;
}

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
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
		Jacobian->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
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
	if (params->linear_type >= params->GPU_GMRES_CPR_AMG)
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
#if 0 // can be enabled if amgdll.dll is available
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

	fluxes.resize(mesh->n_conns);
	std::fill_n(fluxes.begin(), fluxes.size(), 0.0);
	X_init.resize(n_vars * mesh->n_res_blocks);  // initialize only reservoir blocks with mesh->initial_state array
	PV.resize(mesh->n_blocks);
	RV.resize(mesh->n_blocks);
	old_z.resize(nc);
	new_z.resize(nc);
	FIPS.resize(nc);

	X_init = mesh->initial_state;
	X_init.resize(n_vars * mesh->n_blocks);
	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		PV[i] = mesh->volume[i] * mesh->poro[i];
		RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
	}

	op_vals_arr.resize(n_ops * (mesh->n_blocks + mesh->n_bounds));
	op_ders_arr.resize(n_ops * n_vars * (mesh->n_blocks + mesh->n_bounds));

	t = 0;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	stat = sim_stat();

	print_header();

	//acc_flux_op_set->init_timer_node(&timer->node["jacobian assembly"].node["interpolation"]);

	// initialize jacobian structure
	init_jacobian_structure_mpfa(Jacobian);

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
	for (index_t i = 0; i < mesh->n_bounds; i++)
	{
		block_idxs[mesh->op_num[0]].emplace_back(idx++);
	}

	extract_Xop();
	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
		acc_flux_op_set_list[r]->evaluate_with_derivatives(Xop, block_idxs[r], op_vals_arr, op_ders_arr);
	op_vals_arr_n = op_vals_arr;

	time_data.clear();
	time_data_report.clear();

	if (params->log_transform == 0)
	{
		min_zc = acc_flux_op_set_list[0]->get_axis_min(z_var) * params->obl_min_fac;
		max_zc = 1 - min_zc * params->obl_min_fac;
		//max_zc = acc_flux_op_set_list[0]->get_maxzc();
	}
	else if (params->log_transform == 1)
	{
		min_zc = exp(acc_flux_op_set_list[0]->get_axis_min(z_var)) * params->obl_min_fac; //log based composition
		max_zc = exp(acc_flux_op_set_list[0]->get_axis_max(z_var));						  //log based composition
	}

	return 0;
}

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::init_jacobian_structure_mpfa(csr_matrix_base *jacobian)
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

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::assemble_linear_system(value_t deltat)
{
	// switch constraints if needed
	timer->node["jacobian assembly"].start();
	for (ms_well *w : wells)
	{
		w->check_constraints(deltat, X);
	}

	// evaluate all operators and their derivatives
	timer->node["jacobian assembly"].node["interpolation"].start();

	extract_Xop();
	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		int result = acc_flux_op_set_list[r]->evaluate_with_derivatives(Xop, block_idxs[r], op_vals_arr, op_ders_arr);
		if (result < 0)
			return 0;
	}

	timer->node["jacobian assembly"].node["interpolation"].stop();

	// assemble jacobian
	if (appr_mode == AVG_MPFA)
		assemble_jacobian_array_avgmpfa(deltat, X, Jacobian, RHS);
	else if (appr_mode == NLTPFA)
		assemble_jacobian_array_nltpfa(deltat, X, Jacobian, RHS);
	else if (appr_mode == NLMPFA)
		assemble_jacobian_array_nlmpfa(deltat, X, Jacobian, RHS);

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

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::assemble_jacobian_array_avgmpfa(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	// We need extended connection list for that with all connections for each block
	index_t n_blocks = mesh->n_blocks;
	index_t n_res_blocks = mesh->n_res_blocks;
	index_t n_conns = mesh->n_conns;
	value_t *Jac = jacobian->get_values();
	index_t *diag_ind = jacobian->get_diag_ind();
	index_t *rows = jacobian->get_rows_ptr();
	index_t *cols = jacobian->get_cols_ind();
	index_t *row_thread_starts = jacobian->get_row_thread_starts();

	const index_t *block_m = mesh->block_m.data();
	const index_t *block_p = mesh->block_p.data();
	const index_t *stencil = mesh->stencil.data();
	const index_t *offset = mesh->offset.data();
	const value_t *tran = mesh->tran.data();
	const value_t *bc = mesh->bc.data();
	const value_t *rhs = mesh->rhs.data();
	const value_t *f = mesh->f.data();

	CFL_max = 0;

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

		index_t upwd, j, k, diag_idx, conn_id = 0, cur_conn_id, st_id = 0, jac_idx, conn_st_id = 0, upwd_jac_idx;
		value_t buf, flux;
		value_t mu[MATRIX];
		value_t CFL_in[NC], CFL_out[NC];
		value_t CFL_max_local = 0;

		ConnType connection_type;
		std::fill_n(Jac, mesh->n_links * N_VARS * N_VARS, 0.0);
		std::fill_n(fluxes.begin(), fluxes.size(), 0.0);
		std::fill_n(RHS.begin(), RHS.size(), 0.0);

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
				for (uint8_t v = 0; v < N_VARS; v++)
				{
					Jac[diag_idx + c * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];
				}
			}
			// index of first entry for block i in CSR cols array
			index_t csr_idx_start = rows[i];
			// index of last entry for block i in CSR cols array
			index_t csr_idx_end = rows[i + 1];

			for (; block_m[conn_id] == i && conn_id < n_conns; conn_id += connection_type)
			{
				// st_id -- index of cell in full (jacobian) stencil
				// conn_st_id -- index of cell in this particular connection
				if (i < n_res_blocks)
					connection_type = MATRIX;
				else
					connection_type = WELL;
				j = block_p[conn_id];
				if (j >= n_res_blocks && j < n_blocks)
					connection_type = WELL;

				// Calculate the flux
				for (cur_conn_id = conn_id; cur_conn_id < conn_id + connection_type; cur_conn_id++)
				{
					for (conn_st_id = offset[cur_conn_id]; conn_st_id < offset[cur_conn_id + 1]; conn_st_id++)
					{
						buf = (stencil[conn_st_id] < n_blocks) ? X[stencil[conn_st_id] * N_VARS + P_VAR] : bc[stencil[conn_st_id] - n_blocks];
						fluxes[cur_conn_id] += buf * tran[conn_st_id];
					}
					fluxes[cur_conn_id] += rhs[cur_conn_id];
				}

				// Weights
				if (connection_type == WELL)
				{
					mu[0] = 1.0;
				}
				else
				{
					mu[0] = mu[1] = 0.5;
				}
				// Total flux
				flux = 0.0;
				for (k = 0; k < connection_type; k++)
				{
					flux += dt * mu[k] * fluxes[conn_id + k];
				}

				if (flux >= 0)
				{
					upwd = i;
					upwd_jac_idx = diag_ind[i];
					for (uint8_t c = 0; c < NC; c++)
					{
						CFL_out[c] += flux * op_vals_arr[i * N_OPS + FLUX_OP + c];
					}
				}
				else
				{
					upwd = j;
					if (j > n_blocks)
						upwd_jac_idx = csr_idx_end;
					if (j < n_res_blocks)
					{
						for (uint8_t c = 0; c < NC; c++)
						{
							CFL_in[c] += -flux * op_vals_arr[j * N_OPS + FLUX_OP + c];
						}
					}
				}

				for (k = 0; k < connection_type; k++)
				{
					// Inner connections
					conn_st_id = offset[conn_id + k];
					for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + k + 1]; st_id++)
					{
						if (stencil[conn_st_id] == cols[st_id])
						{
							if (flux < 0 && cols[st_id] == j)
								upwd_jac_idx = st_id;

							jac_idx = N_VARS_SQ * st_id;
							for (uint8_t c = 0; c < NC; c++)
							{
								Jac[jac_idx + c * N_VARS] += mu[k] * tran[conn_st_id] * dt * op_vals_arr[upwd * N_OPS + FLUX_OP + c];
							}
							conn_st_id++;
						}
					}
				}

				for (uint8_t c = 0; c < NC; c++)
				{
					RHS[i * N_VARS + c] += flux * op_vals_arr[upwd * N_OPS + FLUX_OP + c]; // flux operators only
					if (upwd_jac_idx < csr_idx_end)
					{
						jac_idx = N_VARS_SQ * upwd_jac_idx;
						for (uint8_t v = 0; v < N_VARS; v++)
							Jac[jac_idx + c * N_VARS + v] += flux * op_ders_arr[(upwd * N_OPS + FLUX_OP + c) * N_VARS + v];
					}
				}
			}
			// calc CFL for reservoir cells, not connected with wells
			if (i < mesh->n_res_blocks)
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

			if (i < mesh->n_res_blocks)
			{
				for (uint8_t d = 0; d < N_VARS; d++)
				{
					RHS[i * N_VARS + d] += op_vals_arr[i * N_OPS + d] * f[i * N_VARS + d];
				}
			}
		}
#ifdef _OPENMP
#pragma omp critical
		{
			if (CFL_max < CFL_max_local)
				CFL_max = CFL_max_local;
		}
	}
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

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::assemble_jacobian_array_nltpfa(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	// We need extended connection list for that with all connections for each block
	index_t n_blocks = mesh->n_blocks;
	index_t n_res_blocks = mesh->n_res_blocks;
	index_t n_conns = mesh->n_conns;
	value_t *Jac = jacobian->get_values();
	index_t *diag_ind = jacobian->get_diag_ind();
	index_t *rows = jacobian->get_rows_ptr();
	index_t *cols = jacobian->get_cols_ind();
	index_t *row_thread_starts = jacobian->get_row_thread_starts();

	const index_t *block_m = mesh->block_m.data();
	const index_t *block_p = mesh->block_p.data();
	const index_t *stencil = mesh->stencil.data();
	const index_t *offset = mesh->offset.data();
	const value_t *tran = mesh->tran.data();
	const value_t *bc = mesh->bc.data();
	const value_t *rhs = mesh->rhs.data();
	const value_t *f = mesh->f.data();

	CFL_max = 0;

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

		index_t upwd, j, k, diag_idx, conn_id = 0, cur_conn_id, st_id = 0, jac_idx, conn_st_id = 0, conn_st, upwd_jac_idx, r_counter, cell_p_st_id;
		value_t buf;
		value_t flux, mu[MATRIX], R[MATRIX], dR[MATRIX][MATRIX], dmu[MATRIX][MATRIX][MATRIX];
		value_t CFL_in[NC], CFL_out[NC];
		value_t CFL_max_local = 0;

		ConnType connection_type;
		std::fill_n(Jac, mesh->n_links * N_VARS * N_VARS, 0.0);
		std::fill_n(fluxes.begin(), fluxes.size(), 0.0);
		std::fill_n(RHS.begin(), RHS.size(), 0.0);

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
				for (uint8_t v = 0; v < N_VARS; v++)
				{
					Jac[diag_idx + c * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];
				}
			}
			// index of first entry for block i in CSR cols array
			index_t csr_idx_start = rows[i];
			// index of last entry for block i in CSR cols array
			index_t csr_idx_end = rows[i + 1];

			for (; block_m[conn_id] == i && conn_id < n_conns; conn_id += connection_type)
			{
				// st_id -- index of cell in full (jacobian) stencil
				// conn_st_id -- index of cell in this particular connection
				if (i < n_res_blocks)
					connection_type = MATRIX;
				else
					connection_type = WELL;
				j = block_p[conn_id];
				if (j >= n_res_blocks && j < n_blocks)
					connection_type = WELL;

				// Calculate the flux & weights
				R[0] = R[1] = 0.0;
				if (connection_type == WELL) // well
				{
					for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
					{
						conn_st = stencil[conn_st_id];
						buf = (conn_st < n_blocks) ? X[conn_st * N_VARS + P_VAR] : bc[conn_st - n_blocks];
						fluxes[conn_id] += buf * tran[conn_st_id];
					}
					fluxes[conn_id] += rhs[conn_id];
				}
				else
				{
					r_counter = 0;
					dR[0][0] = dR[0][1] = 0.0;
					for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
					{
						conn_st = stencil[conn_st_id];
						buf = (conn_st < n_blocks) ? X[conn_st * N_VARS + P_VAR] : X[i * N_VARS + P_VAR];
						fluxes[conn_id] += buf * tran[conn_st_id];
						if (conn_st != i && conn_st != j)
						{
							dR[0][r_counter++] = -tran[conn_st_id];
							R[0] -= buf * tran[conn_st_id];
						}
					}
					r_counter = 0;
					dR[1][0] = dR[1][1] = 0.0;
					for (conn_st_id = offset[conn_id + 1]; conn_st_id < offset[conn_id + 2]; conn_st_id++)
					{
						conn_st = stencil[conn_st_id];
						buf = (conn_st < n_blocks) ? X[conn_st * N_VARS + P_VAR] : X[j * N_VARS + P_VAR];
						fluxes[conn_id + 1] += buf * tran[conn_st_id];
						if (conn_st != i && conn_st != j)
						{
							dR[1][r_counter++] = tran[conn_st_id];
							R[1] += buf * tran[conn_st_id];
						}
					}
				}
				// Weights
				if (connection_type == WELL)
				{
					mu[0] = 1.0;
					mu[1] = 0.0;
					dmu[0][0][0] = dmu[0][0][1] = dmu[0][1][0] = dmu[0][1][1] = 0.0;
					dmu[1][0][0] = dmu[1][0][1] = dmu[1][1][0] = dmu[1][1][1] = 0.0;
				}
				else
				{
					if (R[0] < 0 || R[1] < 0)
						double a = 33;
					assert(R[0] >= 0.0 && R[1] >= 0.0);
					// denominator
					buf = R[0] + R[1] + 2 * EQUALITY_TOLERANCE;
					// mu+
					mu[0] = (R[1] + EQUALITY_TOLERANCE) / buf;
					// dR+/dp
					dmu[0][0][0] = -mu[0] / buf * dR[0][0];
					dmu[0][0][1] = -mu[0] / buf * dR[0][1];
					// dR-/dp
					dmu[0][1][0] = dR[1][0] * (1.0 - mu[0]) / buf;
					dmu[0][1][1] = dR[1][1] * (1.0 - mu[0]) / buf;
					// mu-
					mu[1] = (R[0] + EQUALITY_TOLERANCE) / buf;
					// dR+/dp
					dmu[1][0][0] = dR[0][0] * (1.0 - mu[1]) / buf;
					dmu[1][0][1] = dR[0][1] * (1.0 - mu[1]) / buf;
					// dR-/dp
					dmu[1][1][0] = -mu[1] / buf * dR[1][0];
					dmu[1][1][1] = -mu[1] / buf * dR[1][1];

					assert(mu[0] >= 0.0 && mu[0] <= 1.0);
					assert(mu[1] >= 0.0 && mu[1] <= 1.0);
					assert(fabs(mu[0] + mu[1] - 1.0) < EQUALITY_TOLERANCE);
				}
				// Total flux
				flux = 0.0;
				for (k = 0; k < connection_type; k++)
				{
					flux += dt * mu[k] * fluxes[conn_id + k];
				}

				if (flux >= 0)
				{
					upwd = i;
					upwd_jac_idx = diag_ind[i];
					for (uint8_t c = 0; c < NC; c++)
					{
						CFL_out[c] += flux * op_vals_arr[i * N_OPS + c * N_PHASE_OPS + FLUX_OP + c];
					}
				}
				else
				{
					upwd = j;
					if (j > n_blocks)
						upwd_jac_idx = csr_idx_end;
					if (j < n_res_blocks)
					{
						for (uint8_t c = 0; c < NC; c++)
						{
							CFL_in[c] += -flux * op_vals_arr[j * N_OPS + c * N_PHASE_OPS + FLUX_OP + c];
						}
					}
				}

				for (k = 0; k < connection_type; k++)
				{
					r_counter = 0;
					// Inner contributors
					conn_st_id = offset[conn_id + k];
					for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + k + 1]; st_id++)
					{
						conn_st = stencil[conn_st_id];
						if (conn_st == cols[st_id])
						{
							if (cols[st_id] == j)
							{
								cell_p_st_id = st_id;
								if (flux < 0)
									upwd_jac_idx = st_id;
							}

							// derivative of flux
							jac_idx = N_VARS_SQ * st_id;
							for (uint8_t c = 0; c < NC; c++)
								Jac[jac_idx + c * N_VARS] += dt * mu[k] * tran[conn_st_id] * op_vals_arr[upwd * N_OPS + c * N_PHASE_OPS + FLUX_OP + c];

							// derivative of weight
							if (conn_st != i && conn_st != j)
							{
								for (uint8_t c = 0; c < NC; c++)
								{
									Jac[jac_idx + c * N_VARS] += dt * op_vals_arr[upwd * N_OPS + c * N_PHASE_OPS + FLUX_OP + c] *
																 (dmu[0][k][r_counter] * fluxes[conn_id] + dmu[1][k][r_counter] * fluxes[conn_id + 1]);
								}
								r_counter++;
							}

							conn_st_id++;
						}
					}
					// Boundary contributiors
					for (; conn_st_id < offset[conn_id + k + 1]; conn_st_id++)
					{
						conn_st = stencil[conn_st_id];
						if (conn_st >= n_blocks)
						{
							jac_idx = (k == 0) ? diag_idx : N_VARS_SQ * cell_p_st_id;
							// derivative of flux
							for (uint8_t c = 0; c < NC; c++)
								Jac[jac_idx + c * N_VARS] += dt * mu[k] * tran[conn_st_id] * op_vals_arr[upwd * N_OPS + c * N_PHASE_OPS + FLUX_OP + c];
							// derivative of weight
							for (uint8_t c = 0; c < NC; c++)
							{
								Jac[jac_idx + c * N_VARS] += dt * op_vals_arr[upwd * N_OPS + c * N_PHASE_OPS + FLUX_OP + c] *
															 (dmu[0][k][r_counter] * fluxes[conn_id] + dmu[1][k][r_counter] * fluxes[conn_id + 1]);
							}
							r_counter++;
							conn_st_id++;
						}
					}
				}

				for (uint8_t c = 0; c < NC; c++)
				{
					RHS[i * N_VARS + c] += flux * op_vals_arr[upwd * N_OPS + c * N_PHASE_OPS + FLUX_OP + c]; // flux operators only
					if (upwd_jac_idx < csr_idx_end)
					{
						jac_idx = N_VARS_SQ * upwd_jac_idx;
						for (uint8_t v = 0; v < N_VARS; v++)
							Jac[jac_idx + c * N_VARS + v] += flux * op_ders_arr[(upwd * N_OPS + c * N_PHASE_OPS + FLUX_OP + c) * N_VARS + v];
					}
				}
			}
			// calc CFL for reservoir cells, not connected with wells
			if (i < mesh->n_res_blocks)
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

			if (i < mesh->n_res_blocks)
			{
				for (uint8_t d = 0; d < N_VARS; d++)
				{
					RHS[i * N_VARS + d] += op_vals_arr[i * N_OPS + d * N_PHASE_OPS + DENS_OP] * f[i * N_VARS + d];
				}
			}
		}
#ifdef _OPENMP
#pragma omp critical
		{
			if (CFL_max < CFL_max_local)
				CFL_max = CFL_max_local;
		}
	}
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

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::assemble_jacobian_array_nlmpfa(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	return 0;
}

template <uint8_t NC>
int engine_nc_nl_cpu<NC>::solve_linear_equation()
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

	timer->node["linear solver solve"].start();
	r_code = linear_solver->solve(&RHS[0], &dX[0]);
	timer->node["linear solver solve"].stop();

	/*if (1) //changed this to write jacobian to file!
	{
		Jacobian->write_matrix_to_file("jac_nc_dar.csr");
		write_vector_to_file("jac_nc_dar.rhs", RHS);
		write_vector_to_file("jac_nc_dar.sol", dX);
		//apply_newton_update(deltat);
		write_vector_to_file("X_nc_dar", X);
		write_vector_to_file("Xn_nc_dar", Xn);
		std::vector<value_t> buf(RHS.size(), 0.0);
		//Jacobian->matrix_vector_product(dX.data(), buf.data());
		//std::transform(buf.begin(), buf.end(), RHS.begin(), buf.begin(), std::minus<double>());
		//write_vector_to_file("diff", buf);
		//exit(0);
		//return 0;
	}*/

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

template <uint8_t NC>
void engine_nc_nl_cpu<NC>::extract_Xop()
{
	if (Xop.size() < (mesh->n_blocks + mesh->n_bounds) * NC_)
	{
		Xop.resize((mesh->n_blocks + mesh->n_bounds) * NC_);
	}

	// copy unknown variables
	std::copy(X.begin(), X.end(), Xop.begin());
	std::copy(mesh->pz_bounds.begin(), mesh->pz_bounds.end(), Xop.begin() + N_VARS * mesh->n_blocks);
}

template <uint8_t NC>
int
engine_nc_nl_cpu<NC>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};

template struct recursive_instantiator_nc<engine_nc_nl_cpu, 2, MAX_NC>;
