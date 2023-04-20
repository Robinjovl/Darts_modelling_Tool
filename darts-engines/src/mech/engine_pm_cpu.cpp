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
#include <cmath>
#include <cstring>
#include <time.h>
#include <functional>
#include <string>
#include <iomanip>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "mech/engine_pm_cpu.hpp"

const value_t engine_pm_cpu::BAR_DAY2_TO_PA_S2 = 86400.0 * 86400.0 * 1.E+5;

int engine_pm_cpu::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
						std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
						sim_params *params_, timer_node *timer_)
{
	newton_update_coefficient = 1.0;
	dev_u = dev_p = well_residual_last_dt = std::numeric_limits<value_t>::infinity();
	output_counter = 0;
	FIND_EQUILIBRIUM = false;
	contact_solver = pm::RETURN_MAPPING;
	PRINT_LINEAR_SYSTEM = false;
	TIME_DEPENDENT_DISCRETIZATION = false;
	geomechanics_mode.resize(mesh_->n_blocks, 0);
	dt1 = 0.0;
	momentum_inertia = 0.0;

	init_base(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);
	return 0;
}

int engine_pm_cpu::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
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
#ifndef __linux__
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
#endif
		case sim_params::CPU_GMRES_ILU0:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linear_solver->set_prec(new linsolv_bos_bilu0<N_VARS>);
			break;
		}
#ifdef WITH_HYPRE
		case sim_params::CPU_GMRES_FS_CPR:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linsolv_iface *fs_cpr = new linsolv_bos_fs_cpr<N_VARS>(P_VAR, Z_VAR, U_VAR, NC_);
			static_cast<linsolv_bos_fs_cpr<N_VARS>*>(fs_cpr)->set_prec(new linsolv_bos_amg<1>, new linsolv_hypre_amg<1>, new linsolv_hypre_amg<1>); 
			//static_cast<linsolv_bos_fs_cpr<N_VARS>*>(fs_cpr)->set_block_sizes(mesh->n_matrix, mesh->n_fracs, mesh->n_blocks - mesh->n_res_blocks);
			static_cast<linsolv_bos_fs_cpr<N_VARS>*>(fs_cpr)->set_block_sizes(mesh->n_matrix + mesh->n_fracs, 0, mesh->n_blocks - mesh->n_res_blocks);
			//static_cast<linsolv_bos_fs_cpr<N_VARS>*>(fs_cpr)->set_prec_type(FS_UPG);
			linear_solver->set_prec(fs_cpr);
			break;
		}
#endif
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

			int n_json = 0;

			if (params->linear_params.size() > 0)
			{
				n_json = params->linear_params[0];
			}
			cpr->set_prec(new linsolv_amgx<1>(n_json));
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
	PV.resize(mesh->n_blocks);
	RV.resize(mesh->n_blocks);
	old_z.resize(nc);
	new_z.resize(nc);
	FIPS.resize(nc);
	fluxes.resize(n_vars * mesh->n_conns);
	fluxes_n.resize(n_vars * mesh->n_conns);
	fluxes_biot.resize(n_vars * mesh->n_conns);
	fluxes_biot_n.resize(n_vars * mesh->n_conns);
	fluxes_ref.resize(n_vars * mesh->n_conns, 0.0);
	fluxes_biot_ref.resize(n_vars * mesh->n_conns, 0.0);
	fluxes_ref_n.resize(n_vars * mesh->n_conns, 0.0);
	fluxes_biot_ref_n.resize(n_vars * mesh->n_conns, 0.0);
	eps_vol.resize(mesh->n_matrix);

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		for (uint8_t d = 0; d < ND_; d++)
		{
			X_init[n_vars * i + U_VAR + d] = mesh->displacement[ND_ * i + d];
		}
		X_init[n_vars * i + P_VAR] = mesh->pressure[i];

		PV[i] = mesh->volume[i] * mesh->poro[i];
		RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
	}

	op_vals_arr.resize(n_ops * (mesh->n_blocks + mesh->n_bounds));
	op_ders_arr.resize(n_ops * nc * (mesh->n_blocks + mesh->n_bounds));

	t = 0;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	stat = sim_stat();

	print_header();

	//acc_flux_op_set->init_timer_node(&timer->node["jacobian assembly"].node["interpolation"]);

	// initialize jacobian structure
	init_jacobian_structure_pm(Jacobian);

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

	RHS.resize(n_vars * mesh->n_blocks);
	dX.resize(n_vars * mesh->n_blocks);

	sprintf(buffer, "\nSTART SIMULATION\n-------------------------------------------------------------------------------------------------------------\n");
	std::cout << buffer << std::flush;

	// let wells initialize their state
	for (ms_well *w : wells)
	{
		w->initialize_control(X_init);
	}

	Xn_ref = Xref = Xn = Xn1 = X = X_init;
	for (index_t i = 0; i < mesh->ref_pressure.size(); i++)
		Xref[N_VARS * i + P_VAR] = Xn_ref[N_VARS * i + P_VAR] = mesh->ref_pressure[i];

	dt = params->first_ts;
	prev_usual_dt = dt;

	// initialize arrays for every operator set
	block_idxs.resize(acc_flux_op_set_list.size());
	op_axis_min.resize(acc_flux_op_set_list.size());
	op_axis_max.resize(acc_flux_op_set_list.size());
	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		block_idxs[r].clear();
		op_axis_min[r].resize(nc);
		op_axis_max[r].resize(nc);
		for (int j = 0; j < nc; j++)
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

	/*if (params->log_transform == 0)
	{
		min_zc = acc_flux_op_set_list[0]->get_minzc() * params->obl_min_fac;
		max_zc = 1 - min_zc * params->obl_min_fac;
		//max_zc = acc_flux_op_set_list[0]->get_maxzc();
	}
	else if (params->log_transform == 1)
	{
		min_zc = exp(acc_flux_op_set_list[0]->get_minzc() * params->obl_min_fac); //log based composition
		max_zc = exp(acc_flux_op_set_list[0]->get_maxzc() * params->obl_min_fac); //log based composition
	}*/

	return 0;
}

int engine_pm_cpu::init_jacobian_structure_pm(csr_matrix_base *jacobian)
{
	const char n_vars = get_n_vars();

	// init Jacobian structure
	index_t *rows_ptr = jacobian->get_rows_ptr();
	index_t *diag_ind = jacobian->get_diag_ind();
	index_t *cols_ind = jacobian->get_cols_ind();
	index_t *row_thread_starts = jacobian->get_row_thread_starts();

	const index_t n_blocks = mesh->n_blocks;
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

int engine_pm_cpu::assemble_jacobian_array_time_dependent_discr(value_t _dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	dt = _dt;
	// We need extended connection list for that with all connections for each block

	index_t n_blocks = mesh->n_blocks;
	index_t n_matrix = mesh->n_matrix;
	index_t n_res_blocks = mesh->n_res_blocks;
	index_t n_bounds = mesh->n_bounds;
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

	// current
	const value_t *tran = mesh->tran.data();
	const value_t *rhs = mesh->rhs.data();
	const value_t *tran_biot = mesh->tran_biot.data();
	const value_t *rhs_biot = mesh->rhs_biot.data();
	const value_t *bc = mesh->bc.data();
	// previous
	const value_t *tran_biot_n = mesh->tran_biot_n.data();
	const value_t *rhs_biot_n = mesh->rhs_biot_n.data();
	const value_t *bc_n = mesh->bc_n.data();
	// reference
	const value_t *tran_ref = mesh->tran_ref.data();
	const value_t *rhs_ref = mesh->rhs_ref.data();
	const value_t *tran_biot_ref = mesh->tran_biot_ref.data();
	const value_t *rhs_biot_ref = mesh->rhs_biot_ref.data();
	const value_t *bc_ref = mesh->bc_ref.data();

	const value_t *f = mesh->f.data();
	const value_t *V = mesh->volume.data();
	const value_t *kd = mesh->drained_compressibility.data();
	const value_t *biot = mesh->biot.data();
	const value_t *poro = mesh->poro.data();
	value_t *pz_bounds = mesh->pz_bounds.data();
	//const value_t *p_ref = mesh->ref_pressure.data();
	const value_t *eps_vol_ref = mesh->ref_eps_vol.data();

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

	std::fill_n(Jac, N_VARS * N_VARS * mesh->n_links, 0.0);
	std::fill(RHS.begin(), RHS.end(), 0.0);
	std::fill(fluxes.begin(), fluxes.end(), 0.0);
	std::fill(fluxes_biot.begin(), fluxes_biot.end(), 0.0);

	int connected_with_well;
	index_t i, j, upwd_jac_idx, upwd_idx, diag_idx, jac_idx = 0, conn_id = 0, st_id = 0, conn_st_id = 0, idx, csr_idx_start, csr_idx_end;
	value_t CFL_mech[ND_];
	value_t CFL_max_global = 0, CFL_max_local;
	const value_t *cur_bc, *cur_bc_n, *ref_bc, *buf, *buf_n;
	value_t p_diff, gamma, biot_mult, biot_cur, comp_mult, phi, phi_n, p_ref_cur, *n;
	uint8_t d, v;
	value_t tmp;

	for (i = start; i < end; ++i)
	{
		// index of diagonal block entry for block i in CSR values array
		diag_idx = N_VARS_SQ * diag_ind[i];
		// index of first entry for block i in CSR cols array
		csr_idx_start = rows[i];
		// index of last entry for block i in CSR cols array
		csr_idx_end = rows[i + 1];
		// index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
		//index_t conn_idx = csr_idx_start - i;
		connected_with_well = 0;

		//jac_idx = N_VARS_SQ * csr_idx_start;
		CFL_mech[0] = CFL_mech[1] = CFL_mech[2] = biot_mult = 0.0;
		for (; block_m[conn_id] == i && conn_id < n_conns; conn_id++)
		{
			j = block_p[conn_id];
			if (j >= n_res_blocks && j < n_blocks)
				connected_with_well = 1;

			// Fluid flux evaluation & biot flux
			gamma = 0.0;
			for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
			{
				if (stencil[conn_st_id] < n_blocks)
				{
					buf = &X[N_VARS * stencil[conn_st_id]];
					buf_n = &Xn[N_VARS * stencil[conn_st_id]];
				}
				else
				{
					buf = &bc[N_VARS * (stencil[conn_st_id] - n_blocks)];
					buf_n = &bc_n[N_VARS * (stencil[conn_st_id] - n_blocks)];
				}
				for (v = 0; v < N_VARS; v++)
				{
					gamma += tran[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v] * buf[v];
					biot_mult += tran_biot[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v] * buf[v];
					fluxes_biot[N_VARS * conn_id + P_VAR] += (tran_biot[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v] * buf[v] - 
																tran_biot_n[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v] * buf_n[v]) / dt;
				}
			}

			gamma += op_vals_arr[i * N_OPS + GRAV_OP] * rhs[N_VARS * conn_id + P_VAR];
			biot_mult += op_vals_arr[i * N_OPS + GRAV_OP] * rhs_biot[N_VARS * conn_id + P_VAR];

			// Identify upwind direction
			if (gamma >= 0)
			{
				upwd_idx = i;
				upwd_jac_idx = diag_ind[i];
				fluxes[N_VARS * conn_id + P_VAR] = gamma * op_vals_arr[i * N_OPS + FLUX_OP] / op_vals_arr[i * N_OPS + ACC_OP];
			}
			else
			{
				upwd_idx = j;
				if (j > n_blocks)
					upwd_jac_idx = csr_idx_end;
				fluxes[N_VARS * conn_id + P_VAR] = gamma * op_vals_arr[j * N_OPS + FLUX_OP] / op_vals_arr[j * N_OPS + ACC_OP];
			}

			// Inner contribution
			conn_st_id = offset[conn_id];
			for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + 1]; st_id++)
			{
				if (stencil[conn_st_id] == cols[st_id])
				{
					if (gamma < 0 && stencil[conn_st_id] == j)
						upwd_jac_idx = st_id;

					p_ref_cur = Xref[N_VARS * stencil[conn_st_id] + P_VAR];
					// momentum balance
					for (d = 0; d < ND_; d++)
					{
						for (v = 0; v < ND_; v++)
						{
							fluxes[N_VARS * conn_id + U_VAR + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] * X[stencil[conn_st_id] * N_VARS + U_VAR + v] - 
																	tran_ref[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] * Xref[stencil[conn_st_id] * N_VARS + U_VAR + v];
							fluxes_biot[N_VARS * conn_id + U_VAR + d] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] * X[stencil[conn_st_id] * N_VARS + U_VAR + v] - 
																			tran_biot_ref[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] * Xref[stencil[conn_st_id] * N_VARS + U_VAR + v];
							Jac[st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v];
							Jac[st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v];
						}
						fluxes[N_VARS * conn_id + U_VAR + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR] * X[stencil[conn_st_id] * N_VARS + P_VAR] - 
																tran_ref[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR] * p_ref_cur;
						fluxes_biot[N_VARS * conn_id + U_VAR + d] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR] * X[stencil[conn_st_id] * N_VARS + P_VAR] - 
																		tran_biot_ref[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR] * p_ref_cur;
						Jac[st_id * N_VARS_SQ + d * N_VARS + P_VAR] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR];
						Jac[st_id * N_VARS_SQ + d * N_VARS + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR];
					}
					// mass balance
					for (v = 0; v < N_VARS; v++)
					{
						Jac[st_id * N_VARS_SQ + P_VAR * N_VARS + v] += dt * op_vals_arr[upwd_idx * N_OPS + FLUX_OP] * tran[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v];
						// biot
						fluxes_biot[N_VARS * conn_id + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * X[stencil[conn_st_id] * N_VARS + v] - 
																	tran_biot_ref[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * Xref[stencil[conn_st_id] * N_VARS + v];
						RHS[i * N_VARS + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * op_vals_arr[i * N_OPS + ACC_OP] * X[stencil[conn_st_id] * N_VARS + v] -
													tran_biot_n[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * op_vals_arr_n[i * N_OPS + ACC_OP] * Xn[stencil[conn_st_id] * N_VARS + v];
						Jac[st_id * N_VARS_SQ + P_VAR * N_VARS + v] += op_vals_arr[i * N_OPS + ACC_OP] * tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v];
					}
					conn_st_id++;
				}
			}
			// Boundary contribution
			for (; conn_st_id < offset[conn_id + 1]; conn_st_id++)
			{
				if (stencil[conn_st_id] >= n_blocks)
				{
					idx = N_VARS * (stencil[conn_st_id] - n_blocks);
					cur_bc = &bc[idx];
					cur_bc_n = &bc_n[idx];
					ref_bc = &bc_ref[idx];
					// momentum balance
					for (d = 0; d < ND_; d++)
					{
						for (v = 0; v < N_VARS; v++)
						{
							fluxes[N_VARS * conn_id + U_VAR + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + v] * cur_bc[v] - 
																	tran_ref[conn_st_id * N_VARS_SQ + d * N_VARS + v] * ref_bc[v];
							fluxes_biot[N_VARS * conn_id + U_VAR + d] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + v] * cur_bc[v] - 
																			tran_biot_ref[conn_st_id * N_VARS_SQ + d * N_VARS + v] * ref_bc[v];
						}
					}
					// mass balance
					for (v = 0; v < N_VARS; v++)
					{
						// biot
						fluxes_biot[N_VARS * conn_id + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * cur_bc[v] - 
																	tran_biot_ref[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * ref_bc[v];
						RHS[i * N_VARS + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * op_vals_arr[i * N_OPS + ACC_OP] * cur_bc[v] - 
													tran_biot_n[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * op_vals_arr_n[i * N_OPS + ACC_OP] * cur_bc_n[v];
					}
				}
			}

			for (d = 0; d < ND_; d++)
			{
				fluxes[N_VARS * conn_id + U_VAR + d] += op_vals_arr[i * N_OPS + GRAV_OP] * rhs[N_VARS * conn_id + U_VAR + d];
				fluxes_biot[N_VARS * conn_id + U_VAR + d] += op_vals_arr[i * N_OPS + GRAV_OP] * rhs_biot[N_VARS * conn_id + U_VAR + d];
			}
			fluxes_biot[N_VARS * conn_id + P_VAR] += rhs_biot[N_VARS * conn_id + P_VAR] * op_vals_arr[i * N_OPS + GRAV_OP];

			// gravity

			for (d = 0; d < ND_; d++)
			{
				RHS[i * N_VARS + U_VAR + d] += fluxes_ref[N_VARS * conn_id + U_VAR + d] + fluxes[N_VARS * conn_id + U_VAR + d];
				RHS[i * N_VARS + U_VAR + d] += fluxes_biot_ref[N_VARS * conn_id + U_VAR + d] + fluxes_biot[N_VARS * conn_id + U_VAR + d];
				CFL_mech[d] += fluxes_ref[N_VARS * conn_id + U_VAR + d] + fluxes[N_VARS * conn_id + U_VAR + d] + 
								fluxes_biot_ref[N_VARS * conn_id + U_VAR + d] + fluxes_biot[N_VARS * conn_id + U_VAR + d];
				Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR] += (rhs[NT_ * conn_id + U_VAR + d] + rhs_biot[NT_ * conn_id + U_VAR + d]) * op_ders_arr[(i * N_OPS + GRAV_OP) * NC_];
			}
			RHS[i * N_VARS + P_VAR] += dt * op_vals_arr[upwd_idx * N_OPS + FLUX_OP] * gamma;
			//RHS[i * N_VARS + P_VAR] += op_vals_arr[i * N_OPS + ACC_OP] * (fluxes_biot_ref[N_VARS * conn_id + P_VAR] + fluxes_biot[N_VARS * conn_id + P_VAR]) -
			//							op_vals_arr_n[i * N_OPS + ACC_OP] * (fluxes_biot_ref_n[N_VARS * conn_id + P_VAR] + fluxes_biot_n[N_VARS * conn_id + P_VAR]);

			RHS[i * N_VARS + P_VAR] += rhs_biot[N_VARS * conn_id + P_VAR] * op_vals_arr[i * N_OPS + ACC_OP] * op_vals_arr[i * N_OPS + GRAV_OP] - 
										rhs_biot_n[N_VARS * conn_id + P_VAR] * op_vals_arr_n[i * N_OPS + ACC_OP] * op_vals_arr_n[i * N_OPS + GRAV_OP];

			if (upwd_jac_idx < csr_idx_end)
			{
				Jac[upwd_jac_idx * N_VARS_SQ + P_VAR * N_VARS + P_VAR] += dt * gamma * op_ders_arr[(upwd_idx * N_OPS + FLUX_OP) * NC_];
			}
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] += (dt * op_vals_arr[upwd_idx * N_OPS + FLUX_OP] * rhs[NT_ * conn_id + P_VAR] + rhs_biot[N_VARS * conn_id + P_VAR] * op_vals_arr[i * N_OPS + ACC_OP]) *
														op_ders_arr[(i * N_OPS + GRAV_OP) * NC_];
		}

		// porosity
		if (i >= n_matrix)
		{
			biot_mult = comp_mult = 0.0;
			phi = poro[i];
			phi_n = poro[i];
		}
		else
		{
			eps_vol[i] = biot_mult / V[i];
			biot_mult -= V[i] * eps_vol_ref[i];
			RHS[i * N_VARS + P_VAR] += -V[i] * eps_vol_ref[i] * (op_vals_arr[i * N_OPS + ACC_OP] - op_vals_arr_n[i * N_OPS + ACC_OP]);
			biot_cur = (biot[i * ND_ * ND_] + biot[i * ND_ * ND_ + ND_ + 1] + biot[i * ND_ * ND_ + 2 * ND_ + 2]) / 3.0; // one-third of the Biot tensor trace
			comp_mult = (biot_cur != 0) ? (biot_cur - poro[i]) * (1 - biot_cur) / kd[i] : 1.0 / kd[i];
			phi = poro[i] + comp_mult * (X[i * N_VARS + P_VAR] - Xref[N_VARS * i + P_VAR]);
			phi_n = poro[i] + comp_mult * (Xn[i * N_VARS + P_VAR] - Xn_ref[N_VARS * i + P_VAR]);
		}

		if (FIND_EQUILIBRIUM || geomechanics_mode[i])
		{
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] = V[i];
		}
		else
		{
			RHS[i * N_VARS + P_VAR] += V[i] * (phi * op_vals_arr[i * N_OPS + ACC_OP] - phi_n * op_vals_arr_n[i * N_OPS + ACC_OP]);
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] += (V[i] * phi + biot_mult) * op_ders_arr[(i * N_OPS + ACC_OP) * NC_];
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] += V[i] * comp_mult * op_vals_arr[i * N_OPS + ACC_OP];
		}

		if (!FIND_EQUILIBRIUM)
		{
			// momentum inertia
			if (dt > 0.0)
			{
				for (d = 0; d < ND_; d++)
				{
					RHS[i * N_VARS + U_VAR + d] += momentum_inertia * mesh->volume[i] * (X[i * N_VARS + U_VAR + d] - Xn[i * N_VARS + U_VAR + d]) / dt / dt / engine_pm_cpu::BAR_DAY2_TO_PA_S2;
					Jac[diag_idx + (U_VAR + d) * N_VARS + U_VAR + d] += momentum_inertia * mesh->volume[i] / dt / dt / engine_pm_cpu::BAR_DAY2_TO_PA_S2;
				}

				if (dt1 > 0.0)
				{
					for (d = 0; d < ND_; d++)
					{
						RHS[i * N_VARS + U_VAR + d] += -momentum_inertia * mesh->volume[i] * (Xn[i * N_VARS + U_VAR + d] - Xn1[i * N_VARS + U_VAR + d]) / dt / dt1 / engine_pm_cpu::BAR_DAY2_TO_PA_S2;
					}
				}
			}
		}

		// calc CFL for reservoir cells, not connected with wells
		if (i < n_res_blocks)
		{
			if (fabs(momentum_inertia) > 0.0 && dt > 0.0)
			{
				CFL_max_local = 0.0;
				for (uint8_t d = 0; d < ND_; d++)
				{
					tmp = engine_pm_cpu::BAR_DAY2_TO_PA_S2 * CFL_mech[d] / momentum_inertia / mesh->volume[i] /
						((X[i * N_VARS + U_VAR + d] - Xn[i * N_VARS + U_VAR + d]) / dt / dt);
					CFL_max_local += tmp * tmp;
				}
				if (fabs(X[i * N_VARS + U_VAR + d] - Xn[i * N_VARS + U_VAR + d]) > EQUALITY_TOLERANCE)
					CFL_max_global = std::max(CFL_max_global, sqrt(CFL_max_local));
			}

			// volumetric forces and source/sink 
			for (d = 0; d < ND_; d++)
			{
				RHS[i * N_VARS + U_VAR + d] += V[i] * f[i * N_VARS + d];
			}
			RHS[i * N_VARS + P_VAR] += V[i] * dt * f[i * N_VARS + P_VAR];
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
	CFL_max = CFL_max_global;
#endif
	for (auto &contact : contacts)
	{
		if (FIND_EQUILIBRIUM)
			contact.set_state(pm::TRUE_STUCK);

		if (contact_solver == pm::FLUX_FROM_PREVIOUS_ITERATION)
			contact.add_to_jacobian_linear(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
		else if (contact_solver == pm::RETURN_MAPPING)
			contact.add_to_jacobian_return_mapping(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
		else if (contact_solver == pm::LOCAL_ITERATIONS)
			contact.add_to_jacobian_local_iters(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
	}
	for (ms_well *w : wells)
	{
		value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * N_VARS_SQ]);
		w->add_to_jacobian(dt, X, jac_well_head, RHS);
		for (uint8_t d = 0; d < ND_; d++)
		{
			Jac[N_VARS_SQ * diag_ind[w->well_head_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
			Jac[N_VARS_SQ * diag_ind[w->well_body_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
		}
	}
	return 0;
};

int engine_pm_cpu::assemble_jacobian_array(value_t _dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	dt = _dt;
	// We need extended connection list for that with all connections for each block

	index_t n_blocks = mesh->n_blocks;
	index_t n_matrix = mesh->n_matrix;
	index_t n_res_blocks = mesh->n_res_blocks;
	index_t n_bounds = mesh->n_bounds;
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
	const value_t *tran_biot = mesh->tran_biot.data();
	value_t *bc = mesh->bc.data();
	value_t *bc_prev = mesh->bc_n.data();
	value_t *bc_ref = mesh->bc_ref.data();
	const value_t *rhs = mesh->rhs.data();
	const value_t *rhs_biot = mesh->rhs_biot.data();
	const value_t *f = mesh->f.data();
	const value_t *V = mesh->volume.data();
	const value_t *kd = mesh->drained_compressibility.data();
	const value_t *biot = mesh->biot.data();
	const value_t *poro = mesh->poro.data();
	value_t *pz_bounds = mesh->pz_bounds.data();
	//const value_t *p_ref = mesh->ref_pressure.data();
	const value_t *eps_vol_ref = mesh->ref_eps_vol.data();

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

	std::fill_n(Jac, N_VARS * N_VARS * mesh->n_links, 0.0);
	std::fill(RHS.begin(), RHS.end(), 0.0);
	std::fill(fluxes.begin(), fluxes.end(), 0.0);
	std::fill(fluxes_biot.begin(), fluxes_biot.end(), 0.0);

	int connected_with_well;
	index_t i, j, upwd_jac_idx, upwd_idx, diag_idx, jac_idx = 0, conn_id = 0, st_id = 0, conn_st_id = 0, idx, csr_idx_start, csr_idx_end, cur_conn_id;
	value_t CFL_mech[ND_];
	value_t CFL_max_global = 0, CFL_max_local;
	value_t p_diff, gamma, *cur_bc, *cur_bc_prev, *ref_bc, biot_mult, biot_cur, comp_mult, phi, phi_n, *buf, *buf_prev, p_ref_cur, *n;
	uint8_t d, v;
	value_t tmp;

	for (i = start; i < end; ++i)
	{
		// index of diagonal block entry for block i in CSR values array
		diag_idx = N_VARS_SQ * diag_ind[i];
		// index of first entry for block i in CSR cols array
		csr_idx_start = rows[i];
		// index of last entry for block i in CSR cols array
		csr_idx_end = rows[i + 1];
		// index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
		//index_t conn_idx = csr_idx_start - i;
		connected_with_well = 0;

		//jac_idx = N_VARS_SQ * csr_idx_start;
		CFL_mech[0] = CFL_mech[1] = CFL_mech[2] = biot_mult = 0.0;
		for (; block_m[conn_id] == i && conn_id < n_conns; conn_id++)
		{
			j = block_p[conn_id];
			if (j >= n_res_blocks && j < n_blocks)
				connected_with_well = 1;

			// Fluid flux evaluation & biot flux
			gamma = 0.0;
			for (conn_st_id = offset[conn_id]; conn_st_id < offset[conn_id + 1]; conn_st_id++)
			{
				if (stencil[conn_st_id] < n_blocks)
				{
					buf = &X[N_VARS * stencil[conn_st_id]];
					buf_prev = &Xn[N_VARS * stencil[conn_st_id]];
				}
				else
				{
					buf = &bc[N_VARS * (stencil[conn_st_id] - n_blocks)];
					buf_prev = &bc_prev[N_VARS * (stencil[conn_st_id] - n_blocks)];
				}
				for (v = 0; v < N_VARS; v++)
				{
					gamma += tran[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v] * buf[v];
					biot_mult += tran_biot[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v] * buf[v];
					fluxes_biot[N_VARS * conn_id + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v] * (buf[v] - buf_prev[v]) / dt;
				}
			}

			gamma += op_vals_arr[i * N_OPS + GRAV_OP] * rhs[N_VARS * conn_id + P_VAR];
			biot_mult += op_vals_arr[i * N_OPS + GRAV_OP] * rhs_biot[N_VARS * conn_id + P_VAR];

			// Identify upwind direction
			if (gamma >= 0)
			{
				upwd_idx = i;
				upwd_jac_idx = diag_ind[i];
				fluxes[N_VARS * conn_id + P_VAR] = gamma * op_vals_arr[i * N_OPS + FLUX_OP] / op_vals_arr[i * N_OPS + ACC_OP];
			}
			else
			{
				upwd_idx = j;
				if (j > n_blocks)
					upwd_jac_idx = csr_idx_end;
				fluxes[N_VARS * conn_id + P_VAR] = gamma * op_vals_arr[j * N_OPS + FLUX_OP] / op_vals_arr[j * N_OPS + ACC_OP];
			}

			// Inner contribution
			conn_st_id = offset[conn_id];
			for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + 1]; st_id++)
			{
				if (stencil[conn_st_id] == cols[st_id])
				{
					if (gamma < 0 && stencil[conn_st_id] == j)
						upwd_jac_idx = st_id;

					p_ref_cur = Xref[N_VARS * stencil[conn_st_id] + P_VAR];
					// momentum balance
					for (d = 0; d < ND_; d++)
					{
						for (v = 0; v < ND_; v++)
						{
							fluxes[N_VARS * conn_id + U_VAR + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] * (X[stencil[conn_st_id] * N_VARS + U_VAR + v] - Xref[stencil[conn_st_id] * N_VARS + U_VAR + v]);
							fluxes_biot[N_VARS * conn_id + U_VAR + d] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] * (X[stencil[conn_st_id] * N_VARS + U_VAR + v] - Xref[stencil[conn_st_id] * N_VARS + U_VAR + v]);
							Jac[st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v];
							Jac[st_id * N_VARS_SQ + d * N_VARS + U_VAR + v] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + U_VAR + v];
						}
						fluxes[N_VARS * conn_id + U_VAR + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR] * (X[stencil[conn_st_id] * N_VARS + P_VAR] - p_ref_cur);
						fluxes_biot[N_VARS * conn_id + U_VAR + d] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR] * (X[stencil[conn_st_id] * N_VARS + P_VAR] - p_ref_cur);
						Jac[st_id * N_VARS_SQ + d * N_VARS + P_VAR] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR];
						Jac[st_id * N_VARS_SQ + d * N_VARS + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + P_VAR];
					}
					// mass balance
					for (v = 0; v < N_VARS; v++)
					{
						Jac[st_id * N_VARS_SQ + P_VAR * N_VARS + v] += dt * op_vals_arr[upwd_idx * N_OPS + FLUX_OP] * tran[conn_st_id * N_VARS_SQ + P_VAR * N_VARS + v];
						// biot
						fluxes_biot[N_VARS * conn_id + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * (X[stencil[conn_st_id] * N_VARS + v] - Xref[stencil[conn_st_id] * N_VARS + v]);
						RHS[i * N_VARS + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] *
							(op_vals_arr[i * N_OPS + ACC_OP] * X[stencil[conn_st_id] * N_VARS + v] -
								op_vals_arr_n[i * N_OPS + ACC_OP] * Xn[stencil[conn_st_id] * N_VARS + v]);
						Jac[st_id * N_VARS_SQ + P_VAR * N_VARS + v] += op_vals_arr[i * N_OPS + ACC_OP] * tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v];
					}
					conn_st_id++;
				}
			}
			// Boundary contribution
			for (; conn_st_id < offset[conn_id + 1]; conn_st_id++)
			{
				if (stencil[conn_st_id] >= n_blocks)
				{
					idx = N_VARS * (stencil[conn_st_id] - n_blocks);
					cur_bc = &bc[idx];
					cur_bc_prev = &bc_prev[idx];
					ref_bc = &bc_ref[idx];
					// momentum balance
					for (d = 0; d < ND_; d++)
					{
						for (v = 0; v < N_VARS; v++)
						{
							fluxes[N_VARS * conn_id + U_VAR + d] += tran[conn_st_id * N_VARS_SQ + d * N_VARS + v] * (cur_bc[v] - ref_bc[v]);
							fluxes_biot[N_VARS * conn_id + U_VAR + d] += tran_biot[conn_st_id * N_VARS_SQ + d * N_VARS + v] * (cur_bc[v] - ref_bc[v]);
						}
					}
					// mass balance
					for (v = 0; v < N_VARS; v++)
					{
						// biot
						fluxes_biot[N_VARS * conn_id + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] * (cur_bc[v] - ref_bc[v]);
						RHS[i * N_VARS + P_VAR] += tran_biot[conn_st_id * N_VARS_SQ + N_VARS * P_VAR + v] *
							(op_vals_arr[i * N_OPS + ACC_OP] * cur_bc[v] - op_vals_arr_n[i * N_OPS + ACC_OP] * cur_bc_prev[v]);
					}
				}
			}

			for (d = 0; d < ND_; d++)
			{
				fluxes[N_VARS * conn_id + U_VAR + d] += op_vals_arr[i * N_OPS + GRAV_OP] * rhs[N_VARS * conn_id + U_VAR + d];
				fluxes_biot[N_VARS * conn_id + U_VAR + d] += op_vals_arr[i * N_OPS + GRAV_OP] * rhs_biot[N_VARS * conn_id + U_VAR + d];
			}
			fluxes_biot[N_VARS * conn_id + P_VAR] += rhs_biot[N_VARS * conn_id + P_VAR] * op_vals_arr[i * N_OPS + GRAV_OP];

			// gravity

			for (d = 0; d < ND_; d++)
			{
				RHS[i * N_VARS + U_VAR + d] += fluxes_ref[N_VARS * conn_id + U_VAR + d] + fluxes[N_VARS * conn_id + U_VAR + d];
				RHS[i * N_VARS + U_VAR + d] += fluxes_biot_ref[N_VARS * conn_id + U_VAR + d] + fluxes_biot[N_VARS * conn_id + U_VAR + d];
				CFL_mech[d] += fluxes_ref[N_VARS * conn_id + U_VAR + d] + fluxes[N_VARS * conn_id + U_VAR + d] +
					fluxes_biot_ref[N_VARS * conn_id + U_VAR + d] + fluxes_biot[N_VARS * conn_id + U_VAR + d];
				Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR] += (rhs[NT_ * conn_id + U_VAR + d] + rhs_biot[NT_ * conn_id + U_VAR + d]) * op_ders_arr[(i * N_OPS + GRAV_OP) * NC_];
			}
			RHS[i * N_VARS + P_VAR] += dt * op_vals_arr[upwd_idx * N_OPS + FLUX_OP] * gamma;
			//RHS[i * N_VARS + P_VAR] += op_vals_arr[i * N_OPS + ACC_OP] * (fluxes_biot_ref[N_VARS * conn_id + P_VAR] + fluxes_biot[N_VARS * conn_id + P_VAR]) -
			//							op_vals_arr_n[i * N_OPS + ACC_OP] * (fluxes_biot_ref_n[N_VARS * conn_id + P_VAR] + fluxes_biot_n[N_VARS * conn_id + P_VAR]);

			RHS[i * N_VARS + P_VAR] += rhs_biot[N_VARS * conn_id + P_VAR] *
				(op_vals_arr[i * N_OPS + ACC_OP] * op_vals_arr[i * N_OPS + GRAV_OP] - op_vals_arr_n[i * N_OPS + ACC_OP] * op_vals_arr_n[i * N_OPS + GRAV_OP]);

			if (upwd_jac_idx < csr_idx_end)
			{
				Jac[upwd_jac_idx * N_VARS_SQ + P_VAR * N_VARS + P_VAR] += dt * gamma * op_ders_arr[(upwd_idx * N_OPS + FLUX_OP) * NC_];
			}
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] += (dt * op_vals_arr[upwd_idx * N_OPS + FLUX_OP] * rhs[NT_ * conn_id + P_VAR] + rhs_biot[N_VARS * conn_id + P_VAR] * op_vals_arr[i * N_OPS + ACC_OP]) *
				op_ders_arr[(i * N_OPS + GRAV_OP) * NC_];
		}

		// porosity
		if (i >= n_matrix)
		{
			biot_mult = comp_mult = 0.0;
			phi = poro[i];
			phi_n = poro[i];
		}
		else
		{
			eps_vol[i] = biot_mult / V[i];
			biot_mult -= V[i] * eps_vol_ref[i];
			RHS[i * N_VARS + P_VAR] += -V[i] * eps_vol_ref[i] * (op_vals_arr[i * N_OPS + ACC_OP] - op_vals_arr_n[i * N_OPS + ACC_OP]);
			biot_cur = (biot[i * ND_ * ND_] + biot[i * ND_ * ND_ + ND_ + 1] + biot[i * ND_ * ND_ + 2 * ND_ + 2]) / 3.0; // one-third of the Biot tensor trace
			comp_mult = (biot_cur != 0) ? (biot_cur - poro[i]) * (1 - biot_cur) / kd[i] : 1.0 / kd[i];
			phi = poro[i] + comp_mult * (X[i * N_VARS + P_VAR] - Xref[N_VARS * i + P_VAR]);
			phi_n = poro[i] + comp_mult * (Xn[i * N_VARS + P_VAR] - Xn_ref[N_VARS * i + P_VAR]);
		}

		if (FIND_EQUILIBRIUM || geomechanics_mode[i])
		{
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] = V[i];
		}
		else
		{
			RHS[i * N_VARS + P_VAR] += V[i] * (phi * op_vals_arr[i * N_OPS + ACC_OP] - phi_n * op_vals_arr_n[i * N_OPS + ACC_OP]);
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] += (V[i] * phi + biot_mult) * op_ders_arr[(i * N_OPS + ACC_OP) * NC_];
			Jac[diag_idx + P_VAR * N_VARS + P_VAR] += V[i] * comp_mult * op_vals_arr[i * N_OPS + ACC_OP];
		}

		if (!FIND_EQUILIBRIUM)
		{
			// momentum inertia
			if (dt > 0.0)
			{
				for (d = 0; d < ND_; d++)
				{
					RHS[i * N_VARS + U_VAR + d] += momentum_inertia * mesh->volume[i] * (X[i * N_VARS + U_VAR + d] - Xn[i * N_VARS + U_VAR + d]) / dt / dt / engine_pm_cpu::BAR_DAY2_TO_PA_S2;
					Jac[diag_idx + (U_VAR + d) * N_VARS + U_VAR + d] += momentum_inertia * mesh->volume[i] / dt / dt / engine_pm_cpu::BAR_DAY2_TO_PA_S2;
				}

				if (dt1 > 0.0)
				{
					for (d = 0; d < ND_; d++)
					{
						RHS[i * N_VARS + U_VAR + d] += -momentum_inertia * mesh->volume[i] * (Xn[i * N_VARS + U_VAR + d] - Xn1[i * N_VARS + U_VAR + d]) / dt / dt1 / engine_pm_cpu::BAR_DAY2_TO_PA_S2;
					}
				}
			}
		}

		// calc CFL for reservoir cells, not connected with wells
		if (i < n_res_blocks)
		{
			if (fabs(momentum_inertia) > 0.0 && dt > 0.0)
			{
				CFL_max_local = 0.0;
				for (uint8_t d = 0; d < ND_; d++)
				{
					tmp = engine_pm_cpu::BAR_DAY2_TO_PA_S2 * CFL_mech[d] / momentum_inertia / mesh->volume[i] /
						((X[i * N_VARS + U_VAR + d] - Xn[i * N_VARS + U_VAR + d]) / dt / dt);
					CFL_max_local += tmp * tmp;
				}
				if (fabs(X[i * N_VARS + U_VAR + d] - Xn[i * N_VARS + U_VAR + d]) > EQUALITY_TOLERANCE)
					CFL_max_global = std::max(CFL_max_global, sqrt(CFL_max_local));
			}

			// volumetric forces and source/sink 
			for (d = 0; d < ND_; d++)
			{
				RHS[i * N_VARS + U_VAR + d] += V[i] * f[i * N_VARS + d];
			}
			RHS[i * N_VARS + P_VAR] += V[i] * dt * f[i * N_VARS + P_VAR];
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
	CFL_max = CFL_max_global;
#endif
	for (auto &contact : contacts)
	{
		if (FIND_EQUILIBRIUM)
			contact.set_state(pm::TRUE_STUCK);

		if (contact_solver == pm::FLUX_FROM_PREVIOUS_ITERATION)
			contact.add_to_jacobian_linear(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
		else if (contact_solver == pm::RETURN_MAPPING)
			contact.add_to_jacobian_return_mapping(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
		else if (contact_solver == pm::LOCAL_ITERATIONS)
			contact.add_to_jacobian_local_iters(dt, jacobian, RHS, X, fluxes, fluxes_biot, Xn, fluxes_n, fluxes_biot_n, Xref, fluxes_ref, fluxes_biot_ref, Xn_ref, fluxes_ref_n, fluxes_biot_ref_n);
	}
	for (ms_well *w : wells)
	{
		value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * N_VARS_SQ]);
		w->add_to_jacobian(dt, X, jac_well_head, RHS);
		for (uint8_t d = 0; d < ND_; d++)
		{
			Jac[N_VARS_SQ * diag_ind[w->well_head_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
			Jac[N_VARS_SQ * diag_ind[w->well_body_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
		}
	}
	return 0;
};

void engine_pm_cpu::apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	double max_ratio = 0;
	index_t n_vars_total = X.size();
	index_t n_obl_fixes = 0;

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		value_t *axis_min = &op_axis_min[mesh->op_num[i]][0];
		value_t *axis_max = &op_axis_max[mesh->op_num[i]][0];
		for (index_t v = 0; v < nc; v++)
		{
			value_t new_x = X[i * n_vars + P_VAR + v] - dX[i * n_vars + P_VAR + v];
			if (new_x > axis_max[v])
			{
				dX[i * n_vars + P_VAR + v] = X[i * n_vars + P_VAR + v] - axis_max[v];
				// output only for the first time
				if (n_obl_fixes == 0)
				{
					std::cout << "OBL axis correction: block " << i << " variable " << v << " shoots over axis limit of " << axis_max[v] << " to " << new_x << std::endl;
				}
				n_obl_fixes++;
			}
			else if (new_x < axis_min[v])
			{
				dX[i * n_vars + P_VAR + v] = X[i * n_vars + P_VAR + v] - axis_min[v];
				// output only for the first time
				if (n_obl_fixes == 0)
				{
					std::cout << "OBL axis correction: block " << i << " variable " << v << " shoots under axis limit of " << axis_min[v] << " to " << new_x << std::endl;
				}
				n_obl_fixes++;
			}
		}
	}

	if (n_obl_fixes > 0)
	{
		std::cout << "OBL axis correction applied " << n_obl_fixes << " time(s) \n";
	}
}

void engine_pm_cpu::extract_Xop()
{
	if (Xop.size() < (mesh->n_blocks + mesh->n_bounds) * NC_)
	{
		Xop.resize((mesh->n_blocks + mesh->n_bounds) * NC_);
	}
	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		Xop[i * NC_] = X[i * N_VARS + P_VAR];
	}
	for (index_t i = 0; i < mesh->n_bounds; i++)
	{
		Xop[(mesh->n_blocks + i) * NC_] = mesh->pz_bounds[i * NC_];
	}
}

std::vector<value_t>
engine_pm_cpu::calc_newton_dev()
{
	/*switch (params->nonlinear_norm_type)
	{
	case sim_params::L1:
	{
		return calc_newton_residual_L1();
	}
	case sim_params::L2:
	{
		return calc_newton_residual_L2();
	}
	case sim_params::LINF:
	{
		return calc_newton_residual_Linf();
	}
	default:
	{*/
	return calc_newton_dev_L2();
	//}
	//}
}

std::vector<value_t>
engine_pm_cpu::calc_newton_dev_L2()
{
	std::vector<value_t> two_dev(3, 0);
	std::vector<value_t> dev(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);
	value_t gap_dev = 0.0, norm_gap = 0.0;

	// accumulation of residual
	for (int i = 0; i < mesh->n_matrix; i++)
	{
		for (int c = U_VAR; c < ND_; c++)
		{
			dev[c] += RHS[i * n_vars + c] * RHS[i * n_vars + c];
			norm[c] += mesh->volume[i] * mesh->volume[i];
		}
		dev[P_VAR] += RHS[i * n_vars + P_VAR] * RHS[i * n_vars + P_VAR];
		norm[P_VAR] +=	mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP] * 
						mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP];
	}
	// in faults
	for (int i = mesh->n_matrix; i < mesh->n_res_blocks; i++)
	{
		for (int c = U_VAR; c < ND_; c++)
		{
			//dev[c] += RHS[i * n_vars + c] * RHS[i * n_vars + c];
			//norm[c] += mesh->volume[i] * mesh->volume[i];
			gap_dev += RHS[i * n_vars + c] * RHS[i * n_vars + c];
		}
		dev[P_VAR] += RHS[i * n_vars + P_VAR] * RHS[i * n_vars + P_VAR];
		norm[P_VAR] +=	mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP] *
						mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP];
		norm_gap += 1.0;// mesh->volume[i] * mesh->volume[i];
	}
	// flow norm
	two_dev[0] = 0.0;
	for (int c = 0; c < NC_; c++)
	{
		two_dev[0] = std::max(two_dev[0], sqrt(dev[P_VAR + c] / norm[P_VAR + c]));
	}
	dev_p_prev = dev_p;
	dev_p = two_dev[0];
	//mech norm
	two_dev[1] = 0.0;
	for (int c = 0; c < ND_; c++)
	{
		two_dev[1] = std::max(two_dev[1], sqrt(dev[U_VAR + c] / norm[U_VAR + c]));
	}
	dev_u_prev = dev_u;
	dev_u = two_dev[1];
	// gap norm
	if (mesh->n_res_blocks > mesh->n_matrix)
		two_dev[2] = sqrt(gap_dev / norm_gap);

	return two_dev;
}

double
engine_pm_cpu::calc_well_residual_L2()
{
	double residual = 0;
	std::vector<value_t> res(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);

	std::vector<value_t> av_op(n_vars, 0);
	average_operator(av_op);

	for (ms_well *w : wells)
	{
		// first sum up RHS for well segments which have perforations
		int nperf = w->perforations.size();
		for (int ip = 0; ip < nperf; ip++)
		{
			for (int v = 0; v < n_vars; v++)
			{
				index_t i_w, i_r;
				value_t wi, wid;
				std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

				res[v] += RHS[(w->well_body_idx + i_w) * n_vars + v] * RHS[(w->well_body_idx + i_w) * n_vars + v];
				norm[v] += PV[w->well_body_idx + i_w] * av_op[v] * PV[w->well_body_idx + i_w] * av_op[v];
			}
		}
		// and then add RHS for well control equations
		for (int v = 0; v < n_vars; v++)
		{
			// well constraints should not be normalized, so pre-multiply by norm
			res[v] += RHS[w->well_head_idx * n_vars + v] * RHS[w->well_head_idx * n_vars + v] * PV[w->well_body_idx] * av_op[v] * PV[w->well_body_idx] * av_op[v];
		}
	}

	for (int v = 0; v < n_vars; v++)
	{
		residual = std::max(residual, sqrt(res[v] / norm[v]));
	}

	well_residual_prev_dt = well_residual_last_dt;
	well_residual_last_dt = residual;

	return residual;
}

int engine_pm_cpu::run_single_newton_iteration(value_t deltat)
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

	extract_Xop();
	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		int result = acc_flux_op_set_list[r]->evaluate_with_derivatives(Xop, block_idxs[r], op_vals_arr, op_ders_arr);
		if (result < 0)
			return 0;
	}

	timer->node["jacobian assembly"].node["interpolation"].stop();

	// assemble jacobian
	if (TIME_DEPENDENT_DISCRETIZATION)
		assemble_jacobian_array_time_dependent_discr(deltat, X, Jacobian, RHS);
	else
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

int engine_pm_cpu::apply_newton_update(value_t dt)
{
	/*if (params->newton_type == sim_params::NEWTON_GLOBAL_CHOP)
	{
		// max gap
		/*for (index_t i = mesh->n_matrix; i < mesh->n_res_blocks; i++)
		{
			for (uint8_t d = 0; d < ND_; d++)
				max_gap_change = std::max(max_gap_change, fabs(dX[N_VARS * i + U_VAR + d]));
		}
		if (contacts[0].max_allowed_gap_change > 0 && max_gap_change > contacts[0].max_allowed_gap_change)
		{
			mult = 0.5;
			printf("Global chop activated!\n");
		}

		value_t cur_norm, prev_norm;
		prev_norm = sqrt(dev_p_prev * dev_p_prev + dev_u_prev * dev_u_prev + well_residual_prev_dt * well_residual_prev_dt);
		cur_norm = sqrt(dev_p * dev_p + dev_u * dev_u + well_residual_last_dt * well_residual_last_dt);
		if (cur_norm > 0.9 * prev_norm)
		{
			newton_update_coefficient = 0.1;
			printf("Global chop activated!\n");
		}
	}*/

	//for (auto& contact : contacts)
	//	contact.apply_direction_chop(X, Xn, dX);


	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		for (uint8_t d = 0; d < ND_; d++)
			X[N_VARS * i + U_VAR + d] -= newton_update_coefficient * dX[N_VARS * i + U_VAR + d];
		X[N_VARS * i + P_VAR] -= newton_update_coefficient * dX[N_VARS * i + P_VAR];
	}

	return 0;
}

int engine_pm_cpu::solve_linear_equation()
{
	int r_code;
	char buffer[1024];
	linear_solver_error_last_dt = 0;

	/*if (1) //changed this to write jacobian to file!
	{
		static_cast<csr_matrix<4>*>(Jacobian)->write_matrix_to_file_mm(("jac_nc_dar_" + std::to_string(output_counter++) + ".csr").c_str());
		write_vector_to_file("jac_nc_dar.rhs", RHS);
		write_vector_to_file("jac_nc_dar.sol", dX);
	//apply_newton_update(deltat);
	//write_vector_to_file("X_nc_dar", X);
	//write_vector_to_file("Xn_nc_dar", Xn);
	//std::vector<value_t> buf(RHS.size(), 0.0);
	//Jacobian->matrix_vector_product(dX.data(), buf.data());
	//std::transform(buf.begin(), buf.end(), RHS.begin(), buf.begin(), std::minus<double>());
	//write_vector_to_file("diff", buf);
	//exit(0);
	//return 0;
	}*/

	timer->node["linear solver setup"].start();
	//static_cast<linsolv_bos_fs_cpr<N_VARS>*>(static_cast<linsolv_bos_gmres<N_VARS>*>(linear_solver)->prec)->set_block_sizes(mesh->n_matrix, mesh->n_fracs, mesh->n_blocks - mesh->n_res_blocks);
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

	if (PRINT_LINEAR_SYSTEM) // changed this to write jacobian to file!
	{
                #ifndef OPENDARTS_LINEAR_SOLVERS
	        static_cast<csr_matrix<4>*>(Jacobian)->write_matrix_to_file_mm(("jac_nc_dar_" + std::to_string(output_counter) + ".csr").c_str());
                #endif  // OPENDARTS_LINEAR_SOLVERS
		//Jacobian->write_matrix_to_file(("jac_dar_" + std::to_string(output_counter) + ".csr").c_str());
		write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".rhs", RHS);
		write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".sol", X);
		output_counter++;
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
		sprintf(buffer, "\t #%d (%.4e, %.4e, %.4e): lin %d (%.1e)\n", n_newton_last_dt + 1,
				dev_p, dev_u, well_residual_last_dt,
				linear_solver->get_n_iters(), linear_solver->get_residual());
		std::cout << buffer << std::flush;
		n_linear_last_dt += linear_solver->get_n_iters();
	}
	return 0;
}

int engine_pm_cpu::post_newtonloop(value_t deltat, value_t time)
{
	int converged = 0;
	char buffer[1024];
	double well_tolerance_coefficient = 1e2;

	if (linear_solver_error_last_dt == 1) // linear solver setup failed
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (linear solver setup failed) \n", deltat);
	}
	else if (linear_solver_error_last_dt == 2) // linear solver solve failed
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (linear solver solve failed) \n", deltat);
	}
	else if (newton_residual_last_dt >= params->tolerance_newton) // no reservoir convergence reached
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (newton residual reservoir) \n", deltat);
	}
	else if (well_residual_last_dt > well_tolerance_coefficient * params->tolerance_newton) // no well convergence reached
	{
		sprintf(buffer, "FAILED TO CONVERGE WITH DT = %.3lf (newton residual wells) \n", deltat);
	}
	else
	{
		converged = 1;
	}

	dev_u = dev_p = well_residual_last_dt = std::numeric_limits<value_t>::infinity();

	if (!converged)
	{
		stat.n_newton_wasted += n_newton_last_dt;
		stat.n_linear_wasted += n_linear_last_dt;
		stat.n_timesteps_wasted++;
		converged = 0;

		X = Xn;
		Xref = Xn_ref;
		std::copy(fluxes_n.begin(), fluxes_n.end(), fluxes.begin());
		std::copy(fluxes_biot_n.begin(), fluxes_biot_n.end(), fluxes_biot.begin());
		std::copy(fluxes_ref_n.begin(), fluxes_ref_n.end(), fluxes_ref.begin());
		std::copy(fluxes_biot_ref_n.begin(), fluxes_biot_ref_n.end(), fluxes_biot_ref.begin());
		std::cout << buffer << std::flush;
	}
	else //convergence reached
	{
		stat.n_newton_total += n_newton_last_dt;
		stat.n_linear_total += n_linear_last_dt;
		stat.n_timesteps_total++;
		converged = 1;

		print_timestep(time + deltat, deltat);

		time_data["time"].push_back(time + deltat);

		for (ms_well *w : wells)
		{
			w->calc_rates(X, op_vals_arr, time_data);
		}

		// calculate FIPS
		FIPS.assign(nc, 0);
		for (index_t i = 0; i < mesh->n_res_blocks; i++)
		{
			for (uint8_t c = 0; c < nc; c++)
			{
				// assuming ACC_OP is 0
				FIPS[c] += PV[i] * op_vals_arr[i * n_ops + 0 + c];
			}
		}

		for (uint8_t c = 0; c < nc; c++)
		{
			time_data["FIPS c " + std::to_string(c) + " (kmol)"].push_back(FIPS[c]);
		}

		// fault contacts
		for (auto& contact : contacts)
			contact.states_n = contact.states;

		Xn1 = Xn;
		Xn = X;
		Xn_ref = Xref;
		std::copy(fluxes.begin(), fluxes.end(), fluxes_n.begin());
		std::copy(fluxes_biot.begin(), fluxes_biot.end(), fluxes_biot_n.begin());
		std::copy(fluxes_ref.begin(), fluxes_ref.end(), fluxes_ref_n.begin());
		std::copy(fluxes_biot_ref.begin(), fluxes_biot_ref.end(), fluxes_biot_ref_n.begin());
		//std::copy(fluxes.begin(), fluxes.end(), fluxes_iter.begin());
		op_vals_arr_n = op_vals_arr;

		if (TIME_DEPENDENT_DISCRETIZATION)
		{
			if (FIND_EQUILIBRIUM)
			{
				mesh->tran_ref = mesh->tran;
				mesh->rhs_ref = mesh->rhs;
				mesh->tran_biot_ref = mesh->tran_biot;
				mesh->rhs_biot_ref = mesh->rhs_biot;
			}

			std::copy(mesh->tran_biot.begin(), mesh->tran_biot.end(), mesh->tran_biot_n.begin());
			std::copy(mesh->rhs_biot.begin(), mesh->rhs_biot.end(), mesh->rhs_biot_n.begin());
		}

		for (auto& contact : contacts)
		{
			assert(contact.rsf.theta.size() == contact.rsf.theta_n.size());
			std::copy(contact.rsf.theta.begin(), contact.rsf.theta.end(), contact.rsf.theta_n.begin());
		}

		dt1 = deltat;
		t += deltat;
	}
	return converged;
}

void engine_pm_cpu::update_uu_jacobian()
{
	static_cast<linsolv_bos_fs_cpr<N_VARS>*>(static_cast<linsolv_bos_gmres<N_VARS>*>(linear_solver)->prec)->do_update_uu();
}
int engine_pm_cpu::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};
