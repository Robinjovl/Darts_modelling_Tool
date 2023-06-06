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
#include <math.h>

#include "engine_super_mp_cpu.hpp"
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
int engine_super_mp_cpu<NC, NP, THERMAL>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                            std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                            sim_params *params_, timer_node *timer_)
{
	init_base(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);

	return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_mp_cpu<NC, NP, THERMAL>::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
	std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_, sim_params *params_, timer_node *timer_)
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
	const uint8_t n_state = get_n_state();
	z_var = get_z_var();
	nc_fl = get_n_comps();

	X_init.resize(n_vars * mesh->n_blocks);
	PV.resize(mesh->n_blocks);
	RV.resize(mesh->n_blocks);
	old_z.resize(nc);
	new_z.resize(nc);
	FIPS.resize(nc);
	old_z_fl.resize(nc_fl);
	new_z_fl.resize(nc_fl);

	fluxes.resize(n_vars * mesh->n_conns);
	std::fill_n(fluxes.begin(), fluxes.size(), 0.0);

	Xn = X = X_init;
	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		X_init[n_vars * i + P_VAR] = mesh->pressure[i];
		for (uint8_t c = 0; c < nc - 1; c++)
		{
			X_init[n_vars * i + Z_VAR + c] = mesh->composition[i * (nc - 1) + c];
		}

		PV[i] = mesh->volume[i] * mesh->poro[i];
		RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
	}
	if (THERMAL)
	{
		for (index_t i = 0; i < mesh_->n_blocks; i++)
		{
			X_init[N_VARS * i + T_VAR] = mesh->temperature[i];
		}
	}

	op_vals_arr.resize(n_ops * (mesh->n_blocks + mesh->n_bounds));
	op_ders_arr.resize(n_ops * n_state * (mesh->n_blocks + mesh->n_bounds));

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
	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		block_idxs[r].clear();
		op_axis_min[r].resize(nc + THERMAL);
		op_axis_max[r].resize(nc + THERMAL);
		for (int j = 0; j < nc + THERMAL; j++)
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

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_mp_cpu<NC, NP, THERMAL>::extract_Xop()
{
	const int state_size = NC_ + THERMAL;
	if (Xop.size() < (mesh->n_blocks + mesh->n_bounds) * state_size)
	{
		Xop.resize((mesh->n_blocks + mesh->n_bounds) * state_size);
	}
	// copy unknown variables
	std::copy(X.begin(), X.end(), Xop.begin());
	std::copy(mesh->pz_bounds.begin(), mesh->pz_bounds.end(), Xop.begin() + N_VARS * mesh->n_blocks);
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_mp_cpu<NC, NP, THERMAL>::init_jacobian_structure_mpfa(csr_matrix_base *jacobian)
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

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_mp_cpu<NC, NP, THERMAL>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	index_t n_blocks = mesh->n_blocks;
	index_t n_matrix = mesh->n_matrix;
	index_t n_res_blocks = mesh->n_res_blocks;
	index_t n_bounds = mesh->n_bounds;
	index_t n_conns = mesh->n_conns;

	const index_t *block_m = mesh->block_m.data();
	const index_t *block_p = mesh->block_p.data();
	const index_t *stencil = mesh->stencil.data();
	const index_t *offset = mesh->offset.data();
	const value_t *tran = mesh->tran.data();
	value_t *bc = mesh->bc.data();
	const value_t *rhs = mesh->rhs.data();
	const value_t *f = mesh->f.data();
	const value_t *V = mesh->volume.data();

	const value_t *tranD = mesh->tranD.data();
	const value_t *hcap = mesh->heat_capacity.data();
	const value_t *kin_fac = mesh->kin_factor.data(); // default value of 1
	const value_t *grav_coef = mesh->grav_coef.data();

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

	index_t r_ind, r_ind1, r_ind2, r_ind3, l_ind, l_ind1, upwd_idx[NP], d_upwd_idx[NP * NE];
	index_t j, diag_idx, jac_idx, nebr_jac_idx, csr_idx_start, csr_idx_end, upwd_jac_idx[NP], d_upwd_jac_idx[NP * NE], conn_id = 0, st_id = 0, conn_st_id = 0;
    value_t p_diff, gamma_p_diff, t_diff, gamma_t_diff, phi_i, phi_j, phi_avg, phi_0_avg, pc_diff[NP], diff_diff[NP * NE], phase_p_diff[NP], ZEROS[NP * NE];
	value_t avg_density, *buf, *buf_c, *buf_diff;
	uint8_t d, v, c, p;
	value_t CFL_in[NC], CFL_out[NC];
    value_t CFL_max_local = 0;
    int connected_with_well;

	std::fill_n(Jac, mesh->n_links * N_VARS * N_VARS, 0.0);
	std::fill(RHS.begin(), RHS.end(), 0.0);
	std::fill(fluxes.begin(), fluxes.end(), 0.0);
	std::fill_n(ZEROS, NP * NE, 0.0);

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
		// index of first entry for block i in CSR cols array
		csr_idx_start = rows[i];
		// index of last entry for block i in CSR cols array
		csr_idx_end = rows[i + 1];

		// loop over cell connections
		for (; block_m[conn_id] == i && conn_id < n_conns; conn_id++)
		{
			j = block_p[conn_id];
			if (j >= n_res_blocks && j < n_blocks)
				connected_with_well = 1;

			// [0] transmissibility multiplier
			/*value_t trans_mult = 1;
			value_t trans_mult_der_i[N_STATE];
			value_t trans_mult_der_j[N_STATE];
			if (params->trans_mult_exp > 0 && i < mesh->n_res_blocks && j < mesh->n_res_blocks)
			{
				// Calculate transmissibility multiplier:
				phi_i = op_vals_arr[i * N_OPS + PORO_OP];
				phi_j = op_vals_arr[j * N_OPS + PORO_OP];

				// Take average interface porosity:
				phi_avg = (phi_i + phi_j) * 0.5;
				phi_0_avg = (mesh->poro[i] + mesh->poro[j]) * 0.5;

				trans_mult = params->trans_mult_exp * pow(phi_avg, params->trans_mult_exp - 1) * 0.5;
				for (v = 0; v < N_STATE; v++)
				{
					trans_mult_der_i[v] = trans_mult * op_ders_arr[(i * N_OPS + PORO_OP) * N_STATE + v];
					trans_mult_der_j[v] = trans_mult * op_ders_arr[(j * N_OPS + PORO_OP) * N_STATE + v];
				}
				trans_mult = pow(phi_avg, params->trans_mult_exp);
			}
			else
			{
				for (v = 0; v < N_STATE; v++)
				{
					trans_mult_der_i[v] = 0;
					trans_mult_der_j[v] = 0;
				}
			}*/
	  
			nebr_jac_idx = csr_idx_end;
			// [1] fluid flux evaluation q = -Kn * \nabla p
			p_diff = t_diff = 0.0;
			std::fill_n(pc_diff, NP, 0.0);
			std::fill_n(diff_diff, NP * NE, 0.0);
			conn_st_id = offset[conn_id];
			for (st_id = csr_idx_start; conn_st_id < offset[conn_id + 1]; st_id++)
			{
				// skip entry if cell is different
				if (stencil[conn_st_id] != cols[st_id] && st_id < csr_idx_end) continue;

				// upwind index in jacobian
				if (st_id < csr_idx_end && cols[st_id] == j) nebr_jac_idx = st_id;

				if (stencil[conn_st_id] < n_blocks)	// matrix, fault or well cells
				{
					buf =		&X[N_VARS * stencil[conn_st_id]];
					buf_c =		&op_vals_arr[stencil[conn_st_id] * N_OPS + PC_OP];
					buf_diff =	&op_vals_arr[stencil[conn_st_id] * N_OPS + GRAD_OP];
				}
				else									// boundary condition
				{
					buf = &bc[N_VARS * (stencil[conn_st_id] - n_blocks)];
					buf_c =		&ZEROS[0]; // TODO: zeros only for Neumann
					buf_diff =	&ZEROS[0]; // TODO: zeros only for Neumann
				}


				p_diff += tran[conn_st_id] * buf[P_VAR];
				// heat conduction
				if (THERMAL)
					t_diff += tranD[conn_st_id] * buf[T_VAR];
				
				for (p = 0; p < NP; p++)
				{
					// capillary
					pc_diff[p] += tran[conn_st_id] * buf_c[p];
					// diffusion
					for (c = 0; c < NE; c++)
						diff_diff[c * NP + p] += tranD[conn_st_id] * buf_diff[c * NP + p];
				}

				conn_st_id++;
			}

			// [2] phase fluxes & upwind direction
			for (p = 0; p < NP; p++)
			{
				// avoid non-zero pc within region; TODO: discuss MPFA scheme in the presence of capillarity
				if (fabs(op_vals_arr[i * N_OPS + PC_OP + p] - op_vals_arr[j * N_OPS + PC_OP + p]) < 1.e-8)
					pc_diff[p] = 0.0;

				// calculate gravity term for phase p
				avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] + op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;

				// sum up gravity and cappillary terms
				phase_p_diff[p] = p_diff - pc_diff[p] + avg_density * rhs[conn_id];

				// identify upwind flow direction
				if (phase_p_diff[p] >= 0)
				{
					upwd_idx[p] = i;
					upwd_jac_idx[p] = diag_ind[i];
					for (c = 0; c < NE; c++)
					{
						if (c < NC) CFL_out[c] += phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
						fluxes[N_VARS * conn_id + P_VAR + c] += phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
					}
				}
				else
				{
					upwd_idx[p] = j;
					upwd_jac_idx[p] = nebr_jac_idx;
					for (c = 0; c < NE; c++)
					{
						if (c < NC && j < n_res_blocks) CFL_in[c] += -phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
						fluxes[N_VARS * conn_id + P_VAR + c] += phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
					}
				}

				// identify upwind diffusion direction
				for (c = 0; c < NE; c++)
				{
					if (diff_diff[c * NP + p] >= 0)
					{
						d_upwd_idx[c * NP + p] = i;
						d_upwd_jac_idx[c * NP + p] = diag_ind[i];
					}
					else
					{
						d_upwd_idx[c * NP + p] = j;
						d_upwd_jac_idx[c * NP + p] = nebr_jac_idx;
					}
				}
			}

			// [3] loop over stencil, contribution from UNKNOWNS to flux
			if (j < mesh->n_res_blocks)
				phi_avg = (mesh->poro[i] + mesh->poro[j]) * 0.5; // diffusion term depends on total porosity!
			else
				phi_avg = mesh->poro[i];
			conn_st_id = offset[conn_id];
			for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + 1]; st_id++)
			{
				if (stencil[conn_st_id] == cols[st_id])
				{
					//// mass fluxes
					for (p = 0; p < NP; p++)
					{
						// NE equations
						for (c = 0; c < NE; c++)
						{
							l_ind1 = st_id * N_VARS_SQ + (P_VAR + c) * N_VARS;						// jacobian
							r_ind = upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c;						// flux upwind multiplier
							r_ind1 = (stencil[conn_st_id] * N_OPS + PC_OP + p) * N_VARS;			// capillary operator
							r_ind2 = d_upwd_idx[c * NP + p] * N_OPS + UPSAT_OP + p;					// diffusion upwind multiplier
							r_ind3 = (stencil[conn_st_id] * N_OPS + GRAD_OP + c * NP + p) * N_VARS;	// diffusion operator
							// pressure contribution to flux
							Jac[l_ind1 + P_VAR] += dt * op_vals_arr[r_ind] * tran[conn_st_id];
							for (v = 0; v < NE; v++)
							{
								// capillary contribution to flux
								Jac[l_ind1 + v] -= dt * op_vals_arr[r_ind] * tran[conn_st_id] * op_ders_arr[r_ind1 + v];
								// component diffusion
								if (i < mesh->n_res_blocks && j < mesh->n_res_blocks)
									Jac[l_ind1 + v] += dt * phi_avg * op_vals_arr[r_ind2] * tranD[conn_st_id] * op_ders_arr[r_ind3 + v];
							}
						}
					}
					conn_st_id++;
				}
			}

			// [4] loop over pressure, composition & temperature
			for (p = 0; p < NP; p++)
			{
				for (c = 0; c < NE; c++)
				{
					l_ind = diag_idx + (P_VAR + c) * N_VARS;
					l_ind1 = nebr_jac_idx * N_VARS_SQ + (P_VAR + c) * N_VARS;
					r_ind = upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c;
					r_ind1 = (upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c) * N_STATE;
					for (v = 0; v < NE; v++)
					{
						if (upwd_jac_idx[p] < csr_idx_end)
						{
							Jac[upwd_jac_idx[p] * N_VARS_SQ + (P_VAR + c) * N_VARS + v] += dt * phase_p_diff[p] * op_ders_arr[r_ind1 + v];
						}
						// gravity
						Jac[l_ind + v] += dt * rhs[conn_id] * op_vals_arr[r_ind] * op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_STATE + v] / 2.0;// *grav_pc_der_i[v];
						if (nebr_jac_idx < csr_idx_end)
							Jac[l_ind1 + v] += dt * rhs[conn_id] * op_vals_arr[r_ind] * op_ders_arr[(j * N_OPS + GRAV_OP + p) * N_STATE + v] / 2.0;// *grav_pc_der_j[v];
					}
				}
			}

			// [5] Additional diffusion code here:   (phi_p * S_p) * (rho_p * D_cp * Delta_x_cp)  or (phi_p * S_p) * (kappa_p * Delta_T)
			// Only if block connection is between reservoir and reservoir cells!
			if (i < mesh->n_res_blocks && j < mesh->n_res_blocks)
			{
				// Add diffusion term to the residual:
				for (uint8_t c = 0; c < NE; c++)
				{
					for (uint8_t p = 0; p < NP; p++)
					{
						RHS[i * N_VARS + c] += dt * diff_diff[c * NP + p] * phi_avg * op_vals_arr[d_upwd_idx[c * NP + p] * N_OPS + UPSAT_OP + p]; // diffusion term

						// upwind multiplier
						if (d_upwd_jac_idx[c * NP + p] < csr_idx_end)
						{
							l_ind = d_upwd_jac_idx[c * NP + p] * N_VARS_SQ + c * N_VARS;
							r_ind = (i * N_OPS + UPSAT_OP + p) * N_VARS;
							for (uint8_t v = 0; v < N_VARS; v++)
								Jac[l_ind + v] += diff_diff[c * NP + p] * dt * phi_avg * op_ders_arr[r_ind + v];
						}
					}
				}
			}

			// [6] add heat conduction
			if (THERMAL)
			{
				t_diff = op_vals_arr[j * N_OPS + RE_TEMP_OP] - op_vals_arr[i * N_OPS + RE_TEMP_OP];
				gamma_t_diff = tranD[conn_id] * dt * t_diff;

				if (t_diff < 0)
				{
					// rock heat transfers flows from cell i to j
					RHS[i * N_VARS + T_VAR] -= gamma_t_diff * op_vals_arr[i * N_OPS + ROCK_COND] * (1 - mesh->poro[i]) * mesh->rock_cond[i];
					for (v = 0; v < NC; v++)
					{
						Jac[diag_idx + T_VAR * N_VARS + v] -= gamma_t_diff * op_ders_arr[(i * N_OPS + ROCK_COND) * N_STATE + v] * (1 - mesh->poro[i]) * mesh->rock_cond[i];
					}
					Jac[diag_idx + T_VAR * N_VARS + T_VAR] -= gamma_t_diff * op_ders_arr[(i * N_OPS + ROCK_COND) * N_STATE + T_VAR] * (1 - mesh->poro[i]) * mesh->rock_cond[i];
					if (nebr_jac_idx < csr_idx_end)
						Jac[nebr_jac_idx * N_VARS_SQ + T_VAR * N_VARS + T_VAR] -= tranD[conn_id] * dt * op_vals_arr[i * N_OPS + ROCK_COND] * (1 - mesh->poro[i]) * mesh->rock_cond[i];
					Jac[diag_idx + T_VAR * N_VARS + T_VAR] += tranD[conn_id] * dt * op_vals_arr[i * N_OPS + ROCK_COND] * (1 - mesh->poro[i]) * mesh->rock_cond[i];
				}
				else
				{
					// rock heat transfers flows from cell j to i
					if (j < mesh->n_blocks)
					{
						RHS[i * N_VARS + T_VAR] -= gamma_t_diff * op_vals_arr[j * N_OPS + ROCK_COND] * (1 - mesh->poro[j]) * mesh->rock_cond[j]; // energy cond operator
						for (v = 0; v < NC; v++)
						{
							Jac[nebr_jac_idx * N_VARS_SQ + T_VAR * N_VARS + v] -= gamma_t_diff * op_ders_arr[(j * N_OPS + ROCK_COND) * N_STATE + v] * (1 - mesh->poro[j]) * mesh->rock_cond[j];
						}
						Jac[nebr_jac_idx * N_VARS_SQ + T_VAR * N_VARS + T_VAR] -= gamma_t_diff * op_ders_arr[(j * N_OPS + ROCK_COND) * N_STATE + T_VAR] * (1 - mesh->poro[j]) * mesh->rock_cond[j];
						Jac[diag_idx + NC * N_VARS + T_VAR] += tranD[conn_id] * dt * op_vals_arr[j * N_OPS + ROCK_COND] * (1 - mesh->poro[j]) * mesh->rock_cond[j];
						Jac[nebr_jac_idx * N_VARS_SQ + NC * N_VARS + T_VAR] -= tranD[conn_id] * dt * op_vals_arr[j * N_OPS + ROCK_COND] * (1 - mesh->poro[j]) * mesh->rock_cond[j];
					}
					else
					{
						RHS[i * N_VARS + T_VAR] -= gamma_t_diff * op_vals_arr[j * N_OPS + ROCK_COND] * (1 - mesh->poro[i]) * mesh->rock_cond[j];
						Jac[diag_idx + NC * N_VARS + T_VAR] += tranD[conn_id] * dt * op_vals_arr[j * N_OPS + ROCK_COND] * (1 - mesh->poro[i]) * mesh->rock_cond[j];
					}
				}
			}
			// [7] residual
			for (c = 0; c < NE; c++)
			{
				RHS[i * N_VARS + P_VAR + c] += dt * fluxes[N_VARS * conn_id + P_VAR + c];
			}
		}

		// [8] accumulation terms
		for (c = 0; c < NE; c++)
		{
			RHS[i * N_VARS + P_VAR + c] += PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]);
			for (v = 0; v < N_STATE; v++)
			{
				Jac[diag_idx + (P_VAR + c) * N_VARS + P_VAR + v] += PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_STATE + v];
			}
		}
		if (THERMAL)
		{
			RHS[i * N_VARS + T_VAR] += RV[i] * (op_vals_arr[i * N_OPS + RE_INTER_OP] - op_vals_arr_n[i * N_OPS + RE_INTER_OP]) * hcap[i];

			for (v = 0; v < NE; v++)
			{
				Jac[diag_idx + T_VAR * N_VARS + v] += RV[i] * op_ders_arr[(i * N_OPS + RE_INTER_OP) * N_STATE + v] * hcap[i];
			}
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

		if (i < n_res_blocks)
		{
			for (c = 0; c < NE; c++)
			{
				RHS[i * N_VARS + P_VAR + c] += V[i] * dt * f[i * N_VARS + P_VAR + c];
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

  for (ms_well *w : wells)
  {
    value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
    w->add_to_jacobian(dt, X, jac_well_head, RHS);
  }

  return 0;
};



template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_mp_cpu<NC, NP, THERMAL>::adjoint_gradient_assembly(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
	return 0;
};



template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_mp_cpu<NC, NP, THERMAL>::run_single_newton_iteration(value_t deltat)
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

//template<uint8_t NC, uint8_t NP, , bool THERMAL>
//double
//engine_super_mp_cpu<NC, NP, THERMAL>::calc_newton_residual()
//{
//	double residual = 0, res;
//
//	std::vector <value_t> &hcap = mesh->heat_capacity;
//
//	residual = 0;
//	for (int i = 0; i < mesh->n_blocks; i++)
//	{
//		for (int c = 0; c < NC; c++)
//		{
//			res = fabs(RHS[i * N_VARS + c] / (PV[i] * op_vals_arr[i * N_OPS + c]));
//			if (res > residual)
//				residual = res;
//		}
//
//		if (THERMAL)
//		{
//			res = fabs(RHS[i * N_VARS + T_VAR] / (PV[i] * op_vals_arr[i * N_OPS + NC] + RV[i] * op_vals_arr[i * N_OPS + RE_INTER_OP] * hcap[i]));
//			if (res > residual)
//				residual = res;
//		}
//	}
//	return residual;
//}

// compositional, kinetic (H2O, CO2, Ca+2, CO3-2, CaCO3):
//template class engine_super_mp_cpu<2, 2, 1>;
//template struct recursive_instantiator_nc_np<engine_super_mp_cpu, 2, MAX_NC, 1>;
//template struct recursive_instantiator_nc_np<engine_super_mp_cpu, 2, MAX_NC, 2>;
//template struct recursive_instantiator_nc_np<engine_super_mp_cpu, 2, MAX_NC, 3>;
