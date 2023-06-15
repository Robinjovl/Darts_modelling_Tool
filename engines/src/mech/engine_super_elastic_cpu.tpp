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

#include "mech/engine_super_elastic_cpu.hpp"
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
int engine_super_elastic_cpu<NC, NP, THERMAL>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                            std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                            sim_params *params_, timer_node *timer_)
{
  newton_update_coefficient = 1.0;
  dev_u = dev_p = dev_e = well_residual_last_dt = std::numeric_limits<value_t>::infinity();
  std::fill(dev_z, dev_z + NC_, std::numeric_limits<value_t>::infinity());
  output_counter = 0;
  FIND_EQUILIBRIUM = false;
  contact_solver = pm::RETURN_MAPPING;
  geomechanics_mode.resize(mesh_->n_blocks, 0);

  init_base(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);

  return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
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
		/*case sim_params::CPU_GMRES_FS_CPR:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linsolv_iface *fs_cpr = new linsolv_bos_fs_cpr<N_VARS>(P_VAR, Z_VAR, U_VAR, NC_);
			static_cast<linsolv_bos_fs_cpr<N_VARS> *>(fs_cpr)->set_prec(new linsolv_bos_amg<1>, new linsolv_hypre_amg<1>); //new linsolv_amg1r5<1>);
			linear_solver->set_prec(fs_cpr);
			break;
		}*/
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
	fluxes_n.resize(n_vars * mesh->n_conns);
	fluxes_biot.resize(n_vars * mesh->n_conns);
	fluxes_biot_n.resize(n_vars * mesh->n_conns);
	fluxes_ref.resize(n_vars * mesh->n_conns, 0.0);
	fluxes_biot_ref.resize(n_vars * mesh->n_conns, 0.0);
	fluxes_ref_n.resize(n_vars * mesh->n_conns, 0.0);
	fluxes_biot_ref_n.resize(n_vars * mesh->n_conns, 0.0);
	eps_vol.resize(mesh->n_matrix);

	std::fill_n(fluxes.begin(), fluxes.size(), 0.0);
	std::fill_n(fluxes_n.begin(), fluxes_n.size(), 0.0);

	Xn_ref = Xref = Xn = X = X_init;
	for (index_t i = 0; i < mesh->ref_pressure.size(); i++)
		Xref[N_VARS * i + P_VAR] = Xn_ref[N_VARS * i + P_VAR] = mesh->ref_pressure[i];

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		X_init[n_vars * i + P_VAR] = mesh->pressure[i];
		for (uint8_t c = 0; c < nc - 1; c++)
		{
			X_init[n_vars * i + Z_VAR + c] = mesh->composition[i * (nc - 1) + c];
		}
		for (uint8_t d = 0; d < ND; d++)
		{
			X_init[n_vars * i + U_VAR + d] = mesh->displacement[ND * i + d];
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
	init_jacobian_structure_pme(Jacobian);

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
void engine_super_elastic_cpu<NC, NP, THERMAL>::extract_Xop()
{
	const int state_size = NC_ + THERMAL;

	if (Xop.size() < (mesh->n_blocks + mesh->n_bounds) * state_size)
	{
		Xop.resize((mesh->n_blocks + mesh->n_bounds) * state_size);
	}
	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		Xop[i * state_size] = X[i * N_VARS + P_VAR];
		for (uint8_t c = 0; c < NC_ - 1; c++)
		{
			Xop[i * state_size + c + 1] = X[i * N_VARS + Z_VAR + c];
		}
	}
	for (index_t i = 0; i < mesh->n_bounds; i++)
	{
		Xop[(mesh->n_blocks + i) * state_size] = mesh->pz_bounds[i * state_size];
		for (uint8_t c = 0; c < NC_ - 1; c++)
		{
			Xop[(mesh->n_blocks + i) * state_size + c + 1] = mesh->pz_bounds[i * state_size + c + 1];
		}
	}

	if (THERMAL)
	{
		for (index_t i = 0; i < mesh->n_blocks; i++)
		{
			Xop[i * state_size + NC_] = X[i * N_VARS + T_VAR];
		}
		for (index_t i = 0; i < mesh->n_bounds; i++)
		{
			Xop[(mesh->n_blocks + i) * state_size + NC_] = mesh->pz_bounds[i * state_size + NC_];
		}
	}
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::init_jacobian_structure_pme(csr_matrix_base *jacobian)
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
int engine_super_elastic_cpu<NC, NP, THERMAL>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
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
  const value_t *tran_biot = mesh->tran_biot.data();
  const value_t *tran_th_expn = mesh->tran_th_expn.data();
  value_t *bc = mesh->bc.data();
  value_t *bc_prev = mesh->bc_n.data();
  value_t *bc_ref = mesh->bc_ref.data();
  const value_t *rhs = mesh->rhs.data();
  const value_t *rhs_biot = mesh->rhs_biot.data();
  const value_t *f = mesh->f.data();
  const value_t *V = mesh->volume.data();
  const value_t *kd = mesh->drained_compressibility.data();
  const value_t *th_poro = mesh->th_poro.data();
  const value_t *biot = mesh->biot.data();
  const value_t *poro = mesh->poro.data();
  value_t *pz_bounds = mesh->pz_bounds.data();
  const value_t *p_ref = mesh->ref_pressure.data();
  value_t *t_ref = mesh->ref_temperature.data();
  const value_t *eps_vol_ref = mesh->ref_eps_vol.data();

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

	std::fill_n(Jac, N_VARS * N_VARS * mesh->n_links, 0.0);
	std::fill(RHS.begin(), RHS.end(), 0.0);
	std::fill(fluxes.begin(), fluxes.end(), 0.0);
	std::fill(fluxes_biot.begin(), fluxes_biot.end(), 0.0);

	index_t j, upwd_jac_idx[NP], nebr_jac_idx, upwd_idx[NP], diag_idx, conn_id = 0, st_id = 0, conn_st_id = 0, 
		csr_idx_start, csr_idx_end;
	index_t l_ind, r_ind, l_ind1, r_ind1;
	value_t *cur_bc, *cur_bc_prev, *ref_bc, biot_mult, biot_cur, comp_mult, phi, phi_n, *buf, *buf_prev, p_ref_cur, *n;
	uint8_t d, v, c, p;
	value_t gamma_p_diff, p_diff, phase_p_diff[NP], t_diff, gamma_t_diff, phi_i, phi_j, phi_avg, phi_0_avg;
    value_t CFL_in[NC], CFL_out[NC];
    value_t CFL_max_local = 0;
	value_t avg_density, eff_density;
	const value_t rho_s = 2650.0;

    int connected_with_well;

    for (index_t i = start; i < end; ++i)
    { // loop over grid blocks

      // initialize the CFL_in and CFL_out
      for (c = 0; c < NC; c++)
      {
        CFL_out[c] = 0;
        CFL_in[c] = 0;
      }

	  // index of diagonal block entry for block i in CSR values array
	  diag_idx = N_VARS_SQ * diag_ind[i];
	  // index of first entry for block i in CSR cols array
	  csr_idx_start = rows[i];
	  // index of last entry for block i in CSR cols array
	  csr_idx_end = rows[i + 1];

	  connected_with_well = 0;
	  biot_mult = 0.0;
	  eff_density = 0.0;
	  for (p = 0; p < NP; p++)
	  {
		  eff_density += op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
	  }
	  eff_density -= rho_s;

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
		  // [1] fluid flux evaluation q = -Kn * \nabla p & biot flux qb = u * n
		  p_diff = t_diff = 0.0;
		  conn_st_id = offset[conn_id];
		  for (st_id = csr_idx_start; conn_st_id < offset[conn_id + 1]; st_id++)
		  {
			  // skip entry if cell is different
			  if (stencil[conn_st_id] != cols[st_id] && st_id < csr_idx_end) continue;

			  // upwind index in jacobian
			  if (st_id < csr_idx_end && cols[st_id] == j) nebr_jac_idx = st_id;
			  
			  if (stencil[conn_st_id] < n_blocks)	// matrix, fault or well cells
			  {
				  r_ind = N_VARS * stencil[conn_st_id];
				  buf = &X[r_ind];
				  buf_prev = &Xn[r_ind];
			  }
			  else									// boundary condition
			  {
				  r_ind = N_VARS * (stencil[conn_st_id] - n_blocks);
				  buf = &bc[r_ind];
				  buf_prev = &bc_prev[r_ind];
			  }
			  // displacement contribution
			  r_ind = conn_st_id * NT_SQ + P_VAR_T * NT + U_VAR_T;
			  for (d = 0; d < ND; d++)
			  {
				  // fluid flux
				  p_diff += tran[r_ind + d] * buf[U_VAR + d];
				  // flux of displacements (u * n)
				  biot_mult += tran_biot[r_ind + d] * buf[U_VAR + d];
				  // time derivative of the last flux = flux of matrix mass due to structure movement
				  fluxes_biot[N_VARS * conn_id + P_VAR] += tran_biot[r_ind + d] * 
					  (buf[U_VAR + d] - buf_prev[U_VAR + d]) / dt;
			  }
			  // pressure contribution
			  r_ind = conn_st_id * NT_SQ + P_VAR_T * NT + P_VAR_T;
			  p_diff += tran[r_ind] * buf[P_VAR];
			  biot_mult += tran_biot[r_ind] * buf[P_VAR];
			  fluxes_biot[N_VARS * conn_id + P_VAR] += tran_biot[r_ind] * 
				  (buf[P_VAR] - buf_prev[P_VAR]) / dt;
			  // heat conduction
			  if (THERMAL)
				t_diff += tranD[conn_st_id] * buf[T_VAR];

			  conn_st_id++;
		  }
		  // [2] phase fluxes & upwind direction
		  for (p = 0; p < NP; p++)
		  {
			  // calculate gravity term for phase p
			  avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] + op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;

			  // sum up gravity and cappillary terms
			  phase_p_diff[p] = p_diff + avg_density * rhs[NT * conn_id + P_VAR_T] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];
			  //?????//phase_biot_mult += avg_density * rhs_biot[NT * conn_id + P_VAR];

			  // identify upwind direction
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
		  }
		  // [3] loop over stencil, contribution from UNKNOWNS to flux
		  conn_st_id = offset[conn_id];
		  for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + 1]; st_id++)
		  {
			  if (stencil[conn_st_id] == cols[st_id])
			  {
				  p_ref_cur = p_ref[stencil[conn_st_id]];
				  //// momentum fluxes
				  l_ind = N_VARS * conn_id + U_VAR;
				  r_ind = stencil[conn_st_id] * N_VARS;
				  for (d = 0; d < ND; d++)
				  {
					  // displacements contribution
					  l_ind1 = st_id * N_VARS_SQ + (U_VAR + d) * N_VARS;
					  r_ind1 = conn_st_id * NT_SQ + (U_VAR_T + d) * NT + U_VAR_T;
					  for (v = 0; v < ND; v++)
					  {
						  fluxes[l_ind + d] += tran[r_ind1 + v] * X[r_ind + U_VAR + v];
						  fluxes_biot[l_ind + d] += tran_biot[r_ind1 + v] * X[r_ind + U_VAR + v];
						  Jac[l_ind1 + U_VAR + v] += tran[r_ind1 + v];
						  Jac[l_ind1 + U_VAR + v] += tran_biot[r_ind1 + v];
					  }
					  // pressure contribution
					  r_ind1 = conn_st_id * NT_SQ + (U_VAR_T + d) * NT + P_VAR_T;
					  fluxes[l_ind + d] += tran[r_ind1] * X[r_ind + P_VAR];
					  fluxes_biot[l_ind + d] += tran_biot[r_ind1] * X[r_ind + P_VAR];
					  Jac[l_ind1 + P_VAR] += tran[r_ind1];
					  Jac[l_ind1 + P_VAR] += tran_biot[r_ind1];
					  // subtract reference pressure (when stress = 0)
					  fluxes[l_ind + d] += -tran[r_ind1] * p_ref_cur;
					  fluxes_biot[l_ind + d] += -tran_biot[r_ind1] * p_ref_cur;
				  }
				  //// mass fluxes
				  r_ind = stencil[conn_st_id] * N_VARS;
				  r_ind1 = conn_st_id * NT_SQ + P_VAR_T * NT;
				  for (p = 0; p < NP; p++)
				  {
					  // NE equations
					  for (c = 0; c < NE; c++)
					  {
						  l_ind1 = st_id * N_VARS_SQ + (P_VAR + c) * N_VARS;
						  // displacements contribution to flux
						  for (v = 0; v < ND; v++)
						  {
							  Jac[l_ind1 + U_VAR + v] += dt * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c] * tran[r_ind1 + U_VAR_T + v];
						  }
						  // pressure contribution to flux
						  Jac[l_ind1 + P_VAR] += dt * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c] * tran[r_ind1 + P_VAR_T];
					  }
				  }
				  // biot term in accumulation
				  for (c = 0; c < NE; c++)
				  {
					  l_ind1 = st_id * N_VARS_SQ + (c + P_VAR) * N_VARS;
					  // displacements contribution to flux
					  for (v = 0; v < ND; v++)
					  {
						  RHS[i * N_VARS + P_VAR + c] += tran_biot[r_ind1 + U_VAR_T + v] *
							  (op_vals_arr[i * N_OPS + ACC_OP + c] * X[r_ind + U_VAR + v] - op_vals_arr_n[i * N_OPS + ACC_OP + c] * Xn[r_ind + U_VAR + v]);
						  Jac[l_ind1 + U_VAR + v] += op_vals_arr[i * N_OPS + ACC_OP + c] * tran_biot[r_ind1 + U_VAR_T + v];
					  }
					  // pressure contribution to flux
					  RHS[i * N_VARS + P_VAR + c] += tran_biot[r_ind1 + P_VAR_T] *
						  (op_vals_arr[i * N_OPS + ACC_OP + c] * X[r_ind + P_VAR] - op_vals_arr_n[i * N_OPS + ACC_OP + c] * Xn[r_ind + P_VAR]);
					  Jac[l_ind1 + P_VAR] += op_vals_arr[i * N_OPS + ACC_OP + c] * tran_biot[r_ind1 + P_VAR_T];
				  }
				  // rock energy
				  if (THERMAL)
				  {
					  l_ind1 = st_id * N_VARS_SQ + T_VAR * N_VARS;
					  // displacements contribution to flux
					  for (v = 0; v < ND; v++)
					  {
						  RHS[i * N_VARS + T_VAR] -= hcap[i] * tran_biot[r_ind1 + U_VAR_T + v] *
							  (op_vals_arr[i * N_OPS + RE_INTER_OP] * X[r_ind + U_VAR + v] - op_vals_arr_n[i * N_OPS + RE_INTER_OP] * Xn[r_ind + U_VAR + v]);
						  Jac[l_ind1 + U_VAR + v] -= hcap[i] * tran_biot[r_ind1 + U_VAR_T + v] * op_vals_arr[i * N_OPS + RE_INTER_OP];
					  }
					  // pressure contribution to flux
					  RHS[i * N_VARS + T_VAR] -= hcap[i] * tran_biot[r_ind1 + P_VAR_T] *
						  (op_vals_arr[i * N_OPS + RE_INTER_OP] * X[r_ind + P_VAR] - op_vals_arr_n[i * N_OPS + RE_INTER_OP] * Xn[r_ind + P_VAR]);
					  Jac[l_ind1 + P_VAR] -= hcap[i] * tran_biot[r_ind1 + P_VAR_T] * op_vals_arr[i * N_OPS + RE_INTER_OP];			      
				  }
				  // biot term in porosity in gravitational forces
				  for (d = 0; d < ND; d++)
				  {
					  l_ind1 = st_id * N_VARS_SQ + (U_VAR + d) * N_VARS;
					  // displacements contribution to flux
					  for (v = 0; v < ND; v++)
					  {
						  RHS[i * N_VARS + U_VAR + d] += V[i] * f[i * N_VARS + U_VAR + d] * eff_density * tran_biot[r_ind1 + U_VAR_T + v] * X[r_ind + U_VAR + v];
						  Jac[l_ind1 + U_VAR + v] += V[i] * f[i * N_VARS + U_VAR + d] * eff_density * tran_biot[r_ind1 + U_VAR_T + v];
					  }
					  // pressure contribution to flux
					  RHS[i * N_VARS + U_VAR + d] += V[i] * f[i * N_VARS + U_VAR + d] * eff_density * tran_biot[r_ind1 + P_VAR_T] * X[r_ind + P_VAR];
					  Jac[l_ind1 + P_VAR] += V[i] * f[i * N_VARS + U_VAR + d] * eff_density * tran_biot[r_ind1 + P_VAR_T];
				  }

				  conn_st_id++;
			  }
		  }
		  // [4] loop over stencil, contribution from BOUNDARY CONDITIONS to flux
		  for (; conn_st_id < offset[conn_id + 1]; conn_st_id++)
		  {
			  if (stencil[conn_st_id] >= n_blocks)
			  {
				  r_ind = N_VARS * (stencil[conn_st_id] - n_blocks);
				  cur_bc = &bc[r_ind];
				  cur_bc_prev = &bc_prev[r_ind];
				  ref_bc = &bc_ref[r_ind];
				  // momentum balance
				  l_ind = N_VARS * conn_id + U_VAR;
				  for (d = 0; d < ND; d++)
				  {
					  // displacement contribution
					  r_ind = conn_st_id * NT_SQ + (U_VAR_T + d) * NT;
					  for (v = 0; v < ND; v++)
					  {
						  fluxes[l_ind + d] += tran[r_ind + U_VAR_T + v] * (cur_bc[U_VAR + v] - ref_bc[U_VAR + v]);
						  fluxes_biot[l_ind + d] += tran_biot[r_ind + U_VAR_T + v] * (cur_bc[U_VAR + v] - ref_bc[U_VAR + v]);
					  }
					  // pressure contribution
					  fluxes[l_ind + d] += tran[r_ind + P_VAR_T] * (cur_bc[P_VAR] - ref_bc[P_VAR]);
					  fluxes_biot[l_ind + d] += tran_biot[r_ind + P_VAR_T] * (cur_bc[P_VAR] - ref_bc[P_VAR]);
				  }
				  // mass balance
				  // biot term in accumulation
				  r_ind = conn_st_id * NT_SQ + P_VAR_T * NT;
				  for (c = 0; c < NE; c++)
				  {
					  // displacement contribution
					  for (v = 0; v < ND; v++)
					  {
						  // biot
						  RHS[i * N_VARS + P_VAR + c] += tran_biot[r_ind + U_VAR_T + v] *
							  (op_vals_arr[i * N_OPS + ACC_OP + c] * cur_bc[U_VAR + v] - op_vals_arr_n[i * N_OPS + ACC_OP + c] * cur_bc_prev[U_VAR + v]);
					  }
					  // pressure contribution
					  RHS[i * N_VARS + P_VAR + c] += tran_biot[r_ind + P_VAR_T] *
						  (op_vals_arr[i * N_OPS + ACC_OP + c] * cur_bc[P_VAR] - op_vals_arr_n[i * N_OPS + ACC_OP + c] * cur_bc_prev[P_VAR]);
				  }
				  // rock energy
				  if (THERMAL)
				  {
					  // displacements contribution to flux
					  for (v = 0; v < ND; v++)
					  {
						  RHS[i * N_VARS + T_VAR] -= hcap[i] * tran_biot[r_ind + U_VAR_T + v] *
							  (op_vals_arr[i * N_OPS + RE_INTER_OP] * cur_bc[U_VAR + v] - op_vals_arr_n[i * N_OPS + RE_INTER_OP] * cur_bc_prev[U_VAR + v]);
					  }
					  // pressure contribution to flux
					  RHS[i * N_VARS + T_VAR] -= hcap[i] * tran_biot[r_ind + P_VAR_T] *
						  (op_vals_arr[i * N_OPS + RE_INTER_OP] * cur_bc[P_VAR] - op_vals_arr_n[i * N_OPS + RE_INTER_OP] * cur_bc_prev[P_VAR]);	      
				  }
				  // biot term in porosity in gravitational forces
				  for (d = 0; d < ND; d++)
				  {
					  // displacements contribution to flux
					  for (v = 0; v < ND; v++)
					  {
						  RHS[i * N_VARS + U_VAR + d] += V[i] * f[i * N_VARS + U_VAR + d] * eff_density * tran_biot[r_ind + U_VAR_T + v] * cur_bc[U_VAR + v];
					  }
					  // pressure contribution to flux
					  RHS[i * N_VARS + U_VAR + d] += V[i] * f[i * N_VARS + U_VAR + d] * eff_density * tran_biot[r_ind + P_VAR_T] * cur_bc[P_VAR];
				  }
			  }
		  }
		  // [5] loop over pressure, composition & temperature
		  for (p = 0; p < NP; p++)
		  {
			  // calculate partial derivatives for gravity and capillary terms
			  value_t grav_pc_der_i[N_VARS - ND];
			  value_t grav_pc_der_j[N_VARS - ND];
			  r_ind = (i * N_OPS + GRAV_OP + p) * N_STATE;
			  r_ind1 = (j * N_OPS + GRAV_OP + p) * N_STATE;
			  for (v = 0; v < N_VARS - ND; v++)
			  {
				  grav_pc_der_i[v] = -op_ders_arr[r_ind + v] * rhs[NT * conn_id + P_VAR_T] / 2 - op_ders_arr[r_ind + v];
				  grav_pc_der_j[v] = -op_ders_arr[r_ind1 + v] * rhs[NT * conn_id + P_VAR_T] / 2 + op_ders_arr[r_ind1 + v];
			  }

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
					  Jac[l_ind + v] += dt * rhs[conn_id * NT + P_VAR_T] * op_vals_arr[r_ind] * grav_pc_der_i[v];
					  if (nebr_jac_idx < csr_idx_end)
						  Jac[l_ind1 + v] += dt * rhs[conn_id * NT + P_VAR_T] * op_vals_arr[r_ind] * grav_pc_der_j[v];
				  }
			  }
		  }
		  // [?] extra loop for gravity in biot for flux
		  // [6] (saturation ??? ) thermal expansion & fluid gravity for momentum balance
		  if (THERMAL)
		  {
			  l_ind = N_VARS * conn_id + U_VAR;
			  l_ind1 = upwd_jac_idx[0] * N_VARS_SQ + T_VAR;
			  if (upwd_idx[0] < n_blocks)
			  {
				  cur_bc = &X[upwd_idx[0] * N_VARS + T_VAR];
				  ref_bc = &t_ref[upwd_idx[0]];
			  }
			  else
			  {
				  r_ind = N_STATE * (upwd_idx[0] - n_blocks);
				  cur_bc = &pz_bounds[r_ind + T_VAR];
				  ref_bc = &pz_bounds[r_ind + T_VAR];// &t_ref[upwd_idx[0]];
			  }
			  // thermal induced stresses
			  for (d = 0; d < ND; d++)
			  {
				  fluxes[l_ind + d] += tran_th_expn[conn_id * ND + d] * (cur_bc[0] - ref_bc[0]);
				  if (upwd_jac_idx[0] < csr_idx_end)
					  Jac[l_ind1 + (U_VAR + d) * N_VARS] += tran_th_expn[conn_id * ND + d];
				  //fluxes[N_VARS * conn_id + U_VAR + d] += op_vals_arr[i * N_OPS + GRAV_OP] * rhs[N_VARS * conn_id + U_VAR + d];
				  //fluxes_biot[N_VARS * conn_id + U_VAR + d] += op_vals_arr[i * N_OPS + GRAV_OP] * rhs_biot[N_VARS * conn_id + U_VAR + d];
			  }
		  // [7] add heat conduction
		  /*if (THERMAL)
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
			  }*/
		  }
		  // [8] residual
		  for (c = 0; c < NE; c++)
		  {
			  RHS[i * N_VARS + P_VAR + c] += dt * fluxes[N_VARS * conn_id + P_VAR + c];
		  }
		  for (d = 0; d < ND; d++)
		  {
			  RHS[i * N_VARS + U_VAR + d] += fluxes[N_VARS * conn_id + U_VAR + d];
			  RHS[i * N_VARS + U_VAR + d] += fluxes_biot[N_VARS * conn_id + U_VAR + d];
			  //Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR] += (rhs[NT * conn_id + U_VAR_T + d] + rhs_biot[NT * conn_id + U_VAR_T + d]) * op_ders_arr[(i * N_OPS + GRAV_OP) * N_STATE];
		  }
	  }

	  // [9] accumulation for mass balance
	  // porosity

	  phi = poro[i];
	  phi_n = poro[i];
	  if (i >= n_matrix)
	  {
		  biot_mult = comp_mult = 0.0;
	  }
	  else
	  {
		  eps_vol[i] = biot_mult / V[i];
		  //biot_mult -= V[i] * eps_vol_ref[i];
		  //for (c = 0; c < NE; c++)
		  //	RHS[i * N_VARS + P_VAR + c] += -V[i] * eps_vol_ref[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]);
		  if (!geomechanics_mode[i])
		  {
			  biot_cur = (biot[i * ND * ND] + biot[i * ND * ND + ND + 1] + biot[i * ND * ND + 2 * ND + 2]) / 3.0; // one-third of the Biot tensor trace
			  comp_mult = (biot_cur != 0) ? (biot_cur - poro[i]) * (1 - biot_cur) / kd[i] : 1.0 / kd[i];
			  phi += comp_mult * (X[i * N_VARS + P_VAR] - p_ref[i]) - eps_vol_ref[i];
			  phi_n += comp_mult * (Xn[i * N_VARS + P_VAR] - p_ref[i]) - eps_vol_ref[i];
			  if (THERMAL)
			  {
				  phi -= th_poro[i] * (X[i * N_VARS + T_VAR] - t_ref[i]);
				  phi_n -= th_poro[i] * (Xn[i * N_VARS + T_VAR] - t_ref[i]);
			  }
		  }
		  else
		  {
			  comp_mult = 0.0;
		  }
	  }
	  /*if (FIND_EQUILIBRIUM || geomechanics_mode[i])
	  {
		  for (c = 0; c < NE; c++)
			  Jac[diag_idx + (P_VAR + c) * N_VARS + P_VAR + c] = V[i];
	  }
	  else
	  {*/
		  for (c = 0; c < NE; c++)
		  {
			  RHS[i * N_VARS + P_VAR + c] += V[i] * (phi * op_vals_arr[i * N_OPS + ACC_OP + c] - phi_n * op_vals_arr_n[i * N_OPS + ACC_OP + c]);
			  Jac[diag_idx + (P_VAR + c) * N_VARS + P_VAR] += V[i] * comp_mult * op_vals_arr[i * N_OPS + ACC_OP + c];
			  for (v = 0; v < N_STATE; v++)
			  {
				  Jac[diag_idx + (P_VAR + c) * N_VARS + P_VAR + v] += V[i] * phi * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_STATE + v];
			  }
			  if (!geomechanics_mode[i] && THERMAL)
				Jac[diag_idx + (P_VAR + c) * N_VARS + T_VAR] -= V[i] * th_poro[i] * op_vals_arr[i * N_OPS + ACC_OP + c];
		  }
	  //}

      // [9] finally add rock energy
      // + rock energy (no rock compressibility included in these computations)
      if (THERMAL && !FIND_EQUILIBRIUM)
      {
        RHS[i * N_VARS + T_VAR] += V[i] * ((1.0 - phi) * op_vals_arr[i * N_OPS + RE_INTER_OP] - (1.0 - phi_n) * op_vals_arr_n[i * N_OPS + RE_INTER_OP]) * hcap[i];

        for (v = 0; v < NE; v++)
        {
          Jac[diag_idx + T_VAR * N_VARS + v] +=  V[i] * (1.0 - phi) * op_ders_arr[(i * N_OPS + RE_INTER_OP) * N_STATE + v] * hcap[i];
        } // end of fill offdiagonal part + contribute to diagonal

		Jac[diag_idx + T_VAR * N_VARS + P_VAR] -= V[i] * comp_mult * op_vals_arr[i * N_OPS + RE_INTER_OP] * hcap[i];
		Jac[diag_idx + T_VAR * N_VARS + T_VAR] += V[i] * th_poro[i] * op_vals_arr[i * N_OPS + RE_INTER_OP] * hcap[i];
      }

      // calc CFL for reservoir cells, not connected with wells
      if (i < mesh->n_res_blocks && !connected_with_well)
      {
        for (c = 0; c < NC; c++)
        {
          if ((PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]) > 1e-4)
          {
            CFL_max_local = std::max(CFL_max_local, CFL_in[c] / (PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]));
            CFL_max_local = std::max(CFL_max_local, CFL_out[c] / (PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c]));
          }
        }
      }

	  // gravitational forces
	  if (i < n_res_blocks)
	  {
		  for (c = 0; c < NE; c++)
		  {
			  RHS[i * N_VARS + P_VAR + c] += V[i] * dt * f[i * N_VARS + P_VAR + c];
		  }
		  for (d = 0; d < ND; d++)
		  {
			  for (p = 0; p < NP; p++)
			  {
				  RHS[i * N_VARS + U_VAR + d] += phi * V[i] * f[i * N_VARS + U_VAR + d] * op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
				  Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR] += comp_mult * V[i] * f[i * N_VARS + U_VAR + d] * op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
				  for (v = 0; v < N_STATE; v++)
				  {
					  Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR + v] += phi * V[i] * f[i * N_VARS + U_VAR + d] * 
						  (op_vals_arr[i * N_OPS + SAT_OP + p] * op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_STATE + v] + 
							  op_ders_arr[(i * N_OPS + SAT_OP + p) * N_STATE + v] * op_vals_arr[i * N_OPS + GRAV_OP + p]);
				  }
			  }
			  RHS[i * N_VARS + U_VAR + d] += (1 - phi) * V[i] * f[i * N_VARS + U_VAR + d] * rho_s;
			  Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR] -= comp_mult * V[i] * f[i * N_VARS + U_VAR + d] * rho_s;
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
    value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
	w->add_to_jacobian(dt, X, jac_well_head, RHS);
	for (uint8_t d = 0; d < ND; d++)
	{
		Jac[N_VARS_SQ * diag_ind[w->well_head_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
		Jac[N_VARS_SQ * diag_ind[w->well_body_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
	}
  }

  return 0;
};

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::run_single_newton_iteration(value_t deltat)
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

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::apply_newton_update(value_t dt)
{
	timer->node["newton update"].node["composition correction"].start();
	if (nc > 1)
	{
		if (params->log_transform == 1)
		{
			apply_composition_correction_new(X, dX);
		}
		else
		{
			apply_composition_correction(X, dX);
		}
	}
	timer->node["newton update"].node["composition correction"].stop();

	if (params->newton_type == sim_params::NEWTON_GLOBAL_CHOP)
	{
		if (params->log_transform == 1)
		{
			apply_global_chop_correction_new(X, dX);
		}
		else
		{
			apply_global_chop_correction(X, dX);
		}
	}
	// apply local chop only if number of components is 2 and more
	/*else if (params->newton_type == sim_params::NEWTON_LOCAL_CHOP && nc > 1)
	{
		if (params->log_transform == 1)
		{
			apply_local_chop_correction_new(X, dX);
		}
		else
		{
			apply_local_chop_correction(X, dX);
		}
	}*/

	// apply only if interpolation is used for derivatives
	// make decision based on only the first region
	if (op_axis_min[0].size() > 0)
		apply_obl_axis_local_correction(X, dX);

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		for (uint8_t c = 0; c < NC_ - 1; c++)
			X[N_VARS * i + Z_VAR + c] -= newton_update_coefficient * dX[N_VARS * i + Z_VAR + c];
		for (uint8_t d = 0; d < ND; d++)
			X[N_VARS * i + U_VAR + d] -= newton_update_coefficient * dX[N_VARS * i + U_VAR + d];
		X[N_VARS * i + P_VAR] -= newton_update_coefficient * dX[N_VARS * i + P_VAR];
	}
	if (THERMAL)
	{
		for (index_t i = 0; i < mesh->n_blocks; i++)
		{
			X[N_VARS * i + T_VAR] -= newton_update_coefficient * dX[N_VARS * i + T_VAR];
		}
	}

	return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::solve_linear_equation()
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
		static_cast<csr_matrix<4>*>(Jacobian)->write_matrix_to_file_mm(("jac_nc_dar_" + std::to_string(output_counter++) + ".csr").c_str());
		//Jacobian->write_matrix_to_file(("jac_nc_dar_" + std::to_string(output_counter++) + ".csr").c_str());
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

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::post_newtonloop(value_t deltat, value_t time)
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

	dev_u = dev_p = dev_e = well_residual_last_dt = std::numeric_limits<value_t>::infinity();
	std::fill(dev_z, dev_z + NC_, std::numeric_limits<value_t>::infinity());

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

		Xn = X;
		Xn_ref = Xref;
		std::copy(fluxes.begin(), fluxes.end(), fluxes_n.begin());
		std::copy(fluxes_biot.begin(), fluxes_biot.end(), fluxes_biot_n.begin());
		op_vals_arr_n = op_vals_arr;
		t += dt;
	}
	return converged;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
std::vector<value_t> engine_super_elastic_cpu<NC, NP, THERMAL>::calc_newton_dev()
{
	/*switch (params->nonlinear_norm_type)
	{
	case sim_params::L1:
	{
		return calc_newton_residual_L1();
	}
	case sim_params::L2:
	{*/
	return calc_newton_dev_L2();
	/*}
	case sim_params::LINF:
	{
		return calc_newton_residual_Linf();
	}
	default:
	{*/
	//}
	//}
}
template <uint8_t NC, uint8_t NP, bool THERMAL>
std::vector<value_t> engine_super_elastic_cpu<NC, NP, THERMAL>::calc_newton_dev_L2()
{
	std::vector<value_t> dev_by_balance(THERMAL + NC + 2, 0.0); // mass + momentum + energy
	std::vector<value_t> dev(n_vars, 0);
	std::vector<value_t> norm(n_vars, 0);
	value_t gap_dev = 0.0, norm_gap = 0.0;

	// residual in matrix cells
	for (int i = 0; i < mesh->n_matrix; i++)
	{
		// mass + energy
		for (int c = 0; c < NE; c++)
		{
			dev[P_VAR + c] += RHS[i * n_vars + P_VAR + c] * RHS[i * n_vars + P_VAR + c];
			norm[P_VAR + c] += mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP + c] *
								mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP + c];
		}
		// momentum
		for (int c = 0; c < ND; c++)
		{
			dev[U_VAR + c] += RHS[i * n_vars + U_VAR + c] * RHS[i * n_vars + U_VAR + c];
			norm[U_VAR + c] += mesh->volume[i] * mesh->volume[i];
		}
	}
	// residual in fault cells
	for (int i = mesh->n_matrix; i < mesh->n_res_blocks; i++)
	{
		// mass + energy
		for (int c = 0; c < NE; c++)
		{
			dev[P_VAR + c] += RHS[i * n_vars + P_VAR + c] * RHS[i * n_vars + P_VAR + c];
			norm[P_VAR + c] += mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP + c] *
				mesh->volume[i] * mesh->poro[i] * op_vals_arr[i * N_OPS + ACC_OP + c];
		}
		// momentum
		for (int c = 0; c < ND; c++)
		{
			dev[U_VAR + c] += RHS[i * n_vars + U_VAR + c] * RHS[i * n_vars + U_VAR + c];
			norm[U_VAR + c] += mesh->volume[i] * mesh->volume[i];
			gap_dev += RHS[i * n_vars + U_VAR + c] * RHS[i * n_vars + U_VAR + c];
		}
		norm_gap += 1.0;// mesh->volume[i] * mesh->volume[i];
	}
	// mass normalization
	for (int c = 0; c < NC; c++)
	{
		dev_by_balance[0] = std::max(dev_by_balance[0], sqrt(dev[P_VAR + c] / norm[P_VAR + c]));
	}
	dev_p_prev = dev_p;		dev_p = dev_by_balance[0];
	// energy normalization
	if (THERMAL)
	{
		dev_by_balance[2] = std::max(dev_by_balance[2], sqrt(dev[T_VAR] / norm[T_VAR]));
		dev_e_prev = dev_e;		dev_e = dev_by_balance[2];
	}
	// momentum normalization
	for (int c = 0; c < ND; c++)
	{
		dev_by_balance[1] = std::max(dev_by_balance[1], sqrt(dev[U_VAR + c] / norm[U_VAR + c]));
	}
	dev_u_prev = dev_u;		dev_u = dev_by_balance[1];
	// gap norm
	if (mesh->n_res_blocks > mesh->n_matrix)
		dev_by_balance[3] = sqrt(gap_dev / norm_gap);

	return dev_by_balance;
}
template <uint8_t NC, uint8_t NP, bool THERMAL>
double engine_super_elastic_cpu<NC, NP, THERMAL>::calc_well_residual_L2()
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
			for (int v = 0; v < NE; v++)
			{
				index_t i_w, i_r;
				value_t wi, wid;
				std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

				res[v] += RHS[(w->well_body_idx + i_w) * n_vars + v] * RHS[(w->well_body_idx + i_w) * n_vars + v];
				norm[v] += PV[w->well_body_idx + i_w] * av_op[v] * PV[w->well_body_idx + i_w] * av_op[v];
			}
		}
		// and then add RHS for well control equations
		for (int v = 0; v < NE; v++)
		{
			// well constraints should not be normalized, so pre-multiply by norm
			res[v] += RHS[w->well_head_idx * n_vars + v] * RHS[w->well_head_idx * n_vars + v] * PV[w->well_body_idx] * av_op[v] * PV[w->well_body_idx] * av_op[v];
		}
	}

	for (int v = 0; v < NE; v++)
	{
		residual = std::max(residual, sqrt(res[v] / norm[v]));
	}

	well_residual_prev_dt = well_residual_last_dt;
	well_residual_last_dt = residual;

	return residual;
}



template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_elastic_cpu<NC, NP, THERMAL>::apply_composition_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	value_t sum_z, new_z;
	index_t nb = mesh->n_blocks;
	bool z_corrected;
	index_t n_corrected = 0;

	for (index_t i = 0; i < nb; i++)
	{
		sum_z = 0;
		z_corrected = false;

		// check all but one composition in grid block
		for (char c = 0; c < nc - 1; c++)
		{
			new_z = X[i * N_VARS + Z_VAR + c] - dX[i * N_VARS + Z_VAR + c];
			if (new_z < min_zc)
			{
				new_z = min_zc;
				z_corrected = true;
			}
			else if (new_z > 1 - min_zc)
			{
				new_z = 1 - min_zc;
				z_corrected = true;
			}
			sum_z += new_z;
		}
		// check the last composition
		new_z = 1 - sum_z;
		if (new_z < min_zc)
		{
			new_z = min_zc;
			z_corrected = true;
		}
		sum_z += new_z;

		if (z_corrected)
		{
			// normalize compositions and set appropriate update
			for (char c = 0; c < nc - 1; c++)
			{
				new_z = X[i * N_VARS + Z_VAR + c] - dX[i * N_VARS + Z_VAR + c];

				new_z = std::max(min_zc, new_z);
				new_z = std::min(1 - min_zc, new_z);

				new_z = new_z / sum_z;
				dX[i * N_VARS + Z_VAR + c] = X[i * N_VARS + Z_VAR + c] - new_z;
			}
			n_corrected++;
		}
	}
	if (n_corrected)
		std::cout << "Composition correction applied in " << n_corrected << " block(s)" << std::endl;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_elastic_cpu<NC, NP, THERMAL>::apply_composition_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	/*double sum_z, new_z, temp_sum, min_count;
	std::vector<value_t> check_vec;
	index_t nb = mesh->n_blocks;
	bool z_corrected;
	index_t n_corrected = 0;

	// Check if solving for the log-transform or regular composition:
	if (params->log_transform == 0)
	{
		// No log-transform is applied to nonlinear unknowns (compositions only), proceed normally:
		for (index_t i = 0; i < nb; i++)
		{
			sum_z = 0;
			temp_sum = 0;		  // sum of any composition not set to z_min
			min_count = 0;		  // number of times a composition is set to z_min
			check_vec.resize(nc); // vector that holds 0 for z_c > z_min && 1 for z_c = z_min
			z_corrected = false;

			// check all but one composition in grid block
			for (index_t c = 0; c < nc - 1; c++)
			{
				new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];

				if (new_z < min_zc)
				{
					//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
					new_z = min_zc; //TODO: check if this update is consistent!
					z_corrected = true;
					check_vec[c] = 1;
					min_count += 1;
				}
				else if (new_z > max_zc)
				{
					new_z = max_zc;
					z_corrected = true;
					temp_sum += new_z;
				}
				else
				{
					temp_sum += new_z;
				}
				sum_z += new_z;
			}

			// check the last composition
			new_z = 1 - sum_z;
			if (new_z < min_zc)
			{
				//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
				new_z = min_zc;
				z_corrected = true;
				check_vec[nc - 1] = 1;
				min_count += 1;
			}
			else
			{
				temp_sum += new_z;
			}
			sum_z += new_z;

			if (z_corrected)
			{
				// normalize compositions and set appropriate update
				for (index_t c = 0; c < nc - 1; c++)
				{
					new_z = X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c];

					//new_z = std::max(min_zc * (1 + min_zc), new_z);  //TODO: check if this update is consistent!
					new_z = std::max(min_zc, new_z);
					new_z = std::min(max_zc, new_z);

					if (check_vec[c] != 1)
					{
						//new_z = new_z / temp_sum * (1 - min_count * min_zc * (1 + min_zc));
						new_z = new_z / temp_sum * (1 - min_count * min_zc);
					}

					dX[i * n_vars + z_var + c] = X[i * n_vars + z_var + c] - new_z;
				}
				n_corrected++;
			}
			check_vec.clear();
		}
	}
	else if (params->log_transform == 1)
	{
		// Log-transform is applied to nonlinear unknowns (compositions only), transform back composition exp(log(zc)) to apply correction:
		for (index_t i = 0; i < nb; i++)
		{
			sum_z = 0;
			temp_sum = 0;		  // sum of any composition not set to z_min
			min_count = 0;		  // number of times a composition is set to z_min
			check_vec.resize(nc); // vector that holds 0 for z_c > z_min && 1 for z_c = z_min
			z_corrected = false;

			// check all but one composition in grid block
			for (char c = 0; c < nc - 1; c++)
			{
				new_z = exp(X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c]); //log based composition

				if (new_z < min_zc)
				{
					//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
					new_z = min_zc;
					z_corrected = true;
					check_vec[c] = 1;
					min_count += 1;
				}
				else if (new_z > max_zc)
				{
					new_z = max_zc;
					z_corrected = true;
					temp_sum += new_z;
				}
				else
				{
					temp_sum += new_z;
				}
				sum_z += new_z;
			}

			// check the last composition
			new_z = 1 - sum_z;
			if (new_z < min_zc)
			{
				//new_z = min_zc * (1 + min_zc);  //TODO: check if this update is consistent!
				new_z = min_zc;
				z_corrected = true;
				check_vec[nc - 1] = 1;
				min_count += 1;
			}
			else
			{
				temp_sum += new_z;
			}
			sum_z += new_z;

			if (z_corrected)
			{
				// normalize compositions and set appropriate update
				for (char c = 0; c < nc - 1; c++)
				{
					new_z = exp(X[i * n_vars + z_var + c] - dX[i * n_vars + z_var + c]); //log based composition

					//new_z = std::max(min_zc * (1 + min_zc), new_z);  //TODO: check if this update is consistent!
					new_z = std::max(min_zc, new_z);
					new_z = std::min(max_zc, new_z);

					if (check_vec[c] != 1)
					{
						//new_z = new_z / temp_sum * (1 - min_count * min_zc * (1 + min_zc));
						new_z = new_z / temp_sum * (1 - min_count * min_zc);
					}

					dX[i * n_vars + z_var + c] = log(exp(X[i * n_vars + z_var + c]) / new_z); //log based composition
				}
				n_corrected++;
			}
			check_vec.clear();
		}
	}

	if (n_corrected)
		std::cout << "Composition correction applied in " << n_corrected << " block(s)" << std::endl;
		*/
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_elastic_cpu<NC, NP, THERMAL>::apply_global_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	value_t max_ratio = 0;
	index_t ind, n_blocks = mesh->n_blocks;

	for (index_t i = 0; i < n_blocks; i++)
	{
		for (uint8_t c = 1; c < NC; c++)
		{
			ind = i * N_VARS + P_VAR + c;
			if (fabs(X[ind]) > 1e-4)
			{
				double ratio = fabs(dX[ind]) / fabs(X[ind]);
				max_ratio = (max_ratio < ratio) ? ratio : max_ratio;
			}
		}
	}

	if (max_ratio > params->newton_params[0])
	{
		std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
		for (size_t i = 0; i < n_blocks; i++)
		{
			for (uint8_t c = 1; c < NC; c++)
			{
				dX[i * N_VARS + P_VAR + c] *= params->newton_params[0] / max_ratio;
			}
		}
	}
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_elastic_cpu<NC, NP, THERMAL>::apply_global_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	value_t max_ratio = 0, temp_zc = 0, temp_dz = 0, ratio;
	index_t ind, n_blocks = mesh->n_blocks;

	if (params->log_transform == 0)
	{
		for (index_t i = 0; i < n_blocks; i++)
		{
			for (uint8_t c = 1; c < NC; c++)
			{
				ind = i * N_VARS + P_VAR + c;
				if (fabs(X[ind]) > 1e-4)
				{
					ratio = fabs(dX[ind]) / fabs(X[ind]);
					max_ratio = (max_ratio < ratio) ? ratio : max_ratio;
				}
			}
		}

		if (max_ratio > params->newton_params[0])
		{
			std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
			for (index_t i = 0; i < n_blocks; i++)
			{
				for (uint8_t c = 1; c < NC; c++)
				{
					dX[i * N_VARS + P_VAR + c] *= params->newton_params[0] / max_ratio;
				}
			}
		}
	}
	/*else if (params->log_transform == 1)
	{
		for (index_t i = 0; i < n_blocks; i++)
		{
			for (uint8_t c = 1; c < NC; c++)
			{
				ind = i * N_VARS + P_VAR + c;
				if (fabs(X[ind]) > 1e-4)
				{
					if (i % nc == 0)
					{
						ratio = fabs(dX[i]) / fabs(X[i]);
					}
					else
					{
						temp_zc = exp(X[i] - dX[i]);
						temp_dz = exp(X[i]) - temp_zc;
						ratio = fabs(temp_dz) / fabs(exp(X[i]));
					}
					max_ratio = (max_ratio < ratio) ? ratio : max_ratio;
				}
			}
		}

		if (max_ratio > params->newton_params[0])
		{
			std::cout << "Apply global chop with max changes = " << max_ratio << "\n";
			for (size_t i = 0; i < n_vars_total; i++)
			{
				dX[i] *= params->newton_params[0] / max_ratio; //log based composition
			}
		}
	}*/
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_elastic_cpu<NC, NP, THERMAL>::apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX)
{
	value_t max_ratio = 0;
	index_t n_obl_fixes = 0;
	value_t eps = 1e-15;

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		for (index_t v = 0; v < N_STATE; v++)
		{
			// make sure state values are strictly inside (min;max) interval by using eps
			// otherwise index issues with interpolation can be caused
			value_t axis_min = op_axis_min[mesh->op_num[i]][v] + eps;
			value_t axis_max = op_axis_max[mesh->op_num[i]][v] - eps;
			value_t new_x = X[i * N_VARS + P_VAR + v] - dX[i * N_VARS + P_VAR + v];
			if (new_x > axis_max)
			{
				dX[i * N_VARS + P_VAR + v] = X[i * N_VARS + P_VAR + v] - axis_max;
				// output only for the first time
				if (n_obl_fixes == 0)
				{
					std::cout << "OBL axis correction: block " << i << " variable " << v << " shoots over axis limit of " << axis_max << " to " << new_x << std::endl;
				}
				n_obl_fixes++;
			}
			else if (new_x < axis_min)
			{
				dX[i * N_VARS + P_VAR + v] = X[i * N_VARS + P_VAR + v] - axis_min;
				// output only for the first time
				if (n_obl_fixes == 0)
				{
					std::cout << "OBL axis correction: block " << i << " variable " << v << " shoots under axis limit of " << axis_min << " to " << new_x << std::endl;
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

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
    return 0;
};

//template<uint8_t NC, uint8_t NP, , bool THERMAL>
//double
//engine_super_elastic_cpu<NC, NP, THERMAL>::calc_newton_residual()
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

template class engine_super_elastic_cpu<2, 2, 0>;
template class engine_super_elastic_cpu<2, 2, 1>;
//template struct recursive_instantiator_nc_np<engine_super_elastic_cpu, 2, MAX_NC, 1>;
//template struct recursive_instantiator_nc_np<engine_super_elastic_cpu, 2, MAX_NC, 2>;
//template struct recursive_instantiator_nc_np<engine_super_elastic_cpu, 2, MAX_NC, 3>;
