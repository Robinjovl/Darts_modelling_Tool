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

using std::fill;
using std::fill_n;

template <uint8_t NC, uint8_t NP, bool THERMAL>
const uint8_t engine_super_elastic_cpu<NC, NP, THERMAL>::T2U[5] = {U_VAR, U_VAR + 1, U_VAR + 2, P_VAR, T_VAR};

template <uint8_t NC, uint8_t NP, bool THERMAL>
const uint8_t engine_super_elastic_cpu<NC, NP, THERMAL>::BC2U[5] = { U_BC_VAR, U_BC_VAR + 1, U_BC_VAR + 2, P_BC_VAR, T_BC_VAR };

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                            std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                            sim_params *params_, timer_node *timer_)
{
  newton_update_coefficient = 1.0;
  dev_u = dev_p = dev_e = well_residual_last_dt = std::numeric_limits<value_t>::infinity();
  fill(dev_z, dev_z + NC_, std::numeric_limits<value_t>::infinity());
  FIND_EQUILIBRIUM = false;
  contact_solver = pm::RETURN_MAPPING;
  geomechanics_mode.resize(mesh_->n_blocks, 0);
  gravity = {0.0, 0.0, 0.0};
  discr = nullptr;

  init_base(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);
  this->expose_jacobian();

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
#ifdef WITH_HYPRE
		case sim_params::CPU_GMRES_FS_CPR:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linsolv_iface *fs_cpr = new linsolv_bos_fs_cpr<N_VARS>(P_VAR, Z_VAR, U_VAR);
			if constexpr (NE == 1)
			  static_cast<linsolv_bos_fs_cpr<N_VARS> *>(fs_cpr)->set_prec(new linsolv_bos_amg<1>, new linsolv_hypre_amg<1>(params->finalize_mpi));
			else
			{
			  linsolv_iface* cpr = new linsolv_bos_cpr<NE>;
			  cpr->set_prec(new linsolv_bos_amg<1>);
			  static_cast<linsolv_bos_fs_cpr<N_VARS>*>(fs_cpr)->set_prec(cpr, new linsolv_hypre_amg<1>(params->finalize_mpi));
			}
			static_cast<linsolv_bos_fs_cpr<N_VARS>*>(fs_cpr)->set_block_sizes(mesh->n_matrix + mesh->n_fracs, 0, mesh->n_blocks - mesh->n_res_blocks);
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
	const uint8_t n_state = get_n_state();
	z_var = get_z_var();

	X_init.resize(n_vars * mesh->n_blocks);
	PV.resize(mesh->n_blocks);
	RV.resize(mesh->n_blocks);
	old_z.resize(nc);
	new_z.resize(nc);
	FIPS.resize(nc);
	old_z_fl.resize(nc - n_solid);
	new_z_fl.resize(nc - n_solid);

	darcy_fluxes.resize(mesh->n_conns);
	structural_movement_fluxes.resize(mesh->n_conns);
	fick_fluxes.resize(mesh->n_conns);
	hooke_forces.resize(ND * mesh->n_conns);
	hooke_forces_n.resize(ND * mesh->n_conns);
	biot_forces.resize(ND * mesh->n_conns);
	biot_forces_n.resize(ND * mesh->n_conns);
	if constexpr (THERMAL)
	{
	  fourier_fluxes.resize(mesh->n_conns);
	  thermal_forces.resize(ND * mesh->n_conns);
	  thermal_forces_n.resize(ND * mesh->n_conns);
	}
	eps_vol.resize(mesh->n_matrix);

	const uint8_t n_sym = ND * (ND + 1) / 2;
	total_stresses.resize(n_sym * mesh->n_matrix);
	effective_stresses.resize(n_sym * mesh->n_matrix);
	darcy_velocities.resize(ND * mesh->n_matrix);

	Xn_ref = Xref = Xn = X = X_init;
	for (index_t i = 0; i < mesh->n_res_blocks; i++)
	{
	  // reference
	  Xref[n_vars * i + P_VAR] = Xn_ref[n_vars * i + P_VAR] = mesh->ref_pressure[i];
	  // initial
	
	  for (uint8_t ii = 0; ii < NE; ii++)
	  {
		  X_init[n_vars * i + P_VAR + ii] = mesh->initial_state[i * NE + ii];
	  }
	  for (uint8_t d = 0; d < ND; d++)
	  {
		  X_init[n_vars * i + U_VAR + d] = mesh->displacement[ND * i + d];
	  }
	}
	X_init.resize(n_vars * mesh->n_blocks);

	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		PV[i] = mesh->volume[i] * mesh->poro[i];
	  	RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
	}

	if (THERMAL)
	{
	  for (index_t i = 0; i < mesh_->n_blocks; i++)
	  {
		// reference
		Xref[n_vars * i + T_VAR] = Xn_ref[n_vars * i + T_VAR] = mesh->ref_temperature[i];
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

	if (NC_ > 1)
	{
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
  // sizes
  index_t n_blocks = mesh->n_blocks;
  index_t n_matrix = mesh->n_matrix;
  index_t n_res_blocks = mesh->n_res_blocks;
  index_t n_bounds = mesh->n_bounds;
  index_t n_conns = mesh->n_conns;
  // connections
  const index_t *block_m = mesh->block_m.data();
  const index_t *block_p = mesh->block_p.data();
  const index_t *stencil = mesh->stencil.data();
  const index_t *offset = mesh->offset.data();
  // approximations
  const value_t *darcy_tran = mesh->darcy_tran.data();
  const value_t *darcy_rhs = mesh->darcy_rhs.data();
  const value_t *hooke_tran = mesh->hooke_tran.data();
  const value_t *hooke_rhs = mesh->hooke_rhs.data();
  const value_t *biot_tran = mesh->biot_tran.data();
  const value_t *biot_rhs = mesh->biot_rhs.data();
  const value_t *biot_vol_strain_tran = mesh->vol_strain_tran.data();
  const value_t *biot_vol_strain_rhs = mesh->vol_strain_rhs.data();
  const value_t *thermal_traction_tran = mesh->thermal_traction_tran.data();
  const value_t *fourier_tran = mesh->fourier_tran.data();
  // approximation blocks
  const uint8_t N_DARCY = 1;
  const uint8_t N_HOOKE = ND * NT;
  const uint8_t N_BIOT = ND;
  const uint8_t N_BIOT_STRAIN = NT;
  // boundary rhs values
  value_t *bc = mesh->bc.data();
  value_t *bc_prev = mesh->bc_n.data();
  value_t *bc_ref = mesh->bc_ref.data();
  value_t* pz_bounds = mesh->pz_bounds.data(); // hyperbolic influx variables over boundaries
  // free term
  const value_t *f = mesh->f.data();
  // other properties
  const value_t *V = mesh->volume.data();
  const value_t *cs = mesh->rock_compressibility.data();
  const value_t *poro = mesh->poro.data();
  const value_t *eps_vol_ref = mesh->ref_eps_vol.data();
  const value_t *hcap = mesh->heat_capacity.data();
  const std::vector<value_t>& th_poro = mesh->th_poro;
  // Jacobian as a BCSR matrix
  value_t *Jac = jacobian->get_values();
  index_t *diag_ind = jacobian->get_diag_ind();
  index_t *rows = jacobian->get_rows_ptr();
  index_t *cols = jacobian->get_cols_ind();
  index_t *row_thread_starts = jacobian->get_row_thread_starts();

  CFL_max = 0;

#ifdef _OPENMP
  //#pragma omp parallel reduction (max: CFL_max)
  if (!row_thread_starts)
  {
	  std::cout<<"row_thread_starts are not initialized! Check that linear solvers were compiled with OpenMP\n";
	  exit(1);
  }
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

  fill_n(Jac, N_VARS * N_VARS * mesh->n_links, 0.0);
  fill(RHS.begin(), RHS.end(), 0.0);
  fill(darcy_fluxes.begin(), darcy_fluxes.end(), 0.0);
  fill(structural_movement_fluxes.begin(), structural_movement_fluxes.end(), 0.0);
  fill(fick_fluxes.begin(), fick_fluxes.end(), 0.0);
  fill(hooke_forces.begin(), hooke_forces.end(), 0.0);
  fill(biot_forces.begin(), biot_forces.end(), 0.0);
  if constexpr (THERMAL)
  {
	fill(fourier_fluxes.begin(), fourier_fluxes.end(), 0.0);
	fill(thermal_forces.begin(), thermal_forces.end(), 0.0);
  }

  index_t j, upwd_jac_idx[NP], nebr_jac_idx, upwd_idx[NP], diag_idx, conn_id = 0, st_id = 0, conn_st_id = 0, 
	  csr_idx_start, csr_idx_end;
  index_t l_ind, r_ind, l_ind1, r_ind1, l_ind2, r_ind2, r_ind3, r_ind4, r_ind5;
  value_t *cur_bc, *cur_bc_prev, *ref_bc, biot_mult, biot_cur, comp_mult, phi, phi_n, *buf, *buf_prev, *n;
  uint8_t d, v, c, p, density_cond;
  value_t gamma_p_diff, p_diff, phase_p_diff[NP], t_diff, gamma_t_diff, phi_i, phi_j, phi_avg, phi_0_avg;
  value_t CFL_in[NC], CFL_out[NC], darcy_component_fluxes[NE];
  value_t CFL_max_local = 0;
  value_t avg_density, avg_weigthed_density, avg_weigthed_density_n, eff_density;
  uint8_t* var_map;
  value_t rho_s;

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
	  rho_s = op_vals_arr[i * N_OPS + ROCK_DENS]; // rock density
	  eff_density = 0.0;
	  for (p = 0; p < NP; p++)
	  {
		  eff_density += op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
	  }
	  eff_density -= rho_s;

	  // loop over cell connections
	  for (; conn_id < n_conns && block_m[conn_id] == i; conn_id++)
	  {
		  j = block_p[conn_id];
		  if (j >= n_res_blocks && j < n_blocks)
			  connected_with_well = 1;

		  fill_n(darcy_component_fluxes, NE, 0.0);

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
			  if (st_id < csr_idx_end && stencil[conn_st_id] != cols[st_id]) continue;

			  // upwind index in jacobian
			  if (st_id < csr_idx_end && cols[st_id] == j) nebr_jac_idx = st_id;
			  
			  if (stencil[conn_st_id] < n_blocks)	// matrix, fault or well cells
			  {
				  r_ind = N_VARS * stencil[conn_st_id];
				  buf = &X[r_ind];
				  buf_prev = &Xn[r_ind];
				  var_map = const_cast<uint8_t*>(T2U);
			  }
			  else									// boundary condition
			  {
				  r_ind = N_BC_VARS * (stencil[conn_st_id] - n_blocks);
				  buf = &bc[r_ind];
				  buf_prev = &bc_prev[r_ind];
				  var_map = const_cast<uint8_t*>(BC2U);
			  }

			  // biot * vol_strains
			  r_ind = conn_st_id * N_BIOT_STRAIN;
			  for (d = 0; d < NT; d++)
			  {
				// flux of displacements (u * n)
				biot_mult += biot_vol_strain_tran[r_ind + d] * buf[var_map[d]];
				// time derivative of the last flux = flux of matrix mass due to structure movement
				structural_movement_fluxes[conn_id] += biot_vol_strain_tran[r_ind + d] * (buf[var_map[d]] - buf_prev[var_map[d]]) / dt;
			  }
			  // darcy
			  p_diff += darcy_tran[conn_st_id] * buf[P_VAR];

			  // heat conduction
			  if constexpr (THERMAL)
				t_diff += fourier_tran[conn_st_id] * buf[var_map[NT-1]];

			  conn_st_id++;
		  }
		  // Darcy flux
		  darcy_fluxes[conn_id] = p_diff;
		  // rock heat conduction
		  if constexpr (THERMAL)
			fourier_fluxes[conn_id] = t_diff;

		  // [2] phase fluxes & upwind direction
		  for (p = 0; p < NP; p++)
		  {
			  // etimate average densities between cells for current time step
			  if (op_vals_arr[i * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE && op_vals_arr[j * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
			  {
				avg_density = avg_weigthed_density = 0.0;
			  }
			  else if (op_vals_arr[i * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
			  {
				avg_density = op_vals_arr[j * N_OPS + GRAV_OP + p];
				avg_weigthed_density = op_vals_arr[j * N_OPS + SAT_OP + p] * op_vals_arr[j * N_OPS + GRAV_OP + p];
			  }
			  else if (op_vals_arr[j * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
			  {
				avg_density = op_vals_arr[i * N_OPS + GRAV_OP + p];
				avg_weigthed_density = op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
			  }
			  else
			  {
				avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] + op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;
				avg_weigthed_density = (op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p] +
									    op_vals_arr[j * N_OPS + SAT_OP + p] * op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;
			  }
			  // etimate average density between cells for previous time step
			  if (op_vals_arr_n[i * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE && op_vals_arr_n[j * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
				avg_weigthed_density_n = 0.0;
			  else if (op_vals_arr_n[i * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
				avg_weigthed_density_n = op_vals_arr_n[j * N_OPS + SAT_OP + p] * op_vals_arr_n[j * N_OPS + GRAV_OP + p];
			  else if (op_vals_arr_n[j * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
				avg_weigthed_density_n = op_vals_arr_n[i * N_OPS + SAT_OP + p] * op_vals_arr_n[i * N_OPS + GRAV_OP + p];
			  else
				avg_weigthed_density_n = (op_vals_arr_n[i * N_OPS + SAT_OP + p] * op_vals_arr_n[i * N_OPS + GRAV_OP + p] +
										  op_vals_arr_n[j * N_OPS + SAT_OP + p] * op_vals_arr_n[j * N_OPS + GRAV_OP + p]) / 2;

			  // sum up gravity and cappillary terms
			  phase_p_diff[p] = p_diff + avg_density * darcy_rhs[conn_id] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

			  // sum up gravity for Biot volumetric strain
			  biot_mult += avg_weigthed_density * biot_vol_strain_rhs[conn_id];
			  structural_movement_fluxes[conn_id] += biot_vol_strain_rhs[conn_id] * (avg_weigthed_density - avg_weigthed_density_n) / dt;

			  // sum up gravitational & capillary terms for Darcy flux
			  darcy_fluxes[conn_id] += avg_weigthed_density * darcy_rhs[conn_id] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

			  // identify upwind direction
			  if (phase_p_diff[p] >= 0)
			  {
				  upwd_idx[p] = i;
				  upwd_jac_idx[p] = diag_ind[i];
				  for (c = 0; c < NE; c++)
				  {
					  if (c < NC) CFL_out[c] += phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
					  darcy_component_fluxes[c] += phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
				  }
			  }
			  else
			  {
				  upwd_idx[p] = j;
				  upwd_jac_idx[p] = nebr_jac_idx;
				  for (c = 0; c < NE; c++)
				  {
					  if (c < NC && j < n_res_blocks) CFL_in[c] += -phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
					  darcy_component_fluxes[c] += phase_p_diff[p] * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c];
				  }
			  }
		  }

		  // [3] loop over stencil, contribution from UNKNOWNS to flux
		  conn_st_id = offset[conn_id];
		  for (st_id = csr_idx_start; st_id < csr_idx_end && conn_st_id < offset[conn_id + 1]; st_id++)
		  {
			  if (stencil[conn_st_id] == cols[st_id])
			  {
				  //// momentum fluxes
				  l_ind = ND * conn_id;
				  r_ind = stencil[conn_st_id] * N_VARS;
				  for (d = 0; d < ND; d++)
				  {
					l_ind1 = st_id * N_VARS_SQ + (U_VAR + d) * N_VARS;
					r_ind1 = conn_st_id * N_HOOKE + d * NT;
					for (v = 0; v < NT; v++)
					{
					  // Hooke's forces 
					  hooke_forces[l_ind + d] += hooke_tran[r_ind1 + v] * X[r_ind + T2U[v]];
					  Jac[l_ind1 + T2U[v]] += hooke_tran[r_ind1 + v];
					}

					// Biot's forces
					biot_forces[l_ind + d] += biot_tran[conn_st_id * N_BIOT + d] * X[r_ind + P_VAR];
					Jac[l_ind1 + P_VAR] += biot_tran[conn_st_id * N_BIOT + d];

					// Thermal forces 					
					if constexpr (THERMAL)
					{
					  thermal_forces[l_ind + d] += thermal_traction_tran[conn_st_id * N_BIOT + d] * X[r_ind + T_VAR];
					  Jac[l_ind1 + T_VAR] += thermal_traction_tran[conn_st_id * N_BIOT + d];
					}
				  }
				  //// mass fluxes
				  r_ind = stencil[conn_st_id] * N_VARS;
				  r_ind1 = conn_st_id * N_DARCY;
				  for (p = 0; p < NP; p++)
				  {
					  // NE equations
					  for (c = 0; c < NE; c++)
					  {
						  Jac[st_id * N_VARS_SQ + (P_VAR + c) * N_VARS + P_VAR] += dt * op_vals_arr[upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c] * darcy_tran[r_ind1];
					  }
				  }
				  // biot term in accumulation
				  r_ind1 = conn_st_id * N_BIOT_STRAIN;
				  for (c = 0; c < NE; c++)
				  {
					  l_ind = i * N_VARS + P_VAR + c;
					  l_ind1 = st_id * N_VARS_SQ + (c + P_VAR) * N_VARS;
					  for (v = 0; v < NT; v++)
					  {
						  RHS[l_ind] += biot_vol_strain_tran[r_ind1 + v] *
							  (op_vals_arr[i * N_OPS + ACC_OP + c] * X[r_ind + T2U[v]] - op_vals_arr_n[i * N_OPS + ACC_OP + c] * Xn[r_ind + T2U[v]]);
						  Jac[l_ind1 + T2U[v]] += op_vals_arr[i * N_OPS + ACC_OP + c] * biot_vol_strain_tran[r_ind1 + v];
					  }
				  }
				  // biot term in porosity in gravitational forces
				  /*for (d = 0; d < ND; d++)
				  {
					  l_ind = i * N_VARS + U_VAR + d;
					  l_ind1 = st_id * N_VARS_SQ + (U_VAR + d) * N_VARS;
					  for (v = 0; v < NT; v++)
					  {
						  RHS[l_ind] -= gravity[d] * eff_density * biot_vol_strain_tran[r_ind1 + v] * X[r_ind + T2U[v]];
						  Jac[l_ind1 + T2U[v]] -= gravity[d] * eff_density * biot_vol_strain_tran[r_ind1 + v];
					  }
				  }*/
				  //// heat fluxes
				  if constexpr (THERMAL)
				  {
					  // rock energy
					  l_ind = i * N_VARS + T_VAR;
					  l_ind1 = st_id * N_VARS_SQ + T_VAR * N_VARS;
					  for (v = 0; v < NT; v++)
					  {
						  RHS[l_ind] -= hcap[i] * biot_vol_strain_tran[r_ind1 + v] *
							  (op_vals_arr[i * N_OPS + TEMP_OP] * X[r_ind + T2U[v]] - op_vals_arr_n[i * N_OPS + TEMP_OP] * Xn[r_ind + T2U[v]]);
						  Jac[l_ind1 + T2U[v]] -= hcap[i] * biot_vol_strain_tran[r_ind1 + v] * op_vals_arr[i * N_OPS + TEMP_OP];
					  }

					  // heat conduction
					  Jac[l_ind1 + T_VAR] += dt * fourier_tran[conn_st_id];
				  }

				  conn_st_id++;
			  }
		  }
		  // [4] loop over stencil, contribution from BOUNDARY CONDITIONS to flux
		  for (; conn_st_id < offset[conn_id + 1]; conn_st_id++)
		  {
			  if (stencil[conn_st_id] >= n_blocks)
			  {
				  r_ind = N_BC_VARS * (stencil[conn_st_id] - n_blocks);
				  cur_bc = &bc[r_ind];
				  cur_bc_prev = &bc_prev[r_ind];
				  ref_bc = &bc_ref[r_ind];
				  // momentum balance
				  l_ind = ND * conn_id;
				  for (d = 0; d < ND; d++)
				  {
					  // Hooke's forces 
					  r_ind = conn_st_id * N_HOOKE + d * NT;
					  for (v = 0; v < NT; v++)
					  {
						  hooke_forces[l_ind + d] += hooke_tran[r_ind + v] * (cur_bc[BC2U[v]] - ref_bc[BC2U[v]]);
					  }
					  // Biot's forces
					  biot_forces[l_ind + d] += biot_tran[conn_st_id * N_BIOT + d] * (cur_bc[P_VAR] - ref_bc[P_VAR]);

					  // Thermal forces
					  if constexpr (THERMAL)
						thermal_forces[l_ind + d] += thermal_traction_tran[conn_st_id * N_BIOT + d] * (cur_bc[BC2U[NT - 1]] - ref_bc[BC2U[NT - 1]]);
				  }
				  // mass balance
				  // biot term in accumulation
				  r_ind = conn_st_id * N_BIOT_STRAIN;
				  for (c = 0; c < NE; c++)
				  {
					  l_ind = i * N_VARS + P_VAR + c;
					  for (v = 0; v < NT; v++)
					  {
						  RHS[l_ind] += biot_vol_strain_tran[r_ind + v] *
							  (op_vals_arr[i * N_OPS + ACC_OP + c] * cur_bc[BC2U[v]] - op_vals_arr_n[i * N_OPS + ACC_OP + c] * cur_bc_prev[BC2U[v]]);
					  }
				  }
				  // biot term in porosity in gravitational forces
				  /*for (d = 0; d < ND; d++)
				  {
					  l_ind = i * N_VARS + U_VAR + d;
					  for (v = 0; v < NT; v++)
					  {
						  RHS[l_ind] -= gravity[d] * eff_density * biot_vol_strain_tran[r_ind + v] * cur_bc[T2U[v]];
					  }
				  }*/
				  // rock energy
				  if constexpr (THERMAL)
				  {
					  l_ind = i * N_VARS + T_VAR;
					  for (v = 0; v < NT; v++)
					  {
						  RHS[l_ind] -= hcap[i] * biot_vol_strain_tran[r_ind + v] *
							  (op_vals_arr[i * N_OPS + TEMP_OP] * cur_bc[BC2U[v]] - op_vals_arr_n[i * N_OPS + TEMP_OP] * cur_bc_prev[BC2U[v]]);
					  }
				  }
			  }
		  }
		  // [5] loop over pressure, composition & temperature
		  for (p = 0; p < NP; p++)
		  {
			  density_cond = -1;
			  // calculate partial derivatives for gravity and capillary terms
			  value_t grav_pc_der_i[N_VARS - ND] = { 0.0 };
			  value_t grav_pc_der_j[N_VARS - ND] = { 0.0 };
			  r_ind = (i * N_OPS + GRAV_OP + p) * N_STATE;
			  r_ind1 = (j * N_OPS + GRAV_OP + p) * N_STATE;
			  r_ind2 = (i * N_OPS + PC_OP + p) * N_STATE;
			  r_ind3 = (j * N_OPS + PC_OP + p) * N_STATE;
			  // estimate average gravity between cells
			  if (op_vals_arr[i * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE && op_vals_arr[j * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
			  {
				 avg_weigthed_density = 0.0;
			  }
			  else if (op_vals_arr[i * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
			  {
				avg_weigthed_density = op_vals_arr[j * N_OPS + SAT_OP + p] * op_vals_arr[j * N_OPS + GRAV_OP + p];
				density_cond = 2;
			  }
			  else if (op_vals_arr[j * N_OPS + SAT_OP + p] < EQUALITY_TOLERANCE)
			  {
				avg_weigthed_density = op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
				density_cond = 1;
			  }
			  else
			  {
				avg_weigthed_density = (op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p] +
										op_vals_arr[j * N_OPS + SAT_OP + p] * op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;
				density_cond = 0;
			  }
			  // store derivatives of 'avg_density' coming with (gravitational) free term
			  if (density_cond == 0)
			  {
				for (v = 0; v < NE; v++)
				{
				  grav_pc_der_i[v] = -op_ders_arr[r_ind + v] * darcy_rhs[conn_id] / 2 - op_ders_arr[r_ind2 + v];
				  grav_pc_der_j[v] = -op_ders_arr[r_ind1 + v] * darcy_rhs[conn_id] / 2 + op_ders_arr[r_ind3 + v];
				}
			  }
			  else if (density_cond == 1)
			  {
				for (v = 0; v < NE; v++)
				  grav_pc_der_i[v] = -op_ders_arr[r_ind + v] * darcy_rhs[conn_id] - op_ders_arr[r_ind2 + v];
			  }
			  else if (density_cond == 2)
			  {
				for (v = 0; v < NE; v++)
				  grav_pc_der_j[v] = -op_ders_arr[r_ind1 + v] * darcy_rhs[conn_id] / 2 + op_ders_arr[r_ind3 + v];
			  }
			  // assemble
			  for (c = 0; c < NE; c++)
			  {
				  l_ind = diag_idx + (P_VAR + c) * N_VARS;
				  l_ind1 = nebr_jac_idx * N_VARS_SQ + (P_VAR + c) * N_VARS;
				  r_ind = upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c;
				  r_ind1 = (upwd_idx[p] * N_OPS + FLUX_OP + p * NE + c) * N_STATE;
				  l_ind2 = i * N_VARS + P_VAR + c;
				  r_ind2 = (i * N_OPS + GRAV_OP + p) * N_STATE;
				  r_ind3 = (i * N_OPS + SAT_OP + p) * N_STATE;
				  r_ind4 = (j * N_OPS + GRAV_OP + p) * N_STATE;
				  r_ind5 = (j * N_OPS + SAT_OP + p) * N_STATE;
				  
				  RHS[l_ind2] += avg_weigthed_density * biot_vol_strain_rhs[conn_id] * op_vals_arr[i * N_OPS + ACC_OP + c];
				  for (v = 0; v < NE; v++)
				  {
					  // 1. mobility derivative
					  if (upwd_jac_idx[p] < csr_idx_end) // mobility derivatives
					  {
						  Jac[upwd_jac_idx[p] * N_VARS_SQ + (P_VAR + c) * N_VARS + v] += dt * phase_p_diff[p] * op_ders_arr[r_ind1 + v];
					  }
					  // 2. derivatives of 'avg_density' coming with (gravitational) free term
					  Jac[l_ind + v] += dt * op_vals_arr[r_ind] * grav_pc_der_i[v];
					  // 3. derivatives of 'avg_weigthed_density' coming with (gravitational) free term
					  if (density_cond == 0)
						Jac[l_ind + v] += biot_vol_strain_rhs[conn_id] * op_vals_arr[i * N_OPS + ACC_OP + c] *
						  (op_vals_arr[i * N_OPS + SAT_OP + p] * op_ders_arr[r_ind2 + v] + op_ders_arr[r_ind3 + v] * op_vals_arr[i * N_OPS + GRAV_OP + p]) / 2;
					  else if (density_cond == 1)
						Jac[l_ind + v] += biot_vol_strain_rhs[conn_id] * op_vals_arr[i * N_OPS + ACC_OP + c] *
						  (op_vals_arr[i * N_OPS + SAT_OP + p] * op_ders_arr[r_ind2 + v] + op_ders_arr[r_ind3 + v] * op_vals_arr[i * N_OPS + GRAV_OP + p]);
					  if (nebr_jac_idx < csr_idx_end)
					  {
						// 2. .. with respect to neighbour j
						Jac[l_ind1 + v] += dt * op_vals_arr[r_ind] * grav_pc_der_j[v]; // 1.
						// 3. .. with respect to neighbour j
						if (density_cond == 0)
						  Jac[l_ind1 + v] += biot_vol_strain_rhs[conn_id] * op_vals_arr[i * N_OPS + ACC_OP + c] *
							(op_vals_arr[j * N_OPS + SAT_OP + p] * op_ders_arr[r_ind4 + v] + op_ders_arr[r_ind5 + v] * op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;
						else if (density_cond == 2)
						  Jac[l_ind1 + v] += biot_vol_strain_rhs[conn_id] * op_vals_arr[i * N_OPS + ACC_OP + c] *
							(op_vals_arr[j * N_OPS + SAT_OP + p] * op_ders_arr[r_ind4 + v] + op_ders_arr[r_ind5 + v] * op_vals_arr[j * N_OPS + GRAV_OP + p]);
					  } 
				  }
			  }
			  // 4. derivatives of density coming with (gravitational) free term to porosity in gravitational forces
			  // note that gravitational free term is from Darcy fluxes, gravitational forces are in momentum balance
			  // for clarity: 
			  // biot * vol_strain * \rho_{fluid} + (1 - biot * vol_strain) * \rho_{sk} = (\sum_{stencil, vars} (biot_vol_strain_tran * X) + rho_{fluid} * biot_vol_strain_rhs) * eff_density +
			  // + \rho_{sk}
			  /*for (d = 0; d < ND; d++)
			  {
				l_ind = i * N_VARS + U_VAR + d;
				RHS[l_ind] -= gravity[d] * eff_density * biot_vol_strain_rhs[conn_id] * op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
				l_ind1 = diag_idx + (U_VAR + d) * N_VARS;
				r_ind2 = (i * N_OPS + GRAV_OP + p) * N_STATE;
				r_ind3 = (i * N_OPS + SAT_OP + p) * N_STATE;
				for (v = 0; v < NE; v++)
				{
				  Jac[l_ind1 + v] -= gravity[d] * eff_density * biot_vol_strain_rhs[conn_id] *
							  (op_vals_arr[i * N_OPS + SAT_OP + p] * op_ders_arr[r_ind2 + v] + op_ders_arr[r_ind3 + v] * op_vals_arr[i * N_OPS + GRAV_OP + p]);
				}
			  }*/
		  }
		  // [?] extra loop for gravity in biot for flux
		  // [6] (saturation ??? ) thermal expansion & fluid gravity for momentum balance
		  // [7] add fluid heat conduction
		  /*if (THERMAL)
		  {
			  t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
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
		  }*/
		  // [8] fluxes to residual
		  // mass (Darcy)
		  l_ind = i * N_VARS + P_VAR;
		  for (c = 0; c < NE; c++)
		  {
			  RHS[l_ind + c] += dt * darcy_component_fluxes[c];
		  }
		  // momentum (forces)
		  l_ind = i * N_VARS + U_VAR;
		  r_ind = ND * conn_id;
		  for (d = 0; d < ND; d++)
		  {
			  RHS[l_ind + d] += hooke_forces[r_ind + d] + biot_forces[r_ind + d];
			  if constexpr (THERMAL)
				RHS[l_ind + d] += thermal_forces[r_ind + d];
		  }
		  // energy (heat conduction)
		  if constexpr (THERMAL)
		  {
			l_ind = i * N_VARS + T_VAR;
			RHS[l_ind] += dt * fourier_fluxes[conn_id];
		  }
	  }

	  // [9] accumulation for mass balance
	  // [9.1] porosity

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
			r_ind = i * N_VARS;
			comp_mult = cs[i];
			phi += comp_mult * (X[r_ind + P_VAR] - Xref[r_ind + P_VAR]) - eps_vol_ref[i];
			phi_n += comp_mult * (Xn[r_ind + P_VAR] - Xn_ref[r_ind + P_VAR]) - eps_vol_ref[i];
			if (THERMAL)
			{
				phi -= th_poro[i] * (X[r_ind + T_VAR] - Xref[r_ind + T_VAR]);
				phi_n -= th_poro[i] * (Xn[r_ind + T_VAR] - Xn_ref[r_ind + T_VAR]);
			}
		  }
		  else
		  {
			  comp_mult = 0.0;
		  }
	  }

	  if (FIND_EQUILIBRIUM || geomechanics_mode[i])
	  {
		  for (c = 0; c < NE; c++)
			  Jac[diag_idx + (P_VAR + c) * N_VARS + P_VAR + c] = V[i];
	  }
	  else
	  {
		  for (c = 0; c < NE; c++)
		  {
			  RHS[i * N_VARS + P_VAR + c] += V[i] * (phi * op_vals_arr[i * N_OPS + ACC_OP + c] - phi_n * op_vals_arr_n[i * N_OPS + ACC_OP + c]);
			  Jac[diag_idx + (P_VAR + c) * N_VARS + P_VAR] += V[i] * comp_mult * op_vals_arr[i * N_OPS + ACC_OP + c];
			  for (v = 0; v < N_STATE; v++)
			  {
				  Jac[diag_idx + (P_VAR + c) * N_VARS + P_VAR + v] += V[i] * phi * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_STATE + v];
			  }
			  //if (!geomechanics_mode[i] && THERMAL)
			  //	Jac[diag_idx + (P_VAR + c) * N_VARS + T_VAR] -= V[i] * th_poro[i] * op_vals_arr[i * N_OPS + ACC_OP + c];
		  }
	  }

      // [9.2] add rock energy
      // + rock energy (no rock compressibility included in these computations)
      if (THERMAL && !FIND_EQUILIBRIUM)
      {
        RHS[i * N_VARS + T_VAR] += V[i] * ((1.0 - phi) * op_vals_arr[i * N_OPS + TEMP_OP] - (1.0 - phi_n) * op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];

        for (v = 0; v < NE; v++)
        {
          Jac[diag_idx + T_VAR * N_VARS + v] +=  V[i] * (1.0 - phi) * op_ders_arr[(i * N_OPS + TEMP_OP) * N_STATE + v] * hcap[i];
        } // end of fill offdiagonal part + contribute to diagonal

		Jac[diag_idx + T_VAR * N_VARS + P_VAR] -= V[i] * comp_mult * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i];
		Jac[diag_idx + T_VAR * N_VARS + T_VAR] += V[i] * th_poro[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i];
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

	  if (i < n_res_blocks)
	  {
		  // [9.3] gravitational forces
		  for (d = 0; d < ND; d++)
		  {
			  for (p = 0; p < NP; p++)
			  {
				  RHS[i * N_VARS + U_VAR + d] -= poro[i] * V[i] * gravity[d] * op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
				  // Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR] -= comp_mult * V[i] * gravity[d] * op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];
				  // if constexpr (THERMAL)
					// Jac[diag_idx + (U_VAR + d) * N_VARS + T_VAR] += th_poro[i] * V[i] * gravity[d] * op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p];

				  for (v = 0; v < N_STATE; v++)
				  {
					  Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR + v] -= poro[i] * V[i] * gravity[d] *
						  (op_vals_arr[i * N_OPS + SAT_OP + p] * op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_STATE + v] + 
							  op_ders_arr[(i * N_OPS + SAT_OP + p) * N_STATE + v] * op_vals_arr[i * N_OPS + GRAV_OP + p]);
				  }
			  }
			  RHS[i * N_VARS + U_VAR + d] -= (1 - poro[i]) * V[i] * gravity[d] * rho_s;
			  
			  //Jac[diag_idx + (U_VAR + d) * N_VARS + P_VAR] += comp_mult * V[i] * gravity[d] * rho_s;
			  //if constexpr (THERMAL)
				// Jac[diag_idx + (U_VAR + d) * N_VARS + T_VAR] += th_poro[i] * V[i] * gravity[d] * rho_s;*/
		  }
		  // [9.4] user-defined part
		  for (c = 0; c < NE; c++)
		  {
			RHS[i * N_VARS + P_VAR + c] += V[i] * dt * f[i * N_VARS + P_VAR + c];
		  }
		  for (d = 0; d < ND; d++)
		  {
			RHS[i * N_VARS + U_VAR + d] += V[i] * f[i * N_VARS + U_VAR + d];
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
	for (uint8_t d = 0; d < ND; d++)
	{
		Jac[N_VARS_SQ * diag_ind[w->well_head_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
		Jac[N_VARS_SQ * diag_ind[w->well_body_idx] + (U_VAR + d) * N_VARS + U_VAR + d] = 1.0;
	}
  }

  return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::eval_stresses_and_velocities()
{
  assert(discr != nullptr);
  assert(discr->stress_approx.size() > 0);

  const index_t* adj_matrix_offset = discr->mesh->adj_matrix_offset.data();
  const index_t* adj_matrix = discr->mesh->adj_matrix.data();
  const mesh::Connection* conns = discr->mesh->conns.data();
  const mesh::Vector3* centroids = discr->mesh->centroids.data();
  const dis::Matrix33* biots = discr->biots.data();
  const value_t* stress_approx = discr->stress_approx.data();
  const value_t* velocity_approx = discr->velocity_approx.data();

  const index_t* block_m = mesh->block_m.data();
  const index_t* block_p = mesh->block_p.data();
  const index_t n_matrix = mesh->n_matrix;
  const index_t n_res_blocks = mesh->n_res_blocks;
  const index_t n_blocks = mesh->n_blocks;
  const index_t n_conns = mesh->n_conns;
  const index_t n_bounds = mesh->n_bounds;
  const index_t n_wells = n_blocks - n_res_blocks;
  constexpr uint8_t n_sym = (ND + 1) * ND / 2;

  linalg::Vector3 t_face, n;
  index_t counter, conn_id = 0, j, tmp, tmp1;
  value_t p_grad_vals[ND], p_face;
  value_t cur_total_tractions[ND * mesh::MAX_CONNS_PER_ELEM_GMSH];
  value_t cur_effective_tractions[ND * mesh::MAX_CONNS_PER_ELEM_GMSH];
  value_t cur_darcy_fluxes[mesh::MAX_CONNS_PER_ELEM_GMSH];
  value_t Ndelta[ND][n_sym] = {0.0};
  value_t w[n_sym];

  for (index_t i = 0; i < n_matrix; i++)
  {
	// evaluate pressure gradient
	const auto& p_grad = discr->p_grads[i];

	p_grad_vals[0] = p_grad_vals[1] = p_grad_vals[2] = 0.0;
	// right-hand side
	for (uint8_t d = 0; d < ND; d++)
	{
	  for (uint8_t p = 0; p < NP; p++)
	  {
		p_grad_vals[d] += (op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p]) * p_grad.rhs(d, 0);
	  }
	}
	// stencil assembly 
	for (index_t k = 0; k < p_grad.stencil.size(); k++)
	{
	  for (uint8_t d = 0; d < ND; d++)
	  {
		tmp = p_grad.stencil[k] < n_res_blocks ? p_grad.stencil[k] : p_grad.stencil[k] + n_wells;
		p_grad_vals[d] += p_grad.a(d, k) * Xop[tmp * N_STATE + P_VAR];
	  }
	}

	const auto& b = biots[i];
	w[0] = b(0, 0);	 w[1] = b(1, 1);  w[2] = b(2, 2);	w[3] = b(1, 2);	  w[4] = b(0, 2);	w[5] = b(0, 1);
	fill(std::begin(cur_total_tractions), std::end(cur_total_tractions), 0.0);
	fill(std::begin(cur_effective_tractions), std::end(cur_effective_tractions), 0.0);
	fill(std::begin(cur_darcy_fluxes), std::end(cur_darcy_fluxes), 0.0);
	counter = 0;
	for (index_t face_id = 0; conn_id < n_conns && block_m[conn_id] == i;)
	{
	  j = block_p[conn_id];

	  // skip well connection
	  if (j >= n_res_blocks && j < n_blocks) { conn_id++;  continue; }

	  const auto& conn = conns[adj_matrix[adj_matrix_offset[i] + face_id]];
	  /* assert((i == conn.elem_id1 && j == conn.elem_id2 + n_wells) ||
			  (i == conn.elem_id2 && j == conn.elem_id1 + n_wells) ); */

	  t_face = conn.c - centroids[i];
	  n = (dot(conn.n, t_face) > 0 ? conn.n : -conn.n);
	  Ndelta[0][0] = n.x;              Ndelta[1][1] = n.y;          Ndelta[2][2] = n.z;
	  Ndelta[1][ND + 2] = n.x;         Ndelta[2][ND + 1] = n.x;
	  Ndelta[0][ND + 2] = n.y;         Ndelta[2][ND] = n.y;
	  Ndelta[1][ND] = n.z;			   Ndelta[0][ND + 1] = n.z;
	  // evaluate facial pressure
	  p_face = X[i * N_VARS + P_VAR];
	  p_face += t_face.x * p_grad_vals[0] + t_face.y * p_grad_vals[1] + t_face.z * p_grad_vals[2];

	  // fill fluxes
	  for (uint8_t d = 0; d < ND; d++)
	  {
		tmp = ND * conn_id + d;
		tmp1 = ND * counter + d;
		// total traction
		cur_total_tractions[tmp1] = -hooke_forces[tmp] - biot_forces[tmp];
		if constexpr (THERMAL)
		  cur_total_tractions[tmp1] += -thermal_forces[tmp];
		// effective traction
		cur_effective_tractions[tmp1] = cur_total_tractions[tmp1];
		for (uint8_t c = 0; c < n_sym; c++)
		  cur_effective_tractions[tmp1] += conn.area * p_face * Ndelta[d][c] * w[c];
	  }

	  // Darcy flux
	  cur_darcy_fluxes[counter] = darcy_fluxes[conn_id];

	  counter++;
	  face_id++;
	  conn_id++;
	}

	// stresses
	for (index_t c = 0; c < n_sym; c++)
	{
	  tmp = i * n_sym + c;
	  total_stresses[tmp] = 0.0;
	  effective_stresses[tmp] = 0.0;

	  tmp1 = n_sym * ND * adj_matrix_offset[i] + c * ND * counter;
	  for (index_t k = 0; k < ND * counter; k++)
	  {
		total_stresses[tmp] += stress_approx[tmp1 + k] * cur_total_tractions[k];
		effective_stresses[tmp] += stress_approx[tmp1 + k] * cur_effective_tractions[k];
	  }
	}

	// darcy velocities
	for (index_t d = 0; d < ND; d++)
	{
	  tmp = i * ND + d;
	  darcy_velocities[tmp] = 0.0;

	  tmp1 = ND * adj_matrix_offset[i] + d * counter;
	  for (index_t k = 0; k < counter; k++)
	  {
		darcy_velocities[tmp] += velocity_approx[tmp1 + k] * cur_darcy_fluxes[k];
	  }
	}
  }

  return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::assemble_linear_system(value_t deltat)
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

	if (print_linear_system) //changed this to write jacobian to file!
	{
		const std::string matrix_filename = "jac_nc_dar_" + std::to_string(output_counter) + ".csr";
#ifdef OPENDARTS_LINEAR_SOLVERS
		Jacobian->export_matrix_to_file(matrix_filename, opendarts::linear_solvers::sparse_matrix_export_format::csr);
#else
		Jacobian->write_matrix_to_file_mm(matrix_filename.c_str());
#endif
		//Jacobian->write_matrix_to_file(("jac_nc_dar_" + std::to_string(output_counter) + ".csr").c_str());
		write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".rhs", RHS);
		write_vector_to_file("jac_nc_dar_" + std::to_string(output_counter) + ".sol", dX);
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

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_elastic_cpu<NC, NP, THERMAL>::post_newtonloop(value_t deltat, value_t time, index_t converged)
{
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
		converged *= 1;
	}

	dev_u = dev_p = dev_e = std::numeric_limits<value_t>::infinity();
	fill(dev_z, dev_z + NC_, std::numeric_limits<value_t>::infinity());

	if (!converged)
	{
		stat.n_newton_wasted += n_newton_last_dt;
		stat.n_linear_wasted += n_linear_last_dt;
		stat.n_timesteps_wasted++;
		converged = 0;

		X = Xn;
		Xref = Xn_ref;
		std::copy(hooke_forces_n.begin(), hooke_forces_n.end(), hooke_forces.begin());
		std::copy(biot_forces_n.begin(), biot_forces_n.end(), biot_forces.begin());
		if constexpr (THERMAL)
		  std::copy(thermal_forces_n.begin(), thermal_forces_n.end(), thermal_forces.begin());
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
		std::copy(hooke_forces.begin(), hooke_forces.end(), hooke_forces_n.begin());
		std::copy(biot_forces.begin(), biot_forces.end(), biot_forces_n.begin());
		if constexpr (THERMAL)
		  std::copy(thermal_forces.begin(), thermal_forces.end(), thermal_forces_n.begin());
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
void engine_super_elastic_cpu<NC, NP, THERMAL>::set_discretizer(DiscretizerType* _discr)
{
  discr = _discr;
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
//			res = fabs(RHS[i * N_VARS + T_VAR] / (PV[i] * op_vals_arr[i * N_OPS + NC] + RV[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i]));
//			if (res > residual)
//				residual = res;
//		}
//	}
//	return residual;
//}
