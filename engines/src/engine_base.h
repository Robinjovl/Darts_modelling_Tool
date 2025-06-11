#ifndef ENGINE_BASE_HPP
#define ENGINE_BASE_HPP

#include <vector>
#include <unordered_map>
#include <cmath>
#include <iostream>

#include "globals.h"
#include "conn_mesh.h"
#include "interpolator_base.hpp"
#include "pybind11/py_globals.h"

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/data_types.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_gmres.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_bilu0.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_cpr.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_fs_cpr.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
using namespace opendarts::linear_solvers;
#else
#include "linsolv_bos_gmres.h"
#include "linsolv_bos_bilu0.h"
#include "linsolv_bos_cpr.h"
#include "linsolv_bos_fs_cpr.h"
#include "csr_matrix.h"
#endif // OPENDARTS_LINEAR_SOLVERS 

#ifdef WITH_GPU
#include "linsolv_bos_cpr_gpu.h"
#include "linsolv_aips.h"
#include "linsolv_amgx.h"
#include "linsolv_adgprs_nf.h"
#include "linsolv_cusparse_ilu.h"
#include "linsolv_cusolver.h"
#endif

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/linsolv_bos_amg.hpp"
// #include "openDARTS/linear_solvers/linsolv_amg1r5.h"
#include "openDARTS/linear_solvers/linsolv_superlu.hpp"
#else
#include "linsolv_bos_amg.h"
#include "linsolv_amg1r5.h"
#include "linsolv_superlu.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef WITH_HYPRE
#include "linsolv_hypre_amg.h"
#endif

#ifdef WITH_SAMG
#include "linsolv_samg.h"
#endif

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::auxiliary;
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS


class ms_well;
class operator_set_gradient_evaluator_iface;
class operator_set_gradient_evaluator_iface;

/// This class defines infrastructure for simulation
class engine_base
{
	// methods
public:
	engine_base()
	{
		linear_solver = nullptr;
		Jacobian = nullptr;

		//adjoint method
		linear_solver_ad = 0;
		dg_dx_n_temp = 0;

        dg_dx_T = 0;
        dg_dx_n = 0;
        dg_dT_general = 0;
        dT_du = 0;

		print_linear_system = false;
		output_counter = 0;
		enabled_flux_output = false;
		is_fickian_energy_transport_on = true;
		newton_update_coefficient = 1.0;
		n_solid = 0;
	};

	~engine_base()
	{
		if (linear_solver != nullptr)
			delete linear_solver;
		if (Jacobian != nullptr)
			delete Jacobian;

		//adjoint method
		delete linear_solver_ad;
		delete dg_dx_n_temp;

        delete dg_dx_T;
        delete dg_dx_n;
        delete dg_dT_general;
        delete dT_du;
	};

	// get the number of primary unknowns (per block)
	virtual uint8_t get_n_vars() const = 0;

	// get the number of operators (per block)
	virtual uint8_t get_n_ops() const = 0;

	// get the number of components
	virtual uint8_t get_n_comps() const = 0;
	virtual uint8_t get_n_fl_var() const { return 0; };

	// get the index of Z variable
	virtual uint8_t get_z_var() const = 0;

	// get the number of solid/mineral species
	virtual uint8_t get_n_solid() const { return n_solid; };

	// initialization
	virtual int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_, sim_params *params, timer_node *timer_) = 0;

	template <uint8_t N_VARS>
	int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_, sim_params *params, timer_node *timer_);

	virtual int init_jacobian_structure(csr_matrix_base *jacobian);

	// newton loop
	virtual int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS) = 0;

	virtual double calc_newton_residual();
	virtual double calc_newton_residual_L1();
	virtual double calc_newton_residual_L2();
	virtual double calc_newton_residual_Linf();
	virtual double calc_well_residual();
	virtual double calc_well_residual_L1();
	virtual double calc_well_residual_L2();
	virtual double calc_well_residual_Linf();

	virtual void average_operator(std::vector<value_t> &av_op);

	virtual void apply_composition_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
	virtual void apply_composition_correction_(std::vector<value_t>& X, std::vector<value_t>& dX);

	virtual void apply_global_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
	virtual void apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX);

	void apply_local_chop_correction_with_solid(std::vector<value_t> &X, std::vector<value_t> &dX);

	void apply_composition_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);
	void apply_global_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);
	void apply_local_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);

	virtual int apply_newton_update(value_t dt);

	// Here we make the same thing as inside interpolation, but during Newton update
	// It is correct from architectural point of view - X should be changed by engine, not inside interpolator
	virtual void apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX);

	// output routines

	virtual int print_timestep(value_t time, value_t deltat);

	int print_header();

	/// @brief report for one newton iteration
	virtual int assemble_linear_system(value_t deltat);
	virtual int solve_linear_equation();
	virtual int post_newtonloop(value_t deltat, value_t time);

	/// @brief reports complete information about well regimes
	virtual int report();

	/// @brief print statistics for the current run
	virtual int print_stat();

	virtual int test_assembly(int n_times, int kernel_number = 0, int dump_jacobian = 0);

	virtual int test_spmv(int n_timer, int kernel_number = 0, int dump_result = 0);

	/// @brief row-wise scaling of Jacobian by maximum value
	template<uint16_t N_VARS>
	void dimensionalize_rows()
	{
	  constexpr uint16_t N_VARS_SQ = N_VARS * N_VARS;
	  const index_t n_blocks = mesh->n_blocks;
	  value_t* Jac = Jacobian->get_values();
	  const index_t* rows = Jacobian->get_rows_ptr();

	  // maximum values
	  std::fill_n(max_row_values_inv.data(), n_blocks * N_VARS, 0.0);

	  #pragma omp parallel for
	  for (index_t i = 0; i < n_blocks; i++)
	  {
		index_t csr_start = rows[i];
		index_t csr_end = rows[i + 1];
		for (index_t j = csr_start; j < csr_end; j++)
		{
		  const index_t base = j * N_VARS_SQ;
		  for (uint8_t c = 0; c < N_VARS; c++)
		  {
			value_t current_max = max_row_values_inv[i * N_VARS + c];
			for (uint8_t v = 0; v < N_VARS; v++)
			{
			  const index_t idx = base + c * N_VARS + v;
			  value_t val = fabs(Jac[idx]);
			  if (val > current_max)
				current_max = val;
			}
			max_row_values_inv[i * N_VARS + c] = current_max;
		  }
		}
	  }

	  // compute inverses
	  for (index_t i = 0; i < n_blocks; i++) 
	  {
		for (uint8_t c = 0; c < N_VARS; c++) 
		{
		  value_t& val = max_row_values_inv[i * N_VARS + c];
		  if (val != 0.0)
			val = 1.0 / val;
		  else
			val = 1.0;
		}
	  }

	  // scaling
	  #pragma omp parallel for
	  for (index_t i = 0; i < n_blocks; i++)
	  {
		index_t csr_start = rows[i];
		index_t csr_end = rows[i + 1];
		value_t inv_vals[N_VARS];

		// copy values to local array
		for (uint8_t c = 0; c < N_VARS; c++)
		  inv_vals[c] = max_row_values_inv[i * N_VARS + c];

		// scale jacobian
		for (index_t j = csr_start; j < csr_end; j++) 
		{
		  const index_t base = j * N_VARS_SQ;
		  for (uint8_t c = 0; c < N_VARS; c++) 
		  {
			for (uint8_t v = 0; v < N_VARS; v++) 
			  Jac[base + c * N_VARS + v] *= inv_vals[c];
		  }
		}

		// scale residual
		for (uint8_t c = 0; c < N_VARS; c++)
		  RHS[i * N_VARS + c] *= inv_vals[c];
	  }
	};
	
	/// @} // end of Methods

	// properties
public:
	/** @defgroup Engine_parameters
	 *  Parameters in base engine class exposed to Python
	 *  @{
	 */

	/// @brief space dimension
	const static uint8_t ND = 3;

	/// @brief vector of unknowns in the current timestep
	std::vector<value_t> X;

	/// @brief vector of unknowns in the previous timestep
	std::vector<value_t> Xn;

	/// @brief current timestep
	value_t t;

	/// @brief pointer to mesh
	conn_mesh *mesh;

	/// @brief simulation parameters
	sim_params *params;

	/// @brief simulation statistics
	sim_stat stat;

	/// @brief vector of wells
	std::vector<ms_well *> wells;

	/// @brief unsorted map containing well information (BHP, rates)
	std::unordered_map<std::string, std::vector<value_t>> time_data;

	// @brief python wrapper for jacobian values
	py::array_t<value_t> jac_vals;
	
	// @brief python wrappers for storing BCSR jacobian structure
	py::array_t<index_t> jac_rows, jac_cols, jac_diags;

	// @brief method to initialize python wrappers for Jacobian matrix
	void expose_jacobian()
	{
	  value_t* values = Jacobian->get_values();
	  index_t* rows = Jacobian->get_rows_ptr();
	  index_t* cols = Jacobian->get_cols_ind();
	  index_t* diag_ind = Jacobian->get_diag_ind();

	  jac_vals = get_raw_array(values, n_vars * n_vars * rows[mesh->n_blocks]);
	  jac_rows = get_raw_array(rows, mesh->n_blocks + 1);
	  jac_cols = get_raw_array(cols, rows[mesh->n_blocks]);
	  jac_diags = get_raw_array(diag_ind, mesh->n_blocks);
	};

	/// @} // end of Parameters

	linsolv_iface *linear_solver;

	//operator_set_gradient_evaluator_iface* acc_flux_op_set;
	std::vector<operator_set_gradient_evaluator_iface *> acc_flux_op_set_list;

	uint8_t n_vars;
	uint8_t n_ops;
	uint8_t nc;
	uint8_t z_var;
	// number of mineral/solid species
	uint8_t n_solid;
	double min_zc;
	double max_zc;
	std::vector<value_t> old_z, new_z; // [NC] array for local chop
	std::vector<value_t> old_z_fl, new_z_fl; // [NC_FLUID] array for local chop

	std::vector<value_t> X_init;				   // [N_VARS * n_blocks] array of initial solution
	std::vector<value_t> PV;					   // [n_blocks]     array of initial pore volumes
	std::vector<value_t> RV;					   // [n_blocks]     array of initial rock volumes
	std::vector<std::vector<index_t>> block_idxs;  // [N_OPS_NUM] array of block indices corresponding to given operator set number
	std::vector<std::vector<value_t>> op_axis_min; // [N_OPS_NUM] array of axis minimum values for each operator set
	std::vector<std::vector<value_t>> op_axis_max; // [N_OPS_NUM] array of axis minimum values for each operator set

	// storage for interpolated operator values and derivatives
	std::vector<value_t> op_vals_arr;	// [N_OPS * n_blocks] array of values of operators
	std::vector<value_t> op_ders_arr;	// [N_OPS * N_VARS * n_blocks] array of dedrivatives of operators
	std::vector<value_t> op_vals_arr_n; // [N_OPS * n_blocks] array of values of operators from the last timestep

	std::vector<value_t> darcy_velocities;	// [NP * n_res_blocks * ND] array of phase (Darcy) velocities for every reservoir cell
	std::vector<value_t> molar_weights;		// [n_regions * NC] molar weights of components
	std::vector<value_t> dispersivity;		// [n_regions * NP * NC] dispersion coefficients

	// rates, bhps, FIPs, etc
	std::unordered_map<std::string, std::vector<value_t>> time_data_report;
	std::vector<value_t> FIPS;

	// linear system
	csr_matrix_base *Jacobian;
	std::vector<value_t> X0, RHS, dX;

	value_t dt, prev_usual_dt, stop_time;
	
	index_t output_counter;
	bool print_linear_system;

	// switch on/off heat fluxes related to Fickian mass transport
	bool is_fickian_energy_transport_on;

	// statistics
	value_t CFL_max; // maximum value of CFL for last Jacobian assebly
	index_t n_newton_last_dt, n_linear_last_dt;
	double newton_residual_last_dt;
	double well_residual_last_dt;
	int linear_solver_error_last_dt;

	value_t newton_update_coefficient; // Newton update coefficient for line search

	timer_node *timer;
	timer_node full_step_timer;
	double full_step_run_timer, t_full_step; // for more accurate estimation of time left

	std::string engine_name;

	// flags to apply dimension-based and row-wise scaling respectively
	bool scale_dimless, scale_rows;

	// dimensions for scaling
	value_t e_dim, t_dim, m_dim, p_dim;

	// maximum absolute values in rows of jacobian
	std::vector<value_t> max_row_values_inv;

	// flag to turn on fluxes output
	bool enabled_flux_output;
	virtual void enable_flux_output() {};

	// mass fluxes
	std::vector<value_t> darcy_fluxes;
	std::vector<value_t> diffusion_fluxes;
	std::vector<value_t> dispersion_fluxes;
	// energy fluxes
	std::vector<value_t> heat_darcy_advection_fluxes;
	std::vector<value_t> heat_diffusion_advection_fluxes;
	std::vector<value_t> heat_dispersion_advection_fluxes;
	std::vector<value_t> fourier_fluxes;

	// adjoint method--------------------------------------------------------------------------------------

	// initialize dg_dT_general, which is similar to the jacobian initialization
	int init_adjoint_structure(csr_matrix_base* init_adjoint);  

	// assemble dg_dx_n, dg_dT, dj_dx. This is similar to "init_jacobian_structure" in the forward simulation
	virtual int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS) = 0;

	bool opt_history_matching = false;
	bool optimize_component_rate = false;
	bool optimize_phase_rate = false;

	bool is_mp = false;  // MPFA or MPSA

	bool objfun_prod_phase_rate = false;
	bool objfun_inj_phase_rate = false;
	bool objfun_BHP = false;
	bool objfun_well_tempr = false;
	bool objfun_temperature = false;
	bool objfun_customized_op = false;
	bool objfun_saturation = false;
	std::vector<value_t> Temp_dj_dx, Temp_dj_du;

	csr_matrix_base* dg_dx_T;
	csr_matrix_base* dg_dx_n;
	//csr_matrix_base* dg_dT;
	csr_matrix_base* dg_dT_general;
	csr_matrix_base* dT_du;

	csr_matrix_base* dg_dx_n_temp;

	std::vector<int> col_dT_du;
	index_t n_control_vars;

	std::vector<value_t> Xop_mp;
	std::vector<std::vector<value_t>> X_t, X_t_report, Xop_t;
	std::vector<value_t> dt_t, t_t, dt_t_report, t_t_report;
	std::vector<int> well_head_idx_collection;
	std::vector<int> well_head_tran_idx_collection;
	std::string unit;
	std::vector<int> component_index, phase_index;
	std::vector<std::string> prod_well_name, inj_well_name, BHP_well_name, well_tempr_name;
	std::vector<std::string> prod_phase_name, inj_phase_name;
	std::string well;
	std::vector<value_t> cov_mat_inv, dirac_vec;

	index_t upstream_index, downstream_index;

	linsolv_iface* linear_solver_ad;

	// the total number of the cell interfaces, 
    // including 1. res to res (trans), 2. res to well_body (WI), 3. well_body to well_head
	// n_interfaces = mesh->n_conns / 2;
	index_t n_interfaces;


	std::vector<std::vector<value_t>> Q;
	int add_value_to_Q(std::vector<value_t> A)
	{
		Q.push_back(A);
		return 0;
	};

	int clear_Q()
	{
		Q.clear();
		return 0;
	};




	std::vector<value_t> phase_relative_density;

	typedef std::vector<std::vector<std::vector<value_t>>> vec_3d;

	int prepare_dj_dx(vec_3d q, vec_3d q_inj,
		std::vector<std::vector<value_t>> bhp, std::vector<std::vector<value_t>> well_tempr, 
		std::vector<std::vector<value_t>> temperature, std::vector<std::vector<value_t>> customized_op,
		index_t idx_sim_ts, index_t idx_obs_ts);

	// producer rate, covariance, weights
	vec_3d Q_all;
	std::vector<std::vector<value_t>> Q_p;
	int clear_Q_p() { Q_p.clear(); return 0; };
	int add_value_to_Q_p(std::vector<value_t> A) { Q_p.push_back(A); return 0; };
	int push_back_to_Q_all() { Q_all.push_back(Q_p); return 0; };
	vec_3d cov_mat_inv_prod_all;
	std::vector<std::vector<value_t>> cov_mat_inv_prod_p;
	int clear_cov_prod_p() { cov_mat_inv_prod_p.clear(); return 0; };
	int add_value_to_cov_prod_p(std::vector<value_t> A) { cov_mat_inv_prod_p.push_back(A); return 0; };
	int push_back_to_cov_prod_all() { cov_mat_inv_prod_all.push_back(cov_mat_inv_prod_p); return 0; };
	vec_3d prod_weights_all;
	std::vector<std::vector<value_t>> prod_weights_p;
	int clear_prod_wei_p() { prod_weights_p.clear(); return 0; };
	int add_value_to_prod_wei_p(std::vector<value_t> A) { prod_weights_p.push_back(A); return 0; };
	int push_back_to_prod_wei_all() { prod_weights_all.push_back(prod_weights_p); return 0; };


	// injector rate, covariance, weights
	vec_3d Q_inj_all;
	std::vector<std::vector<value_t>> Q_inj_p;
	int clear_Q_inj_p() { Q_inj_p.clear(); return 0; };
	int add_value_to_Q_inj_p(std::vector<value_t> A) { Q_inj_p.push_back(A); return 0; };
	int push_back_to_Q_inj_all() { Q_inj_all.push_back(Q_inj_p); return 0; };
	vec_3d cov_mat_inv_inj_all;
	std::vector<std::vector<value_t>> cov_mat_inv_inj_p;
	int clear_cov_inj_p() { cov_mat_inv_inj_p.clear(); return 0; };
	int add_value_to_cov_inj_p(std::vector<value_t> A) { cov_mat_inv_inj_p.push_back(A); return 0; };
	int push_back_to_cov_inj_all() { cov_mat_inv_inj_all.push_back(cov_mat_inv_inj_p); return 0; };
	vec_3d inj_weights_all;
	std::vector<std::vector<value_t>> inj_weights_p;
	int clear_inj_wei_p() { inj_weights_p.clear(); return 0; };
	int add_value_to_inj_wei_p(std::vector<value_t> A) { inj_weights_p.push_back(A); return 0; };
	int push_back_to_inj_wei_all() { inj_weights_all.push_back(inj_weights_p); return 0; };


	// BHP, covariance, weights
	std::vector<std::vector<value_t>> BHP_all;
	int push_back_to_BHP_all(std::vector<value_t> A) { BHP_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> cov_mat_inv_BHP_all;
	int push_back_to_cov_BHP_all(std::vector<value_t> A) { cov_mat_inv_BHP_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> BHP_weights_all;
	int push_back_to_BHP_wei_all(std::vector<value_t> A) { BHP_weights_all.push_back(A); return 0; };


	// well temperature, covariance, weights
	std::vector<std::vector<value_t>> well_tempr_all;
	int push_back_to_well_tempr_all(std::vector<value_t> A) { well_tempr_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> cov_mat_inv_well_tempr_all;
	int push_back_to_cov_well_tempr_all(std::vector<value_t> A) { cov_mat_inv_well_tempr_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> well_tempr_weights_all;
	int push_back_to_well_tempr_wei_all(std::vector<value_t> A) { well_tempr_weights_all.push_back(A); return 0; };


	// temperature, covariance, weights
	std::vector<std::vector<value_t>> temperature_all;
	int push_back_to_temperature_all(std::vector<value_t> A) { temperature_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> cov_mat_inv_temperature_all;
	int push_back_to_cov_temperature_all(std::vector<value_t> A) { cov_mat_inv_temperature_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> temperature_weights_all;
	int push_back_to_temperature_wei_all(std::vector<value_t> A) { temperature_weights_all.push_back(A); return 0; };


	// customized operator, covariance, weights
	std::vector<std::vector<value_t>> customized_op_all;
	int push_back_to_customized_op_all(std::vector<value_t> A) { customized_op_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> cov_mat_inv_customized_op_all;
	int push_back_to_cov_customized_op_all(std::vector<value_t> A) { cov_mat_inv_customized_op_all.push_back(A); return 0; };
	std::vector<std::vector<value_t>> customized_op_weights_all;
	int push_back_to_customized_op_wei_all(std::vector<value_t> A) { customized_op_weights_all.push_back(A); return 0; };
	double threshold;
	std::vector<std::vector<value_t>> binary_all;
	int push_back_to_binary_all(std::vector<value_t> A) { binary_all.push_back(A); return 0; };




	int clear_previous_adjoint_assembly()
	{
		X_t.clear();
		dt_t.clear();
		t_t.clear();
		Xop_t.clear();

		X_t_report.clear();
		dt_t_report.clear();
		t_t_report.clear();

		dirac_vec.clear();

		time_data_report_customized.clear();
		time_data_customized.clear();
		return 0;
	};

	// this is the key function to be called in Python to compute the adjoint gradient.
	int calc_adjoint_gradient_dirac_all();
	std::vector<value_t> derivatives;
	double scale_function_value;

	bool customize_operator = false;
	std::vector<index_t> customize_op_num;
	std::vector<std::vector<index_t>> customize_block_idxs;   // array of block indices corresponding to given operator set number
	index_t idx_customized_operator;  // the idx of your customized operator
	std::vector<value_t> op_vals_arr_customized;   // [1 * n_blocks] array of values of operators
	std::vector<value_t> op_ders_arr_customized;   // [1 * N_VARS * n_blocks] array of dedrivatives of operators
	std::vector<std::vector<ms_well>> well_control_arr;  // time, well, and control
	std::vector<std::vector<value_t>> time_data_report_customized;
	std::vector<std::vector<value_t>> time_data_customized;
	std::vector<value_t> X_next;  // the X at the next time step
	index_t idx_ts = 0;


	std::vector<value_t> flux_multiplier;
	value_t test_value = 1.1;
	index_t test_index = 1;
	std::vector<value_t> test_value_vec;
	std::vector<index_t> test_index_vec;

	well_control_iface::WellControlType observation_rate_type;
};

template <uint8_t N_VARS>
int engine_base::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
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
		(static_cast<csr_matrix<N_VARS> *>(Jacobian))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);
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
		(static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_device(mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);
	}
#endif

	std::string linear_solver_type_str;	
	// create linear solver
	if (!linear_solver)
	{
		switch (params->linear_type)
		{
		case sim_params::CPU_GMRES_CPR_AMG:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			if constexpr (N_VARS > 1)
			{
			  linsolv_iface* cpr = new linsolv_bos_cpr<N_VARS>;
			  cpr->set_prec(new linsolv_bos_amg<1>);
			  linear_solver->set_prec(cpr);
			  linear_solver_type_str = "CPU_GMRES_CPR_AMG";
			}
			else
			{
			  linear_solver->set_prec(new linsolv_bos_amg<1>);
			  linear_solver_type_str = "CPU_GMRES_AMG";
			}

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
			linear_solver_type_str = "CPU_GMRES_CPR_AMG1R5";
			break;
		}
#endif 
#endif //_WIN32
		case sim_params::CPU_GMRES_ILU0:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>;
			linear_solver->set_prec(new linsolv_bos_bilu0<N_VARS>);
			linear_solver_type_str = "CPU_GMRES_ILU0";
			break;
		}
		case sim_params::CPU_SUPERLU:
		{
			linear_solver = new linsolv_superlu<N_VARS>;
			linear_solver_type_str = "CPU_SUPERLU";
			break;
		}

#ifdef WITH_GPU
		case sim_params::GPU_GMRES_CPR_AMG:
		{
			if constexpr (N_VARS > 1)
			{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			linsolv_iface *cpr = new linsolv_bos_cpr_gpu<N_VARS>;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 0;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 0;
			((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 1;
			cpr->set_prec(new linsolv_bos_amg<1>);
			linear_solver->set_prec(cpr);
			linear_solver_type_str = "GPU_GMRES_CPR_AMG";
			}
			else
			{
			  linear_solver->set_prec(new linsolv_bos_amg<1>);
			  linear_solver_type_str = "GPU_GMRES_AMG";
			}
			break;
		}
#ifdef WITH_AIPS
		case sim_params::GPU_GMRES_CPR_AIPS:
		{
			if constexpr (N_VARS > 1)
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
			}
			linear_solver_type_str = "GPU_GMRES_CPR_AIPS";
			break;
		}
#endif //WITH_AIPS
		case sim_params::GPU_GMRES_CPR_AMGX_ILU:
		{
			if constexpr (N_VARS > 1)
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
			linear_solver_type_str = "GPU_GMRES_CPR_AMGX_ILU";
			}
			else
			{
			  linear_solver->set_prec(new linsolv_amgx<1>);
			  linear_solver_type_str = "GPU_GMRES_AMGX";
			}
			break;
		}
#ifdef WITH_ADGPRS_NF
		case sim_params::GPU_GMRES_CPR_NF:
		{
			if constexpr (N_VARS > 1)
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
			}
			linear_solver_type_str = "GPU_GMRES_CPR_NF";
			break;
		}
#endif //WITH_ADGPRS_NF
		case sim_params::GPU_GMRES_ILU0:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			linear_solver_type_str = "GPU_GMRES_ILU0";
			break;
		}
#endif
		default:
		{
		    std::cerr << "Linear solver type " << params->linear_type << " is not supported for " << engine_name << std::endl << std::flush;
		    exit(1);
		}
		
		}
	}

	std::cout << "Linear solver type is " << linear_solver_type_str << std::endl;

	n_vars = get_n_vars();
	n_ops = get_n_ops();
	nc = get_n_comps();
	z_var = get_z_var();

	PV.resize(mesh->n_blocks);
	RV.resize(mesh->n_blocks);
	old_z.resize(nc);
	new_z.resize(nc);
	FIPS.resize(nc);
	old_z_fl.resize(nc - n_solid);
	new_z_fl.resize(nc - n_solid);

	X_init = mesh->initial_state;  // initialize only reservoir blocks with mesh->initial_state array
	X_init.resize(n_vars * mesh->n_blocks);
	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
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
	init_jacobian_structure(Jacobian);

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







	// for adjoint method------------------------------------------

	if (opt_history_matching)
	{
		n_interfaces = mesh->n_conns / 2;

		// prepare dg_dx_n_temp
		init_adjoint_structure(dg_dx_n_temp);

		// here we remove wells.size() transmissibility between well head and well body (i.e. segment_transmissibility)
		// because there is no need to optimize segment_transmissibility, which is usually a large value of 100000
		std::vector<int> Temp_1(n_interfaces - wells.size(), 0);  
		col_dT_du = Temp_1;


		// initialization of linear solver
		if (!linear_solver_ad)
		{
			if (0)
			{
				// so far these preconditioner and the linear solver can't be applied to adjoint for some reason
				linear_solver_ad = new linsolv_bos_gmres<1>;
				linear_solver_ad->set_prec(new linsolv_bos_bilu0<1>);

			}
			else
				linear_solver_ad = new linsolv_superlu<1>;
		}
		linear_solver_ad->init_timer_nodes(&timer->node["linear solver for adjoint method - setup"], &timer->node["linear solver for adjoint method - solve"]);

		well_head_idx_collection.clear();
		for (ms_well* w : wells)
		{
			well_head_idx_collection.push_back(w->well_head_idx);
		}

		dg_dx_T = new csr_matrix<1>;
		dg_dx_T->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;

		dg_dx_n = new csr_matrix<1>;
		dg_dx_n->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;

		//dg_dT = new csr_matrix<1>;
		//dg_dT->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;

		dg_dT_general = new csr_matrix<1>;
		dg_dT_general->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;

		(static_cast<csr_matrix<1>*>(dg_dx_T))->init(mesh->n_blocks * n_vars, mesh->n_blocks * n_vars, 1, (mesh->n_conns + mesh->n_blocks) * n_vars * n_vars);
		(static_cast<csr_matrix<1>*>(dg_dx_n))->init(mesh->n_blocks * n_vars, mesh->n_blocks * n_vars, 1, (mesh->n_conns + mesh->n_blocks) * n_vars * n_vars);
		//(static_cast<csr_matrix<1>*>(dg_dT))->init(mesh->n_blocks * n_vars, n_interfaces - wells.size(), 1, ((mesh->n_blocks) * 2 - 2 * wells.size()) * n_vars);
		(static_cast<csr_matrix<1>*>(dg_dT_general))->init(mesh->n_blocks * n_vars, n_interfaces, 1, (mesh->n_conns) * n_vars);
		//init_adjoint_structure(dg_dT);
		init_adjoint_structure(dg_dT_general);


		dT_du = new csr_matrix<1>;
		dT_du->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;

		//(static_cast<csr_matrix<1>*>(dT_du))->init(n_interfaces - wells.size(), n_control_vars, 1, n_interfaces - wells.size());
		(static_cast<csr_matrix<1>*>(dT_du))->init(n_interfaces - wells.size(), n_interfaces - wells.size(), 1, n_interfaces - wells.size());

	}


	if (customize_operator)
	{
		time_data_report_customized.clear();
		time_data_customized.clear();

		// WARNING: this variable shadows a member variable of a different type
        index_t n_ops = 1;  // here '1' is to distinguish the size of the customized operator with the ordinary operator

		op_vals_arr_customized.resize(n_ops * mesh->n_blocks);   // [1 * n_blocks] array of values of operators
		op_ders_arr_customized.resize(n_ops * n_vars * mesh->n_blocks);   // [1 * N_VARS * n_blocks] array of dedrivatives of operators

		// create a block list for the customized operator
		customize_block_idxs.resize(acc_flux_op_set_list.size());
		for (auto op_region : customize_op_num)
		{
			customize_block_idxs[op_region].clear();
		}

		index_t idx = 0;
		for (auto op_region : customize_op_num)
		{
			customize_block_idxs[op_region].emplace_back(idx++);
		}
	}

	well_control_arr.clear();



	return 0;
}

#endif
