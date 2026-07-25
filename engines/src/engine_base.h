#ifndef ENGINE_BASE_HPP
#define ENGINE_BASE_HPP

#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <iostream>

#include "globals.h"
#include "conn_mesh.h"
#include "evaluator_iface.h"

#include <pybind11/numpy.h>
namespace py = pybind11;

template <typename T>
inline py::array_t<T> get_raw_array(T* arr, size_t size) {
  return py::array_t<T>(
    { size },
    { sizeof(T) },
    arr,
    py::capsule(arr, [](void* /*f*/) {})
  );
}

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "linear_solvers_data_types.hpp"
#include "csr_matrix.hpp"
#include "block_csr_matrix.hpp"
#include "linsolv_superlu.hpp"
#include "linsolv_mgr.hpp"
using namespace opendarts::linear_solvers;
#else
#include "linsolv_bos_gmres.h"
#include "linsolv_bos_bilu0.h"
#include "linsolv_bos_cpr.h"
#include "linsolv_bos_fs_cpr.h"
#include "csr_matrix.h"
#include "linsolv_bos_amg.h"
#include "linsolv_amg1r5.h"
#include "linsolv_superlu.h"
#include "linsolv_hypre_amg.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef WITH_GPU
#ifdef OPENDARTS_LINEAR_SOLVERS
// Open-source GPU solver wrappers. aips has no open-source counterpart and
// is intentionally not included here.
#include "linsolv_bos_cpr_gpu.hpp"
#include "linsolv_cusparse_ilu.hpp"
#include "linsolv_cusolv.hpp"
#include "linsolv_bicgstab.hpp"
#ifdef WITH_AMGX
#include "linsolv_amgx.hpp"
#endif
#else
#include "linsolv_bos_cpr_gpu.h"
#include "linsolv_aips.h"
#include "linsolv_amgx.h"
#include "linsolv_cusparse_ilu.h"
#include "linsolv_cusolver.h"
#endif // OPENDARTS_LINEAR_SOLVERS
// AMGX solver availability: bos_solvers always ships AMGX; with the
// open-source solvers it is opt-in via the CMake option WITH_AMGX.
#if !defined(OPENDARTS_LINEAR_SOLVERS) || defined(WITH_AMGX)
#define OPENDARTS_GPU_HAS_AMGX
#endif
#endif // WITH_GPU

#ifdef WITH_SAMG
#include "linsolv_samg.h"
#endif

class ms_well;
class operator_set_gradient_evaluator_iface;
class operator_set_gradient_evaluator_iface;

/// This class defines infrastructure for simulation
class engine_base
{
public:
	enum class StateSpecification
	{
		P = 0,
		PT,
		PH,
		PS,
	};

	// methods
public:
	engine_base()
	{
		linear_solver = nullptr;
		Jacobian = nullptr;

		//adjoint method
		linear_solver_ad = 0;
		linear_solver_ad_owned = true;
		linear_solver_ad_uses_jacobian_transpose = false;
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
		linear_solver_owned = true;  // By default, we own the solver
		newton_chop_mode = sim_params::NEWTON_LOCAL_CHOP;
		newton_chop_factor = 0.1;
		log_transform = 0;
		residual_norm_type = sim_params::L2;
	};

	~engine_base()
	{
		if (linear_solver != nullptr && linear_solver_owned)
			delete linear_solver;
		if (Jacobian != nullptr)
			delete Jacobian;

		//adjoint method
		if (linear_solver_ad != nullptr && linear_solver_ad_owned)
			delete linear_solver_ad;
		delete dg_dx_n_temp;

        delete dg_dx_T;
        delete dg_dx_n;
        delete dg_dT_general;
        delete dT_du;
	};

	// get the number of primary unknowns (per block)
	virtual uint8_t get_n_vars() const = 0;

	// get the number of operators (per block) — widened to uint16_t: super-engine
	// N_OPS up to 272 (273 for super-elastic) at NC=30 / NP=3 thermal exceeds uint8_t.
	// Every override across CPU/GPU/elastic/mech engines must match this signature.
	virtual uint16_t get_n_ops() const = 0;

	// get the number of components
	virtual uint8_t get_n_comps() const = 0;
	virtual uint8_t get_n_fl_var() const { return 0; };

	// get the index of Z variable
	virtual uint8_t get_z_var_idx() const = 0;

	// get the number of solid/mineral species
	virtual uint8_t get_n_solid() const { return n_solid; };

	// Number of per-cell history variables fed to OBL interpolation but not part of the Newton
	// system (e.g. trapped/max-gas saturation for Killough hysteresis). Python sets this before
	// engine.init() via `engine.n_history_runtime = k`; 0 disables the Xop / Xhistory code paths.
	uint8_t n_history_runtime = 0;

	virtual uint8_t get_n_history() const { return n_history_runtime; };

	// get the dimension of the OBL interpolation state: Newton unknowns + history variables
	virtual uint8_t get_n_state() const { return get_n_vars() + get_n_history(); };

	// Allocate / resize the history-aware scratch buffers used by build_Xop and project_xop_ders.
	// No-op when no history variables are active.
	// n_ops_ widened to uint16_t to receive super-engine N_OPS up to 273 without truncation.
	void ensure_history_buffers(const index_t n_total, const uint16_t n_ops_)
	{
		const uint8_t n_history = get_n_history();
		if (n_history == 0)
			return;

		const uint8_t n_state = get_n_state();
		if (Xhistory.size() < (size_t)n_total * n_history)
			Xhistory.assign((size_t)n_total * n_history, 0.0);
		Xop.resize((size_t)n_total * n_state);
		op_ders_arr_ext.resize((size_t)n_total * n_ops_ * n_state);
	}

	// initialization
	virtual int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_, operator_set_gradient_evaluator_iface* thermal_var_etor_, sim_params *params, timer_node *timer_) = 0;

	template <uint8_t N_VARS>
	int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_, operator_set_gradient_evaluator_iface* thermal_var_etor_, sim_params *params, timer_node *timer_);

	// Set external linear solver (from Python). The optional name is a
	// human-readable label (e.g. the LinearSolverSpec registry name) used for
	// the "Linear solver type is ..." log line in the open-source build, where
	// the solver is injected rather than selected by the linear_type enum.
	void set_linear_solver(std::shared_ptr<linsolv_iface> solver, const std::string &name = "")
	{
		// If we previously owned a solver, delete it
		if (linear_solver != nullptr && linear_solver_owned)
		{
			delete linear_solver;
		}

		// Store the external solver
		linear_solver_external = solver;
		linear_solver = solver.get();
		linear_solver_owned = false;  // We don't own it, Python does
		external_solver_name = name;

		// If engine is already initialized, wire timers and initialize solver
		if (linear_solver != nullptr && Jacobian != nullptr && params != nullptr)
		{
			if (timer != nullptr)
			{
				linear_solver->init_timer_nodes(&timer->node["linear solver setup"], &timer->node["linear solver solve"]);
			}
			linear_solver->init(Jacobian, params->max_i_linear, params->tolerance_linear);
			// The engine was already initialised, so this call replaces the
			// solver selected at init time (e.g. a model that swaps in MGR after
			// init()). Report the new type so the log reflects the solver that
			// is actually used.
			if (!name.empty())
				std::cout << "Linear solver type is " << name << std::endl;
		}
	}

	// Set external adjoint linear solver (from Python).
	// The legacy adjoint path assembles a scalar transposed matrix dg_dx_T.
	// MGR/CPR-style adjoint solvers can instead keep the simulator block
	// structure and solve Jacobian^T x = b through solve_transposed().
	void set_adjoint_linear_solver(std::shared_ptr<linsolv_iface> solver, bool use_jacobian_transpose = false)
	{
		if (linear_solver_ad != nullptr && linear_solver_ad_owned)
		{
			delete linear_solver_ad;
		}

		linear_solver_ad_external = solver;
		linear_solver_ad = solver.get();
		linear_solver_ad_owned = false;
		linear_solver_ad_uses_jacobian_transpose = use_jacobian_transpose;

		csr_matrix_base *adjoint_matrix = linear_solver_ad_uses_jacobian_transpose ? Jacobian : dg_dx_T;
		if (linear_solver_ad != nullptr && adjoint_matrix != nullptr && params != nullptr)
		{
			if (timer != nullptr)
			{
				linear_solver_ad->init_timer_nodes(&timer->node["linear solver for adjoint method - setup"],
				                                   &timer->node["linear solver for adjoint method - solve"]);
			}
			linear_solver_ad->init(adjoint_matrix, params->max_i_linear, params->tolerance_linear);
		}
	}

	/** Build and attach the native GPU CPRA adjoint stack (device GMRES +
	 *  CPR with AMGX pressure solves on P and P^T + cuSPARSE block-ILU(0)),
	 *  with use_jacobian_transpose = true. Overridden by the GPU super engine
	 *  when AMGX is built; the base returns -1 ("unsupported"). */
	virtual int set_adjoint_solver_cpra_gpu(int /*restart*/ = 150) { return -1; }

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
	virtual double calc_coupled_well_reservoir_residual(int method);

	virtual void average_operator(std::vector<value_t> &av_op);

	// Apply composition correction on initial state: normalize to within [min_sim_z, max_sim_z]
	virtual void apply_composition_correction(std::vector<value_t>& Xi);
	// Apply composition correction on Newton update: normalize to within [min_sim_z, max_sim_z]
	virtual void apply_composition_correction(std::vector<value_t>& X, std::vector<value_t> &dX);
	// Alternative composition correction on Newton update: find intersection of Newton update with compositional domain (not used currently)
	virtual void apply_composition_correction_(std::vector<value_t>& X, std::vector<value_t>& dX);

	virtual void apply_global_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
	virtual void apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX);

	void apply_local_chop_correction_with_solid(std::vector<value_t> &X, std::vector<value_t> &dX);

	void apply_composition_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);
	void apply_global_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);
	void apply_local_chop_correction_new(std::vector<value_t> &X, std::vector<value_t> &dX);

	virtual void apply_thermal_var_correction(std::vector<value_t>& X, std::vector<value_t>& dX);

	// Staged nonlinear-update kernels operating on the engine's own X/dX.
	// Each guards its own applicability; the Python nonlinear solver composes
	// them into the pre-update pipeline prescribed by the solver spec.
	void correct_composition();
	void correct_chop_global();
	void correct_chop_local();
	void correct_obl_axes();
	void correct_thermal();
	/// @brief plain Newton update X -= newton_update_coefficient * dX (resets the coefficient)
	virtual int apply_update(value_t dt);

	/// @brief legacy composite: correction pipeline selected by newton_chop_mode + apply_update
	virtual int apply_newton_update(value_t dt);

	// Here we make the same thing as inside interpolation, but during Newton update
	// It is correct from architectural point of view - X should be changed by engine, not inside interpolator
	virtual void apply_obl_axis_local_correction(std::vector<value_t> &X, std::vector<value_t> &dX);
	/// @brief Install persistent per-variable OBL axis bounds and clamp the current solution once.
	///
	/// Populates every region's op_axis_min/op_axis_max with the given bounds and applies
	/// apply_obl_axis_local_correction immediately. The bounds PERSIST, so the size()>0 gate
	/// in apply_newton_update keeps clamping the solution after every subsequent Newton update
	/// (CPU and GPU -- the GPU engine reuses the host composite). Signature-compatible with the
	/// nonlinear_refactoring (MR327) overload -- both branches implemented this
	/// identically, so the merge keeps a single virtual definition; it is the one
	/// the spec-driven pipeline calls (OBLBoundsSpec.axis_min/axis_max, where
	/// +/-inf entries leave an axis unbounded). If either vector's size differs
	/// from n_vars, a warning is printed and nothing is installed.
	///
	/// @param axis_min lower bound per state variable, size n_vars
	/// @param axis_max upper bound per state variable, size n_vars
	virtual void correct_obl_axes(const std::vector<value_t> &axis_min, const std::vector<value_t> &axis_max);

	// output routines

	virtual int print_timestep(value_t time, value_t deltat, index_t n_newton, index_t n_linear,
							   value_t newton_residual, value_t well_residual);

	int print_header();

	// Build Xop = [X | Xhistory] for reservoir + boundary cells when n_history > 0. No-op otherwise.
	// mesh->Xhistory_bounds supplies the history values to use at boundary cells.
	void build_Xop();

	// After interpolating into op_ders_arr_ext (sized by n_state), copy the first n_vars derivative
	// columns into op_ders_arr (the Newton-sized buffer) so the assembly kernels can consume it
	// with the standard compile-time N_VARS stride. Derivatives w.r.t. history are dropped, which is
	// correct because history values are not Newton unknowns.
	void project_xop_ders();

	/// @brief report for one newton iteration
	virtual int assemble_linear_system(value_t deltat);
	virtual int solve_linear_equation();
	/// @brief commit (converged) or roll back (failed) the timestep state;
	/// the convergence decision is made by the Python nonlinear solver
	virtual int post_newtonloop(value_t deltat, value_t time, index_t converged);

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

#ifdef _OPENMP
	  #pragma omp parallel for
#endif
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
#ifdef _OPENMP
	  #pragma omp parallel for
#endif
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
	std::shared_ptr<linsolv_iface> linear_solver_external;  // For externally provided solvers (Python)
	std::string external_solver_name;  // human-readable label for an injected solver (open-source build)
	bool linear_solver_owned;  // True if we own the solver (need to delete), false if external

	// operator interfaces
	std::vector<operator_set_gradient_evaluator_iface*> acc_flux_op_set_list;
	operator_set_gradient_evaluator_iface* thermal_var_etor;

	uint8_t n_vars;
	// Widened to uint16_t: caches get_n_ops() up to 273 at NC=30 / NP=3 thermal. Used as
	// stride into op_vals_arr / op_ders_arr; uint8_t would silently truncate to 16 mod 256.
	uint16_t n_ops;
	uint8_t nc;
	uint8_t z_var_idx;
	// number of mineral/solid species
	uint8_t n_solid;
	StateSpecification state_spec;
	double min_axis_z;  // OBL axis min for composition
	double max_axis_z;  // OBL axis max for composition
	double min_sim_z;   // Min composition to remain well above OBL min_axis_z and physical bounds (0): min_axis_z + params->sim_eps
	double max_sim_z;   // Max composition to remain well below OBL max_axis_z and physical bounds (1): max_axis_z - params->sim_eps
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
	// GPU engines keep op_vals_arr_n on the device (op_vals_arr_n_d) and skip the
	// 260MB-class host mirror assignment in post_newtonloop.
	bool keep_host_op_vals_n_mirror = true;
	// Called at the start of the accepted-timestep path, before host op_vals_arr
	// consumers (ms_well::calc_rates, FIPS). GPU engines refresh the host mirror
	// here; keeping the hook inside the converged branch makes it follow the
	// convergence verdict wherever that logic lives (C++ or the Python
	// NonlinearSolver of the nonlinear refactoring).
	virtual void sync_host_data_for_accepted_step() {}

	std::vector<value_t> darcy_velocities;	// [NP * n_res_blocks * ND] array of phase (Darcy) velocities for every reservoir cell
	std::vector<value_t> molar_weights;		// [n_regions * NC] molar weights of components
	std::vector<value_t> dispersivity;		// [n_regions * NP * NC] dispersion coefficients
	// History variables: per-cell quantities that feed OBL interpolation but are not Newton unknowns.
	// Used for path-dependent state such as sg_max in Killough hysteresis, while keeping the storage
	// generic for future OBL history variables.
	std::vector<value_t> Xhistory;				// [(n_blocks + n_bounds) * n_history] history values (reservoir cells then boundary cells)
	std::vector<value_t> Xop;				// [(n_blocks + n_bounds) * n_state] extended state vector fed to interpolator; empty unless n_history > 0
	std::vector<value_t> op_ders_arr_ext;	// [(n_blocks + n_bounds) * n_ops * n_state] scratch for interpolator derivative output when n_history > 0

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

	/// @brief linear iterations/residual of the last solve_linear_equation() call
	/// (accumulated per timestep by the Python nonlinear solver)
	index_t last_linear_iters;
	value_t last_linear_residual;

	index_t get_last_linear_iters() const { return last_linear_iters; }
	value_t get_last_linear_residual() const { return last_linear_residual; }

	value_t newton_update_coefficient; // Newton update coefficient for line search

	// nonlinear update controls, owned by the Python nonlinear solver spec
	// (darts.nonlinear_solvers) and synced before every timestep solve
	index_t newton_chop_mode;   // sim_params::newton_solver_t: 0 = none, 1 = global chop, 2 = local chop
	value_t newton_chop_factor; // max composition change per nonlinear update
	index_t log_transform;      // 1 = log-transformed composition variables
	index_t residual_norm_type; // sim_params::nonlinear_norm_t: 0 = L1, 1 = L2, 2 = LINF

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
	/// Allocates the adjoint matrices/solver (host side). Shared by the CPU
	/// (engine_base::init_base) and GPU (engine_base_gpu::init_base) engines.
	void init_adjoint_base();
	/// Allocates the customized-operator arrays/block lists (host side).
	/// Shared by the CPU and GPU init_base, like init_adjoint_base().
	void init_customized_operator_base();

	// assemble dg_dx_n, dg_dT, dj_dx. This is similar to "init_jacobian_structure" in the forward simulation
	virtual int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS) = 0;

	bool opt_history_matching = false;
	/// Selects the adjoint-gradient assembly implementation on the GPU super
	/// engine: true (default) = device kernel (adjoint_gradient_assembly_kernel);
	/// false = the host reference loop (super_engine_adjoint_assembly). The CPU
	/// engine ignores this and always assembles on the host. Exposed to Python
	/// for testing/benchmarking the two implementations against each other.
	bool adjoint_assembly_on_gpu = true;
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
	std::shared_ptr<linsolv_iface> linear_solver_ad_external;
	bool linear_solver_ad_owned;
	bool linear_solver_ad_uses_jacobian_transpose;

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
						   operator_set_gradient_evaluator_iface* thermal_var_etor_,
						   sim_params *params_, timer_node *timer_)
{
	time_t rawtime;
	struct tm *timeinfo;
	char buffer[1024];

	mesh = mesh_;
	wells = well_list_;
	acc_flux_op_set_list = acc_flux_op_set_list_;
	thermal_var_etor = thermal_var_etor_;
	params = params_;
	timer = timer_;

	// Instantiate Jacobian
	if (!Jacobian)
	{
#ifdef OPENDARTS_LINEAR_SOLVERS
		Jacobian = new block_csr_matrix; // unified block-CSR matrix (section 12)
#else
		Jacobian = new csr_matrix<N_VARS>;
		Jacobian->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
#endif
	}

	// allocate Jacobian: the structure arrays are filled in place afterwards
	// by init_jacobian_structure().
#ifdef OPENDARTS_LINEAR_SOLVERS
	(static_cast<block_csr_matrix *>(Jacobian))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);
	Jacobian->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE; // set after init() (init resets type)
#else
	(static_cast<csr_matrix<N_VARS> *>(Jacobian))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);
#endif
#ifdef WITH_GPU
	if (params->linear_type >= params->GPU_GMRES_CPR_AMG)
	{
#ifndef OPENDARTS_LINEAR_SOLVERS
		(static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_device(mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);
#endif
		// block_csr_matrix allocates device storage lazily (dual_array) -- no init_device.
	}
#endif

	std::string linear_solver_type_str;
	// create linear solver
	// Check if external solver was provided (from Python) - if so, use it instead of creating new one
	if (!linear_solver && !linear_solver_external)
	{
#ifdef OPENDARTS_LINEAR_SOLVERS
		// Open-source build: the enum-driven factory below builds the
		// proprietary bos solvers, which are not available here. The linear
		// solver must be injected from Python -- built from a LinearSolverSpec
		// via the open-source registry; see darts_model._apply_linear_solver_spec().
		// Throw instead of exit(1): engine init is entered through pybind11,
		// which translates the exception into a Python RuntimeError -- the
		// previous exit killed the host process (including Jupyter kernels).
		throw std::runtime_error(
		    "no linear solver was provided for " + engine_name +
		    ". The open-source build requires a linear solver injected via "
		    "set_linear_solver() (a LinearSolverSpec built through the "
		    "darts.solvers registry).");
#else
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
#ifdef OPENDARTS_LINEAR_SOLVERS
		case sim_params::CPU_GMRES_MGR:
		{
			// MGR solver is provided externally (Python) via set_linear_solver.
			linear_solver_type_str = "CPU_GMRES_MGR (external pending)";
			break;
		}
#endif

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
			  // N_VARS == 1: CPR collapses to pure AMG, still needs the GMRES outer solver
			  linear_solver = new linsolv_bos_gmres<N_VARS>(1);
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
#ifdef OPENDARTS_GPU_HAS_AMGX
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
			  // N_VARS == 1: CPR collapses to pure AMGX, still needs the GMRES outer solver
			  linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			  linear_solver->set_prec(new linsolv_amgx<1>);
			  linear_solver_type_str = "GPU_GMRES_AMGX";
			}
			break;
		}
#endif // OPENDARTS_GPU_HAS_AMGX
		case sim_params::GPU_GMRES_ILU0:
		{
			linear_solver = new linsolv_bos_gmres<N_VARS>(1);
			linear_solver_type_str = "GPU_GMRES_ILU0";
			break;
		}
#endif
		default:
		{
		    throw std::runtime_error("Linear solver type " +
		        std::to_string(static_cast<int>(params->linear_type)) +
		        " is not supported for " + engine_name);
		}

		}
#endif // OPENDARTS_LINEAR_SOLVERS
	}

	// In the open-source build the solver is injected via set_linear_solver()
	// (built from data_ts.linear_solver through the darts.solvers registry), so
	// the enum-based naming above never ran and linear_solver_type_str is empty.
	// Fall back to the injected label, or a generic note if none was provided.
	if (linear_solver_type_str.empty())
	{
		linear_solver_type_str = external_solver_name.empty()
		    ? std::string("external (injected via set_linear_solver)")
		    : external_solver_name;
	}
	std::cout << "Linear solver type is " << linear_solver_type_str << std::endl;

	n_vars = get_n_vars();
	n_ops = get_n_ops();
	nc = get_n_comps();
	z_var_idx = get_z_var_idx();

	// Sync mesh n_vars with engine n_vars (needed for reverse_and_sort_one_way with IS_DERS=true)
	mesh->n_vars = n_vars;

	// Composition is clipped to the physical simplex [0, 1] ± sim_eps. The adaptive
	// interpolator cache grows on demand outside the prescribed OBL window, so the
	// solver may freely explore state space; (min_axis_z, max_axis_z) now reflect
	// only the physical bound, not the OBL grid.
	min_axis_z = 0.0;
	max_axis_z = 1.0;
	min_sim_z = min_axis_z + params->sim_eps;
	max_sim_z = max_axis_z - params->sim_eps;

	PV.resize(mesh->n_blocks);
	RV.resize(mesh->n_blocks);
	old_z.resize(nc);
	new_z.resize(nc);
	FIPS.resize(nc);
	old_z_fl.resize(nc - n_solid);
	new_z_fl.resize(nc - n_solid);

	X_init = mesh->initial_state;  // initialize only reservoir blocks with mesh->initial_state array
	this->apply_composition_correction(X_init);  // apply composition correction for initial state

	X_init.resize(n_vars * mesh->n_blocks);
	for (index_t i = 0; i < mesh->n_blocks; i++)
	{
		PV[i] = mesh->volume[i] * mesh->poro[i];
		RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
	}

	op_vals_arr.resize(n_ops * mesh->n_blocks);
	op_ders_arr.resize(n_ops * n_vars * mesh->n_blocks);

	// History buffers: allocated only if the engine reports n_history > 0 (see engine_base::get_n_history).
	// Xhistory stores per-cell history values for reservoir cells followed by boundary cells; boundary
	// entries are seeded from mesh->Xhistory_bounds by build_Xop.
	ensure_history_buffers(mesh->n_blocks + mesh->n_bounds, n_ops);

	t = 0;

	time(&rawtime);
	timeinfo = localtime(&rawtime);


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

	if (linear_solver)
	{
		linear_solver->init_timer_nodes(&timer->node["linear solver setup"], &timer->node["linear solver solve"]);
		// initialize linear solver
		linear_solver->init(Jacobian, params->max_i_linear, params->tolerance_linear);
	}
	else
	{
		std::cerr << "WARNING: Linear solver not set yet; call engine.set_linear_solver(...) before run."
		          << std::endl;
	}

	//Xn.resize (n_vars * mesh->n_blocks);
	RHS.resize(n_vars * mesh->n_blocks);
	dX.resize(n_vars * mesh->n_blocks);

	sprintf(buffer, "\nSTART SIMULATION\n-------------------------------------------------------------------------------------------------------------\n");
	std::cout << buffer << std::flush;

	for (ms_well *w : wells)
	{
		// initialize the state of well segments
		if (w->ms_type == ms_well::MS_Type::EPM)
			w->initialize_control_epm(X_init);
		else if (w->ms_type == ms_well::MS_Type::DFM && w->control.get_well_control_type() > well_control_iface::WellControlType::NONE)
			w->initialize_control_dfm(X_init);
		else if (w->ms_type == ms_well::MS_Type::DFM && w->control.get_well_control_type() == well_control_iface::WellControlType::NONE)
			std::copy(w->init_state.begin(), w->init_state.end(), X_init.begin() + w->well_head_idx * n_vars);
	}

	Xn = X = X_init;
	dt = 0.0; // timestep sizing is owned by the Python driver
	prev_usual_dt = dt;

	// initialize arrays for every operator set
	block_idxs.resize(acc_flux_op_set_list.size());
	op_axis_min.resize(acc_flux_op_set_list.size());
	op_axis_max.resize(acc_flux_op_set_list.size());

	// op_axis_min/op_axis_max are intentionally left empty (default-constructed inner
	// vectors with .size() == 0). The size() == 0 check in apply_newton_update gates
	// apply_obl_axis_local_correction, so leaving these empty disables the per-axis
	// clamp — Newton may freely explore state space and the adaptive cache grows on
	// demand. The per-region block_idxs map below is still populated normally.
	for (int r = 0; r < acc_flux_op_set_list.size(); r++)
	{
		block_idxs[r].clear();
	}

	// create a block list for every operator set
	index_t idx = 0;
	for (auto op_region : mesh->op_num)
	{
		block_idxs[op_region].emplace_back(idx++);
	}

	if (get_n_history() > 0)
	{
		build_Xop();
		for (int r = 0; r < (int)acc_flux_op_set_list.size(); r++)
			acc_flux_op_set_list[r]->evaluate_with_derivatives(Xop, block_idxs[r], op_vals_arr, op_ders_arr_ext);
		project_xop_ders();
	}
	else
	{
		for (int r = 0; r < (int)acc_flux_op_set_list.size(); r++)
			acc_flux_op_set_list[r]->evaluate_with_derivatives(X, block_idxs[r], op_vals_arr, op_ders_arr);
	}
	op_vals_arr_n = op_vals_arr;

	time_data.clear();
	time_data_report.clear();

	// for adjoint method------------------------------------------

	if (opt_history_matching)
	{
		init_adjoint_base();
	}


	if (customize_operator)
	{
		init_customized_operator_base();
	}

	well_control_arr.clear();



	return 0;
}

#endif
