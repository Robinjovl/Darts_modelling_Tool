#ifndef ENGINE_BASE_GPU_H
#define ENGINE_BASE_GPU_H

#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif


#include "engine_base.h"
#ifdef OPENDARTS_LINEAR_SOLVERS
#include "block_csr_matrix.hpp"
#include "csr_matrix.hpp"
#else
#include "csr_matrix.h"
#endif
#include "gpu_tools.h"  // engine-local GPU kernel-launch helpers (engines/src)
#ifdef WITH_GPU
#ifdef OPENDARTS_LINEAR_SOLVERS
#include "linsolv_bicgstab.hpp"
#include "linsolv_gmres_gpu.hpp"
#include "linsolv_cusparse_ilu.hpp"
#include "linsolv_cusolv.hpp"
#ifdef WITH_CUDSS
#include "linsolv_cudss.hpp"
#endif
#include "linsolv_mcsgs.hpp"
#include "linsolv_schur_elim.hpp"
#ifdef OPENDARTS_GPU_HAS_AMGX
#include "linsolv_amgx.hpp"
#include "linsolv_bos_cpr_gpu.hpp"
#endif
#else
#include "linsolv_bicgstab.h"
#endif
#define KERNEL_BLOCK_SIZE 128

#if defined(OPENDARTS_LINEAR_SOLVERS) && defined(OPENDARTS_GPU_HAS_AMGX)
/// Build the in-tree GPU AMGX-CPR chain for block size NV: Krylov outer
/// (GMRES or BiCGStab) around linsolv_bos_cpr_gpu with AMGX on the pressure
/// system and cuSPARSE block-ILU(0) (or a DARTS_CPR_STAGE2 experiment hook)
/// as the full-system stage. Factored out of engine_base_gpu::init_base so
/// the local-elimination wrapper can build the same chain one block size
/// smaller (see params->schur_elim_count).
template <uint8_t NV>
inline opendarts::linear_solvers::linsolv_iface *make_gpu_amgx_cpr_chain(
    int device_num, bool use_bicgstab, std::string &linear_solver_type_str,
    int amgx_reuse_override = -1)
{
  using namespace opendarts::linear_solvers;
  auto *cpr = new linsolv_bos_cpr_gpu<NV>;
  cpr->p_solver_setup_gpu = 1;
  cpr->p_solver_solve_gpu = 1;
  cpr->p_solver_requires_diag_first = 0;
  cpr->set_p_system_prec(new linsolv_amgx<1>(device_num, 1, amgx_reuse_override));
  // Stage-2 experiment hook: DARTS_CPR_STAGE2=amgx swaps the exact
  // (latency-bound) block-ILU(0) for a second AMGX instance on the full
  // system; configure it via amgx_config_bs<NV>.json in the run directory.
  const char *stage2_env = std::getenv("DARTS_CPR_STAGE2");
  if (stage2_env && std::string(stage2_env) == std::string("amgx"))
    cpr->set_prec(new linsolv_amgx<NV>(device_num, 1, amgx_reuse_override));
  else if (stage2_env && std::string(stage2_env) == std::string("amgx_bs1"))
    // scalar-expanded full system: well-row diagonals become invertible
    // scalars, which D^-1-based smoothers (Jacobi/DILU) require
    cpr->set_prec(new linsolv_amgx<NV>(device_num, 1, amgx_reuse_override));
  else if (stage2_env && std::string(stage2_env) == std::string("mcsgs"))
    // opendarts multicolor symmetric block-Gauss-Seidel: latency-friendly
    // stage-2 with identity fallback on singular (well-row) diagonals
    cpr->set_prec(new linsolv_mcsgs<NV>());
  else
    cpr->set_prec(new linsolv_cusparse_ilu<NV>());
  if (use_bicgstab)
  {
    auto *bicgstab = new linsolv_bicgstab<NV>();
    bicgstab->set_prec(cpr);
    linear_solver_type_str = "GPU_BICGSTAB_CPR_AMGX_ILU";
    return bicgstab;
  }
  auto *gmres = new linsolv_gmres_gpu<NV>();
  gmres->set_prec(cpr);
  linear_solver_type_str = "GPU_GMRES_CPR_AMGX_ILU";
  return gmres;
}

/// Wrap the AMGX-CPR chain in the exact K-pair local (block-Schur) elimination
/// (linsolv_schur_elim<NV, KELIM>): the inner chain is built at the reduced
/// block size NV-KELIM. The eliminated (row, column) pairs are explicit.
template <uint8_t NV, uint8_t KELIM>
inline opendarts::linear_solvers::linsolv_iface *make_gpu_schur_elim_chain(
    int device_num, bool use_bicgstab, const std::vector<int> &erows,
    const std::vector<int> &ecols, std::string &tag)
{
  auto *wrap = new opendarts::linear_solvers::linsolv_schur_elim<NV, KELIM>(
      /*on_device=*/true, erows, ecols, 0.0);
  // The reduced pressure system's coefficients change every Newton/timestep
  // (condensation folds in the evolving cell-local dynamics), which invalidates a
  // reused AMGX hierarchy (measured: setup failures, wasted Newtons). Disable
  // adaptive hierarchy reuse for THIS chain's AMGX instances only -- other
  // AMGX instances in the process keep the default behaviour.
  wrap->set_prec(make_gpu_amgx_cpr_chain<NV - KELIM>(device_num, use_bicgstab, tag,
      /*amgx_reuse_override=*/0));
  wrap->set_inner_owned(true);  // engine deletes only the top-level solver
  tag += " + SCHUR_ELIM(K=" + std::to_string((int)KELIM) + ")";
  return wrap;
}
#endif // OPENDARTS_LINEAR_SOLVERS && OPENDARTS_GPU_HAS_AMGX

#endif

/// This class defines infrastructure for simulation
// The GPU engine has-a Jacobian (engine_base::Jacobian) and also IS-a
// csr_matrix_base: the matrix-free path (assembly_kernel == 13) passes the
// engine itself to linear_solver->setup() as the system matrix. The
// csr_matrix_base storage-accessor virtuals delegate to the owned Jacobian.
class engine_base_gpu : public engine_base, public csr_matrix_base
{
  // methods
public:
  engine_base_gpu() { ; };

  ~engine_base_gpu();

  // get the number of primary unknowns (per block)
  virtual uint8_t get_n_vars() const override = 0;

  // get the number of operators (per block) — must match widened base signature.
  virtual uint16_t get_n_ops() const override = 0;

  // get the number of components
  virtual uint8_t get_n_comps() const override = 0;

  // get the index of Z variable
  virtual uint8_t get_z_var_idx() const override = 0;

  // initialization
  virtual int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                   operator_set_gradient_evaluator_iface* thermal_var_etor_, sim_params *params, timer_node *timer_) override = 0;

  template <uint8_t N_VARS>
  int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                operator_set_gradient_evaluator_iface* thermal_var_etor_, sim_params *params, timer_node *timer_);
  int evaluate_operators_d();

  // newton loop
  virtual int assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS) override = 0;
  virtual int adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS) override = 0;

  void apply_global_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX) override;
  void apply_local_chop_correction(std::vector<value_t> &X, std::vector<value_t> &dX) override;

  int apply_newton_update(value_t dt) override;

  /** @defgroup Engine_methods
     *  Methods of base engine class exposed to Python
     *  @{
     */

  /// @brief report for one newton iteration
  virtual int assemble_linear_system(value_t deltat) override;
  virtual int solve_linear_equation() override;
  virtual int post_newtonloop(value_t deltat, value_t time) override;

  // Device-resident residual norms: the per-assembly host mirror of op_vals_arr
  // is gone, so the default (L2) norms reduce on the device. L1/Linf fall back
  // to the host implementation after an explicit refresh.
  virtual double calc_newton_residual_L2() override;
  virtual double calc_newton_residual_L1() override;
  virtual double calc_newton_residual_Linf() override;
  virtual void average_operator(std::vector<value_t> &av_op) override;
  /// refresh the host op_vals_arr mirror from the device (lazy consumers)
  void sync_op_vals_to_host();
  /// accepted-step hook from engine_base::post_newtonloop's converged branch
  void sync_host_data_for_accepted_step() override;

  virtual int test_assembly(int n_times, int kernel_number = 0, int dump_jacobian_rhs = 0) override;

  virtual int test_spmv(int n_times, int kernel_number = 0, int dump_result = 0) override;

  // calc r_d = Jacobian * v_d
  virtual int matrix_vector_product_d0(const value_t *v_d, value_t *r_d);

  // calc r_d += Jacobian * v_d
  virtual int matrix_vector_product_d(const value_t *v_d, value_t *r_d);

  // calc r_d = alpha * Jacobian * u_d + beta * v_d
  virtual int calc_lin_comb_d(value_t alpha, value_t beta, value_t *u_d, value_t *v_d, value_t *r_d);

  // dummy methods to support deriving from csr_matrix_base:
  virtual int matrix_vector_product(const double *v, double *r) { return 0; };
  virtual int calc_lin_comb(const double alpha, const double beta, double *u, double *v, double *r) { return 0; };
  virtual int matrix_vector_product_d_ell(const double *v, double *r) { return 0; };
  virtual int copy_struct_to_device() { return 0; };
  virtual int copy_values_to_device() { return 0; };
  virtual int write_matrix_to_file(const char *file_name, int sort_cols = 0) { return 0; };
  virtual int write_matrix_to_file_mm(const char *file_name) { return 0; };
  virtual int convert_to_ELL() { return 0; };
  virtual csr_matrix_base *get_csr_matrix() { return Jacobian; };

#ifdef OPENDARTS_LINEAR_SOLVERS
  // csr_matrix_base pure-virtual interface. The matrix-free GPU path passes
  // the engine itself as the system matrix, so the storage accessors simply
  // forward to the owned Jacobian.
  value_t *get_values() override { return Jacobian->get_values(); }
  index_t *get_rows_ptr() override { return Jacobian->get_rows_ptr(); }
  index_t *get_cols_ind() override { return Jacobian->get_cols_ind(); }
  index_t *get_diag_ind() override { return Jacobian->get_diag_ind(); }
  index_t *get_row_thread_starts() override { return Jacobian->get_row_thread_starts(); }
  int export_matrix_to_file(const std::string &filename,
      opendarts::linear_solvers::sparse_matrix_export_format export_format) override
  {
    return Jacobian->export_matrix_to_file(filename, export_format);
  }
  int import_matrix_from_file(const std::string &filename,
      opendarts::linear_solvers::sparse_matrix_import_format import_format) override
  {
    return Jacobian->import_matrix_from_file(filename, import_format);
  }
#ifdef WITH_GPU
  value_t *get_values_d() override { return Jacobian->get_values_d(); }
  index_t *get_rows_ptr_d() override { return Jacobian->get_rows_ptr_d(); }
  index_t *get_cols_ind_d() override { return Jacobian->get_cols_ind_d(); }
  index_t *get_diag_ind_d() override { return Jacobian->get_diag_ind_d(); }
#endif
#endif

  // Jacobian device/host pointer accessors -- bridge the open-source
  // csr_matrix_base device-pointer virtuals (OPENDARTS_LINEAR_SOLVERS) and
  // the legacy/bos csr_matrix members, so the *_gpu.cu kernels stay free of
  // #ifdef branching.
  value_t *jac_values_d()
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    return Jacobian->get_values_d();
#else
    return Jacobian->values_d;
#endif
  }
  index_t *jac_rows_ptr_d()
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    return Jacobian->get_rows_ptr_d();
#else
    return Jacobian->rows_ptr_d;
#endif
  }
  index_t *jac_cols_ind_d()
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    return Jacobian->get_cols_ind_d();
#else
    return Jacobian->cols_ind_d;
#endif
  }
  index_t *jac_diag_ind_d()
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    return Jacobian->get_diag_ind_d();
#else
    return Jacobian->diag_ind_d;
#endif
  }
  // Host structure / values -- the get_*() accessors are csr_matrix_base
  // virtuals available in both builds.
  index_t *jac_rows_ptr() { return Jacobian->get_rows_ptr(); }
  value_t *jac_values() { return Jacobian->get_values(); }

  // GPU-specific data (_d postfix means device data)
  // All device pointers are default-initialized to nullptr so the destructor
  // can safely free_device_data() even when init() didn't run (e.g. model
  // construction raised). cudaFree(nullptr) is documented as a no-op.

  // linear system
  value_t *X_d = nullptr, *Xn_d = nullptr, *dX_d = nullptr, *RHS_d = nullptr;      // [N_VARS * n_blocks] arrays for solution, previous timestep solution, update, and right hand side
  value_t *Xop_d = nullptr;                          // [(N_VARS + n_history) * n_blocks] extended OBL state for history-aware interpolation
  value_t *residual_scratch_d = nullptr;             // [2 * NORM_MAX_VARS] accumulator for device-side residual norms / operator averages
  std::vector<void *> pinned_host_ptrs;              // host buffers registered with cudaHostRegister (unpinned in the destructor)
  std::vector<value_t> jac_wells;                    // [n_wells * 2 * N_VARS * N_VARS ] temporary host storage for well equations
  value_t *jac_wells_d = nullptr;                    // [n_wells * 2 * N_VARS * N_VARS ] temporary device storage for well equations
  std::vector<index_t> jac_well_head_idxs;           // [n_wells] well head indexes in jacobian values array
  index_t *jac_well_head_idxs_d = nullptr;           // [n_wells] device storage for well head indexes in jacobian values array

  // interpolation
  value_t *op_vals_arr_d = nullptr;       // [N_OPS * n_blocks] array of values of operators
  value_t *op_ders_arr_d = nullptr;       // [N_OPS * N_VARS * n_blocks] array of dedrivatives of operators
  value_t *op_ders_arr_ext_d = nullptr;   // [N_OPS * (N_VARS + n_history) * n_blocks] extended derivative scratch
  value_t *op_vals_arr_n_d = nullptr;     // [N_OPS * n_blocks] array of values of operators from the last timestep

  std::vector<index_t *> block_idxs_d; // [N_OP_NUM][?] vector of arrays of block indexes corresponding to given operator set

  // input data
  value_t *RV_d = nullptr, *PV_d = nullptr;                // [n_blocks] rock and pore volumes for each block
  value_t *mesh_tran_d = nullptr, *mesh_tranD_d = nullptr; // [n_conns] transmissibility and diffusive transmissibility for each (duplicated) connection
  value_t *mesh_hcap_d = nullptr;                          // [n_blocks] rock heat capacity for each block

  value_t *molar_weights_d = nullptr;          // [n_regions * NC] molar weights of components for reconstruction of Darcy velocities
  value_t *darcy_velocities_d = nullptr;       // [n_res_blocks * NP * ND] array of phase Darcy velocities for every reservoir cell
  value_t *mesh_velocity_appr_d = nullptr;     // coefficients of approximation of Darcy phase velocities over fluxes
  index_t *mesh_velocity_offset_d = nullptr;   // offsets in the approximation of Darcy phase velocities over fluxes
  index_t *mesh_op_num_d = nullptr;            // regions indices for every cell
  value_t *dispersivity_d = nullptr;           // [n_regions * NP * NC] dispersivity coefficients stored in device memory
};

template <uint8_t N_VARS>
int engine_base_gpu::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                               std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                               operator_set_gradient_evaluator_iface* thermal_var_etor_,
                               sim_params *params_, timer_node *timer_)
{
  time_t rawtime;
  struct tm *timeinfo;
  char buffer[1024];

#ifdef _OPENMP
  // Mirror the CPU engine contract (engine_base::print_header): the
  // Jacobian's row_thread_starts partition is sized for
  // omp_get_max_threads() at allocate/init time, so dynamic team sizing must
  // be off BEFORE init_jacobian_structure's first-touch / assembly regions
  // run. The GPU engine reaches print_header() only after structure init,
  // hence the explicit early call here.
  omp_set_dynamic(0);
#endif

  mesh = mesh_;
  wells = well_list_;
  acc_flux_op_set_list = acc_flux_op_set_list_;
  thermal_var_etor = thermal_var_etor_;
  params = params_;
  timer = timer_;

  // Instantiate Jacobian. With the unified matrix layout (plan §12, phase B
  // for CPU; phase C1 for GPU), the open-source build uses the new
  // block_csr_matrix on GPU as well as on CPU -- its device storage is
  // allocated lazily through dual_array (no init_device call needed), and
  // the cuSPARSE BSR SpMV adapter (gpu_bsr_spmv) services
  // csr_matrix_base::matrix_vector_product_d on the device pointers exposed
  // through get_*_d(). The legacy csr_matrix<N> + init_device path is kept
  // for the proprietary build, which still ships its own GPU device layer.
  if (!Jacobian)
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    Jacobian = new block_csr_matrix;
#else
    Jacobian = new csr_matrix<N_VARS>;
    Jacobian->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
#endif
  }
  // for GPU engines we need only structure - rows_ptr and cols_ind
  // they are filled on CPU and later copied to GPU

  // may need full init to be able to dump csr matrix from device
#ifdef OPENDARTS_LINEAR_SOLVERS
  (static_cast<block_csr_matrix *>(Jacobian))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);
  Jacobian->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;  // init() resets type
#else
  (static_cast<csr_matrix<N_VARS> *>(Jacobian))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);
#endif

  int matrix_free = 0;
  if (params->assembly_kernel == 13)
  {
    // enable matrix-free mode for 13th assembly kernel
    matrix_free = 1;
  }

#ifndef OPENDARTS_LINEAR_SOLVERS
  // Legacy GPU path: pre-allocate the cuSPARSE-backed device buffers
  // (values_d, rows_ptr_d, cols_ind_d, diag_ind_d). The open-source
  // block_csr_matrix path allocates these lazily on first access via
  // dual_array::ensure_device_allocated().
  (static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_device(mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);
#endif
  // create linear solver
  // if default CPU solver is used, silently change to default GPU solver
  if (params->linear_type == 0)
  {
#ifdef OPENDARTS_GPU_HAS_AMGX
    params->linear_type = sim_params::GPU_GMRES_CPR_AMGX_ILU;
#else
    // AMGX not built; fall back to the CPR + AMG GPU solver.
    params->linear_type = sim_params::GPU_GMRES_CPR_AMG;
#endif
  }

#ifndef OPENDARTS_GPU_HAS_AMGX
  // AMGX not built into this (open-source) configuration: redirect any
  // explicitly requested AMGX-based GPU solver to the AMG-based CPR GPU
  // solver so the build stays runnable instead of aborting in the switch.
  switch (params->linear_type)
  {
  case sim_params::GPU_GMRES_CPR_AMGX_ILU:
  case sim_params::GPU_GMRES_CPR_AMGX_ILU_SP:
  case sim_params::GPU_GMRES_CPR_AMGX_AMGX:
  case sim_params::GPU_GMRES_AMGX:
  case sim_params::GPU_AMGX:
  case sim_params::GPU_BICGSTAB_CPR_AMGX:
    std::cout << "AMGX not available; using GPU_GMRES_CPR_AMG instead of linear solver type "
              << params->linear_type << std::endl;
    params->linear_type = sim_params::GPU_GMRES_CPR_AMG;
    break;
  default:
    break;
  }
#endif

  std::string linear_solver_type_str;
  if (!linear_solver)
  {
#ifdef OPENDARTS_LINEAR_SOLVERS
    // Open-source GPU build: the proprietary bos GMRES/CPR/AMG solvers are
    // stubbed out, so the full linear_type-driven factory below cannot run.
    // Honor the explicitly requested in-tree GPU solvers (direct solvers:
    // cuDSS when built, cuSOLVER QR always); otherwise fall back to the
    // open-source GPU BiCGStab Krylov solver with a cuSPARSE block-ILU(0)
    // preconditioner. AMGX-based linear_type values were already redirected
    // above when AMGX is absent.
    if (params->linear_type == sim_params::GPU_CUDSS)
    {
#ifdef WITH_CUDSS
      linear_solver = new linsolv_cudss<N_VARS>();
      linear_solver_type_str = "GPU_CUDSS";
#else
      std::cout << "cuDSS not built (WITH_CUDSS=OFF); using the BiCGStab + "
                   "cuSPARSE-ILU(0) GPU solver instead." << std::endl;
#endif
    }
    else if (params->linear_type == sim_params::GPU_CUSOLVER)
    {
      linear_solver = new linsolv_cusolv<N_VARS>();
      linear_solver_type_str = "GPU_CUSOLVER";
    }
#ifdef OPENDARTS_GPU_HAS_AMGX
    else if (params->linear_type == sim_params::GPU_BICGSTAB_CPR_AMGX
             || params->linear_type == sim_params::GPU_GMRES_CPR_AMGX_ILU)
    {
      // In-tree AMGX-CPR stack on the open-source block_csr_matrix Jacobian:
      // GPU-resident GMRES (linsolv_gmres_gpu) around the two-stage CPR
      // (linsolv_bos_cpr_gpu: True-IMPES pressure reduction on device, AMGX
      // AMG on the scalar pressure system, cuSPARSE block-ILU(0) on the full
      // system). Mirrors the proprietary GPU_GMRES_CPR_AMGX_ILU wiring.
      if constexpr (N_VARS > 1)
      {
        const bool use_bicgstab = (params->linear_type == sim_params::GPU_BICGSTAB_CPR_AMGX);
        if (params->schur_elim_count > 0)
        {
          // Mineral-equation Schur elimination: exact per-cell condensation of K
          // cell-local (diagonal-block-only) equations, with the SAME AMGX-CPR chain built at the
          // reduced block size (N_VARS-K) as the inner solver. The reduced block
          // must stay >= 2 for the CPR split, so K <= N_VARS-2.
          const int Kelim = params->schur_elim_count;
          const std::vector<int> &er = params->schur_elim_rows;
          const std::vector<int> &ec = params->schur_elim_cols;
          if ((int)er.size() != Kelim || (int)ec.size() != Kelim)
            throw std::runtime_error("schur_elim: schur_elim_rows/cols length must equal "
                "schur_elim_count (K)");
          linear_solver = nullptr;
          if constexpr (N_VARS >= 3)
          {
            switch (Kelim)
            {
              case 1: if constexpr (N_VARS >= 3) linear_solver = make_gpu_schur_elim_chain<N_VARS, 1>(device_num, use_bicgstab, er, ec, linear_solver_type_str); break;
              case 2: if constexpr (N_VARS >= 4) linear_solver = make_gpu_schur_elim_chain<N_VARS, 2>(device_num, use_bicgstab, er, ec, linear_solver_type_str); break;
              case 3: if constexpr (N_VARS >= 5) linear_solver = make_gpu_schur_elim_chain<N_VARS, 3>(device_num, use_bicgstab, er, ec, linear_solver_type_str); break;
              case 4: if constexpr (N_VARS >= 6) linear_solver = make_gpu_schur_elim_chain<N_VARS, 4>(device_num, use_bicgstab, er, ec, linear_solver_type_str); break;
              default: break;
            }
          }
          if (!linear_solver)
            // An EXPLICIT solver request must fail rather than silently running a
            // different algorithm than the user configured.
            throw std::runtime_error("schur_elim_count=" + std::to_string(Kelim) +
                " unsupported for block size " + std::to_string((int)N_VARS) +
                " (need 1 <= K <= N_VARS-2, K <= 4); disable local (Schur) elimination "
                "or adjust K");
        }
        else
        {
          linear_solver = make_gpu_amgx_cpr_chain<N_VARS>(device_num, use_bicgstab,
              linear_solver_type_str);
        }
      }
      else
      {
        auto *gmres = new linsolv_gmres_gpu<1>();
        gmres->set_prec(new linsolv_amgx<1>(device_num));
        linear_solver = gmres;
        linear_solver_type_str = "GPU_GMRES_AMGX";
      }
    }
#endif // OPENDARTS_GPU_HAS_AMGX
    if (!linear_solver)
    {
      linsolv_bicgstab<N_VARS> *bicgstab = new linsolv_bicgstab<N_VARS>();
      bicgstab->set_prec(new linsolv_cusparse_ilu<N_VARS>());
      linear_solver = bicgstab;
      linear_solver_type_str = "GPU_BICGSTAB_CUSPARSE_ILU";
    }
#else
    switch (params->linear_type)
    {
    case sim_params::GPU_GMRES_CPR_AMG:
    {
      linear_solver = new linsolv_bos_gmres<N_VARS>(1);
      if constexpr (N_VARS > 1)
      {
        linsolv_iface* cpr = new linsolv_bos_cpr_gpu<N_VARS>;
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
	  linear_solver_type_str = "GPU_GMRES_CPR_AIPS";
      break;
    }
#endif //WITH_AIPS
#ifdef OPENDARTS_GPU_HAS_AMGX
    case sim_params::GPU_GMRES_CPR_AMGX_ILU:
    {
      linear_solver = new linsolv_bos_gmres<N_VARS>(1);
      if constexpr (N_VARS > 1)
      {
        linsolv_iface* cpr = new linsolv_bos_cpr_gpu<N_VARS>;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 1;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 1;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 0;

        // set p system prec
        cpr->set_p_system_prec(new linsolv_amgx<1>(device_num));
        // set full system prec
        cpr->set_prec(new linsolv_cusparse_ilu<N_VARS>(matrix_free, 0));
        linear_solver->set_prec(cpr);
        linear_solver_type_str = "GPU_GMRES_CPR_AMGX_ILU";
      }
      else
      {
        linear_solver->set_prec(new linsolv_amgx<1>(device_num));
        linear_solver_type_str = "GPU_GMRES_AMGX";
      }

      break;
    }
    case sim_params::GPU_GMRES_CPR_AMGX_ILU_SP:
    {
      linear_solver = new linsolv_bos_gmres<N_VARS>(1);
      if constexpr (N_VARS > 1)
      {
        linsolv_iface* cpr = new linsolv_bos_cpr_gpu<N_VARS>;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 1;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 1;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 0;

        // set p system prec
        cpr->set_p_system_prec(new linsolv_amgx<1>(device_num));
        // set full system prec
        cpr->set_prec(new linsolv_cusparse_ilu<N_VARS>(matrix_free, 1));
        linear_solver->set_prec(cpr);
        linear_solver_type_str = "GPU_GMRES_CPR_AMGX_ILU_SP";
      }
      else
      {
        linear_solver->set_prec(new linsolv_amgx<1>(device_num));
        linear_solver_type_str = "GPU_GMRES_AMGX_SP";
      }

      break;
    }
    case sim_params::GPU_GMRES_CPR_AMGX_AMGX:
    {
      linear_solver = new linsolv_bos_gmres<N_VARS>(1);
      if constexpr (N_VARS > 1)
      {
        linsolv_iface* cpr = new linsolv_bos_cpr_gpu<N_VARS>;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 1;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 1;
        ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 0;

        int convert_to_bs1 = 0;
        if (params->linear_params.size() > 0)
        {
          convert_to_bs1 = params->linear_params[0];
        }

        // set p system prec
        cpr->set_p_system_prec(new linsolv_amgx<1>(device_num));
        // set full system prec
        cpr->set_prec(new linsolv_amgx<N_VARS>(device_num, convert_to_bs1));
        linear_solver->set_prec(cpr);
        linear_solver_type_str = "GPU_GMRES_CPR_AMGX_AMGX";
      }
      else
      {
        linear_solver->set_prec(new linsolv_amgx<1>(device_num));
        linear_solver_type_str = "GPU_GMRES_AMGX";
      }

      break;
    }
    case sim_params::GPU_GMRES_AMGX:
    {
      int convert_to_bs1 = 0;
      if (params->linear_params.size() > 0)
      {
        convert_to_bs1 = params->linear_params[0];
      }
      linear_solver = new linsolv_bos_gmres<N_VARS>(1);
      linear_solver->set_prec(new linsolv_amgx<N_VARS>(device_num, convert_to_bs1));
	  linear_solver_type_str = "GPU_GMRES_AMGX";
      break;
    }
    case sim_params::GPU_AMGX:
    {
      int convert_to_bs1 = 0;
      if (params->linear_params.size() > 0)
      {
        convert_to_bs1 = params->linear_params[0];
      }
      linear_solver = new linsolv_amgx<N_VARS>(device_num, convert_to_bs1);
	  linear_solver_type_str = "GPU_AMGX";
      break;
    }
#endif // OPENDARTS_GPU_HAS_AMGX
    case sim_params::GPU_GMRES_ILU0:
    {
      linear_solver = new linsolv_bos_gmres<N_VARS>(1);
	  linear_solver_type_str = "GPU_GMRES_ILU0";
      break;
    }
#ifdef OPENDARTS_GPU_HAS_AMGX
    case sim_params::GPU_BICGSTAB_CPR_AMGX:
    {
      linear_solver = new linsolv_bicgstab<N_VARS>();
      linsolv_iface *cpr = new linsolv_bos_cpr_gpu<N_VARS>;
      ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_setup_gpu = 1;
      ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_solve_gpu = 1;
      ((linsolv_bos_cpr_gpu<N_VARS> *)cpr)->p_solver_requires_diag_first = 0;

      cpr->set_prec(new linsolv_amgx<1>(device_num));
      linear_solver->set_prec(cpr);
	  linear_solver_type_str = "GPU_BICGSTAB_CPR_AMGX";
      break;
    }
#endif // OPENDARTS_GPU_HAS_AMGX
    default:
    {
      throw std::runtime_error("Linear solver type " +
          std::to_string(static_cast<int>(params->linear_type)) +
          " is not supported for " + engine_name);
    }
    }
#endif // OPENDARTS_LINEAR_SOLVERS
  }

  std::cout << "Linear solver type is " << params->linear_type << std::endl;

  // *** allocate host data ***

  n_vars = get_n_vars();
  n_ops = get_n_ops();
  nc = get_n_comps();
  z_var_idx = get_z_var_idx();

  // Physical-simplex clipping; OBL window no longer constrains Newton — see engine_base.h
  min_axis_z = 0.0;
  max_axis_z = 1.0;
  min_sim_z = min_axis_z + params->sim_eps;
  max_sim_z = max_axis_z - params->sim_eps;

  X.resize(n_vars * mesh->n_blocks);
  Xn.resize(n_vars * mesh->n_blocks);
  X_init.resize(n_vars * mesh->n_res_blocks);  // initialize only reservoir blocks with mesh->initial_state array
  RHS.resize(n_vars * mesh->n_blocks);
  dX.resize(n_vars * mesh->n_blocks);

  PV.resize(mesh->n_blocks);
  RV.resize(mesh->n_blocks);

  old_z.resize(nc);
  new_z.resize(nc);
  FIPS.resize(nc);
  old_z_fl.resize(nc - n_solid);
	new_z_fl.resize(nc - n_solid);

  op_vals_arr.resize(n_ops * mesh->n_blocks);
  op_ders_arr.resize(n_ops * n_vars * mesh->n_blocks);

  ensure_history_buffers(mesh->n_blocks + mesh->n_bounds, n_ops);

  jac_wells.resize(2 * n_vars * n_vars * wells.size());
  jac_well_head_idxs.resize(wells.size());

  // *** allocate device data ***

  allocate_device_data(X, &X_d);
  allocate_device_data(Xn, &Xn_d);
  allocate_device_data(Xn, &dX_d);
  allocate_device_data(RHS, &RHS_d);

  allocate_device_data(PV, &PV_d);
  allocate_device_data(mesh->tran, &mesh_tran_d);
  allocate_device_data(jac_wells, &jac_wells_d);
  allocate_device_data(jac_well_head_idxs, &jac_well_head_idxs_d);

  allocate_device_data(op_vals_arr, &op_vals_arr_d);
  allocate_device_data(op_vals_arr, &op_vals_arr_n_d);
  // previous-timestep operator values live on the device; skip the host mirror
  keep_host_op_vals_n_mirror = false;

  // Pin the per-Newton host transfer buffers: pageable copies run ~9 GB/s on
  // this host class vs ~26 GB/s pinned. Registration is best-effort.
  auto pin_host_buffer = [this](std::vector<value_t> &v)
  {
    if (!v.empty() && cudaHostRegister(v.data(), v.size() * sizeof(value_t), cudaHostRegisterDefault) == cudaSuccess)
      pinned_host_ptrs.push_back(v.data());
  };
  pin_host_buffer(X);
  pin_host_buffer(dX);
  pin_host_buffer(RHS);
  pin_host_buffer(op_vals_arr);
  allocate_device_data(&op_ders_arr_d, n_ops * n_vars * mesh->n_blocks);
  if (get_n_history() > 0)
  {
    allocate_device_data(Xop, &Xop_d);
    allocate_device_data(op_ders_arr_ext, &op_ders_arr_ext_d);
  }

  // *** initialize host data ***
  X_init = mesh->initial_state;
  this->apply_composition_correction(X_init);  // apply composition correction for initial state

  X_init.resize(n_vars * mesh->n_blocks);
  for (index_t i = 0; i < mesh->n_blocks; i++)
  {
    PV[i] = mesh->volume[i] * mesh->poro[i];
    RV[i] = mesh->volume[i] * (1 - mesh->poro[i]);
  }

  t = 0;

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  stat = sim_stat();

  // initialize jacobian structure
  init_jacobian_structure(Jacobian);

  // for matrix-free, support csr_matrix_base parameters
  is_square = 1;
  n_rows = Jacobian->n_rows;

#ifdef WITH_GPU
#ifdef OPENDARTS_LINEAR_SOLVERS
  // Open-source GPU engine always solves on device -> always mirror the
  // block-CSR structure to the device.
  if (true)
#else
  if (params->linear_type >= sim_params::GPU_GMRES_CPR_AMG)
#endif
  {
    timer->node["jacobian assembly"].node["send_to_device"].start();
    Jacobian->copy_struct_to_device();
    timer->node["jacobian assembly"].node["send_to_device"].stop();
  }
#endif

  linear_solver->init_timer_nodes(&timer->node["linear solver setup"], &timer->node["linear solver solve"]);
  // initialize linear solver
  linear_solver->init(Jacobian, params->max_i_linear, params->tolerance_linear);

  // let wells initialize their state
  int iw = 0;
  for (ms_well *w : wells)
  {
    w->initialize_control_epm(X_init);
    jac_well_head_idxs[iw++] = w->well_head_idx;
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
    // op_axis_min/op_axis_max left empty — disables apply_obl_axis_local_correction
    block_idxs[r].clear();
  }

  // create a block list for every operator set

  // scanning through all blocks, fill the corresponding list with the index of the block
  index_t idx = 0;
  for (auto op_region : mesh->op_num)
  {
    block_idxs[op_region].emplace_back(idx++);
  }

  op_vals_arr_n = op_vals_arr;

  time_data.clear();
  time_data_report.clear();

  // *** initialize device data ***
  copy_data_to_device(X, X_d);
  copy_data_within_device(Xn_d, X_d, X.size());

  // block_idxs have first been initialized at host, now we can allocate&initialize device data
  block_idxs_d.resize(block_idxs.size());
  for (int op_region = 0; op_region < block_idxs.size(); op_region++)
  {
    allocate_device_data(block_idxs[op_region], &block_idxs_d[op_region]);
    copy_data_to_device(block_idxs[op_region], block_idxs_d[op_region]);
  }

  // interpolate initial values
  evaluate_operators_d();
  copy_data_within_device(op_vals_arr_n_d, op_vals_arr_d, op_vals_arr.size());

  copy_data_to_device(PV, PV_d);
  copy_data_to_device(mesh->tran, mesh_tran_d);
  copy_data_to_device(jac_well_head_idxs, jac_well_head_idxs_d);

  // Adjoint gradients: the backward driver and all its matrices are
  // host-resident -- reuse the engine_base allocation path (same blocks that
  // engine_base::init_base runs on the CPU engines).
  if (opt_history_matching)
  {
    init_adjoint_base();
  }

  // Customized operators are evaluated host-side too (post_newtonloop / the
  // adjoint driver via customize_block_idxs).
  if (customize_operator)
  {
    init_customized_operator_base();
  }

  well_control_arr.clear();

  print_header();

  sprintf(buffer, "\nSTART SIMULATION\n-------------------------------------------------------------------------------------------------------------\n");
  std::cout << buffer << std::flush;

  return 0;
}

#endif
