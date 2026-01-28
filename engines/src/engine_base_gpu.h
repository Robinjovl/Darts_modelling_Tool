#ifndef ENGINE_BASE_GPU_H
#define ENGINE_BASE_GPU_H

#include <vector>
#include <unordered_map>
#include <cmath>


#include "engine_base.h"
#include "csr_matrix.h"
#include "gpu_tools.h"
#ifdef WITH_GPU
#include "linsolv_bicgstab.h"
#define KERNEL_BLOCK_SIZE 128

#endif

/// This class defines infrastructure for simulation
class engine_base_gpu : public engine_base, public csr_matrix_base
{
  // methods
public:
  engine_base_gpu() { ; };

  ~engine_base_gpu();

  // get the number of primary unknowns (per block)
  virtual uint8_t get_n_vars() const override = 0;

  // get the number of operators (per block)
  virtual uint8_t get_n_ops() const override = 0;

  // get the number of components
  virtual uint8_t get_n_comps() const override = 0;

  // get the index of Z variable
  virtual uint8_t get_z_var() const override = 0;

  // initialization
  virtual int init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_, sim_params *params, timer_node *timer_) override = 0;

  template <uint8_t N_VARS>
  int init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_, std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_, sim_params *params, timer_node *timer_);

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

  // GPU-specific data (_d postfix means device data)

  // linear system
  value_t *X_d, *Xn_d, *dX_d, *RHS_d;      // [N_VARS * n_blocks] arrays for solution, previous timestep solution, update, and right hand side
  value_t *RHS_wells_d;                    // [N_VARS * n_blocks] temporary device storage for RHS_wells copied async from host while main assembly is done
  std::vector<value_t> jac_wells;          // [n_wells * 2 * N_VARS * N_VARS ] temporary host storage for well equations
  value_t *jac_wells_d;                    // [n_wells * 2 * N_VARS * N_VARS ] temporary device storage for well equations
  std::vector<index_t> jac_well_head_idxs; // [n_wells] well head indexes in jacobian values array
  index_t *jac_well_head_idxs_d;           // [n_wells] device storage for well head indexes in jacobian values array

  // interpolation
  value_t *op_vals_arr_d;   // [N_OPS * n_blocks] array of values of operators
  value_t *op_ders_arr_d;   // [N_OPS * N_VARS * n_blocks] array of dedrivatives of operators
  value_t *op_vals_arr_n_d; // [N_OPS * n_blocks] array of values of operators from the last timestep

  std::vector<index_t *> block_idxs_d; // [N_OP_NUM][?] vector of arrays of block indexes corresponding to given operator set

  // input data
  value_t *RV_d, *PV_d;                // [n_blocks] rock and pore volumes for each block
  value_t *mesh_tran_d, *mesh_tranD_d; // [n_conns] transmissibility and diffusive transmissibility for each (duplicated) connection
  value_t *mesh_hcap_d;                // [n_blocks] rock heat capacity for each block

  value_t *molar_weights_d;            // [n_regions * NC] molar weights of components for reconstruction of Darcy velocities
  value_t *darcy_velocities_d;         // [n_res_blocks * NP * ND] array of phase Darcy velocities for every reservoir cell
  value_t *mesh_velocity_appr_d;       // coefficients of approximation of Darcy phase velocities over fluxes
  index_t *mesh_velocity_offset_d;     // offsets in the approximation of Darcy phase velocities over fluxes
  index_t *mesh_op_num_d;              // regions indices for every cell 
  value_t *dispersivity_d;             // [n_regions * NP * NC] dispersivity coefficients stored in device memory 
};

template <uint8_t N_VARS>
int engine_base_gpu::init_base(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
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
  // for GPU engines we need only structure - rows_ptr and cols_ind
  // they are filled on CPU and later copied to GPU
  //(static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_struct(mesh_->n_blocks, mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);

  // may need full init to be able to dump csr matrix from device
  (static_cast<csr_matrix<N_VARS> *>(Jacobian))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);

  int matrix_free = 0;
  if (params->assembly_kernel == 13)
  {
    // enable matrix-free mode for 13th assembly kernel
    matrix_free = 1;
  }

  (static_cast<csr_matrix<N_VARS> *>(Jacobian))->init_device(mesh_->n_blocks, mesh_->n_conns + mesh_->n_blocks);
  // create linear solver
  // if default CPU solver is used, silently change to default GPU solver
  if (params->linear_type == 0)
  {
    params->linear_type = sim_params::GPU_GMRES_CPR_AMGX_ILU;
  }
  
  std::string linear_solver_type_str;	
  if (!linear_solver)
  {
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
    default:
    {
      std::cerr << "Linear solver type " << params->linear_type << " is not supported for " << engine_name << std::endl << std::flush;
      exit(1);
    }
    }
  }
  
  std::cout << "Linear solver type is " << params->linear_type << std::endl;
	
  // *** allocate host data ***

  n_vars = get_n_vars();
  n_ops = get_n_ops();
  nc = get_n_comps();
  z_var = get_z_var();

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
  //op_ders_arr.resize(n_ops * n_vars * mesh->n_blocks);

  jac_wells.resize(2 * n_vars * n_vars * wells.size());
  jac_well_head_idxs.resize(wells.size());

  // *** allocate device data ***

  allocate_device_data(X, &X_d);
  allocate_device_data(Xn, &Xn_d);
  allocate_device_data(Xn, &dX_d);
  allocate_device_data(RHS, &RHS_d);
  allocate_device_data(RHS, &RHS_wells_d);

  allocate_device_data(PV, &PV_d);
  allocate_device_data(mesh->tran, &mesh_tran_d);
  allocate_device_data(jac_wells, &jac_wells_d);
  allocate_device_data(jac_well_head_idxs, &jac_well_head_idxs_d);

  allocate_device_data(op_vals_arr, &op_vals_arr_d);
  allocate_device_data(op_vals_arr, &op_vals_arr_n_d);
  allocate_device_data(&op_ders_arr_d, n_ops * n_vars * mesh->n_blocks);

  // *** initialize host data ***
  X_init = mesh->initial_state;
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

  // let wells initialize their state
  int iw = 0;
  for (ms_well *w : wells)
  {
    w->initialize_control(X_init);
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

  // scanning through all blocks, fill the corresponding list with the index of the block
  index_t idx = 0;
  for (auto op_region : mesh->op_num)
  {
    block_idxs[op_region].emplace_back(idx++);
  }

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
    max_zc = exp(acc_flux_op_set_list[0]->get_axis_max(z_var));                       //log based composition
  }

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
  for (int r = 0; r < acc_flux_op_set_list.size(); r++)
    acc_flux_op_set_list[r]->evaluate_with_derivatives_d(block_idxs[r].size(), X_d, block_idxs_d[r], op_vals_arr_d, op_ders_arr_d);
  copy_data_within_device(op_vals_arr_n_d, op_vals_arr_d, op_vals_arr.size());

  copy_data_to_device(PV, PV_d);
  copy_data_to_device(mesh->tran, mesh_tran_d);
  copy_data_to_device(jac_well_head_idxs, jac_well_head_idxs_d);

  print_header();

  sprintf(buffer, "\nSTART SIMULATION\n-------------------------------------------------------------------------------------------------------------\n");
  std::cout << buffer << std::flush;

  return 0;
}

#endif
