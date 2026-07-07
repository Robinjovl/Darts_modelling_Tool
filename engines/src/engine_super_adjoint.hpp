#ifndef ENGINE_SUPER_ADJOINT_HPP
#define ENGINE_SUPER_ADJOINT_HPP

#include <vector>
#include <cstring>

/// Shared host-side adjoint assembly for the super engines.
///
/// The bodies below were moved verbatim from
/// engine_super_cpu::adjoint_gradient_assembly so that engine_super_gpu can
/// reuse them without copy-paste divergence: every data member they touch
/// lives on engine_base (host side), and both engines define an identical
/// operator layout (same static constants).
///
/// The routine is split into two stages so the GPU engine can substitute a
/// device kernel for the expensive per-cell/per-connection assembly while
/// still sharing the format-conversion tail:
///   * super_engine_adjoint_assembly_host_loops -- fills the block matrix
///     dg_dx_n_temp (the diagonal blocks of dg/dx^n) and the scalar values of
///     dg_dT_general (dg/dT). This is the part the GPU device kernel mirrors.
///   * super_engine_adjoint_finalize -- rebuilds well_head_tran_idx_collection
///     (structure-only), zeroes the well-head rows of dg_dx_n_temp, and
///     expands the assembled block values into the scalar adjoint matrices the
///     host driver consumes (dg_dx_n always; dg_dx_T only for the legacy
///     scalar-transpose adjoint solvers). Shared by the host and the device
///     assembly paths.
/// super_engine_adjoint_assembly runs both in sequence (host assembly path).
///
/// Include this header from the engine .tpp after the engine class and matrix
/// headers -- it relies on csr_matrix / conn_mesh / ms_well being visible.
template <typename Engine>
int super_engine_adjoint_assembly_host_loops(Engine &e, value_t dt, std::vector<value_t> &X)
{
  // class constants -- identical layout on engine_super_cpu / engine_super_gpu
  constexpr uint8_t NC = Engine::NC_;
  constexpr uint8_t NP = Engine::NP_;
  constexpr bool THERMAL = Engine::N_VARS > Engine::NC_;
  constexpr uint8_t NE = Engine::NE;
  constexpr uint8_t N_VARS = Engine::N_VARS;
  constexpr uint16_t N_VARS_SQ = Engine::N_VARS_SQ;
  constexpr uint8_t P_VAR = Engine::P_VAR;
  constexpr uint8_t N_OPS = Engine::N_OPS;
  constexpr uint8_t ACC_OP = Engine::ACC_OP;
  constexpr uint8_t FLUX_OP = Engine::FLUX_OP;
  constexpr uint8_t DENS_OP = Engine::DENS_OP;
  constexpr uint8_t UPSAT_OP = Engine::UPSAT_OP;
  constexpr uint8_t GRAD_OP = Engine::GRAD_OP;
  constexpr uint8_t KIN_OP = Engine::KIN_OP;
  constexpr uint8_t GRAV_OP = Engine::GRAV_OP;
  constexpr uint8_t PC_OP = Engine::PC_OP;
  constexpr uint8_t MULT_OP = Engine::MULT_OP;
  constexpr uint8_t LAMBDA_OP = Engine::LAMBDA_OP;
  constexpr uint8_t TEMP_OP = Engine::TEMP_OP;

  // engine state -- all engine_base members; the body below is shared verbatim
  auto &mesh = e.mesh;
  auto &Jacobian = e.Jacobian;
  auto &dg_dx_n_temp = e.dg_dx_n_temp;
  auto &dg_dT_general = e.dg_dT_general;
  auto &PV = e.PV;
  auto &RV = e.RV;
  auto &op_vals_arr = e.op_vals_arr;
  auto &op_ders_arr = e.op_ders_arr;
  auto &params = e.params;
  auto &CFL_max = e.CFL_max;

  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  std::vector<value_t> &tran = mesh->tran;
  std::vector<value_t> &tranD = mesh->tranD;
  std::vector<value_t> &kin_fac = mesh->kin_factor; // default value of 1
  std::vector<value_t> &grav_coef = mesh->grav_coef;
  std::vector <index_t>& conn_index_to_one_way = mesh->conn_index_to_one_way;


  index_t* diag_ind = Jacobian->get_diag_ind();
  index_t* rows = Jacobian->get_rows_ptr();
  index_t* cols = Jacobian->get_cols_ind();

  value_t* Jac_n = dg_dx_n_temp->get_values();
  value_t* value_dg_dT = dg_dT_general->get_values();

  CFL_max = 0;

  index_t start = 0;
  index_t end = n_blocks;

  index_t j, diag_idx, jac_idx;
  value_t p_diff, t_diff, gamma_t_diff, mult_i, mult_j;
  value_t phase_presence_mult = 0.0;

  memset(Jac_n, 0, (n_conns + n_blocks) * N_VARS_SQ * sizeof(value_t));
  memset(value_dg_dT, 0, n_conns * N_VARS * sizeof(value_t));


  double value_g_u = 0.0;
  index_t N_element = 0;
  index_t count = 0;

  index_t k_count = 0;
  index_t idx;
  std::vector<index_t> temp_conn_one_way;
  std::vector<index_t> temp_num;


  for (index_t i = start; i < end; ++i)
  { // loop over grid blocks

    // index of diagonal block entry for block i in CSR values array
    diag_idx = N_VARS_SQ * diag_ind[i];

    // [1] fill diagonal part for both mass (and energy equations if needed, only fluid energy is involved here)
    for (uint8_t c = 0; c < NE; c++)
    {
      for (uint8_t v = 0; v < N_VARS; v++)
      {
        Jac_n[diag_idx + c * N_VARS + v] = -(PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v]); // der of accumulation term

        // Include derivatives for reaction term if part of reservoir cells:
        if (i < mesh->n_res_blocks)
        {
          Jac_n[diag_idx + c * N_VARS + v] -= ((PV[i] + RV[i]) * dt * op_ders_arr[(i * N_OPS + KIN_OP + c) * N_VARS + v] * kin_fac[i]); // derivative kinetics
        }
      }
    }

    // index of first entry for block i in CSR cols array
    index_t csr_idx_start = rows[i];
    // index of last entry for block i in CSR cols array
    index_t csr_idx_end = rows[i + 1];
    // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
    index_t conn_idx = csr_idx_start - i;

    jac_idx = N_VARS_SQ * csr_idx_start;

    // number of blocks between the last diagonal block and the current diagonal block in CSR array
    N_element = rows[i + 1] - rows[i] - 1;
    temp_conn_one_way.clear();
    temp_num.clear();
    for (index_t m = 0; m < N_element; m++)
    {
      temp_conn_one_way.push_back(conn_index_to_one_way[conn_idx + m]);
      temp_num.push_back(0);
    }

    for (index_t m = 0; m < N_element; m++)
    {
      for (index_t com : temp_conn_one_way)
      {
        if (com < temp_conn_one_way[m])
          temp_num[m] += 1;
      }
    }

    k_count = 0;

    for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS_SQ)
    { // fill offdiagonal part + contribute to diagonal

      j = cols[csr_idx];
      // skip diagonal
      if (i == j)
        continue;

      value_t trans_mult = 1;
      value_t trans_mult_der_i[N_VARS];
      value_t trans_mult_der_j[N_VARS];
      if (params->enable_permporo && i < mesh->n_res_blocks && j < mesh->n_res_blocks)
      {
        // Calculate transmissibility multiplier:
        mult_i = op_vals_arr[i * N_OPS + MULT_OP];
        mult_j = op_vals_arr[j * N_OPS + MULT_OP];

        // Take average interface porosity:
        trans_mult = 2 * mult_i * mult_j / (mult_i + mult_j);
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          trans_mult_der_i[v] = mult_j * trans_mult / (mult_i + mult_j) * op_ders_arr[(i * N_OPS + MULT_OP) * N_VARS + v];
          trans_mult_der_j[v] = mult_i * trans_mult / (mult_i + mult_j) * op_ders_arr[(j * N_OPS + MULT_OP) * N_VARS + v];
        }
      }
      else
      {
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          trans_mult_der_i[v] = 0;
          trans_mult_der_j[v] = 0;
        }
      }

      p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];


      // [2] fill offdiagonal part + contribute to diagonal, only fluid part is considered in energy equation
      for (uint8_t p = 0; p < NP; p++)
      { // loop over number of phases for convective operator

        // calculate gravity term for phase p
        value_t avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] + op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;

        // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
        value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

        // calculate partial derivatives for gravity and capillary terms
        value_t grav_pc_der_i[N_VARS];
        value_t grav_pc_der_j[N_VARS];
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          grav_pc_der_i[v] = -(op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + PC_OP + p) * N_VARS + v];
          grav_pc_der_j[v] = -(op_ders_arr[(j * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + PC_OP + p) * N_VARS + v];
        }

        double phase_gamma_p_diff = trans_mult * tran[conn_idx] * dt * phase_p_diff;
        (void)phase_gamma_p_diff;

        if (phase_p_diff < 0)
        {
          // mass and energy outflow with effect of gravity and capillarity
          for (uint8_t c = 0; c < NE; c++)
          {
            value_g_u = phase_p_diff * trans_mult * dt * op_vals_arr[i * N_OPS + LAMBDA_OP + p] * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c];
            idx = count + c * N_element + temp_num[k_count];
            value_dg_dT[idx] -= value_g_u;
          }
        }
        else
        {
          // mass and energy inflow with effect of gravity and capillarity
          for (uint8_t c = 0; c < NE; c++)
          {
            value_g_u = phase_p_diff * trans_mult * dt * op_vals_arr[j * N_OPS + LAMBDA_OP + p] * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c];
            idx = count + c * N_element + temp_num[k_count];
            value_dg_dT[idx] -= value_g_u;
          }
        }

      } // end of loop over number of phases for convective operator with gravity and capillarity

      // [3] Additional diffusion code here:   (phi_p * S_p) * (rho_p * D_cp * Delta_x_cp)  or (phi_p * S_p) * (kappa_p * Delta_T)
      // Only if block connection is between reservoir and reservoir cells!
      if (i < mesh->n_res_blocks && j < mesh->n_res_blocks)
      {
        // Add diffusion term to the residual:
        for (uint8_t c = 0; c < NE; c++)
        {
          for (uint8_t p = 0; p < NP; p++)
          {
            value_t grad_con = op_vals_arr[j * N_OPS + GRAD_OP + c * NP + p] - op_vals_arr[i * N_OPS + GRAD_OP + c * NP + p];

            // Determine phase presence multiplier as in the forward assembly
            if (op_vals_arr[i * N_OPS + UPSAT_OP + p] * op_vals_arr[j * N_OPS + UPSAT_OP + p] > params->phase_existence_tolerance)
              phase_presence_mult = 1.0;
            else
              phase_presence_mult = 0.0;


            if (c < NC) // mass
              value_g_u = grad_con * dt * phase_presence_mult * (mesh->poro[i] * op_vals_arr[i * N_OPS + DENS_OP + p] * op_vals_arr[i * N_OPS + UPSAT_OP + p] +
                                                                 mesh->poro[j] * op_vals_arr[j * N_OPS + DENS_OP + p] * op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;
            else // energy
              value_g_u = grad_con * dt * phase_presence_mult * (mesh->poro[i] * op_vals_arr[i * N_OPS + UPSAT_OP + p] +
                                                                 mesh->poro[j] * op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;

            idx = count + c * N_element + temp_num[k_count];
            value_dg_dT[idx] -= value_g_u;
          }
        }
      }

      // [4] add rock conduction
      if (THERMAL)
      {
        t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
        gamma_t_diff = tranD[conn_idx] * dt * t_diff;
        (void)gamma_t_diff;

        value_g_u = dt * t_diff * ((1 - mesh->poro[i]) * mesh->rock_cond[i] +
                                   (1 - mesh->poro[j]) * mesh->rock_cond[j]) / 2;
        idx = count + NC * N_element + temp_num[k_count];
        value_dg_dT[idx] -= value_g_u;
      }

      k_count++;

      conn_idx++;

      //set the values of non-diagonal elements to zero
      memset(&Jac_n[jac_idx], 0, N_VARS * N_VARS);
    }

    if (jac_idx == diag_idx)
      jac_idx += N_VARS_SQ;

    count += N_VARS * N_element;

  } // end of loop over grid blocks

  return 0;
}


/// Structure-only + format-conversion tail shared by the host and device
/// adjoint assembly paths (see the header comment). Consumes the already
/// filled dg_dx_n_temp (block) and dg_dT_general (scalar) values.
template <typename Engine>
int super_engine_adjoint_finalize(Engine &e)
{
  constexpr uint8_t N_VARS = Engine::N_VARS;
  constexpr uint16_t N_VARS_SQ = Engine::N_VARS_SQ;

  auto &mesh = e.mesh;
  auto &Jacobian = e.Jacobian;
  auto &dg_dx_T = e.dg_dx_T;
  auto &dg_dx_n = e.dg_dx_n;
  auto &dg_dx_n_temp = e.dg_dx_n_temp;
  auto &well_head_tran_idx_collection = e.well_head_tran_idx_collection;
  auto &well_head_idx_collection = e.well_head_idx_collection;
  auto &wells = e.wells;
  const index_t n_vars = e.n_vars;
  const bool linear_solver_ad_uses_jacobian_transpose = e.linear_solver_ad_uses_jacobian_transpose;

  index_t n_blocks = mesh->n_blocks;
  index_t* rows = Jacobian->get_rows_ptr();
  index_t* cols = Jacobian->get_cols_ind();
  std::vector<index_t>& conn_index_to_one_way = mesh->conn_index_to_one_way;

  // Rebuild well_head_tran_idx_collection (structure-only; the host assembly
  // loop populated this inline, connection-by-connection, in block-ascending /
  // CSR order -- reproduce the same push order here so both paths and the
  // downstream well-head-stripping in the driver see an identical collection).
  well_head_tran_idx_collection.clear();
  for (index_t i = 0; i < n_blocks; ++i)
  {
    bool is_well_head = false;
    for (index_t wh : well_head_idx_collection)
      if (i == wh) { is_well_head = true; break; }
    if (!is_well_head)
      continue;
    index_t conn_idx = rows[i] - i;
    for (index_t csr_idx = rows[i]; csr_idx < rows[i + 1]; ++csr_idx)
    {
      if (cols[csr_idx] == i)
        continue; // diagonal: no connection, conn_idx not advanced
      well_head_tran_idx_collection.push_back(conn_index_to_one_way[conn_idx]);
      conn_idx++;
    }
  }

  // Zero the well-head rows of dg/dx^n (wells contribute no accumulation term).
  for (ms_well* w : wells)
  {
    value_t *jac_n_well_head = &(dg_dx_n_temp->get_values()[dg_dx_n_temp->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
    memset(jac_n_well_head, 0, 2 * N_VARS_SQ * sizeof(value_t));
    for (uint8_t idx = 0; idx < N_VARS; idx++)
    {
      jac_n_well_head[idx + idx * N_VARS] = 0;
    }
  }

  csr_matrix<1> T2;
  value_t* ad_values = dg_dx_T->get_values();
  index_t* ad_rows = dg_dx_T->get_rows_ptr();
  index_t* ad_cols = dg_dx_T->get_cols_ind();
  if (!linear_solver_ad_uses_jacobian_transpose)
  {
    // Legacy adjoint solvers consume an assembled scalar dg_dx_T. The MGR
    // adjoint path keeps the block Jacobian and calls solve_transposed().
    csr_matrix<1> Temp, T1;
#ifdef OPENDARTS_LINEAR_SOLVERS
    Temp.to_nb_1(Jacobian); // unified block_csr_matrix -> polymorphic scalar expansion
#else
    Temp.to_nb_1(static_cast<csr_matrix<N_VARS>*>(Jacobian));
#endif
    T1.build_transpose(&Temp);

    value_t* T1_values = T1.get_values();
    index_t* T1_rows = T1.get_rows_ptr();
    index_t* T1_cols = T1.get_cols_ind();


    for (index_t i = 0; i <= n_blocks * N_VARS; i++)
    {
      ad_rows[i] = T1_rows[i];
    }

    index_t n_value = (mesh->n_conns + mesh->n_blocks) * N_VARS * N_VARS;
    for (index_t i = 0; i < n_value; i++)
    {
      ad_values[i] = T1_values[i];
      ad_cols[i] = T1_cols[i];
    }
  }


  T2.to_nb_1(static_cast<csr_matrix<N_VARS>*>(dg_dx_n_temp));

  value_t* T2_values = T2.get_values();
  index_t* T2_rows = T2.get_rows_ptr();
  index_t* T2_cols = T2.get_cols_ind();

  value_t* ad_values_n = dg_dx_n->get_values();
  index_t* ad_rows_n = dg_dx_n->get_rows_ptr();
  index_t* ad_cols_n = dg_dx_n->get_cols_ind();


  for (index_t i = 0; i <= n_blocks * N_VARS; i++)
  {
    ad_rows_n[i] = T2_rows[i];
  }

  index_t n_value = (mesh->n_conns + mesh->n_blocks) * N_VARS * N_VARS;
  for (index_t i = 0; i < n_value; i++)
  {
    ad_values_n[i] = T2_values[i];
    ad_cols_n[i] = T2_cols[i];
  }

  return 0;
}


/// Full host adjoint assembly: the per-cell/per-connection loops followed by
/// the shared finalize. Used by engine_super_cpu and by the GPU host path.
template <typename Engine>
int super_engine_adjoint_assembly(Engine &e, value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
  (void)jacobian; // the body reads the engine Jacobian member (same object)
  (void)RHS;      // RHS is a dummy in the adjoint driver -- never written here
  super_engine_adjoint_assembly_host_loops(e, dt, X);
  super_engine_adjoint_finalize(e);
  return 0;
}

#endif // ENGINE_SUPER_ADJOINT_HPP
