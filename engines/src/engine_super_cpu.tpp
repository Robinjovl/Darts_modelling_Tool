#include <algorithm>
#include <time.h>
#include <functional>
#include <string>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iomanip>
#include <limits>
#include <math.h>
#include <stdexcept>

#include "engine_super_cpu.hpp"
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
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_cpu<NC, NP, THERMAL>::prepare_weno_fields()
{
  const index_t n_res_blocks = mesh->n_res_blocks;
  const size_t n_fields = static_cast<size_t>(n_res_blocks) * NP * WENO_FIELDS_PER_PHASE;
  weno_field_values.resize(n_fields);
  weno_field_derivatives.resize(n_fields * N_VARS);

  for (index_t cell = 0; cell < n_res_blocks; ++cell)
  {
    for (uint8_t phase = 0; phase < NP; ++phase)
    {
      const size_t field_base = (static_cast<size_t>(cell) * NP + phase) * WENO_FIELDS_PER_PHASE;
      const index_t lambda_op = cell * N_OPS + LAMBDA_OP + phase;
      const value_t lambda = op_vals_arr[lambda_op];
      weno_field_values[field_base + WENO_LAMBDA_FIELD] = lambda;
      for (uint8_t variable = 0; variable < N_VARS; ++variable)
        weno_field_derivatives[(field_base + WENO_LAMBDA_FIELD) * N_VARS + variable] =
          op_ders_arr[lambda_op * N_VARS + variable];

      for (uint8_t equation = 0; equation < NE; ++equation)
      {
        const index_t flux_op = cell * N_OPS + FLUX_OP + phase * NE + equation;
        const value_t flux = op_vals_arr[flux_op];
        const size_t field = field_base + WENO_FLUX_FIELD + equation;
        weno_field_values[field] = lambda * flux;
        for (uint8_t variable = 0; variable < N_VARS; ++variable)
          weno_field_derivatives[field * N_VARS + variable] =
            flux * op_ders_arr[lambda_op * N_VARS + variable] +
            lambda * op_ders_arr[flux_op * N_VARS + variable];
      }

      if constexpr (THERMAL)
      {
        const index_t density_op = cell * N_OPS + GRAV_OP + phase;
        const value_t density = op_vals_arr[density_op];
        const size_t field = field_base + WENO_POTENTIAL_FIELD;
        weno_field_values[field] = lambda * density;
        for (uint8_t variable = 0; variable < N_VARS; ++variable)
          weno_field_derivatives[field * N_VARS + variable] =
            density * op_ders_arr[lambda_op * N_VARS + variable] +
            lambda * op_ders_arr[density_op * N_VARS + variable];
      }
    }
  }
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
bool engine_super_cpu<NC, NP, THERMAL>::reconstruct_weno_scalar(
  index_t target_cell, uint8_t phase, uint8_t field,
  const value_t *candidate_face_coefficient, bool require_nonnegative,
  bool need_derivatives, WenoReconstruction &result) const
{
  result.value = 0.0;
  result.n_dependencies = 0;
  result.bound_fallback = false;

  if (target_cell < 0 || target_cell >= mesh->n_res_blocks || phase >= NP ||
      field >= WENO_FIELDS_PER_PHASE || !mesh->weno_cell_status[target_cell])
    return false;

  const index_t candidate_begin = mesh->weno_cell_candidate_offset[target_cell];
  const index_t candidate_end = mesh->weno_cell_candidate_offset[target_cell + 1];
  const index_t dependency_begin = mesh->weno_cell_dependency_offset[target_cell];
  const index_t dependency_end = mesh->weno_cell_dependency_offset[target_cell + 1];
  const index_t n_candidates = candidate_end - candidate_begin;
  const index_t n_dependencies = dependency_end - dependency_begin;
  const index_t target_dependency = mesh->weno_cell_target_dependency[target_cell];
  if (n_candidates <= 0 || n_candidates > static_cast<index_t>(WENO_MAX_CANDIDATES) ||
      n_dependencies <= 0 || n_dependencies > static_cast<index_t>(WENO_MAX_DEPENDENCIES) ||
      target_dependency < 0 || target_dependency >= n_dependencies)
    return false;

  const auto field_index = [phase, field](index_t cell)
  {
    return (static_cast<size_t>(cell) * NP + phase) * WENO_FIELDS_PER_PHASE + field;
  };
  const value_t target_value = weno_field_values[field_index(target_cell)];
  result.n_dependencies = n_dependencies;

  auto use_target_value = [&]()
  {
    result.value = target_value;
    if (need_derivatives)
    {
      std::fill_n(result.derivative.begin(), n_dependencies, 0.0);
      result.derivative[target_dependency] = 1.0;
    }
  };

  // Only the used prefixes are written before they are read, so these fixed-size
  // scratch arrays are intentionally left uninitialized (avoids a per-call
  // memset of the full WENO_MAX_* capacity in the assembly hot loop).
  std::array<value_t, WENO_MAX_CANDIDATES> candidate_value;
  std::array<value_t, WENO_MAX_CANDIDATES> indicator;
  std::array<value_t, WENO_MAX_CANDIDATES> weight;
  std::array<value_t, 3 * WENO_MAX_CANDIDATES> sigma;
  value_t alpha_sum = 0.0;

  for (index_t local_candidate = 0; local_candidate < n_candidates; ++local_candidate)
  {
    const index_t candidate = candidate_begin + local_candidate;
    value_t difference[3];
    for (index_t support = 0; support < 3; ++support)
    {
      const index_t support_cell = mesh->weno_candidate_support_cell[3 * candidate + support];
      difference[support] = weno_field_values[field_index(support_cell)] - target_value;
    }

    value_t smoothness = 0.0;
    value_t face_value = target_value;
    for (index_t direction = 0; direction < 3; ++direction)
    {
      value_t slope = 0.0;
      for (index_t support = 0; support < 3; ++support)
        slope += mesh->weno_candidate_inverse[9 * candidate + 3 * direction + support] * difference[support];
      sigma[3 * local_candidate + direction] = slope;
      smoothness += slope * slope;
    }
    for (index_t support = 0; support < 3; ++support)
      face_value += candidate_face_coefficient[3 * local_candidate + support] *
                    difference[support];
    candidate_value[local_candidate] = face_value;
    indicator[local_candidate] = smoothness;

    const value_t denominator = params->weno_epsilon + smoothness;
    value_t denominator_power = 1.0;
    for (index_t exponent = 0; exponent < params->weno_power; ++exponent)
      denominator_power *= denominator;
    const value_t alpha = mesh->weno_candidate_gamma[candidate] / denominator_power;
    weight[local_candidate] = alpha;
    alpha_sum += alpha;
  }

  if (!(alpha_sum > 0.0) || !std::isfinite(alpha_sum))
  {
    use_target_value();
    result.bound_fallback = true;
    return false;
  }

  value_t face_value = 0.0;
  for (index_t local_candidate = 0; local_candidate < n_candidates; ++local_candidate)
  {
    weight[local_candidate] /= alpha_sum;
    face_value += weight[local_candidate] * candidate_value[local_candidate];
  }

  if (!std::isfinite(face_value) ||
      (params->weno_bound_fallback && require_nonnegative && face_value < 0.0))
  {
    use_target_value();
    result.bound_fallback = true;
    return false;
  }
  result.value = face_value;

  // The reconstructed value alone is needed for fields consumed only by output
  // quantities (e.g. phase mobility for CFL/velocity), so skip the analytic
  // sensitivities entirely when the caller does not scatter them.
  if (!need_derivatives)
    return true;

  // d(sum_k w_k q_k) = sum_k w_k dq_k +
  //                     sum_k w_k (q_k - q_face) r_k.
  // Each candidate touches at most the target and its three supports, so merge
  // those local coefficients instead of allocating a candidates-by-dependencies matrix.
  std::fill_n(result.derivative.begin(), n_dependencies, 0.0);
  for (index_t local_candidate = 0; local_candidate < n_candidates; ++local_candidate)
  {
    const index_t candidate = candidate_begin + local_candidate;
    index_t local_dependency[4] = {target_dependency, -1, -1, -1};
    value_t candidate_derivative[4] = {1.0, 0.0, 0.0, 0.0};
    value_t indicator_derivative[4] = {0.0, 0.0, 0.0, 0.0};
    index_t coefficient_count = 1;

    auto accumulate_coefficient = [&](index_t dependency, value_t candidate_coefficient,
                                      value_t indicator_coefficient)
    {
      for (index_t coefficient = 0; coefficient < coefficient_count; ++coefficient)
      {
        if (local_dependency[coefficient] == dependency)
        {
          candidate_derivative[coefficient] += candidate_coefficient;
          indicator_derivative[coefficient] += indicator_coefficient;
          return;
        }
      }
      local_dependency[coefficient_count] = dependency;
      candidate_derivative[coefficient_count] = candidate_coefficient;
      indicator_derivative[coefficient_count] = indicator_coefficient;
      ++coefficient_count;
    };

    for (index_t support = 0; support < 3; ++support)
    {
      const value_t face_coefficient =
        candidate_face_coefficient[3 * local_candidate + support];
      value_t smoothness_coefficient = 0.0;
      for (index_t direction = 0; direction < 3; ++direction)
      {
        const value_t inverse = mesh->weno_candidate_inverse[9 * candidate + 3 * direction + support];
        smoothness_coefficient += 2.0 * sigma[3 * local_candidate + direction] * inverse;
      }
      accumulate_coefficient(target_dependency, -face_coefficient, -smoothness_coefficient);
      accumulate_coefficient(mesh->weno_candidate_support_dependency[3 * candidate + support],
                             face_coefficient, smoothness_coefficient);
    }

    const value_t nonlinear_scale =
      -static_cast<value_t>(params->weno_power) /
      (params->weno_epsilon + indicator[local_candidate]);
    for (index_t coefficient = 0; coefficient < coefficient_count; ++coefficient)
    {
      const index_t dependency = local_dependency[coefficient];
      const value_t r = nonlinear_scale * indicator_derivative[coefficient];
      result.derivative[dependency] += weight[local_candidate] *
        (candidate_derivative[coefficient] +
         (candidate_value[local_candidate] - face_value) * r);
    }
  }
  return true;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_cpu<NC, NP, THERMAL>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                            std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                            operator_set_gradient_evaluator_iface* thermal_var_etor_,
                                            sim_params *params_, timer_node *timer_)
{
  const bool use_weno = params_->transport_scheme == sim_params::WENO2;
  if (use_weno)
  {
    if (!mesh_->weno_enabled || !mesh_->weno_finalized)
      throw std::invalid_argument("WENO2 requires finalized static geometry on conn_mesh");
    if (!(params_->weno_epsilon > 0.0) || !std::isfinite(params_->weno_epsilon) ||
        params_->weno_power <= 0)
      throw std::invalid_argument("WENO2 epsilon and nonlinear-weight power must be positive");
    if (opt_history_matching)
      throw std::invalid_argument("WENO2 adjoint/history-matching assembly is not implemented");
    for (index_t cell = 0; cell < mesh_->n_res_blocks; ++cell)
    {
      const index_t candidate_count = mesh_->weno_cell_candidate_offset[cell + 1] -
                                      mesh_->weno_cell_candidate_offset[cell];
      const index_t dependency_count = mesh_->weno_cell_dependency_offset[cell + 1] -
                                       mesh_->weno_cell_dependency_offset[cell];
      if (candidate_count > static_cast<index_t>(WENO_MAX_CANDIDATES) ||
          dependency_count > static_cast<index_t>(WENO_MAX_DEPENDENCIES))
        throw std::invalid_argument("WENO2 setup exceeds the native fixed-size reconstruction limits");
    }
  }
  else if (mesh_->weno_enabled)
    throw std::invalid_argument("WENO geometry was attached while the transport scheme is SPU");

  // prepare dg_dx_n_temp for adjoint method
  if (opt_history_matching)
  {

    if (!dg_dx_n_temp)
    {
      dg_dx_n_temp = new csr_matrix<N_VARS>;
      dg_dx_n_temp->type = MATRIX_TYPE_CSR_FIXED_STRUCTURE;
    }

      // allocate Adjoint matrices
      (static_cast<csr_matrix<N_VARS>*>(dg_dx_n_temp))->init(mesh_->n_blocks, mesh_->n_blocks, N_VARS, mesh_->n_conns + mesh_->n_blocks);
  }

  engine_base::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, thermal_var_etor_, params_, timer_);
  this->expose_jacobian();

  // Initialize phase velocities at all connections including DFM wells
  one_way_phase_vels.resize(mesh_->n_conns / 2);
  phase_vels.resize(mesh_->n_conns);
  phases_vels.resize(mesh_->n_conns * NP);   // velocities are stored phase-wise

  // Initialize derivatives of phase velocities at all connections including DFM wells
  one_way_phase_vels_ders.resize(mesh_->n_conns / 2 * vel_der_size);
  two_way_phase_vels_ders.resize(mesh_->n_conns * vel_der_size);
  phase_vels_ders.resize(mesh_->n_conns * vel_der_size);
  phases_vels_ders.resize(mesh_->n_conns * vel_der_size * NP);   // velocities derivatives are stored phase-wise

  if constexpr (THERMAL)
  {
      min_axis_temp = thermal_var_etor->get_axis_min(T_VAR);
      max_axis_temp = thermal_var_etor->get_axis_max(T_VAR);
  }

  return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_cpu<NC, NP, THERMAL>::enable_flux_output()
{
  enabled_flux_output = true;

  if (darcy_fluxes.empty())
  {
    // mass fluxes
    darcy_fluxes.resize(NC * NP * mesh->n_conns);
    diffusion_fluxes.resize(NP * NC * mesh->n_conns);

    // energy fluxes
    if (THERMAL)
    {
      heat_darcy_advection_fluxes.resize(NP * mesh->n_conns);
      heat_diffusion_advection_fluxes.resize(NP * NC * mesh->n_conns);
      fourier_fluxes.resize((NP + 1) * mesh->n_conns);
    }
  }
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_cpu<NC, NP, THERMAL>::assemble_jacobian_array(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
    index_t n_blocks = mesh->n_blocks;
    index_t n_res_blocks = mesh->n_res_blocks;
    index_t n_conns = mesh->n_conns;
    const std::vector<value_t>& tran = mesh->tran;
    const std::vector<value_t>& tranD = mesh->tranD;
    const std::vector<value_t>& hcap = mesh->heat_capacity;
    const std::vector<value_t>& kin_fac = mesh->kin_factor; // default value of 1
    const std::vector<value_t>& grav_coef = mesh->grav_coef;
    const std::vector<value_t>& velocity_appr = mesh->velocity_appr;
    const std::vector<index_t>& velocity_offset = mesh->velocity_offset;
    const std::vector<index_t>& op_num = mesh->op_num;
    const std::vector<value_t>& cell_spe = mesh->cell_spe;
    const bool use_weno = params->transport_scheme == sim_params::WENO2;

    if (use_weno)
        prepare_weno_fields();

    value_t* Jac = jacobian->get_values();
    index_t* diag_ind = jacobian->get_diag_ind();
    index_t* rows = jacobian->get_rows_ptr();
    index_t* row_thread_starts = jacobian->get_row_thread_starts();

    // for reconstruction of phase velocities
    if (!mesh->velocity_appr.empty() && darcy_velocities.empty())
        darcy_velocities.resize(n_res_blocks * NP * ND);

    if (!mesh->velocity_appr.empty() && !dispersivity.empty())
    {
      if (dispersion_fluxes.empty())
        dispersion_fluxes.resize(NP * NC * n_conns);

      if (THERMAL && heat_dispersion_advection_fluxes.empty())
        heat_dispersion_advection_fluxes.resize(NP * NC * n_conns);
    }

    std::fill(darcy_velocities.begin(), darcy_velocities.end(), 0.0);

    // if velocity reconstruction enabled, also provide molar weights
    if (!mesh->velocity_appr.empty())
    {
      if (molar_weights.empty())
      {
        printf("Velocity reconstruction is enabled. Provide molar weights!");
        exit(-1);
      }
    }

    // fill fourier_fluxes with zeros
    std::fill(fourier_fluxes.begin(), fourier_fluxes.end(), 0.0);

    if (mesh->has_dfm_well)
    {
        update_two_way_phase_vels_and_ders();
    }

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

    index_t j, diag_idx, jac_idx;
    value_t p_diff, gamma_p_diff, t_diff, gamma_t_i, gamma_t_j, gamma_t, mult_i, mult_j;
    value_t CFL_in[NC], CFL_out[NC];
    value_t CFL_max_local = 0;
    value_t phase_presence_mult;
    index_t cell_conn_idx = 0, cell_conn_num = 0;
    std::array<value_t, NP> phase_fluxes;
    uint64_t weno_geometry_fallback_local = 0;
    uint64_t weno_bound_fallback_local = 0;

    // fluxes for output
    value_t *cur_darcy_fluxes = 0, *cur_diffusion_fluxes = 0, *cur_dispersion_fluxes = 0;
    value_t *cur_heat_darcy_advection_fluxes = 0, *cur_heat_diffusion_advection_fluxes = 0,
            *cur_heat_dispersion_advection_fluxes = 0, *cur_fourier_fluxes = 0;

    int connected_with_well;

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

        // [1] fill diagonal part for both mass (and energy equations if needed, only fluid energy is involved here)
        for (uint8_t c = 0; c < NE; c++)
        {
            RHS[i * N_VARS + c] = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]); // acc operators only

            // Add reaction term to diagonal of reservoir cells (here the volume is pore volume or block volume):
            if (i < n_res_blocks)
                RHS[i * N_VARS + c] += (PV[i] + RV[i]) * dt * op_vals_arr[i * N_OPS + KIN_OP + c] * kin_fac[i]; // kinetics

            // Add potential energy accumulation
            if (THERMAL && c == (NE - 1))
            {
                value_t sat_dens_sum = 0.;
                for (uint8_t p = 0; p < NP; p++)
                {
                    sat_dens_sum += op_vals_arr[i * N_OPS + SAT_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p]
                                    - op_vals_arr_n[i * N_OPS + SAT_OP + p] * op_vals_arr_n[i * N_OPS + GRAV_OP + p];
                }
                RHS[i * N_VARS + c] += PV[i] * cell_spe[i] * sat_dens_sum;
            }

            for (uint8_t v = 0; v < N_VARS; v++)
            {
                Jac[diag_idx + c * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v]; // der of accumulation term

                // Include derivatives for reaction term if part of reservoir cells:
                if (i < n_res_blocks)
                {
                    Jac[diag_idx + c * N_VARS + v] += (PV[i] + RV[i]) * dt * op_ders_arr[(i * N_OPS + KIN_OP + c) * N_VARS + v] * kin_fac[i]; // derivative kinetics
                }

                // Include derivatives for potential energy term
                if (THERMAL && c == (NE - 1))
                {
                    value_t sat_dens_sum_der = 0.;
                    for (uint8_t p = 0; p < NP; p++)
                    {
                        sat_dens_sum_der += op_ders_arr[(i * N_OPS + SAT_OP + p) * N_VARS + v] * op_vals_arr[i * N_OPS + GRAV_OP + p]
                                            + op_vals_arr[i * N_OPS + SAT_OP + p] * op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_VARS + v];
                    }
                    Jac[diag_idx + c * N_VARS + v] += PV[i] * cell_spe[i] * sat_dens_sum_der; // der of potential energy accumulation term
                }
            }
        }

        // for velocity reconstruction
        if (!velocity_offset.empty() && i < n_res_blocks)
            cell_conn_num = velocity_offset[i + 1] - velocity_offset[i];

        cell_conn_idx = 0;
        for (index_t conn_idx = mesh->physical_row_offset[i];
             conn_idx < mesh->physical_row_offset[i + 1]; ++conn_idx)
        { // loop over physical connections; WENO-only columns are not fluxes

            j = mesh->block_p[conn_idx];
            jac_idx = N_VARS_SQ * mesh->connection_jacobian_slot[conn_idx];

            bool DFM_conn = mesh->is_dfm_conn[conn_idx];

            // fluxes for current connection
            if (enabled_flux_output)
            {
                cur_darcy_fluxes = &darcy_fluxes[NP * NC * conn_idx];
                cur_diffusion_fluxes = &diffusion_fluxes[NP * NC * conn_idx];
                if constexpr (THERMAL)
                {
                    cur_heat_darcy_advection_fluxes = &heat_darcy_advection_fluxes[NP * conn_idx];
                    cur_heat_diffusion_advection_fluxes = &heat_diffusion_advection_fluxes[NP * NC * conn_idx];
                    cur_fourier_fluxes = &fourier_fluxes[(NP + 1) * conn_idx];
                }
            }

            value_t trans_mult = 1;
            value_t trans_mult_der_i[N_VARS];
            value_t trans_mult_der_j[N_VARS];
            if (params->enable_permporo && i < n_res_blocks && j < n_res_blocks)
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

            if (j >= n_res_blocks)
                connected_with_well = 1;

            // [2] fill offdiagonal part + contribute to diagonal, only fluid part is considered in energy equation
            for (uint8_t p = 0; p < NP; p++)
            { // loop over number of phases for convective operator

                value_t phase_p_diff;
                value_t grav_pc_der_i[N_VARS];
                value_t grav_pc_der_j[N_VARS];
                if (!DFM_conn)
                {
                    // calculate gravity term for phase p
                    value_t avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] + op_vals_arr[j * N_OPS + GRAV_OP + p]) / 2;

                    // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
                    phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

                    // calculate partial derivatives for gravity and capillary terms
                    for (uint8_t v = 0; v < N_VARS; v++)
                    {
                        grav_pc_der_i[v] = (op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(i * N_OPS + PC_OP + p) * N_VARS + v];
                        grav_pc_der_j[v] = (op_ders_arr[(j * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(j * N_OPS + PC_OP + p) * N_VARS + v];
                    }
                }
                else if (DFM_conn)
                {
                    phase_p_diff = phases_vels[p * n_conns + conn_idx];   // value of phase_p_diff is not important, only its sign is used for DFM connections
                }

                phase_fluxes[p] = 0.0;

                // WENO2 changes only the upstream transport coefficients.  The
                // existing two-point phase potential remains the sign selector
                // and multiplier.  Wells, DFM links, and unsupported target
                // sides continue through the bit-for-bit SPU branch below.
                if (use_weno && !DFM_conn && i < n_res_blocks && j < n_res_blocks)
                {
                    const bool target_is_m = phase_p_diff < 0.0;
                    const index_t target_cell = target_is_m ? i : j;
                    const bool face_is_eligible = target_is_m ?
                        mesh->weno_face_status_m[conn_idx] : mesh->weno_face_status_p[conn_idx];
                    if (face_is_eligible)
                    {
                        const index_t coefficient_begin = target_is_m ?
                            mesh->weno_connection_candidate_coefficient_offset_m[conn_idx] :
                            mesh->weno_connection_candidate_coefficient_offset_p[conn_idx];
                        const std::vector<value_t>& candidate_coefficients = target_is_m ?
                            mesh->weno_connection_candidate_coefficient_m :
                            mesh->weno_connection_candidate_coefficient_p;
                        const value_t *candidate_face_coefficient =
                            &candidate_coefficients[coefficient_begin];
                        const index_t dependency_begin = mesh->weno_cell_dependency_offset[target_cell];
                        const index_t slot_begin = target_is_m ?
                            mesh->weno_connection_dependency_offset_m[conn_idx] :
                            mesh->weno_connection_dependency_offset_p[conn_idx];
                        const std::vector<index_t>& dependency_slots = target_is_m ?
                            mesh->weno_connection_dependency_slot_m :
                            mesh->weno_connection_dependency_slot_p;

                        auto field_storage_index = [p](index_t cell, uint8_t field)
                        {
                            return (static_cast<size_t>(cell) * NP + p) *
                                   WENO_FIELDS_PER_PHASE + field;
                        };
                        auto scatter_reconstruction = [&](const WenoReconstruction& reconstruction,
                                                          uint8_t field, uint8_t equation,
                                                          value_t multiplier)
                        {
                            for (index_t dependency = 0;
                                 dependency < reconstruction.n_dependencies; ++dependency)
                            {
                                const index_t source_cell =
                                    mesh->weno_cell_dependency_cell[dependency_begin + dependency];
                                const index_t jacobian_slot = dependency_slots[slot_begin + dependency];
                                const size_t field_index = field_storage_index(source_cell, field);
                                const value_t scalar_sensitivity = reconstruction.derivative[dependency];
                                for (uint8_t v = 0; v < N_VARS; ++v)
                                    Jac[N_VARS_SQ * jacobian_slot + equation * N_VARS + v] -=
                                        multiplier * scalar_sensitivity *
                                        weno_field_derivatives[field_index * N_VARS + v];
                            }
                        };

                        WenoReconstruction reconstruction;
                        // Reconstructed phase mobility feeds only CFL/velocity output,
                        // so its analytic sensitivities are not scattered.
                        reconstruct_weno_scalar(target_cell, p, WENO_LAMBDA_FIELD,
                                                candidate_face_coefficient, true, false,
                                                reconstruction);
                        if (reconstruction.bound_fallback)
                            ++weno_bound_fallback_local;
                        phase_fluxes[p] = -trans_mult * tran[conn_idx] *
                                          phase_p_diff * reconstruction.value;

                        const value_t advective_multiplier =
                            dt * tran[conn_idx] * trans_mult * phase_p_diff;
                        for (uint8_t c = 0; c < NE; ++c)
                        {
                            const uint8_t field = WENO_FLUX_FIELD + c;
                            reconstruct_weno_scalar(target_cell, p, field,
                                                    candidate_face_coefficient, c < NC,
                                                    true, reconstruction);
                            if (reconstruction.bound_fallback)
                                ++weno_bound_fallback_local;

                            const value_t flux = advective_multiplier * reconstruction.value;
                            RHS[i * N_VARS + c] -= flux;
                            scatter_reconstruction(reconstruction, field, c, advective_multiplier);

                            if (c < NC)
                            {
                                if (target_is_m)
                                    CFL_out[c] -= flux;
                                else
                                    CFL_in[c] += flux;
                                if (enabled_flux_output)
                                    cur_darcy_fluxes[p * NC + c] =
                                        -tran[conn_idx] * trans_mult * phase_p_diff *
                                        reconstruction.value;
                            }
                            else if (enabled_flux_output)
                                cur_heat_darcy_advection_fluxes[p] =
                                    -tran[conn_idx] * trans_mult * phase_p_diff *
                                    reconstruction.value;

                            for (uint8_t v = 0; v < N_VARS; ++v)
                            {
                                const value_t phase_p_diff_der_i = grav_pc_der_i[v] -
                                    (v == P_VAR ? 1.0 : 0.0);
                                const value_t phase_p_diff_der_j = grav_pc_der_j[v] +
                                    (v == P_VAR ? 1.0 : 0.0);
                                const value_t multiplier_der_i = dt * tran[conn_idx] *
                                    (trans_mult_der_i[v] * phase_p_diff +
                                     trans_mult * phase_p_diff_der_i);
                                const value_t multiplier_der_j = dt * tran[conn_idx] *
                                    (trans_mult_der_j[v] * phase_p_diff +
                                     trans_mult * phase_p_diff_der_j);
                                Jac[diag_idx + c * N_VARS + v] -=
                                    multiplier_der_i * reconstruction.value;
                                Jac[jac_idx + c * N_VARS + v] -=
                                    multiplier_der_j * reconstruction.value;
                            }

                            if constexpr (THERMAL)
                            {
                                if (c == NE - 1)
                                {
                                    reconstruct_weno_scalar(target_cell, p, WENO_POTENTIAL_FIELD,
                                                            candidate_face_coefficient, true,
                                                            true, reconstruction);
                                    if (reconstruction.bound_fallback)
                                        ++weno_bound_fallback_local;
                                    const value_t potential_multiplier =
                                        dt * tran[conn_idx] * phase_p_diff * cell_spe[target_cell];
                                    RHS[i * N_VARS + c] -=
                                        potential_multiplier * reconstruction.value;
                                    scatter_reconstruction(reconstruction, WENO_POTENTIAL_FIELD, c,
                                                           potential_multiplier);
                                    for (uint8_t v = 0; v < N_VARS; ++v)
                                    {
                                        const value_t phase_p_diff_der_i = grav_pc_der_i[v] -
                                            (v == P_VAR ? 1.0 : 0.0);
                                        const value_t phase_p_diff_der_j = grav_pc_der_j[v] +
                                            (v == P_VAR ? 1.0 : 0.0);
                                        Jac[diag_idx + c * N_VARS + v] -=
                                            dt * tran[conn_idx] * cell_spe[target_cell] *
                                            phase_p_diff_der_i * reconstruction.value;
                                        Jac[jac_idx + c * N_VARS + v] -=
                                            dt * tran[conn_idx] * cell_spe[target_cell] *
                                            phase_p_diff_der_j * reconstruction.value;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    ++weno_geometry_fallback_local;
                }

                if (phase_p_diff < 0)
                {
                    // mass and energy outflow
                    // calculate phase volumetric rate and partial derivatives for phase volumetric rate
                    value_t phase_volumetric_rate;
                    value_t phase_vol_rate_der_i[N_VARS];
                    value_t phase_vol_rate_der_j[N_VARS];
                    if (!DFM_conn)
                    {
                        // calculate phase volumetric rate at reservoir connection using Darcy's law
                        phase_volumetric_rate = tran[conn_idx] * op_vals_arr[i * N_OPS + LAMBDA_OP + p] * phase_p_diff;

                        // calculate partial derivatives of phase volumetric rate at reservoir connection
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            phase_vol_rate_der_i[v] = tran[conn_idx] * (op_ders_arr[(i * N_OPS + LAMBDA_OP + p) * N_VARS + v] * phase_p_diff + op_vals_arr[i * N_OPS + LAMBDA_OP + p] * grav_pc_der_i[v]);
                            phase_vol_rate_der_j[v] = tran[conn_idx] * op_vals_arr[i * N_OPS + LAMBDA_OP + p] * grav_pc_der_j[v];
                        }
                    }
                    else if (DFM_conn)
                    {
                        // calculate phase volumetric rate at DFM well connection
                        value_t phase_velocity = phases_vels[p * n_conns + conn_idx];

                        phase_volumetric_rate = wells[0]->well_transmissibility * op_vals_arr[i * N_OPS + SAT_OP + p] * phase_velocity;

                        // get partial derivatives of phase volumetric rate at DFM well connection
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            value_t phase_velocity_der_i;
                            value_t phase_velocity_der_j;

                            index_t a = (i < j) ? 0 : N_VARS;
                            index_t b = (i < j) ? N_VARS : 0;
                            phase_velocity_der_i = phases_vels_ders[p * n_conns * vel_der_size + conn_idx * vel_der_size + a + v];
                            phase_velocity_der_j = phases_vels_ders[p * n_conns * vel_der_size + conn_idx * vel_der_size + b + v];

                            phase_vol_rate_der_i[v] = wells[0]->well_transmissibility * (op_ders_arr[(i * N_OPS + SAT_OP + p) * N_VARS + v] * phase_velocity + op_vals_arr[i * N_OPS + SAT_OP + p] * phase_velocity_der_i);
                            phase_vol_rate_der_j[v] = wells[0]->well_transmissibility * op_vals_arr[i * N_OPS + SAT_OP + p] * phase_velocity_der_j;
                        }
                    }
                    // mass and energy outflow with effect of gravity and capillarity
                    for (uint8_t c = 0; c < NE; c++)
                    {
                        value_t c_flux_coef = trans_mult * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c] * dt;

                        if (c < NC)
                        {
                            CFL_out[c] -= phase_volumetric_rate * c_flux_coef; // subtract negative value of flux
                            if (!molar_weights.empty())
                                phase_fluxes[p] += op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c] * op_vals_arr[i * N_OPS + LAMBDA_OP + p] * molar_weights[NC * op_num[i] + c];
                            if (enabled_flux_output) cur_darcy_fluxes[p * NC + c] = -phase_volumetric_rate * c_flux_coef / dt;
                        }
                        else
                            if (enabled_flux_output) cur_heat_darcy_advection_fluxes[p] = -phase_volumetric_rate * c_flux_coef / dt;

                        RHS[i * N_VARS + c] -= phase_volumetric_rate * c_flux_coef;

                        // Add potential energy flux
                        if (THERMAL && c == (NE - 1))
                        {
                            RHS[i * N_VARS + c] -= dt * phase_volumetric_rate * op_vals_arr[i * N_OPS + GRAV_OP + p] * cell_spe[i];
                        }

                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            Jac[diag_idx + c * N_VARS + v] -= (phase_volumetric_rate * trans_mult * op_ders_arr[(i * N_OPS + FLUX_OP + p * NE + c) * N_VARS + v] * dt +
                                phase_volumetric_rate * trans_mult_der_i[v] * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c] * dt);
                            Jac[diag_idx + c * N_VARS + v] -= phase_vol_rate_der_i[v] * trans_mult * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c] * dt;
                            Jac[jac_idx + c * N_VARS + v] -= phase_vol_rate_der_j[v] * trans_mult * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c] * dt;

                            if (!DFM_conn)   // Add derivatives with respect to pressure
                            {
                                if (v == 0)
                                {
                                    Jac[jac_idx + c * N_VARS + v] -= c_flux_coef * tran[conn_idx] * op_vals_arr[i * N_OPS + LAMBDA_OP + p];
                                    Jac[diag_idx + c * N_VARS + v] += c_flux_coef * tran[conn_idx] * op_vals_arr[i * N_OPS + LAMBDA_OP + p];
                                }
                            }

                            // Add derivatives of potential energy
                            if (THERMAL && c == (NE - 1))
                            {
                                Jac[diag_idx + c * N_VARS + v] -= dt * (phase_vol_rate_der_i[v] * op_vals_arr[i * N_OPS + GRAV_OP + p] + phase_volumetric_rate * op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_VARS + v]) * cell_spe[i];
                                Jac[jac_idx + c * N_VARS + v] -= dt * phase_vol_rate_der_j[v] * op_vals_arr[i * N_OPS + GRAV_OP + p] * cell_spe[i];

                                if (!DFM_conn)   // Add derivatives with respect to pressure
                                {
                                    if (v == 0)
                                    {
                                        Jac[diag_idx + c * N_VARS + v] += dt * tran[conn_idx] * op_vals_arr[i * N_OPS + LAMBDA_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p] * cell_spe[i];
                                        Jac[jac_idx + c * N_VARS + v] -= dt * tran[conn_idx] * op_vals_arr[i * N_OPS + LAMBDA_OP + p] * op_vals_arr[i * N_OPS + GRAV_OP + p] * cell_spe[i];
                                    }
                                }
                            }
                        }
                    }
                    if (phase_fluxes[p] != 0.0)
                        phase_fluxes[p] *= -trans_mult * tran[conn_idx] * phase_p_diff / op_vals_arr[i * N_OPS + GRAV_OP + p];
                }
                else
                {
                    // mass and energy inflow
                    // calculate phase volumetric rate and partial derivatives for phase volumetric rate
                    value_t phase_volumetric_rate;
                    value_t phase_vol_rate_der_i[N_VARS];
                    value_t phase_vol_rate_der_j[N_VARS];
                    if (!DFM_conn)
                    {
                        // calculate phase volumetric rate at reservoir connection using Darcy's law
                        phase_volumetric_rate = tran[conn_idx] * op_vals_arr[j * N_OPS + LAMBDA_OP + p] * phase_p_diff;

                        // calculate partial derivatives of phase volumetric rate at reservoir connection
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            phase_vol_rate_der_i[v] = tran[conn_idx] * op_vals_arr[j * N_OPS + LAMBDA_OP + p] * grav_pc_der_i[v];
                            phase_vol_rate_der_j[v] = tran[conn_idx] * (op_ders_arr[(j * N_OPS + LAMBDA_OP + p) * N_VARS + v] * phase_p_diff + op_vals_arr[j * N_OPS + LAMBDA_OP + p] * grav_pc_der_j[v]);
                        }
                    }
                    else if (DFM_conn)
                    {
                        // calculate phase volumetric rate at DFM well connection
                        value_t phase_velocity = phases_vels[p * n_conns + conn_idx];

                        phase_volumetric_rate = wells[0]->well_transmissibility * op_vals_arr[j * N_OPS + SAT_OP + p] * phase_velocity;

                        // get partial derivatives of phase volumetric rate at DFM well connection
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            value_t phase_velocity_der_i;
                            value_t phase_velocity_der_j;

                            index_t a = (i < j) ? 0 : N_VARS;
                            index_t b = (i < j) ? N_VARS : 0;
                            phase_velocity_der_i = phases_vels_ders[p * n_conns * vel_der_size + conn_idx * vel_der_size + a + v];
                            phase_velocity_der_j = phases_vels_ders[p * n_conns * vel_der_size + conn_idx * vel_der_size + b + v];

                            phase_vol_rate_der_i[v] = wells[0]->well_transmissibility * op_vals_arr[j * N_OPS + SAT_OP + p] * phase_velocity_der_i;
                            phase_vol_rate_der_j[v] = wells[0]->well_transmissibility * (op_ders_arr[(j * N_OPS + SAT_OP + p) * N_VARS + v] * phase_velocity + op_vals_arr[j * N_OPS + SAT_OP + p] * phase_velocity_der_j);
                        }
                    }
                    // mass and energy inflow with effect of gravity and capillarity
                    for (uint8_t c = 0; c < NE; c++)
                    {
                        value_t c_flux_coef = trans_mult * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c] * dt;

                        if (c < NC)
                        {
                            CFL_in[c] += phase_volumetric_rate * c_flux_coef;
                            if (!molar_weights.empty())
                                phase_fluxes[p] += op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c] * op_vals_arr[j * N_OPS + LAMBDA_OP + p] * molar_weights[NC * op_num[j] + c];
                            if (enabled_flux_output) cur_darcy_fluxes[p * NC + c] = -phase_volumetric_rate * c_flux_coef / dt;
                        }
                        else
                            if (enabled_flux_output) cur_heat_darcy_advection_fluxes[p] = -phase_volumetric_rate * c_flux_coef / dt;

                        RHS[i * N_VARS + c] -= phase_volumetric_rate * c_flux_coef;

						// Add potential energy flux
                        if (THERMAL && c == (NE - 1))
                        {
                            RHS[i * N_VARS + c] -= dt * phase_volumetric_rate * op_vals_arr[j * N_OPS + GRAV_OP + p] * cell_spe[j];
                        }

                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            Jac[jac_idx + c * N_VARS + v] -= (phase_volumetric_rate * trans_mult * op_ders_arr[(j * N_OPS + FLUX_OP + p * NE + c) * N_VARS + v] * dt +
                                phase_volumetric_rate * trans_mult_der_j[v] * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c] * dt);
                            Jac[diag_idx + c * N_VARS + v] -= phase_vol_rate_der_i[v] * trans_mult * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c] * dt;
                            Jac[jac_idx + c * N_VARS + v] -= phase_vol_rate_der_j[v] * trans_mult * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c] * dt;

                            if (!DFM_conn)   // Add derivatives with respect to pressure
                            {
                                if (v == 0)
                                {
                                    Jac[diag_idx + c * N_VARS + v] += c_flux_coef * tran[conn_idx] * op_vals_arr[j * N_OPS + LAMBDA_OP + p];
                                    Jac[jac_idx + c * N_VARS + v] -= c_flux_coef * tran[conn_idx] * op_vals_arr[j * N_OPS + LAMBDA_OP + p];
                                }
                            }

                            // Add derivatives of potential energy
                            if (THERMAL && c == (NE - 1))
                            {
                                Jac[diag_idx + c * N_VARS + v] -= dt * phase_vol_rate_der_i[v] * op_vals_arr[j * N_OPS + GRAV_OP + p] * cell_spe[j];
                                Jac[jac_idx + c * N_VARS + v] -= dt * (phase_vol_rate_der_j[v] * op_vals_arr[j * N_OPS + GRAV_OP + p] + phase_volumetric_rate * op_ders_arr[(j * N_OPS + GRAV_OP + p) * N_VARS + v]) * cell_spe[j];

                                if (!DFM_conn)   // Add derivatives with respect to pressure
                                {
                                    if (v == 0)
                                    {
                                        Jac[diag_idx + c * N_VARS + v] += dt * tran[conn_idx] * op_vals_arr[j * N_OPS + LAMBDA_OP + p] * op_vals_arr[j * N_OPS + GRAV_OP + p] * cell_spe[j];
                                        Jac[jac_idx + c * N_VARS + v] -= dt * tran[conn_idx] * op_vals_arr[j * N_OPS + LAMBDA_OP + p] * op_vals_arr[j * N_OPS + GRAV_OP + p] * cell_spe[j];
                                    }
                                }
                            }
                        }
                    }
                    if (phase_fluxes[p] != 0.0)
                        phase_fluxes[p] *= -trans_mult * tran[conn_idx] * phase_p_diff / op_vals_arr[j * N_OPS + GRAV_OP + p];
                }
            } // end of loop over number of phases for convective operator with gravity and capillarity

            // [3] Additional diffusion code here:   (phi_p * S_p) * (rho_p * D_cp * Delta_x_cp)  or (phi_p * S_p) * (kappa_p * Delta_T)
            // Only if block connection is between reservoir and reservoir cells!
            if (i < n_res_blocks && j < n_res_blocks)
            {
                // Add diffusion term to the residual:
                for (uint8_t c = 0; c < NE; c++)
                {
                    for (uint8_t p = 0; p < NP; p++)
                    {
                        value_t grad_con = op_vals_arr[j * N_OPS + GRAD_OP + p * NE + c] - op_vals_arr[i * N_OPS + GRAD_OP + p * NE + c];

                        if (op_vals_arr[i * N_OPS + UPSAT_OP + p] * op_vals_arr[j * N_OPS + UPSAT_OP + p] > params->phase_existence_tolerance)
                          phase_presence_mult = 1.0;
                        else
                          phase_presence_mult = 0.0;

						value_t diff_mob_ups_m;
                        if (c < NC)
                            diff_mob_ups_m = dt * phase_presence_mult * mesh->tranD[conn_idx] * (mesh->poro[i] * op_vals_arr[i * N_OPS + DENS_OP + p] * op_vals_arr[i * N_OPS + UPSAT_OP + p] +
                                                                                                    mesh->poro[j] * op_vals_arr[j * N_OPS + DENS_OP + p] * op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;
                        else
                            // energy
                            diff_mob_ups_m = dt * phase_presence_mult * mesh->tranD[conn_idx] * (mesh->poro[i] * op_vals_arr[i * N_OPS + UPSAT_OP + p] +
                                                                                                    mesh->poro[j] * op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;

                        RHS[i * N_VARS + c] -= diff_mob_ups_m * grad_con; // diffusion term
                        if (enabled_flux_output)
                        {
                            if (c < NC)
                                cur_diffusion_fluxes[p * NC + c] = -diff_mob_ups_m * grad_con / dt;
                            else
                                cur_fourier_fluxes[p] = -diff_mob_ups_m * grad_con / dt;
                        }

                        // Add diffusion terms to Jacobian:
                        for (uint8_t v = 0; v < N_VARS; v++){
                            // wrt gradient
                            Jac[diag_idx + c * N_VARS + v] += diff_mob_ups_m * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                            Jac[jac_idx + c * N_VARS + v] -= diff_mob_ups_m * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

                            if (c < NC) {
                                // wrt saturation
                                Jac[diag_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] *
                                    mesh->poro[i] * op_vals_arr[i * N_OPS + DENS_OP + p] * op_ders_arr[(i * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;
                                Jac[jac_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] *
                                    mesh->poro[j] * op_vals_arr[j * N_OPS + DENS_OP + p] * op_ders_arr[(j * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;

                                // wrt density
                                Jac[diag_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] *
                                    mesh->poro[i] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] * op_vals_arr[i * N_OPS + UPSAT_OP + p] / 2;
                                Jac[jac_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] *
                                    mesh->poro[j] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] * op_vals_arr[j * N_OPS + UPSAT_OP + p] / 2;
							}
                            else
                            // energy equation
                            {
                                // wrt saturation
                                Jac[diag_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] *
                                        mesh->poro[i] * op_ders_arr[(i * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;
                                Jac[jac_idx + c * N_VARS + v] -= grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] *
                                        mesh->poro[j] * op_ders_arr[(j * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;
							}

                        }
                        if (is_fickian_energy_transport_on)
                        {
                            // respective heat flux
                            if constexpr (THERMAL)
                            {
                                if (c < NC)
                                {
                                    value_t avg_enthalpy = (op_vals_arr[i * N_OPS + ENTH_OP + p] + op_vals_arr[j * N_OPS + ENTH_OP + p]) / 2.;
                                    RHS[i * N_VARS + NC] -= avg_enthalpy * diff_mob_ups_m * grad_con;
                                    if (enabled_flux_output) cur_heat_diffusion_advection_fluxes[p * NC + c] = -avg_enthalpy * diff_mob_ups_m * grad_con / dt;

                                    for (uint8_t v = 0; v < N_VARS; v++)
                                    {
                                        Jac[diag_idx + NC * N_VARS + v] += avg_enthalpy * diff_mob_ups_m * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                                        Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * diff_mob_ups_m * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

                                        // saturation
                                        Jac[diag_idx + NC * N_VARS + v] -= avg_enthalpy * grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[i] *
                                            op_vals_arr[i * N_OPS + DENS_OP + p] * op_ders_arr[(i * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;
                                        Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[j] *
                                            op_vals_arr[j * N_OPS + DENS_OP + p] * op_ders_arr[(j * N_OPS + UPSAT_OP + p) * N_VARS + v] / 2;

                                        // density
                                        Jac[diag_idx + NC * N_VARS + v] -= avg_enthalpy * grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[i] *
                                            op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] * op_vals_arr[i * N_OPS + UPSAT_OP + p]/ 2;
                                        Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * grad_con * dt * phase_presence_mult * mesh->tranD[conn_idx] * mesh->poro[j] *
                                            op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] * op_vals_arr[j * N_OPS + UPSAT_OP + p]/ 2;

                                        // enthalpy
                                        Jac[diag_idx + NC * N_VARS + v] -= op_ders_arr[(i * N_OPS + ENTH_OP + p) * N_VARS + v] * diff_mob_ups_m * grad_con / 2;
                                        Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + ENTH_OP + p) * N_VARS + v] * diff_mob_ups_m * grad_con / 2;
                                    }
                                }
                            }
                        }
                    }
                }

                // assemble velocities, dispersion is assembled in a separate loop as it requires multiple velocities
                if (!velocity_appr.empty())
                {
                    index_t vel_idx = ND * velocity_offset[i];
                    for (uint8_t p = 0; p < NP; p++)
                    {
                        for (uint8_t d = 0; d < ND; d++)
                            darcy_velocities[NP * ND * i + p * ND + d] += velocity_appr[vel_idx + d * cell_conn_num + cell_conn_idx] * phase_fluxes[p];
                    }
                }
            }

            // [4] add rock conduction (between reservoir cells and between DFM well segments and surrounding formations to accound for lateral heat exchange for DFM wells)
            if (THERMAL)
            {
                t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
                gamma_t_i = tranD[conn_idx] * dt * (1 - mesh->poro[i]) * mesh->rock_cond[i];
                gamma_t_j = tranD[conn_idx] * dt * (1 - mesh->poro[j]) * mesh->rock_cond[j];

                // rock heat transfers flows from cell i to j
                RHS[i * N_VARS + NC] -= t_diff * (gamma_t_i + gamma_t_j) / 2;
                if (enabled_flux_output) cur_fourier_fluxes[NP] = -t_diff * (gamma_t_i + gamma_t_j) / 2 / dt;
                for (uint8_t v = 0; v < N_VARS; v++)
                {
                    Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + TEMP_OP) * N_VARS + v] * (gamma_t_i + gamma_t_j) / 2;
                    Jac[diag_idx + NC * N_VARS + v] += op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * (gamma_t_i + gamma_t_j) / 2;
                }
            }
            if (j < n_res_blocks)
                cell_conn_idx++;
        }

        // [5] finally add rock energy
        // + rock energy (no rock compressibility included in these computations)
        if (THERMAL)
        {
          RHS[i * N_VARS + NC] += RV[i] * (op_vals_arr[i * N_OPS + TEMP_OP] - op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];

          for (uint8_t v = 0; v < N_VARS; v++)
          {
            Jac[diag_idx + NC * N_VARS + v] += RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i];
          } // end of fill offdiagonal part + contribute to diagonal
        }

        // calc CFL for reservoir cells, not connected with wells
        if (i < n_res_blocks && !connected_with_well)
        {
            for (uint8_t c = 0; c < NC; c++)
            {
              double denominator = PV[i] * op_vals_arr[i * N_OPS + ACC_OP + c];
              if (denominator != 0.0)
              {
                CFL_max_local = std::max(CFL_max_local, CFL_in[c] / denominator);
                CFL_max_local = std::max(CFL_max_local, CFL_out[c] / denominator);
              }
            }
        }

    } // end of loop over grid blocks
#ifdef _OPENMP
#pragma omp critical
    {
        if (CFL_max < CFL_max_local)
            CFL_max = CFL_max_local;
        weno_geometry_fallback_count += weno_geometry_fallback_local;
        weno_bound_fallback_count += weno_bound_fallback_local;
    }
  } // end of omp parallel
#else
    CFL_max = CFL_max_local;
    weno_geometry_fallback_count += weno_geometry_fallback_local;
    weno_bound_fallback_count += weno_bound_fallback_local;
#endif

    // dispersion
    if (!velocity_appr.empty() && !dispersivity.empty())
    {
        value_t avg_dispersivity;
        value_t *cur_dispersion_fluxes, *cur_heat_dispersion_advection_fluxes;
        std::array<value_t, ND> avg_velocity;

        for (index_t i = 0; i < n_res_blocks; ++i)
        { // loop over grid blocks
            if (i > n_res_blocks) // skip wells
                continue;
            // index of diagonal block entry for block i in CSR values array
            index_t diag_idx = N_VARS_SQ * diag_ind[i];
            for (index_t conn_idx = mesh->physical_row_offset[i];
                 conn_idx < mesh->physical_row_offset[i + 1]; ++conn_idx)
            {
                index_t j = mesh->block_p[conn_idx];
                index_t jac_idx = N_VARS_SQ * mesh->connection_jacobian_slot[conn_idx];

                if (enabled_flux_output)
                {
                  cur_dispersion_fluxes = &dispersion_fluxes[NP * NC * conn_idx];
                  if constexpr (THERMAL)
                    cur_heat_dispersion_advection_fluxes = &heat_dispersion_advection_fluxes[NP * NC * conn_idx];
                }

                if (j < n_res_blocks)
                {
                    for (uint8_t p = 0; p < NP; p++)
                    {
                        value_t avg_enthalpy = (op_vals_arr[i * N_OPS + ENTH_OP + p] + op_vals_arr[j * N_OPS + ENTH_OP + p]) / 2.;

                        // approximate facial velocity
                        index_t vel_idx_i = ND * NP * i;
                        index_t vel_idx_j = ND * NP * j;
                        for (uint8_t d = 0; d < ND; d++)
                            avg_velocity[d] = (op_vals_arr[i * N_OPS + DENS_OP + p] * darcy_velocities[vel_idx_i + p * ND + d] +
                                                op_vals_arr[j * N_OPS + DENS_OP + p] * darcy_velocities[vel_idx_j + p * ND + d]) / 2.0;

                        for (uint8_t c = 0; c < NC; c++)
                        {
                            value_t grad_con = op_vals_arr[j * N_OPS + GRAD_OP + p * NE + c] - op_vals_arr[i * N_OPS + GRAD_OP + p * NE + c];

                            // Diffusion flows from cell i to j (high to low), use upstream quantity from cell i for compressibility and saturation (mass or energy):
                            value_t vel_norm = sqrt(avg_velocity[0] * avg_velocity[0] +
                                avg_velocity[1] * avg_velocity[1] +
                                avg_velocity[2] * avg_velocity[2]);

                            value_t arith_mean_dispersivity = (dispersivity[NP * NC * op_num[i] + p * NC + c] + dispersivity[NP * NC * op_num[j] + p * NC + c]) / 2.;
                            if (arith_mean_dispersivity > 0.)
                                avg_dispersivity = dispersivity[NP * NC * op_num[i] + p * NC + c] * dispersivity[NP * NC * op_num[j] + p * NC + c] / arith_mean_dispersivity;
                            else
                                avg_dispersivity = 0.0;

                            value_t disp = dt * avg_dispersivity * mesh->tranD[conn_idx] * vel_norm;

                            RHS[i * N_VARS + c] -= disp * grad_con; // diffusion term
                            if (enabled_flux_output) cur_dispersion_fluxes[p * NC + c] = -disp * grad_con;

                            // Add diffusion terms to Jacobian:
                            for (uint8_t v = 0; v < N_VARS; v++){
                                Jac[diag_idx + c * N_VARS + v] += disp * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                                Jac[jac_idx + c * N_VARS + v] -= disp * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

                                Jac[diag_idx + c * N_VARS + v] += -dt * avg_dispersivity * mesh->tranD[conn_idx] * vel_norm * grad_con * op_ders_arr[(i * N_OPS + DENS_OP + p * NE) * N_VARS + v] / 2;
                                Jac[jac_idx + c * N_VARS + v] += -dt * avg_dispersivity * mesh->tranD[conn_idx] * vel_norm * grad_con * op_ders_arr[(j * N_OPS + DENS_OP + p * NE) * N_VARS + v] / 2;
                            }

                            // respective heat fluxes
                            if (is_fickian_energy_transport_on)
                            {
                              if constexpr (THERMAL)
                              {
                                RHS[i * N_VARS + NC] -= avg_enthalpy * disp * grad_con;
                                if (enabled_flux_output) cur_heat_dispersion_advection_fluxes[p * NC + c] = -avg_enthalpy * disp * grad_con;

                                for (uint8_t v = 0; v < N_VARS; v++)
                                {
                                  // fraction gradient derivatives
                                  Jac[diag_idx + NC * N_VARS + v] += avg_enthalpy * disp * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
                                  Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * disp * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

                                  Jac[diag_idx + NC * N_VARS + v] -= avg_enthalpy * dt * avg_dispersivity * mesh->tranD[conn_idx] * vel_norm * grad_con * op_ders_arr[(i * N_OPS + DENS_OP + p * NE) * N_VARS + v] / 2;
                                  Jac[jac_idx + NC * N_VARS + v] -= avg_enthalpy * dt * avg_dispersivity * mesh->tranD[conn_idx] * vel_norm * grad_con * op_ders_arr[(j * N_OPS + DENS_OP + p * NE) * N_VARS + v] / 2;

                                  // enthalpy
                                  Jac[diag_idx + NC * N_VARS + v] -= op_ders_arr[(i * N_OPS + ENTH_OP + p) * N_VARS + v] * disp * grad_con / 2;
                                  Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + ENTH_OP + p) * N_VARS + v] * disp * grad_con / 2;
                                }
                              }
                            }
                        }
                    }
                }
            }
        }
    }

  for (ms_well *w : wells)
  {
      if (w->control.get_well_control_type() > well_control_iface::WellControlType::NONE)
      {
          value_t* jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
          w->add_to_jacobian(dt, X, jac_well_head, RHS);
      }
  }

  return 0;
};

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_cpu<NC, NP, THERMAL>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
  index_t n_blocks = mesh->n_blocks;
  index_t n_conns = mesh->n_conns;
  std::vector<value_t> &tran = mesh->tran;
  std::vector<value_t> &tranD = mesh->tranD;
  std::vector<value_t> &hcap = mesh->heat_capacity;
  std::vector<value_t> &kin_fac = mesh->kin_factor; // default value of 1
  std::vector<value_t> &grav_coef = mesh->grav_coef;
  std::vector <index_t>& conn_index_to_one_way = mesh->conn_index_to_one_way;


  value_t* Jac = Jacobian->get_values();
  index_t* diag_ind = Jacobian->get_diag_ind();
  index_t* rows = Jacobian->get_rows_ptr();
  index_t* cols = Jacobian->get_cols_ind();
  index_t* row_thread_starts = Jacobian->get_row_thread_starts();

  value_t* ad_values = dg_dx_T->get_values();
  index_t* ad_rows = dg_dx_T->get_rows_ptr();
  index_t* ad_cols = dg_dx_T->get_cols_ind();
  index_t* ad_diag = dg_dx_T->get_diag_ind();
  index_t* row_T_thread_starts = dg_dx_T->get_row_thread_starts();

  value_t* Jac_n = dg_dx_n_temp->get_values();
  //value_t* v_g_T = dg_dT->get_values();
  value_t* value_dg_dT = dg_dT_general->get_values();
  well_head_tran_idx_collection.clear();

  CFL_max = 0;

  //#ifdef _OPENMP
  //  //#pragma omp parallel reduction (max: CFL_max)
  //#pragma omp parallel
  //  {
  //    int id = omp_get_thread_num();
  //
  //    //index_t start = row_thread_starts[id];
  //    //index_t end = row_thread_starts[id + 1];
  //
  //    //index_t start = row_T_thread_starts[id];
  //    //index_t end = row_T_thread_starts[id + 1];
  //#else
  //  index_t start = 0;
  //  index_t end = n_blocks;
  //
  //#endif //_OPENMP

  index_t start = 0;
  index_t end = n_blocks;

  index_t j, diag_idx, jac_idx;
  value_t p_diff, gamma_p_diff, t_diff, gamma_t_diff, mult_i, mult_j;
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




      for (index_t wh : well_head_idx_collection)
      {
        if (i == wh)
        {
          well_head_tran_idx_collection.push_back(conn_index_to_one_way[conn_idx]);
        }
      }


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

        if (phase_p_diff < 0)
        {
          // mass and energy outflow with effect of gravity and capillarity
          for (uint8_t c = 0; c < NE; c++)
          {
            //value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[i * N_OPS + LAMBDA_OP + p] * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c];

            //RHS[i * N_VARS + c] -= phase_p_diff * c_flux;

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
            //value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[j * N_OPS + LAMBDA_OP + p] * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c];

            //RHS[i * N_VARS + c] -= phase_p_diff * c_flux;

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

            // Diffusion flows, use arithmetic mean for compressibility and saturation (mass or energy):
            // value_t diff_mob_ups_m = dt * phase_presence_mult * mesh->tranD[conn_idx] * (mesh->poro[i] * op_vals_arr[i * N_OPS + UPSAT_OP + p] +
            // mesh->poro[j] * op_vals_arr[j * N_OPS + UPSAT_OP + p]) / 2;
            // RHS[i * N_VARS + c] -= diff_mob_ups_m * grad_con; // diffusion term

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

          // rock heat transfers flows from cell i to j
          //RHS[i * N_VARS + NC] -= gamma_t_diff * ((1 - mesh->poro[i]) * mesh->rock_cond[i] +
          //                                        (1 - mesh->poro[j]) * mesh->rock_cond[j]) / 2;

        value_g_u = dt * t_diff * ((1 - mesh->poro[i]) * mesh->rock_cond[i] +
                                   (1 - mesh->poro[j]) * mesh->rock_cond[j]) / 2;
        idx = count + NC * N_element + temp_num[k_count];
        value_dg_dT[idx] -= value_g_u;
      }

      k_count++;

      conn_idx++;

      //set the values of non-diagonal elements to zero
      /*for (uint8_t c = 0; c < N_VARS; c++)
      {
        for (uint8_t v = 0; v < N_VARS; v++)
        {
          Jac_n[jac_idx + c * N_VARS + v] = 0;
        }
      }*/
      memset(&Jac_n[jac_idx], 0, N_VARS * N_VARS);
    }


    //// [5] finally add rock energy
    //// + rock energy (no rock compressibility included in these computations)
    //if (THERMAL)
    //{
    //  RHS[i * N_VARS + NC] += RV[i] * (op_vals_arr[i * N_OPS + TEMP_OP] - op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];

    //  for (uint8_t v = 0; v < N_VARS; v++)
    //  {
    //    Jac[diag_idx + NC * N_VARS + v] += RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i];
    //  } // end of fill offdiagonal part + contribute to diagonal
    //}


    if (jac_idx == diag_idx)
      jac_idx += N_VARS_SQ;

    count += N_VARS * N_element;


  } // end of loop over grid blocks



//  value_t CFL_max_local = 0;
//#ifdef _OPENMP
//#pragma omp critical
//  {
//    if (CFL_max < CFL_max_local)
//      CFL_max = CFL_max_local;
//  }
//  }
//#else
//  CFL_max = CFL_max_local;
//#endif

  for (ms_well* w : wells)
  {
    //w->add_to_jacobian(dt, X, dg_dx, RHS);

    value_t *jac_n_well_head = &(dg_dx_n_temp->get_values()[dg_dx_n_temp->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
    memset(jac_n_well_head, 0, 2 * N_VARS_SQ * sizeof(value_t));
    for (uint8_t idx = 0; idx < N_VARS; idx++)
    {
      jac_n_well_head[idx + idx * N_VARS] = 0;
    }
  }

  // we have to convert the csr matrix to the csr matrix with block size of 1
  // because the function "build_transpose" is only applicable for the csr matrix with the block size of 1
  // this is also required by the linear solver "linsolv_superlu<1>", as the preconditioner is not applicable to adjoint so far
  // so this might be improved in the future
  csr_matrix<1> Temp, T1, T2;
  Temp.to_nb_1(static_cast<csr_matrix<N_VARS>*>(Jacobian));
  T1.build_transpose(&Temp);

  value_t* T1_values = T1.get_values();
  index_t* T1_rows = T1.get_rows_ptr();
  index_t* T1_cols = T1.get_cols_ind();
  index_t* T1_diag = T1.get_diag_ind();


  for (index_t i = 0; i <= n_blocks * N_VARS; i++)
  {
    //ad_diag[i] = i;  //so far using superlu, it may need to be fixed if using other linear solver
    ad_rows[i] = T1_rows[i];
  }
  //ad_rows[n_blocks * N_VARS] = T1_rows[n_blocks * N_VARS];

  index_t n_value = (mesh->n_conns + mesh->n_blocks) * N_VARS * N_VARS;
  for (index_t i = 0; i < n_value; i++)
  {
    ad_values[i] = T1_values[i];
    ad_cols[i] = T1_cols[i];
  }


  T2.to_nb_1(static_cast<csr_matrix<N_VARS>*>(dg_dx_n_temp));
  //T2.build_transpose(&Temp);

  value_t* T2_values = T2.get_values();
  index_t* T2_rows = T2.get_rows_ptr();
  index_t* T2_cols = T2.get_cols_ind();
  index_t* T2_diag = T2.get_diag_ind();

  value_t* ad_values_n = dg_dx_n->get_values();
  index_t* ad_rows_n = dg_dx_n->get_rows_ptr();
  index_t* ad_cols_n = dg_dx_n->get_cols_ind();
  index_t* ad_diag_n = dg_dx_n->get_diag_ind();



  for (index_t i = 0; i <= n_blocks * N_VARS; i++)
  {
    //ad_diag_n[i] = i;  //so far using superlu, it may need to be fixed if using other linear solver
    ad_rows_n[i] = T2_rows[i];
  }
  //ad_rows_n[n_blocks * N_VARS] = T2_rows[n_blocks * N_VARS];

  n_value = (mesh->n_conns + mesh->n_blocks) * N_VARS * N_VARS;
  for (index_t i = 0; i < n_value; i++)
  {
    ad_values_n[i] = T2_values[i];
    ad_cols_n[i] = T2_cols[i];
  }


    return 0;
};

/**
 * @brief Update two-way phase velocities and their corresponding derivatives based on the new phase velocities and derivatives for DFM wells.
 *
 * This function is only applicable when at least one DFM well exists.
 *
 * @return void.
 */
template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_cpu<NC, NP, THERMAL>::update_two_way_phase_vels_and_ders()
{
    for (uint8_t p = 0; p < NP; p++)
    {
        for (ms_well* w : wells)
        {
            if (w->ms_type == ms_well::MS_Type::DFM)
            {
                const size_t well_n_conns = w->num_segments - 1;

                std::vector<value_t> well_phase_vels(well_n_conns);
                std::vector<value_t> well_phase_vels_ders(well_n_conns * vel_der_size);

                // Get the velocities of the phase in the DFM well (DFM phase velocities and derivatives are evaluated in Python)
                std::copy_n(w->phases_vels.begin() + p * well_n_conns, well_n_conns, well_phase_vels.begin());
                std::copy_n(w->phases_vels_ders.begin() + p * well_n_conns * vel_der_size, well_n_conns * vel_der_size, well_phase_vels_ders.begin());

                // Update one-way phase velocities
                std::copy(well_phase_vels.begin(), well_phase_vels.end(), one_way_phase_vels.begin() + w->well_head_conn_idx);

                // Update one-way phase velocities derivatives
                std::copy(well_phase_vels_ders.begin(), well_phase_vels_ders.end(), one_way_phase_vels_ders.begin() + w->well_head_conn_idx * vel_der_size);
            }
        }
        // Reverse and sort one-way phase velocities
        mesh->reverse_and_sort_one_way(one_way_phase_vels, phase_vels);
        // Reverse and sort one-way phase velocities derivatives
        mesh->reverse_and_sort_one_way<value_t, true>(one_way_phase_vels_ders, phase_vels_ders);

        // Update the array containing velocities of all phases
        std::copy(phase_vels.begin(), phase_vels.end(), phases_vels.begin() + p * mesh->n_conns);
        std::copy(phase_vels_ders.begin(), phase_vels_ders.end(), phases_vels_ders.begin() + p * mesh->n_conns * vel_der_size);
    }
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
void engine_super_cpu<NC, NP, THERMAL>::apply_thermal_var_correction(std::vector<value_t>& X, std::vector<value_t>& dX)
{
    index_t n_thermal_var_corr{ 0 };  // Number of states corrected for temperature under-/overshoot

    index_t nb = mesh->n_blocks;

    std::vector<value_t> state(n_vars);

    std::vector<value_t> X_new(nb * n_vars);
    std::vector<value_t> op_vals_arr_new(n_ops * nb);
    std::vector<value_t> op_ders_arr_new(n_ops * nb * n_vars);

    for (index_t i = 0; i < dX.size(); i++)
    {
        X_new[i] = X[i] - dX[i];
    }

    for (int r = 0; r < acc_flux_op_set_list.size(); r++)
    {
        int result = acc_flux_op_set_list[r]->evaluate_with_derivatives(X_new, block_idxs[r], op_vals_arr_new, op_ders_arr_new);
        //if (result < 0)
        //	return 0;
    }

    for (index_t i = 0; i < nb; i++)
    {
        // If TEMP_OP out of [T_min, T_max] bounds, use thermal_var_etor to calculate thermal variable at p and T_bound
        value_t new_temperature = op_vals_arr_new[i * n_ops + TEMP_OP];
        if (new_temperature < min_axis_temp || new_temperature > max_axis_temp)
        {
            // Define PT-state
            for (index_t c = 0; c < n_vars - 1; c++)
            {
                state[c] = X[i * n_vars + c] - dX[i * n_vars + c];
            }
            state[T_VAR] = (new_temperature < min_axis_temp) ? min_axis_temp : max_axis_temp;

            // Evaluate thermal_var_etor
            std::vector<value_t> thermal_var_op(1);
            this->thermal_var_etor->evaluate(state, thermal_var_op);

            dX[i * n_vars + T_VAR] = X[i * n_vars + T_VAR] - thermal_var_op[0];

            if (n_thermal_var_corr == 0)
			{
				std::cout << "Thermal variable correction: block " << i;
                std::cout << ((new_temperature < min_axis_temp)
                    ? " shoots under T axis limit of "
                    : " shoots over T axis limit of ");
                std::cout << state[T_VAR] << " to " << new_temperature << "\n";
			}
            new_temperature = state[T_VAR];
            n_thermal_var_corr++;
        }

        // Cap maximum temperature change between updates (needs further investigation)
        if (false)
        {
            value_t dT = std::abs(new_temperature - op_vals_arr_n[i * n_ops + TEMP_OP]);
            value_t dT_max = 20.;

            //value_t ds = std::abs(op_vals_arr_new[i * n_ops + SAT_OP] - op_vals_arr_n[i * n_ops + SAT_OP]);
            //value_t ds_max = 0.2;

            if (dT > dT_max)
            //if (ds > ds_max)
            {
                value_t chopping_factor = dT_max / dT;
                //value_t chopping_factor = ds_max / ds;
                //dX[i * n_vars + P_VAR] *= chopping_factor;
                dX[i * n_vars + T_VAR] *= chopping_factor;
            }
        }
    }

    if (n_thermal_var_corr)
	{
		std::cout << "Thermal variable correction applied " << n_thermal_var_corr << " time(s) \n";
	}
}

//template<uint8_t NC, uint8_t NP, , bool THERMAL>
//double
//engine_super_cpu<NC, NP, THERMAL>::calc_newton_residual()
//{
//  double residual = 0, res;
//
//  std::vector <value_t> &hcap = mesh->heat_capacity;
//
//  residual = 0;
//  for (int i = 0; i < mesh->n_blocks; i++)
//  {
//    for (int c = 0; c < NC; c++)
//    {
//      res = fabs(RHS[i * N_VARS + c] / (PV[i] * op_vals_arr[i * N_OPS + c]));
//      if (res > residual)
//        residual = res;
//    }
//
//    if (THERMAL)
//    {
//      res = fabs(RHS[i * N_VARS + T_VAR] / (PV[i] * op_vals_arr[i * N_OPS + NC] + RV[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i]));
//      if (res > residual)
//        residual = res;
//    }
//  }
//  return residual;
//}

// compositional, kinetic (H2O, CO2, Ca+2, CO3-2, CaCO3):
//template class engine_super_cpu<2, 2, 1>;
//template struct recursive_instantiator_nc_np<engine_super_cpu, 2, MAX_NC, 1>;
//template struct recursive_instantiator_nc_np<engine_super_cpu, 2, MAX_NC, 2>;
//template struct recursive_instantiator_nc_np<engine_super_cpu, 2, MAX_NC, 3>;
