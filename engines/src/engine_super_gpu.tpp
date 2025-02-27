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

#include "engine_super_gpu.hpp"


/**
 * @brief Reconstruct phase velocities based on input data and operators.
 *
 * This kernel (thread per cell) reconstructs the velocities of phases in a computational grid,
 * accounting for various physical parameters such as pressure differences, density,
 * gravity, and phase flux. The result is stored in `darcy_velocities`.
 *
 * @tparam NC Number of components in the fluid.
 * @tparam NP Number of phases.
 * @tparam NE Number of equations.
 * @tparam N_VARS Number of variables per block.
 * @tparam P_VAR Index of pressure variable.
 * @tparam N_OPS Number of operators.
 * @tparam FLUX_OP Index for flux operators.
 * @tparam GRAV_OP Index for gravity operators.
 * @tparam PC_OP Index for capillary pressure operators.
 * @tparam PORO_OP Index for porosity operators.
 *
 * @param[in] n_res_blocks Number of reservoir blocks.
 * @param[in] trans_mult_exp Exponent for transmissibility multiplier calculation.
 * @param[in] X Array of unknowns (pressure, temperature, etc.).
 * @param[in] op_vals_arr Array of operator values.
 * @param[in] op_num Array of operator numbers per block.
 * @param[in] rows CSR row pointer array for the sparse matrix.
 * @param[in] cols CSR column index array for the sparse matrix.
 * @param[in] tran Array of transmissibilities.
 * @param[in] grav_coef Array of gravity coefficients.
 * @param[in] velocity_appr Array of velocity approximations.
 * @param[in] velocity_offset Velocity offset per block.
 * @param[out] darcy_velocities Output array for the reconstructed Darcy velocities.
 * @param[in] molar_weights Array of molar weights for fluid components.
 * @param[in] dt Time step size.
 */
template <uint8_t NC, uint8_t NP, uint8_t NE, uint8_t N_VARS, uint8_t P_VAR, uint8_t N_OPS, uint8_t FLUX_OP,
          uint8_t GRAV_OP, uint8_t PC_OP, uint8_t PORO_OP>
__global__ void
reconstruct_velocities(const unsigned int n_res_blocks, const unsigned int trans_mult_exp,
                      value_t *X, value_t *op_vals_arr, index_t *op_num, index_t *rows,
                      index_t *cols, value_t *tran,  value_t *grav_coef, value_t *velocity_appr,
                      index_t *velocity_offset, value_t *darcy_velocities, value_t *molar_weights,
                      value_t dt)
{
  // mesh grid block number
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > n_res_blocks - 1)
    return;

  // space dimension
  const uint8_t ND = 3;

  // index of first entry for block i in CSR cols array
  index_t csr_idx_start = rows[i];
  // index of last entry for block i in CSR cols array
  index_t csr_idx_end = rows[i + 1];
  // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
  index_t conn_idx = csr_idx_start - i;

  index_t j, cell_conn_idx, vel_idx;
  value_t p_diff, phi_avg, trans_mult, avg_density, phase_p_diff;
  value_t phase_flux;
  index_t cell_conn_num = velocity_offset[i + 1] - velocity_offset[i];

  for (uint8_t p = 0; p < NP; p++)
    for (uint8_t d = 0; d < ND; d++)
      darcy_velocities[NP * ND * i + p * ND + d] = 0.0;

  cell_conn_idx = 0;
  for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++)
  {
    j = cols[csr_idx];
    // skip diagonal
    if (i == j)
      continue;

    if (j >= n_res_blocks)
    {
      conn_idx++;
      continue;
    }

    trans_mult = 1.;
    if (trans_mult_exp > 0)
    {
      // Take average interface porosity:
      phi_avg = (op_vals_arr[i * N_OPS + PORO_OP] + op_vals_arr[j * N_OPS + PORO_OP]) * 0.5;
      trans_mult = pow(phi_avg, (double) trans_mult_exp);
    }

    p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];

    for (uint8_t p = 0; p < NP; p++)
    {
      avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] + op_vals_arr[j * N_OPS + GRAV_OP + p]) * 0.5;
      phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

      phase_flux = 0.0;
      if (phase_p_diff < 0)
      {
        for (uint8_t c = 0; c < NC; c++)
          phase_flux += op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c] * molar_weights[NC * op_num[i] + c];

        if (phase_flux != 0.0)
            phase_flux *= -trans_mult * tran[conn_idx] * phase_p_diff / op_vals_arr[i * N_OPS + GRAV_OP + p];
      }
      else
      {
        for (uint8_t c = 0; c < NC; c++)
          phase_flux += op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c] * molar_weights[NC * op_num[j] + c];

        if (phase_flux != 0.0)
          phase_flux *= -trans_mult * tran[conn_idx] * phase_p_diff / op_vals_arr[j * N_OPS + GRAV_OP + p];
      }

      vel_idx = ND * velocity_offset[i];
      for (uint8_t d = 0; d < ND; d++)
        darcy_velocities[NP * ND * i + p * ND + d] += velocity_appr[vel_idx + d * cell_conn_num + cell_conn_idx] * phase_flux;
    }
    conn_idx++;
    cell_conn_idx++;
  }
}

/**
 * @brief Assemble the dispersion term for a phase in a computational grid.
 *
 * This kernel (thread per component*varible) computes the contribution of dispersion and diffusion terms to the
 * residual and Jacobian matrix, which is used in solving flow and transport equations.
 *
 * @tparam NC Number of components in the fluid.
 * @tparam NP Number of phases.
 * @tparam NE Number of equations.
 * @tparam N_VARS Number of variables per block.
 * @tparam N_OPS Number of operators.
 * @tparam GRAD_OP Index for gradient operators.
 * @tparam ENTH_OP Index for enthalpy operators.
 * @tparam THERMAL Enable or disable thermal effects.
 *
 * @param[in] n_res_blocks Number of reservoir blocks.
 * @param[in] X Array of unknowns (pressure, temperature, etc.).
 * @param[out] RHS Right-hand side array for the residual.
 * @param[in] op_vals_arr Array of operator values.
 * @param[in] op_ders_arr Array of operator derivatives.
 * @param[in] rows CSR row pointer array for the sparse matrix.
 * @param[in] cols CSR column index array for the sparse matrix.
 * @param[out] Jac Output Jacobian matrix.
 * @param[in] diag_ind Array of diagonal indices in the sparse matrix.
 * @param[in] tranD Array of dispersion transmissibilities.
 * @param[in] darcy_velocities Array of Darcy velocities.
 * @param[in] dispersivity Array of phase dispersivities.
 * @param[in] op_num Array of region per block.
 * @param[in] dt Time step size.
 */
template <uint8_t NC, uint8_t NP, uint8_t NE, uint8_t N_VARS, uint8_t N_OPS, uint8_t GRAD_OP, uint8_t ENTH_OP,
          bool THERMAL>
__global__ void
assemble_dispersion(const unsigned int n_res_blocks, value_t *X, value_t *RHS, value_t *op_vals_arr,
                    value_t *op_ders_arr, index_t *rows, index_t *cols, value_t *Jac, index_t *diag_ind,
                    value_t *tranD, value_t *darcy_velocities, value_t *dispersivity, index_t *op_num, value_t dt)
{
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / (NC * N_VARS);
  if (i > n_res_blocks - 1)
    return;

  const int block_pos = (blockIdx.x * blockDim.x + threadIdx.x) % (NC * N_VARS);
  const int v = block_pos % N_VARS;
  const int c = block_pos / N_VARS;

  const int N_VARS_SQ = N_VARS * N_VARS;
  const int ND = 3;
  value_t jac_diag = 0, jac_offd, rhs = 0, avg_dispersivity, avg_enthalpy, grad_con,
          vel_norm, arith_mean_dispersivity, disp;
  value_t avg_velocity[ND];

  // index of diagonal block entry for block i in CSR values array
  index_t diag_idx = N_VARS_SQ * diag_ind[i];
  // index of first entry for block i in CSR cols array
  index_t csr_idx_start = rows[i];
  // index of last entry for block i in CSR cols array
  index_t csr_idx_end = rows[i + 1];
  // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
  index_t conn_idx = csr_idx_start - i;

  index_t jac_idx = N_VARS_SQ * csr_idx_start;
  index_t vel_idx_i, vel_idx_j, j;

  for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++)
  {

    j = cols[csr_idx];
    // skip diagonal
    if (i == j)
      continue;

    if (j >= n_res_blocks)
    {
      conn_idx++;
      continue;
    }

    // initialize offdiagonal value for current connection
    jac_offd = 0;
    for (uint8_t p = 0; p < NP; p++)
    {
      avg_enthalpy = 0.5 * (op_vals_arr[i * N_OPS + ENTH_OP + p] + op_vals_arr[j * N_OPS + ENTH_OP + p]);

      // approximate facial velocity
      vel_idx_i = ND * NP * i;
      vel_idx_j = ND * NP * j;
      for (uint8_t d = 0; d < ND; d++)
          avg_velocity[d] = 0.5 * (darcy_velocities[vel_idx_i + p * ND + d] + darcy_velocities[vel_idx_j + p * ND + d]);

      grad_con = op_vals_arr[j * N_OPS + GRAD_OP + p * NE + c] - op_vals_arr[i * N_OPS + GRAD_OP + p * NE + c];

      // Diffusion flows from cell i to j (high to low), use upstream quantity from cell i for compressibility and saturation (mass or energy):
      vel_norm = sqrt(avg_velocity[0] * avg_velocity[0] + avg_velocity[1] * avg_velocity[1] + avg_velocity[2] * avg_velocity[2]);

      arith_mean_dispersivity = 0.5 * (dispersivity[NP * NC * op_num[i] + p * NC + c] + dispersivity[NP * NC * op_num[j] + p * NC + c]);
      if (arith_mean_dispersivity > 0.)
          avg_dispersivity = dispersivity[NP * NC * op_num[i] + p * NC + c] * dispersivity[NP * NC * op_num[j] + p * NC + c] / arith_mean_dispersivity;
      else
          avg_dispersivity = 0.0;

      disp = dt * avg_dispersivity * tranD[conn_idx] * vel_norm;

      if (v == 0)
      {
        rhs -= disp * grad_con;
      }

      // Add diffusion terms to Jacobian:
      jac_diag += disp * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
      jac_offd -= disp * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

      // respective heat fluxes
      if constexpr (THERMAL)
      {
        // use atomics below because threads for all components are writing into energy balance equation
        if (v == 0)
        {
          atomicAdd(&RHS[i * N_VARS + NC], -avg_enthalpy * disp * grad_con);
        }

          atomicAdd(&Jac[diag_idx + NC * N_VARS + v], avg_enthalpy * disp * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v]);
          atomicAdd(&Jac[jac_idx + NC * N_VARS + v], -avg_enthalpy * disp * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v]);

          atomicAdd(&Jac[diag_idx + NC * N_VARS + v], -op_ders_arr[(i * N_OPS + ENTH_OP + p) * N_VARS + v] * disp * grad_con / 2);
          atomicAdd(&Jac[jac_idx + NC * N_VARS + v], -op_ders_arr[(j * N_OPS + ENTH_OP + p) * N_VARS + v] * disp * grad_con / 2);
      }
    }

    Jac[csr_idx * N_VARS_SQ + block_pos] += jac_offd;
    conn_idx++;
  }

  Jac[diag_idx + block_pos] += jac_diag;

  if (v == 0)
  {
    RHS[i * N_VARS + c] += rhs;
  }
}

/**
 * @brief Assemble the Jacobian matrix for a set of equations with thermal effects.
 *
 * This kernel assembles the Jacobian matrix for a system of flow and transport equations
 * with potential thermal effects. It accounts for accumulation, convection, diffusion, and
 * reaction terms.
 *
 * @tparam NC Number of components in the fluid.
 * @tparam NP Number of phases.
 * @tparam NE Number of equations.
 * @tparam N_VARS Number of variables per block.
 * @tparam P_VAR Index of pressure variable.
 * @tparam T_VAR Index of temperature variable.
 * @tparam N_OPS Number of operators.
 * @tparam ACC_OP Index for accumulation operators.
 * @tparam FLUX_OP Index for flux operators.
 * @tparam UPSAT_OP Index for upstream saturation operators.
 * @tparam GRAD_OP Index for gradient operators.
 * @tparam KIN_OP Index for kinetic operators.
 * @tparam GRAV_OP Index for gravity operators.
 * @tparam PC_OP Index for capillary pressure operators.
 * @tparam PORO_OP Index for porosity operators.
 * @tparam ENTH_OP Index for enthalpy operators.
 * @tparam TEMP_OP Index for temperature operators.
 * @tparam PRES_OP Index for pressure operators.
 * @tparam THERMAL Enable or disable thermal effects.
 *
 * @param[in] n_blocks Total number of blocks.
 * @param[in] n_res_blocks Number of reservoir blocks.
 * @param[in] trans_mult_exp Exponent for transmissibility multiplier.
 * @param[in] phase_existence_tolerance Tolerance value for phase existence.
 * @param[in] dt Time step size.
 * @param[in] X Array of unknowns.
 * @param[out] RHS Right-hand side residual array.
 * @param[in] rows CSR row pointer array for the sparse matrix.
 * @param[in] cols CSR column index array for the sparse matrix.
 * @param[out] Jac Output Jacobian matrix.
 * @param[in] diag_ind Diagonal indices in the sparse matrix.
 * @param[in] op_vals_arr Operator values at the current timestep.
 * @param[in] op_vals_arr_n Operator values at the previous timestep.
 * @param[in] op_ders_arr Operator derivatives array.
 * @param[in] tran Transmissibility array.
 * @param[in] tranD Diffusion transmissibility array.
 * @param[in] hcap Array of heat capacities.
 * @param[in] rock_cond Array of rock conductivities.
 * @param[in] poro Array of porosity values.
 * @param[in] PV Pore volume array.
 * @param[in] RV Rock volume array.
 * @param[in] grav_coef Array of gravity coefficients.
 * @param[in] kin_fac Kinetic factor array.
 */
template <uint8_t NC, uint8_t NP, uint8_t NE, uint8_t N_VARS, uint8_t P_VAR, uint8_t T_VAR, uint8_t N_OPS,
          uint8_t ACC_OP, uint8_t FLUX_OP, uint8_t UPSAT_OP, uint8_t GRAD_OP, uint8_t KIN_OP,
          uint8_t GRAV_OP, uint8_t PC_OP, uint8_t PORO_OP, uint8_t ENTH_OP, uint8_t TEMP_OP, uint8_t PRES_OP,
          bool THERMAL>
__global__ void
assemble_jacobian_array_kernel(const unsigned int n_blocks, const unsigned int n_res_blocks, const unsigned int trans_mult_exp,
                               value_t phase_existence_tolerance,
                               value_t dt, value_t *X, value_t *RHS,
                               index_t *rows, index_t *cols, value_t *Jac, index_t *diag_ind,
                               value_t *op_vals_arr, value_t *op_vals_arr_n, value_t *op_ders_arr,
                               value_t *tran, value_t *tranD, value_t *hcap, value_t *rock_cond, value_t *poro,
                               value_t *PV, value_t *RV, value_t *grav_coef, value_t *kin_fac)
{
  // Each matrix block row is processed by N_VARS * N_VARS threads
  // Memory access is coalesced for most data, while communications minimized

  const int N_VARS_SQ = N_VARS * N_VARS;

  // mesh grid block number
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / N_VARS_SQ;
  // Each thread is pinned to specific position in matrix block
  // and evaluates diagonal and all offdiagonal Jacobian entries at that position for the block row i
  // if the block position corresponds to the first column of the block (v==0, see below),
  // that thread also evaluates RHS value for equation c of the the block row i.
  const int block_pos = (blockIdx.x * blockDim.x + threadIdx.x) % N_VARS_SQ;
  // v (variable) is the column of that position
  const int v = block_pos % N_VARS;
  // c (component, but it is outdated notation- better to use e, equation) is the row of that position
  const int c = block_pos / N_VARS;

  if (i > n_blocks - 1)
    return;

  // local value of jacobian diagonal block value at block_pos
  // gets contributions in the loop over connections and is written to global memory at the end
  value_t jac_diag = 0;
  // local value of RHS according to c
  // gets contributions in the loop over connections and is written to global memory at the end
  value_t rhs = 0;
  // local value of jacobian offdiagonal block value at block_pos
  // is evaluated and written to global memory during each iteration of the loop over connections
  value_t jac_offd;
  // local variable used to eliminate diffusion fluxes in case any of neighbours does not have a phase
  value_t phase_presence_mult;

  index_t j;
  value_t p_diff, t_diff, gamma_t_i, gamma_t_j, phi_i, phi_j, phi_avg;

  // [1] fill diagonal part for both mass (and energy equations if needed, only fluid energy is involved here)
  if (v == 0)
  {
    rhs = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]); // acc operators only

    // Add reaction term to diagonal of reservoir cells (here the volume is pore volume or block volume):
    if (i < n_res_blocks)
      rhs += (PV[i] + RV[i]) * dt * op_vals_arr[i * N_OPS + KIN_OP + c] * kin_fac[i]; // kinetics
  }

  jac_diag = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v]; // der of accumulation term

  // Include derivatives for reaction term if part of reservoir cells:
  if (i < n_res_blocks)
  {
    jac_diag += (PV[i] + RV[i]) * dt * op_ders_arr[(i * N_OPS + KIN_OP + c) * N_VARS + v] * kin_fac[i]; // derivative kinetics
  }

  // if thermal is enabled, full up the last equation
  if (THERMAL && c == (NE - 1))
  {
    if (v == 0)
    {
      rhs += RV[i] * (op_vals_arr[i * N_OPS + TEMP_OP] - op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];
    }

    jac_diag += RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i];
  }

  // index of first entry for block i in CSR cols array
  index_t csr_idx_start = rows[i];
  // index of last entry for block i in CSR cols array
  index_t csr_idx_end = rows[i + 1];
  // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
  index_t conn_idx = csr_idx_start - i;

  // fill offdiagonal part + contribute to diagonal
  for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++)
  {

    j = cols[csr_idx];
    // skip diagonal
    if (i == j)
      continue;

    // initialize offdiagonal value for current connection
    jac_offd = 0;

    value_t trans_mult = 1;
    value_t trans_mult_der_i = 0;
    value_t trans_mult_der_j = 0;
    if (trans_mult_exp > 0 && (i < n_res_blocks && j < n_res_blocks))
    {
      // Calculate transmissibility multiplier:
      phi_i = op_vals_arr[i * N_OPS + PORO_OP];
      phi_j = op_vals_arr[j * N_OPS + PORO_OP];

      // Take average interface porosity:
      phi_avg = (phi_i + phi_j) * 0.5;
      value_t phi_0_avg = (poro[i] + poro[j]) * 0.5;
      trans_mult = trans_mult_exp * pow(phi_avg, (double) trans_mult_exp - 1) * 0.5;
      trans_mult_der_i = trans_mult * op_ders_arr[(i * N_OPS + PORO_OP) * N_VARS + v];
      trans_mult_der_j = trans_mult * op_ders_arr[(j * N_OPS + PORO_OP) * N_VARS + v];
      trans_mult = pow(phi_avg, (double) trans_mult_exp);
    }

    p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];

    // [2] fill offdiagonal part + contribute to diagonal, only fluid part is considered in energy equation
    for (uint8_t p = 0; p < NP; p++)
    { // loop over number of phases for convective operator

      // calculate gravity term for phase p
      value_t avg_density = (op_vals_arr[i * N_OPS + GRAV_OP + p] +
                             op_vals_arr[j * N_OPS + GRAV_OP + p]) /
                            2;

      // p = 1 means oil phase, it's reference phase. pw=po-pcow, pg=po-(-pcog).
      value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] - op_vals_arr[j * N_OPS + PC_OP + p] + op_vals_arr[i * N_OPS + PC_OP + p];

      // calculate partial derivatives for gravity and capillary terms
      value_t grav_pc_der_i;
      value_t grav_pc_der_j;

      grav_pc_der_i = -(op_ders_arr[(i * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 - op_ders_arr[(i * N_OPS + PC_OP + p) * N_VARS + v];
      grav_pc_der_j = -(op_ders_arr[(j * N_OPS + GRAV_OP + p) * N_VARS + v]) * grav_coef[conn_idx] / 2 + op_ders_arr[(j * N_OPS + PC_OP + p) * N_VARS + v];

      double phase_gamma_p_diff = trans_mult * tran[conn_idx] * dt * phase_p_diff;

      if (phase_p_diff < 0)
      {
        // mass and energy outflow with effect of gravity and capillarity
        value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c];
        if (v == 0)
        {
          rhs -= phase_p_diff * c_flux; // flux operators only
          jac_offd -= c_flux;
          jac_diag += c_flux;
        }
        jac_diag -= (phase_gamma_p_diff * op_ders_arr[(i * N_OPS + FLUX_OP + p * NE + c) * N_VARS + v] +
                     tran[conn_idx] * dt * phase_p_diff * trans_mult_der_i * op_vals_arr[i * N_OPS + FLUX_OP + p * NE + c]);
        jac_diag += c_flux * grav_pc_der_i;
        jac_offd += c_flux * grav_pc_der_j;
      }
      else
      {
        // mass and energy inflow with effect of gravity and capillarity
        value_t c_flux = trans_mult * tran[conn_idx] * dt * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c];
        if (v == 0)
        {
          rhs -= phase_p_diff * c_flux; // flux operators only
          jac_diag += c_flux;           //-= Jac[jac_idx + c * N_VARS];
          jac_offd -= c_flux;           // -tran[conn_idx] * dt * op_vals[NC + c];
        }
        jac_offd -= (phase_gamma_p_diff * op_ders_arr[(j * N_OPS + FLUX_OP + p * NE + c) * N_VARS + v] +
                     tran[conn_idx] * dt * phase_p_diff * trans_mult_der_j * op_vals_arr[j * N_OPS + FLUX_OP + p * NE + c]);
        jac_diag += c_flux * grav_pc_der_i;
        jac_offd += c_flux * grav_pc_der_j;
      }
    } // end of loop over number of phases for convective operator with gravity and capillarity

    // [3] Additional diffusion code here:   (phi_p * S_p) * (rho_p * D_cp * Delta_x_cp)  or (phi_p * S_p) * (kappa_p * Delta_T)
    phi_avg = (poro[i] + poro[j]) * 0.5; // diffusion term depends on total porosity!

    // Only if block connection is between reservoir and reservoir cells!
    if (i < n_res_blocks && j < n_res_blocks)
    {
      // Add diffusion term to the residual:
      for (uint8_t p = 0; p < NP; p++)
      {
        value_t grad_con = op_vals_arr[j * N_OPS + GRAD_OP + p * NE + c] - op_vals_arr[i * N_OPS + GRAD_OP + p * NE + c];

        if (op_vals_arr[i * N_OPS + UPSAT_OP + p] * op_vals_arr[j * N_OPS + UPSAT_OP + p] > phase_existence_tolerance)
          phase_presence_mult = 1.0;
        else
          phase_presence_mult = 0.0;

        value_t diff_mob_ups_m = dt * phase_presence_mult * tranD[conn_idx] * (poro[i] * op_vals_arr[i * N_OPS + UPSAT_OP + p] +
                                                                              poro[j] * op_vals_arr[j * N_OPS + UPSAT_OP + p]) * 0.5;

        if (v == 0)
        {
          rhs -= diff_mob_ups_m * grad_con; // diffusion term
        }

        // Add diffusion terms to Jacobian:
        jac_diag += diff_mob_ups_m * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];
        jac_offd -= diff_mob_ups_m * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v];

        jac_diag -= grad_con * dt * phase_presence_mult * tranD[conn_idx] * poro[i] * op_ders_arr[(i * N_OPS + UPSAT_OP + p) * N_VARS + v] * 0.5;
        jac_offd -= grad_con * dt * phase_presence_mult * tranD[conn_idx] * poro[j] * op_ders_arr[(j * N_OPS + UPSAT_OP + p) * N_VARS + v] * 0.5;

        if constexpr (THERMAL)
        {
          if (c < NC)
          {
            value_t avg_enthalpy = (op_vals_arr[i * N_OPS + ENTH_OP + p] + op_vals_arr[j * N_OPS + ENTH_OP + p]) * 0.5;

            // use atomics below because threads for all components are writing into energy balance equation
            if (v == 0)
            {
              atomicAdd(&RHS[i * N_VARS + NC], -avg_enthalpy * diff_mob_ups_m * grad_con);
            }

            index_t diag_energy = diag_ind[i] * N_VARS_SQ + NC * N_VARS + v;
            index_t jac_energy = csr_idx * N_VARS_SQ + NC * N_VARS + v;

            atomicAdd(&Jac[diag_energy], avg_enthalpy * diff_mob_ups_m * op_ders_arr[(i * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v]);
            atomicAdd(&Jac[jac_energy], -avg_enthalpy * diff_mob_ups_m * op_ders_arr[(j * N_OPS + GRAD_OP + p * NE + c) * N_VARS + v]);

            atomicAdd(&Jac[diag_energy], -avg_enthalpy * grad_con * dt * phase_presence_mult * tranD[conn_idx] * poro[i] * op_ders_arr[(i * N_OPS + UPSAT_OP + p) * N_VARS + v] * 0.5);
            atomicAdd(&Jac[jac_energy], -avg_enthalpy * grad_con * dt * phase_presence_mult * tranD[conn_idx] * poro[j] * op_ders_arr[(j * N_OPS + UPSAT_OP + p) * N_VARS + v] * 0.5);

            atomicAdd(&Jac[diag_energy], -op_ders_arr[(i * N_OPS + ENTH_OP + p) * N_VARS + v] * diff_mob_ups_m * grad_con * 0.5);
            atomicAdd(&Jac[jac_energy], -op_ders_arr[(j * N_OPS + ENTH_OP + p) * N_VARS + v] * diff_mob_ups_m * grad_con * 0.5);
          }
        }
      }
    }

    // [4] add rock conduction
    // if thermal is enabled, full up the last equation
    if (THERMAL && (c == NE - 1))
    {
      t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
      gamma_t_i = tranD[conn_idx] * dt * (1 - poro[i]) * rock_cond[i];
      gamma_t_j = tranD[conn_idx] * dt * (1 - poro[j]) * rock_cond[j];

      // rock heat transfers flows from cell i to j
      rhs -= t_diff * (gamma_t_i + gamma_t_j) / 2;
      for (uint8_t v = 0; v < N_VARS; v++)
      {
        jac_offd -= op_ders_arr[(j * N_OPS + TEMP_OP) * N_VARS + v] * (gamma_t_i + gamma_t_j) / 2;
        jac_diag += op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * (gamma_t_i + gamma_t_j) / 2;
      }
    }
    // write down offdiag value
    Jac[csr_idx * N_VARS_SQ + block_pos] = jac_offd;
    conn_idx++;
  }
  // index of diagonal block entry for block i
  index_t diag_idx = N_VARS_SQ * diag_ind[i];
  // write down diag value and RHS
  Jac[diag_idx + block_pos] = jac_diag;

  if (v == 0)
  {
    RHS[i * N_VARS + c] = rhs;
  }
};

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_gpu<NC, NP, THERMAL>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                            std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                            sim_params *params_, timer_node *timer_)
{
  engine_base_gpu::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);

  allocate_device_data(RV, &RV_d);
  allocate_device_data(mesh->heat_capacity, &mesh_hcap_d);
  allocate_device_data(mesh->tranD, &mesh_tranD_d);
  allocate_device_data(mesh->rock_cond, &mesh_rcond_d);
  allocate_device_data(mesh->poro, &mesh_poro_d);
  allocate_device_data(mesh->kin_factor, &mesh_kin_factor_d);
  allocate_device_data(mesh->grav_coef, &mesh_grav_coef_d);

  copy_data_to_device(RV, RV_d);
  copy_data_to_device(mesh->heat_capacity, mesh_hcap_d);
  copy_data_to_device(mesh->tranD, mesh_tranD_d);
  copy_data_to_device(mesh->rock_cond, mesh_rcond_d);
  copy_data_to_device(mesh->poro, mesh_poro_d);
  copy_data_to_device(mesh->kin_factor, mesh_kin_factor_d);
  copy_data_to_device(mesh->grav_coef, mesh_grav_coef_d);

  return 0;
}

template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_gpu<NC, NP, THERMAL>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
  timer->node["jacobian assembly"].node["kernel"].start_gpu();
  //cudaMemset(jacobian->values_d, 0, jacobian->rows_ptr[mesh->n_blocks] * N_VARS_SQ * sizeof(double));

  assemble_jacobian_array_kernel<NC, NP, NE, N_VARS, P_VAR, T_VAR, N_OPS, ACC_OP, FLUX_OP, UPSAT_OP, GRAD_OP, KIN_OP,
                                 GRAV_OP, PC_OP, PORO_OP, ENTH_OP, TEMP_OP, PRES_OP, THERMAL>
      KERNEL_1D(mesh->n_blocks, N_VARS * N_VARS, 64)(mesh->n_blocks, mesh->n_res_blocks, params->trans_mult_exp,
                                                     params->phase_existence_tolerance,
                                                     dt, X_d, RHS_d,
                                                     jacobian->rows_ptr_d, jacobian->cols_ind_d, jacobian->values_d, jacobian->diag_ind_d,
                                                     op_vals_arr_d, op_vals_arr_n_d, op_ders_arr_d,
                                                     mesh_tran_d, mesh_tranD_d, mesh_hcap_d, mesh_rcond_d, mesh_poro_d,
                                                     PV_d, RV_d, mesh_grav_coef_d, mesh_kin_factor_d);

  if (!mesh->velocity_appr.empty()) // reconstruction of phase velocities
  {
    if (molar_weights.empty())
    {
      printf("Velocity reconstruction is enabled. Provide molar weights!");
      exit(-1);
    }

    reconstruct_velocities<NC, NP, NE, N_VARS, P_VAR, N_OPS, FLUX_OP, GRAV_OP, PC_OP, PORO_OP>
        KERNEL_1D(mesh->n_res_blocks, 1, 64)(mesh->n_res_blocks, params->trans_mult_exp,
                                        X_d, op_vals_arr_d, mesh_op_num_d, jacobian->rows_ptr_d,
                                        jacobian->cols_ind_d, mesh_tran_d, mesh_grav_coef_d,
                                        mesh_velocity_appr_d, mesh_velocity_offset_d, darcy_velocities_d, molar_weights_d,
                                        dt);
    copy_data_to_host(darcy_velocities, darcy_velocities_d);

    if (!dispersivity.empty())
    {
      assemble_dispersion<NC, NP, NE, N_VARS, N_OPS, GRAD_OP, ENTH_OP, THERMAL>
        KERNEL_1D(mesh->n_res_blocks, NC * N_VARS, 64)(mesh->n_res_blocks, X_d, RHS_d, op_vals_arr_d,
                                                    op_ders_arr_d, jacobian->rows_ptr_d, jacobian->cols_ind_d, jacobian->values_d, jacobian->diag_ind_d,
                                                    mesh_tranD_d, darcy_velocities_d, dispersivity_d, mesh_op_num_d, dt);
    }
  }

  timer->node["jacobian assembly"].node["kernel"].stop_gpu();
  timer->node["jacobian assembly"].node["wells"].start_gpu();
  int i_w = 0;
  for (ms_well *w : wells)
  {
    value_t *jac_well_head = &(jac_wells[2 * N_VARS * N_VARS * i_w]);
    w->add_to_jacobian(dt, X, jac_well_head, RHS);
    i_w++;
  }
  copy_data_to_device(jac_wells, jac_wells_d);
  copy_data_to_device(RHS, RHS_wells_d);

  i_w = 0;
  for (ms_well *w : wells)
  {
    copy_data_within_device(RHS_d + N_VARS * w->well_head_idx, RHS_wells_d + N_VARS * w->well_head_idx, N_VARS);
    copy_data_within_device(jacobian->values_d + jacobian->rows_ptr[w->well_head_idx] * N_VARS * N_VARS, jac_wells_d + 2 * N_VARS * N_VARS * i_w, 2 * N_VARS * N_VARS);
    i_w++;
  }
  timer->node["jacobian assembly"].node["wells"].stop_gpu();

  // copy_data_to_host(jacobian->values, jacobian->values_d, N_VARS * N_VARS * jacobian->rows_ptr[mesh->n_blocks]);
  // jacobian->write_matrix_to_file("jac_nc_dar_gpu.csr");
  //exit(0);

  return 0;
}


template <uint8_t NC, uint8_t NP, bool THERMAL>
int engine_super_gpu<NC, NP, THERMAL>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	return 0;
};