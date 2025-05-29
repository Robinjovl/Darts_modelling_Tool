#include <algorithm>
#include <time.h>
#include <functional>
#include <string>
#include <iomanip>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cstring>

#include "engine_nce_g_cpu.hpp"

template <uint8_t NC, uint8_t NP>
int engine_nce_g_cpu<NC, NP>::init(conn_mesh *mesh_, std::vector<ms_well *> &well_list_,
                                   std::vector<operator_set_gradient_evaluator_iface *> &acc_flux_op_set_list_,
                                   sim_params *params_, timer_node *timer_)
{
	scale_rows = false;
	scale_dimless = false;
	e_dim = m_dim = p_dim = 1;

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

	
    engine_base::init_base<N_VARS>(mesh_, well_list_, acc_flux_op_set_list_, params_, timer_);
	this->expose_jacobian();

	max_row_values_inv.resize(n_vars * mesh->n_blocks);

    return 0;
}

template <uint8_t NC, uint8_t NP>
void engine_nce_g_cpu<NC, NP>::enable_flux_output()
{
  enabled_flux_output = true;

  if (darcy_fluxes.empty())
  {
	darcy_fluxes.resize(NC * NP * mesh->n_conns);
	heat_darcy_advection_fluxes.resize(NP * mesh->n_conns);
	fourier_fluxes.resize((NP + 1) * mesh->n_conns);
  }
}

template <uint8_t NC, uint8_t NP>
int engine_nce_g_cpu<NC, NP>::assemble_jacobian_array(value_t dt, std::vector<value_t> &X, csr_matrix_base *jacobian, std::vector<value_t> &RHS)
{
    // We need extended connection list for that with all connections for each block

    index_t n_blocks = mesh->n_blocks;
    index_t n_conns = mesh->n_conns;
	index_t n_res_blocks = mesh->n_res_blocks;
    std::vector<index_t> &block_m = mesh->block_m;
    std::vector<index_t> &block_p = mesh->block_p;
    std::vector<value_t> &tran = mesh->tran;
    std::vector<value_t> &tranD = mesh->tranD;
    std::vector<value_t> &hcap = mesh->heat_capacity;
	std::vector<value_t>& velocity_appr = mesh->velocity_appr;
	std::vector<index_t>& velocity_offset = mesh->velocity_offset;
    std::vector<value_t> &grav_coef = mesh->grav_coef;

    value_t *Jac = jacobian->get_values();
    index_t *diag_ind = jacobian->get_diag_ind();
    index_t *rows = jacobian->get_rows_ptr();
    index_t *cols = jacobian->get_cols_ind();
    index_t *row_thread_starts = jacobian->get_row_thread_starts();

	// for reconstruction of phase velocities
	if (!mesh->velocity_appr.empty() && darcy_velocities.empty())
		darcy_velocities.resize(n_res_blocks * NP * ND);

	std::fill(darcy_velocities.begin(), darcy_velocities.end(), 0.0);
    CFL_max = 0;

#ifdef _OPENMP
    //#pragma omp parallel reduction (max: CFL_max)
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        index_t start = row_thread_starts[id];
        index_t end = row_thread_starts[id + 1];
#else
    index_t start = 0;
    index_t end = n_blocks;
#endif //_OPENMP

        index_t j, diag_idx, jac_idx;
        value_t p_diff, gamma_p_diff;
        value_t t_diff, gamma_t_diff;
		index_t cell_conn_idx, cell_conn_num;
		std::array<value_t, NP> phase_fluxes;

        numa_set(Jac, 0, rows[start] * N_VARS_SQ, rows[end] * N_VARS_SQ);

		// fluxes for output
		value_t *cur_darcy_fluxes;
		value_t *cur_heat_darcy_advection_fluxes, *cur_fourier_fluxes;

        for (index_t i = start; i < end; i++)
        {
            //multitable->interpolate (&X[N_VARS * i], op_vals, op_ders);
            //multitable->interpolate_acc_val_only (&Xn[N_VARS * i], op_vals_tmp);
            // index of diagonal block entry for block i in CSR values array
            diag_idx = N_VARS_SQ * diag_ind[i];

			// for velocity reconstruction
			if (!velocity_offset.empty() && i < n_res_blocks)
				cell_conn_num = velocity_offset[i + 1] - velocity_offset[i];

            // fill diagonal part
            // [NC] mass eqns
            for (uint8_t c = 0; c < NC; c++)
            {
                RHS[i * N_VARS + c] = PV[i] * (op_vals_arr[i * N_OPS + ACC_OP + c] - op_vals_arr_n[i * N_OPS + ACC_OP + c]); // acc operators only
                for (uint8_t v = 0; v < N_VARS; v++)
                {
                    Jac[diag_idx + c * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v];
                }
            }

            // [1] energy eqn
            // fluid energy
            RHS[i * N_VARS + NC] = PV[i] * (op_vals_arr[i * N_OPS + FE_ACC_OP] - op_vals_arr_n[i * N_OPS + FE_ACC_OP]);
            // + rock energy (no rock compressibility included in these computations)
            RHS[i * N_VARS + NC] += RV[i] * (op_vals_arr[i * N_OPS + TEMP_OP] - op_vals_arr_n[i * N_OPS + TEMP_OP]) * hcap[i];

            for (uint8_t v = 0; v < N_VARS; v++)
            {
                Jac[diag_idx + NC * N_VARS + v] = PV[i] * op_ders_arr[(i * N_OPS + FE_ACC_OP) * N_VARS + v];
                Jac[diag_idx + NC * N_VARS + v] += RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i];
            }

            // index of first entry for block i in CSR cols array
            index_t csr_idx_start = rows[i];
            // index of last entry for block i in CSR cols array
            index_t csr_idx_end = rows[i + 1];
            // index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
            index_t conn_idx = csr_idx_start - i;

            jac_idx = N_VARS_SQ * csr_idx_start;
			cell_conn_idx = 0;

            // fill offdiagonal part + contribute to diagonal
            for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS_SQ)
            {
                j = cols[csr_idx];
                // skip diagonal
                if (i == j)
                    continue;

				// fluxes for current connection
				if (enabled_flux_output)
				{
				  cur_darcy_fluxes = &darcy_fluxes[NP * NC * conn_idx];
				  cur_heat_darcy_advection_fluxes = &heat_darcy_advection_fluxes[NP * conn_idx];
				  cur_fourier_fluxes = &fourier_fluxes[(NP + 1) * conn_idx];
				}

                p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];
                t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
                gamma_t_diff = tranD[conn_idx] * dt * t_diff;

                for (uint8_t p = 0; p < NP; p++)
                {
                    // calculate gravity term for phase p
                    value_t avg_density = (op_vals_arr[i * N_OPS + DENS_OP + p] + op_vals_arr[j * N_OPS + DENS_OP + p]) * 0.5;

                    value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] * H2O_MW;
                    double phase_gamma_p_diff = tran[conn_idx] * dt * phase_p_diff;
					phase_fluxes[p] = 0.0;

                    if (phase_p_diff < 0)
                    {
                        // mass outflow
                        for (uint8_t c = 0; c < NC; c++)
                        {
                            value_t c_flux = tran[conn_idx] * dt * op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c];
							phase_fluxes[p] += op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c] * H2O_MW;
                            RHS[i * N_VARS + c] -= phase_gamma_p_diff * op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c]; // flux operators only
							if (enabled_flux_output)
							  cur_darcy_fluxes[p * NC + c] = -phase_gamma_p_diff * op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c] / dt;

                            for (uint8_t v = 0; v < N_VARS; v++)
                            {
                                Jac[diag_idx + c * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + FLUX_OP + p * NC + c) * N_VARS + v];
                                Jac[diag_idx + c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                                Jac[jac_idx + c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                                if (v == P_VAR)
                                {
                                    Jac[jac_idx + c * N_VARS] -= c_flux;
                                    Jac[diag_idx + c * N_VARS] += c_flux;
                                }
                                //else
                                //{
                                //    Jac[jac_idx + c * N_VARS + v] = 0;
                                //}
                            }
                        }

                        // energy outflow
                        RHS[i * N_VARS + E_VAR] -= phase_gamma_p_diff * op_vals_arr[i * N_OPS + FE_FLUX_OP + p]; // energy flux
						if (enabled_flux_output)
						  cur_heat_darcy_advection_fluxes[p] = -phase_gamma_p_diff * op_vals_arr[i * N_OPS + FE_FLUX_OP + p] / dt;
                        value_t phase_e_flux = tran[conn_idx] * dt * op_vals_arr[i * N_OPS + FE_FLUX_OP + p];

                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            Jac[diag_idx + NC * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(i * N_OPS + FE_FLUX_OP + p) * N_VARS + v];
                            Jac[diag_idx + NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                            Jac[jac_idx + NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                            if (v == P_VAR)
                            {
                                Jac[jac_idx + NC * N_VARS] -= phase_e_flux;
                                Jac[diag_idx + NC * N_VARS] += phase_e_flux;
                            }
                            //else
                            //{
                            //    Jac[jac_idx + NC * N_VARS + v] = 0;
                            //}
                        }
						if (phase_fluxes[p] != 0.0)
							phase_fluxes[p] *= - tran[conn_idx] * phase_p_diff / op_vals_arr[i * N_OPS + DENS_OP + p];
                    }
                    else
                    {
                        //inflow

                        // mass
                        for (uint8_t c = 0; c < NC; c++)
                        {
                            value_t c_flux = tran[conn_idx] * dt * op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c];
							phase_fluxes[p] += op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c] * H2O_MW;
                            RHS[i * N_VARS + c] -= phase_gamma_p_diff * op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c]; // flux operators only
							if (enabled_flux_output)
							  cur_darcy_fluxes[p * NC + c] = -phase_gamma_p_diff * op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c] / dt;

                            for (uint8_t v = 0; v < N_VARS; v++)
                            {
                                Jac[jac_idx + c * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + FLUX_OP + p * NC + c) * N_VARS + v];
                                Jac[jac_idx + c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                                Jac[diag_idx + c * N_VARS + v] -= c_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                                if (v == P_VAR)
                                {
                                    Jac[diag_idx + c * N_VARS] += c_flux;
                                    Jac[jac_idx + c * N_VARS] -= c_flux;
                                }
                            }
                        }
						if (phase_fluxes[p] != 0.0)
							phase_fluxes[p] *= - tran[conn_idx] * phase_p_diff / op_vals_arr[j * N_OPS + DENS_OP + p];

                        // energy flux
                        RHS[i * N_VARS + E_VAR] -= phase_gamma_p_diff * op_vals_arr[j * N_OPS + FE_FLUX_OP + p]; // energy flux operator
						if (enabled_flux_output)
						  cur_heat_darcy_advection_fluxes[p] = -phase_gamma_p_diff * op_vals_arr[j * N_OPS + FE_FLUX_OP + p] / dt;
						value_t phase_e_flux = tran[conn_idx] * dt * op_vals_arr[j * N_OPS + FE_FLUX_OP + p];
                        for (uint8_t v = 0; v < N_VARS; v++)
                        {
                            Jac[jac_idx + NC * N_VARS + v] -= phase_gamma_p_diff * op_ders_arr[(j * N_OPS + FE_FLUX_OP + p) * N_VARS + v];
                            Jac[jac_idx + NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(j * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                            Jac[diag_idx + NC * N_VARS + v] -= phase_e_flux * grav_coef[conn_idx] * op_ders_arr[(i * N_OPS + DENS_OP + p) * N_VARS + v] * 0.5 * H2O_MW;
                            if (v == P_VAR)
                            {
                                Jac[diag_idx + NC * N_VARS] += phase_e_flux;
                                Jac[jac_idx + NC * N_VARS] -= phase_e_flux;
                            }
                        }
                    }
                }

				if (i < n_res_blocks && j < n_res_blocks)
				{

					// assemble velocities
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

                if (t_diff < 0)
                {
                    // energy outflow

                    // conduction
                    value_t local_cond_dt = tranD[conn_idx] * dt * (op_vals_arr[i * N_OPS + FE_COND_OP] * mesh->poro[i] + (1 - mesh->poro[i]) * mesh->rock_cond[i]);

                    RHS[i * N_VARS + NC] -= local_cond_dt * t_diff;
					if (enabled_flux_output)
					  cur_fourier_fluxes[NP] = -local_cond_dt * t_diff / dt;
                    for (uint8_t v = 0; v < N_VARS; v++)
                    {
                        // conduction part derivative
                        Jac[diag_idx + NC * N_VARS + v] -= gamma_t_diff * op_ders_arr[(i * N_OPS + FE_COND_OP) * N_VARS + v] * mesh->poro[i];
                        // t_diff derivatives
                        Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
                        Jac[diag_idx + NC * N_VARS + v] += op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
                    }
                }
                else
                {
                    //energy inflow
                    // conduction
                    value_t local_cond_dt = tranD[conn_idx] * dt * (op_vals_arr[j * N_OPS + FE_COND_OP] * mesh->poro[j] + (1 - mesh->poro[j]) * mesh->rock_cond[j]);

                    RHS[i * N_VARS + NC] -= local_cond_dt * t_diff;
					if (enabled_flux_output)
					  cur_fourier_fluxes[NP] = -local_cond_dt * t_diff / dt;
                    for (uint8_t v = 0; v < N_VARS; v++)
                    {
                        // conduction part derivative
                        Jac[jac_idx + NC * N_VARS + v] -= gamma_t_diff * op_ders_arr[(j * N_OPS + FE_COND_OP) * N_VARS + v] * mesh->poro[j];
                        // t_diff derivatives
                        Jac[jac_idx + NC * N_VARS + v] -= op_ders_arr[(j * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
                        Jac[diag_idx + NC * N_VARS + v] += op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * local_cond_dt;
                    }
                }
                conn_idx++;
				if (j < n_res_blocks)
					cell_conn_idx++;
            }
        }
#ifdef _OPENMP
    }
#endif
    //Jacobian.write_matrix_to_file("jac_nc_dar_before_wells.csr");
    //write_vector_to_file("jac_nc_dar_before_wells.rhs", RHS);
    for (ms_well *w : wells)
    {
        value_t *jac_well_head = &(jacobian->get_values()[jacobian->get_rows_ptr()[w->well_head_idx] * n_vars * n_vars]);
        w->add_to_jacobian(dt, X, jac_well_head, RHS);
    }

    return 0;
};

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_cpu<NC, NP>::calc_newton_residual_L2()
{
    double residual = 0, res = 0;
    double res_m = 0, res_e = 0;
    std::vector<value_t> &hcap = mesh->heat_capacity;

    for (int i = 0; i < mesh->n_res_blocks; i++)
    {
        for (int c = 0; c < NC; c++)
        {
            res = fabs(RHS[i * N_VARS + c] / (PV[i] * op_vals_arr[i * N_OPS + c]));
            res_m += res * res;
        }

        res = fabs(RHS[i * N_VARS + E_VAR] / (PV[i] * op_vals_arr[i * N_OPS + FE_ACC_OP] + RV[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i]));
        res_e += res * res;
    }
    residual = sqrt(res_m + res_e);
    return residual;
}

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_cpu<NC, NP>::calc_newton_residual_Linf()
{
    double residual = 0, res = 0;
    std::vector<value_t> &hcap = mesh->heat_capacity;

    for (int i = 0; i < mesh->n_res_blocks; i++)
    {
        for (int c = 0; c < NC; c++)
        {
            res = fabs(RHS[i * N_VARS + c] / (PV[i] * op_vals_arr[i * N_OPS + c]));
            if (res > residual)
                residual = res;
        }

        res = fabs(RHS[i * N_VARS + E_VAR] / (PV[i] * op_vals_arr[i * N_OPS + FE_ACC_OP] + RV[i] * op_vals_arr[i * N_OPS + TEMP_OP] * hcap[i]));
        if (res > residual)
            residual = res;
    }
    return residual;
}

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_cpu<NC, NP>::calc_well_residual_L2()
{
    double residual = 0;
    std::vector<value_t> res(n_vars, 0);
    std::vector<value_t> norm(n_vars, 0);

    std::vector<value_t> &hcap = mesh->heat_capacity;

    for (ms_well *w : wells)
    {
        int nperf = w->perforations.size();
        for (int ip = 0; ip < nperf; ip++)
        {
            index_t i_w, i_r;
            value_t wi, wid;
            std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

            for (int c = 0; c < nc; c++)
            {
                res[c] += RHS[(w->well_body_idx + i_w) * n_vars + c] * RHS[(w->well_body_idx + i_w) * n_vars + c];
                norm[c] += PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + c] * PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + c];
            }
            res[E_VAR] += RHS[(w->well_body_idx + i_w) * n_vars + E_VAR] * RHS[(w->well_body_idx + i_w) * n_vars + E_VAR];
            norm[E_VAR] += PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP] * PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP];
        }
        // and then add RHS for well control equations
        for (int c = 0; c < nc; c++)
        {
            // well constraints should not be normalized, so pre-multiply by norm
            res[c] += RHS[w->well_head_idx * n_vars + c] * RHS[w->well_head_idx * n_vars + c] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + c] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + c];
        }
        res[E_VAR] += RHS[(w->well_head_idx) * n_vars + E_VAR] * RHS[(w->well_head_idx) * n_vars + E_VAR] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP] * PV[w->well_body_idx] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP];
    }

    for (int v = 0; v < n_vars; v++)
    {
        residual = std::max(residual, sqrt(res[v] / norm[v]));
    }
    return residual;
}

template <uint8_t NC, uint8_t NP>
double
engine_nce_g_cpu<NC, NP>::calc_well_residual_Linf()
{
    double residual = 0, res = 0;
    std::vector<value_t> &hcap = mesh->heat_capacity;

    for (ms_well *w : wells)
    {
        int nperf = w->perforations.size();
        for (int ip = 0; ip < nperf; ip++)
        {
            index_t i_w, i_r;
            value_t wi, wid;
            std::tie(i_w, i_r, wi, wid) = w->perforations[ip];

            for (int c = 0; c < nc; c++)
            {
                res = fabs(RHS[(w->well_body_idx + i_w) * n_vars + c] / (PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + c]));
                residual = std::max(residual, res);
            }
            res = fabs(RHS[(w->well_body_idx + i_w) * n_vars + E_VAR] / (PV[w->well_body_idx + i_w] * op_vals_arr[w->well_body_idx * N_OPS + FE_ACC_OP]));
            residual = std::max(residual, res);
        }
        // and then add RHS for well control equations
        for (int c = 0; c < nc; c++)
        {
            // well constraints should not be normalized, so pre-multiply by norm
            res = fabs(RHS[w->well_head_idx * n_vars + c]);
            residual = std::max(residual, res);
        }
        res = fabs(RHS[(w->well_head_idx) * n_vars + E_VAR]);
        residual = std::max(residual, res);
    }

    return residual;
}

template <uint8_t NC, uint8_t NP>
int 
engine_nce_g_cpu<NC, NP>::solve_linear_equation()
{
  int r_code;
  char buffer[1024];
  linear_solver_error_last_dt = 0;

  // scaling according to dimensions
  if (scale_dimless)
	make_dimensionless();

  // row-wise scaling
  if (scale_rows)
	dimensionalize_rows<N_VARS>();

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
  }

  if (scale_dimless)
	dimensionalize_unknowns();

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
	sprintf(buffer, "\t #%d (%.4e, %.4e): lin %d (%.1e)\n", n_newton_last_dt + 1, newton_residual_last_dt,
	  well_residual_last_dt, linear_solver->get_n_iters(), linear_solver->get_residual());
	std::cout << buffer << std::flush;
	n_linear_last_dt += linear_solver->get_n_iters();
  }
  return 0;
}

template <uint8_t NC, uint8_t NP>
void 
engine_nce_g_cpu<NC, NP>::make_dimensionless()
{
  const index_t n_blocks = mesh->n_blocks;
  const index_t n_res_blocks = mesh->n_res_blocks;
  value_t* Jac = Jacobian->get_values();
  const index_t* rows = Jacobian->get_rows_ptr();
  const value_t* V = mesh->volume.data();
  index_t csr_idx_start, csr_idx_end;
  value_t mass_dim = m_dim, heat_dim = m_dim * e_dim;

  value_t max_jacobian = 0.0, max_residual = 0.0;
  // value_t min_ratio = std::numeric_limits<value_t>::infinity();
  value_t row_max_jacobian[N_VARS];

  // matrix + fractures
  for (index_t i = 0; i < n_res_blocks; i++)
  {
	// std::fill_n(row_max_jacobian, N_VARS, 0.0);

	csr_idx_start = rows[i];
	csr_idx_end = rows[i + 1];
	for (index_t j = csr_idx_start; j < csr_idx_end; j++)
	{
	  // jacobian (fluid mass)
	  for (uint8_t c = 0; c < NC; c++)
	  {
		// pressure
		Jac[j * N_VARS_SQ + c * N_VARS + P_VAR] /= (mass_dim / p_dim);
		row_max_jacobian[c] = std::max(row_max_jacobian[c], fabs(Jac[j * N_VARS_SQ + c * N_VARS + P_VAR]));
		// composition
		for (uint8_t v = Z_VAR; v < E_VAR; v++)
		{
		  Jac[j * N_VARS_SQ + c * N_VARS + v] /= mass_dim;
		  row_max_jacobian[c] = std::max(row_max_jacobian[c], fabs(Jac[j * N_VARS_SQ + c * N_VARS + v]));
		}
		// enthalpy
		Jac[j * N_VARS_SQ + c * N_VARS + E_VAR] /= (mass_dim / e_dim);
		row_max_jacobian[c] = std::max(row_max_jacobian[c], fabs(Jac[j * N_VARS_SQ + c * N_VARS + E_VAR]));
	  }

	  // jacobian (energy)
	  // pressure
	  Jac[j * N_VARS_SQ + NC * N_VARS + P_VAR] /= (heat_dim / p_dim);
	  row_max_jacobian[NC] = std::max(row_max_jacobian[NC], fabs(Jac[j * N_VARS_SQ + NC * N_VARS + P_VAR]));
	  // composition
	  for (uint8_t v = Z_VAR; v < E_VAR; v++)
	  {
		Jac[j * N_VARS_SQ + NC * N_VARS + v] /= heat_dim;
		row_max_jacobian[NC] = std::max(row_max_jacobian[NC], fabs(Jac[j * N_VARS_SQ + NC * N_VARS + v]));
	  }
	  // enthalpy
	  Jac[j * N_VARS_SQ + NC * N_VARS + E_VAR] /= (heat_dim / e_dim);
	  row_max_jacobian[NC] = std::max(row_max_jacobian[NC], fabs(Jac[j * N_VARS_SQ + NC * N_VARS + E_VAR]));
	}
	// residual
	for (uint8_t c = 0; c < NC; c++)
	{
	  RHS[i * N_VARS + c] /= mass_dim;
	  max_jacobian = std::max(max_jacobian, row_max_jacobian[c]);
	  max_residual = std::max(max_residual, fabs(RHS[i * N_VARS + c]));
	  //if (fabs(RHS[i * N_VARS + c]) > EQUALITY_TOLERANCE)
		//min_ratio = std::min(min_ratio, fabs(RHS[i * N_VARS + c] / row_max_jacobian[c]));
	}
	RHS[i * N_VARS + NC] /= heat_dim;
	max_jacobian = std::max(max_jacobian, row_max_jacobian[NC]);
	max_residual = std::max(max_residual, fabs(RHS[i * N_VARS + NC]));
	//if (fabs(RHS[i * N_VARS + P_VAR]) > EQUALITY_TOLERANCE)
	//  min_ratio = std::min(min_ratio, fabs(RHS[i * N_VARS + P_VAR] / row_max_jacobian[P_VAR]));
  }

  // wells: TODO: add the scaling of well equations
  /*for (ms_well* w : wells)
  {
	if (geomechanics_mode[w->well_body_idx])
	  mass_dim = mass_dim_geom;
	else
	  mass_dim = mass_dim_base;

	// well body
	csr_idx_start = rows[w->well_body_idx];
	csr_idx_end = rows[w->well_body_idx + 1];

	for (index_t j = csr_idx_start; j < csr_idx_end; j++)
	{
	  // jacobian (fluid mass)
	  for (uint8_t v = U_VAR; v < U_VAR + ND_; v++)
	  {
		Jac[j * N_VARS_SQ + P_VAR * N_VARS + v] /= (mass_dim / x_dim);
	  }
	  Jac[j * N_VARS_SQ + P_VAR * N_VARS + P_VAR] /= (mass_dim / p_dim);
	}
	// residual
	RHS[w->well_body_idx * N_VARS + P_VAR] /= (mass_dim);
  }*/

  // printf("max(residual)/max(jacobian) = %e\n", max_residual / max_jacobian);
  // printf("row-wise residual/max(jacobian) = %e\n", min_ratio);
  fflush(stdout);
}

template <uint8_t NC, uint8_t NP>
void 
engine_nce_g_cpu<NC, NP>::dimensionalize_unknowns()
{
  const index_t n_blocks = mesh->n_blocks;
  const index_t n_res_blocks = mesh->n_res_blocks;

  // matrix + fractures
  for (index_t i = 0; i < n_res_blocks; i++)
  {
	// pressure
	dX[i * N_VARS + P_VAR] *= p_dim;
	// enthalpy
	dX[i * N_VARS + E_VAR] *= e_dim;
  }

  // TODO: add well equations
}

template <uint8_t NC, uint8_t NP>
int engine_nce_g_cpu<NC, NP>::adjoint_gradient_assembly(value_t dt, std::vector<value_t>& X, csr_matrix_base* jacobian, std::vector<value_t>& RHS)
{
	index_t n_blocks = mesh->n_blocks;
	index_t n_conns = mesh->n_conns;
	std::vector <index_t> &block_m = mesh->block_m;
	std::vector <index_t> &block_p = mesh->block_p;
	std::vector <value_t> &tran = mesh->tran;
	std::vector <value_t> &tranD = mesh->tranD;
	std::vector <value_t> &hcap = mesh->heat_capacity;
	std::vector <value_t> &grav_coef = mesh->grav_coef;
	std::vector <index_t>& conn_index_to_one_way = mesh->conn_index_to_one_way;


	//std::vector<value_t> sub1(N_VARS * n_blocks, 0);
	//std::vector<value_t> sub2(n_interfaces, 0);

	//Temp_dj_dx.clear();
	//Temp_dj_du.clear();

	//Temp_dj_dx = sub1;
	//Temp_dj_du = sub2;


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
//	//#pragma omp parallel reduction (max: CFL_max)
//#pragma omp parallel
//	{
//		int id = omp_get_thread_num();
//
//		//index_t start = row_thread_starts[id];
//		//index_t end = row_thread_starts[id + 1];
//
//
//		//index_t start = row_T_thread_starts[id];
//		//index_t end = row_T_thread_starts[id + 1];
//#else
//	index_t start = 0;
//	index_t end = n_blocks;
//#endif //_OPENMP


    index_t start = 0;
    index_t end = n_blocks;

	index_t j, diag_idx, jac_idx;
	value_t p_diff, gamma_p_diff;
	value_t t_diff, gamma_t_diff;

	memset(Jac_n, 0, (n_conns + n_blocks) * N_VARS_SQ * sizeof(value_t));
	memset(value_dg_dT, 0, n_conns * N_VARS * sizeof(value_t));


	double value_g_u = 0.0;
	index_t N_element = 0;
	index_t count = 0;

	index_t k_count = 0;
	index_t idx;
	std::vector<index_t> temp_conn_one_way;
	std::vector<index_t> temp_num;
	for (index_t i = start; i < end; i++)
	{

		diag_idx = N_VARS_SQ * diag_ind[i];

		// fill diagonal part
		// [NC] mass eqns
		for (uint8_t c = 0; c < NC; c++)
		{
			for (uint8_t v = 0; v < N_VARS; v++)
			{
				Jac_n[diag_idx + c * N_VARS + v] = -(PV[i] * op_ders_arr[(i * N_OPS + ACC_OP + c) * N_VARS + v]);
			}
		}

		// [1] energy eqn
		// fluid energy 
		for (uint8_t v = 0; v < N_VARS; v++)
		{
			Jac_n[diag_idx + NC * N_VARS + v] = -(PV[i] * op_ders_arr[(i * N_OPS + FE_ACC_OP) * N_VARS + v]);
			Jac_n[diag_idx + NC * N_VARS + v] -= (RV[i] * op_ders_arr[(i * N_OPS + TEMP_OP) * N_VARS + v] * hcap[i]);
		}

		// index of first entry for block i in CSR cols array
		index_t csr_idx_start = rows[i];
		// index of last entry for block i in CSR cols array
		index_t csr_idx_end = rows[i + 1];
		// index of first entry for block i in connection array (has all entries of CSR except diagonals, ordering is identical)
		index_t conn_idx = csr_idx_start - i;

		jac_idx = N_VARS_SQ * csr_idx_start;


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


		// fill offdiagonal part + contribute to diagonal
		for (index_t csr_idx = csr_idx_start; csr_idx < csr_idx_end; csr_idx++, jac_idx += N_VARS_SQ)
		{
			j = cols[csr_idx];
			// skip diagonal
			if (i == j)
				continue;

			p_diff = X[j * N_VARS + P_VAR] - X[i * N_VARS + P_VAR];
			t_diff = op_vals_arr[j * N_OPS + TEMP_OP] - op_vals_arr[i * N_OPS + TEMP_OP];
			gamma_t_diff = tranD[conn_idx] * dt * t_diff;

			for (index_t wh : well_head_idx_collection)
			{
				if (i == wh)
				{
					well_head_tran_idx_collection.push_back(conn_index_to_one_way[conn_idx]);
				}
			}

			for (uint8_t p = 0; p < NP; p++)
			{
				// calculate gravity term for phase p
				value_t avg_density = (op_vals_arr[i * N_OPS + DENS_OP + p] + op_vals_arr[j * N_OPS + DENS_OP + p]) * 0.5;

				value_t phase_p_diff = p_diff + avg_density * grav_coef[conn_idx] * H2O_MW;
				double phase_gamma_p_diff = tran[conn_idx] * dt * phase_p_diff;

				if (phase_p_diff < 0)
				{
					// mass outflow
					for (uint8_t c = 0; c < NC; c++)
					{

						//RHS[i * N_VARS + c] -= phase_gamma_p_diff * op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c]; // flux operators only

						value_g_u = phase_p_diff * dt * op_vals_arr[i * N_OPS + FLUX_OP + p * NC + c];

						idx = count + c * N_element + temp_num[k_count];
						//value_dg_dT[idx] = -value_g_u;
						value_dg_dT[idx] -= value_g_u;

					}

					// energy outflow

					//RHS[i * N_VARS + E_VAR] -= phase_gamma_p_diff * op_vals_arr[i * N_OPS + FE_FLUX_OP + p]; // energy flux

					value_g_u = phase_p_diff * dt * op_vals_arr[i * N_OPS + FE_FLUX_OP + p];

					idx = count + NC * N_element + temp_num[k_count];
					//value_dg_dT[idx] = -value_g_u;
					value_dg_dT[idx] -= value_g_u;

				}
				else
				{
					//inflow

					// mass
					for (uint8_t c = 0; c < NC; c++)
					{

						//RHS[i * N_VARS + c] -= phase_gamma_p_diff * op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c]; // flux operators only

						value_g_u = phase_p_diff * dt * op_vals_arr[j * N_OPS + FLUX_OP + p * NC + c];

						idx = count + c * N_element + temp_num[k_count];
						//value_dg_dT[idx] = -value_g_u;
						value_dg_dT[idx] -= value_g_u;
					}

					// energy flux

					//RHS[i * N_VARS + E_VAR] -= phase_gamma_p_diff * op_vals_arr[j * N_OPS + FE_FLUX_OP + p]; // energy flux operator

					value_g_u = phase_p_diff * dt * op_vals_arr[j * N_OPS + FE_FLUX_OP + p];

					idx = count + NC * N_element + temp_num[k_count];
					//value_dg_dT[idx] = -value_g_u;
					value_dg_dT[idx] -= value_g_u;

				}
			}
			k_count++;

			conn_idx++;


			//set the values of non-diagonal elements to zero
			for (uint8_t c = 0; c < N_VARS; c++)
			{
				for (uint8_t v = 0; v < N_VARS; v++)
				{
					Jac_n[jac_idx + c * N_VARS + v] = 0;
				}
			}
		}

		if (jac_idx == diag_idx)
			jac_idx += N_VARS_SQ;


		count += N_VARS * N_element;
	}



//	value_t CFL_max_local = 0;
//#ifdef _OPENMP
//#pragma omp critical 
//	{
//		if (CFL_max < CFL_max_local)
//			CFL_max = CFL_max_local;
//	}
//	}
//#else
//	CFL_max = CFL_max_local;
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

	//for (ms_well* w : wells)
	//{
	//	w->add_to_jacobian(dt, X, dg_dx_n_temp, RHS);
	//	//w->add_to_jacobian(dt_next, X_next, dg_dx_n_temp, RHS);
	//}

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



template class engine_nce_g_cpu<1, 2>;
//template class engine_nce_g_cpu<2, 2>;
//template class engine_nce_g_cpu<3, 2>;
//template class engine_nce_g_cpu<4, 2>;
//template class engine_nce_g_cpu<5, 2>;
