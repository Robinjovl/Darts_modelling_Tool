#include <iostream>
#include <sstream>
#include <stdexcept>

#include "ms_well.h"

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#else
#include "csr_matrix.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS

ms_well::ms_well()
{
}

// ---------------------------------------------------------------------------
// Perforation flow-law rates: THE single source of truth (review finding R1).
//
// This is the arithmetic of engine_super_cpu::add_perforation_flow_law -- see
// the derivation comment there ([1] total rate, [2] composition, [3] mixture
// properties, [4] basis conversion, [5] component/energy rates). The engine
// calls it with op_ders/drate_* for assembly; ms_well::calc_rates* calls it
// value-only for reporting, so the reported perforation rate IS the assembled
// flux and the two can not drift apart. The runtime loops mirror the engine's
// compile-time loops operation for operation.
// ---------------------------------------------------------------------------
index_t perforation_law_rates(const perforation_flow_law &law,
                              index_t well_block, index_t res_block,
                              const perforation_law_layout &layout,
                              const value_t *X,
                              const value_t *op_vals,
                              const value_t *op_ders,
                              const value_t *cell_spe,
                              value_t *rate,
                              value_t *drate_w_out,
                              value_t *drate_r_out)
{
    const index_t NV = layout.n_vars;
    const index_t NC = layout.nc;
    const index_t NP = layout.np;
    const index_t NE = layout.ne;
    const bool THERMAL = layout.thermal != 0;
    const bool want_der = (drate_w_out != nullptr) && (drate_r_out != nullptr);
    if (want_der && op_ders == nullptr)
        throw std::runtime_error("perforation_law_rates: derivatives were requested without op_ders.");
    if (THERMAL && cell_spe == nullptr)
        throw std::runtime_error("perforation_law_rates: a thermal layout requires cell_spe.");
    if (law.law == perforation_flow_law_type::DARCY)
        throw std::runtime_error("perforation_law_rates: a DARCY perforation has no flow law to evaluate.");

    const index_t wb = well_block;
    const index_t rb = res_block;

    // [1] total rate and its (pressure-only) derivatives
    const value_t q_tot = law.intercept +
                          law.productivity * (X[wb * NV + layout.p_var] - X[rb * NV + layout.p_var] - law.offset);
    const bool up_is_well = (q_tot >= 0.0);
    const index_t up = up_is_well ? wb : rb;

    std::vector<value_t> dq_w(NV, 0.0), dq_r(NV, 0.0);
    dq_w[layout.p_var] = law.productivity;
    dq_r[layout.p_var] = -law.productivity;

    // [2] upstream overall composition from the state, clipped and renormalized
    std::vector<value_t> z(NC, 0.0);
    std::vector<value_t> dz(NC * NV, 0.0);

    if (NC == 1)
    {
        z[0] = 1.0;
    }
    else
    {
        std::vector<value_t> w(NC, 0.0);
        std::vector<value_t> dw(NC * NV, 0.0);

        value_t sum_z = 0.0;
        for (index_t c = 0; c < NC - 1; c++)
        {
            w[c] = X[up * NV + layout.z_var + c];
            dw[c * NV + layout.z_var + c] = 1.0;
            sum_z += w[c];
        }
        w[NC - 1] = 1.0 - sum_z;
        for (index_t c = 0; c < NC - 1; c++)
            dw[(NC - 1) * NV + layout.z_var + c] = -1.0;

        const value_t eps_z = layout.eps_z;
        value_t S = 0.0;
        std::vector<value_t> dS(NV, 0.0);
        for (index_t c = 0; c < NC; c++)
        {
            if (!(w[c] > eps_z))
            {
                w[c] = eps_z;
                for (index_t v = 0; v < NV; v++)
                    dw[c * NV + v] = 0.0;
            }
            S += w[c];
            for (index_t v = 0; v < NV; v++)
                dS[v] += dw[c * NV + v];
        }
        for (index_t c = 0; c < NC; c++)
        {
            z[c] = w[c] / S;
            for (index_t v = 0; v < NV; v++)
                dz[c * NV + v] = (dw[c * NV + v] - z[c] * dS[v]) / S;
        }
    }

    // [3] upstream mixture properties from the interpolated operators.
    // An isothermal MOLAR law needs none of them, and then the flux is an exact
    // function of the state alone.
    const bool need_molar_density = THERMAL || (law.basis != ipr_rate_basis::MOLAR);
    const bool need_molecular_weight = THERMAL || (law.basis == ipr_rate_basis::MASS);

    value_t rho_m = 1.0, Mw = 1.0, h = 0.0;
    std::vector<value_t> drho_m(NV, 0.0), dMw(NV, 0.0), dh(NV, 0.0);

    if (need_molar_density)
    {
        value_t rho_mass = 0.0, rho_h = 0.0;
        std::vector<value_t> drho_mass(NV, 0.0), drho_h(NV, 0.0);
        rho_m = 0.0;

        for (index_t p = 0; p < NP; p++)
        {
            const value_t s_p = op_vals[up * layout.n_ops + layout.sat_op + p];
            const value_t *ds_p = want_der ? &op_ders[(up * layout.n_ops + layout.sat_op + p) * NV] : nullptr;

            value_t f_p = 0.0;
            std::vector<value_t> df_p(NV, 0.0);
            for (index_t c = 0; c < NC; c++)
            {
                f_p += op_vals[up * layout.n_ops + layout.flux_op + p * NE + c];
                if (want_der)
                    for (index_t v = 0; v < NV; v++)
                        df_p[v] += op_ders[(up * layout.n_ops + layout.flux_op + p * NE + c) * NV + v];
            }
            rho_m += s_p * f_p;
            if (want_der)
                for (index_t v = 0; v < NV; v++)
                    drho_m[v] += ds_p[v] * f_p + s_p * df_p[v];

            if (need_molecular_weight)
            {
                const value_t g_p = op_vals[up * layout.n_ops + layout.grav_op + p];
                rho_mass += s_p * g_p;
                if (want_der)
                    for (index_t v = 0; v < NV; v++)
                        drho_mass[v] += ds_p[v] * g_p + s_p * op_ders[(up * layout.n_ops + layout.grav_op + p) * NV + v];
            }

            if (THERMAL)
            {
                const value_t e_p = op_vals[up * layout.n_ops + layout.flux_op + p * NE + NC];
                rho_h += s_p * e_p;
                if (want_der)
                    for (index_t v = 0; v < NV; v++)
                        drho_h[v] += ds_p[v] * e_p + s_p * op_ders[(up * layout.n_ops + layout.flux_op + p * NE + NC) * NV + v];
            }
        }

        if (rho_m > 0.0)
        {
            if (need_molecular_weight)
            {
                Mw = rho_mass / rho_m;
                if (want_der)
                    for (index_t v = 0; v < NV; v++)
                        dMw[v] = (drho_mass[v] - Mw * drho_m[v]) / rho_m;
            }
            if (THERMAL)
            {
                h = rho_h / rho_m;
                if (want_der)
                    for (index_t v = 0; v < NV; v++)
                        dh[v] = (drho_h[v] - h * drho_m[v]) / rho_m;
            }
        }
        else
        {
            // No fluid at the upstream block: nothing can flow through the perforation.
            rho_m = 0.0;
            Mw = 1.0;
            h = 0.0;
            for (index_t v = 0; v < NV; v++)
            {
                drho_m[v] = 0.0;
                dMw[v] = 0.0;
                dh[v] = 0.0;
            }
        }
    }

    // [4] total MOLAR rate and its derivatives w.r.t. both connected blocks
    value_t m_rate = 0.0;
    std::vector<value_t> dm_w(NV, 0.0), dm_r(NV, 0.0);

    switch (law.basis)
    {
    case ipr_rate_basis::MOLAR:
        m_rate = q_tot;
        for (index_t v = 0; v < NV; v++)
        {
            dm_w[v] = dq_w[v];
            dm_r[v] = dq_r[v];
        }
        break;

    case ipr_rate_basis::MASS:
    {
        m_rate = q_tot / Mw;
        value_t *dm_up = up_is_well ? dm_w.data() : dm_r.data();
        for (index_t v = 0; v < NV; v++)
        {
            dm_w[v] = dq_w[v] / Mw;
            dm_r[v] = dq_r[v] / Mw;
        }
        if (want_der)
            for (index_t v = 0; v < NV; v++)
                dm_up[v] -= q_tot * dMw[v] / (Mw * Mw);
        break;
    }

    case ipr_rate_basis::VOLUMETRIC:
    {
        m_rate = q_tot * rho_m;
        value_t *dm_up = up_is_well ? dm_w.data() : dm_r.data();
        for (index_t v = 0; v < NV; v++)
        {
            dm_w[v] = dq_w[v] * rho_m;
            dm_r[v] = dq_r[v] * rho_m;
        }
        if (want_der)
            for (index_t v = 0; v < NV; v++)
                dm_up[v] += q_tot * drho_m[v];
        break;
    }
    }

    // [5] component and energy rates, positive from the well into the reservoir
    for (index_t c = 0; c < NC; c++)
    {
        rate[c] = m_rate * z[c];
        if (want_der)
        {
            for (index_t v = 0; v < NV; v++)
            {
                drate_w_out[c * NV + v] = dm_w[v] * z[c];
                drate_r_out[c * NV + v] = dm_r[v] * z[c];
            }
            value_t *drate_up = up_is_well ? &drate_w_out[c * NV] : &drate_r_out[c * NV];
            for (index_t v = 0; v < NV; v++)
                drate_up[v] += m_rate * dz[c * NV + v];
        }
    }

    if (THERMAL)
    {
        const value_t spe = cell_spe[up];
        const value_t specific = h + spe * Mw;
        rate[NC] = m_rate * specific;
        if (want_der)
        {
            for (index_t v = 0; v < NV; v++)
            {
                drate_w_out[NC * NV + v] = dm_w[v] * specific;
                drate_r_out[NC * NV + v] = dm_r[v] * specific;
            }
            value_t *drate_up = up_is_well ? &drate_w_out[NC * NV] : &drate_r_out[NC * NV];
            for (index_t v = 0; v < NV; v++)
                drate_up[v] += m_rate * (dh[v] + spe * dMw[v]);
        }
    }

    return up;
}

void ms_well::init_physics(int n_vars_, int n_ops_, std::vector<std::string> phase_names_,
    operator_set_gradient_evaluator_iface* well_ctrl_etor_, operator_set_gradient_evaluator_iface* thermal_var_etor_,
    int thermal_)
{
    n_block_size = n_vars_;
    P_VAR = 0;
    n_vars = n_vars_;
    n_ops = n_ops_;
    n_phases = int(phase_names_.size());
    phase_names = phase_names_;
    thermal = thermal_;

    state.resize(n_vars);
    state_neighbour.resize(n_vars);

    control = well_control_iface(n_phases, n_vars - thermal, thermal, well_ctrl_etor_, thermal_var_etor_);
    constraint = well_control_iface(n_phases, n_vars - thermal, thermal, well_ctrl_etor_, thermal_var_etor_);

    // Store well control evaluators for calculation of wellhead and perforation rates and adjoint
    well_ctrl_ops.resize(control.get_n_well_ctrl_ops());
    well_ctrl_etor = well_ctrl_etor_;
    well_ctrl_etor_ad = well_ctrl_etor_;  // adjoint method
}

void ms_well::init_mech_physics(uint8_t N_VARS_, uint8_t P_VAR_, int n_vars_, int n_ops_, std::vector<std::string> phase_names_,
    operator_set_gradient_evaluator_iface* well_ctrl_etor_, operator_set_gradient_evaluator_iface* thermal_var_etor_,
    int thermal_)
{
    n_block_size = N_VARS_;
    P_VAR = P_VAR_;
    n_vars = n_vars_;
    n_ops = n_ops_;
    n_phases = int(phase_names_.size());
    phase_names = phase_names_;
    thermal = thermal_;

    state.resize(n_vars);
    state_neighbour.resize(n_vars);

    control = well_control_iface(n_phases, n_vars - thermal, thermal, well_ctrl_etor_, thermal_var_etor_);
    constraint = well_control_iface(n_phases, n_vars - thermal, thermal, well_ctrl_etor_, thermal_var_etor_);

    // Store well control evaluators for calculation of wellhead and perforation rates and adjoint
    well_ctrl_ops.resize(control.get_n_well_ctrl_ops());
    well_ctrl_etor = well_ctrl_etor_;
    well_ctrl_etor_ad = well_ctrl_etor_;  // adjoint method
}

int ms_well::initialize_control_epm(std::vector<value_t>& X)
{
    if (control.get_well_control_type() == well_control_iface::WellControlType::NONE)
    {
        std::cout << "Well " << name << " has uninitialized well control\n";
        exit(1);
    }
    std::cout << "Well " << name << " initialized with " << control.get_well_control_type_str() << std::endl;

    // Initialize state in well blocks for each perforation - state neighbour is reservoir cell, state is well block
    for (auto& p : perforations)
    {
        index_t i_w, i_r;
        value_t wi, wid;
        std::tie(i_w, i_r, wi, wid) = p;
        i_w += well_body_idx;

        // move the state from X
        std::move(X.begin() + i_w * n_block_size + P_VAR, X.begin() + i_w * n_block_size + P_VAR + n_vars, state.begin());
        // copy neighbour state
        std::copy(X.begin() + i_r * n_block_size + P_VAR, X.begin() + i_r * n_block_size + P_VAR + n_vars, state_neighbour.begin());
        // initialize
        control.initialize_well_block(state, state_neighbour, false);
        // move initialized state back to X
        std::move(state.begin(), state.end(), X.begin() + i_w * n_block_size + P_VAR);
    }
    // Initialize state in well head - state neighbour is well body, state is well head
    // move the state from X
    std::move(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_vars, state.begin());
    // copy neighbour state
    std::copy(X.begin() + well_body_idx * n_block_size + P_VAR, X.begin() + well_body_idx * n_block_size + P_VAR + n_vars, state_neighbour.begin());
    // initialize
    control.initialize_well_block(state, state_neighbour, false);
    // move initialized state back to X
    std::move(state.begin(), state.end(), X.begin() + well_head_idx * n_block_size + P_VAR);
    return 0;
}

int ms_well::initialize_control_dfm(std::vector<value_t>& X)
{
    if (control.get_well_control_type() == well_control_iface::WellControlType::NONE)
    {
        std::cout << "Well " << name << " has uninitialized well control\n";
        exit(1);
    }
    std::cout << "Well " << name << " initialized with " << control.get_well_control_type_str() << std::endl;

    // Initialize all the well blocks. Wellhead state will be overwritten later.
    std::copy(init_state.begin(), init_state.end(), X.begin() + well_head_idx * n_block_size);

    // Initialize state in well head - state neighbour is well body, state is well head
    // move the state from X
    std::move(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_vars, state.begin());
    // copy neighbour state
    std::copy(X.begin() + well_body_idx * n_block_size + P_VAR, X.begin() + well_body_idx * n_block_size + P_VAR + n_vars, state_neighbour.begin());
    // initialize
    control.initialize_well_block(state, state_neighbour, true);
    // move initialized state back to X
    std::move(state.begin(), state.end(), X.begin() + well_head_idx * n_block_size + P_VAR);
    return 0;
}

int ms_well::check_constraints(double dt, std::vector<value_t>& X)
{
    if (constraint.get_well_control_type() > well_control_iface::WellControlType::NONE)
    {
        if (ms_type == ms_well::MS_Type::DFM)
            throw std::runtime_error("DFM wells do not support well constraints yet!");

        if (constraint.check_constraint_violation(dt, well_head_idx, well_transmissibility, n_block_size, P_VAR, X))
        {
            // constraint violation occured, switch control and constrain
            std::swap(control, constraint);
            std::cout << "Well " << name << " switched to " << control.get_well_control_type_str() << " (target: " << control.get_well_control_target_str() << ")\n";
            //initialize_control_epm(X);
        }
    }

    return 0;
}

int ms_well::add_to_jacobian(double dt, std::vector<value_t>& X, value_t* jac_well_head, std::vector<value_t>& RHS)
{
    const bool is_dfm_well = ms_type == ms_well::MS_Type::DFM;
    control.add_to_jacobian(dt, well_head_idx, well_transmissibility, n_block_size, P_VAR, X, jac_well_head, RHS,
        phases_vels, phases_vels_ders, is_dfm_well);

    return 0;
}

int ms_well::calc_rates(std::vector<value_t>& X, std::vector<value_t>& op_vals_arr, std::unordered_map<std::string, std::vector<value_t>>& time_data)
{
    index_t upstream_idx;

    // find upstream state
    value_t p_diff = X[well_head_idx * n_block_size + P_VAR] - X[well_body_idx * n_block_size + P_VAR];
    if (p_diff > 0)
        upstream_idx = well_head_idx; // injector
    else
        upstream_idx = well_body_idx; // producer

    state.assign(X.begin() + upstream_idx * n_block_size + P_VAR, X.begin() + upstream_idx * n_block_size + P_VAR + n_vars);
    state.insert(state.end(), Xhistory_well_default.begin(), Xhistory_well_default.end());

    well_ctrl_etor->evaluate(state, well_ctrl_ops);

    // Energy and volumetric rates
    value_t total_energy = 0.;
    for (int i = 0; i < n_phases; i++)
    {
        time_data[name + " : " + phase_names[i] + " rate (m3/day)"].push_back(well_ctrl_ops[control.get_rate_ctrl_op_idx(well_control_iface::VOLUMETRIC_RATE, i, false)] * p_diff * well_transmissibility);
        total_energy += well_ctrl_ops[control.get_rate_ctrl_op_idx(well_control_iface::ADVECTIVE_HEAT_RATE, i, false)] * p_diff * well_transmissibility;
    }
    time_data[name + " : energy (kJ/day)"].push_back(total_energy);

    // Component molar rates
    index_t nc = n_vars - thermal;
    for (index_t c = 0; c < nc; c++)
    {
        double c_rate_op = 0;

        for (int j = 0; j < n_phases; j++)
        {
            int shift = n_block_size + n_block_size * j;
            c_rate_op += op_vals_arr[upstream_idx * n_ops + shift + c];
        }

        time_data[name + " : c " + std::to_string(c) + " rate (Kmol/day)"].push_back(c_rate_op * p_diff * well_transmissibility);
    }

    int i_p = 0;

    for (auto& p : perforations)
    {
        index_t i_w, i_r;
        value_t wi, wid;
        std::tie(i_w, i_r, wi, wid) = p;
        i_w += well_body_idx;

        const perforation_flow_law &law = get_perforation_flow_law(i_p);
        if (law.law != perforation_flow_law_type::DARCY)
        {
            // Law-aware reporting (review finding R1): a non-Darcy perforation
            // has a ZERO well index by construction, so the Darcy `p_diff * wi`
            // below would export exactly 0.0 while the assembled law flux is
            // not. Report the law flux with the SAME arithmetic assembly uses.
            calc_perforation_law_rates(law, i_p, i_w, i_r, X, op_vals_arr, time_data);
        }
        else
        {
            // find upstream for the perforation
            value_t p_diff = X[i_w * n_block_size + P_VAR] - X[i_r * n_block_size + P_VAR];
            if (p_diff > 0)
                upstream_idx = i_w; // injection perforation
            else
                upstream_idx = i_r; // production perforation

            for (index_t c = 0; c < nc; c++)
            {
                double c_rate_op = 0;

                for (int j = 0; j < n_phases; j++)
                {
                    int shift = nc + nc * j;
                    c_rate_op += op_vals_arr[upstream_idx * n_ops + shift + c];
                }
                time_data[name + " : p " + std::to_string(i_p) + " c " + std::to_string(c) + " rate (Kmol/day)"].push_back(c_rate_op * p_diff * wi);
            }
        }
        time_data[name + " : p " + std::to_string(i_p) + " reservoir P (bar)"].push_back(X[i_r * n_block_size + P_VAR]);

        i_p++;
    }

    // BHP and temperature
    time_data[name + " : BHP (bar)"].push_back(X[well_head_idx * n_block_size + P_VAR]);
    time_data[name + " : temperature (K)"].push_back(well_ctrl_ops[control.get_temp_ctrl_op_idx()]);

    return 0;
}

void ms_well::calc_perforation_law_rates(const perforation_flow_law &law, index_t i_p, index_t i_w, index_t i_r,
    std::vector<value_t>& X, std::vector<value_t>& op_vals_arr,
    std::unordered_map<std::string, std::vector<value_t>>& time_data)
{
    if (!law_layout_set)
    {
        std::ostringstream msg;
        msg << "Well '" << name << "': perforation " << i_p << " carries a non-Darcy flow law "
            << "but the engine never provided the operator-table layout for law-aware rate "
            << "reporting (ms_well::law_layout). Reporting the Darcy p_diff * wi instead "
            << "would silently export a zero rate.";
        throw std::runtime_error(msg.str());
    }

    std::vector<value_t> law_rate(law_layout.ne, 0.0);
    perforation_law_rates(law, i_w, i_r, law_layout, X.data(), op_vals_arr.data(), nullptr,
                          (law_cell_spe && !law_cell_spe->empty()) ? law_cell_spe->data() : nullptr,
                          law_rate.data());

    // Same sign convention as the Darcy branch: positive = injection
    // (from the well into the reservoir), which is the law's own convention.
    const index_t nc_fl = n_vars - thermal;
    for (index_t c = 0; c < nc_fl; c++)
        time_data[name + " : p " + std::to_string(i_p) + " c " + std::to_string(c) + " rate (Kmol/day)"].push_back(law_rate[c]);
}

int ms_well::calc_rates_velocity(std::vector<value_t>& X, std::vector<value_t>& op_vals_arr, std::unordered_map<std::string, std::vector<value_t>>& time_data, index_t n_blocks)
{
    // calculate rate based on velocity unknown; use for decouple velocity engine.
    const bool is_dfm_well = ms_type == ms_well::MS_Type::DFM;

    index_t upstream_idx;

    // find the wellhead connection
    value_t velocity = X[n_block_size * n_blocks + well_head_conn_idx];


    // find upstream state
    value_t p_diff = X[well_head_idx * n_block_size + P_VAR] - X[well_body_idx * n_block_size + P_VAR];
    if (velocity > 0)
        upstream_idx = well_head_idx; // injector
    else
        upstream_idx = well_body_idx; // producer

    state.assign(X.begin() + upstream_idx * n_block_size + P_VAR, X.begin() + upstream_idx * n_block_size + P_VAR + n_vars);
    state.insert(state.end(), Xhistory_well_default.begin(), Xhistory_well_default.end());

    well_ctrl_etor->evaluate(state, well_ctrl_ops);

    // Energy and volumetric rates
    value_t total_energy = 0.;
    for (int i = 0; i < n_phases; i++)
    {
        time_data[name + " : " + phase_names[i] + " rate (m3/day)"].push_back(well_ctrl_ops[control.get_rate_ctrl_op_idx(well_control_iface::VOLUMETRIC_RATE, i, is_dfm_well)] * velocity);
        total_energy += well_ctrl_ops[control.get_rate_ctrl_op_idx(well_control_iface::ADVECTIVE_HEAT_RATE, i, is_dfm_well)] * p_diff * well_transmissibility;
    }
    time_data[name + " : energy (kJ/day)"].push_back(total_energy);

    // Component molar rates
    index_t nc = n_vars - thermal;
    for (index_t c = 0; c < nc; c++)
    {
        double c_rate_op = 0;

        for (int j = 0; j < n_phases; j++)
        {
            index_t shift = n_block_size + n_block_size * j;
            c_rate_op += op_vals_arr[upstream_idx * n_ops + shift + c];
        }

        time_data[name + " : c " + std::to_string(c) + " rate (Kmol/day)"].push_back(c_rate_op * p_diff * well_transmissibility);
    }

    index_t i_p = 0;

    for (auto& p : perforations)
    {
        index_t i_w, i_r;
        value_t wi, wid;
        std::tie(i_w, i_r, wi, wid) = p;
        i_w += well_body_idx;

        const perforation_flow_law &law = get_perforation_flow_law(i_p);
        if (law.law != perforation_flow_law_type::DARCY)
        {
            // Law-aware reporting (review finding R1) -- see calc_rates().
            calc_perforation_law_rates(law, i_p, i_w, i_r, X, op_vals_arr, time_data);
        }
        else
        {
            // find upstream for the perforation
            value_t p_diff = X[i_w * n_vars] - X[i_r * n_vars];
            if (p_diff > 0)
                upstream_idx = i_w; // injection perforation
            else
                upstream_idx = i_r; // production perforation

            for (index_t c = 0; c < nc; c++)
            {
                double c_rate_op = 0;

                for (int j = 0; j < n_phases; j++)
                {
                    index_t shift = nc + nc * j;
                    c_rate_op += op_vals_arr[upstream_idx * n_ops + shift + c];
                }
                time_data[name + " : p " + std::to_string(i_p) + " c " + std::to_string(c) + " rate (Kmol/day)"].push_back(c_rate_op * p_diff * wi);
            }
        }
        time_data[name + " : p " + std::to_string(i_p) + " reservoir P (bar)"].push_back(X[i_r * n_vars]);

        i_p++;
    }

    // BHP and temperature
    time_data[name + " : BHP (bar)"].push_back(X[well_head_idx * n_vars + P_VAR]);
    time_data[name + " : temperature (K)"].push_back(well_ctrl_ops[control.get_temp_ctrl_op_idx()]);

    return 0;
}

void ms_well::addSegment()
{
    double PI = 3.141592;
    for (index_t p = 0; p < n_segments + 1; p++)
    {
        //
        segment s;
        s.diameter = segment_diameter;
        s.length = segment_depth_increment;
        s.area = PI * (s.diameter * s.diameter) / 4;
        s.volume = s.length * s.area;  // volume of the segment
        segments.push_back(s);
    }
}

int ms_well::cross_flow(std::vector<value_t>& X)
{
    /*
    1. check if the well is producer or injector [ based on the name of the well ]
    2. check whether cross-flow happens or not for the given peforation. if it happends  print it out .
    */
    bool is_producer = isProducer();
    for (auto& p : perforations)
    {
        index_t i_w, i_r;
        value_t wi, wid;
        std::tie(i_w, i_r, wi, wid) = p;
        value_t potential_diff = X[(i_w + well_head_idx + 1) * n_block_size + P_VAR] - X[i_r * n_block_size + P_VAR];
        bool is_cross_flow = (is_producer && potential_diff > 0) || (!(is_producer) && potential_diff < 0);
        if (is_cross_flow)
        {
            std::cout << "Cross-flow happens for the well " << name << " for this iteration \n";
        }

    }

    return 0;
}

void ms_well::set_perforation_flow_law(index_t perforation_index, const perforation_flow_law &law)
{
    if (perforation_index < 0 || (size_t)perforation_index >= perforations.size())
    {
        std::ostringstream msg;
        msg << "Well '" << name << "': perforation index " << perforation_index
            << " is out of bounds (" << perforations.size() << " perforations). "
            << "Attach the flow law after the perforation has been added.";
        throw std::runtime_error(msg.str());
    }

    if (law.law != perforation_flow_law_type::DARCY)
    {
        const value_t wi = std::get<2>(perforations[perforation_index]);
        if (wi != 0.0)
        {
            std::ostringstream msg;
            msg << "Well '" << name << "': perforation " << perforation_index
                << " has a non-zero well index (WI=" << wi << ") and a non-Darcy flow law. "
                << "The engine would assemble the Peaceman flux across the very interface "
                << "the flow law carries, double-counting it. Pass well_index=0.0 to "
                << "add_perforation() when giving that perforation a flow law.";
            throw std::runtime_error(msg.str());
        }
        if (law.productivity < 0.0)
        {
            std::ostringstream msg;
            msg << "Well '" << name << "': perforation " << perforation_index
                << " has a negative productivity index (" << law.productivity
                << "). A negative productivity is an unconditionally unstable "
                << "anti-physical feedback (the flux grows with the pressure "
                << "difference it opposes).";
            throw std::runtime_error(msg.str());
        }
    }

    if (perforation_flow_laws.size() < perforations.size())
        perforation_flow_laws.resize(perforations.size());
    perforation_flow_laws[perforation_index] = law;
}

const perforation_flow_law &ms_well::get_perforation_flow_law(index_t perforation_index) const
{
    static const perforation_flow_law darcy_default;
    if (perforation_index < 0 || (size_t)perforation_index >= perforation_flow_laws.size())
        return darcy_default;
    return perforation_flow_laws[perforation_index];
}

bool ms_well::has_non_darcy_perforation() const
{
    for (const perforation_flow_law &law : perforation_flow_laws)
        if (law.law != perforation_flow_law_type::DARCY)
            return true;
    return false;
}
