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
        time_data[name + " : p " + std::to_string(i_p) + " reservoir P (bar)"].push_back(X[i_r * n_block_size + P_VAR]);

        i_p++;
    }

    // BHP and temperature
    time_data[name + " : BHP (bar)"].push_back(X[well_head_idx * n_block_size + P_VAR]);
    time_data[name + " : temperature (K)"].push_back(well_ctrl_ops[control.get_temp_ctrl_op_idx()]);

    return 0;
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
