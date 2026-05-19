#ifndef WELL_CONTROLS_H
#define WELL_CONTROLS_H

#include <algorithm>
#include <cmath>
#include <vector>
#include <optional>
#include "globals.h"
#include "evaluator_iface.h"

/*

 A well control assumed to fill one (blocked) row of jacobian for the well head block.
 The well head block has exactly one connection - to the well body block
 The well head block always has greater index that well body block, therefore
 in the jacobian row the first block stands for well body variables, the second is the diagonal block

 If the base well control class is to be exposed in python, then there is the following strategy choice:
 1. WELL_CONTROL_COPY: A control will receive and fill a small vector, which is then to be copied to big jacobian
    Pros: easy indexing
    Cons: since this is the base class agreement, even C++ controls have to copy to jacobian
    Name:
 2. WELL_CONTROL_FILL: A  control will receive the entire Jacobian, and fill it directly in
    Pros: no excess copy, if Jacobian is based on STL containers, which it should be
    Cons: complex indexing

*/

class well_control_iface
{
public:
    // MOLAR_RATE is 0 because it is the first rate ctrl operator type in the WellCtrlOperators
    enum WellControlType : int
    {
        NONE = -2,
        BHP,
        MOLAR_RATE,
        MASS_RATE,
        VOLUMETRIC_RATE,
        ADVECTIVE_HEAT_RATE,
        NUMBER_OF_RATE_TYPES
    };

    static const int n_state_ctrls = 2;  // pressure and temperature operators
    static const int n_well_ctrl_models = 2;  // EPM and DFM rate ctrl operator families

protected:
    WellControlType control_type = WellControlType::NONE;
    std::optional<index_t> phase_idx = std::nullopt;
    index_t n_phases, n_comps, thermal, n_vars, n_well_ctrl_ops, well_state_offset;
    value_t target, inj_temp;
    std::vector<value_t> inj_comp;
    std::vector<index_t> block_idx{ 0 };
    std::vector<value_t> state;
    std::vector<value_t> well_ctrl_ops;
    std::vector<value_t> well_ctrl_ops_derivs;
    operator_set_gradient_evaluator_iface* well_ctrl_etor, * thermal_var_etor;

    index_t epm_rate_ctrl_ops_offset() const { return 0; }
    index_t state_ctrl_ops_offset() const { return WellControlType::NUMBER_OF_RATE_TYPES * n_phases; }
    index_t dfm_rate_ctrl_ops_offset() const { return state_ctrl_ops_offset() + n_state_ctrls; }

public:
    well_control_iface() {}
    well_control_iface(index_t n_phases_, index_t n_comps_, bool thermal_, operator_set_gradient_evaluator_iface* well_ctrl_etor_,
        operator_set_gradient_evaluator_iface* thermal_var_etor_)
        : n_phases(n_phases_), n_comps(n_comps_), thermal(thermal_), well_ctrl_etor(well_ctrl_etor_), thermal_var_etor(thermal_var_etor_)
    {
        // Evaluate well control operators
        // WellCtrlOperators are defined as follows:
        // NP EPM MOLAR_RATE, NP EPM MASS_RATE, NP EPM VOLUMETRIC_RATE, NP EPM ADVECTIVE_HEAT_RATE ctrl operators,
        // P, T, then NP DFM MOLAR_RATE, NP DFM MASS_RATE, NP DFM VOLUMETRIC_RATE, NP DFM ADVECTIVE_HEAT_RATE ctrl operators.
        n_vars = n_comps + thermal;
        // The logical well ctrl layout may be padded by a fallback interpolator with a larger compiled N_OPS.
        const index_t n_logical_well_ctrl_ops =
            well_control_iface::n_well_ctrl_models * WellControlType::NUMBER_OF_RATE_TYPES * n_phases
            + well_control_iface::n_state_ctrls;
        const index_t n_itor_well_ctrl_ops = well_ctrl_etor_ ? well_ctrl_etor_->get_n_ops() : 0;
        n_well_ctrl_ops = n_itor_well_ctrl_ops > n_logical_well_ctrl_ops
            ? n_itor_well_ctrl_ops
            : n_logical_well_ctrl_ops;
        well_ctrl_ops.resize(n_well_ctrl_ops);
        well_ctrl_ops_derivs.resize(n_well_ctrl_ops * n_vars);
    }

    virtual int set_bhp_control(bool is_inj, value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_);
    virtual int set_rate_control(bool is_inj, well_control_iface::WellControlType control_type_, std::optional<index_t> phase_idx_,
        value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_);

    WellControlType get_well_control_type() { return this->control_type; }
    std::string get_well_control_type_str();
    value_t get_target() const { return this->target; }
    std::string get_well_control_target_str();

    bool is_rate_control() const { return this->control_type > WellControlType::BHP && this->control_type < WellControlType::NUMBER_OF_RATE_TYPES; }
    value_t get_rate_ctrl_residual_scale(value_t absolute_scale, value_t relative_scale) const
    {
        if (!this->is_rate_control())
        {
            return 1.0;
        }
        const value_t abs_scale = std::max(static_cast<value_t>(0.0), absolute_scale);
        const value_t rel_scale = std::max(static_cast<value_t>(0.0), relative_scale);
        return std::max(static_cast<value_t>(1.0e-30), std::max(abs_scale, std::fabs(this->target) * rel_scale));
    }

    index_t get_n_well_ctrl_ops() { return this->n_well_ctrl_ops; }
    index_t get_rate_ctrl_op_idx(WellControlType ctrl_type, index_t phase_idx_, bool is_dfm_well) const;
    index_t get_pres_ctrl_op_idx() const { return state_ctrl_ops_offset(); }
    index_t get_temp_ctrl_op_idx() const { return state_ctrl_ops_offset() + 1; }

    virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour, bool is_dfm_well);

    virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t well_transmissibility,
        uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X);

    virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t well_transmissibility,
        uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t* jacobian_row, std::vector<value_t>& RHS,
        std::vector<value_t>& phases_vels, std::vector<value_t>& phases_vels_ders, bool is_dfm_well);
};

#endif
