#ifndef MS_WELL_H
#define MS_WELL_H

#include <vector>
#include <tuple>
#include <unordered_map>

#include "globals.h"
#include "well_controls.h"
#include "evaluator_iface.h"

// Does not seem to be needed
// class csr_matrix_base;

struct segment
{
    double diameter = 0;
    double length = 0;
    double roughness = 0;
    //SegmentType type;
    double area = 0;
    double volume = 0;
    double diameter_annulus = 0;
    std::vector<std::tuple<index_t, index_t, value_t>> perforations;
};


/// @brief Flow law of a single well perforation: HOW the flux across it is computed.
///
/// The default, DARCY, is the Peaceman/Darcy flux the engine has always assembled
/// from the perforation well index. LINEAR_IPR replaces it with a linear inflow
/// performance relation evaluated by the well assembler, with analytic derivatives
/// with respect to both connected blocks.
enum class perforation_flow_law_type : int
{
    DARCY = 0,      ///< Peaceman/Darcy flux from the perforation well index (default)
    LINEAR_IPR = 1  ///< q_total = intercept + productivity * (p_well - p_res - offset)
};

/// @brief Basis in which the total rate of a LINEAR_IPR perforation is expressed.
enum class ipr_rate_basis : int
{
    MOLAR = 0,      ///< kmol/day      (productivity in kmol/day/bar)
    MASS = 1,       ///< kg/day        (productivity in kg/day/bar)
    VOLUMETRIC = 2  ///< m3/day at upstream in-situ conditions (m3/day/bar)
};

/// @brief Parameters of the flow law of one perforation.
///
/// Only the LINEAR_IPR fields are carried; a DARCY perforation keeps using its
/// well index and ignores all of them.
struct perforation_flow_law
{
    perforation_flow_law_type law = perforation_flow_law_type::DARCY;
    ipr_rate_basis basis = ipr_rate_basis::MOLAR;
    value_t productivity = 0.0;  ///< B, per bar of drawdown, in the rate units of `basis`
    value_t offset = 0.0;        ///< dp [bar]
    value_t intercept = 0.0;     ///< A, in the rate units of `basis`
};

/// Class for a multi-segment well
class ms_well
{
public:

    enum class WellType : int
    {
        PRODUCER = -1,
        INJECTOR = 1
    };

    enum class MS_Type : int
    {
        EPM,
        DFM
    };

    ms_well();

    void init_physics(int n_vars_, int n_ops_, std::vector<std::string> phase_names_,
        operator_set_gradient_evaluator_iface* well_ctrl_etor_, operator_set_gradient_evaluator_iface* thermal_var_etor_,
        int thermal_ = 0);

    void init_mech_physics(uint8_t N_VARS_, uint8_t P_VAR_, int n_vars_, int n_ops_, std::vector<std::string> phase_names_,
        operator_set_gradient_evaluator_iface* well_ctrl_etor_, operator_set_gradient_evaluator_iface* thermal_var_etor_,
        int thermal_ = 0);

    // the function changes (overwrites) jacobian equations for well_head_idx block
    // since well_head_idx has exactly 1 connection, it is assumed that
    // jac_well_head argument points to 2*n_vars*n_vars array of type value_t
    // first n_vars*n_vars correspond to diagonal block (well_head_idx>well_body_idx always)
    // second n_vars*n_vars correspond to offdiagonal
    // X and RHS vector are passed in full (yet)
    void set_bhp_control(bool is_inj, value_t target, std::vector<value_t>& inj_comp, value_t inj_temp)
    {
        this->control.set_bhp_control(is_inj, target, inj_comp, inj_temp);
    }
    void set_bhp_constraint(bool is_inj, value_t target, std::vector<value_t>& inj_comp, value_t inj_temp)
    {
        this->constraint.set_bhp_control(is_inj, target, inj_comp, inj_temp);
    }
    void set_rate_control(bool is_inj, well_control_iface::WellControlType control_type, index_t phase_idx,
        value_t target, std::vector<value_t>& inj_comp, value_t inj_temp)
    {
        this->control.set_rate_control(is_inj, control_type, phase_idx, target, inj_comp, inj_temp);
    }
    void set_rate_constraint(bool is_inj, well_control_iface::WellControlType control_type, index_t phase_idx,
        value_t target, std::vector<value_t>& inj_comp, value_t inj_temp)
    {
        this->constraint.set_rate_control(is_inj, control_type, phase_idx, target, inj_comp, inj_temp);
    }

    int initialize_control_epm(std::vector<value_t>& X);
    int initialize_control_dfm(std::vector<value_t>& X);

    int check_constraints(double dt, std::vector<value_t>& X);

    int add_to_jacobian(double dt, std::vector<value_t>& X, value_t* jac_well_head, std::vector<value_t>& RHS);

    int calc_rates(std::vector<value_t>& X, std::vector<value_t>& op_vals_arr, std::unordered_map<std::string, std::vector<value_t>>& time_data);

    int calc_rates_velocity(std::vector<value_t>& X, std::vector<value_t>& op_vals_arr, std::unordered_map<std::string, std::vector<value_t>>& time_data, index_t n_blocks);

    void addSegment();

    bool isProducer() const { return (well_type == WellType::PRODUCER); }

    int cross_flow(std::vector<value_t>& X);

    /// @brief Attach a flow law to one perforation (the perforation must already exist).
    /// A non-DARCY law requires a zero well index: the engine would otherwise assemble
    /// the Peaceman flux across the very interface the law carries.
    void set_perforation_flow_law(index_t perforation_index, const perforation_flow_law &law);

    /// @brief Flow law of one perforation; DARCY when none was attached.
    const perforation_flow_law &get_perforation_flow_law(index_t perforation_index) const;

    /// @brief Whether any perforation of this well carries a non-DARCY flow law.
    bool has_non_darcy_perforation() const;

    std::string name;
    MS_Type ms_type;
    WellType well_type; // type to be producer or injector
    // segment data type
    std::vector<segment> segments;

    int n_segments = -1;
    value_t segment_volume;
    value_t well_transmissibility;
    value_t well_head_depth;
    value_t well_body_depth;
    value_t segment_depth_increment;
    value_t segment_diameter;
    value_t segment_roughness;

    // Only used for DFM wells; num_segments stays 0 for EPM wells so Python
    // code iterating over wells reads a deterministic value instead of garbage
    std::vector<value_t> segment_volumes;
    std::vector<value_t> segment_depths;
    index_t num_segments = 0;

    index_t well_head_idx;        // index of the wellhead segment, where well controls apply
    index_t well_body_idx;        // index of the well segment right below the wellhead segment
    index_t well_bottom_idx;      // index of the well bottom segment
    index_t well_head_conn_idx;   // index of the connection between the two well segments at the top of the well (for EPM wells, connection is between the ghost segment and the lower segment)

    std::vector<std::tuple<index_t, index_t, value_t, value_t>> perforations;
    /// Per-perforation flow law, parallel to `perforations`. Perforations beyond
    /// its size are DARCY, so a well that never sets one costs nothing.
    std::vector<perforation_flow_law> perforation_flow_laws;
    bool with_lateral_heat_transfer = false;   // only used for a DFM well. If true, lateral heat transfer between the DFM well segments and reservoir blocks is considered.
    std::vector<std::tuple<index_t, index_t, value_t>> connections_for_lateral_heat_transfer; // tuple of (dfm_segment_index, reservoir_block_index, geometric_part_of_the_heat_transfer_equation)

    well_control_iface control;
    well_control_iface constraint;

    std::vector<value_t> init_state;

    std::vector<value_t> phases_vels;        // phases velocities used for a DFM well
    std::vector<value_t> phases_vels_ders;   // phases velocities derivatives used for a DFM well

    std::vector<value_t> well_ctrl_ops;
    operator_set_evaluator_iface* well_ctrl_etor;
    operator_set_gradient_evaluator_iface* well_ctrl_etor_ad;  //adjoint method

    std::vector<value_t> state;
    std::vector<value_t> state_neighbour;
    std::vector<value_t> rates;
    // History values appended to the well state when the physics uses OBL history variables
    // (analogous to mesh->Xhistory_bounds for boundary cells). Empty unless history axes are active.
    std::vector<value_t> Xhistory_well_default;

    // n_block_size -- size of the full block, P_VAR -- index of the start of the state variables within block
    uint8_t n_block_size, P_VAR;
    int n_vars;
    int n_ops;
    int n_phases;
    std::vector<std::string> phase_names;
    int thermal;
};

#endif /* MS_WELL_H */
