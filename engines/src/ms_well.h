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

/// @brief State/operator-table layout the perforation flow-law arithmetic reads.
///
/// The law converts its total rate with upstream mixture properties taken from
/// the assembling engine's interpolated operator table, so the arithmetic needs
/// that engine's layout constants. The engine that assembles the laws fills one
/// of these per law-carrying well at init (engine_super_cpu::init), which is
/// what lets ms_well::calc_rates* report the law flux with the SAME arithmetic
/// the assembly uses (perforation_law_rates) instead of the Darcy `p_diff * wi`
/// -- which is identically zero for a non-Darcy perforation, whose well index
/// is zero by construction.
struct perforation_law_layout
{
    index_t n_vars = 0;    ///< unknowns per block (N_VARS)
    index_t p_var = 0;     ///< pressure index within the block (P_VAR)
    index_t z_var = 1;     ///< first composition index within the block (Z_VAR)
    index_t nc = 0;        ///< number of components (NC)
    index_t np = 0;        ///< number of phases (NP)
    index_t ne = 0;        ///< number of equations (NE = NC + thermal)
    index_t n_ops = 0;     ///< operators per block (N_OPS)
    index_t sat_op = 0;    ///< offset of the phase saturation operators (SAT_OP)
    index_t flux_op = 0;   ///< offset of the phase flux operators (FLUX_OP)
    index_t grav_op = 0;   ///< offset of the phase mass-density operators (GRAV_OP)
    int thermal = 0;       ///< 1 when the physics carries the energy equation
    value_t eps_z = 1e-12; ///< composition clip (params->sim_eps at init)
};

/// @brief Component (and energy) rates of one non-Darcy perforation flow law,
/// with the EXACT arithmetic the CPU super engine assembles (the single source
/// of truth shared by assembly and rate reporting -- review finding R1).
///
/// Computes, positive FROM the well INTO the reservoir,
///
///     q_total = intercept + productivity * (p_well - p_res - offset)
///
/// converted to `rate[0..ne)` -- component molar rates [kmol/day] and, for a
/// thermal layout, the energy rate [kJ/day] at index `nc` -- through the
/// upstream state (composition, clipped at `layout.eps_z` and renormalized)
/// and the upstream mixture properties read from the interpolated operator
/// table (see the derivation comment above
/// engine_super_cpu::add_perforation_flow_law).
///
/// @param law         the perforation's flow law (must not be DARCY)
/// @param well_block  global block index of the perforated well segment
/// @param res_block   global block index of the perforated reservoir cell
/// @param layout      the assembling engine's layout constants
/// @param X           full state vector (block-major, `layout.n_vars` each)
/// @param op_vals     interpolated operator values (`layout.n_ops` per block)
/// @param op_ders     operator derivatives (`layout.n_vars` per operator), or
///                    nullptr when no derivatives are requested
/// @param cell_spe    per-block specific potential energy (mesh->cell_spe), or
///                    nullptr for an isothermal layout
/// @param rate        [out] `layout.ne` rates, positive well -> reservoir
/// @param drate_w     [out, optional] d(rate)/d(state of the well block),
///                    `layout.ne * layout.n_vars`, entry `c * n_vars + v`
/// @param drate_r     [out, optional] d(rate)/d(state of the reservoir block)
/// @return the upstream block index the conversion used
index_t perforation_law_rates(const perforation_flow_law &law,
                              index_t well_block, index_t res_block,
                              const perforation_law_layout &layout,
                              const value_t *X,
                              const value_t *op_vals,
                              const value_t *op_ders,
                              const value_t *cell_spe,
                              value_t *rate,
                              value_t *drate_w = nullptr,
                              value_t *drate_r = nullptr);

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

    /// @brief Law-aware perforation rate REPORTING shared by calc_rates and
    /// calc_rates_velocity (review finding R1): writes the flow-law component
    /// molar rates of perforation `i_p` (blocks `i_w`/`i_r`, global indices)
    /// into `time_data` through perforation_law_rates() -- the same arithmetic
    /// the engine assembles -- instead of the Darcy `p_diff * wi`, which is
    /// identically zero for a non-Darcy perforation. Throws when the engine
    /// never provided `law_layout`.
    void calc_perforation_law_rates(const perforation_flow_law &law, index_t i_p, index_t i_w, index_t i_r,
        std::vector<value_t>& X, std::vector<value_t>& op_vals_arr,
        std::unordered_map<std::string, std::vector<value_t>>& time_data);

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
    /// Layout snapshot for law-aware rate REPORTING in calc_rates*: only the
    /// assembling engine knows its operator-table layout, so it fills this at
    /// init for every well carrying a non-Darcy perforation
    /// (engine_super_cpu::init). calc_rates* throws on a non-Darcy perforation
    /// when it was never provided -- reporting the Darcy `p_diff * wi == 0`
    /// instead would silently claim no flow (review finding R1).
    perforation_law_layout law_layout;
    bool law_layout_set = false;
    /// The assembling engine's mesh->cell_spe (specific potential energy per
    /// block), needed by the law's energy rate; set together with `law_layout`.
    const std::vector<value_t> *law_cell_spe = nullptr;
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
