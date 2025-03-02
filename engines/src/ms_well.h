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



/// Base class for multi-segmented well
class ms_well
{

public:

  enum WellType
  {
    PRODUCER = -1,
    INJECTOR = 1
  };

  ms_well()
  {
    segment_volume = 0.07; // 1 m high, 0.3 m diameter
    segment_transmissibility = 100000;
    well_head_depth = 0;
    well_body_depth = 0;
    segment_depth_increment = 0;
    segment_diameter = 0;
    segment_roughness = 0;
    well_type = PRODUCER;
  };

  void init_rate_parameters(int n_vars_, int n_ops_, std::vector<std::string> phase_names_, 
                            operator_set_gradient_evaluator_iface* well_controls_etor, operator_set_gradient_evaluator_iface* well_init_etor, int thermal_ = 0)
  {
    n_block_size = n_vars_;
    P_VAR = 0;
    n_vars = n_vars_;
    n_phases = int(phase_names_.size());
    n_ops = n_ops_;
    phase_names = phase_names_;
    thermal = thermal_;

    control = well_control_iface(n_phases, n_vars - thermal, thermal, well_controls_etor, well_init_etor);
    constraint = well_control_iface(n_phases, n_vars - thermal, thermal, well_controls_etor, well_init_etor);

    rate_evaluator = well_controls_etor;
    state.resize(n_vars);
    state_neighbour.resize(n_vars);
    rates.resize(well_control_iface::NUMBER_OF_RATE_TYPES * n_phases + well_control_iface::n_state_ctrls);

	  rate_etor_ad = well_controls_etor;  //adjoint method
  };

  void init_mech_rate_parameters(uint8_t N_VARS_, uint8_t P_VAR_, int n_vars_, int n_ops_, std::vector<std::string> phase_names_, 
                                 operator_set_gradient_evaluator_iface* well_controls_etor, operator_set_gradient_evaluator_iface* well_init_etor, int thermal_ = 0)
  {
    n_block_size = N_VARS_;
    P_VAR = P_VAR_;
    n_vars = n_vars_;
    n_phases = int(phase_names_.size());
    n_ops = n_ops_;
    phase_names = phase_names_;
    thermal = thermal_;

    control = well_control_iface(n_phases, n_vars - thermal, thermal, well_controls_etor, well_init_etor);
    constraint = well_control_iface(n_phases, n_vars - thermal, thermal, well_controls_etor, well_init_etor);

    rate_evaluator = well_controls_etor;
    state.resize(n_vars);
    state_neighbour.resize(n_vars);
    rates.resize(well_control_iface::NUMBER_OF_RATE_TYPES* n_phases + well_control_iface::n_state_ctrls);

	  rate_etor_ad = well_controls_etor;  //adjoint method
  };

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

  int add_to_jacobian(double dt, std::vector<value_t> &X, value_t* jac_well_head, std::vector<value_t> &RHS);

  int check_constraints(double dt, std::vector<value_t> &X);

  int calc_rates(std::vector<value_t> &X, std::vector<value_t>& op_vals_arr, std::unordered_map<std::string, std::vector<value_t>> &time_data);

  int calc_rates_velocity(std::vector<value_t> &X, std::vector<value_t>& op_vals_arr, std::unordered_map<std::string, std::vector<value_t>> &time_data, index_t n_blocks);

  int initialize_control(std::vector<value_t> &X);

  int cross_flow(std::vector<value_t> &X);

  // These properties are only used in discretization, before simulation starts
  std::vector<std::tuple<index_t, index_t, value_t, value_t>> perforations;
  value_t segment_volume;
  value_t segment_transmissibility;
  value_t well_head_depth;
  value_t well_body_depth;
  value_t segment_depth_increment;
  value_t segment_diameter;
  value_t segment_roughness;

  // Properties for simulation

  index_t well_head_idx;        // index of the well head block, where well controls apply
  index_t well_body_idx;        // index of the first well segment block, which connects to ghost well block
  index_t well_head_idx_conn;   // index of the first well segment block connection, which connects to ghost well block

  well_control_iface control;
  well_control_iface constraint;

  operator_set_evaluator_iface* rate_evaluator;
  operator_set_gradient_evaluator_iface *rate_etor_ad;  //adjoint method

  std::string name;
  std::vector<std::string> phase_names;
  std::vector<value_t> state;
  std::vector<value_t> state_neighbour;
  std::vector<value_t> rates;
  int n_vars;
  int n_ops;
  int n_segments = -1;
  int n_phases;
  int thermal;
  // n_block_size -- size of the full block, P_VAR -- index of the start of the state variables within block
  uint8_t n_block_size, P_VAR;

  // segment data type
  std::vector<segment> segments;
  void addSegment()
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



  bool isProducer()                 const {
    // what if the user doesn't choose a name equal to Producer. 
    return (well_type == PRODUCER);
  }

  WellType well_type;          // type to be producer or injector
};

#endif
