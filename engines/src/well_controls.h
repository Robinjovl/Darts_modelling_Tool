#ifndef WELL_CONTROLS_H
#define WELL_CONTROLS_H

#include <vector>
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
  // MOLAR_RATE is 0 because it is the first rate operator type in the WellControlOperators
  enum WellControlType : int { NONE = -2, BHP, MOLAR_RATE, MASS_RATE, VOLUMETRIC_RATE, ADVECTIVE_HEAT_RATE, NUMBER_OF_RATE_TYPES };
  static const int n_state_ctrls = 2;  // pressure (BHP) and temperature (BHT) operators

protected:
  WellControlType control_type = NONE;
  index_t phase_idx{ 0 }, n_phases, n_comps, thermal, n_vars, n_ops, well_state_offset;
  value_t target, inj_temp;
  std::vector<value_t> inj_comp;
  std::vector<index_t> block_idx {0};
  std::vector<value_t> state;
  std::vector<value_t> well_control_ops;
  std::vector<value_t> well_control_ops_derivs;
  operator_set_gradient_evaluator_iface *well_controls_etor, *well_init_etor;
  
public:
  well_control_iface() {}
  well_control_iface(index_t n_phases_, index_t n_comps_, bool thermal_, operator_set_gradient_evaluator_iface* well_controls_etor_, operator_set_gradient_evaluator_iface* well_init_etor_) 
  : n_phases(n_phases_), n_comps(n_comps_), thermal(thermal_), well_controls_etor(well_controls_etor_), well_init_etor(well_init_etor_)
  {
	  // Evaluate well control operators
    // WellControlOperators are defined as follows: P, composition, T, NP MOLAR_RATE, NP MASS_RATE, NP VOLUMETRIC_RATE, and NP ADVECTIVE_HEAT_RATE operators
	  n_vars = n_comps + thermal;
    n_ops = WellControlType::NUMBER_OF_RATE_TYPES * n_phases + well_control_iface::n_state_ctrls;
	  well_control_ops.resize(n_ops);
	  well_control_ops_derivs.resize(n_ops * n_vars);
  }

  virtual int set_bhp_control(bool is_inj, value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_);
  virtual int set_rate_control(bool is_inj, well_control_iface::WellControlType control_type_, index_t phase_idx_, 
                               value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_);

  WellControlType get_well_control_type() { return this->control_type; }
  index_t get_well_n_ops() { return this->n_ops; }
  index_t get_well_n_vars() { return this->n_vars; }
  std::string get_well_control_type_str();

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);
  
  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, 
    uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);
};

#endif
