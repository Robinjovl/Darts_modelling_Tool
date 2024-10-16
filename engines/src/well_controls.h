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

#define WELL_CONTROL_FILL

/// Base class work well control/constraint
class well_control_iface
{
public:
  well_control_iface() { 
    block_idx.resize(1);
    block_idx[0] = 0;
  };

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS) = 0;

  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X) = 0;

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour) = 0;

  virtual int add_to_csr_jacobian(value_t /*dt*/, index_t /*well_head_idx*/, value_t /*segment_trans*/,
	  index_t /*n_state_size*/, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS) {
	  return 0;
  };

  std::string name;
  std::vector<index_t> block_idx;
};

/** @defgroup Well_controls
 *  Methods for well control/constraint exposed to Python
 *  @{
 */

/// BHP control for injection compositional well
class bhp_inj_well_control : public well_control_iface
{
public:
  bhp_inj_well_control(value_t target_pressure_, std::vector <value_t> &injection_stream_) : target_pressure(target_pressure_),
                                                                                             injection_stream(injection_stream_)
  {
    name = "BHP injector";
  };

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

  value_t target_pressure;
  std::vector <value_t> injection_stream;
};

/// BHP control for production compositional well
class bhp_prod_well_control : public well_control_iface
{
public:
  bhp_prod_well_control(value_t target_pressure_) : target_pressure(target_pressure_)
  {
    name = "BHP producer";
  };

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);
  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

  value_t target_pressure;
};

/// Volumetric rate control for injection compositional well
class rate_inj_well_control : public well_control_iface
{
public:
  rate_inj_well_control(std::vector <std::string> phase_names_, index_t target_phase_idx_, index_t n_equations_, index_t n_variables_,
                        value_t target_rate_, std::vector <value_t> &injection_stream_, 
                        operator_set_gradient_evaluator_iface* rate_etor_) :
    phase_names(phase_names_), target_phase_idx(target_phase_idx_), n_equations(n_equations_), n_variables(n_variables_),
    target_rate(target_rate_), injection_stream(injection_stream_), rate_etor(rate_etor_)
  {
    name = phase_names[target_phase_idx] + " rate injector";
    state.resize(n_variables);
    rates.resize(phase_names_.size());
    rates_derivs.resize(phase_names_.size() * n_variables);
  };

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);


  index_t target_phase_idx, n_equations, n_variables;
  std::vector <std::string> phase_names;
  value_t target_rate;
  std::vector <value_t> injection_stream;
  operator_set_gradient_evaluator_iface *rate_etor;

  std::vector<value_t> state;
  std::vector<value_t> rates;
  std::vector<value_t> rates_derivs;
};

/// Volumetric rate control for production compositional well 
class rate_prod_well_control : public well_control_iface
{
public:
  rate_prod_well_control(std::vector <std::string> phase_names_, index_t target_phase_idx_, index_t n_equations_, index_t n_variables_,
                        value_t target_rate_, operator_set_gradient_evaluator_iface* rate_etor_) :
    phase_names(phase_names_), target_phase_idx(target_phase_idx_), n_equations(n_equations_), n_variables(n_variables_),
    target_rate(target_rate_), rate_etor(rate_etor_)
  {
    name = phase_names[target_phase_idx] + " rate producer";
    state.resize(n_variables);
    rates.resize(phase_names_.size());
    rates_derivs.resize(phase_names_.size() * n_variables);
  };

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);


  index_t target_phase_idx, n_equations, n_variables;
  std::vector <std::string> phase_names;
  value_t target_rate;
  operator_set_gradient_evaluator_iface *rate_etor;

  std::vector<value_t> state;
  std::vector<value_t> rates;
  std::vector<value_t> rates_derivs;
};



/// BHP and temperature control for injection geothermal well 
class gt_bhp_temp_inj_well_control : public well_control_iface
{
public:
	gt_bhp_temp_inj_well_control(std::vector<std::string> phase_names_, index_t n_block_size_,
		value_t target_pressure_, value_t target_temperature_, std::vector<value_t> injection_stream_,
		operator_set_gradient_evaluator_iface *rate_etor_) :
		phase_names(phase_names_), n_block_size(n_block_size_),
		target_pressure(target_pressure_), target_temperature(target_temperature_), injection_stream(injection_stream_),
		rate_etor(rate_etor_)
	{
		name = "gt BHP TEMP injector";
		n_phases = phase_names.size();
		rate_temp_ops.resize(n_phases);
		rate_temp_ops_derivs.resize((n_phases) * n_block_size);
	};
	virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
		index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
    index_t n_block_size, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

	virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

	virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

	std::vector<std::string> phase_names;
	index_t n_block_size, n_phases;
	value_t target_pressure;
	value_t target_temperature;
	std::vector<value_t> injection_stream;
	std::vector<value_t> state;
	std::vector<value_t> rate_temp_ops;
	std::vector<value_t> rate_temp_ops_derivs;
	operator_set_gradient_evaluator_iface *rate_etor;
};

/// BHP control for production geothermal well 
class gt_bhp_prod_well_control : public well_control_iface
{
public:
	gt_bhp_prod_well_control(value_t target_pressure_) : target_pressure(target_pressure_)
	{
		name = "gt BHP producer";
	};
	virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
		index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
    index_t n_block_size, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

	virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

	virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

	value_t target_pressure;
};

/// Volumetric rate and temperature control for injection geothermal well 
class gt_rate_temp_inj_well_control : public well_control_iface
{
public:
	 gt_rate_temp_inj_well_control(
		 std::vector <std::string> phase_names_, index_t target_phase_idx_, index_t n_variables_,
		 value_t target_rate_, value_t target_temp_, std::vector <value_t> &injection_stream_,
		 operator_set_gradient_evaluator_iface* rate_etor_):
		 phase_names(phase_names_), target_phase_idx(target_phase_idx_), n_variables(n_variables_),
		 target_rate(target_rate_), target_temperature(target_temp_), injection_stream(injection_stream_),
		 rate_etor(rate_etor_)
	 {
		 name = phase_names[target_phase_idx] + " rate injector";
		 state.resize(n_variables);
		 n_phases = phase_names.size();
		 rate_temp_ops.resize(n_phases);
		 rate_temp_ops_derivs.resize(n_phases * n_variables);
	 };

	 virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
		 index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

	 virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

	 virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

	 std::vector<std::string> phase_names;
   index_t target_phase_idx, n_equations, n_variables;
   size_t n_phases;
	 value_t target_rate;
	 value_t target_temperature;
	 std::vector<value_t> injection_stream;
	 std::vector<value_t> state, rate_temp_ops, rate_temp_ops_derivs;
	 operator_set_gradient_evaluator_iface* rate_etor;
};

/// Volumetric rate control for production geothermal well 
class gt_rate_prod_well_control : public well_control_iface
{
public:
	gt_rate_prod_well_control(std::vector<std::string> phase_names_, index_t target_phase_idx_, index_t n_block_size,
		value_t target_rate_,
		operator_set_gradient_evaluator_iface* rate_etor_) :
		phase_names(phase_names_), 	target_phase_idx(target_phase_idx_), n_variables(n_block_size),
		target_rate(target_rate_),
		rate_etor(rate_etor_)
	{
		name = phase_names[target_phase_idx] + " rate producer";
		state.resize(n_variables);
		n_phases = index_t(phase_names.size());
		rate_temp_ops.resize(n_phases);
		rate_temp_ops_derivs.resize(n_phases * n_block_size);
	};

	virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
		index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

	virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

	virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

	std::vector<std::string> phase_names;
	std::vector<value_t> state, rate_temp_ops, rate_temp_ops_derivs;
	index_t target_phase_idx, n_variables, n_phases;
	value_t target_rate;
	operator_set_gradient_evaluator_iface *rate_etor;

};

/// Mass rate and enthalpy control for injection geothermal well 
class gt_mass_rate_enthalpy_inj_well_control : public well_control_iface
{
public:
	gt_mass_rate_enthalpy_inj_well_control(
		std::vector<std::string> phase_names_, index_t target_phase_idx_, index_t n_variables_, std::vector<value_t>injection_stream_,
		value_t target_rate_, value_t target_enthalpy_,
		operator_set_gradient_evaluator_iface *rate_etor_) :
		phase_names(phase_names_), target_phase_idx(target_phase_idx_), n_variables(n_variables_), injection_stream(injection_stream_),
		target_rate(target_rate_), target_enthalpy(target_enthalpy_),
		rate_etor(rate_etor_)
	{
		name = phase_names[target_phase_idx] + " mass rate enthalpy injector";
		state.resize(n_variables);
		n_phases = phase_names.size();
		rate_temp_ops.resize(n_phases);
		rate_temp_ops_derivs.resize(n_phases * n_variables);		
	}

	virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
		index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

	virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

	virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

	std::vector<std::string> phase_names;
	std::vector<value_t> state, injection_stream, rate_temp_ops, rate_temp_ops_derivs;
	index_t target_phase_idx, n_variables, n_phases;
	value_t target_rate, target_enthalpy;
	operator_set_gradient_evaluator_iface *rate_etor;
};

/// Mass rate control for production geothermal well 
class gt_mass_rate_prod_well_control : public well_control_iface
{
public:
	gt_mass_rate_prod_well_control(
		std::vector<std::string> phase_names_, index_t target_phase_idx_, index_t n_variables_,
		value_t target_rate_,
		operator_set_gradient_evaluator_iface *rate_etor_) :
		phase_names(phase_names_), target_phase_idx(target_phase_idx_), n_variables(n_variables_),
		target_rate(target_rate_),
		rate_etor(rate_etor_)
	{
		name = phase_names[target_phase_idx] + " mass rate producer";
		n_phases = phase_names.size();
		state.resize(n_variables);
		rate_temp_ops.resize(n_phases);
		rate_temp_ops_derivs.resize(n_phases * n_variables);
	}

	virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
		index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

	virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

	virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);

	std::vector<std::string> phase_names;
	std::vector<value_t> state, rate_temp_ops, rate_temp_ops_derivs;
	index_t target_phase_idx, n_variables, n_phases;
	value_t target_rate;
	operator_set_gradient_evaluator_iface *rate_etor;
};

/// @} // end of well_controls

/// Rate control based on mass ballance equation for injection compositional well 
class rate_inj_well_control_mass_balance : public well_control_iface
{
public:
  rate_inj_well_control_mass_balance(std::vector <std::string> phase_names_, index_t target_phase_idx_, index_t n_equations_, index_t n_variables_,
    value_t target_rate_, std::vector <value_t> &injection_stream_,
    operator_set_evaluator_iface* rate_etor_, operator_set_gradient_evaluator_iface* sources_etor_) :
    phase_names(phase_names_), target_phase_idx(target_phase_idx_), n_equations(n_equations_), n_variables(n_variables_),
    target_rate(target_rate_), injection_stream(injection_stream_), rate_etor(rate_etor_), sources_etor(sources_etor_)
  {
    name = phase_names[target_phase_idx] + " rate injector (mass balance)";
    state.resize(n_variables);
    rates.resize(phase_names_.size());
    sources.resize(n_equations);
    sources_derivs.resize(n_equations * n_variables);
  };

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);


  index_t target_phase_idx, n_equations, n_variables;
  std::vector <std::string> phase_names;
  value_t target_rate;
  std::vector <value_t> injection_stream;
  operator_set_gradient_evaluator_iface *sources_etor;
  operator_set_evaluator_iface *rate_etor;


  std::vector<value_t> state;
  std::vector<value_t> sources, rates;
  std::vector<value_t> sources_derivs;
};

/// Rate control based on mass ballance equation for production compositional well
class rate_prod_well_control_mass_balance : public well_control_iface
{
public:
  rate_prod_well_control_mass_balance(std::vector <std::string> phase_names_, index_t target_phase_idx_, index_t n_equations_, index_t n_variables_,
    value_t target_rate_,
    operator_set_evaluator_iface* rate_etor_, operator_set_gradient_evaluator_iface* sources_etor_) :
    phase_names(phase_names_), target_phase_idx(target_phase_idx_), n_equations(n_equations_), n_variables(n_variables_),
    target_rate(target_rate_), rate_etor(rate_etor_), sources_etor(sources_etor_)
  {
    name = phase_names[target_phase_idx] + " rate producer (mass balance)";
    state.resize(n_variables);
    rates.resize(phase_names_.size());
    sources.resize(n_equations);
    sources_derivs.resize(n_equations * n_variables);
  };

  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	  index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X);

  virtual int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour);


  std::vector <std::string> phase_names;
  index_t target_phase_idx, n_equations, n_variables;
  value_t target_rate;
  operator_set_evaluator_iface *rate_etor;
  operator_set_gradient_evaluator_iface *sources_etor;


  std::vector<value_t> state;
  std::vector<value_t> sources, rates;
  std::vector<value_t> sources_derivs;
};



/*
class bhp_prod_well_control : public well_control_iface
{
public:
  bhp_prod_well_control(value_t target_pressure_) : target_pressure(target_pressure_)
  {
    name = "BHP PRODUCER";
  };
  virtual int add_to_jacobian(value_t dt, index_t well_head_idx, index_t n_state_size,
                              std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS);

  virtual int check_constraint_violation(value_t dt, index_t well_head_idx, index_t n_state_size, std::vector<value_t> &X);

  value_t target_pressure;
};
*/
#ifdef WELL_CONTROL_FILL

#else

class well_control_iface
{
public:
  well_control_iface() {};

  virtual int add_to_jacobian(value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body,
                                        index_t n_state_size, value_t *jacobian_row, value_t *RHS_well_head) = 0;

  virtual value_t check_constraint_violation, (value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body) = 0;

  std::string name;
};

class bhp_well_control : public well_control_iface
{
public:
  bhp_well_control(value_t target_pressure_)
  {
    target_pressure = target_pressure_;
    name = "BHP control";
  };
  int add_to_jacobian(value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body,
                                index_t n_state_size, value_t *jacobian_row, value_t *RHS_well_head);

  value_t check_constraint_violation, (value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body)
  {
    return X_well_head[0];
  };

  value_t target_pressure;
};

class volume_rate_well_control : public well_control_iface
{
public:
  volume_rate_well_control(index_t target_phase_, value_t target_volume_rate_)
  {
    target_phase = target_phase_;
    target_volume_rate = target_volume_rate_;
  };

  int add_to_jacobian(value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body,
                                index_t n_state_size, value_t *jacobian_row, value_t *RHS_well_head);

  value_t check_constraint_violation, (value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body);

  index_t target_phase;
  value_t target_volume_rate;
};

#endif

#endif
