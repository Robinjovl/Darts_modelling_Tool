#include "well_controls.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

int well_control_iface::set_bhp_control(bool is_inj, value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_) 
{
	this->well_state_offset = (is_inj) ? 0 : 1; // If injection well, evaluates operators with state of well head; for production, it uses well body
	this->control_type = well_control_iface::WellControlType::BHP;
	
	// Fill well control spec
	this->target = target_;
	this->inj_comp = inj_comp_;
	this->inj_temp = inj_temp_;
	return 0;
}

int well_control_iface::set_rate_control(bool is_inj, well_control_iface::WellControlType control_type_, index_t phase_idx_, value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_) 
{ 
	this->well_state_offset = (is_inj) ? 0 : 1; // If injection well, evaluates operators with state of well head; for production, it uses well body
	this->control_type = control_type_;
	this->phase_idx = phase_idx_;
	
	this->target = target_;
	this->inj_comp = inj_comp_;
	this->inj_temp = inj_temp_;
	return 0;
}

std::string well_control_iface::get_well_control_type_str()
{
	std::string out;

	if (this->control_type == WellControlType::NONE)
	{
		out = "uninitialized/deactivated control";
	}
	else if (this->control_type == WellControlType::BHP)
	{
		out = "BHP-control";
	}
	else if (this->control_type > WellControlType::BHP && this->control_type < WellControlType::NUMBER_OF_RATE_TYPES)
	{
		out = "phase " + std::to_string(phase_idx);
		switch (this->control_type)
		{
			case WellControlType::MOLAR_RATE:
			{
				out += " molar rate-control";
				break;
			}
			case WellControlType::MASS_RATE:
			{
				out += " mass rate-control";
				break;
			}
			case WellControlType::VOLUMETRIC_RATE:
			{
				out += " volumetric rate-control";
				break;
			}
			default:
			{
				out += " advective heat rate-control";
			}
		}
	}
	else
	{
		out = "undefined control";
		exit(1);
	}
	return out;
}

int well_control_iface::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
                                  	    index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
  (void) n_state_size;
  // n_vars is number of flow variables
  // n_block_size is size of block which includes flow and mechanics variables
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];

  // fill the jacobian
  const uint16_t n_block_size_sq = n_block_size * n_block_size;
  memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

  // Set first specification from well controls (defined in operators)
  if (this->control_type == WellControlType::BHP)
  {
    // If BHP controlled - pressure constraint
	state.assign(X.begin() + (well_head_idx + 0) * n_block_size + P_VAR, X.begin() + (well_head_idx + 0) * n_block_size + P_VAR + n_vars);
    well_controls_etor->evaluate_with_derivatives(state, block_idx, well_control_ops, well_control_ops_derivs);
  	
	index_t pres_op_idx = WellControlType::NUMBER_OF_RATE_TYPES * n_phases;
	RHS_well_head[0] = well_control_ops[pres_op_idx] - this->target;

	// BHP operator derivatives
	for (int jj = 0; jj < n_vars; jj++)
	{
	  jacobian_row[n_block_size * P_VAR + P_VAR + jj] = well_control_ops_derivs[pres_op_idx * n_vars + jj];
	}
  }
  else
  {
	// If rate controlled, find the pressure difference and calculate rate
	state.assign(X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR, X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR + n_vars);
    well_controls_etor->evaluate_with_derivatives(state, block_idx, well_control_ops, well_control_ops_derivs);
	value_t p_diff = X_well_head[0] - X_well_body[0];
	index_t rate_op_idx = this->control_type * n_phases + phase_idx;  // find correct index in WellControlOperators

	// RHS
	RHS_well_head[0] = well_control_ops[rate_op_idx] * p_diff * segment_trans - this->target;

	// Rate operator derivatives
	for (int jj = 0; jj < n_vars; jj++)
	{
	  jacobian_row[n_block_size * P_VAR + P_VAR + jj] = well_control_ops_derivs[rate_op_idx * n_vars + jj] * p_diff * segment_trans;
	}
	// Product rule for pressure variable
	jacobian_row[n_block_size * P_VAR + P_VAR] += well_control_ops[rate_op_idx] * segment_trans;
	jacobian_row[n_block_size * P_VAR + P_VAR + n_block_size_sq] = -well_control_ops[rate_op_idx] * segment_trans;

	// if target phase does not exist, set a constant small value to pressure derivative
    // it will let the pressure drop and eventually pressure constraint might work
	if (this->well_state_offset && std::fabs(jacobian_row[n_block_size * P_VAR + P_VAR]) < 1e-3)
	{
	  jacobian_row[n_block_size * P_VAR + P_VAR] = 1.;
	}
  }

  // Loop over rest of vector of well controls (defined in operators)
  if (this->well_state_offset)
  {
	// PRODUCTION WELL: specify equal state to well body
	for (index_t ii = 1; ii < n_vars; ii++)
	{
	  RHS_well_head[ii] = X_well_head[ii] - X_well_body[ii];
	  jacobian_row[n_block_size * (P_VAR + ii) + P_VAR + ii] = 1.;
	  jacobian_row[n_block_size * (P_VAR + ii) + P_VAR + ii + n_block_size_sq] = -1.;
	}
  }
  else
  {
	// INJECTION WELL: specify injection stream
	for (index_t ii = 1; ii < n_comps; ii++)
	{
	  RHS_well_head[ii] = X_well_head[ii] - this->inj_comp[ii-1];
	  jacobian_row[n_block_size * (P_VAR + ii) + P_VAR + ii] = 1.;
	}

	// If thermal, specify
	for (index_t ii = n_comps; ii < n_vars; ii++)
	{
	  index_t temp_op_idx = WellControlType::NUMBER_OF_RATE_TYPES * n_phases + 1;
	  RHS_well_head[ii] = well_control_ops[temp_op_idx] - this->inj_temp;  // well_control_ops[1] contains temperature

	  for (int jj = 0; jj < n_vars; jj++)
	  {
		jacobian_row[n_block_size * (P_VAR + ii) + P_VAR + jj] = well_control_ops_derivs[temp_op_idx * n_vars + jj];
	  }
	}
  }

  return 0;
}

int well_control_iface::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, 
 										     	   index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
  (void) n_state_size;
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t p_diff = X_well_head[0] - X_well_body[0];

  if (this->control_type == WellControlType::BHP)
  {
	// Check if BHP constraint is violated
	state.assign(X.begin() + (well_head_idx + 0) * n_block_size + P_VAR, X.begin() + (well_head_idx + 0) * n_block_size + P_VAR + n_vars);
  	well_controls_etor->evaluate(state, well_control_ops);
	index_t pres_op_idx = WellControlType::NUMBER_OF_RATE_TYPES * n_phases;

	return (p_diff > 0.) ?
			well_control_ops[pres_op_idx] > this->target : // injection well
			well_control_ops[pres_op_idx] < this->target;  // production well
  }
  else
  {
	// Check if rate constraint is violated
  	state.assign(X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR, X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR + n_vars);
  	well_controls_etor->evaluate(state, well_control_ops);
	index_t rate_op_idx = this->control_type * n_phases + phase_idx;  // find correct index in WellControlOperators

  	return (this->target > 0.) ? 
			well_control_ops[rate_op_idx] * p_diff * segment_trans > this->target : // injection well
			well_control_ops[rate_op_idx] * p_diff * segment_trans < this->target;  // production well
  }
}

int well_control_iface::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour, bool is_well_head)
{
  // If block to be initialized is well head, initialize according to well control target
  if (is_well_head)
  {
	// Pressure initialization
	if (this->control_type == WellControlType::BHP)
	{
	  // BHP-controlled: set bhp
	  state_block[0] = this->target;
	}
	else
	{
	  // Rate-controlled: initialize with pressure of neighbouring cell ensuring the correct flow direction
	  state_block[0] = (this->target > 0.) ? state_neighbour[0] + 0.001 : state_neighbour[0] - 0.001;
	}
  
	// Other state specifications
	if (this->well_state_offset == 1)  // if production well
	{
	  // PRODUCTION WELL
	  // Initialize production well with state of neighbouring cell
	  for (size_t i = 1; i < state_block.size() - thermal; i++)
	  {
		state_block[i] = state_neighbour[i];
	  }
  
	  // For temperature/enthalpy, use neighbouring cell
	  if (this->thermal)
	  {
		index_t i = state_block.size()-1;
		state_block[i] = state_neighbour[i];
	  }
	}
	else
	{
	  // INJECTION WELL
	  // Initialize injection well with injection stream
	  for (index_t i = 1; i < state_block.size() - thermal; i++)
	  {
		state_block[i] = this->inj_comp[i-1];
	  }
  
	  // For temperature/enthalpy, use neighbouring cell
	  if (this->thermal)
	  {
		index_t i = state_block.size()-1;
		state_block[i] = state_neighbour[i];
	  }
	}
  }
  // In case block to be initialized well block, initialize with neighbouring state (perforated reservoir cell)
  else
  {
	for (size_t i = 0; i < state_block.size(); i++)
	{
	  state_block[i] = state_neighbour[i];
	}
  }

  return 0;
}

#if 0
int bhp_inj_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
                                       index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];

  const uint16_t n_block_size_sq = n_block_size * n_block_size;
  memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

  // first equation - pressure constraint
  RHS_well_head[0] = X_well_head[0] - target_pressure;
  // all the rest
  int idx = 1;
  for (value_t is : injection_stream)
  {
	RHS_well_head[idx] = X_well_head[idx] - is;
    idx++;
  }

  // fill diagonal H block - it`s always the first
  for (int idx = 0; idx < n_state_size; idx++)
  {
    jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
  }

  return 0;
}

int bhp_inj_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
  return X[well_head_idx * n_block_size + P_VAR] > target_pressure;
}

int bhp_inj_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
  state_block[0] = target_pressure;
  for (int i = 1; i < state_block.size(); i++)
  {
    state_block[i] = injection_stream[i - 1];
  }
  return 0;
}

int bhp_prod_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
	value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];
	const uint16_t n_block_size_sq = n_block_size * n_block_size;

	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

	// first equation - pressure constraint
	RHS_well_head[0] = X_well_head[0] - target_pressure;
	// all the rest
	for (int idx = 1; idx < n_state_size; idx++)
	{
		RHS_well_head[idx] = X_well_head[idx] - X_well_body[idx];
	}

	// fill diagonal H block - it`s always the first
	for (int idx = 0; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
	}

	jacobian_row += n_block_size_sq;

	// fill neighbour H block 
	for (int idx = 1; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = -1;
	}

	return 0;
}

int bhp_prod_well_control::check_constraint_violation (value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
  return X[well_head_idx * n_block_size + P_VAR] < target_pressure;
}

int bhp_prod_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
  state_block[0] = target_pressure;
  for (int i = 1; i < state_block.size(); i++)
  {
    state_block[i] = state_neighbour[i];
  }
  return 0;
}
#endif

#if 0
int volume_rate_well_control::add_to_jacobian(value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body,
  index_t n_state_size, value_t *jacobian_row, value_t *RHS_well_head)
{
  jacobian_row[0] = X_well_head[0];

  return 0;
};

value_t volume_rate_well_control::check_constraint_violation, (value_t dt, std::vector<value_t> &X_well_head, std::vector<value_t> &X_well_body)
{
  return X_well_head[0];

  return 0;
};

#endif

#if 0
int rate_inj_well_control::add_to_jacobian(value_t /*dt*/, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];
  const uint16_t n_block_size_sq = n_block_size * n_block_size;
  value_t p_diff = X_well_head[0] - X_well_body[0];
  value_t current_rate;

  state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
  rate_etor->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);
  current_rate = rates[FLUX_OP] * p_diff * segment_trans;

  // first equation - rate control
  RHS_well_head[0] = current_rate - target_rate;
  
  // all the rest - stream control
  int idx = 1;
  for (value_t is : injection_stream)
  {
    RHS_well_head[idx] = X_well_head[idx] - is;
    idx++;
  }

  // fill diagonal H block - it`s always the first
  memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));
  jacobian_row[n_block_size * P_VAR + P_VAR] = rates_derivs[FLUX_OP * n_state_size] * p_diff * segment_trans + rates[FLUX_OP] * segment_trans;
  jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR] = - rates[FLUX_OP] * segment_trans;
  for (int idx = 1; idx < n_state_size; idx++)
  {
	jacobian_row[n_block_size * P_VAR + P_VAR + idx] = rates_derivs[FLUX_OP * n_state_size + idx] * p_diff * segment_trans;
    jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
  }

  return 0;
}

int rate_inj_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t p_diff = X_well_head[0] - X_well_body[0];

  state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
  rate_etor->evaluate(state, rates);

  return rates[FLUX_OP] * p_diff * segment_trans > target_rate;
}

int rate_inj_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
  // initialize by setting a bit higher pressure to enusure correct flow direction
  state_block[0] = state_neighbour[0] * 1.01;
  // also set initial composition equal to target injection stream
  for (int i = 1; i < state_block.size(); i++)
  {
    state_block[i] = injection_stream[i - 1];
  }
  return 0;
}


int rate_inj_well_control_mass_balance::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans, 
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];

  state[0] = X_well_head[0];
  for (int var = 1; var < n_variables; var++)
    state[var] = injection_stream[var - 1];

  sources_etor->evaluate_with_derivatives(state, block_idx, sources, sources_derivs);

  // add source term to every equation
  for (int eq = 0; eq < n_equations; eq++)
  {
    RHS_well_head[eq] -= sources[eq] * dt * target_rate;
    for (int var = 0; var < n_variables; var++)
	  jacobian_row[n_block_size * (P_VAR + eq) + P_VAR + var] -= sources_derivs[var + eq * n_equations] * dt * target_rate;
  }

  return 0;
}

int rate_inj_well_control_mass_balance::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t p_diff = X_well_head[0] - X_well_body[0];

  state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
  rate_etor->evaluate(state, rates);

  return rates[target_phase_idx] * p_diff * segment_trans > target_rate;
}

int rate_inj_well_control_mass_balance::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
  // initialize by setting a bit higher pressure to enusure correct flow direction
  state_block[0] = state_neighbour[0] * 1.01;
  // also set initial composition equal to target injection stream
  for (int i = 1; i < state_block.size(); i++)
  {
    state_block[i] = injection_stream[i - 1];
  }
  return 0;
}

int rate_prod_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans, 
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];
  const uint16_t n_block_size_sq = n_block_size * n_block_size;
  value_t p_diff = X_well_body[0] - X_well_head[0];
  value_t current_rate;

  // take state from well body
  state.assign(X.begin() + (well_head_idx + 1) * n_block_size + P_VAR, X.begin() + (well_head_idx + 1) * n_block_size + P_VAR + n_state_size);
  rate_etor->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);
  current_rate = rates[FLUX_OP] * p_diff * segment_trans;

  // first equation - rate control
  RHS_well_head[0] = current_rate - target_rate;

  // all the rest - upstream control
  for (int i = 1; i < n_variables; i++)
  {
    RHS_well_head[i] = X_well_head[i] - X_well_body[i];
  }

  // fill jacobian
  memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));
  jacobian_row[n_block_size * P_VAR + P_VAR] = -rates[FLUX_OP] * segment_trans;
  
  // if target phase does not exist, set a constant small value to pressure derivative
  // it will let the pressure drop and eventually pressure constraint might work
  if (fabs(jacobian_row[n_block_size * P_VAR + P_VAR]) < 1e-3)
	jacobian_row[n_block_size * P_VAR + P_VAR] = -1;

  jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR] = rates_derivs[FLUX_OP * n_state_size] * p_diff * segment_trans + rates[FLUX_OP] * segment_trans;
  
  for (int idx = 1; idx < n_state_size; idx++)
  {
	jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR + idx] = rates_derivs[FLUX_OP * n_state_size + idx] * p_diff * segment_trans;
	jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
	jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx + n_block_size_sq] = -1;
  }

  return 0;
}

int rate_prod_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t p_diff = X_well_head[0] - X_well_body[0];

  state.assign(X.begin() + (well_head_idx + 1) * n_block_size + P_VAR, X.begin() + (well_head_idx + 1) * n_block_size + P_VAR + n_state_size);
  rate_etor->evaluate(state, rates);
  /*
  for (int i = 0; i < state.size(); i++)
    std::cout << state[i] << " ";
  
  std::cout << std::endl;

  for (int i = 0; i < phase_names.size(); i++)
    std::cout << phase_names[i] << " rate is " << fabs(rates[i] * p_diff * segment_trans) << "  ";
  std::cout << std::endl;
  */
  return fabs(rates[FLUX_OP] * p_diff * segment_trans) > target_rate;
}

int rate_prod_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
  // initialize by setting a bit higher pressure to enusure correct flow direction
  state_block[0] = state_neighbour[0] * 0.99;
  // also set initial composition equal to the neighbour
  for (int i = 1; i < state_block.size(); i++)
  {
    state_block[i] = state_neighbour[i];
  }
  return 0;
}

int rate_prod_well_control_mass_balance::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans, 
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];
  const uint16_t n_block_size_sq = n_block_size * n_block_size;

  for (int var = 0; var < n_variables; var++)
    state[var] = X_well_body[var];

  sources_etor->evaluate_with_derivatives(state, block_idx, sources, sources_derivs);

  // add sink term to every equation's well_body block
  jacobian_row += n_block_size_sq;
  for (int eq = 0; eq < n_equations; eq++)
  {
    RHS_well_head[eq] += sources[eq] * dt * target_rate;
    for (int var = 0; var < n_variables; var++)
	  jacobian_row[n_block_size * (P_VAR + eq) + P_VAR + var] += sources_derivs[var + eq * n_equations] * dt * target_rate;
  }

  return 0;
}



int rate_prod_well_control_mass_balance::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t p_diff = X_well_head[0] - X_well_body[0];

  for (int var = 0; var < n_variables; var++)
    state[var] = X_well_body[var];
  rate_etor->evaluate(state, rates);

  return rates[target_phase_idx] * p_diff * segment_trans > target_rate;
}


int rate_prod_well_control_mass_balance::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
  state_block[0] = state_neighbour[0] * 0.99;
  for (int i = 1; i < state_block.size(); i++)
  {
    state_block[i] = state_neighbour[i];
  }
  return 0;
}


int gt_bhp_temp_inj_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
    int temp_idx = 0;
	value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
	//value_t *X_body_head = X_well_head + n_state_size;
	value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];

	const uint16_t n_block_size_sq = n_block_size * n_block_size;

	state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
	// first equation - pressure constraint
	RHS_well_head[0] = X_well_head[0] - target_pressure;
	// second equation - temperature constraint
	rate_etor->evaluate_with_derivatives(state, block_idx, rate_temp_ops, rate_temp_ops_derivs);

    for (int i = 0; i < n_phases; i++)
    {
        if (phase_names[i] == "temperature")
        {
            temp_idx = i;
        }
    }
	RHS_well_head[1] = rate_temp_ops[temp_idx] - target_temperature;

	// fill diagonal H block - it`s always the first
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

	jacobian_row[n_block_size * P_VAR + P_VAR] = 1;
	for (int idx = 0; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * (P_VAR + 1) + P_VAR + idx] = rate_temp_ops_derivs[(temp_idx)* n_block_size + idx];
	}
	return 0;
}


int gt_bhp_temp_inj_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X)
{
	return X[well_head_idx * n_state_size] > target_pressure;
}

int gt_bhp_temp_inj_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
	state_block[0] = target_pressure;
	return 0;						 
}

int gt_bhp_prod_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
	value_t* X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t* X_body_head = X_well_head + n_block_size;
	value_t* RHS_well_head = &RHS[well_head_idx * n_block_size + P_VAR];
	const uint16_t n_block_size_sq = n_block_size * n_block_size;
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

	// RHS
	RHS_well_head[0] = X_well_head[0] - target_pressure;
	for (int idx = 1; idx < n_state_size; idx++)
	{
		RHS_well_head[idx] = X_well_head[idx] - X_body_head[idx];
	}
	// Jacobian
	for (int idx = 0; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
	}
	jacobian_row += n_block_size_sq;
	for (int idx = 1; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = -1;
	}
	return 0;
}

int gt_bhp_prod_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X)
{
	return X[well_head_idx * n_block_size + P_VAR] < target_pressure;
}

int gt_bhp_prod_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
	state_block[0] = target_pressure;
	for (int i = 1; i < state_block.size(); i++)
	{
		state_block[i] = state_neighbour[i];
	}
	return 0;
}

int gt_rate_temp_inj_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
    int temp_idx = 0;
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t *RHS_well_head = &RHS[well_head_idx * n_block_size + P_VAR];
	const uint16_t n_block_size_sq = n_block_size * n_block_size;

	state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
	value_t p_diff = X_well_head[0] - X_well_body[0];

    for (int i = 0; i < n_phases; i++)
    {
        if (phase_names[i] == "temperature")
        {
            temp_idx = i;
        }
    }
	// RHS
	rate_etor->evaluate_with_derivatives(state, block_idx, rate_temp_ops, rate_temp_ops_derivs);
	RHS_well_head[0] = rate_temp_ops[target_phase_idx] * p_diff * segment_trans - target_rate;
	RHS_well_head[1] = rate_temp_ops[temp_idx] - target_temperature;

	// fill the jacobian
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));
	
	//jacobian_row[0] = rate_temp_ops_derivs[target_phase_idx * n_state_size] * p_diff * segment_trans + rate_temp_ops[target_phase_idx] * segment_trans;
	//jacobian_row[n_block_size_sq] = -rate_temp_ops[target_phase_idx] * segment_trans;

	for (int idx = 0; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * P_VAR + P_VAR + idx] = rate_temp_ops_derivs[target_phase_idx * n_state_size + idx] * p_diff * segment_trans;
	}
	jacobian_row[n_block_size * P_VAR + P_VAR] += rate_temp_ops[target_phase_idx] * segment_trans;
	jacobian_row[n_block_size * P_VAR + P_VAR + n_block_size_sq] = -rate_temp_ops[target_phase_idx] * segment_trans;

	for (int idx = 0; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * (P_VAR + 1) + P_VAR + idx] = rate_temp_ops_derivs[(temp_idx)* n_variables + idx];
	}
	return 0;
};

int gt_rate_temp_inj_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X)
{
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;

	value_t p_diff = X_well_head[0] - X_well_body[0];
	state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
	rate_etor->evaluate(state, rate_temp_ops);

	return (rate_temp_ops[target_phase_idx] * p_diff * segment_trans) > target_rate;
};

int gt_rate_temp_inj_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
	// initialize by setting a bit higher pressure to enusure correct flow direction
	state_block[0] = state_neighbour[0] + 0.001;
	return 0;
};

int gt_rate_prod_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t *RHS_well_head = &RHS[well_head_idx * n_block_size + P_VAR];
	const uint16_t n_block_size_sq = n_block_size * n_block_size;
	value_t p_diff = X_well_body[0] - X_well_head[0];

	state.assign(X.begin() + (well_head_idx + 1) * n_block_size + P_VAR, X.begin() + (well_head_idx + 1) * n_block_size + P_VAR + n_state_size);
	rate_etor->evaluate_with_derivatives(state, block_idx, rate_temp_ops, rate_temp_ops_derivs);

	// RHS
	RHS_well_head[0] = rate_temp_ops[target_phase_idx] * p_diff * segment_trans - target_rate;
	for (int idx = 1; idx < n_state_size; idx++)
	{
		RHS_well_head[idx] = X_well_head[idx] - X_well_body[idx];
	}						 

	// fill the jacobian
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));
	jacobian_row[n_block_size * P_VAR + P_VAR] = - rate_temp_ops[target_phase_idx] * segment_trans ;
	
	if (fabs(jacobian_row[n_block_size * P_VAR + P_VAR]) < 1e-3)
		jacobian_row[n_block_size * P_VAR + P_VAR] = -1;

	jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR] = rate_temp_ops_derivs[target_phase_idx * n_state_size] * p_diff * segment_trans + rate_temp_ops[target_phase_idx] * segment_trans;

	for (int idx = 1; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR + idx] = rate_temp_ops_derivs[target_phase_idx * n_state_size + idx] * p_diff * segment_trans;
		jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
		jacobian_row[n_block_size_sq + n_block_size * (P_VAR + idx) + P_VAR + idx] = -1;
	}
		
	return 0;
};

int gt_rate_prod_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X)
{
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t p_diff = X_well_body[0] - X_well_head[0];
	state.assign(X.begin() + (well_head_idx + 1) * n_block_size + P_VAR, X.begin() + (well_head_idx + 1) * n_block_size + P_VAR + n_state_size);
	rate_etor->evaluate(state, rate_temp_ops);

	return (fabs(rate_temp_ops[target_phase_idx] * p_diff * segment_trans) > target_rate);
};

int gt_rate_prod_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
	// initialize by setting a bit higher pressure to enusure correct flow direction
	state_block[0] = state_neighbour[0] * 0.99;

	// set other initial conditions equal to the neighbour
	for (int i = 1; i < state_block.size(); i++)
	{
		state_block[i] = state_neighbour[i];
	}
	return 0;
};

int gt_mass_rate_enthalpy_inj_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t *RHS_well_head = &RHS[well_head_idx * n_block_size + P_VAR];
	const uint16_t n_block_size_sq = n_block_size * n_block_size;
 	value_t p_diff = X_well_head[0] - X_well_body[0];

	state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
	rate_etor->evaluate_with_derivatives(state, block_idx, rate_temp_ops, rate_temp_ops_derivs);

	//RHS
	RHS_well_head[0] = rate_temp_ops[target_phase_idx] * p_diff * segment_trans -  target_rate;
	RHS_well_head[1] = X_well_head[1] - target_enthalpy;

	//fill the jacobian
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

	jacobian_row[n_block_size * P_VAR + P_VAR] = rate_temp_ops_derivs[(target_phase_idx * n_state_size)] * p_diff * segment_trans + rate_temp_ops[target_phase_idx] * segment_trans;
	jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR] = -rate_temp_ops[target_phase_idx] * segment_trans;
	for (int idx = 1; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size * P_VAR + P_VAR + idx] = rate_temp_ops_derivs[(target_phase_idx * n_state_size + idx)] * p_diff * segment_trans;
		jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
	}
	return 0;
}

int gt_mass_rate_enthalpy_inj_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X)
{
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t p_diff = X_well_head[0] - X_well_body[0];
	state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
	rate_etor->evaluate(state, rate_temp_ops);

	return (rate_temp_ops[target_phase_idx] * p_diff * segment_trans) > target_rate;
}

int gt_mass_rate_enthalpy_inj_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
	state_block[0] = state_neighbour[0] + 0.001;
	return 0;
}

int gt_mass_rate_prod_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t *RHS_well_head = &RHS[well_head_idx * n_block_size + P_VAR];
 	value_t p_diff = X_well_body[0] - X_well_head[0];
	const uint16_t n_block_size_sq = n_block_size * n_block_size;

	state.assign(X.begin() + (well_head_idx + 1) * n_block_size + P_VAR, X.begin() + (well_head_idx + 1) * n_block_size + P_VAR + n_state_size);
	rate_etor->evaluate_with_derivatives(state, block_idx, rate_temp_ops, rate_temp_ops_derivs);

	//RHS
	RHS_well_head[0] = rate_temp_ops[target_phase_idx] * p_diff * segment_trans - target_rate;
	for (int idx = 1; idx < n_state_size; idx++)
	{
		RHS_well_head[idx] = X_well_head[idx] -  X_well_body[idx];
	}
	
	//fill the jacobian
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

	jacobian_row[n_block_size * P_VAR + P_VAR] = -rate_temp_ops[target_phase_idx] * segment_trans;

	if (fabs(jacobian_row[n_block_size * P_VAR + P_VAR]) < 1e-3)
		jacobian_row[n_block_size * P_VAR + P_VAR] = -1;

	jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR] = rate_temp_ops_derivs[target_phase_idx * n_state_size] * p_diff * segment_trans + rate_temp_ops[target_phase_idx] * segment_trans;

	for (int idx = 1; idx < n_state_size; idx++)
	{
		jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR + idx] = rate_temp_ops_derivs[target_phase_idx * n_state_size + idx] * p_diff * segment_trans;
		jacobian_row[n_block_size * (P_VAR + idx) + P_VAR + idx] = 1;
		jacobian_row[n_block_size_sq + n_block_size * (P_VAR + idx) + P_VAR + idx] = -1;
	}
	return 0;
};

int gt_mass_rate_prod_well_control::check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X)
{
	value_t *X_well_head = &X[well_head_idx * n_block_size + P_VAR];
	value_t *X_well_body = X_well_head + n_block_size;
	value_t p_diff = X_well_body[0] - X_well_head[0];
	state.assign(X.begin() + (well_head_idx + 1) * n_block_size + P_VAR, X.begin() + (well_head_idx + 1) * n_block_size + P_VAR + n_state_size);
	rate_etor->evaluate(state, rate_temp_ops);

	return rate_temp_ops[target_phase_idx] * p_diff * segment_trans > target_rate;
}

int gt_mass_rate_prod_well_control::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
	// initialize by setting a bit higher pressure to enusure correct flow direction
	state_block[0] = state_neighbour[0] * 0.99;

	// set other initial conditions equal to the neighbour
	for (int i = 1; i < state_block.size(); i++)
	{
		state_block[i] = state_neighbour[i];
	}
	return 0;

}
#endif
