//*************************************************************************
//    Copyright (c) 2018
//            Mark Khait         M.Khait@tudelft.nl
//            Denis Voskov    D.V.Voskov@tudelft.nl
//    Delft University of Technology, the Netherlands
//
//    This file is part of the Delft Advanced Research Terra Simulator (DARTS)
//
//    DARTS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as 
//    published by the Free Software Foundation, either version 3 of the 
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public 
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************


#include "well_controls.h"
#include <iostream>
#include <cstring>
#include <cmath>


int bhp_inj_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
                                       index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];

  const uint8_t n_block_size_sq = n_block_size * n_block_size;
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
	const uint8_t n_block_size_sq = n_block_size * n_block_size;

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

int rate_inj_well_control::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans, 
	index_t n_state_size, uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{
  value_t *X_well_head = &X[n_block_size * well_head_idx + P_VAR];
  value_t *X_well_body = X_well_head + n_block_size;
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];
  const uint8_t n_block_size_sq = n_block_size * n_block_size;
  value_t p_diff = X_well_head[0] - X_well_body[0];
  value_t current_rate;

  state.assign(X.begin() + well_head_idx * n_block_size + P_VAR, X.begin() + well_head_idx * n_block_size + P_VAR + n_state_size);
  rate_etor->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);
  current_rate = rates[target_phase_idx] * p_diff * segment_trans;

  // first equation - rate constraint
  RHS_well_head[0] = current_rate - target_rate;
  
  // all the rest - injection stream constraint
  int idx = 1;
  for (value_t is : injection_stream)
  {
    RHS_well_head[idx] = X_well_head[idx] - is;
    idx++;
  }

  // fill diagonal H block - it`s always the first
  memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));
  jacobian_row[n_block_size * P_VAR + P_VAR] = rates_derivs[target_phase_idx * n_state_size] * p_diff * segment_trans + rates[target_phase_idx] * segment_trans;
  jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR] = - rates[target_phase_idx] * segment_trans;
  for (int idx = 1; idx < n_state_size; idx++)
  {
	jacobian_row[n_block_size * P_VAR + P_VAR + idx] = rates_derivs[target_phase_idx * n_state_size + idx] * p_diff * segment_trans;
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

  return rates[target_phase_idx] * p_diff * segment_trans > target_rate;
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
  const uint8_t n_block_size_sq = n_block_size * n_block_size;
  value_t p_diff = X_well_body[0] - X_well_head[0];
  value_t current_rate;

  // take state from well body
  state.assign(X.begin() + (well_head_idx + 1) * n_block_size + P_VAR, X.begin() + (well_head_idx + 1) * n_block_size + P_VAR + n_state_size);
  rate_etor->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);
  current_rate = rates[target_phase_idx] * p_diff * segment_trans;

  // first equation - rate constraint
  RHS_well_head[0] = current_rate - target_rate;

  // all the rest - upstream constraint
  for (int i = 1; i < n_variables; i++)
  {
    RHS_well_head[i] = X_well_head[i] - X_well_body[i];
  }

  // fill jacobian
  memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));
  jacobian_row[n_block_size * P_VAR + P_VAR] = -rates[target_phase_idx] * segment_trans;
  
  // if target phase does not exist, set a constant small value to pressure derivative
  // it will let the pressure drop and eventually pressure constraint might work
  if (fabs(jacobian_row[n_block_size * P_VAR + P_VAR]) < 1e-3)
	jacobian_row[n_block_size * P_VAR + P_VAR] = -1;

  jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR] = rates_derivs[target_phase_idx * n_state_size] * p_diff * segment_trans + rates[target_phase_idx] * segment_trans;
  
  for (int idx = 1; idx < n_state_size; idx++)
  {
	jacobian_row[n_block_size_sq + n_block_size * P_VAR + P_VAR + idx] = rates_derivs[target_phase_idx * n_state_size + idx] * p_diff * segment_trans;
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
  return fabs(rates[target_phase_idx] * p_diff * segment_trans) > target_rate;
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
  const uint8_t n_block_size_sq = n_block_size * n_block_size;

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

	const uint8_t n_block_size_sq = n_block_size * n_block_size;

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
	const uint8_t n_block_size_sq = n_block_size * n_block_size;
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
	const uint8_t n_block_size_sq = n_block_size * n_block_size;

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
	const uint8_t n_block_size_sq = n_block_size * n_block_size;
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
	const uint8_t n_block_size_sq = n_block_size * n_block_size;
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
	const uint8_t n_block_size_sq = n_block_size * n_block_size;

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

int bhp_inj_well_control::add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
	value_t *X_well_head = &X[n_state_size * well_head_idx];
	value_t *RHS_well_head = &RHS[n_state_size * well_head_idx];

	int n_block_size_sq = n_state_size * n_state_size;
	memset(jacobian_row, 0, (2 * n_block_size_sq + n_state_size) * sizeof(value_t));

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
		jacobian_row[2 * (idx + idx * n_state_size)] = 1;
	}

	return 0;
}


int bhp_prod_well_control::add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	index_t n_state_size, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
{
	value_t *X_well_head = &X[n_state_size * well_head_idx];
	value_t *X_well_body = X_well_head + n_state_size;
	value_t *RHS_well_head = &RHS[n_state_size * well_head_idx];
	int n_block_size_sq = n_state_size * n_state_size;

	//for (int i = 0; i < n_state_size; i++)
	//    std::cout << X_well_body[i] << " ";

	memset(jacobian_row, 0, (2 * n_block_size_sq + n_state_size) * sizeof(value_t));

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
		jacobian_row[2 * (idx + idx * n_state_size)] = 1;
	}

	jacobian_row += n_state_size;

	// fill neighbour H block 
	for (int idx = 1; idx < n_state_size; idx++)
	{
		jacobian_row[2 * (idx + idx * n_state_size)] = -1;
	}

	return 0;
}


int rate_inj_well_control::add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_state_size, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{
	value_t *X_well_head = &X[n_state_size * well_head_idx];
	value_t *X_well_body = X_well_head + n_state_size;
	value_t *RHS_well_head = &RHS[n_state_size * well_head_idx];
	int n_block_size_sq = n_variables * n_variables;
	value_t p_diff = X_well_head[0] - X_well_body[0];
	value_t current_rate;

	state.assign(X.begin() + well_head_idx * n_state_size, X.begin() + (well_head_idx + 1) * n_state_size);
	rate_etor->evaluate_with_derivatives(state, block_idx, rates, rates_derivs);
	current_rate = rates[target_phase_idx] * p_diff * segment_trans;

	// first equation - rate constraint
	RHS_well_head[0] = current_rate - target_rate;

	// all the rest - injection stream constraint
	int idx = 1;
	for (value_t is : injection_stream)
	{
		RHS_well_head[idx] = X_well_head[idx] - is;
		idx++;
	}

	// fill diagonal H block - it`s always the first
	memset(jacobian_row, 0, (2 * n_block_size_sq + n_state_size) * sizeof(value_t));
	jacobian_row[0] = rates_derivs[target_phase_idx * n_state_size] * p_diff * segment_trans + rates[target_phase_idx] * segment_trans;
	jacobian_row[n_state_size] = -rates[target_phase_idx] * segment_trans;
	for (int idx = 1; idx < n_state_size; idx++)
	{
		jacobian_row[idx] = rates_derivs[target_phase_idx * n_state_size + idx] * p_diff * segment_trans;
		jacobian_row[2 * (idx + idx * n_state_size)] = 1;
	}

	return 0;
}
int gt_bhp_prod_well_control::add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_block_size, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{
  value_t* X_well_head = &X[well_head_idx * n_block_size];
  value_t* X_body_head = X_well_head + n_block_size;
  value_t* RHS_well_head = &RHS[well_head_idx * n_block_size];
  int n_vars_sq = n_block_size * n_block_size;
  memset(jacobian_row, 0, (2 * n_vars_sq + n_block_size) * sizeof(value_t));

  // RHS
  RHS_well_head[0] = X_well_head[0] - target_pressure;
  for (int idx = 1; idx < n_block_size; idx++)
  {
    RHS_well_head[idx] = X_well_head[idx] - X_body_head[idx];
  }

  // fill diagonal H block - it`s always the first
  for (int idx = 0; idx < n_block_size; idx++)
  {
    jacobian_row[2 * (idx + idx * n_block_size)] = 1;
  }

  jacobian_row += n_block_size;

  // fill neighbour H block 
  for (int idx = 1; idx < n_block_size; idx++)
  {
    jacobian_row[2 * (idx + idx * n_block_size)] = -1;
  }

  return 0;
}

int gt_bhp_temp_inj_well_control::add_to_csr_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_block_size, std::vector<value_t>& X, value_t * jacobian_row, std::vector<value_t>& RHS)
{

  int temp_idx = 0;
  value_t *X_well_head = &X[n_block_size * well_head_idx];
  //value_t *X_body_head = X_well_head + n_block_size;
  value_t *RHS_well_head = &RHS[n_block_size * well_head_idx];

  int n_vars_sq = n_block_size * n_block_size;

  state.assign(X.begin() + well_head_idx * n_block_size, X.begin() + (well_head_idx + 1) * n_block_size);
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
  memset(jacobian_row, 0, (2 * n_vars_sq + n_block_size) * sizeof(value_t));

  jacobian_row[0] = 1;
  for (int idx = 0; idx < n_block_size; idx++)
  {
    // for energy row
    jacobian_row[idx + (n_block_size * 2 + 1)] = rate_temp_ops_derivs[(temp_idx)* n_block_size + idx];
  }

  return 0;
}