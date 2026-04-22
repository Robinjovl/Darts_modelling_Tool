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

std::string well_control_iface::get_well_control_target_str()
{
	if (this->control_type == WellControlType::NONE)
	{
		return "";
	}
	else if (this->control_type >= WellControlType::BHP && this->control_type < WellControlType::NUMBER_OF_RATE_TYPES)
	{
		return std::to_string(this->target);
	}
	else
	{
		throw std::runtime_error("Undefined well control type");
	}
}

void well_control_iface::extract_Xopw(std::vector<value_t>& X)
{
	// Build extended well state vector Xopw by appending sgw_max (=1 for wells, hysteresis always on drainage branch)
	const uint8_t nc = n_comps;
	const uint8_t z_var = nc - 1;
	const uint8_t p_var = 0;
	const int state_size = n_vars + 1;
	const int n_block = static_cast<int>(X.size()) / n_vars;

	sgw_max.resize(n_block);
	std::fill(sgw_max.begin(), sgw_max.end(), value_t(1));

	if (static_cast<int>(Xopw.size()) < n_block * state_size)
	{
		Xopw.resize(n_block * state_size);
		xopw_ders_arr.resize(n_block * n_ops * state_size);
	}

	for (index_t i = 0; i < n_block; i++)
	{
		Xopw[i * state_size] = X[i * n_vars + p_var];
		for (uint8_t c = 0; c < nc - 1; c++)
		{
			Xopw[i * state_size + c + 1] = X[i * n_vars + z_var + c];
		}
		Xopw[i * state_size + state_size - 1] = sgw_max[i];
	}

	if (n_vars > nc)
	{
		for (index_t i = 0; i < n_block; i++)
		{
			Xopw[i * state_size + nc] = X[i * n_vars + nc];
		}
	}
}

void well_control_iface::extract_xopw_ders()
{
	// Map extended well derivatives back to standard well_control_ops_derivs, dropping sgw_max column
	const int state_size = n_vars + 1;
	const index_t n_block0 = n_ops * n_vars;
	const index_t n_block1 = n_ops * state_size;

	for (index_t i = 0; i < static_cast<index_t>(block_idx.size()); i++)
		for (index_t op = 0; op < n_ops; op++)
			for (index_t v = 0; v < n_vars; v++)
				well_control_ops_derivs[i * n_block0 + op * n_vars + v] = xopw_ders_arr[i * n_block1 + op * state_size + v];
}

int well_control_iface::add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
	uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t* jacobian_row, std::vector<value_t>& RHS)
{
	(void)dt;
	// n_vars is number of flow variables
	// n_block_size is size of block which includes flow and mechanics variables
	value_t* X_well_head = &X[n_block_size * well_head_idx + P_VAR];
	value_t* X_well_body = X_well_head + n_block_size;
	value_t* RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];

	// fill the jacobian
	const uint16_t n_block_size_sq = n_block_size * n_block_size;
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

	// Set first specification from well controls (defined in operators)
	if (this->control_type == WellControlType::BHP)
	{
		// If BHP controlled - pressure constraint
		if (hysteresis_enabled)
		{
			// Hysteresis: use extended state Xopw (n_vars+1, sg_max=1 for wells)
			extract_Xopw(X);
			xopw_ders_arr.resize(n_ops * (n_vars + 1));
			state.assign(Xopw.begin() + (well_head_idx + 0) * (n_block_size + 1) + P_VAR,
			             Xopw.begin() + (well_head_idx + 0) * (n_block_size + 1) + P_VAR + n_vars + 1);
			well_controls_etor->evaluate_with_derivatives(state, block_idx, well_control_ops, xopw_ders_arr);
			extract_xopw_ders();
		}
		else
		{
			// Standard: use normal state (n_vars)
			state.assign(X.begin() + (well_head_idx + 0) * n_block_size + P_VAR,
			             X.begin() + (well_head_idx + 0) * n_block_size + P_VAR + n_vars);
			well_controls_etor->evaluate_with_derivatives(state, block_idx, well_control_ops, well_control_ops_derivs);
		}

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
		if (hysteresis_enabled)
		{
			// Hysteresis: use extended state Xopw (n_vars+1, sg_max=1 for wells)
			extract_Xopw(X);
			xopw_ders_arr.resize(n_ops * (n_vars + 1));
			state.assign(Xopw.begin() + (well_head_idx + 0) * (n_block_size + 1) + P_VAR,
			             Xopw.begin() + (well_head_idx + 0) * (n_block_size + 1) + P_VAR + n_vars + 1);
			well_controls_etor->evaluate_with_derivatives(state, block_idx, well_control_ops, xopw_ders_arr);
			extract_xopw_ders();
		}
		else
		{
			// Standard: use normal state (n_vars)
			state.assign(X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR,
			             X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR + n_vars);
			well_controls_etor->evaluate_with_derivatives(state, block_idx, well_control_ops, well_control_ops_derivs);
		}
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
			RHS_well_head[ii] = X_well_head[ii] - this->inj_comp[ii - 1];
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
 										     	   uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
	(void)dt;
	value_t* X_well_head = &X[n_block_size * well_head_idx + P_VAR];
	value_t* X_well_body = X_well_head + n_block_size;
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

int well_control_iface::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
{
	// Fill target state with target BHP/rate pressure, composition and target temperature
	std::vector<value_t> target_state(n_vars);

	// Pressure initialization
	if (this->control_type == WellControlType::BHP)
	{
		// BHP-controlled: set bhp
		target_state[0] = this->target;
	}
	else
	{
		// Rate-controlled: initialize with pressure of neighbouring cell ensuring the correct flow direction
		target_state[0] = (this->target > 0.) ? state_neighbour[0] + 0.001 : state_neighbour[0] * 0.99;
	}

	// Other state specifications
	if (this->well_state_offset == 1)  // if production well
	{
		// PRODUCTION WELL
		// Initialize production well with state of neighbouring cell
		for (int i = 1; i < n_vars; i++)
		{
			target_state[i] = state_neighbour[i];
		}
	}
	else
	{
		// INJECTION WELL
		// Initialize injection well with injection stream
		for (int i = 1; i < n_vars - thermal; i++)
		{
			target_state[i] = this->inj_comp[i - 1];
		}

		// For temperature/enthalpy, use specified control
		if (this->thermal)
		{
			// Evaluate ThermalVarOperator to initialize temperature/enthalpy of well head according to specified injection conditions
			target_state[n_vars - 1] = inj_temp;
			std::vector<value_t> thermal_var_op(1);
			this->thermal_var_etor->evaluate(target_state, thermal_var_op);

			target_state[n_vars - 1] = thermal_var_op[0];
		}
	}

	// Fill state block with target state vector
	for (index_t i = 0; i < n_vars; i++)
	{
		state_block[i] = target_state[i];
	}

	return 0;
}
