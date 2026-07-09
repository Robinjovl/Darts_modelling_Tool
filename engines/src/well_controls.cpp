#include "well_controls.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

int well_control_iface::set_bhp_control(bool is_inj, value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_)
{
	this->well_state_offset = (is_inj) ? 0 : 1; // if injection well, evaluates operators with state of well head; for production, it uses well body
	this->control_type = well_control_iface::WellControlType::BHP;
	this->phase_idx = std::nullopt;  // phase_idx is not used for BHP control

	// Fill well control spec
	this->target = target_;
	this->inj_comp = inj_comp_;
	this->inj_temp = inj_temp_;
	return 0;
}

int well_control_iface::set_rate_control(bool is_inj, well_control_iface::WellControlType control_type_, std::optional<index_t> phase_idx_, value_t target_, std::vector<value_t>& inj_comp_, value_t inj_temp_)
{
	this->well_state_offset = (is_inj) ? 0 : 1;  // if injection well, evaluates operators with state of well head; for production, it uses well body
	this->control_type = control_type_;
	this->phase_idx = phase_idx_;  // if phase_idx is nullopt, total rate is controlled

    // Fill well control spec
	this->target = target_;
	this->inj_comp = inj_comp_;
	this->inj_temp = inj_temp_;
	return 0;
}

index_t well_control_iface::get_rate_ctrl_op_idx(WellControlType ctrl_type, index_t phase_idx_, bool is_dfm_well) const
{
	const index_t offset = is_dfm_well ? dfm_rate_ctrl_ops_offset() : epm_rate_ctrl_ops_offset();
	return offset + ctrl_type * n_phases + phase_idx_;
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
	else if (this->control_type > WellControlType::BHP && this->control_type < WellControlType::NUMBER_OF_RATE_TYPES && this->phase_idx.has_value())
	{
		out = "phase " + std::to_string(phase_idx.value());
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
	else if (this->control_type > WellControlType::BHP && this->control_type < WellControlType::NUMBER_OF_RATE_TYPES && !this->phase_idx.has_value())
	{
		switch (this->control_type)
		{
		case WellControlType::MOLAR_RATE:
		{
			out = "total molar rate-control";
			break;
		}
		case WellControlType::MASS_RATE:
		{
			out = "total mass rate-control";
			break;
		}
		case WellControlType::VOLUMETRIC_RATE:
		{
			out = "total volumetric rate-control";
			break;
		}
		default:
		{
			out = "total advective heat rate-control";
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

int well_control_iface::initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour, bool is_dfm_well)
{
	// Fill target state with target BHP/rate pressure, composition and target temperature.
	// Start from the existing block state so DFM production keeps initialized non-pressure variables.
	std::vector<value_t> target_state = state_block;

	// Pressure initialization
	if (this->control_type == WellControlType::BHP)
	{
		// BHP-controlled: set bhp
		target_state[0] = this->target;
	}
	else
	{
		if (is_dfm_well)
		{
			// Rate-controlled DFM wells keep the initialized wellhead pressure.
			target_state[0] = state_block[0];
		}
		else
		{
			// Rate-controlled EPM wells are initialized with pressure of the neighbouring cell,
			// ensuring the correct flow direction.
			target_state[0] = (this->target > 0.) ? state_neighbour[0] + 0.001 : state_neighbour[0] * 0.99;
		}
	}

	// Other state specifications
	if (this->well_state_offset == 1)  // if production well
	{
		if (!is_dfm_well)
		{
			// EPM production wells use the neighbouring state.
			for (int i = 1; i < n_vars; i++)
			{
				target_state[i] = state_neighbour[i];
			}
		}
		// DFM production wells keep initialized non-pressure variables.
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

int well_control_iface::check_constraint_violation(value_t dt, index_t well_head_idx, value_t well_transmissibility,
	uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X)
{
	(void)dt;
	value_t* X_well_head = &X[n_block_size * well_head_idx + P_VAR];
	value_t* X_well_body = X_well_head + n_block_size;
	value_t p_diff = X_well_head[0] - X_well_body[0];

	if (this->control_type == WellControlType::BHP)
	{
		// Check if BHP constraint is violated. EPM is the only supported constraint path for now.
		state.assign(X.begin() + (well_head_idx + 0) * n_block_size + P_VAR, X.begin() + (well_head_idx + 0) * n_block_size + P_VAR + n_vars);
		// Append OBL history values when history axes are active (no-op for plain drainage)
		state.insert(state.end(), Xhistory_well_default.begin(), Xhistory_well_default.end());
		well_ctrl_etor->evaluate(state, well_ctrl_ops);
		index_t pres_op_idx = get_pres_ctrl_op_idx();

		return (p_diff > 0.) ?
			well_ctrl_ops[pres_op_idx] > this->target : // injection well
		well_ctrl_ops[pres_op_idx] < this->target;  // production well
	}
	else
	{
		// Check if rate constraint is violated. EPM is the only supported constraint path for now.
		state.assign(X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR, X.begin() + (well_head_idx + well_state_offset) * n_block_size + P_VAR + n_vars);
		// Append OBL history values when history axes are active (no-op for plain drainage)
		state.insert(state.end(), Xhistory_well_default.begin(), Xhistory_well_default.end());
		well_ctrl_etor->evaluate(state, well_ctrl_ops);
		if (phase_idx.has_value())
		{
			index_t rate_ctrl_op_idx = get_rate_ctrl_op_idx(this->control_type, phase_idx.value(), false);

			return (this->target > 0.) ?
				well_ctrl_ops[rate_ctrl_op_idx] * p_diff * well_transmissibility > this->target : // injection well
			well_ctrl_ops[rate_ctrl_op_idx] * p_diff * well_transmissibility < this->target;  // production well
		}
		else
		{
			// total-rate: sum rates over phases and compare
			value_t total_rate = 0.;
			for (index_t p = 0; p < n_phases; ++p)
			{
				index_t rate_ctrl_op_idx = get_rate_ctrl_op_idx(this->control_type, p, false);
				total_rate += well_ctrl_ops[rate_ctrl_op_idx] * p_diff * well_transmissibility;
			}

			return (this->target > 0.) ?
				total_rate > this->target :
			total_rate < this->target;
		}
	}
}

int well_control_iface::add_to_jacobian(value_t dt, index_t well_head_idx, value_t well_transmissibility,
	uint8_t n_block_size, uint8_t P_VAR, std::vector<value_t>& X, value_t* jacobian_row, std::vector<value_t>& RHS,
	std::vector<value_t>& phases_vels, std::vector<value_t>& phases_vels_ders, bool is_dfm_well)
{
	(void)dt;
	// n_vars is number of variables in the well state
	// n_block_size is size of block which includes flow and mechanics variables
	value_t* X_well_head = &X[n_block_size * well_head_idx + P_VAR];
	value_t* X_well_body = X_well_head + n_block_size;
	value_t* RHS_well_head = &RHS[n_block_size * well_head_idx + P_VAR];

	// Clear the two local column blocks of the wellhead equation row before replacing
	// that row with control equations. This does not touch the body block's own conservation-equation rows.
	const uint16_t n_block_size_sq = n_block_size * n_block_size;
	memset(jacobian_row, 0, 2 * n_block_size_sq * sizeof(value_t));

	const bool is_bhp_ctrl = (this->control_type == WellControlType::BHP);
	// BHP ctrl always uses the pressure at the wellhead.
	// Rate control uses the upstream state depending on whether the well is for injection or production.
	const index_t ctrl_state_block_offset = is_bhp_ctrl ? 0 : well_state_offset;
	const index_t ctrl_state_col_offset = ctrl_state_block_offset * n_block_size_sq;
	state.assign(X.begin() + (well_head_idx + ctrl_state_block_offset) * n_block_size + P_VAR,
	    X.begin() + (well_head_idx + ctrl_state_block_offset) * n_block_size + P_VAR + n_vars);

	// Hysteresis: when OBL history axes are active, the well-ctrl etor expects an extended state
	// [X | Xhistory] and writes derivatives of size n_well_ctrl_ops * (n_vars + n_history).
	// well_ctrl_ops_derivs is sized for n_vars columns only, so project the leading n_vars columns back.
	const uint8_t n_history = (uint8_t)Xhistory_well_default.size();
	if (n_history == 0)
	{
		well_ctrl_etor->evaluate_with_derivatives(state, block_idx, well_ctrl_ops, well_ctrl_ops_derivs);
	}
	else
	{
		state.insert(state.end(), Xhistory_well_default.begin(), Xhistory_well_default.end());
		const uint8_t n_state = n_vars + n_history;
		std::vector<value_t> ops_derivs_ext(n_well_ctrl_ops * n_state, 0.0);
		well_ctrl_etor->evaluate_with_derivatives(state, block_idx, well_ctrl_ops, ops_derivs_ext);
		for (index_t op = 0; op < n_well_ctrl_ops; op++)
			for (index_t v = 0; v < n_vars; v++)
				well_ctrl_ops_derivs[op * n_vars + v] = ops_derivs_ext[op * n_state + v];
	}

	// The first wellhead equation is the BHP/rate ctrl residual. Remaining equations are filled below.
	if (is_bhp_ctrl)
	{
		// BHP ctrl: constrain the pressure at the wellhead.
		index_t pres_op_idx = get_pres_ctrl_op_idx();
		RHS_well_head[0] = well_ctrl_ops[pres_op_idx] - this->target;

		// BHP operator derivatives
		for (int jj = 0; jj < n_vars; jj++)
		{
			jacobian_row[n_block_size * P_VAR + P_VAR + jj] = well_ctrl_ops_derivs[pres_op_idx * n_vars + jj];
		}
	}
	else if (!is_dfm_well)
	{
		// EPM rate ctrl: constrain the phase/total rate at the wellhead connection of EPM wells
		value_t p_diff = X_well_head[0] - X_well_body[0];

		if (phase_idx.has_value())
		{
			// Phase rate ctrl: constrain only the rate of the selected phase.
			index_t rate_ctrl_op_idx = get_rate_ctrl_op_idx(this->control_type, phase_idx.value(), false);

			// RHS
			RHS_well_head[0] = well_ctrl_ops[rate_ctrl_op_idx] * p_diff * well_transmissibility - this->target;

			// Rate operator derivatives belong to the block whose state was used for well_ctrl_ops.
			for (int jj = 0; jj < n_vars; jj++)
			{
				jacobian_row[n_block_size * P_VAR + P_VAR + ctrl_state_col_offset + jj] += well_ctrl_ops_derivs[rate_ctrl_op_idx * n_vars + jj] * p_diff * well_transmissibility;
			}
			// Product rule for pressure variable
			jacobian_row[n_block_size * P_VAR + P_VAR] += well_ctrl_ops[rate_ctrl_op_idx] * well_transmissibility;
			jacobian_row[n_block_size * P_VAR + P_VAR + n_block_size_sq] += -well_ctrl_ops[rate_ctrl_op_idx] * well_transmissibility;

			// if target phase does not exist, set a constant small value to pressure derivative
			// it will let the pressure drop and eventually pressure constraint might work
			if (this->well_state_offset && std::fabs(jacobian_row[n_block_size * P_VAR + P_VAR]) < 1e-3)
			{
				jacobian_row[n_block_size * P_VAR + P_VAR] = 1.;
			}
		}
		else
		{
			// Total rate ctrl: constrain total phase rates.
			value_t total_rate = 0.0;  // accumulate total well rate over all phases

			for (index_t p = 0; p < n_phases; p++)
			{
				index_t rate_ctrl_op_idx = get_rate_ctrl_op_idx(this->control_type, p, false);

				total_rate += well_ctrl_ops[rate_ctrl_op_idx] * p_diff * well_transmissibility;

				// Rate ctrl operator derivatives belong to the block whose state was used for well_ctrl_ops.
				for (int jj = 0; jj < n_vars; jj++)
				{
					jacobian_row[n_block_size * P_VAR + P_VAR + ctrl_state_col_offset + jj] += well_ctrl_ops_derivs[rate_ctrl_op_idx * n_vars + jj] * p_diff * well_transmissibility;
				}
				// Product rule for pressure variable
				jacobian_row[n_block_size * P_VAR + P_VAR] += well_ctrl_ops[rate_ctrl_op_idx] * well_transmissibility;
				jacobian_row[n_block_size * P_VAR + P_VAR + n_block_size_sq] += -well_ctrl_ops[rate_ctrl_op_idx] * well_transmissibility;
			}
			// RHS
			RHS_well_head[0] = total_rate - this->target;
		}
	}
	else
	{
		// DFM rate ctrl: constrain the phase/total rate at the wellhead connection of DFM wells
		// TODO: This state must be chosen based on the sign of phase velocity for each phase,
		// not based on the injection/production type of the well. This matters because I have seen particularly at the beginning of simulation
		// where there is a lot of instability and there is upward fluid flow for an injection well and using the upwind scheme is important for stability.
		index_t n_conns = phases_vels.size() / n_phases;
		index_t well_head_conn_idx_local = 0;

		// Strides needed for finding velocity derivatives
		index_t phase_stride = n_conns * 2 * n_vars;
		index_t conn_stride = 2 * n_vars;

		if (phase_idx.has_value())
		{
			// Phase rate ctrl: constrain only the rate of the selected phase.
			index_t rate_ctrl_op_idx = get_rate_ctrl_op_idx(this->control_type, phase_idx.value(), true);

			value_t phase_vel = phases_vels[n_conns * phase_idx.value() + well_head_conn_idx_local];

			// RHS
			RHS_well_head[0] = well_ctrl_ops[rate_ctrl_op_idx] * phase_vel * well_transmissibility - this->target;

			// Rate operator derivatives belong to the block whose state was used for well_ctrl_ops.
			for (int jj = 0; jj < n_vars; jj++)
			{
				jacobian_row[n_block_size * P_VAR + P_VAR + ctrl_state_col_offset + jj] += well_ctrl_ops_derivs[rate_ctrl_op_idx * n_vars + jj] * phase_vel * well_transmissibility;

				value_t vel_der_head = phases_vels_ders[phase_idx.value() * phase_stride + well_head_conn_idx_local * conn_stride + 0 * n_vars + jj];
				jacobian_row[n_block_size * P_VAR + P_VAR + jj] += well_ctrl_ops[rate_ctrl_op_idx] * vel_der_head * well_transmissibility;

				value_t vel_der_body = phases_vels_ders[phase_idx.value() * phase_stride + well_head_conn_idx_local * conn_stride + 1 * n_vars + jj];
				jacobian_row[n_block_size * P_VAR + P_VAR + n_block_size_sq + jj] += well_ctrl_ops[rate_ctrl_op_idx] * vel_der_body * well_transmissibility;
			}
		}
		else
		{
			// Total rate ctrl: constrain total phase rates.
			value_t total_rate = 0.0;  // accumulate total well rate over all phases

			for (index_t p = 0; p < n_phases; p++)
			{
				index_t rate_ctrl_op_idx = get_rate_ctrl_op_idx(this->control_type, p, true);

				value_t phase_vel = phases_vels[n_conns * p + well_head_conn_idx_local];

				total_rate += well_ctrl_ops[rate_ctrl_op_idx] * phase_vel * well_transmissibility;

				// Rate operator derivatives belong to the block whose state was used for well_ctrl_ops.
				for (int jj = 0; jj < n_vars; jj++)
				{
					jacobian_row[n_block_size * P_VAR + P_VAR + ctrl_state_col_offset + jj] += well_ctrl_ops_derivs[rate_ctrl_op_idx * n_vars + jj] * phase_vel * well_transmissibility;

					value_t vel_der_head = phases_vels_ders[p * phase_stride + well_head_conn_idx_local * conn_stride + 0 * n_vars + jj];
					jacobian_row[n_block_size * P_VAR + P_VAR + jj] += well_ctrl_ops[rate_ctrl_op_idx] * vel_der_head * well_transmissibility;

					value_t vel_der_body = phases_vels_ders[p * phase_stride + well_head_conn_idx_local * conn_stride + 1 * n_vars + jj];
					jacobian_row[n_block_size * P_VAR + P_VAR + n_block_size_sq + jj] += well_ctrl_ops[rate_ctrl_op_idx] * vel_der_body * well_transmissibility;
				}
			}
			// RHS
			RHS_well_head[0] = total_rate - this->target;
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
			index_t temp_op_idx = get_temp_ctrl_op_idx();
			RHS_well_head[ii] = well_ctrl_ops[temp_op_idx] - this->inj_temp;

			for (int jj = 0; jj < n_vars; jj++)
			{
				jacobian_row[n_block_size * (P_VAR + ii) + P_VAR + jj] = well_ctrl_ops_derivs[temp_op_idx * n_vars + jj];
			}
		}
	}

	return 0;
}
