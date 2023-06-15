#ifndef MECH_OPERATORS_HPP_
#define MECH_OPERATORS_HPP_

#include <vector>
#include <array>
#include "conn_mesh.h"
#include "mech/pm_discretizer.hpp"

namespace pm
{
	class mech_operators
	{
	protected:
		conn_mesh* mesh;
		pm_discretizer* discr;
		uint8_t P_VAR, Z_VAR, U_VAR;
		uint8_t P_VAR_T, U_VAR_T;
		uint8_t N_VARS, N_VARS_SQ, N_OPS, NC;
		uint8_t ACC_OP, FLUX_OP, GRAV_OP;
		static const uint8_t ND = 3;
		static const uint8_t NT = 4;
		static const uint8_t N_TRANS_SQ = NT * NT;

		std::map<uint8_t, Matrix> pre_N, pre_R, pre_Ft, pre_F, pre_Nflux, pre_Q;
	public:
		mech_operators();
		~mech_operators();
		void init(conn_mesh* _mesh, pm_discretizer* _discr, uint8_t _P_VAR, uint8_t _Z_VAR, uint8_t _U_VAR, 
			uint8_t _N_VARS, uint8_t _N_OPS, uint8_t _NC, uint8_t _ACC_OP, uint8_t _FLUX_OP, uint8_t _GRAV_OP);

		void init(conn_mesh* _mesh, pm_discretizer* _discr, uint8_t _P_VAR, uint8_t _Z_VAR, uint8_t _U_VAR,
			uint8_t _P_VAR_T, uint8_t _U_VAR_T,	uint8_t _N_VARS, uint8_t _N_OPS, uint8_t _NC, uint8_t _ACC_OP, uint8_t _FLUX_OP, uint8_t _GRAV_OP);
		
		// prepare matrices for reconstruction
		std::vector<Matrix> mat_stress, mat_flux;
		void prepare();

		void eval_stresses(const std::vector<value_t>& fluxes, const std::vector<value_t>& fluxes_biot, std::vector<value_t>& X, std::vector<value_t>& bc_rhs, const std::vector<value_t>& op_vals_arr);
		void eval_porosities(std::vector<value_t>& X, std::vector<value_t>& bc_rhs);
		void eval_unknowns_on_faces(std::vector<value_t>& X, std::vector<value_t>& bc_rhs, std::vector<value_t>& Xref);

		std::vector<std::vector<value_t>> pressures;
		std::vector<value_t> stresses, total_stresses, velocity;
		std::vector<value_t> eps_vol, porosities;
		std::vector<value_t> face_unknowns;
	};
};

#endif /* MECH_OPERATORS_HPP_ */