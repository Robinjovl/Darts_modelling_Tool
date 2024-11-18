#ifndef CONTACT_H_
#define CONTACT_H_

#include <vector>
#include "mech/pm_discretizer.hpp"
#include "conn_mesh.h"

#ifdef OPENDARTS_LINEAR_SOLVERS
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#else
#include "linsolv_iface.h"
#endif // OPENDARTS_LINEAR_SOLVERS

#ifdef OPENDARTS_LINEAR_SOLVERS
using namespace opendarts::linear_solvers;
#endif // OPENDARTS_LINEAR_SOLVERS

namespace pm
{
	enum ContactState { TRUE_STUCK, PEN_STUCK, SLIP, FREE };
	enum FrictionModel { FRICTIONLESS, STATIC, SLIP_DEPENDENT, RSF, RSF_STAB, CNS };
	enum StateLaw { AGEING_LAW, SLIP_LAW, MIXED };
	enum ContactSolver { FLUX_FROM_PREVIOUS_ITERATION, RETURN_MAPPING, LOCAL_ITERATIONS };
	enum CriticalStress { TERZAGHI, BIOT };
	enum NormalCondition { PENALIZED, ZERO_GAP_CHANGE };

	struct RSF_props 
	{ 
		value_t a, b, vel0, Dc; 
		std::vector<value_t> theta, theta_n;
		std::vector<value_t> mu_rate, mu_state;
		value_t min_vel;
		StateLaw law;
	};
	struct SlipDependentFriction_props
	{
		value_t mu_d, Dc;
	};

	/*
	* Class that defines the contact condition for mechanics
	* Serves the set of contact segments that belong to single fault
	*/
	class contact
	{
	protected:
		static const uint8_t ND = 3;
		static const uint8_t MAX_LOCAL_ITER_NUM = 10;
		value_t dgt_norm, dgt_iter_norm, Ft_trial_norm;
		Matrix Fcoef, Fpres_coef, Frhs, flux, flux_n, dg, dg_iter, I3, F_trial;
		std::vector<index_t> st, ind;
		std::map<uint8_t, Matrix> pre_F, pre_Fpres, pre_Fn, pre_Fn_pres;

		index_t file_id;

		pm_discretizer* discr;
		conn_mesh* mesh;

		// data necessary for merging tractions
		index_t n_blocks, n_matrix, n_res_blocks;
		uint8_t permut[ND];
		index_t *block_m, *block_p, *stencil, *offset;
		value_t *tran, *tran_biot, *rhs, *rhs_biot;
		// jacobian arrays
		value_t *Jac;
		index_t *diag_ind, *rows, *cols, diag_idx;

		void merge_tractions_biot(const index_t i,
			const std::vector<value_t>& fluxes, const std::vector<value_t>& fluxes_biot, const std::vector<value_t>& X,
			const std::vector<value_t>& fluxes_n, const std::vector<value_t>& fluxes_biot_n, const std::vector<value_t>& Xn,
			const std::vector<value_t>& fluxes_ref, const std::vector<value_t>& fluxes_biot_ref, const std::vector<value_t>& Xref,
			const std::vector<value_t>& fluxes_ref_n, const std::vector<value_t>& fluxes_biot_ref_n, const std::vector<value_t>& Xn_ref);
		void merge_tractions_terzaghi(const index_t i,
			const std::vector<value_t>& fluxes, const std::vector<value_t>& fluxes_biot, const std::vector<value_t>& X,
			const std::vector<value_t>& fluxes_n, const std::vector<value_t>& fluxes_biot_n, const std::vector<value_t>& Xn,
			const std::vector<value_t>& fluxes_ref, const std::vector<value_t>& fluxes_biot_ref, const std::vector<value_t>& Xref,
			const std::vector<value_t>& fluxes_ref_n, const std::vector<value_t>& fluxes_biot_ref_n, const std::vector<value_t>& Xn_ref);
		int add_to_jacobian_slip(index_t id, value_t dt, std::vector<value_t>& RHS);
		int add_to_jacobian_stuck(index_t id, value_t dt, std::vector<value_t>& RHS);

		// for local iterations
		value_t calc_gap_L2_residual(const std::vector<value_t>& RHS) const;
		int init_local_jacobian_structure();
		//value_t update_fluxes();
		std::vector<value_t> dg_local, rhs_local;
		linsolv_iface* local_solver;
		csr_matrix_base *local_jacobian;
		timer_node* timer;
		index_t output_counter;

		std::vector<value_t> getFrictionCoef(const index_t i, const value_t dt, Matrix slip_vel, const Matrix& slip);
		std::vector<value_t> getStabilizedFrictionCoef(const index_t i, const value_t dt, Matrix slip_vel, const Matrix& slip);
	public:
		uint8_t N_VARS, U_VAR, P_VAR, N_VARS_SQ;
		uint8_t NT, U_VAR_T, P_VAR_T, NT_SQ;
		std::vector<index_t> cell_ids;
		std::vector<ContactState> states, states_n;
		std::vector<Matrix> S, Sinv, S_fault;

		//// friction properties
		// friction model
		FrictionModel friction_model;
		// static friction coefficients for every fault segment
		std::vector<value_t> mu0;
		// total friction coefficients for every fault segment
		std::vector<value_t> mu;
		// sliding potentials for every fault segment
		std::vector<value_t> phi;
		// tangential & normal penalty parameters for every fault segment
		std::vector<value_t> eps_t, eps_n;
		// radiation damping coefficient
		std::vector<value_t> eta;
		// flattened vector of stresses over the fault
		std::vector<value_t> fault_stress;
		// RSF props
		RSF_props rsf;
		SlipDependentFriction_props sd_props;
		// effective stresses used for friction criterion and beyond
		CriticalStress friction_criterion;
		// the type of normal condition used
		NormalCondition normal_condition;
		// fault tag to identify one particular fault in multi-fault system
		index_t fault_tag;

		// penalty parameter ~ f_scale
		value_t f_scale;
		// maximum gap change allowed per iteration
		value_t max_allowed_gap_change;
		// number of segments that with different gap direction
		index_t num_of_change_sign;
		// multiplier to switch to fully-explicit scheme
		value_t implicit_scheme_multiplier;
		// diagonal values of jacobian in the case of fully-explicit scheme
		std::vector<Matrix33> jacobian_explicit_scheme;

		contact();
		~contact();
		int init_friction(pm_discretizer* _discr, conn_mesh* _mesh);
		int init_fault();
		int init_local_iterations();

		// returm-mapping algorithm for penalized contact constraints
		int add_to_jacobian_return_mapping(value_t dt, csr_matrix_base* jacobian, std::vector<value_t>& RHS, const std::vector<value_t>& X, const std::vector<value_t>& fluxes, const std::vector<value_t>& fluxes_biot,
																								const std::vector<value_t>& Xn, const std::vector<value_t>& fluxes_n, const std::vector<value_t>& fluxes_biot_n,
																									std::vector<value_t>& Xref, std::vector<value_t>& fluxes_ref, std::vector<value_t>& fluxes_biot_ref,
																										const std::vector<value_t>& Xn_ref, const std::vector<value_t>& fluxes_ref_n, const std::vector<value_t>& fluxes_biot_ref_n);
		// semi-explicit scheme
		int add_to_jacobian_linear(value_t dt, csr_matrix_base* jacobian, std::vector<value_t>& RHS, const std::vector<value_t>& X, const std::vector<value_t>& fluxes, const std::vector<value_t>& fluxes_biot,
																								const std::vector<value_t>& Xn, const std::vector<value_t>& fluxes_n, const std::vector<value_t>& fluxes_biot_n,
																									std::vector<value_t>& Xref, std::vector<value_t>& fluxes_ref, std::vector<value_t>& fluxes_biot_ref,
																										const std::vector<value_t>& Xn_ref, const std::vector<value_t>& fluxes_ref_n, const std::vector<value_t>& fluxes_biot_ref_n);
		// return-mapping with local iterations for gap vector
		int add_to_jacobian_local_iters(value_t dt, csr_matrix_base* jacobian, std::vector<value_t>& RHS, std::vector<value_t>& X, std::vector<value_t>& fluxes, std::vector<value_t>& fluxes_biot,
																											const std::vector<value_t>& Xn, const std::vector<value_t>& fluxes_n, const std::vector<value_t>& fluxes_biot_n,
																												std::vector<value_t>& Xref, std::vector<value_t>& fluxes_ref, std::vector<value_t>& fluxes_biot_ref,
																													const std::vector<value_t>& Xn_ref, const std::vector<value_t>& fluxes_ref_n, const std::vector<value_t>& fluxes_biot_ref_n);
		int solve_explicit_scheme(std::vector<value_t>& RHS, std::vector<value_t>& dX);

		
		void set_state(const ContactState& state);
		
		int apply_direction_chop(const std::vector<value_t>& X, const std::vector<value_t>& Xn, std::vector<value_t>& dX);
	};
};

#endif /* CONTACT_H_ */
