#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include "mech/contact.h"
//#include "mech/pm_discretizer.hpp"

namespace py = pybind11;
using namespace pm;

PYBIND11_MAKE_OPAQUE(std::vector<Matrix>);

void pybind_contact(py::module &m)
{
	py::class_<contact>(m, "contact") \
		.def(py::init<>()) \
		.def("init_fault", (int(contact::*)())& contact::init_fault) \
		.def("init_local_iterations", (int(contact::*)())& contact::init_local_iterations) \
		.def("init_friction", (int(contact::*)(pm_discretizer*, conn_mesh*))& contact::init_friction) \
		.def("set_state", &contact::set_state) \
		.def_readwrite("cell_ids", &contact::cell_ids) \
		.def_readwrite("f_scale", &contact::f_scale) \
		.def_readwrite("implicit_scheme_multiplier", &contact::implicit_scheme_multiplier) \
		.def_readwrite("num_of_change_sign", &contact::num_of_change_sign) \
		.def_readwrite("N_VARS", &contact::N_VARS) \
		.def_readwrite("U_VAR", &contact::U_VAR) \
		.def_readwrite("P_VAR", &contact::P_VAR) \
		.def_readwrite("NT", &contact::NT) \
		.def_readwrite("U_VAR_T", &contact::U_VAR_T) \
		.def_readwrite("P_VAR_T", &contact::P_VAR_T) \
		.def_readwrite("friction_model", &contact::friction_model) \
		.def_readwrite("friction_criterion", &contact::friction_criterion) \
		.def_readwrite("mu0", &contact::mu0) \
		.def_readwrite("mu", &contact::mu) \
		.def_readwrite("fault_stress", &contact::fault_stress) \
		.def_readwrite("phi", &contact::phi) \
		.def_readwrite("S", &contact::S) \
		.def_readwrite("states", &contact::states) \
		.def_readwrite("fault_tag", &contact::fault_tag) \
		.def_readwrite("eps_t", &contact::eps_t) \
		.def_readwrite("eps_n", &contact::eps_n) \
		.def_readwrite("eta", &contact::eta) \
		.def_readwrite("rsf", &contact::rsf) \
		.def_readwrite("sd_props", &contact::sd_props)
		.def_readwrite("normal_condition", &contact::normal_condition)
		.def(py::pickle(
			[](const contact& c) { // __getstate__
				py::tuple t(18);

				t[0] = c.cell_ids;
				t[1] = c.f_scale;
				t[2] = c.num_of_change_sign;
				t[3] = c.N_VARS;
				t[4] = c.U_VAR;
				t[5] = c.P_VAR;
				t[6] = c.NT;
				t[7] = c.U_VAR_T;
				t[8] = c.P_VAR_T;
				//t[9] = c.friction_model;
				//t[10] = c.friction_criterion;
				t[9] = c.mu0;
				t[10] = c.mu;
				t[11] = c.fault_stress;
				t[12] = c.phi;
				t[13] = c.S;
				//t[16] = c.states;
				t[14] = c.fault_tag;
				t[15] = c.eps_n;
				t[16] = c.eps_t;
				t[17] = c.eta;

				return t;
			},
			[](py::tuple t) { // __setstate__
				contact c;

				c.cell_ids = t[0].cast<std::vector<index_t>>();
				c.f_scale = t[1].cast<value_t>();
				c.num_of_change_sign = t[2].cast<index_t>();
				c.N_VARS = t[3].cast<uint8_t>();
				c.U_VAR = t[4].cast<uint8_t>();
				c.P_VAR = t[5].cast<uint8_t>();
				c.NT = t[6].cast<uint8_t>();
				c.U_VAR_T = t[7].cast<uint8_t>();
				c.P_VAR_T = t[8].cast<uint8_t>();
				//c.friction_model = t[9].cast<FrictionModel>();
				//c.friction_criterion = t[10].cast<CriticalStress>();
				c.mu0 = t[9].cast<std::vector<value_t>>();
				c.mu = t[10].cast<std::vector<value_t>>();
				c.fault_stress = t[11].cast<std::vector<value_t>>();
				c.phi = t[12].cast<std::vector<value_t>>();
				c.S = t[13].cast<std::vector<Matrix>>();
				//c.states = t[16].cast<std::vector<ContactState>>();
				c.fault_tag = t[14].cast<index_t>();
				c.eps_n = t[15].cast<std::vector<value_t>>();
				c.eps_t = t[16].cast<std::vector<value_t>>();
				c.eta = t[17].cast<std::vector<value_t>>();

				return c;
			}));

		py::class_<RSF_props>(m, "rsf_props") \
			.def(py::init<>()) \
			.def_readwrite("theta", &RSF_props::theta) \
			.def_readwrite("theta_n", &RSF_props::theta_n) \
			.def_readwrite("mu_rate", &RSF_props::mu_rate) \
			.def_readwrite("mu_state", &RSF_props::mu_state) \
			.def_readwrite("ref_velocity", &RSF_props::vel0) \
			.def_readwrite("crit_distance", &RSF_props::Dc) \
			.def_readwrite("a", &RSF_props::a) \
			.def_readwrite("b", &RSF_props::b) \
			.def_readwrite("min_vel", &RSF_props::min_vel) \
			.def_readwrite("law", &RSF_props::law)
			.def(py::pickle(
				[](const RSF_props& p) { // __getstate__
					py::tuple t(9);

					t[0] = p.theta;
					t[1] = p.theta_n;
					t[2] = p.vel0;
					t[3] = p.Dc;
					t[4] = p.a;
					t[5] = p.b;
					t[6] = p.min_vel;
					t[7] = p.mu_rate;
					t[8] = p.mu_state;

					return t;
				},
				[](py::tuple t) { // __setstate__
					RSF_props p;

					p.theta = t[0].cast<std::vector<value_t>>();
					p.theta_n = t[1].cast<std::vector<value_t>>();
					p.vel0 = t[2].cast<value_t>();
					p.Dc = t[3].cast<value_t>();
					p.a = t[4].cast<value_t>();
					p.b = t[5].cast<value_t>();
					p.min_vel = t[6].cast<value_t>();
					p.mu_rate = t[7].cast<std::vector<value_t>>();
					p.mu_state = t[8].cast<std::vector<value_t>>();

					return p;
				}));

	py::class_<SlipDependentFriction_props>(m, "sd_props") \
		.def(py::init<>()) \
		.def_readwrite("crit_distance", &SlipDependentFriction_props::Dc) \
		.def_readwrite("mu_dyn", &SlipDependentFriction_props::mu_d)
		.def(py::pickle(
			[](const SlipDependentFriction_props& p) { // __getstate__
				py::tuple t(2);

				t[0] = p.Dc;
				t[1] = p.mu_d;

				return t;
			},
			[](py::tuple t) { // __setstate__
				SlipDependentFriction_props p;

				p.Dc = t[0].cast<value_t>();
				p.mu_d = t[1].cast<value_t>();

				return p;
			}));

	py::enum_<ContactState>(m, "contact_state") \
		.value("TRUE_STUCK", ContactState::TRUE_STUCK) \
		.value("PEN_STUCK", ContactState::PEN_STUCK) \
		.value("SLIP", ContactState::SLIP) \
		.value("FREE", ContactState::FREE)
		.export_values();
	py::bind_vector<std::vector<ContactState>>(m, "state_vector");

	py::enum_<FrictionModel>(m, "friction") \
		.value("FRICTIONLESS", FrictionModel::FRICTIONLESS) \
		.value("STATIC", FrictionModel::STATIC) \
		.value("SLIP_DEPENDENT", FrictionModel::SLIP_DEPENDENT) \
		.value("RSF", FrictionModel::RSF) \
		.value("RSF_STAB", FrictionModel::RSF_STAB) \
		.value("CNS", FrictionModel::CNS)
		.export_values();

	py::enum_<CriticalStress>(m, "critical_stress") \
		.value("TERZAGHI", CriticalStress::TERZAGHI) \
		.value("BIOT", CriticalStress::BIOT)
		.export_values();

	py::enum_<StateLaw>(m, "state_law") \
		.value("AGEING_LAW", StateLaw::AGEING_LAW) \
		.value("SLIP_LAW", StateLaw::SLIP_LAW) \
		.value("MIXED", StateLaw::MIXED)
		.export_values();

	py::enum_<ContactSolver>(m, "contact_solver")
		.value("FLUX_FROM_PREVIOUS_ITERATION", ContactSolver::FLUX_FROM_PREVIOUS_ITERATION) \
		.value("RETURN_MAPPING", ContactSolver::RETURN_MAPPING) \
		.value("LOCAL_ITERATIONS", ContactSolver::LOCAL_ITERATIONS) \
		.export_values();

	py::enum_<NormalCondition>(m, "normal_condition")
		.value("PENALIZED", NormalCondition::PENALIZED) \
		.value("ZERO_GAP_CHANGE", NormalCondition::ZERO_GAP_CHANGE)
		.export_values();
};

#endif /* PYBIND11_ENABLED */
