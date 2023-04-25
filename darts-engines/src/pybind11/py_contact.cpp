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

#ifdef PYBIND11_ENABLED
#include <pybind11.h>
#include "py_globals.h"
#include <stl_bind.h>
#include <stl.h>

#include "mech/contact.h"
//#include "mech/pm_discretizer.hpp"

namespace py = pybind11;
using namespace pm;

PYBIND11_MAKE_OPAQUE(std::vector<Matrix>);

void pybind_contact(py::module &m)
{
	py::class_<contact>(m, "contact") \
		.def(py::init<>()) \
		.def("init_geometry", (int(contact::*)(index_t, pm_discretizer*, conn_mesh*, std::vector<index_t>&))& contact::init_geometry) \
		.def("init_fault", (int(contact::*)())& contact::init_fault) \
		.def("init_local_iterations", (int(contact::*)())& contact::init_local_iterations) \
		.def("init_friction", (int(contact::*)())& contact::init_friction) \
		.def("set_state", &contact::set_state) \
		.def_readwrite("cell_ids", &contact::cell_ids) \
		.def_readwrite("f_scale", &contact::f_scale) \
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
		.def_readwrite("normal_condition", &contact::normal_condition);

	py::class_<RSF_props>(m, "rsf_props") \
		.def(py::init<>()) \
		.def_readwrite("theta", &RSF_props::theta) \
		.def_readwrite("theta_n", &RSF_props::theta_n) \
		.def_readwrite("ref_velocity", &RSF_props::vel0) \
		.def_readwrite("crit_distance", &RSF_props::Dc) \
		.def_readwrite("a", &RSF_props::a) \
		.def_readwrite("b", &RSF_props::b) \
		.def_readwrite("min_vel", &RSF_props::min_vel) \
		.def_readwrite("law", &RSF_props::law);

	py::class_<SlipDependentFriction_props>(m, "sd_props") \
		.def(py::init<>()) \
		.def_readwrite("crit_distance", &SlipDependentFriction_props::Dc) \
		.def_readwrite("mu_dyn", &SlipDependentFriction_props::mu_d);

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
