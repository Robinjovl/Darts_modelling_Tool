#ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include "py_globals.h"
#include <pybind11/stl.h>

namespace py = pybind11;
#include "mech/engine_super_elastic_cpu.hpp"
#include "conn_mesh.h"

PYBIND11_MAKE_OPAQUE(std::vector<pm::contact>);

template <uint8_t NC, uint8_t NP, bool THERMAL>
struct engine_super_elastic_exposer
{
	static void expose(py::module &m)
	{
		std::string short_name, long_name;
		short_name = "engine_super_elastic_cpu" + std::to_string(NC) + "_" + std::to_string(NP);
		if (THERMAL)
		{
			long_name = "Non-isothermal ";
			short_name += "_t";
		}
		else
		{
		    long_name = "Isothermal ";
		}
		long_name += "CPU simulator engine for " + std::to_string(NC) + " components and " + std::to_string(NP) + " phases fluid in poroelastic matrix, diffusion and kinetic reaction";
		py::class_<engine_super_elastic_cpu<NC, NP, THERMAL>, engine_base>(m, short_name.c_str(), long_name.c_str())   \
			.def(py::init<>()) \
			.def("init", (int (engine_super_elastic_cpu<NC, NP, THERMAL>::*)
			(conn_mesh *, std::vector<ms_well*> &, std::vector<operator_set_gradient_evaluator_iface*> &, sim_params*, timer_node*)) &engine_super_elastic_cpu<NC, NP, THERMAL>::init, "Initialize simulator by mesh, tables and wells", py::keep_alive<1, 5>())
			.def("calc_newton_dev", &engine_super_elastic_cpu<NC, NP, THERMAL>::calc_newton_dev) \
			.def("apply_newton_update", &engine_super_elastic_cpu<NC, NP, THERMAL>::apply_newton_update) \
			.def("post_newtonloop", &engine_super_elastic_cpu<NC, NP, THERMAL>::post_newtonloop) \
			.def("set_discretizer", (void (engine_super_elastic_cpu<NC, NP, THERMAL>::*) 
			(typename engine_super_elastic_cpu<NC, NP, THERMAL>::DiscretizerType*)) &engine_super_elastic_cpu<NC, NP, THERMAL>::set_discretizer) \
			.def("eval_stresses_and_velocities", &engine_super_elastic_cpu<NC, NP, THERMAL>::eval_stresses_and_velocities) \
			.def_readwrite("find_equilibrium", &engine_super_elastic_cpu<NC, NP, THERMAL>::FIND_EQUILIBRIUM) \
			.def_readwrite("geomechanics_mode", &engine_super_elastic_cpu<NC, NP, THERMAL>::geomechanics_mode) \
			.def_readwrite("dev_u", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_u) \
			.def_readwrite("dev_p", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_p) \
			.def_readwrite("dev_e", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_e) \
			.def_readwrite("dev_g", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_g) \
			//.def_readwrite("dev_z", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_z) 
			.def_readwrite("dev_u_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_u_prev) \
			.def_readwrite("dev_p_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_p_prev) \
			.def_readwrite("dev_e_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_e_prev) \
			.def_readwrite("dev_g_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_g_prev) \
			//.def_readwrite("dev_z_prev", &engine_super_elastic_cpu<NC, NP, THERMAL>::dev_z_prev) 
			.def_readwrite("well_residual_prev_dt", &engine_super_elastic_cpu<NC, NP, THERMAL>::well_residual_prev_dt) \
			.def_readwrite("darcy_fluxes", &engine_super_elastic_cpu<NC, NP, THERMAL>::darcy_fluxes) \
			.def_readwrite("structural_movement_fluxes", &engine_super_elastic_cpu<NC, NP, THERMAL>::structural_movement_fluxes) \
			.def_readwrite("fourier_fluxes", &engine_super_elastic_cpu<NC, NP, THERMAL>::fourier_fluxes) \
			.def_readwrite("fick_fluxes", &engine_super_elastic_cpu<NC, NP, THERMAL>::fick_fluxes) \
			.def_readwrite("hooke_forces", &engine_super_elastic_cpu<NC, NP, THERMAL>::hooke_forces) \
			.def_readwrite("biot_forces", &engine_super_elastic_cpu<NC, NP, THERMAL>::biot_forces) \
			.def_readwrite("thermal_forces", &engine_super_elastic_cpu<NC, NP, THERMAL>::thermal_forces) \
			.def_readwrite("hooke_forces_n", &engine_super_elastic_cpu<NC, NP, THERMAL>::hooke_forces_n) \
			.def_readwrite("biot_forces_n", &engine_super_elastic_cpu<NC, NP, THERMAL>::biot_forces_n) \
			.def_readwrite("thermal_forces_n", &engine_super_elastic_cpu<NC, NP, THERMAL>::thermal_forces_n) \
			.def_readwrite("total_stresses", &engine_super_elastic_cpu<NC, NP, THERMAL>::total_stresses) \
			.def_readwrite("effective_stresses", &engine_super_elastic_cpu<NC, NP, THERMAL>::effective_stresses) \
			.def_readwrite("darcy_velocities", &engine_super_elastic_cpu<NC, NP, THERMAL>::darcy_velocities) \
			.def_readwrite("dX", &engine_super_elastic_cpu<NC, NP, THERMAL>::dX) \
			.def_readwrite("RHS", &engine_super_elastic_cpu<NC, NP, THERMAL>::RHS) \
			.def_readwrite("contacts", &engine_super_elastic_cpu<NC, NP, THERMAL>::contacts) \
			.def_readwrite("contact_solver", &engine_super_elastic_cpu<NC, NP, THERMAL>::contact_solver) \
			.def_readwrite("eps_vol", &engine_super_elastic_cpu<NC, NP, THERMAL>::eps_vol) \
			.def_readwrite("gravity", &engine_super_elastic_cpu<NC, NP, THERMAL>::gravity) \
			.def_readwrite("Xref", &engine_super_elastic_cpu<NC, NP, THERMAL>::Xref) \
			.def_readwrite("Xn_ref", &engine_super_elastic_cpu<NC, NP, THERMAL>::Xn_ref) \
			.def_property_readonly_static("P_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::P_VAR; }) \
			.def_property_readonly_static("Z_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::Z_VAR; }) \
			.def_property_readonly_static("P_VAR_T", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::P_VAR_T; }) \
			.def_property_readonly_static("U_VAR_T", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::U_VAR_T; }) \
			.def_property_readonly_static("U_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::U_VAR; }) \
			.def_property_readonly_static("T_VAR", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::T_VAR; }) \
			.def_property_readonly_static("N_VARS", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::N_VARS; }) \
			.def_property_readonly_static("NT", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::NT; }) \
			.def_property_readonly_static("N_OPS", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::N_OPS; }) \
			.def_property_readonly_static("NC", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::NC_; }) \
			.def_property_readonly_static("ACC_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::ACC_OP; }) \
			.def_property_readonly_static("FLUX_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::FLUX_OP; }) \
			.def_property_readonly_static("GRAV_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::GRAV_OP; }) \
			.def_property_readonly_static("SAT_OP", [](py::object) {return engine_super_elastic_cpu<NC, NP, THERMAL>::SAT_OP; });
	}
};

void pybind_engine_super_elastic_cpu(py::module &m)
{
  // single-phase isothermal
  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 1, false> re;
  re.expose(m);
  
  // two-phase isothermal
  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 2, false> re1;
  re1.expose(m);

  // three-phase isothermal
  //recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 2, MAX_NC, 3, false> re2;
  //re2.expose(m);

  // single-phase thermal
  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 1, true> re3;
  re3.expose(m);

  // two-phase thermal
  recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 1, MAX_NC, 2, true> re4;
  re4.expose(m);

  // three-phase thermal
  //recursive_exposer_nc_np_t<engine_super_elastic_exposer, py::module, 2, MAX_NC, 3, true> re5;
  //re5.expose(m);
}

#endif //PYBIND11_ENABLED