#ifdef PYBIND11_ENABLED
#include "py_globals.h"
#include <pybind11/stl.h>
#include "well_controls.h"


namespace py = pybind11;

#if 1
class py_well_control_iface : public well_control_iface {
public:
 
  /* Inherit the constructors */
  using well_control_iface::well_control_iface;

  /* Trampoline (need one for each virtual function) */
  int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
                      index_t n_block_size, uint8_t N_VARS, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
  {
    PYBIND11_OVERLOAD_PURE(
      int,                      /* Return type */
      well_control_iface,       /* Parent class */
      add_to_jacobian,          /* Name of function in C++ (must match Python name) */
      dt,                       /* Argument(s) */
      well_head_idx,
	  segment_trans,
      n_block_size,
	  N_VARS,
	  P_VAR,
      X,
      jacobian_row,
      &RHS
    );
  }

  int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_block_size, uint8_t N_VARS, uint8_t P_VAR, std::vector<value_t>& X)
  {
    PYBIND11_OVERLOAD_PURE(
      int,                          /* Return type */
      well_control_iface,           /* Parent class */
      check_constraint_violation,   /* Name of function in C++ (must match Python name) */
      dt,                           /* Argument(s) */
      well_head_idx,
      n_block_size,
	  N_VARS, 
	  P_VAR,
      X
    );
  }

  int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
  {
    PYBIND11_OVERLOAD_PURE(
      int,                          /* Return type */
      well_control_iface,           /* Parent class */
      initialize_control,           /* Name of function in C++ (must match Python name) */
      state_block,                  /* Argument(s) */
      state_neighbour
    );
  }
};

#endif
void pybind_well_controls(py::module &m)
{
#if 1

  py::class_<well_control_iface, py_well_control_iface /* <--- trampoline*/> well_control_iface(m, "well_control_iface");
  well_control_iface
    .def(py::init<>())
    .def("add_to_jacobian", &well_control_iface::add_to_jacobian)
    .def("check_constraint_violation", &well_control_iface::check_constraint_violation);
#endif

  //  py::class_<bhp_inj_well_control>(m, "bhp_inj_well_control", well_control_iface)
  py::class_<bhp_inj_well_control>(m, "bhp_inj_well_control", well_control_iface)
    .def(py::init<value_t, std::vector<value_t> &>())
    .def_readwrite("injection_stream", &bhp_inj_well_control::injection_stream)
    .def_readwrite("target_pressure", &bhp_inj_well_control::target_pressure);

  py::class_<bhp_prod_well_control>(m, "bhp_prod_well_control", well_control_iface)
    .def(py::init<value_t>())
    .def_readwrite("target_pressure", &bhp_prod_well_control::target_pressure);

  py::class_<rate_inj_well_control>(m, "rate_inj_well_control", well_control_iface)
    .def(py::init<std::vector <std::string>, index_t, index_t, index_t,
         value_t, std::vector <value_t> &,
         operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 8>())
    .def_readwrite("injection_stream", &rate_inj_well_control::injection_stream)
    .def_readwrite("target_rate", &rate_inj_well_control::target_rate);

  py::class_<rate_inj_well_control_mass_balance>(m, "rate_inj_well_control_mass_balance", well_control_iface)
    .def(py::init<std::vector <std::string>, index_t, index_t, index_t,
         value_t, std::vector <value_t> &,
         operator_set_evaluator_iface*, operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 8>(), py::keep_alive<1, 9>())
    .def_readwrite("injection_stream", &rate_inj_well_control_mass_balance::injection_stream)
    .def_readwrite("target_rate", &rate_inj_well_control_mass_balance::target_rate);

  py::class_<rate_prod_well_control>(m, "rate_prod_well_control", well_control_iface)
    .def(py::init<std::vector <std::string>, index_t, index_t, index_t,
         value_t, 
         operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 7>())
    .def_readwrite("target_rate", &rate_prod_well_control::target_rate);


  py::class_<rate_prod_well_control_mass_balance>(m, "rate_prod_well_control_mass_balance", well_control_iface)
    .def(py::init<std::vector <std::string>, index_t, index_t, index_t,
         value_t, 
         operator_set_evaluator_iface*, operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 7>(), py::keep_alive<1, 8>())
    .def_readwrite("target_rate", &rate_prod_well_control_mass_balance::target_rate);

  py::class_<gt_bhp_temp_inj_well_control>(m, "gt_bhp_temp_inj_well_control", well_control_iface)
	.def(py::init<std::vector<std::string>, index_t, value_t, value_t, std::vector<value_t>, operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>(), py::keep_alive<1, 5>())
    .def_readwrite("target_temperature", &gt_bhp_temp_inj_well_control::target_temperature)
	.def_readwrite("target_pressure", &gt_bhp_temp_inj_well_control::target_pressure);

  py::class_<gt_bhp_prod_well_control>(m, "gt_bhp_prod_well_control", well_control_iface)
	.def(py::init<value_t>(), py::keep_alive<1, 2>())
	.def_readwrite("target_pressure", &gt_bhp_prod_well_control::target_pressure);

  py::class_<gt_rate_temp_inj_well_control>(m, "gt_rate_temp_inj_well_control", well_control_iface)
	  .def(py::init<std::vector <std::string>, index_t, index_t,
		  value_t, value_t, std::vector <value_t>&,
		  operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 8>())
	  .def_readwrite("target_rate", &gt_rate_temp_inj_well_control::target_rate)
	  .def_readwrite("target_temperature", &gt_rate_temp_inj_well_control::target_temperature);

  py::class_<gt_rate_prod_well_control>(m, "gt_rate_prod_well_control", well_control_iface)
	  .def(py::init<std::vector<std::string>, index_t, index_t, value_t,
		  operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 6>())
	  .def_readwrite("target_rate", &gt_rate_prod_well_control::target_rate);

  py::class_<gt_mass_rate_enthalpy_inj_well_control>(m, "gt_mass_rate_enthalpy_inj_well_control", well_control_iface)
	  .def(py::init<std::vector <std::string>, index_t, index_t, std::vector <value_t>,
		  value_t, value_t, 
		  operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 8>())
	  .def_readwrite("target_rate", &gt_mass_rate_enthalpy_inj_well_control::target_rate)
	  .def_readwrite("target_enthalpy", &gt_mass_rate_enthalpy_inj_well_control::target_enthalpy);

  py::class_<gt_mass_rate_prod_well_control>(m, "gt_mass_rate_prod_well_control", well_control_iface)
	  .def(py::init<std::vector<std::string>, index_t, index_t, value_t,
		  operator_set_gradient_evaluator_iface*>(), py::keep_alive<1, 6>())
	  .def_readwrite("target_rate", &gt_mass_rate_prod_well_control::target_rate);

}

#endif //PYBIND11_ENABLED