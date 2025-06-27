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
  // int add_to_jacobian(value_t dt, index_t well_head_idx, value_t segment_trans,
  //                     index_t n_block_size, uint8_t N_VARS, uint8_t P_VAR, std::vector<value_t> &X, value_t *jacobian_row, std::vector<value_t> &RHS)
  // {
  //   PYBIND11_OVERLOAD_PURE(
  //     int,                      /* Return type */
  //     well_control_iface,       /* Parent class */
  //     add_to_jacobian,          /* Name of function in C++ (must match Python name) */
  //     dt,                       /* Argument(s) */
  //     well_head_idx,
	//   segment_trans,
  //     n_block_size,
	//   N_VARS,
	//   P_VAR,
  //     X,
  //     jacobian_row,
  //     &RHS
  //   );
  // }

  // int check_constraint_violation(value_t dt, index_t well_head_idx, value_t segment_trans, index_t n_block_size, uint8_t N_VARS, uint8_t P_VAR, std::vector<value_t>& X)
  // {
  //   PYBIND11_OVERLOAD_PURE(
  //     int,                          /* Return type */
  //     well_control_iface,           /* Parent class */
  //     check_constraint_violation,   /* Name of function in C++ (must match Python name) */
  //     dt,                           /* Argument(s) */
  //     well_head_idx,
  //     n_block_size,
	//   N_VARS, 
	//   P_VAR,
  //     X
  //   );
  // }

  // int initialize_well_block(std::vector<value_t>& state_block, const std::vector<value_t>& state_neighbour)
  // {
  //   PYBIND11_OVERLOAD_PURE(
  //     int,                          /* Return type */
  //     well_control_iface,           /* Parent class */
  //     initialize_control,           /* Name of function in C++ (must match Python name) */
  //     state_block,                  /* Argument(s) */
  //     state_neighbour
  //   );
  // }

  // int set_bhp_control(std::vector<value_t>& well_control_spec_)
  // {
  //   PYBIND11_OVERLOAD_PURE(
  //     int,                          /* Return type */
  //     well_control_iface,           /* Parent class */
  //     set_bhp_control,              /* Name of function in C++ (must match Python name) */
  //     well_control_spec_,           /* Argument(s) */
  //   );
  // }

  // int set_rate_control(well_control_iface::WellControlType control_type_, index_t phase_idx_, std::vector<value_t>& well_control_spec_)
  // {
  //   PYBIND11_OVERLOAD_PURE(
  //     int,                     /* Return type */
  //     well_control_iface,       /* Parent class */
  //     set_rate_control,         /* Name of function in C++ (must match Python name) */
  //     control_type_,            /* Argument(s) */
  //     phase_idx_,
  //     well_control_spec_,      
  //   );
  // }

};

#endif
void pybind_well_controls(py::module &m)
{
  py::class_<well_control_iface, py_well_control_iface /* <--- trampoline*/> well_control_iface(m, "well_control_iface");
  well_control_iface
    .def(py::init<index_t, index_t, bool, operator_set_gradient_evaluator_iface*, operator_set_gradient_evaluator_iface*>())
    .def("add_to_jacobian", &well_control_iface::add_to_jacobian)
    .def("check_constraint_violation", &well_control_iface::check_constraint_violation)
    .def("set_bhp_control", &well_control_iface::set_bhp_control)
    .def("set_rate_control", &well_control_iface::set_rate_control)
    .def("get_well_control_type_str", &well_control_iface::get_well_control_type_str)
    .def("get_well_control_type", &well_control_iface::get_well_control_type);

  py::enum_<well_control_iface::WellControlType>(well_control_iface, "WellControlType")
    .value("NONE", well_control_iface::WellControlType::NONE)
    .value("BHP", well_control_iface::WellControlType::BHP)
    .value("MOLAR_RATE",	well_control_iface::WellControlType::MOLAR_RATE)
    .value("MASS_RATE", well_control_iface::WellControlType::MASS_RATE)
    .value("VOLUMETRIC_RATE", well_control_iface::WellControlType::VOLUMETRIC_RATE)
    .value("ADVECTIVE_HEAT_RATE", well_control_iface::WellControlType::ADVECTIVE_HEAT_RATE)
	.export_values();
}

#endif //PYBIND11_ENABLED