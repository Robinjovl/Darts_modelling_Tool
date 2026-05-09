#ifndef PY_EVALUATOR_IFACE_H
#define PY_EVALUATOR_IFACE_H

#include "py_globals_interpolation.h"
#include "evaluator_iface.h"

class py_property_evaluator_iface : public property_evaluator_iface {
public:

  /* Inherit the constructors */
  using property_evaluator_iface::property_evaluator_iface;

  /* Trampoline (need one for each virtual function) */
  value_t evaluate(
    const std::vector<value_t> &state  // INPUT: state variables for every block
  )
  {
    py::gil_scoped_acquire acquire;

    PYBIND11_OVERLOAD_PURE(
      value_t,
      property_evaluator_iface,
      evaluate,
      state
    );
    //py::gil_scoped_release release;
  }
  /* Trampoline (need one for each virtual function) */
  int evaluate(
    const std::vector<value_t> &states,     // INPUT: state variables for every block
    index_t n_blocks,                 // INPUT: number of blocks
    std::vector<value_t> &values      // OUTPUT: evaluated property values for every block
  )
  {
    py::gil_scoped_acquire acquire;
    PYBIND11_OVERLOAD_PURE(
      int,
      property_evaluator_iface,
      evaluate,
      states,
      n_blocks,
      values
    );
  }
};


class py_operator_set_evaluator_iface : public operator_set_evaluator_iface {
public:

  /* Inherit the constructors */
  using operator_set_evaluator_iface::operator_set_evaluator_iface;

  /* Trampoline (need one for each virtual function) */
  int evaluate(
    const std::vector<value_t> &state,
    std::vector<value_t> &values
  )
  {
    py::gil_scoped_acquire acquire;

    PYBIND11_OVERLOAD_PURE(
      int,
      operator_set_evaluator_iface,
      evaluate,
      state,
      &values
    );
  }
};

#endif
