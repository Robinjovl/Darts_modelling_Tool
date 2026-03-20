#ifndef PY_EVALUATOR_IFACE_H
#define PY_EVALUATOR_IFACE_H

#include "py_globals.h"
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

  /* Trampoline for batch evaluation — non-pure, has C++ default.
     Manual implementation instead of PYBIND11_OVERLOAD to correctly pass
     values by pointer (so Python writes back to C++ vector) while still
     falling back to C++ default when no Python override exists. */
  int evaluate_batch(
    const std::vector<value_t> &states,   // INPUT: flat [n_points * n_dims]
    int n_points,                         // INPUT: number of points
    std::vector<value_t> &values,         // OUTPUT: flat [n_points * n_ops]
    int n_ops                             // INPUT: operators per point
  )
  {
    py::gil_scoped_acquire acquire;

    // Check if Python subclass overrides evaluate_batch
    pybind11::function overload = pybind11::get_overload(
        static_cast<const operator_set_evaluator_iface *>(this), "evaluate_batch");
    if (overload) {
      auto o = overload(states, n_points, &values, n_ops);
      return o.cast<int>();
    }
    // No Python override — call C++ default
    return operator_set_evaluator_iface::evaluate_batch(states, n_points, values, n_ops);
  }
};

#endif
