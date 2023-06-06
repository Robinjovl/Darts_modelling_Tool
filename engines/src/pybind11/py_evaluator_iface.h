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
};

#endif