//*************************************************************************
//    Copyright (c) 2026
//    Delft University of Technology, the Netherlands
//    Netherlands eScience Center
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the Apache License.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_SOLVER_CONFIG_HPP
#define OPENDARTS_LINEAR_SOLVERS_SOLVER_CONFIG_HPP
//--------------------------------------------------------------------------

#include "data_types.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Base configuration shared by every linear solver.
     *
     *  Each concrete solver defines its own struct deriving from solver_config
     *  and adds the solver-specific parameters. The factory registered in the
     *  solver registry knows the concrete type and down-casts accordingly.
     *  It is the C++ counterpart of the Python LinearSolverSpec classes.
     */
    struct solver_config
    {
      opendarts::config::index_t max_iterations = 50;       // max linear iterations
      opendarts::config::mat_float tolerance = 1e-5;        // relative convergence tolerance
      int print_level = 0;                                  // 0 = silent, higher = more verbose

      virtual ~solver_config() = default;  // polymorphic: enables safe down-cast in factories
    };

    /** Outcome of the last solve, reported uniformly by every linear solver. */
    struct solver_stats
    {
      int iterations = 0;                            // iterations performed in last solve
      opendarts::config::mat_float residual = 0.0;   // final (relative) residual
      bool converged = false;                        // whether the tolerance was reached
      double setup_time = 0.0;                       // wall time of last setup() [s]
      double solve_time = 0.0;                       // wall time of last solve() [s]
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_SOLVER_CONFIG_HPP
//--------------------------------------------------------------------------
