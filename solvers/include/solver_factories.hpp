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
#ifndef OPENDARTS_LINEAR_SOLVERS_SOLVER_FACTORIES_HPP
#define OPENDARTS_LINEAR_SOLVERS_SOLVER_FACTORIES_HPP
//--------------------------------------------------------------------------

namespace opendarts
{
  namespace linear_solvers
  {
    /** Register every open-source solver compiled into this build with the
     *  solver registry.
     *
     *  Currently registers: "mgr" (HYPRE MGR) and "superlu" (direct).
     *
     *  Idempotent -- repeated calls are harmless -- but it must be called once
     *  before create_linear_solver() is used. It is invoked from the pybind
     *  solvers module initialization and from engine initialization.
     */
    void register_builtin_solvers();
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_SOLVER_FACTORIES_HPP
//--------------------------------------------------------------------------
