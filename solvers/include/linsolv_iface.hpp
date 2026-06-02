//*************************************************************************
//    Copyright (c) 2022
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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------

// Backward-compatibility shim: ``linsolv_iface`` is the original name of
// open-DARTS' unified linear-solver interface. The canonical type now lives
// in ``linear_solver.hpp`` -- see that header for the documented API. This
// alias keeps existing call sites (engine, pybind, every concrete solver)
// compiling unchanged while the migration completes; new code should refer
// to ``opendarts::linear_solvers::linear_solver`` directly.

#include "linear_solver.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    using linsolv_iface = opendarts::linear_solvers::linear_solver;
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------
