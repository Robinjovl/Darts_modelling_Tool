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
#ifndef OPENDARTS_LINEAR_SOLVERS_SOLVER_REGISTRY_HPP
#define OPENDARTS_LINEAR_SOLVERS_SOLVER_REGISTRY_HPP
//--------------------------------------------------------------------------

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace opendarts
{
  namespace linear_solvers
  {
    class linear_solver;   // see linear_solver.hpp
    struct solver_config;  // see solver_config.hpp

    /** Factory that builds a solver for a configuration and a matrix block size. */
    using solver_factory = std::function<std::unique_ptr<opendarts::linear_solvers::linear_solver>(
        const opendarts::linear_solvers::solver_config &, int /*block_size*/)>;

    /** Register a solver factory under a unique name.
     *
     *  A solver self-registers from its own translation unit (typically via a
     *  static registrar object), so adding a new solver needs no change to any
     *  enum or dispatch switch -- this replaces sim_params::linear_solver_t.
     *
     *  @return true if registered, false if the name was already taken
     */
    bool register_solver(const std::string &name,
        opendarts::linear_solvers::solver_factory factory);

    /** Build a solver by registered name.
     *
     *  @throws std::runtime_error if the name is not registered in this build --
     *          e.g. a proprietary bos solver when BOS_SOLVERS_DIR was not set.
     */
    std::unique_ptr<opendarts::linear_solvers::linear_solver> create_linear_solver(
        const std::string &name,
        const opendarts::linear_solvers::solver_config &config,
        int block_size);

    /** Whether a solver name is available in this build. */
    bool is_solver_registered(const std::string &name);

    /** Names of all registered solvers, sorted -- for diagnostics and the
     *  Python-side listing of available solvers. */
    std::vector<std::string> registered_solvers();
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_SOLVER_REGISTRY_HPP
//--------------------------------------------------------------------------
