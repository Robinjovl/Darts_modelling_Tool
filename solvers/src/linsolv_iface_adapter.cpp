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

#include "linsolv_iface_adapter.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    linsolv_iface_adapter::linsolv_iface_adapter(
        opendarts::linear_solvers::linsolv_iface *wrapped)
        : wrapped_solver(wrapped), last_stats()
    {
    }

    linsolv_iface_adapter::~linsolv_iface_adapter()
    {
      delete this->wrapped_solver;
    }

    int linsolv_iface_adapter::init(opendarts::linear_solvers::csr_matrix_base *A,
        const opendarts::linear_solvers::solver_config &config)
    {
      this->last_stats = opendarts::linear_solvers::solver_stats();
      return this->wrapped_solver->init(A, config.max_iterations, config.tolerance);
    }

    int linsolv_iface_adapter::setup(opendarts::linear_solvers::csr_matrix_base *A)
    {
      return this->wrapped_solver->setup(A);
    }

    int linsolv_iface_adapter::solve(opendarts::config::mat_float *rhs,
        opendarts::config::mat_float *x)
    {
      const int solve_status = this->wrapped_solver->solve(rhs, x);
      this->last_stats.iterations = this->wrapped_solver->get_n_iters();
      this->last_stats.residual = this->wrapped_solver->get_residual();
      this->last_stats.converged = (solve_status == 0);
      return solve_status;
    }

    void linsolv_iface_adapter::set_prec(opendarts::linear_solvers::linear_solver *prec)
    {
      // The wrapped linsolv_iface expects a linsolv_iface* preconditioner, so
      // only another adapter (from which the wrapped solver can be extracted)
      // can be bridged. A non-adapter preconditioner is silently ignored.
      auto *prec_as_adapter =
          dynamic_cast<opendarts::linear_solvers::linsolv_iface_adapter *>(prec);
      if (prec_as_adapter == nullptr)
        return;
      this->wrapped_solver->set_prec(prec_as_adapter->wrapped_solver);
    }

    opendarts::linear_solvers::solver_stats linsolv_iface_adapter::stats() const
    {
      return this->last_stats;
    }
  } // namespace linear_solvers
} // namespace opendarts
