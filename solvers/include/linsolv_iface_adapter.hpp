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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_ADAPTER_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_ADAPTER_HPP
//--------------------------------------------------------------------------

#include "data_types.hpp"
#include "linear_solver.hpp"
#include "linsolv_iface.hpp"
#include "solver_config.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Adapts a legacy linsolv_iface solver to the unified linear_solver interface.
     *
     *  Lets any existing linsolv_iface-based solver -- including the external
     *  proprietary bos solvers -- be reached through the unified interface and
     *  the solver registry without being ported. The adapter forwards the calls
     *  and maps the generic configuration (tolerance, max iterations); any
     *  solver-specific configuration must be applied to the wrapped solver
     *  before it is handed to the adapter (the registered factory does this).
     *
     *  Ownership: the adapter owns the wrapped solver and deletes it. A
     *  preconditioner passed via set_prec() is NOT owned by the adapter -- its
     *  lifetime is the caller's responsibility. Precise preconditioner-chain
     *  ownership is finalized with the engine factory wiring (plan commit C3).
     */
    class linsolv_iface_adapter : public opendarts::linear_solvers::linear_solver
    {
    public:
      /** Wrap an existing linsolv_iface solver. Takes ownership of `wrapped`. */
      explicit linsolv_iface_adapter(opendarts::linear_solvers::linsolv_iface *wrapped);

      ~linsolv_iface_adapter() override;

      linsolv_iface_adapter(const linsolv_iface_adapter &) = delete;
      linsolv_iface_adapter &operator=(const linsolv_iface_adapter &) = delete;

      int init(opendarts::linear_solvers::csr_matrix_base *A,
          const opendarts::linear_solvers::solver_config &config) override;

      int setup(opendarts::linear_solvers::csr_matrix_base *A) override;

      int solve(opendarts::config::mat_float *rhs,
          opendarts::config::mat_float *x) override;

      /** Attach a preconditioner. Only another linsolv_iface_adapter can be
       *  bridged to the wrapped legacy solver (its wrapped solver is extracted
       *  and forwarded); any other type is ignored. Ownership is not taken.
       */
      void set_prec(opendarts::linear_solvers::linear_solver *prec) override;

      opendarts::linear_solvers::solver_stats stats() const override;

      /** Direct access to the wrapped legacy solver -- e.g. to apply
       *  solver-specific configuration not carried by the unified interface. */
      opendarts::linear_solvers::linsolv_iface *get_wrapped() const
      {
        return this->wrapped_solver;
      }

    private:
      opendarts::linear_solvers::linsolv_iface *wrapped_solver;
      opendarts::linear_solvers::solver_stats last_stats;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_ADAPTER_HPP
//--------------------------------------------------------------------------
