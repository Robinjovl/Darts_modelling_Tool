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
#ifndef OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_HPP
//--------------------------------------------------------------------------

#include "data_types.hpp"
#include "solver_config.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    class csr_matrix_base;  // forward declaration, see csr_matrix_base.hpp

    /** Unified linear-solver interface.
     *
     *  The single base class for every linear solver in open-DARTS: iterative,
     *  direct, CPU, GPU, HYPRE/MGR, and -- through linsolv_iface_adapter -- the
     *  external proprietary bos solvers. It replaces the legacy linsolv_iface,
     *  linsolv_iface_bos<N> and linear_solver_base hierarchies with one type.
     *
     *  Block size is not a template parameter of the interface: it is carried
     *  at run time by csr_matrix_base::n_row_size. Block-templated concrete
     *  solvers assert that n_row_size matches their template parameter before
     *  down-casting (see SOLVER_REFACTORING_PLAN.md, decision OD-6).
     *
     *  Lifecycle, per simulation run:
     *    init()                       -- once, binds matrix structure + config
     *    setup() then solve()         -- once per Newton iteration
     */
    class linear_solver
    {
    public:
      virtual ~linear_solver() = default;

      /** One-time initialization: bind the matrix and apply the configuration.
       *  @param A       sparse matrix (its n_row_size gives the block size)
       *  @param config  solver configuration; concrete solvers expect their own
       *                 solver_config-derived type
       *  @return 0 on success, non-zero on failure
       */
      virtual int init(opendarts::linear_solvers::csr_matrix_base *A,
          const opendarts::linear_solvers::solver_config &config) = 0;

      /** Per-Newton-iteration setup: ingest the updated matrix values and
       *  rebuild the preconditioner / factorization.
       *  @return 0 on success, non-zero on failure
       */
      virtual int setup(opendarts::linear_solvers::csr_matrix_base *A) = 0;

      /** Solve A x = rhs.
       *  @return 0 on success, non-zero on failure
       */
      virtual int solve(opendarts::config::mat_float *rhs,
          opendarts::config::mat_float *x) = 0;

      /** Attach a preconditioner. Self-preconditioning solvers (e.g. MGR,
       *  direct solvers) ignore it; the default is a no-op.
       */
      virtual void set_prec(opendarts::linear_solvers::linear_solver * /*prec*/) {}

      /** Outcome of the last solve(): iterations, residual, timings. */
      virtual opendarts::linear_solvers::solver_stats stats() const = 0;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_HPP
//--------------------------------------------------------------------------
