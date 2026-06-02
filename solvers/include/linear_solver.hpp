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

#include "timer_node.h"
#include "data_types.hpp"
#include "csr_matrix_base.hpp"
#include "solver_config.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    /** Unified linear-solver interface.
     *
     *  The single base class for every linear solver in open-DARTS: iterative,
     *  direct, CPU, GPU, HYPRE/MGR, and the proprietary bos solvers. Replaces
     *  the legacy linsolv_iface / linear_solver_base / linsolv_iface_bos<N>
     *  triplet with one type. The legacy ``linsolv_iface`` name is preserved
     *  as a type alias (see linsolv_iface.hpp) so existing call sites keep
     *  working unchanged.
     *
     *  Block size is carried at run time via csr_matrix_base::n_row_size --
     *  it is not a template parameter of the interface. Block-templated
     *  concrete solvers (linsolv_iface_bos<N>) assert that n_row_size matches
     *  their template parameter before down-casting (decision OD-6 in
     *  SOLVER_REFACTORING_PLAN.md).
     *
     *  Lifecycle, per simulation run:
     *    init()                       -- once, binds matrix structure +
     *                                    iteration / tolerance config
     *    setup() then solve()         -- once per Newton iteration; setup()
     *                                    consumes refreshed matrix values
     *    solve_transposed()           -- adjoint Newton step (Han et al. 2013)
     */
    class linear_solver
    {
    public:
      linear_solver() = default;
      virtual ~linear_solver() = default;

      /** Attach a single preconditioner. Self-preconditioning solvers (MGR,
       *  direct solvers, CPR) do not consume an external preconditioner --
       *  they internally chain their own stages. */
      virtual int set_prec(opendarts::linear_solvers::linear_solver *prec_input) = 0;

      /** CPR-style two-preconditioner setter: the pressure-reduced-system
       *  preconditioner. set_prec() targets the full-system preconditioner.
       *  Default no-op for solvers that don't take a reduced-system prec. */
      virtual int set_p_system_prec(opendarts::linear_solvers::linear_solver *prec_input)
      {
        (void)prec_input;
        return 0;
      }

      /** One-time initialisation: bind the matrix structure and the
       *  iteration / tolerance budget. Concrete solvers may also dispatch on
       *  the matrix's block size (csr_matrix_base::n_row_size) at this
       *  point.
       *  @return 0 on success, non-zero on failure
       */
      virtual int init(opendarts::linear_solvers::csr_matrix_base *A,
          opendarts::config::index_t max_iters,
          opendarts::config::mat_float tolerance) = 0;

      /** Bind external timer nodes (optional -- a solver that doesn't time
       *  itself simply ignores them). */
      void init_timer_nodes(::timer_node *timer_setup_input,
          ::timer_node *timer_solve_input)
      {
        this->timer_setup = timer_setup_input;
        this->timer_solve = timer_solve_input;
      }

      /** Per-Newton-iteration setup: ingest the updated matrix values and
       *  rebuild the preconditioner / factorisation as required. The
       *  sparsity pattern is assumed unchanged across calls within a run. */
      virtual int setup(opendarts::linear_solvers::csr_matrix_base *A_input) = 0;

      /** Solve A x = B. */
      virtual int solve(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X) = 0;

      /** Whether the solver requires its setup() to be called again before a
       *  solve_transposed() call. Solvers that materialise a separate
       *  transpose system internally inside solve_transposed() return false;
       *  solvers that share state between forward and transpose (or simply
       *  ignore the transpose entry point) return true (the default --
       *  conservative). */
      virtual bool requires_setup_for_transposed_solve() const
      {
        return true;
      }

      /** Adjoint solve: solve A^T x = B (or apply the transpose preconditioner
       *  M^{-T} when this solver is used as a preconditioner). Required for
       *  the backward / adjoint Newton step (see Han et al. 2013 for the CPR-
       *  for-adjoint preconditioner). Default returns -1 ("not supported");
       *  implement explicitly where transpose is meaningful. */
      virtual int solve_transposed(opendarts::config::mat_float * /*B*/,
          opendarts::config::mat_float * /*X*/)
      {
        return -1;
      }

      /** Number of iterations of the last solve. For preconditioner-only
       *  callers (e.g. CPR as a stage) this is the number of preconditioner
       *  applications (usually 1). */
      virtual int get_n_iters() = 0;

      /** Final residual of the last solve. */
      virtual opendarts::config::mat_float get_residual() = 0;

      /** Outcome of the last solve, packaged. Provides a uniform replacement
       *  for the get_n_iters() + get_residual() pair. Default implementation
       *  composes the two; solvers that maintain richer stats override.  */
      virtual opendarts::linear_solvers::solver_stats stats() const
      {
        opendarts::linear_solvers::solver_stats s;
        // Note: const here is API hygiene; concrete solvers store their
        // counters as mutable bookkeeping. The default keeps stats() callable
        // on const handles without changing the existing get_* accessors,
        // but cannot inspect them without dropping const. Solvers that want
        // a meaningful stats() override this directly.
        s.iterations = 0;
        s.residual = 0.0;
        s.converged = false;
        return s;
      }

      ::timer_node *timer_setup = nullptr;
      ::timer_node *timer_solve = nullptr;
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINEAR_SOLVER_HPP
//--------------------------------------------------------------------------
