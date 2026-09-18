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
    /** Return-code constants of linear_solver::solve() (see its doc).
     *  Negative codes stay backend-specific; only the sign carries meaning. */
    namespace solve_result
    {
      constexpr int converged = 0;      ///< residual reached the tolerance
      constexpr int not_converged = 1;  ///< budget exhausted, iterate usable

      /** Relative slack allowed when deciding whether the final residual
       *  regressed past the one the solve started from. A Krylov method is
       *  non-increasing only in exact arithmetic, so a few ULPs of growth is
       *  round-off, not breakdown -- rejecting it would make the classification
       *  differ between backends purely by floating-point luck. Every backend
       *  (native C++ and the Python-resident ones) uses this same slack. */
      constexpr double residual_growth_rtol = 1.0e-12;

      /** True when @p final_residual counts as "no worse" than @p initial_residual
       *  under residual_growth_rtol. Starting from an exactly zero residual the
       *  system is already solved, so ANY positive final residual is growth --
       *  the relative slack has nothing to scale and must not be read as a free
       *  pass. */
      inline bool residual_did_not_regress(double final_residual, double initial_residual)
      {
        if (!(initial_residual > 0.0))
          return final_residual <= 0.0;
        return final_residual <= initial_residual * (1.0 + residual_growth_rtol);
      }
    } // namespace solve_result

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

      /** Value-only refresh after the engine has re-assembled the Jacobian
       *  but the sparsity pattern is unchanged. Solvers that own a HYPRE IJ
       *  matrix / AMG hierarchy / direct factorisation can override this to
       *  reuse the structural / hierarchy work and only re-push the new
       *  values -- significantly cheaper than a full setup() on subsequent
       *  Newton iterations. The default implementation falls back to
       *  setup(), so callers can always invoke refresh() safely; only
       *  solvers with a meaningful fast path override. */
      virtual int refresh(opendarts::linear_solvers::csr_matrix_base *A_input)
      {
        return this->setup(A_input);
      }

      /** Solve A x = B.
       *
       *  Unified return convention (signed status, PETSc-style):
       *    - 0                  : converged to tolerance (a direct solver's
       *                           exact solve and a preconditioner's single
       *                           application also report 0);
       *    - solve_result::not_converged (+1)
       *                         : iteration budget exhausted but the iterate is
       *                           USABLE -- the residual is finite and has not
       *                           risen above its initial value. Whether such a
       *                           step is applied or the timestep is cut is the
       *                           NONLINEAR solver's decision
       *                           (NonlinearSolverSpec.on_linear_nonconvergence),
       *                           never this solver's;
       *    - negative           : hard failure -- non-finite residual, residual
       *                           growth (numerical breakdown), or a backend
       *                           error. The iterate must not be applied; the
       *                           engine turns this into a timestep cut.
       *  Preconditioners applied inside a Krylov solver must return only
       *  0 / negative: a single application has no convergence to report, and
       *  the outer solver treats any nonzero as a hard failure. */
      virtual int solve(opendarts::config::mat_float *B,
          opendarts::config::mat_float *X) = 0;

      /** Apply a new configuration to a LIVE solver without rebinding the
       *  matrix. The hard invariant: reconfigure() never reallocates or
       *  touches the bound system matrix, the engine-owned vectors, or any
       *  structure-derived cache (csr_expansion, scalar adapter, thread
       *  partition) -- so it is always safe mid-run, between timesteps.
       *
       *  Field tiers (solver-specific):
       *    hot   -- stored and effective at the next solve/setup at zero
       *             extra cost (tolerances, iteration budgets, thresholds);
       *    warm  -- stored and the internal hierarchies/factorisations are
       *             rebuilt on the next setup() against the SAME bound
       *             matrices;
       *    structural -- cannot be applied in place; the call returns > 0
       *             and the caller falls back to building a fresh solver
       *             (engine_base::set_linear_solver re-inits it against the
       *             existing Jacobian, still with no matrix reallocation).
       *
       *  The factories construct solvers as `new T` + reconfigure(config),
       *  so this is also the single source of truth for config application.
       *
       *  @return 0 = fully applied; > 0 = unsupported in place, rebuild the
       *          solver; < 0 = error. The default cannot know any derived
       *          fields and conservatively requests a rebuild. */
      virtual int reconfigure(const opendarts::linear_solvers::solver_config & /*config*/)
      {
        return 1;
      }

      /** Outer-solver feedback: iteration count of the last outer Krylov
       *  solve. Preconditioners with reuse policies (CPR's adaptive AMG
       *  rebuild) override this; the default ignores it. Replaces the former
       *  hard dynamic_cast from linsolv_gmres to the concrete linsolv_cpr,
       *  so ANY preconditioner can now participate in the adaptive-rebuild
       *  feedback loop. */
      virtual void set_last_outer_iters(int /*n_iters*/) {}

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
