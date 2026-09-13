"""Nonlinear solver specifications and drivers for DARTS.

This package is the single source of input parameters for the nonlinear
(timestep) solve and the home of the Python-side nonlinear solver classes:

- :mod:`darts.nonlinear_solvers.base` — the residual norm enum, the composable
  sub-specs (chopping, line search, OBL bounds, inexact-Newton forcing), the
  divergence-fallback spec, the base :class:`NonlinearSolverSpec` and the
  runtime base classes (status, statistics, :class:`NonlinearSolver`).
- :mod:`darts.nonlinear_solvers.newton` — the Newton family of specs and the
  :class:`NewtonSolver` driver.

Design (specified in ``set_solver()``, the same hook as the linear solver of MR280):

- A model selects a nonlinear solver by assigning a solver *instance* to
  ``DartsModel.nonlinear_solver`` inside an overridden ``set_solver()`` — either
  by replacing it (``NewtonSolver`` accepts a :class:`NewtonSpec` positionally
  or its keyword arguments)::

      def set_solver(self):
          super().set_solver()
          self.nonlinear_solver = NewtonSolver(tolerance=1e-4,
                                               chop=ChopSpec(mode='global'))

  or by tuning the spec of the default solver in place::

      def set_solver(self):
          super().set_solver()
          self.nonlinear_solver.spec.tolerance = 1e-4
          self.nonlinear_solver.spec.chop.factor = 0.2

- The solver instance (created in ``set_solver()``)
  is bound to the model by ``DartsModel.init()`` (and re-bound by ``reset()``). It
  owns ``run_timestep()`` — the nonlinear loop driving the C++ per-iteration
  kernels (assembly, linear solve, dX corrections, residual norms). Every
  iteration is staged into ``pre_iteration`` (user routines), ``update`` (the
  spec-assembled dX-correction pipeline — composition correction, chopping,
  OBL-bounds constraints, thermal — followed by the plain Newton step) and
  ``post_iteration`` (user routines); line-search trials reuse the same
  ``update``. The input spec stays retrievable as ``nonlinear_solver.spec``
  (serializable via ``.to_dict()``).

- On divergence of the primary solver, the ordered ``spec.fallbacks``
  (:class:`FallbackSpec`) are tried on the same timestep — with extra
  built-in/user pre/post routines and/or another solver type — before the
  driver cuts the timestep (``NonlinearSolver.solve_timestep``).

- Timestep control (dt_first/dt_min/dt_mult/dt_max/eta) lives in the model's
  ``ts_control`` structure and is consumed by ``DartsModel.run()``.

The C++ engine keeps only the computationally intensive kernels (assembly,
linear solve, residual norms and the cell-looping dX corrections); all control
flow and convergence decisions live here.
"""

from darts.nonlinear_solvers.base import (
    ChopSpec,
    FallbackSpec,
    NonlinearSolver,
    NonlinearSolverSpec,
    NonlinearStatus,
    Norm,
    OBLBoundsSpec,
    SolverStats,
    write_to_log,
)
from darts.nonlinear_solvers.mechanics import (
    TIME_INTEGRATION_SCHEMES,
    MechanicsNewtonSolver,
    configure_time_integration,
)
from darts.nonlinear_solvers.newton import (
    NewtonSolver,
    NewtonSpec,
)

__all__ = [
    # enums
    "Norm",
    # sub-specs
    "ChopSpec",
    "OBLBoundsSpec",
    "FallbackSpec",
    # solver specs
    "NonlinearSolverSpec",
    "NewtonSpec",
    # runtime
    "NonlinearSolver",
    "NewtonSolver",
    "MechanicsNewtonSolver",
    "configure_time_integration",
    "TIME_INTEGRATION_SCHEMES",
    "NonlinearStatus",
    "SolverStats",
    "write_to_log",
]
