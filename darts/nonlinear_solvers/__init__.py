"""Nonlinear solver specifications and drivers for DARTS.

This package is the single source of input parameters for the nonlinear
(timestep) solve and the home of the Python-side nonlinear solver classes:

- :mod:`darts.nonlinear_solvers.base` — the residual norm enum, the composable
  sub-specs (chopping, line search, OBL bounds, inexact-Newton forcing), the
  divergence-fallback spec, the base :class:`NonlinearSolverSpec` and the
  runtime base classes (status, statistics, :class:`NonlinearSolver`).
- :mod:`darts.nonlinear_solvers.newton` — the Newton family of specs and the
  :class:`NewtonSolver` driver.

Design (mirrors the linear-solver specs of MR280):

- A model selects a nonlinear solver by assigning a spec to
  ``DartsModel.nonlinear_solver`` inside an overridden ``set_solver()`` — the
  same hook where the linear solver is specified (MR280)::

      def set_solver(self):
          super().set_solver()
          self.nonlinear_solver.tolerance = 1e-4
          self.nonlinear_solver.chop.factor = 0.2

  or by replacing the spec entirely::

      self.nonlinear_solver = NewtonSpec(tolerance=1e-4, chop=ChopSpec(mode='global'))

- The spec is materialized into a runtime solver object (``NewtonSolver``) by
  ``DartsModel._apply_nonlinear()`` during ``init()``; the solver owns
  ``run_timestep()`` — the nonlinear loop driving the C++ per-iteration kernels
  (assembly, linear solve, update corrections, residual norms). Every iteration
  is staged into ``pre_iteration`` (user routines + the dX-correction pipeline
  assembled from the spec: composition correction, chopping, OBL-bounds
  constraints), ``update`` and ``post_iteration``.

- On divergence of the primary solver, the ordered ``spec.fallbacks``
  (:class:`FallbackSpec`) are tried on the same timestep — with extra
  built-in/user pre/post routines and/or another solver type — before the
  driver cuts the timestep (``NonlinearSolver.solve_timestep``).

- Timestep control (dt_first/dt_min/dt_mult/dt_max/eta) lives in the model's
  ``data_ts`` structure and is consumed by ``DartsModel.run()``.

The C++ engine keeps only the computationally intensive kernels (assembly,
linear solve, residual norms and the cell-looping dX corrections); all control
flow and convergence decisions live here.
"""

from darts.nonlinear_solvers.base import (
    ChopSpec,
    FallbackSpec,
    InexactNewtonSpec,
    LineSearchSpec,
    NonlinearSolver,
    NonlinearSolverSpec,
    NonlinearStatus,
    Norm,
    OBLBoundsSpec,
    PicardSpec,
    SolverStats,
    write_to_log,
)
from darts.nonlinear_solvers.newton import (
    NewtonSolver,
    NewtonSpec,
    QuasiNewtonSpec,
    TrustRegionNewtonSpec,
    default_nonlinear_solver,
    default_nonlinear_spec,
)

__all__ = [
    # enums
    "Norm",
    # sub-specs
    "ChopSpec",
    "LineSearchSpec",
    "OBLBoundsSpec",
    "InexactNewtonSpec",
    "FallbackSpec",
    # solver specs
    "NonlinearSolverSpec",
    "NewtonSpec",
    "QuasiNewtonSpec",
    "TrustRegionNewtonSpec",
    "PicardSpec",
    "default_nonlinear_solver",
    "default_nonlinear_spec",
    # runtime
    "NonlinearSolver",
    "NewtonSolver",
    "NonlinearStatus",
    "SolverStats",
    "write_to_log",
]
