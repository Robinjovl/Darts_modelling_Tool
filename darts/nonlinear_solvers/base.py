"""Basic nonlinear solver specifications and the runtime base classes.

This module holds everything that is common to all nonlinear solution methods:

- the residual :class:`Norm` enum,
- the composable sub-specs (:class:`ChopSpec`, :class:`LineSearchSpec`,
  :class:`OBLBoundsSpec`, :class:`InexactNewtonSpec`),
- the divergence-fallback spec (:class:`FallbackSpec`),
- the base solver spec :class:`NonlinearSolverSpec` (and the not-yet-implemented
  :class:`PicardSpec`),
- the runtime base classes :class:`NonlinearStatus`, :class:`SolverStats` and
  :class:`NonlinearSolver`, which stage every nonlinear iteration into
  ``pre_nonlinear`` / ``update`` / ``post_nonlinear`` and drive the
  fallback retries of :meth:`NonlinearSolver.solve_timestep`.

Method-specific specs and solvers live in sibling modules (e.g.
:mod:`darts.nonlinear_solvers.newton`).
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ------------------------------------------------------------------ enums


class Norm(Enum):
    """Norm used for the nonlinear residual convergence check."""

    L1 = "L1"
    L2 = "L2"
    LINF = "LINF"


_NORM_TO_ENUM = {Norm.L1: 0, Norm.L2: 1, Norm.LINF: 2}  # engine.residual_norm_type


# ------------------------------------------------------------------ sub-specs


@dataclass
class ChopSpec:
    """Nonlinear-update chopping strategy (composition-change limiting).

    :ivar mode: ``'local'`` (per-block composition chop, default), ``'global'``
        (single scalar max-ratio limiter) or ``None`` (plain update).
    :ivar factor: maximum allowed composition change per iteration
        (previously ``sim_params.newton_params[0]``).
    :ivar log_transform: solve in log-transformed composition variables
        (previously ``sim_params.log_transform``).
    """

    mode: str | None = "local"
    factor: float = 0.1
    log_transform: bool = False

    def __post_init__(self):
        assert self.mode in (None, "local", "global"), (
            f"Unknown chop mode '{self.mode}', expected None, 'local' or 'global'"
        )


@dataclass
class LineSearchSpec:
    """Backtracking line search on the nonlinear update."""

    enabled: bool = False
    min_update: float = 1e-4


@dataclass
class OBLBoundsSpec:
    """Restraining of the nonlinear trajectory within predefined bounds.

    :ivar mode: ``None`` (disabled, default), ``'obl_axes'`` (clamp the update
        into the OBL parametrization box via the C++
        ``apply_obl_axis_local_correction`` kernel) or ``'physical'``
        (per-axis physical box, e.g. ``p >= p_min``; reserved for the
        axis box-constraint extension).
    :ivar axis_min: optional per-axis lower bounds overriding the OBL axes.
    :ivar axis_max: optional per-axis upper bounds overriding the OBL axes.
    """

    mode: str | None = None
    axis_min: list | None = None
    axis_max: list | None = None


@dataclass
class InexactNewtonSpec:
    """Placeholder: inexact Newton via a forcing sequence on the linear tolerance.

    The forcing term is applied purely on the Python side by adapting the
    linear solver tolerance per nonlinear iteration.

    :ivar forcing: ``'constant'`` or the Eisenstat-Walker choices ``'EW1'``/``'EW2'``.
    """

    forcing: str = "constant"
    eta0: float = 1e-1
    eta_max: float = 1e-1


# ------------------------------------------------------------------ solver specs


@dataclass
class FallbackSpec:
    """Fallback applied when the nonlinear solve of a timestep fails, before
    the timestep is cut. Fallbacks are tried in the order they are listed in
    :attr:`NonlinearSolverSpec.fallbacks`; each retries the SAME timestep
    (the engine state was rolled back by the failed attempt).

    :ivar solver: alternative solver spec to retry with (``None`` = retry with
        the primary spec, useful together with extra routines).
    :ivar pre_routines: extra routines ``f(solver, dt, t, iteration)`` run in
        ``pre_nonlinear`` of every retry iteration (built-in or user-defined).
    :ivar post_routines: extra routines run in ``post_nonlinear``.
    """

    solver: "NonlinearSolverSpec | None" = None
    pre_routines: list = field(default_factory=list)
    post_routines: list = field(default_factory=list)


@dataclass
class NonlinearSolverSpec:
    """Base nonlinear solver specification: convergence control common to all methods.

    This object is the single owner of the nonlinear solve settings; it fully
    replaces the Newton-related fields of the retired ``sim_params`` structure.

    :ivar tolerance: residual tolerance of the reservoir blocks.
    :ivar well_tolerance_multiplier: well-block tolerance = ``tolerance * well_tolerance_multiplier``.
    :ivar max_iterations: maximum nonlinear iterations per timestep.
    :ivar stationary_point_tolerance: relative residual-stagnation tolerance
        used for stationary-point detection.
    :ivar norm: residual norm type (:class:`Norm`).
    :ivar coupled_well_res_norm_method: norm evaluation method (1 or 2) for the
        coupled well-reservoir residual of DFM wells.
    :ivar pre_routines: user routines ``f(solver, dt, t, iteration)`` run at the
        start of ``pre_nonlinear`` on every nonlinear iteration.
    :ivar post_routines: user routines run in ``post_nonlinear`` after the update.
    :ivar fallbacks: ordered :class:`FallbackSpec` list tried when the timestep
        solve diverges, before the timestep is cut.
    """

    tolerance: float = 1e-2
    well_tolerance_multiplier: float = 100.0
    max_iterations: int = 20
    stationary_point_tolerance: float = 1e-3
    norm: Norm = Norm.L2
    coupled_well_res_norm_method: int = 1
    pre_routines: list = field(default_factory=list)
    post_routines: list = field(default_factory=list)
    fallbacks: list = field(default_factory=list)

    def make_solver(self, model) -> "NonlinearSolver":
        """Materialize the runtime solver driving this spec."""
        raise NotImplementedError(
            f"{type(self).__name__} has no runtime solver implementation yet"
        )

    def sync_to_engine(self, engine):
        """Write the kernel-level controls of this spec into the C++ engine.
        The engine owns only the raw kernel knobs; they are synced from the
        spec before every timestep solve."""
        engine.residual_norm_type = _NORM_TO_ENUM[self.norm]


@dataclass
class PicardSpec(NonlinearSolverSpec):
    """Placeholder: Picard (fixed-point) iterations."""

    def make_solver(self, model):
        raise NotImplementedError("Picard iterations are not implemented yet")


# ------------------------------------------------------------------ runtime status


class NonlinearStatus:
    """Per-timestep nonlinear solve status, replacing the C++ engine's
    ``n_newton_last_dt``/``n_linear_last_dt``/``newton_residual_last_dt``/
    ``well_residual_last_dt`` members."""

    def __init__(self):
        self.n_newton = 0
        self.n_linear = 0
        self.newton_residual = np.inf
        self.well_residual = np.inf
        self.linear_solver_rc = 0
        self.residual_history = []

    def reset(self):
        self.__init__()


class SolverStats:
    """Cumulative simulation statistics, replacing the C++ ``sim_stat``/``engine.stat``."""

    def __init__(self):
        self.n_timesteps_total = 0
        self.n_timesteps_wasted = 0
        self.n_newton_total = 0
        self.n_newton_wasted = 0
        self.n_linear_total = 0
        self.n_linear_wasted = 0

    def update(self, converged: bool, status: NonlinearStatus):
        if converged:
            self.n_timesteps_total += 1
            self.n_newton_total += status.n_newton
            self.n_linear_total += status.n_linear
        else:
            self.n_timesteps_wasted += 1
            self.n_newton_wasted += status.n_newton
            self.n_linear_wasted += status.n_linear

    def print(self):
        print(
            f"----- TS = {self.n_timesteps_total:d}({self.n_timesteps_wasted:d}), "
            f"NI = {self.n_newton_total:d}({self.n_newton_wasted:d}), "
            f"LI = {self.n_linear_total:d}({self.n_linear_wasted:d}) -----"
        )


def write_to_log(message: str):
    """Write to the darts output stream (the file set by redirect_darts_output),
    matching where the C++ engine used to print these lines."""
    try:
        from darts.engines import write_to_darts_output

        write_to_darts_output(message)
    except ImportError:
        print(message, end="")


# ------------------------------------------------------------------ base solver


class NonlinearSolver:
    """Base runtime nonlinear solver: owns the timestep solve loop.

    Subclasses implement :meth:`run_timestep` and :meth:`build_corrections`.
    The heavy kernels stay on the C++ engine; models can customize behaviour by
    overriding the hooks :meth:`compute_residuals` and :meth:`on_iteration` in a
    subclass, or by attaching routines to the spec's ``pre_routines`` /
    ``post_routines``.
    """

    def __init__(self, model, spec: NonlinearSolverSpec):
        self.model = model
        self.spec = spec
        self.status = NonlinearStatus()
        self.stats = SolverStats()
        # ordered dX-correction pipeline assembled from the spec by build_corrections()
        self._corrections = []
        # extra routines injected by a FallbackSpec retry (empty on the primary solver)
        self.extra_pre_routines = []
        self.extra_post_routines = []

    @property
    def engine(self):
        return self.model.physics.engine

    @property
    def timer(self):
        return self.model.timer

    def run_timestep(self, dt: float, t: float, verbose: int | None = None) -> bool:
        raise NotImplementedError

    def solve_timestep(self, dt: float, t: float, verbose: int | None = None) -> bool:
        """Solve one timestep: run the primary solver and, on divergence, try
        the spec's fallbacks in order (each retries the SAME dt — the failed
        attempt rolled the engine state back). Only when every fallback fails
        does the caller cut the timestep."""
        converged = self.run_timestep(dt, t, verbose)
        for fallback in self.spec.fallbacks:
            if converged:
                break
            solver = self._make_fallback_solver(fallback)
            write_to_log(
                f"Nonlinear fallback: retrying DT = {dt:g} with {type(solver.spec).__name__}\n"
            )
            converged = solver.run_timestep(dt, t, verbose)
            # expose the last attempt's status to the model/driver
            self.status = solver.status
        return converged

    def _make_fallback_solver(self, fallback: FallbackSpec) -> "NonlinearSolver":
        spec = fallback.solver if fallback.solver is not None else self.spec
        solver = spec.make_solver(self.model)
        solver.stats = self.stats  # single cumulative statistics object
        solver.extra_pre_routines = list(fallback.pre_routines)
        solver.extra_post_routines = list(fallback.post_routines)
        return solver

    # -------- per-iteration stages (assembled from the spec)

    def pre_nonlinear(self, dt: float, t: float, iteration: int):
        """Stage run before the update of every nonlinear iteration: user/fallback
        pre-routines, then the spec-assembled dX-correction pipeline
        (composition correction, chopping, OBL-bounds constraints, ...)."""
        for routine in list(self.spec.pre_routines) + self.extra_pre_routines:
            routine(self, dt, t, iteration)
        for step in self._corrections:
            step()

    def update(self, dt: float):
        """Plain nonlinear update ``X -= newton_update_coefficient * dX``."""
        return self.engine.apply_update(dt)

    def post_nonlinear(self, dt: float, t: float, iteration: int):
        """Stage run after the update of every nonlinear iteration."""
        for routine in list(self.spec.post_routines) + self.extra_post_routines:
            routine(self, dt, t, iteration)

    def build_corrections(self) -> list:
        """Assemble the ordered dX-correction pipeline from the spec.
        Subclasses override; each returned callable operates on the engine's
        own X/dX and guards its own applicability."""
        return []

    # -------- overridable hooks

    def compute_residuals(self) -> tuple:
        """Return ``(reservoir_residual, well_residual)`` for the current state."""
        engine = self.engine
        if self.model.has_dfm_well:
            res = engine.calc_coupled_well_reservoir_residual(
                self.spec.coupled_well_res_norm_method
            )
        else:
            res = engine.calc_newton_residual()
        return res, engine.calc_well_residual()

    def on_iteration(self, iteration: int, dt: float, t: float):
        """Hook called after residual evaluation of every nonlinear iteration."""
        pass

    def converged(self) -> bool:
        """Convergence verdict for the finished nonlinear loop (previously the
        re-check inside the C++ ``post_newtonloop``)."""
        status, spec = self.status, self.spec
        return not (
            status.linear_solver_rc != 0
            or status.newton_residual >= spec.tolerance
            or status.well_residual > spec.tolerance * spec.well_tolerance_multiplier
        )

    def _failure_message(self, dt: float) -> str:
        status = self.status
        if status.linear_solver_rc == 1:
            reason = "linear solver setup failed"
        elif status.linear_solver_rc == 2:
            reason = "linear solver solve failed"
        elif status.newton_residual >= self.spec.tolerance:
            reason = "newton residual reservoir"
        else:
            reason = "newton residual wells"
        return f"FAILED TO CONVERGE WITH DT = {dt:.3f} ({reason}) \n"

    def _solve_linear(self) -> int:
        """Solve the linearized system, routing to the Python-resident solvers
        (PETSc/Pardiso) when selected; returns the solver return code."""
        from darts.input.input_data import linear_solver_types

        linear_type = self.model.data_ts.linear_type
        if isinstance(linear_type, linear_solver_types):
            # solvers via Python interface
            if linear_type in [
                linear_solver_types.CPU_PETSC_CPR,
                linear_solver_types.CPU_PETSC_FS,
            ]:
                self.model.petsc_solve_linear_equation()
            elif linear_type in [linear_solver_types.CPU_PARDISO]:
                self.model.pardiso_solve_linear_equation()
            else:
                raise Exception("Unknown linear solver type", linear_type)
            return 0
        # compile-time C++ linear solvers
        engine = self.engine
        rc = engine.solve_linear_equation()
        if rc == 0:
            status = self.status
            status.n_linear += engine.get_last_linear_iters()
            write_to_log(
                f"\t #{status.n_newton + 1:d} ({status.newton_residual:.4e}, {status.well_residual:.4e}): lin {engine.get_last_linear_iters():d} ({engine.get_last_linear_residual():.1e})\n"
            )
        return rc
