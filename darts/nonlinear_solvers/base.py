"""Basic nonlinear solver specifications and the runtime base classes.

This module holds everything that is common to all nonlinear solution methods:

- the residual :class:`Norm` enum,
- the composable sub-specs (:class:`ChopSpec`, :class:`OBLBoundsSpec`),
- the divergence-fallback spec (:class:`FallbackSpec`),
- the base solver spec :class:`NonlinearSolverSpec`,
- the runtime base classes :class:`NonlinearStatus`, :class:`SolverStats` and
  :class:`NonlinearSolver`, which stage every nonlinear iteration into
  ``pre_iteration`` / ``update`` / ``post_iteration`` and drive the
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


# Map each Python Norm to the NAME of the canonical C++ enum member
# (sim_params::nonlinear_norm_t). The integer values are taken from the compiled
# enum at sync time (see sync_to_engine) so there is a single source of truth and
# no hardcoded 0/1/2 that could silently desync if the C++ enum is reordered.
_NORM_TO_CPP_ENUM = {Norm.L1: "L1", Norm.L2: "L2", Norm.LINF: "LINF"}


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
        # explicit ValueError (not assert, which -O strips) — this validates user input
        if self.mode not in (None, "local", "global"):
            raise ValueError(
                f"Unknown chop mode '{self.mode}', expected None, 'local' or 'global'"
            )
        if self.factor <= 0:
            raise ValueError(f"ChopSpec.factor must be > 0, got {self.factor}")


@dataclass
class OBLBoundsSpec:
    """Restraining of the nonlinear trajectory within predefined bounds.

    :ivar mode: ``None`` (disabled, default), ``'obl_axes'`` (clamp the update
        into the OBL parametrization box via the C++
        ``apply_obl_axis_local_correction`` kernel; ``axis_min``/``axis_max``
        are passed to the kernel, populating the engine's per-region op_axis
        bounds) or ``'physical'`` (per-axis physical box, e.g. ``p >= p_min``;
        reserved for the axis box-constraint extension).
    :ivar axis_min: per-axis lower bounds (state-variable order), consumed by
        the ``'obl_axes'`` mode. ``None`` entries leave that axis unbounded;
        composition axes are usually left ``None`` — the composition
        correction kernel already projects onto the simplex.
    :ivar axis_max: per-axis upper bounds, same conventions as ``axis_min``.
    """

    mode: str | None = None
    axis_min: list | None = None
    axis_max: list | None = None

    def __post_init__(self):
        if self.mode not in (None, "obl_axes", "physical"):
            raise ValueError(
                f"Unknown OBL-bounds mode '{self.mode}', expected None, "
                "'obl_axes' or 'physical'"
            )
        # element-wise lower <= upper where both are given (None = unbounded axis)
        if self.axis_min is not None and self.axis_max is not None:
            for j, (lo, hi) in enumerate(
                zip(self.axis_min, self.axis_max, strict=False)
            ):
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError(
                        f"OBLBoundsSpec axis {j}: axis_min ({lo}) > axis_max ({hi})"
                    )


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
        ``pre_iteration`` of every retry iteration (built-in or user-defined).
    :ivar post_routines: extra routines run in ``post_iteration``.
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
        start of ``pre_iteration`` on every nonlinear iteration.
    :ivar post_routines: user routines run in ``post_iteration`` after the update.
    :ivar fallbacks: ordered :class:`FallbackSpec` list tried when the timestep
        solve diverges, before the timestep is cut.
    :ivar on_linear_nonconvergence: policy for a linear solve that exhausted its
        iteration budget on a *usable* iterate (finite, non-regressing residual;
        the unified ``linear_solver::solve()`` convention reports it as a
        positive code). ``'accept'`` (default) applies the inexact-Newton step
        and lets the Newton residual gate decide -- the historical behaviour of
        the default FGMRES+CPR solver; ``'cut'`` treats it as a failed solve and
        cuts the timestep -- the historical behaviour of MGR. A hard linear
        failure (non-finite or growing residual) always cuts, regardless of
        this policy.
    """

    tolerance: float = 1e-3
    well_tolerance_multiplier: float = 100.0
    max_iterations: int = 20
    stationary_point_tolerance: float = 1e-3
    norm: Norm = Norm.L2
    coupled_well_res_norm_method: int = 1
    pre_routines: list = field(default_factory=list)
    post_routines: list = field(default_factory=list)
    fallbacks: list = field(default_factory=list)
    # keyword-only: appending a positional field to this base would shift every
    # positional argument the subclasses inherit (NewtonSpec's `chop` was the
    # 10th), silently rebinding existing call sites.
    on_linear_nonconvergence: str = field(default="accept", kw_only=True)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate the convergence settings (called at construction and once per
        timestep from the runtime solver, so post-construction mutation is caught
        too). Explicit ValueError/TypeError — not ``assert`` (stripped by -O)."""
        if not isinstance(self.norm, Norm):
            raise TypeError(
                f"NonlinearSolverSpec.norm must be a Norm, got {type(self.norm).__name__}"
            )
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {self.tolerance}")
        if self.well_tolerance_multiplier <= 0:
            raise ValueError(
                f"well_tolerance_multiplier must be > 0, got {self.well_tolerance_multiplier}"
            )
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.stationary_point_tolerance <= 0:
            raise ValueError(
                f"stationary_point_tolerance must be > 0, got {self.stationary_point_tolerance}"
            )
        if self.coupled_well_res_norm_method not in (1, 2):
            raise ValueError(
                "coupled_well_res_norm_method must be 1 or 2, got "
                f"{self.coupled_well_res_norm_method}"
            )
        if self.on_linear_nonconvergence not in ("accept", "cut"):
            raise ValueError(
                "on_linear_nonconvergence must be 'accept' or 'cut', got "
                f"{self.on_linear_nonconvergence!r}"
            )

    def make_solver(self, model) -> "NonlinearSolver":
        """Materialize the runtime solver driving this spec."""
        raise NotImplementedError(
            f"{type(self).__name__} has no runtime solver implementation yet"
        )

    def sync_to_engine(self, engine):
        """Write the kernel-level controls of this spec into the C++ engine.
        The engine owns only the raw kernel knobs; they are synced from the
        spec before every timestep solve."""
        from darts.engines import sim_params

        # canonical int comes from the compiled enum (single source of truth)
        engine.residual_norm_type = int(
            getattr(sim_params, _NORM_TO_CPP_ENUM[self.norm])
        )

    def to_dict(self) -> dict:
        """Serialize this spec (and its nested sub-specs) to a plain dict for
        input tracing / autospec (see :meth:`NonlinearSolver.to_spec`).

        Uses :func:`dataclasses.asdict` today; if a spec is later migrated to a
        Pydantic model this transparently delegates to its ``model_dump()``.
        Note the dict retains :class:`Norm` enum members and routine callables,
        so it is dict-serializable but not directly ``json.dumps``-able without
        an enum/callable encoder."""
        dump = getattr(self, "model_dump", None)
        if callable(dump):  # Pydantic-forward-compatible
            return dump()
        from dataclasses import asdict

        return asdict(self)


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
        self.n_linear_nonconverged = 0
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

    The solver is the object a model assigns to ``DartsModel.nonlinear_solver``.
    It is constructed *detached* from its declarative :attr:`spec` (no model
    needed) and later :meth:`bind`\\ s to the model during ``init()``. The spec
    it runs stays retrievable via :attr:`spec` / :meth:`to_spec` (serializable via
    :meth:`NonlinearSolverSpec.to_dict`, Pydantic-forward-compatible) for input
    tracing / serialization.

    Subclasses implement :meth:`run_timestep` and :meth:`build_corrections`.
    The heavy kernels stay on the C++ engine; models can customize behaviour by
    overriding the residual hooks :meth:`compute_reservoir_residual` /
    :meth:`compute_well_residual` (both consumed by the Newton loop and line
    search) and :meth:`on_iteration`, or by attaching routines to the spec's
    ``pre_routines`` / ``post_routines``.
    """

    def __init__(self, spec: NonlinearSolverSpec, model=None):
        self.spec = spec
        self.model = model
        self.status = NonlinearStatus()
        self.stats = SolverStats()
        # ordered dX-correction pipeline assembled from the spec by build_corrections()
        self._corrections = []
        # extra routines injected by a FallbackSpec retry (empty on the primary solver)
        self.extra_pre_routines = []
        self.extra_post_routines = []

    def bind(self, model):
        """Attach this (possibly detached) solver to a model; returns self."""
        self.model = model
        return self

    def to_spec(self) -> NonlinearSolverSpec:
        """Return the specification this solver runs (the live config object the
        constructor kwargs write into). Serializable via
        ``solver.to_spec().to_dict()`` for input tracing (autospec)."""
        return self.spec

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
        if fallback.solver is not None:
            # explicit alternative spec -> build its own default runtime class
            solver = fallback.solver.make_solver(self.model)
        else:
            # same-spec retry: preserve THIS runtime (sub)class so on_iteration/
            # build_corrections/compute_* overrides survive the fallback rather
            # than degrading to a plain NewtonSolver. Requires the subclass to
            # keep a (spec, model=...)-compatible constructor.
            solver = type(self)(self.spec, model=self.model)
        solver.stats = self.stats  # single cumulative statistics object
        solver.extra_pre_routines = list(fallback.pre_routines)
        solver.extra_post_routines = list(fallback.post_routines)
        return solver

    # -------- per-iteration stages (assembled from the spec)

    def pre_iteration(self, dt: float, t: float, iteration: int):
        """Hook run before the update of every nonlinear iteration: the user/
        fallback pre-routines from the spec (once per iteration)."""
        for routine in list(self.spec.pre_routines) + self.extra_pre_routines:
            routine(self, dt, t, iteration)

    def apply_corrections(self):
        """Apply the spec-assembled dX-correction pipeline (composition
        correction, chopping, OBL-bounds constraints, thermal correction) to the
        engine's ``dX``. Applied before every ``apply_update`` — the main Newton
        step and each line-search trial alike."""
        for step in self._corrections:
            step()

    def update(self, dt: float):
        """Nonlinear update: apply the spec-assembled dX corrections, then the
        plain step ``X -= newton_update_coefficient * dX``. Used by both the
        main loop and every line-search trial."""
        self.timer.node["newton update"].start()
        self.apply_corrections()
        self.engine.apply_update(dt)
        self.timer.node["newton update"].stop()

    def post_iteration(self, dt: float, t: float, iteration: int):
        """Hook run after the update of every nonlinear iteration: the user/
        fallback post-routines from the spec (once per iteration)."""
        for routine in list(self.spec.post_routines) + self.extra_post_routines:
            routine(self, dt, t, iteration)

    def build_corrections(self) -> list:
        """Assemble the ordered dX-correction pipeline from the spec.
        Subclasses override; each returned callable operates on the engine's
        own X/dX and guards its own applicability."""
        return []

    # -------- overridable hooks

    def compute_reservoir_residual(self) -> float:
        """Reservoir (matrix) nonlinear residual for the current state. Override
        to define a custom residual (e.g. the mechanics deviatoric norm) without
        reimplementing the whole Newton loop; the loop and line search both call
        this so an override reaches every residual evaluation."""
        engine = self.engine
        if self.model.has_dfm_well:
            return engine.calc_coupled_well_reservoir_residual(
                self.spec.coupled_well_res_norm_method
            )
        return engine.calc_newton_residual()

    def compute_well_residual(self) -> float:
        """Well nonlinear residual for the current state (overridable)."""
        return self.engine.calc_well_residual()

    def compute_residuals(self) -> tuple:
        """Return ``(reservoir_residual, well_residual)`` for the current state.
        NOTE the main Newton loop calls :meth:`compute_reservoir_residual` and
        :meth:`compute_well_residual` separately (the reservoir residual is
        needed for stationary-point detection before the well residual is
        computed); this composite is used where both are needed at once (line
        search)."""
        return self.compute_reservoir_residual(), self.compute_well_residual()

    def on_iteration(self, iteration: int, dt: float, t: float):
        """Hook called after residual evaluation of every nonlinear iteration."""
        pass

    def converged(self) -> bool:
        """Convergence verdict for the finished nonlinear loop (previously the
        re-check inside the C++ ``post_newtonloop``)."""
        status, spec = self.status, self.spec
        # a NaN residual satisfies neither >= nor > below — reject it explicitly
        if not (
            np.isfinite(status.newton_residual) and np.isfinite(status.well_residual)
        ):
            return False
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
        elif status.linear_solver_rc == 3:
            reason = "linear solver did not converge (on_linear_nonconvergence='cut')"
        elif not (
            np.isfinite(status.newton_residual) and np.isfinite(status.well_residual)
        ):
            reason = "newton residual not finite"
        elif status.newton_residual >= self.spec.tolerance:
            reason = "newton residual reservoir"
        else:
            reason = "newton residual wells"
        return f"FAILED TO CONVERGE WITH DT = {dt:.3f} ({reason}) \n"

    def _solve_linear(self) -> int:
        """Solve the linearized system via the model's backend-neutral dispatch
        funnel and account the result uniformly for every backend.

        The backend selection lives in :meth:`DartsModel._solve_linear_equation`,
        which returns ``(rc, n_iters, residual)`` — ``rc`` is ``0`` on success,
        ``1`` on setup failure, ``2`` on solve failure (Python PETSc/Pardiso
        report failure via this same code, so a divergent solve now aborts the
        Newton loop / triggers a fallback exactly like the C++ path). This split
        is the seam the linear-solver refactoring (MR280) later replaces
        wholesale, so the accounting stays backend-agnostic here."""
        rc, n_iters, residual = self.model._solve_linear_equation()
        if rc in (0, 3):
            status = self.status
            status.n_linear += n_iters
            nc_marker = ""
            if rc == 3:
                # Unified solve() convention: the linear solver exhausted its
                # budget on a USABLE iterate (finite, non-regressing residual).
                # Count and mark it, then apply the spec policy: 'accept'
                # (default) takes the inexact-Newton step and lets the Newton
                # residual gate decide; 'cut' fails the step like a hard error.
                status.n_linear_nonconverged += 1
                nc_marker = " NC"
            write_to_log(
                f"\t #{status.n_newton + 1:d} ({status.newton_residual:.4e}, {status.well_residual:.4e}): lin {n_iters:d} ({residual:.1e}){nc_marker}\n"
            )
            if rc == 3 and self.spec.on_linear_nonconvergence == "accept":
                return 0
        return rc
