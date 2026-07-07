"""Adaptive linear-solver switching during a simulation run.

An :class:`AdaptiveSolverSpec` holds an ordered list of candidate
:class:`~darts.solvers.specs.LinearSolverSpec` objects and a policy. After every
timestep ``DartsModel`` evaluates the policy and, when it selects a different
candidate, rebuilds and injects that solver -- so a run can, for example, start
on HYPRE MGR and fall back to SuperLU if MGR fails to converge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .specs import GPUSolverSpec, LinearSolverSpec, PythonLinearSolverSpec


@dataclass
class SolverSwitchContext:
    """State passed to an adaptive switching policy after each timestep.

    :ivar current_index: index of the candidate currently in use.
    :ivar timestep_converged: whether the just-finished timestep converged.
    :ivar linear_solver_error: ``engine.linear_solver_error_last_dt``
        (0 = ok, 1 = setup failed, 2 = solve failed).
    :ivar linear_iterations: linear iterations in the last timestep.
    :ivar newton_iterations: Newton iterations in the last timestep.
    :ivar consecutive_failures: running count of consecutive non-converged
        timesteps.
    :ivar dt: size of the just-attempted timestep (days).
    :ivar simulation_time: engine time after the attempt (days).
    :ivar ls_setup_time: linear-solver SETUP seconds spent in this attempt
        (delta of the engine timer).
    :ivar ls_solve_time: linear-solver SOLVE seconds spent in this attempt.
    :ivar phase: free-form model phase tag (``model.solver_phase``) for
        phase-driven policies (e.g. 'static' / 'dynamic'); ``None`` if unset.

    Together with :class:`SolverAction` this is a ready contextual-bandit
    interface: the context is the observation, the action is a candidate
    switch and/or a parameter update, and (ls_setup_time + ls_solve_time)
    per converged dt is the natural reward signal.
    """

    current_index: int
    timestep_converged: bool
    linear_solver_error: int
    linear_iterations: int
    newton_iterations: int
    consecutive_failures: int
    dt: float = 0.0
    simulation_time: float = 0.0
    ls_setup_time: float = 0.0
    ls_solve_time: float = 0.0
    phase: str | None = None


@dataclass
class SolverAction:
    """What an adaptive policy wants done before the next timestep attempt.

    :ivar index: candidate to switch to (``None`` = stay on the current one).
    :ivar updates: field updates applied to the (possibly newly selected)
        solver through :meth:`DartsModel.update_solver` -- hot/warm fields
        reconfigure the live solver in place, with no Jacobian impact.

    A policy may also simply return an ``int`` (the candidate index) --
    the legacy contract.
    """

    index: int | None = None
    updates: dict | None = None


def fallback_on_failure(context: SolverSwitchContext) -> int:
    """Stock policy: stay on the current solver while it works; advance to the
    next (more robust) candidate after a failed timestep or a linear-solver
    error. With ``candidates=[MGRSolverSpec(), SuperLUSolverSpec()]`` this gives
    "iterate with MGR, fall back to the SuperLU direct solver on failure".
    """
    if not context.timestep_converged or context.linear_solver_error != 0:
        return context.current_index + 1
    return context.current_index


@dataclass
class AdaptiveSolverSpec(LinearSolverSpec):
    """Meta-spec that switches between candidate solvers during a run.

    :param candidates: ordered list of ``LinearSolverSpec``; index 0 is used
        first. Each is built on demand for the model's block size.
    :param policy: ``callable(SolverSwitchContext) -> int | SolverAction``
        evaluated after every timestep. Defaults to
        :func:`fallback_on_failure`.
    :param on_timestep_failed: optional
        ``callable(SolverSwitchContext) -> int | SolverAction | None``
        evaluated when a timestep did NOT converge, BEFORE the engine retries
        it with a cut dt -- so the retry itself runs on the fallback solver /
        tightened parameters. Returning ``None`` defers to ``policy``.
    """

    candidates: list[LinearSolverSpec] = field(default_factory=list)
    policy: Callable[[SolverSwitchContext], int | SolverAction] | None = None
    on_timestep_failed: (
        Callable[[SolverSwitchContext], int | SolverAction | None] | None
    ) = None

    def __post_init__(self):
        if not self.candidates:
            raise ValueError("AdaptiveSolverSpec requires at least one candidate")
        # Validate candidate types NOW: switching rebuilds a candidate through
        # the open-source CPU registry and injects it into the live engine.
        # GPU specs (enum-selected by the engine factory) and Python-resident
        # solvers (owned by the model, not injectable) previously failed only
        # at switch time -- potentially hours into a run.
        for i, cand in enumerate(self.candidates):
            if isinstance(cand, GPUSolverSpec | PythonLinearSolverSpec):
                raise TypeError(
                    f"AdaptiveSolverSpec candidate {i} ({type(cand).__name__}) "
                    "cannot be switched mid-run: only engine-resident CPU "
                    "registry specs (MGRSolverSpec, GMRESSolverSpec, "
                    "CPRSolverSpec, SuperLUSolverSpec, ...) are supported."
                )
            if not isinstance(cand, LinearSolverSpec):
                raise TypeError(
                    f"AdaptiveSolverSpec candidate {i} must be a "
                    f"LinearSolverSpec, got {type(cand).__name__}"
                )
        if self.policy is None:
            self.policy = fallback_on_failure

    def build(self, block_size: int):
        """Build the first candidate's solver. ``DartsModel`` re-injects later
        candidates as the policy selects them."""
        return self.candidates[0].build(block_size)

    def choose(self, context: SolverSwitchContext) -> int:
        """Evaluate the policy, clamped to a valid candidate index."""
        index = int(self.policy(context))
        return max(0, min(index, len(self.candidates) - 1))
