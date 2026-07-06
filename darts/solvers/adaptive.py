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
    """

    current_index: int
    timestep_converged: bool
    linear_solver_error: int
    linear_iterations: int
    newton_iterations: int
    consecutive_failures: int


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
    :param policy: ``callable(SolverSwitchContext) -> int`` returning the index
        of the candidate to use next. Defaults to :func:`fallback_on_failure`.
    """

    candidates: list[LinearSolverSpec] = field(default_factory=list)
    policy: Callable[[SolverSwitchContext], int] | None = None

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
