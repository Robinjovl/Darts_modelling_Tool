"""open-DARTS linear solvers.

This package exposes:

* the compiled ``solvers`` extension -- the solver registry, the MGR API and
  the C++ configuration objects (re-exported here for backward compatibility,
  so ``from darts import solvers; solvers.create_linear_solver(...)`` works);
* the Python :mod:`~darts.solvers.specs` configuration classes
  (:class:`~darts.solvers.specs.MGRSolverSpec`, ``SuperLUSolverSpec``, ...);
* the HYPRE :mod:`~darts.solvers.enums` integer-code enumerations.

Typical use::

    from darts.solvers import MGRSolverSpec
    spec = MGRSolverSpec(tolerance=1e-4, kdim=50)
    solver = spec.build(block_size=3)
"""

from . import solvers  # noqa: F401
from .adaptive import (  # noqa: F401
    AdaptiveSolverSpec,
    SolverSwitchContext,
    fallback_on_failure,
)
from .enums import (  # noqa: F401
    CoarseGrid,
    FRelaxation,
    GlobalSmoother,
    Interpolation,
    Restriction,
    VariableRole,
)
from .solvers import *  # noqa: F401,F403
from .specs import (  # noqa: F401
    LinearSolverSpec,
    MGRLevelSpec,
    MGRSolverSpec,
    SuperLUSolverSpec,
    default_linear_solver,
)
