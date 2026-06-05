"""open-DARTS linear solvers.

This package exposes:

* the compiled ``solvers`` extension -- the solver registry, the MGR API and
  the C++ configuration objects (flattened into the ``darts.solvers``
  namespace so ``from darts import solvers; solvers.create_linear_solver(...)``
  works);
* the Python :mod:`~darts.solvers.specs` configuration classes
  (:class:`~darts.solvers.specs.MGRSolverSpec`, ``SuperLUSolverSpec``, ...);
* the HYPRE :mod:`~darts.solvers.enums` integer-code enumerations.

Typical use::

    from darts.solvers import MGRSolverSpec
    spec = MGRSolverSpec(tolerance=1e-4, kdim=50)
    solver = spec.build(block_size=3)
"""

# The compiled ``solvers`` extension is only built in the open-source
# configuration (``OPENDARTS_LINEAR_SOLVERS`` defined / no ``-b`` passed to
# the build script). A proprietary build that links against the prebuilt
# ``darts-linear-solvers`` library does not produce ``solvers.so``, and the
# C++ registry / MGR API are not available. In that case we expose only the
# Python-side helpers (specs, adaptive policy, enums, python_solvers) so
# ``from darts.solvers import SuperLUSolverSpec`` still works.
try:
    from . import solvers  # noqa: F401
    from .solvers import *  # noqa: F401,F403

    _have_compiled_solvers = True
except ImportError:
    solvers = None  # noqa: F811
    _have_compiled_solvers = False

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
from .python_solvers import (  # noqa: F401
    PardisoSolver,
    PETScSolver,
    PythonLinearSolver,
)
from .specs import (  # noqa: F401
    CPRSolverSpec,
    GMRESSolverSpec,
    LinearSolverSpec,
    MGRLevelSpec,
    MGRSolverSpec,
    PardisoSolverSpec,
    PETScSolverSpec,
    PythonLinearSolverSpec,
    SuperLUSolverSpec,
    default_linear_solver,
)
