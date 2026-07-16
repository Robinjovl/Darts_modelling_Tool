"""open-DARTS linear solvers.

This package exposes:

* the compiled ``linear_solvers`` extension -- the solver registry, the MGR API and
  the C++ configuration objects (flattened into the ``darts.linear_solvers``
  namespace so ``from darts import linear_solvers; linear_solvers.create_linear_solver(...)``
  works);
* the Python :mod:`~darts.linear_solvers.specs` configuration classes
  (:class:`~darts.linear_solvers.specs.MGRSolverSpec`, ``SuperLUSolverSpec``, ...);
* the HYPRE :mod:`~darts.linear_solvers.enums` integer-code enumerations.

Typical use::

    from darts.linear_solvers import MGRSolverSpec
    spec = MGRSolverSpec(tolerance=1e-4, kdim=50)
    solver = spec.build(block_size=3)
"""

# The compiled ``linear_solvers`` extension is only built in the open-source
# configuration (``OPENDARTS_LINEAR_SOLVERS`` defined / no ``-b`` passed to
# the build script). A proprietary build that links against the prebuilt
# ``darts-linear-solvers`` library does not produce ``linear_solvers.so``, and the
# C++ registry / MGR API are not available. In that case we expose only the
# Python-side helpers (specs, adaptive policy, enums, python_solvers) so
# ``from darts.linear_solvers import SuperLUSolverSpec`` still works.
try:
    from . import linear_solvers  # noqa: F401
    from .linear_solvers import *  # noqa: F401,F403

    _have_compiled_solvers = True
except ImportError:
    linear_solvers = None  # noqa: F811
    _have_compiled_solvers = False

from .adaptive import (  # noqa: F401
    AdaptiveSolverSpec,
    SolverAction,
    SolverSwitchContext,
    fallback_on_failure,
)
from .enums import (  # noqa: F401
    BCSRCPRReduction,
    CoarseGrid,
    CompositeMode,
    FRelaxation,
    GlobalSmoother,
    Interpolation,
    LocalFallback,
    LocalPreconditioner,
    Restriction,
    ScalingType,
    VariableRole,
)
from .python_solvers import (  # noqa: F401
    PardisoSolver,
    PETScSolver,
    PythonLinearSolver,
)
from .specs import (  # noqa: F401
    AMGXCPRSolverSpec,
    BCSRCPRSpec,
    BILU0Spec,
    CPRSolverSpec,
    CuDSSSolverSpec,
    FSCPRSolverSpec,
    GMRESSolverSpec,
    GPUBiCGStabCPRSolverSpec,
    GPUCuSolverSpec,
    GPUGMRESILU0SolverSpec,
    GPUSolverSpec,
    LinearSolverSpec,
    LocalCorrectionSpec,
    MGRLevelSpec,
    MGRSolverSpec,
    PardisoSolverSpec,
    PETScSolverSpec,
    PressureAMGSpec,
    PythonLinearSolverSpec,
    SuperLUSolverSpec,
    default_linear_solver,
)
