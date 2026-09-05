"""open-DARTS linear solvers.

This package is the single source of input parameters for the linear solve and
the home of the Python-side linear solver classes, mirroring the design of
:mod:`darts.nonlinear_solvers` (!327):

* the runtime :class:`~darts.linear_solvers.solver.LinearSolver` -- composed onto
  every ``DartsModel`` as ``self.linear_solver`` (created once in
  ``DartsModel.__init__``, never reassigned), it owns the declarative :attr:`spec`
  as well as every method that binds a model to its linear solver: backend
  materialization at ``reset()``/``init()`` time (``handle`` / ``python_solver``),
  mid-run reconfiguration (``update_solver``), adaptive per-timestep switching, and
  the deprecated ``set_sim_params`` family;
* the declarative :mod:`~darts.linear_solvers.specs` configuration classes
  (:class:`~darts.linear_solvers.specs.MGRSolverSpec`, ``SuperLUSolverSpec``, ...),
  retrievable as ``linear_solver.spec`` (serializable via ``spec.to_dict()``);
* the compiled ``linear_solvers`` extension -- the solver registry, the MGR API and
  the C++ configuration objects (flattened into the ``darts.linear_solvers``
  namespace so ``from darts import linear_solvers; linear_solvers.create_linear_solver(...)``
  works);
* the HYPRE :mod:`~darts.linear_solvers.enums` integer-code enumerations.

Model use (in ``set_solver()``, the shared hook of the linear and nonlinear solver)::

    def set_solver(self):
        self.linear_solver.spec = MGRSolverSpec(tolerance=1e-4, kdim=50)
        # or tune the platform default (mirror of the nonlinear form):
        super().set_solver()
        self.linear_solver.spec.tolerance = 1e-6

Standalone use (outside a model)::

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

# The runtime LinearSolver wrapper (the instance DartsModel.linear_solver holds,
# mirror of darts.nonlinear_solvers.NonlinearSolver). Imported after the compiled
# star-import above, so it deliberately shadows the compiled raw-handle class of
# the same name in this package namespace; the raw pybind class stays reachable
# as darts.linear_solvers.linear_solvers.LinearSolver (alias
# LinearSolverInterface) and as the type of LinearSolver.handle.
from .solver import LinearSolver  # noqa: F401,E402
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
    SchurEliminationSpec,
    SuperLUSolverSpec,
)
