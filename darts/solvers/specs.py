"""Python configuration classes for open-DARTS linear solvers.

Each :class:`LinearSolverSpec` subclass describes one solver and all of its
parameters as typed, documented attributes. ``spec.build(block_size)`` produces
a configured C++ solver through the solver registry -- this is the single,
enum-free way to select and configure a linear solver from Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from . import solvers
from .enums import CoarseGrid, FRelaxation, GlobalSmoother, Interpolation, Restriction


@dataclass
class LinearSolverSpec:
    """Base linear-solver specification.

    Subclasses set the class attribute :attr:`registry_name` (the name the
    solver is registered under in C++) and, if they have extra parameters,
    override :meth:`_make_config`.
    """

    tolerance: float = 1e-5
    max_iterations: int = 50
    print_level: int = 0

    #: Name the solver is registered under in the C++ solver registry.
    registry_name: ClassVar[str] = ""

    def _make_config(self) -> solvers.SolverConfig:
        """Build the C++ configuration object for this spec."""
        config = solvers.SolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.print_level = self.print_level
        return config

    def build(self, block_size: int):
        """Create the configured C++ linear solver for the given block size.

        :param block_size: number of equations per cell (matrix block size).
        :returns: a ``solvers.LinearSolver`` handle.
        """
        if not self.registry_name:
            raise NotImplementedError(
                f"{type(self).__name__} does not define a registry_name"
            )
        return solvers.create_linear_solver(
            self.registry_name, self._make_config(), block_size
        )


@dataclass
class MGRLevelSpec:
    """Configuration of one HYPRE MGR reduction level.

    The integer fields accept the :mod:`darts.solvers.enums` ``IntEnum`` values
    (e.g. ``FRelaxation.DIRECT_INVERSE``) or plain ints.
    """

    keep_labels: list[int] = field(default_factory=list)
    frelax_type: int = FRelaxation.NONE
    frelax_iters: int = 0
    interp_type: int = Interpolation.INJECTION
    restrict_type: int = Restriction.INJECTION
    coarse_method: int = CoarseGrid.GALERKIN
    smoother_type: int = GlobalSmoother.NONE
    smoother_iters: int = 0

    def _to_cpp(self) -> solvers.MGRLevelConfig:
        """Convert to the C++ ``MGRLevelConfig``."""
        cpp = solvers.MGRLevelConfig()
        cpp.keep_labels = [int(label) for label in self.keep_labels]
        cpp.frelax_type = int(self.frelax_type)
        cpp.frelax_iters = int(self.frelax_iters)
        cpp.interp_type = int(self.interp_type)
        cpp.restrict_type = int(self.restrict_type)
        cpp.coarse_method = int(self.coarse_method)
        cpp.smoother_type = int(self.smoother_type)
        cpp.smoother_iters = int(self.smoother_iters)
        return cpp


@dataclass
class MGRSolverSpec(LinearSolverSpec):
    """HYPRE MGR solver -- the default open-source CPU solver.

    The scalar defaults match the C++ solver's own defaults. The optional
    ``*_level`` members override individual reduction levels only when set;
    leaving them ``None`` keeps the solver's built-in level defaults.
    """

    registry_name: ClassVar[str] = "mgr"

    kdim: int = 30
    use_mgr: bool = True
    log_level: int = 1
    use_physics_scaling: bool = True
    use_flex_gmres: bool = True
    n_reservoir_blocks: int = 0
    enable_well_level: bool = False
    enable_composition_level: bool = False
    reservoir_variable_roles: list[int] = field(default_factory=list)
    well_variable_roles: list[int] = field(default_factory=list)
    well_strategy: int | None = None
    well_level: MGRLevelSpec | None = None
    composition_level: MGRLevelSpec | None = None
    pressure_level: MGRLevelSpec | None = None
    custom_levels: list[MGRLevelSpec] = field(default_factory=list)

    def _make_config(self) -> solvers.MGRSolverConfig:
        config = solvers.MGRSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.print_level = self.print_level
        config.kdim = self.kdim
        config.use_mgr = self.use_mgr
        config.log_level = self.log_level
        config.use_physics_scaling = self.use_physics_scaling
        config.use_flex_gmres = self.use_flex_gmres
        config.n_reservoir_blocks = self.n_reservoir_blocks
        config.enable_well_level = self.enable_well_level
        config.enable_composition_level = self.enable_composition_level
        config.reservoir_variable_roles = [
            int(role) for role in self.reservoir_variable_roles
        ]
        config.well_variable_roles = [int(role) for role in self.well_variable_roles]
        config.well_strategy = (
            None if self.well_strategy is None else int(self.well_strategy)
        )
        config.well_level = (
            None if self.well_level is None else self.well_level._to_cpp()
        )
        config.composition_level = (
            None if self.composition_level is None else self.composition_level._to_cpp()
        )
        config.pressure_level = (
            None if self.pressure_level is None else self.pressure_level._to_cpp()
        )
        config.custom_levels = [level._to_cpp() for level in self.custom_levels]
        return config


@dataclass
class SuperLUSolverSpec(LinearSolverSpec):
    """SuperLU sparse direct solver.

    A direct solver: it takes no parameters beyond the matrix, so the inherited
    ``tolerance`` / ``max_iterations`` fields are ignored. Robust but memory- and
    time-heavy on large systems -- intended for small problems and verification.
    """

    registry_name: ClassVar[str] = "superlu"


@dataclass
class GMRESSolverSpec(LinearSolverSpec):
    """Open-source restarted GMRES outer Krylov solver (``linsolv_gmres``).

    Right-preconditioned restart-GMRES with modified Gram-Schmidt and Givens
    rotations -- the in-tree replacement for the legacy ``linsolv_bos_gmres``.
    Combine it with any other :class:`LinearSolverSpec` (MGR, HYPRE AMG / ILU,
    SuperLU, ...) via :attr:`prec` to build a full preconditioned solver.

    :param restart: Krylov subspace dimension (restart length).
    :param prec: inner preconditioner spec (any :class:`LinearSolverSpec`);
        ``None`` runs unpreconditioned GMRES.
    """

    registry_name: ClassVar[str] = "gmres"

    restart: int = 30
    prec: LinearSolverSpec | None = None

    def _make_config(self) -> solvers.GMRESSolverConfig:
        config = solvers.GMRESSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.print_level = self.print_level
        config.restart = self.restart
        return config

    def build(self, block_size: int):
        """Build the GMRES solver, attach the inner preconditioner if any.

        The inner solver is also stored on this spec so it outlives any local
        reference at the call site -- the C++ GMRES holds a raw pointer to it.
        """
        gmres = solvers.create_linear_solver(
            self.registry_name, self._make_config(), block_size
        )
        if self.prec is not None:
            self._built_prec = self.prec.build(block_size)
            gmres.set_prec(self._built_prec)
        return gmres


@dataclass
class CPRSolverSpec(LinearSolverSpec):
    """Open-source CPR (Constrained Pressure Residual) two-stage preconditioner.

    The in-tree replacement for the proprietary ``linsolv_bos_cpr``: HYPRE
    BoomerAMG on the scalar pressure subsystem followed by HYPRE ILU(0) on the
    full system. Intended as the inner preconditioner of an outer Krylov
    solver, e.g.::

        GMRESSolverSpec(prec=CPRSolverSpec(), tolerance=1e-5)

    Transposed apply (CPRA, Han et al. 2013) is selected automatically when
    the outer GMRES calls ``solve_transposed`` for the adjoint Newton step.

    :param amg_max_iters: AMG sweeps on the pressure subsystem per CPR apply.
    :param amg_tolerance: AMG inner tolerance (relaxed -- AMG is used as a prec).
    :param ilu_fill_level: full-system ILU(k) fill level (currently 0).
    """

    registry_name: ClassVar[str] = "cpr"

    amg_max_iters: int = 2
    amg_tolerance: float = 1e-2
    ilu_fill_level: int = 0

    def _make_config(self) -> solvers.CPRSolverConfig:
        config = solvers.CPRSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.print_level = self.print_level
        config.amg_max_iters = self.amg_max_iters
        config.amg_tolerance = self.amg_tolerance
        config.ilu_fill_level = self.ilu_fill_level
        return config


@dataclass
class PythonLinearSolverSpec(LinearSolverSpec):
    """Base spec for the Python-resident solvers (PETSc / Pardiso).

    Unlike the engine-resident specs, :meth:`build` returns a
    :class:`~darts.solvers.python_solvers.PythonLinearSolver` -- a stateful
    object that runs the solve in the Python process and is owned by the model,
    not injected into the C++ engine. See ``SOLVER_REFACTORING_PLAN.md``
    section 13.
    """

    def build(self, block_size: int):  # noqa: D102 -- see subclasses
        raise NotImplementedError(f"{type(self).__name__} must override build()")


@dataclass
class PETScSolverSpec(PythonLinearSolverSpec):
    """PETSc (``petsc4py``) Krylov solver.

    Builds the system matrix as a PETSc ``BAIJ`` matrix directly from the
    engine's block-CSR Jacobian -- no block->scalar expansion.

    :param variant: ``"cpr"`` -- CPR preconditioner for flow; ``"fs"`` --
        fixed-stress fieldsplit for poromechanics.
    """

    variant: str = "cpr"

    def build(self, block_size: int):
        """Return a configured :class:`PETScSolver`."""
        from .python_solvers import PETScSolver

        return PETScSolver(
            variant=self.variant,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            print_level=self.print_level,
        )


@dataclass
class PardisoSolverSpec(PythonLinearSolverSpec):
    """Pardiso (``pypardiso`` / Intel MKL) sparse direct solver.

    The block-CSR Jacobian is expanded to scalar CSR once; per Newton iteration
    only the values are gathered into the scalar layout.
    """

    def build(self, block_size: int):
        """Return a configured :class:`PardisoSolver`."""
        from .python_solvers import PardisoSolver

        return PardisoSolver(
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            print_level=self.print_level,
        )


def default_linear_solver(platform: str = "cpu") -> LinearSolverSpec:
    """Return the default linear-solver spec for a platform.

    Single source of truth for the default solver. The CPU default is
    **FGMRES + open-source CPR** -- the in-tree restart-GMRES wrapped around
    the two-stage CPR preconditioner (HYPRE BoomerAMG on the pressure
    subsystem + HYPRE_ILU(0) on the full system). This is the open-source
    equivalent of the legacy ``linsolv_bos_gmres + linsolv_bos_cpr_amg`` stack
    that the proprietary build used as its default. Validated against MGR
    across ``2ph_comp``, ``2ph_do``, ``2ph_geothermal``, and ``3ph_bo``: same
    Newton / linear iteration counts as MGR, with lower per-iteration setup
    overhead.

    :class:`MGRSolverSpec` remains the recommended fallback for matrices
    where the CPR pressure extraction is a bad fit; set
    ``data_ts.linear_solver = MGRSolverSpec()`` to use it.

    :param platform: ``"cpu"`` or ``"gpu"``.

    .. note::
       The GPU default -- the open-source GPU BiCGStab + cuSPARSE-ILU outer
       solver -- is wired directly in the GPU engine factory (``engine_base_gpu``);
       it does not flow through a ``LinearSolverSpec`` and ``data_ts.linear_solver``
       is unused on GPU. Requesting the GPU default through this function
       therefore raises :class:`NotImplementedError`.
    """
    if platform.lower() == "gpu":
        raise NotImplementedError(
            "GPU default solver is configured in the GPU engine factory, not "
            "through a LinearSolverSpec (see engine_base_gpu)."
        )
    return GMRESSolverSpec(restart=50, prec=CPRSolverSpec())
