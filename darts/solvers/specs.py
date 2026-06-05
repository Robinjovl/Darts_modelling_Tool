"""Python configuration classes for open-DARTS linear solvers.

Each :class:`LinearSolverSpec` subclass describes one solver and all of its
parameters as typed, documented attributes. ``spec.build(block_size)`` produces
a configured C++ solver through the solver registry -- this is the single,
enum-free way to select and configure a linear solver from Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

# The compiled ``solvers`` extension is absent in proprietary builds that
# link the prebuilt ``darts-linear-solvers`` library; the spec classes
# below still import (only their ``build()`` calls will fail at runtime).
try:
    from . import solvers
except ImportError:
    solvers = None  # type: ignore[assignment]
from .enums import CoarseGrid, FRelaxation, GlobalSmoother, Interpolation, Restriction


@dataclass
class LinearSolverSpec:
    """Base linear-solver specification.

    Subclasses set the class attribute :attr:`registry_name` (the name the
    solver is registered under in C++) and, if they have extra parameters,
    override :meth:`_make_config`.

    ``tolerance`` and ``max_iterations`` are honoured by the Python-resident
    solvers (PETSc / Pardiso). For engine-resident solvers (MGR / GMRES / CPR /
    SuperLU) the engine's ``linear_solver->init()`` call subsequently overrides
    them with ``data_ts.linear_tol`` / ``data_ts.linear_max_iter`` -- those are
    the authoritative knobs on the Newton loop.

    ``print_level`` controls Python-resident solver verbosity. Engine-resident
    solvers expose their own verbosity knob through the solver-specific spec
    (e.g. :attr:`MGRSolverSpec.log_level`).
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
        BoomerAMG is configured with tol=0 (preconditioner stage); the sweep
        budget is the only AMG knob -- the outer Krylov drives convergence.
    :param ilu_fill_level: full-system ILU(k) fill level (currently 0).
    """

    registry_name: ClassVar[str] = "cpr"

    amg_max_iters: int = 2
    ilu_fill_level: int = 0

    def _make_config(self) -> solvers.CPRSolverConfig:
        config = solvers.CPRSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.amg_max_iters = self.amg_max_iters
        config.ilu_fill_level = self.ilu_fill_level
        return config


@dataclass
class FSCPRSolverSpec(LinearSolverSpec):
    """Open-source FS-CPR (Full-System CPR) poromechanics preconditioner.

    Two-stage poromechanics CPR: HYPRE BoomerAMG correction on the displacement
    (U) subsystem followed by a second BoomerAMG correction on the flow /
    pressure (P or PPSS) subsystem driven by a Schur-complement approximation.
    The in-tree replacement for the proprietary ``linsolv_bos_fs_cpr``. Block
    size 4..8 (ND = 3 hard-coded; NE = block_size - 3 in 1..5).

    .. note::
       The ``(n_res, n_fracs, n_wells)`` partition is normally set by the
       engine at construction time -- setting it on the spec is a manual
       override. ``n_fracs > 0`` is **not yet supported** (FS_UPG / gap
       subsystem path is pending).

    .. note::
       Variable-layout overrides (``p_var`` / ``z_var`` / ``u_var`` / ``nc``)
       let one spec drive either engine. Defaulting them to ``None`` keeps
       the ``engine_super_elastic_cpu`` convention
       (``P_VAR = 0, Z_VAR = 1, U_VAR = NE, NC = NE``). For
       ``engine_pm_cpu`` set them explicitly:
       ``p_var = 3, u_var = 0, z_var = 255, nc = 1``.

    :param force_amg_asymmetric: workaround for HYPRE BoomerAMG's
        symmetric-detection heuristic; doubles the first row of each scalar
        subsystem extraction. Default ``True`` matches the proprietary code.
    :param n_res: number of reservoir (volumetric) cells in the partition;
        ``0`` keeps whatever the engine sets later via ``set_block_sizes``.
    :param n_fracs: number of fracture cells; **must be 0** (FS_UPG pending).
    :param n_wells: number of well blocks in the partition.
    :param u_amg_max_iters: U-block BoomerAMG V-cycle budget per FS-CPR apply.
    :param p_amg_max_iters: PPSS BoomerAMG V-cycle budget per FS-CPR apply.
    :param p_var: explicit pressure-variable block index; ``None`` ->
        engine_super_elastic_cpu default (``0``).
    :param z_var: explicit composition-variable block index; ``None`` ->
        engine_super_elastic_cpu default (``1``). For engine_pm_cpu pass
        ``255`` (the "no composition" sentinel).
    :param u_var: explicit displacement-block start index; ``None`` ->
        engine_super_elastic_cpu default (``NE``). engine_pm_cpu uses ``0``.
    :param nc: explicit composition count (``NE - THERMAL``); ``None`` ->
        engine_super_elastic_cpu default (``NE``). engine_pm_cpu uses ``1``.
    """

    registry_name: ClassVar[str] = "fs_cpr"

    force_amg_asymmetric: bool = True
    n_res: int = 0
    n_fracs: int = 0
    n_wells: int = 0
    u_amg_max_iters: int = 1
    p_amg_max_iters: int = 1
    p_var: int | None = None
    z_var: int | None = None
    u_var: int | None = None
    nc: int | None = None

    def _make_config(self) -> solvers.FSCPRSolverConfig:
        config = solvers.FSCPRSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.force_amg_asymmetric = self.force_amg_asymmetric
        config.n_res = self.n_res
        config.n_fracs = self.n_fracs
        config.n_wells = self.n_wells
        config.u_amg_max_iters = self.u_amg_max_iters
        config.p_amg_max_iters = self.p_amg_max_iters
        # None -> -1 sentinel on the C++ side -> fall back to the
        # engine_super_elastic_cpu convention default derived from block_size
        # inside the factory.
        config.p_var = -1 if self.p_var is None else self.p_var
        config.z_var = -1 if self.z_var is None else self.z_var
        config.u_var = -1 if self.u_var is None else self.u_var
        config.nc = -1 if self.nc is None else self.nc
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
