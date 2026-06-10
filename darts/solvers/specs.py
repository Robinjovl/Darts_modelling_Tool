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
from .enums import (
    BCSRCPRReduction,
    CoarseGrid,
    FRelaxation,
    GlobalSmoother,
    Interpolation,
    LocalFallback,
    Restriction,
)


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
class BILU0Spec:
    """Block-ILU(0) local-solver options for the MGR F-relaxation stage.

    Mirrors the C++ ``mgr_bilu0_config``; defaults match ``linsolv_mgr``'s own.
    """

    pivot_shift: float = 1.0e-12
    fallback_strategy: int = LocalFallback.IDENTITY
    fallback_diagonal_tolerance: float = 1.0e-4
    fallback_shifted_max: float = 1.0e-4
    fallback_shifted_growth: float = 100.0

    def _to_cpp(self) -> solvers.MGRBILU0Config:
        cpp = solvers.MGRBILU0Config()
        cpp.pivot_shift = float(self.pivot_shift)
        cpp.fallback_strategy = int(self.fallback_strategy)
        cpp.fallback_diagonal_tolerance = float(self.fallback_diagonal_tolerance)
        cpp.fallback_shifted_max = float(self.fallback_shifted_max)
        cpp.fallback_shifted_growth = float(self.fallback_shifted_growth)
        return cpp


@dataclass
class LocalCorrectionSpec:
    """MGR local-correction (pressure-block damping) options.

    Mirrors the C++ ``mgr_local_correction_config``.
    """

    alpha: float = 1.0
    adaptive_fallback_threshold: float = -1.0
    adaptive_alpha: float = 0.0
    adaptive_fallback_threshold_high: float = -1.0
    adaptive_alpha_high: float = 0.0
    quality_enabled: bool = False
    quality_min_alpha: float = 0.0

    def _to_cpp(self) -> solvers.MGRLocalCorrectionConfig:
        cpp = solvers.MGRLocalCorrectionConfig()
        cpp.alpha = float(self.alpha)
        cpp.adaptive_fallback_threshold = float(self.adaptive_fallback_threshold)
        cpp.adaptive_alpha = float(self.adaptive_alpha)
        cpp.adaptive_fallback_threshold_high = float(
            self.adaptive_fallback_threshold_high
        )
        cpp.adaptive_alpha_high = float(self.adaptive_alpha_high)
        cpp.quality_enabled = bool(self.quality_enabled)
        cpp.quality_min_alpha = float(self.quality_min_alpha)
        return cpp


@dataclass
class BCSRCPRSpec:
    """BCSR-CPR (block-CSR Constrained Pressure Residual) options.

    Attaching this to :attr:`MGRSolverSpec.bcsr_cpr` enables BCSR-CPR. Mirrors
    the C++ ``mgr_bcsr_cpr_config``. ``transpose_apply`` / ``forward_source``
    default to ``None`` (left to ``linsolv_mgr``'s auto-derivation).
    """

    reduction_type: int = BCSRCPRReduction.TRUE_IMPES
    pressure_variable: int = 0
    weight_max: float = 1.0e6
    reuse_amg_hierarchy: bool = False
    amg_rebuild_interval: int = 1
    adaptive_amg_rebuild: bool = False
    adaptive_li_threshold: int = 80
    adaptive_li_growth_factor: float = 2.0
    adaptive_min_reuse_setups: int = 1
    adaptive_max_reuse_setups: int = 0
    adaptive_pressure_overshoot_threshold: float = -1.0
    adaptive_final_proxy_threshold: float = -1.0
    adaptive_fallback_threshold: float = -1.0
    diagnostics: bool = False
    diagnostic_apply_interval: int = 0
    diagnostic_matrix_interval: int = 0
    pressure_correction_alpha: float = 1.0
    pressure_correction_guard_threshold: float = -1.0
    pressure_correction_guard_min_alpha: float = 0.0
    transpose_apply: bool | None = None
    forward_source: bool | None = None

    def _to_cpp(self) -> solvers.MGRBCSRCPRConfig:
        cpp = solvers.MGRBCSRCPRConfig()
        cpp.reduction_type = int(self.reduction_type)
        cpp.pressure_variable = int(self.pressure_variable)
        cpp.weight_max = float(self.weight_max)
        cpp.reuse_amg_hierarchy = bool(self.reuse_amg_hierarchy)
        cpp.amg_rebuild_interval = int(self.amg_rebuild_interval)
        cpp.adaptive_amg_rebuild = bool(self.adaptive_amg_rebuild)
        cpp.adaptive_li_threshold = int(self.adaptive_li_threshold)
        cpp.adaptive_li_growth_factor = float(self.adaptive_li_growth_factor)
        cpp.adaptive_min_reuse_setups = int(self.adaptive_min_reuse_setups)
        cpp.adaptive_max_reuse_setups = int(self.adaptive_max_reuse_setups)
        cpp.adaptive_pressure_overshoot_threshold = float(
            self.adaptive_pressure_overshoot_threshold
        )
        cpp.adaptive_final_proxy_threshold = float(self.adaptive_final_proxy_threshold)
        cpp.adaptive_fallback_threshold = float(self.adaptive_fallback_threshold)
        cpp.diagnostics = bool(self.diagnostics)
        cpp.diagnostic_apply_interval = int(self.diagnostic_apply_interval)
        cpp.diagnostic_matrix_interval = int(self.diagnostic_matrix_interval)
        cpp.pressure_correction_alpha = float(self.pressure_correction_alpha)
        cpp.pressure_correction_guard_threshold = float(
            self.pressure_correction_guard_threshold
        )
        cpp.pressure_correction_guard_min_alpha = float(
            self.pressure_correction_guard_min_alpha
        )
        cpp.transpose_apply = (
            None if self.transpose_apply is None else bool(self.transpose_apply)
        )
        cpp.forward_source = (
            None if self.forward_source is None else bool(self.forward_source)
        )
        return cpp


@dataclass
class PressureAMGSpec:
    """MGR pressure-subsystem BoomerAMG options.

    Mirrors the C++ ``mgr_pressure_amg_config`` (the three
    ``set_mgr_pressure_amg_*`` setter groups).
    """

    coarsen_type: int = 6
    interp_type: int = 6
    relax_type: int = 6
    agg_num_levels: int = 1
    agg_interp_type: int = 6
    agg_pmax_elmts: int = 20
    relax_order: int = 1
    strong_threshold: float = 0.5
    trunc_factor: float = -1.0
    pmax_elmts: int = -1
    max_levels: int = 0
    solve_max_iter: int = 1
    solve_tolerance: float = 0.0

    def _to_cpp(self) -> solvers.MGRPressureAMGConfig:
        cpp = solvers.MGRPressureAMGConfig()
        cpp.coarsen_type = int(self.coarsen_type)
        cpp.interp_type = int(self.interp_type)
        cpp.relax_type = int(self.relax_type)
        cpp.agg_num_levels = int(self.agg_num_levels)
        cpp.agg_interp_type = int(self.agg_interp_type)
        cpp.agg_pmax_elmts = int(self.agg_pmax_elmts)
        cpp.relax_order = int(self.relax_order)
        cpp.strong_threshold = float(self.strong_threshold)
        cpp.trunc_factor = float(self.trunc_factor)
        cpp.pmax_elmts = int(self.pmax_elmts)
        cpp.max_levels = int(self.max_levels)
        cpp.solve_max_iter = int(self.solve_max_iter)
        cpp.solve_tolerance = float(self.solve_tolerance)
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

    # Composite-preconditioner / local-solver knobs. Each is applied only when
    # set (None / unset), so a bare MGRSolverSpec keeps linsolv_mgr's built-in
    # behaviour; populating them reproduces the composite-model raw build.
    scaling_type: int | None = None
    composite_mode: int | None = None
    local_solver: int | None = None
    use_bcsr_cpr: bool | None = None
    bilu0: BILU0Spec | None = None
    local_correction: LocalCorrectionSpec | None = None
    bcsr_cpr: BCSRCPRSpec | None = None
    pressure_amg: PressureAMGSpec | None = None

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
        config.scaling_type = (
            None if self.scaling_type is None else int(self.scaling_type)
        )
        config.composite_mode = (
            None if self.composite_mode is None else int(self.composite_mode)
        )
        config.local_solver = (
            None if self.local_solver is None else int(self.local_solver)
        )
        config.use_bcsr_cpr = (
            None if self.use_bcsr_cpr is None else bool(self.use_bcsr_cpr)
        )
        config.bilu0 = None if self.bilu0 is None else self.bilu0._to_cpp()
        config.local_correction = (
            None if self.local_correction is None else self.local_correction._to_cpp()
        )
        config.bcsr_cpr = None if self.bcsr_cpr is None else self.bcsr_cpr._to_cpp()
        config.pressure_amg = (
            None if self.pressure_amg is None else self.pressure_amg._to_cpp()
        )
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


@dataclass
class GPUSolverSpec(LinearSolverSpec):
    """Base spec for GPU linear solvers.

    GPU solvers are selected by the GPU engine factory through the
    ``params.linear_type`` (``darts.engines.linear_solver_t``) enum, NOT through
    the open-source ``darts.solvers`` registry. A GPUSolverSpec therefore does not
    build a C++ solver -- it names the enum value via :attr:`linear_type_name`, and
    :meth:`darts.models.darts_model.DartsModel._apply_solver` translates
    ``self.solver`` to ``params.linear_type`` on the GPU platform. :meth:`build`
    raises.

    This keeps ``self.solver`` the single user-facing API on GPU too:
    ``self.solver = AMGXCPRSolverSpec()`` selects the GPU solver, mirroring the way
    a CPU spec selects a registry solver.
    """

    #: name of the ``darts.engines.sim_params`` ``linear_solver_t`` enum value
    linear_type_name: ClassVar[str] = ""

    def build(self, block_size: int):
        raise NotImplementedError(
            f"{type(self).__name__} is a GPU spec; it is selected via "
            f"params.linear_type ({self.linear_type_name or '<unset>'}) by the GPU "
            f"engine factory, not built through the open-source registry."
        )


@dataclass
class AMGXCPRSolverSpec(GPUSolverSpec):
    """GPU GMRES + AMGX-CPR -- the default GPU solver.

    NVIDIA AMGX algebraic multigrid on the pressure subsystem + ILU on the full
    system, wrapped in GMRES. Maps to ``linear_solver_t.gpu_gmres_cpr_amgx_ilu``.
    """

    linear_type_name: ClassVar[str] = "gpu_gmres_cpr_amgx_ilu"


@dataclass
class GPUBiCGStabCPRSolverSpec(GPUSolverSpec):
    """GPU BiCGStab + AMGX-CPR. Maps to ``linear_solver_t.gpu_bicgstab_cpr_amgx``."""

    linear_type_name: ClassVar[str] = "gpu_bicgstab_cpr_amgx"


@dataclass
class GPUGMRESILU0SolverSpec(GPUSolverSpec):
    """GPU GMRES + cuSPARSE-ILU(0) -- a single-stage GPU fallback (no AMG).

    Maps to ``linear_solver_t.gpu_gmres_ilu0``.
    """

    linear_type_name: ClassVar[str] = "gpu_gmres_ilu0"


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
    ``self.solver = MGRSolverSpec()`` in the model's ``set_solver()`` to use it.

    :param platform: ``"cpu"`` or ``"gpu"``.

    The GPU default is :class:`AMGXCPRSolverSpec` -- GMRES + AMGX-CPR (NVIDIA
    AMGX algebraic multigrid on the pressure subsystem + ILU on the full system),
    mapping to ``linear_solver_t.gpu_gmres_cpr_amgx_ilu``. A :class:`GPUSolverSpec`
    does not build a C++ solver; it names the ``params.linear_type`` enum, which
    :meth:`DartsModel._apply_solver` sets on the GPU platform and the GPU engine
    factory (``engine_base_gpu``) consumes. So ``self.solver`` is the single
    user-facing API on GPU too.

    :param platform: ``"cpu"`` or ``"gpu"``.
    """
    if platform.lower() == "gpu":
        return AMGXCPRSolverSpec()
    return GMRESSolverSpec(restart=50, prec=CPRSolverSpec())
