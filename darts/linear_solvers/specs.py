"""Python configuration classes for open-DARTS linear solvers.

Each :class:`LinearSolverSpec` subclass describes one solver and all of its
parameters as typed, documented attributes. ``spec.build(block_size)`` produces
a configured C++ solver through the solver registry -- this is the single,
enum-free way to select and configure a linear solver from Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

# The compiled ``linear_solvers`` extension is absent in proprietary builds that
# link the prebuilt ``darts-linear-solvers`` library; the spec classes
# below still import (only their ``build()`` calls will fail at runtime).
try:
    from . import linear_solvers
except ImportError:
    linear_solvers = None  # type: ignore[assignment]
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

    The spec is the **single owner of the linear-solver settings**: ``tolerance``,
    ``max_iterations`` and ``print_level`` are authoritative for every backend.
    :meth:`DartsModel._apply_solver` mirrors them into ``sim_params``
    (``tolerance_linear`` / ``max_i_linear`` / ``linear_print_level``) after
    ``set_solver()`` and before ``engine.init()``, so the engine re-applies exactly
    these values to whatever solver it (re-)inits -- engine-resident (MGR / GMRES /
    CPR / SuperLU) and Python-resident (PETSc / Pardiso) alike. A model therefore
    configures the linear solve in one place::

        def set_solver(self):
            super().set_solver()                                # platform default spec
            self.linear_solver.spec.tolerance = 1e-6                        # linear knobs
            self.linear_solver.spec.max_iterations = 40

    ``print_level`` is the generic verbosity knob; some engine-resident solvers also
    expose a solver-specific one (e.g. :attr:`MGRSolverSpec.log_level`).
    """

    tolerance: float = 1e-5
    max_iterations: int = 50
    print_level: int = 0

    #: ``darts.engines.linear_solver_t`` enum the *engine factory* should select
    #: on builds without the open-source registry (proprietary ``-a`` build), where
    #: :meth:`build` is unavailable. This is the cross-build fallback that lets a
    #: model declare its solver **only** through ``self.linear_solver`` -- the base
    #: ``_apply_solver`` writes it to ``params.linear_type`` on the proprietary path.
    #: ``None`` => leave ``params.linear_type`` as ``init()`` set it (e.g. the engine
    #: default / GPU default). Ignored on the open-source CPU build (the spec drives).
    proprietary_linear_type: int | None = None

    #: Name the solver is registered under in the C++ solver registry.
    registry_name: ClassVar[str] = ""

    def _make_config(self) -> linear_solvers.SolverConfig:
        """Build the C++ configuration object for this spec."""
        config = linear_solvers.SolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        return config

    def to_dict(self) -> dict:
        """Serialize this spec (and nested sub-specs) to a plain dict for input
        tracing, mirroring ``NonlinearSolverSpec.to_dict`` of
        :mod:`darts.nonlinear_solvers`. Delegates to ``model_dump()`` if the spec
        is later migrated to a Pydantic model.
        """
        dump = getattr(self, "model_dump", None)
        if callable(dump):  # Pydantic-forward-compatible
            return dump()
        from dataclasses import asdict

        return asdict(self)

    def build(self, block_size: int):
        """Create the configured C++ linear solver for the given block size.

        :param block_size: number of equations per cell (matrix block size).
        :returns: a ``linear_solvers.LinearSolver`` handle.
        """
        if not self.registry_name:
            raise NotImplementedError(
                f"{type(self).__name__} does not define a registry_name"
            )
        return linear_solvers.create_linear_solver(
            self.registry_name, self._make_config(), block_size
        )


@dataclass
class MGRLevelSpec:
    """Configuration of one HYPRE MGR reduction level.

    The integer fields accept the :mod:`darts.linear_solvers.enums` ``IntEnum`` values
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

    def _to_cpp(self) -> linear_solvers.MGRLevelConfig:
        """Convert to the C++ ``MGRLevelConfig``."""
        cpp = linear_solvers.MGRLevelConfig()
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

    def _to_cpp(self) -> linear_solvers.MGRBILU0Config:
        cpp = linear_solvers.MGRBILU0Config()
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

    def _to_cpp(self) -> linear_solvers.MGRLocalCorrectionConfig:
        cpp = linear_solvers.MGRLocalCorrectionConfig()
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

    def _to_cpp(self) -> linear_solvers.MGRBCSRCPRConfig:
        cpp = linear_solvers.MGRBCSRCPRConfig()
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

    def _to_cpp(self) -> linear_solvers.MGRPressureAMGConfig:
        cpp = linear_solvers.MGRPressureAMGConfig()
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

    def _make_config(self) -> linear_solvers.MGRSolverConfig:
        config = linear_solvers.MGRSolverConfig()
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

    def _make_config(self) -> linear_solvers.GMRESSolverConfig:
        config = linear_solvers.GMRESSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.restart = self.restart
        return config

    def build(self, block_size: int):
        """Build the GMRES solver, attach the inner preconditioner if any.

        The inner solver is also stored on this spec so it outlives any local
        reference at the call site -- the C++ GMRES holds a raw pointer to it.
        """
        if isinstance(self.prec, MGRSolverSpec):
            # MGR is a full iterative solver (it runs its own FlexGMRES /
            # MGR cycle to a tolerance), i.e. a *varying* operator. Wrapping
            # it inside this non-flexible outer GMRES is mathematically
            # unsound (the Krylov relations no longer hold) and is the known
            # root cause of the HYPRE NaN warnings on 2ph_comp. Use
            # MGRSolverSpec standalone instead -- it already embeds its own
            # (flexible) outer Krylov.
            raise ValueError(
                "GMRESSolverSpec(prec=MGRSolverSpec(...)) is not supported: "
                "MGR is an iterative solver, not a constant-operator "
                "preconditioner, and the composition diverges (HYPRE NaNs). "
                "Use MGRSolverSpec(...) directly as self.linear_solver instead."
            )
        gmres = linear_solvers.create_linear_solver(
            self.registry_name, self._make_config(), block_size
        )
        if self.prec is not None:
            self._built_prec = self.prec.build(block_size)
            gmres.set_prec(self._built_prec)
        return gmres


@dataclass
class CPRSolverSpec(LinearSolverSpec):
    """Open-source CPR (Constrained Pressure Residual) two-stage preconditioner.

    The in-tree replacement for the proprietary ``linsolv_bos_cpr``:
    column-sum True-IMPES pressure decoupling, HYPRE BoomerAMG on the scalar
    pressure subsystem (configured to mirror the proprietary BOS AMG by
    default), followed by an in-tree block ILU(0) on the full block system.
    Intended as the inner preconditioner of an outer Krylov solver, e.g.::

        GMRESSolverSpec(prec=CPRSolverSpec(), tolerance=1e-5)

    Transposed apply (CPRA, Han et al. 2013) is selected automatically when
    the outer GMRES calls ``solve_transposed`` for the adjoint Newton step;
    its hierarchies are built lazily on the first transposed solve (see
    ``eager_adjoint``).

    :param amg_max_iters: AMG V-cycles on the pressure subsystem per CPR
        apply. BoomerAMG is configured with tol=0 (preconditioner stage);
        1 V-cycle matches the proprietary stack.
    :param ilu_fill_level: scalar ILU(k) fill level (``stage2_type == 0``).
    :param weight_scheme: pressure-decoupling weights -- 1 = column-sum
        True-IMPES (BOS parity, default), 0 = diagonal-block only (previous
        behaviour).
    :param stage2_type: full-system smoothing stage -- 1 = in-tree block
        ILU(0) with dense NxN block inverses (BOS ``csr_ilu_prec`` parity,
        default), 0 = HYPRE scalar ILU(k) on the expanded system.
    :param eager_adjoint: build the CPRA transpose hierarchies on every
        setup from the start (previous behaviour) instead of lazily on the
        first transposed solve. Forward-only simulations should keep this
        off -- it roughly doubles the preconditioner setup cost.
    :param amg_coarsen_type: pressure-AMG knobs (HYPRE integer codes;
        negative keeps the HYPRE built-in default). Together the ``amg_*``
        defaults mirror the proprietary BOS AMG: PMIS coarsening, standard
        interpolation with no truncation, strong threshold 0.75, C/F-ordered
        hybrid Gauss-Seidel, Gaussian-elimination coarse solve at <= 100
        rows, no aggressive coarsening.
    :param reuse_amg_hierarchy: skip the BoomerAMG/ILU setup on subsequent
        Newton iterations (reuse the existing hierarchies on refreshed matrix
        values). OFF by default -- the per-Newton rebuild is the proven
        baseline; reuse halves the per-Newton CPR setup cost on
        well-converging runs.
    :param adaptive_amg_rebuild: with reuse on, force a fresh hierarchy after
        ``adaptive_consecutive_bad`` solves in a row exceeded
        ``adaptive_iter_threshold`` outer iterations (the outer GMRES feeds
        the count back after each solve).
    """

    registry_name: ClassVar[str] = "cpr"

    amg_max_iters: int = 1
    ilu_fill_level: int = 0
    weight_scheme: int = 1
    stage2_type: int = 1
    eager_adjoint: bool = False
    amg_coarsen_type: int = 8
    amg_interp_type: int = 8
    amg_relax_type: int = 3
    amg_relax_order: int = 1
    amg_num_sweeps: int = 1
    amg_strong_threshold: float = 0.75
    amg_agg_num_levels: int = 0
    amg_agg_interp_type: int = 6
    amg_agg_pmax_elmts: int = 20
    amg_pmax_elmts: int = 0
    amg_trunc_factor: float = 0.0
    amg_max_levels: int = -1
    amg_cycle_type: int = -1
    amg_max_coarse_size: int = 100
    amg_coarse_relax_type: int = 9
    amg_relax_wt: float = -1.0
    reuse_amg_hierarchy: bool = False
    adaptive_amg_rebuild: bool = False
    adaptive_iter_threshold: int = 15
    adaptive_consecutive_bad: int = 2

    def _make_config(self) -> linear_solvers.CPRSolverConfig:
        config = linear_solvers.CPRSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.amg_max_iters = self.amg_max_iters
        config.ilu_fill_level = self.ilu_fill_level
        config.weight_scheme = self.weight_scheme
        config.stage2_type = self.stage2_type
        config.eager_adjoint = self.eager_adjoint
        config.amg_coarsen_type = self.amg_coarsen_type
        config.amg_interp_type = self.amg_interp_type
        config.amg_relax_type = self.amg_relax_type
        config.amg_relax_order = self.amg_relax_order
        config.amg_num_sweeps = self.amg_num_sweeps
        config.amg_strong_threshold = self.amg_strong_threshold
        config.amg_agg_num_levels = self.amg_agg_num_levels
        config.amg_agg_interp_type = self.amg_agg_interp_type
        config.amg_agg_pmax_elmts = self.amg_agg_pmax_elmts
        config.amg_pmax_elmts = self.amg_pmax_elmts
        config.amg_trunc_factor = self.amg_trunc_factor
        config.amg_max_levels = self.amg_max_levels
        config.amg_cycle_type = self.amg_cycle_type
        config.amg_max_coarse_size = self.amg_max_coarse_size
        config.amg_coarse_relax_type = self.amg_coarse_relax_type
        config.amg_relax_wt = self.amg_relax_wt
        config.reuse_amg_hierarchy = self.reuse_amg_hierarchy
        config.adaptive_amg_rebuild = self.adaptive_amg_rebuild
        config.adaptive_iter_threshold = self.adaptive_iter_threshold
        config.adaptive_consecutive_bad = self.adaptive_consecutive_bad
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
    :param p_amg_max_iters: V-cycle budget for the flow (PPSS) stage's
        BoomerAMG per FS-CPR apply -- the stage-1 AMG of the nested CPR when
        ``NE > 1``.
    :param stage_growth_cap: divergence guard -- a stage apply whose output
        exceeds this multiple of its own input in max-norm (or is non-finite)
        fails the solve, so the nonlinear solver cuts the timestep instead of
        accepting a useless step. A BoomerAMG V-cycle is only a contraction on
        a near-M-matrix; on an operator that is not, it diverges *silently*
        (finite output, no HYPRE error). Default ``1e12`` leaves many orders of
        headroom over any healthy apply; non-positive disables the check.
    :param p_stage_type: flow (PPSS) stage when ``NE = N_VARS - 3 > 1``:
        ``2`` (default) runs a systems BoomerAMG on the block-diagonal-
        decoupled flow block; ``1`` nests a block CPR (true-IMPES-decoupled
        BoomerAMG + block ILU(0)), matching the proprietary FS-CPR; ``0``
        applies a systems BoomerAMG to the *raw* block, which is retained only
        as a diagnostic -- its point-wise smoother diverges on advection-
        dominated flow. Ignored when ``NE == 1``.
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
    p_stage_type: int = 2
    stage_growth_cap: float = 1.0e12
    p_var: int | None = None
    z_var: int | None = None
    u_var: int | None = None
    nc: int | None = None

    def _make_config(self) -> linear_solvers.FSCPRSolverConfig:
        config = linear_solvers.FSCPRSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.force_amg_asymmetric = self.force_amg_asymmetric
        config.n_res = self.n_res
        config.n_fracs = self.n_fracs
        config.n_wells = self.n_wells
        config.u_amg_max_iters = self.u_amg_max_iters
        config.p_amg_max_iters = self.p_amg_max_iters
        config.p_stage_type = self.p_stage_type
        config.stage_growth_cap = self.stage_growth_cap
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
    :class:`~darts.linear_solvers.python_solvers.PythonLinearSolver` -- a stateful
    object that runs the solve in the Python process and is owned by the model,
    not injected into the C++ engine. See ``SOLVER_REFACTORING_PLAN.md``
    section 13.
    """

    def build(self, block_size: int):  # noqa: D102 -- see subclasses
        raise NotImplementedError(f"{type(self).__name__} must override build()")


@dataclass
class PETScSolverSpec(PythonLinearSolverSpec):
    """PETSc (``petsc4py``) Krylov solver.

    Builds the system matrix as a scalar PETSc ``AIJ`` matrix, expanded from the
    engine's block-CSR Jacobian. The CPR / fixed-stress ``PCFIELDSPLIT``
    preconditioners split *within* each cell block (pressure vs transport /
    displacement), which a block ``BAIJ`` matrix cannot express; the structural
    block->scalar expansion is computed once and each Newton iteration only
    gathers the current block values into the scalar CSR (see
    ``_ScalarCSRExpander`` in ``python_solvers.py``).

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
class SchurEliminationSpec(LinearSolverSpec):
    """Exact local (block-Schur) elimination wrapping an inner solver
    (``linsolv_schur_elim``).

    A physics-agnostic transform for any block system containing equations with
    a purely LOCAL stencil -- equations whose Jacobian row is nonzero only in the
    cell's own diagonal block (no coupling to neighbour cells). This wrapper
    eliminates K such equation/unknown pairs per cell by exact static
    condensation -- the reduced system is K variables smaller per cell with the
    SAME sparsity -- and runs :attr:`inner` on it. The elimination is exact:
    outer Newton behaviour is unchanged up to linear-solver tolerance. It both
    shrinks all linear-solver work (vectors, SpMV, ILU blocks) and moves the
    retained (neighbour-coupled) equations to the front, which repairs the CPR
    True-IMPES pressure decoupling that a local-equation-first ordering degrades.

    The motivating example is reactive-transport chemistry: each mineral balance
    is exactly such a local equation (a precipitated solid does not flow, so its
    balance has no inter-cell flux term). But the same machinery serves any model
    whose equations exhibit a diagonal-block-only stencil.

    The eliminated (equation row, unknown column) pairs are supplied
    **explicitly** -- there is no built-in "row 0 / column 1" assumption. E.g. for
    a chemistry model with the ``K`` solids ordered first, the natural choice is
    ``elim_rows=list(range(K))`` (the local mineral-balance equations) and
    ``elim_cols=list(range(1, K+1))`` (their unknowns). Pressure (column 0) must
    stay kept so a CPR-type inner keeps its pressure subsystem.

    Usage (K eliminated pairs, block size N)::

        spec = SchurEliminationSpec(
            inner=GMRESSolverSpec(prec=CPRSolverSpec()),
            elim_rows=list(range(K)), elim_cols=list(range(1, K + 1)),
        )
        spec.tolerance = 1e-6             # set on the WRAPPER: the engine mirrors the
        spec.max_iterations = 500         # top-level spec into sim_params and passes it
        self.linear_solver.spec = spec    # down to the inner solver at init

    Tolerance/max-iteration semantics: the top-level (wrapper) spec is the
    single owner -- ``_sync_solver_to_sim_params`` mirrors ITS values into
    ``sim_params`` and the wrapper's ``init`` forwards them to the inner
    solver, overwriting anything set on the inner spec directly.

    The inner spec is built ``K`` block sizes smaller (N-K) automatically.

    :param inner: spec of the solver to run on the reduced system (required).
    :param elim_rows: preferred eliminated equation rows (length K);
        ``elim_rows[k]`` pairs with ``elim_cols[k]``. Cells where the pairing is
        singular (e.g. well heads) fall back to an invertible alternative row set
        automatically.
    :param elim_cols: eliminated unknown columns (length K), eliminated globally.
    :param pivot_eps: pivots at or below this magnitude disqualify a candidate
        during detection and fail setup on a later Newton iteration.
    """

    registry_name: ClassVar[str] = "schur_elim"

    inner: LinearSolverSpec | None = None
    elim_rows: list[int] | None = None
    elim_cols: list[int] | None = None
    pivot_eps: float = 0.0

    def _make_config(self) -> linear_solvers.SchurElimSolverConfig:
        config = linear_solvers.SchurElimSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.elim_rows = [int(r) for r in self.elim_rows]
        config.elim_cols = [int(c) for c in self.elim_cols]
        config.pivot_eps = self.pivot_eps
        return config

    def build(self, block_size: int):
        """Build the wrapper at ``block_size`` and the inner solver at
        ``block_size - K``, attaching it via ``set_prec``.

        The built inner solver is stored on this spec so it outlives any local
        reference at the call site -- the C++ wrapper holds a raw pointer.
        """
        if self.inner is None:
            raise ValueError(
                "SchurEliminationSpec requires an inner solver spec, e.g. "
                "SchurEliminationSpec(inner=GMRESSolverSpec(prec=CPRSolverSpec()), "
                "elim_rows=[0], elim_cols=[1])."
            )
        if not self.elim_rows or not self.elim_cols:
            raise ValueError(
                "SchurEliminationSpec requires explicit elim_rows and elim_cols "
                "(the local-equation rows and their unknown columns); there is no default."
            )
        k = len(self.elim_cols)
        if len(self.elim_rows) != k:
            raise ValueError(
                f"SchurEliminationSpec: elim_rows ({len(self.elim_rows)}) and elim_cols "
                f"({k}) must have equal length."
            )
        if block_size - k < 1:
            raise ValueError(
                f"SchurEliminationSpec: eliminating {k} of {block_size} equations leaves "
                "no reduced system; K must be < block size."
            )
        if 0 in self.elim_cols:
            raise ValueError(
                "SchurEliminationSpec: column 0 (pressure) cannot be eliminated -- a "
                "CPR-type inner solver needs the pressure subsystem preserved."
            )
        wrapper = linear_solvers.create_linear_solver(
            self.registry_name, self._make_config(), block_size
        )
        self._built_inner = self.inner.build(block_size - k)
        wrapper.set_prec(self._built_inner)
        return wrapper


@dataclass
class GPUSolverSpec(LinearSolverSpec):
    """Base spec for the device-resident (GPU) linear solvers.

    The same GPU solver chain is reachable in two ways, selected by the model's
    platform:

    * platform ``'gpu'`` (GPU engines, device-assembled Jacobian): the spec names a
      ``params.linear_type`` (``darts.engines.linear_solver_t``) enum value via
      :attr:`linear_type_name`; :meth:`~darts.linear_solvers.LinearSolver._apply_solver`
      translates it and the GPU engine factory builds the chain. This keeps
      ``self.linear_solver.spec`` the single user-facing API on GPU too.
    * platform ``'cpu'`` (host-assembled Jacobian: every mechanics / poromechanics
      engine, or a flow engine run on the CPU): :meth:`build` creates the same chain
      through the open-source registry (the ``gpu_*`` names, :attr:`registry_name`)
      wrapped in ``linsolv_host_adapter``. The adapter mirrors the Jacobian values to
      the device at every ``setup()`` (one H2D copy per Newton iteration) and stages
      the host RHS / solution vectors, so the solver is injected and switched exactly
      like the CPU registry specs (``engine.set_linear_solver`` /
      ``model.linear_solver.update_solver(spec=...)``). A run can therefore switch
      between CPU and GPU solvers mid-run, e.g. cuDSS for a quasi-static stage and
      GMRES + FS-CPR for a dynamic one.

    The registry path needs a CUDA build of open-DARTS (cuDSS additionally needs
    ``WITH_CUDSS``, the AMGX-CPR variants ``WITH_AMGX``); :meth:`build` raises
    ``NotImplementedError`` when the name is not registered in this build.
    """

    #: name of the ``darts.engines.sim_params`` ``linear_solver_t`` enum value
    #: (platform 'gpu': selected by the GPU engine factory)
    linear_type_name: ClassVar[str] = ""

    #: name in the open-source solver registry (platform 'cpu': built through
    #: ``linear_solvers.create_linear_solver`` and wrapped in linsolv_host_adapter)
    registry_name: ClassVar[str] = ""

    #: K > 0 wraps the GPU chain in an exact per-cell local (block-Schur)
    #: elimination of K cell-local (diagonal-block-only) equation/unknown pairs
    #: (``linsolv_schur_elim``); the chain is then built at the reduced block
    #: size N-K. Honoured by the AMGX-CPR family of GPU solvers on platform
    #: 'gpu'; mirrors ``SchurEliminationSpec`` on the CPU side (on platform 'cpu'
    #: compose ``SchurEliminationSpec(inner=<GPU spec>)`` instead). When > 0,
    #: :attr:`schur_elim_rows` / :attr:`schur_elim_cols` (each of length K) give
    #: the explicit eliminated (row, column) pairs.
    schur_elim_count: int = 0
    schur_elim_rows: list[int] | None = None
    schur_elim_cols: list[int] | None = None

    #: CUDA device the chain runs on (registry path: ``cudaSetDevice`` at init;
    #: combine with ``CUDA_VISIBLE_DEVICES`` to pick a physical GPU).
    device_num: int = 0

    def _make_config(self) -> linear_solvers.GPUSolverConfig:
        config = linear_solvers.GPUSolverConfig()
        config.tolerance = self.tolerance
        config.max_iterations = self.max_iterations
        config.device_num = self.device_num
        config.restart = int(getattr(self, "restart", 50))
        config.ilu_single_precision = bool(getattr(self, "ilu_single_precision", False))
        config.cudss_ir_steps = int(getattr(self, "ir_steps", 2))
        config.cudss_pivot_epsilon = float(getattr(self, "pivot_epsilon", -1.0))
        # hybrid-memory fields exist in builds that carry the option; keep older
        # compiled modules usable (they run the plain device-memory cuDSS).
        if hasattr(config, "cudss_hybrid_memory"):
            config.cudss_hybrid_memory = bool(getattr(self, "hybrid_memory", False))
            config.cudss_hybrid_device_memory_limit = int(
                getattr(self, "hybrid_device_memory_limit", 0)
            )
        return config

    @classmethod
    def available(cls) -> bool:
        """True when this build's registry carries the solver (CUDA build with the
        required optional library), i.e. :meth:`build` will succeed."""
        return (
            linear_solvers is not None
            and bool(cls.registry_name)
            and bool(linear_solvers.is_solver_registered(cls.registry_name))
        )

    def build(self, block_size: int):
        """Build the GPU chain for a host-assembled system (platform 'cpu').

        :raises NotImplementedError: when the registry of this build does not
            carry :attr:`registry_name` (CPU-only build, or a GPU build without
            the optional library the chain needs).
        """
        if not self.available():
            raise NotImplementedError(
                f"{type(self).__name__} ('{self.registry_name or self.linear_type_name}') "
                "is not available in this build: the gpu_* registry solvers need a "
                "CUDA build of open-DARTS (cuDSS also needs WITH_CUDSS, the AMGX-CPR "
                "chains WITH_AMGX). On platform='gpu' the spec is selected via "
                f"params.linear_type ({self.linear_type_name or '<unset>'}) by the GPU "
                "engine factory instead."
            )
        if self.schur_elim_count:
            raise NotImplementedError(
                "schur_elim_count is honoured by the GPU engine factory (platform "
                "'gpu') only; on the registry path compose "
                "SchurEliminationSpec(inner=<GPU spec>, elim_rows=..., elim_cols=...)."
            )
        return linear_solvers.create_linear_solver(
            self.registry_name, self._make_config(), block_size
        )


@dataclass
class AMGXCPRSolverSpec(GPUSolverSpec):
    """GPU GMRES + AMGX-CPR -- the default GPU solver.

    NVIDIA AMGX algebraic multigrid on the pressure subsystem + cuSPARSE
    block-ILU(0) on the full system, wrapped in the in-tree GPU GMRES. Maps to
    ``linear_solver_t.gpu_gmres_cpr_amgx_ilu`` (platform 'gpu') / registry
    ``gpu_gmres_cpr_amgx`` (platform 'cpu'; needs ``WITH_AMGX``). The CPR split
    assumes the flow-engine variable layout (pressure first): for poromechanics
    (``engine_pm_cpu``, ``[u_x, u_y, u_z, p]``) use ``FSCPRSolverSpec``, a direct
    solver or :class:`GPUGMRESILU0SolverSpec` instead.

    :param restart: GMRES restart length (registry path).
    :param ilu_single_precision: keep the block-ILU(0) factors in float
        (registry path).
    """

    linear_type_name: ClassVar[str] = "gpu_gmres_cpr_amgx_ilu"
    registry_name: ClassVar[str] = "gpu_gmres_cpr_amgx"

    restart: int = 50
    ilu_single_precision: bool = False


@dataclass
class GPUBiCGStabCPRSolverSpec(GPUSolverSpec):
    """GPU BiCGStab + AMGX-CPR. Maps to ``linear_solver_t.gpu_bicgstab_cpr_amgx``
    (platform 'gpu') / registry ``gpu_bicgstab_cpr_amgx`` (platform 'cpu'; needs
    ``WITH_AMGX``; pressure-first layout, see :class:`AMGXCPRSolverSpec`)."""

    linear_type_name: ClassVar[str] = "gpu_bicgstab_cpr_amgx"
    registry_name: ClassVar[str] = "gpu_bicgstab_cpr_amgx"

    ilu_single_precision: bool = False


@dataclass
class GPUGMRESILU0SolverSpec(GPUSolverSpec):
    """GPU GMRES + cuSPARSE block-ILU(0) -- a single-stage GPU solver (no AMG).

    Layout-agnostic (the block-ILU(0) works on any block structure), so it is the
    Krylov GPU option for poromechanics. Maps to ``linear_solver_t.gpu_gmres_ilu0``
    (platform 'gpu') / registry ``gpu_gmres_ilu0`` (platform 'cpu').

    :param restart: GMRES restart length (registry path).
    :param ilu_single_precision: keep the block-ILU(0) factors in float
        (registry path).
    """

    linear_type_name: ClassVar[str] = "gpu_gmres_ilu0"
    registry_name: ClassVar[str] = "gpu_gmres_ilu0"

    restart: int = 50
    ilu_single_precision: bool = False


@dataclass
class CuDSSSolverSpec(GPUSolverSpec):
    """GPU sparse DIRECT solver via NVIDIA cuDSS (``linsolv_cudss``).

    cuDSS (https://docs.nvidia.com/cuda/cudss/) is NVIDIA's sparse
    direct-solver library, the successor of the deprecated cusolverSp QR
    wrapped by ``GPUCuSolverSpec``. Exact solve -- one "linear iteration" per
    Newton step; the GPU counterpart of ``SuperLUSolverSpec`` /
    ``PardisoSolverSpec`` and a robustness fallback when iterative GPU
    solvers struggle. Memory-bound: the LU fill of a 3D problem grows
    super-linearly with the system size (see the solver docs for measured
    figures on the poromechanics fault model).

    Requires a GPU build configured with ``-D WITH_CUDSS=ON`` and the prebuilt
    library (``pip install nvidia-cudss-cu13`` into the build's environment).
    Maps to ``linear_solver_t.gpu_cudss`` (platform 'gpu'; a build without cuDSS
    falls back to the BiCGStab + cuSPARSE-ILU(0) GPU solver with a console
    notice) / registry ``gpu_cudss`` (platform 'cpu').

    :param ir_steps: iterative-refinement steps after each solve (registry
        path; default 2). cuDSS perturbs tiny pivots (static pivoting,
        ``CUDSS_CONFIG_PIVOT_EPSILON``) and reports success regardless, so on a
        badly scaled Jacobian (the poromechanics fault model's rows are scaled
        to unit maximum with entries down to 1e-40) the refinement is a cheap
        safeguard -- two extra triangular solves and one SpMV per solve.
        Measured true residuals on that model were at round-off (1e-14) with
        and without it.
    :param pivot_epsilon: ``CUDSS_CONFIG_PIVOT_EPSILON`` override (``< 0`` =
        library default).
    :param hybrid_memory: ``CUDSS_CONFIG_HYBRID_MEMORY_MODE`` -- keep part of the
        LU factors in host memory so that systems whose factorization does not
        fit the free device memory still solve (slower); the 2M-cell
        displaced-fault mesh (8M unknowns) needs it on a shared 80 GB GPU.
    :param hybrid_device_memory_limit: device share in hybrid mode [bytes]
        (0 = cuDSS default).

    The solve is NOT bit-reproducible run to run (parallel factorization;
    differences at the 1e-11 relative level, which a Newton loop poised at a
    contact / well-control bifurcation can amplify into a different timestep
    path); cuDSS 0.8 rejects its ``CUDSS_CONFIG_DETERMINISTIC_MODE`` for this
    matrix type at the analysis phase, so no deterministic option is offered.
    """

    linear_type_name: ClassVar[str] = "gpu_cudss"
    registry_name: ClassVar[str] = "gpu_cudss"

    ir_steps: int = 2
    pivot_epsilon: float = -1.0
    hybrid_memory: bool = False
    hybrid_device_memory_limit: int = 0


@dataclass
class GPUCuSolverSpec(GPUSolverSpec):
    """GPU sparse direct solver via cuSOLVER QR (``linsolv_cusolv``).

    Legacy GPU direct solve (``cusolverSpDcsrlsvqr``, deprecated by NVIDIA in
    favour of cuDSS -- prefer :class:`CuDSSSolverSpec` when available). Maps
    to ``linear_solver_t.gpu_cusolver`` (platform 'gpu') / registry
    ``gpu_cusolver`` (platform 'cpu').
    """

    linear_type_name: ClassVar[str] = "gpu_cusolver"
    registry_name: ClassVar[str] = "gpu_cusolver"
