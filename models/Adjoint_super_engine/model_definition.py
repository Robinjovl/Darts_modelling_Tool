import numpy as np
from darts.engines import *
from darts.engines import sim_params

from darts import linear_solvers
from darts.linear_solvers import (
    BCSRCPRSpec,
    BILU0Spec,
    LocalCorrectionSpec,
    MGRLevelSpec,
    MGRSolverSpec,
    PressureAMGSpec,
)
from darts.linear_solvers.enums import (
    BCSRCPRReduction,
    CoarseGrid,
    CompositeMode,
    FRelaxation,
    GlobalSmoother,
    Interpolation,
    LocalFallback,
    LocalPreconditioner,
    Restriction,
    VariableRole,
)
from darts.models.darts_model import DartsModel
from darts.models.opt.opt_module_settings import OptModuleSettings
from darts.nonlinear_solvers import ChopSpec, NewtonSolver
from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.flash import ConstantK
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.keyword_file_tools import get_table_keyword


class Model(DartsModel, OptModuleSettings):
    def __init__(
        self,
        T,
        report_step=120,
        perm=300,
        poro=0.2,
        customize_new_operator=False,
        Peaceman_WI=False,
        use_adjoint_mgr=True,
        adjoint_solver=None,
        adjoint_mgr_profile="physical",
        adjoint_mgr_options=None,
    ):
        # call base class constructor
        DartsModel.__init__(self)
        OptModuleSettings.__init__(self)

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.T = T
        self.report_step = report_step
        self.customize_new_operator = customize_new_operator

        # initialize global data to record the well location in vtk output file
        self.global_data = {'well location': 0}  # will be updated later in "run"

        self.set_reservoir(perm, poro)
        self.Peaceman_WI = Peaceman_WI
        self.set_physics()
        self.linear_solver = None
        self.adjoint_solver = None
        self.adjoint_linear_tol = 1e-10
        self.adjoint_linear_max_iter = 300
        self.adjoint_solver_mode = self._normalize_adjoint_solver(
            adjoint_solver, use_adjoint_mgr
        )
        self.use_mgr_for_adjoint = self.adjoint_solver_mode == "mgr"
        self._adjoint_solver_spec = None
        self.adjoint_cpra_options = {
            "restart": 150,
            "cpr_amg_max_iters": 2,
            "cpr_amg_tolerance": 1e-2,
            "cpr_ilu_fill_level": 0,
        }
        self.adjoint_mgr_profile = adjoint_mgr_profile
        self.adjoint_mgr_options = self._make_adjoint_mgr_options(
            adjoint_mgr_profile,
            **(adjoint_mgr_options or {}),
        )

        # Time-stepping and the forward/adjoint linear solvers are configured in
        # set_solver()/reset() (the unified set_solver pattern), invoked during
        # init(). adjoint_mgr_options / adjoint_solver_mode are already set above,
        # so set_adjoint_solver() (called from reset()) has everything it needs.

        self.timer.node["initialization"].stop()

    @staticmethod
    def _normalize_adjoint_solver(adjoint_solver, use_adjoint_mgr=True):
        if adjoint_solver is None:
            return "mgr" if use_adjoint_mgr else "superlu"
        solver = str(adjoint_solver).lower()
        if solver not in {"mgr", "superlu", "cpra", "cpra-gpu"}:
            raise ValueError("adjoint_solver must be 'mgr', 'superlu', 'cpra', or 'cpra-gpu'")
        return solver

    def _make_adjoint_mgr_options(self, profile="physical", **overrides):
        profile = "physical" if profile is None else str(profile).lower()
        true_impes_well_elim = getattr(
            sim_params,
            "mgrCprReductionTrueIMPESWellElim",
            sim_params.mgrCprReductionTrueIMPES,
        )

        common = {
            "profile": profile,
            "use_bcsr_cpr": True,
            "pressure_variable": 0,
            "bcsr_cpr_weight_max": 1e6,
            "pressure_interp": sim_params.mgrInterpInjection,
            "pressure_restrict": sim_params.mgrRestrictBlockColLumped,
            "pressure_coarse": sim_params.mgrCoarseGalerkin,
            "pressure_smoother": sim_params.mgrSmootherHypreILU,
            "pressure_smoother_iters": 1,
            "well_frelax": sim_params.mgrFRelaxDirectInverse,
            "well_frelax_iters": 1,
            "well_interp": sim_params.mgrInterpBlockJacobi,
            "well_restrict": sim_params.mgrRestrictInjection,
            "well_coarse": sim_params.mgrCoarseGalerkin,
            "well_smoother": sim_params.mgrSmootherNone,
            "well_smoother_iters": 0,
            "composition_frelax": sim_params.mgrFRelaxJacobi,
            "composition_frelax_iters": 1,
            "composition_interp": sim_params.mgrInterpJacobi,
            "composition_restrict": sim_params.mgrRestrictInjection,
            "composition_coarse": sim_params.mgrCoarseGalerkin,
            "composition_smoother": sim_params.mgrSmootherNone,
            "composition_smoother_iters": 0,
            "transpose_apply": False,
            "forward_cpr_source": False,
            "local_correction_alpha": 1.0,
            "local_quality_gate": False,
            "local_quality_min_alpha": 0.0,
            "pressure_amg_max_iter": 1,
            "pressure_amg_tolerance": 0.0,
            "pressure_correction_alpha": 1.0,
            "pressure_guard_threshold": 10.0,
            "pressure_guard_min_alpha": 0.05,
            "diagnostics": False,
            "diagnostic_apply_interval": 0,
            "diagnostic_matrix_interval": 0,
        }

        if profile == "baseline":
            options = {
                **common,
                "physics_scaling": False,
                "bcsr_cpr_reduction_type": sim_params.mgrCprReductionTrueIMPES,
                "enable_well_level": False,
                "enable_composition_level": False,
                "pressure_frelax": sim_params.mgrFRelaxNone,
                "pressure_frelax_iters": 0,
            }
        elif profile == "physical":
            options = {
                **common,
                "physics_scaling": True,
                "bcsr_cpr_reduction_type": true_impes_well_elim,
                "enable_well_level": True,
                "enable_composition_level": True,
                "pressure_frelax": sim_params.mgrFRelaxL1Jacobi,
                "pressure_frelax_iters": 1,
                "forward_cpr_source": True,
            }
        else:
            raise ValueError(
                "adjoint_mgr_profile must be 'baseline' or 'physical'"
            )

        for key, value in overrides.items():
            if value is not None:
                options[key] = value
        if options.get("forward_cpr_source"):
            options["transpose_apply"] = True
        return options

    def configure_adjoint_mgr_profile(self, profile=None, **overrides):
        self.adjoint_mgr_options = self._make_adjoint_mgr_options(
            profile or getattr(self, "adjoint_mgr_profile", "physical"),
            **overrides,
        )
        self.adjoint_mgr_profile = self.adjoint_mgr_options["profile"]

    def use_adjoint_mgr_profile(self, profile=None, **overrides):
        self.adjoint_solver_mode = "mgr"
        self.use_mgr_for_adjoint = True
        self._adjoint_solver_spec = None
        self.configure_adjoint_mgr_profile(profile, **overrides)
        self.set_adjoint_solver()

    def use_adjoint_mgr_baseline_profile(self):
        self.use_adjoint_mgr_profile(profile="baseline")

    def adjoint_mgr_profile_summary(self):
        return dict(getattr(self, "adjoint_mgr_options", {}))

    def use_adjoint_cpra_profile(self, **overrides):
        self.adjoint_solver_mode = "cpra"
        self.use_mgr_for_adjoint = False
        options = dict(getattr(self, "adjoint_cpra_options", {}))
        for key, value in overrides.items():
            if value is not None:
                options[key] = value
        self.adjoint_cpra_options = options
        self.set_adjoint_solver()

    def adjoint_cpra_profile_summary(self):
        return dict(getattr(self, "adjoint_cpra_options", {}))

    def use_adjoint_superlu_profile(self):
        self.adjoint_solver_mode = "superlu"
        self.use_mgr_for_adjoint = False
        self._adjoint_solver_spec = None
        self.adjoint_solver = None

    def set_reservoir(self, perm, poro):
        """Reservoir construction"""
        # nx = 20
        # ny = 10
        # nz = 2

        # Default 5x5x2; overridable via env for benchmarking the device vs host
        # adjoint assembly at scale (perm/poro are scalars, broadcast to the grid).
        import os as _os
        nx = int(_os.environ.get("ADJ_NX", 5))
        ny = int(_os.environ.get("ADJ_NY", 5))
        nz = int(_os.environ.get("ADJ_NZ", 2))

        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=30, dy=30, dz=12,
                                         permx=perm, permy=perm, permz=perm, poro=poro, depth=2000)

        return

    def set_wells(self):
        # self.inj_list = [[5, 5]]
        # self.prod_list = [[15, 3], [15, 8]]

        self.inj_list = [[1, 5], [5, 1]]
        self.prod_list = [[1, 1], [5, 5]]

        # well index setting
        if self.Peaceman_WI:
            WI = -1  # use Peaceman function; check the function "add_perforation" for more details
        else:
            WI = 200

        n_perf = self.reservoir.nz
        for i, inj in enumerate(self.inj_list):
            self.reservoir.add_well('I' + str(i + 1))

            for k in range(n_perf):
                self.reservoir.add_perforation('I' + str(i + 1), res_cell_idx=(inj[0], inj[1], k + 1),
                                               well_diameter=0.2, well_index=WI)

        for p, prod in enumerate(self.prod_list):
            self.reservoir.add_well('P' + str(p + 1))

            for k in range(n_perf):
                self.reservoir.add_perforation('P' + str(p + 1), res_cell_idx=(prod[0], prod[1], k + 1),
                                               well_diameter=0.2, well_index=WI)

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        zero = 1e-8
        epsilon = 1e-9
        components = ['CO2', 'C1', 'H2O']
        phases = ['gas', 'oil']
        Mw = [44.01, 16.04, 18.015]
        nc = len(components)

        self.inj_composition = [1.0 - 2 * zero, zero]
        self.ini_stream = [0.1, 0.2]

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=Mw,
                                               eps_z=epsilon, temperature=1.)
        property_container.flash_ev = ConstantK(nc, [4, 2, 1e-1], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('oil', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil"))])

        """ Activate physics """
        thermal = False
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        # [p, z_1, ..., z_{nc-1}]
        nz = len(components) - 1
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[1.5] + [5e-3] * nz,
                                     axes_origin=[1.0] + [epsilon] * nz,
                                     epsilon_z=epsilon, extrapolation_flag=True)
        self.physics.add_property_region(property_container)

        return

    def _reservoir_block_count(self):
        mesh = getattr(self.reservoir, "mesh", None)
        return None if mesh is None else mesh.n_res_blocks

    def _adjoint_reservoir_block_count(self):
        return self._reservoir_block_count()

    def set_solver(self):
        # Idempotent: the forward solver is built once (before engine.init, so the
        # adjoint solver can hold a reference to it). The base reset() calls
        # set_solver() again at its top -- skip the rebuild so that reference stays
        # valid.
        if self.linear_solver is not None:
            return
        # Single per-model home for time-stepping / Newton config (the unified
        # set_solver pattern); the base reset() calls this before engine.init.
        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000 )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-6, max_iterations=10,
            chop=ChopSpec(mode='local'))
        self.params.linear_print_level = 0  # 0 = quiet, 1 = basic, 2 = verbose
        # Forward MGR (BCSR-CPR) via the single unified spec API (self.linear_solver =
        # MGRSolverSpec). The base DartsModel._apply_solver hook builds + injects it
        # before engine.init on the open-source CPU build; the adjoint solver is
        # injected separately by _attach_mgr_solvers_to_engine(). On the proprietary -a
        # build the spec is not built; _apply_solver applies proprietary_linear_type
        # (cpu_gmres_cpr_amg) to params.linear_type instead. Mirrors 2ph_comp's
        # MGRSolverSpec (here diagnostics are off, as in the former raw build).
        block_size = self.physics.n_vars
        reservoir_blocks = self._reservoir_block_count() or 0

        reservoir_roles = [VariableRole.PRESSURE] + [VariableRole.COMPOSITION] * (
            block_size - 1
        )
        well_roles = [VariableRole.WELL_PRESSURE] + [VariableRole.WELL_SECONDARY] * (
            block_size - 1
        )

        self.linear_solver = MGRSolverSpec(
            tolerance=1e-3,
            max_iterations=50,
            log_level=self.params.linear_print_level,
            proprietary_linear_type=sim_params.cpu_gmres_cpr_amg,
            kdim=150,
            use_mgr=True,
            use_flex_gmres=True,
            use_physics_scaling=True,
            composite_mode=CompositeMode.MGR_THEN_LOCAL,
            local_solver=LocalPreconditioner.BLOCK_ILU0,
            bilu0=BILU0Spec(
                pivot_shift=1e-12,
                fallback_strategy=LocalFallback.IDENTITY,
                fallback_diagonal_tolerance=1e-4,
                fallback_shifted_max=1e-4,
                fallback_shifted_growth=100.0,
            ),
            local_correction=LocalCorrectionSpec(
                alpha=1.0,
                adaptive_fallback_threshold=-1.0,
                adaptive_alpha=0.0,
                adaptive_fallback_threshold_high=-1.0,
                adaptive_alpha_high=0.0,
                quality_enabled=False,
                quality_min_alpha=0.0,
            ),
            pressure_amg=PressureAMGSpec(
                coarsen_type=6,
                interp_type=6,
                relax_type=6,
                agg_num_levels=1,
                agg_interp_type=6,
                agg_pmax_elmts=20,
                relax_order=1,
                strong_threshold=0.5,
                trunc_factor=-1.0,
                pmax_elmts=-1,
                max_levels=0,
                solve_max_iter=1,
                solve_tolerance=0.0,
            ),
            bcsr_cpr=BCSRCPRSpec(
                # Forward solver uses True-IMPES reduction; the adjoint solver's
                # reduction is configured separately via adjoint_mgr_options.
                reduction_type=BCSRCPRReduction.TRUE_IMPES,
                pressure_variable=0,
                weight_max=1e6,
                reuse_amg_hierarchy=True,
                amg_rebuild_interval=0,
                adaptive_amg_rebuild=True,
                adaptive_li_threshold=15,
                adaptive_li_growth_factor=1.5,
                adaptive_min_reuse_setups=1,
                adaptive_max_reuse_setups=2,
                adaptive_pressure_overshoot_threshold=-1.0,
                adaptive_final_proxy_threshold=-1.0,
                adaptive_fallback_threshold=-1.0,
                diagnostics=False,
                diagnostic_apply_interval=0,
                diagnostic_matrix_interval=0,
                pressure_correction_alpha=1.0,
                pressure_correction_guard_threshold=10.0,
                pressure_correction_guard_min_alpha=0.05,
            ),
            reservoir_variable_roles=reservoir_roles,
            well_variable_roles=well_roles,
            pressure_level=MGRLevelSpec(
                frelax_type=FRelaxation.NONE,
                frelax_iters=0,
                interp_type=Interpolation.INJECTION,
                restrict_type=Restriction.BLOCK_COL_LUMPED,
                coarse_method=CoarseGrid.GALERKIN,
                smoother_type=GlobalSmoother.HYPRE_ILU,
                smoother_iters=1,
            ),
            n_reservoir_blocks=int(reservoir_blocks),
            enable_well_level=False,
            enable_composition_level=False,
        )
        self.solver_label = "mgr (bcsr-cpr, forward)"

    def set_adjoint_solver(self):
        if getattr(self, "adjoint_solver_mode", "mgr") in {"superlu", "cpra-gpu"}:
            # superlu: the engine's built-in default adjoint solver.
            # cpra-gpu: the stack is built inside the engine by
            # engine.set_adjoint_solver_cpra_gpu() at attach time.
            self.adjoint_solver = None
            self._adjoint_solver_spec = None
            return
        # MGR / CPRA adjoint solvers use the open-source-only registry; in the
        # proprietary -a build leave the adjoint solver unset (engine factory).
        if not self.open_source_solvers_available():
            self.adjoint_solver = None
            self._adjoint_solver_spec = None
            return
        if getattr(self, "adjoint_solver_mode", "mgr") == "cpra":
            self.set_adjoint_cpra_solver()
            return

        block_size = self.physics.n_vars
        adjoint_options = getattr(
            self,
            "adjoint_mgr_options",
            self._make_adjoint_mgr_options("physical"),
        )
        self.adjoint_solver = linear_solvers.create_mgr_solver_for_block_size(block_size)
        self.adjoint_solver.set_max_iterations(self.adjoint_linear_max_iter)
        self.adjoint_solver.set_tolerance(self.adjoint_linear_tol)
        self.adjoint_solver.set_log_level(self.params.linear_print_level)
        self.adjoint_solver.set_kdim(150)
        self.adjoint_solver.set_use_mgr(True)
        self.adjoint_solver.set_use_flex_gmres(True)
        self.adjoint_solver.set_use_physics_scaling(
            bool(adjoint_options["physics_scaling"])
        )
        self.adjoint_solver.set_mgr_composite_mode(1)
        self.adjoint_solver.set_mgr_local_solver(
            getattr(sim_params, "mgrLocalSolverBlockILU0", 2)
        )
        self.adjoint_solver.set_mgr_bilu0_pivot_shift(1e-12)
        self.adjoint_solver.set_mgr_bilu0_fallback_options(
            sim_params.mgrBilu0FallbackIdentity,
            1e-4,
            1e-4,
            100.0,
        )
        self.adjoint_solver.set_mgr_local_correction_options(
            adjoint_options["local_correction_alpha"], -1.0, 0.0, -1.0, 0.0
        )
        self.adjoint_solver.set_mgr_local_correction_quality_options(
            bool(adjoint_options["local_quality_gate"]),
            adjoint_options["local_quality_min_alpha"],
        )
        self.adjoint_solver.set_mgr_pressure_amg_options(6, 6, 6, 1, 6, 20, 1)
        if hasattr(self.adjoint_solver, "set_mgr_pressure_amg_advanced_options"):
            self.adjoint_solver.set_mgr_pressure_amg_advanced_options(
                0.5, -1.0, -1, 0
            )
        self.adjoint_solver.set_mgr_pressure_amg_solve_options(
            adjoint_options["pressure_amg_max_iter"],
            adjoint_options["pressure_amg_tolerance"],
        )
        self.adjoint_solver.set_use_bcsr_cpr(bool(adjoint_options["use_bcsr_cpr"]))
        self.adjoint_solver.set_bcsr_cpr_options(
            adjoint_options["bcsr_cpr_reduction_type"],
            adjoint_options["pressure_variable"],
            adjoint_options["bcsr_cpr_weight_max"],
        )
        self.adjoint_solver.set_bcsr_cpr_reuse_options(True, 0)
        self.adjoint_solver.set_bcsr_cpr_adaptive_rebuild_options(True, 15, 1.5, 1, 2)
        self.adjoint_solver.set_bcsr_cpr_adaptive_quality_options(-1.0, -1.0, -1.0)
        self.adjoint_solver.set_bcsr_cpr_diagnostics_options(
            bool(adjoint_options["diagnostics"]),
            adjoint_options["diagnostic_apply_interval"],
            adjoint_options["diagnostic_matrix_interval"],
        )
        self.adjoint_solver.set_bcsr_cpr_pressure_correction_options(
            adjoint_options["pressure_correction_alpha"],
            adjoint_options["pressure_guard_threshold"],
            adjoint_options["pressure_guard_min_alpha"],
        )
        forward_cpr_source = bool(adjoint_options.get("forward_cpr_source", False))
        transpose_apply = bool(adjoint_options["transpose_apply"]) or forward_cpr_source
        if hasattr(self.adjoint_solver, "set_bcsr_cpr_forward_source"):
            self.adjoint_solver.set_bcsr_cpr_forward_source(forward_cpr_source)
        if hasattr(self.adjoint_solver, "set_bcsr_cpr_transpose_apply"):
            self.adjoint_solver.set_bcsr_cpr_transpose_apply(transpose_apply)

        reservoir_roles = [sim_params.mgrVarPressure] + [
            sim_params.mgrVarComposition
        ] * (block_size - 1)
        well_roles = [sim_params.mgrVarWellPressure] + [
            sim_params.mgrVarWellSecondary
        ] * (block_size - 1)
        self.adjoint_solver.set_mgr_reservoir_variable_roles(
            reservoir_roles
        )
        self.adjoint_solver.set_mgr_well_variable_roles(
            well_roles
        )
        self.adjoint_solver.set_mgr_well_strategy(
            getattr(sim_params, "mgrWellEliminateBlock", 0)
        )
        self.adjoint_solver.set_mgr_well_level_options(
            adjoint_options["well_frelax"],
            adjoint_options["well_frelax_iters"],
            adjoint_options["well_interp"],
            adjoint_options["well_restrict"],
            adjoint_options["well_coarse"],
            adjoint_options["well_smoother"],
            adjoint_options["well_smoother_iters"],
        )
        self.adjoint_solver.set_mgr_composition_level_options(
            adjoint_options["composition_frelax"],
            adjoint_options["composition_frelax_iters"],
            adjoint_options["composition_interp"],
            adjoint_options["composition_restrict"],
            adjoint_options["composition_coarse"],
            adjoint_options["composition_smoother"],
            adjoint_options["composition_smoother_iters"],
        )
        self.adjoint_solver.set_mgr_pressure_level_options(
            adjoint_options["pressure_frelax"],
            adjoint_options["pressure_frelax_iters"],
            adjoint_options["pressure_interp"],
            adjoint_options["pressure_restrict"],
            adjoint_options["pressure_coarse"],
            adjoint_options["pressure_smoother"],
            adjoint_options["pressure_smoother_iters"],
        )
        self.adjoint_solver.set_mgr_enable_well_level(
            bool(adjoint_options["enable_well_level"])
        )
        self.adjoint_solver.set_mgr_enable_composition_level(
            bool(adjoint_options["enable_composition_level"])
        )

        adjoint_reservoir_blocks = self._adjoint_reservoir_block_count()
        if adjoint_reservoir_blocks is not None:
            self.adjoint_solver.set_n_reservoir_blocks(adjoint_reservoir_blocks)

    def set_adjoint_cpra_solver(self):
        block_size = self.physics.n_vars
        options = getattr(self, "adjoint_cpra_options", {})
        # No amg_tolerance: BoomerAMG inside CPR always runs with tol=0 as a
        # preconditioner stage (the parameter was removed in the spec-surface
        # cleanup; the sweep budget amg_max_iters is the only AMG knob).
        cpr_spec = linear_solvers.CPRSolverSpec(
            tolerance=self.adjoint_linear_tol,
            max_iterations=self.adjoint_linear_max_iter,
            print_level=self.params.linear_print_level,
            amg_max_iters=int(options.get("cpr_amg_max_iters", 2)),
            ilu_fill_level=int(options.get("cpr_ilu_fill_level", 0)),
        )
        gmres_spec = linear_solvers.GMRESSolverSpec(
            tolerance=self.adjoint_linear_tol,
            max_iterations=self.adjoint_linear_max_iter,
            print_level=self.params.linear_print_level,
            restart=int(options.get("restart", 150)),
            prec=cpr_spec,
        )
        self._adjoint_solver_spec = gmres_spec
        self.adjoint_solver = gmres_spec.build(block_size)

    def _attach_mgr_solvers_to_engine(self):
        engine = getattr(self.physics, "engine", None)
        if engine is None:
            return

        # The forward solver (self.linear_solver = MGRSolverSpec) is built and injected by
        # the base DartsModel._apply_solver hook from super().reset(). Its
        # n_reservoir_blocks is only FINAL after engine.init (the well/reservoir
        # partition is split there): mesh.n_res_blocks reads the total pre-init and
        # the reservoir-only count post-init. The spec captured the pre-init value,
        # so correct the built forward solver here and re-inject -- exactly as the
        # former raw _attach did. (self._linear_solver is the base-built MGR object.)
        forward_solver = getattr(self, "_linear_solver", None)
        if forward_solver is not None and hasattr(
            forward_solver, "set_n_reservoir_blocks"
        ):
            reservoir_blocks = self._reservoir_block_count()
            if reservoir_blocks is not None:
                forward_solver.set_n_reservoir_blocks(reservoir_blocks)
                engine.set_linear_solver(forward_solver)

        if getattr(self, "adjoint_solver_mode", "mgr") == "cpra-gpu":
            rc = engine.set_adjoint_solver_cpra_gpu()
            if rc != 0:
                raise RuntimeError(
                    "adjoint_solver='cpra-gpu' requires the GPU engine with AMGX "
                    "(run init(platform='gpu') on an AMGX-enabled build); "
                    "engine.set_adjoint_solver_cpra_gpu() returned %d" % rc
                )
            return

        if self.adjoint_solver is not None and hasattr(
            engine, "set_adjoint_linear_solver"
        ):
            adjoint_reservoir_blocks = self._adjoint_reservoir_block_count()
            if adjoint_reservoir_blocks is not None and hasattr(
                self.adjoint_solver, "set_n_reservoir_blocks"
            ):
                self.adjoint_solver.set_n_reservoir_blocks(adjoint_reservoir_blocks)
            engine.set_adjoint_linear_solver(
                self.adjoint_solver, use_jacobian_transpose=True
            )

    def reset(self):
        # super().reset() runs set_solver() (time-stepping + forward MGR spec) and
        # engine.init via the base hook. The adjoint solver is then (re)built fresh
        # each reset -- params.linear_print_level is set by set_solver() above, and
        # _attach_mgr_solvers_to_engine() corrects its post-init reservoir-block count.
        super().reset()
        if getattr(self, "adjoint_solver_mode", "mgr") in {"mgr", "cpra"}:
            self.set_adjoint_solver()
        self._attach_mgr_solvers_to_engine()

    def grad_adjoint_method_all(self, x):
        old_tol = self.params.tolerance_linear
        old_max_iter = self.params.max_i_linear
        self.params.tolerance_linear = self.adjoint_linear_tol
        self.params.max_i_linear = self.adjoint_linear_max_iter
        try:
            return OptModuleSettings.grad_adjoint_method_all(self, x)
        finally:
            self.params.tolerance_linear = old_tol
            self.params.max_i_linear = old_max_iter

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 50.,
                              self.physics.vars[1]: self.ini_stream[0],
                              self.physics.vars[2]: self.ini_stream[1],
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)


    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if "I" in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=140., inj_composition=self.inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=50.)

    def set_op_list(self):
        if self.customize_new_operator:
            customized_component_etor = customized_etor_specific_component()
            # NB: stays on the CPU also for platform='gpu' runs -- the customized
            # operator is only evaluated host-side (engine_base::post_newtonloop /
            # the adjoint driver) via customize_block_idxs; its block_idxs entry is
            # empty, so the GPU engine's device evaluation loop skips it.
            customized_component_itor, _ = self.physics.create_interpolator(customized_component_etor,
                                                                         n_ops=1,
                                                                         platform='cpu', algorithm='multilinear',
                                                                         precision='d',
                                                                         timer_name='customized component interpolation')
            self.physics.engine.customize_operator = self.customize_new_operator

            self.op_list = [self.physics.acc_flux_itor[0], customized_component_itor]

            # specify the index of blocks of customized operator
            idx_in_op_list = 1
            op_num_new = np.array(self.reservoir.mesh.op_num, copy=True)
            op_num_new[:] = idx_in_op_list  # set the second interpolator (i.e. "customized_component_itor") from "self.op_list" to all blocks
            self.physics.engine.idx_customized_operator = idx_in_op_list
            self.physics.engine.customize_op_num = index_vector(op_num_new)
        else:
            # self.op_list = [self.physics.acc_flux_itor]

            self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            n_res = self.reservoir.mesh.n_res_blocks
            self.op_num[n_res:] = 1
            self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]

    def run(self, export_to_vtk=False, file_name='data'):
        import random
        # use seed to generate the same random values every run
        random.seed(3)
        if export_to_vtk:

            well_loc = np.zeros(self.reservoir.n)
            for inj in self.inj_list:
                well_loc[(inj[1] - 1) * self.reservoir.nx + inj[0] - 1] = -1

            for prod in self.prod_list:
                well_loc[(prod[1] - 1) * self.reservoir.nx + prod[0] - 1] = 1

            self.global_data = {'well location': well_loc}

            self.export_vtk(file_name, global_cell_data=self.global_data)

        # now we start to run for the time report--------------------------------------------------------------
        time_step = self.report_step
        even_end = int(self.T / time_step) * time_step
        time_step_arr = np.ones(int(self.T / time_step)) * time_step
        if self.T - even_end > 0:
            time_step_arr = np.append(time_step_arr, self.T - even_end)

        for ts in time_step_arr:
            from darts.engines import well_control_iface
            for i, w in enumerate(self.reservoir.wells):
                if "I1" in w.name:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE,
                                                   is_inj=True, target=40., phase_name='gas', inj_composition=self.inj_composition)
                elif "I" in w.name:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=True, target=140., inj_composition=self.inj_composition)
                else:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=False, target=50.)

            DartsModel.run(self, ts, save_well_data=False, save_reservoir_data=False, verbose=export_to_vtk)
            self.physics.engine.report()
            if export_to_vtk:
                self.export_vtk(file_name)


class customized_etor_specific_component(operator_set_evaluator_iface):
    n_ops = 1

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """

        # temp = self.temperature.evaluate(state)
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        # values[0] = state[0]  # pressure
        values[0] = 1 - state[1]  # comp_1
        return 0
