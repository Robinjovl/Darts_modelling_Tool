from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, well_control_iface, ms_well
from darts.nonlinear_solvers import ChopSpec, NewtonSolver
from darts import linear_solvers
from darts.linear_solvers import (
    BCSRCPRSpec,
    BILU0Spec,
    LinearSolver,
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
import numpy as np

from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic


class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()
        # Time-stepping and linear-solver configuration live in set_solver(),
        # which the base reset() calls before engine.init (see the unified
        # self.linear_solver.spec = <LinearSolverSpec> API).

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=10, dz=10,
                                         permx=100, permy=100, permz=10, poro=0.3, depth=1000)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1))

    def set_physics(self):
        zero = 1e-8
        epsilon = 1e-9

        components = ['CO2', 'C1', 'H2O']
        phases = ['gas', 'aqueous']
        Mw = [44.01, 16.04, 18.015]

        # Create a property container
        property_container = PropertyContainer(phases_name=phases, components_name=components,
                                               Mw=Mw, eps_z=epsilon, temperature=1.)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(components), [4, 2, 1e-1], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('aqueous', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('aqueous', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('aqueous', PhaseRelPerm("oil"))])

        """ Activate physics """
        thermal = False
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        # axes_step-based API: per-axis cell size + per-axis origin. The adaptive
        # multi-index-keyed interpolator caches cells on demand wherever the solver
        # lands; no axes_max, no n_points, no min_p needed.
        p_step = (300 - 1) / (200 - 1)
        z_step = (1 - 3 * epsilon) / (200 - 1)
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[p_step, z_step, z_step],
                                     axes_origin=[1.0, epsilon, epsilon],
                                     epsilon_z=epsilon,
                                     extrapolation_flag=True)
        # property_container.output_props = {
        #     "sat0": lambda: property_container.sat[0],
        #     "dens0": lambda: property_container.dens[0],
        #     "nu0": lambda: property_container.nu[0],
        #     "x00": lambda: property_container.x[0,0]
        #     }

        self.physics.add_property_region(property_container)

        return

    def set_solver(self):
        # Single per-model home for time-stepping / Newton + linear-solver config
        # (the unified set_solver() pattern). Called by the base reset() before
        # engine.init, so these settings feed engine.init().
        self.linear_solver = LinearSolver(model=self)
        self.linear_solver.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000 )
        self.params.linear_print_level = 0  # 0 = quiet, 1 = basic, 2 = verbose

        # MGR (BCSR-CPR) via the single unified spec API (self.linear_solver.spec = MGRSolverSpec).
        # The base DartsModel._apply_solver hook builds + injects it before engine.init
        # on the open-source CPU build. On the proprietary build the spec is not built;
        # _apply_solver instead applies proprietary_linear_type (cpu_gmres_cpr_amg) to
        # params.linear_type, so this is the only place the solver is declared and it
        # stays build-safe everywhere. Verified bit-for-bit against the former raw MGR
        # build: TS=1009 / NI=2234 / LI=4188 (see verify_mgr_spec.py).

        # block_size = 1 (pressure) + (n_components - 1) fractions
        block_size = self.physics.n_vars
        mesh = getattr(self.reservoir, "mesh", None)
        reservoir_blocks = mesh.n_res_blocks if mesh is not None else 0

        reservoir_roles = [VariableRole.PRESSURE] + [VariableRole.COMPOSITION] * (
            block_size - 1
        )
        well_roles = [VariableRole.WELL_PRESSURE] + [VariableRole.WELL_SECONDARY] * (
            block_size - 1
        )

        # Configured directly on self.linear_solver (constructed above) -- no
        # platform default is ever materialized and discarded.
        self.linear_solver.spec = MGRSolverSpec(
            tolerance=1e-4,
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
                # True-IMPES pressure-equation reduction (== sim_params.mgrCprReductionTrueIMPES).
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
                diagnostics=True,
                diagnostic_apply_interval=100,
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
        self.linear_solver.label = "mgr (bcsr-cpr)"
        super().set_solver()  # platform default nonlinear solver
        # NOTE: 1e-3 / 20 (not the historic 1e-2 / 10) -- tightened on this branch by
        # commit 34b55809a 'Fix passing parameters from Python'; the nonlinear
        # refactoring (!327) carried the older values into the NewtonSolver form,
        # so the merge restores ours. verify_mgr_spec.py compares against these.
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3, max_iterations=20, chop=ChopSpec(mode='local'))
        return

    # The MGRSolverSpec above is built and injected by the base
    # LinearSolver._apply_solver() hook (called from reset(), before engine.init).
    # self.linear_solver.label names it in the engine log.

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 50,
                              self.physics.vars[1]: 0.1,
                              self.physics.vars[2]: 0.2
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        zero = self.physics.axes_origin[1]
        inj_composition = [1.0 - 2 * zero*10, zero*10]
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=140., inj_composition=inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=50.)
                # Control total mass rate of the produced fluid
                # self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MASS_RATE,
                #                                is_inj=False, target=4000)
