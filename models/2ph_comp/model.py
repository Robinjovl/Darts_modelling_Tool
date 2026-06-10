from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, well_control_iface, ms_well
from darts import solvers
from darts.solvers import (
    BCSRCPRSpec,
    BILU0Spec,
    LocalCorrectionSpec,
    MGRLevelSpec,
    MGRSolverSpec,
    PressureAMGSpec,
)
from darts.solvers.enums import (
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

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()
        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000, tol_newton=1e-3, tol_linear=1e-4,
                            it_newton=20, it_linear=50, newton_type=sim_params.newton_local_chop)
        self.params.linear_type = sim_params.cpu_gmres_mgr
        self.params.linear_print_level = 0  # 0 = quiet, 1 = basic (default), 2 = verbose
        self.solver = None
        self.use_bcsr_cpr_pressureguard_thr10_profile()
        # self.solver is built and injected by the base DartsModel.set_solver()
        # hook during init() (reset -> _apply_set_solver), after engine.init.

        self.timer.node["initialization"].stop()

    def use_bcsr_cpr_pressureguard_thr10_profile(self, reduction_type=None):
        self.use_mgr_cpr_pressureguard_thr10 = True
        self.bcsr_cpr_reduction_type = (
            sim_params.mgrCprReductionTrueIMPES
            if reduction_type is None
            else reduction_type
        )
        if getattr(self, "solver", None) is not None:
            self.set_solver()

    def use_bcsr_cpr_levelaware_pressureguard_thr10_profile(self):
        self.use_bcsr_cpr_pressureguard_thr10_profile(
            getattr(
                sim_params,
                "mgrCprReductionTrueIMPESWellElim",
                sim_params.mgrCprReductionTrueIMPES,
            )
        )

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
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=200, min_p=1, max_p=300, min_z=0., max_z=1., epsilon_z=epsilon,
                                     extrapolation_flag=True)
        # property_container.output_props = {
        #     "sat0": lambda: property_container.sat[0],
        #     "dens0": lambda: property_container.dens[0],
        #     "nu0": lambda: property_container.nu[0],
        #     "x00": lambda: property_container.x[0,0]
        #     }

        self.physics.add_property_region(property_container)

        return

    def set_sim_params(self, *args, **kwargs):
        super().set_sim_params(*args, **kwargs)

        self.data_ts.linear_type = sim_params.cpu_gmres_mgr
        if self.data_ts.linear_print_level is None:
            self.data_ts.linear_print_level = 0

        self.params.linear_type = sim_params.cpu_gmres_mgr
        self.params.linear_print_level = self.data_ts.linear_print_level

        if getattr(self, "use_mgr_cpr_pressureguard_thr10", False):
            self.set_solver()

    def set_solver(self):
        # MGR (BCSR-CPR) via the unified spec API (self.solver = MGRSolverSpec).
        # The base DartsModel._apply_solver hook builds + injects it before
        # engine.init on the open-source CPU build; in proprietary / GPU builds the
        # spec is ignored and the engine factory selects from params.linear_type
        # (set below for proprietary, and to cpu_gmres_mgr in __init__), so this is
        # build-safe everywhere. Verified bit-for-bit against the former raw MGR
        # build: TS=1009 / NI=2234 / LI=4188 (see verify_mgr_spec.py).
        if not self.open_source_solvers_available():
            self.solver = None
            self.params.linear_type = sim_params.cpu_gmres_cpr
            return

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

        self.solver = MGRSolverSpec(
            tolerance=self.params.tolerance_linear,
            max_iterations=self.params.max_i_linear,
            log_level=self.params.linear_print_level,
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
                # Parametrised by the use_bcsr_cpr_*_profile() methods; the
                # sim_params.mgrCprReduction* int values equal the enum values.
                reduction_type=getattr(
                    self, "bcsr_cpr_reduction_type", BCSRCPRReduction.TRUE_IMPES
                ),
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
        self.solver_label = "mgr (bcsr-cpr)"
        return

    # The MGRSolverSpec above is built and injected by the base
    # DartsModel._apply_solver(stage="pre") hook (called from reset(), before
    # engine.init). No init() override is needed -- self.solver_label names it in
    # the engine log.

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 50,
                              self.physics.vars[1]: 0.1,
                              self.physics.vars[2]: 0.2
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        zero = self.physics.axes_min[1]
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
