from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, well_control_iface, ms_well
from darts import solvers
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
        self.set_solver()

        self.timer.node["initialization"].stop()

    def use_bcsr_cpr_pressureguard_thr10_profile(self):
        self.use_mgr_cpr_pressureguard_thr10 = True
        if getattr(self, "solver", None) is not None:
            self.set_solver()

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
        """Physical properties"""
        zero = 1e-8
        epsilon = 1e-9
        # Create property containers:
        components = ['CO2', 'C1', 'H2O']
        phases = ['gas', 'oil']
        thermal = 0
        Mw = [44.01, 16.04, 18.015]

        property_container = PropertyContainer(phases_name=phases, components_name=components,
                                               Mw=Mw, eps_z=epsilon, temperature=1.)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(components), [4, 2, 1e-1], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('oil', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil"))])

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
        # Create MGR solver with correct block size (pressure + n_components - 1)
        # n_vars = 1 (pressure) + len(components) - 1 (component fractions)
        block_size = self.physics.n_vars  # pressure + (n_components - 1) fractions = n_components
        self.solver = solvers.create_mgr_solver_for_block_size(block_size)

        mesh = getattr(self.reservoir, "mesh", None)
        reservoir_blocks = mesh.n_res_blocks if mesh is not None else None

        self.solver.set_max_iterations(self.params.max_i_linear)
        self.solver.set_tolerance(self.params.tolerance_linear)
        self.solver.set_log_level(self.params.linear_print_level)
        self.solver.set_kdim(150)
        self.solver.set_use_mgr(True)
        self.solver.set_use_flex_gmres(True)
        self.solver.set_use_physics_scaling(True)
        self.solver.set_mgr_composite_mode(1)

        self.solver.set_mgr_local_solver(
            getattr(sim_params, "mgrLocalSolverBlockILU0", 2)
        )
        self.solver.set_mgr_bilu0_pivot_shift(1e-12)
        self.solver.set_mgr_bilu0_fallback_options(
            sim_params.mgrBilu0FallbackIdentity,
            1e-4,
            1e-4,
            100.0,
        )
        self.solver.set_mgr_local_correction_options(1.0, -1.0, 0.0, -1.0, 0.0)
        self.solver.set_mgr_local_correction_quality_options(False, 0.0)

        self.solver.set_mgr_pressure_amg_options(6, 6, 6, 1, 6, 20, 1)
        self.solver.set_mgr_pressure_amg_solve_options(1, 0.0)

        self.solver.set_use_bcsr_cpr(True)
        self.solver.set_bcsr_cpr_options(
            sim_params.mgrCprReductionTrueIMPES,
            0,
            1e6,
        )
        self.solver.set_bcsr_cpr_reuse_options(True, 0)
        self.solver.set_bcsr_cpr_adaptive_rebuild_options(True, 15, 1.5, 1, 2)
        self.solver.set_bcsr_cpr_adaptive_quality_options(-1.0, -1.0, -1.0)
        self.solver.set_bcsr_cpr_diagnostics_options(True, 100, 0)
        self.solver.set_bcsr_cpr_pressure_correction_options(1.0, 10.0, 0.05)

        reservoir_roles = [sim_params.mgrVarPressure] + [
            sim_params.mgrVarComposition
        ] * (block_size - 1)
        well_roles = [sim_params.mgrVarWellPressure] + [
            sim_params.mgrVarWellSecondary
        ] * (block_size - 1)
        self.solver.set_mgr_reservoir_variable_roles(reservoir_roles)
        self.solver.set_mgr_well_variable_roles(well_roles)
        self.solver.set_mgr_pressure_level_options(
            sim_params.mgrFRelaxNone,
            0,
            sim_params.mgrInterpInjection,
            sim_params.mgrRestrictBlockColLumped,
            sim_params.mgrCoarseGalerkin,
            sim_params.mgrSmootherHypreILU,
            1,
        )

        if reservoir_blocks is not None:
            self.solver.set_n_reservoir_blocks(reservoir_blocks)

        self.solver.set_mgr_enable_well_level(False)
        self.solver.set_mgr_enable_composition_level(False)

        return

    def init(self, *args, **kwargs):
        """Override init to set solver before engine initialization"""
        # The engine is created during physics.init_physics() in the base init()
        # init_base() checks for linear_solver_external, so if we set it before init_base runs,
        # it will use our solver. However, engine.init() -> init_base() is called from within
        # super().init(), so we can't easily intercept.
        #
        # Solution: The C++ code now checks for linear_solver_external in init_base,
        # so we need to set it before init_base runs. Since we can't do that directly,
        # we'll set it as early as possible - right after physics.init_physics() creates the engine.
        # But that happens inside super().init().
        #
        # For now, we'll set it after super().init() completes. The init_base will have
        # already created a solver, but set_linear_solver will replace it. This works
        # because set_linear_solver handles cleanup of the old solver.
        super().init(*args, **kwargs)

        # Set the solver on the engine - this replaces any solver created in init_base
        if hasattr(self, 'solver') and self.solver is not None:
            if hasattr(self.physics, 'engine') and self.physics.engine is not None:
                self.solver.set_n_reservoir_blocks(self.reservoir.mesh.n_res_blocks)
                self.physics.engine.set_linear_solver(self.solver)

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
