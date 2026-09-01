from darts.models.thmc_model import THMCModel
from darts.engines import value_vector, sim_params, mech_operators, rsf_props, sd_props, friction, contact_state, state_law, contact_solver, critical_stress, normal_condition
from reservoir import UnstructReservoir
import numpy as np
from darts.engines import vector_linear_solver_params, linear_solver_params
from darts.nonlinear_solvers import MechanicsNewtonSolver


class DisplacedFaultNewtonSolver(MechanicsNewtonSolver):
    """Mechanics Newton driver specialized for the fault-reactivation model:
    the third residual component is the contact-gap residual ``dev_g``, the
    timestep fails early when it exceeds the contact-residual cut-off, and the
    dynamic-mode slip-area gating runs after the loop. (Replaces the copied
    ``run_timestep_python`` loop; the base loop lives in MechanicsNewtonSolver.)"""

    def compute_mech_residual(self):
        engine = self.engine
        res = engine.calc_newton_dev()
        engine.dev_p = res[0]
        engine.dev_u = res[1]
        engine.dev_g = res[2] if len(res) > 2 and res[2] == res[2] else 0.0
        return res[0], res[1], engine.dev_g

    def on_mech_iteration(self, i, dev_p, dev_u, dev_third, well_residual):
        print(
            f"{i}: rp = {dev_p}\tru = {dev_u}\trg = {dev_third}"
            f"\trwell = {well_residual}"
        )

    def check_early_break(self, i) -> bool:
        if self.engine.dev_g > self.model.cut_off_gap_residual:
            print(
                'Restart newton iterations due to exceed of contact residual '
                'cut-off exceeded!!!'
            )
            return True
        return False

    def finalize_convergence(self, converged: int) -> int:
        m = self.model
        if not hasattr(m, 'slip_area'):
            m.slip_area = [0.0]
        cur_area = m.reservoir.calc_slip_areas(engine=m.physics.engine)[0]  # one fault
        print('slip area = ' + str(cur_area))
        if m.enable_dynamic_mode:
            if cur_area - m.slip_area[-1] > 4.2 * m.min_area:
                converged *= 0
            else:
                m.slip_area.append(cur_area)
                converged *= 1
        return converged

from darts.physics.base.property_container import PropertyContainer
from darts.physics.mech.poroelasticity import Poroelasticity
from darts.physics.properties.flash import SinglePhase
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

class Model(THMCModel):
    def __init__(self, config):
        self.physics_type = 'poromechanics'
        self.discretizer_name = 'pm_discretizer'
        self.enable_dynamic_mode = True if config['mode'] != 'quasi_static' else False
        self.max_newt_it_dynamic_mode = 100
        self.depletion_mode = config['depletion']['mode']
        self.depletion_value = config['depletion']['value']
        self.friction_law = config['friction_law']
        self.mesh_file = config['mesh_file']
        if 'cache_discretizer' in config:
            self.cache_discretizer = config['cache_discretizer']
        else:
            self.cache_discretizer = True
        super().__init__()
    def set_physics(self):
        self.fluid_compressibility = 1.e-6
        self.rock_density0 = 2650.0
        self.fluid_density0 = 1020.0
        self.reference_pressure = 130.67381893933677
        self.fluid_viscosity = 1.

        self.zero = 1e-13
        Mw = [18.015]
        components = ['H2O']
        phases = ['wat']
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=Mw, eps_z=self.zero,
                                               temperature=323.15)
        """ properties correlations """
        property_container.flash_ev = SinglePhase(nc=1)
        property_container.density_ev = dict([('wat', DensityBasic(compr=self.fluid_compressibility,
                                                                   dens0=self.fluid_density0,
                                                                   p0=self.reference_pressure))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(self.fluid_viscosity))])

        property_container.rel_perm_ev = dict([('wat', ConstFunc(1.0))])
        # rock compressibility is treated inside engine
        property_container.rock_compr_ev = ConstFunc(1.0)

        property_container.rock_density_ev = ConstFunc(self.rock_density0)
        # create physics
        self.physics = Poroelasticity(components=components, phases=phases, timer=self.timer,
                                      axes_step=[2.0], axes_origin=[-10.],
                                      epsilon_z=self.zero/10, discretizer=self.discretizer_name)
        self.physics.add_property_region(property_container)
        self.physics.init_physics(discr_type=self.discretizer_name, platform='cpu')

        return
    def set_reservoir(self):
        self.reservoir = UnstructReservoir(timer=self.timer, fluid_density=self.fluid_density0,
                                           rock_density=self.rock_density0, mesh_file=self.mesh_file,
                                           cache_discretizer=self.cache_discretizer)
    def update_pressure(self, dt, time):
        if self.enable_dynamic_mode or self.friction_law == 'rsf':
            dp_rate = self.depletion_value
            p = lambda x: dp_rate * dt
        else:
            dp = self.depletion_value
            p = lambda x: dp

        if time == dt or self.enable_dynamic_mode or self.friction_law == 'rsf':
            X = np.asarray(self.physics.engine.X)
            Xn = np.asarray(self.physics.engine.Xn)
            for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
                if cell.prop_id == 99991 or cell.prop_id == 99993:
                    X[4 * cell_id + 3] += p(cell.centroid[0])
                    Xn[4 * cell_id + 3] += p(cell.centroid[0])
                    #Xref[4 * cell_id + 3] = 100.0
                    #Xn_ref[4 * cell_id + 3] = 100.0
            for cell_id, cell in self.reservoir.unstr_discr.frac_cell_info_dict.items():
                if cell.centroid[1] >= -150.0 and cell.centroid[1] <= 150.0:
                    X[4 * cell_id + 3] += p(cell.centroid[0])
                    Xn[4 * cell_id + 3] += p(cell.centroid[0])
    def set_solver(self):
        # Open-source FS-CPR by default (pm_discretizer / engine_pm_cpu). The
        # spec-built solver is injected via set_linear_solver and now genuinely
        # drives the open-source solve (engine_pm_cpu prefers the external
        # solver over its ls_params bank); ls_params remains the
        # proprietary-build / factory path. Mid-run changes (e.g. the dynamic
        # rupture stage in main.py) go through model.linear_solver.update_solver().
        from darts.linear_solvers.specs import FSCPRSolverSpec, GMRESSolverSpec
        mesh = self.reservoir.mesh
        n_res_blks = mesh.n_res_blocks
        n_matrix = getattr(self.reservoir, 'n_matrix', n_res_blks)
        n_fracs_mesh = getattr(self.reservoir, 'n_fracs', 0)
        # engine_pm_cpu variable layout: displacement first (U_VAR=0), pressure at
        # ND=3, no composition variable (Z_VAR=255). Without these overrides the
        # FS-CPR splits the engine_super_elastic_cpu default layout (P_VAR=0) and
        # mis-identifies the pressure/displacement subsystems -- a wrong
        # preconditioner that stalls convergence (and hangs when GMRES can't
        # compensate). Read the indices off the engine so this stays correct if
        # the conventions change. NE = N_VARS - ND = NC for the isothermal cases.
        engine = self.physics.engine
        fs_cpr = FSCPRSolverSpec(
            force_amg_asymmetric=True,
            n_res=n_matrix + n_fracs_mesh,
            n_fracs=0,
            n_wells=mesh.n_blocks - n_res_blks,
            p_var=engine.P_VAR,
            z_var=engine.Z_VAR,
            u_var=engine.U_VAR,
            nc=engine.N_VARS - 3,
        )
        # 1e-10 / 500, not the sim_params defaults (1e-5 / 50). Until !280 this model
        # ran a *direct* solve (ls_params[0] = cpu_superlu), and ref/*/solution_fault1.vtu
        # was generated with it; main.py compares the fault data at rtol 1e-6 / atol 1e-8.
        # The slip-weakening case amplifies the linear residual: mu depends on the slip g,
        # so the momentum-residual floor propagates straight into mu and f_local. At
        # 1e-5 / 50 GMRES+FS-CPR leaves ||ru|| ~ 4e-9 (vs ~2e-14 for the direct solve),
        # which moves mu by ~1.3e-6 and f_local by ~5e-4 -- over the comparison tolerance.
        # At 1e-10 / 500 the Newton path matches the direct solve exactly (NI = 5, same
        # residuals to ~10 digits) and every fault field is within isclose(1e-6, 1e-8) of
        # it, at ~20% more wall time. The static case is insensitive (constant mu) and
        # passes either way. main.py tightens this further (1e-12 / 500) for the dynamic
        # rupture stage via update_solver().
        self.linear_solver.spec = GMRESSolverSpec(prec=fs_cpr, tolerance=1e-10, max_iterations=500, restart=50)
        self.solver_phase = 'static'  # main.py flips to 'dynamic' at rupture

        # Mechanics model: the LINEAR solver comes from params.linear_type /
        # engine.ls_params (THMCModel.linear_solver_from_engine_factory), so the
        # flow CPR/AMG default is not applied; the NONLINEAR solver is configured
        # on its spec below. Called from the base reset(), before engine.init.
        super().set_solver()
        self.nonlinear_solver.spec.tolerance = 1e-6 # Tolerance of newton residual norm ||residual||<tol_newt
        self.nonlinear_solver.spec.chop.mode = 'local'  # Type of newton method (related to chopping strategy?)
        self.nonlinear_solver.spec.chop.factor = 0.2  # Probably chop-criteria(?)
        if self.friction_law == 'rsf':
            self.nonlinear_solver.spec.max_iterations = 20
        else:
            self.nonlinear_solver.spec.max_iterations = 8

        # swap the runtime to the fault-specialized mechanics Newton driver
        # (contact-gap residual + cut-off + slip-area gating), reusing the spec
        self.nonlinear_solver = DisplacedFaultNewtonSolver(self.nonlinear_solver.spec)
        self.nonlinear_solver.bind(self)

        # Idempotent: ls_params is appended once even though set_solver() runs on every reset().
        if len(self.physics.engine.ls_params) == 0:
            ls1 = linear_solver_params()
            # Placeholder in the open-source build (the FS-CPR spec drives the solve,
            # and the neutralised cpu_gmres_fs_cpr factory path crashes there); real
            # selector (bos_fs_cpr) in the proprietary build.
            ls1.linear_type = (sim_params.cpu_superlu if self.linear_solver.open_source_solvers_available()
                               else sim_params.cpu_gmres_fs_cpr)
            self.physics.engine.ls_params.append(ls1)

            # for iterative preconditioner need to repeat AMG setup as Juu is changing
            if ls1.linear_type == sim_params.cpu_gmres_fs_cpr:
                self.physics.engine.update_uu_jacobian()

            # different solver for dynamic simulation -- works with BOS-solvers build only.
            # There, main.py switches the engine-side solver at rupture
            # (active_linear_solver_id = 1). The open-source build reaches the same
            # dynamic-stage settings by reconfiguring the injected GMRES+FS-CPR stack in
            # place (main.py: update_solver(tolerance=1e-12, max_iterations=500)) and never
            # selects ls_params[1], so it must not append this entry: engine_pm_cpu keeps
            # the cpu_gmres_ilu0 case behind #ifndef OPENDARTS_LINEAR_SOLVERS, so the entry
            # would match no case, nothing would be appended to the engine's linear_solvers
            # and engine.init() -- which indexes that by the ls_params index --
            # would read past its end. There is currently no standalone open-source ILU(0)
            # to name here either: the in-tree block ILU(0) is reachable only as the CPR
            # stage-2 smoother, not as a registry solver.
            if self.enable_dynamic_mode and not self.open_source_solvers_available():
                ls2 = linear_solver_params()
                # Same placeholder rule as ls1: cpu_gmres_ilu0 is compiled out of
                # the open-source engine factory, which would now raise instead of
                # silently desyncing ls_params from the solver bank.
                ls2.linear_type = (sim_params.cpu_superlu if self.open_source_solvers_available()
                                   else sim_params.cpu_gmres_ilu0)
                ls2.tolerance_linear = 1.e-12
                ls2.max_i_linear = 500
                self.physics.engine.ls_params.append(ls2)
    def set_wells(self):
        if self.depletion_mode == 'well':
            well_index = 1.E+10
            x = 2000.0
            centroids = np.array([c.centroid for c in self.reservoir.unstr_discr.mat_cell_info_dict.values()])

            pt_left = np.array([-x, (self.reservoir.a - self.reservoir.b) / 2, 0.0])
            self.id_inj = np.linalg.norm(centroids - pt_left, axis=1).argmin()

            # self.reservoir.add_well("INJ001", depth=self.reservoir.depth[self.id_inj])
            # self.reservoir.add_perforation(self.reservoir.wells[-1], int(self.id_inj),
            #                                well_index=well_index)

            pt_right = np.array([x, (-self.reservoir.a + self.reservoir.b) / 2, 0.0])
            print('well perforation location prod:', pt_right)
            self.id_prod = np.linalg.norm(centroids - pt_right, axis=1).argmin()

            self.reservoir.add_well("PROD001", depth=self.reservoir.unstr_discr.depth_all_cells[self.id_prod])
            self.reservoir.add_perforation(self.reservoir.wells[-1], int(self.id_prod),
                                           well_index=well_index)
    def set_input_data(self):
        pass
    def set_initial_conditions(self):
        #self.physics.set_uniform_initial_conditions(self.reservoir.mesh,
        #                                            uniform_pressure=self.reservoir.p_init,
        #                                            uniform_displacement=self.reservoir.u_init)
        self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                    input_distribution={'pressure': self.reservoir.p_init},
                                                    input_displacement=self.reservoir.u_init)
        return 0
    def set_boundary_conditions(self):
        """
        Class method called in the init() class method of parents class
        """
        from darts.engines import well_control_iface
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:
        for i, w in enumerate(self.reservoir.wells):
            # if i == 0:
            # #     # For BHP control in injection well we usually specify pressure and composition (upstream) but here
            # #     # the method is wrapped such  that we only need to specify bhp pressure (see lambda for more info)
            #     w.control = self.physics.new_bhp_inj(self.reservoir.p_init[self.id_inj])
            #     #w.control = self.physics.new_rate_inj(0.0)
            # else:
            #     # Add controls for production well:
            #     # Specify bhp for particular production well:
            self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                           is_inj=False, target=self.reservoir.p_init[self.id_prod] + self.depletion_value)
        return 0
    def setup_contact_friction(self, contact_algorithm: contact_solver):
        if hasattr(self.reservoir, 'contacts'):
            for contact in self.physics.engine.contacts:
                if self.friction_law == 'static':
                    friction_model = friction.STATIC # friction.STATIC # friction.SLIP_DEPENDENT # friction.RSF
                elif self.friction_law == 'slip_weakening':
                    friction_model = friction.SLIP_DEPENDENT # friction.STATIC # friction.SLIP_DEPENDENT # friction.RSF
                elif self.friction_law == 'rsf':
                    friction_model = friction.RSF  # friction.STATIC # friction.SLIP_DEPENDENT # friction.RSF

                # allow to slip
                contact.set_state(contact_state.SLIP)
                # static friction coefficients
                mu0 = 0.52 * np.ones(len(contact.cell_ids))  # initial friction coefficient
                contact.mu0 = value_vector(mu0)
                contact.mu = contact.mu0

                # setup friction model
                contact.friction_model = friction_model
                # setup friction criterion
                contact.friction_criterion = critical_stress.BIOT
                # setup normal condition
                contact.normal_condition = normal_condition.ZERO_GAP_CHANGE

                # Slip dependent model
                if (friction_model == friction.SLIP_DEPENDENT):
                    prop = sd_props()
                    prop.crit_distance = 0.02
                    prop.mu_dyn = 0.2
                    contact.sd_props = prop
                # RSF model
                if (friction_model == friction.RSF or friction_model == friction.RSF_STAB):
                    prop = rsf_props()
                    prop.min_vel = 1.E-13 * 86400

                    prop.a = 0.001 # 0.015#0.008#-0.001  # 0.0008#0.0078
                    prop.b = 0.03 # 0.03
                    prop.crit_distance = 0.02
                    prop.ref_velocity = 1.E-10 * 86400  #0.01 * 1.E-6 * 86400
                    prop.law = state_law.AGEING_LAW

                    # theta
                    theta = prop.crit_distance / prop.ref_velocity * np.ones(len(contact.cell_ids))
                    prop.theta_n = value_vector(theta)
                    prop.theta = value_vector(theta)
                    prop.mu_rate = value_vector(np.zeros(len(contact.cell_ids)))
                    prop.mu_state = value_vector(np.zeros(len(contact.cell_ids)))

                    contact.rsf = prop

                #contact.init_friction(self.reservoir.pm, self.reservoir.mesh)

                # Damping term
                for i in range(len(contact.eta)):
                    contact.eta[i] *= 0.0#1.0e10

                # init local solver in the case of local iterations
                if contact_algorithm == contact_solver.LOCAL_ITERATIONS:
                    contact.init_local_iterations()
