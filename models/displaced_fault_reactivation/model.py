from darts.models.thmc_model import THMCModel
from darts.engines import value_vector, sim_params, mech_operators, rsf_props, sd_props, friction, contact_state, state_law, contact_solver, critical_stress, normal_condition
from reservoir import UnstructReservoir
import numpy as np
from darts.engines import vector_linear_solver_params, linear_solver_params

from darts.physics.super.property_container import PropertyContainer
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
        super().__init__(n_points=256, discretizer=self.discretizer_name)
    def set_physics(self):
        self.fluid_compressibility = 1.e-6
        self.rock_density0 = 2650.0
        self.fluid_density0 = 1020.0
        self.reference_pressure = 130.67381893933677
        self.fluid_viscosity = 1.

        self.zero = 1e-13
        n_points = 501
        Mw = [18.015]
        components = ['H2O']
        phases = ['wat']
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=Mw, min_z=self.zero,
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
        self.physics = Poroelasticity(components=components, phases=phases, timer=self.timer, n_points=n_points,
                                      min_p=-10, max_p=1000, min_z=self.zero, max_z=1 - self.zero, discretizer=self.discretizer_name)
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
    def set_solver_params(self):
        self.params.tolerance_newton = 1e-6 # Tolerance of newton residual norm ||residual||<tol_newt
        self.params.newton_type = sim_params.newton_local_chop  # Type of newton method (related to chopping strategy?)
        self.params.newton_params = value_vector([0.2])  # Probably chop-criteria(?)
        if self.friction_law == 'rsf':
            self.params.max_i_newton = 20
        else:
            self.params.max_i_newton = 8

        ls1 = linear_solver_params()
        ls1.linear_type = sim_params.cpu_superlu
        self.physics.engine.ls_params.append(ls1)

        # for iterative preconditioner need to repeat AMG setup as Juu is changing
        if ls1.linear_type == sim_params.cpu_gmres_fs_cpr:
            m.physics.engine.update_uu_jacobian()

        # different solver for dynamic simulation
        if self.enable_dynamic_mode:
            ls2 = linear_solver_params()
            ls2.linear_type = sim_params.cpu_gmres_ilu0
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
            self.id_prod = np.linalg.norm(centroids - pt_right, axis=1).argmin()

            self.reservoir.add_well("PROD001", depth=self.reservoir.depth[self.id_prod])
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
            self.physics.set_well_controls(well=w, is_control=True, control_type=well_control_iface.BHP,
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
