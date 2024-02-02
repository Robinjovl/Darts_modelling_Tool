from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params, mech_operators, rsf_props, friction, contact_state, state_law, contact_solver, critical_stress, linear_solver_params
from reservoir import UnstructReservoirCustom
import numpy as np
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
from darts.physics.mech.poroelasticity import Poroelasticity
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.flash import SinglePhase
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import DensityBasic

class Model(DartsModel):
    def __init__(self, mesh_file, discretizer='mech_discretizer', mode='poroelastic', n_points=64):
        super().__init__()
        self.n_points = n_points
        self.timer.node["initialization"].start()
        self.physics_type = 'poromechanics'
        self.discretizer_name = discretizer

        self.reservoir = UnstructReservoirCustom(timer=self.timer,
                                                   discretizer=self.discretizer_name,
                                                   mode=mode,
                                                   mesh_file=mesh_file)
        self.set_physics()

        self.reservoir.P_VAR = self.engine.P_VAR
        self.reservoir.U_VAR = self.engine.U_VAR
        self.params.tolerance_newton = 1e-6 # Tolerance of newton residual norm ||residual||<tol_newt
        self.params.newton_type = sim_params.newton_global_chop  # Type of newton method (related to chopping strategy?)
        self.params.newton_params = value_vector([0.2])  # Probably chop-criteria(?)
        self.params.max_i_newton = 10

        if self.discretizer_name == 'mech_discretizer':
            self.params.tolerance_linear = 1e-10  # Tolerance for linear solver ||Ax - b||<tol_linslv
            self.params.linear_type = sim_params.cpu_superlu  # cpu_gmres_fs_cpr # cpu_superlu
            self.params.max_i_linear = 5000
        elif self.discretizer_name == 'pm_discretizer':
            ls1 = linear_solver_params()
            ls1.linear_type = sim_params.cpu_superlu  # cpu_gmres_fs_cpr # cpu_superlu
            ls1.tolerance_linear = 1.e-12
            ls1.max_i_linear = 500
            self.engine.ls_params.append(ls1)
        self.timer.node["initialization"].stop()

    def set_physics(self):
        zero = 1e-8
        # Create property containers:
        components = ['H2O']
        phases = ['wat']
        Mw = [1.0]

        if self.reservoir.thermoporoelasticity:
            self.reservoir.heat_capacity = 2200.0 * 1000.0
            hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
            hcap.fill(self.reservoir.heat_capacity)
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, min_z=zero / 10)
            property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=self.reservoir.heat_capacity, tref=0.0))])
            property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0, tref=0.0)
            property_container.conductivity_ev = dict([('wat', ConstFunc(1.0))])
            # create physics
            self.physics = Poroelasticity(components, phases, self.timer, n_points=200,
                                          min_p=-5, max_p=500, min_z=zero/10, max_z=1-zero/10,
                                          thermal=True, min_t=-10.0, max_t=100.0,
                                          discretizer=self.discretizer_name)
        else:
            property_container = PropertyContainer(phases_name=phases, components_name=components,
                                                   Mw=Mw, min_z=zero / 10, temperature=1.)
            # create physics
            self.physics = Poroelasticity(components, phases, self.timer, n_points=200,
                                          min_p=-1000, max_p=1000, min_z=zero / 10, max_z=1 - zero / 10,
                                          discretizer=self.discretizer_name)

        """ properties correlations """
        property_container.flash_ev = SinglePhase(nc=1)
        property_container.density_ev = dict([('wat', DensityBasic(compr=self.reservoir.fluid_compressibility,
                                                                   dens0=self.reservoir.fluid_density))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(self.reservoir.fluid_viscosity))])
        property_container.rel_perm_ev = dict([('wat', ConstFunc(1.0))])
        property_container.rock_compr_ev = ConstFunc(1.0) # rock compressibility is treated inside engine

        self.physics.add_property_region(property_container)

        self.engine = self.physics.init_physics(discretizer=self.discretizer_name, platform='cpu')
        return

    def init(self):
        if self.discretizer_name == 'pm_discretizer':
            self.reservoir.mech_operators = mech_operators()
            self.reservoir.mech_operators.init(self.reservoir.mesh, self.reservoir.pm,
                                     self.engine.P_VAR, self.engine.Z_VAR, self.engine.U_VAR,
                                     self.engine.N_VARS, self.engine.N_OPS, self.engine.NC,
                                     self.engine.ACC_OP, self.engine.FLUX_OP, self.engine.GRAV_OP)
            self.reservoir.mech_operators.prepare()

        self.set_boundary_conditions()
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells, self.engine)
        self.set_initial_conditions()
        self.set_well_controls()
        self.set_op_list()
        self.reset()

        if self.discretizer_name == 'mech_discretizer':
            self.engine.set_discretizer(self.reservoir.discr)

        Xref = np.array(self.engine.Xref, copy=False)
        Xn_ref = np.array(self.engine.Xn_ref, copy=False)
        Xref[:] = 0.0
        Xn_ref[:] = 0.0

    def set_initial_conditions(self):
        if self.reservoir.thermoporoelasticity:
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                initial_pressure=self.reservoir.p_init,
                                                initial_temperature=self.reservoir.t_init,
                                                initial_displacement=self.reservoir.u_init)
        else:
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                initial_pressure=self.reservoir.p_init,
                                                initial_displacement=self.reservoir.u_init)
        return 0

    def set_boundary_conditions(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # Add controls for production well:
                # Specify bhp for particular production well:
                w.control = self.physics.new_bhp_prod(self.reservoir.p_init - 50)
                # w.control = self.physics.new_bhp_prod(self.reservoir.p_init)
            else:
                # For BHP control in injection well we usually specify pressure and composition (upstream) but here
                # the method is wrapped such  that we only need to specify bhp pressure (see lambda for more info)
                #w.control = self.physics.new_bhp_inj(self.reservoir.p_init + 10)
                w.control = self.physics.new_bhp_inj(self.reservoir.p_init)
        return 0