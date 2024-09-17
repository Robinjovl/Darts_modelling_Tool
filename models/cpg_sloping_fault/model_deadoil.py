import numpy as np

from model_cpg import Model_CPG, fmt

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic

from darts.engines import value_vector
class ModelPropertiesDeadOil(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z,
                         temperature=1.)

    def run_flash(self, pressure, temperature, zc):
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1
        self.nu = zc
        ph = [0, 1]
        return ph
class ModelDeadOil(Model_CPG):
    def __init__(self, case='generate', grid_out_dir=None, n_points=400):
        super().__init__(physics_type='dead_oil', case=case, grid_out_dir=grid_out_dir, n_points=n_points)
    def set_physics(self):
        self.zero = 1e-13
        components = ["w", "o"]
        phases = ["wat", "oil"]

        self.inj = value_vector([1 - self.zero])
        self.ini = value_vector([self.zero])

        property_container = ModelPropertiesDeadOil(phases_name=phases, components_name=components, min_z=self.zero/10)

        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=700))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.89)),
                                                ('oil', ConstFunc(1))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("wat", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])

        # create physics
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=self.n_points,
                                     min_p=0, max_p=1000,
                                     min_z=self.zero, max_z=1 - self.zero)
        self.physics.add_property_region(property_container)

        self.P_initial = np.zeros(self.reservoir.mesh.n_res_blocks) + 400.

        # uniform initial conditions
        self.initial_values = {self.physics.vars[0]: 400,
                               self.physics.vars[1]: self.ini}

    def set_initial_conditions(self):

        self.initial_values['pressure'] = 1 # bars at surface
        super().set_initial_conditions(gradient={'pressure': 0.1})  # bar/m

        depth_array = np.array(self.reservoir.mesh.depth, copy=False)[:self.reservoir.mesh.n_res_blocks]
        water_table_depth = depth_array.mean()

        Sw_initial = np.zeros(self.reservoir.mesh.n_res_blocks) + 0.1
        Sw_initial[depth_array > water_table_depth] = 0.9
        def sat_to_z(p, s):
            # find composition corresponding to particular saturation
            z_range = np.linspace(self.zero, 1 - self.zero, 200)
            for z in z_range:
                # state is pressure and 1 molar fractions out of 2
                state = [p, z]
                sat = self.physics.property_containers[0].compute_saturation_full(state)
                if sat > s:
                    break
            return z

        Z_initial = np.zeros(self.reservoir.mesh.n_res_blocks)
        for ith_cell in range(self.reservoir.mesh.n_res_blocks):
            Z_initial[ith_cell] = sat_to_z(self.P_initial[ith_cell], Sw_initial[ith_cell])
        self.initial_values = {self.physics.vars[0]: self.P_initial, self.physics.vars[1]: Z_initial}

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):  # INJ well
                # BHP control
                w.control = self.physics.new_bhp_inj(450, self.inj)  # bars
                # rate control
                #w.control = self.physics.new_rate_inj(200, self.inj, 0)  # Kmol/day, composition, composition-index
                #w.constraint = self.physics.new_bhp_inj(450, self.inj)   # bars, composition
            else:  # PROD well
                # BHP control
                w.control = self.physics.new_bhp_prod(350)  # bars
                # rate control
                #w.control = self.physics.new_rate_prod(200)   # Kmol/day
                #w.constraint = self.physics.new_bhp_prod(350) # bars

    def get_arrays(self):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        nv = self.physics.n_vars
        nb = nv * self.reservoir.mesh.n_res_blocks
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[:nb:nv]

        print('P range [bars]:', fmt(P.min()), '-', fmt(P.max()))

        return {'PRESSURE': P}