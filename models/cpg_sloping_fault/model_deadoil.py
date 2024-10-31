import numpy as np
import pandas as pd
from scipy import interpolate

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
        ph = np.array([0, 1])
        for i in ph:
            self.x[i, i] = 1
        self.nu = zc
        return ph

    def evaluate_at_cond(self, pressure, zc):

        self.nu = zc
        for j in [0, 1]:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.sat = self.nu / self.dens_m

        return self.sat, self.dens_m

class ModelDeadOil(Model_CPG):
    def __init__(self, case='generate', grid_out_dir=None):
        self.n_points = 400  # OBL points
        super().__init__(physics_type='dead_oil', case=case, grid_out_dir=grid_out_dir)

    def set_physics(self):
        self.zero = 1e-13
        components = ["zoil", "zwat"]
        phases = ["oil", "water"]

        self.inj = value_vector([self.zero])  # injection composition - water
        self.ini = value_vector([1 - self.zero])  # initial composition (above water table depth) - oil

        property_container = ModelPropertiesDeadOil(phases_name=phases, components_name=components, min_z=self.zero/10)

        property_container.density_ev = dict([('water', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=700))])
        property_container.viscosity_ev = dict([('water', ConstFunc(0.89)),
                                                ('oil', ConstFunc(1))])
        property_container.rel_perm_ev = dict([('water', PhaseRelPerm("water", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])

        # create physics
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=self.n_points,
                                     min_p=0, max_p=1000,
                                     min_z=self.zero, max_z=1 - self.zero)
        self.physics.add_property_region(property_container)

    def set_initial_conditions(self):  # override origin set_initial_conditions function from darts_model
        if self.reservoir.nz == 1:
            # uniform initial conditions, # pressure in bars # composition
            P_at_surface = 1.  # bars
            self.initial_values = {self.physics.vars[0]: P_at_surface, self.physics.vars[1]: self.ini}
            self.physics.gradient = {self.physics.vars[0]: 0.1}  # gradient 0.1 bars/m
        else:
            depth_array = np.array(self.reservoir.mesh.depth, copy=False)
            water_table_depth = depth_array.mean()  # specify your value here

            def sat_to_z(p, s):
                # find composition corresponding to particular saturation
                z_range = np.linspace(self.zero, 1 - self.zero, 2000)
                for z in z_range:
                    # state is pressure and 1 molar fractions out of 2
                    state = [p, z]
                    sat = self.physics.property_containers[0].compute_saturation_full(state)
                    if sat > s:
                        break
                return z
            def p_by_depth(depth):  # depth in meters
                return 1 + depth * 0.1  # gradient 0.1 bars/m
            def Sw_by_depth(depth):
                return 0 if depth > water_table_depth else 0.9

            # compute composition at few depth values
            n_depth_discr = self.reservoir.nz
            tbl_depth = np.linspace(depth_array.min(), depth_array.max(), n_depth_discr)
            tbl_z = np.zeros(n_depth_discr)
            for i in range(n_depth_discr):
                p = p_by_depth(tbl_depth[i])
                Sw = Sw_by_depth(tbl_depth[i])
                tbl_z[i] = sat_to_z(p, Sw)

            # and interpolate the resulting tbl_z to the full array (as loop over the variables would be slow)
            z_interp_func = interpolate.interp1d(tbl_depth, tbl_z, fill_value='extrapolate')
            Z_initial = z_interp_func(depth_array)

            P_initial = p_by_depth(depth_array)

            # set initial array for each variable: pressure and composition
            self.initial_values = {self.physics.vars[0]: P_initial, self.physics.vars[1]: Z_initial}

        # call base-class function from dart to transfer self.initial_values to actual arrays used in computation
        super().set_initial_conditions()

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):  # INJ well
                # BHP control
                w.control = self.physics.new_bhp_inj(250, self.inj)  # bars
                # rate control
                #w.control = self.physics.new_rate_inj(200, self.inj, 0)  # Kmol/day, composition, composition-index
                #w.constraint = self.physics.new_bhp_inj(250, self.inj)   # bars, composition
            else:  # PROD well
                # BHP control
                w.control = self.physics.new_bhp_prod(100)  # bars
                # rate control
                #w.control = self.physics.new_rate_prod(200)   # Kmol/day
                #w.constraint = self.physics.new_bhp_prod(100) # bars

    def get_arrays(self):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        a = self.reservoir.input_arrays  # include initial arrays and the grid

        nv = self.physics.n_vars
        nb = nv * self.reservoir.mesh.n_res_blocks
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[:nb:nv]
        a.update({'PRESSURE': P})

        print('P range [bars]:', fmt(P.min()), '-', fmt(P.max()))

        return a

    def print_well_rate(self):
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):
                inj_well = w
            else:
                prod_well = w
        time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
        years = np.array(time_data['time'])[-1] / 365.
        pr_col_name = time_data.filter(like=prod_well.name + ' : oil rate').columns.to_list()
        pp_col_name = time_data.filter(like=prod_well.name + ' : BHP').columns.to_list()
        ir_col_name = time_data.filter(like=inj_well.name + ' : water rate').columns.to_list()
        ip_col_name = time_data.filter(like=inj_well.name + ' : BHP').columns.to_list()
        rate_prod = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
        bhp_prod = np.array(time_data[pp_col_name])[-1][0]  # pick the last timestep value
        bhp_inj = np.array(time_data[ip_col_name])[-1][0]  # pick the last timestep value
        rate_inj = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
        print(fmt(years), 'years:', 'OIL RATE_prod =', fmt(rate_prod), ' WATER RATE_inj =', fmt(rate_inj), 'BHP_prod =',
              fmt(bhp_prod), 'BHP_inj =', fmt(bhp_inj))
