import numpy as np
import pandas as pd
from scipy import interpolate

from model_cpg import Model_CPG, fmt
from darts.input.input_data import InputData
from darts.engines import value_vector
from darts.physics.deadoil import DeadOil, DeadOil2PFluidProps


class ModelDeadOil(Model_CPG):
    def __init__(self, case='generate', grid_out_dir=None):
        self.zero = 1e-13
        super().__init__(physics_type='dead_oil', case=case, grid_out_dir=grid_out_dir)

    def set_physics(self):
        self.physics = DeadOil(self.idata, self.timer, thermal=False)
        self.ini = value_vector([1 - self.zero])  # initial composition (above water table depth) - oil

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
        wctrl = self.idata.wells.controls
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):  # INJ well
                if wctrl.type == 'rate': # rate control
                    w.control = self.physics.new_rate_inj(wctrl.inj_rate, wctrl.inj, wctrl.inj_comp_index)
                    w.constraint = self.physics.new_bhp_inj(wctrl.inj_bhp_constraint, wctrl.inj)
                elif wctrl.type == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_inj(wctrl.inj_bhp, wctrl.inj)
            else:  # PROD well
                if wctrl.type == 'rate': # rate control
                    w.control = self.physics.new_rate_prod(wctrl.prod_rate)
                    w.constraint = self.physics.new_bhp_prod(wctrl.prod_bhp_constraint)
                elif wctrl.type == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_prod(wctrl.prod_bhp)

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

    def set_input_data(self, case=''):
        self.idata = InputData(type_hydr='isothermal', type_mech='none', init_type='uniform')

        # this sets default properties
        self.idata.fluid = DeadOil2PFluidProps() #if twophase else DeadOil3PFluidProps

        # example - how to change the properties
        # self.idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)

        # well controls
        wctrl = self.idata.wells.controls  # short name

        #wctrl.type = 'rate'
        wctrl.type = 'bhp'

        wctrl.inj = value_vector([self.zero])  # injection composition - water

        if wctrl.type == 'bhp':
            self.idata.wells.controls.inj_bhp = 250 # bars
            self.idata.wells.controls.prod_bhp = 100 # bars
        elif wctrl.type == 'rate':
            self.idata.wells.controls.inj_rate = 200 # kmol/day
            self.idata.wells.controls.inj_bhp_constraint = 300 # upper limit for bhp, bars
            self.idata.wells.controls.prod_rate = 200 # kmol/day
            self.idata.wells.controls.prod_bhp_constraint = 70 # lower limit for bhp, bars
        self.idata.wells.controls.inj_bht = 300  # K

        self.idata.obl.n_points = 400
        self.idata.obl.zero = 1e-13
        self.idata.obl.min_p = 0.
        self.idata.obl.max_p = 1000.
        self.idata.obl.min_t = 10.
        self.idata.obl.max_t = 100.
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
