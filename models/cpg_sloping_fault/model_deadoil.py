import numpy as np
import pandas as pd
from scipy import interpolate

from darts.input.input_data import InputData
from darts.engines import value_vector
from darts.physics.deadoil import DeadOil, DeadOil2PFluidProps
from darts.engines import well_control_iface

from model_cpg import Model_CPG, fmt
from set_case import set_input_data


class ModelDeadOil(Model_CPG):
    def __init__(self):
        self.zero = 1e-13
        super().__init__()

    def set_physics(self):
        self.physics = DeadOil(self.idata, self.timer, thermal=False)
        self.ini = value_vector([1 - self.zero])  # initial composition (above water table depth) - oil

    def set_initial_conditions(self):  # override origin set_initial_conditions function from darts_model
        if self.reservoir.nz == 1:
            # uniform initial conditions, # pressure in bars # composition
            # Specify reference depth, values and gradients to construct depth table in super().set_initial_conditions()
            input_depth = [0., np.amax(self.reservoir.mesh.depth)]
            P_at_surface = 1.  # bar
            input_distribution = {'pressure': [P_at_surface, P_at_surface + input_depth[1] * 0.1],  # gradient 0.1 bar/m
                                  self.physics.vars[1]: [self.ini[0], self.ini[0]]
                                  }
            g2l = np.asarray(self.reservoir.discr_mesh.global_to_local)[:self.reservoir.mesh.n_res_blocks]
            return self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                                        global_to_local=g2l,
                                                                        input_distribution=input_distribution,
                                                                        input_depth=input_depth)
        else:
            nb = self.reservoir.mesh.n_res_blocks
            depth_array = np.array(self.reservoir.mesh.depth, copy=False)[:nb]
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
            input_distribution = {self.physics.vars[0]: P_initial,
                                  self.physics.vars[1]: Z_initial
                                  }

            return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                                  input_distribution=input_distribution)

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
        inj_well = None
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):
                inj_well = w
            else:
                prod_well = w
        time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
        years = np.array(time_data['time'])[-1] / 365.
        pr_col_name = time_data.filter(like=prod_well.name + ' : oil rate').columns.to_list()
        pp_col_name = time_data.filter(like=prod_well.name + ' : BHP').columns.to_list()
        rate_prod = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
        bhp_prod = np.array(time_data[pp_col_name])[-1][0]  # pick the last timestep value
        if inj_well is not None:
            ir_col_name = time_data.filter(like=inj_well.name + ' : water rate').columns.to_list()
            ip_col_name = time_data.filter(like=inj_well.name + ' : BHP').columns.to_list()
            bhp_inj = np.array(time_data[ip_col_name])[-1][0]  # pick the last timestep value
            rate_inj = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
        else:
            bhp_inj = rate_inj = 0.
        print(fmt(years), 'years:', 'OIL RATE_prod =', fmt(rate_prod), ' WATER RATE_inj =', fmt(rate_inj), 'BHP_prod =',
              fmt(bhp_prod), 'BHP_inj =', fmt(bhp_inj))

    def set_input_data(self, case=''):
        self.idata = InputData(type_hydr='isothermal', type_mech='none', init_type='uniform')
        set_input_data(self.idata, case)

        self.idata.geom.burden_layers = 0

        # this sets default properties
        self.idata.fluid = DeadOil2PFluidProps() #if twophase else DeadOil3PFluidProps

        # example - how to change the properties
        # self.idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)
        # well controls
        wdata = self.idata.well_data
        wells = wdata.wells  # short name
        # set default injection composition
        inj_comp = value_vector([self.zero * 100])  # injection composition - water

        if 'wbhp' in case:
            for w in wells:
                if self.well_is_inj(w):
                    wdata.add_inj_bhp_control(name=w, bhp=250, phase_name='water', inj_composition=inj_comp, temperature=300)  # kmol/day | bars | K
                else:  # prod
                    wdata.add_prd_bhp_control(name=w, bhp=100)  # kmol/day | bars
        elif 'wrate' in case:
            for w in wells:
                if self.well_is_inj(w): # inject water
                    wdata.add_inj_rate_control(name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE, phase_name='water', inj_composition=inj_comp, bhp_constraint=250)  # kmol/day | bars | K
                else:  # prod
                    wdata.add_prd_rate_control(name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE, phase_name='oil', bhp_constraint=100)  # kmol/day | bars
        elif 'wperiodic' in case:
            y2d = 365.25
            for w in wells:
                if self.well_is_inj(w): # inject water
                    wdata.add_inj_rate_control(time=0*y2d, name=w, rate=1e5, rate_type=well_control_iface.MOLAR_RATE, phase_name='water', inj_composition=inj_comp, bhp_constraint=300)  # kmol/day | bars | K
                    wdata.add_inj_rate_control(time=1*y2d, name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE, phase_name='water', inj_composition=inj_comp, bhp_constraint=300)  # kmol/day | bars | K
                else:  # prod
                    wdata.add_prd_rate_control(time=0*y2d, name=w, rate=1e5, rate_type=well_control_iface.MOLAR_RATE, phase_name='oil', bhp_constraint=70)  # kmol/day | bars
                    wdata.add_prd_rate_control(time=1*y2d, name=w, rate=1e6, rate_type=well_control_iface.MOLAR_RATE, phase_name='oil', bhp_constraint=70)  # kmol/day | bars

        self.idata.obl.n_points = 400
        self.idata.obl.zero = 1e-13
        self.idata.obl.min_p = 0.
        self.idata.obl.max_p = 1000.
        self.idata.obl.min_t = 10.
        self.idata.obl.max_t = 100.
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
