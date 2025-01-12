import numpy as np
import pandas as pd

from darts.engines import value_vector
from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH, GeothermalIAPWSFluidProps, GeothermalPHFluidProps

from darts.input.input_data import InputData
from set_case import set_input_data
from model_cpg import Model_CPG, fmt


class ModelGeothermal(Model_CPG):
    def __init__(self, case='generate', grid_out_dir=None, iapws_physics: bool = True):
        self.iapws_physics = iapws_physics
        super().__init__(physics_type='geothermal', case=case, grid_out_dir=grid_out_dir)

    def set_physics(self):
        if self.iapws_physics:
            self.physics = Geothermal(self.idata, self.timer)
        else:
            self.physics = GeothermalPH(self.idata, self.timer)
            self.physics.determine_obl_bounds(state_min=[self.idata.obl.min_p, 250.],
                                              state_max=[self.idata.obl.max_p, 575.])

    def set_initial_conditions(self):
        if self.idata.initial.type == 'gradient':
            self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                       pressure_grad=self.idata.initial.pressure_gradient,
                                                       temperature_grad=self.idata.initial.temperature_gradient)
        elif self.idata.initial.type == 'uniform':
            state_init = value_vector([self.idata.initial.initial_pressure, 0.])
            enth_init = self.physics.property_containers[0].compute_total_enthalpy(state_init, self.idata.initial.initial_temperature)
            self.initial_values = {self.physics.vars[0]: state_init[0],
                                   self.physics.vars[1]: enth_init}
            super().set_initial_conditions()

    def set_well_controls(self, time: float = 0., verbose=True):
        '''
        :param time: simulation time, [days]
        :return:
        '''
        eps_time = 1e-15  # threshold between the current time and the time for the well control
        for w in self.reservoir.wells:
            # find next well control in controls list for different timesteps
            wctrl = None
            for wctrl_t in self.idata.well_data.wells[w.name].controls:
                if np.fabs(wctrl_t[0] - time) < eps_time:  # check time
                    wctrl = wctrl_t[1]
                    break
            if wctrl is None:  # no control is defined for the current timestep
                continue
            if wctrl.type == 'inj':  # INJ well
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_water_inj(wctrl.rate, wctrl.inj_bht)
                    w.constraint = self.physics.new_bhp_water_inj(wctrl.bhp_constraint, wctrl.inj_bht)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_water_inj(wctrl.bhp, wctrl.inj_bht)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            elif wctrl.type == 'prod':  # PROD well
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_water_prod(wctrl.rate)
                    w.constraint = self.physics.new_bhp_prod(wctrl.bhp_constraint)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_prod(wctrl.bhp)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            else:
                print('Unknown well ctrl.type', wctrl.type)
                exit(1)
            if verbose:
                print('set_well_controls: time=', time, 'well=', w.name, w.control, w.constraint)
            assert w.control is not None, 'well control is not initialized!' + w.name
            if verbose and w.constraint is None and wctrl.mode == 'rate':
                print('A constraint for the well ' + w.name + ' is not initialized!')

    def get_arrays(self):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        a = self.reservoir.input_arrays  # include initial arrays and the grid

        nv = self.physics.n_vars
        n_ops = self.physics.n_ops
        nb = self.reservoir.mesh.n_res_blocks
        Xn = np.array(self.physics.engine.X, copy=False)
        state = value_vector(Xn.T.flatten())

        # Interpolate temperature with property interpolator
        values = value_vector(np.zeros(n_ops * nb))
        values_numpy = np.array(values, copy=False)
        dvalues = value_vector(np.zeros(n_ops * nb * nv))
        i = 0
        for region, prop_itor in self.physics.property_itor.items():
            prop_itor.evaluate_with_derivatives(state, self.physics.engine.region_cell_idx[i], values, dvalues)
            i += 1

        # Get P from state vector and T from interpolated properties
        P = np.array(state[0:nb*nv:nv])
        T = values_numpy[0:nb*n_ops:n_ops]
        T -= 273.15  # K to degrees

        a.update({'PRESSURE': P, 'TEMPERATURE': T})

        print('P range [bars]:', fmt(P.min()), '-', fmt(P.max()), 'T range [degrees]:', fmt(T.min()), '-', fmt(T.max()))

        return a

    def print_well_rate(self):
        inj_well = prd_well = None
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):
                inj_well = w
            else:
                prd_well = w
        time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
        years = np.array(time_data['time'])[-1]/365.25

        rate_inj = rate_prd = temp_prd = temp_inj = 0.
        if prd_well is not None:
            pr_col_name = time_data.filter(like=prd_well.name + ' : water rate').columns.to_list()
            pt_col_name = time_data.filter(like=prd_well.name + ' : temperature').columns.to_list()
            rate_prd = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
            temp_prd = np.array(time_data[pt_col_name])[-1][0]  # pick the last timestep value
        if inj_well is not None:
            ir_col_name = time_data.filter(like=inj_well.name + ' : water rate').columns.to_list()
            it_col_name = time_data.filter(like=inj_well.name + ' : temperature').columns.to_list()
            rate_inj  = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
            temp_inj = np.array(time_data[it_col_name])[-1][0]  # pick the last timestep value
        print(fmt(years), 'years:', 'RATE_prod =', fmt(rate_prd), 'RATE_inj =', fmt(rate_inj), 'TEMP_prod =', fmt(temp_prd), 'TEMP_inj =', fmt(temp_inj))

    def set_input_data(self, case=''):
        #init_type = 'uniform'
        init_type = 'gradient'
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type=init_type)
        set_input_data(self.idata, case)

        if self.iapws_physics:
            self.idata.fluid = GeothermalIAPWSFluidProps()
        else:
            self.idata.fluid = GeothermalPHFluidProps()

        # example - how to change the properties
        # self.idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)

        #from darts.physics.properties.basic import ConstFunc
        #self.idata.fluid.conduction_ev['water'] = ConstFunc(172.8)

        if init_type== 'uniform': # uniform initial conditions
            self.idata.initial.initial_pressure = 200.  # bars
            self.idata.initial.initial_temperature = 350.  # K
        elif init_type == 'gradient':         # gradient by depth
            self.idata.initial.reference_depth_for_pressure = 0  # [m]
            self.idata.initial.pressure_gradient = 100  # [bar/km]
            self.idata.initial.pressure_at_ref_depth = 1 # [bars]

            self.idata.initial.reference_depth_for_temperature = 0  # [m]
            self.idata.initial.temperature_gradient = 30  # [K/km]
            self.idata.initial.temperature_at_ref_depth = 273.15 + 20 # [K]

        # well controls
        wdata = self.idata.well_data
        wells = wdata.wells  # short name

        if 'wbhp' in case:
            for w in wells:
                if self.well_is_inj(w):
                    wdata.add_inj_bhp_control(name=w, bhp=250, temperature=300)  # m3/day | bars | K
                else: # prod
                    wdata.add_prd_bhp_control(name=w, bhp=100) # m3/day | bars
        elif 'wrate' in case:
            for w in wells:
                if self.well_is_inj(w):
                    wdata.add_inj_rate_control(name=w, rate=5500, bhp_constraint=300, temperature=300)  # m3/day | bars | K
                else: # prod
                    wdata.add_prd_rate_control(name=w, rate=5500, bhp_constraint=70) # m3/day | bars
        elif 'wperiodic' in case:
            wname = list(wdata.wells.keys())[0]  # single well
            y2d = 365.25
            for i in range(0, len(self.idata.sim.time_steps), 4):
                # iterate [inj - stop - prod - stop]
                wdata.add_inj_rate_control(time=(i+0)*y2d, name=wname, rate=5500, bhp_constraint=300, temperature=300)
                wdata.add_prd_rate_control(time=(i+1)*y2d, name=wname, rate=0,    bhp_constraint=5)
                wdata.add_prd_rate_control(time=(i+2)*y2d, name=wname, rate=5500, bhp_constraint=5)
                wdata.add_prd_rate_control(time=(i+3)*y2d, name=wname, rate=0,    bhp_constraint=5)
        else:
            assert False, 'Unknown wctrl_type' +  case

        self.idata.obl.n_points = 100
        self.idata.obl.min_p = 50.
        self.idata.obl.max_p = 400.
        self.idata.obl.min_e = 1000.  # kJ/kmol, will be overwritten in PHFlash physics
        self.idata.obl.max_e = 25000.  # kJ/kmol, will be overwritten in PHFlash physics
