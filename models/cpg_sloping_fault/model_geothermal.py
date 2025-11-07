import numpy as np
import pandas as pd

from darts.engines import value_vector
from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH, GeothermalIAPWSFluidProps, GeothermalPHFluidProps
from model_base import Model_CPG, fmt

class ModelGeothermal(Model_CPG):
    def __init__(self, iapws_physics: bool = True):
        self.iapws_physics = iapws_physics
        self.physics_type = 'geothermal'
        super().__init__()

    def set_physics(self):
        # single component, two phase. Pressure and enthalpy are the main variables
        if self.iapws_physics:
            self.physics = Geothermal(self.idata, self.timer)  # IAPWS
        else:
            self.physics = GeothermalPH(self.idata, self.timer)  # Flash
            self.physics.determine_obl_bounds(state_min=[self.idata.obl.min_p, 250.],
                                              state_max=[self.idata.obl.max_p, 575.])

    def set_initial_conditions(self):
        if self.idata.initial.type == 'gradient':
            # Specify reference depth, values and gradients to construct depth table in super().set_initial_conditions()
            input_depth = [0., np.amax(self.reservoir.mesh.depth)]
            input_distribution = {'pressure': [1., 1. + input_depth[1] * self.idata.initial.pressure_gradient/1000],
                                  'temperature': [293.15, 293.15 + input_depth[1] * self.idata.initial.temperature_gradient/1000]
                                  }
            return self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                                        input_distribution=input_distribution,
                                                                        input_depth=input_depth)
        elif self.idata.initial.type == 'uniform':
            input_distribution = {'pressure': self.idata.initial.initial_pressure,
                                  'temperature': self.idata.initial.initial_temperature}
            return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                                  input_distribution=input_distribution)



    def get_arrays(self):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        a = self.reservoir.input_arrays  # include initial arrays and the grid

        nv = self.physics.n_vars
        n_ops = self.output.n_ops
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
            if self.idata.well_is_inj(w.name):
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

