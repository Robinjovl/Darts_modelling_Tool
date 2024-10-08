import numpy as np
import pandas as pd

from model_cpg import Model_CPG, fmt

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer as PropertyContainer
from darts.physics.properties.iapws.iapws_property_vec import enthalpy_to_temperature
from darts.physics.properties.iapws.custom_rock_property import custom_rock_compaction_evaluator

from darts.engines import value_vector
class ModelGeothermal(Model_CPG):
    def __init__(self, case='generate', grid_out_dir=None):
        self.n_points = 100  # OBL points
        super().__init__(physics_type='geothermal', case=case, grid_out_dir=grid_out_dir)

    def set_physics(self):
        # initialize physics for Geothermal
        property_container = PropertyContainer()
        property_container.output_props = {'T,degrees': lambda: property_container.temperature - 273.15}

        # Create rock_compaction object to set rock compressibility
        rock_compressibility = 1e-5  # [1/bars]
        property_container.rock = [value_vector([1, rock_compressibility, 273.15])]
        property_container.rock_compaction_ev = custom_rock_compaction_evaluator(property_container.rock)

        self.physics = Geothermal(timer=self.timer,
                                  n_points=self.n_points,   # number of OBL points
                                  min_p=50, max_p=400,      # pressure range for OBL grid
                                  min_e=1000, max_e=25000,  # enthalpy range for OBL grid
                                  cache=False)
        self.physics.add_property_region(property_container)

        # uniform initial conditions
        T_initial = 350.  # K
        P_initial = 200.  # bars
        state_init = value_vector([P_initial, 0.])
        enth_init = self.physics.property_containers[0].enthalpy_ev['total'](T_initial).evaluate(state_init)
        self.initial_values = {self.physics.vars[0]: state_init[0],
                               self.physics.vars[1]: enth_init}

    def set_initial_conditions(self, initial_values: dict = None, gradient: dict = None):
         self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh, pressure_grad=100, temperature_grad=30)

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):  # INJ well
                inj_temperature = 300  # K
                # rate control
                w.control = self.physics.new_rate_water_inj(5500, inj_temperature)  #  m3/day
                w.constraint = self.physics.new_bhp_water_inj(300, inj_temperature)  # upper limit for bhp, bars
                # BHP control
                #w.control = self.physics.new_bhp_water_inj(250, inj_temperature)  # bars
            else:  # PROD well
                # rate control
                w.control = self.physics.new_rate_water_prod(5500)  #  m3/day
                w.constraint = self.physics.new_bhp_prod(70)  # lower limit for bhp, bars
                # BHP control
                #w.control = self.physics.new_bhp_prod(100)  # bars

    def get_arrays(self):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        a = self.reservoir.input_arrays  # include initial arrays and the grid

        nv = self.physics.n_vars
        nb = nv * self.reservoir.mesh.n_res_blocks
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[:nb:nv]
        T = enthalpy_to_temperature(Xn[:nb])
        T -= 273.15  # K to degrees

        a.update({'PRESSURE': P, 'TEMPERATURE': T})

        print('P range [bars]:', fmt(P.min()), '-', fmt(P.max()), 'T range [degrees]:', fmt(T.min()), '-', fmt(T.max()))

        return a

    def print_well_rate(self):
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):
                inj_well = w
            else:
                prod_well = w
        time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
        years = np.array(time_data['time'])[-1]/365.
        pr_col_name = time_data.filter(like=prod_well.name + ' : water rate').columns.to_list()
        pt_col_name = time_data.filter(like=prod_well.name + ' : temperature').columns.to_list()
        ir_col_name = time_data.filter(like=inj_well.name + ' : water rate').columns.to_list()
        rate_prod = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
        temp_prod = np.array(time_data[pt_col_name])[-1][0]  # pick the last timestep value
        rate_inj  = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
        print(fmt(years), 'years:', 'RATE_prod =', fmt(rate_prod), 'RATE_inj =', fmt(rate_inj), 'TEMP_prod =', fmt(temp_prod))