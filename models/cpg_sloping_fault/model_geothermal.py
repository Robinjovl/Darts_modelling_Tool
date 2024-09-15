from model_cpg import Model_CPG

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer as PropertyContainer

from darts.engines import value_vector
class ModelGeothermal(Model_CPG):
    def __init__(self, discr_type='cpp', case='generate', grid_out_dir=None, n_points=100):
        super().__init__(discr_type=discr_type, case=case, grid_out_dir=grid_out_dir, n_points=n_points)

    def set_physics(self):
        '''
        set Geothermal physics
        :return:
        '''
        # initialize physics for Geothermal
        property_container = PropertyContainer()
        self.physics = Geothermal(timer=self.timer,
                                  n_points=self.n_points,   # number of OBL points
                                  min_p=50, max_p=400,      # pressure range
                                  min_e=1000, max_e=25000,  # enthalpy range
                                  cache=False)
        self.physics.add_property_region(property_container)

        T_init = 350.
        state_init = value_vector([200., 0.])
        enth_init = self.physics.property_containers[0].enthalpy_ev['total'](T_init).evaluate(state_init)
        self.initial_values = {self.physics.vars[0]: state_init[0],
                               self.physics.vars[1]: enth_init
                               }

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):  # INJ well
                # rate control
                w.control = self.physics.new_rate_water_inj(0, 300)  #  m3/day, K
                w.constraint = self.physics.new_bhp_water_inj(500, 300)  # upper limit for bhp, bars
                # BHP control
                #w.control = self.physics.new_bhp_water_inj(250, 300)  # bars
            else:  # PROD well
                # rate control
                w.control = self.physics.new_rate_water_prod(0)  #  m3/day
                w.constraint = self.physics.new_bhp_prod(50)  # lower limit for bhp, bars
                # BHP control
                #w.control = self.physics.new_bhp_prod(100)  # bars