import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.density import DensityBasic, DensityBrineCO2

import numpy as np

class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=100, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

        self.initial_values = {self.physics.vars[0]: 50.,
                               self.physics.vars[1]: self.ini_stream[0],
                               self.physics.vars[2]: self.ini_stream[1],
                               self.physics.vars[3]: self.ini_stream[2]
                               }

    def set_reservoir(self):
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=10, dz=10, permx=100, permy=100,
                                         permz=10, poro=0.3, depth=1000)

        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", cell_index=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", cell_index=(self.reservoir.nx, 1, 1))

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        zero = 1e-8
        components = ['CO2', 'C1', 'H2S', 'H2O']
        phases = ['gas', 'oil', 'wat']
        nc = len(components)
        Mw = [44.01, 16.04, 34.081, 18.015]

        self.inj_stream = [1.0 - 2 * zero, zero, zero]
        self.ini_stream = [0.1, 0.2, 0.6 - zero]

        property_container = ModelProperties(phases_name=phases, components_name=components, Mw=Mw, min_z=zero/10)

        """ properties correlations """
        property_container.flash_ev = ConstantK(nc-1, [4, 2, 1e-2], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600)),
                                              ('wat', DensityBrineCO2(components, compr=1e-5, dens0=1000, co2_mult=0))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('oil', ConstFunc(0.5)),
                                                ('wat', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil")),
                                               ('wat', PhaseRelPerm("wat"))])

        """ Activate physics """
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=200, min_p=1, max_p=300, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_inj(20, self.inj_stream, 0)
                # w.control = self.physics.new_bhp_inj(100, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(50)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11):
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, min_z=min_z, temperature=1.)

    def run_flash(self, pressure, temperature, zc):

        zc_r = zc[:-1] / (1 - zc[-1])
        self.flash_ev.evaluate_PT(pressure, temperature, zc_r)
        flash_results = self.flash_ev.get_flash_results()
        nu = np.array(flash_results.nu)
        xr = np.array(flash_results.X).reshape(self.nph-1, self.nc-1)
        V = nu[0]

        if V <= 0:
            V = 0
            xr[1] = zc_r
            ph = [1, 2]
        elif V >= 1:
            V = 1
            xr[0] = zc_r
            ph = [0, 2]
        else:
            ph = [0, 1, 2]

        for i in range(self.nc - 1):
            for j in range(2):
                self.x[j][i] = xr[j][i]

        self.x[-1][-1] = 1

        self.nu[0] = V * (1 - zc[-1])
        self.nu[1] = (1 - V) * (1 - zc[-1])
        self.nu[2] = zc[-1]

        return np.array(ph, dtype=np.intp)
