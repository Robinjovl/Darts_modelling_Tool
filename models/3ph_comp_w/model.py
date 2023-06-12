from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.density import DensityBasic, DensityBrineCO2
# from darts.models.physics_sup.properties_basic import ConstFunc, Density, DensityBrineCo2, PhaseRelPerm, ConstantK


# Model class creation here!
class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        """Reservoir"""
        self.reservoir = StructReservoir(self.timer, nx=1000, ny=1, nz=1, dx=1, dy=10, dz=10, permx=100, permy=100,
                                         permz=10, poro=0.3, depth=1000)

        #self.reservoir.set_boundary_volume(yz_plus=1e8)

        # overwrite depth to involve gravity
        self.depth_start = self.reservoir.depth[0]
        self.depth_end = self.reservoir.depth[0]                # by changing this, one can change the inclined angle

        # self.layer_1 = np.linspace(self.depth_start, self.depth_end, self.reservoir.nx)       # inclined
        # for i in range(self.reservoir.nz):
        #     self.reservoir.depth[i*self.reservoir.nx: (i+1)*self.reservoir.nx] = self.layer_1 + i*self.reservoir.global_data['dz']

        """well location"""
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1000, j=1, k=1, multi_segment=False)

        # self.reservoir.add_well("P1")
        # for i in range(int(self.reservoir.nz / 2)):
        #     self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.reservoir.nx, j=1, k=i+1, multi_segment=False)

        """Physical properties"""
        # Create property containers:
        self.zero = 1e-8
        components = ['CO2', 'C1', 'H2S', 'H2O']
        phases = ['gas', 'oil', 'wat']
        nc = len(components)
        Mw = [44.01, 16.04, 34.081, 18.015]

        self.inj_stream = [1.0 - 2 * self.zero, self.zero, self.zero]
        self.ini_stream = [0.1, 0.2, 0.6 - self.zero]

        property_container = ModelProperties(phases_name=phases, components_name=components, Mw=Mw, min_z=self.zero/10)

        """ properties correlations """
        property_container.flash_ev = ConstantK(nc-1, [4, 2, 1e-2], self.zero)
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
                                     n_points=200, min_p=1, max_p=300, min_z=self.zero/10, max_z=1-self.zero/10)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 0.001
        self.params.max_ts = 1
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-3
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.newton_params[0] = 0.2

        self.runtime = 100

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 50, self.ini_stream)

        # composition = np.array(self.reservoir.mesh.composition, copy=False)
        # n_half = int(self.reservoir.nx * self.reservoir.ny * self.reservoir.nz / 2)
        # composition[2*n_half:] = 1e-6

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_inj(20, self.inj_stream, 0)
                # w.control = self.physics.new_bhp_inj(100, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(50)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11):
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, min_z, temperature=1.)

    def run_flash(self, pressure, temperature, zc):

        zc_r = zc[:-1] / (1 - zc[-1])
        (nu, xr) = self.flash_ev.evaluate(pressure, temperature, zc_r)
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

        return ph
