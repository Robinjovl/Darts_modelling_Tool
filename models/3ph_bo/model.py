import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.black_oil import *


# Model class creation here!
class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        self.set_sim_params(first_ts=1e-6, mult_ts=2, max_ts=10, runtime=100, tol_newton=1e-3, tol_linear=1e-7,
                            it_newton=10, it_linear=50)

        self.timer.node["initialization"].stop()

        self.initial_values = {self.physics.vars[0]: 330.,
                               self.physics.vars[1]: self.ini_stream[0],
                               self.physics.vars[2]: self.ini_stream[1]
                               }

    def set_reservoir(self):
        """Reservoir"""
        (nx, ny, nz) = (10, 10, 3)
        kx = np.array([500] * 100 + [50] * 100 + [200] * 100)
        ky = kx
        kz = np.array([80] * 100 + [42] * 100 + [20] * 100)
        dz = np.array([6] * 100 + [9] * 100 + [15] * 100)
        depth = np.array([2540] * 100 + [2548] * 100 + [2560] * 100)
        dx = 3000 / nx
        dy = 3000 / ny

        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=kx, permy=ky, permz=kz, poro=0.3, depth=depth)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", cell_index=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", cell_index=(10, 10, 3))

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        zero = 1e-12
        components = ['g', 'o', 'w']
        phases = ['gas', 'oil', 'wat']

        self.inj_stream = [1 - 2 * zero, zero]
        self.ini_stream = [0.001225901537, 0.7711341309]

        """ properties correlations """
        pvt = 'physics.in'
        property_container = ModelProperties(phases_name=phases, components_name=components, pvt=pvt, min_z=zero/10)

        property_container.flash_ev = flash_black_oil(pvt)
        property_container.density_ev = dict([('gas', DensityGas(pvt)),
                                              ('oil', DensityOil(pvt)),
                                              ('wat', DensityWat(pvt))])
        property_container.viscosity_ev = dict([('gas', ViscGas(pvt)),
                                                ('oil', ViscOil(pvt)),
                                                ('wat', ViscWat(pvt))])
        property_container.rel_perm_ev = dict([('gas', GasRelPerm(pvt)),
                                               ('oil', OilRelPerm(pvt)),
                                               ('wat', WatRelPerm(pvt))])
        property_container.capillary_pressure_ev = dict([('pcow', CapillaryPressurePcow(pvt)),
                                                         ('pcgo', CapillaryPressurePcgo(pvt))])

        property_container.rock_compress_ev = RockCompactionEvaluator(pvt)

        """ Activate physics """
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=5000, min_p=1, max_p=450, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_bhp_inj(400, self.inj_stream)
                # w.control = self.physics.new_bhp_inj(100, self.inj_stream)
            else:
                # w.control = self.physics.new_rate_oil_prod(3000)
                w.control = self.physics.new_bhp_prod(70)


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, pvt, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        self.pvt = pvt
        super().__init__(phases_name, components_name, Mw, min_z=min_z, temperature=1.)
        self.surf_dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.surf_oil_dens = self.surf_dens[0]
        self.surf_wat_dens = self.surf_dens[1]
        self.surf_gas_dens = self.surf_dens[2]

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        (xgo, V, pbub) = self.flash_ev.evaluate(pressure, zc)
        for i in range(self.nph):
            self.x[i, i] = 1

        if V < 0:
            self.ph = [1, 2]
        else:  # assume oil and water are always exists
            self.x[1][0] = xgo
            self.x[1][1] = 1 - xgo
            self.ph = [0, 1, 2]

        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, pbub, xgo)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, pbub)  # output in [cp]

        self.nu[2] = zc[2]
        # two phase undersaturated condition
        if pressure > pbub:
            self.nu[0] = 0
            self.nu[1] = zc[1]
        else:
            self.nu[1] = zc[1] / (1 - xgo)
            self.nu[0] = 1 - self.nu[1] - self.nu[2]

        self.compute_saturation(self.ph)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0], self.sat[2])

        pcow = self.capillary_pressure_ev['pcow'].evaluate(self.sat[2])
        pcgo = self.capillary_pressure_ev['pcgo'].evaluate(self.sat[0])

        self.pc = np.array([-pcgo, 0, pcow])

        return

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        if zc[-1] < 0:
            # print(zc)
            zc = self.comp_out_of_bounds(zc)

        self.ph = []
        for j in range(self.nph):
            if zc[j] > self.min_z:
                self.ph.append(j)
            self.dens_m[j] = self.density_ev[self.phases_name[j]].dens_sc

        self.nu = zc
        self.compute_saturation(self.ph)

        return self.sat, self.dens_m
