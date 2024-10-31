from darts.input.input_data import InputData, FluidProps
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.black_oil import *


class BlackOilBase(Compositional):
    def __init__(self, idata, timer):
        super().__init__(idata, timer, BlackOilProperties)

        property_container = BlackOilProperties(idata)
        property_container.density_ev = idata.fluid.density
        property_container.viscosity_ev = idata.fluid.viscosity
        property_container.rel_perm_ev = idata.fluid.rel_perm
        self.add_property_region(property_container)


class BlackOil(Compositional):
    def __init__(self, idata: InputData, timer, thermal):
        super().__init__(idata.fluid.components, idata.fluid.phases, timer,
                         idata.obl.n_points, idata.obl.min_p, idata.obl.max_p, idata.obl.min_z, idata.obl.max_z,
                         idata.obl.min_t, idata.obl.max_t)

        temperature = None if thermal else 1.
        property_container = BlackOilProperties(phases_name=idata.fluid.phases, components_name=idata.fluid.components,
                                                Mw=idata.fluid.Mw, min_z=idata.obl.min_z, temperature=temperature)

        property_container.flash_ev = idata.fluid.flash_ev
        property_container.density_ev = idata.fluid.density
        property_container.viscosity_ev = idata.fluid.viscosity
        property_container.rel_perm_ev = idata.fluid.rel_perm
        property_container.capillary_pressure_ev = idata.fluid.capillary_pressure

        property_container.rock_compress_ev = RockCompactionEvaluator(idata.fluid.pvt)

        self.add_property_region(property_container)


class BlackOilFluidProps(FluidProps):
    def __init__(self, pvt):
        super().__init__()
        self.components = ["g", "o", "w"]
        self.phases = ["gas", "oil", "water"]
        self.Mw = np.ones(len(self.components))

        self.pvt = pvt
        self.flash_ev = flash_black_oil(pvt)
        self.density = dict([('gas', DensityGas(pvt)),
                             ('oil', DensityOil(pvt)),
                             ('water', DensityWat(pvt))])
        self.viscosity = dict([('gas', ViscGas(pvt)),
                               ('oil', ViscOil(pvt)),
                               ('water', ViscWat(pvt))])
        self.rel_perm = dict([('gas', GasRelPerm(pvt)),
                              ('oil', OilRelPerm(pvt)),
                              ('water', WatRelPerm(pvt))])
        self.capillary_pressure = dict([('pcow', CapillaryPressurePcow(pvt)),
                                        ('pcgo', CapillaryPressurePcgo(pvt))])


class BlackOilProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z: float = 1e-11, temperature: float = None):
        # Call base class constructor
        super().__init__(phases_name, components_name, Mw, min_z=min_z, temperature=1.)
        # self.surf_dens = get_table_keyword(idata.fluid.pvt, 'DENSITY')[0]
        # self.surf_oil_dens = self.surf_dens[0]
        # self.surf_wat_dens = self.surf_dens[1]
        # self.surf_gas_dens = self.surf_dens[2]

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
            self.ph = np.array([1, 2])
        else:  # assume oil and water are always exists
            self.x[1][0] = xgo
            self.x[1][1] = 1 - xgo
            self.ph = np.array([0, 1, 2])

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
