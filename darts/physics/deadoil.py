import numpy as np

from darts.input.input_data import InputData, FluidProps
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, DensityBrineCO2


class DeadOilBase(Compositional):
    def __init__(self, idata, timer):
        super().__init__(idata, timer, DeadOilProperties)

        property_container = DeadOilProperties(idata)
        property_container.density_ev = idata.fluid.density
        property_container.viscosity_ev = idata.fluid.viscosity
        property_container.rel_perm_ev = idata.fluid.rel_perm
        self.add_property_region(property_container)


class DeadOil(Compositional):
    def __init__(self, idata: InputData, timer, thermal):
        super().__init__(idata.fluid.components, idata.fluid.phases, timer,
                         idata.obl.n_points, idata.obl.min_p, idata.obl.max_p, idata.obl.min_z, idata.obl.max_z,
                         idata.obl.min_t, idata.obl.max_t)
        self.zero = 1e-13

        temperature = None if thermal else 1.
        property_container = DeadOilProperties(phases_name=idata.fluid.phases, components_name=idata.fluid.components,
                                               Mw=idata.fluid.Mw, min_z=self.zero / 10, temperature=temperature)

        property_container.density_ev = idata.fluid.density
        property_container.viscosity_ev = idata.fluid.viscosity
        property_container.rel_perm_ev = idata.fluid.rel_perm

        self.add_property_region(property_container)


class DeadOil2PFluidProps(FluidProps):#, idata: InputData):
    def __init__(self):
        super().__init__()
        self.components = ["Zo", "Zw"]
        self.phases = ['oil', 'water']
        self.Mw = np.ones(len(self.components))

        self.density = dict([('water', DensityBasic(compr=1e-5, dens0=1014)),
                             ('oil', DensityBasic(compr=5e-3, dens0=700))])
        self.viscosity = dict([('water', ConstFunc(0.89)),
                               ('oil', ConstFunc(1))])
        self.rel_perm = dict([('water', PhaseRelPerm("water", 0.1, 0.1)),
                              ('oil', PhaseRelPerm("oil", 0.1, 0.1))])


class DeadOil3PFluidProps(FluidProps):
    def __init__(self):
        super().__init__()
        self.components = ["g", "o", "w"]
        self.phases = ["gas", "oil", "water"]
        self.Mw = np.ones(len(self.components))

        self.density = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                             ('oil', DensityBasic(compr=1e-5, dens0=600)),
                             ('water', DensityBrineCO2(self.components, compr=1e-5, dens0=1000, co2_mult=0))])
        self.viscosity = dict([('gas', ConstFunc(0.05)),
                               ('oil', ConstFunc(0.5)),
                               ('water', ConstFunc(0.5))])
        self.rel_perm = dict([('gas', PhaseRelPerm("gas")),
                              ('oil', PhaseRelPerm("oil")),
                              ('water', PhaseRelPerm("water"))])


class DeadOilProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11, rock_comp=1e-6, temperature: float = None):
        # Call base class constructor
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z,
                         rock_comp=rock_comp, temperature=temperature)

    def run_flash(self, pressure, temperature, zc):
        ph = np.array([j for j in range(self.nph)])

        for i in range(self.nc):
            self.x[i][i] = 1
        self.nu = zc

        return ph
