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
        self.idata = idata
        self.zero = 1e-13

        property_container = DeadOilProperties(idata=idata, thermal=thermal)

        property_container.density_ev = idata.fluid.density
        property_container.viscosity_ev = idata.fluid.viscosity
        property_container.rel_perm_ev = idata.fluid.rel_perm

        self.add_property_region(property_container)


class DeadOil2PFluidProps(FluidProps):#, idata: InputData):
    def __init__(self):
        super().__init__(phases_name=["oil", "water"], components_name=["o", "w"], Mw=np.ones(2))

        self.density_ev = dict([('water', DensityBasic(compr=1e-5, dens0=1014)),
                                ('oil', DensityBasic(compr=5e-3, dens0=700))])
        self.viscosity_ev = dict([('water', ConstFunc(0.89)),
                                  ('oil', ConstFunc(1))])
        self.rel_perm_ev = dict([('water', PhaseRelPerm("water", 0.1, 0.1)),
                                 ('oil', PhaseRelPerm("oil", 0.1, 0.1))])


class DeadOil3PFluidProps(FluidProps):
    def __init__(self):
        super().__init__(phases_name=["gas", "oil", "water"], components_name=["g", "o", "w"], Mw=np.ones(3))

        self.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                ('oil', DensityBasic(compr=1e-5, dens0=600)),
                                ('water', DensityBrineCO2(self.components_name, compr=1e-5, dens0=1000, co2_mult=0))])
        self.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                  ('oil', ConstFunc(0.5)),
                                  ('water', ConstFunc(0.5))])
        self.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                 ('oil', PhaseRelPerm("oil")),
                                 ('water', PhaseRelPerm("water"))])


class DeadOilProperties(PropertyContainer):
    def __init__(self, idata: InputData, thermal: bool = False):
        # Call base class constructor
        super().__init__(idata=idata, constant_temperature=None if thermal else 1.)

    def run_flash(self, pressure, temperature, zc):
        ph = np.array([j for j in range(self.nph)])

        for i in range(self.nc):
            self.x[i][i] = 1
        self.nu = zc

        return ph
