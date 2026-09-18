import numpy as np

from darts.input.input_data import FluidProps
from darts.physics.properties.black_oil import (
    CapillaryPressurePcgo,
    CapillaryPressurePcow,
    DensityGas,
    DensityOil,
    DensityWat,
    GasRelPerm,
    OilRelPerm,
    ViscGas,
    ViscOil,
    ViscWat,
    WatRelPerm,
    flash_black_oil,
)


class BlackOilFluidProps(FluidProps):
    def __init__(self, pvt):
        super().__init__()
        self.components = ["g", "o", "w"]
        self.phases = ["gas", "oil", "water"]
        self.Mw = np.ones(len(self.components))

        self.pvt = pvt
        self.flash_ev = flash_black_oil(pvt)
        self.density = dict(
            [
                ('gas', DensityGas(pvt)),
                ('oil', DensityOil(pvt)),
                ('water', DensityWat(pvt)),
            ]
        )
        self.viscosity = dict(
            [('gas', ViscGas(pvt)), ('oil', ViscOil(pvt)), ('water', ViscWat(pvt))]
        )
        self.rel_perm = dict(
            [
                ('gas', GasRelPerm(pvt)),
                ('oil', OilRelPerm(pvt)),
                ('water', WatRelPerm(pvt)),
            ]
        )
        self.capillary_pressure = dict(
            [('pcow', CapillaryPressurePcow(pvt)), ('pcgo', CapillaryPressurePcgo(pvt))]
        )
