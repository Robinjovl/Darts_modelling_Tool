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
    def __init__(
        self,
        pvt,
        gas_phase_name: str = "gas",
        oil_phase_name: str = "oil",
        water_phase_name: str = "water",
    ):
        """
        :param pvt: Path to the PVT input file.
        :param gas_phase_name: Label for the gas phase in ``self.phases`` and the
            density/viscosity/rel_perm dict keys, default ``"gas"``.
        :param oil_phase_name: Label for the oil phase, default ``"oil"``.
        :param water_phase_name: Label for the aqueous phase, default ``"water"``.
            Override any of these (e.g. ``"wat"``) to match an existing model's ref files
        """
        super().__init__()
        self.components = ["g", "o", "w"]
        self.phases = [gas_phase_name, oil_phase_name, water_phase_name]
        self.Mw = np.ones(len(self.components))

        self.pvt = pvt
        self.flash_ev = flash_black_oil(pvt)
        self.density = dict(
            [
                (gas_phase_name, DensityGas(pvt)),
                (oil_phase_name, DensityOil(pvt)),
                (water_phase_name, DensityWat(pvt)),
            ]
        )
        self.viscosity = dict(
            [
                (gas_phase_name, ViscGas(pvt)),
                (oil_phase_name, ViscOil(pvt)),
                (water_phase_name, ViscWat(pvt)),
            ]
        )
        self.rel_perm = dict(
            [
                (gas_phase_name, GasRelPerm(pvt)),
                (oil_phase_name, OilRelPerm(pvt)),
                (water_phase_name, WatRelPerm(pvt)),
            ]
        )
        self.capillary_pressure = dict(
            [('pcow', CapillaryPressurePcow(pvt)), ('pcgo', CapillaryPressurePcgo(pvt))]
        )
