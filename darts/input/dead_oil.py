import numpy as np

from darts.input.input_data import FluidProps
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, DensityBrineCO2


class DeadOilFluidProps(FluidProps):
    """
    Fluid-property preset for two- or three-phase dead-oil physics.

    :param n_phases: Number of fluid phases. Supported values are 2 and 3.
    :type n_phases: int
    """

    def __init__(self, n_phases: int = 2):
        super().__init__()
        self.n_phases = n_phases

        if n_phases == 2:
            self.components = ["Zo", "Zw"]
            self.phases = ['oil', 'water']
            self.density = {
                'water': DensityBasic(compr=1e-5, dens0=1014),
                'oil': DensityBasic(compr=5e-3, dens0=700),
            }
            self.viscosity = {
                'water': ConstFunc(0.89),
                'oil': ConstFunc(1),
            }
            self.rel_perm = {
                'water': PhaseRelPerm("water", 0.1, 0.1),
                'oil': PhaseRelPerm("oil", 0.1, 0.1),
            }
        elif n_phases == 3:
            self.components = ["g", "o", "w"]
            self.phases = ["gas", "oil", "water"]
            self.density = {
                'gas': DensityBasic(compr=1e-3, dens0=200),
                'oil': DensityBasic(compr=1e-5, dens0=600),
                'water': DensityBrineCO2(
                    self.components, compr=1e-5, dens0=1000, co2_mult=0
                ),
            }
            self.viscosity = {
                'gas': ConstFunc(0.05),
                'oil': ConstFunc(0.5),
                'water': ConstFunc(0.5),
            }
            self.rel_perm = {
                'gas': PhaseRelPerm("gas"),
                'oil': PhaseRelPerm("oil"),
                'water': PhaseRelPerm("water"),
            }
        else:
            raise ValueError(
                f"DeadOilFluidProps supports 2 or 3 phases, received {n_phases}"
            )

        self.Mw = np.ones(len(self.components))
