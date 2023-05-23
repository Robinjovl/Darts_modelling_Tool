import numpy as np
from .properties import Flash
from dartsflash import Flash as F, NegativeFlash2
import warnings


class EoSFlash(Flash):
    flash: F

    def __init__(self, nph, nc):
        super().__init__(nph, nc)

    def evaluate(self, pressure, temperature, zc):
        error_output = self.flash.evaluate(pressure, temperature, zc)
        if error_output:
            warnings.warn("Flash unable to converge")

        nu = self.flash.getnu()
        X = self.flash.getx()

        x = np.zeros((self.nph, self.nc))
        for j in range(self.nph):
            x[j, :] = X[j * self.nc:(j + 1) * self.nc]

        return nu, x

    def fugacity(self, pressure, temperature, x, j):
        return self.flash.fugacity(pressure, temperature, x, j)


class NF2(EoSFlash):
    def __init__(self, nc, flash_params):
        super().__init__(nph=2, nc=nc)
        self.flash = NegativeFlash2(flash_params)

    def evaluate(self, pressure, temperature, zc):
        nu, x = super().evaluate(pressure, temperature, zc)

        if nu[0] > 0. and nu[1] > 0.:
            return nu, x
        elif nu[0] > 1.:
            return np.array([1., 0.]), np.array([zc, np.zeros(self.nc)])
        else:  # if nu[1] > 1.:
            return np.array([0., 1.]), np.array([np.zeros(self.nc), zc])


# class SF2(EoSFlash):
#     def __init__(self, nc, flash_params):
#         super().__init__(nph=2, nc=nc)
#         self.flash = StabilityFlash2(flash_params)
