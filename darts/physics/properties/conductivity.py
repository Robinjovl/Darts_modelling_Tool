import abc
import warnings
import numpy as np


class Conductivity:
    def __init__(self, components: list = None):
        self.nc = len(components) if components is not None else 0

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x, rho):
        pass


class ConductivityV(Conductivity):
    A = [105.161, 0.9007, 0.0007, 3.5e-15, 3.76e-10, 0.75, 0.0017]

    def __init__(self):
        super().__init__()

    def evaluate(self, pressure, temperature, x, rho):
        kappa = (self.A[0] + self.A[1] * rho + self.A[2] * rho ** 2 + self.A[3] * rho ** 3 * temperature ** 3 +
                 self.A[4] * rho ** 4 + self.A[5] * temperature + self.A[6] * temperature ** 2) / np.sqrt(temperature)
        kappa *= 1e-3 * 3600 * 24  # convert from W / m.K to kJ / m.day.K
        return kappa  # kJ / m.day.K


class ConductivityAq(Conductivity):
    def __init__(self, components: list, Mw: list, ions: list = None):
        super().__init__(components)
        self.Mw = Mw
        self.ni = len(ions) if ions is not None else 0

        self.H2O_idx = components.index("H2O") if "H2O" in components else None
        if not self.H2O_idx:
            warnings.warn("H2O not in list of components")

    def evaluate(self, pressure, temperature, x, rho):
        # Mass of dissolved salt
        S = 55.509 * x[self.nc] / x[self.H2O_idx] * self.Mw if self.ni else 0.

        T_d = temperature / 300
        cond_aq = 0.797015 * T_d ** (-0.194) - 0.251242 * T_d ** (-4.717) + 0.096437 * T_d ** (-6.385) - 0.032696 * T_d ** (-2.134)
        kappa = (cond_aq / (0.00022 * S + 1.)) + 0.00005 * (pressure - 50)
        kappa *= 86400
        return kappa


class ConductivityS(Conductivity):
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure, temperature, x, rho):
        return 0.6


class ConductivityH(Conductivity):
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure, temperature, x, rho):
        return 0.6
