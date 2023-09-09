import abc
import warnings
import numpy as np


class Viscosity:
    def __init__(self, components: list = None):
        self.nc = len(components) if components is not None else 0

    def evaluate(self, pressure, temperature, x, rho):
        pass


class Fenghour1998(Viscosity):
    """
    Correlation for CO2 viscosity: Fenghour, Wakeham & Vesovic (1998) - The Viscosity of CO2
    """
    a = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
    d = [0.4071119e-2, 0.7198037e-4, 0.2411967e-16, 0.2971072e-22, -0.1627888e-22]  # d11, d21, d64, d81, d82

    def __init__(self):
        super().__init__()

    def evaluate(self, pressure, temperature, x, rho):
        # Viscosity in zero density limit
        eps_k = 251.196  # energy scaling parameter eps/k [K]
        T_ = 1/eps_k * temperature
        lnT_ = np.log(T_)

        lnG = self.a[0]
        for i in range(1, 5):
            lnG += self.a[i] * lnT_ ** i
        G = np.exp(lnG)

        n0 = 1.00697*np.sqrt(temperature) / G

        # Excess viscosity
        dn = self.d[0] * rho + self.d[1] * rho ** 2 + self.d[2] * rho ** 6 / (T_ ** 3) + \
             self.d[3] * rho ** 8 + self.d[4] * rho ** 8 / T_

        # Correction of viscosity in vicinity of critical point
        dnc = 0

        n = (n0 + dn + dnc) * 1e-3  # muPa.s to cP
        return n


class Lee1966(Viscosity):
    """
    Correlation for gas mixture viscosity: Lee et al. (1966) - The Viscosity of Natural Gases
    """
    def __init__(self, components: list, Mw: list):
        super().__init__(components)
        self.Mw = Mw

    def evaluate(self, pressure, temperature, x, rho):
        MW = np.sum(x * self.Mw) * 1e-3  # kg/mol

        T_ran = temperature * 1.8  # rankine scale
        a = (9.379 + 0.0160 * MW) * T_ran ** 1.5 / (209.2 + 19.26 * MW + T_ran)
        b = 3.448 + 0.01009 * MW + (986.4 / T_ran)
        c = 2.447 - 0.2224 * b
        rho_lb = rho * 0.0624279606  # convert rho from [kg / m3] -> [lb / ft ^ 3]
        return 1E-4 * a * np.exp(b * (rho_lb / 62.43) ** c)


class MaoDuan2009(Viscosity):
    """
    Correlation for brine + NaCl viscosity: Mao & Duan (2009) - The Viscosity of Aqueous Alkali-Chloride Solutions
                                                                up to 623 K, 1,000 bar, and High Ionic Strength
    """
    a = 9.03591045e1
    b = [3.40285740e4, 8.23556123e8, -9.28022905e8]
    c = [1.40090092e-2, 4.86126399e-2, 5.26696663e-2]
    d = [-1.22757462e-1, 2.15995021e-2, -3.65253919e-4, 1.97270835e-6]

    def __init__(self, components: list, ions: list = None):
        super().__init__(components)
        self.ions = ions
        self.ni = len(ions) if ions is not None else 0

    def evaluate(self, pressure, temperature, x, rho):
        # Viscosity of pure water
        muH2O = self.a
        for i in range(3):
            muH2O += self.b[i] * np.exp(-self.c[i] * temperature)
        for i in range(4):
            muH2O += pressure*0.1 * self.d[i] * (temperature-293.15)**i

        # Viscosity of H2O + salt
        mu_r = 1.
        mu = mu_r * muH2O

        return mu * 1e-3


class Islam2012(MaoDuan2009):
    """
    Correlation for brine + NaCl + CO2 viscosity: Islam & Carlson (2012) - Viscosity Models and Effects of Dissolved CO2
    """
    def __init__(self, components: list, ions: list = None):
        super().__init__(components, ions)
        self.CO2_idx = components.index("CO2") if "CO2" in components else None

    def evaluate(self, pressure, temperature, x, rho):
        mu_brine = super().evaluate(pressure, temperature, x, rho)

        # Viscosity of Aq + CO2
        if self.CO2_idx is not None:
            mu_brine *= 1. + 4.65 * pow(x[self.CO2_idx], 1.0134)

        return mu_brine
