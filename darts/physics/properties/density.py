import abc
import warnings
import numpy as np


class Density:
    def __init__(self, components: list = None):
        self.nc = len(components) if components is not None else 0

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x):
        pass


class DensityBasic(Density):
    def __init__(self, dens0, compr=0., p0=1.):
        super().__init__()
        self.dens0 = dens0
        self.compr = compr
        self.p0 = p0

    def evaluate(self, pressure, temperature: float = None, x: list = None):
        return self.dens0 * (1 + self.compr * (pressure - self.p0))


class DensityBrineCO2(DensityBasic):
    def __init__(self, components, dens0=1000., compr=0., p0=1., co2_mult=0., ions_mult=0.):
        super().__init__(dens0, compr, p0)
        self.co2_mult = co2_mult

        if "CO2" in components:
            self.CO2_idx = components.index("CO2")
        else:
            self.CO2_idx = None

    def evaluate(self, pressure, temperature: float, x: list):
        if self.CO2_idx is not None:
            x_co2 = x[self.CO2_idx]
        else:
            x_co2 = 0.

        density = (self.dens0 + x_co2 * self.co2_mult) * (1 + self.compr * (pressure - self.p0))
        return density


class Density4Ions:
    def __init__(self, density, compressibility=0, p_ref=1, ions_fac=0):
        super().__init__()
        # Density evaluator class based on simple first order compressibility approximation (Taylor expansion)
        self.density_rc = density
        self.cr = compressibility
        self.p_ref = p_ref
        self.ions_fac = ions_fac

    def evaluate(self, pres, ion_liq_molefrac):
        return self.density_rc * (1 + self.cr * (pres - self.p_ref) + self.ions_fac * ion_liq_molefrac)


class Spivey2004(Density):
    """
    Correlation for brine density: Spivey et al. (2004) - Estimating Density, Formation Volume Factor, Compressibility,
                                                        Methane Solubility, and Viscosity for Oilfield Brines at
                                                        Temperatures From 0 to 275˚ C, Pressures to 200 MPa, and
                                                        Salinities to 5.7 mole/kg
    """
    aw = [[-0.127213, 0.645486, 1.03265, -0.070291, 0.639589],
          [4.221, -3.478, 6.221, 0.5182, -0.4405],
          [-11.403, 29.932, 27.952, 0.20684, 0.3768]]
    ab = [[-7.925e-5, -1.93e-6, -3.4254e-4, 0., 0.],
          [1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016],
          [-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685],
          [3.746e-4, -3.328e-4, -3.346e-4, 0., 0.],
          [0., 0., 0.1353, 0., 0.],
          [-1.409, -0.361, -0.2532, 0., 9.216],
          [0., 5.614, 4.6782, -0.307, 2.6069],
          [-0.1127, 0.2047, -0.0452, 0., 0.]]

    def __init__(self, components: list, ions: list = None):
        super().__init__(components)

        self.H2O_idx = components.index("H2O") if "H2O" in components else None
        if self.H2O_idx is None:
            warnings.warn("H2O not present")

        self.ions = ions
        self.ni = len(ions) if ions is not None else 0

    def evaluate(self, pressure, temperature, x):
        tc = temperature - 273.15  # Temp in [Celcius]
        tc_100 = tc/100  # needed many times
        p0 = 700  # reference pressure of 70 MPa

        # Pure water density
        a_w = np.empty(3)
        for i in range(3):
            a_w[i] = (self.aw[i][0] * tc_100 ** 2 + self.aw[i][1] * tc_100 + self.aw[i][2]) / \
                     (self.aw[i][3] * tc_100 ** 2 + self.aw[i][4] * tc_100 + 1.)

        rho_w0 = a_w[0]
        Ew = a_w[1]
        Fw = a_w[2]

        if self.ni == 0:
            # Pure water
            Iw = (1. / Ew) * np.log(np.abs(Ew * (pressure / p0) + Fw))
            Iw0 = (1. / Ew) * np.log(np.abs(Ew * (p0 / p0) + Fw))
            rho = 1000 * rho_w0 * np.exp(Iw - Iw0)
        else:
            # Brine density
            Cm = 55.509 * x[self.nc] / x[self.H2O_idx]

            a_b = np.empty(8)
            for i in range(8):
                a_b[i] = (self.ab[i][0] * tc_100 ** 2 + self.ab[i][1] * tc_100 + self.ab[i][2]) / \
                         (self.ab[i][3] * tc_100 ** 2 + self.ab[i][4] * tc_100 + 1.)
            rho_b0 = rho_w0 + a_b[0] * Cm ** 2 + a_b[1] * Cm ** 1.5 + a_b[2] * Cm + a_b[3] * np.sqrt(Cm)
            Eb = Ew + a_b[4] * Cm
            Fb = Fw + a_b[5] * Cm ** 1.5 + a_b[6] * Cm + a_b[7] * np.sqrt(Cm)

            Ib = (1. / Eb) * np.log(np.abs(Eb * (pressure / p0) + Fb))
            Ib0 = (1. / Eb) * np.log(np.abs(Eb * (p0 / p0) + Fb))
            rho = 1000 * rho_b0 * np.exp(Ib - Ib0)

        return rho  # kg/m3


class Garcia2001(Spivey2004):
    """
    Correlation for brine density with dissolved CO2: Garcia (2001) - Density of aqueous solutions of CO2
    """
    def __init__(self, components: list, ions: list = None):
        super().__init__(components, ions)

        self.CO2_idx = components.index("CO2") if "CO2" in components else None

    def evaluate(self, pressure, temperature, x):
        """"""
        rho_b = super().evaluate(pressure, temperature, x)

        # If CO2 is present, correct density
        if self.CO2_idx is not None:
            # Apparent molar volume of dissolved CO2
            tc = temperature - 273.15  # Temp in [Celcius]
            V_app = (37.51 - 9.585e-2 * tc + 8.740e-4 * tc ** 2 - 5.044e-7 * tc ** 3) * 1e-6  # in [m3 / mol]

            mCO2 = 55.509 * x[self.CO2_idx] / (x[self.H2O_idx])
            MW = 44.01  # molecular weight of CO2
            rho = (1. + mCO2 * MW * 1e-3) / (mCO2 * V_app + 1. / rho_b)  # in [kg / m3]
        else:
            rho = rho_b

        return rho
