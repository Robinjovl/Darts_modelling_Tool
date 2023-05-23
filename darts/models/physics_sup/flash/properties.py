import numpy as np
import warnings


# region Flash
class Flash:
    def __init__(self, nph, nc):
        self.nph = nph
        self.nc = nc

    def evaluate(self, pressure, temperature, zc):
        pass


class SolidFlash:
    def __init__(self, flash: Flash, nc_sol: int = 0, np_sol: int = 0):
        self.flash = flash

        self.nc_fl = flash.nc
        self.np_fl = flash.nph
        self.nc_sol = nc_sol
        self.np_sol = np_sol

    def evaluate(self, pressure, temperature, zc):
        """Evaluate flash normalized for solids"""
        # Normalize compositions
        zc_sol = zc[self.nc_fl:]
        zc_sol_tot = np.sum(zc_sol)
        zc_norm = zc[:self.nc_fl]/(1.-zc_sol_tot)

        # Evaluate flash for normalized composition
        nu, x = self.flash.evaluate(pressure, temperature, zc_norm)

        # Re-normalize solids and append to nu, x
        NU = np.zeros(self.np_fl + self.np_sol)
        X = np.zeros((self.np_fl + self.np_sol, self.nc_fl + self.nc_sol))
        for j in range(self.np_fl):
            NU[j] = nu[j] * (1.-zc_sol_tot)
            X[j, :self.nc_fl] = x[j, :]

        for j in range(self.np_sol):
            NU[self.np_fl+j] = zc_sol[j]
            X[self.np_fl+j, self.nc_fl+j] = 1.

        return NU, X
# endregion


# region Density
class Density:
    def __init__(self, components: list):
        self.nc = len(components) if components is not None else 0

    def evaluate(self, pressure, temperature, x):
        pass


class DensityBrine(Density):
    """
    Spivey (2004) correlation for brine density
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


class DensityBrineCO2(DensityBrine):
    """
    Garcia (2001) correlation for brine density with dissolved CO2
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
# endregion


# region Viscosity
class Viscosity:
    def __init__(self, components: list):
        self.nc = len(components) if components is not None else 0

    def evaluate(self, pressure, temperature, x, rho):
        pass


class ViscosityCO2(Viscosity):
    """
    Correlation for CO2 viscosity: Fenghour, Wakeham & Vesovic (1998)
    """
    a = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
    d = [0.4071119e-2, 0.7198037e-4, 0.2411967e-16, 0.2971072e-22, -0.1627888e-22]  # d11, d21, d64, d81, d82

    def __init__(self, components: list = None):
        super().__init__(components)

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


class ViscosityLee(Viscosity):
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


class ViscosityAq(Viscosity):
    a = 9.03591045e1
    b = [3.40285740e4, 8.23556123e8, -9.28022905e8]
    c = [1.40090092e-2, 4.86126399e-2, 5.26696663e-2]
    d = [-1.22757462e-1, 2.15995021e-2, -3.65253919e-4, 1.97270835e-6]

    def __init__(self, components: list):
        super().__init__(components)

        self.CO2_idx = components.index("CO2") if "CO2" in components else None

    def evaluate(self, pressure, temperature, x, rho):
        """
        Correlation for brine + NaCl + CO2 viscosity
        Brine + NaCl: Mao and Duan (2008)
        Brine + NaCl + CO2: Islam (2012)
        """
        # Viscosity of pure water
        muH2O = self.a
        for i in range(3):
            muH2O += self.b[i] * np.exp(-self.c[i] * temperature)
        for i in range(4):
            muH2O += pressure*0.1 * self.d[i] * (temperature-293.15)**i

        # Viscosity of H2O + salt
        mu_r = 1.
        mu = mu_r * muH2O

        # Viscosity of Aq + CO2
        if self.CO2_idx is not None:
            mu *= 1 + 4.65 * pow(x[self.CO2_idx], 1.0134)

        return mu * 1e-3
# endregion


# region Enthalpy
class Enthalpy:
    def __init__(self, components: list):
        self.nc = len(components)

    def evaluate(self, pressure, temperature, x):
        pass


class EnthalpyIdeal(Enthalpy):
    R = 8.3145
    T_0 = 298.15

    hi_0 = {"H2O": -242000., "CO2": -393800., "N2": 0., "H2S": -20200.,
            "C1": -74900., "C2": -84720., "C3": -103900., "iC4": -134600., "nC4": -126200.,
            "iC5": -165976., "nC5": -146500., "nC6": -167300., "nC7": -187900., "NaCl": -411153.00
            }
    hi_a = {"H2O": [3.8747*R, 0.0231E-2*R, 0.1269E-5*R, -0.4321E-9*R],
            "CO2": [2.6751*R, 0.7188E-2*R, -0.4208E-5*R, 0.8977E-9*R],
            "N2": [3.4736*R, -0.0189E-2*R, 0.0971E-5*R, -0.3453E-9*R],
            "H2S": [3.5577*R, 0.1574E-2*R, 0.0686E-5*R, -0.3959E-9*R],
            "C1": [2.3902*R, 0.6039E-2*R, 0.1525E-5*R, -1.3234E-9*R],
            "C2": [0.8293*R, 2.0752E-2*R, -0.7699E-5*R, 0.8756E-9*R],
            "C3": [-0.4861*R, 3.6629E-2*R, -1.8895E-5*R, 3.8143E-9*R],
            "iC4": [-0.9511*R, 4.9999E-2*R, -2.7651E-5*R, 5.9982E-9*R],
            "nC4": [0.4755*R, 4.4650E-2*R, -2.2041E-5*R, 4.2068E-9*R],
            "iC5": [-1.9942*R, 6.6725E-2*R, -3.9738E-5*R, 9.1735E-9*R],
            "nC5": [0.8142*R, 5.4598E-2*R, -2.6997E-5*R, 5.0824E-9*R],
            "nC6": [0.8338*R, 6.6373E-2*R, -3.444E-5*R, 6.9342E-9*R],
            "nC7": [-0.6184*R, 8.1268E-2*R, -4.388E-5*R, 9.2037E-9*R],
            "NaCl": [5.526*R, 0.1963e-2*R, 0., 0.]
            }

    def __init__(self, components):
        super().__init__(components)
        self.components = components

    def evaluate(self, pressure, temperature, x):
        Hi = 0.
        for i, comp in enumerate(self.components):
            Hi_i = self.hi_0[comp]\
                   + self.hi_a[comp][0] * (temperature - self.T_0)\
                   + self.hi_a[comp][1] / 2. * (temperature ** 2 - self.T_0 ** 2)\
                   + self.hi_a[comp][2] / 3. * (temperature ** 3 - self.T_0 ** 3)\
                   + self.hi_a[comp][3] / 4. * (temperature ** 4 - self.T_0 ** 4)
            Hi += x[i] * Hi_i

        return Hi


class EnthalpyGuo(Enthalpy):
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x):
        return 0.
# endregion


# region Conductivity
class Conductivity:
    def __init__(self, components: list):
        self.nc = len(components)

    def evaluate(self, pressure, temperature, x, rho):
        pass


class ConductivityV(Conductivity):
    A = [105.161, 0.9007, 0.0007, 3.5e-15, 3.76e-10, 0.75, 0.0017]

    def __init__(self, components: list):
        super().__init__(components)

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
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x, rho):
        return 0.6


class ConductivityH(Conductivity):
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x, rho):
        return 0.6
# endregion


# region Kinetics
class Kinetics:
    def __init__(self, stoich: list):
        self.stoich = stoich

    def evaluate(self, pressure, temperature, x, sat):
        pass

    def evaluate_enthalpy(self, pressure, temperature, x, sat):
        pass


class HydrateReactionRate(Kinetics):
    def __init__(self, components: list, Mw, flash: SolidFlash, hydrate_eos, fluid_eos: list, stoich: list = None,
                 perm=300., poro=0.2, k=3.11e12, enthalpy: bool = False):
        super().__init__(stoich)

        self.flash = flash
        self.hydrate_eos = hydrate_eos
        self.fluid_eos = fluid_eos

        self.water_idx = components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1
        self.hydrate_idx = components.index("H")

        self.stoich = stoich
        if stoich is not None:
            self.nH = stoich[self.water_idx]/stoich[self.guest_idx]
        else:
            self.nH = None
        self.Mw = Mw

        self.K = k  # reaction constant [kmol/(m^2 bar day)]
        self.perm = perm * 1E-15  # mD to m2
        self.poro = poro

        self.enthalpy = enthalpy

    def evaluate(self, pressure, temperature, x, sat):
        # Calculate fugacity difference between water in fluid phases and water in hydrate phase
        if x[0, 0] != 0.:
            f0 = self.flash.flash.fugacity(pressure, temperature, x[0, :], self.fluid_eos[0])
        else:
            f0 = self.flash.flash.fugacity(pressure, temperature, x[1, :], self.fluid_eos[1])

        self.hydrate_eos.component_parameters(pressure, temperature)
        fwH = self.hydrate_eos.fwH(f0)

        df = fwH - f0[self.water_idx]  # if df < 0 formation, if df > 0 dissociation
        xH = self.hydrate_eos.xH()

        # Reaction rate following Yin (2018)
        # surface area
        F_A = 1
        r_p = np.sqrt(45 * self.perm * (1 - self.poro) ** 2 / (self.poro ** 3))
        A_s = 0.879 * F_A * (1 - self.poro) / r_p * sat ** (2 / 3)  # hydrate surface area [m2]

        # Thermodynamic parameters
        dE = -81E3  # activation energy [J/mol]
        R = 8.3145  # gas constant [J/(K.mol)]

        # K is reaction cons, A_s hydrate surface area, dE activation energy, driving force is fugacity difference
        self.rate = self.K * A_s * np.exp(dE / (R * temperature)) * df

        return [stoich * self.rate for stoich in self.stoich]

    def evaluate_enthalpy(self, pressure, temperature, x, sat):
        if self.enthalpy:
            # Enthalpy change with dissociation (-ive, rate of hydrate component +ive)
            Cf = 33.72995  # J/kg cal/gmol
            if temperature - 273.15 > 0:
                (C1, C2) = (13521, -4.02)
            else:
                (C1, C2) = (6534, -11.97)
            en = Cf * (C1 + C2 / temperature) * 1e-3  # kJ/kg

            # Calculate molar weight of hydrate component
            if self.nH is None:
                mH = np.sum(x * self.Mw)
            else:
                mH = self.Mw[self.water_idx] * self.nH + self.Mw[self.guest_idx]

            H_diss = -en * mH  # kJ/kg * kg/kmol -> kJ/kmol

            return self.rate * H_diss  # rate * enth/mol
        else:
            return 0.
# endregion
