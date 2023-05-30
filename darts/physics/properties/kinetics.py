import abc
import warnings
import numpy as np
from .flash import SolidFlash


class Kinetics:
    def __init__(self, stoich: list):
        self.stoich = stoich

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x, sat):
        pass

    def evaluate_enthalpy(self, pressure, temperature, x, sat):
        return 0.


class KineticsBasic(Kinetics):
    """
    Law of Mass Action for kinetic reaction
    For reaction aA + bB <-> cC: rate = c * (1 - Q/K) with Q = [C]^c / [A]^a [B]^b
    """
    def __init__(self, stoich: list, prod: list, equi_prod: float, kin_rate_cte: float, combined_ions=True):
        super().__init__(stoich)

        self.prod = prod
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.combined_ions = combined_ions
        self.ne = len(stoich)

    def evaluate(self, pressure, temperature, x, sat_sol):
        # For reaction aA + bB <-> cC
        # Calculate activity product Q = [C]^c / [A]^a [B]^b
        prod = 1.
        for i in range(self.ne):
            prod *= x[i] ** self.stoich[i]

        # Calculate rate = c * As * (1-Q/K)
        rate = self.kin_rate_cte * sat_sol * (1. - prod / self.equi_prod)

        return [stoich * rate for stoich in self.stoich]


class HydrateKinetics(Kinetics):
    def __init__(self, components: list, Mw, flash: SolidFlash, hydrate_eos, fluid_eos: list, stoich: list = None,
                 perm=300., poro=0.2, k=3.11e12, enthalpy: bool = False):
        super().__init__(stoich)

        self.flash = flash
        self.hydrate_eos = hydrate_eos
        self.fluid_eos = fluid_eos

        self.water_idx = components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1
        # self.hydrate_idx = components.index("H")

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

    def calc_df(self, pressure, temperature, x):
        # Calculate fugacity difference between water in fluid phases and water in hydrate phase
        if x[0, 0] != 0.:
            f0 = self.flash.flash.fugacity(pressure, temperature, x[0, :], self.fluid_eos[0])
        else:
            f0 = self.flash.flash.fugacity(pressure, temperature, x[1, :], self.fluid_eos[1])

        self.hydrate_eos.component_parameters(pressure, temperature)
        fwH = self.hydrate_eos.fw(f0)

        df = fwH - f0[self.water_idx]  # if df < 0 formation, if df > 0 dissociation
        xH = self.hydrate_eos.xH()

        return df, xH

    def evaluate(self, pressure, temperature, x, sat):
        df, xH = self.calc_df(pressure, temperature, x)

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
