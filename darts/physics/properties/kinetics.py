import abc
import warnings
import numpy as np
from darts.physics.properties.flash import SolidFlash


class Kinetics:
    def __init__(self, stoich: list):
        self.stoich = stoich

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x, sat):
        pass

    def evaluate_enthalpy(self, pressure, temperature, x, sat):
        return 0.


class KineticBasic:
    def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.kinetic_rate = np.zeros(ne)
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, x, nu_sol):
        if self.combined_ions:
            ion_prod = (x[1][1] / 2) ** 2
            dQ = (1 - ion_prod / self.equi_prod)
            self.kinetic_rate[1] = - self.kin_rate_cte * dQ * nu_sol
            self.kinetic_rate[-1] = - 0.5 * self.kinetic_rate[1]
        else:
            ion_prod = x[1][1] * x[1][2]
            dQ = (1 - ion_prod / self.equi_prod)
            self.kinetic_rate[1] = - self.kin_rate_cte * dQ * nu_sol
            self.kinetic_rate[2] = - self.kin_rate_cte * dQ * nu_sol
            self.kinetic_rate[-1] = - self.kinetic_rate[1]

        return self.kinetic_rate, dQ


class LawOfMassAction(Kinetics):
    """
    Law of Mass Action for kinetic reaction
    For reaction aA + bB <-> cC: rate = c * (1 - Q/K) with Q = [C]^c / [A]^a [B]^b
    """
    def __init__(self, stoich: list, nc_fl: int, fl_idx: int, equi_prod: float, kin_rate_cte: float):
        super().__init__(stoich)

        self.nc_fl = nc_fl
        self.fl_idx = fl_idx
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte

    def evaluate(self, pressure, temperature, x, sat_sol):
        # For reaction aA + bB <-> cC
        # Calculate activity product Q = [C]^c / [A]^a [B]^b
        prod = 1.
        for i in range(self.nc_fl):
            prod = prod * x[self.fl_idx, i] ** self.stoich[i] if self.stoich[i] != 0 else prod

        # Calculate rate = c * As * (1-Q/K)
        dQ = (1. - prod / self.equi_prod)
        rate = self.kin_rate_cte * sat_sol * dQ

        return [stoich * rate for stoich in self.stoich], dQ


class HydrateKinetics(Kinetics):
    def __init__(self, components: list, phases: list, Mw, flash: SolidFlash, hydrate_eos, fluid_eos: list,
                 stoich: list = None, perm: float = 300., poro: float = 0.2, k: float = None, F_a=1.,
                 moridis: bool = True, enthalpy: bool = False):
        super().__init__(stoich)

        self.flash = flash
        self.hydrate_eos = hydrate_eos
        self.fluid_eos = fluid_eos

        self.water_idx = components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1
        # self.hydrate_idx = components.index("H")

        self.a_idx = phases.index("Aq")
        self.v_idx = phases.index("V")
        self.h_idx = phases.index("sI")

        self.stoich = stoich
        if stoich is not None:
            self.nH = stoich[self.water_idx]/stoich[self.guest_idx]
        else:
            self.nH = None
        self.Mw = Mw

        self.K = k if k is not None else 3.6e6*86400  # reaction constant [kmol/(m^2 bar day)]
        self.F_a = F_a
        self.perm = perm * 1E-15  # mD to m2
        self.poro = poro

        if moridis:
            r_p = np.sqrt(45. * self.perm * (1 - self.poro) ** 2 / self.poro ** 3)
            self.A_s = lambda sat: 0.879 * self.F_a * (1 - self.poro) / r_p * sat[self.h_idx] ** (2. / 3.)
        else:
            self.K = 8.06 / Mw[-1] * 1e5 * 86400  # kg/m2.Pa.s
            r_p = 3.75e-4
            beta = 2. / 3.
            self.A_s = lambda sat: (0.879 * (1 - self.poro) / r_p *
                                    sat[self.v_idx] ** (2. / 3.) * sat[self.a_idx] ** beta * (1 - sat[self.h_idx]) ** beta)

        self.enthalpy = enthalpy

    def calc_df(self, pressure, temperature, x):
        # Calculate fugacity difference between water in fluid phases and water in hydrate phase
        if x[0, 0] != 0.:
            f0 = self.fluid_eos[0].fugacity(pressure, temperature, x[0, :])
        else:
            f0 = self.fluid_eos[1].fugacity(pressure, temperature, x[1, :])

        fwH = self.hydrate_eos.fw(pressure, temperature, f0)

        df = fwH - f0[self.water_idx]  # if df < 0 formation, if df > 0 dissociation
        xH = self.hydrate_eos.xH()

        return df, xH

    def evaluate(self, pressure, temperature, x, sat: list):
        df, xH = self.calc_df(pressure, temperature, x)

        # Reaction rate following Yin (2018)
        # surface area
        A_s = self.A_s(sat)

        # Thermodynamic parameters
        dE = -81E3  # activation energy [J/mol]
        R = 8.3145  # gas constant [J/(K.mol)]

        # K is reaction cons, A_s hydrate surface area, dE activation energy, driving force is fugacity difference
        self.rate = self.K * A_s * np.exp(dE / (R * temperature)) * df

        return [stoich * self.rate for stoich in self.stoich], df

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

            return self.rate * H_diss  # rate [kmol/day] * [kJ/kmol] = [kJ/day]
        else:
            return 0.
