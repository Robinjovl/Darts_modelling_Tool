import abc
import warnings
import numpy as np


class Enthalpy:
    def __init__(self, components: list = None):
        self.nc = len(components) if components is not None else 0

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x):
        pass


class EnthalpyBasic(Enthalpy):
    def __init__(self, tref=273.15, hcap=0.0357):
        super().__init__()
        self.tref = tref
        self.hcap = hcap

    def evaluate(self, pressure: float = None, temperature: float = None, x: list = None):
        # Enthalpy based on constant heat capacity
        enthalpy = self.hcap * (temperature - self.tref)
        return enthalpy


class EnthalpyIdeal(Enthalpy):
    """
    Ideal gas enthalpy: Jager et al. (2003) - The next generation of hydrate prediction II.
                                            Dedicated aqueous phase fugacity model for hydrate prediction
    """
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
    """
    Correlation for brine enthalpy with dissolved gases: Guo et al. (2019) - An enthalpy model of CO2-CH4-H2S-N2-brine
                                                                            systems applied in simulation of non-
                                                                            isothermal multiphase and multicomponent flow
                                                                            with high pressure, temperature and salinity
    """
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x):
        return 0.
