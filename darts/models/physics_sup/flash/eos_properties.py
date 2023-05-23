import numpy as np


class EoSProperty:
    def __init__(self, eos):
        self.eos = eos


class EoSDensity(EoSProperty):
    def __init__(self, eos, Mw):
        super().__init__(eos)
        self.Mw = Mw

    def evaluate(self, pressure, temperature, x):
        MW = np.sum(x * self.Mw) * 1e-3  # kg/mol
        return MW / self.eos.V(pressure, temperature, x)  # kg/mol / m3/mol -> kg/m3


from .properties import EnthalpyIdeal
class EoSEnthalpy(EoSProperty):
    def __init__(self, eos, h_ideal: EnthalpyIdeal):
        super().__init__(eos)
        self.h_ideal = h_ideal

    def evaluate(self, pressure, temperature, x):
        Hi = self.h_ideal.evaluate(pressure, temperature, x)
        Hr = self.eos.Hr_TP(pressure, temperature, x)
        return Hi + Hr  # J/mol == kJ/kmol
