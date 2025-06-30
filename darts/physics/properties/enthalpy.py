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

    def evaluate(
        self, pressure: float = None, temperature: float = None, x: list = None
    ):
        # Enthalpy based on constant heat capacity
        enthalpy = self.hcap * (temperature - self.tref)
        return enthalpy
