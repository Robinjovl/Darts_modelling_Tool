import abc
from typing import Literal

from pydantic import Field

from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    register_evaluator,
)


class EnthalpyBasicConfig(EvaluatorConfigBase):
    """Configuration for EnthalpyBasic evaluator."""

    kind: Literal["enthalpy_basic"] = "enthalpy_basic"
    tref: float = Field(273.15)
    hcap: float = Field(0.0357, ge=0)


class Enthalpy(EvaluatorBase):
    """Family ABC for enthalpy evaluators.

    Concrete subclasses implement :meth:`evaluate` returning a phase enthalpy
    for given pressure, temperature, and composition.
    """

    def __init__(self, components: list = None):
        """
        :param components: list of fluid component names (used only to compute ``self.nc``)
        :type components: list[str] | None
        """
        self.nc = len(components) if components is not None else 0

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, x):
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :param x: phase composition (molar fractions)
        :type x: list[float] | np.ndarray
        :return: phase enthalpy
        :rtype: float
        """
        pass


class EnthalpyBasic(Enthalpy):
    """Constant-heat-capacity enthalpy:
    ``h = hcap * (T - Tref)``.
    """

    def __init__(self, tref=273.15, hcap=0.0357):
        """
        :param tref: reference temperature [K]
        :type tref: float
        :param hcap: heat capacity
        :type hcap: float
        """
        super().__init__()
        self.tref = tref
        self.hcap = hcap

    def evaluate(
        self, pressure: float = None, temperature: float = None, x: list = None
    ):
        """
        :param pressure: pressure (unused)
        :type pressure: float | None
        :param temperature: temperature [K]
        :type temperature: float
        :param x: composition (unused)
        :type x: list[float] | None
        :return: enthalpy
        :rtype: float
        """
        # Enthalpy based on constant heat capacity
        enthalpy = self.hcap * (temperature - self.tref)
        return enthalpy


register_evaluator("enthalpy_basic", EnthalpyBasic, EnthalpyBasicConfig)
