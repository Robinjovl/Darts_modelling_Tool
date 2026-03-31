import abc

from pydantic import BaseModel, Field


class EnthalpyBasicConfig(BaseModel):
    """Configuration for EnthalpyBasic evaluator."""

    tref: float = Field(273.15)
    hcap: float = Field(0.0357, ge=0)


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

    @classmethod
    def from_config(cls, config: EnthalpyBasicConfig) -> "EnthalpyBasic":
        """Construct from a validated config object."""
        return cls(tref=config.tref, hcap=config.hcap)
