from typing import Any, Literal

from darts.interpolators import property_evaluator_iface
from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    register_evaluator,
)


class _EvaluatorIfaceMeta(type(property_evaluator_iface), type(EvaluatorBase)):
    """Combined metaclass so a class can inherit both the C++ pybind11
    ``property_evaluator_iface`` and the Python ``EvaluatorBase`` ABC."""


class CustomRockEnergyEvaluatorConfig(EvaluatorConfigBase):
    """Configuration for custom_rock_energy_evaluator."""

    kind: Literal["custom_rock_energy"] = "custom_rock_energy"
    rock: list[list[Any]]


class CustomRockCompactionEvaluatorConfig(EvaluatorConfigBase):
    """Configuration for custom_rock_compaction_evaluator."""

    kind: Literal["custom_rock_compaction"] = "custom_rock_compaction"
    rock: list[list[Any]]


class custom_rock_energy_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """Linear rock-enthalpy evaluator: ``h_rock = (T - Tref)`` using the third
    column of the rock table as ``Tref``."""

    def __init__(self, rock):
        """
        :param rock: rock table rows (each row holds at least
            ``[p_ref, compressibility, T_ref]``)
        :type rock: list[list[float]]
        """
        super().__init__()
        self.rock_table = rock

    def evaluate(self, state, temperature):
        """
        :param state: state vector (unused)
        :type state: list[float] | np.ndarray
        :param temperature: temperature [K]
        :type temperature: float
        :return: rock enthalpy
        :rtype: float
        """
        temperature_ref = self.rock_table[0][2]
        heat_constant = 1
        return heat_constant * (temperature - temperature_ref)

    def to_config(self) -> CustomRockEnergyEvaluatorConfig:
        """Build Config explicitly because the storage attribute
        ``rock_table`` diverges from the Config field name ``rock``.

        :return: serialized config
        :rtype: CustomRockEnergyEvaluatorConfig
        """
        return CustomRockEnergyEvaluatorConfig(
            rock=[list(row) for row in self.rock_table]
        )

    @classmethod
    def from_config(
        cls, config: CustomRockEnergyEvaluatorConfig
    ) -> "custom_rock_energy_evaluator":
        """Build instance explicitly because the constructor uses ``rock`` while
        the class stores it as ``rock_table``.

        :param config: validated config
        :type config: CustomRockEnergyEvaluatorConfig
        :return: custom_rock_energy_evaluator instance
        :rtype: custom_rock_energy_evaluator
        """
        return cls(rock=[list(row) for row in config.rock])


register_evaluator(
    "custom_rock_energy",
    custom_rock_energy_evaluator,
    CustomRockEnergyEvaluatorConfig,
)


class custom_rock_compaction_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """Linear rock-compaction multiplier from the rock table:
    ``mult = 1 + cr * (p - p_ref)``."""

    def __init__(self, rock):
        """
        :param rock: rock table rows (each row holds ``[p_ref, compressibility, ...]``)
        :type rock: list[list[float]]
        """
        super().__init__()
        self.rock_table = rock

    def evaluate(self, state):
        """
        :param state: state vector with pressure as the first element
        :type state: list[float] | np.ndarray
        :return: rock compaction multiplier
        :rtype: float
        """
        pressure = state[0]
        pressure_ref = self.rock_table[0][0]
        compressibility = self.rock_table[0][1]
        return 1.0 + compressibility * (pressure - pressure_ref)

    def to_config(self) -> CustomRockCompactionEvaluatorConfig:
        """Build Config explicitly because the storage attribute
        ``rock_table`` diverges from the Config field name ``rock``.

        :return: serialized config
        :rtype: CustomRockCompactionEvaluatorConfig
        """
        return CustomRockCompactionEvaluatorConfig(
            rock=[list(row) for row in self.rock_table]
        )

    @classmethod
    def from_config(
        cls, config: CustomRockCompactionEvaluatorConfig
    ) -> "custom_rock_compaction_evaluator":
        """Build instance explicitly because the constructor uses ``rock`` while
        the class stores it as ``rock_table``.

        :param config: validated config
        :type config: CustomRockCompactionEvaluatorConfig
        :return: custom_rock_compaction_evaluator instance
        :rtype: custom_rock_compaction_evaluator
        """
        return cls(rock=[list(row) for row in config.rock])


register_evaluator(
    "custom_rock_compaction",
    custom_rock_compaction_evaluator,
    CustomRockCompactionEvaluatorConfig,
)
