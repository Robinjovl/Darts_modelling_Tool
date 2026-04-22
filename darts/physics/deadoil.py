from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic, DensityBrineCO2
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import (
    PropertyContainer,
    PropertyContainerConfig,
)


class DeadOilConfig(BaseModel):
    """Configuration for DeadOil physics (dead-oil PVT, 2ph or 3ph).

    Mirrors the OBL / thermal-toggle fields ``DeadOil.__init__`` reads from
    ``idata.obl`` and ``idata.fluid`` so the ModelConfig-driven path can
    construct a ``DeadOil`` without going through an ``InputData`` shim.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "kind": "dead_oil",
                    "thermal": False,
                    "n_phases": 2,
                    "n_points": 1001,
                    "min_p": 1.0,
                    "max_p": 450.0,
                    "min_z": 1e-13,
                    "max_z": 1.0,
                    "epsilon_z": 1e-13,
                }
            ]
        },
    )

    kind: Literal["dead_oil"] = Field(
        default="dead_oil",
        description="Physics discriminator for ModelConfig.physics union",
    )
    thermal: bool = Field(
        default=False, description="Thermal (PT) vs. isothermal (P) mode"
    )
    n_phases: Literal[2, 3] = Field(
        default=2, description="2 for oil-water, 3 for gas-oil-water"
    )
    n_points: int = Field(default=1001, ge=2)
    min_p: float = Field(default=1.0, ge=0)
    max_p: float = Field(default=450.0, ge=0)
    min_z: float = Field(default=1e-13, ge=0)
    max_z: float = Field(default=1.0, ge=0)
    epsilon_z: float = Field(default=1e-13, ge=0)
    min_t: float | None = Field(default=None)
    max_t: float | None = Field(default=None)
    components: list[str] | None = None
    phases: list[str] | None = None


class DeadOil(Compositional):
    """Dead-oil physics constructed directly from a :class:`DeadOilConfig`.

    Replaces the legacy ``DeadOil(idata, timer, thermal)`` signature.  The
    default 2-phase or 3-phase fluid bundle is built by helper functions
    below, matching the old ``DeadOil{2,3}PFluidProps`` layouts.
    """

    def __init__(self, config: DeadOilConfig, timer: Any):
        """
        :param config: validated DeadOil configuration
        :type config: DeadOilConfig
        :param timer: DARTS timer node
        :type timer: darts.engines.timer_node
        """
        evaluators = _default_dead_oil_evaluators(config.n_phases)
        components = config.components or evaluators["components"]
        phases = config.phases or evaluators["phases"]

        thermal = config.thermal
        state_spec = (
            Compositional.StateSpecification.PT
            if thermal
            else Compositional.StateSpecification.P
        )
        super().__init__(
            components=components,
            phases=phases,
            timer=timer,
            n_points=config.n_points,
            min_p=config.min_p,
            max_p=config.max_p,
            min_z=config.min_z,
            max_z=config.max_z,
            epsilon_z=config.epsilon_z,
            min_t=config.min_t,
            max_t=config.max_t,
            state_spec=state_spec,
            extrapolation_flag=True,
        )
        self.zero = 1e-13

        temperature = None if thermal else 1.0
        property_container = DeadOilProperties(
            phases_name=phases,
            components_name=components,
            Mw=evaluators["Mw"],
            eps_z=config.epsilon_z,
            temperature=temperature,
        )
        property_container.density_ev = evaluators["density"]
        property_container.viscosity_ev = evaluators["viscosity"]
        property_container.rel_perm_ev = evaluators["rel_perm"]
        self.add_property_region(property_container)

    @classmethod
    def from_config(cls, config: DeadOilConfig, *, timer: Any) -> DeadOil:
        """Thin alias for ``DeadOil(config, timer)``."""
        return cls(config, timer)


def _default_dead_oil_evaluators(n_phases: int) -> dict:
    """Build the default dead-oil fluid evaluator bundle for 2 or 3 phases.

    Replaces the legacy ``DeadOil2PFluidProps`` / ``DeadOil3PFluidProps``
    facades — the evaluator dicts now live alongside physics construction
    rather than being attached to ``idata.fluid``.
    """
    if n_phases == 2:
        components = ["Zo", "Zw"]
        phases = ["oil", "water"]
        density = {
            "water": DensityBasic(compr=1e-5, dens0=1014),
            "oil": DensityBasic(compr=5e-3, dens0=700),
        }
        viscosity = {"water": ConstFunc(0.89), "oil": ConstFunc(1)}
        rel_perm = {
            "water": PhaseRelPerm("water", 0.1, 0.1),
            "oil": PhaseRelPerm("oil", 0.1, 0.1),
        }
    elif n_phases == 3:
        components = ["g", "o", "w"]
        phases = ["gas", "oil", "water"]
        density = {
            "gas": DensityBasic(compr=1e-3, dens0=200),
            "oil": DensityBasic(compr=1e-5, dens0=600),
            "water": DensityBrineCO2(components, compr=1e-5, dens0=1000, co2_mult=0),
        }
        viscosity = {
            "gas": ConstFunc(0.05),
            "oil": ConstFunc(0.5),
            "water": ConstFunc(0.5),
        }
        rel_perm = {
            "gas": PhaseRelPerm("gas"),
            "oil": PhaseRelPerm("oil"),
            "water": PhaseRelPerm("water"),
        }
    else:
        raise ValueError(f"Unsupported DeadOil n_phases: {n_phases}")
    return {
        "components": components,
        "phases": phases,
        "Mw": np.ones(len(components)),
        "density": density,
        "viscosity": viscosity,
        "rel_perm": rel_perm,
    }


class DeadOilProperties(PropertyContainer):
    def __init__(
        self,
        phases_name,
        components_name,
        Mw,
        eps_z=1e-11,
        rock_comp=1e-6,
        temperature: float = None,
    ):
        # Call base class constructor
        super().__init__(
            phases_name=phases_name,
            components_name=components_name,
            Mw=Mw,
            eps_z=eps_z,
            rock_comp=rock_comp,
            temperature=temperature,
        )

    @classmethod
    def from_config(cls, config: PropertyContainerConfig) -> DeadOilProperties:
        """Construct from a validated PropertyContainerConfig."""
        eps_z = config.eps_z if config.eps_z is not None else config.min_z
        components_name = config.components_name
        phases_name = config.phases_name
        if components_name is None or phases_name is None:
            raise ValueError("components_name and phases_name are required")
        Mw = config.Mw if config.Mw is not None else [1.0] * len(components_name)
        kwargs: dict = dict(
            phases_name=phases_name,
            components_name=components_name,
            Mw=Mw,
            eps_z=eps_z,
            temperature=config.temperature,
        )
        if config.rock_comp is not None:
            kwargs["rock_comp"] = config.rock_comp
        return cls(**kwargs)

    def run_flash(self, pressure, temperature, zc, evaluate_PT: bool = True):
        ph = np.array([j for j in range(self.nph)])

        for i in range(self.nc):
            self.x[i][i] = 1
        self.nu = zc

        return ph
