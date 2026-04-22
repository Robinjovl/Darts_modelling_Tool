import abc
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from darts.engines import value_vector
from darts.physics.base.property_base import PropertyBase
from darts.physics.geothermal.physics import Geothermal as GeothermalBase
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.iapws.custom_rock_property import *
from darts.physics.properties.iapws.iapws_property import *


class GeothermalConfig(BaseModel):
    """Configuration for Geothermal physics (IAPWS steam-water).

    Mirrors the OBL + rock-compaction parameters ``Geothermal.__init__``
    currently reads out of ``idata.obl`` and ``idata.rock``.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "kind": "geothermal",
                    "state_spec": "PH",
                    "n_points": 128,
                    "min_p": 1.0,
                    "max_p": 351.0,
                    "min_e": 1000.0,
                    "max_e": 10000.0,
                    "rock_compressibility": 1e-5,
                    "rock_compressibility_ref_p": 1.0,
                    "rock_compressibility_ref_T": 273.15,
                }
            ]
        },
    )

    kind: Literal["geothermal"] = Field(
        default="geothermal",
        description="Physics discriminator for ModelConfig.physics union",
    )
    state_spec: Literal["PH", "PT"] = Field(
        default="PH",
        description="Thermodynamic variable pair (enthalpy or temperature)",
    )
    n_points: int = Field(default=128, ge=2)
    min_p: float = Field(default=1.0, ge=0)
    max_p: float = Field(default=351.0, ge=0)
    min_e: float = Field(default=1000.0)
    max_e: float = Field(default=10000.0)
    rock_compressibility: float = Field(default=1e-5, ge=0)
    rock_compressibility_ref_p: float = Field(default=1.0, ge=0)
    rock_compressibility_ref_T: float = Field(default=273.15, ge=0)


class Geothermal(GeothermalBase):
    """Geothermal physics (P-H state, IAPWS fluid) built from
    :class:`GeothermalConfig` — no InputData dependency.
    """

    def __init__(self, config: "GeothermalConfig", timer):
        """
        :param config: validated GeothermalConfig
        :type config: GeothermalConfig
        :param timer: DARTS timer node
        :type timer: darts.engines.timer_node
        """
        super().__init__(
            timer,
            config.n_points,
            config.min_p,
            config.max_p,
            config.min_e,
            config.max_e,
        )

        property_container = GeothermalIAPWSProperties()
        property_container.Mw = [18.015]
        property_container.rock = [
            value_vector(
                [
                    config.rock_compressibility_ref_p,
                    config.rock_compressibility,
                    config.rock_compressibility_ref_T,
                ]
            )
        ]
        property_container.rock_compaction_ev = custom_rock_compaction_evaluator(
            property_container.rock
        )
        property_container.rock_energy_ev = custom_rock_energy_evaluator(
            property_container.rock
        )

        evaluators = _default_geothermal_iapws_evaluators()
        property_container.temperature_ev = evaluators["temperature_ev"]
        property_container.density_ev = evaluators["density_ev"]
        property_container.viscosity_ev = evaluators["viscosity_ev"]
        property_container.relperm_ev = evaluators["relperm_ev"]
        property_container.enthalpy_ev = evaluators["enthalpy_ev"]
        property_container.saturation_ev = evaluators["saturation_ev"]
        property_container.conduction_ev = evaluators["conduction_ev"]

        self.add_property_region(property_container)

    @classmethod
    def from_config(cls, config: "GeothermalConfig", *, timer) -> "Geothermal":
        """Thin alias for ``Geothermal(config, timer)``."""
        return cls(config, timer)


class GeothermalPH(GeothermalBase):
    """Geothermal physics in P-H formulation with explicit flash (built
    from :class:`GeothermalConfig` — no InputData)."""

    def __init__(self, config: "GeothermalConfig", timer):
        """
        :param config: validated GeothermalConfig
        :type config: GeothermalConfig
        :param timer: DARTS timer node
        :type timer: darts.engines.timer_node
        """
        super().__init__(
            timer,
            config.n_points,
            config.min_p,
            config.max_p,
            config.min_e,
            config.max_e,
        )
        property_container = GeothermalPHProperties()

        evaluators = _default_geothermal_iapws_evaluators()
        property_container.flash_ev = evaluators.get("flash_ev")
        property_container.Mw = [18.015]

        property_container.rock = [
            value_vector(
                [
                    config.rock_compressibility_ref_p,
                    config.rock_compressibility,
                    config.rock_compressibility_ref_T,
                ]
            )
        ]
        property_container.rock_compaction_ev = custom_rock_compaction_evaluator(
            property_container.rock
        )
        property_container.rock_energy_ev = custom_rock_energy_evaluator(
            property_container.rock
        )

        property_container.density_ev = evaluators["density_ev"]
        property_container.viscosity_ev = evaluators["viscosity_ev"]
        property_container.relperm_ev = evaluators["relperm_ev"]
        property_container.enthalpy_ev = evaluators["enthalpy_ev"]
        property_container.conduction_ev = evaluators["conduction_ev"]

        self.add_property_region(property_container)

    @classmethod
    def from_config(cls, config: "GeothermalConfig", *, timer) -> "GeothermalPH":
        """Thin alias for ``GeothermalPH(config, timer)``."""
        return cls(config, timer)


def _default_geothermal_iapws_evaluators() -> dict:
    """Build the default IAPWS evaluator bundle for Geothermal physics.

    Replaces the :class:`GeothermalIAPWSFluidProps` facade — evaluators
    now live alongside the physics class rather than on ``idata.fluid``.
    """
    return {
        "temperature_ev": iapws_temperature_evaluator(),
        "enthalpy_ev": {
            "water": iapws_water_enthalpy_evaluator(),
            "steam": iapws_steam_enthalpy_evaluator(),
            "total": iapws_total_enthalpy_evalutor(),
        },
        "density_ev": {
            "water": iapws_water_density_evaluator(),
            "steam": iapws_steam_density_evaluator(),
        },
        "saturation_ev": {
            "water": iapws_water_saturation_evaluator(),
            "steam": iapws_steam_saturation_evaluator(),
        },
        "viscosity_ev": {
            "water": iapws_water_viscosity_evaluator(),
            "steam": iapws_steam_viscosity_evaluator(),
        },
        "conduction_ev": {"water": ConstFunc(172.8), "steam": ConstFunc(0.0)},
        "relperm_ev": {
            "water": iapws_water_relperm_evaluator(),
            "steam": iapws_steam_relperm_evaluator(),
        },
    }


class GeothermalPropertiesBase(PropertyBase):
    nc = 1
    nph = 2

    def __init__(self):
        self.components_name = ["H2O"]
        self.phases_name = ["water", "steam"]
        self.Mw = np.zeros(self.nc)
        self.nu = np.zeros(self.nph)
        self.x = np.zeros((self.nph, self.nc))
        self.dens = np.zeros(self.nph)
        self.dens_m = np.zeros(self.nph)
        self.saturation = np.zeros(self.nph)
        self.mu = np.zeros(self.nph)
        self.kr = np.zeros(self.nph)
        self.pc = np.zeros(self.nph)
        self.enthalpy = np.zeros(self.nph)
        self.conduction = np.zeros(self.nph)
        self.dX = []
        self.mass_source = np.zeros(self.nc)
        self.energy_source = 0.0
        self.temperature = 0.0

        self.phase_props = [
            self.dens,
            self.dens_m,
            self.saturation,
            self.mu,
            self.kr,
            self.pc,
            self.enthalpy,
            self.conduction,
            self.mass_source,
        ]

        self.output_props = {'temperature': lambda: self.temperature}

    @abc.abstractmethod
    def compute_total_enthalpy(self, state_pt):
        pass


class GeothermalIAPWSProperties(GeothermalPropertiesBase):
    def evaluate(self, state):
        self.temperature = self.temperature_ev.evaluate(state)

        for j, phase in enumerate(['water', 'steam']):
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(state)
            self.dens[j] = self.density_ev[phase].evaluate(state)
            self.dens_m[j] = self.dens[j] / self.Mw[0]
            self.saturation[j] = self.saturation_ev[phase].evaluate(state)
            self.mu[j] = self.viscosity_ev[phase].evaluate(state)
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)
            self.kr[j] = self.relperm_ev[phase].evaluate(state)

        self.ph = np.array([j for j in range(self.nph) if self.saturation[j] > 0])
        return

    def compute_total_enthalpy(self, state_pt):
        return self.enthalpy_ev['total'].evaluate(state_pt, state_pt[-1])


class GeothermalPHProperties(GeothermalPropertiesBase):
    def __init__(self):
        super().__init__()
        self.phases_name = ["water", "steam"]

    def run_flash(self, pressure, enthalpy):
        _ = self.flash_ev.evaluate(pressure, enthalpy)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        self.x = np.array(flash_results.X).reshape(self.nph, self.nc)
        self.temperature = flash_results.T

        ph = np.array([j for j in range(self.nph) if self.nu[j] > 0])

        return ph

    def compute_saturation(self, ph):
        # Get saturations [volume fraction]
        if len(ph) == 1:
            self.saturation[ph] = 1.0
        else:
            vol = [self.nu[j] / self.dens_m[j] for j in ph]
            self.saturation[ph] = vol / np.sum(vol)

        return

    def compute_total_enthalpy(self, state_pt):
        _ = self.flash_ev.evaluate_PT(state_pt[0], state_pt[-1])
        flash_results = self.flash_ev.get_flash_results()
        nu = np.array(flash_results.nu)
        x = np.array(flash_results.X).reshape(self.nph, self.nc)

        ph = np.array([j for j in range(self.nph) if nu[j] > 0])

        enthalpy = 0.0
        for j in ph:
            enthalpy += nu[j] * self.enthalpy_ev[self.phases[j]].evaluate(
                state_pt[0], state_pt[-1], x[j, :]
            )

        return enthalpy

    def evaluate(self, state):
        # Clean arrays
        for a in self.phase_props:
            a[:] = 0

        # Evaluate flash
        self.ph = self.run_flash(state[0], state[1])

        # Evaluate phase properties
        for j in self.ph:
            phase = self.phases[j]
            Mw = np.sum(self.Mw * self.x[j, :])
            self.dens[j] = self.density_ev[phase].evaluate(
                state[0], self.temperature, self.x[j, :]
            )
            self.dens_m[j] = self.dens[j] / Mw
            self.mu[j] = self.viscosity_ev[phase].evaluate(
                state[0], self.temperature, self.x[j, :], self.dens[j]
            )
            self.enthalpy[j] = self.enthalpy_ev[phase].evaluate(
                state[0], self.temperature, self.x[j, :]
            )
            self.conduction[j] = self.conduction_ev[phase].evaluate(state)

        # Compute saturation and saturation-based properties
        self.compute_saturation(self.ph)

        # self.pc = self.capillary_pressure_ev.evaluate(self.sat)
        for j in self.ph:
            self.kr[j] = self.relperm_ev[self.phases[j]].evaluate(self.saturation[j])

        return


def _build_geothermal_ph_flash_evaluators() -> dict:
    """Build the IAPWS P-H formulation flash + evaluator bundle.

    Replaces the legacy :class:`GeothermalPHFluidProps` facade (removed
    alongside :class:`FluidProps`).  Callers that previously did
    ``idata.fluid = GeothermalPHFluidProps()`` now get a dict of
    evaluators directly and assign them onto a
    :class:`GeothermalPHProperties` container themselves.
    """
    from dartsflash.components import CompData
    from dartsflash.libflash import (
        AQEoS,
        CubicEoS,
        EoS,
        FlashParams,
        PHFlash,
    )

    from darts.physics.properties.density import Spivey2004
    from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
    from darts.physics.properties.viscosity import MaoDuan2009

    components = ["H2O"]
    phases = ['water', 'steam']

    comp_data = CompData(components=components, setprops=True)
    Mw = comp_data.Mw
    ceos = CubicEoS(comp_data, CubicEoS.PR)
    ceos.set_preferred_roots(0, 0.75, EoS.MAX)
    aq = AQEoS(comp_data, AQEoS.Jager2003)
    aq.set_eos_range(0, [0.6, 1.0])

    flash_params = FlashParams(comp_data)
    flash_params.add_eos("CEOS", ceos)
    flash_params.add_eos("AQ", aq)
    flash_params.eos_order = ["AQ", "CEOS"]
    flash_params.T_min = 250.0
    flash_params.T_max = 575.0
    flash_params.phflash_Htol = 1e-3
    flash_params.phflash_Ttol = 1e-8

    return {
        "components": components,
        "phases": phases,
        "Mw": Mw,
        "flash_ev": PHFlash(flash_params),
        "enthalpy_ev": {
            "water": EoSEnthalpy(aq),
            "steam": EoSEnthalpy(ceos),
        },
        "density_ev": {
            "water": Spivey2004(components),
            "steam": EoSDensity(ceos, comp_data.Mw),
        },
        "viscosity_ev": {
            "water": MaoDuan2009(components),
            "steam": ConstFunc(0.01),
        },
        "conduction_ev": {"water": ConstFunc(172.8), "steam": ConstFunc(0.0)},
        "relperm_ev": {
            "water": PhaseRelPerm("water"),
            "steam": PhaseRelPerm("gas"),
        },
    }
