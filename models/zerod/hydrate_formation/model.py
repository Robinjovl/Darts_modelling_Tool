from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from darts.models.zerod_model import ZerodModel
from darts.physics.super.initialize import Initialize

from physics import HydrateBatchPhysics


@dataclass(frozen=True)
class CaseConfig:
    slug: str
    original_case: str
    guest_component: str
    guest_label: str
    title: str
    runtime_days: float
    first_ts: float
    max_ts: float
    dt_mult: float
    n_points: int
    min_p: float
    max_p: float
    min_t: float
    max_t: float
    poro: float
    perm_md: float
    dens_rock: float
    c_r: float
    bulk_volume: float
    pressure_init: float
    temperature_init: float
    fixed_pressure: bool
    heat_transfer_coeff: float
    boundary_times: tuple[float, ...]
    boundary_temperatures: tuple[float, ...]

    @property
    def pore_volume(self) -> float:
        return self.bulk_volume * self.poro


CH4_CONFIG = CaseConfig(
    slug="ch4",
    original_case="Moridis_CH4form",
    guest_component="C1",
    guest_label="CH4",
    title="CH4 hydrate formation (Yin & Moridis style 0D reduction)",
    runtime_days=6.0 / 24.0,
    first_ts=1e-7,
    max_ts=2e-4,
    dt_mult=1.5,
    n_points=101,
    min_p=10.0,
    max_p=300.0,
    min_t=250.0,
    max_t=333.15,
    poro=0.448,
    perm_md=3830.0,
    dens_rock=2650.0,
    c_r=1.309,
    bulk_volume=np.pi * 0.051**2 * 0.120,
    pressure_init=95.0,
    temperature_init=288.2,
    fixed_pressure=False,
    heat_transfer_coeff=9.5e4,
    boundary_times=(0.0, 0.66 / 24.0, 6.0 / 24.0),
    boundary_temperatures=(288.2, 274.5, 274.5),
)


CO2_INTERIOR_RADIUS = 0.06 - (0.06 - 1.0e-4) / 11.0
CO2_INTERIOR_HEIGHT = 0.16 - 2.0 * (0.16 / 32.0)
CO2_CONFIG = CaseConfig(
    slug="co2",
    original_case="Li_CO2inj",
    guest_component="CO2",
    guest_label="CO2",
    title="CO2 hydrate formation (Li et al. style 0D reduction)",
    runtime_days=120.0 / 1440.0,
    first_ts=1e-8,
    max_ts=1e-4,
    dt_mult=1.5,
    n_points=101,
    min_p=10.0,
    max_p=300.0,
    min_t=250.0,
    max_t=303.15,
    poro=0.312,
    perm_md=30400.0,
    dens_rock=2075.5,
    c_r=0.745,
    bulk_volume=np.pi * CO2_INTERIOR_RADIUS**2 * CO2_INTERIOR_HEIGHT,
    pressure_init=32.0,
    temperature_init=273.15 + 7.25,
    fixed_pressure=True,
    heat_transfer_coeff=2.0e6,
    boundary_times=(0.0, 120.0 / 1440.0),
    boundary_temperatures=(274.15, 274.15),
)


CASE_CONFIGS = {
    CH4_CONFIG.slug: CH4_CONFIG,
    CO2_CONFIG.slug: CO2_CONFIG,
}


class HydrateFormationModel(ZerodModel):
    def __init__(
        self,
        case: str,
        n_points: int | None = None,
        runtime: float | None = None,
        max_ts: float | None = None,
        heat_transfer_coeff: float | None = None,
        fixed_pressure: bool | None = None,
    ):
        if case not in CASE_CONFIGS:
            raise ValueError(f"Unknown hydrate case '{case}'. Expected one of {list(CASE_CONFIGS)}.")

        self.case_config = CASE_CONFIGS[case]
        super().__init__(
            fixed_pressure=(
                self.case_config.fixed_pressure
                if fixed_pressure is None
                else bool(fixed_pressure)
            ),
            fixed_temperature=False,
        )

        self.mode = "obl"
        self.runtime_days = (
            self.case_config.runtime_days if runtime is None else float(runtime)
        )
        self.n_points = self.case_config.n_points if n_points is None else int(n_points)
        self.heat_transfer_coeff = (
            self.case_config.heat_transfer_coeff
            if heat_transfer_coeff is None
            else float(heat_transfer_coeff)
        )

        self.poro = self.case_config.poro
        self.dens_rock = self.case_config.dens_rock
        self.c_r = self.case_config.c_r
        self.bulk_volume = self.case_config.bulk_volume
        self.pore_volume = self.case_config.pore_volume

        self.timer.node["initialization"].start()
        self.set_physics()
        self.set_sim_params(
            n_vars=self.physics.n_vars,
            first_ts=self.case_config.first_ts,
            mult_ts=self.case_config.dt_mult,
            max_ts=self.case_config.max_ts if max_ts is None else float(max_ts),
            runtime=self.runtime_days,
        )
        self.data_ts.eta = np.array([5.0, 2e-3, 2e-3, 0.2], dtype=float)
        self.timer.node["initialization"].stop()

    def set_physics(self):
        cfg = self.case_config
        self.physics = HydrateBatchPhysics(
            timer=self.timer,
            guest_component=cfg.guest_component,
            case_name=cfg.slug,
            n_points=self.n_points,
            min_p=cfg.min_p,
            max_p=cfg.max_p,
            min_t=cfg.min_t,
            max_t=cfg.max_t,
            cache=False,
            extrapolation_flag=True,
        )

        if cfg.slug == "ch4":
            kinetic_rate = 3.6e4 * 1e5 * 86400.0 * 280.0
        else:
            kinetic_rate = 8.4e11 * 1e5 * 86400.0 * 120.0
            kinetic_rate *= 1000.0 / self.physics.property_container.Mw[-1]

        self.physics.set_kinetic_ev(
            k=kinetic_rate,
            perm=cfg.perm_md,
            poro=cfg.poro,
        )
        self.property_container = self.physics.property_container

        self.water_index = self.property_container.components_name.index("H2O")
        self.guest_index = self.property_container.components_name.index(
            cfg.guest_component
        )
        self.hydrate_component_index = self.property_container.components_name.index("H")

        self.aq_phase_index = self.property_container.phases_name.index("Aq")
        self.vap_phase_index = self.property_container.phases_name.index("V")
        self.hyd_phase_index = self.property_container.phases_name.index("sI")

        self.water_mw = float(self.property_container.Mw[self.water_index])
        self.guest_mw = float(self.property_container.Mw[self.guest_index])

    def set_initial_conditions(self):
        cfg = self.case_config
        init = Initialize(self.physics)

        if cfg.slug == "ch4":
            # Publication setup: 183.3 g H2O in the porous volume, negligible initial hydrate.
            water_mass_density = 0.1833 / self.pore_volume
            specs = {
                "pressure": cfg.pressure_init,
                "m_H2O": water_mass_density,
                "H": 1e-5,
                "temperature": cfg.temperature_init,
            }
            guess = [cfg.pressure_init, 0.9, 0.1, cfg.temperature_init]
        else:
            specs = {
                "pressure": cfg.pressure_init,
                "temperature": cfg.temperature_init,
                "satAq": 0.25 - 5e-4,
                "satV": 0.75 - 5e-4,
            }
            guess = [cfg.pressure_init, 0.9, 0.099, cfg.temperature_init]

        self.initial_state = np.asarray(init.solve_state(guess, specs), dtype=float)

    def boundary_temperature(self, t: float) -> float:
        cfg = self.case_config
        return float(
            np.interp(
                t,
                np.asarray(cfg.boundary_times, dtype=float),
                np.asarray(cfg.boundary_temperatures, dtype=float),
            )
        )

    def external_energy_rate(self, t: float, state) -> float:
        state_np = np.asarray(state, dtype=float)
        self.property_container.evaluate(state_np)
        temperature = float(self.property_container.temperature)
        return self.heat_transfer_coeff * (self.boundary_temperature(t) - temperature)

    def get_obl_source_terms(self, t, state, ops, etor):
        source = np.zeros(self.physics.n_vars, dtype=float)
        kin_start = getattr(etor, "KIN_OP", None)
        if kin_start is not None:
            ncopy = min(source.size, max(0, np.asarray(ops).size - kin_start))
            if ncopy > 0:
                source[:ncopy] = -np.asarray(ops[kin_start : kin_start + ncopy], dtype=float)

        if self.physics.n_vars > self.physics.nc:
            source[self.physics.nc] += self.external_energy_rate(t, state)
        return source

    @staticmethod
    def _safe_float(value, fallback=0.0) -> float:
        out = float(np.nan_to_num(value, nan=fallback, posinf=fallback, neginf=fallback))
        return out

    def evaluate_output_properties(self, state):
        state_np = np.asarray(state, dtype=float)
        pc = self.property_container
        pc.evaluate(state_np)

        sat_aq = self._safe_float(pc.sat[self.aq_phase_index])
        sat_vap = self._safe_float(pc.sat[self.vap_phase_index])
        sat_hyd = self._safe_float(pc.sat[self.hyd_phase_index])

        x_guest_aq = self._safe_float(pc.x[self.aq_phase_index, self.guest_index])
        y_h2o_vap = self._safe_float(pc.x[self.vap_phase_index, self.water_index])
        rho_hyd = self._safe_float(pc.dens[self.hyd_phase_index])
        rhom_aq = self._safe_float(pc.dens_m[self.aq_phase_index])
        rhom_vap = self._safe_float(pc.dens_m[self.vap_phase_index])

        mass_scale = 1000.0
        mass_water = mass_scale * self.pore_volume * self.water_mw * (
            sat_aq * rhom_aq * (1.0 - x_guest_aq) + sat_vap * rhom_vap * y_h2o_vap
        )
        mass_guest = mass_scale * self.pore_volume * self.guest_mw * (
            sat_aq * rhom_aq * x_guest_aq + sat_vap * rhom_vap * (1.0 - y_h2o_vap)
        )
        mass_hyd = mass_scale * self.pore_volume * sat_hyd * rho_hyd

        hydrate_fraction = max(0.0, 1.0 - state_np[1] - state_np[2])
        return {
            "pressure_bar": float(state_np[0]),
            "pressure_MPa": float(state_np[0] * 0.1),
            "temperature_K": self._safe_float(pc.temperature),
            "temperature_C": self._safe_float(pc.temperature - 273.15),
            "sat_aq": sat_aq,
            "sat_vap": sat_vap,
            "sat_hyd": sat_hyd,
            f"x_{self.case_config.guest_label}_aq": x_guest_aq,
            "y_H2O_vap": y_h2o_vap,
            "z_hydrate": hydrate_fraction,
            "mass_H2O": mass_water,
            f"mass_{self.case_config.guest_label}": mass_guest,
            "mass_hydrate": mass_hyd,
            "mass_total": mass_water + mass_guest + mass_hyd,
            "rate_hydrate_component": self._safe_float(
                pc.mass_source[self.hydrate_component_index]
            ),
        }
