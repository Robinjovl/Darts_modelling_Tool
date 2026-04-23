import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from dartsflash.libflash import EoS
from scipy.optimize import brentq

from darts.engines import value_vector
from darts.pipes.upstream_mass_node import UpstreamMassNode


@dataclass(frozen=True)
class ChokeBoundaryState:
    pressure: float
    temperature: float
    composition: np.ndarray
    phase_name: str
    molar_enthalpy: float
    molar_entropy: float


@dataclass(frozen=True)
class ChokeEvaluationResult:
    mass_rate_kg_s: float
    discharge_molar_enthalpy: float


class ChokePhysicsHelper:
    _MIN_T_K = 150.0
    _MAX_T_K = 500.0
    _H_BOUNDS = (-23714.0, 8230.0)

    def __init__(self, physics):
        self.physics = physics
        self.pc = physics.property_containers[0]
        self.eos = self.pc.flash_ev.eos["VL"]

    @staticmethod
    def phase_root_flag(phase_name: str):
        if phase_name == "G":
            return EoS.RootFlag.MAX
        if phase_name == "L":
            return EoS.RootFlag.MIN
        raise NotImplementedError(
            "UpstreamPressureNodeWithChoke currently supports phase_name 'G' or 'L'."
        )

    def evaluate_phase_density(
        self,
        phase_name: str,
        pressure: float,
        temperature: float,
        composition,
    ) -> float:
        if phase_name not in ("G", "L"):
            raise NotImplementedError(
                "UpstreamPressureNodeWithChoke currently supports phase_name 'G' or 'L'."
            )
        return float(
            self.pc.density_ev[phase_name].evaluate(
                pressure,
                temperature,
                composition,
            )
        )

    def evaluate_phase_entropy(
        self,
        pressure: float,
        temperature: float,
        composition,
        root_flag,
    ) -> float:
        self.eos.set_root_flag(root_flag)
        return float(eos_entropy(self.eos, pressure, temperature, composition))

    def build_ph_state(self, pressure: float, molar_enthalpy: float, composition):
        if self.physics.n_vars == 2:
            return value_vector([pressure, molar_enthalpy])
        z = np.asarray(composition, dtype=float)
        return value_vector([pressure, molar_enthalpy, *z[:-1]])

    def evaluate_equilibrium_mixture_entropy(
        self,
        pressure: float,
        molar_enthalpy: float,
        composition,
    ) -> float:
        state = self.build_ph_state(pressure, molar_enthalpy, composition)
        self.pc.evaluate(state)
        self.pc.evaluate_thermal(state)

        phase_weights = []
        phase_entropies = []
        for phase_name in ("G", "L"):
            if phase_name not in self.physics.phases:
                continue

            phase_idx = self.physics.phases.index(phase_name)
            sat = float(self.pc.sat[phase_idx])
            dens = float(self.pc.dens[phase_idx])
            if (
                not np.isfinite(sat)
                or not np.isfinite(dens)
                or sat <= 0.0
                or dens <= 0.0
            ):
                continue

            phase_comp = np.asarray(self.pc.x[phase_idx], dtype=float)
            phase_mw = float(np.sum(phase_comp * np.asarray(self.pc.Mw))) * 1e-3
            if phase_mw <= 0.0:
                continue

            phase_weights.append(sat * dens / phase_mw)
            phase_entropies.append(
                self.evaluate_phase_entropy(
                    pressure,
                    float(self.pc.temperature),
                    phase_comp,
                    self.phase_root_flag(phase_name),
                )
            )

        if not phase_weights:
            return np.nan

        weights = np.asarray(phase_weights, dtype=float)
        weights /= np.sum(weights)
        return float(np.dot(weights, np.asarray(phase_entropies, dtype=float)))

    @staticmethod
    def find_bracket(xs: np.ndarray, fn):
        vals = np.asarray([fn(x) for x in xs], dtype=float)
        for idx in range(len(xs) - 1):
            if not np.isfinite(vals[idx]) or not np.isfinite(vals[idx + 1]):
                continue
            if vals[idx] == 0.0:
                return float(xs[idx]), float(xs[idx])
            if vals[idx] * vals[idx + 1] < 0.0:
                return float(xs[idx]), float(xs[idx + 1])
        return None

    def clamp_enthalpy(self, molar_enthalpy: float) -> float:
        return float(np.clip(molar_enthalpy, *self._H_BOUNDS))

    def solve_single_phase_isentropic_enthalpy(
        self,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        phase_name = boundary_state.phase_name
        root_flag = self.phase_root_flag(phase_name)

        def entropy_residual(temperature: float) -> float:
            return (
                self.evaluate_phase_entropy(
                    downstream_pressure,
                    temperature,
                    boundary_state.composition,
                    root_flag,
                )
                - boundary_state.molar_entropy
            )

        bracket = self.find_bracket(
            np.linspace(self._MIN_T_K, self._MAX_T_K, 32),
            entropy_residual,
        )
        if bracket is None:
            return self.clamp_enthalpy(boundary_state.molar_enthalpy)

        if bracket[0] == bracket[1]:
            temperature = bracket[0]
        else:
            temperature = brentq(entropy_residual, bracket[0], bracket[1])

        molar_enthalpy = self.pc.enthalpy_ev[phase_name].evaluate(
            downstream_pressure,
            temperature,
            boundary_state.composition,
        )
        if not np.isfinite(molar_enthalpy):
            return self.clamp_enthalpy(boundary_state.molar_enthalpy)
        return self.clamp_enthalpy(molar_enthalpy)

    def solve_equilibrium_isentropic_enthalpy(
        self,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        h_frozen = self.solve_single_phase_isentropic_enthalpy(
            boundary_state,
            downstream_pressure,
        )

        def entropy_residual(molar_enthalpy: float) -> float:
            return (
                self.evaluate_equilibrium_mixture_entropy(
                    downstream_pressure,
                    molar_enthalpy,
                    boundary_state.composition,
                )
                - boundary_state.molar_entropy
            )

        bracket = self.find_bracket(
            np.linspace(self._H_BOUNDS[0], self._H_BOUNDS[1], 48),
            entropy_residual,
        )
        if bracket is None:
            return h_frozen
        if bracket[0] == bracket[1]:
            return self.clamp_enthalpy(bracket[0])
        return self.clamp_enthalpy(brentq(entropy_residual, bracket[0], bracket[1]))


def eos_entropy(eos, pressure: float, temperature: float, composition) -> float:
    return eos.S(pressure, temperature, np.asarray(composition, dtype=float))


class ValveGeometryModel(ABC):
    @property
    @abstractmethod
    def valve_geometry(self) -> str:
        pass

    @property
    @abstractmethod
    def diameter(self) -> float:
        pass

    @diameter.setter
    @abstractmethod
    def diameter(self, value: float):
        pass

    @property
    @abstractmethod
    def choke_area(self) -> float:
        pass

    @property
    @abstractmethod
    def effective_area(self) -> float:
        pass

    @abstractmethod
    def mass_rate_kg_s(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        pass

    @abstractmethod
    def size_diameter_from_target_rate(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        target_mass_rate_kg_s: float,
        initial_downstream_pressure: float,
    ) -> float:
        pass

    @abstractmethod
    def rescale_diameter_for_target_mass_rate(
        self,
        actual_mass_rate_kg_s: float,
        target_mass_rate_kg_s: float,
    ) -> float:
        pass


class OrificeValveGeometryModel(ValveGeometryModel):
    _PA_PER_BAR = 1e5

    def __init__(
        self,
        diameter: float = None,
        discharge_coefficient: float = 1.0,
        opening: float = 1.0,
        flow_coefficient: float = 1.0,
    ):
        if discharge_coefficient <= 0.0:
            raise ValueError("discharge_coefficient must be positive.")
        if flow_coefficient <= 0.0:
            raise ValueError("flow_coefficient must be positive.")
        if opening <= 0.0:
            raise ValueError("opening must be positive.")
        if diameter is not None and diameter <= 0.0:
            raise ValueError("diameter must be positive.")

        self.discharge_coefficient = float(discharge_coefficient)
        self.opening = float(opening)
        self.flow_coefficient = float(flow_coefficient)
        self._diameter = None if diameter is None else float(diameter)

    @property
    def valve_geometry(self) -> str:
        return "ORIFICE"

    @property
    def diameter(self) -> float:
        return self._diameter

    @diameter.setter
    def diameter(self, value: float):
        if value <= 0.0:
            raise ValueError("diameter must be positive.")
        self._diameter = float(value)

    @property
    def choke_area(self) -> float:
        return math.pi * self._diameter**2 / 4.0

    @property
    def effective_area(self) -> float:
        return (
            self.flow_coefficient
            * self.discharge_coefficient
            * self.opening
            * self.choke_area
        )

    def mass_rate_kg_s(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        delta_p_bar = max(boundary_state.pressure - downstream_pressure, 0.0)
        if delta_p_bar == 0.0:
            return 0.0

        rho_up = helper.evaluate_phase_density(
            boundary_state.phase_name,
            boundary_state.pressure,
            boundary_state.temperature,
            boundary_state.composition,
        )
        return self.effective_area * math.sqrt(
            2.0 * rho_up * delta_p_bar * self._PA_PER_BAR
        )

    def size_diameter_from_target_rate(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        target_mass_rate_kg_s: float,
        initial_downstream_pressure: float,
    ) -> float:
        delta_p_bar = boundary_state.pressure - initial_downstream_pressure
        if delta_p_bar <= 0.0:
            raise ValueError(
                "upstream pressure must be greater than the initial downstream pressure to size the choke."
            )

        rho_up = helper.evaluate_phase_density(
            boundary_state.phase_name,
            boundary_state.pressure,
            boundary_state.temperature,
            boundary_state.composition,
        )
        area = target_mass_rate_kg_s / (
            self.flow_coefficient
            * self.discharge_coefficient
            * self.opening
            * math.sqrt(2.0 * rho_up * delta_p_bar * self._PA_PER_BAR)
        )
        if area <= 0.0:
            raise ValueError("computed choke area must be positive.")
        return math.sqrt(4.0 * area / math.pi)

    def rescale_diameter_for_target_mass_rate(
        self,
        actual_mass_rate_kg_s: float,
        target_mass_rate_kg_s: float,
    ) -> float:
        if self._diameter is None:
            raise ValueError(
                "diameter must be defined before it can be rescaled from a rate mismatch."
            )
        if actual_mass_rate_kg_s <= 0.0:
            raise ValueError("actual_mass_rate_kg_s must be positive.")
        if target_mass_rate_kg_s <= 0.0:
            raise ValueError("target_mass_rate_kg_s must be positive.")
        return self._diameter * math.sqrt(target_mass_rate_kg_s / actual_mass_rate_kg_s)


class RecoveryModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate(self, valve_geometry: str):
        return


class NoRecoveryModel(RecoveryModel):
    @property
    def name(self) -> str:
        return "OFF"


class UnsupportedRecoveryModel(RecoveryModel):
    def __init__(self, recovery: str):
        self._name = recovery.upper()

    @property
    def name(self) -> str:
        return self._name

    def validate(self, valve_geometry: str):
        raise NotImplementedError(
            f"RECOVERY={self._name!r} is not implemented yet for valve_geometry={valve_geometry!r}."
        )


class SlipModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate(self, equilibrium_model_name: str):
        return


class NoSlipModel(SlipModel):
    @property
    def name(self) -> str:
        return "NOSLIP"


class ChisholmSlipModel(SlipModel):
    @property
    def name(self) -> str:
        return "CHISHOLM"

    def validate(self, equilibrium_model_name: str):
        raise NotImplementedError(
            "SLIPMODEL='CHISHOLM' is not implemented yet in the DARTS choke boundary."
        )


class EquilibriumModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate(self, slip_model_name: str):
        return

    @abstractmethod
    def discharge_molar_enthalpy(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        pass


class FrozenEquilibriumModel(EquilibriumModel):
    def __init__(self, thermal_phase_equilibrium: bool = False):
        self.thermal_phase_equilibrium = bool(thermal_phase_equilibrium)

    @property
    def name(self) -> str:
        return "FROZEN"

    def discharge_molar_enthalpy(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        # OLGA default FROZEN behavior keeps liquid properties tied to the upstream
        # state. Gas is expanded isentropically. If thermal phase equilibrium is
        # enabled, use the phase-frozen isentropic path for either phase.
        if boundary_state.phase_name == "L" and not self.thermal_phase_equilibrium:
            return helper.clamp_enthalpy(boundary_state.molar_enthalpy)
        return helper.solve_single_phase_isentropic_enthalpy(
            boundary_state,
            downstream_pressure,
        )


class HenryFauskeEquilibriumModel(EquilibriumModel):
    def __init__(self, thermal_phase_equilibrium: bool = False):
        self.thermal_phase_equilibrium = bool(thermal_phase_equilibrium)
        self._frozen = FrozenEquilibriumModel(
            thermal_phase_equilibrium=thermal_phase_equilibrium
        )

    @property
    def name(self) -> str:
        return "HENRYFAUSKE"

    def discharge_molar_enthalpy(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        # TODO: replace this blend with the full Henry-Fauske throat gas-fraction
        # correction from the OLGA / Henry-Fauske equations.
        h_frozen = self._frozen.discharge_molar_enthalpy(
            helper,
            boundary_state,
            downstream_pressure,
        )
        h_eq = helper.solve_equilibrium_isentropic_enthalpy(
            boundary_state,
            downstream_pressure,
        )
        return helper.clamp_enthalpy(0.5 * (h_frozen + h_eq))


class FullEquilibriumModel(EquilibriumModel):
    @property
    def name(self) -> str:
        return "EQUILIBRIUM"

    def validate(self, slip_model_name: str):
        if slip_model_name != "NOSLIP":
            raise ValueError(
                "EQUILIBRIUMMODEL='EQUILIBRIUM' cannot be combined with a slip model."
            )

    def discharge_molar_enthalpy(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        return helper.solve_equilibrium_isentropic_enthalpy(
            boundary_state,
            downstream_pressure,
        )


def build_valve_geometry_model(
    valve_geometry: str,
    diameter: float = None,
    discharge_coefficient: float = 1.0,
    opening: float = 1.0,
    flow_coefficient: float = 1.0,
) -> ValveGeometryModel:
    valve_geometry = valve_geometry.upper()
    if valve_geometry == "ORIFICE":
        return OrificeValveGeometryModel(
            diameter=diameter,
            discharge_coefficient=discharge_coefficient,
            opening=opening,
            flow_coefficient=flow_coefficient,
        )
    raise NotImplementedError(
        f"VALVEGEOMETRY={valve_geometry!r} is not implemented yet."
    )


def build_equilibrium_model(
    equilibrium_model: str,
    thermal_phase_equilibrium: bool = False,
) -> EquilibriumModel:
    equilibrium_model = equilibrium_model.upper()
    if equilibrium_model == "FROZEN":
        return FrozenEquilibriumModel(
            thermal_phase_equilibrium=thermal_phase_equilibrium
        )
    if equilibrium_model == "HENRYFAUSKE":
        return HenryFauskeEquilibriumModel(
            thermal_phase_equilibrium=thermal_phase_equilibrium
        )
    if equilibrium_model == "EQUILIBRIUM":
        return FullEquilibriumModel()
    raise NotImplementedError(
        f"EQUILIBRIUMMODEL={equilibrium_model!r} is not implemented yet."
    )


def build_recovery_model(recovery: str) -> RecoveryModel:
    recovery = recovery.upper()
    if recovery == "OFF":
        return NoRecoveryModel()
    return UnsupportedRecoveryModel(recovery)


def build_slip_model(slip_model: str) -> SlipModel:
    slip_model = slip_model.upper()
    if slip_model == "NOSLIP":
        return NoSlipModel()
    if slip_model == "CHISHOLM":
        return ChisholmSlipModel()
    raise NotImplementedError(f"SLIPMODEL={slip_model!r} is not implemented yet.")


class ChokeModel:
    def __init__(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        valve_geometry_model: ValveGeometryModel,
        equilibrium_model: EquilibriumModel,
        recovery_model: RecoveryModel,
        slip_model: SlipModel,
    ):
        self.helper = helper
        self.boundary_state = boundary_state
        self.valve_geometry_model = valve_geometry_model
        self.equilibrium_model = equilibrium_model
        self.recovery_model = recovery_model
        self.slip_model = slip_model

        self.recovery_model.validate(self.valve_geometry_model.valve_geometry)
        self.slip_model.validate(self.equilibrium_model.name)
        self.equilibrium_model.validate(self.slip_model.name)

    def size_diameter_from_target_rate(
        self,
        target_mass_rate_kg_s: float,
        initial_downstream_pressure: float,
    ) -> float:
        return self.valve_geometry_model.size_diameter_from_target_rate(
            self.helper,
            self.boundary_state,
            target_mass_rate_kg_s,
            initial_downstream_pressure,
        )

    def evaluate(self, downstream_pressure: float) -> ChokeEvaluationResult:
        mass_rate_kg_s = self.valve_geometry_model.mass_rate_kg_s(
            self.helper,
            self.boundary_state,
            downstream_pressure,
        )
        discharge_molar_enthalpy = self.equilibrium_model.discharge_molar_enthalpy(
            self.helper,
            self.boundary_state,
            downstream_pressure,
        )
        return ChokeEvaluationResult(
            mass_rate_kg_s=mass_rate_kg_s,
            discharge_molar_enthalpy=discharge_molar_enthalpy,
        )


class UpstreamPressureNodeWithChoke(UpstreamMassNode):
    """
    Upstream pressure/temperature source connected to the pipe through a choke.

    The implementation is intentionally split into orthogonal model parts, following
    the OLGA choke documentation:

      - valve geometry
      - equilibrium model
      - recovery model
      - slip model

    Only the ORIFICE geometry and NOSLIP / OFF recovery path are implemented
    presently, but new models can be added without rewriting the boundary node.
    """

    _SEC_PER_DAY = 24.0 * 60.0 * 60.0

    def __init__(
        self,
        pipe_name: str,
        pipe_geom,
        reservoir,
        physics,
        first_ts_size: float,
        segment_idx: int,
        target_molar_rate: float,
        ramp_up_period: float,
        composition,
        pressure: float,
        temperature: float,
        phase_name: str,
        valve_geometry: str = "ORIFICE",
        equilibrium_model: str = "FROZEN",
        diameter: float = None,
        discharge_coefficient: float = 1.0,
        opening: float = 1.0,
        flow_coefficient: float = 1.0,
        thermal_phase_equilibrium: bool = False,
        recovery: str = "OFF",
        slip_model: str = "NOSLIP",
        initial_downstream_pressure: float = None,
        max_molar_rate: float = None,
        verbose: bool = False,
    ):
        if max_molar_rate is not None and max_molar_rate <= 0.0:
            raise ValueError("max_molar_rate must be positive when specified.")

        super().__init__(
            pipe_name=pipe_name,
            pipe_geom=pipe_geom,
            physics=physics,
            first_ts_size=first_ts_size,
            segment_idx=segment_idx,
            target_molar_rate=target_molar_rate,
            ramp_up_period=ramp_up_period,
            composition=composition,
            pressure=pressure,
            temperature=temperature,
            phase_name=phase_name,
            verbose=False,
        )

        self.downstream_block_idx = reservoir.mesh.n_res_blocks + segment_idx
        self.max_molar_rate = max_molar_rate
        self.last_downstream_pressure = None
        self.last_mass_rate_kg_s = None

        helper = ChokePhysicsHelper(self.physics)
        boundary_state = ChokeBoundaryState(
            pressure=self.pressure,
            temperature=self.temperature,
            composition=np.asarray(self.composition, dtype=float),
            phase_name=self.phase_name,
            molar_enthalpy=self.inj_fluid_props["molar_enthalpy"],
            molar_entropy=helper.evaluate_phase_entropy(
                self.pressure,
                self.temperature,
                self.composition,
                helper.phase_root_flag(self.phase_name),
            ),
        )

        geometry_model = build_valve_geometry_model(
            valve_geometry=valve_geometry,
            diameter=diameter,
            discharge_coefficient=discharge_coefficient,
            opening=opening,
            flow_coefficient=flow_coefficient,
        )
        equilibrium_model_obj = build_equilibrium_model(
            equilibrium_model=equilibrium_model,
            thermal_phase_equilibrium=thermal_phase_equilibrium,
        )
        recovery_model = build_recovery_model(recovery)
        slip_model_obj = build_slip_model(slip_model)

        self.choke_model = ChokeModel(
            helper=helper,
            boundary_state=boundary_state,
            valve_geometry_model=geometry_model,
            equilibrium_model=equilibrium_model_obj,
            recovery_model=recovery_model,
            slip_model=slip_model_obj,
        )

        if geometry_model.diameter is None:
            if initial_downstream_pressure is None:
                raise ValueError(
                    "initial_downstream_pressure must be provided when choke diameter is not specified."
                )
            target_mass_rate = self._target_mass_rate_kg_s(target_molar_rate)
            geometry_model.diameter = self.choke_model.size_diameter_from_target_rate(
                target_mass_rate,
                initial_downstream_pressure,
            )

        self.valve_geometry = self.choke_model.valve_geometry_model.valve_geometry
        self.equilibrium_model = self.choke_model.equilibrium_model.name
        self.recovery = self.choke_model.recovery_model.name
        self.slip_model = self.choke_model.slip_model.name
        self.thermal_phase_equilibrium = bool(thermal_phase_equilibrium)
        self.current_discharge_molar_enthalpy = self.inj_fluid_props["molar_enthalpy"]

        if self.ramp_up_period == 0.0:
            self.current_rate = target_molar_rate
        else:
            self.current_rate = 0.0

        if verbose:
            print(
                f'** UpstreamPressureNodeWithChoke for the segment index {segment_idx} of the pipe "{pipe_geom.pipe_name}" is defined!'
            )

    @property
    def diameter(self) -> float:
        return self.choke_model.valve_geometry_model.diameter

    @property
    def choke_area(self) -> float:
        return self.choke_model.valve_geometry_model.choke_area

    @property
    def effective_area(self) -> float:
        return self.choke_model.valve_geometry_model.effective_area

    @property
    def target_mass_rate_kg_s(self) -> float:
        return self._target_mass_rate_kg_s(self.target_rate)

    def suggest_diameter_for_target_mass_rate(
        self,
        target_mass_rate_kg_s: float,
    ) -> float:
        if self.last_mass_rate_kg_s is None:
            raise ValueError(
                "last_mass_rate_kg_s is not available yet. Run the model first before rescaling the choke."
            )
        return (
            self.choke_model.valve_geometry_model.rescale_diameter_for_target_mass_rate(
                self.last_mass_rate_kg_s,
                target_mass_rate_kg_s,
            )
        )

    def suggest_diameter_for_target_rate(self) -> float:
        return self.suggest_diameter_for_target_mass_rate(self.target_mass_rate_kg_s)

    def _target_mass_rate_kg_s(self, target_molar_rate: float) -> float:
        mw_avg = float(
            np.sum(self.physics.property_containers[0].Mw * self.composition)
        )
        return target_molar_rate * mw_avg / self._SEC_PER_DAY

    def update_current_molar_rate(self, simulation_time):
        engine_x = np.asarray(self.physics.engine.X)
        n_vars = self.physics.n_vars
        downstream_pressure = float(engine_x[self.downstream_block_idx * n_vars])
        self.last_downstream_pressure = downstream_pressure

        choke_eval = self.choke_model.evaluate(downstream_pressure)
        self.last_mass_rate_kg_s = choke_eval.mass_rate_kg_s

        mw_avg = float(
            np.sum(self.physics.property_containers[0].Mw * self.composition)
        )
        molar_rate = (
            choke_eval.mass_rate_kg_s * self._SEC_PER_DAY / mw_avg
            if mw_avg > 0.0
            else 0.0
        )

        if self.ramp_up_period > 0.0:
            ramp_factor = min(max(simulation_time, 0.0) / self.ramp_up_period, 1.0)
        else:
            ramp_factor = 1.0

        self.current_rate = ramp_factor * molar_rate
        if self.max_molar_rate is not None:
            self.current_rate = min(self.current_rate, self.max_molar_rate)
        self.current_discharge_molar_enthalpy = choke_eval.discharge_molar_enthalpy
