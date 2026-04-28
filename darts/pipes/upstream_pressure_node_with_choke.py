import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from dartsflash.libflash import EoS
from scipy.optimize import brentq, minimize_scalar

from darts.engines import value_vector
from darts.pipes.upstream_ramp_up_rate import UpstreamRampUpRate


@dataclass(frozen=True)
class ChokeBoundaryState:
    """Thermodynamic state imposed upstream of the choke."""

    pressure: float
    temperature: float
    composition: np.ndarray
    phase_name: str
    molar_enthalpy: float
    molar_entropy: float


@dataclass(frozen=True)
class ChokeEvaluationResult:
    """Result of solving the choke at the current downstream pipe pressure."""

    mass_rate_kg_s: float
    discharge_molar_enthalpy: float
    throat_pressure: float
    flow_regime: str
    discharge_density: float
    discharge_inv_momentum_density: float
    discharge_gas_mass_fraction: float


@dataclass(frozen=True)
class ChokeFlowState:
    """
    Local state used by the choke pressure-drop equations.

    ``inv_momentum_density`` is the quantity integrated in the Bernoulli-like
    inlet-to-throat pressure-drop expression. With the currently implemented
    no-slip model it is the homogeneous mixture specific volume,
    ``x_g / rho_g + (1 - x_g) / rho_l``. The OLGA Chisholm/slip correction is
    intentionally not folded into this field until that model is implemented.
    """

    pressure: float
    temperature: float
    molar_enthalpy: float
    gas_mass_fraction: float
    inv_momentum_density: float
    density: float
    gas_density: float
    liquid_density: float


@dataclass(frozen=True)
class EquilibriumPhaseState:
    """Phase split returned by a PH flash at a trial choke pressure."""

    pressure: float
    temperature: float
    gas_mass_fraction: float
    gas_density: float
    liquid_density: float
    gas_molar_enthalpy: float
    liquid_molar_enthalpy: float
    gas_phase_composition: np.ndarray | None
    liquid_phase_composition: np.ndarray | None


class ChokePhysicsHelper:
    """
    Adapter between the choke equations and open-DARTS thermodynamic evaluators.

    The choke solver repeatedly asks for states along an isentropic pressure
    path from the upstream boundary to a trial throat/downstream pressure. This
    helper centralizes those flash/property calls and bounds every scalar solve
    by the active OBL axes. It is not an OLGA fluid package implementation: it
    uses open-DARTS EOS/property evaluators directly and does not implement OLGA's
    optional entropy-integration fallback for fluids without entropy tables.
    """

    _TEMPERATURE_ROOT_SAMPLES = 32
    _ENTHALPY_ROOT_SAMPLES = 48

    def __init__(self, physics):
        self.physics = physics
        self.pc = physics.property_containers[0]
        self.eos = self.pc.flash_ev.eos["VL"]
        self.pressure_bounds = self._obl_axis_bounds(
            "pressure"
        ) or self._pt_axis_bounds(0, "pressure")
        self.temperature_bounds = self._obl_axis_bounds(
            "temperature"
        ) or self._pt_axis_bounds(-1, "temperature")
        self.enthalpy_bounds = self._obl_axis_bounds(
            "enthalpy"
        ) or self._computed_ph_axis_bounds("enthalpy")

        if self.pressure_bounds is None:
            raise ValueError("Choke model requires pressure OBL bounds.")
        if self.temperature_bounds is None:
            raise ValueError("Choke model requires temperature OBL bounds.")
        if self.enthalpy_bounds is None:
            raise ValueError("Choke model requires enthalpy OBL bounds.")

    @staticmethod
    def _valid_bounds(lower, upper, name: str) -> tuple[float, float]:
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid {name} OBL bounds: ({lower}, {upper}).")
        return lower, upper

    def _obl_axis_bounds(self, axis_name: str) -> tuple[float, float] | None:
        physics_vars = list(getattr(self.physics, "vars", []))
        if axis_name not in physics_vars:
            return None

        axes_min = getattr(self.physics, "axes_min", None)
        axes_max = getattr(self.physics, "axes_max", None)
        if axes_min is None or axes_max is None:
            return None

        axis_idx = physics_vars.index(axis_name)
        return self._valid_bounds(
            axes_min[axis_idx],
            axes_max[axis_idx],
            axis_name,
        )

    def _pt_axis_bounds(
        self,
        axis_idx: int,
        axis_name: str,
    ) -> tuple[float, float] | None:
        axes_min = getattr(self.physics, "PT_axes_min", None)
        axes_max = getattr(self.physics, "PT_axes_max", None)
        if axes_min is None or axes_max is None:
            return None
        return self._valid_bounds(axes_min[axis_idx], axes_max[axis_idx], axis_name)

    def _computed_ph_axis_bounds(self, axis_name: str) -> tuple[float, float] | None:
        physics_vars = list(getattr(self.physics, "vars", []))
        if axis_name in physics_vars:
            axis_idx = physics_vars.index(axis_name)
        elif axis_name == "enthalpy":
            axis_idx = self.physics.n_vars - 1
        else:
            return None

        axes_min = getattr(self.physics, "PT_axes_min", None)
        axes_max = getattr(self.physics, "PT_axes_max", None)
        if axes_min is None or axes_max is None:
            return None

        state_spec_cls = getattr(self.physics, "StateSpecification", None)
        ph_state_spec = getattr(
            state_spec_cls,
            "PH",
            getattr(self.physics, "state_spec", None),
        )
        computed_axes_min, computed_axes_max = self.physics.determine_obl_bounds(
            min_p=axes_min[0],
            max_p=axes_max[0],
            min_t=axes_min[-1],
            max_t=axes_max[-1],
            min_z=axes_min[1 : self.physics.nc],
            max_z=axes_max[1 : self.physics.nc],
            state_spec=ph_state_spec,
        )
        return self._valid_bounds(
            computed_axes_min[axis_idx],
            computed_axes_max[axis_idx],
            axis_name,
        )

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
        physics_vars = list(getattr(self.physics, "vars", []))
        if "pressure" not in physics_vars or "enthalpy" not in physics_vars:
            raise ValueError(
                "Equilibrium choke models require PH physics with pressure and "
                "enthalpy OBL axes."
            )

        state = np.zeros(self.physics.n_vars, dtype=float)
        state[physics_vars.index("pressure")] = float(pressure)
        state[physics_vars.index("enthalpy")] = float(molar_enthalpy)

        z = np.asarray(composition, dtype=float)
        composition_axis_idxs = [
            idx
            for idx, variable in enumerate(physics_vars)
            if variable not in ("pressure", "enthalpy")
        ]
        if len(composition_axis_idxs) != max(len(z) - 1, 0):
            raise ValueError(
                "Composition size is inconsistent with the PH OBL composition axes."
            )
        for axis_idx, zi in zip(composition_axis_idxs, z[:-1], strict=False):
            state[axis_idx] = zi

        return value_vector(state.tolist())

    def evaluate_equilibrium_mixture_entropy(
        self,
        pressure: float,
        molar_enthalpy: float,
        composition,
    ) -> float:
        """
        Estimate mixture entropy after a PH equilibrium flash.

        This is a mass-weighted combination of phase entropies evaluated by the
        EOS at the flashed temperature/compositions. It is sufficient for the
        current isentropic-equilibrium root solve, but it is not OLGA's documented
        fallback integration from enthalpy, mass fraction, and density.
        """
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
        return float(np.clip(molar_enthalpy, *self.enthalpy_bounds))

    def evaluate_phase_enthalpy(
        self,
        phase_name: str,
        pressure: float,
        temperature: float,
        composition,
    ) -> float:
        return float(
            self.pc.enthalpy_ev[phase_name].evaluate(
                pressure,
                temperature,
                composition,
            )
        )

    def solve_phase_temperature_from_enthalpy(
        self,
        phase_name: str,
        pressure: float,
        composition,
        molar_enthalpy: float,
    ) -> float:
        def enthalpy_residual(temperature: float) -> float:
            return (
                self.evaluate_phase_enthalpy(
                    phase_name,
                    pressure,
                    temperature,
                    composition,
                )
                - molar_enthalpy
            )

        bracket = self.find_bracket(
            np.linspace(*self.temperature_bounds, self._TEMPERATURE_ROOT_SAMPLES),
            enthalpy_residual,
        )
        if bracket is None:
            return float(self.temperature_bounds[1])
        if bracket[0] == bracket[1]:
            return bracket[0]
        return float(brentq(enthalpy_residual, bracket[0], bracket[1]))

    def build_single_phase_flow_state(
        self,
        phase_name: str,
        pressure: float,
        temperature: float,
        composition,
        molar_enthalpy: float = None,
    ) -> ChokeFlowState:
        density = self.evaluate_phase_density(
            phase_name,
            pressure,
            temperature,
            composition,
        )
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("single-phase density must be finite and positive.")

        enthalpy = (
            self.evaluate_phase_enthalpy(
                phase_name,
                pressure,
                temperature,
                composition,
            )
            if molar_enthalpy is None
            else float(molar_enthalpy)
        )
        gas_mass_fraction = 1.0 if phase_name == "G" else 0.0
        return ChokeFlowState(
            pressure=float(pressure),
            temperature=float(temperature),
            molar_enthalpy=float(enthalpy),
            gas_mass_fraction=gas_mass_fraction,
            inv_momentum_density=1.0 / density,
            density=density,
            gas_density=density if phase_name == "G" else np.nan,
            liquid_density=density if phase_name == "L" else np.nan,
        )

    def evaluate_equilibrium_phase_state(
        self,
        pressure: float,
        molar_enthalpy: float,
        composition,
    ) -> EquilibriumPhaseState:
        state = self.build_ph_state(pressure, molar_enthalpy, composition)
        self.pc.evaluate(state)
        self.pc.evaluate_thermal(state)

        gas_density = np.nan
        liquid_density = np.nan
        gas_saturation = 0.0
        liquid_saturation = 0.0
        gas_phase_composition = None
        liquid_phase_composition = None
        gas_molar_enthalpy = np.nan
        liquid_molar_enthalpy = np.nan
        temperature = float(self.pc.temperature)

        if "G" in self.physics.phases:
            gas_idx = self.physics.phases.index("G")
            gas_density = float(self.pc.dens[gas_idx])
            gas_saturation = float(self.pc.sat[gas_idx])
            gas_phase_composition = np.asarray(self.pc.x[gas_idx], dtype=float)
            if np.isfinite(gas_density) and gas_density > 0.0 and gas_saturation > 0.0:
                gas_molar_enthalpy = self.evaluate_phase_enthalpy(
                    "G",
                    pressure,
                    temperature,
                    gas_phase_composition,
                )
        if "L" in self.physics.phases:
            liquid_idx = self.physics.phases.index("L")
            liquid_density = float(self.pc.dens[liquid_idx])
            liquid_saturation = float(self.pc.sat[liquid_idx])
            liquid_phase_composition = np.asarray(self.pc.x[liquid_idx], dtype=float)
            if (
                np.isfinite(liquid_density)
                and liquid_density > 0.0
                and liquid_saturation > 0.0
            ):
                liquid_molar_enthalpy = self.evaluate_phase_enthalpy(
                    "L",
                    pressure,
                    temperature,
                    liquid_phase_composition,
                )

        gas_mass = 0.0
        liquid_mass = 0.0
        if np.isfinite(gas_density) and gas_density > 0.0 and gas_saturation > 0.0:
            gas_mass = gas_saturation * gas_density
        if (
            np.isfinite(liquid_density)
            and liquid_density > 0.0
            and liquid_saturation > 0.0
        ):
            liquid_mass = liquid_saturation * liquid_density

        total_mass = gas_mass + liquid_mass
        gas_mass_fraction = gas_mass / total_mass if total_mass > 0.0 else 0.0
        return EquilibriumPhaseState(
            pressure=float(pressure),
            temperature=temperature,
            gas_mass_fraction=float(gas_mass_fraction),
            gas_density=float(gas_density) if np.isfinite(gas_density) else np.nan,
            liquid_density=float(liquid_density)
            if np.isfinite(liquid_density)
            else np.nan,
            gas_molar_enthalpy=float(gas_molar_enthalpy)
            if np.isfinite(gas_molar_enthalpy)
            else np.nan,
            liquid_molar_enthalpy=float(liquid_molar_enthalpy)
            if np.isfinite(liquid_molar_enthalpy)
            else np.nan,
            gas_phase_composition=gas_phase_composition,
            liquid_phase_composition=liquid_phase_composition,
        )

    def _phase_mw_kg_per_kmol(self, composition) -> float:
        return float(
            np.dot(np.asarray(composition, dtype=float), np.asarray(self.pc.Mw))
        )

    def build_mixed_flow_state(
        self,
        pressure: float,
        temperature: float,
        gas_mass_fraction: float,
        gas_density: float,
        liquid_density: float,
        gas_molar_enthalpy: float,
        liquid_molar_enthalpy: float,
        gas_phase_composition,
        liquid_phase_composition,
    ) -> ChokeFlowState:
        """
        Build a homogeneous gas/liquid state from phase properties.

        The returned momentum density assumes both phases move with the same
        velocity. This matches NOSLIP and full-equilibrium assumptions. The OLGA
        documentation's slip-ratio expression should be introduced here, or in a
        dedicated slip model hook, when CHISHOLM support is implemented.
        """
        gas_mass_fraction = float(np.clip(gas_mass_fraction, 0.0, 1.0))
        liquid_mass_fraction = 1.0 - gas_mass_fraction

        if gas_mass_fraction <= 0.0:
            if not np.isfinite(liquid_density) or liquid_density <= 0.0:
                raise ValueError(
                    "liquid_density must be positive for a liquid-only state."
                )
            return ChokeFlowState(
                pressure=float(pressure),
                temperature=float(temperature),
                molar_enthalpy=float(liquid_molar_enthalpy),
                gas_mass_fraction=0.0,
                inv_momentum_density=1.0 / liquid_density,
                density=float(liquid_density),
                gas_density=np.nan,
                liquid_density=float(liquid_density),
            )
        if gas_mass_fraction >= 1.0:
            if not np.isfinite(gas_density) or gas_density <= 0.0:
                raise ValueError("gas_density must be positive for a gas-only state.")
            return ChokeFlowState(
                pressure=float(pressure),
                temperature=float(temperature),
                molar_enthalpy=float(gas_molar_enthalpy),
                gas_mass_fraction=1.0,
                inv_momentum_density=1.0 / gas_density,
                density=float(gas_density),
                gas_density=float(gas_density),
                liquid_density=np.nan,
            )

        if (
            not np.isfinite(gas_density)
            or gas_density <= 0.0
            or not np.isfinite(liquid_density)
            or liquid_density <= 0.0
        ):
            raise ValueError(
                "both gas_density and liquid_density must be positive for a mixed state."
            )

        gas_mw = self._phase_mw_kg_per_kmol(gas_phase_composition)
        liquid_mw = self._phase_mw_kg_per_kmol(liquid_phase_composition)
        if gas_mw <= 0.0 or liquid_mw <= 0.0:
            raise ValueError("phase molecular weights must be positive.")

        gas_mass_specific_h = gas_molar_enthalpy / gas_mw
        liquid_mass_specific_h = liquid_molar_enthalpy / liquid_mw
        mixture_mass_specific_h = (
            gas_mass_fraction * gas_mass_specific_h
            + liquid_mass_fraction * liquid_mass_specific_h
        )
        inv_moles_per_kg = gas_mass_fraction / gas_mw + liquid_mass_fraction / liquid_mw
        if inv_moles_per_kg <= 0.0:
            raise ValueError("mixture molar density in mass space must be positive.")
        mixture_mw = 1.0 / inv_moles_per_kg
        mixture_molar_enthalpy = mixture_mass_specific_h * mixture_mw

        inv_momentum_density = (
            gas_mass_fraction / gas_density + liquid_mass_fraction / liquid_density
        )
        density = 1.0 / inv_momentum_density
        return ChokeFlowState(
            pressure=float(pressure),
            temperature=float(temperature),
            molar_enthalpy=float(mixture_molar_enthalpy),
            gas_mass_fraction=gas_mass_fraction,
            inv_momentum_density=float(inv_momentum_density),
            density=float(density),
            gas_density=float(gas_density),
            liquid_density=float(liquid_density),
        )

    def build_equilibrium_flow_state(
        self,
        pressure: float,
        molar_enthalpy: float,
        composition,
    ) -> ChokeFlowState:
        phase_state = self.evaluate_equilibrium_phase_state(
            pressure,
            molar_enthalpy,
            composition,
        )
        return self.build_mixed_flow_state(
            pressure=pressure,
            temperature=phase_state.temperature,
            gas_mass_fraction=phase_state.gas_mass_fraction,
            gas_density=phase_state.gas_density,
            liquid_density=phase_state.liquid_density,
            gas_molar_enthalpy=phase_state.gas_molar_enthalpy,
            liquid_molar_enthalpy=phase_state.liquid_molar_enthalpy,
            gas_phase_composition=phase_state.gas_phase_composition
            if phase_state.gas_phase_composition is not None
            else composition,
            liquid_phase_composition=phase_state.liquid_phase_composition
            if phase_state.liquid_phase_composition is not None
            else composition,
        )

    def solve_single_phase_isentropic_state(
        self,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> tuple[float, float]:
        """
        Follow a single frozen phase along an isentropic pressure change.

        OLGA's simplified model can use idealized gamma-based gas expansion.
        Here the temperature is solved from the EOS entropy residual and the
        phase enthalpy is then evaluated at that pressure/temperature.
        """
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
            np.linspace(*self.temperature_bounds, self._TEMPERATURE_ROOT_SAMPLES),
            entropy_residual,
        )
        if bracket is None:
            return (
                float(boundary_state.temperature),
                self.clamp_enthalpy(boundary_state.molar_enthalpy),
            )

        if bracket[0] == bracket[1]:
            temperature = bracket[0]
        else:
            temperature = brentq(entropy_residual, bracket[0], bracket[1])

        molar_enthalpy = self.evaluate_phase_enthalpy(
            phase_name,
            downstream_pressure,
            temperature,
            boundary_state.composition,
        )
        if not np.isfinite(molar_enthalpy):
            return (
                float(temperature),
                self.clamp_enthalpy(boundary_state.molar_enthalpy),
            )
        return float(temperature), self.clamp_enthalpy(molar_enthalpy)

    def solve_single_phase_isentropic_enthalpy(
        self,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        _, molar_enthalpy = self.solve_single_phase_isentropic_state(
            boundary_state,
            downstream_pressure,
        )
        return molar_enthalpy

    def solve_equilibrium_isentropic_enthalpy(
        self,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        """
        Find the PH-flash enthalpy that preserves upstream mixture entropy.

        If no robust entropy bracket is found, the model falls back to the frozen
        single-phase isentropic enthalpy. This keeps the boundary stable but is a
        simplification that should be validated for flashing cases.
        """
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
            np.linspace(*self.enthalpy_bounds, self._ENTHALPY_ROOT_SAMPLES),
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
    """
    Geometry-specific area and momentum-balance terms.

    The main choke solver computes the rate from an integrated compressible
    pressure-drop relation. Geometry models only provide the effective areas and
    denominator terms that distinguish ORIFICE from BEAN behavior. The
    ``mass_rate_kg_s`` and sizing methods are lightweight incompressible-style
    estimates used for initial diameter sizing, not for the production choke
    solve in ``ChokeModel.evaluate``.
    """

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

    @property
    @abstractmethod
    def throat_area(self) -> float:
        pass

    @abstractmethod
    def inlet_to_throat_denominator(
        self,
        upstream_inv_momentum_density: float,
        throat_inv_momentum_density: float,
        upstream_area: float,
    ) -> float:
        pass

    @abstractmethod
    def throat_to_downstream_acceleration_term(
        self,
        throat_inv_momentum_density: float,
        downstream_inv_momentum_density: float,
        downstream_area: float,
    ) -> float:
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
    """
    Zero-length orifice geometry.

    The effective throat area is ``CD * opening * flow_coefficient * area``.
    OLGA valve-coefficient tables and multiple-nozzle aggregation are not
    represented here; callers must pass the equivalent scalar opening/diameter.
    """

    _PA_PER_BAR = 1e5

    def __init__(
        self,
        diameter: float = None,
        discharge_coefficient: float = 0.84,
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
        return math.pi / 4.0 * self._diameter**2

    @property
    def throat_area(self) -> float:
        return self.flow_coefficient * self.opening * self.choke_area

    @property
    def effective_area(self) -> float:
        return self.discharge_coefficient * self.throat_area

    def inlet_to_throat_denominator(
        self,
        upstream_inv_momentum_density: float,
        throat_inv_momentum_density: float,
        upstream_area: float,
    ) -> float:
        return (throat_inv_momentum_density / self.effective_area) ** 2 - (
            upstream_inv_momentum_density / upstream_area
        ) ** 2

    def throat_to_downstream_acceleration_term(
        self,
        throat_inv_momentum_density: float,
        downstream_inv_momentum_density: float,
        downstream_area: float,
    ) -> float:
        return (
            throat_inv_momentum_density / self.effective_area
            - downstream_inv_momentum_density / downstream_area
        ) / downstream_area

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


class BeanValveGeometryModel(OrificeValveGeometryModel):
    """
    Bean geometry with a simple vena-contracta recovery correction.

    This adds the contraction-recovery term used by the implemented pressure-drop
    relation and uses the physical throat area for downstream acceleration. It is
    still a reduced model, not a full OLGA valve-table or nozzle-group model.
    """

    @property
    def valve_geometry(self) -> str:
        return "BEAN"

    def inlet_to_throat_denominator(
        self,
        upstream_inv_momentum_density: float,
        throat_inv_momentum_density: float,
        upstream_area: float,
    ) -> float:
        contraction_recovery = (
            2.0
            * (1.0 / self.discharge_coefficient - 1.0)
            * (throat_inv_momentum_density / self.throat_area) ** 2
        )
        return (
            (throat_inv_momentum_density / self.effective_area) ** 2
            - (upstream_inv_momentum_density / upstream_area) ** 2
            + contraction_recovery
        )

    def throat_to_downstream_acceleration_term(
        self,
        throat_inv_momentum_density: float,
        downstream_inv_momentum_density: float,
        downstream_area: float,
    ) -> float:
        return (
            throat_inv_momentum_density / self.throat_area
            - downstream_inv_momentum_density / downstream_area
        ) / downstream_area


class RecoveryModel(ABC):
    """
    Downstream pressure-recovery policy.

    Recovery maps the throat pressure plus an ideal acceleration-based recovery
    term to a predicted downstream pressure. ``ON`` applies a scalar tuning
    factor; it does not model detailed venturi or geometry-specific diffuser
    recovery.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def tuning(self) -> float:
        pass

    @property
    def uses_downstream_recovery(self) -> bool:
        return False

    def validate(self, valve_geometry: str):
        return

    @abstractmethod
    def predicted_downstream_pressure(
        self,
        throat_pressure: float,
        ideal_pressure_recovery_bar: float,
    ) -> float:
        pass


class NoRecoveryModel(RecoveryModel):
    def __init__(self, tuning: float = 1.0):
        self._tuning = float(tuning)

    @property
    def name(self) -> str:
        return "OFF"

    @property
    def tuning(self) -> float:
        return self._tuning

    def predicted_downstream_pressure(
        self,
        throat_pressure: float,
        ideal_pressure_recovery_bar: float,
    ) -> float:
        return float(throat_pressure)


class DownstreamRecoveryModel(RecoveryModel):
    def __init__(self, recovery: str = "ON", tuning: float = 1.0):
        self._name = recovery.upper()
        self._tuning = float(tuning)

    @property
    def name(self) -> str:
        return self._name

    @property
    def tuning(self) -> float:
        return self._tuning

    @property
    def uses_downstream_recovery(self) -> bool:
        return self._tuning > 0.0

    def predicted_downstream_pressure(
        self,
        throat_pressure: float,
        ideal_pressure_recovery_bar: float,
    ) -> float:
        pressure_rise_bar = max(float(ideal_pressure_recovery_bar), 0.0)
        return float(throat_pressure + self._tuning * pressure_rise_bar)


class SlipModel(ABC):
    """Validation hook for future slip models."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate(self, equilibrium_model_name: str):
        return


class NoSlipModel(SlipModel):
    """Homogeneous velocity assumption for gas and liquid."""

    @property
    def name(self) -> str:
        return "NOSLIP"


class ChisholmSlipModel(SlipModel):
    """
    Placeholder for the OLGA Chisholm slip option.

    The OLGA documentation includes a Chisholm slip relation for the momentum
    density. That equation is not implemented yet, so selecting this model fails during
    validation instead of silently using NOSLIP.
    """

    @property
    def name(self) -> str:
        return "CHISHOLM"

    def validate(self, equilibrium_model_name: str):
        raise NotImplementedError(
            "SLIPMODEL='CHISHOLM' is not implemented yet in the DARTS choke boundary."
        )


class EquilibriumModel(ABC):
    """
    Thermodynamic closure used to evaluate states along the choke.

    Each implementation returns a ``ChokeFlowState`` at a trial pressure. The
    hydraulic solver is shared; only the pressure path and phase split differ.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate(self, slip_model_name: str):
        return

    @abstractmethod
    def flow_state(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        pressure: float,
    ) -> ChokeFlowState:
        pass

    def discharge_molar_enthalpy(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        downstream_pressure: float,
    ) -> float:
        return self.flow_state(
            helper,
            boundary_state,
            downstream_pressure,
        ).molar_enthalpy


class FrozenEquilibriumModel(EquilibriumModel):
    """
    Frozen-composition model.

    For liquid without thermal phase equilibrium this keeps density and enthalpy
    fixed at the user-specified upstream boundary pressure, temperature, and
    composition. Gas, and thermal-equilibrium liquid, follow an EOS-based
    isentropic path.
    """

    def __init__(self, thermal_phase_equilibrium: bool = False):
        self.thermal_phase_equilibrium = bool(thermal_phase_equilibrium)

    @property
    def name(self) -> str:
        return "FROZEN"

    def flow_state(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        pressure: float,
    ) -> ChokeFlowState:
        # OLGA default FROZEN behavior keeps liquid properties tied to the upstream
        # liquid state. Gas is expanded isentropically. If thermal phase
        # equilibrium is enabled, the phase still remains frozen, but follows an
        # isentropic path.
        if boundary_state.phase_name == "L" and not self.thermal_phase_equilibrium:
            upstream_liquid_state = helper.build_single_phase_flow_state(
                phase_name="L",
                pressure=boundary_state.pressure,
                temperature=boundary_state.temperature,
                composition=boundary_state.composition,
                molar_enthalpy=boundary_state.molar_enthalpy,
            )
            return ChokeFlowState(
                pressure=float(pressure),
                temperature=float(boundary_state.temperature),
                molar_enthalpy=float(boundary_state.molar_enthalpy),
                gas_mass_fraction=0.0,
                inv_momentum_density=upstream_liquid_state.inv_momentum_density,
                density=upstream_liquid_state.density,
                gas_density=np.nan,
                liquid_density=upstream_liquid_state.density,
            )
        temperature, molar_enthalpy = helper.solve_single_phase_isentropic_state(
            boundary_state,
            pressure,
        )
        return helper.build_single_phase_flow_state(
            phase_name=boundary_state.phase_name,
            pressure=pressure,
            temperature=temperature,
            composition=boundary_state.composition,
            molar_enthalpy=molar_enthalpy,
        )


class HenryFauskeEquilibriumModel(EquilibriumModel):
    """
    Approximate Henry-Fauske delayed-flashing model.

    This implementation is intentionally partial. It starts from the FROZEN
    state, computes an equilibrium PH flash at the same pressure, and moves the
    gas mass fraction toward equilibrium using ``N = min(x_eq / 0.14, 1)``.
    The pressure-derivative form described in the OLGA documentation is not
    solved explicitly here.
    """

    _XE_REF = 0.14

    def __init__(self, thermal_phase_equilibrium: bool = False):
        self.thermal_phase_equilibrium = bool(thermal_phase_equilibrium)
        self._frozen = FrozenEquilibriumModel(
            thermal_phase_equilibrium=thermal_phase_equilibrium
        )

    @property
    def name(self) -> str:
        return "HENRYFAUSKE"

    def flow_state(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        pressure: float,
    ) -> ChokeFlowState:
        # Henry-Fauske keeps the frozen phase-path thermodynamics but corrects the
        # throat gas fraction toward the equilibrium flash. OLGA documents this
        # as a delayed flashing model with N = min(x_eq / 0.14, 1) for the
        # metastable mass-transfer correction.
        frozen_state = self._frozen.flow_state(
            helper,
            boundary_state,
            pressure,
        )
        h_eq = helper.solve_equilibrium_isentropic_enthalpy(boundary_state, pressure)
        eq_phase_state = helper.evaluate_equilibrium_phase_state(
            pressure=pressure,
            molar_enthalpy=h_eq,
            composition=boundary_state.composition,
        )

        x0 = 1.0 if boundary_state.phase_name == "G" else 0.0
        x_eq = float(np.clip(eq_phase_state.gas_mass_fraction, 0.0, 1.0))
        if x_eq <= x0 + 1e-12:
            return frozen_state

        n_hf = min(x_eq / self._XE_REF, 1.0)
        x_hf = float(np.clip(x0 + n_hf * (x_eq - x0), 0.0, 1.0))
        if x_hf <= 1e-12:
            return frozen_state

        if (
            not np.isfinite(eq_phase_state.gas_density)
            or eq_phase_state.gas_density <= 0.0
        ):
            return frozen_state

        if self.thermal_phase_equilibrium:
            liquid_density = eq_phase_state.liquid_density
            liquid_molar_enthalpy = eq_phase_state.liquid_molar_enthalpy
            liquid_temperature = eq_phase_state.temperature
            liquid_phase_composition = (
                eq_phase_state.liquid_phase_composition
                if eq_phase_state.liquid_phase_composition is not None
                else boundary_state.composition
            )
        else:
            liquid_density = frozen_state.liquid_density
            liquid_molar_enthalpy = frozen_state.molar_enthalpy
            liquid_temperature = frozen_state.temperature
            liquid_phase_composition = boundary_state.composition

        if not np.isfinite(liquid_density) or liquid_density <= 0.0:
            return helper.build_single_phase_flow_state(
                phase_name="G",
                pressure=pressure,
                temperature=eq_phase_state.temperature,
                composition=eq_phase_state.gas_phase_composition
                if eq_phase_state.gas_phase_composition is not None
                else boundary_state.composition,
                molar_enthalpy=eq_phase_state.gas_molar_enthalpy,
            )

        gas_phase_composition = (
            eq_phase_state.gas_phase_composition
            if eq_phase_state.gas_phase_composition is not None
            else boundary_state.composition
        )
        mixture_temperature = (
            x_hf * eq_phase_state.temperature + (1.0 - x_hf) * liquid_temperature
        )
        return helper.build_mixed_flow_state(
            pressure=pressure,
            temperature=mixture_temperature,
            gas_mass_fraction=x_hf,
            gas_density=eq_phase_state.gas_density,
            liquid_density=liquid_density,
            gas_molar_enthalpy=eq_phase_state.gas_molar_enthalpy,
            liquid_molar_enthalpy=liquid_molar_enthalpy,
            gas_phase_composition=gas_phase_composition,
            liquid_phase_composition=liquid_phase_composition,
        )


class FullEquilibriumModel(EquilibriumModel):
    """
    Homogeneous full-equilibrium model.

    The state is found from a PH equilibrium flash constrained to the upstream
    entropy. Because the current momentum density is homogeneous, this model is
    only valid with ``SLIPMODEL='NOSLIP'``.
    """

    @property
    def name(self) -> str:
        return "EQUILIBRIUM"

    def validate(self, slip_model_name: str):
        if slip_model_name != "NOSLIP":
            raise ValueError(
                "EQUILIBRIUMMODEL='EQUILIBRIUM' cannot be combined with a slip model."
            )

    def flow_state(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        pressure: float,
    ) -> ChokeFlowState:
        molar_enthalpy = helper.solve_equilibrium_isentropic_enthalpy(
            boundary_state,
            pressure,
        )
        return helper.build_equilibrium_flow_state(
            pressure=pressure,
            molar_enthalpy=molar_enthalpy,
            composition=boundary_state.composition,
        )


def build_valve_geometry_model(
    valve_geometry: str,
    diameter: float = None,
    discharge_coefficient: float = 0.84,
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
    if valve_geometry == "BEAN":
        return BeanValveGeometryModel(
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


def build_recovery_model(
    recovery: str,
    recovery_tuning: float = 1.0,
) -> RecoveryModel:
    recovery = recovery.upper()
    if not 0.0 <= recovery_tuning <= 1.0:
        raise ValueError("recovery_tuning must be between 0 and 1.")
    if recovery == "OFF":
        return NoRecoveryModel(tuning=recovery_tuning)
    if recovery == "ON":
        return DownstreamRecoveryModel(recovery, tuning=recovery_tuning)
    raise NotImplementedError(f"RECOVERY={recovery!r} is not implemented yet.")


def build_slip_model(slip_model: str) -> SlipModel:
    slip_model = slip_model.upper()
    if slip_model == "NOSLIP":
        return NoSlipModel()
    if slip_model == "CHISHOLM":
        return ChisholmSlipModel()
    raise NotImplementedError(f"SLIPMODEL={slip_model!r} is not implemented yet.")


class ChokeModel:
    """
    Numerical choke solver coupling thermodynamics, geometry, and recovery.

    The implemented hydraulic model follows the OLGA-style structure: integrate
    inverse momentum density from throat to upstream pressure, use geometry
    terms to convert that pressure drop into mass rate, then either match the
    requested downstream pressure or choose the critical maximum-rate throat
    pressure. Critical flow is found by numerical maximization, not by the
    analytical derivative equation in the OLGA documentation.
    """

    _PRESSURE_EPS_BAR = 1e-6
    _MIN_PRESSURE_BAR = 1e-3
    _INTEGRATION_POINTS = 8
    _ROOT_SCAN_POINTS = 24

    def __init__(
        self,
        helper: ChokePhysicsHelper,
        boundary_state: ChokeBoundaryState,
        valve_geometry_model: ValveGeometryModel,
        equilibrium_model: EquilibriumModel,
        recovery_model: RecoveryModel,
        slip_model: SlipModel,
        upstream_area: float,
        downstream_area: float,
    ):
        self.helper = helper
        self.boundary_state = boundary_state
        self.valve_geometry_model = valve_geometry_model
        self.equilibrium_model = equilibrium_model
        self.recovery_model = recovery_model
        self.slip_model = slip_model
        self.upstream_area = float(upstream_area)
        self.downstream_area = float(downstream_area)

        self.recovery_model.validate(self.valve_geometry_model.valve_geometry)
        self.slip_model.validate(self.equilibrium_model.name)
        self.equilibrium_model.validate(self.slip_model.name)

        if self.upstream_area <= 0.0:
            raise ValueError("upstream_area must be positive.")
        if self.downstream_area <= 0.0:
            raise ValueError("downstream_area must be positive.")

    def _state_cache_key(self, pressure: float) -> float:
        return round(float(pressure), 8)

    def _lower_pressure_search_bound(self, upper: float) -> float | None:
        lower = max(self.helper.pressure_bounds[0], self._MIN_PRESSURE_BAR)
        if lower >= upper:
            return None
        return lower

    def _flow_state(
        self,
        pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> ChokeFlowState:
        key = self._state_cache_key(pressure)
        if key not in cache:
            cache[key] = self.equilibrium_model.flow_state(
                self.helper,
                self.boundary_state,
                float(pressure),
            )
        return cache[key]

    def _integrate_inverse_momentum_density(
        self,
        throat_pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Integrate specific volume-like momentum term along the pressure path.

        The integral is evaluated with a fixed trapezoid rule for robustness and
        speed inside boundary updates. Increase ``_INTEGRATION_POINTS`` or replace
        this with adaptive quadrature before relying on strongly flashing cases.
        """
        upstream_pressure = self.boundary_state.pressure
        if throat_pressure >= upstream_pressure:
            return 0.0

        pressures = np.linspace(
            throat_pressure,
            upstream_pressure,
            self._INTEGRATION_POINTS,
        )
        inv_rho_m = np.asarray(
            [self._flow_state(p, cache).inv_momentum_density for p in pressures],
            dtype=float,
        )
        return float(np.trapz(inv_rho_m, pressures * 1e5))

    def _mass_rate_from_throat_pressure(
        self,
        throat_pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Compute mass rate for a candidate throat pressure.

        A non-positive denominator means the current geometry/state combination
        cannot produce a physical acceleration term for this trial pressure, so
        the candidate contributes zero flow instead of failing the simulator.
        """
        upstream_state = self._flow_state(self.boundary_state.pressure, cache)
        throat_state = self._flow_state(throat_pressure, cache)
        integral_term = self._integrate_inverse_momentum_density(
            throat_pressure,
            cache,
        )
        denominator = self.valve_geometry_model.inlet_to_throat_denominator(
            upstream_state.inv_momentum_density,
            throat_state.inv_momentum_density,
            self.upstream_area,
        )
        if (
            not np.isfinite(integral_term)
            or integral_term <= 0.0
            or not np.isfinite(denominator)
            or denominator <= 0.0
            or not np.isfinite(throat_state.inv_momentum_density)
            or throat_state.inv_momentum_density <= 0.0
        ):
            return 0.0
        return math.sqrt(2.0 * integral_term / denominator)

    def _predicted_downstream_pressure(
        self,
        throat_pressure: float,
        target_downstream_pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Predict downstream pressure produced by a trial throat pressure.

        With RECOVERY='OFF' this collapses to the throat pressure. With recovery
        enabled, the ideal acceleration recovery is scaled by ``recovery_tuning``.
        """
        throat_state = self._flow_state(throat_pressure, cache)
        downstream_state = self._flow_state(target_downstream_pressure, cache)
        mass_rate = self._mass_rate_from_throat_pressure(throat_pressure, cache)
        recovery_term_bar = (
            mass_rate**2
            * self.valve_geometry_model.throat_to_downstream_acceleration_term(
                throat_state.inv_momentum_density,
                downstream_state.inv_momentum_density,
                self.downstream_area,
            )
            / 1e5
        )
        return self.recovery_model.predicted_downstream_pressure(
            throat_pressure,
            recovery_term_bar,
        )

    def _find_subcritical_throat_pressure(
        self,
        downstream_pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float | None:
        """
        Solve for the throat pressure that exactly matches downstream pressure.

        If no bracket is found, the caller treats the flow as critical. This is
        deliberately conservative for boundary robustness, but it also means
        poor thermodynamic brackets can force critical behavior.
        """
        upper = min(
            downstream_pressure,
            self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR),
        )
        lower = self._lower_pressure_search_bound(upper)
        if lower is None:
            return None

        def residual(throat_pressure: float) -> float:
            return (
                self._predicted_downstream_pressure(
                    throat_pressure,
                    downstream_pressure,
                    cache,
                )
                - downstream_pressure
            )

        samples = np.linspace(lower, upper, self._ROOT_SCAN_POINTS)
        bracket = self.helper.find_bracket(samples, residual)
        if bracket is None:
            return None
        if bracket[0] == bracket[1]:
            return bracket[0]
        return float(brentq(residual, bracket[0], bracket[1]))

    def _critical_solution(
        self,
        cache: dict[float, ChokeFlowState],
    ) -> tuple[float, float]:
        """
        Find the critical-flow point as the maximum mass rate over throat pressure.

        This numerical optimization replaces the explicit derivative equation in
        the OLGA documentation. It avoids deriving model-specific derivatives but
        depends on smooth property evaluations and adequate pressure bounds.
        """
        upper = self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR)
        lower = self._lower_pressure_search_bound(upper)
        if lower is None:
            return upper, 0.0

        def objective(throat_pressure: float) -> float:
            return -self._mass_rate_from_throat_pressure(throat_pressure, cache)

        optimum = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
        )
        throat_pressure = float(optimum.x)
        return throat_pressure, self._mass_rate_from_throat_pressure(
            throat_pressure,
            cache,
        )

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
        """
        Evaluate mass rate and discharge state for the current pipe pressure.

        The result is later converted to the molar source rate used by the DARTS
        boundary. ``flow_regime`` reports whether the downstream pressure could
        be matched subcritically or whether the critical maximum-rate solution was
        selected.
        """
        cache: dict[float, ChokeFlowState] = {}
        if not self.recovery_model.uses_downstream_recovery:
            critical_throat_pressure, critical_mass_rate_kg_s = self._critical_solution(
                cache
            )
            if downstream_pressure >= critical_throat_pressure:
                throat_pressure = min(
                    downstream_pressure,
                    self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR),
                )
                mass_rate_kg_s = self._mass_rate_from_throat_pressure(
                    throat_pressure,
                    cache,
                )
                flow_regime = "subcritical"
            else:
                throat_pressure = critical_throat_pressure
                mass_rate_kg_s = critical_mass_rate_kg_s
                flow_regime = "critical"
        else:
            throat_pressure = self._find_subcritical_throat_pressure(
                downstream_pressure,
                cache,
            )
            if throat_pressure is None:
                throat_pressure, mass_rate_kg_s = self._critical_solution(cache)
                flow_regime = "critical"
            else:
                mass_rate_kg_s = self._mass_rate_from_throat_pressure(
                    throat_pressure,
                    cache,
                )
                flow_regime = "subcritical"

        discharge_state = self._flow_state(
            downstream_pressure,
            cache,
        )
        return ChokeEvaluationResult(
            mass_rate_kg_s=mass_rate_kg_s,
            discharge_molar_enthalpy=discharge_state.molar_enthalpy,
            throat_pressure=float(throat_pressure),
            flow_regime=flow_regime,
            discharge_density=discharge_state.density,
            discharge_inv_momentum_density=discharge_state.inv_momentum_density,
            discharge_gas_mass_fraction=discharge_state.gas_mass_fraction,
        )


class UpstreamPressureNodeWithChoke(UpstreamRampUpRate):
    """
    Upstream pressure/temperature source connected to the pipe through a choke.

    The implementation is intentionally split into orthogonal model parts, following
    the OLGA choke documentation:

      - valve geometry
      - equilibrium model
      - recovery model
      - slip model

    ORIFICE / BEAN geometries and OFF / ON recovery paths are implemented
    presently, and new models can be added without rewriting the boundary node.
    The implementation is a reduced OLGA-style model, not a feature-complete
    OLGA clone: CHISHOLM slip, valve-coefficient tables, multiple nozzles,
    venturi/standing-valve options, and the full Henry-Fauske derivative
    equations are not implemented.

    The API exposes OLGA-style choke inputs explicitly:
      - discharge_coefficient ~= CD
      - gas_liquid_sizing_ratio ~= CF
      - recovery_tuning ~= CR

    CD affects the effective contraction area directly. CF is stored for future
    valve-table / gas-sizing support and is not used by the present rate solve.
    CR scales the downstream pressure-recovery term when RECOVERY='ON'.
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
        discharge_coefficient: float = 0.84,
        opening: float = 1.0,
        flow_coefficient: float = 1.0,
        gas_liquid_sizing_ratio: float = 1.0,
        thermal_phase_equilibrium: bool = False,
        recovery: str = "OFF",
        recovery_tuning: float = 1.0,
        slip_model: str = "NOSLIP",
        upstream_area: float = None,
        initial_downstream_pressure: float = None,
        max_molar_rate: float = None,
        verbose: bool = False,
    ):
        """
        :param valve_geometry: Valve geometry used in the choke model:
                               - ORIFICE: Orifice type with no spatial extension, vena contracta appears behind the valve.
                               - BEAN: Bean type with spatial extension, vena contracta appears inside the valve.
        :param equilibrium_model: Equilibrium model used in the choke model:
                                  - FROZEN: No mass transfer
                                  - HENRYFAUSKE: Partial equilibrium
                                  - EQUILIBRIUM: Gas/liquid equilibrium
        :param diameter: Maximum valve diameter. Multiple nozzle groups are not supported.
        :param discharge_coefficient: Discharge coefficient
        :param opening: Scalar opening multiplier applied to the choke area.
        :param flow_coefficient: Scalar valve coefficient multiplier applied to
                                 the choke area. Valve tables are not supported.
        :param gas_liquid_sizing_ratio: Ratio between gas and liquid sizing coefficients;
                                        currently stored for compatibility only.
        :param thermal_phase_equilibrium: If set to True, thermal equilibrium between gas and liquid is assumed;
                                          otherwise, the gas is expanded isentropically while the liquid is isothermal.
                                          Used by FROZEN and HENRYFAUSKE closures.
        :param recovery: Enable/disable simplified pressure recovery downstream of the throat.
        :param recovery_tuning: 1 gives maximum recovery and 0 gives zero recovery
        :param slip_model: Slip model for choke throat. Only NOSLIP is currently usable;
                           CHISHOLM is declared but raises NotImplementedError.
        """
        if max_molar_rate is not None and max_molar_rate <= 0.0:
            raise ValueError("max_molar_rate must be positive when specified.")
        if gas_liquid_sizing_ratio <= 0.0:
            raise ValueError("gas_liquid_sizing_ratio must be positive.")

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
        self.last_throat_pressure = None
        self.last_flow_regime = None
        if upstream_area is None:
            upstream_area = pipe_geom.pipe_internal_A

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
        recovery_model = build_recovery_model(
            recovery,
            recovery_tuning=recovery_tuning,
        )
        slip_model_obj = build_slip_model(slip_model)

        self.choke_model = ChokeModel(
            helper=helper,
            boundary_state=boundary_state,
            valve_geometry_model=geometry_model,
            equilibrium_model=equilibrium_model_obj,
            recovery_model=recovery_model,
            slip_model=slip_model_obj,
            upstream_area=upstream_area,
            downstream_area=pipe_geom.pipe_internal_A,
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
        self.recovery_tuning = self.choke_model.recovery_model.tuning
        self.slip_model = self.choke_model.slip_model.name
        self.discharge_coefficient = float(discharge_coefficient)
        self.flow_coefficient = float(flow_coefficient)
        self.gas_liquid_sizing_ratio = float(gas_liquid_sizing_ratio)
        self.thermal_phase_equilibrium = bool(thermal_phase_equilibrium)
        self.current_discharge_molar_enthalpy = self.inj_fluid_props["molar_enthalpy"]
        self.current_discharge_density = None
        self.current_discharge_inv_momentum_density = None
        self.current_discharge_gas_mass_fraction = None

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
    def opening(self) -> float:
        return self.choke_model.valve_geometry_model.opening

    @opening.setter
    def opening(self, value: float):
        if value <= 0.0:
            raise ValueError("opening must be positive.")
        self.choke_model.valve_geometry_model.opening = float(value)

    @property
    def choke_area(self) -> float:
        return self.choke_model.valve_geometry_model.choke_area

    @property
    def effective_area(self) -> float:
        return self.choke_model.valve_geometry_model.effective_area

    @property
    def target_mass_rate_kg_s(self) -> float:
        return self._target_mass_rate_kg_s(self.target_rate)

    def suggest_opening_for_target_mass_rate(
        self,
        target_mass_rate_kg_s: float,
    ) -> float:
        if self.last_mass_rate_kg_s is None:
            raise ValueError(
                "last_mass_rate_kg_s is not available yet. Run the model first before rescaling the choke."
            )
        if self.last_mass_rate_kg_s <= 0.0:
            raise ValueError("actual_mass_rate_kg_s must be positive.")
        if target_mass_rate_kg_s <= 0.0:
            raise ValueError("target_mass_rate_kg_s must be positive.")
        return self.opening * target_mass_rate_kg_s / self.last_mass_rate_kg_s

    def suggest_opening_for_target_rate(self) -> float:
        return self.suggest_opening_for_target_mass_rate(self.target_mass_rate_kg_s)

    def _target_mass_rate_kg_s(self, target_molar_rate: float) -> float:
        mw_avg = float(
            np.sum(self.physics.property_containers[0].Mw * self.composition)
        )
        return target_molar_rate * mw_avg / self._SEC_PER_DAY

    def get_boundary_momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        molar_rate: float = None,
    ) -> float:
        rate = self.current_rate if molar_rate is None else molar_rate
        mw = np.asarray(property_container.Mw)
        mass_rate = float(np.sum(rate * self.composition * mw) / self._SEC_PER_DAY)

        if mass_rate == 0.0:
            return 0.0

        if (
            self.current_discharge_inv_momentum_density is None
            or not np.isfinite(self.current_discharge_inv_momentum_density)
            or self.current_discharge_inv_momentum_density <= 0.0
        ):
            return super().get_boundary_momentum_flux(
                property_container,
                pipe_internal_area,
                molar_rate=rate,
            )

        return (
            mass_rate**2
            * self.current_discharge_inv_momentum_density
            / pipe_internal_area
        )

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
        self.current_discharge_density = choke_eval.discharge_density
        self.current_discharge_inv_momentum_density = (
            choke_eval.discharge_inv_momentum_density
        )
        self.current_discharge_gas_mass_fraction = (
            choke_eval.discharge_gas_mass_fraction
        )
        self.last_throat_pressure = choke_eval.throat_pressure
        self.last_flow_regime = choke_eval.flow_regime
