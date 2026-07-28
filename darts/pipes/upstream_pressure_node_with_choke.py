import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from dartsflash.libflash import EoS
from scipy.optimize import brentq, minimize_scalar

from darts.engines import value_vector
from darts.physics.super.property_container import PropertyContainer
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.upstream_ramp_up_rate import UpstreamRampUpRate


@dataclass(frozen=True)
class ChokeBoundaryState:
    """
    Thermodynamic state imposed upstream of the choke.

    :param pressure: Upstream pressure [bar].
    :param temperature: Upstream temperature [K].
    :param composition: Overall upstream composition.
    :param phase_name: Upstream phase name, currently ``G`` or ``L``.
    :param molar_enthalpy: Upstream molar enthalpy [kJ/kmol].
    :param molar_entropy: Upstream molar entropy from the EOS evaluator.
    """

    pressure: float
    temperature: float
    composition: np.ndarray
    phase_name: str
    molar_enthalpy: float
    molar_entropy: float


@dataclass(frozen=True)
class ChokeEvaluationResult:
    """
    Result of solving the choke at the current downstream pipe pressure.

    :param mass_rate_kg_s: Choke mass rate [kg/s].
    :param discharge_molar_enthalpy: Enthalpy assigned to the pipe source [kJ/kmol].
    :param throat_pressure: Pressure at the controlling restriction state [bar].
    :param flow_regime: ``critical`` or ``subcritical``.
    :param discharge_density: Density used for downstream momentum flux [kg/m3].
    :param discharge_inv_momentum_density: Homogeneous momentum-density inverse [m3/kg].
    :param discharge_gas_mass_fraction: Gas mass fraction in the discharge state.
    """

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

    :param pressure: Local pressure [bar].
    :param temperature: Local temperature [K].
    :param molar_enthalpy: Local molar enthalpy [kJ/kmol].
    :param gas_mass_fraction: Local gas mass fraction.
    :param inv_momentum_density: Momentum-density inverse used in the choke equations [m3/kg].
    :param density: Homogeneous mixture density [kg/m3].
    :param gas_density: Gas density [kg/m3], or ``nan`` if gas is absent.
    :param liquid_density: Liquid density [kg/m3], or ``nan`` if liquid is absent.
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
class DelayedHemTransitionState:
    """
    State at the D-HEM superheat-limit transition.

    :param pressure: Superheat-limit pressure [bar].
    :param metastable_state: Liquid state immediately before flashing.
    :param equilibrium_state: Equilibrium state after isenthalpic flashing at the SHL pressure.
    :param equilibrium_entropy: Entropy of the post-SHL equilibrium state.
    """

    pressure: float
    metastable_state: ChokeFlowState
    equilibrium_state: ChokeFlowState
    equilibrium_entropy: float


@dataclass(frozen=True)
class EquilibriumPhaseState:
    """
    Phase split returned by a PH flash at a trial choke pressure.

    :param pressure: Flash pressure [bar].
    :param temperature: Flash temperature [K].
    :param gas_mass_fraction: Gas mass fraction from the flash.
    :param gas_density: Gas density [kg/m3], or ``nan`` if gas is absent.
    :param liquid_density: Liquid density [kg/m3], or ``nan`` if liquid is absent.
    :param gas_molar_enthalpy: Gas molar enthalpy [kJ/kmol].
    :param liquid_molar_enthalpy: Liquid molar enthalpy [kJ/kmol].
    :param gas_phase_composition: Gas phase composition, if present.
    :param liquid_phase_composition: Liquid phase composition, if present.
    """

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

    def __init__(self, physics: object):
        """
        Store the open-DARTS property evaluators and OBL bounds for choke solves.

        :param physics: open-DARTS physics object that owns the property container,
                        OBL bounds, flash evaluator, and EOS/property evaluators
                        used by the choke model.
        """
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
    def _valid_bounds(lower: float, upper: float, name: str) -> tuple[float, float]:
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
        composition: np.ndarray | Sequence[float],
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
        composition: np.ndarray | Sequence[float],
        root_flag: object,
    ) -> float:
        self.eos.set_root_flag(root_flag)
        return float(eos_entropy(self.eos, pressure, temperature, composition))

    def build_ph_state(
        self,
        pressure: float,
        molar_enthalpy: float,
        composition: np.ndarray | Sequence[float],
    ) -> value_vector:
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
        composition: np.ndarray | Sequence[float],
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
    def find_bracket(
        xs: np.ndarray,
        fn: Callable[[float], float],
    ) -> tuple[float, float] | None:
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
        composition: np.ndarray | Sequence[float],
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
        composition: np.ndarray | Sequence[float],
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
        composition: np.ndarray | Sequence[float],
        molar_enthalpy: float | None = None,
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
        composition: np.ndarray | Sequence[float],
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

    def _phase_mw_kg_per_kmol(self, composition: np.ndarray | Sequence[float]) -> float:
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
        gas_phase_composition: np.ndarray | Sequence[float],
        liquid_phase_composition: np.ndarray | Sequence[float],
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
        composition: np.ndarray | Sequence[float],
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


def eos_entropy(
    eos: object,
    pressure: float,
    temperature: float,
    composition: np.ndarray | Sequence[float],
) -> float:
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
    def diameter(self, value: float) -> None:
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
        diameter: float | None = None,
        discharge_coefficient: float = 0.84,
        opening: float = 1.0,
        flow_coefficient: float = 1.0,
    ):
        """
        Initialize a scalar orifice geometry.

        :param diameter: Choke diameter [m]. If omitted, it is sized from target rate.
        :param discharge_coefficient: Discharge coefficient applied to the effective area.
        :param opening: Scalar multiplier on the physical choke area.
        :param flow_coefficient: Scalar valve coefficient multiplier on the choke area.
        """
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
    def diameter(self, value: float) -> None:
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

    Use ``OFF`` only when the downstream pressure supplied to the choke model is
    intended to represent the throat or vena-contracta pressure itself. Use
    ``ON`` when the downstream pressure is a pipe pressure after
    the restriction, where part of the kinetic energy has already recovered as
    static pressure. The Perkins hydraulic model has its own Perry-orifice
    recovery relation for this ``ON`` case.
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

    def validate(self, valve_geometry: str) -> None:
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
    """
    Validation hook for future slip models.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate(self, equilibrium_model_name: str) -> None:
        return


class NoSlipModel(SlipModel):
    """
    Homogeneous velocity assumption for gas and liquid.
    """

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

    def validate(self, equilibrium_model_name: str) -> None:
        raise NotImplementedError(
            "SLIPMODEL='CHISHOLM' is not implemented yet in the open-DARTS choke boundary."
        )


class EquilibriumModel(ABC):
    """
    Define the thermodynamic closure for states along the choke.

    Each implementation returns a ``ChokeFlowState`` at a trial pressure. The
    hydraulic solver is shared; only the pressure path and phase split differ.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate(self, slip_model_name: str) -> None:
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

    def validate(self, slip_model_name: str) -> None:
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
    diameter: float | None = None,
    discharge_coefficient: float = 0.84,
    opening: float = 1.0,
    flow_coefficient: float = 1.0,
) -> ValveGeometryModel:
    """
    Build the geometry object used by the choke hydraulic model.

    :param valve_geometry: Geometry keyword, currently ``ORIFICE`` or ``BEAN``.
    :param diameter: Choke diameter [m]. If omitted, it is sized from target rate.
    :param discharge_coefficient: Discharge coefficient applied to the effective area.
    :param opening: Scalar multiplier on the physical choke area.
    :param flow_coefficient: Scalar valve coefficient multiplier on the choke area.
    :return: Valve geometry model instance.
    """
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
    """
    Build the thermodynamic closure used along the choke pressure path.

    :param equilibrium_model: Closure keyword, currently ``FROZEN``,
                              ``HENRYFAUSKE``, or ``EQUILIBRIUM``.
    :param thermal_phase_equilibrium: Whether frozen/Henry-Fauske liquid states
                                      follow the thermal-equilibrium path.
    :return: Equilibrium model instance.
    """
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
    """
    Build the downstream pressure-recovery policy.

    :param recovery: Recovery keyword, currently ``OFF`` or ``ON``.
    :param recovery_tuning: Scalar recovery factor between 0 and 1.
    :return: Recovery model instance.
    """
    recovery = recovery.upper()
    if not 0.0 <= recovery_tuning <= 1.0:
        raise ValueError("recovery_tuning must be between 0 and 1.")
    if recovery == "OFF":
        return NoRecoveryModel(tuning=recovery_tuning)
    if recovery == "ON":
        return DownstreamRecoveryModel(recovery, tuning=recovery_tuning)
    raise NotImplementedError(f"RECOVERY={recovery!r} is not implemented yet.")


def build_slip_model(slip_model: str) -> SlipModel:
    """
    Build the slip model used by gas/liquid momentum-density calculations.

    :param slip_model: Slip keyword, currently ``NOSLIP`` or declared
                       ``CHISHOLM`` placeholder.
    :return: Slip model instance.
    """
    slip_model = slip_model.upper()
    if slip_model == "NOSLIP":
        return NoSlipModel()
    if slip_model == "CHISHOLM":
        return ChisholmSlipModel()
    raise NotImplementedError(f"SLIPMODEL={slip_model!r} is not implemented yet.")


def build_hydraulic_choke_model(
    hydraulic_model: str,
    helper: ChokePhysicsHelper,
    boundary_state: ChokeBoundaryState,
    valve_geometry_model: ValveGeometryModel,
    equilibrium_model: EquilibriumModel,
    recovery_model: RecoveryModel,
    slip_model: SlipModel,
    upstream_area: float,
    downstream_area: float,
) -> "ChokeModel":
    """
    Construct the hydraulic choke solver and share thermodynamic options.

    ``OLGA_STYLE`` preserves the original integral pressure-drop implementation.
    ``PERKINS`` switches only the critical/subcritical selection and downstream
    pressure-recovery treatment to the Perkins method.
    ``SINTEF_HEM`` uses the quasi-steady homogeneous-equilibrium restricted-flow
    calculation described by the SINTEF CO2 choke-flow work.
    ``SINTEF_DHEM`` adds the SINTEF delayed-flashing SHL/CNT path for pure CO2.

    :param hydraulic_model: Hydraulic model keyword.
    :param helper: Adapter that evaluates open-DARTS thermodynamic properties.
    :param boundary_state: Upstream thermodynamic state.
    :param valve_geometry_model: Geometry model for areas and contraction terms.
    :param equilibrium_model: Thermodynamic closure for trial pressure states.
    :param recovery_model: Downstream pressure-recovery policy.
    :param slip_model: Gas/liquid slip policy.
    :param upstream_area: Flow area upstream of the choke [m2].
    :param downstream_area: Flow area downstream of the choke [m2].
    :return: Hydraulic choke model instance.
    """
    hydraulic_model = hydraulic_model.upper()
    model_args = dict(
        helper=helper,
        boundary_state=boundary_state,
        valve_geometry_model=valve_geometry_model,
        equilibrium_model=equilibrium_model,
        recovery_model=recovery_model,
        slip_model=slip_model,
        upstream_area=upstream_area,
        downstream_area=downstream_area,
    )
    if hydraulic_model in ("OLGA", "OLGA_STYLE", "INTEGRAL"):
        return ChokeModel(**model_args)
    if hydraulic_model == "PERKINS":
        return PerkinsChokeModel(**model_args)
    if hydraulic_model in ("SINTEF_HEM", "HEM"):
        return SintefHemChokeModel(**model_args)
    if hydraulic_model in ("SINTEF_DHEM", "DHEM", "D-HEM"):
        return SintefDelayedHemChokeModel(**model_args)
    raise NotImplementedError(
        f"hydraulic_model={hydraulic_model!r} is not implemented yet."
    )


class ChokeModel:
    """
    Base class for choke solvers coupling thermodynamics and geometry.

    Subclasses provide the hydraulic relation and critical-flow criterion:
    ``OLGA_STYLE`` uses the existing integral pressure-drop solve, ``PERKINS``
    uses Perkins Eq. A-28/A-30, ``SINTEF_HEM`` uses the homogeneous equilibrium
    mass-flux maximum, and ``SINTEF_DHEM`` adds delayed flashing through the
    superheat-limit/CNT construction from the SINTEF CO2 orifice/nozzle model.
    """

    _PRESSURE_EPS_BAR = 1e-6
    _MIN_PRESSURE_BAR = 1e-3
    _INTEGRATION_POINTS = 8
    _ROOT_SCAN_POINTS = 24

    @property
    def hydraulic_model(self) -> str:
        return "OLGA_STYLE"

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
        """
        Store the shared choke solver state.

        :param helper: Adapter that evaluates open-DARTS thermodynamic properties
                       and isentropic states.
        :param boundary_state: User-specified upstream pressure, temperature,
                               phase, composition, enthalpy, and entropy.
        :param valve_geometry_model: Geometry model that supplies choke area,
                                     effective area, and acceleration terms.
        :param equilibrium_model: Thermodynamic closure for the pressure
                                  path through the choke.
        :param recovery_model: Downstream pressure-recovery option.
        :param slip_model: Gas/liquid velocity model. Only NOSLIP is currently
                           usable.
        :param upstream_area: Flow area upstream of the choke, in m2.
        :param downstream_area: Flow area downstream of the choke, in m2.
        """
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
        return float(np.trapezoid(inv_rho_m, pressures * 1e5))

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

        The result is later converted to the molar source rate used by the open-DARTS
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


class SintefHemChokeModel(ChokeModel):
    """
    SINTEF-style homogeneous-equilibrium restricted-flow model.

    This model is intended for dense/liquid CO2 injection where flashing can
    occur inside the restriction. It follows the quasi-steady formulation used
    by the SINTEF CO2 orifice/nozzle work: along an isentropic equilibrium path,
    Eq. (8) gives the local velocity from the stagnation enthalpy and Eq. (9)
    gives the restriction mass flux as ``rho * u``.

    The implemented model is HEM, not the delayed HEM. It allows equilibrium
    flashing and therefore fixes the main limitation of using the Perkins
    frozen-liquid path for liquid CO2, but it does not yet include the
    superheat-limit/CNT delayed phase-transition path used by D-HEM. The
    critical point is selected as the maximum of ``rho * u`` along the
    isentropic equilibrium path; this is the numerical equivalent of the
    sonic-point selection when an explicit speed-of-sound evaluator is not
    available from open-DARTS.

    ``discharge_coefficient`` is applied as an effective-area multiplier. Use a
    value near 1.0 for nozzle-like restrictions and a contraction coefficient
    for sharp-edged orifice-like restrictions when matching the SINTEF setup.
    """

    _CRITICAL_SCAN_POINTS = 64

    @property
    def hydraulic_model(self) -> str:
        return "SINTEF_HEM"

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
        """
        Store the SINTEF HEM hydraulic choke model inputs.

        :param helper: Adapter that evaluates open-DARTS thermodynamic
                       properties and equilibrium isentropic states.
        :param boundary_state: User-specified upstream stagnation pressure,
                               temperature, phase, composition, enthalpy, and
                               entropy.
        :param valve_geometry_model: Restriction geometry. Its effective area
                                     provides the SINTEF HEM flow area.
        :param equilibrium_model: Must be EQUILIBRIUM so flashing evaluates
                                  as a homogeneous-equilibrium PH state.
        :param recovery_model: Present for API consistency. SINTEF HEM uses the
                               downstream pressure directly for subcritical
                               flow and critical flow is independent of it.
        :param slip_model: Must be NOSLIP, consistent with HEM.
        :param upstream_area: Flow area upstream of the restriction, in m2.
        :param downstream_area: Flow area downstream of the restriction, in m2.
        """
        super().__init__(
            helper=helper,
            boundary_state=boundary_state,
            valve_geometry_model=valve_geometry_model,
            equilibrium_model=equilibrium_model,
            recovery_model=recovery_model,
            slip_model=slip_model,
            upstream_area=upstream_area,
            downstream_area=downstream_area,
        )
        if self.equilibrium_model.name != "EQUILIBRIUM":
            raise ValueError(
                "hydraulic_model='SINTEF_HEM' requires equilibrium_model='EQUILIBRIUM' "
                "so liquid CO2 flashing is included."
            )
        if self.slip_model.name != "NOSLIP":
            raise ValueError(
                "hydraulic_model='SINTEF_HEM' requires slip_model='NOSLIP'."
            )

    def _mass_specific_enthalpy_j_kg(
        self,
        molar_enthalpy: float,
        composition: np.ndarray | Sequence[float],
    ) -> float:
        mw_kg_per_kmol = self.helper._phase_mw_kg_per_kmol(composition)
        if mw_kg_per_kmol <= 0.0:
            raise ValueError("Molecular weight must be positive.")
        return float(molar_enthalpy) / mw_kg_per_kmol * 1000.0

    def _sintef_velocity(
        self,
        pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Compute restriction velocity from SINTEF Eq. (8).

        The upstream boundary state is treated as a stagnation state, so the
        stagnation enthalpy is the upstream static enthalpy. open-DARTS stores
        molar enthalpy in kJ/kmol; the energy equation is evaluated in J/kg.
        """
        state = self._flow_state(pressure, cache)
        h_stagnation = self._mass_specific_enthalpy_j_kg(
            self.boundary_state.molar_enthalpy,
            self.boundary_state.composition,
        )
        h_local = self._mass_specific_enthalpy_j_kg(
            state.molar_enthalpy,
            self.boundary_state.composition,
        )
        delta_h = h_stagnation - h_local
        if not np.isfinite(delta_h) or delta_h <= 0.0:
            return 0.0
        return math.sqrt(2.0 * delta_h)

    def _sintef_mass_flux(
        self,
        pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Return ``rho * u`` from SINTEF Eq. (9), in kg/(m2 s).

        :param pressure: Candidate restriction pressure [bar].
        :param cache: Per-evaluation cache of pressure-indexed flow states.
        """
        state = self._flow_state(pressure, cache)
        if not np.isfinite(state.density) or state.density <= 0.0:
            return 0.0
        return state.density * self._sintef_velocity(pressure, cache)

    def _mass_rate_from_throat_pressure(
        self,
        throat_pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Compute mass rate from the SINTEF HEM mass flux and effective area.

        :param throat_pressure: Candidate restriction pressure [bar].
        :param cache: Per-evaluation cache of pressure-indexed flow states.
        """
        if throat_pressure >= self.boundary_state.pressure:
            return 0.0
        mass_flux = self._sintef_mass_flux(throat_pressure, cache)
        return mass_flux * self.valve_geometry_model.effective_area

    def _critical_solution(
        self,
        cache: dict[float, ChokeFlowState],
    ) -> tuple[float, float]:
        """
        Select the critical HEM mass flux as the maximum of Eq. (9).

        SINTEF describes choking as the point where velocity reaches the speed
        of sound on the calculated path. Because open-DARTS does not expose an
        equilibrium sound-speed evaluator here, the same critical point is found
        by maximizing the steady isentropic mass flux over pressure.
        """
        upper = self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR)
        lower = self._lower_pressure_search_bound(upper)
        if lower is None:
            return upper, 0.0

        def objective(pressure: float) -> float:
            return -self._mass_rate_from_throat_pressure(pressure, cache)

        samples = np.linspace(lower, upper, self._CRITICAL_SCAN_POINTS)
        rates = np.asarray(
            [self._mass_rate_from_throat_pressure(p, cache) for p in samples],
            dtype=float,
        )
        if np.all(~np.isfinite(rates)) or np.nanmax(rates) <= 0.0:
            return upper, 0.0

        best_idx = int(np.nanargmax(rates))
        lo_idx = max(best_idx - 1, 0)
        hi_idx = min(best_idx + 1, len(samples) - 1)
        if lo_idx == hi_idx:
            critical_pressure = float(samples[best_idx])
        else:
            optimum = minimize_scalar(
                objective,
                bounds=(float(samples[lo_idx]), float(samples[hi_idx])),
                method="bounded",
            )
            critical_pressure = float(optimum.x)
        return critical_pressure, self._mass_rate_from_throat_pressure(
            critical_pressure,
            cache,
        )

    def evaluate(self, downstream_pressure: float) -> ChokeEvaluationResult:
        """
        Evaluate SINTEF HEM critical/subcritical restricted flow.

        If the supplied downstream pressure is below the critical pressure found
        on the isentropic equilibrium path, the critical mass flux is used.
        Otherwise Eq. (8)/(9) is evaluated directly at the downstream pressure.
        """
        cache: dict[float, ChokeFlowState] = {}
        critical_throat_pressure, critical_mass_rate_kg_s = self._critical_solution(
            cache
        )
        subcritical_pressure = float(
            np.clip(
                downstream_pressure,
                self._lower_pressure_search_bound(
                    self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR)
                )
                or self._MIN_PRESSURE_BAR,
                self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR),
            )
        )

        if downstream_pressure <= critical_throat_pressure:
            throat_pressure = critical_throat_pressure
            mass_rate_kg_s = critical_mass_rate_kg_s
            flow_regime = "critical"
        else:
            throat_pressure = subcritical_pressure
            mass_rate_kg_s = self._mass_rate_from_throat_pressure(
                throat_pressure,
                cache,
            )
            flow_regime = "subcritical"

        discharge_state = self._flow_state(throat_pressure, cache)
        return ChokeEvaluationResult(
            mass_rate_kg_s=mass_rate_kg_s,
            discharge_molar_enthalpy=discharge_state.molar_enthalpy,
            throat_pressure=float(throat_pressure),
            flow_regime=flow_regime,
            discharge_density=discharge_state.density,
            discharge_inv_momentum_density=discharge_state.inv_momentum_density,
            discharge_gas_mass_fraction=discharge_state.gas_mass_fraction,
        )


class SintefDelayedHemChokeModel(SintefHemChokeModel):
    """
    SINTEF delayed homogeneous-equilibrium restricted-flow model for CO2.

    D-HEM follows the same steady energy equation as HEM, but delays flashing
    until the metastable liquid reaches the superheat limit (SHL). The SHL is
    calculated from classical nucleation theory using SINTEF Eqs. (3)-(7):
    nucleation rate, free-energy barrier, critical bubble radius, kinetic
    prefactor, and ``J = Jcrit``. The liquid-vapor surface tension is the
    Rathjen-Straub CO2 correlation,
    ``sigma = 0.08450 * (1 - T / 304.19)**1.280`` in N/m.

    The implemented path is the SINTEF D-HEM path: liquid isentropic expansion
    to SHL, isenthalpic/isobaric equilibrium flashing at the SHL pressure, then
    isentropic HEM expansion from that equilibrium state. If the post-SHL HEM
    path contains a larger mass-flux maximum, that maximum is used; otherwise
    the SHL flux is the choked flux.

    This implementation is intentionally limited to pure CO2 because the
    Rathjen-Straub constants and the saturation-pressure solve used here are
    pure-component relations. open-DARTS PR thermodynamics are used, so exact
    agreement with SINTEF's GERG/Span-Wagner calculations is not guaranteed.
    """

    _AVOGADRO = 6.02214076e23
    _BOLTZMANN_J_K = 1.380649e-23
    _JCRIT_PER_M3_S = 1.0e12
    _RATHJEN_STRAUB_TC_K = 304.19
    _RATHJEN_STRAUB_SIGMA0_N_M = 0.08450
    _RATHJEN_STRAUB_MU = 1.280
    _SATURATION_PRESSURE_SAMPLES = 160
    _SHL_SCAN_POINTS = 240
    _DHEM_ENTHALPY_ROOT_SAMPLES = 96
    _ROOT_Z_SEPARATION_TOL = 1e-6
    _NO_NUCLEATION_LOG_RATIO = -1e6

    @property
    def hydraulic_model(self) -> str:
        return "SINTEF_DHEM"

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
        """
        Store the SINTEF D-HEM hydraulic choke model inputs.

        :param helper: Adapter that evaluates open-DARTS thermodynamic
                       properties, phase fugacity, and equilibrium PH states.
        :param boundary_state: User-specified upstream stagnation liquid CO2
                               state before the restriction.
        :param valve_geometry_model: Restriction geometry. Its effective area
                                     multiplies the selected D-HEM mass flux.
        :param equilibrium_model: Must be EQUILIBRIUM for the isenthalpic
                                  flash at SHL and the post-SHL HEM path.
        :param recovery_model: Present for API consistency. D-HEM uses the
                               downstream pressure directly for subcritical
                               flow and internally selects the critical point.
        :param slip_model: Must be NOSLIP, consistent with HEM and D-HEM.
        :param upstream_area: Flow area upstream of the restriction, in m2.
        :param downstream_area: Flow area downstream of the restriction, in m2.
        """
        super().__init__(
            helper=helper,
            boundary_state=boundary_state,
            valve_geometry_model=valve_geometry_model,
            equilibrium_model=equilibrium_model,
            recovery_model=recovery_model,
            slip_model=slip_model,
            upstream_area=upstream_area,
            downstream_area=downstream_area,
        )
        component_names = list(getattr(self.helper.pc, "components_name", []))
        composition = np.asarray(self.boundary_state.composition, dtype=float)
        if len(composition) != 1 or not np.isclose(composition[0], 1.0):
            raise ValueError(
                "hydraulic_model='SINTEF_DHEM' currently supports pure CO2 only."
            )
        if component_names and component_names[0].upper() != "CO2":
            raise ValueError(
                "hydraulic_model='SINTEF_DHEM' uses CO2-specific "
                "Rathjen-Straub surface-tension constants."
            )
        self._saturation_pressure_cache: dict[float, float | None] = {}

    def _rathjen_straub_surface_tension_n_m(self, temperature: float) -> float:
        """
        Return Rathjen-Straub CO2 liquid-vapor surface tension in N/m.

        :param temperature: Liquid temperature [K].
        """
        tau = 1.0 - float(temperature) / self._RATHJEN_STRAUB_TC_K
        if tau <= 0.0:
            return 0.0
        return self._RATHJEN_STRAUB_SIGMA0_N_M * tau**self._RATHJEN_STRAUB_MU

    def _saturation_lnphi_difference(
        self,
        pressure: float,
        temperature: float,
    ) -> float | None:
        """
        Return liquid-vapor ln(phi) difference for a two-root pure state.

        Cubic EOS MIN and MAX roots can collapse to the same single root outside
        the two-phase region. Those points must not be accepted as saturation
        roots even though their fugacity difference is numerically zero.
        """
        composition = self.boundary_state.composition
        eos = self.helper.eos
        eos.set_root_flag(EoS.RootFlag.MIN)
        liquid_z = eos.Z(pressure, temperature, composition)
        liquid_lnphi = float(eos.lnphi(pressure, temperature, composition)[0])
        eos.set_root_flag(EoS.RootFlag.MAX)
        vapor_z = eos.Z(pressure, temperature, composition)
        vapor_lnphi = float(eos.lnphi(pressure, temperature, composition)[0])
        if abs(vapor_z - liquid_z) < self._ROOT_Z_SEPARATION_TOL:
            return None
        return liquid_lnphi - vapor_lnphi

    def _saturation_pressure_bar(self, temperature: float) -> float | None:
        """
        Solve pure-CO2 saturation pressure from equality of phase fugacity.

        :param temperature: Temperature [K].
        :return: Saturation pressure [bar], or ``None`` if no two-root state is found.
        """
        temperature = float(temperature)
        cache_key = round(temperature, 8)
        if cache_key in self._saturation_pressure_cache:
            return self._saturation_pressure_cache[cache_key]
        if temperature >= self._RATHJEN_STRAUB_TC_K:
            self._saturation_pressure_cache[cache_key] = None
            return None

        critical_point = self.helper.eos.critical_point(self.boundary_state.composition)
        lower = max(self.helper.pressure_bounds[0], self._MIN_PRESSURE_BAR)
        upper = min(float(critical_point.Pc), self.helper.pressure_bounds[1]) * (
            1.0 - self._PRESSURE_EPS_BAR
        )
        samples = np.linspace(lower, upper, self._SATURATION_PRESSURE_SAMPLES)

        def residual(pressure: float) -> float:
            value = self._saturation_lnphi_difference(pressure, temperature)
            if value is None:
                raise ValueError("EOS roots collapsed outside the two-phase region.")
            return value

        previous_pressure = None
        previous_value = None
        for pressure in samples:
            try:
                value = residual(pressure)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            if not np.isfinite(value):
                continue
            if (
                previous_pressure is not None
                and previous_value is not None
                and previous_value * value < 0.0
            ):
                saturation_pressure = brentq(
                    residual,
                    previous_pressure,
                    pressure,
                )
                self._saturation_pressure_cache[cache_key] = float(saturation_pressure)
                return float(saturation_pressure)
            previous_pressure = float(pressure)
            previous_value = float(value)

        self._saturation_pressure_cache[cache_key] = None
        return None

    def _nucleation_log_rate_ratio(
        self,
        pressure: float,
        temperature: float,
        liquid_density: float,
    ) -> float:
        """
        Return ``ln(J / Jcrit)`` from SINTEF Eqs. (3)-(7).

        ``pressure`` is in bar and density is in kg/m3. The pressure difference
        in the critical-radius/free-energy terms is converted to Pa.
        """
        saturation_pressure = self._saturation_pressure_bar(temperature)
        if saturation_pressure is None or saturation_pressure <= pressure:
            return self._NO_NUCLEATION_LOG_RATIO

        sigma = self._rathjen_straub_surface_tension_n_m(temperature)
        delta_p_pa = (saturation_pressure - pressure) * 1e5
        if sigma <= 0.0 or delta_p_pa <= 0.0 or liquid_density <= 0.0:
            return self._NO_NUCLEATION_LOG_RATIO

        mw_kg_per_kmol = self.helper._phase_mw_kg_per_kmol(
            self.boundary_state.composition
        )
        molecule_mass_kg = mw_kg_per_kmol / (1000.0 * self._AVOGADRO)
        number_density = liquid_density / molecule_mass_kg
        free_energy_barrier = 16.0 * math.pi * sigma**3 / (3.0 * delta_p_pa**2)
        log_prefactor = math.log(number_density) + 0.5 * math.log(
            2.0 * sigma / (math.pi * molecule_mass_kg)
        )
        return (
            log_prefactor
            - free_energy_barrier / (self._BOLTZMANN_J_K * temperature)
            - math.log(self._JCRIT_PER_M3_S)
        )

    def _metastable_liquid_state(
        self,
        pressure: float,
    ) -> ChokeFlowState:
        temperature, molar_enthalpy = self.helper.solve_single_phase_isentropic_state(
            self.boundary_state,
            pressure,
        )
        return self.helper.build_single_phase_flow_state(
            phase_name="L",
            pressure=pressure,
            temperature=temperature,
            composition=self.boundary_state.composition,
            molar_enthalpy=molar_enthalpy,
        )

    def _velocity_from_state(self, state: ChokeFlowState) -> float:
        h_stagnation = self._mass_specific_enthalpy_j_kg(
            self.boundary_state.molar_enthalpy,
            self.boundary_state.composition,
        )
        h_local = self._mass_specific_enthalpy_j_kg(
            state.molar_enthalpy,
            self.boundary_state.composition,
        )
        delta_h = h_stagnation - h_local
        if not np.isfinite(delta_h) or delta_h <= 0.0:
            return 0.0
        return math.sqrt(2.0 * delta_h)

    def _find_transition_state(self) -> DelayedHemTransitionState | None:
        """
        Find the SHL point along the metastable isentropic liquid path.

        The first pressure where ``J >= Jcrit`` is used as Point 2 in the
        SINTEF D-HEM construction.
        """
        upper = self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR)
        lower = self._lower_pressure_search_bound(upper)
        if lower is None:
            return None

        def residual(pressure: float) -> float:
            state = self._metastable_liquid_state(pressure)
            return self._nucleation_log_rate_ratio(
                pressure,
                state.temperature,
                state.liquid_density,
            )

        samples = np.linspace(upper, lower, self._SHL_SCAN_POINTS)
        previous_pressure = float(samples[0])
        previous_value = residual(previous_pressure)
        for pressure in samples[1:]:
            try:
                value = residual(float(pressure))
            except (ValueError, RuntimeError, FloatingPointError):
                previous_pressure = float(pressure)
                previous_value = np.nan
                continue
            if (
                np.isfinite(previous_value)
                and np.isfinite(value)
                and previous_value <= 0.0 <= value
            ):
                shl_pressure = brentq(residual, previous_pressure, float(pressure))
                metastable_state = self._metastable_liquid_state(shl_pressure)
                equilibrium_state = self.helper.build_equilibrium_flow_state(
                    pressure=shl_pressure,
                    molar_enthalpy=metastable_state.molar_enthalpy,
                    composition=self.boundary_state.composition,
                )
                equilibrium_entropy = self.helper.evaluate_equilibrium_mixture_entropy(
                    shl_pressure,
                    metastable_state.molar_enthalpy,
                    self.boundary_state.composition,
                )
                return DelayedHemTransitionState(
                    pressure=float(shl_pressure),
                    metastable_state=metastable_state,
                    equilibrium_state=equilibrium_state,
                    equilibrium_entropy=float(equilibrium_entropy),
                )
            previous_pressure = float(pressure)
            previous_value = float(value)
        return None

    def _post_shl_equilibrium_state(
        self,
        pressure: float,
        transition: DelayedHemTransitionState,
    ) -> ChokeFlowState:
        def entropy_residual(molar_enthalpy: float) -> float:
            return (
                self.helper.evaluate_equilibrium_mixture_entropy(
                    pressure,
                    molar_enthalpy,
                    self.boundary_state.composition,
                )
                - transition.equilibrium_entropy
            )

        bracket = self.helper.find_bracket(
            np.linspace(*self.helper.enthalpy_bounds, self._DHEM_ENTHALPY_ROOT_SAMPLES),
            entropy_residual,
        )
        if bracket is None:
            raise ValueError(
                "Could not find an equilibrium entropy bracket for the D-HEM "
                "post-SHL path."
            )
        if bracket[0] == bracket[1]:
            molar_enthalpy = bracket[0]
        else:
            molar_enthalpy = brentq(entropy_residual, bracket[0], bracket[1])
        return self.helper.build_equilibrium_flow_state(
            pressure=pressure,
            molar_enthalpy=self.helper.clamp_enthalpy(molar_enthalpy),
            composition=self.boundary_state.composition,
        )

    def _mass_rate_from_state(self, state: ChokeFlowState) -> float:
        if not np.isfinite(state.density) or state.density <= 0.0:
            return 0.0
        return (
            state.density
            * self._velocity_from_state(state)
            * self.valve_geometry_model.effective_area
        )

    def _post_shl_mass_rate(
        self,
        pressure: float,
        transition: DelayedHemTransitionState,
    ) -> float:
        state = self._post_shl_equilibrium_state(pressure, transition)
        return self._mass_rate_from_state(state)

    def _critical_solution(
        self,
        transition: DelayedHemTransitionState,
    ) -> tuple[float, float, ChokeFlowState]:
        """
        Select the D-HEM choked state from SHL flux and post-SHL HEM flux.

        SINTEF states that, when the post-SHL HEM path also reaches sonic
        conditions, the D-HEM choke flux is the maximum of the SHL flux and the
        post-SHL HEM choke flux.
        """
        shl_mass_rate = self._mass_rate_from_state(transition.metastable_state)
        lower = self._lower_pressure_search_bound(
            transition.pressure * (1.0 - self._PRESSURE_EPS_BAR)
        )
        if lower is None:
            return transition.pressure, shl_mass_rate, transition.equilibrium_state

        def objective(pressure: float) -> float:
            return -self._post_shl_mass_rate(pressure, transition)

        samples = np.linspace(
            lower,
            transition.pressure * (1.0 - self._PRESSURE_EPS_BAR),
            self._CRITICAL_SCAN_POINTS,
        )
        rates = np.asarray(
            [self._post_shl_mass_rate(p, transition) for p in samples],
            dtype=float,
        )
        if np.all(~np.isfinite(rates)) or np.nanmax(rates) <= 0.0:
            return transition.pressure, shl_mass_rate, transition.equilibrium_state

        best_idx = int(np.nanargmax(rates))
        lo_idx = max(best_idx - 1, 0)
        hi_idx = min(best_idx + 1, len(samples) - 1)
        if lo_idx == hi_idx:
            post_critical_pressure = float(samples[best_idx])
        else:
            optimum = minimize_scalar(
                objective,
                bounds=(float(samples[lo_idx]), float(samples[hi_idx])),
                method="bounded",
            )
            post_critical_pressure = float(optimum.x)

        post_critical_state = self._post_shl_equilibrium_state(
            post_critical_pressure,
            transition,
        )
        post_critical_rate = self._mass_rate_from_state(post_critical_state)
        if shl_mass_rate >= post_critical_rate:
            return transition.pressure, shl_mass_rate, transition.equilibrium_state
        return post_critical_pressure, post_critical_rate, post_critical_state

    def evaluate(self, downstream_pressure: float) -> ChokeEvaluationResult:
        """
        Evaluate SINTEF D-HEM critical/subcritical restricted flow.

        Above SHL, the downstream pressure is evaluated on the metastable liquid
        path. Below SHL, the post-transition equilibrium path is used unless the
        SHL or post-SHL maximum has already choked the flow.
        """
        transition = self._find_transition_state()
        if transition is None:
            pressure = float(
                np.clip(
                    downstream_pressure,
                    self._lower_pressure_search_bound(
                        self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR)
                    )
                    or self._MIN_PRESSURE_BAR,
                    self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR),
                )
            )
            state = self._metastable_liquid_state(pressure)
            return ChokeEvaluationResult(
                mass_rate_kg_s=self._mass_rate_from_state(state),
                discharge_molar_enthalpy=state.molar_enthalpy,
                throat_pressure=pressure,
                flow_regime="subcritical",
                discharge_density=state.density,
                discharge_inv_momentum_density=state.inv_momentum_density,
                discharge_gas_mass_fraction=state.gas_mass_fraction,
            )

        critical_pressure, critical_mass_rate, critical_state = self._critical_solution(
            transition
        )
        if downstream_pressure <= critical_pressure:
            state = critical_state
            pressure = critical_pressure
            mass_rate = critical_mass_rate
            regime = "critical"
        elif downstream_pressure > transition.pressure:
            pressure = float(
                np.clip(
                    downstream_pressure,
                    transition.pressure,
                    self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR),
                )
            )
            state = self._metastable_liquid_state(pressure)
            mass_rate = self._mass_rate_from_state(state)
            regime = "subcritical"
        else:
            pressure = float(
                np.clip(
                    downstream_pressure,
                    self._lower_pressure_search_bound(
                        transition.pressure * (1.0 - self._PRESSURE_EPS_BAR)
                    )
                    or self._MIN_PRESSURE_BAR,
                    transition.pressure * (1.0 - self._PRESSURE_EPS_BAR),
                )
            )
            state = self._post_shl_equilibrium_state(pressure, transition)
            mass_rate = self._mass_rate_from_state(state)
            regime = "subcritical"

        return ChokeEvaluationResult(
            mass_rate_kg_s=mass_rate,
            discharge_molar_enthalpy=state.molar_enthalpy,
            throat_pressure=float(pressure),
            flow_regime=regime,
            discharge_density=state.density,
            discharge_inv_momentum_density=state.inv_momentum_density,
            discharge_gas_mass_fraction=state.gas_mass_fraction,
        )


class PerkinsChokeModel(ChokeModel):
    """
    Perkins critical/subcritical choke model with open-DARTS thermodynamics.

    Perkins derives the ideal mass rate from the steady adiabatic energy
    equation for a homogeneous mixture and applies a discharge coefficient to
    obtain the actual rate. This implementation evaluates Perkins Eq. A-28 in
    SI units. The gas polytropic exponent is estimated from an open-DARTS
    isentropic gas expansion at the upstream state; liquid is treated as
    incompressible, consistent with Perkins' assumptions.

    When ``RECOVERY='ON'``, the subcritical throat pressure is inferred from the
    recovered downstream pressure with Perkins' Perry-orifice pressure recovery
    relation,
    ``p_throat = p_up - (p_up - p_downstream) / (1 - (d_choke / d_pipe)**1.85)``.
    When recovery is disabled, the downstream pipe pressure is treated as the
    throat pressure. The model solves Perkins Eq. A-30 for the critical pressure
    ratio. If Eq. A-30 cannot be solved, the model raises an error instead of
    substituting a different critical-flow criterion.
    """

    _PERRY_DIAMETER_RATIO_EXPONENT = 1.85
    _POLYTROPIC_PRESSURE_STEP = 1e-3

    @property
    def hydraulic_model(self) -> str:
        return "PERKINS"

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
        """
        Store the Perkins hydraulic choke model inputs.

        :param helper: Adapter that evaluates open-DARTS thermodynamic properties
                       and isentropic states.
        :param boundary_state: User-specified upstream pressure, temperature,
                               phase, composition, enthalpy, and entropy.
        :param valve_geometry_model: Orifice geometry for the Perkins
                                     pressure-recovery and area terms.
        :param equilibrium_model: Thermodynamic closure for the
                                  isentropic pressure path.
        :param recovery_model: If ON, treats downstream pressure as recovered
                               pipe pressure and estimates throat pressure using
                               Perkins' Perry-orifice relation. If OFF,
                               downstream pressure is treated as throat pressure.
        :param slip_model: Gas/liquid velocity model. Only NOSLIP is currently
                           usable.
        :param upstream_area: Flow area upstream of the choke, in m2.
        :param downstream_area: Flow area downstream of the choke, in m2.
        """
        super().__init__(
            helper=helper,
            boundary_state=boundary_state,
            valve_geometry_model=valve_geometry_model,
            equilibrium_model=equilibrium_model,
            recovery_model=recovery_model,
            slip_model=slip_model,
            upstream_area=upstream_area,
            downstream_area=downstream_area,
        )
        if self.valve_geometry_model.valve_geometry != "ORIFICE":
            raise NotImplementedError(
                "hydraulic_model='PERKINS' currently supports VALVEGEOMETRY='ORIFICE' only."
            )
        self._perkins_parameter_cache = None

    def _estimate_gas_polytropic_exponent(self) -> float:
        """
        Estimate Perkins' gas exponent ``n`` from an isentropic gas expansion.

        Perkins computes ``n`` from heat capacities. Those are not always exposed
        by the open-DARTS property container, so this method evaluates the same
        physical quantity from ``p v_g**n = constant`` over a small pressure step.
        """
        upstream_pressure = self.boundary_state.pressure
        trial_pressure = upstream_pressure * (1.0 - self._POLYTROPIC_PRESSURE_STEP)
        lower = self._lower_pressure_search_bound(upstream_pressure)
        if lower is not None:
            trial_pressure = max(trial_pressure, lower)
        if trial_pressure >= upstream_pressure:
            raise ValueError(
                "Cannot estimate Perkins gas exponent inside pressure bounds."
            )

        gas_entropy = self.helper.evaluate_phase_entropy(
            upstream_pressure,
            self.boundary_state.temperature,
            self.boundary_state.composition,
            self.helper.phase_root_flag("G"),
        )
        gas_enthalpy = self.helper.evaluate_phase_enthalpy(
            "G",
            upstream_pressure,
            self.boundary_state.temperature,
            self.boundary_state.composition,
        )
        gas_boundary_state = ChokeBoundaryState(
            pressure=upstream_pressure,
            temperature=self.boundary_state.temperature,
            composition=self.boundary_state.composition,
            phase_name="G",
            molar_enthalpy=gas_enthalpy,
            molar_entropy=gas_entropy,
        )
        trial_temperature, _ = self.helper.solve_single_phase_isentropic_state(
            gas_boundary_state,
            trial_pressure,
        )
        upstream_gas_density = self.helper.evaluate_phase_density(
            "G",
            upstream_pressure,
            self.boundary_state.temperature,
            self.boundary_state.composition,
        )
        trial_gas_density = self.helper.evaluate_phase_density(
            "G",
            trial_pressure,
            trial_temperature,
            self.boundary_state.composition,
        )
        if upstream_gas_density <= 0.0 or trial_gas_density <= 0.0:
            raise ValueError(
                "Gas density must be positive to estimate Perkins exponent."
            )

        upstream_gas_specific_volume = 1.0 / upstream_gas_density
        trial_gas_specific_volume = 1.0 / trial_gas_density
        exponent = math.log(trial_pressure / upstream_pressure) / math.log(
            upstream_gas_specific_volume / trial_gas_specific_volume
        )
        if not np.isfinite(exponent) or exponent <= 1.0:
            raise ValueError(f"Invalid Perkins gas exponent: {exponent}.")
        return float(exponent)

    def _perkins_parameters(
        self,
        cache: dict[float, ChokeFlowState],
    ) -> tuple[float, float, float, float, float]:
        """
        Return ``fg``, ``alpha1``, ``lambda``, ``n``, and upstream reference volume.

        Perkins defines ``fg`` as upstream gas mass fraction and ``alpha1`` as
        the upstream liquid-volume contribution divided by the upstream gas
        specific volume. The original paper contains separate oil and water
        terms; this implementation uses the single liquid phase exposed by
        open-DARTS.
        """
        if self._perkins_parameter_cache is not None:
            return self._perkins_parameter_cache

        upstream_state = self._flow_state(self.boundary_state.pressure, cache)
        gas_mass_fraction = float(np.clip(upstream_state.gas_mass_fraction, 0.0, 1.0))
        liquid_mass_fraction = 1.0 - gas_mass_fraction
        upstream_mixture_specific_volume = upstream_state.inv_momentum_density
        if (
            not np.isfinite(upstream_mixture_specific_volume)
            or upstream_mixture_specific_volume <= 0.0
        ):
            raise ValueError(
                "Perkins model requires positive upstream specific volume."
            )

        liquid_volume_term = 0.0
        if liquid_mass_fraction > 0.0:
            if (
                not np.isfinite(upstream_state.liquid_density)
                or upstream_state.liquid_density <= 0.0
            ):
                raise ValueError(
                    "Perkins model requires positive upstream liquid density."
                )
            liquid_volume_term = liquid_mass_fraction / upstream_state.liquid_density

        if gas_mass_fraction <= 1e-12:
            upstream_reference_specific_volume = upstream_mixture_specific_volume
            alpha1 = 1.0
            polytropic_exponent = 1.0
            lambda_perkins = 0.0
        else:
            if (
                not np.isfinite(upstream_state.gas_density)
                or upstream_state.gas_density <= 0.0
            ):
                raise ValueError(
                    "Perkins model requires positive upstream gas density."
                )
            upstream_reference_specific_volume = 1.0 / upstream_state.gas_density
            alpha1 = liquid_volume_term / upstream_reference_specific_volume
            polytropic_exponent = self._estimate_gas_polytropic_exponent()
            lambda_perkins = (
                gas_mass_fraction * polytropic_exponent / (polytropic_exponent - 1.0)
            )

        self._perkins_parameter_cache = (
            gas_mass_fraction,
            float(alpha1),
            float(lambda_perkins),
            float(polytropic_exponent),
            float(upstream_reference_specific_volume),
        )
        return self._perkins_parameter_cache

    def _perkins_dimensionless_terms(
        self,
        pressure_ratio: float,
        cache: dict[float, ChokeFlowState],
    ) -> tuple[float, float, float, float, float, float]:
        """
        Evaluate the reusable dimensionless groups in Perkins Eq. A-28/A-30.

        :return: ``fg``, ``alpha1``, ``lambda``, ``n``, ``Y``, and ``D`` where
                 ``Y`` is the numerator energy term and ``D`` is the finite
                 upstream-area correction.
        """
        fg, alpha1, lambda_perkins, n, _ = self._perkins_parameters(cache)
        pressure_ratio = float(pressure_ratio)
        if not 0.0 < pressure_ratio < 1.0:
            raise ValueError("Perkins pressure ratio must be between 0 and 1.")

        gas_liquid_volume_ratio = fg * pressure_ratio ** (-1.0 / n) + alpha1
        if gas_liquid_volume_ratio <= 0.0:
            raise ValueError("Invalid Perkins gas/liquid volume-ratio term.")

        area_ratio = self.valve_geometry_model.throat_area / self.upstream_area
        velocity_ratio = (fg + alpha1) / gas_liquid_volume_ratio
        denominator = 1.0 - area_ratio**2 * velocity_ratio**2
        energy_term = lambda_perkins * (
            1.0 - pressure_ratio ** ((n - 1.0) / n)
        ) + alpha1 * (1.0 - pressure_ratio)
        return fg, alpha1, lambda_perkins, n, energy_term, denominator

    def _perkins_mass_rate_from_pressure(
        self,
        throat_pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Compute actual mass rate from Perkins Eq. A-28.

        The equation is evaluated in SI units, replacing the oilfield-unit
        ``288 g_c`` factor by ``2``. The ideal isentropic rate is multiplied by
        the discharge coefficient to obtain the actual rate.
        """
        upstream_pressure = self.boundary_state.pressure
        if throat_pressure >= upstream_pressure:
            return 0.0

        pressure_ratio = throat_pressure / upstream_pressure
        fg, alpha1, _, n, energy_term, denominator = self._perkins_dimensionless_terms(
            pressure_ratio,
            cache,
        )
        _, _, _, _, upstream_specific_volume = self._perkins_parameters(cache)
        gas_liquid_volume_ratio = fg * pressure_ratio ** (-1.0 / n) + alpha1
        if energy_term <= 0.0 or denominator <= 0.0 or gas_liquid_volume_ratio <= 0.0:
            return 0.0

        upstream_pressure_pa = upstream_pressure * 1e5
        ideal_rate = self.valve_geometry_model.throat_area * math.sqrt(
            2.0
            * upstream_pressure_pa
            / upstream_specific_volume
            * energy_term
            / (denominator * gas_liquid_volume_ratio**2)
        )
        return self.valve_geometry_model.discharge_coefficient * ideal_rate

    def _mass_rate_from_throat_pressure(
        self,
        throat_pressure: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Evaluate Perkins Eq. A-28 instead of the base integral rate equation.

        :param throat_pressure: Candidate throat pressure [bar].
        :param cache: Per-evaluation cache of pressure-indexed flow states.
        """
        return self._perkins_mass_rate_from_pressure(throat_pressure, cache)

    def _perkins_critical_residual(
        self,
        pressure_ratio: float,
        cache: dict[float, ChokeFlowState],
    ) -> float:
        """
        Perkins Eq. A-30 residual for the critical pressure ratio.

        The residual is written as left-minus-right using the notation from the
        paper. A zero residual is equivalent to ``dwi / dpr = 0`` for Eq. A-28.
        """
        fg, alpha1, lambda_perkins, n, energy_term, denominator = (
            self._perkins_dimensionless_terms(pressure_ratio, cache)
        )
        if fg <= 1e-12:
            return np.nan

        area_ratio = self.valve_geometry_model.throat_area / self.upstream_area
        gas_liquid_volume_ratio = fg * pressure_ratio ** (-1.0 / n) + alpha1
        pressure_derivative_term = fg / n * pressure_ratio ** (-(1.0 + n) / n)
        area_derivative_term = (
            area_ratio**2
            * fg
            / n
            * (fg + alpha1) ** 2
            * pressure_ratio ** (-(1.0 + n) / n)
            / gas_liquid_volume_ratio**2
        )
        left = (
            2.0
            * energy_term
            * (denominator * pressure_derivative_term + area_derivative_term)
        )
        right = (
            denominator
            * gas_liquid_volume_ratio
            * (lambda_perkins * (n - 1.0) / n * pressure_ratio ** (-1.0 / n) + alpha1)
        )
        return left - right

    def _critical_solution(
        self,
        cache: dict[float, ChokeFlowState],
    ) -> tuple[float, float] | None:
        """
        Solve Perkins Eq. A-30 for critical pressure ratio.

        Pure-liquid flow has no gas-expansion critical root in Eq. A-30, so this
        returns ``None`` and the caller evaluates the subcritical Perkins rate.
        For gas or gas/liquid flow, failure to bracket Eq. A-30 is treated as an
        error because no alternative critical-flow equation is introduced here.
        """
        upper_pressure = self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR)
        lower_pressure = self._lower_pressure_search_bound(upper_pressure)
        if lower_pressure is None:
            return upper_pressure, 0.0

        gas_mass_fraction, _, _, _, upstream_specific_volume = self._perkins_parameters(
            cache
        )
        if gas_mass_fraction <= 1e-12:
            return None
        if upstream_specific_volume <= 0.0:
            raise ValueError(
                "Perkins model requires positive upstream specific volume."
            )

        lower_ratio = max(lower_pressure / self.boundary_state.pressure, 1e-8)
        upper_ratio = upper_pressure / self.boundary_state.pressure

        def residual(pressure_ratio: float) -> float:
            return self._perkins_critical_residual(pressure_ratio, cache)

        samples = np.linspace(lower_ratio, upper_ratio, self._ROOT_SCAN_POINTS)
        bracket = self.helper.find_bracket(samples, residual)
        if bracket is not None:
            if bracket[0] == bracket[1]:
                critical_ratio = bracket[0]
            else:
                critical_ratio = brentq(residual, bracket[0], bracket[1])
            throat_pressure = float(critical_ratio * self.boundary_state.pressure)
            return throat_pressure, self._perkins_mass_rate_from_pressure(
                throat_pressure,
                cache,
            )

        raise ValueError(
            "Could not bracket Perkins Eq. A-30 for the critical pressure ratio. "
            "No numerical-maximization fallback is used because it is not part of "
            "the Perkins implementation."
        )

    def _perkins_subcritical_throat_pressure(
        self,
        recovered_downstream_pressure: float,
    ) -> float:
        """
        Estimate the subcritical throat pressure from recovered downstream pressure.

        Perkins uses this pressure-recovery correction because the measurable
        downstream pressure is normally taken after turbulent recovery, not at the
        vena contracta. The relation is only defined when the choke diameter is
        smaller than the downstream pipe diameter.
        """
        upper = self.boundary_state.pressure * (1.0 - self._PRESSURE_EPS_BAR)
        if not self.recovery_model.uses_downstream_recovery:
            return float(
                np.clip(
                    recovered_downstream_pressure,
                    self._MIN_PRESSURE_BAR,
                    upper,
                )
            )

        diameter_ratio = math.sqrt(
            self.valve_geometry_model.throat_area / self.downstream_area
        )
        if not 0.0 < diameter_ratio < 1.0:
            raise ValueError(
                "Perkins pressure recovery requires choke diameter smaller than "
                "downstream pipe diameter."
            )

        recovery_denominator = 1.0 - diameter_ratio**self._PERRY_DIAMETER_RATIO_EXPONENT
        if recovery_denominator <= 0.0:
            raise ValueError("Invalid Perkins pressure-recovery denominator.")

        throat_pressure = (
            self.boundary_state.pressure
            - (self.boundary_state.pressure - recovered_downstream_pressure)
            / recovery_denominator
        )
        lower = self._lower_pressure_search_bound(upper)
        if lower is None:
            return upper
        return float(np.clip(throat_pressure, lower, upper))

    def evaluate(self, downstream_pressure: float) -> ChokeEvaluationResult:
        """
        Evaluate the Perkins flow regime and mass rate.

        The critical throat pressure is determined first. If the inferred
        subcritical throat pressure lies below that critical pressure, the flow is
        choked and independent of the recovered downstream pressure. Otherwise
        the inferred throat pressure is used directly in the mass-rate equation.
        Pure-liquid flow has no Eq. A-30 gas-expansion root and is evaluated as
        subcritical.
        """
        cache: dict[float, ChokeFlowState] = {}
        subcritical_throat_pressure = self._perkins_subcritical_throat_pressure(
            downstream_pressure
        )
        critical_solution = self._critical_solution(cache)

        if (
            critical_solution is not None
            and critical_solution[0] > subcritical_throat_pressure
        ):
            critical_throat_pressure, critical_mass_rate_kg_s = critical_solution
            throat_pressure = critical_throat_pressure
            mass_rate_kg_s = critical_mass_rate_kg_s
            flow_regime = "critical"
        else:
            throat_pressure = subcritical_throat_pressure
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

      - hydraulic model
      - valve geometry
      - equilibrium model
      - recovery model
      - slip model

    ORIFICE / BEAN geometries and OFF / ON recovery paths are implemented
    presently, and new models can be added without rewriting the boundary node.
    The implementation is a reduced OLGA-style model, not a feature-complete
    OLGA clone: CHISHOLM slip, valve-coefficient tables, multiple nozzles,
    venturi/standing-valve options, and the full Henry-Fauske derivative
    equations are not implemented. ``hydraulic_model='PERKINS'`` is available
    for the Perkins energy-equation critical/subcritical choke method.
    ``hydraulic_model='SINTEF_HEM'`` is available for dense/liquid CO2 injection
    with homogeneous-equilibrium flashing through the restriction.
    ``hydraulic_model='SINTEF_DHEM'`` is available for pure CO2 injection with
    delayed flashing from the SINTEF SHL/CNT construction.

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
        pipe_geom: PipeGeometry,
        reservoir: object,
        physics: object,
        first_ts_size: float,
        segment_idx: int,
        target_molar_rate: float,
        ramp_up_period: float,
        composition: np.ndarray | Sequence[float],
        pressure: float,
        temperature: float,
        phase_name: str,
        hydraulic_model: str = "OLGA_STYLE",
        valve_geometry: str = "ORIFICE",
        equilibrium_model: str = "FROZEN",
        diameter: float | None = None,
        discharge_coefficient: float = 0.84,
        opening: float = 1.0,
        flow_coefficient: float = 1.0,
        gas_liquid_sizing_ratio: float = 1.0,
        thermal_phase_equilibrium: bool = False,
        recovery: str = "OFF",
        recovery_tuning: float = 1.0,
        slip_model: str = "NOSLIP",
        upstream_area: float | None = None,
        initial_downstream_pressure: float | None = None,
        max_molar_rate: float | None = None,
        verbose: bool = False,
    ):
        """
        Connect an upstream pressure/temperature boundary through a choke.

        :param pipe_name: Name of the pipe/well receiving the boundary source.
        :param pipe_geom: Pipe geometry object that supplies pipe area and name.
        :param reservoir: Reservoir object that locates the connected pipe segment
                          in the global engine state.
        :param physics: open-DARTS physics object for composition, enthalpy,
                        entropy, and density evaluations.
        :param first_ts_size: Size of the first time step [day].
        :param segment_idx: Pipe segment index connected to the boundary source.
        :param target_molar_rate: Ramp target before choke limiting [kmol/day].
        :param ramp_up_period: Period over which the boundary ramps from zero [day].
        :param composition: Injected fluid composition.
        :param pressure: Upstream stagnation pressure before the choke [bar].
        :param temperature: Upstream stagnation temperature before the choke [K].
        :param phase_name: Upstream phase name, currently ``G`` or ``L``.
        :param hydraulic_model: Hydraulic choke method:
                                - OLGA_STYLE: Existing integral pressure-drop solver.
                                - PERKINS: Perkins critical/subcritical method with
                                  Perry-orifice recovery when RECOVERY='ON'.
                                - SINTEF_HEM: SINTEF-style homogeneous-equilibrium
                                  restricted-flow model for flashing dense/liquid CO2.
                                - SINTEF_DHEM: SINTEF-style delayed HEM model
                                  for pure CO2 with CNT superheat-limit flashing.
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
        :param recovery: Enable/disable Perkins/Perry downstream pressure
                         recovery. Use ON with PERKINS when the downstream
                         pressure is a recovered pipe pressure after the choke,
                         as in a well segment pressure. Use OFF with
                         SINTEF_HEM or SINTEF_DHEM; those models evaluate
                         subcritical flow at the downstream pressure and
                         internally select the critical throat pressure when
                         choked.
        :param recovery_tuning: 1 gives maximum recovery and 0 gives zero recovery
        :param slip_model: Slip model for choke throat. Only NOSLIP is currently usable;
                           CHISHOLM is declared but raises NotImplementedError.
        :param upstream_area: Flow area upstream of the choke [m2]. Defaults to
                              the pipe internal area.
        :param initial_downstream_pressure: Initial downstream pressure [bar],
                                            required when ``diameter`` is omitted
                                            so the choke can be sized from the
                                            target rate.
        :param max_molar_rate: Optional upper bound on the computed molar rate
                               [kmol/day].
        :param verbose: Whether to print boundary construction information.
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

        self.choke_model = build_hydraulic_choke_model(
            hydraulic_model=hydraulic_model,
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

        self.hydraulic_model = self.choke_model.hydraulic_model
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
    def opening(self, value: float) -> None:
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
        property_container: PropertyContainer,
        pipe_internal_area: float,
        molar_rate: float | None = None,
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

    def update_current_molar_rate(self, simulation_time: float) -> None:
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
