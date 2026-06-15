import math
from types import SimpleNamespace

import pytest
from scipy.optimize import brentq

from darts.pipes.upstream_pressure_node_with_choke import (
    ChokeFlowState,
    PerkinsChokeModel,
    SintefDelayedHemChokeModel,
    SintefHemChokeModel,
)


class FakePerkinsChokeModel(PerkinsChokeModel):
    """
    Minimal Perkins model for equation-level validation.

    The real choke model obtains ``fg``, ``alpha1``, ``lambda``, ``n``, and
    upstream specific volume from open-DARTS thermodynamics. These tests inject
    those dimensionless Perkins parameters directly so the algebra in Eq. A-28
    and Eq. A-30 can be validated without building a full physics object.
    """

    def __init__(
        self,
        gas_mass_fraction: float,
        alpha1: float,
        lambda_perkins: float,
        polytropic_exponent: float,
        upstream_specific_volume: float,
        upstream_pressure_bar: float,
        upstream_to_throat_area_ratio: float,
        throat_area: float = 1.0,
        discharge_coefficient: float = 1.0,
    ):
        self._params = (
            gas_mass_fraction,
            alpha1,
            lambda_perkins,
            polytropic_exponent,
            upstream_specific_volume,
        )
        self.boundary_state = SimpleNamespace(pressure=upstream_pressure_bar)
        self.valve_geometry_model = SimpleNamespace(
            throat_area=throat_area,
            discharge_coefficient=discharge_coefficient,
        )
        self.upstream_area = throat_area / upstream_to_throat_area_ratio

    def _perkins_parameters(self, cache):
        return self._params


class FakeSintefHemChokeModel(SintefHemChokeModel):
    """
    Minimal SINTEF HEM model for equation-level validation.

    The fake state follows an incompressible isentropic path,
    ``h0 - h = (p0 - p) / rho``. SINTEF Eq. (8)/(9) should then collapse to
    the standard orifice relation without requiring a full open-DARTS physics
    object.
    """

    _MW_KG_PER_KMOL = 44.0

    def __init__(
        self,
        density: float,
        upstream_pressure_bar: float,
        throat_area: float,
        discharge_coefficient: float,
        h_stagnation_j_kg: float = 1.0e5,
    ):
        self._density = density
        self._h_stagnation_j_kg = h_stagnation_j_kg
        self.helper = SimpleNamespace(
            pressure_bounds=(1.0, upstream_pressure_bar),
            _phase_mw_kg_per_kmol=lambda composition: self._MW_KG_PER_KMOL,
        )
        self.boundary_state = SimpleNamespace(
            pressure=upstream_pressure_bar,
            composition=[1.0],
            molar_enthalpy=self._molar_enthalpy(h_stagnation_j_kg),
        )
        self.valve_geometry_model = SimpleNamespace(
            effective_area=discharge_coefficient * throat_area,
        )

    def _molar_enthalpy(self, h_j_kg: float) -> float:
        return h_j_kg / 1000.0 * self._MW_KG_PER_KMOL

    def _flow_state(self, pressure, cache):
        h_local = (
            self._h_stagnation_j_kg
            - (self.boundary_state.pressure - pressure) * 1e5 / self._density
        )
        return ChokeFlowState(
            pressure=pressure,
            temperature=300.0,
            molar_enthalpy=self._molar_enthalpy(h_local),
            gas_mass_fraction=0.0,
            inv_momentum_density=1.0 / self._density,
            density=self._density,
            gas_density=math.nan,
            liquid_density=self._density,
        )


def test_perkins_pure_gas_critical_pressure_ratio_matches_isentropic_limit():
    """
    For pure gas and negligible upstream velocity, Eq. A-30 must reduce to the
    standard isentropic critical pressure ratio.
    """
    n = 1.4
    model = FakePerkinsChokeModel(
        gas_mass_fraction=1.0,
        alpha1=0.0,
        lambda_perkins=n / (n - 1.0),
        polytropic_exponent=n,
        upstream_specific_volume=1.0,
        upstream_pressure_bar=10.0,
        upstream_to_throat_area_ratio=1e-6,
    )

    critical_ratio = brentq(
        lambda pr: model._perkins_critical_residual(pr, {}),
        0.05,
        0.95,
    )

    expected_ratio = (2.0 / (n + 1.0)) ** (n / (n - 1.0))
    assert critical_ratio == pytest.approx(expected_ratio, abs=1e-12)


def test_perkins_pure_liquid_mass_rate_matches_incompressible_orifice_limit():
    """
    For pure incompressible liquid and negligible upstream velocity, Eq. A-28
    must reduce to ``Cd A sqrt(2 rho delta_p)``.
    """
    density = 1000.0
    upstream_pressure_bar = 100.0
    throat_pressure_bar = 90.0
    throat_area = 0.01
    discharge_coefficient = 0.84
    model = FakePerkinsChokeModel(
        gas_mass_fraction=0.0,
        alpha1=1.0,
        lambda_perkins=0.0,
        polytropic_exponent=1.0,
        upstream_specific_volume=1.0 / density,
        upstream_pressure_bar=upstream_pressure_bar,
        upstream_to_throat_area_ratio=1e-6,
        throat_area=throat_area,
        discharge_coefficient=discharge_coefficient,
    )

    mass_rate = model._perkins_mass_rate_from_pressure(throat_pressure_bar, {})
    expected_rate = (
        discharge_coefficient
        * throat_area
        * math.sqrt(2.0 * density * (upstream_pressure_bar - throat_pressure_bar) * 1e5)
    )

    assert mass_rate == pytest.approx(expected_rate, rel=1e-12)


def test_perkins_a30_root_is_stationary_point_of_a28_for_mixture():
    """
    Eq. A-30 is the derivative condition for Eq. A-28. At an Eq. A-30 root, the
    numerical derivative of Eq. A-28 with respect to pressure ratio should vanish.
    """
    gas_mass_fraction = 0.25
    alpha1 = 0.75
    n = 1.25
    model = FakePerkinsChokeModel(
        gas_mass_fraction=gas_mass_fraction,
        alpha1=alpha1,
        lambda_perkins=gas_mass_fraction * n / (n - 1.0),
        polytropic_exponent=n,
        upstream_specific_volume=0.01,
        upstream_pressure_bar=80.0,
        upstream_to_throat_area_ratio=0.15,
        throat_area=0.003,
        discharge_coefficient=0.826,
    )

    critical_ratio = brentq(
        lambda pr: model._perkins_critical_residual(pr, {}),
        0.05,
        0.95,
    )

    def rate_at_ratio(pressure_ratio):
        return model._perkins_mass_rate_from_pressure(
            pressure_ratio * model.boundary_state.pressure,
            {},
        )

    step = 1e-5
    derivative = (
        rate_at_ratio(critical_ratio + step) - rate_at_ratio(critical_ratio - step)
    ) / (2.0 * step)

    assert derivative == pytest.approx(0.0, abs=1e-6)


def test_perkins_parameters_use_gas_specific_volume_as_a28_reference():
    """
    Perkins Eq. A-28 normalizes liquid volume by upstream gas specific volume.
    """
    gas_mass_fraction = 0.2
    liquid_density = 800.0
    gas_density = 10.0
    liquid_volume_term = (1.0 - gas_mass_fraction) / liquid_density
    gas_specific_volume = 1.0 / gas_density
    model = object.__new__(PerkinsChokeModel)
    model._perkins_parameter_cache = None
    model.boundary_state = SimpleNamespace(pressure=100.0)
    model._estimate_gas_polytropic_exponent = lambda: 1.3
    model._flow_state = lambda pressure, cache: ChokeFlowState(
        pressure=pressure,
        temperature=300.0,
        molar_enthalpy=0.0,
        gas_mass_fraction=gas_mass_fraction,
        inv_momentum_density=gas_mass_fraction / gas_density
        + (1.0 - gas_mass_fraction) / liquid_density,
        density=1.0
        / (
            gas_mass_fraction / gas_density + (1.0 - gas_mass_fraction) / liquid_density
        ),
        gas_density=gas_density,
        liquid_density=liquid_density,
    )

    _, alpha1, _, _, upstream_reference_volume = model._perkins_parameters({})

    assert alpha1 == pytest.approx(liquid_volume_term / gas_specific_volume)
    assert upstream_reference_volume == pytest.approx(gas_specific_volume)


def test_sintef_hem_incompressible_limit_matches_orifice_relation():
    """
    For an incompressible isentropic liquid path, SINTEF Eq. (8)/(9) must reduce
    to ``Cd A sqrt(2 rho delta_p)``.
    """
    density = 1000.0
    upstream_pressure_bar = 100.0
    throat_pressure_bar = 90.0
    throat_area = 0.01
    discharge_coefficient = 0.84
    model = FakeSintefHemChokeModel(
        density=density,
        upstream_pressure_bar=upstream_pressure_bar,
        throat_area=throat_area,
        discharge_coefficient=discharge_coefficient,
    )

    mass_rate = model._mass_rate_from_throat_pressure(throat_pressure_bar, {})
    expected_rate = (
        discharge_coefficient
        * throat_area
        * math.sqrt(2.0 * density * (upstream_pressure_bar - throat_pressure_bar) * 1e5)
    )

    assert mass_rate == pytest.approx(expected_rate, rel=1e-12)


def test_sintef_dhem_rathjen_straub_surface_tension_matches_co2_reference():
    """
    The D-HEM model uses the Rathjen-Straub one-term CO2 correlation.
    """
    model = object.__new__(SintefDelayedHemChokeModel)

    sigma = model._rathjen_straub_surface_tension_n_m(280.0)

    assert sigma == pytest.approx(0.0033074, rel=1e-4)


def test_sintef_dhem_cnt_nucleation_rate_uses_molecule_mass():
    """
    SINTEF Eq. (6) uses molecule mass, not kg/kmol molecular weight.
    """
    model = object.__new__(SintefDelayedHemChokeModel)
    model.boundary_state = SimpleNamespace(composition=[1.0])
    model.helper = SimpleNamespace(_phase_mw_kg_per_kmol=lambda composition: 44.01)
    model._saturation_pressure_bar = lambda temperature: 50.0

    pressure_bar = 30.0
    temperature_k = 280.0
    liquid_density_kg_m3 = 800.0
    log_rate_ratio = model._nucleation_log_rate_ratio(
        pressure_bar,
        temperature_k,
        liquid_density_kg_m3,
    )

    sigma = model._rathjen_straub_surface_tension_n_m(temperature_k)
    molecule_mass = 44.01 / (1000.0 * model._AVOGADRO)
    number_density = liquid_density_kg_m3 / molecule_mass
    delta_p_pa = (50.0 - pressure_bar) * 1e5
    free_energy_barrier = 16.0 * math.pi * sigma**3 / (3.0 * delta_p_pa**2)
    expected = (
        math.log(number_density)
        + 0.5 * math.log(2.0 * sigma / (math.pi * molecule_mass))
        - free_energy_barrier / (model._BOLTZMANN_J_K * temperature_k)
        - math.log(model._JCRIT_PER_M3_S)
    )

    assert log_rate_ratio == pytest.approx(expected, rel=1e-12)
