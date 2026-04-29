import math
from types import SimpleNamespace

import pytest
from scipy.optimize import brentq

from darts.pipes.upstream_pressure_node_with_choke import (
    ChokeFlowState,
    PerkinsChokeModel,
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
