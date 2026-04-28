import math
from types import SimpleNamespace

import pytest
from scipy.optimize import brentq

from darts.pipes.upstream_pressure_node_with_choke import PerkinsChokeModel


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
