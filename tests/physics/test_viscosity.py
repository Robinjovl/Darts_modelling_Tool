"""Unit tests for ``AirViscositySutherland`` in ``darts.physics.properties.viscosity``.

Pins Sutherland's law with air constants (mu_ref = 1.716e-5 Pa.s at
T_ref = 273 K, S = 111 K; returned in cP) at two reference temperatures and
verifies the M1b temperature clamp to [50, 3000] K: with the adaptive
multi-index interpolator the supporting-point evaluator can be called far
outside the physical range, and the kernel must return a finite positive
viscosity instead of raising or overflowing.
"""

import numpy as np
import pytest

from darts.physics.properties.viscosity import AirViscositySutherland


@pytest.fixture
def evaluator():
    return AirViscositySutherland()


def test_air_viscosity_at_300_kelvin(evaluator):
    # White (2006) / COMSOL tabulated value for air at 300 K: ~1.846e-5 Pa.s
    mu = evaluator.evaluate(1.0, 300.0, None, None)
    assert mu == pytest.approx(1.8469e-2, rel=1e-3)  # cP


def test_air_viscosity_at_reference_temperature(evaluator):
    # at T = T_ref the law collapses to mu_ref exactly
    mu = evaluator.evaluate(1.0, 273.0, None, None)
    assert mu == pytest.approx(1.716e-2, rel=1e-12)  # cP


def test_pressure_composition_density_are_ignored(evaluator):
    assert evaluator.evaluate(1.0, 300.0, None, None) == evaluator.evaluate(
        500.0, 300.0, [0.5, 0.5], 100.0
    )


@pytest.mark.parametrize("temperature", [-50.0, 0.0, 1e6])
def test_out_of_range_temperature_is_clamped_to_finite_positive(evaluator, temperature):
    mu = evaluator.evaluate(1.0, temperature, None, None)
    assert np.isfinite(mu)
    assert mu > 0.0


def test_clamp_boundaries(evaluator):
    # below 50 K the clamp pins the result to mu(50 K); above 3000 K to mu(3000 K)
    assert evaluator.evaluate(1.0, -50.0, None, None) == evaluator.evaluate(
        1.0, 50.0, None, None
    )
    assert evaluator.evaluate(1.0, 1e6, None, None) == evaluator.evaluate(
        1.0, 3000.0, None, None
    )


def test_monotonic_increase_with_temperature_for_gas(evaluator):
    temperatures = np.array([100.0, 200.0, 300.0, 500.0, 1000.0, 3000.0])
    mus = np.array([evaluator.evaluate(1.0, T, None, None) for T in temperatures])
    assert np.all(np.diff(mus) > 0.0)
