"""Unit tests for the drift-flux closures in ``darts.pipes.pipe`` (M1b).

These exercise the pure-Python closure layer of :class:`darts.pipes.pipe.Pipe`
-- profile parameter (C0) and drift velocity (vD) for the ``shi_t2well``,
``tang_2019``, ``bai_2023`` and ``bhagwat_ghajar_2014`` models -- without a
reservoir, engine assembly, or property flash. Pipes are built through the real
constructor with a minimal stub physics (only the attributes the constructor
reads: ``phases``, ``property_containers[0].np_fl/temperature``, ``thermal``,
``vars``, ``n_vars``); the previous-timestep face state the ``update_*``
methods consume (``iter_phases_props0_face``, ``velocities0``, ``rhoM0_face``)
is set by hand, standing in for what ``eval_phase_vels`` would normally
compute.

Covered guards (added on this branch):
  * density inversion (rhoG >= rhoL) -> C0 = 1 / vD = 0, no crash;
  * B&G C0*sG <= 1 - 1e-8 clamp;
  * shi_t2well Cmax extrapolation clamp a2 >= a1 + 0.02 (with warning);
  * B&G-family ValueError for non-vertical geometry;
  * Tang eta clipped to [0, 1] (C0 == A at low gas fraction, monotonic);
  * Tang Ku floored at 1e-6 for vanishing density difference.
"""

import math
import warnings

import numpy as np
import pytest

# darts.pipes.pipe imports darts.engines (value_vector); skip the module on a
# checkout without the compiled extension (CI builds it before pytest runs).
pytest.importorskip("darts.engines")

from darts.pipes.define_pipe_geometry import PipeGeometry  # noqa: E402
from darts.pipes.pipe import Pipe  # noqa: E402


# ------------------------------------------------------------- minimal stubs
class _StubIFT:
    """Constant gas-liquid interfacial tension [N/m]."""

    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def evaluate(self, rhoG, rhoL, xG_mass, xL_mass):
        return self.sigma


class _StubPropertyContainer:
    np_fl = 2  # two fluid phases: G + L

    def __init__(self):
        self.temperature = 313.15  # isothermal Pipe requires a temperature
        self.IFT_ev = _StubIFT()


class _StubPhysics:
    """Exactly the attribute surface Pipe.__init__ reads."""

    def __init__(self):
        self.phases = ["G", "L"]
        self.property_containers = [_StubPropertyContainer()]
        self.thermal = False
        self.vars = ["pressure", "CO2"]
        self.n_vars = 2


def _make_geometry(inclination=0.0, n_segments=3, name="W1"):
    return PipeGeometry(
        pipe_name=name,
        segment_lengths=[10.0] * n_segments,
        pipe_ID=0.1,
        inclination_angle=inclination,
        wall_roughness=5e-5,
    )


def _make_pipe(drift_flux_model="shi_t2well", inclination=0.0, **kwargs):
    geometry = _make_geometry(inclination=inclination)
    return Pipe(
        pipe_name=geometry.pipe_name,
        pipe_geometry=geometry,
        physics=_StubPhysics(),
        reservoir=None,
        initial_conditions=None,
        drift_flux_model=drift_flux_model,
        **kwargs,
    )


def _set_previous_face_state(
    pipe,
    sG=0.4,
    vM=1.0,
    rhoG=100.0,
    rhoL=900.0,
    muG=1.5e-5,
    muL=5.0e-4,
    vG=None,
    vL=None,
):
    """Hand-set the previous-timestep interface state ``update_profile_parameter``
    and ``update_drift_velocity`` read (uniform two-phase state on every face)."""
    n_i = pipe.geometry.num_interfaces
    ones = np.ones(n_i)
    sG_face = sG * ones
    sL_face = (1.0 - sG) * ones
    xG_mass = np.tile([0.99, 0.01], (n_i, 1))
    xL_mass = np.tile([0.02, 0.98], (n_i, 1))
    pipe.iter_phases_props0_face = [
        xG_mass,
        xL_mass,
        sG_face,
        sL_face,
        rhoG * ones,
        rhoL * ones,
        muG * ones,
        muL * ones,
    ]
    pipe.rhoM0_face = sG_face * rhoG + sL_face * rhoL
    vG = vM if vG is None else vG
    vL = vM if vL is None else vL
    pipe.velocities0 = np.array([pipe.rhoM0_face * vM, vM * ones, vG * ones, vL * ones])


# ------------------------------------------- (a) finite values, all closures
@pytest.mark.parametrize("model", Pipe.supported_drift_flux_models)
@pytest.mark.parametrize("sG", [0.15, 0.5, 0.85])
def test_profile_parameter_and_drift_velocity_finite(model, sG):
    pipe = _make_pipe(model)
    _set_previous_face_state(pipe, sG=sG, vM=0.5, vG=0.8, vL=0.4)

    pipe.update_profile_parameter()
    pipe.update_drift_velocity()

    assert np.all(np.isfinite(pipe.C00)), f"{model}: non-finite C0"
    assert np.all(pipe.C00 > 0.0), f"{model}: non-positive C0"
    assert np.all(np.isfinite(pipe.vD0)), f"{model}: non-finite vD"
    if model in ("shi_t2well", "tang_2019"):
        # C0 = A / (1 + (A - 1) * eta^2) with eta in [0, 1] cannot exceed A
        assert np.all(pipe.C00 <= pipe.profile_A + 1e-12)


# ------------------------------------------------- (b) density inversion
def test_bhagwat_ghajar_density_inversion_returns_no_slip():
    pipe = _make_pipe("bhagwat_ghajar_2014")
    theta = pipe.bhagwat_ghajar_theta[0]
    for rhoG, rhoL in [(800.0, 800.0), (850.0, 800.0)]:  # rhoG >= rhoL
        vD = pipe.bhagwat_ghajar_drift_velocity(
            sL=0.5, j_g=0.1, rhoG=rhoG, rhoL=rhoL, muL=1e-3, sigma=0.03, theta=theta
        )
        assert vD == 0.0
        C0 = pipe.bhagwat_ghajar_profile_parameter(
            sG=0.5,
            sL=0.5,
            j_g=0.1,
            j_l=0.1,
            mixture_velocity=0.2,
            rhoG=rhoG,
            rhoL=rhoL,
            muG=1.5e-5,
            muL=1e-3,
            theta=theta,
        )
        assert C0 == 1.0


def test_update_paths_handle_density_inversion_without_crash():
    """Merging phase densities on the faces -> homogeneous no-slip limit."""
    for model in Pipe.supported_drift_flux_models:
        pipe = _make_pipe(model)
        _set_previous_face_state(pipe, sG=0.4, vM=0.5, rhoG=800.0, rhoL=800.0)
        pipe.update_profile_parameter()
        pipe.update_drift_velocity()
        assert np.allclose(pipe.C00, 1.0), f"{model}: no-slip C0 != 1"
        assert np.allclose(pipe.vD0, 0.0), f"{model}: no-slip vD != 0"


# ------------------------------------------------------- (c) Cmax a2 clamp
def test_cmax_extrapolation_clamps_a2_with_warning():
    with pytest.warns(UserWarning, match="Clamping a2"):
        pipe = _make_pipe("shi_t2well", Cmax=1.4)
    assert pipe.a2 == pytest.approx(pipe.a1 + 0.02)


def test_cmax_within_anchor_range_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe = _make_pipe("shi_t2well", Cmax=1.19)
    assert pipe.a2 > pipe.a1 + 0.02


@pytest.mark.parametrize("bad_cmax", [1.6, 0.9])
def test_cmax_out_of_range_raises(bad_cmax):
    with pytest.raises(ValueError, match="Cmax"):
        _make_pipe("shi_t2well", Cmax=bad_cmax)


# ------------------------------------------- (d) B&G vertical-only geometry
@pytest.mark.parametrize("model", Pipe.bhagwat_ghajar_drift_flux_models)
def test_bhagwat_ghajar_family_rejects_inclined_geometry(model):
    with pytest.raises(ValueError, match="vertical"):
        _make_pipe(model, inclination=30.0)


@pytest.mark.parametrize("model", Pipe.bhagwat_ghajar_drift_flux_models)
def test_bhagwat_ghajar_family_accepts_vertical_geometry(model):
    pipe = _make_pipe(model, inclination=0.0)
    assert pipe.drift_flux_model == model


def test_shi_accepts_inclined_geometry():
    pipe = _make_pipe("shi_t2well", inclination=30.0)
    assert np.all(np.isfinite(pipe.m))


# ------------------------------------------------- B&G C0*sG < 1 clamp
def test_bhagwat_ghajar_c0_sg_clamp_at_high_gas_fraction():
    """At low Re the raw B&G C0 -> 2; with sG near 1 the drift-flux density
    C0*sG*rhoG + (1 - C0*sG)*rhoL would turn unphysical, so update_profile_
    parameter clamps C0*sG <= 1 - 1e-8."""
    pipe = _make_pipe("bhagwat_ghajar_2014")
    sG = 0.995
    _set_previous_face_state(pipe, sG=sG, vM=1e-3)
    pipe.update_profile_parameter()
    assert np.all(pipe.C00 * sG <= 1.0 - 1e-8 + 1e-14)
    assert np.allclose(pipe.C00, (1.0 - 1e-8) / sG)


# --------------------------------------------- (e) Tang eta clip / monotonic
def test_tang_tuffp_c0_equals_profile_a_at_low_gas_fraction():
    """beta below B gives a negative raw eta; the clip to [0, 1] must yield
    C0 == A exactly (not a C0 > A from a negative eta growing the
    denominator), and C0 must be non-increasing in the gas fraction."""
    c0_by_sg = {}
    for sG in (0.05, 0.5, 0.95):
        pipe = _make_pipe("tang_2019", tang_parameter_set="tuffp")
        _set_previous_face_state(pipe, sG=sG, vM=1e-3)
        pipe.update_profile_parameter()
        c0_by_sg[sG] = pipe.C00.copy()
        assert np.all(pipe.C00 <= pipe.profile_A + 1e-12)

    profile_a = 1.088  # tuffp parameterization
    # sG = 0.05 and 0.5 are both below B = 0.833 -> eta clipped to 0 -> C0 = A
    assert np.allclose(c0_by_sg[0.05], profile_a)
    assert np.allclose(c0_by_sg[0.5], profile_a)
    # sG = 0.95 is above B -> eta in (0, 1] -> C0 strictly below A
    assert np.all(c0_by_sg[0.95] < profile_a)
    assert np.all(c0_by_sg[0.5] >= c0_by_sg[0.95])


def test_tang_ku_floor_for_vanishing_density_difference():
    """A tiny (rhoL - rhoG) drives the raw Tang Kutateladze number negative;
    it must be floored at 1e-6 and the closure must stay finite (v_sgf0
    division guards)."""
    pipe = _make_pipe("tang_2019")
    _set_previous_face_state(pipe, sG=0.4, vM=0.5, rhoG=500.0, rhoL=500.0 + 1e-6)
    pipe.update_profile_parameter()
    pipe.update_drift_velocity()
    assert np.all(np.asarray(pipe.Ku0_filtered) >= 1e-6)
    assert np.all(np.isfinite(pipe.C00))
    assert np.all(np.isfinite(pipe.vD0))


# -------------------------------------------------- constructor validation
def test_unknown_drift_flux_model_raises():
    with pytest.raises(ValueError, match="drift_flux_model"):
        _make_pipe("not_a_model")


def test_unknown_tang_parameter_set_raises():
    with pytest.raises(ValueError, match="tang_parameter_set"):
        _make_pipe("tang_2019", tang_parameter_set="bogus")


def test_deg_to_theta_convention():
    """B&G/Tang theta is measured from horizontal: vertical pipe -> -pi/2."""
    pipe = _make_pipe("bhagwat_ghajar_2014")
    assert np.allclose(pipe.bhagwat_ghajar_theta, -math.pi / 2.0)
