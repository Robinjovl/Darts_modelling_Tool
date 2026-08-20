"""Unit tests for the drift-flux closures in ``darts.pipes.drift_flux`` (M1b, M2).

These exercise the pure-Python closure layer -- profile parameter (C0) and drift
velocity (vD) for the ``shi_t2well``, ``tang_2019``, ``bai_2023`` and
``bhagwat_ghajar_2014`` models -- without a reservoir, engine assembly, or
property flash.

Since M2 the closures are a strategy family (:class:`DriftFluxClosure`) that
:class:`darts.pipes.pipe.Pipe` selects in its constructor and stores as
``pipe.drift_flux_closure``; the closure owns its own parameters, its theta
convention, its sign convention and its friction model. The tests therefore come
in two flavours: through the real ``Pipe`` constructor with a minimal stub
physics (only the attributes the constructor reads: ``phases``,
``property_containers[0].np_fl/temperature``, ``thermal``, ``vars``,
``n_vars``), and directly against a closure bound to a bare geometry. The
previous-timestep face state the ``update_*`` methods consume
(``iter_phases_props0_face``, ``velocities0``, ``rhoM0_face``) is set by hand,
standing in for what ``eval_phase_vels`` would normally compute.

Covered guards (added on this branch):
  * density inversion (rhoG >= rhoL) -> C0 = 1 / vD = 0, no crash;
  * B&G C0*sG <= 1 - 1e-8 clamp;
  * shi_t2well Cmax extrapolation clamp a2 >= a1 + 0.02 (with warning);
  * B&G-family ValueError for non-vertical geometry;
  * Tang eta clipped to [0, 1] (C0 == A at low gas fraction, monotonic);
  * Tang Ku floored at 1e-6 for vanishing density difference.

Covered structural properties (M2 extraction):
  * closure-irrelevant constructor arguments are rejected, not ignored;
  * every closure returns the FINAL signed drift velocity (no flip in Pipe);
  * each closure owns its friction model (wang_2014 for bai_2023);
  * the Bai/B&G differences live in Bai2023Closure overrides;
  * Pipe and a bare bound closure produce identical C0/vD.
"""

import math
import warnings

import numpy as np
import pytest

# darts.pipes.pipe imports darts.engines (value_vector); skip the module on a
# checkout without the compiled extension (CI builds it before pytest runs).
pytest.importorskip("darts.engines")

from darts.pipes.define_pipe_geometry import PipeGeometry  # noqa: E402
from darts.pipes.drift_flux import (  # noqa: E402
    BHAGWAT_GHAJAR_DRIFT_FLUX_MODELS,
    SUPPORTED_DRIFT_FLUX_MODELS,
    Bai2023Closure,
    BhagwatGhajar2014Closure,
    ColebrookWhiteFriction,
    DriftFluxClosure,
    FaceProps,
    ShiT2WellClosure,
    Tang2019Closure,
    Wang2014Friction,
    make_drift_flux_closure,
)
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


def _closure(pipe):
    """The drift-flux strategy the pipe selected."""
    return pipe.drift_flux_closure


def _make_closure(drift_flux_model="shi_t2well", inclination=0.0, **kwargs):
    """A closure bound to a bare geometry, without any Pipe around it."""
    return make_drift_flux_closure(drift_flux_model, **kwargs).bind(
        _make_geometry(inclination=inclination)
    )


def _face_state(
    n_interfaces=2,
    sG=0.4,
    vM=1.0,
    rhoG=100.0,
    rhoL=900.0,
    muG=1.5e-5,
    muL=5.0e-4,
    vG=None,
    vL=None,
    sigma=0.05,
):
    """A uniform two-phase :class:`FaceProps` state, as ``Pipe`` would build it."""
    ones = np.ones(n_interfaces)
    sG_face = sG * ones
    sL_face = (1.0 - sG) * ones
    vG = vM if vG is None else vG
    vL = vM if vL is None else vL
    rhoM = sG_face * rhoG + sL_face * rhoL
    face = FaceProps.from_pipe_arrays(
        [
            np.tile([0.99, 0.01], (n_interfaces, 1)),
            np.tile([0.02, 0.98], (n_interfaces, 1)),
            sG_face,
            sL_face,
            rhoG * ones,
            rhoL * ones,
            muG * ones,
            muL * ones,
        ],
        np.array([rhoM * vM, vM * ones, vG * ones, vL * ones]),
        rhoM,
    )
    face.ift = sigma * np.ones(len(face.indices))
    return face


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
        assert np.all(pipe.C00 <= _closure(pipe).profile_A + 1e-12)


# ------------------------------------------------- (b) density inversion
def test_bhagwat_ghajar_density_inversion_returns_no_slip():
    closure = _make_closure("bhagwat_ghajar_2014")
    theta = closure.theta[0]
    for rhoG, rhoL in [(800.0, 800.0), (850.0, 800.0)]:  # rhoG >= rhoL
        vD = closure.interface_drift_velocity(
            sL=0.5, j_g=0.1, rhoG=rhoG, rhoL=rhoL, muL=1e-3, sigma=0.03, theta=theta
        )
        assert vD == 0.0
        C0 = closure.interface_profile_parameter(
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
    assert _closure(pipe).a2 == pytest.approx(_closure(pipe).a1 + 0.02)


def test_cmax_within_anchor_range_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe = _make_pipe("shi_t2well", Cmax=1.19)
    assert _closure(pipe).a2 > _closure(pipe).a1 + 0.02


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
    assert np.all(np.isfinite(_closure(pipe).m))


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
        assert np.all(pipe.C00 <= _closure(pipe).profile_A + 1e-12)

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
    assert np.all(np.asarray(_closure(pipe).Ku0_filtered) >= 1e-6)
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
    closure = _make_closure("bhagwat_ghajar_2014")
    assert np.allclose(closure.theta, -math.pi / 2.0)


# ===================================================================== M2 ===
# The strategy extraction: closure selection, ownership and sign convention.
# ============================================================================


# ------------------------------------------- closure selection and registry
@pytest.mark.parametrize(
    ("model", "closure_class"),
    [
        ("shi_t2well", ShiT2WellClosure),
        ("tang_2019", Tang2019Closure),
        ("bai_2023", Bai2023Closure),
        ("bhagwat_ghajar_2014", BhagwatGhajar2014Closure),
    ],
)
def test_pipe_selects_the_matching_closure_class(model, closure_class):
    pipe = _make_pipe(model)
    closure = _closure(pipe)
    assert isinstance(closure, closure_class)
    assert closure.name == model == pipe.drift_flux_model
    assert closure.geometry is pipe.geometry


def test_registry_matches_pipe_class_attributes():
    """The Pipe class attributes are aliases of the drift_flux registry."""
    assert Pipe.supported_drift_flux_models == SUPPORTED_DRIFT_FLUX_MODELS
    assert Pipe.bhagwat_ghajar_drift_flux_models == BHAGWAT_GHAJAR_DRIFT_FLUX_MODELS
    assert set(BHAGWAT_GHAJAR_DRIFT_FLUX_MODELS) <= set(SUPPORTED_DRIFT_FLUX_MODELS)


def test_pipe_accepts_a_closure_instance():
    """The string form is the documented path, but an instance also works."""
    closure = ShiT2WellClosure(Cmax=1.0)
    pipe = _make_pipe(closure)
    assert _closure(pipe) is closure
    assert pipe.drift_flux_model == "shi_t2well"
    assert closure.a2 == 0.21  # the Cmax = 1.0 anchor set, not the 1.2 one
    assert closure.geometry is pipe.geometry


def test_closure_instance_with_closure_keyword_raises():
    with pytest.raises(ValueError, match="closure instance"):
        _make_pipe(ShiT2WellClosure(), Cmax=1.1)


def test_unknown_closure_type_raises():
    with pytest.raises(ValueError, match="drift_flux_model"):
        _make_pipe(3.14)


# ------------------------------- closure-irrelevant arguments are rejected
@pytest.mark.parametrize("model", ["tang_2019", "bai_2023", "bhagwat_ghajar_2014"])
@pytest.mark.parametrize("keyword", ["Cmax", "Fv"])
def test_shi_only_keywords_rejected_by_other_closures(model, keyword):
    """Cmax/Fv used to bind only inside the shi branch and were silently
    ignored elsewhere; now they raise."""
    with pytest.raises(ValueError, match=keyword):
        _make_pipe(model, **{keyword: 1.1})


@pytest.mark.parametrize("model", ["shi_t2well", "bai_2023", "bhagwat_ghajar_2014"])
def test_tang_only_keyword_rejected_by_other_closures(model):
    with pytest.raises(ValueError, match="tang_parameter_set"):
        _make_pipe(model, tang_parameter_set="tuffp")


def test_closure_specific_keywords_accepted_by_their_owner():
    shi = _closure(_make_pipe("shi_t2well", Cmax=1.0, Fv=0.5))
    assert (shi.Cmax, shi.Fv) == (1.0, 0.5)
    tang = _closure(_make_pipe("tang_2019", tang_parameter_set="tuffp"))
    assert tang.parameter_set == "tuffp"
    assert tang.params.A == 1.088


def test_defaults_unchanged_when_keywords_are_omitted():
    """Omitting the keywords must reproduce the historical defaults."""
    shi = _closure(_make_pipe("shi_t2well"))
    assert (shi.Cmax, shi.Fv) == (1.2, 1.0)
    assert shi.profile_A == 1.2
    assert shi.B == pytest.approx(2 / 1.2 - 1.0667)
    assert (shi.a1, shi.a2) == (0.06, 0.12)
    assert _closure(_make_pipe("tang_2019")).parameter_set == "olgas"


# ------------------------------------------------ each closure owns friction
@pytest.mark.parametrize(
    ("model", "friction_class", "friction_name"),
    [
        ("shi_t2well", ColebrookWhiteFriction, "colebrook_white"),
        ("tang_2019", ColebrookWhiteFriction, "colebrook_white"),
        ("bhagwat_ghajar_2014", ColebrookWhiteFriction, "colebrook_white"),
        ("bai_2023", Wang2014Friction, "wang_2014"),
    ],
)
def test_default_friction_model_per_closure(model, friction_class, friction_name):
    pipe = _make_pipe(model)
    assert isinstance(_closure(pipe).friction, friction_class)
    assert pipe.friction_model == friction_name


def test_friction_model_override():
    pipe = _make_pipe("bai_2023", friction_model="colebrook_white")
    assert isinstance(_closure(pipe).friction, ColebrookWhiteFriction)
    assert pipe.friction_model == "colebrook_white"
    assert np.allclose(
        _closure(pipe).fanning_friction_factor(np.array([5.0e4])),
        _closure(_make_pipe("bhagwat_ghajar_2014")).fanning_friction_factor(
            np.array([5.0e4])
        ),
    )


def test_unknown_friction_model_raises():
    with pytest.raises(ValueError, match="friction_model"):
        _make_pipe("shi_t2well", friction_model="not_a_model")


def test_fanning_friction_factor_regimes():
    """Zero flow -> 0, laminar -> 16/Re, turbulent -> above the laminar line
    but O(1e-3), and the two friction models really differ in the turbulent
    regime."""
    Re = np.array([0.0, 1.0e3, 5.0e4])
    colebrook_ff = _closure(_make_pipe("shi_t2well")).fanning_friction_factor(Re)
    wang_ff = _closure(_make_pipe("bai_2023")).fanning_friction_factor(Re)
    for ff in (colebrook_ff, wang_ff):
        assert ff[0] == 0.0
        assert ff[1] == pytest.approx(16.0 / 1.0e3)
        assert 16.0 / 5.0e4 < ff[2] < 0.02
    assert colebrook_ff[2] != wang_ff[2]


# ------------------------------------------------------- sign convention
@pytest.mark.parametrize("model", SUPPORTED_DRIFT_FLUX_MODELS)
def test_drift_velocity_is_final_and_signed(model):
    """Pipe applies no sign of its own any more: the number the closure returns
    is the number Pipe stores."""
    pipe = _make_pipe(model)
    _set_previous_face_state(pipe, sG=0.4, vM=0.5, vG=0.8, vL=0.4)
    pipe.update_profile_parameter()
    pipe.update_drift_velocity()

    face = pipe._face_props(ift=pipe.IFT_face_filtered)
    expected = _closure(pipe).drift_velocity(face, pipe.C00_filtered)
    assert np.array_equal(pipe.vD0, expected)


def test_buoyant_drift_velocity_points_up_the_pipe_frame():
    """The pipe counts positive from the wellhead downwards, so a buoyant gas
    phase drifts in the negative direction for every closure."""
    for model in SUPPORTED_DRIFT_FLUX_MODELS:
        pipe = _make_pipe(model)
        _set_previous_face_state(pipe, sG=0.4, vM=0.5, vG=0.8, vL=0.4)
        pipe.update_profile_parameter()
        pipe.update_drift_velocity()
        assert np.all(pipe.vD0 < 0.0), f"{model}: gas does not drift upwards"


def test_single_phase_faces_give_zero_drift_and_unit_profile_parameter():
    for model in SUPPORTED_DRIFT_FLUX_MODELS:
        pipe = _make_pipe(model)
        _set_previous_face_state(pipe, sG=1.0, vM=0.5)  # sL == 0 everywhere
        pipe.update_profile_parameter()
        pipe.update_drift_velocity()
        assert np.array_equal(pipe.C00, np.ones(pipe.geometry.num_interfaces))
        assert np.all(pipe.vD0 == 0.0)
        assert len(pipe.IFT_face_filtered) == 0


# -------------------------------------- closure works without a Pipe around
@pytest.mark.parametrize("model", SUPPORTED_DRIFT_FLUX_MODELS)
def test_bare_closure_matches_pipe(model):
    """A closure bound to a geometry reproduces exactly what Pipe computes:
    Pipe contributes the state, the closure contributes the physics."""
    pipe = _make_pipe(model)
    _set_previous_face_state(pipe, sG=0.4, vM=0.5, vG=0.8, vL=0.4)
    pipe.update_profile_parameter()
    pipe.update_drift_velocity()

    closure = _make_closure(model)
    face = _face_state(
        n_interfaces=pipe.geometry.num_interfaces, vM=0.5, vG=0.8, vL=0.4
    )
    C0 = closure.profile_parameter(face)
    vD = closure.drift_velocity(face, C0[face.indices])

    assert np.array_equal(C0, pipe.C00)
    assert np.array_equal(vD, pipe.vD0)


@pytest.mark.parametrize("model", SUPPORTED_DRIFT_FLUX_MODELS)
def test_closure_handles_a_face_state_without_two_phase_interfaces(model):
    closure = _make_closure(model)
    face = _face_state(sG=0.0)  # sG == 0 everywhere -> nothing to close
    assert face.indices.size == 0
    C0 = closure.profile_parameter(face)
    assert np.array_equal(C0, np.ones(closure.geometry.num_interfaces))
    assert np.all(closure.drift_velocity(face, C0[face.indices]) == 0.0)


def test_disabled_profile_parameter_forces_unit_c0_into_the_drift_velocity():
    """enable_profile_parameter=False still needs the closure workspace, so the
    profile parameter is evaluated and then discarded."""
    pipe = _make_pipe("shi_t2well", enable_profile_parameter=False)
    _set_previous_face_state(pipe, sG=0.4, vM=0.5, vG=0.8, vL=0.4)
    pipe.update_drift_velocity()
    assert np.array_equal(pipe.C00, np.ones(pipe.geometry.num_interfaces))
    assert np.array_equal(pipe.C00_filtered, np.ones(pipe.geometry.num_interfaces))

    closure = _make_closure("shi_t2well")
    face = _face_state(
        n_interfaces=pipe.geometry.num_interfaces, vM=0.5, vG=0.8, vL=0.4
    )
    closure.profile_parameter(face)
    expected = closure.drift_velocity(face, np.ones(len(face.indices)))
    assert np.array_equal(pipe.vD0, expected)


# ------------------------------------------------------ FaceProps contract
def test_face_props_maps_the_positional_pipe_arrays():
    """The magic positional indices (``[6]`` = muG, ``[7]`` = muL) are decoded
    in exactly one place now."""
    n = 3
    props = [
        np.full((n, 2), float(i)) if i < 2 else np.full(n, float(i)) for i in range(8)
    ]
    velocities = np.array([np.full(n, 10.0 + i) for i in range(4)])
    rhoM = np.full(n, 99.0)
    face = FaceProps.from_pipe_arrays(props, velocities, rhoM)
    assert np.array_equal(face.sG, props[2])
    assert np.array_equal(face.sL, props[3])
    assert np.array_equal(face.rhoG, props[4])
    assert np.array_equal(face.rhoL, props[5])
    assert np.array_equal(face.muG, props[6])
    assert np.array_equal(face.muL, props[7])
    assert np.array_equal(face.rhoM_vM, velocities[0])
    assert np.array_equal(face.vM, velocities[1])
    assert np.array_equal(face.vG, velocities[2])
    assert np.array_equal(face.vL, velocities[3])
    assert np.array_equal(face.rhoM, rhoM)


def test_face_props_indices_select_two_phase_interfaces():
    face = FaceProps.from_pipe_arrays(
        [
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            np.array([0.0, 0.5, 1.0]),  # sG
            np.array([1.0, 0.5, 0.0]),  # sL
            np.full(3, 100.0),
            np.full(3, 900.0),
            np.full(3, 1.5e-5),
            np.full(3, 5.0e-4),
        ],
        np.zeros((4, 3)),
        np.full(3, 500.0),
    )
    assert np.array_equal(face.indices, np.array([1]))


# ------------------------------------------ Bai vs Bhagwat-Ghajar differences
def test_bai_overrides_are_the_only_difference_to_bhagwat_ghajar():
    """The four internal re-branches on ``self.drift_flux_model`` are now four
    overridden methods on Bai2023Closure."""
    bai = _make_closure("bai_2023")
    bng = _make_closure("bhagwat_ghajar_2014")
    overridden = {
        "c4_downward_low_froude_condition",
        "profile_reynolds_number",
        "laplace_factor",
        "liquid_holdup_factor",
    }
    for name in overridden:
        assert getattr(type(bai), name) is not getattr(type(bng), name), name
    shared = set(dir(bng)) - overridden - {"name", "default_friction_model"}
    for name in sorted(shared):
        if name.startswith("__"):
            continue
        bai_attr = getattr(type(bai), name, None)
        if callable(bai_attr):
            assert bai_attr is getattr(type(bng), name), name


def test_bai_laplace_branch_is_inverted():
    bai, bng = _make_closure("bai_2023"), _make_closure("bhagwat_ghajar_2014")
    small_la, large_la = 0.01, 0.1
    assert bai.laplace_factor(small_la) == 1.0
    assert bai.laplace_factor(large_la) == pytest.approx((large_la / 0.025) ** 0.9)
    assert bng.laplace_factor(large_la) == 1.0
    assert bng.laplace_factor(small_la) == pytest.approx((small_la / 0.025) ** 0.9)


def test_bai_liquid_holdup_factor_is_not_square_rooted():
    bai, bng = _make_closure("bai_2023"), _make_closure("bhagwat_ghajar_2014")
    assert bai.liquid_holdup_factor(0.25) == 0.25
    assert bng.liquid_holdup_factor(0.25) == 0.5


def test_bai_c4_condition_includes_horizontal_flow():
    bai, bng = _make_closure("bai_2023"), _make_closure("bhagwat_ghajar_2014")
    assert bai.c4_downward_low_froude_condition(0.0, 0.05) is True
    assert bng.c4_downward_low_froude_condition(0.0, 0.05) is False
    for closure in (bai, bng):
        assert closure.c4_downward_low_froude_condition(-10.0, 0.05) is True
        assert closure.c4_downward_low_froude_condition(-10.0, 0.5) is False


def test_bai_profile_reynolds_number_uses_the_mixture_properties():
    bai, bng = _make_closure("bai_2023"), _make_closure("bhagwat_ghajar_2014")
    kwargs = dict(
        sG=0.4,
        sL=0.6,
        mixture_velocity=1.0,
        rhoG=100.0,
        rhoL=900.0,
        muG=1.5e-5,
        muL=5e-4,
    )
    rho_m = 0.4 * 100.0 + 0.6 * 900.0
    mu_m = 0.4 * 1.5e-5 + 0.6 * 5e-4
    assert bai.profile_reynolds_number(**kwargs) == pytest.approx(
        rho_m * 1.0 * 0.1 / mu_m
    )
    assert bng.profile_reynolds_number(**kwargs) == pytest.approx(
        900.0 * 1.0 * 0.1 / 5e-4
    )


def test_bai_and_bhagwat_ghajar_give_different_closures():
    """A guard against the two family members collapsing into one."""
    pipes = {}
    for model in BHAGWAT_GHAJAR_DRIFT_FLUX_MODELS:
        pipe = _make_pipe(model)
        _set_previous_face_state(pipe, sG=0.4, vM=0.5, vG=0.8, vL=0.4)
        pipe.update_profile_parameter()
        pipe.update_drift_velocity()
        pipes[model] = (pipe.C00.copy(), pipe.vD0.copy())
    (c0_bai, vd_bai), (c0_bng, vd_bng) = pipes["bai_2023"], pipes["bhagwat_ghajar_2014"]
    assert not np.allclose(c0_bai, c0_bng)
    assert not np.allclose(vd_bai, vd_bng)


# ---------------------------------------------------------- base-class API
def test_base_class_methods_are_abstract_in_spirit():
    class _Incomplete(DriftFluxClosure):
        name = "incomplete"

    closure = _Incomplete().bind(_make_geometry())
    face = _face_state()
    with pytest.raises(NotImplementedError):
        closure.profile_parameter(face)
    with pytest.raises(NotImplementedError):
        closure.drift_velocity(face, np.ones(len(face.indices)))


def test_closure_instance_cannot_be_shared_between_pipes():
    """A closure caches per-interface parameters of its geometry, so sharing one
    instance between pipes is refused instead of silently re-pointing it."""
    closure = ShiT2WellClosure()
    pipe = _make_pipe(closure)
    with pytest.raises(ValueError, match="already bound"):
        _make_pipe(closure)
    assert _closure(pipe) is closure
    assert closure.geometry is pipe.geometry


def test_rebinding_the_same_geometry_is_allowed():
    geometry = _make_geometry()
    closure = Tang2019Closure().bind(geometry)
    assert closure.bind(geometry) is closure
