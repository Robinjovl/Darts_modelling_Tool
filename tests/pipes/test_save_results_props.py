"""Unit tests for ``_get_requested_well_properties`` in ``darts.pipes.save_results``.

The M1b change under test: the requested-property parser re-infers the
``include_*`` flags from the requested names -- ``"z"`` enables the overall
composition, ``"vG"``/``"vL"`` enable the phase velocities, and any
``phase_{molar|mass|volumetric}_rate_{phase}`` name enables the phase rates --
and unknown names raise a ``KeyError``. The coupled model is a pure-Python
stub exposing only what the function reads (``physics.vars`` and
``physics.property_containers[0].output_props`` / ``phases_name``).
"""

import pytest

# save_results imports darts.models.darts_model, which needs the compiled
# extension; skip the module on a checkout without it.
pytest.importorskip("darts.engines")

from darts.pipes.save_results import _get_requested_well_properties  # noqa: E402


class _StubPropertyContainer:
    def __init__(self):
        # only the keys are read (secondary property names)
        self.output_props = {"temperature": None, "sG": None, "rhoG": None}
        self.phases_name = ["G", "L"]


class _StubPhysics:
    def __init__(self):
        self.vars = ["pressure", "CO2"]
        self.property_containers = [_StubPropertyContainer()]


class _StubModel:
    def __init__(self):
        self.physics = _StubPhysics()


def _call(output_properties, overall=False, velocities=False, rates=False):
    return _get_requested_well_properties(
        _StubModel(), output_properties, overall, velocities, rates
    )


def test_requesting_z_enables_overall_composition():
    _, _, overall, velocities, rates = _call(["z"])
    assert overall is True
    assert velocities is False
    assert rates is False


@pytest.mark.parametrize("velocity_name", ["vG", "vL"])
def test_requesting_phase_velocity_enables_phase_velocities(velocity_name):
    _, _, overall, velocities, rates = _call([velocity_name])
    assert velocities is True
    assert overall is False
    assert rates is False


@pytest.mark.parametrize(
    "rate_name",
    ["phase_molar_rate_G", "phase_mass_rate_L", "phase_volumetric_rate_G"],
)
def test_requesting_phase_rate_enables_phase_rates(rate_name):
    _, _, overall, velocities, rates = _call([rate_name])
    assert rates is True
    assert overall is False
    assert velocities is False


def test_explicitly_enabled_flags_are_kept():
    _, _, overall, velocities, rates = _call(
        ["pressure"], overall=True, velocities=True, rates=True
    )
    assert overall is True and velocities is True and rates is True


def test_unknown_property_name_raises_key_error():
    with pytest.raises(KeyError, match="not_a_property"):
        _call(["not_a_property"])


def test_unknown_phase_in_rate_name_raises_key_error():
    # only rates of the declared phases (G, L) are known
    with pytest.raises(KeyError, match="phase_mass_rate_W"):
        _call(["phase_mass_rate_W"])


def test_non_string_entry_raises_type_error():
    with pytest.raises(TypeError, match="strings"):
        _call(["pressure", 42])


def test_single_string_request_is_accepted():
    _, _, overall, _, _ = _call("z")
    assert overall is True


def test_primary_and_secondary_props_are_split():
    primary_idxs, secondary, overall, velocities, rates = _call(
        ["pressure", "sG", "CO2"]
    )
    assert primary_idxs == {"pressure": 0, "CO2": 1}
    assert secondary == ["sG"]
    assert overall is False and velocities is False and rates is False


def test_default_request_takes_all_props_and_appends_temperature():
    primary_idxs, secondary, overall, velocities, rates = _call(None)
    # all primary variables plus all secondary output props are requested
    assert primary_idxs == {"pressure": 0, "CO2": 1}
    assert set(secondary) == {"temperature", "sG", "rhoG"}
    # flags stay as passed (no name-based inference triggered)
    assert overall is False and velocities is False and rates is False
