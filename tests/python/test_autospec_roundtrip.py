"""Round-trip regression tests for :mod:`darts.api.autospec`.

For each model in ``models/json_test_suite.JSON_MODELS``, builds the model
from JSON, has autospec record a ``ModelSpec``-shaped dict, and asserts
two layers of invariants:

1. **Emission soundness** (single test, all models): autospec writes a
   well-formed JSON file containing top-level ``reservoir``, ``physics``,
   and ``wells`` sections. This is the bar for wiring autospec into
   ``run_json_model`` (each run drops a ``resolved_spec.json``).

2. **Field-level fidelity** (parametrized per-field): the emitted JSON
   preserves specific structural fields from the input — reservoir grid
   dimensions, physics components, well count. Any failure here documents
   a real autospec coverage gap.

Tests that surface gaps which the autospec maintainer hasn't addressed
yet are marked ``xfail(strict=False)`` with a reason — they will start
passing automatically as autospec coverage improves.
"""

from __future__ import annotations

import json
import os

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODELS_ROOT = os.path.join(_REPO_ROOT, "models")

# Models that successfully reach ``model.init()`` under autospec recording.
# 2ph_comp_solid is currently excluded because autospec's monkey-patches
# interact with the solid-phase ConstFunc density evaluator in a way that
# breaks ``property_container.evaluate_at_cond`` during init (temperature
# field receives a ConstFuncConfig instance instead of a float). Fixing
# requires changes inside autospec's evaluator-recording wrapper; tracked
# as a follow-up.
JSON_MODELS = [
    ("2ph_comp", "2ph_comp.json"),
    ("2ph_do", "2ph_do.json"),
    ("2ph_do_thermal", "2ph_do_thermal.json"),
    ("3ph_bo", "3ph_bo.json"),
    ("3ph_comp_w", "3ph_comp_w.json"),
    ("3ph_do", "3ph_do.json"),
]


@pytest.fixture
def autospec_session():
    """Enable autospec recording for the test, restore + reset state after.

    Also snapshots and restores ``TYPE_REGISTRY``. Multiple reference models
    register plugins under the same ``type_id`` (e.g. ``pc/ModelProperties@v1``)
    with model-specific constructor paths; without isolation the second
    model silently uses the first model's constructor (registration is
    no-op-on-duplicate in :func:`load_local_plugin_registry`) and Python
    explodes deep inside the wrong ``model.py``.
    """
    from darts.api import autospec
    from darts.api.type_registry import TYPE_REGISTRY

    autospec.enable_autorecording()
    autospec._STATE.spec = {}
    autospec._STATE.wctrl_to_well = {}
    autospec._STATE.pc_by_id = {}
    autospec._STATE.plugin_by_id = {}

    saved_by_id = dict(TYPE_REGISTRY._by_id)
    saved_by_kind = {k: list(v) for k, v in TYPE_REGISTRY._by_kind.items()}
    try:
        yield autospec
    finally:
        autospec.disable_autorecording()
        TYPE_REGISTRY._by_id = saved_by_id
        TYPE_REGISTRY._by_kind = saved_by_kind


def _build_and_init(json_path: str) -> tuple:
    """Construct + init a JsonModel from a JSON spec; returns (model, spec_dict)."""
    from darts.api import ModelBuilder
    from darts.api.json_model import JsonModel
    from darts.models.cicd_model import CICDModel

    class JsonCICDModel(JsonModel, CICDModel):
        pass

    cwd = os.getcwd()
    model_dir = os.path.dirname(json_path)
    try:
        os.chdir(model_dir)
        with open(json_path) as fp:
            spec_dict = json.load(fp)
        model = JsonCICDModel()
        ModelBuilder.apply_dict(spec_dict, model, base_path=model_dir)
        model.init()
        return model, spec_dict
    finally:
        os.chdir(cwd)


def _emit(autospec_mod, tmp_path) -> dict:
    """Emit the recorded spec and return the loaded JSON dict."""
    out = tmp_path / "resolved_spec.json"
    autospec_mod.emit_json(str(out))
    assert out.exists() and out.stat().st_size > 0, (
        "autospec did not write a non-empty resolved spec file"
    )
    with open(out) as fp:
        return json.load(fp)


@pytest.mark.parametrize(
    ("model_dir", "json_name"),
    JSON_MODELS,
    ids=[m[0] for m in JSON_MODELS],
)
def test_autospec_emits_well_formed_json(
    autospec_session, model_dir: str, json_name: str, tmp_path
) -> None:
    """Build → autospec emit → file is valid JSON with the expected sections.

    This is the minimum bar for wiring autospec into the runtime: every
    run should produce a parseable file with at least the reservoir,
    physics, and wells top-level keys.
    """
    json_path = os.path.join(_MODELS_ROOT, model_dir, json_name)
    if not os.path.exists(json_path):
        pytest.skip(f"Model spec missing on disk: {json_path}")

    _build_and_init(json_path)
    emitted = _emit(autospec_session, tmp_path)

    for required_section in ("reservoir", "physics", "wells"):
        assert required_section in emitted, (
            f"Emitted spec missing top-level section: {required_section}"
        )


@pytest.mark.parametrize(
    ("model_dir", "json_name"),
    JSON_MODELS,
    ids=[m[0] for m in JSON_MODELS],
)
def test_autospec_preserves_reservoir_grid(
    autospec_session, model_dir: str, json_name: str, tmp_path
) -> None:
    """Reservoir nx/ny/nz survive the round-trip."""
    from darts.api.presets import resolve_section_presets

    json_path = os.path.join(_MODELS_ROOT, model_dir, json_name)
    if not os.path.exists(json_path):
        pytest.skip(f"Model spec missing on disk: {json_path}")

    _, input_spec = _build_and_init(json_path)
    emitted = _emit(autospec_session, tmp_path)

    in_res = resolve_section_presets(input_spec).get("reservoir") or {}
    out_res = emitted.get("reservoir") or {}
    if in_res.get("type") != "structured":
        pytest.skip("Only structured reservoirs check nx/ny/nz here")
    for axis in ("nx", "ny", "nz"):
        assert out_res.get(axis) == in_res.get(axis), (
            f"reservoir.{axis}: input={in_res.get(axis)} emitted={out_res.get(axis)}"
        )


@pytest.mark.parametrize(
    ("model_dir", "json_name"),
    JSON_MODELS,
    ids=[m[0] for m in JSON_MODELS],
)
def test_autospec_preserves_physics_components(
    autospec_session, model_dir: str, json_name: str, tmp_path
) -> None:
    """``physics.components`` array survives the round-trip."""
    from darts.api.presets import resolve_section_presets

    json_path = os.path.join(_MODELS_ROOT, model_dir, json_name)
    if not os.path.exists(json_path):
        pytest.skip(f"Model spec missing on disk: {json_path}")

    _, input_spec = _build_and_init(json_path)
    emitted = _emit(autospec_session, tmp_path)

    in_comp = (resolve_section_presets(input_spec).get("physics") or {}).get(
        "components"
    )
    out_comp = (emitted.get("physics") or {}).get("components")
    assert in_comp is not None, "Input spec has no physics.components"
    assert out_comp == in_comp, (
        f"physics.components: input={in_comp} emitted={out_comp}"
    )


@pytest.mark.parametrize(
    ("model_dir", "json_name"),
    JSON_MODELS,
    ids=[m[0] for m in JSON_MODELS],
)
@pytest.mark.xfail(
    strict=False,
    reason=(
        "autospec currently records well names but not perforations — emitted "
        "wells.wells[*].perforations is [] which fails Strict validation. "
        "Tracked as an autospec coverage gap to be fixed in a follow-up."
    ),
)
def test_autospec_preserves_well_perforations(
    autospec_session, model_dir: str, json_name: str, tmp_path
) -> None:
    """Each emitted well must carry at least one perforation."""
    json_path = os.path.join(_MODELS_ROOT, model_dir, json_name)
    if not os.path.exists(json_path):
        pytest.skip(f"Model spec missing on disk: {json_path}")

    _build_and_init(json_path)
    emitted = _emit(autospec_session, tmp_path)
    wells = (emitted.get("wells") or {}).get("wells") or []
    assert wells, "No wells recorded"
    for w in wells:
        perfs = w.get("perforations") or []
        assert perfs, f"Well {w.get('name')!r} has no perforations in emission"
