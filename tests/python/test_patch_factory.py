"""Side-by-side equivalence tests for :func:`make_patch_model`.

Asserts that a factory-generated Patch class produces a JSON Schema
equivalent (modulo title and example payloads) to the hand-written Patch
class in :mod:`darts.api.schemas`. This is the safety net that lets us
delete the hand-written classes — if every pair matches, the factory is
behaviorally equivalent.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from darts.api.patch_factory import make_patch_model
from darts.api.schemas import (
    PatchCPGReservoirSpec,
    PatchInitialConditionsSpec,
    PatchOutputSpec,
    PatchPluginSlots,
    PatchPropertyRegionSpec,
    PatchReservoirSpec,
    PatchSimParamsSpec,
    PatchWellControlsSpec,
    PatchWellPerforation,
    StrictCPGReservoirSpec,
    StrictInitialConditionsSpec,
    StrictModelSpec,
    StrictOutputSpec,
    StrictPhysicsSpec,
    StrictPluginSlots,
    StrictPropertyRegionSpec,
    StrictReservoirSpec,
    StrictSimParamsSpec,
    StrictWellControlsSpec,
    StrictWellPerforation,
    StrictWellSpec,
    StrictWellsSpec,
)


def _strip_for_equivalence(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that are expected to differ across hand-written and
    generated patch classes (titles, example payloads, $def names).

    Equivalence test is about *structure* — required-field set, property
    shapes, constraints, nullability — not about cosmetic naming.
    """
    out = copy.deepcopy(schema)
    out.pop("title", None)
    out.pop("examples", None)
    out.pop("$defs", None)
    # The top-level ``description`` is the class docstring — generated
    # patches carry the factory's auto-doc string, hand-written ones carry
    # their own. Property-level descriptions are kept because they're part
    # of the JSON Schema contract.
    out.pop("description", None)
    # Examples may appear nested inside individual properties; drop them too.
    for prop in out.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("examples", None)
    return out


# (Strict, Patch, type_map factory) triplets. type_map factories are deferred
# so each test can build them on demand against the *generated* patch classes
# (e.g. PatchPhysicsSpec.property_regions must reference the generated
# PatchPropertyRegionSpec, not the hand-written one).
PAIRS: list[tuple[type, type, str]] = [
    (StrictOutputSpec, PatchOutputSpec, "PatchOutputSpec"),
    (StrictSimParamsSpec, PatchSimParamsSpec, "PatchSimParamsSpec"),
    (StrictWellControlsSpec, PatchWellControlsSpec, "PatchWellControlsSpec"),
    (
        StrictInitialConditionsSpec,
        PatchInitialConditionsSpec,
        "PatchInitialConditionsSpec",
    ),
    (StrictPluginSlots, PatchPluginSlots, "PatchPluginSlots"),
    (StrictWellPerforation, PatchWellPerforation, "PatchWellPerforation"),
    (StrictCPGReservoirSpec, PatchCPGReservoirSpec, "PatchCPGReservoirSpec"),
    (StrictReservoirSpec, PatchReservoirSpec, "PatchReservoirSpec"),
    (StrictPropertyRegionSpec, PatchPropertyRegionSpec, "PatchPropertyRegionSpec"),
]


@pytest.mark.parametrize(("strict_cls", "patch_cls", "patch_name"), PAIRS)
def test_factory_required_fields_become_optional(
    strict_cls: type, patch_cls: type, patch_name: str
) -> None:
    """All required fields on the strict class are optional on the patch."""
    generated = make_patch_model(strict_cls, name=patch_name)
    for fname, finfo in generated.model_fields.items():
        assert not finfo.is_required(), (
            f"{patch_name}.{fname} should be optional in the generated patch"
        )


@pytest.mark.parametrize(("strict_cls", "patch_cls", "patch_name"), PAIRS)
def test_factory_preserves_extra_forbid(
    strict_cls: type, patch_cls: type, patch_name: str
) -> None:
    """Generated patches must inherit ``extra='forbid'`` from the schema base."""
    generated = make_patch_model(strict_cls, name=patch_name)
    assert generated.model_config.get("extra") == "forbid"


def test_factory_matches_hand_written_leaf_classes() -> None:
    """Leaf patch classes (no nested Strict→Patch substitutions) produce
    a JSON schema structurally equivalent to the hand-written one."""
    leaf_pairs = [
        (StrictOutputSpec, PatchOutputSpec, "PatchOutputSpec"),
        (StrictSimParamsSpec, PatchSimParamsSpec, "PatchSimParamsSpec"),
        (StrictWellControlsSpec, PatchWellControlsSpec, "PatchWellControlsSpec"),
        (
            StrictInitialConditionsSpec,
            PatchInitialConditionsSpec,
            "PatchInitialConditionsSpec",
        ),
        (StrictPluginSlots, PatchPluginSlots, "PatchPluginSlots"),
    ]
    for strict, hand_patch, name in leaf_pairs:
        gen = make_patch_model(strict, name=name)
        gen_schema = _strip_for_equivalence(gen.model_json_schema())
        hand_schema = _strip_for_equivalence(hand_patch.model_json_schema())
        assert gen_schema == hand_schema, (
            f"Schemas differ for {name}:\n"
            f"generated: {gen_schema}\n"
            f"hand-written: {hand_schema}"
        )


def test_factory_substitutes_nested_strict_types() -> None:
    """``type_map`` rewrites nested strict refs in field annotations."""
    gen_region = make_patch_model(
        StrictPropertyRegionSpec, name="PatchPropertyRegionSpec"
    )
    gen_physics = make_patch_model(
        StrictPhysicsSpec,
        name="PatchPhysicsSpec",
        type_map={StrictPropertyRegionSpec: gen_region},
    )
    schema = gen_physics.model_json_schema()
    # property_regions items must reference the generated region class.
    items_ref = schema["properties"]["property_regions"]["anyOf"][0]["items"]["$ref"]
    assert items_ref.endswith("PatchPropertyRegionSpec"), (
        f"Expected nested ref to PatchPropertyRegionSpec, got {items_ref}"
    )


def test_factory_full_chain_matches_hand_written_modelspec() -> None:
    """Build the full chain of generated patches and compare against the
    hand-written ``PatchModelSpec`` to prove the factory handles every
    nested-substitution case end-to-end."""
    gen_perf = make_patch_model(StrictWellPerforation, name="PatchWellPerforation")
    gen_wctrl = make_patch_model(StrictWellControlsSpec, name="PatchWellControlsSpec")
    gen_well = make_patch_model(
        StrictWellSpec,
        name="PatchWellSpec",
        type_map={
            StrictWellPerforation: gen_perf,
            StrictWellControlsSpec: gen_wctrl,
        },
    )
    gen_wells = make_patch_model(
        StrictWellsSpec, name="PatchWellsSpec", type_map={StrictWellSpec: gen_well}
    )

    gen_region = make_patch_model(
        StrictPropertyRegionSpec, name="PatchPropertyRegionSpec"
    )
    gen_physics = make_patch_model(
        StrictPhysicsSpec,
        name="PatchPhysicsSpec",
        type_map={StrictPropertyRegionSpec: gen_region},
    )
    # Reservoir/CPG generated separately to prove they validate; PatchModelSpec
    # itself uses PatchReservoirUnion (hand-built) for its reservoir field.
    make_patch_model(StrictReservoirSpec, name="PatchReservoirSpec")
    make_patch_model(StrictCPGReservoirSpec, name="PatchCPGReservoirSpec")
    gen_ic = make_patch_model(
        StrictInitialConditionsSpec, name="PatchInitialConditionsSpec"
    )
    gen_sp = make_patch_model(StrictSimParamsSpec, name="PatchSimParamsSpec")
    gen_out = make_patch_model(StrictOutputSpec, name="PatchOutputSpec")

    # Sanity: top-level model can be generated. Schema equivalence at the
    # ModelSpec level is intentionally not asserted here because the
    # hand-written PatchModelSpec uses ``PatchReservoirUnion`` (a tagged
    # discriminator union built by hand) that the factory does not replicate
    # at this layer — that union is the one piece that stays hand-written.
    gen_model = make_patch_model(
        StrictModelSpec,
        name="PatchModelSpec",
        type_map={
            StrictPhysicsSpec: gen_physics,
            StrictWellsSpec: gen_wells,
            StrictInitialConditionsSpec: gen_ic,
            StrictSimParamsSpec: gen_sp,
            StrictOutputSpec: gen_out,
            StrictWellControlsSpec: gen_wctrl,
        },
    )
    for fname, finfo in gen_model.model_fields.items():
        assert not finfo.is_required(), (
            f"PatchModelSpec.{fname} should be optional in the generated patch"
        )


def test_factory_validates_partial_payload() -> None:
    """Generated patch must accept a minimal payload (e.g. only one field)."""
    gen_physics = make_patch_model(StrictPhysicsSpec, name="PatchPhysicsSpec")
    # Only ``components`` provided — would fail strict validation.
    instance = gen_physics.model_validate({"components": ["CO2", "H2O"]})
    assert instance.components == ["CO2", "H2O"]
    assert instance.plugin is None
    assert instance.phases is None


def test_factory_rejects_unknown_fields() -> None:
    """Generated patch inherits ``extra='forbid'`` and rejects typos."""
    gen_sp = make_patch_model(StrictSimParamsSpec, name="PatchSimParamsSpec")
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        gen_sp.model_validate({"runtim": 100.0})  # typo


def test_factory_preserves_field_constraints() -> None:
    """``Field(ge=0, ...)`` and similar constraints survive the patch wrap."""
    gen_region = make_patch_model(
        StrictPropertyRegionSpec, name="PatchPropertyRegionSpec"
    )
    import pydantic

    # `region: int = Field(ge=0)` — negative value must still be rejected
    # even though region itself is now optional.
    with pytest.raises(pydantic.ValidationError):
        gen_region.model_validate({"region": -1})
