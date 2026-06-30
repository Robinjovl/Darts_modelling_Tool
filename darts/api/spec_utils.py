"""
Pure utility functions for JSON model specification handling.

Provides RFC-7396 JSON merge-patch, key normalization (synonym mapping),
and validation helpers that check a raw dict against the strict ModelSpec
or the patch PatchModelSpec schema.  All functions are stateless and
operate on plain dicts, making them usable from both batch and interactive
workflows without side effects.
"""

import logging
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from darts.api.schemas import (
    ModelSpec,
    PatchModelSpec,
    StrictPluginSlots,
)

logger = logging.getLogger(__name__)


def json_merge_patch(base: Any, patch: Any) -> Any:
    """RFC-7396 JSON merge-patch: recursively merge *patch* into *base*."""
    if not isinstance(patch, dict):
        return deepcopy(patch)
    if not isinstance(base, dict):
        base = {}
    result = deepcopy(base)
    for k, v in patch.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = json_merge_patch(result.get(k), v)
    return result


_SYNONYMS = {
    "cells_x": "nx",
    "cells_y": "ny",
    "cells_z": "nz",
    "kx": "permx",
    "ky": "permy",
    "kz": "permz",
    "bhp_inj": "inj_bhp",
    "bhp_prod": "prod_bhp",
}


def normalize_keys(d: Any) -> Any:
    """Map known key synonyms to canonical names (recursive)."""
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            ck = _SYNONYMS.get(k, k)
            out[ck] = normalize_keys(v)
        return out
    if isinstance(d, list):
        return [normalize_keys(x) for x in d]
    return d


def migrate_flattened_region_plugins(
    spec: Any,
) -> tuple[Any, list[tuple[int, list[str]]]]:
    """Lift evaluator slots flattened onto a property region into ``plugins``.

    JSON producers (notably LLMs generating a config from the schema) sometimes
    place evaluator fields — the slots declared by
    :class:`darts.api.schemas.StrictPluginSlots` (``flash_ev``, ``density_ev``,
    ``viscosity_ev``, ...) — directly on a ``physics.property_regions[i]``
    object instead of nesting them under that region's ``plugins`` key.
    ``StrictPropertyRegionSpec`` forbids extra fields, so the flattened form is
    rejected even though it has exactly one unambiguous valid interpretation:
    those field names belong only to ``StrictPluginSlots`` and never to a
    region (the two field sets are disjoint). This helper rewrites the
    flattened encoding into the canonical nested one.

    The transform is lossless and non-silent:

    * It operates on a deep copy; the input ``spec`` is never mutated.
    * Only names present in :attr:`StrictPluginSlots.model_fields` (read live,
      never hard-coded) are moved; every other region field is left untouched.
    * If a region's ``plugins`` already holds a key that is also present at the
      region top level, the flattened key is left in place (not merged, not
      overwritten) so strict validation still surfaces the genuine ambiguity.
    * A non-dict ``plugins`` value is left untouched so strict validation
      reports it.
    * The returned report lists what moved so callers can log the rewrite.

    The operation is idempotent: applying it to an already-nested spec is a
    no-op that returns an empty report.

    :param spec: Raw JSON-decoded model spec; non-dict inputs pass through.
    :type spec: Any
    :return: ``(normalized_spec, migrations)`` where ``migrations`` is a list
        of ``(region_index, sorted_field_names)`` tuples describing the moves.
    :rtype: tuple[Any, list[tuple[int, list[str]]]]
    """
    if not isinstance(spec, dict):
        return spec, []
    physics = spec.get("physics")
    if not isinstance(physics, dict):
        return spec, []
    regions = physics.get("property_regions")
    if not isinstance(regions, list):
        return spec, []

    evaluator_fields = set(StrictPluginSlots.model_fields)
    result = deepcopy(spec)
    migrations: list[tuple[int, list[str]]] = []
    for index, region in enumerate(result["physics"]["property_regions"]):
        if not isinstance(region, dict):
            continue
        flattened = [key for key in region if key in evaluator_fields]
        if not flattened:
            continue
        existing = region.get("plugins")
        if existing is not None and not isinstance(existing, dict):
            # Malformed plugins block: leave the region as-is so strict
            # validation reports the real error instead of us guessing.
            continue
        plugins = dict(existing) if isinstance(existing, dict) else {}
        moved: list[str] = []
        for key in flattened:
            if key in plugins:
                # Same slot is both flattened and nested: genuinely ambiguous,
                # so leave the flattened copy for strict validation to flag.
                continue
            plugins[key] = region.pop(key)
            moved.append(key)
        if moved:
            region["plugins"] = plugins
            migrations.append((index, sorted(moved)))
    return result, migrations


def _err(
    field: str, code: str, expected: str, received: Any = None, hint: str = ""
) -> dict[str, Any]:
    return {
        "field": field,
        "code": code,
        "expected": expected,
        "received": received,
        "hint": hint,
    }


def _validate_with_model(model_cls: Any, spec: dict[str, Any]) -> dict[str, Any]:
    migrated, migrations = migrate_flattened_region_plugins(spec)
    if migrations:
        logger.info(
            "Lifted flattened evaluator slots into property-region plugins: %s",
            migrations,
        )
    try:
        model_cls.model_validate(normalize_keys(migrated))
        return {"ok": True}
    except ValidationError as e:
        errs: list[dict[str, Any]] = []
        for er in e.errors():
            loc = ".".join(map(str, er.get("loc", [])))
            msg = er.get("msg", "invalid value")
            errs.append(
                _err(
                    field=loc,
                    code="validation_error",
                    expected=msg,
                    hint="fix this field",
                )
            )
        return {"ok": False, "errors": errs}


def validate_model_spec_dict(
    spec: dict[str, Any], *, strict: bool = True
) -> dict[str, Any]:
    """Validate *spec* against ModelSpec (strict) or PatchModelSpec."""
    model_cls = ModelSpec if strict else PatchModelSpec
    return _validate_with_model(model_cls, spec)


def validate_patch_model_spec_dict(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate *spec* as a partial patch."""
    return _validate_with_model(PatchModelSpec, spec)


__all__ = [
    # Canonical schema aliases (single definitions live in schemas.py)
    "ModelSpec",
    "PatchModelSpec",
    # Utilities
    "json_merge_patch",
    "normalize_keys",
    "migrate_flattened_region_plugins",
    "validate_model_spec_dict",
    "validate_patch_model_spec_dict",
]
