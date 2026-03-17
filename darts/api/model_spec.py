from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from darts.api.schemas import (
    ModelSpec,
    PatchModelSpec,
)


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
    try:
        model_cls.model_validate(normalize_keys(spec))
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
    "validate_model_spec_dict",
    "validate_patch_model_spec_dict",
]
