from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from darts.api.schemas import (
    InitialConditionsSpec,
    ModelSpec,
    OutputSpec,
    PatchModelSpec,
    PhysicsSpec,
    PluginRegistrySpec,
    PluginSlots,
    ReservoirSpec,
    SimParamsSpec,
    WellControlsSpec,
    WellsSpec,
)


def json_merge_patch(base: Any, patch: Any) -> Any:
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
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            ck = _SYNONYMS.get(k, k)
            out[ck] = normalize_keys(v)
        return out
    if isinstance(d, list):
        return [normalize_keys(x) for x in d]
    return d


# -----------------------------
# In-memory spec store helpers
# -----------------------------
_CANONICAL: dict[str, dict[str, Any]] = {}
_APPLIED_KEYS: dict[str, set] = {}


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
    model_cls = ModelSpec if strict else PatchModelSpec
    return _validate_with_model(model_cls, spec)


def validate_patch_model_spec_dict(spec: dict[str, Any]) -> dict[str, Any]:
    return _validate_with_model(PatchModelSpec, spec)


def get_model_spec_dict(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "spec": _CANONICAL.get(session_id, {})}


def upsert_model_spec_dict(
    session_id: str,
    patch: dict[str, Any],
    *,
    mode: str = "merge",
    idempotency_key: str | None = None,
    validate_only: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    base = {} if mode == "replace" else _CANONICAL.get(session_id, {})
    merged = (
        patch if mode == "replace" else json_merge_patch(base, normalize_keys(patch))
    )

    v = validate_model_spec_dict(merged, strict=strict)
    if not v.get("ok", False):
        return v

    if validate_only:
        return {"ok": True, "validated": True, "spec": merged}

    if idempotency_key:
        seen = _APPLIED_KEYS.setdefault(session_id, set())
        if idempotency_key in seen:
            return {
                "ok": True,
                "session_id": session_id,
                "spec": _CANONICAL.get(session_id, {}),
                "idempotent": True,
            }
        seen.add(idempotency_key)

    _CANONICAL[session_id] = merged
    return {"ok": True, "session_id": session_id, "spec": merged}


__all__ = [
    # Canonical schema aliases (single definitions live in schemas.py)
    "ModelSpec",
    "PatchModelSpec",
    "ReservoirSpec",
    "PhysicsSpec",
    "WellsSpec",
    "InitialConditionsSpec",
    "WellControlsSpec",
    "SimParamsSpec",
    "OutputSpec",
    "PluginSlots",
    "PluginRegistrySpec",
    # Utilities/state store
    "json_merge_patch",
    "normalize_keys",
    "validate_model_spec_dict",
    "validate_patch_model_spec_dict",
    "upsert_model_spec_dict",
    "get_model_spec_dict",
]
