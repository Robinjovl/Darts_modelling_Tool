from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from darts.api.type_registry import PluginInstance


class ReservoirSpec(BaseModel):
    type: Literal["structured"] = Field("structured", description="Reservoir type")
    nx: int | None = Field(None, ge=1)
    ny: int | None = Field(None, ge=1)
    nz: int | None = Field(None, ge=1)
    dx: float | None = Field(None, gt=0)
    dy: float | None = Field(None, gt=0)
    dz: float | None = Field(None, gt=0)
    permx: float | None = Field(None, gt=0)
    permy: float | None = Field(None, gt=0)
    permz: float | None = Field(None, gt=0)
    poro: float | None = Field(None, ge=0, le=1)
    depth: float | None = Field(None, ge=0)


class PluginSlots(BaseModel):
    flash_ev: PluginInstance | None = None
    density_ev: dict[str, PluginInstance] | None = None
    viscosity_ev: dict[str, PluginInstance] | None = None
    rel_perm_ev: dict[str, PluginInstance] | None = None
    diffusion_ev: dict[str, PluginInstance] | None = None
    kinetic_rate_ev: dict[str, PluginInstance] | None = None


class PropertyRegionSpec(BaseModel):
    region: int = 0
    property_container: PluginInstance | None = None
    plugins: PluginSlots | None = None


class PhysicsSpec(BaseModel):
    plugin: PluginInstance | None = None
    components: list[str] | None = None
    phases: list[str] | None = None
    property_regions: list[PropertyRegionSpec] | None = None


class WellControlsSpec(BaseModel):
    # Minimal, MCP-friendly controls; advanced schedules can be added later
    inj_bhp: float | None = Field(None, ge=0)
    prod_bhp: float | None = Field(None, ge=0)
    inj_composition: list[float] | None = None
    # Extended controls supported by json_model.py
    inj_rate: float | None = Field(None, ge=0)
    rate_type: (
        Literal[
            "MOLAR_RATE",
            "MASS_RATE",
            "VOLUMETRIC_RATE",
            "ADVECTIVE_HEAT_RATE",
        ]
        | None
    ) = None
    phase_name: str | None = None


class WellPerforation(BaseModel):
    # KJI indexing, 1-based as used in examples
    ijk: list[int] = Field(description="[i,j,k] indices (1-based)")
    well_radius: float | None = Field(None, gt=0)
    skin: float | None = None


class WellSpec(BaseModel):
    name: str
    perforations: list[WellPerforation]
    controls: WellControlsSpec | None = None


class WellsSpec(BaseModel):
    wells: list[WellSpec] | None = None


class InitialConditionsSpec(BaseModel):
    # optional: map component names to values; 'pressure' key for pressure
    by_array: dict[str, Any] | None = None


class SimParamsSpec(BaseModel):
    first_ts: float | None = Field(None, gt=0)
    mult_ts: float | None = Field(None, gt=0)
    max_ts: float | None = Field(None, gt=0)
    runtime: float | None = Field(None, gt=0)
    tol_newton: float | None = Field(None, gt=0)
    tol_linear: float | None = Field(None, gt=0)
    it_newton: int | None = Field(None, ge=1)
    it_linear: int | None = Field(None, ge=1)
    line_search: bool | None = None
    newton_tol_stationary: float | None = Field(None, gt=0)
    newton_type: Literal["newton_local_chop", "default"] | None = None
    min_line_search_update: float | None = Field(None, gt=0)


class OutputSpec(BaseModel):
    folder: str | None = None
    precision: Literal["s", "d"] | None = None
    save_initial: bool | None = None


class ModelSpec(BaseModel):
    apiVersion: Literal["darts/v1alpha1"] = "darts/v1alpha1"
    kind: Literal["Model"] = "Model"
    reservoir: ReservoirSpec | None = None
    physics: PhysicsSpec | None = None
    wells: WellsSpec | None = None
    initial_conditions: InitialConditionsSpec | None = None
    well_controls: WellControlsSpec | None = None
    sim_params: SimParamsSpec | None = None
    output: OutputSpec | None = None


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


def validate_model_spec_dict(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        try:
            ModelSpec.model_validate(normalize_keys(spec))
        except Exception:
            ModelSpec.parse_obj(normalize_keys(spec))
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


def get_model_spec_dict(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "spec": _CANONICAL.get(session_id, {})}


def upsert_model_spec_dict(
    session_id: str,
    patch: dict[str, Any],
    *,
    mode: str = "merge",
    idempotency_key: str | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    base = {} if mode == "replace" else _CANONICAL.get(session_id, {})
    merged = (
        patch if mode == "replace" else json_merge_patch(base, normalize_keys(patch))
    )

    v = validate_model_spec_dict(merged)
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
