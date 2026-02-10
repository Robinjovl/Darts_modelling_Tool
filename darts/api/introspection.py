from typing import Any

from darts.api.schemas import (
    InitialConditionsSpec,
    ModelSpec,
    OutputSpec,
    PhysicsSpec,
    ReservoirSpec,
    SimParamsSpec,
    WellControlsSpec,
    WellsSpec,
)
from darts.api.type_registry import get_kind_schema, list_capabilities


def _json_schema(cls: Any) -> dict[str, Any]:
    try:
        return cls.model_json_schema()
    except Exception:  # pragma: no cover
        return cls.schema()


def get_schema_dict(kind: str | None = None) -> dict[str, Any]:
    mapping = {
        None: ModelSpec,
        "reservoir": ReservoirSpec,
        "physics": PhysicsSpec,
        "wells": WellsSpec,
        "initial_conditions": InitialConditionsSpec,
        "well_controls": WellControlsSpec,
        "sim_params": SimParamsSpec,
        "output": OutputSpec,
    }
    cls = mapping.get(kind, ModelSpec)
    return {"kind": kind or "ModelSpec", "schema": _json_schema(cls)}


def list_capabilities_dict(kind: str | None = None) -> dict[str, Any]:
    return {
        "kinds": list({item["kind"] for item in list_capabilities(None)}),
        "items": list_capabilities(kind),
    }


def get_plugin_schema_dict(kind: str) -> dict[str, Any]:
    return get_kind_schema(kind)
