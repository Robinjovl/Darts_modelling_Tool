"""
JSON Schema and plugin capability export helpers.

Generates the full JSON Schema (Draft 2020-12) for ModelSpec and its
subsections, and lists available plugin types from the TypeRegistry.
Used by the MCP server to expose ``get_model_config_schema`` and
``list_capabilities`` tools, giving LLM agents programmatic access to
the configuration grammar and available physics/evaluator plugins.
"""

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
    if kind == "reservoir":
        # Return the reservoir property schema from ModelSpec so union members
        # (e.g., structured and cpg variants) stay discoverable.
        model_schema = _json_schema(ModelSpec)
        reservoir_schema = model_schema.get("properties", {}).get(
            "reservoir"
        ) or _json_schema(ReservoirSpec)
        return {"kind": "reservoir", "schema": reservoir_schema}

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
