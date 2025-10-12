"""
Public API for schema-first configuration and plugin registry.

Modules:
- model_spec: Pydantic models for ModelSpec and subsections; utilities for merge and normalization.
- type_registry: TypeRegistry and PluginInstance for custom Python types with JSON Schemas.
- builder: ModelBuilder to apply a validated ModelSpec to a DartsModel instance.
- introspection: Helpers to export JSON Schemas and capabilities.
"""

from darts.api.builder import ModelBuilder
from darts.api.introspection import (
    get_plugin_schema_dict,
    get_schema_dict,
    list_capabilities_dict,
)
from darts.api.model_spec import (
    InitialConditionsSpec,
    ModelSpec,
    OutputSpec,
    PhysicsSpec,
    ReservoirSpec,
    SimParamsSpec,
    WellControlsSpec,
    WellsSpec,
    get_model_spec_dict,
    json_merge_patch,
    normalize_keys,
    upsert_model_spec_dict,
    validate_model_spec_dict,
)
from darts.api.type_registry import (
    TYPE_REGISTRY,
    PluginInstance,
    load_entry_point_plugins,
)

__all__ = [
    "ModelSpec",
    "ReservoirSpec",
    "PhysicsSpec",
    "WellsSpec",
    "InitialConditionsSpec",
    "WellControlsSpec",
    "SimParamsSpec",
    "OutputSpec",
    "PluginInstance",
    "TYPE_REGISTRY",
    "ModelBuilder",
    "json_merge_patch",
    "normalize_keys",
    "get_schema_dict",
    "list_capabilities_dict",
    "get_plugin_schema_dict",
    "validate_model_spec_dict",
    "upsert_model_spec_dict",
    "get_model_spec_dict",
    "load_entry_point_plugins",
]
