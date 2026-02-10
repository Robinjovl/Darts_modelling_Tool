"""
Public API for schema-first configuration and plugin registry.

Modules:
- schemas: Canonical Pydantic models for ModelSpec and subsections.
- model_spec: compatibility aliases and utilities for merge and normalization.
- type_registry: TypeRegistry and PluginInstance for custom Python types with JSON Schemas.
- builder: ModelBuilder to apply a validated ModelSpec to a DartsModel instance.
- introspection: Helpers to export JSON Schemas and capabilities.
"""

from darts.api.builder import ModelBuilder
from darts.api.data_refs import register_object, resolve_data_ref
from darts.api.introspection import (
    get_plugin_schema_dict,
    get_schema_dict,
    list_capabilities_dict,
)
from darts.api.model_adapter import JsonModelAdapter, MCPModelAdapter, ModelAdapter
from darts.api.model_spec import (
    get_model_spec_dict,
    json_merge_patch,
    normalize_keys,
    upsert_model_spec_dict,
    validate_model_spec_dict,
    validate_patch_model_spec_dict,
)
from darts.api.schemas import (
    DataRef,
    InitialConditionsSpec,
    ModelSpec,
    OutputSpec,
    PatchModelSpec,
    PhysicsSpec,
    ReservoirSpec,
    SimParamsSpec,
    StrictModelSpec,
    WellControlsSpec,
    WellsSpec,
)
from darts.api.type_registry import (
    TYPE_REGISTRY,
    PluginInstance,
    list_customizable_types,
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
    "StrictModelSpec",
    "PatchModelSpec",
    "DataRef",
    "PluginInstance",
    "TYPE_REGISTRY",
    "list_customizable_types",
    "ModelBuilder",
    "ModelAdapter",
    "JsonModelAdapter",
    "MCPModelAdapter",
    "register_object",
    "resolve_data_ref",
    "json_merge_patch",
    "normalize_keys",
    "get_schema_dict",
    "list_capabilities_dict",
    "get_plugin_schema_dict",
    "validate_model_spec_dict",
    "validate_patch_model_spec_dict",
    "upsert_model_spec_dict",
    "get_model_spec_dict",
    "load_entry_point_plugins",
]
