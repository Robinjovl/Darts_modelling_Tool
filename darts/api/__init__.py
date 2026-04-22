"""
Public API for schema-first model configuration from JSON.

Provides the full pipeline for defining, validating, and applying DARTS
simulation models from a single JSON configuration file.  Also exposes a
plugin registry for user-defined physics evaluators and adapters for both
batch (JSON file) and interactive (MCP tool-by-tool) workflows.

Modules
-------
schemas        Pydantic v2 models for ModelSpec and every subsection (Strict
               and Patch variants).
type_registry  TypeRegistry singleton and built-in plugin entries for physics,
               property containers, and evaluators.
builder        Stateless ModelBuilder that applies a validated ModelSpec to any
               DartsModelProtocol-compatible model.
spec_utils     Pure utility functions: JSON merge-patch, key normalization,
               and spec validation helpers.
model_adapter  Stateful adapters (JsonModelAdapter, MCPModelAdapter) that wrap
               ModelBuilder with spec tracking and idempotency.
json_model     DartsModel subclass with lifecycle overrides for JSON-driven
               execution.
data_refs      DataRef resolution: file paths, URIs, and in-memory objects.
autospec       Reverse-engineering tool that records a ModelSpec by
               monkey-patching DARTS classes during normal Python execution.
introspection  JSON Schema and plugin capability export helpers.
run_json_model CLI entry point for running a model from a JSON file.
"""

from darts.api.builder import DartsModelProtocol, ModelBuilder
from darts.api.data_refs import register_object, resolve_data_ref
from darts.api.introspection import (
    get_plugin_schema_dict,
    get_schema_dict,
    list_capabilities_dict,
)
from darts.api.model_adapter import JsonModelAdapter, MCPModelAdapter, ModelAdapter
from darts.api.model_config import (
    ModelConfig,
    PhysicsConfig,
    load_model_config,
)
from darts.api.presets import (
    Preset,
    PresetMeta,
    iter_presets,
    list_presets,
    load_preset,
    load_preset_config,
    load_preset_dir,
    register_preset,
    register_preset_directory_binding,
)
from darts.api.presets import (
    docs as preset_docs,
)
from darts.api.schemas import (
    CPGReservoirSpec,
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
from darts.api.spec_utils import (
    json_merge_patch,
    normalize_keys,
    validate_model_spec_dict,
    validate_patch_model_spec_dict,
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
    "CPGReservoirSpec",
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
    "DartsModelProtocol",
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
    "load_entry_point_plugins",
    "Preset",
    "PresetMeta",
    "iter_presets",
    "list_presets",
    "load_preset",
    "load_preset_config",
    "load_preset_dir",
    "preset_docs",
    "register_preset",
    "register_preset_directory_binding",
    "ModelConfig",
    "PhysicsConfig",
    "load_model_config",
]
