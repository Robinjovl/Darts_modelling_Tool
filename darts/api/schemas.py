"""
Pydantic v2 data models for the DARTS JSON model specification.

Defines every section of a model config (reservoir, physics, wells, initial
conditions, simulation parameters, output) as strict Pydantic models with
full JSON Schema export.  Provides two parallel hierarchies: Strict variants
for complete validation and Patch variants for RFC-7396 merge-patch updates.
Also includes the DataRef union for external data references and a
discriminated union for reservoir type routing (structured / CPG / DataRef).
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from darts.api.type_registry import PluginInstance
from darts.models.darts_model import (
    InitialConditionsConfig,
    OutputConfig,
    SimParamsConfig,
    WellConfig,
    WellControlsConfig,
    WellPerforationConfig,
    WellsConfig,
)
from darts.reservoirs.cpg_reservoir import CPGReservoirConfig
from darts.reservoirs.struct_reservoir import (
    ReservoirLayerConfig,
    StructReservoirConfig,
)


class SpecBaseModel(BaseModel):
    """Strict base model used by all JSON schema specs."""

    model_config = ConfigDict(extra="forbid")


class DataRef(SpecBaseModel):
    """JSON-only reference to external data (path/URI) or in-memory object.

    Exists because native DARTS configs only accept plain Python values;
    DataRef lets a JSON payload point at data without inlining it. Resolved
    to a concrete value by :func:`darts.api.data_refs.resolve_data_ref`
    before any native config is constructed.
    """

    model_config = ConfigDict(
        title="DataRef",
        json_schema_extra={
            "examples": [{"kind": "path", "value": "data/model.json", "format": "json"}]
        },
    )

    kind: Annotated[
        Literal["path", "uri", "object"],
        Field(description="Reference kind: path, uri, or object"),
    ]
    value: Annotated[
        str, Field(description="Reference value (path, URI, or object key)")
    ]
    format: Annotated[
        Literal["json", "text", "binary"] | None,
        Field(description="Optional data format hint"),
    ] = None
    encoding: Annotated[
        str | None, Field(description="Optional text encoding override")
    ] = None


# Scalar, per-cell array, or DataRef. Native configs use ScalarOrArray;
# adding DataRef here is what makes reservoir fields JSON-loadable from
# external files without widening the native type.
ReservoirValue = float | list[float] | DataRef


def _reservoir_discriminator(v: Any) -> str:
    """Route a reservoir payload to the right spec variant.

    Needed because the reservoir section is a JSON discriminated union
    (structured / cpg / dataref) — native configs have no such union since
    each reservoir class is instantiated directly in Python.
    """
    if isinstance(v, dict):
        if "kind" in v and "value" in v:
            return "dataref"
        return v.get("type", "structured")
    if isinstance(v, DataRef):
        return "dataref"
    return getattr(v, "type", "structured")


class PluginRegistryEntrySpec(SpecBaseModel):
    """JSON-side record for registering a user plugin at load time.

    Native DARTS has no plugin registry — classes are imported and wired
    up directly in Python. This spec lets a JSON model declare an
    importable constructor (and optional config model) to be added to
    :data:`TYPE_REGISTRY` before the model is applied.
    """

    model_config = ConfigDict(
        title="PluginRegistryEntrySpec",
        json_schema_extra={
            "examples": [
                {
                    "type_id": "pc/ModelProperties@v1",
                    "kind": "pc",
                    "constructor": "model.py:ModelProperties",
                    "config_model": "darts.api.type_registry:PropertyContainerConfig",
                    "customizable": True,
                    "doc": "Local ModelProperties property container",
                }
            ]
        },
    )

    type_id: Annotated[str, Field(description="Registered type identifier")]
    kind: Annotated[str, Field(description="Plugin kind (physics, pc, density, etc.)")]
    constructor: Annotated[
        str,
        Field(description="Import path to constructor (module:attr or file.py:attr)"),
    ]
    config_model: Annotated[
        str | None,
        Field(description="Optional import path to a Pydantic config model"),
    ] = None
    doc: Annotated[str | None, Field(description="Optional documentation")] = None
    customizable: Annotated[
        bool | None, Field(description="Whether plugin is user-customizable")
    ] = None


class PluginRegistrySpec(SpecBaseModel):
    """Top-level ``plugin_registry`` section of a JSON model spec.

    Lists modules to import (which self-register via
    ``register_darts_plugins`` / ``DARTS_PLUGIN_ENTRIES``) and/or explicit
    entries. JSON-only: purely a transport format for configuring
    :data:`TYPE_REGISTRY`.
    """

    model_config = ConfigDict(
        title="PluginRegistrySpec",
        json_schema_extra={
            "examples": [
                {
                    "modules": ["models/2ph_do/model.py"],
                    "entries": [
                        {
                            "type_id": "pc/ModelProperties@v1",
                            "kind": "pc",
                            "constructor": "model.py:ModelProperties",
                        }
                    ],
                }
            ]
        },
    )

    modules: Annotated[
        list[str] | None,
        Field(description="Modules or file paths to import for plugin registration"),
    ] = None
    entries: Annotated[
        list[PluginRegistryEntrySpec] | None,
        Field(description="Explicit plugin registry entries"),
    ] = None


class StrictReservoirLayerSpec(ReservoirLayerConfig):
    """JSON variant of :class:`ReservoirLayerConfig` with DataRef support.

    Identical to the native config except per-cell fields accept
    :class:`DataRef` in addition to scalars/arrays. The builder resolves
    DataRefs before constructing the native :class:`ReservoirLayerConfig`.
    """

    dx: Annotated[
        ReservoirValue | None, Field(gt=0, description="Cell size in x [m]")
    ] = None
    dy: Annotated[
        ReservoirValue | None, Field(gt=0, description="Cell size in y [m]")
    ] = None
    dz: Annotated[
        ReservoirValue | None, Field(gt=0, description="Cell size in z [m]")
    ] = None
    permx: Annotated[
        ReservoirValue | None, Field(gt=0, description="Permeability in x [mD]")
    ] = None
    permy: Annotated[
        ReservoirValue | None, Field(gt=0, description="Permeability in y [mD]")
    ] = None
    permz: Annotated[
        ReservoirValue | None, Field(gt=0, description="Permeability in z [mD]")
    ] = None
    poro: Annotated[
        ReservoirValue | None,
        Field(ge=0, le=1, description="Porosity [fraction]"),
    ] = None
    depth: Annotated[
        ReservoirValue | None, Field(ge=0, description="Reference depth [m]")
    ] = None


class StrictPluginInstance(PluginInstance):
    """Plugin reference with required configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="StrictPluginInstance",
        json_schema_extra={
            "examples": [
                {
                    "type_id": "physics/Compositional@v1",
                    "config": {"state_spec": "P", "n_points": 200},
                }
            ]
        },
    )

    type_id: Annotated[
        str, Field(description="Registered type identifier (kind/name@version)")
    ]
    config: Annotated[dict[str, Any], Field(description="Plugin configuration")]


class StrictReservoirSpec(StructReservoirConfig):
    """JSON variant of :class:`StructReservoirConfig` with DataRef support.

    Only diverges from the native config to widen per-cell fields to
    :data:`ReservoirValue` — the single place where DataRef (a
    JSON-transport concern) leaks into a reservoir spec. The builder
    resolves DataRefs and then constructs the native
    :class:`StructReservoirConfig`.
    """

    # Override property fields to accept DataRef in addition to plain scalars/arrays
    dx: Annotated[
        ReservoirValue | None,
        Field(gt=0, description="Cell size in x direction [m]"),
    ] = None
    dy: Annotated[
        ReservoirValue | None,
        Field(gt=0, description="Cell size in y direction [m]"),
    ] = None
    dz: Annotated[
        ReservoirValue | None,
        Field(gt=0, description="Cell size in z direction [m]"),
    ] = None
    permx: Annotated[
        ReservoirValue | None,
        Field(gt=0, description="Permeability in x direction [mD]"),
    ] = None
    permy: Annotated[
        ReservoirValue | None,
        Field(gt=0, description="Permeability in y direction [mD]"),
    ] = None
    permz: Annotated[
        ReservoirValue | None,
        Field(gt=0, description="Permeability in z direction [mD]"),
    ] = None
    poro: Annotated[
        ReservoirValue | None,
        Field(ge=0, le=1, description="Porosity [fraction]"),
    ] = None
    depth: Annotated[
        ReservoirValue | None,
        Field(ge=0, description="Reference depth [m]"),
    ] = None
    hcap: Annotated[
        ReservoirValue | None,
        Field(ge=0, description="Heat capacity [J/kg-K], 0 = non-thermal"),
    ] = None
    rcond: Annotated[
        ReservoirValue | None,
        Field(ge=0, description="Rock thermal conductivity [W/m-K], 0 = non-thermal"),
    ] = None
    layers: Annotated[
        list[StrictReservoirLayerSpec] | None,
        Field(description="Layered overrides for per-cell properties"),
    ] = None


StrictCPGReservoirSpec = CPGReservoirConfig


class StrictPluginSlots(SpecBaseModel):
    """Evaluator plugins attached to a property region."""

    model_config = ConfigDict(
        title="StrictPluginSlots",
        json_schema_extra={
            "examples": [
                {
                    "flash_ev": {
                        "type_id": "flash/ConstantK@v1",
                        "config": {"K": [4.0, 2.0, 0.1], "epsilon": 1e-8},
                    },
                    "density_ev": {
                        "gas": {
                            "type_id": "density/DensityBasic@v1",
                            "config": {"compr": 1e-3, "dens0": 200.0},
                        }
                    },
                }
            ]
        },
    )

    flash_ev: Annotated[
        StrictPluginInstance | None, Field(description="Flash evaluator plugin")
    ] = None
    density_ev: Annotated[
        dict[str, StrictPluginInstance] | None,
        Field(description="Density evaluator plugins per phase"),
    ] = None
    viscosity_ev: Annotated[
        dict[str, StrictPluginInstance] | None,
        Field(description="Viscosity evaluator plugins per phase"),
    ] = None
    enthalpy_ev: Annotated[
        dict[str, StrictPluginInstance] | None,
        Field(description="Enthalpy evaluator plugins per phase"),
    ] = None
    conductivity_ev: Annotated[
        dict[str, StrictPluginInstance] | None,
        Field(description="Thermal conductivity plugins per phase"),
    ] = None
    rel_perm_ev: Annotated[
        dict[str, StrictPluginInstance] | None,
        Field(description="Relative permeability plugins per phase"),
    ] = None
    diffusion_ev: Annotated[
        dict[str, StrictPluginInstance] | None,
        Field(description="Diffusion evaluator plugins per phase"),
    ] = None
    kinetic_rate_ev: Annotated[
        dict[str, StrictPluginInstance] | None,
        Field(description="Kinetic rate plugins keyed by reaction index"),
    ] = None


class StrictPropertyRegionSpec(SpecBaseModel):
    """Property container and evaluators for a region."""

    model_config = ConfigDict(
        title="StrictPropertyRegionSpec",
        json_schema_extra={
            "examples": [
                {
                    "region": 0,
                    "property_container": {
                        "type_id": "pc/SuperPropertyContainer@v1",
                        "config": {
                            "components_name": ["CO2", "C1", "H2O"],
                            "phases_name": ["gas", "oil"],
                            "Mw": [44.01, 16.04, 18.015],
                            "min_z": 1e-9,
                            "temperature": 1.0,
                        },
                    },
                }
            ]
        },
    )

    region: Annotated[int, Field(ge=0, description="Region index")]
    property_container: Annotated[
        StrictPluginInstance, Field(description="Property container plugin instance")
    ]
    plugins: Annotated[
        StrictPluginSlots | None, Field(description="Evaluator plugins for this region")
    ] = None


class StrictPhysicsSpec(SpecBaseModel):
    """Physics configuration driven by plugins."""

    model_config = ConfigDict(
        title="StrictPhysicsSpec",
        json_schema_extra={
            "examples": [
                {
                    "plugin": {
                        "type_id": "physics/Compositional@v1",
                        "config": {"state_spec": "P", "n_points": 200},
                    },
                    "components": ["CO2", "C1", "H2O"],
                    "phases": ["gas", "oil"],
                }
            ]
        },
    )

    plugin: Annotated[
        StrictPluginInstance, Field(description="Physics plugin instance")
    ]
    components: Annotated[
        list[str],
        Field(min_length=1, description="List of component names (order matters)"),
    ]
    phases: Annotated[
        list[str],
        Field(min_length=1, description="List of phase names (order matters)"),
    ]
    property_regions: Annotated[
        list[StrictPropertyRegionSpec] | None,
        Field(description="Property regions for evaluators"),
    ] = None


StrictWellControlsSpec = WellControlsConfig


StrictWellPerforation = WellPerforationConfig


StrictWellSpec = WellConfig


StrictWellsSpec = WellsConfig


StrictInitialConditionsSpec = InitialConditionsConfig


StrictSimParamsSpec = SimParamsConfig


StrictOutputSpec = OutputConfig


ReservoirUnion = Annotated[
    Annotated[StrictReservoirSpec, Tag("structured")]
    | Annotated[StrictCPGReservoirSpec, Tag("cpg")]
    | Annotated[DataRef, Tag("dataref")],
    Discriminator(_reservoir_discriminator),
]


class StrictModelSpec(SpecBaseModel):
    """Full model specification."""

    model_config = ConfigDict(
        title="StrictModelSpec",
        json_schema_extra={
            "examples": [
                {
                    "reservoir": {
                        "type": "structured",
                        "nx": 10,
                        "ny": 10,
                        "nz": 1,
                        "dx": 10.0,
                        "dy": 10.0,
                        "dz": 10.0,
                        "permx": 100.0,
                        "permy": 100.0,
                        "permz": 10.0,
                        "poro": 0.3,
                        "depth": 1000.0,
                    },
                    "physics": {
                        "plugin": {
                            "type_id": "physics/Compositional@v1",
                            "config": {"state_spec": "P", "n_points": 200},
                        },
                        "components": ["CO2", "C1", "H2O"],
                        "phases": ["gas", "oil"],
                    },
                }
            ]
        },
    )

    plugin_registry: Annotated[
        PluginRegistrySpec | DataRef | None,
        Field(description="Local plugin registry configuration"),
    ] = None
    reservoir: Annotated[
        ReservoirUnion | None,
        Field(description="Reservoir configuration"),
    ] = None
    physics: Annotated[
        StrictPhysicsSpec | DataRef | None,
        Field(description="Physics configuration"),
    ] = None
    wells: Annotated[
        StrictWellsSpec | DataRef | None, Field(description="Wells configuration")
    ] = None
    initial_conditions: Annotated[
        StrictInitialConditionsSpec | DataRef | None,
        Field(description="Initial conditions"),
    ] = None
    well_controls: Annotated[
        StrictWellControlsSpec | DataRef | None,
        Field(
            description=(
                "Default well controls applied to all wells. "
                "Per-well controls in wells[].controls take precedence "
                "over these defaults for individual wells."
            )
        ),
    ] = None
    sim_params: Annotated[
        StrictSimParamsSpec | DataRef | None,
        Field(description="Simulation parameters"),
    ] = None
    output: Annotated[
        StrictOutputSpec | DataRef | None,
        Field(description="Output configuration"),
    ] = None


# ---------------------------------------------------------------------------
# Patch* variants
# ---------------------------------------------------------------------------
# Parallel hierarchy to the Strict* specs where every field is optional so
# clients can send partial updates. Used by the MCP adapter to implement
# RFC-7396 JSON Merge Patch: the server validates incoming patches against
# Patch* models, merges them into the accumulated spec, then validates the
# result against Strict* before building. Purely an MCP/JSON-transport
# concern — native DARTS objects are always fully constructed.
# ---------------------------------------------------------------------------


class PatchPluginSlots(StrictPluginSlots):
    """Optional evaluator plugin overrides (see Patch* variants header)."""

    model_config = ConfigDict(
        title="PatchPluginSlots",
        json_schema_extra={
            "examples": [
                {
                    "flash_ev": {
                        "type_id": "flash/ConstantK@v1",
                        "config": {"K": [4.0, 2.0, 0.1], "epsilon": 1e-8},
                    }
                }
            ]
        },
    )


class PatchPropertyRegionSpec(StrictPropertyRegionSpec):
    """Partial property region override."""

    model_config = ConfigDict(
        title="PatchPropertyRegionSpec",
        json_schema_extra={
            "examples": [
                {
                    "region": 0,
                    "plugins": {
                        "density_ev": {
                            "gas": {
                                "type_id": "density/DensityBasic@v1",
                                "config": {"compr": 1e-3, "dens0": 200.0},
                            }
                        }
                    },
                }
            ]
        },
    )

    region: Annotated[int | None, Field(ge=0, description="Region index")] = None
    property_container: Annotated[
        StrictPluginInstance | None,
        Field(description="Property container plugin instance"),
    ] = None
    plugins: Annotated[
        PatchPluginSlots | None, Field(description="Evaluator plugins for this region")
    ] = None


class PatchPhysicsSpec(StrictPhysicsSpec):
    """Partial physics override."""

    model_config = ConfigDict(
        title="PatchPhysicsSpec",
        json_schema_extra={
            "examples": [
                {
                    "plugin": {
                        "type_id": "physics/Compositional@v1",
                        "config": {"state_spec": "P", "n_points": 200},
                    }
                }
            ]
        },
    )

    plugin: Annotated[
        StrictPluginInstance | None, Field(description="Physics plugin")
    ] = None
    components: Annotated[
        list[str] | None,
        Field(min_length=1, description="List of component names (order matters)"),
    ] = None
    phases: Annotated[
        list[str] | None,
        Field(min_length=1, description="List of phase names (order matters)"),
    ] = None
    property_regions: Annotated[
        list[PatchPropertyRegionSpec] | None,
        Field(description="Property regions for evaluators"),
    ] = None


class PatchReservoirSpec(StrictReservoirSpec):
    """Partial reservoir override."""

    model_config = ConfigDict(
        title="PatchReservoirSpec",
        json_schema_extra={
            "examples": [
                {"nx": 20, "ny": 1, "nz": 1, "dx": 5.0, "dy": 10.0, "dz": 10.0}
            ]
        },
    )

    type: Annotated[
        Literal["structured"] | None,
        Field(description="Reservoir type (structured only in v1)"),
    ] = None
    nx: Annotated[
        int | None, Field(ge=1, description="Number of cells in x direction")
    ] = None
    ny: Annotated[
        int | None, Field(ge=1, description="Number of cells in y direction")
    ] = None
    nz: Annotated[
        int | None, Field(ge=1, description="Number of cells in z direction")
    ] = None
    hcap: Annotated[float | None, Field(ge=0, description="Heat capacity [J/kg-K]")] = (
        None
    )
    rcond: Annotated[
        float | None,
        Field(ge=0, description="Rock thermal conductivity [W/m-K]"),
    ] = None


class PatchCPGReservoirSpec(StrictCPGReservoirSpec):
    """Partial CPG reservoir override."""

    model_config = ConfigDict(
        title="PatchCPGReservoirSpec",
        json_schema_extra={
            "examples": [
                {"type": "cpg", "grid_file": "meshes/brugge/grid.grdecl"},
                {"boundary_volume": 1e10},
            ]
        },
    )

    type: Annotated[
        Literal["cpg"] | None,
        Field(description="Reservoir type (corner-point geometry)"),
    ] = None
    grid_file: Annotated[str | None, Field(description="Path to GRDECL grid file")] = (
        None
    )
    prop_file: Annotated[
        str | None,
        Field(description="Path to reservoir property file (GRDECL-like)"),
    ] = None
    fault_file: Annotated[
        str | None,
        Field(description="Optional file with fault transmissibility multipliers"),
    ] = None


PatchReservoirUnion = Annotated[
    Annotated[PatchReservoirSpec, Tag("structured")]
    | Annotated[PatchCPGReservoirSpec, Tag("cpg")]
    | Annotated[DataRef, Tag("dataref")],
    Discriminator(_reservoir_discriminator),
]


class PatchWellControlsSpec(StrictWellControlsSpec):
    """Partial well controls override."""

    model_config = ConfigDict(
        title="PatchWellControlsSpec",
        json_schema_extra={"examples": [{"inj_bhp": 150.0, "rate_type": "MOLAR_RATE"}]},
    )


class PatchWellPerforation(StrictWellPerforation):
    """Partial perforation override."""

    model_config = ConfigDict(
        title="PatchWellPerforation",
        json_schema_extra={"examples": [{"ijk": [1, 1, 1], "skin": 2.0}]},
    )

    ijk: Annotated[
        list[int] | None,
        Field(description="[i,j,k] indices (1-based)", min_length=3, max_length=3),
    ] = None
    k_end: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Optional inclusive K-end index for vertical completion interval; "
                "when provided, expands ijk[2]..k_end into multiple perforations"
            ),
        ),
    ] = None


class PatchWellSpec(StrictWellSpec):
    """Partial well override."""

    model_config = ConfigDict(
        title="PatchWellSpec",
        json_schema_extra={
            "examples": [
                {"name": "P1", "controls": {"prod_bhp": 50.0}},
                {"perforations": [{"ijk": [2, 1, 1]}]},
            ]
        },
    )

    name: Annotated[str | None, Field(description="Well name")] = None
    perforations: Annotated[
        list[PatchWellPerforation] | None, Field(description="List of perforations")
    ] = None
    controls: Annotated[
        PatchWellControlsSpec | None, Field(description="Optional well controls")
    ] = None


class PatchWellsSpec(StrictWellsSpec):
    """Partial wells override."""

    model_config = ConfigDict(
        title="PatchWellsSpec",
        json_schema_extra={
            "examples": [
                {"wells": [{"name": "I1", "perforations": [{"ijk": [1, 1, 1]}]}]}
            ]
        },
    )

    wells: Annotated[
        list[PatchWellSpec] | None,
        Field(min_length=1, description="List of well specifications"),
    ] = None


class PatchInitialConditionsSpec(StrictInitialConditionsSpec):
    """Partial initial conditions override."""

    model_config = ConfigDict(
        title="PatchInitialConditionsSpec",
        json_schema_extra={"examples": [{"by_array": {"pressure": 60.0, "CO2": 0.05}}]},
    )

    by_array: Annotated[
        dict[str, Any] | None,
        Field(description="Initial state variables (pressure, compositions, etc.)"),
    ] = None


class PatchSimParamsSpec(StrictSimParamsSpec):
    """Partial simulation parameters override."""

    model_config = ConfigDict(
        title="PatchSimParamsSpec",
        json_schema_extra={
            "examples": [{"runtime": 500.0, "max_ts": 2.0, "tol_newton": 0.02}]
        },
    )


class PatchOutputSpec(StrictOutputSpec):
    """Partial output override."""

    model_config = ConfigDict(
        title="PatchOutputSpec",
        json_schema_extra={"examples": [{"folder": "output", "precision": "d"}]},
    )


class PatchModelSpec(StrictModelSpec):
    """Partial model spec for merge-patch updates."""

    model_config = ConfigDict(
        title="PatchModelSpec",
        json_schema_extra={
            "examples": [
                {"reservoir": {"nx": 20, "ny": 1, "nz": 1}},
                {"sim_params": {"runtime": 100.0}, "output": {"folder": "output"}},
            ]
        },
    )

    reservoir: Annotated[
        PatchReservoirUnion | None,
        Field(description="Reservoir configuration"),
    ] = None
    physics: Annotated[
        PatchPhysicsSpec | DataRef | None, Field(description="Physics configuration")
    ] = None
    wells: Annotated[
        PatchWellsSpec | DataRef | None, Field(description="Wells configuration")
    ] = None
    initial_conditions: Annotated[
        PatchInitialConditionsSpec | DataRef | None,
        Field(description="Initial conditions"),
    ] = None
    well_controls: Annotated[
        PatchWellControlsSpec | DataRef | None, Field(description="Well controls")
    ] = None
    sim_params: Annotated[
        PatchSimParamsSpec | DataRef | None, Field(description="Simulation parameters")
    ] = None
    output: Annotated[
        PatchOutputSpec | DataRef | None, Field(description="Output configuration")
    ] = None


ModelSpec = StrictModelSpec
ReservoirSpec = StrictReservoirSpec
CPGReservoirSpec = StrictCPGReservoirSpec
PhysicsSpec = StrictPhysicsSpec
WellsSpec = StrictWellsSpec
InitialConditionsSpec = StrictInitialConditionsSpec
WellControlsSpec = StrictWellControlsSpec
SimParamsSpec = StrictSimParamsSpec
OutputSpec = StrictOutputSpec
PluginSlots = StrictPluginSlots
PropertyRegionSpec = StrictPropertyRegionSpec

__all__ = [
    # Strict (canonical) specs
    "StrictModelSpec",
    "StrictReservoirSpec",
    "StrictCPGReservoirSpec",
    "StrictPhysicsSpec",
    "StrictWellsSpec",
    "StrictWellControlsSpec",
    "StrictInitialConditionsSpec",
    "StrictSimParamsSpec",
    "StrictOutputSpec",
    "StrictPluginSlots",
    "StrictPropertyRegionSpec",
    "StrictPluginInstance",
    "StrictWellSpec",
    "StrictWellPerforation",
    "StrictReservoirLayerSpec",
    # Patch specs
    "PatchModelSpec",
    "PatchReservoirSpec",
    "PatchCPGReservoirSpec",
    "PatchPhysicsSpec",
    "PatchWellsSpec",
    "PatchWellControlsSpec",
    "PatchInitialConditionsSpec",
    "PatchSimParamsSpec",
    "PatchOutputSpec",
    "PatchPluginSlots",
    "PatchPropertyRegionSpec",
    "PatchWellSpec",
    "PatchWellPerforation",
    # Convenience aliases
    "ModelSpec",
    "ReservoirSpec",
    "CPGReservoirSpec",
    "PhysicsSpec",
    "WellsSpec",
    "WellControlsSpec",
    "InitialConditionsSpec",
    "SimParamsSpec",
    "OutputSpec",
    "PluginSlots",
    "PropertyRegionSpec",
    # Shared types
    "DataRef",
    "ReservoirValue",
    "ReservoirUnion",
    "PatchReservoirUnion",
    "PluginRegistrySpec",
    "PluginRegistryEntrySpec",
    "SpecBaseModel",
]
