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


def _spec_or_dataref_discriminator(v: Any) -> str:
    """Route a section payload to the inline spec or the DataRef variant.

    Several top-level sections (``physics``, ``plugin_registry``, ``wells``,
    ``initial_conditions``, ``well_controls``, ``sim_params``, ``output``) are a
    JSON union of an inline strict spec and a :class:`DataRef` (path/URI
    reference). Without an explicit discriminator pydantic treats each as a
    smart union and, when the inline branch fails, also reports the DataRef
    branch's errors — flooding a single real mistake (e.g. an evaluator field
    misplaced on a property region, or a misplaced top-level section) with
    unrelated ``DataRef`` noise. ``kind``/``value`` are both required on
    ``DataRef`` and forbidden on every strict spec (each is ``extra="forbid"``
    and none declares those fields), so their presence is an unambiguous,
    zero-overlap discriminator. This predicate matches the builder's runtime
    DataRef detection so static routing agrees with build-time resolution.

    :param v: Section payload (dict, ``DataRef``, or a strict spec instance).
    :type v: Any
    :return: ``"dataref"`` for a reference payload, otherwise ``"spec"``.
    :rtype: str
    """
    if isinstance(v, dict):
        return "dataref" if ("kind" in v and "value" in v) else "spec"
    if isinstance(v, DataRef):
        return "dataref"
    return "spec"


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

    Numeric bounds intentionally omitted on union fields — see the note
    on :class:`StrictReservoirSpec` for why ``Field(gt=0, ...)`` on a
    ``ReservoirValue | DataRef`` union breaks DataRef validation.
    """

    dx: Annotated[ReservoirValue | None, Field(description="Cell size in x [m]")] = None
    dy: Annotated[ReservoirValue | None, Field(description="Cell size in y [m]")] = None
    dz: Annotated[ReservoirValue | None, Field(description="Cell size in z [m]")] = None
    permx: Annotated[
        ReservoirValue | None, Field(description="Permeability in x [mD]")
    ] = None
    permy: Annotated[
        ReservoirValue | None, Field(description="Permeability in y [mD]")
    ] = None
    permz: Annotated[
        ReservoirValue | None, Field(description="Permeability in z [mD]")
    ] = None
    poro: Annotated[
        ReservoirValue | None,
        Field(description="Porosity [fraction]"),
    ] = None
    depth: Annotated[
        ReservoirValue | None, Field(description="Reference depth [m]")
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

    Numeric bounds (``gt=0`` / ``ge=0`` / ``le=1``) deliberately do NOT
    appear here even though they would be natural for cell sizes,
    permeabilities, and porosity.  Pydantic applies field constraints
    member-by-member across the ``ReservoirValue | DataRef`` union and
    raises ``TypeError: Unable to apply constraint`` against the DataRef
    branch (which carries no numeric value at validation time), so any
    such constraint at this layer would break the documented DataRef
    workflow.  Validation of resolved numeric values happens in the
    builder after DataRef resolution.
    """

    # Override property fields to accept DataRef in addition to plain scalars/arrays
    dx: Annotated[
        ReservoirValue | None,
        Field(description="Cell size in x direction [m]"),
    ] = None
    dy: Annotated[
        ReservoirValue | None,
        Field(description="Cell size in y direction [m]"),
    ] = None
    dz: Annotated[
        ReservoirValue | None,
        Field(description="Cell size in z direction [m]"),
    ] = None
    permx: Annotated[
        ReservoirValue | None,
        Field(description="Permeability in x direction [mD]"),
    ] = None
    permy: Annotated[
        ReservoirValue | None,
        Field(description="Permeability in y direction [mD]"),
    ] = None
    permz: Annotated[
        ReservoirValue | None,
        Field(description="Permeability in z direction [mD]"),
    ] = None
    poro: Annotated[
        ReservoirValue | None,
        Field(description="Porosity [fraction]"),
    ] = None
    depth: Annotated[
        ReservoirValue | None,
        Field(description="Reference depth [m]"),
    ] = None
    hcap: Annotated[
        ReservoirValue | None,
        Field(description="Heat capacity [J/kg-K], 0 = non-thermal"),
    ] = None
    rcond: Annotated[
        ReservoirValue | None,
        Field(description="Rock thermal conductivity [W/m-K], 0 = non-thermal"),
    ] = None
    layers: Annotated[
        list[StrictReservoirLayerSpec] | None,
        Field(description="Layered overrides for per-cell properties"),
    ] = None
    # ``op_num`` and ``actnum`` are integer-typed in the native config; the
    # JSON layer widens them to also accept a DataRef so large per-cell index
    # arrays can be referenced by path instead of inlined in the JSON.
    op_num: Annotated[
        int | list[int] | DataRef | None,
        Field(description="Operator region number per cell (PVTNUM, SCALNUM, ...)"),
    ] = 0
    actnum: Annotated[
        int | list[int] | DataRef | None,
        Field(description="Active cell indicator per cell"),
    ] = 1


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
                    # Evaluator slots MUST nest under ``plugins`` (their names
                    # are forbidden directly on the region); see StrictPluginSlots.
                    "plugins": {
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


def _spec_or_dataref_union(spec_cls: Any) -> Any:
    """Build a discriminated ``spec | DataRef`` union for a top-level section.

    Mirrors :data:`ReservoirUnion`: the inline strict spec is tagged ``"spec"``
    and :class:`DataRef` is tagged ``"dataref"``, routed by
    :func:`_spec_or_dataref_discriminator`. Using an explicit discriminator
    (instead of a bare ``spec | DataRef`` smart union) keeps validation errors
    free of the irrelevant DataRef branch.

    :param spec_cls: The inline strict spec class for the section.
    :type spec_cls: Any
    :return: Discriminated ``Annotated`` union of *spec_cls* and ``DataRef``.
    :rtype: Any
    """
    return Annotated[
        Annotated[spec_cls, Tag("spec")] | Annotated[DataRef, Tag("dataref")],
        Discriminator(_spec_or_dataref_discriminator),
    ]


PhysicsUnion = _spec_or_dataref_union(StrictPhysicsSpec)
PluginRegistryUnion = _spec_or_dataref_union(PluginRegistrySpec)
WellsUnion = _spec_or_dataref_union(StrictWellsSpec)
InitialConditionsUnion = _spec_or_dataref_union(StrictInitialConditionsSpec)
WellControlsUnion = _spec_or_dataref_union(StrictWellControlsSpec)
SimParamsUnion = _spec_or_dataref_union(StrictSimParamsSpec)
OutputUnion = _spec_or_dataref_union(StrictOutputSpec)


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
        PluginRegistryUnion | None,
        Field(description="Local plugin registry configuration"),
    ] = None
    reservoir: Annotated[
        ReservoirUnion | None,
        Field(description="Reservoir configuration"),
    ] = None
    physics: Annotated[
        PhysicsUnion | None,
        Field(description="Physics configuration"),
    ] = None
    wells: Annotated[WellsUnion | None, Field(description="Wells configuration")] = None
    initial_conditions: Annotated[
        InitialConditionsUnion | None,
        Field(description="Initial conditions"),
    ] = None
    well_controls: Annotated[
        WellControlsUnion | None,
        Field(
            description=(
                "Default well controls applied to all wells. "
                "Per-well controls in wells[].controls take precedence "
                "over these defaults for individual wells."
            )
        ),
    ] = None
    sim_params: Annotated[
        SimParamsUnion | None,
        Field(description="Simulation parameters"),
    ] = None
    output: Annotated[
        OutputUnion | None,
        Field(description="Output configuration"),
    ] = None


# ---------------------------------------------------------------------------
# Patch* variants — auto-generated from Strict* via make_patch_model
# ---------------------------------------------------------------------------
# Parallel hierarchy to the Strict* specs where every field is optional so
# clients can send partial updates. Used by the MCP adapter to implement
# RFC-7396 JSON Merge Patch: the server validates incoming patches against
# Patch* models, merges them into the accumulated spec, then validates the
# result against Strict* before building. Purely an MCP/JSON-transport
# concern — native DARTS objects are always fully constructed.
#
# These were previously hand-maintained, which drifted from the Strict side
# whenever a new field was added. They are now generated by
# :func:`darts.api.patch_factory.make_patch_model`, which inherits each Strict
# class, makes every required field optional, and rewrites nested Strict
# references to their Patch counterparts via the ``type_map`` arg.
# Equivalence with the previous hand-written shape is verified per pair in
# ``tests/python/test_patch_factory.py``.
# ---------------------------------------------------------------------------

from darts.api.patch_factory import make_patch_model  # noqa: E402

PatchPluginSlots = make_patch_model(
    StrictPluginSlots,
    name="PatchPluginSlots",
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

PatchPropertyRegionSpec = make_patch_model(
    StrictPropertyRegionSpec,
    name="PatchPropertyRegionSpec",
    type_map={StrictPluginSlots: PatchPluginSlots},
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

PatchPhysicsSpec = make_patch_model(
    StrictPhysicsSpec,
    name="PatchPhysicsSpec",
    type_map={StrictPropertyRegionSpec: PatchPropertyRegionSpec},
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

PatchReservoirSpec = make_patch_model(
    StrictReservoirSpec,
    name="PatchReservoirSpec",
    # ``layers`` keeps ``list[StrictReservoirLayerSpec]`` because there is no
    # corresponding ``PatchReservoirLayerSpec`` exported today (see remaining
    # work item P6c in the design doc).
    json_schema_extra={
        "examples": [{"nx": 20, "ny": 1, "nz": 1, "dx": 5.0, "dy": 10.0, "dz": 10.0}]
    },
)

PatchCPGReservoirSpec = make_patch_model(
    StrictCPGReservoirSpec,
    name="PatchCPGReservoirSpec",
    json_schema_extra={
        "examples": [
            {"type": "cpg", "grid_file": "meshes/brugge/grid.grdecl"},
            {"boundary_volume": 1e10},
        ]
    },
)

PatchReservoirUnion = Annotated[
    Annotated[PatchReservoirSpec, Tag("structured")]
    | Annotated[PatchCPGReservoirSpec, Tag("cpg")]
    | Annotated[DataRef, Tag("dataref")],
    Discriminator(_reservoir_discriminator),
]

PatchWellControlsSpec = make_patch_model(
    StrictWellControlsSpec,
    name="PatchWellControlsSpec",
    json_schema_extra={"examples": [{"inj_bhp": 150.0, "rate_type": "MOLAR_RATE"}]},
)

PatchWellPerforation = make_patch_model(
    StrictWellPerforation,
    name="PatchWellPerforation",
    json_schema_extra={"examples": [{"ijk": [1, 1, 1], "skin": 2.0}]},
)

PatchWellSpec = make_patch_model(
    StrictWellSpec,
    name="PatchWellSpec",
    type_map={
        StrictWellPerforation: PatchWellPerforation,
        StrictWellControlsSpec: PatchWellControlsSpec,
    },
    json_schema_extra={
        "examples": [
            {"name": "P1", "controls": {"prod_bhp": 50.0}},
            {"perforations": [{"ijk": [2, 1, 1]}]},
        ]
    },
)

PatchWellsSpec = make_patch_model(
    StrictWellsSpec,
    name="PatchWellsSpec",
    type_map={StrictWellSpec: PatchWellSpec},
    json_schema_extra={
        "examples": [{"wells": [{"name": "I1", "perforations": [{"ijk": [1, 1, 1]}]}]}]
    },
)

PatchInitialConditionsSpec = make_patch_model(
    StrictInitialConditionsSpec,
    name="PatchInitialConditionsSpec",
    json_schema_extra={"examples": [{"by_array": {"pressure": 60.0, "CO2": 0.05}}]},
)

PatchSimParamsSpec = make_patch_model(
    StrictSimParamsSpec,
    name="PatchSimParamsSpec",
    json_schema_extra={
        "examples": [{"runtime": 500.0, "max_ts": 2.0, "tol_newton": 0.02}]
    },
)

PatchOutputSpec = make_patch_model(
    StrictOutputSpec,
    name="PatchOutputSpec",
    json_schema_extra={"examples": [{"folder": "output", "precision": "d"}]},
)

# PatchModelSpec needs one manual override on top of the factory output:
# its ``reservoir`` field uses ``PatchReservoirUnion`` (an
# ``Annotated[..., Tag(...)]`` discriminated union) that ``make_patch_model``
# does not replicate — Tag-based unions are not the kind of nested type the
# factory's ``type_map`` can substitute through.
_PatchModelSpecBase = make_patch_model(
    StrictModelSpec,
    name="_PatchModelSpecBase",
    type_map={
        StrictPhysicsSpec: PatchPhysicsSpec,
        StrictWellsSpec: PatchWellsSpec,
        StrictInitialConditionsSpec: PatchInitialConditionsSpec,
        StrictSimParamsSpec: PatchSimParamsSpec,
        StrictOutputSpec: PatchOutputSpec,
        StrictWellControlsSpec: PatchWellControlsSpec,
    },
)


class PatchModelSpec(_PatchModelSpecBase):
    """Partial model spec for RFC-7396 merge-patch updates."""

    model_config = ConfigDict(
        extra="forbid",
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
