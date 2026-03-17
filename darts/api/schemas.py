from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from darts.api.type_registry import PluginInstance


class SpecBaseModel(BaseModel):
    """Strict base model used by all JSON schema specs."""

    model_config = ConfigDict(extra="forbid")


class DataRef(SpecBaseModel):
    """Reference to external data or stored Python objects."""

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


ReservoirValue = float | list[float] | DataRef


def _reservoir_discriminator(v: Any) -> str:
    """Discriminate reservoir spec variants by the ``type`` field."""
    if isinstance(v, dict):
        if "kind" in v and "value" in v:
            return "dataref"
        return v.get("type", "structured")
    if isinstance(v, DataRef):
        return "dataref"
    return getattr(v, "type", "structured")


class PluginRegistryEntrySpec(SpecBaseModel):
    """Register a local plugin type."""

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
    """Local plugin registry entries loaded from JSON."""

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


class StrictReservoirLayerSpec(SpecBaseModel):
    """Layered overrides for structured reservoirs."""

    model_config = ConfigDict(
        title="StrictReservoirLayerSpec",
        json_schema_extra={
            "examples": [
                {"count": 100, "dz": 6, "permx": 500, "permy": 500, "permz": 80}
            ]
        },
    )

    count: Annotated[
        int, Field(ge=1, description="Number of cells in this layer (KJI order)")
    ]
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
        ReservoirValue | None, Field(ge=0, le=1, description="Porosity [fraction]")
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


class StrictReservoirSpec(SpecBaseModel):
    """Reservoir geometry and rock properties."""

    model_config = ConfigDict(
        title="StrictReservoirSpec",
        json_schema_extra={
            "examples": [
                {
                    "type": "structured",
                    "nx": 1000,
                    "ny": 1,
                    "nz": 1,
                    "dx": 1.0,
                    "dy": 10.0,
                    "dz": 10.0,
                    "permx": 100.0,
                    "permy": 100.0,
                    "permz": 10.0,
                    "poro": 0.3,
                    "depth": 1000.0,
                }
            ]
        },
    )

    type: Annotated[
        Literal["structured"],
        Field(description="Reservoir type (structured only in v1)"),
    ]
    nx: Annotated[int, Field(ge=1, description="Number of cells in x direction")]
    ny: Annotated[int, Field(ge=1, description="Number of cells in y direction")]
    nz: Annotated[int, Field(ge=1, description="Number of cells in z direction")]
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
        float | None,
        Field(ge=0, description="Heat capacity [J/kg-K], 0 = non-thermal"),
    ] = None
    rcond: Annotated[
        float | None,
        Field(ge=0, description="Rock thermal conductivity [W/m-K], 0 = non-thermal"),
    ] = None
    layers: Annotated[
        list[StrictReservoirLayerSpec] | None,
        Field(description="Layered overrides for per-cell properties"),
    ] = None


class StrictCPGReservoirSpec(SpecBaseModel):
    """Corner-point (CPG) reservoir loaded from GRDECL-like files."""

    model_config = ConfigDict(
        title="StrictCPGReservoirSpec",
        json_schema_extra={
            "examples": [
                {
                    "type": "cpg",
                    "grid_file": "meshes/brugge/grid.grdecl",
                    "prop_file": "meshes/brugge/reservoir.in",
                    "minpv": 1e-5,
                    "min_poro": 1e-5,
                    "boundary_volume": 1e10,
                }
            ]
        },
    )

    type: Annotated[
        Literal["cpg"],
        Field(description="Reservoir type (corner-point geometry)"),
    ]
    grid_file: Annotated[str, Field(description="Path to GRDECL grid file")]
    prop_file: Annotated[
        str, Field(description="Path to reservoir property file (GRDECL-like)")
    ]
    fault_file: Annotated[
        str | None,
        Field(description="Optional file with fault transmissibility multipliers"),
    ] = None
    minpv: Annotated[
        float | None,
        Field(ge=0, description="Minimum pore volume threshold for active cells [m3]"),
    ] = None
    min_poro: Annotated[
        float | None,
        Field(ge=0, le=1, description="Optional porosity cutoff for ACTNUM filtering"),
    ] = None
    min_perm: Annotated[
        float | None,
        Field(ge=0, description="Optional lower bound for PERMX/PERMY/PERMZ [mD]"),
    ] = None
    boundary_volume: Annotated[
        float | None,
        Field(
            gt=0, description="Optional lateral boundary volume assigned to edge cells"
        ),
    ] = None


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


class StrictWellControlsSpec(SpecBaseModel):
    """Well controls configuration (per-well or top-level)."""

    model_config = ConfigDict(
        title="StrictWellControlsSpec",
        json_schema_extra={
            "examples": [
                {"inj_bhp": 140.0, "prod_bhp": 50.0, "inj_composition": [1.0, 0.0, 0.0]}
            ]
        },
    )

    inj_bhp: Annotated[
        float | None,
        Field(ge=0, description="Injector bottom-hole pressure [bar]"),
    ] = None
    prod_bhp: Annotated[
        float | None,
        Field(ge=0, description="Producer bottom-hole pressure [bar]"),
    ] = None
    inj_composition: Annotated[
        list[float] | None,
        Field(description="Injector composition (length = nc or nc-1)"),
    ] = None
    inj_temp: Annotated[
        float | None, Field(gt=0, description="Injector temperature [K]")
    ] = None
    inj_rate: Annotated[
        float | None, Field(ge=0, description="Injector target rate")
    ] = None
    rate_type: Annotated[
        Literal[
            "MOLAR_RATE",
            "MASS_RATE",
            "VOLUMETRIC_RATE",
            "ADVECTIVE_HEAT_RATE",
        ]
        | None,
        Field(description="Rate control type for injector"),
    ] = None
    phase_name: Annotated[
        str | None, Field(description="Phase name for rate-controlled injector")
    ] = None


class StrictWellPerforation(SpecBaseModel):
    """Perforation definition with optional well parameters."""

    model_config = ConfigDict(
        title="StrictWellPerforation",
        json_schema_extra={"examples": [{"ijk": [1, 1, 1], "well_radius": 0.0762}]},
    )

    ijk: Annotated[
        list[int],
        Field(description="[i,j,k] indices (1-based)", min_length=3, max_length=3),
    ]
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
    well_radius: Annotated[float | None, Field(gt=0, description="Well radius")] = None
    skin: Annotated[float | None, Field(description="Skin factor")] = None


class StrictWellSpec(SpecBaseModel):
    """Well specification with perforations and optional controls."""

    model_config = ConfigDict(
        title="StrictWellSpec",
        json_schema_extra={
            "examples": [
                {
                    "name": "I1",
                    "perforations": [{"ijk": [1, 1, 1]}],
                    "controls": {"inj_bhp": 140.0},
                }
            ]
        },
    )

    name: Annotated[str, Field(description="Well name")]
    perforations: Annotated[
        list[StrictWellPerforation],
        Field(min_length=1, description="List of perforations"),
    ]
    controls: Annotated[
        StrictWellControlsSpec | None,
        Field(
            description=(
                "Per-well controls that override top-level well_controls defaults"
            )
        ),
    ] = None


class StrictWellsSpec(SpecBaseModel):
    """List of wells."""

    model_config = ConfigDict(
        title="StrictWellsSpec",
        json_schema_extra={
            "examples": [
                {
                    "wells": [
                        {"name": "I1", "perforations": [{"ijk": [1, 1, 1]}]},
                        {"name": "P1", "perforations": [{"ijk": [1000, 1, 1]}]},
                    ]
                }
            ]
        },
    )

    wells: Annotated[
        list[StrictWellSpec],
        Field(min_length=1, description="List of well specifications"),
    ]


class StrictInitialConditionsSpec(SpecBaseModel):
    """Initial conditions mapping for physics variables."""

    model_config = ConfigDict(
        title="StrictInitialConditionsSpec",
        json_schema_extra={
            "examples": [{"by_array": {"pressure": 50.0, "CO2": 0.1, "C1": 0.2}}]
        },
    )

    by_array: Annotated[
        dict[str, Any],
        Field(description="Initial state variables (pressure, compositions, etc.)"),
    ]


class StrictSimParamsSpec(SpecBaseModel):
    """Simulation parameters configuration."""

    model_config = ConfigDict(
        title="StrictSimParamsSpec",
        json_schema_extra={
            "examples": [
                {
                    "first_ts": 0.001,
                    "mult_ts": 2.0,
                    "max_ts": 1.0,
                    "runtime": 1000.0,
                    "tol_newton": 0.01,
                    "tol_linear": 0.001,
                    "it_newton": 10,
                    "it_linear": 50,
                }
            ]
        },
    )

    first_ts: Annotated[
        float | None, Field(gt=0, description="First time step [d]")
    ] = None
    mult_ts: Annotated[
        float | None, Field(gt=0, description="Time step multiplier")
    ] = None
    max_ts: Annotated[
        float | None, Field(gt=0, description="Maximum time step [d]")
    ] = None
    runtime: Annotated[
        float | None, Field(gt=0, description="Simulation runtime [d]")
    ] = None
    tol_newton: Annotated[float | None, Field(gt=0, description="Newton tolerance")] = (
        None
    )
    tol_linear: Annotated[float | None, Field(gt=0, description="Linear tolerance")] = (
        None
    )
    it_newton: Annotated[
        int | None, Field(ge=1, description="Maximum number of Newton iterations")
    ] = None
    it_linear: Annotated[
        int | None, Field(ge=1, description="Maximum number of linear iterations")
    ] = None
    line_search: Annotated[
        bool | None, Field(description="Enable line search for Newton solver")
    ] = None
    newton_tol_stationary: Annotated[
        float | None, Field(gt=0, description="Stationary Newton tolerance")
    ] = None
    newton_type: Annotated[
        Literal["newton_local_chop", "default"] | None,
        Field(description="Newton type identifier"),
    ] = None
    min_line_search_update: Annotated[
        float | None, Field(gt=0, description="Minimum line search update")
    ] = None


class StrictOutputSpec(SpecBaseModel):
    """Output configuration."""

    model_config = ConfigDict(
        title="StrictOutputSpec",
        json_schema_extra={
            "examples": [{"folder": "output", "precision": "d", "save_initial": True}]
        },
    )

    folder: Annotated[str | None, Field(description="Output folder")] = None
    precision: Annotated[
        Literal["s", "d"] | None,
        Field(description="Output precision (s=single, d=double)"),
    ] = None
    save_initial: Annotated[
        bool | None, Field(description="Save initial state to output")
    ] = None


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


class PatchPluginSlots(StrictPluginSlots):
    """Optional evaluator plugin overrides."""

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
