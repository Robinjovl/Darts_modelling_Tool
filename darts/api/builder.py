"""
Stateless model builder that applies a validated ModelSpec to a DARTS model.

ModelBuilder translates each section of a ModelSpec (reservoir, physics,
wells, initial conditions, simulation parameters, output) into the
corresponding DARTS object graph.  It operates on any object satisfying
DartsModelProtocol and does not hold state between calls, making it safe
for reuse across multiple models or adapter instances.

Each ``apply_*`` method is a public static entry point that can be called
independently to construct a single primitive from its JSON spec, enabling
composable test fixtures and standalone primitive construction without a
full ModelSpec envelope.
"""

import logging
import os
from typing import Any, Protocol, runtime_checkable

from darts.api.data_refs import resolve_data_ref
from darts.api.schemas import (
    CPGReservoirSpec,
    DataRef,
    InitialConditionsSpec,
    ModelSpec,
    OutputSpec,
    PhysicsSpec,
    PluginRegistrySpec,
    PluginSlots,
    ReservoirSpec,
    SimParamsSpec,
    WellControlsSpec,
    WellsSpec,
)
from darts.api.type_registry import (
    TYPE_REGISTRY,
    PluginInstance,
    PropertyContainerConfig,
    load_local_plugin_registry,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class DartsModelProtocol(Protocol):
    """Minimal interface expected by ModelBuilder on the *model* argument."""

    timer: Any
    reservoir: Any
    physics: Any

    def set_sim_params(self, **kwargs: Any) -> None: ...


class ModelBuilder:
    """Stateless builder: full-model ``apply`` or per-section ``apply_*``."""

    # ------------------------------------------------------------------
    # Full-model orchestrator
    # ------------------------------------------------------------------

    @staticmethod
    def apply(
        spec: ModelSpec,
        model: DartsModelProtocol,
        *,
        base_path: str | None = None,
        object_store: dict[str, Any] | None = None,
    ) -> None:
        """Apply every non-None section of *spec* to *model* in order."""
        if getattr(spec, "plugin_registry", None):
            preg = ModelBuilder.resolve_section(
                spec.plugin_registry, PluginRegistrySpec, base_path, object_store
            )
            ModelBuilder.apply_plugin_registry(preg, base_path=base_path)

        if spec.reservoir:
            res_raw = ModelBuilder.resolve_section(
                spec.reservoir, None, base_path, object_store
            )
            res = ModelBuilder.validate_reservoir_spec(res_raw)
            ModelBuilder.apply_reservoir(
                res, model, base_path=base_path, object_store=object_store
            )

        if spec.physics:
            phy = ModelBuilder.resolve_section(
                spec.physics, PhysicsSpec, base_path, object_store
            )
            ModelBuilder.apply_physics(phy, model, base_path=base_path)

        if spec.wells:
            wells = ModelBuilder.resolve_section(
                spec.wells, WellsSpec, base_path, object_store
            )
            ModelBuilder.apply_wells(wells, model)

        if spec.initial_conditions:
            ics = ModelBuilder.resolve_section(
                spec.initial_conditions, InitialConditionsSpec, base_path, object_store
            )
            ModelBuilder.apply_initial_conditions(ics, model)

        if spec.sim_params:
            sp = ModelBuilder.resolve_section(
                spec.sim_params, SimParamsSpec, base_path, object_store
            )
            ModelBuilder.apply_sim_params(sp, model)

        if spec.output:
            out = ModelBuilder.resolve_section(
                spec.output, OutputSpec, base_path, object_store
            )
            ModelBuilder.apply_output(out, model)

        if getattr(spec, "well_controls", None):
            wc = ModelBuilder.resolve_section(
                spec.well_controls, WellControlsSpec, base_path, object_store
            )
            ModelBuilder.apply_well_controls(wc, model)

    # ------------------------------------------------------------------
    # Section resolution & validation
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_section(
        section: Any,
        model_cls: type[Any] | None,
        base_path: str | None,
        object_store: dict[str, Any] | None,
    ) -> Any:
        """Resolve DataRefs and optionally validate via *model_cls*."""
        if isinstance(section, DataRef):
            section = resolve_data_ref(
                section, base_path=base_path, object_store=object_store
            )
        elif isinstance(section, dict) and "kind" in section and "value" in section:
            section = resolve_data_ref(
                section, base_path=base_path, object_store=object_store
            )
        if model_cls is None:
            return section
        if isinstance(section, dict):
            return model_cls.model_validate(section)
        return section

    @staticmethod
    def validate_reservoir_spec(section: Any) -> Any:
        """Coerce a raw dict into the appropriate reservoir spec model."""
        if hasattr(section, "type"):
            return section
        if not isinstance(section, dict):
            raise TypeError("Unsupported reservoir section representation")

        rtype = section.get("type")
        target_cls: Any = CPGReservoirSpec if rtype == "cpg" else ReservoirSpec
        return target_cls.model_validate(section)

    # ------------------------------------------------------------------
    # Per-section apply methods (public API)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_plugin_registry(
        preg: PluginRegistrySpec, *, base_path: str | None = None
    ) -> None:
        """Register local plugin types from *preg*."""
        load_local_plugin_registry(preg, base_path=base_path)

    @staticmethod
    def apply_reservoir(
        r: Any,
        model: DartsModelProtocol,
        *,
        base_path: str | None = None,
        object_store: dict[str, Any] | None = None,
    ) -> None:
        """Construct a reservoir object and assign it to *model.reservoir*."""
        if isinstance(r, dict):
            r = ModelBuilder.validate_reservoir_spec(r)

        if getattr(r, "type", None) == "cpg":
            ModelBuilder._apply_cpg_reservoir(r, model, base_path=base_path)
            return

        # Default branch: structured reservoir
        rtype = getattr(r, "type", None)
        if rtype != "structured":
            raise ValueError(f"Unsupported reservoir type: {rtype!r}")

        from darts.reservoirs.struct_reservoir import StructReservoir

        def _resolve_val(val: Any) -> Any:
            if isinstance(val, DataRef) or (
                isinstance(val, dict) and "kind" in val and "value" in val
            ):
                return resolve_data_ref(
                    val, base_path=base_path, object_store=object_store
                )
            return val

        def _maybe_array(val: Any) -> Any:
            if isinstance(val, list | tuple):
                import numpy as np

                return np.asarray(val)
            return val

        layers = getattr(r, "layers", None)
        layer_specs: list[dict[str, Any]] = []
        if layers:
            for layer in layers:
                if hasattr(layer, "model_dump"):
                    layer_specs.append(layer.model_dump(exclude_none=True))
                elif isinstance(layer, dict):
                    layer_specs.append(
                        {k: v for k, v in layer.items() if v is not None}
                    )
                else:
                    raise TypeError("Unsupported layer specification type")
            total_cells = int(r.nx * r.ny * r.nz)

            def _is_list_like(val: Any) -> bool:
                if isinstance(val, list | tuple):
                    return True
                try:
                    import numpy as np

                    return isinstance(val, np.ndarray)
                except Exception:
                    return False

            def _layers_have_field(key: str) -> bool:
                return any(layer.get(key) is not None for layer in layer_specs)

            def _expand_layered_value(key: str, base_val: Any) -> Any:
                if not _layers_have_field(key):
                    return base_val
                if _is_list_like(base_val):
                    if any(layer.get(key) is None for layer in layer_specs):
                        raise ValueError(
                            f"Layered '{key}' requires a scalar default when some layers omit it."
                        )
                out: list[Any] = []
                for layer in layer_specs:
                    count = int(layer.get("count", 0))
                    if count <= 0:
                        raise ValueError("Layer count must be >= 1")
                    val = layer.get(key, None)
                    val = _resolve_val(val) if val is not None else base_val
                    if val is None:
                        raise ValueError(
                            f"Layered '{key}' is missing for a layer and no base value was provided."
                        )
                    if _is_list_like(val):
                        if len(val) != count:
                            raise ValueError(
                                f"Layered '{key}' list length must equal count ({count})."
                            )
                        out.extend(list(val))
                    else:
                        out.extend([val] * count)
                if len(out) != total_cells:
                    raise ValueError(
                        f"Layered '{key}' produced {len(out)} values, expected {total_cells}."
                    )
                return out

        kwargs: dict[str, Any] = {}
        for key in [
            "nx",
            "ny",
            "nz",
            "dx",
            "dy",
            "dz",
            "permx",
            "permy",
            "permz",
            "poro",
            "depth",
            "hcap",
            "rcond",
        ]:
            val = _resolve_val(getattr(r, key))
            if layer_specs:
                val = _expand_layered_value(key, val)
            if val is not None:
                kwargs[key] = _maybe_array(val)
        required = [
            "nx",
            "ny",
            "nz",
            "dx",
            "dy",
            "dz",
            "permx",
            "permy",
            "permz",
            "poro",
        ]
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(
                "Reservoir spec missing required fields: " + ", ".join(missing)
            )
        model.reservoir = StructReservoir(model.timer, **kwargs)

    @staticmethod
    def apply_physics(
        p: PhysicsSpec, model: DartsModelProtocol, *, base_path: str | None = None
    ) -> None:
        """Instantiate physics engine and property regions on *model*."""
        if p.plugin is None:
            raise ValueError("physics.plugin is required")
        if p.components is None or p.phases is None:
            raise ValueError("physics.components and physics.phases are required")

        # Instantiate physics via registry
        pinst = ModelBuilder._as_plugin_instance(p.plugin)
        entry = TYPE_REGISTRY.get(pinst.type_id)
        cfg = entry.config_model(**pinst.config)
        # Resolve relative data files from ModelSpec location (e.g., BlackOil pvt_path).
        if base_path and hasattr(cfg, "pvt_path"):
            pvt_path = cfg.pvt_path
            if isinstance(pvt_path, str) and pvt_path and not os.path.isabs(pvt_path):
                cfg.pvt_path = os.path.join(base_path, pvt_path)
        physics = entry.constructor(
            cfg, components=p.components, phases=p.phases, timer=model.timer
        )
        model.physics = physics

        # Property regions
        if p.property_regions:
            for region in p.property_regions:
                if region.property_container is None:
                    raise ValueError(
                        f"property_container is required for region {region.region}"
                    )
                pci = ModelBuilder._as_plugin_instance(region.property_container)
                pc_entry = TYPE_REGISTRY.get(pci.type_id)
                pc_cfg = pc_entry.config_model(**pci.config)
                # override from physics section if omitted
                if isinstance(pc_cfg, PropertyContainerConfig):
                    if pc_cfg.components_name is None:
                        pc_cfg.components_name = p.components
                    if pc_cfg.phases_name is None:
                        pc_cfg.phases_name = p.phases
                pc = pc_entry.constructor(pc_cfg)

                # attach evaluator plugins (flash, density, viscosity, rel_perm)
                if region.plugins:
                    plugins = region.plugins
                    # type guard
                    if not isinstance(plugins, PluginSlots):
                        if hasattr(plugins, "model_dump"):
                            plugins = PluginSlots(**plugins.model_dump())
                        elif hasattr(plugins, "dict"):
                            plugins = PluginSlots(**plugins.dict())
                        elif isinstance(plugins, dict):
                            plugins = PluginSlots(**plugins)
                        else:
                            raise TypeError("Unsupported plugin slots representation")
                    # flash_ev: single plugin
                    if getattr(plugins, "flash_ev", None) is not None:
                        fi = ModelBuilder._as_plugin_instance(plugins.flash_ev)
                        f_entry = TYPE_REGISTRY.get(fi.type_id)
                        f_cfg = f_entry.config_model(**fi.config)
                        # pass fluid component count (exclude solids if specified)
                        fluid_nc = len(p.components)
                        nc_sol = getattr(pc_cfg, "nc_sol", None)
                        if nc_sol is not None:
                            fluid_nc = max(1, fluid_nc - int(nc_sol))
                        obj = f_entry.constructor(
                            f_cfg,
                            nc=fluid_nc,
                            epsilon=getattr(f_cfg, "epsilon", 1e-8),
                        )
                        pc.flash_ev = obj
                    # Per-phase evaluator slots
                    _phase_slots = [
                        "density_ev",
                        "viscosity_ev",
                        "enthalpy_ev",
                        "conductivity_ev",
                        "rel_perm_ev",
                    ]
                    for slot_name in _phase_slots:
                        slot_dict = getattr(plugins, slot_name, None)
                        if slot_dict is None:
                            continue
                        result = {}
                        for phase, inst in slot_dict.items():
                            ModelBuilder._validate_phase_key(phase, p.phases)
                            pi = ModelBuilder._as_plugin_instance(inst)
                            p_entry = TYPE_REGISTRY.get(pi.type_id)
                            p_cfg = p_entry.config_model(**pi.config)
                            result[phase] = p_entry.constructor(p_cfg)
                        setattr(pc, slot_name, result)
                    # diffusion_ev: map per phase (with scalar expansion)
                    if getattr(plugins, "diffusion_ev", None) is not None:
                        pc.diffusion_ev = {}
                        for phase, inst in plugins.diffusion_ev.items():
                            ModelBuilder._validate_phase_key(phase, p.phases)
                            di2 = ModelBuilder._as_plugin_instance(inst)
                            d2_entry = TYPE_REGISTRY.get(di2.type_id)
                            d2_cfg = d2_entry.config_model(**di2.config)
                            # Expand scalar diffusion value to per-component vector
                            val = d2_cfg.value
                            if isinstance(val, int | float):
                                ncomp = len(p.components)
                                d2_cfg.value = [float(val)] * ncomp
                            pc.diffusion_ev[phase] = d2_entry.constructor(d2_cfg)
                    # kinetic_rate_ev: map by integer key
                    if getattr(plugins, "kinetic_rate_ev", None) is not None:
                        if (
                            not hasattr(pc, "kinetic_rate_ev")
                            or pc.kinetic_rate_ev is None
                        ):
                            pc.kinetic_rate_ev = {}
                        for idx_str, inst in plugins.kinetic_rate_ev.items():
                            try:
                                idx = int(idx_str)
                            except ValueError as err:
                                raise ValueError(
                                    f"kinetic_rate_ev key must be an integer, got {idx_str!r}"
                                ) from err
                            ki = ModelBuilder._as_plugin_instance(inst)
                            k_entry = TYPE_REGISTRY.get(ki.type_id)
                            k_cfg = k_entry.config_model(**ki.config)
                            pc.kinetic_rate_ev[idx] = k_entry.constructor(k_cfg)

                model.physics.add_property_region(pc, region=region.region)

    @staticmethod
    def apply_wells(w: WellsSpec, model: DartsModelProtocol) -> None:
        """Store well specs for deferred application during ``model.init()``."""
        model._wells_spec = w  # type: ignore[attr-defined]

    @staticmethod
    def apply_initial_conditions(
        ic: InitialConditionsSpec, model: DartsModelProtocol
    ) -> None:
        """Store initial conditions for deferred application during ``model.init()``."""
        model._initial_conditions_spec = ic  # type: ignore[attr-defined]

    @staticmethod
    def apply_sim_params(sp: SimParamsSpec, model: DartsModelProtocol) -> None:
        """Apply simulation parameters to *model*."""
        kwargs: dict[str, Any] = {}
        for k in [
            "first_ts",
            "mult_ts",
            "max_ts",
            "runtime",
            "tol_newton",
            "tol_linear",
            "it_newton",
            "it_linear",
            "line_search",
        ]:
            v = getattr(sp, k)
            if v is not None:
                kwargs[k] = v
        # Map optional newton_type
        newton_type = getattr(sp, "newton_type", None)
        if newton_type:
            from darts.engines import sim_params as spm

            if newton_type == "newton_local_chop":
                kwargs["newton_type"] = spm.newton_local_chop
            else:
                logger.warning("Unknown newton_type %r, ignoring", newton_type)
        model.set_sim_params(**kwargs)
        # Post DataTS tweaks
        if getattr(sp, "newton_tol_stationary", None) is not None:
            model.data_ts.newton_tol_stationary = sp.newton_tol_stationary  # type: ignore[attr-defined]
        if getattr(sp, "min_line_search_update", None) is not None:
            model.data_ts.min_line_search_update = sp.min_line_search_update  # type: ignore[attr-defined]

    @staticmethod
    def apply_output(out: OutputSpec, model: DartsModelProtocol) -> None:
        """Store output config for deferred application after ``model.init()``."""
        model._output_spec = out  # type: ignore[attr-defined]

    @staticmethod
    def apply_well_controls(wc: Any, model: DartsModelProtocol) -> None:
        """Store well controls for deferred application at runtime."""
        model._well_controls_spec = wc  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Internal helpers (not part of the public per-section API)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cpg_reservoir(
        r: CPGReservoirSpec,
        model: DartsModelProtocol,
        *,
        base_path: str | None = None,
    ) -> None:
        from darts.reservoirs.cpg_reservoir import (
            CPG_Reservoir,
            check_arrays,
            read_arrays,
        )
        from darts.tools.keyword_file_tools import compressed_file

        def _resolve_path(path: str) -> str:
            if os.path.isabs(path):
                return path
            if base_path:
                return os.path.join(base_path, path)
            return path

        grid_file = _resolve_path(r.grid_file)
        prop_file = _resolve_path(r.prop_file)
        fault_file = _resolve_path(r.fault_file) if r.fault_file else None

        compressed_file(grid_file)
        compressed_file(prop_file)

        arrays = read_arrays(gridfile=grid_file, propfile=prop_file)
        check_arrays(arrays)

        if r.min_poro is not None and "PORO" in arrays and "ACTNUM" in arrays:
            arrays["ACTNUM"][arrays["PORO"] < r.min_poro] = 0
        if r.min_perm is not None:
            for key in ("PERMX", "PERMY", "PERMZ"):
                if key in arrays:
                    arrays[key][arrays[key] < r.min_perm] = r.min_perm

        reservoir = CPG_Reservoir(
            model.timer,
            arrays=arrays,
            faultfile=fault_file,
            minpv=r.minpv if r.minpv is not None else 0.0,
        )
        reservoir.discretize()
        reservoir.input_arrays = arrays

        if r.boundary_volume is not None:
            bv = r.boundary_volume
            reservoir.set_boundary_volume(
                xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv
            )
            reservoir.apply_volume_depth()

        model.reservoir = reservoir

    @staticmethod
    def _as_plugin_instance(obj: Any) -> PluginInstance:
        if isinstance(obj, PluginInstance):
            return obj
        # pydantic model from type_registry.PluginInstance
        if hasattr(obj, "type_id") and hasattr(obj, "config"):
            cfg = obj.config
            if hasattr(cfg, "model_dump"):
                cfg = cfg.model_dump()
            elif hasattr(cfg, "dict"):
                cfg = cfg.dict()
            return PluginInstance(type_id=obj.type_id, config=cfg)
        if isinstance(obj, dict):
            return PluginInstance(**obj)
        raise TypeError("Unsupported plugin instance representation")

    @staticmethod
    def _validate_phase_key(phase: str, phases: list[str]) -> None:
        if phase not in phases:
            raise ValueError(f"Unknown phase key '{phase}', expected one of {phases}")
