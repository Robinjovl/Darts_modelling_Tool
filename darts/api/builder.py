from __future__ import annotations

from typing import Any

from darts.api.model_spec import (
    InitialConditionsSpec,
    ModelSpec,
    OutputSpec,
    PhysicsSpec,
    PluginSlots,
    ReservoirSpec,
    SimParamsSpec,
    WellsSpec,
)
from darts.api.type_registry import (
    TYPE_REGISTRY,
    PluginInstance,
    PropertyContainerConfig,
)


class ModelBuilder:
    @staticmethod
    def apply(spec: ModelSpec, model: Any) -> None:
        if spec.reservoir:
            ModelBuilder._apply_reservoir(spec.reservoir, model)

        if spec.physics:
            ModelBuilder._apply_physics(spec.physics, model)

        if spec.wells:
            ModelBuilder._apply_wells(spec.wells, model)

        if spec.initial_conditions:
            ModelBuilder._apply_initial_conditions(spec.initial_conditions, model)

        if spec.sim_params:
            ModelBuilder._apply_sim_params(spec.sim_params, model)

        if spec.output:
            ModelBuilder._apply_output(spec.output, model)

        if getattr(spec, 'well_controls', None):
            ModelBuilder._apply_well_controls(spec.well_controls, model)

    @staticmethod
    def _apply_reservoir(r: ReservoirSpec, model: Any) -> None:
        # Only structured reservoir supported in v1
        assert r.type == "structured", "Only structured reservoir is supported in v1"
        from darts.reservoirs.struct_reservoir import StructReservoir

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
        ]:
            val = getattr(r, key)
            if val is not None:
                kwargs[key] = val
        model.reservoir = StructReservoir(model.timer, **kwargs)

    @staticmethod
    def _apply_physics(p: PhysicsSpec, model: Any) -> None:
        assert p.plugin is not None, "physics.plugin is required"
        assert p.components is not None and p.phases is not None, (
            "physics.components and physics.phases are required"
        )

        # Instantiate physics via registry
        pinst = ModelBuilder._as_plugin_instance(p.plugin)
        entry = TYPE_REGISTRY.get(pinst.type_id)
        cfg = entry.config_model(**pinst.config)
        physics = entry.constructor(
            cfg, components=p.components, phases=p.phases, timer=model.timer
        )
        model.physics = physics

        # Property regions
        if p.property_regions:
            for region in p.property_regions:
                # property container
                assert region.property_container is not None, (
                    "property_container is required"
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
                    # Mw optional; warn if missing length when provided
                pc = pc_entry.constructor(pc_cfg)

                # attach evaluator plugins (flash, density, viscosity, rel_perm)
                if region.plugins:
                    plugins = region.plugins
                    # type guard
                    if not isinstance(plugins, PluginSlots):
                        # Pydantic will usually ensure this, but guard for dict input
                        plugins = PluginSlots(**plugins)
                    # flash_ev: single plugin
                    if getattr(plugins, "flash_ev", None) is not None:
                        fi = ModelBuilder._as_plugin_instance(plugins.flash_ev)
                        f_entry = TYPE_REGISTRY.get(fi.type_id)
                        f_cfg = f_entry.config_model(**fi.config)
                        # pass fluid component count (exclude solids if specified in property container config)
                        fluid_nc = len(p.components)
                        try:
                            nc_sol = getattr(pc_cfg, "nc_sol", None)
                            if nc_sol is not None:
                                fluid_nc = max(1, fluid_nc - int(nc_sol))
                        except Exception:
                            pass
                        obj = f_entry.constructor(
                            f_cfg,
                            nc=fluid_nc,
                            epsilon=getattr(f_cfg, "epsilon", 1e-8),
                        )
                        pc.flash_ev = obj
                    # density_ev: map per phase
                    if getattr(plugins, "density_ev", None) is not None:
                        pc.density_ev = {}
                        for phase, inst in plugins.density_ev.items():
                            ModelBuilder._validate_phase_key(phase, p.phases)
                            di = ModelBuilder._as_plugin_instance(inst)
                            d_entry = TYPE_REGISTRY.get(di.type_id)
                            d_cfg = d_entry.config_model(**di.config)
                            pc.density_ev[phase] = d_entry.constructor(d_cfg)
                    # viscosity_ev: map per phase
                    if getattr(plugins, "viscosity_ev", None) is not None:
                        pc.viscosity_ev = {}
                        for phase, inst in plugins.viscosity_ev.items():
                            ModelBuilder._validate_phase_key(phase, p.phases)
                            vi = ModelBuilder._as_plugin_instance(inst)
                            v_entry = TYPE_REGISTRY.get(vi.type_id)
                            v_cfg = v_entry.config_model(**vi.config)
                            pc.viscosity_ev[phase] = v_entry.constructor(v_cfg)
                    # rel_perm_ev: map per phase
                    if getattr(plugins, "rel_perm_ev", None) is not None:
                        pc.rel_perm_ev = {}
                        for phase, inst in plugins.rel_perm_ev.items():
                            ModelBuilder._validate_phase_key(phase, p.phases)
                            ri = ModelBuilder._as_plugin_instance(inst)
                            r_entry = TYPE_REGISTRY.get(ri.type_id)
                            r_cfg = r_entry.config_model(**ri.config)
                            pc.rel_perm_ev[phase] = r_entry.constructor(r_cfg)
                    # diffusion_ev: map per phase
                    if getattr(plugins, "diffusion_ev", None) is not None:
                        pc.diffusion_ev = {}
                        for phase, inst in plugins.diffusion_ev.items():
                            ModelBuilder._validate_phase_key(phase, p.phases)
                            di2 = ModelBuilder._as_plugin_instance(inst)
                            d2_entry = TYPE_REGISTRY.get(di2.type_id)
                            d2_cfg = d2_entry.config_model(**di2.config)
                            # Expand scalar diffusion value to per-component vector to match engine expectations
                            try:
                                val = d2_cfg.value
                                is_scalar = isinstance(val, int | float)
                                if is_scalar:
                                    ncomp = len(p.components)
                                    d2_cfg.value = [float(val)] * ncomp
                            except Exception:
                                pass
                            pc.diffusion_ev[phase] = d2_entry.constructor(d2_cfg)
                    # kinetic_rate_ev: map by integer key
                    if getattr(plugins, "kinetic_rate_ev", None) is not None:
                        # ensure container exists
                        if (
                            not hasattr(pc, "kinetic_rate_ev")
                            or pc.kinetic_rate_ev is None
                        ):
                            try:
                                pc.kinetic_rate_ev = {}
                            except Exception:
                                pass
                        for idx_str, inst in plugins.kinetic_rate_ev.items():
                            try:
                                idx = int(idx_str)
                            except Exception:
                                continue
                            ki = ModelBuilder._as_plugin_instance(inst)
                            k_entry = TYPE_REGISTRY.get(ki.type_id)
                            k_cfg = k_entry.config_model(**ki.config)
                            try:
                                pc.kinetic_rate_ev[idx] = k_entry.constructor(k_cfg)
                            except Exception:
                                if isinstance(pc.kinetic_rate_ev, dict):
                                    pc.kinetic_rate_ev[idx] = k_entry.constructor(k_cfg)

                model.physics.add_property_region(pc, region=region.region)

    @staticmethod
    def _as_plugin_instance(obj: Any) -> PluginInstance:
        if isinstance(obj, PluginInstance):
            return obj
        # pydantic model from model_spec.PluginInstance
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
        assert phase in phases, f"Unknown phase key '{phase}', expected one of {phases}"

    @staticmethod
    def _apply_wells(w: WellsSpec, model: Any) -> None:
        # Defer well/perforation creation until DartsModel.init() calls set_wells().
        # Store spec on the model instance for consumption by a custom set_wells() override.
        model._wells_spec = w

    @staticmethod
    def _apply_initial_conditions(ic: InitialConditionsSpec, model: Any) -> None:
        # Defer initial conditions until after reservoir mesh is created in init().
        model._initial_conditions_spec = ic

    @staticmethod
    def _apply_sim_params(sp: SimParamsSpec, model: Any) -> None:
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
            try:
                from darts.engines import sim_params as spm

                if newton_type == "newton_local_chop":
                    kwargs["newton_type"] = spm.newton_local_chop
            except Exception:
                pass
        model.set_sim_params(**kwargs)
        # Post DataTS tweaks
        if getattr(sp, "newton_tol_stationary", None) is not None:
            model.data_ts.newton_tol_stationary = sp.newton_tol_stationary  # type: ignore[attr-defined]
        if getattr(sp, "min_line_search_update", None) is not None:
            model.data_ts.min_line_search_update = sp.min_line_search_update  # type: ignore[attr-defined]

    @staticmethod
    def _apply_output(out: OutputSpec, model: Any) -> None:
        # Defer output configuration until after init(), so attributes like
        # model.restart are available. Store spec for the runtime to consume.
        model._output_spec = out

    @staticmethod
    def _apply_well_controls(wc: Any, model: Any) -> None:
        # Defer well controls to runtime; store minimal spec
        model._well_controls_spec = wc
