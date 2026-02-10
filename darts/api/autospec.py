from __future__ import annotations

import atexit
import json
from collections.abc import Callable
from typing import Any


class _AutoSpecState:
    def __init__(self) -> None:
        # Incrementally built JSON dict following ModelSpec shape
        self.spec: dict[str, Any] = {
            "apiVersion": "darts/v1alpha1",
            "kind": "Model",
        }
        # Map well control object id -> well name
        self.wctrl_to_well: dict[int, str] = {}
        # Map property container id -> recorded PC plugin dict
        self.pc_by_id: dict[int, dict[str, Any]] = {}
        # Map evaluator instance id -> PluginInstance dict
        self.plugin_by_id: dict[int, dict[str, Any]] = {}
        # Original callables for restoring
        self._orig: dict[str, Callable[..., Any]] = {}
        # atexit path if set
        self._emit_path: str | None = None

    # ---------------------
    # Spec builders/helpers
    # ---------------------
    def _ensure(self, key: str, default: Any) -> Any:
        if key not in self.spec:
            self.spec[key] = default
        return self.spec[key]

    def _wells_list(self) -> list[dict[str, Any]]:
        wells = self._ensure("wells", {})
        if "wells" not in wells:
            wells["wells"] = []
        return wells["wells"]

    def _get_or_create_well(self, name: str) -> dict[str, Any]:
        wl = self._wells_list()
        for w in wl:
            if w.get("name") == name:
                return w
        w = {"name": name, "perforations": []}
        wl.append(w)
        return w

    def _physics_dict(self) -> dict[str, Any]:
        return self._ensure("physics", {})

    def record_reservoir_structured(
        self, r: Any, params: dict[str, Any] | None = None
    ) -> None:
        params = params or {}
        res: dict[str, Any] = {"type": "structured"}
        for key in ("nx", "ny", "nz"):
            val = params.get(key, getattr(r, key, None))
            if val is not None:
                res[key] = int(val)
        for key in ("dx", "dy", "dz", "permx", "permy", "permz", "poro", "depth"):
            if key in params and params[key] is not None:
                res[key] = self._as_scalar(params[key])
            else:
                val = getattr(r, key, None)
                sval = self._as_scalar(val)
                if sval is not None:
                    res[key] = sval
        self.spec["reservoir"] = res

    def record_physics_compositional(
        self, components: list[str], phases: list[str], cfg: dict[str, Any]
    ) -> None:
        phy = self._physics_dict()
        phy["plugin"] = {"type_id": "physics/Compositional@v1", "config": cfg}
        phy["components"] = list(components)
        phy["phases"] = list(phases)
        if "property_regions" not in phy:
            phy["property_regions"] = []

    def record_property_container(self, pc: Any, cfg: dict[str, Any]) -> None:
        self.pc_by_id[id(pc)] = {
            "type_id": "pc/SuperPropertyContainer@v1",
            "config": cfg,
        }

    def record_plugin_instance(
        self, obj: Any, type_id: str, config: dict[str, Any]
    ) -> None:
        self.plugin_by_id[id(obj)] = {"type_id": type_id, "config": config}

    def record_property_region(self, pc: Any, region: int) -> None:
        # Build region spec from pc and attached evaluators
        phy = self._physics_dict()
        prop_regions = phy.setdefault("property_regions", [])
        pc_spec = self.pc_by_id.get(id(pc))
        if not pc_spec:
            return
        # Gather plugins
        plugins: dict[str, Any] = {}
        # Flash
        if getattr(pc, "flash_ev", None) is not None:
            pi = self.plugin_by_id.get(id(pc.flash_ev))
            if pi:
                plugins["flash_ev"] = pi
        # Density
        if getattr(pc, "density_ev", None):
            den_map: dict[str, Any] = {}
            for phase, ev in pc.density_ev.items():
                pi = self.plugin_by_id.get(id(ev))
                if pi:
                    den_map[phase] = pi
            if den_map:
                plugins["density_ev"] = den_map
        # Viscosity
        if getattr(pc, "viscosity_ev", None):
            vis_map: dict[str, Any] = {}
            for phase, ev in pc.viscosity_ev.items():
                pi = self.plugin_by_id.get(id(ev))
                if pi:
                    vis_map[phase] = pi
            if vis_map:
                plugins["viscosity_ev"] = vis_map
        # Rel perm
        if getattr(pc, "rel_perm_ev", None):
            rp_map: dict[str, Any] = {}
            for phase, ev in pc.rel_perm_ev.items():
                pi = self.plugin_by_id.get(id(ev))
                if pi:
                    rp_map[phase] = pi
            if rp_map:
                plugins["rel_perm_ev"] = rp_map
        # Diffusion
        if getattr(pc, "diffusion_ev", None):
            dif_map: dict[str, Any] = {}
            for phase, ev in pc.diffusion_ev.items():
                pi = self.plugin_by_id.get(id(ev))
                if pi:
                    dif_map[phase] = pi
            if dif_map:
                plugins["diffusion_ev"] = dif_map
        # Kinetics
        if getattr(pc, "kinetic_rate_ev", None):
            kr_map: dict[str, Any] = {}
            try:
                for idx, ev in pc.kinetic_rate_ev.items():
                    pi = self.plugin_by_id.get(id(ev))
                    if pi:
                        kr_map[str(idx)] = pi
            except Exception:
                pass
            if kr_map:
                plugins["kinetic_rate_ev"] = kr_map

        region_spec: dict[str, Any] = {
            "region": int(region),
            "property_container": pc_spec,
        }
        if plugins:
            region_spec["plugins"] = plugins
        prop_regions.append(region_spec)

    def record_initial_conditions(self, input_distribution: dict[str, Any]) -> None:
        # Store as-provided; json_model will normalize keys
        self.spec.setdefault("initial_conditions", {})["by_array"] = self._to_jsonable(
            input_distribution
        )

    def record_sim_params(self, **kwargs: Any) -> None:
        sp = self.spec.setdefault("sim_params", {})
        # Only whitelist known keys
        allowed = {
            "first_ts",
            "mult_ts",
            "max_ts",
            "runtime",
            "tol_newton",
            "tol_linear",
            "it_newton",
            "it_linear",
            "line_search",
            "newton_type",
            "newton_tol_stationary",
            "min_line_search_update",
        }
        for k, v in kwargs.items():
            if k in allowed and v is not None:
                sp[k] = self._to_jsonable(v)

    def record_output(self, **kwargs: Any) -> None:
        out = self.spec.setdefault("output", {})
        for k in ("folder", "precision", "save_initial"):
            if k in kwargs and kwargs[k] is not None:
                out[k] = kwargs[k]

    def record_well_control(
        self,
        well_name: str,
        *,
        is_inj: bool,
        control_type: Any,
        target: Any,
        phase_name: Any = None,
        inj_composition: Any = None,
    ) -> None:
        # Map control types to strings when rate-controlled
        try:
            from darts.engines import well_control_iface as wci
        except Exception:
            wci = None
        ws = self._get_or_create_well(well_name)
        controls = dict(ws.get("controls", {}))

        if is_inj:
            # Detect BHP vs rate types
            if wci and control_type == getattr(wci, "BHP", None):
                controls["inj_bhp"] = float(target)
            else:
                # Default rate type mapping
                rate_name = None
                if wci is not None:
                    mapping = {
                        getattr(wci, "MOLAR_RATE", object()): "MOLAR_RATE",
                        getattr(wci, "MASS_RATE", object()): "MASS_RATE",
                        getattr(wci, "VOLUMETRIC_RATE", object()): "VOLUMETRIC_RATE",
                        getattr(
                            wci, "ADVECTIVE_HEAT_RATE", object()
                        ): "ADVECTIVE_HEAT_RATE",
                    }
                    rate_name = mapping.get(control_type)
                controls["inj_rate"] = self._to_jsonable(target)
                if rate_name:
                    controls["rate_type"] = rate_name
                if phase_name is not None:
                    controls["phase_name"] = phase_name
                if inj_composition is not None:
                    comp_list = self._to_list(inj_composition)
                    # Pad/truncate to physics.components length for parity with JSON examples
                    try:
                        comps = self.spec.get("physics", {}).get("components", [])
                        nc = len(comps)
                        if nc > 0:
                            if len(comp_list) < nc:
                                comp_list = comp_list + [0.0] * (nc - len(comp_list))
                            elif len(comp_list) > nc:
                                comp_list = comp_list[:nc]
                    except Exception:
                        pass
                    controls["inj_composition"] = comp_list
        else:
            # Producer: record BHP or rate if later extended
            if wci and control_type == getattr(wci, "BHP", None):
                controls["prod_bhp"] = float(target)
            else:
                # Future: producer rate controls
                controls["prod_bhp"] = float(target)

        ws["controls"] = controls

    # ---------------------
    # Utilities
    # ---------------------
    @staticmethod
    def _to_list(v: Any) -> list[float]:
        try:
            return [float(x) for x in list(v)]
        except Exception:
            try:
                return [float(v)]
            except Exception:
                return []

    @staticmethod
    def _to_jsonable(v: Any) -> Any:
        # Preserve native types where possible for cleaner JSON
        try:
            import numpy as np  # type: ignore

            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.ndarray) and v.size == 1:
                return float(v.item())
        except Exception:
            pass
        # Avoid converting ints to floats; keep bools as bools
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return v
        # Leave lists/dicts/strings as-is
        return v

    @staticmethod
    def _as_scalar(v: Any) -> Any | None:
        try:
            import numpy as np  # type: ignore

            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.ndarray):
                if v.size == 1:
                    itm = v.item()
                    if isinstance(itm, np.integer | int):
                        return int(itm)
                    return float(itm)
                return None
        except Exception:
            pass
        if isinstance(v, list | tuple):
            if len(v) == 1:
                try:
                    return int(v[0]) if isinstance(v[0], int) else float(v[0])
                except Exception:
                    return None
            return None
        try:
            return int(v) if isinstance(v, int) else float(v)
        except Exception:
            return None

    # ---------------------
    # Emission
    # ---------------------
    def emit_json(self, path: str) -> None:
        # Ensure minimal structure
        if "physics" in self.spec:
            phy = self.spec["physics"]
            # Clean empty property_regions
            if not phy.get("property_regions"):
                phy.pop("property_regions", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.spec, f, indent=2)


_STATE = _AutoSpecState()


def _patch(target: Any, name: str, wrapper: Callable[..., Any]) -> None:
    key = (
        f"{target.__module__}.{target.__name__}.{name}"
        if hasattr(target, "__name__")
        else f"{target}.{name}"
    )
    if key in _STATE._orig:
        return
    orig = getattr(target, name)
    _STATE._orig[key] = orig
    setattr(target, name, wrapper(orig))


def _restore_all() -> None:
    for key, orig in list(_STATE._orig.items()):
        try:
            mod_cls, meth = key.rsplit(".", 1)
            parts = mod_cls.split(".")
            obj = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
            cls = getattr(obj, parts[-1])
            setattr(cls, meth, orig)
        except Exception:
            pass
    _STATE._orig.clear()


def enable_autorecording() -> None:
    """Monkey-patch DARTS primitives to auto-record a ModelSpec during model construction."""
    # StructReservoir
    try:
        from darts.reservoirs.struct_reservoir import StructReservoir  # type: ignore

        def wrap_sr_init(orig):
            def _wrapped(self, *args, **kwargs):
                orig(self, *args, **kwargs)
                try:
                    # Prefer constructor kwargs (scalars) over internal arrays
                    capture: dict[str, Any] = {}
                    for k in (
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
                    ):
                        if k in kwargs:
                            capture[k] = kwargs[k]
                    _STATE.record_reservoir_structured(self, capture)
                except Exception:
                    pass

            return _wrapped

        def wrap_add_well(orig):
            def _wrapped(self, name: str, *args, **kwargs):
                res = orig(self, name, *args, **kwargs)
                try:
                    # Find the last added well and map control id -> name
                    if getattr(self, "wells", None):
                        w = self.wells[-1]
                        ctrl = getattr(w, "control", None)
                        if ctrl is not None:
                            _STATE.wctrl_to_well[id(ctrl)] = name
                    _STATE._get_or_create_well(name)
                except Exception:
                    pass
                return res

            return _wrapped

        def wrap_add_perf(orig):
            def _wrapped(self, well_name: str, *args, **kwargs):
                res = orig(self, well_name, *args, **kwargs)
                try:
                    ijk = None
                    if "cell_index" in kwargs:
                        ijk = kwargs["cell_index"]
                    elif args:
                        # Signature: (well_name, cell_index, ...)
                        ijk = args[0]
                    if ijk is not None:
                        w = _STATE._get_or_create_well(well_name)
                        perfs = w.setdefault("perforations", [])
                        perfs.append({"ijk": list(ijk)})
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(StructReservoir, "__init__", wrap_sr_init)
        _patch(StructReservoir, "add_well", wrap_add_well)
        _patch(StructReservoir, "add_perforation", wrap_add_perf)
    except Exception:
        pass

    # Compositional physics and property regions
    try:
        from darts.physics.super.physics import Compositional  # type: ignore

        def _state_spec_to_str(state_spec: Any) -> str:
            try:
                name = getattr(state_spec, "name", None)
                if name in ("P", "PT", "PH"):
                    return name
            except Exception:
                pass
            # Fallback to P
            return "P"

        def wrap_comp_init(orig):
            def _wrapped(self, components, phases, timer, *args, **kwargs):
                res = orig(self, components, phases, timer, *args, **kwargs)
                try:
                    cfg: dict[str, Any] = {}
                    if "state_spec" in kwargs:
                        cfg["state_spec"] = _state_spec_to_str(kwargs["state_spec"])
                    for k in ("n_points", "min_p", "max_p", "min_z", "max_z"):
                        if k in kwargs and kwargs[k] is not None:
                            cfg[k] = _STATE._to_jsonable(kwargs[k])
                    _STATE.record_physics_compositional(
                        list(components), list(phases), cfg
                    )
                except Exception:
                    pass
                return res

            return _wrapped

        def wrap_add_region(orig):
            def _wrapped(self, property_container: Any, *args, **kwargs):
                res = orig(self, property_container, *args, **kwargs)
                try:
                    region = kwargs.get("region", 0)
                    _STATE.record_property_region(property_container, int(region))
                except Exception:
                    pass
                return res

            return _wrapped

        def wrap_set_ic(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    input_distribution = kwargs.get("input_distribution")
                    if input_distribution is None and len(args) >= 2:
                        input_distribution = args[1]
                    if isinstance(input_distribution, dict):
                        _STATE.record_initial_conditions(input_distribution)
                except Exception:
                    pass
                return res

            return _wrapped

        def wrap_set_wc(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    wctrl = kwargs.get("wctrl")
                    control_type = kwargs.get("control_type")
                    is_inj = bool(kwargs.get("is_inj"))
                    target = kwargs.get("target")
                    phase_name = kwargs.get("phase_name")
                    inj_comp = kwargs.get("inj_composition")
                    well_name = (
                        _STATE.wctrl_to_well.get(id(wctrl))
                        if wctrl is not None
                        else None
                    )
                    if well_name:
                        _STATE.record_well_control(
                            well_name,
                            is_inj=is_inj,
                            control_type=control_type,
                            target=target,
                            phase_name=phase_name,
                            inj_composition=inj_comp,
                        )
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(Compositional, "__init__", wrap_comp_init)
        _patch(Compositional, "add_property_region", wrap_add_region)
        _patch(Compositional, "set_initial_conditions_from_array", wrap_set_ic)
        _patch(Compositional, "set_well_controls", wrap_set_wc)
    except Exception:
        pass

    # Property container
    try:
        from darts.physics.super.property_container import (
            PropertyContainer,  # type: ignore
        )

        def wrap_pc_init(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    cfg: dict[str, Any] = {}
                    for k in (
                        "phases_name",
                        "components_name",
                        "Mw",
                        "min_z",
                        "temperature",
                        "nc_sol",
                        "np_sol",
                        "rock_comp",
                    ):
                        if k in kwargs and kwargs[k] is not None:
                            v = kwargs[k]
                            cfg[k] = list(v) if isinstance(v, list | tuple) else v
                    _STATE.record_property_container(self, cfg)
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(PropertyContainer, "__init__", wrap_pc_init)
    except Exception:
        pass

    # Evaluators
    try:
        from darts.physics.properties.flash import ConstantK  # type: ignore

        def wrap_ck_init(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    # Signature: (nc, K, epsilon)
                    K = (
                        kwargs.get("K")
                        if "K" in kwargs
                        else (args[1] if len(args) > 1 else None)
                    )
                    eps = (
                        kwargs.get("epsilon")
                        if "epsilon" in kwargs
                        else (args[2] if len(args) > 2 else None)
                    )
                    if K is not None:
                        cfg = {"K": list(K)}
                        if eps is not None:
                            cfg["epsilon"] = _STATE._to_jsonable(eps)
                        _STATE.record_plugin_instance(self, "flash/ConstantK@v1", cfg)
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(ConstantK, "__init__", wrap_ck_init)
    except Exception:
        pass

    try:
        from darts.physics.properties.density import DensityBasic  # type: ignore

        def wrap_db_init(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    compr = (
                        kwargs.get("compr")
                        if "compr" in kwargs
                        else (args[0] if len(args) > 0 else None)
                    )
                    dens0 = (
                        kwargs.get("dens0")
                        if "dens0" in kwargs
                        else (args[1] if len(args) > 1 else None)
                    )
                    if compr is not None and dens0 is not None:
                        cfg = {
                            "compr": _STATE._to_jsonable(compr),
                            "dens0": _STATE._to_jsonable(dens0),
                        }
                        _STATE.record_plugin_instance(
                            self, "density/DensityBasic@v1", cfg
                        )
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(DensityBasic, "__init__", wrap_db_init)
    except Exception:
        pass

    try:
        from darts.physics.properties.basic import (  # type: ignore
            ConstFunc,
            PhaseRelPerm,
        )

        def wrap_cf_init(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    value = (
                        kwargs.get("value")
                        if "value" in kwargs
                        else (args[0] if len(args) > 0 else None)
                    )
                    if value is not None:
                        cfg = {"value": _STATE._to_jsonable(value)}
                        # Reuse viscosity namespace for ConstFunc as in registry
                        _STATE.record_plugin_instance(
                            self, "viscosity/ConstFunc@v1", cfg
                        )
                except Exception:
                    pass
                return res

            return _wrapped

        def wrap_prp_init(orig):
            def _wrapped(self, phase: str, *args, **kwargs):
                res = orig(self, phase, *args, **kwargs)
                try:
                    _STATE.record_plugin_instance(
                        self, "relperm/PhaseRelPerm@v1", {"phase": phase}
                    )
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(ConstFunc, "__init__", wrap_cf_init)
        _patch(PhaseRelPerm, "__init__", wrap_prp_init)
    except Exception:
        pass

    try:
        from darts.physics.properties.kinetics import KineticBasic  # type: ignore

        def wrap_kb_init(orig):
            def _wrapped(self, equi_prod, rate, ne, *args, **kwargs):
                res = orig(self, equi_prod, rate, ne, *args, **kwargs)
                try:
                    cfg = {
                        "equi_prod": _STATE._to_jsonable(equi_prod),
                        "rate": _STATE._to_jsonable(rate),
                        "ne": int(ne),
                    }
                    _STATE.record_plugin_instance(self, "kinetic/KineticBasic@v1", cfg)
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(KineticBasic, "__init__", wrap_kb_init)
    except Exception:
        pass

    # DartsModel hooks for sim params and output
    try:
        from darts.models.darts_model import DartsModel  # type: ignore

        def wrap_set_sim_params(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    # Map engine constant back to string when possible
                    nt = kwargs.get("newton_type")
                    try:
                        from darts.engines import sim_params as spm  # type: ignore

                        if nt == getattr(spm, "newton_local_chop", object()):
                            kwargs["newton_type"] = "newton_local_chop"
                        elif isinstance(nt, int):
                            kwargs["newton_type"] = "default"
                    except Exception:
                        pass
                    _STATE.record_sim_params(**kwargs)
                except Exception:
                    pass
                return res

            return _wrapped

        def wrap_set_output(orig):
            def _wrapped(self, *args, **kwargs):
                res = orig(self, *args, **kwargs)
                try:
                    _STATE.record_output(**kwargs)
                except Exception:
                    pass
                return res

            return _wrapped

        _patch(DartsModel, "set_sim_params", wrap_set_sim_params)
        _patch(DartsModel, "set_output", wrap_set_output)
    except Exception:
        pass


def disable_autorecording() -> None:
    """Restore original methods and stop recording."""
    _restore_all()


def emit_json(path: str) -> None:
    """Write the recorded spec to a JSON file."""
    _STATE.emit_json(path)


def emit_on_exit(path: str) -> None:
    """Schedule writing the recorded spec on interpreter exit."""
    _STATE._emit_path = path

    def _dump() -> None:
        try:
            if _STATE._emit_path:
                _STATE.emit_json(_STATE._emit_path)
        except Exception:
            pass

    atexit.register(_dump)
