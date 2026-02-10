from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Literal

from pydantic import BaseModel, Field

try:  # Pydantic v2
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - v1 fallback
    ConfigDict = None  # type: ignore


class PluginInstance(BaseModel):
    type_id: str
    config: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _TypeEntry:
    type_id: str
    kind: str
    config_model: Any
    constructor: Callable[[Any], Any]
    doc: str = ""
    customizable: bool = False
    source: Literal["builtin", "entry_point", "manual"] = "builtin"
    mcp_defaults: dict[str, Any] | None = None


class TypeRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, _TypeEntry] = {}
        self._by_kind: dict[str, list[str]] = {}

    def register(self, entry: _TypeEntry) -> None:
        if entry.type_id in self._by_id:
            raise RuntimeError(f"Duplicate type_id: {entry.type_id}")
        self._by_id[entry.type_id] = entry
        self._by_kind.setdefault(entry.kind, []).append(entry.type_id)

    def kinds(self) -> builtins.list[str]:
        return sorted(self._by_kind.keys())

    def list(self, kind: str | None = None) -> builtins.list[_TypeEntry]:
        if kind is None:
            return [self._by_id[k] for k in sorted(self._by_id.keys())]
        return [self._by_id[i] for i in sorted(self._by_kind.get(kind, []))]

    def get(self, type_id: str) -> _TypeEntry:
        if type_id not in self._by_id:
            raise KeyError(type_id)
        return self._by_id[type_id]


def _json_schema(cls: Any) -> dict[str, Any]:
    try:
        return cls.model_json_schema()
    except Exception:  # pragma: no cover
        return cls.schema()


def _schema_entry(entry: _TypeEntry) -> dict[str, Any]:
    defaults = entry.mcp_defaults
    if defaults is None:
        try:
            cfg = entry.config_model()  # type: ignore[call-arg]
            if hasattr(cfg, "model_dump"):
                defaults = cfg.model_dump()
            elif hasattr(cfg, "dict"):
                defaults = cfg.dict()
        except Exception:
            defaults = None
    return {
        "type_id": entry.type_id,
        "kind": entry.kind,
        "doc": entry.doc,
        "config_schema": _json_schema(entry.config_model),
        "customizable": entry.customizable,
        "source": entry.source,
        "mcp_defaults": defaults,
    }


TYPE_REGISTRY = TypeRegistry()


# ========================
# Built-in example configs
# ========================


class CompositionalConfig(BaseModel):
    state_spec: Literal["P", "PT", "PH"] = "P"
    n_points: int = Field(200, ge=2)
    min_p: float = Field(1.0, ge=0)
    max_p: float = Field(300.0, ge=0)
    min_z: float = Field(1e-9, ge=0)
    max_z: float = Field(0.999999, ge=0)
    epsilon_z: float = Field(1e-9, ge=0)
    min_t: float | None = Field(None, ge=0)
    max_t: float | None = Field(None, ge=0)
    extrapolation_flag: bool = False


class BlackOilConfig(BaseModel):
    pvt_path: str
    thermal: bool = False
    type_hydr: Literal["isothermal", "thermal"] = "isothermal"
    type_mech: Literal["none", "poroelasticity", "thermoporoelasticity"] = "none"
    init_type: Literal["uniform"] = "uniform"
    n_points: int = Field(5001, ge=2)
    zero: float = Field(1e-12, ge=0)
    epsilon_z: float = Field(1e-13, ge=0)
    min_p: float = Field(1.0, ge=0)
    max_p: float = Field(450.0, ge=0)
    min_t: float = Field(-10.0)
    max_t: float = Field(100.0)
    min_z: float = Field(0.0, ge=0)
    max_z: float = Field(1.0, ge=0)


class PropertyContainerConfig(BaseModel):
    phases_name: list[str] | None = None
    components_name: list[str] | None = None
    Mw: list[float] | None = None
    # Backward compatible: accept min_z as alias for eps_z
    min_z: float | None = Field(1e-9, ge=0)
    eps_z: float | None = Field(None, ge=0)
    temperature: float = Field(1.0)
    # Optional solid/rock options (used by some models)
    nc_sol: int | None = None
    np_sol: int | None = None
    rock_comp: float | None = None


class AnyConfig(BaseModel):
    """Permissive config model for local plugins."""

    if ConfigDict is not None:  # pragma: no cover - pydantic v2
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover - pydantic v1

        class Config:
            extra = "allow"


class ConstantKConfig(BaseModel):
    K: list[float]
    epsilon: float = Field(1e-8, ge=0)


class DensityBasicConfig(BaseModel):
    compr: float = Field(ge=0)
    dens0: float = Field(gt=0)


class ConstFuncConfig(BaseModel):
    value: Any


class EnthalpyBasicConfig(BaseModel):
    tref: float = Field(273.15)
    hcap: float = Field(0.0357, ge=0)


class PhaseRelPermConfig(BaseModel):
    phase: str
    swc: float = Field(0.0, ge=0)
    sgr: float = Field(0.0, ge=0)
    kre: float = Field(1.0, ge=0)
    n: float = Field(2.0, ge=0)


# ========================
# Constructors (adapters)
# ========================


def make_compositional(
    cfg: CompositionalConfig, components: list[str], phases: list[str], timer
) -> Any:
    from darts.physics.super.physics import Compositional

    # StateSpecification enum mapping
    state_map = {
        "P": Compositional.StateSpecification.P,
        "PT": Compositional.StateSpecification.PT,
        "PH": Compositional.StateSpecification.PH,
    }
    state_spec = state_map.get(cfg.state_spec, Compositional.StateSpecification.P)
    min_t = cfg.min_t
    max_t = cfg.max_t
    if state_spec != Compositional.StateSpecification.P and (
        min_t is None or max_t is None
    ):
        min_t = 273.15 if min_t is None else min_t
        max_t = 473.15 if max_t is None else max_t
    return Compositional(
        components,
        phases,
        timer,
        state_spec=state_spec,
        n_points=cfg.n_points,
        min_p=cfg.min_p,
        max_p=cfg.max_p,
        min_z=cfg.min_z,
        max_z=cfg.max_z,
        epsilon_z=cfg.epsilon_z,
        min_t=min_t,
        max_t=max_t,
        extrapolation_flag=cfg.extrapolation_flag,
    )


def make_black_oil(
    cfg: BlackOilConfig, components: list[str], phases: list[str], timer
) -> Any:
    from darts.input.input_data import InputData
    from darts.physics.blackoil import BlackOil, BlackOilFluidProps

    pvt_path = os.path.expanduser(cfg.pvt_path)
    idata = InputData(
        type_hydr=cfg.type_hydr, type_mech=cfg.type_mech, init_type=cfg.init_type
    )
    idata.fluid = BlackOilFluidProps(pvt=pvt_path)
    idata.obl.n_points = cfg.n_points
    idata.obl.zero = cfg.zero
    idata.obl.epsilon_z = cfg.epsilon_z
    idata.obl.min_p = cfg.min_p
    idata.obl.max_p = cfg.max_p
    idata.obl.min_t = cfg.min_t
    idata.obl.max_t = cfg.max_t
    idata.obl.min_z = cfg.min_z
    idata.obl.max_z = cfg.max_z

    if components and components != idata.fluid.components:
        raise ValueError(
            f"BlackOil components mismatch: {components} vs {idata.fluid.components}"
        )
    if phases and phases != idata.fluid.phases:
        raise ValueError(f"BlackOil phases mismatch: {phases} vs {idata.fluid.phases}")

    return BlackOil(idata, timer, thermal=cfg.thermal)


def make_property_container(cfg: PropertyContainerConfig) -> Any:
    from darts.physics.super.property_container import PropertyContainer

    eps_z = cfg.eps_z if cfg.eps_z is not None else cfg.min_z
    kwargs = dict(
        phases_name=cfg.phases_name,
        components_name=cfg.components_name,
        Mw=cfg.Mw,
        eps_z=eps_z,
        temperature=cfg.temperature,
    )
    if cfg.nc_sol is not None:
        kwargs["nc_sol"] = cfg.nc_sol
    if cfg.np_sol is not None:
        kwargs["np_sol"] = cfg.np_sol
    if cfg.rock_comp is not None:
        kwargs["rock_comp"] = cfg.rock_comp
    return PropertyContainer(**kwargs)


def make_constant_k(cfg: ConstantKConfig, nc: int, epsilon: float) -> Any:
    from darts.physics.properties.flash import ConstantK

    assert len(cfg.K) == nc, "Length of K must equal number of components"
    return ConstantK(nc, cfg.K, epsilon)


def make_density_basic(cfg: DensityBasicConfig) -> Any:
    from darts.physics.properties.density import DensityBasic

    return DensityBasic(compr=cfg.compr, dens0=cfg.dens0)


def make_const_func(cfg: ConstFuncConfig) -> Any:
    from darts.physics.properties.basic import ConstFunc

    return ConstFunc(cfg.value)


def make_enthalpy_basic(cfg: EnthalpyBasicConfig) -> Any:
    from darts.physics.properties.enthalpy import EnthalpyBasic

    return EnthalpyBasic(tref=cfg.tref, hcap=cfg.hcap)


def make_phase_relperm(cfg: PhaseRelPermConfig) -> Any:
    from darts.physics.properties.basic import PhaseRelPerm

    return PhaseRelPerm(cfg.phase, cfg.swc, cfg.sgr, cfg.kre, cfg.n)


def make_dead_oil_property_container(cfg: PropertyContainerConfig) -> Any:
    from darts.physics.deadoil import DeadOilProperties

    eps_z = cfg.eps_z if cfg.eps_z is not None else cfg.min_z
    components_name = cfg.components_name
    phases_name = cfg.phases_name
    if components_name is None or phases_name is None:
        raise ValueError("components_name and phases_name are required")
    Mw = cfg.Mw if cfg.Mw is not None else [1.0] * len(components_name)
    kwargs = dict(
        phases_name=phases_name,
        components_name=components_name,
        Mw=Mw,
        eps_z=eps_z,
        temperature=cfg.temperature,
    )
    if cfg.rock_comp is not None:
        kwargs["rock_comp"] = cfg.rock_comp
    return DeadOilProperties(**kwargs)


class KineticBasicConfig(BaseModel):
    equi_prod: float
    rate: float = Field(1.0, ge=0)
    ne: int = Field(ge=1)


def make_kinetic_basic(cfg: KineticBasicConfig) -> Any:
    from darts.physics.properties.kinetics import KineticBasic

    return KineticBasic(cfg.equi_prod, cfg.rate, cfg.ne)


# ========================
# Local registry helpers
# ========================


def _config_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if hasattr(cfg, "dict"):
        return cfg.dict()
    if isinstance(cfg, dict):
        return cfg
    return {}


def _call_with_signature(fn: Any, cfg_dict: dict[str, Any], **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(**{**cfg_dict, **kwargs})
    params = sig.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_kwargs:
        return fn(**{**cfg_dict, **kwargs})
    filtered: dict[str, Any] = {}
    for name in params.keys():
        if name == "self":
            continue
        if name in cfg_dict:
            filtered[name] = cfg_dict[name]
        elif name in kwargs:
            filtered[name] = kwargs[name]
    return fn(**filtered)


def _wrap_constructor(obj: Any) -> Callable[[Any], Any]:
    def _ctor(cfg: Any, **kwargs: Any) -> Any:
        cfg_dict = _config_to_dict(cfg)
        try:
            return _call_with_signature(obj, cfg_dict, **kwargs)
        except TypeError:
            pass
        try:
            return obj(cfg, **kwargs)
        except TypeError:
            pass
        try:
            return obj(cfg=cfg, **kwargs)
        except TypeError:
            pass
        try:
            return obj(config=cfg, **kwargs)
        except TypeError:
            pass
        return obj(cfg_dict, **kwargs)

    return _ctor


def _module_name_from_path(path: str) -> str:
    digest = hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
    base = os.path.splitext(os.path.basename(path))[0]
    return f"darts_local_{base}_{digest}"


def _import_module_from_file(path: str) -> Any:
    module_name = _module_name_from_path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from '{path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_path(ref: str, base_path: str | None) -> str:
    ref = ref.strip()
    if base_path and not os.path.isabs(ref):
        candidate = os.path.join(base_path, ref)
        if os.path.exists(candidate):
            return candidate
    return ref


def _resolve_attr(obj: Any, attr_path: str) -> Any:
    cur = obj
    for part in attr_path.split("."):
        cur = getattr(cur, part)
    return cur


def _import_from_ref(ref: str, base_path: str | None) -> Any:
    module_ref, _, attr_path = ref.partition(":")
    module_ref = _resolve_path(module_ref, base_path)
    if os.path.exists(module_ref):
        module = _import_module_from_file(module_ref)
    else:
        module = importlib.import_module(module_ref)
    if attr_path:
        return _resolve_attr(module, attr_path)
    return module


def _entry_to_type_entry(entry: Any, base_path: str | None) -> _TypeEntry:
    if isinstance(entry, _TypeEntry):
        if entry.source == "builtin":
            entry.source = "manual"
        if entry.customizable is False:
            entry.customizable = True
        return entry
    if hasattr(entry, "model_dump"):
        data = entry.model_dump(exclude_none=True)
    elif hasattr(entry, "dict"):
        data = entry.dict(exclude_none=True)
    elif isinstance(entry, dict):
        data = {k: v for k, v in entry.items() if v is not None}
    else:
        raise TypeError("Unsupported plugin registry entry type")

    type_id = data["type_id"]
    kind = data["kind"]
    constructor_ref = data["constructor"]
    constructor_obj = _import_from_ref(constructor_ref, base_path)
    config_model_ref = data.get("config_model")
    config_model = (
        _import_from_ref(config_model_ref, base_path) if config_model_ref else AnyConfig
    )
    doc = data.get("doc", "")
    customizable = data.get("customizable")
    if customizable is None:
        customizable = True
    return _TypeEntry(
        type_id=type_id,
        kind=kind,
        config_model=config_model,
        constructor=_wrap_constructor(constructor_obj),
        doc=doc,
        customizable=customizable,
        source="manual",
    )


def load_local_plugin_registry(
    spec: Any, *, base_path: str | None = None, registry: TypeRegistry = TYPE_REGISTRY
) -> list[str]:
    """Load local plugin entries from a JSON registry section."""

    if spec is None:
        return []
    if hasattr(spec, "model_dump"):
        spec_dict = spec.model_dump(exclude_none=True)
    elif hasattr(spec, "dict"):
        spec_dict = spec.dict(exclude_none=True)
    elif isinstance(spec, dict):
        spec_dict = spec
    else:
        raise TypeError("Unsupported plugin registry spec")

    added: list[str] = []
    added_set: set[str] = set()

    modules = spec_dict.get("modules") or []
    for mod_ref in modules:
        mod_ref = mod_ref.partition(":")[0]
        module_obj = _import_from_ref(mod_ref, base_path)
        before = {t.type_id for t in registry.list(None)}
        did_register = False

        if hasattr(module_obj, "register_darts_plugins"):
            did_register = True
            result = module_obj.register_darts_plugins(registry)
            if result:
                for entry in result:
                    te = _entry_to_type_entry(entry, base_path)
                    try:
                        registry.register(te)
                        added_set.add(te.type_id)
                    except RuntimeError:
                        pass

        if hasattr(module_obj, "DARTS_PLUGIN_ENTRIES"):
            did_register = True
            for entry in module_obj.DARTS_PLUGIN_ENTRIES:
                te = _entry_to_type_entry(entry, base_path)
                try:
                    registry.register(te)
                    added_set.add(te.type_id)
                except RuntimeError:
                    pass

        if not did_register:
            raise ValueError(
                f"Module '{mod_ref}' must define register_darts_plugins() or DARTS_PLUGIN_ENTRIES"
            )

        after = {t.type_id for t in registry.list(None)}
        added_set |= after - before

    for entry in spec_dict.get("entries") or []:
        te = _entry_to_type_entry(entry, base_path)
        try:
            registry.register(te)
            added_set.add(te.type_id)
        except RuntimeError:
            # Keep local plugin registry loading idempotent across repeated apply() calls.
            pass

    for type_id in sorted(added_set):
        added.append(type_id)
    return added


# ========================
# Registration helpers
# ========================


def _register_defaults() -> None:
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="physics/Compositional@v1",
            kind="physics",
            config_model=CompositionalConfig,
            constructor=lambda cfg, **kw: make_compositional(
                cfg, kw.get("components", []), kw.get("phases", []), kw.get("timer")
            ),
            doc="Compositional physics (super physics).",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="physics/BlackOil@v1",
            kind="physics",
            config_model=BlackOilConfig,
            constructor=lambda cfg, **kw: make_black_oil(
                cfg, kw.get("components", []), kw.get("phases", []), kw.get("timer")
            ),
            doc="Black-oil physics (PVT-driven).",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="pc/SuperPropertyContainer@v1",
            kind="pc",
            config_model=PropertyContainerConfig,
            constructor=lambda cfg, **kw: make_property_container(cfg),
            doc="Super property container with phases and components.",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="pc/DeadOilProperties@v1",
            kind="pc",
            config_model=PropertyContainerConfig,
            constructor=lambda cfg, **kw: make_dead_oil_property_container(cfg),
            doc="Dead-oil property container (no flash, immiscible phases).",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="flash/ConstantK@v1",
            kind="flash",
            config_model=ConstantKConfig,
            constructor=lambda cfg, **kw: make_constant_k(
                cfg, kw.get("nc"), kw.get("epsilon", cfg.epsilon)
            ),
            doc="Constant-K flash evaluator.",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="density/DensityBasic@v1",
            kind="density",
            config_model=DensityBasicConfig,
            constructor=lambda cfg, **kw: make_density_basic(cfg),
            doc="Basic density correlation (linear compressibility).",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="viscosity/ConstFunc@v1",
            kind="viscosity",
            config_model=ConstFuncConfig,
            constructor=lambda cfg, **kw: make_const_func(cfg),
            doc="Constant viscosity function.",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="conductivity/ConstFunc@v1",
            kind="conductivity",
            config_model=ConstFuncConfig,
            constructor=lambda cfg, **kw: make_const_func(cfg),
            doc="Constant thermal conductivity function.",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="enthalpy/EnthalpyBasic@v1",
            kind="enthalpy",
            config_model=EnthalpyBasicConfig,
            constructor=lambda cfg, **kw: make_enthalpy_basic(cfg),
            doc="Constant heat capacity enthalpy model.",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="relperm/PhaseRelPerm@v1",
            kind="relperm",
            config_model=PhaseRelPermConfig,
            constructor=lambda cfg, **kw: make_phase_relperm(cfg),
            doc="Phase relative permeability by phase name.",
            customizable=False,
            source="builtin",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="kinetic/KineticBasic@v1",
            kind="kinetic",
            config_model=KineticBasicConfig,
            constructor=lambda cfg, **kw: make_kinetic_basic(cfg),
            doc="Basic kinetic reaction evaluator.",
            customizable=False,
            source="builtin",
        )
    )


_register_defaults()


def list_capabilities(kind: str | None = None) -> list[dict[str, Any]]:
    return [_schema_entry(t) for t in TYPE_REGISTRY.list(kind)]


def list_customizable_types(kind: str | None = None) -> list[dict[str, Any]]:
    entries = [t for t in TYPE_REGISTRY.list(kind) if t.customizable]
    return [_schema_entry(t) for t in entries]


def get_kind_schema(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "oneOf": [_schema_entry(t) for t in TYPE_REGISTRY.list(kind)],
    }


def load_entry_point_plugins(group: str = "darts.plugins") -> list[str]:
    added: list[str] = []
    try:
        eps = entry_points(group=group)
    except Exception:
        return added
    for ep in eps:
        try:
            factory = ep.load()
            entries = factory()
            for te in entries:
                if isinstance(te, _TypeEntry):
                    if te.source == "builtin":
                        te.source = "entry_point"
                    if te.customizable is False:
                        te.customizable = True
                    if te.type_id not in [x.type_id for x in TYPE_REGISTRY.list(None)]:
                        TYPE_REGISTRY.register(te)
                        added.append(te.type_id)
        except Exception:
            continue
    return added
