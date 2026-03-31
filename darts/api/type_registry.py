"""
Plugin registry for user-defined physics evaluators, property containers,
and other extensible DARTS components.

Provides the TypeRegistry singleton that maps ``kind/Name@version`` keys to
constructor callables.  Built-in DARTS types are registered at import time;
external plugins can be loaded from entry points or local Python files.
The registry is used by ModelBuilder to instantiate physics and evaluator
objects from JSON configuration.
"""

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

# Built-in configs — imported from the core classes they describe.
# Re-exported here for backward compatibility.
from darts.physics.properties.basic import (  # noqa: E402, F401
    ConstFuncConfig,
    PhaseRelPermConfig,
)
from darts.physics.properties.density import DensityBasicConfig  # noqa: E402, F401
from darts.physics.properties.enthalpy import EnthalpyBasicConfig  # noqa: E402, F401
from darts.physics.properties.flash import ConstantKConfig  # noqa: E402, F401
from darts.physics.properties.kinetics import KineticBasicConfig  # noqa: E402, F401
from darts.physics.super.physics import CompositionalConfig  # noqa: E402, F401
from darts.physics.super.property_container import (
    PropertyContainerConfig,  # noqa: E402, F401
)

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


class BlackOilConfig(BaseModel):
    """Configuration for BlackOil physics (remains here until BlackOil gets its own)."""

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


class AnyConfig(BaseModel):
    """Permissive config model for local plugins."""

    if ConfigDict is not None:  # pragma: no cover - pydantic v2
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover - pydantic v1

        class Config:
            extra = "allow"


# ========================
# Constructors (adapters)
# ========================


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


def _lazy_from_config(module: str, cls_name: str, cfg: Any, **kwargs: Any) -> Any:
    """Lazy-import a class and call its ``from_config()`` classmethod."""
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    return cls.from_config(cfg, **kwargs)


def _register_defaults() -> None:
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="physics/Compositional@v1",
            kind="physics",
            config_model=CompositionalConfig,
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.super.physics",
                "Compositional",
                cfg,
                components=kw.get("components", []),
                phases=kw.get("phases", []),
                timer=kw.get("timer"),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.super.property_container",
                "PropertyContainer",
                cfg,
            ),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.deadoil",
                "DeadOilProperties",
                cfg,
            ),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.properties.flash",
                "ConstantK",
                cfg,
                nc=kw.get("nc"),
            ),  # epsilon is read from cfg, not from kwargs
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.properties.density",
                "DensityBasic",
                cfg,
            ),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.properties.basic",
                "ConstFunc",
                cfg,
            ),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.properties.basic",
                "ConstFunc",
                cfg,
            ),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.properties.enthalpy",
                "EnthalpyBasic",
                cfg,
            ),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.properties.basic",
                "PhaseRelPerm",
                cfg,
            ),
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
            constructor=lambda cfg, **kw: _lazy_from_config(
                "darts.physics.properties.kinetics",
                "KineticBasic",
                cfg,
            ),
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
