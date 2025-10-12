from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Literal

from pydantic import BaseModel, Field


class PluginInstance(BaseModel):
    type_id: str
    config: dict[str, Any] = {}


@dataclass
class _TypeEntry:
    type_id: str
    kind: str
    config_model: Any
    constructor: Callable[[Any], Any]
    doc: str = ""


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
    return {
        "type_id": entry.type_id,
        "kind": entry.kind,
        "doc": entry.doc,
        "config_schema": _json_schema(entry.config_model),
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


class PropertyContainerConfig(BaseModel):
    phases_name: list[str] | None = None
    components_name: list[str] | None = None
    Mw: list[float] | None = None
    min_z: float = Field(1e-9, ge=0)
    temperature: float = Field(1.0)
    # Optional solid/rock options (used by some models)
    nc_sol: int | None = None
    np_sol: int | None = None
    rock_comp: float | None = None


class ConstantKConfig(BaseModel):
    K: list[float]
    epsilon: float = Field(1e-8, ge=0)


class DensityBasicConfig(BaseModel):
    compr: float = Field(ge=0)
    dens0: float = Field(gt=0)


class ConstFuncConfig(BaseModel):
    value: Any


class PhaseRelPermConfig(BaseModel):
    phase: str


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
    )


def make_property_container(cfg: PropertyContainerConfig) -> Any:
    from darts.physics.super.property_container import PropertyContainer

    kwargs = dict(
        phases_name=cfg.phases_name,
        components_name=cfg.components_name,
        Mw=cfg.Mw,
        min_z=cfg.min_z,
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


def make_phase_relperm(cfg: PhaseRelPermConfig) -> Any:
    from darts.physics.properties.basic import PhaseRelPerm

    return PhaseRelPerm(cfg.phase)


class KineticBasicConfig(BaseModel):
    equi_prod: float
    rate: float = Field(1.0, ge=0)
    ne: int = Field(ge=1)


def make_kinetic_basic(cfg: KineticBasicConfig) -> Any:
    from darts.physics.properties.kinetics import KineticBasic

    return KineticBasic(cfg.equi_prod, cfg.rate, cfg.ne)


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
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="pc/SuperPropertyContainer@v1",
            kind="pc",
            config_model=PropertyContainerConfig,
            constructor=lambda cfg, **kw: make_property_container(cfg),
            doc="Super property container with phases and components.",
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
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="density/DensityBasic@v1",
            kind="density",
            config_model=DensityBasicConfig,
            constructor=lambda cfg, **kw: make_density_basic(cfg),
            doc="Basic density correlation (linear compressibility).",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="viscosity/ConstFunc@v1",
            kind="viscosity",
            config_model=ConstFuncConfig,
            constructor=lambda cfg, **kw: make_const_func(cfg),
            doc="Constant viscosity function.",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="relperm/PhaseRelPerm@v1",
            kind="relperm",
            config_model=PhaseRelPermConfig,
            constructor=lambda cfg, **kw: make_phase_relperm(cfg),
            doc="Phase relative permeability by phase name.",
        )
    )
    TYPE_REGISTRY.register(
        _TypeEntry(
            type_id="kinetic/KineticBasic@v1",
            kind="kinetic",
            config_model=KineticBasicConfig,
            constructor=lambda cfg, **kw: make_kinetic_basic(cfg),
            doc="Basic kinetic reaction evaluator.",
        )
    )


_register_defaults()


def list_capabilities(kind: str | None = None) -> list[dict[str, Any]]:
    return [_schema_entry(t) for t in TYPE_REGISTRY.list(kind)]


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
                    if te.type_id not in [x.type_id for x in TYPE_REGISTRY.list(None)]:
                        TYPE_REGISTRY.register(te)
                        added.append(te.type_id)
        except Exception:
            continue
    return added
