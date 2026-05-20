"""Named-preset registry for DARTS Configs.

A preset is a JSON file holding two top-level keys:

* ``_meta`` — :class:`PresetMeta` payload describing what the preset is
  (``name``, ``description``, ``references``, ``tags``, ``version``).
* ``config`` — a Config payload that validates against a Pydantic Config class
  registered in DARTS (e.g. ``DensityBasicConfig``, ``ConstantKConfig``).

Presets live as ``.json`` files under a hierarchical directory tree.  The
default tree is :data:`DEFAULT_PRESET_ROOT` — the package-data directory
``darts/api/presets_data/`` (shipped in the wheel); third-party trees can
be loaded with :func:`load_preset_dir`, and the ``DARTS_PRESET_ROOT`` env
var adds a first-priority search root for out-of-tree preset libraries.

The loader resolves an evaluator-config payload via the existing ``kind``
discriminator (see ``darts.physics.properties.evaluator_base``).  Non-evaluator
Config classes can be associated with a preset directory by calling
:func:`register_preset_directory_binding`.

Composition is supported via in-place ``$preset`` references — any nested dict
of the form ``{"$preset": "<preset_name>"}`` is replaced by the referenced
preset's ``config`` payload before validation, with cycle detection.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Preset envelope
# ---------------------------------------------------------------------------


class PresetMeta(BaseModel):
    """Self-describing metadata carried by every preset file.

    Surfaces in :func:`iter_presets` and :func:`docs` so browsing tools (CLI,
    MCP) can show what a preset is for without parsing its config payload.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Short identifier; should match the file stem")
    description: str = Field(description="Human-readable summary of the preset")
    references: list[str] = Field(
        default_factory=list,
        description="Bibliographic references or URLs backing the values",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form keywords for filtering and discovery",
    )
    version: str = Field("1", description="Preset payload version")


@dataclass(frozen=True)
class Preset:
    """A loaded preset: metadata + materialized Config instance.

    :param qualified_name: hierarchical preset name (e.g.
        ``"evaluators/density/co2_brine"``)
    :type qualified_name: str
    :param meta: preset metadata
    :type meta: PresetMeta
    :param config: validated Config instance
    :type config: BaseModel
    :param source_path: file path the preset was loaded from (``None`` for
        runtime-registered presets)
    :type source_path: pathlib.Path | None
    :param property_regions: optional ``property_regions`` payload carried
        alongside ``config`` (e.g. on physics presets that want to ship a
        full evaluator stack rather than requiring the caller to supply
        ``property_regions`` separately).  ``$preset`` refs inside this
        payload are resolved during load.
    :type property_regions: list[dict] | None
    :param plugin_registry: optional plugin-registry block alongside
        ``config`` (lets a preset ship its own code-plugin entries —
        e.g. a custom property container — without the caller naming a
        legacy ``model_type`` template).  Relative ``constructor`` paths
        are rewritten to absolute paths at load time, anchored at the
        preset file's directory.
    :type plugin_registry: dict | None
    """

    qualified_name: str
    meta: PresetMeta
    config: BaseModel
    source_path: Path | None = field(default=None)
    property_regions: list[dict[str, Any]] | None = field(default=None)
    plugin_registry: dict[str, Any] | None = field(default=None)


# ---------------------------------------------------------------------------
# Registry state
# ---------------------------------------------------------------------------


def _default_preset_root() -> Path:
    """Locate the shipped preset tree bundled inside the ``darts`` package.

    The presets live at ``darts/api/presets_data/`` and are declared as
    package data in ``pyproject.toml``, so this resolves correctly for both
    editable/source installs and wheel installs.  ``importlib.resources``
    returns a real filesystem path here because open-darts is always
    installed unzipped (it ships compiled ``.so``/``.pyd`` extensions).
    """
    from importlib.resources import files

    return Path(str(files("darts.api"))) / "presets_data"


DEFAULT_PRESET_ROOT: Path = _default_preset_root()

# qualified_name → Preset
_PRESET_REGISTRY: dict[str, Preset] = {}

# Top-level directory category → ConfigClass for non-evaluator presets.
# Evaluator presets dispatch via the existing ``kind`` discriminator and do not
# need a binding here.
_DIRECTORY_BINDINGS: dict[str, type[BaseModel]] = {}

# Roots searched on-demand by ``load_preset`` when a name is not yet in the
# in-memory registry.  This lets cross-file ``$preset`` references resolve in
# either order without forcing a strict load sequence.
_SEARCH_ROOTS: list[Path] = []


def register_preset_directory_binding(
    directory: str, config_cls: type[BaseModel]
) -> None:
    """Bind a preset top-level directory to a Config class.

    Use this when a preset category has a single canonical Config class with
    no built-in ``kind`` discriminator (e.g. ``property_container/`` →
    :class:`PropertyContainerConfig`).  Evaluator categories do *not* need a
    binding — their ``kind`` field already disambiguates the Config class.

    :param directory: top-level directory under the preset root, e.g.
        ``"property_container"``
    :type directory: str
    :param config_cls: Pydantic Config class to validate every preset in that
        directory against
    :type config_cls: type[pydantic.BaseModel]
    """
    _DIRECTORY_BINDINGS[directory.strip("/")] = config_cls


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _qualified_name_for(path: Path, root: Path) -> str:
    """Convert a preset file path into its hierarchical name.

    ``presets_data/evaluators/density/co2_brine.json`` →
    ``"evaluators/density/co2_brine"``.
    """
    rel = path.relative_to(root).with_suffix("")
    return rel.as_posix()


def _resolve_evaluator_config_cls(kind: str) -> type[BaseModel]:
    """Look up a Pydantic Config class for an evaluator ``kind``."""
    # Local import to avoid a hard dependency at module import time.
    from darts.physics.properties.evaluator_base import resolve_evaluator

    _, config_cls = resolve_evaluator(kind)
    return config_cls


def _config_class_for(qualified_name: str, payload: dict[str, Any]) -> type[BaseModel]:
    """Pick the Config class that should validate a given preset payload.

    Resolution order:

    1. Longest-matching directory-prefix binding (so
       ``physics/compositional`` wins over ``physics`` for a preset named
       ``physics/compositional/co2_brine``).
    2. Evaluator ``kind`` discriminator inside the payload.

    :raises ValueError: if no Config class can be resolved.
    """
    parts = qualified_name.split("/")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "/".join(parts[:i])
        if prefix in _DIRECTORY_BINDINGS:
            return _DIRECTORY_BINDINGS[prefix]
    kind = payload.get("kind") if isinstance(payload, dict) else None
    if kind is None:
        raise ValueError(
            f"Preset '{qualified_name}' has no 'kind' field in its config payload "
            f"and no Config class is bound to any directory prefix of '{qualified_name}'. "
            f"Either add 'kind' or call register_preset_directory_binding(<prefix>, ...)."
        )
    return _resolve_evaluator_config_cls(kind)


def _resolve_preset_refs(value: Any, _seen: tuple[str, ...] = ()) -> Any:
    """Recursively replace ``{"$preset": "<name>"}`` markers with the
    referenced preset's config payload.

    :param value: arbitrary JSON-decoded value
    :param _seen: chain of preset names currently being resolved (for cycle
        detection)
    :raises ValueError: on cycles.
    :raises KeyError: on unknown preset references.
    """
    if isinstance(value, dict):
        if set(value.keys()) == {"$preset"}:
            ref = value["$preset"]
            if not isinstance(ref, str):
                raise ValueError(
                    f"$preset value must be a string, got {type(ref).__name__}"
                )
            ref = ref.strip("/")
            if ref in _seen:
                cycle = " -> ".join((*_seen, ref))
                raise ValueError(f"Cyclic $preset reference detected: {cycle}")
            target = _load_preset_lazy(ref, _seen)
            # Re-serialize the target's config to a dict so further refs nested
            # inside it can also resolve.  ``model_dump`` is deterministic.
            return target.config.model_dump()
        return {k: _resolve_preset_refs(v, _seen) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_preset_refs(v, _seen) for v in value]
    return value


def _find_on_disk(qualified_name: str) -> Path | None:
    """Locate a preset's JSON file under any registered search root."""
    rel = Path(*qualified_name.split("/")).with_suffix(".json")
    for root in _SEARCH_ROOTS:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def _load_preset_lazy(qualified_name: str, _seen: tuple[str, ...]) -> Preset:
    """Return a preset, loading it from disk on demand if necessary.

    Used by ``$preset`` ref resolution so cross-file references inside the
    same ``load_preset_dir`` call don't depend on file iteration order.
    """
    if qualified_name in _PRESET_REGISTRY:
        return _PRESET_REGISTRY[qualified_name]
    on_disk = _find_on_disk(qualified_name)
    if on_disk is None:
        known = sorted(_PRESET_REGISTRY)
        raise KeyError(f"Unknown preset '{qualified_name}'. Known presets: {known}")
    # Find which root this file belongs to so qualified names line up.
    for root in _SEARCH_ROOTS:
        try:
            on_disk.relative_to(root)
        except ValueError:
            continue
        return _load_preset_from_file(on_disk, root, _seen)
    # Should not happen — _find_on_disk returned a path under some root.
    raise RuntimeError(  # pragma: no cover
        f"Preset file {on_disk} not under any registered search root"
    )


def _load_preset_from_file(
    path: Path, root: Path, _seen: tuple[str, ...] = ()
) -> Preset:
    """Read, validate, and ref-resolve a single preset JSON file."""
    qualified_name = _qualified_name_for(path, root)
    with path.open("r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Preset '{qualified_name}' is not valid JSON: {exc}"
            ) from exc

    if not isinstance(raw, dict) or "_meta" not in raw or "config" not in raw:
        raise ValueError(
            f"Preset '{qualified_name}' must be a JSON object with '_meta' and "
            f"'config' top-level keys; got keys {sorted(raw) if isinstance(raw, dict) else type(raw).__name__}"
        )

    meta = PresetMeta.model_validate(raw["_meta"])
    raw_config = _resolve_preset_refs(raw["config"], (*_seen, qualified_name))
    config_cls = _config_class_for(qualified_name, raw_config)
    config = config_cls.model_validate(raw_config)

    # Optional property_regions payload alongside ``config`` (physics
    # presets that ship a full evaluator stack).  We resolve ``$preset``
    # refs inside it but deliberately don't Pydantic-validate here — that
    # happens downstream when the consuming Config (e.g. PhysicsSpec)
    # validates the merged payload.
    property_regions: list[dict[str, Any]] | None = None
    if "property_regions" in raw:
        resolved = _resolve_preset_refs(
            raw["property_regions"], (*_seen, qualified_name)
        )
        if not isinstance(resolved, list):
            raise ValueError(
                f"Preset '{qualified_name}': 'property_regions' must be a JSON "
                f"list; got {type(resolved).__name__}"
            )
        property_regions = resolved

    # Optional plugin_registry block — lets a preset ship its own code
    # plugin entries (e.g. a custom property container) without the user
    # naming a ``model_type`` template.  Relative constructor paths are
    # rewritten to absolute paths anchored at the preset file's directory
    # so the server can dispatch them after the in-memory preset is
    # detached from its filesystem location.
    plugin_registry: dict[str, Any] | None = None
    if "plugin_registry" in raw:
        plugin_registry = _resolve_preset_refs(
            raw["plugin_registry"], (*_seen, qualified_name)
        )
        if not isinstance(plugin_registry, dict):
            raise ValueError(
                f"Preset '{qualified_name}': 'plugin_registry' must be a JSON "
                f"object; got {type(plugin_registry).__name__}"
            )
        plugin_registry = _resolve_plugin_registry_paths(plugin_registry, path.parent)

    preset = Preset(
        qualified_name=qualified_name,
        meta=meta,
        config=config,
        source_path=path,
        property_regions=property_regions,
        plugin_registry=plugin_registry,
    )
    _PRESET_REGISTRY[qualified_name] = preset
    return preset


def _resolve_plugin_registry_paths(
    block: dict[str, Any], base_dir: Path
) -> dict[str, Any]:
    """Rewrite relative ``constructor`` paths in a preset ``plugin_registry``
    block to absolute paths anchored at ``base_dir``.

    A constructor string is of the form ``"<path>:<symbol>"`` where
    ``<path>`` may be relative.  Absolute paths and paths that don't
    resolve to an existing file are left untouched (the latter so the
    downstream resolver can produce a clear error message).
    """
    entries = block.get("entries")
    if not isinstance(entries, list):
        return block
    out_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            out_entries.append(entry)
            continue
        constructor = entry.get("constructor")
        if isinstance(constructor, str) and ":" in constructor:
            ctor_path, _, symbol = constructor.partition(":")
            ctor_path_obj = Path(ctor_path)
            if not ctor_path_obj.is_absolute():
                candidate = (base_dir / ctor_path_obj).resolve()
                if candidate.is_file():
                    entry = {
                        **entry,
                        "constructor": f"{candidate}:{symbol}",
                    }
        out_entries.append(entry)
    return {**block, "entries": out_entries}


def load_preset_dir(root: str | os.PathLike[str]) -> list[str]:
    """Walk a directory tree and register every ``.json`` preset found.

    The root is added to the on-demand search path, so ``$preset`` references
    inside one file can resolve to another file in the same tree even before
    that file has been visited.  Re-running with the same root re-loads each
    preset (useful for tests and hot-reload workflows).

    :param root: directory containing the preset tree
    :type root: str | os.PathLike[str]
    :return: list of qualified names that were loaded (sorted)
    :rtype: list[str]
    """
    _register_default_directory_bindings()
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Preset root does not exist: {root_path}")

    if root_path not in _SEARCH_ROOTS:
        _SEARCH_ROOTS.append(root_path)

    loaded: list[str] = []
    for file_path in sorted(root_path.rglob("*.json")):
        # A sibling file may already have triggered a lazy load via a
        # ``$preset`` reference, but loading again is cheap and keeps the
        # registry holding the freshest copy regardless of iteration order.
        preset = _load_preset_from_file(file_path, root_path)
        loaded.append(preset.qualified_name)
    # Drop duplicates that arise when a file is loaded both lazily and in the
    # explicit walk.
    return sorted(set(loaded))


_DEFAULT_BINDINGS_REGISTERED = False


def _register_default_directory_bindings() -> None:
    """Bind built-in non-evaluator Config classes to their preset directories.

    Lazy-imported so this module stays importable even when downstream
    physics/reservoir modules have not been initialized yet.  Idempotent.
    """
    global _DEFAULT_BINDINGS_REGISTERED
    if _DEFAULT_BINDINGS_REGISTERED:
        return
    _DEFAULT_BINDINGS_REGISTERED = True

    # Each binding is wrapped in a try/except so a missing optional dependency
    # (e.g. the physics module) does not block the registry from being usable.
    try:
        from darts.physics.super.property_container import PropertyContainerConfig

        register_preset_directory_binding("property_container", PropertyContainerConfig)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from darts.physics.super.physics import CompositionalConfig

        register_preset_directory_binding("physics/compositional", CompositionalConfig)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from darts.physics.blackoil import BlackOilConfig

        register_preset_directory_binding("physics/black_oil", BlackOilConfig)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        # ``physics/dead_oil`` presets validate against ``CompositionalConfig``
        # because dead-oil physics is dispatched through the Compositional
        # plugin family (matching the legacy templates under
        # ``models/cpg_deadoil_brugge``, ``models/2ph_do_thermal`` etc).
        from darts.physics.super.physics import CompositionalConfig

        register_preset_directory_binding("physics/dead_oil", CompositionalConfig)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from darts.reservoirs.struct_reservoir import StructReservoirConfig

        register_preset_directory_binding(
            "reservoirs/structured", StructReservoirConfig
        )
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from darts.reservoirs.cpg_reservoir import CPGReservoirConfig

        register_preset_directory_binding("reservoirs/cpg", CPGReservoirConfig)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from darts.models.darts_model import SimParamsConfig

        register_preset_directory_binding("sim_params", SimParamsConfig)
    except Exception:  # pragma: no cover - defensive
        pass

    # Importing optional evaluator subpackages so their kinds register with
    # the global evaluator registry — required for presets that reference,
    # e.g., IAPWS evaluator kinds without the user importing IAPWS first.
    try:
        import darts.physics.properties.iapws.iapws_property  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        pass


def _ensure_default_root_loaded() -> None:
    """Lazily load the default preset root on first registry access.

    The load order is bottom-up: lower-priority roots first, higher-priority
    roots last.  Because ``_load_preset_from_file`` overwrites entries in
    ``_PRESET_REGISTRY`` by qualified name, the *last* root to load a given
    name is the one that wins — so loading ``DARTS_PRESET_ROOT`` after the
    shipped default delivers the documented "env var has first priority"
    behavior.

    Effective precedence (highest first):

    1. ``DARTS_PRESET_ROOT`` env var — explicit override, useful when
       open-darts is installed as a wheel and the source tree lives
       elsewhere, or to ship a per-project preset library that overrides
       individual default presets.
    2. ``DEFAULT_PRESET_ROOT`` — resolved via
       ``importlib.resources.files("darts.api") / "presets_data"`` so it
       works for both editable and wheel installs.
    """
    _register_default_directory_bindings()

    # Always load the shipped default first so the env-var root's presets
    # overwrite same-name shipped ones when both are present.
    if DEFAULT_PRESET_ROOT not in _SEARCH_ROOTS and DEFAULT_PRESET_ROOT.is_dir():
        load_preset_dir(DEFAULT_PRESET_ROOT)

    env_override = os.environ.get("DARTS_PRESET_ROOT")
    if env_override:
        env_root = Path(env_override).expanduser().resolve()
        if env_root not in _SEARCH_ROOTS and env_root.is_dir():
            load_preset_dir(env_root)


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


def load_preset(qualified_name: str) -> Preset:
    """Look up a preset by its hierarchical name.

    :param qualified_name: e.g. ``"evaluators/density/co2_brine"``
    :type qualified_name: str
    :return: the loaded preset
    :rtype: Preset
    :raises KeyError: if the preset is not registered
    """
    _ensure_default_root_loaded()
    qname = qualified_name.strip("/")
    if qname not in _PRESET_REGISTRY:
        known = sorted(_PRESET_REGISTRY)
        raise KeyError(f"Unknown preset '{qname}'. Known presets: {known}")
    return _PRESET_REGISTRY[qname]


def load_preset_config(qualified_name: str) -> BaseModel:
    """Convenience: return only the validated Config from a preset.

    :param qualified_name: hierarchical preset name
    :type qualified_name: str
    :return: validated Pydantic Config instance
    :rtype: pydantic.BaseModel
    """
    return load_preset(qualified_name).config


def list_presets(category: str | None = None) -> list[str]:
    """List all registered preset qualified names.

    :param category: optional directory prefix to filter by, e.g.
        ``"evaluators/density"`` returns only density presets
    :type category: str | None
    :return: sorted list of qualified names
    :rtype: list[str]
    """
    _ensure_default_root_loaded()
    names = sorted(_PRESET_REGISTRY)
    if category is None:
        return names
    prefix = category.strip("/") + "/"
    return [n for n in names if n.startswith(prefix) or n == category.strip("/")]


def iter_presets(category: str | None = None) -> Iterator[Preset]:
    """Iterate over registered presets, optionally filtered by category.

    :param category: optional directory prefix
    :type category: str | None
    :return: iterator over :class:`Preset` instances
    :rtype: Iterator[Preset]
    """
    for name in list_presets(category):
        yield _PRESET_REGISTRY[name]


def register_preset(
    qualified_name: str, meta: PresetMeta | dict[str, Any], config: BaseModel
) -> None:
    """Register a preset in-process (no file required).

    Useful for tests and dynamic registration from third-party packages.

    :param qualified_name: hierarchical preset name
    :type qualified_name: str
    :param meta: preset metadata (instance or dict to validate)
    :type meta: PresetMeta | dict[str, Any]
    :param config: validated Config instance
    :type config: pydantic.BaseModel
    """
    qname = qualified_name.strip("/")
    if isinstance(meta, dict):
        meta = PresetMeta.model_validate(meta)
    _PRESET_REGISTRY[qname] = Preset(
        qualified_name=qname,
        meta=meta,
        config=config,
        source_path=None,
    )


def docs(qualified_name: str) -> dict[str, Any]:
    """Return a documentation dict combining preset metadata with the
    underlying Config's JSON Schema.

    Intended for MCP / CLI browsing tools that want to show users *what* a
    preset is for and *what shape* its config takes.

    :param qualified_name: hierarchical preset name
    :type qualified_name: str
    :return: dict with ``meta``, ``config_class``, ``config``, ``schema``
    :rtype: dict[str, Any]
    """
    preset = load_preset(qualified_name)
    config_cls = type(preset.config)
    return {
        "qualified_name": preset.qualified_name,
        "meta": preset.meta.model_dump(),
        "config_class": f"{config_cls.__module__}.{config_cls.__qualname__}",
        "config": preset.config.model_dump(),
        "schema": config_cls.model_json_schema(),
        "source_path": str(preset.source_path) if preset.source_path else None,
    }


# ---------------------------------------------------------------------------
# Section-level preset resolution
# ---------------------------------------------------------------------------
# A ModelSpec JSON may use ``{"preset": "<name>", ...overrides}`` inside any
# top-level section (``reservoir``, ``physics``, ``sim_params``, ...).  The
# pre-pass below expands those shorthand references into full section payloads
# *before* Pydantic validation, so the single-JSON workflow (``darts --json
# model.json``) accepts the same modular composition that the MCP server
# resolves on its side via ``build_*_patch`` helpers.
# ---------------------------------------------------------------------------


# Mapping from physics ``kind`` discriminator (the ``kind`` literal on a
# physics Config like CompositionalConfig / BlackOilConfig) to the registered
# plugin ``type_id`` used inside ``physics.plugin.type_id``.  Mirrors the map
# used by the MCP adapter (mcp-server-langchain/server/model_adapter.py).
_PHYSICS_KIND_TO_TYPE_ID: dict[str, str] = {
    "compositional": "physics/Compositional@v1",
    "black_oil": "physics/BlackOil@v1",
    "dead_oil": "physics/DeadOil@v1",
    "geothermal": "physics/Geothermal@v1",
}


def _expand_physics_preset(
    preset: Preset, overrides: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Turn a physics preset into a ``StrictPhysicsSpec``-shaped dict.

    Lifts ``components``/``phases`` (defined on the preset's Config so
    presets can ship a default composition) to the top-level PhysicsSpec
    fields, wraps the remaining Config fields inside ``plugin.config`` keyed
    by the discriminator-derived ``type_id``, and carries
    ``preset.property_regions`` through.  ``overrides`` are merge-patched on
    top so callers can override ``n_points``, ``components``, etc.

    :param preset: loaded physics preset
    :type preset: Preset
    :param overrides: the original section dict with ``preset`` key removed
    :type overrides: dict[str, Any]
    :return: ``(physics_dict, plugin_registry_dict_or_None)`` — the second
        value, when not ``None``, must be merged into the spec-level
        ``plugin_registry`` (preset-supplied plugin entries live alongside
        ``physics`` in ModelSpec, not inside it)
    :rtype: tuple[dict[str, Any], dict[str, Any] | None]
    """
    from darts.api.spec_utils import json_merge_patch

    preset_dict = preset.config.model_dump(exclude_none=True)
    kind = preset_dict.pop("kind", None)
    components = preset_dict.pop("components", None)
    phases = preset_dict.pop("phases", None)
    type_id = _PHYSICS_KIND_TO_TYPE_ID.get(kind, f"physics/{kind}@v1") if kind else None
    if type_id is None:
        raise ValueError(
            f"Physics preset '{preset.qualified_name}' has no 'kind' "
            f"discriminator on its config; cannot derive plugin.type_id."
        )
    physics_payload: dict[str, Any] = {
        "plugin": {"type_id": type_id, "config": preset_dict},
    }
    if components is not None:
        physics_payload["components"] = components
    if phases is not None:
        physics_payload["phases"] = phases
    if preset.property_regions is not None:
        physics_payload["property_regions"] = [
            dict(region) for region in preset.property_regions
        ]
    if overrides:
        physics_payload = json_merge_patch(physics_payload, overrides)
    return physics_payload, preset.plugin_registry


def _expand_generic_preset(preset: Preset, overrides: dict[str, Any]) -> dict[str, Any]:
    """Expand a non-physics preset by dumping its config and merging overrides.

    Works for any section whose preset Config maps 1-to-1 onto the section
    schema — reservoir (``StructReservoirConfig`` / ``CPGReservoirConfig``),
    ``sim_params`` (``SimParamsConfig``), etc.

    :param preset: loaded preset
    :type preset: Preset
    :param overrides: the original section dict with ``preset`` key removed
    :type overrides: dict[str, Any]
    :return: expanded section dict, ready for Pydantic validation
    :rtype: dict[str, Any]
    """
    from darts.api.spec_utils import json_merge_patch

    payload = preset.config.model_dump(exclude_none=True)
    if overrides:
        payload = json_merge_patch(payload, overrides)
    return payload


# Section keys that may carry a ``"preset"`` shorthand and the expander to
# apply.  Sections not listed pass through untouched.  Physics is handled
# separately because it must also surface a sibling ``plugin_registry``.
_GENERIC_PRESET_SECTIONS: tuple[str, ...] = (
    "reservoir",
    "sim_params",
    "wells",
    "initial_conditions",
    "well_controls",
    "output",
)


def resolve_section_presets(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Expand every ``{"preset": "<name>", ...overrides}`` section in a raw
    ModelSpec dict.

    Walks the top-level section keys of ``spec_dict`` (``reservoir``,
    ``physics``, ``sim_params``, ...) and for each section dict whose first
    key is ``"preset"`` (or which contains a ``"preset"`` key), loads the
    referenced preset, builds the section-appropriate payload, and merges
    any sibling keys on top as RFC-7396 overrides.  Returns a new dict;
    the input is not mutated.

    For physics presets, any ``plugin_registry`` block shipped by the preset
    is merged into ``spec_dict["plugin_registry"]`` (modules + entries are
    concatenated; explicit caller-provided entries take precedence by
    ``type_id``).  This mirrors the MCP adapter's
    :func:`build_physics_patch` so the single-JSON path is symmetric.

    :param spec_dict: raw JSON-decoded ModelSpec dict
    :type spec_dict: dict[str, Any]
    :return: expanded dict with all section-level presets resolved
    :rtype: dict[str, Any]
    """
    result = deepcopy(spec_dict)
    surfaced_plugin_registry: dict[str, Any] | None = None

    # Physics: special expansion + may surface a sibling plugin_registry.
    physics = result.get("physics")
    if isinstance(physics, dict) and "preset" in physics:
        preset_name = physics["preset"]
        overrides = {k: v for k, v in physics.items() if k != "preset"}
        preset = load_preset(preset_name)
        expanded, surfaced_plugin_registry = _expand_physics_preset(preset, overrides)
        result["physics"] = expanded

    # Generic sections: dump preset config, merge overrides.
    for section_key in _GENERIC_PRESET_SECTIONS:
        section = result.get(section_key)
        if isinstance(section, dict) and "preset" in section:
            preset_name = section["preset"]
            overrides = {k: v for k, v in section.items() if k != "preset"}
            preset = load_preset(preset_name)
            result[section_key] = _expand_generic_preset(preset, overrides)

    # Merge any preset-supplied plugin_registry into the spec-level one.
    if surfaced_plugin_registry is not None:
        existing = result.get("plugin_registry")
        result["plugin_registry"] = _merge_plugin_registries(
            surfaced_plugin_registry, existing
        )

    return result


def _merge_plugin_registries(
    preset_block: dict[str, Any], caller_block: dict[str, Any] | None
) -> dict[str, Any]:
    """Concatenate ``modules`` and ``entries`` from preset and caller blocks.

    Caller-provided entries take precedence over preset-provided entries on
    the same ``type_id`` (caller wins, in keeping with override semantics
    elsewhere in the pre-pass).

    :param preset_block: plugin_registry from a physics preset (constructor
        paths already absolutized at preset load time)
    :type preset_block: dict[str, Any]
    :param caller_block: plugin_registry the caller put in the ModelSpec
        alongside ``physics`` (may be ``None``)
    :type caller_block: dict[str, Any] | None
    :return: merged plugin_registry dict
    :rtype: dict[str, Any]
    """
    caller_block = caller_block or {}
    preset_modules = list(preset_block.get("modules") or [])
    caller_modules = list(caller_block.get("modules") or [])
    merged_modules = preset_modules + [
        m for m in caller_modules if m not in preset_modules
    ]
    preset_entries = list(preset_block.get("entries") or [])
    caller_entries = list(caller_block.get("entries") or [])
    caller_type_ids = {e.get("type_id") for e in caller_entries if isinstance(e, dict)}
    merged_entries: list[dict[str, Any]] = [
        e
        for e in preset_entries
        if not (isinstance(e, dict) and e.get("type_id") in caller_type_ids)
    ]
    merged_entries.extend(caller_entries)
    merged: dict[str, Any] = {}
    if merged_modules:
        merged["modules"] = merged_modules
    if merged_entries:
        merged["entries"] = merged_entries
    # Preserve any extra caller-only keys (e.g. future schema additions).
    for k, v in caller_block.items():
        if k not in ("modules", "entries"):
            merged.setdefault(k, v)
    return merged


__all__ = [
    "DEFAULT_PRESET_ROOT",
    "Preset",
    "PresetMeta",
    "docs",
    "iter_presets",
    "list_presets",
    "load_preset",
    "load_preset_config",
    "load_preset_dir",
    "register_preset",
    "register_preset_directory_binding",
    "resolve_section_presets",
]
