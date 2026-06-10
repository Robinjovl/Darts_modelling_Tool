"""Tests for the named-preset registry under ``darts.api.presets``.

Validates three guarantees:

1. Every JSON file shipped under ``darts/api/presets_data/`` loads cleanly and
   validates against its registered Config class.  This is the early-warning
   for schema drift: rename or remove a Config field and the affected
   presets fail loudly here.
2. Public lookup helpers (``list_presets``, ``load_preset``,
   ``load_preset_config``, ``iter_presets``, ``docs``) behave as documented.
3. ``$preset`` reference resolution composes presets recursively and rejects
   cycles / missing names.
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from darts.api.presets import (
    DEFAULT_PRESET_ROOT,
    PresetMeta,
    _resolve_preset_refs,  # internal — exercised here
    docs,
    iter_presets,
    list_presets,
    load_preset,
    load_preset_config,
    load_preset_dir,
    register_preset,
    resolve_section_presets,
)
from darts.api.schemas import ModelSpec

# ---------------------------------------------------------------------------
# Shipped presets validate
# ---------------------------------------------------------------------------


class TestShippedPresetsValidate:
    """Walk every preset under ``darts/api/presets_data/`` and assert it loads."""

    def test_default_root_exists(self) -> None:
        """Refactor canary: the canonical preset tree must live where the
        loader expects it.
        """
        assert DEFAULT_PRESET_ROOT.is_dir(), (
            f"Default preset root missing: {DEFAULT_PRESET_ROOT}"
        )

    def test_at_least_one_preset_shipped(self) -> None:
        """Sanity check that we are exercising real files, not an empty dir."""
        names = list_presets()
        assert names, "No presets discovered under default root"

    @pytest.mark.parametrize("qualified_name", list_presets() or ["__none__"])
    def test_each_preset_loads(self, qualified_name: str) -> None:
        """Every shipped preset must validate against its Config class.

        Parametrization captures `list_presets()` at collection time so each
        preset shows up as its own test case in pytest output.
        """
        if qualified_name == "__none__":
            pytest.skip("No presets shipped to validate")
        preset = load_preset(qualified_name)
        assert isinstance(preset.meta, PresetMeta)
        assert preset.qualified_name == qualified_name
        # Round-trip the config to ensure model_dump produces a re-validatable
        # payload (catches custom-validator regressions).
        config_cls = type(preset.config)
        config_cls.model_validate(preset.config.model_dump())


# ---------------------------------------------------------------------------
# Public lookup API
# ---------------------------------------------------------------------------


class TestLookupAPI:
    """Behavioural tests for the public preset-lookup helpers."""

    def test_list_presets_returns_sorted(self) -> None:
        """``list_presets`` must return names in a deterministic order so
        downstream tooling (CLI, MCP) can rely on it for stable diffs.
        """
        names = list_presets()
        assert names == sorted(names)

    def test_list_presets_filters_by_category(self) -> None:
        """The ``category`` filter restricts to a directory prefix."""
        density_only = list_presets("evaluators/density")
        assert density_only, "Expected at least one density preset shipped"
        assert all(n.startswith("evaluators/density/") for n in density_only)

    def test_load_preset_unknown_raises_keyerror(self) -> None:
        """Unknown preset lookups raise :class:`KeyError`, not a silent ``None``."""
        with pytest.raises(KeyError):
            load_preset("evaluators/density/__does_not_exist__")

    def test_load_preset_config_returns_config_only(self) -> None:
        """``load_preset_config`` is a thin convenience that strips meta."""
        cfg = load_preset_config("evaluators/density/co2_brine")
        assert cfg.kind == "density_basic"
        assert cfg.dens0 == 1020.0

    def test_iter_presets_yields_full_objects(self) -> None:
        """``iter_presets`` yields :class:`Preset` instances with both meta
        and config populated — the surface MCP tools depend on.
        """
        seen = list(iter_presets("evaluators/density"))
        assert seen, "Expected density presets to iterate"
        for preset in seen:
            assert preset.meta.name
            assert preset.meta.description
            assert preset.config is not None

    def test_docs_combines_meta_and_schema(self) -> None:
        """``docs()`` is the self-documenting surface — it must surface both
        preset metadata and the underlying Config's JSON Schema so browsing
        tools can render a complete view in one call.
        """
        d = docs("evaluators/flash/constant_k_co2_brine")
        assert d["meta"]["name"] == "constant_k_co2_brine"
        assert d["schema"]["title"] == "ConstantKConfig"
        assert "K" in d["schema"]["properties"]
        assert d["config"]["kind"] == "constant_k"
        assert d["source_path"] is not None


# ---------------------------------------------------------------------------
# $preset reference resolution
# ---------------------------------------------------------------------------


class TestPresetRefs:
    """``$preset`` markers are resolved before validation, recursively."""

    def test_resolve_simple_ref(self) -> None:
        """A bare ``{"$preset": ...}`` is replaced by the target's config."""
        out = _resolve_preset_refs({"$preset": "evaluators/density/co2_brine"})
        assert out["kind"] == "density_basic"
        assert out["dens0"] == 1020.0

    def test_resolve_ref_inside_dict(self) -> None:
        """A ref nested under a key resolves to the target config dict."""
        out = _resolve_preset_refs(
            {"density_ev": {"$preset": "evaluators/density/spe1_oil"}}
        )
        assert out["density_ev"]["kind"] == "density_basic"
        assert out["density_ev"]["dens0"] == 740.0

    def test_resolve_ref_inside_list(self) -> None:
        """Refs inside list elements resolve too."""
        out = _resolve_preset_refs(
            [{"$preset": "evaluators/density/co2_brine"}, "literal"]
        )
        assert out[0]["kind"] == "density_basic"
        assert out[1] == "literal"

    def test_unknown_ref_raises(self) -> None:
        """Referring to a non-existent preset surfaces a :class:`KeyError`."""
        with pytest.raises(KeyError):
            _resolve_preset_refs({"$preset": "evaluators/density/__missing__"})

    def test_cycle_detection(self, tmp_path) -> None:
        """A self-referential preset chain raises :class:`ValueError`.

        Built with a temp preset dir so we can craft a cycle without polluting
        the shipped tree.
        """
        cycle_dir = tmp_path / "cycle_evaluators" / "density"
        cycle_dir.mkdir(parents=True)
        # a -> b -> a
        (cycle_dir / "a.json").write_text(
            textwrap.dedent("""
                {
                  "_meta": {"name": "a", "description": "cycle a"},
                  "config": {
                    "kind": "density_basic",
                    "dens0": 1000.0,
                    "compr": 0.0,
                    "_loop": {"$preset": "cycle_evaluators/density/b"}
                  }
                }
            """)
        )
        (cycle_dir / "b.json").write_text(
            textwrap.dedent("""
                {
                  "_meta": {"name": "b", "description": "cycle b"},
                  "config": {
                    "kind": "density_basic",
                    "dens0": 1000.0,
                    "compr": 0.0,
                    "_loop": {"$preset": "cycle_evaluators/density/a"}
                  }
                }
            """)
        )
        # The DensityBasicConfig forbids extras, so even if cycle resolution
        # were to succeed validation would fail.  The cycle check must fire
        # first with a ValueError describing the cycle.
        with pytest.raises(ValueError, match="Cyclic"):
            load_preset_dir(tmp_path)


# ---------------------------------------------------------------------------
# Runtime registration
# ---------------------------------------------------------------------------


class TestRuntimeRegistration:
    """Third-party callers can register presets without writing files."""

    def test_register_preset_makes_it_lookupable(self) -> None:
        """A runtime-registered preset is indistinguishable from a file-based
        one for downstream callers.
        """
        from darts.physics.properties.density import DensityBasicConfig

        register_preset(
            "tests/density/runtime_only",
            {"name": "runtime_only", "description": "registered in-process"},
            DensityBasicConfig(dens0=1234.0, compr=0.0),
        )
        loaded = load_preset("tests/density/runtime_only")
        assert loaded.config.dens0 == 1234.0
        assert loaded.source_path is None

    def test_load_preset_dir_overrides_existing(self, tmp_path) -> None:
        """Re-running ``load_preset_dir`` updates entries in place — useful
        for tests and hot-reloading workflows.
        """
        d = tmp_path / "evaluators" / "density"
        d.mkdir(parents=True)
        path = d / "scratch.json"

        def _write(dens0: float) -> None:
            path.write_text(
                json.dumps(
                    {
                        "_meta": {"name": "scratch", "description": "scratch"},
                        "config": {
                            "kind": "density_basic",
                            "dens0": dens0,
                            "compr": 0.0,
                        },
                    }
                )
            )

        _write(100.0)
        load_preset_dir(tmp_path)
        assert load_preset("evaluators/density/scratch").config.dens0 == 100.0

        _write(200.0)
        load_preset_dir(tmp_path)
        assert load_preset("evaluators/density/scratch").config.dens0 == 200.0


# ---------------------------------------------------------------------------
# Regression: DataRef survives validation on reservoir property fields
# ---------------------------------------------------------------------------


class TestDataRefOnReservoirFields:
    """DataRef on per-cell reservoir fields must not raise during validation.

    Regression for the broken ``Field(gt=0, ...)`` constraint that used to
    sit on ``ReservoirValue | DataRef`` unions and triggered
    ``TypeError: Unable to apply constraint`` against the DataRef branch.
    """

    def test_permx_as_dataref_validates(self) -> None:
        from darts.api.schemas import StrictReservoirSpec

        spec = StrictReservoirSpec.model_validate(
            {
                "type": "structured",
                "nx": 1,
                "ny": 1,
                "nz": 1,
                "dx": 1.0,
                "dy": 1.0,
                "dz": 1.0,
                "permx": {"kind": "path", "value": "permx.json"},
                "permy": 1.0,
                "permz": 1.0,
                "poro": 0.3,
                "depth": 1000.0,
            }
        )
        # permx should land as a DataRef-shaped value, not a number.
        assert spec.permx.kind == "path"
        assert spec.permx.value == "permx.json"


# ---------------------------------------------------------------------------
# Regression: DARTS_PRESET_ROOT actually overrides shipped presets
# ---------------------------------------------------------------------------


class TestPresetRootPrecedence:
    """The env-var preset root must win over the shipped default.

    Regression for the prior load order that registered the env root
    *before* the default root; the default-root files then overwrote
    same-name env-root entries in ``_PRESET_REGISTRY`` because
    ``_load_preset_from_file`` always writes by qualified name.
    """

    def test_env_root_overrides_shipped_default(self, tmp_path, monkeypatch) -> None:
        from darts.api import presets as presets_mod

        # Spell the override exactly the way the shipped preset is named so
        # we know it conflicts with a real default-root entry.
        override = {
            "_meta": {"name": "co2_brine", "description": "test override"},
            "config": {
                "kind": "density_basic",
                "dens0": 9876.0,
                "compr": 0.0,
            },
        }
        target = tmp_path / "evaluators" / "density" / "co2_brine.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(override))

        # Force a clean re-bootstrap so the env-root takes effect.
        monkeypatch.setenv("DARTS_PRESET_ROOT", str(tmp_path))
        presets_mod._PRESET_REGISTRY.clear()
        presets_mod._SEARCH_ROOTS.clear()
        try:
            loaded = load_preset("evaluators/density/co2_brine")
            assert loaded.config.dens0 == 9876.0
            assert str(loaded.source_path).startswith(str(tmp_path))
        finally:
            # Restore the registry from the shipped tree for downstream tests.
            presets_mod._PRESET_REGISTRY.clear()
            presets_mod._SEARCH_ROOTS.clear()


# ---------------------------------------------------------------------------
# Section-level preset resolution (single-JSON pre-pass)
# ---------------------------------------------------------------------------


class TestResolveSectionPresets:
    """Pre-pass that expands ``{"preset": "...", ...overrides}`` in a raw
    ModelSpec dict before Pydantic validation.

    Mirrors what the MCP adapter does on the server side
    (``build_physics_patch`` / ``build_reservoir_patch``) so the single-JSON
    workflow accepts the same modular composition idiom.
    """

    def test_reservoir_preset_expands_with_overrides(self) -> None:
        """Reservoir preset expands to a ReservoirUnion-validatable dict,
        with caller overrides applied on top.
        """
        spec = {
            "reservoir": {
                "preset": "reservoirs/cpg/brugge",
                "grid_file": "/abs/path/to/grid.grdecl",
                "prop_file": "/abs/path/to/reservoir.in",
            }
        }
        expanded = resolve_section_presets(spec)
        assert expanded["reservoir"]["type"] == "cpg"
        assert expanded["reservoir"]["grid_file"] == "/abs/path/to/grid.grdecl"
        # Numerical guards from the preset are preserved.
        assert expanded["reservoir"]["minpv"] == 1e-05

    def test_physics_preset_lifts_components_and_carries_property_regions(
        self,
    ) -> None:
        """Physics preset builds the PhysicsSpec envelope: kind →
        plugin.type_id, components/phases lifted to top level,
        property_regions and plugin_registry surfaced.
        """
        spec = {"physics": {"preset": "physics/dead_oil/cpg_deadoil_brugge"}}
        expanded = resolve_section_presets(spec)
        assert expanded["physics"]["plugin"]["type_id"] == "physics/Compositional@v1"
        assert expanded["physics"]["components"] == ["w", "o"]
        assert expanded["physics"]["phases"] == ["wat", "oil"]
        assert len(expanded["physics"]["property_regions"]) == 1
        # Preset-supplied plugin_registry is bubbled up to the spec level.
        assert "plugin_registry" in expanded
        assert expanded["plugin_registry"]["entries"][0]["type_id"] == (
            "pc/ModelProperties@v1"
        )

    def test_full_brugge_composition_validates_as_modelspec(self) -> None:
        """The headline use case: a six-line ModelSpec composed entirely
        from presets must validate against StrictModelSpec.
        """
        spec = {
            "reservoir": {
                "preset": "reservoirs/cpg/brugge",
                "grid_file": "/abs/path/to/grid.grdecl",
                "prop_file": "/abs/path/to/reservoir.in",
            },
            "physics": {"preset": "physics/dead_oil/cpg_deadoil_brugge"},
            "sim_params": {
                "preset": "sim_params/default_implicit",
                "runtime": 365.0,
            },
        }
        expanded = resolve_section_presets(spec)
        model = ModelSpec.model_validate(expanded)
        assert model.reservoir.type == "cpg"
        assert model.physics.plugin.type_id == "physics/Compositional@v1"
        # Override took effect on top of the preset baseline.
        assert model.sim_params.runtime == 365.0
        # Plugin registry was hoisted from physics preset to spec level.
        assert model.plugin_registry is not None
        assert len(model.plugin_registry.entries) == 1

    def test_sections_without_preset_pass_through_unchanged(self) -> None:
        """Sections that don't use the ``preset`` shorthand are left intact —
        no false positives on raw dicts that happen to mention preset-like
        keys elsewhere.
        """
        raw_reservoir = {
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
        }
        spec = {"reservoir": raw_reservoir}
        expanded = resolve_section_presets(spec)
        assert expanded["reservoir"] == raw_reservoir

    def test_caller_plugin_registry_entries_win_on_type_id_collision(self) -> None:
        """When the caller supplies a ``plugin_registry`` alongside a physics
        preset that ships its own, caller-provided entries override preset
        ones on the same ``type_id``.
        """
        spec = {
            "physics": {"preset": "physics/dead_oil/cpg_deadoil_brugge"},
            "plugin_registry": {
                "entries": [
                    {
                        "type_id": "pc/ModelProperties@v1",
                        "kind": "pc",
                        "constructor": "/caller/path.py:CustomPC",
                    }
                ]
            },
        }
        expanded = resolve_section_presets(spec)
        entries = expanded["plugin_registry"]["entries"]
        assert len(entries) == 1
        assert entries[0]["constructor"] == "/caller/path.py:CustomPC"


# ---------------------------------------------------------------------------
# Allow `python tests/python/test_presets.py` for quick local runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
