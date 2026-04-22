"""Tests for the named-preset registry under ``darts.api.presets``.

Validates three guarantees:

1. Every JSON file shipped under ``models/presets/`` loads cleanly and
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
)

# ---------------------------------------------------------------------------
# Shipped presets validate
# ---------------------------------------------------------------------------


class TestShippedPresetsValidate:
    """Walk every preset under ``models/presets/`` and assert it loads."""

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
# Allow `python tests/python/test_presets.py` for quick local runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
