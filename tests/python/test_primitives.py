"""Tests for standalone primitive construction and composable model specs.

Validates that individual ModelSpec sections (reservoir, physics, sim_params,
wells, initial_conditions, well_controls, output) can be loaded from JSON,
validated, and applied independently via the public ``ModelBuilder.apply_*``
API.  Also tests composing a full model from shared primitive JSON files
using DataRef references.
"""

import json
import os
import sys

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PRIMITIVES = os.path.join(FIXTURES, "primitives")


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schema-level validation of individual primitives
# ---------------------------------------------------------------------------


class TestPrimitiveValidation:
    """Validate that each primitive JSON parses into its schema model."""

    def test_reservoir_structured(self):
        from darts.api.schemas import ReservoirSpec

        data = _load(os.path.join(PRIMITIVES, "reservoir_1d_100.json"))
        spec = ReservoirSpec.model_validate(data)
        assert spec.type == "structured"
        assert spec.nx == 100
        assert spec.poro == 0.2

    def test_reservoir_layered(self):
        from darts.api.schemas import ReservoirSpec

        data = _load(os.path.join(PRIMITIVES, "reservoir_3d_layered.json"))
        spec = ReservoirSpec.model_validate(data)
        assert spec.nx == 10
        assert len(spec.layers) == 3
        assert sum(ly.count for ly in spec.layers) == spec.nx * spec.ny * spec.nz

    def test_sim_params(self):
        from darts.api.schemas import SimParamsSpec

        data = _load(os.path.join(PRIMITIVES, "sim_params_default.json"))
        spec = SimParamsSpec.model_validate(data)
        assert spec.runtime == 100
        assert spec.max_ts == 1.0

    def test_wells(self):
        from darts.api.schemas import WellsSpec

        data = _load(os.path.join(PRIMITIVES, "wells_1d_endpoints.json"))
        spec = WellsSpec.model_validate(data)
        assert len(spec.wells) == 2
        assert spec.wells[0].name == "I1"
        assert spec.wells[1].name == "P1"

    def test_well_controls(self):
        from darts.api.schemas import WellControlsSpec

        data = _load(os.path.join(PRIMITIVES, "well_controls_bhp.json"))
        spec = WellControlsSpec.model_validate(data)
        assert spec.inj_bhp == 140.0
        assert spec.prod_bhp == 50.0

    def test_initial_conditions(self):
        from darts.api.schemas import InitialConditionsSpec

        data = _load(os.path.join(PRIMITIVES, "initial_conditions_2ph_comp.json"))
        spec = InitialConditionsSpec.model_validate(data)
        assert "pressure" in spec.by_array
        assert spec.by_array["CO2"] == 0.1

    def test_output(self):
        from darts.api.schemas import OutputSpec

        data = _load(os.path.join(PRIMITIVES, "output_default.json"))
        spec = OutputSpec.model_validate(data)
        assert spec.folder == "output"
        assert spec.sol_filename == "reservoir_solution_custom.h5"
        assert spec.well_filename == "well_data_custom.h5"
        assert spec.save_initial is False
        assert spec.all_phase_props is True
        assert spec.precision == "d"
        assert spec.compression == "gzip"
        assert spec.compression_level == 4
        assert spec.verbose is True


# ---------------------------------------------------------------------------
# Apply individual primitives via ModelBuilder public API
# ---------------------------------------------------------------------------


class TestPrimitiveApply:
    """Apply individual primitives to a mock model via ModelBuilder.apply_*."""

    @staticmethod
    def _make_model():
        """Create a minimal object satisfying DartsModelProtocol for testing."""

        class _Stub:
            def __init__(self):
                self.timer = None
                self.reservoir = None
                self.physics = None
                self._sim_params = {}

            def set_sim_params(self, **kwargs):
                self._sim_params = kwargs

            def set_sim_params_from_config(self, config):
                self.set_sim_params(**config.model_dump(exclude_none=True))

        return _Stub()

    def test_apply_sim_params(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import SimParamsSpec

        data = _load(os.path.join(PRIMITIVES, "sim_params_default.json"))
        spec = SimParamsSpec.model_validate(data)
        model = self._make_model()
        ModelBuilder.apply_sim_params(spec, model)
        assert model._sim_params["runtime"] == 100
        assert model._sim_params["max_ts"] == 1.0
        assert model._sim_params["it_newton"] == 10

    def test_apply_wells_stores_spec(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import WellsSpec

        data = _load(os.path.join(PRIMITIVES, "wells_1d_endpoints.json"))
        spec = WellsSpec.model_validate(data)
        model = self._make_model()
        ModelBuilder.apply_wells(spec, model)
        assert hasattr(model, "_wells_spec")
        assert model._wells_spec.wells[0].name == "I1"

    def test_apply_initial_conditions_stores_spec(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import InitialConditionsSpec

        data = _load(os.path.join(PRIMITIVES, "initial_conditions_2ph_comp.json"))
        spec = InitialConditionsSpec.model_validate(data)
        model = self._make_model()
        ModelBuilder.apply_initial_conditions(spec, model)
        assert hasattr(model, "_initial_conditions_spec")
        assert model._initial_conditions_spec.by_array["pressure"] == 50.0

    def test_apply_well_controls_stores_spec(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import WellControlsSpec

        data = _load(os.path.join(PRIMITIVES, "well_controls_bhp.json"))
        spec = WellControlsSpec.model_validate(data)
        model = self._make_model()
        ModelBuilder.apply_well_controls(spec, model)
        assert hasattr(model, "_well_controls_spec")

    def test_apply_output_stores_spec(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import OutputSpec

        data = _load(os.path.join(PRIMITIVES, "output_default.json"))
        spec = OutputSpec.model_validate(data)
        model = self._make_model()
        ModelBuilder.apply_output(spec, model)
        assert model._output_spec.folder == "output"
        assert model._output_spec.to_set_output_kwargs() == {
            "output_folder": "output",
            "sol_filename": "reservoir_solution_custom.h5",
            "well_filename": "well_data_custom.h5",
            "save_initial": False,
            "all_phase_props": True,
            "precision": "d",
            "compression": "gzip",
            "compression_level": 4,
            "verbose": True,
        }


# ---------------------------------------------------------------------------
# Resolve individual section via resolve_section
# ---------------------------------------------------------------------------


class TestResolveSection:
    """Test that resolve_section handles both inline dicts and DataRefs."""

    def test_resolve_inline_dict(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import SimParamsSpec

        data = _load(os.path.join(PRIMITIVES, "sim_params_default.json"))
        result = ModelBuilder.resolve_section(data, SimParamsSpec, None, None)
        assert isinstance(result, SimParamsSpec)
        assert result.runtime == 100

    def test_resolve_dataref(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import SimParamsSpec

        ref = {
            "kind": "path",
            "value": "primitives/sim_params_default.json",
            "format": "json",
        }
        result = ModelBuilder.resolve_section(ref, SimParamsSpec, FIXTURES, None)
        assert isinstance(result, SimParamsSpec)
        assert result.runtime == 100

    def test_resolve_reservoir_dataref(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import ReservoirSpec

        ref = {
            "kind": "path",
            "value": "primitives/reservoir_1d_100.json",
            "format": "json",
        }
        raw = ModelBuilder.resolve_section(ref, None, FIXTURES, None)
        spec = ModelBuilder.validate_reservoir_spec(raw)
        assert isinstance(spec, ReservoirSpec)
        assert spec.nx == 100


# ---------------------------------------------------------------------------
# Composed model from shared primitives via DataRef
# ---------------------------------------------------------------------------


class TestComposedModel:
    """Validate a full ModelSpec composed from shared primitive DataRefs."""

    def test_validate_composed_spec(self):
        from darts.api.schemas import ModelSpec

        data = _load(os.path.join(FIXTURES, "composed_2ph_comp.json"))
        # The composed spec uses DataRef objects for most sections.
        # ModelSpec validation should accept DataRef in place of inline specs.
        spec = ModelSpec.model_validate(data)
        assert spec.physics is not None
        assert spec.physics.components == ["CO2", "C1", "H2O"]
        # reservoir, wells, sim_params etc. are DataRefs — check they parsed
        assert spec.reservoir.kind == "path"
        assert spec.wells.kind == "path"
        assert spec.sim_params.kind == "path"

    def test_resolve_all_sections(self):
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import (
            InitialConditionsSpec,
            ModelSpec,
            OutputSpec,
            SimParamsSpec,
            WellControlsSpec,
            WellsSpec,
        )

        data = _load(os.path.join(FIXTURES, "composed_2ph_comp.json"))
        spec = ModelSpec.model_validate(data)
        base = FIXTURES

        # Each DataRef section should resolve to its typed spec
        wells = ModelBuilder.resolve_section(spec.wells, WellsSpec, base, None)
        assert isinstance(wells, WellsSpec)
        assert len(wells.wells) == 2

        sp = ModelBuilder.resolve_section(spec.sim_params, SimParamsSpec, base, None)
        assert isinstance(sp, SimParamsSpec)
        assert sp.runtime == 100

        ics = ModelBuilder.resolve_section(
            spec.initial_conditions, InitialConditionsSpec, base, None
        )
        assert isinstance(ics, InitialConditionsSpec)
        assert ics.by_array["pressure"] == 50.0

        wc = ModelBuilder.resolve_section(
            spec.well_controls, WellControlsSpec, base, None
        )
        assert isinstance(wc, WellControlsSpec)
        assert wc.inj_bhp == 140.0

        out = ModelBuilder.resolve_section(spec.output, OutputSpec, base, None)
        assert isinstance(out, OutputSpec)
        assert out.precision == "d"

        res_raw = ModelBuilder.resolve_section(spec.reservoir, None, base, None)
        res = ModelBuilder.validate_reservoir_spec(res_raw)
        assert res.nx == 100


# ---------------------------------------------------------------------------
# Reusability: same reservoir primitive in different model compositions
# ---------------------------------------------------------------------------


class TestSharedPrimitiveReuse:
    """Verify the same primitive JSON validates in multiple model contexts."""

    def test_same_reservoir_two_sim_params(self):
        """Same reservoir, different sim_params — both validate."""
        from darts.api.schemas import ReservoirSpec, SimParamsSpec

        reservoir = ReservoirSpec.model_validate(
            _load(os.path.join(PRIMITIVES, "reservoir_1d_100.json"))
        )
        sp_default = SimParamsSpec.model_validate(
            _load(os.path.join(PRIMITIVES, "sim_params_default.json"))
        )
        sp_fast = SimParamsSpec.model_validate(
            _load(os.path.join(PRIMITIVES, "sim_params_fast.json"))
        )

        # Both share the same reservoir but differ in sim_params
        assert reservoir.nx == 100
        assert sp_default.runtime == 100
        assert sp_fast.runtime == 300
        assert sp_default.max_ts != sp_fast.max_ts

    def test_apply_different_sim_params_to_same_model(self):
        """Apply two sim_params configs sequentially — second overwrites first."""
        from darts.api.builder import ModelBuilder
        from darts.api.schemas import SimParamsSpec

        class _Stub:
            def __init__(self):
                self.timer = self.reservoir = self.physics = None
                self._kwargs = {}

            def set_sim_params(self, **kwargs):
                self._kwargs = kwargs

            def set_sim_params_from_config(self, config):
                self.set_sim_params(**config.model_dump(exclude_none=True))

        model = _Stub()

        sp1 = SimParamsSpec.model_validate(
            _load(os.path.join(PRIMITIVES, "sim_params_default.json"))
        )
        ModelBuilder.apply_sim_params(sp1, model)
        assert model._kwargs["runtime"] == 100

        sp2 = SimParamsSpec.model_validate(
            _load(os.path.join(PRIMITIVES, "sim_params_fast.json"))
        )
        ModelBuilder.apply_sim_params(sp2, model)
        assert model._kwargs["runtime"] == 300


# ---------------------------------------------------------------------------
# Standalone runner (for ``darts test_primitives.py`` in CI)
# ---------------------------------------------------------------------------


def _run_all() -> int:
    """Discover and run all test methods, return number of failures."""
    test_classes = [
        TestPrimitiveValidation,
        TestPrimitiveApply,
        TestResolveSection,
        TestComposedModel,
        TestSharedPrimitiveReuse,
    ]
    n_total = 0
    n_failed = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            n_total += 1
            label = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                print(f"  PASS  {label}")
            except Exception as exc:
                n_failed += 1
                print(f"  FAIL  {label}: {exc}")
    print(f"\n{n_total - n_failed}/{n_total} passed, {n_failed} failed")
    return n_failed


if __name__ == "__main__":
    sys.exit(_run_all())
