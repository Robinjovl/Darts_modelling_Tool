import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.spec import (
    SCORE_VERSION,
    SEED_STREAMS,
    ModelRef,
    ObservationSpec,
    ParameterSpec,
    RunResult,
    SeedLedger,
    StudySpec,
    validate,
)

PROVENANCE = {
    "input_hashes": {"a": "1"},
    "engine_fingerprint": "e",
    "code_revision": "c",
    "dependency_lock": {},
}


def make_spec(**overrides) -> StudySpec:
    base = dict(
        name="ens-proxy",
        workflow="ensemble",
        model=ModelRef(
            model_dir="models/Uniform_Brugge", adapter="workflows.adapters.brugge_proxy"
        ),
        parameters=[
            ParameterSpec(
                name="logperm", family="LogPermField", args={"n_components": 10}
            )
        ],
        seed_root=20260903,
        observations=ObservationSpec(
            wells=["P1"], quantities=["oil_rate"], report_times=[120.0, 240.0]
        ),
        design={"method": "lhs", "n": 100},
        reference={"n_reference": 2000},
    )
    base.update(overrides)
    return StudySpec(**base)


class SeedLedgerTests(unittest.TestCase):
    def test_streams_are_idempotent_and_independent(self):
        ledger = SeedLedger(42)
        a = ledger.generator("design").random(3)
        b = SeedLedger(42).generator("design").random(3)
        c = ledger.generator("geology").random(3)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(a, c))
        self.assertFalse(np.allclose(a, ledger.generator("design", index=1).random(3)))

    def test_rejects_unknown_stream_bad_root_and_non_integer_index(self):
        with self.assertRaises(KeyError):
            SeedLedger(1).generator("nope")
        with self.assertRaises(ValueError):
            SeedLedger(-1)
        with self.assertRaises(ValueError):
            SeedLedger(1).generator("agent", 2.9)
        with self.assertRaises(ValueError):
            SeedLedger(1).generator("agent", -1)
        self.assertEqual(SeedLedger(7).to_dict()["streams"], list(SEED_STREAMS))


class StudySpecTests(unittest.TestCase):
    def test_round_trip_and_validation(self):
        spec = make_spec()
        data = spec.to_dict()
        validate(data)
        self.assertEqual(StudySpec.from_dict(data), spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "study.json"
            spec.dump(path)
            self.assertEqual(StudySpec.load(path), spec)
            self.assertEqual(json.loads(path.read_text())["workflow"], "ensemble")

    def test_schema_rejects_unknown_workflow_extra_keys_and_empty_parameters(self):
        import jsonschema

        bad = make_spec().to_dict()
        bad["workflow"] = "magic"
        with self.assertRaises(jsonschema.ValidationError):
            validate(bad)
        extra = make_spec().to_dict()
        extra["surprise"] = 1
        with self.assertRaises(jsonschema.ValidationError):
            validate(extra)
        empty = make_spec().to_dict()
        empty["parameters"] = []
        with self.assertRaises(jsonschema.ValidationError):
            validate(empty)

    def test_integral_json_numbers_are_coerced_at_load(self):
        data = json.loads(json.dumps(make_spec().to_dict()))
        data["seed_root"] = 1.0
        data["compute"]["max_workers"] = 4.0
        loaded = StudySpec.from_dict(data)
        self.assertIsInstance(loaded.seed_root, int)
        self.assertIsInstance(loaded.compute.max_workers, int)
        loaded.seeds().generator("design")

    def test_identities_separate_working_and_truth_settings(self):
        spec = make_spec()
        truth = spec.truth_generation_hash(PROVENANCE)
        run = spec.workflow_run_hash()
        other_working = make_spec(design={"method": "lhs", "n": 200})
        self.assertEqual(other_working.truth_generation_hash(PROVENANCE), truth)
        self.assertNotEqual(other_working.workflow_run_hash(), run)
        other_reference = make_spec(reference={"n_reference": 4000})
        self.assertNotEqual(other_reference.truth_generation_hash(PROVENANCE), truth)
        other_inputs = dict(PROVENANCE, input_hashes={"a": "2"})
        self.assertNotEqual(spec.truth_generation_hash(other_inputs), truth)
        with self.assertRaises(KeyError):
            spec.truth_generation_hash({"input_hashes": {}})
        with self.assertRaises(ValueError):
            spec.truth_generation_hash(dict(PROVENANCE, engine_fingerprint=None))
        identities = spec.identities(PROVENANCE)
        self.assertEqual(
            set(identities),
            {"workflow_run_hash", "truth_generation_hash", "score_version"},
        )
        self.assertEqual(identities["score_version"], SCORE_VERSION)

    def test_run_result_defaults(self):
        result = RunResult(
            member="m0",
            attempt=1,
            status="ok",
            spec_hash="s",
            input_hash="i",
            binary_fingerprint="b",
        )
        self.assertTrue(result.equivalent)
        self.assertEqual(result.engine_stats, {})


if __name__ == "__main__":
    unittest.main()
