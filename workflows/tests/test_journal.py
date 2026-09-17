import tempfile
import unittest
from pathlib import Path

from workflows.journal import (
    StudyStore,
    atomic_write_json,
    dependency_lock,
    file_sha256,
    provenance,
)
from workflows.spec import ModelRef, ParameterSpec, RunResult, StudySpec


def make_result(member: str, status: str = "ok", **kwargs) -> RunResult:
    base = dict(
        member=member,
        attempt=1,
        status=status,
        spec_hash="s",
        input_hash="i",
        binary_fingerprint="b",
    )
    base.update(kwargs)
    return RunResult(**base)


class StudyStoreTests(unittest.TestCase):
    def test_layout_journal_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyStore(Path(tmp) / "study")
            spec = StudySpec(
                name="s",
                workflow="ensemble",
                model=ModelRef(model_dir="m", adapter="a"),
                parameters=[ParameterSpec(name="p", family="ScalarParam")],
                seed_root=1,
                design={"method": "lhs", "n": 2},
            )
            store.write_spec(spec)
            self.assertEqual(store.read_spec(), spec)
            store.write_manifest({"workflow_run_hash": spec.workflow_run_hash()})
            self.assertIn("workflow_run_hash", store.read_manifest())
            self.assertFalse(
                store.manifest_path.with_name("manifest.json.tmp").exists()
            )
            store.append_member_event(make_result("m0"))
            store.append_member_event(make_result("m1", status="timestep_failure"))
            store.append_member_event(
                make_result("m1", attempt=2, threads=2, equivalent=False)
            )
            events = store.events()
            self.assertEqual([e["seq"] for e in events], [0, 1, 2])
            self.assertTrue(all("ts" in e for e in events))
            self.assertFalse(events[-1]["equivalent"])
            self.assertEqual(store.completed_members(), {"m0", "m1"})
            self.assertTrue(store.member_dir("m0").is_dir())
            reopened = StudyStore(store.root)
            reopened.append_event({"stage": "done"})
            self.assertEqual(reopened.events()[-1]["seq"], 3)

    def test_member_events_require_identity_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyStore(tmp)
            with self.assertRaises(ValueError):
                store.append_member_event(make_result("m1", spec_hash=""))
            self.assertEqual(store.events(), [])

    def test_atomic_write_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            atomic_write_json(path, {"b": 1, "a": 2})
            text = path.read_text()
            self.assertLess(text.index('"a"'), text.index('"b"'))
            self.assertEqual(len(file_sha256(path)), 64)

    def test_provenance_and_dependency_lock(self):
        record = provenance()
        for key in (
            "code_revision",
            "engine_fingerprint",
            "dependency_lock",
            "platform",
        ):
            self.assertIn(key, record)
        lock = dependency_lock()
        for name in ("python", "numpy", "scipy", "jsonschema", "h5py", "dageo"):
            self.assertIn(name, lock)
        self.assertIsNotNone(lock["numpy"])


if __name__ == "__main__":
    unittest.main()
