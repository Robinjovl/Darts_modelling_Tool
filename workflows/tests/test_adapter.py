import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from workflows.adapter import (
    PROTOCOL_VERSION,
    InputSnapshot,
    ModelAdapter,
    ProtocolMismatch,
    add_generated_input,
    load_adapter,
    snapshot_inputs,
    stage_inputs,
)


class FakeAdapter(ModelAdapter):
    input_patterns = ("inputs/*.in",)

    def build(self, paths, realization):
        return {"inputs": paths["inputs"], "realization": realization}

    def apply(self, model, realization):
        model["applied"] = realization

    def run(self, model, report_times):
        model["ran"] = list(report_times)

    def observe(self, model, observation):
        return {"time": model["ran"], "values": {}}


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_hashed_read_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            (model_dir / "inputs").mkdir(parents=True)
            (model_dir / "inputs" / "a.in").write_text("1 2 3\n")
            (model_dir / "inputs" / "b.in").write_text("4 5 6\n")
            (model_dir / "inputs" / "notes.txt").write_text("ignored\n")
            snap_dir = Path(tmp) / "study" / "inputs"
            adapter = FakeAdapter(model_dir)
            snapshot = adapter.snapshot(snap_dir)
            self.assertEqual(sorted(snapshot.files), ["inputs/a.in", "inputs/b.in"])
            self.assertFalse(os.access(snap_dir / "inputs" / "a.in", os.W_OK))
            self.assertEqual(InputSnapshot.load(snap_dir).hash, snapshot.hash)
            again = snapshot_inputs(model_dir, FakeAdapter.input_patterns, snap_dir)
            self.assertEqual(again.hash, snapshot.hash)
            (model_dir / "inputs" / "a.in").write_text("changed\n")
            changed = snapshot_inputs(model_dir, FakeAdapter.input_patterns, snap_dir)
            self.assertNotEqual(changed.hash, snapshot.hash)
            generated = snap_dir / "mesh.msh"
            generated.write_text("mesh\n")
            with_mesh = add_generated_input(changed, generated)
            self.assertIn("mesh.msh", with_mesh.files)
            self.assertEqual(InputSnapshot.load(snap_dir).files, with_mesh.files)
            resnapshot = snapshot_inputs(
                model_dir, FakeAdapter.input_patterns, snap_dir
            )
            self.assertIn(
                "mesh.msh", resnapshot.files, "generated inputs survive a re-snapshot"
            )
            self.assertEqual(resnapshot.hash, with_mesh.hash)

    def test_stage_symlink_and_copy_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            (model_dir / "inputs").mkdir(parents=True)
            (model_dir / "inputs" / "a.in").write_text("1\n")
            adapter = FakeAdapter(model_dir)
            snapshot = adapter.snapshot(Path(tmp) / "study" / "inputs")
            work = Path(tmp) / "study" / "members" / "m0"
            paths = adapter.stage(snapshot, work)
            staged = paths["inputs"] / "inputs" / "a.in"
            self.assertTrue(staged.is_symlink() and staged.read_text() == "1\n")
            self.assertTrue(paths["inputs"].is_absolute())
            copied = stage_inputs(
                snapshot, Path(tmp) / "study" / "members" / "m1", mode="copy"
            )
            self.assertFalse((copied / "inputs" / "a.in").is_symlink())
            with self.assertRaises(ValueError):
                stage_inputs(snapshot, work, mode="teleport")
            model = adapter.build(paths, {"x": 1})
            adapter.apply(model, {"x": 2})
            adapter.run(model, [10.0, 20.0])
            self.assertEqual(adapter.observe(model, None)["time"], [10.0, 20.0])
            self.assertEqual(adapter.cost_hint(), ModelAdapter.member_seconds)

    def test_load_adapter_checks_protocol_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "fake_adapters"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "good.py").write_text(
                textwrap.dedent(
                    """
                    from workflows.tests.test_adapter import FakeAdapter as Adapter  # noqa: F401
                    """
                )
            )
            (pkg / "old.py").write_text(
                textwrap.dedent(
                    f"""
                    from workflows.tests.test_adapter import FakeAdapter

                    class Adapter(FakeAdapter):
                        PROTOCOL_VERSION = {PROTOCOL_VERSION + 1}
                    """
                )
            )
            sys.path.insert(0, tmp)
            try:
                adapter = load_adapter("fake_adapters.good", tmp)
                self.assertIsInstance(adapter, FakeAdapter)
                with self.assertRaises(ProtocolMismatch):
                    load_adapter("fake_adapters.old", tmp)
                with self.assertRaises(FileNotFoundError):
                    FakeAdapter(Path(tmp) / "missing")
            finally:
                sys.path.remove(tmp)


if __name__ == "__main__":
    unittest.main()
