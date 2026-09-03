"""Darts-gated integration test of the Brugge proxy adapter (``WORKFLOWS_RUN_DARTS=1``)."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.adapter import load_adapter
from workflows.spec import ObservationSpec

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "Uniform_Brugge"


@unittest.skipUnless(
    os.environ.get("WORKFLOWS_RUN_DARTS") == "1", "set WORKFLOWS_RUN_DARTS=1"
)
class BruggeProxyAdapterTests(unittest.TestCase):
    def test_snapshot_stage_build_apply_run_observe(self):
        adapter = load_adapter("workflows.adapters.brugge_proxy", MODEL_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study"
            snapshot = adapter.snapshot(study / "inputs")
            self.assertIn("Brugge_model.msh", snapshot.files)
            self.assertIn("Brugge_struct/physics.in", snapshot.files)
            mesh = Path(snapshot.root) / "Brugge_model.msh"
            mesh_mtime = mesh.stat().st_mtime_ns
            observation = ObservationSpec(
                wells=["P1", "I1"],
                quantities=["oil_rate", "gas_rate", "bhp"],
                report_times=[10.0, 20.0],
            )
            results = []
            for member, realization in enumerate(
                [{}, {"tran_multiplier": 0.5, "bhp": {"P1": 140.0}}]
            ):
                work = study / "members" / f"m{member}"
                paths = adapter.stage(snapshot, work)
                model = adapter.build(paths, realization)
                adapter.apply(model, realization)
                adapter.run(model, observation.report_times)
                results.append(adapter.observe(model, observation))
                output = Path(paths["workdir"]) / "output"
                self.assertTrue((output / "well_data.h5").exists())
                self.assertFalse(
                    any(
                        p.suffix in (".vtu", ".vtk", ".xlsx", ".pkl")
                        for p in output.rglob("*")
                    )
                )
                del model
            self.assertEqual(
                mesh.stat().st_mtime_ns,
                mesh_mtime,
                "staged mesh must not be regenerated",
            )
            base, perturbed = results
            self.assertEqual(base["time"], [10.0, 20.0])
            self.assertEqual(
                set(base["values"]),
                {
                    "P1:oil_rate",
                    "P1:gas_rate",
                    "P1:bhp",
                    "I1:oil_rate",
                    "I1:gas_rate",
                    "I1:bhp",
                },
            )
            self.assertTrue(np.allclose(base["values"]["P1:bhp"], 150.0))
            self.assertTrue(np.allclose(perturbed["values"]["P1:bhp"], 140.0))
            self.assertFalse(
                np.allclose(
                    base["values"]["P1:oil_rate"], perturbed["values"]["P1:oil_rate"]
                )
            )


if __name__ == "__main__":
    unittest.main()
