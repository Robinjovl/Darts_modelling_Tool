"""Darts-gated integration tests of the Brugge proxy adapter and executor (``WORKFLOWS_RUN_DARTS=1``)."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workflows.adapter import load_adapter
from workflows.executor import IsolatedExecutor, MemberTask, run_members
from workflows.journal import StudyStore
from workflows.spec import ObservationSpec

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "Uniform_Brugge"
ADAPTER = "workflows.adapters.brugge_proxy"


@unittest.skipUnless(
    os.environ.get("WORKFLOWS_RUN_DARTS") == "1", "set WORKFLOWS_RUN_DARTS=1"
)
class BruggeProxyAdapterTests(unittest.TestCase):
    def test_snapshot_stage_build_apply_run_observe(self):
        adapter = load_adapter(ADAPTER, MODEL_DIR)
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

    def test_executor_runs_members_and_journals_attempts(self):
        adapter = load_adapter(ADAPTER, MODEL_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            store = StudyStore(Path(tmp) / "study")
            snapshot = adapter.snapshot(store.root / "inputs")
            observation = ObservationSpec(
                wells=["P1"], quantities=["oil_rate"], report_times=[10.0]
            )
            tasks = [
                MemberTask(
                    member=f"m{i}",
                    study_root=str(store.root),
                    adapter=ADAPTER,
                    model_dir=str(MODEL_DIR),
                    snapshot=snapshot.to_dict() | {},
                    realization={"tran_multiplier": 1.0 + 0.1 * i},
                    report_times=[10.0],
                    observation=observation.__dict__,
                    spec_hash="s",
                    input_hash=snapshot.hash,
                    binary_fingerprint="b",
                )
                for i in range(2)
            ]
            for task in tasks:
                task.snapshot = {"root": snapshot.root, "files": snapshot.files}
            results = run_members(
                store, tasks, IsolatedExecutor(n_workers=2, timeout_s=300, retries=0)
            )
            self.assertEqual([r.status for r in results], ["ok", "ok"])
            events = store.events()
            self.assertEqual({e["member"] for e in events}, {"m0", "m1"})
            self.assertTrue(
                all(e["engine_stats"]["n_timesteps_total"] > 0 for e in events)
            )
            self.assertTrue(all("well_data.h5" in e["output_hashes"] for e in events))
            self.assertEqual(store.completed_members(), {"m0", "m1"})
            for task in tasks:
                member_dir = store.member_dir(task.member)
                self.assertTrue((member_dir / "result.json").exists())
                self.assertTrue((member_dir / "member.log").exists())
            first = results[0].payload["observation"]["values"]["P1:oil_rate"]
            second = results[1].payload["observation"]["values"]["P1:oil_rate"]
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    os.environ.get("WORKFLOWS_RUN_DARTS") == "1", "set WORKFLOWS_RUN_DARTS=1"
)
class BruggeProxyEnsembleTests(unittest.TestCase):
    def test_small_lhs_ensemble_runs_resumes_and_analyzes(self):
        from workflows.ensemble import analyze, run_study
        from workflows.spec import ModelRef, ParameterSpec, StudySpec

        with tempfile.TemporaryDirectory() as tmp:
            spec = StudySpec(
                name="ens-smoke",
                workflow="ensemble",
                model=ModelRef(model_dir=str(MODEL_DIR), adapter=ADAPTER),
                parameters=[
                    ParameterSpec(
                        name="tm",
                        family="LogScalarParam",
                        args={"low": 0.5, "high": 2.0},
                    ),
                    ParameterSpec(
                        name="k",
                        family="LogPermField",
                        args={"n_components": 3, "range_m": 1500.0},
                    ),
                ],
                seed_root=11,
                observations=ObservationSpec(
                    wells=["P1"], quantities=["oil_rate"], report_times=[10.0]
                ),
                design={"method": "lhs", "n": 3},
            )
            spec.compute.max_workers = 3
            spec.compute.walltime_s = 300
            study = Path(tmp) / "study"
            first = run_study(
                spec,
                study,
                executor=IsolatedExecutor(n_workers=3, timeout_s=300, retries=0),
            )
            self.assertEqual(first["n_ok"], 3)
            second = run_study(
                spec,
                study,
                executor=IsolatedExecutor(n_workers=3, timeout_s=300, retries=0),
            )
            self.assertEqual(second["n_run"], 0)
            analysis = analyze(study)
            entry = analysis["quantities"]["P1:oil_rate"]
            self.assertIn("P50", entry["percentiles"])
            manifest = StudyStore(study).read_manifest()
            self.assertEqual(manifest["n_members"], 3)
            self.assertEqual(len(manifest["labels"]), 4)
            self.assertIn("truth_generation_hash", manifest["identities"])
            self.assertTrue(
                (
                    Path(
                        manifest["provenance"]["engine_fingerprint"]
                        and study / "inputs" / "geometry.json"
                    )
                ).exists()
            )
