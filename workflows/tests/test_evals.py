import json
import os
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

from workflows.evals import sandbox as sb
from workflows.evals.broker import Broker, submit
from workflows.evals.controller import (
    ShellRunner,
    hash_tree,
    run_case,
    verify_after_exit,
)
from workflows.spec import ModelRef, ObservationSpec, ParameterSpec, StudySpec

REPO = Path(__file__).resolve().parents[2]


def make_spec_dict(name="b") -> dict:
    return StudySpec(
        name=name,
        workflow="ensemble",
        model=ModelRef(
            model_dir=str(REPO / "models" / "Uniform_Brugge"),
            adapter="workflows.adapters.brugge_proxy",
        ),
        parameters=[
            ParameterSpec(
                name="tm", family="LogScalarParam", args={"low": 0.5, "high": 2.0}
            )
        ],
        seed_root=1,
        design={"method": "lhs", "n": 2},
    ).to_dict()


class BrokerTests(unittest.TestCase):
    def test_round_trip_with_fake_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def runner(command, spec_path, study_dir):
                calls.append((command, spec_path.name, study_dir.name))
                return {"result": {"ok": True}}

            broker = Broker(Path(tmp) / "b.sock", Path(tmp) / "results", runner=runner)
            thread = broker.serve_in_thread()
            try:
                reply = submit(
                    Path(tmp) / "b.sock",
                    make_spec_dict(),
                    "study-a",
                    command="estimate",
                )
                self.assertEqual(reply["status"], "ok")
                self.assertEqual(reply["result"], {"ok": True})
                self.assertEqual(calls, [("estimate", "study-a.spec.json", "study-a")])
                bad = submit(Path(tmp) / "b.sock", {"name": "x"}, "study-b")
                self.assertEqual(bad["status"], "error")
                evil = submit(Path(tmp) / "b.sock", make_spec_dict(), "../escape")
                self.assertEqual(evil["status"], "ok")
                self.assertTrue(
                    Path(evil["study_dir"]).parent == (Path(tmp) / "results").resolve()
                )
            finally:
                broker.shutdown()
                broker.server_close()
            self.assertIsInstance(thread, threading.Thread)
            self.assertEqual(len(broker.log), 2)


@unittest.skipUnless(sb.available(), "bwrap not available")
class SandboxTests(unittest.TestCase):
    def test_manifest_and_canary(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = sb.SandboxSpec(run_dir=Path(tmp) / "run", credentials=None)
            spec.run_dir.mkdir()
            cmd = spec.command(["/bin/true"])
            self.assertEqual(cmd[0], "bwrap")
            self.assertIn("--unshare-pid", cmd)
            self.assertIn(str(REPO / "workflows" / "evals" / "truth"), cmd)
            checks = spec.canary()
            self.assertTrue(checks["passed"], checks)

    def test_run_case_with_shell_runner_and_post_exit_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = make_spec_dict("shell-study")
            run_dir = Path(tmp) / "run" / "agent"
            run_dir.mkdir(parents=True)
            spec_path = (
                run_dir / "spec.json"
            )  # inside the writable bind; /tmp is tmpfs in the sandbox
            spec_path.write_text(json.dumps(spec))
            python = sys.executable
            script = (
                "import json, os, sys; sys.path.insert(0, os.environ['PYTHONPATH']);"
                "from workflows.evals.broker import submit;"
                f"spec = json.load(open('{spec_path}'));"
                "r = submit(os.environ['WORKFLOWS_BROKER'], spec, 'shell-study', command='estimate');"
                "print(json.dumps({'usage': {'input_tokens': 0}, 'reply': r['status']}))"
            )
            runner = ShellRunner([python, "-c", script], timeout_s=120)
            case = {"name": "shell-case", "prompt": "noop", "budget": {}}
            with unittest.mock.patch.object(
                Broker,
                "run_workflow",
                staticmethod(
                    lambda command, spec_path, study_dir: {
                        "result": {"command": command}
                    }
                ),
            ):
                record = run_case(case, runner, Path(tmp) / "run")
            self.assertTrue(record["canary"]["passed"], record["canary"])
            self.assertEqual(
                record["agent"]["exit_code"], 0, (record["agent"], runner.last_stderr)
            )
            self.assertEqual(record["agent"]["usage"].get("reply"), "ok")
            self.assertEqual(record["verification"]["broker_requests"], 1)
            self.assertTrue(verify_after_exit(Path(tmp) / "run")["intact"])
            (Path(tmp) / "run" / "results" / "shell-study.spec.json").write_text(
                "tampered"
            )
            self.assertFalse(verify_after_exit(Path(tmp) / "run")["intact"])
            self.assertIn(
                "shell-study.spec.json", hash_tree(Path(tmp) / "run" / "results")
            )


@unittest.skipUnless(
    sb.available() and os.environ.get("WORKFLOWS_RUN_DARTS") == "1",
    "needs bwrap and WORKFLOWS_RUN_DARTS=1",
)
class SandboxDartsTests(unittest.TestCase):
    def test_placement_case_through_broker_and_scorer(self):
        from workflows.evals import scorer

        case = json.loads(
            (
                REPO / "workflows" / "evals" / "cases" / "opt-place-proxy-smoke.json"
            ).read_text()
        )
        spec = StudySpec(
            name=case["study"],
            workflow="optimize",
            model=ModelRef(
                model_dir=str(REPO / "models" / "Uniform_Brugge"),
                adapter="workflows.adapters.brugge_proxy",
            ),
            parameters=[
                ParameterSpec(
                    name="I1", family="ScalarParam", args={"target": "well_xyz"}
                )
            ],
            seed_root=case["seeds"][0],
            observations=ObservationSpec(
                wells=["P1", "P2", "P3"],
                quantities=["oil_rate"],
                report_times=[20.0, 40.0, 60.0],
            ),
            design={
                "driver": "exhaustive",
                "well": "I1",
                "objective": "cumulative_oil",
                "min_spacing_m": 300.0,
                "max_candidates": 2,
            },
        ).to_dict()
        with tempfile.TemporaryDirectory(dir=os.environ.get("SCRATCH")) as tmp:
            run_dir = Path(tmp) / "run" / "agent"
            run_dir.mkdir(parents=True)
            (run_dir / "spec.json").write_text(json.dumps(spec))
            script = (
                "import json, os, sys; sys.path.insert(0, os.environ['PYTHONPATH']);"
                "from workflows.evals.broker import submit;"
                "spec = json.load(open('spec.json'));"
                f"r = submit(os.environ['WORKFLOWS_BROKER'], spec, '{case['study']}');"
                "print(json.dumps({'reply': r['status'], 'result': r.get('result')}))"
            )
            record = run_case(
                case, ShellRunner([sys.executable, "-c", script]), Path(tmp) / "run"
            )
            self.assertEqual(record["agent"]["exit_code"], 0, record["agent"])
            self.assertEqual(record["agent"]["usage"]["reply"], "ok")
            study_dir = Path(tmp) / "run" / "results" / case["study"]
            self.assertTrue((study_dir / "optimize_summary.json").exists())
            with unittest.mock.patch.object(
                scorer.Path, "with_name", lambda self, name: Path(tmp)
            ):
                (Path(tmp) / case["name"]).mkdir()
                (Path(tmp) / case["name"] / "reference.json").write_text(
                    json.dumps(scorer.reference_from_study("optimize", study_dir))
                )
                result = scorer.score(case, Path(tmp) / "run")
            self.assertTrue(result["passed"], result)
            self.assertTrue(verify_after_exit(Path(tmp) / "run")["intact"])


class ToolPrefixTests(unittest.TestCase):
    def test_prefixes_follow_symlinks_and_skip_system_dirs(self):
        from workflows.evals.sandbox import tool_prefixes

        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "nvm" / "v1"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "bin" / "fakenode").write_text("#!/bin/sh\n")
            (prefix / "bin" / "fakenode").chmod(0o755)
            npm = Path(tmp) / "npm"
            (npm / "lib" / "node_modules" / "pkg").mkdir(parents=True)
            (npm / "lib" / "node_modules" / "pkg" / "cli.js").write_text("")
            (npm / "bin").mkdir()
            (npm / "bin" / "fakecli").symlink_to(
                npm / "lib" / "node_modules" / "pkg" / "cli.js"
            )
            (npm / "bin" / "fakecli").chmod(0o755)
            with unittest.mock.patch.dict(
                os.environ, {"PATH": f"{npm / 'bin'}:{prefix / 'bin'}:/usr/bin"}
            ):
                found = tool_prefixes("fakecli", "fakenode", "true")
            self.assertIn(npm, found)
            self.assertIn(prefix, found)
            self.assertFalse(any(str(p).startswith("/usr") for p in found))


class BrokerCliTests(unittest.TestCase):
    def test_submit_command(self):
        from workflows.evals.broker import main

        with tempfile.TemporaryDirectory() as tmp:
            broker = Broker(
                Path(tmp) / "b.sock",
                Path(tmp) / "results",
                runner=lambda command, spec_path, study_dir: {
                    "result": {"cmd": command}
                },
            )
            broker.serve_in_thread()
            try:
                spec_path = Path(tmp) / "s.json"
                spec_path.write_text(json.dumps(make_spec_dict()))
                with unittest.mock.patch("sys.stdout") as out:
                    rc = main(
                        [
                            "--spec",
                            str(spec_path),
                            "--study",
                            "s1",
                            "--command",
                            "estimate",
                            "--socket",
                            str(Path(tmp) / "b.sock"),
                        ]
                    )
                self.assertEqual(rc, 0)
                printed = "".join(c.args[0] for c in out.write.call_args_list)
                self.assertIn('"status": "ok"', printed)
            finally:
                broker.shutdown()
                broker.server_close()


class RecordRunTests(unittest.TestCase):
    def test_record_creates_case_directory(self):
        from workflows.evals import controller

        with tempfile.TemporaryDirectory() as tmp:
            record = {
                "agent": {
                    "exit_code": 0,
                    "wall_s": 1.0,
                    "cpu_s": 0.5,
                    "peak_rss_mb": 10.0,
                    "usage": {
                        "num_turns": 2,
                        "total_cost_usd": 0.1,
                        "result": "done",
                        "usage": {"input_tokens": 1, "output_tokens": 2},
                    },
                },
                "broker_log": [],
                "started": 0.0,
            }
            with unittest.mock.patch.object(
                controller, "__file__", str(Path(tmp) / "controller.py")
            ):
                path = controller.record_run(
                    {"name": "new-case"}, "model-x", Path(tmp), record, {"passed": True}
                )
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, Path(tmp) / "runs" / "new-case")
            self.assertEqual(json.loads(path.read_text())["tokens"]["input_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
