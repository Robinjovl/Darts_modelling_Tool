import os
import tempfile
import time
import unittest
from pathlib import Path

from workflows.executor import Attempt, IsolatedExecutor, MemberFailure, SerialExecutor
from workflows.journal import StudyStore


def ok_task(item):
    return {"value": item * 2, "threads": os.environ.get("OMP_NUM_THREADS")}


def flaky_task(item):
    # fails on the first two attempts (marker file counts attempts), succeeds afterwards
    marker = Path(item)
    count = int(marker.read_text()) + 1 if marker.exists() else 1
    marker.write_text(str(count))
    if count < 3:
        raise MemberFailure("timestep_failure", f"attempt {count}")
    return {"attempts": count}


def hang_task(item):
    while True:
        time.sleep(0.1)


def crash_task(item):
    os._exit(3)


class SerialExecutorTests(unittest.TestCase):
    def test_serial_reports_status_and_payload(self):
        seen = []
        results = SerialExecutor().map(ok_task, [1, 2], on_attempt=seen.append)
        self.assertEqual([r.status for r in results], ["ok", "ok"])
        self.assertEqual(results[1].payload["value"], 4)
        self.assertEqual(len(seen), 2)
        failed = SerialExecutor().map(flaky_task, [str(Path(tempfile.mkdtemp()) / "m")])
        self.assertEqual(failed[0].status, "timestep_failure")


class IsolatedExecutorTests(unittest.TestCase):
    def test_ok_sets_thread_environment_and_accounting(self):
        executor = IsolatedExecutor(
            n_workers=2, threads_per_member=1, timeout_s=30, retries=0
        )
        results = executor.map(ok_task, [1, 2, 3])
        self.assertEqual([r.status for r in results], ["ok"] * 3)
        self.assertEqual([r.payload["value"] for r in results], [2, 4, 6])
        self.assertEqual(results[0].payload["threads"], "1")
        self.assertTrue(
            all(
                r.cpu_s is not None and r.peak_rss_mb > 0 and r.equivalent
                for r in results
            )
        )

    def test_retries_get_distinct_attempts_and_thread_bumps(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = IsolatedExecutor(
                n_workers=1, timeout_s=30, retries=2, thread_bumps=(0, 0, 2)
            )
            seen = []
            results = executor.map(
                flaky_task, [str(Path(tmp) / "marker")], on_attempt=seen.append
            )
            self.assertEqual([a.attempt for a in seen], [1, 2, 3])
            self.assertEqual(
                [a.status for a in seen], ["timestep_failure", "timestep_failure", "ok"]
            )
            self.assertEqual([a.threads for a in seen], [1, 1, 3])
            self.assertEqual([a.equivalent for a in seen], [True, True, False])
            self.assertEqual(results[0].status, "ok")
            self.assertFalse(results[0].equivalent)

    def test_walltime_kill_and_crash_are_recorded_not_raised(self):
        executor = IsolatedExecutor(n_workers=2, timeout_s=1.0, retries=1, poll_s=0.1)
        started = time.monotonic()
        results = executor.map(hang_task, [0])
        self.assertLess(time.monotonic() - started, 20)
        self.assertEqual(results[0].status, "walltime")
        self.assertEqual(results[0].attempt, 2)
        self.assertTrue(results[0].error.startswith("killed"))
        crashed = IsolatedExecutor(n_workers=1, timeout_s=10, retries=0).map(
            crash_task, [0]
        )
        self.assertEqual(crashed[0].status, "native_crash")
        self.assertIn("3", crashed[0].error)

    def test_attempt_record_is_journal_ready(self):
        record = Attempt(
            index=0, attempt=1, status="ok", threads=1, equivalent=True, wall_s=0.1
        )
        self.assertIsNone(record.cpu_s)
        with tempfile.TemporaryDirectory() as tmp:
            StudyStore(tmp)


class RealizationHashTests(unittest.TestCase):
    def test_hash_changes_with_realization(self):
        import numpy as np

        from workflows.executor import realization_hash

        a = realization_hash({"permx": np.array([1.0, 2.0]), "tm": 1.5})
        b = realization_hash({"permx": np.array([1.0, 2.0]), "tm": 1.5})
        c = realization_hash({"permx": np.array([1.0, 2.001]), "tm": 1.5})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class ResumeTests(unittest.TestCase):
    def test_completed_members_are_reused_when_spec_matches(self):
        import json as _json
        import tempfile
        import unittest.mock as mock
        from pathlib import Path

        from workflows.executor import MemberTask, SerialExecutor, run_members
        from workflows.journal import StudyStore

        def fake_run_member(task):
            payload = {
                "member": task.member,
                "observation": {"values": {}},
                "engine_stats": {},
                "output_hashes": {},
            }
            path = Path(task.study_root) / "members" / task.member / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps(payload))
            return payload

        def task(name, spec_hash="h1"):
            return MemberTask(
                member=name,
                study_root=str(root),
                adapter="a",
                model_dir=".",
                snapshot={"root": ".", "files": {}},
                realization={},
                report_times=[1.0],
                observation={},
                spec_hash=spec_hash,
                input_hash="i",
                binary_fingerprint="b",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "study"
            store = StudyStore(root)
            with mock.patch("workflows.executor.run_member", fake_run_member):
                first = run_members(store, [task("m0"), task("m1")], SerialExecutor())
                self.assertEqual([r.status for r in first], ["ok", "ok"])
                n_events = len(
                    [e for e in store.events() if e.get("stage") == "member"]
                )
                second = run_members(
                    store, [task("m0"), task("m1"), task("m2")], SerialExecutor()
                )
                self.assertEqual([r.attempt for r in second], [0, 0, 1])
                self.assertEqual(second[0].payload["member"], "m0")
                n_after = len([e for e in store.events() if e.get("stage") == "member"])
                self.assertEqual(n_after - n_events, 1)  # only m2 ran
                changed = run_members(
                    store, [task("m0", spec_hash="h2")], SerialExecutor()
                )
                self.assertEqual(changed[0].attempt, 1)  # different spec: rerun
                fresh = run_members(store, [task("m0")], SerialExecutor(), resume=False)
                self.assertEqual(fresh[0].attempt, 1)


if __name__ == "__main__":
    unittest.main()
