"""Isolated per-member execution with accounting (design sections 6.2 and 10, principle P4).

Every simulation runs in its own ``spawn`` process with a hard walltime. A task that hangs is
killed and retried; retries get a distinct attempt identity, and a retry with a different OpenMP
thread count is reported as non-equivalent (``equivalent=False``) instead of silently replacing
the originally specified run. Permanent failures are recorded, not raised, so that a study never
loses the members that did complete. The parent records every attempt in the study journal.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from workflows.cost import usable_workers
from workflows.journal import StudyStore, atomic_write_json, file_sha256
from workflows.spec import ObservationSpec, RunResult

THREAD_BUMPS = (0, 0, 2, 1)  # per attempt: same, same, then different thread counts
ENGINE_STAT_FIELDS = (
    "n_timesteps_total",
    "n_timesteps_wasted",
    "n_newton_total",
    "n_newton_wasted",
    "n_linear_total",
    "n_linear_wasted",
)


class MemberFailure(RuntimeError):
    """Raised by adapters to classify a member failure (``failure_class`` from design 6.2)."""

    def __init__(self, failure_class: str, message: str = ""):
        super().__init__(message or failure_class)
        self.failure_class = failure_class


@contextmanager
def contained_output(log_path: str | Path):
    """Redirect process-level stdout/stderr (including C++ and gmsh prints) into a log file."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    with open(log_path, "ab") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


@dataclass
class MemberTask:
    """Everything a worker process needs to run one member (picklable)."""

    member: str
    study_root: str
    adapter: str
    model_dir: str
    snapshot: dict
    realization: dict
    report_times: list
    observation: dict | None
    spec_hash: str
    input_hash: str
    binary_fingerprint: str
    stage_mode: str = "symlink"


def realization_hash(realization: dict) -> str:
    """sha256 of the canonical JSON of a realization (arrays included) for per-member provenance."""
    from workflows.members import jsonable
    from workflows.spec import canonical_json, sha256_text

    return sha256_text(canonical_json(jsonable(realization)))


def run_member(task: MemberTask) -> dict:
    """Worker entry point: stage, build, apply, run, observe; write ``result.json``."""
    from workflows.adapter import InputSnapshot, load_adapter

    workdir = Path(task.study_root) / "members" / task.member
    workdir.mkdir(parents=True, exist_ok=True)
    with contained_output(workdir / "member.log"):
        adapter = load_adapter(task.adapter, task.model_dir)
        paths = adapter.stage(InputSnapshot(**task.snapshot), workdir, task.stage_mode)
        model = adapter.build(paths, task.realization)
        adapter.apply(model, task.realization)
        adapter.run(model, task.report_times)
        observation = ObservationSpec(**task.observation) if task.observation else None
        observed = adapter.observe(model, observation)
        stats = getattr(getattr(model, "nonlinear_solver", None), "stats", None)
        engine_stats = {
            k: int(getattr(stats, k)) for k in ENGINE_STAT_FIELDS if hasattr(stats, k)
        }
    output_dir = workdir / "output"
    output_hashes = (
        {p.name: file_sha256(p) for p in sorted(output_dir.glob("*.h5"))}
        if output_dir.is_dir()
        else {}
    )
    result = {
        "member": task.member,
        "realization": task.realization,
        "observation": observed,
        "engine_stats": engine_stats,
        "output_hashes": output_hashes,
        "realization_hash": realization_hash(task.realization),
        "spec_hash": task.spec_hash,
        "input_hash": task.input_hash,
        "binary_fingerprint": task.binary_fingerprint,
    }
    atomic_write_json(workdir / "result.json", result)
    return result


def _worker_init(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def _isolated_task(fn, item, queue, threads: int) -> None:
    """Child entry point: set the thread environment before the task imports the simulator."""
    _worker_init(threads)
    started = time.monotonic()
    try:
        payload = fn(item)
        status, error = "ok", None
    except MemberFailure as exc:
        payload, status, error = None, exc.failure_class, repr(exc)
    except BaseException as exc:  # noqa: BLE001 - report to the parent, which decides
        payload, status, error = None, "exception", repr(exc)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    queue.put(
        {
            "status": status,
            "payload": payload,
            "error": error,
            "cpu_s": usage.ru_utime + usage.ru_stime,
            "peak_rss_mb": usage.ru_maxrss / 1024.0,
            "child_wall_s": time.monotonic() - started,
        }
    )


def _hang_diagnostics(pid: int) -> list:
    """Best-effort /proc snapshot of a hung process's threads before it is killed."""
    lines = []
    try:
        task_dir = f"/proc/{pid}/task"
        for tid in sorted(os.listdir(task_dir)):
            with open(f"{task_dir}/{tid}/stat", encoding="utf-8") as handle:
                parts = handle.read().split()
            try:
                with open(f"{task_dir}/{tid}/wchan", encoding="utf-8") as handle:
                    wchan = handle.read().strip() or "-"
            except OSError:
                wchan = "?"
            lines.append(
                f"tid {tid}: state {parts[2]} cpu {int(parts[13]) + int(parts[14])} wchan {wchan}"
            )
    except OSError as exc:
        lines.append(f"diagnostics unavailable: {exc}")
    return lines


@dataclass
class Attempt:
    """One attempt of one item, as reported to the caller and to the journal."""

    index: int
    attempt: int
    status: str
    threads: int
    equivalent: bool
    wall_s: float
    cpu_s: float | None = None
    peak_rss_mb: float | None = None
    error: str | None = None
    payload: object = None
    diagnostics: list = field(default_factory=list)


class SerialExecutor:
    """In-process execution for tests and debugging (no isolation, no walltime)."""

    def __init__(self, threads_per_member: int = 1):
        self.threads_per_member = threads_per_member

    def map(self, fn, items, on_attempt=None) -> list:
        results = []
        for index, item in enumerate(items):
            started = time.monotonic()
            try:
                payload, status, error = fn(item), "ok", None
            except MemberFailure as exc:
                payload, status, error = None, exc.failure_class, repr(exc)
            except Exception as exc:  # noqa: BLE001
                payload, status, error = None, "exception", repr(exc)
            attempt = Attempt(
                index,
                1,
                status,
                self.threads_per_member,
                True,
                time.monotonic() - started,
                error=error,
                payload=payload,
            )
            if on_attempt:
                on_attempt(attempt)
            results.append(attempt)
        return results


class IsolatedExecutor:
    """One fresh ``spawn`` process per item with a hard walltime, retries and accounting.

    ``map`` returns the final :class:`Attempt` per item (status ``"ok"`` or a failure class) and
    calls ``on_attempt`` for every attempt, including the ones that were retried.
    """

    def __init__(
        self,
        n_workers: int | None = None,
        threads_per_member: int = 1,
        timeout_s: float = 1800.0,
        retries: int = 2,
        thread_bumps: tuple = THREAD_BUMPS,
        memory_per_member_gb: float | None = None,
        poll_s: float = 0.2,
    ):
        self.n_workers = n_workers or usable_workers(
            threads_per_member, memory_per_member_gb
        )
        self.threads_per_member = int(threads_per_member)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.thread_bumps = tuple(thread_bumps)
        self.poll_s = poll_s

    def _threads_for(self, attempt: int) -> int:
        bump = self.thread_bumps[min(attempt - 1, len(self.thread_bumps) - 1)]
        return max(1, self.threads_per_member + bump)

    def map(self, fn, items, on_attempt=None) -> list:
        import multiprocessing as mp

        items = list(items)
        ctx = mp.get_context("spawn")
        final: list = [None] * len(items)
        attempts = [0] * len(items)
        pending = list(range(len(items)))
        running: dict = {}

        def launch(index: int) -> None:
            attempts[index] += 1
            threads = self._threads_for(attempts[index])
            queue = ctx.Queue(1)
            process = ctx.Process(
                target=_isolated_task, args=(fn, items[index], queue, threads)
            )
            process.start()
            running[index] = (process, queue, threads, time.monotonic())

        def finish(index: int, status: str, wall: float, threads: int, **extra) -> None:
            record = Attempt(
                index=index,
                attempt=attempts[index],
                status=status,
                threads=threads,
                equivalent=(threads == self.threads_per_member),
                wall_s=wall,
                **extra,
            )
            if on_attempt:
                on_attempt(record)
            if status == "ok" or attempts[index] > self.retries:
                final[index] = record
            else:
                launch(index)

        while pending or running:
            while pending and len(running) < self.n_workers:
                launch(pending.pop(0))
            time.sleep(self.poll_s)
            for index in list(running):
                process, queue, threads, started = running[index]
                wall = time.monotonic() - started
                if not queue.empty():
                    report = queue.get()
                    process.join(5)
                    del running[index]
                    finish(
                        index,
                        report["status"],
                        wall,
                        threads,
                        cpu_s=report["cpu_s"],
                        peak_rss_mb=report["peak_rss_mb"],
                        error=report["error"],
                        payload=report["payload"],
                    )
                elif not process.is_alive():
                    del running[index]
                    finish(
                        index,
                        "native_crash",
                        wall,
                        threads,
                        error=f"exit code {process.exitcode}",
                    )
                elif wall > self.timeout_s:
                    diagnostics = _hang_diagnostics(process.pid)
                    process.terminate()
                    process.join(10)
                    if process.is_alive():
                        process.kill()
                        process.join(10)
                    del running[index]
                    finish(
                        index,
                        "walltime",
                        wall,
                        threads,
                        error=f"killed after {wall:.0f}s",
                        diagnostics=diagnostics,
                    )
        return final


def run_members(store: StudyStore, tasks: list, executor) -> list:
    """Run member tasks through ``executor`` and journal every attempt as a ``RunResult``."""

    def journal(attempt: Attempt) -> None:
        task = tasks[attempt.index]
        payload = attempt.payload if isinstance(attempt.payload, dict) else {}
        store.append_member_event(
            RunResult(
                member=task.member,
                attempt=attempt.attempt,
                status=attempt.status,
                spec_hash=task.spec_hash,
                input_hash=task.input_hash,
                binary_fingerprint=task.binary_fingerprint,
                wall_s=attempt.wall_s,
                cpu_s=attempt.cpu_s,
                peak_rss_mb=attempt.peak_rss_mb,
                threads=attempt.threads,
                equivalent=attempt.equivalent,
                engine_stats=payload.get("engine_stats", {}),
                output_hashes=payload.get("output_hashes", {}),
                realization_hash=payload.get("realization_hash"),
            )
        )

    return executor.map(run_member, tasks, on_attempt=journal)


def task_dict(task: MemberTask) -> dict:
    return asdict(task)
