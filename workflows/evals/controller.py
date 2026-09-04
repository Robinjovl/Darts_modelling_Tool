"""Evaluation controller: run one case through an ``AgentRunner`` under the sandbox with the
broker, collect parent-side accounting, verify hashes after exit, and write ``run.json``."""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from workflows.evals.broker import Broker
from workflows.evals.sandbox import SandboxSpec, available
from workflows.journal import atomic_write_json, file_sha256


@dataclass
class AgentRun:
    exit_code: int
    wall_s: float
    cpu_s: float
    peak_rss_mb: float
    stdout: str
    stderr: str
    usage: dict


class AgentRunner(Protocol):
    def run(
        self, prompt: str, sandbox: SandboxSpec, env: dict, budget: dict
    ) -> AgentRun: ...


def _timed(cmd: list, cwd: Path, timeout_s: float) -> AgentRun:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s
        )
        exit_code, out, err = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code, out, err = (
            -1,
            (exc.stdout or b"").decode()
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or ""),
            "timeout",
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    usage = {}
    try:
        usage = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
    except (ValueError, IndexError):
        usage = {}
    return AgentRun(
        exit_code=exit_code,
        wall_s=time.monotonic() - started,
        cpu_s=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
        peak_rss_mb=after.ru_maxrss / 1024.0,
        stdout=out,
        stderr=err,
        usage=usage if isinstance(usage, dict) else {},
    )


class ClaudeCliRunner:
    """Headless Claude Code with pinned model id, JSON output and project-only settings."""

    def __init__(self, model_id: str, max_turns: int = 200, timeout_s: float = 3600.0):
        self.model_id, self.max_turns, self.timeout_s = model_id, max_turns, timeout_s

    def run(
        self, prompt: str, sandbox: SandboxSpec, env: dict, budget: dict
    ) -> AgentRun:
        argv = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model_id,
            "--max-turns",
            str(min(self.max_turns, int(budget.get("max_turns", self.max_turns)))),
            "--permission-mode",
            "default",
            "--setting-sources",
            "project",
        ]
        return _timed(sandbox.command(argv, env), sandbox.run_dir, self.timeout_s)


class ShellRunner:
    """Runs an arbitrary command inside the sandbox (tests and non-Claude harnesses)."""

    def __init__(self, argv: list, timeout_s: float = 600.0):
        self.argv, self.timeout_s = list(argv), timeout_s
        self.last_stderr = ""

    def run(
        self, prompt: str, sandbox: SandboxSpec, env: dict, budget: dict
    ) -> AgentRun:
        cmd = sandbox.command(self.argv, {**env, "EVAL_PROMPT": prompt})
        result = _timed(cmd, sandbox.run_dir, self.timeout_s)
        self.last_stderr = result.stderr
        return result


def hash_tree(root: Path, suffixes: tuple = (".json", ".jsonl", ".npy", ".h5")) -> dict:
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in suffixes
    }


def run_case(
    case: dict,
    runner: AgentRunner,
    run_root: str | Path,
    model_prefix: str | None = None,
) -> dict:
    """Execute one evaluation case; returns the run record (also written as ``run.json``)."""
    if not available():
        raise RuntimeError("bwrap is required for evaluation runs")
    run_root = Path(run_root).resolve()
    run_dir = run_root / "agent"
    results_dir = run_root / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    sandbox = SandboxSpec(
        run_dir=run_dir,
        home_files={
            ".claude/settings.json": json.dumps(
                {"permissions": {"defaultMode": "default"}}
            )
        },
    )
    canary = sandbox.canary()
    if not canary["passed"]:
        raise RuntimeError(f"sandbox canary failed: {canary}")
    # AF_UNIX paths are limited to 108 bytes: use a short private dir and bind it into
    # the sandbox (the run root may be deep).
    socket_dir = Path(tempfile.mkdtemp(prefix="wfb-", dir="/tmp"))
    socket_path = socket_dir / "s"
    sandbox.extra_binds = (socket_dir,)
    broker = Broker(socket_path, results_dir)
    broker.serve_in_thread()
    try:
        env = {
            "WORKFLOWS_BROKER": str(socket_path),
            "WORKFLOWS_RUN_DIR": str(run_dir),
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": str(sandbox.repo),
        }
        agent = runner.run(case["prompt"], sandbox, env, case.get("budget", {}))
    finally:
        broker.shutdown()
        broker.server_close()
        shutil.rmtree(socket_dir, ignore_errors=True)
    results_hashes = hash_tree(results_dir) if results_dir.exists() else {}
    verification = {
        "n_results_files": len(results_hashes),
        "broker_requests": len(broker.log),
        "agent_wrote_results": any(p.is_file() for p in run_dir.rglob("*.h5")),
    }
    record = {
        "case": case.get("name"),
        "canary": canary,
        "network": "unrestricted (bwrap shares the network namespace)",
        "agent": {
            "exit_code": agent.exit_code,
            "wall_s": agent.wall_s,
            "cpu_s": agent.cpu_s,
            "peak_rss_mb": agent.peak_rss_mb,
            "usage": agent.usage,
        },
        "broker_log": broker.log,
        "results_hashes": results_hashes,
        "verification": verification,
    }
    atomic_write_json(run_root / "run.json", record)
    return record


def verify_after_exit(run_root: str | Path) -> dict:
    """Re-hash the parent-owned results and compare with the record written at run end."""
    run_root = Path(run_root)
    with open(run_root / "run.json", encoding="utf-8") as handle:
        record = json.load(handle)
    current = hash_tree(run_root / "results") if (run_root / "results").exists() else {}
    changed = sorted(
        k
        for k in set(current) | set(record["results_hashes"])
        if current.get(k) != record["results_hashes"].get(k)
    )
    return {"intact": not changed, "changed": changed}


def broker_socket_from_env() -> str | None:
    return os.environ.get("WORKFLOWS_BROKER")
