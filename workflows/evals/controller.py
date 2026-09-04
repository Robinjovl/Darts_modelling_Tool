"""Evaluation controller: run one case through an ``AgentRunner`` under the sandbox with the
broker, collect parent-side accounting, verify hashes after exit, and write ``run.json``."""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from workflows.evals.broker import Broker
from workflows.evals.sandbox import SandboxSpec, available, tool_prefixes
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


TIME_BIN = "/usr/bin/time"


def _timed(
    cmd: list, cwd: Path, timeout_s: float, rusage_file: Path | None = None
) -> AgentRun:
    """Run ``cmd`` and account for it. Under ``bwrap --unshare-pid`` the kernel's child rusage
    covers the sandbox root only, so when ``rusage_file`` is given the command is expected to
    have been wrapped with ``/usr/bin/time`` writing ``"%e %U %S %M"`` there (see ``time_wrap``)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s
        )
        exit_code, out, err = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        out = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        exit_code, err = -1, "timeout"
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.monotonic() - started
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    rss_mb = after.ru_maxrss / 1024.0
    if rusage_file is not None and rusage_file.exists():
        try:
            fields = rusage_file.read_text().split()[-4:]
            _, user, system, max_kb = (float(x) for x in fields)
            cpu, rss_mb = user + system, max_kb / 1024.0
        except (ValueError, IndexError):
            pass
    usage = {}
    try:
        usage = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
    except (ValueError, IndexError):
        usage = {}
    return AgentRun(
        exit_code=exit_code,
        wall_s=wall,
        cpu_s=cpu,
        peak_rss_mb=rss_mb,
        stdout=out,
        stderr=err,
        usage=usage if isinstance(usage, dict) else {},
    )


def time_wrap(argv: list, rusage_file: Path) -> list:
    """Prefix ``argv`` with ``/usr/bin/time`` when available (it is inside the sandbox via /usr)."""
    if not Path(TIME_BIN).exists():
        return list(argv)
    return [TIME_BIN, "-f", "%e %U %S %M", "-o", str(rusage_file)] + list(argv)


class ClaudeCliRunner:
    """Headless Claude Code with pinned model id, JSON output and project-only settings.

    The working directory is the repository (read-only in the sandbox) so project skills are
    discovered; the agent writes under ``WORKFLOWS_RUN_DIR``. Tools are allow-listed explicitly
    because headless runs cannot prompt; the sandbox, not the permission system, is the boundary.
    """

    ALLOWED_TOOLS = "Bash,Read,Write,Edit,Glob,Grep,Skill"

    def __init__(self, model_id: str, max_turns: int = 200, timeout_s: float = 3600.0):
        self.model_id, self.max_turns, self.timeout_s = model_id, max_turns, timeout_s

    def run(
        self, prompt: str, sandbox: SandboxSpec, env: dict, budget: dict
    ) -> AgentRun:
        turns = min(self.max_turns, int(budget.get("max_turns", self.max_turns)))
        argv = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model_id,
            "--max-turns",
            str(turns),
            "--permission-mode",
            "default",
            "--allowedTools",
            self.ALLOWED_TOOLS,
            "--setting-sources",
            "project",
        ]
        if budget.get("max_usd"):
            argv += ["--max-budget-usd", str(float(budget["max_usd"]))]
        sandbox.extra_ro_binds = tuple(sandbox.extra_ro_binds) + tool_prefixes(
            "claude", "node"
        )
        sandbox.chdir = sandbox.repo
        env = {**env, "PATH": os.environ.get("PATH", "")}
        rusage_file = Path(sandbox.run_dir) / "rusage.txt"
        return _timed(
            sandbox.command(time_wrap(argv, rusage_file), env),
            sandbox.run_dir,
            self.timeout_s,
            rusage_file,
        )


class ShellRunner:
    """Runs an arbitrary command inside the sandbox (tests and non-Claude harnesses)."""

    def __init__(self, argv: list, timeout_s: float = 600.0):
        self.argv, self.timeout_s = list(argv), timeout_s
        self.last_stderr = ""

    def run(
        self, prompt: str, sandbox: SandboxSpec, env: dict, budget: dict
    ) -> AgentRun:
        rusage_file = Path(sandbox.run_dir) / "rusage.txt"
        argv = time_wrap(self.argv, rusage_file)
        cmd = sandbox.command(argv, {**env, "EVAL_PROMPT": prompt})
        result = _timed(cmd, sandbox.run_dir, self.timeout_s, rusage_file)
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
            "WORKFLOWS_PYTHON": sys.executable,
        }
        prompt = case["prompt"].format(
            repo=sandbox.repo, python=sys.executable, run_dir=run_dir
        )
        agent = runner.run(prompt, sandbox, env, case.get("budget", {}))
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


def record_run(
    case: dict, model_id: str, run_root: Path, record: dict, score: dict | None
) -> Path:
    """Compact, committable run record under ``workflows/evals/runs/<case>/``."""
    usage = record["agent"].get("usage", {})
    tokens = usage.get("usage", {}) if isinstance(usage.get("usage"), dict) else {}
    out = {
        "case": case["name"],
        "model": model_id,
        "started_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record["started"])
        ),
        "git_head": git_head(),
        "exit_code": record["agent"]["exit_code"],
        "wall_s": round(record["agent"]["wall_s"], 1),
        "cpu_s": round(record["agent"]["cpu_s"], 1),
        "peak_rss_mb": round(record["agent"]["peak_rss_mb"], 1),
        "num_turns": usage.get("num_turns"),
        "total_cost_usd": usage.get("total_cost_usd"),
        "tokens": {
            k: tokens.get(k)
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        },
        "broker_requests": record["broker_log"],
        "result_text": (usage.get("result") or "")[:2000],
        "score": score,
        "run_root": str(run_root),
    }
    path = (
        Path(__file__).with_name("runs")
        / case["name"]
        / f"{out['started_utc'].replace(':', '')}-{model_id}.json"
    )
    atomic_write_json(path, out)
    return path


def git_head() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
            or None
        )
    except OSError:
        return None


def main(argv: list | None = None) -> int:
    """``python -m workflows.evals.controller --case cases/x.json --model <id> --run-root DIR``"""
    import argparse

    from workflows.evals import scorer

    parser = argparse.ArgumentParser(prog="python -m workflows.evals.controller")
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--model", required=True, help="exact model id, e.g. claude-sonnet-4-6"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args(argv)
    with open(args.case, encoding="utf-8") as handle:
        case = json.load(handle)
    runner = ClaudeCliRunner(args.model, timeout_s=args.timeout_s)
    started = time.time()
    record = run_case(case, runner, args.run_root)
    record["started"] = started
    try:
        score = scorer.score(case, args.run_root)
    except Exception as exc:  # noqa: BLE001 - a missing study is a legitimate (failed) outcome
        score = {"passed": False, "error": repr(exc)}
    verification = verify_after_exit(args.run_root)
    summary = {
        "exit_code": record["agent"]["exit_code"],
        "score": score,
        "verification": verification,
        "cost_usd": record["agent"].get("usage", {}).get("total_cost_usd"),
        "num_turns": record["agent"].get("usage", {}).get("num_turns"),
        "wall_s": round(record["agent"]["wall_s"], 1),
    }
    if not args.no_record:
        summary["record"] = str(
            record_run(case, args.model, Path(args.run_root), record, score)
        )
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0 if score.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
