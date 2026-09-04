"""``bwrap`` mount manifest for the evaluated agent (design section 10).

The repository and the conda environment are mounted read-only; the hidden truth and scorer
paths are masked with empty tmpfs mounts after the repository bind; one per-run directory is
writable; a scratch HOME carries a read-only bind of the CLI credentials. ``bwrap`` cannot filter
egress, so outbound network stays shared and is recorded as unrestricted.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MASKED_RELATIVE = (
    "workflows/evals/truth",
    "workflows/evals/scorer.py",
    "workflows/evals/cases",
)


@dataclass
class SandboxSpec:
    run_dir: Path
    repo: Path = REPO
    env_prefix: Path = Path(
        os.environ.get("CONDA_PREFIX", "/oahu/data/avnovikov/mambaforge/envs/skills")
    )
    masked: tuple = MASKED_RELATIVE
    credentials: Path | None = Path.home() / ".claude" / ".credentials.json"
    extra_ro: tuple = ("/usr", "/lib", "/lib64", "/bin", "/etc", "/opt")
    home_files: dict = field(
        default_factory=dict
    )  # relative path inside scratch HOME -> content
    extra_binds: tuple = ()  # read-write (path, path) binds, e.g. the broker socket dir

    def command(self, argv: list, env: dict | None = None) -> list:
        """Full ``bwrap`` argument list wrapping ``argv``."""
        run_dir = Path(self.run_dir).resolve()
        home = run_dir / "home"
        (home / ".claude").mkdir(parents=True, exist_ok=True)
        for rel, content in self.home_files.items():
            path = home / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        cmd = [
            "bwrap",
            "--die-with-parent",
            "--unshare-pid",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for path in self.extra_ro:
            if Path(path).exists():
                cmd += ["--ro-bind", path, path]
        cmd += ["--ro-bind", str(self.repo), str(self.repo)]
        if not str(self.env_prefix).startswith(str(self.repo)):
            cmd += ["--ro-bind", str(self.env_prefix), str(self.env_prefix)]
        for rel in self.masked:
            target = self.repo / rel
            if target.exists():
                cmd += (
                    ["--tmpfs", str(target)]
                    if target.is_dir()
                    else ["--ro-bind", "/dev/null", str(target)]
                )
        cmd += ["--bind", str(run_dir), str(run_dir)]
        for path in self.extra_binds:
            cmd += ["--bind", str(path), str(path)]
        if self.credentials and Path(self.credentials).exists():
            cmd += [
                "--ro-bind",
                str(self.credentials),
                str(home / ".claude" / self.credentials.name),
            ]
        cmd += ["--setenv", "HOME", str(home), "--chdir", str(run_dir)]
        for key, value in (env or {}).items():
            cmd += ["--setenv", key, str(value)]
        return cmd + argv

    def canary(self) -> dict:
        """Prove from inside the sandbox that masked paths are unreadable and the run dir writable."""
        checks = {}
        for rel in self.masked:
            target = self.repo / rel
            probe = (
                f"ls -A {target} 2>/dev/null | wc -l"
                if target.is_dir()
                else f"cat {target} 2>/dev/null | wc -c"
            )
            out = subprocess.run(
                self.command(["/bin/sh", "-c", probe]), capture_output=True, text=True
            )
            checks[rel] = out.returncode == 0 and out.stdout.strip() == "0"
        out = subprocess.run(
            self.command(["/bin/sh", "-c", "echo ok > canary.txt && cat canary.txt"]),
            capture_output=True,
            text=True,
        )
        checks["run_dir_writable"] = out.stdout.strip() == "ok"
        out = subprocess.run(
            self.command(["/bin/sh", "-c", f"touch {self.repo}/.canary 2>&1 | wc -c"]),
            capture_output=True,
            text=True,
        )
        checks["repo_read_only"] = out.stdout.strip() != "0"
        checks["passed"] = all(checks.values())
        return checks


def available() -> bool:
    return shutil.which("bwrap") is not None
