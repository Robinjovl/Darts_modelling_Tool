"""Study directory layout, atomic manifests, the append-only journal and provenance capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from workflows.spec import RunResult, StudySpec

JOURNAL_NAME = "journal.jsonl"
MANIFEST_NAME = "manifest.json"
SPEC_NAME = "study.json"
HARD_DEPENDENCIES = ("open-darts", "numpy", "scipy", "pandas", "h5py", "jsonschema")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


class StudyStore:
    """One study directory: ``study.json``, ``manifest.json``, ``journal.jsonl``, ``members/<id>/``.

    The journal is append-only; every event gets a sequence number and a wall-clock stamp. The
    library process is the only writer during evaluation (design section 10); the sequence counter
    is kept in memory and seeded once from the file on open.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.members_dir = self.root / "members"
        self.members_dir.mkdir(parents=True, exist_ok=True)
        self._next_seq = self._count_events()

    @property
    def spec_path(self) -> Path:
        return self.root / SPEC_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def journal_path(self) -> Path:
        return self.root / JOURNAL_NAME

    def member_dir(self, member_id: str) -> Path:
        path = self.members_dir / member_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_spec(self, spec: StudySpec) -> None:
        spec.dump(self.spec_path)

    def read_spec(self) -> StudySpec:
        return StudySpec.load(self.spec_path)

    def write_manifest(self, manifest: dict) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def read_manifest(self) -> dict:
        with open(self.manifest_path, encoding="utf-8") as handle:
            return json.load(handle)

    def append_event(self, event: dict) -> dict:
        """Append a stage-level event (single writer; see class docstring)."""
        record = dict(event)
        record["seq"] = self._next_seq
        record["ts"] = time.time()
        with open(self.journal_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._next_seq += 1
        return record

    def append_member_event(self, result: RunResult) -> dict:
        """Append one simulation attempt; the identity fields of ``RunResult`` are mandatory."""
        for name in ("member", "spec_hash", "input_hash", "binary_fingerprint"):
            if not getattr(result, name):
                raise ValueError(f"member event requires a non-empty {name}")
        return self.append_event({"stage": "member", **asdict(result)})

    def events(self) -> list:
        if not self.journal_path.exists():
            return []
        with open(self.journal_path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def completed_members(self) -> set:
        """Member ids whose latest event reports ``status == "ok"`` (used by ``--resume``)."""
        latest = {}
        for event in self.events():
            if "member" in event and "status" in event:
                latest[event["member"]] = event["status"]
        return {member for member, status in latest.items() if status == "ok"}

    def _count_events(self) -> int:
        if not self.journal_path.exists():
            return 0
        with open(self.journal_path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())


def git_output(repo: str | Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip()


def engine_fingerprint() -> str | None:
    """SHA-256 of the compiled open-DARTS engine module, or ``None`` when darts is not importable."""
    try:
        import darts
    except ImportError:
        return None
    darts_dir = Path(darts.__file__).parent
    libs = sorted(darts_dir.glob("engines*.so")) + sorted(
        darts_dir.glob("engines*.pyd")
    )
    if not libs:
        return None
    return ";".join(f"{lib.name}:{file_sha256(lib)}" for lib in libs)


def dependency_lock() -> dict:
    """Installed versions of the hard dependencies and of every package in ``requirements.txt``.

    Absent packages are recorded as ``None`` so the lock still identifies the environment.
    """
    from importlib import metadata

    names = list(HARD_DEPENDENCIES)
    requirements = Path(__file__).with_name("requirements.txt")
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line.split(">=")[0].split("==")[0].strip())
    lock = {"python": sys.version.split()[0]}
    for name in names:
        try:
            lock[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            lock[name] = None
    return lock


def provenance(repo: str | Path | None = None) -> dict:
    """Provenance record used by manifests and by ``StudySpec.truth_generation_hash``."""
    repo = Path(repo) if repo else Path(__file__).resolve().parents[1]
    status = git_output(repo, "status", "--porcelain")
    return {
        "code_revision": git_output(repo, "rev-parse", "HEAD"),
        "code_dirty_paths": status.count("\n") + 1 if status else 0,
        "engine_fingerprint": engine_fingerprint(),
        "dependency_lock": dependency_lock(),
        "platform": platform.platform(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "affinity_cores": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
