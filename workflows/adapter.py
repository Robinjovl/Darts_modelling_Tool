"""Adapter protocol and immutable input staging (design sections 6.1 and 6.2).

A ``ModelAdapter`` is the only code that touches a ``DartsModel``. Its lifecycle per study is:
``snapshot`` (copy the model's immutable inputs once into the study directory and hash them),
``stage`` (place the snapshot into each worker directory by read-only symlink or copy, returning
absolute paths), ``build`` (construct the model in the worker process from those paths),
``apply`` (cheap per-realization knobs), ``run`` (report-step segments) and ``observe`` (the
observation vector). The rebuild scope of each parameter family is declared by the adapter.
"""

from __future__ import annotations

import abc
import importlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from workflows.journal import atomic_write_json, file_sha256
from workflows.spec import ObservationSpec, canonical_json, sha256_text

PROTOCOL_VERSION = 1
SNAPSHOT_NAME = "snapshot.json"
STAGE_MODES = ("symlink", "copy", "reflink")


class ProtocolMismatch(RuntimeError):
    """Raised when an adapter was written against a different protocol version."""


@dataclass
class InputSnapshot:
    """Immutable inputs of one study: absolute root plus relative path to SHA-256."""

    root: str
    files: dict

    @property
    def hash(self) -> str:
        return sha256_text(canonical_json(self.files))

    def to_dict(self) -> dict:
        return {"root": self.root, "files": dict(self.files), "hash": self.hash}

    @classmethod
    def load(cls, snapshot_dir: str | Path) -> InputSnapshot:
        with open(Path(snapshot_dir) / SNAPSHOT_NAME, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(root=data["root"], files=dict(data["files"]))


def _matching_files(model_dir: Path, patterns) -> list:
    found = []
    for pattern in patterns:
        found.extend(p for p in sorted(model_dir.glob(pattern)) if p.is_file())
    return found


def snapshot_inputs(
    model_dir: str | Path, patterns, snapshot_dir: str | Path
) -> InputSnapshot:
    """Copy the files matching ``patterns`` (relative to ``model_dir``) into ``snapshot_dir``.

    Files are made read-only and hashed; an existing snapshot with identical hashes is reused.
    """
    model_dir = Path(model_dir).resolve()
    snapshot_dir = Path(snapshot_dir).resolve()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in _matching_files(model_dir, patterns):
        rel = source.relative_to(model_dir).as_posix()
        target = snapshot_dir / rel
        digest = file_sha256(source)
        if not (target.exists() and file_sha256(target) == digest):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.chmod(0o644)
            shutil.copy2(source, target)
            target.chmod(0o444)
        files[rel] = digest
    snapshot = InputSnapshot(root=str(snapshot_dir), files=files)
    atomic_write_json(snapshot_dir / SNAPSHOT_NAME, snapshot.to_dict())
    return snapshot


def add_generated_input(snapshot: InputSnapshot, path: str | Path) -> InputSnapshot:
    """Register a file generated inside the snapshot root (e.g. a mesh) as an immutable input."""
    path = Path(path).resolve()
    root = Path(snapshot.root).resolve()
    rel = path.relative_to(root).as_posix()
    path.chmod(0o444)
    files = dict(snapshot.files)
    files[rel] = file_sha256(path)
    updated = InputSnapshot(root=snapshot.root, files=files)
    atomic_write_json(root / SNAPSHOT_NAME, updated.to_dict())
    return updated


def stage_inputs(
    snapshot: InputSnapshot, workdir: str | Path, mode: str = "symlink"
) -> Path:
    """Place the snapshot into ``workdir/inputs`` and return that absolute directory."""
    if mode not in STAGE_MODES:
        raise ValueError(f"stage mode must be one of {STAGE_MODES}, got {mode!r}")
    root = Path(snapshot.root)
    staged = Path(workdir).resolve() / "inputs"
    for rel in snapshot.files:
        source = root / rel
        target = staged / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if mode == "symlink":
            os.symlink(source, target)
        elif mode == "reflink" and shutil.which("cp"):
            os.system(f"cp --reflink=auto '{source}' '{target}'")
        else:
            shutil.copy2(source, target)
    return staged


class ModelAdapter(abc.ABC):
    """Model-specific mapping between a study and a ``DartsModel`` (one subclass per bed)."""

    PROTOCOL_VERSION = PROTOCOL_VERSION
    input_patterns: tuple = ()
    rebuild_scope: dict = {}  # parameter family -> "cheap" | "physics" | "geometry"
    member_seconds: tuple = (10.0, 30.0)

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model directory not found: {self.model_dir}")

    def snapshot(self, snapshot_dir: str | Path) -> InputSnapshot:
        snapshot = snapshot_inputs(self.model_dir, self.input_patterns, snapshot_dir)
        return self.prepare_snapshot(snapshot)

    def prepare_snapshot(self, snapshot: InputSnapshot) -> InputSnapshot:
        """Hook for inputs generated once per study (e.g. a mesh); default: nothing."""
        return snapshot

    def stage(
        self, snapshot: InputSnapshot, workdir: str | Path, mode: str = "symlink"
    ) -> dict:
        return {
            "inputs": stage_inputs(snapshot, workdir, mode),
            "workdir": Path(workdir).resolve(),
        }

    @abc.abstractmethod
    def build(self, paths: dict, realization: dict):
        """Construct and initialize the model in the worker process."""

    @abc.abstractmethod
    def apply(self, model, realization: dict) -> None:
        """Apply cheap per-realization knobs (mesh-level arrays, controls) and reset the engine."""

    @abc.abstractmethod
    def run(self, model, report_times) -> None:
        """Run to each report time in turn so that every report time is a timestep boundary."""

    @abc.abstractmethod
    def observe(self, model, observation: ObservationSpec | None) -> dict:
        """Return ``{"time": [...], "values": {"<well>:<quantity>": [...]}}`` at the report times."""

    def cost_hint(self) -> tuple:
        return self.member_seconds


def load_adapter(dotted: str, model_dir: str | Path) -> ModelAdapter:
    """Import ``<dotted>.Adapter`` and instantiate it after checking the protocol version."""
    module = importlib.import_module(dotted)
    cls = module.Adapter
    if getattr(cls, "PROTOCOL_VERSION", None) != PROTOCOL_VERSION:
        raise ProtocolMismatch(
            f"{dotted}.Adapter declares protocol {getattr(cls, 'PROTOCOL_VERSION', None)}, "
            f"this library implements {PROTOCOL_VERSION}"
        )
    return cls(model_dir)
