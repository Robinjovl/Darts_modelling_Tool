"""Study specifications, identities and the seed ledger (design section 6.2).

A ``StudySpec`` is a plain JSON document validated against ``STUDY_SCHEMA``. Three identities are
derived from it (design section 3, P3): ``workflow_run_hash`` over the whole document,
``truth_generation_hash`` over the truth-relevant subset plus provenance (immutable inputs, engine
fingerprint, code revision, dependency lock), and the scorer's ``SCORE_VERSION``. ``SeedLedger``
derives named, independent random streams from one root seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SPEC_VERSION = 1
SCORE_VERSION = "0.1.0"
WORKFLOWS = ("ensemble", "hm-esmda", "hm-adjoint", "optimize")
PLATFORMS = ("cpu", "gpu")
SEED_STREAMS = (
    "design",
    "geology",
    "observation_noise",
    "assimilation",
    "optimizer",
    "surrogate",
    "agent",
)
RNG_IMPLEMENTATION = (
    "numpy.random.Generator(PCG64) via SeedSequence(root, spawn_key=(stream, index))"
)
TRUTH_PROVENANCE_KEYS = (
    "input_hashes",
    "engine_fingerprint",
    "code_revision",
    "dependency_lock",
)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace. Int and float literals differ (1 vs 1.0)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SeedLedger:
    """Hierarchical, idempotent seed derivation.

    ``generator(stream, index)`` always returns the same stream for the same root, stream name
    and integer index, so common-random-number comparisons can share exactly the intended
    substreams.
    """

    def __init__(self, root: int):
        if isinstance(root, bool) or not isinstance(root, int) or root < 0:
            raise ValueError("seed root must be a non-negative integer")
        self.root = root

    def seed_sequence(self, stream: str, index: int = 0) -> np.random.SeedSequence:
        if stream not in SEED_STREAMS:
            raise KeyError(f"unknown seed stream {stream!r}; known: {SEED_STREAMS}")
        if isinstance(index, bool) or index != int(index) or index < 0:
            raise ValueError(
                f"seed index must be a non-negative integer, got {index!r}"
            )
        return np.random.SeedSequence(
            self.root, spawn_key=(SEED_STREAMS.index(stream), int(index))
        )

    def generator(self, stream: str, index: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence(stream, index))

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "streams": list(SEED_STREAMS),
            "rng": RNG_IMPLEMENTATION,
        }


@dataclass
class ModelRef:
    """Where the model lives and which adapter drives it."""

    model_dir: str
    adapter: str
    module: str = "model"
    cls: str = "Model"
    kwargs: dict = field(default_factory=dict)


@dataclass
class ParameterSpec:
    """One parameter family; ``args`` are interpreted by ``workflows.members``."""

    name: str
    family: str
    args: dict = field(default_factory=dict)


@dataclass
class ObservationSpec:
    """Observation contract: which well quantities, at which report times, with which noise."""

    wells: list
    quantities: list
    report_times: list
    held_out_fraction: float = 0.2
    sigma_abs: float | None = None
    sigma_rel: float | None = None
    sigma_floor: float = 1e-12


@dataclass
class ComputeSpec:
    platform: str = "cpu"
    walltime_s: float = 1800.0
    max_workers: int | None = None
    threads_per_member: int = 1
    retries: int = 2
    memory_per_member_gb: float | None = None


@dataclass
class RunResult:
    """Result of one simulation attempt as recorded by the executor (populated in a later step).

    ``status`` is ``"ok"`` or a failure class; ``attempt`` distinguishes retries, and a retry with a
    different thread count is a distinct, non-equivalent attempt (``equivalent=False``).
    """

    member: str
    attempt: int
    status: str
    spec_hash: str
    input_hash: str
    binary_fingerprint: str
    wall_s: float | None = None
    cpu_s: float | None = None
    peak_rss_mb: float | None = None
    threads: int | None = None
    equivalent: bool = True
    engine_stats: dict = field(default_factory=dict)
    output_hashes: dict = field(default_factory=dict)
    realization_hash: str | None = (
        None  # sha256 of the canonical realization (spec-level inputs)
    )


@dataclass
class StudySpec:
    """The study document.

    ``design`` holds working settings (working ensemble size, optimizer and assimilation settings);
    ``reference`` holds truth-relevant settings (reference design and size).
    """

    name: str
    workflow: str
    model: ModelRef
    parameters: list
    seed_root: int
    observations: ObservationSpec | None = None
    design: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    spec_version: int = SPEC_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StudySpec:
        validate(data)
        obs = data.get("observations")
        compute = dict(data.get("compute", {}))
        if compute.get("max_workers") is not None:
            compute["max_workers"] = int(compute["max_workers"])
        return cls(
            name=data["name"],
            workflow=data["workflow"],
            model=ModelRef(**data["model"]),
            parameters=[ParameterSpec(**p) for p in data["parameters"]],
            seed_root=int(data["seed_root"]),
            observations=ObservationSpec(**obs) if obs else None,
            design=dict(data.get("design", {})),
            reference=dict(data.get("reference", {})),
            compute=ComputeSpec(**compute),
            spec_version=int(data.get("spec_version", SPEC_VERSION)),
        )

    @classmethod
    def load(cls, path: str | Path) -> StudySpec:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def dump(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1, sort_keys=True)

    def seeds(self) -> SeedLedger:
        return SeedLedger(self.seed_root)

    def workflow_run_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    def truth_generation_hash(self, provenance: dict) -> str:
        """Identity of the truth: truth-relevant spec subset plus provenance.

        ``provenance`` must carry ``input_hashes`` (immutable inputs), ``engine_fingerprint``,
        ``code_revision`` and ``dependency_lock``; missing keys or empty engine/code identities
        raise, so a truth is never identified without them.
        """
        missing = [k for k in TRUTH_PROVENANCE_KEYS if k not in provenance]
        if missing:
            raise KeyError(f"truth provenance is missing {missing}")
        empty = [
            k for k in ("engine_fingerprint", "code_revision") if not provenance[k]
        ]
        if empty:
            raise ValueError(f"truth provenance has no value for {empty}")
        subset = {
            "spec_version": self.spec_version,
            "model": asdict(self.model),
            "parameters": [asdict(p) for p in self.parameters],
            "observations": asdict(self.observations) if self.observations else None,
            "reference": self.reference,
            "seed_root": self.seed_root,
            "provenance": {k: provenance[k] for k in TRUTH_PROVENANCE_KEYS},
        }
        return sha256_text(canonical_json(subset))

    def identities(self, provenance: dict) -> dict:
        """The three identities of design section 3 (P3)."""
        return {
            "workflow_run_hash": self.workflow_run_hash(),
            "truth_generation_hash": self.truth_generation_hash(provenance),
            "score_version": SCORE_VERSION,
        }


STUDY_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["name", "workflow", "model", "parameters", "seed_root"],
    "additionalProperties": False,
    "properties": {
        "spec_version": {"type": "integer", "minimum": 1},
        "name": {"type": "string", "minLength": 1},
        "workflow": {"type": "string", "enum": list(WORKFLOWS)},
        "model": {
            "type": "object",
            "required": ["model_dir", "adapter"],
            "additionalProperties": False,
            "properties": {
                "model_dir": {"type": "string", "minLength": 1},
                "adapter": {"type": "string", "minLength": 1},
                "module": {"type": "string"},
                "cls": {"type": "string"},
                "kwargs": {"type": "object"},
            },
        },
        "parameters": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "family"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "family": {"type": "string", "minLength": 1},
                    "args": {"type": "object"},
                },
            },
        },
        "observations": {
            "type": ["object", "null"],
            "required": ["wells", "quantities", "report_times"],
            "additionalProperties": False,
            "properties": {
                "wells": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "quantities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "report_times": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                },
                "held_out_fraction": {
                    "type": "number",
                    "minimum": 0,
                    "exclusiveMaximum": 1,
                },
                "sigma_abs": {"type": ["number", "null"], "minimum": 0},
                "sigma_rel": {"type": ["number", "null"], "minimum": 0},
                "sigma_floor": {"type": "number", "minimum": 0},
            },
        },
        "design": {"type": "object"},
        "reference": {"type": "object"},
        "compute": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "platform": {"type": "string", "enum": list(PLATFORMS)},
                "walltime_s": {"type": "number", "exclusiveMinimum": 0},
                "max_workers": {"type": ["integer", "null"], "minimum": 1},
                "threads_per_member": {"type": "integer", "minimum": 1},
                "retries": {"type": "integer", "minimum": 0},
                "memory_per_member_gb": {
                    "type": ["number", "null"],
                    "exclusiveMinimum": 0,
                },
            },
        },
        "seed_root": {"type": "integer", "minimum": 0},
    },
}


def validate(data: dict) -> None:
    """Validate a study document against ``STUDY_SCHEMA``; raises ``jsonschema.ValidationError``."""
    import jsonschema

    jsonschema.validate(instance=data, schema=STUDY_SCHEMA)
