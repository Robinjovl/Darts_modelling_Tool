"""Optimization drivers (design section 6.4): first slice, exhaustive discrete well placement.

Candidates are feasible cells (not occupied by another well, at least ``min_spacing_m`` from every
other well); every candidate and the baseline (current position) is simulated through the
isolated executor; failed simulations are infeasible outcomes. The exhaustive best is the
reference optimum of the design's evaluation cases; ``regret`` is reported against it.
"""

from __future__ import annotations

import numpy as np

from workflows import gates
from workflows.adapter import load_adapter
from workflows.executor import IsolatedExecutor, MemberTask, run_members
from workflows.journal import StudyStore, atomic_write_json, provenance
from workflows.spec import StudySpec


def cumulative_production(
    observed: dict, wells: list, quantity: str, sign: float = -1.0
) -> float:
    """Trapezoidal cumulative volume over the report times; producer rates are negative here."""
    times = np.asarray(observed["time"], dtype=float)
    total = 0.0
    for well in wells:
        rate = sign * np.asarray(observed["values"][f"{well}:{quantity}"], dtype=float)
        total += float(
            np.trapezoid(rate, np.concatenate([[0.0], times])[1:])
            if rate.size > 1
            else rate[0] * times[0]
        )
    return total


def feasible_candidates(geometry: dict, well: str, min_spacing_m: float) -> list:
    """Cells where ``well`` may be placed: unoccupied and at least ``min_spacing_m`` from other wells."""
    centroids = np.asarray(geometry["centroids"], dtype=float)
    others = {name: info for name, info in geometry["wells"].items() if name != well}
    occupied = {info["cell"] for info in others.values()}
    positions = np.array([info["xyz"][:2] for info in others.values()], dtype=float)
    candidates = []
    for cell, xyz in enumerate(centroids):
        if cell in occupied:
            continue
        if (
            positions.size
            and np.min(np.linalg.norm(positions - xyz[:2], axis=1)) < min_spacing_m
        ):
            continue
        candidates.append({"cell": int(cell), "xyz": [float(v) for v in xyz]})
    return candidates


def _executor(spec: StudySpec):
    return IsolatedExecutor(
        n_workers=spec.compute.max_workers,
        threads_per_member=spec.compute.threads_per_member,
        timeout_s=spec.compute.walltime_s,
        retries=spec.compute.retries,
        memory_per_member_gb=spec.compute.memory_per_member_gb,
    )


def run_exhaustive(
    spec: StudySpec, study_root, executor=None, resume: bool = True
) -> dict:
    d = spec.design
    well, objective = d["well"], d.get("objective", "cumulative_oil")
    quantity, producers = (
        "oil_rate",
        list(d.get("producers") or spec.observations.wells),
    )
    store = StudyStore(study_root)
    store.write_spec(spec)
    adapter = load_adapter(spec.model.adapter, spec.model.model_dir)
    snapshot = adapter.snapshot(store.root / "inputs")
    geometry = adapter.geometry(snapshot)
    prov = provenance()
    prov["input_hashes"] = snapshot.files
    identities = (
        spec.identities(prov)
        if prov["engine_fingerprint"]
        else {"workflow_run_hash": spec.workflow_run_hash()}
    )
    candidates = feasible_candidates(geometry, well, float(d.get("min_spacing_m", 0.0)))
    if d.get("max_candidates") and len(candidates) > int(d["max_candidates"]):
        rng = spec.seeds().generator("optimizer")
        keep = np.sort(
            rng.choice(len(candidates), size=int(d["max_candidates"]), replace=False)
        )
        candidates = [candidates[i] for i in keep]
    members = [{"member": "baseline", "realization": {}}] + [
        {
            "member": f"c{c['cell']:05d}",
            "realization": {"well_xyz": {well: c["xyz"]}},
            **c,
        }
        for c in candidates
    ]
    tasks = [
        MemberTask(
            member=m["member"],
            study_root=str(store.root),
            adapter=spec.model.adapter,
            model_dir=spec.model.model_dir,
            snapshot={"root": snapshot.root, "files": snapshot.files},
            realization=m["realization"],
            report_times=list(spec.observations.report_times),
            observation=spec.observations.__dict__,
            spec_hash=identities["workflow_run_hash"],
            input_hash=snapshot.hash,
            binary_fingerprint=prov.get("engine_fingerprint") or "unknown",
        )
        for m in members
    ]
    store.write_manifest(
        {
            "identities": identities,
            "provenance": prov,
            "seeds": spec.seeds().to_dict(),
            "n_candidates": len(candidates),
            "well": well,
        }
    )
    results = run_members(store, tasks, executor or _executor(spec), resume=resume)
    rows = []
    for m, r in zip(members, results, strict=False):
        value = (
            cumulative_production(r.payload["observation"], producers, quantity)
            if r.status == "ok"
            else None
        )
        rows.append(
            {
                **{k: v for k, v in m.items() if k != "realization"},
                "status": r.status,
                "feasible": r.status == "ok",
                "objective": value,
            }
        )
    baseline = rows[0]["objective"]
    feasible = [row for row in rows[1:] if row["feasible"]]
    best = max(feasible, key=lambda row: row["objective"]) if feasible else None
    summary = {
        "well": well,
        "objective": objective,
        "producers": producers,
        "n_candidates": len(candidates),
        "n_infeasible": sum(1 for row in rows[1:] if not row["feasible"]),
        "baseline_objective": baseline,
        "best": best,
        "regret_of_baseline": gates.regret(baseline, best["objective"], baseline)
        if best and baseline is not None
        else None,
        "gates_version": gates.GATES_VERSION,
    }
    atomic_write_json(store.root / "candidates.json", rows)
    atomic_write_json(store.root / "optimize_summary.json", summary)
    store.append_event(
        {"stage": "optimize-done", "n_candidates": len(candidates), "best": best}
    )
    return summary
