"""Ensemble studies: designs in the unit cube, fan-out, and analysis (design sections 6.2, 6.3, 9).

Designs: Latin hypercube and Sobol sequences (``scipy.stats.qmc``), Morris trajectories and
Saltelli matrices (``N`` a power of two; second-order optional). Failure policy: a failed member is
retried by the executor and otherwise preserved as failed; Saltelli rows are never resampled, and
the conventional Sobol analysis refuses to run with failed members.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from workflows import cost
from workflows.adapter import load_adapter
from workflows.executor import IsolatedExecutor, MemberTask, run_members
from workflows.journal import StudyStore, atomic_write_json, provenance
from workflows.members import Parameterization, jsonable
from workflows.spec import StudySpec

MORRIS_LEVELS = 4


def lhs_design(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return qmc.LatinHypercube(d=dim, seed=rng).random(n)


def sobol_design(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if not cost.is_power_of_two(n):
        raise ValueError("Sobol sequence size must be a power of two")
    return qmc.Sobol(d=dim, scramble=True, seed=rng).random(n)


def morris_design(
    n_trajectories: int, dim: int, rng: np.random.Generator, levels: int = MORRIS_LEVELS
) -> dict:
    """Radial-free Morris trajectories on a ``levels`` grid; returns points and per-step metadata."""
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    points, steps = [], []
    for t in range(n_trajectories):
        base = rng.choice(grid[grid + delta <= 1.0 + 1e-12], size=dim)
        current = base.copy()
        points.append(current.copy())
        for i in rng.permutation(dim):
            current = current.copy()
            current[i] = (
                current[i] + delta
                if current[i] + delta <= 1.0 + 1e-12
                else current[i] - delta
            )
            sign = 1.0 if current[i] > points[-1][i] else -1.0
            points.append(current.copy())
            steps.append(
                {
                    "trajectory": t,
                    "dim": int(i),
                    "before": len(points) - 2,
                    "after": len(points) - 1,
                    "sign": sign,
                }
            )
    return {"points": np.array(points), "steps": steps, "delta": delta}


def saltelli_design(
    n_base: int, dim: int, rng: np.random.Generator, second_order: bool = False
) -> dict:
    cost.saltelli_runs(n_base, dim, second_order)  # validates the power-of-two base
    ab = qmc.Sobol(d=2 * dim, scramble=True, seed=rng).random(n_base)
    a, b = ab[:, :dim], ab[:, dim:]
    blocks = [("A", a), ("B", b)]
    for i in range(dim):
        ab_i = a.copy()
        ab_i[:, i] = b[:, i]
        blocks.append((f"AB{i}", ab_i))
    if second_order:
        for i in range(dim):
            ba_i = b.copy()
            ba_i[:, i] = a[:, i]
            blocks.append((f"BA{i}", ba_i))
    return {
        "points": np.vstack([p for _, p in blocks]),
        "blocks": [name for name, _ in blocks],
        "n_base": n_base,
        "dim": dim,
    }


def make_design(spec: StudySpec, dim: int) -> dict:
    d = spec.design
    rng = spec.seeds().generator("design")
    method = d.get("method", "lhs")
    if method == "lhs":
        return {"method": method, "points": lhs_design(int(d["n"]), dim, rng)}
    if method == "sobol_sequence":
        return {"method": method, "points": sobol_design(int(d["n"]), dim, rng)}
    if method == "morris":
        return {"method": method, **morris_design(int(d["n_trajectories"]), dim, rng)}
    if method == "saltelli":
        return {
            "method": method,
            **saltelli_design(
                int(d["n_base"]), dim, rng, bool(d.get("second_order", False))
            ),
        }
    raise ValueError(f"unknown ensemble design method {method!r}")


def prepare(spec: StudySpec, study_root: str | Path) -> tuple:
    """Snapshot inputs, build the parameterization and design, write spec and manifest."""
    store = StudyStore(study_root)
    store.write_spec(spec)
    adapter = load_adapter(spec.model.adapter, spec.model.model_dir)
    snapshot = adapter.snapshot(store.root / "inputs")
    geometry = adapter.geometry(snapshot) if hasattr(adapter, "geometry") else None
    parameterization = Parameterization.from_specs(spec.parameters, geometry)
    design = make_design(spec, parameterization.dim)
    prov = provenance()
    prov["input_hashes"] = snapshot.files
    manifest = {
        "identities": spec.identities(prov)
        if prov["engine_fingerprint"]
        else {"workflow_run_hash": spec.workflow_run_hash()},
        "provenance": prov,
        "seeds": spec.seeds().to_dict(),
        "design": {k: v for k, v in design.items() if k != "points"},
        "labels": parameterization.labels(),
        "n_members": int(design["points"].shape[0]),
        "planned_simulations": cost.planned_simulations(spec),
    }
    store.write_manifest(manifest)
    np.save(store.root / "design_points.npy", design["points"])
    return store, adapter, snapshot, parameterization, design


def member_tasks(
    store: StudyStore, spec: StudySpec, snapshot, parameterization, design
) -> list:
    manifest = store.read_manifest()
    fingerprint = manifest["provenance"].get("engine_fingerprint") or "unknown"
    tasks = []
    for index, point in enumerate(design["points"]):
        member = f"m{index:05d}"
        tasks.append(
            MemberTask(
                member=member,
                study_root=str(store.root),
                adapter=spec.model.adapter,
                model_dir=spec.model.model_dir,
                snapshot={"root": snapshot.root, "files": snapshot.files},
                realization=jsonable(parameterization.to_realization(point)),
                report_times=list(spec.observations.report_times),
                observation=spec.observations.__dict__ if spec.observations else None,
                spec_hash=manifest["identities"]["workflow_run_hash"],
                input_hash=snapshot.hash,
                binary_fingerprint=fingerprint,
            )
        )
    return tasks


def run_study(
    spec: StudySpec, study_root: str | Path, executor=None, resume: bool = True
) -> dict:
    store, adapter, snapshot, parameterization, design = prepare(spec, study_root)
    tasks = member_tasks(store, spec, snapshot, parameterization, design)
    done = store.completed_members() if resume else set()
    todo = [t for t in tasks if t.member not in done]
    store.append_event(
        {"stage": "run", "n_members": len(tasks), "resumed": len(tasks) - len(todo)}
    )
    if executor is None:
        executor = IsolatedExecutor(
            n_workers=spec.compute.max_workers,
            threads_per_member=spec.compute.threads_per_member,
            timeout_s=spec.compute.walltime_s,
            retries=spec.compute.retries,
            memory_per_member_gb=spec.compute.memory_per_member_gb,
        )
    results = run_members(store, todo, executor) if todo else []
    summary = {
        "study": str(store.root),
        "n_members": len(tasks),
        "n_run": len(todo),
        "n_ok": sum(1 for r in results if r.status == "ok") + len(done),
        "failures": {
            t.member: r.status
            for t, r in zip(todo, results, strict=False)
            if r.status != "ok"
        },
    }
    store.append_event({"stage": "run-done", **summary})
    return summary


def load_results(store: StudyStore) -> tuple:
    """Per-member observation values in design order; ``None`` for members without a result."""
    manifest = store.read_manifest()
    n = manifest["n_members"]
    rows, failed = [], []
    for index in range(n):
        path = store.members_dir / f"m{index:05d}" / "result.json"
        if not path.exists():
            rows.append(None)
            failed.append(index)
            continue
        with open(path, encoding="utf-8") as handle:
            rows.append(json.load(handle)["observation"])
    return rows, failed


def bootstrap_percentiles(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = 500
) -> dict:
    """P10/P50/P90 as statistical percentiles with bootstrap 90 % intervals."""
    out = {}
    for name, q in (("P10", 10), ("P50", 50), ("P90", 90)):
        estimate = float(np.percentile(values, q))
        boots = [
            np.percentile(rng.choice(values, size=values.size, replace=True), q)
            for _ in range(n_boot)
        ]
        out[name] = {
            "value": estimate,
            "ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
        }
    out["convention"] = "statistical percentiles (P10 = 10th percentile of the value)"
    return out


def morris_indices(outputs: np.ndarray, design: dict, labels: list) -> dict:
    effects = {i: [] for i in range(len(labels))}
    for step in design["steps"]:
        ee = (
            (outputs[step["after"]] - outputs[step["before"]])
            * step["sign"]
            / design["delta"]
        )
        effects[step["dim"]].append(ee)
    return {
        labels[i]: {
            "mu_star": float(np.mean(np.abs(e))),
            "mu": float(np.mean(e)),
            "sigma": float(np.std(e, ddof=1)) if len(e) > 1 else 0.0,
        }
        for i, e in effects.items()
    }


def sobol_indices(outputs: np.ndarray, design: dict, labels: list) -> dict:
    """First- and total-order indices (Saltelli 2010 / Jansen estimators)."""
    n, dim = design["n_base"], design["dim"]
    blocks = {
        name: outputs[k * n : (k + 1) * n] for k, name in enumerate(design["blocks"])
    }
    fa, fb = blocks["A"], blocks["B"]
    var = np.var(np.concatenate([fa, fb]), ddof=1)
    result = {}
    for i in range(dim):
        fab = blocks[f"AB{i}"]
        first = np.mean(fb * (fab - fa)) / var
        total = 0.5 * np.mean((fa - fab) ** 2) / var
        result[labels[i]] = {"S1": float(first), "ST": float(total)}
    return result


def analyze(study_root: str | Path) -> dict:
    store = StudyStore(study_root)
    spec = store.read_spec()
    manifest = store.read_manifest()
    rows, failed = load_results(store)
    design = dict(manifest["design"])
    design["points"] = np.load(store.root / "design_points.npy")
    labels = manifest["labels"]
    rng = spec.seeds().generator("design", index=1)
    analysis = {
        "n_members": manifest["n_members"],
        "n_failed": len(failed),
        "failed_members": failed,
        "quantities": {},
    }
    keys = (
        sorted(rows[[i for i, r in enumerate(rows) if r is not None][0]]["values"])
        if len(failed) < len(rows)
        else []
    )
    for key in keys:
        matrix = np.array(
            [
                r["values"][key]
                if r is not None
                else [np.nan] * len(spec.observations.report_times)
                for r in rows
            ]
        )
        final = matrix[:, -1]
        valid = final[~np.isnan(final)]
        entry = {
            "final_time": spec.observations.report_times[-1],
            "percentiles": bootstrap_percentiles(valid, rng),
        }
        if design["method"] == "morris":
            if failed:
                entry["morris"] = {
                    "error": "failed members present; Morris effects require complete trajectories"
                }
            else:
                entry["morris"] = morris_indices(final, design, labels)
        if design["method"] == "saltelli":
            if failed:
                entry["sobol"] = {
                    "error": "failed members present; Saltelli rows are never resampled"
                }
            else:
                entry["sobol"] = sobol_indices(final, design, labels)
        analysis["quantities"][key] = entry
    atomic_write_json(store.root / "analysis.json", analysis)
    store.append_event(
        {"stage": "analyze", "n_failed": len(failed), "quantities": keys}
    )
    return analysis
