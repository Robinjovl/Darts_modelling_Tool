"""Hidden scorer: compares broker-owned study results with the hidden truths using the gates.

Runs only in the parent after the agent has exited; the evaluated agent never sees this file
(masked in the sandbox). ``score_version`` is recorded with every score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from workflows import gates
from workflows.spec import SCORE_VERSION


def score_optimize(study_dir: Path, reference: dict, tolerance: float) -> dict:
    with open(study_dir / "optimize_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    best = summary["best"]
    regret = (
        gates.regret(
            best["objective"], reference["objective"], summary["baseline_objective"]
        )
        if best
        else None
    )
    return {
        "regret": regret,
        "passed": regret is not None and regret <= tolerance,
        "uninformative": regret is None,
    }


def score_hm(study_dir: Path, reference: dict, tolerance: dict) -> dict:
    with open(study_dir / "esmda_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)
    final = summary["steps"][-1]
    checks = {
        "held_out_rmse": final["held_out_rmse"] is not None
        and final["held_out_rmse"] <= tolerance.get("held_out_rmse", 2.0),
        "coverage80": final["coverage80_held_out"] is not None
        and tolerance.get("coverage_low", 0.6) <= final["coverage80_held_out"] <= 1.0,
        "spread_ratio": final["spread_ratio"] >= tolerance.get("spread_ratio_min", 0.1),
    }
    return {**final, "checks": checks, "passed": all(checks.values())}


def score_ensemble(study_dir: Path, reference: dict, tolerance: float) -> dict:
    with open(study_dir / "analysis.json", encoding="utf-8") as handle:
        analysis = json.load(handle)
    checks = {}
    for key, ref in reference["quantities"].items():
        got = analysis["quantities"].get(key, {}).get("percentiles", {})
        for p in ("P10", "P50", "P90"):
            if p in got and p in ref:
                scale = max(abs(ref[p]["value"]), 1e-12)
                checks[f"{key}:{p}"] = abs(
                    got[p]["value"] - ref[p]["value"]
                ) / scale <= tolerance + (ref[p]["ci90"][1] - ref[p]["ci90"][0]) / (
                    2 * scale
                )
    return {"checks": checks, "passed": bool(checks) and all(checks.values())}


def score(case: dict, run_root: str | Path) -> dict:
    run_root = Path(run_root)
    truth_dir = Path(__file__).with_name("truth") / case["name"]
    with open(truth_dir / "reference.json", encoding="utf-8") as handle:
        reference = json.load(handle)
    study_dir = run_root / "results" / case["study"]
    workflow = case["workflow"]
    if workflow == "optimize":
        result = score_optimize(study_dir, reference, case["tolerance"])
    elif workflow == "hm-esmda":
        result = score_hm(study_dir, reference, case["tolerance"])
    elif workflow == "ensemble":
        result = score_ensemble(study_dir, reference, case["tolerance"])
    else:
        raise ValueError(f"no scorer for workflow {workflow!r}")
    return {
        "case": case["name"],
        "score_version": SCORE_VERSION,
        "gates_version": gates.GATES_VERSION,
        **result,
    }


def reference_from_study(workflow: str, study_dir: Path) -> dict:
    """Build a hidden reference from a controller-run study (exhaustive optimum, analysis, truth)."""
    if workflow == "optimize":
        with open(study_dir / "optimize_summary.json", encoding="utf-8") as handle:
            return {"objective": json.load(handle)["best"]["objective"]}
    if workflow == "ensemble":
        with open(study_dir / "analysis.json", encoding="utf-8") as handle:
            return {
                "quantities": {
                    k: v["percentiles"]
                    for k, v in json.load(handle)["quantities"].items()
                }
            }
    if workflow == "hm-esmda":
        with open(study_dir / "truth" / "truth.json", encoding="utf-8") as handle:
            truth = json.load(handle)
        return {"x_true_log10_mean": float(np.mean(truth["x_true_log10"]))}
    raise ValueError(workflow)
