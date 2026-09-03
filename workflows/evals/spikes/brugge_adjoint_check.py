"""Assertive feasibility check: adjoint gradient vs central finite differences on Uniform_Brugge.

Run from a scratch copy of ``models/Uniform_Brugge`` (the model reads ``Brugge_struct/`` relatively
and writes ``Brugge_model.msh`` into the working directory)::

    cp -r models/Uniform_Brugge /tmp/brugge_spike
    cp workflows/evals/spikes/brugge_* /tmp/brugge_spike/
    cd /tmp/brugge_spike
    PYTHONPATH=<repo> OMP_NUM_THREADS=1 <env>/bin/python brugge_adjoint_check.py --out result.json

Verified conventions (design document, Appendix C):

* observations are instantaneous rates from ``engine.time_data`` at report times, production
  negated; ``engine.time_data_report`` holds interval averages and must not be used;
* the CI model's injectors inject gas, so the injector observable is the gas rate;
* the objective must vanish at the truth; adjoint and central differences are compared on a
  stratified control subset over a step-size sweep and along random full-vector directions.

The script exits non-zero when any assertion fails and writes a provenance-rich JSON result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


@contextmanager
def contained_output(log_path: str):
    """Redirect the process-level stdout/stderr (incl. gmsh/C++) into a log file."""
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


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: str, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", repo, *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def provenance(repo: str) -> dict:
    import darts

    darts_dir = Path(darts.__file__).parent
    engine_libs = sorted(darts_dir.glob("engines*.so")) + sorted(
        darts_dir.glob("engines*.pyd")
    )
    inputs = sorted(Path("Brugge_struct").glob("*")) + [Path("Brugge_model.msh")]
    return {
        "repo_head": git(repo, "rev-parse", "HEAD"),
        "repo_dirty_paths": git(repo, "status", "--porcelain").count("\n")
        if git(repo, "status", "--porcelain")
        else 0,
        "darts_file": str(darts.__file__),
        "darts_version": getattr(darts, "__version__", "unknown"),
        "engine_binaries": {p.name: sha256_of(p) for p in engine_libs},
        "input_hashes": {str(p): sha256_of(p) for p in inputs if p.is_file()},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "affinity_cores": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }


def atomic_write_json(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    os.replace(tmp, path)


def build_observations(truth_model) -> pd.DataFrame:
    """Instantaneous rates at report times, production rates negated (report sign convention)."""
    time_data = pd.DataFrame.from_dict(truth_model.physics.engine.time_data)
    report_times = pd.DataFrame.from_dict(truth_model.physics.engine.time_data_report)[
        "time"
    ].to_numpy()
    at_report = np.isclose(
        time_data["time"].to_numpy()[:, None],
        report_times[None, :],
        rtol=0.0,
        atol=1e-9,
    ).any(axis=1)
    obs = time_data[at_report].copy().reset_index(drop=True)
    if len(obs) != len(report_times):
        raise RuntimeError(
            f"report-time matching found {len(obs)} rows for {len(report_times)} report times"
        )
    for column in [c for c in obs.columns if "rate" in c]:
        if column.split(" : ")[0].startswith("P"):
            obs[column] = -obs[column]
    return obs


def interface_pairs(mesh, n_interfaces: int):
    """Unique reservoir interfaces in first-appearance order, or None if the count disagrees."""
    block_m = np.array(mesh.block_m, copy=False)
    block_p = np.array(mesh.block_p, copy=False)
    n_res = mesh.n_res_blocks
    pairs, seen = [], set()
    for m, p in zip(block_m, block_p, strict=False):
        if m < n_res and p < n_res:
            key = (min(m, p), max(m, p))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs if len(pairs) == n_interfaces else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--repo", default=os.environ.get("PYTHONPATH", "").split(os.pathsep)[0] or "."
    )
    parser.add_argument("--horizon", type=float, default=200.0)
    parser.add_argument("--report-step", type=float, default=20.0)
    parser.add_argument("--subset", type=int, default=12)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument(
        "--steps", default="1e-3,1e-4,1e-5", help="relative FD step sizes"
    )
    parser.add_argument("--perturbation", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-angle-deg", type=float, default=5.0)
    parser.add_argument("--max-median-rel-err", type=float, default=1e-3)
    parser.add_argument("--max-direction-rel-err", type=float, default=1e-2)
    parser.add_argument("--out", default="brugge_adjoint_check_result.json")
    parser.add_argument("--log", default="brugge_adjoint_check.log")
    args = parser.parse_args()

    from darts.engines import redirect_darts_output, value_vector, well_control_iface
    from darts.models.opt.opt_module_settings import (
        model_modifier_aggregator,
        transmissibility_modifier,
        well_index_modifier,
    )

    redirect_darts_output(args.log)
    rng = np.random.default_rng(args.seed)
    steps = [float(s) for s in args.steps.split(",")]
    timings = {}

    with contained_output(args.log):
        from brugge_model_adj import Model

        t0 = time.perf_counter()
        truth = Model(args.horizon, args.report_step)
        truth.init()
        truth.set_output(verbose=False)
        timings["truth_build_and_init_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        truth.run()
        timings["truth_forward_s"] = time.perf_counter() - t0
        observations = build_observations(truth)
        del truth

        t0 = time.perf_counter()
        proxy = Model(args.horizon, args.report_step)
        proxy.init()
        proxy.set_output(output_folder="out_proxy", save_initial=False, verbose=False)
        timings["proxy_build_and_init_s"] = time.perf_counter() - t0

    modifiers = model_modifier_aggregator()
    modifiers.append(transmissibility_modifier())
    modifiers.modifiers[-1].norms = 10000.0
    modifiers.append(well_index_modifier())
    modifiers.modifiers[-1].norms = 1000.0
    x_true = modifiers.get_x0(proxy)
    tran = value_vector([])
    tran_d = value_vector([])
    proxy.reservoir.mesh.get_res_tran(tran, tran_d)
    n_tran = len(np.array(tran))
    n_wi = sum(len(w.perforations) for w in proxy.reservoir.wells)
    n_ctrl = n_tran + n_wi
    n_report = len(observations)

    producers = [w.name for w in proxy.reservoir.wells if w.name.startswith("P")]
    injectors = [w.name for w in proxy.reservoir.wells if w.name.startswith("I")]
    proxy.scale_function_value = 1e-5
    proxy.modifier = modifiers
    proxy.x_idx = [0, n_tran, n_ctrl]
    proxy.objfun_prod_phase_rate = True
    proxy.prod_well_name = producers
    proxy.prod_phase_name = ["oil", "wat"]
    proxy.prod_weights = np.ones((len(producers), 2, n_report))
    proxy.objfun_inj_phase_rate = True
    proxy.inj_well_name = injectors
    proxy.inj_phase_name = ["gas"]
    proxy.inj_weights = np.ones((len(injectors), 1, n_report))
    proxy.objfun_BHP = False
    proxy.objfun_well_tempr = False
    proxy.objfun_temperature = False
    proxy.objfun_customized_op = False
    proxy.observation_rate_type = well_control_iface.VOLUMETRIC_RATE
    proxy.activate_opt_options()
    proxy.eps = 1e-7
    proxy.set_objfun(proxy.objfun_assembly)
    proxy.set_observation_data_report(observations)
    proxy.set_observation_data(observations)
    proxy.BHP_report_data = observations
    proxy.well_tempr_report_data = observations
    proxy.job_id = "spike"

    def objective(x: np.ndarray) -> float:
        with contained_output(args.log):
            return float(proxy.make_opt_step_adjoint_method(x))

    f_true = objective(x_true)
    x0 = np.clip(
        x_true * (1.0 + args.perturbation * rng.standard_normal(n_ctrl)), 1e-6, None
    )
    t0 = time.perf_counter()
    f0 = objective(x0)
    timings["forward_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with contained_output(args.log):
        g_adj = np.array(proxy.grad_adjoint_method_all(x0), dtype=float).copy()
    timings["adjoint_gradient_s"] = time.perf_counter() - t0

    # Stratified control subset: near-well vs far-field interfaces when the interface map is
    # recoverable from the mesh, otherwise by sensitivity magnitude; plus well-index controls.
    perforated = {per[1] for w in proxy.reservoir.wells for per in w.perforations}
    pairs = interface_pairs(proxy.reservoir.mesh, n_tran)
    n_per_group = max(1, (args.subset - 4) // 2)
    if pairs is not None:
        near = [
            i for i, (m, p) in enumerate(pairs) if m in perforated or p in perforated
        ]
        far = [i for i in range(n_tran) if i not in set(near)]
        stratification = "near-well/far-field"
    else:
        order = np.argsort(-np.abs(g_adj[:n_tran]))
        near, far = list(order[: n_tran // 2]), list(order[n_tran // 2 :])
        stratification = "sensitivity-magnitude"
    subset = np.concatenate(
        [
            rng.choice(near, min(n_per_group, len(near)), replace=False),
            rng.choice(far, min(n_per_group, len(far)), replace=False),
            n_tran + rng.choice(n_wi, min(4, n_wi), replace=False),
        ]
    ).astype(int)

    rows = []
    for i in subset:
        entry = {
            "control": int(i),
            "family": "T" if i < n_tran else "WI",
            "group": "WI"
            if i >= n_tran
            else ("near-well" if i in near else "far-field"),
            "adjoint": float(g_adj[i]),
            "fd": {},
        }
        for h_rel in steps:
            h = h_rel * max(abs(x0[i]), 1e-3)
            x_plus, x_minus = x0.copy(), x0.copy()
            x_plus[i] += h
            x_minus[i] -= h
            g_fd = (objective(x_plus) - objective(x_minus)) / (2.0 * h)
            rel = abs(g_fd - g_adj[i]) / max(abs(g_fd), abs(g_adj[i]), 1e-30)
            entry["fd"][str(h_rel)] = {"value": float(g_fd), "rel_err": float(rel)}
        entry["best_rel_err"] = min(v["rel_err"] for v in entry["fd"].values())
        rows.append(entry)

    directions = []
    for k in range(args.directions):
        direction = rng.standard_normal(n_ctrl) * np.maximum(np.abs(x0), 1e-3)
        direction /= np.linalg.norm(direction)
        h = 1e-4 * np.linalg.norm(x0)
        dd_fd = (objective(x0 + h * direction) - objective(x0 - h * direction)) / (
            2.0 * h
        )
        dd_adj = float(g_adj @ direction)
        directions.append(
            {
                "index": k,
                "fd": float(dd_fd),
                "adjoint": dd_adj,
                "rel_err": float(
                    abs(dd_fd - dd_adj) / max(abs(dd_fd), abs(dd_adj), 1e-30)
                ),
            }
        )

    best_step = {
        str(h): float(np.median([r["fd"][str(h)]["rel_err"] for r in rows]))
        for h in steps
    }
    adj_sub = np.array([r["adjoint"] for r in rows])
    fd_sub = np.array(
        [r["fd"][min(best_step, key=best_step.get)]["value"] for r in rows]
    )
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(
                    adj_sub
                    @ fd_sub
                    / (np.linalg.norm(adj_sub) * np.linalg.norm(fd_sub)),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    median_best = float(np.median([r["best_rel_err"] for r in rows]))

    checks = {
        "objective_vanishes_at_truth": f_true <= 1e-12 * max(f0, 1.0),
        "subset_angle_below_max": angle < args.max_angle_deg,
        "median_best_rel_err_below_max": median_best < args.max_median_rel_err,
        "directional_derivatives_agree": all(
            d["rel_err"] < args.max_direction_rel_err for d in directions
        ),
        "stratified_groups_present": len({r["group"] for r in rows}) >= 2,
    }
    result = {
        "model": "Uniform_Brugge (CI coarse proxy)",
        "horizon_days": args.horizon,
        "report_step_days": args.report_step,
        "n_controls": n_ctrl,
        "n_transmissibilities": n_tran,
        "n_well_indices": n_wi,
        "objective_at_truth": f_true,
        "objective_at_start": f0,
        "perturbation": args.perturbation,
        "stratification": stratification,
        "subset": rows,
        "median_rel_err_by_step": best_step,
        "median_best_rel_err": median_best,
        "max_best_rel_err": float(max(r["best_rel_err"] for r in rows)),
        "subset_angle_deg": angle,
        "directional_derivatives": directions,
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "checks": checks,
        "passed": all(checks.values()),
        "provenance": provenance(args.repo),
        "seed": args.seed,
        "thresholds": {
            "max_angle_deg": args.max_angle_deg,
            "max_median_rel_err": args.max_median_rel_err,
            "max_direction_rel_err": args.max_direction_rel_err,
        },
    }
    atomic_write_json(args.out, result)
    summary = {
        k: result[k]
        for k in (
            "objective_at_truth",
            "objective_at_start",
            "median_best_rel_err",
            "max_best_rel_err",
            "subset_angle_deg",
            "stratification",
            "checks",
            "passed",
        )
    }
    print(json.dumps(summary))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
