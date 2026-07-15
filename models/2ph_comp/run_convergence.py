"""Driver for the WENO2 vs SPU convergence / oscillation study.

Runs a fixed-domain h-refinement sweep for both schemes, extracts the final
CO2 overall-composition profile, and computes:

  * Richardson self-convergence order in the L1 norm (no external reference).
  * L1/L2 error vs a highly-refined reference (finest WENO run).
  * total-variation and over/undershoot (oscillation) metrics.
  * per-run wall time, Newton/linear iteration counts.

dt is scaled ~ dx^2 (n_steps ~ nx^2) so the backward-Euler temporal error does
not mask the 2nd-order spatial term (paper: quadratic convergence needs
dt ~ dx^2).
"""

import argparse
import json
import time

import numpy as np
from darts.engines import redirect_darts_output
from conv_model import ConvModel


def run_one(scheme, nx, L, regime, base_nx, base_steps, runtime, uniform_mobility):
    redirect_darts_output("conv_run.log")
    n_steps = max(base_steps, int(round(base_steps * (nx / base_nx) ** 2)))
    t0 = time.time()
    m = ConvModel(
        transport_scheme=scheme, nx=nx, domain_length=L, regime=regime,
        n_steps=n_steps, runtime=runtime, uniform_mobility=uniform_mobility,
    )
    m.init()
    m.set_output()
    m.run(runtime)
    wall = time.time() - t0

    X = np.array(m.physics.engine.X, copy=True)
    nc = m.physics.nc
    nb = m.reservoir.mesh.n_res_blocks
    z_co2 = X[1:nb * nc:nc][:nx].copy()
    pressure = X[0:nb * nc:nc][:nx].copy()

    eng = m.physics.engine
    stats = {
        "nx": nx, "n_steps": n_steps, "wall": wall,
        "newton": int(eng.stat.n_newton_total),
        "linear": int(eng.stat.n_linear_total),
        "timesteps": int(eng.stat.n_timesteps_total),
        "wasted_newton": int(eng.stat.n_newton_wasted),
        "jac_time": m.timer.node["simulation"].node["jacobian assembly"].get_timer()
        if "jacobian assembly" in m.timer.node["simulation"].node else None,
        "sim_time": m.timer.node["simulation"].get_timer(),
    }
    weno_fallbacks = None
    if scheme == "weno2":
        weno_fallbacks = {
            "bound": int(getattr(eng, "weno_bound_fallback_count", 0)),
            "geometry": int(getattr(eng, "weno_geometry_fallback_count", 0)),
        }
    stats["weno_fallbacks"] = weno_fallbacks
    return z_co2, pressure, stats


def coarsen(fine, factor):
    """Conservative block-average of a fine profile to a coarser grid."""
    n = len(fine) // factor
    return fine[: n * factor].reshape(n, factor).mean(axis=1)


def total_variation(u):
    return float(np.sum(np.abs(np.diff(u))))


def l1(a, b):
    return float(np.mean(np.abs(a - b)))


def l2(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="smooth", choices=["smooth", "ramp", "sharp"])
    ap.add_argument("--L", type=float, default=100.0)
    ap.add_argument("--runtime", type=float, default=150.0)
    ap.add_argument("--base-nx", type=int, default=32)
    ap.add_argument("--base-steps", type=int, default=60)
    ap.add_argument("--nxs", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--ref-nx", type=int, default=1024)
    ap.add_argument("--nonuniform-mobility", action="store_true")
    ap.add_argument("--out", default="conv_result.json")
    args = ap.parse_args()

    uniform_mobility = not args.nonuniform_mobility
    schemes = ["spu", "weno2"]
    results = {"config": vars(args), "runs": {}}

    # reference: finest WENO run (also a finest SPU run for cross-check)
    ref_profiles = {}
    for scheme in schemes:
        z, p, st = run_one(scheme, args.ref_nx, args.L, args.regime,
                           args.base_nx, args.base_steps, args.runtime, uniform_mobility)
        ref_profiles[scheme] = z
        print(f"[ref {scheme}] nx={args.ref_nx} steps={st['n_steps']} wall={st['wall']:.1f}s "
              f"TV={total_variation(z):.5f} min={z.min():.6f} max={z.max():.6f}", flush=True)
    reference = ref_profiles["weno2"]

    for scheme in schemes:
        results["runs"][scheme] = []
        for nx in args.nxs:
            z, p, st = run_one(scheme, nx, args.L, args.regime,
                               args.base_nx, args.base_steps, args.runtime, uniform_mobility)
            factor = args.ref_nx // nx
            ref_coarse = coarsen(reference, factor)
            err_l1 = l1(z, ref_coarse)
            err_l2 = l2(z, ref_coarse)
            rec = {
                **st,
                "err_l1_vs_ref": err_l1, "err_l2_vs_ref": err_l2,
                "TV": total_variation(z),
                "z_min": float(z.min()), "z_max": float(z.max()),
                "profile": z.tolist(),
            }
            results["runs"][scheme].append(rec)
            print(f"[{scheme}] nx={nx:4d} steps={st['n_steps']:6d} wall={st['wall']:7.2f}s "
                  f"L1err={err_l1:.3e} TV={rec['TV']:.5f} "
                  f"min={rec['z_min']:.6f} max={rec['z_max']:.6f} "
                  f"newton={st['newton']} lin={st['linear']} "
                  f"fb={st['weno_fallbacks']}", flush=True)

    # Richardson self-convergence order (successive-grid differences, block-avg)
    for scheme in schemes:
        runs = {r["nx"]: np.array(r["profile"]) for r in results["runs"][scheme]}
        nxs = sorted(runs)
        diffs = {}
        for i in range(len(nxs) - 1):
            fine, coarse = nxs[i + 1], nxs[i]
            f = coarsen(runs[fine], fine // coarse)
            diffs[(coarse, fine)] = l1(f, runs[coarse])
        orders = []
        keys = list(diffs)
        for i in range(len(keys) - 1):
            d_coarse = diffs[keys[i]]
            d_fine = diffs[keys[i + 1]]
            if d_fine > 0:
                orders.append(float(np.log2(d_coarse / d_fine)))
        results["runs"][scheme + "_richardson"] = {
            "successive_L1_diffs": {f"{k[0]}-{k[1]}": v for k, v in diffs.items()},
            "orders": orders,
        }
        pretty = {f'{k[0]}-{k[1]}': round(v, 6) for k, v in diffs.items()}
        print(f"[Richardson {scheme}] L1 diffs={pretty} "
              f"orders={[round(o,3) for o in orders]}", flush=True)

    # error-vs-ref orders (slope of log(L1err) vs log(nx))
    for scheme in schemes:
        runs = results["runs"][scheme]
        lognx = np.log(np.array([r["nx"] for r in runs]))
        logerr = np.log(np.array([r["err_l1_vs_ref"] for r in runs]))
        slope = float(-np.polyfit(lognx, logerr, 1)[0])
        results["runs"][scheme + "_ref_order"] = slope
        print(f"[Ref-order {scheme}] observed L1 order vs finest-ref = {slope:.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written {args.out}", flush=True)


if __name__ == "__main__":
    main()
