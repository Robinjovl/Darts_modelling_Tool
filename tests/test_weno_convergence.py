"""Regression guard for the WENO2 accuracy benefit and oscillation-free behaviour.

A fast fixed-domain h-refinement (dt ~ dx^2) of a smooth advected composition
bump on a uniform-mobility two-phase system.  It asserts the three properties a
correct WENO2 transport reconstruction must have relative to single-point
upstream (SPU):

  * lower L1 error than SPU at every resolution,
  * a faster error-decay rate than SPU (super-linear vs ~first order),
  * boundedness: no spurious over/undershoot beyond the initial data range.

This is deliberately coarse so it stays a few seconds; the full multi-regime
convergence study lives in ``models/2ph_comp/run_convergence.py``.
"""

import importlib.util
from pathlib import Path

import numpy as np

from darts.engines import redirect_darts_output


def _load_conv_model():
    path = Path(__file__).parents[1] / "models" / "2ph_comp" / "conv_model.py"
    spec = importlib.util.spec_from_file_location("weno_conv_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConvModel


def _run(ConvModel, scheme, nx, n_steps, runtime=8.0, L=100.0):
    redirect_darts_output("weno_conv_test.log")
    model = ConvModel(
        transport_scheme=scheme,
        nx=nx,
        domain_length=L,
        regime="smooth",
        n_steps=n_steps,
        runtime=runtime,
        bump_amp=0.25,
        z_bg=(0.5, 0.2),
    )
    model.init()
    model.set_output()
    model.run(runtime)
    X = np.array(model.physics.engine.X, copy=True)
    nc = model.physics.nc
    nb = model.reservoir.mesh.n_res_blocks
    return X[1 : nb * nc : nc][:nx].copy()


def _coarsen(fine, factor):
    n = len(fine) // factor
    return fine[: n * factor].reshape(n, factor).mean(axis=1)


def test_weno2_beats_spu_and_stays_bounded():
    ConvModel = _load_conv_model()
    ref_nx, base_nx, base_steps = 128, 32, 48

    def steps(nx):
        return max(base_steps, int(round(base_steps * (nx / base_nx) ** 2)))

    reference = _run(ConvModel, "weno2", ref_nx, steps(ref_nx))

    grids = [32, 64]
    err = {"spu": [], "weno2": []}
    for scheme in ("spu", "weno2"):
        for nx in grids:
            z = _run(ConvModel, scheme, nx, steps(nx))
            # bounded: background is 0.5, bump peak <= 0.75; allow tiny FP slack
            assert z.min() >= 0.5 - 1e-6, f"{scheme} nx={nx} undershoot {z.min()}"
            assert z.max() <= 0.75 + 1e-3, f"{scheme} nx={nx} overshoot {z.max()}"
            ref_coarse = _coarsen(reference, ref_nx // nx)
            err[scheme].append(float(np.mean(np.abs(z - ref_coarse))))

    # WENO2 is strictly more accurate than SPU at every resolution.
    for i, nx in enumerate(grids):
        assert err["weno2"][i] < err["spu"][i], (
            f"WENO2 not more accurate than SPU at nx={nx}: "
            f"{err['weno2'][i]:.3e} vs {err['spu'][i]:.3e}"
        )

    # WENO2 error decays faster than SPU under refinement (super-linear vs ~1st).
    weno_rate = err["weno2"][0] / err["weno2"][1]
    spu_rate = err["spu"][0] / err["spu"][1]
    assert weno_rate > spu_rate, (
        f"WENO2 refinement rate {weno_rate:.2f} not faster than SPU {spu_rate:.2f}"
    )
    assert weno_rate > 1.7, f"WENO2 not clearly super-linear (rate {weno_rate:.2f})"
