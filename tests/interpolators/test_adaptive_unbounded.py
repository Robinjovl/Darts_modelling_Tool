"""
Performance + correctness test for the multi-index-keyed multilinear adaptive
CPU interpolator (Phase 1).

Goals:
- Verify in-bounds correctness vs. analytic linear evaluator (bit-exact).
- Verify out-of-bounds queries:
    * return the exact linear extrapolation (cache extended on demand),
    * do not clamp the lookup to the prescribed bounds,
    * grow the cache and remain free of extrapolation-clipping warnings.
- Measure interpolation throughput for in-bounds and out-of-bounds workloads,
  with a cold cache and a warm cache.
- Verify legacy pickle round-trip of point_data (in-bounds cells).

Run with the obl_bounds conda env:
    python tests/interpolators/test_adaptive_unbounded.py
"""

import os
import pickle
import time
from contextlib import contextmanager

import numpy as np
import pytest

# Probe which N_DIMS / N_OPS templates were built.
import darts.interpolators as _itor_module
from darts.engines import index_vector, value_vector
from darts.interpolators import operator_set_evaluator_iface

N_DIMS = 3
N_OPS = 4


def _resolve_cls(base):
    # The CPU adaptive templates are stamped per index type: '_l_' (uint64 index,
    # the one currently built) and '_i_' (uint32, legacy/not compiled). Pick
    # whichever is exposed so the test is robust to that build choice.
    for tag in ("l", "i"):
        name = f"{base}_{tag}_d_{N_DIMS}_{N_OPS}"
        if hasattr(_itor_module, name):
            return name
    raise SystemExit(
        f"No {base}_[l|i]_d_{N_DIMS}_{N_OPS} template exposed in darts.interpolators — "
        f"rebuild with this (n_dims, n_ops) pair."
    )


ML_CLS_NAME = _resolve_cls("multilinear_adaptive_cpu_interpolator")
LIN_CLS_NAME = _resolve_cls("linear_adaptive_cpu_interpolator")
MultilinearAdaptiveCls = getattr(_itor_module, ML_CLS_NAME)
LinearAdaptiveCls = getattr(_itor_module, LIN_CLS_NAME)


class LinearEvaluator(operator_set_evaluator_iface):
    """f_k(x) = A[k, :n_dim] . x + A[k, n_dim] — exactly linear, so multilinear
    interpolation reproduces it bit-perfectly for any state."""

    def __init__(self, n_dim: int, n_ops: int, seed: int = 0):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops
        rng = np.random.default_rng(seed)
        self.A = rng.uniform(-1.0, 1.0, size=(n_ops, n_dim + 1))
        self.n_evals = 0

    def evaluate(self, state: value_vector, values: value_vector):
        s = np.asarray(state)
        v = np.asarray(values)
        v[:] = self.A[:, : self.n_dim] @ s + self.A[:, self.n_dim]
        self.n_evals += 1
        return 0

    def evaluate_batch(self, states, n_points, values, n_ops):
        s = np.asarray(states).reshape(n_points, self.n_dim)
        v = np.asarray(values).reshape(n_points, self.n_ops)
        v[:] = s @ self.A[:, : self.n_dim].T + self.A[:, self.n_dim]
        self.n_evals += n_points
        return 0

    def true_values(self, states_flat: np.ndarray) -> np.ndarray:
        s = np.asarray(states_flat).reshape(-1, self.n_dim)
        return (s @ self.A[:, : self.n_dim].T + self.A[:, self.n_dim]).reshape(-1)


@contextmanager
def captured_stderr():
    """Capture C-level stderr (printf via stdout/stderr is not seen by sys.stderr).
    This is best-effort: we redirect fd 2 to a temp buffer and yield the path."""
    saved = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)
    buf_bytes = bytearray()
    try:
        yield buf_bytes
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        # drain pipe (nonblocking)
        import fcntl

        fl = fcntl.fcntl(r, fcntl.F_GETFL)
        fcntl.fcntl(r, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        try:
            while True:
                chunk = os.read(r, 65536)
                if not chunk:
                    break
                buf_bytes.extend(chunk)
        except BlockingIOError:
            pass
        os.close(r)


def build_itor(axes_min, axes_max, n_points, kind="multilinear"):
    evaluator = LinearEvaluator(N_DIMS, N_OPS, seed=42)
    # The adaptive interpolators are unbounded: parametrized by (axes_origin, axes_step).
    # Derive them from the test's (axes_min, axes_max, n_points) window so the cell
    # spacing is unchanged; the cache simply grows past the window on demand.
    axes_origin = list(axes_min)
    axes_step = [(axes_max[i] - axes_min[i]) / (n_points - 1) for i in range(N_DIMS)]
    if kind == "multilinear":
        itor = MultilinearAdaptiveCls(
            evaluator,
            value_vector(axes_origin),
            value_vector(axes_step),
        )
    elif kind == "linear":
        # Last argument is use_barycentric_interpolation; keep False for the simplex path.
        itor = LinearAdaptiveCls(
            evaluator,
            value_vector(axes_origin),
            value_vector(axes_step),
            False,
        )
    else:
        raise ValueError(f"unknown kind: {kind}")
    itor.init()
    return itor, evaluator


def interpolate_batch(itor, evaluator, states_np: np.ndarray):
    n = states_np.shape[0]
    states = value_vector(states_np.flatten().astype(np.float64))
    values = value_vector(np.zeros(n * N_OPS))
    dvalues = value_vector(np.zeros(n * N_OPS * N_DIMS))
    idxs = index_vector(np.arange(n, dtype=np.int32))
    t0 = time.perf_counter()
    itor.evaluate_with_derivatives(states, idxs, values, dvalues)
    t1 = time.perf_counter()
    return np.asarray(values).copy(), np.asarray(dvalues).copy(), t1 - t0


def fmt(x):
    return f"{x:8.2f}" if abs(x) >= 0.01 or x == 0 else f"{x:8.2e}"


def run_one(kind: str):
    print("=" * 78)
    cls_name = ML_CLS_NAME if kind == "multilinear" else LIN_CLS_NAME
    print(f"Test: multi-index-keyed {kind} adaptive CPU interpolator")
    print(f"  N_DIMS={N_DIMS}, N_OPS={N_OPS}, class={cls_name}")
    print("=" * 78)

    # ── Build interpolator with a *narrow* prescribed window ────────────────
    axes_min = [0.0, 0.0, 0.0]
    axes_max = [1.0, 1.0, 1.0]
    n_points = 11  # 10 cells per axis → 1000 hypercubes total in the window
    itor, evaluator = build_itor(axes_min, axes_max, n_points, kind=kind)
    print(
        f"\nPrescribed window: axes_min={axes_min}, axes_max={axes_max}, "
        f"n_points={n_points} (axes_step={(axes_max[0] - axes_min[0]) / (n_points - 1):.3f})"
    )

    has_cached_hypercubes = hasattr(itor, "get_n_cached_hypercubes")

    # ── (1) Correctness: states well inside the prescribed window ───────────
    print("\n[1] In-bounds queries — expect bit-exact (linear evaluator):")
    rng = np.random.default_rng(seed=1)
    n_in = 1024
    s_in = rng.uniform(low=axes_min, high=axes_max, size=(n_in, N_DIMS))

    n_pts_before = itor.get_n_cached_points()
    n_hc_before = itor.get_n_cached_hypercubes() if has_cached_hypercubes else 0

    vals, _, t = interpolate_batch(itor, evaluator, s_in)
    true_vals = evaluator.true_values(s_in.flatten())
    err = np.max(np.abs(vals - true_vals))

    n_pts_after = itor.get_n_cached_points()
    n_hc_after = itor.get_n_cached_hypercubes() if has_cached_hypercubes else 0

    print(
        f"    queries={n_in:5d}  max_abs_err={err:.3e}  time={t * 1e3:7.2f} ms  "
        f"({n_in / t / 1e6:6.2f} M ops/s)"
    )
    if has_cached_hypercubes:
        print(
            f"    cache grew: points {n_pts_before:5d} -> {n_pts_after:5d}, "
            f"hypercubes {n_hc_before:5d} -> {n_hc_after:5d}"
        )
    else:
        print(f"    cache grew: points {n_pts_before:5d} -> {n_pts_after:5d}")
    assert err < 1e-10, f"in-bounds interpolation not exact: {err}"

    # ── (2) Out-of-bounds: states past axes_max[0] and below axes_min[1] ────
    print("\n[2] Out-of-bounds queries — expect bit-exact extrapolation, no warnings:")
    n_out = 1024
    # Build states deliberately far outside the prescribed window in 2 axes.
    s_out = np.column_stack(
        [
            rng.uniform(1.0, 3.0, size=n_out),  # past axes_max[0]
            rng.uniform(-1.5, 0.0, size=n_out),  # below axes_min[1]
            rng.uniform(0.0, 1.0, size=n_out),  # in-bounds on axis 2
        ]
    )

    n_pts_before2 = itor.get_n_cached_points()
    n_hc_before2 = itor.get_n_cached_hypercubes() if has_cached_hypercubes else 0

    with captured_stderr() as stderr_buf:
        vals2, _, t2 = interpolate_batch(itor, evaluator, s_out)
    true_vals2 = evaluator.true_values(s_out.flatten())
    err2 = np.max(np.abs(vals2 - true_vals2))

    n_pts_after2 = itor.get_n_cached_points()
    n_hc_after2 = itor.get_n_cached_hypercubes() if has_cached_hypercubes else 0
    stderr_text = stderr_buf.decode("utf-8", errors="ignore")
    warning_count = stderr_text.count("Interpolation warning")

    print(
        f"    queries={n_out:5d}  max_abs_err={err2:.3e}  time={t2 * 1e3:7.2f} ms  "
        f"({n_out / t2 / 1e6:6.2f} M ops/s)"
    )
    if has_cached_hypercubes:
        print(
            f"    cache grew: points {n_pts_before2:5d} -> {n_pts_after2:5d}, "
            f"hypercubes {n_hc_before2:5d} -> {n_hc_after2:5d}"
        )
    else:
        print(f"    cache grew: points {n_pts_before2:5d} -> {n_pts_after2:5d}")
    print(f"    extrapolation warnings emitted: {warning_count}")
    if warning_count:
        # show first 200 chars
        print(f"      sample: {stderr_text[:200]!r}")

    assert err2 < 1e-10, f"out-of-bounds extrapolation not exact: {err2}"
    assert n_pts_after2 > n_pts_before2, (
        "expected cache to grow when querying outside the prescribed window"
    )
    assert warning_count == 0, (
        f"expected zero extrapolation warnings with unbounded multi-index keys; "
        f"got {warning_count}"
    )

    # ── (3) Warm-cache throughput: re-query the same states ─────────────────
    print("\n[3] Warm-cache replay throughput (no materialization):")
    s_warm = np.vstack([s_in, s_out])
    # First pass to fully warm
    interpolate_batch(itor, evaluator, s_warm)
    # Second pass — measured
    _, _, t_warm = interpolate_batch(itor, evaluator, s_warm)
    n_warm = s_warm.shape[0]
    print(
        f"    queries={n_warm:5d}  time={t_warm * 1e3:7.2f} ms  "
        f"({n_warm / t_warm / 1e6:6.2f} M ops/s)"
    )

    # ── (4) Larger-batch perf sweep, both in- and out-of-bounds ─────────────
    print("\n[4] Throughput sweep — warm cache:")
    print(f"    {'N':>9} {'in-bounds (M ops/s)':>22} {'mixed (M ops/s)':>20}")
    for N in [10_000, 100_000, 500_000]:
        s_a = rng.uniform(low=axes_min, high=axes_max, size=(N, N_DIMS))
        s_b = np.column_stack(
            [
                rng.uniform(-2.0, 3.0, size=N),
                rng.uniform(-2.0, 3.0, size=N),
                rng.uniform(-2.0, 3.0, size=N),
            ]
        )
        # warm
        interpolate_batch(itor, evaluator, s_a)
        interpolate_batch(itor, evaluator, s_b)
        # measure
        _, _, t_a = interpolate_batch(itor, evaluator, s_a)
        _, _, t_b = interpolate_batch(itor, evaluator, s_b)
        print(f"    {N:>9d} {N / t_a / 1e6:>22.2f} {N / t_b / 1e6:>20.2f}")

    if has_cached_hypercubes:
        print(
            f"\n    final cache size: {itor.get_n_cached_points():>8d} points, "
            f"{itor.get_n_cached_hypercubes():>8d} hypercubes"
        )
    else:
        print(f"\n    final cache size: {itor.get_n_cached_points():>8d} points")

    # ── (5) Pickle round-trip ─────────────────────────────────────────────
    # The adaptive interpolator is unbounded: the canonical cache format is the
    # tuple-keyed point_data_full export. (The legacy integer-keyed point_data view
    # was removed along with axes_points-based mixed-radix packing.)
    n_total_cache = itor.get_n_cached_points()

    # Full tuple-keyed export (preserves out-of-window cells):
    print("\n[5b] Pickle round-trip of point_data_full (tuple-keyed):")
    pdf = itor.point_data_full
    assert isinstance(pdf, dict)
    n_full = len(pdf)
    n_oow = sum(1 for k in pdf.keys() if any(c < 0 or c >= n_points for c in k))
    print(
        f"    exported {n_full} points total, {n_oow} of which are out-of-window "
        f"(total cache: {n_total_cache})"
    )
    assert n_full == n_total_cache, (
        f"point_data_full should expose every cached cell ({n_total_cache}), got {n_full}"
    )
    assert n_oow > 0, "expected some out-of-window cells in the cache from step (2)"

    blob_full = pickle.dumps(pdf)
    pdf2 = pickle.loads(blob_full)
    assert set(pdf2.keys()) == set(pdf.keys())

    itor3, _ = build_itor(axes_min, axes_max, n_points, kind=kind)
    itor3.point_data_full = pdf2
    assert itor3.get_n_cached_points() == n_full, (
        f"point_data_full setter restored {itor3.get_n_cached_points()} of {n_full} cells"
    )
    print(
        f"    full round-trip OK: pickle size = {len(blob_full) / 1024:.1f} KiB "
        f"(preserves {n_oow} out-of-window cells the legacy view dropped)"
    )


def run_axes_step_helper_test():
    """(axes_origin, axes_step)-only construction (no axes_min / axes_max / n_points).

    With multi-index keys the adaptive interpolator is unbounded: its constructor
    takes exactly (evaluator, axes_origin, axes_step) — there is no point count or
    upper bound. The cache grows on demand wherever the solver lands.
    """
    print("=" * 78)
    print(
        "Test: axes_step-only direct construction (no axes_min / axes_max in user code)"
    )
    print("=" * 78)

    axes_step = [0.1, 0.1, 0.1]
    print(
        f"\nConstructing multilinear adaptive itor with axes_step={axes_step} only..."
    )

    origin = [0.0] * N_DIMS
    evaluator = LinearEvaluator(N_DIMS, N_OPS, seed=42)
    itor_ml = MultilinearAdaptiveCls(
        evaluator,
        value_vector(origin),
        value_vector(axes_step),
    )
    itor_ml.init()

    # Query at a mix of in-bounds and far-out-of-bounds states; the unbounded cache
    # materializes cells on demand far past any nominal window.
    rng = np.random.default_rng(seed=7)
    n_q = 4096
    s = rng.uniform(low=-500.0, high=500.0, size=(n_q, N_DIMS))
    with captured_stderr() as buf:
        vals, _, t = interpolate_batch(itor_ml, evaluator, s)
    true_vals = evaluator.true_values(s.flatten())
    err = np.max(np.abs(vals - true_vals))
    warn = buf.decode("utf-8", errors="ignore").count("Interpolation warning")
    print(
        f"    multilinear: queries={n_q}, max_abs_err={err:.3e}, "
        f"time={t * 1e3:.2f} ms, warnings={warn}"
    )
    assert err < 1e-9, f"multilinear correctness fail: {err}"
    assert warn == 0, f"unexpected extrapolation warnings: {warn}"

    print("\nConstructing linear adaptive itor with axes_step only...")
    evaluator2 = LinearEvaluator(N_DIMS, N_OPS, seed=42)
    itor_l = LinearAdaptiveCls(
        evaluator2,
        value_vector(origin),
        value_vector(axes_step),
        False,
    )
    itor_l.init()

    s_l = rng.uniform(low=-50.0, high=50.0, size=(2048, N_DIMS))
    with captured_stderr() as buf:
        vals_l, _, t_l = interpolate_batch(itor_l, evaluator2, s_l)
    true_vals_l = evaluator2.true_values(s_l.flatten())
    err_l = np.max(np.abs(vals_l - true_vals_l))
    warn_l = buf.decode("utf-8", errors="ignore").count("Interpolation warning")
    print(
        f"    linear:      queries=2048, max_abs_err={err_l:.3e}, "
        f"time={t_l * 1e3:.2f} ms, warnings={warn_l}"
    )
    assert err_l < 1e-9, f"linear correctness fail: {err_l}"
    assert warn_l == 0, f"unexpected extrapolation warnings: {warn_l}"

    print(
        "\n    axes_step helper OK — interpolators constructed without specifying bounds."
    )


@pytest.mark.parametrize("kind", ["multilinear", "linear"])
def test_adaptive_unbounded(kind):
    run_one(kind)


def test_axes_step_helper():
    run_axes_step_helper_test()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
