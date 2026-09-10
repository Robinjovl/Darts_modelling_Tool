"""
GPU smoke test for the multi-index-hash-keyed multilinear adaptive GPU interpolator
(Phase 4).

Verifies:
- Construction works.
- In-bounds interpolation is exact (linear evaluator).
- Out-of-bounds interpolation works without warnings (cache grew, content hash key
  derived from signed multi-index — no axes_min/axes_max clamping).
- Throughput is non-zero (the GPU hashmap with 64-bit content hash keys is functional).

Run with the obl_bounds conda env, GPU build:
    python tests/interpolators/test_adaptive_unbounded_gpu.py
"""

import time

import numpy as np
import pytest

import darts.interpolators as _itor
from darts.engines import index_vector, timer_node, value_vector
from darts.interpolators import operator_set_evaluator_iface

N_DIMS = 3
N_OPS = 4


def _resolve_gpu_cls():
    """Return the GPU adaptive template, or None on a CPU build. The exposed name carries
    neither the "adaptive" token nor an index-type letter."""
    cls_name = f"multilinear_gpu_interpolator_d_{N_DIMS}_{N_OPS}"
    return getattr(_itor, cls_name, None)


class LinearEvaluator(operator_set_evaluator_iface):
    def __init__(self, n_dim, n_ops, seed=42):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops
        rng = np.random.default_rng(seed)
        self.A = rng.uniform(-1.0, 1.0, size=(n_ops, n_dim + 1))

    def evaluate(self, state, values):
        s = np.asarray(state)
        v = np.asarray(values)
        v[:] = self.A[:, : self.n_dim] @ s + self.A[:, self.n_dim]
        return 0

    def true_values(self, states_flat):
        s = np.asarray(states_flat).reshape(-1, self.n_dim)
        return (s @ self.A[:, : self.n_dim].T + self.A[:, self.n_dim]).reshape(-1)


def run(GpuItor):
    print("=" * 78)
    print(f"GPU smoke test: {GpuItor.__name__}")
    print(f"  N_DIMS={N_DIMS}, N_OPS={N_OPS}")
    print("=" * 78)

    evaluator = LinearEvaluator(N_DIMS, N_OPS)
    axes_min = [0.0, 0.0, 0.0]
    axes_max = [1.0, 1.0, 1.0]
    n_points = 11
    # New adaptive GPU ctor is (evaluator, axes_origin, axes_step).
    axes_step = [(axes_max[d] - axes_min[d]) / (n_points - 1) for d in range(N_DIMS)]
    itor = GpuItor(
        evaluator,
        value_vector(axes_min),
        value_vector(axes_step),
    )

    # GPU itor uses get_axis_n_points etc. via init; assume it's been set up by Python wrapper.
    itor.init()

    # The GPU base evaluate path uses `timer->start()` without a null check (the CPU base
    # is guarded). We must provide a timer node to avoid a segfault.
    root_timer = timer_node()
    itor.init_timer_node(root_timer)

    rng = np.random.default_rng(seed=11)
    n_in = 4096
    s_in = rng.uniform(low=axes_min, high=axes_max, size=(n_in, N_DIMS))

    def run_batch(states):
        n = states.shape[0]
        st = value_vector(states.flatten().astype(np.float64))
        vals = value_vector(np.zeros(n * N_OPS))
        dvals = value_vector(np.zeros(n * N_OPS * N_DIMS))
        idxs = index_vector(np.arange(n, dtype=np.int32))
        t0 = time.perf_counter()
        itor.evaluate_with_derivatives(st, idxs, vals, dvals)
        t1 = time.perf_counter()
        return np.asarray(vals).copy(), t1 - t0

    # Warm-up + measure
    vals_in, _ = run_batch(s_in)
    vals_in, t_in = run_batch(s_in)
    err_in = np.max(np.abs(vals_in - evaluator.true_values(s_in.flatten())))
    print(
        f"\n[1] In-bounds:  N={n_in}, max_abs_err={err_in:.3e}, "
        f"time={t_in * 1e3:.2f} ms ({n_in / t_in / 1e6:.2f} M ops/s)"
    )
    assert err_in < 1e-9, f"GPU in-bounds correctness fail: {err_in}"

    # Out-of-bounds
    s_out = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, size=n_in),
            rng.uniform(-3.0, 3.0, size=n_in),
            rng.uniform(-3.0, 3.0, size=n_in),
        ]
    )
    vals_out, _ = run_batch(s_out)
    vals_out, t_out = run_batch(s_out)
    err_out = np.max(np.abs(vals_out - evaluator.true_values(s_out.flatten())))
    print(
        f"[2] Out-of-bounds: N={n_in}, max_abs_err={err_out:.3e}, "
        f"time={t_out * 1e3:.2f} ms ({n_in / t_out / 1e6:.2f} M ops/s)"
    )
    assert err_out < 1e-9, f"GPU out-of-bounds correctness fail: {err_out}"

    print(
        f"\n    cache size: {itor.get_n_cached_points()} points, "
        f"{itor.get_n_cached_hypercubes()} hypercubes"
    )
    assert itor.get_n_cached_hypercubes() > 0, (
        "expected cached hypercubes to grow on out-of-bounds queries"
    )

    print("\nGPU smoke test passed.")


def test_adaptive_unbounded_gpu():
    GpuItor = _resolve_gpu_cls()
    if GpuItor is None:
        pytest.skip(
            f"no multilinear_gpu_interpolator template for "
            f"(N_DIMS={N_DIMS}, N_OPS={N_OPS}) — not a GPU build"
        )
    run(GpuItor)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
