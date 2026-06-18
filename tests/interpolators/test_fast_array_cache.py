"""
Correctness test for the bulk array cache I/O on the adaptive interpolators
(get_point_data_arrays / set_point_data_arrays) and the .keys.npy/.vals.npy
snapshot round-trip used by PhysicsBase for fast OBL cache reads.

Verifies that exporting the cache to (keys, vals) numpy arrays and re-importing
them — directly and through np.save/np.load(mmap) — reproduces the exact same
point_data_full as the canonical tuple-keyed pickle path, including negative
(out-of-window) multi-index keys.

Run with the chemistry conda env:
    python tests/interpolators/test_fast_array_cache.py
"""

import os
import tempfile

import numpy as np
import pytest

import darts.interpolators as _itor_module
from darts.engines import index_vector, value_vector
from darts.interpolators import operator_set_evaluator_iface

N_DIMS = 3
N_OPS = 4


def _resolve_cls(base):
    for tag in ("l", "i"):
        name = f"{base}_{tag}_d_{N_DIMS}_{N_OPS}"
        if hasattr(_itor_module, name):
            return getattr(_itor_module, name)
    raise SystemExit(
        f"No {base}_[l|i]_d_{N_DIMS}_{N_OPS} template exposed — rebuild with this "
        f"(n_dims, n_ops) pair."
    )


MultilinearAdaptiveCls = _resolve_cls("multilinear_adaptive_cpu_interpolator")
LinearAdaptiveCls = _resolve_cls("linear_adaptive_cpu_interpolator")


class LinearEvaluator(operator_set_evaluator_iface):
    """Exactly-linear operators so multilinear interpolation is bit-exact."""

    def __init__(self, n_dim: int, n_ops: int, seed: int = 0):
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

    def evaluate_batch(self, states, n_points, values, n_ops):
        s = np.asarray(states).reshape(n_points, self.n_dim)
        v = np.asarray(values).reshape(n_points, self.n_ops)
        v[:] = s @ self.A[:, : self.n_dim].T + self.A[:, self.n_dim]
        return 0


def _build(kind):
    evaluator = LinearEvaluator(N_DIMS, N_OPS, seed=42)
    origin = value_vector([0.0] * N_DIMS)
    step = value_vector([0.1] * N_DIMS)
    if kind == "multilinear":
        itor = MultilinearAdaptiveCls(evaluator, origin, step)
    else:
        itor = LinearAdaptiveCls(evaluator, origin, step, False)
    itor.init()
    return itor, evaluator


def _populate(itor, evaluator):
    """Evaluate a mix of in- and out-of-window states so the cache contains both
    positive and negative multi-index keys."""
    rng = np.random.default_rng(7)
    s = rng.uniform(low=-30.0, high=30.0, size=(4096, N_DIMS))
    states = value_vector(s.flatten().astype(np.float64))
    values = value_vector(np.zeros(s.shape[0] * N_OPS))
    dvalues = value_vector(np.zeros(s.shape[0] * N_OPS * N_DIMS))
    idxs = index_vector(np.arange(s.shape[0], dtype=np.int32))
    itor.evaluate_with_derivatives(states, idxs, values, dvalues)


@pytest.mark.parametrize("kind", ["multilinear", "linear"])
def test_arrays_roundtrip_matches_point_data_full(kind):
    itor, ev = _build(kind)
    _populate(itor, ev)

    d0 = dict(itor.point_data_full)
    n = len(d0)
    assert n > 0
    # The populate step must have produced out-of-window (negative-index) cells,
    # exercising int32 signedness in the array path.
    assert any(any(c < 0 for c in k) for k in d0), "expected negative multi-index keys"

    keys, vals = itor.get_point_data_arrays()
    assert keys.dtype == np.int32 and keys.shape == (n, N_DIMS)
    assert vals.dtype == np.float64 and vals.shape == (n, N_OPS)

    # (1) Direct re-import into a fresh interpolator.
    itor2, _ = _build(kind)
    itor2.set_point_data_arrays(keys, vals)
    d1 = dict(itor2.point_data_full)
    assert d1 == d0, f"{kind}: direct array round-trip changed the cache"

    # (2) Round-trip through np.save / np.load(mmap) — the on-disk snapshot path.
    with tempfile.TemporaryDirectory() as tmp:
        kp = os.path.join(tmp, "k.npy")
        vp = os.path.join(tmp, "v.npy")
        np.save(kp, keys)
        np.save(vp, vals)
        k2 = np.load(kp)
        v2 = np.load(vp, mmap_mode="r")
        itor3, _ = _build(kind)
        itor3.set_point_data_arrays(k2, v2)
        d2 = dict(itor3.point_data_full)
    assert d2 == d0, f"{kind}: npy snapshot round-trip changed the cache"


@pytest.mark.parametrize("kind", ["multilinear", "linear"])
def test_set_empty_arrays(kind):
    itor, _ = _build(kind)
    keys = np.zeros((0, N_DIMS), dtype=np.int32)
    vals = np.zeros((0, N_OPS), dtype=np.float64)
    itor.set_point_data_arrays(keys, vals)
    assert itor.get_n_cached_points() == 0


def test_physics_base_fast_cache_plumbing(tmp_path):
    """Exercise the PhysicsBase fast-cache helpers end to end: write a snapshot,
    confirm freshness, load it into a fresh interpolator, and confirm a pickle that
    is newer than the snapshot invalidates it (so the loader falls back)."""
    from darts.physics.base.physics_base import PhysicsBase

    # Bare instance — the fast-cache helpers do not touch engine/__init__ state.
    pb = PhysicsBase.__new__(PhysicsBase)

    itor, ev = _build("multilinear")
    _populate(itor, ev)
    d0 = dict(itor.point_data_full)

    pkl_path = str(tmp_path / "obl_point_data_test.pkl")
    # A stand-in pickle so freshness comparison has something to compare against.
    with open(pkl_path, "wb") as fp:
        fp.write(b"x")

    pb._write_fast_cache(itor, pkl_path)
    keys_path, vals_path = pb._fast_cache_paths(pkl_path)
    assert os.path.exists(keys_path) and os.path.exists(vals_path)
    assert pb._fast_cache_fresh(pkl_path) is True

    itor2, _ = _build("multilinear")
    n = pb._try_load_fast_cache(itor2, pkl_path)
    assert n == len(d0)
    assert dict(itor2.point_data_full) == d0

    # Make the pickle newer than the snapshot -> snapshot is stale -> loader declines.
    future = os.path.getmtime(keys_path) + 100
    os.utime(pkl_path, (future, future))
    assert pb._fast_cache_fresh(pkl_path) is False
    itor3, _ = _build("multilinear")
    assert pb._try_load_fast_cache(itor3, pkl_path) is None


@pytest.mark.parametrize("kind", ["multilinear", "linear"])
def test_set_rejects_bad_shapes(kind):
    itor, _ = _build(kind)
    with pytest.raises(ValueError):
        itor.set_point_data_arrays(
            np.zeros((2, N_DIMS + 1), dtype=np.int32),
            np.zeros((2, N_OPS), dtype=np.float64),
        )
    with pytest.raises(ValueError):
        itor.set_point_data_arrays(
            np.zeros((2, N_DIMS), dtype=np.int32),
            np.zeros((3, N_OPS), dtype=np.float64),
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
