"""
OBL supporting-point cache codec tests, end to end through a REAL compiled adaptive
interpolator: write the mmap-arena base, append deltas, reload via mmap + delta merge,
recover epochs, recover from an ABI/placement-hash mismatch, and recompact. Skips if the
arena-capable .so isn't built.

Run:  PYTHONPATH=<repo> python -m pytest tests/test_obl_cache.py -q
"""

import multiprocessing
import os

import numpy as np
import pytest

from darts.physics.base.physics import PhysicsBase
from darts.tools.obl_cache import OblCacheCodec

ND, NO = 8, 8


def _itor_cls():
    import darts.interpolators as it

    # Letterless naming (index-type template parameter dropped); legacy _l_ name
    # kept as fallback for older compiled modules.
    cls = getattr(it, f"multilinear_adaptive_cpu_interpolator_d_{ND}_{NO}", None)
    if cls is None:
        cls = getattr(it, f"multilinear_adaptive_cpu_interpolator_l_d_{ND}_{NO}", None)
    if cls is None or not hasattr(cls, "build_arena_file"):
        pytest.skip("arena-capable interpolator template not built")
    return cls


def _make_evaluator():
    from darts.engines import operator_set_evaluator_iface

    class LinEval(operator_set_evaluator_iface):
        def __init__(self):
            super().__init__()
            rng = np.random.default_rng(7)
            self.A = rng.uniform(-1, 1, size=(NO, ND + 1))

        def evaluate(self, state, values):
            np.asarray(values)[:] = self.A[:, :ND] @ np.asarray(state) + self.A[:, ND]
            return 0

        def evaluate_batch(self, states, n, values, nops):
            x = np.asarray(states).reshape(n, ND)
            np.asarray(values).reshape(n, NO)[:] = x @ self.A[:, :ND].T + self.A[:, ND]
            return 0

    return LinEval()


def _new_itor(cls, ev):
    from darts.engines import value_vector

    itor = cls(ev, value_vector([0.0] * ND), value_vector([1.0] * ND))
    itor.init()
    return itor


def _materialize(itor, rng, count):
    from darts.engines import index_vector, value_vector

    s = rng.uniform(-15, 15, size=(count, ND))
    itor.evaluate_with_derivatives(
        value_vector(s.flatten()),
        index_vector(np.arange(count, dtype=np.int32)),
        value_vector(np.zeros(count * NO)),
        value_vector(np.zeros(count * NO * ND)),
    )


def _pd(itor):
    k, v = itor.get_point_data_arrays()
    k = np.asarray(k)
    v = np.asarray(v)
    return {
        tuple(int(x) for x in k[i]): tuple(float(x) for x in v[i])
        for i in range(k.shape[0])
    }


def _new_physics(itor, path, live=False):
    p = PhysicsBase.__new__(PhysicsBase)
    p.created_itors = [(itor, str(path))]
    p._last_flushed_sizes = {}
    p._flushed_point_keys = {}
    p._cache_read_offsets = {}
    p._cache_read_inodes = {}
    p.cache = True
    p.cache_live_reload = live
    p._cache_owner_pid = os.getpid()
    if os.path.exists(path):
        p._cache_read_offsets[id(itor)] = os.path.getsize(path)
        p._cache_read_inodes[id(itor)] = os.stat(path).st_ino
    return p


def _locked_append_worker(path, first_key):
    codec = OblCacheCodec()
    keys = np.arange(first_key, first_key + ND, dtype=np.int32).reshape(1, ND)
    vals = np.full((1, NO), float(first_key), dtype=np.float64)
    with codec.cache_lock(path, exclusive=True):
        codec._append_frame(path, codec._KIND_DELTA, keys, vals)


@pytest.mark.parametrize("exclusive", [False, True])
def test_cache_lock_roundtrip(tmp_path, exclusive):
    path = str(tmp_path / "obl_point_data_test.pkl")

    with OblCacheCodec().cache_lock(path, exclusive=exclusive):
        assert os.path.exists(path + ".lock")


def test_cache_write_reload_roundtrip(tmp_path):
    cls = _itor_cls()
    ev = _make_evaluator()
    itor = _new_itor(cls, ev)
    rng = np.random.default_rng(1)
    _materialize(itor, rng, 200)
    path = tmp_path / "obl_point_data_test.pkl"
    p = _new_physics(itor, path)

    p.write_cache()  # arena base
    assert p._cache_codec._is_cache(str(path))
    base = _pd(itor)

    _materialize(itor, rng, 200)  # new points -> dirty
    p.write_cache()  # delta append
    assert p._cache_codec._is_cache(str(path))
    expected = _pd(itor)
    assert len(expected) > len(base)

    # reload into a fresh interpolator via the load dispatch (mmap arena + delta merge)
    itor2 = _new_itor(cls, ev)
    n = p._load_cache(itor2, str(path))
    assert itor2.has_arena()
    assert n == len(expected)
    assert _pd(itor2) == expected


def test_cache_duplicate_delta_shadow_keeps_exact_size(tmp_path):
    cls = _itor_cls()
    ev = _make_evaluator()
    itor = _new_itor(cls, ev)
    rng = np.random.default_rng(21)
    _materialize(itor, rng, 80)
    path = tmp_path / "obl_point_data_test.pkl"
    p = _new_physics(itor, path)
    p.write_cache()  # arena base
    expected = _pd(itor)

    itor2 = _new_itor(cls, ev)
    n = p._load_cache(itor2, str(path))
    assert itor2.has_arena()
    assert n == len(expected)

    duplicate_key = next(iter(expected))
    replacement = np.arange(NO, dtype=np.float64) + 123.0
    itor2.add_point_data_arrays(
        np.array([duplicate_key], dtype=np.int32),
        replacement.reshape(1, NO),
    )

    assert itor2.point_data_size() == len(expected)
    shadowed = _pd(itor2)
    assert len(shadowed) == len(expected)
    assert shadowed[duplicate_key] == tuple(float(x) for x in replacement)


def test_cache_epochs_roundtrip(tmp_path):
    cls = _itor_cls()
    ev = _make_evaluator()
    itor = _new_itor(cls, ev)
    rng = np.random.default_rng(2)
    _materialize(itor, rng, 120)
    path = tmp_path / "obl_point_data_test.pkl"
    p = _new_physics(itor, path)
    p.write_cache()  # base + epoch frame
    _materialize(itor, rng, 80)
    p.write_cache()  # delta + epoch frame

    eps = PhysicsBase.load_point_epochs(str(path))
    full = _pd(itor)
    assert len(eps) > 0 and set(eps).issubset(set(full))


def test_cache_compaction(tmp_path, monkeypatch):
    cls = _itor_cls()
    ev = _make_evaluator()
    itor = _new_itor(cls, ev)
    rng = np.random.default_rng(4)
    _materialize(itor, rng, 100)
    path = tmp_path / "obl_point_data_test.pkl"
    p = _new_physics(itor, path)
    # Tiny compaction threshold so a small delta tail triggers a rebuild.
    monkeypatch.setattr(OblCacheCodec, "_COMPACT_TRAILING_BYTES", 1)
    p.write_cache()  # base
    arena_end_1 = p._cache_codec._read_header(str(path))["arena_end"]

    _materialize(itor, rng, 100)
    p.write_cache()  # delta append -> trailing > 1 byte -> compaction rebuilds arena
    meta = p._cache_codec._read_header(str(path))
    expected = _pd(itor)
    # after compaction the arena holds the full union; the DELTA tail is gone, but preserved
    # EPOCH frames may still trail the arena (epochs survive compaction).
    assert meta["arena_count"] == len(expected)
    assert os.path.getsize(str(path)) >= meta["arena_end"]
    assert meta["arena_end"] >= arena_end_1

    itor2 = _new_itor(cls, ev)
    n = p._load_cache(itor2, str(path))
    assert n == len(expected) and _pd(itor2) == expected


def test_cache_interpolation_equivalence(tmp_path):
    """Interpolation through a reloaded (mmap'd) arena must equal the original."""
    from darts.engines import index_vector, value_vector

    cls = _itor_cls()
    ev = _make_evaluator()
    itor = _new_itor(cls, ev)
    rng = np.random.default_rng(5)
    _materialize(itor, rng, 300)
    path = tmp_path / "obl_point_data_test.pkl"
    p = _new_physics(itor, path)
    p.write_cache()

    itor2 = _new_itor(cls, ev)
    p._load_cache(itor2, str(path))
    assert itor2.has_arena()

    q = rng.uniform(-12, 12, size=(60, ND))
    idx = index_vector(np.arange(60, dtype=np.int32))
    o0, d0 = value_vector(np.zeros(60 * NO)), value_vector(np.zeros(60 * NO * ND))
    o1, d1 = value_vector(np.zeros(60 * NO)), value_vector(np.zeros(60 * NO * ND))
    itor.evaluate_with_derivatives(value_vector(q.flatten()), idx, o0, d0)
    itor2.evaluate_with_derivatives(value_vector(q.flatten()), idx, o1, d1)
    assert np.allclose(np.asarray(o0), np.asarray(o1))
    assert np.allclose(np.asarray(d0), np.asarray(d1))


def test_cache_abi_mismatch_recovery(tmp_path):
    """An arena written with a DIFFERENT placement hash_id (simulating a rebuilt binary
    whose hash/PLACEMENT_VER changed) must be RECOVERED via the occupied-slot scan -- never
    crash (the NumPy-2 uint64>>int64 regression) and never silently regenerate. Then a flush
    must rewrite it with this binary's hash_id so the next load mmaps cleanly."""
    cls = _itor_cls()
    ev = _make_evaluator()
    itor = _new_itor(cls, ev)
    rng = np.random.default_rng(11)
    _materialize(itor, rng, 250)
    expected = _pd(itor)
    path = tmp_path / "obl_point_data_test.pkl"

    # Write the arena with a deliberately WRONG placement hash_id.
    wrong_hash = (itor.obl_arena_hash_id() ^ 0xABCDEF) & ((1 << 64) - 1)
    p = _new_physics(itor, path)
    p._cache_codec._write_base(itor, str(path), wrong_hash, None, None)
    assert p._cache_codec._read_header(str(path))["hash_id"] == wrong_hash

    # Reload: mismatch -> placement-independent occupied-slot scan into the overlay.
    itor2 = _new_itor(cls, ev)
    pl = _new_physics(itor2, path)
    n = pl._load_cache(itor2, str(path))
    assert n == len(expected), "ABI-mismatch recovery lost/regenerated points"
    assert not itor2.has_arena()  # scan path -> overlay, no mmap
    assert _pd(itor2) == expected
    assert id(itor2) in getattr(pl, "_force_recompact", set())

    # A flush rewrites the arena with the correct hash_id (forced recompaction)...
    pl.write_cache()
    assert (
        pl._cache_codec._read_header(str(path))["hash_id"] == itor2.obl_arena_hash_id()
    )
    # ...so a fresh load now mmaps in place.
    itor3 = _new_itor(cls, ev)
    p3 = _new_physics(itor3, path)
    n3 = p3._load_cache(itor3, str(path))
    assert itor3.has_arena() and n3 == len(expected) and _pd(itor3) == expected


def test_cache_epochs_survive_compaction(tmp_path, monkeypatch):
    """Historical epochs must survive an arena recompaction (they are carried over as raw
    EPOCH frames), not be silently discarded."""
    cls = _itor_cls()
    ev = _make_evaluator()
    itor = _new_itor(cls, ev)
    rng = np.random.default_rng(12)
    path = tmp_path / "obl_point_data_test.pkl"
    p = _new_physics(itor, path)
    _materialize(itor, rng, 120)
    p.write_cache()  # base + epoch frame
    eps_before = PhysicsBase.load_point_epochs(str(path))
    assert len(eps_before) > 0

    monkeypatch.setattr(OblCacheCodec, "_COMPACT_TRAILING_BYTES", 1)  # force compaction
    _materialize(itor, rng, 80)
    p.write_cache()  # delta + epoch frame, then compaction rebuild

    eps_after = PhysicsBase.load_point_epochs(str(path))
    # every pre-compaction epoch is still recoverable, plus the new ones
    assert set(eps_before).issubset(set(eps_after))
    assert len(eps_after) >= len(eps_before)
    # and the point data is intact
    itor2 = _new_itor(cls, ev)
    p._load_cache(itor2, str(path))
    assert _pd(itor2) == _pd(itor)


def test_live_reload_shares_peer_deltas(tmp_path):
    cls = _itor_cls()
    ev = _make_evaluator()
    rng = np.random.default_rng(31)
    path = tmp_path / "obl_point_data_test.pkl"

    itor1 = _new_itor(cls, ev)
    _materialize(itor1, rng, 100)
    p1 = _new_physics(itor1, path, live=True)
    p1.write_cache()
    p1._cache_read_offsets[id(itor1)] = os.path.getsize(path)
    p1._cache_read_inodes[id(itor1)] = os.stat(path).st_ino

    itor2 = _new_itor(cls, ev)
    p2 = _new_physics(itor2, path, live=True)
    assert p2._load_cache(itor2, str(path)) == itor1.point_data_size()
    itor2.clear_point_data_delta()
    p2._last_flushed_sizes[id(itor2)] = itor2.point_data_size()

    _materialize(itor1, rng, 100)
    p1.write_cache()
    imported = p2.reload_cache_deltas()

    assert imported > 0
    assert _pd(itor2) == _pd(itor1)
    dkeys, _ = p2._cache_codec._point_data_delta_arrays(itor2)
    assert dkeys is None or len(dkeys) == 0


def test_interprocess_locked_appends_are_complete(tmp_path):
    cls = _itor_cls()
    ev = _make_evaluator()
    path = tmp_path / "obl_point_data_test.pkl"
    itor = _new_itor(cls, ev)
    _materialize(itor, np.random.default_rng(32), 20)
    p = _new_physics(itor, path)
    p.write_cache()

    workers = [
        multiprocessing.Process(
            target=_locked_append_worker, args=(str(path), 1000 + i * ND)
        )
        for i in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert worker.exitcode == 0

    itor2 = _new_itor(cls, ev)
    assert p._load_cache(itor2, str(path)) == itor.point_data_size() + 4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
