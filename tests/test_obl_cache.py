"""
OBL supporting-point cache codec tests, end to end through a REAL compiled adaptive
interpolator: write the mmap-arena base, append deltas, reload via mmap + delta merge,
recover epochs, recover from an ABI/placement-hash mismatch, and recompact. Skips if the
arena-capable .so isn't built.

Run:  PYTHONPATH=<repo> python -m pytest tests/test_obl_cache.py -q
"""

import os

import numpy as np
import pytest

from darts.physics.base.history_extension import HistoryStateSupport
from darts.physics.base.physics import PhysicsBase
from darts.tools.obl_cache import OblCacheCodec

ND, NO = 8, 8


def _itor_cls():
    import darts.interpolators as it

    # Exposed names carry neither the "adaptive" token (static interpolation is gone)
    # nor an index-type letter (that template parameter was dropped).
    cls = getattr(it, f"multilinear_cpu_interpolator_d_{ND}_{NO}", None)
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


def _new_physics(itor, path):
    p = PhysicsBase.__new__(PhysicsBase)
    p.created_itors = [(itor, str(path))]
    p._last_flushed_sizes = {}
    p._flushed_point_keys = {}
    p.cache = True
    p._cache_owner_pid = os.getpid()
    return p


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


def _cache_physics(tmp_path):
    """A PhysicsBase carrying just enough state to drive create_interpolator()."""
    from darts.engines import timer_node

    p = PhysicsBase.__new__(PhysicsBase)
    p.axes_step = [1.0] * ND
    p.axes_origin = [0.0] * ND
    # The default "no history state" case (has_history False).
    p.history = HistoryStateSupport([])
    p.cache = True
    p.cache_dir = str(tmp_path)
    p.created_itors = []
    p._last_flushed_sizes = {}
    p._flushed_point_keys = {}
    p._cache_owner_pid = os.getpid()
    p.timer = timer_node()
    return p


def _signature_name(evaluator, tmp_path, itor_token):
    """Reproduce create_interpolator's cache file name for one identity token.

    ``itor_token`` is ``'_'`` for the current signature and ``'_adaptive_'`` for the one
    written by versions that still carried the token in the interpolator names.
    """
    import hashlib

    signature = f"{type(evaluator).__name__}{itor_token}d_{ND:d}_{NO:d}_"
    for _ in range(ND):
        signature += f"_origin={0.0:e}_step={1.0:e}"
    signature += "_fmtv2"
    md5 = hashlib.md5(signature.encode()).hexdigest()
    return os.path.join(str(tmp_path), "obl_point_data_" + md5 + ".pkl")


def test_cache_filename_drops_adaptive_token(tmp_path):
    """New caches are written under the simplified (no '_adaptive_') signature."""
    _itor_cls()  # skip unless the (ND, NO) template is built
    ev = _make_evaluator()
    p = _cache_physics(tmp_path)
    itor, _ = p.create_interpolator(ev, "test_itor", NO)

    fname = p.created_itors[-1][1]
    assert fname == _signature_name(ev, tmp_path, "_")
    assert fname != _signature_name(ev, tmp_path, "_adaptive_")

    _materialize(itor, np.random.default_rng(3), 150)
    p.write_cache()
    assert os.path.exists(fname)


def test_legacy_cache_signature_is_recognized(tmp_path):
    """A cache left by an older version keeps being used, in place, for this run."""
    _itor_cls()  # skip unless the (ND, NO) template is built
    ev = _make_evaluator()
    rng = np.random.default_rng(4)

    # Write a cache, then rename it to the file name the old '_adaptive_' signature
    # would have hashed to -- i.e. a cache written before the token was dropped.
    p = _cache_physics(tmp_path)
    itor, _ = p.create_interpolator(ev, "test_itor", NO)
    _materialize(itor, rng, 150)
    p.write_cache()
    expected = _pd(itor)
    legacy_name = _signature_name(ev, tmp_path, "_adaptive_")
    os.rename(p.created_itors[-1][1], legacy_name)

    # A fresh run picks the legacy file up and keeps writing to it (not to a second
    # file under the new name).
    p2 = _cache_physics(tmp_path)
    itor2, _ = p2.create_interpolator(ev, "test_itor", NO)
    assert p2.created_itors[-1][1] == legacy_name
    assert _pd(itor2) == expected
    assert not os.path.exists(_signature_name(ev, tmp_path, "_"))

    # With a cache under the current name present, that one wins.
    p3 = _cache_physics(tmp_path)
    itor3, _ = p3.create_interpolator(ev, "test_itor", NO)
    _materialize(itor3, rng, 50)
    p3.created_itors[-1] = (itor3, _signature_name(ev, tmp_path, "_"))
    p3.write_cache()
    p4 = _cache_physics(tmp_path)
    p4.create_interpolator(ev, "test_itor", NO)
    assert p4.created_itors[-1][1] == _signature_name(ev, tmp_path, "_")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
