"""
Benchmark the new fast OBL cache read (numpy arrays + C++ set_point_data_arrays)
against the legacy pickle read, on a real on-disk obl_point_data_*.pkl.

What it does, given a pickle:
  1. (convert) Load the pickle exactly like PhysicsBase (base object + appended
     DELTA frames), timing it — this is the legacy read cost we are replacing.
     Infer (n_dims, n_ops), build the keys/vals arrays, and write the single
     ``.fastcache`` snapshot file next to the pickle (PhysicsBase format).
  2. (fast read) In a fresh interpolator of the matching (n_dims, n_ops) template,
     time reading the ``.fastcache`` (keys read, vals mmap) + set_point_data_arrays.
  3. Report sizes, per-phase timings, correctness (point count), and speedup.

Use --read-only to time just phase 2 against an already-written snapshot (run it
in a fresh shell for a page-cache-independent number).

Run with the chemistry conda env, e.g.:
    python tests/interpolators/bench_fast_array_cache.py \
        models/chemistry/carbonated_water/obl_point_data_<3GB-hash>.pkl
"""

import argparse
import os
import pickle
import time
import zlib

import numpy as np

import darts.interpolators as _itor_module
from darts.engines import value_vector
from darts.interpolators import operator_set_evaluator_iface
from darts.physics.base.physics_base import PhysicsBase


def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def load_pkl_full(path):
    """Replicate PhysicsBase._safe_pickle_load: base pickle + merged DELTA frames."""
    with open(path, "rb") as fp:
        data = pickle.load(fp)
        if hasattr(data, "update"):
            magic_len = len(PhysicsBase._OBL_DELTA_MAGIC)
            hdr = PhysicsBase._OBL_DELTA_HEADER
            while True:
                magic = fp.read(magic_len)
                if len(magic) != magic_len:
                    break
                if magic not in (
                    PhysicsBase._OBL_DELTA_MAGIC,
                    PhysicsBase._OBL_EPOCH_MAGIC,
                ):
                    break
                header = fp.read(hdr.size)
                if len(header) != hdr.size:
                    break
                payload_len, expected_crc = hdr.unpack(header)
                payload = fp.read(payload_len)
                if len(payload) != payload_len:
                    break
                if (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
                    break
                if magic == PhysicsBase._OBL_EPOCH_MAGIC:
                    continue  # metadata sidecar, not point data
                data.update(pickle.loads(payload))
    return data


class _DummyEvaluator(operator_set_evaluator_iface):
    """Never actually evaluated here — only needed to construct the interpolator."""

    def __init__(self, n_dim, n_ops):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops

    def evaluate(self, state, values):
        np.asarray(values)[:] = 0.0
        return 0


def build_itor(n_dims, n_ops, precision, kind):
    base = (
        "multilinear_adaptive_cpu_interpolator"
        if kind == "multilinear"
        else "linear_adaptive_cpu_interpolator"
    )
    cls = None
    for tag in ("l", "i"):
        name = f"{base}_{tag}_{precision}_{n_dims}_{n_ops}"
        cls = getattr(_itor_module, name, None)
        if cls is not None:
            break
    if cls is None:
        avail = sorted(
            a for a in dir(_itor_module) if a.startswith(base) and "_cpu_" in a
        )
        raise SystemExit(
            f"No {base}_[l|i]_{precision}_{n_dims}_{n_ops} template is compiled.\n"
            f"Available {kind} CPU templates:\n  " + "\n  ".join(avail)
        )
    ev = _DummyEvaluator(n_dims, n_ops)
    origin = value_vector([0.0] * n_dims)
    step = value_vector([1.0] * n_dims)
    if kind == "multilinear":
        return cls(ev, origin, step)
    return cls(ev, origin, step, False)


# Bare PhysicsBase instance just to reuse the single-file (.fastcache) reader/writer.
_PB = PhysicsBase.__new__(PhysicsBase)


def convert(pkl_path):
    print(
        f"[convert] loading pickle {pkl_path} ({human(os.path.getsize(pkl_path))}) ..."
    )
    t0 = time.perf_counter()
    d = load_pkl_full(pkl_path)
    t_pickle = time.perf_counter() - t0
    n = len(d)
    first_k = next(iter(d))
    n_dims = len(first_k)
    n_ops = len(d[first_k])
    print(
        f"[convert] pickle.load + merge: {t_pickle:8.2f} s   "
        f"N={n:,}  n_dims={n_dims}  n_ops={n_ops}"
    )

    t0 = time.perf_counter()
    keys = np.fromiter(
        (c for k in d for c in k), dtype=np.int32, count=n * n_dims
    ).reshape(n, n_dims)
    vals = np.fromiter(
        (c for v in d.values() for c in v), dtype=np.float64, count=n * n_ops
    ).reshape(n, n_ops)
    t_build = time.perf_counter() - t0

    fast_path = _PB._fast_cache_path(pkl_path)
    header = {
        "format": 1,
        "pkl_size": os.path.getsize(pkl_path),
        "n_points": n,
        "n_dims": n_dims,
        "n_ops": n_ops,
        "base_fp": _PB._pkl_fingerprint(
            pkl_path, min(65536, os.path.getsize(pkl_path))
        ),
    }
    t0 = time.perf_counter()
    _PB._write_fastcache_file(fast_path, header, keys, vals)
    t_save = time.perf_counter() - t0
    print(
        f"[convert] build arrays: {t_build:7.2f} s   save .fastcache: {t_save:7.2f} s   "
        f"file={human(os.path.getsize(fast_path))}"
    )
    return t_pickle, n, n_dims, n_ops, d


def fast_read(pkl_path, n_dims, n_ops, precision, kind, expected_n=None):
    itor = build_itor(n_dims, n_ops, precision, kind)
    t0 = time.perf_counter()
    keys, vals = _PB._read_fastcache_arrays(pkl_path)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    itor.set_point_data_arrays(keys, vals)
    t_set = time.perf_counter() - t0
    got = itor.get_n_cached_points()
    print(
        f"[fast]    np.load: {t_load:7.2f} s   set_point_data_arrays: {t_set:7.2f} s   "
        f"total: {t_load + t_set:7.2f} s   points={got:,}"
    )
    if expected_n is not None:
        assert got == expected_n, f"point count mismatch: {got} != {expected_n}"
    return t_load + t_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--kind", default="multilinear", choices=["multilinear", "linear"])
    ap.add_argument("--precision", default="d", choices=["d", "s"])
    ap.add_argument(
        "--read-only",
        action="store_true",
        help="skip convert; time fast read of existing snapshot",
    )
    ap.add_argument(
        "--time-old-ingest",
        action="store_true",
        help="also time the legacy point_data_full setter",
    )
    args = ap.parse_args()

    t_pickle = None
    n_dims = n_ops = expected_n = None
    d = None

    if args.read_only:
        meta = _PB._read_fast_meta(args.pkl)
        if meta is None:
            raise SystemExit(f"No .fastcache snapshot for {args.pkl}")
        expected_n, n_dims, n_ops = meta["n_points"], meta["n_dims"], meta["n_ops"]
        print(f"[read-only] snapshot: N={expected_n:,}  n_dims={n_dims}  n_ops={n_ops}")
    else:
        t_pickle, expected_n, n_dims, n_ops, d = convert(args.pkl)
        if not args.time_old_ingest:
            # Drop the dict before the fast-read phase so peak RAM is not
            # dict + arrays + C++ map all at once.
            import gc

            del d
            d = None
            gc.collect()

    print("-" * 70)
    t_fast = fast_read(args.pkl, n_dims, n_ops, args.precision, args.kind, expected_n)

    if args.time_old_ingest and d is not None:
        itor_old = build_itor(n_dims, n_ops, args.precision, args.kind)
        t0 = time.perf_counter()
        itor_old.point_data_full = d
        t_old_ingest = time.perf_counter() - t0
        print(f"[old]     point_data_full setter: {t_old_ingest:7.2f} s")

    print("-" * 70)
    if t_pickle is not None:
        print(
            f"SUMMARY: legacy pickle.load = {t_pickle:.2f} s   "
            f"new fast read = {t_fast:.2f} s   "
            f"speedup = {t_pickle / t_fast:.1f}x  (read phase only)"
        )
    else:
        print(f"SUMMARY: new fast read = {t_fast:.2f} s")


if __name__ == "__main__":
    main()
