"""
OBL adaptive-interpolator supporting-point cache codec (single mmap-arena format).

Extracted from PhysicsBase so the cache *file format* is a self-contained, stateless
interface, decoupled from the physics lifecycle. PhysicsBase composes an instance of
:class:`OblCacheCodec` and keeps only the orchestration (the write_cache loop over
created_itors, dirty-point tracking, SIGINT/SIGTERM flushing, atexit/__del__ finalize),
delegating every actual file read/write here.

The codec operates purely on (interpolator, path, numpy arrays):
  * interpolators expose get/set/add_point_data_arrays, point_data_delta_arrays,
    point_data_epoch_delta_arrays, point_data_size, clear_point_data_delta, build_arena_file,
    mmap_arena, obl_arena_hash_id, has_arena (the C++ adaptive interpolators);
  * paths are the obl_point_data_<md5>.pkl cache files;
  * no PhysicsBase state is referenced.

On-disk layout (one self-contained file, the ``obl_point_data_<md5>.pkl`` name kept for
cache discovery / signatures):

    magic _MAGIC | uint64 header_len | JSON header | ARENA | {FRAME}*

  * JSON header: {n_dims, n_ops, hash_id, arena_capacity, arena_count, bitmap_off, keys_off,
    vals_off, arena_end, page_size, val_dtype, ...}.
  * ARENA: a page-aligned, open-addressing hash arena built by the C++ point_data_store
    (occupancy bitmap | int32 keys | float64 vals). A reload mmaps it in place -- O(1), no
    ~1B-entry rebuild -- and the resident map is file-backed / demand-paged (no ~150 GB
    anonymous-RAM image). When the placement hash mismatches a rebuilt binary the occupied
    slots are still valid and are scanned into the overlay (and the arena recompacted).
  * Trailing FRAMEs record points materialized after the base was written:
        kind:uint8 (1 DELTA / 2 EPOCH) | n_rows:uint64 | crc32:uint32 | payload (SoA)
        DELTA : int32 keys[n_rows*n_dims]  then float64 vals[n_rows*n_ops]
        EPOCH : int32 keys[n_rows*n_dims]  then uint64  epochs[n_rows]
    Each frame's crc32 covers its payload so a torn final append is detected and ignored.
    On load, DELTA frames merge into the overlay; once the trailing region grows large the
    arena is recompacted to fold them in (see _COMPACT_TRAILING_BYTES).

STATIC (non-adaptive, bounded-grid) interpolators have no arena; their point_data is cached
as a plain pickle via :meth:`_atomic_pickle_dump` / :meth:`_safe_pickle_load`.

(The _MAGIC bytes are retained verbatim so cache files written by earlier builds still load.)
"""

import json
import os
import pickle
import struct
import tempfile
import zlib
from contextlib import contextmanager
from typing import Any

import numpy as np

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _lock_file_ex = _kernel32.LockFileEx
    _lock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _lock_file_ex.restype = wintypes.BOOL
    _unlock_file_ex = _kernel32.UnlockFileEx
    _unlock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _unlock_file_ex.restype = wintypes.BOOL
else:
    import fcntl


def _lock_cache_file(lock_fp, exclusive: bool):
    """Acquire a blocking one-byte advisory lock and return its platform token."""
    if os.name == "nt":
        overlapped = _Overlapped()
        flags = 0x00000002 if exclusive else 0  # LOCKFILE_EXCLUSIVE_LOCK
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(lock_fp.fileno()))
        if not _lock_file_ex(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())
        return overlapped

    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(lock_fp.fileno(), mode)
    return None


def _unlock_cache_file(lock_fp, token) -> None:
    """Release a lock acquired by :func:`_lock_cache_file`."""
    if os.name == "nt":
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(lock_fp.fileno()))
        if not _unlock_file_ex(handle, 0, 1, 0, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        return

    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


class OblCacheCodec:
    """Stateless codec for the OBL supporting-point cache file format."""

    # Trailing-frame codec (DELTA/EPOCH records appended after the arena base).
    _KIND_BASE = 0
    _KIND_DELTA = 1
    _KIND_EPOCH = 2
    _FRAME_HDR = struct.Struct('<BQI')  # kind, n_rows, crc32  (13 bytes)

    # 8-byte file magic. Retained verbatim for compatibility with already-written caches.
    _MAGIC = b'DRTSFC03'
    # Recompact (rebuild a single consolidated arena) once the trailing un-compacted DELTA
    # region exceeds max(this, arena_bytes/4), folding the deltas back into the arena.
    _COMPACT_TRAILING_BYTES = 1 << 30  # 1 GiB

    @contextmanager
    def cache_lock(self, path: str, exclusive: bool = True):
        """Serialize cache readers/writers across independent simulations."""
        lock_path = path + '.lock'
        os.makedirs(os.path.dirname(lock_path) or '.', exist_ok=True)
        with open(lock_path, 'a+b') as lock_fp:
            lock_token = _lock_cache_file(lock_fp, exclusive)
            try:
                yield
            finally:
                _unlock_cache_file(lock_fp, lock_token)

    @staticmethod
    def _point_data_size(itor) -> int:
        if hasattr(itor, 'point_data_size'):
            return itor.point_data_size()
        return len(itor.point_data)

    @staticmethod
    def _point_data_delta(itor) -> dict:
        # Native dirty-point tracking (the C++ adaptive interpolators) avoids copying the
        # large supporting-point map. A non-native itor exposes no dirty tracker, so the
        # codec -- which holds no cross-flush state of its own -- treats its entire
        # point_data as the delta (PhysicsBase owns any incremental-flush bookkeeping).
        if hasattr(itor, 'point_data_delta'):
            return dict(itor.point_data_delta())
        return dict(getattr(itor, 'point_data', {}))

    @staticmethod
    def _point_data_epoch_delta(itor) -> dict:
        # Evaluation epoch (batch-interpolation / nonlinear-iteration index) for each
        # point materialized since the last flush, keyed identically to the point delta.
        # Only the native adaptive interpolators track this; everything else returns {}.
        if hasattr(itor, 'point_data_epoch_delta'):
            try:
                return dict(itor.point_data_epoch_delta())
            except Exception:
                return {}
        return {}

    def _atomic_pickle_dump(self, obj: Any, final_path: str) -> None:
        """
        Atomically write a pickle file via a temporary path followed by ``os.replace``.

        Flush the data to disk before the rename to reduce cache corruption on interruption.
        """
        directory = os.path.dirname(final_path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass

        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(final_path) + ".tmp.", suffix=".pkl", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as fp:
                pickle.dump(obj, fp, protocol=4)
                fp.flush()
                try:
                    os.fsync(fp.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, final_path)
            # Best-effort directory fsync to persist the rename
            try:
                dir_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _is_cache(self, path: str) -> bool:
        """True iff ``path`` begins with the cache magic."""
        try:
            with open(path, 'rb') as fp:
                return fp.read(len(self._MAGIC)) == self._MAGIC
        except Exception:
            return False

    def _read_header(self, path: str):
        """Return the validated cache JSON header dict, or None. Validation guards the
        zero-copy mmap: little-endian, page-aligned region offsets, 8-aligned vals
        (double), and arena_end within the file."""
        try:
            with open(path, 'rb') as fp:
                if fp.read(len(self._MAGIC)) != self._MAGIC:
                    return None
                raw = fp.read(8)
                if len(raw) != 8:
                    return None
                (hlen,) = struct.unpack('<Q', raw)
                hj = fp.read(hlen)
                if len(hj) != hlen:
                    return None
                h = json.loads(hj.decode('utf-8'))
            req = (
                'n_dims',
                'n_ops',
                'hash_id',
                'arena_capacity',
                'arena_count',
                'bitmap_off',
                'keys_off',
                'vals_off',
                'arena_end',
            )
            if not isinstance(h, dict) or any(k not in h for k in req):
                return None
            if h.get('endian', 'L') != 'L':
                return None
            page = int(h.get('page_size', 4096))
            if int(h['vals_off']) % 8 != 0:
                return None
            for k in ('bitmap_off', 'keys_off', 'vals_off'):
                if int(h[k]) % page != 0:
                    return None
            if int(h['arena_end']) > os.path.getsize(path):
                return None
            return h
        except Exception:
            return None

    def _point_data_delta_arrays(self, itor):
        """Points materialized since the last flush as ``(keys int32[M,n_dims],
        vals float64[M,n_ops])``, or ``(None, None)``. Prefers a native contiguous
        C++ export (no per-point boxing) and falls back to the tuple-keyed dict."""
        if hasattr(itor, 'point_data_delta_arrays'):
            try:
                k, v = itor.point_data_delta_arrays()
                if k is None or len(k) == 0:
                    return None, None
                return np.ascontiguousarray(k, np.int32), np.ascontiguousarray(
                    v, np.float64
                )
            except Exception:
                pass
        delta = self._point_data_delta(itor)
        if not delta:
            return None, None
        keys = np.array(list(delta.keys()), dtype=np.int32)
        vals = np.array([list(v) for v in delta.values()], dtype=np.float64)
        return keys, vals

    def _point_data_epoch_delta_arrays(self, itor):
        """Epoch sidecar of the points materialized since the last flush as
        ``(keys int32[M,n_dims], epochs uint64[M])``, or ``(None, None)``."""
        if hasattr(itor, 'point_data_epoch_delta_arrays'):
            try:
                k, e = itor.point_data_epoch_delta_arrays()
                if k is None or len(k) == 0:
                    return None, None
                return np.ascontiguousarray(k, np.int32), np.ascontiguousarray(
                    e, np.uint64
                )
            except Exception:
                pass
        epochs = self._point_data_epoch_delta(itor)
        if not epochs:
            return None, None
        keys = np.array(list(epochs.keys()), dtype=np.int32)
        eps = np.array(list(epochs.values()), dtype=np.uint64)
        return keys, eps

    def _append_frame(self, final_path: str, kind: int, keys, second) -> None:
        """Append one checksummed trailing frame (DELTA or EPOCH) to an existing file.
        The length+crc let the loader ignore a torn final append."""
        keys = np.ascontiguousarray(keys, np.int32)
        if kind == self._KIND_EPOCH:
            second = np.ascontiguousarray(second, np.uint64)
        else:
            second = np.ascontiguousarray(second, np.float64)
        payload = keys.tobytes() + second.tobytes()
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        hdr = self._FRAME_HDR.pack(kind, int(keys.shape[0]), crc)
        with open(final_path, "ab") as fp:
            fp.write(hdr)
            fp.write(payload)
            fp.flush()
            try:
                os.fsync(fp.fileno())
            except Exception:
                pass

    def _extract_epoch_frames_bytes(self, path) -> bytes:
        """Return the raw bytes of every EPOCH frame trailing the arena (13-byte header
        + payload), concatenated, so they can be appended verbatim to a freshly written
        cache file -- this is how historical epochs survive an arena recompaction without
        parsing ~1e9 entries in Python. A torn final frame is ignored."""
        try:
            meta = self._read_header(path)
            if meta is None:
                return b''
            nd, no = int(meta['n_dims']), int(meta['n_ops'])
            start = int(meta['arena_end'])
            out = bytearray()
            filesize = os.path.getsize(path)
            hdr_size = self._FRAME_HDR.size
            offset = start
            with open(path, 'rb') as fp:
                while offset + hdr_size <= filesize:
                    fp.seek(offset)
                    fh = fp.read(hdr_size)
                    if len(fh) != hdr_size:
                        break
                    kind, n_rows, _crc = self._FRAME_HDR.unpack(fh)
                    n_rows = int(n_rows)
                    keys_bytes = n_rows * nd * 4
                    if kind == self._KIND_EPOCH:
                        payload_len = keys_bytes + n_rows * 8
                    else:
                        payload_len = keys_bytes + n_rows * no * 8
                    if offset + hdr_size + payload_len > filesize:
                        break  # torn final frame
                    if kind == self._KIND_EPOCH:
                        fp.seek(offset)
                        out += fp.read(hdr_size + payload_len)
                    offset += hdr_size + payload_len
            return bytes(out)
        except Exception:
            return b''

    def _write_base(
        self,
        itor,
        final_path,
        hash_id,
        epoch_keys=None,
        epoch_vals=None,
        preserve_epochs_from=None,
    ):
        """Build a fresh consolidated arena (C++ build_arena_file) to a temp file,
        carry over historical EPOCH frames from ``preserve_epochs_from`` (read BEFORE the
        atomic replace, so it may be ``final_path`` itself), append this flush's EPOCH
        frame, then atomically ``os.replace`` into place. Used for the first flush, compaction, and ABI-mismatch recompaction."""
        # Extract historical epochs from the OLD file first (it still exists pre-replace).
        preserved = b''
        if preserve_epochs_from is not None:
            preserved = self._extract_epoch_frames_bytes(preserve_epochs_from)
        directory = os.path.dirname(final_path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(final_path) + ".tmp.",
            suffix=".cache",
            dir=directory,
        )
        os.close(fd)
        try:
            itor.build_arena_file(tmp, int(hash_id))  # writes the complete arena
            if preserved:
                with open(tmp, 'ab') as fp:
                    fp.write(preserved)
                    fp.flush()
                    try:
                        os.fsync(fp.fileno())
                    except Exception:
                        pass
            if epoch_keys is not None and len(epoch_keys):
                self._append_frame(tmp, self._KIND_EPOCH, epoch_keys, epoch_vals)
            os.replace(tmp, final_path)
            try:
                dir_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _verify_arena(self, path, expected_rows):
        """True iff ``path`` is a valid cache file whose arena holds ``expected_rows``."""
        meta = self._read_header(path)
        return meta is not None and int(meta.get('arena_count', -1)) == int(
            expected_rows
        )

    def _maybe_compact(self, itor, filename):
        """Rebuild a single consolidated arena (arena ∪ overlay) once the trailing
        un-compacted DELTA region exceeds max(1 GiB, arena_bytes/4)."""
        meta = self._read_header(filename)
        if meta is None:
            return
        arena_end = int(meta['arena_end'])
        try:
            filesize = os.path.getsize(filename)
        except Exception:
            return
        trailing = filesize - arena_end
        threshold = max(self._COMPACT_TRAILING_BYTES, arena_end // 4)
        if trailing <= threshold:
            return
        try:
            hash_id = itor.obl_arena_hash_id()
        except Exception:
            return
        # The current flush's epochs are already appended to `filename` as an EPOCH frame,
        # so preserving the file's epoch frames carries the full history (this flush + all
        # earlier ones) across the rebuild; no separate current-epoch arg is needed.
        print("Compacting cache", filename, f'(trailing {trailing} > {threshold})')
        self._write_base(
            itor, filename, hash_id, None, None, preserve_epochs_from=filename
        )

    def _load(self, itor, path):
        """Load the cache: mmap the arena in place (O(1)) when the on-disk placement
        hash matches this binary, else recover by scanning occupied slots into the
        overlay; then merge any trailing DELTA frames into the overlay. Returns the point
        count, or None to fall back to another reader."""
        if not hasattr(itor, 'mmap_arena'):
            return (
                None  # interpolator built without arena support (e.g. older .so / GPU)
            )
        meta = self._read_header(path)
        if meta is None:
            return None
        nd, no = int(meta['n_dims']), int(meta['n_ops'])
        C, count = int(meta['arena_capacity']), int(meta['arena_count'])
        bo, ko, vo = (
            int(meta['bitmap_off']),
            int(meta['keys_off']),
            int(meta['vals_off']),
        )
        arena_end = int(meta['arena_end'])
        try:
            same_abi = int(meta['hash_id']) == int(itor.obl_arena_hash_id())
        except Exception:
            same_abi = False
        try:
            if same_abi:
                itor.mmap_arena(path, bo, ko, vo, C, count)
            else:
                # Placement hash / ABI changed: the arena's slot layout is unusable, but
                # the occupied slots still hold valid (key, val) pairs. Scan them into the
                # overlay (placement-independent), and FLAG this itor for a forced one-time
                # recompaction on the next flush -- otherwise a pure-replay run (no new
                # points materialized) would never rebuild the arena and would re-pay the
                # full O(N) scan + hold the whole cache in anonymous-RAM overlay every run.
                print(
                    "cache arena placement hash mismatch for",
                    path,
                    "- recovering occupied slots into overlay (forcing recompaction).",
                )
                self._scan_occupied_into_overlay(itor, path, meta)
        except Exception as err:
            print(
                "cache arena load failed for", path, "-", type(err).__name__, str(err)
            )
            return None
        # Merge trailing DELTA frames (past arena_end) into the overlay; EPOCH skipped.
        try:
            self._merge_trailing_frames(itor, path, arena_end, nd, no)
        except Exception as err:
            print(
                "cache tail-merge failed for", path, "-", type(err).__name__, str(err)
            )
            return None
        if hasattr(itor, 'point_data_size'):
            return itor.point_data_size()
        return count

    def _merge_trailing_frames(self, itor, path, start_offset, nd, no):
        """Walk the trailing frames appended after the arena and add DELTA frames to
        the overlay via add_point_data_arrays (crc-checked; torn final frame ignored)."""
        filesize = os.path.getsize(path)
        offset = start_offset
        imported = 0
        hdr_size = self._FRAME_HDR.size
        while offset + hdr_size <= filesize:
            with open(path, 'rb') as fp:
                fp.seek(offset)
                fhdr = fp.read(hdr_size)
            if len(fhdr) != hdr_size:
                break
            kind, n_rows, crc = self._FRAME_HDR.unpack(fhdr)
            n_rows = int(n_rows)
            payload_off = offset + hdr_size
            keys_bytes = n_rows * nd * 4
            if kind == self._KIND_EPOCH:
                epoch_len = keys_bytes + n_rows * 8
                if payload_off + epoch_len > filesize:
                    break  # torn final epoch frame
                offset = payload_off + epoch_len
                continue
            payload_len = keys_bytes + n_rows * no * 8
            if payload_off + payload_len > filesize:
                break  # torn final frame
            if n_rows == 0:
                offset = payload_off + payload_len
                continue
            keys = np.fromfile(
                path, dtype=np.int32, count=n_rows * nd, offset=payload_off
            ).reshape(n_rows, nd)
            vals = np.memmap(
                path,
                dtype=np.float64,
                mode='r',
                offset=payload_off + keys_bytes,
                shape=(n_rows, no),
            )
            if kind == self._KIND_DELTA and crc != 0:
                actual = zlib.crc32(keys.tobytes())
                actual = (
                    zlib.crc32(np.ascontiguousarray(vals).tobytes(), actual)
                    & 0xFFFFFFFF
                )
                if actual != crc:
                    del vals
                    break
            itor.add_point_data_arrays(keys, vals)
            if kind == self._KIND_DELTA:
                imported += n_rows
            del vals
            offset = payload_off + payload_len
        return offset, imported

    def merge_trailing_frames(self, itor, path, start_offset):
        """Import complete frames at or after start_offset incrementally.

        Returns (next_offset, imported_rows). An incomplete final frame leaves
        next_offset at its header so a later live reload can retry it.
        """
        meta = self._read_header(path)
        if meta is None:
            return start_offset, 0
        arena_end = int(meta['arena_end'])
        if start_offset < arena_end:
            start_offset = arena_end
        return self._merge_trailing_frames(
            itor, path, start_offset, int(meta['n_dims']), int(meta['n_ops'])
        )

    def _scan_occupied_into_overlay(self, itor, path, meta):
        """ABI-mismatch recovery: read the arena's occupied slots (bitmap-driven,
        placement-independent) in chunks and add them to the overlay. Memory-bounded."""
        nd, no = int(meta['n_dims']), int(meta['n_ops'])
        C = int(meta['arena_capacity'])
        bo, ko, vo = (
            int(meta['bitmap_off']),
            int(meta['keys_off']),
            int(meta['vals_off']),
        )
        occ = np.memmap(
            path, dtype=np.uint64, mode='r', offset=bo, shape=((C + 63) // 64,)
        )
        keys = np.memmap(path, dtype=np.int32, mode='r', offset=ko, shape=(C, nd))
        vals = np.memmap(path, dtype=np.float64, mode='r', offset=vo, shape=(C, no))
        chunk = 1 << 22
        for s0 in range(0, C, chunk):
            s1 = min(s0 + chunk, C)
            idx = np.arange(s0, s1, dtype=np.int64)
            # Keep the bit-twiddle entirely unsigned: NumPy 2.x forbids uint64 >> int64
            # (a TypeError that, swallowed by _load, would silently regenerate the
            # whole cache — the exact disaster this cache format exists to prevent).
            word = occ[idx >> 6]  # uint64[], indexed by int64 (allowed)
            shift = (idx & 63).astype(np.uint64)
            bit = ((word >> shift) & np.uint64(1)).astype(bool)
            sel = idx[bit]
            if sel.size:
                itor.add_point_data_arrays(
                    np.ascontiguousarray(keys[sel]), np.ascontiguousarray(vals[sel])
                )

    def _load_cache(self, itor, pkl_path: str) -> int | None:
        """
        Load the mmap-arena cache: the arena base is mapped in place (O(1), no per-point
        rebuild) and any trailing DELTA frames are merged into the overlay.

        :returns: number of points loaded, or ``None`` if there is no readable cache here
                  (absent / not this format / unsupported interpolator / load raised) so the
                  caller can fall back to the static pickle path or regenerate.
        """
        if not (hasattr(itor, 'set_point_data_arrays') and self._is_cache(pkl_path)):
            return None
        try:
            return self._load(itor, pkl_path)
        except Exception as err:
            print(
                "cache load failed for",
                pkl_path,
                "-",
                type(err).__name__,
                str(err),
            )
            return None

    def _safe_pickle_load(self, path: str) -> Any | None:
        """
        Load a plain-pickle point-data cache (used by STATIC interpolators, which have no
        mmap-arena equivalent). Returns the unpickled dict, or None.

        A corrupt/unparseable pickle is removed (it cannot be loaded anyway); any other
        error returns None without deleting.
        """
        # Never treat an arena-format cache file as a (corrupt) pickle: it is not pickle, so
        # pickle.load would fail and the os.remove path below would DELETE a perfectly good
        # cache. It is loaded via _load_cache; returning None here (without deleting) is safe.
        if self._is_cache(path):
            return None
        try:
            with open(path, "rb") as fp:
                return pickle.load(fp)
        except Exception as err:
            print(
                "Failed to read cached point data from",
                path,
                "-",
                type(err).__name__,
                str(err),
            )
            try:
                os.remove(path)
                print("Removed corrupted cache file", path)
            except Exception:
                pass
            return None

    @classmethod
    def load_point_epochs(cls, path: str) -> dict:
        """
        Read the per-point evaluation epochs stored in an OBL cache file.

        Returns a dict mapping each supporting-point key (same shape as
        ``point_data_full`` / the point deltas) to the evaluation epoch — the
        batch-interpolation / nonlinear-iteration index at which the point was first
        materialized. Intended for offline analysis of OBL-space sampling and the
        evolution of active hypercubes; it never touches the live interpolator state.

        Points written before this tracking was added (or by interpolators without
        native epoch support) simply do not appear in the returned map.
        """
        return cls._read_point_epochs(path)

    @classmethod
    def _read_point_epochs(cls, path: str) -> dict:
        """Read per-point evaluation epochs from the EPOCH frames appended after the
        arena of a cache file (the walk starts at ``arena_end``)."""
        epochs: dict = {}
        try:
            with open(path, 'rb') as fp:
                if fp.read(len(cls._MAGIC)) != cls._MAGIC:
                    return epochs
                raw = fp.read(8)
                if len(raw) != 8:
                    return epochs
                (hlen,) = struct.unpack('<Q', raw)
                hj = fp.read(hlen)
                if len(hj) != hlen:
                    return epochs
                header = json.loads(hj.decode('utf-8'))
            n_dims = int(header['n_dims'])
            n_ops = int(header['n_ops'])
            filesize = os.path.getsize(path)
            offset = int(header['arena_end'])  # frames start after the arena
            hdr_size = cls._FRAME_HDR.size
            while offset + hdr_size <= filesize:
                with open(path, 'rb') as fp:
                    fp.seek(offset)
                    fhdr = fp.read(hdr_size)
                if len(fhdr) != hdr_size:
                    break
                kind, n_rows, crc = cls._FRAME_HDR.unpack(fhdr)
                n_rows = int(n_rows)
                payload_off = offset + hdr_size
                keys_bytes = n_rows * n_dims * 4
                if kind == cls._KIND_EPOCH:
                    payload_len = keys_bytes + n_rows * 8
                    if payload_off + payload_len > filesize:
                        break
                    if n_rows > 0:
                        keys = np.fromfile(
                            path,
                            dtype=np.int32,
                            count=n_rows * n_dims,
                            offset=payload_off,
                        ).reshape(n_rows, n_dims)
                        eps = np.fromfile(
                            path,
                            dtype=np.uint64,
                            count=n_rows,
                            offset=payload_off + keys_bytes,
                        )
                        for i in range(n_rows):
                            epochs[tuple(int(x) for x in keys[i])] = int(eps[i])
                    offset = payload_off + payload_len
                    continue
                payload_len = keys_bytes + n_rows * n_ops * 8
                if payload_off + payload_len > filesize:
                    break
                offset = payload_off + payload_len
        except Exception:
            pass
        return epochs
