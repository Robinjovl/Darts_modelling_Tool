"""
ParallelEvaluator: multiprocessing-based parallel batch evaluation of supporting points.

Wraps an evaluator factory and dispatches evaluate_batch() calls to a pool of worker
processes, each holding its own reconstructed evaluator instance. This gives true CPU
parallelism for flash/property calculations that are CPU-bound and not thread-safe.

Three pieces are exported:

* :class:`ModelEvaluatorFactory` -- the default, picklable factory. It reconstructs a
  model from its constructor arguments and returns one of the operator-evaluator
  attributes (``reservoir_operators[region]``, ``property_operators[region]``,
  ``well_operators``, ``well_ctrl_operators``, ``thermal_var_operator``, ...),
  reusing the model's own ``set_physics``/``PropertyContainer`` build. No per-model
  duplication of the property stack is required.
* :class:`SharedEvaluatorPool` -- a single multiprocessing pool shared by multiple
  :class:`ParallelEvaluator` wrappers within one PhysicsBase instance. Each worker
  pre-builds one evaluator per wrap key, so a 4-worker pool handles batches for all
  five (reservoir / property / well / well_ctrl / thermal_var) wraps with a single
  set of n_workers processes -- not n_wraps × n_workers.
* :class:`ParallelEvaluator` -- the evaluator wrapper plugged into the C++
  interpolator. Either owns a private pool (back-compat) or routes its batches
  through an externally provided :class:`SharedEvaluatorPool`.

All top-level classes and worker functions pickle correctly under both ``fork`` and
``spawn`` start methods.
"""

import contextlib
import multiprocessing
import os
import pickle
import sys
import warnings

import numpy as np

from darts.engines import operator_set_evaluator_iface, value_vector

# ── Module-level worker state and functions (must be picklable) ────────────

# In each worker process: a dict {key -> evaluator}. One evaluator per wrap key.
_worker_evaluators: dict = {}

# In each worker process: {key -> LocalPointStore} for operator-value rows
# (extrapolation supporting points), and {region -> LocalPointStore} for flash
# rows shared between the operator sets of one region. See _attach_worker_stores.
_worker_op_stores: dict = {}
_worker_flash_stores: dict = {}


class LocalPointStore:
    """
    In-process, dict-backed implementation of the adaptive interpolator's
    single-point API (``try_get_point``/``set_point``), attachable through
    :meth:`OperatorsBase.attach_point_store
    <darts.physics.base.operator_evaluator.OperatorsBase.attach_point_store>`.

    Used in pool worker processes, which cannot reach the parent interpolator's
    C++ point store: attached to a worker evaluator it caches extrapolation
    supporting points across batches for the lifetime of the worker; attached to
    the worker's FlashOperators it caches flash rows the same way (shared
    between the operator sets of one region within the worker). With
    ``track_fresh``, newly inserted keys are remembered in insertion order so
    :meth:`drain_fresh` can ship the rows back to the parent process after each
    batch (see :meth:`ParallelEvaluator.evaluate_batch` and
    :meth:`OperatorsBase.insert_point_rows
    <darts.physics.base.operator_evaluator.OperatorsBase.insert_point_rows>`).

    :param n_slots: Row width (informational; rows are stored as given).
    :param track_fresh: Keep newly inserted keys for :meth:`drain_fresh`.
    """

    def __init__(self, n_slots, track_fresh=False):
        self.n_slots = n_slots
        self.rows = {}
        self._fresh = [] if track_fresh else None

    def try_get_point(self, key):
        row = self.rows.get(tuple(int(k) for k in key))
        return None if row is None else row.copy()

    def set_point(self, key, values):
        k = tuple(int(i) for i in key)
        self.rows[k] = np.asarray(values, dtype=np.float64).copy()
        if self._fresh is not None:
            self._fresh.append(k)

    def drain_fresh(self):
        """Return the (key, row) pairs inserted since the last drain."""
        if not self._fresh:
            return []
        out = [(k, self.rows[k]) for k in self._fresh]
        self._fresh = []
        return out


@contextlib.contextmanager
def _suppress_stdout():
    """
    Suppress Python and native (fd 1) stdout for the duration, then restore.

    ``stderr`` is left untouched so genuine errors still surface. Used to silence the
    duplicate model/evaluator construction a factory performs (the real model has already
    printed the same setup), so only one evaluator's worth of output reaches the console.
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)
    devnull = open(os.devnull, 'w')
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 1)
        os.close(null_fd)
        with contextlib.redirect_stdout(devnull):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        devnull.close()


def _silence_worker_process():
    """
    Permanently silence a worker process so only the main process prints.

    A forked worker inherits the parent's stdout and C++ darts output stream (possibly the
    main run's log file). Without this, every worker re-emits the whole model/evaluator
    construction and would also interleave its C++ output into the owner's log. We redirect
    the darts output to nowhere and send stdout (Python + native fd 1) to /dev/null;
    ``stderr`` is kept so real worker errors are still visible.
    """
    try:
        from darts.engines import redirect_darts_output

        redirect_darts_output('')
    except Exception:
        pass
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 1)
        os.close(null_fd)
    except Exception:
        pass
    try:
        sys.stdout = open(os.devnull, 'w')
    except Exception:
        pass


class _DartsLogStream:
    """A minimal file-like object whose writes go to the darts output stream (the file set
    by ``redirect_darts_output``), so Python prints land in the run log rather than on the
    console. Used when evaluator output is explicitly enabled: it must reach the log only,
    never stdout."""

    def __init__(self):
        from darts.engines import write_to_darts_output

        self._write = write_to_darts_output

    def write(self, s):
        if s:
            try:
                self._write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def _route_worker_to_log():
    """Send an enabled worker's output to the darts log *only* — never to the console.

    The forked worker keeps the inherited darts output redirect (the main run's log), so its
    C++ output already lands there; its Python stdout is forwarded to the same log via
    ``write_to_darts_output``; and the native fd 1 is sent to /dev/null so reaktoro/native
    writes never reach the console. (Several workers sharing the log can interleave — this
    is debug-only output at the highest verbosity.)
    """
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 1)
        os.close(null_fd)
    except Exception:
        pass
    try:
        sys.stdout = _DartsLogStream()
    except Exception:
        pass


@contextlib.contextmanager
def _stdout_to_log():
    """Temporarily route Python stdout to the darts log and native fd 1 to /dev/null, then
    restore. The block's output reaches the log only, never the console."""
    sys.stdout.flush()
    saved_fd = os.dup(1)
    saved_stdout = sys.stdout
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 1)
        os.close(null_fd)
        sys.stdout = _DartsLogStream()
        yield
    finally:
        sys.stdout = saved_stdout
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def _attach_worker_stores(store_configs: dict):
    """
    Attach :class:`LocalPointStore` caches to the evaluators of this worker.

    For every key with a store config, the evaluator gets its own operator-value
    store (extrapolation supporting points, ``track_fresh=True`` so each batch
    ships the new rows back to the parent), and its private ``FlashOperators``
    gets a flash-row store keyed on the primary axes. The flash store is shared
    between all of this worker's evaluators of the same region, so a grid node
    flashed for one operator family (e.g. reservoir) is a hit for the others
    (e.g. property) -- the same row contract the parent's shared flash store
    relies on. Flash rows stay worker-local (not shipped to the parent).

    :param store_configs: dict ``key -> {'axes_origin', 'axes_step',
        'flash_axes_origin', 'flash_axes_step'}`` for the keys to wire; keys
        without a config (e.g. thermal_var_operator, which lives on a different
        grid) are left uncached.
    """
    global _worker_op_stores, _worker_flash_stores
    _worker_op_stores = {}
    _worker_flash_stores = {}
    for key, evaluator in _worker_evaluators.items():
        cfg = store_configs.get(key)
        if cfg is None or not hasattr(evaluator, 'attach_point_store'):
            continue

        n_ops = getattr(evaluator, 'n_ops', None)
        if n_ops:
            store = LocalPointStore(n_ops, track_fresh=True)
            evaluator.attach_point_store(
                store,
                n_slots=n_ops,
                axes_origin=cfg['axes_origin'],
                axes_step=cfg['axes_step'],
            )
            _worker_op_stores[key] = store

        flash = getattr(evaluator, 'flash', None)
        if (
            flash is not None
            and flash is not evaluator
            and hasattr(flash, 'attach_point_store')
        ):
            width = flash.property.flash_row_width()
            region = key[1] if isinstance(key, tuple) and len(key) == 2 else None
            flash_store = _worker_flash_stores.get(region)
            if flash_store is None or flash_store.n_slots != width:
                flash_store = LocalPointStore(width)
                _worker_flash_stores[region] = flash_store
            flash.attach_point_store(
                flash_store,
                n_slots=width,
                axes_origin=cfg['flash_axes_origin'],
                axes_step=cfg['flash_axes_step'],
            )


def _worker_init_multi(
    factories: dict, silence_workers: bool = True, store_configs: dict = None
):
    """Build one evaluator per wrap key in this worker process.

    When ``silence_workers`` (the default) the worker is muted so the model's console /
    log shows only one evaluator's output; pass ``False`` (e.g. at the highest DartsModel
    verbosity) to let every worker print — routed to the run log only, never to stdout.

    ``store_configs`` (see :func:`_attach_worker_stores`) wires worker-local
    point-store caches to the freshly built evaluators.
    """
    global _worker_evaluators
    if silence_workers:
        _silence_worker_process()
    else:
        _route_worker_to_log()
    _worker_evaluators = {key: factory() for key, factory in factories.items()}
    if store_configs:
        _attach_worker_stores(store_configs)


def _worker_evaluate_chunk_multi(key, coords_flat, n_dims, n_ops):
    """
    Evaluate a contiguous chunk of points in this worker, using the evaluator
    registered under ``key``.

    :param key: hashable key identifying which evaluator to use
    :param coords_flat: 1-D numpy array of shape [n_pts * n_dims]
    :param n_dims: number of dimensions per point
    :param n_ops: number of operators per point
    :return: tuple of (1-D numpy array of shape [n_pts * n_ops], list of
        (key, row) supporting-point rows tabulated in this worker's local point
        store during the chunk -- see LocalPointStore.drain_fresh)
    """
    global _worker_evaluators
    evaluator = _worker_evaluators[key]
    n_pts = len(coords_flat) // n_dims
    results = np.empty(n_pts * n_ops, dtype=np.float64)

    for i in range(n_pts):
        sv = value_vector(coords_flat[i * n_dims : (i + 1) * n_dims].copy())
        vv = value_vector(np.zeros(n_ops))
        try:
            evaluator.evaluate(sv, vv)
        except Exception as e:
            warnings.warn(
                f"Worker evaluation failed for key {key!r}, point {i}: {e}",
                stacklevel=2,
            )
            vv_np = np.asarray(vv)
            vv_np[:] = np.nan
        results[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)

    op_store = _worker_op_stores.get(key)
    fresh_rows = op_store.drain_fresh() if op_store is not None else []
    return results, fresh_rows


# ── Default picklable factory: rebuild any of the model's evaluator attrs ───


class ModelEvaluatorFactory:
    """
    Picklable factory that rebuilds a named operator-evaluator attribute of a model.

    This is the default mechanism behind :meth:`DartsModel.get_evaluator_factory`.
    It stores the model class and the (plain-data) constructor arguments captured
    at construction time, plus the attribute name to fetch and an optional region
    key. On every call it reconstructs the model, runs ``physics.set_operators()``,
    and returns either ``physics.<attribute>`` (singular) or
    ``physics.<attribute>[region]`` (per-region). Because it reuses the model's own
    ``set_physics``/``PropertyContainer`` build, there is no per-model duplication
    of the property/flash/kinetics stack.

    It is a plain top-level class (not a closure), so it pickles correctly under
    both the ``fork`` and the ``spawn`` multiprocessing start methods.

    :param model_cls: The :class:`DartsModel` subclass to reconstruct.
    :param init_args: Positional arguments the model was constructed with.
    :param init_kwargs: Keyword arguments the model was constructed with.
    :param attribute: Name of the ``physics`` attribute to fetch
        (e.g. ``'reservoir_operators'``, ``'property_operators'``,
        ``'well_operators'``, ``'well_ctrl_operators'``, ``'thermal_var_operator'``).
        Defaults to ``'reservoir_operators'`` for back-compat with the original
        MR297 signature.
    :param region: Optional per-region key. ``None`` for singular attributes
        (e.g. ``well_ctrl_operators``); an integer region index for per-region
        dicts (e.g. ``reservoir_operators[0]``).
    """

    def __init__(
        self,
        model_cls,
        init_args,
        init_kwargs,
        attribute: str = 'reservoir_operators',
        region=None,
    ):
        self.model_cls = model_cls
        self.init_args = tuple(init_args)
        self.init_kwargs = dict(init_kwargs)
        self.attribute = attribute
        self.region = region

    def __call__(self):
        # Reconstruct the model: runs the model's own set_reservoir/set_physics.
        # init() is intentionally NOT called -- no engine, no nested worker pool.
        model = self.model_cls(*self.init_args, **self.init_kwargs)
        # reservoir_operators are normally populated by init_physics(); build just
        # the operator objects here from the property containers set in set_physics.
        model.physics.set_operators()
        obj = getattr(model.physics, self.attribute)
        return obj[self.region] if self.region is not None else obj


class OutputPropertyOperatorsFactory:
    """
    Picklable factory that rebuilds an output-only ``PropertyOperators`` instance
    in a worker process.

    Unlike :class:`ModelEvaluatorFactory` (which fetches an existing physics
    attribute), this factory *constructs* a fresh ``PropertyOperators`` from a
    picklable :class:`OutputPropertyDescriptor` that lists the phase-property
    keys to expose. Used by :meth:`OutputBase.set_phase_properties` when the
    physics has a live shared evaluator pool: the existing pool is extended
    with new keys ``('output_property_operators', region)`` whose worker-side
    evaluator is built by this factory.

    :param model_cls: The :class:`DartsModel` subclass to reconstruct.
    :param init_args: Positional args the model was constructed with.
    :param init_kwargs: Keyword args the model was constructed with.
    :param region: Region index whose property container drives the dict layout.
    :param descriptor: :class:`OutputPropertyDescriptor` instance describing the
        output-property layout.
    :param thermal: Pass-through to ``PropertyOperators(thermal=...)``.
    :param extrapolation_flag: Pass-through to ``PropertyOperators(extrapolation_flag=...)``.
    :param dz: Pass-through to ``PropertyOperators(dz=...)``.
    """

    def __init__(
        self,
        model_cls,
        init_args,
        init_kwargs,
        region: int,
        descriptor,
        thermal: bool,
        extrapolation_flag: bool,
        dz,
    ):
        self.model_cls = model_cls
        self.init_args = tuple(init_args)
        self.init_kwargs = dict(init_kwargs)
        self.region = region
        self.descriptor = descriptor
        self.thermal = thermal
        self.extrapolation_flag = extrapolation_flag
        self.dz = dz

    def __call__(self):
        from darts.physics.base.operator_evaluator import PropertyOperators

        model = self.model_cls(*self.init_args, **self.init_kwargs)
        model.physics.set_operators()
        pc = model.physics.property_containers[self.region]
        temp_dict = self.descriptor.materialize(pc)
        return PropertyOperators(
            property_container=pc,
            thermal=self.thermal,
            props=temp_dict,
            extrapolation_flag=self.extrapolation_flag,
            dz=self.dz,
        )


def _check_picklable(factory):
    """Raise a clear error if the factory cannot be pickled (required for spawn)."""
    try:
        pickle.dumps(factory)
    except Exception as e:
        raise ValueError(
            "evaluator_factory must be picklable so the multiprocessing pool can "
            "ship it to worker processes under the 'spawn' start method "
            "(the default on Windows and macOS). "
            f"Pickling failed with: {type(e).__name__}: {e}. "
            "Use a top-level callable or ModelEvaluatorFactory instead of a "
            "nested function / lambda / closure."
        ) from e


# ── Shared multiprocessing pool ─────────────────────────────────────────────


class SharedEvaluatorPool:
    """
    Multiprocessing pool shared by multiple :class:`ParallelEvaluator` wrappers.

    Built with a ``{key: factory}`` dict that lists every evaluator the pool must
    be able to dispatch to. Each worker process pre-builds one evaluator per key
    at startup. The total process count is fixed at ``n_workers`` regardless of
    how many wrap targets the pool serves, which addresses the per-region
    multi-pool follow-up flagged in MR297.

    :param factories: dict mapping a hashable key to a picklable factory callable
        ``() -> operator_set_evaluator_iface``. Every factory is pickle-checked
        eagerly so failures surface in the parent, not deep inside a worker.
    :param n_workers: Number of worker processes (default: ``os.cpu_count()``).
    :param start_method: Optional multiprocessing start method (``'fork'``,
        ``'spawn'``, ``'forkserver'``). ``None`` uses the platform default.
    :param store_configs: Optional dict ``key -> axes config`` wiring worker-local
        point-store caches to the worker evaluators (see
        :func:`_attach_worker_stores`). Keys without a config stay uncached.
    """

    def __init__(
        self,
        factories: dict,
        n_workers=None,
        start_method=None,
        silence_workers=True,
        store_configs: dict = None,
    ):
        if not factories:
            raise ValueError("SharedEvaluatorPool requires at least one factory.")
        for f in factories.values():
            _check_picklable(f)

        self.n_workers = n_workers or os.cpu_count()
        self._factories = dict(factories)
        self._store_configs = dict(store_configs) if store_configs else {}

        ctx = multiprocessing.get_context(start_method)
        self._pool = ctx.Pool(
            processes=self.n_workers,
            initializer=_worker_init_multi,
            initargs=(self._factories, silence_workers, self._store_configs),
        )

    def has_key(self, key) -> bool:
        return key in self._factories

    def evaluate_batch(self, key, states_np, n_points, values_np, n_dims, n_ops):
        """
        Dispatch a batch under ``key`` to the worker pool. Splits the batch into
        roughly equal chunks (one per worker), gathers results back into
        ``values_np``. Returns ``(True, fresh_rows)`` on success -- where
        ``fresh_rows`` is the merged list of (key, row) supporting-point rows the
        workers tabulated during this batch (see LocalPointStore.drain_fresh),
        for the caller to insert into the parent interpolator's store -- or
        ``(False, [])`` if the pool raised and the caller should fall back to
        serial evaluation.
        """
        if n_points == 0:
            return True, []

        chunk_size = (n_points + self.n_workers - 1) // self.n_workers
        chunks = []
        for w in range(self.n_workers):
            start = w * chunk_size
            end = min(start + chunk_size, n_points)
            if start >= n_points:
                break
            chunks.append(states_np[start * n_dims : end * n_dims].copy())

        try:
            results = self._pool.starmap(
                _worker_evaluate_chunk_multi,
                [(key, chunk, n_dims, n_ops) for chunk in chunks],
            )
        except Exception as e:
            warnings.warn(
                f"Parallel evaluation for key {key!r} failed: {e}. Falling back to serial.",
                stacklevel=2,
            )
            return False, []

        offset = 0
        fresh_rows = []
        for r, rows in results:
            n = len(r)
            values_np[offset : offset + n] = r
            offset += n
            fresh_rows.extend(rows)
        return True, fresh_rows

    def shutdown(self):
        """Terminate and join the worker pool. Idempotent."""
        pool = getattr(self, '_pool', None)
        if pool is not None:
            pool.terminate()
            pool.join()
            self._pool = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# ── ParallelEvaluator class ─────────────────────────────────────────────────


class ParallelEvaluator(operator_set_evaluator_iface):
    """
    Wraps an evaluator factory and dispatches evaluate_batch() to a multiprocessing pool.

    Two modes:

    * **Owned pool** -- pass only ``evaluator_factory`` (and optionally
      ``n_workers``/``start_method``). The wrapper creates its own
      :class:`SharedEvaluatorPool` with one entry and tears it down on shutdown.
      This is the backwards-compatible MR297 mode.
    * **Shared pool** -- pass ``shared_pool=<SharedEvaluatorPool>`` and ``key=<...>``
      to route batches through a pool that several wraps share. The wrapper does
      not own the pool and does not shut it down.

    Single-point ``evaluate()`` always delegates to a local serial evaluator
    instance constructed from ``evaluator_factory``.

    :param evaluator_factory: Picklable callable ``() -> operator_set_evaluator_iface``.
        Required (used to build the local serial evaluator and, in owned-pool
        mode, the worker evaluators).
    :param n_workers: Number of worker processes when owning a pool
        (default: ``os.cpu_count()``). Ignored when ``shared_pool`` is provided.
    :param start_method: Optional multiprocessing start method when owning a
        pool. Ignored when ``shared_pool`` is provided.
    :param shared_pool: Optional :class:`SharedEvaluatorPool` to route batches
        through. Must already contain ``key``.
    :param key: Hashable wrap key used by the shared pool to pick the right
        worker evaluator. Required when ``shared_pool`` is provided.
    """

    def __init__(
        self,
        evaluator_factory,
        n_workers=None,
        start_method=None,
        *,
        shared_pool: SharedEvaluatorPool = None,
        key=None,
        silence=True,
    ):
        super().__init__()

        # Fail early with a clear message if the factory is not spawn-safe.
        _check_picklable(evaluator_factory)

        self._factory = evaluator_factory
        # Local serial evaluator for single-point evaluate() calls. By default its
        # construction is suppressed (the real model already printed the same thing, so it
        # would be duplicate noise). When ``silence=False`` (highest DartsModel verbosity)
        # the construction is shown, but routed to the run log only — never to stdout.
        ctx = _suppress_stdout() if silence else _stdout_to_log()
        with ctx:
            self._serial_evaluator = evaluator_factory()

        if shared_pool is not None:
            if key is None:
                raise ValueError(
                    "ParallelEvaluator(shared_pool=...) requires a key identifying "
                    "the wrap target within the shared pool."
                )
            if not shared_pool.has_key(key):
                raise ValueError(
                    f"shared_pool does not contain key {key!r}; pool was built with "
                    f"{list(shared_pool._factories.keys())!r}."
                )
            self._owns_pool = False
            self._shared_pool = shared_pool
            self._key = key
            self._n_workers = shared_pool.n_workers
        else:
            # Backwards-compatible: own a private single-key pool.
            self._owns_pool = True
            self._n_workers = n_workers or os.cpu_count()
            self._key = '__default__'
            self._shared_pool = SharedEvaluatorPool(
                {self._key: evaluator_factory},
                n_workers=self._n_workers,
                start_method=start_method,
                silence_workers=silence,
            )

    def __getattr__(self, name):
        """Delegate unknown attribute lookups to the underlying serial evaluator."""
        # Avoid infinite recursion during __init__ before _serial_evaluator exists
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._serial_evaluator, name)

    def evaluate(self, state, values):
        """Single-point evaluate delegates to local serial evaluator."""
        return self._serial_evaluator.evaluate(state, values)

    def evaluate_batch(self, states, n_points, values, n_ops):
        """
        Parallel batch evaluation via the (owned or shared) multiprocessing pool.

        Falls back to serial evaluation when (a) the batch is too small to amortize
        pool overhead (``n_points <= n_workers``) or (b) the pool dispatch fails.

        :param states: Flat array of coordinates [n_points * n_dims]
        :param n_points: Number of points to evaluate
        :param values: Flat output array [n_points * n_ops], pre-allocated
        :param n_ops: Number of operators per point
        :return: 0 if successful
        """
        if n_points == 0:
            return 0

        states_np = np.asarray(states, dtype=np.float64)
        values_np = np.asarray(values)
        n_dims = len(states_np) // n_points

        # For small batches, skip pool overhead
        if n_points <= self._n_workers:
            return self._serial_evaluate_batch(
                states_np, n_points, values_np, n_dims, n_ops
            )

        succeeded, fresh_rows = self._shared_pool.evaluate_batch(
            self._key, states_np, n_points, values_np, n_dims, n_ops
        )
        if not succeeded:
            return self._serial_evaluate_batch(
                states_np, n_points, values_np, n_dims, n_ops
            )
        if fresh_rows:
            # Merge supporting-point rows the workers tabulated during boundary
            # extrapolation into the parent interpolator's point store, so later
            # interpolation/extrapolation and disk persistence reuse them.
            # (The serial evaluator holds the parent-side store attachment; a
            # generic evaluator without one silently drops the rows.)
            insert = getattr(self._serial_evaluator, 'insert_point_rows', None)
            if insert is not None:
                insert(fresh_rows)
        return 0

    def _serial_evaluate_batch(self, states_np, n_points, values_np, n_dims, n_ops):
        """Serial fallback using the local evaluator."""
        for i in range(n_points):
            sv = value_vector(states_np[i * n_dims : (i + 1) * n_dims].copy())
            vv = value_vector(np.zeros(n_ops))
            self._serial_evaluator.evaluate(sv, vv)
            values_np[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)
        return 0

    def shutdown(self):
        """Terminate the worker pool if this wrapper owns it. Idempotent."""
        if getattr(self, '_owns_pool', False):
            pool = getattr(self, '_shared_pool', None)
            if pool is not None:
                pool.shutdown()
                self._shared_pool = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
