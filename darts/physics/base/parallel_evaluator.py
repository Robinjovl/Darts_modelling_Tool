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
import hashlib
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


# Keeps the threadpoolctl limiter alive for the worker's lifetime; dropping the
# returned object would let threadpoolctl restore the original BLAS limits.
_worker_threadpool_limits = None


def _pin_worker_threads():
    """Pin this worker process to a single compute thread, at runtime.

    Setting ``OMP_NUM_THREADS`` alone is inert here: libgomp and OpenBLAS latch
    their thread count when the runtime initialises. Under ``fork`` that happened
    in the parent; under ``spawn`` it happens while unpickling this initializer
    imports numpy / ``darts.engines`` -- both strictly before a pool initializer
    runs. Only the runtime setters take effect, so call them explicitly; the env
    var is still exported for lazily-initialised libraries and grandchildren.
    """
    global _worker_threadpool_limits
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from darts.engines import set_num_threads  # omp_set_num_threads

        set_num_threads(1)
    except Exception:
        pass  # engines built without OpenMP: nothing to pin
    try:
        from threadpoolctl import threadpool_limits  # OpenBLAS / MKL

        _worker_threadpool_limits = threadpool_limits(limits=1)
    except Exception:
        pass  # threadpoolctl optional


def _worker_init_multi(factories: dict, silence_workers: bool = True):
    """Build one evaluator per wrap key in this worker process.

    Also pins the worker's OpenMP/BLAS threading to a single thread (see the
    comment below) before any evaluator is constructed.

    When ``silence_workers`` (the default) the worker is muted so the model's console /
    log shows only one evaluator's output; pass ``False`` (e.g. at the highest DartsModel
    verbosity) to let every worker print — routed to the run log only, never to stdout.

    :param factories: Mapping ``key -> picklable factory`` producing the worker-local
        evaluator for each wrap target served by this pool.
    :param silence_workers: Mute worker stdout (default ``True``).
    """
    global _worker_evaluators
    # Pin worker BLAS/OpenMP threading to 1 before any evaluator is built: the
    # pool typically runs dozens of workers, and inheriting the parent's thread
    # count would oversubscribe the host by an order of magnitude.
    _pin_worker_threads()
    if silence_workers:
        _silence_worker_process()
    else:
        _route_worker_to_log()
    _worker_evaluators = {key: factory() for key, factory in factories.items()}


def _worker_evaluate_chunk_multi(key, coords_flat, n_dims, n_ops):
    """
    Evaluate a contiguous chunk of points in this worker, using the evaluator
    registered under ``key``.

    :param key: hashable key identifying which evaluator to use
    :param coords_flat: 1-D numpy array of shape [n_pts * n_dims]
    :param n_dims: number of dimensions per point
    :param n_ops: number of operators per point
    :return: 1-D numpy array of shape [n_pts * n_ops]
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
    return results


# ── Default picklable factory: rebuild any of the model's evaluator attrs ───


class ModelEvaluatorFactory:
    """
    Picklable factory that rebuilds a named operator-evaluator attribute of a model.

    This is the default mechanism behind :meth:`DartsModel.get_evaluator_factory`.
    It stores the model class and the (plain-data) constructor arguments captured
    at construction time, plus the attribute name to fetch and an optional region
    key. On first call in a process it reconstructs the model and runs
    ``physics.set_operators()``; the reconstructed model is memoized in the
    process-wide :attr:`_model_cache` (keyed by :meth:`_model_key`), so factories
    for other attribute/region targets of the same model reuse it instead of
    paying the full rebuild per target. Every call returns either
    ``physics.<attribute>`` (singular) or ``physics.<attribute>[region]``
    (per-region). Because it reuses the model's own
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

    # Per-process cache of reconstructed models: a worker (or the main process)
    # asking for several wrap targets of the SAME model must not pay the full
    # set_reservoir/set_physics reconstruction once per target -- at 60k+ cells
    # the mesh discretization alone is tens of seconds per rebuild.
    _model_cache: dict = {}

    @classmethod
    def clear_model_cache(cls):
        """Drop every reconstructed model cached in this process.

        The cache is process-wide and unbounded: each entry holds a full mesh and
        physics stack, so a long-lived process that builds several distinct models
        (a parameter sweep, an optimisation loop, a notebook session) would retain
        all of them. Called from :meth:`SharedEvaluatorPool.shutdown`.
        """
        cls._model_cache.clear()

    def _model_key(self):
        """Cache key identifying the model reconstruction this factory performs.

        Two factories that would rebuild an identical model (same class, same
        constructor arguments) map to the same key regardless of which physics
        ``attribute``/``region`` they fetch, so they share one cached model.

        :return: Hashable ``(class, content-digest)`` tuple, or ``None`` when the
            arguments cannot be digested (the model is then simply not cached).
        """
        # repr() must NOT be used here: an argument without __repr__ falls back to
        # object.__repr__, which embeds its memory address -- equivalent models
        # then miss the cache, and a recycled address can make two DIFFERENT
        # models collide on one key. numpy also abbreviates repr() of arrays
        # larger than 1000 elements, so distinct fields collide deterministically.
        # Digest the pickled payload instead; the factory must be picklable to
        # reach a worker at all.
        try:
            payload = pickle.dumps(
                (self.init_args, sorted(self.init_kwargs.items())),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception:
            return None
        return (self.model_cls, hashlib.blake2b(payload, digest_size=16).digest())

    def __call__(self):
        """Return the evaluator object this factory targets.

        Reconstructs the model (``set_reservoir``/``set_physics`` +
        ``physics.set_operators()``; ``init()`` is intentionally NOT called — no
        engine, no nested worker pool) on first use in this process, then serves
        every subsequent call — including other attribute/region targets of the
        same model — from :attr:`_model_cache`.

        :return: ``model.physics.<attribute>`` (indexed by ``region`` when set).
        """
        key = self._model_key()
        model = None if key is None else ModelEvaluatorFactory._model_cache.get(key)
        if model is None:
            # Reconstruct the model: runs the model's own set_reservoir/set_physics.
            # init() is intentionally NOT called -- no engine, no nested worker pool.
            model = self.model_cls(*self.init_args, **self.init_kwargs)
            # reservoir_operators are normally populated by init_physics(); build just
            # the operator objects here from the property containers set in set_physics.
            model.physics.set_operators()
            if key is not None:
                ModelEvaluatorFactory._model_cache[key] = model
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
    """

    def __init__(
        self, factories: dict, n_workers=None, start_method=None, silence_workers=True
    ):
        if not factories:
            raise ValueError("SharedEvaluatorPool requires at least one factory.")
        for f in factories.values():
            _check_picklable(f)

        self.n_workers = n_workers or os.cpu_count()
        self._factories = dict(factories)

        ctx = multiprocessing.get_context(start_method)
        self._pool = ctx.Pool(
            processes=self.n_workers,
            initializer=_worker_init_multi,
            initargs=(self._factories, silence_workers),
        )

    def has_key(self, key) -> bool:
        return key in self._factories

    def evaluate_batch(
        self, key, states_np, n_points, values_np, n_dims, n_ops
    ) -> bool:
        """
        Dispatch a batch under ``key`` to the worker pool. Splits the batch into
        roughly equal chunks (one per worker), gathers results back into
        ``values_np``. Returns ``True`` on success, ``False`` if the pool raised
        and the caller should fall back to serial evaluation.
        """
        if n_points == 0:
            return True

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
            return False

        offset = 0
        for r in results:
            n = len(r)
            values_np[offset : offset + n] = r
            offset += n
        return True

    def shutdown(self):
        """Terminate and join the worker pool. Idempotent."""
        pool = getattr(self, '_pool', None)
        if pool is not None:
            pool.terminate()
            pool.join()
            self._pool = None
            # Release the worker-model cache with the pool that populated it.
            ModelEvaluatorFactory.clear_model_cache()

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

    Single-point ``evaluate()`` always delegates to a local serial evaluator:
    either the instance injected via ``serial_evaluator``, or one constructed
    from ``evaluator_factory``.

    :param evaluator_factory: Picklable callable ``() -> operator_set_evaluator_iface``.
        Required (used to build the worker evaluators in owned-pool mode, and the
        local serial evaluator when ``serial_evaluator`` is not given).
    :param n_workers: Number of worker processes when owning a pool
        (default: ``os.cpu_count()``). Ignored when ``shared_pool`` is provided.
    :param start_method: Optional multiprocessing start method when owning a
        pool. Ignored when ``shared_pool`` is provided.
    :param shared_pool: Optional :class:`SharedEvaluatorPool` to route batches
        through. Must already contain ``key``.
    :param key: Hashable wrap key used by the shared pool to pick the right
        worker evaluator. Required when ``shared_pool`` is provided.
    :param silence: Suppress stdout while the factory builds the local serial
        evaluator (default ``True``; the owning model already printed the same
        construction output). With ``False`` the output is routed to the run
        log. Unused when ``serial_evaluator`` is injected.
    :param serial_evaluator: Optional already-constructed evaluator to use for
        single-point ``evaluate()`` calls. Pass it when the caller owns the
        object being wrapped (the physics wrap path does) to avoid the factory
        reconstructing the entire model — mesh discretization included — once
        per wrap target.
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
        serial_evaluator=None,
    ):
        super().__init__()

        # Fail early with a clear message if the factory is not spawn-safe.
        _check_picklable(evaluator_factory)

        self._factory = evaluator_factory
        # Local serial evaluator for single-point evaluate() calls. When the caller
        # already owns the evaluator being wrapped (the physics wrap path), it is
        # injected directly -- rebuilding it through the factory would reconstruct
        # the entire model (mesh discretization included) once per wrap target.
        if serial_evaluator is not None:
            self._serial_evaluator = serial_evaluator
        else:
            # By default the construction is suppressed (the real model already printed
            # the same thing, so it would be duplicate noise). When ``silence=False``
            # (highest DartsModel verbosity) the construction is shown, but routed to
            # the run log only — never to stdout.
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

        succeeded = self._shared_pool.evaluate_batch(
            self._key, states_np, n_points, values_np, n_dims, n_ops
        )
        if not succeeded:
            return self._serial_evaluate_batch(
                states_np, n_points, values_np, n_dims, n_ops
            )
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
