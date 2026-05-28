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

import multiprocessing
import os
import pickle
import warnings

import numpy as np

from darts.engines import operator_set_evaluator_iface, value_vector

# ── Module-level worker state and functions (must be picklable) ────────────

# In each worker process: a dict {key -> evaluator}. One evaluator per wrap key.
_worker_evaluators: dict = {}


def _worker_init_multi(factories: dict):
    """Build one evaluator per wrap key in this worker process."""
    global _worker_evaluators
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

    def __init__(self, factories: dict, n_workers=None, start_method=None):
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
            initargs=(self._factories,),
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
    ):
        super().__init__()

        # Fail early with a clear message if the factory is not spawn-safe.
        _check_picklable(evaluator_factory)

        self._factory = evaluator_factory
        # Local serial evaluator for single-point evaluate() calls.
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
