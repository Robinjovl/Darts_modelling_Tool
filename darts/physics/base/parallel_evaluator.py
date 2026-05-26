"""
ParallelEvaluator: multiprocessing-based parallel batch evaluation of supporting points.

Wraps an evaluator factory and dispatches evaluate_batch() calls to a pool of worker
processes, each holding its own reconstructed evaluator instance. This gives true CPU
parallelism for flash/property calculations that are CPU-bound and not thread-safe.

Two pieces are exported:

* :class:`ModelEvaluatorFactory` -- the default, picklable factory. It reconstructs a
  model from its constructor arguments and returns ``physics.reservoir_operators[region]``,
  reusing the model's own ``set_physics``/``PropertyContainer`` build. No per-model
  duplication of the property stack is required.
* :class:`ParallelEvaluator` -- the evaluator wrapper holding the worker pool.

Both the factory and the module-level worker functions are top-level objects (not
closures), so they pickle correctly under the ``fork`` start method (Linux default)
*and* the ``spawn`` start method (Windows and macOS default).
"""

import multiprocessing
import os
import pickle
import warnings

import numpy as np

from darts.engines import operator_set_evaluator_iface, value_vector

# ── Module-level worker functions (must be picklable for multiprocessing) ────

_worker_evaluator = None


def _worker_init(factory):
    """Called once per worker process to construct its own evaluator."""
    global _worker_evaluator
    _worker_evaluator = factory()


def _worker_evaluate_chunk(coords_flat, n_dims, n_ops):
    """
    Evaluate a contiguous chunk of points in a worker process.

    :param coords_flat: 1-D numpy array of shape [n_pts * n_dims]
    :param n_dims: number of dimensions per point
    :param n_ops: number of operators per point
    :return: 1-D numpy array of shape [n_pts * n_ops]
    """
    global _worker_evaluator
    n_pts = len(coords_flat) // n_dims
    results = np.empty(n_pts * n_ops, dtype=np.float64)

    for i in range(n_pts):
        sv = value_vector(coords_flat[i * n_dims : (i + 1) * n_dims].copy())
        vv = value_vector(np.zeros(n_ops))
        try:
            _worker_evaluator.evaluate(sv, vv)
        except Exception as e:
            warnings.warn(
                f"Worker evaluation failed for point {i}: {e}",
                stacklevel=2,
            )
            vv_np = np.asarray(vv)
            vv_np[:] = np.nan
        results[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)
    return results


# ── Default picklable factory: rebuild the model's region evaluator ─────────


class ModelEvaluatorFactory:
    """
    Picklable factory that rebuilds a model's reservoir operator evaluator.

    This is the default mechanism behind :meth:`DartsModel.get_evaluator_factory`.
    It stores the model class and the (plain-data) constructor arguments captured
    at construction time. On every call it reconstructs the model and returns
    ``physics.reservoir_operators[region]``. Because it reuses the model's own
    ``set_physics``/``PropertyContainer`` build, there is no per-model duplication
    of the property/flash/kinetics stack.

    It is a plain top-level class (not a closure), so it pickles correctly under
    both the ``fork`` and the ``spawn`` multiprocessing start methods -- the latter
    being the default on Windows and macOS.

    :param model_cls: The :class:`DartsModel` subclass to reconstruct.
    :param init_args: Positional arguments the model was constructed with.
    :param init_kwargs: Keyword arguments the model was constructed with.
    :param region: Region index whose ``reservoir_operators`` entry is returned.
    """

    def __init__(self, model_cls, init_args, init_kwargs, region):
        self.model_cls = model_cls
        self.init_args = tuple(init_args)
        self.init_kwargs = dict(init_kwargs)
        self.region = region

    def __call__(self):
        # Reconstruct the model: runs the model's own set_reservoir/set_physics.
        # init() is intentionally NOT called -- no engine, no nested worker pool.
        model = self.model_cls(*self.init_args, **self.init_kwargs)
        # reservoir_operators are normally populated by init_physics(); build just
        # the operator objects here from the property containers set in set_physics.
        model.physics.set_operators()
        return model.physics.reservoir_operators[self.region]


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


# ── ParallelEvaluator class ─────────────────────────────────────────────────


class ParallelEvaluator(operator_set_evaluator_iface):
    """
    Wraps an evaluator factory and dispatches evaluate_batch() to a multiprocessing pool.

    Each worker process constructs its own evaluator via the factory, providing full
    isolation of PropertyContainer state, flash solvers, and third-party libraries.
    Single-point evaluate() delegates to a local serial evaluator instance.

    :param evaluator_factory: Picklable callable ``() -> operator_set_evaluator_iface``.
        Called once per worker process and once in the parent for the serial path.
        Must be picklable -- use a top-level callable or :class:`ModelEvaluatorFactory`,
        never a nested function / lambda / closure.
    :param n_workers: Number of worker processes. Defaults to ``os.cpu_count()``.
    :param start_method: Optional multiprocessing start method (``'fork'``, ``'spawn'``,
        ``'forkserver'``). ``None`` uses the platform default. Mainly useful to force
        ``'spawn'`` for Windows-parity testing on Linux.
    """

    def __init__(self, evaluator_factory, n_workers=None, start_method=None):
        super().__init__()

        # Fail early with a clear message if the factory is not spawn-safe.
        _check_picklable(evaluator_factory)

        self._factory = evaluator_factory
        self._n_workers = n_workers or os.cpu_count()

        # Local serial evaluator for single-point evaluate() calls
        self._serial_evaluator = evaluator_factory()

        # Create persistent worker pool — amortizes process creation cost.
        # get_context(None) returns the platform-default context.
        ctx = multiprocessing.get_context(start_method)
        self._pool = ctx.Pool(
            processes=self._n_workers,
            initializer=_worker_init,
            initargs=(evaluator_factory,),
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
        Parallel batch evaluation via multiprocessing pool.

        Splits the batch into roughly equal chunks (one per worker), dispatches
        to the pool, and gathers results back into the output values array.

        Falls back to serial evaluation if the pool encounters an error.

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

        # For very small batches, skip pool overhead
        if n_points <= self._n_workers:
            return self._serial_evaluate_batch(
                states_np, n_points, values_np, n_dims, n_ops
            )

        # Split points into chunks for workers
        chunk_size = (n_points + self._n_workers - 1) // self._n_workers
        chunks = []
        for w in range(self._n_workers):
            start = w * chunk_size
            end = min(start + chunk_size, n_points)
            if start >= n_points:
                break
            chunks.append(states_np[start * n_dims : end * n_dims].copy())

        try:
            results = self._pool.starmap(
                _worker_evaluate_chunk,
                [(chunk, n_dims, n_ops) for chunk in chunks],
            )
        except Exception as e:
            warnings.warn(
                f"Parallel evaluation failed: {e}. Falling back to serial.",
                stacklevel=2,
            )
            return self._serial_evaluate_batch(
                states_np, n_points, values_np, n_dims, n_ops
            )

        # Gather results into output array
        offset = 0
        for r in results:
            n = len(r)
            values_np[offset : offset + n] = r
            offset += n

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
        """Terminate and join the worker pool."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.shutdown()
