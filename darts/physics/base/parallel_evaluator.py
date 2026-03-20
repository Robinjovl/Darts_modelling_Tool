"""
ParallelEvaluator: multiprocessing-based parallel batch evaluation of supporting points.

Wraps an existing operator_set_evaluator_iface (or factory thereof) and dispatches
evaluate_batch() calls to a pool of worker processes, each with its own reconstructed
evaluator instance. This provides true CPU parallelism for flash/property calculations
that are CPU-bound and not thread-safe.

Usage:
    from darts.physics.base.parallel_evaluator import ParallelEvaluator

    def my_factory():
        pc = PropertyContainer(...)
        # ... attach flash, density, viscosity evaluators ...
        return ReservoirOperators(pc, thermal=False)

    par_eval = ParallelEvaluator(evaluator_factory=my_factory, n_workers=4)
    # par_eval can be passed to create_interpolator() as the evaluator
"""

import os
import warnings
import multiprocessing
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
            warnings.warn(f"Worker evaluation failed for point {i}: {e}")
            vv_np = np.asarray(vv)
            vv_np[:] = np.nan
        results[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)
    return results


# ── ParallelEvaluator class ─────────────────────────────────────────────────


class ParallelEvaluator(operator_set_evaluator_iface):
    """
    Wraps an evaluator factory and dispatches evaluate_batch() to a multiprocessing pool.

    Each worker process constructs its own evaluator via the factory, providing full
    isolation of PropertyContainer state, flash solvers, and third-party libraries.
    Single-point evaluate() delegates to a local serial evaluator instance.

    :param evaluator_factory: Callable ``() -> operator_set_evaluator_iface``.
        Must be picklable (required by multiprocessing). Called once per worker.
    :param n_workers: Number of worker processes. Defaults to ``os.cpu_count()``.
    """

    def __init__(self, evaluator_factory, n_workers=None):
        super().__init__()
        self._factory = evaluator_factory
        self._n_workers = n_workers or os.cpu_count()

        # Local serial evaluator for single-point evaluate() calls
        self._serial_evaluator = evaluator_factory()

        # Create persistent worker pool — amortizes process creation cost
        self._pool = multiprocessing.Pool(
            processes=self._n_workers,
            initializer=_worker_init,
            initargs=(evaluator_factory,),
        )

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
            return self._serial_evaluate_batch(states_np, n_points, values_np, n_dims, n_ops)

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
                f"Parallel evaluation failed: {e}. Falling back to serial."
            )
            return self._serial_evaluate_batch(states_np, n_points, values_np, n_dims, n_ops)

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
