"""
Shared multiprocessing evaluator for SPE11b region operator batches.

Each worker reconstructs the SPE11 model once, builds all reservoir operators,
and then reuses the operator for whichever region a batch targets.
"""

import multiprocessing
import os
import pickle
import warnings

import numpy as np

from darts.engines import operator_set_evaluator_iface, value_vector


_worker_region_evaluators = None


def _check_picklable(obj, label):
    try:
        pickle.dumps(obj)
    except Exception as e:
        raise ValueError(
            f"{label} must be picklable for shared SPE11b parallel evaluation. "
            f"Pickling failed with {type(e).__name__}: {e}."
        ) from e


def _build_region_evaluators(evaluator_factories):
    regions = list(evaluator_factories)
    first_factory = evaluator_factories[regions[0]]

    if all(
        hasattr(first_factory, attr)
        for attr in ("model_cls", "init_args", "init_kwargs")
    ):
        model = first_factory.model_cls(
            *first_factory.init_args,
            **first_factory.init_kwargs,
        )
        model.physics.set_operators()
        return {
            region: model.physics.reservoir_operators[region] for region in regions
        }

    return {
        region: factory() for region, factory in evaluator_factories.items()
    }


def _shared_worker_init(evaluator_factories):
    global _worker_region_evaluators
    _worker_region_evaluators = _build_region_evaluators(evaluator_factories)


def _shared_worker_evaluate_chunk(region, coords_flat, n_dims, n_ops):
    evaluator = _worker_region_evaluators[region]
    n_pts = len(coords_flat) // n_dims
    results = np.empty(n_pts * n_ops, dtype=np.float64)

    for i in range(n_pts):
        sv = value_vector(coords_flat[i * n_dims : (i + 1) * n_dims].copy())
        vv = value_vector(np.zeros(n_ops))
        try:
            evaluator.evaluate(sv, vv)
        except Exception as e:
            warnings.warn(
                f"SPE11b shared worker evaluation failed for region {region}, "
                f"point {i}: {e}",
                stacklevel=2,
            )
            vv_np = np.asarray(vv)
            vv_np[:] = np.nan
        results[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)

    return results


class SharedParallelEvaluatorPool:
    """
    Own one multiprocessing pool shared by all SPE11b region evaluators.

    :param evaluator_factories: Mapping from region to evaluator factory.
    :type evaluator_factories: dict
    :param n_workers: Total number of worker processes shared across regions.
    :type n_workers: int
    :param start_method: Optional multiprocessing start method.
    :type start_method: str
    """

    def __init__(self, evaluator_factories, n_workers=None, start_method=None):
        self._factories = dict(evaluator_factories)
        _check_picklable(self._factories, "evaluator_factories")

        self.n_workers = n_workers or os.cpu_count() or 1
        ctx = multiprocessing.get_context(start_method)
        self._pool = ctx.Pool(
            processes=self.n_workers,
            initializer=_shared_worker_init,
            initargs=(self._factories,),
        )

    def evaluate_batch(self, region, states_np, n_points, values_np, n_dims, n_ops):
        chunk_size = (n_points + self.n_workers - 1) // self.n_workers
        chunks = []
        for w in range(self.n_workers):
            start = w * chunk_size
            end = min(start + chunk_size, n_points)
            if start >= n_points:
                break
            chunks.append(states_np[start * n_dims : end * n_dims].copy())

        results = self._pool.starmap(
            _shared_worker_evaluate_chunk,
            [(region, chunk, n_dims, n_ops) for chunk in chunks],
        )

        offset = 0
        for result in results:
            n_values = len(result)
            values_np[offset : offset + n_values] = result
            offset += n_values

        return 0

    def shutdown(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.shutdown()


class SharedRegionParallelEvaluator(operator_set_evaluator_iface):
    """
    Region-specific evaluator wrapper backed by one shared SPE11b worker pool.

    :param region: SPE11b property region.
    :type region: int
    :param serial_evaluator: Existing in-process region evaluator.
    :type serial_evaluator: operator_set_evaluator_iface
    :param shared_pool: Shared multiprocessing pool owner.
    :type shared_pool: SharedParallelEvaluatorPool
    """

    def __init__(self, region, serial_evaluator, shared_pool):
        super().__init__()
        self._region = region
        self._serial_evaluator = serial_evaluator
        self._shared_pool = shared_pool

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._serial_evaluator, name)

    def evaluate(self, state, values):
        return self._serial_evaluator.evaluate(state, values)

    def evaluate_batch(self, states, n_points, values, n_ops):
        if n_points == 0:
            return 0

        states_np = np.asarray(states, dtype=np.float64)
        values_np = np.asarray(values)
        n_dims = len(states_np) // n_points

        if n_points <= self._shared_pool.n_workers:
            return self._serial_evaluate_batch(
                states_np, n_points, values_np, n_dims, n_ops
            )

        try:
            return self._shared_pool.evaluate_batch(
                self._region, states_np, n_points, values_np, n_dims, n_ops
            )
        except Exception as e:
            warnings.warn(
                f"SPE11b shared parallel evaluation failed for region "
                f"{self._region}: {e}. Falling back to serial.",
                stacklevel=2,
            )
            return self._serial_evaluate_batch(
                states_np, n_points, values_np, n_dims, n_ops
            )

    def _serial_evaluate_batch(self, states_np, n_points, values_np, n_dims, n_ops):
        for i in range(n_points):
            sv = value_vector(states_np[i * n_dims : (i + 1) * n_dims].copy())
            vv = value_vector(np.zeros(n_ops))
            self._serial_evaluator.evaluate(sv, vv)
            values_np[i * n_ops : (i + 1) * n_ops] = np.asarray(vv)
        return 0
