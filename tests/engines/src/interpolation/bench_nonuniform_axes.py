import time

import numpy as np

import darts.engines as engines
from darts.engines import (
    index_vector,
    operator_set_evaluator_iface,
    timer_node,
    value_vector,
)


class Linear(operator_set_evaluator_iface):
    def __init__(self, n_dim, n_ops):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops
        self.A = np.arange((n_dim + 1) * n_ops, dtype=np.float64).reshape(
            (n_ops, n_dim + 1)
        )

    def evaluate(self, state: value_vector, values: value_vector):
        vec_state = np.asarray(state)
        vec_values = np.asarray(values)
        vec_values[:] = (
            self.A[:, : self.n_dim].dot(vec_state) + self.A[:, self.n_dim]
        ).flatten()
        return 0


def _get_multilinear_name(n_dim, n_ops):
    return f"multilinear_adaptive_cpu_interpolator_l_d_{n_dim}_{n_ops}"


def _make_interpolator(evaluator, axes_points, axes_min, axes_max, axis_nodes=None):
    ctor = getattr(engines, _get_multilinear_name(evaluator.n_dim, evaluator.n_ops))
    if axis_nodes is None:
        interpolator = ctor(
            evaluator,
            index_vector(axes_points),
            value_vector(axes_min),
            value_vector(axes_max),
        )
    else:
        interpolator = ctor(
            evaluator,
            index_vector(axes_points),
            value_vector(axes_min),
            value_vector(axes_max),
            axis_nodes,
        )
    interpolator.init()
    interpolator.init_timer_node(timer_node())
    return interpolator


def _run_pass(interpolator, states, n_ops, n_dim):
    n_states = states.shape[0]
    states_vec = value_vector(states.flatten())
    block_idx = index_vector(np.arange(n_states, dtype=np.int32))
    values = value_vector(np.zeros(n_ops * n_states))
    dvalues = value_vector(np.zeros(n_ops * n_dim * n_states))
    start = time.perf_counter()
    interpolator.evaluate_with_derivatives(states_vec, block_idx, values, dvalues)
    elapsed = time.perf_counter() - start
    return elapsed


def _build_bin_table(axis_nodes):
    n_points = len(axis_nodes)
    intervals = max(n_points - 1, 1)
    bin_count = max(1, min(1024, intervals * 4))
    span = axis_nodes[-1] - axis_nodes[0]
    bin_width = span / bin_count if bin_count > 0 else 0.0
    indices = np.zeros(bin_count, dtype=np.int32)
    idx = 0
    for b in range(bin_count):
        pivot = axis_nodes[0] + b * bin_width
        while idx + 1 < n_points and axis_nodes[idx + 1] <= pivot:
            idx += 1
        indices[b] = min(idx, n_points - 2)
    inv_width = 1.0 / bin_width if bin_width > 0 else 0.0
    return indices, inv_width


def _analyze_corrections(axis_nodes, states):
    axis_nodes = np.asarray(axis_nodes, dtype=np.float64)
    bin_table, inv_width = _build_bin_table(axis_nodes)
    min_v = axis_nodes[0]
    corrections = []
    for value in states:
        bin_idx = int((value - min_v) * inv_width)
        bin_idx = min(len(bin_table) - 1, max(0, bin_idx))
        guess = bin_table[bin_idx]
        exact = np.searchsorted(axis_nodes, value, side="right") - 1
        exact = min(len(axis_nodes) - 2, max(0, exact))
        corrections.append(abs(exact - guess))
    corrections = np.array(corrections, dtype=np.int32)
    return corrections


def benchmark():
    n_dim = 5
    n_ops = 24
    evaluator = Linear(n_dim, n_ops)
    clustered_nodes = [
        [0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0],
        [1.0, 1.02, 1.05, 1.2, 1.5, 2.0, 3.0, 4.0],
        [-5.0, -2.0, -1.0, -0.4, -0.1, 0.0, 0.1],
        [10.0, 11.0, 11.5, 12.0, 12.1, 12.2, 13.0],
        [-2.0, -1.9, -1.5, -0.5, 0.0, 0.4, 1.0],
    ]
    axes_points = [len(a) for a in clustered_nodes]
    axes_min = [a[0] for a in clustered_nodes]
    axes_max = [a[-1] for a in clustered_nodes]

    interpolator_uniform = _make_interpolator(
        evaluator, axes_points, axes_min, axes_max
    )
    interpolator_nonuniform = _make_interpolator(
        evaluator, axes_points, axes_min, axes_max, axis_nodes=clustered_nodes
    )

    rng = np.random.default_rng(1234)
    n_states = 256
    states = rng.uniform(low=axes_min, high=axes_max, size=(n_states, n_dim))
    warmup_states = rng.uniform(low=axes_min, high=axes_max, size=(64, n_dim))

    _run_pass(interpolator_uniform, warmup_states, n_ops, n_dim)
    _run_pass(interpolator_nonuniform, warmup_states, n_ops, n_dim)

    uniform_time = _run_pass(interpolator_uniform, states, n_ops, n_dim)
    nonuniform_time = _run_pass(interpolator_nonuniform, states, n_ops, n_dim)

    corrections = _analyze_corrections(clustered_nodes[0], states[:, 0])
    correction_hist = {
        steps: int(np.sum(corrections == steps))
        for steps in range(corrections.max() + 1)
    }

    print("Benchmark summary (lower is better):")
    print(f"  Uniform grid    : {uniform_time:.4f} s for {n_states} states")
    print(f"  Non-uniform grid: {nonuniform_time:.4f} s for {n_states} states")
    print(
        f"  Throughput ratio: {(uniform_time / nonuniform_time):.3f} (uniform/non-uniform)"
    )
    print("  Correction histogram (per-axis guess vs actual interval):")
    for steps, count in correction_hist.items():
        print(f"    {steps} steps: {count} hits")


if __name__ == "__main__":
    benchmark()
