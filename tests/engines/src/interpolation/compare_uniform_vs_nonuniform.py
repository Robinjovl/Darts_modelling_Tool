#!/usr/bin/env python3
"""
Compare interpolation performance for uniform vs. non-uniform axes.

The script builds two multilinear adaptive CPU interpolators that differ only
by the way axis nodes are specified and times ``evaluate_with_derivatives`` on
the same batch of randomly sampled states.

Run from the repository root:

    python tests/engines/src/interpolation/compare_uniform_vs_nonuniform.py
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np

import darts.engines as engines
from darts.engines import (
    index_vector,
    operator_set_evaluator_iface,
    timer_node,
    value_vector,
)


class Linear(operator_set_evaluator_iface):
    """Simple linear evaluator used to populate supporting points."""

    def __init__(self, n_dim: int, n_ops: int):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops
        coeffs = np.arange((n_dim + 1) * n_ops, dtype=np.float64)
        self.A = coeffs.reshape((n_ops, n_dim + 1))

    def evaluate(self, state: value_vector, values: value_vector):
        vec_state = np.asarray(state)
        vec_values = np.asarray(values)
        vec_values[:] = (
            self.A[:, : self.n_dim].dot(vec_state) + self.A[:, self.n_dim]
        ).flatten()
        return 0


DEFAULT_AXIS_NODES: list[list[float]] = [
    [0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0],
    [1.0, 1.02, 1.05, 1.2, 1.5, 2.0, 3.0, 4.0],
    [-5.0, -2.0, -1.0, -0.4, -0.1, 0.0, 0.1],
    [10.0, 11.0, 11.5, 12.0, 12.1, 12.2, 13.0],
    [-2.0, -1.9, -1.5, -0.5, 0.0, 0.4, 1.0],
]


def _interpolator_name(n_dim: int, n_ops: int) -> str:
    return f"multilinear_adaptive_cpu_interpolator_l_d_{n_dim}_{n_ops}"


def _build_interpolator(
    evaluator: Linear,
    axes_points: Sequence[int],
    axes_min: Sequence[float],
    axes_max: Sequence[float],
    axis_nodes: Sequence[Sequence[float]] | None = None,
):
    ctor = getattr(engines, _interpolator_name(evaluator.n_dim, evaluator.n_ops))
    args = [
        evaluator,
        index_vector(axes_points),
        value_vector(axes_min),
        value_vector(axes_max),
    ]
    if axis_nodes is not None:
        args.append(axis_nodes)
    try:
        interpolator = ctor(*args)
    except TypeError as exc:
        if axis_nodes is not None:
            raise RuntimeError(
                "This DARTS build does not expose the axis_nodes constructor. "
                "Rebuild the engines extension from the non-uniform axes branch."
            ) from exc
        raise
    interpolator.init()
    interpolator.init_timer_node(timer_node())
    return interpolator


def _prepare_buffers(states: np.ndarray, n_ops: int, n_dim: int):
    states = np.asarray(states, dtype=np.float64)
    n_states = states.shape[0]
    states_vec = value_vector(states.ravel())
    block_idx = index_vector(np.arange(n_states, dtype=np.int32))
    values = value_vector(np.zeros(n_ops * n_states, dtype=np.float64))
    dvalues = value_vector(np.zeros(n_ops * n_dim * n_states, dtype=np.float64))
    return n_states, states_vec, block_idx, values, dvalues


def _time_interpolator(
    interpolator,
    states_vec,
    block_idx,
    values_buf,
    dvalues_buf,
    repeats: int,
):
    durations = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        interpolator.evaluate_with_derivatives(
            states_vec, block_idx, values_buf, dvalues_buf
        )
        durations.append(time.perf_counter() - start)
    return min(durations)


def _axes_from_nodes(axis_nodes: Sequence[Sequence[float]]):
    axes_points = [len(nodes) for nodes in axis_nodes]
    axes_min = [nodes[0] for nodes in axis_nodes]
    axes_max = [nodes[-1] for nodes in axis_nodes]
    return axes_points, axes_min, axes_max


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare performance of multilinear interpolation "
            "with uniform vs. non-uniform axis definitions."
        )
    )
    parser.add_argument(
        "--states",
        type=int,
        default=1024,
        help="Number of random states used for timing (default: 1024).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=256,
        help="Number of warm-up states evaluated before timing (default: 256).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="How many timed passes to run per interpolator (best is reported).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Seed for the RNG that samples state vectors.",
    )
    parser.add_argument(
        "--n-ops",
        type=int,
        default=24,
        help="Number of operator outputs requested from the evaluator.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    axis_nodes = DEFAULT_AXIS_NODES
    n_dim = len(axis_nodes)
    axes_points, axes_min, axes_max = _axes_from_nodes(axis_nodes)

    evaluator = Linear(n_dim=n_dim, n_ops=args.n_ops)
    interpolator_uniform = _build_interpolator(
        evaluator, axes_points, axes_min, axes_max
    )
    interpolator_nonuniform = _build_interpolator(
        evaluator, axes_points, axes_min, axes_max, axis_nodes=axis_nodes
    )

    rng = np.random.default_rng(args.seed)
    warmup_states = rng.uniform(low=axes_min, high=axes_max, size=(args.warmup, n_dim))
    states = rng.uniform(low=axes_min, high=axes_max, size=(args.states, n_dim))

    _, warm_states_vec, warm_block_idx, warm_values, warm_dvalues = _prepare_buffers(
        warmup_states, args.n_ops, n_dim
    )
    interpolator_uniform.evaluate_with_derivatives(
        warm_states_vec, warm_block_idx, warm_values, warm_dvalues
    )
    interpolator_nonuniform.evaluate_with_derivatives(
        warm_states_vec, warm_block_idx, warm_values, warm_dvalues
    )

    (
        n_states,
        states_vec,
        block_idx,
        values_buf,
        dvalues_buf,
    ) = _prepare_buffers(states, args.n_ops, n_dim)

    uniform_time = _time_interpolator(
        interpolator_uniform,
        states_vec,
        block_idx,
        values_buf,
        dvalues_buf,
        args.repeats,
    )
    uniform_values = np.asarray(values_buf).copy()
    uniform_dvalues = np.asarray(dvalues_buf).copy()

    nonuniform_time = _time_interpolator(
        interpolator_nonuniform,
        states_vec,
        block_idx,
        values_buf,
        dvalues_buf,
        args.repeats,
    )
    nonuniform_values = np.asarray(values_buf).copy()
    nonuniform_dvalues = np.asarray(dvalues_buf).copy()

    throughput_uniform = n_states / uniform_time
    throughput_nonuniform = n_states / nonuniform_time
    slowdown = uniform_time / nonuniform_time
    value_diff = float(np.max(np.abs(uniform_values - nonuniform_values)))
    deriv_diff = float(np.max(np.abs(uniform_dvalues - nonuniform_dvalues)))

    print("=== Uniform vs. Non-uniform interpolation benchmark ===")
    print(f"Dimensions: {n_dim}, operators: {args.n_ops}, states per run: {n_states}")
    print(
        f"Uniform grid    : {uniform_time:.6f} s ({throughput_uniform:,.0f} states/s)"
    )
    print(
        f"Non-uniform grid: {nonuniform_time:.6f} s "
        f"({throughput_nonuniform:,.0f} states/s)"
    )
    print(f"Relative throughput (uniform / non-uniform): {slowdown:.3f}x")
    print(
        "Output agreement (should be ~1e-12): "
        f"values max |diff| = {value_diff:.3e}, "
        f"derivatives max |diff| = {deriv_diff:.3e}"
    )


if __name__ == "__main__":
    main()
