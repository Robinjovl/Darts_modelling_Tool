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

    def evaluate_with_derivative(self):
        return self.A[:, : self.n_dim].flatten()


def _get_multilinear_name(n_dim, n_ops):
    return f"multilinear_adaptive_cpu_interpolator_l_d_{n_dim}_{n_ops}"


def _make_multilinear(evaluator, axes_points, axes_min, axes_max, axis_nodes=None):
    n_dim = len(axes_points)
    n_ops = evaluator.n_ops
    name = _get_multilinear_name(n_dim, n_ops)
    ctor = getattr(engines, name)
    if axis_nodes is None:
        interpolator = ctor(
            evaluator,
            index_vector(axes_points),
            value_vector(axes_min),
            value_vector(axes_max),
        )
    else:
        axis_nodes_vectors = [value_vector(nodes) for nodes in axis_nodes]
        interpolator = ctor(
            evaluator,
            index_vector(axes_points),
            value_vector(axes_min),
            value_vector(axes_max),
            axis_nodes_vectors,
        )
    interpolator.init()
    interpolator.init_timer_node(timer_node())
    return interpolator


def _evaluate_with_reference(interpolator, evaluator, states):
    n_states, n_dim = states.shape
    n_ops = evaluator.n_ops
    states_vec = value_vector(states.flatten())
    block_idx = index_vector(np.arange(n_states, dtype=np.int32))
    values = value_vector(np.zeros(n_ops * n_states))
    dvalues = value_vector(np.zeros(n_ops * n_dim * n_states))
    interpolator.evaluate_with_derivatives(states_vec, block_idx, values, dvalues)

    true_values = np.zeros(n_ops * n_states, dtype=np.float64)
    true_derivs = np.zeros(n_ops * n_dim * n_states, dtype=np.float64)
    buf = value_vector(np.zeros(n_ops))
    grad = evaluator.evaluate_with_derivative()
    for i in range(n_states):
        evaluator.evaluate(value_vector(states[i]), buf)
        true_values[i * n_ops : (i + 1) * n_ops] = np.asarray(buf)
        true_derivs[i * n_ops * n_dim : (i + 1) * n_ops * n_dim] = grad
    return (
        np.asarray(values),
        np.asarray(dvalues),
        true_values,
        true_derivs,
    )


def test_nonuniform_axes_linear_exact():
    n_dim = 3
    n_ops = 6 * n_dim + 3
    evaluator = Linear(n_dim, n_ops)
    axis_nodes = [
        [0.0, 0.02, 0.2, 0.5, 1.4],
        [2.0, 2.1, 2.5, 3.5],
        [-1.0, -0.5, 0.0, 0.01, 0.2, 0.5],
    ]
    axes_points = [len(a) for a in axis_nodes]
    axes_min = [a[0] for a in axis_nodes]
    axes_max = [a[-1] for a in axis_nodes]
    interpolator = _make_multilinear(
        evaluator, axes_points, axes_min, axes_max, axis_nodes=axis_nodes
    )

    rng = np.random.default_rng(42)
    states = rng.uniform(low=axes_min, high=axes_max, size=(10, n_dim))
    values, dvalues, true_values, true_derivs = _evaluate_with_reference(
        interpolator, evaluator, states
    )

    assert np.allclose(values, true_values, rtol=1e-10, atol=1e-10)
    assert np.allclose(dvalues, true_derivs, rtol=1e-10, atol=1e-10)


def test_nonuniform_uniform_equivalence():
    n_dim = 2
    n_ops = 12
    evaluator = Linear(n_dim, n_ops)
    axes_points = [5, 4]
    axes_min = [0.0, -1.0]
    axes_max = [2.0, 3.0]
    uniform_nodes = [
        list(np.linspace(axes_min[0], axes_max[0], axes_points[0])),
        list(np.linspace(axes_min[1], axes_max[1], axes_points[1])),
    ]

    interpolator_default = _make_multilinear(evaluator, axes_points, axes_min, axes_max)
    interpolator_nodes = _make_multilinear(
        evaluator, axes_points, axes_min, axes_max, axis_nodes=uniform_nodes
    )

    rng = np.random.default_rng(7)
    states = rng.uniform(low=axes_min, high=axes_max, size=(6, n_dim))
    values_default, dvalues_default, _, _ = _evaluate_with_reference(
        interpolator_default, evaluator, states
    )
    values_nodes, dvalues_nodes, _, _ = _evaluate_with_reference(
        interpolator_nodes, evaluator, states
    )

    assert np.allclose(values_default, values_nodes, rtol=1e-12, atol=1e-12)
    assert np.allclose(dvalues_default, dvalues_nodes, rtol=1e-12, atol=1e-12)


def test_nonuniform_bounds_clamping():
    n_dim = 1
    n_ops = 6
    evaluator = Linear(n_dim, n_ops)
    axis_nodes = [[-2.0, -0.5, 0.0, 0.5, 2.0]]
    axes_points = [len(axis_nodes[0])]
    axes_min = [-2.0]
    axes_max = [2.0]
    interpolator = _make_multilinear(
        evaluator, axes_points, axes_min, axes_max, axis_nodes=axis_nodes
    )

    states = np.array([[-3.5], [3.8]], dtype=np.float64)
    values, _, true_values, _ = _evaluate_with_reference(
        interpolator, evaluator, states
    )
    boundary_states = np.array([[axes_min[0]], [axes_max[0]]], dtype=np.float64)
    boundary_values, _, _, _ = _evaluate_with_reference(
        interpolator, evaluator, boundary_states
    )

    assert np.allclose(values[0], boundary_values[0], rtol=1e-10, atol=1e-10)
    assert np.allclose(values[1], boundary_values[1], rtol=1e-10, atol=1e-10)
