import functools

import numpy as np

from darts.engines import *
from darts.interpolators import *
from darts.physics.base.parallel_evaluator import ParallelEvaluator


class Linear(operator_set_evaluator_iface):
    def __init__(self, n_dim, n_ops):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops
        self.A = np.arange((n_dim + 1) * n_ops, dtype=np.float64).reshape(
            (n_ops, n_dim + 1)
        )

    def evaluate(self, state: value_vector, values: value_vector):
        vec_state_as_np = np.asarray(state)
        vec_values_as_np = np.asarray(values)
        vec_values_as_np[:] = (
            self.A[:, : self.n_dim].dot(vec_state_as_np) + self.A[:, self.n_dim]
        ).flatten()
        return 0

    def evaluate_with_derivative(self):
        return self.A[:, : self.n_dim].flatten()


class Nonlinear(operator_set_evaluator_iface):
    def __init__(self, n_dim, n_ops):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops

    def evaluate(self, state: value_vector, values: value_vector):
        vec_state_as_np = np.asarray(state)
        vec_values_as_np = np.asarray(values)
        vec_values_as_np[:] = np.prod(np.sin(2 * np.pi * vec_state_as_np))
        return 0

    def evaluate_with_derivative(
        self, state: value_vector, values: value_vector, dvalues: value_vector
    ):
        vec_state_as_np = np.asarray(state)
        vec_values_as_np = np.asarray(values)
        vec_values_as_np[:] = np.prod(np.sin(2 * np.pi * vec_state_as_np))
        vec_dvalues_as_np = np.asarray(dvalues)
        diff = np.array(
            [
                2
                * np.pi
                * np.cos(2 * np.pi * vec_state_as_np[i])
                * np.prod(
                    np.sin(
                        2
                        * np.pi
                        * np.concatenate(
                            [vec_state_as_np[:i], vec_state_as_np[i + 1 :]]
                        )
                    )
                )
                for i in range(vec_state_as_np.size)
            ]
        )
        vec_dvalues_as_np[:] = np.tile(diff, self.n_ops)
        return 0


def get_interpolator_name(algorithm, mode, platform, precision, n_dims, n_ops):
    itor_name = (
        f"{algorithm}_{mode}_{platform}_interpolator_l_{precision}_{n_dims}_{n_ops}"
    )
    return itor_name


def test_interpolator_convergence(
    itor_type, itor_mode, n_dim, is_barycentric: bool = None, norm=None
):
    zero = 1.0e-9
    n_ops = 6 * n_dim + 17
    axes_min = n_dim * [-1 - zero]
    axes_max = n_dim * [1 + zero]
    evaluator = Nonlinear(n_dim, n_ops)
    itor_name = get_interpolator_name(itor_type, itor_mode, 'cpu', 'd', n_dim, n_ops)
    resolutions = [n_dim * [8], n_dim * [32], n_dim * [128]]

    # generate random states
    n_states = n_dim * [4]
    n_states_plain = np.prod(n_states)
    states = value_vector(
        np.random.uniform(
            low=axes_min, high=axes_max, size=n_states + [n_dim]
        ).flatten()
    )
    states_np = np.asarray(states)
    block_idx = np.arange(n_states_plain).astype(np.int32)

    # allocate memory
    values = value_vector(np.zeros(n_ops * n_states_plain))
    dvalues = value_vector(np.zeros(n_dim * n_ops * n_states_plain))
    values_np = np.asarray(values)
    dvalues_np = np.asarray(dvalues)
    true_values = value_vector(np.zeros(n_ops * n_states_plain))
    true_dvalues = value_vector(np.zeros(n_ops * n_dim * n_states_plain))
    true_values_np = np.asarray(true_values)
    true_dvalues_np = np.asarray(true_dvalues)

    # calculate reference values with evaluator
    buf = value_vector(np.zeros(n_ops))
    dbuf = value_vector(np.zeros(n_ops * n_dim))
    for i in range(n_states_plain):
        evaluator.evaluate_with_derivative(
            states_np[i * n_dim : (i + 1) * n_dim], buf, dbuf
        )
        true_values_np[i * n_ops : (i + 1) * n_ops] = np.asarray(buf)
        true_dvalues_np[i * n_ops * n_dim : (i + 1) * n_ops * n_dim] = np.asarray(dbuf)

    # calculate interpolated values with interpolators of multiple resolutions
    diff = np.zeros((2, len(resolutions)))
    for i in range(len(resolutions)):
        # initialize interpolator
        if itor_type == 'linear':
            itor = eval(itor_name)(
                evaluator,
                index_vector(resolutions[i]),
                value_vector(axes_min),
                value_vector(axes_max),
                is_barycentric,
            )
        else:
            itor = eval(itor_name)(
                evaluator,
                index_vector(resolutions[i]),
                value_vector(axes_min),
                value_vector(axes_max),
            )
        timer = timer_node()
        itor.init()
        itor.init_timer_node(timer)

        # interpolate
        itor.evaluate_with_derivatives(states, index_vector(block_idx), values, dvalues)

        # calculate mismatch
        diff[0, i] = np.linalg.norm(values_np - true_values_np, ord=norm)
        diff[1, i] = np.linalg.norm(dvalues_np - true_dvalues_np, ord=norm)

    dx = 2 / np.array([res[0] for res in resolutions])
    orders = np.diff(np.log(diff), axis=1)[:, -1] / np.diff(np.log(dx))
    success = (orders[0] > 1.4) and (orders[1] > 0.4)

    if success:
        test_status = 'OK'
    else:
        test_status = 'FAILED'
    # print(diff)
    if itor_type == 'linear' and is_barycentric:
        print(
            f'{itor_type} {itor_mode} barycentric interpolation with Delaunay triangulation (n_dim={n_dim}): {test_status}'
        )
    elif itor_type == 'linear' and not is_barycentric:
        print(
            f'{itor_type} {itor_mode} interpolation with standard triangulation (n_dim={n_dim}): {test_status}'
        )
    else:
        print(f'{itor_type} {itor_mode} interpolation (n_dim={n_dim}): {test_status}')
    # print('Conv. order: val = ' + str(orders[0]) + ', der = ' + str(orders[1]))

    assert success, 'Conv. order: val = ' + str(orders[0]) + ', der = ' + str(orders[1])


def test_linearity_preservation(
    itor_type, itor_mode, n_dim, is_barycentric: bool = None
):
    zero = 1.0e-9
    n_ops = 6 * n_dim + 17
    n_axes_points = n_dim * [128]
    axes_min = [1] + (n_dim - 1) * [zero]
    axes_max = [300] + (n_dim - 1) * [1.0 - zero]
    evaluator = Linear(n_dim, n_ops)

    # initialize interpolator
    itor_name = get_interpolator_name(itor_type, itor_mode, 'cpu', 'd', n_dim, n_ops)
    if itor_type == 'linear':
        itor = eval(itor_name)(
            evaluator,
            index_vector(n_axes_points),
            value_vector(axes_min),
            value_vector(axes_max),
            is_barycentric,
        )
    else:
        itor = eval(itor_name)(
            evaluator,
            index_vector(n_axes_points),
            value_vector(axes_min),
            value_vector(axes_max),
        )
    timer = timer_node()
    itor.init()
    itor.init_timer_node(timer)

    # generate random states
    n_states = n_dim * [4]
    n_states_plain = np.prod(n_states)
    states = value_vector(
        np.random.uniform(
            low=axes_min, high=axes_max, size=n_states + [n_dim]
        ).flatten()
    )
    block_idx = np.arange(n_states_plain).astype(np.int32)

    # allocate memory
    values = value_vector(np.zeros(n_ops * n_states_plain))
    dvalues = value_vector(np.zeros(n_dim * n_ops * n_states_plain))

    # interpolate
    itor.evaluate_with_derivatives(states, index_vector(block_idx), values, dvalues)

    # compare interpolated with true values
    values_np = np.asarray(values)
    dvalues_np = np.asarray(dvalues)

    true_values = value_vector(np.zeros(n_ops))
    true_values_np = np.asarray(true_values)
    true_dvalues_np = evaluator.evaluate_with_derivative()

    rtol = 1e-8
    atol = 1e-8
    for i in range(n_states_plain):
        evaluator.evaluate(states[i * n_dim : (i + 1) * n_dim], true_values)
        # check values
        success = (
            np.isclose(
                true_values_np,
                values_np[i * n_ops : (i + 1) * n_ops],
                rtol=rtol,
                atol=atol,
            ).all()
            and np.isclose(
                true_dvalues_np,
                dvalues_np[i * n_ops * n_dim : (i + 1) * n_ops * n_dim],
                rtol=rtol,
                atol=atol,
            ).all()
        )
        assert success

    if success:
        test_status = 'OK'
    else:
        test_status = 'FAILED'

    if itor_type == 'linear' and is_barycentric:
        print(
            f'{itor_type} {itor_mode} barycentric interpolation with Delaunay triangulation (n_dim={n_dim}): {test_status}'
        )
    elif itor_type == 'linear' and not is_barycentric:
        print(
            f'{itor_type} {itor_mode} interpolation with standard triangulation (n_dim={n_dim}): {test_status}'
        )
    else:
        print(f'{itor_type} {itor_mode} interpolation (n_dim={n_dim}): {test_status}')


# ── Tests for the parallel operator update (MR297) ──────────────────────────
#
# These cover the two layers of concurrency added for parallel operator update:
#   * the C++ OpenMP three-phase adaptive interpolator, and
#   * the Python multiprocessing ParallelEvaluator / evaluate_batch path.
# See docs/for_developers/parallel_operators.md.


def _build_multilinear_adaptive(
    evaluator, n_dim, n_ops, n_axes_points, axes_min, axes_max
):
    """Construct and initialize a multilinear adaptive CPU interpolator.

    No timer node is attached on purpose: this exercises the null-timer path of
    the three-phase adaptive update (the ``if (this->timer)`` guards added in MR297).
    """
    itor_name = get_interpolator_name(
        'multilinear', 'adaptive', 'cpu', 'd', n_dim, n_ops
    )
    itor = eval(itor_name)(
        evaluator,
        index_vector(n_axes_points),
        value_vector(axes_min),
        value_vector(axes_max),
    )
    itor.init()
    return itor


def test_evaluate_batch_consistency(n_dim=4):
    """evaluate_batch() must reproduce per-point evaluate() exactly.

    Exercises the C++ default operator_set_evaluator_iface::evaluate_batch
    (Linear does not override it), used by the adaptive interpolator to
    materialize missing supporting points in one call.
    """
    n_ops = 6 * n_dim + 17
    evaluator = Linear(n_dim, n_ops)
    n_pts = 64
    rng = np.random.default_rng(1)
    states_np = rng.uniform(-1.0, 1.0, size=n_pts * n_dim)

    # per-point reference
    reference = np.zeros(n_pts * n_ops)
    for i in range(n_pts):
        buf = value_vector(np.zeros(n_ops))
        evaluator.evaluate(
            value_vector(states_np[i * n_dim : (i + 1) * n_dim].copy()), buf
        )
        reference[i * n_ops : (i + 1) * n_ops] = np.asarray(buf)

    # batch
    out = value_vector(np.zeros(n_pts * n_ops))
    evaluator.evaluate_batch(value_vector(states_np.copy()), n_pts, out, n_ops)

    success = np.allclose(np.asarray(out), reference, rtol=0.0, atol=1e-12)
    print(f'evaluate_batch vs per-point evaluate: {"OK" if success else "FAILED"}')
    assert success


def test_parallel_evaluator(n_dim=4, start_method=None):
    """ParallelEvaluator.evaluate_batch() must reproduce serial evaluation.

    The factory is functools.partial(Linear, ...) — a top-level picklable
    callable. Running with start_method='spawn' reproduces the Windows /
    macOS multiprocessing behaviour on any platform.
    """
    n_ops = 6 * n_dim + 17
    factory = functools.partial(Linear, n_dim, n_ops)
    par_eval = ParallelEvaluator(
        evaluator_factory=factory, n_workers=4, start_method=start_method
    )

    n_pts = 300
    rng = np.random.default_rng(2)
    states_np = rng.uniform(-1.0, 1.0, size=n_pts * n_dim)

    # serial reference
    serial = Linear(n_dim, n_ops)
    reference = np.zeros(n_pts * n_ops)
    for i in range(n_pts):
        buf = value_vector(np.zeros(n_ops))
        serial.evaluate(
            value_vector(states_np[i * n_dim : (i + 1) * n_dim].copy()), buf
        )
        reference[i * n_ops : (i + 1) * n_ops] = np.asarray(buf)

    out = value_vector(np.zeros(n_pts * n_ops))
    par_eval.evaluate_batch(value_vector(states_np.copy()), n_pts, out, n_ops)
    par_eval.shutdown()

    success = np.allclose(np.asarray(out), reference, rtol=0.0, atol=1e-12)
    label = start_method if start_method else 'default'
    print(f'ParallelEvaluator ({label} start method): {"OK" if success else "FAILED"}')
    assert success


def test_parallel_interpolator(n_dim=4):
    """Adaptive interpolator must give identical results whether its missing
    supporting points are evaluated serially or through a ParallelEvaluator.
    """
    zero = 1.0e-9
    n_ops = 4 * n_dim  # 16 for n_dim=4: an instantiated (N_DIMS, N_OPS) template
    n_axes_points = n_dim * [16]
    axes_min = n_dim * [-1 - zero]
    axes_max = n_dim * [1 + zero]
    n_states = 600
    rng = np.random.default_rng(3)
    states_np = rng.uniform(-1.0, 1.0, size=n_states * n_dim)
    block_idx = np.arange(n_states).astype(np.int32)

    def interpolate(evaluator):
        itor = _build_multilinear_adaptive(
            evaluator, n_dim, n_ops, n_axes_points, axes_min, axes_max
        )
        values = value_vector(np.zeros(n_ops * n_states))
        dvalues = value_vector(np.zeros(n_dim * n_ops * n_states))
        itor.evaluate_with_derivatives(
            value_vector(states_np.copy()), index_vector(block_idx), values, dvalues
        )
        return np.asarray(values).copy(), np.asarray(dvalues).copy()

    v_serial, d_serial = interpolate(Nonlinear(n_dim, n_ops))

    par_eval = ParallelEvaluator(
        evaluator_factory=functools.partial(Nonlinear, n_dim, n_ops), n_workers=4
    )
    v_par, d_par = interpolate(par_eval)
    par_eval.shutdown()

    success = np.allclose(v_serial, v_par, rtol=0.0, atol=1e-12) and np.allclose(
        d_serial, d_par, rtol=0.0, atol=1e-12
    )
    print(
        f'parallel interpolator (ParallelEvaluator supporting points): '
        f'{"OK" if success else "FAILED"}'
    )
    assert success


def test_interpolator_thread_consistency(n_dim=4):
    """The C++ three-phase adaptive interpolator must produce bit-identical
    results regardless of OMP_NUM_THREADS — every phase is either serial or
    operates on independent per-cell / per-hypercube data.

    Skipped on single-threaded (ST) builds — e.g. the open-DARTS-solvers (ODLS)
    build, which has no OpenMP and therefore does not expose set/get_num_threads.
    On an ST build the interpolator is serial anyway, so there is nothing to vary.
    """
    import darts.engines as _engines

    if not (
        hasattr(_engines, 'set_num_threads') and hasattr(_engines, 'get_num_threads')
    ):
        print(
            'interpolator thread-count consistency: SKIPPED '
            '(single-threaded build — no OpenMP thread control)'
        )
        return

    zero = 1.0e-9
    n_ops = 4 * n_dim  # 16 for n_dim=4: an instantiated (N_DIMS, N_OPS) template
    n_axes_points = n_dim * [16]
    axes_min = n_dim * [-1 - zero]
    axes_max = n_dim * [1 + zero]
    n_states = 800
    rng = np.random.default_rng(4)
    states_np = rng.uniform(-1.0, 1.0, size=n_states * n_dim)
    block_idx = np.arange(n_states).astype(np.int32)

    def interpolate(n_threads):
        set_num_threads(n_threads)
        itor = _build_multilinear_adaptive(
            Nonlinear(n_dim, n_ops), n_dim, n_ops, n_axes_points, axes_min, axes_max
        )
        values = value_vector(np.zeros(n_ops * n_states))
        dvalues = value_vector(np.zeros(n_dim * n_ops * n_states))
        itor.evaluate_with_derivatives(
            value_vector(states_np.copy()), index_vector(block_idx), values, dvalues
        )
        return np.asarray(values).copy(), np.asarray(dvalues).copy()

    n_threads_default = get_num_threads()
    try:
        v1, d1 = interpolate(1)
        v_many, d_many = interpolate(max(2, n_threads_default))
    finally:
        set_num_threads(n_threads_default)

    success = np.array_equal(v1, v_many) and np.array_equal(d1, d_many)
    print(
        f'interpolator thread-count consistency (1 vs '
        f'{max(2, n_threads_default)} threads): {"OK" if success else "FAILED"}'
    )
    assert success


if __name__ == '__main__':
    print('Linearity-preserving tests for interpolators:')
    test_linearity_preservation(itor_type='multilinear', itor_mode='adaptive', n_dim=4)
    test_linearity_preservation(
        itor_type='linear', itor_mode='adaptive', n_dim=4, is_barycentric=False
    )
    test_linearity_preservation(
        itor_type='linear', itor_mode='adaptive', n_dim=4, is_barycentric=True
    )

    print('Convergence tests for interpolators:')
    test_interpolator_convergence(
        itor_type='multilinear', itor_mode='adaptive', n_dim=4, norm=np.inf
    )
    # test_interpolator_convergence(itor_type='multilinear', itor_mode='static', n_dim=4, norm=np.inf)
    test_interpolator_convergence(
        itor_type='linear',
        itor_mode='adaptive',
        n_dim=4,
        is_barycentric=False,
        norm=np.inf,
    )
    test_interpolator_convergence(
        itor_type='linear',
        itor_mode='adaptive',
        n_dim=4,
        is_barycentric=True,
        norm=np.inf,
    )

    print('Parallel operator update tests:')
    test_evaluate_batch_consistency(n_dim=4)
    test_parallel_evaluator(n_dim=4, start_method=None)
    test_parallel_evaluator(n_dim=4, start_method='spawn')
    test_parallel_interpolator(n_dim=4)
    test_interpolator_thread_consistency(n_dim=4)
