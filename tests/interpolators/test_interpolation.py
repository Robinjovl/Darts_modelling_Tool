import functools

import numpy as np
import pytest

from darts.engines import *
from darts.interpolators import *
from darts.physics.base.parallel_evaluator import (
    ParallelEvaluator,
    SharedEvaluatorPool,
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


def get_interpolator_name(algorithm, platform, precision, n_dims, n_ops):
    # Letterless naming: the index-type template parameter was dropped from the
    # interpolators, so exposed names carry no _i_/_l_ index-type letter. The
    # interpolation-mode token is gone too -- !280 removed the static interpolators,
    # so interpolation is always adaptive and the names no longer spell it out.
    itor_name = f"{algorithm}_{platform}_interpolator_{precision}_{n_dims}_{n_ops}"
    return itor_name


def get_interpolator_class(algorithm, platform, precision, n_dims, n_ops):
    itor_name = get_interpolator_name(algorithm, platform, precision, n_dims, n_ops)
    itor_cls = globals().get(itor_name)
    if itor_cls is None:
        pytest.fail(f'{itor_name} is not exposed in darts.interpolators')
    return itor_name, itor_cls


@pytest.mark.parametrize(
    "itor_type, n_dim, is_barycentric, norm",
    [
        ("multilinear", 4, None, np.inf),
        ("linear", 4, False, np.inf),
        ("linear", 4, True, np.inf),
    ],
)
def test_interpolator_convergence(itor_type, n_dim, is_barycentric, norm):
    zero = 1.0e-9
    n_ops = 6 * n_dim + 17
    axes_min = n_dim * [-1 - zero]
    axes_max = n_dim * [1 + zero]
    evaluator = Nonlinear(n_dim, n_ops)
    _, itor_cls = get_interpolator_class(itor_type, 'cpu', 'd', n_dim, n_ops)
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
        # initialize interpolator. New adaptive ctor is (evaluator, axes_origin,
        # axes_step); derive per-axis step from this resolution so the convergence
        # sweep still refines the cell size as before.
        axes_step = [
            (axes_max[d] - axes_min[d]) / (resolutions[i][d] - 1) for d in range(n_dim)
        ]
        if itor_type == 'linear':
            itor = itor_cls(
                evaluator,
                value_vector(axes_min),
                value_vector(axes_step),
                is_barycentric,
            )
        else:
            itor = itor_cls(
                evaluator,
                value_vector(axes_min),
                value_vector(axes_step),
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
            f'{itor_type} barycentric interpolation with Delaunay triangulation (n_dim={n_dim}): {test_status}'
        )
    elif itor_type == 'linear' and not is_barycentric:
        print(
            f'{itor_type} interpolation with standard triangulation (n_dim={n_dim}): {test_status}'
        )
    else:
        print(f'{itor_type} interpolation (n_dim={n_dim}): {test_status}')
    # print('Conv. order: val = ' + str(orders[0]) + ', der = ' + str(orders[1]))

    assert success, 'Conv. order: val = ' + str(orders[0]) + ', der = ' + str(orders[1])


@pytest.mark.parametrize(
    "itor_type, n_dim, is_barycentric",
    [
        ("multilinear", 4, None),
        ("linear", 4, False),
        ("linear", 4, True),
    ],
)
def test_linearity_preservation(itor_type, n_dim, is_barycentric):
    zero = 1.0e-9
    n_ops = 6 * n_dim + 17
    n_axes_points = n_dim * [128]
    axes_min = [1] + (n_dim - 1) * [zero]
    axes_max = [300] + (n_dim - 1) * [1.0 - zero]
    evaluator = Linear(n_dim, n_ops)

    # initialize interpolator. New adaptive ctor is (evaluator, axes_origin,
    # axes_step); derive step from the (n_axes_points, axes_min, axes_max) window.
    _, itor_cls = get_interpolator_class(itor_type, 'cpu', 'd', n_dim, n_ops)
    axes_step = [
        (axes_max[d] - axes_min[d]) / (n_axes_points[d] - 1) for d in range(n_dim)
    ]
    if itor_type == 'linear':
        itor = itor_cls(
            evaluator,
            value_vector(axes_min),
            value_vector(axes_step),
            is_barycentric,
        )
    else:
        itor = itor_cls(
            evaluator,
            value_vector(axes_min),
            value_vector(axes_step),
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
            f'{itor_type} barycentric interpolation with Delaunay triangulation (n_dim={n_dim}): {test_status}'
        )
    elif itor_type == 'linear' and not is_barycentric:
        print(
            f'{itor_type} interpolation with standard triangulation (n_dim={n_dim}): {test_status}'
        )
    else:
        print(f'{itor_type} interpolation (n_dim={n_dim}): {test_status}')


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
    _, itor_cls = get_interpolator_class('multilinear', 'cpu', 'd', n_dim, n_ops)
    # New adaptive ctor is (evaluator, axes_origin, axes_step); derive step from the
    # legacy (n_axes_points, axes_min, axes_max) window so cell spacing is unchanged.
    axes_step = [
        (axes_max[d] - axes_min[d]) / (n_axes_points[d] - 1) for d in range(n_dim)
    ]
    itor = itor_cls(
        evaluator,
        value_vector(list(axes_min)),
        value_vector(axes_step),
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


@pytest.mark.parametrize("start_method", [None, "spawn"])
def test_parallel_evaluator(start_method, n_dim=4):
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


def test_shared_evaluator_pool(n_dim=4):
    """SharedEvaluatorPool must dispatch per-key to the right worker evaluator.

    Covers the multi-wrap pattern used by PhysicsBase._wrap_evaluators_parallel:
    one pool of n_workers processes, several ParallelEvaluator wrappers each
    routing batches through the shared pool with their own key.
    """
    n_ops = 6 * n_dim + 17
    factory_lin = functools.partial(Linear, n_dim, n_ops)
    factory_nlin = functools.partial(Nonlinear, n_dim, n_ops)

    sp = SharedEvaluatorPool(
        {
            ('reservoir_operators', 0): factory_lin,
            ('reservoir_operators', 1): factory_nlin,
            ('property_operators', 0): factory_lin,
        },
        n_workers=3,
    )
    pe_lin = ParallelEvaluator(
        evaluator_factory=factory_lin, shared_pool=sp, key=('reservoir_operators', 0)
    )
    pe_nlin = ParallelEvaluator(
        evaluator_factory=factory_nlin, shared_pool=sp, key=('reservoir_operators', 1)
    )
    pe_prop = ParallelEvaluator(
        evaluator_factory=factory_lin, shared_pool=sp, key=('property_operators', 0)
    )

    n_pts = 300
    rng = np.random.default_rng(8)
    states_np = rng.uniform(-1.0, 1.0, size=n_pts * n_dim)

    def per_point_ref(ev):
        ref = np.zeros(n_pts * n_ops)
        for i in range(n_pts):
            buf = value_vector(np.zeros(n_ops))
            ev.evaluate(
                value_vector(states_np[i * n_dim : (i + 1) * n_dim].copy()), buf
            )
            ref[i * n_ops : (i + 1) * n_ops] = np.asarray(buf)
        return ref

    ref_lin = per_point_ref(Linear(n_dim, n_ops))
    ref_nlin = per_point_ref(Nonlinear(n_dim, n_ops))

    out_lin = value_vector(np.zeros(n_pts * n_ops))
    out_nlin = value_vector(np.zeros(n_pts * n_ops))
    out_prop = value_vector(np.zeros(n_pts * n_ops))
    pe_lin.evaluate_batch(value_vector(states_np.copy()), n_pts, out_lin, n_ops)
    pe_nlin.evaluate_batch(value_vector(states_np.copy()), n_pts, out_nlin, n_ops)
    pe_prop.evaluate_batch(value_vector(states_np.copy()), n_pts, out_prop, n_ops)

    lin_ok = np.allclose(np.asarray(out_lin), ref_lin, rtol=0.0, atol=1e-12)
    nlin_ok = np.allclose(np.asarray(out_nlin), ref_nlin, rtol=0.0, atol=1e-12)
    prop_ok = np.allclose(np.asarray(out_prop), ref_lin, rtol=0.0, atol=1e-12)

    # Unknown key must raise eagerly (catches typos in wrap-target plumbing).
    try:
        ParallelEvaluator(
            evaluator_factory=factory_lin, shared_pool=sp, key=('bogus', None)
        )
        bad_key_ok = False
    except ValueError:
        bad_key_ok = True

    sp.shutdown()

    success = lin_ok and nlin_ok and prop_ok and bad_key_ok
    print(
        f'SharedEvaluatorPool multi-key dispatch: '
        f'{"OK" if success else "FAILED"} '
        f'(Linear={lin_ok}, Nonlinear={nlin_ok}, property={prop_ok}, '
        f'bad-key-rejected={bad_key_ok})'
    )
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
        pytest.skip(
            'single-threaded build — no OpenMP thread control '
            '(set/get_num_threads absent)'
        )

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
    import sys

    sys.exit(pytest.main([__file__, '-v']))
