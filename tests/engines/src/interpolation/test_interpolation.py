import numpy as np
from darts.engines import *

class Linear(operator_set_evaluator_iface):
    def __init__(self, n_dim, n_ops):
        super().__init__()
        self.n_dim = n_dim
        self.n_ops = n_ops
        self.A = np.arange((n_dim + 1) * n_ops, dtype=np.float64).reshape((n_ops, n_dim + 1))

    def evaluate(self, state: value_vector, values: value_vector):
        vec_state_as_np = np.asarray(state)
        vec_values_as_np = np.asarray(values)
        vec_values_as_np[:] = (self.A[:,:self.n_dim].dot(vec_state_as_np) + self.A[:,self.n_dim]).flatten()
        return 0

    def evaluate_with_derivative(self):
        return self.A[:,:self.n_dim].flatten()

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

    def evaluate_with_derivative(self, state: value_vector, values: value_vector, dvalues: value_vector):
        vec_state_as_np = np.asarray(state)
        vec_values_as_np = np.asarray(values)
        vec_values_as_np[:] = np.prod(np.sin(2 * np.pi * vec_state_as_np))
        vec_dvalues_as_np = np.asarray(dvalues)
        diff = np.array([2 * np.pi * np.cos(2 * np.pi * vec_state_as_np[i]) * np.prod(np.sin(2 * np.pi * \
                            np.concatenate([vec_state_as_np[:i], vec_state_as_np[i+1:]]))) for i in range(vec_state_as_np.size)])
        vec_dvalues_as_np[:] = np.tile(diff, self.n_ops)
        return 0

def get_interpolator_name(algorithm, mode, platform, precision, n_dims, n_ops):
    itor_name = "%s_%s_%s_interpolator_l_%s_%d_%d" % (algorithm,
                                                      mode,
                                                      platform,
                                                      precision,
                                                      n_dims,
                                                      n_ops)
    return itor_name
def test_interpolator_convergence(itor_type, itor_mode, n_dim, is_barycentric: bool = None, norm = None):
    zero = 1.e-9
    n_ops = 6 * n_dim + 12
    axes_min = n_dim * [-1 - zero]
    axes_max = n_dim * [1 + zero]
    evaluator = Nonlinear(n_dim, n_ops)
    itor_name = get_interpolator_name(itor_type, itor_mode, 'cpu', 'd', n_dim, n_ops)
    resolutions = [n_dim * [8], n_dim * [32], n_dim * [128]]

    # generate random states
    n_states = n_dim * [4]
    n_states_plain = np.prod(n_states)
    states = value_vector(np.random.uniform(low=axes_min,high=axes_max, size=n_states + [n_dim]).flatten())
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
        evaluator.evaluate_with_derivative(states_np[i * n_dim:(i + 1) * n_dim], buf, dbuf)
        true_values_np[i * n_ops:(i + 1) * n_ops] = np.asarray(buf)
        true_dvalues_np[i * n_ops * n_dim:(i + 1) * n_ops * n_dim] = np.asarray(dbuf)

    # calculate interpolated values with interpolators of multiple resolutions
    diff = np.zeros((2, len(resolutions)))
    for i in range(len(resolutions)):
        # initialize interpolator
        if itor_type == 'linear':
            itor = eval(itor_name)(evaluator, index_vector(resolutions[i]), value_vector(axes_min), value_vector(axes_max), is_barycentric)
        else:
            itor = eval(itor_name)(evaluator, index_vector(resolutions[i]), value_vector(axes_min), value_vector(axes_max))
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
    success = (orders[0] > 1.6) and (orders[1] > 0.6)

    if success:
        test_status = 'OK'
    else:
        test_status = 'FAILED'
    # print(diff)
    if itor_type == 'linear' and is_barycentric:
        print(f'{itor_type} {itor_mode} barycentric interpolation with Delaunay triangulation (n_dim={n_dim}): {test_status}')
    elif itor_type == 'linear' and not is_barycentric:
        print(f'{itor_type} {itor_mode} interpolation with standard triangulation (n_dim={n_dim}): {test_status}')
    else:
        print(f'{itor_type} {itor_mode} interpolation (n_dim={n_dim}): {test_status}')
    # print('Conv. order: val = ' + str(orders[0]) + ', der = ' + str(orders[1]))

    assert success, 'Conv. order: val = ' + str(orders[0]) + ', der = ' + str(orders[1])


def test_linearity_preservation(itor_type, itor_mode, n_dim, is_barycentric: bool = None):
    zero = 1.e-9
    n_ops = 6 * n_dim + 12
    n_axes_points = n_dim * [128]
    axes_min = [1] + (n_dim - 1) * [zero]
    axes_max = [300] + (n_dim - 1) * [1. - zero]
    evaluator = Linear(n_dim, n_ops)

    # initialize interpolator
    itor_name = get_interpolator_name(itor_type, itor_mode, 'cpu', 'd', n_dim, n_ops)
    if itor_type == 'linear':
        itor = eval(itor_name)(evaluator, index_vector(n_axes_points), value_vector(axes_min), value_vector(axes_max),
                               is_barycentric)
    else:
        itor = eval(itor_name)(evaluator, index_vector(n_axes_points), value_vector(axes_min), value_vector(axes_max))
    timer = timer_node()
    itor.init()
    itor.init_timer_node(timer)

    # generate random states
    n_states = n_dim * [4]
    n_states_plain = np.prod(n_states)
    states = value_vector(np.random.uniform(low=axes_min, high=axes_max, size=n_states + [n_dim]).flatten())
    block_idx = np.arange(n_states_plain).astype(np.int32)

    # allocate memory
    values = value_vector(np.zeros(n_ops * n_states_plain))
    dvalues = value_vector(np.zeros(n_dim * n_ops * n_states_plain))

    # interpolate
    itor.evaluate_with_derivatives(states, index_vector(block_idx), values, dvalues)

    # compare interpolated with true values
    states_np = np.asarray(states)
    values_np = np.asarray(values)
    dvalues_np = np.asarray(dvalues)

    true_values = value_vector(np.zeros(n_ops))
    true_values_np = np.asarray(true_values)
    true_dvalues_np = evaluator.evaluate_with_derivative()

    rtol = 1e-8
    atol = 1e-8
    for i in range(n_states_plain):
        evaluator.evaluate(states[i * n_dim:(i + 1) * n_dim], true_values)
        # check values
        success = np.isclose(true_values_np, values_np[i * n_ops:(i + 1) * n_ops], rtol=rtol, atol=atol).all() and \
                  np.isclose(true_dvalues_np, dvalues_np[i * n_ops * n_dim:(i + 1) * n_ops * n_dim], rtol=rtol, atol=atol).all()
        assert (success)

    if success:
        test_status = 'OK'
    else:
        test_status = 'FAILED'

    if itor_type == 'linear' and is_barycentric:
        print(f'{itor_type} {itor_mode} barycentric interpolation with Delaunay triangulation (n_dim={n_dim}): {test_status}')
    elif itor_type == 'linear' and not is_barycentric:
        print(f'{itor_type} {itor_mode} interpolation with standard triangulation (n_dim={n_dim}): {test_status}')
    else:
        print(f'{itor_type} {itor_mode} interpolation (n_dim={n_dim}): {test_status}')

print('Linearity-preserving tests for interpolators:')
test_linearity_preservation(itor_type='multilinear', itor_mode='adaptive', n_dim=4)
test_linearity_preservation(itor_type='linear', itor_mode='adaptive', n_dim=4, is_barycentric=False)
test_linearity_preservation(itor_type='linear', itor_mode='adaptive', n_dim=4, is_barycentric=True)

print('Convergence tests for interpolators:')
test_interpolator_convergence(itor_type='multilinear', itor_mode='adaptive', n_dim=4, norm=np.inf)
# test_interpolator_convergence(itor_type='multilinear', itor_mode='static', n_dim=4, norm=np.inf)
test_interpolator_convergence(itor_type='linear', itor_mode='adaptive', n_dim=4, is_barycentric=False, norm=np.inf)
test_interpolator_convergence(itor_type='linear', itor_mode='adaptive', n_dim=4, is_barycentric=True, norm=np.inf)