import numpy as np
import pandas as pd
from darts.engines import *
import matplotlib.pyplot as plt
import seaborn as sns


class test_itor:
    def __init__(self, n_dims: int, n_ops: int,
                 axes_n_points: index_vector, axes_min: value_vector, axes_max: value_vector,
                 type: str = 'multilinear', mode: str = 'adaptive', version: str = '',
                 platform: str = 'cpu', precision: str = 'd', index: str = 'i'):
        # create a darts wrapper for function
        class dummy_func(operator_set_evaluator_iface):
            def __init__(self, n_dims: int, n_ops: int):
                super().__init__()
                self.n_dims = n_dims
                self.n_ops = n_ops

            def evaluate(self, state, values):
                values[0] = np.sum(state)
                for i in range(1, n_ops):
                    values[i] = values[i - 1] * values[0]
                return 0

        # create the instance of the wrapper
        evaluator = dummy_func(n_dims, n_ops)
        self.n_dims = n_dims
        self.n_ops = n_ops
        self.type = type
        self.mode = mode
        self.version = version
        self.platform = platform
        self.precision = precision
        self.index = index

        # verify then inputs are valid
        assert len(axes_n_points) == n_dims
        assert len(axes_min) == n_dims
        assert len(axes_max) == n_dims
        for n_p in axes_n_points:
            assert n_p > 1

        self.timer = timer_node()
        self.timer.node['init'] = timer_node()
        self.reset_timing()
        # calculate ibject name using 32 bit index type (i)
        itor_name = "%s_%s%s_%s_interpolator_%s_%s_%d_%d" % (type,
                                                             mode, version,
                                                             platform,
                                                             index,
                                                             precision,
                                                             n_dims,
                                                             n_ops)
        print("Creating %s..." % itor_name)
        self.name = itor_name

        self.timer.node['init'].start()

        self.itor = eval(itor_name)(evaluator, axes_n_points, axes_min, axes_max)
        self.itor.init_timer_node(self.timer)
        self.itor.init()
        self.timer.node['init'].stop()

    def prepare_to_interpolate(self, n_states: int):
        self.n_states = n_states
        self.block_idx = index_vector(np.arange(n_states, dtype=np.int32))

        # values should fit single value per point
        self.values = value_vector([0] * n_states * self.n_ops)

        # derivatives should fit self.n_dim values per point
        self.derivatives = value_vector([0] * n_states * self.n_dims * self.n_ops)

    def interpolate_array(self, X):
        # interpolate and shape the result
        assert (self.n_states * self.n_dims == len(X))
        self.itor.evaluate_with_derivatives(X, self.block_idx, self.values, self.derivatives)
        self.n_interpolations += 1
        self.min_interpolation_time = min(self.min_interpolation_time, self.get_interpolation_time())
        self.max_interpolation_time = max(self.max_interpolation_time, self.get_interpolation_time())
        self.total_interpolation_time += self.get_interpolation_time()
        self.avg_interpolation_time = self.total_interpolation_time / self.n_interpolations
        self.timer.reset_recursive()
        return np.array(self.values, copy=False)

    def validate_last_result(self, reference):
        return np.allclose(reference, self.values)

    def get_init_time(self):
        return self.timer.node['init'].get_timer()

    def get_interpolation_time(self):
        if 'gpu' in self.name:
            return self.timer.node['gpu interpolation'].get_timer()
        else:
            return self.timer.get_timer()

    def reset_timing(self):
        self.timer.reset_recursive()
        self.min_interpolation_time = 9999
        self.max_interpolation_time = 0
        self.avg_interpolation_time = 0
        self.total_interpolation_time = 0
        self.n_interpolations = 0


def make_itors(n_dims=2, n_ops=0, n_points=64, min=0, max=1):
    axes_n_points = index_vector([n_points] * n_dims)
    axes_min = value_vector([min] * n_dims)
    axes_max = value_vector([max] * n_dims)

    # set_num_threads(n_threads)
    redirect_darts_output('')

    if n_ops == 0:
        n_ops = 2 * n_dims

    itors = []

    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'static', '', 'gpu', 'd', 'i'))
    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'static', '2', 'gpu', 'd', 'i'))
    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'static', '', 'gpu', 's', 'i'))
    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'static', '2', 'gpu', 's', 'i'))
    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'adaptive', '3', 'gpu', 'd', 'i'))
    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'adaptive', '3', 'gpu', 's', 'i'))

    return itors
