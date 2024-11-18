from darts_interpolator import DartsInterpolator
import numpy as np

# test evaluator function
def func_norm(my_vars, values):
    '''
    :param my_vars: list of arguments
    :param values: storage for output (only the first value is used here)
    :return:
    '''
    values[0] = 0.
    for i in my_vars:
        values[0] += i**2
    values[0] = np.sqrt(values[0])
    return values

# test precompiled itors for different dimensions
for n_dim in [2, 9]:
    n_interp_points_per_dim = 10
    axes_points = [n_interp_points_per_dim] * n_dim
    axes_min = [0] * n_dim
    axes_max = [1] * n_dim

    norm_itor = DartsInterpolator(func_norm, axes_points=axes_points, axes_min=axes_min, axes_max=axes_max, amount_of_int=n_dim)
    point = (np.arange(n_dim) / n_dim).tolist()  # generate a vector with values from 0 to 1

    exact_value = func_norm(point, [0]*n_dim)

    interp_value = norm_itor.interpolate_point(point)
    print('n_dim =', n_dim, 'exact=', '{:.3}'.format(exact_value[0]), 'interp=', '{:.3}'.format(interp_value[0]),
          'interpolation error =', '{:.3}'.format((interp_value[0] - exact_value[0])/exact_value[0]*100), '%')

    approx_deriv = norm_itor.interpolate_point_with_derivatives(point)
    #print(approx_deriv)

