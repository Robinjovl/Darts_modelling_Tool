import numpy as np


def linear_interp_extrapolate(x_new, x, y):
    """Piecewise-linear interpolation with linear extrapolation beyond the table.

    Reproduces bit-for-bit what
    ``scipy.interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')(x_new)``
    returned up to SciPy 1.17.  SciPy 1.18 re-derived that evaluation in the de Boor
    (convex-combination) form (scipy/scipy#24282); the two forms are algebraically
    identical but differ by up to one ULP, which is enough to move a marginally
    converging simulation onto a different timestep sequence.  Evaluating the
    interpolant here keeps DARTS results independent of the installed SciPy version.

    :param x_new: Points to evaluate the interpolant at (scalar or array)
    :param x: Table abscissae; sorted ascending internally, as interp1d does
    :param y: Table ordinates, same length as ``x``
    :returns: Interpolated values, shaped like ``x_new``
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)

    if np.any(np.diff(x) < 0.0):
        order = np.argsort(x, kind="mergesort")
        x, y = x[order], y[order]

    # Clipping to [1, len(x) - 1] picks the first (last) interval for points below
    # (above) the table, which is what makes the evaluation extrapolate linearly.
    idx = np.searchsorted(x, x_new).clip(1, len(x) - 1).astype(int)
    lo, hi = idx - 1, idx
    slope = (y[hi] - y[lo]) / (x[hi] - x[lo])
    return slope * (x_new - x[lo]) + y[lo]


# Table0-based interpolation procedure
class TableInterpolation:
    def LinearInterP(self, table, x_val, x_index, y_index):
        num = len(table)
        y_val = 0
        for i in range(num):
            if (x_val - table[i][x_index]) <= 0:
                y_val = table[i][y_index] + (x_val - table[i][x_index]) / (
                    table[i][x_index] - table[i - 1][x_index]
                ) * (table[i][y_index] - table[i - 1][y_index])
                break
        return y_val

    def LinearExtraP(self, table, x_val, x_index, y_index):
        num = len(table)
        y_val = 0
        if x_val < table[0][x_index]:
            y_val = table[0][y_index] + (x_val - table[0][x_index]) / (
                table[1][x_index] - table[0][x_index]
            ) * (table[1][y_index] - table[0][y_index])
        elif x_val > table[num - 1][x_index]:
            y_val = table[num - 1][y_index] + (x_val - table[num - 1][x_index]) / (
                table[num - 1][x_index] - table[num - 2][x_index]
            ) * (table[num - 1][y_index] - table[num - 2][y_index])

        return y_val

    def SCALExtraP(self, table, x_val, x_index, y_index):
        num = len(table)
        y_val = 0
        if x_val < table[0][x_index]:
            y_val = table[0][y_index]
        elif x_val > table[num - 1][x_index]:
            y_val = table[num - 1][y_index]

        return y_val

        # extrapolate properties if Rs>Rs_max_table

    def SatExtrapolation(self, table, x_val, x_index, y_index, lnum):
        y_val = table[lnum][y_index] + (x_val - table[lnum][x_index]) / (
            table[lnum][x_index] - table[lnum - 1][x_index]
        ) * (table[lnum][y_index] - table[lnum - 1][y_index])

        return y_val
