import abc
import numpy as np


class Flash:
    nu: []
    X: []

    def __init__(self, nph, nc, ni=0):
        self.nph = nph
        self.nc = nc
        self.ni = ni
        self.ns = nc + ni

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, zc):
        pass

    def get_flash_results(self):
        return self


class SinglePhase(Flash):
    def __init__(self, nc):
        super().__init__(nph=1, nc=nc)

    def evaluate(self, pressure, temperature, zc):
        self.nu, self.X = np.array([1.]), np.array([zc])
        return 0


class ConstantK(Flash):
    def __init__(self, nc, ki, eps=1e-11):
        super().__init__(nph=2, nc=nc)

        self.rr_eps = eps
        self.K_values = np.array(ki)

    def evaluate(self, pressure, temperature, zc):
        self.nu, self.X = RR2(self.K_values, zc, self.rr_eps)
        return 0


from numba import jit
@jit(nopython=True)
def RR2(k, zc, eps):

    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps
    k_minus_1 = k - 1

    max_iter = 200  # use enough iterations for V to converge
    tol = 1e-12  # convergence tolerance

    for i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * k_minus_1 / (V * k_minus_1 + 1))
        if abs(r) < tol:
            break

        if r > 0:
            a = V
        else:
            b = V

    if i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * k_minus_1 + 1)
    y = k * x

    return [V, 1-V], [y, x]
