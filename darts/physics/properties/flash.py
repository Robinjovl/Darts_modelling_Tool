import abc

import numpy as np


class Flash:
    nu: []
    X: []
    temperature: float

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
        self.nu, self.X = np.array([1.0]), np.array([zc])
        self.temperature = temperature
        return 0


class ConstantK(Flash):
    def __init__(self, nc, ki, eps=1e-11):
        super().__init__(nph=2, nc=nc)

        self.rr_eps = eps
        self.K_values = np.array(ki)

    def evaluate(self, pressure, temperature, zc):
        self.nu, self.X = RR2(self.K_values, zc, self.rr_eps)
        self.temperature = temperature
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

    return [V, 1 - V], [y, x]


class IonFlash(Flash):
    def __init__(
        self, flash_ev: Flash, nph: int, nc: int, ni: int, combined_ions: list = None
    ):
        super().__init__(nph, nc, ni)
        self.flash_ev = flash_ev
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, zc):
        # Uncombine ions into Na+ and Cl- mole fractions
        if self.combined_ions is not None:
            ion_weights = self.combined_ions / np.sum(self.combined_ions)
            zc = np.append(zc[:-1], [ion_weights[0] * zc[-1], ion_weights[1] * zc[-1]])
        nc_tot = len(zc)

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        error_output = self.flash_ev.evaluate(pressure, temperature, zc)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        self.X = np.empty(
            (
                self.nph,
                self.nc + 1 if self.combined_ions is not None else self.nc + self.ni,
            )
        )
        self.temperature = flash_results.temperature

        for j in range(self.nph):
            Xj = flash_results.X[j * nc_tot : (j + 1) * nc_tot]

            if self.combined_ions is not None:
                # Normal components +
                self.X[j, : self.nc] = Xj[: self.nc]

                # Sum ions
                self.X[j, self.nc] = np.sum(ion_weights * Xj[self.nc :])
            else:
                self.X[j, :] = Xj

        return 0
