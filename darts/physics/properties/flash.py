import abc
import numpy as np


class Flash:
    nu: []
    x: []

    def __init__(self, nph, nc, ni=0):
        self.nph = nph
        self.nc = nc
        self.ni = ni
        self.ns = nc + ni

    @abc.abstractmethod
    def evaluate(self, pressure, temperature, zc):
        pass

    def getnu(self):
        return self.nu

    def getx(self):
        return self.x


class SinglePhase(Flash):
    def __init__(self, nc):
        super().__init__(nph=1, nc=nc)

    def evaluate(self, pressure, temperature, zc):
        self.nu, self.x = np.array([1.]), np.array([zc])
        return 0


class ConstantK(Flash):
    def __init__(self, nc, ki, min_z=1e-11):
        super().__init__(nph=2, nc=nc)

        self.min_z = min_z
        self.K_values = np.array(ki)

    def evaluate(self, pressure, temperature, zc):
        self.nu, self.x = RR2(self.K_values, zc, self.min_z)
        return 0


from numba import jit
@jit(nopython=True)
def RR2(k, zc, eps):

    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps

    max_iter = 200  # use enough iterations for V to converge
    for i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
        if abs(r) < 1e-12:
            break

        if r > 0:
            a = V
        else:
            b = V

    if i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * (k - 1) + 1)
    y = k * x

    return [V, 1-V], [y, x]


class SolidFlash(Flash):
    def __init__(self, flash: Flash, nc_fl: int, np_fl: int, ni: int = 0, nc_sol: int = 0, np_sol: int = 0):
        super().__init__(np_fl, nc_fl, ni)
        self.flash = flash

        self.nc_fl = self.ns
        self.np_fl = self.nph
        self.nc_sol = nc_sol
        self.np_sol = np_sol

    def evaluate(self, pressure, temperature, zc):
        """Evaluate flash normalized for solids"""
        # Normalize compositions
        zc_sol = zc[self.nc_fl:]
        zc_sol_tot = np.sum(zc_sol)
        zc_norm = zc[:self.nc_fl]/(1.-zc_sol_tot)

        # Evaluate flash for normalized composition
        error_output = self.flash.evaluate(pressure, temperature, zc_norm)
        nu = np.array(self.flash.getnu())
        x = np.array(self.flash.getx())

        # Re-normalize solids and append to nu, x
        NU = np.zeros(self.np_fl + self.np_sol)
        X = np.zeros((self.np_fl + self.np_sol, self.nc_fl + self.nc_sol))
        for j in range(self.np_fl):
            NU[j] = nu[j] * (1.-zc_sol_tot)
            X[j, :self.nc_fl] = x[j, :]

        for j in range(self.np_sol):
            NU[self.np_fl+j] = zc_sol[j]
            X[self.np_fl+j, self.nc_fl+j] = 1.

        self.nu = NU
        self.x = X

        return error_output
