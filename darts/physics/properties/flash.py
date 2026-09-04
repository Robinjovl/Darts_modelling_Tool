import abc

import numpy as np
from numba import jit


class Flash:
    def __init__(self, nph, nc, ni=0):
        self.nph = nph
        self.nc = nc
        self.ni = ni
        self.ns = nc + ni

        self.nu = []
        self.X = []
        self.temperature: float = 0.0

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
    def __init__(self, nc, ki, eps=1e-11, use_dartsflash: bool = False):
        """
        Constant K-value flash, option to make use of DARTS-flash RR solver.
        K-values are defined as Ki = yi/xi, with the y-phase returned as phase 0.
        The DARTS-flash RR solver defines K-values with phase 0 as reference phase
        (Ki = xi1/xi0), so K-values are inverted internally when use_dartsflash is enabled.

        :param nc: Number of components
        :param ki: K-values per component (Ki = yi/xi)
        :param eps: Epsilon value for composition
        :param use_dartsflash: Use DARTS-flash RR solver, default is False
        """
        super().__init__(nph=2, nc=nc)

        self.rr_eps = eps
        self.K_values = np.array(ki)

        self.use_dartsflash = use_dartsflash
        if use_dartsflash:
            from dartsflash.libflash import RR_EqConvex2

            self.rr = RR_EqConvex2(nc=nc, min_z=eps, rr_tol=1e-12, max_iter=100)

    def evaluate(self, pressure, temperature, zc):
        if self.use_dartsflash:
            # darts-flash uses phase 0 as reference phase (Ki = xi1/xi0), so invert
            # the K-values to keep the y-phase as phase 0 in the flash output
            self.rr.solve_rr(zc, 1.0 / self.K_values, np.array([]))
            self.nu, self.X = self.rr.getnu(), self.rr.getx()
            self.X = np.array(self.X).reshape(2, self.nc)

        else:
            self.nu, self.X = RR2(self.K_values, zc, self.rr_eps)

        self.temperature = temperature
        return 0


@jit(nopython=True)
def RR2(k, zc, eps):
    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps
    k_minus_1 = k - 1

    max_iter = 200  # use enough iterations for V to converge
    tol = 1e-12  # convergence tolerance

    for _i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * k_minus_1 / (V * k_minus_1 + 1))
        if abs(r) < tol:
            break

        if r > 0:
            a = V
        else:
            b = V

    if _i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * k_minus_1 + 1)
    y = k * x

    return [V, 1 - V], [y, x]


class SolidFlash(Flash):
    """
    SolidFlash class is a wrapper around a flash of fluid components/phases and normalized solid that reacts kinetically
    This is used in a formulation where the solid is a regular component with mole fractions, just does not flow
    It is a composition of a Flash object.
    During evaluate(), it normalizes fluid composition, evaluates Flash and renormalizes
    """

    def __init__(
        self,
        flash: Flash,
        nc_fl: int,
        np_fl: int,
        ni: int = 0,
        nc_sol: int = 0,
        np_sol: int = 0,
    ):
        """
        Constructor of SolidFlash

        :param flash: Flash object for fluid components/phases
        :param nc_fl: Number of fluid components
        :param np_fl: Number of fluid phases
        :param ni: Number of ions
        :param nc_sol: Number of solid components
        :param np_sol: Number of solid phases
        """
        super().__init__(np_fl, nc_fl, ni)
        self.flash = flash

        self.nc_fl = self.ns
        self.np_fl = self.nph
        self.nc_sol = nc_sol
        self.np_sol = np_sol

    def evaluate(self, pressure, temperature, zc):
        """Evaluate flash normalized for solids.

        Normalizes the fluid part of ``zc`` (solids removed), evaluates the wrapped
        fluid flash, then re-appends the solid phase fractions to ``nu``/``X``.

        If the wrapped flash fails and leaves its phase compositions mis-shaped,
        the fluid part of ``X`` is filled with NaN and the error count is
        incremented instead of raising, so the caller (OBL point generation)
        can detect the failed supporting point downstream.

        :param pressure: Pressure at the evaluated point.
        :param temperature: Temperature (may be None for isothermal physics).
        :param zc: Overall composition including solid components.
        :return: Number of flash errors encountered (0 on success).
        """
        # Normalize compositions
        zc_sol = zc[self.nc_fl :]
        zc_sol_tot = np.sum(zc_sol)
        zc_norm = zc[: self.nc_fl] / (1.0 - zc_sol_tot)

        # Evaluate flash for normalized composition
        error_output = self.flash.evaluate(pressure, temperature, zc_norm)
        flash_results = self.flash.get_flash_results()
        nu = np.array(flash_results.nu)
        try:
            x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1
            # failed flash left X mis-shaped; keep going with NaN phase
            # compositions so the error is detectable downstream instead of
            # crashing on the undefined local
            x = np.full((self.np_fl, self.nc_fl), np.nan)

        # Re-normalize solids and append to nu, x
        NU = np.zeros(self.np_fl + self.np_sol)
        X = np.zeros((self.np_fl + self.np_sol, self.nc_fl + self.nc_sol))
        for j in range(self.np_fl):
            NU[j] = nu[j] * (1.0 - zc_sol_tot)
            X[j, : self.nc_fl] = x[j, :]

        for j in range(self.np_sol):
            NU[self.np_fl + j] = zc_sol[j]
            X[self.np_fl + j, self.nc_fl + j] = 1.0

        self.nu = NU
        self.X = X
        self.temperature = flash_results.temperature

        return error_output
