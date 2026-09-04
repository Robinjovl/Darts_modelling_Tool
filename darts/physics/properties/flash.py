import abc

import numpy as np
from numba import jit


class Flash:
    def __init__(self, nph, nc, nsalt=0):
        self.nph = nph
        self.nc = nc
        self.nsalt = nsalt
        self.ns = nc + nsalt

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


class KineticsFlashWrapper(Flash):
    """
    KineticsFlashWrapper class is a wrapper around a flash of equilibrium components/phases and normalized kinetic components/phases
    This is used in a formulation where the kinetic components are regular components with mole fractions in z, just do not flow
    It is a composition of a Flash object.
    During evaluate(), it normalizes equilibrium composition, evaluates Flash and renormalizes for kinetic components
    """

    def __init__(
        self,
        flash: Flash,
        nc_eq: int,
        np_eq: int,
        nsalt: int = 0,
        nc_kin: int = 0,
        np_kin: int = 0,
    ):
        """
        Constructor of KineticsFlashWrapper

        :param flash: Flash object for fluid components/phases
        :param nc_eq: Number of equilibrium components
        :param np_eq: Number of equilibrium phases
        :param nsalt: Number of ions
        :param nc_kin: Number of kinetic components
        :param np_kin: Number of kinetic phases
        """
        super().__init__(np_eq, nc_eq, nsalt=nsalt)
        self.flash = flash

        self.nc_eq = self.ns
        self.np_eq = self.nph
        self.nc_kin = nc_kin
        self.np_kin = np_kin

    def evaluate(self, pressure, temperature, zc):
        """Evaluate flash normalized for kinetic components.

        Normalizes the fluid part of ``zc`` (kinetic components removed), evaluates the wrapped
        fluid flash, then re-appends the kinetic phase fractions to ``nu``/``X``.

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
        zc_kin = zc[self.nc_eq :]
        zc_kin_tot = np.sum(zc_kin)
        zc_norm = zc[: self.nc_eq] / (1.0 - zc_kin_tot)

        # Evaluate flash for normalized composition
        error_output = self.flash.evaluate(pressure, temperature, zc_norm)
        flash_results = self.flash.get_flash_results()
        nu = np.array(flash_results.nu)
        try:
            x = np.array(flash_results.X).reshape(self.np_eq, self.nc_eq)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1
            # failed flash left X mis-shaped; keep going with NaN phase
            # compositions so the error is detectable downstream instead of
            # crashing on the undefined local
            x = np.full((self.np_eq, self.nc_eq), np.nan)

        # Re-normalize kinetic components and append to nu, x
        NU = np.zeros(self.np_eq + self.np_kin)
        X = np.zeros((self.np_eq + self.np_kin, self.nc_eq + self.nc_kin))
        for j in range(self.np_eq):
            NU[j] = nu[j] * (1.0 - zc_kin_tot)
            X[j, : self.nc_eq] = x[j, :]

        for j in range(self.np_kin):
            NU[self.np_eq + j] = zc_kin[j]
            X[self.np_eq + j, self.nc_eq + j] = 1.0

        self.nu = NU
        self.X = X
        self.temperature = flash_results.temperature

        return error_output
