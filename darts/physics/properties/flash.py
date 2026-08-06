import abc

import numpy as np


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
    def __init__(self, nc, ki, eps=1e-11):
        super().__init__(nph=2, nc=nc)

        from dartsflash.libflash import RR_EqConvex2

        self.rr = RR_EqConvex2(nc=nc, min_z=eps, rr_tol=1e-12, max_iter=100)
        self.rr_eps = eps
        self.K_values = np.array(ki)

    def evaluate(self, pressure, temperature, zc):
        self.rr.solve_rr(zc, self.K_values)
        self.nu, self.X = self.rr.getnu(), self.rr.getx()
        self.temperature = temperature
        return 0


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
        """Evaluate flash normalized for solids"""
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
        self.flash_ev.evaluate(pressure, temperature, zc)
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
