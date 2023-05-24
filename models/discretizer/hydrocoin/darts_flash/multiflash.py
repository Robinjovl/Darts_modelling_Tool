import numpy as np
import flash
import warnings
from .initial import InitialGuess


class Flash:
    def __init__(self, components, ions, phases):
        self.nc = len(components)
        self.ni = len(ions)
        self.ns = self.nc + self.ni
        self.np = len(phases)

        self.z = np.zeros(self.ns)

        self.flash: flash.Flash

        self.it = 0

    def evaluate(self, p, T, z):
        pass

    def results(self):
        nu = self.flash.nu
        X = self.flash.x

        x = np.zeros((self.np, self.ns))
        for j in range(self.np):
            x[j, :] = X[j*self.ns:(j+1)*self.ns]

        return nu, x

    def fugacity(self, p, T, nu, x):
        for j, v in enumerate(nu):
            if v > 0.:
                return self.flash.fugacity(p, T, x[j, :], j)


class Flash2(Flash):
    def __init__(self, components: list, ions: list, phases: list, eos_used: dict, comp_data: dict, init_guess_ev: InitialGuess):
        super().__init__(components, ions, phases)

        self.flash = flash.Flash2(components, ions, phases, eos_used, comp_data)
        self.init_guess_ev = init_guess_ev

    def evaluate(self, p, T, z, max_it=100):
        self.z = z

        # Evaluate initial guess
        lnK = self.init_guess_ev.evaluate(p, T, z)

        # Initialise flash class
        self.flash.init(p, T, z, lnK)

        # Converge to solution
        self.it = 0
        while True:
            # Perform successive substitution step
            self.converged = self.flash.ssi(rr_tol=1e-12, tol=1e-8)
            self.it += 1
            if self.converged or self.it > max_it:
                break

        return self.results()

    def results(self):
        nu, x = super().results()

        if nu[0] >= 0. and nu[1] >= 0.:
            if not self.converged:
                warnings.warn("Flash unable to converge")
            return nu, x
        else:
            if nu[0] < 0.:
                return np.array([0., 1.]), np.array([np.zeros(self.ns), self.z])
            else:
                return np.array([1., 0.]), np.array([self.z, np.zeros(self.ns)])


# class FlashN(Flash):
#     def __init__(self):
#         super().__init__()
#
#         self.flash = flash.FlashN()
#
#     def evaluate(self, p, T, z):
#         return
