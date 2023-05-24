import numpy as np
from select_para import *
from darts.engines import property_evaluator_iface

class property_container(property_evaluator_iface):
    def __init__(self, phase_name, component_name, min_z):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        self.nph = len(phase_name)
        self.nc = len(component_name)
        self.component_name = component_name
        self.phase_name = phase_name
        self.min_z = min_z

        self.rock_comp = 1e-5
        self.p_ref = 1

        # Allocate (empty) evaluators
        self.density_ev = []
        self.viscosity_ev = []
        self.rel_perm_ev = []
        self.flash_ev = 0
        self.foam_STARS_FM_ev = []

class Flash(property_evaluator_iface):
    def __init__(self, components):
        super().__init__()
        self.components = components

    def flash(self, state):

        zc = state[1:]
        zc = np.append(zc, 1-sum(zc))

        ki = np.array([44.5, 2.05e-2])
        # ki = np.array([40, 2.47e-4])

        (x, y, V) = self.RR(zc, ki)

        return x, y, V

    def RR(self, zc, k):

        eps = 1e-12

        a = 1 / (1 - np.max(k)) + eps
        b = 1 / (1 - np.min(k)) - eps

        max_iter = 100 # use enough iterations for V to converge
        for i in range(1, max_iter):
            V = 0.5 * (a + b)

            r = sum(zc * (k - 1) / (V * (k - 1) + 1))

            if r > 0:
                a = V
            else:
                b = V

            if abs(r) < 1 * 10 ** -12:
                break

        x = zc / (V * (k - 1) + 1)
        y = k * x

        return (x, y, V)

class DensityBrine(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, x):
        x_co2 = x[0]
        rho_aq = 980 + x_co2 / 0.0125 * 4
        return rho_aq

class DensityVap(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure):
        p = pressure
        density = 733 * (1 + 1e-7 * (p - 1))

        return density

class ViscosityBrine(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self):

        mu_aq = 0.511

        return mu_aq

class ViscosityVap(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self):
        mu_g = 0.2611

        return mu_g

class PhaseRelPerm(property_evaluator_iface):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase

        self.Swc = 0.2
        self.Sgr = 0.2
        if phase == "wat":
            self.kre = 0.2
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 4.2

        else:
            self.kre = 0.94
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = 1.3


    def evaluate(self, sat):

        if sat >= 1 - self.sr1:
            kr = self.kre

        elif sat <= self.sr:
            kr = 0

        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

            # if self.kre == 0.2:
            #     Se = (sat - self.Swc) / (1 - self.Swc)
            #     kr = Se**4
            # else:
            #     Se = (1 - sat - self.Swc) / (1 - self.Swc)
            #     Swa = 1 - self.Sgr
            #     Sea = (Swa - self.Swc) / (1 - self.Swc)
            #     krna = 0.4 * (1 - Sea ** 2) * (1 - Sea) ** 2
            #     C = krna
            #
            #     kr = 0.4 * (1 - Se ** 2) * (1 - Se) ** 2 - C
            #
            # if kr > 1:
            #     kr = 1
            # elif kr < 0:
            #     kr = 0

        return kr

class FMEvaluator(property_evaluator_iface):
    def __init__(self, foam_paras):
        super().__init__()

        foam = foam_paras
        self.fmmob = foam[0]
        self.fmdry = foam[1]
        self.epdry = foam[2]

    def evaluate(self, sg):
        water_sat = 1 - sg

        Fw = 0.5 + np.arctan(self.epdry * (water_sat - self.fmdry)) / np.pi

        FM = 1/(1 + self.fmmob * Fw)

        return FM