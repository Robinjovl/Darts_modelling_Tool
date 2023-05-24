import numpy as np
from darts.engines import property_evaluator_iface

# from src.cubic_main import *
# from src.Binary_Interactions import *
# from src.flash_funcs import *

class flash_3phase():
    def __init__(self, components, T):
        self.components = components
        self.T = T
        mixture = Mix(components)
        binary = Kij(components)
        mixture.kij_cubic(binary)

        self.eos = preos(mixture, mixrule='qmr', volume_translation=True)

    def evaluate(self, p, zc):
        nu, x, status = multiphase_flash(self.components, zc, self.T, p, self.eos)

        return x, nu


# Uncomment these two lines if numba package is installed and make things happen much faster:
# from numba import jit
# @jit(nopython=True)
def RR_func(zc, k, eps):

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

    return (x, y, V)


class Flash:
    def __init__(self, components, ki, min_z=1e-11):
        self.components = components
        self.nc = len(components)
        self.min_z = min_z
        self.ki = np.array(ki)

    def evaluate(self, pressure, temperature, zc):
        # vec_state_as_np = np.asarray(state)
        # zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        (x, y, V) = self.RR(zc, self.ki)
        if V <= 0:
            V = 0
            x = zc
        elif V >= 1:
            V = 1
            y = zc

        return np.array([1-V, V]), np.array([x, y])

    def RR(self, zc, k):
        return RR_func(zc, k, self.min_z)


# # from numba import jit
# # @jit(nopython=True)
# def RR_func(zc, k, eps):
#
#     a = 1 / (1 - np.max(k)) + eps
#     b = 1 / (1 - np.min(k)) - eps
#
#     max_iter = 200  # use enough iterations for V to converge
#     for i in range(max_iter):
#         V = 0.5 * (a + b)
#         r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
#         if abs(r) < 1e-12:
#             break
#
#         if r > 0:
#             a = V
#         else:
#             b = V
#
#     if i >= max_iter:
#         print("Flash warning!!!")
#
#     x = zc / (V * (k - 1) + 1)
#     y = k * x
#
#     return V, x, y
#
#
# class Flash:
#     def __init__(self, components, ki, min_z=1e-11):
#         self.components = components
#         self.min_z = min_z
#         self.K_values = np.array(ki)
#
#     def evaluate(self, pressure, temperature, zc):
#         v1, x, y = RR_func(zc, self.K_values, self.min_z)
#
#         X = np.zeros((2, len(self.components)))
#         if v1 <= 0:
#             V = [1, 0]
#             X[0, :] = zc
#         elif v1 >= 1:
#             V = [0, 1]
#             X[1, :] = zc
#         else:
#             V = [1-v1, v1]
#             X[0, :] = x
#             X[1, :] = y
#
#         return V, X


class DensitySimple:
    def __init__(self, dens0=1000, compr=0, p0=1, x_mult=0):
        self.compr = compr
        self.p0 = p0
        self.dens0 = dens0
        self.x_max = x_mult

    def evaluate(self, p, T, x):
        density = (self.dens0 + x[0] * self.x_max) * (1 + self.compr * (p - self.p0))
        return density  # kg/m3


class ViscositySimple:
    def __init__(self, visc):
        self.visc = visc

    def evaluate(self, p, T, x, rho):
        return self.visc  # cP


class EnthalpySimple:
    def __init__(self, tref=273.15, hcap=0.0357):
        self.tref = tref
        self.hcap = hcap

    def evaluate(self, p, T, x):
        # methane heat capacity
        enthalpy = self.hcap * (T - self.tref)
        return enthalpy  # kJ/kmol


class PhaseRelPerm:
    def __init__(self, phase, swc=0, sgr=0):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = 1
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 2
        elif phase == 'gas':
            self.kre = 1
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = 2
        else:  # water
            self.kre = 1
            self.sr = 0
            self.sr1 = 0
            self.n = 1

    def evaluate(self, sat):

        if sat >= 1 - self.sr1:
            kr = self.kre

        elif sat <= self.sr:
            kr = 0

        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


class CapillaryPressure:
    def __init__(self, p_entry=0, swc=0, labda=2):
        self.swc = swc
        self.p_entry = p_entry
        self.labda = labda
        self.eps = 1e-3

    def evaluate(self, sat_w):
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = self.p_entry * Se ** (-1/self.labda)

        Pc = np.array([0, pc], dtype=object)

        return Pc


class RockCompactionEvaluator(property_evaluator_iface):
    def __init__(self, pref=1, compres=1.45e-5):
        super().__init__()
        self.Pref = pref
        self.compres = compres

    def evaluate(self, state):
        pressure = state[0]

        return (1.0 + self.compres * (pressure - self.Pref))


class RockEnergyEvaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, temperature):
        T_ref = 273.15
        # c_vr = 3710  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3
        c_vr = 1  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3

        return c_vr * (temperature - T_ref)  # kJ/m3
