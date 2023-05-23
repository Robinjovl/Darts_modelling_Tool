import numpy as np


class ConstFunc:
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0, dummy4=0):
        return self.value


class Flash:
    def __init__(self, nph, nc):
        self.nph = nph
        self.nc = nc

    def evaluate(self, pressure, temperature, zc):
        pass


class flash_3phase():
    def __init__(self, components):
        self.components = components
        mixture = Mix(components)
        binary = Kij(components)
        mixture.kij_cubic(binary)

        self.eos = preos(mixture, mixrule='qmr', volume_translation=True)

    def evaluate(self, p, T, zc):
        nu, x, status = multiphase_flash(self.components, zc, T, p, self.eos)

        return x, nu


# Uncomment these two lines if numba package is installed and make things happen much faster:
from numba import jit
@jit(nopython=True)
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


# from dartsflash import RRN2Convex
# def RR_convex(zc, k, eps):
#     nc = len(zc)
#     rr = RRN2Convex(nc)
#     v = rr.solve_rr(zc, k, 1e-10, 50)
#     x = rr.getx()
#     return x[0, :], x[1, :], v[1]


class ConstantK(Flash):
    def __init__(self, nc, ki, min_z=1e-11):
        super().__init__(nph=2, nc=nc)

        self.min_z = min_z
        self.K_values = np.array(ki)

    def evaluate(self, pressure, temperature, zc):

        (x, y, V) = RR_func(zc, self.K_values, self.min_z)
        return np.array([V, 1-V]), np.array([y, x])


#  Density dependent on compressibility only
class Density:
    def __init__(self, dens0, compr=0., p0=1.):
        self.dens0 = dens0
        self.compr = compr
        self.p0 = p0

    def evaluate(self, pressure, temperature: float = None, x: list = None):
        return self.dens0 * (1 + self.compr * (pressure - self.p0))


class DensityBrineCo2(Density):
    def __init__(self, components, dens0=1000., compr=0., p0=1., x_mult=0.):
        super().__init__(dens0, compr, p0)
        self.x_max = x_mult

        if "CO2" in components:
            self.CO2_idx = components.index("CO2")
        else:
            self.CO2_idx = None

    def evaluate(self, pressure, temperature, x):
        if self.CO2_idx is not None:
            x_co2 = x[self.CO2_idx]
        else:
            x_co2 = 0.

        density = (self.dens0 + x_co2 * self.x_max) * (1 + self.compr * (pressure - self.p0))
        return density


class Density4Ions:
    def __init__(self, density, compressibility=0, p_ref=1, ions_fac=0):
        super().__init__()
        # Density evaluator class based on simple first order compressibility approximation (Taylor expansion)
        self.density_rc = density
        self.cr = compressibility
        self.p_ref = p_ref
        self.ions_fac = ions_fac

    def evaluate(self, pres, ion_liq_molefrac):
        return self.density_rc * (1 + self.cr * (pres - self.p_ref) + self.ions_fac * ion_liq_molefrac)


class Enthalpy:
    def __init__(self, tref=273.15, hcap=0.0357):
        self.tref = tref
        self.hcap = hcap

    def evaluate(self, pressure: float = None, temperature: float = None, x: list = None):
        # Enthalpy based on constant heat capacity
        enthalpy = self.hcap * (temperature - self.tref)
        return enthalpy


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
            self.n = 2

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

    def evaluate(self, sat):
        '''
        default evaluator of capillary pressure Pc based on pow
        :param sat: saturation
        :return: Pc
        '''
        Se = (sat - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = self.p_entry * Se ** (-1/self.labda)

        Pc = np.array([0, pc], dtype=object)

        return Pc


class KineticBasic:
    def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.kinetic_rate = np.zeros(ne)
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, x, nu_sol):
        if self.combined_ions:
            ion_prod = (x[1][1] / 2) ** 2
            self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - 0.5 * self.kinetic_rate[1]
        else:
            ion_prod = x[1][1] * x[1][2]
            self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[2] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - self.kinetic_rate[1]

        return self.kinetic_rate


class RockCompactionEvaluator:
    def __init__(self, pref=1, compres=1.45e-5):
        self.Pref = pref
        self.compres = compres

    def evaluate(self, pressure):
        return 1.0 + self.compres * (pressure - self.Pref)


class RockEnergyEvaluator:
    def __init__(self, c_vr=3710., T_ref=273.15):
        self.c_vr = c_vr  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3
        self.T_ref = T_ref

    def evaluate(self, temperature):
        return self.c_vr * (temperature - self.T_ref)  # kJ/m3
