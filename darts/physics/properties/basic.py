import numpy as np


class ConstFunc:
    def __init__(self, value):
        self.value = value

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0, dummy4=0):
        return self.value


class PhaseRelPerm:
    def __init__(self, phase, swc=0., sgr=0., kre=1., n=2.):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
        else:  # water
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n

    def evaluate(self, sat):
        if sat >= 1 - self.sr1:
            kr = self.kre
        elif sat <= self.sr:
            kr = 0
        else:
            # general Brooks-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


class PhaseRelPerm_VG:  # Van Genuchten
    def __init__(self, phase, swc=0.20, sgr=0, kre=1., n=4.367):
        self.phase = phase
        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
            self.m = 1 - 1/n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
            self.m = 1 - 1/n
        else:  # water
            self.kre = 1
            self.sr = 0
            self.sr1 = 0
            self.n = n
            self.m = 1 - 1/n

    def evaluate(self, sat):
        if self.phase == "oil":
            Se = (sat - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = np.sqrt(Se) * (1 - (1 - Se ** (1/self.m)) ** self.m) ** 2

        elif self.phase == 'gas':
            Se = ((1-sat) - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = (1 - Se) ** (1/3) * (1 - Se ** (1/self.m)) ** (2*self.m)
        return kr


class CapillaryPressure:
    def __init__(self, nph=2, p_entry=0, swc=0, labda=2):
        self.nph = nph
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
        if self.nph > 1:
            Se = (sat[1] - self.swc)/(1 - self.swc)
            if Se < self.eps:
                Se = self.eps
            pc = self.p_entry * Se ** (-1/self.labda)

            Pc = np.zeros(self.nph, dtype=object)
            Pc[1] = pc
        else:
            Pc = [0.0]
        return Pc


class CapillaryPressure_VG:  # Van Genuchten
    def __init__(self, nph=2, p_entry=0, swc=0, labda=3.3e-6, n=4.367): #[labda [Pa^-1]]
        self.nph = nph
        self.swc = swc
        self.p_entry = p_entry
        self.labda = labda
        self.eps = 1e-3
        self.n = n
        self.m = 1 - 1/n

    def evaluate(self, sat):
        '''
        default evaluator of capillary pressure Pc based on pow
        :param sat: saturation
        :return: Pc
        '''
        Se = (sat - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = 1 / self.labda * (Se ** (-1/self.m) - 1) ** (1/self.n)
        Pc = np.zeros(self.nph, dtype=object)
        Pc[1] = pc
        return Pc


class RockCompactionEvaluator:
    def __init__(self, pref=1., compres=1.45e-5):
        self.Pref = pref
        self.compres = compres

    def evaluate(self, pressure):
        return 1.0 + self.compres * (pressure - self.Pref)


class RockEnergyEvaluator:
    def __init__(self, T_ref=273.15):
        self.T_ref = T_ref

    def evaluate(self, temperature):
        return temperature - self.T_ref  # T-T_0, multiplied by rock hcap inside engine
