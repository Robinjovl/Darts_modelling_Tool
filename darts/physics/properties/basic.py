import numpy as np


class ConstFunc:
    def __init__(self, value):
        self.value = value

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0, dummy4=0):
        return self.value


class PhaseRelPerm_Stone:
    def __init__(
        self,
        phase,
        property_container, 
        swc=0.00,
        sorw=0.00,  # OW system
        sgc=0.00,
        sorg=0.00,  # OG system
        krw_end=1.0,
        krocw=1.0,
        nw=2.0,
        now=2.0,
        krg_end=1.0,
        krog0=1.0,
        ng=2.0,
        nog=2.0,
        som=None,
    ):
        self.phase = phase
        self.pc = property_container
        self.wat_idx = self.pc.phases_name.index("wat")
        self.gas_idx = self.pc.phases_name.index("gas")
        self.oil_idx = self.pc.phases_name.index("oil")
        
        self.swc = swc
        self.sorw = sorw
        self.sgc = sgc
        self.sorg = sorg

        self.krw_end = krw_end
        self.krocw = krocw
        self.nw = nw
        self.now = now

        self.krg_end = krg_end
        self.krog0 = krog0
        self.ng = ng
        self.nog = nog

        self.som = som if som is not None else None

    @staticmethod
    def _eff(S, sr, sr1):
        den = np.maximum(1e-12, 1.0 - sr - sr1)
        return np.clip((S - sr) / den, 0.0, 1.0)

    def _two_phase_ow(self, Sw):
        Swe = self._eff(Sw, self.swc, self.sorw)
        krw = self.krw_end * (Swe**self.nw)
        krow = self.krocw * ((1.0 - Swe) ** self.now)
        return krw, krow

    def _two_phase_og(self, Sg):
        Sge = self._eff(Sg, self.sgc, self.sorg)
        krg = self.krg_end * (Sge**self.ng)
        krog = self.krog0 * ((1.0 - Sge) ** self.nog)
        return krg, krog

    def evaluate_kro(self, Sw, Sg):
        Sw = np.asarray(Sw, dtype=float)
        Sg = np.asarray(Sg, dtype=float)
        So = 1.0 - Sw - Sg

        krw, krow = self._two_phase_ow(Sw)
        krg, krog = self._two_phase_og(Sg)

        Som = self.som if self.som is not None else self.sorw
        den = np.maximum(1e-12, 1.0 - self.swc - Som)

        SwD = np.clip((Sw - self.swc) / den, 0.0, 1.0)
        SgD = np.clip(Sg / den, 0.0, 1.0)
        SoD = np.clip((So - Som) / den, 0.0, 1.0)

        eps = 1e-12
        kro = (
            SoD
            * (krow / np.maximum(eps, (1.0 - SwD)))
            * (krog / np.maximum(eps, (1.0 - SgD)))
        )

        # kro = np.where(So <= Som, np.nan, kro)
        krw = np.clip(krw, 0.0, 1.0)
        krg = np.clip(krg, 0.0, 1.0)
        kro = np.clip(kro, 0.0, 1.0)

        return krw, kro, krg

    def visualize(self):
        Nw = 100
        Ng = 100
        Sw_, Sg_ = np.meshgrid(np.linspace(0, 1, Nw), np.linspace(0, 1, Ng))
        Sw, Sg = Sw_.flatten(), Sg_.flatten()
        Sw = np.where(Sw + Sg < 1.0, Sw, np.nan)
        Sg = np.where(Sw + Sg < 1.0, Sg, np.nan)
        
        krw, kro_grid, krg = self.evaluate_kro(Sw, Sg)
          
        from darts.tools.plot_darts import tersurf
        import matplotlib.pyplot as plt 
        tersurf(1 - Sw.flatten() - Sg.flatten(),
                Sg.flatten(),
                Sw.flatten(),
                kro_grid.flatten(),
                ['water', 'oil', 'gas'],
                label = ['oil', 'gas', 'water'],
                title = 'kro'
                )
        plt.show()

        sw = np.linspace(0, 1, 100)
        krw, krow = self._two_phase_ow(sw)

        plt.figure()
        plt.plot(sw, krw, label = 'krw')
        plt.plot(sw, krow, label = 'krow')
        plt.xlabel('sw')
        plt.legend()
        plt.show()

        sg = np.linspace(0, 1, 100)
        krg, krog = self._two_phase_og(sg)

        plt.figure()
        plt.plot(sg, krg, label = 'krg')
        plt.plot(sg, krog, label = 'krog')
        plt.xlabel('sg')
        plt.legend()
        plt.show()

    def evaluate(self, sat):
        if self.phase == 'gas':
            krg, krog = self._two_phase_og(sat)
            return krg

        elif self.phase == 'wat':
            krw, krow = self._two_phase_ow(sat)
            return krw

        elif self.phase == 'oil':
            sat = self.pc.sat
            Sg, Sw = sat[self.gas_idx], sat[self.wat_idx]
            krw, kro, krg = self.evaluate_kro(Sw, Sg)
            return kro


class PhaseRelPerm:
    def __init__(self, phase, swc=0.0, sgr=0.0, kre=1.0, n=2.0):
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
    def __init__(self, phase, swc=0.20, sgr=0, kre=1.0, n=4.367):
        self.phase = phase
        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
            self.m = 1 - 1 / n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
            self.m = 1 - 1 / n
        else:  # water
            self.kre = 1
            self.sr = 0
            self.sr1 = 0
            self.n = n
            self.m = 1 - 1 / n

    def evaluate(self, sat):
        if self.phase == "oil":
            Se = (sat - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = np.sqrt(Se) * (1 - (1 - Se ** (1 / self.m)) ** self.m) ** 2

        elif self.phase == 'gas':
            Se = ((1 - sat) - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = (1 - Se) ** (1 / 3) * (1 - Se ** (1 / self.m)) ** (2 * self.m)
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
            Se = (sat[1] - self.swc) / (1 - self.swc)
            if Se < self.eps:
                Se = self.eps
            pc = self.p_entry * Se ** (-1 / self.labda)

            Pc = np.zeros(self.nph, dtype=object)
            Pc[1] = pc
        else:
            Pc = [0.0]
        return Pc


class CapillaryPressure_VG:  # Van Genuchten
    def __init__(
        self, nph=2, p_entry=0, swc=0, labda=3.3e-6, n=4.367
    ):  # [labda [Pa^-1]]
        self.nph = nph
        self.swc = swc
        self.p_entry = p_entry
        self.labda = labda
        self.eps = 1e-3
        self.n = n
        self.m = 1 - 1 / n

    def evaluate(self, sat):
        '''
        default evaluator of capillary pressure Pc based on pow
        :param sat: saturation
        :return: Pc
        '''
        Se = (sat - self.swc) / (1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = 1 / self.labda * (Se ** (-1 / self.m) - 1) ** (1 / self.n)
        Pc = np.zeros(self.nph, dtype=object)
        Pc[1] = pc
        return Pc


class RockCompactionEvaluator:
    def __init__(self, pref=1.0, compres=1.45e-5):
        self.Pref = pref
        self.compres = compres

    def evaluate(self, pressure):
        return 1.0 + self.compres * (pressure - self.Pref)
