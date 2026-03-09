import numpy as np

class STARSFoamRelPerm:
    def __init__(self, phase, foam, swc=0., sgr=0., kre=1., n=2.):
        super().__init__()
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        self.kre = kre
        self.sr = sgr
        self.sr1 = swc
        self.n = n
        self.fmmob = foam[0]
        self.fmdry = foam[1]
        self.epdry = foam[2]


    def evaluate(self, sat):

        if sat >= 1 - self.sr1:
            kr = self.kre

        elif sat <= self.sr:
            kr = 0

        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        water_sat = 1 - sat

        Fw = 0.5 + np.arctan(self.epdry * (water_sat - self.fmdry)) / np.pi

        ret = kr / (1 + self.fmmob * Fw)

        return ret


class WaterPhaseRelPerm:
    def __init__(self, phase, swc=0.0, sgr=0.0, kre=1.0, n=2.0):

        self.Swc = swc
        self.Sgr = sgr
        self.kre = kre
        self.sr = swc
        self.sr1 = sgr
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