import numpy as np
from darts.physics import *
from select_para import *
from EOS.fugacity_activity import *
from EOS.k_update import *


class flash_ccs_thermal(property_evaluator_iface):
    def __init__(self, components, phases, min_z):
        self.components = components
        self.phases = phases
        self.nc = len(self.components)
        self.nph = len(self.phases)
        self.min_z = min_z

    def flash(self, P, T, zc, Cm):

        if (min(zc) < 0):  # fix for when zc <= 0
            print('-----> fix:   min(zc) <= 0', zc)
            V = 1
            x = zc
            y = zc

        else:
            """ FIXED Initial K-VALUES """
            if self.nc == 2:
                ki = np.array([20, 0.01])
            elif self.nc == 3:
                ki = np.array([10, 20, 0.01])
            else:
                ki = np.array([10, 8, 3, 0.01])

            ki, l = update_k_value_flash(ki, P, T, zc, Cm, self.components, self.phases)

            (x, y, V) = self.RR(zc, ki)

            phi_c, Vm, hg_dev = vapour_liquid_fa(P, T, x, y, zc, self.components)

        return (x, y, V, Vm, hg_dev)

    def RR(self, zc, k):

        eps = self.min_z

        a = 1 / (1 - np.max(k)) + eps
        b = 1 / (1 - np.min(k)) - eps

        max_iter = 200  # use enough iterations for V to converge
        for i in range(1, max_iter):
            V = 0.5 * (a + b)

            r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))

            if r > 0:
                a = V
            else:
                b = V

            if V < 0:
                V = 0
            elif V > 1:
                V = 1
            if abs(r) < 1e-12:
                break

            if i >= max_iter:
                print("Flash warning!!!")

        x = zc / (V * (k - 1) + 1)
        y = k * x
        return (x, y, V)  # V is vapor fraction, x = mole fraction in liquid phase, y = mole fraction in gas phase


class AqDensity(property_evaluator_iface):
    def __init__(self, components, Cm):
        super().__init__()
        self.components = components
        self.Cm = Cm

    def evaluate(self, state, x):
        rho_brine, rho_w = self.rho_brine(state)
        rho_aq = self.rho_brine_Garcia(rho_brine, x, state[3])
        return rho_aq

    def rho_brine(self, state):  # Pressure, Temperature, Molality
        Cm = self.Cm
        P = state[0]
        T = state[3]
        # ref; spivey et al. 2004
        P0 = 700  # 70 MPa reference pressure
        Tc = T - 273.15

        # TEMP IN CELCIUS
        # PRESSURE IN BAR
        # SALINITY IN MOLES/KG
        def eq_3(a1, a2, a3, a4, a5, Tc):
            a = (a1 * (Tc / 100) ** 2 + a2 * (Tc / 100) + a3) / (a4 * (Tc / 100) ** 2 + a5 * (Tc / 100) + 1)
            return a

        # EoS coeff. for pure water
        Dw = np.array([-0.127213, 0.645486, 1.03265, -0.070291, 0.639589])
        E_w = np.array([4.221, -3.478, 6.221, 0.5182, -0.4405])
        F_w = np.array([-11.403, 29.932, 27.952, 0.20684, 0.3768])
        # coeff. for density of brine at ref. pressure 70 MPa as a function of Temp.
        D_cm2 = np.array([-7.925e-5, -1.93e-6, -3.4254e-4, 0, 0])
        D_cm32 = np.array([1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016])
        D_cm1 = np.array([-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685])
        D_cm12 = np.array([3.746e-4, -3.328e-4, -3.346e-4, 0, 0])
        # Coeff. for eq. 12 & 13 for brine compressibility
        E_cm = np.array([0, 0, 0.1353, 0, 0])
        F_cm32 = np.array([-1.409, -0.361, -0.2532, 0, 9.216])
        F_cm1 = np.array([0, 5.614, 4.6782, -0.307, 2.6069])
        F_cm12 = np.array([-0.1127, 0.2047, -0.0452, 0, 0])

        # Equation 3=> function of temp.
        Dcm2 = eq_3(D_cm2[0], D_cm2[1], D_cm2[2], D_cm2[3], D_cm2[4], Tc)
        Dcm32 = eq_3(D_cm32[0], D_cm32[1], D_cm32[2], D_cm32[3], D_cm32[4], Tc)
        Dcm1 = eq_3(D_cm1[0], D_cm1[1], D_cm1[2], D_cm1[3], D_cm1[4], Tc)
        Dcm12 = eq_3(D_cm12[0], D_cm12[1], D_cm12[2], D_cm12[3], D_cm12[4], Tc)

        rho_w0 = eq_3(Dw[0], Dw[1], Dw[2], Dw[3], Dw[4], Tc)  # density of water at 70Mpa
        # density of pure water
        Ew_w = eq_3(E_w[0], E_w[1], E_w[2], E_w[3], E_w[4], Tc)
        Fw_w = eq_3(F_w[0], F_w[1], F_w[2], F_w[3], F_w[4], Tc)
        Iw = (1 / Ew_w) * np.log(abs(Ew_w * (P / P0) + Fw_w))
        Iw0 = (1 / Ew_w) * np.log(abs(Ew_w * (P0 / P0) + Fw_w))

        rho_w = rho_w0 * np.exp(Iw - Iw0)
        # print('rho_w',rho_w)
        ## density of Brine
        Ew = eq_3(E_w[0], E_w[1], E_w[2], E_w[3], E_w[4], Tc)
        Fw = eq_3(F_w[0], F_w[1], F_w[2], F_w[3], F_w[4], Tc)
        Ecm = eq_3(E_cm[0], E_cm[1], E_cm[2], E_cm[3], E_cm[4], Tc)

        Fcm32 = eq_3(F_cm32[0], F_cm32[1], F_cm32[2], F_cm32[3], F_cm32[4], Tc)
        Fcm1 = eq_3(F_cm1[0], F_cm1[1], F_cm1[2], F_cm1[3], F_cm1[4], Tc)
        Fcm12 = eq_3(F_cm12[0], F_cm12[1], F_cm12[2], F_cm12[3], F_cm12[4], Tc)

        # function of Salinity Cm
        rho_b0 = rho_w0 + Dcm2 * Cm ** 2 + Dcm32 * Cm ** (3 / 2) + Dcm1 * Cm + Dcm12 * Cm ** (1 / 2)

        Eb = Ew + Ecm * Cm
        Fb = Fw + Fcm32 * Cm ** (3 / 2) + Fcm1 * Cm + Fcm12 * Cm ** (1 / 2)

        # function of pressure
        Ib_p = (1 / Eb) * np.log(abs(Eb * (P / P0) + Fb))
        Ib_p0 = (1 / Eb) * np.log(abs(Eb * (P0 / P0) + Fb))

        # density of brine
        rho_b = rho_b0 * np.exp(Ib_p - Ib_p0)
        return rho_b * 1e3, rho_w * 1e3

    def rho_brine_Garcia(self, rho_brine, x, T):
        tetha = T - 273.15  # Temp in [Celcius]
        # Ref. Garcia et al. (2001)
        # Mt_aq, Mt_g = Mw(x, components=self.components, y=[0, 0, 1])  # kg/mol
        Mw_h2o = (props('H2O', 'Mw')) / 1000  # kg/mol
        Mw_co2 = (props('CO2', 'Mw')) / 1000  # kg/mol
        # print(Mt_aq)
        # Apparent molar volume of dissolved CO2
        V_m = (37.51 - 9.585e-2 * tetha + 8.740e-4 * tetha ** 2 - 5.044e-7 * tetha ** 3) * 1e-6  # in [m3/mol]
        # V_phi  = (37.36 - 7.109e-2*tetha - 3.812e-5*tetha**2 + 3.296e-6*tetha**3 - 3.702e-9*tetha**4)*1e-6# in [cm3/mol]
        # Mw_brine = 1000 / (1000 / 18.01 + Cm) + 58.44 * Cm / (1000 / 18.01 + Cm)
        # V_m1 = V_phi / 1e6  # in [m3/mol]

        Vm_imp_1 = V_m

        Mw_imp_1 = (props('C1', 'Mw')) / 1000  # M in [kg/mol]
        Mt = Mw_h2o * x[-1] + Mw_co2 * x[0] + Mw_imp_1 * x[1]

        rho_aq = 1 / ((x[0] * V_m) / Mt + (x[1] * Vm_imp_1) / Mt + (Mw_h2o * x[-1]) / (rho_brine * Mt))

        return (rho_aq)  # -15 is correction from validation, this is not based on science!


class GasDensity(property_evaluator_iface):
    def __init__(self, compres, pref):
        super().__init__()
        # self.compres = compres
        # self.pref = pref

    def evaluate(self, Mt_g, Vm):
        rho_g = Mt_g / Vm
        if hasattr(rho_g, "__len__"):
            rho_g = rho_g[0]
        return rho_g


class GasViscosity(property_evaluator_iface):
    def __init__(self):
        super().__init__()
        self.A_CO2 = [-1.146067e-01, 6.978380e-07, 3.976765e-10, 6.336120e-02,
                      -1.166119e-02, 7.142596e-04, 6.519333e-06, -3.567559e-01, 3.180473e-02]
        self.A_C1 = [-2.25711259e-02, -1.31338399e-04, 3.44353097e-06, -4.69476607e-08, 2.23030860e-02,
                     -5.56421194e-03, 2.90880717e-05, -1.90511457e0, 1.14082882e0, -2.25890087e-01]
        self.pc_c1 = props('C1', 'Pc')
        self.Tc_c1 = props('C1', 'Tc')
        self.Mw_C1 = props('C1', 'Mw')
        self.Mw_CO2 = props('CO2', 'Mw')

    def evaluate(self, state, y):  # Ignores CH4 concentration
        T = state[3]
        p = state[0]
        pr = p / self.pc_c1
        Tr = T / self.Tc_c1

        mu_co2 = (self.A_CO2[0] + self.A_CO2[1] * p + self.A_CO2[2] * p ** 2 + self.A_CO2[3] * np.log(T) + self.A_CO2[
            4] * np.log(T) ** 2 + self.A_CO2[5] * np.log(
            T) ** 3) / (1 + self.A_CO2[6] * p + self.A_CO2[7] * np.log(T) + self.A_CO2[8] * np.log(T) ** 2)

        if 72 <= p <= 77:
            mu_co2 = (p - 72) / 5 * mu_co2 + (77 - p) / 5 * 0.018
        elif p < 73:
            mu_co2 = 0.018

        mu_c1 = (self.A_C1[0] + self.A_C1[1] * pr + self.A_C1[2] * pr ** 2 + self.A_C1[3] * pr ** 3 + self.A_C1[
            4] * Tr + self.A_C1[5] * Tr ** 2) / (
                        1 + self.A_C1[6] * pr + self.A_C1[7] * Tr + self.A_C1[8] * Tr ** 2 + self.A_C1[9] * Tr ** 3)

        mu_g = (mu_c1 * y[1] * np.sqrt(self.Mw_C1) + mu_co2 * y[0] * np.sqrt(self.Mw_CO2)) / (
        (y[1] * np.sqrt(self.Mw_C1) + y[0] * np.sqrt(self.Mw_CO2)))

        return mu_g


class AqViscosity(property_evaluator_iface):
    def __init__(self, Cm):
        super().__init__()
        self.Cm = Cm

    def evaluate(self, state, x):  # Ignores CH4 concentration
        Tc = state[3] - 273.15
        S = self.Cm * 55500 / 1e6
        mu_b = 0.1 + 0.333 * S + (1.65 + 91.9 * S * S * S) * np.exp(
            -1 * (0.42 * (S ** 0.8 - 0.17) ** 2 + 0.045) * Tc ** 0.8)
        mu_aq = mu_b * (1 + 4.65 * x[0] ** 1.0134)

        return mu_aq


class PhaseRelPerm(property_evaluator_iface):
    def __init__(self, phase, prop_cont, swc=0.175, sgr=0.175, krwe=1, krge=1.0, nw=1.9, no=3):
        super().__init__()
        self.phase = phase
        self.swc = swc
        self.sgr = sgr
        self.krwe = krwe
        self.krge = krge
        self.nw = nw
        self.no = no
        self.prop_cont = prop_cont

    def evaluate(self, sat):
        if self.phase == 'Aq':
            if sat < self.swc:
                kr = 0
            elif sat > 1 - self.sgr:
                kr = self.krwe
            else:
                kr = self.krwe * ((sat - self.swc) / (1 - self.sgr - self.swc)) ** self.nw
        elif self.phase == 'Gas':
            if sat < self.sgr:
                kr = 0
            elif sat > 1 - self.sgr:
                kr = self.krge
            else:
                kr = self.krge * ((sat - self.sgr) / (1 - self.sgr - self.swc)) ** self.no
        else:
            print('Wrong phase name!')
            kr = 0

        return kr


class CapillaryPressure(property_evaluator_iface):
    def __init__(self, prop_cont, pentry=0.2, lam=0.5, swc=0.2, sgr=0.2):
        super().__init__()
        self.pd = pentry
        self.lam = lam
        self.swc = swc
        self.sgr = sgr
        self.prop_cont = prop_cont

    def evaluate(self, state):
        sw = self.prop_cont.watersat_ev.evaluate(state)
        eps = 1e-4

        pc = self.pd * (sw - self.swc + eps) / (1 - self.swc - self.sgr)

        return pc


class GasEnthalpy(property_evaluator_iface):
    def __init__(self):
        super().__init__()
        # Values for Guo
        self.Tref = 1000  # Kelvin
        self.R = 8.3145
        # Calculate
        # For CH4,CO2, use Yongfan Guo
        # parameters for ideal gas enthalpy for CH4,CO2, H2o
        self.a_ideal = np.array(
            [[-133.8335522, -44.33883961, 214.4210088, 62.37629521, -821.1817892, 1123.866804, -507.3991957,
              177.1006921, -114.0665933, 44.04725269, -10.20113159, 1.305622457, -0.071037683, 28371.1091],
             # 28351.10919],
             [-1.8188731, 12.903022, -9.6634864, 4.2251879, -1.042164, 0.12683515, -0.49939675,
              2.4950242, -0.8272375, 0.15372481, -0.015861243, 0.000860172, 1.92222E-05, 3678.207003],  # 2108.207003],
             [31.04096012, -39.14220805, 37.96952772, -21.8374911, 7.422514946, -1.381789296, 0.108807068,
              -12.07711768, 3.391050789, -0.58452098, 0.058993085, -0.0031297, 6.57461 * 1e-5, 9908]])

    def evaluate(self, state, y, H_dev):
        T = state[3]  # -1 does not work?
        dH_vap_H2O = 40.8 * y[2]
        enthalpy = self.ideal_Guo(y, state[3]) + H_dev
        if hasattr(enthalpy, "__len__"):
            enthalpy = enthalpy[0]
        # vapratio = (100*dH_vap_H2O/(enthalpy/1000)), #Commonly 1-2%
        # print('vaporization enthalpy in kj/mol', dH_vap_H2O, 'ratio to total enthalpy = ', vapratio, '%')
        return enthalpy / 1000 - dH_vap_H2O  # to Kj/mol

    def ideal_Guo(self, y, T):
        # print(a_ideal.shape)
        a_mix = np.zeros(14)
        for ii in range(0, 14):
            a_mix[ii] = self.a_ideal[0, ii] * y[1] + self.a_ideal[1, ii] * y[0] + self.a_ideal[2, ii] * y[2]
        # print(a_mix)
        tau = T / self.Tref
        a_temp = 0
        for ii in range(0, 13):
            if ii < 7:
                a_temp = a_temp + a_mix[ii] / (ii + 1) * tau ** (ii + 1)
            elif ii == 7:
                a_temp = a_temp + a_mix[ii] * np.log(tau)
            elif ii < 13:
                a_temp = a_temp + a_mix[ii] / (7 - ii) * (1 / tau) ** (ii - 7)

        H_ideal = self.R * self.Tref * a_temp + self.R*a_mix[-1]

        return H_ideal


class AqEnthalpy(property_evaluator_iface):  # Assume pure h20 for now
    def __init__(self, Cm):
        super().__init__()
        self.Cm = Cm

    def evaluate(self, state, x):
        P = state[0]
        T = state[3]
        Tc = T - 273.15

        Hw, Hs = self.Hpure(Tc)
        # print(Hw,Hs)
        if Tc < 150:
            Hsb = self.Michalides(Hw, Hs, Tc)
        # elif 99.9 <= Tc <= 130.1:
        #     Hsb1 = self.Michalides(Hw, Hs, Tc)
        #     Hsb2 = self.Lorenz(Hw, Hs, Tc)
        #     Hsb = (Hsb1 + Hsb2) / 2
        # elif 130.1 < Tc <= 300:
        #     Hsb = self.Lorenz(Hw, Hs, Tc)
        else:
            print('T out of bounds for Aq enthalpy')

        # print(Hsb)
        dH_diss_g = self.dH_diss_gas(P, T)
        Hbrine = self.Hrevised(P, T, Hsb)
        # print(dH_diss_g)
        return Hbrine - dH_diss_g * x[0]
        #
        # x1 = 1000 / (1000 + 58.44 * self.Cm)
        # x2 = 58.44 * self.Cm / (1000 + 58.44 * self.Cm)
        # factor = (1000 + 58.44 * self.Cm) / (1000 / 18.015 + self.Cm)
        # Hbrine = (Hw * x1 + Hs * x2) * factor #self.Hrevised(P, T, Hsb, rho_b)

    def rho_brine(self, P, Tc):  # Pressure, Temperature, Molality

        # ref; spivey et al. 2004
        P0 = 700  # 70 MPa reference pressure

        # TEMP IN CELCIUS
        # PRESSURE IN BAR
        # SALINITY IN MOLES/KG
        def eq_3(a1, a2, a3, a4, a5, Tc):
            a = (a1 * (Tc / 100) ** 2 + a2 * (Tc / 100) + a3) / (a4 * (Tc / 100) ** 2 + a5 * (Tc / 100) + 1)
            return a

        # EoS coeff. for pure water
        Dw = np.array([-0.127213, 0.645486, 1.03265, -0.070291, 0.639589])
        E_w = np.array([4.221, -3.478, 6.221, 0.5182, -0.4405])
        F_w = np.array([-11.403, 29.932, 27.952, 0.20684, 0.3768])
        # coeff. for density of brine at ref. pressure 70 MPa as a function of Temp.
        D_cm2 = np.array([-7.925e-5, -1.93e-6, -3.4254e-4, 0, 0])
        D_cm32 = np.array([1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016])
        D_cm1 = np.array([-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685])
        D_cm12 = np.array([3.746e-4, -3.328e-4, -3.346e-4, 0, 0])
        # Coeff. for eq. 12 & 13 for brine compressibility
        E_cm = np.array([0, 0, 0.1353, 0, 0])
        F_cm32 = np.array([-1.409, -0.361, -0.2532, 0, 9.216])
        F_cm1 = np.array([0, 5.614, 4.6782, -0.307, 2.6069])
        F_cm12 = np.array([-0.1127, 0.2047, -0.0452, 0, 0])

        # Equation 3=> function of temp.
        Dcm2 = eq_3(D_cm2[0], D_cm2[1], D_cm2[2], D_cm2[3], D_cm2[4], Tc)
        Dcm32 = eq_3(D_cm32[0], D_cm32[1], D_cm32[2], D_cm32[3], D_cm32[4], Tc)
        Dcm1 = eq_3(D_cm1[0], D_cm1[1], D_cm1[2], D_cm1[3], D_cm1[4], Tc)
        Dcm12 = eq_3(D_cm12[0], D_cm12[1], D_cm12[2], D_cm12[3], D_cm12[4], Tc)

        rho_w0 = eq_3(Dw[0], Dw[1], Dw[2], Dw[3], Dw[4], Tc)  # density of water at 70Mpa
        # density of pure water
        Ew_w = eq_3(E_w[0], E_w[1], E_w[2], E_w[3], E_w[4], Tc)
        Fw_w = eq_3(F_w[0], F_w[1], F_w[2], F_w[3], F_w[4], Tc)
        Iw = (1 / Ew_w) * np.log(abs(Ew_w * (P / P0) + Fw_w))
        Iw0 = (1 / Ew_w) * np.log(abs(Ew_w * (P0 / P0) + Fw_w))

        rho_w = rho_w0 * np.exp(Iw - Iw0)
        # print('rho_w',rho_w)
        ## density of Brine
        Ew = eq_3(E_w[0], E_w[1], E_w[2], E_w[3], E_w[4], Tc)
        Fw = eq_3(F_w[0], F_w[1], F_w[2], F_w[3], F_w[4], Tc)
        Ecm = eq_3(E_cm[0], E_cm[1], E_cm[2], E_cm[3], E_cm[4], Tc)

        Fcm32 = eq_3(F_cm32[0], F_cm32[1], F_cm32[2], F_cm32[3], F_cm32[4], Tc)
        Fcm1 = eq_3(F_cm1[0], F_cm1[1], F_cm1[2], F_cm1[3], F_cm1[4], Tc)
        Fcm12 = eq_3(F_cm12[0], F_cm12[1], F_cm12[2], F_cm12[3], F_cm12[4], Tc)

        # function of Salinity Cm
        rho_b0 = rho_w0 + Dcm2 * self.Cm ** 2 + Dcm32 * self.Cm ** (3 / 2) + Dcm1 * self.Cm + Dcm12 * self.Cm ** (1 / 2)

        Eb = Ew + Ecm * self.Cm
        Fb = Fw + Fcm32 * self.Cm ** (3 / 2) + Fcm1 * self.Cm + Fcm12 * self.Cm ** (1 / 2)

        # function of pressure
        Ib_p = (1 / Eb) * np.log(abs(Eb * (P / P0) + Fb))
        Ib_p0 = (1 / Eb) * np.log(abs(Eb * (P0 / P0) + Fb))

        # density of brine
        rho_b = rho_b0 * np.exp(Ib_p - Ib_p0)

        return rho_b * 1e3, rho_w * 1e3

    # Pure water enthalpy and salt give good correlations, validated
    def Hpure(self, T):  # Keenan, Keyes, Hill and Moore
        Hw = 0.12453e-4 * T ** 3 - 0.4513e-2 * T ** 2 + 4.81155 * T - 29.578
        Hs = (-0.83624e-3 * T ** 3 + 0.16792 * T ** 2 - 25.9293 * T) * (4.184 / 58.44)
        return Hw, Hs  # in kJ/kg  /58.4428/55.55

    def Antoine(self, Tc):
        if Tc < 99.9:
            Psat = 0.00133322 * 10 ** (8.07131 - (1730.63 / (233.426 + Tc)))
        elif 99.9 <= Tc <= 100.1:
            Psat1 = 10 ** (8.07131 - (1730.63 / (233.426 + Tc)))
            Psat2 = 10 ** (8.140191 - (1810.94 / (244.485 + Tc)))
            Psat = 0.00133322 * (Psat1 + Psat2) / 2
        elif 100.1 < Tc <= 374:
            Psat = 0.00133322 * 10 ** (8.140191 - (1810.94 / (244.485 + Tc)))  # To bar
        else:
            print('T out of bounds for Antoine')
        return Psat

    def Michalides(self, Hw, Hs, T):
        # Input in Kj/Kg , mol/kg and Celsius
        aij = np.array([[-9633.6, -4080.0, 286.49],
                        [166.58, 68.577, -4.6856],
                        [-0.90963, -0.36524, 0.0249667],
                        [0.17965e-2, 0.71924e-3, -0.4900e-4]])

        dH_diss_salt = 0
        for ii in [0, 1, 2, 3]:
            for jj in [0, 1, 2]:
                dH_diss_salt += aij[ii, jj] * T ** ii * self.Cm ** jj
        dH_diss_salt *= 4.184 / (1000 + 58.44 * self.Cm)

        x1 = 1000 / (1000 + 58.44 * self.Cm)
        x2 = 58.44 * self.Cm / (1000 + 58.44 * self.Cm)
        return x1 * Hw + x2 * Hs + dH_diss_salt

    # Lorenz Function gives correct results -> Validated
    def Lorenz(self, Hw, Hs, Tc):

        # Input Kj/mol, Kj,mol, mol/kg, C
        # Needs wt% and C so multiply cm by 58.44
        # mol/kg to kg/kg % = 1100g/kg
        wtpct = self.Cm * 58.44 / (1000 + self.Cm * 58.44) * 100

        bij = np.array([[0.2985, -7.257e-2, 1.071e-3],
                        [-7.819e-2, 2.169e-3, -3.343e-5],
                        [3.479e-4, -1.809e-5, 3.450e-7],
                        [-1.203e-6, 5.910e-8, -1.131e-9]])

        dH_diss_salt = 0
        for jj in [1, 2, 3]:
            Ai_t = 0
            for ii in [0, 1, 2, 3]:
                Ai_t += bij[ii, jj - 1] * (Tc ** ii)
            dH_diss_salt += Ai_t * (wtpct ** jj)

        x1 = 1
        x2 = 0

        return x1 * Hw + x2 * Hs + dH_diss_salt

    def dH_diss_gas(self, P, T):
        T2 = T + 0.1  # Forward difference with 1 K
        R = 8.31446  # J/mol/K
        phi_c, Hcoeff1 = aqueous(P, T, ['CO2'], self.Cm)
        phi_c, Hcoeff2 = aqueous(P, T2, ['CO2'], self.Cm)

        dH_diss_g = (np.log(Hcoeff2) - np.log(Hcoeff1)) / (1 / T2 - 1 / T) * R / 44.01  # in Kj/Kg Co2

        return dH_diss_g / (1000 / 44.01)  # convert to Kj/mol

    def Hrevised(self, P, T, Hsb):
        # Input in Bar, Kelvin, mol/kg, kg/m3
        # print(rho_b)
        T_dt = 1
        T_forward = T + T_dt
        rho_b, rho_w = self.rho_brine(P, T - 273.15)
        rho_b_f, rho_w_f = self.rho_brine(P, T_forward - 273.15)
        V = 1 / rho_b
        # print(V)
        dVdT = (1 / rho_b_f - 1 / rho_b) / T_dt
        # print(dVdT)
        Psat = self.Antoine(T - 273.15)  # Antoine equation, outputs in bar
        # print(Psat)
        Interim = (V - (T - 273.15) * dVdT)
        Mw_brine = 1000 / (1000 / 18.01 + self.Cm) + 58.44 * self.Cm / (1000 / 18.01 + self.Cm)
        # print(Mw_brine,'g/mol')
        Hbrine = Hsb / (1000 / (Mw_brine)) + Interim * (P - Psat)
        Factor = 1  # 1000/18.015#(1000 / 18.015 + Cm)/(1000 + 58.44 * Cm)  #to kJ/mol
        # print('Kg/mol', Factor)
        Hbrine = Hbrine / Factor
        # print(Hbrine,'kJ/mol')
        return Hbrine


class AqCondonductivity(property_evaluator_iface):
    def __init__(self, Cm, condfact=1):
        super().__init__()
        self.S = Cm * 58.44
        self.fact = condfact  # Factor to turn on/off for comparison

    def evaluate(self, state):
        T = state[3]
        T_d = T / 300
        cond_aq = 0.797015 * T_d ** -0.194 - 0.251242 * T_d ** -4.717 + 0.096437 * T_d ** -6.385 - 0.032696 * T_d ** -2.134
        cond_brine = (cond_aq / (0.00022 * self.S + 1)) + 0.00005 * (state[0] - 50)
        return cond_brine  #/ 1000 * 3600 * 24 * self.fact  # Convert from W/m/k to kj/m/day/K


class GasCondonductivity(property_evaluator_iface):
    def __init__(self, condfact=1):
        super().__init__()
        self.A = [105.161, 0.9007, 0.0007, 3.5e-15, 3.76e-10, 0.75, 0.0017]
        self.fact = condfact  # Factor to turn on/off for comparison

    def evaluate(self, rho_g, state, sg):
        T = state[3]
        A = self.A
        cond_g = (A[0] + A[1] * rho_g + A[2] * rho_g ** 2 + A[3] * rho_g ** 3 * T ** 3 + A[4] * rho_g ** 4 + A[
            5] * T + A[6] * T ** 2) / np.sqrt(T)
        return cond_g * 1e-3 # / 1000 * 3600 * 24 * self.fact  # Convert from W/m/k to kj/m/day/K


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

    def evaluate(self, state):
        T = state[3]
        T_ref = 273.15
        # c_vr = 3710  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3
        c_vr = 1  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3

        return c_vr * (T - T_ref)  # kJ/m3
