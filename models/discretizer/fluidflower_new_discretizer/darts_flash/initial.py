import numpy as np


class IdealK:
    H0 = {"CO2": 33., "N2": 0.64, "H2S": 100., "C1": 1.4, "C2": 1.9, "C3": 1.5, "iC4": 0.91, "nC4": 1.2,
          "iC5": 0.7, "nC5": 0.8, "nC6": 0.61, "nC7": 0.44, "nC8": 0.31, "nC9": 0.2, "nC10": 0.14}
    dlnH = {"CO2": 2400., "N2": 1600., "H2S": 2100., "C1": 1900., "C2": 2400., "C3": 2700., "iC4": 2700., "nC4": 3100.,
            "iC5": 3400., "nC5": 3400., "nC6": 3800., "nC7": 4100., "nC8": 4300., "nC9": 5000., "nC10": 5000.}

    def __init__(self, components, component_data: dict):
        self.components = components
        self.nc = len(components)
        self.comp_data = component_data

    def wilson(self, p, T, vapour_idx=0):
        ki_wilson = np.zeros(self.nc)

        for i in range(self.nc):
            if self.components[i] == "H2O":
                aii = [-133.67, 0.63288, 3.19211E-3]
                ki_wilson[i] = (aii[0] + aii[1] * T) / p + aii[2] * p  # K_i = x_iV/x_iAq
            else:
                ki_wilson[i] = self.comp_data["Pc"][i] / p * np.exp(5.373 * (1 + self.comp_data["ac"][i]) * (1 - self.comp_data["Tc"][i] / T))  # K_i = x_V/x_Aq

        if vapour_idx == 0:
            return 1/ki_wilson
        else:
            return ki_wilson

    def activity(self, p, T, vapour_idx=0):
        # Initial K - value for phase Aq
        # Ballard(2002) - Appendix A.1.2
        ki_activity = np.zeros(self.nc)
        for i in range(self.nc):
            comp = self.components[i]
            if comp == "H2O":
                # Raoult's law for H2O
                psat = np.exp(12.048399 - 4030.18245 / (T - 38.15))
                j_inf = 1
                ki_activity[i] = psat / p * j_inf
            else:
                # Henry's law for solutes in dilute solution
                x_iV = 1.
                H = self.H0[comp] * np.exp(self.dlnH[comp] * (1 / T - 1 / 298.15))
                ca = H * p
                rho_Aq = 1100
                Vm = self.comp_data["Mw"][self.components.index("H2O")] * 1E-3 / rho_Aq
                x_iAq = ca * Vm
                ki_activity[i] = x_iV / x_iAq

        if vapour_idx == 0:
            return 1/ki_activity
        else:
            return ki_activity

    def ideal(self):

        return

    def pure(self, z, j, z_pure=0.9):
        ki_pure = np.zeros(self.nc)

        for i in range(self.nc):
            if i == j:
                ki_pure[i] = z_pure / z[i]
            else:
                ki_pure[i] = (1-z_pure) / ((self.nc - 1) * z[i])

        return ki_pure


class InitialGuess(IdealK):
    def __init__(self, components, component_data):
        super().__init__(components, component_data)

    def evaluate(self, p, T, z):
        pass


class Wilson(InitialGuess):
    def __init__(self, components, component_data, vap_idx=0):
        super().__init__(components, component_data)

        self.vap_idx = vap_idx

    def evaluate(self, p, T, z):
        lnK = np.log(self.wilson(p, T, vapour_idx=self.vap_idx))
        return lnK


class Activity(InitialGuess):
    def __init__(self, components, component_data, vap_idx=0):
        super().__init__(components, component_data)

        self.vap_idx = vap_idx

    def evaluate(self, p, T, z):
        lnK = np.log(self.activity(p, T, vapour_idx=self.vap_idx))
        return lnK


# def vapour_sI(self):
#     # Initial K - value for sI
#     # Ballard(2002) - Appendix A.1.5
#
#     K_VsI = np.zeros(self.NC)
#     x_wH = 0.88
#
#     a_ = {"CO2": [15.8336435, 3.119, 0, 3760.6324, 1090.27777, 0, 0],
#           "H2S": [31.209396, -4.20751374, 0.761087, 8340.62535, -751.895, 182.905, 0],
#           "C1": [27.474169, -0.8587468, 0, 6604.6088, 50.8806, 1.57577, -1.4011858],
#           "C2": [14.81962, 6.813994, 0, 3463.9937, 2215.3, 0, 0]}
#     a_N2 = {0: [173.2164, -0.5996, 0, 24751.6667, 0, 0, 0, 1.441, -37.0696, -0.287334, -2.07405E-5, 0, 0],
#             1: [71.67484, -1.75377, -0.32788, 25180.56, 0, 0, 0, 56.219655, -140.5394, 0, 8.0641E-4, 366006.5,
#                 978852]}
#
#     for i, comp in enumerate(self.components):
#         if comp == "H2O":
#             # Kw_VAq
#             psat = np.exp(12.048399 - 4030.18245 / (T - 38.15))
#             j_inf = 1
#             Kw_VAq = psat / p * j_inf
#             # Kw_IAq
#             p0 = 6.11657E-3
#             T0 = 273.1576
#             Ti = T0 - 7.404E-3 * (p - p0) - 1.461E-6 * (p - p0) ** 2
#             x_wAq = 1 + 8.33076E-3 * (T - Ti) + 3.91416E-5 * (T - Ti) ** 2
#             Kw_IAq = 1 / x_wAq
#             K_VsI[i] = Kw_VAq / (x_wH * Kw_IAq)
#         elif comp == "N2":
#             if "H2S" in self.components:  # depends on presence of H2S
#                 a = a_N2[1]
#             else:
#                 a = a_N2[0]
#             Ki_wf = np.exp(a[0] + a[1] * np.log(p) + a[2] * (np.log(p)) ** 2 -
#                            (a[3] + a[4] * np.log(p) + a[5] * (np.log(p)) ** 2 + a[6] * (np.log(p)) ** 3) / T
#                                + a[7] / p + a[8] / p ** 2 + a[9] * T + a[10] * p + a[11] * np.log(p) / T ** 2 + a[
#                                    12] / T ** 2)
#             K_VsI[i] = Ki_wf / (1 - x_wH)
#         else:
#             a = a_[comp]
#             Ki_wf = np.exp(a[0] + a[1] * np.log(p) + a[2] * (np.log(p)) ** 2 - (
#                         a[3] + a[4] * np.log(p) + a[5] * (np.log(p)) ** 2 + + a[6] * (np.log(p)) ** 3) / T)
#             K_VsI[i] = Ki_wf / (1 - x_wH)
#
#     return K_VsI
