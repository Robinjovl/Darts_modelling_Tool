import numpy as np


class Viscosity:
    def __init__(self, components: list = None, ions: list = None):
        self.nc = len(components) if components is not None else 0
        self.ni = len(ions) if ions is not None else 0

    def evaluate(self, pressure, temperature, x, rho):
        pass


class Fenghour1998(Viscosity):
    """
    Correlation for CO2 viscosity: Fenghour, Wakeham & Vesovic (1998) - The Viscosity of CO2
    """

    a = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
    d = [
        0.4071119e-2,
        0.7198037e-4,
        0.2411967e-16,
        0.2971072e-22,
        -0.1627888e-22,
    ]  # d11, d21, d64, d81, d82

    def __init__(self):
        super().__init__()

    def evaluate(self, pressure, temperature, x, rho):
        # Viscosity in zero density limit
        eps_k = 251.196  # energy scaling parameter eps/k [K]
        T_ = 1 / eps_k * temperature
        lnT_ = np.log(T_)

        lnG = self.a[0]
        for i in range(1, 5):
            lnG += self.a[i] * lnT_**i
        G = np.exp(lnG)

        n0 = 1.00697 * np.sqrt(temperature) / G

        # Excess viscosity
        dn = (
            self.d[0] * rho
            + self.d[1] * rho**2
            + self.d[2] * rho**6 / (T_**3)
            + self.d[3] * rho**8
            + self.d[4] * rho**8 / T_
        )

        # Correction of viscosity in vicinity of critical point
        dnc = 0

        n = (n0 + dn + dnc) * 1e-3  # muPa.s to cP
        return n


class LBC(Viscosity):
    """
    Condensate / liquid viscosity – Lohrenz, Bray & Clark (1964)

    References:
        SPE915 - Calculating viscosities of Reservoir Fluids from Their Compositions
        https://courses.ems.psu.edu/png520/m19_p4.html

    """

    def __init__(self, comp_data, property_container, heptane_plus=False, a=None):
        """
        comp_data : comp_data class, contains critical pressure/temperature, critical volumes, and molecular weights
        a : list, LBC coefficients
        pahse : str, phase name to reference in the property_container
        pc : DartsModel.property_container

        """
        # super().__init__()

        self.pc = property_container
        self.comp_data = comp_data

        if 'wat' in self.pc.phases_name:
            self.nc = len(comp_data.components) - 1
        else:
            self.nc = len(comp_data.components)

        # lump heptane plus fraction
        self.heptane_plus = heptane_plus

        # LBC coefficients
        if a is None:
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = (
                0.1023,
                0.023364,
                0.058533,
                -0.040758,
                0.0093724,
                -1e-4,
            )
        else:
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = (
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
            )

        # conversion factors
        self.bar2psia = 14.5037738
        self.K2Rankine = 1.8
        self.kgm3_2_lbmft3 = 6.243e-2

        self.Pc = np.array(comp_data.Pc) * self.bar2psia  # critical pressure
        self.Tc = np.array(comp_data.Tc) * self.K2Rankine  # crotocal temperature
        if self.heptane_plus:
            self.SG = np.array(comp_data.SG)  # specific gravity
        else:
            pass
        self.Mw = np.array(comp_data.Mw)
        # self.Vc = np.array(comp_data.Vc) # critical volumes

        # Pure-component critical molar volumes Vc in ft^3/lbmol
        if hasattr(comp_data, "Vc"):
            self.Vc = np.array(comp_data.Vc)  # critical volumes
            # assume m^3/kmol -> ft^3/lbmol
            self.Vc_ft3lbmol = np.asarray(self.Vc, dtype=float) * (
                35.3146667 / 2.20462262185
            )
        else:
            ValueError(
                'Missing pure-component critical molar volumes Vc in comp_data().'
            )

    def evaluate(self, pressure, temperature, x, rho):
        rhoL_si = rho
        temperature *= self.K2Rankine

        # ---- Breakup of the heptane plus fraction
        if self.heptane_plus:
            C7plus_idx = self.Mw > 100
            x_C7plus = np.sum(x[C7plus_idx])
            if 1:
                M_C7plus = (
                    np.sum(x[C7plus_idx] * self.Mw[C7plus_idx]) / x_C7plus
                )  # weighted average
                SG_C7plus = np.sum(x[C7plus_idx] * self.SG[C7plus_idx]) / x_C7plus
            else:
                M_C7plus = 138.9024
                SG_C7plus = 0.727
            Vc_C7plus = (
                21.573
                + 0.015122 * M_C7plus
                - 27.656 * SG_C7plus
                + 0.070615 * M_C7plus * SG_C7plus
            )

        # ---- Calculation of the low-pressure pure components gas viscosities
        XI = (5.4402 * self.Tc ** (1 / 6)) / (
            np.sqrt(self.Mw) * self.Pc ** (2 / 3)
        )  # viscosity reduction parameter, eq. 19.27
        Trj = temperature / self.Tc  # reduced temperature
        mu_j = np.empty(self.nc, dtype=float)
        for i in range(self.nc):
            if Trj[i] < 1.5:
                mu_j[i] = (34e-5 * (Trj[i] ** 0.94)) / XI[i]
            else:
                mu_j[i] = (17.78e-5 * (4.58 * Trj[i] - 1.67) ** 0.625) / XI[i]

        # ---- Calculation of the low pressure mixture gas viscosity (Herning & Zipperer)
        mu_ = np.sum(x * mu_j * np.sqrt(self.Mw)) / np.sum(
            x * np.sqrt(self.Mw)
        )  # eq. 19.27

        # ---- Mixture pseudo-properties (Kay’s mixing rules)
        Tpc = np.sum(x * self.Tc)
        Ppc = np.sum(x * self.Pc)
        # Vpc = np.sum(x * self.Vc)

        # ---- Reduced density of the mixture
        # rhoL_si = self.pc.dens[pc.phases_name.index(self.phase)]
        rhoL_lbmft3 = rhoL_si * self.kgm3_2_lbmft3
        MW_mix = float(np.sum(x * self.Mw))  # lbm/lbmol

        if self.heptane_plus:
            rho_r = (
                rhoL_lbmft3
                * (
                    np.sum(x[~C7plus_idx] * self.Vc_ft3lbmol[~C7plus_idx])
                    + x_C7plus * Vc_C7plus
                )
                / MW_mix
            )
        else:
            rho_r = rhoL_lbmft3 * np.sum(x * self.Vc_ft3lbmol) / MW_mix

        XI_m = (5.4402 * Tpc ** (1 / 6)) / (
            np.sqrt(MW_mix) * Ppc ** (2 / 3)
        )  # eq. 19.28

        # ---- Calculation of the viscosity, Jossi, Stiel & Thodos (1962)
        dense = (
            (
                self.a0
                + self.a1 * rho_r
                + self.a2 * (rho_r**2)
                + self.a3 * (rho_r**3)
                + self.a4 * (rho_r**4)
            )
            ** 4
            + self.a5
        ) / XI_m
        mu = mu_ + dense  # 19.26

        return mu


class Lee1966(Viscosity):
    """
    Correlation for gas mixture viscosity: Lee et al. (1966) - The Viscosity of Natural Gases
    """

    def __init__(self, components: list, Mw: list):
        super().__init__(components)
        self.Mw = Mw

    def evaluate(self, pressure, temperature, x, rho):
        MW = np.sum(x * self.Mw) * 1e-3  # kg/mol

        T_ran = temperature * 1.8  # rankine scale
        a = (9.379 + 0.0160 * MW) * T_ran**1.5 / (209.2 + 19.26 * MW + T_ran)
        b = 3.448 + 0.01009 * MW + (986.4 / T_ran)
        c = 2.447 - 0.2224 * b
        rho_lb = rho * 0.0624279606  # convert rho from [kg / m3] -> [lb / ft ^ 3]
        return 1e-4 * a * np.exp(b * (rho_lb / 62.43) ** c)


class MaoDuan2009(Viscosity):
    """
    Correlation for brine + NaCl viscosity: Mao and Duan (2009) - The Viscosity of Aqueous Alkali-Chloride Solutions
                                                                  up to 623 K, 1,000 bar, and High Ionic Strength
    """

    # # Data from Mao and Duan (2009) for pure water viscosity
    # mu_d = [0.28853170e7, -0.11072577e5, -0.90834095e1, 0.30925651e-1, -0.27407100e-4,
    #         -0.19283851e7, 0.56216046e4, 0.13827250e2, -0.47609523e-1, 0.35545041e-4]

    # Data from Islam and Carlson (2012) for pure water viscosity
    mu_a = 9.03591045e1
    mu_b = [3.40285740e4, 8.23556123e8, -9.28022905e8]
    mu_c = [1.40090092e-2, 4.86126399e-2, 5.26696663e-2]
    mu_d = [-1.22757462e-1, 2.15995021e-2, -3.65253919e-4, 1.97270835e-6]

    # Data from Islam and Carlson (2012) for pure water density
    rho_a = 1.34136579e2
    rho_b = [-4.07743800e3, 1.63192756e4, 1.37091355e3]
    rho_c = [-5.56126409e-3, -1.07149234e-2, -5.46294495e-4]
    rho_d = [4.45861703e-1, -4.51029739e-4]

    # Data from Mao and Duan (2009) for relative viscosity of solutions of NaCl
    a = [-0.21319213, 0.13651589e-2, -0.12191756e-5]
    b = [0.69161945e-1, -0.27292263e-3, 0.20852448e-6]
    c = [-0.25988855e-2, 0.77989227e-5]

    def __init__(self, components: list, ions: list = None, combined_ions: list = None):
        super().__init__(components, ions)

        self.H2O_idx = components.index("H2O")
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, x, rho):
        # Density of pure water (Islam and Carlson, 2012)
        rhoH2O = self.rho_a
        for i in range(3):
            rhoH2O += self.rho_b[i] * 10 ** (self.rho_c[i] * temperature)
        for i in range(2):
            rhoH2O += self.rho_d[i] * pressure ** (i + 1)

        # Viscosity of pure water (Islam and Carlson, 2012)
        muH2O = self.mu_a
        for i in range(3):
            muH2O += self.mu_b[i] * np.exp(-self.mu_c[i] * temperature)
        for i in range(4):
            muH2O += pressure * 0.1 * self.mu_d[i] * (temperature - 293.15) ** i

        # # Viscosity of pure water (Mao and Duan, 2009)
        # muH2O = 0.
        # for i in range(5):
        #     muH2O += self.mu_d[i] * temperature ** (i-3)
        # for i in range(5, 10):
        #     muH2O += self.mu_d[i] * rhoH2O * temperature ** (i-8)
        # muH2O = np.exp(muH2O)

        # Relative viscosity of solution (Mao and Duan, 2009)
        A = self.a[0] + self.a[1] * temperature + self.a[2] * temperature**2
        B = self.b[0] + self.b[1] * temperature + self.b[2] * temperature**2
        C = self.c[0] + self.c[1] * temperature
        if self.combined_ions is not None:
            m = 55.509 * x[self.nc] / x[self.H2O_idx]
        else:
            m = (
                np.sum(
                    [
                        55.509 * x[i] / x[self.H2O_idx]
                        for i in range(self.nc, self.nc + self.ni)
                    ]
                )
                * 0.5
            )  # half because sum of ions molality is double NaCl molality

        mu_r = np.exp(A * m + B * m**2 + C * m**3)
        mu = mu_r * muH2O  # Pa.s

        return mu * 1e-3


class Islam2012(MaoDuan2009):
    """
    Correlation for H2O + NaCl + CO2 viscosity: Islam and Carlson (2012) - Viscosity Models and Effects of Dissolved CO2
    """

    def __init__(self, components: list, ions: list = None, combined_ions: list = None):
        super().__init__(components, ions, combined_ions)

        self.CO2_idx = components.index("CO2") if "CO2" in components else None

    def evaluate(self, pressure, temperature, x, rho):
        mu_brine = super().evaluate(pressure, temperature, x, rho)

        # Viscosity of Aq + CO2
        if self.CO2_idx is not None:
            mu_brine *= 1.0 + 4.65 * x[self.CO2_idx] ** 1.0134

        return mu_brine
