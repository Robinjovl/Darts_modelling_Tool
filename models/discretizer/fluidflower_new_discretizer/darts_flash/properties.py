import numpy as np
# from flash import value_vector, index_vector, string_vector
from flash import AQProperties, VLProperties, HProperties, SProperties
from flash import VdWP


class PropertyCorrelation:
    def __init__(self, components: list):
        self.components = components
        self.nc = len(components)


class Density(PropertyCorrelation):
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x):
        pass


class Viscosity(PropertyCorrelation):
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x, rho):
        pass


class ViscosityCO2(Viscosity):
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x, rho):
        """
        Correlation for CO2 viscosity: Fenghour, Wakeham & Vesovic (1998)
        """
        # Viscosity in zero density limit
        eps_k = 251.196  # energy scaling parameter eps/k [K]
        T_ = 1/eps_k * temperature
        lnT_ = np.log(T_)

        a = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
        lnG = a[0]
        for i in range(1, 5):
            lnG += a[i] * lnT_ ** i
        G = np.exp(lnG)

        n0 = 1.00697*np.sqrt(temperature) / G

        # Excess viscosity
        d = [0.4071119e-2, 0.7198037e-4, 0.2411967e-16, 0.2971072e-22, -0.1627888e-22]  # d11, d21, d64, d81, d82
        dn = d[0] * rho + d[1] * rho ** 2 + d[2] * rho ** 6 / (T_ ** 3) + d[3] * rho ** 8 + d[4] * rho ** 8 / T_

        # Correction of viscosity in vicinity of critical point
        dnc = 0

        n = (n0 + dn + dnc) * 1e-3  # muPa.s to cP
        return n


class ViscosityAq(Viscosity):
    def __init__(self, components: list):
        super().__init__(components)

    def evaluate(self, pressure, temperature, x, rho):
        """
        Correlation for brine + NaCl + CO2 viscosity
        Brine + NaCl + CO2: Islam (2012)
        """
        # Viscosity of pure water
        a = 9.03591045e1
        b = [3.40285740e4, 8.23556123e8, -9.28022905e8]
        c = [1.40090092e-2, 4.86126399e-2, 5.26696663e-2]
        d = [-1.22757462e-1, 2.15995021e-2, -3.65253919e-4, 1.97270835e-6]
        muH2O = a
        for i in range(3):
            muH2O += b[i] * np.exp(-c[i] * temperature)
        for i in range(4):
            muH2O += pressure*0.1 * d[i] * (temperature-293.15)**i

        # Viscosity of H2O + salt
        mu_r = 1
        mu = mu_r * muH2O

        # Viscosity of Aq + CO2
        if "CO2" in self.components:
            mu *= 1 + 4.65 * pow(x[self.components.index("CO2")], 1.0134)

        return mu * 1e-3


# region Aq
class AqProperties:
    ion_charge = {"Na+": 1, "Cl-": -1, "Ca+2": 2, "CO3-2": -2, "I-": -1}

    def __init__(self, components: list, ions: list = None, combined_ions: list = None, eos="AQ3", max_m=0.):
        # Components and ions
        self.components = components
        self.ions = ions if ions is not None else []
        self.species = self.components + self.ions
        self.nc = len(self.components)
        self.ni = len(self.ions)

        self.prop_container = AQProperties(self.components, self.ions, eos, max_m)

        # If ions are combined in composition vector, determine fraction for each separate ion
        if combined_ions is not None:
            self.combined_ions = []
            for i, idxs in enumerate(combined_ions):
                ion_stoich = []
                for idx in idxs:
                    ion_stoich.append(np.abs(self.ion_charge[ions[idx]]))
                ion_stoich[:] /= np.sum(ion_stoich)
                self.combined_ions.append(ion_stoich)
        else:
            self.combined_ions = None

    def separate_ions(self, x):
        """
        If combined ions, un-lump them
        """
        if self.combined_ions is not None:
            X = np.zeros(len(self.species))

            # Regular comp are the same
            X[:self.nc] = x[:self.nc]

            # Separate the combined ions according to stoichiometry
            ii = 0
            for i, ion_stoich in enumerate(self.combined_ions):
                for weight in ion_stoich:
                    X[self.nc+ii] = x[self.nc+i] * weight
                    ii += 1
            return X
        else:
            return x


class AqDensity(AqProperties):
    def __init__(self, components: list, ions: list = None, combined_ions: list = None, max_m=0.):
        super().__init__(components, ions, combined_ions, max_m=max_m)

    def evaluate(self, p, T, x):
        X = self.separate_ions(x)
        return self.prop_container.density(p, T, X)


class AqViscosity(AqProperties):
    def __init__(self, components: list, ions: list = None, combined_ions: list = None, max_m=0.):
        super().__init__(components, ions, combined_ions, max_m=max_m)

    def evaluate(self, p, T, x, rho):
        X = self.separate_ions(x)
        return self.prop_container.viscosity(p, T, X, rho)


class AqEnthalpy(AqProperties):
    def __init__(self, components: list, ions: list = None, combined_ions: list = None, max_m=0.):
        super().__init__(components, ions, combined_ions, max_m=max_m)

    def evaluate(self, p, T, x):
        X = self.separate_ions(x)
        return self.prop_container.enthalpy(p, T, X) * 1000  # kJ/kmol


class AqConductivity(AqProperties):
    def __init__(self, components: list, ions: list = None, combined_ions: list = None, max_m=0.):
        super().__init__(components, ions, combined_ions, max_m=max_m)

    def evaluate(self, p, T, x, rho):
        X = self.separate_ions(x)
        return self.prop_container.conductivity(p, T, X, rho)

# endregion


# region VL
class VlProperties:
    def __init__(self, components: list, eos: str, comp_data: dict):
        self.prop_container = VLProperties(components, eos, comp_data)


class VLDensity(VlProperties):
    def __init__(self, components: list, eos: str, comp_data: dict):
        super().__init__(components, eos, comp_data)

    def evaluate(self, p, T, x):
        return self.prop_container.density(p, T, x)


class VViscosity(VlProperties):
    def __init__(self, components: list, eos: str, comp_data: dict):
        super().__init__(components, eos, comp_data)

    def evaluate(self, p, T, x, rho):
        return self.prop_container.viscosity(p, T, x, rho)


class VLEnthalpy(VlProperties):
    def __init__(self, components: list, eos: str, comp_data: dict):
        super().__init__(components, eos, comp_data)

    def evaluate(self, p, T, x):
        return self.prop_container.enthalpy(p, T, x) * 1000  # kJ/kmol


class VConductivity(VlProperties):
    def __init__(self, components: list, eos: str, comp_data: dict):
        super().__init__(components, eos, comp_data)

    def evaluate(self, p, T, x, rho):
        return self.prop_container.conductivity(p, T, x, rho)
# endregion


# region H
class HydrateProperties:
    def __init__(self, components: list, eos="VdWP", hydrate_type="sI", x: list = None):
        self.components = components
        self.hydrate_type = hydrate_type

        self.prop = HProperties(self.components, eos, hydrate_type)
        self.H_eos = VdWP(self.components, hydrate_type)

        self.water_idx = self.components.index("H2O")
        self.guest_idx = 0 if self.water_idx == 1 else 1
        self.x = x
        if x is not None:
            self.nH = 1/x[self.guest_idx] - 1
        else:
            self.nH = None

    def calc_df(self, pressure, temperature, f0):
        # Calculate fugacity difference
        self.H_eos.component_parameters(pressure, temperature)
        fwH = self.H_eos.fwH(f0)

        df = fwH - f0[self.water_idx]  # if df < 0 formation, if df > 0 dissociation
        xH = self.H_eos.xH()

        return df, xH


class HDensity(HydrateProperties):
    def __init__(self, components: list, eos: str = "VdWP", hydrate_type: str = "sI", x: list = None):
        super().__init__(components, eos, hydrate_type, x)

    def evaluate(self, p, T, x):
        if self.x is not None:
            return self.prop.density(p, T, self.x)
        else:
            return self.prop.density(p, T, x)


class HEnthalpy(HydrateProperties):
    def __init__(self, components: list, eos: str = "VdWP", hydrate_type: str = "sI", x: list = None):
        super().__init__(components, eos, hydrate_type, x)

    def evaluate(self, p, T, x):
        if self.x is not None:
            return (self.nH + 1) * self.prop.enthalpy(p, T, self.x) * 1000
        else:
            nH = 1/x[self.guest_idx] - 1
            return (nH + 1) * self.prop.enthalpy(p, T, x) * 1000


class HConductivity(HydrateProperties):
    def __init__(self, components: list, eos: str = "VdWP", hydrate_type: str = "sI", x: list = None):
        super().__init__(components, eos, hydrate_type, x)

    def evaluate(self, p, T, x, rho):
        if self.x is not None:
            return self.prop.conductivity(p, T, self.x, rho)
        else:
            return self.prop.conductivity(p, T, x, rho)


class HydrateReactionRate(HydrateProperties):
    def __init__(self, components: list, eos: str = "VdWP", hydrate_type: str = "sI", x: list = None, k=3.11e12, reac_enthalpy=0, perm=300, poro=0.2, stoich: list = None):
        super().__init__(components, eos, hydrate_type, x)

        # input parameters for kinetic rate
        self.stoich = stoich
        self.perm = perm
        self.poro = poro
        self.k = k

        self.reac_enthalpy = reac_enthalpy
        if self.nH is not None:
            self.mH = self.nH * 18.015 + 16.043  # g/mol -> kg/kmol
        else:
            self.mH = None

    def evaluate(self, pressure, temperature, f0, sh):
        # Calculate fugacity difference and hydrate composition
        df, xH = self.calc_df(pressure, temperature, f0)

        # Reaction rate following Yin (2018)
        # Constants needs to be determined through history matching with experiments
        K = self.k  # reaction constant [kmol/(m^2 bar day)]

        # surface area following Yin (2018)
        F_A = 1
        perm = self.perm * 1E-15  # mD to m2
        r_p = np.sqrt(45 * perm * (1 - self.poro) ** 2 / (self.poro ** 3))
        A_s = 0.879 * F_A * (1 - self.poro) / r_p * sh ** (2 / 3)  # hydrate surface area [m2]

        # Thermodynamic parameters
        dE = -81E3  # activation energy [J/mol]
        R = 8.3145  # gas constant [J/(K.mol)]

        # # K is reaction cons, A_s hydrate surface area, dE activation energy, driving force is fugacity difference
        kinetic_rate = K * A_s * np.exp(dE / (R * temperature)) * df

        rate = [stoich * kinetic_rate for stoich in self.stoich]

        return rate

    def evaluate_enthalpy(self, pressure, temperature, f0, sh):
        if self.reac_enthalpy:
            rates = self.evaluate(pressure, temperature, f0, sh)

            # Enthalpy change with dissociation (-ive, rate of hydrate component +ive)
            Cf = 33.72995  # J/kg cal/gmol
            if temperature - 273.15 > 0:
                (C1, C2) = (13521, -4.02)
            else:
                (C1, C2) = (6534, -11.97)
            en = Cf * (C1 + C2 / temperature) / 1000  # kJ/kg

            if self.mH is not None:
                mH = self.mH
            else:
                raise Exception("Not implemented")
                # x = None
                # nH = 1/x[self.guest_idx] - 1
                # mH = nH * 18.015 + 16.043
            H_diss = -en * mH  # kJ/kg * kg/kmol -> kJ/kmol

            return rates[2] * H_diss  # rate * enth/mol
        else:
            return 0.
# endregion


# region S
class SolidProperties:
    def __init__(self, solid_type: str):
        self.prop_container = SProperties(solid_type)


class SolidDensity(SolidProperties):
    def __init__(self, solid_type: str):
        super().__init__(solid_type)

    def evaluate(self, p, T, dummy1=0):
        return self.prop_container.density(p, T)


class SolidEnthalpy(SolidProperties):
    def __init__(self, solid_type: str):
        super().__init__(solid_type)

    def evaluate(self, p, T, dummy1=0):
        return self.prop_container.density(p, T)


class SolidConductivity(SolidProperties):
    def __init__(self, solid_type: str):
        super().__init__(solid_type)

    def evaluate(self, p, T, dummy1=0, dummy2=0):
        return self.prop_container.density(p, T)
# endregion
