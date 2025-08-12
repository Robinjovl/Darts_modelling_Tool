import numpy as np
from darts.physics.properties.basic import RockCompactionEvaluator


class PropertyContainer:
    def __init__(self, phase_name, component_name, eps_z, Mw, temperature=None, rock_comp=1e-5):
        self.nph = len(phase_name)
        self.nc = len(component_name)
        self.components = component_name
        self.phases = phase_name
        self.eps_z = eps_z
        self.Mw = Mw

        if temperature is not None:  # constant T specified
            self.thermal = False
            self.temperature = temperature
        else:
            self.thermal = True
            self.temperature = None

        # Allocate (empty) evaluators
        self.density_ev = {}
        self.viscosity_ev = {}
        self.rel_perm_ev = {}
        self.flash_ev = 0
        self.foam_STARS_FM_ev = []
        self.rock_compr_ev = RockCompactionEvaluator(compres=rock_comp)

    def get_state(self, state):
        """
        Get tuple of (pressure, temperature, [z0, ... zn-1]) at current OBL point (state)
        If isothermal, temperature returns initial temperature
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        if zc[-1] < 0:
            zc = self.comp_out_of_bounds(zc)

        if self.thermal:
            temperature = vec_state_as_np[-1]
        else:
            temperature = self.temperature

        return pressure, temperature, zc

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.eps_z:
                vec_composition[ith_comp] = self.eps_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.eps_z:
                vec_composition[ith_comp] = 1 - self.eps_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.eps_z)
        return vec_composition

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        # Perform Flash procedure here:
        self.flash_ev.evaluate(pressure, temperature, zc)
        V = self.flash_ev.getnu()
        x = np.array(self.flash_ev.getx())

        # print('Mt_aq:', Mt_aq, ', Mt_gas:', Mt_g)
        rho_g = 0
        rho_aq = 0

        mu_g = 1
        mu_aq = 1

        rho_g_m = 0
        rho_aq_m = 0

        """" PROPERTIES Evaluation """
        if V[0] <= 1e-2:
            sg = 0
            # single aqueous phase
            x[0, :] = np.zeros(2)
            x[1, :] = zc
            Mt_aq = np.sum(self.Mw * x[1, :])

            rho_aq = self.density_ev['wat'].evaluate(pressure, temperature, x[1, :])  # output in [kg/m3]
            mu_aq = self.viscosity_ev['wat'].evaluate(pressure, temperature, x[1, :])
            rho_aq_m = rho_aq / Mt_aq
            MRF = 1

            # print('firstly, I am in single-phase region and now zc = ', zc)

        elif V[0] >= 1:
            sg = 1
            # single vapor phase
            x[0, :] = zc
            x[1, :] = np.zeros(2)
            Mt_g = np.sum(self.Mw * x[0, :])

            rho_g = self.density_ev['gas'].evaluate(pressure, temperature, x[0, :])  # in [kg/m3]
            mu_g = self.viscosity_ev['gas'].evaluate(pressure, temperature, x[0, :])
            rho_g_m = rho_g / Mt_g
            MRF = 1

        else:
            # two phases
            Mt_aq = np.sum(self.Mw * x[1, :])
            Mt_g = np.sum(self.Mw * x[0, :])

            rho_aq = self.density_ev['wat'].evaluate(pressure, temperature, x[1, :])  # output in [kg/m3]
            mu_aq = self.viscosity_ev['wat'].evaluate(pressure, temperature, x[1, :])  # output in [cp]
            rho_g = self.density_ev['gas'].evaluate(pressure, temperature, x[0, :])  # in [kg/m3]
            mu_g = self.viscosity_ev['gas'].evaluate(pressure, temperature, x[0, :])  # in [cp]
            rho_aq_m = rho_aq / Mt_aq
            rho_g_m = rho_g / Mt_g
            sg = rho_aq_m / (rho_g_m / V[0] - rho_g_m + rho_aq_m)  # saturation using [Kmol/m3]
            MRF = self.foam_STARS_FM_ev.evaluate(sg)

            # print('I am in two-phase region and now zc = ', zc, '\t V = ', V)

        kr_aq = self.rel_perm_ev['wat'].evaluate(1 - sg)
        kr_g = self.rel_perm_ev['gas'].evaluate(sg)

        sat = np.array([sg, 1 - sg], dtype=object)
        rho = np.array([rho_g, rho_aq], dtype=object)
        rho_m = np.array([rho_g_m, rho_aq_m], dtype=object)
        mu = np.array([mu_g, mu_aq], dtype=object)
        kr = np.array([kr_g, kr_aq], dtype=object)
        FM = np.array([MRF, 1], dtype=object)

        # print('state:', state, 'Density:', rho_aq)
        # print('state:', state, 'Viscosity:', mu_aq)

        return sat, x, rho, rho_m, mu, kr, FM
