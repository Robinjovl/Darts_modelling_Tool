import numpy as np
from darts.engines import *
from select_para import *
from properties import *
import os.path as osp

physics_name = osp.splitext(osp.basename(__file__))[0]

# Define our own operator evaluator class
class AccFluxGravityEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.num_comp = property_container.nc
        self.min_z = property_container.min_z
        self.phases = property_container.phase_name
        self.components = property_container.component_name

        self.property = property_container

        self.c_r = 1e-7
        self.p_ref = 1

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc

        interpr = properties_evaluator(self.property)
        (self.sat, self.x, self.rho, self.rho_m, self.mu, self.kr, self.FM) = interpr.evaluate(state)

        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock

        density_tot = np.sum(self.sat * self.rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        """ CONSTRUCT OPERATORS HERE """
        """ Alpha operator represents accumulation term: """
        for i in range(self.num_comp):
            values[i] = self.compr * density_tot * zc[i]

        """ Beta operator represents flux term: """
        for j in range(nph):
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] * self.FM[j] / self.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in range(nph):
            values[shift + j] = self.compr * self.sat[j]

        """ Chi operator for diffusion """
        shift += nph
        for i in range(nc):
            for j in range(nph):
                values[shift + i * nph + j] = 0.0001728 * self.rho_m[j] * self.x[j][i]

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(ne):
            values[shift + i] = 0

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = self.rho[i]

        # E4-> capillarity
        for i in range(nph):
            values[shift + 3 + nph + i] = 0

        # E5_> porosity
        values[shift + 3 + 2 * nph] = 0

        return 0

class AccFluxGravityWellEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.num_comp = property_container.nc
        self.min_z = property_container.min_z
        self.phases = property_container.phase_name
        self.components = property_container.component_name

        self.property = property_container

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nc = self.property.nc
        nph = self.property.nph
        ne = nc

        interpr = properties_evaluator(self.property)
        (self.sat, self.x, self.rho, self.rho_m, self.mu, self.kr, self.FM) = interpr.evaluate(state)

        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock

        density_tot = np.sum(self.sat * self.rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        """ CONSTRUCT OPERATORS HERE """
        """ Alpha operator represents accumulation term: """
        for i in range(self.num_comp):
            values[i] = self.compr * density_tot * zc[i]

        """ Beta operator represents flux term: """
        for j in range(nph):
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in range(nph):
            values[shift + j] = self.compr * self.sat[j]

        """ Chi operator for diffusion """
        shift += nph
        for j in range(nph):
            for i in range(nc):
                values[shift + ne * j + i] = 0.0001728 * self.rho_m[j] * self.x[j][i]

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(ne):
            values[shift + i] = 0

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = 0

        # E4-> capillarity
        for i in range(nph):
            values[shift + 3 + nph + i] = 0
        # E5_> porosity
        values[shift + 3 + 2 * nph] = 0

        return 0

class RateEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.num_comp = property_container.nc
        self.min_z = property_container.min_z
        self.phases = property_container.phase_name
        self.components = property_container.component_name

        self.property = property_container

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        interpr = properties_evaluator(self.property)
        (self.sat, self.x, self.rho, self.rho_m, self.mu, self.kr, self.FM) = interpr.evaluate(state)

        """ CONSTRUCT RATE OPERATORS HERE """
        num_rate_op = self.num_comp  # two p two c just for this case, otherwise need to import phases number

        # step-1
        flux = np.zeros((num_rate_op, 1))
        for i in range(num_rate_op):
            flux[i] = self.rho_m[0] * self.kr[0] * self.x[0][i] / self.mu[0] + self.rho_m[1] * self.kr[1] * self.x[1][i] / self.mu[1]
        # step-2
        flux_sum = np.sum(flux)
        # step-3
        total_density = sum(self.sat*self.rho_m)
        # step-4
        values[0] = self.sat[0] * flux_sum / total_density
        values[1] = self.sat[1] * flux_sum / total_density

        return 0

""" used for plotting in main.py and for properties evaluation in operators """
class properties_evaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.num_comp = property_container.nc
        self.min_z = property_container.min_z
        self.phases = property_container.phase_name
        self.components = property_container.component_name

        self.property = property_container

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        zc = self.comp_out_of_bounds(zc)

        # Perform Flash procedure here:
        (x, y, V) = self.property.flash_ev.flash(vec_state_as_np)
        Mt_aq = 0
        Mt_g = 0
        # molar weight of mixture
        for i in range(self.num_comp):
            Mt_aq = Mt_aq + props(self.components[i], 'Mw') * x[i]
            Mt_g = Mt_g + props(self.components[i], 'Mw') * y[i]

        # print('Mt_aq:', Mt_aq, ', Mt_gas:', Mt_g)
        rho_g = 0
        rho_aq = 0

        mu_g = 1
        mu_aq = 1

        rho_g_m = 0
        rho_aq_m = 0

        """" PROPERTIES Evaluation """
        if V <= 1e-2:
            sg = 0
            # single aqueous phase
            x = zc
            rho_aq = self.property.density_ev['wat'].evaluate(x)  # output in [kg/m3]
            mu_aq = self.property.viscosity_ev['wat'].evaluate()
            rho_aq_m = rho_aq / Mt_aq
            MRF = 1

            # print('firstly, I am in single-phase region and now zc = ', zc)

        elif V >= 1:
            sg = 1
            # single vapor phase
            y = zc
            rho_g = self.property.density_ev['gas'].evaluate(pressure)  # in [kg/m3]
            mu_g = self.property.viscosity_ev['gas'].evaluate()
            rho_g_m = rho_g / Mt_g
            MRF = 1

        else:
            # two phases
            rho_aq = self.property.density_ev['wat'].evaluate(x)  # output in [kg/m3]
            mu_aq = self.property.viscosity_ev['wat'].evaluate()  # output in [cp]
            rho_g = self.property.density_ev['gas'].evaluate(pressure)  # in [kg/m3]
            mu_g = self.property.viscosity_ev['gas'].evaluate()  # in [cp]
            rho_aq_m = rho_aq / Mt_aq
            rho_g_m = rho_g / Mt_g
            sg = rho_aq_m / (rho_g_m / V - rho_g_m + rho_aq_m)  # saturation using [Kmol/m3]
            MRF = self.property.foam_STARS_FM_ev.evaluate(sg)

            # print('I am in two-phase region and now zc = ', zc, '\t V = ', V)

        kr_aq = self.property.rel_perm_ev['wat'].evaluate(1 - sg)
        kr_g = self.property.rel_perm_ev['gas'].evaluate(sg)

        sat = np.array([sg, 1 - sg], dtype=object)
        xx = np.array([y, x], dtype=object)
        rho = np.array([rho_g, rho_aq], dtype=object)
        rho_m = np.array([rho_g_m, rho_aq_m], dtype=object)
        mu = np.array([mu_g, mu_aq], dtype=object)
        kr = np.array([kr_g, kr_aq], dtype=object)
        FM = np.array([MRF, 1], dtype=object)

        # print('state:', state, 'Density:', rho_aq)
        # print('state:', state, 'Viscosity:', mu_aq)

        return sat, xx, rho, rho_m, mu, kr, FM

class PropertyEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.num_comp = property_container.nc
        self.min_z = property_container.min_z
        self.phases = property_container.phase_name
        self.components = property_container.component_name

        self.property = property_container

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        zc = self.comp_out_of_bounds(zc)

        # Perform Flash procedure here:
        (x, y, V) = self.property.flash_ev.flash(vec_state_as_np)
        Mt_aq = 0
        Mt_g = 0
        # molar weight of mixture
        for i in range(self.num_comp):
            Mt_aq = Mt_aq + props(self.components[i], 'Mw') * x[i]
            Mt_g = Mt_g + props(self.components[i], 'Mw') * y[i]

        # print('Mt_aq:', Mt_aq, ', Mt_gas:', Mt_g)
        rho_g = 0
        rho_aq = 0

        mu_g = 1
        mu_aq = 1

        rho_g_m = 0
        rho_aq_m = 0

        """" PROPERTIES Evaluation """
        if V <= 1e-2:
            sg = 0
            # single aqueous phase
            x = zc
            rho_aq = self.property.density_ev['wat'].evaluate(x)  # output in [kg/m3]
            mu_aq = self.property.viscosity_ev['wat'].evaluate()
            rho_aq_m = rho_aq / Mt_aq

        elif V >= 1:
            sg = 1
            # single vapor phase
            y = zc
            rho_g = self.property.density_ev['gas'].evaluate(pressure)  # in [kg/m3]
            mu_g = self.property.viscosity_ev['gas'].evaluate()
            rho_g_m = rho_g / Mt_g

        else:
            # two phases
            rho_aq = self.property.density_ev['wat'].evaluate(x)  # output in [kg/m3]
            mu_aq = self.property.viscosity_ev['wat'].evaluate()  # output in [cp]
            rho_g = self.property.density_ev['gas'].evaluate(pressure)  # in [kg/m3]
            mu_g = self.property.viscosity_ev['gas'].evaluate()  # in [cp]
            rho_aq_m = rho_aq / Mt_aq
            rho_g_m = rho_g / Mt_g
            sg = rho_aq_m / (rho_g_m / V - rho_g_m + rho_aq_m)  # saturation using [Kmol/m3]

        self.rho = np.array([rho_g, rho_aq], dtype=object)
        self.rho_m = np.array([rho_g_m, rho_aq_m], dtype=object)
        self.x = np.array([y, x], dtype=object)

        values[0] = sg
        values[1] = x[0]
        values[2] = rho_aq
        values[3] = rho_g
        values[4] = mu_aq

        return 0