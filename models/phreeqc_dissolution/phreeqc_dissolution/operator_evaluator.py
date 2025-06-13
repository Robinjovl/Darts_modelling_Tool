from darts.physics.base.operators_base import OperatorsBase
from darts.engines import *
import os.path as osp
import numpy as np

physics_name = osp.splitext(osp.basename(__file__))[0]

class my_own_acc_flux_etor(OperatorsBase):
    """
    Reservoir operators working with the following state:
    state: (pressure, overall mineral molar fractions, overall fluid molar fractions)
    """
    def __init__(self, input_data, properties):
        super().__init__(properties, thermal=properties.thermal)
        # Store your input parameters in self here, and initialize other parameters here in self
        self.input_data = input_data
        self.min_z = input_data.min_z
        self.temperature = input_data.temperature
        self.exp_w = input_data.exp_w
        self.exp_g = input_data.exp_g
        self.kin_fact = input_data.kin_fact
        self.property = properties
        self.counter = 0

        # Operator order
        self.ACC_OP = 0  # accumulation operator - ne
        self.FLUX_OP = self.ACC_OP + self.ne  # flux operator - ne * nph
        self.UPSAT_OP = self.FLUX_OP + self.ne * self.nph  # saturation operator (diffusion/conduction term) - nph
        self.GRAD_OP = self.UPSAT_OP + self.nph  # gradient operator (diffusion/conduction term) - ne * nph
        self.KIN_OP = self.GRAD_OP + self.ne * self.nph  # kinetic operator - ne

        # extra operators
        self.GRAV_OP = self.KIN_OP + self.ne  # gravity operator - nph
        self.PC_OP = self.GRAV_OP + self.nph  # capillary operator - nph
        self.PORO_OP = self.PC_OP + self.nph  # porosity operator - 1
        self.ENTH_OP = self.PORO_OP + 1  # enthalpy operator - nph
        self.TEMP_OP = self.ENTH_OP + self.nph  # temperature operator - 1
        self.PRES_OP = self.TEMP_OP + 1
        self.n_ops = self.PRES_OP + 1

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

    def get_overall_composition(self, state):
        if self.thermal:
            z = state[1:-1]
        else:
            z = state[1:]
        z_last = min(max(1 - np.sum(z[self.property.flash_ev.fc_mask[:-1]]), self.min_z), 1 - self.min_z)
        z = np.concatenate([z, [z_last]])
        return z

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # state and values numpy vectors:
        state_np = state.to_numpy()
        values_np = values.to_numpy()

        # pore pressure
        pressure = state_np[0]
        # get overall molar composition
        z = self.get_overall_composition(state_np)

        # call property:
        self.property.evaluate(state_np)

        # Densities
        rho_t = np.sum(self.property.dens_m * self.property.sat_overall[:self.property.nph]) + \
                self.property.dens_m_solid * self.property.sat_overall[self.property.nph]
        rho_f = np.sum(self.property.dens_m * self.property.sat)

        nc = self.property.nc
        ns = self.property.n_solid
        nph = 2
        ne = nc

        """ CONSTRUCT OPERATORS HERE """
        values_np[:] = 0.
        
        """ Alpha operator represents accumulation term: """
        values_np[self.ACC_OP:self.ACC_OP + ns] = z[:ns] * rho_t
        values_np[self.ACC_OP + ns:self.ACC_OP + nc] = (1 - self.property.sat_overall[self.property.nph]) * z[ns:] * rho_f

        """ Beta operator represents flux term: """
        for j in range(nph):
            values_np[self.FLUX_OP + j * self.ne:self.FLUX_OP + j * self.ne + nc] = self.property.x[j] * self.property.dens_m[j] * self.property.kr[j] / self.property.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        for j in range(nph):
            values_np[self.UPSAT_OP + j] = self.property.rock_compr.mean() * self.property.sat[j]

        """ Chi operator for diffusion """
        for j in self.property.ph:
            values_np[self.GRAD_OP + j * self.ne:self.GRAD_OP + (j + 1) * self.ne] = self.property.diffusivity[j] * \
                                                                    self.property.x[j] * self.property.dens_m[j]

        """ Delta operator for reaction """
        for i in range(ne):
            values_np[self.KIN_OP + i] = (self.input_data.stoich_matrix[:, i] * self.property.kin_rates).sum()

        """ Gravity and Capillarity operators """
        # E3-> gravity
        for i in range(nph):
            values_np[self.GRAV_OP + i] = 0

        # E4-> capillarity
        for i in range(nph):
            values_np[self.PC_OP + i] = 0

        # E5_> porosity
        values_np[self.PORO_OP] = 1 - self.property.sat_overall[self.property.nph]

        return 0

class my_own_comp_etor(my_own_acc_flux_etor):
    """
    Operator required for initialization, to convert given volume fraction to molar one
    DIFFERENT state: (pressure, overall mineral volume fractions, fluid molar fractions)
    """
    def __init__(self, input_data, properties):
        super().__init__(input_data, properties)  # Initialize base-class
        self.fluid_mole = self.property.flash_ev.total_moles / 1000 # mol to kmol
        self.counter = 0
        self.props_name = ['z_solid']

    def evaluate(self, state, values):
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        pressure = state_np[0]
        s_minerals = state_np[self.property.s_mask_state]
        ss = s_minerals.sum() # volume fraction in initialization

        # initial flash, non-standard argument
        _, _, _, _, _, fluid_volume, _ = self.property.flash_ev.evaluate(state_np)

        # evaluate molar fraction
        solid_volume = fluid_volume * ss / (1 - ss)         # m3
        mineral_volume = s_minerals * (fluid_volume + solid_volume)
        mineral_mole = np.array([mineral_volume[i] * self.property.rock_density_ev[k].evaluate(pressure) / self.property.Mw[k]
                                 for i, k in enumerate(self.property.rock_density_ev.keys())])
        nu_m = mineral_mole / (mineral_mole.sum() + self.fluid_mole)
        values_np[:self.property.n_solid] = nu_m

        return 0

class my_own_property_evaluator(operator_set_evaluator_iface):
    def __init__(self, input_data, properties):
        # Initialize base-class
        super().__init__()
        self.input_data = input_data
        self.property = properties
        self.props_name = (['z' + prop for prop in properties.flash_ev.phreeqc_species] + ['satV'] +
                           ['Act(H+)', 'Act(CO2)'] + ['SR_' + mineral for mineral in self.property.flash_ev.mineral_names])

    def evaluate(self, state, values):
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        nu_v, _, _, rho_phases, kin_state, _, molar_fractions = self.property.flash_ev.evaluate(state_np)
        values_np[:molar_fractions.size] = molar_fractions

        # gas saturation
        nu_s_minerals = state_np[self.property.s_mask_state]
        nu_s = nu_s_minerals.sum()
        nu_s_rho_s = np.array([nu_s_minerals[i] / v.evaluate(state_np[0]) * self.property.Mw[k]
                      for i, (k, v) in enumerate(self.property.rock_density_ev.items())]).sum()
        nu_v = nu_v * (1 - nu_s)  # convert to overall molar fraction
        nu_a = 1 - nu_v - nu_s
        rho_a, rho_v = rho_phases['aq'], rho_phases['gas']
        if nu_v > 0:
            sv = nu_v / rho_v / (nu_v / rho_v + nu_a / rho_a + nu_s_rho_s)
        else:
            sv = 0
        values_np[molar_fractions.size] = sv

        # extra kinetic props
        values_np[molar_fractions.size + 1] = kin_state['Act(H+)']
        values_np[molar_fractions.size + 2] = kin_state['Act(CO2)']
        for i, mineral in enumerate(self.property.flash_ev.mineral_names):
            values_np[molar_fractions.size + 3 + i] = kin_state['SR_' + mineral]

        return 0
