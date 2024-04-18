import numpy as np
from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.operators_base import OperatorsBase
from darts.physics.super.property_container import PropertyContainer


class OperatorsSuper(OperatorsBase):
    def __init__(self, property_container: PropertyContainer, thermal: bool):
        super().__init__(property_container, thermal)  # Initialize base-class

        self.min_z = property_container.min_z

        # Operator order
        self.ACC_OP = 0  # accumulation operator - ne
        self.FLUX_OP = self.ACC_OP + self.ne  # flux operator - ne * nph
        self.UPSAT_OP = self.FLUX_OP + self.ne * self.nph  # saturation operator (diffusion/conduction term) - nph
        self.GRAD_OP = self.UPSAT_OP + self.nph  # gradient operator (diffusion/conduction term) - ne * nph
        self.KIN_OP = self.GRAD_OP + self.ne * self.nph  # kinetic operator - ne

        # extra operators
        self.RE_INTER_OP = self.KIN_OP + self.ne  # rock internal energy operator - 1
        self.RE_TEMP_OP = self.RE_INTER_OP + 1  # rock temperature operator - 1
        self.ROCK_COND = self.RE_INTER_OP + 2  # rock conduction operator - 1
        self.GRAV_OP = self.RE_INTER_OP + 3  # gravity operator - nph
        self.PC_OP = self.RE_INTER_OP + 3 + self.nph  # capillary operator - nph
        self.PORO_OP = self.RE_INTER_OP + 3 + 2 * self.nph  # porosity operator - 1
        self.n_ops = self.PORO_OP + 1

    def print_operators(self, state: value_vector, values: value_vector):
        """Method for printing operators, grouped"""
        print("================================================")
        print("STATE", state)
        print("ALPHA (accumulation)", values[self.ACC_OP:self.FLUX_OP])
        for j in range(self.nph):
            idx0, idx1 = self.FLUX_OP + j * self.ne, self.FLUX_OP + (j+1) * self.ne
            print("BETA (flux) {}".format(j), values[idx0:idx1])
        print("GAMMA (diffusion)", values[self.UPSAT_OP:self.GRAD_OP])
        for j in range(self.nph):
            idx0, idx1 = self.GRAD_OP + j * self.ne, self.GRAD_OP + (j+1) * self.ne
            print("CHI (diffusion) {}".format(j), values[idx0:idx1])
        print("DELTA (reaction)", values[self.KIN_OP:self.RE_INTER_OP])
        print("GRAVITY", values[self.GRAV_OP:self.PC_OP])
        print("CAPILLARITY", values[self.PC_OP:self.PORO_OP])
        print("POROSITY", values[self.PORO_OP])
        if self.thermal:
            print("ROCK ENERGY, TEMP, COND", values[self.RE_INTER_OP:self.GRAV_OP])
        return


class ReservoirOperators(OperatorsSuper):
    def evaluate(self, state: value_vector, values: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nm = self.property.nm
        nc_fl = self.nc - nm

        for i in range(self.n_ops):
            values[i] = 0

        #  some arrays will be reused in thermal
        self.ph, self.sat, self.x, rho, self.rho_m, self.mu, self.kr, pc, mass_source = self.property.evaluate(state)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(self.sat * self.rho_m)
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        phi = 1 - np.sum(zc[nc_fl:self.nc])

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        for i in range(nc_fl):
            values[self.ACC_OP + i] = self.compr * density_tot * zc[i]
        
        """ and alpha for mineral components """
        for i in range(nm):
            values[self.ACC_OP + nc_fl + i] = self.property.solid_dens[i] * zc[nc_fl + i]

        """ Beta operator represents flux term: """
        for j in self.ph:
            for i in range(nc_fl):
                values[self.FLUX_OP + j * self.ne + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        for j in self.ph:
            values[self.UPSAT_OP + j] = self.compr * self.sat[j]

        """ Chi operator for diffusion """
        for j in self.ph:
            D = self.property.diffusion_ev[self.property.phases_name[j]].evaluate()
            for i in range(nc_fl):
                values[self.GRAD_OP + j * self.ne + i] = D[i] * self.x[j][i] * self.rho_m[j]

        """ Delta operator for reaction """
        for i in range(self.nc):
            values[self.KIN_OP + i] = mass_source[i]

        """ Gravity and Capillarity operators """
        # E3-> gravity
        for j in self.ph:
            values[self.GRAV_OP + j] = rho[j]

        # E4-> capillarity
        for j in self.ph:
            values[self.PC_OP + j] = pc[j]
        # E5_> porosity
        values[self.PORO_OP] = phi

        if self.thermal:
            self.evaluate_thermal(state, values)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state: value_vector, values: value_vector):
        vec_state_as_np = np.asarray(state)
        pressure = state[0]
        temperature = vec_state_as_np[-1]

        rock_energy = self.property.rock_energy_ev.evaluate(temperature=temperature)
        enthalpy, cond, energy_source = self.property.evaluate_thermal(state)

        """ Alpha operator represents accumulation term: """
        for m in self.ph:
            values[self.ACC_OP + self.nc] += self.compr * self.sat[m] * self.rho_m[m] * enthalpy[
                m]  # fluid enthalpy (kJ/m3)
        values[self.ACC_OP + self.nc] -= self.compr * 100 * pressure

        """ Beta operator represents flux term: """
        for j in self.ph:
            values[self.FLUX_OP + j * self.ne + self.nc] = enthalpy[j] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Chi operator for temperature in conduction """
        for j in self.ph:
            values[self.GRAD_OP + j * self.ne + self.nc] = temperature * cond[j]

        """ Delta operator for reaction """
        values[self.KIN_OP + self.nc] = energy_source

        """ Additional energy operators """
        # E1-> rock internal energy
        values[self.RE_INTER_OP] = rock_energy / self.compr  # (T-T_0), multiplied by rock hcap inside engine
        # E2-> rock temperature
        values[self.RE_TEMP_OP] = temperature
        # E3-> rock conduction
        values[self.ROCK_COND] = 1 / self.compr  # multiplied by rock cond inside engine

        return 0


class WellOperators(OperatorsSuper):
    def evaluate(self, state: value_vector, values: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        nm = self.property.nm
        nc_fl = self.nc - nm

        for i in range(self.n_ops):
            values[i] = 0

        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(sat * rho_m)
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        phi = 1

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        for i in range(nc_fl):
            values[self.ACC_OP + i] = self.compr * density_tot * zc[i]

        """ and alpha for mineral components """
        for i in range(nm):
            values[self.ACC_OP + nc_fl + i] = self.property.solid_dens[i] * zc[nc_fl + i]

        """ Beta operator represents flux term: """
        for j in ph:
            for i in range(self.nc):
                values[self.FLUX_OP + j * self.ne + i] = x[j][i] * rho_m[j] * kr[j] / mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """

        """ Chi operator for diffusion """

        """ Delta operator for reaction """
        for i in range(self.nc):
            values[self.KIN_OP + i] = mass_source[i]

        """ Gravity and Capillarity operators """
        # E3-> gravity
        for j in range(self.nph):
            values[self.GRAV_OP + j] = rho[j]

        # E5_> porosity
        values[self.PORO_OP] = phi

        if self.thermal:
            self.evaluate_thermal(state, values)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state: value_vector, values: value_vector):
        return


class RateOperators(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()

        self.nc = property_container.nc
        self.nph = property_container.nph
        self.n_ops = property_container.nph

        self.property = property_container

    def evaluate(self, state: value_vector, values: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        for i in range(self.n_ops):
            values[i] = 0

        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        flux = np.zeros(self.nc)
        # step-1
        for j in ph:
            for i in range(self.nc):
                flux[i] += rho_m[j] * kr[j] * x[j][i] / mu[j]
        # step-2
        flux_sum = np.sum(flux)

        #(sat_sc, rho_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        sat_sc = sat
        rho_m_sc = rho_m

        # step-3
        total_density = np.sum(sat_sc * rho_m_sc)
        # step-4
        for j in ph:
            values[j] = rho_m[j] * kr[j] / mu[j]
            #sat_sc[j] * flux_sum / total_density

        # print(state, values)
        return 0
