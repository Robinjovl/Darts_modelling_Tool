import numpy as np
from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.super.property_container import PropertyContainer


class ReservoirOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal

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
        nm = self.property.nm
        nc_fl = nc - nm
        ne = nc + self.thermal

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        #  some arrays will be reused in thermal
        self.ph, self.sat, self.x, rho, self.rho_m, self.mu, self.kr, pc, mass_source = self.property.evaluate(state)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(self.sat * self.rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        phi = 1 - np.sum(zc[nc_fl:nc])

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        for i in range(nc_fl):
            values[i] = self.compr * density_tot * zc[i]
        
        """ and alpha for mineral components """
        for i in range(nm):
            values[i + nc_fl] = self.property.solid_dens[i] * zc[i + nc_fl]

        """ Beta operator represents flux term: """
        for j in self.ph:
            shift = ne + ne * j
            for i in range(nc_fl):
                values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in self.ph:
            values[shift + j] = self.compr * self.sat[j]

        """ Chi operator for diffusion """
        shift += nph
        for i in range(nc):
            for j in self.ph:
                values[shift + i * nph + j] = self.property.diff_coef * self.x[j][i] * self.rho_m[j]

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(nc):
            values[shift + i] = mass_source[i]

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in self.ph:
            values[shift + 3 + i] = rho[i]

        # E4-> capillarity
        for i in self.ph:
            values[shift + 3 + nph + i] = pc[i]
        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        #print(state, values)
        # self.print_operators(state, values)

        return 0

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        print("================================================")
        print("STATE", state)
        print("ALPHA (accumulation)", values[0:ne])
        for j in self.ph:
            print("BETA (flux) {}".format(j), values[(ne + ne * j):(ne + ne * (j+1))])
        print("GAMMA (diffusion)", values[(ne + ne * nph):(ne + ne * nph + nph + nph * ne)])
        print("DELTA (reaction)", values[(ne + ne * nph + nph + nph * ne):(ne + ne * nph + nph + nph * ne + ne)])
        print("GRAVITY", values[(ne + ne * nph + nph + nph * ne + ne + 3):(ne + ne * nph + nph + nph * ne + ne + 3 + nph)])
        print("CAPILLARITY", values[(ne + ne * nph + nph + nph * ne + ne + 3 + nph):(ne + ne * nph + nph + nph * ne + ne + 3 + nph + nph)])
        print("POROSITY", values[(ne + ne * nph + nph + nph * ne + ne + 3 + nph + nph)])
        print("ROCK ENERGY", values[(ne + ne * nph + nph + nph * ne + ne):(ne + ne * nph + nph + nph * ne + ne + 3)])
        return


class ReservoirThermalOperators(ReservoirOperators):
    def __init__(self, property_container, thermal=1):
        super().__init__(property_container, thermal=thermal)  # Initialize base-class

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        super().evaluate(state, values)

        vec_state_as_np = np.asarray(state)
        pressure = state[0]
        temperature = vec_state_as_np[-1]

        rock_energy = self.property.rock_energy_ev.evaluate(temperature=temperature)
        enthalpy, cond, energy_source = self.property.evaluate_thermal(state)

        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        i = nc  # use this numeration for energy operators
        """ Alpha operator represents accumulation term: """
        for m in self.ph:
            values[i] += self.compr * self.sat[m] * self.rho_m[m] * enthalpy[m]  # fluid enthalpy (kJ/m3)
        values[i] -= self.compr * 100 * pressure

        """ Beta operator represents flux term: """
        for j in self.ph:
            shift = ne + ne * j
            values[shift + i] = enthalpy[j] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Chi operator for temperature in conduction, gamma operators are skipped """
        shift = ne + ne * nph + nph
        for j in range(nph):
            # values[shift + nc * nph + j] = temperature
            values[shift + ne * j + nc] = temperature * cond[j]

        """ Delta operator for reaction """
        shift += nph * ne
        values[shift + i] = energy_source

        """ Additional energy operators """
        shift += ne
        # E1-> rock internal energy
        values[shift] = rock_energy / self.compr  # (T-T_0), multiplied by rock hcap inside engine
        # E2-> rock temperature
        values[shift + 1] = temperature
        # E3-> rock conduction
        values[shift + 2] = 1 / self.compr  # multiplied by rock cond inside engine

        #print(state, values)
        # self.print_operators(state, values)

        return 0


class WellOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal

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
        nm = self.property.nm
        nc_fl = nc - nm
        ne = nc + self.thermal

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(sat * rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        phi = 1

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        for i in range(nc_fl):
            values[i] = self.compr * density_tot * zc[i]

        """ and alpha for mineral components """
        for i in range(nm):
            values[i + nc_fl] = self.property.solid_dens[i] * zc[i + nc_fl]

        """ Beta operator represents flux term: """
        for j in ph:
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = x[j][i] * rho_m[j] * kr[j] / mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph

        """ Chi operator for diffusion """
        shift += nph

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(ne):
            values[shift + i] = mass_source[i]

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = rho[i]

        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        #print(state, values)
        return 0


class RateOperators(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.flux = np.zeros(self.nc)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        for i in range(self.nph):
            values[i] = 0

        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        self.flux[:] = 0
        # step-1
        for j in ph:
            for i in range(self.nc):
                self.flux[i] += rho_m[j] * kr[j] * x[j][i] / mu[j]
        # step-2
        flux_sum = np.sum(self.flux)

        #(sat_sc, rho_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        sat_sc = sat
        rho_m_sc = rho_m

        # step-3
        total_density = np.sum(sat_sc * rho_m_sc)
        # step-4
        for j in ph:
            values[j] = sat_sc[j] * flux_sum / total_density

        #print(state, values)
        return 0


class PropertyOperators(operator_set_evaluator_iface):
    """
    This class contains a set of operators for evaluation of properties.
    An interpolator is created in the :class:`Physics` object to rapidly obtain properties after simulation.

    :ivar prop_idxs: Dictionary of indices for each property as they are returned in property_container.evaluate()
    :type prop_idxs: dict
    """
    prop_idxs = {"sat": 1, "x": 2, "rho": 3, "rho_m": 4, "mu": 5, "kr": 6, "pc": 7, "mass_source": 8}

    def __init__(self, props: list, property_container: PropertyContainer):
        """
        This is the constructor for PropertyOperator.
        The properties to be obtained from the PropertyContainer are passed as a list of tuples.

        :param props: List of tuples with ('name', 'key', index), where key must match self.prop_dict and index is for phase, (component)
        :type props: list[tuple]
        :param property_container: PropertyContainer object to evaluate properties at given state
        """
        super().__init__()  # Initialize base-class

        self.property_container = property_container

        self.props = props
        self.n_props = len(props)
        self.props_name = [prop[0] for prop in props]

    def evaluate(self, state: value_vector, values: value_vector):
        """
        This function evaluates the properties at given `state` (P,z) or (P,T,z) from the :class:`PropertyContainer` object.
        The user-specified properties are stored in the `values` object.

        :param state: Vector of state variables [pres, comp_0, ..., comp_N-1, (temp)]
        :type state: darts.engines.value_vector
        :param values: Vector for storage of operator values
        :type values: darts.engines.value_vector
        """
        prop_output = self.property_container.evaluate(state)

        for i, prop in enumerate(self.props):
            prop_idx = self.prop_idxs[prop[1]]
            values[i] = prop_output[prop_idx][prop[2]]

        return 0
