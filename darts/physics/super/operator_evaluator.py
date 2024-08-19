import numpy as np
import jax
import jax.numpy as jnp

from darts.engines import operator_set_evaluator_iface, value_vector
from darts.physics.operators_base import OperatorsBase
from darts.physics.super.property_container import PropertyContainer


class OperatorsSuper(OperatorsBase):
    property: PropertyContainer

    def __init__(self, property_container: PropertyContainer, thermal: bool):
        super().__init__(property_container, thermal)  # Initialize base-class

        self.min_z = property_container.min_z
        self.nc_fl = property_container.nc_fl
        self.ns = property_container.ns
        self.np_fl = property_container.np_fl

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
            idx0, idx1 = self.FLUX_OP + j * self.ne, self.FLUX_OP + (j + 1) * self.ne
            print("BETA (flux) {}".format(j), values[idx0:idx1])
        print("GAMMA (diffusion)", values[self.UPSAT_OP:self.GRAD_OP])
        for j in range(self.nph):
            idx0, idx1 = self.GRAD_OP + j * self.ne, self.GRAD_OP + (j + 1) * self.ne
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
        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        for i in range(self.n_ops):
            values[i] = 0

        # Evaluate isothermal properties at current state
        self.property.evaluate(state)
        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(self.property.sat[:self.np_fl] * self.property.dens_m[:self.np_fl])
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        self.phi_f = 1. - np.sum(zc[self.nc_fl:])

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        for i in range(self.nc_fl):
            # fluid mass accumulation: c_r phi^T z_c* [-] rho_m^T [kmol/m3]
            values[self.ACC_OP + i] = self.compr * density_tot * zc[i]

        """ and alpha for solid components """
        for i in range(self.ns):
            # solid mass accumulation: c_r phi^T z_s* [-] rho_ms [kmol/m3]
            values[self.ACC_OP + self.nc_fl + i] = self.compr * self.property.dens_m[self.np_fl + i] * zc[self.nc_fl + i]

        """ Beta operator represents flux term: """
        for j in self.property.ph:
            for i in range(self.nc_fl):
                # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] k_rj [-] / mu_j [cP ∝ bar.day] (kmol/m3.bar.day)
                values[self.FLUX_OP + j * self.ne + i] = (self.property.x[j][i] * self.property.dens_m[j] *
                                                          self.property.kr[j] / self.property.mu[j])

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        for j in self.property.ph:
            # fluid diffusive flux sat: c_r phi_f s_j (-)
            values[self.UPSAT_OP + j] = self.compr * self.phi_f * self.property.sat[j]
        for i in range(self.ns):
            # solid conductive flux sat: c_r z_s* (-)
            j = self.np_fl + i
            values[self.UPSAT_OP + j] = self.compr * zc[self.nc_fl + i]

        """ Chi operator for diffusion """
        for j in self.property.ph:
            D = self.property.diffusion_ev[self.property.phases_name[j]].evaluate()
            for i in range(self.nc_fl):
                # fluid diffusive flux: D_cj [m2/day] x_cj [-] rho_mj [kmol/m3] (kmol/m.day)
                values[self.GRAD_OP + j * self.ne + i] = D[i] * self.property.x[j][i] * self.property.dens_m[j]

        """ Delta operator for reaction """
        for i in range(self.nc):
            # fluid/solid mass source: dt [day] n_c [kmol/m3.day] (kmol/m3)
            values[self.KIN_OP + i] = self.property.mass_source[i]

        """ Gravity and Capillarity operators """
        # E3-> gravity
        for j in self.property.ph:
            values[self.GRAV_OP + j] = self.property.dens[j]

        # E4-> capillarity
        for j in self.property.ph:
            values[self.PC_OP + j] = self.property.pc[j]

        # E5_> fluid porosity
        values[self.PORO_OP] = self.phi_f

        if self.thermal:
            self.evaluate_thermal(state, values)

        # self.print_operators(state, values)

        return 0

class ReservoirOperatorsJAX(ReservoirOperators):
    def evaluate_jax(self, state):
        values = jnp.zeros(self.n_ops)

        # Evaluate isothermal properties at current state
        pressure = state[0]
        props = self.property.evaluate_jax(state)
        compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = jnp.sum(props['saturation'][:self.np_fl] * props['molar_density'][:self.np_fl])
        zc = jnp.append(state[1:self.nc], 1 - jnp.sum(state[1:self.nc]))
        phi_f = 1. - jnp.sum(zc[self.nc_fl:])

        # Construct operators
        # Alpha operator for fluid and solid components
        fluid_acc = compr * density_tot * zc[:self.nc_fl]
        solid_acc = compr * props['molar_density'][self.np_fl:self.np_fl+self.ns] * zc[self.nc_fl:self.nc]
        values = values.at[self.ACC_OP : self.ACC_OP + self.nc_fl].set(fluid_acc)
        values = values.at[self.ACC_OP + self.nc_fl : self.ACC_OP + self.nc_fl + self.ns].set(solid_acc)

        # Beta operator for flux terms
        for j in self.property.ph:
            for i in range(self.nc_fl):
                flux_index = self.FLUX_OP + j * self.ne + i
                flux_value = (self.property.x[j][i] * props['molar_density'][j] * \
                              props['relative_permeability'][j] / props['viscosity'][j])
                values = values.at[flux_index].set(flux_value)

        # Gamma operator for diffusion, including solid components
        for j in self.property.ph:
            upsat_value = compr * phi_f * props['saturation'][j]
            values = values.at[self.UPSAT_OP + j].set(upsat_value)

        # Including solid conductive flux saturation
        for i in range(self.ns):
            upsat_solid_index = self.UPSAT_OP + self.np_fl + i
            upsat_solid_value = compr * zc[self.nc_fl + i]
            values = values.at[upsat_solid_index].set(upsat_solid_value)

        # Chi operator for diffusion
        for j in self.property.ph:
            for i in range(self.nc_fl):
                diff_index = self.GRAD_OP + j * self.ne + i
                D = self.property.diffusion_ev[self.property.phases_name[j]].evaluate()  # This must be JAX compatible
                diff_value = D[i] * self.property.x[j][i] * props['molar_density'][j]
                values = values.at[diff_index].set(diff_value)

        # Delta operator for reaction
        for i in range(self.nc):
            values = values.at[self.KIN_OP + i].set(self.property.mass_source[i])

        # Gravity and Capillarity operators
        for j in self.property.ph:
            values = values.at[self.GRAV_OP + j].set(props['density'][j])
            values = values.at[self.PC_OP + j].set(props['capillary_pressure'][j])

        # Fluid porosity
        values = values.at[self.PORO_OP].set(phi_f)

        return values

class MassFluxOperators(OperatorsSuper):
    def __init__(self, property_container: PropertyContainer, thermal: bool):
        super().__init__(property_container, thermal)  # Initialize base-class

        self.n_ops = self.nph * self.nc_fl

    def evaluate(self, state: value_vector, values: value_vector):
        for i in range(self.n_ops):
            values[i] = 0

        self.property.evaluate(state)

        """ Beta operator here represents mass flux term: """
        for j in range(self.nph):
            for i in range(self.nc_fl):
                values[self.nc_fl * j + i] = self.property.x[j][i] * self.property.dens[j] * kr[j] / mu[j]


class GeomechanicsReservoirOperators(ReservoirOperators):
    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Reservoir operators
        super().evaluate(state, values)

        # Rock density operator
        # TODO: function of matrix pressure = I1 / 3 = (s_xx + s_yy + s_zz) / 3
        values[self.PORO_OP + 1] = self.property.rock_density_ev.evaluate()

        return 0

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        super().print_operators(state, values)
        print("ROCK DENSITY", values[self.PORO_OP + 1])
        return


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

        for i in range(self.n_ops):
            values[i] = 0

        self.property.evaluate(state)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(self.property.sat[:self.np_fl] * self.property.dens_m[:self.np_fl])
        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))
        self.phi_f = 1.

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        for i in range(self.nc_fl):
            # fluid mass accumulation: c_r phi^T z_c* [-] rho_m^T [kmol/m3]
            values[self.ACC_OP + i] = self.compr * density_tot * zc[i]

        """ and alpha for solid components """
        for i in range(self.ns):
            # solid mass accumulation: c_r phi^T z_s* [-] rho_ms [kmol/m3]
            values[self.ACC_OP + self.nc_fl + i] = self.compr * self.property.dens_m[self.np_fl + i] * zc[self.nc_fl + i]

        """ Beta operator represents flux term: """
        for j in self.property.ph:
            for i in range(self.nc_fl):
                # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] k_rj [-] / mu_j [cP ∝ bar.day] (kmol/m3.bar.day)
                values[self.FLUX_OP + j * self.ne + i] = (self.property.x[j][i] * self.property.dens_m[j] *
                                                          self.property.kr[j] / self.property.mu[j])

        """ Gamma operator for diffusion (same for thermal and isothermal) """

        """ Chi operator for diffusion """

        """ Delta operator for reaction """
        for i in range(self.nc):
            # fluid/solid mass source: dt [day] n_c [kmol/m3.day] (kmol/m3)
            values[self.KIN_OP + i] = self.property.mass_source[i]

        """ Gravity and Capillarity operators """
        # E3-> gravity
        for j in range(self.np_fl):
            values[self.GRAV_OP + j] = self.property.dens[j]

        # E5_> porosity
        values[self.PORO_OP] = 1.

        if self.thermal:
            self.evaluate_thermal(state, values)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state: value_vector, values: value_vector):
        return


class SinglePhaseGeomechanicsOperators(OperatorsBase):
    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """

        self.property.evaluate(state)
        values[0] = self.property.dens[0]
        values[1] = self.property.dens[0] / self.property.mu[0]

        return 0


class RateOperators(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()

        self.nc = property_container.nc
        self.nc_fl = property_container.nc_fl
        self.nph = property_container.nph
        self.np_fl = property_container.np_fl
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

        self.property.evaluate(state)

        flux = np.zeros(self.nc_fl)
        # step-1
        for j in self.property.ph:
            for i in range(self.nc_fl):
                flux[i] += self.property.dens_m[j] * self.property.kr[j] * self.property.x[j][i] / self.property.mu[j]
        # step-2
        flux_sum = np.sum(flux)

        # (sat_sc, dens_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        sat_sc = self.property.sat[:self.np_fl]
        dens_m_sc = self.property.dens_m[:self.np_fl]

        # step-3
        total_density = np.sum(sat_sc * dens_m_sc)
        # step-4
        for j in self.property.ph:
            values[j] = self.property.dens_m[j] * self.property.kr[j] / self.property.mu[j]
            # sat_sc[j] * flux_sum / total_density

        # print(state, values)
        return 0
