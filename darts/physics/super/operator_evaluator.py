import numpy as np

from darts.engines import operator_set_evaluator_iface
from darts.physics.base.operators_base import OperatorsBase
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
        self.UPSAT_OP = (
            self.FLUX_OP + self.ne * self.nph
        )  # saturation operator (diffusion/conduction term) - nph
        self.GRAD_OP = (
            self.UPSAT_OP + self.nph
        )  # gradient operator (diffusion/conduction term) - ne * nph
        self.KIN_OP = self.GRAD_OP + self.ne * self.nph  # kinetic operator - ne

        # extra operators
        self.GRAV_OP = self.KIN_OP + self.ne  # gravity operator - nph
        self.PC_OP = self.GRAV_OP + self.nph  # capillary operator - nph
        self.MULT_OP = self.PC_OP + self.nph  # permeability multiplier operator - 1
        self.ENTH_OP = self.MULT_OP + 1  # enthalpy operator - nph
        self.TEMP_OP = self.ENTH_OP + self.nph  # temperature operator - 1
        self.PRES_OP = self.TEMP_OP + 1
        self.n_ops = self.PRES_OP + 1

        # Operator names
        self.op_names = [
            (self.ACC_OP, "ACC"),
            (self.FLUX_OP, "FLUX"),
            (self.UPSAT_OP, "UPSAT"),
            (self.GRAD_OP, "GRAD"),
            (self.KIN_OP, "KIN"),
            (self.GRAV_OP, "GRAV"),
            (self.MULT_OP, "MULT"),
            (self.ENTH_OP, "ENTH"),
            (self.TEMP_OP, "TEMP"),
            (self.PRES_OP, "PRES"),
        ]

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        print("================================================")
        print("STATE", state)
        print("ALPHA (accumulation)", values[self.ACC_OP : self.FLUX_OP])
        for j in range(self.nph):
            idx0, idx1 = self.FLUX_OP + j * self.ne, self.FLUX_OP + (j + 1) * self.ne
            print("BETA (flux) {}".format(j), values[idx0:idx1])
        print("GAMMA (diffusion)", values[self.UPSAT_OP : self.GRAD_OP])
        for j in range(self.nph):
            idx0, idx1 = self.GRAD_OP + j * self.ne, self.GRAD_OP + (j + 1) * self.ne
            print("CHI (diffusion) {}".format(j), values[idx0:idx1])
        print("DELTA (reaction)", values[self.KIN_OP : self.GRAV_OP])
        print("GRAVITY", values[self.GRAV_OP : self.PC_OP])
        print("CAPILLARITY", values[self.PC_OP : self.MULT_OP])
        print("ENTHALPY", values[self.ENTH_OP : self.ENTH_OP + self.nph])
        print("PERM_MULT", values[self.MULT_OP])
        print("TEMPERATURE, PRESSURE", values[self.TEMP_OP], values[self.PRES_OP])
        return


class ReservoirOperators(OperatorsSuper):
    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values): value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        # Evaluate isothermal properties at current state
        self.property.evaluate(vec_state_as_np)
        self.compr = self.property.rock_compr_ev.evaluate(vec_state_as_np[0])

        density_tot = np.sum(
            self.property.sat[: self.np_fl] * self.property.dens_m[: self.np_fl]
        )
        zc = np.append(
            vec_state_as_np[1 : self.nc], 1 - np.sum(vec_state_as_np[1 : self.nc])
        )
        self.phi_s = np.sum(zc[self.nc_fl :])
        self.phi_f = 1.0 - self.phi_s

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        # fluid mass accumulation: c_r phi^T z_c* [-] rho_m^T [kmol/m3]
        vec_values_as_np[self.ACC_OP : self.ACC_OP + self.nc_fl] = (
            self.compr * density_tot * zc[: self.nc_fl]
        )

        """ and alpha for mineral components """
        vec_values_as_np[
            self.ACC_OP + self.nc_fl : self.ACC_OP + self.nc_fl + self.ns
        ] = (
            self.compr
            * self.property.dens_m[self.np_fl : self.np_fl + self.ns]
            * zc[self.nc_fl : self.nc_fl + self.ns]
        )

        """ Beta operator represents flux term: """
        for j in self.property.ph:
            # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] k_rj [-] / mu_j [cP ∝ bar.day] (kmol/m3.bar.day)
            vec_values_as_np[
                self.FLUX_OP + j * self.ne : self.FLUX_OP + j * self.ne + self.nc_fl
            ] = (
                self.property.x[j][: self.nc_fl]
                * self.property.dens_m[j]
                * self.property.kr[j]
                / self.property.mu[j]
            )

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        # fluid diffusive flux sat: c_r phi_f s_j (-)
        vec_values_as_np[self.UPSAT_OP + self.property.ph] = (
            self.compr * self.phi_f * self.property.sat[self.property.ph]
        )
        # solid diffusive flux sat: c_r z_s* (-)
        vec_values_as_np[
            self.UPSAT_OP + self.np_fl : self.UPSAT_OP + self.np_fl + self.ns
        ] = (self.compr * zc[self.nc_fl : self.nc_fl + self.ns])

        """ Chi operator for diffusion """
        for j in self.property.ph:
            D = self.property.diffusion_ev[self.property.phases_name[j]].evaluate()
            # fluid diffusive flux: D_cj [m2/day] x_cj [-] rho_mj [kmol/m3] (kmol/m.day)
            vec_values_as_np[
                self.GRAD_OP + j * self.ne : self.GRAD_OP + j * self.ne + self.nc_fl
            ] = (
                D[: self.nc_fl]
                * self.property.x[j][: self.nc_fl]
                * self.property.dens_m[j]
            )

        """ Delta operator for reaction """
        # fluid/solid mass source: dt [day] n_c [kmol/m3.day] (kmol/m3)
        vec_values_as_np[self.KIN_OP : self.KIN_OP + self.nc] = (
            self.property.mass_source
        )

        """ Gravity and Capillarity operators """
        # E3-> gravity
        vec_values_as_np[self.GRAV_OP + self.property.ph] = self.property.dens[
            self.property.ph
        ]

        # E4-> capillarity
        vec_values_as_np[self.PC_OP + self.property.ph] = self.property.pc[
            self.property.ph
        ]

        # E5_> permeability multiplier due to permporo relationship
        vec_values_as_np[self.MULT_OP] = self.property.permporo_mult_ev.evaluate(
            self.phi_f
        )

        # Pressure operator (for generic state specification where no pressure in the state, for instance V,T)
        vec_values_as_np[self.PRES_OP] = vec_state_as_np[0]

        if self.thermal:
            self.evaluate_thermal(vec_state_as_np, vec_values_as_np)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state, values):
        """
        Method to evaluate operators for energy conservation equation

        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        pressure = state[0]
        temperature = state[-1]

        # Evaluate thermal properties at current state
        self.property.evaluate_thermal(state)

        """ Alpha operator represents accumulation term: """
        # fluid enthalpy: s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += (
            self.compr
            * self.phi_f
            * np.sum(
                self.property.sat[self.property.ph]
                * self.property.dens_m[self.property.ph]
                * self.property.enthalpy[self.property.ph]
            )
        )  # fluid enthalpy (kJ/m3)
        # solid enthalpy: s_j [-] rho_mj [kmol/m3] H_j [kJ/kmol] (kJ/m3)
        values[self.ACC_OP + self.nc] += (
            self.compr
            * self.phi_s
            * np.sum(
                self.property.sat[self.np_fl : self.np_fl + self.ns]
                * self.property.dens_m[self.np_fl : self.np_fl + self.ns]
                * self.property.enthalpy[self.np_fl : self.np_fl + self.ns]
            )
        )
        # Enthalpy to internal energy conversion
        values[self.ACC_OP + self.nc] -= self.compr * 100 * pressure

        """ Beta operator represents flux term: """
        # fluid convective energy flux: H_j [kJ/kmol] rho_mj [kmol/m3] k_rj [-] / mu_j [cP ∝ bar.day] (kJ/m3.bar.day)
        values[self.FLUX_OP + self.property.ph * self.ne + self.nc] = (
            self.property.enthalpy[self.property.ph]
            * self.property.dens_m[self.property.ph]
            * self.property.kr[self.property.ph]
            / self.property.mu[self.property.ph]
        )

        """ Chi operator for temperature in conduction """
        # fluid/solid conductive flux: kappa_j [kJ/m.K.day] T [K] (kJ/m.day)
        values[self.GRAD_OP + self.property.ph * self.ne + self.nc] = (
            temperature * self.property.cond[self.property.ph]
        )

        """ Delta operator for reaction """
        # energy source: V [m3] dt [day] c_r phi^T Q [kJ/m3.days] (kJ/m3)
        values[self.KIN_OP + self.nc] = self.property.energy_source

        # Phase enthalpy
        for j in range(self.nph):
            values[self.ENTH_OP + j] = self.property.enthalpy[j]

        """ Additional energy operators """
        # Temperature operator
        values[self.TEMP_OP] = temperature

        return 0


class GeomechanicsReservoirOperators(ReservoirOperators):
    def __init__(self, property_container: PropertyContainer, thermal: bool):
        super().__init__(property_container, thermal)  # Initialize base-class

        self.ROCK_DENS_OP = self.PRES_OP + 1  # used only in mechanical engine
        self.n_ops = self.ROCK_DENS_OP + 1

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values): value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Reservoir operators
        super().evaluate(state, values)

        # Rock density operator
        self.n_ops = self.ROCK_DENS_OP + 1
        # TODO: function of matrix pressure = I1 / 3 = (s_xx + s_yy + s_zz) / 3
        values.to_numpy()[self.ROCK_DENS_OP] = self.property.rock_density_ev.evaluate()

        return 0

    def print_operators(self, state, values):
        """Method for printing operators, grouped"""
        super().print_operators(state, values)
        print("ROCK DENSITY", values[self.ROCK_DENS_OP])
        return


class WellOperators(OperatorsSuper):
    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values): value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        pressure = vec_state_as_np[0]

        vec_values_as_np[:] = 0

        self.property.evaluate(vec_state_as_np)

        self.compr = self.property.rock_compr_ev.evaluate(pressure)

        density_tot = np.sum(
            self.property.sat[: self.np_fl] * self.property.dens_m[: self.np_fl]
        )
        zc = np.append(
            vec_state_as_np[1 : self.nc], 1 - np.sum(vec_state_as_np[1 : self.nc])
        )
        self.phi_f = 1.0

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term """
        # fluid mass accumulation: c_r phi^T z_c* [-] rho_m^T [kmol/m3]
        vec_values_as_np[self.ACC_OP : self.ACC_OP + self.nc_fl] = (
            self.compr * density_tot * zc[: self.nc_fl]
        )

        """ and alpha for mineral components """
        # solid mass accumulation: c_r phi^T z_s* [-] rho_ms [kmol/m3]
        vec_values_as_np[
            self.ACC_OP + self.nc_fl : self.ACC_OP + self.nc_fl + self.ns
        ] = (
            self.compr
            * self.property.dens_m[self.np_fl : self.np_fl + self.ns]
            * zc[self.nc_fl : self.nc_fl + self.ns]
        )

        """ Beta operator represents flux term: """
        for j in self.property.ph:
            # fluid convective mass flux: x_cj [-] rho_mj [kmol/m3] k_rj [-] / mu_j [cP ∝ bar.day] (kmol/m3.bar.day)
            vec_values_as_np[
                self.FLUX_OP + j * self.ne : self.FLUX_OP + j * self.ne + self.nc_fl
            ] = (
                self.property.x[j][: self.nc_fl]
                * self.property.dens_m[j]
                * self.property.kr[j]
                / self.property.mu[j]
            )

        """ Gamma operator for diffusion (same for thermal and isothermal) """

        """ Chi operator for diffusion """

        """ Delta operator for reaction """
        # fluid/solid mass source: dt [day] n_c [kmol/m3.day] (kmol/m3)
        vec_values_as_np[self.KIN_OP : self.KIN_OP + self.nc] = (
            self.property.mass_source
        )

        """ Gravity and Porosity operators """
        # E3-> gravity
        vec_values_as_np[self.GRAV_OP + self.property.ph] = self.property.dens[
            self.property.ph
        ]

        # E5_> permeability multiplier due to permporo relationship
        vec_values_as_np[self.MULT_OP] = 1.0

        # Pressure operator
        vec_values_as_np[self.PRES_OP] = vec_state_as_np[0]

        if self.thermal:
            self.evaluate_thermal(vec_state_as_np, vec_values_as_np)

        # self.print_operators(state, values)

        return 0

    def evaluate_thermal(self, state, values):
        values[self.TEMP_OP] = state[-1]
        return


class SinglePhaseGeomechanicsOperators(OperatorsBase):
    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1, temp]: value_vector in open-darts, pylvarray.Array in GEOS
        :param values: values of the operators (used for storing the operator values): value_vector in open-darts, pylvarray.Array in GEOS
        :return: updated value for operators, stored in values
        """

        vec_state_as_np = state.to_numpy()
        vec_values_as_np = values.to_numpy()
        self.property.evaluate(vec_state_as_np)
        vec_values_as_np[0] = self.property.dens[0]
        vec_values_as_np[1] = self.property.dens[0] / self.property.mu[0]

        return 0
