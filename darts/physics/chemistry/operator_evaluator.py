import numpy as np

from darts.physics.super.operator_evaluator import OperatorsSuper


class ReservoirOperators(OperatorsSuper):
    """
    Reservoir operators working with the following state:
    state:
    p - pressure in [bar]
    z_{1}, ..., z_{n_m} - mineral molar fractions within rock + fluid mixture
    z_{n_m+1}, ..., z_{n_c-1} - fluid molar fractions within only fluid
    values are the same as in OperatorsSuper
    """

    def __init__(self, properties):
        """
        Constructor for ReservoirOperators class.
        :param properties: Property container object
        :type properties: user-defined or built-in PropertyContainer class
        """
        # set some properties to -1 to use OperatorsSuper constructor
        # TODO: refactor in future
        properties.nc_fl = -1
        properties.np_fl = -1
        properties.ns = -1
        super().__init__(properties, thermal=properties.thermal)

        # Store your input parameters in self here, and initialize other parameters here in self
        self.property = properties
        self.counter = 0

    def get_overall_composition(self, state):
        """
        Class method which returns corrected full (inlcuding last) molar composition of the system.
        It ensures last fluid compositions are within (min_z, 1-min_z) range.
        :param state: state variables [p, z_{1}, ..., z_{n_m}, z_{n_m+1}, ..., z_{n_c-1}]
        :type state: np.ndarray
        :return: overall molar composition [z_{1}, ..., z_{n_m}, z_{n_m+1}, ..., z_{n_c}]
        :rtype: np.ndarray
        """
        if self.thermal:
            z = state[1:-1]
        else:
            z = state[1:]
        z_last = min(
            max(1 - np.sum(z[self.property.fc_mask[:-1]]), self.min_z),
            1 - self.min_z,
        )
        z = np.concatenate([z, [z_last]])
        return z

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for
        the element-based formulation of reactive flow physics
        :param state: state variables [p, z_{1}, ..., z_{n_m}, z_{n_m+1}, ..., z_{n_c-1}]
        :type state: value_vector
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector
        :rtype: int
        """
        # state and values numpy vectors:
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        values_np[:] = 0

        # pore pressure
        _p = state_np[0]
        # get overall molar composition
        z = self.get_overall_composition(state_np)
        # call property:
        self.property.evaluate(state_np)

        # Densities
        rho_t = (
            np.sum(
                self.property.dens_m * self.property.sat_overall[: self.property.nph]
            )
            + self.property.dens_m_solid * self.property.sat_overall[self.property.nph]
        )
        rho_f = np.sum(self.property.dens_m * self.property.sat)

        nc = self.property.nc
        ns = self.property.n_solid
        nph = self.property.nph
        ne = nc

        """ CONSTRUCT OPERATORS HERE """

        """ Alpha operator represents accumulation term: """
        values_np[self.ACC_OP : self.ACC_OP + ns] = z[:ns] * rho_t
        values_np[self.ACC_OP + ns : self.ACC_OP + nc] = (
            (1 - self.property.sat_overall[self.property.nph]) * z[ns:] * rho_f
        )

        """ Beta operator represents flux term: """
        for j in range(nph):
            values_np[self.FLUX_OP + j * self.ne : self.FLUX_OP + j * self.ne + nc] = (
                self.property.x[j] * self.property.dens_m[j]
            )

        """ molar density operator """
        values_np[self.DENS_OP + self.property.ph] = self.property.dens_m[
            self.property.ph
        ]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        for j in range(nph):
            values_np[self.UPSAT_OP + j] = (
                self.property.rock_compr.mean() * self.property.sat[j]
            )

        """ Chi operator for diffusion """
        for j in self.property.ph:
            values_np[self.GRAD_OP + j * self.ne : self.GRAD_OP + (j + 1) * self.ne] = (
                self.property.diffusivity[j] * self.property.x[j]
            )

        """ Delta operator for reaction """
        for i in range(ne):
            values_np[self.KIN_OP + i] = (
                self.property.stoich_matrix[:, i] * self.property.kin_rates
            ).sum()

        """ Gravity and Capillarity operators """
        # E3-> gravity
        values_np[self.GRAV_OP + self.property.ph] = self.property.dens[
            self.property.ph
        ]

        # E4-> capillarity
        values_np[self.PC_OP + self.property.ph] = self.property.pc[self.property.ph]

        """ Permeability multiplier k/kmax """
        # E5_> permeability multiplier due to permporo relationship
        values_np[self.MULT_OP] = self.property.permporo_mult_ev.evaluate(
            1 - self.property.sat_overall[self.property.nph]
        )

        """ Lambda operator for velocity calculations """
        # phase mobility: k_rj [-] / mu_j [cP ∝ bar.day] (1/(bar.day))
        values_np[self.LAMBDA_OP + self.property.ph] = (
            self.property.kr[self.property.ph] / self.property.mu[self.property.ph]
        )

        """ Saturation operator """
        # phase saturation: s_j [-]
        values_np[self.SAT_OP + self.property.ph] = self.property.sat[self.property.ph]

        """ Pressure operator """
        # Pressure operator (for generic state specification where no pressure in the state, for instance V,T)
        values_np[self.PRES_OP] = state_np[0]

        if self.thermal:
            self.evaluate_thermal(state_np, values_np)

        return 0

    def evaluate_thermal(self, state, values):
        """
        Class methods which evaluates the state operators for the element-based
        formulation of reactive flow physics for thermal case.
        :param state: state variables [p, z_{1}, ..., z_{n_m}, z_{n_m+1}, ..., z_{n_c-1}, T]
        :type state: value_vector
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector
        :rtype: int
        """
        pass  # TODO: implement thermal evaluation


class ConversionOperators(ReservoirOperators):
    """
    Operator required for initialization, to convert given volume fraction to molar one
    Therefore, it works with state DIFFERENT from ReservoirOperators:
    state:
    p - pressure in [bar]
    \phi_{1}, ..., \phi_{n_m} - mineral volume fractions within rock + fluid mixture
    z_{n_m+1}, ..., z_{n_c-1} - fluid molar fractions within only fluid
    values are mineral molar fractions within rock + fluid mixture
    """

    def __init__(self, properties):
        """
        Constructor for ConversionOperators class.
        :param properties: Property container object
        :type properties: user-defined or built-in PropertyContainer class
        """
        super().__init__(properties)  # Initialize base-class
        self.fluid_mole = self.property.flash_ev.total_moles / 1000  # mol to kmol
        self.counter = 0
        self.props_name = ['z_' + prop for prop in properties.minerals]

    def evaluate(self, state, values):
        """
        Class methods which performs volumetric to molar conversion for
        the element-based formulation of reactive flow physics
        :param state: state variables [p, \phi_{1}, ..., \phi_{n_m}, z_{n_m+1}, ..., z_{n_c-1}]
        :type state: value_vector
        :param values: values of the operators (used for storing the operator values)
        :type values: value_vector
        :return: updated value for operators, stored in values
        :rtype: int
        """
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        pressure = state_np[0]
        s_minerals = state_np[self.property.s_mask_state]
        ss = s_minerals.sum()  # volume fraction in initialization

        # initial flash, non-standard argument
        _, _, _, _, _, fluid_volume, _, _ = self.property.flash_ev.evaluate(state_np)

        # evaluate molar fraction
        solid_volume = fluid_volume * ss / (1 - ss)  # m3
        mineral_volume = s_minerals * (fluid_volume + solid_volume)
        mineral_mole = np.array(
            [
                mineral_volume[i]
                * self.property.rock_density_ev[k].evaluate(pressure)
                / self.property.Mw[k]
                for i, k in enumerate(self.property.rock_density_ev.keys())
            ]
        )
        nu_m = mineral_mole / (mineral_mole.sum() + self.fluid_mole)
        values_np[: self.property.n_solid] = nu_m

        return 0
