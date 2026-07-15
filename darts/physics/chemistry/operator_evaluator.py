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

    def __init__(
        self,
        property_container,
        thermal: bool,
        extrapolation_flag: bool = False,
        dz: float = None,
    ):
        """
        Constructor of ReservoirOperators class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        # set some properties to -1 to use OperatorsSuper constructor
        # TODO: refactor in future
        property_container.nc_fl = -1
        property_container.np_fl = -1
        property_container.ns = -1
        super().__init__(
            property_container=property_container,
            thermal=thermal,
            extrapolation_flag=extrapolation_flag,
            dz=dz,
        )

        # Store your input parameters in self here, and initialize other parameters here in self
        self.counter = 0

    def get_overall_composition(self, state):
        """
        Class method which returns corrected full (including last) molar composition of the system.
        It ensures last fluid compositions are within (obl_min_z, obl_max_z) range.
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
            max(1 - np.sum(z[self.property.fc_mask[:-1]]), self.property.eps_z),
            1.0 - len(z) * self.property.eps_z,
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
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

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

    def __init__(
        self,
        property_container,
        thermal: bool,
        extrapolation_flag: bool = False,
        dz: float = None,
    ):
        """
        Constructor of ConversionOperators class

        :param property_container: Property container of type PropertyBase
        :param thermal: Switch to indicate if energy conservation equation is there
        :param extrapolation_flag: Switch to turn on extrapolation logic (z[last component] < 0 in case nc >= 3)
        :param dz: Composition interval along OBL composition axes to obtain consistent points for extrapolation
                    (must be equal along all composition axes in current setup)
        """
        super().__init__(
            property_container, thermal, extrapolation_flag, dz
        )  # Initialize base-class
        self.fluid_mole = self.property.flash_ev.total_moles / 1000  # mol to kmol
        self.counter = 0
        self.props_name = ['z_' + prop for prop in property_container.minerals]

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
        # Check if extrapolation needs to be applied
        if super().apply_extrapolation(state, values):
            return 0

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


class SmoothFieldsReservoirOperators(ReservoirOperators):
    """
    Nested-kinetics variant of :class:`ReservoirOperators`: the KIN operator slots
    carry SMOOTH FIELDS instead of the sharp composed kinetic rates. The engine-facing
    rates are reconstructed analytically (with exact chain-rule derivatives) by
    ``kinetic_composition_cpu_interpolator`` after interpolation, so the sharp
    ``(1 - SR^p)^q`` sign flip at SR = 1 never enters the interpolated set.

    KIN block layout produced here (n_min = number of kinetic minerals, must satisfy
    2 * n_min + 1 <= ne):

    - slot KIN_OP + m             : SRb_m = min(SR_m, SR_threshold)
    - slot KIN_OP + n_min + m     : A_m = sum_j k_arr_j(T) * act_j^{n_j}  (affinity-free
                                    activity-weighted Arrhenius sum over mechanisms)
    - slot KIN_OP + 2 * n_min     : rho_t = 1 / vol_sum  (total molar density, kmol/m3)
    - remaining KIN slots         : 0

    Exactness: kin_rate_m = -s_init * sat_m * (rho_s_m * 1000) * sum_j rate_j * 86.4 and
    sat_m * rho_s_m = z_m / vol_sum = z_m * rho_t (z_m is the mineral's solid state entry),
    so with mechanisms sharing (p, q) per mineral:
    kin_rate_m == c_m * z_m * rho_t * A_m * (1 - SRb_m^p)^q with c_m = -s_init * 86400.

    NOTE: requires all mechanisms of a mineral to share (p, q) (true for the
    PalandriKharaka carbonates: p = q = 1). Checked at construction.
    """

    #: activity source per mechanism name (mirrors KineticRate.evaluate)
    _ACT_KEYS = {'acidic': 'Act(H+)', 'neutral': None, 'carbonate': 'Act(CO2)'}

    def __init__(self, property_container, thermal, extrapolation_flag=False, dz=None):
        super().__init__(property_container, thermal, extrapolation_flag, dz)
        prop = self.property
        self.mineral_keys = list(prop.rock_compr_ev.keys())  # defines mineral order
        n_min = len(self.mineral_keys)
        if 2 * n_min + 1 > self.ne:
            raise ValueError(
                f"SmoothFieldsReservoirOperators: {n_min} minerals need "
                f"{2 * n_min + 1} field slots > KIN block size ne={self.ne}"
            )
        # Per-mineral shared affinity exponents + composition constants; validate that
        # the shared-affinity factorization holds (all mechanisms share p, q).
        self.p_aff, self.q_aff, self.c_coeffs = [], [], []
        for k in self.mineral_keys:
            kr = prop.kinetic_rate_ev[k]
            pq = {(mech.p, mech.q) for mech in kr.mechanisms}
            if len(pq) > 1:
                raise ValueError(
                    f"SmoothFieldsReservoirOperators: mineral '{k}' mechanisms have "
                    f"mixed affinity exponents {pq}; shared-affinity nesting requires "
                    f"identical (p, q) per mineral"
                )
            p, q = next(iter(pq)) if pq else (1.0, 1.0)
            self.p_aff.append(float(p))
            self.q_aff.append(float(q))
            # c_m = -s_init * (rho_s*1000 folded via rho_t identity) * 86400/1000 * 1000
            self.c_coeffs.append(-kr.surface_area_ev.s_init * 1000.0 * 86400.0 / 1000.0)

    def kin_field_slots(self):
        """(SRb slots, A slots, rho_t slot) — layout consumed by the C++ composition."""
        n_min = len(self.mineral_keys)
        return (
            list(range(self.KIN_OP, self.KIN_OP + n_min)),
            list(range(self.KIN_OP + n_min, self.KIN_OP + 2 * n_min)),
            self.KIN_OP + 2 * n_min,
        )

    def evaluate(self, state, values):
        ret = super().evaluate(state, values)
        if ret != 0:
            return ret
        values_np = values.to_numpy()
        prop = self.property
        kin_state = prop.kin_state
        n_min = len(self.mineral_keys)

        # rho_t = 1/vol_sum, recomputed exactly as property_container.evaluate's local
        # (gas volume term included iff the gas phase is present: nu_v > 0 <=> nu[g] > 0)
        idx_g, idx_a = prop.phase_idx['gas'], prop.phase_idx['aq']
        vol_sum = (
            prop.nu[idx_a] / prop.dens_m[idx_a]
            + (prop.nu_solid / prop.dens_m_solid).sum()
        )
        if prop.nu[idx_g] > 0:
            vol_sum += prop.nu[idx_g] / prop.dens_m[idx_g]

        values_np[self.KIN_OP : self.KIN_OP + self.ne] = 0.0
        T = prop.temperature
        for m, k in enumerate(self.mineral_keys):
            kr = prop.kinetic_rate_ev[k]
            SR = kin_state['SR_' + kr.mineral]
            SRb = SR
            A = 0.0
            for mech in kr.mechanisms:
                SRb = min(SR, mech.SR_threshold)
                act_key = self._ACT_KEYS[mech.name]
                act = 1.0 if act_key is None else kin_state[act_key]
                k_arr = mech.k * np.exp(
                    (-mech.Ea / mech.R) * (1.0 / T - 1.0 / mech.temperature_ref)
                )
                A += k_arr * act**mech.n
            values_np[self.KIN_OP + m] = SRb
            values_np[self.KIN_OP + n_min + m] = A
        values_np[self.KIN_OP + 2 * n_min] = 1.0 / vol_sum
        return 0
