"""
Differences between this script and the DFM velocity evaluator in DWell:
- https://gitlab.com/open-darts/open-darts/-/commit/b0aa26cb9beb90a10bb1b4e3db5291399607f46d
Revert the density averaging method here (now it is similar to DWell):
- https://gitlab.com/open-darts/open-darts/-/commit/0c584d57e8cd20c270057763f3b6bc916cbc695e

Notes:
    - The kinetic energy is not added to the energy conservation equation of the coupled well-reservoir model yet,
    while it is in DWell.

    - When having a DFM pipe:
        - Use 'G' as the name of the gaseous phase
        - Use 'L' as the name of the liquid phase (for a single liquid phase)
        - Use 'L_a' and 'L_b' as the names of the liquid phases (for two liquid phases)
        - Immobile phases after PropertyContainer.np_fl, or phases explicitly listed in Pipe.immobile_phase_names,
          are returned with zero velocity. The drift-flux momentum closure itself is still gas/liquid.
          Mobile-liquid holdup is tracked explicitly as sL (L or L_a + L_b). The C00-adjusted liquid saturations
          retain the original (1 - C00 * sG) form, while explicitly mobile-liquid terms use sL. If an
          immobile phase has nonzero holdup in a DFM pipe segment, the C00-adjusted non-gas factors can still
          treat it as liquid-like holdup; solid blockage or separate immobile-holdup physics is not modeled.
        - Source-boundary momentum calculations assume no immobile phase is present in the source segment. If
          immobile holdup is present there, single-mobile-phase source velocities still use the full pipe area
          rather than an area reduced by immobile-phase saturation.
"""

from darts.engines import index_vector, value_vector
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.drift_flux import (
    BHAGWAT_GHAJAR_DRIFT_FLUX_MODELS,
    GRAVITY_ACCELERATION,
    SUPPORTED_DRIFT_FLUX_MODELS,
    AdjustmentFuncParams,  # noqa: F401  (re-exported for backwards compatibility)
    DriftFluxClosure,
    FaceProps,
    TangUnifiedDFParams,  # noqa: F401  (re-exported for backwards compatibility)
    colebrook,
    make_drift_flux_closure,
    wang_darcy_friction_factor,
)
from darts.pipes.ramp_up_rate import RampUpRate, SegmentState
from darts.pipes.set_initial_conditions import (
    LinearAmbientTemperature,
    SingleAmbientTemperature,
)
from darts.pipes.units import *


class Pipe:
    g = GRAVITY_ACCELERATION  # Gravitational acceleration

    bhagwat_ghajar_drift_flux_models = BHAGWAT_GHAJAR_DRIFT_FLUX_MODELS
    supported_drift_flux_models = SUPPORTED_DRIFT_FLUX_MODELS

    def __init__(
        self,
        pipe_name: str,
        pipe_geometry: PipeGeometry,
        physics,
        reservoir,
        initial_conditions: SingleAmbientTemperature | LinearAmbientTemperature,
        source_sinks: dict = None,
        immobile_phase_names: list[str] | None = None,
        drift_flux_model: str | DriftFluxClosure = "shi_t2well",
        tang_parameter_set: str | None = None,
        friction_model: str | None = None,
        Cmax: float | None = None,
        Fv: float | None = None,
        prop_eval_method: str = "direct",
        diff_method: str = "OBL",
        eps_p: float = 1e-4,
        eps_temp: float = 0.1,
        eps_z: float = 0.00001,
        enable_profile_parameter: bool = True,
        enable_drift_velocity: bool = True,
        exclude_top_control_block_from_velocity: bool = False,
        verbose: bool = False,
    ):
        """
        :param pipe_name: Name of the pipe
        :type pipe_name: str
        :param pipe_geometry: Pipe geometry object
        :type pipe_geometry: PipeGeometry
        :param physics: Physics object for the pipe
        :param reservoir: Reservoir object
        :param initial_conditions: Object containing the initial conditions of the wellbore/pipe
        :type initial_conditions: SingleAmbientTemperature or LinearAmbientTemperature
        :param source_sinks: Dict containing sources or sinks for the momentum equation
        :type source_sinks: dict
        :param immobile_phase_names: Phase names that are thermodynamic phases but are wanted to have zero DFM
                                     velocity in the pipe, e.g. ["Ice"]. These phases are the immobile phases that
                                     are not considered as np_sol in the property container.
        :type immobile_phase_names: list[str] or None
        :param drift_flux_model: Drift-flux closure to use:
                                 - "shi_t2well" retains the historical Holmes/Shi/T2Well style closure
                                 - "tang_2019" uses the unified all-inclination Tang et al. (2019) closure
                                 - "bai_2023" uses the CO2-specific Bai et al. (2023) pipe-flow model, which adopts
                                   Bhagwat and Ghajar (2014) with Bai-specific Reynolds/friction and drift terms
                                 - "bhagwat_ghajar_2014" uses the original Bhagwat and Ghajar (2014) drift-flux
                                   correlation with its Colebrook-White friction term
                                 An already-built darts.pipes.drift_flux.DriftFluxClosure instance is also
                                 accepted; the string form above is the documented default path.
        :type drift_flux_model: str or DriftFluxClosure
        :param tang_parameter_set: Parameterization of the Tang et al. (2019) unified model. Allowed values are
                                   "olgas" (default) and "tuffp". Only meaningful for drift_flux_model="tang_2019";
                                   passing it with another closure raises a ValueError instead of being ignored.
        :type tang_parameter_set: str or None
        :param friction_model: Friction-factor closure to use:
                               - None selects the default of the drift-flux closure, i.e. "wang_2014" for
                                 drift_flux_model="bai_2023" and "colebrook_white" otherwise
                               - "colebrook_white" uses Colebrook-White correlation to calculate the friction factor
                               - "wang_2014" uses the Wang et al. (2014) supercritical-CO2 friction factor adopted by Bai et al.
        :type friction_model: str or None
        :param Cmax: A user-specified maximum profile parameter that can be tuned to match the observations and
                     could have a value between 1.0 and 1.5. It is set to:
                     --> 1.2 in ECLIPSE according to Shi et al. paper (Drift-Flux Modeling of Two-Phase Flow in Wellbores)
                     --> 1.0 in a wellbore simulator in Tonken et al. paper (A transient geothermal wellbore simulator)
                     Only meaningful for drift_flux_model="shi_t2well" (default 1.2); passing it with another
                     closure raises a ValueError instead of being ignored.
        :type Cmax: float or None
        :param Fv: A multiplier on the flooding velocity fraction, set to be 1 by default, and its value can be tuned
                   to fit the observations. Only meaningful for drift_flux_model="shi_t2well"; passing it with
                   another closure raises a ValueError instead of being ignored.
        :type Fv: float or None
        :param prop_eval_method: Method for evaluation of wellbore phase properties to calculate phase velocities
                                 "OBL" for OBL approach and "direct" for direct usage of property evaluators (default)
                                 Note that using the OBL approach may lead to non-zero phase saturation where saturation
                                 is expected to be zero or non-one phase saturation where saturation is expected to be 1.
                                 This leads to instability of evaluation of wellbore phase velocities. To avoid this
                                 either use a larger OBL resolution or use the direct method as a safe approach.
                                 If the OBL approach is used, the following point need to be followed when creating
                                 the model:
                                 - In output_props of the property container, specify phase saturation, density,
                                   viscosity, and mass composition of each phase with appropriate names as keys.
        :type prop_eval_method: str
        :param diff_method: Method for differentiation of wellbore phase velocities with respect to the primary
                            variables. "OBL" for OBL diff (default) and "numerical" for numerical diff
        :type diff_method: str
        :param eps_p: If diff_method is numerical, this variable will be used. It is a very small value used for
                      numerically differentiating pipe phase velocities with respect to pressure
        :type eps_p: float
        :param eps_temp: If diff_method is numerical, this variable will be used. It is a very small value used for
                         numerically differentiating pipe phase velocities with respect to temperature
        :type eps_temp: float
        :param eps_z: If diff_method is numerical, this variable will be used. It is a very small value used for
                      numerically differentiating pipe phase velocities with respect to composition
        :type eps_z: float
        :param verbose: Whether to display extra info about PipeModel
        :param enable_profile_parameter: Whether to calculate and use profile parameter for multiphase flow. If False,
                                         profile parameter is set to one.
        :type enable_profile_parameter: bool
        :param enable_drift_velocity: Whether to calculate and use drift velocity for multiphase flow. If False,
                                      drift velocity is set to zero.
        :type enable_drift_velocity: bool
        :param exclude_top_control_block_from_velocity: If True, interface 0 uses segment 1 for phase properties
                                                        in the DFM velocity closure. The pressure gradient still
                                                        uses segment 0 and 1, but the ghost wellhead/control
                                                        block is not used for saturation, density, viscosity,
                                                        composition, profile parameter, or drift velocity. Enable
                                                        this for DFM production wells controlled by the C++
                                                        wellhead rate-control block in PH formulation. In that
                                                        setup, the C++ control equation keeps the wellhead/control
                                                        block pressure as the rate-control unknown, while the
                                                        remaining primary variables are constrained from the first
                                                        physical segment. For PH formulation, this means the
                                                        control block can have a different pressure but the same
                                                        enthalpy as the first physical segment. Flashing that
                                                        artificial pressure-enthalpy pair can produce nonphysical
                                                        temperature/saturation, so those phase properties should
                                                        not be used in the DFM velocity closure.
        :type exclude_top_control_block_from_velocity: bool
        :type verbose: boolean
        """
        assert pipe_name == pipe_geometry.pipe_name, (
            "Pipe names in pipe_name and pipe_geometry are not identical!"
        )
        self.name = pipe_name
        self.geometry = pipe_geometry

        # Check if mobile phase names and number of mobile phases are as expected. Then, store phase indices.
        phase_names = physics.phases
        pc = physics.property_containers[0]
        immobile_phase_names = (
            [] if immobile_phase_names is None else list(immobile_phase_names)
        )
        unknown_immobile_phases = set(immobile_phase_names) - set(phase_names)
        assert not unknown_immobile_phases, (
            f"Immobile phase(s) {sorted(unknown_immobile_phases)} are not found in the list of phases!"
        )
        thermodynamic_phase_names = phase_names[: pc.np_fl]
        mobile_phase_names = [
            phase_name
            for phase_name in thermodynamic_phase_names
            if phase_name not in immobile_phase_names
        ]
        self.n_mobile_phases = len(mobile_phase_names)
        assert "G" in mobile_phase_names, (
            "Gaseous phase with the name 'G' is not found in the list of phases!"
        )
        self.g_idx = phase_names.index("G")
        if self.n_mobile_phases == 2:
            assert "L" in mobile_phase_names, (
                "Liquid phase with the name 'L' is not found in the list of phases!"
            )
            self.l_idx = phase_names.index("L")
        elif self.n_mobile_phases == 3:
            assert "L_a" in mobile_phase_names, (
                "Liquid phase with the name 'L_a' is not found in the list of phases!"
            )
            assert "L_b" in mobile_phase_names, (
                "Liquid phase with the name 'L_b' is not found in the list of phases!"
            )
            self.la_idx = phase_names.index("L_a")
            self.lb_idx = phase_names.index("L_b")
        else:
            raise Exception(
                f"{self.n_mobile_phases} DFM mobile phase(s) is not supported! "
                f"Mobile phase list: {mobile_phase_names}; "
                f"immobile phase list: {immobile_phase_names}; "
                f"thermodynamic phase list: {thermodynamic_phase_names}; "
                f"full phase list: {phase_names}"
            )

        self.isothermal = not physics.thermal
        if self.isothermal:
            assert pc.temperature is not None, (
                "If model is isothermal, system_temperature must be specified!"
            )

        self.system_temperature = pc.temperature
        self.p_idx = physics.vars.index("pressure")

        self.physics = physics

        self.reservoir = reservoir

        self.initial_conditions = initial_conditions

        source_sinks = {} if source_sinks is None else source_sinks
        for source_sink in source_sinks.values():
            # RampUpRate is supported fow now
            assert isinstance(source_sink, RampUpRate), (
                "source_sink must be of the type RampUpRate!"
            )
            assert pipe_name == source_sink.pipe_name, (
                "Pipe names in pipe_name and source_sink are not identical!"
            )
        self.source_sinks = source_sinks

        # The drift-flux closure owns its own parameters, its theta convention, its
        # sign convention and its friction model (see darts.pipes.drift_flux).
        # Closure-irrelevant constructor keywords are rejected here rather than
        # silently ignored, which is why they default to None.
        self.drift_flux_closure = make_drift_flux_closure(
            drift_flux_model,
            Cmax=Cmax,
            Fv=Fv,
            tang_parameter_set=tang_parameter_set,
        ).bind(pipe_geometry, friction_model=friction_model)
        self.drift_flux_model = self.drift_flux_closure.name
        self.friction_model = self.drift_flux_closure.friction.name
        self.tang_parameter_set = getattr(
            self.drift_flux_closure, "parameter_set", None
        )

        self.g_cos_theta = self.g * np.cos(pipe_geometry.inclination_angle_radian)
        if isinstance(self.g_cos_theta, float):
            self.g_cos_theta = self.g_cos_theta * np.ones(pipe_geometry.num_interfaces)

        assert isinstance(prop_eval_method, str), (
            "prop_eval_method for pipe velocity calculation must be a string!"
        )
        if prop_eval_method in ("direct", "OBL"):
            self.prop_eval_method = prop_eval_method
        else:
            raise ValueError(
                "prop_eval_method for pipe velocity differentiation must be either 'OBL' or 'direct'!"
            )
        # The following vars are used for the property interpolator used if prop_eval_method is OBL
        if self.prop_eval_method == "OBL":
            n_output_props = len(pc.output_props)
            n_vars = self.physics.n_vars
            n_segments = pipe_geometry.num_segments
            if n_output_props < self.physics.n_ops:
                self.n_prop_ops = self.physics.n_ops
            else:
                self.n_prop_ops = n_output_props + n_vars
            self.block_idx = index_vector(np.arange(n_segments).astype(np.int32))
            # This variable (derivatives of props) is not used in calculations. Derivatives of operators are used.
            self.dvalues = value_vector(
                np.zeros((n_segments * self.n_prop_ops) * n_vars)
            )

        # Set the properties required for the specified differentiation method
        assert isinstance(diff_method, str), (
            "diff_method for pipe velocity differentiation must be a string!"
        )
        if diff_method in ("numerical", "OBL"):
            self.diff_method = diff_method
            if self.diff_method == "numerical":
                # Epsilon values for numerical differentiation with respect to pressure, temperature, and overall composition
                self.eps_p = eps_p
                self.eps_temp = eps_temp
                self.eps_z = eps_z
            elif self.diff_method == "OBL":
                self.rhoM_face_der = np.array([])
                self.rhoM_adjusted_face_der = np.array([])
                self.rhoM_vM_der = np.array([])
                self.vG_der, self.vL_der = np.array([]), np.array([])
        else:
            raise ValueError(
                "diff_method for pipe velocity differentiation must be either 'OBL' or 'numerical'!"
            )

        self._build_phase_vel_dense_der_indexers()

        # is_first_first_iter is true only for the first iteration of the first time step.
        self.is_first_first_iter = True
        self._accepted_pipe_state = None

        self.enable_profile_parameter = enable_profile_parameter
        self.enable_drift_velocity = enable_drift_velocity
        self.exclude_top_control_block_from_velocity = (
            exclude_top_control_block_from_velocity
        )

        if verbose:
            print(f'** Model of the pipe "{self.geometry.pipe_name}" is created!')

    def _build_phase_vel_dense_der_indexers(self):
        """
        Builds the indexers needed to extract the dense part of the derivative matrix of phase velocities
        """
        n_conns = self.geometry.num_interfaces
        n_vars = self.physics.n_vars
        vel_der_size = 2 * n_vars

        # Rows and local-block columns used to extract the dense part of the derivative matrix of phase velocities (n_conns, 2 * n_vars)
        self.conn_row_idx = np.arange(n_conns)[:, None]  # (n_conns, 1)
        self.conn_local_col_idx = (
            self.conn_row_idx * n_vars + np.arange(vel_der_size)[None, :]
        )

    @staticmethod
    def _copy_pipe_state_value(value):
        """Copy pipe-state entries without sharing mutable NumPy arrays."""
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, list):
            return [Pipe._copy_pipe_state_value(item) for item in value]
        return value

    def _current_pipe_state(self):
        """Return a snapshot of the current pipe state."""
        required_attrs = (
            "iter_phases_props",
            "rhoM_face",
            "rhoM_vM",
            "vM",
            "vG",
            "vL",
        )
        missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing_attrs:
            raise RuntimeError(
                "Cannot accept pipe state before pipe properties have "
                f"been evaluated. Missing attributes: {', '.join(missing_attrs)}"
            )

        return {
            attr: self._copy_pipe_state_value(getattr(self, attr))
            for attr in required_attrs
        }

    def reset_pipe_state(self):
        """Clear accepted pipe state."""
        self._accepted_pipe_state = None
        self.is_first_first_iter = True

    def accept_pipe_state(self):
        """Store the current pipe state as the last converged (accepted) pipe state."""
        accepted_state = self._current_pipe_state()

        self._accepted_pipe_state = accepted_state
        self.is_first_first_iter = False

    def _load_accepted_pipe_state(self):
        """Load the accepted pipe state."""
        if self._accepted_pipe_state is None:
            raise RuntimeError(
                "Cannot load pipe state before an accepted pipe state has been stored."
            )

        for attr, value in self._accepted_pipe_state.items():
            setattr(self, attr, self._copy_pipe_state_value(value))
        return True

    @property
    def lateral_heat_rate_eval(self):
        return None

    @lateral_heat_rate_eval.setter
    def lateral_heat_rate_eval(self, value):
        raise RuntimeError(
            "lateral_heat_rate_eval is no longer consumed; register "
            "SemiAnalyticalWellLateralHeatTransferHook in model.rhs_flux_hooks instead"
        )

    def eval_phase_vels(
        self, Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter, flag
    ):
        """
        Evaluate pipe phase velocities using the drift-flux model (DFM)

        :param Xn_dfm_well: Vector containing the state of pipe segments (ordered block by block) of the previous time step
        :type Xn_dfm_well: np.ndarray
        :param X_dfm_well: Vector containing the state of pipe segments (ordered block by block) of the current time step
        :type X_dfm_well: np.ndarray
        :param dt: Time step size [day]
        :type dt: float
        :param simulation_time: Simulation time [day]
        :type simulation_time: float
        :param iter_counter: Iteration counter of the current time step
        :type iter_counter: int
        :param flag: Flag indicating if we want to update the solution of the previous time step based on the new solution or not
            If 0  -> Used when we don't want to update the solution of the previous time step, e.g., during numerical differentiation
            If 1  -> Used when we want to update the solution of the previous time step
        :type flag: int
        """
        dt_day = dt  # the rate schedule works in days, everything else in SI
        dt = dt * 24 * 60 * 60  # convert day to second

        geom = self.geometry
        num_segments = geom.num_segments
        num_interfaces = geom.num_interfaces
        pc = self.physics.property_containers[0]
        Mw_fl = np.asarray(pc.Mw[: pc.nc_fl])
        n_vars = self.physics.n_vars

        Xn_dfm_well_2d = Xn_dfm_well.reshape(num_segments, n_vars)
        X_dfm_well_2d = X_dfm_well.reshape(num_segments, n_vars)

        if self.diff_method == "OBL":
            # Get operator indices
            GRAV_OP = self.physics.reservoir_operators[0].GRAV_OP
            SAT_OP = self.physics.reservoir_operators[0].SAT_OP
            PRES_OP = self.physics.reservoir_operators[0].PRES_OP

        # Get phase indices
        g_idx = self.g_idx
        if self.n_mobile_phases == 2:
            l_idx = self.l_idx
        elif self.n_mobile_phases == 3:
            la_idx = self.la_idx
            lb_idx = self.lb_idx

        if iter_counter == 0 and not self.is_first_first_iter and flag == 1:
            self._load_accepted_pipe_state()

        """ Calculate phase props of previous time step at centroids """
        if iter_counter == 0 and self.is_first_first_iter and flag == 1:
            if self.prop_eval_method == "OBL":
                state0 = value_vector(Xn_dfm_well)
                values0 = value_vector(np.zeros(num_segments * self.n_prop_ops))
                prop_itor = self.physics.property_itor[0]
                prop_itor.evaluate_with_derivatives(
                    state0, self.block_idx, values0, self.dvalues
                )

                # Define a property array dictionary
                prop_arr0 = {prop: np.zeros(num_segments) for prop in pc.output_props}

                # Fill the prop dict
                values0_reshaped = np.asarray(values0).reshape(-1, self.n_prop_ops)
                for prop_idx, prop_name in enumerate(pc.output_props):
                    prop_arr0[prop_name] = values0_reshaped[:, prop_idx]

                sG0 = prop_arr0['sG']
                rhoG0 = prop_arr0['rhoG']
                muG0 = prop_arr0['muG'] * 1e-3  # convert cP to Pa.s
                xG_mass0 = np.zeros((num_segments, pc.nc_fl))
                for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                    xG_mass0[:, c_idx] = prop_arr0[f'x{c_name}_in_G_mass']

                if self.n_mobile_phases == 2:
                    sL0 = prop_arr0['sL']
                    rhoL0 = prop_arr0['rhoL']
                    muL0 = prop_arr0['muL'] * 1e-3  # convert cP to Pa.s
                    xL_mass0 = np.zeros((num_segments, pc.nc_fl))
                    for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                        xL_mass0[:, c_idx] = prop_arr0[f'x{c_name}_in_L_mass']

                if self.n_mobile_phases == 3:
                    sL_a_0 = prop_arr0['sL_a']
                    sL_b_0 = prop_arr0['sL_b']
                    sL0 = sL_a_0 + sL_b_0
                    rhoL_a_0 = prop_arr0['rhoL_a']
                    rhoL_b_0 = prop_arr0['rhoL_b']
                    muL_a_0 = prop_arr0['muL_a'] * 1e-3  # convert cP to Pa.s
                    muL_b_0 = prop_arr0['muL_b'] * 1e-3  # convert cP to Pa.s
                    xL_a_mass_0 = np.zeros((num_segments, pc.nc_fl))
                    for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                        xL_a_mass_0[:, c_idx] = prop_arr0[f'x{c_name}_in_L_a_mass']
                    xL_b_mass_0 = np.zeros((num_segments, pc.nc_fl))
                    for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                        xL_b_mass_0[:, c_idx] = prop_arr0[f'x{c_name}_in_L_b_mass']

                    # Calculate averaged liquid props
                    mask = sL0 > 0
                    rhoL0 = np.zeros(num_segments)
                    muL0 = np.zeros(num_segments)
                    xL_mass0 = np.zeros((num_segments, pc.nc_fl))
                    rhoL0[mask] = (
                        rhoL_a_0[mask] * sL_a_0[mask] + rhoL_b_0[mask] * sL_b_0[mask]
                    ) / sL0[mask]
                    muL0[mask] = (
                        muL_a_0[mask] * sL_a_0[mask] + muL_b_0[mask] * sL_b_0[mask]
                    ) / sL0[mask]
                    xL_mass0[mask, :] = (
                        xL_a_mass_0[mask, :] * rhoL_a_0[mask, None] * sL_a_0[mask, None]
                        + xL_b_mass_0[mask, :]
                        * rhoL_b_0[mask, None]
                        * sL_b_0[mask, None]
                    ) / (
                        rhoL_a_0[mask, None] * sL_a_0[mask, None]
                        + rhoL_b_0[mask, None] * sL_b_0[mask, None]
                    )

            elif self.prop_eval_method == "direct":
                sG0 = np.zeros(num_segments)
                sL0 = np.zeros(num_segments)
                rhoG0 = np.zeros(num_segments)
                rhoL0 = np.zeros(num_segments)
                muG0 = np.zeros(num_segments)
                muL0 = np.zeros(num_segments)
                xG_mass0 = np.zeros((num_segments, pc.nc_fl))
                xL_mass0 = np.zeros((num_segments, pc.nc_fl))

                if self.n_mobile_phases == 3:
                    sL_a_0 = np.zeros(num_segments)
                    sL_b_0 = np.zeros(num_segments)
                    rhoL_a_0 = np.zeros(num_segments)
                    rhoL_b_0 = np.zeros(num_segments)
                    muL_a_0 = np.zeros(num_segments)
                    muL_b_0 = np.zeros(num_segments)
                    xL_a_mass_0 = np.zeros((num_segments, pc.nc_fl))
                    xL_b_mass_0 = np.zeros((num_segments, pc.nc_fl))

                for i in range(num_segments):
                    state0 = Xn_dfm_well_2d[i, :]
                    pc.evaluate(state0)
                    if self.physics.thermal:
                        pc.evaluate_thermal(state0)

                    sG0[i] = pc.sat[g_idx]
                    rhoG0[i] = pc.dens[g_idx]
                    muG0[i] = pc.mu[g_idx] * 1e-3  # convert cP to Pa.s
                    if self.n_mobile_phases == 2:
                        sL0[i] = pc.sat[l_idx]
                        rhoL0[i] = pc.dens[l_idx]
                        muL0[i] = pc.mu[l_idx] * 1e-3  # convert cP to Pa.s
                        # Calculate mass fractions of components in each phase
                        x_mass0 = np.zeros((pc.np_fl, pc.nc_fl))
                        for j in pc.ph:
                            x_mass0[j, :] = (pc.x[j, :] * Mw_fl) / sum(
                                pc.x[j, :] * Mw_fl
                            )
                        xG_mass0[i, :], xL_mass0[i, :] = (
                            x_mass0[g_idx, :],
                            x_mass0[l_idx, :],
                        )

                    if self.n_mobile_phases == 3:
                        sL_a_0[i], sL_b_0[i] = pc.sat[la_idx], pc.sat[lb_idx]
                        sL0[i] = sL_a_0[i] + sL_b_0[i]
                        rhoL_a_0[i], rhoL_b_0[i] = pc.dens[la_idx], pc.dens[lb_idx]
                        muL_a_0[i], muL_b_0[i] = (
                            pc.mu[la_idx] * 1e-3,
                            pc.mu[lb_idx] * 1e-3,
                        )
                        # Calculate mass fractions of components in each phase
                        x_mass0 = np.zeros((pc.np_fl, pc.nc_fl))
                        for j in pc.ph:
                            x_mass0[j, :] = (pc.x[j, :] * Mw_fl) / sum(
                                pc.x[j, :] * Mw_fl
                            )
                        xG_mass0[i, :], xL_a_mass_0[i, :], xL_b_mass_0[i, :] = (
                            x_mass0[g_idx, :],
                            x_mass0[la_idx, :],
                            x_mass0[lb_idx, :],
                        )

                        # Calculate averaged liquid props
                        rhoL0[i] = (
                            (rhoL_a_0[i] * sL_a_0[i] + rhoL_b_0[i] * sL_b_0[i])
                            / (sL_a_0[i] + sL_b_0[i])
                            if (sL_a_0[i] + sL_b_0[i]) > 0
                            else 0
                        )
                        muL0[i] = (
                            (muL_a_0[i] * sL_a_0[i] + muL_b_0[i] * sL_b_0[i])
                            / (sL_a_0[i] + sL_b_0[i])
                            if (sL_a_0[i] + sL_b_0[i]) > 0
                            else 0
                        )
                        xL_mass0[i, :] = (
                            (
                                xL_a_mass_0[i, :] * rhoL_a_0[i] * sL_a_0[i]
                                + xL_b_mass_0[i, :] * rhoL_b_0[i] * sL_b_0[i]
                            )
                            / (rhoL_a_0[i] * sL_a_0[i] + rhoL_b_0[i] * sL_b_0[i])
                            if (sL_a_0[i] + sL_b_0[i]) > 0
                            else 0
                        )

            self.iter_phases_props0 = [
                xG_mass0,
                xL_mass0,
                sG0,
                sL0,
                rhoG0,
                rhoL0,
                muG0,
                muL0,
            ]

        elif iter_counter == 0 and not self.is_first_first_iter and flag == 1:
            self.iter_phases_props0 = self.iter_phases_props

        xG_mass0, xL_mass0, sG0, sL0, rhoG0, rhoL0, muG0, muL0 = self.iter_phases_props0

        """ Calculate phase props of current time step at centroids """
        if self.prop_eval_method == "OBL":
            state = value_vector(X_dfm_well)
            values = value_vector(np.zeros(num_segments * self.n_prop_ops))
            prop_itor = self.physics.property_itor[0]
            prop_itor.evaluate_with_derivatives(
                state, self.block_idx, values, self.dvalues
            )

            # Define a property array dictionary
            prop_arr = {prop: np.zeros(num_segments) for prop in pc.output_props}

            # Fill the prop dict
            values_reshaped = np.asarray(values).reshape(-1, self.n_prop_ops)
            for prop_idx, prop_name in enumerate(pc.output_props):
                prop_arr[prop_name] = values_reshaped[:, prop_idx]

            sG = prop_arr['sG']
            rhoG = prop_arr['rhoG']
            muG = prop_arr['muG'] * 1e-3  # convert cP to Pa.s
            xG_mass = np.zeros((num_segments, pc.nc_fl))
            for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                xG_mass[:, c_idx] = prop_arr[f'x{c_name}_in_G_mass']

            if self.n_mobile_phases == 2:
                sL = prop_arr['sL']
                rhoL = prop_arr['rhoL']
                muL = prop_arr['muL'] * 1e-3  # convert cP to Pa.s
                xL_mass = np.zeros((num_segments, pc.nc_fl))
                for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                    xL_mass[:, c_idx] = prop_arr[f'x{c_name}_in_L_mass']

            if self.n_mobile_phases == 3:
                sL_a = prop_arr['sL_a']
                sL_b = prop_arr['sL_b']
                sL = sL_a + sL_b
                rhoL_a = prop_arr['rhoL_a']
                rhoL_b = prop_arr['rhoL_b']
                muL_a = prop_arr['muL_a'] * 1e-3  # convert cP to Pa.s
                muL_b = prop_arr['muL_b'] * 1e-3  # convert cP to Pa.s
                xL_a_mass = np.zeros((num_segments, pc.nc_fl))
                for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                    xL_a_mass[:, c_idx] = prop_arr[f'x{c_name}_in_L_a_mass']
                xL_b_mass = np.zeros((num_segments, pc.nc_fl))
                for c_idx, c_name in enumerate(pc.components_name[: pc.nc_fl]):
                    xL_b_mass[:, c_idx] = prop_arr[f'x{c_name}_in_L_b_mass']

                # Calculate averaged liquid props
                mask = sL > 0
                rhoL = np.zeros(num_segments)
                muL = np.zeros(num_segments)
                xL_mass = np.zeros((num_segments, pc.nc_fl))
                rhoL[mask] = (
                    rhoL_a[mask] * sL_a[mask] + rhoL_b[mask] * sL_b[mask]
                ) / sL[mask]
                muL[mask] = (muL_a[mask] * sL_a[mask] + muL_b[mask] * sL_b[mask]) / sL[
                    mask
                ]
                xL_mass[mask, :] = (
                    xL_a_mass[mask, :] * rhoL_a[mask, None] * sL_a[mask, None]
                    + xL_b_mass[mask, :] * rhoL_b[mask, None] * sL_b[mask, None]
                ) / (
                    rhoL_a[mask, None] * sL_a[mask, None]
                    + rhoL_b[mask, None] * sL_b[mask, None]
                )

        elif self.prop_eval_method == "direct":
            sG = np.zeros(num_segments)
            sL = np.zeros(num_segments)
            rhoG = np.zeros(num_segments)
            rhoL = np.zeros(num_segments)
            muG = np.zeros(num_segments)
            muL = np.zeros(num_segments)
            xG_mass = np.zeros((num_segments, pc.nc_fl))
            xL_mass = np.zeros((num_segments, pc.nc_fl))

            if self.n_mobile_phases == 3:
                sL_a = np.zeros(num_segments)
                sL_b = np.zeros(num_segments)
                rhoL_a = np.zeros(num_segments)
                rhoL_b = np.zeros(num_segments)
                muL_a = np.zeros(num_segments)
                muL_b = np.zeros(num_segments)
                xL_a_mass = np.zeros((num_segments, pc.nc_fl))
                xL_b_mass = np.zeros((num_segments, pc.nc_fl))

            for i in range(num_segments):
                state = X_dfm_well_2d[i, :]
                pc.evaluate(state)
                if self.physics.thermal:
                    pc.evaluate_thermal(state)

                sG[i] = pc.sat[g_idx]
                rhoG[i] = pc.dens[g_idx]
                muG[i] = pc.mu[g_idx] * 1e-3  # convert cP to Pa.s
                if self.n_mobile_phases == 2:
                    sL[i] = pc.sat[l_idx]
                    rhoL[i] = pc.dens[l_idx]
                    muL[i] = pc.mu[l_idx] * 1e-3  # convert cP to Pa.s
                    # Calculate mass fractions of components in each phase
                    x_mass = np.zeros((pc.np_fl, pc.nc_fl))
                    for j in pc.ph:
                        x_mass[j, :] = (pc.x[j, :] * Mw_fl) / sum(pc.x[j, :] * Mw_fl)
                    xG_mass[i, :], xL_mass[i, :] = x_mass[g_idx, :], x_mass[l_idx, :]

                if self.n_mobile_phases == 3:
                    sL_a[i], sL_b[i] = pc.sat[la_idx], pc.sat[lb_idx]
                    sL[i] = sL_a[i] + sL_b[i]
                    rhoL_a[i], rhoL_b[i] = pc.dens[la_idx], pc.dens[lb_idx]
                    muL_a[i], muL_b[i] = pc.mu[la_idx] * 1e-3, pc.mu[lb_idx] * 1e-3
                    # Calculate mass fractions of components in each phase
                    x_mass = np.zeros((pc.np_fl, pc.nc_fl))
                    for j in pc.ph:
                        x_mass[j, :] = (pc.x[j, :] * Mw_fl) / sum(pc.x[j, :] * Mw_fl)
                    xG_mass[i, :], xL_a_mass[i, :], xL_b_mass[i, :] = (
                        x_mass[g_idx, :],
                        x_mass[la_idx, :],
                        x_mass[lb_idx, :],
                    )

                    # Calculate averaged liquid props
                    rhoL[i] = (
                        (rhoL_a[i] * sL_a[i] + rhoL_b[i] * sL_b[i])
                        / (sL_a[i] + sL_b[i])
                        if (sL_a[i] + sL_b[i]) > 0
                        else 0
                    )
                    muL[i] = (
                        (muL_a[i] * sL_a[i] + muL_b[i] * sL_b[i]) / (sL_a[i] + sL_b[i])
                        if (sL_a[i] + sL_b[i]) > 0
                        else 0
                    )
                    xL_mass[i, :] = (
                        (
                            xL_a_mass[i, :] * rhoL_a[i] * sL_a[i]
                            + xL_b_mass[i, :] * rhoL_b[i] * sL_b[i]
                        )
                        / (rhoL_a[i] * sL_a[i] + rhoL_b[i] * sL_b[i])
                        if (sL_a[i] + sL_b[i]) > 0
                        else 0
                    )

        self.iter_phases_props = [xG_mass, xL_mass, sG, sL, rhoG, rhoL, muG, muL]

        # If the differentiation method is OBL, calculate phase property derivatives
        if self.diff_method == "OBL":
            sG_der = self.get_op_der_matrix(op_idx=SAT_OP + g_idx)
            rhoG_der = self.get_op_der_matrix(op_idx=GRAV_OP + g_idx)
            if self.n_mobile_phases == 2:
                sL_der = self.get_op_der_matrix(op_idx=SAT_OP + l_idx)
                rhoL_der = self.get_op_der_matrix(op_idx=GRAV_OP + l_idx)
            elif self.n_mobile_phases == 3:
                rhoL_a_der = self.get_op_der_matrix(op_idx=GRAV_OP + la_idx)
                rhoL_b_der = self.get_op_der_matrix(op_idx=GRAV_OP + lb_idx)
                sL_a_der = self.get_op_der_matrix(op_idx=SAT_OP + la_idx)
                sL_b_der = self.get_op_der_matrix(op_idx=SAT_OP + lb_idx)
                sL_der = sL_a_der + sL_b_der

                sL = sL_a + sL_b
                mask = sL > 0
                rhoL_der = np.zeros_like(rhoL_a_der)
                num = (sL_a[:, None] + sL_b[:, None]) * (
                    sL_a[:, None] * rhoL_a_der
                    + rhoL_a[:, None] * sL_a_der
                    + sL_b[:, None] * rhoL_b_der
                    + rhoL_b[:, None] * sL_b_der
                ) - (
                    rhoL_a[:, None] * sL_a[:, None] + rhoL_b[:, None] * sL_b[:, None]
                ) * (sL_a_der + sL_b_der)
                rhoL_der[mask, :] = num[mask, :] / (sL[mask, None] ** 2)

        """ Calculate phase props of previous time step at interfaces """
        if iter_counter == 0 and flag == 1:
            sG0_face = (sG0[:-1] + sG0[1:]) / 2
            sL0_face = (sL0[:-1] + sL0[1:]) / 2

            # Initialize arrays to store interface properties
            rhoG0_face = np.zeros(num_segments - 1)
            rhoL0_face = np.zeros(num_segments - 1)
            muG0_face = np.zeros(num_segments - 1)
            muL0_face = np.zeros(num_segments - 1)
            # xG_mass0_face and xL_mass0_face for IFT calculation
            xG_mass0_face = np.zeros((num_segments - 1, pc.nc_fl))
            xL_mass0_face = np.zeros((num_segments - 1, pc.nc_fl))

            # Compute interface values using conditional averaging
            for i in range(num_segments - 1):
                if sG0[i] == 0:
                    # If no gas in segment i, use properties from segment i+1
                    rhoG0_face[i] = rhoG0[i + 1]
                    muG0_face[i] = muG0[i + 1]
                    xG_mass0_face[i] = xG_mass0[i + 1]
                elif sG0[i + 1] == 0:
                    # If no gas in segment i+1, use properties from segment i
                    rhoG0_face[i] = rhoG0[i]
                    muG0_face[i] = muG0[i]
                    xG_mass0_face[i] = xG_mass0[i]
                else:
                    # If both segments have gas, use averaging
                    rhoG0_face[i] = (rhoG0[i] + rhoG0[i + 1]) / 2
                    muG0_face[i] = (muG0[i] + muG0[i + 1]) / 2

                    xG_mass0_face[i] = (
                        xG_mass0[i] * rhoG0[i] * sG0[i]
                        + xG_mass0[i + 1] * rhoG0[i + 1] * sG0[i + 1]
                    ) / (rhoG0[i] * sG0[i] + rhoG0[i + 1] * sG0[i + 1])

                liquid_weight_i = rhoL0[i] * sL0[i]
                liquid_weight_ip1 = rhoL0[i + 1] * sL0[i + 1]
                liquid_weight_sum = liquid_weight_i + liquid_weight_ip1

                if liquid_weight_i == 0 and liquid_weight_ip1 > 0:
                    # If no liquid in segment i, use properties from segment i+1
                    rhoL0_face[i] = rhoL0[i + 1]
                    muL0_face[i] = muL0[i + 1]
                    xL_mass0_face[i] = xL_mass0[i + 1]
                elif liquid_weight_ip1 == 0 and liquid_weight_i > 0:
                    # If no liquid in segment i+1, use properties from segment i
                    rhoL0_face[i] = rhoL0[i]
                    muL0_face[i] = muL0[i]
                    xL_mass0_face[i] = xL_mass0[i]
                elif liquid_weight_sum > 0:
                    # If both segments have liquid, use averaging
                    rhoL0_face[i] = (rhoL0[i] + rhoL0[i + 1]) / 2
                    muL0_face[i] = (muL0[i] + muL0[i + 1]) / 2

                    xL_mass0_face[i] = (
                        xL_mass0[i] * rhoL0[i] * sL0[i]
                        + xL_mass0[i + 1] * rhoL0[i + 1] * sL0[i + 1]
                    ) / liquid_weight_sum
                else:
                    rhoL0_face[i] = 0.0
                    muL0_face[i] = 0.0
                    xL_mass0_face[i] = 0.0

            if self.exclude_top_control_block_from_velocity:
                sG0_face[0] = sG0[1]
                sL0_face[0] = sL0[1]
                rhoG0_face[0] = rhoG0[1]
                rhoL0_face[0] = rhoL0[1]
                muG0_face[0] = muG0[1]
                muL0_face[0] = muL0[1]
                xG_mass0_face[0] = xG_mass0[1]
                xL_mass0_face[0] = xL_mass0[1]

            self.iter_phases_props0_face = [
                xG_mass0_face,
                xL_mass0_face,
                sG0_face,
                sL0_face,
                rhoG0_face,
                rhoL0_face,
                muG0_face,
                muL0_face,
            ]

        """ Calculate phase props of current time step at interfaces """
        sG_face = (sG[:-1] + sG[1:]) / 2
        sL_face = (sL[:-1] + sL[1:]) / 2
        if self.diff_method == "OBL":
            sG_face_der = (sG_der[:-1, :] + sG_der[1:, :]) / 2
            sL_face_der = (sL_der[:-1, :] + sL_der[1:, :]) / 2

        # Initialize arrays to store interface properties
        rhoG_face = np.zeros(num_segments - 1)
        rhoL_face = np.zeros(num_segments - 1)

        if self.diff_method == "OBL":
            rhoG_face_der = np.zeros((num_interfaces, num_segments * n_vars))
            rhoL_face_der = np.zeros((num_interfaces, num_segments * n_vars))

        # Compute interface values using conditional averaging
        for i in range(num_interfaces):
            if sG[i] == 0:
                # If no gas in segment i, use properties from segment i+1
                rhoG_face[i] = rhoG[i + 1]
                if self.diff_method == "OBL":
                    rhoG_face_der[i, :] = rhoG_der[i + 1, :]
            elif sG[i + 1] == 0:
                # If no gas in segment i+1, use properties from segment i
                rhoG_face[i] = rhoG[i]
                if self.diff_method == "OBL":
                    rhoG_face_der[i, :] = rhoG_der[i, :]
            else:
                # If both segments have gas, use averaging
                rhoG_face[i] = (rhoG[i] + rhoG[i + 1]) / 2

                if self.diff_method == "OBL":
                    rhoG_face_der[i, :] = (rhoG_der[i, :] + rhoG_der[i + 1, :]) / 2

            if sL[i] == 0:
                # If no liquid in segment i, use properties from segment i+1
                rhoL_face[i] = rhoL[i + 1]
                if self.diff_method == "OBL":
                    rhoL_face_der[i, :] = rhoL_der[i + 1, :]
            elif sL[i + 1] == 0:
                # If no liquid in segment i+1, use properties from segment i
                rhoL_face[i] = rhoL[i]
                if self.diff_method == "OBL":
                    rhoL_face_der[i, :] = rhoL_der[i, :]
            else:
                # If both segments have liquid, use averaging
                rhoL_face[i] = (rhoL[i] + rhoL[i + 1]) / 2

                if self.diff_method == "OBL":
                    rhoL_face_der[i, :] = (rhoL_der[i, :] + rhoL_der[i + 1, :]) / 2

        if self.exclude_top_control_block_from_velocity:
            sG_face[0] = sG[1]
            sL_face[0] = sL[1]
            rhoG_face[0] = rhoG[1]
            rhoL_face[0] = rhoL[1]
            if self.diff_method == "OBL":
                sG_face_der[0, :] = sG_der[1, :]
                sL_face_der[0, :] = sL_der[1, :]
                rhoG_face_der[0, :] = rhoG_der[1, :]
                rhoL_face_der[0, :] = rhoL_der[1, :]

        self.iter_phases_props_face = [sG_face, sL_face, rhoG_face, rhoL_face]
        if self.diff_method == "OBL":
            self.iter_phases_props_face_ders = [
                sG_face_der,
                sL_face_der,
                rhoG_face_der,
                rhoL_face_der,
            ]

        if iter_counter == 0 and self.is_first_first_iter and flag == 1:
            # Initial velocities in the wellbore are zero
            rhoM0_vM0, vM0, vG0, vL0 = (
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
            )
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])
        elif iter_counter == 0 and not self.is_first_first_iter and flag == 1:
            rhoM0_vM0, vM0, vG0, vL0 = self.rhoM_vM, self.vM, self.vG, self.vL
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])

        [_, vM0, vG0, vL0] = self.velocities0

        p = X_dfm_well_2d[:, self.p_idx] * 1e5  # convert bar to Pa
        p_m = p[:-1]
        p_p = p[1:]
        if self.diff_method == "OBL":
            p_der = self.get_op_der_matrix(op_idx=PRES_OP) * 1e5
            p_m_der = p_der[0:-1:1, :]
            p_p_der = p_der[1::1, :]

        self.calc_mixture_densities(iter_counter, flag)

        if iter_counter == 0 and flag == 1:
            # To increase the numerical stability, you may need to use an upwind scheme for the momentum flux like
            # in the paper "A transient geothermal wellbore simulator (2023)"
            delta_interface0 = geom.pipe_internal_A * (
                rhoG0_face * sG0_face * vG0**2 + rhoL0_face * sL0_face * vL0**2
            )

            """ Add momentum boundary conditions """
            momentum_at_first_last_exterfaces = [0, 0]
            for sink_source in self.source_sinks.values():
                segment_idx_source = sink_source.segment_idx
                # Update current rate. Publish the size of the timestep first:
                # the schedule is integrated over [simulation_time,
                # simulation_time + dt] rather than sampled at its start, and
                # publishing it (as with receiving_segment_state) keeps the
                # single-argument signature subclasses override.
                sink_source.timestep_size = dt_day
                sink_source.update_current_molar_rate(simulation_time)
                rate_source = sink_source.current_rate  # Output rate is in kmol/day

                pipe_internal_A = geom.pipe_internal_A

                # Publish the state of the receiving segment. The SegmentProps
                # boundary property model (the plain RampUpRate default) builds
                # the boundary momentum from it; the upstream models ignore it.
                sink_source.receiving_segment_state = SegmentState(
                    sG=sG0[segment_idx_source],
                    sL=sL0[segment_idx_source],
                    rhoG=rhoG0[segment_idx_source],
                    rhoL=rhoL0[segment_idx_source],
                )
                delta_at_bc_interface0 = sink_source.get_boundary_momentum_flux(
                    pc, pipe_internal_A, rate_source
                )

                if segment_idx_source == 0:
                    momentum_at_first_last_exterfaces[0] = delta_at_bc_interface0
                elif segment_idx_source == num_segments - 1:
                    momentum_at_first_last_exterfaces[1] = delta_at_bc_interface0
                else:
                    raise Exception(
                        f"segment_idx_source must be either 0 or {num_segments - 1}! Got {segment_idx_source}!"
                    )

            delta_interface0 = np.insert(
                delta_interface0, 0, momentum_at_first_last_exterfaces[0]
            )
            delta_interface0 = np.insert(
                delta_interface0, num_segments, momentum_at_first_last_exterfaces[1]
            )

            delta_segment0 = (
                geom.pipe_internal_A * delta_interface0[0:-1] / geom.D[0:-1]
                + geom.pipe_internal_A * delta_interface0[1:] / geom.D[1:]
            ) / (
                geom.pipe_internal_A / geom.D[0:-1] + geom.pipe_internal_A / geom.D[1:]
            )
            self.delta_m0 = delta_segment0[0:-1]
            self.delta_p0 = delta_segment0[1:]

            ff0 = self.calc_Fanning_friction_factor()
            self.w0 = 1 / (
                1 / dt + geom.perimeter * ff0 * abs(vM0) / (2 * geom.pipe_internal_A)
            )

        self.rhoM_vM = (
            -self.w0 * (p_p - p_m) / (geom.z_p - geom.z_m)
            + self.w0 * self.g_cos_theta * self.rhoM_face
            - self.w0
            * (
                (self.delta_p0 - self.delta_m0)
                / (geom.pipe_internal_A * (geom.z_p - geom.z_m))
                - self.rhoM0_face * vM0 / dt
            )
        )

        if self.diff_method == "OBL":
            self.rhoM_vM_der = (
                -self.w0[:, None]
                * (p_p_der - p_m_der)
                / (geom.z_p[:, None] - geom.z_m[:, None])
                + self.w0[:, None] * self.g_cos_theta[:, None] * self.rhoM_face_der
            )

        self.vM = self.rhoM_vM / self.rhoM_face

        if iter_counter == 0 and flag == 1 and self.is_first_first_iter:
            self.vD0 = np.zeros(num_interfaces)
        elif iter_counter == 0 and flag == 1 and not self.is_first_first_iter:
            if self.enable_drift_velocity:
                self.update_drift_velocity()
            else:
                self.vD0 = np.zeros(num_interfaces)

        # If the differentiation method is OBL, preallocate derivative matrices
        if self.diff_method == "OBL":
            self.vG_der = np.zeros((num_interfaces, num_segments * n_vars))
            self.vL_der = np.zeros((num_interfaces, num_segments * n_vars))

        # Gas velocity at wellbore interfaces
        # self.vG = self.C00 * self.rhoM_vM / self.rhoM_adjusted_face + rhoL_face * self.vD0 / self.rhoM_adjusted_face
        self.vG = np.zeros(num_interfaces)
        for i in range(num_interfaces):
            if sG_face[i] != 0:
                self.vG[i] = (
                    self.C00[i] * self.rhoM_vM[i] / self.rhoM_adjusted_face[i]
                    + rhoL_face[i] * self.vD0[i] / self.rhoM_adjusted_face[i]
                )

                if self.diff_method == "OBL":
                    self.vG_der[i, :] = self.C00[i] * (
                        self.rhoM_vM_der[i, :] * self.rhoM_adjusted_face[i]
                        - self.rhoM_adjusted_face_der[i, :] * self.rhoM_vM[i]
                    ) / (self.rhoM_adjusted_face[i] ** 2) + self.vD0[i] * (
                        rhoL_face_der[i, :] * self.rhoM_adjusted_face[i]
                        - self.rhoM_adjusted_face_der[i, :] * rhoL_face[i]
                    ) / (self.rhoM_adjusted_face[i] ** 2)

        # Liquid velocity at wellbore interfaces
        self.vL = np.zeros(num_interfaces)
        for i in range(num_interfaces):
            if sL_face[i] > 0:
                self.vL[i] = (1 - self.C00[i] * sG_face[i]) * self.rhoM_vM[i] / (
                    sL_face[i] * self.rhoM_adjusted_face[i]
                ) - sG_face[i] * rhoG_face[i] * self.vD0[i] / (
                    sL_face[i] * self.rhoM_adjusted_face[i]
                )

                if self.diff_method == "OBL":
                    self.vL_der[i, :] = (
                        (
                            -self.C00[i] * sG_face_der[i, :] * self.rhoM_vM[i]
                            + (1 - self.C00[i] * sG_face[i]) * self.rhoM_vM_der[i, :]
                        )
                        * sL_face[i]
                        * self.rhoM_adjusted_face[i]
                        - (
                            sL_face_der[i, :] * self.rhoM_adjusted_face[i]
                            + sL_face[i] * self.rhoM_adjusted_face_der[i, :]
                        )
                        * (1 - self.C00[i] * sG_face[i])
                        * self.rhoM_vM[i]
                    ) / ((sL_face[i] * self.rhoM_adjusted_face[i]) ** 2) - (
                        (
                            self.vD0[i]
                            * (
                                sG_face_der[i, :] * rhoG_face[i]
                                + sG_face[i] * rhoG_face_der[i, :]
                            )
                        )
                        * (sL_face[i] * self.rhoM_adjusted_face[i])
                        - (
                            sL_face_der[i, :] * self.rhoM_adjusted_face[i]
                            + sL_face[i] * self.rhoM_adjusted_face_der[i, :]
                        )
                        * sG_face[i]
                        * rhoG_face[i]
                        * self.vD0[i]
                    ) / ((sL_face[i] * self.rhoM_adjusted_face[i]) ** 2)

        for i in range(num_interfaces):
            if i == 0 and self.exclude_top_control_block_from_velocity:
                # The ghost wellhead/control block is excluded from the DFM velocity
                # closure, so gate interface 0 with the first physical segment's
                # saturations for both flow directions.
                sG_upwind_pos = sG_upwind_neg = sG[1]
                sL_upwind_pos = sL_upwind_neg = sL[1]
            else:
                sG_upwind_pos, sG_upwind_neg = sG[i], sG[i + 1]
                sL_upwind_pos, sL_upwind_neg = sL[i], sL[i + 1]

            if self.vG[i] > 0 and sG_upwind_pos == 0:
                self.vG[i] = 0
                if self.diff_method == "OBL":
                    self.vG_der[i, :] = 0
            elif self.vG[i] < 0 and sG_upwind_neg == 0:
                self.vG[i] = 0
                if self.diff_method == "OBL":
                    self.vG_der[i, :] = 0

            if self.vL[i] > 0 and sL_upwind_pos == 0:
                self.vL[i] = 0
                if self.diff_method == "OBL":
                    self.vL_der[i, :] = 0
            elif self.vL[i] < 0 and sL_upwind_neg == 0:
                self.vL[i] = 0
                if self.diff_method == "OBL":
                    self.vL_der[i, :] = 0

        # Concatenate phase velocities and convert m/s to m/day
        phase_vels = np.concatenate((self.vG * 24 * 60 * 60, self.vL * 24 * 60 * 60))

        if self.diff_method == "OBL":
            self.vG_der *= 24 * 60 * 60
            self.vL_der *= 24 * 60 * 60

        return phase_vels

    def calc_mixture_densities(self, iter_counter, flag):
        if iter_counter == 0 and self.is_first_first_iter and flag == 1:
            _, _, sG0, sL0, rhoG0, rhoL0, _, _ = self.iter_phases_props0

            _, _, sG0_face, sL0_face, rhoG0_face, rhoL0_face, _, _ = (
                self.iter_phases_props0_face
            )
            self.rhoM0_face = (sG0_face * rhoG0_face + sL0_face * rhoL0_face) / (
                sG0_face + sL0_face
            )

        elif iter_counter == 0 and not self.is_first_first_iter and flag == 1:
            self.rhoM0_face = self.rhoM_face

        # Calculate mixture density
        _, _, sG, sL, rhoG, rhoL, _, _ = self.iter_phases_props

        [sG_face, sL_face, rhoG_face, rhoL_face] = self.iter_phases_props_face

        num = sG_face * rhoG_face + sL_face * rhoL_face
        den = sG_face + sL_face
        self.rhoM_face = num / den
        if self.diff_method == "OBL":
            sG_face_der, sL_face_der, rhoG_face_der, rhoL_face_der = (
                self.iter_phases_props_face_ders
            )
            self.rhoM_face_der = (
                (
                    sG_face_der * rhoG_face[:, None]
                    + sG_face[:, None] * rhoG_face_der
                    + sL_face_der * rhoL_face[:, None]
                    + sL_face[:, None] * rhoL_face_der
                )
                * den[:, None]
                - (sG_face_der + sL_face_der) * num[:, None]
            ) / den[:, None] ** 2

        # Calculate adjusted-mixture density
        if iter_counter == 0 and flag == 1 and self.is_first_first_iter:
            # At the beginning, there is no flow, so C00 is considered 1 everywhere.
            self.C00 = np.ones(self.geometry.num_interfaces)
        elif iter_counter == 0 and flag == 1 and not self.is_first_first_iter:
            if self.enable_profile_parameter:
                self.update_profile_parameter()
            else:
                self.C00 = np.ones(self.geometry.num_interfaces)

        self.rhoM_adjusted_face = (
            self.C00 * sG_face * rhoG_face + (1 - self.C00 * sG_face) * rhoL_face
        )
        if self.diff_method == "OBL":
            self.rhoM_adjusted_face_der = (
                self.C00[:, None]
                * (sG_face_der * rhoG_face[:, None] + sG_face[:, None] * rhoG_face_der)
                - self.C00[:, None] * sG_face_der * rhoL_face[:, None]
                + (1 - self.C00[:, None] * sG_face[:, None]) * rhoL_face_der
            )

    def calc_Fanning_friction_factor(self):
        geom = self.geometry

        """ Start calculating the Reynolds number """
        _, _, sG0_face, sL0_face, _, _, muG0_face, muL0_face = (
            self.iter_phases_props0_face
        )
        [_, vM0, _, _] = self.velocities0

        # Saturation-weighted average is used to calculate the mixture viscosity of two phases. The method is used in
        # Beggs and Brill's book: Eq. 1.38
        # My production engineering notebook: Pressure drop calc in wellbore for 2-phase flow with Beggs and Brill's method
        self.muM0 = (sG0_face * muG0_face + sL0_face * muL0_face) / (
            sG0_face + sL0_face
        )

        # Viscosity averaging method in https://doi.org/10.1016/j.ijmultiphaseflow.2021.103590
        # _, _, _, _, rhoG0_face, rhoL0_face, _, _ = self.iter_phases_props0_face
        # xg = sG0_face * rhoG0_face / (sG0_face * rhoG0_face + sL0_face * rhoL0_face)
        # denominator_1 = xg / muG0_face
        # denominator_1 = np.nan_to_num(denominator_1, nan=0.0)
        # denominator_2 = (1 - xg) / muL0_face
        # denominator_2 = np.nan_to_num(denominator_2, nan=0.0)
        # self.muM0 = 1 / (denominator_1 + denominator_2)

        Re0 = self.calc_Reynolds_number(vM0, self.rhoM0_face, self.muM0, geom.pipe_ID)
        """ End calculating the Reynolds number """

        # The wall-friction model belongs to the drift-flux closure: "wang_2014"
        # for bai_2023 and "colebrook_white" otherwise, unless the Pipe caller
        # asked for a specific friction_model.
        self.ff0 = self.drift_flux_closure.fanning_friction_factor(Re0)
        return self.ff0

    @staticmethod
    def calc_Reynolds_number(v, rho, mu, pipe_ID):
        """
        Calculate the Reynolds number

        :param v: Fluid velocity
        :param rho: Fluid density
        :param mu: Fluid viscosity
        :param pipe_ID: Pipe inside diameter
        """

        Re0 = rho * abs(v) * pipe_ID / mu

        return Re0

    # The friction-factor correlations and the Bhagwat-Ghajar helper terms now
    # live in darts.pipes.drift_flux, where each closure family owns its own
    # variant of them. These two aliases keep the historical Pipe entry points.
    colebrook = staticmethod(colebrook)
    wang_darcy_friction_factor = staticmethod(wang_darcy_friction_factor)

    def _face_props(self, ift=None):
        """
        Previous-timestep interface state handed to the drift-flux closure.

        :param ift: Interfacial tension on the two-phase interfaces [N/m], if known.
        """
        return FaceProps.from_pipe_arrays(
            self.iter_phases_props0_face,
            self.velocities0,
            self.rhoM0_face,
            ift=ift,
        )

    def _interfacial_tension(self, indices):
        """
        Gas-liquid interfacial tension on the two-phase interfaces.

        Constant or variable IFT using FluidModel depending on the class used by the user.

        :param indices: Interfaces carrying both phases.
        """
        xG_mass0_face, xL_mass0_face, _, _, rhoG0_face, rhoL0_face, _, _ = (
            self.iter_phases_props0_face
        )
        IFT_ev = self.physics.property_containers[0].IFT_ev
        IFT0_face = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            IFT0_face[i] = IFT_ev.evaluate(
                rhoG0_face[idx],
                rhoL0_face[idx],
                xG_mass0_face[idx],
                xL_mass0_face[idx],
            )
        return IFT0_face

    def update_profile_parameter(self):
        """
        Update the profile parameter C00 from the solution of the previous time step.

        The correlation itself belongs to the drift-flux closure; the pipe only
        supplies the interface state and the interfacial tension.
        """
        face = self._face_props()
        indices = face.indices
        self.IFT_face_filtered = (
            self._interfacial_tension(indices) if indices.size else np.array([])
        )
        face.ift = self.IFT_face_filtered

        self.C00 = self.drift_flux_closure.profile_parameter(face)
        self.C00_filtered = self.C00[indices]

    def update_drift_velocity(self):
        """
        Update the drift velocity vD0 from the solution of the previous time step.

        The closure returns the final, signed drift velocity in the pipe frame
        (positive from the wellhead downwards), so no sign is applied here.
        """
        face = self._face_props()
        indices = face.indices

        if indices.size and not self.enable_profile_parameter:
            # Use this function to evaluate the closure workspace (Kutateladze
            # number, characteristic velocity) and the interfacial tension the
            # drift velocity needs; reset C00 afterward because the profile
            # parameter itself is disabled.
            self.update_profile_parameter()
            self.C00 = np.ones(self.geometry.num_interfaces)
            self.C00_filtered = np.ones(len(indices))

        C00_filtered = None
        if indices.size:
            face.ift = self.IFT_face_filtered
            C00_filtered = self.C00_filtered

        self.vD0 = self.drift_flux_closure.drift_velocity(face, C00_filtered)

    def eval_phase_vels_and_ders(
        self, Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter
    ):
        """
        Evaluate pipe phase velocities and their derivatives with respect to primary variables

        :param Xn_dfm_well: Vector containing the state of pipe segments (ordered block by block) of the previous time step
        :type Xn_dfm_well: np.ndarray
        :param X_dfm_well: Vector containing the state of pipe segments (ordered block by block) of the current time step
        :type X_dfm_well: np.ndarray
        :param dt: Time step size [day]
        :type dt: float
        :param simulation_time: Simulation time [day]
        :type simulation_time: float
        :param iter_counter: Iteration counter of the current time step
        :type iter_counter: int
        """
        n_conns = self.geometry.num_interfaces
        n_vars = self.physics.n_vars
        nph = self.physics.nph

        # Evaluate phase velocities
        phase_vels = self.eval_phase_vels(
            Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter, flag=1
        )

        # Construct the matrix of derivatives of phase velocities
        if self.diff_method == "numerical":
            vel_der_matrix_G, vel_der_matrix_L = (
                self.differentiate_phase_vels_with_perturb(
                    phase_vels,
                    Xn_dfm_well,
                    X_dfm_well,
                    dt,
                    simulation_time,
                    iter_counter,
                )
            )
        elif self.diff_method == "OBL":
            vel_der_matrix_G = self.vG_der
            vel_der_matrix_L = self.vL_der

        # Extract the dense matrix of phase velocity derivatives
        vel_der_dense_G = vel_der_matrix_G[
            self.conn_row_idx, self.conn_local_col_idx
        ].astype(np.float64, copy=False)
        vel_der_dense_L = vel_der_matrix_L[
            self.conn_row_idx, self.conn_local_col_idx
        ].astype(np.float64, copy=False)

        # Process phase velocities for two- and three-phase flow
        vG, vL = phase_vels[:n_conns], phase_vels[n_conns:]
        phase_vels = np.zeros((n_conns, nph), dtype=np.float64)
        phase_vels[:, self.g_idx] = vG
        if self.n_mobile_phases == 2:
            phase_vels[:, self.l_idx] = vL
        elif self.n_mobile_phases == 3:
            phase_vels[:, self.la_idx] = vL
            phase_vels[:, self.lb_idx] = vL
        phase_vels = phase_vels.ravel(order="F")

        # Process derivatives of phase velocities for two- and three-phase flow
        vel_der_size_all = n_conns * 2 * n_vars
        phase_vels_ders = np.zeros((vel_der_size_all, nph), dtype=np.float64)
        phase_vels_ders[:, self.g_idx] = vel_der_dense_G.ravel()
        if self.n_mobile_phases == 2:
            phase_vels_ders[:, self.l_idx] = vel_der_dense_L.ravel()
        elif self.n_mobile_phases == 3:
            phase_vels_ders[:, self.la_idx] = vel_der_dense_L.ravel()
            phase_vels_ders[:, self.lb_idx] = vel_der_dense_L.ravel()
        phase_vels_ders = phase_vels_ders.ravel(order="F")

        return phase_vels, phase_vels_ders

    def get_op_der_matrix(self, op_idx):
        """
        Extract the derivative matrix of the specified operator for the well. This operator derivative
        matrix is used for differentiating phase velocities using the OBL approach.

        :param op_idx: Index of the desired operator
        :type op_idx: int

        Returns
        der : ndarray, shape (num_segments, num_segments * n_vars)
        """
        n = self.geometry.num_segments
        n_vars = self.physics.n_vars
        n_ops = self.physics.n_ops
        n_all_cells = self.reservoir.mesh.n_blocks

        well_obj = self.reservoir.get_well(self.name)
        start = well_obj.well_head_idx
        stop = well_obj.well_head_idx + n

        op_ders_arr = np.asarray(self.physics.engine.op_ders_arr)
        arr = op_ders_arr.reshape(n_all_cells, n_ops, n_vars)

        # (n, n_vars): derivative values for segments of this well, for this operator
        local = arr[start:stop, op_idx, :]

        der = np.zeros((n, n, n_vars), dtype=local.dtype)
        idx = np.arange(n)
        der[idx, idx, :] = local
        return der.reshape(n, n * n_vars)

    def differentiate_phase_vels_with_perturb(
        self, phase_vels, Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter
    ):
        """
        Differentiate phase velocities using the perturbation-based numerical differentiation
        """
        n_conns = self.geometry.num_interfaces
        n_vars = self.physics.n_vars
        n_phase_vels = n_conns * 2
        n_primary_vars = len(X_dfm_well)
        n_segments = self.geometry.num_segments

        # Preallocate the matrix of derivatives of phase velocities
        vel_der_matrix = np.zeros((n_phase_vels, n_primary_vars))

        for i in range(n_segments):
            # Derivatives of all the phase velocities with respect to the pressure of segment i
            X_dfm_well[i * n_vars] += self.eps_p
            vel_der_matrix[:, i * n_vars] = (
                self.eval_phase_vels(
                    Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter, flag=0
                )
                - phase_vels
            ) / self.eps_p
            X_dfm_well[i * n_vars] -= self.eps_p

            for j in range(1, self.physics.nc):
                # Derivatives of all the phase velocities with respect to the mole fraction of component j in segment i
                X_dfm_well[i * n_vars + j] += self.eps_z
                vel_der_matrix[:, i * n_vars + j] = (
                    self.eval_phase_vels(
                        Xn_dfm_well,
                        X_dfm_well,
                        dt,
                        simulation_time,
                        iter_counter,
                        flag=0,
                    )
                    - phase_vels
                ) / self.eps_z
                X_dfm_well[i * n_vars + j] -= self.eps_z

            if not self.isothermal:
                # Derivatives of all the phase velocities with respect to the temperature of segment i
                X_dfm_well[i * n_vars + n_vars - 1] += self.eps_temp
                vel_der_matrix[:, i * n_vars + n_vars - 1] = (
                    self.eval_phase_vels(
                        Xn_dfm_well,
                        X_dfm_well,
                        dt,
                        simulation_time,
                        iter_counter,
                        flag=0,
                    )
                    - phase_vels
                ) / self.eps_temp
                X_dfm_well[i * n_vars + n_vars - 1] -= self.eps_temp

        # Update properties at the current time step with the original primary variables (original X_dfm_well)
        # unaffected by eps_p, eps_temp, and eps_z
        _ = self.eval_phase_vels(
            Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter, flag=0
        )

        vel_der_matrix_G = vel_der_matrix[: n_phase_vels // 2, :]
        vel_der_matrix_L = vel_der_matrix[n_phase_vels // 2 :, :]

        return vel_der_matrix_G, vel_der_matrix_L
