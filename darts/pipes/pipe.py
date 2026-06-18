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

import math
from dataclasses import dataclass

from scipy.optimize import fsolve

from darts.engines import index_vector, value_vector
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.ramp_up_rate import RampUpRate
from darts.pipes.set_initial_conditions import (
    LinearAmbientTemperature,
    SingleAmbientTemperature,
)
from darts.pipes.units import *


@dataclass(frozen=True)
class AdjustmentFuncParams:
    # The following parameters are used in the adjustment function f(G,X) used for calculating the drift velocity
    Xm1 = 1.0
    Xm2 = 0.94
    Gm1 = 300.0
    Gm2 = 700.0
    alpha = 0.001
    lambdaa = 199.0  # lambdaa is used because lambda is a reserved keyword in Python


@dataclass(frozen=True)
class TangUnifiedDFParams:
    A: float
    B: float
    a1: float
    a2: float
    N1: float
    N2: float
    N3: float
    N4: float
    m1: float
    m2: float
    m3: float


class Pipe:
    g = 9.80665 * meter() / second() ** 2  # Gravitational acceleration

    adjustment_func_params = AdjustmentFuncParams()
    bhagwat_ghajar_drift_flux_models = ("bai_2023", "bhagwat_ghajar_2014")
    supported_drift_flux_models = (
        "shi_t2well",
        "tang_2019",
        *bhagwat_ghajar_drift_flux_models,
    )

    def __init__(
        self,
        pipe_name: str,
        pipe_geometry: PipeGeometry,
        physics,
        reservoir,
        initial_conditions: SingleAmbientTemperature | LinearAmbientTemperature,
        source_sinks: dict = None,
        immobile_phase_names: list[str] | None = None,
        drift_flux_model: str = "shi_t2well",
        tang_parameter_set: str = "olgas",
        friction_model: str | None = None,
        Cmax: float = 1.2,
        Fv: float = 1,
        prop_eval_method: str = "direct",
        diff_method: str = "OBL",
        eps_p: float = 1e-4,
        eps_temp: float = 0.1,
        eps_z: float = 0.00001,
        enable_profile_parameter: bool = True,
        enable_drift_velocity: bool = True,
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
        :type drift_flux_model: str
        :param tang_parameter_set: If tang_2019 is used as the drift-flux model, parameterization of the
                                   Tang et al. (2019) unified model. Allowed values are "olgas" and "tuffp".
        :type tang_parameter_set: str
        :param friction_model: Friction-factor closure to use:
                               - None selects "wang_2014" for drift_flux_model="bai_2023" and "colebrook_white" otherwise
                               - "colebrook_white" uses Colebrook-White correlation to calculate the friction factor
                               - "wang_2014" uses the Wang et al. (2014) supercritical-CO2 friction factor adopted by Bai et al.
        :type friction_model: str or None
        :param Cmax: A user-specified maximum profile parameter that can be tuned to match the observations and
                     could have a value between 1.0 and 1.5. It is set to:
                     --> 1.2 in ECLIPSE according to Shi et al. paper (Drift-Flux Modeling of Two-Phase Flow in Wellbores)
                     --> 1 in a wellbore simulator in Tonken et al. paper (A transient geothermal wellbore simulator)
        :type Cmax: float
        :param Fv: A multiplier on the flooding velocity fraction, set to be 1 by default, and its value can be tuned
                   to fit the observations.
        :type Fv: float
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

        if drift_flux_model not in self.supported_drift_flux_models:
            raise ValueError(
                "drift_flux_model must be one of "
                f"{', '.join(repr(model) for model in self.supported_drift_flux_models)}."
            )
        self.drift_flux_model = drift_flux_model

        if friction_model is None:
            friction_model = (
                "wang_2014"
                if self.drift_flux_model == "bai_2023"
                else "colebrook_white"
            )
        if friction_model not in ("colebrook_white", "wang_2014"):
            raise ValueError(
                "friction_model must be either 'colebrook_white' or 'wang_2014'."
            )
        self.friction_model = friction_model

        if tang_parameter_set not in ("olgas", "tuffp"):
            raise ValueError("tang_parameter_set must be either 'olgas' or 'tuffp'.")
        self.tang_parameter_set = tang_parameter_set

        self.Cku = 142
        self.Cw = 0.008

        if self.drift_flux_model == "shi_t2well":
            # I did not see anywhere to tell if I can use linear interp and extra here or not.
            if Cmax == 1:
                a1 = 0.06
                a2 = 0.21
                m0 = 1.85
                n1 = 0.21
                n2 = 0.95
            elif 1 < Cmax < 1.2:
                # Linear interpolation
                a1 = 0.06
                a2 = np.interp(Cmax, [1, 1.2], [0.21, 0.12])
                m0 = np.interp(Cmax, [1, 1.2], [1.85, 1.27])
                n1 = np.interp(Cmax, [1, 1.2], [0.21, 0.24])
                n2 = np.interp(Cmax, [1, 1.2], [0.95, 1.08])
            elif Cmax == 1.2:
                a1 = 0.06
                a2 = 0.12
                m0 = 1.27
                n1 = 0.24
                n2 = 1.08
            elif 1.2 < Cmax <= 1.5:
                # Linear extrapolation
                a1 = 0.06
                a2 = np.interp(Cmax, [1, 1.2], [0.21, 0.12])
                m0 = np.interp(Cmax, [1, 1.2], [1.85, 1.27])
                n1 = np.interp(Cmax, [1, 1.2], [0.21, 0.24])
                n2 = np.interp(Cmax, [1, 1.2], [0.95, 1.08])
            else:
                raise ValueError("Cmax value is out of the allowed range [1 to 1.5]")

            self.Fv = Fv
            self.profile_A = Cmax
            self.B = 2 / Cmax - 1.0667
            if isinstance(pipe_geometry.inclination_angle_radian, float):
                self.m = (
                    m0
                    * (np.cos(pipe_geometry.inclination_angle_radian) ** n1)
                    * (1 + np.sin(pipe_geometry.inclination_angle_radian)) ** n2
                    * np.ones(pipe_geometry.num_interfaces)
                )
            elif isinstance(pipe_geometry.inclination_angle_radian, np.ndarray):
                self.m = (
                    m0
                    * (np.cos(pipe_geometry.inclination_angle_radian) ** n1)
                    * (1 + np.sin(pipe_geometry.inclination_angle_radian)) ** n2
                )
            self.a1 = a1
            self.a2 = a2
            self.tang_df_params = None
        elif self.drift_flux_model == "tang_2019":
            if tang_parameter_set == "olgas":
                self.tang_df_params = TangUnifiedDFParams(
                    A=1.000,
                    B=0.773,
                    a1=0.591,
                    a2=0.786,
                    N1=1.968,
                    N2=1.759,
                    N3=0.574,
                    N4=0.477,
                    m1=1.000,
                    m2=2.300,
                    m3=1.000,
                )
            elif tang_parameter_set == "tuffp":
                self.tang_df_params = TangUnifiedDFParams(
                    A=1.088,
                    B=0.833,
                    a1=0.577,
                    a2=0.769,
                    N1=1.981,
                    N2=1.759,
                    N3=0.574,
                    N4=0.477,
                    m1=1.017,
                    m2=2.303,
                    m3=1.000,
                )
            self.profile_A = self.tang_df_params.A
            self.B = self.tang_df_params.B
            self.a1 = self.tang_df_params.a1
            self.a2 = self.tang_df_params.a2
            self.m = np.ones(pipe_geometry.num_interfaces)
            if isinstance(pipe_geometry.inclination_angle_radian, float):
                self.tang_theta = (
                    pipe_geometry.inclination_angle_radian - math.pi / 2.0
                ) * np.ones(pipe_geometry.num_interfaces)
            else:
                self.tang_theta = pipe_geometry.inclination_angle_radian - math.pi / 2.0

        elif self.drift_flux_model in self.bhagwat_ghajar_drift_flux_models:
            self.profile_A = None
            self.B = None
            self.a1 = None
            self.a2 = None
            self.m = np.ones(pipe_geometry.num_interfaces)
            self.tang_df_params = None
            if isinstance(pipe_geometry.inclination_angle_radian, float):
                self.bhagwat_ghajar_theta = (
                    pipe_geometry.inclination_angle_radian - math.pi / 2.0
                ) * np.ones(pipe_geometry.num_interfaces)
            else:
                self.bhagwat_ghajar_theta = (
                    pipe_geometry.inclination_angle_radian - math.pi / 2.0
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

        self.lateral_heat_rate_eval = None

        self.enable_profile_parameter = enable_profile_parameter
        self.enable_drift_velocity = enable_drift_velocity

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
                # Update current rate
                sink_source.update_current_molar_rate(simulation_time)
                rate_source = sink_source.current_rate  # Output rate is in kmol/day

                if sink_source.inflow_or_outflow == "inflow":
                    # comp_source in kmol/kmol
                    comp_source = sink_source.inj_fluid_props["composition"]
                    Mw = pc.Mw
                    # mass_rate in kg/s
                    mass_rate = sum(
                        rate_source * np.array(comp_source) * np.array(Mw)
                    ) / (24 * 60 * 60)
                elif sink_source.inflow_or_outflow == "outflow":
                    # TODO: For outflow, we have rate_source, which is in kmol/day, but we don't have comp_source, which
                    # is in kmol/kmol, from the user. Instead, we have xG_mass0 and xL_mass0, which are mass fractions.
                    # Need to see how we can get the overall composition of the source block in kmol/kmol.
                    mass_rate = 0

                pipe_internal_A = geom.pipe_internal_A

                # If UpstreamRampUpRate is used, which calculates boundary momentum using the properties of the boundary itself:
                if hasattr(sink_source, "get_boundary_momentum_flux"):
                    delta_at_bc_interface0 = sink_source.get_boundary_momentum_flux(
                        pc, pipe_internal_A, rate_source
                    )
                # If RampUpRate is used:
                else:
                    # The props of the fluid of the segment on which the constant mass rate source is defined are used.
                    sG0_source = sG0[segment_idx_source]
                    sL0_source = sL0[segment_idx_source]
                    rhoG0_source = rhoG0[segment_idx_source]
                    rhoL0_source = rhoL0[segment_idx_source]

                    has_mobile_liquid0 = rhoL0_source > 0 and sL0_source > 1e-12

                    # This section is written under the assumption that there is no solid phase in the source.
                    if sG0_source == 0 and has_mobile_liquid0:
                        vG0_source = 0
                        liquid_mass_fraction0 = 1
                        liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
                        vL0_source = (
                            liquid_mass_rate0 / rhoL0_source / (pipe_internal_A * 1)
                        )
                    elif sG0_source == 1 or not has_mobile_liquid0:
                        vL0_source = 0
                        gas_mass_fraction0 = 1
                        gas_mass_rate0 = mass_rate * gas_mass_fraction0
                        vG0_source = (
                            gas_mass_rate0 / rhoG0_source / (pipe_internal_A * 1)
                        )
                    elif sG0_source > 0 and has_mobile_liquid0:
                        gas_mass_fraction0 = (
                            sG0_source
                            * rhoG0_source
                            / (sG0_source * rhoG0_source + sL0_source * rhoL0_source)
                        )
                        gas_mass_rate0 = mass_rate * gas_mass_fraction0
                        vG0_source = (
                            gas_mass_rate0
                            / rhoG0_source
                            / (pipe_internal_A * sG0_source)
                        )

                        liquid_mass_fraction0 = 1 - gas_mass_fraction0
                        liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
                        vL0_source = (
                            liquid_mass_rate0
                            / rhoL0_source
                            / (pipe_internal_A * sL0_source)
                        )
                    else:
                        raise Exception(
                            "sG0_source is out of correct range (from 0 to 1)!"
                        )

                    delta_at_bc_interface0 = pipe_internal_A * (
                        rhoG0_source * sG0_source * vG0_source**2
                        + rhoL0_source * sL0_source * vL0_source**2
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
            if self.vG[i] > 0 and sG[i] == 0:
                self.vG[i] = 0
                if self.diff_method == "OBL":
                    self.vG_der[i, :] = 0
            elif self.vG[i] < 0 and sG[i + 1] == 0:
                self.vG[i] = 0
                if self.diff_method == "OBL":
                    self.vG_der[i, :] = 0

            if self.vL[i] > 0 and sL[i] == 0:
                self.vL[i] = 0
                if self.diff_method == "OBL":
                    self.vL_der[i, :] = 0
            elif self.vL[i] < 0 and sL[i + 1] == 0:
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
        _, _, sG0, sL0, _, _, _, _ = self.iter_phases_props0

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

        self.ff0 = np.zeros(geom.num_interfaces)
        if self.friction_model == "wang_2014":
            relative_roughness = geom.wall_roughness / geom.pipe_ID
            for i, Re in enumerate(Re0):
                self.ff0[i] = (
                    self.wang_darcy_friction_factor(Re, relative_roughness) / 4.0
                )
            return self.ff0

        # Laminar connections (Re==0 stays 0)
        lam = (Re0 > 0.0) & (Re0 < 2400.0)
        self.ff0[lam] = 16.0 / Re0[lam]

        # Turbulent connections
        turb_idx = np.nonzero(Re0 > 2400.0)[0]

        initial_guess = 0.005
        relative_roughness = geom.wall_roughness / geom.pipe_ID
        for i in turb_idx:
            # Colebrook-White correlation (implicit method)
            self.ff0[i] = fsolve(
                self.colebrook, initial_guess, args=(Re0[i], relative_roughness)
            )[0]
            # self.ff0[i] = fsolve(self.colebrook, initial_guess, args=(Re0[i], relative_roughness), xtol=1e-10)[0]

            # # T2Well
            # self.ff0[i] = ((1 / (-4 * math.log10(2 * relative_roughness / 3.7 - 5.02 / Re0[i]
            #                 * math.log10(2 * relative_roughness / 3.7 + 13 / Re0[i])))) ** 2)

            # # Chen's correlation (The explicit form of Colebrook-White's correlation)
            # self.ff0[i] = (1 / (-4 * math.log10(relative_roughness / 3.7065 - 5.0452 / Re0[i]
            #                 * math.log10(relative_roughness ** 1.1098 / 2.8257 + (7.149 / Re0[i]) ** 0.8981)))) ** 2

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

    @staticmethod
    def colebrook(f, Re, relative_roughness):
        """
        Calculate the friction factor using the Colebrook-White correlation (implicit method)

        :param f: Guessed Fanning friction factor
        :param Re: Reynolds number
        :param relative_roughness: Pipe relative roughness
        """
        f = f[0]  # Make sure f passed to math.sqrt is a float
        # Ensure the friction factor doesn't go negative or zero
        # Return a large value to prevent sqrt of negative number
        if f <= 0:
            return 1e6

        sqrt_f = math.sqrt(f)
        return 1.0 / sqrt_f + 4.0 * math.log10(
            relative_roughness / 3.7065 + 1.2613 / (Re * sqrt_f)
        )

    @staticmethod
    def wang_darcy_friction_factor(Re: float, relative_roughness: float) -> float:
        """
        Calculate the Darcy friction factor with the Wang et al. (2014)
        supercritical-CO2 correlation adopted by Bai et al. (2023).

        Bai et al. use this correlation in their two-phase pure-CO2
        pressure-gradient model because it was developed for CO2 pipe
        flow. The returned value is a Darcy friction factor; callers that
        use the historical open-DARTS Fanning-friction convention must
        divide this value by 4.

        :param Re: Mixture Reynolds number.
        :param relative_roughness: Pipe relative roughness, wall roughness divided by pipe diameter.
        :return: Darcy friction factor.
        """
        if Re <= 0.0:
            return 0.0
        if Re < 2400.0:
            return 64.0 / Re

        inner = (relative_roughness / 29.36) ** 0.95 + (18.35 / Re) ** 1.108
        argument = relative_roughness / 1.72 - (9.26 / Re) * math.log10(inner)
        if argument <= 0.0:
            # Fall back to the existing Colebrook Fanning form converted to Darcy.
            return float(
                4.0 * fsolve(Pipe.colebrook, 0.005, args=(Re, relative_roughness))[0]
            )
        return (-2.34 * math.log10(argument)) ** -2.0

    @staticmethod
    def bhagwat_ghajar_gas_froude_number(
        j_g: float, rhoG: float, rhoL: float, theta: float, pipe_ID: float
    ) -> float:
        """
        Calculate the gas superficial Froude number used by Bhagwat and Ghajar.

        :param j_g: Gas superficial velocity [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :param theta: Pipe inclination angle [rad], measured from horizontal.
        :param pipe_ID: Pipe internal diameter [m].
        :return: Gas superficial Froude number.
        """
        density_difference = rhoL - rhoG
        cos_theta = math.cos(theta)
        if density_difference <= 0.0 or cos_theta <= np.finfo(float).eps:
            return math.inf
        return (
            math.sqrt(rhoG / density_difference)
            * abs(j_g)
            / math.sqrt(Pipe.g * pipe_ID * cos_theta)
        )

    @staticmethod
    def bhagwat_ghajar_gas_volumetric_fraction(j_g: float, j_l: float) -> float:
        """
        Calculate gas volumetric flow fraction from superficial velocities.

        :param j_g: Gas superficial velocity [m/s].
        :param j_l: Liquid superficial velocity [m/s].
        :return: Gas volumetric flow fraction.
        """
        total = abs(j_g) + abs(j_l)
        return 0.0 if total <= 0.0 else abs(j_g) / total

    @staticmethod
    def bhagwat_ghajar_gas_quality(
        j_g: float, j_l: float, rhoG: float, rhoL: float
    ) -> float:
        """
        Calculate gas mass quality from superficial velocities and densities.

        :param j_g: Gas superficial velocity [m/s].
        :param j_l: Liquid superficial velocity [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :return: Gas mass quality.
        """
        gas_mass_flux = rhoG * abs(j_g)
        liquid_mass_flux = rhoL * abs(j_l)
        total = gas_mass_flux + liquid_mass_flux
        return 0.0 if total <= 0.0 else gas_mass_flux / total

    def bhagwat_ghajar_c01_downward_low_froude_condition(
        self, theta_deg: float, Fr_sg: float
    ) -> bool:
        """
        Check the near-horizontal downward-flow condition used for C0,1.
        """
        return -50.0 <= theta_deg <= 0.0 and Fr_sg <= 0.1

    def bhagwat_ghajar_c4_downward_low_froude_condition(
        self, theta_deg: float, Fr_sg: float
    ) -> bool:
        """
        Check the near-horizontal downward-flow condition used for C4.
        """
        if self.drift_flux_model == "bai_2023":
            return -50.0 <= theta_deg <= 0.0 and Fr_sg <= 0.1
        return -50.0 <= theta_deg < 0.0 and Fr_sg <= 0.1

    def bhagwat_ghajar_profile_friction_factor(self, Re: float) -> float:
        """
        Calculate the Fanning friction factor used inside the B&G-family C0,1 term.
        """
        relative_roughness = self.geometry.wall_roughness / self.geometry.pipe_ID
        if self.drift_flux_model == "bai_2023":
            return self.wang_darcy_friction_factor(Re, relative_roughness) / 4.0
        if Re <= 0.0:
            return 0.0
        if Re < 2400.0:
            return 16.0 / Re
        return float(fsolve(self.colebrook, 0.005, args=(Re, relative_roughness))[0])

    def bhagwat_ghajar_profile_parameter(
        self,
        sG: float,
        sL: float,
        j_g: float,
        j_l: float,
        mixture_velocity: float,
        rhoG: float,
        rhoL: float,
        muG: float,
        muL: float,
        theta: float,
    ) -> float:
        """
        Calculate the B&G-family distribution coefficient.

        :param sG: Gas volume fraction.
        :param sL: Liquid volume fraction.
        :param j_g: Gas superficial velocity [m/s].
        :param j_l: Liquid superficial velocity [m/s].
        :param mixture_velocity: Mixture velocity magnitude [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :param muG: Gas viscosity [Pa.s].
        :param muL: Liquid viscosity [Pa.s].
        :param theta: Pipe inclination angle [rad], measured from horizontal.
        :return: Distribution coefficient C0.
        """
        if self.drift_flux_model == "bai_2023":
            rho_m = (sG * rhoG + sL * rhoL) / (sG + sL)
            mu_m = (sG * muG + sL * muL) / (sG + sL)
            Re = (
                rho_m
                * abs(mixture_velocity)
                * self.geometry.pipe_ID
                / max(mu_m, np.finfo(float).eps)
            )
        else:
            Re = (
                rhoL
                * abs(mixture_velocity)
                * self.geometry.pipe_ID
                / max(muL, np.finfo(float).eps)
            )

        fanning_f = self.bhagwat_ghajar_profile_friction_factor(Re)
        beta = self.bhagwat_ghajar_gas_volumetric_fraction(j_g, j_l)
        quality = self.bhagwat_ghajar_gas_quality(j_g, j_l, rhoG, rhoL)
        Fr_sg = self.bhagwat_ghajar_gas_froude_number(
            j_g, rhoG, rhoL, theta, self.geometry.pipe_ID
        )
        theta_deg = math.degrees(theta)

        if self.bhagwat_ghajar_c01_downward_low_froude_condition(theta_deg, Fr_sg):
            C01 = 0.0
        else:
            C01 = (
                0.2
                * (1.0 - math.sqrt(rhoG / rhoL))
                * ((2.6 - beta) ** 0.15 - math.sqrt(max(fanning_f, 0.0)))
                * (1.0 - quality) ** 1.5
            )

        density_ratio = rhoG / rhoL
        cos_theta = math.cos(theta)
        denominator = max(1.0 + cos_theta, np.finfo(float).eps)
        base = math.sqrt(
            max(
                (1.0 + density_ratio**2 * cos_theta) / denominator,
                np.finfo(float).eps,
            )
        )
        # In Equation 10 of Bhagwat,and Ghajar (2014) paper, sL and 2/5 need to be multiplied, but
        # in Equation 13 of Bai et al. (2023) paper, sL is powered by 2/5. I tried both for the
        # two CI tests comparing isothermal two-phase flow in wellbore with OLGA. The results using
        # power are much closer and make more sense.
        exponent = sL ** (2.0 / 5.0)
        low_re_term = (2.0 - density_ratio**2) / (1.0 + (Re / 1000.0) ** 2)
        high_re_term = (base**exponent + C01) / (
            1.0 + (1000.0 / max(Re, np.finfo(float).eps)) ** 2
        )
        return low_re_term + high_re_term

    def bhagwat_ghajar_drift_velocity(
        self,
        sL: float,
        j_g: float,
        rhoG: float,
        rhoL: float,
        muL: float,
        sigma: float,
        theta: float,
    ) -> float:
        """
        Calculate the B&G-family drift velocity.

        :param sL: Liquid volume fraction.
        :param j_g: Gas superficial velocity [m/s].
        :param rhoG: Gas density [kg/m3].
        :param rhoL: Liquid density [kg/m3].
        :param muL: Liquid dynamic viscosity [Pa.s].
        :param sigma: Gas-liquid surface tension [N/m].
        :param theta: Bai pipe inclination angle [rad], measured from horizontal.
        :return: Drift velocity [m/s] in the pipe coordinate system.
        """
        viscosity_ratio = muL / 0.001
        if viscosity_ratio > 10.0:
            C2 = (0.434 / math.log10(viscosity_ratio)) ** 0.15
        else:
            C2 = 1.0

        La = math.sqrt(sigma / (self.g * (rhoL - rhoG))) / self.geometry.pipe_ID
        if self.drift_flux_model == "bai_2023":
            C3 = (La / 0.025) ** 0.9 if La > 0.025 else 1.0
            liquid_holdup_factor = sL
        else:
            C3 = (La / 0.025) ** 0.9 if La < 0.025 else 1.0
            liquid_holdup_factor = math.sqrt(sL)
        theta_deg = math.degrees(theta)
        Fr_sg = self.bhagwat_ghajar_gas_froude_number(
            j_g, rhoG, rhoL, theta, self.geometry.pipe_ID
        )
        C4 = (
            -1.0
            if self.bhagwat_ghajar_c4_downward_low_froude_condition(theta_deg, Fr_sg)
            else 1.0
        )
        return (
            (0.35 * math.sin(theta) + 0.45 * math.cos(theta))
            * math.sqrt(self.g * self.geometry.pipe_ID * (rhoL - rhoG) / rhoL)
            * liquid_holdup_factor
            * C2
            * C3
            * C4
        )

    def update_profile_parameter(self):
        geom = self.geometry
        num_interfaces = geom.num_interfaces
        [rhoM0_vM0, vM0_all, vG0, vL0] = self.velocities0
        [
            xG_mass0_face,
            xL_mass0_face,
            sG0_face,
            sL0_face,
            rhoG0_face,
            rhoL0_face,
            _,
            _,
        ] = self.iter_phases_props0_face
        self.IFT_face_filtered = np.array([])

        mask = (sG0_face > 0) & (sL0_face > 0)
        at_least_one_true = any(mask)
        if at_least_one_true:
            indices = np.where(mask)[0]
            sG0_face_filtered = sG0_face[indices]
            rhoG0_face_filtered = rhoG0_face[indices]
            rhoL0_face_filtered = rhoL0_face[indices]
            rhoM0_vM0_filtered = rhoM0_vM0[indices]
            rhoM0_face_filtered = self.rhoM0_face[indices]
            # For IFT calculation
            xG_mass0_face_filtered = xG_mass0_face[indices]
            xL_mass0_face_filtered = xL_mass0_face[indices]

            # Constant or variable IFT using FluidModel depending on the class used by the user
            IFT0_face = np.zeros(len(indices))
            for i in range(len(indices)):
                IFT0_face[i] = self.physics.property_containers[0].IFT_ev.evaluate(
                    rhoG0_face_filtered[i],
                    rhoL0_face_filtered[i],
                    xG_mass0_face_filtered[i],
                    xL_mass0_face_filtered[i],
                )
            self.IFT_face_filtered = IFT0_face

            # self.Ku0_filtered = np.zeros(len(indices))
            # self.vC0_filtered = np.zeros(len(indices))

            # Calculate C00 from the solution of the previous time step
            vM0 = rhoM0_vM0_filtered / rhoM0_face_filtered
            NB0 = (geom.pipe_ID**2) * (
                self.g * (rhoL0_face_filtered - rhoG0_face_filtered) / IFT0_face
            )
            if self.drift_flux_model == "tang_2019":
                Dhat0 = np.sqrt(NB0)
                self.Ku0_filtered = np.clip(3.587 - 19.105 / (Dhat0 + 3.333), 0.0, 3.2)
            else:
                self.Ku0_filtered = np.sqrt(
                    self.Cku
                    / np.sqrt(NB0)
                    * (np.sqrt(1 + NB0 / (self.Cku**2 * self.Cw)) - 1)
                )
            self.vC0_filtered = (
                self.g
                * IFT0_face
                * (rhoL0_face_filtered - rhoG0_face_filtered)
                / rhoL0_face_filtered**2
            ) ** 0.25
            v_sgf0 = (
                self.Ku0_filtered
                * np.sqrt(rhoL0_face_filtered / rhoG0_face_filtered)
                * self.vC0_filtered
            )

            if self.drift_flux_model == "shi_t2well":
                flooding_fraction = self.Fv * sG0_face_filtered * abs(vM0) / v_sgf0
                beta0 = np.maximum(sG0_face_filtered, flooding_fraction)
                beta0 = np.clip(beta0, 0, 1)  # T2Well imposes 0 <= beta0 <= 1
                eta0 = (beta0 - self.B) / (1 - self.B)
                eta0 = np.clip(eta0, 0, 1)  # Shi et al. impose 0 <= eta <= 1
                C00_filtered = self.profile_A / (1 + (self.profile_A - 1) * eta0**2)
            elif self.drift_flux_model == "tang_2019":
                flooding_fraction = sG0_face_filtered * abs(vM0) / v_sgf0
                beta0 = np.maximum(sG0_face_filtered, flooding_fraction)
                beta0 = np.clip(beta0, 0, 1)  # T2Well imposes 0 <= beta0 <= 1
                eta0 = (beta0 - self.B) / (1 - self.B)
                eta0_squared = np.minimum(eta0**2, 1.0)
                C00_filtered = self.profile_A / (
                    1 + (self.profile_A - 1) * eta0_squared
                )
            elif self.drift_flux_model in self.bhagwat_ghajar_drift_flux_models:
                muG0_face = self.iter_phases_props0_face[6]
                muL0_face = self.iter_phases_props0_face[7]
                C00_filtered = np.zeros(len(indices))
                for i, idx in enumerate(indices):
                    jG0 = sG0_face[idx] * vG0[idx]
                    jL0 = sL0_face[idx] * vL0[idx]
                    if jG0 == 0.0 and jL0 == 0.0:
                        jG0 = sG0_face[idx] * vM0_all[idx]
                        jL0 = sL0_face[idx] * vM0_all[idx]
                    C00_filtered[i] = self.bhagwat_ghajar_profile_parameter(
                        sG0_face[idx],
                        sL0_face[idx],
                        jG0,
                        jL0,
                        abs(vM0_all[idx]),
                        rhoG0_face[idx],
                        rhoL0_face[idx],
                        muG0_face[idx],
                        muL0_face[idx],
                        self.bhagwat_ghajar_theta[idx],
                    )

            C00 = np.ones(num_interfaces)  # C00 all ones first
            self.C00 = np.ones(num_interfaces)
            self.C00_filtered = np.ones(len(indices))

            for i, idx in enumerate(indices):
                C00[idx] = C00_filtered[i]

            self.C00_filtered = C00_filtered
            self.C00 = C00
            # Set all profile parameters equal to 1
            # self.C00 = np.ones(num_interfaces)

        else:
            self.C00_filtered = 1
            self.C00 = np.ones(num_interfaces)

    def update_drift_velocity(self):
        geom = self.geometry
        num_interfaces = geom.num_interfaces
        # if np.all(self.C00 == 1):
        #     vD0 = np.zeros(num_interfaces)
        # else:
        if any(
            (sG > 0) and (sL > 0)
            for sG, sL in zip(
                self.iter_phases_props0_face[2],
                self.iter_phases_props0_face[3],
                strict=False,
            )
        ):
            [_, _, sG0_face, sL0_face, rhoG0_face, rhoL0_face, _, muL0_face] = (
                self.iter_phases_props0_face
            )
            [_, vM0, vG0, vL0] = self.velocities0

            mask = (sG0_face > 0) & (sL0_face > 0)
            indices = np.where(mask)[0]
            sG0_face_filtered = sG0_face[indices]
            sL0_face_filtered = sL0_face[indices]
            rhoG0_face_filtered = rhoG0_face[indices]
            rhoL0_face_filtered = rhoL0_face[indices]
            vM0_filtered = vM0[indices]
            rhoM0_face_filtered = self.rhoM0_face[indices]

            if not self.enable_profile_parameter:
                # Use this function to evaluate Ku0_filtered and vC0_filtered for drift velocity;
                # reset C00 afterward because the profile parameter itself is disabled.
                self.update_profile_parameter()
                self.C00 = np.ones(num_interfaces)
                self.C00_filtered = np.ones(len(indices))

            if self.drift_flux_model in self.bhagwat_ghajar_drift_flux_models:
                vD0 = np.zeros(num_interfaces)
                for index, value in enumerate(indices):
                    jG0 = sG0_face[value] * vG0[value]
                    jL0 = sL0_face[value] * vL0[value]
                    if jG0 == 0.0 and jL0 == 0.0:
                        jG0 = sG0_face[value] * vM0[value]
                        jL0 = sL0_face[value] * vM0[value]
                    vD0[value] = self.bhagwat_ghajar_drift_velocity(
                        sL0_face[value],
                        jG0,
                        rhoG0_face[value],
                        rhoL0_face[value],
                        muL0_face[value],
                        self.IFT_face_filtered[index],
                        self.bhagwat_ghajar_theta[value],
                    )
                self.vD0 = vD0
                return

            # Calculate the K function to make a smooth transition of drift velocity between
            # the bubble-rise and film-flooding stages
            K0_filtered = np.zeros(len(self.C00_filtered))
            # If type of self.Ku0_filtered is float, this if block turns it into a list.
            if type(self.Ku0_filtered) is float:
                self.Ku0_filtered = [self.Ku0_filtered]

            for index, value in enumerate(indices):
                if sG0_face[value] <= self.a1:
                    K0_filtered[index] = 1.53
                elif self.a1 < sG0_face[value] < self.a2:
                    K0_filtered[index] = 1.53 + (
                        self.C00_filtered[index] * self.Ku0_filtered[index] - 1.53
                    ) / 2 * (
                        1
                        - np.cos(
                            math.pi * (sG0_face[value] - self.a1) / (self.a2 - self.a1)
                        )
                    )
                elif sG0_face[value] >= self.a2:
                    K0_filtered[index] = (
                        self.C00_filtered[index] * self.Ku0_filtered[index]
                    )

            if self.drift_flux_model == "tang_2019":
                eps = np.finfo(float).eps
                safe_rhoG_face = np.maximum(rhoG0_face_filtered, eps)
                safe_rhoL_face = np.maximum(rhoL0_face_filtered, eps)
                safe_muL_face = np.maximum(muL0_face[indices], eps)

                tang_params = self.tang_df_params
                theta_filtered = self.tang_theta[indices]

                denominator_vd = (
                    self.C00_filtered
                    * sG0_face_filtered
                    * np.sqrt(safe_rhoG_face / safe_rhoL_face)
                    + 1
                    - self.C00_filtered * sG0_face_filtered
                )
                vDv = (
                    (1 - self.C00_filtered * sG0_face_filtered)
                    * self.vC0_filtered
                    * K0_filtered
                    / np.maximum(denominator_vd, eps)
                )

                N_l = safe_muL_face / np.maximum(
                    (safe_rhoL_face - safe_rhoG_face)
                    * np.power(geom.pipe_ID, 1.5)
                    * math.sqrt(self.g),
                    eps,
                )
                N_Eo = (
                    self.g
                    * (safe_rhoL_face - safe_rhoG_face)
                    * (geom.pipe_ID**2)
                    / np.maximum(self.IFT_face_filtered, eps)
                )
                vDh = (
                    np.sqrt(self.g * geom.pipe_ID)
                    * (
                        tang_params.N1
                        - tang_params.N2
                        * (N_l**tang_params.N4)
                        / np.maximum(
                            N_Eo**tang_params.N3,
                            eps,
                        )
                    )
                    * sG0_face_filtered
                    * sL0_face_filtered
                )

                transition_angle = theta_filtered + np.deg2rad(
                    tang_params.m2 * vM0_filtered
                )
                transition_argument = 50.0 * np.sin(transition_angle)
                transition = 1 - 2 / (
                    1 + np.exp(np.clip(transition_argument, -700.0, 700.0))
                )
                Re_L = (
                    np.abs(vM0_filtered) * safe_rhoL_face * geom.pipe_ID / safe_muL_face
                )
                low_re_multiplier = (1 + 1000.0 / (Re_L + 1000.0)) ** tang_params.m3

                vD_filtered = (
                    tang_params.m1 * vDv * np.sin(theta_filtered)
                    + transition * vDh * np.cos(theta_filtered)
                ) * low_re_multiplier

                vD0 = np.zeros(num_interfaces)
                vD0[indices] = vD_filtered
                self.vD0 = vD0
                return

            # Calculate the adjustment function for the mist flow regime
            Xm1 = self.adjustment_func_params.Xm1
            Xm2 = self.adjustment_func_params.Xm2
            Gm1 = self.adjustment_func_params.Gm1
            Gm2 = self.adjustment_func_params.Gm2
            alpha = self.adjustment_func_params.alpha
            lambdaa = self.adjustment_func_params.lambdaa

            # I'm not sure if X should be multiplied by C0 or not.
            # Calculate gas mass fraction [dimensionless]
            X0 = (
                sG0_face_filtered
                * rhoG0_face_filtered
                / (
                    sG0_face_filtered * rhoG0_face_filtered
                    + sL0_face_filtered * rhoL0_face_filtered
                )
            )

            # Calculate G0: the total mass flux (or total mass flow rate per unit cross-sectional area) [kg/m2/s]
            G0 = rhoM0_face_filtered * abs(vM0_filtered)

            # Determinant of the matrix
            numerator = alpha * (
                X0 * (Gm1 - Gm2) - G0 * (Xm1 - Xm2) + (Xm1 * Gm2 - Xm2 * Gm1)
            )

            denominator = np.sqrt((Xm2 - Xm1) ** 2 + (alpha * Gm2 - alpha * Gm1) ** 2)
            Dm = numerator / denominator
            f0 = np.maximum(
                0.0, 1 - np.minimum(1, G0 / Gm1) * np.exp(-lambdaa * Dm * abs(Dm))
            )

            # Calculate drift velocity
            vD0 = np.zeros(num_interfaces)  # vD0 all zeros first
            for index, value in enumerate(indices):
                vD0[value] = (
                    (1 - self.C00_filtered[index] * sG0_face_filtered[index])
                    * self.vC0_filtered[index]
                    * K0_filtered[index]
                    * self.m[value]
                    * f0[index]
                    / (
                        self.C00_filtered[index]
                        * sG0_face_filtered[index]
                        * np.sqrt(
                            rhoG0_face_filtered[index] / rhoL0_face_filtered[index]
                        )
                        + 1
                        - self.C00_filtered[index] * sG0_face_filtered[index]
                    )
                )
                # vD0[value] = (1 - self.C00_filtered[index] * sG0_face_filtered[index]) * self.vC0_filtered[index] * K0_filtered[index] * self.m[value] / (self.C00_filtered[index] * sG0_face_filtered[index] * np.sqrt(rhoG0_face_filtered[index] / rhoL0_face_filtered[index]) + 1 - self.C00_filtered[index] * sG0_face_filtered[index])
        else:
            vD0 = np.zeros(num_interfaces)
        self.vD0 = -vD0  # I multiplied the drift velocity by -1 because I changed the positive direction of the well from top to bottom.

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
