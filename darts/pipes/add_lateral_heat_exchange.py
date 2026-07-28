import numpy as np

from darts.pipes.define_pipe_geometry import PipeGeometry


class SemiAnalyticalWellLateralHeatTransfer:
    """
    Semi-analytical wellbore-to-earth heat exchange.

    The implementation follows Ramey's notation for the formation time function:

        Ramey Jr., H. J. (1962), "Wellbore Heat Transmission",
        Journal of Petroleum Technology, 14(04), 427-435.

        Chiu, K.-W. and Thakur, S. C. (1991), "Modeling of Wellbore
        Heat Losses in Directional Wells Under Changing Injection Conditions",
        SPE 22870.

        Zhang, Y., Pan, L., Pruess, K., and Finsterle, S. (2011),
        "A Time-Convolution Approach for Modeling Heat Exchange Between a
        Wellbore and Surrounding Formation", Geothermics, 40(4), 261-266.

    In Ramey's Appendix, the transient radial conduction from the wellbore outer
    boundary to the undisturbed earth is written as

        dq = 2*pi*K_earth*(T_boundary - T_earth)*dL / f(t)

    where f(t) is a dimensionless formation resistance. For long times Ramey gives

        f(t) = -ln(r_h / (2*sqrt(alpha*t))) - 0.29

    Ramey's line-source expression is a long-time asymptotic result. Ramey
    states that the convergence time is on the order of one week for many
    reservoir problems and that the line-source result is useful for times
    greater than one week. Chiu and Thakur also note that this Ramey expression
    fails for times less than about seven days. For shorter simulated times,
    use Chiu and Thakur's empirical time function or a numerical surrounding
    grid instead of the Ramey option.

    Chiu and Thakur proposed an empirical time function that matches the exact
    finite-radius formation solution well at early and late times:

        f(t) = 0.982*ln(1 + 1.81*sqrt(alpha*t)/r_h)

    The Zhang option follows the finite-radius formation time function used by
    the OGS WellboreSimulator:

        beta = (pi*t_d)^(-1/2) + 1/2
               - (1/4)*sqrt(t_d/pi) + t_d/8,             t_d < 2.8

        beta = 2*[1/(ln(4*t_d) - 2*gamma)
                  - gamma/(ln(4*t_d) - 2*gamma)^2],      t_d >= 2.8

        t_d = alpha*t/r_h^2

    It is represented in the common resistance form by f(t) = 1/beta. This
    reproduces the instantaneous OGS heat-rate expression, but not Zhang et
    al.'s full time-convolution superposition for a changing boundary
    temperature.

    This implementation does not implement Chiu and Thakur's full WHAP model,
    a superposition treatment for changing injection conditions, a
    pressure-drop model, or Willhite U calculation.

    In all time-function options, r_h is the hole or outer-boundary radius used
    for the formation solution and is set from outermost_layer_OD/2. Ramey's main
    result also includes the wellbore thermal resistance through an overall
    heat-transfer coefficient U between the fluid and the outer boundary.
    Eliminating the unknown outer-boundary temperature gives the fluid-to-earth
    form used here:

        q = 2*pi*K_earth*L*(T_earth - T_fluid)
            / (f(t) + K_earth/(r_U*U))

    where r_U is the radius on which U is based. For the constant-Ui branch,
    r_U is pipe_geometry.pipe_IR, so Ui must be based on the inside pipe area.
    If U tends to infinity, the formula reduces to the pure formation-conduction
    expression with only f(t) in the denominator.
    """

    def __init__(
        self,
        pipe_name: str,
        pipe_geometry: PipeGeometry,
        earth_thermal_props: dict,
        outermost_layer_OD: float,
        Ui: float = None,
        perforated_segments: list = None,
        well_layers_props: dict = None,
        time_function_name: str = "Chiu&Thakur",
        verbose: bool = False,
    ):
        """
        This class defines lateral heat transfer between the wellbore the geometry of which is entered as the first
        input argument of the constructor and the surrounding rock/soil using a semi-analytical lateral heat
        transfer model.
        Note that from the input args "U" and "well_layers_props", only one must be specified.

        :param pipe_name: Name of the pipe (well) for which SemiAnalyticalWellLateralHeatTransfer is added.
        :type pipe_name: str
        :param pipe_geometry: The geometry of the pipe (well) for which lateral heat transfer is intended to be defined
        :type pipe_geometry: PipeGeometry
        :param earth_thermal_props: A dictionary containing earth thermal properties including these keys:
        "T": Earth temperature with the number of elements equal to the number of segments of the wellbore (list)
        "c": Earth specific heat capacity (float or list)
        "K": Earth thermal conductivity (float or list)
        "rho": Earth density (float or list)
        :type earth_thermal_props: dict
        :param outermost_layer_OD: The outside diameter of the outermost layer of the wellbore before the formation, so
        it could be a casing, a cement sheath, etc.
        :type outermost_layer_OD: float
        :param perforated_segments: The list of the indices of the segments which are perforated. If not specified,
        it is assumed that the pipe has no perforated segments.
        :type perforated_segments: list
        :param Ui: Overall heat transfer coefficient based on the inner pipe diameter. If Ui is not specified,
        well_layers_props must be specified.
        :type Ui: float
        :param well_layers_props: The properties of the layers surrounding the fluid in the wellbore to thermal
        calculations. If well_layers_props is not specified, Ui must be specified.
        :type well_layers_props: dict
        :param time_function_name: The name of the time function used for transient calculation of heat transfer.
        Available options are "Ramey", "Chiu&Thakur", and "Zhang". Default is "Chiu&Thakur".
        "Ramey" is a long-time asymptotic expression and should not be used for
        simulated times shorter than about seven days.
        "Zhang" is the finite-radius Carslaw-Jaeger response used by the OGS
        WellboreSimulator.
        :type time_function_name: str
        :param verbose: Whether to display extra info about SemiAnalyticalWellLateralHeatTransfer
        :type verbose: boolean
        """
        assert pipe_geometry.pipe_name == pipe_name, (
            "The names of the pipes in PipeGeometry and SemiAnalyticalWellLateralHeatTransfer are not identical!"
        )
        self.well_name = pipe_name

        T_earth = earth_thermal_props["T"]
        assert len(T_earth) == pipe_geometry.num_segments
        self.T_earth = np.array(T_earth)

        c_earth = earth_thermal_props["c"]
        if isinstance(c_earth, float) or isinstance(c_earth, int):
            c_earth = [c_earth] * pipe_geometry.num_segments
        self.c_earth = np.array(c_earth)

        K_earth = earth_thermal_props["K"]
        if isinstance(K_earth, float) or isinstance(K_earth, int):
            K_earth = [K_earth] * pipe_geometry.num_segments
        self.K_earth = np.array(K_earth)

        rho_earth = earth_thermal_props["rho"]
        if isinstance(rho_earth, float) or isinstance(rho_earth, int):
            rho_earth = [rho_earth] * pipe_geometry.num_segments
        self.rho_earth = np.array(rho_earth)

        # Calculate earth thermal diffusivity
        self.alpha = self.K_earth / (self.rho_earth * self.c_earth)

        if well_layers_props is not None:
            # The ID of the smallest pipe specified in well_layers_props
            # must be the same value as the pip_IR in the class PipeGeometry
            self.well_layers_props = well_layers_props
            # Calculate U

        self.tubing_IR = pipe_geometry.pipe_IR
        self.Ui = Ui

        self.time_function_name = time_function_name
        self.outermost_layer_OD = outermost_layer_OD

        self.perforated_segments = (
            [] if perforated_segments is None else perforated_segments
        )
        assert isinstance(self.perforated_segments, list), (
            "perforated_segments must be a list!"
        )
        assert all(
            [
                0 <= perf_idx < pipe_geometry.num_segments
                for perf_idx in self.perforated_segments
            ]
        ), "Indices of perforated segments must be valid well segment indices!"

        self.segment_lengths = pipe_geometry.segment_lengths

        self.q_lateral_heat = []

        if verbose:
            print(
                f'** SemiAnalyticalWellLateralHeatTransfer for the well "{pipe_name}" is added!'
            )

    def evaluate(self, T_segments, simulation_timer):
        """
        :param T_segments: Fluid temperature inside the segment [Kelvin]
        :param simulation_timer: Simulation timer [day]
        :return Lateral heat rate [kJ/day]
        """
        simulation_timer = simulation_timer * 24 * 60 * 60
        # Time function evaluation
        if self.time_function_name == "Ramey":
            # Ramey's long-time asymptotic expression is not suitable before about seven days.
            f_t = -np.log(
                (self.outermost_layer_OD / 2)
                / (2 * np.sqrt(self.alpha * simulation_timer))
            )
            f_t -= 0.29
        elif self.time_function_name == "Chiu&Thakur":
            # Chiu and Thakur time function: Provides a reasonable approximation of transient wellbore-formation heat
            # exchange while avoiding the early time discontinuity that results from using Ramey's time function.
            f_t = 0.982 * np.log(
                1
                + 1.81
                * np.sqrt(self.alpha * simulation_timer)
                / (self.outermost_layer_OD / 2)
            )
        elif self.time_function_name == "Zhang":
            t_d = self.alpha * simulation_timer / (self.outermost_layer_OD / 2) ** 2
            beta = np.empty_like(t_d)
            early_time = t_d < 2.8
            beta[early_time] = (
                np.power(np.pi * t_d[early_time], -0.5)
                + 0.5
                - 0.25 * np.sqrt(t_d[early_time] / np.pi)
                + 0.125 * t_d[early_time]
            )
            log_term = np.log(4 * t_d[~early_time]) - 2 * 0.57722
            beta[~early_time] = 2 * (1 / log_term - 0.57722 / np.square(log_term))
            f_t = 1 / beta
        else:
            raise TypeError(
                "Unrecognized time function name " + self.time_function_name
            )

        # Lateral heat rate evaluation
        if self.Ui is not None:
            # For constant overall heat transfer coefficient
            self.q_lateral_heat = (
                2
                * np.pi
                * self.K_earth
                * self.segment_lengths
                * (self.T_earth - T_segments)
                / (f_t + self.K_earth / (self.tubing_IR * self.Ui))
            )
        elif self.well_layers_props is not None:
            # Calculate the overall heat transfer coefficient using Willhite's formula
            U_to = "Willhite's formula"
            r_to = "tubing_outside_radius"
            self.q_lateral_heat = (
                2
                * np.pi
                * self.K_earth
                * self.segment_lengths
                * (self.T_earth - T_segments)
                / (f_t + self.K_earth / (r_to * U_to))
            )

        # Set the lateral heat rate of the perforated well segments to zero
        self.q_lateral_heat[self.perforated_segments] = 0

        return (
            self.q_lateral_heat * 24 * 60 * 60 / 1000
        )  # Multiplying the heat rate by 24 * 60 * 60 / 1000 converts the unit from Joule/second to kJ/day


class SemiAnalyticalWellLateralHeatTransferHook:
    """
    Hook that connects a SemiAnalyticalWellLateralHeatTransfer evaluator to the Newton solver.

    At each Newton iteration, it reads the current segment temperatures from the engine
    state vector, evaluates the lateral heat rates, and subtracts them from the energy
    equation entries of the RHS vector.

    Create an instance in set_wells() and register it by appending to model.rhs_flux_hooks::

        lateral_heat_ev = SemiAnalyticalWellLateralHeatTransfer(...)
        self.rhs_flux_hooks.append(
            SemiAnalyticalWellLateralHeatTransferHook(self, self.reservoir.get_well(well_name), lateral_heat_ev)
        )
    """

    def __init__(self, model, well, lateral_heat_ev):
        self.model = model
        self.well = well
        self.lateral_heat_ev = lateral_heat_ev

    def apply(self, dt: float, t: float):
        physics = self.model.physics
        well = self.well
        n_vars = physics.n_vars
        X = np.asarray(physics.engine.X)
        X_well = X[
            well.well_head_idx * n_vars : (well.well_head_idx + well.num_segments)
            * n_vars
        ].reshape(well.num_segments, n_vars)

        if physics.state_spec == physics.StateSpecification.PT:
            T_segments = X_well[:, -1]
        elif physics.state_spec == physics.StateSpecification.PH:
            T_segments = np.zeros(well.num_segments)
            for i in range(well.num_segments):
                physics.property_containers[0].evaluate(X_well[i])
                T_segments[i] = physics.property_containers[0].temperature
        else:
            raise NotImplementedError(
                f"SemiAnalyticalWellLateralHeatTransferHook does not support state_spec={physics.state_spec!r}."
            )

        lateral_heat_rate = self.lateral_heat_ev.evaluate(T_segments, t + dt)
        rhs = np.asarray(physics.engine.RHS)
        rhs_well = rhs[
            well.well_head_idx * n_vars : (well.well_head_idx + well.num_segments)
            * n_vars
        ].reshape(well.num_segments, n_vars)
        rhs_well[:, -1] -= lateral_heat_rate * dt


def add_numerical_well_lateral_heat_transfer(
    well_name: str,
    well_geometry: PipeGeometry,
    reservoir,
    well_wall_cells_idx: np.ndarray,
    well_wall_thickness: float,
    verbose: bool = False,
    geometry_approximation: str = "linear",
):
    """
    This function adds lateral heat transfer between the wellbore the geometry of which is entered as the second
    input argument of the function and the surrounding well (and beyond) using a numerical lateral heat
    transfer model.

    :param well_name: Name of the well for which numerical lateral heat transfer is intended to be added
    :type well_name: str
    :param well_geometry: The geometry of the well for which lateral heat transfer is intended to be added
    :type well_geometry: PipeGeometry
    :param well_wall_cells_idx: Indices of all the cells of the well wall. The number of these indices must be equal
    to the number of the well segments. These indices need to be retrieved from reservoir cell indices.
    :type well_wall_cells_idx: np.ndarray of integers
    :param well_wall_thickness: Thickness of the well wall [meters]
    :type well_wall_thickness: float
    :param verbose: Whether to display extra info about the function
    :type verbose: boolean
    :param geometry_approximation: Approximation used for the well-to-wall-cell geometric coefficient.
                                   "linear" uses A / (wall_thickness / 2).
                                   "radial" uses the cylindrical logarithmic shape factor
                                   2*pi*segment_length / ln((r_w + wall_thickness / 2) / r_w).
    :type geometry_approximation: str
    """
    assert well_geometry.pipe_name == well_name, (
        "The names of the wells in PipeGeometry and add_numerical_lateral_heat_transfer are not identical!"
    )

    assert isinstance(well_wall_cells_idx, np.ndarray), (
        "well_wall_cells_idx must be a numpy array!"
    )
    assert len(well_wall_cells_idx) == well_geometry.num_segments, (
        "The number of well_wall_cells_idx must be equal to the number of the well segments!"
    )

    assert isinstance(well_geometry.pipe_IR, float), (
        "Well radius must be a float; otherwise, it's not supported!"
    )
    assert isinstance(well_wall_thickness, float), (
        "Well wall thickness must be a float; otherwise, it's not supported!"
    )
    assert well_wall_thickness > 0, "Well wall thickness must be positive!"

    assert geometry_approximation in ("linear", "radial"), (
        "geometry_approximation must be either 'linear' or 'radial'!"
    )

    if geometry_approximation == "linear":
        # Thermal transmissibility simply equals geom_coef = A / L
        well_perimeter = 2 * np.pi * well_geometry.pipe_IR
        A = well_perimeter * well_geometry.segment_lengths
        L = well_wall_thickness / 2
        geom_coef = A / L
    else:
        r_w = well_geometry.pipe_IR
        r_wall_cell_center = r_w + well_wall_thickness / 2
        geom_coef = (
            2 * np.pi * well_geometry.segment_lengths / np.log(r_wall_cell_center / r_w)
        )
    well_indexD = geom_coef

    well_indexD *= 2  # Because in cpp code, (gamma_t_i + gamma_t_j) is divided by 2, but we don't want this division
    # in lateral heat exchange because either gamma_t_i or gamma_t_j is always zero since T is at well-pipe interface.

    cpp_well = reservoir.get_well(well_name)

    cpp_well.with_lateral_heat_transfer = True

    well_segments_idx = np.arange(0, cpp_well.num_segments)

    lateral_heat_connections = []
    for idx, segment_idx in enumerate(well_segments_idx):
        lateral_heat_connections = lateral_heat_connections + [
            (segment_idx, well_wall_cells_idx[idx], well_indexD[idx])
        ]

    cpp_well.connections_for_lateral_heat_transfer = lateral_heat_connections

    if verbose:
        print(
            f'** Numerical lateral heat transfer for the well "{well_name}" is added!'
        )

    return
