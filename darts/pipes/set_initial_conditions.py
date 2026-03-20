import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.units import *

g = 9.80665 * meter() / second() ** 2  # Gravitational acceleration


class SingleAmbientTemperature:
    """
    This class is used when the ambient temperature along the pipe is a single value, and so the temperature of
    the fluid in the pipe does not change along the pipe.
    """

    def __init__(
        self,
        pipe_name: str,
        pipe_geom: PipeGeometry,
        physics,
        ambient_temperature: float,
        pipe_head_pressure: float,
        pipe_head_segment_index: int,
        initial_conditions_dict: dict,
        verbose: bool = False,
    ):
        """
        :param pipe_name: Name of the pipe for which the initial conditions are going to be set
        :type pipe_name: str
        :param pipe_geom: Pipe geometry object
        :type pipe_geom: PipeGeometry
        :param physics: physics object is used for density calculations, etc.
        :param ambient_temperature: The temperature of the fluid surrounding the pipe
        :type ambient_temperature: float
        :param pipe_head_pressure: Pressure at one of the heads of the pipe the index of which is specified in pip_head_segment_index [bar]
        :type pipe_head_pressure: float
        :param pipe_head_segment_index: The index of the segment of the pipe at which the pipe head pressure is specified. The index starts from 0.
        :type pipe_head_segment_index: int
        :param initial_conditions_dict: Initial fluid conditions in the pipe including the names of the phases present
        in the pipe, the composition of the phases [mole fractions], and the true vertical depth (TVD) intervals of the
        pipe in which those phases are present [meter].
        :type initial_conditions_dict: dict consisting three key-value pairs: list of strings, list of lists, list of lists
        :param verbose: Whether to display extra info about SingleAmbientTemperature
        :type verbose: boolean
        :return: Initial pressure and temperature profile along the pipe
        """
        assert pipe_name == pipe_geom.pipe_name, (
            "Pipe names for PipeGeometry and SingleAmbientTemperature are not identical!"
        )
        self.pipe_name = pipe_name
        self.pipe_geom = pipe_geom
        self.physics = physics
        self.ambient_temperature = ambient_temperature
        self.pipe_head_pressure = pipe_head_pressure
        assert pipe_head_segment_index in (
            0,
            self.pipe_geom.num_segments - 1,
        ), (
            f"The specified pipe_head_segment_index is neither 0 nor {self.pipe_geom.num_segments - 1}"
        )
        self.pipe_head_segment_index = pipe_head_segment_index

        self.check_initial_fluid_conditions(initial_conditions_dict)
        self.initial_conditions_dict = initial_conditions_dict

        # Get initial conditions
        self.get_initial_temperature_profile()
        self.get_initial_pressure_profile()

        self.initial_conditions_vector = np.zeros(
            self.physics.n_vars * self.pipe_geom.num_segments
        )
        self.assemble_initial_conditions_vector()

        if verbose:
            print(
                f'** Initial conditions (SingleAmbientTemperature) of the pipe "{pipe_name}" are set!'
            )

    def check_initial_fluid_conditions(self, initial_conditions_dict):
        for phase_composition in initial_conditions_dict["phases_compositions"]:
            assert np.isclose(sum(phase_composition), 1, atol=1e-12, rtol=1e-12), (
                "Summation of initial fluid mole fractions must be equal to 1!"
            )
            assert len(phase_composition) == self.physics.property_containers[0].nc, (
                "Number of specified initial fluid mole fractions must be equal to the number of components in the fluid!"
            )

        num_phase_compositions = len(initial_conditions_dict["phases_compositions"])
        num_phase_names = len(initial_conditions_dict["phases_names"])
        num_pipe_intervals = len(initial_conditions_dict["pipe_intervals"])

        assert num_pipe_intervals == num_phase_names == num_phase_compositions, (
            "Number of the specified pipe intervals and their corresponding fluid properties must be equal!"
        )

    def get_initial_temperature_profile(self):
        num_segments = self.pipe_geom.num_segments
        self.temp_init_segments = self.ambient_temperature * np.ones(num_segments)

    def get_initial_pressure_profile(self):
        def dpdz(TVD, p):
            for i, interval in enumerate(
                self.initial_conditions_dict["pipe_intervals"]
            ):
                if interval[0] <= TVD <= interval[1]:
                    phase_name = self.initial_conditions_dict["phases_names"][i]
                    initial_phase_composition = self.initial_conditions_dict[
                        "phases_compositions"
                    ][i]

            density = (
                self.physics.property_containers[0]
                .density_ev[phase_name]
                .evaluate(p[0], temp, initial_phase_composition)
            )

            return g * density * 1e-5  # Convert Pascal to bar

        p_head1 = self.pipe_head_pressure  # Initial solution for the ODE
        if (
            self.pipe_head_segment_index == 0
        ):  # This is used when the pipe-head pressure is the pressure of the top head
            TVD_head1 = self.pipe_geom.TVD_segments[
                0
            ]  # TVD of the initial solution p_head1
            TVD_head2 = self.pipe_geom.TVD_segments[-1]
            TVD_seg_interfaces = self.pipe_geom.TVD_seg_interfaces

            temp = self.ambient_temperature
            # Seg and face together
            sol_seg_interfaces = solve_ivp(
                dpdz, [TVD_head1, TVD_head2], [p_head1], t_eval=TVD_seg_interfaces
            )
            p_seg_interfaces = sol_seg_interfaces.y[0]

        elif (
            self.pipe_head_segment_index == self.pipe_geom.num_segments - 1
        ):  # This is used when the pipe-head pressure is the pressure of the bottom head
            TVD_head1 = self.pipe_geom.TVD_segments[
                -1
            ]  # TVD of the initial solution (p_head1)
            TVD_head2 = self.pipe_geom.TVD_segments[0]
            TVD_seg_interfaces = self.pipe_geom.TVD_seg_interfaces[::-1]

            temp = self.ambient_temperature
            # Seg and face together
            sol_seg_interfaces = solve_ivp(
                dpdz, [TVD_head1, TVD_head2], [p_head1], t_eval=TVD_seg_interfaces
            )
            p_seg_interfaces = sol_seg_interfaces.y[0]

            p_seg_interfaces = p_seg_interfaces[::-1]

        self.p_init_segments = p_seg_interfaces[0::2]
        # Pressures at interfaces are calculated. Maybe, they'll be used later.
        _p_init_interfaces = p_seg_interfaces[1::2]

    def assemble_initial_conditions_vector(self):
        self.initial_conditions_vector[0 :: self.physics.n_vars] = self.p_init_segments
        for var_idx in range(self.physics.nc - 1):
            for segment_idx in range(self.pipe_geom.num_segments):
                for interval_idx, pipe_interval in enumerate(
                    self.initial_conditions_dict["pipe_intervals"]
                ):
                    if (
                        pipe_interval[0]
                        <= self.pipe_geom.TVD_segments[segment_idx]
                        <= pipe_interval[1]
                    ):
                        self.initial_conditions_vector[
                            self.physics.n_vars * segment_idx + var_idx + 1
                        ] = self.initial_conditions_dict["phases_compositions"][
                            interval_idx
                        ][var_idx]

        if self.physics.thermal:
            self.initial_conditions_vector[
                self.physics.n_vars - 1 :: self.physics.n_vars
            ] = self.temp_init_segments

            if (
                self.physics.state_spec == self.physics.StateSpecification.PH
            ):  # replace T with H in the vector
                for seg_idx in range(self.pipe_geom.num_segments):
                    seg_state = self.initial_conditions_vector[
                        seg_idx * self.physics.n_vars : seg_idx * self.physics.n_vars
                        + self.physics.n_vars
                    ]
                    enth = self.physics.property_containers[0].compute_total_enthalpy(
                        seg_state
                    )
                    self.initial_conditions_vector[
                        seg_idx * self.physics.n_vars + self.physics.n_vars - 1
                    ] = enth


class LinearAmbientTemperature:
    """
    This class is used when the ambient temperature along the pipe changes linearly, and so the temperature of
    the fluid in the pipe changes linearly.
    """

    def __init__(
        self,
        pipe_name: str,
        pipe_geom: PipeGeometry,
        physics,
        pipe_head_pressure: float,
        pipe_head_temperature: float,
        temp_grad: float,
        pipe_head_segment_index: int,
        initial_conditions_dict: dict,
        verbose: bool = False,
    ):
        """
        :param pipe_name: Name of the pipe for which the initial conditions are going to be set
        :type pipe_name: str
        :param pipe_geom: Pipe geometry object
        :type pipe_geom: PipeGeometry
        :param physics: physics object is used for density calculations
        :param pipe_head_pressure: Pressure at one of the heads of the pipe the index of which is specified in pipe_head_segment_index [bar]
        :type pipe_head_pressure: float
        :param pipe_head_temperature: Temperature at one of the heads of the pipe the index of which is specified in pipe_head_segment_index [Kelvin]
        :type pipe_head_temperature: float
        :param temp_grad: Temperature gradient [Kelvin/meter]
        :type temp_grad: float
        :param pipe_head_segment_index: The index of the segment of the pipe at which the pipe head pressure is specified. The index starts from 0.
        :type pipe_head_segment_index: int
        :param initial_conditions_dict: Initial fluid conditions in the pipe including the names of the phases present
        in the pipe, the composition of the phases [mole fractions], and the true vertical depth (TVD) intervals of the
        pipe in which those phases are present [meter].
        :type initial_conditions_dict: dict consisting three key-value pairs: list of strings, list of lists, list of lists
        :param verbose: Whether to display extra info about LinearAmbientTemperature
        :type verbose: boolean
        :return: Initial pressure and temperature profile along the pipe
        """
        assert pipe_name == pipe_geom.pipe_name, (
            "Pipe names for PipeGeometry and LinearAmbientTemperature are not identical!"
        )
        self.pipe_name = pipe_name
        self.pipe_geom = pipe_geom
        self.physics = physics
        self.pipe_head_pressure = pipe_head_pressure
        self.pipe_head_temperature = pipe_head_temperature
        self.temp_grad = temp_grad
        assert pipe_head_segment_index in (
            0,
            self.pipe_geom.num_segments - 1,
        ), (
            f"The specified pipe_head_segment_index is neither 0 nor {self.pipe_geom.num_segments - 1}"
        )
        self.pipe_head_segment_index = pipe_head_segment_index

        self.check_initial_fluid_conditions(initial_conditions_dict)
        self.initial_conditions_dict = initial_conditions_dict

        # Get initial conditions
        self.get_initial_temperature_profile()
        self.get_initial_pressure_profile()

        self.initial_conditions_vector = np.zeros(
            self.physics.n_vars * self.pipe_geom.num_segments
        )
        self.assemble_initial_conditions_vector()

        if verbose:
            print(
                f'** Initial conditions (LinearAmbientTemperature) of the pipe "{pipe_name}" are set!'
            )

    def check_initial_fluid_conditions(self, initial_conditions_dict):
        for phase_composition in initial_conditions_dict["phases_compositions"]:
            assert np.isclose(sum(phase_composition), 1, atol=1e-12, rtol=1e-12), (
                "Summation of initial fluid mole fractions must be equal to 1!"
            )
            assert len(phase_composition) == self.physics.property_containers[0].nc, (
                "Number of specified initial fluid mole fractions must be equal to the number of components in the fluid!"
            )

        num_phase_compositions = len(initial_conditions_dict["phases_compositions"])
        num_phase_names = len(initial_conditions_dict["phases_names"])
        num_pipe_intervals = len(initial_conditions_dict["pipe_intervals"])

        assert num_pipe_intervals == num_phase_names == num_phase_compositions, (
            "Number of the specified pipe intervals and their corresponding fluid properties must be equal!"
        )

    def get_initial_temperature_profile(self):
        print(
            "Pipe head temperature is assumed to be the lowest temperature for the initial temperature calculation. "
            "If it's the opposite, change the sign of the temperature gradient."
        )
        self.temp_init_segments = np.array(
            self.pipe_head_temperature
            + self.temp_grad
            * (self.pipe_geom.TVD_segments - self.pipe_geom.TVD_segments[0])
        )
        self.temp_init_interfaces = np.array(
            self.pipe_head_temperature
            + self.temp_grad
            * (self.pipe_geom.TVD_interfaces - self.pipe_geom.TVD_segments[0])
        )  # Temperatures at interfaces are calculated even though they're not used in any part of the code.

        temp_init_seg_interfaces = np.zeros(
            self.pipe_geom.num_segments + self.pipe_geom.num_interfaces
        )
        temp_init_seg_interfaces[0::2] = self.temp_init_segments
        temp_init_seg_interfaces[1::2] = self.temp_init_interfaces
        self.temp_init_seg_interfaces = temp_init_seg_interfaces  # This is used to calculate pressures at segments and interfaces together even though the pressure values at interfaces are not used in any part of the code, but this variable is used for calculating initial pressure profile along the wellbore more easily.

    def get_initial_pressure_profile(self):
        def dpdz(TVD, p):
            for i, interval in enumerate(
                self.initial_conditions_dict["pipe_intervals"]
            ):
                if interval[0] <= TVD <= interval[1]:
                    temp = temp_func(TVD)
                    phase_name = self.initial_conditions_dict["phases_names"][i]
                    initial_phase_composition = self.initial_conditions_dict[
                        "phases_compositions"
                    ][i]

            density = (
                self.physics.property_containers[0]
                .density_ev[phase_name]
                .evaluate(p[0], temp, initial_phase_composition)
            )

            return g * density * 1e-5  # Convert Pas to bar

        p_head1 = self.pipe_head_pressure  # Initial solution for the ODE
        if self.pipe_head_segment_index == 0:
            TVD_head1 = self.pipe_geom.TVD_segments[
                0
            ]  # TVD of the initial solution p_head1
            TVD_head2 = self.pipe_geom.TVD_segments[-1]
            TVD_seg_interfaces = self.pipe_geom.TVD_seg_interfaces

            temp_func = interp1d(
                TVD_seg_interfaces,
                self.temp_init_seg_interfaces,
                fill_value="extrapolate",
            )

            # Seg and face together
            sol_seg_interfaces = solve_ivp(
                dpdz, [TVD_head1, TVD_head2], [p_head1], t_eval=TVD_seg_interfaces
            )
            p_seg_interfaces = sol_seg_interfaces.y[0]

        elif self.pipe_head_segment_index == self.pipe_geom.num_segments - 1:
            TVD_head1 = self.pipe_geom.TVD_segments[-1]
            TVD_head2 = self.pipe_geom.TVD_segments[
                0
            ]  # TVD of the initial solution (p_head1)
            TVD_seg_interfaces = self.pipe_geom.TVD_seg_interfaces[::-1]

            temp_init_seg_interfaces = self.temp_init_seg_interfaces[::-1]
            temp_func = interp1d(
                TVD_seg_interfaces, temp_init_seg_interfaces, fill_value="extrapolate"
            )

            # Seg and face together
            sol_seg_interfaces = solve_ivp(
                dpdz, [TVD_head1, TVD_head2], [p_head1], t_eval=TVD_seg_interfaces
            )
            p_seg_interfaces = sol_seg_interfaces.y[0]

            p_seg_interfaces = p_seg_interfaces[::-1]

        self.p_init_segments = p_seg_interfaces[0::2]
        # Pressures at interfaces are calculated. Maybe, they'll be used later.
        _p_init_interfaces = p_seg_interfaces[1::2]

    def assemble_initial_conditions_vector(self):
        self.initial_conditions_vector[0 :: self.physics.n_vars] = self.p_init_segments
        for var_idx in range(self.physics.nc - 1):
            for segment_idx in range(self.pipe_geom.num_segments):
                for interval_idx, pipe_interval in enumerate(
                    self.initial_conditions_dict["pipe_intervals"]
                ):
                    if (
                        pipe_interval[0]
                        <= self.pipe_geom.TVD_segments[segment_idx]
                        <= pipe_interval[1]
                    ):
                        self.initial_conditions_vector[
                            self.physics.n_vars * segment_idx + var_idx + 1
                        ] = self.initial_conditions_dict["phases_compositions"][
                            interval_idx
                        ][var_idx]

        if self.physics.thermal:
            self.initial_conditions_vector[
                self.physics.n_vars - 1 :: self.physics.n_vars
            ] = self.temp_init_segments

            if (
                self.physics.state_spec == self.physics.StateSpecification.PH
            ):  # replace T with H in the vector
                for seg_idx in range(self.pipe_geom.num_segments):
                    seg_state = self.initial_conditions_vector[
                        seg_idx * self.physics.n_vars : seg_idx * self.physics.n_vars
                        + self.physics.n_vars
                    ]
                    enth = self.physics.property_containers[0].compute_total_enthalpy(
                        seg_state
                    )
                    self.initial_conditions_vector[
                        seg_idx * self.physics.n_vars + self.physics.n_vars - 1
                    ] = enth
