"""
Differences between this script and the DFM velocity evaluator in the standalone well model:
- https://gitlab.com/open-darts/open-darts/-/commit/b0aa26cb9beb90a10bb1b4e3db5291399607f46d
Revert the density averaging method here (now it is similar to DWell):
- https://gitlab.com/open-darts/open-darts/-/commit/0c584d57e8cd20c270057763f3b6bc916cbc695e

Notes:
    - The kinetic energy is not added to the energy conservation equation of the coupled model yet, while it was in the
    standalone wellbore model.
"""

import math
from dataclasses import dataclass

from scipy.optimize import fsolve

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


class Pipe:
    g = 9.80665 * meter() / second() ** 2  # Gravitational acceleration
    Cku = 142
    Cw = 0.008

    adjustment_func_params = AdjustmentFuncParams

    def __init__(
        self,
        pipe_name: str,
        pipe_geometry: PipeGeometry,
        physics,
        reservoir,
        initial_conditions: SingleAmbientTemperature | LinearAmbientTemperature,
        source_sinks: dict = None,
        Cmax: float = 1.2,
        Fv: float = 1,
        diff_method: str = "OBL",
        eps_p: float = 1e-4,
        eps_temp: float = 0.1,
        eps_z: float = 0.00001,
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
        :param Cmax: A user-specified maximum profile parameter that can be tuned to match the observations and
        could have a value between 1.0 and 1.5. It is set to:
        --> 1.2 in ECLIPSE according to Shi et al. paper (Drift-Flux Modeling of Two-Phase Flow in Wellbores)
        --> 1 in a wellbore simulator in Tonken et al. paper (A transient geothermal wellbore simulator)
        :type Cmax: float
        :param Fv: A multiplier on the flooding velocity fraction, set to be 1 by default, and its value can be tuned
        to fit the observations.
        :type Fv: float
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
        :type verbose: boolean
        """
        assert pipe_name == pipe_geometry.pipe_name, (
            "Pipe names in pipe_name and pipe_geometry are not identical!"
        )
        self.name = pipe_name
        self.geometry = pipe_geometry
        self.physics = physics
        self.reservoir = reservoir

        self.isothermal = not physics.thermal

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

        if self.isothermal:
            assert self.physics.property_containers[0].temperature is not None, (
                "If model is isothermal, system_temperature must be specified!"
            )
        elif not self.isothermal:
            assert self.physics.property_containers[0].temperature is None, (
                "If model is non-isothermal, system_temperature must not be specified!"
            )
        self.system_temperature = self.physics.property_containers[0].temperature

        self.Cmax = Cmax
        self.B = 2 / Cmax - 1.0667
        self.Fv = Fv

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

        # For phase velocity evaluation
        self.g_cos_theta = self.g * np.cos(pipe_geometry.inclination_angle_radian)
        if isinstance(self.g_cos_theta, float):
            self.g_cos_theta = self.g_cos_theta * np.ones(pipe_geometry.num_interfaces)

        assert isinstance(diff_method, str), (
            "diff_method for pipe velocity differentiation must be a string!"
        )
        if diff_method in ["numerical", "OBL"]:
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

        self.is_first_first_iter = True  # first_iter_in_first_ts_identifier

        self.lateral_heat_rate_eval = None

        if verbose:
            print(f'** Model of the pipe "{self.geometry.pipe_name}" is created!')

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

        num_segments = self.geometry.num_segments
        num_interfaces = self.geometry.num_interfaces
        nc = self.physics.nc
        n_vars = self.physics.n_vars

        pc = self.physics.property_containers[0]

        """ Calculate phase props of previous time step at centroids """
        if iter_counter == 0 and self.is_first_first_iter is True and flag == 1:
            sG0 = np.zeros(num_segments)
            rhoG0 = np.zeros(num_segments)
            rhoL0 = np.zeros(num_segments)
            miuG0 = np.zeros(num_segments)
            miuL0 = np.zeros(num_segments)
            xG_mass0 = np.zeros((num_segments, nc))
            xL_mass0 = np.zeros((num_segments, nc))

            if pc.nph == 3:
                sL_a_0 = np.zeros(num_segments)
                sL_b_0 = np.zeros(num_segments)
                rhoL_a_0 = np.zeros(num_segments)
                rhoL_b_0 = np.zeros(num_segments)
                miuL_a_0 = np.zeros(num_segments)
                miuL_b_0 = np.zeros(num_segments)
                xL_a_mass_0 = np.zeros((num_segments, nc))
                xL_b_mass_0 = np.zeros((num_segments, nc))

            for i in range(num_segments):
                state0 = Xn_dfm_well[i * n_vars : (i + 1) * n_vars]
                pc.evaluate(state0)
                if self.physics.thermal:
                    pc.evaluate_thermal(state0)

                if pc.nph == 2:
                    sG0[i] = pc.sat[0]
                    rhoG0[i], rhoL0[i] = pc.dens[0], pc.dens[1]
                    miuG0[i], miuL0[i] = (
                        pc.mu[0] * 1e-3,
                        pc.mu[1] * 1e-3,
                    )  # convert cP to Pa.s
                    # Calculate mass fractions of components in each phase
                    x_mass0 = np.zeros((pc.nph, nc))
                    for j in pc.ph:
                        x_mass0[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
                    xG_mass0[i, :], xL_mass0[i, :] = x_mass0[0, :], x_mass0[1, :]

                if pc.nph == 3:
                    # sG0[i], sL_a_0[i], sL_b_0[i] = pc.sat[1], pc.sat[0], pc.sat[2]
                    # rhoG0[i], rhoL_a_0[i], rhoL_b_0[i] = pc.dens[1], pc.dens[0], pc.dens[2]
                    # miuG0[i], miuL_a_0[i], miuL_b_0[i] = pc.mu[1] * 1e-3, pc.mu[0] * 1e-3, pc.mu[2] * 1e-3
                    sG0[i], sL_a_0[i], sL_b_0[i] = pc.sat[0], pc.sat[1], pc.sat[2]
                    rhoG0[i], rhoL_a_0[i], rhoL_b_0[i] = (
                        pc.dens[0],
                        pc.dens[1],
                        pc.dens[2],
                    )
                    miuG0[i], miuL_a_0[i], miuL_b_0[i] = (
                        pc.mu[0] * 1e-3,
                        pc.mu[1] * 1e-3,
                        pc.mu[2] * 1e-3,
                    )
                    # Calculate mass fractions of components in each phase
                    x_mass0 = np.zeros((pc.nph, nc))
                    for j in pc.ph:
                        x_mass0[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
                    # xG_mass0[i, :], xL_a_mass_0[i, :], xL_b_mass_0[i, :] = x_mass0[1, :], x_mass0[0, :], x_mass0[2, :]
                    xG_mass0[i, :], xL_a_mass_0[i, :], xL_b_mass_0[i, :] = (
                        x_mass0[0, :],
                        x_mass0[1, :],
                        x_mass0[2, :],
                    )

                    # Calculate averaged liquid props
                    rhoL0[i] = (
                        (rhoL_a_0[i] * sL_a_0[i] + rhoL_b_0[i] * sL_b_0[i])
                        / (sL_a_0[i] + sL_b_0[i])
                        if (sL_a_0[i] + sL_b_0[i]) > 0
                        else 0
                    )
                    miuL0[i] = (
                        (miuL_a_0[i] * sL_a_0[i] + miuL_b_0[i] * sL_b_0[i])
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
                rhoG0,
                rhoL0,
                miuG0,
                miuL0,
            ]

        elif iter_counter == 0 and self.is_first_first_iter is False and flag == 1:
            self.iter_phases_props0 = self.iter_phases_props

        xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, miuG0, miuL0 = self.iter_phases_props0

        """ Calculate phase props of current time step at centroids """
        sG = np.zeros(num_segments)
        rhoG = np.zeros(num_segments)
        rhoL = np.zeros(num_segments)
        miuG = np.zeros(num_segments)
        miuL = np.zeros(num_segments)
        xG_mass = np.zeros((num_segments, nc))
        xL_mass = np.zeros((num_segments, nc))

        if pc.nph == 3:
            sL_a = np.zeros(num_segments)
            sL_b = np.zeros(num_segments)
            rhoL_a = np.zeros(num_segments)
            rhoL_b = np.zeros(num_segments)
            miuL_a = np.zeros(num_segments)
            miuL_b = np.zeros(num_segments)
            xL_a_mass = np.zeros((num_segments, nc))
            xL_b_mass = np.zeros((num_segments, nc))

        for i in range(num_segments):
            state = X_dfm_well[i * n_vars : (i + 1) * n_vars]
            pc.evaluate(state)
            if self.physics.thermal:
                pc.evaluate_thermal(state)

            if pc.nph == 2:
                sG[i] = pc.sat[0]
                rhoG[i], rhoL[i] = pc.dens[0], pc.dens[1]
                miuG[i], miuL[i] = (
                    pc.mu[0] * 1e-3,
                    pc.mu[1] * 1e-3,
                )  # convert cP to Pa.s
                # Calculate mass fractions of components in each phase
                x_mass = np.zeros((pc.nph, nc))
                for j in pc.ph:
                    x_mass[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
                xG_mass[i, :], xL_mass[i, :] = x_mass[0, :], x_mass[1, :]

            if pc.nph == 3:
                # sG[i], sL_a[i], sL_b[i] = pc.sat[1], pc.sat[0], pc.sat[2]
                # rhoG[i], rhoL_a[i], rhoL_b[i] = pc.dens[1], pc.dens[0], pc.dens[2]
                # miuG[i], miuL_a[i], miuL_b[i] = pc.mu[1] * 1e-3, pc.mu[0] * 1e-3, pc.mu[2] * 1e-3
                sG[i], sL_a[i], sL_b[i] = pc.sat[0], pc.sat[1], pc.sat[2]
                rhoG[i], rhoL_a[i], rhoL_b[i] = pc.dens[0], pc.dens[1], pc.dens[2]
                miuG[i], miuL_a[i], miuL_b[i] = (
                    pc.mu[0] * 1e-3,
                    pc.mu[1] * 1e-3,
                    pc.mu[2] * 1e-3,
                )
                # Calculate mass fractions of components in each phase
                x_mass = np.zeros((pc.nph, nc))
                for j in pc.ph:
                    x_mass[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
                # xG_mass[i, :], xL_a_mass[i, :], xL_b_mass[i, :] = x_mass[1, :], x_mass[0, :], x_mass[2, :]
                xG_mass[i, :], xL_a_mass[i, :], xL_b_mass[i, :] = (
                    x_mass[0, :],
                    x_mass[1, :],
                    x_mass[2, :],
                )

                # Calculate averaged liquid props
                rhoL[i] = (
                    (rhoL_a[i] * sL_a[i] + rhoL_b[i] * sL_b[i]) / (sL_a[i] + sL_b[i])
                    if (sL_a[i] + sL_b[i]) > 0
                    else 0
                )
                miuL[i] = (
                    (miuL_a[i] * sL_a[i] + miuL_b[i] * sL_b[i]) / (sL_a[i] + sL_b[i])
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

        self.iter_phases_props = [xG_mass, xL_mass, sG, rhoG, rhoL, miuG, miuL]

        # If the differentiation method is OBL, calculate phase property derivatives
        if self.diff_method == "OBL":
            sG_der = self.get_operator_der_matrix_for_well(
                op_idx=self.physics.reservoir_operators[0].SAT_OP + 0
            )
            rhoG_der = self.get_operator_der_matrix_for_well(
                op_idx=self.physics.reservoir_operators[0].GRAV_OP + 0
            )
            rhoL_der = self.get_operator_der_matrix_for_well(
                op_idx=self.physics.reservoir_operators[0].GRAV_OP + 1
            )
            self.iter_phases_props_der = [sG_der, rhoG_der, rhoL_der]

        """ Calculate phase props of previous time step at interfaces """
        if iter_counter == 0 and flag == 1:
            sG0_face = (sG0[0:-1] + sG0[1:]) / 2

            # Initialize arrays to store interface properties
            rhoG0_face = np.zeros(num_segments - 1)
            rhoL0_face = np.zeros(num_segments - 1)
            miuG0_face = np.zeros(num_segments - 1)
            miuL0_face = np.zeros(num_segments - 1)
            # xG_mass0_face and xL_mass0_face for IFT calculation
            xG_mass0_face = np.zeros((num_segments - 1, nc))
            xL_mass0_face = np.zeros((num_segments - 1, nc))

            # Compute interface values using conditional averaging
            for i in range(num_segments - 1):
                if sG0[i] == 0:
                    # If no gas in segment i, use properties from segment i+1
                    rhoG0_face[i] = rhoG0[i + 1]
                    miuG0_face[i] = miuG0[i + 1]
                    xG_mass0_face[i] = xG_mass0[i + 1]
                elif sG0[i + 1] == 0:
                    # If no gas in segment i+1, use properties from segment i
                    rhoG0_face[i] = rhoG0[i]
                    miuG0_face[i] = miuG0[i]
                    xG_mass0_face[i] = xG_mass0[i]
                else:
                    # If both segments have gas, use averaging
                    rhoG0_face[i] = (rhoG0[i] + rhoG0[i + 1]) / 2
                    miuG0_face[i] = (miuG0[i] + miuG0[i + 1]) / 2

                    xG_mass0_face[i] = (
                        xG_mass0[i] * rhoG0[i] * sG0[i]
                        + xG_mass0[i + 1] * rhoG0[i + 1] * sG0[i + 1]
                    ) / (rhoG0[i] * sG0[i] + rhoG0[i + 1] * sG0[i + 1])

                if sG0[i] == 1:
                    # If no liquid in segment i, use properties from segment i+1
                    rhoL0_face[i] = rhoL0[i + 1]
                    miuL0_face[i] = miuL0[i + 1]
                    xL_mass0_face[i] = xL_mass0[i + 1]
                elif sG0[i + 1] == 1:
                    # If no liquid in segment i+1, use properties from segment i
                    rhoL0_face[i] = rhoL0[i]
                    miuL0_face[i] = miuL0[i]
                    xL_mass0_face[i] = xL_mass0[i]
                else:
                    # If both segments have liquid, use averaging
                    rhoL0_face[i] = (rhoL0[i] + rhoL0[i + 1]) / 2
                    miuL0_face[i] = (miuL0[i] + miuL0[i + 1]) / 2

                    xL_mass0_face[i] = (
                        xL_mass0[i] * rhoL0[i] * (1 - sG0[i])
                        + xL_mass0[i + 1] * rhoL0[i + 1] * (1 - sG0[i + 1])
                    ) / (rhoL0[i] * (1 - sG0[i]) + rhoL0[i + 1] * (1 - sG0[i + 1]))

            self.iter_phases_props0_face = [
                xG_mass0_face,
                xL_mass0_face,
                sG0_face,
                rhoG0_face,
                rhoL0_face,
                miuG0_face,
                miuL0_face,
            ]

        """ Calculate phase props of current time step at interfaces """
        sG_face = (sG[0:-1] + sG[1:]) / 2
        if self.diff_method == "OBL":
            sG_face_der = (sG_der[0:-1, :] + sG_der[1:, :]) / 2

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

            if sG[i] == 1:
                # If no liquid in segment i, use properties from segment i+1
                rhoL_face[i] = rhoL[i + 1]
                if self.diff_method == "OBL":
                    rhoL_face_der[i, :] = rhoL_der[i + 1, :]
            elif sG[i + 1] == 1:
                # If no liquid in segment i+1, use properties from segment i
                rhoL_face[i] = rhoL[i]
                if self.diff_method == "OBL":
                    rhoL_face_der[i, :] = rhoL_der[i, :]
            else:
                # If both segments have liquid, use averaging
                rhoL_face[i] = (rhoL[i] + rhoL[i + 1]) / 2

                if self.diff_method == "OBL":
                    rhoL_face_der[i, :] = (rhoL_der[i, :] + rhoL_der[i + 1, :]) / 2

        self.iter_phases_props_face = [sG_face, rhoG_face, rhoL_face]
        if self.diff_method == "OBL":
            self.iter_phases_props_face_ders = [
                sG_face_der,
                rhoG_face_der,
                rhoL_face_der,
            ]

        if iter_counter == 0 and self.is_first_first_iter is True and flag == 1:
            # Initial velocities in the wellbore are zero
            rhoM0_vM0, vM0, vG0, vL0 = (
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
            )
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])
        elif iter_counter == 0 and self.is_first_first_iter is False and flag == 1:
            rhoM0_vM0, vM0, vG0, vL0 = self.rhoM_vM, self.vM, self.vG, self.vL
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])

        [_, vM0, vG0, vL0] = self.velocities0

        p = X_dfm_well[0::n_vars] * 1e5  # convert bar to Pa
        p_m = p[0:-1:1]
        p_p = p[1::1]
        if self.diff_method == "OBL":
            p_der = (
                self.get_operator_der_matrix_for_well(
                    op_idx=self.physics.reservoir_operators[0].PRES_OP
                )
                * 1e5
            )
            p_m_der = p_der[0:-1:1, :]
            p_p_der = p_der[1::1, :]

        self.calc_mixture_densities(iter_counter, flag)

        pg = self.geometry

        if iter_counter == 0 and flag == 1:
            # To increase the numerical stability, you may need to use an upwind scheme for the momentum flux like
            # in the paper "A transient gothermal wellbore simulator (2023)
            delta_interface0 = pg.pipe_internal_A * (
                rhoG0_face * sG0_face * vG0**2 + rhoL0_face * (1 - sG0_face) * vL0**2
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
                    Mw = self.physics.property_containers[0].Mw
                    # mass_rate in kg/s
                    mass_rate = sum(
                        rate_source * np.array(comp_source) * np.array(Mw)
                    ) / (24 * 60 * 60)
                elif sink_source.inflow_or_outflow == "outflow":
                    # TODO: For outflow, we have rate_source, which is in kmol/day, but we don't have comp_source, which
                    # is in kmol/kmol, from the user. Instead, we have xG_mass0 and xL_mass0, which are mass fractions.
                    # Need to see how we can get the overall composition of the source block in kmol/kmol.
                    mass_rate = 0

                pipe_internal_A = self.geometry.pipe_internal_A

                # The props of the fluid of the segment on which the constant mass rate source is defined are used.
                sG0_source = sG0[segment_idx_source]
                rhoG0_source = rhoG0[segment_idx_source]
                rhoL0_source = rhoL0[segment_idx_source]

                if sG0_source == 0:
                    vG0_source = 0
                    liquid_mass_fraction0 = 1
                    liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
                    vL0_source = (
                        liquid_mass_rate0 / rhoL0_source / (pipe_internal_A * 1)
                    )
                elif sG0_source == 1:
                    vL0_source = 0
                    gas_mass_fraction0 = 1
                    gas_mass_rate0 = mass_rate * gas_mass_fraction0
                    vG0_source = gas_mass_rate0 / rhoG0_source / (pipe_internal_A * 1)
                elif 0 < sG0_source < 1:
                    gas_mass_fraction0 = (
                        sG0_source
                        * rhoG0_source
                        / (sG0_source * rhoG0_source + (1 - sG0_source) * rhoL0_source)
                    )
                    gas_mass_rate0 = mass_rate * gas_mass_fraction0
                    vG0_source = (
                        gas_mass_rate0 / rhoG0_source / (pipe_internal_A * sG0_source)
                    )

                    liquid_mass_fraction0 = 1 - gas_mass_fraction0
                    liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
                    vL0_source = (
                        liquid_mass_rate0
                        / rhoL0_source
                        / (pipe_internal_A * (1 - sG0_source))
                    )
                else:
                    raise Exception("sG0_source is out of correct range (from 0 to 1)!")

                delta_at_bc_interface0 = pipe_internal_A * (
                    rhoG0_source * sG0_source * vG0_source**2
                    + rhoL0_source * (1 - sG0_source) * vL0_source**2
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
                pg.pipe_internal_A * delta_interface0[0:-1] / pg.D[0:-1]
                + pg.pipe_internal_A * delta_interface0[1:] / pg.D[1:]
            ) / (pg.pipe_internal_A / pg.D[0:-1] + pg.pipe_internal_A / pg.D[1:])
            self.delta_m0 = delta_segment0[0:-1]
            self.delta_p0 = delta_segment0[1:]

            ff0 = self.calc_Fanning_friction_factor()
            self.w0 = 1 / (
                1 / dt + pg.perimeter * ff0 * abs(vM0) / (2 * pg.pipe_internal_A)
            )

        self.rhoM_vM = (
            -self.w0 * (p_p - p_m) / (pg.z_p - pg.z_m)
            + self.w0 * self.g_cos_theta * self.rhoM_face
            - self.w0
            * (
                (self.delta_p0 - self.delta_m0)
                / (pg.pipe_internal_A * (pg.z_p - pg.z_m))
                - self.rhoM0_face * vM0 / dt
            )
        )

        if self.diff_method == "OBL":
            self.rhoM_vM_der = (
                -self.w0[:, None]
                * (p_p_der - p_m_der)
                / (pg.z_p[:, None] - pg.z_m[:, None])
                + self.w0[:, None] * self.g_cos_theta[:, None] * self.rhoM_face_der
            )

        self.vM = self.rhoM_vM / self.rhoM_face

        if iter_counter == 0 and flag == 1 and self.is_first_first_iter is True:
            self.vD0 = np.zeros(num_interfaces)
        elif iter_counter == 0 and flag == 1 and self.is_first_first_iter is False:
            self.update_drift_velocity()

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
            if sG_face[i] != 1:
                self.vL[i] = (1 - self.C00[i] * sG_face[i]) * self.rhoM_vM[i] / (
                    (1 - sG_face[i]) * self.rhoM_adjusted_face[i]
                ) - sG_face[i] * rhoG_face[i] * self.vD0[i] / (
                    (1 - sG_face[i]) * self.rhoM_adjusted_face[i]
                )

                if self.diff_method == "OBL":
                    self.vL_der[i, :] = (
                        (
                            -self.C00[i] * sG_face_der[i, :] * self.rhoM_vM[i]
                            + (1 - self.C00[i] * sG_face[i]) * self.rhoM_vM_der[i, :]
                        )
                        * (1 - sG_face[i])
                        * self.rhoM_adjusted_face[i]
                        - (
                            -sG_face_der[i, :] * self.rhoM_adjusted_face[i]
                            + (1 - sG_face[i]) * self.rhoM_adjusted_face_der[i, :]
                        )
                        * (1 - self.C00[i] * sG_face[i])
                        * self.rhoM_vM[i]
                    ) / (((1 - sG_face[i]) * self.rhoM_adjusted_face[i]) ** 2) - (
                        (
                            self.vD0[i]
                            * (
                                sG_face_der[i, :] * rhoG_face[i]
                                + sG_face[i] * rhoG_face_der[i, :]
                            )
                        )
                        * ((1 - sG_face[i]) * self.rhoM_adjusted_face[i])
                        - (
                            -sG_face_der[i, :] * self.rhoM_adjusted_face[i]
                            + (1 - sG_face[i]) * self.rhoM_adjusted_face_der[i, :]
                        )
                        * sG_face[i]
                        * rhoG_face[i]
                        * self.vD0[i]
                    ) / (((1 - sG_face[i]) * self.rhoM_adjusted_face[i]) ** 2)

        for i in range(num_interfaces):
            if self.vG[i] > 0 and sG[i] == 0:
                self.vG[i] = 0
                if self.diff_method == "OBL":
                    self.vG_der[i, :] = 0
            elif self.vG[i] < 0 and sG[i + 1] == 0:
                self.vG[i] = 0
                if self.diff_method == "OBL":
                    self.vG_der[i, :] = 0

            if self.vL[i] > 0 and (sG[i] - 1) == 0:
                self.vL[i] = 0
                if self.diff_method == "OBL":
                    self.vL_der[i, :] = 0
            elif self.vL[i] < 0 and (sG[i + 1] - 1) == 0:
                self.vL[i] = 0
                if self.diff_method == "OBL":
                    self.vL_der[i, :] = 0

        # Concatenate phase velocities and convert m/s to m/day
        phase_velocities = np.concatenate(
            (self.vG * 24 * 60 * 60, self.vL * 24 * 60 * 60)
        )

        if self.diff_method == "OBL":
            self.vG_der *= 24 * 60 * 60
            self.vL_der *= 24 * 60 * 60

        # is_first_first_iter is true only for the first iteration of the first time step.
        self.is_first_first_iter = False

        return phase_velocities

    def calc_mixture_densities(self, iter_counter, flag):
        if iter_counter == 0 and self.is_first_first_iter is True and flag == 1:
            _, _, sG0, rhoG0, rhoL0, _, _ = self.iter_phases_props0

            _, _, sG0_face, rhoG0_face, rhoL0_face, _, _ = self.iter_phases_props0_face
            self.rhoM0_face = sG0_face * rhoG0_face + (1 - sG0_face) * rhoL0_face

        elif iter_counter == 0 and self.is_first_first_iter is False and flag == 1:
            self.rhoM0_face = self.rhoM_face

        # Calculate mixture density
        _, _, sG, rhoG, rhoL, _, _ = self.iter_phases_props

        [sG_face, rhoG_face, rhoL_face] = self.iter_phases_props_face

        self.rhoM_face = sG_face * rhoG_face + (1 - sG_face) * rhoL_face
        if self.diff_method == "OBL":
            sG_face_der, rhoG_face_der, rhoL_face_der = self.iter_phases_props_face_ders
            self.rhoM_face_der = (
                sG_face_der * rhoG_face[:, None]
                + sG_face[:, None] * rhoG_face_der
                - sG_face_der * rhoL_face[:, None]
                + (1 - sG_face[:, None]) * rhoL_face_der
            )

        # Calculate adjusted-mixture density
        if iter_counter == 0 and flag == 1 and self.is_first_first_iter is False:
            self.update_profile_parameter()
        elif iter_counter == 0 and flag == 1 and self.is_first_first_iter is True:
            # At the beginning, there is no flow, so C00 is considered 1 everywhere.
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
        pg = self.geometry

        """ Start calculating the Reynolds number """
        _, _, sG0, _, _, _, _ = self.iter_phases_props0

        _, _, sG0_face, _, _, miuG0_face, miuL0_face = self.iter_phases_props0_face
        [_, vM0, _, _] = self.velocities0

        # Saturation-weighted average is used to calculate the mixture viscosity of two phases. The method is used in
        # Beggs and Brill's book: Eq. 1.38
        # My production engineering notebook: Pressure drop calc in wellbore for 2-phase flow with Beggs and Brill's method
        self.miuM0 = sG0_face * miuG0_face + (1 - sG0_face) * miuL0_face

        # Viscosity averaging method in https://doi.org/10.1016/j.ijmultiphaseflow.2021.103590
        # _, _, _, rhoG0_face, rhoL0_face, _, _ = self.iter_phases_props0_face
        # xg = sG0_face * rhoG0_face / (sG0_face * rhoG0_face + (1 - sG0_face) * rhoL0_face)
        # denominator_1 = xg / miuG0_face
        # denominator_1 = np.nan_to_num(denominator_1, nan=0.0)
        # denominator_2 = (1 - xg) / miuL0_face
        # denominator_2 = np.nan_to_num(denominator_2, nan=0.0)
        # self.miuM0 = 1 / (denominator_1 + denominator_2)

        Re0 = self.calc_Reynolds_number(vM0, self.rhoM0_face, self.miuM0, pg.pipe_ID)
        """ End calculating the Reynolds number """

        self.ff0 = np.zeros(pg.num_interfaces)

        # Laminar connections (Re==0 stays 0)
        lam = (Re0 > 0.0) & (Re0 < 2400.0)
        self.ff0[lam] = 16.0 / Re0[lam]

        # Turbulent connections
        turb_idx = np.nonzero(Re0 > 2400.0)[0]

        initial_guess = 0.005
        relative_roughness = pg.wall_roughness / pg.pipe_ID
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
    def calc_Reynolds_number(v, rho, miu, pipe_ID):
        """
        Calculate the Reynolds number

        :param v: Fluid velocity
        :param rho: Fluid density
        :param miu: Fluid viscosity
        :param pipe_ID: Pipe inside diameter
        """

        Re0 = rho * abs(v) * pipe_ID / miu

        return Re0

    @staticmethod
    def colebrook(f, Re, relative_roughness):
        """
        Calculate the friction factor using the Colebrook-White correlation (implicit method)

        :param f: Guessed Fanning friction factor
        :param Re: Reynolds number
        :param relative_roughness: Pipe relative roughness
        """
        # Ensure the friction factor doesn't go negative or zero
        # Return a large value to prevent sqrt of negative number
        if f <= 0:
            return 1e6

        sqrt_f = math.sqrt(f)
        return 1.0 / sqrt_f + 4.0 * math.log10(
            relative_roughness / 3.7065 + (1.2613 / (Re * sqrt_f))
        )

    def update_profile_parameter(self):
        pg = self.geometry
        num_interfaces = self.geometry.num_interfaces
        [rhoM0_vM0, _, _, _] = self.velocities0
        [xG_mass0_face, xL_mass0_face, sG0_face, rhoG0_face, rhoL0_face, _, _] = (
            self.iter_phases_props0_face
        )

        mask = (sG0_face > 0) & (sG0_face < 1)
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

            # self.Ku0_filtered = np.zeros(len(indices))
            # self.vC0_filtered = np.zeros(len(indices))

            # Calculate C00 from the solution of the previous time step
            vM0 = rhoM0_vM0_filtered / rhoM0_face_filtered
            NB0 = (pg.pipe_ID**2) * (
                self.g * (rhoL0_face_filtered - rhoG0_face_filtered) / IFT0_face
            )
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
            beta0 = np.maximum(
                sG0_face_filtered, self.Fv * sG0_face_filtered * abs(vM0) / v_sgf0
            )
            beta0 = np.clip(beta0, 0, 1)  # beta0 is subject to limits 0 <= beta0 <= 1
            eta0 = (beta0 - self.B) / (1 - self.B)  # B is calculated in the constructor
            C00_filtered = self.Cmax / (1 + (self.Cmax - 1) * eta0**2)

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
            # self.C00 = np.ones(num_interfaces)

    def update_drift_velocity(self):
        num_interfaces = self.geometry.num_interfaces
        # if np.all(self.C00 == 1):
        #     vD0 = np.zeros(num_interfaces)
        # else:
        if any(0 < sG < 1 for sG in self.iter_phases_props0_face[2]):
            [_, _, sG0_face, rhoG0_face, rhoL0_face, _, _] = (
                self.iter_phases_props0_face
            )
            [_, vM0, _, _] = self.velocities0

            mask = (sG0_face > 0) & (sG0_face < 1)
            indices = np.where(mask)[0]
            sG0_face_filtered = sG0_face[indices]
            rhoG0_face_filtered = rhoG0_face[indices]
            rhoL0_face_filtered = rhoL0_face[indices]
            vM0_filtered = vM0[indices]
            rhoM0_face_filtered = self.rhoM0_face[indices]

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
                    + (1 - sG0_face_filtered) * rhoL0_face_filtered
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
        num_segments = self.geometry.num_segments
        num_conn = self.geometry.num_interfaces
        num_phase_velocities = num_conn * 2
        num_primary_vars = len(Xn_dfm_well)
        n_vars = self.physics.n_vars

        # Preallocate the matrix of derivatives of phase velocities
        vel_der_matrix = np.zeros((num_phase_velocities, num_primary_vars))

        phase_velocities = self.eval_phase_vels(
            Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter, flag=1
        )

        # Construct the matrix of derivatives of phase velocities
        if self.diff_method == "numerical":
            for i in range(num_segments):
                # Derivatives of all the phase velocities with respect to the pressure of segment i
                X_dfm_well[i * n_vars] += self.eps_p
                vel_der_matrix[:, i * n_vars] = (
                    self.eval_phase_vels(
                        Xn_dfm_well,
                        X_dfm_well,
                        dt,
                        simulation_time,
                        iter_counter,
                        flag=0,
                    )
                    - phase_velocities
                ) / self.eps_p
                X_dfm_well[i * n_vars] -= self.eps_p

                for j in range(self.physics.nc - 1):
                    # Derivatives of all the phase velocities with respect to the mole fraction of component j in segment i
                    X_dfm_well[i * n_vars + j + 1] += self.eps_z
                    vel_der_matrix[:, i * n_vars + j + 1] = (
                        self.eval_phase_vels(
                            Xn_dfm_well,
                            X_dfm_well,
                            dt,
                            simulation_time,
                            iter_counter,
                            flag=0,
                        )
                        - phase_velocities
                    ) / self.eps_z
                    X_dfm_well[i * n_vars + j + 1] -= self.eps_z

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
                        - phase_velocities
                    ) / self.eps_temp
                    X_dfm_well[i * n_vars + n_vars - 1] -= self.eps_temp

            # Update properties at the current time step with the original primary variables (original X_dfm_well)
            # unaffected by eps_p, eps_temp, and eps_z
            phase_velocities = self.eval_phase_vels(
                Xn_dfm_well, X_dfm_well, dt, simulation_time, iter_counter, flag=0
            )

            vel_der_matrix_G = vel_der_matrix[: num_phase_velocities // 2, :]
            vel_der_matrix_L = vel_der_matrix[num_phase_velocities // 2 :, :]
        elif self.diff_method == "OBL":
            vel_der_matrix_G = self.vG_der
            vel_der_matrix_L = self.vL_der

        vel_der_matrix_G_clean = np.zeros((num_conn, 2 * n_vars))
        vel_der_matrix_L_clean = np.zeros((num_conn, 2 * n_vars))
        for a in range(num_conn):
            vel_der_matrix_G_clean[a] = vel_der_matrix_G[
                a, a * n_vars : a * n_vars + 2 * n_vars
            ]
            vel_der_matrix_L_clean[a] = vel_der_matrix_L[
                a, a * n_vars : a * n_vars + 2 * n_vars
            ]

        # Flatten and concatenate both arrays
        phase_velocities_derivatives = np.concatenate(
            (
                vel_der_matrix_G_clean.flatten(),
                vel_der_matrix_L_clean.flatten(),
            )
        )

        return phase_velocities, phase_velocities_derivatives

    def get_operator_der_matrix_for_well(self, op_idx):
        """
        Extract the derivative matrix of the specified operator for a well. This operator derivative
        matrix is used for differentiating phase velocities using the OBL approach.

        :param op_idx: Index of the desired operator
        :type op_idx: int

        Returns
        der : ndarray, shape (num_segments, num_segments * n_vars)
        """
        op_ders_arr = np.array(self.physics.engine.op_ders_arr)
        n_vars = self.physics.n_vars
        n_ops = self.physics.n_ops
        total_cells = self.reservoir.mesh.n_blocks

        well_obj = self.reservoir.get_well(self.name)
        start = well_obj.well_head_idx
        stop = well_obj.well_head_idx + self.geometry.num_segments

        # [all_cells, ops, vars]
        arr = op_ders_arr.reshape(total_cells, n_ops, n_vars)
        well_indices = np.arange(start, stop)

        local = arr[well_indices, op_idx, :]  # (num_segments, n_vars)

        n = self.geometry.num_segments
        der = np.zeros((n, n, n_vars), dtype=local.dtype)
        idx = np.arange(n)
        der[idx, idx, :] = local
        return der.reshape(n, n * n_vars)
