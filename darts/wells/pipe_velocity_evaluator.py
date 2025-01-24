import math
import numpy as np

from darts.engines import value_vector
# from darts.models.darts_model import DartsModel

from darts.wells.define_pipe_geometry import PipeGeometry


class PipeVelocityEvaluator:
    g = 9.80665   # * meter() / second()**2  # Gravitational acceleration
    Cku = 142
    Cw = 0.008

    def __init__(self, pipe_geometry: PipeGeometry, physics, darts_model, Cmax: float = 1.2, Fv: float = 1,
                 eps_p: float = 1e-4, eps_temp: float = 0.1, eps_z: float = 0.00001, verbose: bool = False):
        """
        :param pipe_geometry: Pipe geometry object
        :type pipe_geometry: PipeGeometry
        :param physics: Physics object for the pipe
        :param Cmax: A user-specified maximum profile parameter that can be tuned to match the observations and
        could have a value between 1.0 and 1.5. It is set to:
        --> 1.2 in ECLIPSE according to Shi et al. paper (Drift-Flux Modeling of Two-Phase Flow in Wellbores)
        --> 1 in a wellbore simulator in Tonken et al. paper (A transient geothermal wellbore simulator)
        :type Cmax: float
        :param Fv: A multiplier on the flooding velocity fraction, set to be 1 by default, and its value can be tuned
        to fit the observations.
        :type Fv: float
        :param eps_p: A very small number used for numerically differentiating residual equations with respect to pressure.
        :type eps_p: float
        :param eps_temp: A very small number used for numerically differentiating residual equations with respect to temperature.
        :type eps_temp: float
        :param eps_z: A very small number used for numerically differentiating residual equations with respect to composition
        :type eps_z: float
        :param verbose: Whether to display extra info about PipeModel
        :type verbose: boolean
        """
        self.pipe_geometry = pipe_geometry

        self.physics = physics
        self.isothermal = not physics.thermal
        if self.isothermal:
            assert self.physics.property_containers[0].temperature is not None, \
                "If model is isothermal, system_temperature must be specified!"
        elif not self.isothermal:
            assert self.physics.property_containers[0].temperature is None, \
                "If model is non-isothermal, system_temperature must not be specified!"
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

        self.m = m0 * ((math.cos(pipe_geometry.inclination_angle_radian)) ** n1) * (
                    1 + math.sin(pipe_geometry.inclination_angle_radian)) ** n2

        self.a1 = a1
        self.a2 = a2

        self.g_cos_theta = self.g * math.cos(pipe_geometry.inclination_angle_radian)

        # Epsilon values for numerical differentiation with respect to pressure, temperature, and overall composition
        self.eps_p = eps_p
        self.eps_temp = eps_temp
        self.eps_z = eps_z

        self.darts_model = darts_model

        self.is_first_first_iter = True   # first_iter_in_first_ts_identifier

        if verbose:
            print("** Model of the pipe \"%s\" is created!" % self.pipe_geometry.pipe_name)

    def evaluate_phase_velocities(self, Xn_ms_well, X_ms_well, dt, flag):
        iter_counter = self.darts_model.iter_counter

        dt = dt * 24 * 60 * 60   # convert day to second

        num_segments = self.pipe_geometry.num_segments
        nc = self.physics.nc
        n_vars = self.physics.n_vars

        pc = self.physics.property_containers[0]

        """ Calculate phase props of previous time step at centroids """
        if iter_counter == 0 and self.is_first_first_iter is True and flag == 1:
            # Reshape into blocks
            Xn_ms_well_reshaped = Xn_ms_well.reshape(-1, nc)
            # Reverse the order of the blocks
            Xn_ms_well_reversed_blocks = Xn_ms_well_reshaped[::-1]
            # Flatten back to a 1D array
            Xn_ms_well = Xn_ms_well_reversed_blocks.flatten()

            """ From here on, instead of G, use A, and instead of L, use B. A and B represent the first and second 
            phases specified by the user, respectively. """
            sG0 = np.zeros(num_segments)
            rhoG0 = np.zeros(num_segments)
            rhoL0 = np.zeros(num_segments)
            miuG0 = np.zeros(num_segments)
            miuL0 = np.zeros(num_segments)
            xG_mass0 = np.zeros((num_segments, nc))
            xL_mass0 = np.zeros((num_segments, nc))

            for i in range(num_segments):
                state0 = Xn_ms_well[i * n_vars:(i + 1) * n_vars]
                pc.evaluate(state0)
                if self.physics.thermal:
                    pc.evaluate_thermal(state0)

                sG0[i] = pc.sat[0]
                rhoG0[i], rhoL0[i] = pc.dens[0], pc.dens[1]
                miuG0[i], miuL0[i] = pc.mu[0] * 1e-3, pc.mu[1] * 1e-3   # convert cP to Pa.s
                # Calculate mass fractions of components in each phase
                x_mass0 = np.zeros((pc.nph, nc))
                for j in pc.ph:
                    x_mass0[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
                xG_mass0[i, :], xL_mass0[i, :] = x_mass0[0, :], x_mass0[1, :]

            self.iter_phases_props0 = [xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, miuG0, miuL0]

        elif iter_counter == 0 and self.is_first_first_iter is False and flag == 1:
            self.iter_phases_props0 = self.iter_phases_props

        xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, _, _ = self.iter_phases_props0

        """ Calculate phase props of current time step at centroids """
        # Reshape into blocks
        X_ms_well_reshaped = X_ms_well.reshape(-1, nc)
        # Reverse the order of the blocks
        X_ms_well_reversed_blocks = X_ms_well_reshaped[::-1]
        # Flatten back to a 1D array
        X_ms_well = X_ms_well_reversed_blocks.flatten()

        sG = np.zeros(num_segments)
        rhoG = np.zeros(num_segments)
        rhoL = np.zeros(num_segments)
        miuG = np.zeros(num_segments)
        miuL = np.zeros(num_segments)
        xG_mass = np.zeros((num_segments, nc))
        xL_mass = np.zeros((num_segments, nc))

        for i in range(num_segments):
            state = X_ms_well[i * n_vars:(i + 1) * n_vars]
            pc.evaluate(state)
            if self.physics.thermal:
                pc.evaluate_thermal(state)

            sG[i] = pc.sat[0]
            rhoG[i], rhoL[i] = pc.dens[0], pc.dens[1]
            miuG[i], miuL[i] = pc.mu[0] * 1e-3, pc.mu[1] * 1e-3   # convert cP to Pa.s
            # Calculate mass fractions of components in each phase
            x_mass = np.zeros((pc.nph, nc))
            for j in pc.ph:
                x_mass[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
            xG_mass[i, :], xL_mass[i, :] = x_mass[0, :], x_mass[1, :]

        self.iter_phases_props = [xG_mass, xL_mass, sG, rhoG, rhoL, miuG, miuL]

        """ Calculate phase props of previous time step at interfaces """
        if iter_counter == 0 and flag == 1:
            # Method 1
            # # xG_mass0_face and xL_mass0_face for IFT calculation
            # # xG_mass0_face = (xG_mass0[0:-1] + xG_mass0[1:]) / 2
            # # xL_mass0_face = (xL_mass0[0:-1] + xL_mass0[1:]) / 2
            # xG_mass0_face = xG_mass0[0:-1]
            # xL_mass0_face = xL_mass0[0:-1]
            #
            # sG0_face = (sG0[0:-1] + sG0[1:]) / 2
            # rhoG0_face = (rhoG0[0:-1] + rhoG0[1:]) / 2
            # rhoL0_face = (rhoL0[0:-1] + rhoL0[1:]) / 2

            # Method 2
            sG0_face = (sG0[0:-1] + sG0[1:]) / 2

            # Initialize arrays to store interface properties
            rhoG0_face = np.zeros(num_segments - 1)
            rhoL0_face = np.zeros(num_segments - 1)
            xG_mass0_face = np.zeros((num_segments - 1, nc))
            xL_mass0_face = np.zeros((num_segments - 1, nc))

            # Compute interface values using conditional averaging
            for i in range(num_segments - 1):
                if sG0[i] == 0:
                    # If no gas in segment i, use properties from segment i+1
                    rhoG0_face[i] = rhoG0[i + 1]
                    xG_mass0_face[i] = xG_mass0[i + 1]
                elif sG0[i + 1] == 0:
                    # If no gas in segment i+1, use properties from segment i
                    rhoG0_face[i] = rhoG0[i]
                    xG_mass0_face[i] = xG_mass0[i]
                else:
                    # If both segments have gas, use arithmetic averaging
                    rhoG0_face[i] = (rhoG0[i] + rhoG0[i + 1]) / 2
                    xG_mass0_face[i] = (xG_mass0[i] + xG_mass0[i + 1]) / 2

                if sG0[i] == 1:
                    # If no liquid in segment i, use properties from segment i+1
                    rhoL0_face[i] = rhoL0[i + 1]
                    xL_mass0_face[i] = xL_mass0[i + 1]
                elif sG0[i + 1] == 1:
                    # If no liquid in segment i+1, use properties from segment i
                    rhoL0_face[i] = rhoL0[i]
                    xL_mass0_face[i] = xL_mass0[i]
                else:
                    # If both segments have liquid, use arithmetic averaging
                    rhoL0_face[i] = (rhoL0[i] + rhoL0[i + 1]) / 2
                    xL_mass0_face[i] = (xL_mass0[i] + xL_mass0[i + 1]) / 2

            self.iter_phases_props0_face = [xG_mass0_face, xL_mass0_face, sG0_face, rhoG0_face, rhoL0_face]

        """ Calculate phase props of current time step at interfaces """
        # Method 1
        # sG_face = (sG[0:-1] + sG[1:])/2
        # rhoG_face = (rhoG[0:-1] + rhoG[1:]) / 2
        # rhoL_face = (rhoL[0:-1] + rhoL[1:]) / 2

        # Method 2
        sG_face = (sG[0:-1] + sG[1:]) / 2

        # Initialize arrays to store interface properties
        rhoG_face = np.zeros(num_segments - 1)
        rhoL_face = np.zeros(num_segments - 1)

        # Compute interface values using conditional averaging
        for i in range(num_segments - 1):
            if sG[i] == 0:
                # If no gas in segment i, use properties from segment i+1
                rhoG_face[i] = rhoG[i + 1]
            elif sG[i + 1] == 0:
                # If no gas in segment i+1, use properties from segment i
                rhoG_face[i] = rhoG[i]
            else:
                # If both segments have gas, use arithmetic averaging
                rhoG_face[i] = (rhoG[i] + rhoG[i + 1]) / 2

            if sG[i] == 1:
                # If no liquid in segment i, use properties from segment i+1
                rhoL_face[i] = rhoL[i + 1]
            elif sG[i + 1] == 1:
                # If no liquid in segment i+1, use properties from segment i
                rhoL_face[i] = rhoL[i]
            else:
                # If both segments have liquid, use arithmetic averaging
                rhoL_face[i] = (rhoL[i] + rhoL[i + 1]) / 2

        self.iter_phases_props_face = [sG_face, rhoG_face, rhoL_face]

        if iter_counter == 0 and self.is_first_first_iter is True and flag == 1:
            # Initial velocities in the wellbore are zero
            rhoM0_vM0, vM0, vG0, vL0 = np.array([0]), np.array([0]), np.array([0]), np.array([0])
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])
        elif iter_counter == 0 and self.is_first_first_iter is False and flag == 1:
            rhoM0_vM0, vM0, vG0, vL0 = self.rhoM_vM, self.vM, self.vG, self.vL
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])

        [_, vM0, vG0, vL0] = self.velocities0

        p = X_ms_well[0::n_vars] * 1e5   # convert bar to Pa
        p_m = p[0:-1:1]
        p_p = p[1::1]

        self.calc_mixture_densities(iter_counter, flag)

        pg = self.pipe_geometry

        if iter_counter == 0 and flag == 1:
            # delta_segment = (pg.pipe_internal_A * delta_interface / pg.D + pg.pipe_internal_A * delta_interface / pg.D) / (pg.pipe_internal_A / pg.D + pg.pipe_internal_A / pg.D)
            # To increase the numerical stability, you may need to use an upwind scheme for the momentum flux like
            # in the paper "A transient gothermal wellbore simulator (2023)
            delta_interface0 = pg.pipe_internal_A * (rhoG0_face * sG0_face * vG0 ** 2 +
                                                     rhoL0_face * (1 - sG0_face) * vL0 ** 2)

            """ Add momentum boundary conditions """
            momentum_at_first_last_exterfaces = [0, 0]
            # if self.darts_model.reservoir.wells[0].control.target_rate:
            if hasattr(self.darts_model.reservoir.wells[0].control, 'target_rate'):
                # TODO: This segment_index_source and mass_rate should be directly received from the well control class, but now done manually
                segment_index_source = num_segments - 1
                # mass_rate = self.darts_model.reservoir.wells[0].control.target_rate   # must be in kg/s
                mass_rate = sum(58895.98 * np.array([1.0 - 2 * 1e-5, 1e-5, 1e-5]) * [44.0098, 16.04288, 18.0152]) / (24 * 60 * 60)

                pipe_internal_A = self.pipe_geometry.pipe_internal_A

                # The props of the fluid of the segment on which the constant mass rate source is defined are used.
                sG0_source = sG0[segment_index_source]
                rhoG0_source = rhoG0[segment_index_source]
                rhoL0_source = rhoL0[segment_index_source]

                if sG0_source == 0:
                    vG0 = 0
                    liquid_mass_fraction0 = 1
                    liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
                    vL0 = liquid_mass_rate0 / rhoL0_source / (pipe_internal_A * 1)
                elif sG0_source == 1:
                    vL0 = 0
                    gas_mass_fraction0 = 1
                    gas_mass_rate0 = mass_rate * gas_mass_fraction0
                    vG0 = gas_mass_rate0 / rhoG0_source / (pipe_internal_A * 1)
                elif 0 < sG0_source < 1:
                    gas_mass_fraction0 = sG0_source * rhoG0_source / (sG0_source * rhoG0_source + (1 - sG0_source) * rhoL0_source)
                    gas_mass_rate0 = mass_rate * gas_mass_fraction0
                    vG0 = gas_mass_rate0 / rhoG0_source / (pipe_internal_A * sG0_source)

                    liquid_mass_fraction0 = 1 - gas_mass_fraction0
                    liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
                    vL0 = liquid_mass_rate0 / rhoL0_source / (pipe_internal_A * (1 - sG0_source))
                else:
                    raise Exception("sG0_source is out of correct range (from 0 to 1)!")

                delta_at_bc_interface0 = pipe_internal_A * (rhoG0_source * sG0_source * vG0 ** 2 + rhoL0_source * (1 - sG0_source) * vL0 ** 2)

                if segment_index_source == 0:
                    momentum_at_first_last_exterfaces[0] = delta_at_bc_interface0
                elif segment_index_source == num_segments - 1:
                    momentum_at_first_last_exterfaces[1] = delta_at_bc_interface0

            delta_interface0 = np.insert(delta_interface0, 0, momentum_at_first_last_exterfaces[0])
            delta_interface0 = np.insert(delta_interface0, num_segments, momentum_at_first_last_exterfaces[1])

            delta_segment0 = (delta_interface0[0:-1:1] + delta_interface0[1::1]) / 2
            self.delta_m0 = delta_segment0[0:-1:1]
            self.delta_p0 = delta_segment0[1::1]

            ff0 = self.calc_Fanning_friction_factor()
            self.w0 = 1 / (1 / dt + pg.perimeter * ff0 * abs(vM0) / (2 * pg.pipe_internal_A))

        self.rhoM_vM = (- self.w0 * (p_p - p_m) / (pg.z_p - pg.z_m)
                        - self.w0 * self.g_cos_theta * self.rhoM_face
                        - self.w0 * ((self.delta_p0 - self.delta_m0) / (pg.pipe_internal_A * (pg.z_p - pg.z_m)) - self.rhoM0_face * vM0 / dt))

        self.vM = self.rhoM_vM / self.rhoM_face

        if iter_counter == 0 and flag == 1 and self.is_first_first_iter is True:
            self.vD0 = np.zeros(self.pipe_geometry.num_interfaces)
        elif iter_counter == 0 and flag == 1 and self.is_first_first_iter is False:
            self.calc_drift_velocity()

        # Gas velocity at wellbore interfaces
        # self.vG = self.C00 * self.rhoM_vM / self.rhoM_adjusted_face + rhoL_face * self.vD0 / self.rhoM_adjusted_face
        self.vG = np.zeros(self.pipe_geometry.num_interfaces)
        for i in range(self.pipe_geometry.num_interfaces):
            if sG_face[i] != 0:
                self.vG[i] = self.C00[i] * self.rhoM_vM[i] / self.rhoM_adjusted_face[i] + rhoL_face[i] * self.vD0[i] / self.rhoM_adjusted_face[i]

        # Liquid velocity at wellbore interfaces
        self.vL = np.zeros(self.pipe_geometry.num_interfaces)
        for i in range(self.pipe_geometry.num_interfaces):
            if sG_face[i] != 1:
                self.vL[i] = ((1 - self.C00[i] * sG_face[i]) * self.rhoM_vM[i] / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i])
                         - sG_face[i] * rhoG_face[i] * self.vD0[i] / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i]))

        # concatenate phase velocities, reverse the order, and convert m/s to m/day
        phase_velocities = np.concatenate((self.vG[::-1] * 24 * 60 * 60, self.vL[::-1] * 24 * 60 * 60))

        self.is_first_first_iter = False   # For the next iterations will be False

        return phase_velocities

    def calc_mixture_densities(self, iter_counter, flag):
        if iter_counter == 0 and self.is_first_first_iter is True and flag == 1:
            _, _, sG0, rhoG0, rhoL0, _, _ = self.iter_phases_props0

            rhoM0 = sG0 * rhoG0 + (1 - sG0) * rhoL0
            self.rhoM0_face = (rhoM0[0:-1] + rhoM0[1:]) / 2
            # _, _, sG0_face, rhoG0_face, rhoL0_face = self.iter_phases_props0_face
            # self.rhoM0_face = sG0_face * rhoG0_face + (1 - sG0_face) * rhoL0_face

        elif iter_counter == 0 and self.is_first_first_iter is False and flag == 1:
            self.rhoM0_face = self.rhoM_face

        # Calculate mixture density
        _, _, sG, rhoG, rhoL, _, _ = self.iter_phases_props

        [sG_face, rhoG_face, rhoL_face] = self.iter_phases_props_face
        rhoM = sG * rhoG + (1 - sG) * rhoL
        self.rhoM_face = (rhoM[0:-1] + rhoM[1:]) / 2
        # self.rhoM_face = sG_face * rhoG_face + (1 - sG_face) * rhoL_face

        # Calculate adjusted-mixture density
        if iter_counter == 0 and flag == 1 and self.is_first_first_iter is False:
            self.calc_profile_parameter()
        elif iter_counter == 0 and flag == 1 and self.is_first_first_iter is True:
            # At the beginning, there is no flow, so C00 is considered 1 everywhere.
            self.C00 = np.ones(self.pipe_geometry.num_interfaces)
        self.rhoM_adjusted_face = self.C00 * sG_face * rhoG_face + (1 - self.C00 * sG_face) * rhoL_face

    def calc_Fanning_friction_factor(self):
        pg = self.pipe_geometry
        Re0 = self.calc_Reynolds_number()
        ff0 = []
        for i in range(pg.num_interfaces):
            if Re0[i] == 0:
                ff0.append(0)

            elif Re0[i] != 0:
                if Re0[i] < 2400:
                    ff0.append(16 / Re0[i])
                elif Re0[i] > 2400:
                    # T2Well
                    # ff0.append((1 / (-4 * math.log10(2 * pg.wall_roughness / (3.7 * pg.pipe_ID) - (5.02 / Re0[i])
                    #             * math.log10(2 * pg.wall_roughness / (3.7 * pg.pipe_ID) + 13 / Re0[i])))) ** 2)

                    # # Chen's correlation (The explicit form of Colebrook-White's correlation)
                    # relative_roughness = pg.wall_roughness / pg.pipe_ID
                    # Fanning_friction_factor = (1 / (-4 * math.log10(relative_roughness / 3.7065 - 5.0452 / Re0[i] * math.log10(relative_roughness ** 1.1098 / 2.8257 + (7.149 / Re0[i]) ** 0.8981)))) ** 2
                    # ff0.append(Fanning_friction_factor)

                    # Define Colebrook-White correlation
                    from scipy.optimize import fsolve
                    def colebrook(f, Re, relative_roughness):
                        if f <= 0:  # Ensure the friction factor doesn't go negative or zero
                            return 1e6  # Return a large value to prevent sqrt of negative number
                        return 1 / math.sqrt(f) + 4 * math.log10(
                            relative_roughness / 3.7065 + (1.2613 / (Re * math.sqrt(f))))

                    # Initial guess for f
                    initial_guess = 0.005

                    # Solve for f using fsolve
                    Fanning_friction_factor = \
                    fsolve(colebrook, initial_guess, args=(Re0[i], pg.wall_roughness / pg.pipe_ID))[0]
                    ff0.append(Fanning_friction_factor)

        self.ff0 = np.array(ff0)

        return self.ff0

    def calc_Reynolds_number(self):
        _, _, _, rhoG0, rhoL0, miuG0, miuL0 = self.iter_phases_props0

        _, _, sG0_face, _, _ = self.iter_phases_props0_face
        [_, vM0, _, _] = self.velocities0

        miuG0_face = (miuG0[0:-1] + miuG0[1:]) / 2 if len(miuG0) != 1 else miuG0
        miuL0_face = (miuL0[0:-1] + miuL0[1:]) / 2 if len(miuL0) != 1 else miuL0

        # Saturation-weighted average is used to calculate the mixture viscosity of two phases. The method is used in
        # Beggs and Brill's book: Eq. 1.38
        # My production engineering notebook: Pressure drop calc in wellbore for 2-phase flow with Beggs and Brill's method
        self.miuM0 = sG0_face * miuG0_face + (1 - sG0_face) * miuL0_face

        pg = self.pipe_geometry

        self.Re0 = self.rhoM0_face * abs(vM0) * pg.pipe_ID / self.miuM0

        return self.Re0

    def calc_profile_parameter(self):
        pg = self.pipe_geometry
        [rhoM0_vM0, _, _, _] = self.velocities0
        [xG_mass0_face, xL_mass0_face, sG0_face, rhoG0_face, rhoL0_face] = self.iter_phases_props0_face

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
                IFT0_face[i] = self.physics.property_containers[0].IFT_ev.evaluate(rhoG0_face_filtered[i], rhoL0_face_filtered[i],
                                                                                   xG_mass0_face_filtered[i], xL_mass0_face_filtered[i])

            # self.Ku0_filtered = np.zeros(len(indices))
            # self.vC0_filtered = np.zeros(len(indices))

            # Calculate C00 from the solution of the previous time step
            vM0 = rhoM0_vM0_filtered / rhoM0_face_filtered
            NB0 = (pg.pipe_ID ** 2) * (self.g * (rhoL0_face_filtered - rhoG0_face_filtered) / IFT0_face)
            self.Ku0_filtered = np.sqrt(self.Cku / np.sqrt(NB0) * (np.sqrt(1 + NB0 / (self.Cku ** 2 * self.Cw)) - 1))
            self.vC0_filtered = (self.g * IFT0_face * (rhoL0_face_filtered - rhoG0_face_filtered) / rhoL0_face_filtered ** 2) ** 0.25
            v_sgf0 = self.Ku0_filtered * np.sqrt(rhoL0_face_filtered / rhoG0_face_filtered) * self.vC0_filtered
            beta0 = np.maximum(sG0_face_filtered, self.Fv * sG0_face_filtered * abs(vM0) / v_sgf0)
            beta0 = np.clip(beta0, 0, 1)   # beta0 is subject to limits 0 <= beta0 <= 1
            eta0 = (beta0 - self.B) / (1 - self.B)   # B is calculated in the constructor
            C00_filtered = self.Cmax / (1 + (self.Cmax - 1) * eta0 ** 2)

            C00 = np.ones(self.pipe_geometry.num_interfaces)   # C00 all ones first
            self.C00 = np.ones(self.pipe_geometry.num_interfaces)
            self.C00_filtered = np.ones(len(indices))

            for i, idx in enumerate(indices):
                C00[idx] = C00_filtered[i]

            self.C00_filtered = C00_filtered
            self.C00 = C00
            # Set all profile parameters equal to 1
            # self.C00 = np.ones(self.pipe_geometry.num_interfaces)

        else:
            self.C00_filtered = 1
            # self.C00 = np.ones(self.pipe_geometry.num_interfaces)

    def calc_drift_velocity(self):
        # if np.all(self.C00 == 1):
        #     vD0 = np.zeros(self.pipe_geometry.num_interfaces)
        # else:
        if any(0 < sG < 1 for sG in self.iter_phases_props0_face[2]):
            pg = self.pipe_geometry
            [_, _, sG0_face, rhoG0_face, rhoL0_face] = self.iter_phases_props0_face
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
            if type(self.Ku0_filtered) is float:   # If type of self.Ku0_filtered is float, this turns it into a list.
                self.Ku0_filtered = [self.Ku0_filtered]

            for index, value in enumerate(indices):
                if sG0_face[value] <= self.a1:
                    K0_filtered[index] = 1.53
                elif self.a1 < sG0_face[value] < self.a2:
                    K0_filtered[index] = 1.53 + (self.C00_filtered[index] * self.Ku0_filtered[index] - 1.53) / 2 * (1 - np.cos(math.pi * (sG0_face[value] - self.a1) / (self.a2 - self.a1)))
                elif sG0_face[value] >= self.a2:
                    K0_filtered[index] = self.C00_filtered[index] * self.Ku0_filtered[index]

            # Calculate the adjustment function for the mist flow regime
            # I'm not sure if X should be multiplied by C0 or not.
            Xm1 = 1
            Xm2 = 0.94
            Gm1 = 300
            Gm2 = 700
            alpha = 0.001
            lambdaa = 199
            X0 = sG0_face_filtered * rhoG0_face_filtered / (sG0_face_filtered * rhoG0_face_filtered + (1 - sG0_face_filtered) * rhoL0_face_filtered)   # gas mass fraction [dimensionless]
            G0 = rhoM0_face_filtered * abs(vM0_filtered)   # Total mass flux (or total mass flow rate per unit cross-sectional area) [kg/m2/s]
            numerator = np.zeros(len(X0))
            for i in range(len(X0)):
                numerator[i] = np.linalg.det(np.array([[X0[i], alpha * G0[i], 1], [Xm1, alpha * Gm1, 1], [Xm2, alpha * Gm2, 1]]))

            denominator = np.sqrt((Xm2 - Xm1)**2 + (alpha*Gm2 - alpha*Gm1)**2)
            Dm = numerator/denominator
            f0 = np.maximum(0, 1 - np.minimum(1, G0/Gm1) * np.exp(-lambdaa * Dm*abs(Dm)))

            # Calculate drift velocity
            vD0 = np.zeros(self.pipe_geometry.num_interfaces)   # vD0 all zeros first
            for index, value in enumerate(indices):
                # Ignore the consideration of the adjustment function for the mist flow regime for now
                vD0[value] = (1 - self.C00_filtered[index] * sG0_face_filtered[index]) * self.vC0_filtered[index] * K0_filtered[index] * self.m * f0[index] / (self.C00_filtered[index] * sG0_face_filtered[index] * np.sqrt(rhoG0_face_filtered[index] / rhoL0_face_filtered[index]) + 1 - self.C00_filtered[index] * sG0_face_filtered[index])
                # vD0[value] = (1 - self.C00_filtered[index] * sG0_face_filtered[index]) * self.vC0_filtered[index] * K0_filtered[index] * self.m / (self.C00_filtered[index] * sG0_face_filtered[index] * np.sqrt(rhoG0_face_filtered[index] / rhoL0_face_filtered[index]) + 1 - self.C00_filtered[index] * sG0_face_filtered[index])
        else:
            vD0 = np.zeros(self.pipe_geometry.num_interfaces)
        self.vD0 = vD0

    def evaluate_phase_velocities_and_derivatives(self, Xn_ms_well, X_ms_well, dt):
        Xn_ms_well = Xn_ms_well.to_numpy()
        X_ms_well = X_ms_well.to_numpy()

        num_segments = self.pipe_geometry.num_segments
        num_conn = self.pipe_geometry.num_interfaces
        num_phase_velocities = num_conn * 2
        num_primary_vars = len(Xn_ms_well)
        n_vars = self.physics.n_vars

        jac = np.zeros((num_phase_velocities, num_primary_vars))

        phase_velocities = self.evaluate_phase_velocities(Xn_ms_well, X_ms_well, dt, flag=1)

        # Construct the Jacobian matrix
        for i in range(num_segments):
            # Derivatives of all the phase velocities with respect to the pressure of segment i
            X_ms_well[i * n_vars] += self.eps_p
            jac[:, i * n_vars] = (self.evaluate_phase_velocities(Xn_ms_well, X_ms_well, dt, flag=0) - phase_velocities) / self.eps_p
            X_ms_well[i * n_vars] -= self.eps_p

            for j in range(self.physics.nc - 1):
                # Derivatives of all the phase velocities with respect to the mole fraction of component j in segment i
                X_ms_well[i * n_vars + j + 1] += self.eps_z
                jac[:, i * n_vars + j + 1] = (self.evaluate_phase_velocities(Xn_ms_well, X_ms_well, dt, flag=0)
                                              - phase_velocities) / self.eps_z
                X_ms_well[i * n_vars + j + 1] -= self.eps_z

            if not self.isothermal:
                # Derivatives of all the phase velocities with respect to the temperature of segment i
                X_ms_well[i * n_vars + n_vars - 1] += self.eps_temp
                jac[:, i * n_vars + n_vars - 1] = (self.evaluate_phase_velocities(
                    Xn_ms_well, X_ms_well, dt, flag=0) - phase_velocities) / self.eps_temp
                X_ms_well[i * n_vars + n_vars - 1] -= self.eps_temp

        # Update properties at the current time step with the original primary variables (original X_ms_well)
        # unaffected by eps_p, eps_temp, and eps_z
        phase_velocities = self.evaluate_phase_velocities(Xn_ms_well, X_ms_well, dt, flag=-1)

        jac_phase_A = jac[:num_phase_velocities // 2, :]
        jac_phase_B = jac[num_phase_velocities // 2:, :]
        jac_phase_A_clean_flat = np.zeros((num_conn, 2 * n_vars))
        jac_phase_B_clean_flat = np.zeros((num_conn, 2 * n_vars))
        for a in range(num_conn):
            jac_phase_A_clean_flat[a] = jac_phase_A[a, a*n_vars : a*n_vars+2*n_vars]
            jac_phase_B_clean_flat[a] = jac_phase_B[a, a*n_vars : a*n_vars+2*n_vars]

        # Flatten and concatenate both arrays
        phase_velocities_derivatives = np.concatenate((jac_phase_A_clean_flat.flatten(), jac_phase_B_clean_flat.flatten()))

        return value_vector(phase_velocities), value_vector(phase_velocities_derivatives)
