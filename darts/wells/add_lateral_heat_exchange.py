from darts.wells.define_pipe_geometry import PipeGeometry

import numpy as np

class WellLateralHeatTransfer:
    def __init__(self, pipe_name: str, pipe_geometry: PipeGeometry, earth_thermal_props: dict, outermost_layer_OD: float,
                 Ui: float = None, well_layers_props: dict = None, time_function_name: str = "Chiu&Thakur",
                 verbose: bool = False):
        """
        This class defines lateral heat transfer between the wellbore the geometry of which is entered as the first
        input argument of the constructor and the surrounding rock/soil using a semi-analytical lateral heat
        transfer model.
        Note that from the input args "U" and "well_layers_props", only one must be specified.

        :param pipe_name: Name of the pipe (well) for which WellLateralHeatTransfer is added.
        :type pipe_name: str
        :param pipe_geometry: The geometry of the pipe (well) for which lateral heat transfer is intended to be defined
        :type pipe_geometry: PipeGeometry
        :param earth_thermal_props: A dictionary containing earth thermal properties including these keys:
        "T": Earth temperature with the number of elements equal to the number of segments of the wellbore (list)
        "c": Earth specific heat capacity (float or list)
        "K": Earth thermal conductivity (float or list)
        "rho": Earth density (float or list)
        :type earth_thermal_props: dict
        :param Ui: Overall heat transfer coefficient based on the inner pipe diameter. If Ui is not specified,
        well_layers_props must be specified.
        :type Ui: float
        :param well_layers_props: The properties of the layers surrounding the fluid in the wellbore to thermal
        calculations. If well_layers_props is not specified, Ui must be specified.
        :type well_layers_props: dict
        :param time_function_name: The name of the time function used for transient calculation of heat transfer
        Available options are "Ramey" and "Chiu&Thakur". Default is "Chiu&Thakur"
        :type time_function_name: str
        :param verbose: Whether to display extra info about WellLateralHeatTransfer
        :type verbose: boolean
        """
        assert pipe_geometry.pipe_name == pipe_name, \
            "The names of the pipes in PipeGeometry and WellLateralHeatTransfer are not identical!"
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

        if Ui is not None:
            self.tubing_IR = pipe_geometry.pipe_IR
            self.Ui = Ui

        self.time_function_name = time_function_name
        self.outermost_layer_OD = outermost_layer_OD

        self.segments_lengths = pipe_geometry.segments_lengths

        self.q_lateral_heat = []

        if verbose:
            print("** WellLateralHeatTransfer for the well \"%s\" is added!" % pipe_name)

    def evaluate(self, T_segments, simulation_timer):
        """
        :param T_segments: Fluid temperature inside the segment
        :param simulation_timer: Simulation timer in seconds
        :return Lateral heat rate
        """
        # Time function evaluation
        # outermost_layer_OD is the outside diameter of the outermost layer of the wellbore before the formation, so
        # it could be a casing, a cement sheath, etc.
        if self.time_function_name == "Ramey":
            # Ramey's time function: Gives reasonably good results for long times but fails for times less than seven days.
            f_t = 1 / (- np.log((self.outermost_layer_OD/2) / (2 * np.sqrt(self.alpha * simulation_timer))) - 0.29)
        elif self.time_function_name == "Chiu&Thakur":
            # Chiu and Thakur time function: Provides a reasonable approximation of transient wellbore-formation heat
            # exchange while avoiding the early time discontinuity that results from using Ramey’s time function.
            f_t = 0.982 * np.log(1 + 1.81 * np.sqrt(self.alpha * simulation_timer) / (self.outermost_layer_OD))
        else:
            raise TypeError("Unrecognized time function name " + self.time_function_name)

        # Lateral heat rate evaluation
        if self.Ui is not None:
            # For constant overall heat transfer coefficient
            # I should see if U is based on ID or OD of the pipe. I think it's based on ID.
            self.q_lateral_heat = (2 * np.pi * self.tubing_IR * self.segments_lengths) * self.Ui * (self.T_earth - T_segments) / f_t
            return self.q_lateral_heat
        elif self.well_layers_props is not None:
            # Calculate the overall heat transfer coefficient using Willhite's formula
            U_to = "Willhite's formula"
            r_to = "tubing_outside_radius"
            self.q_lateral_heat = (2 * np.pi * self.K_earth * self.segments_lengths * (self.T_earth - T_segments)
                                   / (f_t + self.K_earth / (r_to * U_to)))
            return self.q_lateral_heat