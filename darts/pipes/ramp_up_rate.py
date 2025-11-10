import numpy as np

from darts.physics.super.physics import Compositional
from darts.pipes.define_pipe_geometry import PipeGeometry


class RampUpRate:
    """
    This class is used to create a ramp-up rate for the start-up of production or injection.
    """

    def __init__(
        self,
        pipe_name: str,
        pipe_geom: PipeGeometry,
        physics,
        first_ts_size: float,
        segment_idx: int,
        inflow_or_outflow: str,
        target_molar_rate: float,
        ramp_up_period: float,
        inj_fluid_props: dict = None,
    ):
        """
        :param pipe_name: Name of the pipe for which the ramp-up rate is going to be defined
        :type pipe_name: str
        :param pipe_geom: Pipe geometry object of the pipe for which the RampUpRate is going to be defined. It is used for assertion purposes.
        :type pipe_geom: PipeGeometry
        :param physics: physics object is used for assertion purposes and for evaluation of molar enthalpy for thermal scenarios
        :param first_ts_size: Size of the first time step from the class DataTS in darts_model.py [day]
        :type first_ts_size: float
        :param segment_idx: The index of the segment which fluid will be injected into or produced from
        :type segment_idx: int
        :param inflow_or_outflow: Whether the fluid is coming in ("inflow") or going out ("outflow") of the pipe
        :type inflow_or_outflow: str
        :param target_molar_rate: The target molar rate of inflow or outflow. For inflow is positive and for outflow is negative [kmol/day]
        :type target_molar_rate: float
        :param ramp_up_period: The period during which the rate ramps up from zero to the specified target rate [day]
        :type ramp_up_period: float
        :param inj_fluid_props: If inflow_or_outflow is "inflow", the props of the injected fluid must be specified as a dict
        with the following keys:
            "composition"
            For thermal scenarios the following keys are also required:
                Method 1: "pressure", "temperature", "phase_name"
                Method 2: "molar_enthalpy"
        :type inj_fluid_props: dict
        """
        assert pipe_name == pipe_geom.pipe_name, (
            "Pipe names for PipeGeometry and RampUpRate are not identical!"
        )
        self.pipe_name = pipe_name
        self.physics = physics

        assert isinstance(first_ts_size, float), "first_ts_size must be a float!"
        assert 0 < first_ts_size, "first_ts_size must be positive!"
        assert first_ts_size < 0.01 / 24 / 60 / 60, (
            "first_ts_size had better be smaller than 0.01 seconds because during this time step I set the rate to zero!"
        )

        assert isinstance(segment_idx, int), "segment_idx must be an int!"
        assert 0 <= segment_idx < pipe_geom.num_segments, "Invalid segment index!"
        self.segment_idx = segment_idx

        assert inflow_or_outflow in ["inflow", "outflow"], (
            "inflow_or_outflow must be either inflow or outflow!"
        )
        self.inflow_or_outflow = inflow_or_outflow

        assert isinstance(target_molar_rate, float), (
            "Specified target rate is not a float!"
        )
        if inflow_or_outflow == "inflow":
            assert target_molar_rate > 0, "Target rate must be positive!"
        elif inflow_or_outflow == "outflow":
            assert target_molar_rate < 0, "Target rate must be negative!"
        else:
            raise ValueError("target_rate must be non-zero!")
        self.target_rate = target_molar_rate

        assert isinstance(ramp_up_period, float), (
            "Specified ramp_up_period is not a float!"
        )
        assert ramp_up_period >= 0, "ramp_up_period must not be negative!"
        self.ramp_up_period = ramp_up_period

        if inflow_or_outflow == "inflow":
            assert isinstance(inj_fluid_props, dict), "inj_fluid_props must be a dict!"

            assert "composition" in inj_fluid_props, (
                "inj_fluid_props must contain a 'composition' key!"
            )
            comp = inj_fluid_props["composition"]
            assert isinstance(comp, np.ndarray | list), (
                "Specified composition is neither a numpy array nor a list!"
            )
            comp = comp if isinstance(comp, np.ndarray) else np.array(comp)
            assert len(comp) == physics.nc, (
                "Number of elements in composition is not equal to the number of components!"
            )
            assert np.isclose(sum(comp), 1, atol=1e-12, rtol=1e-12), (
                "Summation of injected fluid mole fractions must be equal to 1!"
            )
            inj_fluid_props["composition"] = comp

            if physics.thermal:
                assert (
                    all(
                        i in inj_fluid_props
                        for i in ("pressure", "temperature", "phase_name")
                    )
                ) ^ ("molar_enthalpy" in inj_fluid_props), (
                    "Provide either the pressure, temperature, and phase_name keys or the molar_enthalpy key in inj_fluid_props!"
                )

                if "pressure" in inj_fluid_props:
                    assert isinstance(inj_fluid_props["pressure"], float), (
                        "Specified pressure must be a float!"
                    )

                if "temperature" in inj_fluid_props:
                    assert isinstance(inj_fluid_props["temperature"], float), (
                        "Specified temperature must be a float!"
                    )
                    assert inj_fluid_props["temperature"] > 273.15, (
                        "Specified temperature must be in Kelvin!"
                    )

                if "phase_name" in inj_fluid_props:
                    ph_name = inj_fluid_props["phase_name"]
                    assert isinstance(ph_name, str), (
                        "Specified phase_name is not a string!"
                    )
                    assert ph_name in physics.phases, (
                        "Specified phase_name is not in the list of the phase names in physics!"
                    )

                if "molar_enthalpy" in inj_fluid_props:
                    assert all(
                        i not in inj_fluid_props
                        for i in ("pressure", "temperature", "phase_name")
                    ), (
                        "If molar_enthalpy is specified, pressure, temperature, phase_name must not be specified!"
                    )
                    assert isinstance(inj_fluid_props["molar_enthalpy"], float), (
                        "Specified molar_enthalpy must be a float!"
                    )

                if "molar_enthalpy" not in inj_fluid_props:
                    # Calculate and store injected_fluid_molar_enthalpy
                    injected_fluid_molar_enthalpy = (
                        physics.property_containers[0]
                        .enthalpy_ev[ph_name]
                        .evaluate(
                            inj_fluid_props["pressure"],
                            inj_fluid_props["temperature"],
                            inj_fluid_props["composition"],
                        )
                    )
                    inj_fluid_props["molar_enthalpy"] = injected_fluid_molar_enthalpy
                elif "molar_enthalpy" in inj_fluid_props:
                    # molar_enthalpy is directly specified by the user
                    pass

            else:
                assert "pressure" not in inj_fluid_props, (
                    "inj_fluid_props must not contain a 'pressure' key for isothermal scenarios!"
                )
                assert "temperature" not in inj_fluid_props, (
                    "inj_fluid_props must not contain a 'temperature' key for isothermal scenarios!"
                )
                assert "phase_name" not in inj_fluid_props, (
                    "inj_fluid_props must not contain a 'phase_name' key for isothermal scenarios!"
                )
        elif inflow_or_outflow == "outflow":
            assert inj_fluid_props is None, (
                "For outflow, inj_fluid_props must not be specified!"
            )
            assert physics.state_spec == Compositional.StateSpecification.PH, (
                "Thermal production only work with the PH formulation for multiphase flow accurately!"
            )

        self.inj_fluid_props = inj_fluid_props

        # If the user sets ramp_up_period to zero means that they don't want to have a ramp-up rate
        if ramp_up_period == 0.0:
            self.current_rate = target_molar_rate
        # Rate starts from zero, so initial rate is zero:
        elif ramp_up_period > 0.0:
            self.current_rate = 0.0

    def update_current_molar_rate(self, simulation_time):
        """
        :param simulation_time: Simulation time [day]
        :type simulation_time: float
        """
        if simulation_time < self.ramp_up_period:
            self.current_rate = (
                simulation_time / self.ramp_up_period
            ) * self.target_rate
        else:
            self.current_rate = self.target_rate
