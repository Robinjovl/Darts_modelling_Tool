import numpy as np

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.ramp_up_rate import RampUpRate


class UpstreamRampUpRate(RampUpRate):
    """
    RampUpRate, but using the upstream properties to calculate boundary momentum.

    Unlike RampUpRate, which uses the properties of the pipe segment to calculate boundary momentum,
    this class uses the properties of the upstream node to calculate boundary momentum.
    """

    def __init__(
        self,
        pipe_name: str,
        pipe_geom: PipeGeometry,
        physics,
        first_ts_size: float,
        segment_idx: int,
        target_molar_rate: float,
        ramp_up_period: float,
        composition,
        pressure: float = None,
        temperature: float = None,
        phase_name: str = None,
        molar_enthalpy: float = None,
        verbose: bool = False,
    ):
        """
        In RampUpRate, if molar_enthalpy is provided, pressure, temperature, and phase_name must not be specified,
        but in UpstreamRampUpRate, pressure, temperature, and phase_name must be specified all the time because
        they are needed to calculate the boundary momentum.
        """
        if not physics.thermal:
            raise ValueError("UpstreamRampUpRate currently requires thermal physics.")

        inj_fluid_props = {"composition": composition}
        if molar_enthalpy is None:
            if pressure is None or temperature is None or phase_name is None:
                raise ValueError(
                    "pressure, temperature, and phase_name must be provided when molar_enthalpy is not specified."
                )
            inj_fluid_props.update(
                {
                    "pressure": pressure,
                    "temperature": temperature,
                    "phase_name": phase_name,
                }
            )
        else:
            inj_fluid_props["molar_enthalpy"] = float(molar_enthalpy)

        super().__init__(
            pipe_name=pipe_name,
            pipe_geom=pipe_geom,
            physics=physics,
            first_ts_size=first_ts_size,
            segment_idx=segment_idx,
            inflow_or_outflow="inflow",
            target_molar_rate=target_molar_rate,
            ramp_up_period=ramp_up_period,
            inj_fluid_props=inj_fluid_props,
            verbose=False,
        )
        if pressure is not None:
            self.inj_fluid_props["pressure"] = float(pressure)
        if temperature is not None:
            self.inj_fluid_props["temperature"] = float(temperature)
        if phase_name is not None:
            self.inj_fluid_props["phase_name"] = phase_name

        if verbose:
            print(
                f'** UpstreamRampUpRate for the segment index {segment_idx} of the pipe "{pipe_geom.pipe_name}" is defined!'
            )

    @property
    def pressure(self) -> float:
        return self.inj_fluid_props.get("pressure")

    @property
    def temperature(self) -> float:
        return self.inj_fluid_props.get("temperature")

    @property
    def phase_name(self) -> str:
        return self.inj_fluid_props.get("phase_name")

    @property
    def composition(self) -> np.ndarray:
        return self.inj_fluid_props["composition"]

    def get_component_energy_rates(
        self,
        physics,
        specific_potential_energy: float = 0.0,
        molar_rate: float = None,
    ) -> np.ndarray:
        """
        Return component and energy rates in open-DARTS equation order.

        Component rates are in kmol/day. Energy rate is in kJ/day and includes
        potential energy when a segment-specific potential energy is provided.
        """
        rate = self.current_rate if molar_rate is None else molar_rate
        component_rate = rate * self.composition

        if not physics.thermal:
            return component_rate

        mw_avg = float(np.sum(physics.property_containers[0].Mw * self.composition))
        molar_potential_energy = specific_potential_energy * mw_avg
        molar_energy = self.inj_fluid_props["molar_enthalpy"] + molar_potential_energy

        return np.append(component_rate, rate * molar_energy)

    def get_boundary_momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        molar_rate: float = None,
    ) -> float:
        """
        Return A * sum(rho_phase * saturation_phase * velocity_phase**2).

        This is the inlet momentum term used by the DFM pipe momentum equation.
        It is evaluated from the properties of the upstream node, not
        from the receiving pipe segment.
        """
        rate = self.current_rate if molar_rate is None else molar_rate
        mw = np.asarray(property_container.Mw)
        mass_rate = float(np.sum(rate * self.composition * mw) / (24.0 * 60.0 * 60.0))

        if mass_rate == 0.0:
            return 0.0

        if self.phase_name == "G":
            rho = property_container.density_ev["G"].evaluate(
                self.pressure, self.temperature, self.composition
            )
            return mass_rate**2 / (rho * pipe_internal_area)

        if self.phase_name == "L":
            rho = property_container.density_ev["L"].evaluate(
                self.pressure, self.temperature, self.composition
            )
            return mass_rate**2 / (rho * pipe_internal_area)

        raise NotImplementedError(
            "UpstreamRampUpRate inlet momentum currently supports phase_name 'G' or 'L'."
        )
