import numpy as np

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.ramp_up_rate import RampUpRate


class UpstreamMassNode(RampUpRate):
    """
    Upstream source node for DFM pipe injection.

    The specified pressure and temperature define the thermodynamic state of the
    injected stream. The pressure is not imposed as the pipe-segment pressure;
    it is used as the upstream mass-node state for injected enthalpy and inlet
    momentum.
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
        pressure: float,
        temperature: float,
        phase_name: str,
        apply_pressure_boundary: bool = False,
        verbose: bool = False,
    ):
        if not physics.thermal:
            raise ValueError("UpstreamMassNode currently requires thermal physics.")

        inj_fluid_props = {
            "composition": composition,
            "pressure": pressure,
            "temperature": temperature,
            "phase_name": phase_name,
        }

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
        self.apply_pressure_boundary = apply_pressure_boundary

        if verbose:
            print(
                f'** UpstreamMassNode for the segment index {segment_idx} of the pipe "{pipe_geom.pipe_name}" is defined!'
            )

    @property
    def pressure(self) -> float:
        return self.inj_fluid_props["pressure"]

    @property
    def temperature(self) -> float:
        return self.inj_fluid_props["temperature"]

    @property
    def phase_name(self) -> str:
        return self.inj_fluid_props["phase_name"]

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
        Return source rates in open-DARTS equation order.

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
        It is evaluated from the upstream mass-node pressure/temperature, not
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
            "UpstreamMassNode inlet momentum currently supports phase_name 'G' or 'L'."
        )
