from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.ramp_up_rate import (
    ConstantInjectedProps,
    RampUpRate,
    UpstreamFlashProps,
)


class UpstreamRampUpRate(RampUpRate):
    """
    The same as RampUpRate, but it uses the upstream properties to calculate boundary momentum.

    Unlike RampUpRate, which uses the properties of the pipe segment to calculate boundary momentum,
    this class uses the properties of the upstream node to calculate boundary momentum.

    This is a thin adapter: the upstream boundary state lives in the
    :class:`~darts.pipes.ramp_up_rate.BoundaryPropertyModel` this constructor
    builds — :class:`~darts.pipes.ramp_up_rate.ConstantInjectedProps` for a
    single upstream phase, :class:`~darts.pipes.ramp_up_rate.UpstreamFlashProps`
    when explicit ``phase_mass_rates`` are given — and the momentum/enthalpy
    machinery is the one of the base class.
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
        phase_name: str | None = None,
        phase_mass_rates: dict[str, float] | None = None,
        molar_enthalpy: float = None,
        verbose: bool = False,
    ):
        """
        In RampUpRate, if molar_enthalpy is provided, pressure, temperature, and phase_name must not be specified,
        but in UpstreamRampUpRate, pressure, temperature, and phase_name must be specified all the time because
        they are needed to calculate the boundary momentum.

        :param phase_name: Single upstream mobile phase name used when
            phase_mass_rates is None. Accepted inputs are None or a string, e.g. "G" or "L".
            Each name must be present in property_container.phases_name[:property_container.np_fl].
        :param phase_mass_rates: Optional explicit upstream phase mass rates at
            the target source rate [kg/s]. Accepted inputs are None or a
            non-empty dictionary keyed by phase name, e.g.
            {"G": gas_rate, "L": liquid_rate}. If phase_mass_rates is provided,
            phase_name must be None and the upstream phase names are inferred
            from the dictionary keys. The sum of phase_mass_rates must match the
            total mass rate implied by target_molar_rate and composition.
        """
        if phase_name is not None and phase_mass_rates is not None:
            raise ValueError("Specify either phase_name or phase_mass_rates, not both.")

        if phase_mass_rates is not None and not isinstance(phase_mass_rates, dict):
            raise TypeError("phase_mass_rates must be None or a dictionary.")

        if phase_mass_rates is not None:
            if not phase_mass_rates:
                raise ValueError("phase_mass_rates must contain at least one phase.")
            if not all(isinstance(key, str) for key in phase_mass_rates):
                raise TypeError("phase_mass_rates keys must be strings.")
            phase_names = list(phase_mass_rates)
        else:
            if phase_name is None:
                raise ValueError(
                    "phase_name must define at least one upstream phase when "
                    "phase_mass_rates is not provided."
                )
            if not isinstance(phase_name, str):
                raise TypeError("phase_name must be None or a string.")
            phase_names = [phase_name]

        if pressure is None or temperature is None:
            raise ValueError(
                "pressure and temperature must be provided for upstream boundary momentum."
            )
        pc = physics.property_containers[0]
        unknown_phase_names = set(phase_names) - set(pc.phases_name[: pc.np_fl])
        if unknown_phase_names:
            raise ValueError(
                "Unknown upstream phase names: "
                + ", ".join(sorted(unknown_phase_names))
            )

        if phase_mass_rates is None:
            boundary_props = ConstantInjectedProps(
                pressure=pressure,
                temperature=temperature,
                phase_name=phase_name,
            )
        else:
            boundary_props = UpstreamFlashProps(
                pressure=pressure,
                temperature=temperature,
                phase_names=phase_names,
                phase_mass_rates=phase_mass_rates,
            )

        inj_fluid_props = {"composition": composition}
        if physics.thermal:
            if molar_enthalpy is not None:
                inj_fluid_props["molar_enthalpy"] = float(molar_enthalpy)
        else:
            if molar_enthalpy is not None:
                raise ValueError(
                    "molar_enthalpy must not be specified for isothermal upstream boundaries."
                )

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
            boundary_props=boundary_props,
        )

        if verbose:
            print(
                f'** UpstreamRampUpRate for the segment index {segment_idx} of the pipe "{pipe_geom.pipe_name}" is defined!'
            )

    @property
    def pressure(self) -> float:
        return self.boundary_props.pressure

    @property
    def temperature(self) -> float:
        return self.boundary_props.temperature

    @property
    def phase_name(self) -> str | None:
        return self.boundary_props.phase_name

    @property
    def phase_names(self) -> list[str]:
        return self.boundary_props.phase_names

    @property
    def phase_mass_rates(self) -> dict[str, float] | None:
        return getattr(self.boundary_props, "phase_mass_rates", None)
