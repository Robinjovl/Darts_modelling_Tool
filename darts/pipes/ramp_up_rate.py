"""Ramp-up rate sources for DFM pipes and their boundary property strategies.

A :class:`RampUpRate` is a molar source/sink attached to one segment of a
:class:`~darts.pipes.pipe.Pipe`. It owns

* the rate schedule (:meth:`RampUpRate.update_current_molar_rate`),
* the component/energy source a model adds to the residual of the segment
  block (:meth:`RampUpRate.get_component_energy_rates`),
* the inlet momentum term the DFM momentum equation needs at the boundary
  interface (:meth:`RampUpRate.get_boundary_momentum_flux`).

Where the boundary fluid properties behind the last two come from is the job of
a :class:`BoundaryPropertyModel` strategy, so that the copies of the boundary
momentum term (the inline block that used to live in ``pipe.py``, the
``UpstreamRampUpRate`` override and the choke override in the private models)
collapse onto one call path:

* :class:`SegmentProps` — properties of the RECEIVING segment; the plain
  ``RampUpRate`` default and the historical ``pipe.py`` behaviour,
* :class:`ConstantInjectedProps` — a single upstream phase at a constant
  (pressure, temperature, composition) state (``UpstreamRampUpRate`` with
  ``phase_name``),
* :class:`UpstreamFlashProps` — a multiphase upstream node whose phase split
  comes from ONE PT flash, evaluated when the strategy is bound
  (``UpstreamRampUpRate`` with ``phase_mass_rates``).

A strategy is bound once, from the :class:`RampUpRate` constructor, and that is
the only place it is allowed to touch the (shared, mutable) property container:
everything the per-timestep calls need is cached at bind time.
"""

import inspect
from dataclasses import dataclass

import numpy as np

from darts.pipes.define_pipe_geometry import PipeGeometry

SECONDS_PER_DAY = 24.0 * 60.0 * 60.0


def molar_to_mass_rate(molar_rate, composition, Mw) -> float:
    """Convert a molar rate to a mass rate — THE single conversion.

    :param molar_rate: molar rate [kmol/day]
    :param composition: overall composition [kmol/kmol]
    :param Mw: component molar weights [kg/kmol]
    :returns: mass rate [kg/s]
    """
    return float(
        np.sum(molar_rate * np.asarray(composition) * np.asarray(Mw)) / SECONDS_PER_DAY
    )


def _run_flash_supports_evaluate_pt(property_container) -> bool:
    """
    Probe once, at bind time, whether run_flash accepts a PT flash via the
    evaluate_PT argument. A run_flash whose signature cannot be introspected is
    treated as not supporting it, so a required PT flash fails with a clear
    error instead of passing temperature into an enthalpy slot.
    """
    run_flash = getattr(property_container, "run_flash", None)
    if run_flash is None:
        return False
    try:
        parameters = inspect.signature(run_flash).parameters
    except (TypeError, ValueError):
        return False
    return "evaluate_PT" in parameters


@dataclass(frozen=True)
class SegmentState:
    """Phase state of the pipe segment that receives a boundary source.

    Published by :class:`~darts.pipes.pipe.Pipe` (as
    :attr:`RampUpRate.receiving_segment_state`) before every boundary momentum
    evaluation, and consumed by :class:`SegmentProps`.

    :ivar sG: gas saturation of the segment [-]
    :ivar sL: mobile liquid saturation of the segment [-]
    :ivar rhoG: gas mass density of the segment [kg/m3]
    :ivar rhoL: mobile liquid mass density of the segment [kg/m3]
    """

    sG: float
    sL: float
    rhoG: float
    rhoL: float


class BoundaryPropertyModel:
    """Strategy supplying the boundary fluid properties of a :class:`RampUpRate`.

    A strategy answers two questions about the boundary of a pipe source/sink:

    * :meth:`molar_enthalpy` — the molar enthalpy carried by the injected
      stream (thermal physics only, ``None`` otherwise),
    * :meth:`momentum_flux` — the inlet momentum term
      ``A * sum(rho_phase * saturation_phase * velocity_phase**2)``.

    :param pressure: boundary pressure [bar], or ``None``
    :param temperature: boundary temperature [K], or ``None``
    :param phase_name: boundary phase name, or ``None``
    """

    def __init__(
        self,
        pressure: float = None,
        temperature: float = None,
        phase_name: str = None,
    ):
        self.pressure = None if pressure is None else float(pressure)
        self.temperature = None if temperature is None else float(temperature)
        self.phase_name = phase_name
        self.source = None
        self.property_container = None

    def bind(self, source, property_container):
        """Attach the strategy to its source and property container.

        Called once, from the :class:`RampUpRate` constructor. Strategies that
        need thermodynamic evaluations do them HERE and cache the results.

        :param source: the :class:`RampUpRate` owning this strategy
        :param property_container: the physics property container
        """
        self.source = source
        self.property_container = property_container

    @property
    def composition(self):
        """Overall composition of the injected fluid [kmol/kmol]."""
        return self.source.composition

    @property
    def phase_names(self) -> list:
        """Boundary phase names (empty when the boundary defines no phase)."""
        return [] if self.phase_name is None else [self.phase_name]

    def molar_enthalpy(self):
        """Molar enthalpy of the injected fluid [kJ/kmol].

        ``None`` when the boundary state does not define one (then the user
        must supply ``molar_enthalpy`` for thermal physics).
        """
        if self.phase_name is None or self.pressure is None or self.temperature is None:
            return None
        pc = self.property_container
        return pc.enthalpy_ev[self.phase_name].evaluate(
            self.pressure,
            self.temperature,
            self.composition[: pc.nc_fl],
        )

    def momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        mass_rate: float,
        molar_rate: float,
        segment_state: SegmentState = None,
    ) -> float:
        """Return ``A * sum(rho_phase * saturation_phase * velocity_phase**2)``.

        :param property_container: the physics property container
        :param pipe_internal_area: internal cross-section of the pipe [m2]
        :param mass_rate: boundary mass rate [kg/s]
        :param molar_rate: boundary molar rate [kmol/day]
        :param segment_state: state of the receiving segment, when published
        """
        raise NotImplementedError(
            f"{type(self).__name__}.momentum_flux() is not implemented"
        )


class SegmentProps(BoundaryPropertyModel):
    """Boundary momentum from the state of the RECEIVING pipe segment.

    The historical plain-:class:`RampUpRate` behaviour (relocated verbatim from
    ``pipe.py``): the boundary stream is assumed to enter with the phase split,
    densities and holdups of the segment it is injected into. The gas/liquid
    pair matches the drift-flux momentum closure, and the section assumes there
    is no solid phase in the source segment.

    ``pressure``/``temperature``/``phase_name`` are optional and used only by
    :meth:`BoundaryPropertyModel.molar_enthalpy` (the "Method 1" enthalpy of
    ``inj_fluid_props``); they play no part in the momentum term.
    """

    def momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        mass_rate: float,
        molar_rate: float,
        segment_state: SegmentState = None,
    ) -> float:
        if segment_state is None:
            raise ValueError(
                "SegmentProps needs the state of the receiving pipe segment. "
                "Pipe publishes it as RampUpRate.receiving_segment_state before "
                "evaluating the boundary momentum; a source evaluated outside a "
                "Pipe must set it or use an upstream boundary property model."
            )
        pipe_internal_A = pipe_internal_area

        # The props of the fluid of the segment on which the constant mass rate source is defined are used.
        sG0_source = segment_state.sG
        sL0_source = segment_state.sL
        rhoG0_source = segment_state.rhoG
        rhoL0_source = segment_state.rhoL

        has_mobile_liquid0 = rhoL0_source > 0 and sL0_source > 1e-12

        # This section is written under the assumption that there is no solid phase in the source.
        if sG0_source == 0 and has_mobile_liquid0:
            vG0_source = 0
            liquid_mass_fraction0 = 1
            liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
            vL0_source = liquid_mass_rate0 / rhoL0_source / (pipe_internal_A * 1)
        elif sG0_source == 1 or not has_mobile_liquid0:
            vL0_source = 0
            gas_mass_fraction0 = 1
            gas_mass_rate0 = mass_rate * gas_mass_fraction0
            vG0_source = gas_mass_rate0 / rhoG0_source / (pipe_internal_A * 1)
        elif sG0_source > 0 and has_mobile_liquid0:
            gas_mass_fraction0 = (
                sG0_source
                * rhoG0_source
                / (sG0_source * rhoG0_source + sL0_source * rhoL0_source)
            )
            gas_mass_rate0 = mass_rate * gas_mass_fraction0
            vG0_source = gas_mass_rate0 / rhoG0_source / (pipe_internal_A * sG0_source)

            liquid_mass_fraction0 = 1 - gas_mass_fraction0
            liquid_mass_rate0 = mass_rate * liquid_mass_fraction0
            vL0_source = (
                liquid_mass_rate0 / rhoL0_source / (pipe_internal_A * sL0_source)
            )
        else:
            raise Exception("sG0_source is out of correct range (from 0 to 1)!")

        return pipe_internal_A * (
            rhoG0_source * sG0_source * vG0_source**2
            + rhoL0_source * sL0_source * vL0_source**2
        )


class UpstreamBoundaryProps(BoundaryPropertyModel):
    """Common part of the strategies that evaluate the boundary momentum from
    the UPSTREAM node state instead of from the receiving segment.

    Subclasses implement :meth:`upstream_momentum_flux`; the zero-rate shortcut
    is shared.
    """

    def momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        mass_rate: float,
        molar_rate: float,
        segment_state: SegmentState = None,
    ) -> float:
        if mass_rate == 0.0:
            return 0.0
        return self.upstream_momentum_flux(
            property_container, pipe_internal_area, mass_rate, molar_rate
        )

    def upstream_momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        mass_rate: float,
        molar_rate: float,
    ) -> float:
        raise NotImplementedError(
            f"{type(self).__name__}.upstream_momentum_flux() is not implemented"
        )


class ConstantInjectedProps(UpstreamBoundaryProps):
    """Single upstream phase at a constant (pressure, temperature, composition).

    The boundary momentum is ``mass_rate**2 / (rho * A)`` with ``rho`` the
    density of the single upstream phase at the boundary state.
    """

    def __init__(self, pressure: float, temperature: float, phase_name: str):
        super().__init__(
            pressure=pressure, temperature=temperature, phase_name=phase_name
        )

    def upstream_momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        mass_rate: float,
        molar_rate: float,
    ) -> float:
        rho = property_container.density_ev[self.phase_name].evaluate(
            self.pressure,
            self.temperature,
            self.composition,
        )
        return mass_rate**2 / (rho * pipe_internal_area)


class UpstreamFlashProps(UpstreamBoundaryProps):
    """Multiphase upstream node: the phase split comes from ONE PT flash.

    The boundary state (pressure, temperature, composition) is constant, so the
    flash is evaluated once in :meth:`bind` and the per-phase density,
    saturation and mass fraction are cached; the per-timestep momentum
    evaluation is then pure arithmetic and never touches the shared property
    container.

    :param pressure: upstream pressure [bar]
    :param temperature: upstream temperature [K]
    :param phase_names: upstream mobile phase names
    :param phase_mass_rates: optional explicit upstream phase mass rates at the
        target source rate [kg/s]; when omitted the flash mass fractions split
        the total mass rate.
    """

    def __init__(
        self,
        pressure: float,
        temperature: float,
        phase_names,
        phase_mass_rates: dict = None,
    ):
        super().__init__(pressure=pressure, temperature=temperature, phase_name=None)
        self._phase_names = list(phase_names)
        self.phase_mass_rates = phase_mass_rates
        self.phase_props = None
        self._molar_enthalpy = None

    @property
    def phase_names(self) -> list:
        return self._phase_names

    def bind(self, source, property_container):
        super().bind(source, property_container)
        # One PT flash for the whole simulation: the boundary state is constant.
        self.phase_props, self._molar_enthalpy = self._evaluate_upstream_phase_props(
            evaluate_enthalpy=bool(source.physics.thermal)
        )

    def molar_enthalpy(self):
        return self._molar_enthalpy

    @staticmethod
    def _copy_property_container_attrs(property_container, attrs):
        saved = {}
        for attr in attrs:
            if not hasattr(property_container, attr):
                continue
            value = getattr(property_container, attr)
            saved[attr] = value.copy() if isinstance(value, np.ndarray) else value
        return saved

    @staticmethod
    def _restore_property_container_attrs(property_container, saved):
        for attr, value in saved.items():
            setattr(property_container, attr, value)

    def _evaluate_upstream_phase_props(self, evaluate_enthalpy=False):
        """
        Evaluate upstream equilibrium phase properties at the boundary PT state.

        The returned phase saturation and mass fraction are used only for the
        upstream boundary momentum term. The property container is restored
        afterwards because it is a mutable object shared with the simulation.
        """
        property_container = self.property_container
        pressure = self.pressure
        temperature = self.temperature
        phase_names = self._phase_names
        saved = self._copy_property_container_attrs(
            property_container,
            (
                "nu",
                "x",
                "x_mass",
                "ph",
                "temperature",
                "pressure",
                "dens",
                "dens_m",
                "sat",
                "mu",
            ),
        )
        try:
            composition = np.asarray(self.composition, dtype=float)
            if not _run_flash_supports_evaluate_pt(property_container):
                raise ValueError(
                    "The property container's run_flash does not accept the "
                    "evaluate_PT argument, so the PT flash required for the "
                    "upstream boundary state cannot be evaluated. Use a "
                    "property container whose run_flash supports evaluate_PT."
                )
            ph = property_container.run_flash(
                pressure,
                temperature,
                composition[: property_container.nc_fl],
                evaluate_PT=True,
            )
            nu = np.asarray(property_container.nu, dtype=float)
            x = np.asarray(property_container.x, dtype=float)
            mw = np.asarray(property_container.Mw[: property_container.nc_fl])

            phase_props = {}
            volumes = {}
            masses = {}
            molar_enthalpy = 0.0

            for phase_idx in ph:
                phase_name = property_container.phases_name[phase_idx]
                if phase_name not in phase_names:
                    if nu[phase_idx] > np.finfo(float).eps:
                        raise ValueError(
                            f"Upstream flash at pressure={pressure}, "
                            f"temperature={temperature}, composition="
                            f"{composition[: property_container.nc_fl].tolist()} "
                            f"produced phase '{phase_name}' with molar fraction "
                            f"{nu[phase_idx]}, which is not listed in phase_names="
                            f"{list(phase_names)}; its saturation and enthalpy "
                            "contributions cannot be silently dropped."
                        )
                    continue

                phase_comp = x[phase_idx, : property_container.nc_fl]
                phase_mw = float(np.sum(mw * phase_comp))
                density = float(
                    property_container.density_ev[phase_name].evaluate(
                        pressure,
                        temperature,
                        phase_comp,
                    )
                )
                if density <= 0.0:
                    raise ValueError(
                        f"Upstream phase {phase_name} density must be positive."
                    )

                molar_density = density / phase_mw
                volumes[phase_name] = nu[phase_idx] / molar_density
                masses[phase_name] = nu[phase_idx] * phase_mw
                if evaluate_enthalpy:
                    molar_enthalpy += nu[phase_idx] * float(
                        property_container.enthalpy_ev[phase_name].evaluate(
                            pressure,
                            temperature,
                            phase_comp,
                        )
                    )
                phase_props[phase_name] = {
                    "density": density,
                    "composition": phase_comp.copy(),
                }

            total_volume = sum(volumes.values())
            total_mass = sum(masses.values())
            if total_volume <= 0.0 or total_mass <= 0.0:
                raise ValueError("No mobile upstream phase was found at the PT state.")

            for phase_name, props in phase_props.items():
                props["saturation"] = volumes[phase_name] / total_volume
                props["mass_fraction"] = masses[phase_name] / total_mass

            return phase_props, float(molar_enthalpy) if evaluate_enthalpy else None
        finally:
            self._restore_property_container_attrs(property_container, saved)

    def _explicit_phase_mass_rates(self, mass_rate, molar_rate):
        phase_mass_rates = self.phase_mass_rates
        if phase_mass_rates is None:
            return None

        rate_scale = molar_rate / self.source.target_rate
        rates = {
            phase_name: float(phase_mass_rates.get(phase_name, 0.0)) * rate_scale
            for phase_name in self._phase_names
        }

        if not np.isclose(
            sum(rates.values()),
            mass_rate,
            rtol=1.0e-8,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Sum of upstream phase_mass_rates is not equal to the source mass rate."
            )
        return rates

    def upstream_momentum_flux(
        self,
        property_container,
        pipe_internal_area: float,
        mass_rate: float,
        molar_rate: float,
    ) -> float:
        upstream_phase_props = self.phase_props

        phase_mass_rates = self._explicit_phase_mass_rates(mass_rate, molar_rate)
        if phase_mass_rates is None:
            phase_mass_rates = {
                phase_name: mass_rate
                * upstream_phase_props[phase_name]["mass_fraction"]
                for phase_name in upstream_phase_props
            }

        momentum_flux = 0.0
        for phase_name, phase_mass_rate in phase_mass_rates.items():
            if phase_mass_rate == 0.0:
                continue
            if phase_name not in upstream_phase_props:
                raise ValueError(
                    f"Upstream phase {phase_name} is not present at the PT state."
                )
            phase_props = upstream_phase_props[phase_name]
            density = phase_props["density"]
            saturation = phase_props["saturation"]
            if saturation <= 0.0:
                raise ValueError(
                    f"Upstream phase {phase_name} saturation must be positive."
                )
            momentum_flux += phase_mass_rate**2 / (
                density * saturation * pipe_internal_area
            )

        return momentum_flux


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
        verbose: bool = False,
        boundary_props: BoundaryPropertyModel = None,
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
        :param verbose: Whether to display extra info about RampUpRate
        :type verbose: bool
        :param boundary_props: Strategy supplying the boundary fluid properties
            (enthalpy and momentum). Defaults to :class:`SegmentProps`, i.e. the
            properties of the receiving segment. When a strategy is given, the
            boundary state (pressure, temperature, phase) lives in it and must
            not be repeated in inj_fluid_props.
        :type boundary_props: BoundaryPropertyModel
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
        pc = physics.property_containers[0]

        # State of the segment receiving this source, published by Pipe before
        # every boundary momentum evaluation (consumed by SegmentProps).
        self.receiving_segment_state = None

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

            if boundary_props is not None:
                # The boundary state belongs to the strategy (S12): it must not
                # be smuggled through inj_fluid_props as well.
                assert all(
                    i not in inj_fluid_props
                    for i in ("pressure", "temperature", "phase_name")
                ), (
                    "When boundary_props is specified, the boundary pressure, temperature and "
                    "phase_name belong to the boundary property model, not to inj_fluid_props!"
                )
                if "molar_enthalpy" in inj_fluid_props:
                    assert isinstance(inj_fluid_props["molar_enthalpy"], float), (
                        "Specified molar_enthalpy must be a float!"
                    )
            elif physics.thermal:
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
                    assert inj_fluid_props["temperature"] > 200.0, (
                        "Specified temperature must be in Kelvin!"
                    )

                if "phase_name" in inj_fluid_props:
                    ph_name = inj_fluid_props["phase_name"]
                    assert isinstance(ph_name, str), (
                        "The specified phase is not a string!"
                    )
                    assert ph_name in pc.phases_name[: pc.np_fl], (
                        f'The specified phase "{ph_name}" is not in the list of mobile phases defined in the physics!'
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
            assert physics.state_spec == physics.StateSpecification.PH, (
                "Thermal production only work with the PH formulation for multiphase flow accurately!"
            )

        self.inj_fluid_props = inj_fluid_props

        if boundary_props is None:
            # Default: the boundary momentum uses the receiving segment state.
            # Any "Method 1" state in inj_fluid_props only feeds the enthalpy.
            props = inj_fluid_props if inj_fluid_props is not None else {}
            boundary_props = SegmentProps(
                pressure=props.get("pressure"),
                temperature=props.get("temperature"),
                phase_name=props.get("phase_name"),
            )
        self.boundary_props = boundary_props
        boundary_props.bind(self, pc)

        if (
            inflow_or_outflow == "inflow"
            and physics.thermal
            and "molar_enthalpy" not in inj_fluid_props
        ):
            # Calculate and store injected_fluid_molar_enthalpy
            injected_fluid_molar_enthalpy = boundary_props.molar_enthalpy()
            if injected_fluid_molar_enthalpy is None:
                raise ValueError(
                    f"{type(boundary_props).__name__} does not define a molar "
                    "enthalpy for the injected fluid, so molar_enthalpy must be "
                    "specified in inj_fluid_props for thermal physics."
                )
            inj_fluid_props["molar_enthalpy"] = injected_fluid_molar_enthalpy

        # If the user sets ramp_up_period to zero means that they don't want to have a ramp-up rate
        if ramp_up_period == 0.0:
            self.current_rate = target_molar_rate
        # Rate starts from zero, so initial rate is zero:
        elif ramp_up_period > 0.0:
            self.current_rate = 0.0

        if verbose:
            print(
                f'** RampUpRate for the segment index {segment_idx} of the pipe "{pipe_geom.pipe_name}" is defined!'
            )

    @property
    def composition(self):
        """Overall composition of the injected fluid [kmol/kmol] (``None`` for an outflow)."""
        if self.inj_fluid_props is None:
            return None
        return self.inj_fluid_props["composition"]

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

    def get_component_energy_rates(
        self,
        physics=None,
        specific_potential_energy: float = 0.0,
        molar_rate: float = None,
    ) -> np.ndarray:
        """
        Return component and energy rates in open-DARTS equation order.

        Component rates are in kmol/day. Energy rate is in kJ/day and includes
        potential energy when a segment-specific potential energy is provided.
        For isothermal physics only the component rates are returned.

        :param physics: physics object; defaults to the one of this source
        :param specific_potential_energy: specific potential energy of the
            receiving block [kJ/kg]
        :param molar_rate: molar rate to evaluate at; defaults to the current
            ramp-up rate [kmol/day]
        """
        if self.inj_fluid_props is None:
            raise ValueError(
                "get_component_energy_rates is defined for an inflow source only: "
                "the composition of an outflow is not known to the source."
            )
        physics = self.physics if physics is None else physics
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
        Which properties it is evaluated from is decided by the
        :class:`BoundaryPropertyModel` of this source: the receiving segment
        (default) or the upstream node.

        The receiving segment state, needed by :class:`SegmentProps` only, is
        read from :attr:`receiving_segment_state`, which Pipe publishes before
        each evaluation, so that this signature stays the one subclasses
        override (and call back into through ``super()``).
        """
        rate = self.current_rate if molar_rate is None else molar_rate

        if self.inflow_or_outflow == "inflow":
            # comp_source in kmol/kmol, mass_rate in kg/s
            mass_rate = molar_to_mass_rate(
                rate, self.composition, property_container.Mw
            )
        else:
            # TODO: For outflow, we have rate_source, which is in kmol/day, but we don't have comp_source, which
            # is in kmol/kmol, from the user. Instead, we have xG_mass0 and xL_mass0, which are mass fractions.
            # Need to see how we can get the overall composition of the source block in kmol/kmol.
            mass_rate = 0

        return self.boundary_props.momentum_flux(
            property_container,
            pipe_internal_area,
            mass_rate,
            rate,
            self.receiving_segment_state,
        )
