import inspect

import numpy as np

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.ramp_up_rate import RampUpRate


class UpstreamRampUpRate(RampUpRate):
    """
    The same as RampUpRate, but it uses the upstream properties to calculate boundary momentum.

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
        self._pt_flash_supported = self._run_flash_supports_evaluate_pt(pc)

        inj_fluid_props = {"composition": composition}
        if physics.thermal:
            if molar_enthalpy is None and phase_mass_rates is None:
                inj_fluid_props.update(
                    {
                        "pressure": pressure,
                        "temperature": temperature,
                        "phase_name": phase_name,
                    }
                )
            elif molar_enthalpy is None:
                molar_enthalpy = self._evaluate_upstream_mixture_enthalpy(
                    pc,
                    float(pressure),
                    float(temperature),
                    composition,
                    phase_names,
                    self._pt_flash_supported,
                )
                inj_fluid_props["molar_enthalpy"] = float(molar_enthalpy)
            else:
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
        )
        if pressure is not None:
            self.inj_fluid_props["pressure"] = float(pressure)
        if temperature is not None:
            self.inj_fluid_props["temperature"] = float(temperature)
        if phase_name is not None:
            self.inj_fluid_props["phase_name"] = phase_name
        self.inj_fluid_props["phase_names"] = phase_names
        if phase_mass_rates is not None:
            self.inj_fluid_props["phase_mass_rates"] = phase_mass_rates

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
    def phase_name(self) -> str | None:
        return self.inj_fluid_props.get("phase_name")

    @property
    def phase_names(self) -> list[str]:
        return self.inj_fluid_props["phase_names"]

    @property
    def composition(self) -> np.ndarray:
        return self.inj_fluid_props["composition"]

    @staticmethod
    def _run_flash_supports_evaluate_pt(property_container) -> bool:
        """
        Probe once, at construction, whether run_flash accepts a PT flash via
        the evaluate_PT argument. A run_flash whose signature cannot be
        introspected is treated as not supporting it, so a required PT flash
        fails with a clear error instead of passing temperature into an
        enthalpy slot.
        """
        run_flash = getattr(property_container, "run_flash", None)
        if run_flash is None:
            return False
        try:
            parameters = inspect.signature(run_flash).parameters
        except (TypeError, ValueError):
            return False
        return "evaluate_PT" in parameters

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

    @classmethod
    def _evaluate_upstream_phase_props(
        cls,
        property_container,
        pressure,
        temperature,
        composition,
        phase_names,
        pt_flash_supported,
        evaluate_enthalpy=False,
    ):
        """
        Evaluate upstream equilibrium phase properties at the specified PT state.

        The returned phase saturation and mass fraction are used only for the
        upstream boundary momentum term. The property container is restored
        afterwards because it is a mutable object shared with the simulation.
        """
        saved = cls._copy_property_container_attrs(
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
            composition = np.asarray(composition, dtype=float)
            if not pt_flash_supported:
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

            return phase_props, float(molar_enthalpy)
        finally:
            cls._restore_property_container_attrs(property_container, saved)

    @classmethod
    def _evaluate_upstream_mixture_enthalpy(
        cls,
        property_container,
        pressure,
        temperature,
        composition,
        phase_names,
        pt_flash_supported,
    ):
        _, molar_enthalpy = cls._evaluate_upstream_phase_props(
            property_container,
            pressure,
            temperature,
            composition,
            phase_names,
            pt_flash_supported,
            evaluate_enthalpy=True,
        )
        return molar_enthalpy

    def _explicit_phase_mass_rates(self, mass_rate, molar_rate):
        phase_mass_rates = self.inj_fluid_props.get("phase_mass_rates")
        if phase_mass_rates is None:
            return None

        rate_scale = molar_rate / self.target_rate
        rates = {
            phase_name: float(phase_mass_rates.get(phase_name, 0.0)) * rate_scale
            for phase_name in self.phase_names
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

        if self.inj_fluid_props.get("phase_mass_rates") is None:
            phase_name = self.phase_name
            rho = property_container.density_ev[phase_name].evaluate(
                self.pressure,
                self.temperature,
                self.composition,
            )
            return mass_rate**2 / (rho * pipe_internal_area)

        upstream_phase_props, _ = self._evaluate_upstream_phase_props(
            property_container,
            self.pressure,
            self.temperature,
            self.composition,
            self.phase_names,
            self._pt_flash_supported,
        )

        phase_mass_rates = self._explicit_phase_mass_rates(mass_rate, rate)
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
