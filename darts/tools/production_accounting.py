"""
Numerical production-accounting methods for :class:`darts.models.output.Output`.

One module deliberately contains both the DataFrame arithmetic and the
DARTS-aware inventory/flash calculations. The public API is
``Output`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

VALID_WELL_ROLES = {"producer", "injector"}


def _working_frame(time_data, inplace: bool) -> pd.DataFrame:
    if isinstance(time_data, pd.DataFrame):
        return time_data if inplace else time_data.copy(deep=True)
    if isinstance(time_data, Mapping):
        return pd.DataFrame(time_data)
    raise TypeError("time_data must be a pandas DataFrame or mapping of columns")


def _require_columns(time_data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in time_data.columns]
    if missing:
        raise KeyError("Missing required time-data columns: " + ", ".join(missing))


def validate_time(time_data: pd.DataFrame) -> np.ndarray:
    """
    Return a validated, strictly increasing time vector in days.

    :param time_data: Well time-series data containing a ``time`` column.
    :type time_data: pandas.DataFrame
    :return: Time vector [day].
    :rtype: numpy.ndarray
    """
    _require_columns(time_data, ["time"])
    time = time_data["time"].to_numpy(dtype=float)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("time must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(time)):
        raise ValueError("time contains non-finite values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return time


def cumulative_integral(
    rate: Sequence[float],
    time: Sequence[float],
    method: str = "trapezoidal",
) -> np.ndarray:
    """
    Integrate a rate time series using the requested reporting convention.

    :param rate: Rate values.
    :type rate: Sequence[float]
    :param time: Strictly increasing reporting times.
    :type time: Sequence[float]
    :param method: ``trapezoidal``, ``left``, or ``right`` integration.
    :type method: str
    :return: Cumulative integral beginning at zero.
    :rtype: numpy.ndarray
    """
    rate = np.asarray(rate, dtype=float)
    time = np.asarray(time, dtype=float)
    if rate.shape != time.shape:
        raise ValueError("rate and time must have the same shape")
    if rate.ndim != 1:
        raise ValueError("rate and time must be one-dimensional")
    if not np.all(np.isfinite(rate)):
        raise ValueError("rate contains non-finite values")
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ValueError("time must be strictly increasing")
    method = method.lower()
    if method == "trapezoidal":
        increments = 0.5 * (rate[:-1] + rate[1:]) * dt
    elif method == "left":
        increments = rate[:-1] * dt
    elif method == "right":
        increments = rate[1:] * dt
    else:
        raise ValueError("integration must be 'trapezoidal', 'left', or 'right'")
    return np.concatenate(([0.0], np.cumsum(increments)))


def _resolve_well_roles(
    time_data: pd.DataFrame,
    well_names: Sequence[str],
    component_names: Sequence[str],
    well_roles: Mapping[str, str] | None = None,
    tolerance: float = 1e-12,
) -> dict[str, str]:
    """
    Validate roles or infer fixed roles from signed component molar rates.

    :return: Mapping of every well to ``producer`` or ``injector``.
    :rtype: dict[str, str]
    """
    names = list(well_names)
    if well_roles is not None:
        roles = {str(name): str(role).lower() for name, role in well_roles.items()}
        missing = [name for name in names if name not in roles]
        extra = [name for name in roles if name not in names]
        invalid = {
            name: role for name, role in roles.items() if role not in VALID_WELL_ROLES
        }
        if missing or extra or invalid:
            raise ValueError(
                f"Invalid well_roles: missing={missing}, unknown={extra}, invalid={invalid}"
            )
        return {name: roles[name] for name in names}
    roles = {}
    for well in names:
        columns = [
            f"well_{well}_molar_rate_{component}_by_sum_perfs"
            for component in component_names
        ]
        _require_columns(time_data, columns)
        total_rate = time_data[columns].sum(axis=1).to_numpy(dtype=float)
        has_injection = np.any(total_rate > tolerance)
        has_production = np.any(total_rate < -tolerance)
        if has_injection and has_production:
            raise ValueError(
                f"Well {well!r} changes flow direction; provide well_roles explicitly."
            )
        if not has_injection and not has_production:
            raise ValueError(
                f"Cannot infer a role for zero-rate well {well!r}; provide well_roles explicitly."
            )
        roles[well] = "injector" if has_injection else "producer"
    return roles


def _calculate_field_totals(
    time_data: pd.DataFrame,
    well_roles: Mapping[str, str],
    phase_names: Sequence[str],
    integration: str = "trapezoidal",
    conditions: Sequence[str] = ("rc", "sc"),
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add field phase rates and cumulative gross, injected, and net volumes.

    :return: Enriched time data.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(time_data, inplace)
    time = validate_time(result)
    invalid = {
        well: role for well, role in well_roles.items() if role not in VALID_WELL_ROLES
    }
    if invalid:
        raise ValueError(f"Invalid well roles: {invalid}")
    for condition in conditions:
        if condition not in {"rc", "sc"}:
            raise ValueError("conditions may contain only 'rc' and 'sc'")
        suffix = "at_wh" if condition == "rc" else "at_wh_sc"
        for phase in phase_names:
            columns = {
                well: f"well_{well}_volumetric_rate_{phase}_{suffix}"
                for well in well_roles
            }
            _require_columns(result, list(columns.values()))
            production_rate = np.zeros(len(result), dtype=float)
            injection_rate = np.zeros(len(result), dtype=float)
            for well, column in columns.items():
                signed_rate = result[column].to_numpy(dtype=float)
                if not np.all(np.isfinite(signed_rate)):
                    raise ValueError(f"Column {column!r} contains non-finite values")
                if well_roles[well] == "producer":
                    production_rate -= signed_rate
                else:
                    injection_rate += signed_rate
            net_rate = production_rate - injection_rate
            result[f"field_{phase}_production_rate_{condition}"] = production_rate
            result[f"field_{phase}_injection_rate_{condition}"] = injection_rate
            result[f"field_{phase}_net_production_rate_{condition}"] = net_rate
            for direction, rate_values in (
                ("production", production_rate),
                ("injection", injection_rate),
                ("net_production", net_rate),
            ):
                result[f"cumulative_{phase}_{direction}_{condition}"] = (
                    cumulative_integral(rate_values, time, integration)
                )
    production = (
        result[[f"field_{phase}_production_rate_rc" for phase in phase_names]]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    injection = (
        result[[f"field_{phase}_injection_rate_rc" for phase in phase_names]]
        .sum(axis=1)
        .to_numpy(dtype=float)
    )
    result["VRR"] = np.divide(
        injection,
        production,
        out=np.full(len(result), np.nan, dtype=float),
        where=production > 0.0,
    )
    return result


def _calculate_injected_pore_volumes(
    field_totals: pd.DataFrame,
    pore_volume: float,
    phase_names: Sequence[str],
    volume_condition: str = "rc",
    column_name: str = "injected_pv",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add cumulative injected pore volumes from field totals.

    :return: Enriched field-total data.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(field_totals, inplace)
    pore_volume = float(pore_volume)
    if not np.isfinite(pore_volume) or pore_volume <= 0.0:
        raise ValueError("pore_volume must be finite and positive")
    if volume_condition not in {"rc", "sc"}:
        raise ValueError("volume_condition must be 'rc' or 'sc'")
    columns = [
        f"cumulative_{phase}_injection_{volume_condition}" for phase in phase_names
    ]
    _require_columns(result, columns)
    result[column_name] = (
        result[columns].sum(axis=1).to_numpy(dtype=float) / pore_volume
    )
    return result


def calculate_production_gor(
    time_data: pd.DataFrame,
    producer_wells: Sequence[str],
    gas_phase: str = "gas",
    oil_phase: str = "oil",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add standard-condition GOR for producers and for the field.

    :return: Enriched time data.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(time_data, inplace)
    gas_total = np.zeros(len(result), dtype=float)
    oil_total = np.zeros(len(result), dtype=float)
    required = []
    for well in producer_wells:
        required.extend(
            [
                f"well_{well}_volumetric_rate_{gas_phase}_at_wh_sc",
                f"well_{well}_volumetric_rate_{oil_phase}_at_wh_sc",
            ]
        )
    _require_columns(result, required)
    for well in producer_wells:
        gas_rate = np.abs(result[f"well_{well}_volumetric_rate_{gas_phase}_at_wh_sc"])
        oil_rate = np.abs(result[f"well_{well}_volumetric_rate_{oil_phase}_at_wh_sc"])
        gor = np.divide(
            gas_rate,
            oil_rate,
            out=np.full(len(result), np.nan, dtype=float),
            where=oil_rate > 0.0,
        )
        result[f"well_{well}_GOR_m3_m3"] = gor
        result[f"well_{well}_GOR_scf_bbl"] = gor * (35.31467 / 6.289811)
        gas_total += gas_rate
        oil_total += oil_rate
    field_gor = np.divide(
        gas_total,
        oil_total,
        out=np.full(len(result), np.nan, dtype=float),
        where=oil_total > 0.0,
    )
    result["field_GOR_m3_m3"] = field_gor
    result["field_GOR_scf_bbl"] = field_gor * (35.31467 / 6.289811)
    return result


def _calculate_volume_recovery_factors(
    field_totals: pd.DataFrame,
    simple_stoiip: float,
    compositional_stoiip: float,
    oil_phase: str = "oil",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add gross/net simple and compositional volume-based oil RFs.

    :return: Enriched field-total data.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(field_totals, inplace)
    produced_column = f"cumulative_{oil_phase}_production_sc"
    injected_column = f"cumulative_{oil_phase}_injection_sc"
    _require_columns(result, [produced_column, injected_column])
    produced = result[produced_column].to_numpy(dtype=float)
    injected = result[injected_column].to_numpy(dtype=float)
    for method, stoiip in (
        ("simple", float(simple_stoiip)),
        ("compositional", float(compositional_stoiip)),
    ):
        if not np.isfinite(stoiip) or stoiip <= 0.0:
            raise ValueError(f"{method}_stoiip must be finite and positive")
        result[f"RF_volume_gross_{method}"] = produced / stoiip
        result[f"RF_volume_net_{method}"] = (produced - injected) / stoiip
        result[f"STOIIP_{method}"] = stoiip
    return result


def _calculate_mass_recovery_factors(
    time_data: pd.DataFrame,
    initial_component_masses: Mapping[str, float],
    well_roles: Mapping[str, str],
    component_names: Sequence[str],
    integration: str = "trapezoidal",
) -> pd.DataFrame:
    """
    Return gross and net mass-based component recovery factors.

    :return: Component mass-accounting data.
    :rtype: pandas.DataFrame
    """
    time = validate_time(time_data)
    result = pd.DataFrame({"time": time})
    for component in component_names:
        columns = {
            well: f"well_{well}_mass_rate_{component}_by_sum_perfs"
            for well in well_roles
        }
        _require_columns(time_data, list(columns.values()))
        production_rate = np.zeros(len(time_data), dtype=float)
        injection_rate = np.zeros(len(time_data), dtype=float)
        for well, column in columns.items():
            signed_rate = time_data[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(signed_rate)):
                raise ValueError(f"Column {column!r} contains non-finite values")
            if well_roles[well] == "producer":
                production_rate -= signed_rate
            else:
                injection_rate += signed_rate
        produced = cumulative_integral(production_rate, time, integration)
        injected = cumulative_integral(injection_rate, time, integration)
        initial_mass = float(initial_component_masses[component])
        if not np.isfinite(initial_mass) or initial_mass <= 0.0:
            raise ValueError(
                f"Initial mass for {component!r} must be finite and positive"
            )
        result[f"field_{component}_production_mass_rate"] = production_rate
        result[f"field_{component}_injection_mass_rate"] = injection_rate
        result[f"cumulative_produced_mass_{component}"] = produced
        result[f"cumulative_injected_mass_{component}"] = injected
        result[f"cumulative_net_produced_mass_{component}"] = produced - injected
        result[f"initial_mass_{component}"] = initial_mass
        result[f"RF_mass_gross_{component}"] = produced / initial_mass
        result[f"RF_mass_net_{component}"] = (produced - injected) / initial_mass
    return result


@dataclass(frozen=True)
class InitialInventory:
    """
    Initial reservoir fluid inventory used by recovery-factor calculations.

    :param pore_volume: Total initial reservoir pore volume [m3].
    :type pore_volume: float
    :param phase_volumes_reservoir: Initial phase volumes at reservoir conditions [m3].
    :type phase_volumes_reservoir: dict[str, float]
    :param component_masses: Initial component masses across all fluid phases [kg].
    :type component_masses: dict[str, float]
    :param oil_volume_standard: Initial oil volume after a standard-condition flash [m3].
    :type oil_volume_standard: float
    """

    pore_volume: float
    phase_volumes_reservoir: dict[str, float]
    component_masses: dict[str, float]
    oil_volume_standard: float


def _containers(self) -> dict:
    containers = self.physics.property_containers
    if not containers:
        raise ValueError("Production accounting requires a property container.")
    return containers


def _fluid_definition(self) -> tuple[list[str], list[str]]:
    containers = self._containers()
    first = next(iter(containers.values()))
    components = list(first.components_name[: first.nc_fl])
    phases = list(first.phases_name[: first.np_fl])
    for region, container in containers.items():
        region_components = list(container.components_name[: container.nc_fl])
        region_phases = list(container.phases_name[: container.np_fl])
        if region_components != components or region_phases != phases:
            raise ValueError(
                "Production accounting requires identical fluid component and "
                f"phase ordering in every property region; region {region!r} differs."
            )
    return components, phases


def _well_names(self) -> list[str]:
    return [well.name for well in self.reservoir.wells]


def _pore_volumes(self) -> np.ndarray:
    n_cells = int(self.reservoir.n)
    volumes = np.asarray(self.reservoir.mesh.volume, dtype=float)[:n_cells]
    porosity = np.asarray(self.reservoir.mesh.poro, dtype=float)[:n_cells]
    if volumes.size != n_cells or porosity.size != n_cells:
        raise ValueError("Volume and porosity are required for every reservoir cell.")
    pore_volumes = volumes * porosity
    if not np.all(np.isfinite(pore_volumes)) or np.any(pore_volumes < 0.0):
        raise ValueError("Reservoir pore volumes must be finite and non-negative.")
    if np.sum(pore_volumes) <= 0.0:
        raise ValueError("Total reservoir pore volume must be positive.")
    return pore_volumes


def _region_ids(self) -> np.ndarray:
    n_cells = int(self.reservoir.n)
    region_ids = np.asarray(self.op_num, dtype=int)[:n_cells]
    if region_ids.size != n_cells:
        raise ValueError("A property-region number is required for every cell.")
    missing = sorted(set(region_ids) - set(self._containers()))
    if missing:
        raise ValueError(f"Missing property-container regions: {missing}.")
    return region_ids


def _resolve_well_regions(
    self, well_regions: Mapping[str, int] | None
) -> dict[str, int]:
    wells = list(self.reservoir.wells)
    containers = self._containers()
    if well_regions is not None:
        regions = {str(name): int(region) for name, region in well_regions.items()}
        missing = [well.name for well in wells if well.name not in regions]
        extra = [name for name in regions if name not in self._well_names()]
        invalid = {
            name: region for name, region in regions.items() if region not in containers
        }
        if missing or extra or invalid:
            raise ValueError(
                "Invalid well_regions: "
                f"missing={missing}, unknown={extra}, unavailable={invalid}."
            )
        return regions

    cell_regions = self._region_ids()
    regions = {}
    for well in wells:
        perforation_cells = [int(perf[1]) for perf in well.perforations]
        if not perforation_cells:
            raise ValueError(
                f"Well {well.name!r} has no perforations; provide well_regions."
            )
        if min(perforation_cells) < 0 or max(perforation_cells) >= len(cell_regions):
            raise ValueError(
                f"Well {well.name!r} references an invalid reservoir cell."
            )
        connected_regions = set(cell_regions[perforation_cells].tolist())
        if len(connected_regions) != 1:
            raise ValueError(
                f"Well {well.name!r} connects property regions "
                f"{sorted(connected_regions)}; provide one well_regions entry."
            )
        regions[well.name] = connected_regions.pop()
    return regions


def _standard_phase_state(
    container, pressure: float, temperature: float, composition: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # run_flash mutates temperature and phase arrays. In isothermal physics
    # temperature is also the next reservoir evaluation's input, not just output.
    previous_temperature = container.temperature
    previous_x, previous_nu = container.x, container.nu
    saved_x, saved_nu = previous_x.copy(), previous_nu.copy()
    try:
        return _evaluate_standard_phase_state(
            container, pressure, temperature, composition
        )
    finally:
        container.temperature = previous_temperature
        previous_x[:] = saved_x
        previous_nu[:] = saved_nu
        container.x, container.nu = previous_x, previous_nu


def _evaluate_standard_phase_state(
    container, pressure: float, temperature: float, composition: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray(
        container.run_flash(
            pressure,
            temperature,
            composition,
            evaluate_PT=bool(getattr(container, "evaluate_PT_bool", False)),
        ),
        dtype=int,
    )
    nu = np.asarray(container.nu, dtype=float).copy()
    x = np.asarray(container.x, dtype=float).copy()
    dens_m = np.zeros(container.np_fl, dtype=float)
    molecular_weights = np.asarray(container.Mw[: container.nc_fl], dtype=float)
    for phase_index in active:
        mixture_mw = float(np.dot(molecular_weights, x[phase_index]))
        if not np.isfinite(mixture_mw) or mixture_mw <= 0.0:
            raise ValueError("standard-condition phase molecular weight is invalid.")
        phase = container.phases_name[phase_index]
        density = float(
            container.density_ev[phase].evaluate(pressure, temperature, x[phase_index])
        )
        dens_m[phase_index] = density / mixture_mw
        if not np.isfinite(dens_m[phase_index]) or dens_m[phase_index] <= 0.0:
            raise ValueError(
                f"standard-condition molar density for phase {phase!r} is invalid."
            )
    return active, nu, x, dens_m


def resolve_well_roles(
    self,
    time_data,
    well_roles: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Validate explicit roles or infer them from signed component molar rates.

    :param time_data: Well time-series data.
    :type time_data: pandas.DataFrame
    :param well_roles: Optional mapping of well names to producer/injector.
    :type well_roles: Mapping[str, str], optional
    :return: Validated well-role mapping.
    :rtype: dict[str, str]
    """
    components, _ = self._fluid_definition()
    return _resolve_well_roles(
        pd.DataFrame(time_data), self._well_names(), components, well_roles
    )


def convert_rates_to_standard_conditions(
    self,
    time_data,
    standard_pressure: float = 1.013235,
    standard_temperature: float = 288.15,
    well_regions: Mapping[str, int] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Flash signed component molar well rates to phase volumes at standard conditions.

    :param time_data: Well time-series data containing component molar rates.
    :type time_data: pandas.DataFrame or Mapping
    :param standard_pressure: Standard pressure [bar].
    :type standard_pressure: float
    :param standard_temperature: Standard temperature [K].
    :type standard_temperature: float
    :param well_regions: Optional property-container region for each well.
    :type well_regions: Mapping[str, int], optional
    :param inplace: Modify an input DataFrame when True.
    :type inplace: bool
    :return: Time data with standard-condition phase-volume rates.
    :rtype: pandas.DataFrame
    """
    components, phases = self._fluid_definition()
    standard_columns = [
        f"well_{well}_volumetric_rate_{phase}_at_wh_sc"
        for well in self._well_names()
        for phase in phases
    ]
    result = _working_frame(time_data, inplace)
    if all(column in result for column in standard_columns):
        return result
    regions = self._resolve_well_regions(well_regions)
    n_times = len(result)

    for well in self.reservoir.wells:  # calculate rates per well!
        columns = [
            f"well_{well.name}_molar_rate_{component}_by_sum_perfs"
            for component in components
        ]
        missing = [column for column in columns if column not in result]
        if missing:
            raise KeyError("Missing required time-data columns: " + ", ".join(missing))
        component_rates = result[columns].to_numpy(dtype=float).T
        if not np.all(np.isfinite(component_rates)):
            raise ValueError(f"Component rates for well {well.name!r} are non-finite.")
        total_rate = np.sum(
            component_rates, axis=0
        )  # total number of moles being produced / day
        phase_rates = {phase: np.zeros(n_times, dtype=float) for phase in phases}
        container = self._containers()[regions[well.name]]

        # calcualte molar fraction
        for component_index, component in enumerate(components):
            result[f"well_{well.name}_molar_fraction_{component}"] = np.divide(
                component_rates[component_index],
                total_rate,
                out=np.zeros(n_times, dtype=float),
                where=np.abs(total_rate) > 0.0,
            )  # convert molar rates to molar fractions

        # per time step take the molar fraction and flash it at standard conditiosn
        for time_index, signed_total_rate in enumerate(total_rate):
            if abs(signed_total_rate) <= 0.0:
                continue
            composition = component_rates[:, time_index] / signed_total_rate
            if np.any(composition < -1e-10) or not np.isclose(np.sum(composition), 1.0):
                raise ValueError(
                    f"Component rates for well {well.name!r} do not define a valid "
                    f"composition at row {time_index}."
                )
            composition = np.maximum(composition, 0.0)
            composition /= np.sum(composition)  # zc
            active, nu, _, dens_m = self._standard_phase_state(
                container,
                float(standard_pressure),
                float(standard_temperature),
                composition,
            )  # determine what phases the molar stream flashes into at standard conditions
            for phase_index in active:
                phase = phases[phase_index]
                phase_rates[phase][time_index] = (
                    signed_total_rate * nu[phase_index] / dens_m[phase_index]
                )  # sum(comp_rates) * phase_fraction / molar_phase_density --> (mol/day)/(mol/sm3) = sm3/day

        for phase, rates in phase_rates.items():
            result[f"well_{well.name}_volumetric_rate_{phase}_at_wh_sc"] = rates

    return result


def calculate_field_totals(
    self,
    time_data,
    well_roles: Mapping[str, str] | None = None,
    standard_pressure: float = 1.013235,
    standard_temperature: float = 288.15,
    well_regions: Mapping[str, int] | None = None,
    integration: str = "trapezoidal",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Calculate field phase rates and cumulative volumes.

    Standard-condition rates are calculated only when they are absent.

    :return: Time data with reservoir- and standard-condition field totals.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(time_data, inplace)
    _, phases = self._fluid_definition()
    standard_columns = [
        f"well_{well}_volumetric_rate_{phase}_at_wh_sc"
        for well in self._well_names()
        for phase in phases
    ]
    if not all(column in result for column in standard_columns):
        result = self.convert_rates_to_standard_conditions(
            result, standard_pressure, standard_temperature, well_regions, True
        )
    roles = self.resolve_well_roles(result, well_roles)
    return _calculate_field_totals(result, roles, phases, integration, inplace=True)


def calculate_injected_pore_volumes(
    self,
    time_data,
    initial_inventory: InitialInventory | None = None,
    well_roles: Mapping[str, str] | None = None,
    standard_pressure: float = 1.013235,
    standard_temperature: float = 288.15,
    well_regions: Mapping[str, int] | None = None,
    integration: str = "trapezoidal",
    inplace: bool = False,
    **inventory_options,
) -> pd.DataFrame:
    """
    Calculate cumulative injected pore volume.

    Field totals are calculated first when their required columns are absent.

    :return: Time data with ``injected_pv``.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(time_data, inplace)
    _, phases = self._fluid_definition()
    required = [f"cumulative_{phase}_injection_rc" for phase in phases]
    if not all(column in result for column in required):
        result = self.calculate_field_totals(
            result,
            well_roles=well_roles,
            standard_pressure=standard_pressure,
            standard_temperature=standard_temperature,
            well_regions=well_regions,
            integration=integration,
            inplace=True,
        )
    inventory = initial_inventory or self.calculate_initial_inventory(
        **inventory_options
    )
    return _calculate_injected_pore_volumes(
        result, inventory.pore_volume, phases, inplace=True
    )


def calculate_volume_recovery_factors(
    self,
    time_data,
    simple_stoiip: float | None = None,
    oil_formation_volume_factor: float | None = None,
    initial_inventory: InitialInventory | None = None,
    well_roles: Mapping[str, str] | None = None,
    standard_pressure: float = 1.013235,
    standard_temperature: float = 288.15,
    well_regions: Mapping[str, int] | None = None,
    integration: str = "trapezoidal",
    inplace: bool = False,
    **inventory_options,
) -> pd.DataFrame:
    """
    Calculate gross and net volume-based recovery factors.

    The required standard-condition field totals are calculated when absent.

    :return: Time data with gross/net simple and compositional RF columns.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(time_data, inplace)
    required = ["cumulative_oil_production_sc", "cumulative_oil_injection_sc"]
    if not all(column in result for column in required):
        result = self.calculate_field_totals(
            result,
            well_roles=well_roles,
            standard_pressure=standard_pressure,
            standard_temperature=standard_temperature,
            well_regions=well_regions,
            integration=integration,
            inplace=True,
        )
    inventory = initial_inventory or self.calculate_initial_inventory(
        **inventory_options
    )
    if simple_stoiip is None:
        if oil_formation_volume_factor is None:
            raise ValueError(
                "Provide simple_stoiip or oil_formation_volume_factor to calculate "
                "simple volume-based RF."
            )
        simple_stoiip = inventory.phase_volumes_reservoir["oil"] / float(
            oil_formation_volume_factor
        )
    return _calculate_volume_recovery_factors(
        result,
        float(simple_stoiip),
        inventory.oil_volume_standard,
        inplace=True,
    )


def _initial_states(
    self,
    mode: str,
    pore_volumes: np.ndarray,
    initial_pressure,
    initial_composition,
    initial_state_specification,
) -> np.ndarray:
    components, _ = self._fluid_definition()
    mode = {"actual": "cellwise"}.get(str(mode).lower(), str(mode).lower())
    if mode not in {"constant", "average", "cellwise"}:
        raise ValueError(
            "initial_state_mode must be 'constant', 'average', 'cellwise', or 'actual'"
        )
    n_cells = int(self.reservoir.n)
    first = next(iter(self._containers().values()))

    if mode == "constant":
        if initial_pressure is None or initial_composition is None:
            raise ValueError(
                "constant initial_state_mode requires initial_pressure and "
                "initial_composition"
            )
        pressure_values = np.asarray(initial_pressure, dtype=float).reshape(-1)
        if pressure_values.size != 1:
            raise ValueError("initial_pressure must be scalar in constant mode")
        composition = np.asarray(initial_composition, dtype=float).reshape(-1)
        if composition.size == len(components) - 1:
            composition = np.append(composition, 1.0 - np.sum(composition))
        if composition.size != len(components):
            raise ValueError("initial_composition must contain every fluid component")
        if np.any(composition < 0.0) or not np.isclose(np.sum(composition), 1.0):
            raise ValueError("initial_composition must be non-negative and sum to one")
        state = [float(pressure_values[0]), *composition[:-1]]
        if first.thermal:
            if initial_state_specification is None:
                raise ValueError(
                    "Thermal physics requires initial_state_specification in constant mode"
                )
            state.append(float(initial_state_specification))
        return np.tile(np.asarray(state, dtype=float), (n_cells, 1))

    variable_names = list(self.physics.vars)
    _, properties = self.output_properties(
        output_properties=variable_names, ts_idx=0, engine=False
    )
    states = np.column_stack(
        [
            np.asarray(properties[name], dtype=float).reshape(-1)[:n_cells]
            for name in variable_names
        ]
    )
    if states.shape != (n_cells, len(variable_names)):
        raise ValueError("Saved initial state does not contain every reservoir cell.")
    if not np.all(np.isfinite(states)):
        raise ValueError("Saved initial state contains non-finite values.")
    if mode == "average":
        weights = pore_volumes / np.sum(pore_volumes)
        averaged_state = np.sum(states * weights[:, None], axis=0)
        states = np.tile(averaged_state, (n_cells, 1))
    return states


def calculate_initial_inventory(
    self,
    standard_pressure: float = 1.013235,
    standard_temperature: float = 288.15,
    initial_state_mode: str = "cellwise",
    initial_pressure=None,
    initial_composition=None,
    initial_state_specification=None,
    oil_phase: str = "oil",
) -> InitialInventory:
    """
    Calculate regional initial phase volumes, component masses, and STOIIP.

    :param standard_pressure: Reference pressure [bar].
    :type standard_pressure: float
    :param standard_temperature: Reference temperature [K].
    :type standard_temperature: float
    :param initial_state_mode: ``constant``, ``average``, or ``cellwise``/``actual``.
    :type initial_state_mode: str
    :param initial_pressure: Constant-mode reservoir pressure [bar].
    :type initial_pressure: float, optional
    :param initial_composition: Constant-mode overall fluid composition.
    :type initial_composition: Sequence[float], optional
    :param initial_state_specification: Constant-mode thermal state variable [K or kJ/kmol].
    :type initial_state_specification: float, optional
    :param oil_phase: Phase treated as stock-tank oil.
    :type oil_phase: str
    :return: Initial reservoir inventory.
    :rtype: InitialInventory
    """
    components, phases = self._fluid_definition()
    if oil_phase not in phases:
        raise ValueError(f"Phase {oil_phase!r} is required to calculate STOIIP.")
    pore_volumes = self._pore_volumes()
    region_ids = self._region_ids()
    states = self._initial_states(
        initial_state_mode,
        pore_volumes,
        initial_pressure,
        initial_composition,
        initial_state_specification,
    )
    phase_volumes = dict.fromkeys(phases, 0.0)
    component_masses = dict.fromkeys(components, 0.0)
    oil_volume_standard = 0.0

    for region in np.unique(region_ids):
        container = self._containers()[int(region)]
        molecular_weights = np.asarray(container.Mw[: container.nc_fl], dtype=float)
        oil_index = phases.index(oil_phase)
        for cell in np.flatnonzero(region_ids == region):
            container.evaluate(states[cell].copy())
            saturation = np.asarray(
                container.sat[: container.np_fl], dtype=float
            ).copy()
            dens_m = np.asarray(container.dens_m[: container.np_fl], dtype=float).copy()
            phase_composition = np.asarray(container.x, dtype=float).copy()
            if not (
                np.all(np.isfinite(saturation))
                and np.all(np.isfinite(dens_m))
                and np.all(np.isfinite(phase_composition))
            ):
                raise ValueError(
                    f"Invalid initial properties in reservoir cell {cell}."
                )

            for phase_index, phase in enumerate(phases):
                phase_volume = pore_volumes[cell] * max(saturation[phase_index], 0.0)
                phase_volumes[phase] += phase_volume
                if phase_volume == 0.0:
                    continue
                if dens_m[phase_index] <= 0.0:
                    raise ValueError(
                        f"Non-positive molar density for phase {phase!r} in cell {cell}."
                    )
                phase_moles = phase_volume * dens_m[phase_index]
                phase_mass = (
                    phase_moles * phase_composition[phase_index] * molecular_weights
                )
                for component_index, component in enumerate(components):
                    component_masses[component] += phase_mass[component_index]

            reservoir_oil_moles = (
                pore_volumes[cell] * saturation[oil_index] * dens_m[oil_index]
            )
            if reservoir_oil_moles <= 0.0:
                continue
            oil_composition = phase_composition[oil_index].copy()
            active, nu, _, standard_dens_m = self._standard_phase_state(
                container,
                float(standard_pressure),
                float(standard_temperature),
                oil_composition,
            )
            if oil_index in active:
                oil_volume_standard += (
                    reservoir_oil_moles * nu[oil_index] / standard_dens_m[oil_index]
                )

    return InitialInventory(
        pore_volume=float(np.sum(pore_volumes)),
        phase_volumes_reservoir={
            key: float(value) for key, value in phase_volumes.items()
        },
        component_masses={key: float(value) for key, value in component_masses.items()},
        oil_volume_standard=float(oil_volume_standard),
    )


def calculate_compositional_stoiip(self, **kwargs) -> float:
    """
    Return initial oil volume at the requested standard conditions.

    :return: Compositional STOIIP [m3].
    :rtype: float
    """
    return self.calculate_initial_inventory(**kwargs).oil_volume_standard


def calculate_compositional_recovery_factor(
    self, cumulative_oil_production, **inventory_options
):
    """
    Calculate gross compositional volume-based oil recovery factor.

    :param cumulative_oil_production: Cumulative standard-condition oil production [m3].
    :type cumulative_oil_production: array-like
    :return: Recovery factor and compositional STOIIP.
    :rtype: tuple[numpy.ndarray or float, float]
    """
    stoiip = self.calculate_compositional_stoiip(**inventory_options)
    production = np.asarray(cumulative_oil_production, dtype=float)
    if not np.isfinite(stoiip) or stoiip <= 0.0:
        result = np.full_like(production, np.nan, dtype=float)
    else:
        result = production / stoiip
    if result.ndim == 0:
        return float(result), stoiip
    return result, stoiip


def calculate_gor(
    self,
    time_data,
    well_roles: Mapping[str, str] | None = None,
    producer_wells: Sequence[str] | None = None,
    standard_pressure: float = 1.013235,
    standard_temperature: float = 288.15,
    well_regions: Mapping[str, int] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Calculate standard-condition GOR for production wells and the field.

    :return: Time data with GOR columns.
    :rtype: pandas.DataFrame
    """
    result = _working_frame(time_data, inplace)
    required = [
        f"well_{well}_volumetric_rate_{phase}_at_wh_sc"
        for well in self._well_names()
        for phase in ("gas", "oil")
    ]
    if not all(column in result for column in required):
        result = self.convert_rates_to_standard_conditions(
            result, standard_pressure, standard_temperature, well_regions, True
        )
    roles = self.resolve_well_roles(result, well_roles)
    if producer_wells is None:
        producer_wells = [well for well, role in roles.items() if role == "producer"]
    invalid = [well for well in producer_wells if roles.get(well) != "producer"]
    if invalid:
        raise ValueError(f"GOR requested for non-production wells: {invalid}")
    if not producer_wells:
        raise ValueError("At least one production well is required for GOR.")
    _, phases = self._fluid_definition()
    if "gas" not in phases or "oil" not in phases:
        raise ValueError("GOR requires phases named 'gas' and 'oil'.")
    return calculate_production_gor(result, producer_wells, "gas", "oil", True)


def compute_component_rf(
    self,
    time_data,
    components: Sequence[str] | None = None,
    well_roles: Mapping[str, str] | None = None,
    initial_inventory: InitialInventory | None = None,
    integration: str = "trapezoidal",
    inplace: bool = False,
    **inventory_options,
) -> pd.DataFrame:
    """
    Calculate gross and net mass-based component recovery factors.

    :return: Component mass-accounting table.
    :rtype: pandas.DataFrame
    """
    fluid_components, _ = self._fluid_definition()
    selected = fluid_components if components is None else list(components)
    unknown = sorted(set(selected) - set(fluid_components))
    if unknown:
        raise ValueError(f"Unknown fluid components: {unknown}")
    roles = self.resolve_well_roles(time_data, well_roles)
    inventory = initial_inventory or self.calculate_initial_inventory(
        **inventory_options
    )
    result = _working_frame(time_data, inplace)
    missing_mass = [
        f"well_{well}_mass_rate_{component}_by_sum_perfs"
        for well in self._well_names()
        for component in selected
        if f"well_{well}_mass_rate_{component}_by_sum_perfs" not in result
    ]
    if missing_mass:
        for well in self._well_names():
            for index, component in enumerate(fluid_components):
                mass_column = f"well_{well}_mass_rate_{component}_by_sum_perfs"
                molar_column = f"well_{well}_molar_rate_{component}_by_sum_perfs"
                if mass_column not in result and molar_column in result:
                    result[mass_column] = result[molar_column].to_numpy(
                        dtype=float
                    ) * float(next(iter(self._containers().values())).Mw[index])
        missing_mass = [column for column in missing_mass if column not in result]
        if missing_mass:
            raise KeyError(
                "Missing required mass-rate columns: " + ", ".join(missing_mass)
            )
    mass = _calculate_mass_recovery_factors(
        result,
        inventory.component_masses,
        roles,
        selected,
        integration,
    )
    for column in mass.columns:
        if column != "time":
            result[column] = mass[column].to_numpy()
    return result


def calculate(
    self,
    time_data=None,
    well_roles: Mapping[str, str] | None = None,
    producer_wells: Sequence[str] | None = None,
    well_regions: Mapping[str, int] | None = None,
    standard_pressure: float = 1.013235,
    standard_temperature: float = 288.15,
    initial_state_mode: str = "cellwise",
    initial_pressure=None,
    initial_composition=None,
    initial_state_specification=None,
    oil_formation_volume_factor: float | None = None,
    simple_stoiip: float | None = None,
    integration: str = "trapezoidal",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Build a complete production-accounting table without plotting or file I/O.

    ``simple_stoiip`` can be supplied directly. Otherwise a positive oil
    formation-volume factor is required, and simple STOIIP is calculated as
    initial reservoir-condition oil volume divided by that factor.


    :param time_data: Existing time data or None to calculate it from well output.
    :type time_data: pandas.DataFrame or Mapping, optional

    :param well_roles: Optional producer/injector mapping.
    :type well_roles: Mapping[str, str], optional

    :param producer_wells: Optional subset of production wells for GOR.
    :type producer_wells: Sequence[str], optional

    :param well_regions: Optional property-container region for each well flash.
    :type well_regions: Mapping[str, int], optional

    :param standard_pressure: Standard pressure [bar].
    :type standard_pressure: float

    :param standard_temperature: Standard temperature [K].
    :type standard_temperature: float

    :param initial_state_mode: ``constant``, ``average``, or ``cellwise``/``actual``.
    :type initial_state_mode: str

    :param initial_pressure: Constant-mode reservoir pressure [bar].
    :type initial_pressure: float, optional

    :param initial_composition: Constant-mode overall fluid composition.
    :type initial_composition: Sequence[float], optional

    :param initial_state_specification: Constant-mode thermal state variable.
    :type initial_state_specification: float, optional

    :param oil_formation_volume_factor: Oil formation-volume factor for simple STOIIP.
    :type oil_formation_volume_factor: float, optional

    :param simple_stoiip: Explicit simple STOIIP denominator [m3].
    :type simple_stoiip: float, optional

    :param integration: Cumulative integration convention.
    :type integration: str

    :param inplace: Modify an input DataFrame when True.
    :type inplace: bool

    :return: Enriched production-accounting table.
    :rtype: pandas.DataFrame
    """
    if time_data is None:  # if time data is not passed compute it
        time_data = self.store_well_time_data(
            phase_molar_rates=True,
            phase_mass_rates=True,
            phase_volumetric_rates=True,
            component_molar_rates=True,
            component_mass_rates=True,
            advective_heat_rates=False,
            save_output_files=False,
        )
    result = _working_frame(time_data, inplace)
    _, phases = self._fluid_definition()
    if "oil" not in phases:
        raise ValueError("Production recovery accounting requires an 'oil' phase.")

    # ---- convert well output to standard a.k.a. surface conditions
    result = self.convert_rates_to_standard_conditions(
        result,
        standard_pressure,
        standard_temperature,
        well_regions,
        inplace=True,
    )

    # ---- calculate field totals
    result = self.calculate_field_totals(
        result,
        well_roles=well_roles,
        standard_pressure=standard_pressure,
        standard_temperature=standard_temperature,
        well_regions=well_regions,
        integration=integration,
        inplace=True,
    )

    # ---- calculate STOIIP, and mass per component
    inventory_options = {
        "standard_pressure": standard_pressure,
        "standard_temperature": standard_temperature,
        "initial_state_mode": initial_state_mode,  # cst, avg, true values
        "initial_pressure": initial_pressure,
        "initial_composition": initial_composition,
        "initial_state_specification": initial_state_specification,
    }
    inventory = self.calculate_initial_inventory(**inventory_options)

    # ---- pore volumes injected
    result = self.calculate_injected_pore_volumes(
        result,
        initial_inventory=inventory,
        well_roles=well_roles,
        standard_pressure=standard_pressure,
        standard_temperature=standard_temperature,
        well_regions=well_regions,
        integration=integration,
        inplace=True,
    )
    # ---- compute volume-based recovery factors
    result = self.calculate_volume_recovery_factors(
        result,
        simple_stoiip=simple_stoiip,
        oil_formation_volume_factor=oil_formation_volume_factor,
        initial_inventory=inventory,
        well_roles=well_roles,
        standard_pressure=standard_pressure,
        standard_temperature=standard_temperature,
        well_regions=well_regions,
        integration=integration,
        inplace=True,
    )

    # ---- compute mass-based recovery factors
    result = self.compute_component_rf(
        result,
        well_roles=well_roles,
        initial_inventory=inventory,
        integration=integration,
        inplace=True,
    )

    # ---- compute gas oil ratio at standard conditions
    if "gas" in phases:
        result = self.calculate_gor(
            result,
            well_roles=well_roles,
            producer_wells=producer_wells,
            standard_pressure=standard_pressure,
            standard_temperature=standard_temperature,
            well_regions=well_regions,
            inplace=True,
        )

    return result
