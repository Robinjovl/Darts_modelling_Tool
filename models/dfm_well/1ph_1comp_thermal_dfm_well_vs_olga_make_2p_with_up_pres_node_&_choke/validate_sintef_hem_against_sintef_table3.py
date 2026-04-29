"""
Diagnostic comparison of open-DARTS choke implementations against SINTEF data.

The data below are taken from Table 3 of Hammer et al., "Experiments and
modelling of choked flow of CO2 in orifices and nozzles". The script compares
two choices:

* PERKINS with frozen single-liquid CO2, which is useful as a baseline and is
  expected to be biased for flashing releases.
* SINTEF_HEM with equilibrium flashing and contraction coefficients from the
  SINTEF HEM comparison.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXAMPLE_DIR.parents[2]

for path in (REPO_ROOT, EXAMPLE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from darts.pipes.upstream_pressure_node_with_choke import (  # noqa: E402
    ChokeBoundaryState,
    ChokePhysicsHelper,
    build_equilibrium_model,
    build_hydraulic_choke_model,
    build_recovery_model,
    build_slip_model,
    build_valve_geometry_model,
)
from model import Model  # noqa: E402


@dataclass(frozen=True)
class SintefCase:
    test_no: int
    geometry: str
    diameter_mm: float
    initial_temperature_c: float
    initial_pressure_mpa: float
    plateau_pressure_mpa: float
    measured_mass_rate_kg_s: float
    measured_mass_flux_t_m2_s: float
    sintef_hem_mass_flux_t_m2_s: float
    contraction_coefficient: float


@dataclass(frozen=True)
class ValidationResult:
    case: SintefCase
    upstream_temperature_k: float
    liquid_density_kg_m3: float
    predicted_mass_rate_kg_s: float
    predicted_mass_flux_t_m2_s: float
    measured_error_pct: float
    sintef_hem_error_pct: float | None
    throat_pressure_bar: float
    flow_regime: str


SINTEF_CASES = (
    SintefCase(13, "orifice", 12.7, 24.6, 12.77, 9.61, 8.592, 67.8, 63.9, 0.75),
    SintefCase(16, "orifice", 4.5, 24.4, 12.17, 11.58, 1.600, 100.6, 74.8, 0.74),
    SintefCase(17, "nozzle", 4.5, 25.2, 12.40, 11.74, 1.807, 113.6, 101.6, 1.0),
    SintefCase(18, "nozzle", 12.7, 25.1, 12.41, 8.81, 10.072, 79.5, 76.1, 1.0),
    SintefCase(20, "nozzle", 9.0, 22.7, 11.40, 9.40, 5.515, 86.7, 83.7, 1.0),
    SintefCase(21, "orifice", 9.0, 22.0, 11.50, 9.94, 4.208, 66.2, 66.4, 0.74),
)

PIPE_DIAMETER_M = 0.0408
ATMOSPHERIC_PRESSURE_BAR = 1.01325
PERKINS_AVERAGE_DISCHARGE_COEFFICIENT = 0.826
CO2_COMPOSITION = np.array([1.0])


def _mpa_to_bar(pressure_mpa: float) -> float:
    return 10.0 * pressure_mpa


def _diameter_to_area(diameter_m: float) -> float:
    return math.pi * diameter_m**2 / 4.0


def _liquid_state_at_plateau(
    helper: ChokePhysicsHelper,
    case: SintefCase,
) -> ChokeBoundaryState:
    """
    Build the restriction-upstream state used by the Perkins comparison.

    Table 3 reports initial pipe pressure/temperature and plateau pressure.
    Because it does not list plateau temperature at the restriction, this script
    follows an isentropic liquid path from the initial pipe state to the plateau
    pressure before evaluating the choke.
    """
    initial_pressure_bar = _mpa_to_bar(case.initial_pressure_mpa)
    initial_temperature_k = case.initial_temperature_c + 273.15
    plateau_pressure_bar = _mpa_to_bar(case.plateau_pressure_mpa)

    initial_enthalpy = helper.evaluate_phase_enthalpy(
        "L",
        initial_pressure_bar,
        initial_temperature_k,
        CO2_COMPOSITION,
    )
    initial_entropy = helper.evaluate_phase_entropy(
        initial_pressure_bar,
        initial_temperature_k,
        CO2_COMPOSITION,
        helper.phase_root_flag("L"),
    )
    initial_state = ChokeBoundaryState(
        pressure=initial_pressure_bar,
        temperature=initial_temperature_k,
        composition=CO2_COMPOSITION,
        phase_name="L",
        molar_enthalpy=initial_enthalpy,
        molar_entropy=initial_entropy,
    )
    plateau_temperature_k, plateau_enthalpy = helper.solve_single_phase_isentropic_state(
        initial_state,
        plateau_pressure_bar,
    )
    plateau_entropy = helper.evaluate_phase_entropy(
        plateau_pressure_bar,
        plateau_temperature_k,
        CO2_COMPOSITION,
        helper.phase_root_flag("L"),
    )
    return ChokeBoundaryState(
        pressure=plateau_pressure_bar,
        temperature=plateau_temperature_k,
        composition=CO2_COMPOSITION,
        phase_name="L",
        molar_enthalpy=plateau_enthalpy,
        molar_entropy=plateau_entropy,
    )


def evaluate_case(
    helper: ChokePhysicsHelper,
    case: SintefCase,
    *,
    hydraulic_model: str,
    equilibrium_model: str,
    discharge_coefficient: float,
    recovery: str,
) -> ValidationResult:
    boundary_state = _liquid_state_at_plateau(helper, case)
    choke_diameter_m = case.diameter_mm * 1e-3
    choke_model = build_hydraulic_choke_model(
        hydraulic_model=hydraulic_model,
        helper=helper,
        boundary_state=boundary_state,
        valve_geometry_model=build_valve_geometry_model(
            valve_geometry="ORIFICE",
            diameter=choke_diameter_m,
            discharge_coefficient=discharge_coefficient,
        ),
        equilibrium_model=build_equilibrium_model(
            equilibrium_model=equilibrium_model,
            thermal_phase_equilibrium=equilibrium_model == "EQUILIBRIUM",
        ),
        recovery_model=build_recovery_model(recovery),
        slip_model=build_slip_model("NOSLIP"),
        upstream_area=_diameter_to_area(PIPE_DIAMETER_M),
        downstream_area=_diameter_to_area(PIPE_DIAMETER_M),
    )
    result = choke_model.evaluate(ATMOSPHERIC_PRESSURE_BAR)
    area = _diameter_to_area(choke_diameter_m)
    predicted_mass_flux = result.mass_rate_kg_s / area / 1000.0
    measured_error = (
        100.0
        * (predicted_mass_flux - case.measured_mass_flux_t_m2_s)
        / case.measured_mass_flux_t_m2_s
    )
    sintef_hem_error = None
    if hydraulic_model == "SINTEF_HEM":
        sintef_hem_error = (
            100.0
            * (predicted_mass_flux - case.sintef_hem_mass_flux_t_m2_s)
            / case.sintef_hem_mass_flux_t_m2_s
        )
    density = helper.evaluate_phase_density(
        "L",
        boundary_state.pressure,
        boundary_state.temperature,
        boundary_state.composition,
    )
    return ValidationResult(
        case=case,
        upstream_temperature_k=boundary_state.temperature,
        liquid_density_kg_m3=density,
        predicted_mass_rate_kg_s=result.mass_rate_kg_s,
        predicted_mass_flux_t_m2_s=predicted_mass_flux,
        measured_error_pct=measured_error,
        sintef_hem_error_pct=sintef_hem_error,
        throat_pressure_bar=result.throat_pressure,
        flow_regime=result.flow_regime,
    )


def _print_results(
    title: str,
    results: list[ValidationResult],
    *,
    compare_to_sintef_hem: bool = False,
) -> None:
    print(title)
    print(
        "Assumption: plateau pressure is the restriction-upstream pressure; "
        "plateau temperature is estimated by an isentropic liquid path."
    )
    print()
    header = (
        "test geom      d_mm  T_up_K  rho_kg_m3  "
        "pred_j  meas_j  meas_err_%"
    )
    if compare_to_sintef_hem:
        header += "  sintef_hem_j  hem_err_%"
    header += "  p_throat_bar  regime"
    print(header)
    for item in results:
        row = (
            f"{item.case.test_no:>4} "
            f"{item.case.geometry:<7} "
            f"{item.case.diameter_mm:>5.1f} "
            f"{item.upstream_temperature_k:>7.3f} "
            f"{item.liquid_density_kg_m3:>10.3f} "
            f"{item.predicted_mass_flux_t_m2_s:>7.3f} "
            f"{item.case.measured_mass_flux_t_m2_s:>7.3f} "
            f"{item.measured_error_pct:>10.2f} "
        )
        if compare_to_sintef_hem:
            row += (
                f"{item.case.sintef_hem_mass_flux_t_m2_s:>13.3f} "
                f"{item.sintef_hem_error_pct:>9.2f} "
            )
        row += (
            f"{item.throat_pressure_bar:>13.3f} "
            f"{item.flow_regime}"
        )
        print(row)

    abs_errors = [abs(item.measured_error_pct) for item in results]
    signed_errors = [item.measured_error_pct for item in results]
    print()
    print(
        f"measured mean signed error: {sum(signed_errors) / len(signed_errors):.2f}%"
    )
    print(f"measured mean absolute error: {sum(abs_errors) / len(abs_errors):.2f}%")
    print(f"measured max absolute error: {max(abs_errors):.2f}%")
    if compare_to_sintef_hem:
        hem_errors = [
            item.sintef_hem_error_pct
            for item in results
            if item.sintef_hem_error_pct is not None
        ]
        abs_hem_errors = [abs(error) for error in hem_errors]
        print(
            f"SINTEF HEM mean signed error: {sum(hem_errors) / len(hem_errors):.2f}%"
        )
        print(
            f"SINTEF HEM mean absolute error: "
            f"{sum(abs_hem_errors) / len(abs_hem_errors):.2f}%"
        )
        print(f"SINTEF HEM max absolute error: {max(abs_hem_errors):.2f}%")


def main() -> None:
    model = Model()
    helper = ChokePhysicsHelper(model.physics)
    perkins_results = [
        evaluate_case(
            helper,
            case,
            hydraulic_model="PERKINS",
            equilibrium_model="FROZEN",
            discharge_coefficient=PERKINS_AVERAGE_DISCHARGE_COEFFICIENT,
            recovery="ON",
        )
        for case in SINTEF_CASES
    ]
    hem_results = [
        evaluate_case(
            helper,
            case,
            hydraulic_model="SINTEF_HEM",
            equilibrium_model="EQUILIBRIUM",
            discharge_coefficient=case.contraction_coefficient,
            recovery="OFF",
        )
        for case in SINTEF_CASES
    ]
    _print_results(
        "Perkins FROZEN-liquid comparison against SINTEF Table 3 "
        f"(Cd={PERKINS_AVERAGE_DISCHARGE_COEFFICIENT}, recovery=ON). "
        "Caveat: flashing is disabled.",
        perkins_results,
    )
    print()
    _print_results(
        "SINTEF_HEM equilibrium-flashing comparison "
        "(Cd=case contraction coefficient, recovery=OFF).",
        hem_results,
        compare_to_sintef_hem=True,
    )


if __name__ == "__main__":
    main()
