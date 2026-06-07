"""
Injection of pure gaseous CO₂ at a constant mass injection rate with a constant enthalpy into a well containing
gaseous CO₂ using a standalone well model to compare its results with those in OLGA for a thermal two-phase scenario.

Lessons learned:
    1- TODO

    2- TODO

OLGA example with which this DARTS-well example is compared is available here:
    https://gitlab.com/open-darts/TODO
Comparison of the results are available in the following Excel file:
    https://gitlab.com/open-darts/TODO

Corresponding OLGA file is in my_old_laptop/Desktop/march/non-isothermal single-phase model validation with CO2

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh
from darts.pipes.viz.plot_heat_map_contourf import plot_heat_map_contourf

from model import Model


def build_model(choke_opening=1.0) -> Model:
    coupled_model = Model(choke_opening=choke_opening)
    coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
    return coupled_model


def run_model(
    choke_opening=1.0,
    save_output: bool = False,
) -> Model:
    coupled_model = build_model(choke_opening=choke_opening)
    coupled_model.init()
    coupled_model.set_output()

    output_props = coupled_model.physics.vars + coupled_model.output.properties
    if save_output:
        coupled_model.output.well_output_to_vtp(
            ith_step=0,
            output_properties=output_props,
        )

    coupled_model.run(10 / 24 / 60)

    if save_output:
        coupled_model.output.well_output_to_vtp(
            ith_step=1,
            output_properties=output_props,
        )

    return coupled_model


def size_choke_for_target_rate(
    max_outer_iters: int = 3,
    rel_tol: float = 1e-4,
):
    choke_opening = 1.0
    sizing_history = []

    for outer_iter in range(1, max_outer_iters + 1):
        coupled_model = run_model(
            choke_opening=choke_opening,
            save_output=False,
        )
        choke = coupled_model.wells["I1"].source_sinks["UpstreamPressureNodeWithChoke1"]
        actual_mass_rate_kg_s = choke.last_mass_rate_kg_s
        target_mass_rate_kg_s = choke.target_mass_rate_kg_s

        if actual_mass_rate_kg_s is None or actual_mass_rate_kg_s <= 0.0:
            raise RuntimeError(
                "Choke sizing failed because the evaluated mass rate is not positive."
            )

        rel_error = abs(actual_mass_rate_kg_s - target_mass_rate_kg_s) / max(
            target_mass_rate_kg_s,
            1e-30,
        )
        suggested_opening = choke.suggest_opening_for_target_rate()
        sizing_history.append(
            {
                "outer_iter": outer_iter,
                "opening": float(choke.opening),
                "diameter_m": float(choke.diameter),
                "actual_mass_rate_kg_s": float(actual_mass_rate_kg_s),
                "target_mass_rate_kg_s": float(target_mass_rate_kg_s),
                "rel_error": float(rel_error),
                "suggested_opening": float(suggested_opening),
            }
        )

        if rel_error <= rel_tol:
            return float(choke.opening), sizing_history

        choke_opening = suggested_opening

    return float(choke_opening), sizing_history


redirect_darts_output('run.log')

final_opening, sizing_history = size_choke_for_target_rate()
coupled_model = run_model(
    choke_opening=final_opening,
    save_output=True,
)

coupled_model.print_timers()
save_dfm_well_props('I1', coupled_model)
plot_heat_map_pcolormesh('I1', coupled_model, show_plot=False)
plot_heat_map_contourf('I1', coupled_model, show_plot=False)

choke = coupled_model.wells["I1"].source_sinks["UpstreamPressureNodeWithChoke1"]
for sizing_step in sizing_history:
    print(
        "Choke sizing iteration "
        f"{sizing_step['outer_iter']}: "
        f"opening={sizing_step['opening']:.9f}, "
        f"diameter={sizing_step['diameter_m']:.9f} m, "
        f"actual_rate={sizing_step['actual_mass_rate_kg_s']:.6f} kg/s, "
        f"target_rate={sizing_step['target_mass_rate_kg_s']:.6f} kg/s, "
        f"rel_error={sizing_step['rel_error']:.6e}, "
        f"suggested_opening={sizing_step['suggested_opening']:.9f}"
    )

print(
    "Final choke run: "
    f"opening={choke.opening:.9f}, "
    f"diameter={choke.diameter:.9f} m, "
    f"actual_rate={choke.last_mass_rate_kg_s:.6f} kg/s, "
    f"target_rate={choke.target_mass_rate_kg_s:.6f} kg/s, "
    f"flow_regime={choke.last_flow_regime}, "
    f"throat_pressure={choke.last_throat_pressure:.6f} bar"
)
