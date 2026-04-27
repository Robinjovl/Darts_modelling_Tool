"""
Injection of pure gaseous CO₂ at a constant mass injection rate into an inclined well containing water using a standalone
well model to compare its results with those in OLGA for an isothermal two-phase scenario.

Lessons learned:
    1- TODO

    2- TODO

OLGA example with which this DARTS-well example is compared is available here:
    https://gitlab.com/open-darts/TODO
Comparison of the results are available in the following Excel file:
    https://gitlab.com/open-darts/TODO
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


REFERENCE_WELL_PRIMARY_VARIABLES = np.array(
    [
        [5.015943251532377e01, 9.999997370789937e-01],
        [5.054523668238304e01, 9.999954275818572e-01],
        [5.093702159997892e01, 9.999622677856230e-01],
        [5.133480370512198e01, 9.998046407806931e-01],
        [5.173863982032162e01, 9.992923212408485e-01],
        [5.214857906001833e01, 9.981053659281838e-01],
        [5.256482348884938e01, 9.957010200337159e-01],
        [5.298805482711848e01, 9.862112490653944e-01],
        [5.342420571740497e01, 8.885159434484585e-01],
        [5.395562416525289e01, 4.259069818066613e-01],
        [5.496892575641048e01, 1.103615786658956e-01],
        [5.697059120110149e01, 3.975977317671529e-02],
        [5.992336052145273e01, 1.908170246949150e-02],
        [6.331008174583686e01, 1.372587154094766e-02],
        [6.671597534137200e01, 9.565264021569460e-03],
        [7.011615453515066e01, 6.359344949973211e-03],
        [7.351225012022188e01, 4.041396435393687e-03],
        [7.690559988538514e01, 2.458799213059893e-03],
        [8.029727895317311e01, 1.434665868647259e-03],
        [8.369550079908929e01, 8.041929026612709e-04],
    ],
    dtype=float,
)
PRIMARY_VARIABLE_NAMES = ("pressure", "z_CO2")
PRIMARY_VARIABLE_ATOL = np.array([1e-3, 1e-5], dtype=float)


def extract_well_primary_variables(coupled_model: Model) -> np.ndarray:
    n_vars = coupled_model.physics.n_vars
    n_blocks = coupled_model.reservoir.mesh.n_blocks
    n_res_blocks = coupled_model.reservoir.mesh.n_res_blocks
    state = np.asarray(coupled_model.physics.engine.X).reshape(n_blocks, n_vars)
    return state[n_res_blocks:, :].copy()


def assert_reference_well_primary_variables(coupled_model: Model) -> None:
    actual = extract_well_primary_variables(coupled_model)
    expected = REFERENCE_WELL_PRIMARY_VARIABLES

    if actual.shape != expected.shape:
        raise AssertionError(
            f"Unexpected well primary-variable shape {actual.shape}; expected {expected.shape}."
        )

    for idx, (name, atol) in enumerate(
        zip(PRIMARY_VARIABLE_NAMES, PRIMARY_VARIABLE_ATOL)
    ):
        abs_diff = np.abs(actual[:, idx] - expected[:, idx])
        max_abs_diff = float(np.max(abs_diff))
        if max_abs_diff > float(atol):
            raise AssertionError(
                f"Primary-variable regression failed for '{name}': "
                f"max abs diff {max_abs_diff:.6e} exceeds tolerance {float(atol):.6e}."
            )


redirect_darts_output("run.log")
coupled_model = Model()
coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
coupled_model.init()
coupled_model.set_output()

if 1:
    output_props = coupled_model.physics.vars + coupled_model.output.properties
    coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)  # saves initial well conditions

    time_steps = [
        # 5 min for well profiles validation
        5 / 60 / 24,

        # 1 hour for time series validation
        # 20 / 60 / 24,
        # 20 / 60 / 24,
        # 20 / 60 / 24,
    ]

    for i, dt in enumerate(time_steps):
        if i == 1:
            coupled_model.data_ts.dt_max = 5 / (24 * 60 * 60)
        elif i == 2:
            coupled_model.data_ts.dt_max = 10 / (24 * 60 * 60)
        elif i == 3:
            coupled_model.data_ts.dt_max = 15 / (24 * 60 * 60)

        coupled_model.run(dt)
        coupled_model.output.well_output_to_vtp(ith_step=i + 1, output_properties=output_props)

    assert_reference_well_primary_variables(coupled_model)
    coupled_model.print_timers()
else:
    save_dfm_well_props('I1', coupled_model)

    plot_heat_map_pcolormesh('I1', coupled_model)
    plot_heat_map_contourf('I1', coupled_model)
