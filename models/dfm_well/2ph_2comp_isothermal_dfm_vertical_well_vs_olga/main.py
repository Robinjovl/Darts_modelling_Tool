"""
Injection of pure gaseous CO₂ at a constant mass injection rate into a vertical well containing water using a standalone
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
        [5.391740349099573e01, 9.999986647105943e-01],
        [5.453992405160889e01, 9.999756880302456e-01],
        [5.517539542611163e01, 9.997961798066283e-01],
        [5.582414271546871e01, 9.989631757040373e-01],
        [5.648674876236742e01, 9.963377723087214e-01],
        [5.716495304293696e01, 9.860630113159974e-01],
        [5.786856171856417e01, 9.144304484236350e-01],
        [5.868270030900496e01, 5.573894843865207e-01],
        [6.001220142769438e01, 1.776395682891973e-01],
        [6.228565463558456e01, 8.378507971567931e-02],
        [6.574919481562932e01, 3.624891116884556e-02],
        [7.028087053893761e01, 1.970380718659471e-02],
        [7.515427755975841e01, 1.321031062004633e-02],
        [8.001723333154247e01, 8.504564977136090e-03],
        [8.487181866719531e01, 5.215255043614462e-03],
        [8.972101890766935e01, 3.047729880069187e-03],
        [9.456713928886896e01, 1.700902041932439e-03],
        [9.941182034242964e01, 9.084510059438370e-04],
        [1.042561410916646e02, 4.652444280954061e-04],
        [1.091052456532159e02, 2.288737524780077e-04],
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
