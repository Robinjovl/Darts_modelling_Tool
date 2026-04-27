"""
Injection of pure gaseous CO₂ at a constant mass injection rate with a constant temperature into a well containing
gaseous CO₂ using a standalone well model to compare its results with those in OLGA for a thermal single-phase scenario.

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


REFERENCE_WELL_PRIMARY_VARIABLES = np.array(
    [
        [9.059293292292640, -585.690271389921008],
        [8.985941472634265, -564.110585459582126],
        [8.928697303806334, -542.530641084318290],
        [8.869550299240844, -520.950892195863617],
        [8.808383264196106, -499.371321108596021],
        [8.745130475364874, -477.791877121725975],
        [8.679722825668993, -456.212113168560961],
        [8.612087389381667, -434.699140732011756],
        [8.542147082580863, -413.096096238414248],
        [8.469820241293679, -391.499616691897472],
        [8.395020344839031, -369.908164170666623],
        [8.317655442645545, -348.321097229673569],
        [8.237627742909698, -326.737394703395353],
        [8.154832957116927, -305.156107846131874],
        [8.069159842971247, -283.556356655662682],
        [7.980489312168285, -262.015762485290054],
        [7.888693351367723, -240.419726953966830],
        [7.793634565622257, -218.852456034543280],
        [7.695130360127781, -197.300202034441440],
        [7.616119998784487, -175.678313611993843],
    ],
    dtype=float,
)
PRIMARY_VARIABLE_NAMES = ("pressure", "molar_enthalpy")
PRIMARY_VARIABLE_ATOL = np.array([1e-3, 1e-2], dtype=float)


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
        10 / 24 / 60,  # 10 minutes
    ]

    for i, dt in enumerate(time_steps):
        coupled_model.run(dt)
        coupled_model.output.well_output_to_vtp(ith_step=i + 1, output_properties=output_props)

    assert_reference_well_primary_variables(coupled_model)
    coupled_model.print_timers()
else:
    save_dfm_well_props('I1', coupled_model)

    plot_heat_map_pcolormesh('I1', coupled_model)
    plot_heat_map_contourf('I1', coupled_model)
