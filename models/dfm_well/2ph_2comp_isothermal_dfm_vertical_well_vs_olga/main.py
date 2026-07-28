"""
Injection of pure gaseous CO₂ at a constant mass rate into a vertical well containing water using a standalone
well model to compare its results with those in OLGA for an isothermal two-phase scenario.
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


redirect_darts_output("run.log")
coupled_model = Model()
coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
coupled_model.init()
coupled_model.set_output()

output_props = coupled_model.physics.vars + coupled_model.output.properties
coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)  # saves initial well conditions

time_steps = [
    # 5 min for well profiles benchmark
    5 / 60 / 24,

    # 1 hour for time series benchmark
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

coupled_model.print_timers()

save_dfm_well_props(
    'I1',
    coupled_model,
    include_overall_composition=True,
    include_phase_velocities=True,
    include_phase_rates=True,
)

plot_heat_map_pcolormesh('I1', coupled_model, show_plot=False)
# plot_heat_map_contourf('I1', coupled_model, show_plot=False)
