"""
Injection of pure gaseous CO₂ at a constant mass rate with a constant temperature into a well containing
gaseous CO₂ using a standalone well model to compare its results with those in OLGA for a thermal single-phase scenario.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh
from darts.pipes.viz.plot_heat_map_contourf import plot_heat_map_contourf
from darts.pipes.viz.plot_well_segment_property_vs_time import plot_well_segment_property_vs_time

from model import Model


redirect_darts_output("run.log")
coupled_model = Model()
coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
coupled_model.init()
coupled_model.set_output()

output_props = coupled_model.physics.vars + coupled_model.output.properties
coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)  # saves initial well conditions

time_steps = [
    10 / 24 / 60,  # 10 minutes
]

for i, dt in enumerate(time_steps):
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

# plot_heat_map_pcolormesh('I1', coupled_model, show_plot=False)
plot_heat_map_contourf('I1', coupled_model, show_plot=False)

plot_well_segment_property_vs_time(
    [(coupled_model.output_folder, 'DARTS-well')],
    'I1', 'pressure', segment_index=-1,
    output_path=os.path.join(coupled_model.output_folder, 'bottom_segment_pressure_vs_time.png'),
    show_plot=False,
)
