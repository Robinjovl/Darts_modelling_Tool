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


redirect_darts_output('run.log')
coupled_model = Model()
coupled_model.reservoir.grav_acceleration_for_spe = 9.80665
coupled_model.init()
coupled_model.set_output()

if 1:
    output_props = coupled_model.physics.vars + coupled_model.output.properties
    coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)  # saves initial well conditions

    time_steps = [
        10 / 24 / 60,   # 10 minutes
                 ]

    for i, dt in enumerate(time_steps):
        coupled_model.run(dt)
        coupled_model.output.well_output_to_vtp(ith_step=i + 1, output_properties=output_props)

    coupled_model.print_timers()
else:
    save_dfm_well_props('I1', coupled_model)

    plot_heat_map_pcolormesh('I1', coupled_model, show_plot=False)
    plot_heat_map_contourf('I1', coupled_model, show_plot=False)
