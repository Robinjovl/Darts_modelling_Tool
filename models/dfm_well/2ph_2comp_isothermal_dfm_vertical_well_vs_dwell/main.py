"""
Injection of gaseous CO₂ at a constant mass rate into a vertical well containing water using a standalone
well model to compare its results with those in DWell for an isothermal two-phase scenario.

Lessons learned:
    1- The results of this scenario are very close to the results of the example
    main_full_column_of_water_variable_K_2-comp_validation_scenario2.py in DWell even though the averaging methods are
    a bit changed in DARTS-well for better numerical stability (Changes can be found here:
    https://gitlab.com/open-darts/open-darts/-/commit/b0aa26cb9beb90a10bb1b4e3db5291399607f46d). I tried the new
    averaging method, which is used in DARTS-well, in DWell too, and I could say the results are almost identical to
    the results of the old method used in DWell.
    I reverted the change in the averaging method of density here:
    https://gitlab.com/open-darts/open-darts/-/commit/0c584d57e8cd20c270057763f3b6bc916cbc695e

DWell example with which this DARTS-well example is compared is available here:
    https://gitlab.com/open-darts/dwell/-/blob/compare_with_darts_well/examples/two_phase/main_full_column_of_water_variable_K_2-comp_validation_scenario2.py?ref_type=heads
Comparison of the results are available in the following Excel file:
    https://gitlab.com/open-darts/dwell/-/blob/compare_with_darts_well/examples/two_phase/dwell_darts_well_comparison.xlsx?ref_type=heads
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
        100 / 60 / 60 / 24,   # 100 seconds
                 ]

    for i, dt in enumerate(time_steps):
        coupled_model.run(dt)
        coupled_model.output.well_output_to_vtp(ith_step=i + 1, output_properties=output_props)

    coupled_model.print_timers()
else:
    save_dfm_well_props('I1', coupled_model)

    plot_heat_map_pcolormesh('I1', coupled_model)
    plot_heat_map_contourf('I1', coupled_model)
