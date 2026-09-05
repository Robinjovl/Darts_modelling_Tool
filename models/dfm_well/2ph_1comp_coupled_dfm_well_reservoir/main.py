"""
Single-component (CO2), 2-phase (gas and liquid) system
Injection of liquid CO2 into a well and reservoir containing gaseous CO2

When using DFM wells:
    for a 2-phase system, use G as the name of the gaseous phase and L as the name of the liquid phase.
    for a 3-phase system, use G as the name of the gaseous phase, L_a, as the name of one liquid phase,
    and L_b as the name of the other liquid phase.
"""

import numpy as np
import os

from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh
from darts.pipes.viz.plot_heat_map_contourf import plot_heat_map_contourf
from darts.pipes.viz.plot_line_graphs import plot_line_graphs

from model import Model


redirect_darts_output('run.log')
coupled_model = Model()
coupled_model.init()
coupled_model.set_output()

if 1:
    output_props = coupled_model.physics.vars + coupled_model.output.properties + ["temperature"]
    coupled_model.output.output_to_vtk(ith_step=0, output_properties=output_props)   # saves initial reservoir conditions
    coupled_model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)   # saves initial well conditions

    report_steps = [
        0.5 / 24 / 60,  # 30 seconds
        0.5 / 24 / 60,  # 1 minute
        # 1 / 24 / 60,  # 2 minute
        # 1 / 24 / 60,  # 3 minute
        # 2 / 24 / 60,  # 5 minute
        # 5 / 24 / 60,  # 10 minute
        # 10 / 24 / 60,  # 20 minute
        # 10 / 24 / 60,  # 30 minute
        # 10 / 24 / 60,  # 40 minutes
        # 10 / 24 / 60,  # 50 minute
        # 10 / 24 / 60,  # 1 hour
        # 1 / 24,  # 2 hour
        # 1 / 24,  # 3 hour
        # 2 / 24,  # 5 hour
        # 5 / 24,  # 10 hour
        # 14 / 24,  # 1 day
        # 1,  # 2 day
        # 1,  # 3 day
        # 2,  # 5 day
        # 5,  # 10 day
        # 10,  # 20 day
        # 10,  # 30 day
        # 20,  # 50 day
        # 50,  # 100 day
        # 100,  # 200 day
        # 165,  # 365 day
    ]

    for i, dt in enumerate(report_steps):

        if i == 1:
            coupled_model.ts_control.dt_max = 5 / (24 * 60 * 60)
        elif i == 4:
            coupled_model.ts_control.dt_max = 10 / (24 * 60 * 60)
        elif i == 7:
            coupled_model.ts_control.dt_max = 20 / (24 * 60 * 60)
        elif i == 8:
            coupled_model.ts_control.dt_max = 30 / (24 * 60 * 60)
        elif i == 11:
            coupled_model.ts_control.dt_max = 60 / (24 * 60 * 60)
        elif i == 12:
            coupled_model.ts_control.dt_max = 10 / (24 * 60)
        elif i == 17:
            coupled_model.ts_control.dt_max = 2 / 24
        elif i == 20:
            coupled_model.ts_control.dt_max = 10 / 24
        elif i == 21:
            coupled_model.ts_control.dt_max = 1

        # # For injection at a constant WHP
        # if i == 1:
        #     coupled_model.ts_control.dt_max = 1 / (24 * 60 * 60)
        # elif i == 2:
        #     coupled_model.ts_control.dt_max = 5 / (24 * 60 * 60)

        coupled_model.run(dt)
        coupled_model.output.output_to_vtk(ith_step=i+1, output_properties=output_props)
        coupled_model.output.well_output_to_vtp(ith_step=i+1, output_properties=output_props)

    coupled_model.print_timers()
else:
    save_dfm_well_props('I1', coupled_model)

    plot_heat_map_pcolormesh('I1', coupled_model)
    plot_heat_map_contourf('I1', coupled_model, y_axis_tick_interval=250)

    # Use line graphs if injection rate is controlled because the wellhead state might change a lot (for numerical reasons)
    # at the beginning of simulation and this may create confusion if plot_heat_map_pcolormesh or plot_heat_map_contourf is used.
    plot_line_graphs('I1', coupled_model)
