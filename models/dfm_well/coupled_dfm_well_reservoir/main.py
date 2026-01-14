"""
Single-component (CO2), 2-phase (gas and liquid) system
Injection of liquid CO2 into a well and reservoir containing gaseous CO2

Notes:
    When using a DFM well for a two-phase system, the order of phases is important: first the low density phase (gas),
    second the high density phase (liquid)
"""

import numpy as np
import os

from darts.engines import redirect_darts_output
from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.pipes.save_results import save_segments_primary_vars_and_phase_props
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh
from darts.pipes.viz.plot_heat_map_contourf import plot_heat_map_contourf

from model import Model

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()
coupled_model.set_output()

if 1:
    output_props = coupled_model.physics.vars + coupled_model.output.properties + ["temperature"]
    coupled_model.output_to_vtk(ith_step=0, output_properties=output_props)   # saves initial conditions

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
            coupled_model.data_ts.dt_max = 5 / (24 * 60 * 60)
        elif i == 4:
            coupled_model.data_ts.dt_max = 10 / (24 * 60 * 60)
        elif i == 7:
            coupled_model.data_ts.dt_max = 20 / (24 * 60 * 60)
        elif i == 8:
            coupled_model.data_ts.dt_max = 30 / (24 * 60 * 60)
        elif i == 11:
            coupled_model.data_ts.dt_max = 60 / (24 * 60 * 60)
        elif i == 12:
            coupled_model.data_ts.dt_max = 10 / (24 * 60)
        elif i == 17:
            coupled_model.data_ts.dt_max = 2 / 24
        elif i == 20:
            coupled_model.data_ts.dt_max = 10 / 24
        elif i == 21:
            coupled_model.data_ts.dt_max = 1
        coupled_model.run(dt)
        coupled_model.output_to_vtk(ith_step=i+1, output_properties=output_props)
else:
    well_data_file_path = os.path.join(coupled_model.output.output_folder, "well_data.h5")
    h5_well_data = load_hdf5_to_dict(well_data_file_path)
    save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model)

    primary_vars_and_phase_props_file_address = os.path.join(coupled_model.output.output_folder, "well_primary_vars_and_phase_props.pkl")
    plot_heat_map_pcolormesh(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model)
    plot_heat_map_contourf(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model, y_axis_tick_interval=250)
