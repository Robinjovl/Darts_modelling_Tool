"""
Uniform temperature in the entire reservoir
Smaller time steps are needed for convergence compared to isothermal scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import Model
from darts.engines import redirect_darts_output
from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.pipes.save_results import save_segments_primary_vars_and_phase_props
from darts.pipes.visualize_results_heat_maps import visualize_results_heat_maps
from darts.pipes.visualize_results_line_graphs import visualize_results_line_graphs

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

if 1:
    output_props = coupled_model.physics.property_operators[0].props_name
    coupled_model.output_to_vtk(ith_step=0, output_properties=output_props)   # saves initial conditions

    time_steps = [100/60/24/60,
                  ]

    for i, dt in enumerate(time_steps):
        coupled_model.run(dt)
        coupled_model.output_to_vtk(ith_step=i+1, output_properties=output_props)

else:
    well_data_file_path = "output/well_data.h5"
    h5_well_data = load_hdf5_to_dict(well_data_file_path)
    save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model)

    primary_vars_and_phase_props_file_address = "output/stored_primary_vars_and_phase_props.pkl"
    visualize_results_heat_maps(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model)
    # visualize_results_line_graphs(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model, 3)
