"""
2-phase flow of a gaseous phase and liquid CO2-rich phase
Injection of liquid CO2 at a constant specific enthalpy into a depleted reservoir (to achieve convergence, the injected
specific enthalpy must be chosen in a way that the injected phase must contain a little bit of gaseous phase as well;
otherwise, if we inject only liquid CO2, after a while divergence may happen due to sudden complete vaporization of
liquid CO2 in blocks in which pressure may drop below the saturation pressure of CO2.)
"""

import numpy as np

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

    time_steps = [60/60/24/60,   # 1 minute
                  60.001/60/24/60,   # 2 minute
                  59.999 / 60 / 24 / 60,   # 3 minute
                  60.001 / 60 / 24 / 60,   # 4 minute
                  60.01 / 60 / 24 / 60,   # 5 minute
                  60.01 / 60 / 24 / 60,
                  60.01 / 60 / 24 / 60,
                  60.01 / 60 / 24 / 60,
                  60.01 / 60 / 24 / 60,
                  60.01 / 60 / 24 / 60,
                  ]
    time_steps = time_steps + list(10.01 / (24 * 60) * np.ones(5))

    for i, dt in enumerate(time_steps):
        if i >= 10:
            coupled_model.params.max_ts = 10/(24*60*60)

        coupled_model.run(dt)
        coupled_model.output_to_vtk(ith_step=i+1, output_properties=output_props)

else:
    well_data_file_path = "output/well_data.h5"
    h5_well_data = load_hdf5_to_dict(well_data_file_path)
    save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model)

    primary_vars_and_phase_props_file_address = "output/stored_primary_vars_and_phase_props.pkl"
    visualize_results_heat_maps(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model)
    visualize_results_line_graphs(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model, 3)
