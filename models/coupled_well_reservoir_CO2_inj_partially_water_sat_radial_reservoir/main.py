import pandas as pd

from model import Model
from darts.engines import redirect_darts_output
from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.wells.save_results import save_segments_primary_vars_and_phase_props
from darts.wells.visualize_results_heat_maps import visualize_results_heat_maps

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

if 1:
    coupled_model.run(1)
    time_data = pd.DataFrame.from_dict(coupled_model.physics.engine.time_data)
    time_data.to_pickle("darts_time_data.pkl")
    coupled_model.save_data_to_h5('solution')

    #%% Store output as .vtk files
    step = 0
    while True:
        try:
            # Export the VTK file
            # if step % 50 == 0:
            coupled_model.output_to_vtk(ith_step=step, output_directory='output')
            step += 1
        except:  # Bare except, since this is not an error
            break  # Gracefully exit the loop without printing anything

else:
    well_data_file_path = "output/well_data.h5"
    h5_well_data = load_hdf5_to_dict(well_data_file_path)
    save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model)

    primary_vars_and_phase_props_file_address = "output/stored_primary_vars_and_phase_props.pkl"
    visualize_results_heat_maps(primary_vars_and_phase_props_file_address, h5_well_data, coupled_model)