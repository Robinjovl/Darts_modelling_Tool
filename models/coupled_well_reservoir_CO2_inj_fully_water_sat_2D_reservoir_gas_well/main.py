import numpy as np
import pandas as pd
from model import Model
import matplotlib.pyplot as plt

from darts.engines import redirect_darts_output
from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.wells.save_results import save_segments_primary_vars_and_phase_props

redirect_darts_output('run_log.log')
coupled_model = Model()
coupled_model.init()

if True:
    coupled_model.run(1)
    # n.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
    # n.run_python(300, restart_dt=1e-3)
    coupled_model.print_timers()
    coupled_model.print_stat()
    time_data = pd.DataFrame.from_dict(coupled_model.physics.engine.time_data)
    time_data.to_pickle("darts_time_data.pkl")
    # n.save_restart_data()
    coupled_model.save_data_to_h5('solution')
    writer = pd.ExcelWriter('time_data.xlsx')
    time_data.to_excel(writer, sheet_name='Sheet1')
    writer.close()

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

    #%% Plot primary variables instantly and store the figure
    centroids = coupled_model.reservoir.discretizer.centroids_all_cells[:,0]

    Xn = np.array(coupled_model.physics.engine.X, copy=False)
    num_primary_vars = coupled_model.physics.nc + coupled_model.physics.thermal
    num_reservoir_cells = coupled_model.reservoir.mesh.n_res_blocks

    p = Xn[0 : num_reservoir_cells*num_primary_vars : num_primary_vars]
    z_CO2 = Xn[1 : num_reservoir_cells*num_primary_vars : num_primary_vars]
    z_CH4 = Xn[2 : num_reservoir_cells*num_primary_vars : num_primary_vars]

    font_size = 18

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20,12))
    # Plot 1
    axs[0,0].plot(centroids, p, color='blue', linestyle='-', marker='o')
    axs[0,0].set_xlabel('Reservoir cell centroid position [m]', fontsize=font_size)
    axs[0,0].set_ylabel('Reservoir pressure [bar]', fontsize=font_size)
    axs[0,0].set_xlim(0, centroids[-1])
    axs[0,0].set_xticks(np.arange(0, 101, 10))
    # axs[0,0].set_ylim(0, None)
    axs[0,0].tick_params(axis='both', labelsize=font_size)
    axs[0,0].grid()
    # Plot 2
    axs[0,1].plot(centroids, z_CO2, color='blue', linestyle='-', marker='o')
    axs[0,1].set_xlabel('Reservoir cell centroid position [m]', fontsize=font_size)
    axs[0,1].set_ylabel('$CO_2$ overall mole fraction [-]', fontsize=font_size)
    axs[0,1].set_xlim(0, centroids[-1])
    axs[0,1].set_xticks(np.arange(0, 101, 10))
    # axs[0,1].set_ylim(0, None)
    axs[0,1].tick_params(axis='both', labelsize=font_size)
    axs[0,1].grid()
    # Plot 3
    axs[1,0].plot(centroids, z_CH4, color='blue', linestyle='-', marker='o')
    axs[1,0].set_xlabel('Reservoir cell centroid position [m]', fontsize=font_size)
    axs[1,0].set_ylabel('$CH_4$ overall mole fraction [-]', fontsize=font_size)
    axs[1,0].set_xlim(0, centroids[-1])
    axs[1,0].set_xticks(np.arange(0, 101, 10))
    # axs[1,0].set_ylim(0, None)
    axs[1,0].tick_params(axis='both', labelsize=font_size)
    axs[1,0].grid()

    # Add props for the figure
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig('output.png')

else:
    well_filepath = "output/well_data.h5"
    h5_well_data = load_hdf5_to_dict(well_filepath)
    save_segments_primary_vars_and_phase_props(h5_well_data, coupled_model)
