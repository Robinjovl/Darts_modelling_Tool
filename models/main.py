import numpy as np
import pandas as pd
import sys, os, shutil
import xarray as xr
import h5py
import matplotlib.pyplot as plt
import time
import importlib.util

from darts.tools.hdf5_tools import *
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from darts.print_build_info import *

#%%

def read_data(sol_filepath, well_filepath, timestep = None):
    # read reservoir data
    time, cell_id, X, var_names = n.output.read_specific_data(sol_filepath, timestep = timestep)
    print('time', time)
    print('cell id:', cell_id)
    print('vars:', var_names)
    print('X[time, cell_id, variable], shape:', X.shape)

    # read well data
    time, cell_id, X, var_names = n.output.read_specific_data(well_filepath, timestep = timestep)
    print('time', time)
    print('cell id:', cell_id)
    print('vars:', var_names)
    print('X[time, cell_id, variable], shape:', X.shape)

#%%

accepted_dirs = [
    # '2ph_comp',
    # '2ph_comp_solid',
    # '2ph_do',
    # '2ph_do_thermal',
    # '2ph_geothermal',
    # '2ph_geothermal_mass_flux',
    # '3ph_comp_w',
    # '3ph_do',
    # '3ph_bo',
    # 'Uniform_Brugge',
    # 'Chem_benchmark_new',
    # 'CO2_foam_CCS',
    'GeoRising',
    # 'CoaxWell'
    ]

# Store the initial working directory
initial_dir = os.getcwd()

for mdir in accepted_dirs:
    # Navigate to the model directory
    os.chdir(mdir)
    print(f"Changed to directory: {os.getcwd()}")

    # Load model
    module_path = os.path.join(os.getcwd(), 'model.py')
    spec = importlib.util.spec_from_file_location("model", module_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    # init model
    n = model.Model()
    n.init()
    output_folder = os.path.join(os.getcwd(), 'data/n')
    n.set_output(output_folder=output_folder,
                 sol_filename='reservoir_solution.h5',
                 save_initial=True,
                 all_phase_props=True,
                 precision='d',
                 verbose=False)

    # n.output.print_simulation_parameters()
    redirect_darts_output(os.path.join(n.output_folder, 'run_n.log'))

    # run model
    Nt = 5
    for i in range(Nt):
        n.run(1, verbose=True, save_well_data=True, save_reservoir_data=True, save_well_data_after_run=False)
    # read_data(n.sol_filepath, n.well_filepath)

    # evaluating properties
    output_props = n.physics.vars + n.output.properties
    time_vector, property_array = n.output.output_properties(filepath = None, output_properties = output_props, timestep = -1, engine = False)

    # for var in property_array.keys():
    #     plt.figure()
    #     plt.title(var)
    #     plt.plot(property_array[var][0])
    #     plt.savefig(os.path.join(n.output_folder + '/figures', var))
    # plt.close('all')

    # # evaluating filtered properties
    # n.output.filter_phase_props(['dens_oil', 'mu_gas', 'x_oil_CO2'])
    # output_props = n.physics.vars + n.output.properties
    # time_vector, property_array = n.output.output_properties(filepath=None, output_properties=output_props, timestep=-1,
    #                                                          engine=False)

    # for var in property_array.keys():
    #     plt.figure()
    #     plt.title(var)
    #     plt.plot(property_array[var][0])
    #     plt.savefig(os.path.join(n.output_folder + '/filtered_figures', var))
    # plt.close('all')

    xarray_data = n.output.output_to_xarray(output_properties = output_props)

    # for var in output_props:
    #     timestep = -1
    #     z = 0
    #     plt.figure()
    #     xarray_data[var].isel(time=timestep, z=z).plot()
    #     plt.show()

    # for i in [-1]:
    #     n.output.plot_xarray(xarray_data, timestep=i, z = 0)

    # export to .vtk
    # n.output.output_to_vtk()

    if 1:
        # restart model
        m = model.Model()
        m.init()
        restarted_output_folder = os.path.join(os.getcwd(), 'data/m_restarted')
        m.set_output(output_folder = restarted_output_folder, sol_filename = 'reservoir_solution.h5',
                     save_initial = False, all_phase_props = False, precision = 'd', verbose=False)
        m.load_restart_data(reservoir_filename = n.sol_filepath, well_filename = n.well_filepath, timestep=-1)
        redirect_darts_output(os.path.join(m.output_folder, 'run_m.log'))

        for i in range(5):
            m.run(5)

        output_props = m.physics.vars + m.output.properties
        xarray_data = m.output.output_to_xarray(output_properties=output_props)
        # for i in range(5):
        #     m.output.plot_xarray(xarray_data, timestep=i, z=0)

        # m.output.output_to_vtk()

        # evaluate well time-series
        well_rates_dict = n.output.store_and_plot_well_time_data(plot_figs=True)

        # well_rates_dict = n.output.plot_well_rates(['phases_molar_rates'])
        # td = pd.DataFrame.from_dict(well_rates_dict)
        # td.to_pickle(os.path.join(n.output_folder, "darts_time_data.pkl"))
        # writer = pd.ExcelWriter(os.path.join(n.output_folder, 'time_data.xlsx'))
        # td.to_excel(writer, sheet_name='Sheet1')
        # writer.close()

        n.print_timers()

        os.chdir(initial_dir)
        print("------------------------------------------------------------------------------------------")