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
    'GeoRising',
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

    n = model.Model()
    n.init()
    output_folder = os.path.join(os.getcwd(), 'testing')
    n.set_output(output_folder=output_folder, sol_filename='reservoir_solution.h5', save_initial=True,
                 all_phase_props=True, precision='d', verbose=False)
    redirect_darts_output(os.path.join(n.output_folder, 'run_n.log'))

    Nt = 5
    for i in range(Nt):
        n.run(5, verbose=False, save_well_data=True, save_reservoir_data=True)
    # read_data(n.sol_filepath, n.well_filepath)

    xarray_data = n.output.output_to_xarray() # evaluate properties from *.h5 and save as *.nc file
    for i in range(Nt + 1):
        n.output.plot_xarray(xarray_data, timestep=i, z = 1)

    n.output.output_to_vtk()

    well_rates_dict = n.output.plot_well_rates(['phases_molar_rates'])
    td = pd.DataFrame.from_dict(well_rates_dict)
    td.to_pickle(os.path.join(n.output_folder, "darts_time_data.pkl"))
    writer = pd.ExcelWriter(os.path.join(n.output_folder, 'time_data.xlsx'))
    td.to_excel(writer, sheet_name='Sheet1')
    writer.close()

    n.print_timers()

    os.chdir(initial_dir)

    print("------------------------------------------------------------------------------------------")




