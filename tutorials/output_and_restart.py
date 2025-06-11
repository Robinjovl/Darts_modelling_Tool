"""
In this model the new features of the output class are demonstrated.
"""

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

RESTART = True # if true the models will be restarted

# list of models
accepted_dirs = [
    '2ph_comp',
    '2ph_comp_solid',
    '2ph_do',
    '2ph_do_thermal',
    '2ph_geothermal',
    '2ph_geothermal_mass_flux',
    '3ph_comp_w',
    '3ph_do',
    '3ph_bo',
    'GeoRising',
    'CoaxWell'
    ] # directory of cicd models

#%%

# Store the initial working directory
initial_dir = os.getcwd()
parent_dir = os.path.dirname(initial_dir)
models_dir = os.path.join(parent_dir, 'models')
os.chdir(models_dir)

for mdir in accepted_dirs:
    # Navigate to the model directory
    os.chdir(mdir)
    print(f"Changed to directory: {os.getcwd()}")

    # Load model
    module_path = os.path.join(os.getcwd(), 'model.py')
    spec = importlib.util.spec_from_file_location("model", module_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    """ --------------------- RUN A MODEL --------------------- """
    # init model
    n = model.Model()
    n.init()
    output_folder = os.path.join(os.getcwd(), 'output_data/n')
    n.set_output(output_folder=output_folder,
                 sol_filename='reservoir_solution_double.h5',
                 save_initial=True, # if you do not want to save initial conditions set this to False
                 all_phase_props=False,
                 precision='d',
                 verbose=False)

    # n.output.print_simulation_parameters()
    redirect_darts_output(os.path.join(n.output_folder, 'run_n.log'))

    # run model
    Nt = 2
    for i in range(Nt):
        n.run(365/10/2, verbose=True, save_well_data=True, save_reservoir_data=True, save_well_data_after_run=False)
    # read_data(n.sol_filepath, n.well_filepath)

    """ --------------------- EVALUATING PROPERTIES --------------------- """
    # when reading hdf5 files the timestep defaults to None. In which case all available timesteps are returned
    sol_filepath = n.sol_filepath # in the 'resrvoir_solution_double.h5' the state in every reservoir/grid block is saved
    time, cell_id, X, var_names = n.output.read_specific_data(sol_filepath, timestep = None)

    well_filpath = n.well_filepath  # in the well_data.h5 the state in the perforated reservoir block and well block is saved
    time_well, cell_id_well, X_well, var_names_well = n.output.read_specific_data(sol_filepath, timestep=None)

    # collect all available primary and secondary variables in a list
    primary_variables = n.physics.vars # state variables
    secondary_variables = n.output.properties # properties available in physics.property_itors()
    output_props =  primary_variables + secondary_variables # properties list

    # default behaviour for dartsmodel.output.output_properties() returns a dictionary of primary variables
    # for all timesteps saved in .../dartsmodel.output_folder/reservoir_solution.h5
    time_vector, property_array = n.output.output_properties(filepath = None, output_properties = None, timestep = None, engine = False)
    # save property array in the output folder
    n.output.save_property_array(time_vector, property_array)
    # load property array
    loaded_time_vector, loaded_property_array = n.output.load_property_array(file_directory=n.output_folder + '/property_array.h5')

    try:
        # save properties as an *.nc file
        # !! only for structured reservoir class !!
        xarray_dataset = n.output.output_to_xarray(output_properties = output_props)
        n.output.plot_xarray(xarray_dataset, timestep=Nt, x = None, y = None, z = 0)
    except:
        pass

    # evaluate a specific timestep
    time_vector, property_array = n.output.output_properties(timestep = 0) # initial conditions
    time_vector, property_array = n.output.output_properties(timestep = Nt) # final timestep
    # time_vector, property_array = n.output.output_properties(timestep = -1) # alternatively, final timestep

    # run model without saving anything
    n.run(1, verbose=True, save_well_data=False, save_reservoir_data=False, save_well_data_after_run=False)
    time_vector, property_array = n.output.output_properties(engine=True) # return dictionary from engine.X at engine.t

    # include only specified properties in a list
    time_vector, property_array = n.output.output_properties(output_properties = [n.physics.vars[0]]) # returns only pressure
    time_vector, property_array = n.output.output_properties(output_properties = n.output.properties) # returns only secondary variables defined in physics.property_containers[i].output_props

    # compare property_array evaluated from double and single precision saved data
    try:
        time_vector, property_array_single = n.output.output_properties(filepath = n.output_folder + '/reservoir_solution_single.h5',
                                                                        output_properties = n.output.properties)
        norm = np.mean(np.abs(property_array_single[n.output.properties[0]] - property_array[n.output.properties[0]]))
        print(norm)
    except:
        pass

    # different errors
    # time_vector, property_array = n.output.output_properties(output_properties='pressure')
    # time_vector, property_array = n.output.output_properties(timestep = 6) # raises an IndexError
    # time_vector, property_array = n.output.output_properties(timestep = 5.5) # raises a TypeError
    # time_vector, property_array = n.output.output_properties(filepath = output_folder + 'bublegum') # raises FileNotFoundError

    """ ----------------------------- WELL DATA ----------------------------- """

    # compute well time data
    time_data_dict = n.output.store_well_time_data()
    time_data_df = pd.DataFrame.from_dict(time_data_dict) # data frame for plotting

    # save well time data
    time_data_df.to_pickle(os.path.join(n.output_folder, "well_time_data.pkl"))  # as a pickle file
    writer = pd.ExcelWriter(os.path.join(n.output_folder, "well_time_data.xlsx"))  # as an excel file
    time_data_df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()

    n.output.plot_well_time_data(types_of_well_rates=["phases_volumetric_rates"])

    """ ------------------------ POST PROCESSING ------------------------ """
    m = model.Model() # a new model
    m.init()
    output_folder = os.path.join(os.getcwd(), 'output_data/n')
    m.set_output(output_folder=output_folder,
                 sol_filename='reservoir_solution_double.h5',
                 save_initial=False, # !! SET SAVE INITIAL TO FALSE OTHERWISE YOU WILL OVERWRITE YOUR DATA !!
                 all_phase_props=True,
                 precision='d',
                 verbose=False)

    time_vector, property_array = m.output.output_properties(output_properties = m.physics.vars + m.output.properties)
    m.output.save_property_array(time_vector, property_array) # save property array in the output folder

    # export and save properties as an *.nc file
    xarray_dataset = m.output.output_to_xarray(output_properties = m.physics.vars + m.output.properties)
    m.output.plot_xarray(xarray_dataset, timestep=Nt, x=None, y=None, z=0)

    # output_to_vtk
    try:
        # output_to_vtk acts identically to the output_properties
        m.output.output_to_vtk() # dumps all available data points into a default vtk file directory, just primary varibales
        m.output.output_to_vtk(output_directory = output_folder + '/vtks_everything',  output_properties = m.physics.vars + m.output.properties)
        # m.output.output_to_vtk(output_director = output_foler + '/vtks_primary_variables' + )
    except:
         print('nope')
         pass

    # filter properties to
    m.output.filter_phase_props(new_prop_keys = [m.output.properties[0]])
    # m.output.filter_phase_props(new_prop_keys=['somethimgsomething']) # raises a ValueError
    time_vector, property_array = m.output.output_properties(output_properties=m.output.properties)
    xarray_dataset = m.output.output_to_xarray(output_properties=m.output.properties)
    m.output.plot_xarray(xarray_dataset, timestep=Nt, x=None, y=None, z=0)

    """ --------------------- RESTART MODEL --------------------- """
    if RESTART:
        print('----------------- Restarting model ------------------')
        m_restarted = model.Model()
        m_restarted.init(restart=True)
        m_restarted.set_output(output_folder='output_data/n_restarted', # ensure you use a different output folder
                               all_phase_props=True
                               )

        reservoir_filename = n.sol_filepath # path to the data you want to restart from
        m_restarted.load_restart_data(reservoir_filename, timestep=1)
        m_restarted.run(1+365/10/2, restart_dt=1e-5) # use a smaller timestep than normal

        output_props = m_restarted.physics.vars + m_restarted.output.properties
        xarray_dataset = m_restarted.output.output_to_xarray(output_properties=output_props)
        for i in range(len(xarray_dataset['time'])):
            m_restarted.output.plot_xarray(xarray_dataset, timestep = i, z=0)

        try:
            m_restarted.output.output_to_vtk(output_properties=output_props)
        except:
            pass

        X = np.copy(m_restarted.physics.engine.X)
        X_restarted = np.copy(n.physics.engine.X)
        for i, var in enumerate(n.physics.vars):
            plt.figure(dpi = 100)
            plt.title(mdir)
            plt.plot(X[i::n.physics.n_vars], label = f'reference @ {n.physics.engine.t}')
            plt.plot(X_restarted[i::n.physics.n_vars], '--', label = f'restarted @ {m_restarted.physics.engine.t}')
            plt.legend()
            plt.ylabel(var)
            plt.savefig('output_data/' + f'{var}_comparison_restart.png')
            plt.close()

            # X[i::n.physics.n_vars] = X[i::n.physics.n_vars]/(np.max(X[i::n.physics.n_vars])-np.min(X[i::n.physics.n_vars]))
            # X_restarted[i::n.physics.n_vars] = X_restarted[i::n.physics.n_vars] / (np.max(X_restarted[i::n.physics.n_vars]) - np.min(X_restarted[i::n.physics.n_vars]))
        # assert np.isclose(X, X_restarted, rtol=0, atol=0.1).all()

    os.chdir(models_dir)
