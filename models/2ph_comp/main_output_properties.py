import numpy as np
import pandas as pd
import sys
import xarray as xr
import h5py
import matplotlib.pyplot as plt
import time

from darts.tools.hdf5_tools import *
from model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from darts.print_build_info import *

#%%

def prop_plot(path, property_array):
    for i, name in enumerate(property_array.keys()):
        plt.figure()
        plt.title(name)
        plt.plot(property_array[name][:, :1000].T)
        plt.savefig(os.path.join(path, name + '.png'))
    plt.close()

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

    print('--------------------------------------------------------------------------------------------------------------')

#%% Run model and save data

from model import Model
n = Model()
n.init()
n.set_output(output_folder='data\case_0', sol_filename='solution.h5',
             save_initial=True, all_phase_props=False, precision='d', compression = None, verbose=False)
print(type(n.reservoir))
# print(n.sol_filename)
# print(n.sol_filepath)
redirect_darts_output(n.output_folder + '/run_n.log')
Nt = 5
for i in range(Nt):
    n.run(1, verbose = False, save_well_data = True, save_reservoir_data = True)
n.print_timers()

# read_data(n.sol_filepath, n.well_filepath)
# read_data(sol_filepath='2ph_comp\data\case_0\solution_double_precision.h5',
#           well_filepath='2ph_comp\data\case_0\well_data.h5')

# time, cell_id, X1, var_names = n.output.read_specific_data(n.sol_filepath, timestep = None)
# time, cell_id, X2, var_names = n.output.read_specific_data('2ph_comp\data\case_0\solution_double_precision.h5', timestep = None)

#%% evaluate properties

# evaluate all available properties and timesteps from *.h5
time, property_array1 = n.output.output_properties(filepath = None, output_properties = None, timestep = None, engine = False)
# prop_plot(n.output_folder, property_array1)

# evaluate all available properties from last saved timestep
# time, property_array2 = n.output.output_properties(filepath = None, output_properties = None, timestep = -1, engine = False)
# prop_plot(n.output_folder, property_array2)

# evaluate specific properties from last saved timestep
# time, property_array3 = n.output.output_properties(filepath = None, output_properties = ['dens0'], timestep = -1, engine = False)
# prop_plot(n.output_folder, property_array3)

# evaluate all available properties and timesteps from engine
# time, property_array4 = n.output.output_properties(filepath = None, output_properties = None, engine = True)
# prop_plot(n.output_folder, property_array4)

#%% Well output

well_time_data = n.output.store_well_time_data()
types_of_well_time_data = ["phases_molar_rates", "phases_mass_rates", "phases_volumetric_rates", "components_molar_rates", "components_mass_rates", "advective_heat_rate", "BHP", "BHT"]
n.output.plot_well_time_data(types_of_well_time_data)

plt.figure()
plt.plot(well_time_data['time'], well_time_data['well_P1_molar_rate_oil_at_wh'], marker='o')
plt.xlabel("Simulation time [days]")
plt.ylabel("Oil molar production rate of well P1 [kmole/day]")
plt.tight_layout()
plt.show()

#%% plot results

xarray_data = n.output.output_to_xarray() # evaluate properties from *.h5 and save as *.nc file
for i in range(Nt + 1):
    n.output.plot_xarray(xarray_data, timestep=i, y=0)

# evaluate all properties, at every time step and output to .vtk
n.output.output_to_vtk()

# evaluate density at the last time step from *.h5 file
# n.output.output_to_vtk(ith_step = 4, output_directory = n.output_folder + '/vtk_files1', output_properties = ['dens0'], engine = False)

# evaluate density at the last time step from engine
# n.output.output_to_vtk(ith_step = 4, output_directory = n.output_folder + '/vtk_files2', output_properties = ['dens0'], engine = True)

#%% restart a model

from model import Model
m = Model()
m.init()
m.set_output(output_folder = 'data/restarted_from_case0',
             sol_filename = 'solution.h5', save_initial = False, all_phase_props = True, precision = 'd', verbose = False)
redirect_darts_output(m.output_folder + '/run_restarted_model.log')

# load point from which to restart
m.load_restart_data(reservoir_filename = 'data\case_0\solution.h5',
                    well_filename = 'data\case_0\well_data.h5',
                    timestep = -1)
m.params.first_ts = 1e-9
m.params.max_ts = 1.0
Nt = 5
for i in range(Nt):
    m.run(1, verbose = True, save_well_data = True, save_reservoir_data = True)
# m.print_timers()
# read_data(m.sol_filepath, m.well_filepath)

# evaluate properties
for i in range(Nt):
    time, property_array = m.output.output_properties(output_properties = m.output.properties, timestep = i)


xarray_data = m.output.output_to_xarray(output_properties = m.output.properties) # evaluate properties from *.h5 and save as *.nc file
for i in range(Nt + 1):
    m.output.plot_xarray(xarray_data, timestep=i, y=0)
m.output.output_to_vtk()

# # filter properties to only evaluate the properties of interest
# m.output.filter_phase_props(['dens_gas', 'dens_oil', 'sat_gas', 'sat_oil', 'nu_gas'])
# m.output.output_to_vtk(output_directory = m.output_folder + '/vtk_files1')
