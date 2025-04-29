import os
import numpy as np
import h5py
import xarray as xr
import matplotlib.pyplot as plt
import shutil
import pickle
import pandas as pd

from darts.physics.blackoil import BlackOil
from darts.physics.super.physics import Compositional
from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH

from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.engines import value_vector, index_vector, timer_node, ms_well_vector, op_vector
from darts.tools.calc_well_rates import *
from darts.physics.base.operators_base import PropertyOperators

#%%

class Output:
    """
    Base class for all output related functionality
    """

    def __init__(self, timer: timer_node, reservoir, physics, op_list, params, well_head_conn_id, well_perf_conn_ids,
                 output_folder: str, sol_filename: str, well_filename: str, save_initial: bool,
                 all_phase_props: bool, precision: str, compression: str, verbose: bool):
        """
        Class constructor method for output related functionalities including saving primary variables (state variables),
        evaulating secondary variables (properties) and creating visualizations.

        :param timer: timer object to measure time spent saviving data, and evaluating properties
        :reservoir: reservoir object
        :param physics: physics object
        :param op_list: list of operator interpolators
        :param params: engine params
        :param well_head_conn_id: dictionary of wellhead indices of wells (values are integers)
        :param well_perf_conn_ids: dictionary of perforation indices of wells (values are lists)
        :param output_folder: output folder for saved data and figures
        :param sol_filename: hdf5 filename for saving reservoir solution
        :param well_filename: hdf5 filename for saving well solution
        :param save_initial: boolean flag to save initial conditions of reservoir
        :param all_phase_props: boolean flag to define properties (secondary variables) aaccording to a predefined list.
        :param compression: boolean flag to enable compression of hdf5 data
        :param verbose: boolean flag to enable verbose output
        """
        super().__init__()

        self.reservoir = reservoir
        self.physics = physics
        self.op_list = op_list
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)

        self.params = params
        self.well_head_conn_id = well_head_conn_id
        self.well_perf_conn_ids = well_perf_conn_ids
        self.verbose = verbose

        self.master_timer = timer
        self.timer = timer.node['output']
        self.timer.node["saving_reservoir_data"] = timer_node()
        self.timer.node["saving_well_data"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()
        self.timer.node["output_well_time_data"] = timer_node()

        self.output_folder = output_folder
        self.sol_filename = sol_filename
        self.well_filename = well_filename
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        self.precision = precision
        self.compression = compression
        self.precision_map = {
            'd': np.float64,
            's': np.float32
        }

        self.properties = list(self.physics.property_containers[0].output_props.keys())
        if len(self.properties) < self.physics.n_ops:
            self.n_ops = self.physics.n_ops
        else:
            self.n_ops = len(self.properties) + self.physics.n_vars

        if save_initial:
            self.save_data_to_h5(kind='reservoir')

        if all_phase_props:
            self.set_phase_properties()

        self.unit_dictionary = {
            'dens': '[kg/m3]',
            'densm': '[kmol/m3]',
            'sat': '[-]',
            'mu': '[cP]',
            'kr': '[-]',
            'pc': '[Bar]',
            'pressure': '[Bar]',
            'enthalpy': '[kJ]',
            'cond': '[kJ/m/day/K]',
            'temperature': '[K]'}

        self.set_units()

    def set_units(self):
        """
        Function to construct a dictionary of units for all the variables
        """

        self.variable_units = {}
        for name in self.physics.vars:
            try:
                self.variable_units[name] = self.unit_dictionary[name]
            except:
                self.variable_units[name] = ''

        for name in self.properties:
            try:
                self.variable_units[name] = self.unit_dictionary[name.split('_')[0]]
            except:
                self.variable_units[name] = ''

        return

    def set_phase_properties(self):
        """
        This function constructs a predefined set of property operators for the compositiol/geothermal physics class.
        """

        if type(self.physics) is Compositional or type(self.physics) is BlackOil:
            phase_props_labels = ['dens', 'densm', 'sat', 'mu', 'kr', 'pc', 'enthalpy', 'cond']
            self.physics.property_itor = {}

            for region in self.physics.regions:  # loop over the different sets of operators
                pc = self.physics.property_containers[region]
                temp_dict = {} # output_properties dictionary

                # Loop through each property label and phase name
                for i, name in enumerate(phase_props_labels):
                    for j in range(len(pc.phase_props[i])):
                        temp_dict[f"{name}_{self.physics.phases[j]}"] = lambda i=i, j=j: pc.phase_props[i][j]

                # Add molar phase fractions
                for i in range(pc.x.shape[1]):
                    for j in range(pc.x.shape[0]):
                        temp_dict[f"x_{self.physics.phases[j]}_{pc.components_name[i]}"] = lambda i=i, j=j: pc.x[j, i]

                self.physics.property_operators[region] = PropertyOperators(pc, self.physics.thermal, temp_dict)
                self.physics.property_itor[region] = self.physics.create_interpolator(self.physics.property_operators[region],
                                                                                    n_ops=self.physics.n_ops,
                                                                                    axes_min=self.physics.axes_min, axes_max=self.physics.axes_max,
                                                                                    platform='cpu', algorithm='multilinear',
                                                                                    mode='adaptive', precision='d',
                                                                                    timer_name='property %d interpolation' % region,
                                                                                    region=str(region))

                # Assign the temporary dictionary to output_props for the region
                self.physics.property_containers[region].output_props = temp_dict
                self.n_ops = self.physics.n_ops

        elif type(self.physics) is Geothermal or type(self.physics) is GeothermalPH:
            phase_props_labels = ['dens', 'densm', 'sat', 'mu', 'kr', 'pc', 'enthalpy'] # 'cond'
            self.physics.property_itor = {}

            for region in self.physics.regions:  # loop over the different sets of operators
                pc = self.physics.property_containers[region]
                temp_dict = {}

                temp_dict['temperature'] = lambda: pc.temperature # add temperature

                # Loop through each property label and phase name
                for i, name in enumerate(phase_props_labels):
                    # for j, phase_name in enumerate(self.physics.property_containers[region].nph):
                    for j in range(self.physics.property_containers[region].nph):
                        temp_dict[f"{name}_{self.physics.phases[j]}"] = lambda i=i, j=j: pc.phase_props[i][j]

                self.physics.property_operators[region] = PropertyOperators(pc, thermal = False, props = temp_dict)
                self.physics.property_itor[region] = self.physics.create_interpolator(self.physics.property_operators[region],
                                                                                      n_ops = self.physics.property_operators[region].n_ops,
                                                                                      axes_min = self.physics.axes_min, axes_max = self.physics.axes_max,
                                                                                      platform = 'cpu', algorithm='multilinear',
                                                                                      mode='adaptive', precision='d',
                                                                                      timer_name='property %d interpolation' % region,
                                                                                      region=str(region))

                # Assign the temporary dictionary to output_props for the region
                self.physics.property_containers[region].output_props = temp_dict
                self.n_ops = self.physics.property_operators[0].n_ops

        # Update the properties list
        self.properties = list(self.physics.property_containers[0].output_props.keys())

        return

    def filter_phase_props(self, new_prop_keys):
        """
        Filter default list of properties to only evaluate desired properties listed in new_prop_keys.

        :param new_prop_keys: list of properties to keep
        :type new_prop_keys: list

        :raises ValueError: If any key in `new_prop_keys` is not an available property.
        """
        for region in self.physics.regions:
            output_dictionary = self.physics.property_containers[region].output_props
            prop_keys = list(output_dictionary.keys())

            # Warn if any key is missing in the available properties
            for key in new_prop_keys:
                if key not in prop_keys:
                    raise ValueError(
                        f"The following properties are not available: {key}. "
                        f"Choose properties from: {prop_keys}"
                    )

            for key in prop_keys:
                if key not in new_prop_keys:
                    del output_dictionary[key]

            self.physics.property_containers[region].output_props = output_dictionary

            # Initialize physics and engine settings
            self.physics.init_physics()
            self.physics.engine.init(self.reservoir.mesh,
                                     ms_well_vector(self.reservoir.wells),
                                     op_vector(self.op_list),
                                     self.params,
                                     self.master_timer.node["simulation"])
            self.properties = list(output_dictionary.keys())

        return 0

    def save_property_array(self, time_vector, property_array, filename="property_array.h5"):
        """
        Saves property_array to an HDF5 file.

        :param time_vector : Array of timesteps
        :param property_array : Dictionary where keys are property names and values are NumPy arrays.
        :param filename : Name of the HDF5 file to save to.
        """
        
        compression_level = 2
        output_directory = os.path.join(self.output_folder, filename)
        
        with h5py.File(output_directory, "w") as h5f:
            # Save the time vector with compression
            h5f.create_dataset("time_vector", data=time_vector, compression="gzip", compression_opts=compression_level)

            # Save each property array with compression
            for key, array in property_array.items():
                h5f.create_dataset(key, data=array, compression="gzip", compression_opts=compression_level)
        
        return 0 

    def load_property_array(self, file_directory="property_array.h5"):
        """
        Load saved properties back into a dictionary.
        :param file_directory : filepath to saved property_array.h5

        :return time_vector:  available timesteps
        :return property_array: dictionary of properties
        """
        property_array = {}

        with h5py.File(file_directory, "r") as h5f:
            # Load time vector
            time_vector = np.array(h5f["time_vector"])

           # Load each property array
            for key in h5f.keys():
                if key != "time_vector":  # Skip time vector in property dictionary
                    property_array[key] = np.array(h5f[key])

        return time_vector, property_array
        
    def print_simulation_parameters(self, mode = 'table'):
        """
        Function that prints class variables into a .txt file
        """
        filepath = os.path.join(self.output_folder, 'simulation_input_parameters.txt')

        if mode == 'dump':
            obj_list = [self.params, self.reservoir, self.physics]
            with open(filepath, 'w') as f:
                for i, obj in enumerate(obj_list):
                    f.write(f"------- {i + 1}: {obj.__class__.__name__} -------\n")
                    for attr in dir(obj):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(obj, attr)
                                f.write(f"{attr}: {value}\n")
                            except Exception as e:
                                f.write(f"{attr}: <error: {e}>\n")
                    f.write('\n')  # Add a blank line between objects
        else:
            with open(filepath, 'w') as f:
                f.write("-----------------------------PHYSICS------------------------\n")
                f.write("-- Physics:\n")
                f.write(f"{type(self.physics)}\n")

                f.write("-- Components:\n")
                f.write(f"{self.physics.components}\n")

                f.write("-- Phases:\n")
                f.write(f"{self.physics.phases}\n")

                f.write("-- Numerical variables:\n")
                f.write(f"{self.physics.vars}\n")

                f.write("-- Thermal:\n")
                f.write(f"{self.physics.thermal}\n")

                f.write("-- State specification:\n")
                f.write(f"{self.physics.state_spec}\n")

                f.write("-- OBL axes minimums:\n")
                f.write(f"{self.physics.axes_min[:]}\n")

                f.write("-- OBL axes maximums:\n")
                f.write(f"{self.physics.axes_max[:]}\n")

                f.write("-- OBL axes maximums:\n")
                f.write(f"{self.physics.n_axes_points[:]}\n")

                f.write("-- Regions:\n")
                f.write(f"{self.physics.regions}\n")

                f.write("------------------------RESERVOIR-----------------------\n")
                f.write("-- Reservoir:\n")
                f.write(f"{type(self.reservoir)}\n")

                f.write("-- n_blocks:\n")
                f.write(f"{self.reservoir.mesh.n_blocks}\n")

                f.write("-- n_res_blocks:\n")
                f.write(f"{self.reservoir.mesh.n_res_blocks}\n")

                # UPDATE THIS PART FOR THE NEW CONTROLS
                ### UPDATE TO NEW WELL CONTROLS
                # f.write("------------------------WELLS-----------------------\n")
                # f.write("-- wells and perforations:\n")
                # for i, w in enumerate(self.reservoir.wells):
                #     if 'I' in w.name:  # Injector well
                #         if hasattr(w.control, 'target_pressure'):
                #             f.write(
                #                 f"Well {w.name} perforated at {w.perforations} with {type(w.control).__name__} control at pressure {w.control.target_pressure} and injection stream {w.control.injection_stream}.\n")
                #         elif hasattr(w.control, 'target_rate'):
                #             f.write(
                #                 f"Well {w.name} perforated at {w.perforations} with {type(w.control).__name__} control at rate {w.control.target_rate} and injection stream {w.control.injection_stream}.\n")
                #     else:  # Producer well
                #         if hasattr(w.control, 'target_pressure'):
                #             f.write(
                #                 f"Well {w.name} perforated at {w.perforations} with {type(w.control).__name__} control at pressure {w.control.target_pressure}.\n")
                #         elif hasattr(w.control, 'target_rate'):
                #             f.write(
                #                 f"Well {w.name} perforated at {w.perforations} with {type(w.control).__name__} control at rate {w.control.target_rate}.\n")

        return 0

    def configure_h5_output(self, filename: str, cell_ids, description, add_static_data: bool = False):
        """
        Configuration of *.h5 output

        :param filename: *.h5 filename
        :param cell_ids: np.array of cell indexes for output
        :param description: description for *.h5
        :param add_static_data: flag to add static output
        """

        with h5py.File(filename, 'w') as f:
            # add static data group
            if add_static_data:
                static_group = f.create_group('static')
                block_m = np.array(self.reservoir.mesh.block_m, copy=False)
                block_p = np.array(self.reservoir.mesh.block_p, copy=False)
                static_group.create_dataset('block_m', data=block_m)
                static_group.create_dataset('block_p', data=block_p)

            # add dynamic data group
            dynamic_group = f.create_group('dynamic')
            dynamic_group.create_dataset('time', shape=(0,), maxshape=(None,), dtype=self.precision_map[self.precision])

            # add solution
            if self.reservoir.mesh.n_blocks > 0 and self.physics.n_vars > 0:
                nb = cell_ids.size
                cell_ids_dataset = dynamic_group.create_dataset('cell_id', shape=(nb,), dtype=np.int32)
                cell_ids_dataset[:] = cell_ids

            dynamic_group.create_dataset('X', shape=(0, nb, self.physics.n_vars),
                                         maxshape=(None, nb, self.physics.n_vars),
                                         dtype=self.precision_map[self.precision],
                                         compression=self.compression)

            # add variable names
            datatype = h5py.special_dtype(vlen=str)  # dtype for variable-length strings
            dynamic_group.create_dataset('variable_names', data=np.array(self.physics.vars, dtype=datatype))
            #var_names = dynamic_group.create_dataset('variable_names', (self.physics.n_vars,), dtype=datatype)
            #var_names[:] = self.physics.vars

            # write brief description
            f.attrs['description'] = description

    def configure_output(self, kind: str):
        """
        Configuration of output

        :param kind: 'well' for well output or 'solution' to write the whole solution vector
        :type kind: str
        :param restart: Boolean to check if existing file should be overwritten or appended
        :type restart: bool
        """

        # Ensure the directory and subdirectory exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, 'figures'), exist_ok=True)

        # solution ouput
        if kind == 'reservoir':
            sol_output_path = os.path.join(self.output_folder, self.sol_filename)
            if os.path.exists(sol_output_path): #and not restart:
                os.remove(sol_output_path)
            self.configure_h5_output(filename=sol_output_path, cell_ids=np.arange(self.reservoir.mesh.n_res_blocks),
                                     add_static_data=False, description='Reservoir data')

        # Find relevant connections for well data
        if kind == 'well':
            block_m = np.array(self.reservoir.mesh.block_m, copy=False)
            block_p = np.array(self.reservoir.mesh.block_p, copy=False)
            well_conn_ids = np.argwhere(block_p >= self.reservoir.mesh.n_res_blocks)[:, 0]
            self.id_well_data = np.unique(block_m[well_conn_ids])

            # well output
            well_output_path = os.path.join(self.output_folder, self.well_filename)
            if os.path.exists(well_output_path):
                os.remove(well_output_path)
            self.configure_h5_output(filename=well_output_path, cell_ids=self.id_well_data,
                                     add_static_data=True, description='Well data')

        if hasattr(self, 'output_configured'):
            self.output_configured.append(kind)
        else:
            self.output_configured = [kind]

    def save_specific_data(self, filename):
        """
        Function to write output to *.h5 file

        :param filename: path to *.h5 filename to append data to
        :type filename: str
        """
        X = np.array(self.physics.engine.X, copy=False)  # [:self.physics.n_vars*self.reservoir.n]

        # Open the HDF5 file in append mode
        with h5py.File(filename, "a") as f:
            # Append to time dataset under the dynamic group
            time_dataset = f["dynamic/time"]
            time_dataset.resize((time_dataset.shape[0] + 1,))
            time_dataset[-1] = self.physics.engine.t

            cell_id = f["dynamic/cell_id"][:]

            x_dataset = f["dynamic/X"]
            x_dataset.resize((x_dataset.shape[0] + 1, x_dataset.shape[1], x_dataset.shape[2]))
            x_dataset[x_dataset.shape[0] - 1, :, :] = X.reshape((self.reservoir.mesh.n_blocks, self.physics.n_vars))[cell_id]

        if self.verbose:
            print(f'Saving data to {filename} at time = {self.physics.engine.t}')

    def save_data_to_h5(self, kind):
        """
        Function to write output solution or well output to *.h5 file

        :param kind: 'well' for well output or 'solution' to write the whole solution vector
        :type kind: str
        """

        if not hasattr(self, 'output_configured') or kind not in self.output_configured:
            self.configure_output(kind=kind)

        if kind == 'well':
            path = os.path.join(self.output_folder, self.well_filename)
            self.timer.start(); self.timer.node['saving_well_data'].start()
            self.save_specific_data(path)
            self.timer.node['saving_well_data'].stop(); self.timer.stop()

        elif kind == 'reservoir':
            path = os.path.join(self.output_folder, self.sol_filename)
            self.timer.start(); self.timer.node['saving_reservoir_data'].start()
            self.save_specific_data(path)
            self.timer.node['saving_reservoir_data'].stop(); self.timer.stop()

        else:
            print("Please use either kind='well' or kind='solution' in save_data_to_h5")

    def read_specific_data(self, filename: str, timestep: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts time and data (primary variables) from an HDF5 file for a given timestep

        :param filename: Path to the HDF5 file
        :type file_path: str
        :param timestep: The timestep to extract data for.
        :type timestep: int

        :return: time, ndarray with extracted timesteps
        :return cell_id: ndarray with cell_id of each of the saved grid blocks
        :return X: ndarray with data, shape: (number_of_timesteps, number_of_cells, number_of_vars)
        :return var_names: ndarray with variable names

        :raises FileNotFoundError: If the file does not exist.
        :raises IndexError: If `timestep` is out of range.
        """

        try:
            with h5py.File(filename, 'r') as file:
                if timestep is None:
                    # datapoints = file['dynamic/X'].shape[0] * file['dynamic/X'].shape[1] * file['dynamic/X'].shape[2]
                    # print('WARNING: %s contains %d data points...' % (filename, datapoints)) if self.verbose

                    cell_id = file['dynamic/cell_id'][:]
                    var_names = file['dynamic/variable_names'][:]
                    time = file['dynamic/time'][:]
                    X = file['dynamic/X'][:]
                else:

                    if not isinstance(timestep, int):
                        raise TypeError(f"Expected 'timestep' to be an int, but got {type(timestep).__name__}")

                    cell_id = file['dynamic/cell_id'][:]
                    var_names = file['dynamic/variable_names'][:]

                    try:
                        time = file['dynamic/time'][timestep].reshape(1)
                        X = file['dynamic/X'][timestep].reshape(1, len(cell_id), len(var_names))

                    except IndexError:
                        raise IndexError(
                            f"Timestep {timestep} does not exist in {filename}.")

            for i, name in enumerate(var_names):
                var_names[i] = name.decode()

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}.")

        return time, cell_id, X, var_names

    def output_properties(self, filepath: str = None, output_properties: list = None, timestep: int = None, engine = False) -> tuple[np.ndarray, dict]:
        """
        Evaluates and returns properties from saved data (HDF5 file) or a simulation engine.

        :param filepath: Path to the solution HDF5 file. Defaults to None, in which case the dartsmodel.sol_filepath is used.
        :type filepath: str, optional
        :param output_properties: List of properties to evaluate. Defaults to None, which returns an array containing only state variables.
        :type output_properties: list, optional
        :param timestep: Timestep at which to evaluate properties. Defaults to None, which will evaluate all saved timesteps.
        :type timestep: int, optional
        :param engine: If true, state variables are evaluated directly from engine.X. Defaults to False, which reads properties from the HDF5 file.
        :type engine: bool, optional
    
        :return property_array: A dictionary where keys are primary/secondary variables and values are NumPy arrays of the requested properties for each grid block. The shape of each array is (number_of_timesteps, number_of_gridblocks).
        :type property_array: dict
        :return timesteps: A NumPy array of the time labels.
        :type timesteps: np.ndarray

        :raises KeyError: If specified property in `output_properties` is not found in any property container
        :raises TypeError: If output_properties is not a list
        """

        if output_properties is not None and not isinstance(output_properties, list):
            raise TypeError(f"Expected 'output_properties' to be a list, but got {type(output_properties).__name__}.")

        if not engine:
            # Evaluate properties from the HDF5 file
            if filepath is None: # Establish filepath/name to HDF5 file
                path = os.path.join(self.output_folder, self.sol_filename)
            else:
                path = filepath
            timesteps, cell_id, X, var_names = self.read_specific_data(path, timestep) # Read data from HDF5 file
        else:
            # Evaluate properties from the physics.engine.X
            timesteps = np.array(self.physics.engine.t).reshape(1,) # current time
            cell_id = np.arange(self.reservoir.mesh.n_res_blocks) # cell ids
            X = np.array(self.physics.engine.X[:self.physics.n_vars*self.reservoir.mesh.n_res_blocks], copy = True) # solution at current time
            var_names = self.physics.vars # primary variable names

        # n_ops = self.n_ops
        n_vars = len(var_names) # number of primary variables
        nb = len(cell_id) # number of grid blocks
        # complete list of properties
        output_properties = output_properties if output_properties is not None else list(self.physics.vars)

        # List of primary variables i.e. state variables
        primary_props = [prop for prop in output_properties if prop in var_names]
        primary_prop_idxs = {prop: list(var_names).index(prop) for prop in primary_props}

        # List of secondary properties
        secondary_props = [prop for prop in output_properties if prop not in var_names]
        secondary_prop_idxs = {}
        for prop in secondary_props:
            for container in self.physics.property_containers.values():
                if prop in container.output_props:
                    secondary_prop_idxs[prop] = list(container.output_props.keys()).index(prop)
                    break
            else:
                raise KeyError(f"Secondary property '{prop}' not found in any property container.")

        # define property array dictionary
        property_array = {prop: np.zeros((len(timesteps), nb)) for prop in primary_props + secondary_props}

        # Loop over available timesteps
        for k, t in enumerate(timesteps):
            # Extract primary properties from X vector
            for var_name, var_idx in primary_prop_idxs.items():
                if engine is False:
                    property_array[var_name][k] = X[k, :nb, var_idx]
                else:
                    property_array[var_name][k] = X[var_idx::n_vars]

            # Interpolate secondary properties
            if secondary_props: # if empty this part is skipped
                if engine is False:
                    state = value_vector(np.stack([X[k, :nb, j] for j in range(n_vars)]).T.flatten().astype(np.float64))
                else:
                    state = value_vector(np.stack([X[j::n_vars] for j in range(n_vars)]).T.flatten())

                values = value_vector(np.zeros(self.n_ops * nb))
                values_numpy = np.array(values, copy=False)
                dvalues = value_vector(np.zeros(self.n_ops * nb * n_vars))

                for region, prop_itor in self.physics.property_itor.items():

                    block_idx = np.where(self.op_num == region)[0].astype(np.int32)
                    prop_itor.evaluate_with_derivatives(state, index_vector(block_idx), values, dvalues)

                    for prop_name, prop_idx in secondary_prop_idxs.items():
                        temp = values_numpy[prop_idx::self.n_ops]
                        property_array[prop_name][k][block_idx] = temp[block_idx]

        return timesteps, property_array

    def output_to_vtk(self, sol_filepath : str = None, ith_step: int = None, output_directory: str = None, output_properties: list = None, engine : bool = False):
        """
        Function to export results at timestamp t into `.vtk` format for viewing in Paraview.

        :param filepath: Path to the solution HDF5 file. Defaults to None, in which case the dartsmodel.sol_filepath is used.
        :type filepath: str, optional
        :param ith_step: i'th reporting step indicates which timestep to create a .vtk from. Defaults to None, in which case all saved data points are evaluated.
        :type ith_step: int
        :param output_directory: directory of where to save .vtk file. Defaults to none in which case the 'self.output_folder/vtk' is used.
        :type output_directory: str
        :param output_properties: List of properties to include in .vtk file. Defaults to None in which case only primary (state) variables are evaluated.
        :type output_properties: list
        """
        self.timer.start(); self.timer.node["vtk_output"].start()

        # Set default output directory
        if output_directory is None:
            output_directory = os.path.join(self.output_folder, 'vtk_files')
        os.makedirs(output_directory, exist_ok=True)

        timesteps, property_array = self.output_properties(self.sol_filepath if sol_filepath is None else sol_filepath,
                                                           output_properties,
                                                           ith_step,
                                                           engine)

        # prop_names = {prop: i for i, prop in enumerate(property_array.keys())}
        # units to prop names
        prop_names = {}
        for i, name in enumerate(property_array.keys()):
            prop_names[name] = name + self.variable_units[name]
        # prop_names = {prop: f"{prop} {self.variable_units.get(prop, '')}" for prop in property_array.keys()}

        for t, time in enumerate(timesteps):
            data = np.zeros((len(property_array), self.reservoir.mesh.n_res_blocks))
            for i, name in enumerate(property_array.keys()):
                data[i, :] = property_array[name][t]

            if ith_step is None:
                self.reservoir.output_to_vtk(t, time, output_directory, prop_names, data)
            else:
                self.reservoir.output_to_vtk(ith_step, time, output_directory, prop_names, data)

        self.timer.node["vtk_output"].stop(); self.timer.stop()

    def output_to_xarray(self, filepath: str = None, output_properties: list = None, timestep: int = None, engine: bool = False) -> xr.Dataset:
        """
        Generates an xarray Dataset of properties and saves it as a NetCDF file.
        State variables area obtained from the engine or *.h5 file.
        Properties are interpolated by the property iterator.

        :param filepath: Path to the solution HDF5 file. Defaults to None, in which case the dartsmodel.sol_filepath is used.
        :type filepath: str, optional
        :param output_properties: List of properties to include in the dataset. If None, all properties are included.
        :type output_properties: list, optional
        :param timestep: Specific timestep to output. If None, all timesteps are included.
        :type timestep: int, optional
        :param engine: import state variable from engine if True. Default is False.
        :type engine: bool, optional
        :returns: xarray Dataset containing the property data.
        :rtype: xarray.Dataset
        """

        # Interpolate properties
        time, data = self.output_properties(filepath, output_properties, timestep, engine)
        props = list(data.keys())

        # Initialize coords and data_vars for Xarray Dataset
        array_shape = (len(time), self.reservoir.nz, self.reservoir.ny, self.reservoir.nx)
        for prop, array in data.items():
            data[prop] = array.reshape(array_shape)

        # Initialize coords and data_vars for Xarray Dataset
        if type(self.reservoir).__name__ == 'StructReservoir':
            dx, dy, dz = self.reservoir.global_data['dx'], self.reservoir.global_data['dy'], self.reservoir.global_data['dz']
            x = np.cumsum(dx[:, 0, 0]) - dx[0, 0, 0] * 0.5
            y = np.cumsum(dy[0, :, 0]) - dy[0, 0, 0] * 0.5
            z = np.cumsum(dz[0, 0, :]) - dz[0, 0, 0] * 0.5
        else:
            raise ValueError('Reservoir type is not supported.')

        coords = {'time': time, 'z': z, 'y': y, 'x': x}
        data_vars = {prop: (list(coords.keys()), data[prop]) for prop in props}
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)

        # Attach units
        dataset['time'].attrs['units'] = 'days'
        dataset['x'].attrs['units'] = 'm'
        dataset['y'].attrs['units'] = 'm'
        dataset['z'].attrs['units'] = 'm'
        for var in data.keys():
            try:
                # first_part = var.split('_')[0]
                dataset[var].attrs['units'] = self.variable_units[var][1:-1]
            except:
                dataset[var].attrs['units'] = ''

        if self.precision == 'd':
            encoding = {prop: {'dtype': 'float64'} for prop in data.keys()}
        else:
            encoding = {prop: {'dtype': 'float32'} for prop in data.keys()}

        # Save to NetCDF with specified encoding
        dataset.to_netcdf(os.path.join(self.output_folder, self.sol_filename[:-3] + '.nc'),
                          engine='netcdf4',
                          encoding=encoding)

        return dataset

    def plot_xarray(self, xarray_data, timestep: int = -1, x: int = None, y: int = None, z: int = None):
        """
        :param xarray_data: xarray data set
        :param timestep: time index
        :param x: index in x-dimension
        :param y: index in y-dimension
        :param z: index in z-dimension
        """

        from darts.reservoirs.struct_reservoir import StructReservoir
        if type(self.reservoir) is not StructReservoir:
            raise AttributeError("Reservoir class must be exactly of type StructReservoir.")

        output_directory = os.path.join(self.output_folder, 'figures')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        assert isinstance(timestep, int) and timestep < len(xarray_data['time']), \
            f"Timestep should be an integer less than {len(xarray_data['time'])}."

        var_names = list(xarray_data.data_vars)
        nrows = len(var_names)
        for i, var in enumerate(var_names):
            fig = plt.figure()
            if z is not None:
                assert z < len(xarray_data['z']), 'z-level step should be less than %d' % len(xarray_data['z'])
                xarray_data[var].isel(time=timestep, z=z).plot()
                # plot.colorbar.set_label(self.variable_units[var])
                plt.savefig(output_directory + '/%s ts%d z%d.png'%(var, timestep, z))

            elif y is not None:
                assert y < len(xarray_data['y']), 'y-level step should be less than %d' % len(xarray_data['y'])
                xarray_data[var].isel(time=timestep, y=y).plot()
                plt.savefig(output_directory + '/%s ts%d y%d.png' % (var, timestep, y))

            elif x is not None:
                assert x < len(xarray_data['x']), 'x-level step should be less than %d' % len(xarray_data['x'])
                xarray_data[var].isel(time=timestep, x=x).plot()
                plt.savefig(output_directory + '/%s ts%d zx%d.png'%(var, timestep, z))

            else:
                # model is a 1D reservoir
                xarray_data[var].isel(time=timestep).plot()
                plt.savefig(output_directory + '/%s ts%d.png' % (var, timestep))
        plt.close('all')

    def store_well_time_data(self, types_of_well_rates=None):
        """
        Compute and store well time data including rates and bottom-hole conditions (BHT and BHP).
        Rates are calculated for each perforation and also total rate of each well. Total rates are calculated using
        two different methods:
        1- summing up the rates of perforations
        2- calculating the rates directly at the wellhead connection

        :param types_of_well_rates: List of types of well rates that can be computed:
        "phases_molar_rates"
        "phases_mass_rates"
        "phases_volumetric_rates"
        "components_molar_rates"
        "components_mass_rates"
        "advective_heat_rate" for thermal scenarios
        :type types_of_well_rates: list
        """
        # Start timer for store_well_time_data
        self.timer.start(); self.timer.node["output_well_time_data"].start()

        h5_well_data = load_hdf5_to_dict(self.well_filepath)
        pc = self.configure_physics()

        time = h5_well_data['dynamic']['time']
        time_data_dict = {'time': time}

        perfs_conn_ids, well_head_conn_ids, geometric_WI, well_head_conn_trans = self.get_connection_info()

        if types_of_well_rates is None:
            types_of_well_rates = [
                "phases_molar_rates",
                "phases_mass_rates",
                "phases_volumetric_rates",
                "components_molar_rates",
                "components_mass_rates",
            ]
            if self.physics.thermal:
                types_of_well_rates.append("advective_heat_rate")

        # Store BHP and BHT
        self.store_bhp_bht(h5_well_data, time_data_dict, pc)

        for rate_type in types_of_well_rates:
            # Compute perforation rates
            rates_perfs = calc_rates_at_connections(h5_well_data, perfs_conn_ids, geometric_WI, self.physics.thermal, pc, rate_type)
            # Store perforation rates
            self.store_perf_rates(time_data_dict, rates_perfs, rate_type, pc)
            # Store well rates by summing perforation rates
            self.store_well_rates_sums(time_data_dict, rates_perfs, rate_type, pc)
            # Compute wellhead rates
            rates_wellhead = calc_rates_at_connections(h5_well_data, well_head_conn_ids, well_head_conn_trans, self.physics.thermal, pc, rate_type)
            # Store wellhead rates
            self.store_wellhead_rates(time_data_dict, rates_wellhead, rate_type, pc)

        # Export time_data_dict
        df = pd.DataFrame(time_data_dict)
        df.to_pickle(os.path.join(self.output_folder, 'well_time_data.pkl'))
        with pd.ExcelWriter(os.path.join(self.output_folder, 'well_time_data.xlsx')) as w:
            df.to_excel(w, sheet_name='Sheet1')

        # End timer for store_well_time_data
        self.timer.node["output_well_time_data"].stop(); self.timer.stop()
        return time_data_dict

    def plot_well_time_data(self, types_of_well_rates=None):
        """
        Plots well time data that are specified in the list types_of_well_time_data over time, including
        phases_molar_rates, phases_mass_rates, phases_volumetric_rates, components_molar_rates, components_mass_rates,
        advective_heat_rate, BHP (bottom-hole pressure), and BHT (bottom-hole temperature)
        """
        main_dir = os.path.join(self.output_folder, 'figures/well_time_plots')

        # Reset_directory
        if os.path.exists(main_dir):
            shutil.rmtree(main_dir)
        os.makedirs(main_dir)

        self.create_perf_dirs(main_dir)

        df = pd.read_pickle(os.path.join(self.output_folder, 'well_time_data.pkl'))
        time = df['time']
        pc = self.physics.property_containers[0]

        # Specify types of well rates that will be plotted if types_of_well_rates is not entered by the user
        if types_of_well_rates is None:
            types_of_well_rates = [
                "phases_molar_rates",
                "phases_mass_rates",
                "phases_volumetric_rates",
                "components_molar_rates",
                "components_mass_rates",
            ]
            if self.physics.thermal:
                types_of_well_rates.append("advective_heat_rate")

        for rtype in types_of_well_rates:
            for w in self.reservoir.wells:
                well_dir = os.path.join(main_dir, f'well_{w.name}')
                for perf in w.perforations:
                    subdir = os.path.join(well_dir, f'perf_{perf[0]}')
                    keys = self.create_perf_keys(rtype, w.name, perf[0], pc)
                    for key, ylabel in keys:
                        arr = df[key]
                        plt.figure()
                        plt.plot(time, arr, marker='o')
                        plt.xlabel('Time [day]')
                        plt.ylabel(ylabel)
                        plt.tight_layout()
                        plt.savefig(os.path.join(subdir, f'{key}.png'))
                        plt.close()
                # total and wellhead plots
                total_keys = self.create_total_keys(rtype, w.name, pc)
                for key, ylabel in total_keys:
                    plt.figure()
                    plt.plot(time, df[key], marker='o')
                    plt.xlabel('Time [day]')
                    plt.ylabel(ylabel)
                    plt.tight_layout()
                    plt.savefig(os.path.join(well_dir, f'{key}.png'))
                    plt.close()

        # BHP and BHT are plotted all the time
        for w in self.reservoir.wells:
            well_dir = os.path.join(main_dir, f'well_{w.name}')

            BHP_key = f'well_{w.name}_BHP'
            BHP = df[BHP_key]

            plt.figure()
            plt.plot(time, BHP, marker='o')
            plt.xlabel('Time [day]')
            plt.ylabel('Bottom-hole pressure [bar]')
            plt.tight_layout()
            plt.savefig(os.path.join(well_dir, f'{BHP_key}.png'))
            plt.close()

            BHT_key = f'well_{w.name}_BHT'
            BHT = df[BHT_key]

            plt.figure()
            plt.plot(time, BHT, marker='o')
            plt.xlabel('Time [day]')
            plt.ylabel('Bottom-hole pressure [bar]')
            plt.tight_layout()
            plt.savefig(os.path.join(well_dir, f'{BHT_key}.png'))
            plt.close()

        return df

    def configure_physics(self):
        pc = self.physics.property_containers[0]
        pc.physics_type = "super_engine"
        physics_name = type(self.physics).__name__
        if physics_name in ('Geothermal', 'GeothermalPH'):
            pc.physics_type = "geothermal_engine"
            pc.phases_name = self.physics.phases[:pc.nph]
            pc.nc_fl = 1
            pc.components_name = ['H2O']
            self.physics.thermal = True
        return pc

    def get_connection_info(self):
        perfs_conn_ids = [item for sublist in self.well_perf_conn_ids.values() for item in sublist]
        well_head_conn_ids = list(self.well_head_conn_id.values())

        # Get well indices for each perforation
        geometric_WI = np.array([p[2] for well in self.reservoir.wells for p in well.perforations])
        # Get transmissibility for each wellhead connection
        well_head_conn_trans = np.array([well.segment_transmissibility for well in self.reservoir.wells])

        # This change is done to make sure that if the user has not set multi_segment to True in the function
        # add_perforation, the perforations have different indices.
        for well in self.reservoir.wells:
            new_perforations = []
            for idx, perf in enumerate(well.perforations):
                new_perf = (idx, *perf[1:])
                new_perforations.append(new_perf)
            well.perforations = new_perforations

        return perfs_conn_ids, well_head_conn_ids, geometric_WI, well_head_conn_trans

    def store_perf_rates(self, time_data_dict, rates_perfs, rate_type, pc):
        perf_idx = 0
        for well in self.reservoir.wells:
            for perf in well.perforations:
                tag = f'well_{well.name}_perf_{perf[0]}'
                if rate_type.startswith('phases_'):
                    for phase_idx, phase_name in enumerate(pc.phases_name):
                        arr = rates_perfs[:, perf_idx, phase_idx]
                        time_data_dict[f'{tag}_{rate_type.split("_")[1]}_rate_{phase_name}'] = arr
                elif rate_type.startswith('components_'):
                    for c_idx in range(pc.nc_fl):
                        arr = np.sum(rates_perfs[:, perf_idx, c_idx::pc.nc_fl], axis=1)
                        time_data_dict[f'{tag}_{rate_type.split("_")[1]}_rate_{pc.components_name[c_idx]}'] = arr
                elif rate_type.startswith('advective_heat_'):
                    for phase_idx, phase_name in enumerate(pc.phases_name):
                        arr = rates_perfs[:, perf_idx, phase_idx]
                        time_data_dict[f'{tag}_advective_heat_rate_{phase_name}'] = arr
                perf_idx += 1

    def store_well_rates_sums(self, time_data_dict, rates_perfs, rate_type, pc):
        perf_idx = 0
        for well in self.reservoir.wells:
            tag = f'well_{well.name}'
            if rate_type.startswith('phases_'):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    total = sum(rates_perfs[:, perf_idx + j, phase_idx]
                                for j in range(len(well.perforations)))
                    time_data_dict[f'{tag}_{rate_type.split("_")[1]}_rate_{phase_name}_by_sum_perfs'] = total
                perf_idx += len(well.perforations)
            elif rate_type.startswith('components_'):
                for c_idx in range(pc.nc_fl):
                    total = sum(np.sum(rates_perfs[:, perf_idx + j, c_idx::pc.nc_fl], axis=1)
                                for j in range(len(well.perforations)))
                    time_data_dict[f'{tag}_{rate_type.split("_")[1]}_rate_{pc.components_name[c_idx]}_by_sum_perfs'] = total
                perf_idx += len(well.perforations)
            elif rate_type.startswith('advective_heat_'):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    total = sum(rates_perfs[:, perf_idx + j, phase_idx]
                                for j in range(len(well.perforations)))
                    time_data_dict[f'{tag}_advective_heat_rate_{phase_name}_by_sum_perfs'] = total
                perf_idx += len(well.perforations)

    def store_wellhead_rates(self, time_data_dict, wh_rates, rate_type, pc):
        for well_idx, well in enumerate(self.reservoir.wells):
            tag = f'well_{well.name}'
            if rate_type.startswith('phases_'):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    time_data_dict[f'{tag}_{rate_type.split("_")[1]}_rate_{phase_name}_at_wh'] = wh_rates[:, well_idx, phase_idx]
            elif rate_type.startswith('components_'):
                for c_idx, c_name in enumerate(pc.components_name):
                    arr = np.sum(wh_rates[:, well_idx, c_idx::pc.nc_fl], axis=1)
                    time_data_dict[f'{tag}_{rate_type.split("_")[1]}_rate_{c_name}_at_wh'] = arr
            elif rate_type.startswith('advective_heat_'):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    time_data_dict[f'{tag}_advective_heat_rate_{phase_name}_at_wh'] = wh_rates[:, well_idx, phase_idx]

    def store_bhp_bht(self, h5_well_data, time_data_dict, pc):
        dyn = h5_well_data['dynamic']
        nt = len(dyn['time'])
        for well in self.reservoir.wells:
            BHP = np.zeros(nt)
            BHT = np.zeros(nt) if self.physics.thermal else np.full(nt, pc.temperature)
            wellhead_cell_idx = find_one_array_in_another_indices([well.well_head_idx], dyn['cell_id'])
            p_idx = dyn['variable_names'].index('pressure')
            for i in range(nt):
                p = dyn['X'][i, :, p_idx]
                BHP[i] = p[wellhead_cell_idx]
                if self.physics.thermal:
                    if 'temperature' in dyn['variable_names']:
                        idx_T = dyn['variable_names'].index('temperature')
                        BHT[i] = dyn['X'][i, :, idx_T][wellhead_cell_idx]
                    else:
                        h_idx = dyn['variable_names'].index('enthalpy')
                        BHT[i] = pc.temperature_ev.evaluate([BHP[i], dyn['X'][i, wellhead_cell_idx, h_idx]])
            time_data_dict[f'well_{well.name}_BHP'] = BHP
            time_data_dict[f'well_{well.name}_BHT'] = BHT

    def create_perf_dirs(self, main_dir):
        for well in self.reservoir.wells:
            well_dir = os.path.join(main_dir, f'well_{well.name}')
            os.makedirs(well_dir, exist_ok=True)
            for perf in well.perforations:
                os.makedirs(os.path.join(well_dir, f'perf_{perf[0]}'), exist_ok=True)

    def create_perf_keys(self, rtype, well_name, perf_idx, pc):
        keys = []
        tag = f'well_{well_name}_perf_{perf_idx}_'
        if rtype.startswith('phases_'):
            unit = {'molar': 'kmol/day', 'mass': 'kg/day', 'volumetric': 'm^3/day'}[rtype.split('_')[1]]
            for phase_name in pc.phases_name:
                key = f'{tag}{rtype.split("_")[1]}_rate_{phase_name}'
                ylabel = f'{phase_name} {rtype.split("_")[1]} rate [{unit}]'
                keys.append((key, ylabel))
        elif rtype.startswith('components_'):
            unit = {'molar': 'kmol/day', 'mass': 'kg/day'}[rtype.split('_')[1]]
            for component_name in pc.components_name:
                key = f'{tag}{rtype.split("_")[1]}_rate_{component_name}'
                ylabel = f'{component_name} {rtype.split("_")[1]} rate [{unit}]'
                keys.append((key, ylabel))
        elif rtype.startswith('advective_heat_'):
            for phase_name in pc.phases_name:
                key = f'{tag}advective_heat_rate_{phase_name}'
                keys.append((key, f'{phase_name} advective heat rate [kJ/day]'))
        return keys

    def create_total_keys(self, rtype, well_name, pc):
        keys = []
        base = f'well_{well_name}_'
        if rtype.startswith('phases_'):
            unit = {'molar': 'kmol/day', 'mass': 'kg/day', 'volumetric': 'm^3/day'}[rtype.split('_')[1]]
            for phase_name in pc.phases_name:
                keys.extend([
                    (
                    f'{base}{rtype.split("_")[1]}_rate_{phase_name}_by_sum_perfs', f'{phase_name} {rtype.split("_")[1]} rate [{unit}]'),
                    (f'{base}{rtype.split("_")[1]}_rate_{phase_name}_at_wh', f'{phase_name} {rtype.split("_")[1]} rate [{unit}]')
                ])
        elif rtype.startswith('components_'):
            unit = {'molar': 'kmol/day', 'mass': 'kg/day'}[rtype.split('_')[1]]
            for component_name in pc.components_name:
                keys.extend([
                    (f'{base}{rtype.split("_")[1]}_rate_{component_name}_by_sum_perfs',
                     f'{component_name} {rtype.split("_")[1]} rate [{unit}]'),
                    (f'{base}{rtype.split("_")[1]}_rate_{component_name}_at_wh', f'{component_name} {rtype.split("_")[1]} rate [{unit}]')
                ])
        elif rtype.startswith('advective_heat_'):
            for phase_name in pc.phases_name:
                keys.extend([
                    (f'{base}advective_heat_rate_{phase_name}_by_sum_perfs', f'{phase_name} advective heat rate [kJ/day]'),
                    (f'{base}advective_heat_rate_{phase_name}_at_wh', f'{phase_name} advective heat rate [kJ/day]')
                ])
        elif rtype in ('BHP', 'BHT'):
            label = 'Bottom-hole pressure [bar]' if rtype == 'BHP' else 'Bottom-hole temperature [K]'
            keys.append((f'{base}{rtype}', label))
        return keys
