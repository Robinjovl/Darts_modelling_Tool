import os
import numpy as np
import h5py
import xarray as xr
import matplotlib.pyplot as plt
import shutil

from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.engines import value_vector, timer_node, ms_well_vector, op_vector
from darts.tools.plot_well_rates import *
from darts.physics.base.operators_base import PropertyOperators
# from darts.tools.plot_well_rates import *

#%%

class Output:
    """
    Base class for all output related functionality
    """
    def __init__(self, timer: timer_node, reservoir, physics, op_list, params,
                 output_folder : str, sol_filename : str, well_filename : str, save_initial : bool,
                 all_phase_props : bool, precision : str, compression : str, verbose : bool):
        """
        Class constructor method for output related functionalities including saving primary variables (state variables),
        evaulating secondary variables (properties) and creating visualizations.

        :param timer: timer object to measure time spent saviving data, and evaluating properties
        :reservoir: reservoir object
        :param physics: physics object
        :param op_list: list of operator interpolators
        :param params: engine params
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
        self.params = params
        self.verbose = verbose

        self.timer = timer.node['output']
        self.timer.node["output_reservoir"] = timer_node()
        self.timer.node["output_well"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()

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

        if save_initial:
            self.save_data_to_h5(kind='reservoir')

        if all_phase_props:
            from darts.physics.super.physics import Compositional
            from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH

            if type(self.physics) is Compositional:

                phase_props_labels = ['dens', 'dens_m', 'sat', 'mu', 'kr', 'pc', 'enthalpy', 'cond']
                # self.phase_props_units  = ['kmol/m3', 'kg/m3', '', 'cP', '', 'Bar', '', '']

                self.physics.property_itor = {}

                for region in self.physics.regions:  # loop over the different sets of operators
                    temp_dict = {}

                    # Loop through each property label and phase name
                    for i, name in enumerate(phase_props_labels):
                        for j, phase_name in enumerate(self.physics.phases):
                            temp_dict[f"{name}_{phase_name}"] = lambda i=i, j=j: self.physics.property_containers[region].phase_props[i][j]

                    for i, comp_name in enumerate(self.physics.property_containers[region].components_name):  # loop over components
                        for j, phase_name in enumerate(self.physics.phases):  # loop over phases
                            temp_dict[f"x_{phase_name}_{comp_name}"] = lambda i=i, j=j: self.physics.property_containers[region].x[j, i]

                    self.physics.property_operators[region] = PropertyOperators(self.physics.property_containers[region], self.physics.thermal, temp_dict)

                    self.physics.property_itor[region] = self.physics.create_interpolator(self.physics.property_operators[region],
                                                                                          n_ops=self.physics.n_ops,
                                                                                          platform='cpu', algorithm='multilinear',
                                                                                          mode='adaptive', precision='d',
                                                                                          timer_name='property %d interpolation' % region,
                                                                                          region=str(region))

                    # Assign the temporary dictionary to output_props for the region
                    self.physics.property_containers[region].output_props = temp_dict

                # Initialize physics and engine settings
                self.physics.init_physics()
                self.physics.engine.init(self.reservoir.mesh,
                                         ms_well_vector(self.reservoir.wells),
                                         op_vector(op_list),
                                         params,
                                         timer.node["simulation"])

            elif type(self.physics) is Geothermal or type(self.physics) is GeothermalPH:

                phase_props_labels = ['dens', 'dens_m', 'sat', 'mu', 'kr', 'pc', 'enthalpy', 'cond'] #, 'temperature']

                self.physics.property_itor = {}

                for region in self.physics.regions:  # loop over the different sets of operators
                    temp_dict = {}

                    # Loop through each property label and phase name
                    for i, name in enumerate(phase_props_labels):
                        # for j, phase_name in enumerate(self.physics.property_containers[region].nph):
                        for j in range(self.physics.property_containers[region].nph):
                            temp_dict[f"{name}_{self.physics.phases[j]}"] = lambda i=i, j=j: self.physics.property_containers[region].phase_props[i][j]

                    # add temperature
                    # temp_dict[phase_props_labels[-1]] = lambda: self.physics.property_containers[region].temperature

                    self.physics.property_operators[region] = PropertyOperators(self.physics.property_containers[region], thermal = True, props = temp_dict)

                    self.physics.property_itor[region] = self.physics.create_interpolator(self.physics.property_operators[region],
                                                                                          n_ops=self.physics.property_operators[region].n_ops,
                                                                                          platform='cpu', algorithm='multilinear',
                                                                                          mode='adaptive', precision='d',
                                                                                          timer_name='property %d interpolation' % region,
                                                                                          region=str(region))

                    # Assign the temporary dictionary to output_props for the region
                    self.physics.property_containers[region].output_props = temp_dict

                # Initialize physics and engine settings
                self.physics.init_physics()
                self.physics.engine.init(self.reservoir.mesh,
                                         ms_well_vector(self.reservoir.wells),
                                         op_vector(op_list),
                                         params,
                                         timer.node["simulation"])
            # self.reset()

        # Update the properties list
        self.properties = list(self.physics.property_containers[0].output_props.keys())

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
                        f"The following properties are not available: {missing_keys}. "
                        f"Choose properties from: {prop_keys}"
                    )

            # Create a new dictionary with only the available keys from new_prop_keys
            new_output_dictionary = {}
            for name in new_prop_keys:
                if name in output_dictionary: # Only add if the key is available
                    new_output_dictionary[name] = output_dictionary[name]

            # Update the output properties and reinitialize physics
            self.physics.property_containers[region].output_props = new_output_dictionary
            self.physics.init_physics()
            self.physics.engine.init(self.reservoir.mesh, ms_well_vector(self.reservoir.wells),
                                     op_vector(self.op_list), self.params, self.timerr.node["simulation"])
            self.properties = list(new_output_dictionary.keys())

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
            dynamic_group.create_dataset('time', shape=(0,), maxshape=(None,))

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
        X = np.array(self.physics.engine.X, copy=False)#[:self.physics.n_vars*self.reservoir.n]

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
            self.timer.start()
            self.timer.node['output_well'].start()
            self.save_specific_data(path)
            self.timer.node['output_well'].stop()
            self.timer.stop()

        elif kind == 'reservoir':
            path = os.path.join(self.output_folder, self.sol_filename)
            self.timer.start()
            self.timer.node['output_reservoir'].start()
            self.save_specific_data(path)
            self.timer.node['output_reservoir'].stop()
            self.timer.stop()

        else:
            print("Please use either kind='well' or kind='solution' in save_data_to_h5")

    def read_specific_data(self, filename: str, timestep: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts time and data from an HDF5 file for a given timestep.

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
                    cell_id = file['dynamic/cell_id'][:]
                    var_names = file['dynamic/variable_names'][:]
                    try:
                        time = file['dynamic/time'][timestep].reshape(1)
                        X = file['dynamic/X'][timestep].reshape(1, len(cell_id), len(var_names))
                    except IndexError:
                        raise IndexError(
                            f"Timestep {timestep} is out of range in the HDF5 file.")

            for i, name in enumerate(var_names):
                var_names[i] = name.decode()

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")

        return time, cell_id, X, var_names

    def output_properties(self, filepath: str = None, output_properties: list = None, timestep: int = None, engine = False) -> tuple[np.array, dict]:
        """
        Evaluates and returns properties from saved data (HDF5 file) or a simulation engine.

        :param filepath: Path to the solution HDF5 file. Defaults to None, which uses a previously defined output folder.
        :type filepath: str, optional
        :param output_properties: List of properties to evaluate. Defaults to None, which returns an array containing only state variables.
        :type output_properties: list, optional
        :param timestep: Timestep at which to evaluate properties. Defaults to None, which evaluates all saved timesteps.
        :type timestep: int, optional
        :param engine: Whether to evaluate properties directly from the simulation engine. Defaults to False, which reads properties from the HDF5 file.
        :type engine: bool, optional
    
        :return property_array: A dictionary where keys are primary/secondary variables and values are NumPy arrays of the requested properties for each grid block. The shape of each array is (number_of_timesteps, number_of_gridblocks).
        :type property_array: dict
        :return timesteps: A NumPy array of the time labels.
        :type timesteps: np.ndarray

        :raises ValueError: If specified property in `output_properties` is not found in any property container
        """

        if not engine:
            # Evaluate properties from the HDF5 file
            if filepath is None:
                path = os.path.join(self.output_folder, self.sol_filename)
            else:
                path = filepath

            timesteps, cell_id, X, var_names = self.read_specific_data(path, timestep)

        else:
            # Evaluate properties from the engine
            timesteps = np.array(self.physics.engine.t).reshape(1,)
            cell_id = np.arange(self.reservoir.mesh.n_res_blocks)
            X = np.array(self.physics.engine.X[:self.physics.n_vars*self.reservoir.mesh.n_res_blocks], copy = True)
            var_names = self.physics.vars

        # Initialize property_array
        n_vars = len(var_names)

        nb = len(cell_id)

        output_properties = output_properties if output_properties is not None else list(self.physics.vars)

        # primary properties i.e. state variables
        primary_props = [prop for prop in output_properties if prop in var_names]
        primary_prop_idxs = {prop: list(var_names).index(prop) for prop in primary_props}

        # secondary properties defined
        secondary_props = [prop for prop in output_properties if prop not in var_names]
        # secondary_prop_idxs = {prop: list(self.physics.property_containers[next(iter(self.physics.property_containers))].output_props.keys()).index(prop) for prop in secondary_props}
        secondary_prop_idxs = {}
        for prop in secondary_props:
            for container in self.physics.property_containers.values():
                if prop in container.output_props:
                    secondary_prop_idxs[prop] = list(container.output_props.keys()).index(prop)
                    break
            else:
                raise ValueError(f"Secondary property '{prop}' not found in any property container.")

        # define property array
        property_array = {prop: np.zeros((len(timesteps), nb)) for prop in primary_props + secondary_props}

        # Loop over timesteps
        for k, timestep in enumerate(timesteps):

            # Extract primary properties from X vector
            for var_name, var_idx in primary_prop_idxs.items():
                if engine is False:
                    property_array[var_name][k] = X[k, :nb, var_idx]
                else:
                    property_array[var_name][k] = X[var_idx::n_vars]

            # Interpolate secondary properties
            if secondary_props:
                if engine is False:
                    state = value_vector(np.stack([X[k, :nb, j] for j in range(n_vars)]).T.flatten())
                else:
                    state = value_vector(np.stack([X[j::n_vars] for j in range(n_vars)]).T.flatten())

                i = 0
                n_ops = self.physics.property_operators[i].n_ops
                values = value_vector(np.zeros(n_ops * nb))
                values_numpy = np.array(values, copy=False)
                dvalues = value_vector(np.zeros(n_ops * nb * n_vars))

                for region, prop_itor in self.physics.property_itor.items():
                    prop_itor.evaluate_with_derivatives(state, self.physics.engine.region_cell_idx[i], values, dvalues)
                    # prop_itor.evaluate(state, values)
                    i += 1

                for prop_name, prop_idx in secondary_prop_idxs.items():
                    property_array[prop_name][k] = values_numpy[prop_idx::n_ops]

        return timesteps, property_array

    def output_to_xarray(self, filepath: str = None, output_properties: list = None, timestep: int = None, engine: bool = False) -> xr.Dataset:
        """
        Generates an xarray Dataset of properties and saves it as a NetCDF file.
        State variables area obtained from the engine or *.h5 file.
        Properties are interpolated by the property iterator.

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
        dx, dy, dz = self.reservoir.global_data['dx'], self.reservoir.global_data['dy'], self.reservoir.global_data['dz']
        x = np.cumsum(dx[:, 0, 0]) - dx[0, 0, 0] * 0.5
        y = np.cumsum(dy[0, :, 0]) - dy[0, 0, 0] * 0.5
        z = np.cumsum(dz[0, 0, :]) - dz[0, 0, 0] * 0.5
        coords = {'time': time, 'z': z, 'y': y, 'x': x}
        data_vars = {prop: (list(coords.keys()), data[prop]) for prop in props}
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)
        if self.precision == 'd':
            encoding = {prop: {'dtype': 'float64'} for prop in data.keys()}
        else:
            encoding = {prop: {'dtype': 'float32'} for prop in data.keys()}

        # Save to NetCDF with specified encoding
        dataset.to_netcdf(os.path.join(self.output_folder, self.sol_filename[:-3] + '.nc'), engine='netcdf4', encoding=encoding)

        return dataset

    def plot_xarray(self, xarray_data, timestep: int = -1, x: int = None, y: int =None, z: int = None):
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

        assert timestep < len(xarray_data['time']), 'time step should be less than %d' % len(xarray_data['time'])

        var_names = list(xarray_data.data_vars)
        nrows = len(var_names)
        for i, var in enumerate(var_names):
            plt.figure()
            if z is not None:
                assert z < len(xarray_data['z']), 'z-level step should be less than %d' % len(xarray_data['z'])
                xarray_data[var].isel(time=timestep, z=z).plot()
                plt.savefig(output_directory + '/%s_ts%d_z%d.png'%(var, timestep, z))

            elif y is not None:
                assert y < len(xarray_data['y']), 'y-level step should be less than %d' % len(xarray_data['y'])
                xarray_data[var].isel(time=timestep, y=y).plot()
                plt.savefig(output_directory + '/%s_ts%d_y%d.png' % (var, timestep, y))

            elif x is not None:
                assert x < len(xarray_data['x']), 'x-level step should be less than %d' % len(xarray_data['x'])
                xarray_data[var].isel(time=timestep, x=x).plot()
                plt.savefig(output_directory + '/%s_ts%d_zx%d.png'%(var, timestep, z))

            else:
                # model is a 1D reservoir
                xarray_data[var].isel(time=timestep).plot()
                plt.savefig(output_directory + '/%s_ts%d.png' % (var, timestep))
            # plt.close()

    def output_to_vtk(self, filepath: str = None, ith_step: int = None, output_directory: str = None, output_properties: list = None, engine : bool = False):
        """
        Function to export results at timestamp t into `.vtk` format.

        :param ith_step: i'th reporting step
        :type ith_step: int
        :param output_directory: Name to save .vtk file
        :type output_directory: str
        :param output_properties: List of properties to include in .vtk file, default is None which will pass all
        :type output_properties: list
        """
        self.timer.start(); self.timer.node["vtk_output"].start()

        # Set default output directory
        if output_directory is None:
            output_directory = self.output_folder

        main_dir = os.path.join(output_directory, 'vtk_files')
        if not os.path.exists(main_dir):
            os.makedirs(main_dir, exist_ok = True)

        # timesteps, property_array = self.output_properties(output_properties=list(prop_idxs.keys()), timestep=ith_step)
        timesteps, property_array = self.output_properties(filepath, output_properties, ith_step, engine)
        prop_names = {prop: i for i, prop in enumerate(property_array.keys())}

        for t, time in enumerate(timesteps):
            data = np.zeros((len(property_array), self.reservoir.mesh.n_res_blocks))
            for i, name in enumerate(property_array.keys()):
                data[i, :] = property_array[name][t]

            # Pass to Reservoir.output_to_vtk() method
            if ith_step is None:
                self.reservoir.output_to_vtk(t, time, main_dir, prop_names, data)
            else:
                self.reservoir.output_to_vtk(ith_step, time, main_dir, prop_names, data)

        self.timer.node["vtk_output"].stop(); self.timer.stop()

    def plot_well_rates(self, types_of_well_rates: list, save_figs : bool = True) -> dict:
        """
        Plots the following types of rates for each perforation and each well:
        'phases_molar_rates'
        'phases_mass_rates'
        'phases_volumetric_rates'
        'components_molar_rates'
        'components_mass_rates'
        'heat_rate'

        :param types_of_well_rates: Type or types of well rates the user wants to plot
        :type types_of_well_rates: list
        :param m: An instance of DartsModel
        :type m: DartsModel
        """

        # h5_well_data_address = os.path.join(self.output_folder, self.well_filename)
        h5_well_data = load_hdf5_to_dict(self.well_filepath)

        # Calculate well rates
        perfs = [p for well in self.reservoir.wells for p in well.perforations]
        block_m = np.array(self.reservoir.mesh.block_m, copy=False)
        block_p = np.array(self.reservoir.mesh.block_p, copy=False)
        perfs_conn_ids = find_conn_ids_for_perfs(perfs, block_m, block_p, self.reservoir.mesh.n_res_blocks)
        
        # Get well indices for each perforation
        geometric_WI = np.array([p[2] for well in self.reservoir.wells for p in well.perforations])
        
        # Get property container for operators calculations
        property_container = self.physics.property_containers
        pc = property_container[0]

        # Create new folders in which well rates output will be stored
        main_dir = os.path.join(self.output_folder, 'figures/output_well_rates')
        if not os.path.exists(main_dir):
            os.makedirs(main_dir, exist_ok = True)
        elif os.path.exists(main_dir):
            shutil.rmtree(main_dir)
            os.makedirs(main_dir, exist_ok = True)

        for well in self.reservoir.wells:
            well_dir = os.path.join(main_dir, 'well_' + well.name)
            os.makedirs(well_dir, exist_ok = True)
            for perf in well.perforations:
                perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                os.makedirs(perf_dir, exist_ok = True)

        well_rates_dict = {}
        for rate_type in types_of_well_rates:
            # if rate_type == 'heat_rate' and not self.physics.thermal:
            if rate_type == 'heat_rate':
                continue

            time, rates = calc_rates_at_perforations(h5_well_data, perfs_conn_ids, geometric_WI, rate_type, self.physics.thermal, pc)
            well_rates_dict['time'] = time

            """""""""  Plot well rates over time """""""""
            """ Rates for each perforation """
            perf_counter = 0
            for well in self.reservoir.wells:
                for perf in well.perforations:
                    if rate_type in ['phases_molar_rates', 'phases_mass_rates', 'phases_volumetric_rates']:
                        for phase in range(pc.nph):
                            phase_rate_for_perf = - rates[:, perf_counter, phase]

                            plt.figure()
                            plt.plot(time, phase_rate_for_perf, color='r', marker='o', markersize=5)
                            plt.xlabel('time [day]', fontsize=16)

                            if rate_type == 'phases_molar_rates':
                                plt.ylabel(f'{pc.phases_name[phase]} molar rate [kmol/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                os.makedirs(perf_dir, exist_ok=True) # Ensure directory exists before saving
                                if save_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.phases_name[phase]}.png'))
                                well_rates_dict[f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.phases_name[phase]}'] = phase_rate_for_perf

                                well_rates_dict['well_' + well.name + '_perf_' + str(perf[0]) + '_molar_rate_' +
                                            pc.phases_name[phase]] = phase_rate_for_perf

                            elif rate_type == 'phases_mass_rates':
                                plt.ylabel(f'{pc.phases_name[phase]} mass rate [kg/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                os.makedirs(perf_dir, exist_ok=True)
                                if save_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.phases_name[phase]}.png'))
                                well_rates_dict[f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.phases_name[phase]}'] = phase_rate_for_perf

                                well_rates_dict['\well_' + well.name + '_perf_' + str(perf[0]) + '_mass_rate_' +
                                            pc.phases_name[phase]] = phase_rate_for_perf

                            elif rate_type == 'phases_volumetric_rates':
                                plt.ylabel(f'{pc.phases_name[phase]} volumetric rate [m3/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                os.makedirs(perf_dir, exist_ok=True)
                                plt.tight_layout()
                                if save_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_volumetric_rate_{pc.phases_name[phase]}.png'))
                                well_rates_dict[f'well_{well.name}_perf_{str(perf[0])}_volumetric_rate_{pc.phases_name[phase]}'] = phase_rate_for_perf
                                well_rates_dict['well_' + well.name + '_perf_' + str(perf[0]) + '_volumetric_rate_' +
                                    pc.phases_name[phase]] = phase_rate_for_perf

                    elif rate_type in ['components_molar_rates', 'components_mass_rates']:
                        for component in range(pc.nc_fl):
                            component_rate_for_perf = - np.sum(rates[:, perf_counter, component::pc.nc_fl], axis=1)

                            plt.figure()
                            plt.plot(time, component_rate_for_perf, color='r', marker='o', markersize=5)

                            plt.xlabel('time [day]', fontsize=16)

                            if rate_type == 'components_molar_rates':
                                plt.ylabel(f'{pc.components_name[component]} molar rate [kmole/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                plt.tight_layout()
                                os.makedirs(perf_dir, exist_ok=True)
                                if save_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.components_name[component]}.png'))
                                well_rates_dict[f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.components_name[component]}'] = component_rate_for_perf

                                well_rates_dict['well_' + well.name + '_perf_' + str(perf[0]) + '_molar_rate_' +
                                            pc.components_name[component]] = component_rate_for_perf
                            elif rate_type == 'components_mass_rates':
                                plt.ylabel(f'{pc.components_name[component]} mass rate [kg/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                plt.tight_layout()
                                os.makedirs(perf_dir, exist_ok=True)
                                if save_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.components_name[component]}.png'))
                                well_rates_dict[f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.components_name[component]}'] = component_rate_for_perf
                                well_rates_dict['well_' + well.name + '_perf_' + str(perf[0]) + '_mass_rate_' +
                                            pc.components_name[component]] = component_rate_for_perf

                    elif rate_type == 'heat_rate':
                        heat_rate_of_all_phases_for_perf = - np.sum(rates[:, perf_counter, :], axis=1)

                        plt.figure()
                        plt.plot(time, heat_rate_of_all_phases_for_perf, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel('heat rate [kJ/day]', fontsize=16)
                        perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                        plt.tight_layout()
                        os.makedirs(perf_dir, exist_ok=True)
                        if save_figs:
                            plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_heat_rate.png'))
                        well_rates_dict[f'well_{well.name}_perf_{str(perf[0])}_heat_rate'] = heat_rate_of_all_phases_for_perf
                        well_rates_dict['well_' + well.name + '_perf_' + str(perf[0]) + '_heat_rate'] = heat_rate_of_all_phases_for_perf

                    perf_counter += 1

            """ Total rates for each well """
            if rate_type == 'phases_molar_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for phase in range(pc.nph):
                        phase_molar_rate_for_well = 0
                        for perf in well.perforations:
                            phase_molar_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_molar_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' molar rate [kmol/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if save_figs:
                            plt.savefig(os.path.join(well_dir, f'well_{well.name}_molar_rate_{pc.phases_name[phase]}.png'))
                        well_rates_dict[f'well_{well.name}_molar_rate_{pc.phases_name[phase]}'] = phase_molar_rate_for_well
                        well_rates_dict['well_' + well.name + '_molar_rate_' + pc.phases_name[phase]] = phase_molar_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'phases_mass_rates':
                perf_counter = 0
                for well in self.reservoir.wells:

                    for phase in range(pc.nph):
                        phase_mass_rate_for_well = 0
                        for perf in well.perforations:
                            phase_mass_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_mass_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' mass rate [kg/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if save_figs:
                            plt.savefig(os.path.join(well_dir, f'well_{well.name}_mass_rate_{pc.phases_name[phase]}.png'))
                        well_rates_dict[f'well_{well.name}_mass_rate_{pc.phases_name[phase]}'] = phase_mass_rate_for_well
                    perf_counter += len(well.perforations)

            elif rate_type == 'phases_volumetric_rates':
                perf_counter = 0
                for well in self.reservoir.wells:

                    for phase in range(pc.nph):
                        phase_volumetric_rate_for_well = 0
                        for perf in well.perforations:
                            phase_volumetric_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_volumetric_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' volumetric rate [m$^3$/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if save_figs:
                            plt.savefig(os.path.join(well_dir, f'well_{well.name}_volumetric_rate_{pc.phases_name[phase]}.png'))
                        well_rates_dict[f'well_{well.name}_volumetric_rate_{pc.phases_name[phase]}'] = phase_volumetric_rate_for_well
                        well_rates_dict['well_' + well.name + '_volumetric_rate_' + pc.phases_name[phase]] = phase_volumetric_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'components_molar_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for component in range(pc.nc_fl):
                        component_molar_rate_for_well = 0
                        for perf in well.perforations:
                            component_molar_rate_for_well += - np.sum(
                                rates[:, perf_counter, component::pc.nc_fl], axis=1)

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, component_molar_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.components_name[component] + ' molar rate [kmole/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if save_figs:
                            plt.savefig(os.path.join(well_dir,f'well_{well.name}_molar_rate_{pc.components_name[component]}.png'))
                        well_rates_dict[f'well_{well.name}_molar_rate_{pc.components_name[component]}'] = component_molar_rate_for_well
                        well_rates_dict['well_' + well.name + '_molar_rate_' + pc.components_name[component]] = component_molar_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'components_mass_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for component in range(pc.nc_fl):
                        component_mass_rate_for_well = 0
                        for perf in well.perforations:
                            component_mass_rate_for_well += - np.sum(rates[:, perf_counter, component::pc.nc_fl], axis=1)

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, component_mass_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.components_name[component] + ' mass rate [kg/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if save_figs:
                            plt.savefig(os.path.join(well_dir,f'well_{well.name}_mass_rate_{pc.components_name[component]}.png'))
                        well_rates_dict[f'well_{well.name}_mass_rate_{pc.components_name[component]}'] = component_mass_rate_for_well
                        well_rates_dict['well_' + well.name + '_mass_rate_' + pc.components_name[component]] = component_mass_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'heat_rate':
                perf_counter = 0
                for well in self.reservoir.wells:

                    heat_rate_for_well = 0
                    for phase in range(pc.nph):
                        for perf in well.perforations:
                            heat_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                    plt.figure()
                    plt.plot(time, heat_rate_for_well, color='r', marker='o', markersize=5)
                    plt.xlabel('time [day]', fontsize=16)
                    plt.ylabel('heat rate [kJ/day]', fontsize=16)
                    well_dir = os.path.join(main_dir, 'well_' + well.name)
                    plt.tight_layout()
                    os.makedirs(well_dir, exist_ok=True)
                    if save_figs:
                        plt.savefig(os.path.join(well_dir, f'well_{well.name}_heat_rate.png'))
                    well_rates_dict[f'well_{well.name}_heat_rate'] = heat_rate_for_well

                    perf_counter += len(well.perforations)

        plt.close()

        return well_rates_dict

    def store_well_rates_bhp_and_bht_over_time_as_dict(self, plot_figs : bool = False):
        """
        Stores well rates (phases_molar_rates, phases_mass_rates, phases_volumetric_rates, components_molar_rates,
        components_mass_rates, and heat_rate), bottom-hole pressure (BHP), and bottom-hole temperature (BHT) over time.
        """
        h5_well_data = load_hdf5_to_dict(self.well_filepath)

        # Calculate well rates
        perfs = [p for well in self.reservoir.wells for p in well.perforations]
        block_m = np.array(self.reservoir.mesh.block_m, copy=False)
        block_p = np.array(self.reservoir.mesh.block_p, copy=False)
        perfs_conn_ids = find_conn_ids_for_perfs(perfs, block_m, block_p, self.reservoir.mesh.n_res_blocks)

        # Get well indices for each perforation
        geometric_WI = np.array([p[2] for well in self.reservoir.wells for p in well.perforations])

        # Get property container for operators calculations
        property_container = self.physics.property_containers
        pc = property_container[0]

        # Path of the folder in which figures will be saved
        main_dir = os.path.join(self.output_folder, 'figures/output_well_rates')
        if plot_figs:
            # Create new folders in which well figures will be stored
            if not os.path.exists(main_dir):
                os.makedirs(main_dir, exist_ok=True)
            elif os.path.exists(main_dir):
                shutil.rmtree(main_dir)
                os.makedirs(main_dir, exist_ok=True)

            for well in self.reservoir.wells:
                well_dir = os.path.join(main_dir, 'well_' + well.name)
                os.makedirs(well_dir, exist_ok=True)
                for perf in well.perforations:
                    perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                    os.makedirs(perf_dir, exist_ok=True)
        elif plot_figs is False:
            # If plot_figs is False and the directory in which well figures were saved in the past exists,
            # it removes that folder to avoid confusion.
            if os.path.exists(main_dir):
                shutil.rmtree(main_dir)

        well_output_dict = {}
        types_of_well_rates = ["phases_molar_rates", "phases_mass_rates", "phases_volumetric_rates",
                               "components_molar_rates", "components_mass_rates", "heat_rate"]
        for rate_type in types_of_well_rates:
            # if rate_type == 'heat_rate' and not self.physics.thermal:
            if rate_type == 'heat_rate':
                continue

            time, rates = calc_rates_at_perforations(h5_well_data, perfs_conn_ids, geometric_WI, rate_type,
                                                     self.physics.thermal, pc)
            well_output_dict['time'] = time

            """""""""  Plot well rates over time """""""""
            """ Rates for each perforation """
            perf_counter = 0
            for well in self.reservoir.wells:
                for perf in well.perforations:
                    if rate_type in ['phases_molar_rates', 'phases_mass_rates', 'phases_volumetric_rates']:
                        for phase in range(pc.nph):
                            phase_rate_for_perf = - rates[:, perf_counter, phase]

                            plt.figure()
                            plt.plot(time, phase_rate_for_perf, color='r', marker='o', markersize=5)
                            plt.xlabel('time [day]', fontsize=16)

                            if rate_type == 'phases_molar_rates':
                                plt.ylabel(f'{pc.phases_name[phase]} molar rate [kmol/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                os.makedirs(perf_dir, exist_ok=True)  # Ensure directory exists before saving
                                if plot_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.phases_name[phase]}.png'))
                                well_output_dict[f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.phases_name[phase]}'] = phase_rate_for_perf

                            elif rate_type == 'phases_mass_rates':
                                plt.ylabel(f'{pc.phases_name[phase]} mass rate [kg/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                os.makedirs(perf_dir, exist_ok=True)
                                if plot_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.phases_name[phase]}.png'))
                                well_output_dict[f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.phases_name[phase]}'] = phase_rate_for_perf

                            elif rate_type == 'phases_volumetric_rates':
                                plt.ylabel(f'{pc.phases_name[phase]} volumetric rate [m3/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                os.makedirs(perf_dir, exist_ok=True)
                                plt.tight_layout()
                                if plot_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_volumetric_rate_{pc.phases_name[phase]}.png'))
                                well_output_dict[f'well_{well.name}_perf_{str(perf[0])}_volumetric_rate_{pc.phases_name[phase]}'] = phase_rate_for_perf

                    elif rate_type in ['components_molar_rates', 'components_mass_rates']:
                        for component in range(pc.nc_fl):
                            component_rate_for_perf = - np.sum(rates[:, perf_counter, component::pc.nc_fl], axis=1)

                            plt.figure()
                            plt.plot(time, component_rate_for_perf, color='r', marker='o', markersize=5)

                            plt.xlabel('time [day]', fontsize=16)

                            if rate_type == 'components_molar_rates':
                                plt.ylabel(f'{pc.components_name[component]} molar rate [kmole/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                plt.tight_layout()
                                os.makedirs(perf_dir, exist_ok=True)
                                if plot_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.components_name[component]}.png'))
                                well_output_dict[f'well_{well.name}_perf_{str(perf[0])}_molar_rate_{pc.components_name[component]}'] = component_rate_for_perf

                            elif rate_type == 'components_mass_rates':
                                plt.ylabel(f'{pc.components_name[component]} mass rate [kg/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, f'perf_{str(perf[0])}')
                                plt.tight_layout()
                                os.makedirs(perf_dir, exist_ok=True)
                                if plot_figs:
                                    plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.components_name[component]}.png'))
                                well_output_dict[f'well_{well.name}_perf_{str(perf[0])}_mass_rate_{pc.components_name[component]}'] = component_rate_for_perf

                    elif rate_type == 'heat_rate':
                        heat_rate_of_all_phases_for_perf = - np.sum(rates[:, perf_counter, :], axis=1)

                        plt.figure()
                        plt.plot(time, heat_rate_of_all_phases_for_perf, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel('heat rate [kJ/day]', fontsize=16)
                        perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                        plt.tight_layout()
                        os.makedirs(perf_dir, exist_ok=True)
                        if plot_figs:
                            plt.savefig(os.path.join(perf_dir, f'well_{well.name}_perf_{str(perf[0])}_heat_rate.png'))
                        well_output_dict[f'well_{well.name}_perf_{str(perf[0])}_heat_rate'] = heat_rate_of_all_phases_for_perf

                    perf_counter += 1

            """ Total rates for each well """
            if rate_type == 'phases_molar_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for phase in range(pc.nph):
                        phase_molar_rate_for_well = 0
                        for perf in well.perforations:
                            phase_molar_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_molar_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' molar rate [kmol/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if plot_figs:
                            plt.savefig(os.path.join(well_dir, f'well_{well.name}_molar_rate_{pc.phases_name[phase]}.png'))
                        well_output_dict[f'well_{well.name}_molar_rate_{pc.phases_name[phase]}'] = phase_molar_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'phases_mass_rates':
                perf_counter = 0
                for well in self.reservoir.wells:

                    for phase in range(pc.nph):
                        phase_mass_rate_for_well = 0
                        for perf in well.perforations:
                            phase_mass_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_mass_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' mass rate [kg/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if plot_figs:
                            plt.savefig(
                                os.path.join(well_dir, f'well_{well.name}_mass_rate_{pc.phases_name[phase]}.png'))
                        well_output_dict[f'well_{well.name}_mass_rate_{pc.phases_name[phase]}'] = phase_mass_rate_for_well
                    perf_counter += len(well.perforations)

            elif rate_type == 'phases_volumetric_rates':
                perf_counter = 0
                for well in self.reservoir.wells:

                    for phase in range(pc.nph):
                        phase_volumetric_rate_for_well = 0
                        for perf in well.perforations:
                            phase_volumetric_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_volumetric_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' volumetric rate [m$^3$/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if plot_figs:
                            plt.savefig(
                                os.path.join(well_dir, f'well_{well.name}_volumetric_rate_{pc.phases_name[phase]}.png'))
                        well_output_dict[f'well_{well.name}_volumetric_rate_{pc.phases_name[phase]}'] = phase_volumetric_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'components_molar_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for component in range(pc.nc_fl):
                        component_molar_rate_for_well = 0
                        for perf in well.perforations:
                            component_molar_rate_for_well += - np.sum(
                                rates[:, perf_counter, component::pc.nc_fl], axis=1)

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, component_molar_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.components_name[component] + ' molar rate [kmole/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if plot_figs:
                            plt.savefig(os.path.join(well_dir, f'well_{well.name}_molar_rate_{pc.components_name[component]}.png'))
                        well_output_dict[f'well_{well.name}_molar_rate_{pc.components_name[component]}'] = component_molar_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'components_mass_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for component in range(pc.nc_fl):
                        component_mass_rate_for_well = 0
                        for perf in well.perforations:
                            component_mass_rate_for_well += - np.sum(rates[:, perf_counter, component::pc.nc_fl],
                                                                     axis=1)

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, component_mass_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.components_name[component] + ' mass rate [kg/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.tight_layout()
                        os.makedirs(well_dir, exist_ok=True)
                        if plot_figs:
                            plt.savefig(os.path.join(well_dir, f'well_{well.name}_mass_rate_{pc.components_name[component]}.png'))
                        well_output_dict[f'well_{well.name}_mass_rate_{pc.components_name[component]}'] = component_mass_rate_for_well

                    perf_counter += len(well.perforations)

            elif rate_type == 'heat_rate':
                perf_counter = 0
                for well in self.reservoir.wells:

                    heat_rate_for_well = 0
                    for phase in range(pc.nph):
                        for perf in well.perforations:
                            heat_rate_for_well += - rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                    plt.figure()
                    plt.plot(time, heat_rate_for_well, color='r', marker='o', markersize=5)
                    plt.xlabel('time [day]', fontsize=16)
                    plt.ylabel('heat rate [kJ/day]', fontsize=16)
                    well_dir = os.path.join(main_dir, 'well_' + well.name)
                    plt.tight_layout()
                    os.makedirs(well_dir, exist_ok=True)
                    if plot_figs:
                        plt.savefig(os.path.join(well_dir, f'well_{well.name}_heat_rate.png'))
                    well_output_dict[f'well_{well.name}_heat_rate'] = heat_rate_for_well

                    perf_counter += len(well.perforations)

        plt.close()

        return well_output_dict
