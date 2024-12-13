import os
import numpy as np
import h5py
import xarray as xr
import matplotlib.pyplot as plt
import shutil

from darts.tools.hdf5_tools import load_hdf5_to_dict
from darts.engines import value_vector, timer_node, ms_well_vector, op_vector
from darts.tools.plot_well_rates import *
# from darts.tools.plot_well_rates import *

#%%

class Output:
    """
    Base class for all output related functionality
    """
    def __init__(self, timer: timer_node, reservoir, physics, op_list, params, output_folder, sol_filename, restart, all_phase_props, precision):
        super().__init__()

        self.reservoir = reservoir
        self.physics = physics
        self.op_list = op_list
        self.params = params

        self.timerr = timer
        self.timer = timer.node['output']
        self.timer.node["output_reservoir"] = timer_node()
        self.timer.node["output_well"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()

        self.sol_filename = sol_filename
        self.well_filename = 'well_data.h5'
        self.output_folder = output_folder
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)
        self.precision = precision

        self.restart = restart
        if self.restart is False:
            self.save_data_to_h5(kind='reservoir')

        if all_phase_props:
            phase_props_labels = ['dens', 'dens_m', 'sat', 'nu', 'mu', 'kr', 'pc', 'enthalpy', 'cond']

            for region in self.physics.regions:  # loop over the different sets of operators
                temp_dict = {}

                # Loop through each property label and phase name
                for i, name in enumerate(phase_props_labels):
                    for j, phase_name in enumerate(self.physics.phases):
                        temp_dict[f"{name} {phase_name}"] = lambda i=i, j=j: self.physics.property_containers[region].phase_props[i][j]

                for i, comp_name in enumerate(self.physics.property_containers[region].components_name):  # loop over components
                    for j, phase_name in enumerate(self.physics.phases):  # loop over phases
                        temp_dict[f"x_{phase_name}_{comp_name}"] = lambda i=i, j=j: self.physics.property_containers[region].x[j,i]

                # Assign the temporary dictionary to output_props for the region
                self.physics.property_containers[region].output_props = temp_dict

            # Initialize physics and engine settings
            self.physics.init_physics()
            self.physics.engine.init(self.reservoir.mesh,
                                     ms_well_vector(self.reservoir.wells),
                                     op_vector(op_list),
                                     params,
                                     timer.node["simulation"])

            # Update the properties list
            self.properties = list(self.physics.property_containers[0].output_props.keys())

        else:
            # If all_phase_props is False, update properties list based on output_props
            self.properties = list(self.physics.property_containers[0].output_props.keys())

    def filter_phase_props(self, new_prop_keys = ['sat1', 'dens0']):
        for region in self.physics.regions:
            output_dictionary = self.physics.property_containers[region].output_props
            prop_keys = list(output_dictionary.keys())
            print(f'Available properties in region {region} are {prop_keys}')

            # Warn if any key is missing in the available properties
            for key in new_prop_keys:
                if key not in prop_keys:
                    print(f"Warning: '{key}' is not an available property, choose properties out of {prop_keys}")
                    # break

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

        return 0

    def load_restart_data(self, filename: str = os.path.join('restart', 'reservoir_solution.h5'), timestep: int = -1):
        """
        Loads data from a previous simulation and sets it for the current simulation.
        :param filename (str): Path to the restart file (default: 'restart/reservoir_solution.h5').
        :param timestep (int): The timestep to load from the file (default: -1 for the last timestep).
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"The restart file does not exist: {filename}")

        # Read data from the file
        time, cell_id, X, var_names = self.read_specific_data(filename, timestep)

        print(f"Restarting from {filename} at time = {time[0]:.6f} days")

        # Update the simulation engine with the loaded data
        self.physics.engine.t = time[0]
        self.physics.engine.X = value_vector(X.flatten())
        self.physics.engine.Xn = value_vector(X.flatten())

        # Save the data
        self.save_data_to_h5(kind='reservoir')

    def configure_h5_output(self, filename: str, cell_ids, description, add_static_data: bool = False):
        """
        Configuration of *.h5 output

        :param filename: *.h5 filename
        :param cell_ids: np.array of cell indexes for output
        :param description: description for *.h5
        :param add_static_data: flag to add static output
        """

        precision_map = {
            'd': np.float64, # Double precision
            's': np.float32  # Single precision
        }

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
                                         maxshape=(None, nb, self.physics.n_vars), dtype=precision_map[self.precision])

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

        # Ensure the directory exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            os.makedirs(os.path.join(self.output_folder, 'figures'))

        # solution ouput
        if kind == 'reservoir':
            sol_output_path = os.path.join(self.output_folder, self.sol_filename)
            if os.path.exists(sol_output_path): #and not restart:
                os.remove(sol_output_path)
            self.configure_h5_output(filename=sol_output_path, cell_ids=np.arange(self.reservoir.mesh.n_blocks),
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
        X = np.array(self.physics.engine.X, copy=False)

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
            return

    def read_specific_data(self, filename: str, timestep: int = None):
        """
        Function to read *.h5 files contents.
        :param filename: path to *.h5 filename to append data to
        :param timestep:
        :return time: time of the saved data in days
        :rtype: np.ndarray
        :return cell_id: cell id of each of the saved grid blocks
        :rtype: np.ndarray
        :return X: variables names
        :rtype: np.ndarray
        """

        # Open the HDF5 file
        with h5py.File(filename, 'r') as file:
            if timestep is None:
                datapoints = file['dynamic/X'].shape[0] * file['dynamic/X'].shape[1] * file['dynamic/X'].shape[2]
                print('WARNING: %s contains %d data points...' % (filename, datapoints))

                cell_id = file['dynamic/cell_id'][:]
                var_names = file['dynamic/variable_names'][:]
                time = file['dynamic/time'][:]
                X = file['dynamic/X'][:]
            else:
                cell_id = file['dynamic/cell_id'][:]
                var_names = file['dynamic/variable_names'][:]
                time = file['dynamic/time'][timestep].reshape(1)
                X = file['dynamic/X'][timestep].reshape(1, len(cell_id), len(var_names))


        for i, name in enumerate(var_names):
            var_names[i] = name.decode()

        return time, cell_id, X, var_names

    def output_properties(self, filepath: str = None, output_properties: list = None, timestep: int = None, engine = False) -> tuple:
        """
        Function to read *.h5 data and evaluate properties per grid block, per timestep
        :param output_properties: list of properties to evaluate for output, default = None (no properties)
        :param filepath: solution filepath, e.g. /solution.h5
        :param timestep: timestep at which you want to evaluate properties, default = None (all the timesteps are evaluated)
        :return property_array : dict of property arrays per timestep, per gridblock
        :return timesteps: ndarray the time labels of the evaluated timesteps
        """
        # Read binary file
        if filepath is None:
            path = os.path.join(self.output_folder, self.sol_filename)
        else:
            path = filepath

        if not os.path.exists(path) and not engine:
            raise FileNotFoundError(f"The specified file does not exist: {path}")

        if engine is False:
            # get data from file
            timesteps, cell_id, X, var_names = self.read_specific_data(path, timestep)
        else:
            # get data from engine
            timesteps = np.array(self.physics.engine.t).reshape(1,)
            cell_id = np.arange(self.reservoir.mesh.n_blocks)
            X = np.array(self.physics.engine.X, copy = True)
            var_names = self.physics.vars

        # Initialize property_array
        n_vars = len(var_names)
        n_ops = self.physics.n_ops
        nb = len(cell_id)
        props = list(var_names) + output_properties if output_properties is not None else list(var_names)
        property_array = {prop: np.zeros((len(timesteps), nb)) for prop in props}
        if output_properties is not None:
            prop_idxs = [self.properties.index(prop) for prop in output_properties]

        # Loop over timesteps
        for ts, timestep in enumerate(timesteps):
            # Extract vector of states
            for j, variable in enumerate(var_names):
                if engine is False:
                    property_array[variable][ts, :] = X[ts,:nb,j]
                else:
                    property_array[variable][ts, :] = X[j::n_vars]

            if output_properties is not None:
                states_numpy = np.stack([property_array[var][ts] for var in var_names]).T.flatten()
                state = value_vector(states_numpy)
                values = value_vector(np.zeros(n_ops * nb))
                values_numpy = np.array(values, copy=False)
                dvalues = value_vector(np.zeros(n_ops * nb * n_vars))
                i = 0
                for region, prop_itor in self.physics.property_itor.items():
                    prop_itor.evaluate_with_derivatives(state, self.physics.engine.region_cell_idx[i], values, dvalues)
                    i += 1

                for j, prop in enumerate(output_properties):
                    property_array[prop][ts] = values_numpy[prop_idxs[j]::n_ops]

        return timesteps, property_array

    def output_to_xarray(self, output_properties: list = None, timestep: int = None):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :returns: property_array
        :rtype: np.ndarray
        """
        # Interpolate properties
        if timestep is None:
            timesteps, data = self.output_properties(output_properties)
        else:
            timesteps, data = self.output_properties(output_properties, timestep)
        props = list(data.keys())

        # Initialize coords and data_vars for Xarray Dataset
        array_shape = (len(timesteps), self.reservoir.nz, self.reservoir.ny, self.reservoir.nx)
        for prop, array in data.items():
            data[prop] = array.reshape(array_shape)

        # Initialize coords and data_vars for Xarray Dataset
        dx, dy, dz = self.reservoir.global_data['dx'], self.reservoir.global_data['dy'], self.reservoir.global_data[
            'dz']
        x = np.cumsum(dx[:, 0, 0]) - dx[0, 0, 0] * 0.5
        y = np.cumsum(dy[0, :, 0]) - dy[0, 0, 0] * 0.5
        z = np.cumsum(dz[0, 0, :]) - dz[0, 0, 0] * 0.5
        # coords = {'time': timesteps, 'x': x, 'y': y, 'z': z}
        coords = {'time': timesteps, 'z': z, 'y': y, 'x': x}
        data_vars = {prop: (list(coords.keys()), data[prop]) for prop in props}
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)

        dataset.to_netcdf(os.path.join(self.output_folder, 'reservoir_xarray.nc'))

        return dataset

    def plot_xarray(self, xarray_data, timestep: int = -1, x: int = None, y: int =None, z: int = None):
        """
        :param xarray_data: xarray data set
        :param timestep: time index
        :param x: index in x-dimension
        :param y: index in y-dimension
        :param z: index in z-dimension
        """

        assert timestep < len(xarray_data['time']), 'time step should be less than %d' % len(xarray_data['time'])

        var_names = list(xarray_data.data_vars)
        nrows = len(var_names)
        for i, var in enumerate(var_names):
            plt.figure()
            if z is not None:
                assert z < len(xarray_data['z']), 'z-level step should be less than %d' % len(xarray_data['z'])
                xarray_data[var].isel(time=timestep, z=z).plot(cmap='jet')
                plt.savefig(
                    os.path.join(self.output_folder, 'figures') + '\\%s_ts%d_z%d.png'%(var, timestep, z)
                            )

            elif y is not None:
                assert y < len(xarray_data['y']), 'y-level step should be less than %d' % len(xarray_data['y'])
                xarray_data[var].isel(time=timestep, y=y).plot(cmap='jet')
                plt.savefig(self.output_folder + '/figures/%s_ts%d_y%d.png' % (var, timestep, y))

            elif x is not None:
                assert x < len(xarray_data['x']), 'x-level step should be less than %d' % len(xarray_data['x'])
                xarray_data[var].isel(time=timestep, x=x).plot(cmap='jet')
                plt.savefig(self.output_folder + '/figures/%s_ts%d_zx%d.png'%(var, timestep, z))

            else:
                # model is 1D reservoir
                xarray_data[var].isel(time=timestep).plot()
                plt.savefig(self.output_folder + '/figures/%s_ts%d.png' % (var, timestep))
        plt.close()

    def output_to_vtk(self, ith_step: int = None, output_directory: str = None, output_properties: list = None):
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

        main_dir = os.path.join(self.output_folder, 'vtk_files')
        if not os.path.exists(main_dir):
            os.mkdir(main_dir)

        # Set default output directory
        if output_directory is None:
            output_directory = self.output_folder

        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        tot_props = self.physics.vars + ev_props

        if output_properties is None:
            # If None, all variables and properties from property_operators will be passed
            # prop_idxs = {prop: i for i, prop in enumerate(tot_props)}
            prop_idxs = {prop: i for i, prop in enumerate(ev_props)}
        else:
            # Else, it finds the indices of output_properties in the output data
            prop_idxs = {prop: tot_props.index(prop) for prop in output_properties}

        timesteps, property_array = self.output_properties(output_properties=list(prop_idxs.keys()), timestep=ith_step)

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

    def plot_well_rates(self, types_of_well_rates: list):
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

        h5_well_data_address = os.path.join(self.output_folder, self.well_filename)
        h5_well_data = load_hdf5_to_dict(h5_well_data_address)

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
            os.mkdir(main_dir)
        elif os.path.exists(main_dir):
            shutil.rmtree(main_dir)
            os.mkdir(main_dir)
        for well in self.reservoir.wells:
            well_dir = os.path.join(main_dir, 'well_' + well.name)
            os.mkdir(well_dir)
            for perf in well.perforations:
                perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                os.mkdir(perf_dir)

        for rate_type in types_of_well_rates:
            if rate_type == 'heat_rate' and not self.physics.thermal:
                continue

            time, Python_rates = calc_rates_at_perforations(h5_well_data, perfs_conn_ids, geometric_WI, rate_type, self.physics.thermal,pc)

            """""""""  Plot well rates over time """""""""
            """ Rates for each perforation """
            perf_counter = 0
            for well in self.reservoir.wells:
                for perf in well.perforations:
                    if rate_type in ['phases_molar_rates', 'phases_mass_rates', 'phases_volumetric_rates']:
                        for phase in range(pc.nph):
                            phase_molar_rate_for_perf = - Python_rates[:, perf_counter, phase]

                            plt.figure()
                            plt.plot(time, phase_molar_rate_for_perf, color='r', marker='o', markersize=5)
                            plt.xlabel('time [day]', fontsize=16)

                            if rate_type == 'phases_molar_rates':
                                plt.ylabel(pc.phases_name[phase] + ' molar rate [kmol/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                                plt.savefig(perf_dir + '\well_' + well.name + '_perf_' + str(perf[0]) + '_molar_rate_' +
                                            pc.phases_name[phase] + '.png')
                            elif rate_type == 'phases_mass_rates':
                                plt.ylabel(pc.phases_name[phase] + ' mass rate [kg/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                                plt.savefig(perf_dir + '\well_' + well.name + '_perf_' + str(perf[0]) + '_mass_rate_' +
                                            pc.phases_name[phase] + '.png')
                            elif rate_type == 'phases_volumetric_rates':
                                plt.ylabel(pc.phases_name[phase] + ' volumetric rate [m$^3$/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                                plt.savefig(
                                    perf_dir + '\well_' + well.name + '_perf_' + str(perf[0]) + '_volumetric_rate_' +
                                    pc.phases_name[phase] + '.png')

                    elif rate_type in ['components_molar_rates', 'components_mass_rates']:
                        for component in range(pc.nc_fl):
                            if rate_type == 'components_molar_rates':
                                component_molar_rate_for_perf = - np.sum(
                                    Python_rates[:, perf_counter, component::pc.nc_fl], axis=1)
                            elif rate_type == 'components_mass_rates':
                                component_mass_rate_for_perf = - np.sum(
                                    Python_rates[:, perf_counter, component::pc.nc_fl], axis=1) * pc.Mw[component]

                            plt.figure()
                            if rate_type == 'components_molar_rates':
                                plt.plot(time, component_molar_rate_for_perf, color='r', marker='o', markersize=5)
                            elif rate_type == 'components_mass_rates':
                                plt.plot(time, component_mass_rate_for_perf, color='r', marker='o', markersize=5)

                            plt.xlabel('time [day]', fontsize=16)

                            if rate_type == 'components_molar_rates':
                                plt.ylabel(pc.components_name[component] + ' molar rate [kmole/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                                plt.savefig(perf_dir + '\well_' + well.name + '_perf_' + str(perf[0]) + '_molar_rate_' +
                                            pc.components_name[component] + '.png')
                            elif rate_type == 'components_mass_rates':
                                plt.ylabel(pc.components_name[component] + ' mass rate [kg/day]', fontsize=16)
                                perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                                plt.savefig(perf_dir + '\well_' + well.name + '_perf_' + str(perf[0]) + '_mass_rate_' +
                                            pc.components_name[component] + '.png')

                    elif rate_type == 'heat_rate':
                        heat_rate_of_all_phases_for_perf = - np.sum(Python_rates[:, perf_counter, :], axis=1)

                        plt.figure()
                        plt.plot(time, heat_rate_of_all_phases_for_perf, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel('heat rate [kJ/day]', fontsize=16)
                        perf_dir = os.path.join(main_dir, 'well_' + well.name, 'perf_' + str(perf[0]))
                        plt.savefig(perf_dir + '\well_' + well.name + '_perf_' + str(perf[0]) + '_heat_rate.png')

                    perf_counter += 1

            """ Total rates for each well """
            if rate_type == 'phases_molar_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for phase in range(pc.nph):
                        phase_molar_rate_for_well = 0
                        for perf in well.perforations:
                            phase_molar_rate_for_well += - Python_rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_molar_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' molar rate [kmol/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.savefig(well_dir + '\well_' + well.name + '_molar_rate_' + pc.phases_name[phase] + '.png')

                    perf_counter += len(well.perforations)

            elif rate_type == 'phases_mass_rates':
                perf_counter = 0
                for well in self.reservoir.wells:

                    for phase in range(pc.nph):
                        phase_mass_rate_for_well = 0
                        for perf in well.perforations:
                            phase_mass_rate_for_well += - Python_rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_mass_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' mass rate [kg/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.savefig(well_dir + '/well_' + well.name + '_mass_rate_' + pc.phases_name[phase] + '.png')

                    perf_counter += len(well.perforations)

            elif rate_type == 'phases_volumetric_rates':
                perf_counter = 0
                for well in self.reservoir.wells:

                    for phase in range(pc.nph):
                        phase_volumetric_rate_for_well = 0
                        for perf in well.perforations:
                            phase_volumetric_rate_for_well += - Python_rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, phase_volumetric_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.phases_name[phase] + ' volumetric rate [m$^3$/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.savefig(
                            well_dir + '\well_' + well.name + '_volumetric_rate_' + pc.phases_name[phase] + '.png')

                    perf_counter += len(well.perforations)

            elif rate_type == 'components_molar_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for component in range(pc.nc_fl):
                        component_molar_rate_for_well = 0
                        for perf in well.perforations:
                            component_molar_rate_for_well += - np.sum(
                                Python_rates[:, perf_counter, component::pc.nc_fl], axis=1)

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, component_molar_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.components_name[component] + ' molar rate [kmole/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.savefig(
                            well_dir + '\well_' + well.name + '_molar_rate_' + pc.components_name[component] + '.png')

                    perf_counter += len(well.perforations)

            elif rate_type == 'components_mass_rates':
                perf_counter = 0
                for well in self.reservoir.wells:
                    for component in range(pc.nc_fl):
                        component_mass_rate_for_well = 0
                        for perf in well.perforations:
                            component_mass_rate_for_well += - np.sum(Python_rates[:, perf_counter, component::pc.nc_fl],
                                                                     axis=1) * pc.Mw[component]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                        plt.figure()
                        plt.plot(time, component_mass_rate_for_well, color='r', marker='o', markersize=5)
                        plt.xlabel('time [day]', fontsize=16)
                        plt.ylabel(pc.components_name[component] + ' mass rate [kg/day]', fontsize=16)
                        well_dir = os.path.join(main_dir, 'well_' + well.name)
                        plt.savefig(
                            well_dir + '/well_' + well.name + '_mass_rate_' + pc.components_name[component] + '.png')

                    perf_counter += len(well.perforations)

            elif rate_type == 'heat_rate':
                perf_counter = 0
                for well in self.reservoir.wells:

                    heat_rate_for_well = 0
                    for phase in range(pc.nph):
                        for perf in well.perforations:
                            heat_rate_for_well += - Python_rates[:, perf_counter, phase]

                            perf_counter += 1
                        perf_counter -= len(well.perforations)

                    plt.figure()
                    plt.plot(time, heat_rate_for_well, color='r', marker='o', markersize=5)
                    plt.xlabel('time [day]', fontsize=16)
                    plt.ylabel('heat rate [kJ/day]', fontsize=16)
                    well_dir = os.path.join(main_dir, 'well_' + well.name)
                    plt.savefig(well_dir + '/well_' + well.name + '_heat_rate.png')

                    perf_counter += len(well.perforations)

    # def output_phase_properties(self, phase_props_labels=['dens', 'dens_m', 'sat', 'nu', 'mu', 'kr', 'pc', 'enthalpy', 'cond'], timestep=None, verbose=False):
    #     """
    #     By default the PropertyContainer() contains a number of phase properties. With the this function enables fast interpretation of
    #     phase properties at every grid block.
    #
    #     :param phase_props_labels: list of desired phase properties
    #     :param timestep : desired timestape at which you want to evaluate properties
    #     :return: property_array: property_array containing all the desired props at each grid cell
    #     """
    #
    #     for region in self.physics.regions:  # loop over the different sets of operators
    #         phase_props = self.physics.property_containers[region].phase_props
    #         temp_dict = {}
    #         for i, name in enumerate(phase_props_labels):
    #             for j, phase_name in enumerate(self.physics.phases):
    #                 temp_dict[f"{name} {phase_name}"] = lambda i=i, j=j: phase_props[i][j]
    #
    #         self.physics.property_containers[region].output_props = temp_dict
    #
    #     self.physics.init_physics(verbose=True)
    #     self.reset()
    #     self.set_output()
    #
    #     target_solution_file = os.path.join('one_to_rule_them_all', 'restart_data.h5')
    #     filepath = target_solution_file
    #     timesteps, property_array = self.output_properties(filepath=filepath,
    #                                                        output_properties=self.output_properties,
    #                                                        timestep=None)
    #
    #     if verbose:
    #         for i, name in enumerate(property_array.keys()):
    #             plt.figure()
    #             plt.title(name)
    #             plt.plot(property_array[name].T)
    #             plt.show()
    #
    #     return property_array