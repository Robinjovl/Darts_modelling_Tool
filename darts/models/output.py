import os
import numpy as np
import h5py

from darts.engines import value_vector

class Output:
    """
    Base class for all output related functionality
    """
    def __init__(self, reservoir, physics, output_folder: str = None, restart: bool = False):
        super().__init__()

        self.reservoir = reservoir
        self.physics = physics

        self.sol_filename = "reservoir.h5"
        self.well_filename = 'well_data.h5'

        self.output_folder = 'output'
        if output_folder is not None:
            self.output_folder = output_folder

        if restart is False:
            self.save_data_to_h5(kind = 'reservoir')

    def load_restart_data(self, filename: str = os.path.join('restart', 'solution.h5'), timestep = -1):
        """
        Function to load data from previous simulation and uses them for following simulation.
        :param output_folder: restart_data filename
        :type output_folder: str
        """
        time, cell_id, X, var_names = self.read_specific_data(filename, timestep)

        print('Restarting from %s at time = %f days' % (filename, time[0]))

        self.physics.engine.t = time[0]
        self.physics.engine.X = value_vector(X.flatten())
        self.physics.engine.Xn = value_vector(X.flatten())

        self.save_data_to_h5(kind='solution')

    def configure_h5_output(self, filename: str, cell_ids, description, add_static_data: bool = False):
        """
        Configuration of *.h5 output

        :param filename: *.h5 filename
        :param cell_ids: np.array of cell indexes for output
        :param description: description for *.h5
        :param add_static_data: flag to add static output
        """
        with h5py.File(filename, 'w') as f:
            ## static data group
            if add_static_data:
                static_group = f.create_group('static')
                block_m = np.array(self.reservoir.mesh.block_m, copy=False)
                block_p = np.array(self.reservoir.mesh.block_p, copy=False)
                static_group.create_dataset('block_m', data=block_m)
                static_group.create_dataset('block_p', data=block_p)

            ## dynamic data group
            dynamic_group = f.create_group('dynamic')
            dynamic_group.create_dataset('time', shape=(0,), maxshape=(None,))

            # add solution
            if self.reservoir.mesh.n_blocks > 0 and self.physics.n_vars > 0:
                nb = cell_ids.size
                cell_ids_dataset = dynamic_group.create_dataset('cell_id', shape=(nb,), dtype=np.int32)
                cell_ids_dataset[:] = cell_ids
                dynamic_group.create_dataset('X', shape=(0, nb, self.physics.n_vars),
                                             maxshape=(None, nb, self.physics.n_vars), dtype = np.float64)

            # add variable names
            datatype = h5py.special_dtype(vlen=str)  # dtype for variable-length strings
            var_names = dynamic_group.create_dataset('variable_names', (self.physics.n_vars,), dtype=datatype)
            var_names[:] = self.physics.vars

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
        elif kind == 'reservoir':
            path = os.path.join(self.output_folder, self.sol_filename)
        else:
            print("Please use either kind='well' or kind='solution' in save_data_to_h5")
            return
        self.save_specific_data(path)

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

    def output_properties(self, output_properties: list = None, timestep: int = None) -> tuple:
        """
        Function to read *.h5 data and evaluate properties per grid block, per timestep
        :param output_properties: List of properties to evaluate for output
        :return property_array : dictionary containing the states and evaluated properties
        :return timesteps: np.ndarray containing the timesteps at which the properties were evaluated
        :rtype: tuple
        """
        # Read binary file
        path = os.path.join(self.output_folder, self.sol_filename)
        if timestep is None:
            timesteps, cell_id, X, var_names = self.read_specific_data(path)
        else:
            timesteps, cell_id, X, var_names = self.read_specific_data(path, timestep)

        # Initialize property_array
        n_vars = len(var_names)
        n_ops = self.physics.n_ops
        nb = self.reservoir.mesh.n_res_blocks
        props = list(var_names) + output_properties if output_properties is not None else list(var_names)
        property_array = {prop: np.zeros((len(timesteps), nb)) for prop in props}
        prop_idxs = [list(self.physics.property_containers[0].output_props.keys()).index(prop)
                     for prop in output_properties]

        # Loop over timesteps
        for k, timestep in enumerate(timesteps):
            # Extract vector of states
            for j, variable in enumerate(var_names):
                property_array[variable][k, :] = X[k, :nb, j]

            if output_properties is not None:
                state = value_vector(np.stack([property_array[var][k] for var in var_names]).T.flatten())
                values = value_vector(np.zeros(n_ops * nb))
                values_numpy = np.array(values, copy=False)
                dvalues = value_vector(np.zeros(n_ops * nb * n_vars))
                i = 0
                for region, prop_itor in self.physics.property_itor.items():
                    prop_itor.evaluate_with_derivatives(state, self.physics.engine.region_cell_idx[i], values, dvalues)
                    i += 1

                for j, prop in enumerate(output_properties):
                    property_array[prop][k] = values_numpy[prop_idxs[j]::n_ops]

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
        array_shape = (len(timesteps), self.reservoir.nx, self.reservoir.ny, self.reservoir.nz)
        for prop, array in data.items():
            data[prop] = array.reshape(array_shape)

        # Initialize coords and data_vars for Xarray Dataset
        dx, dy, dz = self.reservoir.global_data['dx'], self.reservoir.global_data['dy'], self.reservoir.global_data[
            'dz']
        x = np.cumsum(dx[:, 0, 0]) - dx[0, 0, 0] * 0.5
        y = np.cumsum(dy[0, :, 0]) - dy[0, 0, 0] * 0.5
        z = np.cumsum(dz[0, 0, :]) - dz[0, 0, 0] * 0.5
        coords = {'time': timesteps, 'x': x, 'y': y, 'z': z}
        data_vars = {prop: (list(coords.keys()), data[prop]) for prop in props}
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)

        dataset.to_netcdf(os.path.join(self.output_folder, 'solution_xarray.nc'))

        return dataset

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
        self.timer.node["vtk_output"].start()
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
                self.reservoir.output_to_vtk(t, time, output_directory, prop_names, data)
            else:
                self.reservoir.output_to_vtk(ith_step, time, output_directory, prop_names, data)

        self.timer.node["vtk_output"].stop()