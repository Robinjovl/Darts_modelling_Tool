import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from darts.engines import (
    index_vector,
    timer_node,
    value_vector,
    well_control_iface,
)
from darts.physics.base.operators_base import PropertyOperators
from darts.physics.blackoil import BlackOil
from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH
from darts.physics.super.physics import Compositional
from darts.tools.hdf5_tools import load_hdf5_to_dict

# %%


class Output:
    """
    Base class for all output related functionality
    """

    def __init__(
        self,
        timer: timer_node,
        reservoir,
        physics,
        op_list,
        params,
        output_folder: str,
        sol_filename: str,
        well_filename: str,
        save_initial: bool,
        all_phase_props: bool,
        precision: str,
        compression: str,
        verbose: bool,
    ):
        """
        Class constructor method for output related functionalities including saving primary variables (state variables),
        evaluating secondary variables (properties) and creating visualizations.

        :param timer: timer object to measure time spent saving data, and evaluating properties
        :reservoir: reservoir object
        :param physics: physics object
        :param op_list: list of operator interpolators
        :param params: engine params
        :param output_folder: output folder for saved data and figures
        :param sol_filename: hdf5 filename for saving reservoir solution
        :param well_filename: hdf5 filename for saving well solution
        :param save_initial: boolean flag to save initial conditions of reservoir
        :param all_phase_props: boolean flag to define properties (secondary variables) according to a predefined list.
        :param compression: boolean flag to enable compression of hdf5 data
        :param verbose: boolean flag to enable verbose output
        """
        super().__init__()

        self.reservoir = reservoir
        self.physics = physics
        self.op_list = op_list
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)

        self.params = params
        self.verbose = verbose

        self.master_timer = timer
        self.timer = timer.node["output"]
        self.timer.node["saving_reservoir_data"] = timer_node()
        self.timer.node["saving_well_data"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()
        self.timer.node["output_well_time_data"] = timer_node()
        self.timer.node["exporting_property_array"] = timer_node()

        self.output_folder = output_folder
        self.sol_filename = sol_filename
        self.well_filename = well_filename
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        self.precision = precision
        self.compression = compression
        self.precision_map = {"d": np.float64, "s": np.float32}

        self.properties = list(self.physics.property_containers[0].output_props.keys())
        if len(self.properties) < self.physics.n_ops:
            self.n_ops = self.physics.n_ops
        else:
            self.n_ops = len(self.properties) + self.physics.n_vars

        if save_initial:
            self.save_data_to_h5(kind="reservoir")

        if all_phase_props:
            self.set_phase_properties()

        self.unit_dictionary = {
            "dens": "[kg/m3]",
            "densm": "[kmol/m3]",
            "sat": "[-]",
            "mu": "[cP]",
            "kr": "[-]",
            "pc": "[Bar]",
            "pressure": "[Bar]",
            "enthalpy": "[kJ]",
            "cond": "[kJ/m/day/K]",
            "temperature": "[K]",
        }

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
                self.variable_units[name] = ""

        for name in self.properties:
            try:
                self.variable_units[name] = self.unit_dictionary[name.split("_")[0]]
            except:
                self.variable_units[name] = ""

        return

    def set_phase_properties(self):
        """
        This function constructs a predefined set of property operators for the compositional/geothermal physics class.
        """

        if type(self.physics) is Compositional or type(self.physics) is BlackOil:
            phase_props_labels = [
                "dens",
                "densm",
                "sat",
                "mu",
                "kr",
                "pc",
                "enthalpy",
                "cond",
            ]
            self.physics.property_itor = {}

            for (
                region
            ) in self.physics.regions:  # loop over the different sets of operators
                pc = self.physics.property_containers[region]
                temp_dict = {}  # output_properties dictionary

                # Loop through each property label and phase name
                for i, name in enumerate(phase_props_labels):
                    for j in range(len(pc.phase_props[i])):
                        temp_dict[f"{name}_{self.physics.phases[j]}"] = (
                            lambda ii=i,
                            jj=j,
                            rr=region: self.physics.property_containers[rr].phase_props[
                                ii
                            ][jj]
                        )

                # Add molar phase fractions
                for i in range(pc.x.shape[1]):
                    for j in range(pc.x.shape[0]):
                        temp_dict[
                            f"x_{self.physics.phases[j]}_{pc.components_name[i]}"
                        ] = (
                            lambda ii=i,
                            jj=j,
                            rr=region: self.physics.property_containers[rr].x[jj, ii]
                        )

                self.physics.property_operators[region] = PropertyOperators(
                    pc, self.physics.thermal, temp_dict
                )
                self.physics.property_itor[region], n_ops = (
                    self.physics.create_interpolator(
                        self.physics.property_operators[region],
                        n_ops=self.physics.n_ops,
                        axes_min=self.physics.axes_min,
                        axes_max=self.physics.axes_max,
                        platform='cpu',
                        algorithm='multilinear',
                        mode='adaptive',
                        precision='d',
                        timer_name=f'property {region:d} interpolation',
                        region=str(region),
                    )
                )

                # Assign the temporary dictionary to output_props for the region
                self.physics.property_containers[region].output_props = temp_dict
                self.n_ops = n_ops

        elif type(self.physics) is Geothermal or type(self.physics) is GeothermalPH:
            phase_props_labels = [
                "dens",
                "densm",
                "sat",
                "mu",
                "kr",
                "pc",
                "enthalpy",
            ]  # 'cond'
            self.physics.property_itor = {}

            for (
                region
            ) in self.physics.regions:  # loop over the different sets of operators
                pc = self.physics.property_containers[region]
                temp_dict = {}

                # add temperature
                temp_dict['temperature'] = lambda container=pc: container.temperature

                # Loop through each property label and phase name
                for i, name in enumerate(phase_props_labels):
                    for j in range(self.physics.property_containers[region].nph):
                        temp_dict[f"{name}_{self.physics.phases[j]}"] = (
                            lambda ii=i,
                            jj=j,
                            rr=region: self.physics.property_containers[rr].phase_props[
                                ii
                            ][jj]
                        )

                self.physics.property_operators[region] = PropertyOperators(
                    pc, thermal=False, props=temp_dict
                )
                self.physics.property_itor[region], n_ops = (
                    self.physics.create_interpolator(
                        self.physics.property_operators[region],
                        n_ops=self.physics.property_operators[region].n_ops,
                        axes_min=self.physics.axes_min,
                        axes_max=self.physics.axes_max,
                        platform='cpu',
                        algorithm='multilinear',
                        mode='adaptive',
                        precision='d',
                        timer_name=f'property {region:d} interpolation',
                        region=str(region),
                    )
                )

                # Assign the temporary dictionary to output_props for the region
                self.physics.property_containers[region].output_props = temp_dict
                self.n_ops = n_ops

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

            self.physics.property_operators[region] = PropertyOperators(
                self.physics.property_containers[region],
                self.physics.thermal,
                output_dictionary,
            )
            self.physics.property_itor[region], n_ops = (
                self.physics.create_interpolator(
                    self.physics.property_operators[region],
                    n_ops=self.physics.n_ops,
                    axes_min=self.physics.axes_min,
                    axes_max=self.physics.axes_max,
                    platform='cpu',
                    algorithm='multilinear',
                    mode='adaptive',
                    precision='d',
                    timer_name=f'property {region:d} interpolation',
                    region=str(region),
                )
            )
            self.n_ops = n_ops
            self.properties = list(output_dictionary.keys())

        return

    def save_array(self, array, filename, compression_level=1):
        """
        This function saved any dictionary as an h5 file with compression

        : param array: data
        : type array: dict
        : param filename: filename in the format filename.h5
        : type filename: str
        : param compression_level : int value between 0 and 9
        : type : int
        """
        output_directory = os.path.join(self.output_folder, filename)
        with h5py.File(output_directory, "w") as h5f:
            for key, arr in array.items():
                h5f.create_dataset(
                    key,
                    data=arr,
                    compression="gzip",
                    compression_opts=compression_level,
                )
        return 0

    def load_array(self, file_directory):
        """
        This function loads any saved data in h5 file format

        : param file_directory: filename in the format filename.h5
        : type file_directory: str
        """

        array = {}
        with h5py.File(file_directory, "r") as h5f:
            for key in h5f.keys():
                array[key] = np.array(h5f[key])
        return array

    def append_properties_to_reservoir(self, time: float, property_array: dict):
        """
        Appends secondary properties to the existing 'reservoir.h5' file under the group 'properties'.

        :param time: timestep index to write properties for.
        :param property_array: Dictionary with property names as keys and arrays (1D over cells) as values.
        """

        with h5py.File(self.sol_filepath, "a") as f:
            time_vector = f["dynamic/time"][:]
            if time in time_vector:
                timestep = int(np.where(time_vector == time)[0][0])

                if "properties" not in f:
                    f.create_group("properties")
                else:
                    pass

                prop_group = f["properties"]
                max_ts = f["dynamic/time"].shape[0]

                for key, data in property_array.items():
                    data = np.asarray(data).reshape(-1)  # Ensure 1D array

                    if key not in prop_group:
                        shape = (max_ts, len(data))
                        dset = prop_group.create_dataset(
                            key,
                            shape=shape,
                            maxshape=(
                                None,
                                len(data),
                            ),  # max shape none ensures that we can append as much data as possible
                            dtype=data.dtype,
                            compression="gzip",
                            compression_opts=2,
                        )

                    else:
                        dset = prop_group[key]
                        if len(data) != dset.shape[1]:
                            raise ValueError(
                                f"Shape mismatch for property '{key}': expected {dset.shape[1]}, got {len(data)}"
                            )
                        if timestep >= dset.shape[0]:
                            dset.resize((timestep + 1, dset.shape[1]))

                    dset[timestep, :] = data

            else:
                raise ValueError(
                    f"Timestamp {time} does not exist in the solution.h5 file."
                )

    def save_property_array(self, time_vector, property_array, filename=None):
        """
        Saves property_array to an HDF5 file.

        :param time_vector : Array of timesteps
        :param property_array : Dictionary where keys are property names and values are NumPy arrays.
        :param filename : Name of the HDF5 file to save to.
        """

        self.timer.start()
        self.timer.node["exporting_property_array"].start()

        if filename is None:
            self.append_properties_to_reservoir(time_vector, property_array)
        else:
            compression_level = 2
            output_directory = os.path.join(self.output_folder, filename)

            with h5py.File(output_directory, "w") as h5f:
                # Save the time vector with compression
                h5f.create_dataset(
                    "time_vector",
                    data=time_vector,
                    compression="gzip",
                    compression_opts=compression_level,
                )

                # Save each property array with compression
                for key, array in property_array.items():
                    h5f.create_dataset(
                        key,
                        data=array,
                        compression="gzip",
                        compression_opts=compression_level,
                    )

        self.timer.node["exporting_property_array"].stop()
        self.timer.stop()

        return

    def load_property_array(self, file_directory="property_array.h5"):
        """
        Load saved properties back into a dictionary.
        :param file_directory : filepath to saved property_array.h5

        :return time_vector:  available timesteps
        :return property_array: dictionary of properties
        """
        try:
            property_array = {}
            with h5py.File(file_directory, "r") as h5f:
                # Load time vector
                time_vector = np.array(h5f["time_vector"])

                # Load each property array
                for key in h5f.keys():
                    if key != "time_vector":  # Skip time vector in property dictionary
                        property_array[key] = np.array(h5f[key])
        except Exception as _err:
            with h5py.File(self.sol_filepath, "r") as f:
                if "properties" not in f:
                    raise KeyError(
                        "No 'properties' group found in the reservoir.h5 file."
                    ) from _err

                time_vector = np.array(f["dynamic/time"][:])
                prop_group = f["properties"]

                property_array = {
                    key: np.array(dset) for key, dset in prop_group.items()
                }

        return time_vector, property_array

    def print_simulation_parameters(self, mode="table"):
        """
        Function that prints class variables into a .txt file
        """
        filepath = os.path.join(self.output_folder, "simulation_input_parameters.txt")

        if mode == "dump":
            obj_list = [self.params, self.reservoir, self.physics]
            with open(filepath, "w") as f:
                for i, obj in enumerate(obj_list):
                    f.write(f"------- {i + 1}: {obj.__class__.__name__} -------\n")
                    for attr in dir(obj):
                        if not attr.startswith("_"):
                            try:
                                value = getattr(obj, attr)
                                f.write(f"{attr}: {value}\n")
                            except Exception as e:
                                f.write(f"{attr}: <error: {e}>\n")
                    f.write("\n")  # Add a blank line between objects
        else:
            with open(filepath, "w") as f:
                f.write(
                    "-----------------------------PHYSICS------------------------\n"
                )
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

        return 0

    def configure_h5_output(
        self, filename: str, cell_ids, description, add_static_data: bool = False
    ):
        """
        Configuration of *.h5 output

        :param filename: *.h5 filename
        :param cell_ids: np.array of cell indexes for output
        :param description: description for *.h5
        :param add_static_data: flag to add static output
        """

        with h5py.File(filename, "w") as f:
            # add static data group
            if add_static_data:
                static_group = f.create_group("static")
                block_m = np.array(self.reservoir.mesh.block_m, copy=False)
                block_p = np.array(self.reservoir.mesh.block_p, copy=False)
                static_group.create_dataset("block_m", data=block_m)
                static_group.create_dataset("block_p", data=block_p)

            # add dynamic data group
            dynamic_group = f.create_group("dynamic")
            dynamic_group.create_dataset(
                "time",
                shape=(0,),
                maxshape=(None,),
                dtype=self.precision_map[self.precision],
            )

            # add CFL_max dataset, maximum taken over all cells
            dynamic_group.create_dataset(
                "CFL_max",
                shape=(0,),
                maxshape=(None,),
                dtype=self.precision_map[self.precision],
            )

            # add solution
            if self.reservoir.mesh.n_blocks > 0 and self.physics.n_vars > 0:
                nb = cell_ids.size
                cell_ids_dataset = dynamic_group.create_dataset(
                    "cell_id", shape=(nb,), dtype=np.int32
                )
                cell_ids_dataset[:] = cell_ids

            dynamic_group.create_dataset(
                "X",
                shape=(0, nb, self.physics.n_vars),
                maxshape=(None, nb, self.physics.n_vars),
                dtype=self.precision_map[self.precision],
                compression=self.compression,
            )

            # add variable names
            datatype = h5py.special_dtype(vlen=str)  # dtype for variable-length strings
            dynamic_group.create_dataset(
                "variable_names", data=np.array(self.physics.vars, dtype=datatype)
            )

            # write brief description
            f.attrs["description"] = description

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
        os.makedirs(os.path.join(self.output_folder, "figures"), exist_ok=True)

        # solution ouput
        if kind == "reservoir":
            sol_output_path = os.path.join(self.output_folder, self.sol_filename)
            if os.path.exists(sol_output_path):  # and not restart:
                os.remove(sol_output_path)
            self.configure_h5_output(
                filename=sol_output_path,
                cell_ids=np.arange(self.reservoir.mesh.n_res_blocks),
                add_static_data=False,
                description="Reservoir data",
            )

        # Find relevant connections for well data
        if kind == "well":
            block_m = np.array(self.reservoir.mesh.block_m, copy=False)
            block_p = np.array(self.reservoir.mesh.block_p, copy=False)
            well_conn_ids = np.argwhere(block_p >= self.reservoir.mesh.n_res_blocks)[
                :, 0
            ]
            self.id_well_data = np.unique(block_m[well_conn_ids])

            # well output
            well_output_path = os.path.join(self.output_folder, self.well_filename)
            if os.path.exists(well_output_path):
                os.remove(well_output_path)
            self.configure_h5_output(
                filename=well_output_path,
                cell_ids=self.id_well_data,
                add_static_data=True,
                description="Well data",
            )

        if hasattr(self, "output_configured"):
            self.output_configured.append(kind)
        else:
            self.output_configured = [kind]

    def save_specific_data(self, filename, X_data=None):
        """
        Save simulation data to an HDF5 file.

        :param filename: Path to the HDF5 file.
        :type filename: str
        :param X_data: Optional tuple containing (time array, data array). If None, current engine state is saved.
        :type X_data: tuple or None
        """
        is_batch = X_data is not None

        with h5py.File(filename, "a") as f:
            time_dataset = f["dynamic/time"]
            x_dataset = f["dynamic/X"]
            cell_id = f["dynamic/cell_id"][:]
            cfl_dataset = f["dynamic/CFL_max"]

            if is_batch:
                times, data_array, cfl_values = X_data
                times = np.asarray(times)
                data_array = np.asarray(data_array)
                cfl_values = np.asarray(cfl_values)
                n_new = len(times)
            else:
                times = np.array([self.physics.engine.t])
                X = np.asarray(self.physics.engine.X)
                reshaped = X.reshape(
                    (self.reservoir.mesh.n_blocks, self.physics.n_vars)
                )[cell_id]
                data_array = np.expand_dims(
                    reshaped, axis=0
                )  # shape (1, n_cells, n_vars)
                cfl_values = np.array([self.physics.engine.CFL_max])
                n_new = 1

            # Resize datasets
            start_idx = time_dataset.shape[0]
            time_dataset.resize((start_idx + n_new,))
            x_dataset.resize(
                (start_idx + n_new, x_dataset.shape[1], x_dataset.shape[2])
            )
            cfl_dataset.resize((start_idx + n_new,))

            # Write data
            time_dataset[start_idx : start_idx + n_new] = times
            x_dataset[start_idx : start_idx + n_new, :, :] = data_array
            cfl_dataset[start_idx : start_idx + n_new] = cfl_values

        if self.verbose:
            mode = "batch" if is_batch else "single"
            print(f"[{mode}] Saved {n_new} entry(ies) to {filename}")

    def save_data_to_h5(self, kind):
        """
        Function to write output solution or well output to *.h5 file

        :param kind: 'well' for well output or 'solution' to write the whole solution vector
        :type kind: str
        """

        if not hasattr(self, "output_configured") or kind not in self.output_configured:
            self.configure_output(kind=kind)

        if kind == "well":
            path = os.path.join(self.output_folder, self.well_filename)
            self.timer.start()
            self.timer.node["saving_well_data"].start()
            self.save_specific_data(path)
            self.timer.node["saving_well_data"].stop()
            self.timer.stop()

        elif kind == "reservoir":
            path = os.path.join(self.output_folder, self.sol_filename)
            self.timer.start()
            self.timer.node["saving_reservoir_data"].start()
            self.save_specific_data(path)
            self.timer.node["saving_reservoir_data"].stop()
            self.timer.stop()

        else:
            print(
                "Please use either kind='well' or kind='reservoir' in save_data_to_h5"
            )

    def read_specific_data(
        self, filename: str, timestep: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            with h5py.File(filename, "r") as file:
                if timestep is None:
                    cell_id = file["dynamic/cell_id"][:]
                    var_names = file["dynamic/variable_names"][:]
                    time = file["dynamic/time"][:]

                    # memory check
                    dataset = file[
                        "dynamic/X"
                    ]  # does not load data into memory since we are not slicing
                    convert2MB = 1e6
                    estimated_size_mb = (
                        dataset.size * dataset.dtype.itemsize / convert2MB
                    )  # number of bytes per element
                    if estimated_size_mb > 1000:  # throw a warning if more than 1GB
                        print(
                            f"WARNING: Dataset 'X' is approximately {estimated_size_mb:.1f} MB. Loading it may impact memory performance."
                        )

                    X = file["dynamic/X"][:]

                else:
                    if not isinstance(timestep, int):
                        raise TypeError(
                            f"Expected 'timestep' to be an int, but got {type(timestep).__name__}"
                        )

                    cell_id = file["dynamic/cell_id"][:]
                    var_names = file["dynamic/variable_names"][:]

                    try:
                        time = file["dynamic/time"][timestep].reshape(1)
                        X = file["dynamic/X"][timestep].reshape(
                            1, len(cell_id), len(var_names)
                        )

                    except IndexError as err:
                        raise IndexError(
                            f"Timestep {timestep} does not exist in {filename}."
                        ) from err

            for i, name in enumerate(var_names):
                var_names[i] = name.decode()

        except FileNotFoundError as err:
            raise FileNotFoundError(f"File not found: {filename}.") from err

        return time, cell_id, X, var_names

    def get_mass_components(self, property_array):
        component_names = self.physics.property_containers[0].components_name
        Mw = np.array(self.physics.property_containers[0].Mw).reshape(-1, 1)

        # Extract properties from property_array
        sg = property_array["sat_g"][0]
        so = property_array['sat_o'][0]

        rhoV = property_array["dens_g"][0]
        rho_m_o = property_array["densm_o"][0]
        try:
            rho_m_w = property_array["densm_w"][0]
        except:
            pass

        self.x_components, self.y_components, self.w_components = [], [], []
        # for phase_name in self.physics.phases:
        for component_name in self.physics.components:
            self.x_components.append(property_array[f"x{component_name}"][0])  # oil
            self.y_components.append(property_array[f"y{component_name}"][0])  # gas

            try:
                self.w_components.append(property_array[f"w{component_name}"][0])
            except:
                pass

        self.x_components, self.y_components = (
            np.array(self.x_components),
            np.array(self.y_components),
        )

        # Compute molecular weight of the aqueous phase
        MWo = (
            np.sum(self.y_components[1:, :] * Mw[1:], axis=0)
            + (1 - np.sum(self.y_components[1:, :], axis=0)) * Mw[0]
        )

        # Mass fractions in oil phase
        w_components_vapor = (self.y_components * Mw) / MWo

        # Pore volume
        V = np.array(self.reservoir.mesh.volume, copy=False)[: self.reservoir.n]
        phi = np.array(self.reservoir.mesh.poro, copy=False)[: self.reservoir.n]

        # Calculate total mass for each component
        mass_components = {}
        mass_oil = {}
        mass_vapor = {}
        mass_water = {}
        for i, component_name in enumerate(component_names):
            # gas phase mass contribution
            mass_vapor[component_name] = phi * V * w_components_vapor[i] * sg * rhoV

            # oil phase mass contribution
            mass_oil[component_name] = (
                phi * V * so * self.x_components[i] * rho_m_o * Mw[i]
            )

            # water phase mass conribution
            try:
                mass_water[component_name] = (
                    phi * V * (1 - so - sg) * self.w_components[i] * rho_m_w * Mw[i]
                )
            except:
                mass_water[component_name] = 0

            # Total mass
            mass_components[component_name] = (
                mass_vapor[component_name]
                + mass_oil[component_name]
                + mass_water[component_name]
            )

        return mass_components, mass_vapor, mass_oil, mass_water

    def output_properties(
        self,
        filepath: str = None,
        output_properties: list = None,
        timestep: int = None,
        engine=False,
    ) -> tuple[np.ndarray, dict]:
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

        if self.verbose:
            print(f'Processing properties {output_properties} at timestep {timestep}')

        if output_properties is not None and not isinstance(output_properties, list):
            raise TypeError(
                f"Expected 'output_properties' to be a list, but got {type(output_properties).__name__}."
            )

        if not engine:
            # Evaluate properties from the HDF5 file
            if filepath is None:  # Establish filepath/name to HDF5 file
                path = os.path.join(self.output_folder, self.sol_filename)
            else:
                path = filepath
            timesteps, cell_id, X, var_names = self.read_specific_data(
                path, timestep
            )  # Read data from HDF5 file
        else:
            # Evaluate properties from the physics.engine.X
            timesteps = np.array(self.physics.engine.t).reshape(
                1,
            )  # current time
            cell_id = np.arange(self.reservoir.mesh.n_res_blocks)  # cell ids
            X = np.array(
                self.physics.engine.X[
                    : self.physics.n_vars * self.reservoir.mesh.n_res_blocks
                ],
                copy=True,
            )  # solution at current time
            var_names = self.physics.vars  # primary variable names

        n_vars = len(var_names)  # number of primary variables
        nb = len(cell_id)  # number of grid blocks
        output_properties = (
            output_properties
            if output_properties is not None
            else list(self.physics.vars)
        )  # complete list of properties

        # List of primary variables i.e. state variables
        primary_props = [prop for prop in output_properties if prop in var_names]
        primary_prop_idxs = {
            prop: list(var_names).index(prop) for prop in primary_props
        }

        # List of secondary properties
        secondary_props = [prop for prop in output_properties if prop not in var_names]
        secondary_prop_idxs = {}
        for prop in secondary_props:
            for container in self.physics.property_containers.values():
                if prop in container.output_props:
                    secondary_prop_idxs[prop] = list(
                        container.output_props.keys()
                    ).index(prop)
                    break
            else:
                raise KeyError(
                    f"Secondary property '{prop}' not found in any property container."
                )

        # define property array dictionary
        property_array = {
            prop: np.zeros((len(timesteps), nb))
            for prop in primary_props + secondary_props
        }

        # Loop over available timesteps
        for k, _t in enumerate(timesteps):
            # Extract primary properties from X vector
            for var_name, var_idx in primary_prop_idxs.items():
                if engine is False:
                    property_array[var_name][k] = X[k, :nb, var_idx]
                else:
                    property_array[var_name][k] = X[var_idx::n_vars]

            # Interpolate secondary properties
            if secondary_props:  # if empty this part is skipped
                if engine is False:
                    state = value_vector(
                        np.stack([X[k, :nb, j] for j in range(n_vars)])
                        .T.flatten()
                        .astype(np.float64)
                    )
                else:
                    state = value_vector(
                        np.stack([X[j::n_vars] for j in range(n_vars)]).T.flatten()
                    )

                values = value_vector(np.zeros(self.n_ops * nb))
                values_numpy = np.array(values, copy=False)
                dvalues = value_vector(np.zeros(self.n_ops * nb * n_vars))

                for region, prop_itor in self.physics.property_itor.items():
                    block_idx = np.where(self.op_num == region)[0].astype(np.int32)
                    prop_itor.evaluate_with_derivatives(
                        state, index_vector(block_idx), values, dvalues
                    )

                    for prop_name, prop_idx in secondary_prop_idxs.items():
                        temp = values_numpy[prop_idx :: self.n_ops]
                        property_array[prop_name][k][block_idx] = temp[block_idx]

        return timesteps, property_array

    def output_to_vtk(
        self,
        sol_filepath: str = None,
        ith_step: int = None,
        output_directory: str = None,
        output_properties: list = None,
        engine: bool = False,
        output_data: list = None,
    ):
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
        :param output_data: List [array of timesteps, dictionary of propertiy arrays]. Defaults to None, in which case properties are evaluated from the HDF5 file or engine
        :type output_data: list, optional
        """
        self.timer.start()
        self.timer.node["vtk_output"].start()

        # Set default output directory
        if output_directory is None:
            output_directory = os.path.join(self.output_folder, "vtk_files")
        os.makedirs(output_directory, exist_ok=True)

        if output_data is None:
            timesteps, property_array = self.output_properties(
                self.sol_filepath if sol_filepath is None else sol_filepath,
                output_properties,
                ith_step,
                engine,
            )
        else:
            timesteps, property_array = output_data[0], output_data[1]

        non_conform = (
            1
            if hasattr(self.reservoir.discretizer, "frac_cells_tot")
            and self.reservoir.discretizer.frac_cells_tot > 0
            else 0
        )

        if non_conform:
            n = int(self.reservoir.mesh.n_res_blocks)
            n_frac = int(self.reservoir.discretizer.frac_cells_tot)

            frac_property_array = {}
            to_drop = []

            for k, v in list(property_array.items()):
                arr = np.asarray(v)

                if arr.shape == (1, n_frac):
                    frac_property_array[k] = arr
                    to_drop.append(k)

                elif arr.shape == (1, n):
                    pass

                else:
                    raise ValueError(
                        f"Property '{k}' has an invalid shape {arr.shape}, "
                        f"expected (1, {n}) or (1, {n_frac}) in case of fracture-property-arrays."
                        f"WARNING: Multiple timesteps are not supported when using fracture-property-arrays."
                    )

            for k in to_drop:
                property_array.pop(k, None)
        else:
            frac_property_array = {}

        # units to prop names
        self.set_units()
        prop_names = {}
        for _i, name in enumerate(property_array.keys()):
            if name in self.properties + self.physics.vars:
                prop_names[name] = name + self.variable_units[name]
            else:
                prop_names[name] = name

        for t, time in enumerate(timesteps):
            data = np.array([property_array[name][t] for name in property_array])

            self.reservoir.frac_property_array = (
                frac_property_array  # fracture-property-array
            )

            self.reservoir.output_to_vtk(
                t if ith_step is None else ith_step,
                time,
                output_directory,
                prop_names,
                data,  # this array only contains reservoir properties
            )

        self.timer.node["vtk_output"].stop()
        self.timer.stop()

    def output_to_xarray(
        self,
        filepath: str = None,
        output_properties: list = None,
        timestep: int = None,
        engine: bool = False,
    ) -> xr.Dataset:
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
        time, data = self.output_properties(
            filepath, output_properties, timestep, engine
        )
        props = list(data.keys())

        # Initialize coords and data_vars for Xarray Dataset
        array_shape = (
            len(time),
            self.reservoir.nz,
            self.reservoir.ny,
            self.reservoir.nx,
        )
        for prop, array in data.items():
            data[prop] = array.reshape(array_shape)

        # Initialize coords and data_vars for Xarray Dataset
        if type(self.reservoir).__name__ == "StructReservoir":
            dx, dy, dz = (
                self.reservoir.global_data["dx"],
                self.reservoir.global_data["dy"],
                self.reservoir.global_data["dz"],
            )
            x = np.cumsum(dx[:, 0, 0]) - dx[0, 0, 0] * 0.5
            y = np.cumsum(dy[0, :, 0]) - dy[0, 0, 0] * 0.5
            z = np.cumsum(dz[0, 0, :]) - dz[0, 0, 0] * 0.5
        else:
            raise ValueError("Reservoir type is not supported.")

        coords = {"time": time, "z": z, "y": y, "x": x}
        data_vars = {prop: (list(coords.keys()), data[prop]) for prop in props}
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)

        # Attach units
        dataset["time"].attrs["units"] = "days"
        dataset["x"].attrs["units"] = "m"
        dataset["y"].attrs["units"] = "m"
        dataset["z"].attrs["units"] = "m"
        for var in data.keys():
            try:
                # first_part = var.split('_')[0]
                dataset[var].attrs["units"] = self.variable_units[var][1:-1]
            except:
                dataset[var].attrs["units"] = ""

        if self.precision == "d":
            encoding = {prop: {"dtype": "float64"} for prop in data.keys()}
        else:
            encoding = {prop: {"dtype": "float32"} for prop in data.keys()}

        # Save to NetCDF with specified encoding
        dataset.to_netcdf(
            os.path.join(self.output_folder, self.sol_filename[:-3] + ".nc"),
            engine="netcdf4",
            encoding=encoding,
        )

        return dataset

    def plot_xarray(
        self,
        xarray_data,
        timestep: int = -1,
        x: int = None,
        y: int = None,
        z: int = None,
    ):
        """
        :param xarray_data: xarray data set
        :param timestep: time index
        :param x: index in x-dimension
        :param y: index in y-dimension
        :param z: index in z-dimension
        """

        from darts.reservoirs.struct_reservoir import StructReservoir

        if type(self.reservoir) is not StructReservoir:
            raise AttributeError(
                "Reservoir class must be exactly of type StructReservoir."
            )

        output_directory = os.path.join(self.output_folder, "figures")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        assert isinstance(timestep, int) and timestep < len(xarray_data['time']), (
            f"Timestep should be an integer less than {len(xarray_data['time'])}."
        )

        var_names = list(xarray_data.data_vars)
        for _i, var in enumerate(var_names):
            plt.figure()
            if z is not None:
                assert z < len(xarray_data['z']), (
                    f"z-level step should be less than {len(xarray_data['z']):d}"
                )
                xarray_data[var].isel(time=timestep, z=z).plot()
                plt.savefig(output_directory + f'/{var} ts{timestep:d} z{z:d}.png')

            elif y is not None:
                assert y < len(xarray_data['y']), (
                    f"y-level step should be less than {len(xarray_data['y']):d}"
                )
                xarray_data[var].isel(time=timestep, y=y).plot()
                plt.savefig(output_directory + f'/{var} ts{timestep:d} y{y:d}.png')

            elif x is not None:
                assert x < len(xarray_data['x']), (
                    f"x-level step should be less than {len(xarray_data['x']):d}"
                )
                xarray_data[var].isel(time=timestep, x=x).plot()
                plt.savefig(
                    output_directory
                    + f'/{var} ts{timestep:d} zx{z if z is not None else 0:d}.png'
                )

            else:
                # model is a 1D reservoir
                xarray_data[var].isel(time=timestep).plot()
                plt.savefig(output_directory + f'/{var} ts{timestep:d}.png')
        plt.close('all')

    def store_well_time_data(
        self, types_of_well_rates: list = None, save_output_files: bool = False
    ):
        """
        Compute and store well time data including rates and bottom-hole conditions (BHT and BHP)

        Rates are calculated for each perforation and also total rate of each well. Total rates are calculated using
        two different methods:
        1- summing up the rates of perforations
        2- calculating the rates directly at the wellhead connection

        :param types_of_well_rates: List of types of well rates that can be computed:
                                    "phase_molar_rates"
                                    "phase_mass_rates"
                                    "phase_volumetric_rates"
                                    "component_molar_rates"
                                    "component_mass_rates"
                                    "advective_heat_rates" for thermal scenarios
        :type types_of_well_rates: list
        :param save_output_files: Flag to save time_data as a .pkl and .xlsx file in the output folder, default false
        :type save_output_files: bool
        """
        # Start timer for store_well_time_data
        self.timer.start()
        self.timer.node["output_well_time_data"].start()

        h5_well_data = load_hdf5_to_dict(self.well_filepath)
        self.configure_physics()

        time = h5_well_data["dynamic"]["time"]
        time_data_dict = {"time": time}

        (
            perfs_conn_ids,
            well_head_conn_ids,
            geometric_WI,
            well_head_conn_trans,
        ) = self.get_wellhead_perf_connection_info()

        if types_of_well_rates is None:
            types_of_well_rates = [
                "phase_molar_rates",
                "phase_mass_rates",
                "phase_volumetric_rates",
                "component_molar_rates",
                "component_mass_rates",
            ]
            if self.physics.thermal:
                types_of_well_rates.append("advective_heat_rates")

        # Store BHP and BHT
        self.store_bhp_bht(h5_well_data, time_data_dict)

        for rate_type in types_of_well_rates:
            if (
                rate_type == "component_molar_rates"
                or rate_type == "component_mass_rates"
            ) and self.physics.property_containers[
                0
            ].physics_type == "geothermal_engine":
                continue
            # Compute perforation rates
            rates_perfs = self.calc_rates_at_connections(
                h5_well_data,
                perfs_conn_ids,
                geometric_WI,
                self.physics.thermal,
                rate_type,
            )
            # Store perforation rates
            self.store_perf_rates(time_data_dict, rates_perfs, rate_type)
            # Store well rates by summing perforation rates
            self.store_well_rates_sums(time_data_dict, rates_perfs, rate_type)
            # Compute wellhead rates
            rates_wellhead = self.calc_rates_at_connections(
                h5_well_data,
                well_head_conn_ids,
                well_head_conn_trans,
                self.physics.thermal,
                rate_type,
            )
            # Store wellhead rates
            self.store_wellhead_rates(time_data_dict, rates_wellhead, rate_type)

        # Export time_data_dict
        if save_output_files:
            df = pd.DataFrame(time_data_dict)
            df.to_pickle(os.path.join(self.output_folder, "well_time_data.pkl"))
            with pd.ExcelWriter(
                os.path.join(self.output_folder, "well_time_data.xlsx")
            ) as w:
                df.to_excel(w, sheet_name="Sheet1")

        # End timer for store_well_time_data
        self.timer.node["output_well_time_data"].stop()
        self.timer.stop()
        return time_data_dict

    def configure_physics(self):
        """
        This function makes the physics of the geothermal engine compatible with how the physics of the super engine
        is defined. This function is used in the method store_well_time_data of the current class.
        """
        pc = self.physics.property_containers[0]
        pc.physics_type = "super_engine"
        physics_name = type(self.physics).__name__
        if physics_name in ("Geothermal", "GeothermalPH"):
            pc.physics_type = "geothermal_engine"
            pc.phases_name = self.physics.phases[: pc.nph]
            pc.nc_fl = 1
            pc.components_name = ["H2O"]
            self.physics.thermal = True

    def get_wellhead_perf_connection_info(self):
        """
        This function gives information of the connections, including perforations and wellhead, for evaluation of
        perforation and wellhead rates in the method store_well_time_data of the current class.
        """
        block_m = np.array(self.reservoir.mesh.block_m, copy=False)
        block_p = np.array(self.reservoir.mesh.block_p, copy=False)
        # Create a dictionary containing connection indices of perforations for each well (values are lists)
        well_perf_conn_ids = {}
        # Create a dictionary containing connection index of wellhead for each well (values are integers)
        well_head_conn_id = {}
        for idx, well in enumerate(self.reservoir.wells):
            res_cell_ids = [perf[1] for perf in well.perforations]

            # Find ids of those connections which 1. block_p is in res_cell_ids, 2. block_m is in the desired well
            if idx + 1 < len(self.reservoir.wells):  # If there is a next well
                next_well = self.reservoir.wells[idx + 1]
                mask = np.logical_and(
                    np.isin(block_p, res_cell_ids),
                    np.logical_and(
                        block_m >= well.well_head_idx, block_m < next_well.well_head_idx
                    ),
                )
            else:  # If there is no next well
                mask = np.logical_and(
                    np.isin(block_p, res_cell_ids), block_m >= well.well_head_idx
                )

            conn_ids = np.nonzero(mask)
            well_perf_conn_ids[well.name] = conn_ids[0]
            assert (
                well_perf_conn_ids[well.name].size == len(well.perforations)
                and (
                    block_m[well_perf_conn_ids[well.name]]
                    > self.reservoir.mesh.n_res_blocks
                ).all()
            )

            # Find id of well_head -> well_body connection in the connection list
            wh_conn_id = np.where(
                np.logical_and(
                    block_m == well.well_head_idx, block_p == well.well_body_idx
                )
            )[0]
            assert len(wh_conn_id) == 1
            well_head_conn_id[well.name] = wh_conn_id[0]

        # Get perforation connection indices for all wells
        perfs_conn_ids = [
            item for sublist in well_perf_conn_ids.values() for item in sublist
        ]
        # Get wellhead connection indices for all wells
        well_head_conn_ids = list(well_head_conn_id.values())

        # Get well indices (WI) for each perforation
        geometric_WI = np.array(
            [p[2] for well in self.reservoir.wells for p in well.perforations]
        )
        # Get transmissibility for each wellhead connection
        well_head_conn_trans = np.array(
            [well.well_transmissibility for well in self.reservoir.wells]
        )

        return perfs_conn_ids, well_head_conn_ids, geometric_WI, well_head_conn_trans

    def store_perf_rates(
        self, time_data_dict: dict, rates_perfs: np.ndarray, rate_type: str
    ):
        """
        This function stores perforation rates from the 3D numpy array rates_perfs for the rate type rate_type in the
        dict time_data_dict. This function is used in the method store_well_time_data of the current class.

        :param time_data_dict: Dictionary in which well time series will be stored
        :type time_data_dict: dict
        :param rates_perfs: A 3D numpy array in which perforation rates are stored for different time steps,
        perforations, and phases or components.
        :type rates_perfs: np.ndarray
        :param rate_type: Type of the well rate
        :type rate_type: str
        """
        pc = self.physics.property_containers[0]
        total_perf_idx = 0
        for well in self.reservoir.wells:
            for perf_idx in range(len(well.perforations)):
                tag = f"well_{well.name}_perf_{perf_idx}"
                if rate_type.startswith("phase_"):
                    for phase_idx, phase_name in enumerate(pc.phases_name):
                        arr = rates_perfs[:, total_perf_idx, phase_idx]
                        time_data_dict[
                            f'{tag}_{rate_type.split("_")[1]}_rate_{phase_name}'
                        ] = arr
                elif rate_type.startswith("component_"):
                    for c_idx in range(pc.nc_fl):
                        arr = np.sum(
                            rates_perfs[:, total_perf_idx, c_idx :: pc.nc_fl], axis=1
                        )
                        time_data_dict[
                            f'{tag}_{rate_type.split("_")[1]}_rate_{pc.components_name[c_idx]}'
                        ] = arr
                elif rate_type.startswith("advective_heat_"):
                    for phase_idx, phase_name in enumerate(pc.phases_name):
                        arr = rates_perfs[:, total_perf_idx, phase_idx]
                        time_data_dict[f"{tag}_advective_heat_rate_{phase_name}"] = arr
                total_perf_idx += 1

    def store_well_rates_sums(
        self, time_data_dict: dict, rates_perfs: np.ndarray, rate_type: str
    ):
        """
        This function stores summation of perforation rates for each well from the 3D numpy array rates_perfs for the
        rate type rate_type in the dict time_data_dict. This function is used in the method store_well_time_data of
        the current class.

        :param time_data_dict: Dictionary in which well time series will be stored
        :type time_data_dict: dict
        :param rates_perfs: A 3D numpy array in which perforation rates are stored for different time steps,
        perforations, and phases or components.
        :type rates_perfs: np.ndarray
        :param rate_type: Type of the well rate
        :type rate_type: str
        """
        pc = self.physics.property_containers[0]
        total_perf_idx = 0
        for well in self.reservoir.wells:
            tag = f"well_{well.name}"
            if rate_type.startswith("phase_"):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    total = sum(
                        rates_perfs[:, total_perf_idx + j, phase_idx]
                        for j in range(len(well.perforations))
                    )
                    time_data_dict[
                        f'{tag}_{rate_type.split("_")[1]}_rate_{phase_name}_by_sum_perfs'
                    ] = total
                total_perf_idx += len(well.perforations)
            elif rate_type.startswith("component_"):
                for c_idx in range(pc.nc_fl):
                    total = sum(
                        np.sum(
                            rates_perfs[:, total_perf_idx + j, c_idx :: pc.nc_fl],
                            axis=1,
                        )
                        for j in range(len(well.perforations))
                    )
                    time_data_dict[
                        f'{tag}_{rate_type.split("_")[1]}_rate_{pc.components_name[c_idx]}_by_sum_perfs'
                    ] = total
                total_perf_idx += len(well.perforations)
            elif rate_type.startswith("advective_heat_"):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    total = sum(
                        rates_perfs[:, total_perf_idx + j, phase_idx]
                        for j in range(len(well.perforations))
                    )
                    time_data_dict[
                        f"{tag}_advective_heat_rate_{phase_name}_by_sum_perfs"
                    ] = total
                total_perf_idx += len(well.perforations)

    def store_wellhead_rates(
        self, time_data_dict: dict, wh_rates: np.ndarray, rate_type: str
    ):
        """
        This function stores wellhead rate for each well from the 3D numpy array rates_perfs for the rate type
        rate_type in the dict time_data_dict. This function is used in the method store_well_time_data of the
        current class.

        :param time_data_dict: Dictionary in which well time series will be stored
        :type time_data_dict: dict
        :param wh_rates: A 3D numpy array in which wellhead rates are stored for different time steps,
        wellheads, and phases or components.
        :type wh_rates: np.ndarray
        :param rate_type: Type of the well rate
        :type rate_type: str
        """
        pc = self.physics.property_containers[0]
        for well_idx, well in enumerate(self.reservoir.wells):
            tag = f"well_{well.name}"
            if rate_type.startswith("phase_"):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    time_data_dict[
                        f'{tag}_{rate_type.split("_")[1]}_rate_{phase_name}_at_wh'
                    ] = wh_rates[:, well_idx, phase_idx]
            elif rate_type.startswith("component_"):
                for c_idx, c_name in enumerate(pc.components_name):
                    arr = np.sum(wh_rates[:, well_idx, c_idx :: pc.nc_fl], axis=1)
                    time_data_dict[
                        f'{tag}_{rate_type.split("_")[1]}_rate_{c_name}_at_wh'
                    ] = arr
            elif rate_type.startswith("advective_heat_"):
                for phase_idx, phase_name in enumerate(pc.phases_name):
                    time_data_dict[f"{tag}_advective_heat_rate_{phase_name}_at_wh"] = (
                        wh_rates[:, well_idx, phase_idx]
                    )

    def store_bhp_bht(self, h5_well_data: dict, time_data_dict: dict):
        """
        This function stores bottom-hole pressure (BHP) and temperature (BHT) of wells over time in time_data_dict.
        This function is used in the method store_well_time_data of the current class.

        :param h5_well_data: Dictionary extracted from the HDF5 file that stores well primary variables, etc.
        :type h5_well_data: dict
        :param time_data_dict: Dictionary in which well time series will be stored
        :type time_data_dict: dict
        """
        nt = len(h5_well_data["dynamic"]["time"])
        cell_id = h5_well_data["dynamic"]["cell_id"]
        variable_names = h5_well_data["dynamic"]["variable_names"]
        X = h5_well_data["dynamic"]["X"]
        pc = self.physics.property_containers[0]

        for well in self.reservoir.wells:
            BHP = np.zeros(nt)
            BHT = np.zeros(nt) if self.physics.thermal else np.full(nt, pc.temperature)
            wellhead_cell_idx = self.find_values_in_an_array(
                [well.well_head_idx], cell_id
            )
            p_idx = variable_names.index("pressure")
            for i in range(nt):
                p = X[i, :, p_idx]
                BHP[i] = p[wellhead_cell_idx]
                if self.physics.thermal:
                    if "temperature" in variable_names:
                        t_idx = variable_names.index("temperature")
                        BHT[i] = X[i, :, t_idx][wellhead_cell_idx]
                    else:
                        h_idx = variable_names.index("enthalpy")
                        BHT[i] = pc.temperature_ev.evaluate(
                            [BHP[i], X[i, wellhead_cell_idx, h_idx]]
                        )
            time_data_dict[f"well_{well.name}_BHP"] = BHP
            time_data_dict[f"well_{well.name}_BHT"] = BHT

    def calc_rates_at_connections(
        self,
        h5_well_data: dict,
        conn_ids: list,
        trans: np.ndarray,
        thermal: bool,
        rate_type: str,
    ):
        """
        This function calculates different types of rates at perforations or wellhead connections of wells.
        This function is used in the method store_well_time_data of the current class.

        :param h5_well_data: Well data stored in the HDF5 file
        :type h5_well_data: dict
        :param conn_ids: IDs of connections
        :type conn_ids: list
        :param trans: Transmissibility (For perforations, it is geometric part of well index)
        :type trans: np.ndarray
        :param thermal: If the model is thermal or not
        :type thermal: bool
        :param rate_type: Type of well rate to calculate
        :type rate_type: str
        """
        # Evaluate position of block_m, block_p in stored data, for every connection
        block_m = h5_well_data["static"]["block_m"]
        block_p = h5_well_data["static"]["block_p"]
        cell_id = h5_well_data["dynamic"]["cell_id"]
        cell_m = self.find_values_in_an_array(block_m[conn_ids], cell_id)  # well cells
        cell_p = self.find_values_in_an_array(
            block_p[conn_ids], cell_id
        )  # reservoir cells
        num_conn = len(conn_ids)
        assert cell_m.size == num_conn and cell_p.size == num_conn

        num_ts = h5_well_data["dynamic"]["time"].size

        pc = self.physics.property_containers[0]

        p_idx = h5_well_data["dynamic"]["variable_names"].index("pressure")
        if thermal:
            if self.physics.state_spec == self.physics.StateSpecification.PT:
                t_idx = h5_well_data["dynamic"]["variable_names"].index(
                    "temperature"
                )  # This does not work for geothermal engine
            elif self.physics.state_spec == self.physics.StateSpecification.PH:
                pass
            else:
                raise Exception(
                    "Neither temperature nor enthalpy exists in the list of variables!"
                )
        elif not thermal and rate_type == "advective_heat_rates":
            raise Exception(
                "The model is isothermal, so advective heat rate cannot be calculated for it!"
            )

        p = h5_well_data["dynamic"]["X"][:, :, p_idx]

        dp = p[:, cell_p] - p[:, cell_m]

        id_upwind = np.where(dp < 0, cell_m, cell_p)

        # This adds a new axis, turning a 1D array into a 2D column vector
        time_idx = np.arange(num_ts)[:, None]

        states = h5_well_data["dynamic"]["X"][time_idx, id_upwind]

        if self.precision == "s":
            states = np.clip(
                states,
                self.physics.axes_min[None, None, :],
                self.physics.axes_max[None, None, :],
            )

        batch_size = num_ts * num_conn
        flat_states = states.reshape(batch_size, self.physics.n_vars)

        states_vec = value_vector(flat_states.ravel())

        if rate_type in [
            "phase_molar_rates",
            "phase_mass_rates",
            "phase_volumetric_rates",
            "advective_heat_rates",
        ]:
            values = value_vector(
                np.zeros(num_ts * num_conn * self.physics.well_ctrl_operators.n_ops)
            )
            dvalues = value_vector(
                np.zeros(
                    (num_ts * num_conn * self.physics.well_ctrl_operators.n_ops)
                    * self.physics.n_vars
                )
            )

            block_idx = np.arange(num_ts * num_conn).astype(np.int32)
            self.physics.well_ctrl_itor.evaluate_with_derivatives(
                states_vec, index_vector(block_idx), values, dvalues
            )

            # self.physics.well_ctrl_itor.evaluate(states_vec, values)

            values_reshaped = np.asarray(values).reshape(
                batch_size, self.physics.well_ctrl_operators.n_ops
            )

        elif rate_type in ["component_molar_rates", "component_mass_rates"]:
            values = value_vector(
                np.zeros(num_ts * num_conn * self.physics.reservoir_operators[0].n_ops)
            )
            dvalues = value_vector(
                np.zeros(
                    (num_ts * num_conn * self.physics.reservoir_operators[0].n_ops)
                    * self.physics.n_vars
                )
            )

            block_idx = np.arange(num_ts * num_conn).astype(np.int32)
            self.physics.acc_flux_itor[0].evaluate_with_derivatives(
                states_vec, index_vector(block_idx), values, dvalues
            )

            values_reshaped = np.asarray(values).reshape(
                batch_size, self.physics.reservoir_operators[0].n_ops
            )

        else:
            raise Exception(
                "The rate type is not entered correctly or is not supported!"
            )

        if rate_type == "phase_molar_rates":
            start = int(well_control_iface.MOLAR_RATE) * pc.nph
            ops = values_reshaped[:, start : start + pc.nph]
        elif rate_type == "phase_mass_rates":
            start = int(well_control_iface.MASS_RATE) * pc.nph
            ops = values_reshaped[:, start : start + pc.nph]
        elif rate_type == "phase_volumetric_rates":
            op_start = int(well_control_iface.VOLUMETRIC_RATE) * pc.nph
            ops = values_reshaped[:, op_start : op_start + pc.nph]
        elif rate_type == "component_molar_rates":
            op_start = self.physics.reservoir_operators[0].FLUX_OP
            ops = values_reshaped[
                :, op_start : op_start + pc.nc_fl * pc.nph
            ]  # molar ops
        elif rate_type == "component_mass_rates":
            op_start = self.physics.reservoir_operators[0].FLUX_OP
            molar_ops = values_reshaped[:, op_start : op_start + pc.nc_fl * pc.nph]
            mw = np.array(self.physics.property_containers[0].Mw[: pc.nc_fl])
            mw_tiled = np.tile(mw, pc.nph)
            ops = molar_ops * mw_tiled
        elif rate_type == "advective_heat_rates":
            op_start = int(well_control_iface.ADVECTIVE_HEAT_RATE) * pc.nph
            ops = values_reshaped[:, op_start : op_start + pc.nph]

            # Calc heat operators for the dead state (1 atm and 15 deg C)
            if self.physics.state_spec == self.physics.StateSpecification.PT:
                flat_states[:, p_idx] = 1.01325  # Dead pressure (1 atm)
                flat_states[:, t_idx] = 273.15 + 15  # Dead temperature (15 deg C)
                states_vec_dead = value_vector(flat_states.ravel())
                self.physics.well_ctrl_itor.evaluate_with_derivatives(
                    states_vec_dead, index_vector(block_idx), values, dvalues
                )
                op_start = int(well_control_iface.ADVECTIVE_HEAT_RATE) * pc.nph
                values_reshaped_dead = np.asarray(values).reshape(
                    batch_size, self.physics.well_ctrl_operators.n_ops
                )
                ops_dead = values_reshaped_dead[:, op_start : op_start + pc.nph]
            elif self.physics.state_spec == self.physics.StateSpecification.PH:
                # TODO This does not work properly if the super engine is of the PH type
                enthalpy_w, dens_m_w, kr_w, miu_w = (
                    -44582.229072,
                    55.457385,
                    1,
                    1.132781,
                )  # Water properties under dead conditions (1 atm, 15 deg C, and zH2O = 1)
                ops_dead_phase = enthalpy_w * dens_m_w * kr_w / miu_w
                ops_dead = np.zeros(ops.shape)
                ops_dead[ops != 0.0] = (
                    ops_dead_phase  # If value is zero, no need to subtract ops_dead_phase from it
                )

            ops = ops - ops_dead

        # Reshape arrays
        if rate_type in [
            "phase_molar_rates",
            "phase_mass_rates",
            "phase_volumetric_rates",
        ]:
            ops_reshaped = ops.reshape(num_ts, num_conn, pc.nph)
        elif rate_type in ["component_molar_rates", "component_mass_rates"]:
            ops_reshaped = ops.reshape(num_ts, num_conn, -1)
        elif rate_type == "advective_heat_rates":
            ops_reshaped = ops.reshape(num_ts, num_conn, pc.nph)

        trans_exp = trans[None, :, None]
        dp_exp = dp[:, :, None]
        rates = -ops_reshaped * trans_exp * dp_exp

        return rates

    # %% Auxiliary functions
    def find_conn_ids_for_perfs(
        self, perfs: list, block_m: np.ndarray, block_p: np.ndarray, n_res_blocks: int
    ):
        """
        This function finds the connection IDs of perforations

        :param perfs: List of perforations (well_block_index, reservoir_block_index, well_index, well_indexD)
        :type perfs: List
        :param block_m: block_m of the connection list
        :type block_m: np.ndarray
        :param block_p: block_p of the connection list
        :type block_p: np.ndarray
        :param n_res_blocks: Number of reservoir blocks
        :type n_res_blocks: int
        """
        res_cell_ids = [perf[1] for perf in perfs]
        perfs_conn_ids = np.nonzero(
            np.logical_and(np.isin(block_p, res_cell_ids), block_m >= n_res_blocks)
        )[0]
        assert (
            len(perfs_conn_ids) == len(perfs)
            and (block_m[perfs_conn_ids] > n_res_blocks).all()
        )
        return perfs_conn_ids

    def find_values_in_an_array(self, to_find: np.ndarray | list, in_array: np.ndarray):
        """
        :param to_find: The values the indices of which we want to find in in_array
        :type to_find: np.ndarray or list
        :param in_array: The array in which we want to find the values in to_find
        :type in_array: np.ndarray
        """
        indices = []
        for element in to_find:
            id = np.where(in_array == element)[0]
            if id.size > 0:
                indices.append(id[0])
        return np.array(indices, dtype=np.intp)

    def plot_well_time_data(self, types_of_well_rates: list = None):
        """
        Plots well time data that are specified in the list types_of_well_time_data over time, including
        phase_molar_rates, phase_mass_rates, phase_volumetric_rates, component_molar_rates, component_mass_rates,
        advective_heat_rates, BHP (bottom-hole pressure), and BHT (bottom-hole temperature)

        :param types_of_well_rates: List of types of well rates that can be computed:
                                    "phase_molar_rates"
                                    "phase_mass_rates"
                                    "phase_volumetric_rates"
                                    "component_molar_rates"
                                    "component_mass_rates"
                                    "advective_heat_rates" for thermal scenarios
        :type types_of_well_rates: list
        """
        main_dir = os.path.join(self.output_folder, "figures/well_time_plots")

        # Reset_directory
        if os.path.exists(main_dir):
            shutil.rmtree(main_dir)
        os.makedirs(main_dir)

        self.create_perf_dirs(main_dir)

        df = pd.read_pickle(os.path.join(self.output_folder, "well_time_data.pkl"))
        time = df["time"]

        # Specify types of well rates that will be plotted if types_of_well_rates is not entered by the user
        if types_of_well_rates is None:
            types_of_well_rates = [
                "phase_molar_rates",
                "phase_mass_rates",
                "phase_volumetric_rates",
                "component_molar_rates",
                "component_mass_rates",
            ]
            if self.physics.thermal:
                types_of_well_rates.append("advective_heat_rates")

        self.unit_dict = {
            "molar": "kmol/day",
            "mass": "kg/day",
            "volumetric": "m^3/day",
            "heat": "kJ/day",
        }

        for rtype in types_of_well_rates:
            for well in self.reservoir.wells:
                well_dir = os.path.join(main_dir, f"well_{well.name}")
                for perf_idx in range(len(well.perforations)):
                    subdir = os.path.join(well_dir, f"perf_{perf_idx}")
                    keys = self.create_perf_keys(rtype, well.name, perf_idx)
                    for key, ylabel in keys:
                        if key not in df.keys():
                            continue
                        arr = df[key]
                        plt.figure()
                        plt.plot(time, arr, marker="o")
                        plt.xlabel("Time [day]")
                        plt.ylabel(ylabel)
                        plt.tight_layout()
                        plt.savefig(os.path.join(subdir, f"{key}.png"))
                        plt.close()
                # total and wellhead plots
                total_keys = self.create_total_keys(rtype, well.name)
                for key, ylabel in total_keys:
                    if key not in df.keys():
                        continue
                    plt.figure()
                    plt.plot(time, df[key], marker="o")
                    plt.xlabel("Time [day]")
                    plt.ylabel(ylabel)
                    plt.tight_layout()
                    plt.savefig(os.path.join(well_dir, f"{key}.png"))
                    plt.close()

        # BHP and BHT are plotted all the time
        for well in self.reservoir.wells:
            well_dir = os.path.join(main_dir, f"well_{well.name}")

            BHP_key = f"well_{well.name}_BHP"
            BHP = df[BHP_key]

            plt.figure()
            plt.plot(time, BHP, marker="o")
            plt.xlabel("Time [day]")
            plt.ylabel("Bottom-hole pressure [bar]")
            plt.tight_layout()
            plt.savefig(os.path.join(well_dir, f"{BHP_key}.png"))
            plt.close()

            BHT_key = f"well_{well.name}_BHT"
            BHT = df[BHT_key]

            plt.figure()
            plt.plot(time, BHT, marker="o")
            plt.xlabel("Time [day]")
            plt.ylabel("Bottom-hole temperature [K]")
            plt.tight_layout()
            plt.savefig(os.path.join(well_dir, f"{BHT_key}.png"))
            plt.close()

        return df

    def create_perf_dirs(self, main_dir: str):
        """
        This function creates a new directory (folder) for each perforation of each well. The rates for each perforation
        will be stored in their corresponding directory later. This function is used in the method plot_well_time_data
        of the current class.

        :param main_dir: Directory in which a folder for each well already exists or will be created. Folder
        for each perforation will be created in the corresponding well folder.
        :type main_dir: str
        """
        for well in self.reservoir.wells:
            well_dir = os.path.join(main_dir, f"well_{well.name}")
            os.makedirs(well_dir, exist_ok=True)
            for perf_idx in range(len(well.perforations)):
                os.makedirs(os.path.join(well_dir, f"perf_{perf_idx}"), exist_ok=True)

    def create_perf_keys(self, rtype: str, well_name: str, perf_idx: int):
        """
        This function creates keys for perforation rates. This function is used in the method plot_well_time_data
        of the current class.

        :param rtype: Type of the well rate
        :type rtype: str
        :param well_name: Name of the well
        :type well_name: str
        :param perf_idx: Index of the perforation. This index starts from zero and the order depends on the order
        at which perforations are added to the wellbore using the add_perforation method.
        :type perf_idx: int
        """
        pc = self.physics.property_containers[0]
        keys = []
        tag = f"well_{well_name}_perf_{perf_idx}_"
        rate_type = rtype.split("_")[1]
        unit = self.unit_dict[rate_type]
        if rtype.startswith("phase_"):
            for phase_name in pc.phases_name:
                key = f"{tag}{rate_type}_rate_{phase_name}"
                ylabel = f"{phase_name} {rate_type} rate [{unit}]"
                keys.append((key, ylabel))
        elif rtype.startswith("component_"):
            for component_name in pc.components_name:
                key = f"{tag}{rate_type}_rate_{component_name}"
                ylabel = f"{component_name} {rate_type} rate [{unit}]"
                keys.append((key, ylabel))
        elif rtype.startswith("advective_heat_"):
            for phase_name in pc.phases_name:
                key = f"{tag}advective_heat_rate_{phase_name}"
                ylabel = f"{phase_name} advective {rate_type} rate [{unit}]"
                keys.append((key, ylabel))
        return keys

    def create_total_keys(self, rtype: str, well_name: str):
        """
        This function creates keys for summation rates, wellhead rates, and BHP and BHT. This function is used in the
        method plot_well_time_data of the current class.

        :param rtype: Type of well rate
        :type rtype: str
        :param well_name: Name of the well
        :type well_name: str
        """
        pc = self.physics.property_containers[0]
        keys = []
        base = f"well_{well_name}_"
        rate_type = rtype.split("_")[1]
        unit = self.unit_dict[rate_type]
        if rtype.startswith("phase_"):
            for phase_name in pc.phases_name:
                keys.extend(
                    [
                        (
                            f"{base}{rate_type}_rate_{phase_name}_by_sum_perfs",
                            f"{phase_name} {rate_type} rate [{unit}]",
                        ),
                        (
                            f"{base}{rate_type}_rate_{phase_name}_at_wh",
                            f"{phase_name} {rate_type} rate [{unit}]",
                        ),
                    ]
                )
        elif rtype.startswith("component_"):
            for component_name in pc.components_name:
                keys.extend(
                    [
                        (
                            f"{base}{rate_type}_rate_{component_name}_by_sum_perfs",
                            f"{component_name} {rate_type} rate [{unit}]",
                        ),
                        (
                            f"{base}{rate_type}_rate_{component_name}_at_wh",
                            f"{component_name} {rate_type} rate [{unit}]",
                        ),
                    ]
                )
        elif rtype.startswith("advective_heat_"):
            for phase_name in pc.phases_name:
                keys.extend(
                    [
                        (
                            f"{base}advective_heat_rate_{phase_name}_by_sum_perfs",
                            f"{phase_name} advective {rate_type} rate [{unit}]",
                        ),
                        (
                            f"{base}advective_heat_rate_{phase_name}_at_wh",
                            f"{phase_name} advective {rate_type} rate [{unit}]",
                        ),
                    ]
                )
        elif rtype in ("BHP", "BHT"):
            label = (
                "Bottom-hole pressure [bar]"
                if rtype == "BHP"
                else "Bottom-hole temperature [K]"
            )
            keys.append((f"{base}{rtype}", label))
        return keys
