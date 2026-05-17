import os
import shutil
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vtk
import xarray as xr
from vtk.util.numpy_support import numpy_to_vtk

from darts.engines import (
    index_vector,
    timer_node,
    value_vector,
    well_control_iface,
)
from darts.physics.base.operators_base import PropertyOperators
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.geothermal.physics import Geothermal
from darts.physics.super.physics import Compositional
from darts.tools.hdf5_tools import load_hdf5_to_dict


class Output:
    """
    This class handles simulation output including primary variables, secondary variables,
    well reporting and visualizations (pyplots, .vtk files). All simulation output is saved
    into HDF5 files. To view the contents of these HDF5 files users are recommended to use an HDF5 viewer.
    Alternatively, primary and secondary variables can also be processed into xarray format.

    * **Primary variables** (state/unknowns) for reservoir blocks and well blocks are written
      to an HDF5 file named by default ``...\\reservoir_solution.h5`` and ``...\\well_data.h5``.
    * **Secondary variables** ("properties"; e.g., saturations, densities, ...) are evaluated
      using the DARTS interpolator and either stored in a separate file or appended to the
      ``reservoir_solution.h5`` file under the dedicated group ``/properties``.
    * **Well output** saved data in ``...\\well_data.h5`` is processed with the function ``store_well_time_data()``
      for well rates. The resulting time series data is saved as an ``.xlsx`` file or ``*.pkl`` file.

    Notes
    -----
    * The storage precision, compression algorithm, and compression level are configurable.
      For large-scale runs, compression and precision have a significant impact on both
      write performance and file size.
    * You can find a tutorial on how to use the various output-related functionalities
      `here <https://gitlab.com/open-darts/open-darts/-/blob/main/tutorials/output_and_restart.py?ref_type=heads>`_.
    * The key naming formats for the rates stored in the ``time_data`` dictionary can be
      found `here <https://open-darts.gitlab.io/open-darts/technical_reference/wells.html>`_.
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
        compression_level: int,
        verbose: bool,
        wells: dict = None,
        has_dfm_well: bool = False,
    ):
        """
        :param timer: timer object, measurs time spent saving data, and evaluating properties.
        :param reservoir: reservoir object.
        :param physics: physics object.
        :param wells: dict of well objects if the DFM well is used
        :param op_list: list of operator interpolators.
        :param params: engine params.
        :param output_folder: output folder for saved data and figures.
        :param sol_filename: hdf5 filename for saving reservoir solution.
        :param well_filename: hdf5 filename for saving well solution.
        :param save_initial: boolean flag to save initial conditions of reservoir.
        :param all_phase_props: boolean flag to enable evaluation of phase properties according to a predefined list.
        :param precision: data precision of saved data ('s' single precision, 'd' double precision).
        :param compression: default 'gzip'.
        :param compression_level: 0 (no compression, fast) and 9 (maximum compression, slow), default is 1.
        :param verbose: boolean flag to enable verbose output.
        """
        super().__init__()

        self.reservoir = reservoir
        self.physics = physics
        self.op_list = op_list
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.wells = wells
        self.has_dfm_well = has_dfm_well

        self.params = params
        self.verbose = verbose

        self.master_timer = timer
        self.timer = timer.node["output"]
        self.timer.node["saving_reservoir_data"] = timer_node()
        self.timer.node["saving_well_data"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()
        self.timer.node["vtp_output"] = timer_node()
        self.timer.node["output_well_time_data"] = timer_node()
        self.timer.node["exporting_property_array"] = timer_node()

        self.output_folder = output_folder
        self.sol_filename = sol_filename
        self.well_filename = well_filename
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        self.precision = precision
        self.compression = compression
        self.compression_level = compression_level
        self.precision_map = {"d": np.float64, "s": np.float32}

        self.thermal = self.physics.state_spec >= PhysicsBase.StateSpecification.PT
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
            "pc": "[bar]",
            "pressure": "[bar]",
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

        Notes
        -----
        * The properties for the super engine class include phase properties (density, molar density, saturation, viscosity, relative permeability, capillary pressure, enthalpy and conductivity) and molar phase fractions.
        * The properties for the geothermal engine class include phase properties (density, molar density, saturation, viscosity, relative permeability, capillary pressure, enthalpy) and temperature.
        * The declared interpolator is adaptive multilinear.
        """

        if isinstance(self.physics, Compositional):
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
                    property_container=pc,
                    thermal=self.thermal,
                    props=temp_dict,
                    extrapolation_flag=self.physics.extrapolation_flag,
                    dz=self.physics.dz,
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

        elif isinstance(self.physics, Geothermal):
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
                    property_container=pc,
                    thermal=False,
                    props=temp_dict,
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

    def filter_phase_props(self, new_prop_keys: list):
        """
        Filter default list of properties to only evaluate desired properties listed in new_prop_keys.

        :param new_prop_keys: list of properties to keep
        :type
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
                property_container=self.physics.property_containers[region],
                thermal=self.thermal,
                props=output_dictionary,
                extrapolation_flag=self.physics.extrapolation_flag,
                dz=self.physics.dz,
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

    def save_array(self, array: dict, filename: str, compression_level: int = 1):
        """
        This function saves any dictionary as an HDF5 file with compression.

        :param array: data.
        :param filename: Name of the output file. The file will be created inside `self.output_folder`.
        :param compression_level: 0 (no compression) and 9 (maximum compression), default is 1.

        Notes
        -----
        * Each key in the input dictionary is written as a separate dataset at the root level of the HDF5 file.
        * Existing files with the same name will be overwritten.

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

    def load_array(self, filepath: str):
        """
        This function loads any saved data in h5 file format.

        :param filepath: Path to the HDF5 file to load (typically ending in ``.h5``).
        :return array: Dictionary mapping dataset names to NumPy arrays.

        Notes
        -----
        * All datasets located at the root level of the HDF5 file are read and returned as NumPy arrays.
        * Dataset names are used as dictionary keys.
        """
        array = {}
        with h5py.File(filepath, "r") as h5f:
            for key in h5f.keys():
                array[key] = np.array(h5f[key])
        return array

    def append_properties_to_reservoir(
        self, time: float, property_array: dict, compression_level: int = 1
    ):
        """
        Append per-cell secondary properties into an existing HDF5 solution file.

        :param time: Simulation time to write properties for.
        :param property_array: Dictionary with property names as keys and arrays (1D over cells) as values.
        :param compression_level: 0 (no compression) and 9 (maximum compression), default is 1.

        Notes
        -----
        * Writes properties into the root-level HDF5 group ``/properties`` using a time-indexed 2D layout: ``(n_timesteps, n_cells)``.
        * The time index is determined by matching the provided ``time`` value against ``/dynamic/time``.
        """

        with h5py.File(self.sol_filepath, "a") as f:
            time_vector = f["dynamic/time"][:]
            if time in time_vector:
                ts_idx = int(np.where(time_vector == time)[0][0])

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
                            compression_opts=compression_level,
                        )

                    else:
                        dset = prop_group[key]
                        if len(data) != dset.shape[1]:
                            raise ValueError(
                                f"Shape mismatch for property '{key}': expected {dset.shape[1]}, got {len(data)}"
                            )
                        if ts_idx >= dset.shape[0]:
                            dset.resize((ts_idx + 1, dset.shape[1]))

                    dset[ts_idx, :] = data

            else:
                raise ValueError(
                    f"Timestamp {time} does not exist in the solution.h5 file."
                )

    def save_property_array(
        self,
        time_vector: np.ndarray,
        property_array: dict,
        filename=None,
        compression_level: int = 1,
    ):
        """
        Function to save property_darry dictionary to HDF5 file.

        :param time_vector: Array of timesteps
        :param property_array: Dictionary where keys are property names and values are NumPy arrays.
        :param filename: Name of the HDF5 file to save to.
        :param compression_level: 0 (no compression) and 9 (maximum compression), default is 1 .

        Notes
        -----
        * If ``filename`` is ``None``, properties are written into the root-level HDF5 group ``/properties`` in ``sol_filename.h5`` using a time-indexed 2D layout:``(n_timesteps, n_cells)``.
        * The time index is determined by matching the provided ``time`` value against ``/dynamic/time``.
        * If ``filename`` is specified, a new HDF5 file is created containing only the ``property_array`` data. Careful not to overwrite an existing file.
        """

        self.timer.start()
        self.timer.node["exporting_property_array"].start()

        if filename is None:
            self.append_properties_to_reservoir(time_vector, property_array)
        else:
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

    def load_property_array(self, filepath: str = "property_array.h5"):
        """
        Load properties into a dictionary.

        :param filepath: Path to a standalone properties HDF5 file (default: ``property_array.h5``).
        :returns: Tuple ``(time_vector, property_array)`` where:
            - ``time_vector`` is a 1D NumPy array of timesteps.
            - ``property_array`` is a dictionary mapping property names to NumPy arrays.
        :raises KeyError: If fallback to ``self.sol_filepath`` is triggered and the file does not contain a ``/properties`` group.

        Notes
        -----
        * Properties are saved as a standalone HDF5 file (``property_array.h5`` by default) or appended to ``reservoir_solution.h5`` under the group ``/properties``.
        * If ``property_array.h5`` does not exist this function looks for data in ``reservoir_solution.h5``.
        """
        try:
            property_array = {}
            with h5py.File(filepath, "r") as h5f:
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
                        f"No 'properties' group found in the {self.sol_filepath} file."
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
                f.write(f"{self.thermal}\n")

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

    def _get_output_cell_centers(self):
        """
        Resolve cell center coordinates for the entire reservoir.

        :returns: numpy array of shape (n_cells, 3) with [x, y, z] coordinates
        :raises ValueError: when centroids are missing or have unexpected shape

        Notes
        -----
        This method supports multiple reservoir/discretizer variants with different
        centroid storage fields and returns all available centroids as-is.
        """
        # Resolve the centroids array from known reservoir/discretizer fields.
        centroids = None
        if hasattr(self.reservoir, "discretizer"):
            if hasattr(self.reservoir.discretizer, "centroids_all_cells"):
                centroids = self.reservoir.discretizer.centroids_all_cells
            elif hasattr(self.reservoir.discretizer, "centroid_all_cells"):
                centroids = self.reservoir.discretizer.centroid_all_cells

        if centroids is None and hasattr(self.reservoir, "centroids_all_cells"):
            centroids = self.reservoir.centroids_all_cells
        if centroids is None and hasattr(self.reservoir, "centroids"):
            centroids = self.reservoir.centroids
        if centroids is None and hasattr(self.reservoir, "discr_mesh"):
            if hasattr(self.reservoir.discr_mesh, "centroids"):
                centroids = self.reservoir.discr_mesh.centroids
        if centroids is None and hasattr(self.reservoir, "mesh"):
            if hasattr(self.reservoir.mesh, "centroids"):
                centroids = self.reservoir.mesh.centroids

        if centroids is None:
            raise ValueError(
                "Cell centroids are not available for this reservoir type."
            )

        # Convert possible wrapped containers to a numpy array.
        centroids = np.asarray(centroids)
        if (
            centroids.ndim == 1
            and centroids.size > 0
            and hasattr(centroids[0], "values")
        ):
            centroids = np.vstack(
                [np.asarray(c.values, dtype=np.float64) for c in centroids]
            )

        # Expect a 2D array with x/y/z columns.
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                f"Expected centroids with shape (n_cells, 3), got {centroids.shape}."
            )

        return centroids

    def configure_h5_output(
        self, sol_filepath: str, cell_ids, description, add_static_data: bool = False
    ):
        """
        Create and initialize an HDF5 output file for simulation results.

        :param sol_filepath: Path of the HDF5 file to create.
        :param cell_ids: Cell/block indices to include in the dynamic output.
        :param description: Text description stored as the file attribute ``description``.
        :param add_static_data: If True, also write ``/static/block_m``, ``/static/block_p`` and
            ``/static/grav_coef``. Default is False.

        .. rubric:: HDF5 layout
        **/static** (optional)

            - ``cell_centers``: ``(n_blocks, dim)`` float. Written only when ``cell_ids`` covers all reservoir blocks.
            - ``block_m``, ``block_p``, ``grav_coef``: arrays. Written when ``add_static_data=True``.

        **/dynamic**

            - ``time``: ``(n_t,)`` float, extensible along time.
            - ``CFL_max``: ``(n_t,)`` float, extensible along time.
            - ``cell_id``: ``(n_selected,)`` int32.
            - ``X``: ``(n_t, n_selected, n_vars)`` float, extensible along time.
            - ``variable_names``: ``(n_vars,)`` variable-length strings.

        **/file attributes**

            - ``description``: string stored as a file attribute.

        .. rubric:: Notes
        - Open and read HDF5 files using an HDF5 viewer.
        - Careful not to overwrite existing files in ``output_folder\\...``.
        """

        with h5py.File(sol_filepath, "w") as f:
            # add static data group
            need_static = add_static_data or (
                cell_ids.size == self.reservoir.mesh.n_res_blocks
            )
            static_group = f.require_group("static") if need_static else None
            if cell_ids.size == self.reservoir.mesh.n_res_blocks:
                cell_centers = self._get_output_cell_centers().astype(
                    self.precision_map[self.precision], copy=False
                )
                static_group.create_dataset(
                    "cell_centers",
                    data=cell_centers,
                    dtype=self.precision_map[self.precision],
                )
            if add_static_data:
                block_m = np.array(self.reservoir.mesh.block_m, copy=False)
                block_p = np.array(self.reservoir.mesh.block_p, copy=False)
                grav_coef = np.array(self.reservoir.mesh.grav_coef, copy=False)
                static_group.create_dataset("block_m", data=block_m)
                static_group.create_dataset("block_p", data=block_p)
                static_group.create_dataset("grav_coef", data=grav_coef)

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
                compression_opts=self.compression_level,
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
        Configuration of output files.

        :param kind: ``'well'`` for well output or ``'reservoir'`` to write the reservoir solution vector.
        :type kind: str
        """

        # Ensure the directory and subdirectory exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "figures"), exist_ok=True)

        # solution output
        if kind == "reservoir":
            sol_filepath = os.path.join(self.output_folder, self.sol_filename)
            if os.path.exists(sol_filepath):  # and not restart:
                os.remove(sol_filepath)
            self.configure_h5_output(
                sol_filepath=sol_filepath,
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
                sol_filepath=well_output_path,
                cell_ids=self.id_well_data,
                add_static_data=True,
                description="Well data",
            )

        if hasattr(self, "output_configured"):
            self.output_configured.append(kind)
        else:
            self.output_configured = [kind]

    def save_specific_data(self, sol_filepath, X_data=None):
        """
        Save simulation data to an HDF5 file.

        :param sol_filepath: Path to the HDF5 file.
        :type sol_filepath: str
        :param X_data: Tuple containing (time array, data array). If None, current engine state is saved.
        :type X_data: tuple or list or None
        """
        is_batch = X_data is not None

        with h5py.File(sol_filepath, "a") as f:
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
            print(f"[{mode}] Saved {n_new} entry(ies) to {sol_filepath}")

    def save_data_to_h5(self, kind):
        """
        Function to write output solution or well output to *.h5 file

        :param kind: 'well' for well output or 'solution' to write the whole solution vector
        :type kind: str
        :raises ValueError: If ``kind`` is not ``'well'`` or ``'reservoir'``.

        Notes
        -----
        * This function is called after DartsModel.run(save_reservoir_data = True) unless explicitly stated otherwise with the flag.
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
            raise ValueError("kind must be either 'well' or 'reservoir'.")

    def read_specific_data(
        self,
        sol_filepath: str,
        ts_idx: int = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract time and data (primary variables) from an HDF5 file for a given timestep

        :param sol_filepath: Path to the HDF5 file.
        :type sol_filepath: str
        :param ts_idx: The timestep index to extract data for.
        :type ts_idx: int
        :returns:
            * **time** – ndarray with extracted timesteps.
            * **cell_id** – ndarray with cell_id of each of the saved grid blocks.
            * **X** – ndarray with data ``(number_of_timesteps, number_of_cells, number_of_vars)``.
            * **var_names** – ndarray with variable names.
        :raises FileNotFoundError: If the file does not exist.
        :raises IndexError: If `ts_idx` is out of range.
        """
        try:
            with h5py.File(sol_filepath, "r") as file:
                if ts_idx is None:
                    cell_id = file["dynamic/cell_id"][:]
                    var_names = file["dynamic/variable_names"][:]
                    time = file["dynamic/time"][:]

                    # memory check
                    # does not load data into memory since we are not slicing
                    dataset = file["dynamic/X"]
                    convert2MB = 1e6
                    # number of bytes per element
                    estimated_size_mb = (
                        dataset.size * dataset.dtype.itemsize / convert2MB
                    )
                    if estimated_size_mb > 1000:  # throw a warning if more than 1GB
                        print(
                            f"WARNING: Dataset 'X' is approximately {estimated_size_mb:.1f} MB. Loading it may impact memory performance."
                        )

                    X = file["dynamic/X"][:]

                else:
                    if not isinstance(ts_idx, int):
                        raise TypeError(
                            f"Expected 'ts_idx' to be an int, but got {type(ts_idx).__name__}"
                        )

                    cell_id = file["dynamic/cell_id"][:]
                    var_names = file["dynamic/variable_names"][:]

                    try:
                        time = file["dynamic/time"][ts_idx].reshape(1)
                        X = file["dynamic/X"][ts_idx].reshape(
                            1, len(cell_id), len(var_names)
                        )

                    except IndexError as err:
                        raise IndexError(
                            f"Timestep {ts_idx} does not exist in {sol_filepath}."
                        ) from err

            for i, name in enumerate(var_names):
                var_names[i] = name.decode()

        except FileNotFoundError as err:
            raise FileNotFoundError(f"File not found: {sol_filepath}.") from err

        return time, cell_id, X, var_names

    def output_properties(
        self,
        sol_filepath: str = None,
        output_properties: list = None,
        ts_idx: int = None,
        engine: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """
        Evaluate reservoir properties from saved data (HDF5 file) or a simulation engine.

        :param sol_filepath: Path to the solution HDF5 file. Defaults to None, in which case the dartsmodel.sol_filepath is used.
        :type sol_filepath: str
        :param output_properties: List of properties to evaluate. Defaults to None, which returns an array containing only state variables.
        :type output_properties: list
        :param ts_idx: Timestep index at which to evaluate properties. Defaults to None, which will evaluate all saved timesteps.
        :type ts_idx: int
        :param engine: If true, state variables are evaluated directly from engine.X. Defaults to False, which reads properties from the HDF5 file.
        :type engine: bool
        :returns:
            * **property_array** (dict) - A dictionary where keys are primary/secondary variables and values are NumPy arrays of the requested properties for each grid block. The shape of each array is (number_of_timesteps, number_of_gridblocks).
            * **timesteps** (ndarray) - A NumPy array of the time labels
        :raises KeyError: If specified property in `output_properties` is not found in any property container
        :raises TypeError: If output_properties is not a list

        Notes
        -----
        * ith_step indexes ``dynamic/time`` in the ``reservoir_solution.h5`` file.
        """

        if self.verbose:
            print(
                f'Processing properties {output_properties} at timestep index {ts_idx}'
            )

        if output_properties is not None and not isinstance(output_properties, list):
            raise TypeError(
                f"Expected 'output_properties' to be a list, but got {type(output_properties).__name__}."
            )

        if not engine:
            # Evaluate properties from the HDF5 file
            if sol_filepath is None:  # Establish sol_filepath/name to HDF5 file
                path = os.path.join(self.output_folder, self.sol_filename)
            else:
                path = sol_filepath
            timesteps, cell_id, X, var_names = self.read_specific_data(
                path, ts_idx
            )  # Read data from HDF5 file
        else:
            # Evaluate properties from the physics.engine.X
            # Get current time
            timesteps = np.array(self.physics.engine.t).reshape(1)

            X = np.array(
                self.physics.engine.X[
                    : self.physics.n_vars * self.reservoir.mesh.n_res_blocks
                ],
                copy=True,
            )  # reservoir solution at current time
            var_names = self.physics.vars  # primary variable names

        n_vars = len(var_names)  # number of primary variables
        nb = self.reservoir.mesh.n_res_blocks  # number of reservoir blocks
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
        Function to for creating `.vtk` files for viewing results in Paraview.

        :param sol_filepath: Path to the solution HDF5 file. Defaults to None, in which case the default path is used.
        :type sol_filepath: str
        :param ith_step: ith reporting step indicates which timestep to create a .vtk from. Defaults to None, in which case all saved data points are evaluated.
        :type ith_step: int
        :param output_directory: directory of where to save .vtk file. Defaults to none in which case the 'self.output_folder/vtk' is used.
        :type output_directory: str
        :param output_properties: List of properties to include in .vtk file. Defaults to None in which case only primary (state) variables are evaluated.
        :type output_properties: list
        :param output_data: List [array of timesteps, dictionary of property arrays]. Defaults to None, in which case properties are evaluated from the HDF5 file or engine
        :type output_data: list

        Notes
        -----
        * If no function inputs are specified .vtk files are created from all the available data in ``reservoir_solution.h5``.
        * The input ``ith_step`` indexes ``dynamic/time`` in the ``reservoir_solution.h5`` file.
        * If output_data = [timesteps, property_array] is passed directly as input, ``ith_step`` merely functions as a label in the created .vtk filename.
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
        sol_filepath: str = None,
        output_properties: list = None,
        ts_idx: int = None,
        engine: bool = False,
        output_data: list = None,
    ) -> xr.Dataset:
        """
        Generates an xarray Dataset of properties and saves it as a NetCDF file.
        State variables area obtained from the engine or *.h5 file.
        Properties are interpolated by the property iterator.

        :param sol_filepath: Path to the solution HDF5 file. Defaults to None, in which case the default path is used.
        :type sol_filepath: str
        :param output_properties: List of properties to include in the dataset. If None, all properties are included.
        :type output_properties: list
        :param ts_idx: Timestep index to output. If None, all timesteps are included.
        :type ts_idx: int
        :param engine: import state variable from engine if True. Default is False.
        :type engine: bool
        :param output_data: List [array of timesteps, dictionary of property arrays]. Defaults to None, in which case properties are evaluated from the HDF5 file or engine
        :type output_data: list
        :returns: xarray Dataset containing the property data.
        :rtype: xarray.Dataset
        """
        from darts.reservoirs.struct_reservoir import StructReservoir

        if output_data is None:
            time, data = self.output_properties(
                self.sol_filepath if sol_filepath is None else sol_filepath,
                output_properties,
                ts_idx,
                engine,
            )
        else:
            time, data = output_data[0], output_data[1]

        # # Interpolate properties
        # time, data = self.output_properties(
        #     sol_filepath, output_properties, ts_idx, engine
        # )
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
        if isinstance(self.reservoir, StructReservoir):
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

    def output_to_plt(
        self,
        sol_filepath: str = None,
        xarray_data: xr.Dataset = None,
        output_properties: list = None,
        ts_idx: int = None,
        x_slice: int = None,
        y_slice: int = None,
        z_slice: int = None,
        lims: dict = None,
        fig_size: tuple = None,
        axs_shape: tuple = None,
        aspect_ratio: str = "equal",
        logx: bool = False,
        plot_zeros: bool = True,
        cmap: str = "jet",
        colorbar_loc: str = "right",
    ):
        """
        Method for plotting output using matplotlib library. !! Requires an xarray_data as an input !!

        :param sol_filepath: Path to the solution HDF5 file. Defaults to None, in which case the default path is used.
        :type sol_filepath: str
        :param xarray_data: Data for output
        :type xarray_data: xr.Dataset
        :param output_properties: List of properties to plot
        :type output_properties: list
        :param ts_idx: Timestep index to plot (int or list of int)
        :type ts_idx: int
        :param x_slice: index for cross-section in x-dimension
        :param y_slice: index for cross-section in y-dimension
        :param z_slice: index for cross-section in z-dimension
        :param lims: Dictionary of lists with [lower, upper] limits for output variables, will default to [None, None]
        :type lims: dict
        :param fig_size: Tuple of (width, height) for figure
        :param axs_shape: Tuple of (rows, columns) for figure
        :param aspect_ratio: Aspect ratio ('equal', 'auto', or float), default is 'equal'
        :param logx: Bool to plot x-axis in logscale, default is False
        :param plot_zeros: Bool to plot zero values, default is True
        :param cmap: plt.Colourmap, default is 'jet'
        :param colorbar_loc: Location of colorbar ('right' or 'bottom'), default is 'right'
        """
        from darts.reservoirs.struct_reservoir import StructReservoir

        if not isinstance(self.reservoir, StructReservoir):
            raise AttributeError("Reservoir class must be of type StructReservoir.")
        dims_to_plot = (
            self.reservoir.ndims
            - (x_slice is not None)
            - (y_slice is not None)
            - (z_slice is not None)
        )
        assert dims_to_plot <= 2, "No implementation exists for 3D plt"

        output_directory = os.path.join(self.output_folder, "figures")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        # Check what data to use
        if sol_filepath is not None and xarray_data is not None:
            warnings.warn(
                "Both solution filepath and xarray Dataset were provided to output_to_plt(), choosing xarray",
                stacklevel=2,
            )
        # If no xarray_data has been provided, either generate from solution file or engine.X
        if xarray_data is None:
            xarray_data = self.output_to_xarray(
                sol_filepath=sol_filepath,
                output_properties=output_properties,
                ts_idx=ts_idx,
                engine=(sol_filepath is None),  # get from engine if no file provided
            )

        # Check if slices are consistent
        if x_slice is not None:
            assert x_slice < len(xarray_data['x']), (
                f"x-level step should be less than {len(xarray_data['x']):d}"
            )
        if y_slice is not None:
            assert y_slice < len(xarray_data['y']), (
                f"y-level step should be less than {len(xarray_data['y']):d}"
            )
        if z_slice is not None:
            assert z_slice < len(xarray_data['z']), (
                f"z-level step should be less than {len(xarray_data['z']):d}"
            )

        # Set subplots, shape and size
        axs_shape = axs_shape if axs_shape is not None else (1, len(output_properties))
        fig_size = (
            fig_size
            if fig_size is not None
            else (axs_shape[1] * 3.5, axs_shape[0] * 3.5)
        )

        # Define limits for properties
        lims = lims if lims is not None else {}
        for prop in output_properties:
            if prop not in lims.keys():
                lims[prop] = [None, None]

        # Slice dataset in space
        slices = {
            key: val
            for key, val in {'z': z_slice, 'y': y_slice, 'x': x_slice}.items()
            if val is not None
        }
        data = xarray_data.isel(slices)

        # Set zeros to nan if plot_zeros = False
        if not plot_zeros:
            data = data.where(data != 0.0, np.nan)

        # For 1D plot
        if dims_to_plot == 1:
            # Plot each timestep (plot over old fig object if provided)
            for ith_timestep, _ts in enumerate(data['time']):
                fig, axs = plt.subplots(
                    nrows=axs_shape[0],
                    ncols=axs_shape[1],
                    figsize=fig_size,
                    dpi=100,
                    facecolor="w",
                    edgecolor="k",
                )
                for j, prop in enumerate(output_properties):
                    axs[j].set_title(prop)

                # Plot all output_properties
                for j, prop in enumerate(output_properties):
                    ax = fig.axes[j]

                    # Horizontal slice
                    if (self.reservoir.nx > 1 and x_slice is None) or (
                        self.reservoir.ny > 1 and y_slice is None
                    ):
                        x = (
                            self.reservoir.discretizer.centroids_all_cells[:, 0]
                            if x_slice is None
                            else self.reservoir.discretizer.centroids_all_cells[:, 1]
                        )

                        ax.plot(x, data[prop].isel(time=ith_timestep).squeeze().values)
                        ax.set(ylim=lims[prop])
                        if logx:
                            ax.set_xscale("log")
                            ax.set_xlim([np.min(x), np.max(x)])
                    # Vertical slice
                    elif self.reservoir.nz > 1 and z_slice is None:
                        z = self.reservoir.discretizer.centroids_all_cells[:, 2]
                        ax.plot(data[prop].isel(time=ith_timestep), z)
                        if prop in lims.keys():
                            ax.set(xlim=lims[prop])
                    else:
                        raise AssertionError(
                            "Slices for 1D plot inconsistent with reservoir dimensions"
                        )

                # Save figure
                filename = (
                    # if ts_idx is not None and not 0, use ts_idx, else ith_timestep as idx
                    (
                        f'ts{ts_idx:d}'
                        if ith_timestep == 0 and ts_idx
                        else f'ts{ith_timestep:d}'
                    )
                    + (f' x{x_slice:d}' if x_slice is not None else '')
                    + (f' y{y_slice:d}' if y_slice is not None else '')
                    + (f' z{x_slice:d}' if z_slice is not None else '')
                    + '.png'
                )
                plt.savefig(os.path.join(output_directory, filename))

        # For 2D plot
        else:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            dx, dy, dz = (
                self.reservoir.global_data["dx"],
                self.reservoir.global_data["dy"],
                self.reservoir.global_data["dz"],
            )
            # Use dx or dy for xgrid, depending on whether x-dimension/y-dimension exists
            x_used, y_used = False, False  # track which dimensions
            if self.reservoir.nx > 1 and x_slice is None:
                xgrid = np.append(0, np.cumsum(dx[:, 0, 0]))
                x_used = True
            elif self.reservoir.ny > 1 and y_slice is None:
                xgrid = np.append(0, np.cumsum(dy[0, :, 0]))
                y_used = True
            else:
                raise AssertionError(
                    "Slices for 2D plot inconsistent with reservoir dimensions"
                )

            # Use dy or dz for ygrid, depending on whether y-dimension/z-dimension exists
            # If y-dimension was already used for xgrid, use z-dimension
            if self.reservoir.ny > 1 and y_slice is None and not y_used:
                ygrid = np.append(0, np.cumsum(dy[0, :, 0]))
                y_used = True
            elif self.reservoir.nz > 1 and z_slice is None:
                ygrid = np.append(0, np.cumsum(dz[0, 0, :]))
            else:
                raise AssertionError(
                    "Slices for 2D plot inconsistent with reservoir dimensions"
                )

            # Create meshgrid object and determine size
            X, Y = np.meshgrid(xgrid, ygrid)
            if x_used and y_used:  # Both: x, y
                shape = (self.reservoir.ny, self.reservoir.nx)
                # transpose = False
            elif x_used:  # Only x: x, z
                shape = (self.reservoir.nz, self.reservoir.nx)
                # transpose = True
            else:  # Not x: y, z
                shape = (self.reservoir.nz, self.reservoir.ny)
                # transpose = True

            # Plot each timestep (plot over old fig object if provided)
            for ith_timestep, _ts in enumerate(data['time']):
                fig, axs = plt.subplots(
                    nrows=axs_shape[0],
                    ncols=axs_shape[1],
                    figsize=fig_size,
                    dpi=100,
                    facecolor="w",
                    edgecolor="k",
                )
                for j, prop in enumerate(output_properties):
                    axs[j].set_title(prop)

                # Plot all output_properties
                for j, prop in enumerate(output_properties):
                    ax = fig.axes[j]

                    im = ax.pcolormesh(
                        X,
                        Y,
                        data[prop]
                        .isel(time=ith_timestep)
                        .squeeze()
                        .values.reshape(shape),
                        cmap=cmap,
                        vmin=lims[prop][0],
                        vmax=lims[prop][1],
                    )
                    if not x_used or not y_used:  # z-dimension used, invert y-axis
                        ax.invert_yaxis()
                    if logx:
                        ax.set_xscale("log")
                        ax.set_xlim([xgrid[1], xgrid[-1]])
                        ax.set_aspect("auto")
                    else:
                        ax.set_aspect(aspect_ratio)

                    divider = make_axes_locatable(ax)
                    if colorbar_loc == 'right':
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im, cax=cax, orientation='vertical')
                    else:
                        cax = divider.append_axes('bottom', size='15%', pad=0.3)
                        fig.colorbar(im, cax=cax, orientation='horizontal')
                    # cbar.set_ticks(np.linspace(lims[j][0], lims[j][1], 6))
                    # cbar.set_ticklabels(["{:.1f}".format(xx) for xx in np.linspace(lims[j][0], lims[j][1], 6)])
                plt.tight_layout()

                # Save figure
                filename = (
                    # if ts_idx is not None and not 0, use ts_idx, else ith_timestep as idx
                    (
                        f'ts{ts_idx:d}'
                        if ith_timestep == 0 and ts_idx
                        else f'ts{ith_timestep:d}'
                    )
                    + (f' x{x_slice:d}' if x_slice is not None else '')
                    + (f' y{y_slice:d}' if y_slice is not None else '')
                    + (f' z{x_slice:d}' if z_slice is not None else '')
                    + '.png'
                )
                plt.savefig(os.path.join(output_directory, filename))

        plt.close('all')
        return fig

    def well_output_to_vtp(
        self,
        ith_step: int,
        output_properties: list = None,
        output_directory: str = None,
    ):
        """
        Evaluate and store well primary and secondary variables of the ith step in vtp files

        :param output_properties: List of properties to evaluate. Defaults to None, which considers only primary vars.
        :type output_properties: list
        :param ith_step: ith reporting step for which you want to create vtp files for
        :type ith_step: int
        :param output_directory: Directory of where to save vtp files
        :type: str
        """
        if not self.has_dfm_well:
            return

        self.timer.start()
        self.timer.node["vtp_output"].start()

        # Set default output directory
        if output_directory is None:
            output_directory = os.path.join(self.output_folder, "vtp_files")
        os.makedirs(output_directory, exist_ok=True)

        # Evaluate well secondary variables of the current time from engine.X
        time, output_data = self.well_output_properties(
            output_properties=output_properties, ith_step=ith_step
        )

        # Store well primary and seconday props in vtp files
        for w_name in self.wells.keys():
            # If the well has n segments, so n+1 nodes
            z_nodes = np.concatenate(
                (
                    [0],
                    self.wells[w_name].geometry.TVD_interfaces,
                    [self.wells[w_name].geometry.pipe_length],
                )
            )
            # Flip depth sign for VTP (positive z in DARTS is downward, while negative z in ParaView is downward)
            z_nodes = -z_nodes
            x_nodes = np.zeros_like(
                z_nodes
            )  # x is zero since the well is located at the center of the cylindrical grid
            y_nodes = np.zeros_like(
                z_nodes
            )  # y is zero since the well is located at the center of the cylindrical grid
            nodes_coords = np.column_stack((x_nodes, y_nodes, z_nodes))

            self.write_well_output_properties_to_vtp(
                well_name=w_name,
                nodes_xyz=nodes_coords,
                output_properties=output_data,
                ith_step=ith_step,
                time=time,
                output_directory=output_directory,
            )

        self.timer.node["vtp_output"].stop()
        self.timer.stop()

    def well_output_properties(
        self,
        output_properties: list = None,
        ith_step: int = None,
    ):
        """
        Evaluate well secondary variables of the current time from engine.X

        :param output_properties: List of properties to evaluate. Defaults to None, which returns an array containing only state variables.
        :type output_properties: list
        :param ith_step: ith reporting step for which you want to evaluate seconday variables
        :type ith_step: int

        :return timesteps: A NumPy array of the time labels
        :type timesteps: np.ndarray
        :return property_array: A dictionary where keys are primary/secondary variables and values are NumPy arrays of the requested properties for each grid block. The shape of each array is (number_of_timesteps, number_of_gridblocks).
        :type property_array: dict
        """
        if self.verbose:
            print(
                f'Processing well properties {output_properties} at timestep index {ith_step}'
            )

        if output_properties is not None and not isinstance(output_properties, list):
            raise TypeError(
                f"Expected 'output_properties' to be a list, but got {type(output_properties).__name__}."
            )

        # Evaluate properties from the physics.engine.X
        time = self.physics.engine.t
        # Get well solution at current time
        n_vars = self.physics.n_vars
        X = np.array(
            self.physics.engine.X[n_vars * self.reservoir.mesh.n_res_blocks :],
            copy=True,
        )
        # number of well blocks
        nb = self.reservoir.mesh.n_blocks - self.reservoir.mesh.n_res_blocks
        output_properties = (
            output_properties
            if output_properties is not None
            else list(self.physics.vars)
        )  # complete list of properties

        # List of primary variables
        var_names = self.physics.vars  # primary variable names
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

        # Define property array dictionary
        property_array = {
            prop: np.zeros((1, nb)) for prop in (primary_props + secondary_props)
        }

        # Extract primary properties from X vector
        for var_name, var_idx in primary_prop_idxs.items():
            property_array[var_name][0] = X[var_idx::n_vars]

        # Interpolate secondary properties
        if secondary_props:  # if empty this part is skipped
            state = value_vector(
                np.stack([X[j::n_vars] for j in range(n_vars)]).T.flatten()
            )

            values = value_vector(np.zeros(self.n_ops * nb))
            values_numpy = np.array(values, copy=False)
            dvalues = value_vector(np.zeros(self.n_ops * nb * n_vars))

            for _region, prop_itor in self.physics.property_itor.items():
                block_idx = index_vector(np.arange(nb).astype(np.int32))
                prop_itor.evaluate_with_derivatives(
                    state, index_vector(block_idx), values, dvalues
                )

                for prop_name, prop_idx in secondary_prop_idxs.items():
                    temp = values_numpy[prop_idx :: self.n_ops]
                    property_array[prop_name][0][block_idx] = temp[block_idx]

        return time, property_array

    def write_well_output_properties_to_vtp(
        self,
        well_name: str,
        nodes_xyz: np.ndarray,
        output_properties: dict,
        ith_step: int,
        time: float,
        output_directory: str,
        active: bool = None,
    ):
        """
        Write well trajectory as .vtp (VTK PolyData) with segment-based primary and secondary vars as CELL data.

        :param well_name: Name of the well
        :type well_name: str
        :param nodes_xyz: XYZ coordinates of the nodes of the well (n_seg+1, 3)
        :type nodes_xyz: np.ndarray
        :param output_properties: Dict of properties to include in the vtp file
        :type output_properties: dict
        :param ith_step: i'th reporting step for which you want to create a .vtp file for
        :type ith_step: int
        :param time: Current simulation time
        :type time: float
        :param output_directory: Directory of where to save the vtp file
        :type: str
        :param active: Optional name of variable to set as active scalars
        :type active: bool
        """
        coords = np.asarray(nodes_xyz, dtype=float)
        npts = coords.shape[0]
        nseg = npts - 1

        # Points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetNumberOfPoints(npts)
        for i, (x, y, z) in enumerate(coords):
            vtk_points.SetPoint(i, float(x), float(y), float(z))

        # Lines
        vtk_lines = vtk.vtkCellArray()
        for i in range(nseg):
            vtk_lines.InsertNextCell(2)
            vtk_lines.InsertCellPoint(i)
            vtk_lines.InsertCellPoint(i + 1)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetLines(vtk_lines)

        # Add time
        tarr = vtk.vtkDoubleArray()
        tarr.SetName("TimeValue")
        tarr.SetNumberOfTuples(1)
        tarr.SetValue(0, float(time))
        poly.GetFieldData().AddArray(tarr)

        # Cell data (segment-based)
        cd = poly.GetCellData()
        for name, vals in output_properties.items():
            arr = np.asarray(vals).reshape((-1, 1))
            if arr.shape[0] != nseg:
                raise ValueError(f"'{name}' length {arr.shape[0]} != Nseg {nseg}")
            vtk_arr = numpy_to_vtk(arr.astype(float), deep=True)
            vtk_arr.SetName(name)
            cd.AddArray(vtk_arr)

        if active is None and output_properties:
            active = next(iter(output_properties.keys()))
        if active is not None:
            cd.SetActiveScalars(active)

        # Write
        writer = vtk.vtkXMLPolyDataWriter()
        output_file_name = f"solution_well_{well_name}_ts{ith_step:d}.vtp"
        output_file_path = os.path.join(output_directory, output_file_name)
        writer.SetFileName(output_file_path)
        writer.SetInputData(poly)
        writer.Write()

    def store_well_time_data(
        self,
        phase_molar_rates: bool = True,
        phase_mass_rates: bool = True,
        phase_volumetric_rates: bool = True,
        component_molar_rates: bool = True,
        component_mass_rates: bool = True,
        advective_heat_rates: bool = True,
        save_output_files: bool = False,
    ):
        """
        Compute and store well time data including rates and bottom-hole pressure and temperature (BHT and BHP)

        Rates are calculated for each perforation and also total rate of each well. Total rates are calculated using
        two different methods:
        1- summing up the rates of perforations
        2- calculating the rates directly at the wellhead connection

        :param phase_molar_rates: Compute phase molar rates
        :type phase_molar_rates: bool
        :param phase_mass_rates: Compute phase mass rates
        :type phase_mass_rates:bool
        :param phase_volumetric_rates: Compute phase volumetric rates
        :type phase_volumetric_rates:bool
        :param component_molar_rates: Compute component molar rates
        :type component_molar_rates: bool
        :param component_mass_rates: Compute component mass rates
        :type component_mass_rates: bool
        :param advective_heat_rates: Compute advective heat rates for thermal scenarios
        :type advective_heat_rates: bool
        :param save_output_files: Flag to save time_data as a .pkl and .xlsx file in the output folder
        :type save_output_files: bool
        """
        # Start timer for storing well time data
        self.timer.start()
        self.timer.node["output_well_time_data"].start()

        h5_well_data = load_hdf5_to_dict(self.well_filepath)

        time = h5_well_data["dynamic"]["time"]
        time_data_dict = {"time": time}

        (
            perfs_conn_idxs,
            well_head_conn_idxs,
            geometric_WI,
            well_head_conn_trans,
        ) = self.get_wellhead_perf_connection_info()

        # Store BHP and BHT
        self.store_bhp_bht(h5_well_data, time_data_dict)

        # Store types of well rates in a list to be calculated
        rate_types = []
        rate_types += ["phase_molar_rates"] if phase_molar_rates else []
        rate_types += ["phase_mass_rates"] if phase_mass_rates else []
        rate_types += ["phase_volumetric_rates"] if phase_volumetric_rates else []
        rate_types += ["component_molar_rates"] if component_molar_rates else []
        rate_types += ["component_mass_rates"] if component_mass_rates else []
        rate_types += (
            ["advective_heat_rates"] if advective_heat_rates and self.thermal else []
        )

        for rate_type in rate_types:
            # Since the geothermal engine supports only a single component, components rates are not needed.
            if rate_type in (
                "component_molar_rates",
                "component_mass_rates",
            ) and isinstance(self.physics, Geothermal):
                continue

            # Compute perforation rates
            rates_perfs = self.calc_rates_at_conns(
                h5_well_data,
                perfs_conn_idxs,
                geometric_WI,
                self.thermal,
                rate_type,
            )
            # Store perforation rates
            self.store_perf_rates(time_data_dict, rates_perfs, rate_type)
            # Store well rates by summing perforation rates
            self.store_perf_rate_sums(time_data_dict, rates_perfs, rate_type)

            # Compute wellhead rates
            rates_wellhead = self.calc_rates_at_conns(
                h5_well_data,
                well_head_conn_idxs,
                well_head_conn_trans,
                self.thermal,
                rate_type,
            )
            # Store wellhead rates
            self.store_wellhead_rates(time_data_dict, rates_wellhead, rate_type)

        # Export the dict
        if save_output_files:
            df = pd.DataFrame(time_data_dict)
            file_base = os.path.join(self.output_folder, "well_time_data")
            df.to_pickle(f"{file_base}.pkl")
            df.to_excel(f"{file_base}.xlsx", index=False)

        # End timer for storing well time data
        self.timer.node["output_well_time_data"].stop()
        self.timer.stop()
        return time_data_dict

    def get_wellhead_perf_connection_info(self):
        """
        Give information of the connections, including perforations and wellhead, for evaluation of
        perforation and wellhead rates in the method store_well_time_data of the current class.
        """
        block_m = np.array(self.reservoir.mesh.block_m, copy=False)
        block_p = np.array(self.reservoir.mesh.block_p, copy=False)
        # Create a dictionary containing connection indices of perforations for each well (values are lists)
        well_perf_conn_idxs = {}
        # Create a dictionary containing connection index of wellhead for each well (values are integers)
        well_head_conn_idx = {}
        for well in self.reservoir.wells:
            res_cell_idxs = [perf[1] for perf in well.perforations]

            # Find indices of perforations in the connection list (those connections
            # which 1. block_m is in the desired well and 2. block_p is in res_cell_idxs)
            mask = np.logical_and(
                np.isin(block_p, res_cell_idxs),
                np.logical_and(
                    block_m >= well.well_head_idx, block_m <= well.well_bottom_idx
                ),
            )

            conn_idxs = np.nonzero(mask)
            well_perf_conn_idxs[well.name] = conn_idxs[0]
            assert well_perf_conn_idxs[well.name].size == len(
                well.perforations
            ) and np.all(
                block_m[well_perf_conn_idxs[well.name]]
                > self.reservoir.mesh.n_res_blocks
            )

            # Find idx of well_head-well_body connection in the connection list
            wh_conn_idx = np.where(
                np.logical_and(
                    block_m == well.well_head_idx, block_p == well.well_body_idx
                )
            )[0]
            assert len(wh_conn_idx) == 1
            well_head_conn_idx[well.name] = wh_conn_idx[0]

        # Get perforation connection indices for all wells
        perfs_conn_idxs = [
            item for sublist in well_perf_conn_idxs.values() for item in sublist
        ]
        # Get wellhead connection indices for all wells
        well_head_conn_idxs = list(well_head_conn_idx.values())

        # Get well indices (WI) for each perforation
        geometric_WI = np.array(
            [p[2] for well in self.reservoir.wells for p in well.perforations]
        )
        # Get transmissibility for each wellhead connection
        well_head_conn_trans = np.array(
            [well.well_transmissibility for well in self.reservoir.wells]
        )

        return perfs_conn_idxs, well_head_conn_idxs, geometric_WI, well_head_conn_trans

    def store_perf_rates(
        self, time_data_dict: dict, rates_perfs: np.ndarray, rate_type: str
    ):
        """
        Store perforation rates from the 3D numpy array rates_perfs for the rate type rate_type in the
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

    def store_perf_rate_sums(
        self, time_data_dict: dict, rates_perfs: np.ndarray, rate_type: str
    ):
        """
        Store summation of perforation rates for each well from the 3D numpy array rates_perfs for the
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
        Store wellhead rate for each well from the 3D numpy array rates_perfs for the rate type
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
        Store bottom-hole pressure (BHP) and temperature (BHT) of wells over time in time_data_dict.
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
            BHT = np.zeros(nt) if self.thermal else np.full(nt, pc.temperature)
            wellhead_cell_idx = self.find_values_in_an_array(
                [well.well_head_idx], cell_id
            )
            wellhead_cell_idx = wellhead_cell_idx[0]  # convert it to a scalar
            p_idx = variable_names.index("pressure")
            for i in range(nt):
                BHP[i] = X[i, wellhead_cell_idx, p_idx]
                if self.thermal:
                    if "temperature" in variable_names:
                        t_idx = variable_names.index("temperature")
                        BHT[i] = X[i, wellhead_cell_idx, t_idx]
                    else:
                        pc.evaluate(X[i, wellhead_cell_idx, :])
                        BHT[i] = pc.temperature
            time_data_dict[f"well_{well.name}_BHP"] = BHP
            time_data_dict[f"well_{well.name}_BHT"] = BHT

    @staticmethod
    def get_gravity_and_capillary_pressure_ops(
        physics,
        reservoir_operator,
        reservoir_ops: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return phase density and capillary-pressure operators for well-rate upwinding.

        Super/chemistry-style operators expose explicit gravity and capillary-pressure
        operator offsets. The geothermal engine stores molar density in its density
        operator slice and has no capillary-pressure operator, so convert molar density
        to mass density to match the phase-potential term used by the engine.
        """
        pc = physics.property_containers[0]

        if hasattr(reservoir_operator, "GRAV_OP"):
            grav_start = reservoir_operator.GRAV_OP
            grav = reservoir_ops[:, grav_start : grav_start + pc.nph]

            if hasattr(reservoir_operator, "PC_OP"):
                pc_start = reservoir_operator.PC_OP
                capillary = reservoir_ops[:, pc_start : pc_start + pc.nph]
            else:
                capillary = np.zeros_like(grav)
            return grav, capillary

        if isinstance(physics, Geothermal):
            if hasattr(reservoir_operator, "DENS_OP"):
                dens_start = reservoir_operator.DENS_OP
            else:
                dens_start = pc.nc + pc.nc * pc.nph + pc.nph + 2

            molar_density = reservoir_ops[:, dens_start : dens_start + pc.nph]
            phase_mw = np.asarray(pc.Mw)[0]
            grav = molar_density * phase_mw
            capillary = np.zeros_like(grav)
            return grav, capillary

        raise AttributeError(
            "Reservoir operators must expose GRAV_OP/PC_OP or a supported "
            "engine-specific density layout for well-rate upwinding."
        )

    def calc_rates_at_conns(
        self,
        h5_well_data: dict,
        conn_idxs: list,
        trans: np.ndarray,
        thermal: bool,
        rate_type: str,
    ):
        """
        Calculate different types of rates at perforations or wellhead connections of wells.
        This function is used in the method store_well_time_data of the current class.

        To calculate connection rates, gravity is included, while capillary pressure is ignored in this
        postprocessing calculation.

        :param h5_well_data: Well data stored in the HDF5 file
        :type h5_well_data: dict
        :param conn_idxs: Indices of the desired connections in the connection list
        :type conn_idxs: list
        :param trans: Transmissibility (For perforations, it is geometric part of well index)
        :type trans: np.ndarray
        :param thermal: If the model is thermal or not
        :type thermal: bool
        :param rate_type: Type of well rate to calculate
        :type rate_type: str
        """
        # Evaluate position of block_m, block_p in stored data for every connection
        block_m = h5_well_data["static"]["block_m"]
        block_p = h5_well_data["static"]["block_p"]
        cell_id = h5_well_data["dynamic"]["cell_id"]
        # Line below finds indices of the m cells of the connections in cell_id
        cell_m = self.find_values_in_an_array(block_m[conn_idxs], cell_id)
        # Line below finds indices of the p cells of the connections in cell_id
        cell_p = self.find_values_in_an_array(block_p[conn_idxs], cell_id)
        n_conns = len(conn_idxs)
        assert cell_m.size == n_conns and cell_p.size == n_conns

        n_ts = h5_well_data["dynamic"]["time"].size

        physics = self.physics

        pc = physics.property_containers[0]
        ne = physics.reservoir_operators[0].ne

        p_idx = h5_well_data["dynamic"]["variable_names"].index("pressure")
        if thermal:
            if physics.state_spec == physics.StateSpecification.PT:
                t_idx = h5_well_data["dynamic"]["variable_names"].index("temperature")
            elif physics.state_spec == physics.StateSpecification.PH:
                t_idx = h5_well_data["dynamic"]["variable_names"].index("enthalpy")
            else:
                raise Exception(
                    "Neither temperature nor enthalpy exists in the list of variables!"
                )
        elif not thermal and rate_type == "advective_heat_rates":
            raise Exception(
                "The model is isothermal, so advective heat rate cannot be calculated for it!"
            )

        # This adds a new axis, turning a 1D array into a 2D column vector
        time_idx = np.arange(n_ts)[:, None]

        batch_size = n_ts * n_conns
        n_well_ctrl_ops = physics.well_ctrl_operators.n_ops
        n_reservoir_ops = physics.reservoir_operators[0].n_ops
        n_vars = physics.n_vars
        block_idx = index_vector(np.arange(batch_size).astype(np.int32))

        states_m = h5_well_data["dynamic"]["X"][time_idx, cell_m]
        states_p = h5_well_data["dynamic"]["X"][time_idx, cell_p]

        if self.precision == "s":
            axes_min = np.array(physics.axes_min)
            axes_max = np.array(physics.axes_max)
            states_m = np.clip(states_m, axes_min, axes_max)
            states_p = np.clip(states_p, axes_min, axes_max)

        states_m_2d = states_m.reshape(batch_size, n_vars)
        states_p_2d = states_p.reshape(batch_size, n_vars)

        def evaluate_ops(states_2d, n_ops, evaluator):
            values = value_vector(np.zeros(batch_size * n_ops))
            dvalues = value_vector(np.zeros((batch_size * n_ops) * n_vars))
            evaluator.evaluate_with_derivatives(
                value_vector(states_2d.ravel()), block_idx, values, dvalues
            )
            return np.asarray(values).reshape(batch_size, n_ops)

        reservoir_ops_m = evaluate_ops(
            states_m_2d, n_reservoir_ops, physics.acc_flux_itor[0]
        )
        reservoir_ops_p = evaluate_ops(
            states_p_2d, n_reservoir_ops, physics.acc_flux_itor[0]
        )

        p = h5_well_data["dynamic"]["X"][:, :, p_idx]
        dp = p[:, cell_p] - p[:, cell_m]

        reservoir_operator = physics.reservoir_operators[0]
        grav_m, _ = self.get_gravity_and_capillary_pressure_ops(
            physics, reservoir_operator, reservoir_ops_m
        )
        grav_p, _ = self.get_gravity_and_capillary_pressure_ops(
            physics, reservoir_operator, reservoir_ops_p
        )
        grav_m = grav_m.reshape(n_ts, n_conns, pc.nph)
        grav_p = grav_p.reshape(n_ts, n_conns, pc.nph)

        grav_coef = h5_well_data["static"]["grav_coef"]
        grav_coef = np.asarray(grav_coef)[conn_idxs][None, :, None]

        phase_p_diff = dp[:, :, None] + 0.5 * (grav_m + grav_p) * grav_coef
        upwind_m = phase_p_diff.reshape(batch_size, pc.nph) < 0

        if rate_type in [
            "phase_molar_rates",
            "phase_mass_rates",
            "phase_volumetric_rates",
            "advective_heat_rates",
        ]:
            well_ops_m = evaluate_ops(
                states_m_2d, n_well_ctrl_ops, physics.well_ctrl_itor
            )
            well_ops_p = evaluate_ops(
                states_p_2d, n_well_ctrl_ops, physics.well_ctrl_itor
            )
        elif rate_type not in ["component_molar_rates", "component_mass_rates"]:
            raise Exception(
                "The rate type is not entered correctly or is not supported!"
            )

        if rate_type == "phase_molar_rates":
            start = int(well_control_iface.MOLAR_RATE) * pc.nph
            ops = np.where(
                upwind_m,
                well_ops_m[:, start : start + pc.nph],
                well_ops_p[:, start : start + pc.nph],
            )
        elif rate_type == "phase_mass_rates":
            start = int(well_control_iface.MASS_RATE) * pc.nph
            ops = np.where(
                upwind_m,
                well_ops_m[:, start : start + pc.nph],
                well_ops_p[:, start : start + pc.nph],
            )
        elif rate_type == "phase_volumetric_rates":
            op_start = int(well_control_iface.VOLUMETRIC_RATE) * pc.nph
            ops = np.where(
                upwind_m,
                well_ops_m[:, op_start : op_start + pc.nph],
                well_ops_p[:, op_start : op_start + pc.nph],
            )
        elif rate_type == "component_molar_rates":
            op_start = reservoir_operator.FLUX_OP
            flux_ops_m = reservoir_ops_m[:, op_start : op_start + ne * pc.nph]
            flux_ops_p = reservoir_ops_p[:, op_start : op_start + ne * pc.nph]
            flux_ops_m = flux_ops_m.reshape(batch_size, pc.nph, ne)
            flux_ops_p = flux_ops_p.reshape(batch_size, pc.nph, ne)
            flux_ops = np.where(upwind_m[:, :, None], flux_ops_m, flux_ops_p)

            op_start = reservoir_operator.LAMBDA_OP
            lambda_op_m = reservoir_ops_m[:, op_start : op_start + pc.nph]
            lambda_op_p = reservoir_ops_p[:, op_start : op_start + pc.nph]
            lambda_op = np.where(upwind_m, lambda_op_m, lambda_op_p)
            lambda_op = lambda_op[:, :, np.newaxis]

            molar_ops = flux_ops[:, :, : pc.nc_fl] * lambda_op
            molar_ops = molar_ops.reshape(batch_size, pc.nph * pc.nc_fl)

            ops = molar_ops
        elif rate_type == "component_mass_rates":
            op_start = reservoir_operator.FLUX_OP
            flux_ops_m = reservoir_ops_m[:, op_start : op_start + ne * pc.nph]
            flux_ops_p = reservoir_ops_p[:, op_start : op_start + ne * pc.nph]
            flux_ops_m = flux_ops_m.reshape(batch_size, pc.nph, ne)
            flux_ops_p = flux_ops_p.reshape(batch_size, pc.nph, ne)
            flux_ops = np.where(upwind_m[:, :, None], flux_ops_m, flux_ops_p)

            op_start = reservoir_operator.LAMBDA_OP
            lambda_op_m = reservoir_ops_m[:, op_start : op_start + pc.nph]
            lambda_op_p = reservoir_ops_p[:, op_start : op_start + pc.nph]
            lambda_op = np.where(upwind_m, lambda_op_m, lambda_op_p)
            lambda_op = lambda_op[:, :, np.newaxis]

            molar_ops = flux_ops[:, :, : pc.nc_fl] * lambda_op
            molar_ops = molar_ops.reshape(batch_size, pc.nph * pc.nc_fl)

            mw = np.array(pc.Mw[: pc.nc_fl])
            mw_tiled = np.tile(mw, pc.nph)
            ops = molar_ops * mw_tiled
        elif rate_type == "advective_heat_rates":
            op_start = int(well_control_iface.ADVECTIVE_HEAT_RATE) * pc.nph
            ops = np.where(
                upwind_m,
                well_ops_m[:, op_start : op_start + pc.nph],
                well_ops_p[:, op_start : op_start + pc.nph],
            )

            # Calculate dead operators
            p_dead = 1.01325  # Dead pressure (1 atm)
            T_dead = 273.15 + 15  # Dead temperature (15 deg C)

            if not (
                self.physics.PT_axes_min[p_idx]
                <= p_dead
                <= self.physics.PT_axes_max[p_idx]
                and self.physics.PT_axes_min[t_idx]
                <= T_dead
                <= self.physics.PT_axes_max[t_idx]
            ):
                # Since the dead pressure or temperature for well advective heat rate calculation is outside the OBL bounds, leave it zero.
                return np.zeros((n_ts, n_conns, pc.nph))

            states_m_dead = states_m_2d.copy()
            states_p_dead = states_p_2d.copy()
            states_m_dead[:, p_idx] = p_dead
            states_p_dead[:, p_idx] = p_dead
            states_m_dead[:, t_idx] = T_dead
            states_p_dead[:, t_idx] = T_dead

            # Calculate heat operators at the dead state (1 atm and 15 deg C)
            if physics.state_spec == physics.StateSpecification.PT:
                values_dead_m = evaluate_ops(
                    states_m_dead, n_well_ctrl_ops, physics.well_ctrl_itor
                )
                values_dead_p = evaluate_ops(
                    states_p_dead, n_well_ctrl_ops, physics.well_ctrl_itor
                )
                op_start = int(well_control_iface.ADVECTIVE_HEAT_RATE) * pc.nph
                ops_dead = np.where(
                    upwind_m,
                    values_dead_m[:, op_start : op_start + pc.nph],
                    values_dead_p[:, op_start : op_start + pc.nph],
                )
            elif physics.state_spec == physics.StateSpecification.PH:
                # Calculate enthalpy for the dead state with temperature as the thermal variable
                n_thermal_var_op = physics.thermal_var_operator.n_ops

                enthalpies_dead_m = value_vector(
                    np.zeros(batch_size * n_thermal_var_op)
                )
                denthalpies_dead_m = value_vector(
                    np.zeros((batch_size * n_thermal_var_op) * n_vars)
                )
                physics.thermal_var_itor.evaluate_with_derivatives(
                    value_vector(states_m_dead.ravel()),
                    block_idx,
                    enthalpies_dead_m,
                    denthalpies_dead_m,
                )
                enthalpies_dead_p = value_vector(
                    np.zeros(batch_size * n_thermal_var_op)
                )
                denthalpies_dead_p = value_vector(
                    np.zeros((batch_size * n_thermal_var_op) * n_vars)
                )
                physics.thermal_var_itor.evaluate_with_derivatives(
                    value_vector(states_p_dead.ravel()),
                    block_idx,
                    enthalpies_dead_p,
                    denthalpies_dead_p,
                )

                # Update the dead state array (containing temperatures) with calculated enthalpies
                states_m_dead[:, t_idx] = np.asarray(enthalpies_dead_m)
                states_p_dead[:, t_idx] = np.asarray(enthalpies_dead_p)

                # Now pass the dead state array with enthalpies to the well control interpolator
                values_dead_m = evaluate_ops(
                    states_m_dead, n_well_ctrl_ops, physics.well_ctrl_itor
                )
                values_dead_p = evaluate_ops(
                    states_p_dead, n_well_ctrl_ops, physics.well_ctrl_itor
                )
                op_start = int(well_control_iface.ADVECTIVE_HEAT_RATE) * pc.nph
                ops_dead = np.where(
                    upwind_m,
                    values_dead_m[:, op_start : op_start + pc.nph],
                    values_dead_p[:, op_start : op_start + pc.nph],
                )

            ops = ops - ops_dead

        # Reshape arrays
        if rate_type in [
            "phase_molar_rates",
            "phase_mass_rates",
            "phase_volumetric_rates",
        ]:
            ops_reshaped = ops.reshape(n_ts, n_conns, pc.nph)
        elif rate_type in ["component_molar_rates", "component_mass_rates"]:
            ops_reshaped = ops.reshape(n_ts, n_conns, -1)
        elif rate_type == "advective_heat_rates":
            ops_reshaped = ops.reshape(n_ts, n_conns, pc.nph)

        tran = trans[None, :, None]
        if rate_type in ["component_molar_rates", "component_mass_rates"]:
            pressure_term = np.repeat(phase_p_diff, pc.nc_fl, axis=2)
        else:
            pressure_term = phase_p_diff
        rates = -ops_reshaped * tran * pressure_term

        return rates

    def plot_well_time_data(
        self,
        phase_molar_rates: bool = True,
        phase_mass_rates: bool = True,
        phase_volumetric_rates: bool = True,
        component_molar_rates: bool = True,
        component_mass_rates: bool = True,
        advective_heat_rates: bool = True,
    ):
        """
        Plot well time data.

        Bottom-hole pressure (BHP) and bottom-hole temperature (BHT) are
        always included in the plots. Additional well rate categories can
        be enabled or disabled using the corresponding boolean input arguments.

        :param phase_molar_rates: Plot phase molar rates, default is True
        :type phase_molar_rates: bool
        :param phase_mass_rates: Plot phase mass rates, default is True
        :type phase_mass_rates:bool
        :param phase_volumetric_rates: Plot phase volumetric rates, default is True
        :type phase_volumetric_rates:bool
        :param component_molar_rates: Plot component molar rates, default is True
        :type component_molar_rates: bool
        :param component_mass_rates: Plot component mass rates, default is True
        :type component_mass_rates: bool
        :param advective_heat_rates: Plot advective heat rates for thermal scenarios, default is True
        :type advective_heat_rates: bool
        """
        main_dir = os.path.join(self.output_folder, "figures/well_time_plots")

        # Reset_directory
        if os.path.exists(main_dir):
            shutil.rmtree(main_dir)
        os.makedirs(main_dir)

        self.create_perf_dirs(main_dir)

        df = pd.read_pickle(os.path.join(self.output_folder, "well_time_data.pkl"))
        time = df["time"]

        # Store types of well rates in a list to be plotted
        rate_types = []
        rate_types += ["phase_molar_rates"] if phase_molar_rates else []
        rate_types += ["phase_mass_rates"] if phase_mass_rates else []
        rate_types += ["phase_volumetric_rates"] if phase_volumetric_rates else []
        rate_types += ["component_molar_rates"] if component_molar_rates else []
        rate_types += ["component_mass_rates"] if component_mass_rates else []
        rate_types += (
            ["advective_heat_rates"] if advective_heat_rates and self.thermal else []
        )

        self.unit_dict = {
            "molar": "kmol/day",
            "mass": "kg/day",
            "volumetric": "m^3/day",
            "heat": "kJ/day",
        }

        for rtype in rate_types:
            for well in self.reservoir.wells:
                well_dir = os.path.join(main_dir, f"well_{well.name}")
                # perforation rate plots
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
                # total rate plots
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
        Create a new directory (folder) for each perforation of each well. The rates for each perforation
        will be stored in their corresponding directory later. This function is used in the method plot_well_time_data
        of the current class.

        :param main_dir: Directory in which a folder for each well already exists or will be created. Folder for each perforation will be created in the corresponding well folder.
        :type main_dir: str
        """
        for well in self.reservoir.wells:
            well_dir = os.path.join(main_dir, f"well_{well.name}")
            os.makedirs(well_dir, exist_ok=True)
            for perf_idx in range(len(well.perforations)):
                os.makedirs(os.path.join(well_dir, f"perf_{perf_idx}"), exist_ok=True)

    def create_perf_keys(self, rtype: str, well_name: str, perf_idx: int):
        """
        Create keys for perforation rates. This function is used in the method plot_well_time_data
        of the current class.

        :param rtype: Type of the well rate
        :type rtype: str
        :param well_name: Name of the well
        :type well_name: str
        :param perf_idx: Index of the perforation. This index starts from zero and the order depends on the order at which perforations are added to the wellbore using the add_perforation method.
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
        Create keys for summation rates, wellhead rates, and BHP and BHT. This function is used in the
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

    # Auxiliary functions
    def find_values_in_an_array(self, to_find: np.ndarray | list, arr: np.ndarray):
        """
        :param to_find: The values the indices of which we want to find in arr
        :type to_find: np.ndarray or list
        :param arr: The array in which we want to find the values in to_find
        :type arr: np.ndarray
        """
        indices = []
        for element in to_find:
            idx = np.where(arr == element)[0]
            if idx.size > 0:
                indices.append(idx[0])
        return np.array(indices, dtype=np.intp)
