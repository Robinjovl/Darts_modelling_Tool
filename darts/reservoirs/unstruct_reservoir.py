import os

import meshio
import numpy as np

from darts.engines import conn_mesh, index_vector, timer_node, value_vector
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.reservoir_base import ReservoirBase


class UnstructReservoir(ReservoirBase):
    """
    Class for generating unstructured mesh

    :param timer: Timer object
    :type timer: timer_node
    :param mesh_file: Mesh file
    :type mesh_file: str
    :param permx: Matrix permeability in the x-direction (scalar or vector)
    :param permy: Matrix permeability in the y-direction (scalar or vector)
    :param permz: Matrix permeability in the z-direction (scalar or vector)
    :param poro: Matrix (and fracture?) porosity (scalar or vector)
    :param rcond: Matrix rock conduction (scalar or vector)
    :param hcap: Matrix heat capacity (scalar or vector)
    :param frac_aper: Aperture of the fracture (scalar or vector)
    :param op_num: Index of operator set
    :param cache: Switch to load/save cache of discretization
    :type cache: bool
    """

    def __init__(
        self,
        timer: timer_node,
        mesh_file: str,
        permx,
        permy,
        permz,
        poro,
        rcond=0,
        hcap=0,
        frac_aper=0,
        op_num=0,
        cache: bool = False,
    ):
        super().__init__(timer, cache)

        self.mesh_file = mesh_file

        self.permx = permx
        self.permy = permy
        self.permz = permz
        self.poro = poro
        self.frac_aper = frac_aper
        self.rcond = rcond
        self.hcap = hcap
        self.op_num = op_num

        self.physical_tags = {
            "matrix": [],
            "fracture": [],
            "boundary": [],
            "fracture_boundary": [],
            "output": [],
        }

        # parameters for optional fracture aperture computation depending on principal stresses
        self.sh_max = None
        self.sh_min = None
        self.sh_max_azimuth = None
        self.sigma_c = None

    def discretize(self, verbose: bool = False) -> conn_mesh:
        # Construct instance of Unstructured Discretization class:
        self.discretizer = UnstructDiscretizer(
            mesh_file=self.mesh_file, physical_tags=self.physical_tags, verbose=verbose
        )

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh(
            permx=self.permx,
            permy=self.permy,
            permz=self.permz,
            frac_aper=self.frac_aper,
            cache=self.cache,
        )

        if self.frac_aper is not None and self.sh_max is not None:
            self.discretizer.calc_frac_aper_by_stress(
                self.frac_aper,
                self.sh_max,
                self.sh_min,
                self.sh_max_azimuth,
                self.sigma_c,
            )

        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroids_all_cells()

        # Assign layer properties
        self.set_layer_properties()

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = (
            self.discretizer.calc_connections_all_cells(cache=self.cache)
        )

        # Initialize mesh using built connection list
        mesh = conn_mesh()
        mesh.init(
            index_vector(cell_m),
            index_vector(cell_p),
            value_vector(tran),
            value_vector(tran_thermal),
        )

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.op_num, copy=False)[:] = self.op_num
        n_elements = self.discretizer.mat_cells_tot + self.discretizer.frac_cells_tot
        np.array(mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells[
            :n_elements
        ]
        np.array(mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells[
            :n_elements
        ]

        return mesh

    def set_boundary_volume(self, boundary_volumes: dict):
        # Set-up dictionary with data for boundary cells:
        boundary_data = dict()  # Dictionary containing boundary condition data (coordinate and value of boundary):
        boundary_data['first_boundary_dir'] = (
            'X'  # Indicates the boundary is located at constant X (in this case!)
        )
        # Constant X-coordinate value at which the boundary is located (used to be 3.40885):
        boundary_data["first_boundary_val"] = np.min(
            self.discretizer.mesh_data.points[:, 0]
        )

        # Same as above but for the second boundary condition!
        boundary_data["second_boundary_dir"] = "X"
        # Constant X-coordinate value at which the boundary is located (used to be 13.0014):
        boundary_data["second_boundary_val"] = np.max(
            self.discretizer.mesh_data.points[:, 0]
        )

        # Calculate boundary cells using the calc_boundary_cells method:
        self.left_boundary_cells, self.right_boundary_cells = (
            self.discretizer.calc_boundary_cells(boundary_data)
        )

        # Calc maximum size of well cells (used to have more homogeneous injection conditions by scaling the WI):
        dummy_vol = np.array(self.volume, copy=True)
        self.max_well_vol = np.max(
            [
                np.max(dummy_vol[self.left_boundary_cells]),
                np.max(dummy_vol[self.right_boundary_cells]),
            ]
        )

        self.volume[self.right_boundary_cells] = (
            self.volume[self.right_boundary_cells] * 1e8
        )
        return

    def add_perforation(
        self,
        well_name: str,
        res_cell_idx: int,
        well_seg_idx: int = None,
        well_diameter: float = 0.3048,
        well_index: float = None,
        well_indexD: float = None,
        segment_direction: str = "z_axis",
        skin: float = 0.0,
        ms_epm: bool = False,
        verbose: bool = False,
    ):
        """
        Function to add a perforation to the well

        :param well_seg_idx: Currently, this is only used for struct_reservoir.
        :param res_cell_idx: Index of reservoir cell to be perforated
        :type res_cell_idx: Reservoir cell index for the unstructured reservoir grid must be an integer.
        """
        well = self.get_well(well_name)

        perf_indices = np.array(well.perforations, dtype=int)
        # res_cell_idx has index=1 in perforation element: (well_block, res_cell_idx, well_index, well_indexD)
        perf_indices = perf_indices[:, 1] if len(well.perforations) > 0 else []
        if res_cell_idx in perf_indices:
            print(
                "There are at least 2 wells locating in the same grid block!!! The mesh file should be modified!"
            )
            exit()

        #  update well depth
        perf_indices = np.append(perf_indices, res_cell_idx).astype(
            int
        )  # add current cell to previous perforation list
        # set well depth to the top perforation depth
        well.well_head_depth = np.array(self.mesh.depth, copy=False)[perf_indices].min()
        well.well_body_depth = well.well_head_depth

        if well_index is None or well_indexD is None:
            # calculate well index and get local index of reservoir block
            wi, wid = self.discretizer.calc_equivalent_well_index(
                res_cell_idx, well_diameter, skin
            )
            well_index = wi if well_index is None else well_index
            well_indexD = wid if well_indexD is None else well_indexD

        assert well_index >= 0
        assert well_indexD >= 0

        # set well segment index (well block) equal to index of perforation layer
        if ms_epm:
            well_block = len(well.perforations)
        else:
            well_block = 0

        well.perforations = well.perforations + [
            (well_block, res_cell_idx, well_index, well_indexD)
        ]

        if verbose:
            print(
                f'Added perforation for well {well.name} to block {res_cell_idx:d} with WI={well_index:f}, WID={well_indexD:f}'
            )
        return

    def find_cell_index(self, coord: list | np.ndarray) -> int:
        """
        Function to find nearest cell to specified coordinate

        :param coord: XYZ-coordinates
        :type coord: list or np.ndarray
        :returns: Index of cell
        :rtype: int
        """
        min_dis = None
        idx = None
        for j, centroid in enumerate(self.discretizer.centroids_all_cells):
            dis = np.linalg.norm(np.array(coord) - centroid)
            if (min_dis is not None and dis < min_dis) or min_dis is None:
                min_dis = dis
                idx = j
        return idx

    def init_vtk(self, output_directory: str, export_grid_data: bool = True):
        """
        Method to initialize objects required for output of unstructured reservoir into `.vtk` format.
        This method can also export the mesh properties, e.g. porosity, permeability, etc.
        Matrix and fracture cells are written to separate VTK files.

        :param output_directory: Path for output
        :type output_directory: str
        :param export_grid_data: Switch for mesh properties output, default is True
        :type export_grid_data: bool
        """

        self.vtk_filenames_and_times = {}
        self.vtk_filenames_and_times_frac = {}

        self.vtk_initialized = True
        self.discretizer.find_vtk_output_cells()

        if export_grid_data:
            mesh_geom_dtype = np.float32
            matrix_props = {
                "poro": self.poro,
                "permx": self.permx,
                "permy": self.permy,
                "permz": self.permz,
                "hcap": self.hcap,
                "rcond": self.rcond,
                "op_num": self.op_num,
            }
            # order of values in volume_all_cells: FRACTURE MATRIX
            matrix_props["volume"] = np.array(self.mesh.volume, copy=False)
            # order of values in depth_all_cells: FRACTURE MATRIX BOUNDARY
            matrix_props['depth'] = np.array(self.mesh.depth, copy=False)
            matrix_props['center_x'] = self.discretizer.centroids_all_cells[:, 0]
            matrix_props['center_y'] = self.discretizer.centroids_all_cells[:, 1]
            matrix_props['center_z'] = self.discretizer.centroids_all_cells[:, 2]

            # --- Matrix mesh file ---
            mat_nodes = self.discretizer.vtk_output_nodes_to_cells["matrix"]
            mat_idxs = self.discretizer.vtk_output_cell_idxs["matrix"]
            n_frac = self.discretizer.frac_cells_tot
            mat_cell_data = {key: [] for key in matrix_props}
            for prop, data in matrix_props.items():
                for _geometry, cell_idxs in mat_idxs.items():
                    if np.isscalar(data):
                        if type(data) is int:
                            mat_cell_data[prop].append(
                                (data * np.ones(len(cell_idxs))).tolist()
                            )
                        elif type(data) is float:
                            mat_cell_data[prop].append(
                                (
                                    data
                                    * np.ones(len(cell_idxs), dtype=mesh_geom_dtype)
                                ).tolist()
                            )
                    else:
                        # cell_idxs are solution-vector indices [frac..frac+mat-1].
                        # Combined (frac+mat) arrays (volume, depth, centroid) are
                        # indexed directly. Matrix-only arrays (poro, permx, …) have
                        # length mat_cells_tot and need 0-based local indices.
                        if len(data) == self.discretizer.mat_cells_tot:
                            local_idxs = np.array(cell_idxs) - n_frac
                        else:
                            local_idxs = cell_idxs
                        mat_cell_data[prop].append(data[local_idxs].tolist())

            print('Writing mesh data to VTK file')
            meshio.write(
                f"{output_directory:s}/mesh.vtk",
                meshio.Mesh(
                    points=self.discretizer.mesh_data.points,
                    cells=mat_nodes,
                    cell_data=mat_cell_data,
                ),
            )

            # --- Fracture mesh file ---
            if self.discretizer.frac_cells_tot:
                frac_props = self.frac_property_array  # filled in output.py
                frac_nodes = self.discretizer.vtk_output_nodes_to_cells["fracture"]
                frac_idxs = self.discretizer.vtk_output_cell_idxs["fracture"]
                frac_cell_data = {key: [] for key in frac_props}
                for prop, data in frac_props.items():
                    for _geometry, cell_idxs in frac_idxs.items():
                        if np.isscalar(data):
                            if type(data) is int:
                                frac_cell_data[prop].append(
                                    (data * np.ones(len(cell_idxs))).tolist()
                                )
                            elif type(data) is float:
                                frac_cell_data[prop].append(
                                    (
                                        data
                                        * np.ones(len(cell_idxs), dtype=mesh_geom_dtype)
                                    ).tolist()
                                )
                        else:
                            frac_cell_data[prop].append(
                                data.flatten()[cell_idxs].tolist()
                            )

                meshio.write(
                    f"{output_directory:s}/mesh_frac.vtk",
                    meshio.Mesh(
                        points=self.discretizer.mesh_data.points,
                        cells=frac_nodes,
                        cell_data=frac_cell_data,
                    ),
                )

    def output_to_vtk(
        self,
        ith_step: int,
        t: float,
        output_directory: str,
        prop_names: list,
        data: dict,
    ):
        """
        Function to export reservoir results of unstructured reservoir at timestamp t into `.vtk` format.
        Matrix and fracture cells are written to separate VTK files.

        :param ith_step: i'th reporting step
        :type ith_step: int
        :param t: Current time [days]
        :type t: float
        :param output_directory: Path to save .vtk file
        :type output_directory: str
        :param prop_names: List of keys for properties
        :type prop_names: list
        :param data: Data for output
        :type data: nd.array
        """
        from pyevtk.vtk import VtkGroup

        mesh_geom_dtype = np.float32
        frac_props = self.frac_property_array

        # First check if output directory already exists:
        os.makedirs(output_directory, exist_ok=True)
        if not self.vtk_initialized:
            self.init_vtk(output_directory, export_grid_data=True)

        print(f'Writing data to VTK file for {ith_step:d}-th reporting step')

        # --- Matrix file ---
        mat_nodes = self.discretizer.vtk_output_nodes_to_cells["matrix"]
        mat_idxs = self.discretizer.vtk_output_cell_idxs["matrix"]
        mat_cell_data = {prop_names[prop]: [] for prop in prop_names}
        for i, prop in enumerate(prop_names):
            for _geometry, cell_idxs in mat_idxs.items():
                mat_cell_data[prop_names[prop]].append(data[i][cell_idxs])

        vtk_mat_file = output_directory + f'/solution_ts{ith_step}.vtu'
        meshio.write(
            vtk_mat_file,
            meshio.Mesh(
                points=self.discretizer.mesh_data.points,
                cells=mat_nodes,
                cell_data=mat_cell_data,
            ),
        )
        self.vtk_filenames_and_times[vtk_mat_file] = t
        vtk_group = VtkGroup(os.path.join(output_directory, "solution"))
        for fname, sim_t in self.vtk_filenames_and_times.items():
            vtk_group.addFile(fname, sim_t)
        vtk_group.save()

        # --- Fracture file ---
        if self.discretizer.frac_cells_tot:
            frac_nodes = self.discretizer.vtk_output_nodes_to_cells["fracture"]
            frac_idxs = self.discretizer.vtk_output_cell_idxs["fracture"]

            # simulation props for fracture cells
            frac_cell_data = {prop_names[prop]: [] for prop in prop_names}
            for i, prop in enumerate(prop_names):
                for _geometry, cell_idxs in frac_idxs.items():
                    frac_cell_data[prop_names[prop]].append(data[i][cell_idxs])

            # static fracture-specific properties
            for prop, fdata in frac_props.items():
                frac_cell_data[prop] = []
                for _geometry, cell_idxs in frac_idxs.items():
                    if np.isscalar(fdata):
                        if type(fdata) is int:
                            frac_cell_data[prop].append(
                                (fdata * np.ones(len(cell_idxs))).tolist()
                            )
                        elif type(fdata) is float:
                            frac_cell_data[prop].append(
                                (
                                    fdata
                                    * np.ones(len(cell_idxs), dtype=mesh_geom_dtype)
                                ).tolist()
                            )
                    else:
                        frac_cell_data[prop].append(fdata.flatten()[cell_idxs].tolist())

            vtk_frac_file = output_directory + f'/solution_frac_ts{ith_step}.vtu'
            meshio.write(
                vtk_frac_file,
                meshio.Mesh(
                    points=self.discretizer.mesh_data.points,
                    cells=frac_nodes,
                    cell_data=frac_cell_data,
                ),
            )
            self.vtk_filenames_and_times_frac[vtk_frac_file] = t
            vtk_group_frac = VtkGroup(os.path.join(output_directory, "solution_frac"))
            for fname, sim_t in self.vtk_filenames_and_times_frac.items():
                vtk_group_frac.addFile(fname, sim_t)
            vtk_group_frac.save()
