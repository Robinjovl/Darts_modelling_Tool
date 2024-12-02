from darts.reservoirs.reservoir_base import ReservoirBase
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.engines import timer_node, ms_well, conn_mesh, value_vector, index_vector
import numpy as np
import os
import meshio
from typing import Union


class UnstructReservoir(ReservoirBase):
    """
    Class for generating unstructered mesh

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
    physical_tags = {'matrix': [], 'fracture': [], 'boundary': [], 'fracture_boundary': [], 'output': []}

    def __init__(self, timer: timer_node, mesh_file: str, permx, permy, permz, poro, rcond=0, hcap=0,
                 frac_aper=0, op_num=0, cache: bool = False):
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

        # parameters for optional fracture aperture computation depending on principal stresses
        self.sh_max = None
        self.sh_min = None
        self.sh_max_azimuth = None
        self.sigma_c = None

    def discretize(self, verbose: bool = False) -> conn_mesh:
        # Construct instance of Unstructured Discretization class:
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=self.physical_tags,
                                               verbose=verbose)

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=self.frac_aper,
                                   cache=False)

        if self.frac_aper is not None and self.sh_max is not None:
            self.discretizer.calc_frac_aper_by_stress(self.frac_aper, self.sh_max, self.sh_min, self.sh_max_azimuth,
                                                      self.sigma_c)

        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroid_all_cells()

        # Assign layer properties
        self.set_layer_properties()

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_connections_all_cells()

        # Initialize mesh using built connection list
        mesh = conn_mesh()
        mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.op_num, copy=False)[:] = self.op_num
        n_elements = self.discretizer.mat_cells_tot + self.discretizer.frac_cells_tot
        np.array(mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells[:n_elements]
        np.array(mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells[:n_elements]

        return mesh

    def set_boundary_volume(self, boundary_volumes: dict):
        # Set-up dictionary with data for boundary cells:
        boundary_data = dict()  # Dictionary containing boundary condition data (coordinate and value of boundary):
        boundary_data['first_boundary_dir'] = 'X'  # Indicates the boundary is located at constant X (in this case!)
        # Constant X-coordinate value at which the boundary is located (used to be 3.40885):
        boundary_data['first_boundary_val'] = np.min(self.discretizer.mesh_data.points[:, 0])

        # Same as above but for the second boundary condition!
        boundary_data['second_boundary_dir'] = 'X'
        # Constant X-coordinate value at which the boundary is located (used to be 13.0014):
        boundary_data['second_boundary_val'] = np.max(self.discretizer.mesh_data.points[:, 0])

        # Calculate boundary cells using the calc_boundary_cells method:
        self.left_boundary_cells, self.right_boundary_cells = self.discretizer.calc_boundary_cells(boundary_data)

        # Calc maximum size of well cells (used to have more homogeneous injection conditions by scaling the WI):
        dummy_vol = np.array(self.volume, copy=True)
        self.max_well_vol = np.max([np.max(dummy_vol[self.left_boundary_cells]),
                                    np.max(dummy_vol[self.right_boundary_cells])])

        self.volume[self.right_boundary_cells] = self.volume[self.right_boundary_cells] * 1e8
        return

    def add_perforation(self, well_name: str, cell_index: int, well_radius: float = 0.1524,
                        well_index: float = None, well_indexD: float = None, segment_direction: str = 'z_axis',
                        skin: float = 0, multi_segment: bool = False, verbose: bool = False):
        """
        Function to add perforations to wells.
        """
        well = self.get_well(well_name)

        perf_indices = np.array(well.perforations, dtype=int)
        # cell_index has index=1 in perforation element: (well_block, cell_index, well_index, well_indexD)
        perf_indices = perf_indices[:, 1] if len(well.perforations) > 0 else []
        if cell_index in perf_indices:
            print('There are at least 2 wells locating in the same grid block!!! The mesh file should be modified!')
            exit()

        #  update well depth
        perf_indices = np.append(perf_indices, cell_index).astype(int)  # add current cell to previous perforation list
        # set well depth to the top perforation depth 
        well.well_head_depth = np.array(self.mesh.depth, copy=False)[perf_indices].min()  
        well.well_body_depth = well.well_head_depth

        if well_index is None or well_indexD is None:
            # calculate well index and get local index of reservoir block
            wi, wid = self.discretizer.calc_equivalent_well_index(cell_index, well_radius, skin)
            well_index = wi if well_index is None else well_index
            well_indexD = wid if well_indexD is None else well_indexD

        assert well_index >= 0
        assert well_indexD >= 0

        # set well segment index (well block) equal to index of perforation layer
        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0

        well.perforations = well.perforations + [(well_block, cell_index, well_index, well_indexD)]

        if verbose:
            print('Added perforation for well %s to block %d with WI=%f, WID=%f' % (well.name, cell_index,
                                                                                    well_index, well_indexD))
        return

    def find_cell_index(self, coord: Union[list, np.ndarray]) -> int:
        """
        Function to find nearest cell to specified coordinate

        :param coord: XYZ-coordinates
        :type coord: list or np.ndarray
        :returns: Index of cell
        :rtype: int
        """
        min_dis = None
        idx = None
        for j, centroid in enumerate(self.discretizer.centroid_all_cells):
            dis = np.linalg.norm(np.array(coord) - centroid)
            if (min_dis is not None and dis < min_dis) or min_dis is None:
                min_dis = dis
                idx = j
        return idx

    def init_vtk(self, output_directory: str, export_grid_data: bool = True):
        """
        Method to initialize objects required for output of unstructured reservoir into `.vtk` format.
        This method can also export the mesh properties, e.g. porosity, permeability, etc.

        :param output_directory: Path for output
        :type output_directory: str
        :param export_grid_data: Switch for mesh properties output, default is True
        :type export_grid_data: bool
        """
        self.vtk_initialized = True
        self.discretizer.find_vtk_output_cells()

        if export_grid_data:
            mesh_geom_dtype = np.float32
            matrix_props = {'poro': self.poro, 'permx': self.permx, 'permy': self.permy, 'permz': self.permz,
                            'hcap': self.hcap, 'rcond': self.rcond, 'op_num': self.op_num,
                            }
            # order of values in volume_all_cells: FRACTURE MATRIX
            matrix_props['volume'] = np.array(self.mesh.volume, copy=False)
            # order of values in depth_all_cells: FRACTURE MATRIX BOUNDARY
            matrix_props['depth'] = np.array(self.mesh.depth, copy=False)
            matrix_props['center_x'] = self.discretizer.centroid_all_cells[:, 0]
            matrix_props['center_y'] = self.discretizer.centroid_all_cells[:, 1]
            matrix_props['center_z'] = self.discretizer.centroid_all_cells[:, 2]
            frac_props = {'frac_aper': self.frac_aper}

            # Create empty lists for each geometry type - {**{}} operator merges dictionaries
            output_nodes = self.discretizer.vtk_output_nodes_to_cells['matrix'] if not self.discretizer.frac_cells_tot \
                else {**self.discretizer.vtk_output_nodes_to_cells['fracture'], **self.discretizer.vtk_output_nodes_to_cells['matrix']}
            output_idxs = self.discretizer.vtk_output_cell_idxs
            geometries = output_nodes.keys()
            props = {**matrix_props, **frac_props}.keys() if self.discretizer.frac_cells_tot else matrix_props.keys()
            cell_data = {key: [[] for geometry in geometries] for key in props}

            # Loop over matrix cell properties
            for prop, data in matrix_props.items():
                ith_geometry = 0

                # Fill fracture cells with zeros
                for geometry, cell_idxs in output_idxs['fracture'].items():
                    cell_data[prop][ith_geometry] += [0.] * len(cell_idxs)
                    ith_geometry += 1
                # Fill matrix cells with data
                for geometry, cell_idxs in output_idxs['matrix'].items():
                    if np.isscalar(data):
                        if type(data) is int:
                            cell_data[prop][ith_geometry] += (data * np.ones(len(cell_idxs))).tolist()
                        elif type(data) is float:
                            cell_data[prop][ith_geometry] += (data * np.ones(len(cell_idxs), dtype=mesh_geom_dtype)).tolist()
                    else:
                        cell_data[prop][ith_geometry] += data[cell_idxs].tolist()
                    ith_geometry += 1

            # Loop over fracture cell properties
            if self.discretizer.frac_cells_tot:
                for prop, data in frac_props.items():
                    ith_geometry = 0

                    # Fill fracture cells with data
                    for geometry, cell_idxs in output_idxs['fracture'].items():
                        if np.isscalar(data):
                            if type(data) is int:
                                cell_data[prop][ith_geometry] += (data * np.ones(len(cell_idxs))).tolist()
                            elif type(data) is float:
                                cell_data[prop][ith_geometry] += (data * np.ones(len(cell_idxs), dtype=mesh_geom_dtype)).tolist()
                        else:
                            cell_data[prop][ith_geometry] += data[cell_idxs].tolist()
                        ith_geometry += 1
                    # Fill matrix cells with zeros
                    for geometry, cell_idxs in output_idxs['matrix'].items():
                        cell_data[prop][ith_geometry] += [0.] * len(cell_idxs)
                        ith_geometry += 1

            # Distinguish fracture cells from matrix cells
            cell_data['matrix_cell_bool'] = [[] for geometry in geometries]
            ith_geometry = 0
            for geometry, cell_idxs in self.discretizer.vtk_output_cell_idxs['fracture'].items():
                cell_data['matrix_cell_bool'][ith_geometry] += np.zeros(
                    len(cell_idxs)).tolist()  # fill fracture cells with zeros
                ith_geometry += 1
            for geometry, cell_idxs in self.discretizer.vtk_output_cell_idxs['matrix'].items():
                cell_data['matrix_cell_bool'][ith_geometry] += np.ones(
                    len(cell_idxs)).tolist()  # fill matrix cells with ones
                ith_geometry += 1

            mesh = meshio.Mesh(
                points=self.discretizer.mesh_data.points,  # list of point coordinates
                cells=output_nodes,  # list of cell geometries and idxs for reporting
                # Each item in cell data must match the cells array
                cell_data=cell_data
            )

            print('Writing mesh data to VTK file')
            meshio.write("{:s}/mesh.vtk".format(output_directory), mesh)

    def output_to_vtk(self, ith_step: int, time_steps: float, output_directory: str, prop_names: list, data: dict):
        """
        Function to export results of unstructured reservoir at timestamp t into `.vtk` format.

        :param ith_step: i'th reporting step
        :type ith_step: int
        :param t: Current time [days]
        :type t: float
        :param output_directory: Path to save .vtk file
        :type output_directory: str
        :param prop_names: List of keys for properties
        :type prop_names: list
        :param data: Data for output
        :type data: dict
        """
        # First check if output directory already exists:
        os.makedirs(output_directory, exist_ok=True)
        if not self.vtk_initialized:
            self.init_vtk(output_directory, export_grid_data=True)

        # Create empty lists for each geometry type - {**{}} operator merges dictionaries
        output_nodes = self.discretizer.vtk_output_nodes_to_cells['matrix'] if not self.discretizer.frac_cells_tot \
            else {**self.discretizer.vtk_output_nodes_to_cells['fracture'], **self.discretizer.vtk_output_nodes_to_cells['matrix']}
        output_idxs = self.discretizer.vtk_output_cell_idxs['matrix'] if not self.discretizer.frac_cells_tot \
            else {**self.discretizer.vtk_output_cell_idxs['fracture'], **self.discretizer.vtk_output_cell_idxs['matrix']}
        geometries = output_nodes.keys()
        cell_data = {prop: [[] for geometry in geometries] for prop in prop_names}

        # Distinguish fracture cells from matrix cells
        cell_data['matrix_cell_bool'] = [[] for geometry in geometries]
        ith_geometry = 0
        for geometry, cell_idxs in self.discretizer.vtk_output_cell_idxs['fracture'].items():
            cell_data['matrix_cell_bool'][ith_geometry] += np.zeros(len(cell_idxs)).tolist()  # fill fracture cells with zeros
            ith_geometry += 1
        for geometry, cell_idxs in self.discretizer.vtk_output_cell_idxs['matrix'].items():
            cell_data['matrix_cell_bool'][ith_geometry] += np.ones(len(cell_idxs)).tolist()  # fill matrix cells with ones
            ith_geometry += 1

        for ts, t in enumerate(time_steps):
            if len(time_steps) == 1:
                vtk_file_name = output_directory + '/solution_ts{}'.format(ith_step)
            else:
                vtk_file_name = output_directory + '/solution_ts{}'.format(ts)

            # Loop over output properties
            for prop in prop_names:
                # Loop over fracture and matrix cells (in that order)
                for ith_geometry, (geometry, cell_idxs) in enumerate(output_idxs.items()):
                    cell_data[prop][ith_geometry] += data[prop][ts][cell_idxs].tolist()

            # Temporarily store mesh_data in copy:
            mesh = meshio.Mesh(
                points=self.discretizer.mesh_data.points,  # list of point coordinates
                cells=output_nodes,  # list of cell geometries and idxs for reporting
                # Each item in cell data must match the cells array
                cell_data=cell_data
            )

            print('Writing data to VTK file for {:d}-th reporting step'.format(ts))
            meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ts), mesh)
