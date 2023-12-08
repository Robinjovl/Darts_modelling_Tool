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
    physical_tags = {'matrix': [], 'boundary': [], 'fracture': [], 'fracture_shape': [], 'output': []}

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

        # Create empty list of wells:
        self.wells = []

    def discretize(self, verbose: bool = False) -> None:
        # Construct instance of Unstructured Discretization class:
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=self.physical_tags,
                                               permx=self.permx, permy=self.permy, permz=self.permz,
                                               frac_aper=self.frac_aper, verbose=verbose)

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh()

        # Calculate cell information of each geometric element in the .msh file:
        self.discretizer.calc_cell_information()

        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroid_all_cells()

        # Assign layer properties
        self.set_layer_properties()

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_connections_all_cells()

        # Initialize mesh using built connection list
        self.mesh = conn_mesh()
        self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(self.mesh.poro, copy=False)[:] = self.poro
        np.array(self.mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(self.mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(self.mesh.op_num, copy=False)[:] = self.op_num
        np.array(self.mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells
        np.array(self.mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        return

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

    def calc_boundary_cells(self, boundary_data):
        """
        Class method which calculates constant boundary values at a specif constant x,y,z-coordinate

        :param boundary_data: dictionary with the boundary location (X,Y,Z, and location)
        :return:
        """
        # Specify boundary cells, simply set specify the single coordinate which is not-changing and its value:
        # First boundary:
        index = []  # Dynamic list containing indices of the nodes (points) which lay on the boundary:
        if boundary_data['first_boundary_dir'] == 'X':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 0] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 1] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 2] == boundary_data['first_boundary_val']

        # Convert dynamic list to numpy array:
        left_boundary_points = np.array(list(compress(range(len(index)), index)))

        # Second boundary (same as above):
        index = []
        if boundary_data['second_boundary_dir'] == 'X':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 0] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 1] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 2] == boundary_data['second_boundary_val']

        right_boundary_points = np.array(list(compress(range(len(index)), index)))

        # Find cells containing boundary cells, for wedges or hexahedrons, the boundary cells must contain,
        # on the X or Y boundary four nodes exactly!
        #     0------0          0
        #    /     / |         /  \
        #  0------0  0        0----0
        #  |      | /         |    |
        #  0------0           0----0
        # Hexahedron       Wedge (prism)
        # Create loop over all matrix cells which are of the geometry 'matrix_cell_type'
        left_count = 0  # Counter for number of left matrix cells on the boundary
        left_boundary_cells = {}  # Dictionary with matrix cells on the left boundary
        for geometry in self.discretizer.geometries_in_mesh_file:
            if geometry in self.discretizer.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the left_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.discretizer.mesh_data.cells_dict[geometry]):

                    if len(set.intersection(set(ith_row), set(left_boundary_points))) == 4:
                        # Store cell since it is on the left boundary:
                        left_boundary_cells[left_count] = ith_cell
                        left_count += 1

        right_count = 0
        right_boundary_cells = {}
        for geometry in self.discretizer.geometries_in_mesh_file:
            if geometry in self.discretizer.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the right_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.discretizer.mesh_data.cells_dict[geometry]):
                    if len(set.intersection(set(ith_row), set(right_boundary_points))) == 4:
                        # Store cell since it is on the left boundary:
                        right_boundary_cells[right_count] = ith_cell
                        right_count += 1

        self.left_boundary_cells = np.array(list(left_boundary_cells.values()), dtype=int) + \
                                   self.discretizer.fracture_cell_count
        self.right_boundary_cells = np.array(list(right_boundary_cells.values()), dtype=int) + \
                                    self.discretizer.fracture_cell_count
        return 0

    def add_perforation(self, well_name: str, cell_index: Union[int, tuple], well_radius: float = 0.1524,
                        well_index: float = None, well_indexD: float = None, segment_direction: str = 'z_axis',
                        skin: float = 0, multi_segment: bool = False, verbose: bool = False):
        """
        Function to add perforations to wells.
        """
        well = self.get_well(well_name)

        if cell_index in well.perforations:
            print('There are at least 2 wells locating in the same grid block!!! The mesh file should be modified!')
            exit()

        # well.well_head_depth = np.array(self.mesh.depth, copy=False)[cell_index]
        # well.well_body_depth = well.well_head_depth

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

    def output_to_vtk(self, output_directory: str, output_filename: str, property_data: dict, ith_step: int):
        """
        Class method which writes output of unstructured grid to VTK format

        :param output_directory: directory of output files
        :param property_array: np.array containing all cell properties (N_cells x N_prop)
        :param cell_property: list with property names (visible in ParaView (format strings)
        :param ith_step: integer containing the output step
        :return:
        """
        # First check if output directory already exists:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Allocate empty new cell_data_dict dictionary:
        cell_data_dict = dict()

        for ith_prop in range(len(cell_property)):
            cell_data_dict[cell_property[ith_prop]] = []
            left_bound = 0
            right_bound = 0
            for ith_geometry in self.mesh_data.cells_dict:
                left_bound = right_bound
                right_bound = right_bound + self.mesh_data.cells_dict[ith_geometry].shape[0]
                cell_data_dict[cell_property[ith_prop]].append(list(property_array[left_bound:right_bound, ith_prop]))

        cell_data_dict['matrix_cell_bool'] = []
        left_bound = 0
        right_bound = 0
        for ith_geometry in self.mesh_data.cells_dict:
            left_bound = right_bound
            right_bound = right_bound + self.mesh_data.cells_dict[ith_geometry].shape[0]

            if (ith_geometry in self.available_fracture_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.zeros(((right_bound - left_bound),))))

            elif (ith_geometry in self.available_matrix_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.ones(((right_bound - left_bound),))))

        # Temporarily store mesh_data in copy:
        # Mesh = meshio.read(self.mesh_file)

        mesh = meshio.Mesh(
            # Mesh.points,
            # Mesh.cells_dict.items(),
            self.mesh_data.points,  # list of point coordinates
            self.mesh_data.cells_dict.items(),  # list of
            # Each item in cell data must match the cells array
            cell_data=cell_data_dict)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)
        return 0
