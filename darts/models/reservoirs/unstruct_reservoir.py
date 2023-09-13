

class UnstructReservoir:

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
            index = self.unstr_discr.mesh_data.points[:, 0] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.unstr_discr.mesh_data.points[:, 1] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.unstr_discr.mesh_data.points[:, 2] == boundary_data['first_boundary_val']

        # Convert dynamic list to numpy array:
        left_boundary_points = np.array(list(compress(range(len(index)), index)))

        # Second boundary (same as above):
        index = []
        if boundary_data['second_boundary_dir'] == 'X':
            # Check if first coordinate of points is on the boundary:
            index = self.unstr_discr.mesh_data.points[:, 0] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.unstr_discr.mesh_data.points[:, 1] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.unstr_discr.mesh_data.points[:, 2] == boundary_data['second_boundary_val']

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
        for geometry in self.unstr_discr.geometries_in_mesh_file:
            if geometry in self.unstr_discr.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the left_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.unstr_discr.mesh_data.cells_dict[geometry]):

                    if len(set.intersection(set(ith_row), set(left_boundary_points))) == 4:
                        # Store cell since it is on the left boundary:
                        left_boundary_cells[left_count] = ith_cell
                        left_count += 1

        right_count = 0
        right_boundary_cells = {}
        for geometry in self.unstr_discr.geometries_in_mesh_file:
            if geometry in self.unstr_discr.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the right_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.unstr_discr.mesh_data.cells_dict[geometry]):
                    if len(set.intersection(set(ith_row), set(right_boundary_points))) == 4:
                        # Store cell since it is on the left boundary:
                        right_boundary_cells[right_count] = ith_cell
                        right_count += 1

        self.left_boundary_cells = np.array(list(left_boundary_cells.values()), dtype=int) + \
                                   self.unstr_discr.fracture_cell_count
        self.right_boundary_cells = np.array(list(right_boundary_cells.values()), dtype=int) + \
                                    self.unstr_discr.fracture_cell_count
        return 0

    def add_well(self, name, depth):
        """
        Class method which adds wells heads to the reservoir (Note: well head is not equal to a perforation!)

        :param name:
        :param depth:
        :return:
        """
        well = ms_well()
        well.name = name
        well.segment_volume = 0.0785 * 40  # 2.5 * pi * 0.15**2 / 4
        well.well_head_depth = depth
        well.well_body_depth = depth
        well.segment_transmissibility = 1e5
        well.segment_depth_increment = 1
        self.wells.append(well)
        return 0

    def add_perforation(self, well, res_block, well_index=-1, well_indexD=-1, well_radius=0.1524, skin=0.,
                        multi_segment=True, verbose=False):
        """
        Class method which ads perforation to each (existing!) well

        :param well: data object which contains data of the particular well
        :param res_block: reservoir block in which the well has a perforation
        :param well_index: well index (productivity index)
        :return:
        """
        
        # calculate well index and get local index of reservoir block
        wi, wid = self.unstr_discr.calc_equivalent_well_index(res_block, well_radius=well_radius, skin=skin)

        if well_index == -1: 
            well_index = wi

        if well_indexD == -1:
            well_indexD = wid
        
        # set well segment index (well block) equal to index of perforation layer
        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0
            
        well.perforations = well.perforations + [(well_block, res_block, well_index, well_indexD)]
        
        if verbose:
            print('Added perforation for well %s to block %d with WI=%f' % (well.name, res_block, well_index))
            
        return 0
