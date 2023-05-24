from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector
import numpy as np
from math import inf, pi
from darts.mesh.unstruct_discretizer import UnstructDiscretizer
from itertools import compress


# Definitions for the unstructured reservoir class:
class UnstructReservoir:
    def __init__(self, permx, permy, permz, frac_aper, mesh_file, poro):
        """
        Class constructor for UnstructReservoir class
        :param permx: Matrix permeability in the x-direction (scalar or vector)
        :param permy: Matrix permeability in the y-direction (scalar or vector)
        :param permz: Matrix permeability in the z-direction (scalar or vector)
        :param frac_aper: Aperture of the fracture (scalar or vector)
        :param mesh_file: Name and relative path to the mesh-file (string)
        :param poro: Matrix (and fracture?) porosity (scalar or vector)
        :param bound_cond: switch which determines the type of boundary conditions used (string)
        """
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()

        # Specify well index and store matrix geometry:
        self.file_path = mesh_file

        # Construct instance of Unstructured Discretization class:
        self.unstr_discr = UnstructDiscretizer(permx=permx, permy=permy, permz=permz, frac_aper=frac_aper,
                                               mesh_file=mesh_file)

        # Use class method load_mesh to load the GMSH file specified above:
        self.unstr_discr.load_mesh()

        # Calculate cell information of each geometric element in the .msh file:
        self.unstr_discr.calc_cell_information()

        # Store volumes and depth to single numpy arrays:
        self.unstr_discr.store_volume_all_cells()
        self.unstr_discr.store_depth_all_cells()
        self.unstr_discr.store_centroid_all_cells()

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.unstr_discr.calc_connections_all_cells()

        # Write to files (in case someone needs this for Eclipse or other simulator):
        # self.unstr_discr.write_conn2p_therm_to_file(cell_m, cell_p, tran, tran_thermal, file_name='conn2p.dat')
        # self.unstr_discr.write_volume_to_file(file_name='vol.dat')
        # self.unstr_discr.write_depth_to_file(file_name='depth.dat')

        # Initialize mesh using built connection list
        self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.unstr_discr.volume_all_cells.size
        self.num_frac = self.unstr_discr.fracture_cell_count
        self.num_mat = self.unstr_discr.matrix_cell_count

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.depth = np.array(self.mesh.depth, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)

        # rock thermal properties
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.conduction = np.array(self.mesh.rock_cond, copy=False)

        # Since we use copy==False above, we have to store the values by using the Python slicing option, if we don't
        # do this we will overwrite the variable, e.g. self.poro = poro --> overwrite self.poro with the variable poro
        # instead of storing the variable poro in self.mesh.poro (therefore "numpy array wrapped around mesh data!!!):
        self.poro[:] = poro
        self.depth[:] = self.unstr_discr.depth_all_cells
        self.volume[:] = self.unstr_discr.volume_all_cells

        # Calculate well_index (very primitive way....):
        self.well_index = np.mean(tran) * 1
        # self.well_index = 10

        # Set-up dictionary with data for boundary cells:
        boundary_data = dict()  # Dictionary containing boundary condition data (coordinate and value of boundary):
        boundary_data['first_boundary_dir'] = 'X'  # Indicates the boundary is located at constant X (in this case!)
        # Constant X-coordinate value at which the boundary is located (used to be 3.40885):
        boundary_data['first_boundary_val'] = np.min(self.unstr_discr.mesh_data.points[:, 0])

        # Same as above but for the second boundary condition!
        boundary_data['second_boundary_dir'] = 'X'
        # Constant X-coordinate value at which the boundary is located (used to be 13.0014):
        boundary_data['second_boundary_val'] = np.max(self.unstr_discr.mesh_data.points[:, 0])

        # Calculate boundary cells using the calc_boundary_cells method:
        self.left_boundary_cells, self.right_boundary_cells = self.unstr_discr.calc_boundary_cells(boundary_data)

        # Calc maximum size of well cells (used to have more homogeneous injection conditions by scaling the WI):
        dummy_vol = np.array(self.volume, copy=True)
        self.max_well_vol = np.max([np.max(dummy_vol[self.left_boundary_cells]),
                                    np.max(dummy_vol[self.right_boundary_cells])])

        self.volume[self.right_boundary_cells] = self.volume[self.right_boundary_cells]*1e8

        # Create empty list of wells:
        self.wells = []

    def add_well(self, name, depth, wellbore_diameter):
        """
        Class method which adds wells heads to the reservoir (Note: well head is not equal to a perforation!)
        :param name:
        :param depth:
        :param wellbore_diameter:
        :return:
        """
        well = ms_well()
        well.name = name
        well.segment_volume = pi * wellbore_diameter ** 2 / 4
        well.well_head_depth = 0
        well.well_body_depth = 0
        well.segment_transmissibility = 1e5
        well.segment_depth_increment = 0
        self.wells.append(well)
        return 0

    def add_perforation(self, well, res_block, well_index):
        """
        Class method which ads perforation to each (existing!) well
        :param well: data object which contains data of the particular well
        :param res_block: reservoir block in which the well has a perforation
        :param well_index: well index (productivity index)
        :return:
        """
        well_block = 0
        well.perforations = well.perforations + [(well_block, res_block, well_index)]
        return 0

    def init_wells(self):
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """
        # Add injection well for CO2:
        self.add_well("I1", depth=5, wellbore_diameter=0.1)
        # Perforate all boundary cells:
        for nth_perf in range(len(self.left_boundary_cells)):
            well_index = self.mesh.volume[self.left_boundary_cells[nth_perf]] / self.max_well_vol * self.well_index
            self.add_perforation(well=self.wells[-1], res_block=self.left_boundary_cells[nth_perf],
                                 well_index=well_index)

        # Add wells to the DARTS mesh object and sort connection (DARTS related):
        self.mesh.add_wells(ms_well_vector(self.wells))
        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()
        return 0