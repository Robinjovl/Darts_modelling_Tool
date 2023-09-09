from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector
import numpy as np
from math import inf, pi
from darts.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.models.reservoirs.unstruct_reservoir import UnstructReservoir
import sys

# Definitions for the unstructured reservoir class:
class UnstructReservoirBrugge(UnstructReservoir):
    def __init__(self, permx, permy, permz, frac_aper, mesh_file, poro, thickness, calc_equiv_WI=True):
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
        self.calc_equiv_WI = calc_equiv_WI
        self.thickness = thickness

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
        self.cac = self.unstr_discr.centroid_all_cells

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.unstr_discr.calc_connections_all_cells()

        # # Write to files (in case someone needs this for Eclipse or other simulator):
        # self.unstr_discr.write_conn2p_to_file(cell_m, cell_p, tran, file_name='conn2p.dat')
        # self.unstr_discr.write_conn2p_therm_to_file(cell_m, cell_p, tran, tran_thermal, file_name='conn2p.dat.connsn')
        # self.unstr_discr.write_volume_to_file(file_name='vol.dat')
        # self.unstr_discr.write_depth_to_file(file_name='depth.dat')

        # Initialize mesh with all four parameters (cell_m, cell_p, trans, trans_D):
        self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))
        # self.mesh.init('conn2p.dat.connsn')

        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.unstr_discr.volume_all_cells.size
        self.num_frac = self.unstr_discr.fracture_cell_count
        self.num_mat = self.unstr_discr.matrix_cell_count

        self.permx = permx * np.ones(self.nb)
        self.permy = permy * np.ones(self.nb)

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.depth = np.array(self.mesh.depth, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)

        # Since we use copy==False above, we have to store the values by using the Python slicing option, if we don't
        # do this we will overwrite the variable, e.g. self.poro = poro --> overwrite self.poro with the variable poro
        # instead of storing the variable poro in self.mesh.poro (therefore "numpy array wrapped around mesh data!!!):
        self.poro[:] = poro
        self.depth[:] = self.unstr_discr.depth_all_cells
        self.volume[:] = self.unstr_discr.volume_all_cells

        # Create empty list of wells:
        self.wells = []



    def add_well(self, name, wellbore_diameter=0.15):
        """
        Class method which adds wells heads to the reservoir (Note: well head is not equal to a perforation!)
        :param name:
        :param wellbore_diameter:
        :return:
        """
        well = ms_well()
        well.name = name
        # so far put only area here,
        # to be multiplied by segment length later

        well.segment_volume = pi * wellbore_diameter ** 2 / 4

        # also to be filled up when the first perforation is made
        well.well_head_depth = 0
        well.well_body_depth = 0
        well.segment_depth_increment = 0
        self.wells.append(well)
        return well



    def init_wells(self):
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """
        well_coord = np.genfromtxt('Brugge_struct/well_coord_Brugge.txt')
        self.index_cell = []

        for i, wc in enumerate(well_coord):
            distance = []
            for j, centroid in enumerate(self.cac):
                distance.append(np.linalg.norm(wc - centroid))
            min_dis = np.min(distance)
            self.index_cell.append(distance.index(min_dis))

        try:
            assert (len(set(self.index_cell)) == len(well_coord))
        except AssertionError:
            print('There are at least 2 wells locating in the same grid block!!! The mesh file should be modified!')
            sys.exit()

        self.well_index_list = []
        if self.calc_equiv_WI:
            self.well_index_list = [-1] * len(well_coord)
        else:
            self.well_index_list = [296.65303668, 69.71905642, 27.14929434, 27.58575654, 53.01869826,
                                    135.80457602, 345.34715322, 80.69146768, 74.07293499, 243.34286931,
                                    482.48726884, 592.37938461, 307.72095815, 542.44279506, 63.58456305,
                                    499.53586907, 213.09805386, 303.97957677, 78.80966839, 791.4220401,
                                    829.44984229, 794.13401768, 761.08006179, 62.37906275, 616.74501491,
                                    475.63754963, 397.50862698, 478.21742722, 504.06328513, 655.32259614]


        n_injector = 10  # the first 10 wells are injectors
        n_wells = 30  # the number of wells
        self.inj_wells = []
        self.prod_wells = []

        for i in range(np.size(self.well_index_list)):
            if i < n_injector:
                self.add_well("I" + str(i + 1))
                self.add_perforation(well=self.wells[-1], res_block=int(self.index_cell[i]),
                                     well_index=self.well_index_list[i], well_indexD=0,
                                     multi_segment=True, verbose=True)
                self.inj_wells.append(self.wells[i])
            else:
                self.add_well("P" + str(i + 1 - n_injector))
                self.add_perforation(well=self.wells[-1], res_block=int(self.index_cell[i]),
                                     well_index=self.well_index_list[i], well_indexD=0,
                                     multi_segment=True, verbose=True)
                self.prod_wells.append(self.wells[i])

        # Add wells to the DARTS mesh object and sort connection (DARTS related):
        self.mesh.add_wells(ms_well_vector(self.wells))
        self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()
        return 0


