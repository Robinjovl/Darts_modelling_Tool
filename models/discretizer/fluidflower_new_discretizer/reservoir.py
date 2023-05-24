from darts.engines import conn_mesh, ms_well, ms_well_vector
from scipy.spatial.transform import Rotation
import numpy as np
import meshio
from math import inf, pi
from itertools import compress
import os
import sys
import inspect
from darts.mesh.unstruct_discretizer import UnstructDiscretizer
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import darts.discretizer as dis
from darts.discretizer import Mesh, Elem, Discretizer, BoundaryCondition, elem_loc, elem_type, index_vector, value_vector, matrix33, vector_matrix33, vector_vector3, matrix
import datetime
from dataclasses import dataclass, field
from darts.mesh.transcalc import TransCalculations
import copy

@dataclass
class PorPerm:
    tag: int
    type: str
    poro: float
    perm: float
    anisotropy: list

# Definitions for the unstructured reservoir class:
class UnstructReservoir:
    def __init__(self, discr_type='mpfa', problem_type='reservoir', mpfa_type='mpfa'):
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
        self.discr_type = discr_type
        self.problem_type = problem_type
        self.mpfa_type = mpfa_type
        self.curvature = False
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        self.fluidflower1()

        if discr_type == 'mpfa':
            # self.write_mpfa_conn_to_file('conn_tpfa_mp.dat')
            self.tranD = copy.copy(self.discr.flux_vals)
            tranD = np.array(self.tranD, copy=False)
            tranD *= self.diff_mult
            # self.mesh.init_mpfa(self.discr.cell_m, self.discr.cell_p,
            #                     self.discr.flux_stencil, self.discr.flux_offset,
            #                     self.discr.flux_vals, self.discr.flux_rhs,
            #                     self.n_matrix, self.n_bounds)
            self.mesh.init_mpfa_e(self.discr.cell_m, self.discr.cell_p,
                                self.discr.flux_stencil, self.discr.flux_offset,
                                self.discr.flux_vals, self.tranD, self.discr.flux_rhs,
                                self.n_matrix, self.n_bounds, self.n_fracs)
            self.poro = np.array(self.mesh.poro, copy=False)
            self.depth = np.array(self.mesh.depth, copy=False)
            self.volume = np.array(self.mesh.volume, copy=False)
            self.bc = np.array(self.mesh.bc, copy=False)

            self.poro[:] = self.porosity
            self.depth[:] = self.depth_all_cells
            self.volume[:] = self.volume_all_cells
            self.bc[:self.bc_flow.size] = self.bc_flow
            self.mesh.pz_bounds.resize(2 * self.n_bounds)
            mean_cell_width = np.cbrt(np.mean(self.volume_all_cells[:self.n_matrix]))
            self.min_depth = self.water_column_height - max(self.depth_all_cells[:self.n_matrix])
        else:
            cell_m, cell_p, tran, tran_thermal = self.unstr_discr.calc_connections_all_cells()

            # self.write_mpfa_conn_to_file()
            # self.unstr_discr.write_volume_to_file(file_name='vol.dat')
            # self.unstr_discr.write_depth_to_file(file_name='depth.dat')
            #self.unstr_discr.write_conn2p_to_file(cell_m, cell_p, tran, file_name='conn_tpfa.dat')
            self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))
            self.nb = len(self.mesh.depth)
            self.depth = np.array(self.mesh.depth, copy=False)
            self.volume = np.array(self.mesh.volume, copy=False)
            self.min_depth = self.water_column_height - max(self.unstr_discr.depth_all_cells[:self.nb])
            self.depth[:] = self.water_column_height - self.unstr_discr.depth_all_cells[:self.nb]
            self.volume[:] = self.unstr_discr.volume_all_cells[:self.nb]
            self.poro = np.array(self.mesh.poro, copy=False)
            self.poro[:] = self.cell_poro
            mean_cell_width = np.cbrt(np.mean(self.unstr_discr.volume_all_cells[:self.nb]))

        # rock thermal properties
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.conduction = np.array(self.mesh.rock_cond, copy=False)

        # Since we use copy==False above, we have to store the values by using the Python slicing option, if we don't
        # do this we will overwrite the variable, e.g. self.poro = poro --> overwrite self.poro with the variable poro
        # instead of storing the variable poro in self.mesh.poro (therefore "numpy array wrapped around mesh data!!!):

        # self.pz_bounds[:] = self.pz_bounds

        # Calculate well_index (very primitive way....):
        # rw = 0.1
        # dx = mean_cell_width
        # dy = mean_cell_width
        # dz = mean_cell_width
        # # WIx
        # wi_x = 0.0
        # # WIy
        # wi_y = 0.0
        # # WIz
        # hz = dz
        # mean_perm_xx = 1000#np.mean(np.array([self.discr.perms[cell_id].values[0] for cell_id in self.well_cells]))
        # mean_perm_yy = 1000#np.mean(np.array([self.discr.perms[cell_id].values[4] for cell_id in self.well_cells]))
        # rp_z = 0.28 * np.sqrt((mean_perm_yy / mean_perm_xx) ** 0.5 * dx ** 2 +
        #                       (mean_perm_xx / mean_perm_yy) ** 0.5 * dy ** 2) / \
        #        ((mean_perm_xx / mean_perm_yy) ** 0.25 + (mean_perm_yy / mean_perm_xx) ** 0.25)
        # wi_z = 2 * np.pi * np.sqrt(mean_perm_xx * mean_perm_yy) * hz / np.log(rp_z / rw)
        # self.well_index = np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)

        self.wells = []

    def fluidflower0(self, discr_type='mpfa'):
        frac_aper = 1.E-4
        self.mesh_file = 'meshes/geometry_coarse_boundaries1.msh'
        #self.mesh_file = '../meshes/ladder/hexahedron_adaptive.msh'
        self.water_column_height = 1.5
        self.well_centers = [[[0.925, 0.0, 0.3294151397849463], [0.925, 0.025, 0.3294151397849463]],
                             [[1.7280622758620692, 0.0, 0.7275171397849464], [1.7280622758620692, 0.025, 0.7275171397849464]]]  # specified manually for now
        self.porperm = [PorPerm(tag=90001, type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
                   PorPerm(tag=90002, type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
                   PorPerm(tag=90003, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                   PorPerm(tag=90004, type='ESF', poro=0.43, perm=44000.0, anisotropy=[1, 1, 0.747972992]),
                   PorPerm(tag=90005, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                   PorPerm(tag=90006, type='Fault-1', poro=0.44, perm=6446953.86, anisotropy=[1, 1, 1]),
                   PorPerm(tag=90007, type='Fault-2', poro=0.44, perm=2859242.02, anisotropy=[1, 1, 1]),
                   PorPerm(tag=90008, type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
                   PorPerm(tag=90009, type='W', poro=0.44, perm=100000000.0, anisotropy=[1, 1, 1])]

        if discr_type == 'mpfa':
            self.mesh_data = meshio.read(self.mesh_file)

            # physical tags from Gmsh scripts (see meshes/*.geo)
            domain_tags = dict()
            domain_tags[elem_loc.MATRIX] = set([90001, 90002, 90003, 90004, 9001, 90005, 90006, 90007, 90008, 90009])
            domain_tags[elem_loc.FRACTURE] = set([])
            domain_tags[elem_loc.BOUNDARY] = set([9991, 9992, 9993, 9994, 9995, 9996])
            domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()

            self.discr_mesh = Mesh()
            self.discr_mesh.gmsh_mesh_processing(self.mesh_file, domain_tags)

            self.discr = Discretizer()
            self.discr.grav_vec = matrix([0.0, 0.0, -9.80665e-5], 1, 3)
            self.cpp_bc = self.set_boundary_conditions(domain_tags)
            self.discr.set_mesh(self.discr_mesh)
            self.discr.init()

            n_vars = 2
            self.n_matrix = self.discr_mesh.region_ranges[elem_loc.MATRIX][1] - self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
            self.n_fracs = self.discr_mesh.region_ranges[elem_loc.FRACTURE][1] - self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
            self.n_bounds = self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]

            self.depth_all_cells = np.zeros(self.n_matrix + self.n_fracs + self.n_bounds)
            self.volume_all_cells = np.zeros(self.n_matrix + self.n_fracs)
            self.porosity = np.zeros(self.n_matrix + self.n_fracs)

            # self.perm_base = np.diag([1000.0, 10.0, 10.0])
            # L = max([pt.values[0] for pt in self.discr_mesh.nodes])
            tags = np.array([elem.tag for elem in self.discr_mesh.elems], dtype=np.int64)
            self.well_cells = np.where(tags == 9001)[0]
            self.layers = [[], [], [], [], [], [], [], [], []]
            centroids = np.array(self.discr_mesh.centroids, copy=False)
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            # loop over elements
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0], self.discr_mesh.region_ranges[elem_loc.FRACTURE][1])):
                tag = tags[i]
                c = centroids[cell_id].values
                # # permeability
                # if c[0] + c[1] <= L / 2:
                #     r = Rotation.from_rotvec([0, 0, np.pi / 4])
                # elif c[0] + c[1] <= L:
                #     r = Rotation.from_rotvec([0, 0, 0])
                # elif c[0] + c[1] <= 3 * L / 2:
                #     r = Rotation.from_rotvec([0, 0, np.pi / 2])
                # else:
                #     r = Rotation.from_rotvec([0, 0, np.pi / 4])

                #self.discr.perms[i] = matrix33(list((r.as_matrix().dot(self.perm_base).dot(r.as_matrix().T).flatten())))
                #self.discr.perms[i] = matrix33(list(self.perm_base.flatten()))
                if tag > 90000:
                    l_id = tag - 90001
                    pp = self.porperm[l_id]
                else:
                    l_id = self.discr_mesh.elems[self.discr_mesh.adj_matrix_cols[self.discr_mesh.adj_matrix_offset[i]]].tag - 90000
                    pp = self.porperm[l_id]

                self.discr.perms.append(matrix33(pp.perm * pp.anisotropy[0],
                                               pp.perm * pp.anisotropy[1],
                                               pp.perm * pp.anisotropy[2]))
                self.porosity[i] = pp.poro


                self.depth_all_cells[i] = c[2]
                self.volume_all_cells[i] = volumes[cell_id]

                self.layers[l_id].append(cell_id)

            self.layers[4].append(self.well_cells[0])
            self.layers[4].append(self.well_cells[1])

            #self.discr.reconstruct_pressure_gradients(self.cpp_bc)
            #self.discr.calc_mpfa_transmissibilities(self.cpp_bc)
            self.discr.calc_tpfa_transmissibilities(domain_tags)

            # self.pz_bounds = np.zeros(n_vars * self.n_bounds)
            # self.pz_bounds[::n_vars] = self.p_init
            # self.pz_bounds[1::n_vars] = self.s_init
            self.bc_flow = np.zeros(n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))
            bc_a = np.array(self.cpp_bc.a, copy=False)
            bc_b = np.array(self.cpp_bc.b, copy=False)
            bc_r = np.array(self.cpp_bc.r, copy=False)
            self.top_cells = []
            cell_p = np.array(self.discr.cell_p, copy=False)
            cell_m = np.array(self.discr.cell_m, copy=False)
            # loop over boundaries
            for i, bound_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0], self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1])):
                if tags[bound_id] == 9996:
                    self.top_cells.append(cell_m[cell_p == bound_id][0])
                c = centroids[bound_id].values
                self.depth_all_cells[bound_id] = c[2]
                #cur_a = bc_a[i]
                #cur_b = bc_b[i]
                #cur_r = bc_r[i]
                self.bc_flow[n_vars * i:n_vars * (i + 1)] = bc_r[i]
                #self.bc_flow.extend([cur_a, cur_b, cur_r])
                #TODO use merged a,b,r

                #if cur_a == 1 and cur_b == 0:
                #    self.pz_bounds[id * n_vars] = cur_r
                #    self.pz_bounds[id * n_vars + 1] = self.s_init
            #self.bc_flow = np.array(self.bc_flow)
        else:
            # Assign layer properties
            self.layer_poro = np.zeros(len(self.porperm))
            self.layer_perm = np.zeros((3, len(self.porperm)))
            for i in range(len(self.porperm)):
                self.layer_poro[i] = self.porperm[i].poro
                for j in range(3):
                    self.layer_perm[j, i] = self.porperm[i].perm * self.porperm[i].anisotropy[j]

            self.unstr_discr = UnstructDiscretizer(permx=self.layer_perm[0, 0],
                                                   permy=self.layer_perm[1, 0],
                                                   permz=self.layer_perm[2, 0],
                                                   frac_aper=0,
                                                   mesh_file=self.mesh_file,
                                                   poro=self.layer_poro[0])
            self.unstr_discr.n_dim = 3

            self.unstr_discr.physical_tags['matrix'] = [90001, 90002, 90003, 90004, 90005, 90006, 90007, 90008,
                                                        90009, 9001]  # C, D, E, ESF, F, Fault-1, Fault-2, G, W(ater)
            #self.unstr_discr.physical_tags['well'] = [9001]
            self.unstr_discr.physical_tags['boundary'] = []
            self.unstr_discr.physical_tags['fracture'] = []
            self.unstr_discr.physical_tags['fracture_shape'] = []
            self.unstr_discr.physical_tags['output'] = []

            # self.well_centers = [[[0.925, 0.0, 0.3294151397849463], [0.925, 0.025, 0.3294151397849463]],
            #                      [[1.7280622758620692, 0.0, 0.7275171397849464],
            #                       [1.7280622758620692, 0.025, 0.7275171397849464]]]  # specified manually for now
            # well_radii = [0.005, 0.005]

            # Use class method load_mesh to load the GMSH file specified above:
            self.unstr_discr.load_mesh_with_bounds()

            # Calculate cell information of each geometric element in the .msh file:
            self.unstr_discr.calc_cell_information()

            # Store volumes and depth to single numpy arrays:
            self.unstr_discr.store_volume_all_cells()
            self.unstr_discr.store_depth_all_cells()
            self.unstr_discr.store_centroid_all_cells()

            self.well_cells = []
            self.unstr_discr.layers = [[], [], [], [], [], [], [], [], []]
            for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items():
                if cell.prop_id > 90000:
                    self.unstr_discr.layers[cell.prop_id - 90001].append(cell_id)
                else:
                    self.unstr_discr.layers[4].append(cell_id)
                    self.well_cells.append(cell_id)

                assert(cell.centroid[2] == self.unstr_discr.depth_all_cells[cell_id])

            self.cell_poro = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_permx = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_permy = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_permz = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_to_layer = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.assign_layer_properties(self.layer_perm, self.layer_poro)

    def fluidflower1(self):
        frac_aper = 1.E-4
        self.mesh_file = 'meshes/cut_fluidflower1.msh'
        self.water_column_height = 1.5
        self.well_centers = np.array([[0.925, 0.0, 0.32942],
                                    [1.72806, 0.0, 0.72757]])
        self.well_cells = -np.ones(self.well_centers.shape[0], dtype=np.int64)

        self.diff_coef = 1.0

        if self.problem_type == 'surface':
            self.porperm = [PorPerm(tag=900001, type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900002, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900003, type='ESF', poro=0.43, perm=44000.0, anisotropy=[1, 1, 0.747972992]),
                           PorPerm(tag=900004, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900005, type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900006, type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900007, type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900008, type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900009, type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900010, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900011, type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900012, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                           #PorPerm(tag=900013, type='ESF', poro=0.43, perm=44000.0, anisotropy=[1, 1, 0.747972992]),
                           PorPerm(tag=900013, type='ESF', poro=0.1, perm=0.1, anisotropy=[1, 1, 0.747972992]),
                           PorPerm(tag=900014, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900015, type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900016, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900017, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900018, type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900019, type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900020, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900021, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900022, type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900023, type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900024, type='G', poro=0.44, perm=2276750.01, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900025, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                           PorPerm(tag=900026, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900027, type='E', poro=0.45, perm=1441352.77, anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900028, type='D', poro=0.44, perm=1878990.31, anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900029, type='C', poro=0.44, perm=385597.374, anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900030, type='ESF', poro=0.43, perm=44000.0, anisotropy=[1, 1, 0.747972992]),
                           PorPerm(tag=900031, type='W', poro=0.44, perm=100000000.0, anisotropy=[1, 1, 1]),
                           # PorPerm(tag=900034, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                           # PorPerm(tag=900036, type='F', poro=0.45, perm=3246353.97, anisotropy=[1, 1, 1]),
                           ]
            self.diff_mult = self.diff_coef / self.porperm[11].perm / TransCalculations.darcy_constant
        elif self.problem_type == 'reservoir':
            permg = {
                "G": 2e4,
                "E": 2e4,
                "D": 5e3,
                "C": 1e3,
                "F": 1e3,
                "ESF": 5e2
            }
            poro = 0.2
            self.porperm = [PorPerm(tag=900001, type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900002, type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900003, type='ESF', poro=poro, perm=permg['ESF'], anisotropy=[1, 1, 0.747972992]),
                           PorPerm(tag=900004, type='E', poro=poro, perm=permg['E'], anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900005, type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900006, type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900007, type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900008, type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900009, type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900010, type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900011, type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900012, type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900013, type='ESF', poro=poro, perm=1.e-6 * permg['ESF'], anisotropy=[1, 1, 0.747972992]),
                           PorPerm(tag=900014, type='E', poro=poro, perm=permg['E'], anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900015, type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900016, type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900017, type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900018, type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900019, type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900020, type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900021, type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900022, type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900023, type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900024, type='G', poro=poro, perm=permg['G'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900025, type='F', poro=poro, perm=permg['F'], anisotropy=[1, 1, 1]),
                           PorPerm(tag=900026, type='E', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900027, type='E', poro=poro, perm=permg['E'], anisotropy=[1, 1, 0.9]),
                           PorPerm(tag=900028, type='D', poro=poro, perm=permg['D'], anisotropy=[1, 1, 0.8]),
                           PorPerm(tag=900029, type='C', poro=poro, perm=permg['C'], anisotropy=[1, 1, 0.7]),
                           PorPerm(tag=900030, type='ESF', poro=poro, perm=permg['ESF'], anisotropy=[1, 1, 0.747972992]),
                           PorPerm(tag=900031, type='W', poro=poro, perm=10000, anisotropy=[1, 1, 1]),
                           ]
            self.diff_mult = self.diff_coef / permg['F'] / TransCalculations.darcy_constant

        dist = 1e+10 * np.ones(self.well_centers.shape[0])
        if self.discr_type == 'mpfa':
            self.mesh_data = meshio.read(self.mesh_file)

            # physical tags from Gmsh scripts (see meshes/*.geo)
            domain_tags = dict()
            domain_tags[elem_loc.MATRIX] = set([900001, 900002, 900003, 900004, 900005, 900006, 900007, 900008, 900009, 900010,
                                                900011, 900012, 900013, 900014, 900015, 900016, 900017, 900018, 900019, 900020,
                                                900021, 900022, 900023, 900024, 900025, 900026, 900027, 900028, 900029, 900030,
                                                900031])
            domain_tags[elem_loc.FRACTURE] = set([])
            domain_tags[elem_loc.BOUNDARY] = set([9991, 9992, 9993, 9994, 9995, 9996])
            domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()

            self.discr_mesh = Mesh()
            self.discr_mesh.gmsh_mesh_processing(self.mesh_file, domain_tags)

            self.discr = Discretizer()
            self.discr.grav_vec = matrix([0.0, 0.0, -9.80665e-5], 1, 3)
            self.cpp_bc = self.set_boundary_conditions(domain_tags)
            self.discr.set_mesh(self.discr_mesh)
            self.discr.init()

            n_vars = 2
            self.n_matrix = self.discr_mesh.region_ranges[elem_loc.MATRIX][1] - self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
            self.n_fracs = self.discr_mesh.region_ranges[elem_loc.FRACTURE][1] - self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
            self.n_bounds = self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]

            self.depth_all_cells = np.zeros(self.n_matrix + self.n_fracs + self.n_bounds)
            self.volume_all_cells = np.zeros(self.n_matrix + self.n_fracs)
            self.porosity = np.zeros(self.n_matrix + self.n_fracs)

            # self.perm_base = np.diag([1000.0, 10.0, 10.0])
            # L = max([pt.values[0] for pt in self.discr_mesh.nodes])
            tags = np.array(self.discr_mesh.tags, copy=False)
            centroids = np.array(self.discr_mesh.centroids, copy=False)
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.layers = [[] for i in range(len(self.porperm))]
            # loop over elements
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0], self.discr_mesh.region_ranges[elem_loc.FRACTURE][1])):
                tag = tags[i]
                c = centroids[cell_id].values
                l_id = tag - 900001
                pp = self.porperm[l_id]

                self.discr.perms.append(matrix33(pp.perm * pp.anisotropy[0],
                                               pp.perm * pp.anisotropy[1],
                                               pp.perm * pp.anisotropy[2]))
                self.porosity[i] = pp.poro


                self.depth_all_cells[i] = c[2]
                self.volume_all_cells[i] = volumes[cell_id]
                self.layers[l_id].append(cell_id)
                # # find well cells
                cur_dist = np.linalg.norm(c - self.well_centers, axis=1)
                ids = cur_dist < dist
                dist[ids] = cur_dist[ids]
                self.well_cells[ids] = np.array([cell_id, cell_id], dtype=np.int64)[ids]

            if self.mpfa_type == 'mpfa':
                self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_bc)
                self.discr.calc_mpfa_transmissibilities(self.cpp_bc)
            else:
                self.discr.calc_tpfa_transmissibilities(domain_tags)

            # self.pz_bounds = np.zeros(n_vars * self.n_bounds)
            # self.pz_bounds[::n_vars] = self.p_init
            # self.pz_bounds[1::n_vars] = self.s_init
            self.bc_flow = np.zeros(n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))
            bc_a = np.array(self.cpp_bc.a, copy=False)
            bc_b = np.array(self.cpp_bc.b, copy=False)
            bc_r = np.array(self.cpp_bc.r, copy=False)
            self.top_cells = []
            cell_p = np.array(self.discr.cell_p, copy=False)
            cell_m = np.array(self.discr.cell_m, copy=False)
            # loop over boundaries
            for i, bound_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0], self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1])):
                if tags[bound_id] == 9996:
                    matrix_cell = cell_m[cell_p == bound_id][0]
                    if tags[matrix_cell] == 900008:
                        self.top_cells.append(matrix_cell)
                c = centroids[bound_id].values
                self.depth_all_cells[bound_id] = c[2]
                #cur_a = bc_a[i]
                #cur_b = bc_b[i]
                #cur_r = bc_r[i]
                self.bc_flow[n_vars * i:n_vars * (i + 1)] = bc_r[i]
                #self.bc_flow.extend([cur_a, cur_b, cur_r])
                #TODO use merged a,b,r

                #if cur_a == 1 and cur_b == 0:
                #    self.pz_bounds[id * n_vars] = cur_r
                #    self.pz_bounds[id * n_vars + 1] = self.s_init
            #self.bc_flow = np.array(self.bc_flow)
        else:
            # Assign layer properties
            self.layer_poro = np.zeros(len(self.porperm))
            self.layer_perm = np.zeros((3, len(self.porperm)))
            for i in range(len(self.porperm)):
                self.layer_poro[i] = self.porperm[i].poro
                for j in range(3):
                    self.layer_perm[j, i] = self.porperm[i].perm * self.porperm[i].anisotropy[j]

            self.unstr_discr = UnstructDiscretizer(permx=self.layer_perm[0, 0],
                                                   permy=self.layer_perm[1, 0],
                                                   permz=self.layer_perm[2, 0],
                                                   frac_aper=0,
                                                   mesh_file=self.mesh_file,
                                                   poro=self.layer_poro[0])
            self.unstr_discr.n_dim = 3

            self.unstr_discr.physical_tags['matrix'] = [900001, 900002, 900003, 900004, 900005, 900006, 900007, 900008, 900009, 900010,
                                                        900011, 900012, 900013, 900014, 900015, 900016, 900017, 900018, 900019, 900020,
                                                        900021, 900022, 900023, 900024, 900025, 900026, 900027, 900028, 900029, 900030,
                                                        900031]
            #self.unstr_discr.physical_tags['well'] = [9001]
            self.unstr_discr.physical_tags['boundary'] = []
            self.unstr_discr.physical_tags['fracture'] = []
            self.unstr_discr.physical_tags['fracture_shape'] = []
            self.unstr_discr.physical_tags['output'] = []

            # Use class method load_mesh to load the GMSH file specified above:
            self.unstr_discr.load_mesh_with_bounds()

            # Calculate cell information of each geometric element in the .msh file:
            self.unstr_discr.calc_cell_information()

            # Store volumes and depth to single numpy arrays:
            self.unstr_discr.store_volume_all_cells()
            self.unstr_discr.store_depth_all_cells()
            self.unstr_discr.store_centroid_all_cells()

            self.unstr_discr.layers = [[] for i in range(len(self.porperm))]
            for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items():
                self.unstr_discr.layers[cell.prop_id - 900001].append(cell_id)
                assert(cell.centroid[2] == self.unstr_discr.depth_all_cells[cell_id])
                # # find well cells
                cur_dist = np.linalg.norm(cell.centroid - self.well_centers, axis=1)
                ids = cur_dist < dist
                dist[ids] = cur_dist[ids]
                self.well_cells[ids] = np.array([cell_id, cell_id], dtype=np.int64)[ids]

            self.cell_poro = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_permx = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_permy = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_permz = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.cell_to_layer = np.zeros(self.unstr_discr.volume_all_cells.size)
            self.assign_layer_properties(self.layer_perm, self.layer_poro)

    def find_well_cells(self, well_center):
        dist0 = None

        if self.discr_type == 'tpfa':
            for l, centroid in enumerate(self.unstr_discr.centroid_all_cells):
                if self.curvature:
                    r = np.sqrt(centroid[0] ** 2 + centroid[1] ** 2)  # radius of circle at centroid
                    theta = pi / 8 + asin(centroid[0] / r)  # angle from left end of domain

                    X = r * theta  # arc length L == x
                else:
                    X = centroid[0]
                Z = centroid[2]
                dist1 = np.sqrt((X - well_center[0]) ** 2 + (Z - well_center[2]) ** 2)
                if dist0 is None or dist1 < dist0:
                    idx = l
                    dist0 = dist1
        elif self.discr_type == 'mpfa':
            centroids = np.array(self.discr_mesh.centroids, copy=False)[:self.n_matrix]
            for l, centroid in enumerate(centroids):
                centroid = np.array(centroid.values, copy=False)
                if self.curvature:
                    r = np.sqrt(centroid[0] ** 2 + centroid[1] ** 2)  # radius of circle at centroid
                    theta = pi / 8 + asin(centroid[0] / r)  # angle from left end of domain

                    X = r * theta  # arc length L == x
                else:
                    X = centroid[0]
                Z = centroid[2]
                dist1 = np.sqrt((X - well_center[0]) ** 2 + (Z - well_center[2]) ** 2)
                if dist0 is None or dist1 < dist0:
                    idx = l
                    dist0 = dist1

        well_depth = self.depth[idx]
        return [idx], well_depth

    def assign_layer_properties(self, perm, poro):
        # Assign properties to layers
        for i, ith_layer in enumerate(self.unstr_discr.layers):
            layer_perm = perm[:, i]
            layer_poro = poro[i]
            for ith_cell in ith_layer:
                self.cell_to_layer[ith_cell] = i
                self.unstr_discr.mat_cell_info_dict[ith_cell].permeability = [layer_perm[0], layer_perm[1], layer_perm[2]]
                self.cell_poro[ith_cell] = layer_poro
                self.cell_permx[ith_cell] = layer_perm[0]
                self.cell_permy[ith_cell] = layer_perm[1]
                self.cell_permz[ith_cell] = layer_perm[2]
            if i == 3:
                self.seal = ith_layer

        return 0

    def set_boundary_conditions(self, physical_tags):
        bc = BoundaryCondition()

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        a = np.zeros(boundary_range[1] - boundary_range[0])
        b = np.zeros(boundary_range[1] - boundary_range[0])
        r = np.zeros(boundary_range[1] - boundary_range[0])

        #TODO: move changes from test_discretizer_performance
        ##range_bnd = range(boundary_range[0], boundary_range[1])
        # no-flow (impermeable) bc
        a[:] = 0.
        b[:] = 1.
        r[:] = 0.
        #TODO: implement without loop in case of different boundary conditions
        P_MINUS = 10
        P_PLUS = 20
        '''
        for i, cell_id in enumerate(range(boundary_range[0], boundary_range[1])):
            # el = mesh.elems[cell_id]
            # assert(el.loc == elem_loc.BOUNDARY)
            if False:  # el.loc == 991: # Dirichlet
                a[i] = 1.0
                b[i] = 0.0
                r[i] = P_MINUS
            elif False:  # el.loc == 992 # Dirichlet
                a[i] = 1.0
                b[i] = 0.0
                r[i] = P_PLUS
            else:  # no-flow (impermeable) bc
                a[i] = 0.0
                b[i] = 1.0
                r[i] = 0.0
        '''
        bc.a = value_vector(a)
        bc.b = value_vector(b)
        bc.r = value_vector(r)

        return bc

    def add_well(self, name, depth):
        """
        Class method which adds wells heads to the reservoir (Note: well head is not equal to a perforation!)
        :param name:
        :param depth:
        :return:
        """
        well = ms_well()
        well.name = name
        well.segment_volume = 1e-4  # 2.5 * pi * 0.15**2 / 4
        well.well_head_depth = depth
        well.well_body_depth = depth
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
        well.perforations = well.perforations + [(well_block, res_block, well_index, 0.0)]
        return 0

    def init_wells(self):
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """
        # Add injection well:
        #self.add_well("I1", 0.5)
        #if self.bound_cond == 'const_pres_rate':
        #    # Perforate all boundary cells:
        #    for nth_perf in range(len(self.left_boundary_cells)):
        #        well_index = self.mesh.volume[self.left_boundary_cells[nth_perf]] / self.max_well_vol * self.well_index
        #        self.add_perforation(well=self.wells[-1], res_block=self.left_boundary_cells[nth_perf],
        #                             well_index=well_index)

        # Add production well:
        #self.add_well("P1", 0.5)
        #if self.bound_cond == 'const_pres_rate':
            # Perforate all boundary cells:
        #    for nth_perf in range(len(self.right_boundary_cells)):
        #        well_index = self.mesh.volume[self.right_boundary_cells[nth_perf]] / self.max_well_vol * self.well_index
        #        self.add_perforation(self.wells[-1], res_block=self.right_boundary_cells[nth_perf],
        #                             well_index=well_index)

        # Add wells to the DARTS mesh object and sort connection (DARTS related):
        if self.discr_type == 'mpfa':
            self.mesh.add_wells_mpfa(ms_well_vector(self.wells), self.P_VAR)
            self.mesh.reverse_and_sort_mpfa()
        else:
            self.mesh.add_wells(ms_well_vector(self.wells))
            self.mesh.reverse_and_sort()
        self.mesh.init_grav_coef()
        return 0

    def reconstruct_velocities(self, p):
        n_dim = 3
        rhs = {}#np.zeros((self.unstr_discr.matrix_cell_count, faces_per_cell))
        a = {}#np.zeros((self.unstr_discr.matrix_cell_count, faces_per_cell, n_dim))
        face_id = -1
        cell_m_prev = self.cell_m[0]
        n_blocks = self.mesh.n_blocks
        n_res_blocks = self.mesh.n_res_blocks
        for id, cell_m in enumerate(self.cell_m):
            cell_p = self.cell_p[id]
            if cell_m >= n_res_blocks or \
                    (cell_p >= n_res_blocks and cell_p < n_blocks): continue

            faces = self.unstr_discr.faces[cell_m]

            face_id = face_id + 1 if cell_m == cell_m_prev else 0
            face = list(self.unstr_discr.faces[cell_m].values())[face_id]
            assert(face.cell_id2 == cell_p or face.face_id2 + n_blocks == cell_p)

            if cell_m not in rhs:
                rhs[cell_m] = np.array([])
                a[cell_m] = np.empty((0,n_dim))

            f = self.rhs[id]
            for k in range(self.offset[id], self.offset[id+1]):
                if self.stencil[k] >= n_blocks:
                    var = self.unstr_discr.bc_flow[self.stencil[k] - n_blocks]
                else:
                    var = p[self.stencil[k]]
                f += self.trans[k] * var
            rhs[cell_m] = np.append(rhs[cell_m], f)
            n = face.n
            sign = -np.sign((face.centroid - self.unstr_discr.mat_cell_info_dict[cell_m].centroid).dot(n))
            a[cell_m] = np.append(a[cell_m], sign * face.area * n[np.newaxis], axis=0)

        vel = np.zeros((self.unstr_discr.matrix_cell_count, n_dim))
        for cell_id in range(self.unstr_discr.matrix_cell_count):
            vel[cell_id] = np.linalg.inv(a[cell_id].T.dot(a[cell_id])).dot(a[cell_id].T).dot(rhs[cell_id])
        return vel

    def write_mpfa_conn_to_file(self, path = 'mpfa_conn.dat'):
        stencil = np.array(self.discr.flux_stencil, copy=False)
        trans = np.array(self.discr.flux_vals, copy=False)

        f = open(path, 'w')
        f.write(str(len(self.discr.cell_m)) + '\n')

        for conn_id in range(len(self.discr.cell_m)):
            cells = stencil[self.discr.flux_offset[conn_id]:self.discr.flux_offset[conn_id + 1]]
            coefs= trans[self.discr.flux_offset[conn_id]:self.discr.flux_offset[conn_id + 1]]
            #row = str(self.discr.cell_m[conn_id]) + '\t' + str(self.discr.cell_p[conn_id])
            row = str(self.discr.cell_m[conn_id]) + '\t' + str(self.discr.cell_p[conn_id]) + '\t\t'
            #row_cells = ''#str(cells)
            #row_vals = ''#str(coefs)
            for i in range(cells.size):
                if np.abs(coefs[i]) > 1.E-10:
                    row += str(cells[i]) + '\t' + str('{:.8e}'.format(coefs[i])) + '\t'
                    #row_cells += str(cells[i]) + '\t'
                    #row_vals += str('{:.2e}'.format(coefs[i])) + '\t'
            f.write(row + '\n')# + row_cells + '\n' + row_vals + '\n')
        f.close()

    def write_to_vtk_mpfa(self, output_directory, cell_property, ith_step, output_props, physics):
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

        # Allocate empty new cell_data dictionary:
        props_num = len(cell_property)
        property_array = np.array(physics.engine.X, copy=False)
        available_matrix_geometries_cpp = [elem_type.HEX, elem_type.PRISM, elem_type.TETRA, elem_type.PYRAMID]
        available_fracture_geometries_cpp = [elem_type.QUAD, elem_type.TRI]
        available_matrix_geometries = {'hexahedron': elem_type.HEX,
                                       'wedge': elem_type.PRISM,
                                       'tetra': elem_type.TETRA,
                                       'pyramid': elem_type.PYRAMID}
        available_fracture_geometries = ['quad', 'triangle']

        # Matrix
        cells = []
        cell_data = {}
        for cell_block in self.mesh_data.cells:
            if cell_block.type in available_matrix_geometries:
                cells.append(cell_block)
                cell_ids = np.array(self.discr_mesh.elem_type_map[available_matrix_geometries[cell_block.type]], copy=False, dtype=np.int64)
                for i in range(props_num):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(property_array[props_num * cell_ids + i])

                n_props = output_props.n_props
                values = value_vector(np.zeros(n_props))
                property_data = np.zeros((len(cell_ids), n_props))
                for i, cell_id in enumerate(cell_ids):
                    state = []
                    for j in range(props_num):
                        state.append(property_array[props_num * cell_id + j])
                    state = value_vector(np.asarray(state))
                    physics.property_itor.evaluate(state, values)

                    for j in range(n_props):
                        property_data[i, j] = values[j]

                for i in range(n_props):
                    if output_props.props[i] not in cell_data: cell_data[output_props.props[i]] = []
                    cell_data[output_props.props[i]].append(property_data[:,i])

                if ith_step == 0:
                    if 'perm' not in cell_data: cell_data['perm'] = []
                    # if 'cell_id' not in cell_data: cell_data['cell_id'] = []
                    cell_data['perm'].append(np.zeros((len(cell_ids), 9), dtype=np.float64))
                    # cell_data['cell_id'].append(np.zeros(len(cell_ids), dtype=np.int64))
                    for i, cell_id in enumerate(cell_ids):
                        cell_data['perm'][-1][i] = np.array(self.discr.perms[cell_id].values)
                        # cell_data['cell_id'][-1][i] = cell_id

                # if 'vel' not in cell_data: cell_data['vel'] = []
                # vel = self.reconstruct_velocities(property_array[::2])
                # cell_data['vel'].append(vel)

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            self.mesh_data.points,
            cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)


        # for ith_geometry in self.unstr_discr.mesh_data.cells:
        #     # Extract left and right bound of array slicing:
        #
        #     # Store matrix or fractures cells in appropriate location
        #     if ith_geometry == 'hexahedron' or ith_geometry == 'wedge':
        #         # Add matrix data to dictionary:
        #         cell_data[ith_geometry] = {}
        #         for i in range(len(cell_property)):
        #             cell_data[ith_geometry][cell_property[i]] = property_array[:, i]
        #
        #         cell_data[ith_geometry]['matrix_cell_bool'] = np.ones(((self.unstr_discr.matrix_cell_count),))
        #         cell_data[ith_geometry]['perm'] = np.zeros((self.unstr_discr.matrix_cell_count, 9))
        #         for cell_id in self.unstr_discr.mat_cell_info_dict.keys():
        #             cell_data[ith_geometry]['perm'][cell_id] = self.unstr_discr.permeability[cell_id].flatten()
        #
        #         vel = self.reconstruct_velocities(property_array[:, 0])
        #         cell_data[ith_geometry]['velocity'] = vel
        #         cell_data[ith_geometry]['matrix_cell_bool'] = np.ones(((self.unstr_discr.matrix_cell_count),))
        #
        #         # vel = self.reconstruct_velocities(property_array[:, 0])
        #         # cell_data[ith_geometry]['velocity'] = vel
        #
        # # Store solution for each time-step:
        # Mesh.cell_data = cell_data
        # print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        # meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), Mesh)
        return 0

    def write_to_vtk_tpfa(self, output_directory, property_array, cell_property, ith_step):
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
        cells_dict = {'wedge': self.unstr_discr.mesh_data.cells_dict['wedge']}
        cell_data_dict = dict()

        for ith_prop in range(len(cell_property)):
            cell_data_dict[cell_property[ith_prop]] = []
            left_bound = 0
            right_bound = 0
            for ith_geometry in cells_dict:
                left_bound = right_bound
                right_bound = right_bound + cells_dict[ith_geometry].shape[0]
                cell_data_dict[cell_property[ith_prop]].append(list(property_array[left_bound:right_bound, ith_prop]))

        cell_data_dict['matrix_cell_bool'] = []
        left_bound = 0
        right_bound = 0
        for ith_geometry in cells_dict:
            left_bound = right_bound
            right_bound = right_bound + cells_dict[ith_geometry].shape[0]

            if (ith_geometry in self.unstr_discr.available_fracture_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.zeros(((right_bound - left_bound),))))

            elif (ith_geometry in self.unstr_discr.available_matrix_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.ones(((right_bound - left_bound),))))

        # Temporarily store mesh_data in copy:
        # Mesh = meshio.read(self.mesh_file)

        mesh = meshio.Mesh(
            # Mesh.points,
            # Mesh.cells_dict.items(),
            self.unstr_discr.mesh_data.points,  # list of point coordinates
            cells_dict.items(),  # list of
            # Each item in cell data must match the cells array
            cell_data=cell_data_dict)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)
        return 0
