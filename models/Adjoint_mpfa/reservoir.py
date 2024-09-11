from darts.engines import conn_mesh, ms_well, ms_well_vector
from scipy.spatial.transform import Rotation
import numpy as np
import meshio
from math import inf, pi
from itertools import compress
import os
import sys
import inspect
from scipy.spatial.transform import Rotation as R
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import darts.discretizer as dis
from darts.discretizer import Mesh, Elem, Discretizer, BoundaryCondition, elem_loc, elem_type, matrix33, vector_matrix33, vector_vector3, matrix, value_vector, index_vector
import datetime
from dataclasses import dataclass, field
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.mesh.transcalc import TransCalculations
import copy

@dataclass
class PorPerm:
    id: int
    type: str
    poro: float
    perm: float
    anisotropy: list

# Definitions for the unstructured reservoir class:
class UnstructReservoir:
    def __init__(self, discr_type, mesh_file, n_vars):
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
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()

        self.n_vars = n_vars

        self.ladder(discr_type, mesh_file)
        if discr_type == 'mpfa':
            # self.prepare_velocity_reconstruction()
            self.mesh.init_mpfa(self.discr.cell_m, self.discr.cell_p,
                                self.discr.flux_stencil, self.discr.flux_offset,
                                self.discr.flux_vals, self.discr.flux_rhs, self.discr.flux_vals_homo, self.discr.flux_vals_thermal,
                                self.n_matrix, self.n_bounds, self.n_fracs, self.n_vars)

            self.bc = np.array(self.mesh.bc, copy=False)
            self.depth = np.array(self.mesh.depth, copy=False)
            self.volume = np.array(self.mesh.volume, copy=False)
            self.depth[:] = self.depth_all_cells
            self.volume[:] = self.volume_all_cells
            self.bc[:self.bc_input.size] = self.bc_input
        else:
            cell_m, cell_p, tran, tran_thermal = self.unstr_discr.calc_connections_all_cells()
            # self.unstr_discr.write_conn2p_to_file(cell_m, cell_p, tran, file_name='conn2p.dat')
            self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))
            self.depth = np.array(self.mesh.depth, copy=False)
            self.volume = np.array(self.mesh.volume, copy=False)
            self.depth[:] = self.unstr_discr.depth_all_cells[:self.unstr_discr.mat_cells_tot]
            self.volume[:] = self.unstr_discr.volume_all_cells

        # Write to files (in case someone needs this for Eclipse or other simulator):
        #self.unstr_discr.write_volume_to_file(file_name='vol.dat')
        #self.unstr_discr.write_depth_to_file(file_name='depth.dat')

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.poro[:] = self.porosity
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.conduction = np.array(self.mesh.rock_cond, copy=False)

        # Calculate well_index (very primitive way....):
        rw = 0.1
        mean_cell_width = np.cbrt(np.mean(self.volume[:self.mesh.n_res_blocks]))
        dx = mean_cell_width
        dy = mean_cell_width
        dz = 10.0
        # WIx
        wi_x = 0.0
        # WIy
        wi_y = 0.0
        # WIz
        hz = dz
        mean_perm_xx = self.perm_mat#np.mean(np.array([self.discr.perms[cell_id].values[0] for cell_id in self.well_cells]))
        mean_perm_yy = self.perm_mat#np.mean(np.array([self.discr.perms[cell_id].values[4] for cell_id in self.well_cells]))
        rp_z = 0.28 * np.sqrt((mean_perm_yy / mean_perm_xx) ** 0.5 * dx ** 2 +
                              (mean_perm_xx / mean_perm_yy) ** 0.5 * dy ** 2) / \
               ((mean_perm_xx / mean_perm_yy) ** 0.25 + (mean_perm_yy / mean_perm_xx) ** 0.25)
        wi_z = 2 * np.pi * np.sqrt(mean_perm_xx * mean_perm_yy) * hz / np.log(rp_z / rw)
        self.well_index = np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)

        self.wells = []

    def ladder(self, discr_type, mesh_file):
        self.perm_mat = 1000
        self.porperm = [PorPerm(id=99991, type='C', poro=0.2, perm=self.perm_mat, anisotropy=[1.0, 0.01, 1.0])]
        self.mesh_file = mesh_file

        self.diff_coef = 0.0#8.64e-6
        self.diff_mult = self.diff_coef / self.porperm[0].perm / TransCalculations.darcy_constant
        frac_aper = 1.E-4

        if discr_type == 'mpfa':
            self.mesh_data = meshio.read(self.mesh_file)

            # physical tags from Gmsh scripts (see meshes/*.geo)
            domain_tags = dict()
            domain_tags[elem_loc.MATRIX] = set([99991])
            domain_tags[elem_loc.FRACTURE] = set([])#9991, 9992])
            domain_tags[elem_loc.BOUNDARY] = set([991, 992, 993, 994, 995, 996])
            domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()  # is this for poromechanics??

            self.discr_mesh = Mesh()

            self.n_fracs = 0
            # for i, block in enumerate(self.mesh_data.cell_data['gmsh:physical']):
            #     self.n_fracs += np.isin(self.mesh_data.cell_data['gmsh:physical'][i],
            #                        list(domain_tags[elem_loc.FRACTURE])).sum()
            # self.discr_mesh.init_apertures = value_vector(frac_aper * np.ones(self.n_fracs))

            self.discr_mesh.gmsh_mesh_processing(self.mesh_file, domain_tags)

            self.discr = Discretizer()
            self.discr.grav_vec = matrix([0.0, 0.0, 0.0], 1, 3)  # 0.0??
            self.tags = np.array(self.discr_mesh.tags, copy=False)
            self.cpp_bc_flow, self.cpp_bc_heat = self.set_boundary_conditions(domain_tags)
            self.discr.set_mesh(self.discr_mesh)
            self.discr.init()

            self.n_matrix = self.discr_mesh.region_ranges[elem_loc.MATRIX][1] - self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
            #self.n_fracs = self.discr_mesh.region_ranges[elem_loc.FRACTURE][1] - self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
            self.n_bounds = self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]

            self.depth_all_cells = np.zeros(self.n_matrix + self.n_fracs + self.n_bounds)
            self.volume_all_cells = np.zeros(self.n_matrix + self.n_fracs)
            self.porosity = np.zeros(self.n_matrix + self.n_fracs)

            # self.perm_base = np.diag([1000.0, 10.0, 10.0])
            # L = max([pt.values[0] for pt in self.discr_mesh.nodes])

            centroids = np.array(self.discr_mesh.centroids)
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            L = max([pt.values[0] for pt in self.discr_mesh.nodes])
            pp = self.porperm[0]
            self.perm_base = np.diag((pp.perm * pp.anisotropy[0], pp.perm * pp.anisotropy[1], pp.perm * pp.anisotropy[2]))
            self.heat_cond = 0.2 * 86400.0
            # loop over elements
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0], self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
                c = centroids[cell_id].values
                pp = self.porperm[0]
                if c[0] + c[1] <= L / 2:
                    r = R.from_rotvec([0, 0, np.pi / 4])
                elif c[0] + c[1] <= L:
                    r = R.from_rotvec([0, 0, 0])
                elif c[0] + c[1] <= 3 * L / 2:
                    r = R.from_rotvec([0, 0, np.pi / 2])
                else:
                    r = R.from_rotvec([0, 0, np.pi / 4])
                perm = r.as_matrix().dot(self.perm_base).dot(r.as_matrix().T)
                self.discr.perms.append(matrix33(value_vector(perm.flatten())))
                self.discr.heat_conductions.append(matrix33(self.heat_cond, self.heat_cond, self.heat_cond))
                self.porosity[i] = pp.poro
                self.depth_all_cells[i] = c[2]  # why not c[2]??
                self.volume_all_cells[i] = volumes[cell_id]
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.FRACTURE][0], self.discr_mesh.region_ranges[elem_loc.FRACTURE][1])):
                c = centroids[cell_id].values
                pp = self.porperm[1]
                self.discr.perms.append(matrix33(pp.perm * pp.anisotropy[0], pp.perm * pp.anisotropy[1], pp.perm * pp.anisotropy[2]))
                self.porosity[self.n_matrix + i] = pp.poro
                self.depth_all_cells[self.n_matrix + i] = c[2]  # why not c[2]??
                self.volume_all_cells[self.n_matrix + i] = volumes[cell_id]

            #self.discr.calc_tpfa_transmissibilities(domain_tags)
            self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_bc_flow, self.cpp_bc_heat)
            self.discr.calc_mpfa_transmissibilities(True)

            # self.pz_bounds = np.zeros(self.n_vars * self.n_bounds)
            # self.pz_bounds[::self.n_vars] = self.p_init
            # self.pz_bounds[1::self.n_vars] = self.s_init
            self.bc_input = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))
            # loop over boundaries
            for i, bound_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0], self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1])):
                # elem = self.discr_mesh.elems[bound_id]
                c = centroids[bound_id].values
                self.depth_all_cells[bound_id] = c[2]
                #cur_a = bc_a[i]
                #cur_b = bc_b[i]
                #cur_r = bc_r[i]
                self.bc_input[self.n_vars * i] = 0.0
                self.bc_input[self.n_vars * i + self.n_vars - 1] = 0.0
                #self.bc_flow.extend([cur_a, cur_b, cur_r])
                #TODO use merged a,b,r

                #if cur_a == 1 and cur_b == 0:
                #    self.pz_bounds[id * self.n_vars] = cur_r
                #    self.pz_bounds[id * self.n_vars + 1] = self.s_init
        else:
            self.mesh_data = meshio.read(self.mesh_file)
            pp = self.porperm[0]
            self.porosity = pp.poro
            # Construct instance of Unstructured Discretization class:
            physical_tags = {'matrix': [99991], 'boundary': [991, 992, 993, 994, 995, 996],
                             'fracture': [], 'fracture_boundary': []}

            self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=physical_tags)

            self.unstr_discr.load_mesh(permx=pp.perm * pp.anisotropy[0],
                                       permy=pp.perm * pp.anisotropy[1],
                                       permz=pp.perm * pp.anisotropy[2],
                                       frac_aper=frac_aper,
                                       cache=False)
            # Store volumes and depth to single numpy arrays:
            self.unstr_discr.store_volume_all_cells()
            self.unstr_discr.store_depth_all_cells()
            self.unstr_discr.store_centroid_all_cells()

    def set_boundary_conditions(self, physical_tags):
        bc_flow = BoundaryCondition()
        bc_heat = BoundaryCondition()

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        a = np.zeros(boundary_range[1] - boundary_range[0])
        b = np.ones(boundary_range[1] - boundary_range[0])
        r = np.zeros(boundary_range[1] - boundary_range[0])

        # P_MINUS = 300.0

        # top = self.tags[boundary_range[0]:boundary_range[1]] == 994
        # a[top] = 1.0
        # b[top] = 0.0

        # grav = -self.discr.grav_vec.values[1]
        # for id in np.where(top)[0]:
        #     r[id] = 0.001#1000 * grav * self.discr_mesh.centroids[boundary_range[0] + id].values[1]

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

        # a, b, r ??
        bc_flow.a = value_vector(a)
        bc_flow.b = value_vector(b)
        bc_heat.a = value_vector(a)
        bc_heat.b = value_vector(b)


        return bc_flow, bc_heat

    def add_well(self, name, depth, wellbore_diameter=0.15):
        """
        Class method which adds wells heads to the reservoir (Note: well head is not equal to a perforation!)
        :param name:
        :param depth:
        :return:
        """
        well = ms_well()
        well.name = name
        well.segment_volume =  pi * wellbore_diameter ** 2 / 4
        well.well_head_depth = depth
        well.well_body_depth = depth
        well.segment_transmissibility = 1e5
        well.segment_depth_increment = 1
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

    def init_reservoir(self, verbose):
        pass

    def prepare_velocity_reconstruction(self):
        n_dim = 3
        self.mat_flux = {}
        adj_mat_offset = np.array(self.discr_mesh.adj_matrix_offset, copy=False)
        adj_mat_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        adj_mat = np.array(self.discr_mesh.adj_matrix, copy=False)
        for cell_id in range(self.discr_mesh.n_cells):
            # a = np.empty((0, n_dim))
            a = np.zeros((adj_mat_offset[cell_id + 1] - adj_mat_offset[cell_id], n_dim))
            cell_centroid = np.array(self.discr_mesh.centroids[cell_id].values, copy=False)
            ids = np.argsort(adj_mat_cols[adj_mat_offset[cell_id]:adj_mat_offset[cell_id + 1]])
            for i, k in enumerate(range(adj_mat_offset[cell_id], adj_mat_offset[cell_id + 1])):
                conn = self.discr_mesh.conns[adj_mat[k]]
                n = np.array(conn.n.values, copy=False)
                face_centroid = np.array(conn.c.values, copy=False)
                sign = np.sign((face_centroid - cell_centroid).dot(n))
                a[ids[i]] = sign * conn.area * n[np.newaxis]
            self.mat_flux[cell_id] = np.linalg.inv(a.T.dot(a)).dot(a.T)

    def reconstruct_velocities(self, fluxes):
        vels = np.zeros((self.discr_mesh.n_cells, 3))
        conn_id = 0
        max_conns = 10
        for cell_m in range(self.discr_mesh.n_cells):
            rhs = np.zeros(max_conns)
            face_id = 0
            while self.mesh.block_m[conn_id] == cell_m:
                cell_p = self.mesh.block_p[conn_id]
                # skip well connections
                if not (cell_p >= self.mesh.n_res_blocks and cell_p < self.mesh.n_blocks):
                    rhs[face_id] = fluxes[conn_id]
                    # face = self.unstr_discr.faces[cell_m][face_id]
                    # assert(self.discr_mesh.adj_matrix_cols[self.discr_mesh.adj_matrix_offset[cell_m] + face_id] == cell_p)
                    # assert(face.cell_id2 == cell_p or face.face_id2 + self.mesh.n_blocks == cell_p)
                    face_id += 1
                conn_id += 1

            assert(face_id == self.mat_flux[cell_m].shape[1])
            vels[cell_m] = self.mat_flux[cell_m].dot(rhs[:face_id])
        return vels

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
                    row += str(cells[i]) + '\t' + str('{:.2e}'.format(coefs[i])) + '\t'
                    #row_cells += str(cells[i]) + '\t'
                    #row_vals += str('{:.2e}'.format(coefs[i])) + '\t'
            f.write(row + '\n')# + row_cells + '\n' + row_vals + '\n')
        f.close()

    def write_to_vtk_old_discretizer(self, output_directory, cell_property, ith_step, physics):
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

        # Temporarily store mesh_data in copy:
        Mesh = meshio.read(self.unstr_discr.mesh_file)

        # Allocate empty new cell_data dictionary:
        props_num = len(cell_property)
        if props_num > physics.n_vars: props_num = physics.n_vars
        property_array = np.array(physics.engine.X, copy=False)
        available_matrix_geometries = ['hexahedron', 'wedge', 'tetra']
        available_fracture_geometries = ['quad', 'triangle']

        # Matrix
        geom_id = 0
        Mesh.cells = []
        cell_data = {}
        for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
            if ith_geometry in available_matrix_geometries:
                Mesh.cells.append(self.unstr_discr.mesh_data.cells[geom_id])
                for i in range(props_num):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(property_array[i:props_num * self.unstr_discr.mat_cells_tot:props_num])

                if ith_step == 0:
                    if 'perm' not in cell_data: cell_data['perm'] = []
                #     if 'cell_id' not in cell_data: cell_data['cell_id'] = []
                    cell_data['perm'].append(np.zeros((self.unstr_discr.mat_cells_tot, 3), dtype=np.float64))
                    cell_data['perm'][-1][:,0] = self.unstr_discr.perm_x_cell
                    cell_data['perm'][-1][:,1] = self.unstr_discr.perm_y_cell
                    cell_data['perm'][-1][:,2] = self.unstr_discr.perm_z_cell
                #     cell_data['cell_id'].append(np.zeros(len(cell_ids), dtype=np.int64))
                #     for i, cell_id in enumerate(cell_ids):
                #         cell_data['perm'][-1][i] = np.array(self.discr.perms[cell_id].values)
                #         cell_data['cell_id'][-1][i] = cell_id

                # if 'cell_id' not in cell_data:
                #     cell_data['cell_id'] = []
                # cell_data['cell_id'].append(np.array([cell_id for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items() if cell.geometry_type == ith_geometry], dtype=np.int64))

                # if 'vel' not in cell_data: cell_data['vel'] = []
                # vel = self.reconstruct_velocities(property_array[::2])
                # cell_data['vel'].append(vel)
            geom_id += 1

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            Mesh.points,
            Mesh.cells,
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

    def write_to_vtk(self, output_directory, cell_property, ith_step, physics):
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
        if props_num > physics.n_vars: props_num = physics.n_vars
        property_array = np.array(physics.engine.X, copy=False)
        available_matrix_geometries_cpp = [elem_type.HEX, elem_type.PRISM, elem_type.TETRA, elem_type.PYRAMID]
        available_fracture_geometries_cpp = [elem_type.QUAD, elem_type.TRI]
        available_matrix_geometries = {'hexahedron': elem_type.HEX,
                                       'wedge': elem_type.PRISM,
                                       'tetra': elem_type.TETRA,
                                       'pyramid': elem_type.PYRAMID}
        available_fracture_geometries = ['quad', 'triangle']

        fluxes = np.array(physics.engine.fluxes, copy=False)
        # vels = self.reconstruct_velocities(fluxes[physics.engine.P_VAR::physics.n_vars])

        # Matrix
        cells = []
        cell_data = {}
        for cell_block in self.mesh_data.cells:
            if cell_block.type in available_matrix_geometries:
                cells.append(cell_block)
                cell_ids = np.array(self.discr_mesh.elem_type_map[available_matrix_geometries[cell_block.type]], dtype=np.int64)
                for i in range(props_num):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(property_array[props_num * cell_ids + i])

                if ith_step == 0:
                    if 'perm' not in cell_data: cell_data['perm'] = []
                    if 'cell_id' not in cell_data: cell_data['cell_id'] = []
                    cell_data['perm'].append(np.zeros((len(cell_ids), 9), dtype=np.float64))
                    cell_data['cell_id'].append(np.zeros(len(cell_ids), dtype=np.int64))
                    for i, cell_id in enumerate(cell_ids):
                        cell_data['perm'][-1][i] = np.array(self.discr.perms[cell_id].values)
                        cell_data['cell_id'][-1][i] = cell_id

                # if 'cell_id' not in cell_data:
                #     cell_data['cell_id'] = []
                # cell_data['cell_id'].append(np.array([cell_id for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items() if cell.geometry_type == ith_geometry], dtype=np.int64))

                # if 'vel' not in cell_data:
                #     cell_data['vel'] = [vels]

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
