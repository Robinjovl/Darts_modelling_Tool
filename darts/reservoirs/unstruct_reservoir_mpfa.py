import numpy as np
import meshio

from darts.engines import conn_mesh, ms_well, ms_well_vector, timer_node
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.discretizer import Mesh, Elem, Discretizer, BoundaryCondition, elem_loc, elem_type, \
    matrix33, vector_matrix33, vector_vector3, matrix, value_vector, index_vector


class UnstructReservoirMPFA(UnstructReservoir):

    def __init__(self, timer: timer_node, mesh_file: str, permx, permy, permz, poro, rcond=0, hcap=0,
                 frac_aper=0, n_vars=0):
        super().__init__(timer=timer, mesh_file=mesh_file,
                         permx=permx, permy=permy, permz=permz, poro=poro, hcap=hcap, rcond=rcond)

        if self.discr_type == 'tpfa_py':
            return

        self.mesh_file = mesh_file
        self.discr_type = 'mpfa'
        self.n_vars = n_vars
        self.frac_aper = frac_aper

        self.permx_value = permx
        self.permy_value = permy
        self.permz_value = permz
        self.poro_value = poro

        self.heat_cond = rcond
        self.heat_cap = hcap


    def discretize(self):
        if self.discr_type == 'tpfa_py':
            return
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()

        self.mesh_data = meshio.read(self.mesh_file)

        # physical tags from Gmsh scripts (see meshes/*.geo)
        domain_tags = dict()
        domain_tags[elem_loc.MATRIX] = set([900001, 900002])
        domain_tags[elem_loc.FRACTURE] = set([])  # 9991, 9992])
        domain_tags[elem_loc.BOUNDARY] = set([90001, 90002, 90003, 90004, 90006, 90007])
        domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()  # is this for poromechanics??

        self.discr_mesh = Mesh()

        # for i, block in enumerate(self.mesh_data.cell_data['gmsh:physical']):
        #     self.n_fracs += np.isin(self.mesh_data.cell_data['gmsh:physical'][i],
        #                        list(domain_tags[elem_loc.FRACTURE])).sum()
        # self.discr_mesh.init_apertures = value_vector(frac_aper * np.ones(self.n_fracs))

        self.discr_mesh.gmsh_mesh_processing(self.mesh_file, domain_tags)

        self.discretizer = Discretizer()
        self.discretizer.grav_vec = matrix([0.0, 0.0, 0.0], 1, 3)  # 0.0??
        self.tags = np.array(self.discr_mesh.tags, copy=False)
        self.cpp_bc = self.set_boundary_conditions(domain_tags)
        self.discretizer.set_mesh(self.discr_mesh)
        self.discretizer.init()

        self.n_matrix = self.discr_mesh.region_ranges[elem_loc.MATRIX][1] - \
                        self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
        self.n_fracs = self.discr_mesh.region_ranges[elem_loc.FRACTURE][1] - self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
        self.n_bounds = self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - \
                        self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]

        self.boundary_cells = self.discr_mesh.get_boundary_cells()

        self.depth_all_cells = np.zeros(self.n_matrix + self.n_fracs + self.n_bounds)
        self.volume_all_cells = np.zeros(self.n_matrix + self.n_fracs)
        self.porosity = np.zeros(self.n_matrix + self.n_fracs)

        centroids = np.array(self.discr_mesh.centroids, copy=False)
        volumes = np.array(self.discr_mesh.volumes, copy=False)
        # loop over matrix elements
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            c = centroids[cell_id].values
            self.discretizer.perms.append(matrix33(self.permx_value, self.permy_value, self.permz_value))
            self.discretizer.heat_conductions.append(matrix33(self.heat_cond, self.heat_cond, self.heat_cond))
            self.porosity[i] = self.poro_value
            self.depth_all_cells[i] = c[2]
            self.volume_all_cells[i] = volumes[cell_id]

        # loop over fracture elements
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.FRACTURE][0],
                                          self.discr_mesh.region_ranges[elem_loc.FRACTURE][1])):
            c = centroids[cell_id].values
            self.discretizer.perms.append(matrix33(self.permx_value, self.permy_value, self.permz_value)) #TODO ?
            #TODO heat_conductions ?
            self.porosity[self.n_matrix + i] = 1.
            self.depth_all_cells[self.n_matrix + i] = c[2]
            self.volume_all_cells[self.n_matrix + i] = volumes[cell_id]

        if self.discr_type == 'tpfa':
            self.discretizer.calc_tpfa_transmissibilities(domain_tags)
        elif self.discr_type == 'mpfa':
            self.discretizer.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_bc)
            self.discretizer.calc_mpfa_transmissibilities(self.cpp_bc, True)

        # self.pz_bounds = np.zeros(self.n_vars * self.n_bounds)
        # self.pz_bounds[::self.n_vars] = self.p_init
        # self.pz_bounds[1::self.n_vars] = self.s_init
        self.bc_input = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))
        bc_r_p = np.array(self.cpp_bc.r_p, copy=False)
        bc_r_th = np.array(self.cpp_bc.r_th, copy=False)
        # loop over boundaries
        for i, bound_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0], self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1])):
            # elem = self.discr_mesh.elems[bound_id]
            c = centroids[bound_id].values
            self.depth_all_cells[bound_id] = c[2]
            #cur_a = bc_a[i]
            #cur_b = bc_b[i]
            #cur_r = bc_r[i]
            self.bc_input[self.n_vars * i] = bc_r_p[i]
            self.bc_input[self.n_vars * i + self.n_vars - 1] = bc_r_th[i]
            #self.bc_flow.extend([cur_a, cur_b, cur_r])
            #TODO use merged a,b,r

            #if cur_a == 1 and cur_b == 0:
            #    self.pz_bounds[id * self.n_vars] = cur_r
            #    self.pz_bounds[id * self.n_vars + 1] = self.s_init

        self.mesh.init_mpfa(self.discretizer.cell_m, self.discretizer.cell_p,
                            self.discretizer.flux_stencil, self.discretizer.flux_offset,
                            self.discretizer.flux_vals, self.discretizer.flux_rhs, self.discretizer.flux_vals_homo,
                            self.discretizer.flux_vals_thermal,
                            self.n_matrix, self.n_bounds, self.n_fracs, self.n_vars)

        self.bc = np.array(self.mesh.bc, copy=False)
        self.depth = np.array(self.mesh.depth, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.depth[:] = self.depth_all_cells
        self.volume[:] = self.volume_all_cells
        self.bc[:self.bc_input.size] = self.bc_input

        # Write to files (in case someone needs this for Eclipse or other simulator):
        #self.unstr_discr.write_volume_to_file(file_name='vol.dat')
        #self.unstr_discr.write_depth_to_file(file_name='depth.dat')

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.poro[:] = self.porosity
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.hcap.fill(self.heat_cap)
        self.conduction = np.array(self.mesh.rock_cond, copy=False)
        self.conduction.fill(self.heat_cond)

        self.wells = []


    def init_wells(self):
        if self.discr_type == 'tpfa_py':
            super().init_wells()
        elif self.discr_type == 'tpfa':
            self.mesh.add_wells(ms_well_vector(self.wells))
            self.mesh.reverse_and_sort()
            self.mesh.init_grav_coef()
        elif self.discr_type == 'mpfa':
            self.mesh.add_wells_mpfa(ms_well_vector(self.wells), 0) #TODO self.P_VAR)
            self.mesh.reverse_and_sort_mpfa()
            self.mesh.init_grav_coef()

    def init_reservoir(self, verbose=False):
        super().init_reservoir(verbose=verbose)

    def set_boundary_conditions(self, physical_tags):
        bc = BoundaryCondition()

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        a = np.zeros(boundary_range[1] - boundary_range[0])
        b = np.ones(boundary_range[1] - boundary_range[0])
        r = np.zeros(boundary_range[1] - boundary_range[0])

        # P_MINUS = 300.0

        # top = self.tags[boundary_range[0]:boundary_range[1]] == 994
        # a[top] = 1.0
        # b[top] = 0.0

        # grav = -self.discretizer.grav_vec.values[1]
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
        bc.a_p = value_vector(a)
        bc.b_p = value_vector(b)
        bc.r_p = value_vector(r)
        bc.a_th = value_vector(a)
        bc.b_th = value_vector(b)
        bc.r_th = value_vector(r)


        return bc

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
        stencil = np.array(self.discretizer.flux_stencil, copy=False)
        trans = np.array(self.discretizer.flux_vals, copy=False)

        f = open(path, 'w')
        f.write(str(len(self.discretizer.cell_m)) + '\n')

        for conn_id in range(len(self.discretizer.cell_m)):
            cells = stencil[self.discretizer.flux_offset[conn_id]:self.discretizer.flux_offset[conn_id + 1]]
            coefs= trans[self.discretizer.flux_offset[conn_id]:self.discretizer.flux_offset[conn_id + 1]]
            #row = str(self.discretizer.cell_m[conn_id]) + '\t' + str(self.discretizer.cell_p[conn_id])
            row = str(self.discretizer.cell_m[conn_id]) + '\t' + str(self.discretizer.cell_p[conn_id]) + '\t\t'
            #row_cells = ''#str(cells)
            #row_vals = ''#str(coefs)
            for i in range(cells.size):
                if np.abs(coefs[i]) > 1.E-10:
                    row += str(cells[i]) + '\t' + str('{:.2e}'.format(coefs[i])) + '\t'
                    #row_cells += str(cells[i]) + '\t'
                    #row_vals += str('{:.2e}'.format(coefs[i])) + '\t'
            f.write(row + '\n')# + row_cells + '\n' + row_vals + '\n')
        f.close()


    def write_to_vtk(self, output_directory, cell_property, ith_step, engine):
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
        if props_num > self.n_vars: props_num = self.n_vars
        property_array = np.array(engine.X, copy=False)
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
