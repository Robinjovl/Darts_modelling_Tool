import numpy as np

from darts.engines import conn_mesh, index_vector, value_vector
from darts.engines import ms_well, ms_well_vector
from darts.engines import matrix33 as engine_matrix33
from darts.discretizer import matrix33 as disc_matrix33
from darts.engines import Stiffness as engine_stiffness
from darts.engines import matrix, Face, vector_face_vector, face_vector, vector_matrix33, stf_vector, critical_stress
from darts.discretizer import Stiffness as disc_stiffness
from darts.discretizer import Mesh, Elem, elem_loc, elem_type, conn_type
from darts.engines import pm_discretizer
from darts.discretizer import poro_mech_discretizer, thermoporo_mech_discretizer
from darts.discretizer import THMBoundaryCondition, BoundaryCondition
from darts.discretizer import vector_matrix33, vector_vector3, matrix, value_vector, index_vector

from darts.reservoirs.unstruct_reservoir import UnstructReservoir

class bound_cond:
    '''
    General representation of boundary condition: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
    '''
    def __init__(self):
        # flow
        self.NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        self.AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}

        # mechanics
        self.ROLLER = {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        self.FREE = {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        self.STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        # Fn, Ft are normal and tangential load [UNIT?]
        self.LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        # the same as ROLLER except rn is non-zero
        self.STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

def set_domain_tags(matrix_tags,
                    bnd_xm_tag, bnd_xp_tag,
                    bnd_ym_tag, bnd_yp_tag,
                    bnd_zm_tag, bnd_zp_tag,
                    fracture_tags=[], frac_bnd_tags=[]):
    '''
    :param matrix_tag: list of integers
    :param bnd_tags: list of integers
    :param fracture_tag: list of integers
    :param frac_bnd_tag: list of integers
    :return: dictionary of sets containing integer tags for each element type; dictionary of tags for 6 boundaries
    '''
    boundary_tags = [bnd_xm_tag, bnd_xp_tag, bnd_ym_tag, bnd_yp_tag, bnd_zm_tag, bnd_zp_tag]
    domain_tags = dict()
    domain_tags[elem_loc.MATRIX] = set(matrix_tags)
    domain_tags[elem_loc.FRACTURE] = set(fracture_tags)
    domain_tags[elem_loc.BOUNDARY] = set(boundary_tags)
    domain_tags[elem_loc.FRACTURE_BOUNDARY] = set(frac_bnd_tags)

    bnd_tags = dict()
    bnd_tags['BND_X-'] = bnd_xm_tag
    bnd_tags['BND_X+'] = bnd_xp_tag
    bnd_tags['BND_Y-'] = bnd_ym_tag
    bnd_tags['BND_Y+'] = bnd_yp_tag
    bnd_tags['BND_Z-'] = bnd_zm_tag
    bnd_tags['BND_Z+'] = bnd_zp_tag

    return domain_tags, bnd_tags


class UnstructReservoirMech(): #TODO: inherit from UnstructReservoir to have add_well functions from there
    '''
    Class for Poroelasticity/ThermoPoroElasticity coupled model
    '''
    def __init__(self, timer, discretizer='mech_discretizer', thermoporoelacticity=False):
        self.timer = timer
        self.discretizer_name = discretizer
        self.thermoporoelacticity = thermoporoelacticity
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        self.n_dim = 3
        self.bc_type = bound_cond()
    
        if thermoporoelacticity:
            self.cell_property = ['p', 't', 'ux', 'uy', 'uz']
            self.n_state = 2
            self.n_vars = 5
            self.t_var = 1
            self.u_var = 2
            self.p_var = 0
            assert (discretizer == 'mech_discretizer')
        else: # poroelasticity
            if discretizer == 'mech_discretizer':
                self.p_var = 0
                self.u_var = 1
                self.cell_property = ['p', 'ux', 'uy', 'uz']
            elif discretizer == 'pm_discretizer':
                self.u_var = 0
                self.p_var = self.n_dim
                self.cell_property = ['ux', 'uy', 'uz', 'p']
            self.n_vars = 4
            self.n_state = 1
            
    def init_arrays(self):
        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.bc = np.array(self.mesh.bc, copy=False)
        self.bc_prev = np.array(self.mesh.bc_prev, copy=False)
        self.bc_ref = np.array(self.mesh.bc_ref, copy=False)
        self.mesh.f.resize(self.n_vars * (self.n_fracs + self.n_matrix))
        self.f = np.array(self.mesh.f, copy=False)
        self.biot_arr = np.array(self.mesh.biot, copy=False)
        self.kd = np.array(self.mesh.kd, copy=False)
        self.mesh.pz_bounds.resize(self.n_state * self.n_bounds)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.p_ref = np.array(self.mesh.ref_pressure, copy=False)
        self.poro[:self.n_matrix] = self.porosity
        self.poro[self.n_matrix:] = 1  # fractures

        if self.discretizer_name == 'mech_discretizer':
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.volume[:self.n_matrix] = volumes[:self.n_matrix]
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            # self.biot_arr[:] = np.tile([0,0,0,
            #                             0,0,0,
            #                             0,0,0], self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.biot_arr[:] = self.biot_mean
            self.kd[:] = self.kd_cur
            self.pz_bounds[self.p_var::self.n_state] = self.p_init
            if self.thermoporoelacticity:
                self.pz_bounds[self.t_var::self.n_state] = self.t_init
            # self.pz_bounds[:] = self.pz_bounds
            # self.p_ref[:] = self.p_ref
            # self.f[:] = self.f
        elif self.discretizer_name == 'pm_discretizer':
            self.volume[:self.unstr_discr.mat_cells_tot] = self.unstr_discr.volume_all_cells[self.unstr_discr.frac_cells_tot:]
            for i in range(self.unstr_discr.mat_cells_tot, self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
                self.volume[i] = self.unstr_discr.faces[i][4].area * self.frac_apers[i-self.unstr_discr.mat_cells_tot]
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            self.bc_ref[:] = self.bc_rhs_ref
            # self.biot_arr[:] = np.tile([0,0,0,
            #                             0,0,0,
            #                             0,0,0], self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.biot_arr[:] = self.biot_mean
            self.kd[:] = self.kd_cur
            self.pz_bounds[:] = self.unstr_discr.pz_bounds
            self.p_ref[:] = self.unstr_discr.p_ref
            self.f[:] = self.unstr_discr.f

    def init_arrays_boundary_condition(self):
        if self.discretizer_name == 'mech_discretizer':
            # mapping boundary connections
            adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
            adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)
            id_sorted = np.argsort(adj_matrix_cols)[-self.n_bounds:]
            self.id_boundary_conns = adj_matrix[id_sorted]

            ap = np.ones(self.n_bounds)
            bp = np.zeros(self.n_bounds)
            amn = np.zeros(self.n_bounds)
            bmn = np.zeros(self.n_bounds)
            amt = np.zeros(self.n_bounds)
            bmt = np.zeros(self.n_bounds)
            at = np.zeros(self.n_bounds)
            bt = np.zeros(self.n_bounds)
            self.bc_rhs = np.zeros(self.n_vars * self.n_bounds)
            self.bc_rhs_prev = np.zeros(self.n_vars * self.n_bounds)

            for tag in self.domain_tags[elem_loc.BOUNDARY]:
                ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
                bc = self.boundary_conditions[tag]
                ap[ids] = bc['flow']['a']
                bp[ids] = bc['flow']['b']
                amn[ids] = bc['mech']['an']
                bmn[ids] = bc['mech']['bn']
                amt[ids] = bc['mech']['at']
                bmt[ids] = bc['mech']['bt']
                at[ids] = bc['temp']['a']
                bt[ids] = bc['temp']['b']
                # flow
                self.bc_rhs[self.n_vars * ids + self.p_var] = bc['flow']['r']
                # energy
                self.bc_rhs[self.n_vars * ids + self.t_var] = bc['temp']['r']
                # mechanics
                for id in ids:
                    assert(adj_matrix_cols[id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                    conn = self.conns[self.id_boundary_conns[id]]
                    n = np.array(conn.n.values, copy=False)
                    conn_c = np.array(conn.c.values, copy=False)
                    c1 = np.array(self.centroids[conn.elem_id1].values, copy=False)
                    if n.dot(conn_c - c1) < 0: n *= -1.0
                    self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']

            self.cpp_bc = THMBoundaryCondition()
            self.cpp_bc.flow.a = value_vector(ap)
            self.cpp_bc.flow.b = value_vector(bp)
            self.cpp_bc.mech_normal.a = value_vector(amn)
            self.cpp_bc.mech_normal.b = value_vector(bmn)
            self.cpp_bc.mech_tangen.a = value_vector(amt)
            self.cpp_bc.mech_tangen.b = value_vector(bmt)
            self.cpp_bc.thermal.a = value_vector(at)
            self.cpp_bc.thermal.b = value_vector(bt)
            # to use base discretizer's class function reconstruct_pressure_gradients_per_cell
            # which doesn't know the new THMBoundaryCondition class yet
            self.cpp_flow = BoundaryCondition()
            self.cpp_flow.a = value_vector(ap)
            self.cpp_flow.b = value_vector(bp)
            self.cpp_heat = BoundaryCondition()
            self.cpp_heat.a = value_vector(at)
            self.cpp_heat.b = value_vector(bt)


    def init_pm_discretizer(self):
        self.unstr_discr.x_new = np.ones((self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot, 4))
        self.unstr_discr.x_new[:, 0] = self.u_init[0]
        self.unstr_discr.x_new[:, 1] = self.u_init[1]
        self.unstr_discr.x_new[:, 2] = self.u_init[2]
        self.unstr_discr.x_new[:, 3] = self.p_init
        dt = 0.0
        self.pm.x_prev = value_vector(np.concatenate((self.unstr_discr.x_new.flatten(), self.bc_rhs_prev)))
        self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot,
                     index_vector(self.ref_contact_cells))
        self.pm.reconstruct_gradients_per_cell(dt)
        self.pm.calc_all_fluxes_once(dt)

        self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p,
                          self.pm.stencil, self.pm.offset,
                          self.pm.tran, self.pm.rhs,
                          self.pm.tran_biot, self.pm.rhs_biot,
                          self.unstr_discr.mat_cells_tot,
                          self.unstr_discr.bound_cells_tot,
                          self.unstr_discr.frac_cells_tot)
        self.unstr_discr.store_volume_all_cells()
        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_cells_tot

    def init_mech_discretizer(self):
        self.discr_mesh = Mesh()
        self.discr_mesh.gmsh_mesh_processing(self.mesh_filename, self.domain_tags)

        self.a = np.max([node.values[0] for node in self.discr_mesh.nodes])
        self.b = np.max([node.values[1] for node in self.discr_mesh.nodes])
        if self.thermoporoelacticity:
            self.discr = thermoporo_mech_discretizer()
        else:
            self.discr = poro_mech_discretizer()
        self.discr.grav_vec = matrix([0.0, 0.0, 0.0], 1, 3)  # 0.0??
        self.tags = np.array(self.discr_mesh.tags, copy=False)
        self.discr.set_mesh(self.discr_mesh)
        self.discr.init()

        self.n_matrix = self.discr_mesh.region_ranges[elem_loc.MATRIX][1] - \
                        self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
        self.n_fracs =  self.discr_mesh.region_ranges[elem_loc.FRACTURE][1] - \
                        self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
        self.n_bounds = self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - \
                        self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]

        self.conns = np.array(self.discr_mesh.conns, copy=False)
        self.centroids = np.array(self.discr_mesh.centroids, copy=False)

    def init_uniform_properties(self):
        if self.discretizer_name == 'mech_discretizer':
            self.biot_mean = np.zeros(9 * (self.n_matrix + self.n_fracs))
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                              self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
                self.discr.perms.append(disc_matrix33(self.permx, self.permy, self.permz))
                self.discr.biots.append(disc_matrix33(self.biot))
                self.discr.stfs.append(disc_stiffness(self.lam, self.mu))
                self.biot_mean[9 * cell_id] = self.biot
                self.biot_mean[9 * cell_id + 4] = self.biot
                self.biot_mean[9 * cell_id + 8] = self.biot
                if self.thermoporoelacticity:
                    self.discr.heat_conductions.append(disc_matrix33(self.th_conductivity))
                    self.discr.thermal_expansions.append(disc_matrix33(self.th_expn))
        elif self.discretizer_name == 'pm_discretizer':
            self.biot_mean = np.zeros(9 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
            for cell_id in range(self.unstr_discr.mat_cells_tot):
                faces = self.unstr_discr.faces[cell_id]
                fs = face_vector()
                for face_id in range(len(faces)):
                    face = faces[face_id]
                    fs.append(Face(face.type.value, face.cell_id1, face.cell_id2,
                                   face.face_id1, face.face_id2,
                                   face.area, list(face.n), list(face.centroid), index_vector(face.pts_id)))
                self.pm.faces.append(fs)

                cell = self.unstr_discr.mat_cell_info_dict[cell_id]
                self.pm.cell_centers.append(matrix(list(cell.centroid), cell.centroid.size, 1))
                self.pm.perms.append(engine_matrix33(self.permx, self.permy, self.permz))
                self.pm.biots.append(engine_matrix33(self.biot))
                self.pm.stfs.append(engine_stiffness(self.lam, self.mu))
                self.biot_mean[9 * cell_id] = self.biot
                self.biot_mean[9 * cell_id + 4] = self.biot
                self.biot_mean[9 * cell_id + 8] = self.biot

    def update_trans(self, dt, x):
        #self.pm.x_prev = value_vector(np.concatenate((x, self.bc_rhs_prev)))
        #self.pm.reconstruct_gradients_per_cell(dt)
        #self.pm.calc_all_fluxes(dt)
        #self.write_pm_conn_to_file(t_step=t_step)
        #self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
        #                  self.unstr_discr.mat_cells_tot, self.unstr_discr.bound_cells_tot, 0)

        # update transient sources / sinks
        # self.f[:] = self.unstr_discr.f
        # update boundaries at n+1 / n timesteps
        self.bc[:] = self.bc_rhs
        self.bc_prev[:] = self.bc_rhs_prev
        #self.init_wells()

    def update(self, dt, time):
        # update local array
        #if time > dt:
        self.bc_rhs_prev = np.copy(self.bc_rhs)

    def init_wells(self):
        # # Add wells to the DARTS mesh object and sort connection (DARTS related):
        self.mesh.add_wells_mpfa(ms_well_vector(self.wells), self.P_VAR)
        if self.discretizer_name == 'mech_discretizer':
            if self.n_vars == 4:
                self.mesh.reverse_and_sort_pm_mech_discretizer()
            elif self.n_vars == 5:
                self.mesh.reverse_and_sort_pme_mech_discretizer()
        elif self.discretizer_name == 'pm_discretizer':
            self.mesh.reverse_and_sort_pm()
        #self.mesh.init_grav_coef()
        return 0

    def get_normal_to_bound_face(self, b_id):
        assert self.discretizer_name == 'pm_discretizer'
        cell = self.unstr_discr.bound_cell_info_dict[b_id]
        cells = [self.unstr_discr.mat_cells_to_node[pt] for pt in cell.nodes_to_cell]
        cell_id = next(iter(set(cells[0]).intersection(*cells)))
        for face in self.unstr_discr.faces[cell_id].values():
            if face.cell_id1 == face.cell_id2 and face.face_id2 == b_id:
                t_face = cell.centroid - self.unstr_discr.mat_cell_info_dict[cell_id].centroid
                n = face.n
                if np.inner(t_face, n) < 0: n = -n
                return n

    def write_pm_conn_to_file(self, t_step, path='pm_conn.dat'):
        assert self.discretizer_name == 'pm_discretizer'
        #self.check_positive_negative_sides()
        path = 'pm_conn' + str(t_step) + '.dat'
        block_size = 4
        f = open(path, 'w')
        f.write(str(len(self.pm.cell_m)) + '\n')
        for i, cell_id1 in enumerate(self.pm.cell_m):
            cell_id2 = self.pm.cell_p[i]
            f.write(str(cell_id1) + '\t' + str(cell_id2) + '\n')

            for k in range(block_size):
                row = 'F' + str(k) + '\t{:.5e}'.format(self.pm.rhs[i * block_size + k])
                for j in range(self.pm.offset[i], self.pm.offset[i + 1]):
                    row += '\t' + str(self.pm.stencil[j]) + '\t[' + ', '.join(
                    ['{:.5e}'.format(self.pm.tran[n]) for n in range((j * block_size + k) * block_size, (j * block_size + k + 1) * block_size)]) + str(']')
                f.write(row + '\n')
            # Biot
            for k in range(block_size):
                row = 'b' + str(k) + '\t{:.5e}'.format(self.pm.rhs_biot[i * block_size + k])
                for j in range(self.pm.offset[i], self.pm.offset[i + 1]):
                    row += '\t' + str(self.pm.stencil[j]) + '\t[' + ', '.join(
                    ['{:.5e}'.format(self.pm.tran_biot[n]) for n in range((j * block_size + k) * block_size, (j * block_size + k + 1) * block_size)]) + str(']')
                f.write(row + '\n')
            # if self.pm.cell_p[i] < self.unstr_discr.mat_cells_tot:
            #     st = np.array(self.pm.stencil[self.pm.offset[i]:self.pm.offset[i+1]],dtype=np.intp)
            #     all_trans_biot = np.array(self.pm.tran_biot)[(self.pm.offset[i] * block_size) * block_size: (self.pm.offset[i + 1] * block_size) * block_size].reshape(self.pm.offset[i + 1] - self.pm.offset[i], block_size, block_size)
            #     sum = np.sum(all_trans_biot, axis=0)
            #     #sum_no_bound = np.sum(all_trans[st < self.unstr_discr.mat_cells_tot], axis=0)
            #     assert((abs(sum[:3,:3]) < 1.E-10).all())
        f.close()


    def write_to_vtk_mech_discretizer(self, output_directory, ith_step, engine):
        """
        Class method which writes output of unstructured grid to VTK format
        :param output_directory: directory of output files
        :param property_array: np.array containing all cell properties (N_cells x N_prop)
        :param cell_property: list with property names (visible in ParaView (format strings)
        :param ith_step: integer containing the output step
        :return:
        """
        assert self.discretizer_name == 'mech_discretizer'

        # First check if output directory already exists:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Temporarily store mesh_data in copy:
        Mesh = meshio.read(self.mesh_filename)

        # Allocate empty new cell_data dictionary:
        property_array = np.array(engine.X, copy=False)
        available_matrix_geometries = ['hexahedron', 'wedge', 'tetra']
        available_fracture_geometries = ['quad', 'triangle']

        # Stresses and velocities
        engine.eval_stresses_and_velocities()
        total_stresses = np.array(engine.total_stresses, copy=False)
        effective_stresses = np.array(engine.effective_stresses, copy=False)
        darcy_velocities = np.array(engine.darcy_velocities, copy=False)

        # Matrix
        geom_id = 0
        Mesh.cells = []
        cell_data = {}
        for ith_geometry in self.mesh_data.cells_dict.keys():
            if ith_geometry in available_matrix_geometries:
                Mesh.cells.append(self.mesh_data.cells[geom_id])
                # Add unknowns to dictionary:
                for i in range(self.n_vars):
                    if self.cell_property[i] not in cell_data: cell_data[self.cell_property[i]] = []
                    cell_data[self.cell_property[i]].append(property_array[i:self.n_vars * self.n_matrix:self.n_vars])

                # Add post-processed data to dictionary
                if 'velocity' not in cell_data: cell_data['velocity'] = []
                cell_data['velocity'].append(np.zeros((self.n_matrix, 3), dtype=np.float64))
                for i in range(3):
                    cell_data['velocity'][-1][:, i] = darcy_velocities[i::3]
                if 'stress' not in cell_data: cell_data['stress'] = []
                cell_data['stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []
                cell_data['tot_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for i in range(6):
                    cell_data['stress'][-1][:, i] = effective_stresses[i::6]
                    cell_data['tot_stress'][-1][:, i] = total_stresses[i::6]

            geom_id += 1

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            Mesh.points,
            Mesh.cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        return 0

    def write_to_vtk_pm_discretizer(self, output_directory, ith_step, engine):
        """
        Class method which writes output of unstructured grid to VTK format
        :param output_directory: directory of output files
        :param property_array: np.array containing all cell properties (N_cells x N_prop)
        :param cell_property: list with property names (visible in ParaView (format strings)
        :param ith_step: integer containing the output step
        :return:
        """
        assert self.discretizer_name == 'pm_discretizer'

        # First check if output directory already exists:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Temporarily store mesh_data in copy:
        Mesh = meshio.read(self.unstr_discr.mesh_file)

        # Allocate empty new cell_data dictionary:
        cell_property = ['u_x', 'u_y', 'u_z', 'p']
        props_num = len(cell_property)
        property_array = np.array(engine.X, copy=False)
        available_matrix_geometries = ['hexahedron', 'wedge', 'tetra']
        available_fracture_geometries = ['quad', 'triangle']

        # if ith_step != 0:
        fluxes = np.array(engine.fluxes, copy=False)
        # fluxes_n = np.array(physics.engine.fluxes_n, copy=False)
        fluxes_biot = np.array(engine.fluxes_biot, copy=False)
        #vels = self.reconstruct_velocities(fluxes[physics.engine.P_VAR::physics.engine.N_VARS],
        #                                  fluxes_biot[physics.engine.P_VAR::physics.engine.N_VARS])
        self.mech_operators.eval_porosities(engine.X, self.mesh.bc)
        self.mech_operators.eval_stresses(engine.fluxes, engine.fluxes_biot, engine.X,
                                          self.mesh.bc, engine.op_vals_arr)
        # else:
        #    self.mech_operators.eval_porosities(physics.engine.X, self.mesh.bc_prev)
        #    self.mech_operators.eval_stresses(physics.engine.X, self.mesh.bc_prev, physics.engine.op_vals_arr)

        # Matrix
        geom_id = 0
        Mesh.cells = []
        cell_data = {}
        for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
            if ith_geometry in available_matrix_geometries:
                Mesh.cells.append(self.unstr_discr.mesh_data.cells[geom_id])
                # Add matrix data to dictionary:
                for i in range(props_num):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(property_array[i:props_num * self.unstr_discr.mat_cells_tot:props_num])

                #if 'velocity' not in cell_data: cell_data['velocity'] = []
                #cell_data['velocity'].append(vels)
                # if hasattr(self.unstr_discr, 'E') and hasattr(self.unstr_discr, 'nu'):
                #     cell_data[ith_geometry]['E'] = np.zeros(self.unstr_discr.mat_cells_tot, dtype=np.float64)
                #     cell_data[ith_geometry]['nu'] = np.zeros(self.unstr_discr.mat_cells_tot, dtype=np.float64)
                #     for id, cell in enumerate(self.unstr_discr.mat_cell_info_dict.values()):
                #         cell_data[ith_geometry]['E'][id] = self.unstr_discr.E[cell.prop_id]
                #         cell_data[ith_geometry]['nu'][id] = self.unstr_discr.nu[cell.prop_id]
                if 'eps_vol' not in cell_data: cell_data['eps_vol'] = []
                if 'porosity' not in cell_data: cell_data['porosity'] = []
                if 'stress' not in cell_data: cell_data['stress'] = []
                if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []

                cell_data['eps_vol'].append(np.array(self.mech_operators.eps_vol, copy=False))
                cell_data['porosity'].append(np.array(self.mech_operators.porosities, copy=False))
                cell_data['stress'].append(np.zeros((self.unstr_discr.mat_cells_tot, 6), dtype=np.float64))
                cell_data['tot_stress'].append(np.zeros((self.unstr_discr.mat_cells_tot, 6), dtype=np.float64))

                stress = np.array(self.mech_operators.stresses, copy=False)
                total_stress = np.array(self.mech_operators.total_stresses, copy=False)
                for i in range(6):
                    cell_data['stress'][-1][:, i] = stress[i::6]
                    cell_data['tot_stress'][-1][:, i] = total_stress[i::6]

                if 'cell_id' not in cell_data: cell_data['cell_id'] = []
                cell_data['cell_id'].append(np.array([cell_id for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items() if cell.geometry_type == ith_geometry], dtype=np.int64))
                # if ith_step == 0:
                #     cell_data[ith_geometry]['permx'] = self.permx[:]
                #     cell_data[ith_geometry]['permy'] = self.permy[:]
                #     cell_data[ith_geometry]['permz'] = self.permz[:]
            geom_id += 1

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            Mesh.points,
            Mesh.cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        return 0