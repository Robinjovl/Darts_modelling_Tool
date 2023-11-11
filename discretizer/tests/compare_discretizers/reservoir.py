from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, contact, contact_vector, vector_matrix, scheme_type
from darts.engines import matrix33 as engine_matrix33
from darts.discretizer import matrix33 as disc_matrix33
from darts.engines import Stiffness as engine_stiffness
from darts.discretizer import Stiffness as disc_stiffness
from darts.engines import matrix, pm_discretizer, Face, vector_face_vector, face_vector, vector_matrix33, stf_vector, critical_stress
import numpy as np
from math import inf, pi
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.mesh.geometrymodule import FType
from darts.engines import timer_node
from itertools import compress
import meshio
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.linalg import null_space
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
import scipy.optimize as opt
import scipy
from scipy.special import erfc as erfc

import darts.discretizer as dis
from darts.discretizer import Mesh, Elem, poro_mech_discretizer, THMBoundaryCondition, BoundaryCondition, elem_loc, elem_type, conn_type
from darts.discretizer import vector_matrix33, vector_vector3, matrix, value_vector, index_vector

# Definitions for the unstructured reservoir class:
class UnstructReservoir:
    def __init__(self, discretizer='new_discretizer', mesh='rect'):
        self.discretizer_name = discretizer
        self.n_vars = 4
        self.n_dim = 3

        if mesh == 'rect':
            self.mesh_path = 'meshes/unit_trans.msh'
        elif mesh == 'tetra':
            self.mesh_path = 'meshes/unit_tetra.msh'

        # material properties
        self.perm = [25,    2,      39,
                2,     42,     7,
                39,    7,      100]
        self.biot = [1,     6,      5,
                6,     67,     27,
                5,     27,     76]
        self.stf =  [93,     46,     22,     13,     72,     35,
                46,     95,     41,     62,     56,     24,
                22,     41,     89,     25,     33,     21,
                13,     62,     25,     87,     13,     25,
                72,     56,     33,     13,     99,     57,
                35,     24,     21,     25,     57,     78]

        if discretizer == 'new_discretizer':
            self.unit_cube_new_discretizer()

            self.offset = np.array(self.discr.flux_offset, copy=False)
            self.stencil = np.array(self.discr.flux_stencil, copy=False)
            self.hooke_trans = np.array(self.discr.hooke, copy=False)
            self.hooke_rhs = np.array(self.discr.hooke_rhs, copy=False)
            self.biot_traction_trans = np.array(self.discr.biot_traction, copy=False)
            self.biot_traction_rhs = np.array(self.discr.biot_traction_rhs, copy=False)
            self.biot_vol_strain_trans = np.array(self.discr.biot_vol_strain, copy=False)
            self.biot_vol_strain_rhs = np.array(self.discr.biot_vol_strain_rhs, copy=False)
            self.darcy_trans = np.array(self.discr.darcy, copy=False)
            self.darcy_rhs = np.array(self.discr.darcy_rhs, copy=False)
            self.fick_trans = np.array(self.discr.fick, copy=False)
            self.fick_rhs = np.array(self.discr.fick_rhs, copy=False)

        elif discretizer == 'pm_discretizer':
            self.unit_cube_pm_discretizer()

            dt = 0.0
            self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot, index_vector([]))
            self.pm.reconstruct_gradients_per_cell(dt)
            self.pm.calc_all_fluxes_once(dt)

            self.offset = np.array(self.pm.offset, copy=False)
            self.stencil = np.array(self.pm.stencil, copy=False)
            self.tran = np.array(self.pm.tran, copy=False)
            self.rhs = np.array(self.pm.rhs, copy=False)
            self.tran_biot = np.array(self.pm.tran_biot, copy=False)
            self.rhs_biot = np.array(self.pm.rhs_biot, copy=False)

        self.W = np.zeros((9, 6))
        self.W[0, 0] = 1.0
        self.W[1, 5] = 1.0
        self.W[2, 4] = 1.0
        self.W[3, 5] = 1.0
        self.W[4, 1] = 1.0
        self.W[5, 3] = 1.0
        self.W[6, 4] = 1.0
        self.W[7, 3] = 1.0
        self.W[8, 2] = 1.0

    # new discretizer
    def unit_cube_new_discretizer(self):
        # assign tags
        domain_tags = dict()
        domain_tags[elem_loc.MATRIX] = set([99991])
        domain_tags[elem_loc.FRACTURE] = set([])
        domain_tags[elem_loc.BOUNDARY] = set([991, 992, 993, 994, 995, 996])
        domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()

        # initialize mesh
        self.discr_mesh = Mesh()
        self.discr_mesh.gmsh_mesh_processing(self.mesh_path, domain_tags)

        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        self.boundary_conditions = {}
        self.boundary_conditions[991] = { 'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]) }
        self.boundary_conditions[992] = { 'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]) }
        self.boundary_conditions[993] = { 'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]) }
        self.boundary_conditions[994] = { 'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]) }
        self.boundary_conditions[995] = { 'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]) }
        self.boundary_conditions[996] = { 'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]) }

        # initialize poromechanics discretizer
        self.discr = poro_mech_discretizer()
        self.discr.grav_vec = matrix([0.0, 0.0, 0.0], 1, 3)  # 0.0??
        self.tags = np.array(self.discr_mesh.tags, copy=False)
        self.discr.set_mesh(self.discr_mesh)
        self.discr.init()

        self.n_matrix = self.discr_mesh.region_ranges[elem_loc.MATRIX][1] - \
                        self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
        self.n_fracs = 0 # self.discr_mesh.region_ranges[elem_loc.FRACTURE][1] - self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
        self.n_bounds = self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - \
                        self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]

        self.solution = np.zeros(self.n_vars * (self.n_matrix + self.n_bounds))
        # filling material properties
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            self.discr.perms.append(disc_matrix33(self.perm))
            self.discr.biots.append(disc_matrix33(self.biot))
            self.discr.stfs.append(disc_stiffness(self.stf))
            self.solution[self.n_vars * cell_id : self.n_vars * (cell_id + 1)] = \
                ref1(np.append(np.array(self.discr_mesh.centroids[cell_id].values), 0.0))

        ap = np.ones(self.n_bounds)
        bp = np.zeros(self.n_bounds)
        an = np.ones(self.n_bounds)
        bn = np.zeros(self.n_bounds)
        at = np.ones(self.n_bounds)
        bt = np.zeros(self.n_bounds)

        # right-hand side of boundary conditions
        for i, bound_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0],
                                           self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1])):
            c = np.array(self.discr_mesh.centroids[bound_id].values, copy=False)
            bc = self.boundary_conditions[self.discr_mesh.tags[bound_id]]
            self.solution[self.n_vars * bound_id: self.n_vars * (bound_id + 1)] = ref1(np.append(c, 0.0))
        # specify boundary conditions, loop over tags for speedup
        for tag in domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            ap[ids] = bc['flow']['a']
            bp[ids] = bc['flow']['b']
            an[ids] = bc['mech']['an']
            bn[ids] = bc['mech']['bn']
            at[ids] = bc['mech']['at']
            bt[ids] = bc['mech']['bt']

        self.cpp_bc = THMBoundaryCondition()
        self.cpp_bc.flow.a = value_vector(ap)
        self.cpp_bc.flow.b = value_vector(bp)

        self.cpp_bc.mech_normal.a = value_vector(an)
        self.cpp_bc.mech_normal.b = value_vector(bn)
        self.cpp_bc.mech_tangen.a = value_vector(at)
        self.cpp_bc.mech_tangen.b = value_vector(bt)

        self.cpp_flow = BoundaryCondition()
        self.cpp_flow.a_p = value_vector(ap)
        self.cpp_flow.b_p = value_vector(bp)

        # gradient reconstruction
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_mpfa_mpsa_transmissibilities()

    # old discretizer
    def unit_cube_pm_discretizer(self):
        self.permx = self.permy = self.permz = 10.0 # any value
        self.unstr_discr = UnstructDiscretizer(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0,
                                               mesh_file=self.mesh_path)

        #self.unstr_discr.init_matrix_stiffness({99991: {'E': self.E, 'nu': self.nu}})
        self.unstr_discr.physical_tags['matrix'] = [99991]
        self.unstr_discr.physical_tags['fracture'] = []
        self.unstr_discr.physical_tags['fracture_shape'] = []
        self.unstr_discr.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996]
        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)

        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER = {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE = {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0,
                                   'rt': np.array([0.0, 0.0, 0.0])}

        self.unstr_discr.boundary_conditions[991] = {'flow': AQUIFER(0.0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': AQUIFER(0.0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': AQUIFER(0.0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': AQUIFER(0.0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': AQUIFER(0.0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': AQUIFER(0.0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.pm.neumann_boundaries_grad_reconstruction = True
        self.pm.grav = matrix([0.0, 0.0, 0.0], 1, 3)
        self.pm.visc = 1  # 9.81e-2
        self.solution = np.zeros(self.n_vars * (self.unstr_discr.mat_cells_tot + self.unstr_discr.bound_cells_tot))
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
            self.pm.perms.append(engine_matrix33(self.perm))
            self.pm.biots.append(engine_matrix33(self.biot))
            self.pm.stfs.append(engine_stiffness(self.stf))
            self.solution[self.n_vars * cell_id : self.n_vars * (cell_id + 1)] = ref1(np.append(cell.centroid, 0.0))

        for bound_id in range(self.unstr_discr.bound_cells_tot):
            b_cell = self.unstr_discr.bound_cell_info_dict[bound_id]
            mech = self.unstr_discr.boundary_conditions[b_cell.prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[b_cell.prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            cell_id = self.unstr_discr.mat_cells_tot + bound_id
            sol = ref1(np.append(b_cell.centroid, 0.0))
            self.solution[self.n_vars * cell_id:self.n_vars * (cell_id + 1)] = sol

    # calculate gradients, old discretizer
    def get_gradients_pm_discretizer(self, cell_id: int):
        st, coef = self.pm.get_gradient(cell_id)
        stencil = np.array(st, copy=False)
        stencil_cols = np.concatenate([
            np.arange(i * self.n_vars, i * self.n_vars + self.n_vars) for i in stencil])
        trans = np.array(coef, copy=False).reshape(3 * self.n_vars, stencil.size * self.n_vars)
        assert((np.sum(trans, axis=1) < 1.e-8).all())

        grad = trans.dot(self.solution[stencil_cols])
        return grad
    # calculate gradients, new discretizer
    def get_gradients_new_discretizer(self, cell_id: int):
        p_grad = self.discr.p_grads[cell_id]
        u_grad = self.discr.u_grads[cell_id]
        stencil_cols = np.concatenate([
            np.arange(i * self.n_vars, i * self.n_vars + self.n_vars) for i in u_grad.stencil])

        p_trans = np.array(p_grad.a.values).reshape(3, len(p_grad.stencil))
        u_trans = np.array(u_grad.a.values, copy=False).\
            reshape(3 * 3, len(u_grad.stencil) * self.n_vars)
        assert((np.sum(p_trans, axis=1) < 1.e-8).all())
        assert((np.sum(u_trans, axis=1) < 1.e-8).all())
        nabla_u = u_trans.dot(self.solution[stencil_cols])
        nabla_p = p_trans.dot(self.solution[self.n_vars * np.array(p_grad.stencil) + 3])

        return np.append(nabla_u, nabla_p)
    # calculate analytical fluxes
    def get_analytical_fluxes(self, x, n):
        grad_an = nabla_ref1(x)
        stf = np.array(self.stf).reshape(6, 6)
        hooke_stress = self.W.dot(stf.dot(self.W.T)).\
            dot(grad_an[:self.n_dim, :self.n_dim].flatten()).\
            reshape(self.n_dim, self.n_dim)
        hooke_traction = hooke_stress.dot(n)
        return -hooke_traction
    # calculate fluxes, old discretizer
    def get_fluxes_pm_discretizer(self, flux_id):
        n_block = 4
        stencil = self.stencil[self.offset[flux_id]:
                               self.offset[flux_id + 1]]
        stencil_cols = np.concatenate([
            np.arange(i * self.n_vars, i * self.n_vars + self.n_vars) for i in stencil])
        main_terms = self.tran[n_block * n_block * self.offset[flux_id]:
                                 n_block * n_block * self.offset[flux_id + 1]].\
            reshape((stencil.size, n_block, n_block))
        main_terms = np.transpose(main_terms, (1, 0, 2)).reshape(n_block, n_block * stencil.size)
        hooke_coefs = main_terms[:self.n_dim, :]
        main_rhs = self.rhs[n_block * flux_id:n_block * (flux_id + 1)]
        hooke = hooke_coefs.dot(self.solution[stencil_cols]) + main_rhs[:self.n_dim]

        return hooke
    # calculate fluxes, new discretizer
    def get_fluxes_new_discretizer(self, flux_id):
        n_block = self.n_vars # for poroelastic mode in discretizer
        n_hooke = n_block * self.n_dim
        stencil = self.stencil[self.offset[flux_id]:
                               self.offset[flux_id + 1]]
        if stencil.size > 0:
            stencil_cols = np.concatenate([
                np.arange(i * self.n_vars, i * self.n_vars + self.n_vars) for i in stencil])
            hooke_coefs = self.hooke_trans[n_hooke * self.offset[flux_id]:
                                           n_hooke * self.offset[flux_id + 1]].\
                    reshape((stencil.size, self.n_dim, n_block))
            hooke_coefs = np.transpose(hooke_coefs, (1, 0, 2)).reshape(self.n_dim, n_block * stencil.size)

            hooke_rhs = self.hooke_rhs[self.n_dim * flux_id:self.n_dim * (flux_id + 1)]
            hooke = hooke_coefs.dot(self.solution[stencil_cols]) + hooke_rhs
        else:
            hooke = np.array([0.0, 0.0, 0.0])
        return hooke

    def get_normal_to_bound_face(self, b_id):
        cell = self.unstr_discr.bound_cell_info_dict[b_id]
        cells = [self.unstr_discr.mat_cells_to_node[pt] for pt in cell.nodes_to_cell]
        cell_id = next(iter(set(cells[0]).intersection(*cells)))
        for face in self.unstr_discr.faces[cell_id].values():
            if face.cell_id1 == face.cell_id2 and face.face_id2 == b_id:
                t_face = cell.centroid - self.unstr_discr.mat_cell_info_dict[cell_id].centroid
                n = face.n
                if np.inner(t_face, n) < 0: n = -n
                return n


# reference solution
def ref1(x):
    A = np.array([[1, 2, 3, 4],
                  [6, 7, 8, 9],
                  [11, 12, 13, 14],
                  [16, 17, 18, 19]])
    b = np.array([5, 10, 15, 20])
    if len(x.shape) == 1:
        return A.dot(x) + b
    else:
        return A.dot(x) + b[:,np.newaxis]

def nabla_ref1(x):
    A = np.array([[1, 2, 3, 4],
                  [6, 7, 8, 9],
                  [11, 12, 13, 14],
                  [16, 17, 18, 19]])
    return A