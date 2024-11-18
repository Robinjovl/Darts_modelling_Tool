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
from darts.discretizer import Mesh, Elem, poro_mech_discretizer, thermoporo_mech_discretizer, THMBoundaryCondition, BoundaryCondition, elem_loc, elem_type, conn_type
from darts.discretizer import vector_matrix33, vector_vector3, matrix, value_vector, index_vector

# Definitions for the unstructured reservoir class:
class UnstructReservoir:
    def __init__(self, discretizer='new_discretizer', mesh='rect', thermal=False):
        self.discretizer_name = discretizer
        self.n_dim = 3
        self.fluid_density = 1000.0
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.thermal = thermal
        if not self.thermal:
            self.n_vars = 4
        else:
            self.n_vars = 5
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
        self.conduction = [25,    2,      39,
                2,     42,     7,
                39,    7,      100]
        self.therm_expn = [1,     6,      5,
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
            if self.thermal:
                self.thermal_traction_trans = np.array(self.discr.thermal_traction, copy=False)
                self.fourier_trans = np.array(self.discr.fourier, copy=False)
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

        if self.thermal: # no heat flux boundary condition
            self.boundary_conditions[991]['heat'] = AQUIFER(0)
            self.boundary_conditions[992]['heat'] = AQUIFER(0)
            self.boundary_conditions[993]['heat'] = AQUIFER(0)
            self.boundary_conditions[994]['heat'] = AQUIFER(0)
            self.boundary_conditions[995]['heat'] = AQUIFER(0)
            self.boundary_conditions[996]['heat'] = AQUIFER(0)

        # initialize poromechanics discretizer
        if self.thermal:
            self.discr = thermoporo_mech_discretizer()
        else:
            self.discr = poro_mech_discretizer()
        self.discr.grav_vec = matrix(list(self.gravity), 1, 3)  # 0.0??
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
            x = np.append(np.array(self.discr_mesh.centroids[cell_id].values), 0.)
            self.solution[self.n_vars * cell_id : self.n_vars * (cell_id + 1)] = self.ref1(x)
            if self.thermal:
                self.discr.heat_conductions.append(disc_matrix33(self.conduction))
                self.discr.thermal_expansions.append(disc_matrix33(self.therm_expn))

        ap = np.ones(self.n_bounds)
        bp = np.zeros(self.n_bounds)
        amn = np.ones(self.n_bounds)
        bmn = np.zeros(self.n_bounds)
        amt = np.ones(self.n_bounds)
        bmt = np.zeros(self.n_bounds)
        if self.thermal:
            at = np.ones(self.n_bounds)
            bt = np.zeros(self.n_bounds)

        # right-hand side of boundary conditions
        for i, bound_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0],
                                           self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1])):
            c = np.array(self.discr_mesh.centroids[bound_id].values)
            #bc = self.boundary_conditions[self.discr_mesh.tags[bound_id]]
            x = np.append(c, 0.0)
            self.solution[self.n_vars * bound_id: self.n_vars * (bound_id + 1)] = self.ref1(x)

        # specify boundary conditions, loop over tags for speedup
        for tag in domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            ap[ids] = bc['flow']['a']
            bp[ids] = bc['flow']['b']
            amn[ids] = bc['mech']['an']
            bmn[ids] = bc['mech']['bn']
            amt[ids] = bc['mech']['at']
            bmt[ids] = bc['mech']['bt']
            if self.thermal:
                at[ids] = bc['heat']['a']
                bt[ids] = bc['heat']['b']

        self.cpp_bc = THMBoundaryCondition()
        self.cpp_bc.flow.a = value_vector(ap)
        self.cpp_bc.flow.b = value_vector(bp)

        if self.thermal:
            self.cpp_bc.thermal.a = value_vector(at)
            self.cpp_bc.thermal.b = value_vector(bt)

        self.cpp_bc.mech_normal.a = value_vector(amn)
        self.cpp_bc.mech_normal.b = value_vector(bmn)
        self.cpp_bc.mech_tangen.a = value_vector(amt)
        self.cpp_bc.mech_tangen.b = value_vector(bmt)

        self.cpp_flow = BoundaryCondition()
        self.cpp_flow.a = value_vector(ap)
        self.cpp_flow.b = value_vector(bp)
        if self.thermal:
            self.cpp_heat = BoundaryCondition()
            self.cpp_heat.a = value_vector(at)
            self.cpp_heat.b = value_vector(bt)

        # gradient reconstruction
        if self.thermal:
            self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_flow, self.cpp_heat)
        else:
            self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()

    # old discretizer
    def unit_cube_pm_discretizer(self):
        self.permx = self.permy = self.permz = 10.0 # any value
        physical_tags = {}
        physical_tags['matrix'] = [99991]
        physical_tags['fracture'] = []
        physical_tags['fracture_shape'] = []
        physical_tags['boundary'] = [991, 992, 993, 994, 995, 996]
        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_path, physical_tags=physical_tags)
        self.unstr_discr.load_mesh(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0)

        #self.unstr_discr.init_matrix_stiffness({99991: {'E': self.E, 'nu': self.nu}})

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
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.pm.neumann_boundaries_grad_reconstruction = True
        self.pm.grav = matrix(list(self.gravity), 1, 3)
        self.pm.visc = 1  # 9.81e-2
        self.solution = np.zeros(self.n_vars * (self.unstr_discr.mat_cells_tot + self.unstr_discr.bound_faces_tot))
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
            self.solution[self.n_vars * cell_id : self.n_vars * (cell_id + 1)] = self.ref1(np.append(cell.centroid, 0.0))

        for bound_id in range(self.unstr_discr.bound_faces_tot):
            b_cell = self.unstr_discr.bound_face_info_dict[bound_id]
            mech = self.unstr_discr.boundary_conditions[b_cell.prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[b_cell.prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            cell_id = self.unstr_discr.mat_cells_tot + bound_id
            sol = self.ref1(np.append(b_cell.centroid, 0.0))
            self.solution[self.n_vars * cell_id:self.n_vars * (cell_id + 1)] = sol

    def get_normal_to_bound_face(self, b_id):
        cell = self.unstr_discr.bound_face_info_dict[b_id]
        cells = [self.unstr_discr.mat_cells_to_node[pt] for pt in cell.nodes_to_cell]
        cell_id = next(iter(set(cells[0]).intersection(*cells)))
        for face in self.unstr_discr.faces[cell_id].values():
            if face.cell_id1 == face.cell_id2 and face.face_id2 == b_id:
                t_face = cell.centroid - self.unstr_discr.mat_cell_info_dict[cell_id].centroid
                n = face.n
                if np.inner(t_face, n) < 0: n = -n
                return n

    # calculate gradients, old discretizer
    def get_gradients_pm_discretizer(self, cell_id: int):
        st, coef = self.pm.get_gradient(cell_id)
        stencil = np.array(st, copy=False)
        stencil_cols = np.concatenate([
            np.arange(i * self.n_vars, i * self.n_vars + self.n_vars) for i in stencil])
        trans = np.array(coef).reshape(3 * self.n_vars, stencil.size * self.n_vars)
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
        u_trans = np.array(u_grad.a.values).reshape(3 * 3, len(u_grad.stencil) * self.n_vars)
        assert((np.sum(p_trans, axis=1) < 1.e-8).all())
        assert((np.sum(u_trans, axis=1) < 1.e-8).all())
        nabla_u = u_trans.dot(self.solution[stencil_cols])
        nabla_p = p_trans.dot(self.solution[self.n_vars * np.array(p_grad.stencil) + 3])

        grad = np.append(nabla_u, nabla_p)
        if self.thermal:
            t_grad = self.discr.t_grads[cell_id]
            t_trans = np.array(t_grad.a.values).reshape(3, len(t_grad.stencil))
            assert ((np.sum(t_trans, axis=1) < 1.e-8).all())
            nabla_t = t_trans.dot(self.solution[self.n_vars * np.array(t_grad.stencil) + 4])
            grad = np.append(grad, nabla_t)
        return grad
    # calculate analytical fluxes
    def get_analytical_fluxes(self, x, n, x_cell):
        sol_an = self.ref1(x)
        grad_an = self.nabla_ref1(x)
        stf = np.array(self.stf).reshape(6, 6)
        biot = np.array(self.biot).reshape(3, 3)
        perm = TC.darcy_constant * np.array(self.perm).reshape(3, 3)
        hooke_stress = self.W.dot(stf.dot(self.W.T)).\
            dot(grad_an[:self.n_dim, :self.n_dim].flatten()).\
            reshape(self.n_dim, self.n_dim)
        hooke_traction = -hooke_stress.dot(n)
        biot_traction = sol_an[3] * biot.dot(n) # sol_an[3] - pressure at the interface
        darcy = -perm.dot(n).dot(grad_an[self.n_dim, :self.n_dim] - self.fluid_density * self.gravity)
        vol_strain = (sol_an[:self.n_dim]).dot(biot.dot(n))# - self.ref1(x_cell)[:self.n_dim]).dot(biot.dot(n))
        result = [hooke_traction, biot_traction, darcy, vol_strain]
        if self.thermal:
            therm_expn = np.array(self.therm_expn).reshape(3, 3)
            conduction = np.array(self.conduction).reshape(3, 3)
            thermal_traction = sol_an[4] * therm_expn.dot(n) # sol_an[4] - temperature at the interface
            fourier = -conduction.dot(n).dot(grad_an[self.n_dim + 1, :self.n_dim])
            result += [thermal_traction, fourier]
        return result
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
        main_rhs = self.rhs[n_block * flux_id:n_block * (flux_id + 1)]
        biot_terms = self.tran_biot[n_block * n_block * self.offset[flux_id]:
                                 n_block * n_block * self.offset[flux_id + 1]].\
            reshape((stencil.size, n_block, n_block))
        biot_terms = np.transpose(biot_terms, (1, 0, 2)).reshape(n_block, n_block * stencil.size)
        biot_rhs = self.rhs_biot[n_block * flux_id:n_block * (flux_id + 1)]

        # Hooke's
        hooke_coefs = main_terms[:self.n_dim, :]
        hooke = hooke_coefs.dot(self.solution[stencil_cols]) + self.fluid_density * main_rhs[:self.n_dim]
        # Biot's
        biot_coefs = biot_terms[:self.n_dim, :]
        biot = biot_coefs.dot(self.solution[stencil_cols]) + self.fluid_density * biot_rhs[:self.n_dim]
        # Darcy's
        darcy_coefs = main_terms[self.n_dim, :]
        darcy = darcy_coefs.dot(self.solution[stencil_cols]) + self.fluid_density * main_rhs[self.n_dim]
        # Vols strain
        vol_strain_coefs = biot_terms[self.n_dim, :]
        vol_strain = vol_strain_coefs.dot(self.solution[stencil_cols]) + self.fluid_density * biot_rhs[self.n_dim]

        return hooke, biot, darcy, vol_strain
    # calculate fluxes, new discretizer
    def get_fluxes_new_discretizer(self, flux_id):
        n_block = self.n_vars # for poroelastic mode in discretizer
        n_hooke = n_block * self.n_dim
        n_biot = self.n_dim
        stencil = self.stencil[self.offset[flux_id]:
                               self.offset[flux_id + 1]]
        stencil_cols = np.concatenate([
            np.arange(i * self.n_vars, i * self.n_vars + self.n_vars) for i in stencil])
        # Hooke's
        hooke_coefs = self.hooke_trans[n_hooke * self.offset[flux_id]:
                               n_hooke * self.offset[flux_id + 1]].reshape((stencil.size, self.n_dim, n_block))
        hooke_coefs = np.transpose(hooke_coefs, (1, 0, 2)).reshape(self.n_dim, n_block * stencil.size)
        hooke_rhs = self.hooke_rhs[self.n_dim * flux_id:self.n_dim * (flux_id + 1)]
        hooke = hooke_coefs.dot(self.solution[stencil_cols]) + self.fluid_density * hooke_rhs
        # Biot's
        biot_coefs = self.biot_traction_trans[n_biot * self.offset[flux_id]:
                               n_biot * self.offset[flux_id + 1]].reshape((stencil.size, self.n_dim, 1))
        biot_coefs = np.transpose(biot_coefs, (1, 0, 2)).reshape(self.n_dim, stencil.size)
        biot_rhs = self.biot_traction_rhs[self.n_dim * flux_id:self.n_dim * (flux_id + 1)]
        biot = biot_coefs.dot(self.solution[stencil * self.n_vars + 3]) + self.fluid_density * biot_rhs
        # Darcy's
        darcy_coefs = self.darcy_trans[self.offset[flux_id]:self.offset[flux_id + 1]].reshape((stencil.size, 1, 1))
        darcy_coefs = np.transpose(darcy_coefs, (1, 0, 2)).reshape(1, stencil.size)
        darcy_rhs = self.darcy_rhs[flux_id]
        darcy = darcy_coefs.dot(self.solution[stencil * self.n_vars + 3])[0] + self.fluid_density * darcy_rhs
        # Volumetric strain
        vol_strain_coefs = self.biot_vol_strain_trans[n_block * self.offset[flux_id]:
                                n_block * self.offset[flux_id + 1]].reshape((stencil.size, 1, n_block))
        vol_strain_coefs = np.transpose(vol_strain_coefs, (1, 0, 2)).reshape(1, n_block * stencil.size)
        vol_strain_rhs = self.biot_vol_strain_rhs[flux_id]
        vol_strain = vol_strain_coefs.dot(self.solution[stencil_cols])[0] + self.fluid_density * vol_strain_rhs

        result = [hooke, biot, darcy, vol_strain]
        if self.thermal:
            # thermal_traction
            thermal_coefs = self.thermal_traction_trans[n_biot * self.offset[flux_id]:
                            n_biot * self.offset[flux_id + 1]].reshape((stencil.size, self.n_dim, 1))
            thermal_coefs = np.transpose(thermal_coefs, (1, 0, 2)).reshape(self.n_dim, stencil.size)
            thermal_traction = thermal_coefs.dot(self.solution[stencil * self.n_vars + 4])
            # Fourier
            fourier_coefs = self.fourier_trans[self.offset[flux_id]:self.offset[flux_id + 1]].reshape((stencil.size, 1, 1))
            fourier_coefs = np.transpose(fourier_coefs, (1, 0, 2)).reshape(1, stencil.size)
            fourier = fourier_coefs.dot(self.solution[stencil * self.n_vars + 4])[0]
            result += [thermal_traction, fourier]
        return result

# reference solution
# a linear function of (x,y,z) and time
    def nabla_ref1(self, x):
        if not self.thermal:
            A = np.array([[1, 2, 3, 4],
                          [6, 7, 8, 9],
                          [11, 12, 13, 14],
                          [16, 17, 18, 19]])
        else:
            A = np.array([[1,   2,  3,  4],   # ux
                          [6,   7,  8,  9],   # uy
                          [11, 12, 13, 14],   # uz
                          [16, 17, 18, 19],   # p
                          [21, 22, 23, 24]])  # temperature
        return A

    def nabla_ref1_b(self):
        if not self.thermal:
            return np.array([5, 10, 15, 20])
        else:
            return np.array([5, 10, 15, 20, 25])
    def ref1(self, x):
        '''
        :param x:
        :return: A*x +b
        '''
        A = self.nabla_ref1(x)
        b = self.nabla_ref1_b()
        if len(x.shape) == 1:
            return A.dot(x) + b
        else:
            return A.dot(x) + b[:,np.newaxis]





