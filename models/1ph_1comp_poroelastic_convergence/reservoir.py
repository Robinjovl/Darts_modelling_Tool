from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, contact, contact_vector, vector_matrix
from darts.engines import matrix33, matrix, pm_discretizer, Face, vector_face_vector, face_vector, vector_matrix33, Stiffness, stf_vector, critical_stress
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
from t2 import Rhs
from scipy.linalg import null_space
from darts.reservoirs.mesh.transcalc import TransCalculations as TC

import darts.discretizer as dis
from darts.discretizer import Mesh, Elem, poro_mech_discretizer, THMBoundaryCondition, BoundaryCondition, elem_loc, elem_type, conn_type
from darts.discretizer import vector_matrix33, vector_vector3, matrix, value_vector, index_vector
from darts.discretizer import matrix33 as disc_matrix33
from darts.engines import Stiffness as engine_stiffness
from darts.discretizer import Stiffness as disc_stiffness

# Definitions for the unstructured reservoir class:
class UnstructReservoir:
    def __init__(self, timer, discretizer, mesh_file):
        self.timer = timer
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        self.discretizer_name = discretizer
        self.mesh_file = mesh_file
        self.n_vars = 4
        self.n_dim = 3

        # Specify elastic properties, mesh & boundaries
        self.timer.node["discretization"] = timer_node()
        if self.discretizer_name == 'pm_discretizer':
            self.u_var = 0
            self.p_var = self.n_dim
            self.convergence_study_setup_pm_discretizer()
        elif self.discretizer_name == 'mech_discretizer':
            self.p_var = 0
            self.u_var = 1
            self.convergence_study_setup_mech_discretizer()

        self.x_new = np.ones((self.n_matrix + self.n_fracs, self.n_vars))
        self.x_new[:, self.u_var] = self.u_init[0]
        self.x_new[:, self.u_var + 1] = self.u_init[1]
        self.x_new[:, self.u_var + 2] = self.u_init[2]
        self.x_new[:, self.p_var] = self.p_init

        dt = 0.0
        if self.discretizer_name == 'pm_discretizer':
            self.timer.node["discretization"].start()

            self.pm.x_prev = value_vector(np.concatenate((self.x_new.flatten(), self.bc_rhs_prev)))
            self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot, index_vector(self.ref_contact_cells))
            self.pm.reconstruct_gradients_per_cell(dt)
            self.pm.calc_all_fluxes_once(dt)

            self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
                              self.pm.tran_biot, self.pm.rhs_biot,
                              self.unstr_discr.mat_cells_tot, self.unstr_discr.bound_cells_tot, self.unstr_discr.frac_cells_tot)
            self.timer.node["discretization"].stop()

            self.unstr_discr.store_volume_all_cells()
        elif self.discretizer_name == 'mech_discretizer':
            self.mesh.init_pm_mech_discretizer(self.discr.cell_m, self.discr.cell_p,
                                  self.discr.flux_stencil, self.discr.flux_offset,
                                  self.discr.hooke, self.discr.hooke_rhs,
                                  self.discr.biot_traction, self.discr.biot_traction_rhs,
                                  self.discr.darcy, self.discr.darcy_rhs,
                                  self.discr.biot_vol_strain, self.discr.biot_vol_strain_rhs,
                                  self.n_matrix, self.n_bounds, self.n_fracs)

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.bc = np.array(self.mesh.bc, copy=False)
        self.bc_prev = np.array(self.mesh.bc_prev, copy=False)
        self.bc_ref = np.array(self.mesh.bc_ref, copy=False)
        self.mesh.f.resize(self.n_vars * (self.n_matrix + self.n_fracs))
        self.f = np.array(self.mesh.f, copy=False)
        self.biot_arr = np.array(self.mesh.biot, copy=False)
        self.kd = np.array(self.mesh.kd, copy=False)
        self.mesh.pz_bounds.resize(self.n_bounds)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.p_ref = np.array(self.mesh.ref_pressure, copy=False)

        self.poro[:self.n_matrix] = self.porosity
        self.poro[self.n_matrix:] = 1
        if self.discretizer_name == 'pm_discretizer':
            self.volume[:self.n_matrix] = self.unstr_discr.volume_all_cells[:self.n_matrix]
            for i in range(self.unstr_discr.mat_cells_tot, self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
                self.volume[i] = self.unstr_discr.faces[i][4].area * self.frac_apers[i-self.unstr_discr.mat_cells_tot]
        elif self.discretizer_name == 'mech_discretizer':
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.volume[:self.n_matrix] = volumes[:self.n_matrix]

        self.bc_prev[:] = self.bc_rhs_prev
        self.bc[:] = self.bc_rhs
        self.bc_ref[:] = self.bc_rhs_ref
        # self.biot_arr[:] = np.tile([self.biot,0,0,
        #                             0,self.biot,0,
        #                             0,0,self.biot], self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
        self.biot_arr[:] = np.tile([0,0,0,
                                    0,0,0,
                                    0,0,0], self.n_matrix + self.n_fracs)
        self.kd[:] = 1 / self.c / self.porosity#self.lam + 2 * self.mu / 3
        # self.pz_bounds[:] = self.unstr_discr.pz_bounds
        self.p_ref[:] = self.p_init
        self.f[:] = self.f_prep

        self.wells = []
    def calc_deviations(self, engine):
        vol = np.array(self.mesh.volume, copy=False)
        total_vol = vol.sum()
        x_num = np.array(engine.X, copy=False)
        x_num = (x_num.reshape((self.n_matrix, self.n_vars))).T
        time = engine.t

        if self.discretizer_name == 'pm_discretizer':
            x = np.array([np.array([c.centroid[0], c.centroid[1], c.centroid[2], time]) for c in self.unstr_discr.mat_cell_info_dict.values()]).T
            x_an = reference_solution(x)
        elif self.discretizer_name == 'mech_discretizer':
            x_an = reference_solution(self.x_all[:, :self.n_matrix])

        dev_u = np.sqrt((vol * ((x_num[self.u_var:self.u_var+self.n_dim] - x_an[:self.n_dim]) ** 2).sum(axis=0)).sum() / total_vol)
        dev_p = np.sqrt((vol * ((x_num[self.p_var] - x_an[self.n_dim]) ** 2)).sum() / total_vol)

        return dev_u, dev_p
    def update_trans(self, dt, x):
        #self.pm.x_prev = value_vector(np.concatenate((x, self.bc_rhs_prev)))
        #self.pm.reconstruct_gradients_per_cell(dt)
        #self.pm.calc_all_fluxes(dt)
        #self.write_pm_conn_to_file(t_step=t_step)
        #self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
        #                  self.unstr_discr.mat_cells_tot, self.unstr_discr.bound_cells_tot, 0)

        # update transient sources / sinks
        self.f[:] = self.f_prep
        # update boundaries at n+1 / n timesteps
        self.bc[:] = self.bc_rhs
        self.bc_prev[:] = self.bc_rhs_prev
        #self.init_wells()
    def update_contact_condition(self, x, ith_iter):
        self.unstr_discr.ith_iter = ith_iter
        if ith_iter == 1:
            self.unstr_discr.x_prev = np.copy(x)
            self.unstr_discr.x_iter = np.copy(x)
        else:
            self.unstr_discr.x_iter = np.copy(self.unstr_discr.x_new)
        self.unstr_discr.x_new = np.copy(x)
        self.unstr_discr.f = np.zeros(self.unstr_discr.n_dim * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
        cell_m, cell_p, stress_stencil, stress_offset, stress_trans = self.unstr_discr.calc_mpsa_contact_connections_update()
        self.unstr_discr.write_mpsa_conn_to_file()
        self.mesh.init_mpsa(index_vector(cell_m), index_vector(cell_p),
                            index_vector(stress_stencil), index_vector(stress_offset), value_vector(stress_trans),
                            self.unstr_discr.n_dim,
                            self.unstr_discr.matrix_cell_count, self.unstr_discr.bound_cell_count,
                            self.unstr_discr.fracture_cell_count)
        self.f[:] = self.unstr_discr.f
    def update(self, dt, time):
        # update local array
        #if time > dt:
        self.bc_rhs_prev = np.copy(self.bc_rhs)

    def update_pm_discretizer(self, time):
        # Boundary conditions
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            c = self.unstr_discr.bound_cell_info_dict[bound_id].centroid
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            sol = reference_solution(np.append(c, time))
            u = sol[:3]
            p = sol[3]
            prop_id = self.unstr_discr.bound_cell_info_dict[bound_id].prop_id
            mech = self.unstr_discr.boundary_conditions[prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = n.dot(u) * n + P.dot(u)
            self.bc_rhs[4 * bound_id + 3] = p
        # RHS terms
        for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items():
            self.f_prep[4 * cell_id:4 * cell_id + 3] = -np.array(self.r.f.subs([(self.r.x, cell.centroid[0]),
                                                    (self.r.y, cell.centroid[1]),
                                                    (self.r.z, cell.centroid[2]),
                                                    (self.r.t, time)]).evalf())[:,0]
            self.f_prep[4 * cell_id + 3] = -(self.c * self.porosity *
                                                    self.r.acc.subs([(self.r.x, cell.centroid[0]),
                                                    (self.r.y, cell.centroid[1]),
                                                    (self.r.z, cell.centroid[2]),
                                                    (self.r.t, time)]).evalf() +
                                                    self.r.flow.subs([(self.r.x, cell.centroid[0]),
                                                    (self.r.y, cell.centroid[1]),
                                                    (self.r.z, cell.centroid[2]),
                                                    (self.r.t, time)]).evalf())
    def convergence_study_setup_pm_discretizer(self):
        self.porosity = 0.1
        self.unstr_discr = UnstructDiscretizer(permx=1, permy=1, permz=1, frac_aper=1.E-4,
                                               mesh_file=self.mesh_file)
        E = 10000 # in bars
        nu = 0.25
        lam = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)

        self.unstr_discr.physical_tags['matrix'] = [99991]
        #self.unstr_discr.init_matrix_stiffness({99991: {'E': E, 'nu': nu}})
        self.unstr_discr.physical_tags['fracture'] = [9991]
        self.unstr_discr.physical_tags['fracture_shape'] = []
        self.unstr_discr.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996]

        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}

        mech_xm = STUCK(0.0, [0.0, 0.0, 0.0])
        mech_xp = STUCK(0.0, [0.0, 0.0, 0.0])
        mech_ym = STUCK(0.0, [0.0, 0.0, 0.0])
        mech_yp = STUCK(0.0, [0.0, 0.0, 0.0])
        mech_zm = STUCK(0.0, [0.0, 0.0, 0.0])
        mech_zp = STUCK(0.0, [0.0, 0.0, 0.0])

        flow_xm = AQUIFER(0)
        flow_xp = AQUIFER(0)
        flow_ym = AQUIFER(0)
        flow_yp = AQUIFER(0)
        flow_zm = AQUIFER(0)
        flow_zp = AQUIFER(0)

        self.unstr_discr.boundary_conditions[991] = {'flow': flow_xm, 'mech': mech_xm, 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': flow_xp, 'mech': mech_xp, 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': flow_ym, 'mech': mech_ym, 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': flow_yp, 'mech': mech_yp, 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': flow_zm, 'mech': mech_zm, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': flow_zp, 'mech': mech_zp, 'cells': []}
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        perm = [1.5,    0.5,    0.35,
                0.5,    1.5,    0.45,
                0.35,   0.45,   1.5]
        biot = [1.5,    0.1,    0.5,
                0.1,    1.5,    0.15,
                0.5,    0.15,   1.5]
        stf =  [1.323, 0.0726, 0.263, 0.108, -0.08, -0.239,
                0.0726, 1.276, -0.318, 0.383, 0.108, 0.501,
                0.263, -0.318, 0.943, -0.183, 0.146, 0.182,
                0.108, 0.383, -0.183, 1.517, -0.0127, -0.304,
                -0.08, 0.108, 0.146, -0.0127, 1.209, -0.326,
                -0.239, 0.501, 0.182, -0.304, -0.326, 1.373]
        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.pm.visc = 1.0#9.81e-2
        self.grav = 9.81e-2
        self.rho_f = 978
        #self.rho_s = 2500
        self.fluid_compressibility = 0.0
        self.fluid_viscosity = 1e-2
        self.pm.grav = matrix([0.0, 0.0, self.grav], 1, 3)
        for cell_id in range(self.unstr_discr.mat_cells_tot):
            faces = self.unstr_discr.faces[cell_id]
            fs = face_vector()
            for face_id in range(len(faces)):
                face = faces[face_id]
                fs.append(Face(face.type.value, face.cell_id1, face.cell_id2,
                                        face.face_id1, face.face_id2,
                                        face.area, list(face.n), list(face.centroid)))
            self.pm.faces.append(fs)

            cell = self.unstr_discr.mat_cell_info_dict[cell_id]
            self.pm.cell_centers.append(matrix(list(cell.centroid), cell.centroid.size, 1))
            self.pm.perms.append(matrix33(perm))
            self.pm.biots.append(matrix33(biot))
            self.pm.stfs.append(Stiffness(stf))
        # Initial conditions
        self.p_init = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
        self.u_init = np.zeros((self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot, 3))
        time = 0.0
        for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items():
            sol = reference_solution(np.append(cell.centroid, time))
            self.u_init[cell_id] = sol[:3]
            self.p_init[cell_id] = sol[3]
        self.u_init = [self.u_init[:,0], self.u_init[:,1], self.u_init[:,2]]
        # Boundary conditions
        self.ref_contact_cells = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intc)
        self.bc_rhs = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs_ref = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs_prev = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            c = self.unstr_discr.bound_cell_info_dict[bound_id].centroid
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            sol = reference_solution(np.append(c, time))
            u = sol[:3]
            p = sol[3]
            prop_id = self.unstr_discr.bound_cell_info_dict[bound_id].prop_id
            mech = self.unstr_discr.boundary_conditions[prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = n.dot(u) * n + P.dot(u)
            self.bc_rhs[4 * bound_id + 3] = p
        self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc
        # RHS (force) term
        self.c = 1.4503768e-05
        self.r = Rhs(stf, biot, perm, self.fluid_viscosity, self.grav, self.rho_f)
        self.f_prep = np.zeros((self.unstr_discr.mat_cells_tot, 4))
        for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items():
            self.f_prep[cell_id, :3] = -np.array(self.r.f.subs([(self.r.x, cell.centroid[0]),
                                                    (self.r.y, cell.centroid[1]),
                                                    (self.r.z, cell.centroid[2]),
                                                    (self.r.t, time)]).evalf())[:,0]
            self.f_prep[cell_id, 3] = -(self.c * self.porosity *
                                                    self.r.acc.subs([(self.r.x, cell.centroid[0]),
                                                    (self.r.y, cell.centroid[1]),
                                                    (self.r.z, cell.centroid[2]),
                                                    (self.r.t, time)]).evalf() +
                                                    self.r.flow.subs([(self.r.x, cell.centroid[0]),
                                                    (self.r.y, cell.centroid[1]),
                                                    (self.r.z, cell.centroid[2]),
                                                    (self.r.t, time)]).evalf())
        self.f_prep = self.f_prep.flatten()

        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_bounds = self.unstr_discr.bound_cells_tot

    def update_mech_discretizer(self, time):
        # Boundary conditions
        self.x_all[3, :] = time
        sol = reference_solution(self.x_all[:, self.n_matrix + self.n_fracs:])
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]

            for id in ids:
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                conn_c = np.array(conn.c.values, copy=False)
                c1 = np.array(self.centroids[conn.elem_id1].values, copy=False)
                if n.dot(conn_c - c1) < 0: n *= -1.0
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = sol[:3, id]
                self.bc_rhs[self.n_vars * id + self.p_var] = sol[3, id]
        # RHS terms
        for cell_id in range(self.n_matrix):
            c = self.centroids[cell_id]
            self.f_prep[self.n_vars * cell_id + self.u_var:
                   self.n_vars * cell_id + self.u_var + self.n_dim] = -np.array(self.r.f.subs([
                    (self.r.x, c.values[0]),
                    (self.r.y, c.values[1]),
                    (self.r.z, c.values[2]),
                    (self.r.t, time)]).evalf())[:, 0]
            self.f_prep[self.n_vars * cell_id + self.p_var] = -(self.c * self.porosity *
               self.r.acc.subs([
                   (self.r.x, c.values[0]),
                   (self.r.y, c.values[1]),
                   (self.r.z, c.values[2]),
                   (self.r.t, time)]).evalf() +
               self.r.flow.subs([
                   (self.r.x, c.values[0]),
                   (self.r.y, c.values[1]),
                   (self.r.z, c.values[2]),
                   (self.r.t, time)]).evalf())
    def convergence_study_setup_mech_discretizer(self):
        self.mesh_data = meshio.read(self.mesh_file)
        self.domain_tags = dict()
        self.domain_tags[elem_loc.MATRIX] = set([99991])
        self.domain_tags[elem_loc.FRACTURE] = set([])  # 9991, 9992])
        self.domain_tags[elem_loc.BOUNDARY] = set([991, 992, 993, 994, 995, 996])
        self.domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()  # is this for poromechanics??

        # params
        self.porosity = 0.1
        perm = [1.5,    0.5,    0.35,
                0.5,    1.5,    0.45,
                0.35,   0.45,   1.5]
        biot = [1.5,    0.1,    0.5,
                0.1,    1.5,    0.15,
                0.5,    0.15,   1.5]
        stf =  [1.323, 0.0726, 0.263, 0.108, -0.08, -0.239,
                0.0726, 1.276, -0.318, 0.383, 0.108, 0.501,
                0.263, -0.318, 0.943, -0.183, 0.146, 0.182,
                0.108, 0.383, -0.183, 1.517, -0.0127, -0.304,
                -0.08, 0.108, 0.146, -0.0127, 1.209, -0.326,
                -0.239, 0.501, 0.182, -0.304, -0.326, 1.373]
        self.grav = 9.81e-2
        self.rho_f = 978.0
        self.fluid_compressibility = 0.0
        self.fluid_viscosity = 1e-2

        # prescribe boundary conditions
        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}

        self.boundary_conditions = {}
        self.boundary_conditions[991] = {'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.boundary_conditions[992] = {'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.boundary_conditions[993] = {'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.boundary_conditions[994] = {'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.boundary_conditions[995] = {'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.boundary_conditions[996] = {'flow': AQUIFER(0), 'mech': STUCK(0.0, [0.0, 0.0, 0.0]), 'cells': []}

        # initialize discretizer
        self.discr_mesh = Mesh()
        self.discr_mesh.gmsh_mesh_processing(self.mesh_file, self.domain_tags)
        self.discr = poro_mech_discretizer()
        self.discr.grav_vec = matrix([0.0, 0.0, self.grav], 1, 3)  # 0.0??
        self.tags = np.array(self.discr_mesh.tags, copy=False)
        self.discr.set_mesh(self.discr_mesh)
        self.discr.init()

        # number of elements
        self.n_matrix = self.discr_mesh.region_ranges[elem_loc.MATRIX][1] - \
                        self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
        self.n_fracs = self.discr_mesh.region_ranges[elem_loc.FRACTURE][1] - \
                        self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
        self.n_bounds = self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] - \
                        self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]

        # bulk properties
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            self.discr.perms.append(disc_matrix33(perm))
            self.discr.biots.append(disc_matrix33(biot))
            self.discr.stfs.append(disc_stiffness(stf))

        # initial conditions
        # mapping boundary connections
        adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)
        id_sorted = np.argsort(adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = adj_matrix[id_sorted]
        self.conns = np.array(self.discr_mesh.conns, copy=False)
        self.centroids = np.array(self.discr_mesh.centroids, copy=False)
        time = 0.0
        self.x_all = np.array([np.array([c.values[0], c.values[1], c.values[2], time]) for c in self.centroids]).T
        sol = reference_solution(self.x_all)
        self.u_init = sol[:self.n_dim, :self.n_matrix]
        self.p_init = sol[self.n_dim, :self.n_matrix]

        # assign boundary conditions element-by-element
        ap = np.ones(self.n_bounds)
        bp = np.zeros(self.n_bounds)
        amn = np.ones(self.n_bounds)
        bmn = np.zeros(self.n_bounds)
        amt = np.ones(self.n_bounds)
        bmt = np.zeros(self.n_bounds)
        self.bc_rhs = np.zeros(self.n_vars * self.n_bounds)
        self.bc_rhs_prev = np.zeros(self.n_vars * self.n_bounds)
        self.bc_rhs_ref = np.zeros(self.n_vars * self.n_bounds)

        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            ap[ids] = bc['flow']['a']
            bp[ids] = bc['flow']['b']
            amn[ids] = bc['mech']['an']
            bmn[ids] = bc['mech']['bn']
            amt[ids] = bc['mech']['at']
            bmt[ids] = bc['mech']['bt']

        self.cpp_bc = THMBoundaryCondition()
        self.cpp_bc.flow.a = value_vector(ap)
        self.cpp_bc.flow.b = value_vector(bp)
        self.cpp_bc.mech_normal.a = value_vector(amn)
        self.cpp_bc.mech_normal.b = value_vector(bmn)
        self.cpp_bc.mech_tangen.a = value_vector(amt)
        self.cpp_bc.mech_tangen.b = value_vector(bmt)
        self.cpp_flow = BoundaryCondition()
        self.cpp_flow.a = value_vector(ap)
        self.cpp_flow.b = value_vector(bp)

        # perform discretization
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.timer.node["discretization"].stop()

        # RHS term
        self.c = 1.4503768e-05
        self.r = Rhs(stf, biot, perm, self.fluid_viscosity, self.grav, self.rho_f)
        self.f_prep = np.zeros(self.n_matrix * self.n_vars)
        for cell_id in range(self.n_matrix):
            c = self.centroids[cell_id]
            self.f_prep[self.n_vars * cell_id + self.u_var:
                   self.n_vars * cell_id + self.u_var + self.n_dim] = -np.array(self.r.f.subs([
                                                    (self.r.x, c.values[0]),
                                                    (self.r.y, c.values[1]),
                                                    (self.r.z, c.values[2]),
                                                    (self.r.t, time)]).evalf())[:,0]
            self.f_prep[self.n_vars * cell_id + self.p_var] = -(self.c * self.porosity *
                                                    self.r.acc.subs([
                                                    (self.r.x, c.values[0]),
                                                    (self.r.y, c.values[1]),
                                                    (self.r.z, c.values[2]),
                                                    (self.r.t, time)]).evalf() +
                                                    self.r.flow.subs([
                                                    (self.r.x, c.values[0]),
                                                    (self.r.y, c.values[1]),
                                                    (self.r.z, c.values[2]),
                                                    (self.r.t, time)]).evalf())

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
        self.mesh.add_wells_mpfa(ms_well_vector(self.wells), self.P_VAR)
        if self.discretizer_name == 'mech_discretizer':
            self.mesh.reverse_and_sort_pm_mech_discretizer()
        elif self.discretizer_name == 'pm_discretizer':
            self.mesh.reverse_and_sort_pm()
        return 0

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
    def get_parametrized_fault_props(self):
        ref_id = next(iter(self.unstr_discr.frac_cell_info_dict))
        tags = np.array([cell.prop_id for cell in self.unstr_discr.frac_cell_info_dict.values()])
        tag_ids = {}
        t0 = {}
        coords = {}
        z_coords = {}
        for tag in self.unstr_discr.physical_tags['fracture']:
            ids = np.argwhere(tags == tag)[:,0]
            tag_ids[tag] = ref_id + ids
            n0 = self.unstr_discr.faces[ref_id + ids[0]][4].n[:2]
            t0[tag] = np.identity(2) - np.outer(n0, n0)
            coords[tag] = np.array([self.unstr_discr.frac_cell_info_dict[i].centroid for i in tag_ids[tag]])
            z_coords[tag] = np.unique(coords[tag][:,2])
        def dist_sort_key(id):
            c = self.unstr_discr.frac_cell_info_dict[id].centroid
            return c[0] ** 2 + c[1] ** 2
        def eval_frac_proj(tag, coords):
            return np.linalg.norm(t0[tag].dot(coords), axis=0)

        output_layers = 1
        output_var_num = {tag: int(output_layers * inds.size / z_coords[tag].size) for tag, inds in tag_ids.items()}
        faults_num = len(self.unstr_discr.physical_tags['fracture'])

        s = {tag: np.zeros(num) for tag, num in output_var_num.items()}
        #gap = np.zeros( (output_var_num, 3) )
        #Ftan = np.zeros( (output_var_num, 3) )
        #Fnorm = np.zeros( output_var_num )
        inds = {tag: np.zeros((output_layers, num), dtype=np.int64) for tag, num in output_var_num.items()}
        s_ref_prev = 0
        for tag, ids in tag_ids.items():
            counter = 0
            for l, z in enumerate(z_coords[tag][:output_layers]):
                z_inds = list(ids[np.argwhere( np.logical_and(coords[tag][:,2] > z-1.E-5, coords[tag][:,2] < z+1.E-5) )[:,0]])
                z_inds.sort(key=dist_sort_key)
                pts = self.unstr_discr.frac_cell_info_dict[z_inds[0]].coord_nodes_to_cell
                s_ref = np.min(eval_frac_proj(tag, pts[:,:2].T))
                inds[tag][l] = np.array(z_inds) - ref_id
                for id in z_inds:
                    c = self.unstr_discr.frac_cell_info_dict[id].centroid[:2]
                    s[tag][counter] = eval_frac_proj(tag, c) - s_ref + s_ref_prev
                    #gap[counter] = g[id - ref_id]
                    #Ftan[counter] = Ft[id - ref_id]
                    #Fnorm[counter] = Fn[id - ref_id]
                    counter += 1
                pts = self.unstr_discr.frac_cell_info_dict[z_inds[-1]].coord_nodes_to_cell
                s_ref_prev += np.max(eval_frac_proj(tag, pts[:,:2].T)) - s_ref
        z_output = {tag: z[:output_layers] for tag, z in z_coords.items()}
        return s, z_output, inds#gap, Ftan, Fnorm
    def write_data_field(self, filename, u, s = None):
        r = np.array([cell.centroid for cell in self.unstr_discr.mat_cell_info_dict.values()])
        inds = list(np.arange(len(r)))
        inds.sort(key=lambda id: r[id][0] + 1000 * r[id][1] + 1.E+6 * r[id][2])
        if s is self.write_data_field.__defaults__[0]:
            np.savetxt(filename, np.c_[r[inds,0], r[inds,1], r[inds, 2], u[inds,0], u[inds,1], u[inds,2]])
        else:
            np.savetxt(filename, np.c_[r[inds, 0], r[inds, 1], r[inds, 2], u[inds, 0], u[inds, 1], u[inds,2],
                s[inds, 0], s[inds, 1], s[inds, 2], s[inds, 3], s[inds, 4], s[inds, 5]])
    def check_positive_negative_sides(self):
        block_size = 4
        cell_m = np.array(self.pm.cell_m,dtype=np.intp)
        cell_p = np.array(self.pm.cell_p,dtype=np.intp)
        for i, cell_id1 in enumerate(cell_m):
            cell_id2 = cell_p[i]
            if cell_id2 < self.unstr_discr.mat_cells_tot:
                # find other one
                st1 = np.array(self.pm.stencil[self.pm.offset[i]:self.pm.offset[i + 1]], dtype=np.intp)
                all_trans1 = np.array(self.pm.tran)[(self.pm.offset[i] * block_size) * block_size: (self.pm.offset[i + 1] * block_size) * block_size].reshape(self.pm.offset[i + 1] - self.pm.offset[i], block_size, block_size)
                j = np.where(np.logical_and(cell_m == cell_id2, cell_p == cell_id1))[0][0]
                st2 = np.array(self.pm.stencil[self.pm.offset[j]:self.pm.offset[j + 1]], dtype=np.intp)
                all_trans2 = np.array(self.pm.tran)[(self.pm.offset[j] * block_size) * block_size: (self.pm.offset[j + 1] * block_size) * block_size].reshape(self.pm.offset[j + 1] - self.pm.offset[j], block_size, block_size)
                assert(set(st1) == set(st2))
                ids1 = np.argsort(st1)
                ids2 = np.argsort(st2)
                diff = all_trans1[ids1] + all_trans2[ids2]
                assert((np.abs(diff) < 1.E-10).all())
    def write_pm_conn_to_file(self, t_step, path='pm_conn.dat'):
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
    def write_to_vtk(self, output_directory, ith_step, engine):
        """
        Class method which writes output of unstructured grid to VTK format
        :param output_directory: directory of output files
        :param property_array: np.array containing all cell properties (N_cells x N_prop)
        :param cell_property: list with property names (visible in ParaView (format strings)
        :param ith_step: integer containing the output step
        :param engine: engine that manages required data
        :return:
        """
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
        # fluxes_n = np.array(engine.fluxes_n, copy=False)
        fluxes_biot = np.array(engine.fluxes_biot, copy=False)
        #vels = self.reconstruct_velocities(fluxes[engine.P_VAR::engine.N_VARS],
        #                                  fluxes_biot[engine.P_VAR::engine.N_VARS])
        # self.mech_operators.eval_porosities(engine.X, self.mesh.bc)
        # self.mech_operators.eval_stresses(engine.fluxes, engine.fluxes_biot, engine.X,
        #                                   self.mesh.bc, engine.op_vals_arr)

        # Matrix
        geom_id = 0
        Mesh.cells = []
        cell_data = {}
        for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
            if ith_geometry in available_matrix_geometries:
                Mesh.cells.append(self.unstr_discr.mesh_data.cells[geom_id])
                x_an = np.array([np.append(cell.centroid, engine.t) for cell in self.unstr_discr.mat_cell_info_dict.values()])
                sol_an = reference_solution(x_an.T)
                # Add matrix data to dictionary:
                for i in range(props_num):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(property_array[i:props_num * self.unstr_discr.mat_cells_tot:props_num])
                    if cell_property[i] + '_an' not in cell_data: cell_data[cell_property[i] + '_an'] = []
                    cell_data[cell_property[i] + '_an'].append(sol_an[i,:])

                #if 'velocity' not in cell_data: cell_data['velocity'] = []
                #cell_data['velocity'].append(vels)
                # if hasattr(self.unstr_discr, 'E') and hasattr(self.unstr_discr, 'nu'):
                #     cell_data[ith_geometry]['E'] = np.zeros(self.unstr_discr.mat_cells_tot, dtype=np.float64)
                #     cell_data[ith_geometry]['nu'] = np.zeros(self.unstr_discr.mat_cells_tot, dtype=np.float64)
                #     for id, cell in enumerate(self.unstr_discr.mat_cell_info_dict.values()):
                #         cell_data[ith_geometry]['E'][id] = self.unstr_discr.E[cell.prop_id]
                #         cell_data[ith_geometry]['nu'][id] = self.unstr_discr.nu[cell.prop_id]
                # if 'eps_vol' not in cell_data: cell_data['eps_vol'] = []
                # if 'porosity' not in cell_data: cell_data['porosity'] = []
                # if 'stress' not in cell_data: cell_data['stress'] = []
                # if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []
                #
                # cell_data['eps_vol'].append(np.array(self.mech_operators.eps_vol, copy=False))
                # cell_data['porosity'].append(np.array(self.mech_operators.porosities, copy=False))
                # cell_data['stress'].append(np.zeros((self.unstr_discr.mat_cells_tot, 6), dtype=np.float64))
                # cell_data['tot_stress'].append(np.zeros((self.unstr_discr.mat_cells_tot, 6), dtype=np.float64))

                # stress = np.array(self.mech_operators.stresses, copy=False)
                # total_stress = np.array(self.mech_operators.total_stresses, copy=False)
                # for i in range(6):
                #     cell_data['stress'][-1][:, i] = stress[i::6]
                #     cell_data['tot_stress'][-1][:, i] = total_stress[i::6]

                # if 'cell_id' not in cell_data: cell_data['cell_id'] = []
                # cell_data['cell_id'].append(np.array([cell_id for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items() if cell.geometry_type == ith_geometry], dtype=np.int64))
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

    def get_fault_props(self, property_array, ith_step, physics):
        n_vars = 4
        n_dim = 3
        fluxes = np.array(physics.engine.fluxes, copy=False)
        fluxes_biot = np.array(physics.engine.fluxes_biot, copy=False)
        S_eng = vector_matrix(physics.engine.contacts[0].S)
        frac_prop = property_array[n_vars * self.unstr_discr.mat_cells_tot:n_vars * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)].reshape(self.unstr_discr.frac_cells_tot, n_vars)
        fstress = np.array(physics.engine.contacts[0].fault_stress, copy=False)

        frac_data = {}
        frac_data['tag'] = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intp)
        frac_data['g_local'] = np.zeros((self.unstr_discr.frac_cells_tot, n_dim))
        frac_data['f_local'] = np.zeros((self.unstr_discr.frac_cells_tot, n_dim))
        frac_data['mu'] = np.array(physics.engine.contacts[0].mu, copy=False)

        for cell_id, cell in self.unstr_discr.frac_cell_info_dict.items():
            cell_id -= self.unstr_discr.mat_cells_tot
            frac_data['tag'][cell_id] = int(cell.prop_id)
            face = self.unstr_discr.faces[cell_id + self.unstr_discr.mat_cells_tot][4]
            S = np.array(S_eng[cell_id].values).reshape((n_dim, n_dim))
            f = fstress[n_dim * cell_id:n_dim * (cell_id + 1)] / face.area
            frac_data['f_local'][cell_id] = S.dot(f)
            frac_data['g_local'][cell_id] = S.dot(frac_prop[cell_id,:n_dim])

        phi = np.array(physics.engine.contacts[0].phi, copy=False)
        #states = phi > 0
        frac_data['phi'] = phi

        #if ith_step == 0:
        #    self.time_file.write(str(0.0) + '\t' + str(self.un_top) + '\n')
        #else:
        #    self.time_file.write(str(physics.engine.t * 86400.0) + '\t' + str(self.un_top) + '\n')
        #self.time_file.flush()

        return frac_data
    def write_fault_props(self, output_directory, property_array, ith_step, physics):
        n_vars = 4
        n_dim = 3
        fluxes = np.array(physics.engine.fluxes, copy=False)
        fluxes_biot = np.array(physics.engine.fluxes_biot, copy=False)
        s, z_coords, inds = self.get_parametrized_fault_props()
        x = property_array.reshape(int(property_array.size / n_vars), n_vars)[self.unstr_discr.mat_cells_tot:self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot]
        g = {}
        glocal = {}
        flocal = {}
        mu = {}
        for tag, ids in inds.items():
            if ids.size * z_coords[tag].size < self.unstr_discr.frac_cells_tot: return 0

            g[tag] = np.array(x[ids[0],:n_dim])
            glocal[tag] = np.zeros((len(ids[0]), 3))
            flocal[tag] = np.zeros((len(ids[0]), 3))
            S_eng = vector_matrix(physics.engine.contacts[0].S)
            mu[tag] = np.array(physics.engine.contacts[0].mu, copy=False)
            fstress = np.array(physics.engine.contacts[0].fault_stress, copy=False)
            for i, id in enumerate(ids[0]):
                face = self.unstr_discr.faces[id + self.unstr_discr.mat_cells_tot][4]
                S = np.array(S_eng[id].values).reshape((n_dim, n_dim))
                flocal[tag][i] = S.dot(fstress[n_dim * id:n_dim * (id + 1)] / face.area)
                #n = self.unstr_discr.faces[self.unstr_discr.mat_cells_tot][max(self.unstr_discr.faces[self.unstr_discr.mat_cells_tot].keys())].n
                #S = np.zeros((n_dim, n_dim))
                #S[:n_dim - 1] = null_space(np.array([-n])).T
                #S[n_dim - 1] = -n
                glocal[tag][i] = S.dot(g[tag][i])

            #if ith_step == 0:
            self.fig, self.ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 10))
            self.ax0 = self.ax[0].twinx()
            self.ax1 = self.ax[1].twinx()
            self.ax11 = self.ax[1].twinx()
            #self.ax[0].set_ylabel('normal gap, $g_N$')
            self.ax[0].set_ylabel('friction coefficient, $\mu$')
            self.ax0.set_ylabel('slip, $g_T$')
            self.ax[1].set_ylabel('normal traction, $F_N$')
            self.ax1.set_ylabel('tangential traction, $F_T$')
            self.ax[1].set_xlabel('distance')
                #self.ax1.set_ylabel('distance along fault')

            phi = np.array(physics.engine.contacts[0].phi, copy=False)
            states = phi > 0
            for tag, s_cur in s.items():
                Fn = flocal[tag][:,0]
                Ft = flocal[tag][:,1]
                #self.ax[0].plot(s_cur, glocal[tag][:,0], color='b', linestyle='-', marker='o', label=str(tag) + r': $g_N$')
                if (mu[tag] != 0.0).all() and (Fn != 0.0).all():
                    self.ax[0].plot(s_cur, mu[tag], color='b', linestyle='-', marker='o', label=str(tag) + r': $\mu$')
                    self.ax[0].plot(s_cur, Ft / Fn, color='r', linestyle='--', marker='o', label=str(tag) + r': $\mu * SCU$')
                self.ax0.plot(s_cur, -glocal[tag][:,1], color='r', linestyle='-', marker='o', label=str(tag) + r': $g_T$')
                self.ax[1].plot(s_cur, Fn, color='b', linestyle='-', marker='o', label=str(tag) + r': $F_N$')
                self.ax1.plot(s_cur, Ft, color='r', linestyle='-', marker='o', label=str(tag) + r': $F_T$')
                self.ax11.plot(s_cur, states[ids[0]], color='g', linestyle=':', marker='x')
                if states[ids[0]][0] == 0:
                    self.ax11.text(0, 0, 'STUCK', fontsize=15)
                elif states[ids[0]][0] == 1:
                    self.ax11.text(0, 1, 'SLIP', fontsize=15)

                np.savetxt(output_directory + '/fault_step_' + str(ith_step) + '_tag_' + str(tag) + ".txt",
                           np.c_[s_cur, glocal[tag][:, 0], glocal[tag][:, 1], glocal[tag][:, 2], Fn, Ft, mu[tag]])

            self.ax[0].grid(axis='x')
            self.ax[1].grid(axis='x')
            self.ax0.grid(axis='y')
            self.ax1.grid(axis='y')
            self.ax[0].legend(loc='upper left')
            self.ax0.legend(loc='upper right')
            self.ax[1].legend(loc='upper left')
            self.ax1.legend(loc='upper right')
            if mu[tag].min() != mu[tag].max():
                self.ax[0].set_ylim([0.98 * mu[tag].min(), 1.02 * mu[tag].max()])
            else:
                self.ax[0].set_ylim([0.0, 1.0])
            #self.ax0.set_ylim([-0.0002, 0.0006])
            #self.ax[1].set_ylim([19, 21])
            #self.ax1.set_ylim([-0.1, 0.1])
            self.ax11.get_yaxis().set_visible(False)

            self.fig.tight_layout()
            self.fig.savefig(output_directory + '/fig_' + str(ith_step) + '.png')
            plt.close(self.fig)

    def write_fault_props_old(self, output_directory, property_array, ith_step, fluxes, physics):
        n_vars = 4
        n_dim = 3
        s, z_coords, inds = self.get_parametrized_fault_props()
        x = property_array.reshape(int(property_array.size / n_vars), n_vars)[self.unstr_discr.mat_cells_tot:self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot]
        g = {}
        glocal = {}
        f = [{}, {}]
        flocal = [{}, {}]
        for tag, ids in inds.items():
            g[tag] = np.array(x[ids[0],:n_dim])
            glocal[tag] = np.zeros((len(ids[0]), 3))
            f[0][tag] = np.zeros((len(ids[0]), 3))
            f[1][tag] = np.zeros((len(ids[0]), 3))
            flocal[0][tag] = np.zeros((len(ids[0]), 3))
            flocal[1][tag] = np.zeros((len(ids[0]), 3))
            S_eng = vector_matrix(physics.engine.contacts[0].S)
            for i, id in enumerate(ids[0]):
                conn_id = self.mesh.fault_conn_id[id][0]
                dx = self.unstr_discr.mat_cell_info_dict[self.mesh.block_p[conn_id]].centroid - \
                        self.unstr_discr.mat_cell_info_dict[self.mesh.block_m[conn_id]].centroid
                sign = 1.0 if self.unstr_discr.faces[self.unstr_discr.mat_cells_tot][max(self.unstr_discr.faces[self.unstr_discr.mat_cells_tot].keys())].n.dot(dx) > 0 else -1
                for k in range(0, 2):
                    conn_id = self.mesh.fault_conn_id[id][k]
                    face = self.unstr_discr.faces[id + self.unstr_discr.mat_cells_tot][4]
                    if k > 0: sign = -sign

                    #n = face.n#np.array(self.mesh.fault_normals[id])
                    #dx = self.unstr_discr.mat_cell_info_dict[self.mesh.block_p[conn_id]].centroid - \
                    #     self.unstr_discr.mat_cell_info_dict[self.mesh.block_m[conn_id]].centroid
                    #n = n if n.dot(dx) > 0 else -n
                    #S = np.zeros((n_dim, n_dim))
                    #S[1:] = null_space(np.array([n])).T
                    #S[0] = n

                    S = np.array(S_eng[id].values).reshape((n_dim, n_dim))

                    f[k][tag][i] = sign * fluxes[n_vars * conn_id:n_vars * conn_id + 3]# / face.area
                    flocal[k][tag][i] = S.dot(f[k][tag][i])
                    if k == 0:
                        #n = self.unstr_discr.faces[self.unstr_discr.mat_cells_tot][max(self.unstr_discr.faces[self.unstr_discr.mat_cells_tot].keys())].n
                        #S = np.zeros((n_dim, n_dim))
                        #S[:n_dim - 1] = null_space(np.array([-n])).T
                        #S[n_dim - 1] = -n
                        glocal[tag][i] = S.dot(g[tag][i])

            #if ith_step == 0:
                self.fig, self.ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 10))
                self.ax0 = self.ax[0].twinx()
                self.ax1 = self.ax[1].twinx()
                self.ax11 = self.ax[1].twinx()
                self.ax[0].set_ylabel('normal gap, $g_N$')
                self.ax0.set_ylabel('slip, $g_T$')
                self.ax[1].set_ylabel('normal traction, $F_N$')
                self.ax1.set_ylabel('tangential traction, $F_T$')
                self.ax[1].set_xlabel('distance')
                #self.ax1.set_ylabel('distance along fault')

            phi = np.array(physics.engine.contacts[0].phi, copy=False)
            states = phi > 0
            for tag, s_cur in s.items():
                self.ax[0].plot(s_cur, glocal[tag][:,0], color='b', linestyle='-', marker='o', label=str(tag) + r': $g_N$')
                self.ax0.plot(s_cur, -glocal[tag][:,1], color='r', linestyle='-', marker='o', label=str(tag) + r': $g_T$')
                self.ax[1].plot(s_cur, flocal[0][tag][:,0], color='b', linestyle='--', marker='x', label=str(tag) + r': $F_N^+$')
                self.ax1.plot(s_cur, flocal[0][tag][:,1], color='r', linestyle='--', marker='x', label=str(tag) + r': $F_T^+$')
                self.ax[1].plot(s_cur, flocal[1][tag][:,0], color='b', linestyle='--', marker='^', label=str(tag) + r': $F_N^-$')
                self.ax1.plot(s_cur, flocal[1][tag][:,1], color='r', linestyle='--', marker='^', label=str(tag) + r': $F_T^-$')
                self.ax[1].plot(s_cur, (flocal[0][tag][:,0] + flocal[1][tag][:,0]) / 2, color='b', linestyle='-', marker='o', label=str(tag) + r': $F_N$')
                self.ax1.plot(s_cur, (flocal[0][tag][:,1] + flocal[1][tag][:,1]) / 2, color='r', linestyle='-', marker='o', label=str(tag) + r': $F_T$')
                self.ax11.plot(s_cur, states[ids[0]], color='g', linestyle=':', marker='x')
                if states[ids[0]][0] == 0:
                    self.ax11.text(0, 0, 'STUCK', fontsize=15)
                elif states[ids[0]][0] == 1:
                    self.ax11.text(0, 1, 'SLIP', fontsize=15)

            self.ax[0].grid(axis='x')
            self.ax[1].grid(axis='x')
            self.ax0.grid(axis='y')
            self.ax1.grid(axis='y')
            self.ax[0].legend(loc='upper left')
            self.ax0.legend(loc='upper right')
            self.ax[1].legend(loc='upper left')
            self.ax1.legend(loc='upper right')
            self.ax[0].set_ylim([0, 0.001])
            #self.ax0.set_ylim([-0.0002, 0.0006])
            #self.ax[1].set_ylim([19, 21])
            #self.ax1.set_ylim([-0.1, 0.1])
            self.ax11.get_yaxis().set_visible(False)

            self.fig.tight_layout()
            self.fig.savefig(output_directory + '/fig_' + str(ith_step) + '.png')

            np.savetxt(output_directory + '/fault_step_' + str(ith_step) + '_tag_' + str(tag) + ".txt",
                       np.c_[s[tag], glocal[tag][:,0], glocal[tag][:,1], glocal[tag][:,2],
                    (flocal[0][tag][:,0] + flocal[1][tag][:,0]) / 2, (flocal[0][tag][:,1] + flocal[1][tag][:,1]) / 2])

    def write_diff_to_vtk(self, output_directory, property_array, cell_property, ith_step, time):
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
        cell_data = {}
        geom_id = 0
        Mesh.cells = []
        for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
            if ith_geometry == 'hexahedron' or ith_geometry == 'wedge' or ith_geometry == 'tetra':
                # Add matrix data to dictionary:
                Mesh.cells.append(self.unstr_discr.mesh_data.cells[geom_id])
                x_an = np.array([np.append(cell.centroid, time) for cell in self.unstr_discr.mat_cell_info_dict.values()])
                sol_an = reference_solution(x_an.T)
                for i in range(len(cell_property)):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(np.abs(property_array[i::4] - sol_an[i,:]))

                if hasattr(self.unstr_discr, 'E') and hasattr(self.unstr_discr, 'nu'):
                    cell_data[ith_geometry]['E'] = np.zeros(self.unstr_discr.mat_cells_tot, dtype=np.float64)
                    cell_data[ith_geometry]['nu'] = np.zeros(self.unstr_discr.mat_cells_tot, dtype=np.float64)
                    for id, cell in enumerate(self.unstr_discr.mat_cell_info_dict.values()):
                        cell_data[ith_geometry]['E'][id] = self.unstr_discr.E[cell.prop_id]
                        cell_data[ith_geometry]['nu'][id] = self.unstr_discr.nu[cell.prop_id]
            geom_id += 1
        #if self.unstr_discr.frac_cells_tot > 0 and np.fabs(np.sum(property_array)) > 0.0:
        #    self.write_fault_props(output_directory,property_array, ith_step, by_terms=True)
        # Store solution for each time-step:
        mesh = meshio.Mesh(
            Mesh.points,
            Mesh.cells,
            cell_data=cell_data)
        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)
        return 0

def reference_solution(x):
    if len(x.shape) == 1:
        sol = np.zeros(4)
    else:
        sol = np.zeros(x.shape)
    sol[:3] = (x[:3] - 0.5) ** 2
    sol[0] -= x[1] + x[2]
    sol[1] -= x[0] + x[2]
    sol[2] -= x[0] + x[1]
    sol[:3] *= (1 + x[3] ** 2)
    sol[3] = np.sin((1 - x[0]) * (1 - x[1]) * (1 - x[2])) / 2 / np.sin(1) + \
           ((1 - x[0]) ** 3) * ((1 - x[1]) ** 2) * (1 - x[2]) * (1 + x[3] ** 2) / 2
    return sol