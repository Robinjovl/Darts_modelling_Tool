from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, contact, contact_vector, vector_matrix, scheme_type
from darts.engines import matrix33 as engine_matrix33
from darts.discretizer import matrix33 as disc_matrix33
from darts.engines import Stiffness as engine_stiffness
from darts.discretizer import Stiffness as disc_stiffness
from darts.engines import matrix, pm_discretizer, Face, vector_face_vector, face_vector, vector_matrix33, stf_vector, critical_stress
import numpy as np
from math import inf, pi
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.unstruct_reservoir_mech import bound_cond, set_domain_tags, UnstructReservoirMech
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
class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, case='mandel', discretizer='mech_discretizer', mesh='rect'):
        thermoporoelacticity = True if case == 'bai' else False
        super().__init__(timer, discretizer, thermoporoelacticity)
        # define correspondence between the physical tags in msh file and mesh elements types
        self.domain_tags, self.bnd_tags = set_domain_tags(matrix_tags=[99991],
                    bnd_xm_tag=991, bnd_xp_tag=992,
                    bnd_ym_tag=993, bnd_yp_tag=994,
                    bnd_zm_tag=995, bnd_zp_tag=996)

        # Specify elastic properties, mesh & boundaries
        if case == 'mandel':
            if discretizer == 'mech_discretizer':
                self.mandel_north_dirichlet_mech_discretizer(mesh)
            elif discretizer == 'pm_discretizer':
                self.mandel_north_dirichlet_pm_discretizer(mesh)
        elif case == 'terzaghi':
            if discretizer == 'mech_discretizer':
                self.terzaghi_mech_discretizer(mesh)
            elif discretizer == 'pm_discretizer':
                self.terzaghi_pm_discretizer(mesh)
        elif case == 'terzaghi_two_layers':
            if discretizer == 'mech_discretizer':
                self.terzaghi_two_layers_mech_discretizer(mesh)
            elif discretizer == 'pm_discretizer':
                self.terzaghi_two_layers_pm_discretizer(mesh)
        elif case == 'terzaghi_two_layers_no_analytics':
            if discretizer == 'pm_discretizer':
                self.terzaghi_two_layers_no_analytics_pm_discretizer(mesh)
        elif case == 'bai':
            self.bai_thermoporoelastic_consolidation(mesh)

        if discretizer == 'mech_discretizer':
            if case == 'bai':
                self.mesh.init_pme_mech_discretizer(self.discr.cell_m, self.discr.cell_p,
                                  self.discr.flux_stencil, self.discr.flux_offset,
                                  self.discr.hooke, self.discr.hooke_rhs,
                                  self.discr.biot_traction, self.discr.biot_traction_rhs,
                                  self.discr.darcy, self.discr.darcy_rhs,
                                  self.discr.biot_vol_strain, self.discr.biot_vol_strain_rhs,
                                  self.discr.thermal_traction, self.discr.fourier,
                                  self.n_matrix, self.n_bounds, self.n_fracs)
            else:
                self.mesh.init_pm_mech_discretizer(self.discr.cell_m, self.discr.cell_p,
                                  self.discr.flux_stencil, self.discr.flux_offset,
                                  self.discr.hooke, self.discr.hooke_rhs,
                                  self.discr.biot_traction, self.discr.biot_traction_rhs,
                                  self.discr.darcy, self.discr.darcy_rhs,
                                  self.discr.biot_vol_strain, self.discr.biot_vol_strain_rhs,
                                  self.n_matrix, self.n_bounds, self.n_fracs)
        elif discretizer == 'pm_discretizer':
            if case == 'bai':
                print(case, 'not supported in', discretizer)
                assert False
            self.init_pm_discretizer()

        self.init_arrays()

        self.wells = []

    # Mandel
    def mandel_north_dirichlet_mech_discretizer(self, mesh='rect'):
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge.msh'
        elif mesh == 'hex':
            mesh_file = 'meshes/hexahedron.msh'
        self.mesh_filename = mesh_file

        self.mesh_data = meshio.read(mesh_file)

        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        self.porosity = 0.375
        self.permx = self.permy = self.permz = 10.0 / 9.81
        self.E = 10000 # in bars
        self.nu = 0.25
        self.lam = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.E / 2 / (1 + self.nu)
        self.biot = 0.9
        self.kd_cur = self.E / 3 / (1 - 2 * self.nu)
        self.fluid_compressibility = 1.e-5
        self.fluid_viscosity = 1.0
        self.M = 1.0 / ((self.biot - self.porosity) * (1 - self.biot) / self.kd_cur +
                            self.porosity * self.fluid_compressibility)

        self.set_mandel_boundary_conditions()

        self.init_mech_discretizer()

        self.F = -100.0 * self.a # bar * m

        self.porosity = self.porosity * np.ones(self.n_matrix + self.n_fracs)

        self.biot_mean = np.zeros(9 * (self.n_matrix + self.n_fracs))

        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            self.discr.perms.append(disc_matrix33(self.permx, self.permy, self.permz))
            self.discr.biots.append(disc_matrix33(self.biot))
            self.discr.stfs.append(disc_stiffness(self.lam, self.mu))
            self.biot_mean[9 * cell_id] = self.biot
            self.biot_mean[9 * cell_id + 4] = self.biot
            self.biot_mean[9 * cell_id + 8] = self.biot

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        ap = np.zeros(boundary_range[1] - boundary_range[0])
        bp = np.zeros(boundary_range[1] - boundary_range[0])
        an = np.zeros(boundary_range[1] - boundary_range[0])
        bn = np.zeros(boundary_range[1] - boundary_range[0])
        at = np.zeros(boundary_range[1] - boundary_range[0])
        bt = np.ones(boundary_range[1] - boundary_range[0])
        self.bc_rhs = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] -
                                                self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))
        self.bc_rhs_prev = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] -
                                                self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))

        # mapping boundary connections
        adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)
        id_sorted = np.argsort(adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = adj_matrix[id_sorted]
        self.conns = np.array(self.discr_mesh.conns, copy=False)
        self.centroids = np.array(self.discr_mesh.centroids, copy=False)
        u_var = 1

        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            ap[ids] = bc['flow']['a']
            bp[ids] = bc['flow']['b']
            an[ids] = bc['mech']['an']
            bn[ids] = bc['mech']['bn']
            at[ids] = bc['mech']['at']
            bt[ids] = bc['mech']['bt']
            # flow
            self.bc_rhs[self.n_vars * ids + self.p_var] = bc['flow']['r']
            self.bc_rhs_prev[self.n_vars * ids + self.p_var] = bc['flow']['r']

            for id in ids:
                assert(adj_matrix_cols[id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                conn_c = np.array(conn.c.values, copy=False)
                c1 = np.array(self.centroids[conn.elem_id1].values, copy=False)
                if n.dot(conn_c - c1) < 0: n *= -1.0
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']
                self.bc_rhs_prev[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']

        self.cpp_bc = THMBoundaryCondition()
        self.cpp_bc.flow.a = value_vector(ap)
        self.cpp_bc.flow.b = value_vector(bp)

        self.cpp_bc.mech_normal.a = value_vector(an)
        self.cpp_bc.mech_normal.b = value_vector(bn)
        self.cpp_bc.mech_tangen.a = value_vector(at)
        self.cpp_bc.mech_tangen.b = value_vector(bt)
        # to use base discretizer's class function reconstruct_pressure_gradients_per_cell
        # which doesn't know the new THMBoundaryCondition class yet
        self.cpp_flow = BoundaryCondition()
        self.cpp_flow.a = value_vector(ap)
        self.cpp_flow.b = value_vector(bp)

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

        MR = 0.9869 * 1.E-15 * self.permx / self.fluid_viscosity / 1.E-3
        K_dr = self.E / (3 * (1 - 2 * self.nu))
        self.K_nu = (K_dr + (4 / 3) * self.mu)
        Cv = 1.e+5 * MR * self.M * self.K_nu / (self.K_nu + self.biot ** 2 * self.M)
        self.tD = self.a ** 2 / Cv / 86400
        self.pD = abs(self.F / self.a) / 2

        # from compare_grad_discr import compare_gradients
        # compare_gradients('pm.pkl', new_cache_filename=None, orig_pm_arg=None, new_pm_arg=self.discr)
    def mandel_north_dirichlet_pm_discretizer(self, mesh='rect'):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        self.porosity = 0.375
        self.permx = self.permy = self.permz = 10.0 / 9.81
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge.msh'
        elif mesh == 'hex':
            mesh_file = 'meshes/hexahedron.msh'
        self.mesh_filename = mesh_file
        self.unstr_discr = UnstructDiscretizer(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0,
                                               mesh_file=mesh_file)
        self.unstr_discr.eps_t = 1.E+0
        self.unstr_discr.eps_n = 1.E+0
        self.unstr_discr.mu = 3.2
        self.unstr_discr.P12 = 0
        self.unstr_discr.Prol = 1
        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3
        self.unstr_discr.physical_tags['matrix'] = list(self.domain_tags[elem_loc.MATRIX])
        # lam = 1.0 * 10000  # in bar
        # mu = 1.0 * 10000
        # nu = lam / 2 / (lam + mu)
        # E = lam * (1 + nu) * (1 - 2 * nu) / nu

        self.E = 10000  # in bars
        self.nu = 0.25
        self.lam = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.E / 2 / (1 + self.nu)
        self.biot = 0.9
        self.kd_cur = self.E / 3 / (1 - 2 * self.nu)
        self.fluid_compressibility = 1.e-5
        self.fluid_viscosity = 1.0
        self.M = 1.0 / ((self.biot - self.porosity) * (1 - self.biot) / self.kd_cur +
                        self.porosity * self.fluid_compressibility)

        self.unstr_discr.init_matrix_stiffness({self.unstr_discr.physical_tags['matrix'][0]: {'E': self.E, 'nu': self.nu}})
        self.unstr_discr.physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        self.unstr_discr.physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        self.unstr_discr.physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        mech_xm = self.bc_type.ROLLER
        mech_xp = self.bc_type.FREE
        mech_ym = self.bc_type.ROLLER
        mech_yp = self.bc_type.STUCK_ROLLER(0.0)
        mech_zm = self.bc_type.ROLLER
        mech_zp = self.bc_type.ROLLER

        flow_xm = self.bc_type.NO_FLOW
        flow_xp = self.bc_type.AQUIFER(self.p_init)
        flow_ym = self.bc_type.NO_FLOW
        flow_yp = self.bc_type.NO_FLOW
        flow_zm = self.bc_type.NO_FLOW
        flow_zp = self.bc_type.NO_FLOW

        self.unstr_discr.boundary_conditions[self.bnd_tags['BND_X-']] = {'flow': flow_xm, 'mech': mech_xm, 'cells': []}
        self.unstr_discr.boundary_conditions[self.bnd_tags['BND_X+']] = {'flow': flow_xp, 'mech': mech_xp, 'cells': []}
        self.unstr_discr.boundary_conditions[self.bnd_tags['BND_Y-']] = {'flow': flow_ym, 'mech': mech_ym, 'cells': []}
        self.unstr_discr.boundary_conditions[self.bnd_tags['BND_Y+']] = {'flow': flow_yp, 'mech': mech_yp, 'cells': []}
        self.unstr_discr.boundary_conditions[self.bnd_tags['BND_Z-']] = {'flow': flow_zm, 'mech': mech_zm, 'cells': []}
        self.unstr_discr.boundary_conditions[self.bnd_tags['BND_Z+']] = {'flow': flow_zp, 'mech': mech_zp, 'cells': []}
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        self.b = np.max(self.unstr_discr.mesh_data.points[:, 1])
        self.F = -100.0 * self.a  # bar * m

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        scheme = 'non_stabilized'
        if scheme == 'stabilized':
            self.pm.scheme = scheme_type.apply_eigen_splitting_new
            self.pm.min_alpha_stabilization = 0.5
        elif scheme == 'non_stabilized':
            pass
        else:
            print('Error: unsupported scheme', scheme)
            exit(1)
        self.pm.neumann_boundaries_grad_reconstruction = True
        self.pm.grav = matrix([0.0, 0.0, 0.0], 1, 3)
        self.pm.visc = 1  # 9.81e-2
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

        self.ref_contact_cells = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intc)
        self.bc_rhs_ref = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs_prev = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.unstr_discr.pz_bounds = np.zeros(self.unstr_discr.bound_cells_tot)
        self.unstr_discr.pz_bounds = self.p_init
        self.unstr_discr.p_ref = np.zeros(self.unstr_discr.mat_cells_tot)
        self.unstr_discr.p_ref[:] = self.p_init
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['flow']
            # if flow['a'] == 1.0:
            #    c = self.unstr_discr.bound_cell_info_dict[bound_id].centroid
            #    if c[1] > 250 and c[1] < 750: bc.extend([flow['a'], flow['b'], 0.5 * self.p_init])
            #    else: bc.extend([0.0, 1.0, 0.0])
            # else:
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
            self.bc_rhs[4 * bound_id + 3] = flow['r']
            self.bc_rhs_prev[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
            self.bc_rhs_ref[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_ref[4 * bound_id + 3] = flow['r']
        # self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc
        self.unstr_discr.f = np.zeros(4 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
        self.unstr_discr.f[3::4] = self.p_init - self.unstr_discr.p_ref[:]

        MR = 0.9869 * 1.E-15 * self.permx / self.fluid_viscosity / 1.E-3
        K_dr = self.E / (3 * (1 - 2 * self.nu))
        self.K_nu = (K_dr + (4 / 3) * self.mu)
        Cv = 1.e+5 * MR * self.M * self.K_nu / (self.K_nu + self.biot ** 2 * self.M)
        self.tD = self.a ** 2 / Cv / 86400
        self.pD = abs(self.F / self.a) / 2
    def update_mandel_boundary(self, dt, time, physics):
        if self.discretizer_name == 'mech_discretizer':
            self.update_mandel_boundary_mech_discretizer(time)
        elif self.discretizer_name == 'pm_discretizer':
            self.update_mandel_boundary_pm_discretizer(time)

    def set_boundary_conditions_pm_discretizer(self):
        if self.discretizer_name == 'pm_discretizer':
            self.unstr_discr.boundary_conditions = self.boundary_conditions
            for key in self.boundary_conditions.keys():
                self.boundary_conditions[key]['cells'] = []

    def set_mandel_boundary_conditions(self, v_north=0.):
        self.boundary_conditions = {}
        self.boundary_conditions[self.bnd_tags['BND_X-']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.boundary_conditions[self.bnd_tags['BND_X+']] = {'flow': self.bc_type.AQUIFER(self.p_init),  'mech': self.bc_type.FREE}
        self.boundary_conditions[self.bnd_tags['BND_Y-']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.boundary_conditions[self.bnd_tags['BND_Y+']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.STUCK_ROLLER(v_north)}
        self.boundary_conditions[self.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.boundary_conditions[self.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.set_boundary_conditions_pm_discretizer()

    def set_terzaghi_boundary_conditions(self):
        self.boundary_conditions = {}
        self.boundary_conditions[self.bnd_tags['BND_X-']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.boundary_conditions[self.bnd_tags['BND_X+']] = {'flow': self.bc_type.AQUIFER(self.p_init),  'mech': self.bc_type.LOAD(self.F, [0.0, 0.0, 0.0])}
        self.boundary_conditions[self.bnd_tags['BND_Y-']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.boundary_conditions[self.bnd_tags['BND_Y+']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.boundary_conditions[self.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.boundary_conditions[self.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER}
        self.set_boundary_conditions_pm_discretizer()
    def set_bai_boundary_conditions(self):
        self.boundary_conditions = {}
        self.boundary_conditions[991] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER,                         'temp': self.bc_type.NO_FLOW }
        self.boundary_conditions[992] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER,                         'temp': self.bc_type.NO_FLOW }
        self.boundary_conditions[993] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER,                         'temp': self.bc_type.NO_FLOW }
        self.boundary_conditions[994] = {'flow': self.bc_type.AQUIFER(self.p_init),  'mech': self.bc_type.LOAD(self.F, [0.0, 0.0, 0.0]),  'temp': self.bc_type.AQUIFER(self.t_top) }
        self.boundary_conditions[995] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER,                         'temp': self.bc_type.NO_FLOW }
        self.boundary_conditions[996] = {'flow': self.bc_type.NO_FLOW,               'mech': self.bc_type.ROLLER,                         'temp': self.bc_type.NO_FLOW }

    def update_mandel_boundary_mech_discretizer(self, time):

        v_north = self.get_vertical_displacement_north_mandel(time)

        self.set_mandel_boundary_conditions(v_north)

        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            # flow
            self.bc_rhs[self.n_vars * ids + self.p_var] = bc['flow']['r']

            for id in ids:
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                conn_c = np.array(conn.c.values, copy=False)
                c1 = np.array(self.centroids[conn.elem_id1].values, copy=False)
                if n.dot(conn_c - c1) < 0: n *= -1.0
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']
    def update_mandel_boundary_pm_discretizer(self, time):

        v_north = self.get_vertical_displacement_north_mandel(time)

        self.set_mandel_boundary_conditions(v_north)

        self.pm.bc.clear()
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
            self.bc_rhs[4 * bound_id + 3] = flow['r']
    # Terzaghi
    def terzaghi_mech_discretizer(self, mesh='rect'):
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge.msh'
        elif mesh == 'hex':
            mesh_file = 'meshes/hexahedron.msh'
        self.mesh_filename = mesh_file

        self.mesh_data = meshio.read(mesh_file)

        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        self.porosity = 0.375
        self.permx = self.permy = self.permz = 10.0 / 9.81
        self.E = 10000 # in bars
        self.nu = 0.25
        self.lam = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.E / 2 / (1 + self.nu)
        self.biot = 0.9
        self.kd_cur = self.E / 3 / (1 - 2 * self.nu)
        self.fluid_compressibility = 1.e-5
        self.fluid_viscosity = 1.0
        self.M = 1.0 / ((self.biot - self.porosity) * (1 - self.biot) / self.kd_cur +
                            self.porosity * self.fluid_compressibility)
        self.F = -100.0 # bar * m

        self.set_terzaghi_boundary_conditions()

        self.init_mech_discretizer()

        self.porosity = self.porosity * np.ones(self.n_matrix + self.n_fracs)

        self.biot_mean = np.zeros(9 * (self.n_matrix + self.n_fracs))

        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):

            self.discr.perms.append(disc_matrix33(self.permx, self.permy, self.permz))
            self.discr.biots.append(disc_matrix33(self.biot))
            self.discr.stfs.append(disc_stiffness(self.lam, self.mu))
            self.biot_mean[9 * cell_id] = self.biot
            self.biot_mean[9 * cell_id + 4] = self.biot
            self.biot_mean[9 * cell_id + 8] = self.biot

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        ap = np.zeros(boundary_range[1] - boundary_range[0])
        bp = np.zeros(boundary_range[1] - boundary_range[0])
        an = np.zeros(boundary_range[1] - boundary_range[0])
        bn = np.zeros(boundary_range[1] - boundary_range[0])
        at = np.zeros(boundary_range[1] - boundary_range[0])
        bt = np.ones(boundary_range[1] - boundary_range[0])
        self.bc_rhs = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] -
                                                self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))
        self.bc_rhs_prev = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] -
                                                self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))

        # mapping boundary connections
        adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)
        id_sorted = np.argsort(adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = adj_matrix[id_sorted]
        self.conns = np.array(self.discr_mesh.conns, copy=False)
        self.centroids = np.array(self.discr_mesh.centroids, copy=False)

        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            ap[ids] = bc['flow']['a']
            bp[ids] = bc['flow']['b']
            an[ids] = bc['mech']['an']
            bn[ids] = bc['mech']['bn']
            at[ids] = bc['mech']['at']
            bt[ids] = bc['mech']['bt']
            # flow
            self.bc_rhs[self.n_vars * ids + self.p_var] = bc['flow']['r']
            self.bc_rhs_prev[self.n_vars * ids + self.p_var] = bc['flow']['r']

            for id in ids:
                assert(adj_matrix_cols[id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                conn_c = np.array(conn.c.values, copy=False)
                c1 = np.array(self.centroids[conn.elem_id1].values, copy=False)
                if n.dot(conn_c - c1) < 0: n *= -1.0
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']
                self.bc_rhs_prev[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']

        self.cpp_bc = THMBoundaryCondition()
        self.cpp_bc.flow.a = value_vector(ap)
        self.cpp_bc.flow.b = value_vector(bp)

        self.cpp_bc.mech_normal.a = value_vector(an)
        self.cpp_bc.mech_normal.b = value_vector(bn)
        self.cpp_bc.mech_tangen.a = value_vector(at)
        self.cpp_bc.mech_tangen.b = value_vector(bt)
        # to use base discretizer's class function reconstruct_pressure_gradients_per_cell
        # which doesn't know the new THMBoundaryCondition class yet
        self.cpp_flow = BoundaryCondition()
        self.cpp_flow.a = value_vector(ap)
        self.cpp_flow.b = value_vector(bp)

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

        MR = 0.9869 * 1.E-15 * self.permx / self.fluid_viscosity / 1.E-3
        K_dr = self.E / (3 * (1 - 2 * self.nu))
        self.K_nu = (K_dr + (4 / 3) * self.mu)
        Cv = 1.e+5 * MR * self.M * self.K_nu / (self.K_nu + self.biot ** 2 * self.M)
        self.tD = self.a ** 2 / Cv / 86400
        self.pD = abs(self.F / self.a) / 2

        # from compare_grad_discr import compare_gradients
        # compare_gradients('pm.pkl', new_cache_filename=None, orig_pm_arg=None, new_pm_arg=self.discr)
    def terzaghi_pm_discretizer(self, mesh='rect'):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        self.porosity = 0.375
        self.permx = self.permy = self.permz = 10.0 / 9.81
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge.msh'
        elif mesh == 'hex':
            mesh_file = 'meshes/hexahedron.msh'
        self.mesh_filename = mesh_file
        self.unstr_discr = UnstructDiscretizer(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0,
                                               mesh_file=mesh_file)
        self.unstr_discr.eps_t = 1.E+0
        self.unstr_discr.eps_n = 1.E+0
        self.unstr_discr.mu = 3.2
        self.unstr_discr.P12 = 0
        self.unstr_discr.Prol = 1
        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3
        self.unstr_discr.physical_tags['matrix'] = list(self.domain_tags[elem_loc.MATRIX])
        #lam = 1.0 * 10000  # in bar
        #mu = 1.0 * 10000
        #nu = lam / 2 / (lam + mu)
        #E = lam * (1 + nu) * (1 - 2 * nu) / nu

        self.E = 10000 # in bars
        self.nu = 0.25
        self.lam = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.E / 2 / (1 + self.nu)
        self.biot = 0.9
        self.kd_cur = self.E / 3 / (1 - 2 * self.nu)
        self.fluid_compressibility = 1.e-5
        self.fluid_viscosity = 1.0
        self.M = 1.0 / ((self.biot - self.porosity) * (1 - self.biot) / self.kd_cur +
                            self.porosity * self.fluid_compressibility)

        self.unstr_discr.init_matrix_stiffness({self.unstr_discr.physical_tags['matrix'][0]: {'E': self.E, 'nu': self.nu}})
        self.unstr_discr.physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        self.unstr_discr.physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        self.unstr_discr.physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.F = -100.0 # bar * m

        self.set_terzaghi_boundary_conditions()
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        scheme = 'non_stabilized'
        if scheme == 'stabilized':
            self.pm.scheme = scheme_type.apply_eigen_splitting_new
            self.pm.min_alpha_stabilization = 0.5
        elif scheme == 'non_stabilized':
            pass
        else:
            print('Error: unsupported scheme', scheme)
            exit(1)
        self.pm.neumann_boundaries_grad_reconstruction = True
        self.pm.grav = matrix([0.0, 0.0, 0.0], 1, 3)
        self.pm.visc = 1#9.81e-2
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

        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_cells_tot
        self.ref_contact_cells = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intc)
        self.bc_rhs_ref = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs_prev = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.unstr_discr.pz_bounds = np.zeros(self.unstr_discr.bound_cells_tot)
        self.unstr_discr.pz_bounds = self.p_init
        self.unstr_discr.p_ref = np.zeros(self.unstr_discr.mat_cells_tot)
        self.unstr_discr.p_ref[:] = self.p_init
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['flow']
            #if flow['a'] == 1.0:
            #    c = self.unstr_discr.bound_cell_info_dict[bound_id].centroid
            #    if c[1] > 250 and c[1] < 750: bc.extend([flow['a'], flow['b'], 0.5 * self.p_init])
            #    else: bc.extend([0.0, 1.0, 0.0])
            #else:
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
            self.bc_rhs[4 * bound_id + 3] = flow['r']
            self.bc_rhs_prev[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
            self.bc_rhs_ref[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_ref[4 * bound_id + 3] = flow['r']
        #self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc
        self.unstr_discr.f = np.zeros(4 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
        self.unstr_discr.f[3::4] = self.p_init - self.unstr_discr.p_ref[:]

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        MR = 0.9869 * 1.E-15 * self.permx / self.fluid_viscosity / 1.E-3
        K_dr = self.E / (3 * (1 - 2 * self.nu))
        self.K_nu = (K_dr + (4 / 3) * self.mu)
        Cv = 1.e+5 * MR * self.M * self.K_nu / (self.K_nu + self.biot ** 2 * self.M)
        self.tD = self.a ** 2 / Cv / 86400
        self.pD = np.fabs(self.F)
    # Two-layer Terzaghi
    def terzaghi_two_layers_pm_discretizer(self, mesh='rect'):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite_two_layers.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge_two_layers.msh'
        self.mesh_filename = mesh_file
        self.unstr_discr = UnstructDiscretizer(permx=1, permy=1, permz=1, frac_aper=0,
                                               mesh_file=mesh_file)

        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3

        self.fluid_compressibility = 1.e-10
        self.fluid_viscosity = 1.0

        # define correspondence between the physical tags in msh file and mesh elements types
        # two regions for different properties
        self.m1_tag = 99991
        self.m2_tag = 99992
        self.domain_tags, self.bnd_tags = set_domain_tags(matrix_tags=[self.m1_tag, self.m2_tag],
                    bnd_xm_tag=991, bnd_xp_tag=992,
                    bnd_ym_tag=993, bnd_yp_tag=994,
                    bnd_zm_tag=995, bnd_zp_tag=996)
        
        self.props = {      self.m1_tag: { 'h': 0.25, 'E': 10000, 'nu': 0.15, 'b': 0.9, 'poro': 0.15, 'perm': 1 },
                            self.m2_tag: { 'h': 0.75, 'E': 10000, 'nu': 0.15, 'b': 0.01, 'poro': 0.001, 'perm': 1  }     }
        x = (self.props[self.m2_tag]['b'] / self.props[self.m1_tag]['b'] * (3 * (self.props[self.m1_tag]['b'] - self.props[self.m1_tag]['poro']) * (1 - self.props[self.m1_tag]['b']) * (1 - self.props[self.m1_tag]['nu']) / (1 + self.props[self.m1_tag]['nu']) + self.props[self.m1_tag]['b'] ** 2) -
             self.props[self.m2_tag]['b'] ** 2) / 3 / (self.props[self.m2_tag]['b'] - self.props[self.m2_tag]['poro']) / (1 - self.props[self.m2_tag]['b'])
        nu2 = (1 - x) / (1 + x)
        self.props[self.m2_tag]['nu'] = nu2
        assert(nu2 < 0.5 and nu2 > 0)

        kd1 = self.props[self.m1_tag]['E'] / 3 / (1 - 2 * self.props[self.m1_tag]['nu'])
        self.props[self.m1_tag]['kd'] = kd1
        self.props[self.m1_tag]['M'] = 1.0 / ((self.props[self.m1_tag]['b'] - self.props[self.m1_tag]['poro']) * (1 - self.props[self.m1_tag]['b']) / kd1 +
                                        self.props[self.m1_tag]['poro'] * self.fluid_compressibility)

        kd2 = self.props[self.m2_tag]['E'] / 3 / (1 - 2 * self.props[self.m2_tag]['nu'])
        self.props[self.m2_tag]['kd'] = kd2
        self.props[self.m2_tag]['M'] = 1.0 / ((self.props[self.m2_tag]['b'] - self.props[self.m2_tag]['poro']) * (1 - self.props[self.m2_tag]['b']) / kd2 +
                                        self.props[self.m2_tag]['poro'] * self.fluid_compressibility)


        # some numbers for analytics
        for tag, p in self.props.items():
            p['m'] = (1 + p['nu']) * (1 - 2 * p['nu']) / p['E'] / (1 - p['nu'])
            # if tag == m2:
                # p['kd'] = kd1 * self.props[m1]['b'] * self.props[m1]['m'] / self.props[m2]['b'] / self.props[m2]['m'] / \
                #               (1 + kd1 * self.props[m1]['b'] * self.props[m1]['m'] * (self.props[m1]['b'] - self.props[m2]['b']))
            p['skempton'] = p['b'] * p['m'] * p['M'] / (1 + p['b'] ** 2 * p['m'] * p['M'])
            p['c'] = TC.darcy_constant * p['perm'] / self.fluid_viscosity * p['M'] / (1 + p['b'] ** 2 * p['m'] * p['M'])

        assert( np.fabs(self.props[self.m1_tag]['skempton'] - self.props[self.m2_tag]['skempton']) < 1.e-6 )

        self.unstr_discr.init_matrix_stiffness(self.props)
        self.unstr_discr.physical_tags['matrix'] = [self.m1_tag, self.m2_tag]
        self.unstr_discr.physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        self.unstr_discr.physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        self.unstr_discr.physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.F = -100.0 # bar * m

        self.set_terzaghi_boundary_conditions()
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_cells_tot

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.pm.visc = 1.0
        scheme = 'non_stabilized'
        if scheme == 'stabilized':
            self.pm.scheme = scheme_type.apply_eigen_splitting_new
            self.pm.min_alpha_stabilization = 0.5
        elif scheme == 'non_stabilized':
            pass
        else:
            print('Error: unsupported scheme', scheme)
            exit(1)
        self.pm.neumann_boundaries_grad_reconstruction = False
        self.pm.grav = matrix([0.0, 0.0, 0.0], 1, 3)
        self.kd_cur = np.zeros(self.n_matrix)
        self.porosity = np.zeros(self.n_matrix)
        self.biot_mean = np.zeros(9 * (self.n_matrix))
        for cell_id in range(self.n_matrix):
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

            E = self.props[cell.prop_id]['E']
            nu = self.props[cell.prop_id]['nu']
            biot = self.props[cell.prop_id]['b']
            k = self.props[cell.prop_id]['perm']
            kd = self.props[cell.prop_id]['kd']
            poro = self.props[cell.prop_id]['poro']
            lam = E * nu / (1 + nu) / (1 - 2 * nu)
            mu = E / 2 / (1 + nu)
            self.pm.stfs.append(engine_stiffness(lam, mu))
            self.pm.perms.append(engine_matrix33(k, k, k))
            self.pm.biots.append(engine_matrix33(biot))
            self.kd_cur[cell_id] = kd #(biot - self.porosity) * (1 - biot) * kd
            self.biot_mean[9 * cell_id] = biot
            self.biot_mean[9 * cell_id + 4] = biot
            self.biot_mean[9 * cell_id + 8] = biot
            self.porosity[cell_id] = poro

        self.bc_rhs_ref = np.zeros(self.n_vars * self.n_bounds)
        self.bc_rhs = np.zeros(self.n_vars * self.n_bounds)
        self.bc_rhs_prev = np.zeros(self.n_vars * self.n_bounds)
        self.ref_contact_cells = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intc)
        self.unstr_discr.pz_bounds = np.zeros(self.n_bounds)
        self.unstr_discr.pz_bounds = self.p_init
        self.unstr_discr.p_ref = np.zeros(self.n_matrix)
        self.unstr_discr.p_ref[:] = self.p_init
        for bound_id in range(self.n_bounds):
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['flow']
            #if flow['a'] == 1.0:
            #    c = self.unstr_discr.bound_cell_info_dict[bound_id].centroid
            #    if c[1] > 250 and c[1] < 750: bc.extend([flow['a'], flow['b'], 0.5 * self.p_init])
            #    else: bc.extend([0.0, 1.0, 0.0])
            #else:
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
            self.bc_rhs[4 * bound_id + 3] = flow['r']
            self.bc_rhs_prev[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
            self.bc_rhs_ref[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_ref[4 * bound_id + 3] = flow['r']
        #self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc
        self.unstr_discr.f = np.zeros(self.n_vars * self.n_matrix)
        self.unstr_discr.f[3::4] = self.p_init - self.unstr_discr.p_ref[:]

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        self.omega = self.approximate_roots_two_layers_terzaghi()
        self.tD = 1.0
        self.pD = 1.0
    def terzaghi_two_layers_no_analytics_pm_discretizer(self, mesh='rect'):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite_two_layers.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge_two_layers.msh'
        self.mesh_filename = mesh_file
        self.unstr_discr = UnstructDiscretizer(permx=1, permy=1, permz=1, frac_aper=0,
                                               mesh_file=mesh_file)

        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3

        # define correspondence between the physical tags in msh file and mesh elements types
        # two regions for different properties
        self.m1_tag = 99991
        self.m2_tag = 99992
        self.domain_tags, self.bnd_tags = set_domain_tags(matrix_tags=[self.m1_tag, self.m2_tag],
                    bnd_xm_tag=991, bnd_xp_tag=992,
                    bnd_ym_tag=993, bnd_yp_tag=994,
                    bnd_zm_tag=995, bnd_zp_tag=996)

        self.visc = 1#9.81e-2
        self.props = {      self.m1_tag: { 'h': 0.25, 'E': 10000, 'nu': 0.15, 'b': 0.0, 'poro': 0.0, 'perm': 1e-10 },
                            self.m2_tag: { 'h': 0.75, 'E': 10000, 'nu': 0.15, 'b': 0.9, 'poro': 0.15, 'perm': 1  }     }

        kd1 = self.props[self.m1_tag]['E'] / 3 / (1 - 2 * self.props[self.self.m1_tag]['nu'])
        self.props[self.m1_tag]['kd'] = kd1
        #self.props[self.m2_tag]['M'] = kd1 / (self.props[self.m1_tag]['b'] - self.props[self.m1_tag]['poro']) / (1 - self.props[self.m1_tag]['b'])

        kd2 = self.props[self.m2_tag]['E'] / 3 / (1 - 2 * self.props[self.m2_tag]['nu'])
        self.props[self.m2_tag]['kd'] = kd2
        #self.props[self.m2_tag]['M'] = kd2 / (self.props[self.m2_tag]['b'] - self.props[self.m2_tag]['poro']) / (1 - self.props[self.m2_tag]['b'])

        self.unstr_discr.init_matrix_stiffness(self.props)
        self.unstr_discr.physical_tags['matrix'] = [self.m1_tag, self.m2_tag]
        self.unstr_discr.physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        self.unstr_discr.physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        self.unstr_discr.physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])
        self.F = -100.0 # bar * m

        self.set_terzaghi_boundary_conditions()
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.pm.visc = self.visc
        scheme = 'non_stabilized'
        if scheme == 'stabilized':
            self.pm.scheme = scheme_type.apply_eigen_splitting_new
            self.pm.min_alpha_stabilization = 0.5
        elif scheme == 'non_stabilized':
            pass
        else:
            print('Error: unsupported scheme', scheme)
            exit(1)
        self.pm.neumann_boundaries_grad_reconstruction = False
        self.pm.grav = matrix([0.0, 0.0, 0.0], 1, 3)
        self.kd_cur = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
        self.porosity = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
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

            E = self.props[cell.prop_id]['E']
            nu = self.props[cell.prop_id]['nu']
            biot = self.props[cell.prop_id]['b']
            k = self.props[cell.prop_id]['perm']
            kd = self.props[cell.prop_id]['kd']
            poro = self.props[cell.prop_id]['poro']
            lam = E * nu / (1 + nu) / (1 - 2 * nu)
            mu = E / 2 / (1 + nu)
            self.pm.stfs.append(Stiffness(lam, mu))
            self.pm.perms.append(matrix33(k, k, k))
            self.pm.biots.append(matrix33(biot))
            self.kd_cur[cell_id] = kd #(biot - self.porosity) * (1 - biot) * kd
            self.biot_mean[9 * cell_id] = biot
            self.biot_mean[9 * cell_id + 4] = biot
            self.biot_mean[9 * cell_id + 8] = biot
            self.porosity[cell_id] = poro

        self.ref_contact_cells = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intc)
        self.bc_rhs_ref = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs_prev = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.unstr_discr.pz_bounds = np.zeros(self.unstr_discr.bound_cells_tot)
        self.unstr_discr.pz_bounds = self.p_init
        self.unstr_discr.p_ref = np.zeros(self.unstr_discr.mat_cells_tot)
        self.unstr_discr.p_ref[:] = self.p_init
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['flow']
            #if flow['a'] == 1.0:
            #    c = self.unstr_discr.bound_cell_info_dict[bound_id].centroid
            #    if c[1] > 250 and c[1] < 750: bc.extend([flow['a'], flow['b'], 0.5 * self.p_init])
            #    else: bc.extend([0.0, 1.0, 0.0])
            #else:
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
            self.bc_rhs[4 * bound_id + 3] = flow['r']
            self.bc_rhs_prev[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
            self.bc_rhs_ref[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_ref[4 * bound_id + 3] = flow['r']
        #self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc
        self.unstr_discr.f = np.zeros(4 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
        self.unstr_discr.f[3::4] = self.p_init - self.unstr_discr.p_ref[:]

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        self.tD = 1.0
        self.pD = 1.0
    def terzaghi_two_layers_mech_discretizer(self, mesh='rect'):
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite_two_layers.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge_two_layers.msh'
        self.mesh_filename = mesh_file

        self.mesh_data = meshio.read(mesh_file)

        # define correspondence between the physical tags in msh file and mesh elements types
        # two regions for different properties
        self.m1_tag = 99991
        self.m2_tag = 99992
        self.domain_tags, self.bnd_tags = set_domain_tags(matrix_tags=[self.m1_tag, self.m2_tag],
                                                            bnd_xm_tag=991, bnd_xp_tag=992,
                                                            bnd_ym_tag=993, bnd_yp_tag=994,
                                                            bnd_zm_tag=995, bnd_zp_tag=996)

        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        self.fluid_compressibility = 1.e-10
        self.fluid_viscosity = 1.0
        self.F = -100.0 # bar * m

        self.props = {      self.m1_tag: { 'h': 0.25, 'E': 10000, 'nu': 0.15, 'b': 0.9, 'poro': 0.15, 'perm': 1 },
                            self.m2_tag: { 'h': 0.75, 'E': 10000, 'nu': 0.15, 'b': 0.01, 'poro': 0.001, 'perm': 1  }     }
        x = (self.props[self.m2_tag]['b'] / self.props[self.m1_tag]['b'] * (3 * (self.props[self.m1_tag]['b'] - self.props[self.m1_tag]['poro']) * (1 - self.props[self.m1_tag]['b']) * (1 - self.props[self.m1_tag]['nu']) / (1 + self.props[self.m1_tag]['nu']) + self.props[self.m1_tag]['b'] ** 2) -
             self.props[self.m2_tag]['b'] ** 2) / 3 / (self.props[self.m2_tag]['b'] - self.props[self.m2_tag]['poro']) / (1 - self.props[self.m2_tag]['b'])
        nu2 = (1 - x) / (1 + x)
        self.props[self.m2_tag]['nu'] = nu2
        assert(nu2 < 0.5 and nu2 > 0)

        kd1 = self.props[self.m1_tag]['E'] / 3 / (1 - 2 * self.props[self.m1_tag]['nu'])
        self.props[self.m1_tag]['kd'] = kd1
        self.props[self.m1_tag]['M'] = 1.0 / ((self.props[self.m1_tag]['b'] - self.props[self.m1_tag]['poro']) * (1 - self.props[self.m1_tag]['b']) / kd1 +
                                        self.props[self.m1_tag]['poro'] * self.fluid_compressibility)

        kd2 = self.props[self.m2_tag]['E'] / 3 / (1 - 2 * self.props[self.m2_tag]['nu'])
        self.props[self.m2_tag]['kd'] = kd2
        self.props[self.m2_tag]['M'] = 1.0 / ((self.props[self.m2_tag]['b'] - self.props[self.m2_tag]['poro']) * (1 - self.props[self.m2_tag]['b']) / kd2 +
                                        self.props[self.m2_tag]['poro'] * self.fluid_compressibility)

        # some numbers for analytics
        for tag, p in self.props.items():
            p['m'] = (1 + p['nu']) * (1 - 2 * p['nu']) / p['E'] / (1 - p['nu'])
            # if tag == m2:
                # p['kd'] = kd1 * self.props[m1]['b'] * self.props[m1]['m'] / self.props[m2]['b'] / self.props[m2]['m'] / \
                #               (1 + kd1 * self.props[m1]['b'] * self.props[m1]['m'] * (self.props[m1]['b'] - self.props[m2]['b']))
            p['skempton'] = p['b'] * p['m'] * p['M'] / (1 + p['b'] ** 2 * p['m'] * p['M'])
            p['c'] = TC.darcy_constant * p['perm'] / self.fluid_viscosity * p['M'] / (1 + p['b'] ** 2 * p['m'] * p['M'])

        assert( np.fabs(self.props[self.m1_tag]['skempton'] - self.props[self.m2_tag]['skempton']) < 1.e-6 )

        self.set_terzaghi_boundary_conditions()

        self.init_mech_discretizer()

        self.kd_cur = np.zeros(self.n_matrix)
        self.porosity = np.zeros(self.n_matrix)
        self.biot_mean = np.zeros(9 * (self.n_matrix))

        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            tag = self.tags[cell_id]
            E = self.props[tag]['E']
            nu = self.props[tag]['nu']
            biot = self.props[tag]['b']
            k = self.props[tag]['perm']
            kd = self.props[tag]['kd']
            poro = self.props[tag]['poro']
            lam = E * nu / (1 + nu) / (1 - 2 * nu)
            mu = E / 2 / (1 + nu)

            self.discr.perms.append(disc_matrix33(k, k, k))
            self.discr.biots.append(disc_matrix33(biot))
            self.discr.stfs.append(disc_stiffness(lam, mu))
            self.biot_mean[9 * cell_id] = biot
            self.biot_mean[9 * cell_id + 4] = biot
            self.biot_mean[9 * cell_id + 8] = biot
            self.porosity[cell_id] = poro
            self.kd_cur[cell_id] = kd

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        ap = np.zeros(boundary_range[1] - boundary_range[0])
        bp = np.zeros(boundary_range[1] - boundary_range[0])
        an = np.zeros(boundary_range[1] - boundary_range[0])
        bn = np.zeros(boundary_range[1] - boundary_range[0])
        at = np.zeros(boundary_range[1] - boundary_range[0])
        bt = np.ones(boundary_range[1] - boundary_range[0])
        self.bc_rhs = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] -
                                                self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))
        self.bc_rhs_prev = np.zeros(self.n_vars * (self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1] -
                                                self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]))

        # mapping boundary connections
        adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)
        id_sorted = np.argsort(adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = adj_matrix[id_sorted]
        self.conns = np.array(self.discr_mesh.conns, copy=False)
        self.centroids = np.array(self.discr_mesh.centroids, copy=False)
        self.ref_contact_cells = np.zeros(self.n_fracs, dtype=np.intc)

        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            ap[ids] = bc['flow']['a']
            bp[ids] = bc['flow']['b']
            an[ids] = bc['mech']['an']
            bn[ids] = bc['mech']['bn']
            at[ids] = bc['mech']['at']
            bt[ids] = bc['mech']['bt']
            # flow
            self.bc_rhs[self.n_vars * ids + self.p_var] = bc['flow']['r']
            self.bc_rhs_prev[self.n_vars * ids + self.p_var] = bc['flow']['r']

            for id in ids:
                assert(adj_matrix_cols[id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                conn_c = np.array(conn.c.values, copy=False)
                c1 = np.array(self.centroids[conn.elem_id1].values, copy=False)
                if n.dot(conn_c - c1) < 0: n *= -1.0
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']
                self.bc_rhs_prev[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']

        self.cpp_bc = THMBoundaryCondition()
        self.cpp_bc.flow.a = value_vector(ap)
        self.cpp_bc.flow.b = value_vector(bp)

        self.cpp_bc.mech_normal.a = value_vector(an)
        self.cpp_bc.mech_normal.b = value_vector(bn)
        self.cpp_bc.mech_tangen.a = value_vector(at)
        self.cpp_bc.mech_tangen.b = value_vector(bt)
        # to use base discretizer's class function reconstruct_pressure_gradients_per_cell
        # which doesn't know the new THMBoundaryCondition class yet
        self.cpp_flow = BoundaryCondition()
        self.cpp_flow.a = value_vector(ap)
        self.cpp_flow.b = value_vector(bp)

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

        self.omega = self.approximate_roots_two_layers_terzaghi()
        self.tD = 1.0
        self.pD = 1.0

    # Bai, 2005 (unidimensional thermoporoelastic consolidation)
    def bai_thermoporoelastic_consolidation(self, mesh='rect'):
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite_bai.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge_bai.msh'
        elif mesh == 'hex':
            mesh_file = 'meshes/hexahedron_bai.msh'
        self.mesh_filename = mesh_file

        self.mesh_data = meshio.read(mesh_file)

        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        self.t_init = 0.0#273.15
        self.t_top = self.t_init + 50
        self.p_top = self.p_init
        self.porosity = 0.2
        self.permx = self.permy = self.permz = 4.e+6 / 0.9869
        self.E = 0.06 # in bars
        self.nu = 0.4
        self.lam = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.E / 2 / (1 + self.nu)
        self.biot = 1.0
        self.kd_cur = self.E / 3 / (1 - 2 * self.nu)
        self.fluid_compressibility = 0.0
        self.fluid_viscosity = 1.0
        self.F = -1.e-5
        self.th_expn_coef = 9.0 * 1.E-7
        self.th_expn = self.th_expn_coef * self.kd_cur
        self.th_conductivity = 0.836 * 86400.0 * 1000

        self.set_bai_boundary_conditions()

        self.init_mech_discretizer()

        self.biot_mean = np.zeros(9 * (self.n_matrix + self.n_fracs))

        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            self.discr.perms.append(disc_matrix33(self.permx, self.permy, self.permz))
            self.discr.biots.append(disc_matrix33(self.biot))
            self.discr.stfs.append(disc_stiffness(self.lam, self.mu))
            self.discr.heat_conductions.append(disc_matrix33(self.th_conductivity))
            self.discr.thermal_expansions.append(disc_matrix33(self.th_expn))
            self.biot_mean[9 * cell_id] = self.biot
            self.biot_mean[9 * cell_id + 4] = self.biot
            self.biot_mean[9 * cell_id + 8] = self.biot

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

        # mapping boundary connections
        adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)
        id_sorted = np.argsort(adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = adj_matrix[id_sorted]
        self.conns = np.array(self.discr_mesh.conns, copy=False)
        self.centroids = np.array(self.discr_mesh.centroids, copy=False)

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

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_flow,  self.cpp_heat)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

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
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """
        # Add injection well:
        # self.add_well("I1", 0.5)
        # # Perforate all boundary cells:
        # for nth_perf in range(len(self.left_boundary_cells)):
        #     well_index = self.mesh.volume[self.left_boundary_cells[nth_perf]] / self.max_well_vol * self.well_index
        #     self.add_perforation(well=self.wells[-1], res_block=self.left_boundary_cells[nth_perf],
        #                          well_index=well_index)
        #
        # # Add production well:
        # self.add_well("P1", 0.5)
        # # Perforate all boundary cells:
        # for nth_perf in range(len(self.right_boundary_cells)):
        #     well_index = self.mesh.volume[self.right_boundary_cells[nth_perf]] / self.max_well_vol * self.well_index
        #     self.add_perforation(self.wells[-1], res_block=self.right_boundary_cells[nth_perf],
        #                          well_index=well_index)
        #
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
    # Mandel analytics
    def get_vertical_displacement_north_mandel(self, t):
        # Parameters
        F = np.fabs(self.F)
        K_s = self.lam + 2 * self.mu / 3
        skempton = self.biot * self.M / (K_s + self.M * self.biot ** 2)
        nu_s = self.nu
        nu_u = (3 * self.nu + self.biot * skempton * (1 - 2 * self.nu)) / (3 - self.biot * skempton * (1 - 2 * self.nu))
        mu_s = self.mu
        mu_f = self.fluid_viscosity
        k_s = self.permx / self.fluid_viscosity
        c_f = TC.darcy_constant * (2 * k_s * (skempton ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / ( 9 * mu_f * (1 - nu_u) * (nu_u - nu_s) )

        # Calculate constants
        aa_n = self.approximate_roots()[:, np.newaxis]

        cy0 = (-F * (1 - nu_s)) / (2 * mu_s * self.a)
        cy1 = F * (1 - nu_u) / (mu_s * self.a)

        # Calculate exact north boundary condition
        uy_sum = np.sum(
            ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
            * np.exp((-(aa_n**2) * c_f * t) / (self.a**2)),
            axis=0,
        )

        north_bc = (cy0 + cy1 * uy_sum) * self.b
        return north_bc
    def approximate_roots(self) -> np.ndarray:
        """
        f(x) = tan(x) - ((1-nu)/(nu_u-nu)) x
        """
        # Parameters
        K_s = self.lam + 2 * self.mu / 3
        skempton = self.biot * self.M / (K_s + self.M * self.biot ** 2)
        nu_s = self.nu
        nu_u = (3 * self.nu + self.biot * skempton * (1 - 2 * self.nu)) / (3 - self.biot * skempton * (1 - 2 * self.nu))
        # Function f(x)
        def f(x):
            y = np.tan(x) - ((1 - nu_s) / (nu_u - nu_s)) * x
            return y

        n_series = 200
        a_n = np.zeros(n_series)  # initializing roots array
        x0 = 0  # initial point
        for i in range(n_series):
            a_n[i] = opt.bisect(
                f,  # function
                x0 + np.pi / 4,  # left point
                x0 + np.pi / 2 - 10000000 * 2.2204e-16,  # right point
                xtol=1e-30,  # absolute tolerance
                rtol=1e-14,  # relative tolerance
            )
            x0 += np.pi  # apply a phase change of pi to get the next root

        return a_n
    def mandel_exact_pressure(self, t, xc) -> np.ndarray:
        """
        Pressure solution for a given time `t`.
        """
        # Parameters
        F = np.fabs(self.F)
        K_s = self.lam + 2 * self.mu / 3
        skempton = self.biot * self.M / (K_s + self.M * self.biot ** 2)
        nu_s = self.nu
        nu_u = (3 * self.nu + self.biot * skempton * (1 - 2 * self.nu)) / (3 - self.biot * skempton * (1 - 2 * self.nu))
        mu_s = self.mu
        mu_f = self.fluid_viscosity
        k_s = self.permx / self.fluid_viscosity
        c_f = TC.darcy_constant * (2 * k_s * (skempton ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / ( 9 * mu_f * (1 - nu_u) * (nu_u - nu_s) )

        if t == 0.0:  # initial condition has its own expression
            p = ((F * skempton * (1 + nu_u)) / (3 * self.a)) * np.ones(xc.size)
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact p
            c0 = (2 * F * skempton * (1 + nu_u)) / (3 * self.a)
            p_sum_0 = np.sum(
                ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * (np.cos((aa_n * xc) / self.a) - np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (self.a**2)),
                axis=0,
            )
            p = c0 * p_sum_0

        return p
    def mandel_exact_displacements(self, t, xc) -> np.ndarray:
        """
        Exact pressure solution for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            p (sd.num_cells, ): Exact pressure solution.

        """

        # Retrieve physical data
        F = np.fabs(self.F)
        K_s = self.lam + 2 * self.mu / 3
        skempton = self.biot * self.M / (K_s + self.M * self.biot ** 2)
        nu_s = self.nu
        nu_u = (3 * self.nu + self.biot * skempton * (1 - 2 * self.nu)) / (3 - self.biot * skempton * (1 - 2 * self.nu))
        mu_s = self.mu
        mu_f = self.fluid_viscosity
        k_s = self.permx / self.fluid_viscosity
        c_f = TC.darcy_constant * (2 * k_s * (skempton ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / ( 9 * mu_f * (1 - nu_u) * (nu_u - nu_s) )

        # -----> Compute exact fluid pressure

        if t == 0.0:  # initial condition has its own expression
            p = ((F * skempton * (1 + nu_u)) / (3 * self.a)) * np.ones(xc.shape[0])
            ux = F / self.mu / self.a * nu_u * xc[:, 0] / 2
            uy = F / self.mu / self.a * (nu_u - 1) * xc[:, 1] / 2
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact p
            c0 = (2 * F * skempton * (1 + nu_u)) / (3 * self.a)
            p_sum_0 = np.sum(
                ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * (np.cos((aa_n * xc[:,0]) / self.a) - np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (self.a**2)),
                axis=0,
            )
            p = c0 * p_sum_0
            # Exact ux
            ux_sum_0 = np.sum(
                ( (self.a * np.sin((aa_n * xc[:,0]) / self.a) - nu_u * xc[:,0] * np.sin(aa_n)) * np.cos(aa_n) /
                  (aa_n - (np.sin(aa_n) * np.cos(aa_n))) ) * np.exp((-(aa_n ** 2) * c_f * t) / (self.a ** 2)),
                axis=0,
            )
            ux = F / self.mu / self.a * (self.nu * xc[:,0] / 2 + ux_sum_0)
            # Exact uy
            uy_sum_0 = np.sum(
                ( np.sin(aa_n) * np.cos(aa_n) /
                  (aa_n - (np.sin(aa_n) * np.cos(aa_n))) ) * np.exp((-(aa_n ** 2) * c_f * t) / (self.a ** 2)),
                axis=0,
            )
            uy = F / self.mu / self.a * ((self.nu - 1) * xc[:,1] / 2 - (nu_u - 1) * xc[:,1] * uy_sum_0)

        return p, ux, uy
    # Terzaghi analytics
    def terzaghi_exact_pressure0(self, t, xc) -> np.ndarray:
        h = self.a
        vertical_load = self.F
        dimless_t = t / self.tD

        n = 1000

        sum_series = np.zeros_like(xc)
        for i in range(1, n + 1):
            sum_series += (
                    (((-1) ** (i - 1)) / (2 * i - 1))
                    * np.cos((2 * i - 1) * (np.pi / 2) * (xc / h))
                    * np.exp((-((2 * i - 1) ** 2)) * (np.pi ** 2 / 4) * dimless_t) )
        p = (4 / np.pi) * vertical_load * sum_series
        return p
    def terzaghi_exact_pressure(self, t, xc) -> np.ndarray:
        # Parameters
        K_s = self.lam + 2 * self.mu / 3
        skempton = self.biot * self.M / (K_s + self.M * self.biot ** 2)
        nu_s = self.nu
        nu_u = (3 * self.nu + self.biot * skempton * (1 - 2 * self.nu)) / (3 - self.biot * skempton * (1 - 2 * self.nu))
        mu_s = self.mu
        k_s = self.permx / self.fluid_viscosity
        c_f = TC.darcy_constant * (2 * k_s * (skempton ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / ( 9 * (1 - nu_u) * (nu_u - nu_s) )

        h = self.a
        vertical_load = np.fabs(self.F)
        dimless_t = t# / self.tD

        n = 1000

        p0 = vertical_load * skempton * (1 + nu_u) / 3 / (1 - nu_u)
        c = TC.darcy_constant * 2 * k_s * self.mu * (1 - nu_s) * (nu_u - nu_s) / self.biot ** 2 / (1 - nu_u) / (1 - 2 * nu_s) ** 2

        if dimless_t > 0:
            sum_series = np.zeros_like(xc)
            for m in range(0, n):
                sum_series += (-1) ** m * (erfc( ((1 + 2*m) * h + xc) / np.sqrt(4 * c * dimless_t) ) +
                                           erfc( ((1 + 2*m) * h - xc) / np.sqrt(4 * c * dimless_t) ) )
            p =  p0 * (1 - sum_series)
        else:
            p = p0
        return p
    def terzaghi_exact_displacements(self, t, xc) -> np.ndarray:
        """Compute exact pressure.
        Args:
            t: Time in seconds.
        Returns:
            Exact pressure for the given time `t`.
        """
        # Retrieve physical data
        K_s = self.lam + 2 * self.mu / 3
        skempton = self.biot * self.M / (K_s + self.M * self.biot ** 2)
        nu_s = self.nu
        nu_u = (3 * self.nu + self.biot * skempton * (1 - 2 * self.nu)) / (3 - self.biot * skempton * (1 - 2 * self.nu))
        mu_s = self.mu
        k_s = self.permx / self.fluid_viscosity
        c_f = TC.darcy_constant * (2 * k_s * (skempton ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / ( 9 * (1 - nu_u) * (nu_u - nu_s) )

        h = self.a
        vertical_load = np.fabs(self.F)
        dimless_t = t# / self.tD

        n = 1000

        u0 = -xc * vertical_load * (1  - 2 * self.nu) / 2 / self.mu / (1 - self.nu)
        c = TC.darcy_constant * 2 * k_s * self.mu * (1 - nu_s) * (nu_u - nu_s) / self.biot ** 2 / (1 - nu_u) / (1 - 2 * nu_s) ** 2
        coef = 4 * vertical_load * h * (nu_u - self.nu) / np.pi ** 2 / self.mu / (1 - self.nu) / (1 - nu_u)

        if dimless_t > 0:
            sum_series = np.zeros_like(xc)
            for m in range(0, n):
                sum_series += np.exp(-(2 * m + 1) ** 2 * np.pi ** 2 * c * dimless_t / 4 / h ** 2) * \
                              np.cos((2 * m + 1) * np.pi * (xc + h) / 2 / h) / (2 * m + 1) ** 2
            u = u0 - coef * sum_series
        else:
            u = u0
        return u
    # Two-layer Terzaghi analytics
    def approximate_roots_two_layers_terzaghi(self) -> np.ndarray:
        # Retrieve physical data
        p1 = self.props[self.m1_tag]
        p2 = self.props[self.m2_tag]
        self.beta = p2['perm'] / p1['perm'] * p1['c'] / p2['c']
        self.theta = p1['h'] / p2['h'] * np.sqrt(p2['c'] / p1['c'])

        # Define algebraic function
        def f(x):
            y = np.cos(x) - (self.beta - 1) / (self.beta + 1) * np.cos((self.theta - 1) / (self.theta + 1) * x)
            return y
        def dfdx(x):
            ydot = -np.sin(x) + (self.theta - 1) / (self.theta + 1) * (self.beta - 1) / (self.beta + 1) * np.sin((self.theta - 1) / (self.theta + 1) * x)
            return ydot

        n_series = 1000
        a_n = np.zeros(n_series)  # initializing roots array
        x0 = np.pi / 2  # initial point
        for i in range(n_series):
            a_n[i] = opt.newton(
                func=f,         # function
                x0=x0,          # point
                fprime=dfdx,    # derivative
                tol=1e-30,      # absolute tolerance
                rtol=1e-14,     # relative tolerance
            )
            x0 += np.pi  # apply a phase change of pi to get the next root

        assert( np.unique(a_n).size == a_n.size )

        return a_n / (1 + self.theta)
    def terzaghi_two_layers_exact_pressure(self, t, xc, n_roots=1000) -> np.ndarray:
        """Compute exact pressure.
        Args:
            t: Time in seconds.
        Returns:
            Exact pressure for the given time `t`.
        """
        # Retrieve physical data
        h1 = self.a * self.props[self.m1_tag]['h']
        h2 = self.a * self.props[self.m2_tag]['h']
        c2 = self.props[self.m2_tag]['c']
        xi = xc - h2
        skempton = self.props[self.m1_tag]['skempton']

        assert(n_roots <= self.omega.size)

        g = 2 * skempton * np.fabs(self.F) / self.omega * \
            np.exp(-c2 * t * self.omega ** 2 / h2 ** 2 ) / \
            ( (1 + self.beta * self.theta) * np.cos(self.theta * self.omega) * np.sin(self.omega) + \
              (self.beta + self.theta) * np.sin(self.theta * self.omega) * np.cos(self.omega) )

        t1 = np.cos(self.omega) * np.cos(self.theta * np.outer(xi, self.omega) / h1) - \
                self.beta * np.sin(self.omega) * np.sin(self.theta * np.outer(xi, self.omega) / h1)
        t2 = np.cos(self.omega) * np.cos(np.outer(xi, self.omega) / h2) - \
                np.sin(self.omega) * np.sin(np.outer(xi, self.omega) / h2)

        p = np.sum((g * t1)[:,:n_roots], axis=1)
        p[xi < 0] = np.sum((g * t2)[:,:n_roots], axis=1)[xi < 0]

        return p
    def terzaghi_two_layers_exact_displacement(self, t, xc, n_roots=1000) -> np.ndarray:
        """Compute exact pressure.
        Args:
            t: Time in seconds.
        Returns:
            Exact pressure for the given time `t`.
        """
        # Retrieve physical data
        h1 = self.a * self.props[self.m1_tag]['h']
        h2 = self.a * self.props[self.m2_tag]['h']
        b1 = self.props[self.m1_tag]['b']
        b2 = self.props[self.m2_tag]['b']
        m1 = self.props[self.m1_tag]['m']
        m2 = self.props[self.m2_tag]['m']
        c2 = self.props[self.m2_tag]['c']
        xi = xc - h2
        skempton = self.props[self.m1_tag]['skempton']

        assert(n_roots <= self.omega.size)

        g = 2 * skempton * np.fabs(self.F) / self.omega * \
            np.exp(-c2 * t * self.omega ** 2 / h2 ** 2 ) / \
            ( (1 + self.beta * self.theta) * np.cos(self.theta * self.omega) * np.sin(self.omega) + \
              (self.beta + self.theta) * np.sin(self.theta * self.omega) * np.cos(self.omega) )

        t1 = b1 * m1 * h1 * (np.cos(self.omega) * np.sin(self.theta * np.outer(xi, self.omega) / h1) + \
                self.beta * np.sin(self.omega) * np.cos(self.theta * np.outer(xi, self.omega) / h1)) - \
            b1 * m1 * h1 * self.beta * np.sin(self.omega) + b2 * m2 * h2 * self.theta * np.sin(self.omega)
        t2 = b2 * m2 * h2 * self.theta * ( np.cos(self.omega) * np.sin(np.outer(xi, self.omega) / h2) + \
                np.sin(self.omega) * np.cos(np.outer(xi, self.omega) / h2) )

        u = np.fabs(self.F) * (m1 * xi + m2 * h2) \
            - np.sum((g * t1 / self.omega)[:,:n_roots], axis=1) / self.theta
        u[xi < 0] = (np.fabs(self.F) * m2 * (xi + h2) \
            - np.sum((g * t2 / self.omega)[:,:n_roots], axis=1) / self.theta)[xi < 0]

        return -u
