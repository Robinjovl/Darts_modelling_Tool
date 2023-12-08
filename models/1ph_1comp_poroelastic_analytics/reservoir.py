from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, contact, contact_vector, vector_matrix, scheme_type
from darts.engines import matrix33 as engine_matrix33
from darts.discretizer import matrix33 as disc_matrix33
from darts.engines import Stiffness as engine_stiffness
from darts.discretizer import Stiffness as disc_stiffness
from darts.engines import matrix, pm_discretizer, Face, vector_face_vector, face_vector, vector_matrix33, stf_vector, critical_stress
import numpy as np
from math import inf, pi
from darts.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.mesh.geometrymodule import FType
from darts.engines import timer_node
from itertools import compress
import meshio
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.linalg import null_space
from darts.mesh.transcalc import TransCalculations as TC
import scipy.optimize as opt
import scipy
from scipy.special import erfc as erfc

import darts.discretizer as dis
from darts.discretizer import Mesh, Elem, poro_mech_discretizer, THMBoundaryCondition, BoundaryCondition, elem_loc, elem_type, conn_type
from darts.discretizer import vector_matrix33, vector_vector3, matrix, value_vector, index_vector

# Definitions for the unstructured reservoir class:
class UnstructReservoir:
    def __init__(self, timer, case='mandel', discretizer='new_discretizer', mesh='rect'):
        self.timer = timer
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        self.discretizer_name = discretizer
        self.n_vars = 4

        # Specify elastic properties, mesh & boundaries
        if case == 'mandel':
            if discretizer == 'new_discretizer':
                self.mandel_north_dirichlet(mesh)
            elif discretizer == 'pm_discretizer':
                self.mandel_north_dirichlet_pm_discretizer(mesh)
        elif case == 'terzaghi':
            if discretizer == 'new_discretizer':
                self.terzaghi(mesh)
            elif discretizer == 'pm_discretizer':
                self.terzaghi_pm_discretizer(mesh)
        elif case == 'terzaghi_two_layers':
            self.terzaghi_two_layers(mesh)
        elif case == 'terzaghi_two_layers_no_analytics':
            self.terzaghi_two_layers_no_analytics(mesh)

        dt = 0.0
        if discretizer == 'new_discretizer':
            self.mesh.init_pm_new(self.discr.cell_m, self.discr.cell_p,
                              self.discr.flux_stencil, self.discr.flux_offset,
                              self.discr.hooke, self.discr.hooke_rhs,
                              self.discr.biot_traction, self.discr.biot_traction_rhs,
                              self.discr.darcy, self.discr.darcy_rhs,
                              self.discr.biot_vol_strain, self.discr.biot_vol_strain_rhs,
                              self.n_matrix, self.n_bounds, self.n_fracs)
        elif discretizer == 'pm_discretizer':
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

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.bc = np.array(self.mesh.bc, copy=False)
        self.bc_prev = np.array(self.mesh.bc_prev, copy=False)
        self.bc_ref = np.array(self.mesh.bc_ref, copy=False)
        if discretizer == 'new_discretizer':
            self.mesh.f.resize(4 * (self.n_fracs + self.n_matrix))
            self.f = np.array(self.mesh.f, copy=False)
            self.biot_arr = np.array(self.mesh.biot, copy=False)
            self.kd = np.array(self.mesh.kd, copy=False)
            self.mesh.pz_bounds.resize(self.n_bounds)
            self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
            self.p_ref = np.array(self.mesh.ref_pressure, copy=False)

            self.poro[:self.n_matrix] = self.porosity
            self.poro[self.n_matrix:] = 1
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.volume[:self.n_matrix] = volumes[:self.n_matrix]
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            # self.biot_arr[:] = np.tile([0,0,0,
            #                             0,0,0,
            #                             0,0,0], self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.biot_arr[:] = self.biot_mean
            self.kd[:] = self.kd_cur
            # self.pz_bounds[:] = self.pz_bounds
            # self.p_ref[:] = self.p_ref
            # self.f[:] = self.f
        elif discretizer == 'pm_discretizer':
            self.mesh.f.resize(4 * (self.unstr_discr.frac_cells_tot + self.unstr_discr.mat_cells_tot))
            self.f = np.array(self.mesh.f, copy=False)
            self.biot_arr = np.array(self.mesh.biot, copy=False)
            self.kd = np.array(self.mesh.kd, copy=False)
            self.mesh.pz_bounds.resize(self.unstr_discr.bound_cells_tot)
            self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
            self.p_ref = np.array(self.mesh.ref_pressure, copy=False)

            self.poro[:self.unstr_discr.mat_cells_tot] = self.porosity
            self.poro[self.unstr_discr.mat_cells_tot:] = 1
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

        self.wells = []
    def update_mandel_boundary(self, dt, time, physics):
        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        v_north = self.get_vertical_displacement_north_mandel(time)

        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        self.boundary_conditions = {}
        self.boundary_conditions[991] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[992] = {'flow': AQUIFER(self.p_init),  'mech': FREE}
        self.boundary_conditions[993] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[994] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(v_north)}
        self.boundary_conditions[995] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[996] = {'flow': NO_FLOW,               'mech': ROLLER}

        p_var = 0
        u_var = 1
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            # flow
            self.bc_rhs[self.n_vars * ids + p_var] = bc['flow']['r']
            self.bc_rhs[self.n_vars * ids + p_var] = bc['flow']['r']

            for id in ids:
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                self.bc_rhs[self.n_vars * id + u_var:self.n_vars * id + u_var + 3] = bc['mech']['rn'] * n + bc['mech']['rt']

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

    # new discretizer
    def mandel_north_dirichlet(self, mesh='rect'):
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge.msh'
        elif mesh == 'hex':
            mesh_file = 'meshes/hexahedron.msh'
        self.file_path = mesh_file

        self.mesh_data = meshio.read(mesh_file)
        self.domain_tags = dict()
        self.domain_tags[elem_loc.MATRIX] = set([99991])
        self.domain_tags[elem_loc.FRACTURE] = set([])  # 9991, 9992])
        self.domain_tags[elem_loc.BOUNDARY] = set([991, 992, 993, 994, 995, 996])
        self.domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()  # is this for poromechanics??

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

        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        self.boundary_conditions = {}
        self.boundary_conditions[991] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[992] = {'flow': AQUIFER(self.p_init),  'mech': FREE}
        self.boundary_conditions[993] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[994] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(0.0)}
        self.boundary_conditions[995] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[996] = {'flow': NO_FLOW,               'mech': ROLLER}

        self.discr_mesh = Mesh()

        # init poromechanics discretizer
        self.discr_mesh.gmsh_mesh_processing(mesh_file, self.domain_tags)

        self.a = np.max([node.values[0] for node in self.discr_mesh.nodes])
        self.F = -100.0 * self.a # bar * m

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
            self.bc_rhs[self.n_vars * ids + 3] = bc['flow']['r']
            self.bc_rhs_prev[self.n_vars * ids + 3] = bc['flow']['r']
            self.bc_rhs[self.n_vars * ids + 3] = bc['flow']['r']
            self.bc_rhs_prev[self.n_vars * ids + 3] = bc['flow']['r']

            for id in ids:
                assert(adj_matrix_cols[id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                self.bc_rhs[self.n_vars * id + u_var:self.n_vars * id + u_var + 3] = bc['mech']['rn'] * n + bc['mech']['rt']
                self.bc_rhs_prev[self.n_vars * id + u_var:self.n_vars * id + u_var + 3] = bc['mech']['rn'] * n + bc['mech']['rt']

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
        self.timer.node["discretization"].stop()

        MR = 0.9869 * 1.E-15 * self.permx / self.fluid_viscosity / 1.E-3
        K_dr = self.E / (3 * (1 - 2 * self.nu))
        self.K_nu = (K_dr + (4 / 3) * self.mu)
        Cv = 1.e+5 * MR * self.M * self.K_nu / (self.K_nu + self.biot ** 2 * self.M)
        self.tD = self.a ** 2 / Cv / 86400
        self.pD = abs(self.F / self.a) / 2

        # from compare_grad_discr import compare_gradients
        # compare_gradients('pm.pkl', new_cache_filename=None, orig_pm_arg=None, new_pm_arg=self.discr)
    def terzaghi(self, mesh='rect'):
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge.msh'
        elif mesh == 'hex':
            mesh_file = 'meshes/hexahedron.msh'
        self.file_path = mesh_file

        self.mesh_data = meshio.read(mesh_file)
        self.domain_tags = dict()
        self.domain_tags[elem_loc.MATRIX] = set([99991])
        self.domain_tags[elem_loc.FRACTURE] = set([])  # 9991, 9992])
        self.domain_tags[elem_loc.BOUNDARY] = set([991, 992, 993, 994, 995, 996])
        self.domain_tags[elem_loc.FRACTURE_BOUNDARY] = set()  # is this for poromechanics??

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

        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        self.boundary_conditions = {}
        self.boundary_conditions[991] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[992] = {'flow': AQUIFER(self.p_init),  'mech': LOAD(self.F, [0.0, 0.0, 0.0])}
        self.boundary_conditions[993] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[994] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[995] = {'flow': NO_FLOW,               'mech': ROLLER}
        self.boundary_conditions[996] = {'flow': NO_FLOW,               'mech': ROLLER}

        self.discr_mesh = Mesh()

        # init poromechanics discretizer
        self.discr_mesh.gmsh_mesh_processing(mesh_file, self.domain_tags)

        self.a = np.max([node.values[0] for node in self.discr_mesh.nodes])

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
            self.bc_rhs[self.n_vars * ids + 3] = bc['flow']['r']
            self.bc_rhs_prev[self.n_vars * ids + 3] = bc['flow']['r']
            self.bc_rhs[self.n_vars * ids + 3] = bc['flow']['r']
            self.bc_rhs_prev[self.n_vars * ids + 3] = bc['flow']['r']

            for id in ids:
                assert(adj_matrix_cols[id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values, copy=False)
                self.bc_rhs[self.n_vars * id + u_var:self.n_vars * id + u_var + 3] = bc['mech']['rn'] * n + bc['mech']['rt']
                self.bc_rhs_prev[self.n_vars * id + u_var:self.n_vars * id + u_var + 3] = bc['mech']['rn'] * n + bc['mech']['rt']

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
        self.file_path = mesh_file
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
        self.unstr_discr.physical_tags['matrix'] = [99991]
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

        self.unstr_discr.init_matrix_stiffness({99991: {'E': self.E, 'nu': self.nu}})
        self.unstr_discr.physical_tags['fracture'] = [9991]
        self.unstr_discr.physical_tags['fracture_shape'] = []
        self.unstr_discr.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996]
        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)

        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        self.F = -100.0 # bar * m

        mech_xm = ROLLER
        mech_xp = LOAD(self.F, [0.0, 0.0, 0.0])
        mech_ym = ROLLER
        mech_yp = ROLLER
        mech_zm = ROLLER
        mech_zp = ROLLER

        flow_xm = NO_FLOW
        flow_xp = AQUIFER(self.p_init)
        flow_ym = NO_FLOW
        flow_yp = NO_FLOW
        flow_zm = NO_FLOW
        flow_zp = NO_FLOW

        self.unstr_discr.boundary_conditions[991] = {'flow': flow_xm, 'mech': mech_xm, 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': flow_xp, 'mech': mech_xp, 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': flow_ym, 'mech': mech_ym, 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': flow_yp, 'mech': mech_yp, 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': flow_zm, 'mech': mech_zm, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': flow_zp, 'mech': mech_zp, 'cells': []}
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
            self.pm.perms.append(matrix33(self.permx, self.permy, self.permz))
            self.pm.biots.append(matrix33(self.biot))
            self.pm.stfs.append(Stiffness(self.lam, self.mu))
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
    def terzaghi_two_layers(self, mesh='rect'):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite_two_layers.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge_two_layers.msh'
        self.file_path = mesh_file
        self.unstr_discr = UnstructDiscretizer(permx=1, permy=1, permz=1, frac_aper=0,
                                               mesh_file=mesh_file)

        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3

        self.fluid_compressibility = 1.e-10
        self.fluid_viscosity = 1.0
        self.props = {      99991: { 'h': 0.25, 'E': 10000, 'nu': 0.15, 'b': 0.9, 'poro': 0.15, 'perm': 1 },
                            99992: { 'h': 0.75, 'E': 10000, 'nu': 0.15, 'b': 0.01, 'poro': 0.001, 'perm': 1  }     }
        x = (self.props[99992]['b'] / self.props[99991]['b'] * (3 * (self.props[99991]['b'] - self.props[99991]['poro']) * (1 - self.props[99991]['b']) * (1 - self.props[99991]['nu']) / (1 + self.props[99991]['nu']) + self.props[99991]['b'] ** 2) -
             self.props[99992]['b'] ** 2) / 3 / (self.props[99992]['b'] - self.props[99992]['poro']) / (1 - self.props[99992]['b'])
        nu2 = (1 - x) / (1 + x)
        self.props[99992]['nu'] = nu2
        assert(nu2 < 0.5 and nu2 > 0)

        kd1 = self.props[99991]['E'] / 3 / (1 - 2 * self.props[99991]['nu'])
        self.props[99991]['kd'] = kd1
        self.props[99991]['M'] = 1.0 / ((self.props[99991]['b'] - self.props[99991]['poro']) * (1 - self.props[99991]['b']) / kd1 +
                                        self.props[99991]['poro'] * self.fluid_compressibility)

        kd2 = self.props[99992]['E'] / 3 / (1 - 2 * self.props[99992]['nu'])
        self.props[99992]['kd'] = kd2
        self.props[99992]['M'] = 1.0 / ((self.props[99992]['b'] - self.props[99992]['poro']) * (1 - self.props[99992]['b']) / kd2 +
                                        self.props[99992]['poro'] * self.fluid_compressibility)


        # some numbers for analytics
        for tag, p in self.props.items():
            p['m'] = (1 + p['nu']) * (1 - 2 * p['nu']) / p['E'] / (1 - p['nu'])
            # if tag == 99992:
                # p['kd'] = kd1 * self.props[99991]['b'] * self.props[99991]['m'] / self.props[99992]['b'] / self.props[99992]['m'] / \
                #               (1 + kd1 * self.props[99991]['b'] * self.props[99991]['m'] * (self.props[99991]['b'] - self.props[99992]['b']))
            p['skempton'] = p['b'] * p['m'] * p['M'] / (1 + p['b'] ** 2 * p['m'] * p['M'])
            p['c'] = TC.darcy_constant * p['perm'] / self.fluid_viscosity * p['M'] / (1 + p['b'] ** 2 * p['m'] * p['M'])

        assert( np.fabs(self.props[99991]['skempton'] - self.props[99992]['skempton']) < 1.e-6 )

        self.unstr_discr.init_matrix_stiffness(self.props)
        self.unstr_discr.physical_tags['matrix'] = [99991, 99992]
        self.unstr_discr.physical_tags['fracture'] = [9991]
        self.unstr_discr.physical_tags['fracture_shape'] = []
        self.unstr_discr.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996]
        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)

        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        self.F = -100.0 # bar * m

        mech_xm = ROLLER
        mech_xp = LOAD(self.F, [0.0, 0.0, 0.0])
        mech_ym = ROLLER
        mech_yp = ROLLER
        mech_zm = ROLLER
        mech_zp = ROLLER

        flow_xm = NO_FLOW
        flow_xp = AQUIFER(self.p_init)
        flow_ym = NO_FLOW
        flow_yp = NO_FLOW
        flow_zm = NO_FLOW
        flow_zp = NO_FLOW

        self.unstr_discr.boundary_conditions[991] = {'flow': flow_xm, 'mech': mech_xm, 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': flow_xp, 'mech': mech_xp, 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': flow_ym, 'mech': mech_ym, 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': flow_yp, 'mech': mech_yp, 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': flow_zm, 'mech': mech_zm, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': flow_zp, 'mech': mech_zp, 'cells': []}
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

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
        self.omega = self.approximate_roots_two_layers_terzaghi()
        self.tD = 1.0
        self.pD = 1.0
    def terzaghi_two_layers_no_analytics(self, mesh='rect'):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init = 0.0
        if mesh == 'rect':
            mesh_file = 'meshes/transfinite_two_layers.msh'
        elif mesh == 'wedge':
            mesh_file = 'meshes/wedge_two_layers.msh'
        self.file_path = mesh_file
        self.unstr_discr = UnstructDiscretizer(permx=1, permy=1, permz=1, frac_aper=0,
                                               mesh_file=mesh_file)

        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3

        self.visc = 1#9.81e-2
        self.props = {      99991: { 'h': 0.25, 'E': 10000, 'nu': 0.15, 'b': 0.0, 'poro': 0.0, 'perm': 1e-10 },
                            99992: { 'h': 0.75, 'E': 10000, 'nu': 0.15, 'b': 0.9, 'poro': 0.15, 'perm': 1  }     }

        kd1 = self.props[99991]['E'] / 3 / (1 - 2 * self.props[99991]['nu'])
        self.props[99991]['kd'] = kd1
        #self.props[99991]['M'] = kd1 / (self.props[99991]['b'] - self.props[99991]['poro']) / (1 - self.props[99991]['b'])

        kd2 = self.props[99992]['E'] / 3 / (1 - 2 * self.props[99992]['nu'])
        self.props[99992]['kd'] = kd2
        #self.props[99992]['M'] = kd2 / (self.props[99992]['b'] - self.props[99992]['poro']) / (1 - self.props[99992]['b'])

        self.unstr_discr.init_matrix_stiffness(self.props)
        self.unstr_discr.physical_tags['matrix'] = [99991, 99992]
        self.unstr_discr.physical_tags['fracture'] = [9991]
        self.unstr_discr.physical_tags['fracture_shape'] = []
        self.unstr_discr.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996]
        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)

        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

        self.F = -100.0 # bar * m

        mech_xm = ROLLER
        mech_xp = LOAD(self.F, [0.0, 0.0, 0.0])
        mech_ym = ROLLER
        mech_yp = ROLLER
        mech_zm = ROLLER
        mech_zp = ROLLER

        flow_xm = NO_FLOW
        flow_xp = AQUIFER(self.p_init)
        flow_ym = NO_FLOW
        flow_yp = NO_FLOW
        flow_zm = NO_FLOW
        flow_zp = NO_FLOW

        self.unstr_discr.boundary_conditions[991] = {'flow': flow_xm, 'mech': mech_xm, 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': flow_xp, 'mech': mech_xp, 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': flow_ym, 'mech': mech_ym, 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': flow_yp, 'mech': mech_yp, 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': flow_zm, 'mech': mech_zm, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': flow_zp, 'mech': mech_zp, 'cells': []}
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

    # old discretizer
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
        self.file_path = mesh_file
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
        self.unstr_discr.physical_tags['matrix'] = [99991]
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

        self.unstr_discr.init_matrix_stiffness({99991: {'E': self.E, 'nu': self.nu}})
        self.unstr_discr.physical_tags['fracture'] = [9991]
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

        mech_xm = ROLLER
        mech_xp = FREE
        mech_ym = ROLLER
        mech_yp = STUCK_ROLLER(0.0)
        mech_zm = ROLLER
        mech_zp = ROLLER

        flow_xm = NO_FLOW
        flow_xp = AQUIFER(self.p_init)
        flow_ym = NO_FLOW
        flow_yp = NO_FLOW
        flow_zm = NO_FLOW
        flow_zp = NO_FLOW

        self.unstr_discr.boundary_conditions[991] = {'flow': flow_xm, 'mech': mech_xm, 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': flow_xp, 'mech': mech_xp, 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': flow_ym, 'mech': mech_ym, 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': flow_yp, 'mech': mech_yp, 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': flow_zm, 'mech': mech_zm, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': flow_zp, 'mech': mech_zp, 'cells': []}
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
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
        self.mesh.reverse_and_sort_pm_mech_discretizer()
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
    def write_to_vtk(self, output_directory, ith_step, engine):
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
        Mesh = meshio.read(self.file_path)

        # Allocate empty new cell_data dictionary:
        cell_property = ['p', 'u_x', 'u_y', 'u_z']
        props_num = len(cell_property)
        property_array = np.array(engine.X, copy=False)
        available_matrix_geometries = ['hexahedron', 'wedge', 'tetra']
        available_fracture_geometries = ['quad', 'triangle']

        # if ith_step != 0:
        # fluxes = np.array(physics.engine.fluxes, copy=False)
        # # fluxes_n = np.array(physics.engine.fluxes_n, copy=False)
        # fluxes_biot = np.array(physics.engine.fluxes_biot, copy=False)
        # #vels = self.reconstruct_velocities(fluxes[physics.engine.P_VAR::physics.engine.N_VARS],
        # #                                  fluxes_biot[physics.engine.P_VAR::physics.engine.N_VARS])
        # self.mech_operators.eval_porosities(physics.engine.X, self.mesh.bc)
        # self.mech_operators.eval_stresses(physics.engine.fluxes, physics.engine.fluxes_biot, physics.engine.X,
        #                                   self.mesh.bc, physics.engine.op_vals_arr)
        # else:
        #    self.mech_operators.eval_porosities(physics.engine.X, self.mesh.bc_prev)
        #    self.mech_operators.eval_stresses(physics.engine.X, self.mesh.bc_prev, physics.engine.op_vals_arr)

        # Matrix
        geom_id = 0
        Mesh.cells = []
        cell_data = {}
        for ith_geometry in self.mesh_data.cells_dict.keys():
            if ith_geometry in available_matrix_geometries:
                Mesh.cells.append(self.mesh_data.cells[geom_id])
                # Add matrix data to dictionary:
                for i in range(props_num):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(property_array[i:props_num * self.n_matrix:props_num])

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
    # Analytics
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
        a = np.max(self.mesh_data.points, axis=0)[0]
        b = np.max(self.mesh_data.points, axis=0)[1]

        # Calculate constants
        aa_n = self.approximate_roots()[:, np.newaxis]

        cy0 = (-F * (1 - nu_s)) / (2 * mu_s * a)
        cy1 = F * (1 - nu_u) / (mu_s * a)

        # Calculate exact north boundary condition
        uy_sum = np.sum(
            ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
            * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
            axis=0,
        )

        north_bc = (cy0 + cy1 * uy_sum) * b
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
        mu_f = self.pm.visc
        k_s = self.permx / self.fluid_viscosity
        c_f = TC.darcy_constant * (2 * k_s * (skempton ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / ( 9 * mu_f * (1 - nu_u) * (nu_u - nu_s) )
        a = np.max(self.unstr_discr.mesh_data.points[:, 0])

        if t == 0.0:  # initial condition has its own expression
            p = ((F * skempton * (1 + nu_u)) / (3 * a)) * np.ones(xc.size)
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots()[:, np.newaxis]
            # Exact p
            c0 = (2 * F * skempton * (1 + nu_u)) / (3 * a)
            p_sum_0 = np.sum(
                ((np.sin(aa_n)) / (aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                * (np.cos((aa_n * xc) / a) - np.cos(aa_n))
                * np.exp((-(aa_n**2) * c_f * t) / (a**2)),
                axis=0,
            )
            p = c0 * p_sum_0

        return p
    # Terzaghi
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
    # Two-layer Terzaghi
    def approximate_roots_two_layers_terzaghi(self) -> np.ndarray:
        # Retrieve physical data
        p1 = self.props[99991]
        p2 = self.props[99992]
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
        h1 = self.a * self.props[99991]['h']
        h2 = self.a * self.props[99992]['h']
        c2 = self.props[99992]['c']
        xi = xc - h2
        skempton = self.props[99991]['skempton']

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
        h1 = self.a * self.props[99991]['h']
        h2 = self.a * self.props[99992]['h']
        b1 = self.props[99991]['b']
        b2 = self.props[99992]['b']
        m1 = self.props[99991]['m']
        m2 = self.props[99992]['m']
        c2 = self.props[99992]['c']
        xi = xc - h2
        skempton = self.props[99991]['skempton']

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
