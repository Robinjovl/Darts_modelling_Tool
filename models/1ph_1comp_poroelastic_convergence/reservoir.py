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
from t2 import RhsPoroelastic, RhsThermoporoelastic
from scipy.linalg import null_space
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags


import darts.discretizer as dis
from darts.discretizer import Mesh, Elem, poro_mech_discretizer, THMBoundaryCondition, BoundaryCondition, elem_loc, elem_type, conn_type
from darts.discretizer import vector_matrix33, vector_vector3, matrix, value_vector, index_vector
from darts.discretizer import matrix33 as disc_matrix33
from darts.engines import Stiffness as engine_stiffness
from darts.discretizer import Stiffness as disc_stiffness

# Definitions for the unstructured reservoir class:
class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, discretizer, mode, mesh_file):
        thermoporoelasticity = True if mode == 'thermoporoelastic' else False
        super().__init__(timer, discretizer, thermoporoelasticity)
        # define correspondence between the physical tags in msh file and mesh elements types
        self.domain_tags, self.bnd_tags = set_domain_tags(matrix_tags=[99991],
                    bnd_xm_tag=991, bnd_xp_tag=992,
                    bnd_ym_tag=993, bnd_yp_tag=994,
                    bnd_zm_tag=995, bnd_zp_tag=996)
        self.mesh_filename = mesh_file

        # Specify elastic properties, mesh & boundaries
        self.timer.node["discretization"] = timer_node()
        if self.discretizer_name == 'pm_discretizer':
            self.convergence_study_setup_pm_discretizer()
        elif self.discretizer_name == 'mech_discretizer':
            if self.thermoporoelasticity:
                self.convergence_study_setup_mech_discretizer_thermoporoelasticity()
            else:
                self.convergence_study_setup_mech_discretizer_poroelasticity()

        self.x_new = np.ones((self.n_matrix + self.n_fracs, self.n_vars))
        self.x_new[:, self.u_var] = self.u_init[0]
        self.x_new[:, self.u_var + 1] = self.u_init[1]
        self.x_new[:, self.u_var + 2] = self.u_init[2]
        self.x_new[:, self.p_var] = self.p_init
        if self.thermoporoelasticity:
            self.x_new[:, self.t_var] = self.t_init

        self.biot_mean = 0.0 # TODO: replace biot_arr, kd with matrix_compressibility
        self.kd_cur = 0.0 # TODO: replace biot_arr, kd with matrix_compressibility
        self.t_init = 0.0 # TODO: fix/generalize self.set_pz_bounds()
        self.p_init_copy = np.copy(self.p_init) # TODO: FFFIIIXXX self.pz_bounds()
        self.p_init = 0.0 # TODO: FFFIIIXXX self.pz_bounds()
        if self.discretizer_name == 'pm_discretizer':
            self.unstr_discr.p_ref = self.p_init
            self.unstr_discr.f = self.f_prep
        self.init_reservoir_main()
        self.p_init = np.copy(self.p_init_copy) # TODO: FFFIIIXXX self.pz_bounds()

        self.bc_prev[:] = self.bc_rhs_prev
        self.bc[:] = self.bc_rhs
        self.bc_ref[:] = self.bc_rhs_ref
        self.mesh.pz_bounds.resize(self.n_state * self.n_bounds)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.pz_bounds[:] = self.pz_bounds_rhs
        if self.thermoporoelasticity:
            self.biot_arr[:] = np.tile([self.porosity,0,0,
                                        0,self.porosity,0,
                                        0,0,self.porosity], self.n_matrix + self.n_fracs)
        else:
            self.biot_arr[:] = np.tile([0,0,0,
                                        0,0,0,
                                        0,0,0], self.n_matrix + self.n_fracs)
        if self.thermoporoelasticity:
            self.kd[:] = 1.0
        else:
            self.kd[:] = 1 / self.c / self.porosity
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
            x_an = reference_solution_poroelastic(x)
            self.mech_operators.eval_stresses(engine.fluxes, engine.fluxes_biot, engine.X,
                                              self.mesh.bc, engine.op_vals_arr)
            total_stresses = np.array(self.mech_operators.total_stresses, copy=True).reshape((self.n_matrix, 6))
            effective_stresses = np.array(self.mech_operators.stresses, copy=True).reshape((self.n_matrix, 6))
            darcy_velocities = np.array(self.mech_operators.velocities, copy=True).reshape((self.n_matrix, 3))
        elif self.discretizer_name == 'mech_discretizer':
            # unknowns
            if self.thermoporoelasticity:
                x_an = reference_solution_thermoporoelastic(self.x_all[:, :self.n_matrix])
            else:
                x_an = reference_solution_poroelastic(self.x_all[:, :self.n_matrix])
            # stresses
            engine.eval_stresses_and_velocities()
            total_stresses = np.array(engine.total_stresses, copy=True).reshape((self.n_matrix, 6))
            effective_stresses = np.array(engine.effective_stresses, copy=True).reshape((self.n_matrix, 6))
            darcy_velocities = np.array(engine.darcy_velocities, copy=True).reshape((self.n_matrix, 3))

        dev_u = np.sqrt((vol * ((x_num[self.u_var:self.u_var+self.n_dim] - x_an[:self.n_dim]) ** 2).sum(axis=0)).sum() / total_vol)
        dev_p = np.sqrt((vol * ((x_num[self.p_var] - x_an[self.n_dim]) ** 2)).sum() / total_vol)
        dev_s = np.sqrt((vol * ((total_stresses - self.total_stress_an) ** 2).sum(axis=1)).sum() / total_vol)
        dev_seff = np.sqrt((vol * ((effective_stresses - self.effective_stresses_an) ** 2).sum(axis=1)).sum() / total_vol)
        dev_v = np.sqrt((vol * ((darcy_velocities - self.darcy_velocities_an) ** 2).sum(axis=1)).sum() / total_vol)

        if self.thermoporoelasticity:
            dev_t = np.sqrt((vol * ((x_num[self.t_var] - x_an[self.n_dim + 1]) ** 2)).sum() / total_vol)
            return dev_u, dev_p, dev_s, dev_seff, dev_v, dev_t
        else:
            return dev_u, dev_p, dev_s, dev_seff, dev_v
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
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.pz_bounds[:] = self.pz_bounds_rhs
        #self.init_wells()
    def set_boundary_conditions(self):
        self.boundary_conditions = {}
        self.boundary_conditions[self.bnd_tags['BND_X-']] = {'flow': self.bc_type.AQUIFER(0),
                                                             'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0]),
                                                             'temp': self.bc_type.AQUIFER(0) }
        self.boundary_conditions[self.bnd_tags['BND_X+']] = {'flow': self.bc_type.AQUIFER(0),
                                                             'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0]),
                                                             'temp': self.bc_type.AQUIFER(0) }
        self.boundary_conditions[self.bnd_tags['BND_Y-']] = {'flow': self.bc_type.AQUIFER(0),
                                                             'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0]),
                                                             'temp': self.bc_type.AQUIFER(0) }
        self.boundary_conditions[self.bnd_tags['BND_Y+']] = {'flow': self.bc_type.AQUIFER(0),
                                                             'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0]),
                                                             'temp': self.bc_type.AQUIFER(0) }
        self.boundary_conditions[self.bnd_tags['BND_Z-']] = {'flow': self.bc_type.AQUIFER(0),
                                                             'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0]),
                                                             'temp': self.bc_type.AQUIFER(0) }
        self.boundary_conditions[self.bnd_tags['BND_Z+']] = {'flow': self.bc_type.AQUIFER(0),
                                                             'mech': self.bc_type.STUCK(0.0, [0.0, 0.0, 0.0]),
                                                             'temp': self.bc_type.AQUIFER(0) }
        self.set_boundary_conditions_pm_discretizer()

    # pm_discretizer: poroelastic
    def update_pm_discretizer(self, time):
        # Boundary conditions
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            c = self.unstr_discr.bound_cell_info_dict[bound_id].centroid
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            sol = reference_solution_poroelastic(np.append(c, time))
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
            self.f_prep[4 * cell_id:4 * cell_id + 3] = -np.array(self.r.f_func(cell.centroid[0],
                                                                               cell.centroid[1],
                                                                               cell.centroid[2], time))[:, 0]
            self.f_prep[4 * cell_id + 3] = -self.r.acc_func(cell.centroid[0], cell.centroid[1], cell.centroid[2], time) - \
                                         self.r.flow_func(cell.centroid[0], cell.centroid[1], cell.centroid[2], time)
            self.total_stress_an[cell_id] = self.r.total_stress_func(cell.centroid[0], cell.centroid[1], cell.centroid[2], time)[:,0]
            self.effective_stresses_an[cell_id] = self.r.effective_stress_func(cell.centroid[0], cell.centroid[1], cell.centroid[2], time)[:,0]
            self.darcy_velocities_an[cell_id] = self.r.darcy_velocity_func(cell.centroid[0], cell.centroid[1], cell.centroid[2], time)[:,0]
    def convergence_study_setup_pm_discretizer(self):
        self.porosity = 0.1
        self.unstr_discr = UnstructDiscretizer(permx=1, permy=1, permz=1, frac_aper=1.E-4,
                                               mesh_file=self.mesh_filename)
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
        self.fluid_density = 1.0
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
            sol = reference_solution_poroelastic(np.append(cell.centroid, time))
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
            sol = reference_solution_poroelastic(np.append(c, time))
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
        self.r = RhsPoroelastic(stf=stf, biot=biot, perm=perm,
                                visc=self.fluid_viscosity, grav=self.grav, rho_f=self.fluid_density,
                                comp_s=self.c * self.porosity, poro0=self.porosity)
        self.f_prep = np.zeros((self.unstr_discr.mat_cells_tot, 4))
        for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items():
            self.f_prep[cell_id, :3] = -np.array(self.r.f_func(cell.centroid[0],
                                                               cell.centroid[1],
                                                               cell.centroid[2],
                                                               time))[:, 0]
            self.f_prep[cell_id, 3] = -self.r.acc_func(cell.centroid[0], cell.centroid[1], cell.centroid[2], time) - \
                                        self.r.flow_func(cell.centroid[0], cell.centroid[1], cell.centroid[2], time)
        self.f_prep = self.f_prep.flatten()

        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_bounds = self.unstr_discr.bound_cells_tot
        self.total_stress_an = np.zeros((self.n_matrix, 6))
        self.effective_stresses_an = np.zeros((self.n_matrix, 6))
        self.darcy_velocities_an = np.zeros((self.n_matrix, 3))

    # mech_discretizer: poroelastic
    def update_mech_discretizer_poroelasticity(self, time):
        # Boundary conditions
        self.x_all[3, :] = time
        sol = reference_solution_poroelastic(self.x_all[:, self.n_matrix + self.n_fracs:])
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            for id in ids:
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = sol[:3, id]
                self.bc_rhs[self.n_vars * id + self.p_var] = sol[3, id]
                self.pz_bounds_rhs[self.n_state * id + self.p_var] = sol[3, id]

        for cell_id in range(self.n_matrix):
            c = self.centroids[cell_id]
            self.f_prep[self.n_vars * cell_id + self.u_var:self.n_vars * cell_id + self.u_var + self.n_dim] = \
                -np.array(self.r.f_func(c.values[0], c.values[1], c.values[2], time))[:, 0]
            self.f_prep[self.n_vars * cell_id + self.p_var] = \
                -self.r.acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.flow_func(c.values[0], c.values[1], c.values[2], time)
            self.total_stress_an[cell_id] = self.r.total_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.effective_stresses_an[cell_id] = self.r.effective_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.darcy_velocities_an[cell_id] = self.r.darcy_velocity_func(c.values[0], c.values[1], c.values[2], time)[:,0]
    def convergence_study_setup_mech_discretizer_poroelasticity(self):
        self.mesh_data = meshio.read(self.mesh_filename)

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
        self.fluid_density = 978.0
        self.fluid_compressibility = 0.0
        self.fluid_viscosity = 1e-2

        self.set_boundary_conditions()
        self.init_mech_discretizer()
        self.discr.grav_vec = matrix([0.0, 0.0, self.grav], 1, 3)

        # bulk properties
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            self.discr.perms.append(disc_matrix33(perm))
            self.discr.biots.append(disc_matrix33(biot))
            self.discr.stfs.append(disc_stiffness(stf))

        # mapping boundary connections
        id_sorted = np.argsort(self.adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = self.adj_matrix[id_sorted]
        time = 0.0
        self.x_all = np.array([np.array([c.values[0], c.values[1], c.values[2], time]) for c in self.centroids]).T
        sol = reference_solution_poroelastic(self.x_all)
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
        self.pz_bounds_rhs = np.zeros(self.n_state * self.n_bounds)

        sol = reference_solution_poroelastic(self.x_all[:, self.n_matrix + self.n_fracs:])
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            ap[ids] = bc['flow']['a']
            bp[ids] = bc['flow']['b']
            amn[ids] = bc['mech']['an']
            bmn[ids] = bc['mech']['bn']
            amt[ids] = bc['mech']['at']
            bmt[ids] = bc['mech']['bt']

            for id in ids:
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = sol[:3, id]
                self.bc_rhs[self.n_vars * id + self.p_var] = sol[3, id]
                self.pz_bounds_rhs[self.n_state * id + self.p_var] = sol[3, id]

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
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

        # RHS term
        self.c = 1.4503768e-05
        self.r = RhsPoroelastic(stf=stf, biot=biot, perm=perm,
                                visc=self.fluid_viscosity, grav=self.grav, rho_f=self.fluid_density,
                                comp_s=self.c * self.porosity, poro0=self.porosity)
        self.f_prep = np.zeros(self.n_matrix * self.n_vars)
        self.total_stress_an = np.zeros((self.n_matrix, 6))
        self.effective_stresses_an = np.zeros((self.n_matrix, 6))
        self.darcy_velocities_an = np.zeros((self.n_matrix, 3))
        for cell_id in range(self.n_matrix):
            c = self.centroids[cell_id]
            self.f_prep[self.n_vars * cell_id + self.u_var:self.n_vars * cell_id + self.u_var + self.n_dim] = \
                -np.array(self.r.f_func(c.values[0], c.values[1], c.values[2], time))[:, 0]
            self.f_prep[self.n_vars * cell_id + self.p_var] = \
                -self.r.acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.flow_func(c.values[0], c.values[1], c.values[2], time)

    # mech_discretizer: thermoporoelastic
    def update_mech_discretizer_thermoporoelasticity(self, time):
        # Boundary conditions
        self.x_all[3, :] = time
        sol = reference_solution_thermoporoelastic(self.x_all[:, self.n_matrix + self.n_fracs:])
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            for id in ids:
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = sol[:3, id]
                self.bc_rhs[self.n_vars * id + self.p_var] = sol[3, id]
                self.bc_rhs[self.n_vars * id + self.t_var] = sol[4, id]
                self.pz_bounds_rhs[self.n_state * id + self.p_var] = sol[3, id]
                self.pz_bounds_rhs[self.n_state * id + self.t_var] = sol[4, id]

        for cell_id in range(self.n_matrix):
            c = self.centroids[cell_id]
            self.f_prep[self.n_vars * cell_id + self.u_var:self.n_vars * cell_id + self.u_var + self.n_dim] = \
                -np.array(self.r.f_func(c.values[0], c.values[1], c.values[2], time))[:, 0]
            self.f_prep[self.n_vars * cell_id + self.t_var] = \
                -self.r.energy_acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.energy_flow_func(c.values[0], c.values[1], c.values[2], time)
            self.f_prep[self.n_vars * cell_id + self.p_var] = \
                -self.r.acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.flow_func(c.values[0], c.values[1], c.values[2], time)
            self.total_stress_an[cell_id] = self.r.total_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.effective_stresses_an[cell_id] = self.r.effective_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.darcy_velocities_an[cell_id] = self.r.darcy_velocity_func(c.values[0], c.values[1], c.values[2], time)[:,0]
    def convergence_study_setup_mech_discretizer_thermoporoelasticity(self):
        self.mesh_data = meshio.read(self.mesh_filename)

        # params
        self.porosity = 0.1
        perm = [1.5,    0.5,    0.35,
                0.5,    1.5,    0.45,
                0.35,   0.45,   1.5]
        biot = [1.5,    0.1,    0.5,
                0.1,    1.5,    0.15,
                0.5,    0.15,   1.5]
        heat_cond = 1.e+6 * np.array([1.5,    0.1,    0.5,
                                        0.1,    1.5,    0.15,
                                        0.5,    0.15,   1.5])
        therm_expn = [1.5,    0.5,    0.35,
                      0.5,    1.5,    0.45,
                      0.35,   0.45,   1.5]
        stf =  [1.323, 0.0726, 0.263, 0.108, -0.08, -0.239,
                0.0726, 1.276, -0.318, 0.383, 0.108, 0.501,
                0.263, -0.318, 0.943, -0.183, 0.146, 0.182,
                0.108, 0.383, -0.183, 1.517, -0.0127, -0.304,
                -0.08, 0.108, 0.146, -0.0127, 1.209, -0.326,
                -0.239, 0.501, 0.182, -0.304, -0.326, 1.373]
        self.th_expn_poro = self.th_expn_coef = self.kd_cur = 0.0 # TODO: why parent ask self.th_expn_coef and self.kd_cur?
        self.grav = 9.81e-2
        self.fluid_density = 978.0
        self.fluid_compressibility = 0.0
        self.fluid_viscosity = 1e-2
        self.heat_capacity = 1.0#2200.0 * 1000.0

        self.set_boundary_conditions()
        self.init_mech_discretizer()
        self.discr.grav_vec = matrix([0.0, 0.0, self.grav], 1, 3)

        # bulk properties
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            self.discr.perms.append(disc_matrix33(perm))
            self.discr.biots.append(disc_matrix33(biot))
            self.discr.stfs.append(disc_stiffness(stf))
            self.discr.heat_conductions.append(disc_matrix33(heat_cond))
            self.discr.thermal_expansions.append(disc_matrix33(therm_expn))

        # mapping boundary connections
        id_sorted = np.argsort(self.adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = self.adj_matrix[id_sorted]
        time = 0.0
        self.x_all = np.array([np.array([c.values[0], c.values[1], c.values[2], time]) for c in self.centroids]).T
        sol = reference_solution_thermoporoelastic(self.x_all)
        self.u_init = sol[:self.n_dim, :self.n_matrix]
        self.p_init = sol[self.n_dim, :self.n_matrix]
        self.t_init = sol[self.n_dim + 1, :self.n_matrix]

        # assign boundary conditions element-by-element
        ap = np.ones(self.n_bounds)
        bp = np.zeros(self.n_bounds)
        amn = np.ones(self.n_bounds)
        bmn = np.zeros(self.n_bounds)
        amt = np.ones(self.n_bounds)
        bmt = np.zeros(self.n_bounds)
        at = np.ones(self.n_bounds)
        bt = np.zeros(self.n_bounds)
        self.bc_rhs = np.zeros(self.n_vars * self.n_bounds)
        self.bc_rhs_prev = np.zeros(self.n_vars * self.n_bounds)
        self.bc_rhs_ref = np.zeros(self.n_vars * self.n_bounds)
        self.pz_bounds_rhs = np.zeros(self.n_state * self.n_bounds)

        sol = reference_solution_thermoporoelastic(self.x_all[:, self.n_matrix + self.n_fracs:])
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
            for id in ids:
                self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = sol[:3, id]
                self.bc_rhs[self.n_vars * id + self.p_var] = sol[3, id]
                self.bc_rhs[self.n_vars * id + self.t_var] = sol[4, id]
                self.pz_bounds_rhs[self.n_state * id + self.p_var] = sol[3, id]
                self.pz_bounds_rhs[self.n_state * id + self.t_var] = sol[4, id]

        # mechanics
        self.cpp_bc = THMBoundaryCondition()
        self.cpp_bc.flow.a = value_vector(ap)
        self.cpp_bc.flow.b = value_vector(bp)
        self.cpp_bc.mech_normal.a = value_vector(amn)
        self.cpp_bc.mech_normal.b = value_vector(bmn)
        self.cpp_bc.mech_tangen.a = value_vector(amt)
        self.cpp_bc.mech_tangen.b = value_vector(bmt)
        self.cpp_bc.thermal.a = value_vector(at)
        self.cpp_bc.thermal.b = value_vector(bt)
        # flow
        self.cpp_flow = BoundaryCondition() # TODO: why do we duplicate them?
        self.cpp_flow.a = value_vector(ap)
        self.cpp_flow.b = value_vector(bp)
        # heat
        self.cpp_heat = BoundaryCondition() # TODO: why do we duplicate them?
        self.cpp_heat.a = value_vector(at)
        self.cpp_heat.b = value_vector(bt)

        # perform discretization
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_flow, self.cpp_heat)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

        # RHS term
        self.r = RhsThermoporoelastic(stf=stf, biot=biot, perm=perm, th_expn=therm_expn, heat_cond=heat_cond,
                                visc=self.fluid_viscosity, grav=self.grav, rho_f=self.fluid_density,
                                comp_s=0.0, poro0=self.porosity,
                                th_expn_poro=self.th_expn_poro, heat_capacity=self.heat_capacity)
        self.f_prep = np.zeros(self.n_matrix * self.n_vars)
        self.total_stress_an = np.zeros((self.n_matrix, 6))
        self.effective_stresses_an = np.zeros((self.n_matrix, 6))
        self.darcy_velocities_an = np.zeros((self.n_matrix, 3))
        for cell_id in range(self.n_matrix):
            c = self.centroids[cell_id]
            self.f_prep[self.n_vars * cell_id + self.u_var:self.n_vars * cell_id + self.u_var + self.n_dim] = \
                -np.array(self.r.f_func(c.values[0], c.values[1], c.values[2], time))[:, 0]
            self.f_prep[self.n_vars * cell_id + self.t_var] = \
                -self.r.energy_acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.energy_flow_func(c.values[0], c.values[1], c.values[2], time)
            self.f_prep[self.n_vars * cell_id + self.p_var] = \
                -self.r.acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.flow_func(c.values[0], c.values[1], c.values[2], time)

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
    def write_data_field(self, filename, u, s = None):
        r = np.array([cell.centroid for cell in self.unstr_discr.mat_cell_info_dict.values()])
        inds = list(np.arange(len(r)))
        inds.sort(key=lambda id: r[id][0] + 1000 * r[id][1] + 1.E+6 * r[id][2])
        if s is self.write_data_field.__defaults__[0]:
            np.savetxt(filename, np.c_[r[inds,0], r[inds,1], r[inds, 2], u[inds,0], u[inds,1], u[inds,2]])
        else:
            np.savetxt(filename, np.c_[r[inds, 0], r[inds, 1], r[inds, 2], u[inds, 0], u[inds, 1], u[inds,2],
                s[inds, 0], s[inds, 1], s[inds, 2], s[inds, 3], s[inds, 4], s[inds, 5]])
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
                sol_an = reference_solution_poroelastic(x_an.T)
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
                sol_an = reference_solution_poroelastic(x_an.T)
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

def reference_solution_poroelastic(x):
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

def reference_solution_thermoporoelastic(x):
    if len(x.shape) == 1:
        sol = np.zeros(5)
    else:
        sol = np.zeros((5, x.shape[1]))
    sol[:3] = (x[:3] - 0.5) ** 2
    sol[0] -= x[1] + x[2]
    sol[1] -= x[0] + x[2]
    sol[2] -= x[0] + x[1]
    sol[:3] *= (1 + x[3] ** 2)
    sol[3] = 3.0 - x[0] - x[1] - x[2]
    sol[4] = np.sin((1 - x[0]) * (1 - x[1]) * (1 - x[2])) / 2 / np.sin(1) + \
           ((1 - x[0]) ** 3) * ((1 - x[1]) ** 2) * (1 - x[2]) * (1 + x[3] ** 2) / 2
    return sol
