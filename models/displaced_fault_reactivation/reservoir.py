from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, contact, contact_vector, vector_matrix
from darts.engines import matrix33, matrix, pm_discretizer, Face, vector_face_vector, face_vector, vector_matrix33, Stiffness, stf_vector, critical_stress, scheme_type
import numpy as np
from math import inf, pi
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.mesh.geometrymodule import FType
from darts.engines import timer_node
from itertools import compress
from darts.reservoirs.unstruct_reservoir_mech import get_rock_compressibility
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags, get_lambda_mu, get_bulk_modulus, get_biot_modulus
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
import meshio
import os
import pandas as pd
import pickle
import xml.dom.minidom
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from matplotlib import rcParams
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
rcParams["text.usetex"]=False
# rcParams["font.sans-serif"] = ["Liberation Sans"]
# rcParams["font.serif"] = ["Liberation Serif"]
from utils import dict_hash, hash_array

class UnstructReservoir(UnstructReservoirMech):
    def __init__(self, timer, fluid_density, rock_density, mesh_file, cache_discretizer: bool = True):
        super().__init__(timer, discretizer='pm_discretizer', fluid_vars=['p'], thermoporoelasticity=False)
        self.timer = timer
        self.rho_s = rock_density
        self.rho_f = fluid_density
        self.mesh_file = mesh_file
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()

        self.cache_discretizer = cache_discretizer
        self.cache_filename = 'cached_preprocessing.pkl'
        cached_var_names = ['self.unstr_discr',
                            'self.pm',
                            'self.porosity',
                            'self.frac_apers',
                            'self.bc_rhs_prev',
                            'self.bc_rhs',
                            'self.bc_rhs_ref',
                            'self.biot',
                            'self.lam',
                            'self.mu',
                            'self.permx',
                            'self.permy',
                            'self.permz',
                            'self.ref_contact_cells',
                            'self.contacts',
                            'self.mesh.fault_normals',
                            'self.p_init',
                            'self.u_init',
                            'self.a',
                            'self.b']

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()

        if self.cache_discretizer:
            # flag checking if update of saved cache needed
            save_cache_discretizer = False
            if os.path.exists(self.cache_filename):
                with open(self.cache_filename, "rb") as fp:
                    cached_data = pickle.load(fp)
                    # check a hash is the same
                    cached_data_no_hash = cached_data.copy()
                    cached_data_no_hash.pop('hash')
                    hash = dict_hash(cached_data_no_hash)
                    if cached_data['hash'] != hash:
                        print('geometry was changed, ', self.cache_filename, 'will be updated')
                        print('current hash=', hash, 'cached hash=', cached_data['hash'])
                        save_cache_discretizer = True
            else:
                save_cache_discretizer = True

            # update needed, re-do discretization
            if save_cache_discretizer:
                self.reservoir_depletion()

                self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot,
                             index_vector(self.ref_contact_cells))
                dt = 0
                self.pm.reconstruct_gradients_per_cell(dt)
                self.pm.calc_all_fluxes_once(dt)

                cached_data = {var_name: eval(var_name, {'self': self}) for var_name in cached_var_names}

                hash = dict_hash(cached_data)
                cached_data.update({'hash': hash})
                print('saving cache, hash=', hash)

                with open(self.cache_filename, "wb") as fp:
                    pickle.dump(cached_data, fp, 4)

            # update not needed, just load and use
            else:
                print('discretizer cache is used')
                for var_name, var in cached_data.items():
                    if var_name == 'self.rho_f':
                        assert var == self.rho_f
                    elif var_name == 'self.rho_s':
                        assert var == self.rho_s
                    exec(var_name + " = var")

                self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot,
                             index_vector(self.ref_contact_cells))
        else:
            self.reservoir_depletion()

            # initialize and run discretizer
            dt = 0
            self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot,
                         index_vector(self.ref_contact_cells))
            self.pm.reconstruct_gradients_per_cell(dt)
            self.pm.calc_all_fluxes_once(dt)

        # check sparsity of gradients
        # for cell_id in range(self.unstr_discr.mat_cells_tot):
        #     st, vals = self.pm.get_gradient(cell_id)
        #     st = np.array(st, dtype=np.intp)
        #     vals = np.array(vals).reshape(12, 4 * st.size)
        #     #assert((np.abs(vals[:9,3::4]) < 1.E-10).all())
        #     assert((np.abs(vals[9:12,0::4]) < 1.E-6).all())
        #     assert((np.abs(vals[9:12,1::4]) < 1.E-6).all())
        #     assert((np.abs(vals[9:12,2::4]) < 1.E-6).all())

        # check sparsity of coupled multi-point approximation
        # for i, cell_m in enumerate(self.pm.cell_m):
        #     cell_p = self.pm.cell_p[i]
        #     for k in range(self.pm.offset[i], self.pm.offset[i+1]):
        #         id = self.pm.stencil[k]
        #         tran = np.array(self.pm.tran[16*k:16*(k+1)]).reshape(4,4)
        #         tran_biot = np.array(self.pm.tran_biot[16*k:16*(k + 1)]).reshape(4,4)
        #         assert((tran[3,:3] == 0.0).all()) # no displacements in flow
                #assert((tran_biot[3,:] == 0.0).all()) # no displacements in flow
                #assert(tran_biot[3] == 0.0) # no pressures in volumetric strain
                #if id != cell_m and id != cell_p:
                #    assert(tran[3,3] == 0.0)    # TPFA

        self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
                          self.pm.tran_biot, self.pm.rhs_biot, self.unstr_discr.mat_cells_tot, \
                          self.unstr_discr.bound_faces_tot + self.unstr_discr.frac_bound_faces_tot, self.unstr_discr.frac_cells_tot)
        # self.write_pm_conn_to_file(t_step=0)
        self.timer.node["discretization"].stop()

        self.unstr_discr.store_volume_all_cells()
        self.unstr_discr.store_depth_all_cells()

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.depth = np.array(self.mesh.depth, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.bc = np.array(self.mesh.bc, copy=False)
        self.bc_prev = np.array(self.mesh.bc_prev, copy=False)
        self.bc_ref = np.array(self.mesh.bc_ref, copy=False)
        self.mesh.f.resize(4 * (self.unstr_discr.frac_cells_tot + self.unstr_discr.mat_cells_tot))
        self.f = np.array(self.mesh.f, copy=False)
        self.rock_compressibility = np.asarray(self.mesh.rock_compressibility)
        self.mesh.pz_bounds.resize(self.unstr_discr.bound_faces_tot + self.unstr_discr.frac_bound_faces_tot)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.p_ref = np.array(self.mesh.ref_pressure, copy=False)

        # Since we use copy==False above, we have to store the values by using the Python slicing option, if we don't
        # do this we will overwrite the variable, e.g. self.poro = poro --> overwrite self.poro with the variable poro
        # instead of storing the variable poro in self.mesh.poro (therefore "numpy array wrapped around mesh data!!!):
        self.poro[:] = self.porosity
        self.depth[:] = self.unstr_discr.depth_all_cells#[:self.unstr_discr.frac_cells_tot + self.unstr_discr.mat_cells_tot]
        self.volume[:self.unstr_discr.mat_cells_tot] = self.unstr_discr.volume_all_cells[self.unstr_discr.frac_cells_tot:]
        for i in range(self.unstr_discr.mat_cells_tot, self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
            self.volume[i] = self.unstr_discr.faces[i][4].area * self.frac_apers[i-self.unstr_discr.mat_cells_tot]
        self.bc_prev[:4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs_prev
        self.bc[:4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs
        self.bc_ref[:4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs_ref
        self.rock_compressibility[:] = get_rock_compressibility(kd=self.lam + 2 * self.mu / 3,
                                                             biot=self.biot, poro0=self.porosity)
        self.pz_bounds[:self.unstr_discr.bound_faces_tot] = self.unstr_discr.pz_bounds
        self.p_ref[:] = self.unstr_discr.p_ref
        self.f[:] = self.unstr_discr.f
        # Create empty list of wells:
        self.wells = []

    def init_reservoir(self, verbose):
        pass
    def set_wells(self, verbose):
        pass

    def update(self, dt, time):
        # update local array
        #if time > dt:
        self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc
    def update_trans(self, dt, x):
        #self.pm.x_prev = value_vector(np.concatenate((x, self.bc_rhs_prev)))
        #self.pm.reconstruct_gradients_per_cell(dt)
        #self.pm.calc_all_fluxes(dt)
        #self.write_pm_conn_to_file(t_step=t_step)
        #self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
        #                  self.unstr_discr.mat_cells_tot, self.unstr_discr.bound_cells_tot, 0)

        # update transient sources / sinks
        self.f[:] = self.unstr_discr.f
        # update boundaries at n+1 / n timesteps
        self.bc[:4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs
        self.bc_prev[:4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs_prev

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
    def initial_stage(self):
        n_dim = 3
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init0 = 350.0
        self.porosity = 0.15
        self.permx = self.permy = self.permz = 100.0
        mesh_file = 'meshes/new_setup_three_point_stick1.msh'
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
        #lam = 1.0 * 10000  # in bar
        #mu = 1.0 * 10000
        #nu = lam / 2 / (lam + mu)
        #E = lam * (1 + nu) * (1 - 2 * nu) / nu

        self.mu = 65000 # in bars
        nu = 0.15
        E = 2 * self.mu * (1 + nu)
        self.lam = E * nu / (1 + nu) / (1 - 2 * nu)
        self.mu = E / 2 / (1 + nu)

        self.unstr_discr.init_matrix_stiffness({99991: {'E': E, 'nu': nu}, 99992: {'E': E, 'nu': nu}})
        self.unstr_discr.physical_tags['matrix'] = [99991, 99992]
        self.unstr_discr.physical_tags['fracture_shape'] = [1]
        self.unstr_discr.physical_tags['fracture'] = [9991]
        self.unstr_discr.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996, 998, 999]
        self.unstr_discr.physical_tags['output'] = []
        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)

        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}
        STUCK_T_LOAD_N = lambda Fn, ut: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}

        gravity = 9.81
        self.rho_s = 2650.0
        self.rho_f = 1020.0
        self.rho_total = (1 - self.porosity) * self.rho_s + self.porosity * self.rho_f
        #assert(np.fabs(self.fv - 295.0) < 1.0)
        self.K0 = 0.5
        self.biot = 0.9
        self.H = 4500.0

        self.sigma_yy = lambda y: self.rho_total * gravity * (y - 3500.0) / 1.E+5
        self.p0 = lambda y: self.p_init0 - self.rho_f * gravity * y / 1.E+5
        self.fh = lambda y: -self.K0 * (self.sigma_yy(y) + self.biot * self.p0(y)) + self.biot * self.p0(y)
        self.eps_yy = lambda y: (1 - 2 * nu) / 2 / self.mu / (1 - nu) * (self.sigma_yy(y) + self.biot * self.p0(y))

        self.unstr_discr.boundary_conditions[991] = {'flow': NO_FLOW,           'mech': LOAD(-0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': NO_FLOW,           'mech': LOAD(-0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': NO_FLOW,           'mech': LOAD(self.sigma_yy(-self.H / 2), [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': NO_FLOW,           'mech': LOAD(self.sigma_yy(self.H / 2), [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': NO_FLOW,           'mech': ROLLER, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': NO_FLOW,           'mech': ROLLER, 'cells': []}
        self.unstr_discr.boundary_conditions[998] = {'flow': NO_FLOW,           'mech': STUCK_T_LOAD_N(self.sigma_yy(-self.H / 2), [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[999] = {'flow': NO_FLOW,           'mech': STUCK_T_LOAD_N(0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[1] =   {'flow': {'a': 0.0, 'b': 1.0, 'r': 0.0},
                                                        'mech': {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 1.0, 'bt': 0.0, 'rt': np.array([0.0, 0.0, 0.0])}, 'cells': [] }
        self.unstr_discr.fracture_aperture = 1
        self.unstr_discr.load_mesh_with_bounds()
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.pm.grav = matrix([0.0, 0.0, 0.0], 1, 3)
        self.pm.visc = 1#9.81e-2
        self.unstr_discr.f = np.zeros(4 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
        self.p_init = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
        self.unstr_discr.p_ref = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
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
            self.p_init[cell_id] = self.p0(cell.centroid[1])

            self.pm.cell_centers.append(matrix(list(cell.centroid), cell.centroid.size, 1))
            if cell.prop_id == 99991:
                self.pm.perms.append(matrix33(self.permx, self.permy, self.permz))
            else:
                self.pm.perms.append(matrix33(100.0, 100.0, 100.0))
            self.pm.biots.append(matrix33(self.biot))

            E = self.unstr_discr.E[cell.prop_id]
            nu = self.unstr_discr.nu[cell.prop_id]
            lam = E * nu / (1 + nu) / (1 - 2 * nu)
            mu = E / (1 + nu) / 2
            self.pm.stfs.append(Stiffness(lam, mu))

            self.unstr_discr.f[4 * cell_id + 1] = self.rho_total * gravity / 1.E+5

        # fracture
        self.frac_aper = 1.E-5
        self.frac_apers = self.frac_aper * np.ones(self.unstr_discr.frac_cells_tot)
        for frac_id in range(self.unstr_discr.mat_cells_tot,
                             self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
            frac = self.unstr_discr.frac_cell_info_dict[frac_id]
            faces = self.unstr_discr.faces[frac_id]
            self.pm.cell_centers.append(matrix(list(frac.centroid), frac.centroid.size, 1))
            self.pm.frac_apers.append(self.frac_apers[frac_id - self.unstr_discr.mat_cells_tot])
            fs = face_vector()
            for face_id, face in faces.items():
                face = faces[face_id]
                fs.append(Face(face.type.value, face.cell_id1, face.cell_id2,
                               face.face_id1, face.face_id2,
                               face.area, list(face.n), list(face.centroid)))
            self.pm.faces.append(fs)

            face1 = faces[4]
            face2 = faces[5]
            self.mesh.fault_normals.append(face1.n[0])
            self.mesh.fault_normals.append(face1.n[1])
            self.mesh.fault_normals.append(face1.n[2])
            # Local basis
            S = np.zeros((n_dim, n_dim))
            S[:n_dim - 1] = null_space(np.array([face1.n])).T
            S[n_dim - 1] = face1.n
            Sinv = np.linalg.inv(S)
            K = np.zeros((n_dim, n_dim))
            K[0, 0] = K[1, 1] = self.permx
            K[n_dim - 1, n_dim - 1] = self.permx
            K = Sinv.dot(K).dot(S)
            self.pm.perms.append(matrix33(list(K.flatten())))

            self.p_init[frac_id] = self.p0(frac.centroid[1])
        # contact
        self.contacts = contact_vector()
        for tag in self.unstr_discr.physical_tags['fracture']:
            con = contact()
            con.f_scale = 1.E+4
            cell_ids = [cell_id for cell_id, cell in self.unstr_discr.frac_cell_info_dict.items() if
                        cell.prop_id == tag]
            fric_coef = 0.7 * np.ones(len(cell_ids))
            con.init_geometry(tag, self.pm, self.mesh, index_vector(cell_ids))
            con.init_friction(value_vector(fric_coef))
            self.contacts.append(con)
        self.bc_rhs_ref = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.bc_rhs_prev = np.zeros(4 * len(self.unstr_discr.bound_cell_info_dict))
        self.unstr_discr.pz_bounds = np.zeros(self.unstr_discr.bound_cells_tot)
        self.unstr_discr.pz_bounds[:] = self.p_init0
        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
            b_cell = self.unstr_discr.bound_cell_info_dict[bound_id]
            prop_id = b_cell.prop_id
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            mech = self.unstr_discr.boundary_conditions[prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            if (prop_id == 991 or prop_id == 992 or prop_id == 999):
                mech['rn'] = -self.fh(b_cell.centroid[1])
                #mech['rn'] = -np.interp(b_cell.centroid[1], self.ux_pt, self.ux)
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
            self.bc_rhs[4 * bound_id + 3] = flow['r']
            self.bc_rhs_prev[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
            self.bc_rhs_ref[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_ref[4 * bound_id + 3] = flow['r']

        for bound_id in range(len(self.unstr_discr.bound_cell_info_dict), self.unstr_discr.bound_cell_count):
            mech = self.unstr_discr.boundary_conditions[self.unstr_discr.frac_bound_cell_info_dict[bound_id].prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[self.unstr_discr.frac_bound_cell_info_dict[bound_id].prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
        self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc
    def reservoir_depletion(self):
        n_dim = 3
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init0 = 350.0
        self.porosity = 0.16 #0.15
        self.permx = self.permy = self.permz = 100.0
        physical_tags = {}
        physical_tags['matrix'] = [99991, 99992, 99993]
        physical_tags['fracture_boundary'] = [1, 2]
        physical_tags['fracture'] = [9991]
        physical_tags['output'] = []
        physical_tags['boundary'] = [991, 981, 992, 982, 993, 994, 995, 996, 998]
        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=physical_tags)
        #TODO can remove this:
        self.unstr_discr.eps_t = 1.E+0
        self.unstr_discr.eps_n = 1.E+0
        self.unstr_discr.mu = 3.2
        self.unstr_discr.P12 = 0
        self.unstr_discr.Prol = 1
        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3

        self.mu = 65000 # in bars
        nu = 0.15
        E = 2 * self.mu * (1 + nu)
        self.lam = E * nu / (1 + nu) / (1 - 2 * nu)
        self.mu = E / 2 / (1 + nu)

        self.unstr_discr.init_matrix_stiffness({99991: {'E': E, 'nu': nu}, 99992: {'E': E, 'nu': nu}, 99993: {'E': E, 'nu': nu}})
        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)

        NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        ROLLER =    {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        FREE =      {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}
        STUCK_T_LOAD_N = lambda Fn, ut: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}

        gravity = 9.81
        self.rho_s = 2650.0
        self.rho_f = 1020.0
        self.rho_total = (1 - self.porosity) * self.rho_s + self.porosity * self.rho_f
        #assert(np.fabs(self.fv - 295.0) < 1.0)
        self.K0 = 0.5
        self.biot = 0.9
        self.H = 4500.0

        self.sigma_yy = lambda y: self.rho_total * gravity * (y - 3500.0) / 1.E+5
        self.p0 = lambda y: self.p_init0 - self.rho_f * gravity * y / 1.E+5
        self.fh = lambda y: -self.K0 * (self.sigma_yy(y) + self.biot * self.p0(y)) + self.biot * self.p0(y)
        self.eps_yy = lambda y: (1 - 2 * nu) / 2 / self.mu / (1 - nu) * (self.sigma_yy(y) + self.biot * self.p0(y))

        # data991 = pd.read_csv('sol_poromechanics/stage1_ddouble_width/xm.csv', delimiter=',')
        data991 = pd.read_csv('xm.csv', delimiter=',')
        self.pt991 = np.array(data991['Points:1'],  dtype=np.float64)
        self.ux991 = np.array(data991['u_x'],       dtype=np.float64)
        #self.uy991 = np.array(data991['u_y'],       dtype=np.float64)

        # data992 = pd.read_csv('sol_poromechanics/stage1_ddouble_width/xp.csv', delimiter=',')
        data992 = pd.read_csv('xp.csv', delimiter=',')
        self.pt992 = np.array(data992['Points:1'],  dtype=np.float64)
        self.ux992 = np.array(data992['u_x'],       dtype=np.float64)
        #self.uy992 = np.array(data992['u_y'],       dtype=np.float64)

        # data993 = pd.read_csv('sol_poromechanics/stage1_ddouble_width/ym.csv', delimiter=',')
        data993 = pd.read_csv('ym.csv', delimiter=',')
        self.pt993 = np.array(data993['Points:0'], dtype=np.float64)
        #self.ux993 = np.array(data993['u_x'], dtype=np.float64)
        self.uy993 = np.array(data993['u_y'], dtype=np.float64)

        # data994 = pd.read_csv('sol_poromechanics/stage1_ddouble_width/yp.csv', delimiter=',')
        data994 = pd.read_csv('yp.csv', delimiter=',')
        self.pt994 = np.array(data994['Points:0'], dtype=np.float64)
        #self.ux994 = np.array(data994['u_x'], dtype=np.float64)
        self.uy994 = np.array(data994['u_y'], dtype=np.float64)

        self.unstr_discr.boundary_conditions[991] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[981] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[982] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': NO_FLOW,               'mech': STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': NO_FLOW,               'mech': ROLLER, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': NO_FLOW,               'mech': ROLLER, 'cells': []}
        self.unstr_discr.boundary_conditions[1] =   {'flow': {'a': 0.0, 'b': 1.0, 'r': 0.0},
                                                        'mech': {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 1.0, 'bt': 0.0, 'rt': np.array([0.0, 0.0, 0.0])}, 'cells': [] }
        self.unstr_discr.boundary_conditions[2] =   {'flow': {'a': 0.0, 'b': 1.0, 'r': 0.0},
                                                        'mech': {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}, 'cells': [] }
        self.unstr_discr.fracture_aperture = 1
        self.unstr_discr.load_mesh(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=self.unstr_discr.fracture_aperture)
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.pm.grav = matrix([0.0, 0.0, 0.0], 1, 3)
        self.pm.visc = 1#9.81e-2
        self.unstr_discr.f = np.zeros(4 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
        self.p_init = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
        self.unstr_discr.p_ref = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
        perm_mult = 1.e-8
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
            self.p_init[cell_id] = self.p0(cell.centroid[1])

            self.pm.cell_centers.append(matrix(list(cell.centroid), cell.centroid.size, 1))
            if cell.prop_id == 99991 or cell.prop_id == 99993:
                self.pm.perms.append(matrix33(self.permx, self.permy, self.permz))
            else:
                self.pm.perms.append(matrix33(perm_mult * self.permx, perm_mult * self.permy, perm_mult * self.permz))
            self.pm.biots.append(matrix33(self.biot))

            E = self.unstr_discr.E[cell.prop_id]
            nu = self.unstr_discr.nu[cell.prop_id]
            lam = E * nu / (1 + nu) / (1 - 2 * nu)
            mu = E / (1 + nu) / 2
            self.pm.stfs.append(Stiffness(lam, mu))

            self.unstr_discr.f[4 * cell_id + 1] = self.rho_total * gravity / 1.E+5

        # fracture
        self.frac_aper = 1.E-5
        self.a = 75.0
        self.b = 150.0
        self.frac_apers = self.frac_aper * np.ones(self.unstr_discr.frac_cells_tot)
        for frac_id in range(self.unstr_discr.mat_cells_tot,
                             self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
            frac = self.unstr_discr.frac_cell_info_dict[frac_id]
            faces = self.unstr_discr.faces[frac_id]
            self.pm.cell_centers.append(matrix(list(frac.centroid), frac.centroid.size, 1))
            self.pm.frac_apers.append(self.frac_apers[frac_id - self.unstr_discr.mat_cells_tot])
            fs = face_vector()
            for face_id, face in faces.items():
                face = faces[face_id]
                fs.append(Face(face.type.value, face.cell_id1, face.cell_id2,
                               face.face_id1, face.face_id2,
                               face.area, list(face.n), list(face.centroid)))
            self.pm.faces.append(fs)

            face1 = faces[4]
            face2 = faces[5]
            self.mesh.fault_normals.append(face1.n[0])
            self.mesh.fault_normals.append(face1.n[1])
            self.mesh.fault_normals.append(face1.n[2])
            # Local basis
            S = np.zeros((n_dim, n_dim))
            S[:n_dim - 1] = null_space(np.array([face1.n])).T
            S[n_dim - 1] = face1.n
            Sinv = np.linalg.inv(S)
            K = np.zeros((n_dim, n_dim))
            K[0, 0] = K[1, 1] = self.permx
            K[n_dim - 1, n_dim - 1] = self.permx
            K = Sinv.dot(K).dot(S)
            self.pm.perms.append(matrix33(list(K.flatten())))
            self.pm.biots.append(matrix33(self.biot))

            self.p_init[frac_id] = self.p0(frac.centroid[1])
        # contact
        self.ref_contact_cells = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intc)
        self.contacts = contact_vector()
        for tag in self.unstr_discr.physical_tags['fracture']:
            con = contact()
            con.f_scale = 1.E+8
            cell_ids = [cell_id for cell_id, cell in self.unstr_discr.frac_cell_info_dict.items() if
                        cell.prop_id == tag]
            fric_coef = 0.7 * np.ones(len(cell_ids))
            #con.init_geometry(tag, self.pm, self.mesh, index_vector(cell_ids))
            self.ref_contact_cells[np.array(cell_ids, dtype=np.intp) - self.unstr_discr.mat_cells_tot] = cell_ids[0]
            con.mu0 = value_vector(fric_coef)
            con.mu = con.mu0
            con.fault_tag = tag
            con.cell_ids = index_vector(cell_ids)
            con.friction_criterion = critical_stress.BIOT
            #con.init_friction()
            self.contacts.append(con)
        self.bc_rhs_ref = np.zeros(4 * len(self.unstr_discr.bound_face_info_dict))
        self.bc_rhs = np.zeros(4 * len(self.unstr_discr.bound_face_info_dict))
        self.bc_rhs_prev = np.zeros(4 * len(self.unstr_discr.bound_face_info_dict))
        self.unstr_discr.pz_bounds = np.zeros(self.unstr_discr.bound_faces_tot)
        self.unstr_discr.pz_bounds[:] = self.p_init0
        for bound_id in range(len(self.unstr_discr.bound_face_info_dict)):
            b_cell = self.unstr_discr.bound_face_info_dict[bound_id]
            prop_id = b_cell.prop_id
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            mech = self.unstr_discr.boundary_conditions[prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
            if prop_id == 991:
                mech['rn'] = -np.interp(b_cell.centroid[1], self.pt991, self.ux991)
            elif prop_id == 981:
                mech['rn'] = -np.interp(b_cell.centroid[1], self.pt991, self.ux991)
                #flow['r'] = self.p0(b_cell.centroid[1]) - 200
            elif prop_id == 992:
                mech['rn'] = np.interp(b_cell.centroid[1], self.pt992, self.ux992)
            elif prop_id == 982:
                mech['rn'] = np.interp(b_cell.centroid[1], self.pt992, self.ux992)
                #flow['r'] = self.p0(b_cell.centroid[1]) - 300
            elif prop_id == 993:
               mech['rn'] = -np.interp(b_cell.centroid[0], self.pt993, self.uy993)
            elif (prop_id == 994):
               mech['rn'] = np.interp(b_cell.centroid[0], self.pt994, self.uy994)
            self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
            self.bc_rhs[4 * bound_id + 3] = flow['r']
            self.bc_rhs_prev[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
            self.bc_rhs_ref[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
            self.bc_rhs_ref[4 * bound_id + 3] = flow['r']
        for bound_id in range(len(self.unstr_discr.bound_face_info_dict), \
                              self.unstr_discr.bound_faces_tot + self.unstr_discr.frac_bound_faces_tot):
            mech = self.unstr_discr.boundary_conditions[self.unstr_discr.frac_bound_face_info_dict[bound_id].prop_id]['mech']
            flow = self.unstr_discr.boundary_conditions[self.unstr_discr.frac_bound_face_info_dict[bound_id].prop_id]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))
        self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc

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
            c_ref = self.unstr_discr.frac_cell_info_dict[list(self.unstr_discr.frac_cell_info_dict.keys())[0]].centroid
            c = self.unstr_discr.frac_cell_info_dict[id].centroid
            return (c[0] - c_ref[0]) ** 2 + (c[1] - c_ref[1]) ** 2
        def eval_frac_proj(tag, coords):
            c_ref = self.unstr_discr.frac_cell_info_dict[list(self.unstr_discr.frac_cell_info_dict.keys())[0]].centroid
            coords1 = np.copy(coords)
            coords1[0] -= c_ref[0]
            coords1[1] -= c_ref[1]
            return np.linalg.norm(t0[tag].dot(coords1), axis=0)

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

    def write_to_vtk(self, output_directory, ith_step, engine, dt = 0):
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
        property_array_n = np.array(engine.Xn, copy=False)
        property_array_n1 = np.array(engine.Xn1, copy=False)
        dX = np.array(engine.dX, copy=False)
        dt1 = engine.dt1
        time = engine.t if ith_step > 0 else 0.0
        available_matrix_geometries = ['hexahedron', 'wedge', 'tetra']
        available_fracture_geometries = ['quad', 'triangle']


        # vels = self.reconstruct_velocities(fluxes[engine.P_VAR::engine.N_VARS],
        #                                   fluxes_biot[engine.P_VAR::engine.N_VARS])
        self.mech_operators.eval_porosities(engine.X, self.mesh.bc)
        self.mech_operators.eval_stresses(engine.fluxes, engine.fluxes_biot, engine.X,
                                          self.mesh.bc, engine.op_vals_arr)
        #self.mech_operators.eval_stresses(engine.fluxes, engine.fluxes_biot, engine.X,
        #                                  self.mesh.bc, engine.op_vals_arr)

        # Matrix
        Mesh.cells = []
        cell_data = {}
        start_geom_cell_id = 0
        for ith_geometry in self.unstr_discr.geom_order:
            if ith_geometry in available_matrix_geometries:
                Mesh.cells.append(next(cell for cell in self.unstr_discr.mesh_data.cells if cell.type == ith_geometry))
                # Add matrix data to dictionary:
                cell_size = self.unstr_discr.mesh_data.cells_dict[ith_geometry].shape[0]
                for i in range(props_num):
                    if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(property_array[props_num * start_geom_cell_id + i:props_num * (cell_size + start_geom_cell_id):props_num])

                if 'porosity' not in cell_data: cell_data['porosity'] = []
                if 'eps_vol' not in cell_data: cell_data['eps_vol'] = []
                if 'stress' not in cell_data: cell_data['stress'] = []
                if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []

                if ith_step == 0:
                    cell_data['eps_vol'].append(np.array(self.mech_operators.eps_vol, copy=False))
                    cell_data['porosity'].append(np.array(self.mech_operators.porosities[start_geom_cell_id:cell_size + start_geom_cell_id], copy=False))
                else:
                    eps_vol = np.array(engine.eps_vol, copy=False)
                    cell_data['eps_vol'].append(eps_vol)
                    comp_mult = self.rock_compressibility
                    eps_vol_ref = np.array(self.mesh.ref_eps_vol, copy=False)
                    poro = self.porosity + (comp_mult * (property_array[self.P_VAR:props_num * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):props_num] - self.p_ref))[:self.unstr_discr.mat_cells_tot] + (eps_vol - eps_vol_ref)
                    poro[poro < 0.0] = self.poro[:self.unstr_discr.mat_cells_tot][poro < 0.0]
                    cell_data['porosity'].append(poro[start_geom_cell_id:cell_size + start_geom_cell_id])
                cell_data['stress'].append(np.zeros((cell_size, 6), dtype=np.float64))
                cell_data['tot_stress'].append(np.zeros((cell_size, 6), dtype=np.float64))

                stress = np.array(self.mech_operators.stresses, copy=False)
                total_stress = np.array(self.mech_operators.total_stresses, copy=False)
                for i in range(6):
                    cell_data['stress'][-1][:, i] = stress[6 * start_geom_cell_id + i:6 * (start_geom_cell_id + cell_size):6]
                    cell_data['tot_stress'][-1][:, i] = total_stress[6 * start_geom_cell_id + i:6 * (start_geom_cell_id + cell_size):6]

                # dynamic simulation
                if engine.momentum_inertia > 0.0 and dt != 0:
                    # velocity
                    if 'v_x' not in cell_data: cell_data['v_x'] = []
                    if 'v_y' not in cell_data: cell_data['v_y'] = []
                    if 'v_z' not in cell_data: cell_data['v_z'] = []
                    cell_data['v_x'].append(-dX[props_num * start_geom_cell_id:props_num * (cell_size + start_geom_cell_id):props_num] / dt / 86400.0)
                    cell_data['v_y'].append(-dX[props_num * start_geom_cell_id + 1:props_num * (cell_size + start_geom_cell_id):props_num] / dt / 86400.0)
                    cell_data['v_z'].append(-dX[props_num * start_geom_cell_id + 2:props_num * (cell_size + start_geom_cell_id):props_num] / dt / 86400.0)
                    # acceleration
                    # if 'a_x' not in cell_data: cell_data['a_x'] = []
                    # if 'a_y' not in cell_data: cell_data['a_y'] = []
                    # if 'a_z' not in cell_data: cell_data['a_z'] = []
                    #
                    # ax = np.array(cell_data['v_x'][-1]) / dt / 86400.0
                    # ay = np.array(cell_data['v_y'][-1]) / dt / 86400.0
                    # az = np.array(cell_data['v_z'][-1]) / dt / 86400.0
                    # if dt1 != 0.0:
                    #     ax -= (property_array_n[props_num * start_geom_cell_id:props_num * (cell_size + start_geom_cell_id):props_num] -
                    #             property_array_n1[props_num * start_geom_cell_id:props_num * (cell_size + start_geom_cell_id):props_num]) / dt1 / dt / 86400.0 / 86400.0
                    #     ay -= (property_array_n[props_num * start_geom_cell_id + 1:props_num * (cell_size + start_geom_cell_id):props_num] -
                    #             property_array_n1[props_num * start_geom_cell_id + 1:props_num * (cell_size + start_geom_cell_id):props_num]) / dt1 / dt / 86400.0 / 86400.0
                    #     az -= (property_array_n[props_num * start_geom_cell_id + 2:props_num * (cell_size + start_geom_cell_id):props_num] -
                    #             property_array_n1[props_num * start_geom_cell_id + 2:props_num * (cell_size + start_geom_cell_id):props_num]) / dt1 / dt / 86400.0 / 86400.0
                    # if 'a_x' not in cell_data: cell_data['a_x'] = []
                    # if 'a_y' not in cell_data: cell_data['a_y'] = []
                    # if 'a_z' not in cell_data: cell_data['a_z'] = []
                    # cell_data['a_x'].append(ax)
                    # cell_data['a_y'].append(ay)
                    # cell_data['a_z'].append(az)

                # if 'cell_id' not in cell_data: cell_data['cell_id'] = []
                # cell_data['cell_id'].append(np.array([cell_id for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items() if cell.geometry_type == ith_geometry], dtype=np.intc))
                # if ith_step == 0:
                #     cell_data[ith_geometry]['permx'] = self.permx[:]
                #     cell_data[ith_geometry]['permy'] = self.permy[:]
                #     cell_data[ith_geometry]['permz'] = self.permz[:]
                # tags = np.array([cell.prop_id for cell in self.unstr_discr.mat_cell_info_dict.values()], dtype=np.int)
                # if 'tags' not in cell_data: cell_data['tags'] = []
                # cell_data['tags'].append(tags)

                start_geom_cell_id += cell_size

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            Mesh.points,
            Mesh.cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtu".format(output_directory, ith_step), mesh)

        self.write_pvd_file(ith_step, time, output_directory)

        # Fractures
        geom_id = 0
        Mesh.cells = []
        cell_data = {}
        for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
            if ith_geometry in available_fracture_geometries:
                # fracture geometry
                frac_ids = np.argwhere(np.in1d(self.unstr_discr.mesh_data.cell_data['gmsh:physical'][geom_id],
                                                                                              self.unstr_discr.physical_tags['fracture']))[:, 0]
                if len(frac_ids):
                    Mesh.cells.append(meshio.CellBlock(ith_geometry, data=self.unstr_discr.mesh_data.cells[geom_id].data[frac_ids]))
                    data = property_array[4 * self.unstr_discr.mat_cells_tot:4 * (
                            self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)].reshape(
                        self.unstr_discr.frac_cells_tot, 4)
                    for i in range(props_num):
                        if cell_property[i] not in cell_data: cell_data[cell_property[i]] = []
                        cell_data[cell_property[i]].append(data[:, i])

                # output geometry
                out_ids = np.argwhere(np.in1d(self.unstr_discr.mesh_data.cell_data['gmsh:physical'][geom_id],
                                              self.unstr_discr.physical_tags['output']))[:, 0]
                if len(out_ids):
                    Mesh.cells.append(meshio.CellBlock(ith_geometry, data=self.unstr_discr.mesh_data.cells[geom_id].data[out_ids]))

            geom_id += 1
        # fracture output
        if self.unstr_discr.frac_cells_tot > 0:
            #self.write_fault_props(output_directory, property_array, ith_step, engine)
            frac_data = self.get_fault_props(property_array, ith_step, engine)
            for key, val in frac_data.items():
               if key not in cell_data: cell_data[key] = []
               cell_data[key].append(val)
        # just output
        if self.unstr_discr.output_faces_tot > 0:
            out_data = self.get_props_over_output(property_array, ith_step, engine)
            for key, val in out_data.items():
               if key not in cell_data: cell_data[key] = []
               cell_data[key].append(val)

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            Mesh.points,
            Mesh.cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution_fault{:d}.vtu".format(output_directory, ith_step), mesh)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        return 0

    def get_props_over_output(self, property_array, ith_step, engine):
        n_vars = 4
        n_dim = 3
        fluxes = np.array(engine.fluxes, copy=False)
        fluxes_biot = np.array(engine.fluxes_biot, copy=False)
        cell_m = np.array(self.mesh.block_m, copy=False)
        cell_p = np.array(self.mesh.block_p, copy=False)

        #S_eng = vector_matrix(engine.contacts[0].S)
        #frac_prop = property_array[n_vars * self.unstr_discr.mat_cells_tot:n_vars * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)].reshape(self.unstr_discr.frac_cells_tot, n_vars)
        #fstress = np.array(engine.contacts[0].fault_stress, copy=False)

        S = np.zeros((n_dim, n_dim))
        frac_data = {}
        #frac_data['tag'] = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intp)
        frac_data['f_local'] = np.zeros((self.unstr_discr.output_face_tot, n_dim))

        ref_id = self.unstr_discr.output_face_to_face[0]
        ref_face = self.unstr_discr.faces[ref_id[0]][ref_id[1]]
        S[1:n_dim] = null_space(np.array([ref_face.n])).T
        S[0] = ref_face.n
        for face_id, ids in self.unstr_discr.output_face_to_face.items():
            #cell_id -= self.unstr_discr.mat_cells_tot
            #frac_data['tag'][cell_id] = int(cell.prop_id)
            face = self.unstr_discr.faces[ids[0]][ids[1]]
            sign = np.sign((self.unstr_discr.mat_cell_info_dict[face.cell_id2].centroid - \
                            self.unstr_discr.mat_cell_info_dict[face.cell_id1].centroid).dot(ref_face.n))
            flux_ids = np.argwhere(np.logical_and(cell_m == face.cell_id1, cell_p == face.cell_id2))[0]
            flux = sign * fluxes[n_vars * flux_ids[0]:n_vars * flux_ids[0] + n_dim] / face.area

            if len(flux_ids):
                frac_data['f_local'][face_id] = S.dot(flux)
            else:
                return 0

        for face_id in range(self.unstr_discr.output_face_tot):
            print(str(face_id + self.unstr_discr.mat_cells_tot) + ' ' + str(frac_data['f_local'][face_id][0] * 1.E+5) + ' ' +
                  str(frac_data['f_local'][face_id][1] * 1.E+5) + ' ' + str(frac_data['f_local'][face_id][2] * 1.E+5))

        return frac_data
    def get_fault_props(self, property_array, ith_step, engine):
        n_vars = 4
        n_dim = 3
        fluxes = np.array(engine.fluxes, copy=False)
        fluxes_biot = np.array(engine.fluxes_biot, copy=False)
        S_eng = vector_matrix(engine.contacts[0].S)
        frac_prop = property_array[n_vars * self.unstr_discr.mat_cells_tot:n_vars * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)].reshape(self.unstr_discr.frac_cells_tot, n_vars)
        fstress = np.array(engine.contacts[0].fault_stress, copy=False)

        frac_data = {}
        #frac_data['tag'] = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intp)
        frac_data['g_local'] = np.zeros((self.unstr_discr.frac_cells_tot, n_dim))
        frac_data['f_local'] = np.zeros((self.unstr_discr.frac_cells_tot, n_dim))
        frac_data['mu'] = np.array(engine.contacts[0].mu, copy=False)

        for cell_id, cell in self.unstr_discr.frac_cell_info_dict.items():
            cell_id -= self.unstr_discr.mat_cells_tot
            #frac_data['tag'][cell_id] = int(cell.prop_id)
            face = self.unstr_discr.faces[cell_id + self.unstr_discr.mat_cells_tot][4]
            S = np.array(S_eng[cell_id].values).reshape((n_dim, n_dim))
            f = fstress[n_dim * cell_id:n_dim * (cell_id + 1)] / face.area
            frac_data['f_local'][cell_id] = S.dot(f)
            frac_data['g_local'][cell_id] = S.dot(frac_prop[cell_id,:n_dim])

        phi = np.array(engine.contacts[0].phi, copy=False)
        #states = phi > 0
        frac_data['phi'] = phi

        return frac_data
    def write_fault_props(self, output_directory, property_array, ith_step, engine):
        n_vars = 4
        n_dim = 3
        fluxes = np.array(engine.fluxes, copy=False)
        fluxes_biot = np.array(engine.fluxes_biot, copy=False)
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
            S_eng = vector_matrix(engine.contacts[0].S)
            mu[tag] = np.array(engine.contacts[0].mu, copy=False)
            fstress = np.array(engine.contacts[0].fault_stress, copy=False)
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
            #self.ax[0].set_ylabel(r'normal gap, $g_N$')
            self.ax[0].set_ylabel(r'friction coefficient, $\mu$')
            self.ax0.set_ylabel(r'slip, $g_T$')
            self.ax[1].set_ylabel(r'normal traction, $F_N$')
            self.ax1.set_ylabel(r'tangential traction, $F_T$')
            self.ax[1].set_xlabel('distance')
                #self.ax1.set_ylabel('distance along fault')

            phi = np.array(engine.contacts[0].phi, copy=False)
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

    def write_pvd_file(self, ith_step, time, output_directory):
        # writing *.pvd file
        if not hasattr(self, 'matpvd_doc'):  # do just once, at the first call
            self.matpvd_doc = xml.dom.minidom.parseString("<VTKFile/>")
            self.matpvd_root = self.matpvd_doc.documentElement
            self.matpvd_root.setAttribute("type", "Collection")
            self.matpvd_root.setAttribute("version", "0.1")
            self.matpvd_collection = self.matpvd_doc.createElement("Collection")

            self.faultpvd_doc = xml.dom.minidom.parseString("<VTKFile/>")
            self.faultpvd_root = self.faultpvd_doc.documentElement
            self.faultpvd_root.setAttribute("type", "Collection")
            self.faultpvd_root.setAttribute("version", "0.1")
            self.faultpvd_collection = self.faultpvd_doc.createElement("Collection")

        # *.pvd
        snap = self.matpvd_doc.createElement("DataSet")
        snap.setAttribute("timestep", str(time))
        snap.setAttribute("file", 'solution{:d}.vtu'.format(ith_step))
        self.matpvd_collection.appendChild(snap)
        root = self.matpvd_root
        root.appendChild(self.matpvd_collection)
        self.matpvd_doc.writexml(open(str(output_directory) + '/solution.pvd', 'w'),
                     indent="  ",
                     addindent="  ",
                     newl='\n')

        # *.pvd
        snap = self.faultpvd_doc.createElement("DataSet")
        snap.setAttribute("timestep", str(time))
        snap.setAttribute("file", 'solution_fault{:d}.vtu'.format(ith_step))
        self.faultpvd_collection.appendChild(snap)
        root = self.faultpvd_root
        root.appendChild(self.faultpvd_collection)
        self.faultpvd_doc.writexml(open(str(output_directory) + '/solution_fault.pvd', 'w'),
                     indent="  ",
                     addindent="  ",
                     newl='\n')
