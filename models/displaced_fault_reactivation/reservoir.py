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
        mesh_file = 'meshes/new_setup_three_point_stick1.geo'
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

        data991 = pd.read_csv(os.path.join('data', 'analytics', 'xm.csv'), delimiter=',')
        self.pt991 = np.array(data991['Points:1'],  dtype=np.float64)
        self.ux991 = np.array(data991['u_x'],       dtype=np.float64)

        data992 = pd.read_csv(os.path.join('data', 'analytics', 'xp.csv'), delimiter=',')
        self.pt992 = np.array(data992['Points:1'],  dtype=np.float64)
        self.ux992 = np.array(data992['u_x'],       dtype=np.float64)

        data993 = pd.read_csv(os.path.join('data', 'analytics', 'ym.csv'), delimiter=',')
        self.pt993 = np.array(data993['Points:0'], dtype=np.float64)
        self.uy993 = np.array(data993['u_y'], dtype=np.float64)

        data994 = pd.read_csv(os.path.join('data', 'analytics','yp.csv'), delimiter=',')
        self.pt994 = np.array(data994['Points:0'], dtype=np.float64)
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


