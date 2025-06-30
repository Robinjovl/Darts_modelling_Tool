import numpy as np
import os
import pandas as pd
import pickle
#from math import inf, pi
#from itertools import compress

from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, contact, contact_vector, vector_matrix
from darts.engines import matrix33, matrix, pm_discretizer, Face, vector_face_vector, face_vector, vector_matrix33, Stiffness, stf_vector, critical_stress, scheme_type
from darts.engines import timer_node

from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
#from darts.reservoirs.mesh.geometrymodule import FType
from darts.reservoirs.unstruct_reservoir_mech import get_rock_compressibility, bound_cond
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags, get_lambda_mu, get_bulk_modulus, get_biot_modulus
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech

from darts.input.input_data import InputData

from utils import dict_hash

class UnstructReservoir(UnstructReservoirMech):
    def __init__(self, timer, fluid_density, rock_density, mesh_file, cache_discretizer: bool = True):
        super().__init__(timer, discretizer='pm_discretizer', fluid_vars=['p'], thermoporoelasticity=False)

        self.rho_s = rock_density
        self.rho_f = fluid_density
        self.mesh_file = mesh_file

        self.bc_type = bound_cond()  # get predefined constants for boundary conditions

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

        self.idata = InputData(type_hydr='isothermal', type_mech='poroelasticity', init_type='uniform')
        self.idata.rock.biot = 0.9
        self.idata.rock.perm = 100.0
        self.idata.other.friction = 0.7
        self.idata.other.frac_apers = 1e-5

        self.reservoir_depletion(idata=self.idata)

        if self.cache_discretizer:
            # flag checking if update of saved cache needed
            save_cache_discretizer = False
            if os.path.exists(self.cache_filename):
                with (open(self.cache_filename, "rb") as fp):
                    cached_data = pickle.load(fp)

                    cached_var_names_short = cached_var_names.copy()
                    for it in ['self.unstr_discr', 'self.pm']:
                        cached_var_names_short.remove(it)

                    # check a hash is the same
                    curr_data = {var_name: eval(var_name, {'self': self}) for var_name in cached_var_names_short}
                    curr_hash = dict_hash(curr_data)

                    if cached_data['hash'] != curr_hash:
                        print('geometry/rock props were changed, ', self.cache_filename, 'will be updated')
                        print('current hash=', curr_hash, 'cached hash=', cached_data['hash'])
                        save_cache_discretizer = True
            else:
                save_cache_discretizer = True

            # update needed, re-do discretization
            if save_cache_discretizer:
                self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot,
                             index_vector(self.ref_contact_cells))
                dt = 0
                self.pm.reconstruct_gradients_per_cell(dt)
                self.pm.calc_all_fluxes_once(dt)

                cached_data = {var_name: eval(var_name, {'self': self}) for var_name in cached_var_names}

                cached_var_names_short = cached_var_names.copy()
                for it in ['self.unstr_discr', 'self.pm']:
                    cached_var_names_short.remove(it)
                cached_data_short = {var_name: eval(var_name, {'self': self}) for var_name in cached_var_names_short}
                hash = dict_hash(cached_data_short)
                cached_data.update({'hash': hash})
                print('saving cache, hash=', hash)

                with open(self.cache_filename, "wb") as fp:
                    pickle.dump(cached_data, fp, 4)

            # update not needed, just load and use
            else:
                print('discretizer cache is used. hash=', cached_data['hash'])
                for var_name, var in cached_data.items():
                    if var_name == 'self.rho_f':
                        assert var == self.rho_f
                    elif var_name == 'self.rho_s':
                        assert var == self.rho_s
                    exec(var_name + " = var")

                self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot,
                             index_vector(self.ref_contact_cells))
        else:
            # initialize and run discretizer
            dt = 0
            self.pm.init(self.unstr_discr.mat_cells_tot, self.unstr_discr.frac_cells_tot,
                         index_vector(self.ref_contact_cells))
            self.pm.reconstruct_gradients_per_cell(dt)
            self.pm.calc_all_fluxes_once(dt)

        self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
                          self.pm.tran_biot, self.pm.rhs_biot, self.unstr_discr.mat_cells_tot, \
                          self.unstr_discr.bound_faces_tot + self.unstr_discr.frac_bound_faces_tot, self.unstr_discr.frac_cells_tot)
        # self.write_pm_conn_to_file(t_step=0)
        self.timer.node["discretization"].stop()

        self.unstr_discr.store_volume_all_cells()
        self.unstr_discr.store_depth_all_cells()

        self.set_vars_pm_discretizer()
        self.cs = get_rock_compressibility(kd=self.lam + 2 * self.mu / 3, biot=self.biot, poro0=self.porosity)
        self.init_arrays(idata=self.idata)

        self.mesh.pz_bounds.resize(self.unstr_discr.bound_faces_tot + self.unstr_discr.frac_bound_faces_tot)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.p_ref = np.array(self.mesh.ref_pressure, copy=False)
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

    def initial_stage(self, idata: InputData):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init0 = 350.0
        self.porosity = 0.15

        mesh_file = 'meshes/new_setup_three_point_stick1.geo'
        self.file_path = mesh_file
        self.permx, self.permy, self.permz = idata.rock.get_permxyz()
        self.unstr_discr = UnstructDiscretizer(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0,
                                               mesh_file=mesh_file)

        self.mu = 65000 # in bars
        nu = 0.15
        E = 2 * self.mu * (1 + nu)
        self.lam, self.mu = self.get_lambda_mu(E, nu)

        self.unstr_discr.init_matrix_stiffness({99991: {'E': E, 'nu': nu}, 99992: {'E': E, 'nu': nu}})
        self.unstr_discr.physical_tags['matrix'] = [99991, 99992]
        self.unstr_discr.physical_tags['fracture_shape'] = [1]
        self.unstr_discr.physical_tags['fracture'] = [9991]
        self.unstr_discr.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996, 998, 999]
        self.unstr_discr.physical_tags['output'] = []
        # General representation of BC: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)

        gravity = 9.81
        self.rho_s = 2650.0
        self.rho_f = 1020.0
        self.rho_total = (1 - self.porosity) * self.rho_s + self.porosity * self.rho_f
        #assert(np.fabs(self.fv - 295.0) < 1.0)
        self.K0 = 0.5
        self.biot = self.idata.rock.biot
        self.H = 4500.0

        self.sigma_yy = lambda y: self.rho_total * gravity * (y - 3500.0) / 1.E+5
        self.p0 = lambda y: self.p_init0 - self.rho_f * gravity * y / 1.E+5
        self.fh = lambda y: -self.K0 * (self.sigma_yy(y) + self.biot * self.p0(y)) + self.biot * self.p0(y)
        self.eps_yy = lambda y: (1 - 2 * nu) / 2 / self.mu / (1 - nu) * (self.sigma_yy(y) + self.biot * self.p0(y))

        self.unstr_discr.boundary_conditions[991] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.LOAD(-0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.LOAD(-0.0, [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.LOAD(self.sigma_yy(-self.H / 2), [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.LOAD(self.sigma_yy(self.H / 2), [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.ROLLER, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.ROLLER, 'cells': []}
        self.unstr_discr.boundary_conditions[998] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_T_LOAD_N(self.sigma_yy(-self.H / 2), [0.0, 0.0, 0.0]), 'cells': []}
        self.unstr_discr.boundary_conditions[999] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_T_LOAD_N(0.0, [0.0, 0.0, 0.0]), 'cells': []}
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

        self.init_fractures(idata=idata)

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

    def reservoir_depletion(self, idata: InputData):
        self.u_init = [0.0, 0.0, 0.0]
        self.p_init0 = 350.0
        self.porosity = 0.16 #0.15
        self.permx, self.permy, self.permz = idata.rock.get_permxyz()
        physical_tags = {}
        physical_tags['matrix'] = [99991, 99992, 99993]
        physical_tags['fracture_boundary'] = [1, 2]
        physical_tags['fracture'] = [9991]
        physical_tags['output'] = []
        physical_tags['boundary'] = [991, 981, 992, 982, 993, 994, 995, 996, 998]
        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=physical_tags)

        self.mu = 65000 # in bars
        nu = 0.15
        E = 2 * self.mu * (1 + nu)
        self.lam, self.mu = get_lambda_mu(E, nu)

        self.unstr_discr.init_matrix_stiffness({99991: {'E': E, 'nu': nu}, 99992: {'E': E, 'nu': nu}, 99993: {'E': E, 'nu': nu}})

        gravity = 9.81
        self.rho_s = 2650.0
        self.rho_f = 1020.0
        self.rho_total = (1 - self.porosity) * self.rho_s + self.porosity * self.rho_f
        #assert(np.fabs(self.fv - 295.0) < 1.0)
        self.K0 = 0.5
        self.biot = self.idata.rock.biot
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

        self.unstr_discr.boundary_conditions[991] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[981] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[992] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[982] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[993] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[994] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK_ROLLER(0.0), 'cells': []}
        self.unstr_discr.boundary_conditions[995] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.ROLLER, 'cells': []}
        self.unstr_discr.boundary_conditions[996] = {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.ROLLER, 'cells': []}
        # top and bottom fracture boundaries
        self.unstr_discr.boundary_conditions[1] =   {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.STUCK(0.0, 0.0), 'cells': [] }
        # side fracture boundaries
        self.unstr_discr.boundary_conditions[2] =   {'flow': self.bc_type.NO_FLOW, 'mech': self.bc_type.ROLLER, 'cells': [] }

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
        perm_mult = 1.e-8 # outer reservoir has lower permeability
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
            if cell.prop_id in [99991, 99993]: # inner part of matrix
                self.pm.perms.append(matrix33(self.permx, self.permy, self.permz))
            else:  # outer part of matrix
                self.pm.perms.append(matrix33(perm_mult * self.permx, perm_mult * self.permy, perm_mult * self.permz))
            self.pm.biots.append(matrix33(self.biot))

            E = self.unstr_discr.E[cell.prop_id]
            nu = self.unstr_discr.nu[cell.prop_id]
            lam, mu = get_lambda_mu(E, nu)
            self.pm.stfs.append(Stiffness(lam, mu))

            self.unstr_discr.f[4 * cell_id + 1] = self.rho_total * gravity / 1.E+5

        # fracture
        self.a = 75.0
        self.b = 150.0
        self.init_fractures(idata=idata)
        for frac_id in range(self.unstr_discr.mat_cells_tot,
                             self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
            frac = self.unstr_discr.frac_cell_info_dict[frac_id]
            self.p_init[frac_id] = self.p0(frac.centroid[1])

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


