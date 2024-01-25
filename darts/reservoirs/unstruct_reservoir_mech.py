import numpy as np
import os
import meshio

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

def get_lambda_mu(E, nu):
    '''
    :param E: Young modulus [bars]
    :param nu: Poisson ratio
    :return: lambda and mu coefficitents for Stiffness matrix
    '''
    lam = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2. / (1 + nu)
    return lam, mu

def get_kd_cur(E, nu):
    kd_cur = E / 3. / (1 - 2 * nu)
    return kd_cur

def get_M(biot, porosity, kd_cur, fluid_compressibility):
    if biot == 1. and fluid_compressibility == 0.:  # avoid divizion by zero
        M = None
    else:
        M = 1.0 / ((biot - porosity) * (1 - biot) / kd_cur + porosity * fluid_compressibility)
    return M

class GeoMechInputData():
    '''
    Class for input data Poroelasticity/ThermoPoroElasticity coupled model
    '''
    def __init__(self):
        self.porosity = None
        self.permx = self.permy = self.permz = None  # Permeability [mD]
        self.E = None   # Young modulus [bars]
        self.nu = 0.25  # Poisson ratio

class UnstructReservoirMech(): 
    #TODO: inherit from UnstructReservoirBase to have add_well functions from there
    #TODO: create a py wrapper reservoir class UnstructReservoirCPP for C++ discretizer (flow only, MPFA)
    #TODO: crate an abstract  class UnstructReservoirBase for existing Python class and UnstructReservoirCPP
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

        self.a = np.max([node.values[0] for node in self.discr_mesh.nodes])  # max value of X coordinate
        self.b = np.max([node.values[1] for node in self.discr_mesh.nodes])  # max value of Y coordinate
        if self.thermoporoelacticity:
            self.discr = thermoporo_mech_discretizer()
        else:
            self.discr = poro_mech_discretizer()
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
        self.adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        self.adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)

        if hasattr(self, 'E'):  # if uniform geomechanical properties
            self.kd_cur = get_kd_cur(self.E, self.nu)
            self.lam, self.mu = get_lambda_mu(self.E, self.nu)
            self.M = get_M(self.biot, self.porosity, self.kd_cur, self.fluid_compressibility)
        if self.thermoporoelacticity:
            self.th_expn = self.th_expn_coef * self.kd_cur

        self.ref_contact_cells = np.zeros(self.n_fracs, dtype=np.intc)

        #TODO: instead of biot and kd, calc compressibility in python and pass this to engines
        # comp_mult = (biot_cur != 0) ? (biot_cur - poro[i]) * (1 - biot_cur) / kd[i] : 1.0 / kd[i];

        #self.kd_cur = np.zeros(self.n_matrix) # corresponds to drained_compressibility in conn_mesh
        #if not hasattr(self, 'porosity'):
        #    self.porosity = 0.
        #self.porosity = self.porosity + np.zeros(self.n_matrix + self.n_fracs)
        #self.biot_mean = np.zeros(9 * (self.n_matrix))

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
        self.p_ref = np.array(self.mesh.ref_pressure, copy=False)
        if self.thermoporoelacticity:
            self.t_ref = np.array(self.mesh.ref_temperature, copy=False)
            self.th_expn_poro_arr = np.array(self.mesh.th_poro, copy=False)

        # specify properties
        self.poro[:self.n_matrix] = self.porosity
        self.poro[self.n_matrix:] = 1  # fractures

        if self.discretizer_name == 'mech_discretizer':
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.volume[:self.n_matrix] = volumes[:self.n_matrix]  #TODO init frac volumes
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            self.biot_arr[:] = self.biot_mean
            self.kd[:] = self.kd_cur
            self.p_ref[:] = self.p_init
            if self.thermoporoelacticity:
                self.t_ref[:] = self.t_init
                self.th_expn_poro_arr[:] = self.th_expn_poro
        elif self.discretizer_name == 'pm_discretizer':
            self.volume[:self.unstr_discr.mat_cells_tot] = self.unstr_discr.volume_all_cells[self.unstr_discr.frac_cells_tot:]
            for i in range(self.unstr_discr.mat_cells_tot, self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
                self.volume[i] = self.unstr_discr.faces[i][4].area * self.frac_apers[i-self.unstr_discr.mat_cells_tot]
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            self.bc_ref[:] = self.bc_rhs_ref
            self.biot_arr[:] = self.biot_mean
            self.kd[:] = self.kd_cur
            self.p_ref[:] = self.unstr_discr.p_ref
            self.f[:] = self.unstr_discr.f


    def set_pz_bounds(self, p, z=None, t=None):
        '''
        # sets boundary values of pressures, (inflow) fractions at boundaries, and temperatures
        # should be called after conn_mesh initialization
        :param p: pressure values, bars
        :param z: composition values TODO: implement
        :param t: temperatures values, degrees
        :return: None
        '''
        self.mesh.pz_bounds.resize(self.n_state * self.n_bounds)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        if self.discretizer_name == 'mech_discretizer':
            self.pz_bounds[self.p_var::self.n_state] = p
            if self.thermoporoelacticity:
                self.pz_bounds[self.t_var::self.n_state] = t
        elif self.discretizer_name == 'pm_discretizer':
            self.pz_bounds[:] = p

    def init_bc_rhs(self):
        if self.discretizer_name == 'mech_discretizer':
            for tag in self.domain_tags[elem_loc.BOUNDARY]:
                ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
                bc = self.boundary_conditions[tag]
                # flow
                self.bc_rhs[self.n_vars * ids + self.p_var] = bc['flow']['r']
                # energy
                if self.thermoporoelacticity:
                    self.bc_rhs[self.n_vars * ids + self.t_var] = bc['temp']['r']
                # mechanics
                for id in ids:
                    assert(self.adj_matrix_cols[self.id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                    conn = self.conns[self.id_boundary_conns[id]]
                    n = np.array(conn.n.values, copy=False)
                    conn_c = np.array(conn.c.values, copy=False)
                    c1 = np.array(self.centroids[conn.elem_id1].values, copy=False)
                    if n.dot(conn_c - c1) < 0: n *= -1.0
                    self.bc_rhs[self.n_vars * id + self.u_var:self.n_vars * id + self.u_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']
        elif self.discretizer_name == 'pm_discretizer':
            self.pm.bc.clear()
            for bound_id in range(len(self.unstr_discr.bound_cell_info_dict)):
                n = self.get_normal_to_bound_face(bound_id)
                # P = np.identity(3) - np.outer(n, n)
                mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['mech']
                flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_cell_info_dict[bound_id].prop_id]['flow']
                bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
                self.pm.bc.append(matrix(bc, len(bc), 1))
                self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']
                self.bc_rhs[4 * bound_id + 3] = flow['r']

    def init_arrays_boundary_condition(self):
        if self.discretizer_name == 'mech_discretizer':
            # mapping boundary connections
            self.id_sorted = np.argsort(self.adj_matrix_cols)[-self.n_bounds:] # store it to self. as it will be used in init_bc_rhs() further
            self.id_boundary_conns = self.adj_matrix[self.id_sorted]

            ap = np.ones(self.n_bounds)
            bp = np.zeros(self.n_bounds)
            amn = np.zeros(self.n_bounds)
            bmn = np.zeros(self.n_bounds)
            amt = np.zeros(self.n_bounds)
            bmt = np.zeros(self.n_bounds)
            if self.thermoporoelacticity:
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
                if self.thermoporoelacticity:
                    at[ids] = bc['temp']['a']
                    bt[ids] = bc['temp']['b']

            self.init_bc_rhs()

            self.cpp_bc = THMBoundaryCondition()
            self.cpp_bc.flow.a = value_vector(ap)
            self.cpp_bc.flow.b = value_vector(bp)
            self.cpp_bc.mech_normal.a = value_vector(amn)
            self.cpp_bc.mech_normal.b = value_vector(bmn)
            self.cpp_bc.mech_tangen.a = value_vector(amt)
            self.cpp_bc.mech_tangen.b = value_vector(bmt)
            if self.thermoporoelacticity:
                self.cpp_bc.thermal.a = value_vector(at)
                self.cpp_bc.thermal.b = value_vector(bt)
            # to use base discretizer's class function reconstruct_pressure_gradients_per_cell
            # which doesn't know the new THMBoundaryCondition class yet
            self.cpp_flow = BoundaryCondition()
            self.cpp_flow.a = value_vector(ap)
            self.cpp_flow.b = value_vector(bp)
            if self.thermoporoelacticity:
                self.cpp_heat = BoundaryCondition()
                self.cpp_heat.a = value_vector(at)
                self.cpp_heat.b = value_vector(bt)
        elif self.discretizer_name == 'pm_discretizer':
            pass

    def set_boundary_conditions_pm_discretizer(self):
        if self.discretizer_name == 'pm_discretizer':
            self.unstr_discr.boundary_conditions = self.boundary_conditions
            for key in self.boundary_conditions.keys():
                self.boundary_conditions[key]['cells'] = []

    def init_gravity(self, gravity_on=False):
        # set gravity vector
        if gravity_on:
            from scipy import gravitational_constant
            grav_coeff = gravitational_constant / 1e5  # convert units
        else:
            grav_coeff = 0.
        grav_vec = matrix([0.0, 0.0, grav_coeff], 1, 3)
        if self.discretizer_name == 'mech_discretizer':
            self.discr.grav_vec = grav_vec
        elif self.discretizer_name == 'pm_discretizer':
            self.pm.grav = grav_vec

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
                cell = self.unstr_discr.mat_cell_info_dict[cell_id]
                self.pm.cell_centers.append(matrix(list(cell.centroid), cell.centroid.size, 1))
                self.pm.perms.append(engine_matrix33(self.permx, self.permy, self.permz))
                self.pm.biots.append(engine_matrix33(self.biot))
                self.pm.stfs.append(engine_stiffness(self.lam, self.mu))
                self.biot_mean[9 * cell_id] = self.biot
                self.biot_mean[9 * cell_id + 4] = self.biot
                self.biot_mean[9 * cell_id + 8] = self.biot

    def init_heterogeneous_properties(self):
        '''
        set matrix poperties using self.props[tag]
        :return:
        '''
        if self.discretizer_name == 'mech_discretizer':
            self.biot_mean = np.zeros(9 * (self.n_matrix + self.n_fracs))
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                              self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
                tag = self.tags[cell_id]
                E = self.props[tag]['E']
                nu = self.props[tag]['nu']
                biot = self.props[tag]['b']
                k = self.props[tag]['perm']
                kd = self.props[tag]['kd']
                poro = self.props[tag]['poro']
                lam, mu = get_lambda_mu(E, nu)

                self.discr.perms.append(disc_matrix33(k, k, k))
                self.discr.biots.append(disc_matrix33(biot))
                self.discr.stfs.append(disc_stiffness(lam, mu))
                self.biot_mean[9 * cell_id] = biot
                self.biot_mean[9 * cell_id + 4] = biot
                self.biot_mean[9 * cell_id + 8] = biot
                self.porosity[cell_id] = poro
                self.kd_cur[cell_id] = kd
        elif self.discretizer_name == 'pm_discretizer':
            self.kd_cur = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.porosity = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.biot_mean = np.zeros(9 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))
            for cell_id in range(self.unstr_discr.mat_cells_tot):
                E = self.props[cell.prop_id]['E']
                nu = self.props[cell.prop_id]['nu']
                biot = self.props[cell.prop_id]['b']
                k = self.props[cell.prop_id]['perm']
                kd = self.props[cell.prop_id]['kd']
                poro = self.props[cell.prop_id]['poro']
                lam, mu = get_lambda_mu(E, nu)
                self.pm.stfs.append(Stiffness(lam, mu))
                self.pm.perms.append(matrix33(k, k, k))
                self.pm.biots.append(matrix33(biot))
                self.kd_cur[cell_id] = kd  # (biot - self.porosity) * (1 - biot) * kd
                self.biot_mean[9 * cell_id] = biot
                self.biot_mean[9 * cell_id + 4] = biot
                self.biot_mean[9 * cell_id + 8] = biot
                self.porosity[cell_id] = poro

    def set_uniform_initial_conditions(self, u_init=[0., 0., 0.], p_init=0., t_init=0.):  #TODO: check units
        self.u_init = u_init  # initial displacements U_x, U_y, U_z [m.]
        self.p_init = p_init  # initial pressure [bars]
        if self.thermoporoelacticity:
            self.t_init = t_init  # initial temperature [degrees]
        else:
            self.t_init = None

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

    def init_faces_centers_pm_discretizer(self):
        assert self.discretizer_name == 'pm_discretizer'
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

    def set_scheme_pm_discretizer(self, scheme='non_stabilized'):
        assert self.discretizer_name == 'pm_discretizer'
        if scheme == 'stabilized':
            self.pm.scheme = scheme_type.apply_eigen_splitting_new
            self.pm.min_alpha_stabilization = 0.5
        elif scheme == 'non_stabilized':
            pass
        else:
            print('Error: unsupported scheme', scheme)
            exit(1)

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