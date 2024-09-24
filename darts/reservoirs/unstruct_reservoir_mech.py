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
from darts.input.input_data import InputData

class bound_cond:
    '''
    General representation of boundary condition: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
    '''
    def __init__(self):
        # flow
        self.NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        self.AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        self.FLOW = lambda flow: {'a': 0.0, 'b': 1.0, 'r': flow} #TODO normed to area ? units?

        # mechanics
        self.ROLLER = {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        self.FREE = {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        self.STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        # Fn, Ft are normal and tangential load [UNIT?]
        self.LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        # the same as ROLLER except rn is non-zero
        self.STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

def set_domain_tags(matrix_tags,
                    bnd_xm_tag=None, bnd_xp_tag=None,
                    bnd_ym_tag=None, bnd_yp_tag=None,
                    bnd_zm_tag=None, bnd_zp_tag=None,
                    bnd_tags=None,  # list of boundary tags
                    fracture_tags=[], frac_bnd_tags=[]):
    '''
    :param matrix_tag: list of integers
    :param bnd_tags: list of integers
    :param fracture_tag: list of integers
    :param frac_bnd_tag: list of integers
    :return: dictionary of sets containing integer tags for each element type; dictionary of tags for 6 boundaries
    '''
    # one can specify tags for the each boundary side, or the list of boundary tags (bnd_tags)
    if bnd_tags is None:
        boundary_tags = [bnd_xm_tag, bnd_xp_tag, bnd_ym_tag, bnd_yp_tag, bnd_zm_tag, bnd_zp_tag]
    else:
        boundary_tags = bnd_tags
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

def get_bulk_modulus(E, nu):
    kd = E / 3. / (1 - 2 * nu)
    return kd

def get_biot_modulus(biot, poro0, kd, cf):
    eps = 1e-100 # avoid divizion by zero
    M = 1.0 / (get_rock_compressibility(kd=kd, biot=biot, poro0=poro0) + \
               poro0 * cf + eps)
    return M

def get_rock_compressibility(kd, biot, poro0):
    if np.isscalar(biot):
        return (biot - poro0) * (1 - biot) / kd
    elif np.isscalar(biot[0]):
        return (biot - poro0) * (1 - biot) / kd
    elif biot.shape[-2:] == (3, 3):
        psi = np.trace(biot, axis1=biot.ndim-2, axis2=biot.ndim-1) / 3
        return (psi - poro0) * (1 - psi) / kd
    else:
        assert False

def get_isotropic_stiffness(E, nu):
    la = nu * E / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    if np.isscalar(la):
        return np.array([[la + 2 * mu, la, la, 0, 0, 0],
                         [la, la + 2 * mu, la, 0, 0, 0],
                         [la, la, la + 2 * mu, 0, 0, 0],
                         [0, 0, 0, mu, 0, 0],
                         [0, 0, 0, 0, mu, 0],
                         [0, 0, 0, 0, 0, mu]])
    else: # If non-scalar, initialize an empty list to store matrices
        matrices = []
        # Iterate over each element in la (and mu if mu is also an array)
        for l, m in zip(la, mu):
            # Create a matrix for each pair of la and mu
            matrix = np.array([[l + 2 * m, l, l, 0, 0, 0],
                               [l, l + 2 * m, l, 0, 0, 0],
                               [l, l, l + 2 * m, 0, 0, 0],
                               [0, 0, 0, m, 0, 0],
                               [0, 0, 0, 0, m, 0],
                               [0, 0, 0, 0, 0, m]])
            matrices.append(matrix)
        # Return the list of matrices
        return matrices

#TODO add cache of discretizer, recompute if something changed

class UnstructReservoirMech(): 
    #TODO: inherit from UnstructReservoirBase to have add_well functions from there
    #TODO: create a py wrapper reservoir class UnstructReservoirCPP for C++ discretizer (flow only, MPFA)
    #TODO: create an abstract  class UnstructReservoirBase for existing Python class and UnstructReservoirCPP
    '''
    Class for Poroelasticity/ThermoPoroElasticity coupled model
    '''
    def __init__(self, timer, discretizer='mech_discretizer', thermoporoelasticity=False, fluid_vars=['p']):
        self.timer = timer
        self.discretizer_name = discretizer
        self.thermoporoelasticity = thermoporoelasticity
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        self.n_dim = 3
        self.n_dim_sq = self.n_dim * self.n_dim
        self.bc_type = bound_cond()
        self.cell_property = fluid_vars + ['ux', 'uy', 'uz']
        ne = len(fluid_vars)
        nc = ne - thermoporoelasticity
        self.p_var = self.p_bc_var = 0
        self.z_var = self.p_var + 1 if nc > 1 else None
        self.t_var = self.p_var + nc if thermoporoelasticity else None
        self.t_bc_var = self.p_var + 1 if thermoporoelasticity else None
        self.u_var = self.p_var + ne
        self.u_bc_var = self.t_bc_var + 1 if thermoporoelasticity else self.p_bc_var + 1

        if thermoporoelasticity:
            assert (discretizer == 'mech_discretizer')
        else: # poroelasticity
            if discretizer == 'pm_discretizer':
                self.u_var = self.u_bc_var = 0
                self.p_var = self.p_bc_var = self.n_dim
                self.z_var = None
                self.t_var = None
                self.cell_property = ['ux', 'uy', 'uz', 'p']

        self.n_state = ne
        self.n_vars = self.n_state + self.n_dim
        self.n_bc_vars = 1 + thermoporoelasticity + self.n_dim

    def set_equilibrium(self, zero_conduction: bool=False):
        # store original transmissibilities
        self.vol_strain_tran = np.array(self.mesh.vol_strain_tran, copy=True)
        self.vol_strain_rhs = np.array(self.mesh.vol_strain_rhs, copy=True)

        # turn off some terms for evaluation of momentum equilibrium
        vol_strain_tran = np.array(self.mesh.vol_strain_tran, copy=False)
        vol_strain_rhs = np.array(self.mesh.vol_strain_rhs, copy=False)
        vol_strain_tran[:] = 0.0
        vol_strain_rhs[:] = 0.0

        if zero_conduction:
            self.darcy_tran = np.array(self.mesh.darcy_tran, copy=True)
            self.darcy_rhs = np.array(self.mesh.darcy_rhs, copy=True)
            darcy_tran = np.array(self.mesh.darcy_tran, copy=False)
            darcy_rhs = np.array(self.mesh.darcy_rhs, copy=False)
            darcy_tran[:] = 0.0
            darcy_rhs[:] = 0.0

            if self.thermoporoelasticity:
                self.fourier_tran = np.array(self.mesh.fourier_tran, copy=True)
                fourier_tran = np.array(self.mesh.fourier_tran, copy=False)
                fourier_tran[:] = 0.0

    def turn_off_equilibrium(self, zero_conduction: bool=False):
        vol_strain_tran = np.array(self.mesh.vol_strain_tran, copy=False)
        vol_strain_rhs = np.array(self.mesh.vol_strain_rhs, copy=False)
        vol_strain_tran[:] = self.vol_strain_tran
        vol_strain_rhs[:] = self.vol_strain_rhs

        if zero_conduction:
            darcy_tran = np.array(self.mesh.darcy_tran, copy=False)
            darcy_rhs = np.array(self.mesh.darcy_rhs, copy=False)
            darcy_tran[:] = self.darcy_tran
            darcy_rhs[:] = self.darcy_rhs

            if self.thermoporoelasticity:
                fourier_tran = np.array(self.mesh.fourier_tran, copy=False)
                fourier_tran[:] = self.fourier_tran

    def init_matrix_stiffness(self, props):
        self.unstr_discr.stiffness = {}
        self.unstr_discr.stf = {}
        for id, prop in props.items():
            self.unstr_discr.stiffness[id] = prop['stiffness']
            self.unstr_discr.stf[id] = self.unstr_discr.get_stiffness_submatrices(self.unstr_discr.stiffness[id])

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
                          self.unstr_discr.bound_faces_tot,
                          self.unstr_discr.frac_cells_tot)
        self.unstr_discr.store_volume_all_cells()
        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_faces_tot

    def init_mech_discretizer(self, idata: InputData):
        self.discr_mesh = Mesh()
        self.discr_mesh.gmsh_mesh_processing(self.mesh_filename, self.domain_tags)

        self.a = np.max([node.values[0] for node in self.discr_mesh.nodes])  # max value of X coordinate
        self.b = np.max([node.values[1] for node in self.discr_mesh.nodes])  # max value of Y coordinate
        if self.thermoporoelasticity:
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

        self.conns = np.array(self.discr_mesh.conns)
        self.centroids = np.array(self.discr_mesh.centroids)
        self.adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        self.adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)

        self.ref_contact_cells = np.zeros(self.n_fracs, dtype=np.intc)

        #TODO: instead of biot and kd, calc compressibility in python and pass this to engines
        # comp_mult = (biot_cur != 0) ? (biot_cur - poro[i]) * (1 - biot_cur) / kd[i] : 1.0 / kd[i];

        #self.kd_cur = np.zeros(self.n_matrix) # corresponds to drained_compressibility in conn_mesh
        #if not hasattr(self, 'porosity'):
        #    self.porosity = 0.
        #self.porosity = self.porosity + np.zeros(self.n_matrix + self.n_fracs)

    def init_arrays(self, idata: InputData):
        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.poro = np.array(self.mesh.poro, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)
        self.bc = np.array(self.mesh.bc, copy=False)
        self.bc_prev = np.array(self.mesh.bc_prev, copy=False)
        self.bc_ref = np.array(self.mesh.bc_ref, copy=False)
        self.mesh.f.resize(self.n_vars * (self.n_fracs + self.n_matrix))
        self.f = np.array(self.mesh.f, copy=False)
        self.rock_compressibility = np.array(self.mesh.rock_compressibility, copy=False)
        self.p_ref = np.array(self.mesh.ref_pressure, copy=False)
        hcap = np.array(self.mesh.heat_capacity, copy=False)
        if self.thermoporoelasticity:
            self.t_ref = np.array(self.mesh.ref_temperature, copy=False)
            self.th_expn_poro_arr = np.array(self.mesh.th_poro, copy=False)
            
        # specify properties
        self.poro[:self.n_matrix] = self.porosity
        self.poro[self.n_matrix:] = 1  # fractures
        if self.thermoporoelasticity:
            hcap[:] = self.hcap
        self.rock_compressibility[:] = self.cs

        if self.discretizer_name == 'mech_discretizer':
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.volume[:self.n_matrix] = volumes[:self.n_matrix]  #TODO init frac volumes
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            self.p_ref[:] = self.p_init
            if self.thermoporoelasticity:
                self.t_ref[:] = self.t_init
                self.th_expn_poro_arr[:] = idata.rock.th_expn_poro
        elif self.discretizer_name == 'pm_discretizer':
            self.volume[:self.unstr_discr.mat_cells_tot] = self.unstr_discr.volume_all_cells[self.unstr_discr.frac_cells_tot:]
            for i in range(self.unstr_discr.mat_cells_tot, self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot):
                self.volume[i] = self.unstr_discr.faces[i][4].area * self.frac_apers[i-self.unstr_discr.mat_cells_tot]
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            self.bc_ref[:] = self.bc_rhs_ref
            self.p_ref[:] = self.unstr_discr.p_ref
            self.f[:] = self.unstr_discr.f

    def set_pzt_bounds(self, p, z=None, t=None):
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
            if self.z_var is not None:
                self.pz_bounds[self.z_var::self.n_state] = z
            if self.thermoporoelasticity:
                self.pz_bounds[self.t_var::self.n_state] = t
        elif self.discretizer_name == 'pm_discretizer':
            self.pz_bounds[:] = p

    def set_bounds(self, p_z_t):
        '''
        :param p_z_t: array of boundary values: pressure, compositions, temperature
        '''
        self.mesh.pz_bounds.resize(self.n_state * self.n_bounds)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.pz_bounds[:] = p_z_t

    def init_bc_rhs(self):
        if self.discretizer_name == 'mech_discretizer':
            for tag in self.domain_tags[elem_loc.BOUNDARY]:
                ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
                bc = self.boundary_conditions[tag]
                # flow
                self.bc_rhs[self.n_bc_vars * ids + self.p_bc_var] = bc['flow']['r']
                # energy
                if self.thermoporoelasticity:
                    self.bc_rhs[self.n_bc_vars * ids + self.t_bc_var] = bc['temp']['r']
                # mechanics
                for id in ids:
                    assert(self.adj_matrix_cols[self.id_sorted[id]] == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                    conn = self.conns[self.id_boundary_conns[id]]
                    n = np.array(conn.n.values)
                    conn_c = np.array(conn.c.values)
                    c1 = np.array(self.centroids[conn.elem_id1].values)
                    if n.dot(conn_c - c1) < 0: n *= -1.0
                    self.bc_rhs[self.n_bc_vars * id + self.u_bc_var:self.n_bc_vars * id + self.u_bc_var + self.n_dim] = bc['mech']['rn'] * n + bc['mech']['rt']
        elif self.discretizer_name == 'pm_discretizer':
            self.pm.bc.clear()
            for id in range(len(self.unstr_discr.bound_face_info_dict)):
                n = self.get_normal_to_bound_face(id)
                # P = np.identity(3) - np.outer(n, n)
                mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_face_info_dict[id].prop_id]['mech']
                flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_face_info_dict[id].prop_id]['flow']
                bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
                self.pm.bc.append(matrix(bc, len(bc), 1))
                self.bc_rhs[self.n_bc_vars * id + self.u_bc_var:self.n_bc_vars * id + self.u_bc_var + self.n_dim] = mech['rn'] * n + mech['rt']
                self.bc_rhs[self.n_bc_vars * id + self.p_bc_var] = flow['r']

    def init_arrays_boundary_condition(self):
        self.set_vars_pm_discretizer()

        self.bc_rhs = np.zeros(self.n_bc_vars * self.n_bounds)
        self.bc_rhs_prev = np.zeros(self.n_bc_vars * self.n_bounds)
        self.bc_rhs_ref = np.zeros(self.n_bc_vars * self.n_bounds)
        self.pz_bounds_rhs = np.zeros(self.n_state * self.n_bounds)

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
            if self.thermoporoelasticity:
                at = np.zeros(self.n_bounds)
                bt = np.zeros(self.n_bounds)

            for tag in self.domain_tags[elem_loc.BOUNDARY]:
                ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
                bc = self.boundary_conditions[tag]
                ap[ids] = bc['flow']['a']
                bp[ids] = bc['flow']['b']
                amn[ids] = bc['mech']['an']
                bmn[ids] = bc['mech']['bn']
                amt[ids] = bc['mech']['at']
                bmt[ids] = bc['mech']['bt']
                if self.thermoporoelasticity:
                    at[ids] = bc['temp']['a']
                    bt[ids] = bc['temp']['b']

            self.cpp_bc = THMBoundaryCondition()
            self.cpp_bc.flow.a = value_vector(ap)
            self.cpp_bc.flow.b = value_vector(bp)
            self.cpp_bc.mech_normal.a = value_vector(amn)
            self.cpp_bc.mech_normal.b = value_vector(bmn)
            self.cpp_bc.mech_tangen.a = value_vector(amt)
            self.cpp_bc.mech_tangen.b = value_vector(bmt)
            if self.thermoporoelasticity:
                self.cpp_bc.thermal.a = value_vector(at)
                self.cpp_bc.thermal.b = value_vector(bt)
            # to use base discretizer's class function reconstruct_pressure_gradients_per_cell
            # which doesn't know the new THMBoundaryCondition class yet
            self.cpp_flow = BoundaryCondition()
            self.cpp_flow.a = value_vector(ap)
            self.cpp_flow.b = value_vector(bp)
            if self.thermoporoelasticity:
                self.cpp_heat = BoundaryCondition()
                self.cpp_heat.a = value_vector(at)
                self.cpp_heat.b = value_vector(bt)
        elif self.discretizer_name == 'pm_discretizer':
            self.ref_contact_cells = np.zeros(self.n_fracs, dtype=np.intc)
            self.unstr_discr.p_ref = np.zeros(self.n_matrix)
            self.unstr_discr.p_ref[:] = self.p_init
            for bound_id in range(len(self.unstr_discr.bound_face_info_dict)):
                n = self.get_normal_to_bound_face(bound_id)
                P = np.identity(3) - np.outer(n, n)
                mech = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_face_info_dict[bound_id].prop_id]['mech']
                flow = self.unstr_discr.boundary_conditions[self.unstr_discr.bound_face_info_dict[bound_id].prop_id]['flow']
                # if flow['a'] == 1.0:
                #    c = self.unstr_discr.bound_face_info_dict[bound_id].centroid
                #    if c[1] > 250 and c[1] < 750: bc.extend([flow['a'], flow['b'], 0.5 * self.p_init])
                #    else: bc.extend([0.0, 1.0, 0.0])
                # else:
                bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
                self.pm.bc.append(matrix(bc, len(bc), 1))
                self.bc_rhs[4 * bound_id:4 * bound_id + 3] = mech['rn'] * n + mech['rt']  # TODO use init_bc_rhs
                self.bc_rhs[4 * bound_id + 3] = flow['r']
                self.bc_rhs_prev[4 * bound_id:4 * bound_id + 3] = np.array(
                    [0, 0, 0])  # prev can be not inited if bnd cond doesn't change
                self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
                self.bc_rhs_ref[4 * bound_id:4 * bound_id + 3] = np.array([0, 0, 0])
                self.bc_rhs_ref[4 * bound_id + 3] = flow['r']
            # self.bc_rhs_prev = np.copy(self.bc_rhs)
            self.pm.bc_prev = self.pm.bc
            self.unstr_discr.f = np.zeros(4 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot))

    def set_boundary_conditions_pm_discretizer(self):
        if self.discretizer_name == 'pm_discretizer':
            self.unstr_discr.boundary_conditions = self.boundary_conditions
            for key in self.boundary_conditions.keys():
                self.boundary_conditions[key]['cells'] = []

    def init_gravity(self, gravity_on: bool=False, gravity_coeff: float=None, gravity_direction: str ='z+'):
        '''
        sets gravity vector in discretizer
        '''
        if gravity_on:
            from scipy.constants import gravitational_constant
            if gravity_coeff is None:
                g_coeff = gravitational_constant / 1e5  # convert units
            else:
                g_coeff = gravity_coeff

            if '-' in gravity_direction: #TODO check/test
                g_coeff = -g_coeff
        else:
            g_coeff = 0.

        if gravity_direction in ['x+', 'x-']:
            grav_vec = [g_coeff, 0.0, 0.0]
        elif gravity_direction in ['y+', 'y-']:
            grav_vec = [0.0, g_coeff, 0.0]
        elif gravity_direction in ['z+', 'z-']:
            grav_vec = [0.0, 0.0, g_coeff]
        else:
            raise('Unknown gravity_direction', gravity_direction)
        grav_vec = matrix(grav_vec, 1, 3)  # n_dims=3
        if self.discretizer_name == 'mech_discretizer':
            self.discr.grav_vec = grav_vec
        elif self.discretizer_name == 'pm_discretizer':
            self.pm.grav = grav_vec

    def init_uniform_properties(self, idata: InputData):
        if self.discretizer_name == 'mech_discretizer':
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                              self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
                #self.discr.poro.append(idata.rock.porosity)  # for cell activity filtering
                if idata.rock.perm is None:
                    self.discr.perms.append(disc_matrix33(idata.rock.permx, idata.rock.permy, idata.rock.permz))
                else:
                    self.discr.perms.append(disc_matrix33(idata.rock.perm))
                stif_tmp = np.array(idata.rock.stiffness).flatten().tolist()
                self.discr.stfs.append(disc_stiffness(stif_tmp))
                self.discr.biots.append(disc_matrix33(idata.rock.biot))

                if self.thermoporoelasticity:
                    self.discr.heat_conductions.append(disc_matrix33(idata.rock.conductivity))
                    self.discr.thermal_expansions.append(disc_matrix33(idata.rock.th_expn))
        elif self.discretizer_name == 'pm_discretizer':
            for cell_id in range(self.unstr_discr.mat_cells_tot):
                cell = self.unstr_discr.mat_cell_info_dict[cell_id]
                self.pm.cell_centers.append(matrix(list(cell.centroid), cell.centroid.size, 1))
                perm = idata.rock.get_permxyz()
                if len(perm) == 3: # permx, permy, permz
                    self.pm.perms.append(engine_matrix33(perm[0], perm[1], perm[2]))
                else:
                    self.pm.perms.append(engine_matrix33(perm)) # tensor
                self.pm.biots.append(engine_matrix33(idata.rock.biot))
                self.pm.stfs.append(engine_stiffness(np.array(idata.rock.stiffness).flatten()))
        if self.thermoporoelasticity:
            self.hcap = idata.rock.heat_capacity
        self.porosity = idata.rock.porosity
        self.cs = idata.rock.compressibility

    def set_props_tags(self, idata: InputData, matrix_tags: list):
        # loop over idata.rock. objects and fill self.props, for example:
        # if idata.rock.poro=[0.2, 0.1], matrix_tags=[90,91]  =>  props = { 90: {'poro': 0.2}, 91: {'poro': 0.1}}
        self.props = {}
        for i, m in enumerate(matrix_tags):
            self.props[m] = dict()
            for k1 in idata.__dict__.keys():
                if k1 not in ['rock', 'other']:
                    continue
                sub_obj = idata.__getattribute__(k1)
                for prop in sub_obj.__dict__.keys():
                    val = sub_obj.__getattribute__(prop)
                    if val is not None:
                        if np.isscalar(val):
                            self.props[m][prop] = val
                        else:
                            self.props[m][prop] = val[i]

    def init_heterogeneous_properties(self):
        '''
        set matrix poperties using self.props[tag]
        :return:
        '''
        if self.discretizer_name == 'mech_discretizer':
            self.porosity = np.zeros(self.n_matrix + self.n_fracs)
            self.cs = np.zeros(self.n_matrix + self.n_fracs)
            self.hcap = np.zeros(self.n_matrix + self.n_fracs)
            for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                              self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
                tag = self.tags[cell_id]
                E = self.props[tag]['E']
                nu = self.props[tag]['nu']
                biot = self.props[tag]['biot']
                if 'perm' in self.props[tag].keys():
                    kx = ky = kz = self.props[tag]['perm']
                else:
                    kx, ky, kz = self.props[tag]['permx'], self.props[tag]['permy'], self.props[tag]['permz']
                kd = self.props[tag]['kd']
                poro = self.props[tag]['porosity']
                if self.thermoporoelasticity:
                    hcap = self.props[tag]['heat_capacity']
                    rcond = self.props[tag]['conductivity']
                    th_expn = self.props[tag]['th_expn']
                lam, mu = get_lambda_mu(E, nu)

                self.discr.perms.append(disc_matrix33(kx, ky, kz))
                self.discr.biots.append(disc_matrix33(biot))
                self.discr.stfs.append(disc_stiffness(lam, mu))
                if self.thermoporoelasticity:
                    self.discr.heat_conductions.append(disc_matrix33(rcond))
                    self.discr.thermal_expansions.append(disc_matrix33(th_expn))
                    self.hcap[cell_id] = hcap
                self.porosity[cell_id] = poro
                self.cs[cell_id] = get_rock_compressibility(kd=kd, biot=biot, poro0=poro)
        elif self.discretizer_name == 'pm_discretizer':
            self.cs = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.porosity = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.hcap = np.zeros(self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            for cell_id in range(self.unstr_discr.mat_cells_tot):
                cell = self.unstr_discr.mat_cell_info_dict[cell_id]
                E = self.props[cell.prop_id]['E']
                nu = self.props[cell.prop_id]['nu']
                biot = self.props[cell.prop_id]['biot']
                if 'perm' in self.props[cell.prop_id].keys():
                    kx = ky = kz = self.props[cell.prop_id]['perm']
                else:
                    kx, ky, kz = self.props[cell.prop_id]['permx'], self.props[cell.prop_id]['permy'], self.props[cell.prop_id]['permz']
                kd = self.props[cell.prop_id]['kd']
                poro = self.props[cell.prop_id]['porosity']
                lam, mu = get_lambda_mu(E, nu)
                self.pm.stfs.append(engine_stiffness(lam, mu))
                self.pm.perms.append(engine_matrix33(kx, ky, kz))
                self.pm.biots.append(engine_matrix33(biot))
                self.cs[cell_id] = get_rock_compressibility(kd=kd, biot=biot, poro0=poro)
                self.porosity[cell_id] = poro

    def set_uniform_initial_conditions(self, idata: InputData):
        self.u_init = idata.initial.initial_displacements
        self.p_init = idata.initial.initial_pressure
        self.z_init = idata.initial.initial_composition
        if self.thermoporoelasticity:
            self.t_init = idata.initial.initial_temperature
        else:
            self.t_init = None

    def set_initial_conditions_by_gradients(self):
        mesh = self.reservoir.mesh
        depth = np.array(mesh.depth, copy=False)

        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = (depth - self.idata.initial.reference_depth_for_pressure) * self.idata.initial.pressure_gradient + \
                      self.idata.initial.pressure_initial

        temperature = np.array(mesh.temperature, copy=False)
        temperature[:] = (depth - self.idata.initial.reference_depth_for_temperature) * self.idata.initial.temperature_gradient + \
                         + self.idata.initial.temperature_initial

        #print('depth:      ', depth.min(), '-', depth.max(), 'm.')
        #print('pressure:   ', pressure.min(), '-', pressure.max(), 'bars.')
        #print('temperature:', temperature.min()-273.15, '-', temperature.max()-273.15, 'C.')

    def init_reservoir_main(self, idata:InputData):
        # allocate arrays in C++ (conn_mesh)
        if self.discretizer_name == 'mech_discretizer':
            if self.thermoporoelasticity:
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
        elif self.discretizer_name == 'pm_discretizer':
            if self.thermoporoelasticity:
                print('thermoporoelasticity is not supported in', self.discretizer_name)
                assert False
            self.init_pm_discretizer()
        self.init_arrays(idata)
        self.wells = []

    def update_trans(self, dt, x):
        #self.pm.x_prev = value_vector(np.concatenate((x, self.bc_rhs_prev)))
        #self.pm.reconstruct_gradients_per_cell(dt)
        #self.pm.calc_all_fluxes(dt)
        #self.write_pm_conn_to_file(t_step=t_step)
        #self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
        #                  self.unstr_discr.mat_cells_tot, self.unstr_discr.bound_faces_tot, 0)

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

    def set_vars_pm_discretizer(self):
        # make vars with the same name as in mech_discretize to avoid code duplication
        if self.discretizer_name == 'pm_discretizer':
            self.n_matrix = self.unstr_discr.mat_cells_tot
            self.n_fracs = self.unstr_discr.frac_cells_tot
            self.n_bounds = self.unstr_discr.bound_faces_tot
            self.n_elements = self.n_matrix + self.n_fracs

    def init_wells(self):
        # # Add wells to the DARTS mesh object and sort connection (DARTS related):
        self.mesh.add_wells_mpfa(ms_well_vector(self.wells), self.P_VAR)
        if self.discretizer_name == 'mech_discretizer':
            if self.thermoporoelasticity:
                self.mesh.reverse_and_sort_pme_mech_discretizer()
            else:
                self.mesh.reverse_and_sort_pm_mech_discretizer()
        elif self.discretizer_name == 'pm_discretizer':
            self.mesh.reverse_and_sort_pm()
        #self.mesh.init_grav_coef()
        return 0

    def get_normal_to_bound_face(self, b_id):
        assert self.discretizer_name == 'pm_discretizer'
        cell = self.unstr_discr.bound_face_info_dict[b_id]
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