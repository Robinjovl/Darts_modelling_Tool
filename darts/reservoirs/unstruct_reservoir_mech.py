import numpy as np

from darts.discretizer import elem_loc
from darts.engines import conn_mesh
from darts.engines import index_vector, value_vector

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


class UnstructReservoirMech:
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
        self.mesh.pz_bounds.resize(self.n_state * self.n_bounds)
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.p_ref = np.array(self.mesh.ref_pressure, copy=False)
        self.poro[:self.n_matrix] = self.porosity
        self.poro[self.n_matrix:] = 1

        if self.discretizer_name == 'mech_discretizer':
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.volume[:self.n_matrix] = volumes[:self.n_matrix]
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            # self.biot_arr[:] = np.tile([0,0,0,
            #                             0,0,0,
            #                             0,0,0], self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            self.biot_arr[:] = self.biot_mean
            self.kd[:] = self.kd_cur
            self.pz_bounds[self.p_var::self.n_state] = self.p_init
            if self.thermoporoelacticity:
                self.pz_bounds[self.t_var::self.n_state] = self.t_init
            # self.pz_bounds[:] = self.pz_bounds
            # self.p_ref[:] = self.p_ref
            # self.f[:] = self.f
        elif self.discretizer_name == 'pm_discretizer':
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


