import os
import xml.dom.minidom

import meshio
import numpy as np
from scipy.linalg import null_space

from darts.discretizer import (
    BoundaryCondition,
    Elem,
    Mesh,
)
from darts.discretizer import Stiffness as disc_stiffness
from darts.discretizer import (
    THMBoundaryCondition,
    conn_type,
    elem_loc,
    elem_type,
    index_vector,
    matrix,
)
from darts.discretizer import matrix33 as disc_matrix33
from darts.discretizer import (
    poro_mech_discretizer,
    thermoporo_mech_discretizer,
    value_vector,
    vector_matrix33,
    vector_vector3,
)
from darts.engines import (
    Face,
)
from darts.engines import Stiffness as engine_stiffness
from darts.engines import (
    conn_mesh,
    contact,
    contact_vector,
    critical_stress,
    face_vector,
    index_vector,
    matrix,
)
from darts.engines import matrix33 as engine_matrix33
from darts.engines import (
    ms_well,
    ms_well_vector,
    pm_discretizer,
    stf_vector,
)
from darts.engines import value_vector
from darts.engines import value_vector as engine_value_vector
from darts.engines import (
    vector_face_vector,
    vector_matrix,
    vector_matrix33,
)
from darts.input.input_data import InputData
from darts.reservoirs.unstruct_reservoir import UnstructReservoir


class bound_cond:
    '''
    General representation of boundary condition: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
    '''

    def __init__(self):
        # flow
        self.NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        self.AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        self.FLOW = lambda flow: {
            'a': 0.0,
            'b': 1.0,
            'r': flow,
        }  # TODO normed to area ? units?

        # mechanics
        # for mechanical boundary conditions, Dirichlet means setting displacements [m.] and Neumann means setting load [bars]
        # roller allows sliding along the boundary, but not moving away from it, so normal displacement is zero and shear stress is zero
        #  1*displ_n + 0*stress_n = 0, 'n' means normal to the boundary face
        #  0*displ_t + 1*stress_t = 0, 't' means parallel to the boundary face
        self.ROLLER = {
            'an': 1.0,
            'bn': 0.0,
            'rn': 0.0,
            'at': 0.0,
            'bt': 1.0,
            'rt': np.array([0, 0, 0]),
        }
        self.FREE = {
            'an': 0.0,
            'bn': 1.0,
            'rn': 0.0,
            'at': 0.0,
            'bt': 1.0,
            'rt': np.array([0, 0, 0]),
        }
        self.STUCK = lambda un, ut: {
            'an': 1.0,
            'bn': 0.0,
            'rn': un,
            'at': 1.0,
            'bt': 0.0,
            'rt': np.array(ut),
        }
        # Fn, Ft are normal and tangential load
        self.LOAD = lambda Fn, Ft: {
            'an': 0.0,
            'bn': 1.0,
            'rn': Fn,
            'at': 0.0,
            'bt': 1.0,
            'rt': np.array(Ft),
        }
        # the same as ROLLER except rn is non-zero
        self.STUCK_ROLLER = lambda un: {
            'an': 1.0,
            'bn': 0.0,
            'rn': un,
            'at': 0.0,
            'bt': 1.0,
            'rt': np.array([0.0, 0.0, 0.0]),
        }
        # doesn allow shearing, but allows normal displacement and apply load in the normal direction
        self.STUCK_T_LOAD_N = lambda Fn, ut: {
            'an': 0.0,
            'bn': 1.0,
            'rn': Fn,
            'at': 1.0,
            'bt': 0.0,
            'rt': np.array(ut),
        }
        # | TYPE             |  a_n  |  b_n  |   r_n   |  a_t  |  b_t  |    r_t     |  comments          |
        # -------------------------------------------------------------------------------------------------
        # | ROLLER           |  1.0   |  0.0  |   0.0   |  0.0  |  1.0  |   [0,0,0] | un = 0, st = 0
        # | STUCK_ROLLER     |  1.0   |  0.0  |   un    |  0.0  |  1.0  |   [0,0,0] | un = Un, st = 0
        # | STUCK            |  1.0   |  0.0  |   un    |  1.0  |  0.0  |   ut      | un = Un, ut = Ut
        # | FREE             |  0.0   |  1.0  |   0.0   |  0.0  |  1.0  |   [0,0,0] | sn = 0, st = 0
        # | LOAD             |  0.0   |  1.0  |   Fn    |  0.0  |  1.0  |   Ft      | sn = Fn, st = Ft
        # | STUCK_T_LOAD_N   |  0.0   |  1.0  |   Fn    |  1.0  |  0.0  |   ut      | sn = Fn, ut = Ut


def set_domain_tags(
    matrix_tags,
    bnd_xm_tag=None,
    bnd_xp_tag=None,
    bnd_ym_tag=None,
    bnd_yp_tag=None,
    bnd_zm_tag=None,
    bnd_zp_tag=None,
    bnd_tags=None,  # list of boundary tags
    fracture_tags=[],
    frac_bnd_tags=[],
):
    '''
    :param matrix_tag: list of integers
    :param bnd_tags: list of integers
    :param fracture_tag: list of integers
    :param frac_bnd_tag: list of integers
    :return: dictionary of sets containing integer tags for each element type; dictionary of tags for 6 boundaries
    '''
    domain_tags = dict()
    domain_tags[elem_loc.MATRIX] = set(matrix_tags)
    domain_tags[elem_loc.FRACTURE] = set(fracture_tags)
    domain_tags[elem_loc.BOUNDARY] = set(bnd_tags)
    domain_tags[elem_loc.FRACTURE_BOUNDARY] = set(frac_bnd_tags)
    return domain_tags


def E_nu_from_Vp_Vs(density, Vp, Vs):
    G = density * Vs**2
    nu = 0.5 * (Vp**2 - 2 * Vs**2) / (Vp**2 - Vs**2)
    E = 2 * G * (1 + nu)
    return E, nu


def get_lambda_mu(E, nu):
    '''
    :param E: Young modulus [bars]
    :param nu: Poisson ratio
    :return: lambda and mu coefficitents for Stiffness matrix
    '''
    lam = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2.0 / (1 + nu)
    return lam, mu


def get_bulk_modulus(E, nu):
    kd = E / 3.0 / (1 - 2 * nu)
    return kd


def get_biot_modulus(biot, poro0, kd, cf):
    eps = 1e-100  # avoid divizion by zero
    M = 1.0 / (
        get_rock_compressibility(kd=kd, biot=biot, poro0=poro0) + poro0 * cf + eps
    )
    return M


def get_rock_compressibility(kd, biot, poro0):
    if np.isscalar(biot):
        return (biot - poro0) * (1 - biot) / kd
    elif np.isscalar(biot[0]):
        return (biot - poro0) * (1 - biot) / kd
    elif biot.shape[-2:] == (3, 3):
        psi = np.trace(biot, axis1=biot.ndim - 2, axis2=biot.ndim - 1) / 3
        return (psi - poro0) * (1 - psi) / kd
    else:
        assert False


def get_isotropic_stiffness(E, nu):
    la = nu * E / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    if np.isscalar(la):
        return np.array(
            [
                [la + 2 * mu, la, la, 0, 0, 0],
                [la, la + 2 * mu, la, 0, 0, 0],
                [la, la, la + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ]
        )
    else:  # If non-scalar, initialize an empty list to store matrices
        matrices = []
        # Iterate over each element in la (and mu if mu is also an array)
        for l, m in zip(la, mu):
            # Create a matrix for each pair of la and mu
            matrix = np.array(
                [
                    [l + 2 * m, l, l, 0, 0, 0],
                    [l, l + 2 * m, l, 0, 0, 0],
                    [l, l, l + 2 * m, 0, 0, 0],
                    [0, 0, 0, m, 0, 0],
                    [0, 0, 0, 0, m, 0],
                    [0, 0, 0, 0, 0, m],
                ]
            )
            matrices.append(matrix)
        # Return the list of matrices
        return matrices


# TODO add cache of discretizer, recompute if something changed


class UnstructReservoirMech:
    # TODO: inherit from UnstructReservoirBase to have add_well functions from there
    # TODO: create a py wrapper reservoir class UnstructReservoirCPP for C++ discretizer (flow only, MPFA)
    # TODO: create an abstract  class UnstructReservoirBase for existing Python class and UnstructReservoirCPP
    '''
    Class for Poroelasticity/ThermoPoroElasticity coupled model
    '''

    def __init__(
        self,
        timer,
        discretizer='mech_discretizer',
        thermoporoelasticity=False,
        fluid_vars=['p'],
    ):
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
            assert discretizer == 'mech_discretizer'
        else:  # poroelasticity
            if discretizer == 'pm_discretizer':
                self.u_var = self.u_bc_var = 0
                self.p_var = self.p_bc_var = self.n_dim
                self.z_var = None
                self.t_var = None
                self.cell_property = ['ux', 'uy', 'uz', 'p']

        self.n_state = ne
        self.n_vars = self.n_state + self.n_dim
        self.n_bc_vars = 1 + thermoporoelasticity + self.n_dim

    def set_equilibrium(self, zero_conduction: bool = False):
        if self.discretizer_name == 'pm_discretizer':
            # store original transmissibilities
            self.tran = np.array(self.mesh.tran, copy=True)
            self.rhs = np.array(self.mesh.rhs, copy=True)
            self.tran_biot = np.array(self.mesh.tran_biot, copy=True)
            self.rhs_biot = np.array(self.mesh.rhs_biot, copy=True)

            # turn off some terms for evaluation of momentum equilibrium
            self.apply_geomechanics_mode(mode=1)

            self.unstr_discr.f[3::4] = 0.0  # self.p_init - self.unstr_discr.p_ref[:]
            self.f[:] = self.unstr_discr.f
        elif self.discretizer_name == 'mech_discretizer':
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

    def turn_off_equilibrium(self, zero_conduction: bool = False):
        if self.discretizer_name == 'pm_discretizer':
            tran = np.array(self.mesh.tran, copy=False)
            rhs = np.array(self.mesh.rhs, copy=False)
            tran_biot = np.array(self.mesh.tran_biot, copy=False)
            rhs_biot = np.array(self.mesh.rhs_biot, copy=False)
            tran[12::16] = self.tran[12::16]
            tran[13::16] = self.tran[13::16]
            tran[14::16] = self.tran[14::16]
            tran[15::16] = self.tran[15::16]
            rhs[3::4] = self.rhs[3::4]

            tran_biot[12::16] = self.tran_biot[12::16]
            tran_biot[13::16] = self.tran_biot[13::16]
            tran_biot[14::16] = self.tran_biot[14::16]
            tran_biot[15::16] = self.tran_biot[15::16]

            rhs_biot[3::4] = self.rhs_biot[3::4]

        elif self.discretizer_name == 'mech_discretizer':
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

    def apply_geomechanics_mode(self, physics=None, mode: int = 0):
        '''
        :param physics: None if mode is 0 or 1
        :param mode:
            0 - nullify biot terms,
            1 - nullify biot and flow terms,
            2 - nullify biot and flow terms and set physics.engine.geomechanics_mode array to 1
        :return:
        '''
        if self.discretizer_name == 'pm_discretizer':
            cell_m = np.array(self.mesh.block_m, copy=False)
            cell_p = np.array(self.mesh.block_p, copy=False)
            offset = np.array(self.mesh.offset, copy=False)
            tran = np.array(self.mesh.tran, copy=False)
            tran_biot = np.array(self.mesh.tran_biot, copy=False)
            rhs = np.array(self.mesh.rhs, copy=False)
            rhs_biot = np.array(self.mesh.rhs_biot, copy=False)

            if mode > 0:
                if mode == 2:
                    assert (
                        physics is not None
                    ), 'physics should be passed when mode is 2'
                    geom_mode = np.array(physics.engine.geomechanics_mode, copy=False)
                    geom_mode[:] = 1
                tran[12::16] = 0.0
                tran[13::16] = 0.0
                tran[14::16] = 0.0
                tran[15::16] = 0.0
                rhs[3::4] = 0.0

            tran_biot[12::16] = 0.0
            tran_biot[13::16] = 0.0
            tran_biot[14::16] = 0.0
            tran_biot[15::16] = 0.0
            rhs_biot[3::4] = 0.0

        elif self.discretizer_name == 'mech_discretizer':
            assert False, 'Not implemented'

    def init_matrix_stiffness(self, props):
        self.unstr_discr.stiffness = {}
        self.unstr_discr.stf = {}
        for id, prop in props.items():
            self.unstr_discr.stiffness[id] = prop['stiffness']
            self.unstr_discr.stf[id] = self.unstr_discr.get_stiffness_submatrices(
                self.unstr_discr.stiffness[id]
            )

    def init_pm_discretizer(self):
        self.unstr_discr.x_new = np.ones(
            (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot, 4)
        )
        self.unstr_discr.x_new[:, 0] = self.u_init[0]
        self.unstr_discr.x_new[:, 1] = self.u_init[1]
        self.unstr_discr.x_new[:, 2] = self.u_init[2]
        self.unstr_discr.x_new[:, 3] = self.p_init
        dt = 0.0
        self.pm.x_prev = value_vector(
            np.concatenate((self.unstr_discr.x_new.flatten(), self.bc_rhs_prev))
        )
        self.pm.init(
            self.unstr_discr.mat_cells_tot,
            self.unstr_discr.frac_cells_tot,
            index_vector(self.ref_contact_cells),
        )
        self.pm.reconstruct_gradients_per_cell(dt)
        self.pm.calc_all_fluxes_once(dt)

        self.mesh.init_pm(
            self.pm.cell_m,
            self.pm.cell_p,
            self.pm.stencil,
            self.pm.offset,
            self.pm.tran,
            self.pm.rhs,
            self.pm.tran_biot,
            self.pm.rhs_biot,
            self.unstr_discr.mat_cells_tot,
            self.unstr_discr.bound_faces_tot,
            self.unstr_discr.frac_cells_tot,
        )
        self.unstr_discr.store_volume_all_cells()
        self.unstr_discr.store_depth_all_cells()
        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_faces_tot

    def init_mech_discretizer(self, idata: InputData):
        self.discr_mesh = Mesh()
        self.discr_mesh.gmsh_mesh_processing(self.mesh_filename, self.domain_tags)

        self.a = np.max(
            [node.values[0] for node in self.discr_mesh.nodes]
        )  # max value of X coordinate
        self.b = np.max(
            [node.values[1] for node in self.discr_mesh.nodes]
        )  # max value of Y coordinate
        if self.thermoporoelasticity:
            self.discr = thermoporo_mech_discretizer()
        else:
            self.discr = poro_mech_discretizer()
        self.tags = np.array(self.discr_mesh.tags, copy=False)
        self.discr.set_mesh(self.discr_mesh)
        self.discr.init()

        self.n_matrix = (
            self.discr_mesh.region_ranges[elem_loc.MATRIX][1]
            - self.discr_mesh.region_ranges[elem_loc.MATRIX][0]
        )
        self.n_fracs = (
            self.discr_mesh.region_ranges[elem_loc.FRACTURE][1]
            - self.discr_mesh.region_ranges[elem_loc.FRACTURE][0]
        )
        self.n_bounds = (
            self.discr_mesh.region_ranges[elem_loc.BOUNDARY][1]
            - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
        )

        self.conns = np.array(self.discr_mesh.conns)
        self.centroids = np.array(self.discr_mesh.centroids)
        self.adj_matrix_cols = np.array(self.discr_mesh.adj_matrix_cols, copy=False)
        self.adj_matrix = np.array(self.discr_mesh.adj_matrix, copy=False)

        self.ref_contact_cells = np.zeros(self.n_fracs, dtype=np.intc)

        # TODO: instead of biot and kd, calc compressibility in python and pass this to engines
        # comp_mult = (biot_cur != 0) ? (biot_cur - poro[i]) * (1 - biot_cur) / kd[i] : 1.0 / kd[i];

        # self.kd_cur = np.zeros(self.n_matrix) # corresponds to drained_compressibility in conn_mesh
        # if not hasattr(self, 'porosity'):
        #    self.porosity = 0.
        # self.porosity = self.porosity + np.zeros(self.n_matrix + self.n_fracs)

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
        self.poro[: self.n_matrix] = self.porosity
        self.poro[self.n_matrix :] = 1  # fractures
        if self.thermoporoelasticity:
            hcap[:] = self.hcap
        self.rock_compressibility[:] = self.cs

        frac_apers = self.get_frac_apers(idata)

        if self.discretizer_name == 'mech_discretizer':
            volumes = np.array(self.discr_mesh.volumes, copy=False)
            self.volume[: self.n_matrix] = volumes[
                : self.n_matrix
            ]  # TODO init frac volumes
            self.bc_prev[:] = self.bc_rhs_prev
            self.bc[:] = self.bc_rhs
            self.p_ref[:] = self.p_init
            if self.thermoporoelasticity:
                self.t_ref[:] = self.t_init
                self.th_expn_poro_arr[:] = idata.rock.th_expn_poro
        elif self.discretizer_name == 'pm_discretizer':
            self.volume[: self.unstr_discr.mat_cells_tot] = (
                self.unstr_discr.volume_all_cells[self.unstr_discr.frac_cells_tot :]
            )
            if frac_apers is not None:  # set volume for fractures
                for i in range(
                    self.unstr_discr.mat_cells_tot,
                    self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot,
                ):
                    self.volume[i] = (
                        self.unstr_discr.faces[i][4].area
                        * frac_apers[i - self.unstr_discr.mat_cells_tot]
                    )
            # exclude fractures
            self.bc_prev[: 4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs_prev
            self.bc[: 4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs
            self.bc_ref[: 4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs_ref

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
            self.pz_bounds[self.p_var :: self.n_state] = p
            if self.z_var is not None:
                self.pz_bounds[self.z_var :: self.n_state] = z
            if self.thermoporoelasticity:
                self.pz_bounds[self.t_var :: self.n_state] = t
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
                ids = (
                    np.where(self.tags == tag)[0]
                    - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
                )
                bc = self.boundary_conditions[tag]
                # flow
                self.bc_rhs[self.n_bc_vars * ids + self.p_bc_var] = bc['flow']['r']
                # energy
                if self.thermoporoelasticity:
                    self.bc_rhs[self.n_bc_vars * ids + self.t_bc_var] = bc['temp']['r']
                # mechanics
                for id in ids:
                    assert (
                        self.adj_matrix_cols[self.id_sorted[id]]
                        == id + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
                    )
                    conn = self.conns[self.id_boundary_conns[id]]
                    n = np.array(conn.n.values)
                    conn_c = np.array(conn.c.values)
                    c1 = np.array(self.centroids[conn.elem_id1].values)
                    if n.dot(conn_c - c1) < 0:
                        n *= -1.0
                    self.bc_rhs[
                        self.n_bc_vars * id
                        + self.u_bc_var : self.n_bc_vars * id
                        + self.u_bc_var
                        + self.n_dim
                    ] = (bc['mech']['rn'] * n + bc['mech']['rt'])
        elif self.discretizer_name == 'pm_discretizer':
            self.pm.bc.clear()
            for id in range(len(self.unstr_discr.bound_face_info_dict)):
                n = self.get_normal_to_bound_face(id)
                # P = np.identity(3) - np.outer(n, n)
                mech = self.unstr_discr.boundary_conditions[
                    self.unstr_discr.bound_face_info_dict[id].prop_id
                ]['mech']
                flow = self.unstr_discr.boundary_conditions[
                    self.unstr_discr.bound_face_info_dict[id].prop_id
                ]['flow']
                bc = [
                    mech['an'],
                    mech['bn'],
                    mech['at'],
                    mech['bt'],
                    flow['a'],
                    flow['b'],
                ]
                self.pm.bc.append(matrix(bc, len(bc), 1))
                self.bc_rhs[
                    self.n_bc_vars * id
                    + self.u_bc_var : self.n_bc_vars * id
                    + self.u_bc_var
                    + self.n_dim
                ] = (mech['rn'] * n + mech['rt'])
                self.bc_rhs[self.n_bc_vars * id + self.p_bc_var] = flow['r']

    def init_arrays_boundary_condition(self):
        self.set_vars_pm_discretizer()

        self.bc_rhs = np.zeros(self.n_bc_vars * self.n_bounds)
        self.bc_rhs_prev = np.zeros(self.n_bc_vars * self.n_bounds)
        self.bc_rhs_ref = np.zeros(self.n_bc_vars * self.n_bounds)
        self.pz_bounds_rhs = np.zeros(self.n_state * self.n_bounds)

        if self.discretizer_name == 'mech_discretizer':
            # mapping boundary connections
            self.id_sorted = np.argsort(self.adj_matrix_cols)[
                -self.n_bounds :
            ]  # store it to self. as it will be used in init_bc_rhs() further
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
                ids = (
                    np.where(self.tags == tag)[0]
                    - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
                )
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
            self.unstr_discr.p_ref = np.zeros(self.n_matrix + self.n_fracs)
            self.unstr_discr.p_ref[:] = self.p_init
            for bound_id in range(len(self.unstr_discr.bound_face_info_dict)):
                n = self.get_normal_to_bound_face(bound_id)
                P = np.identity(3) - np.outer(n, n)
                mech = self.unstr_discr.boundary_conditions[
                    self.unstr_discr.bound_face_info_dict[bound_id].prop_id
                ]['mech']
                flow = self.unstr_discr.boundary_conditions[
                    self.unstr_discr.bound_face_info_dict[bound_id].prop_id
                ]['flow']
                # if flow['a'] == 1.0:
                #    c = self.unstr_discr.bound_face_info_dict[bound_id].centroid
                #    if c[1] > 250 and c[1] < 750: bc.extend([flow['a'], flow['b'], 0.5 * self.p_init])
                #    else: bc.extend([0.0, 1.0, 0.0])
                # else:
                bc = [
                    mech['an'],
                    mech['bn'],
                    mech['at'],
                    mech['bt'],
                    flow['a'],
                    flow['b'],
                ]
                self.pm.bc.append(matrix(bc, len(bc), 1))
                self.bc_rhs[4 * bound_id : 4 * bound_id + 3] = (
                    mech['rn'] * n + mech['rt']
                )  # TODO use init_bc_rhs
                self.bc_rhs[4 * bound_id + 3] = flow['r']
                self.bc_rhs_prev[4 * bound_id : 4 * bound_id + 3] = np.array(
                    [0, 0, 0]
                )  # prev can be not inited if bnd cond doesn't change
                self.bc_rhs_prev[4 * bound_id + 3] = flow['r']
                self.bc_rhs_ref[4 * bound_id : 4 * bound_id + 3] = np.array([0, 0, 0])
                self.bc_rhs_ref[4 * bound_id + 3] = flow['r']
            # self.bc_rhs_prev = np.copy(self.bc_rhs)
            self.pm.bc_prev = self.pm.bc
            self.unstr_discr.f = np.zeros(
                4 * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            )

    def init_bc_fractures(self):
        assert self.discretizer_name == 'pm_discretizer'
        for bound_id in range(
            len(self.unstr_discr.bound_face_info_dict),
            self.unstr_discr.bound_faces_tot + self.unstr_discr.frac_bound_faces_tot,
        ):
            mech = self.unstr_discr.boundary_conditions[
                self.unstr_discr.frac_bound_face_info_dict[bound_id].prop_id
            ]['mech']
            flow = self.unstr_discr.boundary_conditions[
                self.unstr_discr.frac_bound_face_info_dict[bound_id].prop_id
            ]['flow']
            bc = [mech['an'], mech['bn'], mech['at'], mech['bt'], flow['a'], flow['b']]
            self.pm.bc.append(matrix(bc, len(bc), 1))

    def set_boundary_conditions_pm_discretizer(self):
        if self.discretizer_name == 'pm_discretizer':
            self.unstr_discr.boundary_conditions = self.boundary_conditions
            for key in self.boundary_conditions.keys():
                self.boundary_conditions[key]['cells'] = []

    def init_gravity(
        self,
        gravity_on: bool = False,
        gravity_coeff: float = None,
        gravity_direction: str = 'z+',
    ):
        '''
        sets gravity vector in discretizer
        '''
        if gravity_on:
            from scipy.constants import gravitational_constant

            if gravity_coeff is None:
                g_coeff = gravitational_constant / 1e5  # convert units
            else:
                g_coeff = gravity_coeff

            if '-' in gravity_direction:  # TODO check/test
                g_coeff = -g_coeff
        else:
            g_coeff = 0.0

        if gravity_direction in ['x+', 'x-']:
            grav_vec = [g_coeff, 0.0, 0.0]
        elif gravity_direction in ['y+', 'y-']:
            grav_vec = [0.0, g_coeff, 0.0]
        elif gravity_direction in ['z+', 'z-']:
            grav_vec = [0.0, 0.0, g_coeff]
        else:
            raise ('Unknown gravity_direction', gravity_direction)
        grav_vec = matrix(grav_vec, 1, 3)  # n_dims=3
        if self.discretizer_name == 'mech_discretizer':
            self.discr.grav_vec = grav_vec
        elif self.discretizer_name == 'pm_discretizer':
            self.pm.grav = grav_vec

    def init_uniform_properties(self, idata: InputData):
        if self.discretizer_name == 'mech_discretizer':
            for i, cell_id in enumerate(
                range(
                    self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                    self.discr_mesh.region_ranges[elem_loc.MATRIX][1],
                )
            ):
                # self.discr.poro.append(idata.rock.porosity)  # for cell activity filtering
                if idata.rock.perm is None:
                    self.discr.perms.append(
                        disc_matrix33(
                            idata.rock.permx, idata.rock.permy, idata.rock.permz
                        )
                    )
                else:
                    self.discr.perms.append(disc_matrix33(idata.rock.perm))
                stif_tmp = np.array(idata.rock.stiffness).flatten().tolist()
                self.discr.stfs.append(disc_stiffness(stif_tmp))
                self.discr.biots.append(disc_matrix33(idata.rock.biot))

                if self.thermoporoelasticity:
                    self.discr.heat_conductions.append(
                        disc_matrix33(idata.rock.conductivity)
                    )
                    self.discr.thermal_expansions.append(
                        disc_matrix33(idata.rock.th_expn)
                    )
        elif self.discretizer_name == 'pm_discretizer':
            for cell_id in range(self.unstr_discr.mat_cells_tot):
                cell = self.unstr_discr.mat_cell_info_dict[cell_id]
                self.pm.cell_centers.append(
                    matrix(list(cell.centroid), cell.centroid.size, 1)
                )
                perm = idata.rock.get_permxyz()
                if len(perm) == 3:  # permx, permy, permz
                    self.pm.perms.append(engine_matrix33(perm[0], perm[1], perm[2]))
                else:
                    self.pm.perms.append(engine_matrix33(perm))  # tensor
                self.pm.biots.append(engine_matrix33(idata.rock.biot))
                self.pm.stfs.append(
                    engine_stiffness(np.array(idata.rock.stiffness).flatten())
                )
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
                    else:
                        if (
                            prop in ['permx', 'permy', 'permz']
                            and 'perm' in self.props[m]
                        ):
                            pass
                        else:
                            print('in set_props_tags: ', prop + ' is None')
                            exit(1)

    def init_heterogeneous_properties(self):
        '''
        set matrix poperties using self.props[tag]
        :return:
        '''
        if self.discretizer_name == 'mech_discretizer':
            self.porosity = np.zeros(self.n_matrix + self.n_fracs)
            self.cs = np.zeros(self.n_matrix + self.n_fracs)
            self.hcap = np.zeros(self.n_matrix + self.n_fracs)
            for i, cell_id in enumerate(
                range(
                    self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                    self.discr_mesh.region_ranges[elem_loc.MATRIX][1],
                )
            ):
                tag = self.tags[cell_id]
                E = self.props[tag]['E']
                nu = self.props[tag]['nu']
                biot = self.props[tag]['biot']
                if 'perm' in self.props[tag].keys():
                    kx = ky = kz = self.props[tag]['perm']
                else:
                    kx, ky, kz = (
                        self.props[tag]['permx'],
                        self.props[tag]['permy'],
                        self.props[tag]['permz'],
                    )
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
                self.cs[cell_id] = get_rock_compressibility(
                    kd=kd, biot=biot, poro0=poro
                )
        elif self.discretizer_name == 'pm_discretizer':
            self.cs = np.zeros(
                self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot
            )
            self.porosity = np.zeros(
                self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot
            )
            self.hcap = np.zeros(
                self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot
            )
            for cell_id in range(self.unstr_discr.mat_cells_tot):
                cell = self.unstr_discr.mat_cell_info_dict[cell_id]
                E = self.props[cell.prop_id]['E']
                nu = self.props[cell.prop_id]['nu']
                biot = self.props[cell.prop_id]['biot']
                if 'perm' in self.props[cell.prop_id].keys():
                    kx = ky = kz = self.props[cell.prop_id]['perm']
                else:
                    kx, ky, kz = (
                        self.props[cell.prop_id]['permx'],
                        self.props[cell.prop_id]['permy'],
                        self.props[cell.prop_id]['permz'],
                    )
                kd = self.props[cell.prop_id]['kd']
                poro = self.props[cell.prop_id]['porosity']
                lam, mu = get_lambda_mu(E, nu)
                self.pm.stfs.append(engine_stiffness(lam, mu))
                self.pm.perms.append(engine_matrix33(kx, ky, kz))
                self.pm.biots.append(engine_matrix33(biot))
                self.cs[cell_id] = get_rock_compressibility(
                    kd=kd, biot=biot, poro0=poro
                )
                self.porosity[cell_id] = poro

    def set_uniform_initial_conditions(self, idata: InputData):
        self.u_init = idata.initial.initial_displacements
        self.p_init = idata.initial.initial_pressure
        self.z_init = idata.initial.initial_composition
        if self.thermoporoelasticity:
            self.t_init = idata.initial.initial_temperature
        else:
            self.t_init = None

    def init_reservoir_main(self, idata: InputData):
        # allocate arrays in C++ (conn_mesh)
        if self.discretizer_name == 'mech_discretizer':
            if self.thermoporoelasticity:
                self.mesh.init_pme_mech_discretizer(
                    self.discr.cell_m,
                    self.discr.cell_p,
                    self.discr.flux_stencil,
                    self.discr.flux_offset,
                    self.discr.hooke,
                    self.discr.hooke_rhs,
                    self.discr.biot_traction,
                    self.discr.biot_traction_rhs,
                    self.discr.darcy,
                    self.discr.darcy_rhs,
                    self.discr.biot_vol_strain,
                    self.discr.biot_vol_strain_rhs,
                    self.discr.thermal_traction,
                    self.discr.fourier,
                    self.n_matrix,
                    self.n_bounds,
                    self.n_fracs,
                )
            else:
                self.mesh.init_pm_mech_discretizer(
                    self.discr.cell_m,
                    self.discr.cell_p,
                    self.discr.flux_stencil,
                    self.discr.flux_offset,
                    self.discr.hooke,
                    self.discr.hooke_rhs,
                    self.discr.biot_traction,
                    self.discr.biot_traction_rhs,
                    self.discr.darcy,
                    self.discr.darcy_rhs,
                    self.discr.biot_vol_strain,
                    self.discr.biot_vol_strain_rhs,
                    self.n_matrix,
                    self.n_bounds,
                    self.n_fracs,
                )
        elif self.discretizer_name == 'pm_discretizer':
            if self.thermoporoelasticity:
                print('thermoporoelasticity is not supported in', self.discretizer_name)
                assert False
            self.init_pm_discretizer()
        self.init_arrays(idata)
        self.wells = []

    def update_trans(self, dt, x):
        if self.discretizer_name == 'mech_discretizer':
            self.bc[:] = self.bc_rhs
            self.bc_prev[:] = self.bc_rhs_prev
        elif self.discretizer_name == 'pm_discretizer':
            # update transient sources / sinks
            self.f[:] = self.unstr_discr.f
            # update boundaries at n+1 / n timesteps
            self.bc[: 4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs
            self.bc_prev[: 4 * self.unstr_discr.bound_faces_tot] = self.bc_rhs_prev

    def update(self, dt, time):
        # update local array
        self.bc_rhs_prev = np.copy(self.bc_rhs)
        if self.discretizer_name == 'pm_discretizer':
            self.pm.bc_prev = self.pm.bc

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
        # self.mesh.init_grav_coef()
        return 0

    def get_normal_to_bound_face(self, b_id):
        assert self.discretizer_name == 'pm_discretizer'
        cell = self.unstr_discr.bound_face_info_dict[b_id]
        cells = [self.unstr_discr.mat_cells_to_node[pt] for pt in cell.nodes_to_cell]
        cell_id = next(iter(set(cells[0]).intersection(*cells)))
        for face in self.unstr_discr.faces[cell_id].values():
            if face.cell_id1 == face.cell_id2 and face.face_id2 == b_id:
                t_face = (
                    cell.centroid
                    - self.unstr_discr.mat_cell_info_dict[cell_id].centroid
                )
                n = face.n
                if np.inner(t_face, n) < 0:
                    n = -n
                return n

    def init_faces_centers_pm_discretizer(self):
        assert self.discretizer_name == 'pm_discretizer'
        for cell_id in range(self.unstr_discr.mat_cells_tot):
            faces = self.unstr_discr.faces[cell_id]
            fs = face_vector()
            for face_id in range(len(faces)):
                face = faces[face_id]
                fs.append(
                    Face(
                        face.type.value,
                        face.cell_id1,
                        face.cell_id2,
                        face.face_id1,
                        face.face_id2,
                        face.area,
                        list(face.n),
                        list(face.centroid),
                        index_vector(face.pts_id),
                    )
                )
            self.pm.faces.append(fs)
            cell = self.unstr_discr.mat_cell_info_dict[cell_id]
            self.pm.cell_centers.append(
                matrix(list(cell.centroid), cell.centroid.size, 1)
            )

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
        # self.check_positive_negative_sides()
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
                    row += (
                        '\t'
                        + str(self.pm.stencil[j])
                        + '\t['
                        + ', '.join(
                            [
                                '{:.5e}'.format(self.pm.tran[n])
                                for n in range(
                                    (j * block_size + k) * block_size,
                                    (j * block_size + k + 1) * block_size,
                                )
                            ]
                        )
                        + str(']')
                    )
                f.write(row + '\n')
            # Biot
            for k in range(block_size):
                row = (
                    'b'
                    + str(k)
                    + '\t{:.5e}'.format(self.pm.rhs_biot[i * block_size + k])
                )
                for j in range(self.pm.offset[i], self.pm.offset[i + 1]):
                    row += (
                        '\t'
                        + str(self.pm.stencil[j])
                        + '\t['
                        + ', '.join(
                            [
                                '{:.5e}'.format(self.pm.tran_biot[n])
                                for n in range(
                                    (j * block_size + k) * block_size,
                                    (j * block_size + k + 1) * block_size,
                                )
                            ]
                        )
                        + str(']')
                    )
                f.write(row + '\n')
            # if self.pm.cell_p[i] < self.unstr_discr.mat_cells_tot:
            #     st = np.array(self.pm.stencil[self.pm.offset[i]:self.pm.offset[i+1]],dtype=np.intp)
            #     all_trans_biot = np.array(self.pm.tran_biot)[(self.pm.offset[i] * block_size) * block_size: (self.pm.offset[i + 1] * block_size) * block_size].reshape(self.pm.offset[i + 1] - self.pm.offset[i], block_size, block_size)
            #     sum = np.sum(all_trans_biot, axis=0)
            #     #sum_no_bound = np.sum(all_trans[st < self.unstr_discr.mat_cells_tot], axis=0)
            #     assert((abs(sum[:3,:3]) < 1.E-10).all())
        f.close()

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
        well.perforations = well.perforations + [
            (well_block, res_block, well_index, 0.0)
        ]
        return 0

    def get_props_over_output(self, property_array, ith_step, engine):
        if self.discretizer_name == 'mesh_discretizer':
            return None
        elif self.discretizer_name == 'pm_discretizer':
            n_vars = 4
            n_dim = 3
            fluxes = np.array(engine.fluxes, copy=False)
            fluxes_biot = np.array(engine.fluxes_biot, copy=False)
            cell_m = np.array(self.mesh.block_m, copy=False)
            cell_p = np.array(self.mesh.block_p, copy=False)

            # S_eng = vector_matrix(engine.contacts[0].S)
            # frac_prop = property_array[n_vars * self.unstr_discr.mat_cells_tot:n_vars * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)].reshape(self.unstr_discr.frac_cells_tot, n_vars)
            # fstress = np.array(engine.contacts[0].fault_stress, copy=False)

            S = np.zeros((n_dim, n_dim))
            frac_data = {}
            # frac_data['tag'] = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intp)
            frac_data['f_local'] = np.zeros((self.unstr_discr.output_face_tot, n_dim))

            ref_id = self.unstr_discr.output_face_to_face[0]
            ref_face = self.unstr_discr.faces[ref_id[0]][ref_id[1]]
            S[1:n_dim] = null_space(np.array([ref_face.n])).T
            S[0] = ref_face.n
            for face_id, ids in self.unstr_discr.output_face_to_face.items():
                # cell_id -= self.unstr_discr.mat_cells_tot
                # frac_data['tag'][cell_id] = int(cell.prop_id)
                face = self.unstr_discr.faces[ids[0]][ids[1]]
                sign = np.sign(
                    (
                        self.unstr_discr.mat_cell_info_dict[face.cell_id2].centroid
                        - self.unstr_discr.mat_cell_info_dict[face.cell_id1].centroid
                    ).dot(ref_face.n)
                )
                flux_ids = np.argwhere(
                    np.logical_and(cell_m == face.cell_id1, cell_p == face.cell_id2)
                )[0]
                flux = (
                    sign
                    * fluxes[n_vars * flux_ids[0] : n_vars * flux_ids[0] + n_dim]
                    / face.area
                )

                if len(flux_ids):
                    frac_data['f_local'][face_id] = S.dot(flux)
                else:
                    return 0

            for face_id in range(self.unstr_discr.output_face_tot):
                print(
                    str(face_id + self.unstr_discr.mat_cells_tot)
                    + ' '
                    + str(frac_data['f_local'][face_id][0] * 1.0e5)
                    + ' '
                    + str(frac_data['f_local'][face_id][1] * 1.0e5)
                    + ' '
                    + str(frac_data['f_local'][face_id][2] * 1.0e5)
                )

            return frac_data

    def get_fault_props(self, property_array, ith_step, engine):
        if self.discretizer_name == 'mesh_discretizer':
            return None
        elif self.discretizer_name == 'pm_discretizer':
            n_vars = 4
            n_dim = 3
            fluxes = np.array(engine.fluxes, copy=False)
            fluxes_biot = np.array(engine.fluxes_biot, copy=False)
            S_eng = vector_matrix(engine.contacts[0].S)
            frac_prop = property_array[
                n_vars
                * self.unstr_discr.mat_cells_tot : n_vars
                * (self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot)
            ].reshape(self.unstr_discr.frac_cells_tot, n_vars)
            fstress = np.array(engine.contacts[0].fault_stress, copy=False)

            frac_data = {}
            # frac_data['tag'] = np.zeros(self.unstr_discr.frac_cells_tot, dtype=np.intp)
            frac_data['g_local'] = np.zeros((self.unstr_discr.frac_cells_tot, n_dim))
            frac_data['f_local'] = np.zeros((self.unstr_discr.frac_cells_tot, n_dim))
            frac_data['mu'] = np.array(engine.contacts[0].mu, copy=False)

            for cell_id, cell in self.unstr_discr.frac_cell_info_dict.items():
                cell_id -= self.unstr_discr.mat_cells_tot
                # frac_data['tag'][cell_id] = int(cell.prop_id)
                face = self.unstr_discr.faces[cell_id + self.unstr_discr.mat_cells_tot][
                    4
                ]
                S = np.array(S_eng[cell_id].values).reshape((n_dim, n_dim))
                f = fstress[n_dim * cell_id : n_dim * (cell_id + 1)] / face.area
                frac_data['f_local'][cell_id] = S.dot(f)
                frac_data['g_local'][cell_id] = S.dot(frac_prop[cell_id, :n_dim])

            phi = np.array(engine.contacts[0].phi, copy=False)
            # states = phi > 0
            frac_data['phi'] = phi

            return frac_data

    def add_initial_properties(self, cell_data):
        pass

    def write_to_vtk(self, output_directory, ith_step, engine, dt=0.0):
        if self.discretizer_name == 'mech_discretizer':
            self.write_to_vtk_mech_discretizer(output_directory, ith_step, engine, dt)
        elif self.discretizer_name == 'pm_discretizer':
            self.write_to_vtk_pm_discretizer(output_directory, ith_step, engine, dt)
        time = engine.t if ith_step > 0 else 0.0
        self.write_pvd_file(ith_step, time, output_directory)

    def write_to_vtk_mech_discretizer(self, output_directory, ith_step, engine, dt):
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
                    if self.cell_property[i] not in cell_data:
                        cell_data[self.cell_property[i]] = []
                    cell_data[self.cell_property[i]].append(
                        property_array[i : self.n_vars * self.n_matrix : self.n_vars]
                    )

                # Add post-processed data to dictionary
                if 'velocity' not in cell_data:
                    cell_data['velocity'] = []
                cell_data['velocity'].append(
                    np.zeros((self.n_matrix, 3), dtype=np.float64)
                )
                for i in range(3):
                    cell_data['velocity'][-1][:, i] = darcy_velocities[i::3]
                if 'stress' not in cell_data:
                    cell_data['stress'] = []
                cell_data['stress'].append(
                    np.zeros((self.n_matrix, 6), dtype=np.float64)
                )
                if 'tot_stress' not in cell_data:
                    cell_data['tot_stress'] = []
                cell_data['tot_stress'].append(
                    np.zeros((self.n_matrix, 6), dtype=np.float64)
                )
                for i in range(6):
                    cell_data['stress'][-1][:, i] = effective_stresses[i::6]
                    cell_data['tot_stress'][-1][:, i] = total_stresses[i::6]

            geom_id += 1

        if ith_step == 0:
            self.add_initial_properties(cell_data)

        # Store solution for each time-step:
        mesh = meshio.Mesh(Mesh.points, Mesh.cells, cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtu".format(output_directory, ith_step), mesh)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        return 0

    def write_to_vtk_pm_discretizer(self, output_directory, ith_step, engine, dt):
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
        # vels = self.reconstruct_velocities(fluxes[physics.engine.P_VAR::physics.engine.N_VARS],
        #                                  fluxes_biot[physics.engine.P_VAR::physics.engine.N_VARS])
        self.mech_operators.eval_porosities(engine.X, self.mesh.bc)
        self.mech_operators.eval_stresses(
            engine.fluxes,
            engine.fluxes_biot,
            engine.X,
            self.mesh.bc,
            engine.op_vals_arr,
        )
        # else:
        #    self.mech_operators.eval_porosities(physics.engine.X, self.mesh.bc_prev)
        #    self.mech_operators.eval_stresses(physics.engine.X, self.mesh.bc_prev, physics.engine.op_vals_arr)

        # Matrix
        Mesh.cells = []
        cell_data = {}
        start_geom_cell_id = 0
        for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
            if ith_geometry in available_matrix_geometries:
                Mesh.cells.append(
                    next(
                        cell
                        for cell in self.unstr_discr.mesh_data.cells
                        if cell.type == ith_geometry
                    )
                )
                # Add matrix data to dictionary:
                cell_size = self.unstr_discr.mesh_data.cells_dict[ith_geometry].shape[0]
                for i in range(props_num):
                    if cell_property[i] not in cell_data:
                        cell_data[cell_property[i]] = []
                    cell_data[cell_property[i]].append(
                        property_array[
                            props_num * start_geom_cell_id
                            + i : props_num
                            * (cell_size + start_geom_cell_id) : props_num
                        ]
                    )

                if 'eps_vol' not in cell_data:
                    cell_data['eps_vol'] = []
                if 'porosity' not in cell_data:
                    cell_data['porosity'] = []
                if 'stress' not in cell_data:
                    cell_data['stress'] = []
                if 'tot_stress' not in cell_data:
                    cell_data['tot_stress'] = []

                cell_data['eps_vol'].append(
                    np.array(self.mech_operators.eps_vol, copy=False)
                )
                cell_data['porosity'].append(
                    np.array(self.mech_operators.porosities, copy=False)
                )
                cell_data['stress'].append(np.zeros((cell_size, 6), dtype=np.float64))
                cell_data['tot_stress'].append(
                    np.zeros((cell_size, 6), dtype=np.float64)
                )

                stress = np.array(self.mech_operators.stresses, copy=False)
                total_stress = np.array(self.mech_operators.total_stresses, copy=False)
                for i in range(6):
                    cell_data['stress'][-1][:, i] = stress[
                        6 * start_geom_cell_id
                        + i : 6 * (start_geom_cell_id + cell_size) : 6
                    ]
                    cell_data['tot_stress'][-1][:, i] = total_stress[
                        6 * start_geom_cell_id
                        + i : 6 * (start_geom_cell_id + cell_size) : 6
                    ]

                if engine.momentum_inertia > 0.0 and dt != 0:  # dynamic simulation
                    # velocity
                    days2sec = 86400
                    if 'v_x' not in cell_data:
                        cell_data['v_x'] = []
                    if 'v_y' not in cell_data:
                        cell_data['v_y'] = []
                    if 'v_z' not in cell_data:
                        cell_data['v_z'] = []
                    cell_data['v_x'].append(
                        -dX[
                            props_num
                            * start_geom_cell_id : props_num
                            * (cell_size + start_geom_cell_id) : props_num
                        ]
                        / dt
                        / days2sec
                    )
                    cell_data['v_y'].append(
                        -dX[
                            props_num * start_geom_cell_id
                            + 1 : props_num
                            * (cell_size + start_geom_cell_id) : props_num
                        ]
                        / dt
                        / days2sec
                    )
                    cell_data['v_z'].append(
                        -dX[
                            props_num * start_geom_cell_id
                            + 2 : props_num
                            * (cell_size + start_geom_cell_id) : props_num
                        ]
                        / dt
                        / days2sec
                    )

                if 'cell_id' not in cell_data:
                    cell_data['cell_id'] = []
                cell_data['cell_id'].append(
                    np.array(
                        [
                            cell_id
                            for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items()
                            if cell.geometry_type == ith_geometry
                        ],
                        dtype=np.int64,
                    )
                )
                # if ith_step == 0:
                #     cell_data[ith_geometry]['permx'] = self.permx[:]
                #     cell_data[ith_geometry]['permy'] = self.permy[:]
                #     cell_data[ith_geometry]['permz'] = self.permz[:]
                start_geom_cell_id += 1

        # Store solution for each time-step:
        mesh = meshio.Mesh(Mesh.points, Mesh.cells, cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtu".format(output_directory, ith_step), mesh)

        # Faults and boundary surfaces
        if self.unstr_discr.frac_cells_tot > 0 or self.unstr_discr.output_faces_tot > 0:
            geom_id = 0
            Mesh.cells = []
            cell_data = {}
            for ith_geometry in self.unstr_discr.mesh_data.cells_dict.keys():
                if ith_geometry in available_fracture_geometries:
                    # fracture geometry
                    frac_ids = np.argwhere(
                        np.in1d(
                            self.unstr_discr.mesh_data.cell_data['gmsh:physical'][
                                geom_id
                            ],
                            self.unstr_discr.physical_tags['fracture'],
                        )
                    )[:, 0]
                    if len(frac_ids):
                        Mesh.cells.append(
                            meshio.CellBlock(
                                ith_geometry,
                                data=self.unstr_discr.mesh_data.cells[geom_id].data[
                                    frac_ids
                                ],
                            )
                        )
                        data = property_array[
                            4
                            * self.unstr_discr.mat_cells_tot : 4
                            * (
                                self.unstr_discr.mat_cells_tot
                                + self.unstr_discr.frac_cells_tot
                            )
                        ].reshape(self.unstr_discr.frac_cells_tot, 4)
                        for i in range(props_num):
                            if cell_property[i] not in cell_data:
                                cell_data[cell_property[i]] = []
                            cell_data[cell_property[i]].append(data[:, i])

                    # output geometry
                    out_ids = np.argwhere(
                        np.in1d(
                            self.unstr_discr.mesh_data.cell_data['gmsh:physical'][
                                geom_id
                            ],
                            self.unstr_discr.physical_tags['output'],
                        )
                    )[:, 0]
                    if len(out_ids):
                        Mesh.cells.append(
                            meshio.CellBlock(
                                ith_geometry,
                                data=self.unstr_discr.mesh_data.cells[geom_id].data[
                                    out_ids
                                ],
                            )
                        )

                geom_id += 1
            # faults output
            if self.unstr_discr.frac_cells_tot > 0:
                # self.write_fault_props(output_directory, property_array, ith_step, engine)
                frac_data = self.get_fault_props(property_array, ith_step, engine)
                for key, val in frac_data.items():
                    if key not in cell_data:
                        cell_data[key] = []
                    cell_data[key].append(val)
            # boundary surfaces output
            if self.unstr_discr.output_faces_tot > 0:
                out_data = self.get_props_over_output(property_array, ith_step, engine)
                for key, val in out_data.items():
                    if key not in cell_data:
                        cell_data[key] = []
                    cell_data[key].append(val)

            # Store solution for each time-step:
            mesh = meshio.Mesh(Mesh.points, Mesh.cells, cell_data=cell_data)
            meshio.write(
                "{:s}/solution_fault{:d}.vtu".format(output_directory, ith_step), mesh
            )

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        return 0

    def write_pvd_file(self, ith_step, time, output_directory):
        # write *.pvd file for matrix and *.pvd file for faults if there are some
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

        # matrix
        snap = self.matpvd_doc.createElement("DataSet")
        snap.setAttribute("timestep", str(time))
        snap.setAttribute("file", 'solution{:d}.vtu'.format(ith_step))
        self.matpvd_collection.appendChild(snap)
        root = self.matpvd_root
        root.appendChild(self.matpvd_collection)
        self.matpvd_doc.writexml(
            open(str(output_directory) + '/solution.pvd', 'w'),
            indent="  ",
            addindent="  ",
            newl='\n',
        )

        # faults or boundary surfaces
        faults = False
        if self.discretizer_name == 'pm_discretizer':
            if (
                self.unstr_discr.frac_cells_tot > 0
                or self.unstr_discr.output_faces_tot > 0
            ):
                faults = True

        if faults:
            snap = self.faultpvd_doc.createElement("DataSet")
            snap.setAttribute("timestep", str(time))
            snap.setAttribute("file", 'solution_fault{:d}.vtu'.format(ith_step))
            self.faultpvd_collection.appendChild(snap)
            root = self.faultpvd_root
            root.appendChild(self.faultpvd_collection)
            self.faultpvd_doc.writexml(
                open(str(output_directory) + '/solution_fault.pvd', 'w'),
                indent="  ",
                addindent="  ",
                newl='\n',
            )

    def get_normal_to_bound_face(self, b_id):
        cell = self.unstr_discr.bound_face_info_dict[b_id]
        cells = [self.unstr_discr.mat_cells_to_node[pt] for pt in cell.nodes_to_cell]
        cell_id = next(iter(set(cells[0]).intersection(*cells)))
        for face in self.unstr_discr.faces[cell_id].values():
            if face.cell_id1 == face.cell_id2 and face.face_id2 == b_id:
                t_face = (
                    cell.centroid
                    - self.unstr_discr.mat_cell_info_dict[cell_id].centroid
                )
                n = face.n
                if np.inner(t_face, n) < 0:
                    n = -n
                return n

    def get_parametrized_fault_props(self):
        ref_id = next(iter(self.unstr_discr.frac_cell_info_dict))
        tags = np.array(
            [cell.prop_id for cell in self.unstr_discr.frac_cell_info_dict.values()]
        )
        tag_ids = {}
        t0 = {}
        coords = {}
        z_coords = {}
        for tag in self.unstr_discr.physical_tags['fracture']:
            ids = np.argwhere(tags == tag)[:, 0]
            tag_ids[tag] = ref_id + ids
            n0 = self.unstr_discr.faces[ref_id + ids[0]][4].n[:2]
            t0[tag] = np.identity(2) - np.outer(n0, n0)
            coords[tag] = np.array(
                [self.unstr_discr.frac_cell_info_dict[i].centroid for i in tag_ids[tag]]
            )
            z_coords[tag] = np.unique(coords[tag][:, 2])

        def dist_sort_key(id):
            c_ref = self.unstr_discr.frac_cell_info_dict[
                list(self.unstr_discr.frac_cell_info_dict.keys())[0]
            ].centroid
            c = self.unstr_discr.frac_cell_info_dict[id].centroid
            return (c[0] - c_ref[0]) ** 2 + (c[1] - c_ref[1]) ** 2

        def eval_frac_proj(tag, coords):
            c_ref = self.unstr_discr.frac_cell_info_dict[
                list(self.unstr_discr.frac_cell_info_dict.keys())[0]
            ].centroid
            coords1 = np.copy(coords)
            coords1[0] -= c_ref[0]
            coords1[1] -= c_ref[1]
            return np.linalg.norm(t0[tag].dot(coords1), axis=0)

        output_layers = 1
        output_var_num = {
            tag: int(output_layers * inds.size / z_coords[tag].size)
            for tag, inds in tag_ids.items()
        }
        faults_num = len(self.unstr_discr.physical_tags['fracture'])

        s = {tag: np.zeros(num) for tag, num in output_var_num.items()}
        # gap = np.zeros( (output_var_num, 3) )
        # Ftan = np.zeros( (output_var_num, 3) )
        # Fnorm = np.zeros( output_var_num )
        inds = {
            tag: np.zeros((output_layers, num), dtype=np.int64)
            for tag, num in output_var_num.items()
        }
        s_ref_prev = 0
        for tag, ids in tag_ids.items():
            counter = 0
            for l, z in enumerate(z_coords[tag][:output_layers]):
                z_inds = list(
                    ids[
                        np.argwhere(
                            np.logical_and(
                                coords[tag][:, 2] > z - 1.0e-5,
                                coords[tag][:, 2] < z + 1.0e-5,
                            )
                        )[:, 0]
                    ]
                )
                z_inds.sort(key=dist_sort_key)
                pts = self.unstr_discr.frac_cell_info_dict[
                    z_inds[0]
                ].coord_nodes_to_cell
                s_ref = np.min(eval_frac_proj(tag, pts[:, :2].T))
                inds[tag][l] = np.array(z_inds) - ref_id
                for id in z_inds:
                    c = self.unstr_discr.frac_cell_info_dict[id].centroid[:2]
                    s[tag][counter] = eval_frac_proj(tag, c) - s_ref + s_ref_prev
                    # gap[counter] = g[id - ref_id]
                    # Ftan[counter] = Ft[id - ref_id]
                    # Fnorm[counter] = Fn[id - ref_id]
                    counter += 1
                pts = self.unstr_discr.frac_cell_info_dict[
                    z_inds[-1]
                ].coord_nodes_to_cell
                s_ref_prev += np.max(eval_frac_proj(tag, pts[:, :2].T)) - s_ref
        z_output = {tag: z[:output_layers] for tag, z in z_coords.items()}
        return s, z_output, inds  # gap, Ftan, Fnorm

    def write_fault_props(self, output_directory, property_array, ith_step, engine):
        n_vars = 4
        n_dim = 3
        fluxes = np.array(engine.fluxes, copy=False)
        fluxes_biot = np.array(engine.fluxes_biot, copy=False)
        s, z_coords, inds = self.get_parametrized_fault_props()
        x = property_array.reshape(int(property_array.size / n_vars), n_vars)[
            self.unstr_discr.mat_cells_tot : self.unstr_discr.mat_cells_tot
            + self.unstr_discr.frac_cells_tot
        ]
        g = {}
        glocal = {}
        flocal = {}
        mu = {}
        for tag, ids in inds.items():
            if ids.size * z_coords[tag].size < self.unstr_discr.frac_cells_tot:
                return 0

            g[tag] = np.array(x[ids[0], :n_dim])
            glocal[tag] = np.zeros((len(ids[0]), 3))
            flocal[tag] = np.zeros((len(ids[0]), 3))
            S_eng = vector_matrix(engine.contacts[0].S)
            mu[tag] = np.array(engine.contacts[0].mu, copy=False)
            fstress = np.array(engine.contacts[0].fault_stress, copy=False)
            for i, id in enumerate(ids[0]):
                face = self.unstr_discr.faces[id + self.unstr_discr.mat_cells_tot][4]
                S = np.array(S_eng[id].values).reshape((n_dim, n_dim))
                flocal[tag][i] = S.dot(
                    fstress[n_dim * id : n_dim * (id + 1)] / face.area
                )
                # n = self.unstr_discr.faces[self.unstr_discr.mat_cells_tot][max(self.unstr_discr.faces[self.unstr_discr.mat_cells_tot].keys())].n
                # S = np.zeros((n_dim, n_dim))
                # S[:n_dim - 1] = null_space(np.array([-n])).T
                # S[n_dim - 1] = -n
                glocal[tag][i] = S.dot(g[tag][i])

            # if ith_step == 0:
            self.fig, self.ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 10))
            self.ax0 = self.ax[0].twinx()
            self.ax1 = self.ax[1].twinx()
            self.ax11 = self.ax[1].twinx()
            # self.ax[0].set_ylabel(r'normal gap, $g_N$')
            self.ax[0].set_ylabel(r'friction coefficient, $\mu$')
            self.ax0.set_ylabel(r'slip, $g_T$')
            self.ax[1].set_ylabel(r'normal traction, $F_N$')
            self.ax1.set_ylabel(r'tangential traction, $F_T$')
            self.ax[1].set_xlabel('distance')
            # self.ax1.set_ylabel('distance along fault')

            phi = np.array(engine.contacts[0].phi, copy=False)
            states = phi > 0
            for tag, s_cur in s.items():
                Fn = flocal[tag][:, 0]
                Ft = flocal[tag][:, 1]
                # self.ax[0].plot(s_cur, glocal[tag][:,0], color='b', linestyle='-', marker='o', label=str(tag) + r': $g_N$')
                if (mu[tag] != 0.0).all() and (Fn != 0.0).all():
                    self.ax[0].plot(
                        s_cur,
                        mu[tag],
                        color='b',
                        linestyle='-',
                        marker='o',
                        label=str(tag) + r': $\mu$',
                    )
                    self.ax[0].plot(
                        s_cur,
                        Ft / Fn,
                        color='r',
                        linestyle='--',
                        marker='o',
                        label=str(tag) + r': $\mu * SCU$',
                    )
                self.ax0.plot(
                    s_cur,
                    -glocal[tag][:, 1],
                    color='r',
                    linestyle='-',
                    marker='o',
                    label=str(tag) + r': $g_T$',
                )
                self.ax[1].plot(
                    s_cur,
                    Fn,
                    color='b',
                    linestyle='-',
                    marker='o',
                    label=str(tag) + r': $F_N$',
                )
                self.ax1.plot(
                    s_cur,
                    Ft,
                    color='r',
                    linestyle='-',
                    marker='o',
                    label=str(tag) + r': $F_T$',
                )
                self.ax11.plot(
                    s_cur, states[ids[0]], color='g', linestyle=':', marker='x'
                )
                if states[ids[0]][0] == 0:
                    self.ax11.text(0, 0, 'STUCK', fontsize=15)
                elif states[ids[0]][0] == 1:
                    self.ax11.text(0, 1, 'SLIP', fontsize=15)

                np.savetxt(
                    output_directory
                    + '/fault_step_'
                    + str(ith_step)
                    + '_tag_'
                    + str(tag)
                    + ".txt",
                    np.c_[
                        s_cur,
                        glocal[tag][:, 0],
                        glocal[tag][:, 1],
                        glocal[tag][:, 2],
                        Fn,
                        Ft,
                        mu[tag],
                    ],
                )

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
            # self.ax0.set_ylim([-0.0002, 0.0006])
            # self.ax[1].set_ylim([19, 21])
            # self.ax1.set_ylim([-0.1, 0.1])
            self.ax11.get_yaxis().set_visible(False)

            self.fig.tight_layout()
            self.fig.savefig(output_directory + '/fig_' + str(ith_step) + '.png')
            plt.close(self.fig)

    def get_frac_apers(self, idata: InputData):
        '''
        :param idata: InputData
        :return: numpy array of fracture apertures, size = number of fractures; None if no fractures
        '''
        frac_apers = None
        if hasattr(idata.other, 'frac_apers'):
            if np.isscalar(idata.other.frac_apers):  # one value for all fractures
                frac_apers = idata.other.frac_apers * np.ones(
                    self.unstr_discr.frac_cells_tot
                )
            else:  # if already an array (each fracture has its aperture)
                frac_apers = idata.other.frac_apers
        return frac_apers

    def init_fractures(self, idata: InputData):
        '''
        Initializes fractures in the unstructured discretizer:
        1.Appends data to: self.pm.cell_centers, self.pm.frac_apers, self.pm.faces, self.mesh.fault_normals, self.pm.perms
        self.pm.biots, self.p_init
        2. Initialize contacts
        :param idata: InputData (frac apertures, perm, biot, initial_pressure)
        :return:
        '''
        frac_apers = self.get_frac_apers(idata)
        if self.discretizer_name == 'pm_discretizer':
            # fracture
            for frac_id in range(
                self.unstr_discr.mat_cells_tot,
                self.unstr_discr.mat_cells_tot + self.unstr_discr.frac_cells_tot,
            ):
                frac = self.unstr_discr.frac_cell_info_dict[frac_id]
                faces = self.unstr_discr.faces[frac_id]
                self.pm.cell_centers.append(
                    matrix(list(frac.centroid), frac.centroid.size, 1)
                )
                self.pm.frac_apers.append(
                    frac_apers[frac_id - self.unstr_discr.mat_cells_tot]
                )
                fs = face_vector()
                for face_id, face in faces.items():
                    face = faces[face_id]
                    fs.append(
                        Face(
                            face.type.value,
                            face.cell_id1,
                            face.cell_id2,
                            face.face_id1,
                            face.face_id2,
                            face.area,
                            list(face.n),
                            list(face.centroid),
                        )
                    )
                self.pm.faces.append(fs)

                face1 = faces[4]
                face2 = faces[5]
                self.mesh.fault_normals.append(face1.n[0])
                self.mesh.fault_normals.append(face1.n[1])
                self.mesh.fault_normals.append(face1.n[2])
                # Local basis
                S = np.zeros((self.n_dim, self.n_dim))
                S[: self.n_dim - 1] = null_space(np.array([face1.n])).T
                S[self.n_dim - 1] = face1.n
                Sinv = np.linalg.inv(S)
                K = np.zeros((self.n_dim, self.n_dim))
                for i in range(self.n_dim):
                    K[i, i] = idata.rock.get_permxyz()[i]
                K = Sinv.dot(K).dot(S)
                self.pm.perms.append(engine_matrix33(list(K.flatten())))
                self.pm.biots.append(engine_matrix33(idata.rock.biot))

                if not np.isscalar(self.p_init):
                    self.p_init[frac_id] = idata.initial.initial_pressure

            # contact
            self.ref_contact_cells = np.zeros(
                self.unstr_discr.frac_cells_tot, dtype=np.intc
            )
            self.contacts = contact_vector()
            for tag in self.unstr_discr.physical_tags['fracture']:
                con = contact()
                con.f_scale = 1.0e8  # multiplier for penalty parameter
                cell_ids = [
                    cell_id
                    for cell_id, cell in self.unstr_discr.frac_cell_info_dict.items()
                    if cell.prop_id == tag
                ]

                self.ref_contact_cells[
                    np.array(cell_ids, dtype=np.intp) - self.unstr_discr.mat_cells_tot
                ] = cell_ids[0]
                con.fault_tag = tag
                con.cell_ids = index_vector(cell_ids)

                con.friction_criterion = critical_stress.BIOT
                fric_coef = idata.other.friction * np.ones(len(cell_ids))
                con.mu0 = value_vector(fric_coef)
                con.mu = con.mu0

                self.contacts.append(con)
        elif self.discretizer_name == 'mech_discretizer':
            assert (
                self.unstr_discr.frac_cells_tot == 0
            ), "Fractures are not supported in mech discretizer"
