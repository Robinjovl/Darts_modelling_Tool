import numpy as np
from math import inf, pi
import scipy.optimize as opt
from scipy.linalg import null_space
from scipy.special import erfc as erfc
from itertools import compress
import meshio
from matplotlib import pyplot as plt
from matplotlib import rcParams

from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, contact, contact_vector, vector_matrix, scheme_type
from darts.engines import matrix33 as engine_matrix33
from darts.engines import Stiffness as engine_stiffness
from darts.engines import matrix, pm_discretizer, Face, vector_face_vector, face_vector, vector_matrix33, stf_vector, critical_stress

from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags, get_lambda_mu, get_bulk_modulus, get_biot_modulus
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
from darts.reservoirs.mesh.geometrymodule import FType
from darts.engines import timer_node
from darts.discretizer import elem_loc
from darts.discretizer import vector_matrix33, vector_vector3, matrix, value_vector, index_vector
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
from darts.input.input_data import InputData

def get_mesh_filename(mesh='rect', suffix='', prop=''):
    if 'rect' in mesh:
        mesh_filename = ('meshes/transfinite')
        if len(mesh.split('_')) > 1:  # for the convergence_analysis
            mesh_filename += '_' + mesh.split('_')[1]
    elif mesh == 'wedge':
        mesh_filename = 'meshes/wedge'
    elif mesh == 'hex':
        mesh_filename = 'meshes/hexahedron'
    return mesh_filename + suffix + '.msh'

# Definitions for the unstructured reservoir class:
class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, idata: InputData, fluid_vars, case='mandel', discretizer='mech_discretizer'):
        thermoporoelasticity = idata.type_mech == 'thermoporoelasticity'
        super().__init__(timer, discretizer=discretizer, fluid_vars=fluid_vars, thermoporoelasticity=thermoporoelasticity)

        # define correspondence between the physical tags in msh file and mesh elements types
        self.bnd_tags = idata.mesh.bnd_tags
        self.domain_tags = set_domain_tags(matrix_tags=idata.mesh.matrix_tags, bnd_tags=list(self.bnd_tags.values()))

        self.mesh_filename = idata.mesh.mesh_filename
        # Specify elastic properties, mesh & boundaries
        if case == 'mandel':
            if discretizer == 'mech_discretizer':
                self.mandel_north_dirichlet_mech_discretizer(idata=idata)
            elif discretizer == 'pm_discretizer':
                self.mandel_north_dirichlet_pm_discretizer(idata=idata)
        elif case == 'terzaghi':
            if discretizer == 'mech_discretizer':
                self.terzaghi_mech_discretizer(idata=idata)
            elif discretizer == 'pm_discretizer':
                self.terzaghi_pm_discretizer(idata=idata)
        elif case == 'terzaghi_two_layers':
            if discretizer == 'mech_discretizer':
                self.terzaghi_two_layers_mech_discretizer(idata=idata)
            elif discretizer == 'pm_discretizer':
                self.terzaghi_two_layers_pm_discretizer(idata=idata)
        elif case == 'terzaghi_two_layers_no_analytics':
            if discretizer == 'pm_discretizer':
                self.terzaghi_two_layers_no_analytics_pm_discretizer(idata=idata)
        elif case == 'bai':
            self.bai_thermoporoelastic_consolidation(idata=idata)
        else:
            print('Error: wrong case', case)
            exit(-1)

        self.init_reservoir_main(idata=idata)
        self.set_pzt_bounds(p=self.p_init, z=None, t=self.t_init)

    def init_tD_pD(self, idata: InputData, F, a=1):
        '''
        set self.tD and self.pD, they used to get dimensionless solution to compare with analytic solution
        '''
        MR = 0.9869 * 1.E-15 * idata.rock.perm / idata.fluid.viscosity / 1.E-3
        K_dr = idata.rock.E / (3 * (1 - 2 * idata.rock.nu))
        self.K_nu = (K_dr + (4 / 3) * self.mu)
        Cv = 1.e+5 * MR * self.M * self.K_nu / (self.K_nu + idata.rock.biot ** 2 * self.M)
        self.tD = self.a ** 2 / Cv / 86400
        self.pD = abs(F / a) / 2
    # Mandel
    def mandel_north_dirichlet_mech_discretizer(self, idata: InputData):
        self.mesh_data = meshio.read(idata.mesh.mesh_filename)
        self.set_uniform_initial_conditions(idata=idata)
        self.set_mandel_boundary_conditions(idata)
        self.init_mech_discretizer(idata=idata)
        # mandel case has a specific setting of the mechanical boundary condition, see update_mandel_boundary()
        self.F = idata.other.Fa * self.a  # vertical load [bar * m]
        self.lam, self.mu = get_lambda_mu(idata.rock.E, idata.rock.nu)
        self.M = get_biot_modulus(biot=idata.rock.biot, poro0=idata.rock.porosity,
                                  kd=get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu),
                                  cf=idata.fluid.compressibility)
        self.init_uniform_properties(idata=idata)
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

        self.init_tD_pD(idata, self.F, self.a)
        # from compare_grad_discr import compare_gradients
        # compare_gradients('pm.pkl', new_cache_filename=None, orig_pm_arg=None, new_pm_arg=self.discr)

    def mandel_north_dirichlet_pm_discretizer(self, idata: InputData):
        self.set_uniform_initial_conditions(idata=idata)
        physical_tags = {}
        physical_tags['matrix'] = list(self.domain_tags[elem_loc.MATRIX])
        physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_filename, physical_tags=physical_tags)
        self.unstr_discr.eps_t = 1.E+0
        self.unstr_discr.eps_n = 1.E+0
        self.unstr_discr.mu = 3.2
        self.unstr_discr.P12 = 0
        self.unstr_discr.Prol = 1
        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3
        self.lam, self.mu = get_lambda_mu(idata.rock.E, idata.rock.nu)
        self.M = get_biot_modulus(biot=idata.rock.biot, poro0=idata.rock.porosity,
                                  kd=get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu),
                                  cf=idata.fluid.compressibility)
        self.init_matrix_stiffness({self.unstr_discr.physical_tags['matrix'][0]:
                                                    {'E': idata.rock.E, 'nu': idata.rock.nu, 'stiffness': idata.rock.stiffness}})
        self.set_boundary_conditions(idata)
        self.unstr_discr.load_mesh(permx=1, permy=1, permz=1, frac_aper=0)
        self.unstr_discr.calc_cell_neighbours()

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        self.b = np.max(self.unstr_discr.mesh_data.points[:, 1])
        self.F = idata.other.Fa * self.a  # bar * m

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.set_scheme_pm_discretizer()
        self.pm.neumann_boundaries_grad_reconstruction = True
        self.init_gravity(gravity_on=False)

        self.init_faces_centers_pm_discretizer()
        self.init_uniform_properties(idata=idata)
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()
        self.init_tD_pD(idata, self.F, self.a)
    def set_boundary_conditions(self, idata):
        self.boundary_conditions = idata.boundary
        self.bnd_tags = idata.mesh.bnd_tags
        self.set_boundary_conditions_pm_discretizer()
    def set_mandel_boundary_conditions(self, idata, v_north=0.):
        self.set_boundary_conditions(idata)
        self.boundary_conditions[self.bnd_tags['BND_Y+']]['mech'] = self.bc_type.STUCK_ROLLER(v_north)
        self.set_boundary_conditions_pm_discretizer()
    def update_mandel_boundary(self, time, idata: InputData):
        '''
        time-dependent boundary condition from the analytic solution
        :param time: time in days
        '''
        v_north = self.get_vertical_displacement_north_mandel(time, idata)
        self.set_mandel_boundary_conditions(idata, v_north)
        self.init_bc_rhs()

    # Terzaghi
    def terzaghi_mech_discretizer(self, idata: InputData):
        self.mesh_data = meshio.read(idata.mesh.mesh_filename)

        self.set_uniform_initial_conditions(idata=idata)
        self.F = idata.other.F
        self.lam, self.mu = get_lambda_mu(idata.rock.E, idata.rock.nu)
        self.M = get_biot_modulus(biot=idata.rock.biot, poro0=idata.rock.porosity,
                                  kd=get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu),
                                  cf=idata.fluid.compressibility)
        self.set_boundary_conditions(idata)
        self.init_mech_discretizer(idata=idata)
        self.init_uniform_properties(idata=idata)
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

        self.init_tD_pD(idata, self.a)

        # from compare_grad_discr import compare_gradients
        # compare_gradients('pm.pkl', new_cache_filename=None, orig_pm_arg=None, new_pm_arg=self.discr)
    def terzaghi_pm_discretizer(self, idata: InputData):
        self.set_uniform_initial_conditions(idata=idata)
        physical_tags = {'matrix': list(self.domain_tags[elem_loc.MATRIX])}
        physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_filename, physical_tags=physical_tags)
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

        self.F = idata.other.F
        self.lam, self.mu = get_lambda_mu(idata.rock.E, idata.rock.nu)
        self.M = get_biot_modulus(biot=idata.rock.biot, poro0=idata.rock.porosity,
                                  kd=get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu),
                                  cf=idata.fluid.compressibility)
        self.init_matrix_stiffness({self.unstr_discr.physical_tags['matrix'][0]:
                                                    {'E': idata.rock.E, 'nu': idata.rock.nu, 'stiffness': idata.rock.stiffness}})
        self.set_boundary_conditions(idata)
        self.unstr_discr.load_mesh(permx=1, permy=1, permz=1, frac_aper=0)
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.set_scheme_pm_discretizer()
        self.pm.neumann_boundaries_grad_reconstruction = True
        self.init_gravity(gravity_on=False)

        self.init_faces_centers_pm_discretizer()
        self.init_uniform_properties(idata=idata)
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()

        self.n_fracs = self.unstr_discr.frac_cells_tot #TODO
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_faces_tot

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        self.init_tD_pD(idata, self.F, a=0.5)

    # Two-layer Terzaghi
    def two_layers_pm_discretizer(self, idata: InputData):
        self.set_uniform_initial_conditions(idata=idata)

        # define correspondence between the physical tags in msh file and mesh elements types
        # two regions for different properties
        [self.m1_tag, self.m2_tag] = idata.mesh.matrix_tags
        self.set_props_tags(idata=idata, matrix_tags=idata.mesh.matrix_tags)

        physical_tags = {}
        physical_tags['matrix'] = [self.m1_tag, self.m2_tag]
        physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_filename, physical_tags=physical_tags)

        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3
        self.init_matrix_stiffness(self.props)
        self.F = idata.other.F
        self.set_boundary_conditions(idata)
        self.unstr_discr.load_mesh(permx=1, permy=1, permz=1, frac_aper=0)
        self.unstr_discr.calc_cell_neighbours()

        self.n_fracs = self.unstr_discr.frac_cells_tot
        self.n_matrix = self.unstr_discr.mat_cells_tot
        self.n_bounds = self.unstr_discr.bound_faces_tot

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.set_scheme_pm_discretizer()
        self.pm.neumann_boundaries_grad_reconstruction = False
        self.init_gravity(gravity_on=False)
        self.init_faces_centers_pm_discretizer()
        self.init_heterogeneous_properties()
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()
        self.unstr_discr.f[3::4] = self.p_init - self.unstr_discr.p_ref[:]

    def terzaghi_two_layers_pm_discretizer(self, idata: InputData):
        self.two_layers_pm_discretizer(idata)

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        self.omega = self.approximate_roots_two_layers_terzaghi()
        self.tD = 1.0
        self.pD = 1.0

    def terzaghi_two_layers_no_analytics_pm_discretizer(self, idata: InputData):
        self.set_uniform_initial_conditions(idata=idata)

        # define correspondence between the physical tags in msh file and mesh elements types
        # two regions for different properties
        [self.m1_tag, self.m2_tag] = idata.mesh.matrix_tags
        self.set_props_tags(idata=idata, matrix_tags=idata.mesh.matrix_tags)

        physical_tags = {}
        physical_tags['matrix'] = idata.mesh.matrix_tags
        physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_filename, physical_tags=physical_tags)

        self.unstr_discr.n_dim = 3
        self.unstr_discr.bcf_num = 3
        self.unstr_discr.bcm_num = self.unstr_discr.n_dim + 3
        self.init_matrix_stiffness(self.props)
        self.F = idata.other.F
        self.set_boundary_conditions(idata)
        self.unstr_discr.load_mesh(permx=1, permy=1, permz=1, frac_aper=0)
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.set_scheme_pm_discretizer()
        self.pm.neumann_boundaries_grad_reconstruction = False
        self.init_gravity(gravity_on=False)

        self.init_faces_centers_pm_discretizer()
        self.init_heterogeneous_properties()
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()

        self.a = np.max(self.unstr_discr.mesh_data.points[:, 0])
        self.tD = 1.0
        self.pD = 1.0

    def two_layers_mech_discretizer(self, idata: InputData):
        self.mesh_data = meshio.read(idata.mesh.mesh_filename)
        # define correspondence between the physical tags in msh file and mesh elements types
        # two regions for different properties
        [self.m1_tag, self.m2_tag] = idata.mesh.matrix_tags
        self.set_props_tags(idata=idata, matrix_tags=idata.mesh.matrix_tags)
        self.set_uniform_initial_conditions(idata=idata)
        self.F = idata.other.F
        self.set_boundary_conditions(idata)
        self.init_mech_discretizer(idata=idata)
        self.init_heterogeneous_properties()
        self.init_arrays_boundary_condition()
        self.init_bc_rhs()

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

    def terzaghi_two_layers_mech_discretizer(self, idata: InputData):
        self.two_layers_mech_discretizer(idata)
        self.omega = self.approximate_roots_two_layers_terzaghi()
        self.tD = 1.0
        self.pD = 1.0

    # Bai, 2005 (unidimensional thermoporoelastic consolidation)
    def bai_thermoporoelastic_consolidation(self, idata: InputData):
        self.mesh_data = meshio.read(idata.mesh.mesh_filename)
        self.set_uniform_initial_conditions(idata=idata)
        self.F = idata.other.F
        self.lam, self.mu = get_lambda_mu(idata.rock.E, idata.rock.nu)
        self.set_boundary_conditions(idata=idata)
        self.init_mech_discretizer(idata=idata)

        if len(idata.mesh.matrix_tags) == 1:
            self.init_uniform_properties(idata=idata)
        else:
            #[self.m1_tag, self.m2_tag] = idata.mesh.matrix_tags
            self.set_props_tags(idata=idata, matrix_tags=idata.mesh.matrix_tags)

        self.init_arrays_boundary_condition()
        self.init_bc_rhs()

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_flow,  self.cpp_heat)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

    # Mandel analytics
    def get_params_analytic(self, idata: InputData):
        F = np.fabs(self.F)
        K_s = self.lam + 2 * self.mu / 3
        skempton = idata.rock.biot * self.M / (K_s + self.M * idata.rock.biot ** 2)
        nu_s = idata.rock.nu
        nu_u = (3 * idata.rock.nu + idata.rock.biot * skempton * (1 - 2 * idata.rock.nu)) / (3 - idata.rock.biot * skempton * (1 - 2 * idata.rock.nu))
        mu_s = self.mu
        mu_f = idata.fluid.viscosity
        k_s = idata.rock.perm / idata.fluid.viscosity
        c_f = TC.darcy_constant * (2 * k_s * (skempton ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / ( 9 * mu_f * (1 - nu_u) * (nu_u - nu_s) )

        cy0 = (-F * (1 - nu_s)) / (2 * mu_s * self.a)
        cy1 = F * (1 - nu_u) / (mu_s * self.a)
        return c_f, cy0, cy1, skempton, nu_u, nu_s, k_s

    def get_vertical_displacement_north_mandel(self, t, idata: InputData):
        c_f, cy0, cy1, skempton, nu_u, nu_s, k_s = self.get_params_analytic(idata)

        # Calculate constants
        aa_n = self.approximate_roots(idata)[:, np.newaxis]

        # Calculate exact north boundary condition
        uy_sum = np.sum(
            ((np.sin(aa_n) * np.cos(aa_n)) / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
            * np.exp((-(aa_n**2) * c_f * t) / (self.a**2)),
            axis=0,
        )

        north_bc = (cy0 + cy1 * uy_sum) * self.b
        return north_bc
    def approximate_roots(self, idata: InputData) -> np.ndarray:
        """
        f(x) = tan(x) - ((1-nu)/(nu_u-nu)) x
        """
        c_f, cy0, cy1, skempton, nu_u, nu_s, k_s = self.get_params_analytic(idata)
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
    def mandel_exact_pressure(self, idata:InputData, t, xc) -> np.ndarray:
        """
        Pressure solution for a given time `t`.
        """
        # Parameters
        F = np.fabs(self.F)
        c_f, cy0, cy1, skempton, nu_u, nu_s, k_s = self.get_params_analytic(idata=idata)

        if t == 0.0:  # initial condition has its own expression
            p = ((F * skempton * (1 + nu_u)) / (3 * self.a)) * np.ones(xc.size)
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots(idata)[:, np.newaxis]
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
    def mandel_exact_displacements(self, idata:InputData, t, xc) -> np.ndarray:
        """
        Exact pressure solution for a given time `t`.

        Args:
            t: Time in seconds.

        Returns:
            p (sd.num_cells, ): Exact pressure solution.

        """

        # Retrieve physical data
        F = np.fabs(self.F)
        c_f, cy0, cy1, skempton, nu_u, nu_s, k_s = self.get_params_analytic(idata=idata)
        # -----> Compute exact fluid pressure

        if t == 0.0:  # initial condition has its own expression
            p = ((F * skempton * (1 + nu_u)) / (3 * self.a)) * np.ones(xc.shape[0])
            ux = F / self.mu / self.a * nu_u * xc[:, 0] / 2
            uy = F / self.mu / self.a * (nu_u - 1) * xc[:, 1] / 2
        else:
            # Retrieve approximated roots
            aa_n = self.approximate_roots(idata)[:, np.newaxis]
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
            ux = F / self.mu / self.a * (idata.rock.nu * xc[:,0] / 2 + ux_sum_0)
            # Exact uy
            uy_sum_0 = np.sum(
                ( np.sin(aa_n) * np.cos(aa_n) /
                  (aa_n - (np.sin(aa_n) * np.cos(aa_n))) ) * np.exp((-(aa_n ** 2) * c_f * t) / (self.a ** 2)),
                axis=0,
            )
            uy = F / self.mu / self.a * ((idata.rock.nu - 1) * xc[:,1] / 2 - (nu_u - 1) * xc[:,1] * uy_sum_0)

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
    def terzaghi_exact_pressure(self, idata: InputData, t, xc) -> np.ndarray:
        # Parameters
        c_f, cy0, cy1, skempton, nu_u, nu_s, k_s = self.get_params_analytic(idata=idata)
        h = self.a
        vertical_load = np.fabs(idata.other.F)
        dimless_t = t# / self.tD

        n = 1000

        p0 = vertical_load * skempton * (1 + nu_u) / 3 / (1 - nu_u)
        c = TC.darcy_constant * 2 * k_s * self.mu * (1 - nu_s) * (nu_u - nu_s) / idata.rock.biot ** 2 / (1 - nu_u) / (1 - 2 * nu_s) ** 2

        if dimless_t > 0:
            sum_series = np.zeros_like(xc)
            for m in range(0, n):
                sum_series += (-1) ** m * (erfc( ((1 + 2*m) * h + xc) / np.sqrt(4 * c * dimless_t) ) +
                                           erfc( ((1 + 2*m) * h - xc) / np.sqrt(4 * c * dimless_t) ) )
            p =  p0 * (1 - sum_series)
        else:
            p = p0
        return p
    def terzaghi_exact_displacements(self, idata: InputData, t, xc) -> np.ndarray:
        """Compute exact pressure.
        Args:
            t: Time in seconds.
        Returns:
            Exact pressure for the given time `t`.
        """
        # Retrieve physical data
        c_f, cy0, cy1, skempton, nu_u, nu_s, k_s = self.get_params_analytic(idata=idata)
        h = self.a
        vertical_load = np.fabs(idata.other.F)
        dimless_t = t# / self.tD

        n = 1000

        u0 = -xc * vertical_load * (1  - 2 * idata.rock.nu) / 2 / self.mu / (1 - idata.rock.nu)
        c = TC.darcy_constant * 2 * k_s * self.mu * (1 - nu_s) * (nu_u - nu_s) / idata.rock.biot ** 2 / (1 - nu_u) / (1 - 2 * nu_s) ** 2
        coef = 4 * vertical_load * h * (nu_u - idata.rock.nu) / np.pi ** 2 / self.mu / (1 - idata.rock.nu) / (1 - nu_u)

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
        b1 = self.props[self.m1_tag]['biot']
        b2 = self.props[self.m2_tag]['biot']
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
