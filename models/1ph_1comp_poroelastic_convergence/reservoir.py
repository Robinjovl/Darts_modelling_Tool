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
from rhs import RhsPoroelastic, RhsThermoporoelastic
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
from darts.input.input_data import InputData
# Definitions for the unstructured reservoir class:
class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, idata: InputData, discretizer, mode, mesh_filename, fluid_vars):
        thermoporoelasticity = True if mode == 'thermoporoelastic' else False
        super().__init__(timer, discretizer, thermoporoelasticity, fluid_vars)
        # define correspondence between the physical tags in msh file and mesh elements types
        self.bnd_tags = idata.mesh.bnd_tags
        self.domain_tags = set_domain_tags(matrix_tags=idata.mesh.matrix_tags, bnd_tags=list(self.bnd_tags.values()))

        self.mesh_filename = mesh_filename

        # Specify elastic properties, mesh & boundaries
        self.timer.node["discretization"] = timer_node()
        self.grav = 9.81e-2  # to make gravity term the same order as orther terms in equation
        if self.discretizer_name == 'pm_discretizer':
            self.convergence_study_setup_pm_discretizer(idata=idata)
        elif self.discretizer_name == 'mech_discretizer':
            if self.thermoporoelasticity:
                self.convergence_study_setup_mech_discretizer_thermoporoelasticity(idata=idata)
            else:
                self.convergence_study_setup_mech_discretizer_poroelasticity(idata=idata)

        self.x_new = np.ones((self.n_matrix + self.n_fracs, self.n_vars))
        self.x_new[:, self.u_var] = self.u_init[0]
        self.x_new[:, self.u_var + 1] = self.u_init[1]
        self.x_new[:, self.u_var + 2] = self.u_init[2]
        self.x_new[:, self.p_var] = self.p_init
        if self.thermoporoelasticity:
            self.x_new[:, self.t_var] = self.t_init

        if self.discretizer_name == 'pm_discretizer':
            self.unstr_discr.p_ref = self.p_init
            self.unstr_discr.f = self.f_prep
        self.init_reservoir_main(idata=idata)

        self.bc_prev[:] = self.bc_rhs_prev
        self.bc[:] = self.bc_rhs
        self.bc_ref[:] = self.bc_rhs_ref
        self.set_bounds(self.pz_bounds_rhs)

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
    def calc_peclet_number(self, idata: InputData, time):
        assert(self.thermoporoelasticity)

        per_day_2_per_sec = 86400.0
        vel = self.r.darcy_velocity_func(self.a / 2, self.a / 2, self.a / 2, time)[:, 0] / idata.fluid.viscosity / per_day_2_per_sec
        hc = np.linalg.norm(idata.rock.conductivity)
        self.peclet = idata.rock.heat_capacity * idata.fluid.density * np.linalg.norm(vel) * self.a / hc
        return self.peclet

    def update_trans(self, dt, x):
        #self.pm.x_prev = value_vector(np.concatenate((x, self.bc_rhs_prev)))
        #self.pm.reconstruct_gradients_per_cell(dt)
        #self.pm.calc_all_fluxes(dt)
        #self.write_pm_conn_to_file(t_step=t_step)
        #self.mesh.init_pm(self.pm.cell_m, self.pm.cell_p, self.pm.stencil, self.pm.offset, self.pm.tran, self.pm.rhs,
        #                  self.unstr_discr.mat_cells_tot, self.unstr_discr.bound_faces_tot, 0)

        # update transient sources / sinks
        self.f[:] = self.f_prep
        # update boundaries at n+1 / n timesteps
        self.bc[:] = self.bc_rhs
        self.bc_prev[:] = self.bc_rhs_prev
        self.pz_bounds = np.array(self.mesh.pz_bounds, copy=False)
        self.pz_bounds[:] = self.pz_bounds_rhs
        #self.init_wells()
    def set_boundary_conditions(self, idata):
        self.boundary_conditions = idata.boundary
        self.bnd_tags = idata.mesh.bnd_tags
        self.set_boundary_conditions_pm_discretizer()

    # pm_discretizer: poroelastic
    def update_pm_discretizer(self, time):
        # Boundary conditions
        for bound_id in range(len(self.unstr_discr.bound_face_info_dict)):
            c = self.unstr_discr.bound_face_info_dict[bound_id].centroid
            n = self.get_normal_to_bound_face(bound_id)
            P = np.identity(3) - np.outer(n, n)
            sol = reference_solution_poroelastic(np.append(c, time))
            u = sol[:3]
            p = sol[3]
            prop_id = self.unstr_discr.bound_face_info_dict[bound_id].prop_id
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
    def convergence_study_setup_pm_discretizer(self, idata: InputData):
        physical_tags = {}
        physical_tags['matrix'] = list(self.domain_tags[elem_loc.MATRIX])
        physical_tags['fracture'] = list(self.domain_tags[elem_loc.FRACTURE])
        physical_tags['fracture_shape'] = list(self.domain_tags[elem_loc.FRACTURE_BOUNDARY])
        physical_tags['boundary'] = list(self.domain_tags[elem_loc.BOUNDARY])

        self.unstr_discr = UnstructDiscretizer(mesh_file=self.mesh_filename, physical_tags=physical_tags)

        self.set_boundary_conditions(idata=idata)
        self.unstr_discr.load_mesh(permx=1, permy=1, permz=1, frac_aper=1.E-4)
        self.unstr_discr.calc_cell_neighbours()

        # init poromechanics discretizer
        self.pm = pm_discretizer()
        self.init_uniform_properties(idata=idata)
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)
        self.init_faces_centers_pm_discretizer()
        self.set_vars_pm_discretizer()

        # Initial conditions
        self.p_init = np.zeros(self.n_elements)
        self.u_init = np.zeros((self.n_elements, 3))
        time = 0.0
        for cell_id, cell in self.unstr_discr.mat_cell_info_dict.items():
            sol = reference_solution_poroelastic(np.append(cell.centroid, time))
            self.u_init[cell_id] = sol[:3]
            self.p_init[cell_id] = sol[3]
        self.u_init = [self.u_init[:,0], self.u_init[:,1], self.u_init[:,2]]
        # Boundary conditions
        self.init_arrays_boundary_condition()
        # RHS (force) term
        self.r = RhsPoroelastic(stf=idata.rock.stiffness, biot=idata.rock.biot, perm=idata.rock.perm,
                                visc=idata.fluid.viscosity, grav=self.grav, rho_f=idata.fluid.density,
                                rho_s=idata.rock.density, comp_s=idata.rock.compressibility,
                                poro0=idata.rock.porosity)
        self.f_prep = np.zeros(self.n_matrix * self.n_vars)
        self.total_stress_an = np.zeros((self.n_matrix, 6))
        self.effective_stresses_an = np.zeros((self.n_matrix, 6))
        self.darcy_velocities_an = np.zeros((self.n_matrix, 3))
        self.update_pm_discretizer(time=0)
        self.bc_rhs_prev = np.copy(self.bc_rhs)
        self.pm.bc_prev = self.pm.bc

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
                -np.array(self.r.f_func(c.values[0], c.values[1], c.values[2], time))[:, 0] - \
                np.array(self.r.momentum_acc_func(c.values[0], c.values[1], c.values[2], time))[:, 0]
            self.f_prep[self.n_vars * cell_id + self.p_var] = \
                -self.r.acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.flow_func(c.values[0], c.values[1], c.values[2], time)
            self.total_stress_an[cell_id] = self.r.total_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.effective_stresses_an[cell_id] = self.r.effective_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.darcy_velocities_an[cell_id] = self.r.darcy_velocity_func(c.values[0], c.values[1], c.values[2], time)[:,0]
    def convergence_study_setup_mech_discretizer_poroelasticity(self, idata: InputData):
        self.mesh_data = meshio.read(self.mesh_filename)
        self.set_boundary_conditions(idata=idata)
        self.init_mech_discretizer(idata=idata)
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)
        self.init_uniform_properties(idata=idata)

        # mapping boundary connections
        id_sorted = np.argsort(self.adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = self.adj_matrix[id_sorted]
        time = 0.0
        self.x_all = np.array([np.array([c.values[0], c.values[1], c.values[2], time]) for c in self.centroids]).T
        sol = reference_solution_poroelastic(self.x_all)
        self.u_init = sol[:self.n_dim, :self.n_matrix]
        self.p_init = sol[self.n_dim, :self.n_matrix]

        self.init_arrays_boundary_condition()
        # RHS term
        self.r = RhsPoroelastic(stf=idata.rock.stiffness, biot=idata.rock.biot, perm=idata.rock.perm,
                                visc=idata.fluid.viscosity, grav=self.grav, rho_f=idata.fluid.density,
                                rho_s=idata.rock.density, comp_s=idata.rock.compressibility,
                                poro0=idata.rock.porosity)
        self.f_prep = np.zeros(self.n_matrix * self.n_vars)
        self.total_stress_an = np.zeros((self.n_matrix, 6))
        self.effective_stresses_an = np.zeros((self.n_matrix, 6))
        self.darcy_velocities_an = np.zeros((self.n_matrix, 3))

        self.update_mech_discretizer_poroelasticity(time=0)
        # perform discretization
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

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
                -np.array(self.r.f_func(c.values[0], c.values[1], c.values[2], time))[:, 0]  - \
                    np.array(self.r.momentum_acc_func(c.values[0], c.values[1], c.values[2], time))[:, 0]
            self.f_prep[self.n_vars * cell_id + self.t_var] = \
                -self.r.energy_acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.energy_flow_func(c.values[0], c.values[1], c.values[2], time)
            self.f_prep[self.n_vars * cell_id + self.p_var] = \
                -self.r.acc_func(c.values[0], c.values[1], c.values[2], time) - \
                self.r.flow_func(c.values[0], c.values[1], c.values[2], time)
            self.total_stress_an[cell_id] = self.r.total_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.effective_stresses_an[cell_id] = self.r.effective_stress_func(c.values[0], c.values[1], c.values[2], time)[:,0]
            self.darcy_velocities_an[cell_id] = self.r.darcy_velocity_func(c.values[0], c.values[1], c.values[2], time)[:,0]
    def convergence_study_setup_mech_discretizer_thermoporoelasticity(self, idata: InputData):
        self.mesh_data = meshio.read(self.mesh_filename)
        self.set_boundary_conditions(idata=idata)
        self.init_mech_discretizer(idata=idata)
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)
        self.init_uniform_properties(idata=idata)
        # mapping boundary connections
        id_sorted = np.argsort(self.adj_matrix_cols)[-self.n_bounds:]
        self.id_boundary_conns = self.adj_matrix[id_sorted]
        time = 0.0
        self.x_all = np.array([np.array([c.values[0], c.values[1], c.values[2], time]) for c in self.centroids]).T
        sol = reference_solution_thermoporoelastic(self.x_all)
        self.u_init = sol[:self.n_dim, :self.n_matrix]
        self.p_init = sol[self.n_dim, :self.n_matrix]
        self.t_init = sol[self.n_dim + 1, :self.n_matrix]

        self.init_arrays_boundary_condition()
        # RHS term
        self.r = RhsThermoporoelastic(stf=idata.rock.stiffness, biot=idata.rock.biot, perm=idata.rock.perm,
                                        th_expn=idata.rock.th_expn, heat_cond=idata.rock.conductivity,
                                        visc=idata.fluid.viscosity, grav=self.grav, rho_f=idata.fluid.density,
                                        rho_s=idata.rock.density, comp_s=idata.rock.compressibility,
                                        poro0=idata.rock.porosity, th_expn_poro=idata.rock.th_expn_poro,
                                        heat_capacity=idata.rock.heat_capacity)
        self.f_prep = np.zeros(self.n_matrix * self.n_vars)
        self.total_stress_an = np.zeros((self.n_matrix, 6))
        self.effective_stresses_an = np.zeros((self.n_matrix, 6))
        self.darcy_velocities_an = np.zeros((self.n_matrix, 3))

        self.update_mech_discretizer_thermoporoelasticity(time=0)

        # perform discretization
        self.timer.node["discretization"].start()
        self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_flow, self.cpp_heat)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

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
