import numpy as np
import os
import meshio
from darts.discretizer import elem_type, elem_loc
from darts.discretizer import matrix33 as disc_matrix33
from darts.discretizer import Stiffness as disc_stiffness
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags, get_lambda_mu, get_biot_modulus
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
from darts.input.input_data import InputData
from darts.engines import timer_node, ms_well, ms_well_vector
import copy

class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, idata: InputData, model_folder, fluid_vars=['p'], uniform_props=False):
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        thermoporoelasticity = True if 'temperature' in fluid_vars else False
        super().__init__(timer, discretizer='mech_discretizer',
                         thermoporoelasticity=thermoporoelasticity, fluid_vars=fluid_vars)

        self.bnd_tags = idata.mesh.bnd_tags
        self.domain_tags = set_domain_tags(matrix_tags=idata.mesh.matrix_tags, bnd_tags=list(self.bnd_tags.values()))

        self.spe10(model_folder=model_folder, idata=idata, uniform_props=uniform_props)
        self.init_reservoir_main(idata=idata)
        t1 = None
        if thermoporoelasticity:
            t1 = np.mean(self.t_init)
        self.set_pzt_bounds(p=np.mean(self.p_init), z=self.z_init, t=t1)
        self.wells = []

    def get_reservoir_temperature(self, depths):
        top = -3608.832
        bot = -3657.6
        t_top = 300.0
        t_bot = 350.0
        temp_grad = (t_bot - t_top) / (bot - top)
        return t_top + temp_grad * (depths - top)

    def spe10(self, idata: InputData, model_folder, uniform_props=False):
        self.mesh_filename = model_folder + '/spe10.msh'
        self.mesh_data = meshio.read(self.mesh_filename)

        self.set_uniform_initial_conditions(idata=idata)
        self.set_boundary_conditions(idata=idata)
        self.init_mech_discretizer(idata=idata)
        self.grav = -9.80665e-5
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)

        # specify initial temperature
        if self.thermoporoelasticity:
            self.depths = np.array([c.values[2] for c in self.centroids])
            self.t_init = self.get_reservoir_temperature(self.depths[:self.n_matrix])

        if uniform_props:
            self.init_uniform_properties(idata=idata)
        else:
            self.init_heterogeneous_properties(idata=idata)
        self.init_arrays_boundary_condition()
        self.update_boundary_conditions()

        # Discretization
        self.timer.node["discretization"] = timer_node()
        self.timer.node["discretization"].start()
        if self.thermoporoelasticity:
            self.discr.reconstruct_pressure_temperature_gradients_per_cell(self.cpp_flow, self.cpp_heat)
        else:
            self.discr.reconstruct_pressure_gradients_per_cell(self.cpp_flow)
        self.discr.reconstruct_displacement_gradients_per_cell(self.cpp_bc)
        self.discr.calc_interface_approximations()
        self.discr.calc_cell_centered_stress_velocity_approximations()
        self.timer.node["discretization"].stop()

    def set_boundary_conditions(self, idata: InputData):
        self.F = -900.0
        self.boundary_conditions = {}
        self.boundary_conditions[idata.mesh.bnd_tags['BND_X-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_X+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Y-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Y+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.LOAD(self.F, [0.0, 0.0, 0.0]) }

        if self.thermoporoelasticity:
            for key, bc in self.boundary_conditions.items():
                bc['temp'] = self.bc_type.AQUIFER(0.0)

    def update_boundary_conditions(self):
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            # flow
            self.bc_rhs[self.n_bc_vars * ids + self.p_bc_var] = bc['flow']['r']
            # energy
            if self.thermoporoelasticity:
                self.bc_rhs[self.n_bc_vars * ids + self.t_bc_var] = \
                    self.get_reservoir_temperature(self.depths[ids + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]])
            # mechanics
            for id in ids:
                assert (self.adj_matrix_cols[self.id_sorted[id]] == id +
                        self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0])
                conn = self.conns[self.id_boundary_conns[id]]
                n = np.array(conn.n.values)
                conn_c = np.array(conn.c.values)
                c1 = np.array(self.centroids[conn.elem_id1].values)
                if n.dot(conn_c - c1) < 0: n *= -1.0
                self.bc_rhs[self.n_bc_vars * id + self.u_bc_var:self.n_bc_vars * id + self.u_bc_var + self.n_dim] = \
                    bc['mech']['rn'] * n + bc['mech']['rt']

    def init_heterogeneous_properties(self, idata: InputData):
        '''
        set matrix properties using InputData
        :return:
        '''
        self.porosity = idata.rock.porosity
        lam, mu = get_lambda_mu(E=idata.rock.E, nu=idata.rock.nu)
        self.cs = idata.rock.compressibility

        self.hcap = np.zeros(self.n_matrix + self.n_fracs)
        for i, cell_id in enumerate(range(self.discr_mesh.region_ranges[elem_loc.MATRIX][0],
                                          self.discr_mesh.region_ranges[elem_loc.MATRIX][1])):
            permx = idata.rock.permx[3 * cell_id]
            permy = idata.rock.permy[3 * cell_id + 1]
            permz = idata.rock.permz[3 * cell_id + 2]
            self.discr.perms.append(disc_matrix33(permx, permy, permz))
            self.discr.biots.append(disc_matrix33(idata.rock.biot))
            self.discr.stfs.append(disc_stiffness(lam[cell_id], mu[cell_id]))
            if self.thermoporoelasticity:
                self.discr.heat_conductions.append(disc_matrix33(idata.rock.conductivity))
                self.discr.thermal_expansions.append(disc_matrix33(idata.rock.th_expn[cell_id]))

    def write_to_vtk(self, output_directory, ith_step, engine):
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

        # Allocate empty new cell_data dictionary:
        property_array = np.array(engine.X, copy=False)
        props_num = self.n_vars
        available_matrix_geometries_cpp = [elem_type.HEX, elem_type.PRISM, elem_type.TETRA, elem_type.PYRAMID]
        available_fracture_geometries_cpp = [elem_type.QUAD, elem_type.TRI]
        available_matrix_geometries = {'hexahedron': elem_type.HEX,
                                       'wedge': elem_type.PRISM,
                                       'tetra': elem_type.TETRA,
                                       'pyramid': elem_type.PYRAMID}
        available_fracture_geometries = ['quad', 'triangle']

        # Stresses and velocities
        engine.eval_stresses_and_velocities()
        total_stresses = np.array(engine.total_stresses, copy=False)

        # Matrix
        cells = []
        cell_data = {}
        for cell_block in self.mesh_data.cells:
            if cell_block.type in available_matrix_geometries:
                cells.append(cell_block)
                cell_ids = np.array(self.discr_mesh.elem_type_map[available_matrix_geometries[cell_block.type]], dtype=np.int64)
                for i in range(props_num):
                    if self.cell_property[i] not in cell_data: cell_data[self.cell_property[i]] = []
                    cell_data[self.cell_property[i]].append(property_array[props_num * cell_ids + i])

                if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []
                cell_data['tot_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for i in range(6):
                    cell_data['tot_stress'][-1][:, i] = total_stresses[i::6]

                if ith_step == 0:
                    if 'perm' not in cell_data: cell_data['perm'] = []
                    if 'E' not in cell_data: cell_data['E'] = []
                    if 'poro' not in cell_data: cell_data['poro'] = []
                    cell_data['perm'].append(np.zeros((len(cell_ids), 9), dtype=np.float64))
                    cell_data['E'].append(np.zeros(len(cell_ids), dtype=np.float64))
                    cell_data['poro'].append(np.array(self.mesh.poro, copy=False)[:self.n_matrix])
                    for i, cell_id in enumerate(cell_ids):
                        cell_data['perm'][-1][i] = np.array(self.discr.perms[cell_id].values)
                        stf = np.array(self.discr.stfs[cell_id].values)
                        la = stf[1]
                        mu = (stf[0] - la) / 2
                        E = mu * (3 * la + 2 * mu) / (la + mu)
                        cell_data['E'][-1][i] = E

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            self.mesh_data.points,
            cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)

        return 0
