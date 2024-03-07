import numpy as np
import os
import meshio
from darts.discretizer import elem_type
from darts.reservoirs.unstruct_reservoir_mech import set_domain_tags, get_lambda_mu, get_biot_modulus
from darts.reservoirs.unstruct_reservoir_mech import UnstructReservoirMech
from darts.input.input_data import InputData
from darts.engines import timer_node, ms_well_vector
import copy

class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, idata: InputData, model_folder, fluid_vars=['p']):
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        super().__init__(timer, discretizer='mech_discretizer', thermoporoelasticity=False, fluid_vars=fluid_vars)
        # self.n_vars = n_vars
        self.domain_tags, self.bnd_tags = set_domain_tags(matrix_tags=[99991],
                    bnd_xm_tag=991, bnd_xp_tag=992,
                    bnd_ym_tag=993, bnd_yp_tag=994,
                    bnd_zm_tag=995, bnd_zp_tag=996)

        self.spe10(model_folder=model_folder, idata=idata)
        self.init_reservoir_main(idata=idata)
        self.set_pzt_bounds(p=self.p_init, z=self.z_init, t=self.t_init)
        self.wells = []

    def spe10(self, idata: InputData, model_folder):
        self.mesh_filename = model_folder + '/spe10.msh'
        self.mesh_data = meshio.read(self.mesh_filename)

        self.set_uniform_initial_conditions(idata=idata)
        # self.F = -100.0  # bar * m
        self.set_boundary_conditions()
        self.init_mech_discretizer(idata=idata)
        self.grav = -9.80665e-5
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)
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

    def set_boundary_conditions(self):
        self.F = -900.0
        self.boundary_conditions = {}
        self.boundary_conditions[self.bnd_tags['BND_X-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[self.bnd_tags['BND_X+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[self.bnd_tags['BND_Y-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[self.bnd_tags['BND_Y+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[self.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[self.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.LOAD(self.F, [0.0, 0.0, 0.0]) }

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
                cell_ids = np.array(self.discr_mesh.elem_type_map[available_matrix_geometries[cell_block.type]], copy=False, dtype=np.int64)
                for i in range(props_num):
                    if self.cell_property[i] not in cell_data: cell_data[self.cell_property[i]] = []
                    cell_data[self.cell_property[i]].append(property_array[props_num * cell_ids + i])

                if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []
                cell_data['tot_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for i in range(6):
                    cell_data['tot_stress'][-1][:, i] = total_stresses[i::6]

                if ith_step == 0:
                    if 'perm' not in cell_data: cell_data['perm'] = []
                    cell_data['perm'].append(np.zeros((len(cell_ids), 9), dtype=np.float64))
                    for i, cell_id in enumerate(cell_ids):
                        cell_data['perm'][-1][i] = np.array(self.discr.perms[cell_id].values)

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            self.mesh_data.points,
            cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)

        return 0
