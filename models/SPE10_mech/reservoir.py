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
from scipy.interpolate import griddata as gd
from functools import reduce

class UnstructReservoirCustom(UnstructReservoirMech):
    def __init__(self, timer, idata: InputData, model_folder, fluid_vars=['p'], uniform_props=False, generate_mesh=False):
        self.idata = idata
        
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        thermoporoelasticity = True if 'temperature' in fluid_vars else False
        super().__init__(timer, discretizer='mech_discretizer',
                         thermoporoelasticity=thermoporoelasticity, fluid_vars=fluid_vars)

        self.bnd_tags = idata.mesh.bnd_tags
        self.domain_tags = set_domain_tags(matrix_tags=idata.mesh.matrix_tags, bnd_tags=list(self.bnd_tags.values()))

        self.field_reservoir(model_folder=model_folder, idata=idata, uniform_props=uniform_props, generate_mesh=generate_mesh)
        self.init_reservoir_main(idata=idata)
        t1 = None
        if thermoporoelasticity:
            t1 = np.mean(self.t_init)
        self.set_pzt_bounds(p=np.mean(self.p_init), z=self.z_init, t=t1) #TODO how this mean() affects when gradient is applied
        self.wells = []


    #def init_reservoir(self, verbose=False): # dummy, just to make run
    #    pass
    #    #super.init_reservoir()

    def get_reservoir_initial_pressure(self, depths):
        return self.idata.initial.pressure_at_ref_depth + self.idata.initial.pressure_gradient * depths

    def get_reservoir_initial_temperature(self, depths):
        return self.idata.initial.temperature_at_ref_depth + self.idata.initial.temperature_gradient * depths

    def field_reservoir(self, idata: InputData, model_folder, uniform_props=False, generate_mesh=False):

        self.mesh_filename = os.path.join(model_folder, 'mesh.msh')
        nx, ny, nz = idata.other.nx, idata.other.ny, idata.other.nz

        if generate_mesh:
            print('Mesh generation started')
            # define permeable reservoir geometric boundaries
            self.rsv_top = idata.other.rsv_top
            self.rsv_bottom = idata.other.rsv_bottom
            self.rsv_xy = idata.other.rsv_xy
            self.rsv_x1 = idata.other.rsv_x1
            self.rsv_x2 = idata.other.rsv_x2
            self.rsv_y1 = idata.other.rsv_y1
            self.rsv_y2 = idata.other.rsv_y2
            self.Xc = idata.other.Xc
            self.Yc = idata.other.Yc
            self.Zc = idata.other.Zc

            # refine by Z also around rsv
            #self.Zc = np.hstack([np.arange(0, self.rsv_top-100, 100), np.arange(self.rsv_top-100, self.rsv_bottom+100, 20),np.arange(self.rsv_bottom+100, 6000, 100)])

            # extend by Z more
            #self.Zc = np.hstack([np.arange(0, self.rsv_top, 100),np.arange(self.rsv_top, self.rsv_bottom, 20), np.arange(self.rsv_bottom, 6000, 100), np.array([6500, 10000, 15000])])

            # check case name ane generated arrays are consistent
            assert nx == self.Xc.size-1, "nx = {0}, Xc.size = {1}".format(nx, self.Xc.size)
            assert ny == self.Yc.size-1, "ny = {0}, Yc.size = {1}".format(nx, self.Yc.size)
            assert nz == self.Zc.size-1, "nz = {0}, Zc.size = {1}".format(nz, self.Zc.size)

            # check layers boundaries defined without layers deterioration
            assert np.unique(self.Xc).size == self.Xc.size, "Xc has duplicates {0}".format(self.Xc)
            assert np.unique(self.Yc).size == self.Yc.size, "Yc has duplicates {0}".format(self.Yc)
            assert np.unique(self.Zc).size == self.Zc.size, "Zc has duplicates {0}".format(self.Zc)

            print('nx = ', self.Xc.size-1, 'ny = ', self.Yc.size-1, 'nz = ', self.Zc.size-1)
            print('self.rsv_top', self.rsv_top)
            print('self.rsv_bottom', self.rsv_bottom)
            print('self.rsv_xy', self.rsv_xy)
            print('Zc', self.Zc)

            from gen_msh import generate_box_3d
            generate_box_3d(X=2000, Y=2000, Z=4000, NX=21, NY=21, NZ=21, tags=idata.mesh.tags,  # XYZ are ignored since Xc, Yc, Zc are passed
                                       is_transfinite=True, is_recombine=True, Xc=self.Xc, Yc=self.Yc, Zc=self.Zc)
            print('Mesh generation finished')

        print('Mesh reading...')
        self.mesh_data = meshio.read(self.mesh_filename)
        print('Init reservoir...')
        #self.set_uniform_initial_conditions(idata=idata)
        self.u_init = [0., 0., 0.]  # [m]
        self.p_init = None
        self.z_init = None
        self.set_boundary_conditions(idata=idata)
        self.init_mech_discretizer(idata=idata)

        self.grav = 9.80665e-5
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)
        #self.init_gravity(gravity_on=False)

        self.depths = np.array([c.values[2] for c in self.centroids])
        # specify initial temperature and pressure #TODO get it from model.set_initial_conditions()
        self.p_init = self.get_reservoir_initial_pressure(self.depths[:self.n_matrix])
        if self.thermoporoelasticity: # specify initial temperature
            self.t_init = self.get_reservoir_initial_temperature(self.depths[:self.n_matrix])

        if uniform_props:
            self.init_uniform_properties(idata=idata)
        else:
            self.set_heterogeneous_props_by_interpolation(idata=idata)
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
        print('Init reservoir finished')

    def set_boundary_conditions(self, idata: InputData):
        self.boundary_conditions = {}
        self.boundary_conditions[idata.mesh.bnd_tags['BND_X-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_X+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Y-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        self.boundary_conditions[idata.mesh.bnd_tags['BND_Y+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
        if True:  # free Z-
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.FREE }
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.STUCK(0.,0.)}
        else:     # free Z+ 
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.FREE}

        if self.thermoporoelasticity:
            for key, bc in self.boundary_conditions.items():
                bc['temp'] = self.bc_type.AQUIFER(0.0)

    def update_boundary_conditions(self):
        return
        for tag in self.domain_tags[elem_loc.BOUNDARY]:
            ids = np.where(self.tags == tag)[0] - self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]
            bc = self.boundary_conditions[tag]
            # flow
            self.bc_rhs[self.n_bc_vars * ids + self.p_bc_var] = bc['flow']['r']
            # energy
            if self.thermoporoelasticity:
                # keep initial temperature at the boundary
                self.bc_rhs[self.n_bc_vars * ids + self.t_bc_var] = \
                    self.get_reservoir_initial_temperature(self.depths[ids + self.discr_mesh.region_ranges[elem_loc.BOUNDARY][0]])
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
            #permx = idata.rock.permx[3 * cell_id]
            #permy = idata.rock.permy[3 * cell_id + 1]
            #permz = idata.rock.permz[3 * cell_id + 2]
            permx = idata.rock.permx[cell_id]
            permy = idata.rock.permy[cell_id]
            permz = idata.rock.permz[cell_id]
            self.discr.perms.append(disc_matrix33(permx, permy, permz))
            self.discr.biots.append(disc_matrix33(idata.rock.biot))
            self.discr.stfs.append(disc_stiffness(lam[cell_id], mu[cell_id]))
            if self.thermoporoelasticity:
                self.discr.heat_conductions.append(disc_matrix33(idata.rock.thermal_conductivity))
                self.discr.thermal_expansions.append(disc_matrix33(idata.rock.th_expn))#[cell_id]))

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
        total_stresses = -np.array(engine.total_stresses, copy=False)# make positive for compressive stresses

        if not hasattr(self, 'displs_initial'):
            self.displs_initial = dict()
        if not hasattr(self, 'tot_stress_initial'):
            self.tot_stress_initial = total_stresses.copy() 

        # Matrix
        cells = []
        cell_data = {}
        for cell_block in self.mesh_data.cells:
            if cell_block.type in available_matrix_geometries:
                cells.append(cell_block)
                cell_ids = np.array(self.discr_mesh.elem_type_map[available_matrix_geometries[cell_block.type]], dtype=np.int64)
                for i in range(props_num):
                    if self.cell_property[i] in ['ux', 'uy', 'uz']:
                        if self.cell_property[i] not in self.displs_initial:
                            self.displs_initial[self.cell_property[i]] = property_array[props_num * cell_ids + i]
                    if self.cell_property[i] == 'pressure':
                        pressure = property_array[props_num * cell_ids + i]
                        if not hasattr(self, 'pressure_initial') :
                            self.pressure_initial = property_array[props_num * cell_ids + i].copy()
                    if self.cell_property[i] == 'temperature':
                        temperature = property_array[props_num * cell_ids + i]
                        if not hasattr(self, 'temperature_initial'):
                            self.temperature_initial = property_array[props_num * cell_ids + i].copy()

                    if self.cell_property[i] not in cell_data: cell_data[self.cell_property[i]] = []
                    if self.cell_property[i] in ['ux', 'uy', 'uz']:  # eliminate displacements got after the initialization stage (equilibration)
                        cell_data[self.cell_property[i]].append(property_array[props_num * cell_ids + i] - self.displs_initial[self.cell_property[i]])
                    else:
                        cell_data[self.cell_property[i]].append(property_array[props_num * cell_ids + i])

                if 'tot_stress' not in cell_data: cell_data['tot_stress'] = []
                cell_data['tot_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for j in range(6):
                    cell_data['tot_stress'][-1][:, j] = total_stresses[j::6]

                if 'eff_stress' not in cell_data: cell_data['eff_stress'] = []
                cell_data['eff_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for j in range(6):
                    cell_data['eff_stress'][-1][:, j] = np.fabs(cell_data['tot_stress'][-1][:, j]) - self.idata.rock.biot * pressure

                if 'delta_tot_stress' not in cell_data: cell_data['delta_tot_stress'] = []
                cell_data['delta_tot_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for j in range(6):
                    cell_data['delta_tot_stress'][-1][:, j] = total_stresses[j::6] - self.tot_stress_initial[j::6]

                delta_pressure = pressure - self.pressure_initial
                if 'delta_pressure' not in cell_data: cell_data['delta_pressure'] = []
                cell_data['delta_pressure'].append(np.zeros(self.n_matrix, dtype=np.float64))
                cell_data['delta_pressure'][-1][:] = delta_pressure

                if 'delta_eff_stress' not in cell_data: cell_data['delta_eff_stress'] = []
                cell_data['delta_eff_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for j in range(6):
                    cell_data['delta_eff_stress'][-1][:, j] = cell_data['delta_tot_stress'][-1][:, j] - self.idata.rock.biot * delta_pressure

                if hasattr(self, 'temperature_initial'): # if thermal simulation
                    if 'delta_temperature' not in cell_data: cell_data['delta_temperature'] = []
                    cell_data['delta_temperature'].append(np.zeros(self.n_matrix, dtype=np.float64))
                    cell_data['delta_temperature'][-1][:] = temperature - self.temperature_initial

                if True:#ith_step == 0:
                    if 'perm' not in cell_data: cell_data['perm'] = []
                    if 'E' not in cell_data: cell_data['E'] = []
                    if 'poisson' not in cell_data: cell_data['poisson'] = []
                    if 'poro' not in cell_data: cell_data['poro'] = []
                    cell_data['perm'].append(np.zeros((len(cell_ids), 9), dtype=np.float64))
                    cell_data['E'].append(np.zeros(len(cell_ids), dtype=np.float64))
                    cell_data['poisson'].append(np.zeros(len(cell_ids), dtype=np.float64))
                    cell_data['poro'].append(np.array(self.mesh.poro, copy=False)[:self.n_matrix])
                    for i, cell_id in enumerate(cell_ids):
                        cell_data['perm'][-1][i] = np.array(self.discr.perms[cell_id].values)
                        stf = np.array(self.discr.stfs[cell_id].values)
                        la = stf[1]
                        mu = (stf[0] - la) / 2
                        E = mu * (3 * la + 2 * mu) / (la + mu)
                        poisson = la / (2 * (la + mu))
                        cell_data['E'][-1][i] = E
                        cell_data['poisson'][-1][i] = poisson

                # compute strain from stress and geomech props
                # https://en.wikipedia.org/wiki/Hooke%27s_law, In matrix form, Hooke's law for isotropic materials can be written as
                if 'strain' not in cell_data: cell_data['strain'] = []
                cell_data['strain'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                stress = cell_data['delta_eff_stress'][-1]
                E = cell_data['E'][-1]
                poisson = cell_data['poisson'][-1]
                cell_data['strain'][-1][:, 0] = -(stress[:, 0] - poisson * (stress[:, 1] + stress[:, 2])) / E
                cell_data['strain'][-1][:, 1] = -(stress[:, 1] - poisson * (stress[:, 0] + stress[:, 2])) / E
                cell_data['strain'][-1][:, 2] = -(stress[:, 2] - poisson * (stress[:, 0] + stress[:, 1])) / E
                for k in range(3,6):  # shear part
                    cell_data['strain'][-1][:, k] = -(2 + 2 * poisson) * stress[:, k]/E

        # Store solution for each time-step:
        mesh = meshio.Mesh(
            self.mesh_data.points,
            cells,
            cell_data=cell_data)
        meshio.write("{:s}/solution{:d}.vtu".format(output_directory, ith_step), mesh)

        time = engine.t if ith_step > 0 else 0.0
        self.write_pvd_file(ith_step, time, output_directory)

        return 0

    def set_heterogeneous_props_by_interpolation(self, idata):
        # set different values in the reservoir and lateral surrounding+over/under-burden
        # first, create a struct grid to easily set heterogeneous rock properties
        # second, interpolate them to unstructured mesh used for computation
        self.nx, self.ny, self.nz  = idata.other.nx, idata.other.ny, idata.other.nz

        # fill the whole array with non-rsv values, the rsv part will be replaced later on
        porosity_struct = np.zeros(self.nz * self.ny * self.nx) + idata.rock.poro_non_rsv
        permeability_struct = np.zeros(self.nz * self.ny * self.nx) + idata.rock.perm_non_rsv # mD
        E_struct = np.zeros(self.nz * self.ny * self.nx) + idata.rock.E_non_rsv # [bars]

        centers = np.array([np.array(c.values) for c in self.centroids[:self.n_matrix]])
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]

        # rsv cell centers
        xs = (self.Xc[1:] + self.Xc[:-1]) * 0.5
        ys = (self.Yc[1:] + self.Yc[:-1]) * 0.5
        zs = (self.Zc[1:] + self.Zc[:-1]) * 0.5

        centers_struct_x, centers_struct_y, centers_struct_z = np.meshgrid(xs, ys, zs)
        centers_struct_x, centers_struct_y, centers_struct_z = centers_struct_x.flatten(), centers_struct_y.flatten(), centers_struct_z.flatten()

        rsv = reduce(np.logical_and, [self.rsv_top <= centers_struct_z, centers_struct_z <= self.rsv_bottom,
                                      self.rsv_y1 <= centers_struct_y,  centers_struct_y <= self.rsv_y2,
                                      self.rsv_x1 <= centers_struct_x,  centers_struct_x <= self.rsv_x2])
        
        # set juxtaposed rsv
        if False:
            rsv_thickness = np.fabs(self.rsv_bottom - self.rsv_top)
            self.rsv_z_middle_1 = self.rsv_top + rsv_thickness * 0.25
            self.rsv_z_middle_2 = self.rsv_top + rsv_thickness * 0.75
            self.rsv_x_middle = (self.rsv_x1 + self.rsv_x2) * 0.5
            rsv_left = reduce(np.logical_and, [self.rsv_z_middle_1 <= centers_struct_z, centers_struct_z <= self.rsv_bottom,
                                          self.rsv_y1 <= centers_struct_y,  centers_struct_y <= self.rsv_y2,
                                          self.rsv_x1 <= centers_struct_x,  centers_struct_x <= self.rsv_x_middle])
            rsv_right = reduce(np.logical_and, [self.rsv_top <= centers_struct_z, centers_struct_z <= self.rsv_z_middle_2,
                                          self.rsv_y1 <= centers_struct_y,  centers_struct_y <= self.rsv_y2,
                                          self.rsv_x_middle <= centers_struct_x,  centers_struct_x <= self.rsv_x2])
            rsv = reduce(np.logical_or, [rsv_left, rsv_right])
        
        porosity_struct[rsv] = idata.rock.porosity
        permeability_struct[rsv] = idata.rock.permx # [mD]
        E_struct[rsv] = idata.rock.E #[bars]

        porosity = np.zeros(self.nz * self.ny * self.nx)
        permeability = np.zeros(self.nz * self.ny * self.nx)
        E = np.zeros(self.nz * self.ny * self.nx)

        arrays = [porosity, permeability, E]
        arrays_struct = [porosity_struct, permeability_struct, E_struct]

        for arr, arr_struct in zip(arrays, arrays_struct):
            arr[:] = gd((centers_struct_x, centers_struct_y, centers_struct_z), arr_struct, (x, y, z), method='nearest')

        # porosity = np.flip(np.swapaxes(porosity.reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()
        # permeability = np.flip(np.swapaxes(permeability.reshape((self.nz, self.ny, self.nx, 3)), 0, 2), axis=2).flatten()
        # E = np.flip(np.swapaxes(E.reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()
        #p_init = np.flip(np.swapaxes(p_init.reshape(self.nz, self.ny, self.nx), 0, 2), axis=2).flatten()

        idata.rock.porosity = porosity

        idata.rock.permx = idata.rock.permy = idata.rock.permz = permeability
        #permeability_xyz = np.zeros((self.nz * self.ny * self.nx, 3))
        #permeability_xyz[:, 0] = permeability_xyz[:, 1] =  permeability_xyz[:, 2] = permeability
        #idata.rock.permx = idata.rock.permy = idata.rock.permz = permeability_xyz

        idata.rock.E = E  # bars

    def decouple_geomech(self):
        '''
        turns off mechanics->porosity (so pressure and flow) influence
        :return:
        '''
        vol_strain_tran = np.array(self.mesh.vol_strain_tran, copy=False)
        vol_strain_rhs = np.array(self.mesh.vol_strain_rhs, copy=False)
        vol_strain_tran[:] = 0.0
        vol_strain_rhs[:] = 0.0

    def create_vtk_wells(self, output_directory: str, prolongation=-3000, tube_radius=20, dz=10):
        '''
        creates a file wells.vtk with a tube per well based on its first perforation
        :param output_directory:
        :return:
        '''
    
        import vtk
        well_vtk_filename = os.path.join(output_directory, 'wells.vtk')
        # Append multiple cylinders into one polydata
        appendFilter = vtk.vtkAppendPolyData()

        def create_tube(center, prolongation, tube_radius):
            # Create points for the polyline
            points = vtk.vtkPoints()
            points.InsertNextPoint(center[0], center[1], -center[2] + prolongation)  # Point 1
            points.InsertNextPoint(center[0], center[1], -center[2])  # Point 2

            # Create a polyline that connects the points
            lines = vtk.vtkCellArray()
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)  # Number of points
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            lines.InsertNextCell(line)

            # Create a polydata to hold the points and the polyline
            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(lines)

            # Apply vtkTubeFilter to create a tube around the polyline
            tubeFilter = vtk.vtkTubeFilter()
            tubeFilter.SetInputData(polyData)
            tubeFilter.SetRadius(tube_radius)  # Tube radius
            tubeFilter.SetNumberOfSides(50)  # Smoothness of the tube
            tubeFilter.Update()

            return tubeFilter.GetOutput()

        for w in self.wells:
            is_first = True 
            for p in w.perforations:
                well_block, res_block_local, well_index, well_indexD = p
                c = np.array(self.centroids[res_block_local].values, copy=True)
                c[2] = -c[2]
                if is_first:
                    cyl = create_tube(c, prolongation=prolongation, tube_radius=tube_radius)
                    appendFilter.AddInputData(cyl)
                    is_first = False
                c[2] -= dz * 0.5
                cyl = create_tube(c, prolongation=dz, tube_radius=tube_radius * 2)
                appendFilter.AddInputData(cyl)
                #break  # use only the first perf

        # Update the append filter to combine the polydata
        appendFilter.Update()

        # Write the cylinders to a VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(well_vtk_filename)
        writer.SetInputConnection(appendFilter.GetOutputPort())
        writer.Write()
        