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
    def __init__(self, timer, idata: InputData, model_folder, fluid_vars=['p'], uniform_props=False, generate_mesh=False):
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        thermoporoelasticity = True if 'temperature' in fluid_vars else False
        super().__init__(timer, discretizer='mech_discretizer',
                         thermoporoelasticity=thermoporoelasticity, fluid_vars=fluid_vars)

        self.bnd_tags = idata.mesh.bnd_tags
        self.domain_tags = set_domain_tags(matrix_tags=idata.mesh.matrix_tags, bnd_tags=list(self.bnd_tags.values()))

        self.spe10(model_folder=model_folder, idata=idata, uniform_props=uniform_props, generate_mesh=generate_mesh)
        self.init_reservoir_main(idata=idata)
        t1 = None
        if thermoporoelasticity:
            t1 = np.mean(self.t_init)
        self.set_pzt_bounds(p=np.mean(self.p_init), z=self.z_init, t=t1)
        self.wells = []

    def get_reservoir_initial_pressure(self, depths):
        #return 290. + 0.0 * depths  # uniform initial pressure
        return 1. + 0.1 * depths  # by gradient 0.1 bars/m

    def get_reservoir_initial_temperature(self, depths):
        #return 273.15 + 0. * depths # uniform initial temperature
        return 273.15 + 10 + 30. / 1000 * depths # by gradient 30 degrees/km

    def spe10(self, idata: InputData, model_folder, uniform_props=False, generate_mesh=False):

        self.mesh_filename = model_folder + '/spe10.msh'
        nx, ny, nz = list(map(int, os.path.basename(model_folder).split('_')))

        if generate_mesh:
            print('Mesh generation started')
            tags = dict()
            tags['BND_X-'] = 991
            tags['BND_X+'] = 992
            tags['BND_Y-'] = 993
            tags['BND_Y+'] = 994
            tags['BND_Z-'] = 995
            tags['BND_Z-F'] = 997
            tags['BND_Z+'] = 996
            tags['MATRIX_1'] = 99991
            tags['MATRIX_2'] = 99992

            # define permeable reservoir geometric boundaries
            self.rsv_top = 2100
            self.rsv_bottom = 2200
            #self.rsv_xy = 1000   # laterally limited
            self.rsv_xy = 100000  # "infinite" laterally

            if nx == 6: # for debugging, -4..4 km XY
                self.Xc = np.array([-4000, -2000, -1000, 0, 1000, 2000, 4000])
            elif nx == 16: # -4..4 km XY, dx = 100 m in the reservoir, outside 500-2000 m
                self.Xc = np.array([-4000, -2000, -1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000, 2000, 4000])
            elif nx == 28: # -15..15 km XY, dx = 100 m in the reservoir, outside 1000-7000 m
                self.Xc = np.array([-15000, -8000, -4000, -2000, -1000] + np.arange(-900, 1000, 100).tolist() + [1000, 2000, 4000, 8000, 15000])
            elif nx == 34: # -15..15 km XY, dx = 100 m in the reservoir, outside 1000-15000 m
                self.Xc = np.array([-15000,-8000,-4000,-2400,-1600,-1200,-1100,-1000] + np.arange(-900, 1000, 100).tolist() + [1000, 1100,1200, 1600, 2400, 4000,8000,15000])
            elif nx == 41: # 41x41
                pass
                #rsv = np.arange(-900, 1000, 200)
                #side = np.arange(1000, 6500, 1000)
                #self.Xc = np.hstack([-side, rsv, side])
            else:
                print('not found an option to mesh with nx = ', nx)
                exit(1)

            if nz == 5: # for debugging
                self.Zc = np.array([0, 1000, 2000, 2100, 2120, 3000])
            elif nz == 15:  # dz = 100-1000 m for over and underburden and 20m for the reservoir
                self.Zc = np.array([0, 1000, 1500, 2000, 2100, 2120, 2140, 2160, 2180, 2200, 2300, 2500, 3000, 4000, 5000, 6000])
            elif nz == 29:  # dz = 200 m for over and underburden and 20m for the reservoir
                self.Zc = np.hstack([np.arange(0, self.rsv_top, 200), np.arange(self.rsv_top, self.rsv_bottom, 20), np.arange(self.rsv_bottom, 5000, 200)])
            elif nz == 37:  # dz = 200 m for over and underburden and 20m for the reservoir
                self.Zc = np.hstack([np.arange(0, self.rsv_top, 150), np.arange(self.rsv_top, self.rsv_bottom, 20), np.arange(self.rsv_bottom, 5000, 150)])
            elif nz == 53:  # dz = 100 m for over and underburden and 20m for the reservoir
                self.Zc = np.hstack([np.arange(0, self.rsv_top, 100), np.arange(self.rsv_top, self.rsv_bottom, 20), np.arange(self.rsv_bottom, 5000, 100)])
            elif nz == 56:  # uniform dz = 100 m
                self.Zc = np.hstack([np.arange(0, self.rsv_top - 100, 100),
                                     self.rsv_top - 100,
                                     np.arange(self.rsv_top - 50, self.rsv_bottom, 20),
                                     self.rsv_bottom + 50,
                                     np.arange(self.rsv_bottom + 100, 5000, 100)])
            elif nz == 60:  # uniform dz = 100 m
                np.linspace(0, 6000, num=61)  # mesh Z range
            else:
                print('not found an option to mesh with nz = ', nz)
                exit(1)

            # refine by Z also around rsv
            #self.Zc = np.hstack([np.arange(0, self.rsv_top-100, 100), np.arange(self.rsv_top-100, self.rsv_bottom+100, 20),np.arange(self.rsv_bottom+100, 6000, 100)])

            # extend by Z more
            #self.Zc = np.hstack([np.arange(0, self.rsv_top, 100),np.arange(self.rsv_top, self.rsv_bottom, 20), np.arange(self.rsv_bottom, 6000, 100), np.array([6500, 10000, 15000])])

            # check case name ane generated arrays are consistent
            assert nx == self.Xc.size-1, "nx = {0}, Xc.size = {1}".format(nx, self.Xc.size)
            assert nz == self.Zc.size-1, "nz = {0}, Zc.size = {1}".format(nz, self.Zc.size)

            # check layers boundaries defined without layers deterioration
            assert np.unique(self.Xc).size == self.Xc.size, "Xc has duplicates {0}".format(self.Xc)
            assert np.unique(self.Zc).size == self.Zc.size, "Zc has duplicates {0}".format(self.Zc)

            print('nx = ', self.Xc.size-1, 'nz = ', self.Zc.size-1)
            print('self.rsv_top', self.rsv_top)
            print('self.rsv_bottom', self.rsv_bottom)
            print('self.rsv_xy', self.rsv_xy)
            print('Zc', self.Zc)

            self.Yc = self.Xc
            from gen_msh import generate_box_3d
            generate_box_3d(X=2000, Y=2000, Z=4000, NX=21, NY=21, NZ=21, tags=tags,  # XYZ are ignored since Xc, Yc, Zc are passed
                                       is_transfinite=True, is_recombine=True, Xc=self.Xc, Yc=self.Yc, Zc=self.Zc)
            print('Mesh generation finished')

        print('Mesh reading...')
        self.mesh_data = meshio.read(self.mesh_filename)
        print('Init reservoir...')
        self.set_uniform_initial_conditions(idata=idata)
        self.set_boundary_conditions(idata=idata)
        self.init_mech_discretizer(idata=idata)

        self.grav = 9.80665e-5
        self.init_gravity(gravity_on=True, gravity_coeff=self.grav)
        #self.init_gravity(gravity_on=True, gravity_coeff=0.)
        #self.init_gravity(gravity_on=False)

        self.depths = np.array([c.values[2] for c in self.centroids])
        # specify initial temperature and pressure, which will be used for well controls setting, but overridden in model.set_initial_conditions()
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
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER}
        else:     # free Z+
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z-']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.ROLLER }
            self.boundary_conditions[idata.mesh.bnd_tags['BND_Z+']] = {'flow': self.bc_type.NO_FLOW,  'mech': self.bc_type.FREE}

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
                self.discr.heat_conductions.append(disc_matrix33(idata.rock.conductivity))
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
        total_stresses = np.array(engine.total_stresses, copy=False)

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

                if 'tot_delta_stress' not in cell_data: cell_data['tot_delta_stress'] = []
                cell_data['tot_delta_stress'].append(np.zeros((self.n_matrix, 6), dtype=np.float64))
                for j in range(6):
                    cell_data['tot_delta_stress'][-1][:, j] = total_stresses[j::6] - self.tot_stress_initial[j::6]

                if 'delta_pressure' not in cell_data: cell_data['delta_pressure'] = []
                cell_data['delta_pressure'].append(np.zeros(self.n_matrix, dtype=np.float64))
                cell_data['delta_pressure'][-1][:] = pressure - self.pressure_initial

                if hasattr(self, 'temperature_initial'): # if thermal simulation
                    if 'delta_temperature' not in cell_data: cell_data['delta_temperature'] = []
                    cell_data['delta_temperature'].append(np.zeros(self.n_matrix, dtype=np.float64))
                    cell_data['delta_temperature'][-1][:] = temperature - self.temperature_initial

                if True:#ith_step == 0:
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

        from scipy.interpolate import griddata as gd

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

        from functools import reduce
        rsv = reduce(np.logical_and, [self.rsv_top <= centers_struct_z, centers_struct_z <= self.rsv_bottom,
                                      -self.rsv_xy <= centers_struct_y,  centers_struct_y <= self.rsv_xy,
                                      -self.rsv_xy <= centers_struct_x,  centers_struct_x <= self.rsv_xy])
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

    def create_vtk_wells(self, output_directory: str, prolongation=-3000, tube_radius=20):
        '''
        creates a file wells.vtk with a tube per well based on its first perforation
        :param output_directory:
        :return:
        '''
        return
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
            for p in w.perforations:
                well_block, res_block_local, well_index, well_indexD = p
                #c = self.centroids_all_cells[res_block_local].values
                c = np.array(self.centroids[res_block_local].values)
                c[2] = -c[2]
                cyl = create_tube(c, prolongation=prolongation, tube_radius=tube_radius)
                appendFilter.AddInputData(cyl)
                #prolongation = 0
                break  # use only the first perf

        # Update the append filter to combine the polydata
        appendFilter.Update()

        # Write the cylinders to a VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(well_vtk_filename)
        writer.SetInputConnection(appendFilter.GetOutputPort())
        writer.Write()