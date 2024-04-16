# Section of the Python code where we import all dependencies on third party Python modules/libaries or our own
# libraries (exposed C++ code to Python, i.e. darts.engines && darts.physics)
from darts.engines import value_vector, sim_params
#from darts.models.physics.dead_oil import DeadOil
from darts.physics.geothermal.physics import Geothermal
from darts.models.darts_model import DartsModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.physics.properties.iapws.iapws_property import iapws_total_enthalpy_evalutor
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
import os
import numpy as np
import meshio

def fmt(x):
    return '{:.3}'.format(x)

# Here the Model class is defined (child-class from DartsModel) in which most of the data and properties for the
# simulation are defined, e.g. for the reservoir/physics/sim_parameters/etc.
class Model(DartsModel):
    def __init__(self, input_data, n_points=1000):
        """
        Class constructor of Model class
        :param n_points: number of discretization points for the parameter space
        """
        # Call base class constructor (see darts/models/darts_model.py for more info as well as the links in main.py
        # on OOP)
        super().__init__()
        self.input_data = input_data

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.bound_cond = input_data['well_loc_type']

        # Some permeability input data for the simulation
        poro = input_data['poro']  # Matrix porosity [-]
        frac_aper = input_data['frac_aper']  # Aperture of fracture cells (but also takes a list of apertures for each segment) [m]

        self.inj_well_coords = input_data['inj_well_coords']
        self.prod_well_coords = input_data['prod_well_coords']

        self.rate_prod = input_data['rate_prod']
        self.rate_inj = input_data['rate_inj']
        self.delta_temp = input_data['delta_temp']  # inj_temp = initial_temp - delta_temp
        self.delta_p_inj  = input_data['delta_p_inj']   # inj_bhp  = initial_pressure + delta_p_inj
        self.delta_p_prod = input_data['delta_p_prod']  # prod_bhp = initial_pressure - delta_p_prod

        self.perm_file = input_data['perm_file']

        fname = '_' + input_data['mesh_prefix'] + '_' + str(input_data['char_len']) + '.msh'
        mesh_file = os.path.join('meshes_' + input_data['case_name'], input_data['case_name'] + fname)

        # set heterogeneous/uniform permeability
        if self.perm_file is not None:
            permx = self.get_perm_unstr_from_struct_grid(self.perm_file, self.input_data)
            permy = permx
            permz = permx * 0.1
        else:
            permx = input_data['permx']
            permy = input_data['permy']
            permz = input_data['permz']

        # initialize reservoir
        self.reservoir = UnstructReservoir(timer=self.timer, mesh_file=mesh_file,
                                      permx=permx, permy=permy, permz=permz,
                                      poro=poro,
                                      rcond=input_data['conduction'],
                                      hcap=input_data['hcap'],
                                      frac_aper=frac_aper)

        # parameters for fracture aperture computation depending on principal stresses
        if 'Sh_max' in input_data:
            self.reservoir.sh_max = input_data['Sh_max']
            self.reservoir.sh_min = input_data['Sh_min']
            self.reservoir.sh_max_azimuth = input_data['SHmax_azimuth']
            self.reservoir.sigma_c = input_data['sigma_c']

        # read mesh to get the number of fractures for tags specification
        # assume mesh is extruded and fractures have a quad shape
        # fracture tags start from 90000 according to .geo file generation code
        msh = meshio.read(mesh_file)
        c = msh.cell_data_dict['gmsh:physical']
        n_fractures = (np.unique(c['quad']) >= 90000).sum()
        n_fractures = n_fractures * (1 + int(input_data['overburden_layers']>0) + int(input_data['underburden_layers']>0))

        # 9991 - rsv, 9992 - overburden, 9993 - underburden, 9994 - overburden2, 9995 - underburden2
        self.reservoir.physical_tags['matrix'] = [9991 + i for i in range(5)]
        # multiplied by 3 because physical surfaces for fracture are also in underburden and overburden
        self.reservoir.physical_tags['fracture'] = [90000 + i for i in range(n_fractures)]

        self.reservoir.physical_tags['boundary'] = [2, 1, 3, 4, 5, 6]  # order: Z- (bottom); Z+ (top) ; Y-; X+; Y+; X-

        '''     matrix_tag   surface_tag                             fracture_tag    test_case
                ----------      2     overburden2 top                                     }
                | 9994                    overburden2                                     }
                ----------      2     overburden top       ------------- 90003        }   }
                | 9992                    overburden       | FRACTURE  |              }   }case_1_burden_2
                ----------      2     reservoir top        |-----------| 90001    }   }case_1_burden
                | 9991                    RESERVOIR        | FRACTURE  | 90000    }case_1 }
                ----------      1     reservoir bottom     |-----------| 90002    }   }   }
                | 9993                    underburden      | FRACTURE  |              }   }
                ----------      1     underburden bottom   ------------- 90004        }   }
                | 9995                    underburden2                                    }
                ----------      1     underburden2 bottom                                 }
        '''

        # discretize
        self.reservoir.init_reservoir(verbose=True)

        # set boundary volume XY
        boundary_cells = []
        for bnd_tag in [1, 2, 3, 4, 5, 6]:
            boundary_cells += self.reservoir.discretizer.find_cells(bnd_tag, 'face')
        boundary_cells = np.array(boundary_cells) + self.reservoir.discretizer.frac_cells_tot
        #bnd_vol = 1e+8
        bnd_vol_mult = 5
        # for vtk output
        self.reservoir.discretizer.volume_all_cells[boundary_cells] *= bnd_vol_mult  # = bnd_vol
        # for engines
        np.array(self.reservoir.mesh.volume, copy=False)[boundary_cells] *= bnd_vol_mult # = bnd_vol

        # initialize physics
        self.cell_property = ['pressure', 'enthalpy', 'temperature']

        from darts.physics.geothermal.property_container import PropertyContainer
        property_container = PropertyContainer()
        property_container.output_props = {'T,degrees': lambda: property_container.temperature - 273.15}

        # Create rock_compaction object to set rock compressibility (it is 0 by default)
        from darts.physics.properties.iapws.custom_rock_property import custom_rock_compaction_evaluator
        property_container.rock = [value_vector([1, input_data['rock_compressibility'], 273.15])]
        property_container.rock_compaction_ev = custom_rock_compaction_evaluator(property_container.rock)

        self.physics = Geothermal(timer=self.timer, n_points=n_points, min_p=100, max_p=500,
                                  min_e=1000, max_e=25000, cache=False)
        self.physics.add_property_region(property_container)

        # Some tuning parameters:
        self.params.first_ts = 1e-6  # Size of the first time-step [days]
        self.params.mult_ts = 1.5  # Time-step multiplier if newton is converged (i.e. dt_new = dt_old * mult_ts)
        self.params.max_ts = 60  # Max size of the time-step [days]
        self.params.tolerance_newton = 1e-4  # Tolerance of newton residual norm ||residual||<tol_newt
        self.params.tolerance_linear = 1e-5  # Tolerance for linear solver ||Ax - b||<tol_linslv
        self.params.newton_type = sim_params.newton_local_chop  # Type of newton method (related to chopping strategy?)
        self.params.newton_params = value_vector([0.2])  # Probably chop-criteria(?)
        # direct linear solver
        #if int(input_data['overburden_layers']) + int(input_data['underburden_layers']) > 0:
        #    self.params.linear_type = sim_params.cpu_superlu

        self.runtime = 2000  # Total simulations time [days], this parameters is overwritten in main.py!

        # End timer for model initialization:
        self.timer.node["initialization"].stop()

    def print_range(self, time, full=0):
        depth = np.array(self.reservoir.mesh.depth, copy=True)
        if not full:
            nb = self.reservoir.discretizer.mat_cells_tot
            D = depth[:nb]
        else:
            D = depth
        P = self.get_pressure(full=full)
        T = self.get_temperature(full=full)
        suf = '(M)'  # matrix cells
        if full:
            suf = '(M+F)'  # matrix+fracture cells
        print('Time', fmt(time/365), ' years; ', time, 'days, '
              'D_range:', fmt(D.min()), '-', fmt(D.max()), 'm; ',
              'P_range:', fmt(P.min()), '-', fmt(P.max()), 'bars; ',
              'T_range:', fmt(T.min()), '-', fmt(T.max()), 'degrees', suf)

    def set_nonuniform_initial_conditions(self, mesh):
        """""
        Function to set non-uniform initial reservoir condition by p and T gradients
        Arguments:
            -mesh: mesh object
        uses global object input_data: dictionary with parameters
        """

        depth = np.array(mesh.depth, copy=True)
        print('depth:', depth.min(), '-', depth.max(), 'm.')
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = (depth - self.input_data['reference_depth_for_pressure']) * self.input_data['pressure_gradient'] + \
                      self.input_data['pressure_initial']

        enthalpy = np.array(mesh.enthalpy, copy=False)
        init_temperature = (depth - self.input_data['reference_depth_for_temperature']) * self.input_data['temperature_gradient'] + \
                      + self.input_data['temperature_initial'] + \
                      293.15  # convert to K

        self.pressure_initial_mean = pressure.mean()
        self.temperature_initial_mean = init_temperature.mean()

        for j in range(mesh.n_blocks):
            state = value_vector([pressure[j], 0])
            E = iapws_total_enthalpy_evalutor(init_temperature[j])
            enthalpy[j] = E.evaluate(state)

    def set_initial_conditions(self):
        """
        :return:
        """
        if self.input_data['initial_uniform'] == True:
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=self.input_data['uniform_pressure'],
                                                        uniform_temperature=self.input_data['uniform_temperature'])
            self.pressure_initial_mean = self.input_data['uniform_pressure']
            self.temperature_initial_mean = self.input_data['uniform_temperature']
        else:
            # Takes care of uniform initialization of pressure and temperature in this example:
            #self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=350,uniform_temperature=348.15)
            self.set_nonuniform_initial_conditions(self.reservoir.mesh)

        return 0

    def set_well_controls(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                # Add controls for injection well:
                # Specify both pressure and temperature (since it's upstream for injection well)
                if self.rate_inj is None:
                    w.control = self.physics.new_bhp_water_inj(self.pressure_initial_mean + self.delta_p_inj, self.temperature_initial_mean - self.delta_temp)
                else:
                    w.control = self.physics.new_rate_water_inj(self.rate_inj, self.temperature_initial_mean - self.delta_temp)
                    w.constraint = self.physics.new_bhp_water_inj(450, self.temperature_initial_mean - self.delta_temp)
            else:
                # Add controls for production well:
                # Specify bhp for particular production well:
                if self.rate_prod is None:
                    w.control = self.physics.new_bhp_prod(self.pressure_initial_mean - self.delta_p_prod)
                else:
                    w.control = self.physics.new_rate_water_prod(self.rate_prod)
                    w.constraint = self.physics.new_bhp_prod(50)
            print(w.name,
                  w.well_head_depth,
                  w.control.target_pressure if hasattr(w.control, 'target_pressure') else '',
                  w.control.target_temperature if hasattr(w.control, 'target_temperature') else '',
                  w.control.target_rate if hasattr(w.control, 'target_rate') else '')
        return 0


    def enthalpy_to_temperature(self, data): # data is X = (p, T)
        data_len = int(len(data) / 2) # 2: p,E
        T = np.zeros(data_len)
        T[:] = _Backward1_T_Ph_vec(data[::2] / 10, data[1::2] / 18.015)
        return T

    def get_pressure(self, full=0):
        '''
        if full == 1 return array with well blocks
        '''
        nb = self.reservoir.discretizer.mat_cells_tot
        Xn = np.array(self.physics.engine.X, copy=True)
        if full == 1:
          P = Xn[::2]
        else:
          P = Xn[:2*nb:2]
        return P

    def get_temperature(self, full=0):
        nb = self.reservoir.discretizer.mat_cells_tot
        Xn = np.array(self.physics.engine.X, copy=True)
        if full != 1:
          Xn = Xn[:2*nb]
        T = self.enthalpy_to_temperature(Xn) - 273.15  # to degrees
        return T

    def get_saturation(self, full=0):
        nb = self.reservoir.discretizer.mat_cells_tot
        Xn = np.array(self.physics.engine.X, copy=True)
        if full == 1:
          S = Xn[1::2]
        else:
          S = Xn[1:2*nb:2]
        return S

    def calc_well_loc(self):
        #TODO use idx = self.reservoir.find_cell_index(wc)
        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.reservoir.discretizer.volume_all_cells.size
        self.num_frac = self.reservoir.discretizer.frac_cells_tot
        self.num_mat = self.reservoir.discretizer.mat_cells_tot
        if self.bound_cond == 'wells_in_frac':
            offset = 0
            left_int = 0
            right_int = self.num_frac
        elif self.bound_cond == 'wells_in_mat':
            offset = self.num_frac
            left_int = self.num_frac
            right_int = self.num_frac + self.num_mat
        elif self.bound_cond == 'wells_in_nearest_cell':
            offset = 0
            left_int = 0
            right_int = self.num_frac + self.num_mat
        else:
            raise('error: wrong self.bound_cond')

        # Find closest control volume to dummy_well point:
        self.injection_wells = []
        dummy_well_inj = self.inj_well_coords

        self.store_dist_to_well_inj = np.zeros((len(dummy_well_inj),))
        self.store_coord_well_inj = np.zeros((len(dummy_well_inj), 3))
        ii = 0
        for ith_inj in dummy_well_inj:
            dist_to_well_point = np.linalg.norm(self.reservoir.discretizer.centroid_all_cells[left_int:right_int] - ith_inj,
                                                axis=1)
            cell_id = np.argmin(dist_to_well_point) + offset
            self.injection_wells.append(cell_id)

            self.store_coord_well_inj[ii, :] = self.reservoir.discretizer.centroid_all_cells[cell_id]
            self.store_dist_to_well_inj[ii] = np.min(dist_to_well_point)
            ii += 1

        self.production_wells = []
        dummy_well_prod = self.prod_well_coords

        self.store_dist_to_well_prod = np.zeros((len(dummy_well_prod),))
        self.store_coord_well_prod = np.zeros((len(dummy_well_prod), 3))
        ii = 0
        for ith_prod in dummy_well_prod:
            dist_to_well_point = np.linalg.norm(self.reservoir.discretizer.centroid_all_cells[left_int:right_int] - ith_prod,
                                                axis=1)
            cell_id = np.argmin(dist_to_well_point) + offset
            self.production_wells.append(cell_id)

            self.store_coord_well_prod[ii, :] = self.reservoir.discretizer.centroid_all_cells[cell_id]
            self.store_dist_to_well_prod[ii] = np.min(dist_to_well_point)
            ii += 1

        self.well_perf_loc = np.array([self.injection_wells, self.production_wells])

    def set_wells(self, well_index=100):
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """
        self.calc_well_loc()

        for i in range(len(self.well_perf_loc[0])):
            self.reservoir.add_well(f'I{i + 1}')
            self.reservoir.add_perforation(self.reservoir.wells[-1].name, cell_index=self.well_perf_loc[0][i],
                                 well_index=well_index, well_indexD=0, verbose=True)

        for i in range(len(self.well_perf_loc[1])):
            self.reservoir.add_well(f'P{i + 1}')
            self.reservoir.add_perforation(self.reservoir.wells[-1].name, cell_index=self.well_perf_loc[1][i],
                                 well_index=well_index, well_indexD=0, verbose=True)

    def get_perm_unstr_from_struct_grid(self, perm_file, input_data):
        # Set non-uniform permeability
        if perm_file != None:
            [xx, yy, perm_rect_2d] = np.load(perm_file, allow_pickle=True)
            perm_rect_1d = perm_rect_2d.flatten()  # TODO: check XY-order
            cntr = self.discretizer.centroid_all_cells[self.discretizer.fracture_cell_count:]
            z_middle = input_data['z_top'] + input_data['height_res'] * 0.5  # middle depth of the reservoir
            rect_grid = np.vstack((xx.flatten(), yy.flatten(), np.zeros(xx.flatten().shape) + z_middle)).transpose()
            perm_unstr = np.zeros(cntr.size)
            c_idx = 0
            for c in cntr:
                dist_to_point = np.linalg.norm(c - rect_grid, axis=1)
                cell_id = np.argmin(dist_to_point)
                perm_from_rect = perm_rect_1d.flatten()[cell_id]
                perm_unstr[c_idx] = perm_from_rect
                c_idx += 1
            return perm_unstr