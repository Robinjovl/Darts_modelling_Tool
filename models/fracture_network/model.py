from darts.engines import value_vector, sim_params, well_control_iface
from darts.physics.geothermal.geothermal import Geothermal
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import enthalpy_to_temperature
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
import os
import numpy as np
import meshio
from darts.input.input_data import InputData

def fmt(x):
    return '{:.3}'.format(x)

# Here the Model class is defined (child-class from DartsModel) in which most of the data and properties for the
# simulation are defined, e.g. for the reservoir/physics/sim_parameters/etc.
class Model(CICDModel):
    def __init__(self, idata : InputData):
        # base class constructor
        super().__init__()
        self.idata = idata

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.bound_cond = idata.geom['well_loc_type']

        # Some permeability input data for the simulation
        poro = idata.rock.porosity  # Matrix porosity [-]
        frac_aper = idata.geom['frac_aper']  # Aperture of fracture cells (but also takes a list of apertures for each segment) [m]

        self.inj_well_coords = idata.geom['inj_well_coords']
        self.prod_well_coords = idata.geom['prod_well_coords']

        fname = '_' + idata.geom['mesh_prefix'] + '_' + str(idata.geom['char_len']) + '.msh'
        mesh_file = os.path.join('meshes_' + idata.geom['case_name'], idata.geom['case_name'] + fname)

        if idata.rock.perm_file is not None: # set heterogeneous permeability from a file
            permx = self.get_perm_unstr_from_struct_grid(idata.rock.perm_file, self.input_data)
            permy = permx
            permz = permx * 0.1
        else: # set homogeneous permeability
            permx = idata.rock.permx
            permy = idata.rock.permy
            permz = idata.rock.permz

        # initialize reservoir
        self.reservoir = UnstructReservoir(timer=self.timer, mesh_file=mesh_file,
                                           permx=permx, permy=permy, permz=permz,
                                           poro=poro,
                                           rcond=idata.rock.conductivity,
                                           hcap=idata.rock.heat_capacity,
                                           frac_aper=frac_aper)

        # parameters for fracture aperture computation depending on principal stresses
        if 'Sh_max' in idata.stress:
            self.reservoir.sh_max = idata.stress['Sh_max']
            self.reservoir.sh_min = idata.stress['Sh_min']
            self.reservoir.sh_max_azimuth = idata.stress['SHmax_azimuth']
            self.reservoir.sigma_c = idata.stress['sigma_c']

        # read mesh to get the number of fractures for tags specification
        # assume mesh is extruded and fractures have a quad shape
        # fracture tags start from 90000 according to .geo file generation code
        msh = meshio.read(mesh_file)
        c = msh.cell_data_dict['gmsh:physical']
        n_fractures = (np.unique(c['quad']) >= 90000).sum()
        n_fractures = n_fractures * (1 + int(idata.geom['overburden_layers']>0) + int(idata.geom['underburden_layers']>0))

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

        self.physics = Geothermal(self.idata, self.timer)

        # Some tuning parameters:
        self.set_sim_params(first_ts=1e-6, mult_ts=1.5, max_ts=60, tol_newton=1e-4, tol_linear=1e-5)
        self.params.newton_type = sim_params.newton_local_chop  # Type of newton method (related to chopping strategy?)
        self.params.newton_params = value_vector([0.2])  # Probably chop-criteria(?)
        # direct linear solver
        #if int(input_data['overburden_layers']) + int(input_data['underburden_layers']) > 0:
        #    self.params.linear_type = sim_params.cpu_superlu

        # End timer for model initialization:
        self.timer.node["initialization"].stop()

    def print_range(self, time, part='cells'):
        depth = np.array(self.reservoir.mesh.depth, copy=True)
        start, end = self.get_mat_frac_range(part)
        D = depth[start:end]
        P = self.get_pressure(part)
        T = self.get_temperature(part)
        suf = '(MAT)'  # matrix cells
        if part == 'full':
            suf = '(MAT+FRAC)'  # matrix+fracture cells
        elif part == 'fracs':
            suf = '(FRAC)'  # matrix+fracture cells
        print('Time', fmt(time/365.25), ' years; ', time, 'days, '
              'D_range:', fmt(D.min()), '-', fmt(D.max()), 'm; ',
              'P_range:', fmt(P.min()), '-', fmt(P.max()), 'bars; ',
              'T_range:', fmt(T.min()), '-', fmt(T.max()), 'K', suf)

    def set_initial_conditions(self):
        if self.idata.initial.type == 'gradient':
            # Specify reference depth, values and gradients to construct depth table in super().set_initial_conditions()
            input_depth = [0., np.amax(self.reservoir.mesh.depth)]
            input_distribution = {'pressure': [1., 1. + input_depth[1] * self.idata.initial.pressure_gradient / 1000],
                                  'temperature': [293.15, 293.15 + input_depth[
                                      1] * self.idata.initial.temperature_gradient / 1000]
                                  }
            return self.physics.set_initial_conditions_from_depth_table(self.reservoir.mesh,
                                                                        input_distribution=input_distribution,
                                                                        input_depth=input_depth)
        elif self.idata.initial.type == 'uniform':
            input_distribution = {'pressure': self.idata.initial.initial_pressure,
                                  'temperature': self.idata.initial.initial_temperature}
            return self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                                  input_distribution=input_distribution)

    def well_is_inj(self, wname : str):  # determine well control by its name
        return "I" in wname

    def set_well_controls(self):
        wctrl = self.idata.well_data.controls
        inj_rate = wctrl.inj_rate
        prod_rate = wctrl.prod_rate

        P = self.get_pressure('full')
        T = self.get_temperature('full')
        if P.size:
            self.pressure_initial_mean = P.mean()
            self.temperature_initial_mean = T.mean()
        else:
            self.pressure_initial_mean = self.idata.initial.initial_pressure
            self.temperature_initial_mean = self.idata.initial.initial_temperature

        if True:
            inj_bhp = self.pressure_initial_mean + wctrl.delta_p_inj
            inj_temp = self.temperature_initial_mean - wctrl.delta_temp
            prod_bhp = self.pressure_initial_mean - wctrl.delta_p_prod
        else:  # if engine is not initialized yet, set well control rate=0
            inj_temp = 0
            inj_rate = 0
            prod_rate = 0

        for i, w in enumerate(self.reservoir.wells):
            if self.well_is_inj(w.name):
                if inj_rate is None:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=True, target=inj_bhp, inj_composition=[], inj_temp=inj_temp)
                else:
                    # Control
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=True, target=inj_rate, phase_name='water', inj_composition=[], inj_temp=inj_temp)
                    # Constraint
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=True, target=wctrl.inj_bhp_constraint, inj_composition=[],
                                                   inj_temp=inj_temp)
            else:
                if prod_rate is None:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=False, target=prod_bhp)
                else:
                    # Control
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=False, target=-np.abs(prod_rate), phase_name='water')
                    # Constraint
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=False, target=wctrl.prod_bhp_constraint)

            # print(w.name,
            #       w.well_head_depth,
            #       w.control.target_pressure if hasattr(w.control, 'target_pressure') else '',
            #       w.control.target_temperature if hasattr(w.control, 'target_temperature') else '',
            #       w.control.target_rate if hasattr(w.control, 'target_rate') else '')
        return 0

    def get_mat_frac_range(self, part):
        start = 0
        if part == 'full':
            end = self.reservoir.discretizer.frac_cells_tot + self.reservoir.discretizer.mat_cells_tot
        elif part == 'cells':
            end = self.reservoir.discretizer.mat_cells_tot
            start = self.reservoir.discretizer.frac_cells_tot
        elif part == 'fracs':
            end = self.reservoir.discretizer.frac_cells_tot
        return [start, end]

    def get_pressure(self, part='cells'):
        nvars = 2
        start, end = self.get_mat_frac_range(part)
        Xn = np.array(self.physics.engine.X, copy=True)
        P = Xn[nvars*start:nvars*end:nvars]
        return P

    def get_temperature(self, part='cells'):
        nvars = 2
        start, end = self.get_mat_frac_range(part)
        Xn = np.array(self.physics.engine.X, copy=True)
        T = enthalpy_to_temperature(Xn[nvars*start:nvars*end])
        return T

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