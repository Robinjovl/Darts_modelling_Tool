import numpy as np
import os

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, read_arrays, check_arrays, make_burden_layers, make_full_cube
from darts.discretizer import load_single_float_keyword
from darts.engines import value_vector

from darts.tools.gen_cpg_grid import gen_cpg_grid

from darts.models.cicd_model import CICDModel

def get_case_files(case: str):
    prefix = os.path.join('meshes', case)
    grid_file = os.path.join(prefix, 'grid.grdecl')
    prop_file = os.path.join(prefix, 'reservoir.in')
    sch_file = os.path.join(prefix, 'sch.inc')
    assert os.path.exists(grid_file)
    assert os.path.exists(prop_file)
    return grid_file, prop_file, sch_file

def fmt(x):
    return '{:.3}'.format(x)

#####################################################

class Model_CPG(CICDModel):
    def __init__(self, physics_type : str, case : str, grid_out_dir=None):
        super().__init__()
        self.physics_type = physics_type
        self.case = case

        self.set_input_data(case=case)

        if self.idata.generate_grid:
            if grid_out_dir is None:
                self.idata.gridname = None
                self.idata.propname = None
            else:  # save generated grid to grdecl files
                os.makedirs(grid_out_dir, exist_ok=True)
                self.idata.gridname = os.path.join(grid_out_dir, 'grid.grdecl')
                self.idata.propname = os.path.join(grid_out_dir, 'reservoir.in')
            arrays = gen_cpg_grid(nx=self.idata.geom.nx, ny=self.idata.geom.ny, nz=self.idata.geom.nz,
                                  dx=self.idata.geom.dx, dy=self.idata.geom.dy, dz=self.idata.geom.dz,
                                  start_z=self.idata.geom.start_z,
                                  permx=self.idata.rock.permx, permy=self.idata.rock.permy, permz=self.idata.rock.permz,
                                  poro=self.idata.rock.poro,
                                  gridname=self.idata.gridname, propname=self.idata.propname)
        else:
            # read grid and props.
            # Use read_arrays(self.idata.gridfile, self.idata.gridfile) if all the data is in a single file
            arrays = read_arrays(self.idata.gridfile, self.idata.propfile)
            check_arrays(arrays)
            if self.physics_type == 'dead_oil':  # set inactive cells with small porosity (isothermal case)
                arrays['ACTNUM'][arrays['PORO'] < 1e-5] = 0
            elif self.physics_type == 'geothermal':  # process cells with small poro (thermal case)
                for arr in ['PORO', 'PERMX', 'PERMY', 'PERMZ']:
                    arrays[arr][arrays['PORO'] < 1e-5] = 1e-5

        if self.physics_type == 'geothermal':
            # add over- and underburden layers
            make_burden_layers(number_of_burden_layers=self.idata.geom.burden_layers,
                               initial_thickness=self.idata.geom.burden_init_thickness,
                               property_dictionary=arrays,
                               burden_layer_prop_value=self.idata.rock.burden_prop)
        else:
            self.idata.geom.burden_layers = 0

        self.reservoir = CPG_Reservoir(self.timer, arrays, minpv=1e-5)
        self.reservoir.discretize()

        # store modified arrrays (with burden layers) for output to grdecl
        self.reservoir.input_arrays = arrays

        volume = np.array(self.reservoir.mesh.volume, copy=False)
        poro = np.array(self.reservoir.mesh.poro, copy=False)
        print("Pore volume = " + str(sum(volume[:self.reservoir.mesh.n_blocks] * poro)))

        # imitate open-boundaries with a large volume
        bv = 1e10   # volume, will be assigned to each boundary cell [m3]
        self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
        self.reservoir.apply_volume_depth()

        poro_shale_threshold = self.idata.rock.poro_shale_threshold  # short name
        poro = np.array(self.reservoir.mesh.poro)
        self.reservoir.conduction[poro <= poro_shale_threshold] = self.idata.rock.conduction_shale
        self.reservoir.conduction[poro > poro_shale_threshold] = self.idata.rock.conduction_sand
        self.reservoir.hcap[poro <= poro_shale_threshold] = self.idata.rock.hcap_shale
        self.reservoir.hcap[poro > poro_shale_threshold] = self.idata.rock.hcap_sand

        # add hcap and rcond to be saved into mesh.vtu
        l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
        self.reservoir.global_data.update({'heat_capacity': make_full_cube(self.reservoir.hcap, l2g, g2l),
                                           'rock_conduction': make_full_cube(self.reservoir.conduction, l2g, g2l) })

        self.set_physics()

        # time stepping and convergence parameters
        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=92, runtime=300, tol_newton=1e-2, tol_linear=1e-4)
        #self.params.linear_type = self.params.linear_solver_t.cpu_superlu

        self.timer.node["initialization"].stop()

    def set_wells(self):
        # read well locations from a file
        if hasattr(self.idata, 'schfile'):
            self.reservoir.read_and_add_perforations(self.idata.schfile)
        else:
            # add wells and perforations, 1-based indices
            for wname, wdata in self.idata.well_data.wells.items():
                self.reservoir.add_well(wname)
                for k in range(1 + self.idata.geom.burden_layers,  self.reservoir.nz+1-self.idata.geom.burden_layers):
                    self.reservoir.add_perforation(wname,
                                                   cell_index=(wdata.location.I, wdata.location.J, k),
                                                   well_index=None, multi_segment=False, verbose=True)

    def set_initial_pressure_from_file(self, fname : str):
        # set initial pressure
        p_cpp = value_vector()
        load_single_float_keyword(p_cpp, fname, 'PRESSURE', -1)
        p_file = np.array(p_cpp, copy=False)
        p_mesh = np.array(self.reservoir.mesh.pressure, copy=False)
        try:
            actnum = np.array(self.reservoir.actnum, copy=False) # CPG Reservoir
        except:
            actnum = self.reservoir.global_data['actnum']  #Struct reservoir
        p_mesh[:self.reservoir.mesh.n_res_blocks * 2] = p_file[actnum > 0]


    def save_grdecl(self, fname):
        '''
        saves cubes into a text file (grdecl format), nx*ny*nz values, I is the fastest index
        fname - file name to output
        '''
        arrays_save = self.get_arrays()
        actnum = self.reservoir.global_data['actnum']
        suffix = 'struct'
        if type(self.reservoir) == CPG_Reservoir:
            suffix = 'cpg'
        fname_suf = fname + '_' + suffix + '.grdecl'

        if suffix == 'cpg':
            local_to_global = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
            global_to_local = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)

            save_array(actnum, fname_suf, 'ACTNUM', local_to_global, global_to_local, 'w')
            for arr_name in arrays_save.keys():
                make_full = True
                if arr_name in ['SPECGRID', 'COORD', 'ZCORN']:
                    make_full = False
                save_array(arrays_save[arr_name], fname_suf, arr_name, local_to_global, global_to_local, 'a', make_full)
        else:
            print('save_array is not implemented yet for Struct Reservoir')
            return

    def well_is_inj(self, wname : str):  # determine well control by its name
        return "INJ" in wname

    def set_input_data(self, case):
        class InputDataGeom():  # to group geometry input data
            def __init__(self):
                pass
        self.idata.generate_grid = 'generate' in case
        self.idata.geom = InputDataGeom()
        geom = self.idata.geom  # a short name

        well_data = self.idata.well_data
        if self.idata.generate_grid:
            if case == 'generate_51x51x1':   # 4x4x0.1 km
                geom.nx = 51
                geom.ny = 51
                geom.nz = 1
                geom.dx = 4000. / geom.nx
                geom.dy = geom.dx
                geom.dz = 100. / geom.nz
                geom.start_z = 2000  # top reservoir depth
                geom.burden_layers = 4  # the number of overburden layers (= the number of underburden layers), used only in the thermal case
                # vertical wells locations, 1-based indices
                well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(geom.nx // 2 - int(500 // geom.dx), geom.ny // 2, -1)) # I = 0.5 km to the left from the center
                well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(geom.nx // 2 + int(500 // geom.dx), geom.ny // 2, -1))# I = 0.5 km to the right from the center
            elif case == 'generate_5x3x4':
                geom.nx = 5
                geom.ny = 3
                geom.nz = 4
                geom.start_z = 1000  # top reservoir depth
                # non-uniform layers thickness
                geom.dx = np.array([500, 200, 100, 300, 500])
                geom.dy = np.array([1000, 700, 300])
                geom.dz = np.array([100, 150, 180, 120])
                geom.burden_layers = 4
                well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(1, 1, -1))
                well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(3, 3, -1))
                #one might use wells.add_well(name='PRD', loc_type='xyz', loc_xyz=(250.0, 500.0, 890.0))
            elif case == 'generate_100x100x100':
                geom.nx = geom.ny = geom.nz = 100
                geom.dx = geom.dy = 10
                geom.dz = 1
                geom.start_z = 2000  # top reservoir depth
                geom.burden_layers = 4
                # vertical wells locations, 1-based indices
                well_data.add_well(name='PRD', loc_type='ijk', loc_ijk=(50, 20, -1))
                well_data.add_well(name='INJ', loc_type='ijk', loc_ijk=(50, 80, -1))
            elif self.case == 'your_case':
                pass  # add your case here

            self.idata.rock.poro = 0.2
            self.idata.rock.permx = 100  # mD
            self.idata.rock.permy = 100  # mD
            self.idata.rock.permz = 10   # mD
        else:  # read from files
            # setup filenames
            gridfile, propfile, schfile = get_case_files(case)
            self.idata.gridfile = gridfile
            self.idata.propfile = propfile if os.path.exists(propfile) else gridfile
            self.idata.schfile = schfile

        # rock compressibility
        self.idata.rock.compressibility = 1e-5  # [1/bars]
        self.idata.rock.compressibility_ref_p = 1 # [bars]
        self.idata.rock.compressibility_ref_T = 273.15  # [K]

        #########################################################################
        # only for the thermal case (Geothermal physics):
        geom.burden_init_thickness = 10  # first layer thickness, [m.]
        self.idata.rock.burden_prop = 1e-5  # perm and poro value for burden layers

        self.idata.rock.conduction_shale = 2.2 * 86.4 # Shale conductivity kJ/m/day/K
        self.idata.rock.conduction_sand = 3 * 86.4 # Sandstone conductivity kJ/m/day/K
        self.idata.rock.hcap_shale = 2300 # Shale heat capacity kJ/m3/K
        self.idata.rock.hcap_sand = 2450 # Sandstone heat capacity kJ/m3/K

        # the cells with lower poro will be treated as shale when setting the rock thermal properties
        self.idata.rock.poro_shale_threshold = 1e-3
        ############################################################################