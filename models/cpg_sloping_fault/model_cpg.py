import numpy as np
import os

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, read_arrays, check_arrays, make_burden_layers, make_full_cube
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
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
        self.generate_grid = 'generate' in case

        if self.generate_grid:
            if case == 'generate_51x51x1':   # 4x4x0.1 km
                self.nx = 51
                self.ny = 51
                self.nz = 1
                self.dx = 4000. / self.nx
                self.dy = self.dx
                self.dz = 100. / self.nz
                self.start_z = 2000  # top reservoir depth
            elif case == 'generate_5x3x4':
                self.nx = 5
                self.ny = 3
                self.nz = 4
                self.start_z = 1000  # top reservoir depth
                # non-uniform layers thickness
                self.dx = np.array([500, 200, 100, 300, 500])
                self.dy = np.array([1000, 700, 300])
                self.dz = np.array([100, 150, 180, 120])
            elif case == 'generate_100x100x100':
                self.nx = self.ny = self.nz = 100
                self.dx = self.dy = 10
                self.dz = 1
                self.start_z = 2000  # top reservoir depth
            poro = 0.2
            permx = 100
            permy = 100
            permz = 10
        else:  # read from files
            # setup filenames
            gridfile, propfile, schfile = get_case_files(case)
            self.gridfile = gridfile
            self.propfile = gridfile if propfile == '' else propfile

        if self.generate_grid:
            if grid_out_dir is None:
                gridname = None
                propname = None
            else:  # save generated grid to grdecl files
                os.makedirs(grid_out_dir, exist_ok=True)
                gridname = os.path.join(grid_out_dir, 'grid.grdecl')
                propname = os.path.join(grid_out_dir, 'reservoir.in')
            arrays = gen_cpg_grid(nx=self.nx, ny=self.ny, nz=self.nz,
                                  dx=self.dx, dy=self.dy, dz=self.dz, start_z=self.start_z,
                                  permx=permx, permy=permy, permz=permz, poro=poro,
                                  gridname=gridname, propname=propname)
        else:
            # read grid and props.
            # Use read_arrays(self.gridfile, self.gridfile) if all the data is in a single file
            arrays = read_arrays(self.gridfile, self.propfile)
            check_arrays(arrays)
            if self.physics_type == 'dead_oil':  # set inactive cells with small porosity (isothermal case)
                arrays['ACTNUM'][arrays['PORO'] < 1e-5] = 0
            elif self.physics_type == 'geothermal':  # process cells with small poro (thermal case)
                for arr in ['PORO', 'PERMX', 'PERMY', 'PERMZ']:
                    arrays[arr][arrays['PORO'] < 1e-5] = 1e-5

        self.burden_layers = 0
        if self.physics_type == 'geothermal':
            self.burden_layers = 4
            # add over- and underburden layers
            make_burden_layers(number_of_burden_layers=self.burden_layers, initial_thickness=10, property_dictionary=arrays,
                               burden_layer_prop_value=1e-5)

        self.reservoir = CPG_Reservoir(self.timer, arrays, minpv=1e-1)
        self.reservoir.discretize()

        # store modified arrrays (with burden layers) for output to grdecl
        self.reservoir.input_arrays = arrays

        volume = np.array(self.reservoir.mesh.volume, copy=False)
        poro = np.array(self.reservoir.mesh.poro, copy=False)
        print("Pore volume = " + str(sum(volume[:self.reservoir.mesh.n_blocks] * poro)))

        # add "open" boundaries
        bv = 1e10   # boundary volume
        self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
        self.reservoir.apply_volume_depth()

        poro_shale_threshold = 1e-3
        poro = np.array(self.reservoir.mesh.poro)
        self.reservoir.conduction[poro <= poro_shale_threshold] = 2.2 * 86.4 # Shale conductivity kJ/m/day/K
        self.reservoir.conduction[poro > poro_shale_threshold] = 3 * 86.4 # Sandstone conductivity kJ/m/day/K
        self.reservoir.hcap[poro <= poro_shale_threshold] = 2300 # Shale heat capacity kJ/m3/K
        self.reservoir.hcap[poro > poro_shale_threshold] = 2450 # Sandstone heat capacity kJ/m3/K

        # add hcap and rcond to be saved into mesh.vtu
        l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
        self.reservoir.global_data.update({'heat_capacity': make_full_cube(self.reservoir.hcap, l2g, g2l),
                                           'rock_conduction': make_full_cube(self.reservoir.conduction, l2g, g2l) })

        self.set_physics()

        # time stepping and convergence parameters
        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=92, runtime=300, tol_newton=1e-2, tol_linear=1e-4)

        self.timer.node["initialization"].stop()

    def set_wells(self):
        # one can read well locations from a file
        #self.reservoir.read_and_add_perforations(self.sch_fname)

        # add wells and perforations, 1-based indices
        if self.case == 'generate_51x51x1':
            i1, j1 = self.nx // 2 - int(500 // self.dx), self.ny // 2  # I = 0.5 km to the left from the center
            i2, j2 = self.nx // 2 + int(500 // self.dx), self.ny // 2  # I = 0.5 km to the right from the center
        elif self.case == 'generate_5x3x4':
            i1, j1 = 1, 1
            i2, j2 = 5, 3
            #i1, j1, k1 = self.reservoir.get_ijk_from_xyz(250.0, 500.0, 890.0)
            #i2, j2, k2 = self.reservoir.get_ijk_from_xyz(1350.0, 1850.0, 1700.0)
        elif self.case == 'generate_100x100x100':
            i1, j1 = 50, 20
            i2, j2 = 50, 80
        elif self.case == 'brugge':
            i1, j1 = 41, 31  # production well
            i2, j2 = 96, 31  # injection well
        elif self.case == 'case_40x40x10':
            i1, j1 = 10, 20  # production well
            i2, j2 = 30, 20  # injection well
        elif self.case == 'your_case':
            pass

        self.reservoir.add_well('PRD')
        for k in range(1 + self.burden_layers,  self.reservoir.nz+1-self.burden_layers):
            self.reservoir.add_perforation('PRD', cell_index=(i1, j1, k), well_index=None, multi_segment=False,
                                           verbose=True)
        self.reservoir.add_well('INJ')
        for k in range(1 + self.burden_layers, self.reservoir.nz+1-self.burden_layers):
            self.reservoir.add_perforation('INJ', cell_index=(i2, j2, k), well_index=None, multi_segment=False,
                                           verbose=True)
        print('PRD well:', i1, j1, 'INJ well:', i2, j2)

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


