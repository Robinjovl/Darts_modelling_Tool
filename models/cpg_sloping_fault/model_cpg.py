import numpy as np
import time
import os
import pandas as pd

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array, read_arrays
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
from darts.engines import value_vector

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.gen_cpg_grid import gen_cpg_grid

from darts.models.cicd_model import CICDModel

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer as PropertyContainerGeothermal


def get_case_files(case: str):
    prefix = r'meshes/case_' + case
    gridfile = prefix + r'/grid.grdecl'
    propfile = prefix + r'/reservoir.in'
    sch_file = prefix + r'/SCH.INC'
    assert os.path.exists(gridfile)
    assert os.path.exists(propfile)
    assert os.path.exists(sch_file)
    return gridfile, propfile, sch_file

#####################################################
class Model(CICDModel):
    def __init__(self, discr_type='cpp', case='generate', n_points=100):
        super().__init__()
        self.n_points = n_points
        self.discr_type = discr_type
        self.generate_grid = case == 'generate'

        if self.generate_grid:
            self.nx = 51
            self.ny = 51
            self.nz = 1
            self.dx = 4000. / self.nx
            self.dy = self.dx
            self.dz = 100. / self.nz
            poro = 0.2
            permx = 100
            permy = 100
            permz = 10
            start_z = 2000  # top reservoir depth
            depth_1d = start_z + (0.5 + np.arange(self.nz)) * self.dz
            # make 3d depth array
            depth = np.array([])
            for d in depth_1d:
                depth = np.append(depth, np.zeros(self.nx * self.ny) + d)
        else:  # read from files
            gridfile, propfile, sch_fname = get_case_files(case)
            self.gridfile = gridfile
            self.propfile = gridfile if propfile == '' else propfile
            self.sch_fname = sch_fname

        hcap = 2200
        rcond = 181.44

        if discr_type == 'cpp':
            if self.generate_grid:
                gridname = None; propname = None
                #gridname = 'grid.grdecl'; propname = 'reservoir.in'  # uncomment if want to save generated grid to grdecl files
                arrays = gen_cpg_grid(nx=self.nx, ny=self.ny, nz=self.nz,
                                      dx=self.dx, dy=self.dy, dz=self.dz, start_z=start_z,
                                      permx=permx, permy=permy, permz=permz, poro=poro,
                                      gridname=gridname, propname=propname)
            else:
                arrays = read_arrays(self.gridfile, self.propfile)
                # set inactive cells with small porosity (isothermal case)
                # arrays['ACTNUM'][arrays['PORO'] < 1e-5] = 0
                # process cells with small poro (thermal case)
                # arrays['PORO'][arrays['PORO'] < 1e-5] = 1e-5

            self.reservoir = CPG_Reservoir(self.timer, arrays)
            self.reservoir.discretize()
            self.reservoir.hcap[:] = hcap
            self.reservoir.conduction[:] = rcond
        elif discr_type == 'python':
            if case == 'generate':
                self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz,
                                                 dx=self.dx, dy=self.dy, dz=self.dz,
                                                 permx=permx, permy=permy, permz=permz, poro=poro,
                                                 hcap=hcap, rcond=rcond,
                                                 depth=depth)
            else:
                self.set_reservoir()

        self.set_physics()

        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=90, runtime=300, tol_newton=1e-3, tol_linear=1e-6)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        dims_cpp = index_vector_discr()
        load_single_int_keyword(dims_cpp, self.gridfile, "SPECGRID", 3)
        dims = np.array(dims_cpp, copy=False)

        permx_cpp, permy_cpp, permz_cpp = value_vector_discr(), value_vector_discr(), value_vector_discr()
        load_single_float_keyword(permx_cpp, self.propfile, 'PERMX', -1)
        load_single_float_keyword(permy_cpp, self.propfile, 'PERMY', -1)
        permx = np.array(permx_cpp, copy=False)
        permy = np.array(permy_cpp, copy=False)
        for perm_str in ['PERMEABILITYXY', 'PERMEABILITY']:
            if permx.size == 0 or permy.size == 0:
                load_single_float_keyword(permx_cpp, self.propfile, perm_str, -1)
                permy_cpp = permx_cpp
                permx = np.array(permx_cpp, copy=False)
                permy = np.array(permy_cpp, copy=False)
        load_single_float_keyword(permz_cpp, self.propfile, 'PERMZ', -1)
        permz = np.array(permz_cpp, copy=False)

        poro_cpp = value_vector_discr()
        load_single_float_keyword(poro_cpp, self.propfile, 'PORO', -1)
        poro = np.array(poro_cpp, copy=False)

        coord_cpp = value_vector_discr()
        load_single_float_keyword(coord_cpp, self.gridfile, 'COORD', -1)
        coord = np.array(coord_cpp, copy=False)

        zcorn_cpp = value_vector_discr()
        load_single_float_keyword(zcorn_cpp, self.gridfile, 'ZCORN', -1)
        zcorn = np.array(zcorn_cpp, copy=False)

        actnum_cpp = index_vector_discr()
        actnum = np.array([])
        for fname in [self.gridfile, self.propfile]:
            if actnum.size == 0:
                load_single_int_keyword(actnum_cpp, fname, 'ACTNUM', -1)
                actnum = np.array(actnum_cpp, copy=False)
        if actnum.size == 0:
            actnum = np.ones(dims[0] * dims[1] * dims[2])
            print('No ACTNUM found in input files. ACTNUM=1 will be used')

        depth = 0
        dx, dy, dz = [], [], []
        dx, dy, dz = 0, 0, 0

        # make cells with zero porosity (make sense if not thermal)
        # self.actnum[self.poro == 0.0] = 0

        # makes sense for thermal
        #self.poro = np.array(self.reservoir.mesh.poro, copy=False)
        #self.poro[self.poro == 0.0] = 1.E-4

        self.reservoir = StructReservoir(self.timer, nx=dims[0], ny=dims[1], nz=dims[2], dx=dx, dy=dy, dz=dz,
                                         permx=permx, permy=permy, permz=permz, poro=poro, hcap=2200, rcond=181.44,
                                         depth=depth, actnum=actnum, coord=coord, zcorn=zcorn, is_cpg=True)
        return

    def set_wells(self):
        # add wells
        if not self.generate_grid:
            self.read_and_add_perforations(self.reservoir, sch_fname=self.sch_fname, verbose=True)
        else:
            i1, j1 = self.nx//2 - int(500//self.dx), self.ny//2  # # I = 0.5 km to the left from the center
            self.reservoir.add_well('PRD')
            for k in range(1, self.reservoir.nz+1):
                self.reservoir.add_perforation('PRD', cell_index=(i1, j1, k), well_index=None, multi_segment=False,
                                               verbose=True)

            i2, j2 = self.nx//2 + int(500//self.dx), self.ny//2  # I = 0.5 km to the right from the center
            self.reservoir.add_well('INJ')
            for k in range(1, self.reservoir.nz+1):
                self.reservoir.add_perforation('INJ', cell_index=(i2, j2, k), well_index=None, multi_segment=False,
                                               verbose=True)
            print('DX', self.dx, 'DY', self.dy, 'PRD:', i1, j1, 'INJ:', i2, j2)

    def set_physics(self):
        '''
        set Geothermal physics
        :return:
        '''
        # initialize physics for Geothermal
        property_container = PropertyContainerGeothermal()
        self.physics = Geothermal(timer=self.timer,
                                  n_points=101,        # number of OBL points
                                  min_p=50, max_p=400,       # pressure range
                                  min_e=1000, max_e=25000,  # enthalpy range
                                  cache=False
        )
        self.physics.add_property_region(property_container)

        T_init = 350.
        state_init = value_vector([200., 0.])
        enth_init = self.physics.property_containers[0].enthalpy_ev['total'](T_init).evaluate(state_init)
        self.initial_values = {self.physics.vars[0]: state_init[0],
                               self.physics.vars[1]: enth_init
                               }
        return

    def set_initial_pressure_from_file(self, fname):
        # set initial pressure
        p_cpp = value_vector()
        load_single_float_keyword(p_cpp, fname, 'PRESSURE', -1)
        p_file = np.array(p_cpp, copy=False)

        p_mesh = np.array(self.reservoir.mesh.pressure, copy=False)
        try:
            actnum = np.array(self.reservoir.actnum, copy=False) # CPG Reservoir
            #nb = self.reservoir.mesh.n_cells
        except:
            actnum = self.reservoir.global_data['actnum']  #Struct reservoir
        nb = self.reservoir.mesh.n_blocks
        p_mesh[:self.reservoir.mesh.n_res_blocks * 2] = p_file[actnum > 0]

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                if self.generate_grid:
                    w.control = self.physics.new_rate_water_inj(7500, 300)  # 7500 m3/day, 300 K
                    w.constraint = self.physics.new_bhp_water_inj(500, 300)  # 500 bars upper limit for bhp
                else:
                    w.control = self.physics.new_bhp_water_inj(250, 300)
            else:
                if self.generate_grid:
                    w.control = self.physics.new_rate_water_prod(7500)
                    w.constraint = self.physics.new_bhp_prod(50)
                else:
                    w.control = self.physics.new_bhp_prod(100)

    #TODO: combine this function with save_few_keywords
    def save_cubes(self, fname, arr_list = [], arr_names = []):
        '''
        arr - list of numpy arrays to save, size=nactive
        arr_names - list of array names (keyword)
        '''
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[0:self.reservoir.mesh.n_res_blocks * 2:2]
        print('P range:', P.min(), P.max())

        actnum = self.reservoir.global_data['actnum']
        suffix = 'struct'
        if type(self.reservoir) == CPG_Reservoir:
            suffix = 'cpg'

        fname_suf = fname + '_' + suffix + '.grdecl'

        arr_list_ = arr_list.copy()
        arr_list_ += [P]
        arr_names += ['PRESSURE']

        if suffix == 'cpg':
            local_to_global = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
            global_to_local = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)

            save_array(actnum, fname_suf, 'ACTNUM', local_to_global, global_to_local, 'w')
            for i in range(len(arr_list)):
                save_array(arr_list_[i], fname_suf, arr_names[i], local_to_global, global_to_local, 'a')
        else:
            print('save_array is not implemented yet for Struct Reservoir')
            return
            save_array(actnum, fname_suf, 'ACTNUM', actnum, 'w')
            for i in range(len(arr_list_)):
                save_array(arr_list[i], fname_suf, arr_names[i], actnum, 'a')

    def read_and_add_perforations(self, reservoir, sch_fname, well_index: float = None, verbose: bool = False):
        '''
        read COMPDAT from SCH file in Eclipse format, add wells and perforations
        note: uses only I,J,K1,K2 parameters from COMPDAT
        '''
        if sch_fname is None:
            return
        print('reading wells (COMPDAT) from', sch_fname)
        well_dia = 0.152
        well_rad = well_dia / 2

        keep_reading = True
        prev_well_name = ''
        with open(sch_fname) as f:
            while keep_reading:
                buff = f.readline()
                if 'COMPDAT' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            CompDat = buff.split()
                            wname = CompDat[0].strip('"').strip("'") #remove quotas (" and ')
                            if len(CompDat) != 0 and '/' != wname:  # skip the empty line and '/' line
                                # define well
                                if wname == prev_well_name:
                                    pass
                                else:
                                    reservoir.add_well(wname)
                                    prev_well_name = wname
                                # define perforation
                                i1 = int(CompDat[1])
                                j1 = int(CompDat[2])
                                k1 = int(CompDat[3])
                                k2 = int(CompDat[4])

                                for k in range(k1, k2 + 1):
                                    reservoir.add_perforation(wname, cell_index=(i1, j1, k), well_radius=well_rad,
                                                              well_index=well_index, multi_segment=False, verbose=verbose)

                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break
        print('WELLS read from SCH file:', len(reservoir.wells))

    def print_well_rate(self):
        # set inj target rate for the next timestep with the production rate value from the previous timestep
        for i, w in enumerate(self.reservoir.wells):
            if not "I" in w.name:
                prod_well = w
            else:
                inj_well = w
        time_data = pd.DataFrame.from_dict(self.physics.engine.time_data)
        years = np.array(time_data['time'])[-1]/365.
        pr_col_name = time_data.filter(like=prod_well.name + ' : water rate').columns.to_list()
        pt_col_name = time_data.filter(like=prod_well.name + ' : temperature').columns.to_list()
        ir_col_name = time_data.filter(like=inj_well.name + ' : water rate').columns.to_list()
        rate_prod = np.array(time_data[pr_col_name])[-1][0]  # pick the last timestep value
        temp_prod = np.array(time_data[pt_col_name])[-1][0]  # pick the last timestep value
        rate_inj  = np.array(time_data[ir_col_name])[-1][0]  # pick the last timestep value
        print(years, 'years:', 'RATE_prod =', rate_prod, 'RATE_inj =', rate_inj, 'TEMP_prod =', temp_prod)

