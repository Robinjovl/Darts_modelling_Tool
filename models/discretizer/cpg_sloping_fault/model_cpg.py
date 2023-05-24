import numpy as np
import time

from reservoir import CPG_Reservoir
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
from darts.engines import value_vector

from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.tools.keyword_file_tools import save_few_keywords
from cpg_tools import save_array

# inherit from darts-models/2ph_do model to use its physics; self.reservoir will be replaced in this file
# add path to import
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
darts_dir = os.path.dirname(os.path.dirname(current_dir))  # 2 levels up
model_dir = os.path.join(darts_dir, '2ph_do')
#model_dir = os.path.join(darts_dir, 'Uniform_Brugge')
sys.path.insert(0, model_dir)
from model import Model as DO_Model

#from model_3ph_bo import Model as BO_Model
class Model(DO_Model):
    def __init__(self, discr_type='cpp', gridfile='', propfile='', sch_fname='', n_points=1000):
        # measure time spend on reading/initialization
        #self.timer.node["initialization"].start()
        # call base class constructor
        #super().__init__(pvt='physics.in')
        super().__init__()
        self.n_points = n_points

        self.discr_type = discr_type
        self.gridfile = gridfile
        self.propfile = propfile
        self.sch_fname = sch_fname

        if discr_type == 'cpp':
            self.reservoir = CPG_Reservoir(self.gridfile, self.propfile)
        elif discr_type == 'python':
            self.init_struct_rsv()
        #self.timer.node["initialization"].stop()
    def init_struct_rsv(self):
        self.dims_cpp = index_vector_discr()
        load_single_int_keyword(self.dims_cpp, self.gridfile, "SPECGRID", 3)
        self.dims = np.array(self.dims_cpp, copy=False)

        self.permx_cpp, self.permy_cpp, self.permz_cpp = value_vector_discr(), value_vector_discr(), value_vector_discr()
        load_single_float_keyword(self.permx_cpp, self.propfile, 'PERMX', -1)
        load_single_float_keyword(self.permy_cpp, self.propfile, 'PERMY', -1)
        self.permx = np.array(self.permx_cpp, copy=False)
        self.permy = np.array(self.permy_cpp, copy=False)
        for perm_str in ['PERMEABILITYXY', 'PERMEABILITY']:
            if self.permx.size == 0 or self.permy.size == 0:
                load_single_float_keyword(self.permx_cpp, self.propfile, perm_str, -1)
                self.permy_cpp = self.permx_cpp
                self.permx = np.array(self.permx_cpp, copy=False)
                self.permy = np.array(self.permy_cpp, copy=False)
        load_single_float_keyword(self.permz_cpp, self.propfile, 'PERMZ', -1)
        self.permz = np.array(self.permz_cpp, copy=False)

        self.poro_cpp = value_vector_discr()
        load_single_float_keyword(self.poro_cpp, self.propfile, 'PORO', -1)
        self.poro = np.array(self.poro_cpp, copy=False)

        self.coord_cpp = value_vector_discr()
        load_single_float_keyword(self.coord_cpp, self.gridfile, 'COORD', -1)
        self.coord = np.array(self.coord_cpp, copy=False)

        self.zcorn_cpp = value_vector_discr()
        load_single_float_keyword(self.zcorn_cpp, self.gridfile, 'ZCORN', -1)
        self.zcorn = np.array(self.zcorn_cpp, copy=False)

        self.actnum_cpp = index_vector_discr()
        self.actnum = np.array([])
        for fname in [gridfile, propfile]:
            if self.actnum.size == 0:
                load_single_int_keyword(self.actnum_cpp, fname, 'ACTNUM', -1)
                self.actnum = np.array(self.actnum_cpp, copy=False)
        if self.actnum.size == 0:
            self.actnum = np.ones(self.dims[0] * self.dims[1] * self.dims[2])
            print('No ACTNUM found in input files. ACTNUM=1 will be used')

        self.depth = 0
        self.dx, self.dy, self.dz = [], [], []
        self.dx, self.dy, self.dz = 0, 0, 0

        # make cells with zero porosity (make sense if not thermal)
        # self.actnum[self.poro == 0.0] = 0

        # makes sense for thermal
        self.poro = np.array(self.reservoir.mesh.poro, copy=False)
        self.poro[self.poro == 0.0] = 1.E-4

        self.reservoir = StructReservoir(self.timer, nx=self.dims[0], ny=self.dims[1], nz=self.dims[2],
                                         dx=self.dx, dy=self.dy, dz=self.dz,
                                         permx=self.permx, permy=self.permy, permz=self.permz, poro=self.poro,
                                         depth=self.depth, actnum=self.actnum, coord=self.coord, zcorn=self.zcorn,
                                         is_cpg=True)

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
                                                    uniform_composition=[0.001])
                                                    #uniform_composition=[0.001225901537, 0.7711341309])
        #self.set_initial_pressure_from_file(self.gridfile)

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

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                w.control = self.physics.new_bhp_inj(250, self.inj)
            else:
                w.control = self.physics.new_bhp_prod(100)

    def add_wells(self, mode='generate', sch_fname=None, well_index=-1, verbose=False):
        self.read_and_add_perforations(sch_fname, well_index=well_index, verbose=verbose)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                w.control = self.physics.new_bhp_inj(250, value_vector([0.999]))
            else:
                w.control = self.physics.new_bhp_prod(100)

    def set_wells(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ" in w.name:
                w.control = self.physics.new_bhp_inj(250, value_vector([0.999]))
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
        try:
            actnum = np.array(self.reservoir.actnum, copy=False)  # CPG Reservoir doesn't have 'global_data' object
            suffix = 'cpg'
        except:
            actnum = self.reservoir.global_data['actnum']  # Struct Reservoir
            suffix = 'struct'
        fname_suf = fname + '_' + suffix + '.grdecl'

        arr_list += [P]
        arr_names += ['PRESSURE']

        save_array(actnum, fname_suf, 'ACTNUM', actnum, 'w')
        for i in range(len(arr_list)):
            save_array(arr_list[i], fname_suf, arr_names[i], actnum, 'a')

    def read_and_add_perforations(self, sch_fname, well_index=-1, verbose=False):
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
                                    self.reservoir.add_well(wname)
                                    prev_well_name = wname
                                # define perforation
                                i1 = int(CompDat[1])
                                j1 = int(CompDat[2])
                                k1 = int(CompDat[3])
                                k2 = int(CompDat[4])
                                for i in range(k1, k2 + 1):
                                    self.reservoir.add_perforation(self.reservoir.wells[-1],
                                                                   i1, j1, i,
                                                                   well_radius=well_rad, well_index=well_index,
                                                                   multi_segment=False, verbose=verbose)

                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break
        print('WELLS read from SCH file:', len(self.reservoir.wells))
