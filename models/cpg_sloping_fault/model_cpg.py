import numpy as np
import time

from darts.reservoirs.cpg_reservoir import CPG_Reservoir, save_array
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
from darts.engines import value_vector

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.keyword_file_tools import save_few_keywords

from darts.models.cicd_model import CICDModel
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic


# inherit from darts-models/2ph_do model to use its physics; self.reservoir will be replaced in this file
# add path to import
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
darts_dir = os.path.dirname(current_dir)  # 1 level up
model_dir = os.path.join(darts_dir, '2ph_do')
#model_dir = os.path.join(darts_dir, 'Uniform_Brugge')
sys.path.insert(0, model_dir)
from model import Model as DO_Model


#from model_3ph_bo import Model as BO_Model
# class Model(DO_Model):
class Model(CICDModel):
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
            reservoir = super().set_reservoir(CPG_Reservoir(self.timer, self.gridfile, self.propfile))
        elif discr_type == 'python':
            reservoir = self.set_reservoir()

        self.set_wells()

        self.set_physics()

        self.set_sim_params(first_ts=0.01, mult_ts=2, max_ts=5, runtime=300, tol_newton=1e-3, tol_linear=1e-6)

        self.timer.node["initialization"].stop()

        self.initial_values = {self.physics.vars[0]: 200,
                               self.physics.vars[1]: 0.001,
                               }

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

        reservoir = StructReservoir(self.timer, nx=dims[0], ny=dims[1], nz=dims[2], dx=dx, dy=dy, dz=dz,
                                    permx=permx, permy=permy, permz=permz, poro=poro,
                                    depth=depth, actnum=actnum, coord=coord, zcorn=zcorn, is_cpg=True)
        return super().set_reservoir(reservoir)

    def set_wells(self):
        # add wells
        if True:
            self.add_wells(self.reservoir, mode='read', sch_fname=self.sch_fname,
                           verbose=True)  # , well_index=1000) # add only perforations
        else:
            self.add_wells(self.reservoir, mode='generate', sch_fname=self.sch_fname)
        return super().set_wells()

    def set_physics(self):
        """Physical properties"""
        zero = 1e-13
        components = ['w', 'o']
        phases = ['wat', 'oil']

        self.inj = value_vector([zero])
        self.ini = value_vector([1 - zero])

        property_container = ModelProperties(phases_name=phases, components_name=components, min_z=zero/10)

        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('oil', DensityBasic(compr=5e-3, dens0=500))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('oil', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("wat", 0.1, 0.1)),
                                               ('oil', PhaseRelPerm("oil", 0.1, 0.1))])

        # create physics
        physics = Compositional(components, phases, self.timer,
                                n_points=400, min_p=0, max_p=1000, min_z=zero, max_z=1 - zero)
        physics.add_property_region(property_container)

        return super().set_physics(physics)

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

    def add_wells(self, reservoir, mode='generate', sch_fname: str = None, well_index: float = None, verbose: bool = False):
        self.read_and_add_perforations(reservoir, sch_fname, well_index=well_index, verbose=verbose)

    def set_well_controls(self):
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
        Xn = np.array(self.engine.X, copy=False)
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
                                    prev_well_name = wname
                                # define perforation
                                i1 = int(CompDat[1])
                                j1 = int(CompDat[2])
                                k1 = int(CompDat[3])
                                k2 = int(CompDat[4])

                                reservoir.add_well(wname)
                                for k in range(k1, k2 + 1):
                                    reservoir.add_perforation(wname, (i1, j1, k), well_radius=well_rad,
                                                              well_index=well_index, multi_segment=False, verbose=verbose)

                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break
        print('WELLS read from SCH file:', len(reservoir.wells))


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z, temperature=1.)

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        ph = [0, 1]

        for j in ph:
            # molar weight of mixture
            M = np.sum(self.x[j, :] * self.Mw)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu = zc
        self.compute_saturation(ph)

        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        mass_source = np.zeros(self.nc)

        return ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source

    def evaluate_at_cond(self, pressure, zc):
        self.sat[:] = 0

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m
