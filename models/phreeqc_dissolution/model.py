from phreeqc_dissolution.conversions import convert_composition, correct_composition, calculate_injection_stream, \
    get_mole_fractions, convert_rate, bar2atm
from phreeqc_dissolution.physics import PhreeqcDissolution

import darts
from darts.models.cicd_model import CICDModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.unstruct_reservoir import UnstructReservoir

from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.basic import ConstFunc

from darts.engines import sim_params, well_control_iface, value_vector
from phreeqc_dissolution.conversions import bar2pa
from iapws._iapws import _Viscosity

import numpy as np
import pickle, h5py
import os, sys
from math import fabs
import warnings

try:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc

# Definition of your input parameter data structure,
# change as you see fit (when you need more constant values, etc.)!!
class MyOwnDataStruct:
    def __init__(self, nc, zmin, temp, stoich_matrix, pressure_init, kin_fact, n_init_ops, n_prop_ops, exp_w=1, exp_g=1):
        """
        Data structure class which holds various input parameters for simulation
        :param nc: number of components used in simulation
        :param zmin: actual 0 used for composition (usually >0, around some small epsilon)
        :param temp: temperature
        """
        self.num_comp = nc
        self.min_z = zmin
        self.temperature = temp
        self.stoich_matrix = stoich_matrix
        self.exp_w = exp_w
        self.exp_g = exp_g
        self.pressure_init = pressure_init
        self.kin_fact = kin_fact
        self.n_init_ops = n_init_ops
        self.n_prop_ops = n_prop_ops

# Actual Model class creation here!
class Model(CICDModel):
    def __init__(self, domain: str = '1D', nx: int = 200, mesh_filename: str = None, 
                 poro_filename: str = None, minerals: list = ['calcite'], 
                 kinetic_mechanisms=['acidic', 'neutral', 'carbonate'], 
                 n_obl_mult: int = 1, co2_injection: float = 0.1):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()
        self.minerals = minerals
        self.kinetic_mechanisms = kinetic_mechanisms
        self.n_obl_mult = n_obl_mult
        self.n_solid = len(minerals)
        self.co2_injection = co2_injection
        self.co2_injection_cutoff = 0.4

        self.set_reservoir(domain=domain, nx=nx, mesh_filename=mesh_filename, poro_filename=poro_filename)
        self.set_physics()

        self.set_sim_params(first_ts=1e-5, max_ts=1e-3, tol_newton=1e-4, tol_linear=1e-6, it_newton=20, it_linear=200)
        self.params.newton_type = sim_params.newton_local_chop
        self.params.newton_params[0] = 0.2
        self.runtime = 1

        self.timer.node["initialization"].stop()

    def set_physics(self):
        # some properties
        self.temperature = 323.15           # K
        self.pressure_init = 100            # bar

        if self.domain == '1D':
            self.inj_rate = self.volume * 24     # output: m3/day
        elif self.domain == '2D':
            self.inj_rate = 10 * self.volume * 24 / self.inj_cells.size     # output: m3/day
        elif self.domain == '3D':
            self.inj_rate = 1.12 * 60 * 24 / 1000 / 1000

        self.min_z = 1e-11
        self.obl_min = self.min_z / 10

        # Several parameters here related to components used, OBL limits, and injection composition:
        self.phases = ['gas', 'liq']

        if set(self.minerals) == {'calcite'}:
            # purely for initialization
            self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid_CaCO3']
            self.elements = ['Solid_CaCO3', 'Ca', 'C', 'O', 'H']
            self.fc_mask = np.array([False, True, True, True, True], dtype=bool)
            Mw = {'Solid_CaCO3': 100.0869, 'Ca': 40.078, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol
            self.n_points = list(self.n_obl_mult * np.array([101, 201, 101, 101, 101], dtype=np.intp))
            self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, 0.3]
            self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.01, 0.02, 0.37]
            # Rate annihilation matrix
            self.E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                               [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                               [1, 0, 1, 2, 3, 3, 3, 0, 1, 3, 0],
                               [2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]])
            # Mineral decomposition into elements
            stoich_matrix = np.array([[-1, 1, 1, 3, 0]])
            # Mineral properties
            rock_props = {'Solid_CaCO3': {'density': 2710., 'compressibility': 1.e-6}}
            # Dimensions of initial, property interpolators
            n_init_ops = 1
            n_prop_ops = 19
        elif set(self.minerals) == {'calcite', 'dolomite'}:
            # purely for initialization
            self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2',
                               'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid_CaCO3',                  # calcite-related
                               'CaMg(CO3)2', 'Mg+2', 'MgOH+', 'MgHCO3+', 'Solid_CaMg(CO3)2']        # dolomite-related
            self.elements = ['Solid_CaCO3', 'Solid_CaMg(CO3)2', 'Ca', 'Mg', 'C', 'O', 'H']
            self.fc_mask = np.array([False, False, True, True, True, True, True], dtype=bool)
            Mw = {'Solid_CaCO3': 100.0869, 'Solid_CaMg(CO3)2': 184.401,
                    'Ca': 40.078, 'Mg': 24.305, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol
            self.n_points = list(self.n_obl_mult * np.array([101, 201, 201, 101, 101, 101, 101], dtype=np.intp))
            if self.co2_injection < self.co2_injection_cutoff:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.3]
                self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.2, 0.01, 0.001, 0.02, 0.37]
            else:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.25]
                self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.2, 0.01, 0.001, 0.07, 0.37]
            # Rate annihilation matrix
            self.E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],    # Solid_CaCO3
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],    # Solid_CaMg(CO3)2
                               [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],    # Ca
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],    # Mg
                               [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 2, 0, 0, 1, 0],    # C
                               [1, 0, 1, 2, 3, 3, 3, 0, 1, 3, 0, 6, 0, 1, 3, 0],    # O
                               [2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0]])   # H
            # Mineral decomposition into elements
            stoich_matrix = np.array([[-1, 0, 1, 0, 1, 3, 0],
                                      [0, -1, 1, 1, 2, 6, 0]])
            # Mineral properties
            rock_props = {'Solid_CaCO3': {'density': 2710., 'compressibility': 1.e-6},
                          'Solid_CaMg(CO3)2': {'density': 2840., 'compressibility': 1.e-6}}
            # Dimensions of initial, property interpolators
            n_init_ops = 10
            n_prop_ops = 25
        elif set(self.minerals) == {'calcite', 'dolomite', 'magnesite'}:
            # purely for initialization
            self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2',
                               'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid_CaCO3',                  # calcite-related
                               'CaMg(CO3)2', 'Mg+2', 'MgOH+', 'MgHCO3+', 'Solid_CaMg(CO3)2', 'Solid_MgCO3'] # dolomite-related
            self.elements = ['Solid_CaCO3', 'Solid_CaMg(CO3)2', 'Solid_MgCO3', 'Ca', 'Mg', 'C', 'O', 'H']
            self.fc_mask = np.array([False, False, False, True, True, True, True, True], dtype=bool)
            Mw = {'Solid_CaCO3': 100.0869, 'Solid_CaMg(CO3)2': 184.401, 'Solid_MgCO3': 84.31,
                    'Ca': 40.078, 'Mg': 24.305, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol
            self.n_points = list(self.n_obl_mult * np.array([101, 201, 201, 201, 101, 101, 101, 101], dtype=np.intp))
            self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.3]
            self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.2, 0.01, 0.01, 0.001, 0.02, 0.37]
            # Rate annihilation matrix
            self.E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],    # Solid_CaCO3
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],    # Solid_CaMg(CO3)2
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],    # Solid_MgCO3
                               [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],    # Ca
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],    # Mg
                               [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 2, 0, 0, 1, 0, 0],    # C
                               [1, 0, 1, 2, 3, 3, 3, 0, 1, 3, 0, 6, 0, 1, 3, 0, 0],    # O
                               [2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]])   # H
            # Mineral decomposition into elements
            stoich_matrix = np.array([[-1, 0, 0, 1, 0, 1, 3, 0],
                                      [0, -1, 0, 1, 1, 2, 6, 0],
                                      [0, 0, -1, 0, 1, 1, 3, 0]])
            # Mineral properties
            rock_props = {'Solid_CaCO3': {'density': 2710., 'compressibility': 1.e-6},
                          'Solid_CaMg(CO3)2': {'density': 2840., 'compressibility': 1.e-6},
                          'Solid_MgCO3': {'density': 2958., 'compressibility': 1.e-6}}
            # Dimensions of initial, property interpolators
            n_init_ops = 10
            n_prop_ops = 28

        self.nc = len(self.elements)

        # Create property containers:
        is_gas_spec = False if self.co2_injection < self.co2_injection_cutoff else True
        property_container = ModelProperties(phases_name=self.phases, components_name=self.elements, Mw=Mw,
                                             kinetic_mechanisms=self.kinetic_mechanisms, min_z=self.obl_min,
                                             temperature=self.temperature, fc_mask=self.fc_mask, is_gas_spec=is_gas_spec)

        property_container.diffusion_ev = {ph: ConstFunc(np.concatenate([np.zeros(self.n_solid), \
                                         np.ones(self.nc - self.n_solid)]) * 5.2e-10 * 86400) for ph in self.phases}

        for min, props in rock_props.items():
            property_container.rock_compr_ev[min] = ConstFunc(props['compressibility'])
            property_container.rock_density_ev[min] = DensityBasic(compr=props['compressibility'], dens0=props['density'], p0=1.)

        # self.kin_fact = self.property.rock_density_ev['solid'].evaluate(pressure) / self.property.Mw['Solid'] * np.mean(self.solid_sat)
        self.kin_fact = 1

        # Create instance of data-structure for simulation (and chemical) input parameters:
        input_data_struct = MyOwnDataStruct(nc=self.nc, zmin=self.obl_min, temp=self.temperature,
                                            stoich_matrix=stoich_matrix, pressure_init=self.pressure_init,
                                            kin_fact=self.kin_fact, n_init_ops=n_init_ops, n_prop_ops=n_prop_ops)

        # Create instance of (own) physics class:
        self.physics = PhreeqcDissolution(timer=self.timer, elements=self.elements, n_points=self.n_points, 
                                          axes_min=self.axes_min, axes_max=self.axes_max,
                                          input_data_struct=input_data_struct, properties=property_container, cache=False)

        self.physics.add_property_region(property_container, 0)

        # Compute injection stream
        mole_water, mole_co2 = calculate_injection_stream(1.1, self.co2_injection, self.temperature, self.pressure_init) # input - m3 of water, co2
        mole_fraction_water, mole_fraction_co2 = get_mole_fractions(mole_water, mole_co2)

        # Define injection stream composition,
        self.inj_stream_components = np.zeros(len(self.components))
        self.inj_stream_components[self.components.index('H2O')] = mole_fraction_water     # H2O
        self.inj_stream_components[self.components.index('CO2')] = mole_fraction_co2       # CO2
        self.inj_stream = convert_composition(self.inj_stream_components, self.E)
        self.inj_stream = correct_composition(self.inj_stream, self.min_z)

        # prepare arrays for evaluation of properties
        n_prop_ops = self.physics.input_data_struct.n_prop_ops
        n_vars = self.physics.n_vars
        self.prop_states = value_vector([0.] * self.n_res_blocks * (n_vars + 1))
        self.prop_states_np = np.asarray(self.prop_states)
        self.prop_values = value_vector([0.] * n_prop_ops * self.n_res_blocks)
        self.prop_values_np = np.asarray(self.prop_values)
        self.prop_dvalues = value_vector([0.] * n_prop_ops * self.n_res_blocks * n_vars)

    def set_reservoir(self, domain, nx, mesh_filename, poro_filename):
        self.domain = domain

        if self.domain == '1D':
            # grid
            self.domain_sizes = np.array([0.1, 0.001 * 7, 0.058905 / 7])
            self.domain_cells = np.array([nx, 1, 1])
            self.n_res_blocks = np.prod(self.domain_cells)
            self.cell_sizes = self.domain_sizes / self.domain_cells

            # properties
            depth = 1                      # m
            self.poro = 1                            # [-]
            self.params.trans_mult_exp = 4
            perm = 1.25e4 * self.poro ** self.params.trans_mult_exp
            self.solid_sat = np.zeros((self.n_res_blocks, self.n_solid))
            if set(self.minerals) == {'calcite'}:
                self.solid_sat[:, 0] = 0.7
            elif set(self.minerals) == {'calcite', 'dolomite'}:
                self.solid_sat[:, 0] = 0.6
                self.solid_sat[:, 1] = 0.1
            elif set(self.minerals) == {'calcite', 'dolomite', 'magnesite'}:
                self.solid_sat[:, 0] = 0.6
                self.solid_sat[:, 1] = 0.05
                self.solid_sat[:, 2] = 0.05
            self.inj_cells = np.array([0])

            self.volume = np.prod(self.domain_sizes)
            self.reservoir = StructReservoir(self.timer,
                                             nx=self.domain_cells[0], ny=self.domain_cells[1], nz=self.domain_cells[2],
                                             dx=self.cell_sizes[0], dy=self.cell_sizes[1], dz=self.cell_sizes[2],
                                             permx=perm, permy=perm, permz=perm, poro=self.poro, depth=depth)
        elif self.domain == '2D':
            # grid
            self.domain_sizes = np.array([0.09, 0.09, 0.006])
            self.domain_cells = np.array([nx, nx, 1])
            self.n_res_blocks = np.prod(self.domain_cells)
            self.cell_sizes = self.domain_sizes / self.domain_cells

            # properties
            depth = 1                      # m
            self.poro = 1                       # [-]
            self.params.trans_mult_exp = 4
            perm = 1.25e4 * self.poro ** self.params.trans_mult_exp

            # porosity
            if poro_filename == None:
                poro = 0.3 + np.random.uniform(-0.1, 0.1, self.n_res_blocks)
            else:
                poro = 0.3 + 0.05 * np.loadtxt(poro_filename).flatten()
                assert np.prod(self.domain_cells) == poro.size
            poro[poro < 1.e-4] = 1.e-4
            poro[poro > 1 - 1.e-4] = 1 - 1.e-4
            self.solid_sat = np.zeros((self.n_res_blocks, self.n_solid))
            self.solid_sat[:, 0] = 1 - poro
            self.inj_cells = self.domain_cells[0] * np.arange(self.domain_cells[1])

            self.volume = np.prod(self.domain_sizes)
            self.reservoir = StructReservoir(self.timer,
                                             nx=self.domain_cells[0], ny=self.domain_cells[1], nz=self.domain_cells[2],
                                             dx=self.cell_sizes[0], dy=self.cell_sizes[1], dz=self.cell_sizes[2],
                                             permx=perm, permy=perm, permz=perm, poro=self.poro, depth=depth)
        elif self.domain == '3D':
            depth = 1
            poro = 1
            self.params.trans_mult_exp = 4
            perm = 1.25e4 * poro ** self.params.trans_mult_exp
            mesh_file = mesh_filename
            self.reservoir = UnstructReservoir(timer=self.timer, permx=perm, permy=perm, permz=perm, frac_aper=0,
                                               mesh_file=mesh_file, poro=poro)
            self.reservoir.physical_tags['matrix'] = [99991]
            self.reservoir.physical_tags['boundary'] = [991, 992, 993]
            self.reservoir.init_reservoir()
            self.volume = np.asarray(self.reservoir.mesh.volume).sum()
            self.n_res_blocks = self.reservoir.mesh.n_blocks
            if poro_filename == None:
                poro = 0.3 + np.random.uniform(-0.1, 0.1, self.n_res_blocks)
            else:
                poro = 0.3 + 0.05 * np.loadtxt(poro_filename).flatten()
                assert self.n_res_blocks == poro.size
            self.solid_sat = np.zeros((self.n_res_blocks, self.n_solid))
            self.solid_sat[:, 0] = 1 - poro

            # identifying injection/production cells
            a = 2 / 3 * np.cbrt(self.volume / self.reservoir.mesh.n_blocks / 0.1)
            h = self.reservoir.discretizer.mesh_data.points[:,2].max()
            # initial guesses
            self.prd_cells = np.where(self.reservoir.discretizer.centroid_all_cells[:, 2] < a)[0]
            self.inj_cells = np.where(self.reservoir.discretizer.centroid_all_cells[:, 2] > h - a)[0]
            # exact filtering
            self.prd_cells = [id for id in self.prd_cells if np.count_nonzero(self.reservoir.discretizer.mat_cell_info_dict[id].coord_nodes_to_cell[:, 2] < 1e-4 * a) > 2]
            self.inj_cells = [id for id in self.inj_cells if np.count_nonzero(self.reservoir.discretizer.mat_cell_info_dict[id].coord_nodes_to_cell[:, 2] > h - 1e-4 * a) > 2]
            self.prd_cells = np.array(self.prd_cells, dtype=np.intp)
            self.inj_cells = np.array(self.inj_cells, dtype=np.intp)
        else:
            print(f'domain={self.domain} is not supported')
            exit(-1)

    def set_initial_conditions(self):
        # ====================================== Initialize reservoir composition ======================================
        print('\nInitializing compositions...')

        # Component-defined composition of a non-solid phase (pure water here)
        self.initial_comp_components = np.zeros(len(self.components))
        self.initial_comp_components[0] = 1.0
        self.solid_frac = np.zeros((self.n_res_blocks, self.n_solid))
        self.initial_comp = np.zeros((self.n_res_blocks + 2, self.nc - 1))

        # Interpolated values of non-solid volume (second value always 0 due to no (5,1) interpolator)
        values = value_vector([0] * self.physics.input_data_struct.n_init_ops)
        values_np = np.asarray(values)

        # Iterate over solid saturation and call interpolator
        for i in range(len(self.solid_sat)):
            # There are 5 values in the state
            composition_full = convert_composition(self.initial_comp_components, self.E)
            composition = correct_composition(composition_full, self.min_z)
            init_state = value_vector(np.hstack((self.physics.input_data_struct.pressure_init, self.solid_sat[i],
                                                 composition[self.n_solid:])))

            # Call interpolator
            self.physics.comp_itor[0].evaluate(init_state, values)

            # Assemble initial composition
            self.solid_frac[i] = values_np[:self.n_solid]
            initial_comp_with_solid = composition_full # np.multiply(composition_full, 1 - self.solid_frac[i])
            initial_comp_with_solid[:self.n_solid] = self.solid_frac[i]
            self.initial_comp[i, :] = initial_comp_with_solid[:-1] # correct_composition(initial_comp_with_solid, self.min_z)

        # Define initial composition for wells
        # for i in range(n_matrix, n_matrix + 2):
        #     self.initial_comp[i, :] = np.array(self.inj_stream)

        for i in range(self.n_res_blocks, self.n_res_blocks + 2):
            self.initial_comp[i, :] = np.array(self.initial_comp[0, :])

        # print('\tNegative composition occurrence while initializing:', self.physics.comp_itor[0].counter, '\n')

        nb = self.reservoir.mesh.n_res_blocks
        input_distribution = {
            'pressure': self.pressure_init,
            **{
                var: self.initial_comp[:nb, j]
                for j, var in enumerate(self.physics.vars[1:])
            }
        }

        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_wells(self):
        d_w = 1.5
        r_w = d_w / 2
        well_index = 5

        # self.reservoir.add_well("I1", wellbore_diameter=d_w)
        # for idx in range(self.domain_cells[1]):
        #     self.reservoir.add_perforation(well_name='I1', cell_index=(1, idx + 1, 1), multi_segment=False,
        #                                    verbose=True, well_radius=r_w, well_index=well_index,
        #                                    well_indexD=well_index)

        self.reservoir.add_well("P1", wellbore_diameter=d_w)
        if self.domain == '3D':
            for idx in self.prd_cells:
                self.reservoir.add_perforation(well_name='P1', cell_index=idx, multi_segment=False,
                                               verbose=True, well_radius=r_w, well_index=well_index,
                                               well_indexD=well_index)
        else:
            for idx in range(self.domain_cells[1]):
                self.reservoir.add_perforation(well_name='P1', cell_index=(self.domain_cells[0], idx + 1, 1), multi_segment=False,
                                               verbose=True, well_radius=r_w, well_index=well_index,
                                               well_indexD=well_index)

    def set_rhs_flux(self, t: float = None):
        nv = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        rhs_flux = np.zeros(nb * nv)

        rho_m_h20 = 1000 / 18.015 # kmol/m3
        for cell_id in self.inj_cells:
            for i in range(nv - 1):
                rhs_flux[cell_id * nv + i] = -self.inj_rate * rho_m_h20 * self.inj_stream[i]
            rhs_flux[cell_id * nv + nv - 1] = -self.inj_rate * rho_m_h20 * (1 - np.sum(self.inj_stream))

        return rhs_flux

    def set_boundary_conditions(self):
        # New boundary condition by adding wells:
        w = self.reservoir.wells[0]
        self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP, is_inj=False,
                                       target=self.pressure_init)

    def output_properties(self, output_properties: list = None, timestep: int = None) -> tuple:
        timesteps = [timestep] if timestep is not None else [0]
        if output_properties is None:
            prop_names = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        else:
            prop_names = output_properties

        X = np.asarray(self.physics.engine.X)
        nb = self.n_res_blocks
        nv = self.physics.n_vars
        nops = len(prop_names)
        n_interp_size = self.physics.input_data_struct.n_prop_ops

        # unknowns
        property_array = {var: np.array([X[i:nb * nv:nv]]) for i, var in enumerate(self.physics.vars)}
        # properties
        self.physics.property_itor[0].evaluate_with_derivatives(self.physics.engine.X, self.physics.engine.region_cell_idx[0],
                                                                self.prop_values, self.prop_dvalues)

        for i, prop in enumerate(prop_names):
            property_array[prop] = np.array([self.prop_values_np[i::n_interp_size]])

        # porosity
        n_vars = self.physics.nc
        op_vals = np.asarray(self.physics.engine.op_vals_arr).reshape(self.reservoir.mesh.n_blocks, self.physics.n_ops)
        poro = op_vals[:self.reservoir.mesh.n_res_blocks, self.physics.reservoir_operators[0].PORO_OP]
        property_array['porosity'] = poro[np.newaxis]

        # hydrogen
        property = self.physics.property_operators[next(iter(self.physics.property_operators))].property
        fc = property.components_name[property.fc_mask]
        property_array[fc[-1]] = 1 - sum(property_array[c] for c in fc[:-1])

        # write to *.h5
        if self.domain == '1D':
            path = os.path.join(self.output_folder, self.sol_filename)
            with h5py.File(path, "a") as f:
                current_index = f["dynamic/time"].shape[0] - 1
                written_vars = list(f["dynamic/variable_names"].asstr())
                new_keys = [prop for prop in property_array.keys() if prop not in written_vars]
                new_vars_num = len(new_keys)

                if "properties" not in f["dynamic"]:
                    f["dynamic"].create_dataset("properties", shape=(0, nb, new_vars_num),
                                                maxshape=(None, nb, new_vars_num), dtype=np.float64)

                extra_dataset = f["dynamic/properties"]
                if extra_dataset.shape[0] <= current_index:
                    extra_dataset.resize((current_index + 1, nb, new_vars_num))

                for i, key in enumerate(new_keys):
                    extra_dataset[current_index, :, i] = property_array[key]

                if "properties_name" not in f["dynamic"]:
                    datatype = h5py.special_dtype(vlen=str)  # dtype for variable-length strings
                    var_names = f["dynamic"].create_dataset('properties_name', (new_vars_num,), dtype=datatype)
                    var_names[:] = new_keys

        return timesteps, property_array

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, kinetic_mechanisms, nc_sol=0, np_sol=0, 
                 min_z=1e-11, rate_ann_mat=None, temperature=None, fc_mask=None, is_gas_spec=False):
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, nc_sol=nc_sol, np_sol=np_sol,
                         min_z=min_z, rate_ann_mat=rate_ann_mat, temperature=temperature)
        self.components_name = np.array(self.components_name)

        # Define primary fluid constituents
        if fc_mask is None:
            self.fc_mask = self.nc * [True]
        else:
            self.fc_mask = fc_mask
        self.fc_idx = {comp: i for i, comp in enumerate(self.components_name[self.fc_mask])}
        self.Mw_array = np.array([self.Mw[c] for c in self.components_name])
        self.n_solid = (self.fc_mask == False).sum()

        # to retrieve fluid component fractions from state
        self.f_mask_state = np.concatenate([[False], self.fc_mask[:-1]])
        # to retrieve solid component fractions from state
        self.s_mask_state = np.concatenate([[False], ~self.fc_mask[:-1]])

        # figure out spec
        self.minerals = self.components_name[~self.fc_mask]

        self.sat_overall = np.zeros(self.nph + 1)
        self.diffusivity = np.zeros((self.nph, self.nc))
        self.sat_minerals = np.zeros(self.n_solid)
        self.kin_rates = np.zeros(self.n_solid)
        self.rock_compr = np.zeros(self.n_solid)

        # Define custom evaluators
        self.rock_density_ev = {}
        self.rock_compr_ev = {}
        self.flash_ev = self.Flash(min_z=self.min_z, fc_mask=self.fc_mask, fc_idx=self.fc_idx,
                                   f_mask_state=self.f_mask_state, temperature=self.temperature,
                                   minerals=self.minerals, is_gas_spec=is_gas_spec)

        self.kinetic_rate_ev = {m: self.CustomKineticRate(self.temperature, self.min_z, m.split('_', 1)[1], kinetic_mechanisms) for m in self.minerals}
        self.rel_perm_ev = {ph: self.CustomRelPerm(2) for ph in phases_name[:2]}  # Relative perm for first two phases
        self.viscosity_ev = { phases_name[0]: self.GasViscosity(), phases_name[1]: self.LiquidViscosity() }

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector

        :return: updated value for operators, stored in values
        """
        nu_v, x, y, rho_phases, self.kin_state, _, _ = self.flash_ev.evaluate(state)
        self.nu_solid = state[self.s_mask_state]
        self.nu[0] = nu_v * (1 - self.nu_solid.sum()) # convert to overall molar fraction
        self.nu[1] = 1 - nu_v - self.nu_solid.sum()

        pressure = state[0]
        # molar densities in kmol/m3
        self.dens_m[1], self.dens_m[0] = rho_phases['aq'], rho_phases['gas']
        self.dens_m_solid = np.array([v.evaluate(pressure) / self.Mw[k] for k, v in self.rock_density_ev.items()])
        self.ph = np.array([0, 1], dtype=np.intp)

        # Get saturations
        if nu_v > 0:
            sum = self.nu[0] / self.dens_m[0] + self.nu[1] / self.dens_m[1] + (self.nu_solid / self.dens_m_solid).sum()
            self.sat_overall[0] = self.nu[0] / self.dens_m[0] / sum
            self.sat_overall[1] = self.nu[1] / self.dens_m[1] / sum
            self.sat_overall[2] = (self.nu_solid / self.dens_m_solid).sum() / sum
        else:
            sum = self.nu[1] / self.dens_m[1] + (self.nu_solid / self.dens_m_solid).sum()
            self.sat_overall[0] = 0
            self.sat_overall[1] = self.nu[1] / self.dens_m[1] / sum
            self.sat_overall[2] = (self.nu_solid / self.dens_m_solid).sum() / sum
        self.sat_minerals = self.nu_solid / self.dens_m_solid / sum

        self.x = np.array([y, x])

        for j in self.ph:
            M = np.sum(self.Mw_array * self.x[j])
            self.dens[j] = self.dens_m[j] * M
            self.sat[j] = self.sat_overall[j] / np.sum(self.sat_overall[:self.nph])
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.diffusivity[j] = self.diffusion_ev[self.phases_name[j]].evaluate()

        # gas
        self.mu[0] = self.viscosity_ev[self.phases_name[0]].evaluate(pressure=pressure, temperature=self.temperature)
        # liquid
        self.mu[1] = self.viscosity_ev[self.phases_name[1]].evaluate(density=self.dens[1], temperature=self.temperature)

        for i, k in enumerate(self.rock_compr_ev.keys()):
            self.rock_compr[i] = self.rock_compr_ev[k].evaluate(pressure)
            self.kin_rates[i] = self.kinetic_rate_ev[k].evaluate(self.kin_state, self.sat_minerals[i], self.dens_m_solid[i])

    # default flash working with molar fractions
    class Flash:
        def __init__(self, min_z, fc_mask, fc_idx, f_mask_state, minerals, temperature=None, 
                    is_gas_spec=False):
            """
            :param min_z: minimal composition value
            :param fc_mask: boolean mask for extraction of fluid components from all components
            :param fc_idx: dictionary for mapping names of fluid components to filtered (via mask) state
            :param temperature: temperature for isothermal case
            """
            self.fc_mask = fc_mask
            self.fc_idx = fc_idx
            self.f_mask_state = f_mask_state
            self.n_fluid = (self.fc_mask == True).sum()
            self.n_solid = (self.fc_mask == False).sum()
            self.minerals = minerals
            self.mineral_names = [item.split('_', 1)[1] for item in self.minerals]

            if temperature is None:
                self.thermal = True
            else:
                self.thermal = False
                self.temperature = temperature - 273.15
            self.min_z = min_z
            self.total_moles = 1000
            self.molar_weight_h2o = 0.018016

            # phreeqc
            root = os.path.dirname(darts.__file__)
            if sys.platform.startswith("win"):
                libname = "IPhreeqc.dll"
            else:
                libname = "libIPhreeqc.so"
            self.phreeqc = IPhreeqc(os.path.join(root, libname))
            self.load_database(self.phreeqc, "phreeqc.dat")
            self.pitzer = IPhreeqc(os.path.join(root, libname))
            self.load_database(self.pitzer, "pitzer.dat")
            # self.phreeqc.phreeqc.OutputFileOn = True
            # self.phreeqc.phreeqc.SelectedOutputFileOn = True

            if set(self.minerals) == {'Solid_CaCO3'}: # pure calcite
                self.spec = 0
                self.phreeqc_species = ["OH-", "H+", "H2O", "CH4", "HCO3-", "CO2", "CO3-2", "CaHCO3+", "CaCO3", "(CO2)2", "Ca+2", "CaOH+", "H2", "O2"]
                self.species_2_element_moles = np.array([2, 1, 3, 5, 5, 3, 4, 6, 5, 6, 1, 3, 2, 2])
                species_headings = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
                species_punch = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
                if is_gas_spec:
                    self.phreeqc_template = f"""
                        USER_PUNCH            
                        -headings   Ca(mol)       C(mol)       O(mol)       H(mol)       Vol_aq   SR            ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
                        10 PUNCH    TOTMOLE("Ca") TOTMOLE("C") TOTMOLE("O") TOTMOLE("H") SOLN_VOL SR("Calcite") ACT("H+") ACT("CO2") ACT("H2O") {species_punch}
            
                        SELECTED_OUTPUT
                        -selected_out    true
                        -user_punch      true
                        -reset           false
                        -high_precision  true
                        -gases           CO2(g) H2O(g)
            
                        SOLUTION 1
                        temp      {{temperature:.2f}}
                        pressure  {{pressure:.4f}}
                        pH        7 charge
                        -water    {{water_mass:.10f}} # kg
            
                        REACTION 1
                        Ca        {{calcium:.10f}}
                        C         {{carbon:.10f}}
                        O         {{oxygen:.10f}}
                        H         {{hydrogen:.10f}}
                        1
            
                        KNOBS
                        -convergence_tolerance  1e-10
            
                        GAS_PHASE 1
                        pressure  {{pressure:.4f}}       
                        temp      {{temperature:.2f}}  
                        CO2(g)    {{co2_pressure:.4f}}
                        H2O(g)    {{h2o_pressure:.4f}}
            
                        END
                        """
                else:
                    self.phreeqc_template = f"""
                        USER_PUNCH            
                        -headings   Ca(mol)       C(mol)       O(mol)       H(mol)       Vol_aq   SR            ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
                        10 PUNCH    TOTMOLE("Ca") TOTMOLE("C") TOTMOLE("O") TOTMOLE("H") SOLN_VOL SR("Calcite") ACT("H+") ACT("CO2") ACT("H2O") {species_punch}
            
                        SELECTED_OUTPUT
                        -selected_out    true
                        -user_punch      true
                        -reset           false
                        -high_precision  true
                        -gases           CO2(g) H2O(g)
            
                        SOLUTION 1
                        temp      {{temperature:.2f}}
                        pressure  {{pressure:.4f}}
                        pH        7 charge
                        -water    {{water_mass:.10f}} # kg
            
                        REACTION 1
                        Ca        {{calcium:.10f}}
                        C         {{carbon:.10f}}
                        O         {{oxygen:.10f}}
                        H         {{hydrogen:.10f}}
                        1
            
                        KNOBS
                        -convergence_tolerance  1e-10
            
                        GAS_PHASE 1
                        pressure  {{pressure:.4f}}       
                        temp      {{temperature:.2f}}  
                        CO2(g)    0.0#{{co2_pressure:.4f}}
                        # H2O(g)    {{h2o_pressure:.4f}}
            
                        END
                        """
            elif set(self.minerals) == {'Solid_CaCO3', 'Solid_CaMg(CO3)2'}: # calcite and dolomite
                self.spec = 1
                self.phreeqc_species = ["OH-", "H+", "H2O", "CH4", "HCO3-", "CO2", "CO3-2", "CaHCO3+", "MgHCO3+",
                                        "CaCO3", "MgCO3", "(CO2)2", "Ca+2", "CaOH+", "H2", "Mg+2", "MgOH+", "O2"]
                self.species_2_element_moles = np.array([2, 1, 3, 5, 5, 3, 4, 6, 6,
                                                         5, 5, 6, 1, 3, 2, 1, 3, 2])
                species_headings = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
                species_punch = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
                if is_gas_spec:
                    self.phreeqc_template = f"""
                        USER_PUNCH            
                        -headings    Ca(mol)      Mg(mol)       C(mol)       O(mol)       H(mol)       Vol_aq   SR_Calcite    SR_Dolomite    ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
                        10 PUNCH    TOTMOLE("Ca") TOTMOLE("Mg") TOTMOLE("C") TOTMOLE("O") TOTMOLE("H") SOLN_VOL SR("Calcite") SR("Dolomite") ACT("H+") ACT("CO2") ACT("H2O") {species_punch}

                        SELECTED_OUTPUT
                        -selected_out    true
                        -user_punch      true
                        -reset           false
                        -high_precision  true
                        -gases           CO2(g) H2O(g)

                        SOLUTION 1
                        temp      {{temperature:.2f}}
                        pressure  {{pressure:.4f}}
                        pH        7 charge
                        -water    {{water_mass:.10f}} # kg

                        REACTION 1
                        Ca        {{calcium:.10f}}
                        Mg        {{magnesium:.10f}}
                        C         {{carbon:.10f}}
                        O         {{oxygen:.10f}}
                        H         {{hydrogen:.10f}}
                        1

                        KNOBS
                        -convergence_tolerance  1e-10

                        GAS_PHASE 1
                        pressure  {{pressure:.4f}}       
                        temp      {{temperature:.2f}}  
                        CO2(g)    {{co2_pressure:.4f}}
                        H2O(g)    {{h2o_pressure:.4f}}

                        END
                        """
                else:
                    self.phreeqc_template = f"""
                        USER_PUNCH            
                        -headings    Ca(mol)      Mg(mol)       C(mol)       O(mol)       H(mol)       Vol_aq   SR_Calcite    SR_Dolomite    ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
                        10 PUNCH    TOTMOLE("Ca") TOTMOLE("Mg") TOTMOLE("C") TOTMOLE("O") TOTMOLE("H") SOLN_VOL SR("Calcite") SR("Dolomite") ACT("H+") ACT("CO2") ACT("H2O") {species_punch}

                        SELECTED_OUTPUT
                        -selected_out    true
                        -user_punch      true
                        -reset           false
                        -high_precision  true
                        -gases           CO2(g) H2O(g)

                        SOLUTION 1
                        temp      {{temperature:.2f}}
                        pressure  {{pressure:.4f}}
                        pH        7 charge
                        -water    {{water_mass:.10f}} # kg

                        REACTION 1
                        Ca        {{calcium:.10f}}
                        Mg        {{magnesium:.10f}}
                        C         {{carbon:.10f}}
                        O         {{oxygen:.10f}}
                        H         {{hydrogen:.10f}}
                        1

                        KNOBS
                        -convergence_tolerance  1e-10

                        GAS_PHASE 1
                        pressure  {{pressure:.4f}}       
                        temp      {{temperature:.2f}}  
                        CO2(g)    0#{{co2_pressure:.4f}}
                        #H2O(g)    {{h2o_pressure:.4f}}

                        END
                        """
            elif set(self.minerals) == {'Solid_CaCO3', 'Solid_CaMg(CO3)2', 'Solid_MgCO3'}: # calcite, dolomite and magnesite
                self.spec = 2
                self.phreeqc_species = ["OH-", "H+", "H2O", "CH4", "HCO3-", "CO2", "CO3-2", "CaHCO3+", "MgHCO3+",
                                        "CaCO3", "MgCO3", "(CO2)2", "Ca+2", "CaOH+", "H2", "Mg+2", "MgOH+", "O2"]
                self.species_2_element_moles = np.array([2, 1, 3, 5, 5, 3, 4, 6, 6,
                                                         5, 5, 6, 1, 3, 2, 1, 3, 2])
                species_headings = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
                species_punch = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
                self.phreeqc_template = f"""
                    USER_PUNCH            
                    -headings    Ca(mol)      Mg(mol)       C(mol)       O(mol)       H(mol)       Vol_aq   SR_Calcite    SR_Dolomite    SR_Magnesite    ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
                    10 PUNCH    TOTMOLE("Ca") TOTMOLE("Mg") TOTMOLE("C") TOTMOLE("O") TOTMOLE("H") SOLN_VOL SR("Calcite") SR("Dolomite") SR("Magnesite") ACT("H+") ACT("CO2") ACT("H2O") {species_punch}

                    SELECTED_OUTPUT
                    -selected_out    true
                    -user_punch      true
                    -reset           false
                    -high_precision  true
                    -gases           CO2(g) H2O(g)

                    SOLUTION 1
                    temp      {{temperature:.2f}}
                    pressure  {{pressure:.4f}}
                    pH        7 charge
                    -water    {{water_mass:.10f}} # kg

                    REACTION 1
                    Ca        {{calcium:.10f}}
                    Mg        {{magnesium:.10f}}
                    C         {{carbon:.10f}}
                    O         {{oxygen:.10f}}
                    H         {{hydrogen:.10f}}
                    1

                    KNOBS
                    -convergence_tolerance  1e-10

                    GAS_PHASE 1
                    pressure  {{pressure:.4f}}       
                    temp      {{temperature:.2f}}  
                    CO2(g)     0

                    END
                    """

        def load_database(self, database, db_path):
            try:
                database.load_database(db_path)
            except Exception as e:
                warnings.warn(f"Failed to load '{db_path}': {e}.", Warning)

        def interpret_results(self, database):
            results_array = np.array(database.get_selected_output_array()[2])

            volume_gas = results_array[2] / 1000  # liters to m3
            co2_gas_mole = results_array[3]
            h2o_gas_mole = results_array[4]
            total_mole_gas = 3 * (co2_gas_mole + h2o_gas_mole)

            # interpret aqueous phase
            mole_aq = results_array[5:5 + self.n_fluid]
            # hydrogen_mole_aq = results_array[5]
            # oxygen_mole_aq = results_array[6]
            # carbon_mole_aq = results_array[7]
            # calcium_mole_aq = results_array[8]

            volume_aq = results_array[5 + self.n_fluid] / 1000  # liters to m3
            total_mole_aq = mole_aq.sum()  # mol
            rho_aq = total_mole_aq / volume_aq / 1000  # kmol/m3

            # molar fraction of elements in aqueous phase
            nc = self.n_solid + self.n_fluid
            x = np.zeros(nc)
            x[self.n_solid:] = mole_aq / total_mole_aq

            # in gaseous phase
            y = np.zeros(nc)
            if total_mole_gas > 1.e-8:
                rho_g = total_mole_gas / volume_gas / 1000  # kmol/m3
                y[-3] = co2_gas_mole / total_mole_gas
                y[-2] = (2 * co2_gas_mole + h2o_gas_mole) / total_mole_gas
                y[-1] = 2 * h2o_gas_mole / total_mole_gas
            else:
                rho_g = 0.0

            # molar densities
            rho_phases = {'aq': rho_aq, 'gas': rho_g}
            # molar fraction of gaseous phase in fluid
            nu_v = total_mole_gas / (total_mole_aq + total_mole_gas)

            # interpret kinetic parameters
            counter = self.n_fluid + 5 + 1
            kin_state = {}
            for i, name in enumerate(self.mineral_names):
                kin_state['SR_' + name] = results_array[counter]
                counter += 1
            kin_state['Act(H+)'] = results_array[counter]
            kin_state['Act(CO2)'] = results_array[counter + 1]
            kin_state['Act(H2O)'] = results_array[counter + 2]
            species_molalities = results_array[counter + 3:]

            return nu_v, x, y, rho_phases, kin_state, volume_aq + volume_gas, species_molalities

        def get_fluid_composition(self, state):
            if self.thermal:
                z = state[-1][self.f_mask_state]
            else:
                z = state[self.f_mask_state]
            z_last = min(max(1 - np.sum(z), self.min_z), 1 - self.min_z)
            z = np.concatenate([z, [z_last]])
            return z

        def evaluate(self, state):
            """
            :param state: state vector with fluid composition accessible by fc_mask
            :type state: np.ndarray
            :return: phase molar fraction, molar composition of aqueous and vapour phases, kinetic params, solution volume
            """
            # extract pressure and fluid composition
            pressure_atm = bar2atm(state[0])

            # check for negative composition occurrence
            fluid_composition = self.get_fluid_composition(state)

            # calculate amount of moles of each component in 1000 moles of mixture
            fluid_moles = self.total_moles * fluid_composition

            # adjust oxygen and hydrogen moles for water formation
            init_h_moles, init_o_moles = fluid_moles[self.fc_idx['H']], fluid_moles[self.fc_idx['O']]
            if init_h_moles / 2 <= init_o_moles:
                water_mass = init_h_moles / 2 * self.molar_weight_h2o
                fluid_moles[self.fc_idx['H']] = 0
                fluid_moles[self.fc_idx['O']] = init_o_moles - init_h_moles / 2
            else:
                water_mass = init_o_moles * self.molar_weight_h2o
                fluid_moles[self.fc_idx['H']] = init_h_moles - 2 * init_o_moles
                fluid_moles[self.fc_idx['O']] = 0

            # Check if solvent (water) is enough
            ion_strength = np.sum(fluid_moles) / (water_mass + 1.e-8)
            if ion_strength > 20:
                print(f'ion_strength = {ion_strength}')
            # assert ion_strength < 7, "Not enough water to form a realistic brine"

            # Generate and execute PHREEQC input
            if self.spec == 0:
                input_string = self.phreeqc_template.format(
                    temperature=self.temperature,
                    pressure=pressure_atm,
                    co2_pressure=0.99 * pressure_atm,
                    h2o_pressure=0.01 * pressure_atm,
                    water_mass=water_mass,
                    hydrogen=fluid_moles[self.fc_idx['H']],
                    oxygen=fluid_moles[self.fc_idx['O']],
                    carbon=fluid_moles[self.fc_idx['C']],
                    calcium=fluid_moles[self.fc_idx['Ca']]
                )
            elif self.spec == 1 or self.spec == 2:
                input_string = self.phreeqc_template.format(
                    temperature=self.temperature,
                    pressure=pressure_atm,
                    co2_pressure=0.99 * pressure_atm,
                    h2o_pressure=0.01 * pressure_atm,
                    water_mass=water_mass,
                    hydrogen=fluid_moles[self.fc_idx['H']],
                    oxygen=fluid_moles[self.fc_idx['O']],
                    carbon=fluid_moles[self.fc_idx['C']],
                    calcium=fluid_moles[self.fc_idx['Ca']],
                    magnesium=fluid_moles[self.fc_idx['Mg']]
                )

            try:
                self.phreeqc.run_string(input_string)
                nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molalities = self.interpret_results(self.phreeqc)
            except Exception as e:
                warnings.warn(f"Failed to run PHREEQC: {e}", Warning)
                if self.spec == 0:
                    print(f"h20_mass={water_mass}, p={state[0]}, Ca={fluid_moles[self.fc_idx['Ca']]}, C={fluid_moles[self.fc_idx['C']]}, O={fluid_moles[self.fc_idx['O']]}, H={fluid_moles[self.fc_idx['H']]}")
                elif self.spec == 1 or self.spec == 2:
                    print(f"h20_mass={water_mass}, p={state[0]}, Ca={fluid_moles[self.fc_idx['Ca']]}, Mg={fluid_moles[self.fc_idx['Mg']]}, C={fluid_moles[self.fc_idx['C']]}, O={fluid_moles[self.fc_idx['O']]}, H={fluid_moles[self.fc_idx['H']]}")
                self.pitzer.run_string(input_string)
                nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molalities = self.interpret_results(self.pitzer)

            species_molar_fractions = species_molalities * water_mass * self.species_2_element_moles / self.total_moles
            return nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molar_fractions

    class CustomKineticRate:
        def __init__(self, temperature, min_z, mineral, kinetic_mechanisms):
            self.temperature = temperature
            self.min_z = min_z
            self.mineral = mineral
            self.specific_sa = 0.925                 # [m2/mol], default = 0.925
            self.SR_name = 'SR_' + self.mineral

            # doi: 10.3133/ofr20041068 for 25 celsius
            t_ref = 273.15 + 25
            if mineral == 'CaCO3':                   # calcite
                acidic = self.ReactionMechanism(name='acidic',
                                           temperature_ref=t_ref,
                                           k=10 ** (-0.3),
                                           Ea=14400,
                                           n=1)
                neutral = self.ReactionMechanism(name='neutral',
                                           temperature_ref=t_ref,
                                           k=10 ** (-5.81),
                                           Ea=23500,
                                           n=0)
                carbonate = self.ReactionMechanism(name='carbonate',
                                           temperature_ref=t_ref,
                                           k=10 ** (-3.48),
                                           Ea=35400,
                                           n=1)
            elif mineral == 'CaMg(CO3)2':            # dolomite
                acidic = self.ReactionMechanism(name='acidic',
                                           temperature_ref=t_ref,
                                           k=10 ** (-3.19),
                                           Ea=36100,
                                           n=0.5)
                neutral = self.ReactionMechanism(name='neutral',
                                           temperature_ref=t_ref,
                                           k=10 ** (-7.53),
                                           Ea=52200,
                                           n=0)
                carbonate = self.ReactionMechanism(name='carbonate',
                                           temperature_ref=t_ref,
                                           k=10 ** (-5.11),
                                           Ea=34800,
                                           n=0.5)
            elif mineral == 'MgCO3':                 # magnesite
                acidic = self.ReactionMechanism(name='acidic',
                                           temperature_ref=t_ref,
                                           k=10 ** (-6.38),
                                           Ea=14400,
                                           n=1)
                neutral = self.ReactionMechanism(name='neutral',
                                           temperature_ref=t_ref,
                                           k=10 ** (-9.34),
                                           Ea=23500,
                                           n=0)
                carbonate = self.ReactionMechanism(name='carbonate',
                                           temperature_ref=t_ref,
                                           k=10 ** (-5.22),
                                           Ea=62800,
                                           n=1)
            self.mechanisms = []
            if 'acidic' in kinetic_mechanisms:
                self.mechanisms.append(acidic)
            if 'neutral' in kinetic_mechanisms:
                self.mechanisms.append(neutral)
            if 'carbonate' in kinetic_mechanisms:
                self.mechanisms.append(carbonate)

        def evaluate(self, kin_state, solid_saturation, rho_s):
            activities = [kin_state['Act(H+)'], 1.0, kin_state['Act(CO2)']]
            rates = [m.evaluate(temperature=self.temperature, activity=activities[i], SR=kin_state[self.SR_name]) \
                     for i, m in enumerate(self.mechanisms)]

            # [mol/s/m3]
            kinetic_rate = -self.specific_sa * solid_saturation * (rho_s * 1000) * sum(rates)

            # [kmol/d/m3]
            kinetic_rate *= 60 * 60 * 24 / 1000
            return kinetic_rate

        class ReactionMechanism:
            def __init__(self, name, temperature_ref, k, Ea, n, p=1, q=1):
                self.SR_threshold = 100
                self.R = 8.314472

                self.name = name
                self.temperature_ref = temperature_ref  # gas constant [J/mol/Kelvin]
                self.k = k  # [mol * m-2 * s-1]
                self.Ea = Ea  # [J * mol-1]
                self.n = n  # reaction order with respect to given activity/anything
                self.p = p  # chemical affinity parameter
                self.q = q  # chemical affinity parameter

            def evaluate(self, temperature, activity, SR):
                k_arr = self.k * np.exp((-self.Ea / self.R) * (1 / temperature - 1 / self.temperature_ref))
                SR_bound = min(SR, self.SR_threshold)
                k_aff = (1 - SR_bound ** self.p) ** self.q
                rate = k_arr * k_aff * activity ** self.n
                return rate

    class CustomRelPerm:
        def __init__(self, exp, sr=0):
            self.exp = exp
            self.sr = sr

        def evaluate(self, sat):
            return (sat - self.sr) ** self.exp

    class GasViscosity:
        def __init__(self):
            pass
        def evaluate(self, pressure, temperature):
            return 0.0278

    class LiquidViscosity:
        def __init__(self):
            pass
        def evaluate(self, density, temperature):
            visc = _Viscosity(rho=density, T=temperature)
            return visc * 1000


