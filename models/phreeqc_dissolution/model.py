import numpy as np
import pickle, h5py
import os, sys

import darts
from darts.models.output import Output
from darts.models.cicd_model import CICDModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.basic import ConstFunc
from darts.physics.phreeqc.physics import PhreeqcDissolution
from darts.physics.properties.phreeqc import Flash, KineticRate
from darts.engines import sim_params, well_control_iface, value_vector, timer_node

from iapws._iapws import _Viscosity
from conversions import convert_composition, correct_composition, calculate_injection_stream, \
    get_mole_fractions


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

class MyOutput(Output):
    def __init__(self, timer: timer_node, reservoir, physics, op_list, params, output_folder: str, sol_filename: str,
                 well_filename: str, save_initial: bool, all_phase_props: bool, precision: str, compression: str,
                 verbose: bool):

        super().__init__(timer=timer, reservoir=reservoir, physics=physics, op_list=op_list, params=params,
                         output_folder=output_folder, sol_filename=sol_filename, well_filename=well_filename,
                         save_initial=save_initial, all_phase_props=all_phase_props, precision=precision,
                         compression=compression, verbose=verbose)

        # prepare arrays for evaluation of properties
        n_prop_ops = self.physics.input_data_struct.n_prop_ops
        n_vars = self.physics.n_vars
        n_res_blocks = self.reservoir.mesh.n_res_blocks
        self.prop_states = value_vector([0.] * n_res_blocks * (n_vars + 1))
        self.prop_states_np = np.asarray(self.prop_states)
        self.prop_values = value_vector([0.] * n_prop_ops * n_res_blocks)
        self.prop_values_np = np.asarray(self.prop_values)
        self.prop_dvalues = value_vector([0.] * n_prop_ops * n_res_blocks * n_vars)

        # extend units
        op = self.physics.property_operators[next(iter(self.physics.property_operators))]
        self.variable_units.update({name: '' for name in op.props_name})
        self.variable_units['porosity'] = ''
        self.variable_units[op.property.components_name[op.property.fc_mask][-1]] = ''

    def output_properties(self, filepath: str = None, output_properties: list = None, timestep: int = None, engine = False) -> tuple[np.ndarray, dict]:
        timesteps = [timestep] if timestep is not None else [0]
        if output_properties is None:
            prop_names = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        else:
            prop_names = output_properties

        X = np.asarray(self.physics.engine.X)
        nb = self.reservoir.mesh.n_res_blocks
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

        # hydrogen
        property = self.physics.property_operators[next(iter(self.physics.property_operators))].property
        fc = property.components_name[property.fc_mask]
        property_array[fc[-1]] = 1 - sum(property_array[c] for c in fc[:-1])

        # write to *.h5
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

        # locate tip of wormhole
        ids = np.where(property_array['porosity'][0] > 0.95)[0]
        if ids.size:
            if hasattr(self.reservoir.discretizer, 'centroids_all_cells'):
                id = np.argmax(self.reservoir.discretizer.centroids_all_cells[ids, 0])
                max_propagation = self.reservoir.discretizer.centroids_all_cells[:, 0].max()
                self.reservoir.wh_propagation_ratio = self.reservoir.discretizer.centroids_all_cells[ids, 0][id] / max_propagation
            else:
                warnings.warn("Centroids not available, setting wh_propagation_ratio to 0.0")
                self.reservoir.wh_propagation_ratio = 0.0
        else:
            self.reservoir.wh_propagation_ratio = 0.0
        print('WH propagation ratio:', self.reservoir.wh_propagation_ratio)

        return timesteps, property_array

# Actual Model class creation here!
class Model(CICDModel):
    def __init__(self, domain: str = '1D', nx: int = 200, mesh_filename: str = None,
                 poro_filename: str = None, minerals: list = ['calcite'],
                 kinetic_mechanisms=['acidic', 'neutral', 'carbonate'],
                 n_obl_mult: int = 1, co2_injection: float = 0.1, h2o_injection: float = 1.1,
                 inj_rate: float = None, perm_poro: str = 'power_8'):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()
        self.minerals = minerals
        self.kinetic_mechanisms = kinetic_mechanisms
        self.n_obl_mult = n_obl_mult
        self.n_solid = len(minerals)
        self.co2_injection = co2_injection
        self.h2o_injection = h2o_injection
        self.co2_injection_cutoff = 0.4
        self.inj_rate = inj_rate
        self.perm_poro = perm_poro

        self.set_reservoir(domain=domain, nx=nx, mesh_filename=mesh_filename, poro_filename=poro_filename)
        self.set_physics()

        self.set_sim_params(first_ts=1e-5, max_ts=1e-3, tol_newton=1e-4, tol_linear=1e-6, it_newton=15, it_linear=200)
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.nonlinear_norm_type = sim_params.nonlinear_norm_t.LINF
        # self.params.linear_type = sim_params.cpu_superlu
        self.params.newton_params[0] = 0.2
        self.runtime = 1

        self.timer.node["initialization"].stop()

    def set_output(self, output_folder: str = 'output', sol_filename: str = 'reservoir_solution.h5',
                   well_filename: str = 'well_data.h5', save_initial: bool = True, all_phase_props : bool = False,
                   precision : str = 'd', compression : str = 'gzip', verbose : bool = False):
        self.output_folder = output_folder
        self.sol_filename  = sol_filename
        self.well_filename = well_filename
        self.sol_filepath  = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        self.output = MyOutput(self.timer, self.reservoir, self.physics, self.op_list, self.params, self.output_folder,
                               self.sol_filename, self.well_filename, save_initial, all_phase_props, precision, compression, verbose)

    def set_physics(self):
        # some properties
        self.temperature = 323.15           # K
        self.pressure_init = 100            # bar

        if self.inj_rate is None:
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
            n_prop_ops = 24
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
                self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.4, 0.01, 0.01, 0.02, 0.37]
            else:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.25]
                self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.4, 0.01, 0.01, 0.1, 0.37]
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
            n_prop_ops = 32
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
            if self.co2_injection < self.co2_injection_cutoff:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.3]
                self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.4, 0.2, 0.01, 0.01, 0.02, 0.37]
            else:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.25]
                self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.4, 0.2, 0.01, 0.01, 0.1, 0.37]

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
            n_prop_ops = 36

        self.nc = len(self.elements)

        # Create property containers:
        is_gas_spec = False if self.co2_injection < self.co2_injection_cutoff else True
        property_container = ModelProperties(phases_name=self.phases, components_name=self.elements, Mw=Mw,
                                             kinetic_mechanisms=self.kinetic_mechanisms, min_z=self.obl_min,
                                             temperature=self.temperature, fc_mask=self.fc_mask, is_gas_spec=is_gas_spec)

        property_container.permporo_mult_ev = self.permporo
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
                                          input_data_struct=input_data_struct, properties=property_container, cache=True)

        self.physics.add_property_region(property_container, 0)

        # Compute injection stream
        mole_water, mole_co2 = calculate_injection_stream(self.h2o_injection, self.co2_injection, self.temperature, self.pressure_init) # input - m3 of water, co2
        mole_fraction_water, mole_fraction_co2 = get_mole_fractions(mole_water, mole_co2)
        print(f'rate={self.inj_rate} m3/day\t\t\tzH2O = {mole_fraction_water:.5f}\t\t\tzCO2 = {mole_fraction_co2:.5f}')

        # Define injection stream composition,
        self.inj_stream_components = np.zeros(len(self.components))
        self.inj_stream_components[self.components.index('H2O')] = mole_fraction_water     # H2O
        self.inj_stream_components[self.components.index('CO2')] = mole_fraction_co2       # CO2
        self.inj_stream = convert_composition(self.inj_stream_components, self.E)
        self.inj_stream = correct_composition(self.inj_stream, self.min_z)

    def set_reservoir(self, domain, nx, mesh_filename, poro_filename):
        self.domain = domain


        # permporo relationship
        self.params.enable_permporo = True
        true_initial_mean_poro = 0.3
        type, exp = self.perm_poro.split('_')
        self.perm_init = 1.25e4 * true_initial_mean_poro ** 4
        if type == 'power':
            self.permporo = PermPoroRelationship(exp=float(exp))
        else:
            print('Other than power law are not supported for permeability-porosity relationship')

        self.poro = 1  # self.poro=1 is for reservoir, poro is for initial state
        self.perm_max = self.perm_init / self.permporo.evaluate(true_initial_mean_poro)
        print(f'k_init = {self.perm_init/ 1e3} D\t\tk_max = {self.perm_max / 1e3} D')
        perm = self.perm_max * self.permporo.evaluate(self.poro)

        if self.domain == '1D':
            # grid
            self.domain_sizes = np.array([0.1, 0.001 * 7, 0.058905 / 7])
            self.domain_cells = np.array([nx, 1, 1])
            self.n_res_blocks = np.prod(self.domain_cells)
            self.cell_sizes = self.domain_sizes / self.domain_cells

            # properties
            depth = 1                      # m
            self.solid_sat = np.zeros((self.n_res_blocks, self.n_solid))
            if set(self.minerals) == {'calcite'}:
                self.solid_sat[:, 0] = 1 - true_initial_mean_poro
            elif set(self.minerals) == {'calcite', 'dolomite'}:
                self.solid_sat[:, 0] = (1 - true_initial_mean_poro) * 0.45
                self.solid_sat[:, 1] = (1 - true_initial_mean_poro) * 0.55
            elif set(self.minerals) == {'calcite', 'dolomite', 'magnesite'}:
                self.solid_sat[:, 0] = (1 - true_initial_mean_poro) * 0.35
                self.solid_sat[:, 1] = (1 - true_initial_mean_poro) * 0.45
                self.solid_sat[:, 2] = (1 - true_initial_mean_poro) * 0.2
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
            # porosity
            if poro_filename == None:
                poro = true_initial_mean_poro + np.random.uniform(-0.1, 0.1, self.n_res_blocks)
            else:
                poro = true_initial_mean_poro + 0.05 * np.loadtxt(poro_filename).flatten()
                assert np.prod(self.domain_cells) == poro.size
            poro[poro < 1.e-4] = 1.e-4
            poro[poro > 1 - 1.e-4] = 1 - 1.e-4

            self.solid_sat = np.zeros((self.n_res_blocks, self.n_solid))
            if set(self.minerals) == {'calcite'}:
                self.solid_sat[:, 0] = 1 - poro
            elif set(self.minerals) == {'calcite', 'dolomite'}:
                self.solid_sat[:, 0] = 0.8 * (1 - poro)
                self.solid_sat[:, 1] = 0.2 * (1 - poro)
            elif set(self.minerals) == {'calcite', 'dolomite', 'magnesite'}:
                self.solid_sat[:, 0] = 0.7 * (1 - poro)
                self.solid_sat[:, 1] = 0.2 * (1 - poro)
                self.solid_sat[:, 2] = 0.1 * (1 - poro)

            self.inj_cells = self.domain_cells[0] * np.arange(self.domain_cells[1])

            self.volume = np.prod(self.domain_sizes)
            self.reservoir = StructReservoir(self.timer,
                                             nx=self.domain_cells[0], ny=self.domain_cells[1], nz=self.domain_cells[2],
                                             dx=self.cell_sizes[0], dy=self.cell_sizes[1], dz=self.cell_sizes[2],
                                             permx=perm, permy=perm, permz=perm, poro=self.poro, depth=depth)
        elif self.domain == '3D':
            depth = 1
            mesh_file = mesh_filename
            self.reservoir = UnstructReservoir(timer=self.timer, permx=perm, permy=perm, permz=perm, frac_aper=0,
                                               mesh_file=mesh_file, poro=poro)
            self.reservoir.physical_tags['matrix'] = [99991]
            self.reservoir.physical_tags['boundary'] = [991, 992, 993]
            self.reservoir.init_reservoir()
            self.volume = np.asarray(self.reservoir.mesh.volume).sum()
            self.n_res_blocks = self.reservoir.mesh.n_blocks
            if poro_filename == None:
                poro = true_initial_mean_poro + np.random.uniform(-0.1, 0.1, self.n_res_blocks)
            else:
                poro = true_initial_mean_poro + 0.05 * np.loadtxt(poro_filename).flatten()
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

        # ensure passing compositional correction with zero update
        self.initial_comp[self.initial_comp < self.min_z] = self.min_z

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

    def run(self,
            days: float = None,
            restart_dt: float = 0.,
            save_well_data: bool = True,
            save_reservoir_data: bool = True,
            save_well_data_after_run: bool = True,
            verbose: bool = True):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        assert hasattr(self, 'output'), "self.output does not exist, please call m.set_output() after m.init()"
        days = days if days is not None else self.runtime
        data_ts = self.data_ts

        self.output.save_well_after_run = save_well_data_after_run

        if save_well_data_after_run:
            if not hasattr(self, "_well_output_configured"):
                self.output.configure_output(kind="well")
                self._well_output_configured = True
            else:
                pass

            self.output.well_time_labels = []
            self.output.well_data = []
            self.output.well_cfl = []

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
            dt = data_ts.dt_first
        elif restart_dt > 0.:
            dt = restart_dt
        else:
            dt = min(self.prev_dt*data_ts.dt_mult, days, data_ts.dt_max)

        self.prev_dt = dt

        ts = 0

        nc = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        max_dx = np.zeros(nc)

        n_good_steps = 0
        n_bad_steps = 0

        if np.fabs(data_ts.dt_mult - 1) < 1e-10:
            omega = 0.
        else:
            omega = 1 / (data_ts.dt_mult - 1)  # inversion assuming mult = (1 + omega) / omega

        while t < stop_time:
            xn = np.array(self.physics.engine.Xn, copy=True)[:nb * nc]  # need to copy since Xn will be updated Xn = X
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                self.physics.engine.t = t
                ts += 1

                x = np.array(self.physics.engine.X, copy=False)[:nb * nc]
                dt_mult_new = data_ts.dt_mult
                for i in range(nc):
                    max_dx[i] = np.max(abs(xn[i::nc] - x[i::nc]))
                    mult = ((1 + omega) * data_ts.eta[i]) / (max_dx[i] + omega * data_ts.eta[i])
                    if mult < dt_mult_new:
                        dt_mult_new = mult

                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d\tDT_MULT=%3.3g\tdX=%4s"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt,
                             dt_mult_new, np.round(max_dx, 3)))

                if fabs(dt - data_ts.dt_max) < 1.e-10 and self.physics.engine.n_newton_last_dt < self.ni_dt_increase_cutoff:
                    n_good_steps += 1
                else:
                    n_good_steps = 0

                if self.physics.engine.n_newton_last_dt > self.ni_dt_decrease_cutoff:
                    data_ts.dt_max /= 2 * data_ts.dt_mult
                    n_good_steps = 0

                if n_good_steps > self.n_good_ts:
                    data_ts.dt_max *= 2 * data_ts.dt_mult
                    n_good_steps = 0

                dt = min(dt * dt_mult_new, data_ts.dt_max)

                if np.fabs(t + dt - stop_time) < data_ts.dt_min:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

                n_bad_steps = 0

                # save well data at every converged time step
                if save_well_data and save_well_data_after_run is False:
                    self.output.save_data_to_h5(kind="well")
                else:
                    self.output.well_time_labels.append(self.physics.engine.t)
                    X = np.array(self.physics.engine.X, copy=False)

                    self.output.well_data.append(
                        X.reshape(self.reservoir.mesh.n_blocks, self.physics.n_vars)[
                            self.output.id_well_data
                        ]
                    )
                    self.output.well_cfl.append(self.physics.engine.CFL_max)
            else:
                dt /= data_ts.dt_mult
                n_good_steps = 0
                n_bad_steps += 1

                if n_bad_steps > 1:
                    data_ts.dt_max /= 2.
                    n_bad_steps = 0

                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                assert dt > data_ts.dt_min, ('Stop simulation. Reason: reached min. timestep '
                                                 + str(data_ts.dt_min) + ' dt=' + str(dt))

        # update current engine time
        self.physics.engine.t = stop_time

        # save well data after run
        if save_well_data and save_well_data_after_run is True:
            path = os.path.join(self.output_folder, self.well_filename)

            self.output.timer.start()
            self.output.timer.node["saving_well_data"].start()
            self.output.save_specific_data(
                path, [self.output.well_time_labels, self.output.well_data, self.output.well_cfl]
            )
            self.output.timer.node["saving_well_data"].stop()
            self.output.timer.stop()

        # save solution vector
        if save_reservoir_data:
            self.output.save_data_to_h5(kind="reservoir")

        if verbose:
            print(
                f"----- TS = {self.physics.engine.stat.n_timesteps_total:d}({self.physics.engine.stat.n_timesteps_wasted:d}), "
                f"NI = {self.physics.engine.stat.n_newton_total:d}({self.physics.engine.stat.n_newton_wasted:d}), "
                f"LI = {self.physics.engine.stat.n_linear_total:d}({self.physics.engine.stat.n_linear_wasted:d}) -----"
            )

        return 0

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
        self.flash_ev = Flash(min_z=self.min_z, fc_mask=self.fc_mask, fc_idx=self.fc_idx,
                                   f_mask_state=self.f_mask_state, temperature=self.temperature,
                                   minerals=self.minerals, is_gas_spec=is_gas_spec)

        self.kinetic_rate_ev = {m: KineticRate(self.temperature, self.min_z, m.split('_', 1)[1], kinetic_mechanisms) for m in self.minerals}
        self.rel_perm_ev = {ph: self.CustomRelPerm(2) for ph in phases_name[:2]}  # Relative perm for first two phases
        self.viscosity_ev = { phases_name[0]: self.GasViscosity(), phases_name[1]: self.LiquidViscosity() }

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector

        :return: updated value for operators, stored in values
        """
        nu_v, x, y, rho_phases, self.kin_state, _, _, _ = self.flash_ev.evaluate(state)
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

class PermPoroRelationship:
    def __init__(self, exp):
        self.exp = exp
    def evaluate(self, poro):
        return poro ** self.exp
