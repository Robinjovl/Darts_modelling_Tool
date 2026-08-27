import numpy as np
from math import fabs
import h5py
import os

import darts
from darts.models.output import Output
from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.physics.chemistry.property_container import (
    OutputPropertyContainer,
    PropertyContainer,
)
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.basic import ConstFunc
from darts.physics.chemistry.physics import ElementBasedReactiveFlow
from darts.engines import sim_params, well_control_iface, value_vector, index_vector, timer_node, ms_well
from darts.physics.properties.kinetics import (
    KineticRate,
    LinearReactionSurfaceArea,
)
from darts.physics.properties.phreeqc import Flash as PhreeqcFlash, PhreeqcFlashError
from darts.physics.properties.reaktoro import Flash as ReaktoroFlash
from darts.physics.properties.flash_exceptions import FlashError
from darts.nonlinear_solvers import NewtonSolver, ChopSpec, Norm

from iapws._iapws import _Viscosity
from conversions import convert_composition, correct_composition, calculate_injection_stream, \
    get_mole_fractions


class MyOutput(Output):
    def __init__(self, timer: timer_node, reservoir, physics, op_list, params, output_folder: str, sol_filename: str,
                 well_filename: str, save_initial: bool, all_phase_props: bool, precision: str, compression: str,
                 compression_level : int, verbose: bool):

        super().__init__(timer=timer, reservoir=reservoir, physics=physics, op_list=op_list, params=params,
                         output_folder=output_folder, sol_filename=sol_filename, well_filename=well_filename,
                         save_initial=save_initial, all_phase_props=all_phase_props, precision=precision,
                         compression=compression, compression_level = compression_level, verbose=verbose)

        # prepare arrays for evaluation of properties
        n_prop_ops = self.physics.n_property_itor_ops
        n_vars = self.physics.n_vars
        n_res_blocks = self.reservoir.mesh.n_res_blocks
        self.prop_states = value_vector([0.] * n_res_blocks * (n_vars + 1))
        self.prop_states_np = np.asarray(self.prop_states)
        self.prop_values = value_vector([0.] * n_prop_ops * n_res_blocks)
        self.prop_values_np = np.asarray(self.prop_values)
        self.prop_dvalues = value_vector([0.] * n_prop_ops * n_res_blocks * n_vars)

        # extend units
        op = self.physics.output_property_containers[next(iter(self.physics.output_property_containers))]
        self.variable_units.update({name: '' for name in op.output_props.keys()})
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
        n_interp_size = self.physics.n_property_itor_ops

        # unknowns
        property_array = {var: np.array([X[i:nb * nv:nv]]) for i, var in enumerate(self.physics.vars)}
        # properties
        self.physics.property_itor[0].evaluate_with_derivatives(self.physics.engine.X, self.physics.engine.region_cell_idx[0],
                                                                self.prop_values, self.prop_dvalues)

        for i, prop in enumerate(prop_names):
            property_array[prop] = np.array([self.prop_values_np[i::n_interp_size]])

        # hydrogen
        property = self.physics.property_containers[next(iter(self.physics.property_containers))]
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
            axis = self.reservoir.wh_propagation_axis
            direction = self.reservoir.wh_propagation_direction
            coords = self.reservoir.discretizer.centroids_all_cells[:, axis]
            extent = coords.max() - coords.min()
            tip = coords[ids].max() if direction > 0 else coords[ids].min()
            reference = tip - coords.min() if direction > 0 else coords.max() - tip
            self.reservoir.wh_propagation_ratio = reference / extent if extent > 0 else 0.0
        else:
            self.reservoir.wh_propagation_ratio = 0.0
        print('WH propagation ratio:', self.reservoir.wh_propagation_ratio)

        return timesteps, property_array

# Actual Model class creation here!
class Model(DartsModel):
    def __init__(self, domain: str = '1D', nx: int = 200, mesh_filename: str = None,
                 poro_filename: str = None, minerals: list = ['calcite'],
                 kinetic_mechanisms=['acidic', 'neutral', 'carbonate'],
                 n_obl_mult: int = 1, co2_injection: float = 0.1, h2o_injection: float = 1.1,
                 inj_rate: float = None, perm_poro: str = 'power_8', flash: str = 'phreeqc',
                 database: str = 'phreeqc'):
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
        self.flash = flash
        self.database = database

        self.set_reservoir(domain=domain, nx=nx, mesh_filename=mesh_filename, poro_filename=poro_filename)
        self.set_physics()

        # initialize wormhole propagation ratio
        self.reservoir.wh_propagation_ratio = 0.0

        # Time-stepping / Newton / linear-solver config (see DartsModel.set_solver,
        # called from reset()).
        self.runtime = 1
        # default timestep control thresholds (overridable by callers)
        self.ni_dt_increase_cutoff = 5
        self.ni_dt_decrease_cutoff = 8
        self.n_good_ts = 10
        # Persistent count of consecutive "good" timesteps (at dt_max with few Newton
        # iterations). Kept on self so streaks accumulate ACROSS m.run() invocations:
        # the driver in main.py advances the simulation through many short run() calls,
        # and a per-call counter could never reach n_good_ts within a single short call,
        # freezing dt_max even when Newton convergence is trivial.
        self._n_good_steps = 0
        # Max number of Newton iterations within a timestep that may rely on the PHREEQC
        # dilution fallback before the timestep is abandoned and cut. The fallback handles
        # unreachable, over-concentrated OBL supporting points; if more than this many
        # nonlinear iterations need it, the step is not converging healthily -> cut dt.
        self.dilution_max_newton_iters = 3
        self._n_diluted_newton_iters = 0

        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.set_sim_params(first_ts=1e-5, max_ts=1e-3  )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-4, max_iterations=15,
            chop=ChopSpec(mode='local', factor=0.2))
        self.nonlinear_solver.spec.chop.mode = 'local'
        # self.params.nonlinear_norm_type = sim_params.nonlinear_norm_t.LINF
        self.nonlinear_solver.spec.chop.factor = 0.2
        # GPU -> AMGX-CPR; CPU -> FGMRES + CPR/AMG
        tolerance = 1e-6
        max_iterations = 500
        # Exact local (block-Schur) elimination of the mineral-balance equations
        # is ON BY DEFAULT for every mineral present (they are the cell-local /
        # diagonal-block-only equations here).
        K = self.n_solid
        n_vars = getattr(self.physics, 'n_vars', None)
        elim = K > 0 and (n_vars is None or n_vars - K >= 2)
        elim_rows = list(range(K))
        elim_cols = list(range(1, K + 1))
        if getattr(self, 'platform', 'cpu') == 'gpu':
            from darts.linear_solvers import AMGXCPRSolverSpec
            if elim:
                # NOTE: local elimination is incompatible with AMGX adaptive
                # hierarchy reuse (the reduced pressure COEFFICIENTS change every
                # Newton/timestep as condensation folds in the evolving local-
                # equation dynamics; a reused hierarchy goes stale -> AMGX setup
                # fails -> wasted Newton -> dt cut; measured on the 60k core:
                # reuse on made elimination +12% overall with 210 wasted Newtons,
                # reuse off -40% with none). The engine chain builder disables
                # reuse for the elimination chain's OWN AMGX instances (per-
                # instance ctor override) -- no process-global environment
                # mutation, other AMGX instances keep the default adaptive reuse.
                self.linear_solver = AMGXCPRSolverSpec(
                    max_iterations=max_iterations, tolerance=tolerance,
                    schur_elim_count=K, schur_elim_rows=elim_rows,
                    schur_elim_cols=elim_cols)
            else:
                self.linear_solver = AMGXCPRSolverSpec(
                    max_iterations=max_iterations, tolerance=tolerance)
        else:
            from darts.linear_solvers import CPRSolverSpec, GMRESSolverSpec
            spec = GMRESSolverSpec(restart=50, prec=CPRSolverSpec())
            spec.tolerance = tolerance
            spec.max_iterations = max_iterations
            if elim:
                from darts.linear_solvers import SchurEliminationSpec
                wrap = SchurEliminationSpec(inner=spec, elim_rows=elim_rows,
                                            elim_cols=elim_cols)
                wrap.tolerance = tolerance
                wrap.max_iterations = max_iterations
                spec = wrap
            self.linear_solver = spec

    def set_output(self, output_folder: str = 'output', sol_filename: str = 'reservoir_solution.h5',
                   well_filename: str = 'well_data.h5', save_initial: bool = True, all_phase_props : bool = False,
                   precision : str = 'd', compression : str = 'gzip', compression_level : int = 0, verbose : bool = False):
        self.output_folder = output_folder
        self.sol_filename  = sol_filename
        self.well_filename = well_filename
        self.sol_filepath  = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        self.output = MyOutput(self.timer, self.reservoir, self.physics, self.op_list, self.params, self.output_folder,
                               self.sol_filename, self.well_filename, save_initial, all_phase_props, precision, compression,
                               compression_level, verbose)

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
        gas = 'gas'
        liq = 'liq'
        self.phases = {gas: 0, liq: 1}
        phase_name = [list(self.phases.keys())[list(self.phases.values()).index(id)] for id in range(len(self.phases))]

        if self.domain == '3D':
            p_obl_max = self.pressure_init + 70
            n_obl_pressure = 2001
        else:
            p_obl_max = self.pressure_init + 5
            n_obl_pressure = 201
        if set(self.minerals) == {'calcite'}:
            # purely for initialization
            self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid_CaCO3']
            self.elements = ['Solid_CaCO3', 'Ca', 'C', 'O', 'H']
            self.fc_mask = np.array([False, True, True, True, True], dtype=bool)
            Mw = {'Solid_CaCO3': 100.0869, 'Ca': 40.078, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol

            self.n_points = list(self.n_obl_mult * np.array([n_obl_pressure, 201, 251, 251, 401], dtype=np.intp))
            self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, 0.2]
            self.axes_max = [p_obl_max] + [1 - self.obl_min, 0.1, 0.1, 0.6]
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
        elif set(self.minerals) == {'calcite', 'dolomite'}:
            # purely for initialization
            self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2',
                               'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid_CaCO3',                  # calcite-related
                               'CaMg(CO3)2', 'Mg+2', 'MgOH+', 'MgHCO3+', 'Solid_CaMg(CO3)2']        # dolomite-related
            self.elements = ['Solid_CaCO3', 'Solid_CaMg(CO3)2', 'Ca', 'Mg', 'C', 'O', 'H']
            self.fc_mask = np.array([False, False, True, True, True, True, True], dtype=bool)
            Mw = {'Solid_CaCO3': 100.0869, 'Solid_CaMg(CO3)2': 184.401,
                    'Ca': 40.078, 'Mg': 24.305, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol
            self.n_points = list(self.n_obl_mult * np.array([n_obl_pressure, 201, 201, 101, 101, 101, 101], dtype=np.intp))
            if self.co2_injection < self.co2_injection_cutoff:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.3]
                self.axes_max = [p_obl_max] + [1 - self.obl_min, 0.4, 0.01, 0.01, 0.02, 0.37]
            else:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.25]
                self.axes_max = [p_obl_max] + [1 - self.obl_min, 0.4, 0.01, 0.01, 0.1, 0.37]
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
        elif set(self.minerals) == {'calcite', 'dolomite', 'magnesite'}:
            # purely for initialization
            self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2',
                               'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid_CaCO3',                  # calcite-related
                               'CaMg(CO3)2', 'Mg+2', 'MgOH+', 'MgHCO3+', 'Solid_CaMg(CO3)2', 'Solid_MgCO3'] # dolomite-related
            self.elements = ['Solid_CaCO3', 'Solid_CaMg(CO3)2', 'Solid_MgCO3', 'Ca', 'Mg', 'C', 'O', 'H']
            self.fc_mask = np.array([False, False, False, True, True, True, True, True], dtype=bool)
            Mw = {'Solid_CaCO3': 100.0869, 'Solid_CaMg(CO3)2': 184.401, 'Solid_MgCO3': 84.31,
                    'Ca': 40.078, 'Mg': 24.305, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol
            self.n_points = list(self.n_obl_mult * np.array([n_obl_pressure, 201, 201, 201, 101, 101, 101, 101], dtype=np.intp))
            if self.co2_injection < self.co2_injection_cutoff:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.3]
                self.axes_max = [p_obl_max] + [1 - self.obl_min, 0.4, 0.2, 0.01, 0.01, 0.02, 0.37]
            else:
                self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.25]
                self.axes_max = [p_obl_max] + [1 - self.obl_min, 0.4, 0.2, 0.01, 0.01, 0.1, 0.37]

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

        self.nc = len(self.elements)

        # Create property containers:
        property_container = PropertyContainer(phases=self.phases, components_name=self.elements, Mw=Mw,
                                            stoich_matrix=stoich_matrix, eps_z=self.obl_min, temperature=self.temperature,
                                            fc_mask=self.fc_mask)
        property_container.permporo_mult_ev = self.permporo
        property_container.diffusion_ev = {ph: ConstFunc(np.concatenate([np.zeros(self.n_solid), \
                                         np.ones(self.nc - self.n_solid)]) * 5.2e-10 * 86400) for ph in self.phases}
        property_container.rel_perm_ev = {ph: CustomRelPerm(2) for ph in self.phases}
        property_container.viscosity_ev = { gas: GasViscosity(), liq: LiquidViscosity() }

        # flash, also here to be able to setup modified PHREEQC/reaktoro flashes
        if self.flash == 'phreeqc':
            # PHREEQC backend expects .dat filenames
            db_filename = f"{self.database}.dat"
            property_container.flash_ev = PhreeqcFlash(
                min_z=property_container.eps_z,
                minerals=property_container.minerals,
                components=property_container.components_name[property_container.fc_mask],
                temperature=property_container.temperature,
                database_filename=db_filename,
            )
        elif self.flash == 'reaktoro':
            # Reaktoro expects 'supcrtbl' without .dat; PHREEQC DBs with .dat
            db_filename = 'supcrtbl' if self.database == 'supcrtbl' else f"{self.database}.dat"
            property_container.flash_ev = ReaktoroFlash(
                min_z=property_container.eps_z,
                minerals=property_container.minerals,
                components=property_container.components_name[property_container.fc_mask],
                temperature=property_container.temperature,
                database_filename=db_filename,
            )
        else:
            raise ValueError(f'Invalid flash type: {self.flash}')

        # kinetics
        surface_area_ev = LinearReactionSurfaceArea(initial_area_per_mol=0.925)
        property_container.kinetic_rate_ev = {
            m: KineticRate(
                min_z=self.obl_min,
                mineral_name=m.split('_', 1)[1],
                mechanisms=self.kinetic_mechanisms,
                surface_area_ev=surface_area_ev,
            )
            for m in property_container.minerals
        }

        # Mineral properties
        for min, props in rock_props.items():
            property_container.rock_compr_ev[min] = ConstFunc(props['compressibility'])
            property_container.rock_density_ev[min] = DensityBasic(compr=props['compressibility'], dens0=props['density'], p0=1.)

        output_property_container = OutputPropertyContainer(property_container)

        axes_min_arr = np.asarray(self.axes_min, dtype=float)
        axes_max_arr = np.asarray(self.axes_max, dtype=float)
        n_points_arr = np.asarray(self.n_points, dtype=float)
        axes_step = ((axes_max_arr - axes_min_arr) / np.maximum(n_points_arr - 1, 1)).tolist()
        axes_origin = axes_min_arr.tolist()

        self.physics = ElementBasedReactiveFlow(timer=self.timer, elements=self.elements, phases=phase_name,
                                                axes_step=axes_step, axes_origin=axes_origin,
                                                epsilon_z=property_container.eps_z, extrapolation_flag=False,
                                                cache=True)
        self.physics.add_property_region(property_container, output_property_container, 0)

        # Bound the in-memory derived hypercube cache so the adaptive 8-D OBL
        # interpolator cannot exhaust RAM. The hypercube payloads are a pure derived,
        # never-persisted cache (rebuilt from supporting points with no flash), so
        # capping them does NOT touch the supporting-point cache or its *.pkl/*.fastcache
        # files. Read by PhysicsBase.create_interpolator. Sized to ~20x the reservoir
        # cell count (>> the per-Newton-iteration working set) so the hot-path hit rate
        # stays ~100%.
        self.physics.hypercube_cap = max(200_000, 20 * int(self.n_res_blocks))

        # Flashes whose per-iteration dilution fallback we police in run_timestep /
        # apply_rhs_flux. Only those exposing pop_dilution_report() (PHREEQC) qualify; the
        # reaktoro flash is silently ignored. NOTE: with parallel_evaluation=True the model is
        # reconstructed per worker (base DartsModel.get_evaluator_factory / ModelEvaluatorFactory),
        # so each worker uses its own flash copy; the budget/warning are only enforced in the
        # default in-process (parallel_evaluation=False) path; the flash still degrades
        # gracefully per worker regardless.
        self._tracked_flashes = [
            property_container.flash_ev
        ] if hasattr(property_container.flash_ev, 'pop_dilution_report') else []

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
                self.solid_sat[:, 0] = (1 - true_initial_mean_poro) * 0.4
                self.solid_sat[:, 1] = (1 - true_initial_mean_poro) * 0.2
                self.solid_sat[:, 2] = (1 - true_initial_mean_poro) * 0.4
            self.inj_cells = np.array([0])

            self.volume = np.prod(self.domain_sizes)
            self.reservoir = StructReservoir(self.timer,
                                             nx=self.domain_cells[0], ny=self.domain_cells[1], nz=self.domain_cells[2],
                                             dx=self.cell_sizes[0], dy=self.cell_sizes[1], dz=self.cell_sizes[2],
                                             permx=perm, permy=perm, permz=perm, poro=self.poro, depth=depth)
            self.reservoir.wh_propagation_axis = 0
            self.reservoir.wh_propagation_direction = 1
        elif self.domain == '2D':
            # grid
            if mesh_filename is None:
                self.domain_sizes = np.array([0.09, 0.09, 0.006])
                self.domain_cells = np.array([nx, nx, 1])
                self.n_res_blocks = np.prod(self.domain_cells)
                self.cell_sizes = self.domain_sizes / self.domain_cells

                self.volume = np.prod(self.domain_sizes)
                self.reservoir = StructReservoir(timer=self.timer,
                                                nx=self.domain_cells[0], ny=self.domain_cells[1], nz=self.domain_cells[2],
                                                dx=self.cell_sizes[0], dy=self.cell_sizes[1], dz=self.cell_sizes[2],
                                                permx=perm, permy=perm, permz=perm, poro=1, depth=1)
                self.inj_cells = self.domain_cells[0] * np.arange(self.domain_cells[1])
            else:
                self.reservoir = UnstructReservoir(timer=self.timer, permx=perm, permy=perm, permz=perm, frac_aper=0, cache=True,
                                                mesh_file=mesh_filename, poro=1)
                self.reservoir.physical_tags['matrix'] = [99991]
                self.reservoir.physical_tags['boundary'] = [991, 992, 993, 994, 995, 996]
                self.reservoir.init_reservoir()
                self.volume = np.asarray(self.reservoir.mesh.volume).sum()
                self.n_res_blocks = self.reservoir.mesh.n_blocks

                # identifying injection/production cells
                # for wedge
                angle = np.pi / 3 # regular triangle
                a = 2 / 3 * np.sqrt(self.volume / self.reservoir.mesh.n_blocks / 0.006 / np.sin(angle))
                max_x = self.reservoir.discretizer.mesh_data.points[:, 0].max()
                # initial guesses
                self.inj_cells = np.where(self.reservoir.discretizer.centroids_all_cells[:, 0] < 1.5 * a)[0]
                self.prd_cells = np.where(self.reservoir.discretizer.centroids_all_cells[:, 0] > max_x - 1.5 * a)[0]
                # exact filtering
                self.inj_cells = [id for id in self.inj_cells if np.count_nonzero(self.reservoir.discretizer.mat_cell_info_dict[id].coord_nodes_to_cell[:, 0] < 1e-4 * a) > 2]
                self.prd_cells = [id for id in self.prd_cells if np.count_nonzero(self.reservoir.discretizer.mat_cell_info_dict[id].coord_nodes_to_cell[:, 0] > max_x - 1e-4 * a) > 2]
                self.inj_cells = np.array(self.inj_cells, dtype=np.intp)
                self.prd_cells = np.array(self.prd_cells, dtype=np.intp)
            self.reservoir.wh_propagation_axis = 0
            self.reservoir.wh_propagation_direction = 1

            # porosity
            if poro_filename == None:
                poro = true_initial_mean_poro + np.random.uniform(-0.1, 0.1, self.n_res_blocks)
            else:
                poro = true_initial_mean_poro + 0.05 * np.loadtxt(poro_filename).flatten()
                if mesh_filename is None:
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
        elif self.domain == '3D':
            depth = 1
            mesh_file = mesh_filename
            self.reservoir = UnstructReservoir(timer=self.timer, permx=perm, permy=perm, permz=perm, frac_aper=0, cache=True,
                                               mesh_file=mesh_file, poro=1)
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
            self.prd_cells = np.where(self.reservoir.discretizer.centroids_all_cells[:, 2] < a)[0]
            self.inj_cells = np.where(self.reservoir.discretizer.centroids_all_cells[:, 2] > h - a)[0]
            # exact filtering
            self.prd_cells = [id for id in self.prd_cells if np.count_nonzero(self.reservoir.discretizer.mat_cell_info_dict[id].coord_nodes_to_cell[:, 2] < 1e-4 * a) > 2]
            self.inj_cells = [id for id in self.inj_cells if np.count_nonzero(self.reservoir.discretizer.mat_cell_info_dict[id].coord_nodes_to_cell[:, 2] > h - 1e-4 * a) > 2]
            self.prd_cells = np.array(self.prd_cells, dtype=np.intp)
            self.inj_cells = np.array(self.inj_cells, dtype=np.intp)
            self.reservoir.wh_propagation_axis = 2
            self.reservoir.wh_propagation_direction = -1
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

        # The fluid composition is identical for every reservoir block; only the per-block
        # solid saturation varies. Compute the composition once, assemble all per-block OBL
        # states in one contiguous array, and evaluate the initialization interpolator for
        # every block in a SINGLE batched call. The batch path materializes all missing
        # supporting points through evaluate_batch() (the parallel worker pool) in one shot,
        # instead of one point at a time through the single-point evaluate() (which runs
        # every reaktoro flash serially in the main process and used to dominate init time).
        # The numerical result is identical to the former per-block loop.
        composition_full = convert_composition(self.initial_comp_components, self.E)
        composition = correct_composition(composition_full, self.min_z)
        comp_tail = np.ascontiguousarray(composition[self.n_solid:], dtype=np.float64)

        n_blocks = self.n_res_blocks
        n_ops = self.physics.n_comp_itor_ops
        n_dims = 1 + self.n_solid + comp_tail.size  # [pressure | solid_sat | comp_tail]

        # One contiguous [n_blocks, n_dims] state array (row-major == point-major layout the
        # interpolator expects after ravel()).
        states = np.empty((n_blocks, n_dims), dtype=np.float64)
        states[:, 0] = self.pressure_init
        states[:, 1:1 + self.n_solid] = self.solid_sat
        states[:, 1 + self.n_solid:] = comp_tail  # broadcast the shared fluid tail

        # Single batched interpolation over all blocks; missing supporting points are
        # materialized in parallel via evaluate_batch(). Derivatives are required by the
        # C++ signature but unused here.
        values = value_vector(np.zeros(n_blocks * n_ops))
        dvalues = value_vector(np.zeros(n_blocks * n_ops * n_dims))
        block_idxs = index_vector(np.arange(n_blocks, dtype=np.int32))
        self.physics.comp_itor[0].evaluate_with_derivatives(
            value_vector(states.ravel()), block_idxs, values, dvalues)
        values_np = np.asarray(values).reshape(n_blocks, n_ops)

        # Assemble the initial composition: per-block solid fractions + the shared fluid
        # tail (vectorized over all blocks).
        self.solid_frac[:] = values_np[:, :self.n_solid]
        self.initial_comp[:n_blocks, :self.n_solid] = self.solid_frac
        self.initial_comp[:n_blocks, self.n_solid:] = composition_full[self.n_solid:-1]

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
        w_d = 1.5
        well_index = 5

        # self.reservoir.add_well("I1", well_diameter=w_d)
        # for idx in range(self.domain_cells[1]):
        #     self.reservoir.add_perforation(well_name='I1', res_cell_idx=(1, idx + 1, 1), ms_epm=False,
        #                                    verbose=True, well_diameter=w_d, well_index=well_index,
        #                                    well_indexD=well_index)

        self.reservoir.add_well("P1", well_diameter=w_d)
        if isinstance(self.reservoir, UnstructReservoir):
            for idx in self.prd_cells:
                self.reservoir.add_perforation(well_name='P1', res_cell_idx=idx, ms_epm=False,
                                               verbose=False, well_diameter=w_d, well_index=well_index,
                                               well_indexD=well_index)
        elif isinstance(self.reservoir, StructReservoir):
            for idx in range(self.domain_cells[1]):
                self.reservoir.add_perforation(well_name='P1', res_cell_idx=(self.domain_cells[0], idx + 1, 1), ms_epm=False,
                                               verbose=True, well_diameter=w_d, well_index=well_index,
                                               well_indexD=well_index)

    def set_rhs_flux(self, t: float = None):
        nv = self.physics.n_vars
        nb = self.reservoir.mesh.n_blocks
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

    def run_timestep(self, dt: float, t: float, verbose: bool = True):
        """
        Newton loop with PHREEQC dilution-fallback policing.

        Delegates to the base Newton loop but (1) resets the per-timestep dilution budget
        and the flashes' per-iteration trackers, and (2) converts any :class:`FlashError`
        into a non-convergence so the existing dt-cut machinery in :meth:`run` reduces dt
        and retries from the last converged state. This covers a PHREEQC failure (when even
        maximal dilution fails, or when the dilution fallback was needed in more than
        ``self.dilution_max_newton_iters`` nonlinear iterations — see :meth:`apply_rhs_flux`)
        as well as a Reaktoro solver failure (``ReaktoroFlashError``), keeping the
        simulation alive in either case.
        """
        self._n_diluted_newton_iters = 0
        for fl in getattr(self, '_tracked_flashes', []):
            fl.reset_dilution_tracker()
        try:
            return super().run_timestep(dt, t, verbose)
        except FlashError as e:
            if verbose:
                print(f"Flash non-convergence -> cutting timestep (dt={dt:.6g}): {e}")
            # The simulation timer was started inside the base run_timestep and is not
            # stopped on the exception path; stop it so timing/print_timers stay consistent.
            try:
                self.timer.node["simulation"].stop()
            except Exception:
                pass
            # post_newtonloop (which does X = Xn on non-convergence) is skipped on the
            # exception path, so restore the last converged iterate explicitly; the
            # smaller-dt retry then starts clean and stays clear of the unreachable corner.
            try:
                X = np.array(self.physics.engine.X, copy=False)
                Xn = np.array(self.physics.engine.Xn, copy=False)
                X[:] = Xn
            except Exception:
                pass
            # Mirror the base method's per-step history bookkeeping for the failed step.
            try:
                self.time.append(t)
                self.n_newton_iters.append(self.nonlinear_solver.status.n_newton)
                self.time_step_size.append(dt)
            except Exception:
                pass
            return 0  # converged = False -> run() else-branch cuts dt

    def apply_rhs_flux(self, dt: float, t: float):
        """
        Apply the injection RHS flux, then police the PHREEQC dilution fallback.

        Called once per Newton iteration immediately after ``assemble_linear_system`` (so any
        supporting-point dilution that happened during this assembly is now recorded in the
        tracked flashes). Emits a single accumulated warning per nonlinear iteration with
        min/max state statistics, and enforces the per-timestep dilution-iteration budget by
        raising :class:`PhreeqcFlashError` (caught in :meth:`run_timestep`) once exceeded.
        """
        super().apply_rhs_flux(dt, t)

        tracked = getattr(self, '_tracked_flashes', [])
        reports = [r for r in (fl.pop_dilution_report() for fl in tracked) if r]
        if not reports:
            return

        count = sum(r['count'] for r in reports)
        state_min = np.min([r['state_min'] for r in reports], axis=0)
        state_max = np.max([r['state_max'] for r in reports], axis=0)
        factor_min = min(r['factor_min'] for r in reports)
        factor_max = max(r['factor_max'] for r in reports)
        molality_min = min(r['molality_min'] for r in reports)
        molality_max = max(r['molality_max'] for r in reports)

        self._n_diluted_newton_iters += 1
        print(
            f"[flash dilution] t={t:.6g} dt={dt:.3g} NL-iter "
            f"#{self._n_diluted_newton_iters}/{self.dilution_max_newton_iters}: "
            f"diluted {count} supporting point(s); nominal molality "
            f"{molality_min:.1f}-{molality_max:.1f}, dilution factor "
            f"{factor_min:.2f}-{factor_max:.2f}; state min="
            f"{np.array2string(state_min, precision=4, suppress_small=True)} max="
            f"{np.array2string(state_max, precision=4, suppress_small=True)}"
        )

        if self._n_diluted_newton_iters > self.dilution_max_newton_iters:
            raise PhreeqcFlashError(
                f"dilution fallback needed in more than {self.dilution_max_newton_iters} "
                f"nonlinear iterations this timestep (dt={dt:.6g})"
            )

    def run(self,
            days: float = None,
            restart_dt: float = 0.,
            save_well_data: bool = True,
            save_reservoir_data: bool = True,
            save_well_data_after_run: bool = True,
            verbose: int | None = None):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Verbosity level (``int``; ``bool`` accepted). Defaults to
            ``None``, meaning inherit :attr:`self.verbose`. ``>=VERBOSE_TIMERS`` (2)
            additionally prints timers at the end of every run() invocation.
        :type verbose: int
        """
        verbose = self.verbose if verbose is None else verbose
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

        # NOTE: self._n_good_steps is deliberately NOT reset here so the good-step streak
        # carries across consecutive run() calls (see __init__).
        n_bad_steps = 0
        # Whether the dt of the step about to be taken was shortened only to land exactly
        # on stop_time (a reporting boundary). Such a step is not at dt_max by physics, so
        # it must not break the good-step streak.
        dt_truncated = False

        if np.fabs(data_ts.dt_mult - 1) < 1e-10:
            omega = 0.
        else:
            omega = 1 / (data_ts.dt_mult - 1)  # inversion assuming mult = (1 + omega) / omega

        # Per-timestep Python orchestration outside run_timestep (state copies, dt/CFL
        # control, well-data accumulation) is otherwise untimed; bracket it into the
        # "run loop overhead" node instead of leaving it in the root "Total elapsed" gap.
        overhead = self.timer.node["run loop overhead"]
        while t < stop_time:
            overhead.start()
            xn = np.array(self.physics.engine.Xn, copy=True)[:nb * nc]  # need to copy since Xn will be updated Xn = X
            overhead.stop()
            converged = self.run_timestep(dt, t, verbose)
            status = self.nonlinear_solver.status

            overhead.start()
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
                    max_dx_str = '[' + ', '.join(f'{v:.1e}' for v in max_dx) + ']'
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d\tDT_MULT=%3.3g\tdX=%s"
                          % (ts, t, dt, status.n_newton, status.n_linear,
                             dt_mult_new, max_dx_str))

                if dt_truncated:
                    # Boundary-truncated step: it carries no information about whether
                    # dt_max is sustainable, so leave the streak untouched (neither
                    # increment nor reset).
                    pass
                elif fabs(dt - data_ts.dt_max) < 1.e-10 and status.n_newton < self.ni_dt_increase_cutoff:
                    self._n_good_steps += 1
                else:
                    self._n_good_steps = 0

                if status.n_newton > self.ni_dt_decrease_cutoff:
                    data_ts.dt_max /= 2 * data_ts.dt_mult
                    self._n_good_steps = 0

                if self._n_good_steps > self.n_good_ts:
                    data_ts.dt_max *= 2 * data_ts.dt_mult
                    self._n_good_steps = 0

                dt = min(dt * dt_mult_new, data_ts.dt_max)

                dt_truncated = False
                if np.fabs(t + dt - stop_time) < data_ts.dt_min:
                    dt = stop_time - t
                    dt_truncated = True

                if t + dt > stop_time:
                    dt = stop_time - t
                    dt_truncated = True
                else:
                    self.prev_dt = dt

                n_bad_steps = 0

                # save well data at every converged time step
                if save_well_data and save_well_data_after_run is False:
                    # save_data_to_h5 brackets its own output/saving_well_data timer; pause
                    # the overhead bracket so the h5 write is not double-counted.
                    overhead.stop()
                    self.output.save_data_to_h5(kind="well")
                    overhead.start()
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
                if getattr(self, '_linear_solver_rc_last', 0) != 0:
                    dt /= 10.0
                    n_bad_steps += 2
                else:
                    dt /= data_ts.dt_mult
                    n_bad_steps += 1
                self._n_good_steps = 0
                dt_truncated = False

                if n_bad_steps > 1:
                    data_ts.dt_max /= 2.
                    n_bad_steps = 0

                if verbose:
                    print("Cut timestep to %2.10f (solver rc=%d)"
                          % (dt, getattr(self, '_linear_solver_rc_last', 0)))
                if dt <= data_ts.dt_min:
                    overhead.stop()  # keep the bracket balanced before aborting the run
                    raise RuntimeError('Stop simulation. Reason: reached min. timestep '
                                       + str(data_ts.dt_min) + ' dt=' + str(dt))

            overhead.stop()

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

        # Flush OBL adaptive cache between snapshots so progress survives SIGTERM / job cancel.
        if getattr(self.physics, 'cache', False):
            self.timer.node["cache I/O"].start()
            self.physics.write_cache()
            self.timer.node["cache I/O"].stop()

        if verbose:
            stats = self.nonlinear_solver.stats
            print(
                f"----- TS = {stats.n_timesteps_total:d}({stats.n_timesteps_wasted:d}), "
                f"NI = {stats.n_newton_total:d}({stats.n_newton_wasted:d}), "
                f"LI = {stats.n_linear_total:d}({stats.n_linear_wasted:d}) -----"
            )

        # At higher verbosity, print the timer breakdown at the end of every run()
        # invocation (mirrors DartsModel.run(), which this override otherwise bypasses).
        # Automatic prints go to the redirected darts log rather than stdout.
        if verbose >= self.VERBOSE_TIMERS:
            self.print_timers(to_log=True)

        return 0


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
    # IAPWS validity envelope: outside it the correlation's exp() underflows to
    # exactly 0.0 (seen at rho ~ 2280 kg/m3 from extreme single-phase-aq flash
    # results at unreachable OBL corners), and mu = 0 turns the mobility operator
    # kr/mu into +inf, silently poisoning the OBL point cache and every hypercube
    # (and hence Jacobian) that touches it. Clamp the density into the correlation
    # range and floor the result so mobility stays finite.
    RHO_MAX = 1200.0   # kg/m3, upper edge of the IAPWS viscosity correlation range
    MU_MIN = 1e-3      # cP, positive floor (gas-like); only hit on degenerate inputs
    def __init__(self):
        pass
    def evaluate(self, density, temperature):
        visc = _Viscosity(rho=min(density, self.RHO_MAX), T=temperature)
        return max(visc * 1000, self.MU_MIN)

class PermPoroRelationship:
    def __init__(self, exp):
        self.exp = exp
    def evaluate(self, poro):
        return poro ** self.exp
