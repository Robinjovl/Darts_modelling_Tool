import numpy as np
import os
import h5py
import warnings
from darts.models.output import Output
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params, well_control_iface, timer_node
from darts.engines import copy_data_to_device
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.basic import ConstFunc
from darts.physics.chemistry.property_container import (
    OutputPropertyContainer,
    PropertyContainer,
)
from darts.physics.chemistry.physics import ElementBasedReactiveFlow
from darts.nonlinear_solvers import NewtonSolver
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.linear_solvers import SuperLUSolverSpec
from darts.physics.properties.kinetics import (
    KineticRate,
    LinearReactionSurfaceArea,
)
from darts.physics.properties.reaktoro import Flash as ReaktoroFlash
from reaktoro import (
    AqueousPhase,
    GaseousPhase,
    SupcrtDatabase,
    ChemicalSystem,
    ActivityModelPitzer,
    ActivityModelPengRobinsonPhreeqcOriginal,
    speciate,
)

class MyOutput(Output):
    def __init__(self, timer: timer_node, reservoir, physics, op_list, params, output_folder: str, sol_filename: str,
                 well_filename: str, save_initial: bool, all_phase_props: bool, precision: str, compression: str,
                 compression_level : int, verbose: bool):

        super().__init__(timer=timer, reservoir=reservoir, physics=physics, op_list=op_list, params=params,
                         output_folder=output_folder, sol_filename=sol_filename, well_filename=well_filename,
                         save_initial=save_initial, all_phase_props=all_phase_props, precision=precision,
                         compression=compression, compression_level=compression_level, verbose=verbose)

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

        return timesteps, property_array


class Model(DartsModel):
    def __init__(self, n_obl_mult: int = 9):
        super().__init__()
        self.n_obl_mult = n_obl_mult

        self.timer.node["initialization"].start()
        self.set_reservoir()
        self.set_physics()
        # Time-stepping and the linear solver are configured in set_solver()
        # (the unified self.linear_solver = <LinearSolverSpec> pattern), which the base
        # reset() calls before engine.init.
        self.runtime = 1
        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.set_sim_params(first_ts=1e-5, max_ts=1e-3  )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-5, max_iterations=15)
        # SuperLU direct solve for this small, stiff chemistry system, declared solely
        # through self.linear_solver (replaces the params.linear_type = cpu_superlu carrier,
        # which the base FGMRES+CPR default had been shadowing). proprietary_linear_type
        # carries the same enum for the proprietary build's engine factory.
        self.linear_solver = SuperLUSolverSpec(tolerance=1e-6, max_iterations=200,
                                        proprietary_linear_type=sim_params.cpu_superlu)

    def set_reservoir(self):

        self.reservoir = StructReservoir(self.timer, nx=2, ny=1, nz=1, dx=1, dy=1, dz=1,
                                         permx=100, permy=100, permz=100, poro=0.2, depth=100)
        return

    def set_physics(self):
        self.min_z = 1e-11
        # ambient conditions
        self.temperature = 298.15         # K
        self.pressure_init = 1            # bar

        # given input concentration of species
        h2o_mass = 1.0          # kg
        h2o_mole = h2o_mass / 0.018016          # mol
        co2_mole = 5.0          # mol
        calcite_mole = 1.0      # mol
        dolomite_mole = 0.0     # mol
        magnesite_mole = 1.0    # mol
        o2_mole = 1.e-6         # mol

        # convert to compostion
        fluid_moles = 3 * h2o_mole + 3 * co2_mole + 2 * o2_mole
        self.zCa = max(self.min_z, 0.0 / fluid_moles)
        self.zMg = max(self.min_z, 0.0 / fluid_moles)
        self.zC = max(self.min_z, co2_mole / fluid_moles)
        self.zO = max(self.min_z, (h2o_mole + 2 * (co2_mole + o2_mole)) / fluid_moles)
        self.zH = max(self.min_z, 2 * h2o_mole / fluid_moles)
        solid_moles = calcite_mole + dolomite_mole + magnesite_mole
        self.zCalcite = max(self.min_z, calcite_mole / (solid_moles + fluid_moles))
        self.zDolomite = max(self.min_z, dolomite_mole / (solid_moles + fluid_moles))
        self.zMagnesite = max(self.min_z, magnesite_mole / (solid_moles + fluid_moles))

        self.obl_min = self.min_z / 10
        gas = 'gas'
        liq = 'liq'
        self.phases = {gas: 0, liq: 1}
        phase_name = [list(self.phases.keys())[list(self.phases.values()).index(id)] for id in range(len(self.phases))]

        self.minerals = ['calcite', 'dolomite', 'magnesite']
        self.n_solid = len(self.minerals)

        self.elements = ['Solid_CaCO3', 'Solid_CaMg(CO3)2', 'Solid_MgCO3', 'Ca', 'Mg', 'C', 'O', 'H']
        Mw = {'Solid_CaCO3': 100.0869, 'Solid_CaMg(CO3)2': 184.401, 'Solid_MgCO3': 84.31,
                    'Ca': 40.078, 'Mg': 24.305, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol
        self.n_points = list(self.n_obl_mult * np.array([101, 201, 201, 201, 101, 101, 101, 101], dtype=np.intp))
        self.fc_mask = np.array([False, False, False, True, True, True, True, True], dtype=bool)
        self.axes_min = [self.pressure_init - 0.1] + [self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, self.obl_min, 0.3]
        self.axes_max = [self.pressure_init + 0.1] + [1 - self.obl_min, 0.4, 0.2, 0.01, 0.01, 0.1, 0.37]
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
        kinetic_mechanisms = ['acidic', 'neutral', 'carbonate']
        property_container = PropertyContainer(phases=self.phases, components_name=self.elements, Mw=Mw,
                                            stoich_matrix=stoich_matrix, eps_z=self.obl_min, temperature=self.temperature,
                                            fc_mask=self.fc_mask)

        property_container.permporo_mult_ev = ConstFunc(1.0)
        property_container.diffusion_ev = {ph: ConstFunc(np.concatenate([np.zeros(self.n_solid), \
                                         np.ones(self.nc - self.n_solid)]) * 5.2e-10 * 86400) for ph in self.phases}
        property_container.rel_perm_ev = {ph: CustomRelPerm(2) for ph in self.phases}
        property_container.viscosity_ev = { gas: GasViscosity(), liq: LiquidViscosity() }

        property_container.flash_ev = Flash(
            min_z=property_container.eps_z,
            minerals=property_container.minerals,
            components=property_container.components_name[property_container.fc_mask],
            temperature=property_container.temperature,
            database_filename='supcrtbl',
        )

        for min, props in rock_props.items():
            property_container.rock_compr_ev[min] = ConstFunc(props['compressibility'])
            property_container.rock_density_ev[min] = DensityBasic(compr=props['compressibility'], dens0=props['density'], p0=1.)

        # kinetics
        area_per_volume = 6.0 * 100.0 # cm2/cm3 -> m2/m3
        for m in property_container.minerals:
            rho_m = property_container.rock_density_ev[m].evaluate(self.pressure_init, self.temperature) / property_container.Mw[m]
            area_per_mol = area_per_volume / rho_m / 1000.0 # m2/mol
            surface_area_ev = LinearReactionSurfaceArea(initial_area_per_mol=area_per_mol * 10)
            property_container.kinetic_rate_ev[m] = KineticRate(
                min_z=self.obl_min,
                mineral_name=m.split('_', 1)[1],
                mechanisms=kinetic_mechanisms,
                surface_area_ev=surface_area_ev,
            )

        output_property_container = MyOutputPropertyContainer(property_container)

        axes_min_arr = np.asarray(self.axes_min, dtype=float)
        axes_max_arr = np.asarray(self.axes_max, dtype=float)
        n_points_arr = np.asarray(self.n_points, dtype=float)
        axes_step = ((axes_max_arr - axes_min_arr) / np.maximum(n_points_arr - 1, 1)).tolist()
        axes_origin = axes_min_arr.tolist()

        # Create instance of (own) physics class:
        self.physics = ElementBasedReactiveFlow(timer=self.timer, elements=self.elements, phases=phase_name,
                                                axes_step=axes_step, axes_origin=axes_origin,
                                                epsilon_z=property_container.eps_z, extrapolation_flag=False,
                                                cache=False)

        self.physics.add_property_region(property_container, output_property_container, 0)

        return

    def set_output(self, output_folder: str = 'output', sol_filename: str = 'reservoir_solution.h5',
                   well_filename: str = 'well_data.h5', save_initial: bool = True, all_phase_props : bool = False,
                   precision : str = 'd', compression : str = 'gzip', compression_level = 0, verbose : bool = False):
        self.output_folder = output_folder
        self.sol_filename  = sol_filename
        self.well_filename = well_filename
        self.sol_filepath  = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        self.output = MyOutput(self.timer, self.reservoir, self.physics, self.op_list, self.params, self.output_folder,
                               self.sol_filename, self.well_filename, save_initial, all_phase_props, precision, compression,
                               compression_level, verbose)

    def set_initial_conditions(self):
        input_distribution = {'pressure': self.pressure_init,
                            self.physics.vars[1]: self.zCalcite,
                            self.physics.vars[2]: self.zDolomite,
                            self.physics.vars[3]: self.zMagnesite,
                            self.physics.vars[4]: self.zCa,
                            self.physics.vars[5]: self.zMg,
                            self.physics.vars[6]: self.zC,
                            self.physics.vars[7]: self.zO
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                        input_distribution=input_distribution)

    def run_timestep(self, dt: float, t: float, verbose: bool = True):
        """
        Method to solve Newton loop for specified timestep

        :param dt: Timestep size [days]
        :type dt: float
        :param t: Current time [days]
        :type t: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        max_newt = self.nonlinear_solver.spec.max_iterations
        max_residual = np.zeros(max_newt + 1)
        solver = self.nonlinear_solver
        status = solver.status
        status.reset()
        # This loop never computes a well residual (the old C++ member stayed at
        # its initial value, so the well check in post_newtonloop never failed);
        # keep that behavior by zeroing it instead of leaving reset()'s inf.
        status.well_residual = 0.0
        self.timer.node["simulation"].start()
        residual_history = []
        for i in range(max_newt + 1):
            np.asarray(self.physics.engine.X)[::self.physics.n_vars] = self.pressure_init

            # assemble Jacobian and residual of reservoir and well blocks
            self.physics.engine.assemble_linear_system(dt)

            self.apply_rhs_flux(dt, t)  # apply RHS flux
            if self.platform == "gpu":
                copy_data_to_device(
                    self.physics.engine.RHS, self.physics.engine.get_RHS_d()
                )

            # calc norm of residual
            RHS = np.asarray(self.physics.engine.RHS)
            status.newton_residual = np.linalg.norm(RHS)
            # status.newton_residual = self.physics.engine.calc_newton_residual()

            max_residual[i] = status.newton_residual
            counter = 0
            for j in range(i):
                denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                if (
                    abs(max_residual[i] - max_residual[j]) / denom
                    < self.nonlinear_solver.spec.stationary_point_tolerance
                ):
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            residual_history.append(status.newton_residual)
            print(f'Newton iteration {i}: residual = {status.newton_residual}')

            status.n_newton = i
            #  check tolerance if it converges
            if status.newton_residual < self.nonlinear_solver.spec.tolerance or \
                    status.n_newton == max_newt:
                if i > 0:  # min_i_newton
                    break

            # Unified spec-driven dispatch (!280) + (rc, n_iters, residual) contract (!327)

            r_code, n_lin, _ = self._solve_linear_equation()

            status.linear_solver_rc = r_code

            if r_code != 0:

                self._linear_solver_rc_last = r_code

                break

            status.n_linear += n_lin
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop: convergence verdict previously made by the C++
        # post_newtonloop (linear solver rc + residual re-check), now in Python.
        converged = not (status.linear_solver_rc != 0 or
                         status.newton_residual >= self.nonlinear_solver.spec.tolerance or
                         status.well_residual > 1e2 * self.nonlinear_solver.spec.tolerance)
        converged = self.physics.engine.post_newtonloop(dt, t, converged)
        solver.stats.update(converged, status)

        self.timer.node["simulation"].stop()
        return converged


class CustomRelPerm:
    def __init__(self, exp, sr=0):
        self.exp = exp
        self.sr = sr

    def evaluate(self, sat):
        return 0.0 #(sat - self.sr) ** self.exp

class GasViscosity:
    def __init__(self):
        pass
    def evaluate(self, pressure, temperature):
        return 0.0278

class LiquidViscosity:
    def __init__(self):
        pass
    def evaluate(self, density, temperature):
        return 1.0

class PermPoroRelationship:
    def __init__(self, exp):
        self.exp = exp
    def evaluate(self, poro):
        return poro ** self.exp

class Flash(ReaktoroFlash):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_reaktoro_system(self):
        try:
            # Load SUPCRT database
            self.db = SupcrtDatabase(self.database_filename)
            self.aq_ending = "(aq)"
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.database_filename} with Reaktoro: {exc}"
            ) from exc

        # define equilibrium system
        try:
            elements_str = " ".join(self.components)
            aq = AqueousPhase(speciate(elements_str))
            gas = GaseousPhase("CO2(g)")
            self.system = ChemicalSystem(self.db,
            aq.set(ActivityModelPitzer()),
            gas.set(ActivityModelPengRobinsonPhreeqcOriginal()))
        except Exception as exc:
            raise RuntimeError(f"Failed to build ChemicalSystem: {exc}") from exc

        # get aqueous and sort them for consistent output
        phase_idx = self.system.phases().find("AqueousPhase")
        self.aqueous_species = [
            sp.name() for sp in self.system.phases()[phase_idx].species()
        ]
        phase_idx = self.system.phases().find("GaseousPhase")
        self.gas_species = [
            sp.name() for sp in self.system.phases()[phase_idx].species()
        ]

class MyOutputPropertyContainer(OutputPropertyContainer):
    def __init__(self, property_container, props_name: list[str] | None = None):
        super().__init__(property_container, props_name)

        self.dens_m = np.zeros(2)
        self.sat = np.zeros(2)
        self.dens_m_solid = np.zeros(len(self.property.flash_ev.mineral_names))
        self.sat_minerals = np.zeros(len(self.property.flash_ev.mineral_names))

        for i, ph in enumerate(self.property.phases_name):
            self.output_props['dens_m_' + ph] = lambda i=i: self.dens_m[i]
            self.output_props['sat_' + ph] = lambda i=i: self.sat[i]

        for i, m in enumerate(self.property.flash_ev.mineral_names):
            self.output_props['dens_m_solid_' + m] = lambda i=i: self.dens_m_solid[i]
            self.output_props['sat_' + m] = lambda i=i: self.sat_minerals[i]

    def evaluate(self, state):
        super().evaluate(state)
        self.property.evaluate(state)

        self.dens_m = self.property.dens_m
        self.sat = self.property.sat
        self.dens_m_solid = self.property.dens_m_solid
        self.sat_minerals = self.property.sat_minerals
