from math import fabs
import pickle
import os
import numpy as np

from darts.engines import conn_mesh, engine_base
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.physics.physics_base import PhysicsBase

from darts.engines import timer_node, sim_params, value_vector, index_vector, op_vector, ms_well_vector
from darts.engines import print_build_info as engines_pbi
from darts.discretizer import print_build_info as discretizer_pbi
from darts.print_build_info import print_build_info as package_pbi


class DartsModel:
    """
    This is a base class for creating a model in DARTS.
    A model is composed of a :class:`darts.models.Reservoir` object and a `darts.physics.Physics` object.
    Initialization and communication between these two objects takes place through the Model object

    :ivar reservoir: Reservoir object
    :type reservoir: :class:`ReservoirBase`
    :ivar mesh: Mesh object
    :type mesh: :class:`darts.engines.conn_mesh`
    :ivar physics: Physics object
    :type physics: :class:`PhysicsBase`
    :ivar engine: Engine object
    :type engine: :class:`darts.engines.engine`
    """
    reservoir: ReservoirBase
    mesh: conn_mesh
    physics: PhysicsBase
    engine: engine_base
    wells: ms_well_vector

    def __init__(self):
        """"
        Initialize DartsModel class.

        :ivar timer: Timer object
        :type timer: :class:`darts.engines.timer_node`
        :ivar params: Object to set simulation parameters
        :type params: :class:`darts.engines.sim_params`
        """
        # print out build information
        engines_pbi()
        discretizer_pbi()
        package_pbi()
        self.timer = timer_node()  # Create time_node object for time record
        self.timer.start()  # Start time record
        self.timer.node["simulation"] = timer_node()  # Create timer.node called "simulation" to record simulation time
        self.timer.node["newton update"] = timer_node()
        self.timer.node[
            "initialization"] = timer_node()  # Create timer.node called "initialization" to record initialization time
        self.timer.node["initialization"].start()  # Start recording "initialization" time

        self.params = sim_params()  # Create sim_params object to set simulation parameters

        self.timer.node["initialization"].stop()  # Stop recording "initialization" time

    def init(self):
        """
        Function to initialize the model, which includes:
        - initialize well (perforation) position
        - initialize well rate parameters
        - initialize reservoir initial conditions
        - initialize well control settings
        - define list of operator interpolators for accumulation-flux regions and wells
        - initialize engine
        """
        self.set_boundary_conditions()
        self.physics.init_wells(self.reservoir.wells)
        self.set_initial_conditions()
        self.set_well_controls()
        self.set_op_list()
        self.reset()

    def reset(self):
        """
        Function to initialize the engine by calling 'engine.init()' method.
        """
        self.engine.init(self.reservoir.mesh, ms_well_vector(self.reservoir.wells), op_vector(self.op_list),
                         self.params, self.timer.node["simulation"])

    def set_reservoir(self, reservoir: ReservoirBase, verbose: bool = False) -> None:
        """
        Function to define reservoir and initialize :class:`Reservoir` object.

        :param reservoir: :class:`Reservoir` object
        :type reservoir: ReservoirBase
        :param verbose: Set verbose level
        :type verbose: bool
        """
        self.reservoir = reservoir
        self.reservoir.init_reservoir(verbose=verbose)
        return

    def set_wells(self, verbose: bool = False) -> None:
        """
        Function to set (optionally) predefined wells in :class:`Reservoir` object
        and initialize :class:`ms_wells` object.

        :param verbose: Set verbose level
        :type verbose: bool
        """
        self.reservoir.set_wells(verbose=verbose)
        self.wells = self.reservoir.init_wells(verbose=verbose)
        return

    def set_physics(self, physics: PhysicsBase, discr_type: str = 'tpfa', platform: str = 'cpu',
                    verbose: bool = False) -> None:
        """
        Function to define properties and regions and initialize :class:`Physics` object.

        :param physics: :class:`Physics` object
        :type physics: PhysicsBase
        :param discr_type: Discretization type, 'tpfa' (default) or 'mpfa'
        :type discr_type: str
        :param platform: Switch for CPU/GPU engine, 'cpu' (default) or 'gpu'
        :type platform: str
        :param verbose: Set verbose level
        :type verbose: bool
        """
        self.physics = physics
        self.engine = self.physics.init_physics(discr_type=discr_type, platform=platform, verbose=verbose)
        if platform =='gpu':
            self.params.linear_type = sim_params.gpu_gmres_cpr_amgx_ilu
        return

    def set_initial_conditions(self, initial_values: dict = None, gradient: dict = None):
        """
        Function to set initial conditions. Passes initial conditions to :class:`Mesh` object.

        :param initial_values: Map of scalars/arrays of initial values for each primary variable, keys are the variables
        :type initial_values: dict
        :param gradient: Map of scalars of gradients for initial values
        :type gradient: dict
        """
        initial_values = initial_values if initial_values is not None else self.initial_values
        gradient = gradient if gradient is not None else (self.gradient if hasattr(self, 'gradient') else None)

        for i, variable in enumerate(self.physics.vars):
            # Check if variable exists in initial values dictionary
            if variable not in initial_values.keys():
                raise RuntimeError("Primary variable {} was not assigned initial values.".format(variable))

            self.reservoir.mesh.composition.resize(self.reservoir.mesh.n_blocks * (self.physics.nc - 1))
            if variable == 'pressure':
                values = np.array(self.reservoir.mesh.pressure, copy=False)
            elif variable == 'temperature':
                values = np.array(self.reservoir.mesh.temperature, copy=False)
            elif variable == 'enthalpy':
                values = np.array(self.reservoir.mesh.enthalpy, copy=False)
            else:
                values = np.array(self.reservoir.mesh.composition, copy=False)

            # values = np.array(self.reservoir.mesh.values[i], copy=False)
            initial_value = initial_values[variable]

            if variable not in ['pressure', 'temperature', 'enthalpy']:
                c = i-1
                values[c::(self.physics.nc - 1)] = initial_value
            elif isinstance(initial_values[variable], (list, np.ndarray)):
                # If initial value is an array, assign array
                values[:] = initial_value
            elif gradient is not None and variable in gradient.keys():
                # If gradient has been defined, calculate distribution over depth and assign to array
                for ith_cell in range(self.reservoir.mesh.n_res_blocks):
                    values[ith_cell] = initial_value + self.reservoir.mesh.depth[ith_cell] * gradient[variable]
            else:
                # Else, assign constant value to each cell in array
                values.fill(initial_value)

        return

    def set_boundary_conditions(self):
        """
        Function to set boundary conditions. Passes boundary conditions to :class:`Physics` object and wells.

        This function is empty in DartsModel, needs to be overloaded in child Model.
        """
        pass

    def set_well_controls(self):
        """
        Function to set well controls. Passes well controls to :class:`Physics` object and wells.

        This function is empty in DartsModel, needs to be overloaded in child Model.
        """
        pass

    def set_op_list(self):
        """
        Function to define list of operator interpolators for accumulation-flux regions and wells.

        Operator list is in order [acc_flux_itor[0], ..., acc_flux_itor[n-1], acc_flux_w_itor]
        """
        if type(self.physics.acc_flux_itor) == dict:
            self.op_list = list(self.physics.acc_flux_itor.values()) + [self.physics.acc_flux_w_itor]
            self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            self.op_num[self.reservoir.mesh.n_res_blocks:] = len(self.op_list) - 1
        else: # for backward compatibility
            self.op_list = [self.physics.acc_flux_itor]

    def set_sim_params(self, first_ts: float = None, mult_ts: float = None, max_ts: float = None, runtime: float = 1000,
                       tol_newton: float = None, tol_linear: float = None, it_newton: int = None, it_linear: int = None,
                       newton_type=None, newton_params=None):
        """
        Function to set simulation parameters.

        :param first_ts: First timestep
        :type first_ts: float
        :param mult_ts: Timestep multiplier
        :type mult_ts: float
        :param max_ts: Maximum timestep
        :type max_ts: float
        :param runtime: Total runtime in days, default is 1000
        :type runtime: float
        :param tol_newton: Tolerance for Newton iterations
        :type tol_newton: float
        :param tol_linear: Tolerance for linear iterations
        :type tol_linear: float
        :param it_newton: Maximum number of Newton iterations
        :type it_newton: int
        :param it_linear: Maximum number of linear iterations
        :type it_linear: int
        :param newton_type:
        :param newton_params:
        """
        self.params.first_ts = first_ts if first_ts is not None else self.params.first_ts
        self.params.mult_ts = mult_ts if mult_ts is not None else self.params.mult_ts
        self.params.max_ts = max_ts if max_ts is not None else self.params.max_ts
        self.runtime = runtime

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = tol_newton if tol_newton is not None else self.params.tolerance_newton
        self.params.tolerance_linear = tol_linear if tol_linear is not None else self.params.tolerance_linear
        self.params.max_i_newton = it_newton if it_newton is not None else self.params.max_i_newton
        self.params.max_i_linear = it_linear if it_linear is not None else self.params.max_i_linear

        self.params.newton_type = newton_type if newton_type is not None else self.params.newton_type
        self.params.newton_params = newton_params if newton_params is not None else self.params.newton_params

    def run(self, days: float = None):
        runtime = days if days is not None else self.runtime

        self.engine.run(runtime)

    def run_python(self, days: float = None, restart_dt: float = 0, timestep_python: bool = False):
        runtime = days if days is not None else self.runtime
        mult_dt = self.params.mult_ts
        max_dt = self.params.max_ts

        # get current engine time
        t = self.engine.t

        # same logic as in engine.run
        if fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0:
            dt = restart_dt
        else:
            dt = self.params.max_ts

        # evaluate end time
        runtime += t
        ts = 0

        while t < runtime:
            if timestep_python:
                 converged = self.engine.run_timestep(dt, t)
            else:
                 converged = self.run_timestep_python(dt, t)

            if converged:
                t += dt
                ts = ts + 1
                print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                      % (ts, t, dt, self.engine.n_newton_last_dt, self.engine.n_linear_last_dt))

                dt *= mult_dt
                if dt > max_dt:
                    dt = max_dt

                if t + dt > runtime:
                    dt = runtime - t

            else:
                dt /= mult_dt
                print("Cut timestep to %2.3f" % dt)
                if dt < 1e-12:
                    break
        # update current engine time
        self.engine.t = runtime

        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (self.engine.stat.n_timesteps_total, self.engine.stat.n_timesteps_wasted,
                                                         self.engine.stat.n_newton_total, self.engine.stat.n_newton_wasted,
                                                         self.engine.stat.n_linear_total, self.engine.stat.n_linear_wasted))

    def set_rhs_flux(self) -> np.ndarray:
        """
        Function to specify modifications to RHS vector. User can implement his own boundary conditions here.

        This function is empty in DartsModel, needs to be overloaded in child Model.

        :return: Vector of modification to RHS vector
        :rtype: np.ndarray
        """
        pass

    def apply_rhs_flux(self, dt: float):
        """
        Function to apply modifications to RHS vector.

        If self.set_rhs_flux() is defined in Model, this function will add its values to rhs

        :param dt: timestep [days]
        :type dt: float
        """
        if type(self).set_rhs_flux is DartsModel.set_rhs_flux:
            # If the function has not been overloaded, pass
            return
        rhs = np.array(self.engine.RHS, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs[:n_res] += self.set_rhs_flux() * dt
        return

    def run_timestep_python(self, dt, t):
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.engine.n_linear_last_dt = 0
        well_tolerance_coefficient = 1e2
        self.timer.node['simulation'].start()
        for i in range(max_newt+1):
            self.engine.run_single_newton_iteration(dt)
            self.apply_rhs_flux(dt)
            self.engine.newton_residual_last_dt = self.engine.calc_newton_residual()

            max_residual[i] = self.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j])/max_residual[i] < 1e-3:
                    counter += 1
            if counter > 2:
                print("Stationary point detected!")
                break

            self.engine.well_residual_last_dt = self.engine.calc_well_residual()
            self.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.engine.newton_residual_last_dt < self.params.tolerance_newton and
                 self.engine.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton) or
                    self.engine.n_newton_last_dt == self.params.max_i_newton):
                if i > 0:  # min_i_newton
                    break
            r_code = self.engine.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.engine.post_newtonloop(dt, t)
        self.timer.node['simulation'].stop()
        return converged

    def output_properties(self):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :returns: property_array
        :rtype: np.ndarray
        """
        # Initialize property_array
        n_vars = self.physics.n_vars
        n_props = self.physics.n_props
        tot_props = n_vars + n_props
        nb = self.reservoir.mesh.n_res_blocks
        property_array = np.zeros((tot_props, nb))

        # Obtain primary variables from engine
        for j in range(n_vars):
            property_array[j, :] = self.engine.X[j:nb * n_vars:n_vars]

        # If it has been defined, interpolate secondary variables in property_itor,
        if self.physics.property_operators is not None:
            values = value_vector(np.zeros(self.physics.n_ops))

            for i in range(nb):
                state = value_vector(property_array[0:n_vars, i])
                self.physics.property_itor.evaluate(state, values)

                for j in range(n_props):
                    property_array[j + n_vars, i] = values[j]

        return property_array

    def export_vtk(self, file_name: str = 'data', local_cell_data: dict = {}, global_cell_data: dict = {},
                   vars_data_dtype: type = np.float32, export_grid_data: bool = True):
        """
        Function to export results at timestamp t into `.vtk` format.

        :param file_name: Name to save .vtk file
        :type file_name: str
        :param local_cell_data: Local cell data (active cells)
        :type local_cell_data: dict
        :param global_cell_data: Global cell data (all cells including actnum)
        :type global_cell_data: dict
        :param vars_data_dtype:
        :type vars_data_dtype: type
        :param export_grid_data:
        :type export_grid_data: bool
        """
        # get current engine time
        t = self.engine.t
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        X = np.array(self.engine.X, copy=False)

        for v in range(nv):
            local_cell_data[self.physics.vars[v]] = X[v:nb * nv:nv].astype(vars_data_dtype)

        self.reservoir.output_to_vtk(file_name, t, local_cell_data, global_cell_data, export_grid_data)

    def load_restart_data(self, filename: str = 'restart.pkl'):
        """
        Function to load data from previous simulation and uses them for following simulation.
        :param filename: restart_data filename
        :type filename: str
        """
        if os.path.exists(filename):
            with open(filename, "rb") as fp:
                data = pickle.load(fp)
                days, X, arr_n = data
                self.engine.t = days
                self.engine.X = value_vector(X)
                self.engine.Xn = value_vector(X)
                self.engine.op_vals_arr_n = value_vector(arr_n)

    def save_restart_data(self, filename: str = 'restart.pkl'):
        """
        Function to save the simulation data for restart usage.
        :param filename: Name of the file where restart_data stores.
        :type filename: str
        """
        t = np.copy(self.engine.t)
        X = np.copy(self.engine.X)
        arr_n = np.copy(self.engine.op_vals_arr_n)
        data = [t, X, arr_n]
        with open(filename, "wb") as fp:
            pickle.dump(data, fp, 4)

    def print_timers(self):
        """
        Function to print the time information, including total time elapsed,
        time consumption at different stages of the simulation, etc..
        """
        print(self.timer.print("", ""))

    def print_stat(self):
        """
        Function to print the statistics information, including total timesteps, Newton iteration, linear iteration, etc..
        """
        self.engine.print_stat()

    # destructor to force to destroy all created C objects and free memory
    def __del__(self):
        for name in list(vars(self).keys()):
            delattr(self, name)
