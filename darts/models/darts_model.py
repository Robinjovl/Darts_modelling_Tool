from math import fabs
import pickle
import os
import numpy as np

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
    :ivar physics: Physics object
    :type physics: :class:`PhysicsBase`
    """

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
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.set_initial_conditions()
        self.set_boundary_conditions()
        self.set_op_list()
        self.reset()

    def reset(self):
        """
        Function to initialize the engine by calling 'physics.engine.init()' method.
        """
        self.physics.engine.init(self.reservoir.mesh, ms_well_vector(self.reservoir.wells),
                                 op_vector(self.op_list),
                                 self.params, self.timer.node["simulation"])

    def set_physics(self):
        """
        Function to define properties and regions and initialize :class:`Physics` object.

        This function is virtual in DartsModel, needs to be defined in child Model.
        """
        pass

    def set_wells(self):
        """
        Function to define wells and initialize :class:`Reservoir` object.

        This function is virtual in DartsModel, needs to be defined in child Model.
        """
        pass

    def set_initial_conditions(self):
        """
        Function to set initial conditions. Passes initial conditions to :class:`Physics` object.

        This function is virtual in DartsModel, needs to be defined in child Model.
        """
        pass

    def set_boundary_conditions(self):
        """
        Function to set boundary conditions. Passes boundary conditions to :class:`Physics` object and wells.

        This function is virtual in DartsModel, needs to be defined in child Model.
        """
        pass

    def set_op_list(self):
        """
        Function to define list of operator interpolators for accumulation-flux regions and wells.

        Operator list is in order [acc_flux_itor[0], ..., acc_flux_itor[n-1], acc_flux_w_itor]
        """
        if type(self.physics.acc_flux_itor) == dict:
            self.op_list = [acc_flux_itor for acc_flux_itor in self.physics.acc_flux_itor.values()] + [self.physics.acc_flux_w_itor]
            self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            # self.op_num[self.reservoir.nb:] = len(self.op_list) - 1
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

        self.physics.engine.run(runtime)

    def run_python(self, days: float, restart_dt: float = 0, timestep_python: bool = False):
        mult_dt = self.params.mult_ts
        max_dt = self.params.max_ts
        self.e = self.physics.engine

        # get current engine time
        t = self.e.t

        # same logic as in engine.run
        if fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0:
            dt = restart_dt
        else:
            dt = self.params.max_ts

        # evaluate end time
        runtime = t + days
        ts = 0

        while t < runtime:
            if timestep_python:
                 converged = self.e.run_timestep(dt, t)
            else:
                 converged = self.run_timestep_python(dt, t)

            if converged:
                t += dt
                ts = ts + 1
                print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                      % (ts, t, dt, self.e.n_newton_last_dt, self.e.n_linear_last_dt))

                dt *= mult_dt
                if dt > max_dt:
                    dt = max_dt

                if t + dt > runtime:
                    dt = runtime - t

            else:
                dt /= mult_dt
                print("Cut timestep to %2.3f" % dt)
                if dt < 1e-8:
                    break
        # update current engine time
        self.e.t = runtime

        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (self.e.stat.n_timesteps_total, self.e.stat.n_timesteps_wasted,
                                                         self.e.stat.n_newton_total, self.e.stat.n_newton_wasted,
                                                         self.e.stat.n_linear_total, self.e.stat.n_linear_wasted))

    def apply_rhs_flux(self, dt: float):
        '''
        if self.rhs_flux is defined and it is not None, add its values to rhs
        :param dt: timestep [days]
        '''
        if not hasattr(self, 'rhs_flux') or self.rhs_flux is None:
            return
        rhs = np.array(self.physics.engine.RHS, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs[:n_res] += self.rhs_flux * dt


    def run_timestep_python(self, dt, t):
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.e.n_linear_last_dt = 0
        well_tolerance_coefficient = 1e2
        self.timer.node['simulation'].start()
        for i in range(max_newt+1):
            self.e.run_single_newton_iteration(dt)
            self.apply_rhs_flux(dt)
            self.e.newton_residual_last_dt = self.e.calc_newton_residual()

            max_residual[i] = self.e.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j])/max_residual[i] < 1e-3:
                    counter += 1
            if counter > 2:
                print("Stationary point detected!")
                break

            self.e.well_residual_last_dt = self.e.calc_well_residual()
            self.e.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.e.newton_residual_last_dt < self.params.tolerance_newton and
                 self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton) or
                    self.e.n_newton_last_dt == self.params.max_i_newton):
                if i > 0:  # min_i_newton
                    break
            r_code = self.e.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.e.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.e.post_newtonloop(dt, t)
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
        property_array = np.zeros((self.reservoir.nb, tot_props))

        # Obtain primary variables from engine
        for j in range(n_vars):
            property_array[:, j] = self.physics.engine.X[j:self.reservoir.nb * n_vars:n_vars]

        # If it has been defined, interpolate secondary variables in property_itor,
        if self.physics.property_operators is not None:
            values = value_vector(np.zeros(self.physics.n_ops))

            for i in range(self.reservoir.nb):
                state = []
                for j in range(n_vars):
                    state.append(property_array[i, j])
                state = value_vector(np.asarray(state))
                self.physics.property_itor.evaluate(state, values)

                for j in range(n_props):
                    property_array[i, j + n_vars] = values[j]

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
        t = self.physics.engine.t
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        X = np.array(self.physics.engine.X, copy=False)

        for v in range(nv):
            local_cell_data[self.physics.vars[v]] = X[v:nb * nv:nv].astype(vars_data_dtype)

        self.reservoir.export_vtk(file_name, t, local_cell_data, global_cell_data, export_grid_data)

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
                self.physics.engine.t = days
                self.physics.engine.X = value_vector(X)
                self.physics.engine.Xn = value_vector(X)
                self.physics.engine.op_vals_arr_n = value_vector(arr_n)

    def save_restart_data(self, filename: str = 'restart.pkl'):
        """
        Function to save the simulation data for restart usage.
        :param filename: Name of the file where restart_data stores.
        :type filename: str
        """
        t = np.copy(self.physics.engine.t)
        X = np.copy(self.physics.engine.X)
        arr_n = np.copy(self.physics.engine.op_vals_arr_n)
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
        self.physics.engine.print_stat()

    # destructor to force to destroy all created C objects and free memory
    def __del__(self):
        for name in list(vars(self).keys()):
            delattr(self, name)
