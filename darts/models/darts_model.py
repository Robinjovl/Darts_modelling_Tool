from math import fabs
import pickle
import xarray as xr
import h5py
import os
import numpy as np
from scipy.interpolate import interp1d

from darts.reservoirs.reservoir_base import ReservoirBase
from darts.physics.base.physics_base import PhysicsBase

from darts.engines import timer_node, sim_params, value_vector, index_vector, op_vector, ms_well_vector
from darts.engines import print_build_info as engines_pbi
from darts.discretizer import print_build_info as discretizer_pbi
from darts.print_build_info import print_build_info as package_pbi


class DartsModel:
    """
    This is a base class for creating a model in DARTS.
    A model is composed of a :class:`Reservoir` object and a :class:`Physics` object.
    Initialization and communication between these two objects takes place through the Model object

    :ivar reservoir: Reservoir object
    :type reservoir: :class:`ReservoirBase`
    :ivar physics: Physics object
    :type physics: :class:`PhysicsBase`
    """
    reservoir: ReservoirBase
    physics: PhysicsBase

    def __init__(self):
        """
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
        self.timer.node["vtk_output"] = timer_node()
        self.timer.node["initialization"] = timer_node()  # Create timer.node called "initialization" to record initialization time
        self.timer.node["initialization"].start()  # Start recording "initialization" time
        self.output_folder = 'output'
        self.sol_filename = "solution.h5"
        self.well_filename = 'well_data.h5'

        self.params = sim_params()  # Create sim_params object to set simulation parameters

        self.timer.node["initialization"].stop()  # Stop recording "initialization" time

    def init(self, discr_type: str = 'tpfa', platform: str = 'cpu', restart: bool = False,
             verbose: bool = False, output_folder: str = None, itor_mode: str = 'adaptive',
             itor_type: str = 'multilinear', is_barycentric: bool = False):
        """
        Function to initialize the model, which includes:
        - initialize well (perforation) position
        - initialize well rate parameters
        - initialize reservoir initial conditions
        - initialize well control settings
        - define list of operator interpolators for accumulation-flux regions and wells
        - initialize engine

        :param discr_type: 'tpfa' for using Python implementation of TPFA, 'mpfa' activates C++ implementation of MPFA
        :type discr_type: str
        :param platform: 'cpu' for CPU, 'gpu' for using GPU for matrix assembly/solvers/interpolators
        :type platform: str
        :param restart: Boolean to check if existing file should be overwritten or appended
        :type restart: bool
        :param verbose: Switch for verbose
        :type verbose: bool
        :param output_folder: folder for h5 output files
        :type output_folder: str
        :param itor_mode: specifies either 'static' or 'adaptive' interpolator
        :type itor_mode: str
        :param itor_type: specifies either 'linear' or 'multilinear' interpolator
        :type itor_type: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        """
        # Initialize reservoir and Mesh object
        assert self.reservoir is not None, "Reservoir object has not been defined"
        self.reservoir.init_reservoir(verbose)
        self.set_wells()

        # Initialize physics and Engine object
        assert self.physics is not None, "Physics object has not been defined"
        self.physics.init_physics(discr_type=discr_type, platform=platform, verbose=verbose,
                                  itor_mode=itor_mode, itor_type=itor_type, is_barycentric=is_barycentric)
        if platform == 'gpu':
            self.params.linear_type = sim_params.gpu_gmres_cpr_amgx_ilu

        # Initialize well objects
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.init_well_rates()

        if output_folder is not None:
            self.output_folder = output_folder

        self.set_op_list()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.set_well_controls()
        self.reset()

        # self.restart = restart

        # save solution vector
        if restart is False:
            self.save_data_to_h5(kind = 'solution')

    def reset(self):
        """
        Function to initialize the engine by calling 'engine.init()' method.
        """
        self.physics.engine.init(self.reservoir.mesh, ms_well_vector(self.reservoir.wells), op_vector(self.op_list),
                                 self.params, self.timer.node["simulation"])

    def set_output(self, output_folder = 'output', sol_filename = 'reservoir_solution.h5', restart = False, all_phase_props = True, precision = 'd'):
        """
       Function to initialize output class
        : param output_folder: folder for h5 output files
        : param sol_filename: filename of output file
        : param restart: Boolean to check if existing file should be overwritten or appended, deault is False (overwritten)
        : param all_phase_props: Boolean to output all phase properties
        : param precision: data precision of saved data ('s' single precision, 'd' double precssion)
        """
        self.output = Output(self.timer, self.reservoir, self.physics, self.op_list, self.params, output_folder, sol_filename, restart, all_phase_props, precision)
        return

    def set_wells(self, verbose: bool = False):
        """
        Function to define wells. The default method of DartsModel.set_wells() calls Reservoir.set_wells().

        :param verbose: Switch for verbose
        :type verbose: bool
        """
        self.reservoir.set_wells(verbose)
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

        self.reservoir.mesh.composition.resize(self.reservoir.mesh.n_blocks * (self.physics.nc - 1))

        for i, variable in enumerate(self.physics.vars):
            # Check if variable exists in initial values dictionary
            if variable not in initial_values.keys():
                raise RuntimeError("Primary variable {} was not assigned initial values.".format(variable))

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
                c = i - 1
                values[c::(self.physics.nc - 1)] = initial_value
            elif isinstance(initial_values[variable], (list, np.ndarray)):
                # If initial value is an array, assign array
                values[:] = initial_value
            elif gradient is not None and variable in gradient.keys():
                # If gradient has been defined, calculate distribution over depth and assign to array
                values[:self.reservoir.mesh.n_res_blocks] = initial_value + \
                    np.asarray(self.reservoir.mesh.depth)[:self.reservoir.mesh.n_res_blocks] * gradient[variable]
            else:
                # Else, assign constant value to each cell in array
                values.fill(initial_value)

        return

    def set_initial_conditions_from_depth_table(self, depth, initial_distribution: dict):
        """
        Function to set initial conditions from given distribution of properties over depth.

        :param depth: depth
        :param initial_distribution: initial distributions of unknowns over depth,
                                    must have keys equal to self.physics.vars
        :type initial_distribution: dict
        """

        # all depths
        depths = np.asarray(self.reservoir.mesh.depth)

        # adjust the size of composition array in c++
        self.reservoir.mesh.composition.resize(self.reservoir.mesh.n_blocks * (self.physics.nc - 1))

        z_counter = 0
        nz_vars = self.physics.nc - 1
        for variable in self.physics.vars:
            if variable not in initial_distribution.keys():
                raise RuntimeError("Primary variable {} was not assigned initial values.".format(variable))

            values_foo = interp1d(depth, initial_distribution[variable], kind='linear', fill_value='extrapolate')

            if variable == 'pressure':
                np.asarray(self.reservoir.mesh.pressure)[:] = values_foo(depths)
            elif variable == 'temperature':
                np.asarray(self.reservoir.mesh.temperature)[:] = values_foo(depths)
            elif variable == 'enthalpy':
                np.asarray(self.reservoir.mesh.enthalpy)[:] = values_foo(depths)
            else:           # compositions
                np.asarray(self.reservoir.mesh.composition)[z_counter::nz_vars] = values_foo(depths)
                z_counter += 1

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
        self.op_list = [self.physics.acc_flux_itor[region] for region in self.physics.regions] + [self.physics.acc_flux_w_itor]
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.op_num[self.reservoir.mesh.n_res_blocks:] = len(self.op_list) - 1

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

    def run_simple(self, physics, params, days):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        days = days if days is not None else self.runtime
        self.physics = physics
        self.params = params
        verbose = False

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0.:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * self.params.mult_ts, self.params.max_ts)
        self.prev_dt = dt

        ts = 0

        while t < stop_time:
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                ts += 1
                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

                dt = min(dt * self.params.mult_ts, self.params.max_ts)

                # if the current dt almost covers the rest time amount needed to reach the stop_time, add the rest
                # to not allow the next time step be smaller than min_ts
                if np.fabs(t + dt - stop_time) < self.params.min_ts:
                    dt = stop_time - t


                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

            else:
                dt /= self.params.mult_ts
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                if dt < self.params.min_ts:
                    break
                    
        # update current engine time
        self.physics.engine.t = stop_time

        if verbose:
            print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                  % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                     self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                     self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))

    def run(self, days: float = None, restart_dt: float = 0., save_well_data : bool = True, save_solution_data : bool = True,
            log_3d_body_path: bool = False, verbose: bool = True):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        :param save_well_data: if True save states of well blocks at every time step to 'well_data.h5', default is True
        :type save_well_data: bool
        :param save_solution_data: if True save states of all reservoir blocks at the end of run to 'solution.h5', default is True
        :type save_solution_data: bool
        :param log_3d_body_path: hypercube output
        :type verbose: bool
        """
        days = days if days is not None else self.runtime
        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
            dt = self.params.first_ts
        elif restart_dt > 0.:
            dt = restart_dt
        else:

            dt = min(self.prev_dt*self.params.mult_ts, days, self.params.max_ts)
        self.prev_dt = dt

        ts = 0

        if log_3d_body_path:
            self.physics.body_path_start(output_folder=self.output_folder)

        while t < stop_time:
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                self.physics.engine.t = t
                ts += 1
                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

                dt = min(dt * self.params.mult_ts, self.params.max_ts)

                # if the current dt almost covers the rest time amount needed to reach the stop_time, add the rest
                # to not allow the next time step be smaller than min_ts
                if np.fabs(t + dt - stop_time) < self.params.min_ts:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

                if log_3d_body_path:
                    self.physics.body_path_add_bodys(output_folder=self.output_folder, time=t)

                if save_well_data:
                    self.output.save_data_to_h5(kind='well')

            else:
                dt /= self.params.mult_ts
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                if dt < self.params.min_ts:
                    print('Stop simulation. Reason: reached min. timestep', self.params.min_ts, 'dt=', dt)
                    return -1

        # update current engine time
        self.physics.engine.t = stop_time

        # save solution vector
        if save_reservoir_data:
            self.output.save_data_to_h5(kind='reservoir')

        if verbose:
            print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                  % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                     self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                     self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))
        return 0

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
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        self.timer.node['simulation'].start()
        for i in range(max_newt+1):
            # self.physics.engine.run_single_newton_iteration(dt)
            self.physics.engine.assemble_linear_system(dt)  # assemble Jacobian and residual of reservoir and well blocks
            self.apply_rhs_flux(dt, t)  # apply RHS flux
            self.physics.engine.newton_residual_last_dt = self.physics.engine.calc_newton_residual()  # calc norm of residual

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j])/max_residual[i] < self.params.stationary_point_tolerance:
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            self.physics.engine.well_residual_last_dt = self.physics.engine.calc_well_residual()
            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.physics.engine.newton_residual_last_dt < self.params.tolerance_newton and
                 self.physics.engine.well_residual_last_dt < self.params.well_tolerance_coefficient * self.params.tolerance_newton) or
                    self.physics.engine.n_newton_last_dt == self.params.max_i_newton):
                if i > 0:  # min_i_newton
                    break
            r_code = self.physics.engine.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.timer.node['simulation'].stop()
        return converged

    def do_after_step(self):
        '''
        can be overrided by an user to be executed in the 'run_simulation()'
        '''
        pass

    def run_simulation(self):
        time = 0.0
        for ith_step, dt in enumerate(self.idata.sim.time_steps):
            self.set_well_controls(time=time)
            ret = self.run(dt)
            if ret != 0:
                print('run() failed for the step=', ith_step, 'dt=', dt)
                return 1
            self.do_after_step()
            time += dt
        return 0

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        """
        Function to specify modifications to RHS vector. User can implement his own boundary conditions here.

        This function is empty in DartsModel, needs to be overloaded in child Model.

        :param t: current time [days]
        :type t: float
        :return: Vector of modification to RHS vector
        :rtype: np.ndarray
        """
        pass

    def apply_rhs_flux(self, dt: float, t: float):
        """
        Function to apply modifications to RHS vector.

        If self.set_rhs_flux() is defined in Model, this function will add its values to rhs

        :param dt: timestep [days]
        :type dt: float
        :param t: current time [days]
        :type t: float
        """
        if type(self).set_rhs_flux is DartsModel.set_rhs_flux:
            # If the function has not been overloaded, pass
            return
        rhs = np.array(self.physics.engine.RHS, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs[:n_res] += self.set_rhs_flux(t) * dt
        return

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

    def init_well_rates(self):
        """
        Function that prepare data for the calculation of well rates (in Python)
        """
        block_m = np.array(self.reservoir.mesh.block_m, copy=False)
        block_p = np.array(self.reservoir.mesh.block_p, copy=False)
        self.well_perf_conn_ids = {}
        self.well_head_conn_id = {}
        for well in self.reservoir.wells:
            res_cell_ids = [perf[1] for perf in well.perforations]

            # find ids of those connections which 1. block_p is in res_cell_ids, 2. block_m is well cell
            conn_ids = np.nonzero(np.logical_and(np.isin(block_p, res_cell_ids), \
                                                 block_m >= self.reservoir.mesh.n_res_blocks))
            self.well_perf_conn_ids[well.name] = conn_ids[0]
            assert (self.well_perf_conn_ids[well.name].size == len(well.perforations) and \
                    (block_m[self.well_perf_conn_ids[well.name]] > self.reservoir.mesh.n_res_blocks).all())
            # find id of well_head -> well_body connection in the connection list
            well_head_conn_id = np.where(np.logical_and(block_m == well.well_head_idx, block_p == well.well_body_idx))[0]
            assert(len(well_head_conn_id) == 1)
            self.well_head_conn_id[well.name] = well_head_conn_id[0]

    def reconstruct_velocities(self):
        # velocity discretization
        values, offset = self.reservoir.discretizer.discretize_velocities(cell_m=np.asarray(self.reservoir.mesh.block_m),
                                                                            cell_p=np.asarray(self.reservoir.mesh.block_p),
                                                                            geom_coef=np.asarray(self.reservoir.mesh.tranD),
                                                                            n_res_blocks=self.reservoir.mesh.n_res_blocks)
        self.reservoir.mesh.velocity_appr.resize(len(values))
        self.reservoir.mesh.velocity_offset.resize(len(offset))

        velocity_appr = np.asarray(self.reservoir.mesh.velocity_appr)
        velocity_appr[:] = values
        velocity_offset = np.asarray(self.reservoir.mesh.velocity_offset)
        velocity_offset[:] = offset

        # specify molar weights to get rid of molar density multiplier in flux terms
        nc = self.physics.nc
        self.physics.engine.molar_weights.resize(nc * len(self.physics.regions))
        molar_weights = np.asarray(self.physics.engine.molar_weights)
        for i, region in enumerate(self.physics.regions):
            molar_weights[i * nc:(i + 1) * nc] = self.physics.property_containers[region].Mw

        # resize storage for velocities inside engine
        self.physics.engine.darcy_velocities.resize(self.reservoir.mesh.n_res_blocks * self.physics.nph * 3)

        # allocate & transfer data to device
        if self.platform == 'gpu':
            from darts.engines import copy_data_to_device, allocate_device_data
            # velocity_appr
            velocity_appr_d = self.physics.engine.get_velocity_appr_d()
            allocate_device_data(self.reservoir.mesh.velocity_appr, velocity_appr_d)
            copy_data_to_device(self.reservoir.mesh.velocity_appr, velocity_appr_d)
            # velocity_offset_d
            velocity_offset_d = self.physics.engine.get_velocity_offset_d()
            allocate_device_data(self.reservoir.mesh.velocity_offset, velocity_offset_d)
            copy_data_to_device(self.reservoir.mesh.velocity_offset, velocity_offset_d)
            # darcy_velocities_d
            darcy_velocities_d = self.physics.engine.get_darcy_velocities_d()
            allocate_device_data(self.physics.engine.darcy_velocities, darcy_velocities_d)
            # molar_weights_d
            molar_weights_d = self.physics.engine.get_molar_weights_d()
            allocate_device_data(self.physics.engine.molar_weights, molar_weights_d)
            copy_data_to_device(self.physics.engine.molar_weights, molar_weights_d)
            # op_num_d
            op_num_d = self.physics.engine.get_op_num_d()
            allocate_device_data(self.reservoir.mesh.op_num, op_num_d)
            copy_data_to_device(self.reservoir.mesh.op_num, op_num_d)

    # destructor to force to destroy all created C objects and free memory
    def __del__(self):
        for name in list(vars(self).keys()):
            delattr(self, name)
