import os
import pickle
import warnings
from math import fabs

import numpy as np
from scipy.interpolate import interp1d

from darts.models.output import Output
from darts.physics.base.physics_base import PhysicsBase
from darts.reservoirs.reservoir_base import ReservoirBase

try:
    from darts.engines import copy_data_to_device
except ImportError:
    pass

from darts.discretizer import print_build_info as discretizer_pbi
from darts.engines import (
    index_vector,
    ms_well_vector,
    op_vector,
)
from darts.engines import print_build_info as engines_pbi
from darts.engines import (
    sim_params,
    timer_node,
    value_vector,
)
from darts.print_build_info import print_build_info as package_pbi


class DataTS:

    def __init__(self, n_vars):
        self.eta = 1e20 * np.ones(
            n_vars
        )  # controls the timestep by the variable change from the previous newton iteration
        # dX = Xn - X . Eta has a size of number of DOFs per cell. It set to a large value by default, so doesn't affect the timestep choice

        # default values
        self.dt_first = 1.0  # initial timestep [days]
        self.dt_min = 1e-12  # minimal allowed timestep [days]
        self.dt_mult = 2.0  # timestep multiplier, affects the next timestep choice
        self.dt_max = 10.0  # maximal allowed timestep [days]
        self.newton_tol = 1e-2  # newton solver residual
        self.newton_tol_wel_mult = 100.0  # used to compute the newton solver residual for wells = tol_res * tol_wel_mult
        self.newton_tol_stationary = 1e-3  # tolerance for stationary point detection in the newton solver (by residual)
        self.newton_max_iter = 20  # maximum newton iterations allowed
        self.linear_tol = 1e-5
        self.linear_max_iter = 50  # maximum linear iterations allowed
        self.linear_type = None  # linear solver and preconditioner type
        #
        self.line_search = False
        self.min_line_search_update = 1e-4

    def print(self):
        print('Simulation parameters:')
        for k in self.__dict__.keys():
            value = self.__getattribute__(k)
            print('\t', k, '=', value)


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
        self.timer.node["simulation"] = (
            timer_node()
        )  # Create timer.node called "simulation" to record simulation time
        self.timer.node["newton update"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()
        self.timer.node["output"] = timer_node()
        self.timer.node["initialization"] = (
            timer_node()
        )  # Create timer.node called "initialization" to record initialization time
        self.timer.node[
            "initialization"
        ].start()  # Start recording "initialization" time

        self.params = (
            sim_params()
        )  # Create sim_params object to set simulation parameters

        self.timer.node["initialization"].stop()  # Stop recording "initialization" time

    def init(
        self,
        discr_type: str = 'tpfa',
        platform: str = 'cpu',
        restart: bool = False,
        verbose: bool = False,
        itor_mode: str = 'adaptive',
        itor_type: str = 'multilinear',
        is_barycentric: bool = False,
    ):
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
        self.platform = platform
        self.physics.init_physics(
            discr_type=discr_type,
            platform=platform,
            verbose=verbose,
            itor_mode=itor_mode,
            itor_type=itor_type,
            is_barycentric=is_barycentric,
        )
        if platform == 'gpu':
            self.params.linear_type = sim_params.gpu_gmres_cpr_amgx_ilu

        # Initialize well objects
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.init_well_rates()

        self.set_op_list()
        self.set_boundary_conditions()
        self.set_well_controls()

        # when restarting the initial conditions are set in self.load_restart_data() and the engine is reset.
        self.restart = restart
        if restart is False:
            self.set_initial_conditions()
            self.reset()
        self.data_ts.print()
        if (
            self.params.linear_type == sim_params.linear_solver_t.cpu_superlu
            and self.reservoir.mesh.n_res_blocks > 30000
        ):
            warnings.warn(
                'The number of cells looks too big to use a direct linear solver: '
                + str(self.reservoir.mesh.n_res_blocks)
                + ' > 30000'
            )

    def reset(self):
        """
        Function to initialize the engine by calling 'engine.init()' method.
        """
        self.physics.engine.init(
            self.reservoir.mesh,
            ms_well_vector(self.reservoir.wells),
            op_vector(self.op_list),
            self.params,
            self.timer.node["simulation"],
        )

    def load_restart_data(self, reservoir_filename: str, timestep: int = -1):
        """
        Loads data from a previous simulation and sets it for the current simulation.
        Beware that loading restart data resets the engine.

        :param reservoir_filename (str): Path to the restart file containing reservoir block data.
        :param well_filename (str): Path to the restart file containing well block data.
        :param timestep (int): The timestep to load from the file (default: -1 for the last timestep).
        """

        # check if the files with data exist
        if not os.path.exists(
            reservoir_filename
        ):  # or not os.path.exists(well_filename):
            raise FileNotFoundError(
                f"The restart file does not exist: {reservoir_filename}"
            )

        # Read data from the file
        time_res, reservoir_cell_id, Xres, var_names = self.output.read_specific_data(
            reservoir_filename, timestep
        )

        # load data as initial conditions
        initial_values = {}
        for i, name in enumerate(var_names):
            initial_values[name] = Xres[:, :, i].flatten()
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=initial_values
        )

        self.reset()
        self.physics.engine.t = time_res[0]

        # save initial conditions to *.h5 file
        print(fr'Restarting model from {reservoir_filename} at day {time_res[0]}.')
        self.output.save_data_to_h5(kind='reservoir')

        return

    def set_output(
        self,
        output_folder: str = 'output',
        sol_filename: str = 'reservoir_solution.h5',
        well_filename: str = 'well_data.h5',
        save_initial: bool = True,
        all_phase_props: bool = False,
        precision: str = 'd',
        compression: str = 'gzip',
        verbose: bool = False,
    ):
        """
        Function to initialize output class

        : param output_folder: folder for h5 output files
        : param sol_filename: filename of output file
        : param save_inital:
        : param all_phase_props: Boolean to output all phase properties
        : param precision: data precision of saved data ('s' single precision, 'd' double precision)
        : param compression: default 'gzip'
        : param verbose:
        """

        self.output_folder = output_folder
        self.sol_filename = sol_filename
        self.well_filename = well_filename
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        if self.restart:
            save_initial = False

        self.output = Output(
            self.timer,
            self.reservoir,
            self.physics,
            self.op_list,
            self.params,
            self.well_head_conn_id,
            self.well_perf_conn_ids,
            self.output_folder,
            self.sol_filename,
            self.well_filename,
            save_initial,
            all_phase_props,
            precision,
            compression,
            verbose,
        )

        return

    def set_wells(self, verbose: bool = False):
        """
        Function to define wells. The default method of DartsModel.set_wells() calls Reservoir.set_wells().

        :param verbose: Switch for verbose
        :type verbose: bool
        """
        self.reservoir.set_wells(verbose)
        return

    def set_initial_conditions(self):
        """
        Function to set initial conditions. Passes initial conditions to :class:`Mesh` object.

        Initial conditions can be specified in multiple ways:
        1) Uniform or array -> specify constant or array of values for each variable to self.physics.set_initial_conditions_by_array()
        2) Depth table -> specify depth table with depths and initial distributions of unknowns over depth
                          to self.physics.set_initial_conditions_by_depth_table()
        """
        raise NotImplementedError('Model.set_initial_conditions() not implemented.')

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
        self.op_list = [
            self.physics.acc_flux_itor[region] for region in self.physics.regions
        ] + [self.physics.acc_flux_w_itor]
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.op_num[self.reservoir.mesh.n_res_blocks :] = len(self.op_list) - 1

    def set_sim_params_data_ts(self, data_ts):
        self.data_ts = DataTS(self.physics.n_vars)
        # copy attributes except eta
        for k in data_ts.__dict__.keys():
            if k == 'eta':
                continue
            value = data_ts.__getattribute__(k)
            self.data_ts.__setattr__(k, value)
        self.copy_data_ts_to_sim_params()

    def set_sim_params(
        self,
        first_ts: float = None,
        mult_ts: float = None,
        min_ts=1e-15,
        max_ts: float = None,
        runtime: float = 1000,
        tol_newton: float = None,
        tol_linear: float = None,
        it_newton: int = None,
        it_linear: int = None,
        newton_type=None,
        newton_params=None,
        line_search: bool = False,
    ):
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
        self.data_ts = DataTS(self.physics.n_vars)

        # Time stepping parameters. if None, default value will be used
        self.data_ts.dt_first = (
            first_ts if first_ts is not None else self.data_ts.dt_first
        )
        self.data_ts.dt_min = min_ts if min_ts is not None else self.data_ts.dt_min
        self.data_ts.dt_max = max_ts if max_ts is not None else self.data_ts.dt_max
        self.data_ts.dt_mult = mult_ts if mult_ts is not None else self.data_ts.dt_mult

        # Non linear solver parameters. if None, default value will be used
        self.data_ts.newton_max_iter = (
            it_newton if it_newton is not None else self.data_ts.newton_max_iter
        )
        self.data_ts.newton_tol = (
            tol_newton if tol_newton is not None else self.data_ts.newton_tol
        )

        self.params.newton_type = (
            newton_type if newton_type is not None else self.params.newton_type
        )
        self.params.newton_params = (
            newton_params if newton_params is not None else self.params.newton_params
        )

        self.data_ts.line_search = line_search

        # Linear solver parameters. if None, default value will be used
        self.data_ts.linear_tol = (
            tol_linear if tol_linear is not None else self.data_ts.linear_tol
        )
        self.data_ts.linear_max_iter = (
            it_linear if it_linear is not None else self.data_ts.linear_max_iter
        )

        self.runtime = runtime

        self.copy_data_ts_to_sim_params()

    def copy_data_ts_to_sim_params(self):
        self.params.first_ts = self.data_ts.dt_first
        self.params.max_ts = self.data_ts.dt_max
        self.params.mult_ts = self.data_ts.dt_mult
        self.params.tolerance_newton = self.data_ts.newton_tol
        self.params.max_i_newton = self.data_ts.newton_max_iter
        self.params.tolerance_linear = self.data_ts.linear_tol
        self.params.max_i_linear = self.data_ts.linear_max_iter
        if self.data_ts.linear_type is not None:
            self.params.linear_type = self.data_ts.linear_type

    def run_simple(self, physics, data_ts, days):
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
        self.data_ts = data_ts
        verbose = False

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15:
            dt = self.data_ts.dt_first
        elif restart_dt > 0.0:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * self.data_ts.dt_mult, self.data_ts.dt_max)
        self.prev_dt = dt

        ts = 0

        while t < stop_time:
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                ts += 1
                if verbose:
                    print(
                        "# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                        % (
                            ts,
                            t,
                            dt,
                            self.physics.engine.n_newton_last_dt,
                            self.physics.engine.n_linear_last_dt,
                        )
                    )

                dt = min(dt * self.data_ts.dt_mult, self.data_ts.dt_max)

                # if the current dt almost covers the rest time amount needed to reach the stop_time, add the rest
                # to not allow the next time step be smaller than min_ts
                if np.fabs(t + dt - stop_time) < self.data_ts.dt_min:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

            else:
                dt /= self.data_ts.dt_mult
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                if dt < self.data_ts.dt_min:
                    break

        # update current engine time
        self.physics.engine.t = stop_time

        if verbose:
            print(
                "TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                % (
                    self.physics.engine.stat.n_timesteps_total,
                    self.physics.engine.stat.n_timesteps_wasted,
                    self.physics.engine.stat.n_newton_total,
                    self.physics.engine.stat.n_newton_wasted,
                    self.physics.engine.stat.n_linear_total,
                    self.physics.engine.stat.n_linear_wasted,
                )
            )

    def run(
        self,
        days: float = None,
        restart_dt: float = 0.0,
        save_well_data: bool = True,
        save_well_data_after_run: bool = False,
        save_reservoir_data: bool = True,
        verbose: bool = True,
    ):
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
        """
        assert hasattr(
            self, 'output'
        ), "self.output does not exist, please call m.set_output() after m.init()"
        days = days if days is not None else self.runtime
        data_ts = self.data_ts

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
            dt = data_ts.dt_first
        elif restart_dt > 0.0:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * data_ts.dt_mult, days, data_ts.dt_max)

        self.prev_dt = dt

        ts = 0

        nc = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        max_dx = np.zeros(nc)

        if np.fabs(data_ts.dt_mult - 1) < 1e-10:
            omega = 0.0
        else:
            omega = 1 / (
                data_ts.dt_mult - 1
            )  # inversion assuming mult = (1 + omega) / omega

        while t < stop_time:
            xn = np.array(self.physics.engine.Xn, copy=True)[
                : nb * nc
            ]  # need to copy since Xn will be updated Xn = X
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                self.physics.engine.t = t
                ts += 1

                x = np.array(self.physics.engine.X, copy=False)[: nb * nc]
                dt_mult_new = data_ts.dt_mult
                for i in range(nc):
                    max_dx[i] = np.max(abs(xn[i::nc] - x[i::nc]))
                    mult = ((1 + omega) * data_ts.eta[i]) / (
                        max_dx[i] + omega * data_ts.eta[i]
                    )
                    if mult < dt_mult_new:
                        dt_mult_new = mult

                if verbose:
                    print(
                        "# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d\tDT_MULT=%3.3g\tdX=%4s"
                        % (
                            ts,
                            t,
                            dt,
                            self.physics.engine.n_newton_last_dt,
                            self.physics.engine.n_linear_last_dt,
                            dt_mult_new,
                            np.round(max_dx, 3),
                        )
                    )

                dt = min(dt * dt_mult_new, data_ts.dt_max)

                if np.fabs(t + dt - stop_time) < data_ts.dt_min:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

                # save well data at every converged time step
                if save_well_data and save_well_data_after_run is False:
                    self.output.save_data_to_h5(kind='well')

            else:
                dt /= data_ts.dt_mult
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                assert dt > data_ts.dt_min, (
                    'Stop simulation. Reason: reached min. timestep '
                    + str(data_ts.dt_min)
                    + ' dt='
                    + str(dt)
                )

        # update current engine time
        self.physics.engine.t = stop_time

        # save well data after run
        if save_well_data and save_well_data_after_run is True:
            self.output.save_data_to_h5(kind='well')

        # save solution vector
        if save_reservoir_data:
            self.output.save_data_to_h5(kind='reservoir')

        if verbose:
            print(
                "TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                % (
                    self.physics.engine.stat.n_timesteps_total,
                    self.physics.engine.stat.n_timesteps_wasted,
                    self.physics.engine.stat.n_newton_total,
                    self.physics.engine.stat.n_newton_wasted,
                    self.physics.engine.stat.n_linear_total,
                    self.physics.engine.stat.n_linear_wasted,
                )
            )

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
        max_newt = self.data_ts.newton_max_iter
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        self.timer.node['simulation'].start()
        residual_history = []
        for i in range(max_newt + 1):
            self.physics.engine.assemble_linear_system(
                dt
            )  # assemble Jacobian and residual of reservoir and well blocks
            self.apply_rhs_flux(dt, t)  # apply RHS flux
            if self.platform == 'gpu':
                copy_data_to_device(
                    self.physics.engine.RHS, self.physics.engine.get_RHS_d()
                )

            self.physics.engine.newton_residual_last_dt = (
                self.physics.engine.calc_newton_residual()
            )  # calc norm of residual

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if (
                    abs(max_residual[i] - max_residual[j]) / max_residual[i]
                    < self.data_ts.newton_tol_stationary
                ):
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            self.physics.engine.well_residual_last_dt = (
                self.physics.engine.calc_well_residual()
            )
            residual_history.append(
                (
                    self.physics.engine.newton_residual_last_dt,  # matrix residual
                    self.physics.engine.well_residual_last_dt,  # well residual
                    1.0,
                )
            )  # Newton update coefficient

            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if (
                self.physics.engine.newton_residual_last_dt < self.data_ts.newton_tol
                and self.physics.engine.well_residual_last_dt
                < self.data_ts.newton_tol * self.data_ts.newton_tol_wel_mult
            ) or self.physics.engine.n_newton_last_dt == max_newt:
                if i > 0:  # min_i_newton
                    break

            # line search
            if (
                self.data_ts.line_search
                and i > 0
                and residual_history[-1][0] > 0.9 * residual_history[-2][0]
            ):
                coef = np.array([0.0, 1.0])
                history = np.array([residual_history[-2], residual_history[-1]])
                residual_history[-1] = self.line_search(dt, t, coef, history, verbose)
                max_residual[i] = residual_history[-1][0]

                # check stationary point after line search
                counter = 0
                for j in range(i):
                    if (
                        abs(max_residual[i] - max_residual[j]) / max_residual[i]
                        < self.data_ts.newton_tol_stationary
                    ):
                        counter += 1
                if counter > 2:
                    if verbose:
                        print("Stationary point detected!")
                    break
            else:
                r_code = self.physics.engine.solve_linear_equation()
                self.timer.node["newton update"].start()
                self.physics.engine.apply_newton_update(dt)
                self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.timer.node['simulation'].stop()
        return converged

    def line_search(self, dt, t, coef, history, verbose: bool = False):
        """
        Performs a line search to find the optimal coefficient that minimizes residuals.

        :param dt: Time step for the update process.
        :type dt: float
        :param t: Current time.
        :type t: float
        :param coef: Array of current coefficients used in the line search.
        :type coef: numpy.ndarray
        :param history: Historical residuals, where each entry contains residuals for 'r_mat' and 'r_well'.
        :type history: list or numpy.ndarray
        :param verbose: If True, prints detailed debug information during execution.
        :type verbose: bool
        :return: Tuple containing the minimum residual achieved, a placeholder value (0.0), and the coefficient corresponding to the minimum residual.
        :rtype: tuple(float, float, float)
        """

        if verbose:
            print(
                'LS: '
                + str(coef[0])
                + '\t'
                + 'r_mat = '
                + str(history[0][0])
                + '\tr_well = '
                + str(history[0][1])
            )
            print(
                'LS: '
                + str(coef[1])
                + '\t'
                + 'r_mat = '
                + str(history[1][0])
                + '\tr_well = '
                + str(history[1][1])
            )
        res_history = np.array([history[0][0], history[1][0]])

        for iter in range(5):
            if coef.size > 2:
                id = res_history.argmin()
                closest_left = np.where(coef < coef[id])[0]
                closest_right = np.where(coef > coef[id])[0]
                if closest_left.size and closest_right.size:
                    left = closest_left[coef[closest_left].argmax()]
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[left] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[left]) / 2)
                    elif res_history[right] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[right]) / 2)
                    else:
                        if res_history[left] < res_history[right]:
                            coef = np.append(
                                coef, coef[id] - (coef[id] - coef[left]) / 4
                            )
                        else:
                            coef = np.append(
                                coef, coef[id] + (coef[right] - coef[id]) / 4
                            )
                elif closest_left.size:
                    left = closest_left[coef[closest_left].argmax()]
                    if res_history[left] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[left]) / 2)
                    else:
                        coef = np.append(coef, coef[id] + (coef[id] - coef[left]) / 2)
                elif closest_right.size:
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[right] < res_history[id]:
                        coef = np.append(coef, (coef[id] + coef[right]) / 2)
                    else:
                        coef = np.append(coef, coef[id] - (coef[right] - coef[id]) / 2)
                if coef[-1] <= 0:
                    coef[-1] = self.data_ts.min_line_search_update
                if coef[-1] >= 1:
                    coef[-1] = 1.0 - self.data_ts.min_line_search_update
            else:
                coef = np.append(coef, coef[-1] / 2)

            self.physics.engine.newton_update_coefficient = coef[-1] - coef[-2]
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
            self.physics.engine.assemble_linear_system(dt)
            self.apply_rhs_flux(dt, t)
            if self.platform == 'gpu':
                copy_data_to_device(
                    self.physics.engine.RHS, self.physics.engine.get_RHS_d()
                )
            res = (
                self.physics.engine.calc_newton_residual(),
                self.physics.engine.calc_well_residual(),
            )
            res_history = np.append(res_history, res[0])
            if verbose:
                print(
                    'LS: '
                    + str(coef[-1])
                    + '\t'
                    + 'r_mat = '
                    + str(res[0])
                    + '\tr_well = '
                    + str(res[1])
                )

        final_id = res_history.argmin()
        self.physics.engine.newton_update_coefficient = coef[final_id] - coef[-1]
        self.timer.node["newton update"].start()
        self.physics.engine.apply_newton_update(dt)
        self.timer.node["newton update"].stop()

        return res_history[final_id], 0.0, coef[final_id]

    def do_after_step(self):
        '''
        can be overrided by an user to be executed in the 'run_simulation()'
        '''
        pass

    def run_simulation(self):
        time = 0.0
        for ith_step, dt in enumerate(self.idata.sim.time_steps):
            self.set_well_controls_idata(time=time)
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
            conn_ids = np.nonzero(
                np.logical_and(
                    np.isin(block_p, res_cell_ids),
                    block_m >= self.reservoir.mesh.n_res_blocks,
                )
            )
            self.well_perf_conn_ids[well.name] = conn_ids[0]
            assert (
                self.well_perf_conn_ids[well.name].size == len(well.perforations)
                and (
                    block_m[self.well_perf_conn_ids[well.name]]
                    > self.reservoir.mesh.n_res_blocks
                ).all()
            )
            # find id of well_head -> well_body connection in the connection list
            well_head_conn_id = np.where(
                np.logical_and(
                    block_m == well.well_head_idx, block_p == well.well_body_idx
                )
            )[0]
            assert len(well_head_conn_id) == 1
            self.well_head_conn_id[well.name] = well_head_conn_id[0]

    def reconstruct_velocities(self):
        # velocity discretization
        values, offset = self.reservoir.discretizer.discretize_velocities(
            cell_m=np.asarray(self.reservoir.mesh.block_m),
            cell_p=np.asarray(self.reservoir.mesh.block_p),
            geom_coef=np.asarray(self.reservoir.mesh.tranD),
            n_res_blocks=self.reservoir.mesh.n_res_blocks,
        )
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
            molar_weights[i * nc : (i + 1) * nc] = self.physics.property_containers[
                region
            ].Mw

        # resize storage for velocities inside engine
        self.physics.engine.darcy_velocities.resize(
            self.reservoir.mesh.n_res_blocks * self.physics.nph * 3
        )

        # allocate & transfer data to device
        if self.platform == 'gpu':
            from darts.engines import allocate_device_data, copy_data_to_device

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
            allocate_device_data(
                self.physics.engine.darcy_velocities, darcy_velocities_d
            )
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

    def set_well_controls_idata(self, time: float = 0.0, verbose=True):
        '''
        :param time: simulation time, [days]
        :return:
        '''
        from darts.engines import well_control_iface

        eps_time = 1e-15  # threshold between the current time and the time for the well control
        for w in self.reservoir.wells:
            # find next well control in controls list for different timesteps
            wctrl = None
            for wctrl_t in self.idata.well_data.wells[w.name].controls:
                if np.fabs(wctrl_t[0] - time) < eps_time:  # check time
                    wctrl = wctrl_t[1]
                    break
            if wctrl is None:  # no control is defined for the current timestep
                continue
            if wctrl.type == 'inj':  # INJ well
                inj_temp = wctrl.inj_bht if self.physics.thermal else None
                if wctrl.mode == 'rate':  # rate control
                    # Control
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=wctrl.rate_type,
                        is_inj=True,
                        target=wctrl.rate,
                        phase_name=wctrl.phase_name,
                        inj_composition=wctrl.inj_composition,
                        inj_temp=inj_temp,
                    )
                    # Constraint
                    if wctrl.bhp_constraint is not None:
                        self.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=wctrl.bhp_constraint,
                            inj_composition=wctrl.inj_composition,
                            inj_temp=inj_temp,
                        )
                elif wctrl.mode == 'bhp':  # BHP control
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=well_control_iface.BHP,
                        is_inj=True,
                        target=wctrl.bhp,
                        inj_composition=wctrl.inj_composition,
                        inj_temp=inj_temp,
                    )
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            elif wctrl.type == 'prod':  # PROD well
                if wctrl.mode == 'rate':  # rate control
                    # Control
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=wctrl.rate_type,
                        is_inj=False,
                        target=-np.abs(wctrl.rate),
                        phase_name=wctrl.phase_name,
                    )
                    # Constraint
                    if wctrl.bhp_constraint is not None:
                        self.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=False,
                            target=wctrl.bhp_constraint,
                        )
                elif wctrl.mode == 'bhp':  # BHP control
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=well_control_iface.BHP,
                        is_inj=False,
                        target=wctrl.bhp,
                    )
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            else:
                print('Unknown well ctrl.type', wctrl.type)
                exit(1)
            if verbose:
                print(
                    'set_well_controls_idata: time=',
                    time,
                    'well',
                    w.name,
                    'control=[',
                    w.control.get_well_control_type_str(),
                    '],',
                    'constraint=[',
                    w.constraint.get_well_control_type_str(),
                    ']',
                )

        # check
        for w in self.reservoir.wells:
            assert w.control.get_well_control_type() != well_control_iface.NONE, (
                'well control is not initialized for the well ' + w.name
            )
            if (
                verbose
                and w.constraint.get_well_control_type() == well_control_iface.NONE
                and 'rate' in w.control.get_well_control_type_str()
            ):
                print('A constraint for the well ' + w.name + ' is not initialized!')
