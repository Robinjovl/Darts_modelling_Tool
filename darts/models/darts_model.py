import os
import warnings
from math import fabs

import matplotlib.pyplot as plt

# import h5py
import numpy as np

from darts.models.output import Output

try:
    from darts.engines import copy_data_to_device
except ImportError:
    pass

from darts.discretizer import print_build_info as discretizer_pbi
from darts.engines import (
    ms_well,
    ms_well_vector,
    op_vector,
    sim_params,
    timer_node,
    value_vector,
)
from darts.engines import print_build_info as engines_pbi
from darts.input.input_data import linear_solver_types
from darts.pipes.add_lateral_heat_exchange import SemiAnalyticalWellLateralHeatTransfer
from darts.print_build_info import print_build_info as package_pbi


class DataTS:
    def __init__(self, n_vars):
        self.eta = (
            1e20 * np.ones(n_vars)
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
        self.linear_print_level = None  # linear solver messages printing level (used only for PETSC option), 0 - no messages, 10 - all messages
        #
        self.line_search = False
        self.min_line_search_update = 1e-4

        # For coupled well-reservoir model
        self.coupled_well_res_norm_method = 1

    def print(self):
        print("Simulation parameters:")
        for k in self.__dict__.keys():
            value = self.__getattribute__(k)
            print("\t", k, "=", value)


class DartsModel:
    """
    This is a base class for creating a model in DARTS.
    A model is composed of a :class:`Reservoir` object and a :class:`Physics` object.
    Initialization and communication between these two objects takes place through the Model object

    :ivar reservoir: Reservoir object
    :type reservoir: :class:`ReservoirBase`
    :ivar physics: Physics object
    :type physics: :class:`PhysicsBase`
    :ivar timer: Timer object
    :type timer: :class:`darts.engines.timer_node`
    :ivar params: Object to set simulation parameters
    :type params: :class:`darts.engines.sim_params`
    """

    def __init__(self):
        """
        Initialize DartsModel class.
        """
        # print out build information
        engines_pbi()
        discretizer_pbi()
        package_pbi()

        # Create member variables reservoir and physics
        self.reservoir = None
        self.physics = None

        # Create time_node object for time record
        self.timer = timer_node()

        # Start time record
        self.timer.start()

        # Create timer.node called "simulation" to record simulation time
        self.timer.node["simulation"] = timer_node()

        self.timer.node["newton update"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()
        self.timer.node["output"] = timer_node()

        # Create timer.node called "initialization" to record initialization time
        self.timer.node["initialization"] = timer_node()

        # Start recording "initialization" time
        self.timer.node["initialization"].start()

        # Create sim_params object to set simulation parameters
        self.params = sim_params()

        self.time = []
        self.n_newton_iters = []
        self.time_step_size = []

        # For live plotting
        self.live_plots = False
        self.figs = []
        self.axes = []
        self.lines = []

        # Stop recording "initialization" time
        self.timer.node["initialization"].stop()

    def init(
        self,
        discr_type: str = "tpfa",
        platform: str = "cpu",
        restart: bool = False,
        verbose: bool = False,
        itor_mode: str = "adaptive",
        itor_type: str = "multilinear",
        is_barycentric: bool = False,
        n_solid: int = None,
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
        :param n_solid: Number of solid minerals for element-based reactive flow
        :type n_solid: int
        """
        # Initialize reservoir and Mesh object
        assert self.reservoir is not None, "Reservoir object has not been defined"
        self.reservoir.init_reservoir(verbose)
        self.set_wells()
        self.has_dfm_well = any(
            well.ms_type == ms_well.MS_Type.DFM for well in self.reservoir.wells
        )
        if self.has_dfm_well:
            self.timer.node["simulation"].node["dfm_well_velocity_calculation"] = (
                timer_node()
            )

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
        if platform == "gpu":
            self.params.linear_type = sim_params.gpu_gmres_cpr_amgx_ilu
        self.params.sim_eps = self.physics.sim_eps

        # Initialize well objects
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)

        self.set_op_list()
        self.set_boundary_conditions()
        self.set_well_controls()

        # for separate mineral fraction in reactive flow formulations
        if n_solid is not None:
            self.physics.engine.n_solid = n_solid

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
                "The number of cells looks too big to use a direct linear solver: "
                + str(self.reservoir.mesh.n_res_blocks)
                + ' > 30000',
                stacklevel=2,
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

        :param reservoir_filename: Path to the restart file containing reservoir block data.
        :type reservoir_filename: str
        :param timestep: The timestep to load from the file (default: -1 for the last timestep)
        :type timestep: int
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
        print(rf'Restarting model from {reservoir_filename} at day {time_res[0]}.')
        self.output.save_data_to_h5(kind='reservoir')

        return

    def set_output(
        self,
        output_folder: str = "output",
        sol_filename: str = "reservoir_solution.h5",
        well_filename: str = "well_data.h5",
        save_initial: bool = True,
        all_phase_props: bool = False,
        precision: str = "d",
        compression: str = "gzip",
        compression_level: int = 0,
        verbose: bool = False,
    ):
        """
        Function to initialize output class

        :param output_folder: directory for all output files, images etc.
        :param sol_filename: filename for saving reservoir blocks data.
        :param well_filename: filename for saving well block data.
        :param save_initial: boolean flag to save initial conditions to *.h5, default is True.
        :param all_phase_props: Boolean flag to enable evaluation of all phase properties with property interpolators.
        :param precision: data precision of saved data ('s' single precision, 'd' double precision).
        :param compression: default 'gzip'.
        :param compression_level: 0 (no compression, fast) and 9 (maximum compression, slow), default is 1.
        :param verbose: boolean flag to enable verbose mode.
        """

        self.output_folder = output_folder
        self.sol_filename = sol_filename
        self.well_filename = well_filename
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.well_filepath = os.path.join(self.output_folder, self.well_filename)

        if self.restart:
            save_initial = False

        self.output = Output(
            timer=self.timer,
            reservoir=self.reservoir,
            physics=self.physics,
            op_list=self.op_list,
            params=self.params,
            output_folder=self.output_folder,
            sol_filename=self.sol_filename,
            well_filename=self.well_filename,
            save_initial=save_initial,
            all_phase_props=all_phase_props,
            precision=precision,
            compression=compression,
            compression_level=compression_level,
            verbose=verbose,
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
        raise NotImplementedError("Model.set_initial_conditions() not implemented.")

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
            if k == "eta":
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
        coupled_well_res_norm_method: int = 1,
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
        :param coupled_well_res_norm_method: Method of norm evaluation of residuals for the coupled well-reservoir model
        :type coupled_well_res_norm_method: int
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

        assert coupled_well_res_norm_method in [1, 2], (
            "Method number for calculating the norm of coupled "
            "well-reservoir residuals must be either 1 or 2."
        )
        self.data_ts.coupled_well_res_norm_method = coupled_well_res_norm_method

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
            if (
                type(self.data_ts.linear_type) is not linear_solver_types
            ):  # it's not needed to copy it to params for PETSC option
                self.params.linear_type = self.data_ts.linear_type

    def run_simple(self, physics, data_ts, days, restart_dt=0.0):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param physics:
        :param data_ts:
        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        """
        self.physics = physics
        self.data_ts = data_ts

        days = days if days is not None else self.runtime
        assert days > 0, "Time must be a positive value!"

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
                        f"# {ts:d}\tT = {t:3g}\tDT = {dt:2g}\tNI = {self.physics.engine.n_newton_last_dt:d}\tLI={self.physics.engine.n_linear_last_dt:d}"
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
                    print(f"Cut timestep to {dt:2.10f}")
                if dt < self.data_ts.dt_min:
                    break

        # update current engine time
        self.physics.engine.t = stop_time

        if verbose:
            print(
                f"TS = {self.physics.engine.stat.n_timesteps_total:d}({self.physics.engine.stat.n_timesteps_wasted:d}), NI = {self.physics.engine.stat.n_newton_total:d}({self.physics.engine.stat.n_newton_wasted:d}), LI = {self.physics.engine.stat.n_linear_total:d}({self.physics.engine.stat.n_linear_wasted:d})"
            )

    def run(
        self,
        days: float = None,
        restart_dt: float = 0.0,
        save_well_data: bool = True,
        save_well_data_after_run: bool = True,
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
        :param save_well_data_after_run: Switch to save well data only after runtime of `days`
        :param save_reservoir_data: if True save states of all reservoir blocks at the end of run to 'solution.h5', default is True
        :type save_reservoir_data: bool
        """
        assert hasattr(self, 'output'), (
            "self.output does not exist, please call m.set_output() after m.init()"
        )

        days = days if days is not None else self.runtime
        assert days > 0, "Time must be a positive value!"

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

            save_well_data = False

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15 or not hasattr(self, "prev_dt"):
            dt = min(data_ts.dt_first, days)
        elif restart_dt > 0.0:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * data_ts.dt_mult, days, data_ts.dt_max)

        self.prev_dt = dt

        ts_counter = 0

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
                ts_counter += 1

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
                    max_dx_str = '[' + ', '.join(f'{v:.1e}' for v in max_dx) + ']'
                    print(
                        f"#{ts_counter:d}\tT={t:3g}\tDT={dt:2g}\tNI={self.physics.engine.n_newton_last_dt:d}\tLI={self.physics.engine.n_linear_last_dt:d}\tDT_MULT={dt_mult_new:3.3g}\tdX={max_dx_str}"
                    )

                dt = min(dt * dt_mult_new, data_ts.dt_max)

                if np.fabs(t + dt - stop_time) < data_ts.dt_min:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

                if save_well_data:
                    # save well data at every converged time step
                    self.output.save_data_to_h5(kind="well")

                if save_well_data_after_run:
                    # store well data to save later
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
                if verbose:
                    print(f"Cut timestep to {dt:2.10f}")
                assert dt > data_ts.dt_min, (
                    "Stop simulation. Reason: reached min. timestep "
                    + str(data_ts.dt_min)
                    + " dt="
                    + str(dt)
                )

        # update current engine time
        self.physics.engine.t = stop_time

        # save well data after run
        if save_well_data_after_run:
            path = os.path.join(self.output_folder, self.well_filename)

            self.output.timer.start()
            self.output.timer.node["saving_well_data"].start()
            self.output.save_specific_data(
                path,
                [
                    self.output.well_time_labels,
                    self.output.well_data,
                    self.output.well_cfl,
                ],
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
        assert dt > 0, "Time step size must be a positive value!"

        max_newt = self.data_ts.newton_max_iter
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        self.timer.node["simulation"].start()

        residual_history = []
        for i in range(max_newt + 1):
            # Update well phase velocities and derivatives if DFM wells are used
            if self.has_dfm_well:
                self.update_dfm_well_vels_and_ders(dt, t, i)

            # assemble Jacobian and residual of reservoir and well blocks
            self.physics.engine.assemble_linear_system(dt)

            # apply RHS flux
            self.apply_rhs_flux(dt, t)

            if self.has_dfm_well:
                self.apply_dfm_well_lateral_heat_flux(dt, t)

            if self.platform == "gpu":
                copy_data_to_device(
                    self.physics.engine.RHS, self.physics.engine.get_RHS_d()
                )

            if not self.has_dfm_well:
                self.physics.engine.newton_residual_last_dt = (
                    self.physics.engine.calc_newton_residual()
                )  # calc norm of residual
            # TODO Function line_search is not updated for the coupled model.
            elif self.has_dfm_well:
                # Method is either 1 or 2
                self.physics.engine.newton_residual_last_dt = (
                    self.physics.engine.calc_coupled_well_reservoir_residual(
                        self.data_ts.coupled_well_res_norm_method
                    )
                )

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                if (
                    abs(max_residual[i] - max_residual[j]) / denom
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
                    denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                    if (
                        abs(max_residual[i] - max_residual[j]) / denom
                        < self.data_ts.newton_tol_stationary
                    ):
                        counter += 1
                if counter > 2:
                    if verbose:
                        print("Stationary point detected!")
                    break
            else:
                if (
                    type(self.data_ts.linear_type) is linear_solver_types
                ):  # solvers via Python interface
                    if self.data_ts.linear_type in [
                        linear_solver_types.CPU_PETSC_CPR,
                        linear_solver_types.CPU_PETSC_FS,
                    ]:
                        self.petsc_solve_linear_equation()
                    elif self.data_ts.linear_type in [linear_solver_types.CPU_PARDISO]:
                        self.pardiso_solve_linear_equation()
                    else:
                        raise Exception(
                            "Unknown linear solver type", self.data_ts.linear_type
                        )
                else:  # compile-tyme C++ linear solvers
                    self.physics.engine.solve_linear_equation()
                self.timer.node["newton update"].start()
                self.physics.engine.apply_newton_update(dt)
                self.timer.node["newton update"].stop()
                # Plot live results for every Newton-Raphson iteration
                # if self.live_plots:
                #     self.update_live_plots()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.time.append(t)
        self.n_newton_iters.append(self.physics.engine.n_newton_last_dt)
        self.time_step_size.append(dt)

        # # Plot live results for every time step
        # if self.live_plots:
        #     self.update_live_plots()

        self.timer.node["simulation"].stop()
        return converged

    def init_live_plots(self):
        """
        Initialize figure/axes/artist
        """
        plt.ion()

        """ Start initializing the figure containing axes for solver properties and profiles of wellbore properties """
        fig0, axes0 = plt.subplots(2, 9, figsize=(22, 7), constrained_layout=True)

        self.figs.append(fig0)
        self.axes.append(axes0)

        ax0 = self.axes[0][0, 0]
        ax1 = self.axes[0][1, 0]
        # Well props
        ax2 = self.axes[0][0, 1]
        ax3 = self.axes[0][0, 2]
        ax4 = self.axes[0][0, 3]
        ax5 = self.axes[0][0, 4]
        ax6 = self.axes[0][0, 5]
        ax7 = self.axes[0][0, 6]
        ax8 = self.axes[0][0, 7]
        ax9 = self.axes[0][0, 8]
        # Reservoir props
        ax10 = self.axes[0][1, 1]
        ax11 = self.axes[0][1, 2]
        ax12 = self.axes[0][1, 3]
        ax13 = self.axes[0][1, 4]
        ax14 = self.axes[0][1, 5]
        ax15 = self.axes[0][1, 6]
        ax16 = self.axes[0][1, 7]
        ax17 = self.axes[0][1, 8]

        # Axes for number of Newton iterations
        (line0,) = ax0.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        # ax0.set_xscale("log")
        ax0.set_xlabel("Time [days]")
        ax0.set_ylabel("Number of Newton iterations [-]")
        ax0.set_title("** Solver props **")

        # Axes for time step size
        (line1,) = ax1.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        # ax1.set_xscale("log")
        ax1.set_xlabel("Time [days]")
        ax1.set_ylabel("Time step size [days]")

        # Axes for wellbore pressure
        (line2,) = ax2.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax2.set_xlabel("Pressure [bar]")
        ax2.set_ylabel("Segment index [-]")
        ax2.set_title("** Pressure **")
        ax2.invert_yaxis()

        # Axes for wellbore temperature
        (line3,) = ax3.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax3.set_xlabel(r"Temperature [$^\circ$C]")
        ax3.set_ylabel("Segment index [-]")
        ax3.set_title("** Temperature **")
        ax3.invert_yaxis()

        # Axes for wellbore gas volume fraction
        (line4,) = ax4.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax4.set_xlabel("Gas volume fraction [-]")
        ax4.set_ylabel("Segment index [-]")
        ax4.set_title("** Gas volume fraction **")
        ax4.invert_yaxis()

        # Axes for wellbore liquid volume fraction
        (line5,) = ax5.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax5.set_xlabel("Liquid volume fraction [-]")
        ax5.set_ylabel("Segment index [-]")
        ax5.set_title("** Liquid volume fraction **")
        ax5.invert_yaxis()

        # Axes for wellbore gas density
        (line6,) = ax6.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax6.set_xlabel(r"Gas density [kg/m$^3$]")
        ax6.set_ylabel("Segment index [-]")
        ax6.set_title("** Gas density **")
        ax6.invert_yaxis()

        # Axes for wellbore liquid density
        (line7,) = ax7.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax7.set_xlabel(r"Liquid density [kg/m$^3$]")
        ax7.set_ylabel("Segment index [-]")
        ax7.set_title("** Liquid density **")
        ax7.invert_yaxis()

        # Axes for wellbore gas viscosity
        (line8,) = ax8.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax8.set_xlabel("Gas viscosity [cP]")
        ax8.set_ylabel("Segment index [-]")
        ax8.set_title("** Gas viscosity **")
        ax8.invert_yaxis()

        # Axes for wellbore liquid viscosity
        (line9,) = ax9.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax9.set_xlabel("Liquid viscosity [cP]")
        ax9.set_ylabel("Segment index [-]")
        ax9.set_title("** Liquid viscosity **")
        ax9.invert_yaxis()

        # Axes for reservoir pressure
        (line10,) = ax10.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax10.set_xscale("log")
        ax10.set_xlabel("Reservoir radial distance [m]")
        ax10.set_ylabel("Pressure [bar]")

        # Axes for reservoir temperature
        (line11,) = ax11.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax11.set_xscale("log")
        ax11.set_xlabel("Reservoir radial distance [m]")
        ax11.set_ylabel("Temperature [$^\circ$C]")

        # Axes for reservoir gas volume fraction
        (line12,) = ax12.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax12.set_xscale("log")
        ax12.set_xlabel("Reservoir radial distance [m]")
        ax12.set_ylabel("Gas volume fraction [-]")

        # Axes for reservoir liquid volume fraction
        (line13,) = ax13.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax13.set_xscale("log")
        ax13.set_xlabel("Reservoir radial distance [m]")
        ax13.set_ylabel("Liquid volume fraction [-]")

        # Axes for reservoir gas density
        (line14,) = ax14.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax14.set_xscale("log")
        ax14.set_xlabel("Reservoir radial distance [m]")
        ax14.set_ylabel(r"Gas density [kg/m$^3$]")

        # Axes for reservoir liquid density
        (line15,) = ax15.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax15.set_xscale("log")
        ax15.set_xlabel("Reservoir radial distance [m]")
        ax15.set_ylabel(r"Liquid density [kg/m$^3$]")

        # Axes for reservoir gas viscosity
        (line16,) = ax16.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax16.set_xscale("log")
        ax16.set_xlabel("Reservoir radial distance [m]")
        ax16.set_ylabel("Gas viscosity [cP]")

        # Axes for reservoir liquid viscosity
        (line17,) = ax17.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax17.set_xscale("log")
        ax17.set_xlabel("Reservoir radial distance [m]")
        ax17.set_ylabel("Liquid viscosity [cP]")

        self.lines.append(
            [
                line0,
                line1,
                line2,
                line3,
                line4,
                line5,
                line6,
                line7,
                line8,
                line9,
                line10,
                line11,
                line12,
                line13,
                line14,
                line15,
                line16,
                line17,
            ]
        )

        self.figs[0].show()

        """ Stop initializing the figure containing axes for solver properties and profiles of wellbore properties """

        """ Start initializing the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """
        fig1, axes1 = plt.subplots(figsize=(10, 6), constrained_layout=True)

        self.figs.append(fig1)
        self.axes.append(axes1)

        ax0 = self.axes[1]

        p_bounds = (self.physics.PT_axes_min[0], self.physics.PT_axes_max[0])
        # t_bounds = (self.physics.PT_axes_min[1], self.physics.PT_axes_max[1])
        h_bounds = (self.physics.axes_min[1], self.physics.axes_max[1])

        # Resolution of the PH diagram
        n_p, n_h = self.physics.n_axes_points[0], self.physics.n_axes_points[1]

        p_range = np.linspace(p_bounds[0], p_bounds[1], n_p)
        h_range = np.linspace(h_bounds[0], h_bounds[1], n_h)

        # Calculate the property matrix
        prop_matrix = np.empty((n_p, n_h))
        for idx_p, p in enumerate(p_range):
            for idx_h, h in enumerate(h_range):
                state_ph = [p, h]
                self.physics.property_containers[0].evaluate(state_ph)
                prop_matrix[idx_p, idx_h] = self.physics.property_containers[
                    0
                ].temperature

        prop_matrix = np.where(
            (prop_matrix == 100) | (prop_matrix == 1000), np.nan, prop_matrix
        )  # for temperature
        # prop_matrix = np.where(prop_matrix == 0, np.nan, prop_matrix)   # for density and gas viscosity
        # prop_matrix = np.where((prop_matrix > 4.0) | (prop_matrix == 0), np.nan, prop_matrix)  # for liquid viscosity
        n_cmap_bins = 50
        levels = np.linspace(
            np.nanmin(prop_matrix), np.nanmax(prop_matrix), n_cmap_bins
        )

        # Filled contour (colored areas)
        cax = ax0.contourf(h_range, p_range, prop_matrix, levels=levels, cmap='jet')

        # Contour lines at the same levels
        contours = ax0.contour(
            h_range, p_range, prop_matrix, levels=levels, colors='black', linewidths=0.5
        )

        # Label each contour line with its property value
        ax0.clabel(contours, fmt='%1.1f', inline=True, fontsize=7)

        ax0.set_xlim(h_bounds[0], h_bounds[1])
        ax0.set_ylim(p_bounds[0], p_bounds[1])
        ax0.set_xlabel('Specific enthalpy [kJ/kmole]')
        ax0.set_ylabel('Pressure [bar]')

        # Add a colorbar
        cbar = fig1.colorbar(cax, ax=ax0)
        cbar.set_label("Temperature [K]")

        # Create a plot for adding bottom-hole points (states) to the PH diagram
        (line0,) = ax0.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
            label='Bottom-hole state',
        )

        self.lines.append(line0)

        self.figs[1].show()

        """ Stop initializing the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """

    def update_live_plots(self):
        """
        Plot properties vs current time
        """
        # Initialize once (first call only)
        if not self.figs or not self.axes or not self.lines:
            self.init_live_plots()

        """ Start updating the figure containing axes for solver properties and profiles of wellbore properties """
        self.lines[0][0].set_data(self.time, self.n_newton_iters)
        self.axes[0][0, 0].relim()
        self.axes[0][0, 0].autoscale_view()

        self.lines[0][1].set_data(self.time, self.time_step_size)
        self.axes[0][1, 0].relim()
        self.axes[0][1, 0].autoscale_view()

        i_start_well = self.reservoir.wells[0].well_head_idx
        i_end_well = self.reservoir.wells[0].well_bottom_idx
        p_idx = self.physics.vars.index('pressure')
        h_idx = self.physics.vars.index('enthalpy')
        X_np = np.asarray(self.physics.engine.X).reshape(-1, self.physics.n_vars)
        p_well = X_np[i_start_well : i_end_well + 1, p_idx]
        h_well = X_np[i_start_well : i_end_well + 1, h_idx]
        # Get the property container to evaluate phase props
        pc = self.physics.property_containers[0]
        n_segments = self.wells['I1'].geometry.num_segments
        n_res_blocks = self.reservoir.mesh.n_res_blocks
        n_vars = self.physics.n_vars
        T_well = np.zeros(n_segments)
        for i in range(n_res_blocks, n_res_blocks + n_segments):
            state = np.asarray(self.physics.engine.X)[i * n_vars : (i + 1) * n_vars]
            pc.evaluate(state)
            if self.physics.thermal:
                pc.evaluate_thermal(state)
            T_well[i - n_res_blocks] = pc.temperature - 273.15

        till_this_res_cell = 50  # Plot till this reservoir cell index
        assert till_this_res_cell <= n_res_blocks
        p_res = X_np[:i_start_well, p_idx][:till_this_res_cell]
        x_res = self.reservoir.global_data['dx'].reshape(-1)[:till_this_res_cell]

        # Calculate reservoir phase props
        nc = self.physics.nc
        # Preallocate phase props arrays
        T_res = np.zeros(till_this_res_cell)
        sG_res = np.zeros(till_this_res_cell)
        rhoG_res = np.zeros(till_this_res_cell)
        rhoL_res = np.zeros(till_this_res_cell)
        miuG_res = np.zeros(till_this_res_cell)
        miuL_res = np.zeros(till_this_res_cell)
        xG_mass_res = np.zeros((till_this_res_cell, nc))
        xL_mass_res = np.zeros((till_this_res_cell, nc))

        for i in range(till_this_res_cell):
            state = np.asarray(self.physics.engine.X)[i * n_vars : (i + 1) * n_vars]
            pc.evaluate(state)
            if self.physics.thermal:
                pc.evaluate_thermal(state)
            T_res[i] = pc.temperature - 273.15
            sG_res[i] = pc.sat[0]
            rhoG_res[i] = pc.dens[0]
            rhoL_res[i] = pc.dens[1]
            miuG_res[i] = pc.mu[0]
            miuL_res[i] = pc.mu[1]
            x_mass0 = np.zeros((pc.nph, nc))
            for j in pc.ph:
                x_mass0[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
            xG_mass_res[i, :], xL_mass_res[i, :] = x_mass0[0, :], x_mass0[1, :]

        [
            xG_mass_well,
            xL_mass_well,
            sG_well,
            rhoG_well,
            rhoL_well,
            miuG_well,
            miuL_well,
        ] = self.wells['I1'].iter_phases_props

        # Well props
        self.lines[0][2].set_data(p_well, np.arange(n_segments))
        self.axes[0][0, 1].relim()
        self.axes[0][0, 1].autoscale_view()

        self.lines[0][3].set_data(T_well, np.arange(n_segments))
        self.axes[0][0, 2].relim()
        self.axes[0][0, 2].autoscale_view()

        self.lines[0][4].set_data(sG_well, np.arange(n_segments))
        self.axes[0][0, 3].relim()
        self.axes[0][0, 3].autoscale_view()

        self.lines[0][5].set_data(1 - sG_well, np.arange(n_segments))
        self.axes[0][0, 4].relim()
        self.axes[0][0, 4].autoscale_view()

        self.lines[0][6].set_data(rhoG_well, np.arange(n_segments))
        self.axes[0][0, 5].relim()
        self.axes[0][0, 5].autoscale_view()

        self.lines[0][7].set_data(rhoL_well, np.arange(n_segments))
        self.axes[0][0, 6].relim()
        self.axes[0][0, 6].autoscale_view()

        self.lines[0][8].set_data(miuG_well, np.arange(n_segments))
        self.axes[0][0, 7].relim()
        self.axes[0][0, 7].autoscale_view()

        self.lines[0][9].set_data(miuL_well, np.arange(n_segments))
        self.axes[0][0, 8].relim()
        self.axes[0][0, 8].autoscale_view()

        # Reservoir props
        self.lines[0][10].set_data(x_res, p_res)
        self.axes[0][1, 1].relim()
        self.axes[0][1, 1].autoscale_view()

        self.lines[0][11].set_data(x_res, T_res)
        self.axes[0][1, 2].relim()
        self.axes[0][1, 2].autoscale_view()

        self.lines[0][12].set_data(x_res, sG_res)
        self.axes[0][1, 3].relim()
        self.axes[0][1, 3].autoscale_view()

        self.lines[0][13].set_data(x_res, 1 - sG_res)
        self.axes[0][1, 4].relim()
        self.axes[0][1, 4].autoscale_view()

        self.lines[0][14].set_data(x_res, rhoG_res)
        self.axes[0][1, 5].relim()
        self.axes[0][1, 5].autoscale_view()

        self.lines[0][15].set_data(x_res, rhoL_res)
        self.axes[0][1, 6].relim()
        self.axes[0][1, 6].autoscale_view()

        self.lines[0][16].set_data(x_res, miuG_res)
        self.axes[0][1, 7].relim()
        self.axes[0][1, 7].autoscale_view()

        self.lines[0][17].set_data(x_res, miuL_res)
        self.axes[0][1, 8].relim()
        self.axes[0][1, 8].autoscale_view()

        # Refresh display
        self.figs[0].canvas.draw_idle()
        self.figs[0].canvas.flush_events()

        """ Stop updating the figure containing axes for solver properties and profiles of wellbore properties """

        """ Start updating the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """
        # Plot bottom-hole state
        # Get existing bottom-hole data and append them
        x = list(self.lines[1].get_xdata())
        y = list(self.lines[1].get_ydata())
        p_bottom_hole = p_well[-1]
        enthalpy_bottom_hole = h_well[-1]
        x.append(enthalpy_bottom_hole)
        y.append(p_bottom_hole)

        self.lines[1].set_data(x, y)

        self.axes[1].relim()
        self.axes[1].autoscale_view()

        self.figs[1].canvas.draw_idle()
        self.figs[1].canvas.flush_events()
        """ Stop updating the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """

        # plt.pause(0.5)

    def update_dfm_well_vels_and_ders(self, dt, t, iter_counter):
        """
        Update phase velocities and their corresponding derivatives in DFM wells

        :param dt: Time step size [day]
        :type dt: float
        :param t: Simulation time [day]
        :type t: float
        :param iter_counter: Newton-Raphson iteration counter for the current time step
        :type iter_counter: int
        """
        self.timer.node["simulation"].node["dfm_well_velocity_calculation"].start()
        for w in self.reservoir.wells:
            if w.ms_type == ms_well.MS_Type.DFM:
                start = w.well_head_idx * self.physics.n_vars
                stop = (w.well_head_idx + w.num_segments) * self.physics.n_vars

                Xn_dfm_well = np.asarray(self.physics.engine.Xn)[start:stop]
                X_dfm_well = np.asarray(self.physics.engine.X)[start:stop]
                well_phase_v, well_phase_v_d = self.wells[
                    w.name
                ].eval_phase_vels_and_ders(Xn_dfm_well, X_dfm_well, dt, t, iter_counter)
                w.phases_vels = value_vector(well_phase_v)
                w.phases_vels_ders = value_vector(well_phase_v_d)
        self.timer.node["simulation"].node["dfm_well_velocity_calculation"].stop()

    def apply_dfm_well_lateral_heat_flux(self, dt, t):
        for well in self.reservoir.wells:
            if (
                well.ms_type == ms_well.MS_Type.DFM
                and self.wells[well.name].lateral_heat_rate_eval is not None
            ):
                # Get temperatures of segments
                if self.physics.state_spec == self.physics.StateSpecification.PT:
                    T_segments = self.physics.engine.X[
                        well.well_head_idx * self.physics.n_vars
                        + (self.physics.n_vars - 1) : (
                            well.well_head_idx + well.num_segments
                        )
                        * self.physics.n_vars
                        + (self.physics.n_vars - 1) : self.physics.n_vars
                    ]
                elif self.physics.state_spec == self.physics.StateSpecification.PH:
                    T_segments = np.zeros(well.num_segments)
                    for i in range(well.num_segments):
                        state = self.physics.engine.X[
                            (well.well_head_idx + i) * self.physics.n_vars : (
                                well.well_head_idx + i + 1
                            )
                            * self.physics.n_vars
                        ]
                        self.physics.property_containers[0].evaluate(state)
                        T_segments[i] = self.physics.property_containers[0].temperature

                # Evaluate lateral heat rates and add them to the rhs
                if isinstance(
                    self.wells[well.name].lateral_heat_rate_eval,
                    SemiAnalyticalWellLateralHeatTransfer,
                ):
                    well_lateral_heat_rate = self.wells[
                        well.name
                    ].lateral_heat_rate_eval.evaluate(T_segments, t + dt)
                    rhs = np.array(self.physics.engine.RHS, copy=False)
                    rhs[
                        well.well_head_idx * self.physics.n_vars
                        + (self.physics.n_vars - 1) : (
                            well.well_head_idx + well.num_segments
                        )
                        * self.physics.n_vars
                        + (self.physics.n_vars - 1) : self.physics.n_vars
                    ] -= well_lateral_heat_rate * dt
                else:
                    raise TypeError(
                        f"The provided lateral heat rate evaluator for the well {well.name} is not recognized!"
                    )

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
                "LS: "
                + str(coef[0])
                + "\t"
                + "r_mat = "
                + str(history[0][0])
                + "\tr_well = "
                + str(history[0][1])
            )
            print(
                "LS: "
                + str(coef[1])
                + "\t"
                + "r_mat = "
                + str(history[1][0])
                + "\tr_well = "
                + str(history[1][1])
            )
        res_history = np.array([history[0][0], history[1][0]])

        for _iter in range(5):
            if coef.size > 2:
                idx_min = res_history.argmin()
                closest_left = np.where(coef < coef[idx_min])[0]
                closest_right = np.where(coef > coef[idx_min])[0]
                if closest_left.size and closest_right.size:
                    left = closest_left[coef[closest_left].argmax()]
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[left] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[left]) / 2)
                    elif res_history[right] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[right]) / 2)
                    else:
                        if res_history[left] < res_history[right]:
                            coef = np.append(
                                coef, coef[idx_min] - (coef[idx_min] - coef[left]) / 4
                            )
                        else:
                            coef = np.append(
                                coef, coef[idx_min] + (coef[right] - coef[idx_min]) / 4
                            )
                elif closest_left.size:
                    left = closest_left[coef[closest_left].argmax()]
                    if res_history[left] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[left]) / 2)
                    else:
                        coef = np.append(
                            coef, coef[idx_min] + (coef[idx_min] - coef[left]) / 2
                        )
                elif closest_right.size:
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[right] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[right]) / 2)
                    else:
                        coef = np.append(
                            coef, coef[idx_min] - (coef[right] - coef[idx_min]) / 2
                        )
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
            if self.platform == "gpu":
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
                    "LS: "
                    + str(coef[-1])
                    + "\t"
                    + "r_mat = "
                    + str(res[0])
                    + "\tr_well = "
                    + str(res[1])
                )

        final_id = res_history.argmin()
        self.physics.engine.newton_update_coefficient = coef[final_id] - coef[-1]
        self.timer.node["newton update"].start()
        self.physics.engine.apply_newton_update(dt)
        self.timer.node["newton update"].stop()

        return res_history[final_id], 0.0, coef[final_id]

    def do_after_step(self):
        """
        can be overrided by an user to be executed in the 'run_simulation()'
        """
        pass

    def run_simulation(self):
        time = 0.0
        for ith_step, dt in enumerate(self.idata.sim.time_steps):
            self.set_well_controls_idata(time=time)
            ret = self.run(dt)
            if ret != 0:
                print("run() failed for the step=", ith_step, "dt=", dt)
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
        rhs += self.set_rhs_flux(t) * dt
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
        if self.platform == "gpu":
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
        """
        :param time: simulation time, [days]
        :return:
        """
        from darts.engines import well_control_iface

        # store next control index for each well in idata.well_data.wells_next_control_idx
        if not hasattr(self.idata.well_data, "wells_next_control_idx"):
            self.idata.well_data.wells_next_control_idx = dict()
            for w in self.reservoir.wells:
                self.idata.well_data.wells_next_control_idx[w.name] = 0

        for w in self.reservoir.wells:
            # find next well control in controls list for different timesteps
            wctrl = None
            start_idx = self.idata.well_data.wells_next_control_idx[w.name]
            for wctrl_t in self.idata.well_data.wells[w.name].controls[start_idx:]:
                # if the simulation time passed the well control change time and the control is not already set
                if wctrl_t[0] <= time:
                    wctrl = wctrl_t[1]
                    self.idata.well_data.wells_next_control_idx[w.name] += 1
                    break
            if wctrl is None:  # no control is defined for the current timestep
                continue
            if wctrl.type == "inj":  # INJ well
                inj_temp = wctrl.inj_bht if self.physics.thermal else None
                if wctrl.mode == "rate":  # rate control
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
                elif wctrl.mode == "bhp":  # BHP control
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=well_control_iface.BHP,
                        is_inj=True,
                        target=wctrl.bhp,
                        inj_composition=wctrl.inj_composition,
                        inj_temp=inj_temp,
                    )
                else:
                    print("Unknown well ctrl.mode", wctrl.mode)
                    exit(1)
            elif wctrl.type == "prod":  # PROD well
                if wctrl.mode == "rate":  # rate control
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
                elif wctrl.mode == "bhp":  # BHP control
                    self.physics.set_well_controls(
                        wctrl=w.control,
                        control_type=well_control_iface.BHP,
                        is_inj=False,
                        target=wctrl.bhp,
                    )
                else:
                    print("Unknown well ctrl.mode", wctrl.mode)
                    exit(1)
            else:
                print("Unknown well ctrl.type", wctrl.type)
                exit(1)
            if verbose:
                print(
                    "set_well_controls_idata: time=",
                    time,
                    "well",
                    w.name,
                    "control=[",
                    w.control.get_well_control_type_str(),
                    "],",
                    "constraint=[",
                    w.constraint.get_well_control_type_str(),
                    "]",
                )

        # check
        for w in self.reservoir.wells:
            assert w.control.get_well_control_type() != well_control_iface.NONE, (
                "well control is not initialized for the well " + w.name
            )
            if (
                verbose
                and w.constraint.get_well_control_type() == well_control_iface.NONE
                and "rate" in w.control.get_well_control_type_str()
            ):
                print('A constraint for the well ' + w.name + ' is not initialized!')

    def get_linear_system(self):
        # returns scipy sparse matrix and pointers to RHS and dX
        from scipy.sparse import bsr_matrix

        # get current jacobian and rhs from the engine
        indptr = np.asarray(self.physics.engine.jac_rows)
        indices = np.asarray(self.physics.engine.jac_cols)
        data = np.asarray(self.physics.engine.jac_vals)

        rhs = np.array(self.physics.engine.RHS, copy=False)
        sol = np.array(self.physics.engine.dX, copy=False)

        nonzeros = indices.size

        if nonzeros == 0:
            print(f'linear solver type is {self.data_ts.linear_type}')

        assert nonzeros > 0, (
            'Jacobian is not exposed to python! Probably superlu set as a linear solver!'
        )

        b = int(np.sqrt(data.size / nonzeros))
        data = data.reshape(nonzeros, b, b)

        mat = bsr_matrix((data, indices, indptr))

        mat_csr = mat.tocsr()  # TODO  avoid this conversion to non-blocked matrix

        # print('mat', mat)
        # print('mat_csr', mat_csr)

        return mat_csr, rhs, sol

    def petsc_solve_linear_equation(self):
        print_level = self.data_ts.linear_print_level

        mat, rhs, sol = self.get_linear_system()

        # TODO the variable might be used somewhere, but could not set it here since it was not exposed to python
        # self.physics.engine.linear_solver_error_last_dt = 0

        import petsc4py

        # Petsc Command-Line arguments, there are different ways to pass them as well
        args = ""
        # Monitor residual
        if print_level >= 2:
            args += "-ksp_monitor_short "
        if print_level >= 5:
            args += "-omp_view "  # print number of OpenMP threads
        # Right preconditioner
        args += "-ksp_pc_side right "
        # Iteration limit and tolerance
        args += "-ksp_max_it " + str(self.data_ts.linear_max_iter) + " "
        args += "-ksp_rtol " + str(self.data_ts.linear_tol) + " "

        if self.data_ts.linear_type == linear_solver_types.CPU_PETSC_CPR:
            # Setting up CPR as a composite pc. 1st stage - fieldsplit, 2nd stage - ilu
            args += "-pc_type composite -pc_composite_type multiplicative -pc_composite_pcs fieldsplit,ilu "
            # 1st stage will do AMG on pressure block and "nothing" on transport block
            args += "-sub_0_pc_fieldsplit_type schur -sub_0_pc_fieldsplit_schur_fact_type upper "
            # We build a schur complement diagonal approximation to "decouple" pressure from transport
            args += "-sub_0_pc_fieldsplit_schur_precondition selfp "
            # transport subsolver (for some reason "do nothing" does not work, so do one jacobi iteration)
            args += "-sub_0_fieldsplit_transport_ksp_type preonly "
            args += "-sub_0_fieldsplit_transport_pc_type jacobi "
            # # pressure subsolver (do AMG)
            args += "-sub_0_fieldsplit_pressure_ksp_type preonly "
            args += "-sub_0_fieldsplit_pressure_pc_type gamg "
        elif self.data_ts.linear_type == linear_solver_types.CPU_PETSC_FS:
            # Use U^-1 D^-1 as a preconditioner in block LDU factorization
            args += "-pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper "
            # Use diagonal to approximate S. This should be replaced by the fixed stress approx.
            args += "-pc_fieldsplit_schur_precondition selfp "
            # displacement subsolver
            args += "-fieldsplit_displacement_ksp_type preonly "
            args += "-fieldsplit_displacement_pc_type gamg "
            # pressure subsolver
            args += "-fieldsplit_pressure_ksp_type preonly "
            args += "-fieldsplit_pressure_pc_type gamg "
        else:
            raise AssertionError('Unknown linear solver type for PETSC')

        petsc4py.init(args)
        # Important to import PETSc after petsc4py.init
        from petsc4py import PETSc

        # Create matrix
        petsc_mat = PETSc.Mat().createAIJ(
            size=mat.shape, csr=(mat.indptr, mat.indices, mat.data)
        )
        petsc_mat.setFromOptions()
        petsc_mat.setUp()

        # Create rhs
        petsc_rhs = PETSc.Vec().createWithArray(rhs, rhs.size)
        petsc_rhs.setFromOptions()
        petsc_rhs.setUp()

        # Create sol
        petsc_sol = PETSc.Vec().createWithArray(sol, sol.size)
        petsc_sol.setFromOptions()
        petsc_sol.setUp()

        # Create petsc linear solver
        petsc_ksp = PETSc.KSP().create()
        petsc_ksp.setFromOptions()
        petsc_ksp.setOperators(petsc_mat, petsc_mat)

        # Inform petsc about our fields
        if self.data_ts.linear_type == linear_solver_types.CPU_PETSC_CPR:
            pressure_idx = np.arange(0, mat.shape[0], 2)
            transport_idx = np.arange(1, mat.shape[0], 2)
            petsc_is_pressure = PETSc.IS().createGeneral(pressure_idx.astype("int32"))
            petsc_is_transport = PETSc.IS().createGeneral(transport_idx.astype("int32"))

            # Getting Composite PC
            petsc_pc = petsc_ksp.getPC()
            petsc_pc.setUp()
            # Getting the 1st stage (fieldsplit)
            petsc_pc_1st_stage = petsc_pc.getCompositePC(0)
            petsc_pc_1st_stage.setFieldSplitIS(
                ("transport", petsc_is_transport), ("pressure", petsc_is_pressure)
            )

            # Here, AMG setup happens
            petsc_pc_1st_stage.setOperators(petsc_mat, petsc_mat)
            petsc_pc_1st_stage.setUp()
            # ILU setup
            petsc_pc_2nd_stage = petsc_pc.getCompositePC(1)
            petsc_pc_2nd_stage.setUp()
        elif self.data_ts.linear_type == linear_solver_types.CPU_PETSC_FS:
            pressure_idx = np.arange(0, mat.shape[0], 4)
            displacement_idx = np.stack(
                [
                    np.arange(1, mat.shape[0], 4),
                    np.arange(2, mat.shape[0], 4),
                    np.arange(3, mat.shape[0], 4),
                ]
            ).ravel(order="F")
            petsc_is_pressure = PETSc.IS().createGeneral(pressure_idx.astype("int32"))
            petsc_is_displacement = PETSc.IS().createGeneral(
                displacement_idx.astype("int32")
            )
            # Displacement is a vector problem
            petsc_is_displacement.setBlockSize(3)

            # Setting fieldsplit fields
            petsc_pc = petsc_ksp.getPC()
            petsc_pc.setFromOptions()
            petsc_pc.setFieldSplitIS(
                ("displacement", petsc_is_displacement), ("pressure", petsc_is_pressure)
            )

        petsc_ksp.setUp()

        # This prints the solver information to stdout
        if print_level >= 4:
            petsc_ksp.view()

        if print_level >= 1:
            print("PETSC: start solving")

        petsc_ksp.solve(petsc_rhs, petsc_sol)

        if print_level >= 1:
            print('PETSC: True residual =', np.linalg.norm(mat.dot(sol) - rhs))

        # TODO check when solver fails https://petsc.org/main/petsc4py/reference/petsc4py.PETSc.KSP.html#petsc4py.PETSc.KSP.solve

    def pardiso_solve_linear_equation(self):
        import pypardiso

        mat, rhs, sol = self.get_linear_system()
        sol[:] = pypardiso.spsolve(mat, rhs)
