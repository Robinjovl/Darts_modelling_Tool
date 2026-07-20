import os
import warnings
from math import fabs

import numpy as np

from darts.discretizer import print_build_info as discretizer_pbi
from darts.engines import (
    ms_well,
    ms_well_vector,
    sim_params,
    timer_node,
    value_vector,
)
from darts.engines import print_build_info as engines_pbi
from darts.input.input_data import linear_solver_types
from darts.interpolators import op_vector
from darts.models.output import Output
from darts.nonlinear_solvers import default_nonlinear_solver
from darts.pipes.add_lateral_heat_exchange import SemiAnalyticalWellLateralHeatTransfer
from darts.print_build_info import print_build_info as package_pbi


class DataTS:
    """Timestep-control (and, transitionally, linear-solver) parameters.

    Holds ONLY the timestep controls (``dt_first``/``dt_min``/``dt_mult``/
    ``dt_max``/``eta``) and the linear-solver settings (``linear_*``, plain
    attributes until the linear-solver spec branch (MR280) is merged). The
    nonlinear-solver settings are NOT mirrored here — they live at the single
    source of truth ``DartsModel.nonlinear_solver.spec`` (a
    :class:`darts.nonlinear_solvers.NonlinearSolverSpec`).
    """

    _FIELDS = (
        "eta",
        "dt_first",
        "dt_min",
        "dt_mult",
        "dt_max",
        "linear_tol",
        "linear_max_iter",
        "linear_type",
        "linear_print_level",
    )

    def __init__(self, n_vars):
        # timestep control (owned by this structure)
        self.eta = (
            1e20 * np.ones(n_vars)
        )  # controls the timestep by the variable change from the previous newton iteration
        # dX = Xn - X. Eta has a size of number of DOFs per cell. Set to a large value by default, so doesn't affect the timestep choice
        self.dt_first = 1.0  # initial timestep [days]
        self.dt_min = 1e-12  # minimal allowed timestep [days]
        self.dt_mult = 2.0  # timestep multiplier, affects the next timestep choice
        self.dt_max = 10.0  # maximal allowed timestep [days]

        # linear solver settings (plain attributes until MR280 merge)
        self.linear_tol = 1e-5
        self.linear_max_iter = 50  # maximum linear iterations allowed
        self.linear_type = None  # linear solver and preconditioner type
        self.linear_print_level = None  # linear solver messages printing level (used only for PETSC option), 0 - no messages, 10 - all messages

    def print(self):
        print("Simulation parameters:")
        for k in self._FIELDS:
            print("\t", k, "=", getattr(self, k))


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

    # Verbosity levels accepted by :meth:`run` (and other ``verbose`` switches).
    # ``verbose`` is an integer; legacy ``bool`` values map to 0/1 transparently
    # (Python ``False``/``True`` are ``0``/``1``), so existing callers are unaffected.
    VERBOSE_SILENT = 0  # no per-timestep or end-of-run output
    VERBOSE_DEFAULT = 1  # per-timestep lines + end-of-run statistics (legacy True)
    VERBOSE_TIMERS = 2  # additionally print timers at the end of every run() call
    VERBOSE_EVALUATORS = 3  # additionally let every parallel-evaluation worker print
    #                         (default: only one evaluator's output is shown)

    # The build-info banner (engines/discretizer/package) is process-global. Print it only
    # on the first DartsModel construction so the silently-built evaluator-factory sub-models
    # (one per parallel-evaluation wrap target) and forked/spawned workers don't duplicate it
    # N times in the run log.
    _build_info_printed = False

    def __new__(cls, *args, **kwargs):
        """
        Capture the constructor arguments so the model can be reconstructed in a
        worker process by :class:`ModelEvaluatorFactory` (the default mechanism
        behind :meth:`get_evaluator_factory`). The arguments are stored verbatim;
        they must be picklable for ``parallel_evaluation=True`` to work.
        """
        instance = super().__new__(cls)
        instance._init_args = args
        instance._init_kwargs = kwargs
        return instance

    def __init__(self):
        """
        Initialize DartsModel class.
        """
        # print out build information once per process (see DartsModel._build_info_printed)
        if not DartsModel._build_info_printed:
            engines_pbi()
            discretizer_pbi()
            package_pbi()
            DartsModel._build_info_printed = True

        # Create member variables reservoir and physics
        self.reservoir = None
        self.physics = None

        # Create member variable wells (it is needed only for DFM wells)
        self.wells = None

        # Single source of truth for verbosity. Methods with a ``verbose`` parameter
        # default to ``None`` and fall back to this attribute, so the level is set once
        # (here or by the caller) instead of being re-supplied at every call. See the
        # VERBOSE_* constants for the meaning of each level.
        self.verbose = self.VERBOSE_DEFAULT

        # Create time_node object for time record
        self.timer = timer_node()

        # Start time record
        self.timer.start()

        # Create timer.node called "simulation" to record simulation time
        self.timer.node["simulation"] = timer_node()

        self.timer.node["newton update"] = timer_node()
        self.timer.node["output"] = timer_node()

        # Previously-untimed wall-clock now gets its own root-level nodes so print_timers
        # attributes it instead of leaving it in the "Total elapsed" gap:
        #   run loop overhead -- per-timestep Python orchestration in run() outside run_timestep
        #   cache I/O         -- periodic OBL adaptive-cache flushes (write_cache) during run()
        self.timer.node["run loop overhead"] = timer_node()
        self.timer.node["cache I/O"] = timer_node()

        # Create timer.node called "initialization" to record initialization time
        self.timer.node["initialization"] = timer_node()

        # Start recording "initialization" time
        self.timer.node["initialization"].start()

        # Create sim_params object to set simulation parameters
        self.params = sim_params()

        # Nonlinear solver instance (a NewtonSolver; see darts.nonlinear_solvers)
        # built from its declarative spec. Assigned lazily by set_solver() and
        # bound to this model in init(). Its input spec is DartsModel
        # .nonlinear_solver.spec (retrievable for tracing/serialization).
        self.nonlinear_solver = None
        self._data_ts = (
            None  # lazy timestep-control structure, see the data_ts property
        )

        self.time = []
        self.n_newton_iters = []
        self.time_step_size = []

        # Stop recording "initialization" time
        self.timer.node["initialization"].stop()

    def get_evaluator_factory(self, attribute='reservoir_operators', region=None):
        """
        Return a picklable factory callable ``() -> operator_set_evaluator_iface``
        that constructs a fresh, independent evaluator for the given physics
        attribute and (optional) region, used by :class:`ParallelEvaluator` when
        ``parallel_evaluation=True``.

        The default implementation returns a :class:`ModelEvaluatorFactory`, which
        reconstructs this model from its constructor arguments (captured in
        :meth:`__new__`) and returns ``physics.<attribute>`` (singular) or
        ``physics.<attribute>[region]`` (per-region). This reuses the model's own
        ``set_physics``/``PropertyContainer`` build, so no per-model duplication of
        the property stack is required and it works for any model whose constructor
        arguments are picklable.

        Override this method only if model reconstruction is too expensive to
        repeat per worker, or if the constructor arguments are not picklable.

        :param attribute: Name of the physics attribute to wrap
            (``'reservoir_operators'``, ``'property_operators'``, ``'well_operators'``,
            ``'well_ctrl_operators'``, ``'thermal_var_operator'``, or chemistry's
            ``'initial_operators'``).
        :type attribute: str
        :param region: Region index for per-region operator dicts; ``None`` for
            singular attributes such as ``well_ctrl_operators``.
        :type region: int | None
        :return: Picklable factory callable that creates a fresh evaluator
        :rtype: callable
        """
        from darts.physics.base.parallel_evaluator import ModelEvaluatorFactory

        return ModelEvaluatorFactory(
            type(self),
            getattr(self, '_init_args', ()),
            getattr(self, '_init_kwargs', {}),
            attribute=attribute,
            region=region,
        )

    def init(
        self,
        discr_type: str = "tpfa",
        platform: str = "cpu",
        restart: bool = False,
        verbose: int | None = None,
        itor_mode: str = "adaptive",
        itor_type: str = "multilinear",
        is_barycentric: bool = False,
        n_solid: int = None,
        parallel_evaluation: bool = False,
        n_workers: int = None,
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
        :param verbose: Verbosity level (``int``; ``bool`` accepted for backward
            compatibility). Defaults to ``None``, meaning inherit :attr:`self.verbose`.
        :type verbose: int
        :param itor_mode: specifies either 'static' or 'adaptive' interpolator
        :type itor_mode: str
        :param itor_type: specifies either 'linear' or 'multilinear' interpolator
        :type itor_type: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        :param n_solid: Number of solid minerals for element-based reactive flow
        :type n_solid: int
        :param parallel_evaluation: Enable parallel batch evaluation of supporting points via multiprocessing.
            All five evaluator-interpolator pairs in the physics (reservoir, property, well,
            well_ctrl, thermal_var) are wrapped through a single shared multiprocessing pool.
            Requires the model to implement ``get_evaluator_factory(attribute, region=None)``.
        :type parallel_evaluation: bool
        :param n_workers: Number of worker processes for parallel evaluation (default: os.cpu_count())
        :type n_workers: int
        """
        verbose = self.verbose if verbose is None else verbose

        # Time the (previously untimed) initialization phases. init() runs the big up-front
        # costs -- mesh build, physics/Engine construction and OBL interpolator setup / cache
        # loading, and initial-condition interpolation -- that otherwise vanish into the root
        # "Total elapsed" gap. Accumulates into the same "initialization" node the subclass
        # __init__ uses, so the node covers construction + init() together.
        init_timer = self.timer.node["initialization"]
        init_timer.start()

        # Initialize reservoir and Mesh object
        assert self.reservoir is not None, "Reservoir object has not been defined"
        init_timer.node["reservoir init"] = timer_node()
        init_timer.node["reservoir init"].start()
        self.reservoir.init_reservoir(bool(verbose))
        init_timer.node["reservoir init"].stop()
        self.set_wells()
        self.has_dfm_well = any(
            well.ms_type == ms_well.MS_Type.DFM for well in self.reservoir.wells
        )
        if self.has_dfm_well:
            self.timer.node["simulation"].node["dfm_well_velocity_calculation"] = (
                timer_node()
            )
        else:
            # If there are no DFM wells, no Python well objects are needed.
            self.wells = None

        # Initialize physics and Engine object
        assert self.physics is not None, "Physics object has not been defined"
        self.platform = platform
        # Build evaluator_factory_hook from model's get_evaluator_factory if available
        evaluator_factory_hook = None
        if parallel_evaluation:
            evaluator_factory_hook = self.get_evaluator_factory

        init_timer.node["physics init & OBL cache load"] = timer_node()
        init_timer.node["physics init & OBL cache load"].start()
        self.physics.init_physics(
            discr_type=discr_type,
            platform=platform,
            verbose=bool(verbose),
            itor_mode=itor_mode,
            itor_type=itor_type,
            is_barycentric=is_barycentric,
            n_solid=n_solid,
            parallel_evaluation=parallel_evaluation,
            n_workers=n_workers,
            evaluator_factory_hook=evaluator_factory_hook,
            verbose_evaluators=int(verbose) >= self.VERBOSE_EVALUATORS,
        )
        init_timer.node["physics init & OBL cache load"].stop()
        if platform == "gpu":
            self.params.linear_type = sim_params.gpu_gmres_cpr_amgx_ilu
        self.params.sim_eps = self.physics.sim_eps

        # Initialize well objects
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)

        self.set_op_list()
        self.set_boundary_conditions()
        self.set_well_controls()

        # Materialize the nonlinear solver spec (and default specs/data_ts if the
        # model did not configure them) before the engine is initialized.
        self._apply_nonlinear()

        # when restarting the initial conditions are set in self.load_restart_data() and the engine is reset.
        self.restart = restart
        if restart is False:
            init_timer.node["initial conditions"] = timer_node()
            init_timer.node["initial conditions"].start()
            self.set_initial_conditions()
            init_timer.node["initial conditions"].stop()
            init_timer.node["engine init"] = timer_node()
            init_timer.node["engine init"].start()
            self.reset()
            init_timer.node["engine init"].stop()
            self.initialize_history_fields()
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

        init_timer.stop()

    def reset(self):
        """
        Function to initialize the engine by calling 'engine.init()' method.
        """
        self.physics.engine.init(
            self.reservoir.mesh,
            ms_well_vector(self.reservoir.wells),
            op_vector(self.op_list),
            self.physics.thermal_var_itor,
            self.params,
            self.timer.node["simulation"],
        )

    def initialize_history_fields(self):
        """Seed ``engine.Xhistory`` with the per-field default value for every reservoir cell.

        No-op when the physics has no ``history_fields`` configured (the engine then also has
        ``n_history_runtime == 0`` and no ``Xhistory`` buffer). Called by :meth:`init` right after
        :meth:`reset`, which is where the C++ engine allocates ``Xhistory``.

        :returns: None
        """
        if not getattr(self.physics, "history_fields", None):
            return

        n_blocks = self.reservoir.mesh.n_blocks
        for field in self.physics.history_fields:
            self.physics.set_engine_history_array(
                field.label,
                field.default,
                n_blocks=n_blocks,
            )

    def after_converged_timestep(self):
        """Hook called after each converged Newton timestep. Advances history fields by default.

        Subclasses that override this should call ``super().after_converged_timestep()`` to
        preserve the history-field update. The base implementation simply delegates to
        :meth:`update_history_fields_after_timestep`.

        :returns: None
        """
        self.update_history_fields_after_timestep()

    def update_history_fields_after_timestep(self):
        """User hook to advance OBL history variables (e.g. ``sg_max``) between timesteps.

        The base implementation is a no-op. Subclasses backing a hysteretic physics should
        override this to read the current Newton state, compute the updated history value
        per cell, and write it back via :meth:`PhysicsBase.set_engine_history_array` (or by
        mutating the underlying ``engine.Xhistory`` vector directly).

        :returns: None
        """
        return

    def load_restart_data(self, reservoir_filepath: str, ts_idx: int = -1):
        """
        Loads data from a previous simulation and sets it for the current simulation.
        Beware that loading restart data resets the engine.

        :param reservoir_filepath: Path to the restart file containing reservoir block data.
        :type reservoir_filepath: str
        :param ts_idx: The timestep index to load from the file (default: -1 for the last timestep)
        :type ts_idx: int
        """
        # check if the files with data exist
        if not os.path.exists(
            reservoir_filepath
        ):  # or not os.path.exists(well_filepath):
            raise FileNotFoundError(
                f"The restart file does not exist: {reservoir_filepath}"
            )

        # Read data from the file
        time_res, reservoir_cell_id, Xres, var_names = self.output.read_specific_data(
            reservoir_filepath, ts_idx
        )

        # Split columns: primary Newton unknowns (self.physics.vars) go through
        # set_initial_conditions_from_array; OBL history columns (self.physics.history_fields)
        # go through set_engine_history_array so sg_max and friends survive restart.
        primary_names = list(self.physics.vars)
        history_labels = set()
        if hasattr(self.physics, "history_fields"):
            history_labels = {h.label for h in self.physics.history_fields}

        initial_values = {}
        history_values = {}
        for i, name in enumerate(var_names):
            key = name.decode() if isinstance(name, bytes) else name
            col = Xres[:, :, i].flatten()
            if key in primary_names:
                initial_values[key] = col
            elif key in history_labels:
                history_values[key] = col
            else:
                initial_values[key] = col  # unknown key: preserve legacy routing
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=initial_values
        )

        self.reset()
        self.physics.engine.t = time_res[0]

        # Push the restored history columns into engine.Xhistory. reset() has already allocated
        # the buffer, so set_engine_history_array only needs to overwrite its contents.
        for label, values in history_values.items():
            self.physics.set_engine_history_array(
                label,
                values,
                n_blocks=self.reservoir.mesh.n_res_blocks,
            )

        # save initial conditions to *.h5 file
        print(rf'Restarting model from {reservoir_filepath} at day {time_res[0]}.')
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
        verbose: int | None = None,
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
        :param verbose: Verbosity level (``int``; ``bool`` accepted). Defaults to
            ``None``, meaning inherit :attr:`self.verbose`.
        """
        verbose = self.verbose if verbose is None else verbose

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
            verbose=bool(verbose),
            wells=self.wells,
            has_dfm_well=self.has_dfm_well,
        )

        return

    def set_wells(self, verbose: int | None = None):
        """
        Function to define wells. The default method of DartsModel.set_wells() calls Reservoir.set_wells().

        :param verbose: Verbosity level (``int``; ``bool`` accepted). Defaults to
            ``None``, meaning inherit :attr:`self.verbose`.
        :type verbose: int
        """
        verbose = self.verbose if verbose is None else verbose
        self.reservoir.set_wells(bool(verbose))
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

    def set_solver(self):
        """Hook to specify the solvers of the model (consistent with the
        linear-solver ``set_solver()`` of MR280 — after the merge both the
        linear and the nonlinear solver are specified here).

        ``self.nonlinear_solver`` holds the solver *instance* built from its
        declarative spec; the input spec stays retrievable as
        ``self.nonlinear_solver.spec`` (serializable via ``.to_dict()``, for tracing).

        The default implementation is idempotent and lazy: it keeps any solver a
        subclass already assigned and otherwise materializes the default.
        Override in a model to select/tune the nonlinear solver, either by
        replacing it::

            def set_solver(self):
                super().set_solver()
                self.nonlinear_solver = NewtonSolver(tolerance=1e-4,
                                                     chop=ChopSpec(mode='global'))

        or by tuning the spec of the default::

            def set_solver(self):
                super().set_solver()
                self.nonlinear_solver.spec.tolerance = 1e-4
                self.nonlinear_solver.spec.chop.factor = 0.2
        """
        if getattr(self, "nonlinear_solver", None) is None:
            self.nonlinear_solver = default_nonlinear_solver()

    @property
    def data_ts(self):
        """Timestep-control (and transitional linear-solver) structure, see
        :class:`DataTS`. Created lazily so it can be read/written both before and
        after ``init()``. The nonlinear-solver settings live on
        ``self.nonlinear_solver.spec``, not here."""
        if self._data_ts is None:
            n_vars = self.physics.n_vars if getattr(self, "physics", None) else 0
            self._data_ts = DataTS(n_vars)
        return self._data_ts

    @data_ts.setter
    def data_ts(self, value):
        self._data_ts = value

    def _apply_nonlinear(self):
        """Bind the nonlinear solver to this model and make sure
        ``data_ts``/``sim_params`` exist. Called from init()."""
        self.set_solver()
        if self._data_ts is None:
            self.data_ts = DataTS(self.physics.n_vars)
        # the structure may have been created pre-init with n_vars=0: size eta now
        if len(self._data_ts.eta) < self.physics.n_vars:
            self._data_ts.eta = 1e20 * np.ones(self.physics.n_vars)
        # ALWAYS mirror the (possibly user-set) linear settings into sim_params —
        # not only when data_ts was just created. Reading model.data_ts before
        # init() materializes _data_ts, which previously skipped this copy and
        # silently dropped a user's data_ts.linear_tol/linear_max_iter.
        self.copy_data_ts_to_sim_params()
        # bind the (possibly detached) solver to this model
        self.nonlinear_solver.bind(self)

    def set_sim_params_data_ts(self, data_ts):
        """Deprecated: assign ``nonlinear_solver`` and set timestep controls on
        ``data_ts`` instead."""
        warnings.warn(
            "set_sim_params_data_ts() is deprecated; specify DartsModel.nonlinear_solver "
            "in set_solver() and set timestep controls on DartsModel.data_ts instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_solver()
        self.data_ts = DataTS(self.physics.n_vars)
        # copy attributes except eta
        for k in DataTS._FIELDS:
            if k == "eta":
                continue
            setattr(self.data_ts, k, getattr(data_ts, k))
        self.copy_data_ts_to_sim_params()

    def set_sim_params(
        self,
        first_ts: float = None,
        mult_ts: float = None,
        min_ts=1e-15,
        max_ts: float = None,
        runtime: float = 1000,
        tol_linear: float = None,
        it_linear: int = None,
        **legacy,
    ):
        """
        Function to set the timestep and linear solver parameters.

        The nonlinear solver parameters are NOT set here anymore — specify them
        on ``self.nonlinear_solver`` (a :class:`darts.nonlinear_solvers.NewtonSolver`),
        typically in a ``set_solver()`` override::

            self.nonlinear_solver = NewtonSolver(tolerance=1e-4, max_iterations=15,
                                                 chop=ChopSpec(mode='local', factor=0.2))

        For one deprecation cycle the removed nonlinear keyword arguments
        (``tol_newton``, ``it_newton``, ``newton_type``, ``newton_params``,
        ``line_search``, ``coupled_well_res_norm_method``) are still accepted:
        they emit a :class:`DeprecationWarning` and are mapped onto
        ``self.nonlinear_solver.spec``. Any other unexpected keyword still raises
        :class:`TypeError`.

        :param first_ts: First timestep
        :type first_ts: float
        :param mult_ts: Timestep multiplier
        :type mult_ts: float
        :param max_ts: Maximum timestep
        :type max_ts: float
        :param runtime: Total runtime in days, default is 1000
        :type runtime: float
        :param tol_linear: Tolerance for linear iterations
        :type tol_linear: float
        :param it_linear: Maximum number of linear iterations
        :type it_linear: int

        .. deprecated::
            Set timestep controls on ``self.data_ts`` and the linear solver via
            the linear-solver spec (MR280) instead.
        """
        warnings.warn(
            "set_sim_params() is deprecated; set timestep controls on "
            "DartsModel.data_ts and specify DartsModel.nonlinear_solver in set_solver()",
            DeprecationWarning,
            stacklevel=2,
        )
        # nonlinear settings are NOT set here — they live on self.nonlinear_solver
        self.set_solver()

        # one-cycle migration: route any legacy nonlinear kwargs onto the spec
        if legacy:
            self._migrate_legacy_nonlinear_kwargs(legacy)

        # fresh timestep-control structure
        self.data_ts = DataTS(self.physics.n_vars)
        ts = self.data_ts

        # Time stepping parameters. if None, default value will be used
        ts.dt_first = first_ts if first_ts is not None else ts.dt_first
        ts.dt_min = min_ts if min_ts is not None else ts.dt_min
        ts.dt_max = max_ts if max_ts is not None else ts.dt_max
        ts.dt_mult = mult_ts if mult_ts is not None else ts.dt_mult

        # Linear solver parameters. if None, default value will be used
        ts.linear_tol = tol_linear if tol_linear is not None else 1e-5
        ts.linear_max_iter = it_linear if it_linear is not None else 50

        self.runtime = runtime

        self.copy_data_ts_to_sim_params()

    def _migrate_legacy_nonlinear_kwargs(self, legacy: dict):
        """One-deprecation-cycle shim: map removed ``set_sim_params`` nonlinear
        keyword arguments onto ``self.nonlinear_solver.spec`` and warn. Unknown
        keys raise TypeError so genuine typos still fail loudly."""
        from darts.nonlinear_solvers.newton import _ENUM_TO_CHOP_MODE

        spec = self.nonlinear_solver.spec
        handled = []
        if "tol_newton" in legacy:
            spec.tolerance = legacy.pop("tol_newton")
            handled.append("tol_newton -> nonlinear_solver.spec.tolerance")
        if "it_newton" in legacy:
            spec.max_iterations = legacy.pop("it_newton")
            handled.append("it_newton -> nonlinear_solver.spec.max_iterations")
        if "line_search" in legacy:
            spec.line_search.enabled = bool(legacy.pop("line_search"))
            handled.append("line_search -> nonlinear_solver.spec.line_search.enabled")
        if "coupled_well_res_norm_method" in legacy:
            spec.coupled_well_res_norm_method = legacy.pop(
                "coupled_well_res_norm_method"
            )
            handled.append(
                "coupled_well_res_norm_method -> "
                "nonlinear_solver.spec.coupled_well_res_norm_method"
            )
        if "newton_type" in legacy:
            nt = legacy.pop("newton_type")
            # accept the legacy int (0/1/2), the mode string, or None
            spec.chop.mode = (
                _ENUM_TO_CHOP_MODE.get(nt, nt) if isinstance(nt, int) else nt
            )
            spec.chop.__post_init__()  # validate the mapped mode
            handled.append("newton_type -> nonlinear_solver.spec.chop.mode")
        if "newton_params" in legacy:
            np_val = legacy.pop("newton_params")
            spec.chop.factor = np_val[0] if isinstance(np_val, list | tuple) else np_val
            handled.append("newton_params[0] -> nonlinear_solver.spec.chop.factor")
        if legacy:
            raise TypeError(
                f"set_sim_params() got unexpected keyword argument(s) {sorted(legacy)}"
            )
        warnings.warn(
            "set_sim_params() nonlinear keyword arguments are removed; mapped for "
            "this release only (" + "; ".join(handled) + "). Migrate to "
            "self.nonlinear_solver = NewtonSolver(...) in set_solver().",
            DeprecationWarning,
            stacklevel=3,
        )

    def copy_data_ts_to_sim_params(self):
        """Transitional: mirror the linear solver settings into the C++
        ``sim_params`` fields the engine still reads. The nonlinear controls
        are synced directly into the engine by the nonlinear solver."""
        self.params.tolerance_linear = self.data_ts.linear_tol
        self.params.max_i_linear = self.data_ts.linear_max_iter
        if self.data_ts.linear_type is not None:
            if (
                type(self.data_ts.linear_type) is not linear_solver_types
            ):  # it's not needed to copy it to params for PETSC option
                self.params.linear_type = self.data_ts.linear_type

    def run_simple(self, physics, data_ts, days, restart_dt=0.0):
        """
        Run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param physics:
        :param data_ts:
        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        """
        self.physics = physics
        self.data_ts = data_ts
        # bind the model's nonlinear solver (its spec is the single config source)
        self.set_solver()
        self.nonlinear_solver.bind(self)

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
                self.after_converged_timestep()
                if verbose:
                    print(
                        f"# {ts:d}\tT = {t:3g}\tDT = {dt:2g}\tNI = {self.nonlinear_solver.status.n_newton:d}\tLI={self.nonlinear_solver.status.n_linear:d}"
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
                f"TS = {self.nonlinear_solver.stats.n_timesteps_total:d}({self.nonlinear_solver.stats.n_timesteps_wasted:d}), NI = {self.nonlinear_solver.stats.n_newton_total:d}({self.nonlinear_solver.stats.n_newton_wasted:d}), LI = {self.nonlinear_solver.stats.n_linear_total:d}({self.nonlinear_solver.stats.n_linear_wasted:d})"
            )

    def run(
        self,
        days: float = None,
        restart_dt: float = 0.0,
        save_well_data: bool = True,
        save_well_data_after_run: bool = True,
        save_reservoir_data: bool = True,
        verbose: int | None = None,
    ):
        """
        Run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Verbosity level. Defaults to ``None``, meaning inherit
            :attr:`self.verbose`. Accepts a bool for backward compatibility
            (``False``/``True`` map to ``0``/``1``). Levels:
            ``0`` (:attr:`VERBOSE_SILENT`) no output;
            ``1`` (:attr:`VERBOSE_DEFAULT`) per-timestep lines + end-of-run statistics;
            ``>=2`` (:attr:`VERBOSE_TIMERS`) additionally print timers at the end of
            every ``run()`` invocation (otherwise timers are only printed when
            :meth:`print_timers` is called manually, usually from ``main.py``).
        :type verbose: int
        :param save_well_data: if True save states of well blocks at every time step to 'well_data.h5', default is True
        :type save_well_data: bool
        :param save_well_data_after_run: Switch to save well data only after runtime of `days`
        :param save_reservoir_data: if True save states of all reservoir blocks at the end of run to 'solution.h5', default is True
        :type save_reservoir_data: bool
        """
        verbose = self.verbose if verbose is None else verbose

        assert hasattr(self, 'output'), (
            "self.output does not exist, please call m.set_output() after m.init()"
        )

        days = days if days is not None else self.runtime
        assert days > 0, "Time must be a positive value!"

        data_ts = self.data_ts

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

        nc = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        max_dx = np.zeros(nc)

        if np.fabs(data_ts.dt_mult - 1) < 1e-10:
            omega = 0.0
        else:
            # inversion assuming mult = (1 + omega) / omega
            omega = 1 / (data_ts.dt_mult - 1)

        ts_counter = 0

        # Per-timestep Python orchestration outside run_timestep (state copies, dt/CFL
        # control, well-data accumulation) is otherwise untimed; bracket it into the
        # "run loop overhead" node instead of leaving it in the root "Total elapsed" gap.
        overhead = self.timer.node["run loop overhead"]
        while t < stop_time:
            # need to copy since Xn will be updated Xn = X
            overhead.start()
            xn = np.array(self.physics.engine.Xn, copy=True)[: nb * nc]
            overhead.stop()
            converged = self.run_timestep(dt, t, verbose)

            overhead.start()
            if converged:
                t += dt
                self.physics.engine.t = t
                ts_counter += 1
                self.after_converged_timestep()

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
                        f"#{ts_counter:d}\tT={t:3g}\tDT={dt:2g}\tNI={self.nonlinear_solver.status.n_newton:d}\tLI={self.nonlinear_solver.status.n_linear:d}\tDT_MULT={dt_mult_new:3.3g}\tdX={max_dx_str}"
                    )

                dt = min(dt * dt_mult_new, data_ts.dt_max)

                if np.fabs(t + dt - stop_time) < data_ts.dt_min:
                    dt = stop_time - t

                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

                if save_well_data:
                    # save well data at every converged time step. save_data_to_h5 brackets
                    # its own output/saving_well_data timer; pause the overhead bracket so
                    # the h5 write is not double-counted.
                    overhead.stop()
                    self.output.save_data_to_h5(kind="well")
                    overhead.start()

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
                if dt <= data_ts.dt_min:
                    overhead.stop()  # keep the bracket balanced before the assert aborts
                assert dt > data_ts.dt_min, (
                    "Stop simulation. Reason: reached min. timestep "
                    + str(data_ts.dt_min)
                    + " dt="
                    + str(dt)
                )

            overhead.stop()

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

        # If adaptive OBL-point caching is enabled, flush OBL cache at the end of each run/report interval
        # to preserve newly evaluated points, so the cache progress survives SIGTERM/job cancel.
        if getattr(self.physics, 'cache', False):
            self.timer.node["cache I/O"].start()
            self.physics.write_cache()
            self.timer.node["cache I/O"].stop()

        if verbose:
            print(
                f"----- TS = {self.nonlinear_solver.stats.n_timesteps_total:d}({self.nonlinear_solver.stats.n_timesteps_wasted:d}), "
                f"NI = {self.nonlinear_solver.stats.n_newton_total:d}({self.nonlinear_solver.stats.n_newton_wasted:d}), "
                f"LI = {self.nonlinear_solver.stats.n_linear_total:d}({self.nonlinear_solver.stats.n_linear_wasted:d}) -----"
            )

        # At higher verbosity, print the timer breakdown at the end of every run()
        # invocation (the default behaviour only prints timers when print_timers() is
        # called explicitly, typically from main.py after the full simulation). These
        # automatic prints go to the redirected darts log rather than stdout, so they
        # land in the run log instead of cluttering the console.
        if verbose >= self.VERBOSE_TIMERS:
            self.print_timers(to_log=True)

        return 0

    def run_timestep(self, dt: float, t: float, verbose: int | None = None):
        """
        Solve the nonlinear loop for the specified timestep.

        Delegates to the runtime nonlinear solver built from
        :attr:`nonlinear_solver` (see :mod:`darts.nonlinear_solvers`).

        :param dt: Timestep size [days]
        :type dt: float
        :param t: Current time [days]
        :type t: float
        :param verbose: Verbosity level (``int``; ``bool`` accepted). Defaults to
            ``None``, meaning inherit :attr:`self.verbose`.
        :type verbose: int
        """
        return self.nonlinear_solver.bind(self).solve_timestep(dt, t, verbose)

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

    def line_search(
        self,
        dt: float,
        t: float,
        coef,
        history,
        verbose: int | None = None,
        iter_counter: int = None,
    ):
        """
        Perform a line search to find the optimal coefficient that minimizes residuals.

        Delegates to the runtime nonlinear solver (see :mod:`darts.nonlinear_solvers`).
        """
        return self.nonlinear_solver.line_search(
            dt, t, coef, history, verbose, iter_counter
        )

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

    def print_timers(self, to_log: bool = False):
        """
        Function to print the time information, including total time elapsed,
        time consumption at different stages of the simulation, etc..

        :param to_log: When ``False`` (default) the timer tree is written to Python's
            stdout via ``print``. When ``True`` it is written to the darts output stream
            instead — i.e. the file passed to :func:`redirect_darts_output` (the same
            destination the C++ engine output, e.g. ``print_stat``, goes to), or the
            terminal if no redirect is active. Use this so the timers land in the run log
            alongside the engine output rather than on the console.
        :type to_log: bool
        """
        timers_str = self.timer.print("", "")
        if to_log:
            from darts.engines import write_to_darts_output

            write_to_darts_output(timers_str + "\n")
        else:
            print(timers_str)

    def print_stat(self):
        """
        Function to print the statistics information, including total timesteps, Newton iteration, linear iteration, etc..
        """
        stats = self.nonlinear_solver.stats
        print(
            f"Total steps {stats.n_timesteps_total} ({stats.n_timesteps_wasted}) "
            f"newton {stats.n_newton_total} ({stats.n_newton_wasted}) "
            f"linear {stats.n_linear_total} ({stats.n_linear_wasted})"
        )
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

    def set_well_controls_idata(self, time: float = 0.0, verbose: int | None = None):
        """
        :param time: simulation time, [days]
        :param verbose: Verbosity level (``int``; ``bool`` accepted). Defaults to
            ``None``, meaning inherit :attr:`self.verbose`.
        :return:
        """
        verbose = self.verbose if verbose is None else verbose
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

    def _solve_linear_equation(self):
        """Backend-neutral linear-solve dispatch funnel used by the nonlinear
        solver (:meth:`darts.nonlinear_solvers.NonlinearSolver._solve_linear`).

        Returns ``(rc, n_iters, residual)`` for every backend — ``rc`` is ``0``
        on success, ``1`` on setup failure, ``2`` on solve failure. Centralizing
        the dispatch here (rather than in the nonlinear solver) is also the seam
        the linear-solver refactoring (MR280) replaces wholesale with
        spec-driven routing, keeping the nonlinear driver backend-agnostic."""
        from darts.input.input_data import linear_solver_types

        linear_type = self.data_ts.linear_type
        if isinstance(linear_type, linear_solver_types):
            # Python-resident solvers
            if linear_type in (
                linear_solver_types.CPU_PETSC_CPR,
                linear_solver_types.CPU_PETSC_FS,
            ):
                return self.petsc_solve_linear_equation()
            elif linear_type in (linear_solver_types.CPU_PARDISO,):
                return self.pardiso_solve_linear_equation()
            raise Exception("Unknown linear solver type", linear_type)
        # compile-time C++ linear solvers
        engine = self.physics.engine
        rc = engine.solve_linear_equation()
        return rc, engine.get_last_linear_iters(), engine.get_last_linear_residual()

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

        reason = petsc_ksp.getConvergedReason()  # >0 converged, <0 diverged
        n_iters = petsc_ksp.getIterationNumber()
        residual = petsc_ksp.getResidualNorm()

        if print_level >= 1:
            print('PETSC: True residual =', np.linalg.norm(mat.dot(sol) - rhs))

        # Treat only hard breakdowns / non-finite / preconditioner failures as
        # a solver failure (rc=2 -> abort Newton / trigger fallback). Max-iter
        # exhaustion (DIVERGED_ITS) is deliberately NOT fatal, mirroring the C++
        # GMRES BOS-parity convention where a partial solve is accepted and the
        # Newton residual gate decides.
        fatal = {
            PETSc.KSP.ConvergedReason.DIVERGED_NANORINF,
            PETSc.KSP.ConvergedReason.DIVERGED_BREAKDOWN,
            PETSc.KSP.ConvergedReason.DIVERGED_BREAKDOWN_BICG,
            PETSc.KSP.ConvergedReason.DIVERGED_PC_FAILED,
        }
        rc = 2 if (reason in fatal or not np.isfinite(sol).all()) else 0
        return rc, int(n_iters), float(residual)

    def pardiso_solve_linear_equation(self):
        import pypardiso

        mat, rhs, sol = self.get_linear_system()
        try:
            sol[:] = pypardiso.spsolve(mat, rhs)
        except Exception:
            sol[:] = 0.0
            return 2, 0, np.inf
        # direct solve: count as one "iteration"; guard against a non-finite result
        if not np.isfinite(sol).all():
            return 2, 0, np.inf
        return 0, 1, float(np.linalg.norm(mat.dot(sol) - rhs))
