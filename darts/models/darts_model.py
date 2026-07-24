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
from darts.interpolators import op_vector
from darts.models.output import Output
from darts.nonlinear_solvers import ChopSpec, NewtonSolver, Norm, OBLBoundsSpec
from darts.pipes.add_lateral_heat_exchange import SemiAnalyticalWellLateralHeatTransfer
from darts.print_build_info import print_build_info as package_pbi

# Open-source linear-solver registry (the darts.linear_solvers package). It is absent
# in proprietary (-a / -b) builds, where the engine's built-in factory selects
# the solver from params.linear_type; the import is therefore guarded.
# The Python helpers (specs, adaptive policy) still import in proprietary
# builds, but their build() calls reach into the compiled extension --
# guard on _have_compiled_solvers, not just the import succeeding.
try:
    from darts import linear_solvers as _darts_solvers_pkg
    from darts.linear_solvers import (
        AdaptiveSolverSpec,
        LinearSolver,
        LinearSolverSpec,
        PythonLinearSolverSpec,
        SolverAction,
        SolverSwitchContext,
        default_linear_solver,
        default_linear_solver_spec,
    )
    from darts.linear_solvers.solver import (
        is_compiled_solver_handle as _is_compiled_solver_handle,
    )

    _HAVE_SOLVER_REGISTRY = getattr(_darts_solvers_pkg, "_have_compiled_solvers", True)
except ImportError:  # proprietary build without the open-source solvers
    _HAVE_SOLVER_REGISTRY = False
    LinearSolver = None


def _describe_solver_spec(spec) -> str:
    """A short human-readable label for a LinearSolverSpec, e.g. ``gmres+cpr``.

    Used only for the engine's "Linear solver type is ..." log line; falls back
    to the class name and tolerates arbitrary specs.
    """
    try:
        name = getattr(spec, "registry_name", "") or type(spec).__name__
        prec = getattr(spec, "prec", None)
        if prec is not None:
            inner = getattr(prec, "registry_name", "") or type(prec).__name__
            return f"{name}+{inner}"
        return name
    except Exception:
        return ""


class DataTS:
    """Timestep-control parameters.

    Holds ONLY the timestep controls (``dt_first``/``dt_min``/``dt_mult``/
    ``dt_max``/``eta``). Neither solver's settings are mirrored here — each
    lives at its own single source of truth:
    ``DartsModel.nonlinear_solver.spec`` (a
    :class:`darts.nonlinear_solvers.NonlinearSolverSpec`, !327) and
    ``DartsModel.linear_solver.spec`` (a
    :class:`darts.linear_solvers.LinearSolverSpec`, !280 — the transitional
    ``linear_*`` attributes this structure carried are now removed).
    """

    _FIELDS = (
        "eta",
        "dt_first",
        "dt_min",
        "dt_mult",
        "dt_max",
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

    #: True when the model configures its LINEAR solver through the engine
    #: factory (``params.linear_type`` / ``engine.ls_params``) instead of a
    #: :class:`~darts.linear_solvers.LinearSolverSpec` -- the mechanics / THMC
    #: path. :meth:`_apply_solver` then leaves the engine in charge *unless* the
    #: model explicitly chose a spec. (Before !327 this was discriminated by
    #: ``data_ts is None``; ``data_ts`` is now a lazy property that always
    #: materializes, so the intent is stated explicitly here.)
    linear_solver_from_engine_factory = False

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

        # The single source of truth for the linear solver: a LinearSolverSpec
        # (e.g. MGRSolverSpec / GMRESSolverSpec(prec=CPRSolverSpec()) / SuperLUSolverSpec
        # / AdaptiveSolverSpec) set by set_solver(); default = default_linear_solver().
        # It owns ALL linear-solver settings -- solver + preconditioner choice,
        # tolerance, max_iterations, print_level -- which _apply_solver() builds,
        # injects and mirrors into sim_params before engine.init().
        # The attribute is a normalizing property: it always HOLDS a runtime
        # darts.linear_solvers.LinearSolver instance (mirror of
        # DartsModel.nonlinear_solver holding a NewtonSolver, !327), while a model
        # may ASSIGN the ergonomic forms -- a LinearSolverSpec (auto-wrapped), a
        # LinearSolver instance, or a raw compiled handle (fine-control path).
        self.linear_solver = None
        # The spec object set_solver() materialised as the platform default (identity
        # compared against self.linear_solver by _solver_is_default(), so that a model which
        # calls super().set_solver() and then replaces self.linear_solver still counts as
        # having *chosen* its solver).
        self._default_solver_obj = None
        self.solver_label = (
            None  # optional short name for self.linear_solver (model-side, log)
        )

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
        _solver_is_superlu = (
            self.params.linear_type == sim_params.linear_solver_t.cpu_superlu
            or type(self._resolve_solver_spec()).__name__ == "SuperLUSolverSpec"
        )
        if _solver_is_superlu and self.reservoir.mesh.n_res_blocks > 30000:
            warnings.warn(
                "The number of cells looks too big to use a direct linear solver: "
                + str(self.reservoir.mesh.n_res_blocks)
                + ' > 30000',
                stacklevel=2,
            )

        init_timer.stop()

    def reset(self):
        """
        Configure the solver/time-stepping via set_solver(), then initialize the engine.

        set_solver() runs first -- the reservoir/mesh and the engine object already
        exist (so block sizes and n_res_blocks are final), but engine.init() has not
        run yet. So any timestepping/Newton/linear params it sets feed engine.init().

        The linear solver (``self.linear_solver``, a runtime
        :class:`darts.linear_solvers.LinearSolver` instance whose declarative spec is
        ``linear_solver.spec``; default = :func:`default_linear_solver`) is then bound,
        built and injected by :meth:`_apply_solver` before ``engine.init``, so the
        engine adopts its ``handle`` and bypasses its own factory -- the mirror of
        ``nonlinear_solver.bind(self)`` in the !327 design. In proprietary / GPU builds
        no backend is built and the engine factory selects the solver from
        ``params.linear_type``.
        """
        self.set_solver()
        self._apply_solver()
        self.physics.engine.init(
            self.reservoir.mesh,
            ms_well_vector(self.reservoir.wells),
            op_vector(self.op_list),
            self.physics.thermal_var_itor,
            self.params,
            self.timer.node["simulation"],
        )

    def _resolve_solver_spec(self):
        """Return the effective linear-solver spec: ``self.linear_solver`` when it is a
        :class:`LinearSolverSpec` (the single source of truth), else ``None``.

        Shared by :meth:`_apply_solver` and :meth:`_maybe_switch_linear_solver` so
        both see the same solver. A ``None`` result means "no spec is available"
        (nothing assigned yet, or a raw-handle wrapper).
        """
        solver = getattr(self, "linear_solver", None)
        spec = getattr(solver, "spec", None)
        return spec if isinstance(spec, LinearSolverSpec) else None

    @property
    def linear_solver(self):
        """The runtime linear solver: a :class:`darts.linear_solvers.LinearSolver`
        instance (mirror of :attr:`nonlinear_solver` holding a ``NewtonSolver``).
        Its declarative configuration is ``linear_solver.spec``; after ``init()``
        the live backend is ``linear_solver.handle`` (engine-injected C++ solver)
        or ``linear_solver.python_solver`` (PETSc / Pardiso). Assignment is
        normalizing -- a model may assign a :class:`LinearSolverSpec` (ergonomic
        form, auto-wrapped), a ``LinearSolver`` instance, or a raw compiled
        solver handle (fine-control path, wrapped via
        ``LinearSolver.from_handle``)."""
        return getattr(self, "_linear_solver_obj", None)

    @linear_solver.setter
    def linear_solver(self, value):
        if value is None or LinearSolver is None or isinstance(value, LinearSolver):
            self._linear_solver_obj = value
        elif isinstance(value, LinearSolverSpec):
            self._linear_solver_obj = LinearSolver(value)
        elif _is_compiled_solver_handle(value):
            # raw compiled darts.linear_solvers handle (fine-control path)
            self._linear_solver_obj = LinearSolver.from_handle(
                value, label=getattr(self, "solver_label", None)
            )
        else:
            raise TypeError(
                "DartsModel.linear_solver must be a LinearSolver, a "
                "LinearSolverSpec or a compiled darts.linear_solvers solver "
                f"handle, got {type(value).__name__}"
            )

    @property
    def _linear_solver(self):
        """Built C++ solver handle of :attr:`linear_solver` (read-only internal
        alias; ``None`` before :meth:`_apply_solver` and on GPU / proprietary
        builds)."""
        return getattr(self.linear_solver, "handle", None)

    @property
    def _python_solver(self):
        """Python-resident PETSc / Pardiso backend of :attr:`linear_solver`
        (read-only internal alias; ``None`` unless a
        :class:`PythonLinearSolverSpec` is active)."""
        return getattr(self.linear_solver, "python_solver", None)

    def _block_size(self) -> int:
        """Matrix block size for building a solver: ``engine.N_VARS`` for mechanics
        engines (it includes the displacement DOFs that ``physics.n_vars`` does not
        count), else ``physics.n_vars`` (pressure + n_components - 1 for flow)."""
        engine = getattr(self.physics, "engine", None)
        if engine is not None and hasattr(engine, "N_VARS"):
            return engine.N_VARS
        return self.physics.n_vars

    def _apply_solver(self):
        """Build the linear solver from ``self.linear_solver`` and inject it into the engine.

        Replaces the former ``_apply_linear_solver_spec``. Called once by :meth:`reset`
        before ``engine.init``: the resolved :class:`LinearSolverSpec` (``self.linear_solver``,
        or the platform default for models that only configure time-stepping) is built
        and injected, so ``engine.init`` adopts it and bypasses its own factory. No-op
        in proprietary / GPU builds and when no ``data_ts`` exists -- there the engine
        factory / ``params.linear_type`` selects the solver.
        """
        engine = getattr(self.physics, "engine", None)
        if engine is None:
            return
        solver = self.linear_solver
        if solver is not None and hasattr(solver, "bind"):
            # Bind the (possibly detached) runtime solver to this model -- mirror
            # of nonlinear_solver.bind(self) in !327's _apply_nonlinear(). Binding
            # resolves the platform-default spec, so it must precede the
            # sim_params mirroring below.
            solver.bind(self)
        # self.linear_solver owns the linear-solver settings: mirror them into sim_params,
        # which is what engine.init() re-applies to the solver it (re-)inits.
        self._sync_solver_to_sim_params()
        # Python-resident backend (PETSc / Pardiso); None unless a
        # PythonLinearSolverSpec is selected. Reset on every (re)build.
        if solver is not None and hasattr(solver, "python_solver"):
            solver.python_solver = None
        platform = getattr(self, "platform", "cpu")
        if platform == "gpu":
            # GPU: a GPUSolverSpec on self.linear_solver names a params.linear_type enum;
            # the GPU engine factory builds the actual solver. Translate it here.
            self._apply_gpu_solver()
            return
        # Backend guard: never call spec.build() without the compiled registry.
        # Proprietary build (no registry): the engine factory selects from
        # params.linear_type. Honour the spec's proprietary fallback enum so a model
        # declares its solver only via self.linear_solver -- no model-level
        # params.linear_type. None leaves params.linear_type as init() set it.
        if not _HAVE_SOLVER_REGISTRY:
            spec = self._resolve_solver_spec()
            proprietary_type = getattr(spec, "proprietary_linear_type", None)
            # Only a spec the model actually chose selects the enum; the default spec
            # materialised by set_solver() leaves the proprietary factory's own default
            # in charge (params.linear_type as engine init() set it).
            if proprietary_type is not None and not self._solver_is_default():
                self.params.linear_type = proprietary_type
            return
        # Other non-CPU platforms (gpu handled above): leave the factory in charge.
        if platform != "cpu":
            return
        spec = self._resolve_solver_spec()
        if self._solver_is_default() and self.linear_solver_from_engine_factory:
            # Mechanics / THMC models configure engine.ls_params + params directly and
            # never chose a spec: leave the engine factory / ls_params in charge.
            return
        if spec is None and getattr(solver, "handle", None) is not None:
            # A raw compiled solver object built outside the spec API (wrapped by
            # LinearSolver.from_handle) -- the documented fine-control path.
            # Previously it was silently replaced by the default GMRES+CPR.
            self._adaptive_solver_index = 0
            self._adaptive_failures = 0
            self.physics.engine.set_linear_solver(
                solver.handle,
                solver.label or getattr(self, "solver_label", None) or "raw",
            )
            return
        if not isinstance(spec, LinearSolverSpec):
            # No solver assigned at all (self.linear_solver is None -- reset() ran
            # without set_solver(), e.g. a THMC subclass override). Apply the
            # platform default when this model uses the registry path, i.e. it has
            # a data_ts -- matching the former _apply_linear_solver_spec, which
            # defaulted to FGMRES + CPR/AMG. Mechanics / THMC models without
            # data_ts leave the engine factory / ls_params in charge.
            if self.linear_solver_from_engine_factory:
                return
            solver = default_linear_solver().bind(self)
            self.linear_solver = solver
            spec = solver.spec
        # Reset adaptive-switching state whenever the solver is (re)built.
        self._adaptive_solver_index = 0
        self._adaptive_failures = 0
        block_size = self._block_size()
        if isinstance(spec, PythonLinearSolverSpec):
            # PETSc / Pardiso run in the Python process and are owned by the model
            # (invoked from _solve_linear_equation). The engine still needs a C++
            # solver to satisfy engine.init(); inject the CPU default -- it is
            # constructed but never used at solve time.
            solver.python_solver = spec.build(block_size)
            solver.handle = default_linear_solver_spec("cpu").build(block_size)
        else:
            # Engine-resident solver (MGR / GMRES / CPR / FSCPR / SuperLU /
            # Adaptive); keep a reference so it outlives the engine's raw pointer.
            solver.handle = spec.build(block_size)
        # Human-readable label for the engine's "Linear solver type is ..." log line.
        self.physics.engine.set_linear_solver(
            solver.handle, _describe_solver_spec(spec)
        )

    def _apply_gpu_solver(self):
        """Translate a GPU ``LinearSolverSpec`` on ``self.linear_solver`` to ``params.linear_type``.

        GPU solvers are selected by the GPU engine factory (``engine_base_gpu``) via
        the ``linear_solver_t`` enum, not the open-source registry. A
        :class:`~darts.linear_solvers.specs.GPUSolverSpec` names the enum value
        (``linear_type_name``); set it on ``params.linear_type`` here, before
        ``engine.init`` adopts it. A ``None`` / non-GPU ``self.linear_solver`` leaves
        ``params.linear_type`` as :meth:`init` set it (``gpu_gmres_cpr_amgx_ilu``).
        """
        spec = self._resolve_solver_spec()
        linear_type_name = getattr(spec, "linear_type_name", None)
        if not linear_type_name:
            if isinstance(spec, LinearSolverSpec):
                # A CPU registry spec (MGR / GMRES+CPR / SuperLU / ...) on the
                # GPU platform cannot be honoured -- the GPU engine factory
                # selects from params.linear_type. Previously this was silent
                # and the model ran the GPU default while the user believed
                # their spec was active.
                warnings.warn(
                    f"self.linear_solver = {type(spec).__name__} is a CPU registry spec; "
                    "on platform='gpu' the engine uses the GPU factory "
                    "(params.linear_type) and this spec is ignored. Use a "
                    "GPUSolverSpec subclass (AMGXCPRSolverSpec, CuDSSSolverSpec, ...) "
                    "or run on platform='cpu'.",
                    stacklevel=2,
                )
            return
        enum_value = getattr(sim_params, linear_type_name, None)
        if enum_value is None:
            raise ValueError(
                f"GPU solver spec {type(spec).__name__} names linear_type "
                f"'{linear_type_name}', which does not exist in this build's "
                "darts.engines.sim_params -- the requested GPU solver is not "
                "available."
            )
        self.params.linear_type = enum_value
        # Local (block-Schur) elimination: the GPU engine factory wraps the
        # selected chain in linsolv_schur_elim<N,K> when this counter is set. The
        # explicit eliminated (row, column) pairs are passed through params.
        schur_elim = int(getattr(spec, "schur_elim_count", 0) or 0)
        if hasattr(self.params, "schur_elim_count"):
            # sim_params vector<int> members are opaque-bound (index_vector),
            # so wrap the Python lists rather than assigning them directly.
            from darts.engines import index_vector

            if schur_elim:
                rows = getattr(spec, "schur_elim_rows", None)
                cols = getattr(spec, "schur_elim_cols", None)
                if (
                    not rows
                    or not cols
                    or len(rows) != schur_elim
                    or len(cols) != schur_elim
                ):
                    raise ValueError(
                        f"{type(spec).__name__}: schur_elim_count={schur_elim} requires "
                        "schur_elim_rows and schur_elim_cols each of that length."
                    )
                self.params.schur_elim_count = schur_elim
                self.params.schur_elim_rows = index_vector([int(r) for r in rows])
                self.params.schur_elim_cols = index_vector([int(c) for c in cols])
            else:
                # ALWAYS clear: a solver re-selection (reset() with a new spec)
                # must not inherit elimination from a previously applied spec.
                self.params.schur_elim_count = 0
                self.params.schur_elim_rows = index_vector([])
                self.params.schur_elim_cols = index_vector([])

    def set_solver(self):
        """Configure the model's solver and time-stepping (override hook).

        This is the single per-model place to declare all time-stepping / Newton /
        linear-solver settings. It is called at the start of :meth:`reset` (after
        the reservoir/mesh and engine object exist, before ``engine.init``), so it
        may freely:

        * call ``self.set_sim_params(...)`` and set ``self.params.* / self.data_ts.*``
          (these feed ``engine.init`` and the run);
        * set ``self.linear_solver = <LinearSolverSpec>`` -- the single, **build-safe** way
          to pick a solver (``SuperLUSolverSpec``, ``GMRESSolverSpec(prec=CPRSolverSpec())``,
          ``MGRSolverSpec``, ``AdaptiveSolverSpec([...])``, ...). The assignment is
          normalizing: the attribute stores a runtime
          :class:`darts.linear_solvers.LinearSolver` instance wrapping the spec
          (``linear_solver.spec``), mirroring ``nonlinear_solver`` holding a
          ``NewtonSolver`` (!327). Explicit instances (``LinearSolver(spec)`` /
          ``LinearSolver(tolerance=1e-6)``) are accepted too. In proprietary / GPU
          builds no backend is built and the engine factory uses ``params.linear_type``,
          so a spec is safe in any build;
        * for fine control a model may still build a raw C++ solver object into
          ``self.linear_solver`` (e.g. ``linear_solvers.create_mgr_solver_for_block_size(...)``),
          valid only in the open-source build (guard with
          :meth:`open_source_solvers_available`); it is injected after ``engine.init``
          (see :meth:`_apply_solver`), and ``self.solver_label`` names it in the log.
          This raw path is being retired in favour of specs (MGRSolverSpec now covers
          the full MGR configuration).

        Default implementation: select **FGMRES + open-source CPR/AMG** -- the
        in-tree restart-GMRES around the two-stage CPR preconditioner (HYPRE
        BoomerAMG on the pressure subsystem + ILU(0) on the full system), the
        open-source equivalent of the legacy ``bos_gmres + bos_cpr_amg`` default.
        A subclass that wants a different solver overrides this method (setting
        ``self.linear_solver = <Spec>``); one that wants CPR/AMG plus its own time-stepping
        calls ``set_sim_params(...)`` then leaves ``self.linear_solver`` unset (or calls
        ``super().set_solver()``).
        """
        # Idempotent: a subclass that already chose a solver (spec, instance or
        # raw handle) keeps its choice, so super().set_solver() composition is safe.
        self._ensure_default_solvers()

    def _ensure_default_solvers(self):
        """Materialize the platform-default nonlinear and linear solvers when the
        model has not chosen them.

        Factored out of :meth:`set_solver` so internal call sites
        (:meth:`set_sim_params`, :meth:`_migrate_legacy_solver_kwargs`) can
        guarantee the solvers exist WITHOUT re-entering the overridable
        ``set_solver()`` hook -- models call ``set_sim_params()`` from their
        ``set_solver()`` override, so calling back would recurse infinitely.
        """
        if getattr(self, "nonlinear_solver", None) is None:
            # Default nonlinear solver, with every parameter stated explicitly.
            # These values mirror the NewtonSpec/NonlinearSolverSpec field
            # defaults — keep the two in sync when changing a default.
            self.nonlinear_solver = NewtonSolver(
                tolerance=1e-3,  # reservoir-block residual tolerance
                well_tolerance_multiplier=100.0,  # well tol = tolerance * this
                max_iterations=20,  # max Newton iterations per timestep
                stationary_point_tolerance=1e-3,  # residual-stagnation detection
                norm=Norm.L2,  # residual norm
                coupled_well_res_norm_method=1,  # DFM coupled well-res norm (1 or 2)
                chop=ChopSpec(
                    mode="local",  # 'local' | 'global' | None
                    factor=0.1,  # max composition change per iteration
                    log_transform=False,  # solve in log-composition variables
                ),
                obl_bounds=OBLBoundsSpec(
                    mode=None,  # None (off) | 'obl_axes'
                    axis_min=None,  # per-state-variable lower bounds
                    axis_max=None,  # per-state-variable upper bounds
                ),
            )
            # pre_routines / post_routines / fallbacks default to empty lists

        solver = getattr(self, "linear_solver", None)
        if solver is not None:
            # A pre-assigned detached LinearSolver (e.g. LinearSolver(tolerance=...))
            # may still be unresolved; bind it now so linear_solver.spec is
            # materialized and tunable right after super().set_solver().
            if (
                hasattr(solver, "bind")
                and getattr(solver, "spec", None) is None
                and getattr(solver, "handle", None) is None
            ):
                solver.bind(self)
            return
        # Materialise the platform default so that self.linear_solver always holds
        # a runtime LinearSolver instance a subclass can tune -- the single home
        # for linear-solver settings (mirror of the nonlinear form of !327):
        #
        #     def set_solver(self):
        #         self.set_sim_params(first_ts=..., max_ts=...)      # time-stepping
        #         super().set_solver()                              # default solvers
        #         self.nonlinear_solver.spec.tolerance = 1e-3       # nonlinear knobs
        #         self.linear_solver.spec.tolerance = 1e-6          # linear knobs
        #         self.linear_solver.spec.max_iterations = 40
        #
        # The default is detached (its platform-default spec resolves when
        # _apply_solver() binds it, where model.platform is known).
        # _solver_is_default() records that nobody *chose* this solver. Builds/
        # platforms that do not use the open-source registry (proprietary) and
        # models with linear_solver_from_engine_factory = True (mechanics / THMC,
        # which drive engine.ls_params + params directly) keep the engine factory
        # in charge, so _apply_solver() must not build or otherwise act on a
        # solver the model never asked for -- it checks those flags.
        if LinearSolver is None:  # proprietary build without darts.linear_solvers
            return
        # bind() here (not only in _apply_solver) so the platform-default spec is
        # resolved immediately and the documented tuning form
        # `self.linear_solver.spec.tolerance = ...` works right after super().set_solver().
        self.linear_solver = default_linear_solver().bind(self)
        self._default_solver_obj = self.linear_solver

    @staticmethod
    def open_source_solvers_available() -> bool:
        """True when the compiled ``darts.linear_solvers`` registry is present (open-source
        build). False in proprietary ``-a`` builds and where the extension failed to
        load -- there ``create_mgr_solver_*`` / specs are unavailable and the engine
        factory selects the solver from ``params.linear_type`` instead. Use this to
        guard raw-object solver builds in :meth:`set_solver`."""
        try:
            from darts import linear_solvers

            return bool(getattr(linear_solvers, "_have_compiled_solvers", False))
        except Exception:
            return False

    def _linear_solver_timer_totals(self):
        """Cumulative (setup, solve) seconds of the engine's linear-solver
        timer nodes; (0, 0) when the tree is not available."""
        try:
            sim = self.timer.node["simulation"]
            return (
                sim.node["linear solver setup"].get_timer(),
                sim.node["linear solver solve"].get_timer(),
            )
        except (AttributeError, KeyError):
            return (0.0, 0.0)

    def _build_switch_context(self, timestep_converged: bool, dt: float):
        """Assemble the :class:`SolverSwitchContext` for the adaptive policy,
        including per-attempt linear-solver timer deltas and the model's
        ``solver_phase`` tag."""
        engine = self.physics.engine
        setup_total, solve_total = self._linear_solver_timer_totals()
        d_setup = setup_total - getattr(self, "_ls_setup_time_prev", 0.0)
        d_solve = solve_total - getattr(self, "_ls_solve_time_prev", 0.0)
        self._ls_setup_time_prev = setup_total
        self._ls_solve_time_prev = solve_total
        # Per-timestep counters live on the nonlinear solver's NonlinearStatus
        # since !327 (they were engine members before).
        status = getattr(getattr(self, "nonlinear_solver", None), "status", None)
        return SolverSwitchContext(
            current_index=getattr(self, "_adaptive_solver_index", 0),
            timestep_converged=timestep_converged,
            linear_solver_error=int(getattr(status, "linear_solver_rc", 0) or 0),
            linear_iterations=int(getattr(status, "n_linear", 0) or 0),
            newton_iterations=int(getattr(status, "n_newton", 0) or 0),
            consecutive_failures=getattr(self, "_adaptive_failures", 0),
            dt=float(dt),
            simulation_time=float(getattr(engine, "t", 0.0)),
            ls_setup_time=d_setup,
            ls_solve_time=d_solve,
            phase=getattr(self, "solver_phase", None),
        )

    def _maybe_switch_linear_solver(self, timestep_converged: bool, dt: float = 0.0):
        """Adaptive linear-solver switching, evaluated after each timestep
        attempt (converged or not -- a failed attempt is followed by a dt-cut
        retry, so acting here lets the RETRY run on the fallback solver).

        When ``self.linear_solver`` is an :class:`AdaptiveSolverSpec` its policy is
        evaluated; the policy may return a candidate index (legacy) or a
        :class:`SolverAction` combining a switch with in-place parameter
        updates. A no-op for plain specs, in proprietary builds, and on GPU.
        """
        if not _HAVE_SOLVER_REGISTRY:
            return
        spec = self._resolve_solver_spec()
        if not isinstance(spec, AdaptiveSolverSpec):
            return
        engine = self.physics.engine
        current_index = getattr(self, "_adaptive_solver_index", 0)
        if not timestep_converged:
            self._adaptive_failures = getattr(self, "_adaptive_failures", 0) + 1
        else:
            self._adaptive_failures = 0
        context = self._build_switch_context(timestep_converged, dt)

        # The failure hook fires BEFORE the dt-cut retry; None defers to the
        # regular policy.
        result = None
        if not timestep_converged and spec.on_timestep_failed is not None:
            result = spec.on_timestep_failed(context)
        if result is None:
            result = spec.policy(context)

        if isinstance(result, SolverAction):
            new_index = result.index if result.index is not None else current_index
            updates = result.updates
        else:
            new_index = int(result)
            updates = None
        new_index = max(0, min(new_index, len(spec.candidates) - 1))

        if new_index != current_index:
            self._adaptive_solver_index = new_index
            solver = self.linear_solver
            solver.handle = spec.candidates[new_index].build(self._block_size())
            engine.set_linear_solver(
                solver.handle, _describe_solver_spec(spec.candidates[new_index])
            )
            print(
                f"[adaptive solver] switched to candidate {new_index}: "
                f"{type(spec.candidates[new_index]).__name__}"
            )
        if updates:
            # In-place parameter updates on the (possibly just-selected)
            # candidate -- reconfigures the live solver, no Jacobian impact.
            self.update_solver(**updates)

    def _apply_spec_field_updates(self, target, field_updates):
        """Apply ``field_updates`` to ``target`` (a LinearSolverSpec) or, for
        a GMRES+prec stack, to its ``prec`` spec -- with field-name
        validation. Returns the list of spec objects actually modified."""
        import dataclasses

        top_fields = {f.name for f in dataclasses.fields(target)}
        prec = getattr(target, "prec", None)
        prec_fields = (
            {f.name for f in dataclasses.fields(prec)}
            if isinstance(prec, LinearSolverSpec)
            else set()
        )
        touched = []
        for name, value in field_updates.items():
            if name in top_fields:
                setattr(target, name, value)
                if target not in touched:
                    touched.append(target)
            elif name in prec_fields:
                setattr(prec, name, value)
                if prec not in touched:
                    touched.append(prec)
            else:
                known = sorted(top_fields | prec_fields)
                raise AttributeError(
                    f"update_solver: unknown solver field {name!r} for "
                    f"{type(target).__name__}"
                    + (f" / {type(prec).__name__}" if prec_fields else "")
                    + f"; known fields: {known}"
                )
        return touched

    def update_solver(self, spec=None, **field_updates):
        """Change the linear solver or its parameters DURING a simulation.

        Two modes (combinable):

        * ``update_solver(spec=<LinearSolverSpec>)`` -- switch to a different
          solver. The new solver is built and injected into the live engine;
          ``engine_base::set_linear_solver`` re-inits it against the EXISTING
          Jacobian, so **no matrix reallocation** takes place. Costs one full
          preconditioner setup at the next Newton iteration.
        * ``update_solver(field=value, ...)`` -- update parameters of the
          current solver in place. Fields are resolved on the active spec
          and, for a ``GMRESSolverSpec(prec=...)`` stack, on its ``prec``
          spec. Hot fields (tolerances, iteration/cycle budgets, thresholds,
          weight scheme) take effect at the next solve; warm fields (the CPR
          pressure-AMG profile, ILU fill, the whole MGR configuration)
          trigger an internal hierarchy rebuild on the next setup against the
          same bound matrices. Structural changes (e.g. CPR ``stage2_type``)
          transparently fall back to the switch path.

        The spec on ``self.linear_solver`` stays authoritative: a later ``reset()``
        or solver rebuild reproduces the updated configuration.

        :returns: ``"reconfigured"`` (applied in place) or ``"rebuilt"``
            (fresh solver injected -- still no Jacobian reallocation).
        """
        if getattr(self, "platform", "cpu") == "gpu":
            raise NotImplementedError(
                "update_solver is not available on platform='gpu' yet: GPU "
                "solvers are enum-selected by the engine factory and cannot "
                "be reconfigured or re-injected mid-run."
            )
        if not _HAVE_SOLVER_REGISTRY:
            raise RuntimeError(
                "update_solver requires the open-source solver registry "
                "(darts.linear_solvers) -- unavailable in this build."
            )

        if spec is not None:
            if not isinstance(spec, LinearSolverSpec):
                raise TypeError(
                    f"update_solver(spec=...) expects a LinearSolverSpec, "
                    f"got {type(spec).__name__}"
                )
            if field_updates:
                self._apply_spec_field_updates(spec, field_updates)
            self.linear_solver = spec
            self._sync_linear_params(field_updates, spec)
            self._apply_solver()
            return "rebuilt"

        cur = self._resolve_solver_spec()
        if cur is None:
            raise RuntimeError(
                "update_solver: self.linear_solver holds no LinearSolverSpec to "
                "update (raw solver objects can only be replaced via "
                "update_solver(spec=...))."
            )
        target = cur
        if isinstance(cur, AdaptiveSolverSpec):
            target = cur.candidates[getattr(self, "_adaptive_solver_index", 0)]

        touched = self._apply_spec_field_updates(target, field_updates)
        self._sync_linear_params(field_updates, target)

        # Python-resident solvers (PETSc / Pardiso) are owned by the model;
        # rebuilding them is cheap relative to their solves.
        if isinstance(target, PythonLinearSolverSpec):
            self._apply_solver()
            return "rebuilt"

        solver = getattr(self, "_linear_solver", None)
        if solver is None or not touched:
            return "reconfigured"  # not built yet -- the spec carries it all

        # In-place reconfiguration through the live handle. The outer solver
        # forwards configs it does not recognise to its preconditioner, so
        # both the GMRES fields and the CPR fields of a GMRES+CPR stack are
        # reachable through the single outer handle.
        rc = 0
        for touched_spec in touched:
            try:
                rc = max(rc, int(solver.reconfigure(touched_spec._make_config())))
            except AttributeError:
                rc = 1  # handle without reconfigure (e.g. Python-side stub)
        if rc > 0:
            # Structural change: rebuild + inject on the existing Jacobian.
            self._apply_solver()
            return "rebuilt"
        return "reconfigured"

    def _solver_is_default(self) -> bool:
        """True when ``self.linear_solver`` is the platform default that :meth:`set_solver`
        materialised, i.e. the model never *chose* a linear solver. Identity-compared,
        so a model that calls ``super().set_solver()`` and then assigns its own spec
        counts as having chosen one."""
        default = getattr(self, "_default_solver_obj", None)
        return default is not None and self.linear_solver is default

    def _sync_solver_to_sim_params(self):
        """Mirror the linear-solver settings of ``self.linear_solver`` into ``sim_params``.

        ``self.linear_solver`` (a :class:`LinearSolverSpec`) is the single owner of the linear
        knobs; ``sim_params`` is the C++ mirror the engine reads -- it re-applies
        ``tolerance_linear`` / ``max_i_linear`` to whatever solver it (re-)inits, and
        passes ``linear_print_level`` to the Python-resident solvers. Called by
        :meth:`_apply_solver` -- i.e. after :meth:`set_solver`, before ``engine.init``.

        Models that never chose a solver *and* set
        ``linear_solver_from_engine_factory`` (mechanics / THMC: they drive
        ``engine.ls_params`` + ``params`` directly) are left alone.
        """
        spec = self._resolve_solver_spec()
        if spec is None:
            return  # raw compiled solver handle: it carries its own configuration
        if self._solver_is_default() and self.linear_solver_from_engine_factory:
            return
        # An AdaptiveSolverSpec delegates to the candidate currently in use.
        candidates = getattr(spec, "candidates", None)
        if candidates:
            spec = candidates[getattr(self, "_adaptive_solver_index", 0)]
        self.params.tolerance_linear = spec.tolerance
        self.params.max_i_linear = spec.max_iterations
        if getattr(spec, "print_level", None) is not None:
            self.params.linear_print_level = spec.print_level

    def _sync_linear_params(self, field_updates, spec):
        """Keep the engine-side linear-solve knobs coherent with a spec
        update: params.tolerance_linear / max_i_linear are what the engine
        re-applies to any solver it (re-)inits."""
        if "tolerance" in field_updates:
            self.params.tolerance_linear = field_updates["tolerance"]
        if "max_iterations" in field_updates:
            self.params.max_i_linear = field_updates["max_iterations"]
        # Note: a bare spec switch deliberately does NOT overwrite
        # params.tolerance_linear / max_i_linear -- engine-resident solvers
        # take those from the engine at (re-)init (the documented convention),
        # and silently replacing e.g. a model's 1e-2 with a spec DEFAULT of
        # 1e-5 would change every subsequent solve.

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
        Linear-solver parameters are NOT set here either: the linear solver is
        configured through ``self.linear_solver`` (a
        :class:`darts.linear_solvers.LinearSolver`, tuned via
        ``self.linear_solver.spec.tolerance`` / ``.max_iterations``) in
        :meth:`set_solver`. For one deprecation cycle the removed linear keyword
        arguments (``tol_linear``, ``it_linear``) are accepted on the same terms
        as the nonlinear ones above.

        .. deprecated::
            Set timestep controls on ``self.data_ts``, the nonlinear solver on
            ``self.nonlinear_solver`` and the linear solver on
            ``self.linear_solver`` instead.
        """
        warnings.warn(
            "set_sim_params() is deprecated; set timestep controls on "
            "DartsModel.data_ts and specify DartsModel.nonlinear_solver in set_solver()",
            DeprecationWarning,
            stacklevel=2,
        )
        # nonlinear settings are NOT set here — they live on self.nonlinear_solver.
        # Materialize the defaults WITHOUT re-entering the overridable set_solver()
        # hook (models call set_sim_params() from their set_solver()).
        self._ensure_default_solvers()

        # one-cycle migration: route any legacy solver kwargs onto the specs
        if legacy:
            self._migrate_legacy_solver_kwargs(legacy)

        # fresh timestep-control structure
        self.data_ts = DataTS(self.physics.n_vars)
        ts = self.data_ts

        # Time stepping parameters. if None, default value will be used
        ts.dt_first = first_ts if first_ts is not None else ts.dt_first
        ts.dt_min = min_ts if min_ts is not None else ts.dt_min
        ts.dt_max = max_ts if max_ts is not None else ts.dt_max
        ts.dt_mult = mult_ts if mult_ts is not None else ts.dt_mult

        # NOTE: neither solver's parameters are accepted here -- this method
        # configures time-stepping only. Nonlinear settings live on
        # self.nonlinear_solver.spec (!327), linear settings on
        # self.linear_solver.spec (!280):
        #     super().set_solver()                       # platform default solvers
        #     self.nonlinear_solver.spec.tolerance = 1e-3
        #     self.linear_solver.spec.tolerance = 1e-6
        # Legacy kwargs of either family are mapped for one deprecation cycle by
        # _migrate_legacy_solver_kwargs() above.

        self.runtime = runtime

        self.copy_data_ts_to_sim_params()

    def _migrate_legacy_solver_kwargs(self, legacy: dict):
        """One-deprecation-cycle shim: map removed ``set_sim_params`` solver
        keyword arguments -- nonlinear (!327) and linear (!280) alike -- onto
        ``self.nonlinear_solver.spec`` / ``self.linear_solver.spec`` and warn.
        Unknown keys raise TypeError so genuine typos still fail loudly."""
        from darts.nonlinear_solvers.newton import _ENUM_TO_CHOP_MODE

        spec = self.nonlinear_solver.spec
        handled = []
        # --- linear family (MR280): the spec is the single owner ---
        if "tol_linear" in legacy or "it_linear" in legacy:
            self._ensure_default_solvers()  # materialise self.linear_solver if not chosen yet
            lin = getattr(self.linear_solver, "spec", None)
            if lin is None:
                raise RuntimeError(
                    "set_sim_params(tol_linear=/it_linear=) cannot be migrated: "
                    "self.linear_solver holds no LinearSolverSpec (raw handle)."
                )
            if "tol_linear" in legacy:
                lin.tolerance = legacy.pop("tol_linear")
                handled.append("tol_linear -> linear_solver.spec.tolerance")
            if "it_linear" in legacy:
                lin.max_iterations = legacy.pop("it_linear")
                handled.append("it_linear -> linear_solver.spec.max_iterations")
        if "line_search" in legacy:
            # !327 removed the line-search solver entirely; accept and ignore.
            legacy.pop("line_search")
            handled.append("line_search -> removed (no longer supported)")
        if "tol_newton" in legacy:
            spec.tolerance = legacy.pop("tol_newton")
            handled.append("tol_newton -> nonlinear_solver.spec.tolerance")
        if "it_newton" in legacy:
            spec.max_iterations = legacy.pop("it_newton")
            handled.append("it_newton -> nonlinear_solver.spec.max_iterations")
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
            # Accept the legacy int (0/1/2), the compiled sim_params.newton_solver_t
            # enum (int-CONVERTIBLE but not an int instance -- isinstance(nt, int)
            # is False for a pybind enum), the mode string, or None.
            if nt is not None and not isinstance(nt, str):
                try:
                    nt = int(nt)
                except (TypeError, ValueError):
                    pass
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
            "set_sim_params() solver keyword arguments are removed; mapped for "
            "this release only (" + "; ".join(handled) + "). Migrate to "
            "self.nonlinear_solver = NewtonSolver(...) / self.linear_solver = "
            "<LinearSolverSpec> in set_solver().",
            DeprecationWarning,
            stacklevel=3,
        )

    def copy_data_ts_to_sim_params(self):
        """No-op retained for compatibility: nothing is mirrored into ``sim_params``
        any more.

        * timestep controls live on ``data_ts`` and are read directly by :meth:`run`;
        * nonlinear settings are synced into the engine by the nonlinear solver
          (``NonlinearSolverSpec.sync_to_engine``, !327);
        * linear settings are mirrored by :meth:`_sync_solver_to_sim_params` from
          ``self.linear_solver.spec`` before ``engine.init()`` (!280).

        The corresponding C++ ``sim_params`` fields (``first_ts``/``max_ts``/
        ``mult_ts``/``tolerance_newton``/``max_i_newton``) no longer exist.
        """

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
            self._maybe_switch_linear_solver(converged, dt=dt)

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
            print(
                f'linear solver is {self.linear_solver} (params.linear_type={self.params.linear_type})'
            )

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

        Returns ``(rc, n_iters, residual)`` for every backend -- ``rc`` is ``0``
        on success, ``1`` on setup failure, ``2`` on solve failure -- so the
        nonlinear driver stays backend-agnostic (!327 contract).

        Two solver kinds are dispatched here (!280 spec-driven routing, which
        replaced the former ``data_ts.linear_type`` enum dispatch):

        * a Python-resident solver built from a
          :class:`~darts.linear_solvers.specs.PythonLinearSolverSpec` (PETSc /
          Pardiso) selected through ``self.linear_solver``;
        * the C++ engine solver (the default path).
        """
        python_solver = getattr(self, "_python_solver", None)
        if python_solver is not None:
            # Python-resident solver (PETSc / Pardiso); stateful, it performs
            # its one-time setup on the first call. A nonzero rc means a hard
            # failure (non-finite solution) -- the nonlinear solver records it
            # in NonlinearStatus.linear_solver_rc and aborts the Newton loop.
            return python_solver.solve_system(self.physics.engine)
        # C++ linear solver held by the engine
        engine = self.physics.engine
        rc = engine.solve_linear_equation()
        return rc, engine.get_last_linear_iters(), engine.get_last_linear_residual()
