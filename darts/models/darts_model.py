import os
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
from darts.linear_solvers import (
    AMGXCPRSolverSpec,
    CPRSolverSpec,
    GMRESSolverSpec,
    LinearSolver,
)
from darts.models.output import Output
from darts.nonlinear_solvers import ChopSpec, NewtonSolver, Norm, OBLBoundsSpec
from darts.print_build_info import print_build_info as package_pbi
from darts.timestep_control import TimestepControl


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
    #: ``ts_control is None``; ``ts_control`` now always exists as a plain member, so
    #: the intent is stated explicitly here.)
    linear_solver_from_engine_factory = False

    #: Whether this model declares OBL history-state fields (e.g. ``sg_max`` for
    #: Killough hysteresis) on its physics. Off by default so existing models that
    #: never touch ``history_fields`` are unaffected; a model that wants hysteresis
    #: support overrides this -- e.g. as a constructor parameter it assigns to
    #: ``self.hysteresis`` -- before calling :meth:`set_physics`, and passes
    #: ``history_fields=[...] if self.hysteresis else []`` into ``PhysicsBase``.
    hysteresis = False

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

        # The single source of truth for the linear solver: a composed LinearSolver instance,
        # created once here and never reassigned afterward (mirrors nonlinear_solver's binding
        # pattern, except LinearSolver takes its model at construction, so no separate bind()
        # step is needed). It owns ALL linear-solver settings:
        # - self.linear_solver.spec: solver + preconditioner choice
        # - tolerance, max_iterations, print_level)
        # Its own _apply_solver() builds, injects and mirrors into sim_params before engine.init(),
        # plus the solver-binding/orchestration and deprecated set_sim_params() methods.
        # The platform-default spec is materialized lazily by set_solver(); a model may instead
        # assign its own spec before/after calling set_solver() via
        # self.linear_solver.spec = <LinearSolverSpec>.
        self.linear_solver = LinearSolver(model=self)

        # Nonlinear solver instance (a NewtonSolver; see darts.nonlinear_solvers)
        # built from its declarative spec. Assigned lazily by set_solver() and
        # bound to this model in init(). Its input spec is DartsModel
        # .nonlinear_solver.spec (retrievable for tracing/serialization).
        self.nonlinear_solver = None

        # Timestep-control structure (see darts.timestep_control.TimestepControl): a plain
        # settings holder, unlike the two solvers above it has no bind/build step,
        # so it's just a regular member -- constructed here with n_vars=0 and
        # resized to the physics' actual n_vars by init() (physics doesn't exist
        # yet at this point). Read/write directly,
        # e.g. self.ts_control.dt_first = ..., self.ts_control.runtime = ....
        self.ts_control = TimestepControl()

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

        # Materialize the solvers (and default specs/ts_control if the model did
        # not configure them) before the engine is initialized.
        # ts_control was constructed in __init__ with n_vars=0 (physics did not exist
        # yet), so eta is still empty. Size it BEFORE set_solver(): overrides tune
        # individual degrees of freedom (self.ts_control.eta[i] = ...), which needs
        # the array to have its final length already.
        self.ts_control.resize(self.physics.n_vars)
        self.set_solver()
        # ...and again afterwards, in case set_solver() replaced ts_control outright
        # (the mechanics models install one from their idata).
        self.ts_control.resize(self.physics.n_vars)
        # fail loudly on an obviously-broken timestepping config, matching the
        # per-timestep spec.validate() the Newton loop already does
        self.ts_control.validate()

        # bind the (possibly detached) solver to this model
        self.nonlinear_solver.bind(self)

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
        self.ts_control.print()
        self.linear_solver._warn_if_direct_solver_oversized()

        init_timer.stop()

    def reset(self):
        """
        Configure the solver/time-stepping via set_solver(), then initialize the engine.

        set_solver() runs first -- the reservoir/mesh and the engine object already
        exist (so block sizes and n_res_blocks are final), but engine.init() has not
        run yet. So any timestepping/Newton/linear params it sets feed engine.init().

        The linear solver (``self.linear_solver``, a runtime
        :class:`darts.linear_solvers.LinearSolver` instance whose declarative spec is
        ``linear_solver.spec``; the default spec is materialized in :meth:`set_solver`) is
        always already bound to this model (constructed once in ``__init__``) and is
        built and injected by :meth:`_apply_solver` before ``engine.init``, so the
        engine adopts its ``handle`` and bypasses its own factory. The nonlinear
        solver is (re)bound here too: ``set_solver()`` runs a second time (the first
        was in ``init()``, right after the reservoir/mesh and engine object exist),
        and a model's override may unconditionally
        reassign ``self.nonlinear_solver`` on every call (no existing-instance
        guard), leaving a fresh, unbound instance otherwise -- unlike
        ``linear_solver``, whose constructor takes the model directly, so
        reassigning it (e.g. ``self.linear_solver.spec = ...``, or replacing it
        outright with ``model=self``) never leaves it unbound. In proprietary / GPU
        builds no linear backend is built and the engine factory selects the solver
        from ``params.linear_type``.
        """
        self.set_solver()
        self.nonlinear_solver.bind(self)
        self.linear_solver._apply_solver()
        self.physics.engine.init(
            self.reservoir.mesh,
            ms_well_vector(self.reservoir.wells),
            op_vector(self.op_list),
            self.physics.thermal_var_itor,
            self.params,
            self.timer.node["simulation"],
        )

    def set_solver(self):
        """Configure the model's solvers and time-stepping (override hook).

        This is the single per-model place to declare all time-stepping,
        nonlinear-solver and linear-solver settings. It is called at the start of
        :meth:`reset` (after the reservoir/mesh and engine object exist, before
        ``engine.init``), so it may freely:

        * set ``self.ts_control.dt_first`` / ``.dt_mult`` / ``.dt_max`` / ``.runtime``
          etc. (time-stepping only);
        * set ``self.nonlinear_solver = <NonlinearSolver>`` (a
          :class:`~darts.nonlinear_solvers.NewtonSolver`, which accepts a
          ``NewtonSpec`` positionally or its keyword arguments);
        * set ``self.linear_solver.spec = <LinearSolverSpec>`` -- the **build-safe**
          way to pick a solver (``SuperLUSolverSpec``,
          ``GMRESSolverSpec(prec=CPRSolverSpec())``, ``MGRSolverSpec``,
          ``AdaptiveSolverSpec([...])``, ...). In proprietary / GPU builds no
          backend is built and the engine factory uses ``params.linear_type``, so a
          spec is safe in any build;
        * for fine control a model may still build a raw C++ solver object and assign
          it directly to ``self.linear_solver.handle`` (e.g.
          ``linear_solvers.create_mgr_solver_for_block_size(...)``), valid only in the
          open-source build (guard with
          :meth:`~darts.linear_solvers.LinearSolver.open_source_solvers_available`);
          ``self.linear_solver.label`` names it in the log.

        ``self.linear_solver`` (a :class:`darts.linear_solvers.LinearSolver`, mirror
        of ``nonlinear_solver`` holding a ``NewtonSolver``) is constructed once in
        ``__init__`` and never reassigned, so it always exists and is already bound
        to this model -- unlike ``nonlinear_solver``, whose constructor takes no
        ``model`` argument. Only its declarative ``spec`` is left unset until this
        method's own default-construction below, or a subclass assigning
        ``self.linear_solver.spec = <LinearSolverSpec>``.

        The default implementation is idempotent and lazy: it keeps any spec a
        subclass already assigned and otherwise materializes the defaults below --
        which spell out **every** default parameter explicitly, so the effective
        configuration of a model that does not override it is readable here instead
        of being hidden in the spec dataclass defaults.

        Override in a model either by replacing a solver::

            def set_solver(self):
                super().set_solver()
                self.nonlinear_solver = NewtonSolver(tolerance=1e-4,
                                                     chop=ChopSpec(mode='global'))
                self.linear_solver.spec = MGRSolverSpec(tolerance=1e-4)

        or by tuning the spec of the default::

            def set_solver(self):
                self.ts_control.dt_first = ...                  # time-stepping (not restricted to self.set_solver())
                self.ts_control.dt_max = ...
                super().set_solver()                            # default solvers
                self.nonlinear_solver.spec.tolerance = 1e-4
                self.linear_solver.spec.tolerance = 1e-6

        or by assigning the whole spec up front, skipping the platform default::

            def set_solver(self):
                self.linear_solver.spec = MGRSolverSpec(tolerance=1e-4)
                super().set_solver()
        """
        # ------------------------------------------------------------ nonlinear
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
                on_linear_nonconvergence="accept",  # non-converged linear solve with a
                # usable iterate: 'accept' the inexact-Newton step (historical
                # FGMRES+CPR behaviour) or 'cut' the timestep (historical MGR)
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

        # --------------------------------------------------------------- linear
        # self.linear_solver is constructed once in __init__ and never reassigned,
        # so it's always bound to this model here already
        # only the platform default spec is still materialized lazily, below.

        # Platform default with every parameter stated explicitly (mirrors the spec
        # dataclass field defaults — keep the two in sync). _default_spec marks
        # this solver as unchosen, so _apply_solver() leaves the engine factory in
        # charge for proprietary builds and models with
        # linear_solver_from_engine_factory = True (mechanics / THMC).
        if self.linear_solver.spec is None:
            if getattr(self, "platform", "cpu") == "gpu":
                # GPU default: GMRES + AMGX-CPR (AMGX on the pressure subsystem +
                # ILU on the full system). A GPUSolverSpec builds no C++ solver: it
                # names the params.linear_type enum (gpu_gmres_cpr_amgx_ilu) the GPU
                # engine factory consumes. The AMG configuration lives in the engine
                # factory / AMGX JSON, so the only Python knobs are the ones below.
                spec = AMGXCPRSolverSpec(
                    tolerance=1e-5,  # linear residual tolerance
                    max_iterations=50,  # max Krylov iterations per solve
                    print_level=0,  # solver verbosity
                    proprietary_linear_type=None,  # enum for non-registry builds
                    # Cell-local (kinetic, no flux/diffusion term) equations are
                    # auto-detected by init_physics() (see
                    # PropertyContainer.schur_eliminable_comp_idxs()) and, by this
                    # default schur_elim_kinetic=True, Schur-eliminated by
                    # LinearSolver._apply_gpu_solver(). Override with
                    # self.linear_solver.spec.schur_elim_kinetic = False to disable.
                )
            else:
                # CPU default: FGMRES around the two-stage CPR preconditioner
                # (HYPRE BoomerAMG on the pressure subsystem + ILU(0) on the full
                # system). Unlike the GPU spec, CPR is built from Python through the
                # solver registry, so every BoomerAMG knob is a settable field below.
                spec = GMRESSolverSpec(
                    tolerance=1e-5,  # linear residual tolerance
                    max_iterations=50,  # max Krylov iterations per solve
                    print_level=0,  # solver verbosity
                    proprietary_linear_type=None,  # enum for non-registry builds
                    restart=50,  # FGMRES restart (Krylov subspace dimension)
                    prec=CPRSolverSpec(
                        tolerance=1e-5,  # unused: CPR runs as a preconditioner
                        max_iterations=50,  # unused: single application per solve
                        print_level=0,  # preconditioner verbosity
                        proprietary_linear_type=None,
                        amg_max_iters=1,  # AMG V-cycles on the pressure stage
                        ilu_fill_level=0,  # ILU(0) on the full system (stage 2)
                        weight_scheme=1,  # 1 = True-IMPES pressure weights
                        stage2_type=1,  # 1 = ILU second stage
                        eager_adjoint=False,  # build the transpose stack up front
                        # --- HYPRE BoomerAMG configuration of the pressure stage ---
                        amg_coarsen_type=8,  # PMIS coarsening
                        amg_interp_type=8,  # extended+i interpolation
                        amg_relax_type=3,  # hybrid Gauss-Seidel smoother
                        amg_relax_order=1,  # C/F relaxation ordering
                        amg_num_sweeps=1,  # smoother sweeps per level
                        amg_strong_threshold=0.75,  # strength-of-connection threshold
                        amg_agg_num_levels=0,  # aggressive-coarsening levels
                        amg_agg_interp_type=6,  # interpolation on aggressive levels
                        amg_agg_pmax_elmts=20,  # max elements/row, aggressive levels
                        amg_pmax_elmts=0,  # max elements/row (0 = unlimited)
                        amg_trunc_factor=0.0,  # interpolation truncation factor
                        amg_max_levels=-1,  # max levels (-1 = HYPRE default)
                        amg_cycle_type=-1,  # cycle type (-1 = HYPRE default, V)
                        amg_max_coarse_size=100,  # stop coarsening below this size
                        amg_coarse_relax_type=9,  # Gaussian elimination on the coarsest level
                        amg_relax_wt=-1.0,  # relaxation weight (-1 = HYPRE default)
                        # --- hierarchy reuse across Newton iterations ---
                        reuse_amg_hierarchy=False,  # reuse the AMG setup
                        adaptive_amg_rebuild=False,  # rebuild when iterations degrade
                        adaptive_iter_threshold=15,  # LI above which to rebuild
                        adaptive_consecutive_bad=2,  # bad solves before rebuilding
                    ),
                )
            self.linear_solver.spec = spec
            self.linear_solver._default_spec = spec

        # Deprecated set_sim_params(tol_newton=/tol_linear=...) kwargs deferred from
        # before the solvers existed: apply them now that both specs are materialized.
        # Model tuning after super().set_solver() runs later and still wins.
        pending = self.__dict__.pop("_pending_legacy_solver_kwargs", None)
        if pending:
            self.linear_solver._migrate_legacy_solver_kwargs(pending)

    def initialize_history_fields(self):
        """Seed ``engine.Xhistory`` with the per-field default value for every reservoir cell.

        OBL history state is off by default (``physics.has_history is False``), so for almost
        every model this returns immediately -- the engine then also has
        ``n_history_runtime == 0`` and no ``Xhistory`` buffer. Called by :meth:`init` right after
        :meth:`reset`, which is where the C++ engine allocates ``Xhistory``.

        :returns: None
        """
        if not self.physics.has_history:
            return

        n_blocks = self.reservoir.mesh.n_blocks
        for field in self.physics.history.fields:
            self.physics.history.set_engine_history_array(
                self.physics.engine,
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
        per cell, and write it back via :meth:`HistoryStateSupport.set_engine_history_array`
        (``self.physics.history.set_engine_history_array(...)``, or by mutating the
        underlying ``engine.Xhistory`` vector directly).

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
        # set_initial_conditions_from_array; OBL history columns (self.physics.history.fields)
        # go through history.set_engine_history_array so sg_max and friends survive restart.
        # history.fields is empty unless the physics opted into history state
        # (physics.has_history), so this set is empty for almost every model.
        primary_names = list(self.physics.vars)
        history_labels = {h.label for h in self.physics.history.fields}

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
            self.physics.history.set_engine_history_array(
                self.physics.engine,
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

    def print_config(self):
        """Print the effective solver configuration in one place — timestepping
        (``ts_control``) + nonlinear (``nonlinear_solver.spec``) + linear
        (``linear_solver.spec``) — using the same ``to_dict()`` serialization each
        family exposes. Available after ``set_solver()`` / ``init()``."""
        print("=== Effective configuration ===")
        print("Timestepping (ts_control):")
        for k, v in self.ts_control.to_dict().items():
            print(f"\t{k} = {v}")
        ns = getattr(self, "nonlinear_solver", None)
        ns_spec = getattr(ns, "spec", None) if ns is not None else None
        if ns_spec is not None and hasattr(ns_spec, "to_dict"):
            print("Nonlinear solver (nonlinear_solver.spec):")
            for k, v in ns_spec.to_dict().items():
                print(f"\t{k} = {v}")
        ls = getattr(self, "linear_solver", None)
        ls_spec = getattr(ls, "spec", None) if ls is not None else None
        if ls_spec is not None and hasattr(ls_spec, "to_dict"):
            print("Linear solver (linear_solver.spec):")
            for k, v in ls_spec.to_dict().items():
                print(f"\t{k} = {v}")

    def run_simple(self, physics, ts_control, days, restart_dt=0.0):
        """Removed. Use :meth:`run` after configuring the model normally.

        ``run_simple()`` re-assigned ``self.physics`` / ``self.ts_control`` from its
        arguments, which a run method must not do. Configure the model (physics,
        ``ts_control``, ``set_solver()``) and call ``run(days)`` instead.

        .. deprecated::
            Scheduled for deletion after one deprecation cycle.
        """
        raise NotImplementedError(
            "DartsModel.run_simple() was removed: it re-wired the model from its "
            "arguments. Configure the model and call run(days) instead."
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

        days = days if days is not None else self.ts_control.runtime
        assert days > 0, "Time must be a positive value!"

        ts_control = self.ts_control

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
            dt = min(ts_control.dt_first, days)
        elif restart_dt > 0.0:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * ts_control.dt_mult, days, ts_control.dt_max)

        self.prev_dt = dt

        nc = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        max_dx = np.zeros(nc)

        if np.fabs(ts_control.dt_mult - 1) < 1e-10:
            omega = 0.0
        else:
            # inversion assuming mult = (1 + omega) / omega
            omega = 1 / (ts_control.dt_mult - 1)

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
            self.linear_solver._maybe_switch_linear_solver(converged, dt=dt)

            overhead.start()
            if converged:
                t += dt
                self.physics.engine.t = t
                ts_counter += 1
                self.after_converged_timestep()

                x = np.array(self.physics.engine.X, copy=False)[: nb * nc]
                dt_mult_new = ts_control.dt_mult
                for i in range(nc):
                    max_dx[i] = np.max(abs(xn[i::nc] - x[i::nc]))
                    mult = ((1 + omega) * ts_control.eta[i]) / (
                        max_dx[i] + omega * ts_control.eta[i]
                    )
                    if mult < dt_mult_new:
                        dt_mult_new = mult

                if verbose:
                    max_dx_str = '[' + ', '.join(f'{v:.1e}' for v in max_dx) + ']'
                    print(
                        f"#{ts_counter:d}\tT={t:3g}\tDT={dt:2g}\tNI={self.nonlinear_solver.status.n_newton:d}\tLI={self.nonlinear_solver.status.n_linear:d}\tDT_MULT={dt_mult_new:3.3g}\tdX={max_dx_str}"
                    )

                dt = min(dt * dt_mult_new, ts_control.dt_max)

                if np.fabs(t + dt - stop_time) < ts_control.dt_min:
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
                dt /= ts_control.dt_mult
                if verbose:
                    print(f"Cut timestep to {dt:2.10f}")
                if dt <= ts_control.dt_min:
                    overhead.stop()  # keep the bracket balanced before the assert aborts
                assert dt > ts_control.dt_min, (
                    "Stop simulation. Reason: reached min. timestep "
                    + str(ts_control.dt_min)
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
        return self.nonlinear_solver.solve_timestep(dt, t, verbose)

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

    def do_after_step(self):
        """
        Hook for per-report-step actions (e.g. reporting, saving); can be
        overridden by a user to be executed in run_simulation().
        """
        pass

    def run_simulation(self):
        """
        Run the reporting loop over idata.sim.time_steps: for every report
        step, apply the idata well controls (well control switch can be defined in idata),
        and invoke the do_after_step hook.

        :return: 0 on success, 1 if :meth:`run` failed on some step
        :rtype: int
        """
        # simulation time at the START of the current report step, [days]
        time = 0.0
        # idata.sim.time_steps holds report-step LENGTHS, not absolute times
        for ith_step, dt in enumerate(self.idata.sim.time_steps):
            # apply the well controls scheduled for this point in time
            self.set_well_controls_idata(time=time)

            # advance by one report step -- run() drives its own adaptive
            # timestepping inside dt and returns non-zero if it could not finish
            ret = self.run(dt)
            if ret != 0:
                print("run() failed for the step=", ith_step, "dt=", dt)
                return 1

            # per-report-step user hook (reporting, saving, ...)
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
