"""Linear-solver binding of :class:`~darts.models.darts_model.DartsModel`.

Everything that connects a model to its linear solver lives here, extracted
from ``darts_model.py`` (review feedback: the base model was overcrowded with
solver plumbing). :class:`LinearSolverBinding` is a mixin consumed by
``DartsModel`` -- it defines no state of its own beyond what ``DartsModel``
initializes, and every method is unchanged from its previous in-class form.

Scope:

* the ``linear_solver`` property (normalizing setter: spec / runtime
  ``LinearSolver`` / raw compiled handle) and its backing helpers;
* backend materialization at ``reset()``/``init()`` time (``_apply_solver`` /
  ``_apply_gpu_solver``) and the ``sim_params`` mirroring for the
  proprietary / GPU engine factories;
* (``set_solver()`` itself -- the user-facing override hook that constructs
  the explicit platform defaults -- stays on ``DartsModel`` for readability;
  this mixin only consumes it);
* mid-run reconfiguration (``update_solver``) and the adaptive per-timestep
  switching driven by :class:`~darts.linear_solvers.AdaptiveSolverSpec`;
* the linear-solve entry points used by the nonlinear loop
  (``_solve_linear_equation``, ``get_linear_system``).
"""

import warnings

import numpy as np

from darts.engines import sim_params

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
    )
    from darts.linear_solvers.solver import (
        is_compiled_solver_handle as _is_compiled_solver_handle,
    )
    from darts.linear_solvers.specs import (
        CPRSolverSpec,
        GMRESSolverSpec,
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


class LinearSolverBinding:
    """Mixin holding the linear-solver side of ``DartsModel`` (see module docstring)."""

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
            # The model's set_solver() override never materialized one: apply the
            # BASE defaults explicitly (not self.set_solver(), which would re-enter
            # the override that just declined to set it). set_solver lives on
            # DartsModel (the user-facing override hook); import lazily to avoid
            # a module cycle.
            from darts.models.darts_model import DartsModel

            DartsModel.set_solver(self)
            solver = self.linear_solver
            spec = self._resolve_solver_spec()
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
            # engine.init() requires a C++ solver; inject the CPU default stack as a
            # placeholder -- it is constructed but never used at solve time.
            solver.handle = GMRESSolverSpec(restart=50, prec=CPRSolverSpec()).build(
                block_size
            )
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
