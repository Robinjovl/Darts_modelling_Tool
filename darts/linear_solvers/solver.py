"""Runtime linear solver: the instance ``DartsModel.linear_solver`` holds.

Mirrors the runtime-solver design of :mod:`darts.nonlinear_solvers` (!327):
``DartsModel.nonlinear_solver`` holds a detached :class:`NewtonSolver` instance that
binds to the model during ``init()``; :class:`LinearSolver` is the linear twin, except
that it takes its model at construction instead of binding later. It needs no physics /
engine at that point (only the back-reference), holds the
declarative :attr:`spec` (a :class:`~darts.linear_solvers.specs.LinearSolverSpec`),
and materializes its computational backend when it is bound during ``reset()`` -- the
point where the matrix block size and the engine object exist (before
``engine.init()``, which adopts the injected backend).

Unlike the nonlinear side, where the *method* identity lives in the runtime class
(``NewtonSolver`` vs ``MechanicsNewtonSolver``), the linear method identity lives in
the **spec** (``MGRSolverSpec``, ``GMRESSolverSpec(prec=CPRSolverSpec())``, ...); one
uniform runtime wrapper therefore serves every backend:

- ``handle`` -- the built C++ solver injected into the engine (open-source CPU
  builds; ``None`` elsewhere);
- ``python_solver`` -- the Python-resident PETSc / Pardiso backend when the spec is
  a :class:`~darts.linear_solvers.specs.PythonLinearSolverSpec` (``None`` otherwise);
- neither on GPU / proprietary builds, where the engine factory constructs the
  solver internally from ``params.linear_type`` (the spec still drives the enum and
  the tolerances mirrored into ``sim_params``).

``self`` (composition, not inheritance) -- this class also owns every method that
connects a model to its linear solver: created once in ``DartsModel.__init__`` as
``self.linear_solver = LinearSolver(model=self)`` and never reassigned afterward, it
is mutated in place (``self.linear_solver.spec = <LinearSolverSpec>``, mirroring the
``self.nonlinear_solver.spec.tolerance = ...`` tuning convention already used for the
nonlinear side). Scope:

* backend materialization at ``reset()``/``init()`` time (``_apply_solver`` /
  ``_apply_gpu_solver``) and the ``sim_params`` mirroring for the proprietary / GPU
  engine factories;
* mid-run reconfiguration (``update_solver``) and the adaptive per-timestep
  switching driven by :class:`~darts.linear_solvers.AdaptiveSolverSpec`;
* the linear-solve entry points used by the nonlinear loop
  (``_solve_linear_equation``, ``get_linear_system``);
* the one-deprecation-cycle mapping of removed nonlinear/linear keyword arguments
  (previously reachable via the now-removed ``set_sim_params`` shim) onto the
  solver specs (``_migrate_legacy_solver_kwargs``, still invoked from
  :meth:`darts.timestep_control.TimestepControl.set_sim_params`) -- scheduled for
  removal when the deprecation cycle ends.

(``set_solver()`` itself -- the user-facing override hook that constructs the
explicit platform defaults -- stays on ``DartsModel`` for readability; this class only
consumes it.)
"""

import warnings

import numpy as np

from darts.engines import sim_params
from darts.linear_solvers.specs import LinearSolverSpec

# Open-source linear-solver registry (the darts.linear_solvers package). It is absent
# in proprietary (-a / -b) builds, where the engine's built-in factory selects
# the solver from params.linear_type; the import is therefore guarded.
# The Python helpers (specs, adaptive policy) still import in proprietary
# builds, but their build() calls reach into the compiled extension --
# guard on _have_compiled_solvers, not just the import succeeding.
#
# Imported from their defining submodules (not the darts.linear_solvers package
# itself): this module is imported BY darts/linear_solvers/__init__.py partway
# through building that package's namespace, so `from darts.linear_solvers import
# ...` here would see a partially-initialized package and spuriously ImportError
# on names its __init__ has not reached yet (e.g. PythonLinearSolverSpec, bound
# only after the .solver import further down __init__.py).
try:
    from darts import linear_solvers as _darts_solvers_pkg
    from darts.linear_solvers.adaptive import (
        AdaptiveSolverSpec,
        SolverAction,
        SolverSwitchContext,
    )
    from darts.linear_solvers.specs import (
        CPRSolverSpec,
        GMRESSolverSpec,
        PythonLinearSolverSpec,
    )

    _HAVE_SOLVER_REGISTRY = getattr(_darts_solvers_pkg, "_have_compiled_solvers", True)
except ImportError:  # proprietary build without the open-source solvers
    _HAVE_SOLVER_REGISTRY = False


def is_compiled_solver_handle(obj) -> bool:
    """True when ``obj`` is a raw solver object from the compiled
    ``darts.linear_solvers.linear_solvers`` extension (the handle returned by
    ``spec.build()``, an ``MGRSolver_N`` facade, ...) -- i.e. something
    ``engine.set_linear_solver`` accepts directly. Compiled handles expose no common
    Python-visible solve API (their virtuals live on the C++ side), so the
    discriminator is the defining module. A duck-typed ``setup``/``solve`` pair is
    accepted as an escape hatch for Python-implemented handles."""
    if type(obj).__module__ == "darts.linear_solvers.linear_solvers":
        return True
    return hasattr(obj, "setup") and hasattr(obj, "solve")


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


class LinearSolver:
    """Runtime linear solver bound to a model at ``reset()``/``init()`` time.

    Near-mirror of :class:`darts.nonlinear_solvers.NonlinearSolver`: the declarative
    configuration stays retrievable as :attr:`spec` (serializable via
    ``spec.to_dict()``). Unlike the nonlinear solver it takes its ``model`` at
    construction (``DartsModel.__init__`` composes it as
    ``LinearSolver(model=self)``) and is never reassigned, so no separate
    :meth:`bind` step is needed before the backend is materialized.

    :param spec: the :class:`LinearSolverSpec` to run. ``None`` until the model's
        ``set_solver()`` materializes the platform default (or a raw handle is
        assigned to :attr:`handle` directly -- the fine-control path).
    :param model: the :class:`~darts.models.darts_model.DartsModel` this solver is
        composed onto. Set once by ``DartsModel.__init__``.
    """

    # Slots make stale pre-instance idioms fail loudly: assigning
    # solver.tolerance = 1e-6 (the old spec-attribute form) raises
    # AttributeError instead of silently parking the value on the wrapper --
    # the knob belongs on solver.spec.tolerance.
    __slots__ = (
        "spec",
        "model",
        "handle",
        "python_solver",
        "label",
        "_default_spec",
        "_adaptive_solver_index",
        "_adaptive_failures",
        "_ls_setup_time_prev",
        "_ls_solve_time_prev",
    )

    def __init__(self, spec: LinearSolverSpec = None, model=None):
        if spec is not None and not isinstance(spec, LinearSolverSpec):
            raise TypeError(
                f"LinearSolver expects a LinearSolverSpec, got {type(spec).__name__}"
            )
        self.spec = spec
        self.model = model
        #: built C++ solver handle injected into the engine (open-source CPU
        #: builds; None on GPU / proprietary builds and before _apply_solver)
        self.handle = None
        #: Python-resident PETSc / Pardiso backend (None unless the spec is a
        #: PythonLinearSolverSpec)
        self.python_solver = None
        #: optional human-readable label for the engine log (raw-handle path)
        self.label = None
        #: identity marker: the spec set_solver() materialised as the platform
        #: default, so that a model which calls super().set_solver() and then
        #: replaces self.linear_solver.spec still counts as having *chosen* one
        self._default_spec = None
        self._adaptive_solver_index = 0
        self._adaptive_failures = 0
        self._ls_setup_time_prev = 0.0
        self._ls_solve_time_prev = 0.0

    # ------------------------------------------------------------ construction

    @classmethod
    def from_handle(cls, raw, label: str = None, model=None) -> "LinearSolver":
        """Wrap a raw compiled solver handle built outside the spec API (the
        documented fine-control path, e.g.
        ``linear_solvers.create_mgr_solver_for_block_size(...)``). The wrapper
        has no :attr:`spec`; the handle is injected as-is.

        Pass ``model=`` when the result is assigned to ``DartsModel.linear_solver``:
        the composed instance always carries its model (there is no separate bind
        step), so a replacement must carry it too."""
        if not is_compiled_solver_handle(raw):
            raise TypeError(
                "LinearSolver.from_handle expects a compiled darts.linear_solvers "
                f"solver handle, got {type(raw).__name__}"
            )
        solver = cls(model=model)
        solver.handle = raw
        solver.label = label
        return solver

    # ------------------------------------------------------------ binding

    def bind(self, model) -> "LinearSolver":
        """Re-point this solver at ``model``; returns self.

        Not needed on the normal path -- ``DartsModel.__init__`` composes the
        instance with ``model=self`` and nothing reassigns it. Kept for the
        standalone case (a solver built outside a model, then attached) and for
        symmetry with :meth:`darts.nonlinear_solvers.NonlinearSolver.bind`."""
        self.model = model
        return self

    # ------------------------------------------------------------ introspection

    def to_spec(self) -> LinearSolverSpec:
        """Return the specification this solver runs (mirror of
        ``NonlinearSolver.to_spec``); serializable via ``to_spec().to_dict()``.
        ``None`` for a raw-handle wrapper (:meth:`from_handle`)."""
        return self.spec

    def __repr__(self):
        if self.spec is not None:
            head = type(self.spec).__name__
        elif self.handle is not None:
            head = "raw-handle"
        else:
            head = "unconfigured"
        parts = [head]
        if self.handle is not None:
            parts.append("built")
        if self.python_solver is not None:
            parts.append("python-resident")
        return f"LinearSolver({', '.join(parts)})"

    # ------------------------------------------------------------ model binding (former LinearSolverBinding mixin)

    def _resolve_solver_spec(self):
        """Return the effective linear-solver spec: :attr:`spec` when it is a
        :class:`LinearSolverSpec` (the single source of truth), else ``None``.

        Shared by :meth:`_apply_solver` and :meth:`_maybe_switch_linear_solver` so
        both see the same solver. A ``None`` result means "no spec is available"
        (nothing assigned yet, or a raw-handle wrapper).
        """
        return self.spec if isinstance(self.spec, LinearSolverSpec) else None

    def _warn_if_direct_solver_oversized(self):
        """Warn when a direct (SuperLU) solver is selected on a mesh too large for
        it to scale -- called once by ``DartsModel.init()`` after the mesh is
        built and the solver is resolved. ``stacklevel=3`` skips over this method
        and ``init()`` to attribute the warning to ``init()``'s caller."""
        model = self.model
        is_superlu = (
            model.params.linear_type == sim_params.linear_solver_t.cpu_superlu
            or type(self._resolve_solver_spec()).__name__ == "SuperLUSolverSpec"
        )
        if is_superlu and model.reservoir.mesh.n_res_blocks > 30000:
            warnings.warn(
                "The number of cells looks too big to use a direct linear solver: "
                + str(model.reservoir.mesh.n_res_blocks)
                + ' > 30000',
                stacklevel=3,
            )

    def _block_size(self) -> int:
        """Matrix block size for building a solver: ``engine.N_VARS`` for mechanics
        engines (it includes the displacement DOFs that ``physics.n_vars`` does not
        count), else ``physics.n_vars`` (pressure + n_components - 1 for flow)."""
        engine = getattr(self.model.physics, "engine", None)
        if engine is not None and hasattr(engine, "N_VARS"):
            return engine.N_VARS
        return self.model.physics.n_vars

    def _apply_solver(self):
        """Build the linear solver from ``self`` and inject it into the engine.

        Replaces the former ``_apply_linear_solver_spec``. Called once by
        ``DartsModel.reset`` before ``engine.init``: the resolved
        :class:`LinearSolverSpec` (``self.spec``, or the platform default for models
        that only configure time-stepping) is built and injected, so ``engine.init``
        adopts it and bypasses its own factory. No-op in proprietary / GPU builds and
        for models that set ``linear_solver_from_engine_factory`` without choosing a
        spec (mechanics / THMC) -- there the engine factory / ``params.linear_type``
        selects the solver.
        """
        model = self.model
        if model is None:
            raise RuntimeError(
                "LinearSolver is not attached to a model: DartsModel.linear_solver is "
                "composed in __init__ with model=self, so a replacement must pass "
                "model= too (LinearSolver(spec, model=self) / "
                "LinearSolver.from_handle(raw, model=self))."
            )
        engine = getattr(model.physics, "engine", None)
        if engine is None:
            return
        # self owns the linear-solver settings: mirror them into sim_params, which is
        # what engine.init() re-applies to the solver it (re-)inits.
        self._sync_solver_to_sim_params()
        # Python-resident backend (PETSc / Pardiso); None unless a
        # PythonLinearSolverSpec is selected. Reset on every (re)build.
        self.python_solver = None
        platform = getattr(model, "platform", "cpu")
        if platform == "gpu":
            # GPU: a GPUSolverSpec on self names a params.linear_type enum;
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
                model.params.linear_type = proprietary_type
            return
        # Other non-CPU platforms (gpu handled above): leave the factory in charge.
        if platform != "cpu":
            return
        spec = self._resolve_solver_spec()
        if self._solver_is_default() and getattr(
            model, "linear_solver_from_engine_factory", False
        ):
            # Mechanics / THMC models configure engine.ls_params + params directly and
            # never chose a spec: leave the engine factory / ls_params in charge.
            return
        if spec is None and self.handle is not None:
            # A raw compiled solver object built outside the spec API (wrapped by
            # LinearSolver.from_handle, or assigned directly to .handle) -- the
            # documented fine-control path. Previously it was silently replaced by
            # the default GMRES+CPR.
            self._adaptive_solver_index = 0
            self._adaptive_failures = 0
            model.physics.engine.set_linear_solver(
                self.handle,
                self.label or "raw",
            )
            return
        if not isinstance(spec, LinearSolverSpec):
            # No solver assigned at all (self.spec is None -- reset() ran without
            # set_solver(), e.g. a THMC subclass override). Apply the platform default
            # when this model uses the registry path -- matching the former
            # _apply_linear_solver_spec, which defaulted to FGMRES + CPR/AMG. Mechanics /
            # THMC models (linear_solver_from_engine_factory) leave the engine factory /
            # ls_params in charge instead.
            if getattr(model, "linear_solver_from_engine_factory", False):
                return
            # The model's set_solver() override never materialized one: apply the
            # BASE defaults explicitly (not model.set_solver(), which would re-enter
            # the override that just declined to set it). set_solver lives on
            # DartsModel (the user-facing override hook); import lazily to avoid
            # a module cycle.
            from darts.models.darts_model import DartsModel

            DartsModel.set_solver(model)
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
            self.python_solver = spec.build(block_size)
            # engine.init() requires a C++ solver; inject the CPU default stack as a
            # placeholder -- it is constructed but never used at solve time.
            self.handle = GMRESSolverSpec(restart=50, prec=CPRSolverSpec()).build(
                block_size
            )
        else:
            # Engine-resident solver (MGR / GMRES / CPR / FSCPR / SuperLU /
            # Adaptive); keep a reference so it outlives the engine's raw pointer.
            self.handle = spec.build(block_size)
        # Human-readable label for the engine's "Linear solver type is ..." log line.
        model.physics.engine.set_linear_solver(self.handle, _describe_solver_spec(spec))

    def _apply_gpu_solver(self):
        """Translate a GPU ``LinearSolverSpec`` on ``self`` to ``params.linear_type``.

        GPU solvers are selected by the GPU engine factory (``engine_base_gpu``) via
        the ``linear_solver_t`` enum, not the open-source registry. A
        :class:`~darts.linear_solvers.specs.GPUSolverSpec` names the enum value
        (``linear_type_name``); set it on ``params.linear_type`` here, before
        ``engine.init`` adopts it. A ``None`` / non-GPU spec leaves
        ``params.linear_type`` as :meth:`init` set it (``gpu_gmres_cpr_amgx_ilu``).
        """
        model = self.model
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
                    f"self.linear_solver.spec = {type(spec).__name__} is a CPU registry "
                    "spec; on platform='gpu' the engine uses the GPU factory "
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
        model.params.linear_type = enum_value
        # Local (block-Schur) elimination: the GPU engine factory wraps the
        # selected chain in linsolv_schur_elim<N,K> when this counter is set. The
        # explicit eliminated (row, column) pairs are passed through params.
        schur_elim = int(getattr(spec, "schur_elim_count", 0) or 0)
        if hasattr(model.params, "schur_elim_count"):
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
                model.params.schur_elim_count = schur_elim
                model.params.schur_elim_rows = index_vector([int(r) for r in rows])
                model.params.schur_elim_cols = index_vector([int(c) for c in cols])
            else:
                # ALWAYS clear: a solver re-selection (reset() with a new spec)
                # must not inherit elimination from a previously applied spec.
                model.params.schur_elim_count = 0
                model.params.schur_elim_rows = index_vector([])
                model.params.schur_elim_cols = index_vector([])

    @staticmethod
    def open_source_solvers_available() -> bool:
        """True when the compiled ``darts.linear_solvers`` registry is present (open-source
        build). False in proprietary ``-a`` builds and where the extension failed to
        load -- there ``create_mgr_solver_*`` / specs are unavailable and the engine
        factory selects the solver from ``params.linear_type`` instead. Use this to
        guard raw-object solver builds in :meth:`DartsModel.set_solver`."""
        try:
            from darts import linear_solvers

            return bool(getattr(linear_solvers, "_have_compiled_solvers", False))
        except Exception:
            return False

    def _linear_solver_timer_totals(self):
        """Cumulative (setup, solve) seconds of the engine's linear-solver
        timer nodes; (0, 0) when the tree is not available."""
        try:
            sim = self.model.timer.node["simulation"]
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
        model = self.model
        engine = model.physics.engine
        setup_total, solve_total = self._linear_solver_timer_totals()
        d_setup = setup_total - self._ls_setup_time_prev
        d_solve = solve_total - self._ls_solve_time_prev
        self._ls_setup_time_prev = setup_total
        self._ls_solve_time_prev = solve_total
        # Per-timestep counters live on the nonlinear solver's NonlinearStatus
        # since !327 (they were engine members before).
        status = getattr(getattr(model, "nonlinear_solver", None), "status", None)
        return SolverSwitchContext(
            current_index=self._adaptive_solver_index,
            timestep_converged=timestep_converged,
            linear_solver_error=int(getattr(status, "linear_solver_rc", 0) or 0),
            linear_iterations=int(getattr(status, "n_linear", 0) or 0),
            newton_iterations=int(getattr(status, "n_newton", 0) or 0),
            consecutive_failures=self._adaptive_failures,
            dt=float(dt),
            simulation_time=float(getattr(engine, "t", 0.0)),
            ls_setup_time=d_setup,
            ls_solve_time=d_solve,
            phase=getattr(model, "solver_phase", None),
        )

    def _maybe_switch_linear_solver(self, timestep_converged: bool, dt: float = 0.0):
        """Adaptive linear-solver switching, evaluated after each timestep
        attempt (converged or not -- a failed attempt is followed by a dt-cut
        retry, so acting here lets the RETRY run on the fallback solver).

        When ``self.spec`` is an :class:`AdaptiveSolverSpec` its policy is
        evaluated; the policy may return a candidate index (legacy) or a
        :class:`SolverAction` combining a switch with in-place parameter
        updates. A no-op for plain specs, in proprietary builds, and on GPU.
        """
        if not _HAVE_SOLVER_REGISTRY:
            return
        spec = self._resolve_solver_spec()
        if not isinstance(spec, AdaptiveSolverSpec):
            return
        engine = self.model.physics.engine
        current_index = self._adaptive_solver_index
        if not timestep_converged:
            self._adaptive_failures += 1
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
            self.handle = spec.candidates[new_index].build(self._block_size())
            engine.set_linear_solver(
                self.handle, _describe_solver_spec(spec.candidates[new_index])
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

        The spec on ``self.spec`` stays authoritative: a later ``reset()``
        or solver rebuild reproduces the updated configuration.

        :returns: ``"reconfigured"`` (applied in place) or ``"rebuilt"``
            (fresh solver injected -- still no Jacobian reallocation).
        """
        model = self.model
        if getattr(model, "platform", "cpu") == "gpu":
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
            self.spec = spec
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
            target = cur.candidates[self._adaptive_solver_index]

        touched = self._apply_spec_field_updates(target, field_updates)
        self._sync_linear_params(field_updates, target)

        # Python-resident solvers (PETSc / Pardiso) are owned by the model;
        # rebuilding them is cheap relative to their solves.
        if isinstance(target, PythonLinearSolverSpec):
            self._apply_solver()
            return "rebuilt"

        if self.handle is None or not touched:
            return "reconfigured"  # not built yet -- the spec carries it all

        # In-place reconfiguration through the live handle. The outer solver
        # forwards configs it does not recognise to its preconditioner, so
        # both the GMRES fields and the CPR fields of a GMRES+CPR stack are
        # reachable through the single outer handle.
        rc = 0
        for touched_spec in touched:
            try:
                rc = max(rc, int(self.handle.reconfigure(touched_spec._make_config())))
            except AttributeError:
                rc = 1  # handle without reconfigure (e.g. Python-side stub)
        if rc > 0:
            # Structural change: rebuild + inject on the existing Jacobian.
            self._apply_solver()
            return "rebuilt"
        return "reconfigured"

    def _solver_is_default(self) -> bool:
        """True when ``self.spec`` is the platform default that
        ``DartsModel.set_solver`` materialised, i.e. the model never *chose* a
        linear solver. Identity-compared, so a model that calls
        ``super().set_solver()`` and then assigns its own spec counts as having
        chosen one."""
        return self._default_spec is not None and self.spec is self._default_spec

    def _sync_solver_to_sim_params(self):
        """Mirror the linear-solver settings of ``self`` into ``sim_params``.

        ``self.spec`` (a :class:`LinearSolverSpec`) is the single owner of the linear
        knobs; ``sim_params`` is the C++ mirror the engine reads -- it re-applies
        ``tolerance_linear`` / ``max_i_linear`` to whatever solver it (re-)inits, and
        passes ``linear_print_level`` to the Python-resident solvers. Called by
        :meth:`_apply_solver` -- i.e. after ``set_solver()``, before ``engine.init``.

        Models that never chose a solver *and* set
        ``linear_solver_from_engine_factory`` (mechanics / THMC: they drive
        ``engine.ls_params`` + ``params`` directly) are left alone.
        """
        model = self.model
        spec = self._resolve_solver_spec()
        if spec is None:
            return  # raw compiled solver handle: it carries its own configuration
        if self._solver_is_default() and getattr(
            model, "linear_solver_from_engine_factory", False
        ):
            return
        # An AdaptiveSolverSpec delegates to the candidate currently in use.
        candidates = getattr(spec, "candidates", None)
        if candidates:
            spec = candidates[self._adaptive_solver_index]
        model.params.tolerance_linear = spec.tolerance
        model.params.max_i_linear = spec.max_iterations
        if getattr(spec, "print_level", None) is not None:
            model.params.linear_print_level = spec.print_level

    def _sync_linear_params(self, field_updates, spec):
        """Keep the engine-side linear-solve knobs coherent with a spec
        update: params.tolerance_linear / max_i_linear are what the engine
        re-applies to any solver it (re-)inits."""
        model = self.model
        if "tolerance" in field_updates:
            model.params.tolerance_linear = field_updates["tolerance"]
        if "max_iterations" in field_updates:
            model.params.max_i_linear = field_updates["max_iterations"]
        # Note: a bare spec switch deliberately does NOT overwrite
        # params.tolerance_linear / max_i_linear -- engine-resident solvers
        # take those from the engine at (re-)init (the documented convention),
        # and silently replacing e.g. a model's 1e-2 with a spec DEFAULT of
        # 1e-5 would change every subsequent solve.

    def get_linear_system(self):
        # returns scipy sparse matrix and pointers to RHS and dX
        from scipy.sparse import bsr_matrix

        model = self.model
        # get current jacobian and rhs from the engine
        indptr = np.asarray(model.physics.engine.jac_rows)
        indices = np.asarray(model.physics.engine.jac_cols)
        data = np.asarray(model.physics.engine.jac_vals)

        rhs = np.array(model.physics.engine.RHS, copy=False)
        sol = np.array(model.physics.engine.dX, copy=False)

        nonzeros = indices.size

        if nonzeros == 0:
            print(
                f'linear solver is {self} (params.linear_type={model.params.linear_type})'
            )

        assert nonzeros > 0, (
            'Jacobian is not exposed to python! Probably superlu set as a linear solver!'
        )

        b = int(np.sqrt(data.size / nonzeros))
        data = data.reshape(nonzeros, b, b)

        mat = bsr_matrix((data, indices, indptr))

        mat_csr = mat.tocsr()  # TODO  avoid this conversion to non-blocked matrix

        return mat_csr, rhs, sol

    def _solve_linear_equation(self):
        """Backend-neutral linear-solve dispatch funnel used by the nonlinear
        solver (:meth:`darts.nonlinear_solvers.NonlinearSolver._solve_linear`).

        Returns ``(rc, n_iters, residual)`` for every backend -- ``rc`` is ``0``
        on success, ``1`` on setup failure, ``2`` on solve failure -- so the
        nonlinear driver stays backend-agnostic (!327 contract).

        Two solver kinds are dispatched here (!280 spec-driven routing, which
        replaced the former ``ts_control.linear_type`` enum dispatch):

        * a Python-resident solver built from a
          :class:`~darts.linear_solvers.specs.PythonLinearSolverSpec` (PETSc /
          Pardiso) selected through ``self.spec``;
        * the C++ engine solver (the default path).
        """
        if self.python_solver is not None:
            # Python-resident solver (PETSc / Pardiso); stateful, it performs
            # its one-time setup on the first call. A nonzero rc means a hard
            # failure (non-finite solution) -- the nonlinear solver records it
            # in NonlinearStatus.linear_solver_rc and aborts the Newton loop.
            return self.python_solver.solve_system(self.model.physics.engine)
        # C++ linear solver held by the engine
        engine = self.model.physics.engine
        rc = engine.solve_linear_equation()
        return rc, engine.get_last_linear_iters(), engine.get_last_linear_residual()

    # ------------------------------------------------------------ deprecated config shims (former LegacyConfigShims mixin)

    def _migrate_legacy_solver_kwargs(self, legacy: dict):
        """One-deprecation-cycle shim: map removed ``set_sim_params`` solver
        keyword arguments -- nonlinear (!327) and linear (!280) alike -- onto
        ``model.nonlinear_solver.spec`` / ``self.spec`` and warn. Unknown keys
        raise TypeError so genuine typos still fail loudly."""
        from darts.nonlinear_solvers.newton import _ENUM_TO_CHOP_MODE

        model = self.model
        spec = model.nonlinear_solver.spec
        handled = []
        # --- linear family (MR280): the spec is the single owner ---
        if "tol_linear" in legacy or "it_linear" in legacy:
            lin = self.spec
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
            "self.nonlinear_solver = NewtonSolver(...) / self.linear_solver.spec = "
            "<LinearSolverSpec> in set_solver().",
            DeprecationWarning,
            stacklevel=3,
        )
