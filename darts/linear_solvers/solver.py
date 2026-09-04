"""Runtime linear solver: the instance a model assigns to ``DartsModel.linear_solver``.

Mirrors the runtime-solver design of :mod:`darts.nonlinear_solvers` (!327):
``DartsModel.nonlinear_solver`` holds a detached :class:`NewtonSolver` instance that
binds to the model during ``init()``; :class:`LinearSolver` is the linear twin.
It is constructed *detached* from the model (no physics / engine needed), holds the
declarative :attr:`spec` (a :class:`~darts.linear_solvers.specs.LinearSolverSpec`),
and materializes its computational backend when :meth:`DartsModel._apply_solver`
binds it during ``reset()`` — the point where the matrix block size and the engine
object exist (before ``engine.init()``, which adopts the injected backend).

Unlike the nonlinear side, where the *method* identity lives in the runtime class
(``NewtonSolver`` vs ``MechanicsNewtonSolver``), the linear method identity lives in
the **spec** (``MGRSolverSpec``, ``GMRESSolverSpec(prec=CPRSolverSpec())``, …); one
uniform runtime wrapper therefore serves every backend:

- ``handle`` — the built C++ solver injected into the engine (open-source CPU
  builds; ``None`` elsewhere);
- ``python_solver`` — the Python-resident PETSc / Pardiso backend when the spec is
  a :class:`~darts.linear_solvers.specs.PythonLinearSolverSpec` (``None`` otherwise);
- neither on GPU / proprietary builds, where the engine factory constructs the
  solver internally from ``params.linear_type`` (the spec still drives the enum and
  the tolerances mirrored into ``sim_params``).
"""

from darts.linear_solvers.specs import LinearSolverSpec


def is_compiled_solver_handle(obj) -> bool:
    """True when ``obj`` is a raw solver object from the compiled
    ``darts.linear_solvers.linear_solvers`` extension (the ``LinearSolver``
    handle returned by ``spec.build()``, an ``MGRSolver_N`` facade, ...) --
    i.e. something ``engine.set_linear_solver`` accepts directly. Compiled
    handles expose no common Python-visible solve API (their virtuals live on
    the C++ side), so the discriminator is the defining module. A duck-typed
    ``setup``/``solve`` pair is accepted as an escape hatch for Python-
    implemented handles."""
    if type(obj).__module__ == "darts.linear_solvers.linear_solvers":
        return True
    return hasattr(obj, "setup") and hasattr(obj, "solve")


class LinearSolver:
    """Runtime linear solver bound to a model at ``reset()``/``init()`` time.

    Mirror of :class:`darts.nonlinear_solvers.NonlinearSolver`: constructed
    detached, the declarative configuration stays retrievable as :attr:`spec`
    (serializable via ``spec.to_dict()``), and the model binds it via
    :meth:`bind` before materializing the backend.

    :param spec: the :class:`LinearSolverSpec` to run. ``None`` only for the
        raw-handle path (:meth:`from_handle`); the platform defaults are
        constructed explicitly in ``DartsModel.set_solver()``.
    :param model: optional model to bind at construction (usually left ``None``).
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

    # ------------------------------------------------------------ construction

    @classmethod
    def from_handle(cls, raw, label: str = None) -> "LinearSolver":
        """Wrap a raw compiled solver handle built outside the spec API (the
        documented fine-control path, e.g.
        ``linear_solvers.create_mgr_solver_for_block_size(...)``). The wrapper
        has no :attr:`spec`; the handle is injected as-is."""
        if not is_compiled_solver_handle(raw):
            raise TypeError(
                "LinearSolver.from_handle expects a compiled darts.linear_solvers "
                f"solver handle, got {type(raw).__name__}"
            )
        solver = cls()
        solver.handle = raw
        solver.label = label
        return solver

    # ------------------------------------------------------------ binding

    def bind(self, model) -> "LinearSolver":
        """Attach this (possibly detached) solver to a model; returns self. The
        backend is materialized separately by :meth:`DartsModel._apply_solver`
        (it needs the matrix block size and the engine object)."""
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
