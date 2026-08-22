"""Unified Python-side boundary/source conditions applied after engine assembly.

This module is the minimum-viable unified conditions layer: a small framework
through which models add Python-side source terms and interface fluxes to the
assembled residual/Jacobian. It is the ONLY such channel: the ad-hoc
``set_rhs_flux`` override and the untyped ``rhs_flux_hooks`` list it replaced
are gone from ``DartsModel`` -- the base class defines neither, and
``apply_rhs_flux`` is the conditions stage plus the observer stage and nothing
else. A model that still spells a contribution either of the old ways registers
a :class:`ConditionItem` instead; see the *Conditions* page of the technical
reference for the mechanical translation.

Structure:

- :class:`BlockCSRView` — a numpy view of the engine's block-CSR Jacobian with
  position lookup and in-place block accumulation.
- :class:`AssemblyContext` — the per-iteration snapshot handed to every item.
- :class:`ConditionItem` — the item base class; its docstring is THE CONTRACT.
- :class:`ConditionSet` — the validated collection owned by ``DartsModel``
  (``model.conditions``), compiled once at the end of ``init()``.
- Item types: :class:`CellSource`, :class:`SegmentSource`, :class:`PipeSourceTerm`,
  :class:`InterfaceFlux`.
- :class:`ConstantStateBC` — a DECLARATIVE item: it writes nothing, it declares
  the open / constant-state far field expressed by the huge-boundary-volume
  trick and validates at bind time that the volumes really reached the mesh
  before the engine cached the pore volumes.
- :class:`DirichletPin` — a prescribed state value for selected (cell, equation)
  pairs, in two modes: ``"state"`` (the projection the models used to hand-roll:
  overwrite the entry in the state vector) and ``"row"`` (the assembly-consistent
  block-CSR row replacement). See its docstring for the trade-off.

The contract hardening added in M4 (review items E5-E9):

- **Declared stencils** (:meth:`ConditionItem.declare_stencil`) — an item reports
  the ``(row_block, col_block)`` couplings it will write BEFORE the engine
  allocates its matrix, and :meth:`ConditionSet.declare_stencil` adds the missing
  ones to the mesh as zero-transmissibility connections. This is what removes the
  need for a "fake" zero-well-index perforation whose only purpose was to smuggle
  four blocks into the sparsity pattern.
- **Pattern versioning** (:func:`pattern_identity`) — compiled CSR positions are
  stamped with the identity of the pattern they were resolved against, and are
  re-resolved instead of silently written at stale offsets when the engine is
  rebuilt (restart, or any second :meth:`~darts.models.darts_model.DartsModel.reset`).
- **Additive vs replacement** (:attr:`ConditionItem.contribution`) — an item
  either ADDS to a row or CLAIMS it as a constraint; the two are mutually
  exclusive per row and :meth:`ConditionSet.compile` rejects a conflict, naming
  both items.
- **Typed observers** (:class:`NonlinearIterationObserver`) — read-only
  per-iteration diagnostics/policy checks, registered alongside conditions,
  handed a context whose arrays are not writable.
- **Restart** (:attr:`ConditionItem.carries_restart_state`) — an item declares
  whether it carries state that must survive a restart and how it serializes; an
  item with state is either restored from the sidecar file or refuses to restart.
- **Selectors** (:class:`Selector`) — a compile-time description of a set of
  blocks (explicit indices, a predicate over cell centroids, or a named region),
  accepted anywhere a plain index array is accepted. Strictly additive: every
  existing call that passes indices keeps working unchanged.
"""

import inspect
import json
import os
from dataclasses import dataclass

import numpy as np

#: contribution kinds, see :attr:`ConditionItem.contribution`
ADDITIVE = "additive"
REPLACEMENT = "replacement"
NO_CONTRIBUTION = "none"
_CONTRIBUTION_KINDS = (ADDITIVE, REPLACEMENT, NO_CONTRIBUTION)


def pattern_identity(model) -> tuple:
    """Identity of the Jacobian sparsity pattern currently owned by the engine.

    Compiled CSR positions (and the numpy views into ``jac_vals``) are only valid
    for the pattern they were resolved against. ``engine.init()`` reallocates the
    matrix, so a model that is re-initialized -- the restart flow calls
    :meth:`~darts.models.darts_model.DartsModel.reset` a second time, and so does
    any manual re-init -- silently invalidates them: the row/column arrays may
    even come back the same size, so comparing sizes alone is not enough.

    The identity therefore combines

    * the engine object identity,
    * ``DartsModel._pattern_version``, a counter bumped by every
      :meth:`~darts.models.darts_model.DartsModel.reset` (the authoritative half),
    * the row / column / value array sizes (a cheap independent cross-check).

    :param model: the model (or anything exposing ``physics.engine``)
    :returns: an opaque, comparable tuple; ``None`` when no engine is available
    """
    engine = getattr(getattr(model, "physics", None), "engine", None)
    if engine is None:
        return None
    try:
        n_rows = len(engine.jac_rows)
        n_cols = len(engine.jac_cols)
        n_vals = len(engine.jac_vals)
    except (AttributeError, TypeError):
        n_rows = n_cols = n_vals = -1
    return (
        id(engine),
        int(getattr(model, "_pattern_version", 0)),
        n_rows,
        n_cols,
        n_vals,
    )


class BlockCSRView:
    """Numpy view of the engine's block-CSR Jacobian (rows/cols/diags/values).

    Wraps ``engine.jac_rows`` / ``jac_cols`` / ``jac_diags`` / ``jac_vals`` as
    numpy views (no copies — writes go straight into the engine matrix). Each
    CSR position addresses one dense ``n_vars x n_vars`` block stored
    equation-major: entry ``(c, v)`` of block ``pos`` lives at flat value index
    ``pos * n_vars**2 + c * n_vars + v``.
    """

    def __init__(self, engine, n_vars: int, pattern=None):
        jac_diags = getattr(engine, "jac_diags", None)
        if jac_diags is None or len(jac_diags) == 0:
            raise RuntimeError(
                "The engine does not expose its Jacobian to Python "
                "(jac_rows/jac_cols/jac_diags/jac_vals are empty). Python-side "
                "Jacobian contributions require a CPU engine family with an "
                "exposed block-CSR matrix (the iterative-solver flow engines); "
                "direct-solver configurations (e.g. SuperLU) and the GPU "
                "engines do not expose it."
            )
        self.jac_rows = np.asarray(engine.jac_rows)
        self.jac_cols = np.asarray(engine.jac_cols)
        self.jac_diags = np.asarray(jac_diags)
        self.jac_vals = np.asarray(engine.jac_vals)
        self.n_vars = int(n_vars)
        self.block_size = self.n_vars * self.n_vars
        #: identity of the pattern these positions/views address (see
        #: :func:`pattern_identity`); ``None`` when the caller did not stamp one.
        self.pattern = pattern

    def read_only(self) -> "BlockCSRView":
        """A twin of this view whose value array cannot be written.

        Handed to :class:`NonlinearIterationObserver` implementations so a
        read-only observer cannot mutate the Jacobian even by accident (a write
        raises ``ValueError: assignment destination is read-only``).
        """
        twin = object.__new__(BlockCSRView)
        twin.jac_rows = self.jac_rows
        twin.jac_cols = self.jac_cols
        twin.jac_diags = self.jac_diags
        twin.jac_vals = _frozen(self.jac_vals)
        twin.n_vars = self.n_vars
        twin.block_size = self.block_size
        twin.pattern = self.pattern
        return twin

    def block_pos(self, row_block: int, col_block: int) -> int:
        """Flat CSR position of dense block ``(row_block, col_block)``.

        Binary search over the (sorted) column range of the row; raises
        ``RuntimeError`` when the block is absent from the sparsity pattern.
        """
        row_start = int(self.jac_rows[row_block])
        row_end = int(self.jac_rows[row_block + 1])
        pos = row_start + int(
            np.searchsorted(self.jac_cols[row_start:row_end], col_block)
        )
        if pos == row_end or self.jac_cols[pos] != col_block:
            raise RuntimeError(
                f"CSR block ({row_block}, {col_block}) was not found in the "
                "Jacobian pattern."
            )
        return pos

    def diag_pos(self, block: int) -> int:
        """Flat CSR position of the diagonal dense block of ``block``."""
        return int(self.jac_diags[block])

    def add_block(self, pos: int, dense: np.ndarray):
        """In-place ``+=`` of one dense ``n_vars x n_vars`` block at CSR
        position ``pos`` (equation-major ``c * n_vars + v`` layout)."""
        start = pos * self.block_size
        self.jac_vals[start : start + self.block_size] += np.asarray(dense).reshape(-1)


def _frozen(array: np.ndarray) -> np.ndarray:
    """A non-writable view of ``array`` (the array itself stays writable)."""
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


# ------------------------------------------------------------------ selectors
class Selector:
    """A compile-time description of a SET OF BLOCKS (review item E7).

    A selector answers one question -- *which blocks?* -- and answers it once,
    at bind time, against the model. It is the "where" half of the
    selector/law split: the "what" half (the behaviour: a rate, a pinned value,
    a flux law) stays in the item.

    Selectors are accepted ANYWHERE a plain index array is accepted today
    (:class:`CellSource`, :class:`DirichletPin`, and the block members of an
    :class:`InterfaceFlux` connection list), through :func:`resolve_blocks`.
    This is strictly additive: passing indices keeps working exactly as before,
    and no existing item constructor changed shape.

    Subclasses implement :meth:`resolve`.
    """

    def resolve(self, model) -> np.ndarray:
        """Return the selected block indices as an int64 array."""
        raise NotImplementedError(f"{type(self).__name__}.resolve() is not implemented")

    def __len__(self):
        raise TypeError(
            f"{type(self).__name__} has no length before it is resolved against a "
            "model; the item resolves it in bind()."
        )


class BlockIndices(Selector):
    """The identity selector: an explicit list of block indices.

    :param indices: iterable of block indices.
    """

    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64).ravel()

    def resolve(self, model) -> np.ndarray:
        return self.indices


class Where(Selector):
    """Blocks whose cell centroid satisfies a predicate.

    :param predicate: callable ``f(x, y, z) -> bool array`` over the reservoir
        cell centroid coordinate arrays (each of length ``n_res_blocks``).
    :param centroids: optional explicit ``(n, 3)`` centroid array. When omitted
        the centroids are resolved from the reservoir (see
        :func:`cell_centroids`), which is not available for every reservoir
        family -- pass them explicitly when it raises.

    Only RESERVOIR blocks are considered: a well segment has no reservoir
    centroid, and selecting one by geometry would be ambiguous.
    """

    def __init__(self, predicate, centroids=None):
        if not callable(predicate):
            raise TypeError("Where(predicate=...) expects a callable f(x, y, z).")
        self.predicate = predicate
        self.centroids = (
            None if centroids is None else np.asarray(centroids, dtype=float)
        )

    def resolve(self, model) -> np.ndarray:
        centroids = (
            self.centroids if self.centroids is not None else cell_centroids(model)
        )
        centroids = np.asarray(centroids, dtype=float)
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                f"Where: centroids must have shape (n_cells, 3), got {centroids.shape}."
            )
        mask = np.asarray(
            self.predicate(centroids[:, 0], centroids[:, 1], centroids[:, 2])
        )
        if mask.shape != (centroids.shape[0],):
            raise ValueError(
                f"Where: the predicate returned shape {mask.shape}, expected "
                f"{(centroids.shape[0],)} (one bool per cell)."
            )
        return np.flatnonzero(mask.astype(bool)).astype(np.int64)


class NamedRegion(Selector):
    """Blocks of a named region.

    Resolution order, first hit wins:

    1. ``reservoir.regions[name]`` / ``reservoir.cell_groups[name]`` -- a
       reservoir that keeps its own named cell groups (index arrays or masks);
    2. an OPERATOR region tag: reservoir blocks whose ``mesh.op_num`` slot maps
       to ``name`` in ``physics.regions`` (slot ``i`` is region
       ``physics.regions[i]``, the same convention the engine and
       ``DartsModel.set_op_list`` use).

    :param name: the region key / operator-region tag.
    """

    def __init__(self, name):
        self.name = name

    def resolve(self, model) -> np.ndarray:
        reservoir = getattr(model, "reservoir", None)
        for attribute in ("regions", "cell_groups"):
            groups = getattr(reservoir, attribute, None)
            if isinstance(groups, dict) and self.name in groups:
                selected = np.asarray(groups[self.name])
                if selected.dtype == bool:
                    return np.flatnonzero(selected).astype(np.int64)
                return selected.astype(np.int64).ravel()

        physics = getattr(model, "physics", None)
        regions = list(getattr(physics, "regions", None) or [])
        if self.name not in regions:
            raise KeyError(
                f"NamedRegion({self.name!r}): no such named cell group on "
                f"{type(reservoir).__name__} and no such operator region; the "
                f"registered operator regions are {regions}."
            )
        slot = regions.index(self.name)
        mesh = reservoir.mesh
        n_res_blocks = int(mesh.n_res_blocks)
        op_num = np.asarray(mesh.op_num)[:n_res_blocks]
        return np.flatnonzero(op_num == slot).astype(np.int64)


def cell_centroids(model) -> np.ndarray:
    """Reservoir cell centroids as an ``(n_res_blocks, 3)`` array.

    Tries, in order, ``reservoir.centroids``, ``reservoir.discr_mesh.centroids``
    and ``reservoir.discretizer.centroids_all_cells``; raises when the reservoir
    family exposes none of them.

    :param model: model owning ``reservoir``
    :raises RuntimeError: when no centroid array can be found
    """
    reservoir = getattr(model, "reservoir", None)
    n_res_blocks = int(reservoir.mesh.n_res_blocks)
    candidates = (
        getattr(reservoir, "centroids", None),
        getattr(getattr(reservoir, "discr_mesh", None), "centroids", None),
        getattr(getattr(reservoir, "discretizer", None), "centroids_all_cells", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        array = np.asarray([[c[0], c[1], c[2]] for c in candidate], dtype=float)
        if array.ndim == 2 and array.shape[1] == 3 and array.shape[0] >= n_res_blocks:
            return array[:n_res_blocks]
    raise RuntimeError(
        f"Where(...): {type(reservoir).__name__} exposes no cell centroids "
        "(tried reservoir.centroids, reservoir.discr_mesh.centroids and "
        "reservoir.discretizer.centroids_all_cells). Pass them explicitly: "
        "Where(predicate, centroids=my_centroids)."
    )


def resolve_blocks(value, model) -> np.ndarray:
    """Block indices of ``value``: a :class:`Selector`, or an index array.

    The single entry point every item uses, so a selector is accepted exactly
    where an index array is.

    :param value: a :class:`Selector` instance or anything ``np.asarray`` turns
        into an integer index array
    :param model: the model the selector resolves against
    :returns: int64 array of block indices
    """
    if isinstance(value, Selector):
        return np.asarray(value.resolve(model), dtype=np.int64).ravel()
    return np.asarray(value, dtype=np.int64).ravel()


@dataclass
class AssemblyContext:
    """Per-Newton-iteration snapshot handed to :meth:`ConditionItem.apply`.

    :ivar rhs: engine residual vector (numpy view, writable).
    :ivar jac: :class:`BlockCSRView` of the Jacobian, or ``None`` when the
        engine does not expose it (then only RHS writes are possible).
    :ivar X: current state vector (numpy view).
    :ivar Xn: state vector at the previous accepted timestep (numpy view).
    :ivar dt: timestep size [days].
    :ivar t: time at the START of the timestep [days].
    :ivar iteration: Newton iteration index within the current timestep.
    :ivar n_vars: unknowns (and equations) per block.
    :ivar n_res_blocks: number of reservoir blocks (well blocks follow).
    """

    rhs: np.ndarray
    jac: "BlockCSRView | None"
    X: np.ndarray
    Xn: np.ndarray
    dt: float
    t: float
    iteration: int
    n_vars: int
    n_res_blocks: int

    def read_only(self) -> "AssemblyContext":
        """A twin of this context whose arrays cannot be written.

        Handed to every :class:`NonlinearIterationObserver`: ``rhs``, ``X``,
        ``Xn`` and the Jacobian values are non-writable numpy views, so an
        observer that tries to mutate the assembled system raises
        ``ValueError: assignment destination is read-only`` instead of quietly
        changing the answer.
        """
        return AssemblyContext(
            rhs=_frozen(self.rhs),
            jac=None if self.jac is None else self.jac.read_only(),
            X=_frozen(self.X),
            Xn=_frozen(self.Xn),
            dt=self.dt,
            t=self.t,
            iteration=self.iteration,
            n_vars=self.n_vars,
            n_res_blocks=self.n_res_blocks,
        )


class NonlinearIterationObserver:
    """Read-only observer of the assembled system, once per Newton iteration.

    THE TYPED SUCCESSOR OF ``DartsModel.after_assembly`` (review item E8). An
    observer is for everything that must SEE every assembled system and
    contribute NOTHING to it: policing what the property evaluators did during
    the assembly, gathering diagnostics, or raising to force a timestep cut.

    Registered alongside conditions::

        model.conditions.add(MyObserver())      # or add_observer(...)

    and called from the ``conditions`` stage of
    :meth:`~darts.models.darts_model.DartsModel.apply_rhs_flux`, AFTER every
    condition item has contributed, so what an observer sees is the final
    system of that iteration.

    **It is forbidden to mutate the residual or the Jacobian**, and the
    framework enforces it rather than trusting it: the context handed to
    :meth:`observe` is :meth:`AssemblyContext.read_only`, whose ``rhs``, ``X``,
    ``Xn`` and ``jac.jac_vals`` are non-writable numpy views. An observer that
    needs to CHANGE the system is not an observer -- it is a
    :class:`ConditionItem`.

    An exception raised in :meth:`observe` propagates out of the Newton loop;
    a model that raises deliberately (to force a timestep cut) is responsible
    for catching it, exactly as with the legacy hook.

    .. note::
       ``DartsModel.after_assembly(dt, t)`` remains supported as the LEGACY
       path and still runs last, after the observers. It is untyped (it gets no
       context and nothing stops it from writing), and new code should register
       a ``NonlinearIterationObserver`` instead.
    """

    def observe(self, ctx: AssemblyContext):
        """Inspect the assembled system. Must not mutate anything.

        :param ctx: the read-only per-iteration context.
        """
        raise NotImplementedError(f"{type(self).__name__}.observe() is not implemented")

    def on_timestep_start(self, dt: float, t: float):
        """Called once before each timestep solve attempt."""
        pass

    def on_timestep_converged(self, dt: float, t: float):
        """Called once when a timestep converged."""
        pass

    def on_timestep_failed(self, dt: float, t: float):
        """Called once when a timestep solve failed (before the dt cut)."""
        pass


class ConditionItem:
    """Base class of a unified condition — THE CONTRACT.

    :meth:`apply` is called AFTER the engine assembly on EVERY Newton
    iteration, so contributions must be re-derived from the context state each
    call. Contributions must be dt-scaled residual-unit ADDs following the
    engine sign convention: the residual is ``R = acc - dt * (inflow -
    outflow)``, so injecting a rate ``q_c`` (per day, positive into the block)
    into block ``b`` means ``ctx.rhs[b * n_vars + c] -= q_c * dt``, and the
    matching Jacobian contribution is ``-d(q_c)/d(state) * dt``. Jacobian
    mutation is CPU-only (``ctx.jac`` is ``None`` otherwise); items that write
    the Jacobian declare ``provides_jacobian = True`` so
    :meth:`ConditionSet.compile` can validate the platform/engine up front.

    ADDITIVE VERSUS REPLACEMENT (review item E6). The two are distinct, typed
    and mutually exclusive per row, declared by :attr:`contribution`:

    - ``"additive"`` — the item ADDS to the rows it touches. Several additive
      items may share a row; their contributions sum.
    - ``"replacement"`` — the item CLAIMS the rows it touches: it overwrites
      them with a constraint (``DirichletPin(mode="row")`` is the shipped
      example). A claimed row admits exactly ONE claimant and no additive
      contribution — otherwise the added term would either be overwritten
      (silently dropped) or corrupt the constraint, depending on the order the
      items happen to run in.
    - ``"none"`` — the item writes neither residual nor Jacobian. Declarative
      items (:class:`ConstantStateBC`) and pure projections
      (``DirichletPin(mode="state")``, which writes the STATE, not the system)
      are in this class and take part in no row conflict.

    The rows an item writes are reported by :meth:`written_rows`, and
    :meth:`ConditionSet.compile` rejects a conflict naming both items. An item
    that does not implement :meth:`written_rows` reports no rows and is
    therefore exempt from the check — implement it.

    :ivar provides_jacobian: item adds analytic Jacobian contributions.
    :ivar requires_platform: restrict to one platform (``"cpu"``/``"gpu"``),
        or ``None`` for any.
    :ivar adjoint_transparent: contributions are correctly differentiated by
        the adjoint machinery (they are not, unless an item proves otherwise);
        the opt/history-matching driver rejects opaque items.
    :ivar contribution: ``"additive"`` (default), ``"replacement"`` or
        ``"none"``, as described above.
    :ivar carries_restart_state: the item owns internal state that must survive
        a restart; see :meth:`save_restart_state`.
    """

    provides_jacobian: bool = False
    requires_platform: str | None = None
    adjoint_transparent: bool = False
    contribution: str = ADDITIVE
    carries_restart_state: bool = False

    def bind(self, model):
        """Resolve model-dependent data (indices, CSR positions). Called once
        by :meth:`ConditionSet.compile` after the engine exists.

        An implementation that caches CSR positions must stamp them with
        :func:`pattern_identity` and re-resolve when the pattern changes; the
        framework calls :meth:`rebind_if_stale` before every :meth:`apply` for
        items that opt in by storing ``self._pattern`` (see
        :meth:`stamp_pattern`)."""
        pass

    def declare_stencil(self, model):
        """Declare the Jacobian couplings this item will write (review item E5).

        Called ONCE, BEFORE the mesh connection list is frozen and long before
        the engine allocates its matrix, so an item may introduce a coupling
        that does not exist in the mesh: :meth:`ConditionSet.declare_stencil`
        adds every declared-but-absent coupling as a zero-transmissibility
        connection, and the block-CSR pattern then contains the blocks the item
        needs.

        Return an iterable of ``(row_block, col_block)`` pairs. Each pair
        declares the OFF-DIAGONAL blocks in BOTH directions -- ``(row, col)``
        and ``(col, row)`` -- because a mesh connection is symmetric in the
        pattern; diagonal blocks always exist and need not be declared. Order
        within a pair is irrelevant.

        Because it runs before the engine exists, an implementation may only use
        mesh/well information: block indices, well head/body indices and
        perforations. It must not touch ``physics.engine``.

        The default declares nothing, which is what every item that only writes
        blocks the mesh already contains (a diagonal source, a flux across an
        existing connection) should do.

        :param model: the model being initialized
        :returns: iterable of ``(row_block, col_block)`` pairs; empty by default
        """
        return ()

    def written_rows(self, model):
        """Flat row indices (``block * n_vars + equation``) this item writes.

        Used by :meth:`ConditionSet.compile` for the additive/replacement
        conflict check (review item E6). Called AFTER :meth:`bind`, so resolved
        indices are available. The default reports nothing, which exempts the
        item from the check.

        :param model: the model being initialized
        :returns: iterable of flat row indices
        """
        return ()

    def apply(self, ctx: AssemblyContext):
        """Add this item's contributions to ``ctx.rhs`` (and ``ctx.jac``)."""
        raise NotImplementedError(f"{type(self).__name__}.apply() is not implemented")

    # -------------------------------------------------------- pattern version
    def stamp_pattern(self, model):
        """Record the pattern the CSR positions just resolved in :meth:`bind`
        belong to. Call at the END of a :meth:`bind` that caches positions."""
        self._model = model
        self._pattern = pattern_identity(model)

    def rebind_if_stale(self, ctx: AssemblyContext):
        """Re-run :meth:`bind` when the Jacobian pattern was rebuilt.

        A no-op for an item that never called :meth:`stamp_pattern`, and for the
        overwhelmingly common case of a pattern that never changed (one tuple
        comparison per item per Newton iteration).
        """
        stamped = getattr(self, "_pattern", None)
        if stamped is None:
            return
        current = ctx.jac.pattern if ctx.jac is not None else None
        if current is not None and current != stamped:
            self.bind(self._model)

    # ------------------------------------------------------- restart contract
    def save_restart_state(self) -> dict:
        """Serialize the internal state that must survive a restart.

        Only called for an item with :attr:`carries_restart_state` set. The
        returned mapping must be JSON-serializable (numbers, strings, lists,
        nested dicts); numpy arrays should be converted with ``.tolist()``.

        :returns: the item's state
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets carries_restart_state = True but does "
            "not implement save_restart_state()."
        )

    def load_restart_state(self, state: dict):
        """Restore the state produced by :meth:`save_restart_state`.

        :param state: the mapping previously returned by
            :meth:`save_restart_state`
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets carries_restart_state = True but does "
            "not implement load_restart_state()."
        )

    def project_state(self, t: float):
        """Project the engine state vector BEFORE the engine assembly.

        The counterpart of :meth:`apply` for the (rare) items that constrain the
        STATE rather than contribute to the residual: :meth:`apply` runs after
        assembly, so a state written there is not the state the residual and the
        Jacobian were built from. Items that must be seen BY the assembly
        implement this instead (or in addition), and a model whose Newton loop
        wants that guarantee calls ``self.conditions.project_state(t)`` at the
        top of every iteration, immediately before
        ``engine.assemble_linear_system(dt)``.

        The default is a no-op, and the framework never calls it on its own: the
        stock nonlinear solver offers no pre-assembly stage, so only models with
        their own Newton loop (``DartsModel.run_timestep`` overrides) can drive
        it today. Implementations must be idempotent — a model may legitimately
        call it and then let :meth:`apply` re-assert the same values.

        :param t: time at the START of the timestep [days].
        """
        pass

    def on_timestep_start(self, dt: float, t: float):
        """Called once before each timestep solve attempt."""
        pass

    def on_timestep_converged(self, dt: float, t: float):
        """Called once when a timestep converged (single acceptance point)."""
        pass

    def on_timestep_failed(self, dt: float, t: float):
        """Called once when a timestep solve failed (before the dt cut)."""
        pass


class ConditionSet:
    """The validated collection of :class:`ConditionItem` on a model.

    ``DartsModel`` owns one instance as ``model.conditions``; items are added
    with :meth:`add` (typically in ``set_wells``/``set_boundary_conditions``)
    and validated+bound once by :meth:`compile` at the end of ``init()``.

    The set also owns the read-only :class:`NonlinearIterationObserver` list
    (added through the same :meth:`add`, or :meth:`add_observer`) and, in
    ``init()``, the stencil-declaration stage (:meth:`declare_stencil` /
    :meth:`verify_stencil`) that runs before the mesh connection list is frozen.
    """

    def __init__(self):
        self.items = []
        self._model = None  # set by compile(); used by the apply-time adjoint re-check
        self.observers = []
        #: filled by :meth:`declare_stencil`
        self.declared_couplings = ()
        self.added_couplings = ()
        #: True when compile() deferred binding because the engine did not exist
        #: yet (the restart flow); DartsModel re-runs compile() after reset().
        self.deferred = False

    def __bool__(self):
        return bool(self.items) or bool(self.observers)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def add(self, item):
        """Register a :class:`ConditionItem` or a
        :class:`NonlinearIterationObserver`; returns it for chaining."""
        if isinstance(item, NonlinearIterationObserver):
            return self.add_observer(item)
        if not isinstance(item, ConditionItem):
            raise TypeError(
                "ConditionSet.add expects a ConditionItem or a "
                f"NonlinearIterationObserver, got {type(item).__name__}"
            )
        if item.contribution not in _CONTRIBUTION_KINDS:
            raise ValueError(
                f"{type(item).__name__}.contribution is {item.contribution!r}; "
                f"expected one of {list(_CONTRIBUTION_KINDS)}."
            )
        self.items.append(item)
        return item

    def add_observer(
        self, observer: NonlinearIterationObserver
    ) -> NonlinearIterationObserver:
        """Register a read-only per-iteration observer (review item E8)."""
        if not isinstance(observer, NonlinearIterationObserver):
            raise TypeError(
                "ConditionSet.add_observer expects a NonlinearIterationObserver, "
                f"got {type(observer).__name__}"
            )
        self.observers.append(observer)
        return observer

    # ------------------------------------------------- stencil declaration (E5)
    def stencil_declarers(self, model) -> list:
        """The registered items that declare a Jacobian stencil.

        Only items that actually OVERRIDE :meth:`ConditionItem.declare_stencil`
        are returned, so the whole stage is skipped -- with no side effect of
        any kind -- for every model that declares nothing.

        :param model: the model being initialized (unused; the items are the
            single registry since the legacy hook list was removed)
        :returns: list of declaring items, in registration order
        """
        del model
        return [
            item
            for item in self.items
            if type(item).declare_stencil is not ConditionItem.declare_stencil
        ]

    def declare_stencil(self, model, declarers=None) -> tuple:
        """Add the declared-but-absent couplings to the mesh connection list.

        THE TIMING IS THE WHOLE POINT and it is narrow. The block-CSR sparsity
        pattern is built by ``engine.init_jacobian_structure()`` from the
        SORTED, TWO-WAY connection arrays ``mesh.block_m``/``mesh.block_p``, and
        those are built once by ``conn_mesh::reverse_and_sort()``, which
        ``ReservoirBase.init_wells()`` calls right after ``mesh.add_wells()``.
        Before that call the connection list is still open and
        ``conn_mesh::add_conn`` appends to it; after it, it is frozen --
        ``reverse_and_sort()`` doubles ``n_conns`` in place and cannot be run
        twice, and ``n_conns`` is not writable from Python. The only valid
        window is therefore BETWEEN ``add_wells()`` (which assigns
        ``well_head_idx`` / ``well_body_idx``, so block indices exist) and
        ``reverse_and_sort()``, which is exactly where
        :meth:`~darts.models.darts_model.DartsModel._init_wells_with_declared_stencil`
        calls this method.

        A declared coupling that the mesh ALREADY contains must not be added
        again: a duplicate ``(row, col)`` would appear twice in the block row
        and the matrix would be malformed. Existing couplings are reconstructed
        from the Python-side sources of every connection the mesh holds at this
        point (:meth:`_existing_couplings`), and :meth:`verify_stencil` then
        checks the reconstruction against the real, sorted connection arrays
        once ``reverse_and_sort()`` has run -- so the reconstruction is
        VERIFIED at every run, not assumed.

        The added connections carry ``trans = 0`` and ``transD = 0`` and are not
        DFM connections, so they contribute NOTHING to the residual on their
        own: every flux term of the assembly is multiplied by ``tran[conn]``
        (Darcy/advection), ``tranD[conn]`` (diffusion, conduction and
        dispersion) or -- for a DFM connection -- by a phase velocity this one
        does not carry. Their only effect is the pair of off-diagonal blocks
        they add to the pattern for the declaring item to write.

        :param model: the model being initialized
        :param declarers: the result of :meth:`stencil_declarers` (recomputed
            when omitted)
        :returns: tuple of the couplings that were ADDED to the mesh
        """
        declarers = (
            self.stencil_declarers(model) if declarers is None else list(declarers)
        )
        self.declared_couplings = ()
        self.added_couplings = ()
        if not declarers:
            return ()

        mesh = model.reservoir.mesh
        n_blocks = int(mesh.n_blocks)
        n_res_blocks = int(mesh.n_res_blocks)

        declared = []
        owners = {}
        for declarer in declarers:
            for coupling in declarer.declare_stencil(model) or ():
                row, col = (int(coupling[0]), int(coupling[1]))
                for block in (row, col):
                    if not 0 <= block < n_blocks:
                        raise IndexError(
                            f"{type(declarer).__name__}.declare_stencil() declared "
                            f"the coupling ({row}, {col}), but block {block} is "
                            f"outside [0, {n_blocks})."
                        )
                if row == col:
                    raise ValueError(
                        f"{type(declarer).__name__}.declare_stencil() declared the "
                        f"self-coupling ({row}, {col}). The diagonal block of every "
                        "block always exists in the pattern; declare only "
                        "off-diagonal couplings."
                    )
                pair = (min(row, col), max(row, col))
                declared.append(pair)
                owners.setdefault(pair, type(declarer).__name__)

        self.declared_couplings = tuple(dict.fromkeys(declared))
        if not self.declared_couplings:
            return ()

        self._reject_wellhead_couplings(model, self.declared_couplings, owners)

        existing = self._existing_couplings(
            model, self.declared_couplings, n_res_blocks, owners
        )
        added = []
        for pair in self.declared_couplings:
            if pair in existing:
                continue
            mesh.add_conn(pair[0], pair[1], 0.0, 0.0, False)
            existing.add(pair)
            added.append(pair)
        self.added_couplings = tuple(added)
        return self.added_couplings

    @staticmethod
    def _reject_wellhead_couplings(model, couplings, owners):
        """Refuse a coupling that would add a column to a WELLHEAD row.

        The wellhead (ghost) block of every well carries the well-CONTROL
        equations, which ``well_controls.cpp`` writes into a block row it
        assumes holds exactly two column blocks -- its diagonal and the well
        body. ``conn_mesh::add_connection_for_lateral_heat_exchange_for_dfm``
        skips segment 0 for the same reason. An extra column there would be
        overwritten by the control assembly (at best) or mis-addressed (at
        worst), so it is refused here instead.
        """
        wellheads = {
            int(well.well_head_idx): well.name
            for well in getattr(model.reservoir, "wells", ()) or ()
        }
        for pair in couplings:
            for block in pair:
                if block in wellheads:
                    raise ValueError(
                        f"{owners.get(pair, 'A condition item')} declared the "
                        f"coupling {pair}, which attaches block {block} -- the "
                        f"WELLHEAD block of well {wellheads[block]!r} -- to another "
                        "block. The wellhead row carries the well-control "
                        "equations and must keep exactly its diagonal and the "
                        "well-body column; declare the coupling on a well BODY "
                        "segment instead."
                    )

    @staticmethod
    def _existing_couplings(model, declared, n_res_blocks: int, owners: dict) -> set:
        """Which of the DECLARED couplings the connection list already holds.

        Reconstructed from the Python-visible sources of every connection
        ``conn_mesh`` contains at declaration time, because the sorted two-way
        arrays do not exist yet (they are built by ``reverse_and_sort()``, the
        very call this stage must precede) and the one-way arrays are not
        exposed to Python. Every connection has exactly one of these sources:

        * WELL connections -- perforations (well body segment
          ``well_head_idx + 1 + i_w`` to reservoir block ``i_r``), the well
          segment chain ``(head + s, head + s + 1)``, DFM lateral-heat
          connections (``well.connections_for_lateral_heat_transfer``, minus the
          wellhead segment and the segments a perforation already covers), and
          ``reservoir.connected_well_segments``. There are few of these, so they
          are enumerated into a set;
        * RESERVOIR-internal connections -- ``reservoir.cell_m`` /
          ``reservoir.cell_p``, the discretizer output the reservoir keeps.
          There can be millions, so they are never enumerated: a declared
          reservoir-to-reservoir pair is looked up with one vectorized scan
          instead.

        Only the declared pairs are answered, and only reservoir pairs touch the
        big arrays -- the common case (a well segment coupled to a cell) never
        does.

        :raises RuntimeError: when a declared reservoir-to-reservoir coupling
            cannot be checked because the reservoir keeps no connection list.
        """
        from darts.engines import ms_well

        reservoir = model.reservoir
        existing = set()

        for well in getattr(reservoir, "wells", ()) or ():
            head = int(well.well_head_idx)
            perforated_segments = set()
            for perforation in well.perforations:
                i_w, i_r = int(perforation[0]), int(perforation[1])
                block = head + 1 + i_w
                existing.add((min(block, i_r), max(block, i_r)))
                perforated_segments.add(i_w + 1)
            n_segment_conns = (
                int(well.num_segments) - 1
                if well.ms_type == ms_well.MS_Type.DFM
                else int(well.n_segments)
            )
            for segment in range(n_segment_conns):
                existing.add((head + segment, head + segment + 1))
            if getattr(well, "with_lateral_heat_transfer", False):
                for lateral in well.connections_for_lateral_heat_transfer:
                    i_w, i_r = int(lateral[0]), int(lateral[1])
                    if i_w == 0 or i_w in perforated_segments:
                        continue  # skipped by add_connection_for_lateral_heat_exchange_for_dfm
                    block = head + i_w
                    existing.add((min(block, i_r), max(block, i_r)))

        for pair, segments in (
            getattr(reservoir, "connected_well_segments", None) or {}
        ).items():
            first = reservoir.get_well(pair[0])
            second = reservoir.get_well(pair[1])
            for seg_1, seg_2 in segments:
                a = int(first.well_head_idx) + int(seg_1)
                b = int(second.well_head_idx) + int(seg_2)
                existing.add((min(a, b), max(a, b)))

        # reservoir-to-reservoir pairs: one vectorized lookup each, no enumeration
        reservoir_pairs = [
            pair for pair in declared if pair not in existing and pair[1] < n_res_blocks
        ]
        if reservoir_pairs:
            cell_m = getattr(reservoir, "cell_m", None)
            cell_p = getattr(reservoir, "cell_p", None)
            if cell_m is None or cell_p is None:
                raise RuntimeError(
                    f"{owners[reservoir_pairs[0]]} declared the reservoir-to-"
                    f"reservoir coupling {reservoir_pairs[0]}, but "
                    f"{type(reservoir).__name__} does not expose its connection "
                    "list (cell_m/cell_p), so the framework cannot tell whether "
                    "that coupling already exists -- and adding a duplicate "
                    "connection would corrupt the matrix."
                )
            cell_m = np.asarray(cell_m, dtype=np.int64).ravel()
            cell_p = np.asarray(cell_p, dtype=np.int64).ravel()
            low = np.minimum(cell_m, cell_p)
            high = np.maximum(cell_m, cell_p)
            for pair in reservoir_pairs:
                if np.any((low == pair[0]) & (high == pair[1])):
                    existing.add(pair)

        return existing

    def verify_stencil(self, model) -> "ConditionSet":
        """Check the declared couplings against the FROZEN connection arrays.

        Runs after ``reverse_and_sort()``, when ``mesh.block_m``/``block_p`` are
        the authoritative, sorted, two-way connection list the engine will build
        its pattern from. It verifies that every declared coupling is present
        exactly ONCE in each direction -- which fails loudly both when a needed
        coupling was not added (the item would write into a block that does not
        exist) and when a duplicate was created (the reconstruction in
        :meth:`_existing_couplings` missed an existing connection, and the
        matrix would be malformed).

        :param model: the model being initialized
        :raises RuntimeError: on a missing or duplicated declared coupling
        """
        if not self.declared_couplings:
            return self
        mesh = model.reservoir.mesh
        block_m = np.asarray(mesh.block_m, dtype=np.int64)
        block_p = np.asarray(mesh.block_p, dtype=np.int64)
        for pair in self.declared_couplings:
            forward = int(np.count_nonzero((block_m == pair[0]) & (block_p == pair[1])))
            reverse = int(np.count_nonzero((block_m == pair[1]) & (block_p == pair[0])))
            if forward == 1 and reverse == 1:
                continue
            added = "added" if pair in self.added_couplings else "already present"
            raise RuntimeError(
                f"The declared Jacobian coupling {pair} ({added} by the stencil "
                f"declaration stage) appears {forward} time(s) as "
                f"({pair[0]}, {pair[1]}) and {reverse} time(s) as "
                f"({pair[1]}, {pair[0]}) in the frozen mesh connection list, "
                "expected exactly one of each. "
                + (
                    "A count of 2 means the coupling was added although the mesh "
                    "already contained it (duplicate columns in the block row); a "
                    "count of 0 means it was not added at all."
                )
            )
        return self

    @staticmethod
    def _refuse_legacy_channels(model):
        """Refuse a model still carrying ``set_rhs_flux`` / ``rhs_flux_hooks``.

        Both were removed from
        :class:`~darts.models.darts_model.DartsModel`: ``apply_rhs_flux()`` is
        the conditions stage plus the observer stage and nothing else, and no
        hook list is consulted. ``self.rhs_flux_hooks.append(...)`` therefore
        fails loudly by itself (there is no such attribute), but the two
        remaining spellings would not:

        * a subclass that still overrides ``set_rhs_flux()`` -- nothing calls
          it, so the model runs on and quietly omits its source term;
        * a subclass that assigns its own ``self.rhs_flux_hooks = []`` and
          appends to it -- the appends succeed and nothing reads the list.

        Both are refused here, at ``init()`` time, with the migration named.
        """
        offenders = []
        if callable(getattr(type(model), "set_rhs_flux", None)):
            offenders.append(
                "overrides set_rhs_flux(), which nothing calls any more: its "
                "return value would be dropped and the model would run on "
                "without the source term"
            )
        if hasattr(model, "rhs_flux_hooks"):
            offenders.append(
                "carries a rhs_flux_hooks attribute, which nothing reads any "
                "more: whatever is appended to it would never be applied"
            )
        if not offenders:
            return
        raise RuntimeError(
            f"{type(model).__name__} "
            + "; and ".join(offenders)
            + ". The set_rhs_flux() override and the rhs_flux_hooks list "
            "were REMOVED in favour of model.conditions: "
            "register a ConditionItem (CellSource, SegmentSource, "
            "PipeSourceTerm, InterfaceFlux, DirichletPin, or a purpose-built "
            "subclass) with self.conditions.add(...) -- from set_wells() if it "
            "declares a Jacobian stencil, otherwise from "
            "set_boundary_conditions(). NOTE the sign convention: the legacy "
            "vector was applied as rhs += value * dt, so a positive entry "
            "REMOVED mass from the cell, whereas CellSource rates are positive "
            "INTO the cell. See the Conditions page of the technical reference."
        )

    def compile(self, model) -> "ConditionSet":
        """Validate the set against the model/engine and bind every item.

        Raises with a specific message when (0) the model still carries one of
        the REMOVED legacy channels (a ``set_rhs_flux`` override or a
        ``rhs_flux_hooks`` attribute -- its contribution would now be dropped
        without a trace), (i) the model overrides ``apply_rhs_flux`` (items
        would silently never run), (ii) a Jacobian-providing item runs off-CPU
        or on an engine without an exposed block-CSR matrix, (iii) the model
        runs a mechanics engine (rows are rescaled inside assembly,
        post-assembly writes would be mis-scaled), or (iv) the
        adjoint/history-matching driver is active and an item is not
        ``adjoint_transparent``. It then rejects any additive/replacement row
        conflict (review item E6).

        BINDING IS DEFERRED WHEN THE ENGINE DOES NOT EXIST YET. ``init()`` runs
        this at its end, but on the RESTART path it has not called
        :meth:`~darts.models.darts_model.DartsModel.reset` -- the engine is
        initialized later, inside
        :meth:`~darts.models.darts_model.DartsModel.load_restart_data`. Binding
        there would resolve CSR positions against a matrix that does not exist,
        so the set records :attr:`deferred` and ``load_restart_data`` re-runs
        ``compile()`` once the engine is up.
        """
        self._model = model
        # The engine caches PV = volume * poro ONCE, in engine.init() (run by
        # DartsModel.reset() from DartsModel.init()). compile() is the single
        # point that runs after it for EVERY model, so latch the reservoir cell
        # volumes here: later writes would be silently ignored by the engine and
        # the reservoir entry points now say so instead. Skipped when the engine
        # has not been initialized yet (e.g. the restart flow, which initializes
        # it in load_restart_data() after init() returned).
        self._freeze_reservoir_pore_volumes(model)

        # (0) The removed legacy channels. This check runs BEFORE the
        # no-items early return on purpose: the model it is for registers
        # nothing here, and that is exactly what makes it dangerous.
        # ``DartsModel`` defines neither name any more, so a model that still
        # has one is unmigrated, and its source term would now be dropped
        # silently -- the one failure mode the removal must not produce.
        self._refuse_legacy_channels(model)

        if not self.items and not self.observers:
            self.deferred = False
            return self

        from darts.models.darts_model import DartsModel
        from darts.nonlinear_solvers.mechanics import MechanicsNewtonSolver

        # (i) an overridden apply_rhs_flux bypasses the conditions stage
        if type(model).apply_rhs_flux is not DartsModel.apply_rhs_flux:
            registered = (
                f"{len(self.items)} registered condition item(s)"
                if self.items
                else f"{len(self.observers)} registered observer(s)"
            )
            raise RuntimeError(
                f"{type(model).__name__} overrides apply_rhs_flux(), so the "
                f"{registered} would silently never run. Move a post-assembly "
                "policy check into a NonlinearIterationObserver (or the legacy "
                "DartsModel.after_assembly(dt, t) hook), and migrate a "
                "source/flux override into condition items -- apply_rhs_flux() "
                "is the framework's own entry point and is not an extension "
                "point."
            )

        if not self.items:
            self.deferred = False
            return self

        if getattr(model, "restart", False) and not self._engine_initialized(model):
            # restart path: init() skipped reset(), so the engine (and its
            # Jacobian) does not exist yet. Bind in load_restart_data(), which
            # resets the engine and re-runs compile().
            self.deferred = True
            return self
        self.deferred = False

        # (iii) mechanics engines rescale equation rows inside assembly
        if isinstance(getattr(model, "nonlinear_solver", None), MechanicsNewtonSolver):
            raise RuntimeError(
                "Condition items are not supported on mechanics models yet: "
                "the pm/mech engines rescale equation rows inside assembly, so "
                "post-assembly RHS/Jacobian writes would be mis-scaled."
            )

        engine = model.physics.engine

        # (iv) adjoint guard -- see assert_adjoint_supported(). Checked here AND
        # on every apply(), because the optimization driver sets the flag after
        # init() has already run.
        self.assert_adjoint_supported(engine)

        # (ii) Jacobian-providing items need the CPU block-CSR matrix
        jac_items = [item for item in self.items if item.provides_jacobian]
        if jac_items:
            names = ", ".join(type(item).__name__ for item in jac_items)
            if model.platform != "cpu":
                raise RuntimeError(
                    f"Condition item(s) {names} provide Jacobian contributions, "
                    f"which are CPU-only (model platform is '{model.platform}')."
                )
            try:
                BlockCSRView(engine, model.physics.n_vars)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Condition item(s) {names} provide Jacobian contributions "
                    f"but the engine does not expose one: {exc}"
                ) from exc

        for item in self.items:
            if (
                item.requires_platform is not None
                and model.platform != item.requires_platform
            ):
                raise RuntimeError(
                    f"Condition item {type(item).__name__} requires "
                    f"platform='{item.requires_platform}' but the model runs "
                    f"on '{model.platform}'."
                )

        for item in self.items:
            item.bind(model)

        self._check_row_conflicts(model)
        return self

    def _check_row_conflicts(self, model):
        """Reject additive/replacement row conflicts (review item E6).

        A row CLAIMED by a replacement item admits exactly one claimant and no
        additive contribution. Both failures are silent otherwise -- whichever
        item happens to run last wins -- so they are refused at ``init()`` time,
        naming both items and the offending ``(block, equation)``.
        """
        n_vars = int(model.physics.n_vars)
        claims, additions = [], []
        for item in self.items:
            if item.contribution == NO_CONTRIBUTION:
                continue
            written = item.written_rows(model)
            if written is None:
                continue
            rows = np.asarray(written, dtype=np.int64).ravel()
            if not rows.size:
                continue
            target = claims if item.contribution == REPLACEMENT else additions
            target.append((item, rows))

        if not claims:
            return  # nothing is claimed, so nothing can conflict
        claimed = np.concatenate([rows for _, rows in claims])

        # (a) two claims on the same row
        unique, counts = np.unique(claimed, return_counts=True)
        duplicated = unique[counts > 1]
        if duplicated.size:
            row = int(duplicated[0])
            first, second = self._two_owners(claims, row)
            raise RuntimeError(
                self._conflict_message(
                    "Two condition items claim the same equation row",
                    row,
                    n_vars,
                    first,
                    second,
                    "A row replaced by a constraint admits exactly one claimant: "
                    "the second claim would silently overwrite the first.",
                )
            )

        # (b) an additive contribution to a claimed row
        for item, rows in additions:
            overlap = rows[np.isin(rows, unique)]
            if overlap.size:
                row = int(overlap[0])
                claimant, _ = self._two_owners(claims, row)
                raise RuntimeError(
                    self._conflict_message(
                        "A condition item contributes additively to an equation row "
                        "another item claims as a constraint",
                        row,
                        n_vars,
                        claimant,
                        item,
                        "An additive contribution to a claimed row is either "
                        "silently discarded (the constraint overwrites it) or "
                        "corrupts the constraint, depending on registration order.",
                    )
                )

    @staticmethod
    def _two_owners(claims, row: int):
        """The first (and, when present, second) claimant of ``row``."""
        owners = [item for item, rows in claims if np.any(rows == row)]
        return owners[0], (owners[1] if len(owners) > 1 else owners[0])

    @staticmethod
    def _conflict_message(headline, row, n_vars, first, second, explanation):
        return (
            f"{headline}: block {row // n_vars}, equation {row % n_vars} "
            f"(flat row {row}). {type(first).__name__} claims it; "
            f"{type(second).__name__} also writes it. {explanation}"
        )

    @staticmethod
    def _engine_initialized(model) -> bool:
        """True when ``engine.init()`` has run (the state vector is allocated)."""
        engine = getattr(getattr(model, "physics", None), "engine", None)
        try:
            return len(engine.X) > 0
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _freeze_reservoir_pore_volumes(model):
        """Tell the reservoir that the engine has cached ``PV = volume * poro``."""
        reservoir = getattr(model, "reservoir", None)
        freeze = getattr(reservoir, "freeze_pore_volumes", None)
        if not callable(freeze):
            return
        if ConditionSet._engine_initialized(model):
            freeze()

    def assert_adjoint_supported(self, engine):
        """Refuse to contribute to a system whose gradient will not see it.

        The C++ adjoint replays assembly from the stored trajectory and knows
        nothing about Python-side contributions, so an opaque item yields a
        silently incomplete gradient rather than an error.

        This is deliberately re-checked on every ``apply()`` and not only at
        compile time: ``OptModuleSettings`` sets ``engine.opt_history_matching``
        AFTER ``init()`` has returned and then calls ``reset()``, so a
        compile-time-only check never fires in the driver's own call order.
        """
        if not self.items or not getattr(engine, "opt_history_matching", False):
            return
        opaque = [
            type(item).__name__ for item in self.items if not item.adjoint_transparent
        ]
        if opaque:
            raise RuntimeError(
                "The adjoint/history-matching driver is active "
                "(engine.opt_history_matching) but these condition items are "
                "not adjoint_transparent: " + ", ".join(opaque) + ". Their "
                "contributions would be missing from the adjoint gradient."
            )

    def apply(self, ctx: AssemblyContext):
        if self.items and self._model is not None:
            self.assert_adjoint_supported(self._model.physics.engine)
        for item in self.items:
            item.rebind_if_stale(ctx)
            item.apply(ctx)
        self.observe(ctx)

    def observe(self, ctx: AssemblyContext):
        """Run every registered :class:`NonlinearIterationObserver`.

        Called at the end of :meth:`apply` with a READ-ONLY twin of the context
        (see :meth:`AssemblyContext.read_only`), so what an observer sees is the
        final assembled system of the iteration and what it can do to it is
        nothing.
        """
        if not self.observers:
            return
        read_only = ctx.read_only()
        for observer in self.observers:
            observer.observe(read_only)

    def project_state(self, t: float):
        """Run every item's pre-assembly state projection (see
        :meth:`ConditionItem.project_state`).

        A model with its own Newton loop calls this immediately before
        ``engine.assemble_linear_system(dt)``; it is a no-op for every item type
        that does not constrain the state, and for an empty set.

        :param t: time at the START of the timestep [days].
        """
        for item in self.items:
            item.project_state(t)

    def on_timestep_start(self, dt: float, t: float):
        for item in self.items:
            item.on_timestep_start(dt, t)
        for observer in self.observers:
            observer.on_timestep_start(dt, t)

    def on_timestep_converged(self, dt: float, t: float):
        for item in self.items:
            item.on_timestep_converged(dt, t)
        for observer in self.observers:
            observer.on_timestep_converged(dt, t)

    def on_timestep_failed(self, dt: float, t: float):
        for item in self.items:
            item.on_timestep_failed(dt, t)
        for observer in self.observers:
            observer.on_timestep_failed(dt, t)

    # ------------------------------------------------------------ restart (E9)
    #: sidecar file name suffix appended to the restart (reservoir) file path
    RESTART_SIDECAR_SUFFIX = ".conditions.json"

    @classmethod
    def restart_sidecar_path(cls, reservoir_filepath: str) -> str:
        """Path of the condition-state sidecar next to a restart file."""
        return str(reservoir_filepath) + cls.RESTART_SIDECAR_SUFFIX

    def stateful_items(self) -> list:
        """Items that declared :attr:`ConditionItem.carries_restart_state`."""
        return [item for item in self.items if item.carries_restart_state]

    def save_restart_state(self, reservoir_filepath: str):
        """Write the sidecar holding every stateful item's serialized state.

        Called by :meth:`~darts.models.darts_model.DartsModel.save_restart_state`.
        Writes nothing when no item carries state.

        :param reservoir_filepath: the restart (``.h5``) file the sidecar
            accompanies
        :returns: the sidecar path, or ``None`` when nothing was written
        """
        stateful = self.stateful_items()
        if not stateful:
            return None
        payload = {
            "items": [
                {
                    "index": self.items.index(item),
                    "type": type(item).__name__,
                    "state": item.save_restart_state(),
                }
                for item in stateful
            ]
        }
        path = self.restart_sidecar_path(reservoir_filepath)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=1)
        return path

    def load_restart_state(self, reservoir_filepath: str):
        """Restore every stateful item, or REFUSE the restart (review item E9).

        The restart file carries reservoir block data and the OBL history
        columns -- nothing else. An item that declares
        :attr:`ConditionItem.carries_restart_state` therefore has its state
        restored from the sidecar written by :meth:`save_restart_state`, and if
        that sidecar is missing, or does not describe the same set of items, the
        restart is REFUSED with a message that says what to do. Continuing would
        restart the item from its constructor defaults while the reservoir state
        is two years old -- a silently wrong answer, which is the one outcome
        the contract must not allow.

        :param reservoir_filepath: the restart file being loaded
        :raises RuntimeError: when a stateful item cannot be restored
        """
        stateful = self.stateful_items()
        if not stateful:
            return
        path = self.restart_sidecar_path(reservoir_filepath)
        names = ", ".join(type(item).__name__ for item in stateful)
        if not os.path.exists(path):
            raise RuntimeError(
                f"Cannot restart: condition item(s) {names} declare "
                "carries_restart_state = True, but the state sidecar "
                f"'{path}' does not exist. The restart file holds reservoir "
                "block data only, so their internal state cannot be recovered "
                "from it and restarting would silently continue from the "
                "constructor defaults. Write the sidecar in the original run "
                "(DartsModel.save_restart_state(<restart file>), alongside every "
                "save_data_to_h5), or make the item stateless."
            )
        with open(path) as handle:
            payload = json.load(handle)
        by_index = {int(entry["index"]): entry for entry in payload.get("items", ())}
        for item in stateful:
            index = self.items.index(item)
            entry = by_index.get(index)
            if entry is None or entry["type"] != type(item).__name__:
                found = entry["type"] if entry else "nothing"
                raise RuntimeError(
                    f"Cannot restart: the state sidecar '{path}' describes "
                    f"{found} at condition index {index}, but the model "
                    f"registered a {type(item).__name__} there. The sidecar was "
                    "written by a model with a different set of conditions; "
                    "restarting would restore one item's state into another."
                )
            item.load_restart_state(entry["state"])


def _callable_takes_states(func) -> bool:
    """True when ``func`` accepts a second positional argument (the states)."""
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in parameters
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters
    )
    return len(positional) >= 2 or has_var_positional


class CellSource(ConditionItem):
    """Source/sink rates in a fixed set of blocks (vectorized by the framework).

    :param cells: the blocks the rates apply to: an int array of block indices,
        or a :class:`Selector` resolved at bind time.
    :param rates: injection rates ``q`` per block, positive INTO the block, in
        engine residual units per day (e.g. kmol/day per component equation,
        kJ/day for the energy equation). One of:

        - a constant array of shape ``(n_cells, n_vars)``,
        - a callable ``f(t) -> array (n_cells, n_vars)``,
        - a callable ``f(t, states) -> array (n_cells, n_vars)`` where
          ``states`` is the current ``(n_cells, n_vars)`` state of the cells.
    :param d_rates: optional derivative ``d q / d state`` enabling the
        analytic diagonal-block Jacobian: a constant array or callable
        ``f(t, states)`` of shape ``(n_cells, n_vars, n_vars)``, equation-major
        (``d_rates[k, c, v] = d q_c / d x_v`` of cell ``k``). CPU-only.

    The framework applies ``rhs[b * n_vars + c] -= q_c * dt`` and, when
    ``d_rates`` is given, ``jac_diag(b) -= d_rates[k] * dt``.
    """

    def __init__(self, cells, rates, d_rates=None):
        # A Selector is kept as-is and resolved in bind(); an index array is
        # materialized now, exactly as before.
        self.cells = (
            cells
            if isinstance(cells, Selector)
            else np.asarray(cells, dtype=np.int64).ravel()
        )
        self.rates = rates
        self.d_rates = d_rates
        self.provides_jacobian = d_rates is not None
        self._rates_take_states = callable(rates) and _callable_takes_states(rates)
        self._rhs_idx = None
        self._diag_pos = None
        self._diag_flat_idx = None
        self._n_vars = None

    def written_rows(self, model):
        if self._rhs_idx is None:  # not bound yet: nothing resolved to report
            return ()
        return self._rhs_idx

    def bind(self, model):
        n_vars = model.physics.n_vars
        n_blocks = model.reservoir.mesh.n_blocks
        self.cells = resolve_blocks(self.cells, model)
        if len(self.cells) and (self.cells.min() < 0 or self.cells.max() >= n_blocks):
            raise IndexError(
                f"{type(self).__name__}: cell indices must be within [0, {n_blocks})."
            )
        self._n_vars = n_vars
        self._rhs_idx = (
            self.cells[:, None] * n_vars + np.arange(n_vars)[None, :]
        ).ravel()
        if not callable(self.rates):
            self.rates = np.asarray(self.rates, dtype=float)
            self._check_shape(self.rates, (len(self.cells), n_vars), "rates")
        if self.provides_jacobian:
            view = BlockCSRView(model.physics.engine, n_vars)
            self._diag_pos = np.array(
                [view.diag_pos(int(b)) for b in self.cells], dtype=np.int64
            )
            # Flat jac_vals indices of every entry of every diagonal block, so
            # apply() scatters once instead of calling add_block() per cell:
            # the loop cost grows with the number of cells, which is exactly
            # what a contribution is not allowed to do (review item E10).
            self._diag_flat_idx = (
                self._diag_pos[:, None] * view.block_size
                + np.arange(view.block_size)[None, :]
            ).ravel()
            if not callable(self.d_rates):
                self.d_rates = np.asarray(self.d_rates, dtype=float)
                self._check_shape(
                    self.d_rates, (len(self.cells), n_vars, n_vars), "d_rates"
                )
            self.stamp_pattern(model)

    def _check_shape(self, array, expected, name):
        if array.shape != expected:
            raise ValueError(
                f"{type(self).__name__}: {name} has shape {array.shape}, "
                f"expected {expected}."
            )

    def _states(self, ctx: AssemblyContext) -> np.ndarray:
        return ctx.X[self._rhs_idx].reshape(len(self.cells), self._n_vars)

    def evaluate_rates(self, ctx: AssemblyContext) -> np.ndarray:
        if callable(self.rates):
            if self._rates_take_states:
                rates = np.asarray(self.rates(ctx.t, self._states(ctx)), dtype=float)
            else:
                rates = np.asarray(self.rates(ctx.t), dtype=float)
            self._check_shape(rates, (len(self.cells), self._n_vars), "rates")
            return rates
        return self.rates

    def evaluate_d_rates(self, ctx: AssemblyContext) -> np.ndarray:
        if callable(self.d_rates):
            d_rates = np.asarray(self.d_rates(ctx.t, self._states(ctx)), dtype=float)
            self._check_shape(
                d_rates,
                (len(self.cells), self._n_vars, self._n_vars),
                "d_rates",
            )
            return d_rates
        return self.d_rates

    def apply(self, ctx: AssemblyContext):
        if not len(self.cells):
            return
        rates = self.evaluate_rates(ctx)
        ctx.rhs[self._rhs_idx] -= (rates * ctx.dt).ravel()
        if self.provides_jacobian:
            d_rates = self.evaluate_d_rates(ctx)
            # np.add.at rather than += : a cell may legitimately appear twice
            # in `cells`, and the duplicate contributions must accumulate.
            np.add.at(
                ctx.jac.jac_vals,
                self._diag_flat_idx,
                (-ctx.dt * d_rates).ravel(),
            )


class SegmentSource(CellSource):
    """Wellhead-anchored :class:`CellSource` over the BODY segments of a well.

    The constructor takes the well name; :meth:`bind` resolves
    ``well_head_idx``/``num_segments`` via ``model.reservoir.get_well``. The
    rates apply ONLY to segments ``1 .. num_segments - 1``: the wellhead block
    (segment 0) rows carry the well-control equations and must never receive
    source terms — this exclusion is structural, not optional. Rate arrays /
    callables therefore have ``n_cells = num_segments - 1`` rows, row ``k``
    addressing segment ``k + 1``.
    """

    def __init__(self, well_name: str, rates, d_rates=None):
        super().__init__(
            cells=np.empty(0, dtype=np.int64), rates=rates, d_rates=d_rates
        )
        self.well_name = well_name

    def bind(self, model):
        well = model.reservoir.get_well(self.well_name)
        if well is None:
            available = ", ".join(w.name for w in model.reservoir.wells)
            raise KeyError(
                f"SegmentSource: well '{self.well_name}' not found; available "
                f"wells: [{available}]."
            )
        # segment 0 (the wellhead block) is structurally excluded — its rows
        # are the well-control equations
        self.cells = well.well_head_idx + np.arange(
            1, well.num_segments, dtype=np.int64
        )
        super().bind(model)


class PipeSourceTerm(ConditionItem):
    """Component/energy source of a DFM pipe source/sink, in its segment block.

    This is the unified item that replaced the hand-rolled ``set_rhs_flux``
    overrides of the DFM well models. The rate schedule stays where it belongs
    — in the pipe's source/sink object (a
    :class:`~darts.pipes.ramp_up_rate.RampUpRate` or subclass), which
    :class:`~darts.pipes.pipe.Pipe` updates once per timestep — and this item
    only turns the current rate into the residual contribution of the receiving
    segment block:

    - the block is ``mesh.n_res_blocks + source_sink.segment_idx``, the
      addressing the models used;
    - the rates come from
      :meth:`~darts.pipes.ramp_up_rate.RampUpRate.get_component_energy_rates`
      (component rates in kmol/day plus, for thermal physics, the energy rate
      in kJ/day including the potential energy of the receiving block);
    - the framework sign convention applies: ``rhs[block] -= rates * dt``,
      which is what those overrides wrote as ``rhs_flux[block] = -rates``
      followed by ``rhs += rhs_flux * dt`` on the caller side.

    RHS-only (``provides_jacobian = False``), exactly as the legacy path: for
    the plain ramp-up schedule the rate does not depend on the state, so the
    analytic Jacobian contribution is zero. Rate models that DO depend on the
    segment state (e.g. the choke boundary node) are therefore lagged by one
    Newton iteration; giving them derivative blocks is the intended follow-up.

    :param well_name: key of the pipe in ``model.wells``.
    :param source_sink_name: key of the source/sink in ``pipe.source_sinks``.
    """

    def __init__(self, well_name: str, source_sink_name: str):
        self.well_name = well_name
        self.source_sink_name = source_sink_name
        self.source_sink = None
        self.block = None
        self._mesh = None
        self._physics = None
        self._thermal = False
        self._rhs_slice = None

    def written_rows(self, model):
        if self.block is None:  # not bound yet
            return ()
        n_vars = int(model.physics.n_vars)
        return range(self.block * n_vars, (self.block + 1) * n_vars)

    def bind(self, model):
        wells = getattr(model, "wells", None) or {}
        if self.well_name not in wells:
            available = ", ".join(sorted(wells))
            raise KeyError(
                f"PipeSourceTerm: pipe '{self.well_name}' not found in "
                f"model.wells; available pipes: [{available}]."
            )
        source_sinks = getattr(wells[self.well_name], "source_sinks", None) or {}
        if self.source_sink_name not in source_sinks:
            available = ", ".join(sorted(source_sinks))
            raise KeyError(
                f"PipeSourceTerm: source/sink '{self.source_sink_name}' not "
                f"found on pipe '{self.well_name}'; available source/sinks: "
                f"[{available}]."
            )
        self.source_sink = source_sinks[self.source_sink_name]

        self._mesh = model.reservoir.mesh
        self._physics = model.physics
        self._thermal = bool(model.physics.thermal)
        n_vars = model.physics.n_vars
        self.block = self._mesh.n_res_blocks + self.source_sink.segment_idx
        start = self.block * n_vars
        self._rhs_slice = slice(start, start + n_vars)

        n_rates = len(
            np.atleast_1d(
                self.source_sink.get_component_energy_rates(model.physics, 0.0)
            )
        )
        if n_rates != n_vars:
            raise ValueError(
                f"PipeSourceTerm: source/sink '{self.source_sink_name}' returns "
                f"{n_rates} rate(s) but the physics has {n_vars} equations per block."
            )

    def apply(self, ctx: AssemblyContext):
        # Specific potential energy of the receiving block; read only for
        # thermal physics, as the isothermal models did.
        specific_potential_energy = (
            self._mesh.cell_spe[self.block] if self._thermal else 0.0
        )
        rates = self.source_sink.get_component_energy_rates(
            self._physics, specific_potential_energy
        )
        ctx.rhs[self._rhs_slice] -= rates * ctx.dt


class InterfaceFlux(ConditionItem):
    """Two-block coupling flux with analytic 4-block derivatives (CPU-only).

    Subclasses provide the per-connection flux and its derivatives; the
    framework owns the CSR positions, the state gather / residual scatter and
    the dt scaling. ``connections`` is a list of ``(row_block, col_block)``
    pairs; either member may be a :class:`Selector` resolving to exactly one
    block.

    The pairs no longer have to be neighbours in the mesh: this item DECLARES
    them (:meth:`declare_stencil`), so a pair the mesh does not connect is added
    as a zero-transmissibility connection before the engine allocates its matrix
    and its four blocks exist by the time :meth:`bind` resolves their positions.
    Declaring a coupling the mesh already has is free -- the declaration stage
    adds nothing for it.

    Subclasses implement :meth:`connection_flux` returning
    ``(flux, d_flux_d_row, d_flux_d_col)`` where ``flux`` (shape
    ``(n_vars,)``) is the rate INTO ``row_block`` (and out of ``col_block``)
    per day, and the derivative blocks (shape ``(n_vars, n_vars)``,
    equation-major) differentiate the flux w.r.t. the two block states. The
    framework then applies::

        rhs[row] -= flux * dt        rhs[col] += flux * dt
        jac[row, row] -= d_row * dt  jac[row, col] -= d_col * dt
        jac[col, row] += d_row * dt  jac[col, col] += d_col * dt
    """

    provides_jacobian = True

    def __init__(self, connections):
        self._raw_connections = list(connections)
        # Either member of a pair may be a Selector, which needs a model to
        # resolve; plain indices are materialized right away, as before.
        self.connections = (
            None
            if any(
                isinstance(block, Selector)
                for pair in self._raw_connections
                for block in pair
            )
            else [(int(r), int(c)) for r, c in self._raw_connections]
        )
        self._positions = None
        self._n_vars = None

    def resolve_connections(self, model):
        """Materialize the connection list, resolving any :class:`Selector`.

        A selector used as a connection member must resolve to exactly ONE
        block: a connection couples one block to one other block.
        """
        if self.connections is not None:
            return self.connections
        resolved = []
        for pair in self._raw_connections:
            blocks = []
            for block in pair:
                indices = resolve_blocks(block, model)
                if indices.size != 1:
                    raise ValueError(
                        f"{type(self).__name__}: {type(block).__name__} selected "
                        f"{indices.size} blocks for one side of a connection; a "
                        "connection couples exactly one block to one other block."
                    )
                blocks.append(int(indices[0]))
            resolved.append((blocks[0], blocks[1]))
        self.connections = resolved
        return self.connections

    def declare_stencil(self, model):
        """Declare every connection's off-diagonal pair (see :class:`InterfaceFlux`)."""
        return tuple(self.resolve_connections(model))

    def written_rows(self, model):
        n_vars = int(model.physics.n_vars)
        rows = []
        for row, col in self.resolve_connections(model):
            rows.extend(range(row * n_vars, (row + 1) * n_vars))
            rows.extend(range(col * n_vars, (col + 1) * n_vars))
        return rows

    def bind(self, model):
        n_vars = model.physics.n_vars
        self._n_vars = n_vars
        self.resolve_connections(model)
        view = BlockCSRView(model.physics.engine, n_vars)
        self._positions = [
            (
                view.diag_pos(row),
                view.block_pos(row, col),
                view.block_pos(col, row),
                view.diag_pos(col),
            )
            for row, col in self.connections
        ]
        self.stamp_pattern(model)

    def connection_flux(self, t: float, state_row: np.ndarray, state_col: np.ndarray):
        """Return ``(flux, d_flux_d_row, d_flux_d_col)`` for one connection."""
        raise NotImplementedError(
            f"{type(self).__name__}.connection_flux() is not implemented"
        )

    def apply(self, ctx: AssemblyContext):
        n_vars = self._n_vars
        for (row, col), (pos_rr, pos_rc, pos_cr, pos_cc) in zip(
            self.connections, self._positions, strict=True
        ):
            state_row = ctx.X[row * n_vars : (row + 1) * n_vars]
            state_col = ctx.X[col * n_vars : (col + 1) * n_vars]
            flux, d_row, d_col = self.connection_flux(ctx.t, state_row, state_col)
            q_dt = np.asarray(flux, dtype=float) * ctx.dt
            ctx.rhs[row * n_vars : (row + 1) * n_vars] -= q_dt
            ctx.rhs[col * n_vars : (col + 1) * n_vars] += q_dt
            d_row_dt = ctx.dt * np.asarray(d_row, dtype=float)
            d_col_dt = ctx.dt * np.asarray(d_col, dtype=float)
            ctx.jac.add_block(pos_rr, -d_row_dt)
            ctx.jac.add_block(pos_rc, -d_col_dt)
            ctx.jac.add_block(pos_cr, d_row_dt)
            ctx.jac.add_block(pos_cc, d_col_dt)


class ConstantStateBC(ConditionItem):
    """Declared open / constant-state far field (the huge-boundary-volume trick).

    ~13 models express an open boundary by giving the outermost cells an
    enormous volume: their pore volume then dwarfs anything that flows in or
    out over the simulated time, so their state stays (numerically) constant
    and the boundary behaves as an infinite-acting aquifer / constant-pressure
    far field. The mechanism is a MESH property, not a residual contribution,
    so this item deliberately writes NOTHING in :meth:`apply`.

    What it adds is the missing half of that trick: a place that SEES it and
    CHECKS it. The volumes are written by the reservoir exactly where they are
    written today (``reservoir.boundary_volumes`` applied inside
    ``StructReservoir.discretize()``, ``CPG_Reservoir.set_boundary_volume()``
    plus ``apply_volume_depth()``); this item reads that dict at bind time and
    validates that the values actually reached ``mesh.volume`` before the
    engine cached the pore volumes.

    The ordering matters and is invisible otherwise: the engine caches
    ``PV = volume * poro`` ONCE, in ``engine.init()``, which ``DartsModel.init()``
    runs (via :meth:`~darts.models.darts_model.DartsModel.reset`) AFTER
    ``set_boundary_conditions()``. A boundary volume written any later — in
    ``set_boundary_conditions()``, or after ``init()`` returned — is silently
    ignored and the model quietly runs with a CLOSED boundary. Registering this
    item turns that silent wrong answer into an error at ``init()`` time::

        self.reservoir.boundary_volumes['yz_minus'] = 1e8   # in set_reservoir()
        ...
        self.conditions.add(ConstantStateBC(faces='yz_minus'))

    (The complementary guard lives in the reservoir: once the engine ran,
    :meth:`~darts.reservoirs.reservoir_base.ReservoirBase.freeze_pore_volumes`
    makes further boundary-volume writes raise instead of vanish.)

    :param faces: face name or iterable of face names that must carry a
        far-field volume; ``None`` (default) means "whichever faces the
        reservoir has a boundary volume for", and then at least one is required.
    :param mode: ``"volume"`` (default) — the huge-volume mechanism.
        ``"dirichlet"`` (pinning the boundary state through a penalized
        diagonal) is reserved for the ``DirichletPin`` follow-up and raises
        :class:`NotImplementedError`.
    :param rtol: relative tolerance used to recognize the requested volume in
        ``mesh.volume``.

    :ivar volumes: ``face -> requested volume``, resolved at bind time.
    :ivar n_cells: ``face -> number of mesh cells carrying that volume``.
    """

    #: the six face slabs of the structured reservoir family
    FACES = ("xy_minus", "xy_plus", "yz_minus", "yz_plus", "xz_minus", "xz_plus")

    provides_jacobian = False
    # The item contributes nothing to the residual/Jacobian, so the system the
    # adjoint differentiates is exactly the one it always was.
    adjoint_transparent = True
    # Declarative: neither an additive contribution nor a row claim.
    contribution = NO_CONTRIBUTION

    def __init__(self, faces=None, mode: str = "volume", rtol: float = 1e-9):
        if mode == "dirichlet":
            raise NotImplementedError(
                "ConstantStateBC(mode='dirichlet') is not implemented: this item "
                "declares the huge-boundary-volume far field, which is a mesh "
                "property, and pinning the boundary state instead is a different "
                "condition with different results. Use the default mode='volume' "
                "(the huge-boundary-volume far field), or register a DirichletPin "
                "on the boundary cells explicitly if that is what you mean."
            )
        if mode != "volume":
            raise ValueError(
                f"ConstantStateBC: unknown mode '{mode}'; supported: 'volume'."
            )
        if faces is None:
            self.faces = None
        else:
            faces = (faces,) if isinstance(faces, str) else tuple(faces)
            unknown = [face for face in faces if face not in self.FACES]
            if unknown:
                raise ValueError(
                    f"ConstantStateBC: unknown face(s) {unknown}; expected any of "
                    f"{list(self.FACES)}."
                )
            self.faces = faces
        self.mode = mode
        self.rtol = float(rtol)
        self.volumes = {}
        self.n_cells = {}

    def bind(self, model):
        reservoir = model.reservoir
        boundary_volumes = getattr(reservoir, "boundary_volumes", None)
        if not isinstance(boundary_volumes, dict):
            raise RuntimeError(
                f"ConstantStateBC(mode='volume') reads the boundary-volume map of "
                f"the reservoir, but {type(reservoir).__name__} does not expose "
                "'boundary_volumes'. The huge-boundary-volume far field is "
                "implemented by the structured reservoir family (StructReservoir, "
                "StructRadialReservoir, CPG_Reservoir); other reservoirs must "
                "express an open boundary differently."
            )

        requested = {
            face: float(value)
            for face, value in boundary_volumes.items()
            if value is not None
        }
        faces = (
            self.faces
            if self.faces is not None
            else tuple(face for face in self.FACES if face in requested)
        )
        if not faces:
            raise RuntimeError(
                "ConstantStateBC declares an open (constant-state) far field but "
                f"no face of {type(reservoir).__name__} carries a boundary volume. "
                "Set reservoir.boundary_volumes['<face>'] (CPG_Reservoir: call "
                "set_boundary_volume(...) + apply_volume_depth()) before "
                "model.init()."
            )
        missing = [face for face in faces if face not in requested]
        if missing:
            raise RuntimeError(
                f"ConstantStateBC declares face(s) {missing}, but no boundary "
                f"volume was set for them on {type(reservoir).__name__} "
                f"(faces with a volume: {sorted(requested)})."
            )

        volume = np.asarray(model.reservoir.mesh.volume, dtype=float)
        n_res_blocks = int(
            getattr(model.reservoir.mesh, "n_res_blocks", len(volume)) or len(volume)
        )
        volume = volume[:n_res_blocks]
        for face in faces:
            value = requested[face]
            n_cells = int(
                np.count_nonzero(np.isclose(volume, value, rtol=self.rtol, atol=0.0))
            )
            if n_cells == 0:
                raise RuntimeError(
                    f"ConstantStateBC: the far-field volume {value:g} declared for "
                    f"face '{face}' never reached the mesh — no cell of "
                    f"{type(reservoir).__name__} carries it (mesh volume range "
                    f"[{volume.min():g}, {volume.max():g}]). The engine caches the "
                    "pore volume PV = volume * poro ONCE, in engine.init() "
                    "(DartsModel.reset(), called from DartsModel.init()), so the "
                    "boundary volumes must be in mesh.volume by then: populate "
                    "reservoir.boundary_volumes BEFORE model.init() (the structured "
                    "reservoir applies them inside discretize(), which init() runs "
                    "first), and for CPG_Reservoir call set_boundary_volume(...) "
                    "followed by apply_volume_depth() before model.init(). A volume "
                    "written after that point is silently ignored and the boundary "
                    "stays closed."
                )
            self.volumes[face] = value
            self.n_cells[face] = n_cells

    def apply(self, ctx: AssemblyContext):
        """No-op: the far field is carried by the mesh volumes, not the residual."""
        return


class DirichletPin(ConditionItem):
    """A prescribed state value for selected ``(cell, equation)`` pairs.

    Two modes, and the difference between them is not cosmetic.

    ``mode="state"`` (the DEFAULT) is a PROJECTION: it overwrites the entry of
    the state vector, ``X[cell * n_vars + equation] = value``, and touches
    neither the residual nor the Jacobian. The pinned equation is still
    assembled from the (overwritten) state and its accumulation term is still
    computed — and then silently discarded, because the next projection
    overwrites whatever the Newton update did to that entry. The constraint is
    therefore invisible to the linear solver and to any preconditioner: they see
    an unconstrained system, and the pin is re-imposed behind their back. It has
    two virtues: it needs no Jacobian, so it works on the GPU engines and with
    direct solvers, and it reproduces bit-for-bit what the models that
    hand-rolled this pinning did before the conditions layer existed.

    ``mode="row"`` is the assembly-consistent formulation: the block-CSR row of
    the pinned equation is replaced by the constraint itself — every block in
    that block row has the equation's row zeroed, the diagonal block gets a unit
    entry, and the residual becomes ``X - value``. That is the physically
    honest statement (the linear solver and the preconditioner now see the
    constraint, the pinned equation no longer competes with its own
    accumulation term), and it converges to the same pinned value. It is
    CPU-only (it needs ``ctx.jac``, so ``provides_jacobian = True`` and
    ``requires_platform = "cpu"``), and because it changes the assembled system
    it changes results — it is strictly OPT-IN and no shipped model selects it.

    WHEN THE PROJECTION IS APPLIED (state mode). The pin is written at two
    points, and both are idempotent:

    - :meth:`project_state`, BEFORE ``engine.assemble_linear_system(dt)``. This
      is the load-bearing one: it is what makes the assembly — the residual, the
      Jacobian and the convergence test — see the pinned state. The framework's
      stock nonlinear solver has no pre-assembly stage, so a model must call
      ``self.conditions.project_state(t)`` at the top of its own Newton loop,
      exactly where it used to call its hand-rolled pinning function.
    - :meth:`apply`, AFTER assembly, from the ``conditions`` stage of
      :meth:`~darts.models.darts_model.DartsModel.apply_rhs_flux`. This is the
      only stage the framework drives by itself; it guarantees that the state
      the Newton update starts from — and the state accepted at the end of the
      timestep — satisfies the pin.

    A model that calls only the second one gets a WEAKER condition than the
    legacy pinning code: the assembly then evaluates at the un-pinned state that
    the previous Newton update left behind, which perturbs the residual, the
    Jacobian and the iteration counts. Measured on SPE11b (42x12, 2 simulated
    years, single-threaded): same timestep and Newton counts, one linear
    iteration fewer, and a 3e-7 relative shift in the final state — small, but
    not the bit-for-bit reproduction the reference pickles need. Migrate a
    hand-rolled pin by calling BOTH.

    GPU. On the GPU platform the engine state lives on the device and
    ``engine.X`` is only its host mirror; the framework's assembly context does
    not round-trip it (it round-trips the RHS, which is a different buffer).
    This item therefore performs exactly the round-trip the models performed
    around their hand-rolled pin, and for the same reason::

        copy_data_to_host(engine.X, engine.get_X_d())   # refresh the host mirror
        X[pinned] = value                               # write on the host
        copy_data_to_device(engine.X, engine.get_X_d()) # push the whole vector back

    The refresh is not optional: the write is a host write and the push copies
    the WHOLE vector back, so a stale host mirror would overwrite the device
    state everywhere else. The consequence is an ordering rule — on GPU this
    item's write wins over any host-side write to ``engine.X`` made earlier in
    the same ``apply_rhs_flux`` pass, because the refresh discards it. (Items
    that only read ``ctx.X`` are unaffected, and a pin registered FIRST leaves
    the host mirror fresh for them, which is how SPE11b's CO2 source reads a
    valid well-block pressure on GPU.)

    CONTRIBUTION TYPE (review item E6). ``mode="row"`` is the shipped
    REPLACEMENT case: it claims the pinned rows, so no other item may
    contribute additively to them and no second item may claim them.
    ``mode="state"`` is a projection of the STATE, not a contribution to the
    system — it declares ``contribution = "none"`` and takes part in no row
    conflict, because an additive source on a pinned equation is perfectly
    meaningful there (it changes the residual, which the projection then
    ignores).

    :param cells: the blocks to pin: an int array of block indices, or a
        :class:`Selector` resolved at bind time.
    :param equation: index of the pinned equation/variable within the block —
        an int applied to every cell, or one index per cell.
    :param values: the pinned value(s): a scalar, an array of one value per
        cell, or a callable ``f(t) -> array`` of one value per cell.
    :param mode: ``"state"`` (default) or ``"row"``, as described above.
    """

    def __init__(self, cells, equation, values, mode: str = "state"):
        if mode not in ("state", "row"):
            raise ValueError(
                f"DirichletPin: unknown mode '{mode}'; supported: 'state', 'row'."
            )
        self.mode = mode
        self.cells = (
            cells
            if isinstance(cells, Selector)
            else np.asarray(cells, dtype=np.int64).ravel()
        )
        self.equation = equation
        self.values = values
        # row mode replaces Jacobian rows; state mode writes nothing but X
        self.provides_jacobian = mode == "row"
        self.requires_platform = "cpu" if mode == "row" else None
        self.contribution = REPLACEMENT if mode == "row" else NO_CONTRIBUTION
        self.equations = None
        self._x_idx = None  # flat X/rhs indices of the pinned entries
        self._zero_idx = None  # row mode: flat jac_vals indices to zero
        self._diag_idx = None  # row mode: flat jac_vals indices of the unit entries
        self._engine = None
        self._on_gpu = False
        self._copy_to_host = None
        self._copy_to_device = None

    def written_rows(self, model):
        return () if self._x_idx is None else self._x_idx

    def bind(self, model):
        n_vars = model.physics.n_vars
        n_blocks = model.reservoir.mesh.n_blocks
        self.cells = resolve_blocks(self.cells, model)
        if len(self.cells) and (self.cells.min() < 0 or self.cells.max() >= n_blocks):
            raise IndexError(
                f"DirichletPin: cell indices must be within [0, {n_blocks})."
            )
        equations = np.asarray(self.equation, dtype=np.int64).ravel()
        if equations.size == 1:
            equations = np.full(len(self.cells), int(equations[0]), dtype=np.int64)
        elif equations.size != len(self.cells):
            raise ValueError(
                f"DirichletPin: equation has {equations.size} entries, expected 1 "
                f"or {len(self.cells)} (one per cell)."
            )
        if equations.size and (equations.min() < 0 or equations.max() >= n_vars):
            raise IndexError(
                f"DirichletPin: equation indices must be within [0, {n_vars})."
            )
        self.equations = equations
        self._x_idx = self.cells * n_vars + equations
        if not callable(self.values):
            self.values = np.broadcast_to(
                np.asarray(self.values, dtype=float), (len(self.cells),)
            ).astype(float)

        self._engine = model.physics.engine
        self._on_gpu = model.platform == "gpu"
        if self._on_gpu:
            # Same import the models used for their hand-rolled round-trip; it
            # only exists in a GPU-enabled build.
            from darts.engines import copy_data_to_device, copy_data_to_host

            self._copy_to_host = copy_data_to_host
            self._copy_to_device = copy_data_to_device

        if self.mode == "row":
            view = BlockCSRView(model.physics.engine, n_vars)
            block_size = view.block_size
            zero_idx, diag_idx = [], []
            for cell, equation in zip(self.cells, equations, strict=True):
                row_start = int(view.jac_rows[cell])
                row_end = int(view.jac_rows[cell + 1])
                for pos in range(row_start, row_end):
                    start = pos * block_size + int(equation) * n_vars
                    zero_idx.extend(range(start, start + n_vars))
                diag_idx.append(
                    view.diag_pos(int(cell)) * block_size
                    + int(equation) * n_vars
                    + int(equation)
                )
            self._zero_idx = np.asarray(zero_idx, dtype=np.int64)
            self._diag_idx = np.asarray(diag_idx, dtype=np.int64)
            self.stamp_pattern(model)

    def evaluate_values(self, t: float) -> np.ndarray:
        """The pinned value of every cell at time ``t``."""
        if callable(self.values):
            values = np.asarray(self.values(t), dtype=float).ravel()
            if values.size != len(self.cells):
                raise ValueError(
                    f"DirichletPin: values(t) returned {values.size} value(s), "
                    f"expected {len(self.cells)} (one per cell)."
                )
            return values
        return self.values

    def _write_state(self, t: float):
        """Overwrite the pinned entries of the engine state vector.

        On GPU this is the documented host round-trip; on CPU ``engine.X`` is
        the state vector itself (and the very buffer ``ctx.X`` views), so the
        write lands directly.
        """
        if not len(self.cells):
            return
        engine = self._engine
        if self._on_gpu:
            self._copy_to_host(engine.X, engine.get_X_d())
        np.asarray(engine.X)[self._x_idx] = self.evaluate_values(t)
        if self._on_gpu:
            self._copy_to_device(engine.X, engine.get_X_d())

    def project_state(self, t: float):
        """Pre-assembly projection (state mode only); a no-op in row mode."""
        if self.mode == "state":
            self._write_state(t)

    def apply(self, ctx: AssemblyContext):
        if not len(self.cells):
            return
        if self.mode == "state":
            self._write_state(ctx.t)
            return
        # row mode: replace the block-CSR row of the pinned equation
        ctx.jac.jac_vals[self._zero_idx] = 0.0
        ctx.jac.jac_vals[self._diag_idx] = 1.0
        ctx.rhs[self._x_idx] = ctx.X[self._x_idx] - self.evaluate_values(ctx.t)
