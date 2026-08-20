"""Unified Python-side boundary/source conditions applied after engine assembly.

This module is the minimum-viable unified conditions layer: a small framework
through which models add Python-side source terms and interface fluxes to the
assembled residual/Jacobian, replacing the ad-hoc ``set_rhs_flux`` override and
the ``rhs_flux_hooks`` list (both deprecated, still supported).

Structure:

- :class:`BlockCSRView` — a numpy view of the engine's block-CSR Jacobian with
  position lookup and in-place block accumulation.
- :class:`AssemblyContext` — the per-iteration snapshot handed to every item.
- :class:`ConditionItem` — the item base class; its docstring is THE CONTRACT.
- :class:`ConditionSet` — the validated collection owned by ``DartsModel``
  (``model.conditions``), compiled once at the end of ``init()``.
- Item types: :class:`CellSource`, :class:`SegmentSource`, :class:`PipeSourceTerm`,
  :class:`InterfaceFlux`.

A ``DirichletPin`` item (pinning a state variable via a penalized diagonal) is
deliberately NOT part of this MVP — pinning interacts with the engine's
accumulation terms and chopping and needs its own design pass.
"""

import inspect
from dataclasses import dataclass

import numpy as np


class BlockCSRView:
    """Numpy view of the engine's block-CSR Jacobian (rows/cols/diags/values).

    Wraps ``engine.jac_rows`` / ``jac_cols`` / ``jac_diags`` / ``jac_vals`` as
    numpy views (no copies — writes go straight into the engine matrix). Each
    CSR position addresses one dense ``n_vars x n_vars`` block stored
    equation-major: entry ``(c, v)`` of block ``pos`` lives at flat value index
    ``pos * n_vars**2 + c * n_vars + v``.
    """

    def __init__(self, engine, n_vars: int):
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

    :ivar provides_jacobian: item adds analytic Jacobian contributions.
    :ivar requires_platform: restrict to one platform (``"cpu"``/``"gpu"``),
        or ``None`` for any.
    :ivar adjoint_transparent: contributions are correctly differentiated by
        the adjoint machinery (they are not, unless an item proves otherwise);
        the opt/history-matching driver rejects opaque items.
    """

    provides_jacobian: bool = False
    requires_platform: str | None = None
    adjoint_transparent: bool = False

    def bind(self, model):
        """Resolve model-dependent data (indices, CSR positions). Called once
        by :meth:`ConditionSet.compile` after the engine exists."""
        pass

    def apply(self, ctx: AssemblyContext):
        """Add this item's contributions to ``ctx.rhs`` (and ``ctx.jac``)."""
        raise NotImplementedError(f"{type(self).__name__}.apply() is not implemented")

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
    """

    def __init__(self):
        self.items = []

    def __bool__(self):
        return bool(self.items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def add(self, item: ConditionItem) -> ConditionItem:
        """Register an item; returns it for chaining/keeping a reference."""
        if not isinstance(item, ConditionItem):
            raise TypeError(
                f"ConditionSet.add expects a ConditionItem, got {type(item).__name__}"
            )
        self.items.append(item)
        return item

    def compile(self, model) -> "ConditionSet":
        """Validate the set against the model/engine and bind every item.

        Raises with a specific message when (i) the model overrides
        ``apply_rhs_flux`` (items would silently never run), (ii) a
        Jacobian-providing item runs off-CPU or on an engine without an
        exposed block-CSR matrix, (iii) the model runs a mechanics engine
        (rows are rescaled inside assembly, post-assembly writes would be
        mis-scaled), or (iv) the adjoint/history-matching driver is active and
        an item is not ``adjoint_transparent``.
        """
        if not self.items:
            return self

        from darts.models.darts_model import DartsModel
        from darts.nonlinear_solvers.mechanics import MechanicsNewtonSolver

        # (i) an overridden apply_rhs_flux bypasses the conditions stage
        if type(model).apply_rhs_flux is not DartsModel.apply_rhs_flux:
            raise RuntimeError(
                f"{type(model).__name__} overrides apply_rhs_flux(), so the "
                f"{len(self.items)} registered condition item(s) would silently "
                "never run. Either call super().apply_rhs_flux(dt, t) from the "
                "override or migrate the override into condition items."
            )

        # (iii) mechanics engines rescale equation rows inside assembly
        if isinstance(getattr(model, "nonlinear_solver", None), MechanicsNewtonSolver):
            raise RuntimeError(
                "Condition items are not supported on mechanics models yet: "
                "the pm/mech engines rescale equation rows inside assembly, so "
                "post-assembly RHS/Jacobian writes would be mis-scaled."
            )

        engine = model.physics.engine

        # (iv) adjoint guard: the opt driver replays assembly for gradients and
        # knows nothing about Python-side contributions
        if getattr(engine, "opt_history_matching", False):
            opaque = [
                type(item).__name__
                for item in self.items
                if not item.adjoint_transparent
            ]
            if opaque:
                raise RuntimeError(
                    "The adjoint/history-matching driver is active "
                    "(engine.opt_history_matching) but these condition items "
                    "are not adjoint_transparent: " + ", ".join(opaque) + ". "
                    "Their contributions would be missing from the adjoint "
                    "gradient."
                )

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
        return self

    def apply(self, ctx: AssemblyContext):
        for item in self.items:
            item.apply(ctx)

    def on_timestep_start(self, dt: float, t: float):
        for item in self.items:
            item.on_timestep_start(dt, t)

    def on_timestep_converged(self, dt: float, t: float):
        for item in self.items:
            item.on_timestep_converged(dt, t)

    def on_timestep_failed(self, dt: float, t: float):
        for item in self.items:
            item.on_timestep_failed(dt, t)


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

    :param cells: int array of block indices the rates apply to.
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
        self.cells = np.asarray(cells, dtype=np.int64).ravel()
        self.rates = rates
        self.d_rates = d_rates
        self.provides_jacobian = d_rates is not None
        self._rates_take_states = callable(rates) and _callable_takes_states(rates)
        self._rhs_idx = None
        self._diag_pos = None
        self._n_vars = None

    def bind(self, model):
        n_vars = model.physics.n_vars
        n_blocks = model.reservoir.mesh.n_blocks
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
            if not callable(self.d_rates):
                self.d_rates = np.asarray(self.d_rates, dtype=float)
                self._check_shape(
                    self.d_rates, (len(self.cells), n_vars, n_vars), "d_rates"
                )

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
            for k, pos in enumerate(self._diag_pos):
                ctx.jac.add_block(int(pos), -ctx.dt * d_rates[k])


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

    This is the unified replacement for the hand-rolled ``set_rhs_flux``
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
      which is what the models wrote as ``rhs_flux[block] = -rates`` followed
      by ``rhs += rhs_flux * dt``.

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
    pairs; both blocks of every pair must be neighbours in the Jacobian
    sparsity pattern (their off-diagonal blocks must exist).

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
        self.connections = [(int(r), int(c)) for r, c in connections]
        self._positions = None
        self._n_vars = None

    def bind(self, model):
        n_vars = model.physics.n_vars
        self._n_vars = n_vars
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
