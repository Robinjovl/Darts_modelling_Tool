"""Unit tests for the unified conditions framework (``darts.models.conditions``).

Everything runs against pure-numpy stubs -- no reservoir, no assembly:

* :class:`BlockCSRView` over a hand-built 3-block CSR (2x2 dense blocks):
  ``block_pos``/``diag_pos``/``add_block`` semantics, the missing-block
  ``RuntimeError``, and the not-exposed-Jacobian ``RuntimeError``;
* :class:`CellSource`: constant / callable rates scattered into a fake
  ``ctx.rhs`` with the engine sign convention (``-= q * dt``), the analytic
  diagonal-block Jacobian through the real ``BlockCSRView``, and the shape /
  index validation;
* :class:`SegmentSource`: structural wellhead exclusion (cells start at
  ``well_head_idx + 1``) and the unknown-well ``KeyError``;
* :class:`InterfaceFlux`: the 4-block scatter signs of a linear exchange flux;
* :class:`ConditionSet.compile` guards, exercised on real ``DartsModel``
  subclasses instantiated via ``object.__new__`` (no ``__init__``; only the
  attributes ``compile`` touches are set): overridden ``apply_rhs_flux``,
  Jacobian items off-CPU, the adjoint/history-matching guard, and the
  per-item ``requires_platform`` check.
"""

import numpy as np
import pytest

# ConditionSet.compile imports darts.models.darts_model (compiled extension);
# skip the module on a checkout without it.
pytest.importorskip("darts.engines")

from darts.models.conditions import (  # noqa: E402
    AssemblyContext,
    BlockCSRView,
    CellSource,
    ConditionItem,
    ConditionSet,
    InterfaceFlux,
    SegmentSource,
)

N_VARS = 2
BLOCK_SIZE = N_VARS * N_VARS


class FakeCSREngine:
    """3-block CSR with 2x2 dense blocks and pattern::

    row 0: (0,0) (0,1)
    row 1: (1,0) (1,1) (1,2)
    row 2:       (2,1) (2,2)
    """

    def __init__(self):
        self.jac_rows = np.array([0, 2, 5, 7], dtype=np.int64)
        self.jac_cols = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int64)
        self.jac_diags = np.array([0, 3, 6], dtype=np.int64)
        self.jac_vals = np.zeros(7 * BLOCK_SIZE)
        self.opt_history_matching = False


class _StubMesh:
    def __init__(self, n_blocks, n_res_blocks=None):
        self.n_blocks = n_blocks
        self.n_res_blocks = n_res_blocks if n_res_blocks is not None else n_blocks


class _StubWell:
    def __init__(self, name, well_head_idx, num_segments):
        self.name = name
        self.well_head_idx = well_head_idx
        self.num_segments = num_segments


class _StubReservoir:
    def __init__(self, n_blocks, wells=()):
        self.mesh = _StubMesh(n_blocks)
        self.wells = list(wells)

    def get_well(self, well_name):
        for well in self.wells:
            if well.name == well_name:
                return well


class _StubPhysics:
    def __init__(self, engine):
        self.n_vars = N_VARS
        self.engine = engine


class _StubModel:
    def __init__(self, n_blocks=3, engine=None, wells=()):
        self.physics = _StubPhysics(engine if engine is not None else FakeCSREngine())
        self.reservoir = _StubReservoir(n_blocks, wells)
        self.platform = "cpu"


def _make_ctx(model, dt=0.5, t=1.0, jac=None, X=None):
    n = model.reservoir.mesh.n_blocks * N_VARS
    return AssemblyContext(
        rhs=np.zeros(n),
        jac=jac,
        X=X if X is not None else np.arange(n, dtype=float),
        Xn=np.zeros(n),
        dt=dt,
        t=t,
        iteration=0,
        n_vars=N_VARS,
        n_res_blocks=model.reservoir.mesh.n_res_blocks,
    )


# --------------------------------------------------------------- BlockCSRView
def test_block_csr_view_positions_and_add_block():
    engine = FakeCSREngine()
    view = BlockCSRView(engine, N_VARS)
    assert view.block_pos(0, 0) == 0
    assert view.block_pos(0, 1) == 1
    assert view.block_pos(1, 2) == 4
    assert view.block_pos(2, 1) == 5
    assert view.diag_pos(0) == 0
    assert view.diag_pos(1) == 3
    assert view.diag_pos(2) == 6

    dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    view.add_block(view.block_pos(1, 2), dense)
    view.add_block(view.block_pos(1, 2), dense)  # accumulation, not overwrite
    start = 4 * BLOCK_SIZE
    assert engine.jac_vals[start : start + BLOCK_SIZE] == pytest.approx(
        2.0 * dense.reshape(-1)
    )
    # no other position was touched
    untouched = np.ones(7, dtype=bool)
    untouched[4] = False
    assert np.all(engine.jac_vals.reshape(7, BLOCK_SIZE)[untouched] == 0.0)


def test_block_csr_view_missing_block_raises():
    view = BlockCSRView(FakeCSREngine(), N_VARS)
    with pytest.raises(RuntimeError, match=r"\(0, 2\)"):
        view.block_pos(0, 2)
    with pytest.raises(RuntimeError, match=r"\(2, 0\)"):
        view.block_pos(2, 0)


def test_block_csr_view_requires_exposed_jacobian():
    class _NoJacEngine:
        jac_diags = np.array([], dtype=np.int64)

    with pytest.raises(RuntimeError, match="does not expose"):
        BlockCSRView(_NoJacEngine(), N_VARS)
    with pytest.raises(RuntimeError, match="does not expose"):
        BlockCSRView(object(), N_VARS)  # no jac_diags attribute at all


# ----------------------------------------------------------------- CellSource
def test_cell_source_constant_rates_scatter_into_rhs():
    model = _StubModel()
    item = CellSource(cells=[1], rates=[[2.0, 3.0]])
    assert item.provides_jacobian is False
    item.bind(model)
    ctx = _make_ctx(model, dt=0.5)
    item.apply(ctx)
    # engine convention: R -= q * dt, only in the addressed block
    assert ctx.rhs == pytest.approx([0.0, 0.0, -1.0, -1.5, 0.0, 0.0])


def test_cell_source_callable_rates_receive_time_and_states():
    model = _StubModel()
    seen = {}

    def rates(t, states):
        seen["t"] = t
        seen["states"] = states.copy()
        return np.full((1, N_VARS), 4.0)

    item = CellSource(cells=[2], rates=rates)
    item.bind(model)
    ctx = _make_ctx(model, dt=0.25, t=7.0)
    item.apply(ctx)
    assert seen["t"] == 7.0
    assert seen["states"] == pytest.approx(np.array([[4.0, 5.0]]))  # X of block 2
    assert ctx.rhs[4:6] == pytest.approx([-1.0, -1.0])


def test_cell_source_d_rates_writes_diagonal_block():
    engine = FakeCSREngine()
    model = _StubModel(engine=engine)
    d_rates = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    item = CellSource(cells=[1], rates=[[1.0, 1.0]], d_rates=d_rates)
    assert item.provides_jacobian is True
    item.bind(model)
    view = BlockCSRView(engine, N_VARS)
    ctx = _make_ctx(model, dt=0.5, jac=view)
    item.apply(ctx)
    # jac_diag(1) -= dt * d_rates
    start = view.diag_pos(1) * BLOCK_SIZE
    assert engine.jac_vals[start : start + BLOCK_SIZE] == pytest.approx(
        -0.5 * d_rates[0].reshape(-1)
    )
    other = np.ones(7, dtype=bool)
    other[view.diag_pos(1)] = False
    assert np.all(engine.jac_vals.reshape(7, BLOCK_SIZE)[other] == 0.0)


def test_cell_source_shape_and_index_validation():
    model = _StubModel()
    with pytest.raises(ValueError, match="rates"):
        CellSource(cells=[0], rates=[[1.0, 2.0, 3.0]]).bind(model)  # bad n_vars
    with pytest.raises(IndexError, match="cell indices"):
        CellSource(cells=[3], rates=[[1.0, 2.0]]).bind(model)  # out of range
    with pytest.raises(ValueError, match="d_rates"):
        CellSource(cells=[0], rates=[[1.0, 2.0]], d_rates=np.zeros((1, 3, 3))).bind(
            model
        )


# -------------------------------------------------------------- SegmentSource
def test_segment_source_excludes_wellhead_block():
    well = _StubWell("W1", well_head_idx=1, num_segments=2)
    model = _StubModel(wells=[well])
    item = SegmentSource("W1", rates=[[2.0, 2.0]])  # one BODY segment
    item.bind(model)
    assert item.cells.tolist() == [2]  # starts at well_head_idx + 1
    ctx = _make_ctx(model, dt=1.0)
    item.apply(ctx)
    # wellhead block rows (well-control equations) untouched
    assert np.all(ctx.rhs[2:4] == 0.0)
    assert ctx.rhs[4:6] == pytest.approx([-2.0, -2.0])


def test_segment_source_unknown_well_raises_key_error():
    model = _StubModel(wells=[_StubWell("W1", 1, 2)])
    item = SegmentSource("NOPE", rates=[[1.0, 1.0]])
    with pytest.raises(KeyError, match="NOPE"):
        item.bind(model)


# -------------------------------------------------------------- InterfaceFlux
class _LinearExchange(InterfaceFlux):
    """flux into row block = k * (state_col - state_row), per variable."""

    k = 2.0

    def connection_flux(self, t, state_row, state_col):
        eye = np.eye(N_VARS)
        return self.k * (state_col - state_row), -self.k * eye, self.k * eye


def test_interface_flux_scatters_rhs_and_four_jacobian_blocks():
    engine = FakeCSREngine()
    model = _StubModel(engine=engine)
    item = _LinearExchange(connections=[(0, 1)])
    item.bind(model)
    view = BlockCSRView(engine, N_VARS)
    X = np.array([1.0, 2.0, 5.0, 8.0, 0.0, 0.0])
    ctx = _make_ctx(model, dt=0.5, jac=view, X=X)
    item.apply(ctx)

    flux_dt = 2.0 * (X[2:4] - X[0:2]) * 0.5  # k * (col - row) * dt
    assert ctx.rhs[0:2] == pytest.approx(-flux_dt)
    assert ctx.rhs[2:4] == pytest.approx(flux_dt)
    assert np.all(ctx.rhs[4:6] == 0.0)

    blocks = engine.jac_vals.reshape(7, N_VARS, N_VARS)
    d_row_dt = -2.0 * np.eye(N_VARS) * 0.5
    d_col_dt = 2.0 * np.eye(N_VARS) * 0.5
    assert blocks[view.diag_pos(0)] == pytest.approx(-d_row_dt)
    assert blocks[view.block_pos(0, 1)] == pytest.approx(-d_col_dt)
    assert blocks[view.block_pos(1, 0)] == pytest.approx(d_row_dt)
    assert blocks[view.diag_pos(1)] == pytest.approx(d_col_dt)


def test_interface_flux_requires_neighbouring_blocks():
    model = _StubModel()
    item = _LinearExchange(connections=[(0, 2)])  # (0, 2) not in the pattern
    with pytest.raises(RuntimeError, match=r"\(0, 2\)"):
        item.bind(model)


# ---------------------------------------------------------- ConditionSet API
def test_condition_set_add_rejects_non_items():
    conditions = ConditionSet()
    with pytest.raises(TypeError, match="ConditionItem"):
        conditions.add(object())
    item = CellSource(cells=[], rates=np.zeros((0, N_VARS)))
    assert conditions.add(item) is item
    assert len(conditions) == 1 and bool(conditions)


def test_condition_item_apply_is_abstract():
    with pytest.raises(NotImplementedError, match="apply"):
        ConditionItem().apply(None)


# ------------------------------------------------------ ConditionSet.compile
# Real DartsModel subclasses instantiated WITHOUT __init__ (object.__new__),
# with only the attributes compile() touches set by hand.
def _bare_darts_model(cls, engine=None, platform="cpu", n_blocks=3):
    model = object.__new__(cls)
    model.nonlinear_solver = None
    model.platform = platform
    model.physics = _StubPhysics(engine if engine is not None else FakeCSREngine())
    model.reservoir = _StubReservoir(n_blocks)
    return model


def test_compile_empty_set_is_noop():
    conditions = ConditionSet()
    # an empty set never touches the model: any object works
    assert conditions.compile(object()) is conditions


def test_compile_rejects_overridden_apply_rhs_flux():
    from darts.models.darts_model import DartsModel

    class _OverridingModel(DartsModel):
        def apply_rhs_flux(self, dt, t):
            pass

    model = _bare_darts_model(_OverridingModel)
    conditions = ConditionSet()
    conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    with pytest.raises(RuntimeError, match="apply_rhs_flux"):
        conditions.compile(model)


def test_compile_rejects_jacobian_items_off_cpu():
    from darts.models.darts_model import DartsModel

    class _PlainModel(DartsModel):
        pass

    model = _bare_darts_model(_PlainModel, platform="gpu")
    conditions = ConditionSet()
    conditions.add(
        CellSource(cells=[0], rates=[[1.0, 1.0]], d_rates=np.zeros((1, 2, 2)))
    )
    with pytest.raises(RuntimeError, match="CPU-only"):
        conditions.compile(model)


def test_compile_rejects_opaque_items_under_adjoint():
    from darts.models.darts_model import DartsModel

    class _PlainModel(DartsModel):
        pass

    engine = FakeCSREngine()
    engine.opt_history_matching = True
    model = _bare_darts_model(_PlainModel, engine=engine)
    conditions = ConditionSet()
    conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    with pytest.raises(RuntimeError, match="adjoint_transparent"):
        conditions.compile(model)


def test_compile_enforces_requires_platform():
    from darts.models.darts_model import DartsModel

    class _PlainModel(DartsModel):
        pass

    item = CellSource(cells=[0], rates=[[1.0, 1.0]])
    item.requires_platform = "gpu"
    model = _bare_darts_model(_PlainModel, platform="cpu")
    conditions = ConditionSet()
    conditions.add(item)
    with pytest.raises(RuntimeError, match="requires"):
        conditions.compile(model)


def test_compile_success_binds_items():
    from darts.models.darts_model import DartsModel

    class _PlainModel(DartsModel):
        pass

    model = _bare_darts_model(_PlainModel)
    conditions = ConditionSet()
    item = conditions.add(
        CellSource(cells=[1], rates=[[1.0, 1.0]], d_rates=np.zeros((1, 2, 2)))
    )
    assert conditions.compile(model) is conditions
    # bind resolved the RHS scatter indices and the CSR diagonal position
    assert item._rhs_idx.tolist() == [2, 3]
    assert item._diag_pos.tolist() == [3]
