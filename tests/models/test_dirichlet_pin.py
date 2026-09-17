"""Unit tests for :class:`darts.models.conditions.DirichletPin`.

Pure-numpy stubs -- no reservoir, no assembly -- covering the two modes and the
properties the two migrated models (SPE11b, chemistry/reaktoro_kinetics) rely on:

* ``mode="state"``: the projection writes ONLY the pinned entries of the state
  vector, through ``engine.X`` (the very buffer ``ctx.X`` views), leaving the
  residual and the Jacobian untouched; it is idempotent, so a model may drive it
  both from :meth:`~darts.models.conditions.ConditionItem.project_state` (before
  assembly, where it is load-bearing) and from ``apply`` (after assembly);
* the GPU host round-trip -- refresh the host mirror from the device, write,
  push the WHOLE vector back -- including the consequence that a stale host
  value elsewhere in the vector is discarded rather than pushed to the device;
* ``mode="row"``: the block-CSR row replacement (the pinned equation's row
  zeroed across the whole block row, unit diagonal, residual ``X - value``),
  its ``provides_jacobian``/``requires_platform`` flags and the
  :meth:`~darts.models.conditions.ConditionSet.compile` guard they trigger;
* argument validation (mode, cell/equation ranges, per-cell shapes) and the
  empty-selection no-op;
* :meth:`~darts.models.conditions.ConditionSet.project_state` fan-out, and its
  no-op default on items that do not constrain the state.
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
    DirichletPin,
)

N_VARS = 2
BLOCK_SIZE = N_VARS * N_VARS
N_BLOCKS = 3


class _FakeCSREngine:
    """3-block CSR with 2x2 dense blocks and pattern::

    row 0: (0,0) (0,1)
    row 1: (1,0) (1,1) (1,2)
    row 2:       (2,1) (2,2)

    ``X`` is a plain numpy array, exactly like the host state buffer the real
    engine exposes: ``np.asarray(engine.X)`` is a view of it.
    """

    def __init__(self):
        self.jac_rows = np.array([0, 2, 5, 7], dtype=np.int64)
        self.jac_cols = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int64)
        self.jac_diags = np.array([0, 3, 6], dtype=np.int64)
        self.jac_vals = np.zeros(7 * BLOCK_SIZE)
        self.X = np.arange(N_BLOCKS * N_VARS, dtype=float)
        self.opt_history_matching = False


class _StubMesh:
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks
        self.n_res_blocks = n_blocks


class _StubReservoir:
    def __init__(self, n_blocks):
        self.mesh = _StubMesh(n_blocks)
        self.wells = []


class _StubPhysics:
    def __init__(self, engine):
        self.n_vars = N_VARS
        self.engine = engine


class _StubModel:
    def __init__(self, engine=None, platform="cpu", n_blocks=N_BLOCKS):
        self.physics = _StubPhysics(engine if engine is not None else _FakeCSREngine())
        self.reservoir = _StubReservoir(n_blocks)
        self.platform = platform


def _make_ctx(model, dt=0.5, t=1.0, jac=None):
    """Context viewing the engine buffers, as ``DartsModel`` builds it."""
    engine = model.physics.engine
    return AssemblyContext(
        rhs=np.zeros(model.reservoir.mesh.n_blocks * N_VARS),
        jac=jac,
        X=np.asarray(engine.X),
        Xn=np.zeros(model.reservoir.mesh.n_blocks * N_VARS),
        dt=dt,
        t=t,
        iteration=0,
        n_vars=N_VARS,
        n_res_blocks=model.reservoir.mesh.n_res_blocks,
    )


# ------------------------------------------------------------- mode="state"
def test_state_mode_is_the_default_and_needs_no_jacobian():
    pin = DirichletPin(cells=[1], equation=0, values=7.0)
    assert pin.mode == "state"
    assert pin.provides_jacobian is False
    assert pin.requires_platform is None
    # pinning is never differentiated by the adjoint machinery
    assert pin.adjoint_transparent is False


def test_state_mode_apply_writes_only_the_pinned_entries():
    model = _StubModel()
    pin = DirichletPin(cells=[0, 2], equation=1, values=[10.0, 20.0])
    pin.bind(model)
    ctx = _make_ctx(model)
    before = ctx.X.copy()

    pin.apply(ctx)

    expected = before.copy()
    expected[0 * N_VARS + 1] = 10.0
    expected[2 * N_VARS + 1] = 20.0
    assert ctx.X.tolist() == expected.tolist()
    # the write lands in the engine buffer that ctx.X views
    assert np.asarray(model.physics.engine.X).tolist() == expected.tolist()
    # ... and nothing else is touched
    assert np.all(ctx.rhs == 0.0)
    assert np.all(model.physics.engine.jac_vals == 0.0)


def test_project_state_writes_the_same_values_without_a_context():
    model = _StubModel()
    pin = DirichletPin(cells=[1], equation=0, values=42.0)
    pin.bind(model)

    pin.project_state(t=0.0)

    assert model.physics.engine.X[1 * N_VARS + 0] == 42.0


def test_state_mode_is_idempotent_across_both_stages():
    """A model may project before assembly AND let apply() re-assert: same X."""
    model = _StubModel()
    pin = DirichletPin(cells=[0, 1], equation=1, values=[3.0, 4.0])
    pin.bind(model)

    pin.project_state(t=0.0)
    once = np.asarray(model.physics.engine.X).copy()
    pin.apply(_make_ctx(model))
    pin.project_state(t=0.0)
    twice = np.asarray(model.physics.engine.X).copy()

    assert once.tolist() == twice.tolist()


def test_state_mode_accepts_a_scalar_a_per_cell_array_and_a_callable():
    model = _StubModel()
    scalar = DirichletPin(cells=[0, 1], equation=0, values=5.0)
    scalar.bind(model)
    assert scalar.evaluate_values(0.0).tolist() == [5.0, 5.0]

    per_cell = DirichletPin(cells=[0, 1], equation=0, values=[1.0, 2.0])
    per_cell.bind(model)
    assert per_cell.evaluate_values(0.0).tolist() == [1.0, 2.0]

    schedule = DirichletPin(
        cells=[0, 1], equation=0, values=lambda t: np.array([t, 2 * t])
    )
    schedule.bind(model)
    assert schedule.evaluate_values(3.0).tolist() == [3.0, 6.0]
    schedule.project_state(t=3.0)
    assert model.physics.engine.X[0] == 3.0
    assert model.physics.engine.X[N_VARS] == 6.0


def test_per_cell_equation_indices():
    model = _StubModel()
    pin = DirichletPin(cells=[0, 1], equation=[0, 1], values=[8.0, 9.0])
    pin.bind(model)
    pin.project_state(t=0.0)
    assert model.physics.engine.X[0 * N_VARS + 0] == 8.0
    assert model.physics.engine.X[1 * N_VARS + 1] == 9.0


def test_empty_selection_is_a_noop():
    model = _StubModel()
    pin = DirichletPin(cells=[], equation=0, values=1.0)
    pin.bind(model)
    before = np.asarray(model.physics.engine.X).copy()
    pin.project_state(t=0.0)
    pin.apply(_make_ctx(model))
    assert np.asarray(model.physics.engine.X).tolist() == before.tolist()


# ------------------------------------------------------------ GPU round-trip
class _FakeDeviceEngine(_FakeCSREngine):
    """Engine whose authoritative state lives 'on the device'."""

    def __init__(self):
        super().__init__()
        self.X = np.zeros(N_BLOCKS * N_VARS)  # stale host mirror
        self.device_X = np.arange(N_BLOCKS * N_VARS, dtype=float) + 100.0

    def get_X_d(self):
        return self.device_X


def test_gpu_mode_refreshes_the_host_mirror_then_pushes_the_whole_vector(monkeypatch):
    import darts.engines

    calls = []

    def fake_copy_data_to_host(host, device):
        calls.append("to_host")
        np.asarray(host)[:] = device

    def fake_copy_data_to_device(host, device):
        calls.append("to_device")
        device[:] = np.asarray(host)

    monkeypatch.setattr(
        darts.engines, "copy_data_to_host", fake_copy_data_to_host, raising=False
    )
    monkeypatch.setattr(
        darts.engines, "copy_data_to_device", fake_copy_data_to_device, raising=False
    )

    engine = _FakeDeviceEngine()
    model = _StubModel(engine=engine, platform="gpu")
    pin = DirichletPin(cells=[1], equation=0, values=-1.0)
    pin.bind(model)

    # a stale host value that must NOT survive the round-trip
    engine.X[0] = 12345.0
    pin.project_state(t=0.0)

    assert calls == ["to_host", "to_device"]
    expected = np.arange(N_BLOCKS * N_VARS, dtype=float) + 100.0
    expected[1 * N_VARS + 0] = -1.0
    assert engine.device_X.tolist() == expected.tolist()
    assert engine.X.tolist() == expected.tolist()  # host mirror agrees again


def test_cpu_mode_does_not_touch_the_device_helpers(monkeypatch):
    import darts.engines

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the CPU path must not copy to/from a device")

    monkeypatch.setattr(darts.engines, "copy_data_to_host", explode, raising=False)
    monkeypatch.setattr(darts.engines, "copy_data_to_device", explode, raising=False)

    model = _StubModel()  # platform="cpu"
    pin = DirichletPin(cells=[1], equation=0, values=1.0)
    pin.bind(model)
    pin.project_state(t=0.0)
    assert model.physics.engine.X[N_VARS] == 1.0


# --------------------------------------------------------------- mode="row"
def test_row_mode_declares_its_requirements():
    pin = DirichletPin(cells=[1], equation=0, values=1.0, mode="row")
    assert pin.provides_jacobian is True
    assert pin.requires_platform == "cpu"


def test_row_mode_replaces_the_block_row_and_the_residual():
    engine = _FakeCSREngine()
    model = _StubModel(engine=engine)
    engine.jac_vals[:] = np.arange(engine.jac_vals.size, dtype=float) + 1.0
    reference = engine.jac_vals.copy()

    pin = DirichletPin(cells=[1], equation=0, values=2.5, mode="row")
    pin.bind(model)
    view = BlockCSRView(engine, N_VARS)
    ctx = _make_ctx(model, jac=view)
    ctx.rhs[:] = 7.0
    pin.apply(ctx)

    blocks = engine.jac_vals.reshape(-1, N_VARS, N_VARS)
    row_start, row_end = int(engine.jac_rows[1]), int(engine.jac_rows[1 + 1])
    diag = view.diag_pos(1)
    for pos in range(row_start, row_end):
        # the pinned equation's row of every block in the block row is zeroed,
        # except the unit entry on the diagonal block
        expected_row = np.zeros(N_VARS)
        if pos == diag:
            expected_row[0] = 1.0
        assert blocks[pos, 0].tolist() == expected_row.tolist()
        # the other equation of the same blocks is untouched
        assert (
            blocks[pos, 1].tolist()
            == reference.reshape(-1, N_VARS, N_VARS)[pos, 1].tolist()
        )
    # rows of the OTHER blocks are untouched entirely
    for pos in list(range(0, row_start)) + list(range(row_end, 7)):
        assert (
            blocks[pos].ravel().tolist()
            == reference.reshape(-1)[pos * BLOCK_SIZE : (pos + 1) * BLOCK_SIZE].tolist()
        )

    # residual of the pinned equation becomes X - value; the rest is untouched
    assert ctx.rhs[1 * N_VARS + 0] == ctx.X[1 * N_VARS + 0] - 2.5
    assert ctx.rhs[1 * N_VARS + 1] == 7.0
    # ... and the state itself is NOT overwritten in row mode
    assert ctx.X.tolist() == np.arange(N_BLOCKS * N_VARS, dtype=float).tolist()


def test_row_mode_project_state_is_a_noop():
    engine = _FakeCSREngine()
    model = _StubModel(engine=engine)
    pin = DirichletPin(cells=[1], equation=0, values=2.5, mode="row")
    pin.bind(model)
    before = np.asarray(engine.X).copy()
    pin.project_state(t=0.0)
    assert np.asarray(engine.X).tolist() == before.tolist()


def test_row_mode_newton_step_lands_exactly_on_the_pinned_value():
    """R = X - value with a unit diagonal makes X - dX == value in one step."""
    engine = _FakeCSREngine()
    model = _StubModel(engine=engine)
    pin = DirichletPin(cells=[1], equation=0, values=2.5, mode="row")
    pin.bind(model)
    ctx = _make_ctx(model, jac=BlockCSRView(engine, N_VARS))
    pin.apply(ctx)

    # DARTS solves J dX = R and updates X -= dX; the pinned row is 1 * dX = R
    d_x = ctx.rhs[1 * N_VARS + 0]
    assert ctx.X[1 * N_VARS + 0] - d_x == 2.5


# ------------------------------------------------------------- ConditionSet
def test_condition_set_project_state_fans_out_and_defaults_to_noop():
    model = _StubModel()
    conditions = ConditionSet()
    pin = conditions.add(DirichletPin(cells=[2], equation=1, values=99.0))
    # an item that does not constrain the state inherits the no-op default
    source = conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    pin.bind(model)
    source.bind(model)

    conditions.project_state(t=0.0)

    assert model.physics.engine.X[2 * N_VARS + 1] == 99.0
    assert ConditionItem().project_state(0.0) is None


def test_compile_rejects_row_mode_off_cpu():
    from darts.models.darts_model import DartsModel

    class _PlainModel(DartsModel):
        pass

    model = object.__new__(_PlainModel)
    model.nonlinear_solver = None
    model.platform = "gpu"
    model.physics = _StubPhysics(_FakeCSREngine())
    model.reservoir = _StubReservoir(N_BLOCKS)

    conditions = ConditionSet()
    conditions.add(DirichletPin(cells=[1], equation=0, values=1.0, mode="row"))
    with pytest.raises(RuntimeError, match="CPU-only"):
        conditions.compile(model)


def test_compile_accepts_state_mode_off_cpu(monkeypatch):
    """The projection needs no Jacobian, so it is allowed on the GPU engines."""
    import darts.engines
    from darts.models.darts_model import DartsModel

    monkeypatch.setattr(
        darts.engines, "copy_data_to_host", lambda host, dev: None, raising=False
    )
    monkeypatch.setattr(
        darts.engines, "copy_data_to_device", lambda host, dev: None, raising=False
    )

    class _PlainModel(DartsModel):
        pass

    model = object.__new__(_PlainModel)
    model.nonlinear_solver = None
    model.platform = "gpu"
    model.physics = _StubPhysics(_FakeDeviceEngine())
    model.reservoir = _StubReservoir(N_BLOCKS)

    conditions = ConditionSet()
    pin = conditions.add(DirichletPin(cells=[1], equation=0, values=1.0))
    assert conditions.compile(model) is conditions
    assert pin._x_idx.tolist() == [N_VARS]


# --------------------------------------------------------------- validation
def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        DirichletPin(cells=[0], equation=0, values=1.0, mode="penalty")


def test_out_of_range_cells_raise():
    model = _StubModel()
    pin = DirichletPin(cells=[N_BLOCKS], equation=0, values=1.0)
    with pytest.raises(IndexError, match="cell indices"):
        pin.bind(model)


def test_out_of_range_equation_raises():
    model = _StubModel()
    pin = DirichletPin(cells=[0], equation=N_VARS, values=1.0)
    with pytest.raises(IndexError, match="equation indices"):
        pin.bind(model)


def test_equation_length_mismatch_raises():
    model = _StubModel()
    pin = DirichletPin(cells=[0, 1], equation=[0, 1, 0], values=1.0)
    with pytest.raises(ValueError, match="one per cell"):
        pin.bind(model)


def test_callable_values_of_the_wrong_length_raise():
    model = _StubModel()
    pin = DirichletPin(cells=[0, 1], equation=0, values=lambda t: np.zeros(3))
    pin.bind(model)
    with pytest.raises(ValueError, match="one per cell"):
        pin.evaluate_values(0.0)
