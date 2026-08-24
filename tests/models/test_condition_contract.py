"""M4 contract hardening: observers, the restart contract and selectors.

Review items E8 (a typed :class:`NonlinearIterationObserver` replacing the
untyped ``after_assembly`` seam), E9 (an item declares whether it carries state
that must survive a restart, and is either restored or refuses to restart) and
E7 (a :class:`Selector` accepted anywhere block indices are accepted).

Everything here runs against numpy stubs -- no engine, no reservoir -- except
where a real ``DartsModel`` subclass is needed for the ``compile``/restart
plumbing, and those are built with ``object.__new__`` so nothing is simulated.
"""

import json

import numpy as np
import pytest

pytest.importorskip("darts.engines")

from darts.models.conditions import (  # noqa: E402
    NO_CONTRIBUTION,
    AssemblyContext,
    BlockCSRView,
    BlockIndices,
    CellSource,
    ConditionItem,
    ConditionSet,
    DirichletPin,
    InterfaceFlux,
    NamedRegion,
    NonlinearIterationObserver,
    Selector,
    Where,
    resolve_blocks,
)

N_VARS = 2
BLOCK_SIZE = N_VARS * N_VARS


class _FakeCSREngine:
    """3-block CSR with 2x2 dense blocks (the pattern of test_conditions.py)."""

    def __init__(self):
        self.jac_rows = np.array([0, 2, 5, 7], dtype=np.int64)
        self.jac_cols = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int64)
        self.jac_diags = np.array([0, 3, 6], dtype=np.int64)
        self.jac_vals = np.zeros(7 * BLOCK_SIZE)
        self.X = np.zeros(3 * N_VARS)
        self.opt_history_matching = False


class _StubMesh:
    def __init__(self, n_blocks=3, op_num=None):
        self.n_blocks = n_blocks
        self.n_res_blocks = n_blocks
        self.op_num = (
            np.zeros(n_blocks, dtype=np.int64) if op_num is None else np.asarray(op_num)
        )


class _StubReservoir:
    def __init__(self, n_blocks=3, **attributes):
        self.mesh = _StubMesh(n_blocks)
        self.wells = []
        for key, value in attributes.items():
            setattr(self, key, value)

    def get_well(self, name):
        return None


class _StubPhysics:
    def __init__(self, engine, regions=(0,)):
        self.n_vars = N_VARS
        self.engine = engine
        self.regions = list(regions)


class _StubModel:
    def __init__(self, reservoir=None, engine=None, regions=(0,)):
        self.reservoir = reservoir if reservoir is not None else _StubReservoir()
        self.physics = _StubPhysics(
            engine if engine is not None else _FakeCSREngine(), regions
        )
        self.platform = "cpu"
        self._pattern_version = 0


def _ctx(model=None, jac=None, n_blocks=3):
    n = n_blocks * N_VARS
    return AssemblyContext(
        rhs=np.zeros(n),
        jac=jac,
        X=np.arange(n, dtype=float),
        Xn=np.zeros(n),
        dt=0.5,
        t=1.0,
        iteration=0,
        n_vars=N_VARS,
        n_res_blocks=n_blocks,
    )


# ------------------------------------------------------------- E8: observers
class _Recorder(NonlinearIterationObserver):
    def __init__(self):
        self.seen = []
        self.events = []

    def observe(self, ctx):
        self.seen.append((ctx.t, ctx.dt, ctx.rhs.copy()))

    def on_timestep_start(self, dt, t):
        self.events.append(("start", dt, t))

    def on_timestep_converged(self, dt, t):
        self.events.append(("converged", dt, t))

    def on_timestep_failed(self, dt, t):
        self.events.append(("failed", dt, t))


def test_add_dispatches_observers_and_items():
    conditions = ConditionSet()
    observer = conditions.add(_Recorder())
    item = conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    assert conditions.observers == [observer]
    assert conditions.items == [item]
    assert len(conditions) == 1  # len() still counts ITEMS
    assert bool(conditions)


def test_a_set_holding_only_observers_is_truthy():
    """so DartsModel.apply_rhs_flux still enters the conditions stage"""
    conditions = ConditionSet()
    conditions.add(_Recorder())
    assert bool(conditions)


def test_add_observer_rejects_a_condition_item():
    with pytest.raises(TypeError, match="NonlinearIterationObserver"):
        ConditionSet().add_observer(CellSource(cells=[0], rates=[[1.0, 1.0]]))


def test_observers_run_after_every_item_and_see_the_final_residual():
    conditions = ConditionSet()
    model = _StubModel()
    item = conditions.add(CellSource(cells=[1], rates=[[3.0, 5.0]]))
    item.bind(model)
    observer = conditions.add(_Recorder())
    ctx = _ctx(model)
    conditions.apply(ctx)
    (_, _, seen_rhs) = observer.seen[0]
    # rhs[block 1] == -q * dt: the observer saw the contribution, not the zero
    np.testing.assert_allclose(seen_rhs[2:4], [-1.5, -2.5])
    np.testing.assert_array_equal(seen_rhs, ctx.rhs)


def test_an_observer_cannot_write_the_residual():
    conditions = ConditionSet()

    class _Writer(NonlinearIterationObserver):
        def observe(self, ctx):
            ctx.rhs[0] = 1.0

    conditions.add(_Writer())
    with pytest.raises(ValueError, match="read-only"):
        conditions.apply(_ctx())


def test_an_observer_cannot_write_the_jacobian():
    conditions = ConditionSet()
    engine = _FakeCSREngine()

    class _Writer(NonlinearIterationObserver):
        def observe(self, ctx):
            ctx.jac.add_block(0, np.ones((N_VARS, N_VARS)))

    conditions.add(_Writer())
    with pytest.raises(ValueError, match="read-only"):
        conditions.apply(_ctx(jac=BlockCSRView(engine, N_VARS)))
    assert not np.any(engine.jac_vals)


def test_an_observer_cannot_write_the_state():
    conditions = ConditionSet()

    class _Writer(NonlinearIterationObserver):
        def observe(self, ctx):
            ctx.X[0] = 99.0

    conditions.add(_Writer())
    with pytest.raises(ValueError, match="read-only"):
        conditions.apply(_ctx())


def test_read_only_context_still_reads_everything():
    engine = _FakeCSREngine()
    engine.jac_vals[:] = np.arange(engine.jac_vals.size)
    view = BlockCSRView(engine, N_VARS, pattern=("p",))
    ctx = _ctx(jac=view)
    frozen = ctx.read_only()
    assert frozen.dt == ctx.dt and frozen.t == ctx.t
    assert frozen.n_vars == N_VARS and frozen.iteration == ctx.iteration
    assert frozen.jac.pattern == ("p",)
    assert frozen.jac.block_pos(1, 2) == view.block_pos(1, 2)
    assert frozen.jac.diag_pos(2) == view.diag_pos(2)
    np.testing.assert_array_equal(frozen.jac.jac_vals, engine.jac_vals)
    np.testing.assert_array_equal(frozen.X, ctx.X)
    # the original stays writable
    ctx.rhs[0] = 7.0


def test_observers_receive_the_timestep_callbacks():
    conditions = ConditionSet()
    observer = conditions.add(_Recorder())
    conditions.on_timestep_start(0.5, 1.0)
    conditions.on_timestep_converged(0.5, 1.0)
    conditions.on_timestep_failed(0.25, 1.5)
    assert observer.events == [
        ("start", 0.5, 1.0),
        ("converged", 0.5, 1.0),
        ("failed", 0.25, 1.5),
    ]


def test_observer_only_set_still_refuses_an_apply_rhs_flux_override():
    from darts.models.darts_model import DartsModel

    class _Overriding(DartsModel):
        def apply_rhs_flux(self, dt, t):
            pass

    model = object.__new__(_Overriding)
    model.nonlinear_solver = None
    model.platform = "cpu"
    model.physics = _StubPhysics(_FakeCSREngine())
    model.reservoir = _StubReservoir()
    conditions = ConditionSet()
    conditions.add(_Recorder())
    with pytest.raises(RuntimeError, match="observer"):
        conditions.compile(model)


# --------------------------------------------------------------- E9: restart
class _StatefulItem(ConditionItem):
    """An item whose accumulated state would be lost across a restart."""

    contribution = NO_CONTRIBUTION
    carries_restart_state = True

    def __init__(self, produced=0.0):
        self.produced = float(produced)

    def apply(self, ctx):
        return

    def save_restart_state(self):
        return {"produced": self.produced}

    def load_restart_state(self, state):
        self.produced = float(state["produced"])


def test_stateless_sets_write_no_sidecar(tmp_path):
    conditions = ConditionSet()
    conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    restart = tmp_path / "reservoir_solution.h5"
    restart.write_text("")
    assert conditions.save_restart_state(str(restart)) is None
    assert list(tmp_path.iterdir()) == [restart]
    conditions.load_restart_state(str(restart))  # and no refusal


def test_state_round_trips_through_the_sidecar(tmp_path):
    restart = str(tmp_path / "reservoir_solution.h5")
    saved = ConditionSet()
    saved.add(_StatefulItem(produced=42.5))
    path = saved.save_restart_state(restart)
    assert path == restart + ConditionSet.RESTART_SIDECAR_SUFFIX
    payload = json.loads(open(path).read())
    assert payload["items"][0]["state"] == {"produced": 42.5}

    restored = ConditionSet()
    item = restored.add(_StatefulItem())
    assert item.produced == 0.0
    restored.load_restart_state(restart)
    assert item.produced == 42.5


def test_a_missing_sidecar_refuses_the_restart_loudly(tmp_path):
    restart = str(tmp_path / "reservoir_solution.h5")
    conditions = ConditionSet()
    conditions.add(_StatefulItem(produced=1.0))
    with pytest.raises(RuntimeError, match="carries_restart_state") as excinfo:
        conditions.load_restart_state(restart)
    assert "_StatefulItem" in str(excinfo.value)
    assert "save_restart_state" in str(excinfo.value)


def test_a_sidecar_from_a_different_model_refuses_the_restart(tmp_path):
    restart = str(tmp_path / "reservoir_solution.h5")
    saved = ConditionSet()
    saved.add(_StatefulItem(produced=3.0))
    saved.save_restart_state(restart)

    class _OtherStateful(_StatefulItem):
        pass

    other = ConditionSet()
    other.add(_OtherStateful())
    with pytest.raises(RuntimeError, match="different set of conditions"):
        other.load_restart_state(restart)


def test_declaring_state_without_implementing_it_is_an_error():
    class _Incomplete(ConditionItem):
        carries_restart_state = True

        def apply(self, ctx):
            return

    conditions = ConditionSet()
    conditions.add(_Incomplete())
    with pytest.raises(NotImplementedError, match="save_restart_state"):
        conditions.save_restart_state("whatever.h5")


def test_restart_defers_binding_until_the_engine_exists():
    """init(restart=True) never calls reset(), so there is no matrix to bind to."""
    from darts.models.darts_model import DartsModel

    class _Plain(DartsModel):
        pass

    class _EmptyEngine:
        X = []
        jac_rows = jac_cols = jac_diags = jac_vals = []
        opt_history_matching = False

    model = object.__new__(_Plain)
    model.nonlinear_solver = None
    model.platform = "cpu"
    model.restart = True
    model.physics = _StubPhysics(_EmptyEngine())
    model.reservoir = _StubReservoir()
    conditions = ConditionSet()
    item = conditions.add(
        CellSource(cells=[1], rates=[[1.0, 1.0]], d_rates=np.zeros((1, 2, 2)))
    )
    conditions.compile(model)
    assert conditions.deferred is True
    assert item._diag_pos is None  # nothing was resolved against a missing matrix

    # ... and once load_restart_data() has reset the engine, compile() binds
    model.restart = False
    model.physics.engine = _FakeCSREngine()
    conditions.compile(model)
    assert conditions.deferred is False
    assert item._diag_pos.tolist() == [3]


# ------------------------------------------------------------- E7: selectors
def test_block_indices_selector_resolves_to_its_indices():
    model = _StubModel()
    assert BlockIndices([2, 0]).resolve(model).tolist() == [2, 0]


def test_resolve_blocks_accepts_plain_indices_unchanged():
    model = _StubModel()
    assert resolve_blocks([1, 2], model).tolist() == [1, 2]
    assert resolve_blocks(np.array([0]), model).tolist() == [0]
    assert resolve_blocks(BlockIndices([1]), model).tolist() == [1]


def test_where_selects_by_centroid_predicate():
    centroids = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    selector = Where(lambda x, y, z: x >= 10.0, centroids=centroids)
    assert selector.resolve(_StubModel()).tolist() == [1, 2]


def test_where_resolves_centroids_from_the_reservoir():
    reservoir = _StubReservoir(
        n_blocks=3, centroids=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    )
    model = _StubModel(reservoir)
    assert Where(lambda x, y, z: x > 0.5).resolve(model).tolist() == [1, 2]


def test_where_says_so_when_no_centroids_are_available():
    with pytest.raises(RuntimeError, match="centroids"):
        Where(lambda x, y, z: x > 0).resolve(_StubModel())


def test_where_rejects_a_predicate_of_the_wrong_shape():
    centroids = np.zeros((3, 3))
    with pytest.raises(ValueError, match="one bool per cell"):
        Where(lambda x, y, z: np.array([True]), centroids=centroids).resolve(
            _StubModel()
        )


def test_named_region_uses_a_reservoir_cell_group():
    reservoir = _StubReservoir(n_blocks=4, cell_groups={"aquifer": [2, 3]})
    assert NamedRegion("aquifer").resolve(_StubModel(reservoir)).tolist() == [2, 3]


def test_named_region_accepts_a_boolean_mask():
    reservoir = _StubReservoir(
        n_blocks=3, cell_groups={"top": np.array([False, True, True])}
    )
    assert NamedRegion("top").resolve(_StubModel(reservoir)).tolist() == [1, 2]


def test_named_region_falls_back_to_the_operator_region():
    reservoir = _StubReservoir(n_blocks=4)
    reservoir.mesh.op_num = np.array([0, 1, 1, 0])
    model = _StubModel(reservoir, regions=("shale", "sand"))
    assert NamedRegion("sand").resolve(model).tolist() == [1, 2]
    assert NamedRegion("shale").resolve(model).tolist() == [0, 3]


def test_named_region_lists_the_available_regions_when_unknown():
    model = _StubModel(_StubReservoir(), regions=("shale",))
    with pytest.raises(KeyError, match="shale"):
        NamedRegion("nope").resolve(model)


def test_cell_source_accepts_a_selector():
    model = _StubModel()
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    item = CellSource(
        cells=Where(lambda x, y, z: x >= 1.0, centroids=centroids),
        rates=lambda t: np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    item.bind(model)
    assert item.cells.tolist() == [1, 2]
    ctx = _ctx(model)
    item.apply(ctx)
    np.testing.assert_allclose(ctx.rhs, [0.0, 0.0, -0.5, -1.0, -1.5, -2.0])


def test_dirichlet_pin_accepts_a_selector():
    model = _StubModel()
    pin = DirichletPin(cells=BlockIndices([2]), equation=1, values=9.0)
    pin.bind(model)
    assert pin.cells.tolist() == [2]
    assert pin._x_idx.tolist() == [5]


def test_a_bare_selector_has_no_length_before_it_is_resolved():
    with pytest.raises(TypeError, match="no length"):
        len(BlockIndices([1, 2]))


def test_selector_base_class_must_be_implemented():
    with pytest.raises(NotImplementedError, match="resolve"):
        Selector().resolve(_StubModel())


def test_interface_flux_accepts_selectors_as_connection_members():
    class _Flux(InterfaceFlux):
        def connection_flux(self, t, state_row, state_col):
            return (
                np.zeros(N_VARS),
                np.zeros((N_VARS, N_VARS)),
                np.zeros((N_VARS, N_VARS)),
            )

    model = _StubModel()
    flux = _Flux([(BlockIndices([1]), 2)])
    assert flux.connections is None  # deferred until a model resolves it
    assert flux.resolve_connections(model) == [(1, 2)]


def test_interface_flux_rejects_a_multi_block_selector():
    class _Flux(InterfaceFlux):
        def connection_flux(self, t, state_row, state_col):
            return (
                np.zeros(N_VARS),
                np.zeros((N_VARS, N_VARS)),
                np.zeros((N_VARS, N_VARS)),
            )

    flux = _Flux([(BlockIndices([0, 1]), 2)])
    with pytest.raises(ValueError, match="exactly one block"):
        flux.resolve_connections(_StubModel())


def test_interface_flux_without_selectors_resolves_at_construction():
    class _Flux(InterfaceFlux):
        def connection_flux(self, t, state_row, state_col):
            return (
                np.zeros(N_VARS),
                np.zeros((N_VARS, N_VARS)),
                np.zeros((N_VARS, N_VARS)),
            )

    assert _Flux([(1, 2)]).connections == [(1, 2)]


# ----------------------------------------------- R4: frozen CSR STRUCTURE too
# A "read-only" BlockCSRView twin froze only jac_vals; the structural arrays
# (rows/cols/diags) passed through writable, so an observer could corrupt the
# CSR pattern -- worse than writing a value. All four must refuse writes.


def _observer_context_with_jacobian():
    engine = _FakeCSREngine()
    view = BlockCSRView(engine, N_VARS)
    return engine, _ctx(jac=view)


@pytest.mark.parametrize("array", ["jac_rows", "jac_cols", "jac_diags", "jac_vals"])
def test_an_observer_cannot_write_any_jacobian_array(array):
    engine, ctx = _observer_context_with_jacobian()
    before = np.array(getattr(engine, array), copy=True)

    class _Writer(NonlinearIterationObserver):
        def observe(self, observed):
            getattr(observed.jac, array)[0] = 7

    conditions = ConditionSet()
    conditions.add(_Writer())
    with pytest.raises(ValueError, match="read-only"):
        conditions.apply(ctx)
    np.testing.assert_array_equal(getattr(engine, array), before)


@pytest.mark.parametrize("array", ["jac_rows", "jac_cols", "jac_diags", "jac_vals"])
def test_the_read_only_twin_freezes_every_array_directly(array):
    engine = _FakeCSREngine()
    twin = BlockCSRView(engine, N_VARS).read_only()
    frozen = getattr(twin, array)
    assert frozen.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        frozen[0] = 7
    # freezing is a view property: the engine's own array stays writable
    np.asarray(getattr(engine, array))[0] = np.asarray(getattr(engine, array))[0]


def test_the_read_only_twin_still_navigates_the_pattern():
    engine = _FakeCSREngine()
    view = BlockCSRView(engine, N_VARS)
    twin = view.read_only()
    assert twin.block_pos(1, 2) == view.block_pos(1, 2)
    assert twin.diag_pos(2) == view.diag_pos(2)


# ------------------------------- R6: observer-only sets on a mechanics model
# ConditionSet.compile() returns early for observer-only sets BEFORE its
# mechanics rejection, so such a set compiles on a mechanics model. The runtime
# guard in DartsModel.apply_rhs_flux() must agree: it triggers on contributing
# ITEMS, not on the truthiness of the whole set, so the observer-only model
# survives its first assembly (the exact scenario of the review).


def _mechanics_stub_model(conditions):
    from darts.models.darts_model import DartsModel
    from darts.nonlinear_solvers.mechanics import MechanicsNewtonSolver

    class _MechModel(DartsModel):
        pass

    engine = _FakeCSREngine()
    engine.RHS = np.zeros(3 * N_VARS)
    engine.Xn = np.zeros(3 * N_VARS)

    model = object.__new__(_MechModel)
    model.nonlinear_solver = object.__new__(MechanicsNewtonSolver)
    model.platform = "cpu"
    model.physics = _StubPhysics(engine)
    model.reservoir = _StubReservoir()
    model.conditions = conditions
    model._conditions_csr_view = None
    model._assembly_iteration = 0
    model._pattern_version = 0
    return model


def test_observer_only_mechanics_model_compiles_and_survives_its_first_apply():
    conditions = ConditionSet()
    observer = conditions.add(_Recorder())
    model = _mechanics_stub_model(conditions)

    # compiles: the observer-only early return precedes the mechanics rejection
    assert conditions.compile(model) is conditions

    # ... and the first assembly must AGREE with that decision, not raise
    model.apply_rhs_flux(dt=0.5, t=1.0)
    assert len(observer.seen) == 1
    (t_seen, dt_seen, rhs_seen) = observer.seen[0]
    assert (t_seen, dt_seen) == (1.0, 0.5)
    np.testing.assert_array_equal(rhs_seen, np.zeros(3 * N_VARS))


def test_item_carrying_sets_still_refuse_the_mechanics_runtime_path():
    conditions = ConditionSet()
    conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    model = _mechanics_stub_model(conditions)
    with pytest.raises(RuntimeError, match="mechanics"):
        model.apply_rhs_flux(dt=0.5, t=1.0)
