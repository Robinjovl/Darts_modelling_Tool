"""M4 contract hardening: declared stencils, pattern versioning, conflict typing.

Review items E5 (declared stencil + pattern version) and E6 (additive versus
replacement), exercised at two levels.

**Stub level** (no engine, no reservoir): a recording ``_StubMesh`` captures the
``add_conn`` calls :meth:`ConditionSet.declare_stencil` makes, so the decisive
behaviours are asserted directly -- an already-connected pair adds NOTHING, an
unconnected pair adds exactly one zero-transmissibility connection, a wellhead
coupling and a self-coupling are refused, and :meth:`verify_stencil` catches both
a missing and a duplicated coupling against the frozen connection arrays.

**Model level** (the real coupled DFM/OLGA benchmark, CPU): the E5 payoff. Two
models are built from the same source -- one as shipped, with the
zero-well-index "fake" perforation whose only purpose was to smuggle the four
IPR blocks into the sparsity pattern, and one with NO PERFORATION AT ALL, where
``LinearDFMWellIPRHook`` declares the coupling instead. They must produce the
same Jacobian pattern and the same simulation.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("darts.engines")

from darts.models.conditions import (  # noqa: E402
    ADDITIVE,
    NO_CONTRIBUTION,
    REPLACEMENT,
    AssemblyContext,
    BlockCSRView,
    CellSource,
    ConditionItem,
    ConditionSet,
    DirichletPin,
    InterfaceFlux,
    pattern_identity,
)

N_VARS = 2
BLOCK_SIZE = N_VARS * N_VARS


# ------------------------------------------------------------------- stubs
class _StubMesh:
    """Records ``add_conn`` and exposes the frozen two-way arrays.

    ``block_m``/``block_p`` start as the pre-frozen (empty) state; the test
    calls :meth:`freeze` to emulate ``reverse_and_sort()`` producing the sorted
    two-way list from the one-way one.
    """

    def __init__(self, n_blocks, n_res_blocks, one_way=()):
        self.n_blocks = n_blocks
        self.n_res_blocks = n_res_blocks
        self.op_num = np.zeros(n_blocks, dtype=np.int64)
        self.one_way = list(one_way)
        self.added = []
        self.block_m = []
        self.block_p = []

    def add_conn(self, block_m, block_p, trans, transD, is_dfm):
        self.added.append((block_m, block_p, trans, transD, is_dfm))
        self.one_way.append((block_m, block_p))

    def freeze(self, extra=()):
        pairs = list(self.one_way) + list(extra)
        both = [(m, p) for m, p in pairs] + [(p, m) for m, p in pairs]
        both.sort()
        self.block_m = [m for m, _ in both]
        self.block_p = [p for _, p in both]


class _StubWell:
    class MS_Type:  # noqa: N801 - mirrors darts.engines.ms_well.MS_Type
        EPM = 0
        DFM = 1

    def __init__(self, name, well_head_idx, num_segments, perforations=(), ms_type=1):
        self.name = name
        self.well_head_idx = well_head_idx
        self.well_body_idx = well_head_idx + 1
        self.num_segments = num_segments
        self.n_segments = num_segments - 1
        self.perforations = list(perforations)
        self.ms_type = ms_type
        self.with_lateral_heat_transfer = False
        self.connections_for_lateral_heat_transfer = []


class _StubReservoir:
    def __init__(self, mesh, wells=(), cell_m=None, cell_p=None):
        self.mesh = mesh
        self.wells = list(wells)
        if cell_m is not None:
            self.cell_m = cell_m
            self.cell_p = cell_p

    def get_well(self, name):
        for well in self.wells:
            if well.name == name:
                return well


class _StubPhysics:
    def __init__(self, engine=None, n_vars=N_VARS):
        self.n_vars = n_vars
        self.engine = engine
        self.regions = [0]


class _StubModel:
    def __init__(self, reservoir, engine=None):
        self.reservoir = reservoir
        self.physics = _StubPhysics(engine)
        self.platform = "cpu"
        self._pattern_version = 0


class _DeclaringItem(ConditionItem):
    """Minimal item that declares couplings and writes nothing."""

    contribution = NO_CONTRIBUTION

    def __init__(self, couplings):
        self.couplings = tuple(couplings)

    def declare_stencil(self, model):
        return self.couplings

    def apply(self, ctx):
        return


class _WellCouplingItem(ConditionItem):
    """A declaring item that also reports the well its coupling connects.

    ``DartsModel._require_declared_well_coupling`` reads
    ``declared_well_names`` so a well with NO perforation is accepted when an
    item supplies its reservoir coupling instead.
    """

    contribution = NO_CONTRIBUTION

    def __init__(self, couplings, wells=()):
        self.couplings = tuple(couplings)
        self.wells = set(wells)

    def declare_stencil(self, model):
        return self.couplings

    def declared_well_names(self, model):
        return self.wells

    def apply(self, ctx):
        return


def _model_with_one_well(perforations=((1, 1, 0.0, 0.0),)):
    """4 reservoir blocks + a 3-segment DFM well starting at block 4."""
    mesh = _StubMesh(n_blocks=7, n_res_blocks=4)
    well = _StubWell("W1", well_head_idx=4, num_segments=3, perforations=perforations)
    reservoir = _StubReservoir(
        mesh,
        wells=[well],
        cell_m=[0, 1, 2],
        cell_p=[1, 2, 3],
    )
    # what add_wells() put into the one-way list: perforations then the chain
    for i_w, i_r, _, _ in perforations:
        mesh.one_way.append((well.well_body_idx + i_w, i_r))
    for segment in range(well.num_segments - 1):
        mesh.one_way.append((4 + segment, 5 + segment))
    mesh.one_way[:0] = list(zip(reservoir.cell_m, reservoir.cell_p, strict=True))
    return _StubModel(reservoir), mesh, well


# ------------------------------------------------------- E5: declarer discovery
def test_stencil_declarers_empty_when_nothing_declares():
    model, _, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    assert conditions.stencil_declarers(model) == []


def test_stencil_declarers_finds_every_declaring_item_in_order():
    model, _, _ = _model_with_one_well()
    conditions = ConditionSet()
    first = conditions.add(_DeclaringItem([(6, 2)]))
    second = conditions.add(_WellCouplingItem([(5, 0)]))
    # a non-declaring item is not reported, and registration order is kept
    conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    assert conditions.stencil_declarers(model) == [first, second]


def test_declare_stencil_is_a_noop_without_declarers():
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(CellSource(cells=[0], rates=[[1.0, 1.0]]))
    assert conditions.declare_stencil(model) == ()
    assert mesh.added == []


# ---------------------------------------------------- E5: the decisive behaviour
def test_declared_coupling_that_already_exists_adds_nothing():
    """A pair a perforation already connects must NOT be added again."""
    model, mesh, well = _model_with_one_well()
    conditions = ConditionSet()
    # the perforation connects well block well_body_idx + i_w to reservoir cell 1
    perforated_block = well.well_body_idx + well.perforations[0][0]
    conditions.add(_DeclaringItem([(perforated_block, well.perforations[0][1])]))
    added = conditions.declare_stencil(model)
    assert added == ()
    assert mesh.added == []
    assert conditions.declared_couplings == (
        (min(perforated_block, 1), max(perforated_block, 1)),
    )


def test_declared_coupling_that_is_absent_is_added_with_zero_transmissibility():
    """This is what removes the fake perforation: the pattern gains the pair."""
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(6, 2)]))  # well segment 6 <-> reservoir cell 2
    added = conditions.declare_stencil(model)
    assert added == ((2, 6),)
    # exactly one connection, zero trans AND zero diffusive trans, not DFM
    assert mesh.added == [(2, 6, 0.0, 0.0, False)]


def test_reservoir_internal_coupling_is_recognized_as_existing():
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(1, 0)]))  # already a reservoir connection
    assert conditions.declare_stencil(model) == ()
    assert mesh.added == []


def test_duplicate_declarations_of_the_same_pair_add_one_connection():
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(6, 2)]))
    conditions.add(_DeclaringItem([(2, 6)]))  # same pair, other order
    assert conditions.declare_stencil(model) == ((2, 6),)
    assert len(mesh.added) == 1


def test_segment_chain_coupling_is_recognized_as_existing():
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(5, 6)]))  # body-to-body: a chain connection
    assert conditions.declare_stencil(model) == ()
    assert mesh.added == []


def test_a_chain_coupling_onto_the_wellhead_is_still_refused():
    """Declaring it means intending to WRITE the well-control row (finding V1)."""
    model, _, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(4, 5)]))
    with pytest.raises(ValueError, match="WELLHEAD"):
        conditions.declare_stencil(model)


def test_wellhead_coupling_is_refused():
    model, _, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(4, 0)]))
    with pytest.raises(ValueError, match="WELLHEAD"):
        conditions.declare_stencil(model)


def test_self_coupling_is_refused():
    model, _, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(6, 6)]))
    with pytest.raises(ValueError, match="self-coupling"):
        conditions.declare_stencil(model)


def test_out_of_range_block_is_refused():
    model, _, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(6, 99)]))
    with pytest.raises(IndexError, match="99"):
        conditions.declare_stencil(model)


def test_reservoir_coupling_refused_without_a_connection_list():
    mesh = _StubMesh(n_blocks=4, n_res_blocks=4)
    model = _StubModel(_StubReservoir(mesh))  # no cell_m/cell_p
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(0, 2)]))
    with pytest.raises(RuntimeError, match="does not expose its connection list"):
        conditions.declare_stencil(model)


# ------------------------------------------------------------ E5: verification
def test_verify_stencil_accepts_a_correctly_added_coupling():
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(6, 2)]))
    conditions.declare_stencil(model)
    mesh.freeze()
    assert conditions.verify_stencil(model) is conditions


def test_verify_stencil_catches_a_missing_coupling():
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(6, 2)]))
    conditions.declare_stencil(model)
    mesh.one_way.remove((2, 6))  # the connection never reached the frozen list
    mesh.freeze()
    with pytest.raises(RuntimeError, match="expected exactly one of each"):
        conditions.verify_stencil(model)


def test_verify_stencil_catches_a_duplicated_coupling():
    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_DeclaringItem([(6, 2)]))
    conditions.declare_stencil(model)
    mesh.freeze(extra=[(2, 6)])  # the pair existed after all -> duplicate columns
    with pytest.raises(RuntimeError, match="expected exactly one of each"):
        conditions.verify_stencil(model)


def test_interface_flux_declares_its_connections():
    class _Flux(InterfaceFlux):
        def connection_flux(self, t, state_row, state_col):
            return np.zeros(N_VARS), np.zeros((N_VARS,) * 2), np.zeros((N_VARS,) * 2)

    model, mesh, _ = _model_with_one_well()
    conditions = ConditionSet()
    conditions.add(_Flux([(6, 2)]))
    assert conditions.declare_stencil(model) == ((2, 6),)
    assert mesh.added == [(2, 6, 0.0, 0.0, False)]


# --------------------------------------------------------- E5: pattern version
class _PatternEngine:
    def __init__(self, n_positions=7):
        self.jac_rows = np.array([0, 2, 5, 7], dtype=np.int64)
        self.jac_cols = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int64)
        self.jac_diags = np.array([0, 3, 6], dtype=np.int64)
        self.jac_vals = np.zeros(n_positions * BLOCK_SIZE)
        self.opt_history_matching = False


def test_pattern_identity_changes_with_the_version_counter():
    model = _StubModel(_StubReservoir(_StubMesh(3, 3)), engine=_PatternEngine())
    first = pattern_identity(model)
    assert pattern_identity(model) == first
    model._pattern_version += 1
    assert pattern_identity(model) != first


def test_pattern_identity_changes_when_the_matrix_is_reallocated():
    model = _StubModel(_StubReservoir(_StubMesh(3, 3)), engine=_PatternEngine())
    first = pattern_identity(model)
    model.physics.engine = _PatternEngine(n_positions=9)
    assert pattern_identity(model) != first


def test_stale_positions_are_re_resolved_instead_of_written():
    """The failure this prevents: writing yesterday's offsets into a new matrix."""

    class _Counting(ConditionItem):
        contribution = NO_CONTRIBUTION

        def __init__(self):
            self.binds = 0

        def bind(self, model):
            self.binds += 1
            self.stamp_pattern(model)

        def apply(self, ctx):
            return

    model = _StubModel(_StubReservoir(_StubMesh(3, 3)), engine=_PatternEngine())
    item = _Counting()
    item.bind(model)
    assert item.binds == 1

    def _ctx():
        view = BlockCSRView(
            model.physics.engine, N_VARS, pattern=pattern_identity(model)
        )
        return AssemblyContext(
            rhs=np.zeros(6),
            jac=view,
            X=np.zeros(6),
            Xn=np.zeros(6),
            dt=1.0,
            t=0.0,
            iteration=0,
            n_vars=N_VARS,
            n_res_blocks=3,
        )

    item.rebind_if_stale(_ctx())
    assert item.binds == 1  # same pattern: no work

    model._pattern_version += 1  # what DartsModel.reset() does
    item.rebind_if_stale(_ctx())
    assert item.binds == 2  # re-resolved


def test_unstamped_items_are_never_rebound():
    class _Plain(ConditionItem):
        contribution = NO_CONTRIBUTION

        def __init__(self):
            self.binds = 0

        def bind(self, model):
            self.binds += 1

        def apply(self, ctx):
            return

    model = _StubModel(_StubReservoir(_StubMesh(3, 3)), engine=_PatternEngine())
    item = _Plain()
    item.bind(model)
    view = BlockCSRView(model.physics.engine, N_VARS, pattern=("something", "else"))
    ctx = AssemblyContext(
        rhs=np.zeros(6),
        jac=view,
        X=np.zeros(6),
        Xn=np.zeros(6),
        dt=1.0,
        t=0.0,
        iteration=0,
        n_vars=N_VARS,
        n_res_blocks=3,
    )
    item.rebind_if_stale(ctx)
    assert item.binds == 1


def test_reset_bumps_the_pattern_version_and_drops_the_cached_view():
    from darts.models.darts_model import DartsModel

    class _Model(DartsModel):
        pass

    class _Engine:
        def __init__(self):
            self.inits = 0

        def init(self, *args):
            self.inits += 1

    model = object.__new__(_Model)
    model._pattern_version = 3
    model._conditions_csr_view = "stale"
    model.physics = _StubPhysics(_Engine())
    model.physics.thermal_var_itor = None

    class _Mesh:
        pass

    class _Reservoir:
        mesh = _Mesh()
        wells = []

    model.reservoir = _Reservoir()
    model.op_list = []
    model.params = None

    class _Timer:
        node = {"simulation": None}

    model.timer = _Timer()
    model.reset()
    assert model._pattern_version == 4
    assert model._conditions_csr_view is None


# ------------------------------------------------ E6: additive vs replacement
class _FakeCSREngine(_PatternEngine):
    pass


def _row_conflict_model():
    from darts.models.darts_model import DartsModel

    class _Plain(DartsModel):
        pass

    model = object.__new__(_Plain)
    model.nonlinear_solver = None
    model.platform = "cpu"
    model.physics = _StubPhysics(_FakeCSREngine())
    model.reservoir = _StubReservoir(_StubMesh(3, 3))
    model._pattern_version = 0
    return model


def test_additive_contribution_to_a_claimed_row_is_rejected():
    model = _row_conflict_model()
    conditions = ConditionSet()
    conditions.add(DirichletPin(cells=[1], equation=0, values=5.0, mode="row"))
    conditions.add(CellSource(cells=[1], rates=[[1.0, 1.0]]))
    with pytest.raises(RuntimeError, match="claims it") as excinfo:
        conditions.compile(model)
    message = str(excinfo.value)
    assert "DirichletPin" in message and "CellSource" in message
    assert "block 1, equation 0" in message


def test_two_claims_on_the_same_row_are_rejected():
    model = _row_conflict_model()
    conditions = ConditionSet()
    conditions.add(DirichletPin(cells=[1], equation=0, values=5.0, mode="row"))
    conditions.add(DirichletPin(cells=[1], equation=0, values=7.0, mode="row"))
    with pytest.raises(RuntimeError, match="claim the same equation row"):
        conditions.compile(model)


def test_claims_on_different_rows_of_the_same_block_are_fine():
    model = _row_conflict_model()
    conditions = ConditionSet()
    conditions.add(DirichletPin(cells=[1], equation=0, values=5.0, mode="row"))
    conditions.add(DirichletPin(cells=[1], equation=1, values=7.0, mode="row"))
    assert conditions.compile(model) is conditions


def test_state_mode_pin_is_neither_additive_nor_a_claim():
    """A projection of the STATE does not compete with a source on that row."""
    model = _row_conflict_model()
    pin = DirichletPin(cells=[1], equation=0, values=5.0)  # mode="state"
    assert pin.contribution == NO_CONTRIBUTION
    conditions = ConditionSet()
    conditions.add(pin)
    conditions.add(CellSource(cells=[1], rates=[[1.0, 1.0]]))
    assert conditions.compile(model) is conditions


def test_two_additive_items_may_share_a_row():
    model = _row_conflict_model()
    conditions = ConditionSet()
    conditions.add(CellSource(cells=[1], rates=[[1.0, 1.0]]))
    conditions.add(CellSource(cells=[1], rates=[[2.0, 2.0]]))
    assert conditions.compile(model) is conditions


def test_row_mode_pin_declares_replacement_and_default_is_additive():
    assert DirichletPin(cells=[0], equation=0, values=1.0, mode="row").contribution == (
        REPLACEMENT
    )
    assert CellSource(cells=[0], rates=[[1.0, 1.0]]).contribution == ADDITIVE


def test_unknown_contribution_kind_is_rejected_at_registration():
    class _Bad(ConditionItem):
        contribution = "sometimes"

        def apply(self, ctx):
            return

    with pytest.raises(ValueError, match="sometimes"):
        ConditionSet().add(_Bad())


# ------------------------------------------------- the real-model E5 payoff
MODEL_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "models"
    / "dfm_well"
    / "2ph_2comp_isothermal_dfm_vertical_well_vs_olga"
)


def _load_olga_model_module():
    spec = importlib.util.spec_from_file_location(
        "declared_stencil_olga_model", MODEL_DIR / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODEL_DIR))
    return module


def _build_olga_models():
    """The shipped (perforation-free) model and a perforated twin, initialized.

    The shipped model IS the perforation-free configuration now: its hook
    addresses the (well segment, reservoir block) pair directly and the
    declaration stage adds the coupling. The twin reconstructs the HISTORICAL
    configuration -- a zero-well-index perforation whose only purpose is to
    smuggle the mesh connection in, and a hook addressed through it -- so the
    equivalence of the two spellings stays proven in both directions.
    """
    pytest.importorskip("dartsflash")
    if not MODEL_DIR.is_dir():
        pytest.skip(f"model directory not found: {MODEL_DIR}")
    module = _load_olga_model_module()
    from darts.pipes.linear_dfm_well_ipr import (
        LinearDFMWellIPRConnection,
        LinearDFMWellIPRHook,
    )

    class _PerforationModel(module.Model):
        """Identical, except the coupling is carried by a zero-WI perforation.

        The perforation supplies the mesh connection (so the declaration stage
        finds the coupling already present and adds nothing), and the hook is
        re-addressed through ``perforation_index``, exactly as every model
        spelled it before stencil declaration existed.
        """

        def set_wells(self, verbose=None):
            super().set_wells()
            (hook,) = [
                item
                for item in self.conditions.items
                if isinstance(item, LinearDFMWellIPRHook)
            ]
            (connection,) = hook.connections
            self.reservoir.add_perforation(
                "I1",
                res_cell_idx=(1, 1, 1),
                well_seg_idx=int(connection.well_segment_index),
                well_diameter=self.well_1_ID,
                well_index=0.0,
                well_indexD=0.0,
            )
            well = self.reservoir.get_well("I1")
            self.conditions.items = [
                item
                for item in self.conditions.items
                if not isinstance(item, LinearDFMWellIPRHook)
            ]
            self.conditions.add(
                LinearDFMWellIPRHook(
                    self,
                    [
                        LinearDFMWellIPRConnection(
                            well_name="I1",
                            perforation_index=len(well.perforations) - 1,
                            pi=connection.pi,
                            pi_type=connection.pi_type,
                            ipr_pressure_offset=connection.ipr_pressure_offset,
                            ipr_intercept=connection.ipr_intercept,
                        )
                    ],
                )
            )

    cwd = pathlib.Path.cwd()
    import os

    os.chdir(MODEL_DIR)
    try:
        shipped = module.Model()
        shipped.init(platform="cpu", verbose=0)
        perforated = _PerforationModel()
        perforated.init(platform="cpu", verbose=0)
    finally:
        os.chdir(cwd)
    return shipped, perforated


@pytest.fixture(scope="module")
def olga_models():
    return _build_olga_models()


def test_shipped_model_no_longer_creates_the_dummy_perforation(olga_models):
    shipped, _ = olga_models
    assert len(shipped.reservoir.get_well("I1").perforations) == 0
    assert shipped.conditions.declared_couplings  # the hook declared its pair
    # the coupling the perforation used to smuggle in is DECLARED and added
    assert shipped.conditions.added_couplings == shipped.conditions.declared_couplings


def test_the_perforated_twin_declares_a_coupling_that_already_exists(olga_models):
    shipped, perforated = olga_models
    assert len(perforated.reservoir.get_well("I1").perforations) == 1
    assert perforated.conditions.declared_couplings == (
        shipped.conditions.declared_couplings
    )
    assert perforated.conditions.added_couplings == ()  # the perforation supplied it


def test_the_declared_pattern_equals_the_fake_perforation_pattern(olga_models):
    shipped, perforated = olga_models
    for attribute in ("jac_rows", "jac_cols", "jac_diags"):
        np.testing.assert_array_equal(
            np.asarray(getattr(shipped.physics.engine, attribute)),
            np.asarray(getattr(perforated.physics.engine, attribute)),
            err_msg=f"{attribute} differs between the two configurations",
        )
    assert len(shipped.reservoir.mesh.block_m) == len(perforated.reservoir.mesh.block_m)


def _ipr_hook(model):
    from darts.pipes.linear_dfm_well_ipr import LinearDFMWellIPRHook

    (hook,) = [
        item for item in model.conditions if isinstance(item, LinearDFMWellIPRHook)
    ]
    return hook


def _hook_contribution(model, p_well, p_res, dt=1e-3):
    """RHS and Jacobian the IPR hook adds at a prescribed state."""
    engine = model.physics.engine
    n_vars = model.physics.n_vars
    hook = _ipr_hook(model)
    resolved = hook._get_resolved_connections(
        BlockCSRView(engine, n_vars, pattern=pattern_identity(model))
    )[0]
    well_block, res_block = resolved["well_block_idx"], resolved["res_block_idx"]

    X = np.asarray(engine.X)
    saved = X.copy()
    rhs = np.asarray(engine.RHS)
    jac = np.asarray(engine.jac_vals)
    ctx = AssemblyContext(
        rhs=rhs,
        jac=BlockCSRView(engine, n_vars, pattern=pattern_identity(model)),
        X=X,
        Xn=np.asarray(engine.Xn),
        dt=dt,
        t=0.0,
        iteration=0,
        n_vars=n_vars,
        n_res_blocks=model.reservoir.mesh.n_res_blocks,
    )
    try:
        X[well_block * n_vars] = p_well
        X[res_block * n_vars] = p_res
        rhs[:] = 0.0
        jac[:] = 0.0
        hook.apply(ctx)
        return rhs.copy(), jac.copy(), (well_block, res_block)
    finally:
        X[:] = saved
        rhs[:] = 0.0
        jac[:] = 0.0


@pytest.mark.parametrize("p_well,p_res", [(110.0, 102.0), (95.0, 102.0)])
def test_the_declared_configuration_produces_the_same_flux(olga_models, p_well, p_res):
    """The point of E5: identical flux and identical Jacobian, no perforation."""
    shipped, perforated = olga_models
    rhs_a, jac_a, blocks_a = _hook_contribution(shipped, p_well, p_res)
    rhs_b, jac_b, blocks_b = _hook_contribution(perforated, p_well, p_res)
    assert blocks_a == blocks_b
    np.testing.assert_array_equal(rhs_a, rhs_b)
    np.testing.assert_array_equal(jac_a, jac_b)
    assert np.any(rhs_a != 0.0)  # the comparison is not vacuous


def test_the_declared_configuration_simulates_identically(tmp_path):
    """End to end: the fake perforation is provably unnecessary.

    Both configurations are integrated over the same (short) interval from the
    same initial state; the state vectors must agree bit for bit, because a
    zero-well-index perforation and a declared zero-transmissibility connection
    are literally the same ``add_conn`` call -- one made by ``add_wells()``, the
    other by the stencil-declaration stage.
    """
    import os

    shipped, perforated = _build_olga_models()
    cwd = pathlib.Path.cwd()
    os.chdir(MODEL_DIR)
    try:
        for index, model in enumerate((shipped, perforated)):
            model.set_output(output_folder=str(tmp_path / f"out{index}"))
            model.run(
                days=1e-3,
                save_well_data=False,
                save_reservoir_data=False,
                verbose=0,
            )
    finally:
        os.chdir(cwd)
    np.testing.assert_array_equal(
        np.asarray(shipped.physics.engine.X),
        np.asarray(perforated.physics.engine.X),
    )
