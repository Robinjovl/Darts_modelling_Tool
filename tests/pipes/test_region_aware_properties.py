"""Region-aware property resolution in the ``darts.pipes`` hooks (M3.5, E2).

Both hooks used to reach for ``physics.property_containers[0]`` regardless of
which operator region the block they evaluate belongs to, so in a multi-region
model they silently evaluated the wrong fluid. These tests pin the corrected
behaviour with synthetic TWO-REGION physics whose regions carry deliberately
different property containers, so a fallback to region 0 is observable:

* the resolver maps a reservoir block through ``mesh.op_num`` (an index into
  ``op_list``, i.e. slot ``i`` is ``physics.regions[i]``) and a well block
  through the well-operator region ``physics.regions[0]`` -- the container
  ``PhysicsBase.set_operators`` hands to ``WellOperators``;
* :class:`LinearDFMWellIPRHook` resolves the well side and the reservoir side
  INDEPENDENTLY and evaluates with the container of the UPSTREAM side, so the
  molar rate of a mass-PI connection changes with the region of the connected
  reservoir cell;
* :class:`SemiAnalyticalWellLateralHeatTransferHook` flashes the PH well
  segments with the WELL region's container even when that region is not the
  one tagged ``0``;
* an unregistered region raises, naming the block, its region and the
  registered regions, instead of falling back to region 0.
"""

import enum

import numpy as np
import pytest

pytest.importorskip("darts.engines")

from darts.engines import ms_well  # noqa: E402
from darts.models.conditions import (  # noqa: E402
    AssemblyContext,
    BlockCSRView,
)
from darts.pipes.add_lateral_heat_exchange import (  # noqa: E402
    SemiAnalyticalWellLateralHeatTransfer,
    SemiAnalyticalWellLateralHeatTransferHook,
)
from darts.pipes.define_pipe_geometry import PipeGeometry  # noqa: E402
from darts.pipes.linear_dfm_well_ipr import (  # noqa: E402
    LinearDFMWellIPRConnection,
    LinearDFMWellIPRHook,
    PI_Type,
    block_region,
    property_container_for_block,
    registered_regions,
    reservoir_cell_region,
    well_cell_property_container,
    well_cell_region,
)

# ------------------------------------------------------------------ geometry
# 3 reservoir blocks (regions 0, 1, 1) + a 2-segment well (head 3, body 4).
N_RES_BLOCKS = 3
WELL_HEAD_IDX = 3
WELL_BODY_IDX = 4
N_BLOCKS = 5
N_VARS = 2  # isothermal two-component: (p, z_C1)
NC = 2
PERFORATED_RES_BLOCK = 1  # region 1 -- deliberately NOT region 0
DT = 0.5

# Two regions whose mean molecular weights differ by a factor of 10, so a
# fallback to region 0 shows up directly in the molar rate.
MW_REGION_0 = [10.0, 20.0]
MW_REGION_1 = [100.0, 200.0]


class _StateSpec(enum.Enum):
    P = 0
    PT = 1
    PH = 2


class _PropertyContainer:
    """Minimal container: molecular weights plus a PH-flash stand-in."""

    def __init__(self, Mw, temperature_offset=0.0):
        self.Mw = np.array(Mw, dtype=float)
        self.temperature_offset = float(temperature_offset)
        self.temperature = 0.0

    def evaluate(self, state):
        # PH flash stand-in: temperature is the enthalpy entry plus a
        # per-region offset, so which container was used is observable.
        self.temperature = float(state[-1]) + self.temperature_offset


class _Physics:
    StateSpecification = _StateSpec

    def __init__(
        self,
        property_containers,
        regions=None,
        engine=None,
        thermal=False,
        state_spec=_StateSpec.P,
        n_vars=N_VARS,
        nc=NC,
    ):
        self.property_containers = property_containers
        self.regions = list(property_containers) if regions is None else list(regions)
        self.engine = engine
        self.thermal = thermal
        self.state_spec = state_spec
        self.n_vars = n_vars
        self.nc = nc
        self.sim_eps = 1e-12


class _Mesh:
    def __init__(self, op_num, n_res_blocks=N_RES_BLOCKS):
        self.op_num = np.array(op_num, dtype=np.int64)
        self.n_res_blocks = n_res_blocks
        self.cell_spe = np.zeros(len(self.op_num))


class _Well:
    def __init__(self, name="W1", perforations=None):
        self.name = name
        self.ms_type = ms_well.MS_Type.DFM
        # (perf_segment_local, res_block_idx, well_index, well_indexD)
        self.perforations = (
            [(0, PERFORATED_RES_BLOCK, 0.0, 0.0)]
            if perforations is None
            else perforations
        )
        self.well_head_idx = WELL_HEAD_IDX
        self.well_body_idx = WELL_BODY_IDX
        self.num_segments = 2


class _Reservoir:
    def __init__(self, mesh, wells):
        self.mesh = mesh
        self.wells = list(wells)

    def get_well(self, name):
        for well in self.wells:
            if well.name == name:
                return well
        return None


class _Engine:
    """Dense block-CSR over ``N_BLOCKS`` blocks, plus X/RHS numpy views."""

    def __init__(self, n_blocks=N_BLOCKS, n_vars=N_VARS):
        self.X = np.zeros(n_blocks * n_vars)
        self.RHS = np.zeros(n_blocks * n_vars)
        self.jac_rows = np.arange(n_blocks + 1, dtype=np.int64) * n_blocks
        self.jac_cols = np.tile(np.arange(n_blocks, dtype=np.int64), n_blocks)
        self.jac_diags = np.arange(n_blocks, dtype=np.int64) * (n_blocks + 1)
        self.jac_vals = np.zeros(n_blocks * n_blocks * n_vars * n_vars)


class _Model:
    def __init__(self, physics, reservoir):
        self.physics = physics
        self.reservoir = reservoir


def _bind_and_apply(hook, model, dt, t=0.0, with_jacobian=True):
    """Drive the item the way ``ConditionSet`` does: bind once, then apply it
    with an :class:`AssemblyContext` over the stub engine's arrays."""
    engine = model.physics.engine
    n_vars = model.physics.n_vars
    hook.bind(model)
    ctx = AssemblyContext(
        rhs=engine.RHS,
        jac=BlockCSRView(engine, n_vars) if with_jacobian else None,
        X=engine.X,
        Xn=np.zeros_like(engine.X),
        dt=dt,
        t=t,
        iteration=0,
        n_vars=n_vars,
        n_res_blocks=model.reservoir.mesh.n_res_blocks,
    )
    hook.apply(ctx)


def _two_region_model(
    property_containers=None,
    regions=None,
    op_num=None,
    **physics_kwargs,
):
    """A 2-region model: reservoir blocks in regions (0, 1, 1), well in
    ``regions[0]``. ``mesh.op_num`` of the well blocks carries the
    well-operator slot, exactly as ``DartsModel.set_op_list`` leaves it."""
    if property_containers is None:
        property_containers = {
            0: _PropertyContainer(MW_REGION_0),
            1: _PropertyContainer(MW_REGION_1, temperature_offset=1000.0),
        }
    n_regions = len(property_containers)
    if op_num is None:
        op_num = [0, 1, 1] + [n_regions] * (N_BLOCKS - N_RES_BLOCKS)
    engine = _Engine()
    physics = _Physics(
        property_containers, regions=regions, engine=engine, **physics_kwargs
    )
    reservoir = _Reservoir(_Mesh(op_num), [_Well()])
    return _Model(physics, reservoir)


# --------------------------------------------------------- (a) the resolver
def test_registered_regions_prefers_physics_regions_order():
    """The engine's op_list order is physics.regions, which need not be
    sorted, nor equal to the property_containers key order."""
    containers = {
        0: _PropertyContainer(MW_REGION_0),
        1: _PropertyContainer(MW_REGION_1),
    }
    physics = _Physics(containers, regions=[1, 0])
    assert registered_regions(physics) == [1, 0]


def test_registered_regions_falls_back_to_the_container_mapping():
    """A physics double exposing only property_containers still resolves."""

    class _Bare:
        property_containers = [_PropertyContainer(MW_REGION_0)]

    assert registered_regions(_Bare()) == [0]

    class _BareDict:
        property_containers = {7: _PropertyContainer(MW_REGION_0)}

    assert registered_regions(_BareDict()) == [7]


def test_reservoir_block_region_comes_from_op_num():
    model = _two_region_model()
    assert reservoir_cell_region(model, 0) == 0
    assert reservoir_cell_region(model, PERFORATED_RES_BLOCK) == 1
    assert reservoir_cell_region(model, 2) == 1
    assert block_region(model, PERFORATED_RES_BLOCK) == 1
    assert (
        property_container_for_block(model, PERFORATED_RES_BLOCK)
        is model.physics.property_containers[1]
    )


def test_op_num_maps_through_physics_regions_not_by_tag():
    """op_num is an INDEX into op_list, so slot i is physics.regions[i]."""
    containers = {
        0: _PropertyContainer(MW_REGION_0),
        1: _PropertyContainer(MW_REGION_1),
    }
    model = _two_region_model(property_containers=containers, regions=[1, 0])
    # op_num 0 -> regions[0] == 1, op_num 1 -> regions[1] == 0
    assert reservoir_cell_region(model, 0) == 1
    assert reservoir_cell_region(model, PERFORATED_RES_BLOCK) == 0
    assert property_container_for_block(model, 0) is containers[1]


def test_well_block_region_is_the_well_operator_region():
    """Well blocks use property_containers[physics.regions[0]] -- the
    container PhysicsBase.set_operators gives WellOperators -- and never
    mesh.op_num, which set_op_list overwrote with the well-operator slot."""
    containers = {
        0: _PropertyContainer(MW_REGION_0),
        1: _PropertyContainer(MW_REGION_1),
    }
    model = _two_region_model(property_containers=containers, regions=[1, 0])
    assert well_cell_region(model.physics) == 1
    assert block_region(model, WELL_BODY_IDX) == 1
    assert property_container_for_block(model, WELL_BODY_IDX) is containers[1]
    assert well_cell_property_container(model.physics) is containers[1]


def test_out_of_range_op_num_raises_naming_block_region_and_regions():
    model = _two_region_model(op_num=[0, 5, 1, 2, 2])
    with pytest.raises(KeyError) as excinfo:
        reservoir_cell_region(model, 1)
    message = str(excinfo.value)
    assert "Block 1" in message
    assert "op_num=5" in message
    assert "[0, 1]" in message  # the registered regions


def test_region_without_a_property_container_raises():
    """physics.regions advertises region 1 but no container is registered for
    it: fail loudly instead of quietly evaluating region 0's fluid."""
    containers = {0: _PropertyContainer(MW_REGION_0)}
    model = _two_region_model(property_containers=containers, regions=[0, 1])
    with pytest.raises(KeyError) as excinfo:
        property_container_for_block(model, PERFORATED_RES_BLOCK)
    message = str(excinfo.value)
    assert "Block 1" in message
    assert "region 1" in message
    assert "[0, 1]" in message


def test_no_registered_region_at_all_raises():
    model = _two_region_model(property_containers={}, regions=[])
    with pytest.raises(KeyError, match="no registered property container"):
        property_container_for_block(model, WELL_BODY_IDX)


# ------------------------------------------------- (b) the linear IPR hook
def _apply_ipr(model, p_well, z_well, p_res, z_res, pi=1.0):
    engine = model.physics.engine
    engine.X[:] = 0.0
    engine.RHS[:] = 0.0
    engine.jac_vals[:] = 0.0
    engine.X[WELL_BODY_IDX * N_VARS : WELL_BODY_IDX * N_VARS + N_VARS] = [
        p_well,
        z_well,
    ]
    engine.X[PERFORATED_RES_BLOCK * N_VARS : PERFORATED_RES_BLOCK * N_VARS + N_VARS] = [
        p_res,
        z_res,
    ]
    hook = LinearDFMWellIPRHook(
        model,
        [
            LinearDFMWellIPRConnection(
                well_name="W1", perforation_index=0, pi=pi, pi_type=PI_Type.MASS
            )
        ],
    )
    _bind_and_apply(hook, model, DT)
    return hook


def _component_rate_from_rhs(engine):
    """Component molar rate the hook added to the well block (RHS is dt-scaled)."""
    start = WELL_BODY_IDX * N_VARS
    return engine.RHS[start : start + NC] / DT


def test_ipr_resolves_both_sides_from_their_own_block_region():
    model = _two_region_model()
    hook = _apply_ipr(model, p_well=110.0, z_well=0.4, p_res=100.0, z_res=0.7)
    (resolved,) = hook._resolved_connections
    containers = model.physics.property_containers
    assert resolved["well_region"] == 0
    assert resolved["res_region"] == 1
    assert resolved["well_pc"] is containers[0]
    assert resolved["res_pc"] is containers[1]


def test_ipr_well_upstream_uses_the_well_region_fluid():
    """p_well > p_res: the well (region 0, Mw 10/20) sets the fluid."""
    model = _two_region_model()
    _apply_ipr(model, p_well=110.0, z_well=0.4, p_res=100.0, z_res=0.7)

    z = np.array([0.4, 0.6])
    mw_avg = float(np.dot(MW_REGION_0, z))  # 16.0
    expected = (10.0 / mw_avg) * z  # total mass rate 10 kg/day
    assert _component_rate_from_rhs(model.physics.engine) == pytest.approx(
        expected, rel=1e-12
    )


def test_ipr_reservoir_upstream_uses_the_connected_block_region_fluid():
    """p_well < p_res: the perforated cell is in region 1 (Mw 100/200), NOT
    region 0 -- resolving to region 0 would inflate the molar rate ~10x."""
    model = _two_region_model()
    _apply_ipr(model, p_well=90.0, z_well=0.4, p_res=100.0, z_res=0.7)

    z = np.array([0.7, 0.3])
    mw_avg = float(np.dot(MW_REGION_1, z))  # 130.0
    expected = (-10.0 / mw_avg) * z
    assert _component_rate_from_rhs(model.physics.engine) == pytest.approx(
        expected, rel=1e-12
    )

    # the region-0 fallback this test exists to forbid
    wrong = (-10.0 / float(np.dot(MW_REGION_0, z))) * z
    assert not np.allclose(expected, wrong)


def test_ipr_raises_when_the_connected_block_region_has_no_container():
    containers = {0: _PropertyContainer(MW_REGION_0)}
    model = _two_region_model(property_containers=containers, regions=[0, 1])
    with pytest.raises(KeyError) as excinfo:
        _apply_ipr(model, p_well=110.0, z_well=0.4, p_res=100.0, z_res=0.7)
    message = str(excinfo.value)
    assert "Block 1" in message
    assert "region 1" in message


# ------------------------------------------- (c) the lateral-heat PH branch
HEAT_SEGMENTS = 2


def _heat_evaluator():
    geometry = PipeGeometry(
        pipe_name="W1",
        segment_lengths=[25.0] * HEAT_SEGMENTS,
        pipe_ID=0.1,
        inclination_angle=0.0,
        wall_roughness=5e-5,
    )
    return SemiAnalyticalWellLateralHeatTransfer(
        pipe_name="W1",
        pipe_geometry=geometry,
        earth_thermal_props={
            "T": [330.0] * HEAT_SEGMENTS,
            "c": 1000.0,
            "K": 2.0,
            "rho": 2500.0,
        },
        outermost_layer_OD=0.22,
        Ui=50.0,
    )


def _apply_heat_hook(model, enthalpies):
    physics = model.physics
    engine = physics.engine
    engine.X[:] = 0.0
    engine.RHS[:] = 0.0
    x2d = engine.X.reshape(-1, N_VARS)
    x2d[WELL_HEAD_IDX : WELL_HEAD_IDX + HEAT_SEGMENTS, -1] = enthalpies
    hook = SemiAnalyticalWellLateralHeatTransferHook(
        model, model.reservoir.wells[0], _heat_evaluator()
    )
    _bind_and_apply(hook, model, DT, 1.0, with_jacobian=False)
    return hook


def test_lateral_heat_ph_flashes_with_the_well_region_container():
    """regions[0] == 1 here, so the hook must flash with container 1 (which
    reports T = h + 1000), not with the container tagged 0."""
    containers = {
        0: _PropertyContainer(MW_REGION_0),
        1: _PropertyContainer(MW_REGION_1, temperature_offset=1000.0),
    }
    enthalpies = np.array([300.0, 305.0])
    model = _two_region_model(
        property_containers=containers,
        regions=[1, 0],
        state_spec=_StateSpec.PH,
        thermal=True,
    )
    hook = _apply_heat_hook(model, enthalpies)
    assert hook._property_container is containers[1]

    expected_q = _heat_evaluator().evaluate(enthalpies + 1000.0, 1.0 + DT)
    rhs_well = model.physics.engine.RHS.reshape(-1, N_VARS)[
        WELL_HEAD_IDX : WELL_HEAD_IDX + HEAT_SEGMENTS
    ]
    assert np.all(rhs_well[0] == 0.0)  # wellhead control rows untouched
    assert rhs_well[1:, -1] == pytest.approx(-expected_q[1:] * DT, rel=1e-12)

    # what the region-0 fallback would have produced
    wrong_q = _heat_evaluator().evaluate(enthalpies, 1.0 + DT)
    assert not np.allclose(expected_q[1:], wrong_q[1:])


def test_lateral_heat_ph_resolves_once_not_per_newton_iteration():
    model = _two_region_model(state_spec=_StateSpec.PH, thermal=True)
    hook = _apply_heat_hook(model, np.array([300.0, 305.0]))
    resolved = hook._property_container
    assert resolved is model.physics.property_containers[0]
    # a second Newton iteration reuses the bound container
    _bind_and_apply(hook, model, DT, 1.0, with_jacobian=False)
    assert hook._property_container is resolved


def test_lateral_heat_ph_raises_when_the_well_region_has_no_container():
    model = _two_region_model(
        property_containers={},
        regions=[],
        state_spec=_StateSpec.PH,
        thermal=True,
    )
    with pytest.raises(KeyError, match="no registered property container"):
        _apply_heat_hook(model, np.array([300.0, 305.0]))


def test_ipr_well_side_is_resolved_from_the_well_operator_region():
    """The well side is resolved independently too: with regions [1, 0] the
    well fluid is region 1's, while the perforated cell (op_num 1 -> slot 1 ->
    region 0) keeps region 0's -- the exact opposite of the previous
    hard-coded ``property_containers[0]`` for both sides."""
    containers = {
        0: _PropertyContainer(MW_REGION_0),
        1: _PropertyContainer(MW_REGION_1),
    }
    model = _two_region_model(property_containers=containers, regions=[1, 0])
    hook = _apply_ipr(model, p_well=110.0, z_well=0.4, p_res=100.0, z_res=0.7)
    (resolved,) = hook._resolved_connections
    assert resolved["well_region"] == 1
    assert resolved["res_region"] == 0

    z = np.array([0.4, 0.6])
    expected = (10.0 / float(np.dot(MW_REGION_1, z))) * z  # mw_avg 160.0
    assert _component_rate_from_rhs(model.physics.engine) == pytest.approx(
        expected, rel=1e-12
    )
