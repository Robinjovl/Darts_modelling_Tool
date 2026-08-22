"""Unit tests for ``darts.pipes.add_lateral_heat_exchange`` (M1b).

Cover the semi-analytical wellbore-to-earth heat-exchange evaluator
(:class:`SemiAnalyticalWellLateralHeatTransfer`) -- constructor validation,
the three formation time functions (Ramey with its early-time clamp+warning,
Chiu&Thakur against a hand-computed value, Zhang's two ``t_d`` branches),
the analytic ``conductance(t)`` against a finite difference of ``evaluate`` --
and the Newton hook (:class:`SemiAnalyticalWellLateralHeatTransferHook`)
against pure-numpy engine/physics stubs: wellhead (segment-0) rows must stay
untouched, body-segment energy rows receive ``-q * dt``, and on the PT path
the analytic diagonal ``+conductance * dt`` lands on the (energy, T) entry of
the body-segment diagonal Jacobian blocks (verified through a real
``BlockCSRView`` over a synthetic block-diagonal CSR).
"""

import enum
import warnings

import numpy as np
import pytest

from darts.models.conditions import AssemblyContext, BlockCSRView
from darts.pipes.add_lateral_heat_exchange import (
    SemiAnalyticalWellLateralHeatTransfer,
    SemiAnalyticalWellLateralHeatTransferHook,
)
from darts.pipes.define_pipe_geometry import PipeGeometry

N_SEGMENTS = 4
SECONDS_PER_DAY = 24 * 60 * 60


def _make_geometry(name="W1"):
    return PipeGeometry(
        pipe_name=name,
        segment_lengths=[25.0] * N_SEGMENTS,
        pipe_ID=0.1,
        inclination_angle=0.0,
        wall_roughness=5e-5,
    )


def _earth_props(T_earth=None):
    return {
        "T": list(T_earth) if T_earth is not None else [330.0] * N_SEGMENTS,
        "c": 1000.0,  # J/kg/K
        "K": 2.0,  # W/m/K
        "rho": 2500.0,  # kg/m3
    }


def _make_evaluator(
    time_function_name="Chiu&Thakur",
    Ui=50.0,
    perforated_segments=None,
    T_earth=None,
    outermost_layer_OD=0.22,
):
    geometry = _make_geometry()
    return SemiAnalyticalWellLateralHeatTransfer(
        pipe_name=geometry.pipe_name,
        pipe_geometry=geometry,
        earth_thermal_props=_earth_props(T_earth),
        outermost_layer_OD=outermost_layer_OD,
        Ui=Ui,
        perforated_segments=perforated_segments,
        time_function_name=time_function_name,
    )


# ------------------------------------------------ (a) constructor validation
def test_both_ui_and_well_layers_props_raise():
    geometry = _make_geometry()
    with pytest.raises(ValueError, match="exactly one"):
        SemiAnalyticalWellLateralHeatTransfer(
            pipe_name=geometry.pipe_name,
            pipe_geometry=geometry,
            earth_thermal_props=_earth_props(),
            outermost_layer_OD=0.22,
            Ui=50.0,
            well_layers_props={"layers": []},
        )


def test_neither_ui_nor_well_layers_props_raises():
    geometry = _make_geometry()
    with pytest.raises(ValueError, match="exactly one"):
        SemiAnalyticalWellLateralHeatTransfer(
            pipe_name=geometry.pipe_name,
            pipe_geometry=geometry,
            earth_thermal_props=_earth_props(),
            outermost_layer_OD=0.22,
        )


def test_well_layers_props_not_implemented():
    geometry = _make_geometry()
    with pytest.raises(NotImplementedError, match="well_layers_props"):
        SemiAnalyticalWellLateralHeatTransfer(
            pipe_name=geometry.pipe_name,
            pipe_geometry=geometry,
            earth_thermal_props=_earth_props(),
            outermost_layer_OD=0.22,
            well_layers_props={"layers": []},
        )


# ----------------------------------------- (b) Chiu&Thakur hand-computed q
def test_chiu_thakur_evaluate_matches_hand_computed_rate():
    Ui = 50.0
    K = 2.0
    perforated = [2]
    ev = _make_evaluator(Ui=Ui, perforated_segments=perforated)
    T_segments = np.array([300.0, 305.0, 310.0, 320.0])
    t_days = 30.0

    q = ev.evaluate(T_segments, t_days)

    # Hand computation, Ramey's fluid-to-earth form with the C&T time function
    alpha = K / (2500.0 * 1000.0)
    r_h = 0.22 / 2.0
    r_i = 0.05  # pipe_IR of the 0.1 m ID pipe
    L = 25.0
    f_t = 0.982 * np.log(1.0 + 1.81 * np.sqrt(alpha * t_days * SECONDS_PER_DAY) / r_h)
    expected_w = (
        2.0 * np.pi * K * L * (330.0 - T_segments) / (f_t + K / (r_i * Ui))
    )  # J/s
    expected = expected_w * SECONDS_PER_DAY / 1000.0  # kJ/day
    expected[perforated] = 0.0

    assert q == pytest.approx(expected, rel=1e-12)
    # heating from a warmer earth: positive rate everywhere but the perforation
    assert np.all(q[[0, 1, 3]] > 0.0)
    assert q[2] == 0.0


# -------------------------------------------------- (c) Ramey early-time clamp
def test_ramey_early_time_warns_once_and_stays_finite():
    ev = _make_evaluator(time_function_name="Ramey")
    T_segments = np.full(N_SEGMENTS, 300.0)  # T_earth = 330 > T -> q > 0
    t_early = 1e-6  # days; deep inside the invalid range of the asymptote

    with pytest.warns(UserWarning, match="Ramey"):
        q = ev.evaluate(T_segments, t_early)
    assert np.all(np.isfinite(q))
    assert np.all(q > 0.0)  # clamp keeps the correct heating sign

    # the warning is a one-time latch: a second early-time call stays silent
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        q2 = ev.evaluate(T_segments, t_early)
    assert not [w for w in record if "Ramey" in str(w.message)]
    assert q2 == pytest.approx(q)


# --------------------------------------------------- (d) Zhang t_d branches
def _zhang_reference(t_d):
    gamma = 0.57722
    if t_d < 2.8:
        beta = (np.pi * t_d) ** -0.5 + 0.5 - 0.25 * np.sqrt(t_d / np.pi) + 0.125 * t_d
    else:
        log_term = np.log(4.0 * t_d) - 2.0 * gamma
        beta = 2.0 * (1.0 / log_term - gamma / log_term**2)
    return 1.0 / beta


@pytest.mark.parametrize("t_d", [2.7, 2.9])
def test_zhang_time_function_branches(t_d):
    ev = _make_evaluator(time_function_name="Zhang")
    alpha = 2.0 / (2500.0 * 1000.0)
    r_h = 0.22 / 2.0
    t_seconds = t_d * r_h**2 / alpha

    f_t = ev._time_function(t_seconds)

    assert f_t == pytest.approx(np.full(N_SEGMENTS, _zhang_reference(t_d)), rel=1e-12)


def test_unknown_time_function_raises():
    ev = _make_evaluator(time_function_name="NotATimeFunction")
    with pytest.raises(TypeError, match="NotATimeFunction"):
        ev.evaluate(np.full(N_SEGMENTS, 300.0), 1.0)


# --------------------------------- (e) conductance == -dq/dT_fluid (analytic)
@pytest.mark.parametrize("time_function_name", ["Chiu&Thakur", "Zhang"])
def test_conductance_matches_finite_difference_of_evaluate(time_function_name):
    perforated = [1]
    ev = _make_evaluator(
        time_function_name=time_function_name, perforated_segments=perforated
    )
    T_segments = np.array([300.0, 305.0, 310.0, 320.0])
    t_days = 12.0
    h = 1e-4  # evaluate() is linear in T, so the central FD is exact

    conductance = ev.conductance(t_days)
    q_plus = ev.evaluate(T_segments + h, t_days)
    q_minus = ev.evaluate(T_segments - h, t_days)
    fd = -(q_plus - q_minus) / (2.0 * h)

    assert conductance == pytest.approx(fd, rel=1e-9)
    assert conductance[perforated] == pytest.approx(0.0)
    assert np.all(conductance[[0, 2, 3]] > 0.0)


# --------------------------------------------------------- (f) hook + stubs
class _StateSpec(enum.Enum):
    PT = 1
    PH = 2


class _StubEngine:
    """X/RHS numpy views plus (optionally) a synthetic block-diagonal CSR."""

    def __init__(self, n_blocks, n_vars, with_jacobian=False):
        self.X = np.zeros(n_blocks * n_vars)
        self.RHS = np.zeros(n_blocks * n_vars)
        if with_jacobian:
            # block-diagonal pattern: one dense n_vars x n_vars block per row
            self.jac_rows = np.arange(n_blocks + 1, dtype=np.int64)
            self.jac_cols = np.arange(n_blocks, dtype=np.int64)
            self.jac_diags = np.arange(n_blocks, dtype=np.int64)
            self.jac_vals = np.zeros(n_blocks * n_vars * n_vars)


class _StubPHPropertyContainer:
    """PH flash stand-in: 'temperature' is just the last state entry."""

    temperature = 0.0

    def evaluate(self, state):
        self.temperature = state[-1]


class _StubPhysics:
    StateSpecification = _StateSpec

    def __init__(self, state_spec, n_vars, engine):
        self.state_spec = state_spec
        self.n_vars = n_vars
        self.engine = engine
        self.property_containers = [_StubPHPropertyContainer()]


class _StubModel:
    def __init__(self, physics):
        self.physics = physics


class _StubWell:
    def __init__(self, well_head_idx, num_segments):
        self.well_head_idx = well_head_idx
        self.num_segments = num_segments


def _fill_temperatures(engine, well, n_vars, temperatures):
    x2d = engine.X.reshape(-1, n_vars)
    x2d[well.well_head_idx : well.well_head_idx + well.num_segments, -1] = temperatures


def _bind_and_apply(hook, model, dt, t, with_jacobian=False):
    """Drive the item the way ``ConditionSet`` does: bind once, then apply with
    an :class:`AssemblyContext` over the stub engine's arrays."""
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
        n_res_blocks=0,  # the stub has no reservoir; the item never reads it
    )
    hook.apply(ctx)
    return ctx


def test_hook_ph_writes_body_energy_rhs_and_skips_wellhead():
    """PH spec: RHS-only hook (provides_jacobian is False); segment 0 carries
    the well-control equations and must never receive the heat source."""
    n_vars = 3  # (p, z, h); the stub flash returns h as the temperature
    well = _StubWell(well_head_idx=2, num_segments=N_SEGMENTS)
    n_blocks = well.well_head_idx + well.num_segments
    engine = _StubEngine(n_blocks, n_vars)
    physics = _StubPhysics(_StateSpec.PH, n_vars, engine)
    model = _StubModel(physics)
    perforated = [3]
    ev = _make_evaluator(perforated_segments=perforated)

    hook = SemiAnalyticalWellLateralHeatTransferHook(model, well, ev)
    assert hook.provides_jacobian is False

    temperatures = np.array([300.0, 305.0, 310.0, 320.0])
    _fill_temperatures(engine, well, n_vars, temperatures)
    dt, t = 2.0, 1.0
    expected_q = _make_evaluator(perforated_segments=perforated).evaluate(
        temperatures, t + dt
    )

    _bind_and_apply(hook, model, dt, t)

    rhs_well = engine.RHS.reshape(-1, n_vars)[
        well.well_head_idx : well.well_head_idx + well.num_segments
    ]
    # wellhead row (segment 0) untouched, all equations
    assert np.all(rhs_well[0] == 0.0)
    # body-segment energy rows got -q * dt; perforated segment stays zero
    assert rhs_well[1:, -1] == pytest.approx(-expected_q[1:] * dt)
    assert rhs_well[3, -1] == 0.0
    # non-energy equations untouched everywhere
    assert np.all(rhs_well[:, :-1] == 0.0)
    # blocks outside the well untouched
    assert np.all(engine.RHS[: well.well_head_idx * n_vars] == 0.0)


def test_hook_pt_adds_conductance_dt_on_diagonal_energy_temperature_entry():
    """PT spec: the hook also writes the analytic Jacobian diagonal
    +conductance * dt at the (energy, T) entry of the body-segment diagonal
    blocks, through a real BlockCSRView over the synthetic CSR."""
    n_vars = 2  # (p, T)
    well = _StubWell(well_head_idx=1, num_segments=N_SEGMENTS)
    n_blocks = well.well_head_idx + well.num_segments
    engine = _StubEngine(n_blocks, n_vars, with_jacobian=True)
    physics = _StubPhysics(_StateSpec.PT, n_vars, engine)
    model = _StubModel(physics)
    perforated = [2]
    ev = _make_evaluator(perforated_segments=perforated)

    hook = SemiAnalyticalWellLateralHeatTransferHook(model, well, ev)
    assert hook.provides_jacobian is True

    temperatures = np.array([300.0, 305.0, 310.0, 320.0])
    _fill_temperatures(engine, well, n_vars, temperatures)
    dt, t = 0.5, 3.0
    twin = _make_evaluator(perforated_segments=perforated)
    expected_q = twin.evaluate(temperatures, t + dt)
    expected_c = twin.conductance(t + dt)

    _bind_and_apply(hook, model, dt, t, with_jacobian=True)

    rhs_well = engine.RHS.reshape(-1, n_vars)[
        well.well_head_idx : well.well_head_idx + well.num_segments
    ]
    assert np.all(rhs_well[0] == 0.0)
    assert rhs_well[1:, -1] == pytest.approx(-expected_q[1:] * dt)

    block_size = n_vars * n_vars
    jac_blocks = engine.jac_vals.reshape(n_blocks, block_size)
    # wellhead diagonal block and non-well blocks untouched
    assert np.all(jac_blocks[: well.well_head_idx + 1] == 0.0)
    # body-segment diagonal blocks: only the (energy=T-row, T-col) entry moved
    energy_temp_entry = (n_vars - 1) * n_vars + (n_vars - 1)
    for segment in range(1, well.num_segments):
        block = jac_blocks[well.well_head_idx + segment]
        assert block[energy_temp_entry] == pytest.approx(expected_c[segment] * dt)
        other = np.delete(block, energy_temp_entry)
        assert np.all(other == 0.0)
    # perforated segment: conductance is zero, so no diagonal contribution
    assert jac_blocks[well.well_head_idx + 2][energy_temp_entry] == 0.0
