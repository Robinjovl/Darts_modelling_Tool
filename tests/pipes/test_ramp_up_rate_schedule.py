"""Unit tests for the rate schedule of :class:`darts.pipes.ramp_up_rate.RampUpRate` (M3.5).

The schedule used to be sampled at the START of the timestep, which made the
first step of every ramped model inject nothing at all; the class compensated
with an assertion demanding a first timestep smaller than 0.01 s ("because
during this time step I set the rate to zero!"). Review item E3 replaced the
sample by the EXACT AVERAGE of the ramp over ``[t_n, t_n + dt]`` -- which is
what the residual asks for, since the source enters it as ``rate * dt`` -- and
deleted the assertion.

These tests pin down:
  * the average against numerical quadrature of the schedule;
  * that stepping through the ramp injects exactly the integral of the
    schedule, for any sequence of timestep sizes;
  * that a first timestep above the old 0.01 s limit is accepted and does not
    start from a zero rate;
  * that a zero ramp-up period still gives the target rate, bit for bit;
  * that ``Pipe`` publishes the timestep in DAYS (the units the ramp-up period
    and the simulation time are in), not in the seconds the momentum equation
    works in.
"""

import numpy as np
import pytest

# darts.pipes.pipe imports darts.engines (value_vector); skip the module on a
# checkout without the compiled extension (CI builds it before pytest runs).
pytest.importorskip("darts.engines")

from darts.pipes.define_pipe_geometry import PipeGeometry  # noqa: E402
from darts.pipes.ramp_up_rate import RampUpRate  # noqa: E402

SECONDS_PER_DAY = 24.0 * 60.0 * 60.0


class _StubPropertyContainer:
    np_fl = 2
    nc_fl = 2
    phases_name = ["G", "L"]
    Mw = np.array([44.01, 18.015])


class _StubPhysics:
    """The attribute surface RampUpRate.__init__ reads for an isothermal inflow."""

    nc = 2
    thermal = False

    def __init__(self):
        self.property_containers = [_StubPropertyContainer()]


def _make_geometry(n_segments=3):
    return PipeGeometry(
        pipe_name="W1",
        segment_lengths=[10.0] * n_segments,
        pipe_ID=0.1,
        inclination_angle=0.0,
        wall_roughness=5e-5,
    )


def _make_source(ramp_up_period, target_molar_rate=100.0, first_ts_size=1e-7):
    geometry = _make_geometry()
    return RampUpRate(
        pipe_name=geometry.pipe_name,
        pipe_geom=geometry,
        physics=_StubPhysics(),
        first_ts_size=first_ts_size,
        segment_idx=0,
        inflow_or_outflow="inflow",
        target_molar_rate=target_molar_rate,
        ramp_up_period=ramp_up_period,
        inj_fluid_props={"composition": np.array([1.0, 0.0])},
    )


def _quadrature(period, t_start, dt, n=200001):
    """Average of min(t/period, 1) over [t_start, t_start + dt], by quadrature."""
    t = np.linspace(t_start, t_start + dt, n)
    return np.trapezoid(np.minimum(t / period, 1.0), t) / dt


# ------------------------------------------------------------ the average
@pytest.mark.parametrize("period", [1e-3, 0.5, 2.0])
@pytest.mark.parametrize(
    ("t_start", "dt"),
    [
        (0.0, 1e-9),  # the first step of a model that keeps a tiny dt_first
        (0.0, 0.25),  # a first step far bigger than the deleted 0.01 s limit
        (0.3, 0.4),  # somewhere inside the ramp
        (0.9, 0.4),  # straddling the end of the ramp
        (3.0, 1.0),  # entirely on the plateau
    ],
)
def test_ramp_factor_is_the_exact_average_over_the_timestep(period, t_start, dt):
    source = _make_source(ramp_up_period=period)
    factor = source.ramp_factor(t_start, dt)
    assert 0.0 <= factor <= 1.0
    assert factor == pytest.approx(_quadrature(period, t_start, dt), rel=1e-9)


def test_ramp_factor_without_a_timestep_samples_the_start_of_the_step():
    """The historical point sample stays available (and is the fallback when no
    timestep has been published), so a caller outside Pipe keeps its behaviour."""
    source = _make_source(ramp_up_period=2.0)
    assert source.timestep_size == 0.0
    for t in (0.0, 0.5, 1.9, 2.0, 5.0):
        source.update_current_molar_rate(t)
        assert source.current_rate == min(t / 2.0, 1.0) * source.target_rate


def test_ramp_factor_is_monotonic_in_time():
    source = _make_source(ramp_up_period=2.0)
    factors = [source.ramp_factor(t, 0.1) for t in np.linspace(0.0, 3.0, 31)]
    assert np.all(np.diff(factors) >= 0.0)
    assert factors[0] > 0.0  # the first step is no longer identically zero
    assert factors[-1] == 1.0


# ------------------------------------------------------- injected amount
def test_stepping_through_the_ramp_injects_the_exact_integral():
    """The physical statement of the fix: the schedule integrates exactly over
    ANY sequence of timesteps, because the residual multiplies the rate by dt.
    The old start-of-step sample under-injects by target*dt/(2*period) a step."""
    period, target = 2.0, 100.0
    source = _make_source(ramp_up_period=period, target_molar_rate=target)
    steps = [1e-7, 0.3, 0.3, 0.5, 0.5, 0.9, 1.0]

    t, injected, injected_old_scheme = 0.0, 0.0, 0.0
    for dt in steps:
        source.timestep_size = dt
        source.update_current_molar_rate(t)
        injected += source.current_rate * dt
        injected_old_scheme += min(t / period, 1.0) * target * dt
        t += dt

    exact = target * (0.5 * period + (t - period))
    assert injected == pytest.approx(exact, rel=1e-12)
    # ... and the scheme it replaces really was biased low.
    assert injected_old_scheme < exact * (1.0 - 1e-3)


def test_the_timestep_can_be_passed_explicitly():
    source = _make_source(ramp_up_period=2.0)
    source.timestep_size = 1.0
    source.update_current_molar_rate(0.0, dt=0.4)
    assert source.current_rate == pytest.approx(0.2 / 2.0 * source.target_rate)


# ------------------------------------------------- the deleted assertion
@pytest.mark.parametrize("first_ts_size", [1e-7, 0.01 / SECONDS_PER_DAY, 0.1, 1.0])
def test_any_positive_first_timestep_is_accepted(first_ts_size):
    """The 'first_ts_size < 0.01 seconds' assertion is gone: it existed only to
    hide the zero rate of the first step."""
    source = _make_source(ramp_up_period=1.0, first_ts_size=float(first_ts_size))
    assert source.first_ts_size == first_ts_size

    source.timestep_size = first_ts_size
    source.update_current_molar_rate(0.0)
    assert source.current_rate > 0.0
    assert source.current_rate <= source.target_rate


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_first_timestep_is_still_rejected(bad):
    with pytest.raises(AssertionError, match="first_ts_size"):
        _make_source(ramp_up_period=1.0, first_ts_size=bad)


# --------------------------------------------------------- degenerate ramp
@pytest.mark.parametrize("dt", [0.0, 1e-7, 1.0])
def test_a_zero_ramp_up_period_gives_the_target_rate_bit_for_bit(dt):
    """Models that switch the ramp off must be untouched by the fix."""
    source = _make_source(ramp_up_period=0.0, target_molar_rate=123.456)
    assert source.current_rate == source.target_rate
    source.timestep_size = dt
    source.update_current_molar_rate(0.0)
    assert source.current_rate == source.target_rate
    source.update_current_molar_rate(17.0)
    assert source.current_rate == source.target_rate


def test_a_negative_start_time_does_not_produce_a_negative_rate():
    source = _make_source(ramp_up_period=1.0)
    assert source.ramp_factor(-1.0, 0.0) == 0.0
    assert source.ramp_factor(-1.0, 0.5) == pytest.approx(0.25)


# ------------------------------------------------------------ Pipe wiring
def test_pipe_publishes_the_timestep_in_days():
    """Pipe converts dt to seconds for the momentum equation; the schedule works
    in days, so what it publishes must be the unconverted value."""
    import inspect

    from darts.pipes.pipe import Pipe

    source = " ".join(inspect.getsource(Pipe.eval_phase_vels).split())
    assert "sink_source.timestep_size = dt_day" in source
    assert source.index("dt_day = dt") < source.index("dt = dt * 24 * 60 * 60"), (
        "the day-valued timestep must be captured before the conversion to seconds"
    )
