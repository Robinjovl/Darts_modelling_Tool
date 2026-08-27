"""Focused unit tests for the Python nonlinear-solver refactoring (MR327).

These exercise the pure-Python Newton driver (``darts.nonlinear_solvers``)
against a recording ``FakeEngine`` (see conftest.py) — no full simulation. They
lock in the behaviours fixed in MR327 review: residual hooks, pre/update/post
ordering, line-search accepted status, linear-solver failure return codes and
statistics, OBL-bounds wiring, fallback subclass preservation, spec validation,
DataTS->sim_params propagation, the legacy set_sim_params shim, and NaN/Inf
rejection.

``darts.engines`` is required only for the canonical enum constants that
``NonlinearSolverSpec.sync_to_engine`` reads; the engine's numerical kernels are
never used (FakeEngine stands in for them).
"""

import numpy as np
import pytest

# sync_to_engine reads sim_params enum constants; skip the whole module on a
# checkout without the compiled extension (CI builds it before pytest runs).
pytest.importorskip("darts.engines")

from darts.nonlinear_solvers import (  # noqa: E402
    ChopSpec,
    FallbackSpec,
    NewtonSolver,
    NewtonSpec,
    Norm,
    OBLBoundsSpec,
)


# ----------------------------------------------------------------- F3 hooks
def test_on_iteration_called_once_per_iteration(make_newton):
    calls = []

    class SpySolver(NewtonSolver):
        def on_iteration(self, iteration, dt, t):
            calls.append((iteration, dt, t))

    _, model, engine, _ = make_newton()  # build fake infra
    solver = SpySolver(NewtonSpec())
    solver.bind(model)
    solver.run_timestep(0.5, 10.0)
    # residuals [1e-3, 1e-9]: i=0 (no converge), i=1 (converge) -> 2 calls
    assert [c[0] for c in calls] == [0, 1]
    assert calls[0][1] == 0.5 and calls[0][2] == 10.0


def test_compute_residual_hooks_are_used(make_newton):
    """The loop must route residuals through the overridable hooks."""
    seen = {"res": 0, "well": 0}

    class SpySolver(NewtonSolver):
        def compute_reservoir_residual(self):
            seen["res"] += 1
            return super().compute_reservoir_residual()

        def compute_well_residual(self):
            seen["well"] += 1
            return super().compute_well_residual()

    _, model, engine, _ = make_newton()
    solver = SpySolver(NewtonSpec())
    solver.bind(model)
    solver.run_timestep(1.0, 0.0)
    assert seen["res"] >= 1 and seen["well"] >= 1


# --------------------------------------------------- F6 pre/update/post order
def test_pre_update_post_ordering(make_newton):
    spec = NewtonSpec()
    solver, model, engine, elog = make_newton(spec=spec, res_seq=[1.0, 1e-12])
    # share the engine's event log so pre/post routines interleave with kernels
    spec.pre_routines = [lambda s, dt, t, i: elog.append("pre")]
    spec.post_routines = [lambda s, dt, t, i: elog.append("post")]
    solver.run_timestep(1.0, 0.0)
    # one full update cycle at i=0, then i=1 converges & breaks before update
    order = [
        e for e in elog if e in ("pre", "correct_composition", "apply_update", "post")
    ]
    assert order == ["pre", "correct_composition", "apply_update", "post"]
    # default chop mode is 'local'; obl mode None => NO correct_obl_axes step (F2)
    assert "correct_chop_local" in elog
    assert "correct_obl_axes" not in elog


# ---------------------------------------------------- F5 linear-solve results
def test_linear_success_counts_n_linear(make_newton):
    solver, model, engine, _ = make_newton(res_seq=[1.0, 1e-12], solve_rcs=[0])
    solver.run_timestep(1.0, 0.0)
    assert solver.status.n_linear == engine.get_last_linear_iters()  # one solve


@pytest.mark.parametrize("rc,reason", [(1, "setup"), (2, "solve")])
def test_linear_failure_aborts_and_marks_wasted(make_newton, rc, reason):
    solver, model, engine, _ = make_newton(res_seq=[1.0, 1.0], solve_rcs=[rc])
    converged = solver.run_timestep(1.0, 0.0)
    assert converged is False
    assert solver.status.linear_solver_rc == rc
    assert model._linear_solver_rc_last == rc
    assert engine.n_apply_update == 0  # never applied a stale update
    assert reason in solver._failure_message(1.0)
    assert solver.stats.n_timesteps_wasted == 1


# -------------------------------------------------------------- F2 OBL modes
def test_build_corrections_obl_modes(make_newton):
    # mode=None (default): no OBL step at all
    solver, model, engine, _ = make_newton(spec=NewtonSpec())
    names = [getattr(s, "__name__", "partial") for s in solver.build_corrections()]
    assert "correct_obl_axes" not in names

    # mode='obl_axes' without explicit bounds: the no-arg kernel is appended
    solver2, *_ = make_newton(
        spec=NewtonSpec(obl_bounds=OBLBoundsSpec(mode="obl_axes"))
    )
    names2 = [getattr(s, "__name__", "partial") for s in solver2.build_corrections()]
    assert "correct_obl_axes" in names2


def test_obl_axis_length_mismatch_raises(make_newton):
    solver, model, engine, _ = make_newton(
        spec=NewtonSpec(
            obl_bounds=OBLBoundsSpec(mode="obl_axes", axis_min=[0.0])  # n_vars=2
        ),
        n_vars=2,
    )
    with pytest.raises(ValueError):
        solver.build_corrections()


# ------------------------------------------------------- F7 fallback subclass
def test_fallback_preserves_runtime_subclass(make_newton):
    class MySolver(NewtonSolver):
        pass

    _, model, engine, _ = make_newton()
    solver = MySolver(NewtonSpec())
    solver.bind(model)
    # same-spec retry (solver=None) must reconstruct MySolver, not plain NewtonSolver
    same = solver._make_fallback_solver(FallbackSpec())
    assert type(same) is MySolver
    # explicit alternative spec builds that spec's default runtime class
    other = solver._make_fallback_solver(FallbackSpec(solver=NewtonSpec()))
    assert type(other) is NewtonSolver


def test_fallback_solve_timestep_succeeds(make_newton):
    spec = NewtonSpec(max_iterations=1, fallbacks=[FallbackSpec()])
    # primary consumes [1.0, 1.0] (fails at max_iter); fallback consumes
    # [1.0, 1e-12] (converges). Shared FakeEngine residual cursor advances.
    solver, model, engine, _ = make_newton(
        spec=spec, res_seq=[1.0, 1.0, 1.0, 1e-12], solve_rcs=[0, 0, 0, 0]
    )
    converged = solver.solve_timestep(1.0, 0.0)
    assert converged is True
    assert solver.stats.n_timesteps_wasted >= 1  # the primary attempt was wasted


# ------------------------------------------------------------ F9 NaN/Inf + serialization
@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_nonfinite_residual_rejected(make_newton, bad):
    solver, model, engine, _ = make_newton()
    solver.status.newton_residual = bad
    solver.status.well_residual = 0.0
    assert solver.converged() is False
    assert "not finite" in solver._failure_message(1.0)


def test_spec_to_dict_roundtrips_nested():
    spec = NewtonSpec(tolerance=1e-4, chop=ChopSpec(mode="global", factor=0.3))
    d = spec.to_dict()
    assert d["tolerance"] == 1e-4
    assert d["chop"]["mode"] == "global" and d["chop"]["factor"] == 0.3
    assert not hasattr(spec, "model_dump")  # stdlib dataclass, no pydantic today


# ------------------------------------------------------------- F12 validation
def test_spec_validation_errors():
    with pytest.raises(ValueError):
        ChopSpec(mode="bogus")
    with pytest.raises(ValueError):
        ChopSpec(factor=0.0)
    with pytest.raises(ValueError):
        NewtonSpec(tolerance=-1.0)
    with pytest.raises(ValueError):
        NewtonSpec(max_iterations=0)
    with pytest.raises(ValueError):
        NewtonSpec(coupled_well_res_norm_method=3)
    with pytest.raises(TypeError):
        NewtonSpec(norm="L2")  # must be a Norm, not a str
    with pytest.raises(ValueError):
        OBLBoundsSpec(mode="obl_axes", axis_min=[1.0, 5.0], axis_max=[2.0, 3.0])
    with pytest.raises(ValueError):
        OBLBoundsSpec(mode="bogus")


def test_validate_catches_post_construction_mutation():
    spec = NewtonSpec()
    spec.chop.factor = -1.0  # bypasses __post_init__
    with pytest.raises(ValueError):
        spec.validate()


# ---------------------------------------------------------- F13 enum sourcing
def test_sync_to_engine_uses_canonical_enum_ints(make_newton):
    from darts.engines import sim_params

    solver, model, engine, _ = make_newton(
        spec=NewtonSpec(norm=Norm.LINF, chop=ChopSpec(mode="global", factor=0.2))
    )
    solver.spec.sync_to_engine(engine)
    assert engine.residual_norm_type == int(sim_params.LINF)
    assert engine.newton_chop_mode == int(sim_params.newton_global_chop)


# ------------------------------------------- F1 linear-spec -> sim_params (!280)
def test_linear_spec_settings_reach_params():
    """After !280 the linear settings are owned by ``linear_solver.spec`` (they
    were transitional ``data_ts.linear_*`` attributes before) and are mirrored
    into sim_params by ``LinearSolver._sync_solver_to_sim_params``."""
    import types

    from darts.engines import sim_params
    from darts.linear_solvers import GMRESSolverSpec, LinearSolver

    # a minimal stand-in model: the sync path only reads model.params and
    # model.linear_solver_from_engine_factory (via getattr)
    m = types.SimpleNamespace(params=sim_params())
    m.linear_solver = LinearSolver(
        GMRESSolverSpec(tolerance=7.5e-9, max_iterations=777), model=m
    )
    m.linear_solver._sync_solver_to_sim_params()
    assert m.params.tolerance_linear == 7.5e-9
    assert m.params.max_i_linear == 777


# --------------------------------------------------------- F10 legacy shim
def test_set_sim_params_legacy_kwargs_map_and_warn():
    import types

    from darts.linear_solvers import LinearSolver

    m = types.SimpleNamespace(nonlinear_solver=NewtonSolver(NewtonSpec()))
    ls = LinearSolver(model=m)
    with pytest.warns(DeprecationWarning):
        ls._migrate_legacy_solver_kwargs(
            {
                "tol_newton": 1e-4,
                "it_newton": 7,
                "newton_type": 1,  # legacy int -> 'global'
                "newton_params": [0.25],
                "coupled_well_res_norm_method": 2,
            },
        )
    s = m.nonlinear_solver.spec
    assert s.tolerance == 1e-4
    assert s.max_iterations == 7
    assert s.chop.mode == "global"
    assert s.chop.factor == 0.25
    assert s.coupled_well_res_norm_method == 2
    # a genuine typo still fails loudly
    with pytest.raises(TypeError):
        ls._migrate_legacy_solver_kwargs({"bogus": 1})


# --------------------------------------------- MechanicsNewtonSolver (F4 Tier2)
def test_mechanics_per_component_convergence(make_mechanics):
    # default tol=1e-3; i=0 not converged (does an update), i=1 all components
    # below tol and well 0 < well_mult*tol -> converge
    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1.0, 1.0), (1e-9, 1e-9)], well_seq=[0.0, 0.0]
    )
    converged = solver.run_timestep(1.0, 0.0)
    assert converged  # post_newtonloop passes the verdict through
    assert engine.n_apply_update == 1  # one update at i=0, then i=1 converges
    assert solver.status.n_newton == 1
    assert "apply_newton_update" in engine.log  # C++ composite, not the Python update


def test_mechanics_break_before_update_on_failed_solve(make_mechanics):
    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1.0, 1.0)], well_seq=[0.0], solve_rcs=[1]
    )
    converged = solver.run_timestep(1.0, 0.0)
    assert not converged
    assert solver.status.linear_solver_rc == 1
    assert model._linear_solver_rc_last == 1
    assert engine.n_apply_update == 0  # no stale update after a failed solve


def test_mechanics_early_break_and_finalize_hooks(make_mechanics):
    from darts.nonlinear_solvers import MechanicsNewtonSolver

    events = []

    class FaultLike(MechanicsNewtonSolver):
        def check_early_break(self, i):
            events.append("early")
            return True  # fail the timestep before the linear solve

        def finalize_convergence(self, converged):
            events.append("finalize")
            return 0  # veto

    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1.0, 1.0)], well_seq=[0.0], solver_cls=FaultLike
    )
    converged = solver.run_timestep(1.0, 0.0)
    assert not converged
    assert engine.n_apply_update == 0  # broke before the update
    assert "early" in events and "finalize" in events
    assert "solve_linear:0" not in engine.log  # never reached the solve


def test_mechanics_thermal_third_component(make_mechanics):
    model_thermo = True
    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1e-9, 1e-9, 1.0), (1e-9, 1e-9, 1e-9)], well_seq=[0.0, 0.0]
    )
    model.reservoir.thermoporoelasticity = model_thermo
    solver.run_timestep(1.0, 0.0)
    # i=0: third=1.0 >= tol -> NOT converged (does an update); i=1: third tiny -> converge
    assert engine.n_apply_update == 1
    assert engine.dev_e == 1e-9  # thermal component recorded on the engine
