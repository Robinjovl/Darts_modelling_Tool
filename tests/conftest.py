"""Shared fixtures for the pure-Python nonlinear-solver unit tests.

The nonlinear driver (``darts.nonlinear_solvers``) only lazily imports the
compiled ``darts.engines`` (for the residual-norm/chop enum constants synced in
``sync_to_engine``), so the Newton loop can be exercised against a Python
``FakeEngine`` that records the driver's calls and returns scripted residuals /
linear-solve return codes -- no compiled simulator needed for the control flow.
"""

from collections import defaultdict

import pytest

from darts.models.conditions import ConditionSet


class _FakeTimerNode:
    def start(self):
        pass

    def stop(self):
        pass


class _FakeTimer:
    def __init__(self):
        self.node = defaultdict(_FakeTimerNode)


class FakeEngine:
    """Recording spy exposing exactly the method surface the Newton driver uses.

    Scripted inputs:
      * ``res_seq``   -- reservoir residuals returned per Newton iteration
      * ``well_seq``  -- well residuals returned per iteration
      * ``solve_rcs`` -- linear-solve return codes returned per solve (0/1/2)
      * ``dev_seq``   -- deviatoric per-component residual tuples for the
                         mechanics driver: ``(dev_p, dev_u)`` or
                         ``(dev_p, dev_u, dev_third)`` per iteration
    """

    def __init__(
        self, res_seq=None, well_seq=None, solve_rcs=None, log=None, dev_seq=None
    ):
        self.res_seq = list(res_seq if res_seq is not None else [1e-3, 1e-9])
        self.well_seq = list(well_seq if well_seq is not None else [0.0, 0.0])
        self.solve_rcs = list(solve_rcs) if solve_rcs is not None else None
        self.log = log if log is not None else []
        self.dev_seq = list(
            dev_seq if dev_seq is not None else [(1e-3, 1e-3), (1e-9, 1e-9)]
        )
        # writable kernel knobs the spec syncs
        self.residual_norm_type = 0
        self.newton_chop_mode = 0
        self.newton_chop_factor = 0.0
        self.log_transform = 0
        self.newton_update_coefficient = 1.0
        self.opt_history_matching = False
        self._res_i = 0
        self._well_i = 0
        self._solve_i = 0
        self._dev_i = 0
        self.RHS = []
        self.n_apply_update = 0

    # --- residuals
    def calc_newton_residual(self):
        v = self.res_seq[min(self._res_i, len(self.res_seq) - 1)]
        self._res_i += 1
        return v

    def calc_coupled_well_reservoir_residual(self, method):
        return self.calc_newton_residual()

    def calc_well_residual(self):
        v = self.well_seq[min(self._well_i, len(self.well_seq) - 1)]
        self._well_i += 1
        return v

    def calc_newton_dev(self):
        """Mechanics deviatoric per-component residual."""
        v = self.dev_seq[min(self._dev_i, len(self.dev_seq) - 1)]
        self._dev_i += 1
        return v

    # --- assembly / corrections / update (record order)
    def assemble_linear_system(self, dt):
        self.log.append("assemble")

    def correct_composition(self):
        self.log.append("correct_composition")

    def correct_chop_local(self):
        self.log.append("correct_chop_local")

    def correct_chop_global(self):
        self.log.append("correct_chop_global")

    def correct_obl_axes(self, *args):
        self.log.append("correct_obl_axes")

    def correct_thermal(self):
        self.log.append("correct_thermal")

    def apply_update(self, dt):
        self.log.append("apply_update")
        self.n_apply_update += 1
        return 0

    def apply_newton_update(self, dt):
        """Mechanics C++ composite update."""
        self.log.append("apply_newton_update")
        self.n_apply_update += 1
        return 0

    # --- linear solve
    def solve_linear_equation(self):
        if self.solve_rcs is None:
            rc = 0
        else:
            rc = self.solve_rcs[min(self._solve_i, len(self.solve_rcs) - 1)]
            self._solve_i += 1
        self.log.append(f"solve_linear:{rc}")
        return rc

    def get_last_linear_iters(self):
        return 3

    def get_last_linear_residual(self):
        return 1e-8

    # --- timestep commit/rollback + logging
    def post_newtonloop(self, dt, t, converged):
        self.log.append(f"post_newtonloop:{int(bool(converged))}")
        return converged

    def print_timestep(self, *args):
        self.log.append("print_timestep")


class _Reservoir:
    thermoporoelasticity = False


class FakeModel:
    """Minimal stand-in for DartsModel exposing what the driver reads, plus the
    backend-neutral ``_solve_linear_equation`` funnel (C++-solver path only).

    Mirrors the post-assembly surface of the real model: the (empty) unified
    ``conditions`` set and the legacy ``rhs_flux_hooks`` list that
    ``DartsModel.apply_rhs_flux`` consults, and the ``after_assembly`` policy hook
    it ends with.
    """

    def __init__(self, engine, n_vars=2):
        self.verbose = 0
        self.has_dfm_well = False
        self.platform = "cpu"
        self.timer = _FakeTimer()
        self._linear_solver_rc_last = 0
        self.time = []
        self.n_newton_iters = []
        self.time_step_size = []
        self.reservoir = _Reservoir()
        # Python-side condition/source surface of DartsModel (nothing registered).
        self.conditions = ConditionSet()
        self.rhs_flux_hooks = []

        class _Physics:
            pass

        self.physics = _Physics()
        self.physics.engine = engine
        self.physics.n_vars = n_vars

        class _DataTS:
            linear_type = None

        self.data_ts = _DataTS()

    def apply_rhs_flux(self, dt, t):
        # Nothing registered -> no contribution to add; DartsModel still ends with
        # the post-assembly hook, so the fake does too.
        self.after_assembly(dt, t)

    def after_assembly(self, dt, t):
        pass

    def _solve_linear_equation(self):
        engine = self.physics.engine
        rc = engine.solve_linear_equation()
        return rc, engine.get_last_linear_iters(), engine.get_last_linear_residual()


@pytest.fixture
def make_newton():
    """Factory: build a bound NewtonSolver over a FakeModel/FakeEngine.

    Returns ``(solver, model, engine, log)``.
    """
    from darts.nonlinear_solvers import NewtonSolver, NewtonSpec

    def _make(spec=None, res_seq=None, well_seq=None, solve_rcs=None, n_vars=2):
        log = []
        engine = FakeEngine(
            res_seq=res_seq, well_seq=well_seq, solve_rcs=solve_rcs, log=log
        )
        model = FakeModel(engine, n_vars=n_vars)
        solver = NewtonSolver(spec if spec is not None else NewtonSpec())
        solver.bind(model)
        return solver, model, engine, log

    return _make


@pytest.fixture
def make_mechanics():
    """Factory: build a bound MechanicsNewtonSolver over a FakeModel/FakeEngine.

    Returns ``(solver, model, engine, log)``.
    """
    from darts.nonlinear_solvers import MechanicsNewtonSolver, NewtonSpec

    def _make(
        spec=None,
        dev_seq=None,
        well_seq=None,
        solve_rcs=None,
        n_vars=2,
        solver_cls=MechanicsNewtonSolver,
    ):
        log = []
        engine = FakeEngine(
            dev_seq=dev_seq, well_seq=well_seq, solve_rcs=solve_rcs, log=log
        )
        model = FakeModel(engine, n_vars=n_vars)
        solver = solver_cls(spec if spec is not None else NewtonSpec())
        solver.bind(model)
        return solver, model, engine, log

    return _make
