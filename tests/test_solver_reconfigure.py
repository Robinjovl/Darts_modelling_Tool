"""On-the-fly solver reconfiguration tests (SOLVER_DYNAMIC_RECONFIG_PLAN.md).

Guards the two invariants of the dynamic-reconfiguration API:

1. **No Jacobian reallocation** -- neither in-place ``reconfigure`` (hot/warm
   fields) nor a full solver switch (``update_solver(spec=...)`` /
   ``AdaptiveSolverSpec``) may reallocate or rebind the engine Jacobian.
   Verified by pointer identity of the zero-copy ``engine.jac_vals`` view.
2. **Correctness across actions** -- the run keeps converging on the normal
   trajectory after hot updates, warm updates, structural fallbacks and
   policy-driven switches.

Uses the small ``models/2ph_do`` model (runs in seconds).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "2ph_do"


def _jac_ptr(engine):
    return np.asarray(engine.jac_vals).__array_interface__["data"][0]


@pytest.fixture()
def model():
    solvers = pytest.importorskip("darts.solvers")
    if not getattr(solvers, "_have_compiled_solvers", False):
        pytest.skip("open-source solver registry not available in this build")
    cwd = os.getcwd()
    os.chdir(MODEL_DIR)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        from model import Model

        from darts.engines import redirect_darts_output

        redirect_darts_output("test_solver_reconfigure.log")
        m = Model()
        yield m
    finally:
        sys.path.remove(str(MODEL_DIR))
        os.chdir(cwd)


def test_reconfigure_and_switch_without_jacobian_reallocation(model):
    from darts.solvers import (
        CPRSolverSpec,
        GMRESSolverSpec,
        SuperLUSolverSpec,
    )

    m = model
    m.solver = GMRESSolverSpec(restart=40, prec=CPRSolverSpec())
    m.init()
    m.set_output(output_folder="test_solver_reconfigure_out")
    m.run(20, verbose=False)

    ptr0 = _jac_ptr(m.physics.engine)
    assert ptr0 != 0

    # --- hot updates: applied in place ---------------------------------
    assert m.update_solver(restart=25) == "reconfigured"
    assert m.update_solver(tolerance=1e-4, max_iterations=100) == "reconfigured"
    assert m.update_solver(amg_max_iters=2) == "reconfigured"  # prec field
    assert m.params.tolerance_linear == 1e-4
    assert m.params.max_i_linear == 100
    m.run(20, verbose=False)
    assert _jac_ptr(m.physics.engine) == ptr0

    # --- warm update: pressure-AMG profile, hierarchies rebuilt in place
    assert m.update_solver(amg_strong_threshold=0.6) == "reconfigured"
    m.run(20, verbose=False)
    assert _jac_ptr(m.physics.engine) == ptr0

    # --- structural update: stage swap falls back to a rebuild ---------
    assert m.update_solver(stage2_type=0) == "rebuilt"
    m.run(20, verbose=False)
    assert _jac_ptr(m.physics.engine) == ptr0

    # --- full spec switch -----------------------------------------------
    assert m.update_solver(spec=SuperLUSolverSpec()) == "rebuilt"
    m.run(10, verbose=False)
    assert _jac_ptr(m.physics.engine) == ptr0

    # unknown fields are rejected loudly
    with pytest.raises(AttributeError):
        m.update_solver(no_such_field=1)

    # spec stays authoritative for later rebuilds
    m.update_solver(spec=GMRESSolverSpec(restart=33, prec=CPRSolverSpec()))
    assert m._resolve_solver_spec().restart == 33

    assert float(m.physics.engine.t) >= 70.0  # all segments converged


def test_adaptive_policy_actions(model):
    from darts.solvers import (
        AdaptiveSolverSpec,
        CPRSolverSpec,
        GMRESSolverSpec,
        SolverAction,
        SuperLUSolverSpec,
    )

    m = model
    seen = []

    def policy(ctx):
        seen.append(
            (
                ctx.current_index,
                ctx.timestep_converged,
                round(ctx.dt, 6),
                ctx.phase,
                ctx.ls_setup_time >= 0.0,
                ctx.ls_solve_time >= 0.0,
            )
        )
        if len(seen) == 3:
            # switch candidate AND tighten the tolerance in one action
            return SolverAction(index=1, updates={"tolerance": 1e-6})
        return ctx.current_index

    m.solver = AdaptiveSolverSpec(
        candidates=[
            GMRESSolverSpec(restart=40, prec=CPRSolverSpec()),
            SuperLUSolverSpec(),
        ],
        policy=policy,
    )
    m.solver_phase = "static"
    m.init()
    m.set_output(output_folder="test_solver_reconfigure_out2")

    ptr0 = _jac_ptr(m.physics.engine)
    m.run(60, verbose=False)

    assert len(seen) >= 4
    assert seen[0][3] == "static"  # phase tag propagated
    assert m._adaptive_solver_index == 1  # the SolverAction switch happened
    assert m.params.tolerance_linear == 1e-6  # and its update applied
    assert _jac_ptr(m.physics.engine) == ptr0
    assert float(m.physics.engine.t) >= 60.0
