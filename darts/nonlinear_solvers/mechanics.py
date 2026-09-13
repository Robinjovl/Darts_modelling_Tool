"""Newton driver for the poromechanics engines (pm_discretizer / mech_discretizer).

The geomechanics engines converge differently from the flow engines, so the
mechanics models historically each carried a *copy* of the whole Newton loop
(``run_timestep_python``). :class:`MechanicsNewtonSolver` collapses those copies
into one runtime subclass of :class:`~darts.nonlinear_solvers.NewtonSolver`,
overriding only what is genuinely mechanics-specific:

- the residual is the **deviatoric per-component** norm from
  ``engine.calc_newton_dev()`` (``dev_p``, ``dev_u`` and an optional third
  component — thermal ``dev_e`` or contact-gap ``dev_g``), and convergence is a
  **per-component** test (each component below ``tolerance``), not the composite
  L2 gate of the flow solver;
- the nonlinear update is the C++ ``apply_newton_update`` composite applied
  directly (no Python correction pipeline, no OBL/chop sync);
- the engine's ``post_newtonloop`` does **not** veto the Python convergence
  verdict (its residual re-check only selects a failure message), so the verdict
  is passed through;
- the historic ``converged = 1`` while ``i < max_newt`` rule is preserved.

Model-specific behaviour (e.g. displaced-fault contact cut-off / slip-area
gating) is added by subclassing and overriding the hooks
:meth:`compute_mech_residual`, :meth:`on_mech_iteration`,
:meth:`check_early_break` and :meth:`finalize_convergence` — not by copying the
loop again.

Dynamic (inertial) runs of ``engine_pm_cpu`` select the time integration of the
momentum inertia term with :func:`configure_time_integration`; the Bathe
composite scheme needs two consecutive engine sub-steps per timestep, which
:meth:`MechanicsNewtonSolver.run_timestep` drives transparently (the
:meth:`on_substep` hook lets a model update time-dependent boundary data at the
sub-step time).
"""

import numpy as np

from darts.nonlinear_solvers.newton import NewtonSolver

#: known schemes for :func:`configure_time_integration`
TIME_INTEGRATION_SCHEMES = (
    "backward_euler",
    "newmark",
    "generalized_alpha",
    "hht",
    "bathe",
)


def configure_time_integration(engine, scheme: str = "backward_euler", **params):
    """Select the time integration of the inertia term of ``engine_pm_cpu``.

    Call it when switching a run to dynamic mode (after ``engine.momentum_inertia``
    is set). It zeroes the engine's velocity/acceleration state
    (``reset_dynamic_state``: the quasi-static state is at rest) and sets:

    - ``'backward_euler'``: legacy 3-point backward Euler (first order, strongly
      dissipative); no parameters.
    - ``'newmark'``: Newmark-beta in displacement form; ``gamma`` (0.5),
      ``beta`` (0.25). ``gamma > 0.5`` with ``beta = (gamma + 0.5)**2 / 4`` gives
      first-order numerical damping.
    - ``'generalized_alpha'``: Chung-Hulbert generalized-alpha from ``rho_inf``
      in [0, 1] (default 0.8; 1 = no dissipation).
    - ``'hht'``: Hilber-Hughes-Taylor from ``alpha`` in [-1/3, 0] (default
      -0.05).
    - ``'bathe'``: Bathe composite scheme (trapezoidal sub-step over
      ``bathe_gamma * dt`` followed by a 3-point backward sub-step), driven by
      :meth:`MechanicsNewtonSolver.run_timestep`; ``bathe_gamma`` (0.5).

    Common option ``kv_damping`` (default 0): Kelvin-Voigt artificial viscosity
    ``q`` adding ``q * K * (u^{n+1} - u^n)`` (eta = q dt) to the momentum balance
    of matrix cells -- the stiffness-proportional damping that removes grid-scale
    ringing behind sharp wave fronts (Day et al. 2005 use q ~ 0.1-0.25).
    """
    from darts.engines import time_integration as TI

    if scheme not in TIME_INTEGRATION_SCHEMES:
        raise ValueError(
            f"unknown time integration scheme {scheme!r}; "
            f"choose from {TIME_INTEGRATION_SCHEMES}"
        )
    known = {"gamma", "beta", "rho_inf", "alpha", "bathe_gamma", "kv_damping"}
    unknown = set(params) - known
    if unknown:
        raise ValueError(f"unknown time integration parameter(s): {sorted(unknown)}")
    engine.reset_dynamic_state()
    if scheme == "backward_euler":
        engine.time_integration = TI.BACKWARD_EULER
    elif scheme == "newmark":
        engine.set_newmark(params.get("gamma", 0.5), params.get("beta", 0.25))
    elif scheme == "generalized_alpha":
        engine.set_generalized_alpha(params.get("rho_inf", 0.8))
    elif scheme == "hht":
        engine.set_hht_alpha(params.get("alpha", -0.05))
    elif scheme == "bathe":
        engine.time_integration = TI.BATHE
        engine.bathe_gamma = params.get("bathe_gamma", 0.5)
        engine.bathe_substep = 1
    engine.kv_damping = params.get("kv_damping", 0.0)


def _uses_bathe(engine) -> bool:
    """True when the engine integrates inertia with the Bathe composite scheme."""
    ti = getattr(engine, "time_integration", None)
    if ti is None or getattr(engine, "momentum_inertia", 0.0) == 0.0:
        return False
    try:
        from darts.engines import time_integration as TI
    except ImportError:  # pragma: no cover - fake engines in unit tests
        return False
    return ti == TI.BATHE


class MechanicsNewtonSolver(NewtonSolver):
    """Common Newton loop for the geomechanics engines (see module docstring)."""

    #: well-block tolerance = ``well_tolerance_coefficient * spec.tolerance``
    #: (hardcoded to the historic ``1e2`` in the model loops; override if needed)
    well_tolerance_coefficient: float = 1e2

    # ------------------------------------------------------------ hooks
    def compute_mech_residual(self):
        """Return ``(dev_p, dev_u, dev_third)`` for the current state.

        ``dev_third`` is the thermal component ``dev_e`` (thermoporoelasticity) or
        ``0.0``. Subclasses override for a different third component (e.g. the
        contact-gap residual ``dev_g``)."""
        engine = self.engine
        res = engine.calc_newton_dev()
        engine.dev_p = res[0]
        engine.dev_u = res[1]
        if getattr(self.model.reservoir, "thermoporoelasticity", False):
            engine.dev_e = res[2]
            return res[0], res[1], res[2]
        return res[0], res[1], 0.0

    def on_mech_iteration(self, i, dev_p, dev_u, dev_third, well_residual):
        """Per-iteration diagnostics (default: print the residual components).
        Cosmetic only — override for a custom format or to silence."""
        cfl = getattr(self.engine, "CFL_max", None)
        msg = (
            f"{i}: rp = {dev_p}\tru = {dev_u}\tr3 = {dev_third}"
            f"\trwell = {well_residual}"
        )
        if cfl is not None:
            msg += f"\tCFL = {cfl}"
        print(msg)

    def check_early_break(self, i) -> bool:
        """Hook run after the convergence test, before the linear solve: return
        ``True`` to fail the timestep early (e.g. a contact-residual cut-off).
        Default: never break early."""
        return False

    def finalize_convergence(self, converged: int) -> int:
        """Hook run after the loop, before ``post_newtonloop`` (e.g. dynamic-mode
        slip-area gating). Default: pass the verdict through unchanged."""
        return converged

    def on_substep(self, substep: int, dt_sub: float, t_sub: float):
        """Hook run before each Bathe sub-step (``substep`` 1 or 2) covering
        ``[t_sub, t_sub + dt_sub]``: a model with time-dependent boundary data
        should update it here for the sub-step end time. Default: nothing
        (boundary data set once per timestep by the driver applies to both
        sub-steps)."""
        return None

    # ------------------------------------------------------------ loop
    def run_timestep(self, dt: float, t: float, verbose: int | None = None) -> bool:
        """Solve one timestep. With the Bathe composite scheme the timestep is
        split into two engine sub-steps (trapezoidal over ``bathe_gamma * dt``,
        then 3-point backward over the rest), each committed by the engine; a
        failed sub-step leaves the engine at the last committed sub-state and the
        driver cuts ``dt`` as usual (after a failed second sub-step the engine
        state is at ``t + bathe_gamma * dt``)."""
        engine = self.engine
        if not _uses_bathe(engine):
            return self._run_single_step(dt, t, verbose)

        gamma = engine.bathe_gamma
        n_newton = n_linear = 0
        t_sub = t
        converged = False
        for substep, dt_sub in ((1, gamma * dt), (2, (1.0 - gamma) * dt)):
            engine.bathe_substep = substep
            self.on_substep(substep, dt_sub, t_sub)
            converged = self._run_single_step(dt_sub, t_sub, verbose)
            # status.n_newton reports the Newton count of the harder sub-step (the per-solve
            # measure that timestep-growth rules compare against), status.n_linear the total;
            # self.stats accumulates both sub-steps as separate solves
            n_newton = max(n_newton, self.status.n_newton)
            n_linear += self.status.n_linear
            if not converged:
                break
            t_sub += dt_sub
        engine.bathe_substep = 1
        self.status.n_newton = n_newton
        self.status.n_linear = n_linear
        return converged

    def _run_single_step(self, dt: float, t: float, verbose: int | None = None) -> bool:
        model = self.model
        engine = self.engine
        spec = self.spec
        assert dt > 0, "Time step size must be a positive value!"
        spec.validate()

        max_newt = spec.max_iterations
        self.status.reset()
        status = self.status
        model._linear_solver_rc_last = 0
        # Push the spec's kernel controls into the engine, exactly as
        # NewtonSolver.run_timestep does. Without this the engine keeps its
        # constructor defaults -- newton_chop_mode = NEWTON_LOCAL_CHOP, whose
        # branch is commented out in engine_super_elastic_cpu -- so the global
        # chop THMCModel asks for (chop.mode='global', factor=0.2) never runs
        # and multi-component mechanics models lose Newton chopping entirely.
        spec.sync_to_engine(engine)
        tol = spec.tolerance
        well_tol = self.well_tolerance_coefficient * tol
        converged = 0
        self.timer.node["simulation"].start()

        for i in range(max_newt + 1):
            engine.assemble_linear_system(dt)
            dev_p, dev_u, dev_third = self.compute_mech_residual()
            status.newton_residual = np.sqrt(dev_u**2 + dev_p**2 + dev_third**2)
            status.well_residual = engine.calc_well_residual()
            status.n_newton = i
            self.on_mech_iteration(i, dev_p, dev_u, dev_third, status.well_residual)

            # per-component convergence test (mechanics), or iteration budget spent
            if (
                dev_p < tol
                and dev_u < tol
                and dev_third < tol
                and status.well_residual < well_tol
            ) or status.n_newton == max_newt:
                if i > 0:  # min_i_newton
                    converged = 1 if i < max_newt else 0
                    break

            if self.check_early_break(i):
                converged = 0
                break

            rc = self._solve_linear()
            status.linear_solver_rc = rc
            if rc != 0:
                # failed linear solve: do NOT apply a stale update; fail the step
                model._linear_solver_rc_last = rc
                converged = 0
                break
            self.timer.node["newton update"].start()
            engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
            if i < max_newt:
                converged = 1

        # dynamic-mode / model-specific post-loop gating, then the engine commit/
        # rollback. The mech post_newtonloop does not veto `converged`.
        converged = self.finalize_convergence(converged)
        converged = engine.post_newtonloop(dt, t, converged)
        self.stats.update(converged, status)
        self.timer.node["simulation"].stop()
        return converged
