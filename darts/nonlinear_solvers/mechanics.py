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
"""

import numpy as np

from darts.nonlinear_solvers.newton import NewtonSolver


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

    # ------------------------------------------------------------ loop
    def run_timestep(self, dt: float, t: float, verbose: int | None = None) -> bool:
        model = self.model
        engine = self.engine
        spec = self.spec
        assert dt > 0, "Time step size must be a positive value!"
        spec.validate()

        max_newt = spec.max_iterations
        self.status.reset()
        status = self.status
        model._linear_solver_rc_last = 0
        tol = spec.tolerance
        well_tol = self.well_tolerance_coefficient * tol
        converged = 0
        self.timer.node["simulation"].start()

        for i in range(max_newt + 1):
            engine.assemble_linear_system(dt)
            # Python-side sources run right after assembly, matching the flow
            # Newton loop. The pm/mech engines rescale equation rows inside
            # assembly, so DartsModel.apply_rhs_flux raises (instead of
            # mis-scaling silently) when any condition item or observer is
            # registered on a mechanics model.
            model.apply_rhs_flux(dt, t)
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
