"""Newton-Raphson nonlinear solver: spec and runtime driver.

Holds :class:`NewtonSpec` and the :class:`NewtonSolver` that drives the C++
per-iteration kernels (assembly, linear solve, dX corrections, residual norms).
"""

from dataclasses import dataclass, field

import numpy as np

from darts.nonlinear_solvers.base import (
    ChopSpec,
    NonlinearSolver,
    NonlinearSolverSpec,
    OBLBoundsSpec,
    write_to_log,
)

# Python chop mode -> NAME of the canonical C++ enum member
# (sim_params::newton_solver_t). The integer written to engine.newton_chop_mode
# is read from the compiled enum at sync time (single source of truth; robust to
# a future C++ enum reordering).
_CHOP_MODE_TO_CPP_ENUM = {
    None: "newton_std",
    "global": "newton_global_chop",
    "local": "newton_local_chop",
}
# Reverse map (legacy int -> mode) used only by the set_sim_params(newton_type=..)
# deprecation shim in darts_model.py.
_ENUM_TO_CHOP_MODE = {0: None, 1: "global", 2: "local"}


# ------------------------------------------------------------------ specs


@dataclass
class NewtonSpec(NonlinearSolverSpec):
    """Newton-Raphson solver with the full analytic (OBL-derivative) Jacobian.

    :ivar chop: chopping strategy of the nonlinear update (:class:`ChopSpec`).
    :ivar obl_bounds: restraining of the Newton trajectory (:class:`OBLBoundsSpec`).
    """

    chop: ChopSpec = field(default_factory=ChopSpec)
    obl_bounds: OBLBoundsSpec = field(default_factory=OBLBoundsSpec)

    def validate(self):
        """Validate the base convergence fields and re-validate the sub-specs so
        post-construction mutation (e.g. ``spec.chop.factor = -1``) is caught by
        the per-timestep ``validate()`` call, not just at construction."""
        super().validate()
        # sub-spec __post_init__ bodies are pure validators (raise-or-pass)
        self.chop.__post_init__()
        self.obl_bounds.__post_init__()

    def make_solver(self, model=None) -> "NewtonSolver":
        if self.obl_bounds.mode == "physical":
            raise NotImplementedError(
                "Physical per-axis Newton bounds are not implemented yet"
            )
        return NewtonSolver(self, model=model)

    def sync_to_engine(self, engine):
        """Sync the residual norm (base) plus the Newton kernel controls."""
        super().sync_to_engine(engine)
        from darts.engines import sim_params

        # canonical int comes from the compiled enum (single source of truth)
        engine.newton_chop_mode = int(
            getattr(sim_params, _CHOP_MODE_TO_CPP_ENUM[self.chop.mode])
        )
        engine.newton_chop_factor = self.chop.factor
        engine.log_transform = 1 if self.chop.log_transform else 0


# ------------------------------------------------------------------ solver


class NewtonSolver(NonlinearSolver):
    """Newton-Raphson driver: the Python Newton loop over the C++ kernels.

    Constructed from a :class:`NewtonSpec` (or its keyword arguments) and
    detached from the model until :meth:`bind`; the driving Newton loop is
    ported from ``DartsModel.run_timestep`` and is behaviour-identical to the
    historic implementation.

    :param spec: the :class:`NewtonSpec` to run, or ``None`` to build one from
        the keyword arguments.
    :param model: optional model to bind to at construction (usually left
        ``None`` — the model binds the solver during ``init()``).
    """

    def __init__(self, spec: NewtonSpec = None, model=None, **spec_kwargs):
        if spec is None:
            spec = NewtonSpec(**spec_kwargs)
        elif spec_kwargs:
            raise TypeError(
                "NewtonSolver: pass either a NewtonSpec or its keyword "
                "arguments, not both"
            )
        elif not isinstance(spec, NewtonSpec):
            raise TypeError(
                f"NewtonSolver expects a NewtonSpec, got {type(spec).__name__}"
            )
        super().__init__(spec, model=model)

    def build_corrections(self) -> list:
        """Newton dX-correction pipeline prescribed by the spec, mirroring the
        legacy C++ ``apply_newton_update`` composite: composition correction,
        chop (per :class:`ChopSpec`), OBL-axes clamp (per :class:`OBLBoundsSpec`)
        and thermal-variable correction (self-guarded by the state
        specification).

        OBL-axes behaviour by ``obl_bounds.mode``:

        - ``None`` (default): NO OBL step is appended — the correction is
          skipped entirely, so it never consults engine ``op_axis`` state. (The
          legacy no-arg kernel was a no-op on a clean engine anyway, so default
          models are byte-identical; this additionally prevents a prior bounded
          solve — a fallback or live spec switch — from leaking its bounds into
          a later unbounded solve.)
        - ``'obl_axes'`` WITH ``axis_min``/``axis_max``: the bounds are pushed to
          the C++ kernel (which writes the engine's per-region op_axis bounds).
        - ``'obl_axes'`` WITHOUT bounds: the no-arg kernel is appended (reuse
          whatever op_axis bounds the engine already holds)."""
        engine, spec = self.engine, self.spec
        steps = [engine.correct_composition]
        if spec.chop.mode == "global":
            steps.append(engine.correct_chop_global)
        elif spec.chop.mode == "local":
            steps.append(engine.correct_chop_local)
        obl = spec.obl_bounds
        if obl.mode == "obl_axes" and (
            obl.axis_min is not None or obl.axis_max is not None
        ):
            from functools import partial

            from darts.engines import value_vector

            n_vars = self.model.physics.n_vars
            for name in ("axis_min", "axis_max"):
                bound = getattr(obl, name)
                if bound is not None and len(bound) != n_vars:
                    raise ValueError(
                        f"OBLBoundsSpec.{name} has {len(bound)} entries, "
                        f"expected n_vars = {n_vars}"
                    )
            # None entries leave an axis unbounded (+/-inf never clamps)
            inf = float("inf")
            lo = obl.axis_min if obl.axis_min is not None else [None] * n_vars
            hi = obl.axis_max if obl.axis_max is not None else [None] * n_vars
            axis_min = value_vector([-inf if v is None else v for v in lo])
            axis_max = value_vector([inf if v is None else v for v in hi])
            steps.append(partial(engine.correct_obl_axes, axis_min, axis_max))
        elif obl.mode == "obl_axes":
            # 'obl_axes' without explicit bounds: reuse the engine's op_axis box
            steps.append(engine.correct_obl_axes)
        # obl.mode is None -> append nothing (disabled; never touches engine state)
        steps.append(engine.correct_thermal)
        return steps

    def run_timestep(self, dt: float, t: float, verbose: int | None = None) -> bool:
        """
        Solve the Newton loop for the specified timestep.

        :param dt: Timestep size [days]
        :param t: Current time [days]
        :param verbose: Verbosity level; ``None`` inherits ``model.verbose``.
        :return: True if the timestep converged.
        """
        model = self.model
        engine = self.engine
        spec = self.spec
        verbose = model.verbose if verbose is None else verbose
        assert dt > 0, "Time step size must be a positive value!"
        # revalidate the spec (catches post-construction mutation of any field)
        spec.validate()

        max_newt = spec.max_iterations
        max_residual = np.zeros(max_newt + 1)
        self.status.reset()
        status = self.status
        model._linear_solver_rc_last = 0
        spec.sync_to_engine(engine)
        self._corrections = self.build_corrections()
        self.timer.node["simulation"].start()

        residual_history = self.status.residual_history
        for i in range(max_newt + 1):
            # Update well phase velocities and derivatives if DFM wells are used
            if model.has_dfm_well:
                model.update_dfm_well_vels_and_ders(dt, t, i)

            # assemble Jacobian and residual of reservoir and well blocks
            engine.assemble_linear_system(dt)

            # apply RHS flux
            model.apply_rhs_flux(dt, t)

            if model.has_dfm_well:
                model.apply_dfm_well_lateral_heat_flux(dt, t)

            if model.platform == "gpu":
                from darts.engines import copy_data_to_device

                copy_data_to_device(engine.RHS, engine.get_RHS_d())

            # reservoir residual (overridable via compute_reservoir_residual)
            status.newton_residual = self.compute_reservoir_residual()

            max_residual[i] = status.newton_residual
            counter = 0
            for j in range(i):
                denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                if (
                    abs(max_residual[i] - max_residual[j]) / denom
                    < spec.stationary_point_tolerance
                ):
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            # well residual (overridable via compute_well_residual)
            status.well_residual = self.compute_well_residual()
            residual_history.append(
                (
                    status.newton_residual,  # matrix residual
                    status.well_residual,  # well residual
                    1.0,
                )
            )  # Newton update coefficient

            status.n_newton = i
            self.on_iteration(i, dt, t)

            #  check tolerance if it converges
            if (
                status.newton_residual < spec.tolerance
                and status.well_residual
                < spec.tolerance * spec.well_tolerance_multiplier
            ) or status.n_newton == max_newt:
                if i > 0:
                    break

            rc = self._solve_linear()
            if rc != 0:
                # Abort the Newton loop on a failed linear solve without
                # burning the full max_newt budget on stale dX updates.
                status.linear_solver_rc = rc
                model._linear_solver_rc_last = rc
                break
            self.pre_iteration(dt, t, i)
            self.update(dt)
            self.post_iteration(dt, t, i)

        # End of newton loop: the convergence decision is made here and the
        # engine only commits (converged) or rolls back (failed) its state.
        converged = self.converged()
        if not converged:
            write_to_log(self._failure_message(dt))
        converged = engine.post_newtonloop(dt, t, converged)
        self.stats.update(converged, status)
        if converged and not engine.opt_history_matching:
            engine.print_timestep(
                t + dt,
                dt,
                status.n_newton,
                status.n_linear,
                status.newton_residual,
                status.well_residual,
            )

        model.time.append(t)
        model.n_newton_iters.append(status.n_newton)
        model.time_step_size.append(dt)

        self.timer.node["simulation"].stop()
        return converged
