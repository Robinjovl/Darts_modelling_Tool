"""Newton-Raphson nonlinear solver: specs and runtime driver.

Holds the Newton family of specifications (:class:`NewtonSpec` and the
not-yet-implemented :class:`QuasiNewtonSpec` / :class:`TrustRegionNewtonSpec`)
and the :class:`NewtonSolver` that drives the C++ per-iteration kernels
(assembly, linear solve, dX corrections, residual norms).
"""

from dataclasses import dataclass, field

import numpy as np

from darts.nonlinear_solvers.base import (
    ChopSpec,
    InexactNewtonSpec,
    LineSearchSpec,
    NonlinearSolver,
    NonlinearSolverSpec,
    OBLBoundsSpec,
    write_to_log,
)

# chop mode <-> engine.newton_chop_mode (sim_params::newton_solver_t)
_CHOP_MODE_TO_ENUM = {None: 0, "global": 1, "local": 2}
_ENUM_TO_CHOP_MODE = {0: None, 1: "global", 2: "local"}


# ------------------------------------------------------------------ specs


@dataclass
class NewtonSpec(NonlinearSolverSpec):
    """Newton-Raphson solver with the full analytic (OBL-derivative) Jacobian.

    :ivar chop: chopping strategy of the nonlinear update (:class:`ChopSpec`).
    :ivar line_search: backtracking line search (:class:`LineSearchSpec`).
    :ivar obl_bounds: restraining of the Newton trajectory (:class:`OBLBoundsSpec`).
    :ivar inexact: optional inexact-Newton forcing sequence (not implemented yet).
    """

    chop: ChopSpec = field(default_factory=ChopSpec)
    line_search: LineSearchSpec = field(default_factory=LineSearchSpec)
    obl_bounds: OBLBoundsSpec = field(default_factory=OBLBoundsSpec)
    inexact: InexactNewtonSpec | None = None

    def make_solver(self, model=None) -> "NewtonSolver":
        if self.inexact is not None:
            raise NotImplementedError(
                "Inexact Newton (forcing sequences) is not implemented yet"
            )
        if self.obl_bounds.mode == "physical":
            raise NotImplementedError(
                "Physical per-axis Newton bounds are not implemented yet"
            )
        return NewtonSolver(self, model=model)

    def sync_to_engine(self, engine):
        """Sync the residual norm (base) plus the Newton kernel controls."""
        super().sync_to_engine(engine)
        engine.newton_chop_mode = _CHOP_MODE_TO_ENUM[self.chop.mode]
        engine.newton_chop_factor = self.chop.factor
        engine.log_transform = 1 if self.chop.log_transform else 0


@dataclass
class QuasiNewtonSpec(NewtonSpec):
    """Placeholder: quasi-Newton (lagged/frozen Jacobian) variant.

    :ivar jacobian_lag: number of iterations the Jacobian is reused for
        (1 = full Newton).
    """

    jacobian_lag: int = 1

    def make_solver(self, model):
        raise NotImplementedError("Quasi-Newton is not implemented yet")


@dataclass
class TrustRegionNewtonSpec(NewtonSpec):
    """Placeholder: trust-region Newton tracking inflection points of the flux operators."""

    def make_solver(self, model):
        raise NotImplementedError("Trust-region Newton is not implemented yet")


def default_nonlinear_spec() -> NewtonSpec:
    """Default nonlinear solver spec: local-chop Newton, matching the historic
    sim_params defaults."""
    return NewtonSpec()


def default_nonlinear_solver() -> "NewtonSolver":
    """Default nonlinear solver: a detached local-chop Newton solver, matching
    the historic sim_params defaults."""
    return NewtonSolver(default_nonlinear_spec())


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
        chop (per :class:`ChopSpec`), OBL-axes clamp (per :class:`OBLBoundsSpec`;
        when the spec provides axis bounds they are passed to the C++ kernel,
        otherwise the no-argument kernel is inert while the engine's op_axis
        bounds are unset) and thermal-variable correction (self-guarded by the
        state specification)."""
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
        elif obl.mode is None or obl.mode == "obl_axes":
            steps.append(engine.correct_obl_axes)
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

            if not model.has_dfm_well:
                status.newton_residual = (
                    engine.calc_newton_residual()
                )  # calc norm of residual
            else:
                # Method is either 1 or 2
                status.newton_residual = engine.calc_coupled_well_reservoir_residual(
                    spec.coupled_well_res_norm_method
                )

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

            status.well_residual = engine.calc_well_residual()
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

            # line search
            if (
                spec.line_search.enabled
                and i > 0
                and residual_history[-1][0] > 0.9 * residual_history[-2][0]
            ):
                coef = np.array([0.0, 1.0])
                history = np.array([residual_history[-2], residual_history[-1]])
                residual_history[-1] = self.line_search(
                    dt, t, coef, history, verbose, iter_counter=i
                )
                max_residual[i] = residual_history[-1][0]

                # check stationary point after line search
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
            else:
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

    def line_search(
        self,
        dt: float,
        t: float,
        coef: np.ndarray,
        history: list | np.ndarray,
        verbose: int | None = None,
        iter_counter: int = None,
    ):
        """
        Perform a line search to find the optimal coefficient that minimizes residuals.

        :param dt: Time step for the update process.
        :param t: Current time.
        :param coef: Array of current coefficients used in the line search.
        :param history: Historical residuals, where each entry contains residuals for 'r_mat' and 'r_well'.
        :param verbose: Verbosity level; ``None`` inherits ``model.verbose``.
        :param iter_counter: Newton-Raphson iteration counter for the current time step. Used by DFM well velocity updates.

        :return: Tuple containing the minimum residual achieved, a placeholder value (0.0), and the coefficient
                 corresponding to the minimum residual.
        :rtype: tuple(float, float, float)
        """
        model = self.model
        engine = self.engine
        verbose = model.verbose if verbose is None else verbose
        newton_iter_counter = (
            self.status.n_newton if iter_counter is None else iter_counter
        )

        if verbose:
            print(
                "LS: "
                + str(coef[0])
                + "\t"
                + "r_mat = "
                + str(history[0][0])
                + "\tr_well = "
                + str(history[0][1])
            )
            print(
                "LS: "
                + str(coef[1])
                + "\t"
                + "r_mat = "
                + str(history[1][0])
                + "\tr_well = "
                + str(history[1][1])
            )
        res_history = np.array([history[0][0], history[1][0]])

        for _iter in range(5):
            if coef.size > 2:
                idx_min = res_history.argmin()
                closest_left = np.where(coef < coef[idx_min])[0]
                closest_right = np.where(coef > coef[idx_min])[0]
                if closest_left.size and closest_right.size:
                    left = closest_left[coef[closest_left].argmax()]
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[left] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[left]) / 2)
                    elif res_history[right] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[right]) / 2)
                    else:
                        if res_history[left] < res_history[right]:
                            coef = np.append(
                                coef, coef[idx_min] - (coef[idx_min] - coef[left]) / 4
                            )
                        else:
                            coef = np.append(
                                coef, coef[idx_min] + (coef[right] - coef[idx_min]) / 4
                            )
                elif closest_left.size:
                    left = closest_left[coef[closest_left].argmax()]
                    if res_history[left] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[left]) / 2)
                    else:
                        coef = np.append(
                            coef, coef[idx_min] + (coef[idx_min] - coef[left]) / 2
                        )
                elif closest_right.size:
                    right = closest_right[coef[closest_right].argmin()]
                    if res_history[right] < res_history[idx_min]:
                        coef = np.append(coef, (coef[idx_min] + coef[right]) / 2)
                    else:
                        coef = np.append(
                            coef, coef[idx_min] - (coef[right] - coef[idx_min]) / 2
                        )
                if coef[-1] <= 0:
                    coef[-1] = self.spec.line_search.min_update
                if coef[-1] >= 1:
                    coef[-1] = 1.0 - self.spec.line_search.min_update
            else:
                coef = np.append(coef, coef[-1] / 2)

            engine.newton_update_coefficient = coef[-1] - coef[-2]
            self.update(dt)
            if model.has_dfm_well:
                model.update_dfm_well_vels_and_ders(dt, t, newton_iter_counter)
            engine.assemble_linear_system(dt)
            model.apply_rhs_flux(dt, t)
            if model.has_dfm_well:
                model.apply_dfm_well_lateral_heat_flux(dt, t)
            if model.platform == "gpu":
                from darts.engines import copy_data_to_device

                copy_data_to_device(engine.RHS, engine.get_RHS_d())
            if model.has_dfm_well:
                res = (
                    engine.calc_coupled_well_reservoir_residual(
                        self.spec.coupled_well_res_norm_method
                    ),
                    engine.calc_well_residual(),
                )
            else:
                res = (
                    engine.calc_newton_residual(),
                    engine.calc_well_residual(),
                )
            res_history = np.append(res_history, res[0])
            if verbose:
                print(
                    "LS: "
                    + str(coef[-1])
                    + "\t"
                    + "r_mat = "
                    + str(res[0])
                    + "\tr_well = "
                    + str(res[1])
                )

        final_id = res_history.argmin()
        engine.newton_update_coefficient = coef[final_id] - coef[-1]
        self.update(dt)
        if model.has_dfm_well:
            # The accepted line-search coefficient can differ from the last tested coefficient.
            # Recompute DFM velocities and derivatives so stored well data matches the accepted state.
            model.update_dfm_well_vels_and_ders(dt, t, newton_iter_counter)

        return res_history[final_id], 0.0, coef[final_id]
