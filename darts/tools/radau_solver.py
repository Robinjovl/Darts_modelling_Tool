from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp import radau as scipy_radau


@dataclass(frozen=True)
class RadauStepResult:
    success: bool
    nfev: int = 0
    njev: int = 0
    nlu: int = 0
    message: str = ""


class RadauWithCompositionCorrection(scipy_radau.Radau):
    """
    Radau solver with an optional callback to correct stage states between
    internal Newton iterations.
    """

    def __init__(self, *args, composition_correction=None, **kwargs):
        self._composition_correction = composition_correction
        super().__init__(*args, **kwargs)

    def _solve_collocation_system(
        self, fun, t, y, h, Z0, scale, tol, LU_real, LU_complex
    ):
        """
        Solve Radau collocation system and apply optional state correction after
        each Newton update.
        """
        n = y.shape[0]
        M_real = scipy_radau.MU_REAL / h
        M_complex = scipy_radau.MU_COMPLEX / h

        W = scipy_radau.TI.dot(Z0)
        Z = Z0

        F = np.empty((3, n))
        ch = h * scipy_radau.C

        dW_norm_old = None
        dW = np.empty_like(W)
        converged = False
        rate = None

        for k in range(scipy_radau.NEWTON_MAXITER):
            for i in range(3):
                F[i] = fun(t + ch[i], y + Z[i])

            if not np.all(np.isfinite(F)):
                break

            f_real = F.T.dot(scipy_radau.TI_REAL) - M_real * W[0]
            f_complex = F.T.dot(scipy_radau.TI_COMPLEX) - M_complex * (W[1] + 1j * W[2])

            dW_real = self.solve_lu(LU_real, f_real)
            dW_complex = self.solve_lu(LU_complex, f_complex)

            dW[0] = dW_real
            dW[1] = dW_complex.real
            dW[2] = dW_complex.imag

            W_trial = W + dW
            Z_trial = scipy_radau.T.dot(W_trial)

            if self._composition_correction is not None:
                corrected = self._composition_correction(y, Z_trial)
                if corrected is not None:
                    Z_trial = corrected
                W_trial = scipy_radau.TI.dot(Z_trial)
                dW = W_trial - W

            dW_norm = scipy_radau.norm(dW / scale)
            if dW_norm_old is not None:
                rate = dW_norm / dW_norm_old

            if rate is not None and (
                rate >= 1
                or rate ** (scipy_radau.NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol
            ):
                break

            W = W_trial
            Z = Z_trial

            if dW_norm == 0 or (rate is not None and rate / (1 - rate) * dW_norm < tol):
                converged = True
                break

            dW_norm_old = dW_norm

        return converged, k + 1, Z, rate

    # Adapted from scipy.integrate._ivp.radau.Radau._step_impl to route
    # collocation solves through _solve_collocation_system().
    def _step_impl(self):
        t = self.t
        y = self.y
        f = self.f

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        J = self.J
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Z0 = np.zeros((3, y.shape[0]))
            else:
                Z0 = self.sol(t + h * scipy_radau.C).T - y

            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    LU_real = self.lu(scipy_radau.MU_REAL / h * self.I - J)
                    LU_complex = self.lu(scipy_radau.MU_COMPLEX / h * self.I - J)

                converged, n_iter, Z, rate = self._solve_collocation_system(
                    self.fun,
                    t,
                    y,
                    h,
                    Z0,
                    scale,
                    self.newton_tol,
                    LU_real,
                    LU_complex,
                )

                if not converged:
                    if current_jac:
                        break

                    J = self.jac(t, y, f)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            y_new = y + Z[-1]
            ZE = Z.T.dot(scipy_radau.E) / h
            error = self.solve_lu(LU_real, f + ZE)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = scipy_radau.norm(error / scale)
            safety = (
                0.9
                * (2 * scipy_radau.NEWTON_MAXITER + 1)
                / (2 * scipy_radau.NEWTON_MAXITER + n_iter)
            )

            if rejected and error_norm > 1:
                error = self.solve_lu(LU_real, self.fun(t, y + error) + ZE)
                error_norm = scipy_radau.norm(error / scale)

            if error_norm > 1:
                factor = scipy_radau.predict_factor(
                    h_abs, h_abs_old, error_norm, error_norm_old
                )
                h_abs *= max(scipy_radau.MIN_FACTOR, safety * factor)

                LU_real = None
                LU_complex = None
                rejected = True
            else:
                step_accepted = True

        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = scipy_radau.predict_factor(
            h_abs, h_abs_old, error_norm, error_norm_old
        )
        factor = min(scipy_radau.MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        f_new = self.fun(t_new, y_new)
        if recompute_jac:
            J = jac(t_new, y_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y

        self.t = t_new
        self.y = y_new
        self.f = f_new

        self.Z = Z

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.J = J

        self.t_old = t
        self.sol = self._compute_dense_output()

        return step_accepted, message


def resolve_composition_correction_config(property_container, physics=None, z_var=1):
    """
    Resolve composition-correction bounds and indexing.

    Follows engine workflow:
    min_sim_z = min_axis_z + sim_eps
    max_sim_z = max_axis_z - sim_eps
    """
    if property_container is None:
        return None

    nc = int(getattr(property_container, "nc", 0))
    if nc <= 1:
        return None

    n_solid = int(getattr(property_container, "n_solid", 0))
    n_solid = max(0, min(n_solid, nc - 1))

    min_axis_z = None
    max_axis_z = None
    sim_eps = 0.0
    if physics is not None:
        sim_eps = float(getattr(physics, "sim_eps", 0.0))
        axes_min = np.asarray(getattr(physics, "axes_min", []), dtype=float)
        axes_max = np.asarray(getattr(physics, "axes_max", []), dtype=float)
        n_indep = nc - 1
        if axes_min.size >= z_var + n_indep and axes_max.size >= z_var + n_indep:
            min_axis_z = np.asarray(axes_min[z_var : z_var + n_indep], dtype=float)
            max_axis_z = np.asarray(axes_max[z_var : z_var + n_indep], dtype=float)
        else:
            pt_axes_min = np.asarray(getattr(physics, "PT_axes_min", []), dtype=float)
            pt_axes_max = np.asarray(getattr(physics, "PT_axes_max", []), dtype=float)
            if (
                pt_axes_min.size >= z_var + n_indep
                and pt_axes_max.size >= z_var + n_indep
            ):
                min_axis_z = np.asarray(
                    pt_axes_min[z_var : z_var + n_indep], dtype=float
                )
                max_axis_z = np.asarray(
                    pt_axes_max[z_var : z_var + n_indep], dtype=float
                )

    if min_axis_z is None or max_axis_z is None:
        eps_z = float(getattr(property_container, "eps_z", 1e-12))
        n_indep = nc - 1
        min_axis_z = np.full(n_indep, eps_z, dtype=float)
        max_axis_z = np.full(n_indep, 1.0 - eps_z, dtype=float)
        if sim_eps <= 0:
            sim_eps = eps_z

    min_axis_z = np.asarray(min_axis_z, dtype=float)
    max_axis_z = np.asarray(max_axis_z, dtype=float)
    min_sim_z = min_axis_z + sim_eps
    max_sim_z = max_axis_z - sim_eps

    invalid = max_sim_z <= min_sim_z
    if np.any(invalid):
        eps = np.finfo(float).eps
        min_sim_z[invalid] = min_axis_z[invalid] + eps
        max_sim_z[invalid] = max_axis_z[invalid] - eps
        if np.any(max_sim_z <= min_sim_z):
            return None

    return {
        "nc": nc,
        "n_solid": n_solid,
        "z_var": int(z_var),
        "min_sim_z": min_sim_z,
        "max_sim_z": max_sim_z,
    }


def apply_composition_correction_group(
    state, z_var, c_start, c_end, min_sim_z, max_sim_z
):
    """
    Apply DARTS-style composition correction for one composition group.
    """
    if c_end <= c_start:
        return False

    sum_z = 0.0
    z_corrected = False
    for c in range(c_start, c_end):
        idx = z_var + c
        min_z = min_sim_z[c]
        max_z = max_sim_z[c]
        new_z = state[idx]
        if new_z < min_z:
            new_z = min_z
            z_corrected = True
        elif new_z > max_z:
            new_z = max_z
            z_corrected = True
        state[idx] = new_z
        sum_z += new_z

    last_z = 1.0 - sum_z
    # last component has no explicit axis in state; enforce a conservative lower bound
    # consistent with the active composition group.
    min_last = float(np.min(min_sim_z[c_start:c_end]))
    max_last = float(np.max(max_sim_z[c_start:c_end]))
    if last_z < min_last:
        last_z = sum_z * min_last if sum_z > max_last else min_last
        z_corrected = True
    sum_z += last_z

    if z_corrected:
        state[z_var + c_start : z_var + c_end] /= sum_z

    return z_corrected


def apply_composition_correction_state(state, config):
    """
    Apply DARTS composition correction to one state vector.
    """
    nc = config["nc"]
    n_solid = config["n_solid"]
    z_var = config["z_var"]
    min_sim_z = config["min_sim_z"]
    max_sim_z = config["max_sim_z"]

    solid_corr = int(
        apply_composition_correction_group(
            state, z_var, 0, n_solid, min_sim_z, max_sim_z
        )
    )
    fluid_corr = int(
        apply_composition_correction_group(
            state, z_var, n_solid, nc - 1, min_sim_z, max_sim_z
        )
    )
    return solid_corr, fluid_corr


def make_stage_composition_correction(config, stats):
    """
    Build callback injected into Radau Newton loop.
    """

    def _correction(y_base, Z_trial):
        # Z_trial has shape (3, n_vars): one stage-state increment per Radau node.
        for i in range(Z_trial.shape[0]):
            stage_state = y_base + Z_trial[i]
            solid_corr, fluid_corr = apply_composition_correction_state(
                stage_state, config
            )
            if solid_corr or fluid_corr:
                Z_trial[i] = stage_state - y_base
                stats["solid"] += solid_corr
                stats["fluid"] += fluid_corr
        return Z_trial

    return _correction


def make_default_stage_composition_correction(
    property_container, physics=None, stats=None, z_var=1
):
    """
    Build default correction callback based on DARTS-style composition config.
    """
    if stats is None:
        stats = {"solid": 0, "fluid": 0}
    config = resolve_composition_correction_config(
        property_container=property_container, physics=physics, z_var=z_var
    )
    if config is None:
        return None
    return make_stage_composition_correction(config=config, stats=stats)


def solve_radau_step(
    rhs,
    state_prev,
    dt,
    t0=0.0,
    rtol=1e-8,
    atol=1e-10,
    composition_correction=None,
):
    """
    Integrate a single timestep with SciPy Radau, optionally with stage-state
    composition correction.
    """
    solve_method = "Radau"
    solve_options = {}
    if composition_correction is not None:
        solve_method = RadauWithCompositionCorrection
        solve_options["composition_correction"] = composition_correction

    state_prev_np = np.asarray(state_prev, dtype=float)
    try:
        sol = solve_ivp(
            rhs,
            (t0, t0 + dt),
            state_prev_np,
            method=solve_method,
            t_eval=[t0 + dt],
            rtol=rtol,
            atol=atol,
            **solve_options,
        )
        nlu = int(sol.nlu) if sol.nlu is not None else 0
        njev = int(sol.njev) if sol.njev is not None else 0
        success = bool(sol.success and sol.y.size)
        result = RadauStepResult(
            success=success,
            nfev=int(sol.nfev),
            njev=njev,
            nlu=nlu,
            message=str(sol.message),
        )
        if result.success:
            return result, sol.y[:, -1].copy()
    except Exception as exc:
        result = RadauStepResult(success=False, message=str(exc))
        return result, state_prev_np.copy()

    return result, state_prev_np.copy()
