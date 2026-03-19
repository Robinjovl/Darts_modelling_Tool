import os

import h5py
import numpy as np

from darts.engines import index_vector, value_vector
from darts.models.darts_model import DartsModel, DataTS
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.super.physics import Compositional
from darts.tools.radau_solver import (
    make_default_stage_composition_correction,
    solve_radau_step,
)


class ZerodModel(DartsModel):
    """
    ZerodModel implements 0D mass/energy balance using OBL operator values and
    derivatives, with Radau time integration and SciPy nonlinear solve.

    This class assumes the physics/engine are configured by a derived model.
    """

    def __init__(
        self,
        fixed_pressure=False,
        fixed_temperature=False,
        energy_source=None,
    ):
        """
        Initialize the model.
        :param fixed_pressure: flag to fix pressure or not
        :type fixed_pressure: bool
        :param fixed_temperature: flag to fix temperature or not
        :type fixed_temperature: bool
        :param energy_source: energy source, in kJ/day/m3
        :type energy_source: float
        """
        super().__init__()

        self.fixed_pressure = fixed_pressure
        self.fixed_temperature = fixed_temperature
        self.energy_source = energy_source

        self.state = None
        self.state_history = []
        self.time_history = []
        self.property_history = {}

        # Nonlinear solver defaults
        self.nonlin_max_iter = 50
        self.nonlin_tol_res = 1e-10
        self.nonlin_tol_dx = 1e-10

        # Radau defaults
        self.radau_rtol = 1e-8
        self.radau_atol = 1e-10
        self.radau_use_composition_correction = True
        self.radau_last_correction_stats = {"solid": 0, "fluid": 0}

        # HDF5 output defaults (single-cell 0D output)
        self.h5_output_enabled = True
        self.h5_compression_level = 2
        self.output_folder = "output"
        self.sol_filename = "zerod_solution.h5"
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.precision = "d"
        self.compression = "gzip"
        self.precision_map = {"d": np.float64, "s": np.float32}
        self._h5_ready = False

    def init(self, output_folder='output', h5_output_enabled=True):
        """
        Initialize the model state and history.
        :param output_folder: output folder
        :type output_folder: str
        :param h5_output_enabled: flag to enable HDF5 output
        :type h5_output_enabled: bool
        """
        if self.mode == "obl":
            self.physics.init_physics(
                discr_type="tpfa",
                platform="cpu",
                verbose=False,
                itor_mode="adaptive",
                itor_type="multilinear",
                is_barycentric=False,
            )

        self.set_initial_conditions()
        self.state = self.initial_state
        self.state_history = []
        self.time_history = []
        self.property_history = {}
        self.h5_output_enabled = h5_output_enabled
        # HDF5 structure is initialized in set_output.
        if h5_output_enabled:
            self.set_output(output_folder=output_folder)
        self.record_state(0.0, self.state)

    def set_output(
        self,
        output_folder: str = "output",
        sol_filename: str = "zerod_solution.h5",
        precision: str = "d",
        compression: str = "gzip",
    ):
        """
        Configure one-cell HDF5 output.
        """
        if precision not in self.precision_map:
            raise ValueError(
                f"Unsupported precision '{precision}'. Use one of {list(self.precision_map.keys())}."
            )

        self.output_folder = output_folder
        self.sol_filename = sol_filename
        self.sol_filepath = os.path.join(self.output_folder, self.sol_filename)
        self.precision = precision
        self.compression = compression
        self._h5_ready = False

        # HDF5 file initialization is handled centrally here.
        if self.h5_output_enabled:
            n_vars = self._resolve_output_n_vars()
            self._initialize_h5_output_file(n_vars=n_vars, overwrite=True)

        # If states already exist (e.g. set_output() called after init/run),
        # rewrite full history to the new file.
        if self.time_history and self.state_history:
            self.save_data_to_h5()

    def set_sim_params(
        self,
        n_vars: int,
        first_ts: float = None,
        mult_ts: float = None,
        min_ts=1e-15,
        max_ts: float = None,
        runtime: float = 1000,
        tol_newton: float = None,
        it_newton: int = None,
        newton_type=None,
    ):
        """
        Function to set simulation parameters.

        :param n_vars: Number of variables
        :type n_vars: int
        :param first_ts: First timestep
        :type first_ts: float
        :param mult_ts: Timestep multiplier
        :type mult_ts: float
        :param min_ts: Minimum timestep
        :type min_ts: float
        :param max_ts: Maximum timestep
        :type max_ts: float
        :param runtime: Total runtime in days, default is 1000
        :type runtime: float
        :param tol_newton: Tolerance for Newton iterations
        :type tol_newton: float
        :param it_newton: Maximum number of Newton iterations
        :type it_newton: int
        :param newton_type:
        :type newton_type: str
        """
        self.data_ts = DataTS(n_vars)

        # Time stepping parameters. if None, default value will be used
        self.data_ts.dt_first = (
            first_ts if first_ts is not None else self.data_ts.dt_first
        )
        self.data_ts.dt_min = min_ts if min_ts is not None else self.data_ts.dt_min
        self.data_ts.dt_max = max_ts if max_ts is not None else self.data_ts.dt_max
        self.data_ts.dt_mult = mult_ts if mult_ts is not None else self.data_ts.dt_mult

        # Non linear solver parameters. if None, default value will be used
        self.data_ts.newton_max_iter = (
            it_newton if it_newton is not None else self.data_ts.newton_max_iter
        )
        self.data_ts.newton_tol = (
            tol_newton if tol_newton is not None else self.data_ts.newton_tol
        )

        self.params.newton_type = (
            newton_type if newton_type is not None else self.params.newton_type
        )

        self.runtime = runtime

    def set_initial_conditions(self):
        """
        Set initial conditions for the model.

        This method should be implemented in derived classes.
        """
        pass

    def record_state(self, t, state):
        """
        Store the current state and time in history.

        :param t: Simulation time.
        :type t: float
        :param state: State vector to store.
        :type state: array-like
        """
        # Keep state and property snapshots aligned (single source of truth).
        state_np = np.asarray(state, dtype=float).copy()
        props = self.evaluate_output_properties(state_np)
        if not isinstance(props, dict):
            raise TypeError("evaluate_output_properties() must return a dictionary.")

        self.time_history.append(float(t))
        self.state_history.append(state_np)

        # Start a new slot for this timestep for existing properties.
        for name in self.property_history:
            self.property_history[name].append(None)

        # Store current timestep values.
        props_snapshot = {}
        for name, value in props.items():
            arr = self._to_numeric_property_array(value)
            if arr is None:
                continue

            arr = arr.copy()
            props_snapshot[name] = arr
            if name not in self.property_history:
                self.property_history[name] = [None] * (len(self.time_history) - 1)
                self.property_history[name].append(arr)
            else:
                self.property_history[name][-1] = arr

        if self.h5_output_enabled:
            self._append_timestep_to_h5(float(t), state_np, props_snapshot)

    def _format_state_for_log(self, state):
        """
        Format pressure, composition, and thermal state for logging.

        :param state: State vector to format.
        :type state: array-like
        :return: Formatted state string.
        :rtype: str
        """
        props = self.property_container
        num_width = 12
        p_val = state[0]

        nc = props.nc
        if nc == 1:
            zc = np.array([1.0])
        else:
            zc = np.zeros(nc)
            zc[: nc - 1] = state[1:nc]
            zc[-1] = 1.0 - np.sum(zc[: nc - 1])

        comp_names = getattr(props, "components_name", None)
        if comp_names is None or len(comp_names) != nc:
            comp_names = [f"comp{i + 1}" for i in range(nc)]
        z_parts = " ".join(
            f"z_{comp_names[i]}={zc[i]:>{num_width}.6g}" for i in range(nc)
        )

        thermal_val = None
        if state.size > nc:
            thermal_val = state[nc]

        thermal_label = "t"
        if self.mode == "analytical":
            spec = getattr(props, "state_spec", None)
            if spec is not None:
                spec_str = spec.name if hasattr(spec, "name") else str(spec)
                if "ENTHALPY" in spec_str or "PH" in spec_str:
                    thermal_label = "h"
        elif self.mode == "obl":
            if self.physics.state_spec == Compositional.StateSpecification.PH:
                thermal_label = "h"

        temp_val = None
        if self.mode == "analytical":
            try:
                if thermal_label == "t" and thermal_val is not None:
                    temp_val = thermal_val
                elif hasattr(props, "evaluate"):
                    temp_val = getattr(props, "temperature", None)
            except Exception:
                temp_val = None
        elif self.mode == "obl" and self.physics.n_vars > self.physics.nc:
            itor = self.physics.acc_flux_itor[0]
            etor = self.physics.reservoir_operators[0]
            ops_cpp = value_vector(np.zeros(self.physics.n_ops))
            itor.evaluate(value_vector(state), ops_cpp)
            temp_val = ops_cpp[etor.TEMP_OP]

        if thermal_val is not None:
            state_str = (
                f"p={p_val:>{int(num_width / 2)}.6g} {z_parts} "
                f"{thermal_label}={thermal_val:>{num_width}.6g}"
            )
        else:
            state_str = f"p={p_val:>{int(num_width / 2)}.6g} {z_parts}"

        if temp_val is not None:
            state_str += f" temp={temp_val:>{num_width}.6g}"
        return state_str

    def get_obl_source_terms(self, t, state, ops, etor):
        """
        Return OBL right-hand-side source terms dA/dt for current state.

        By default this reads kinetic/source operators (DELTA/KIN). Derived
        models can override this method to add custom forcing terms.
        """
        del state
        source = np.zeros(self.physics.n_vars)
        kin_start = getattr(etor, "KIN_OP", None)
        if kin_start is None:
            ncopy = 0
        else:
            ncopy = min(source.size, max(0, ops.size - kin_start))
            if ncopy > 0:
                source[:ncopy] = -ops[kin_start : kin_start + ncopy]

        # Backward compatibility for OBL models that set self.energy_source
        # but do not populate thermal KIN operator values.
        if (
            self.energy_source is not None
            and source.size > self.physics.nc
            and ncopy > 0
            and np.allclose(source, 0.0)
        ):
            source[self.physics.nc] = self.energy_source.evaluate(t)
        return source

    def _apply_state_constraints(self, n_vars, nc):
        """
        Return fixed-derivative constraints (variable index -> value).

        For robust fixed-state handling we avoid replacing conservation rows.
        Instead, derivatives of constrained variables are prescribed and the
        remaining derivatives are computed from all balance equations.
        """
        constraints = {}
        if self.fixed_pressure and n_vars > 0:
            # state[0] is pressure
            constraints[0] = 0.0

        if self.fixed_temperature and n_vars > nc:
            state_spec = getattr(getattr(self, "physics", None), "state_spec", None)
            if state_spec == PhysicsBase.StateSpecification.PT:
                # thermal unknown index for PT state specification
                constraints[nc] = 0.0

        return constraints

    @staticmethod
    def _solve_with_constraints(dAdx, b, n_vars, constraints):
        """
        Solve dAdx * dxdt = b with optional fixed derivative constraints.

        For constrained cases (e.g., fixed pressure), we prefer exact solves on
        a square, independent subset of balance rows to avoid smoothing errors
        introduced by least-squares on highly stiff chemistry systems.
        """
        if not constraints:
            return np.linalg.solve(dAdx, b)

        fixed_idx = np.array(sorted(constraints.keys()), dtype=np.intp)
        fixed_vals = np.array([constraints[i] for i in fixed_idx], dtype=float)
        free_idx = np.array(
            [i for i in range(n_vars) if i not in constraints], dtype=np.intp
        )

        rhs = np.asarray(b, dtype=float).copy()
        if fixed_idx.size > 0:
            rhs -= dAdx[:, fixed_idx].dot(fixed_vals)

        A_free = dAdx[:, free_idx]
        n_eq, n_free = A_free.shape
        if n_eq == n_free:
            x_free = np.linalg.solve(A_free, rhs)
        elif n_eq == n_free + 1:
            # Drop one row and keep an exact solve on the remaining square system.
            best_x = None
            best_res = np.inf
            for drop_row in range(n_eq):
                row_mask = np.ones(n_eq, dtype=bool)
                row_mask[drop_row] = False
                A_sq = A_free[row_mask, :]
                b_sq = rhs[row_mask]
                try:
                    x_try = np.linalg.solve(A_sq, b_sq)
                except np.linalg.LinAlgError:
                    continue
                # Pick the solution that best matches all rows.
                res_try = np.linalg.norm(A_free.dot(x_try) - rhs, ord=np.inf)
                if res_try < best_res:
                    best_res = res_try
                    best_x = x_try

            if best_x is None:
                x_free, *_ = np.linalg.lstsq(A_free, rhs, rcond=None)
            else:
                x_free = best_x
        elif n_eq > n_free:
            # Generic fallback for cases with more constraints.
            x_free, *_ = np.linalg.lstsq(A_free, rhs, rcond=None)
        else:
            # Underdetermined fallback (should not happen in current 0D setup).
            x_free, *_ = np.linalg.lstsq(A_free, rhs, rcond=None)

        dxdt = np.zeros(n_vars, dtype=float)
        dxdt[free_idx] = x_free
        dxdt[fixed_idx] = fixed_vals
        return dxdt

    def step_radau(
        self,
        state_prev,
        dt,
        t0=0.0,
        rtol=None,
        atol=None,
        composition_correction=None,
    ):
        """
        Advance the state using the Radau integrator.

        :param state_prev: State vector at the beginning of the timestep.
        :type state_prev: array-like
        :param dt: Timestep size.
        :type dt: float
        :param t0: Start time of the timestep.
        :type t0: float
        :param rtol: Relative tolerance for Radau.
        :type rtol: float or None
        :param atol: Absolute tolerance for Radau.
        :type atol: float or None
        :param composition_correction: Apply DARTS composition correction
                                       between Radau nonlinear iterations.
        :type composition_correction: bool or None
        :return: Result of the Radau step.
        :rtype: RadauStepResult
        """
        rtol = self.radau_rtol if rtol is None else rtol
        atol = self.radau_atol if atol is None else atol
        if composition_correction is None:
            composition_correction = self.radau_use_composition_correction
        if composition_correction:
            correction_stats = {"solid": 0, "fluid": 0}
            correction_callback = make_default_stage_composition_correction(
                property_container=getattr(self, "property_container", None),
                physics=getattr(self, "physics", None),
                stats=correction_stats,
                z_var=1,
            )
            self.radau_last_correction_stats = correction_stats
        else:
            correction_callback = None
            self.radau_last_correction_stats = {"solid": 0, "fluid": 0}

        # Radau solver considers system resolved w.r.t. unknowns,
        # thus transition from dA/dt = dA/dX * dX/dt = g(t, p, z, T/h) to dX/dt = f(t, p, z, T/h)
        # is necessary.
        def rhs_analytical(_t, x):
            props = self.property_container
            props.evaluate(x)
            ph = props.ph
            n_vars = props.n_vars
            nc = props.nc
            poro = float(self.poro)
            dens_rock = float(self.dens_rock)
            c_r = float(self.c_r)

            ## accumulation
            dAdx = np.zeros((n_vars, n_vars))

            # mass accumulation: M = phi * z_c * sum(sat * rho_m) [kmol/m3]
            zc = np.append(x[1:nc], 1 - np.sum(x[1:nc]))
            dz_dx = np.zeros((nc, n_vars))
            for i in range(nc - 1):
                dz_dx[i, i + 1] = 1.0
                dz_dx[nc - 1, i + 1] = -1.0
            rho_t = np.sum(props.sat[ph] * props.dens_m[ph])
            drho_t_dx = np.sum(
                props.sat_ders[ph] * props.dens_m[ph, None], axis=0
            ) + np.sum(props.sat[ph, None] * props.dens_m_ders[ph], axis=0)
            dAdx[:nc] = poro * (dz_dx * rho_t + zc * drho_t_dx)

            if n_vars > nc:
                # solid energy accumulation: Es = phi * rho_r * c_r * (T - T0) [kJ/m3]
                dEs_dx = np.zeros(n_vars)
                dEs_dx[0] = (1 - poro) * dens_rock * c_r * props.dTdP
                dEs_dx[1:nc] = (1 - poro) * dens_rock * c_r * props.dTdzk[: nc - 1]
                dEs_dx[nc] = (1 - poro) * dens_rock * c_r * props.dTdX
                dAdx[nc] += dEs_dx

                # fluid energy accumulation: Ef = phi * sum(sat * rho_m * h) [kJ/m3]
                dEf_dx = (
                    np.sum(
                        props.sat_ders[ph]
                        * (props.dens_m[ph] * props.enthalpy[ph])[:, None],
                        axis=0,
                    )
                    + np.sum(
                        (props.sat[ph] * props.enthalpy[ph])[:, None]
                        * props.dens_m_ders[ph],
                        axis=0,
                    )
                    + np.sum(
                        (props.sat[ph] * props.dens_m[ph])[:, None]
                        * props.enthalpy_ders[ph],
                        axis=0,
                    )
                )
                dEf_dx[0] -= 100
                dEf_dx *= poro
                dAdx[nc] += dEf_dx

            ## rate -> right-hand side
            b = np.zeros(n_vars)
            if self.energy_source is not None and n_vars > nc:
                b[nc] = self.energy_source.evaluate(_t)

            try:
                constraints = self._apply_state_constraints(n_vars, nc)
                dxdt = self._solve_with_constraints(dAdx, b, n_vars, constraints)
            except np.linalg.LinAlgError:
                dxdt = np.zeros_like(x)
            return dxdt

        def rhs_obl(_t, x):
            n_vars = self.physics.n_vars
            nc = self.physics.nc
            etor = self.physics.reservoir_operators[0]
            itor = self.physics.acc_flux_itor[0]
            ops_cpp = value_vector(np.zeros(self.physics.n_ops))
            ders_cpp = value_vector(np.zeros(self.physics.n_ops * n_vars))
            itor.evaluate_with_derivatives(
                value_vector(x), index_vector([0]), ops_cpp, ders_cpp
            )
            ops = np.asarray(ops_cpp)
            ders = np.asarray(ders_cpp).reshape(self.physics.n_ops, n_vars)
            poro = float(self.poro)
            dens_rock = float(self.dens_rock)
            c_r = float(self.c_r)

            dAdx = np.zeros((n_vars, n_vars))

            # mass accumulation
            dAdx[:nc] = poro * ders[etor.ACC_OP : etor.ACC_OP + nc]
            if n_vars > nc:
                # energy accumulation
                dAdx[nc] = poro * ders[etor.ACC_OP + nc]
                temp_op = getattr(etor, "TEMP_OP", None)
                if temp_op is not None:
                    dAdx[nc] += (1 - poro) * dens_rock * c_r * ders[temp_op]

            ## rate -> right-hand side
            b = np.asarray(self.get_obl_source_terms(_t, x, ops, etor), dtype=float)
            if b.size != n_vars:
                raise ValueError(
                    f"get_obl_source_terms must return {n_vars} values, got {b.size}."
                )

            try:
                constraints = self._apply_state_constraints(n_vars, nc)
                dxdt = self._solve_with_constraints(dAdx, b, n_vars, constraints)
            except np.linalg.LinAlgError:
                dxdt = np.zeros_like(x)

            return dxdt

        if self.mode == "analytical":
            rhs = rhs_analytical
        elif self.mode == "obl":
            rhs = rhs_obl

        result, state_next = solve_radau_step(
            rhs=rhs,
            state_prev=state_prev,
            dt=dt,
            t0=t0,
            rtol=rtol,
            atol=atol,
            composition_correction=correction_callback,
        )
        self.state = state_next
        return result

    def run_timestep(self, dt, t, method="radau", **kwargs):
        """
        Run a single timestep using the selected method.

        :param dt: Timestep size.
        :type dt: float
        :param t: Current simulation time.
        :type t: float
        :param method: Integration method identifier.
        :type method: str
        :param kwargs: Extra arguments passed to the integrator.
        :type kwargs: dict
        :return: Result of the timestep.
        :rtype: RadauStepResult
        """
        if self.state is None:
            raise RuntimeError("State is not initialized.")
        if dt <= 0:
            raise ValueError("Timestep size must be positive.")
        state_prev = self.state.copy()
        if method.lower() == "radau":
            result = self.step_radau(state_prev, dt, t0=t, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if result.success:
            self.record_state(t + dt, self.state)
        return result

    def run(
        self,
        days: float = None,
        method: str = "radau",
        restart_dt: float = 0.0,
        ts_adaptive: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Run the simulation for the requested duration.

        :param days: Duration to simulate.
        :type days: float or None
        :param method: Integration method identifier.
        :type method: str
        :param restart_dt: Initial timestep size override.
        :type restart_dt: float
        :param ts_adaptive: Enable adaptive timestep selection.
        :type ts_adaptive: bool
        :param verbose: Enable stdout logging.
        :type verbose: bool
        :param kwargs: Extra arguments passed to the integrator.
        :type kwargs: dict
        :return: True if the run reached the final time.
        :rtype: bool
        """
        if self.state is None:
            raise RuntimeError("State is not initialized.")

        days = self.runtime if days is None else days
        if days <= 0:
            raise ValueError("Runtime must be positive.")

        data_ts = self.data_ts
        t = self.time_history[-1] if self.time_history else 0.0
        if not self.time_history:
            self.record_state(t, self.state)
        stop_time = t + days

        if abs(t) < 1e-15 or not hasattr(self, "prev_dt"):
            dt = data_ts.dt_first
        elif restart_dt > 0.0:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * data_ts.dt_mult, data_ts.dt_max)
        if t + dt > stop_time:
            dt = stop_time - t

        ts = 0
        while t < stop_time:
            state_prev = self.state.copy()
            result = self.run_timestep(dt, t, method=method, **kwargs)

            if result.success:
                t += dt
                ts += 1
                if verbose:
                    state_str = self._format_state_for_log(self.state)
                    print(
                        f"T={t:>10.4g}  DT={dt:>10.4g}  "
                        f"NI={result.nlu:>3d}  NFEV={result.nfev:>4d}  {state_str}"
                    )

                if ts_adaptive and data_ts.eta is not None:
                    dx = np.abs(self.state - state_prev)
                    dx = np.maximum(dx, np.finfo(float).eps)
                    eta_arr = np.asarray(data_ts.eta, dtype=float)
                    if eta_arr.size == 1:
                        eta_arr = np.full_like(dx, float(eta_arr))

                    if abs(data_ts.dt_mult - 1.0) < 1e-12:
                        omega = 0.0
                    else:
                        omega = 1.0 / (data_ts.dt_mult - 1.0)

                    mult_vec = ((1.0 + omega) * eta_arr) / (dx + omega * eta_arr)
                    mult = float(min(np.min(mult_vec), data_ts.dt_mult))
                    dt = min(dt * mult, data_ts.dt_max)
                else:
                    if ts_adaptive and data_ts.eta is None and verbose:
                        print("Falling back to simple growth (eta is None).")
                    dt = min(dt * data_ts.dt_mult, data_ts.dt_max)

                if np.fabs(t + dt - stop_time) < data_ts.dt_min or t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

            else:
                dt /= data_ts.dt_mult
                if verbose:
                    reason = f" ({result.message})" if result.message else ""
                    print(f"Cut timestep to {dt:2.10f}{reason}")
                if dt < data_ts.dt_min:
                    print("Convergence error: reached minimum timestep.")
                    return False

        return True

    def _resolve_output_n_vars(self) -> int:
        """Resolve number of state variables for HDF5 dataset allocation."""
        if self.state is not None:
            return int(np.asarray(self.state, dtype=float).size)
        if getattr(self, "initial_state", None) is not None:
            return int(np.asarray(self.initial_state, dtype=float).size)
        if hasattr(self, "property_container") and hasattr(
            self.property_container, "n_vars"
        ):
            return int(self.property_container.n_vars)
        if hasattr(self, "physics") and hasattr(self.physics, "n_vars"):
            return int(self.physics.n_vars)
        raise RuntimeError("Cannot determine state size for HDF5 output.")

    def _dataset_write_kwargs(self):
        """Return common dataset kwargs for HDF5 output."""
        kwargs = {}
        if self.compression:
            kwargs["compression"] = self.compression
            if self.compression == "gzip":
                kwargs["compression_opts"] = self.h5_compression_level
        return kwargs

    def _initialize_h5_output_file(self, n_vars: int, overwrite: bool = True):
        """Create an empty HDF5 file with dynamic/property groups."""
        output_dir = os.path.dirname(self.sol_filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if overwrite and os.path.exists(self.sol_filepath):
            os.remove(self.sol_filepath)

        dtype = self.precision_map[self.precision]
        ds_kwargs = self._dataset_write_kwargs()
        var_names = self._state_variable_names(n_vars)

        with h5py.File(self.sol_filepath, "w") as f:
            dynamic_group = f.create_group("dynamic")
            dynamic_group.create_dataset(
                "time",
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                **ds_kwargs,
            )
            dynamic_group.create_dataset(
                "X",
                shape=(0, 1, n_vars),
                maxshape=(None, 1, n_vars),
                dtype=dtype,
                **ds_kwargs,
            )
            str_dtype = h5py.special_dtype(vlen=str)
            dynamic_group.create_dataset(
                "variable_names", data=np.array(var_names, dtype=str_dtype)
            )
            f.create_group("properties")
            f.attrs["description"] = "0D model solution and evaluated properties"

        self._h5_ready = True

    def _append_timestep_to_h5(self, t: float, state: np.ndarray, props_snapshot: dict):
        """Append one accepted timestep (state + properties) to HDF5."""
        if not self._h5_ready:
            raise RuntimeError(
                "HDF5 output is not initialized. Call set_output(...) first."
            )

        dtype = self.precision_map[self.precision]
        ds_kwargs = self._dataset_write_kwargs()

        with h5py.File(self.sol_filepath, "a") as f:
            dynamic_group = f["dynamic"]
            time_ds = dynamic_group["time"]
            x_ds = dynamic_group["X"]

            idx = time_ds.shape[0]

            time_ds.resize((idx + 1,))
            time_ds[idx] = t

            x_ds.resize((idx + 1, 1, x_ds.shape[2]))
            x_ds[idx, 0, :] = np.asarray(state, dtype=dtype, copy=False)

            prop_group = f["properties"]
            for name, arr in props_snapshot.items():
                arr_flat = np.asarray(arr, dtype=dtype, copy=False).reshape(-1)
                nvals = arr_flat.size

                if name not in prop_group:
                    dset = prop_group.create_dataset(
                        name,
                        shape=(0, nvals),
                        maxshape=(None, nvals),
                        dtype=dtype,
                        **ds_kwargs,
                    )
                else:
                    dset = prop_group[name]
                    if dset.shape[1] != nvals:
                        raise ValueError(
                            f"Property '{name}' size changed from {dset.shape[1]} to {nvals}."
                        )

                dset.resize((idx + 1, nvals))
                dset[idx, :] = arr_flat

    @staticmethod
    def _to_numeric_property_array(value):
        if value is None:
            return None
        if isinstance(value, bool | np.bool_):
            return None

        if np.isscalar(value):
            if isinstance(value, str | bytes):
                return None
            return np.asarray(value, dtype=float)

        try:
            arr = np.asarray(value)
        except Exception:
            return None

        if arr.dtype.kind in {"U", "S", "O", "V", "b"}:
            return None

        try:
            return arr.astype(float, copy=False)
        except Exception:
            return None

    @staticmethod
    def _stack_property_series(series: list):
        """Stack one property time series into a single array."""
        ref_shape = None
        for item in series:
            if item is None:
                continue
            if ref_shape is None:
                ref_shape = item.shape
            elif item.shape != ref_shape:
                ref_shape = None
                break

        n_steps = len(series)
        if ref_shape is not None:
            data = np.full((n_steps,) + ref_shape, np.nan, dtype=float)
            for i, item in enumerate(series):
                if item is not None:
                    data[i] = item
            return data, False

        max_size = max((item.size for item in series if item is not None), default=0)
        if max_size == 0:
            return None, False

        data = np.full((n_steps, max_size), np.nan, dtype=float)
        for i, item in enumerate(series):
            if item is None:
                continue
            flat = item.reshape(-1)
            data[i, : flat.size] = flat
        return data, True

    def _state_variable_names(self, n_vars: int) -> list[str]:
        if self.mode == "obl" and hasattr(self, "physics"):
            names = list(getattr(self.physics, "vars", []))
            if len(names) == n_vars:
                return names

        props = getattr(self, "property_container", None)
        nc = int(getattr(props, "nc", 0)) if props is not None else 0
        names = ["pressure"]

        if nc > 0:
            comp_names = list(getattr(props, "components_name", []))
            if len(comp_names) == nc:
                names += [f"z_{comp}" for comp in comp_names[: max(0, nc - 1)]]
            else:
                names += [f"z{i}" for i in range(max(0, nc - 1))]
            if n_vars > nc:
                # 0D thermal state is usually enthalpy for PH, temperature otherwise.
                state_spec = getattr(props, "state_spec", None)
                state_spec = str(
                    state_spec.name if hasattr(state_spec, "name") else state_spec
                ).upper()
                names.append(
                    "enthalpy"
                    if ("PH" in state_spec or "ENTHALPY" in state_spec)
                    else "temperature"
                )

        if len(names) < n_vars:
            names += [f"state_{i}" for i in range(len(names), n_vars)]
        return names[:n_vars]

    def evaluate_output_properties(self, state):
        """
        Evaluate output properties for the current state.

        Derived models can override this method to expose a custom property set.
        """
        container = getattr(self, "property_container", None)
        if container is None:
            raise RuntimeError("property_container is not defined.")

        state_np = np.asarray(state, dtype=float)
        if self.mode == "obl":
            prop_itor = self.physics.property_itor[0]
            prop_names = self.physics.property_operators[0].props_name
            n_ops = self.physics.n_property_itor_ops
            values = value_vector(np.zeros(int(n_ops)))
            prop_itor.evaluate(value_vector(state_np), values)
            values_np = np.array(values, copy=False)

            props = {}
            for i, name in enumerate(prop_names):
                if i >= values_np.size:
                    break
                props[name] = values_np[i]
            return props

        container.evaluate(state_np)

        evaluate_thermal = getattr(container, "evaluate_thermal", None)
        if callable(evaluate_thermal):
            try:
                evaluate_thermal(state_np)
            except Exception:
                # Thermal part is optional in some property containers.
                pass

        output_props = getattr(container, "output_props", None)
        if not output_props:
            return {}

        props = {}
        for name, prop in output_props.items():
            props[name] = prop()

        return props

    def extract_property_history(self):
        """
        Return cached per-timestep properties (already evaluated during run).
        """
        if not self.time_history:
            return np.array([], dtype=float), {}

        property_array = {}
        for name, series in self.property_history.items():
            stacked, _flattened = self._stack_property_series(series)
            if stacked is not None:
                property_array[name] = stacked

        return np.asarray(self.time_history, dtype=float), property_array

    def save_data_to_h5(
        self,
        filepath: str = None,
    ):
        """
        Save 0D state and property histories to HDF5.

        Output layout:
        - ``dynamic/time``
        - ``dynamic/X`` (shape: [n_t, 1, n_vars])
        - ``dynamic/variable_names``
        - ``properties/<property_name>``
        """
        if not self.state_history or not self.time_history:
            raise RuntimeError(
                "State history is empty. Run the model before saving HDF5."
            )

        path = filepath if filepath is not None else self.sol_filepath
        if path is None:
            raise ValueError("Output filepath could not be determined.")

        old_path = self.sol_filepath
        old_ready = self._h5_ready
        self.sol_filepath = path

        dtype = self.precision_map[self.precision]
        times = np.asarray(self.time_history, dtype=dtype)
        states = np.asarray(self.state_history, dtype=dtype)
        _, props = self.extract_property_history()
        self._initialize_h5_output_file(n_vars=states.shape[1], overwrite=True)

        with h5py.File(self.sol_filepath, "a") as f:
            dynamic_group = f["dynamic"]
            dynamic_group["time"].resize((times.shape[0],))
            dynamic_group["time"][:] = times
            dynamic_group["X"].resize((states.shape[0], 1, states.shape[1]))
            dynamic_group["X"][:] = states[:, np.newaxis, :]

            prop_group = f["properties"]
            ds_kwargs = self._dataset_write_kwargs()
            for name, arr in props.items():
                data = np.asarray(arr, dtype=dtype, copy=False)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                elif data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
                prop_group.create_dataset(name, data=data, **ds_kwargs)

        if filepath is not None:
            self.sol_filepath = old_path
            self._h5_ready = old_ready
        else:
            self._h5_ready = True

    def get_state_path(self):
        """
        Return pressure and thermal state histories.

        :return: Tuple of (pressure, thermal) arrays.
        :rtype: tuple(numpy.ndarray, numpy.ndarray or None)
        """
        if not self.state_history:
            return np.array([]), np.array([])
        states = np.asarray(self.state_history)
        pressure = states[:, 0]
        thermal = (
            states[:, -1]
            if self.property_container.n_vars > self.property_container.nc
            else None
        )
        return pressure, thermal

    def plot_pt_path(
        self,
        p_min=5e-4,
        p_max=1.3e-2,
        t_min=250,
        t_max=500,
        use_log_p=False,
        output_path="pt_path.png",
    ):
        """
        Plot the pressure-temperature path on a PT diagram.

        :param p_min: Minimum pressure for the background grid.
        :type p_min: float
        :param p_max: Maximum pressure for the background grid.
        :type p_max: float
        :param t_min: Minimum temperature for the background grid.
        :type t_min: float
        :param t_max: Maximum temperature for the background grid.
        :type t_max: float
        :param use_log_p: Use logarithmic pressure axis.
        :type use_log_p: bool
        :param output_path: File path for saving the PT diagram.
        :type output_path: str
        Note: If state history exists, limits are set from the path
        using 0.9 * min and 1.1 * max (sign-aware).
        """
        import matplotlib.pyplot as plt
        from dartsflash.dartsflash import DARTSFlash
        from dartsflash.libflash import EoS
        from dartsflash.plot import PlotFlash

        if (
            self.physics is not None
            and self.physics.state_spec != PhysicsBase.StateSpecification.PT
        ):
            raise RuntimeError("PT plot requires PT formulation.")

        pressure, thermal = self.get_state_path()
        if pressure.size and thermal is not None and thermal.size:
            p_min_val = np.nanmin(pressure)
            p_max_val = np.nanmax(pressure)
            p_min = p_min_val * (0.9 if p_min_val >= 0 else 1.1)
            p_max = p_max_val * (1.1 if p_max_val >= 0 else 0.9)
            t_min_val = np.nanmin(thermal)
            t_max_val = np.nanmax(thermal)
            t_min = t_min_val * (0.9 if t_min_val >= 0 else 1.1)
            t_max = t_max_val * (1.1 if t_max_val >= 0 else 0.9)

        flash = getattr(self, "flash_pt", None)
        if flash is None:
            flash = getattr(self.property_container, "flash_ev", None)
            flash_type = getattr(flash, "flash_type", None)
            if flash_type is not None and flash_type != DARTSFlash.FlashType.PTFlash:
                flash = None

        if flash is None:
            raise RuntimeError(
                "PT flash is not initialized. Provide self.flash_pt or a PT "
                "flash in property_container.flash_ev."
            )

        components = list(flash.components)
        compositions = {components[0]: 1.0}

        f = flash

        if use_log_p:
            state_spec_background = {
                "pressure": np.logspace(np.log10(p_min), np.log10(p_max), 100),
                "temperature": np.linspace(t_min, t_max, 100),
            }
        else:
            state_spec_background = {
                "pressure": np.linspace(p_min, p_max, 100),
                "temperature": np.linspace(t_min, t_max, 100),
            }

        results_pt = f.evaluate_flash(
            state_spec=state_spec_background,
            compositions=compositions,
            mole_fractions=True,
            print_state="Flash",
        )
        props = f.evaluate_properties_np(
            state_spec=state_spec_background,
            compositions=compositions,
            state_variables=f.state_vars + components,
            flash_results=results_pt,
            total_properties_to_evaluate={"H": EoS.Property.ENTHALPY},
            print_state="Properties",
        )

        pt_plot = PlotFlash.pt(
            f,
            results_pt,
            composition=[1.0],
            plot_phase_fractions=False,
            logP=use_log_p,
            pt_props=props,
        )

        ax = pt_plot.ax[0]
        pressure, thermal = self.get_state_path()
        if thermal is None:
            raise RuntimeError("Thermal state is not available for PT plot.")
        ax.plot(
            thermal, pressure, color="tab:red", linewidth=1, marker="o", markersize=1
        )
        pt_plot.add_attributes(
            title=f"PT-Diagram ({p_min:.1e} to {p_max:.1e} bar, {t_min} to {t_max} K)"
        )
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def plot_ph_path(
        self,
        p_min=5e-4,
        p_max=5e-3,
        t_min=200,
        t_max=270,
        use_log_p=False,
        output_path="ph_path.png",
    ):
        """
        Plot the pressure-enthalpy path on a PH diagram.

        :param p_min: Minimum pressure for the background grid.
        :type p_min: float
        :param p_max: Maximum pressure for the background grid.
        :type p_max: float
        :param t_min: Minimum temperature for the background grid.
        :type t_min: float
        :param t_max: Maximum temperature for the background grid.
        :type t_max: float
        :param use_log_p: Use logarithmic pressure axis.
        :type use_log_p: bool
        :param output_path: File path for saving the PH diagram.
        :type output_path: str
        Note: If state history exists, pressure and enthalpy limits are
        set from the path using 0.9 * min and 1.1 * max (sign-aware).
        """
        import matplotlib.pyplot as plt
        from dartsflash.plot import PlotFlash

        # if (
        #     self.physics is not None
        #     and self.physics.state_spec != PhysicsBase.StateSpecification.PH
        # ):
        #     raise RuntimeError("PH plot requires PH formulation.")

        pressure, thermal = self.get_state_path()
        enthalpy_min = None
        enthalpy_max = None
        if pressure.size and thermal is not None and thermal.size:
            p_min_val = np.nanmin(pressure)
            p_max_val = np.nanmax(pressure)
            p_min = p_min_val * (0.9 if p_min_val >= 0 else 1.1)
            p_max = p_max_val * (1.1 if p_max_val >= 0 else 0.9)
            h_min_val = np.nanmin(thermal)
            h_max_val = np.nanmax(thermal)
            enthalpy_min = h_min_val * (0.9 if h_min_val >= 0 else 1.1)
            enthalpy_max = h_max_val * (1.1 if h_max_val >= 0 else 0.9)

        flash = getattr(self, "flash_ph", None)
        if flash is None:
            raise RuntimeError(
                "PH flash is not initialized. Provide self.flash_ph or a PH "
                "flash in property_container.flash_ev."
            )

        f = flash

        flash_params = getattr(f, "flash_params", None)
        if flash_params is not None:
            t_min_env = min(t_min, getattr(flash_params, "T_min", t_min))
            t_max_env = max(t_max, getattr(flash_params, "T_max", t_max))
        else:
            t_min_env, t_max_env = t_min, t_max

        Xrange = f.get_ranges(
            prange=[p_min, p_max],
            trange=[t_min_env, t_max_env],
            composition=[1.0],
        )
        if enthalpy_min is None or enthalpy_max is None:
            enthalpy_min, enthalpy_max = Xrange[0], Xrange[1]
        else:
            enthalpy_min = min(enthalpy_min, Xrange[0])
            enthalpy_max = max(enthalpy_max, Xrange[1])
            if enthalpy_min >= enthalpy_max:
                enthalpy_min, enthalpy_max = Xrange[0], Xrange[1]
        if use_log_p:
            state_spec_background = {
                "pressure": np.logspace(np.log10(p_min), np.log10(p_max), 100),
                "enthalpy": np.linspace(enthalpy_min, enthalpy_max, 1000),
            }
        else:
            state_spec_background = {
                "pressure": np.linspace(p_min, p_max, 100),
                "enthalpy": np.linspace(enthalpy_min, enthalpy_max, 1000),
            }
        results_ph = f.evaluate_flash_1c(state_spec=state_spec_background)
        np_min = None
        np_max = None
        if "np" in results_ph:
            results_ph = results_ph.copy()
            np_values = np.nan_to_num(results_ph["np"].values, nan=0.0)
            np_min = int(np.nanmin(np_values))
            np_max = int(np.nanmax(np_values))
            np_da = results_ph["np"].copy(data=np_values.round().astype(int))
            results_ph = results_ph.assign(np=np_da)

        ph_diagram = PlotFlash.ph(
            f,
            results_ph,
            composition=[1.0],
            plot_phase_fractions=False,
            logP=use_log_p,
            min_val=np_min,
            max_val=np_max,
        )
        ax = ph_diagram.ax[0]

        pressure, thermal = self.get_state_path()
        if thermal is None:
            raise RuntimeError("Thermal state is not available for PH plot.")
        ax.plot(
            thermal, pressure, color="tab:red", linewidth=1, marker="o", markersize=1
        )
        ph_diagram.add_attributes(title=f"PH-Diagram ({p_min:.1e} to {p_max:.1e} bar)")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def plot_state_history(
        self,
        output_path="state_history.png",
        show=False,
        figsize=None,
        dpi=150,
        linewidth=2.0,
        alpha=1.0,
        grid=True,
        legend=True,
        label_fontsize=12,
        title_fontsize=12,
        tick_fontsize=12,
        legend_fontsize=12,
        time_label="time",
        time_units="days",
        pressure_label="pressure, bar",
        temperature_label="temperature, K",
        enthalpy_label="enthalpy, kJ/kmol",
        pressure_color="tab:blue",
        temperature_color="tab:orange",
        enthalpy_color="tab:green",
        composition_colors=None,
        use_log_p=False,
    ):
        """
        Plot pressure, compositions, phase saturations, and optional
        enthalpy/temperature vs time.

        :param output_path: File path for saving the plot (None to skip saving).
        :type output_path: str or None
        :param show: If True, display the plot window.
        :type show: bool
        :param figsize: Matplotlib figure size.
        :type figsize: tuple(float, float) or None
        :param dpi: Figure DPI.
        :type dpi: int
        :param linewidth: Line width for all series.
        :type linewidth: float
        :param alpha: Line transparency.
        :type alpha: float
        :param grid: Enable grid on subplots.
        :type grid: bool
        :param legend: Show legends on subplots.
        :type legend: bool
        :param label_fontsize: Font size for axis labels.
        :type label_fontsize: int
        :param title_fontsize: Font size for subplot titles.
        :type title_fontsize: int
        :param tick_fontsize: Font size for tick labels.
        :type tick_fontsize: int
        :param legend_fontsize: Font size for legends.
        :type legend_fontsize: int
        :param time_label: Label for the time axis.
        :type time_label: str
        :param time_units: Units for the time axis.
        :type time_units: str
        :param pressure_label: Label for pressure axis.
        :type pressure_label: str
        :param temperature_label: Label for temperature axis.
        :type temperature_label: str
        :param enthalpy_label: Label for enthalpy axis.
        :type enthalpy_label: str
        :param pressure_color: Line color for pressure.
        :type pressure_color: str
        :param temperature_color: Line color for temperature.
        :type temperature_color: str
        :param enthalpy_color: Line color for enthalpy.
        :type enthalpy_color: str
        :param composition_colors: Colors for compositions (list or dict).
        :type composition_colors: list[str] or dict[str, str] or None
        :param use_log_p: Use logarithmic scale for pressure axis.
        :type use_log_p: bool
        :return: Matplotlib figure and axes.
        :rtype: tuple(matplotlib.figure.Figure, list[matplotlib.axes.Axes])
        """
        import matplotlib.pyplot as plt

        if not self.time_history or not self.state_history:
            raise RuntimeError("State history is empty. Run the model before plotting.")

        time_arr = np.asarray(self.time_history, dtype=float)
        state_arr = np.asarray(self.state_history, dtype=float)
        n_steps = min(time_arr.size, state_arr.shape[0])
        time_arr = time_arr[:n_steps]
        state_arr = state_arr[:n_steps]
        n_vars = state_arr.shape[1]

        props = self.property_container
        nc = getattr(props, "nc", 0)
        if nc <= 0:
            raise RuntimeError("Property container is missing component count.")
        has_thermal_state = n_vars > nc

        pressure = state_arr[:, 0]

        if nc == 1:
            zc = np.ones((n_steps, 1))
        else:
            zc = np.zeros((n_steps, nc))
            zc[:, : nc - 1] = state_arr[:, 1:nc]
            zc[:, -1] = 1.0 - np.sum(zc[:, : nc - 1], axis=1)

        comp_names = getattr(props, "components_name", None)
        if comp_names is None or len(comp_names) != nc:
            comp_names = [f"comp{i + 1}" for i in range(nc)]

        nph = getattr(props, "nph", 0)
        if nph <= 0:
            raise RuntimeError("Property container is missing phase count.")
        phase_names = getattr(props, "phases_name", None)
        if phase_names is None or len(phase_names) != nph:
            phase_names = [f"phase{i + 1}" for i in range(nph)]

        enthalpy = np.full(n_steps, np.nan) if has_thermal_state else None
        temperature = np.full(n_steps, np.nan) if has_thermal_state else None
        sat_history = np.full((n_steps, nph), np.nan)
        spec = getattr(props, "state_spec", None)
        spec_str = spec.name if hasattr(spec, "name") else str(spec)
        if self.mode == "analytical":
            is_ph = "ENTHALPY" in spec_str or "PH" in spec_str
        elif self.mode == "obl":
            is_ph = self.physics.state_spec == Compositional.StateSpecification.PH
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        etor = None
        itor = None
        ops_cpp = None
        if self.mode == "obl":
            etor = self.physics.reservoir_operators[0]
            itor = self.physics.acc_flux_itor[0]
            ops_cpp = value_vector(np.zeros(self.physics.n_ops))

        if has_thermal_state:
            if is_ph:
                enthalpy[:] = state_arr[:, nc]
            else:
                temperature[:] = state_arr[:, nc]

        for i in range(n_steps):
            try:
                if self.mode == "analytical":
                    props.evaluate(state_arr[i])
                    sat_values = np.asarray(getattr(props, "sat", []), dtype=float)
                    if sat_values.size >= nph:
                        sat_history[i, :] = sat_values[:nph]
                    if has_thermal_state and is_ph:
                        temperature[i] = getattr(props, "temperature", np.nan)
                elif self.mode == "obl":
                    itor.evaluate(value_vector(state_arr[i]), ops_cpp)
                    for p in range(nph):
                        sat_history[i, p] = ops_cpp[etor.SAT_OP + p]
                    if has_thermal_state and is_ph:
                        temperature[i] = ops_cpp[etor.TEMP_OP]
            except Exception:
                continue

        nplots = 1 + nc + 1 + (2 if has_thermal_state else 0)
        fig_height = 2.5 * nplots if figsize is None else None
        fig, axes = plt.subplots(
            nrows=nplots,
            ncols=1,
            sharex=True,
            figsize=figsize or (8.0, fig_height),
            dpi=dpi,
        )
        if nplots == 1:
            axes = [axes]

        idx = 0
        axes[idx].plot(
            time_arr,
            pressure,
            color=pressure_color,
            linewidth=linewidth,
            alpha=alpha,
            label=pressure_label,
        )
        if use_log_p:
            if np.any(pressure <= 0):
                raise ValueError("Log pressure scale requires positive pressures.")
            axes[idx].set_yscale("log")
        axes[idx].set_ylabel(pressure_label, fontsize=label_fontsize)
        axes[idx].set_title("Pressure", fontsize=title_fontsize)
        idx += 1

        color_cycle = None
        if isinstance(composition_colors, dict):
            color_cycle = [composition_colors.get(name) for name in comp_names]
        elif isinstance(composition_colors, list | tuple):
            color_cycle = list(composition_colors)

        for i, name in enumerate(comp_names):
            color = None
            if color_cycle:
                color = color_cycle[i % len(color_cycle)]
            axes[idx].plot(
                time_arr,
                zc[:, i],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=f"z_{name}",
            )
            axes[idx].set_ylabel(f"z_{name}", fontsize=label_fontsize)
            axes[idx].set_title(f"Composition {name}", fontsize=title_fontsize)
            idx += 1

        for j, name in enumerate(phase_names):
            axes[idx].plot(
                time_arr,
                sat_history[:, j],
                linewidth=linewidth,
                alpha=alpha,
                label=f"S_{name}",
            )
        axes[idx].set_ylabel("Saturation", fontsize=label_fontsize)
        axes[idx].set_title("Phase Saturations", fontsize=title_fontsize)
        idx += 1

        if has_thermal_state:
            axes[idx].plot(
                time_arr,
                enthalpy,
                color=enthalpy_color,
                linewidth=linewidth,
                alpha=alpha,
                label=enthalpy_label,
            )
            axes[idx].set_ylabel(enthalpy_label, fontsize=label_fontsize)
            axes[idx].set_title("Enthalpy", fontsize=title_fontsize)
            idx += 1

            axes[idx].plot(
                time_arr,
                temperature,
                color=temperature_color,
                linewidth=linewidth,
                alpha=alpha,
                label=temperature_label,
            )
            axes[idx].set_ylabel(temperature_label, fontsize=label_fontsize)
            axes[idx].set_title("Temperature", fontsize=title_fontsize)

        for ax in axes:
            ax.tick_params(labelsize=tick_fontsize)
            if grid:
                ax.grid(True, alpha=0.3)
            if legend:
                ax.legend(fontsize=legend_fontsize)

        axes[-1].set_xlabel(f"{time_label} ({time_units})", fontsize=label_fontsize)
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axes
