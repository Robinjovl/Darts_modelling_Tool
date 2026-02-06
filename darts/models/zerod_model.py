from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from darts.engines import index_vector, value_vector
from darts.models.darts_model import DartsModel, DataTS
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.super.physics import Compositional


@dataclass(frozen=True)
class RadauStepResult:
    success: bool
    nfev: int = 0
    njev: int = 0
    nlu: int = 0
    message: str = ""


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
    ):
        """
        Initialize the model.
        :param fixed_pressure: flag to fix pressure or not
        :type fixed_pressure: bool
        :param fixed_temperature: flag to fix temperature or not
        :type fixed_temperature: bool
        """
        super().__init__()

        self.fixed_pressure = fixed_pressure
        self.fixed_temperature = fixed_temperature

        self.state = None
        self.state_history = []
        self.time_history = []

        # Nonlinear solver defaults
        self.nonlin_max_iter = 50
        self.nonlin_tol_res = 1e-10
        self.nonlin_tol_dx = 1e-10

        # Radau defaults
        self.radau_rtol = 1e-8
        self.radau_atol = 1e-10

    def init(self):
        """
        Initialize the model state and history.
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
        self.record_state(0.0, self.state)

    def record_state(self, t, state):
        """
        Store the current state and time in history.

        :param t: Simulation time.
        :type t: float
        :param state: State vector to store.
        :type state: array-like
        """
        self.time_history.append(float(t))
        self.state_history.append(np.asarray(state, dtype=float).copy())

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

        comp_names = props.components_name
        if not comp_names or len(comp_names) != nc:
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
        elif self.mode == "obl":
            itor = self.physics.acc_flux_itor[0]
            etor = self.physics.reservoir_operators[0]
            ops_cpp = value_vector(np.zeros(self.physics.n_ops))
            itor.evaluate(value_vector(state), ops_cpp)
            temp_val = ops_cpp[etor.TEMP_OP]

        if thermal_val is not None:
            state_str = (
                f"p={p_val:>{num_width}.6g} {z_parts} "
                f"{thermal_label}={thermal_val:>{num_width}.6g}"
            )
        else:
            state_str = f"p={p_val:>{num_width}.6g} {z_parts}"

        if temp_val is not None:
            state_str += f" temp={temp_val:>{num_width}.6g}"
        return state_str

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

    def step_radau(self, state_prev, dt, t0=0.0, rtol=None, atol=None):
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
        :return: Result of the Radau step.
        :rtype: RadauStepResult
        """
        rtol = self.radau_rtol if rtol is None else rtol
        atol = self.radau_atol if atol is None else atol

        # Radau solver considers system resolved w.r.t. unknowns,
        # thus transition from dA/dt = dA/dX * dX/dt = g(t, p, z, T/h) to dX/dt = f(t, p, z, T/h)
        # is necessary.
        def rhs_analytical(_t, x):
            props = self.property_container
            props.evaluate(x)
            ph = props.ph
            n_vars = props.n_vars
            nc = props.nc

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
            dAdx[:nc] = self.poro * (dz_dx * rho_t + zc * drho_t_dx)

            # solid energy accumulation: Es = phi * rho_r * c_r * (T - T0) [kJ/m3]
            dEs_dx = np.zeros(n_vars)
            dEs_dx[0] = (1 - self.poro) * self.dens_rock * self.c_r * props.dTdP
            dEs_dx[1:nc] = (
                (1 - self.poro) * self.dens_rock * self.c_r * props.dTdzk[: nc - 1]
            )
            dEs_dx[nc] = (1 - self.poro) * self.dens_rock * self.c_r * props.dTdX
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
            dEf_dx *= self.poro
            dAdx[nc] += dEf_dx

            ## rate -> right-hand side
            b = np.zeros(n_vars)
            if self.energy_source is not None:
                b[-1] = self.energy_source.evaluate(_t)

            try:
                dxdt = np.linalg.solve(dAdx, b)
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
            ders = np.asarray(ders_cpp).reshape(self.physics.n_ops, n_vars)

            dAdx = np.zeros((n_vars, n_vars))

            # mass accumulation
            dAdx[:nc] = self.poro * ders[etor.ACC_OP : etor.ACC_OP + nc]
            # energy accumulation
            dAdx[nc] = (1 - self.poro) * self.dens_rock * self.c_r * ders[
                etor.TEMP_OP
            ] + self.poro * ders[etor.ACC_OP + nc]

            ## rate -> right-hand side
            b = np.zeros(n_vars)
            if self.energy_source is not None:
                b[-1] = self.energy_source.evaluate(_t)

            try:
                dxdt = np.linalg.solve(dAdx, b)
            except np.linalg.LinAlgError:
                dxdt = np.zeros_like(x)

            return dxdt

        if self.mode == "analytical":
            rhs = rhs_analytical
        elif self.mode == "obl":
            rhs = rhs_obl

        try:
            sol = solve_ivp(
                rhs,
                (t0, t0 + dt),
                state_prev,
                method="Radau",
                t_eval=[t0 + dt],
                rtol=rtol,
                atol=atol,
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
                self.state = sol.y[:, -1].copy()
                return result
        except Exception as exc:
            result = RadauStepResult(success=False, message=str(exc))
        self.state = np.asarray(state_prev, dtype=float).copy()
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
            thermal, pressure, color="tab:red", linewidth=2, marker="o", markersize=3
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
            thermal, pressure, color="tab:red", linewidth=2, marker="o", markersize=3
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
        Plot pressure, compositions, enthalpy, and temperature vs time.

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

        props = self.property_container
        nc = getattr(props, "nc", 0)
        if nc <= 0:
            raise RuntimeError("Property container is missing component count.")

        pressure = state_arr[:, 0]

        if nc == 1:
            zc = np.ones((n_steps, 1))
        else:
            zc = np.zeros((n_steps, nc))
            zc[:, : nc - 1] = state_arr[:, 1:nc]
            zc[:, -1] = 1.0 - np.sum(zc[:, : nc - 1], axis=1)

        comp_names = getattr(props, "components_name", None)
        if not comp_names or len(comp_names) != nc:
            comp_names = [f"comp{i + 1}" for i in range(nc)]

        enthalpy = np.full(n_steps, np.nan)
        temperature = np.full(n_steps, np.nan)
        spec = getattr(props, "state_spec", None)
        spec_str = spec.name if hasattr(spec, "name") else str(spec)
        if self.mode == "analytical":
            is_ph = "ENTHALPY" in spec_str or "PH" in spec_str
        elif self.mode == "obl":
            is_ph = self.physics.state_spec == Compositional.StateSpecification.PH
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if state_arr.shape[1] > nc:
            if is_ph:
                enthalpy[:] = state_arr[:, nc]
            else:
                temperature[:] = state_arr[:, nc]

        for i in range(n_steps):
            try:
                if is_ph:
                    if self.mode == "analytical":
                        props.evaluate(state_arr[i])
                        temperature[i] = getattr(props, "temperature", np.nan)
                    elif self.mode == "obl":
                        etor = self.physics.reservoir_operators[0]
                        itor = self.physics.acc_flux_itor[0]
                        ops_cpp = value_vector(np.zeros(self.physics.n_ops))
                        itor.evaluate(value_vector(state_arr[i]), ops_cpp)
                        temperature[i] = ops_cpp[etor.TEMP_OP]
                else:
                    raise ValueError(
                        f"plot_state_history: Enthalpy evaluation is not supported for mode: {self.mode}"
                    )
            except Exception:
                continue

        nplots = 1 + nc + 2
        fig_height = 2.2 * nplots if figsize is None else None
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
