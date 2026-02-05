from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from darts.models.darts_model import DartsModel, DataTS
from darts.physics.base.physics_base import PhysicsBase


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
        state_arr = np.asarray(state, dtype=float)
        num_width = 12
        p_val = state_arr[0] if state_arr.size else float("nan")

        nc = getattr(props, "nc", None)
        if not nc:
            return f"p={p_val:>{num_width}.6g}"

        if nc == 1:
            zc = np.array([1.0])
        else:
            zc = np.zeros(nc)
            zc[: nc - 1] = state_arr[1:nc]
            zc[-1] = 1.0 - np.sum(zc[: nc - 1])

        comp_names = getattr(props, "components_name", None)
        if not comp_names or len(comp_names) != nc:
            comp_names = [f"comp{i + 1}" for i in range(nc)]
        z_parts = " ".join(
            f"z_{comp_names[i]}={zc[i]:>{num_width}.6g}" for i in range(nc)
        )

        thermal_val = None
        if state_arr.size > nc:
            thermal_val = state_arr[nc]

        thermal_label = "t"
        spec = getattr(props, "state_spec", None)
        if spec is not None:
            spec_str = spec.name if hasattr(spec, "name") else str(spec)
            if "ENTHALPY" in spec_str or "PH" in spec_str:
                thermal_label = "h"

        if thermal_val is not None:
            return (
                f"p={p_val:>{num_width}.6g} {z_parts} "
                f"{thermal_label}={thermal_val:>{num_width}.6g}"
            )
        return f"p={p_val:>{num_width}.6g} {z_parts}"

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

    def step_radau_analytical(self, state_prev, dt, t0=0.0, rtol=None, atol=None):
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
        props = self.property_container

        # Radau solver considers system resolved w.r.t. unknowns,
        # thus transition from dA/dt = dA/dX * dX/dt = g(t, p, z, T/h) to dX/dt = f(t, p, z, T/h)
        # is necessary.
        def rhs(_t, x):
            props.evaluate(x)
            ph = props.ph
            n_vars = props.n_vars
            nc = props.nc

            ## accumulation
            dAdt = np.zeros((n_vars, n_vars))

            # mass accumulation: M = c_r * PV * z_c * sum(sat * rho_m) [kmol/m3]
            zc = np.append(x[1:nc], 1 - np.sum(x[1:nc]))
            dz_dx = np.zeros((nc, n_vars))
            for i in range(nc - 1):
                dz_dx[i, i + 1] = 1.0
                dz_dx[nc - 1, i + 1] = -1.0
            rho_t = np.sum(props.sat[ph] * props.dens_m[ph])
            drho_t_dx = np.sum(props.sat_ders[ph] * props.dens_m[ph], axis=0) + np.sum(
                props.sat[ph] * props.dens_m_ders[ph], axis=0
            )
            dAdt[:nc] = self.poro * self.volume * (dz_dx * rho_t + zc * drho_t_dx)

            # solid energy accumulation: Es = c_r * RV * c_r * rho_m_r * T - T0 [kJ/m3]
            dEs_dx = np.zeros(n_vars)
            dEs_dx[0] = (
                (1 - self.poro) * self.volume * self.dens_rock * self.c_r * props.dTdP
            )
            dEs_dx[1:nc] = (
                (1 - self.poro)
                * self.volume
                * self.dens_rock
                * self.c_r
                * props.dTdzk[: nc - 1]
            )
            dEs_dx[nc] = (
                (1 - self.poro) * self.volume * self.dens_rock * self.c_r * props.dTdX
            )
            dAdt[nc] += dEs_dx

            # fluid energy accumulation: Ef = c_r * PV * sum(sat * rho_m * h) [kJ/m3]
            dEf_dx = (
                np.sum(
                    props.sat_ders[ph] * props.dens_m[ph] * props.enthalpy[ph], axis=0
                )
                + np.sum(
                    props.sat[ph] * props.dens_m_ders[ph] * props.enthalpy[ph], axis=0
                )
                + np.sum(
                    props.sat[ph] * props.dens_m[ph] * props.enthalpy_ders[ph], axis=0
                )
            )
            dEf_dx[0] -= 100
            dEf_dx *= self.poro * self.volume
            dAdt[nc] += dEf_dx

            ## rate -> right-hand side
            b = np.zeros(n_vars)
            if self.energy_source is not None:
                b[-1] = self.energy_source.evaluate(_t)

            try:
                dxdt = np.linalg.solve(dAdt, b)
            except np.linalg.LinAlgError:
                dxdt = np.zeros_like(x)
            return dxdt

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
            result = self.step_radau_analytical(state_prev, dt, t0=t, **kwargs)
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
