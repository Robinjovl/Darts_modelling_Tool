import matplotlib.pyplot as plt
import numpy as np

try:
    from darts.engines import copy_data_to_device
except ImportError:
    pass
from darts.input.input_data import linear_solver_types
from darts.models.darts_model import DartsModel


class PlotLiveDiagrams(DartsModel):
    def __init__(
        self,
        with_live_plots: bool = False,
        live_plots_for_every_newton_iter: bool = False,
    ):
        """
        Initialize the PlotLiveDiagrams class

        Use this class as the super class of your model to plot live diagrams for
        - solver properties (time step size and number of Newton iterations) and profiles of DFM well properties
        - tracking the state of a wellbore or reservoir block on a pressure-enthalpy diagram

        :param with_live_plots: Whether or not to plot live diagrams
        :type with_live_plots: bool
        :param live_plots_for_every_newton_iter: If true, it plots live diagrams for every Newton-Raphson
                                                 iteration. Otherwise, it plots only for every time step.
        """
        super().__init__()

        # For live plotting
        self.with_live_plots = with_live_plots
        self.live_plots_for_every_newton_iter = live_plots_for_every_newton_iter
        self.figs = []
        self.axes = []
        self.lines = []

    def run_timestep(self, dt: float, t: float, verbose: bool = True):
        """
        Method to solve Newton loop for specified timestep

        :param dt: Timestep size [days]
        :type dt: float
        :param t: Current time [days]
        :type t: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        assert dt > 0, "Time step size must be a positive value!"

        max_newt = self.data_ts.newton_max_iter
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        self.timer.node["simulation"].start()

        residual_history = []
        for i in range(max_newt + 1):
            # Update well phase velocities and derivatives if DFM wells are used
            if self.has_dfm_well:
                self.update_dfm_well_vels_and_ders(dt, t, i)

            # assemble Jacobian and residual of reservoir and well blocks
            self.physics.engine.assemble_linear_system(dt)

            # apply RHS flux
            self.apply_rhs_flux(dt, t)

            if self.has_dfm_well:
                self.apply_dfm_well_lateral_heat_flux(dt, t)

            if self.platform == "gpu":
                copy_data_to_device(
                    self.physics.engine.RHS, self.physics.engine.get_RHS_d()
                )

            if not self.has_dfm_well:
                self.physics.engine.newton_residual_last_dt = (
                    self.physics.engine.calc_newton_residual()
                )  # calc norm of residual
            # TODO Function line_search is not updated for the coupled model.
            elif self.has_dfm_well:
                # Method is either 1 or 2
                self.physics.engine.newton_residual_last_dt = (
                    self.physics.engine.calc_coupled_well_reservoir_residual(
                        self.data_ts.coupled_well_res_norm_method
                    )
                )

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                if (
                    abs(max_residual[i] - max_residual[j]) / denom
                    < self.data_ts.newton_tol_stationary
                ):
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            self.physics.engine.well_residual_last_dt = (
                self.physics.engine.calc_well_residual()
            )
            residual_history.append(
                (
                    self.physics.engine.newton_residual_last_dt,  # matrix residual
                    self.physics.engine.well_residual_last_dt,  # well residual
                    1.0,
                )
            )  # Newton update coefficient

            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if (
                self.physics.engine.newton_residual_last_dt < self.data_ts.newton_tol
                and self.physics.engine.well_residual_last_dt
                < self.data_ts.newton_tol * self.data_ts.newton_tol_wel_mult
            ) or self.physics.engine.n_newton_last_dt == max_newt:
                if i > 0:  # min_i_newton
                    break

            # line search
            if (
                self.data_ts.line_search
                and i > 0
                and residual_history[-1][0] > 0.9 * residual_history[-2][0]
            ):
                coef = np.array([0.0, 1.0])
                history = np.array([residual_history[-2], residual_history[-1]])
                residual_history[-1] = self.line_search(dt, t, coef, history, verbose)
                max_residual[i] = residual_history[-1][0]

                # check stationary point after line search
                counter = 0
                for j in range(i):
                    denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                    if (
                        abs(max_residual[i] - max_residual[j]) / denom
                        < self.data_ts.newton_tol_stationary
                    ):
                        counter += 1
                if counter > 2:
                    if verbose:
                        print("Stationary point detected!")
                    break
            else:
                if (
                    type(self.data_ts.linear_type) is linear_solver_types
                ):  # solvers via Python interface
                    if self.data_ts.linear_type in [
                        linear_solver_types.CPU_PETSC_CPR,
                        linear_solver_types.CPU_PETSC_FS,
                    ]:
                        self.petsc_solve_linear_equation()
                    elif self.data_ts.linear_type in [linear_solver_types.CPU_PARDISO]:
                        self.pardiso_solve_linear_equation()
                    else:
                        raise Exception(
                            "Unknown linear solver type", self.data_ts.linear_type
                        )
                else:  # compile-tyme C++ linear solvers
                    self.physics.engine.solve_linear_equation()
                self.timer.node["newton update"].start()
                self.physics.engine.apply_newton_update(dt)
                self.timer.node["newton update"].stop()

                """ Start live plotting """
                # Plot live results for every Newton-Raphson iteration
                if self.with_live_plots and self.live_plots_for_every_newton_iter:
                    self.update_live_plots()
                """ End live plotting """

        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.time.append(t)
        self.n_newton_iters.append(self.physics.engine.n_newton_last_dt)
        self.time_step_size.append(dt)

        """ Start live plotting """
        # Plot live results for every time step
        if self.with_live_plots and not self.live_plots_for_every_newton_iter:
            self.update_live_plots()
        """ End live plotting """

        self.timer.node["simulation"].stop()
        return converged

    def init_live_plots(self):
        """
        Initialize figure/axes/artist
        """
        plt.ion()

        """ Start initializing the figure containing axes for solver properties and profiles of wellbore properties """
        fig0, axes0 = plt.subplots(2, 9, figsize=(22, 7), constrained_layout=True)

        self.figs.append(fig0)
        self.axes.append(axes0)

        ax0 = self.axes[0][0, 0]
        ax1 = self.axes[0][1, 0]
        # Well props
        ax2 = self.axes[0][0, 1]
        ax3 = self.axes[0][0, 2]
        ax4 = self.axes[0][0, 3]
        ax5 = self.axes[0][0, 4]
        ax6 = self.axes[0][0, 5]
        ax7 = self.axes[0][0, 6]
        ax8 = self.axes[0][0, 7]
        ax9 = self.axes[0][0, 8]
        # Reservoir props
        ax10 = self.axes[0][1, 1]
        ax11 = self.axes[0][1, 2]
        ax12 = self.axes[0][1, 3]
        ax13 = self.axes[0][1, 4]
        ax14 = self.axes[0][1, 5]
        ax15 = self.axes[0][1, 6]
        ax16 = self.axes[0][1, 7]
        ax17 = self.axes[0][1, 8]

        # Axes for number of Newton iterations
        (line0,) = ax0.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        # ax0.set_xscale("log")
        ax0.set_xlabel("Time [days]")
        ax0.set_ylabel("Number of Newton iterations [-]")
        ax0.set_title("** Solver props **")

        # Axes for time step size
        (line1,) = ax1.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        # ax1.set_xscale("log")
        ax1.set_xlabel("Time [days]")
        ax1.set_ylabel("Time step size [days]")

        # Axes for wellbore pressure
        (line2,) = ax2.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax2.set_xlabel("Pressure [bar]")
        ax2.set_ylabel("Segment index [-]")
        ax2.set_title("** Pressure **")
        ax2.invert_yaxis()

        # Axes for wellbore temperature
        (line3,) = ax3.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax3.set_xlabel(r"Temperature [$^\circ$C]")
        ax3.set_ylabel("Segment index [-]")
        ax3.set_title("** Temperature **")
        ax3.invert_yaxis()

        # Axes for wellbore gas volume fraction
        (line4,) = ax4.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax4.set_xlabel("Gas volume fraction [-]")
        ax4.set_ylabel("Segment index [-]")
        ax4.set_title("** Gas volume fraction **")
        ax4.invert_yaxis()

        # Axes for wellbore liquid volume fraction
        (line5,) = ax5.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax5.set_xlabel("Liquid volume fraction [-]")
        ax5.set_ylabel("Segment index [-]")
        ax5.set_title("** Liquid volume fraction **")
        ax5.invert_yaxis()

        # Axes for wellbore gas density
        (line6,) = ax6.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax6.set_xlabel(r"Gas density [kg/m$^3$]")
        ax6.set_ylabel("Segment index [-]")
        ax6.set_title("** Gas density **")
        ax6.invert_yaxis()

        # Axes for wellbore liquid density
        (line7,) = ax7.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax7.set_xlabel(r"Liquid density [kg/m$^3$]")
        ax7.set_ylabel("Segment index [-]")
        ax7.set_title("** Liquid density **")
        ax7.invert_yaxis()

        # Axes for wellbore gas viscosity
        (line8,) = ax8.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax8.set_xlabel("Gas viscosity [cP]")
        ax8.set_ylabel("Segment index [-]")
        ax8.set_title("** Gas viscosity **")
        ax8.invert_yaxis()

        # Axes for wellbore liquid viscosity
        (line9,) = ax9.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax9.set_xlabel("Liquid viscosity [cP]")
        ax9.set_ylabel("Segment index [-]")
        ax9.set_title("** Liquid viscosity **")
        ax9.invert_yaxis()

        # Axes for reservoir pressure
        (line10,) = ax10.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax10.set_xscale("log")
        ax10.set_xlabel("Reservoir radial distance [m]")
        ax10.set_ylabel("Pressure [bar]")

        # Axes for reservoir temperature
        (line11,) = ax11.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax11.set_xscale("log")
        ax11.set_xlabel("Reservoir radial distance [m]")
        ax11.set_ylabel("Temperature [$^\circ$C]")

        # Axes for reservoir gas volume fraction
        (line12,) = ax12.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax12.set_xscale("log")
        ax12.set_xlabel("Reservoir radial distance [m]")
        ax12.set_ylabel("Gas volume fraction [-]")

        # Axes for reservoir liquid volume fraction
        (line13,) = ax13.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax13.set_xscale("log")
        ax13.set_xlabel("Reservoir radial distance [m]")
        ax13.set_ylabel("Liquid volume fraction [-]")

        # Axes for reservoir gas density
        (line14,) = ax14.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax14.set_xscale("log")
        ax14.set_xlabel("Reservoir radial distance [m]")
        ax14.set_ylabel(r"Gas density [kg/m$^3$]")

        # Axes for reservoir liquid density
        (line15,) = ax15.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax15.set_xscale("log")
        ax15.set_xlabel("Reservoir radial distance [m]")
        ax15.set_ylabel(r"Liquid density [kg/m$^3$]")

        # Axes for reservoir gas viscosity
        (line16,) = ax16.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax16.set_xscale("log")
        ax16.set_xlabel("Reservoir radial distance [m]")
        ax16.set_ylabel("Gas viscosity [cP]")

        # Axes for reservoir liquid viscosity
        (line17,) = ax17.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
        )

        ax17.set_xscale("log")
        ax17.set_xlabel("Reservoir radial distance [m]")
        ax17.set_ylabel("Liquid viscosity [cP]")

        self.lines.append(
            [
                line0,
                line1,
                line2,
                line3,
                line4,
                line5,
                line6,
                line7,
                line8,
                line9,
                line10,
                line11,
                line12,
                line13,
                line14,
                line15,
                line16,
                line17,
            ]
        )

        self.figs[0].show()
        """ Stop initializing the figure containing axes for solver properties and profiles of wellbore properties """

        """ Start initializing the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """
        fig1, axes1 = plt.subplots(figsize=(10, 6), constrained_layout=True)

        self.figs.append(fig1)
        self.axes.append(axes1)

        ax0 = self.axes[1]

        p_bounds = (self.physics.PT_axes_min[0], self.physics.PT_axes_max[0])
        # t_bounds = (self.physics.PT_axes_min[1], self.physics.PT_axes_max[1])
        h_bounds = (self.physics.axes_min[1], self.physics.axes_max[1])

        # Resolution of the PH diagram
        n_p, n_h = self.physics.n_axes_points[0], self.physics.n_axes_points[1]

        p_range = np.linspace(p_bounds[0], p_bounds[1], n_p)
        h_range = np.linspace(h_bounds[0], h_bounds[1], n_h)

        # Calculate the property matrix
        prop_matrix = np.empty((n_p, n_h))
        for idx_p, p in enumerate(p_range):
            for idx_h, h in enumerate(h_range):
                state_ph = [p, h]
                self.physics.property_containers[0].evaluate(state_ph)
                prop_matrix[idx_p, idx_h] = self.physics.property_containers[
                    0
                ].temperature

        prop_matrix = np.where(
            (prop_matrix == 100) | (prop_matrix == 1000), np.nan, prop_matrix
        )  # for temperature
        # prop_matrix = np.where(prop_matrix == 0, np.nan, prop_matrix)   # for density and gas viscosity
        # prop_matrix = np.where((prop_matrix > 4.0) | (prop_matrix == 0), np.nan, prop_matrix)  # for liquid viscosity
        n_cmap_bins = 50
        levels = np.linspace(
            np.nanmin(prop_matrix), np.nanmax(prop_matrix), n_cmap_bins
        )

        # Filled contour (colored areas)
        cax = ax0.contourf(h_range, p_range, prop_matrix, levels=levels, cmap='jet')

        # Contour lines at the same levels
        contours = ax0.contour(
            h_range, p_range, prop_matrix, levels=levels, colors='black', linewidths=0.5
        )

        # Label each contour line with its property value
        ax0.clabel(contours, fmt='%1.1f', inline=True, fontsize=7)

        ax0.set_xlim(h_bounds[0], h_bounds[1])
        ax0.set_ylim(p_bounds[0], p_bounds[1])
        ax0.set_xlabel('Specific enthalpy [kJ/kmole]')
        ax0.set_ylabel('Pressure [bar]')

        # Add a colorbar
        cbar = fig1.colorbar(cax, ax=ax0)
        cbar.set_label("Temperature [K]")

        # Create a plot for adding bottom-hole points (states) to the PH diagram
        (line0,) = ax0.plot(
            [],
            [],
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=6,
            color='red',
            markerfacecolor='red',
            markeredgecolor='red',
            label='Bottom-hole state',
        )

        self.lines.append(line0)

        self.figs[1].show()

        """ Stop initializing the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """

    def update_live_plots(self):
        """
        Plot properties vs current time
        """
        # Initialize once (first call only)
        if not self.figs or not self.axes or not self.lines:
            self.init_live_plots()

        """ Start updating the figure containing axes for solver properties and profiles of wellbore properties """
        self.lines[0][0].set_data(self.time, self.n_newton_iters)
        self.axes[0][0, 0].relim()
        self.axes[0][0, 0].autoscale_view()

        self.lines[0][1].set_data(self.time, self.time_step_size)
        self.axes[0][1, 0].relim()
        self.axes[0][1, 0].autoscale_view()

        i_start_well = self.reservoir.wells[0].well_head_idx
        i_end_well = self.reservoir.wells[0].well_bottom_idx
        p_idx = self.physics.vars.index('pressure')
        h_idx = self.physics.vars.index('enthalpy')
        X_np = np.asarray(self.physics.engine.X).reshape(-1, self.physics.n_vars)
        p_well = X_np[i_start_well : i_end_well + 1, p_idx]
        h_well = X_np[i_start_well : i_end_well + 1, h_idx]
        # Get the property container to evaluate phase props
        pc = self.physics.property_containers[0]
        n_segments = self.wells['I1'].geometry.num_segments
        n_res_blocks = self.reservoir.mesh.n_res_blocks
        n_vars = self.physics.n_vars
        T_well = np.zeros(n_segments)
        for i in range(n_res_blocks, n_res_blocks + n_segments):
            state = np.asarray(self.physics.engine.X)[i * n_vars : (i + 1) * n_vars]
            pc.evaluate(state)
            if self.physics.thermal:
                pc.evaluate_thermal(state)
            T_well[i - n_res_blocks] = pc.temperature - 273.15

        till_this_res_cell = 50  # Plot till this reservoir cell index
        assert till_this_res_cell <= n_res_blocks
        p_res = X_np[:i_start_well, p_idx][:till_this_res_cell]
        x_res = self.reservoir.global_data['dx'].reshape(-1)[:till_this_res_cell]

        # Calculate reservoir phase props
        nc = self.physics.nc
        # Preallocate phase props arrays
        T_res = np.zeros(till_this_res_cell)
        sG_res = np.zeros(till_this_res_cell)
        rhoG_res = np.zeros(till_this_res_cell)
        rhoL_res = np.zeros(till_this_res_cell)
        miuG_res = np.zeros(till_this_res_cell)
        miuL_res = np.zeros(till_this_res_cell)
        xG_mass_res = np.zeros((till_this_res_cell, nc))
        xL_mass_res = np.zeros((till_this_res_cell, nc))

        for i in range(till_this_res_cell):
            state = np.asarray(self.physics.engine.X)[i * n_vars : (i + 1) * n_vars]
            pc.evaluate(state)
            if self.physics.thermal:
                pc.evaluate_thermal(state)
            T_res[i] = pc.temperature - 273.15
            sG_res[i] = pc.sat[0]
            rhoG_res[i] = pc.dens[0]
            rhoL_res[i] = pc.dens[1]
            miuG_res[i] = pc.mu[0]
            miuL_res[i] = pc.mu[1]
            x_mass0 = np.zeros((pc.nph, nc))
            for j in pc.ph:
                x_mass0[j, :] = (pc.x[j, :] * pc.Mw) / sum(pc.x[j, :] * pc.Mw)
            xG_mass_res[i, :], xL_mass_res[i, :] = x_mass0[0, :], x_mass0[1, :]

        [
            xG_mass_well,
            xL_mass_well,
            sG_well,
            rhoG_well,
            rhoL_well,
            miuG_well,
            miuL_well,
        ] = self.wells['I1'].iter_phases_props

        # Well props
        self.lines[0][2].set_data(p_well, np.arange(n_segments))
        self.axes[0][0, 1].relim()
        self.axes[0][0, 1].autoscale_view()

        self.lines[0][3].set_data(T_well, np.arange(n_segments))
        self.axes[0][0, 2].relim()
        self.axes[0][0, 2].autoscale_view()

        self.lines[0][4].set_data(sG_well, np.arange(n_segments))
        self.axes[0][0, 3].relim()
        self.axes[0][0, 3].autoscale_view()

        self.lines[0][5].set_data(1 - sG_well, np.arange(n_segments))
        self.axes[0][0, 4].relim()
        self.axes[0][0, 4].autoscale_view()

        self.lines[0][6].set_data(rhoG_well, np.arange(n_segments))
        self.axes[0][0, 5].relim()
        self.axes[0][0, 5].autoscale_view()

        self.lines[0][7].set_data(rhoL_well, np.arange(n_segments))
        self.axes[0][0, 6].relim()
        self.axes[0][0, 6].autoscale_view()

        self.lines[0][8].set_data(miuG_well, np.arange(n_segments))
        self.axes[0][0, 7].relim()
        self.axes[0][0, 7].autoscale_view()

        self.lines[0][9].set_data(miuL_well, np.arange(n_segments))
        self.axes[0][0, 8].relim()
        self.axes[0][0, 8].autoscale_view()

        # Reservoir props
        self.lines[0][10].set_data(x_res, p_res)
        self.axes[0][1, 1].relim()
        self.axes[0][1, 1].autoscale_view()

        self.lines[0][11].set_data(x_res, T_res)
        self.axes[0][1, 2].relim()
        self.axes[0][1, 2].autoscale_view()

        self.lines[0][12].set_data(x_res, sG_res)
        self.axes[0][1, 3].relim()
        self.axes[0][1, 3].autoscale_view()

        self.lines[0][13].set_data(x_res, 1 - sG_res)
        self.axes[0][1, 4].relim()
        self.axes[0][1, 4].autoscale_view()

        self.lines[0][14].set_data(x_res, rhoG_res)
        self.axes[0][1, 5].relim()
        self.axes[0][1, 5].autoscale_view()

        self.lines[0][15].set_data(x_res, rhoL_res)
        self.axes[0][1, 6].relim()
        self.axes[0][1, 6].autoscale_view()

        self.lines[0][16].set_data(x_res, miuG_res)
        self.axes[0][1, 7].relim()
        self.axes[0][1, 7].autoscale_view()

        self.lines[0][17].set_data(x_res, miuL_res)
        self.axes[0][1, 8].relim()
        self.axes[0][1, 8].autoscale_view()

        # Refresh display
        self.figs[0].canvas.draw_idle()
        self.figs[0].canvas.flush_events()
        """ Stop updating the figure containing axes for solver properties and profiles of wellbore properties """

        """ Start updating the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """
        # Plot bottom-hole state
        # Get existing bottom-hole data and append them
        x = list(self.lines[1].get_xdata())
        y = list(self.lines[1].get_ydata())
        # p_bottom_hole = X_np[3, 0]
        # enthalpy_bottom_hole = X_np[3, 1]
        p_bottom_hole = p_well[-1]
        enthalpy_bottom_hole = h_well[-1]
        x.append(enthalpy_bottom_hole)
        y.append(p_bottom_hole)

        self.lines[1].set_data(x, y)

        self.axes[1].relim()
        self.axes[1].autoscale_view()

        self.figs[1].canvas.draw_idle()
        self.figs[1].canvas.flush_events()
        """ Stop updating the figure containing a pair of axes for the PH diagram of a property (e.g., temperature) """

        # plt.pause(0.5)
