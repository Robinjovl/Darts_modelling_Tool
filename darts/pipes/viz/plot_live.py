import os
import shutil
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

try:
    from darts.engines import copy_data_to_device
except ImportError:
    pass
from darts.input.input_data import linear_solver_types
from darts.models.darts_model import DartsModel


@dataclass
class LivePlotConfig:
    # Flag to enable live plotting of solver properties (iteration counter and time-step size)
    enable_solver_props: bool = False

    # Flag to enable live plotting of the PH diagram
    enable_ph_diagram: bool = False
    # Index of the block which will be tracked on the PH diagram
    tracked_block_idx: int = 0

    # Flag to enable live plotting of well-reservoir property profiles
    enable_well_res_profiles: bool = False
    # For coupled well-(1D)reservoir, plot reservoir property profile until this reservoir cell
    plot_till_this_res_cell: int = 50

    # Whether to plot for every Newton iteration or not (i.e., for every time step)
    every_newton_iter: bool = False
    # Template of the figure title
    title_template: str = "Time: {time:.4e} \nNR iteration counter: {iter_counter}"

    # "show" opens interactive live plots; "save" saves snapshots without showing interactive live plots.
    output_mode: str = "show"
    # If output_mode is "save", snapshot_folder is the subfolder inside DartsModel.output_folder where the snapshots are saved.
    snapshot_folder: str = "live_plot_snapshots"
    snapshot_format: str = "png"
    snapshot_dpi: int = 150


class DartsModelWithLivePlots(DartsModel):
    def __init__(self):
        """
        Initialize the class

        Use this class as the super class of your model to create live plots for
            - solver properties (number of Newton iterations and time-step size) over time
            - tracking the state of your desired block on the PH diagram
            - profiles of DFM well and 1D reservoir properties

        To enable live plotting, activate appropriate flags available in self.live_plot_config
        """
        super().__init__()

        self.live_plot_config = LivePlotConfig()
        self._live_plot_store = {}
        self._live_plot_snapshot_counter = 0

    def _get_live_plot_output_mode(self) -> str:
        output_mode = self.live_plot_config.output_mode
        if output_mode not in ["show", "save"]:
            raise ValueError('live_plot_config.output_mode must be "show" or "save"')
        return output_mode

    def _show_live_plot_figures(self) -> bool:
        return self._get_live_plot_output_mode() == "show"

    def _save_live_plot_snapshots(self) -> bool:
        return self._get_live_plot_output_mode() == "save"

    def _get_live_plot_snapshot_folder(self) -> str:
        snapshot_folder = self.live_plot_config.snapshot_folder
        if os.path.isabs(snapshot_folder):
            raise ValueError(
                "live_plot_config.snapshot_folder must be relative to self.output_folder"
            )
        if os.path.normpath(snapshot_folder) == ".":
            raise ValueError(
                "live_plot_config.snapshot_folder must name a subfolder inside self.output_folder"
            )

        output_folder = os.path.abspath(self.output_folder)
        snapshot_folder = os.path.abspath(os.path.join(output_folder, snapshot_folder))
        if os.path.commonpath([output_folder, snapshot_folder]) != output_folder:
            raise ValueError(
                "live_plot_config.snapshot_folder must stay inside self.output_folder"
            )

        return snapshot_folder

    def _prepare_live_plot_snapshot_folder(self):
        snapshot_folder = self._get_live_plot_snapshot_folder()
        if os.path.isdir(snapshot_folder):
            for item in os.scandir(snapshot_folder):
                if item.is_dir():
                    shutil.rmtree(item.path)

        os.makedirs(snapshot_folder, exist_ok=True)

    def _init_live_plot_snapshot_dir(self, figure_name: str):
        snapshot_dir = os.path.join(self._get_live_plot_snapshot_folder(), figure_name)
        os.makedirs(snapshot_dir, exist_ok=True)
        return snapshot_dir

    def _save_live_plot_snapshot(
        self,
        fig,
        snapshot_dir: str,
        snapshot_index: int,
    ):
        snapshot_format = self.live_plot_config.snapshot_format.lstrip(".")
        file_name = f"{snapshot_index:06d}.{snapshot_format}"
        fig.savefig(
            os.path.join(snapshot_dir, file_name),
            dpi=self.live_plot_config.snapshot_dpi,
        )

    def init_live_plots(self):
        """
        Initialize live plots
        """
        if self._show_live_plot_figures():
            plt.ion()
        else:
            self._prepare_live_plot_snapshot_folder()

        """ Start initializing the figure containing axes for the properties of the Newton solver """
        if self.live_plot_config.enable_solver_props:
            fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

            ax0 = axes[0]
            ax1 = axes[1]

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
            # Use integers for the labels of the y-axis
            ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

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

            snapshot_dir = None
            if self._show_live_plot_figures():
                fig.show()
            else:
                snapshot_dir = self._init_live_plot_snapshot_dir("solver_props")
            self._live_plot_store["solver_fig"] = {
                "fig": fig,
                "axes": axes,
                "lines": [line0, line1],
                "snapshot_dir": snapshot_dir,
            }
        """ End initializing the figure containing axes for the properties of the Newton solver """

        """ Start initializing the figure containing a pair of axes for the PH diagram """
        # PH diagram is only supported for the PH formulation (at the moment with a single component).
        if (
            self.live_plot_config.enable_ph_diagram
            and self.physics.state_spec == self.physics.StateSpecification.PH
        ):
            if not self.physics.n_vars == 2:
                raise Exception(
                    "Plotting the live PH diagram is supported for the PH formulation and a single component!"
                )

            fig, axes = plt.subplots(figsize=(10, 6), constrained_layout=True)

            p_idx = self.physics.vars.index("pressure")
            h_idx = self.physics.vars.index("enthalpy")

            # Get the bounds of the OBL domain
            p_bounds = (
                self.physics.PT_axes_min[p_idx],
                self.physics.PT_axes_max[p_idx],
            )
            h_bounds = (self.physics.axes_min[h_idx], self.physics.axes_max[h_idx])

            # Resolution of the PH diagram
            n_p, n_h = (
                self.physics.n_axes_points[p_idx],
                self.physics.n_axes_points[h_idx],
            )

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

            n_cmap_bins = 50
            levels = np.linspace(
                np.nanmin(prop_matrix), np.nanmax(prop_matrix), n_cmap_bins
            )

            # Filled contour (colored areas)
            cax = axes.contourf(
                h_range, p_range, prop_matrix, levels=levels, cmap='jet'
            )

            # Contour lines at the same levels
            contours = axes.contour(
                h_range,
                p_range,
                prop_matrix,
                levels=levels,
                colors='black',
                linewidths=0.5,
            )

            # Label each contour line with its property value
            axes.clabel(contours, fmt='%1.1f', inline=True, fontsize=7)

            axes.set_xlim(h_bounds[0], h_bounds[1])
            axes.set_ylim(p_bounds[0], p_bounds[1])
            axes.set_xlabel('Specific enthalpy [kJ/kmole]')
            axes.set_ylabel('Pressure [bar]')

            # Add a colorbar
            cbar = fig.colorbar(cax, ax=axes)
            cbar.set_label("Temperature [K]")

            # Create a plot for adding bottom-hole points (states) to the PH diagram
            (line,) = axes.plot(
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

            snapshot_dir = None
            if self._show_live_plot_figures():
                fig.show()
            else:
                snapshot_dir = self._init_live_plot_snapshot_dir("ph_diagram")
            # Update the figure store
            self._live_plot_store["ph_fig"] = {
                "fig": fig,
                "axes": axes,
                "lines": [line],
                "snapshot_dir": snapshot_dir,
            }
        """ Stop initializing the figure containing a pair of axes for the PH diagram """

        """ Start initializing the figure containing axes for profiles of wellbore and 1D reservoir properties """
        if self.live_plot_config.enable_well_res_profiles and self.has_dfm_well:
            fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)

            pc = self.physics.property_containers[0]
            phase_names = list(pc.phases_name)
            mobile_phase_names = phase_names[: pc.np_fl]

            def make_profile_line(ax, *, label=None, color='red'):
                (line,) = ax.plot(
                    [],
                    [],
                    linestyle='-',
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=label,
                )
                return line

            # Well props
            ax0 = axes[0, 0]
            ax1 = axes[0, 1]
            ax2 = axes[0, 2]
            ax3 = axes[0, 3]
            ax4 = axes[0, 4]
            # Reservoir props
            ax5 = axes[1, 0]
            ax6 = axes[1, 1]
            ax7 = axes[1, 2]
            ax8 = axes[1, 3]
            ax9 = axes[1, 4]

            # Axes for wellbore pressure
            line0 = make_profile_line(ax0)

            ax0.set_xlabel("Pressure [bar]")
            ax0.set_ylabel("Segment index [-]")
            ax0.set_title("** Pressure **")
            ax0.invert_yaxis()

            # Axes for wellbore temperature
            line1 = make_profile_line(ax1)

            ax1.set_xlabel(r"Temperature [$^\circ$C]")
            ax1.set_ylabel("Segment index [-]")
            ax1.set_title("** Temperature **")
            ax1.invert_yaxis()

            # Axes for wellbore phase saturations
            well_sat_lines = [
                make_profile_line(ax2, label=phase_name, color=f"C{j}")
                for j, phase_name in enumerate(phase_names)
            ]

            ax2.set_xlabel("Phase saturation [-]")
            ax2.set_ylabel("Segment index [-]")
            ax2.set_title("** Phase saturation **")
            ax2.set_xlim(0.0, 1.0)
            ax2.invert_yaxis()
            ax2.legend(loc="best", fontsize=8)

            # Axes for wellbore phase densities
            well_density_lines = [
                make_profile_line(ax3, label=phase_name, color=f"C{j}")
                for j, phase_name in enumerate(phase_names)
            ]

            ax3.set_xlabel(r"Phase density [kg/m$^3$]")
            ax3.set_ylabel("Segment index [-]")
            ax3.set_title("** Phase density **")
            ax3.invert_yaxis()
            ax3.legend(loc="best", fontsize=8)

            # Axes for wellbore phase viscosities.  Solids are excluded because
            # PropertyContainer stores viscosity only for mobile fluid phases.
            well_viscosity_lines = [
                make_profile_line(ax4, label=phase_name, color=f"C{j}")
                for j, phase_name in enumerate(mobile_phase_names)
            ]

            ax4.set_xlabel("Phase viscosity [cP]")
            ax4.set_ylabel("Segment index [-]")
            ax4.set_title("** Phase viscosity **")
            ax4.invert_yaxis()
            ax4.legend(loc="best", fontsize=8)

            # Axes for reservoir pressure
            line2 = make_profile_line(ax5)

            ax5.set_xscale("log")
            ax5.set_xlabel("Reservoir radial distance [m]")
            ax5.set_ylabel("Pressure [bar]")

            # Axes for reservoir temperature
            line3 = make_profile_line(ax6)

            ax6.set_xscale("log")
            ax6.set_xlabel("Reservoir radial distance [m]")
            ax6.set_ylabel("Temperature [$^\circ$C]")

            # Axes for reservoir phase saturations
            res_sat_lines = [
                make_profile_line(ax7, label=phase_name, color=f"C{j}")
                for j, phase_name in enumerate(phase_names)
            ]

            ax7.set_xscale("log")
            ax7.set_xlabel("Reservoir radial distance [m]")
            ax7.set_ylabel("Phase saturation [-]")
            ax7.set_ylim(0.0, 1.0)
            ax7.legend(loc="best", fontsize=8)

            # Axes for reservoir phase densities
            res_density_lines = [
                make_profile_line(ax8, label=phase_name, color=f"C{j}")
                for j, phase_name in enumerate(phase_names)
            ]

            ax8.set_xscale("log")
            ax8.set_xlabel("Reservoir radial distance [m]")
            ax8.set_ylabel(r"Phase density [kg/m$^3$]")
            ax8.legend(loc="best", fontsize=8)

            # Axes for reservoir phase viscosities.  Solids are excluded because
            # PropertyContainer stores viscosity only for mobile fluid phases.
            res_viscosity_lines = [
                make_profile_line(ax9, label=phase_name, color=f"C{j}")
                for j, phase_name in enumerate(mobile_phase_names)
            ]

            ax9.set_xscale("log")
            ax9.set_xlabel("Reservoir radial distance [m]")
            ax9.set_ylabel("Phase viscosity [cP]")
            ax9.legend(loc="best", fontsize=8)

            snapshot_dir = None
            if self._show_live_plot_figures():
                fig.show()
            else:
                snapshot_dir = self._init_live_plot_snapshot_dir("well_res_profiles")
            lines = {
                "well_pressure": line0,
                "well_temperature": line1,
                "reservoir_pressure": line2,
                "reservoir_temperature": line3,
            }

            self._live_plot_store["well_fig"] = {
                "fig": fig,
                "axes": axes,
                "lines": lines,
                "phase_names": phase_names,
                "mobile_phase_names": mobile_phase_names,
                "well_sat_lines": well_sat_lines,
                "res_sat_lines": res_sat_lines,
                "well_density_lines": well_density_lines,
                "res_density_lines": res_density_lines,
                "well_viscosity_lines": well_viscosity_lines,
                "res_viscosity_lines": res_viscosity_lines,
                "snapshot_dir": snapshot_dir,
            }
        """ Stop initializing the figure containing axes for profiles of wellbore and 1D reservoir properties """

    def update_live_plots(
        self,
        time: float,
        iter_counter: int,
        time_step_size: float,
    ):
        """
        Initialize (only for the first call) and update live plots

        :param time: Current time [day]
        :type time: float
        :param iter_counter: Newton-Raphson iteration counter [-]
        :type iter_counter: int
        :param time_step_size: Size of the current time step [day]
        :type time_step_size: float
        """
        # Initialize once (first call only)
        if not self._live_plot_store:
            self.init_live_plots()

        self._live_plot_snapshot_counter += 1
        snapshot_index = self._live_plot_snapshot_counter

        """ Start updating the figure containing axes for the properties of the Newton solver """
        if self.live_plot_config.enable_solver_props:
            fig = self._live_plot_store["solver_fig"]["fig"]
            axes = self._live_plot_store["solver_fig"]["axes"]
            lines = self._live_plot_store["solver_fig"]["lines"]

            # Update iteration counter vs time
            x = list(lines[0].get_xdata())
            y = list(lines[0].get_ydata())
            x.append(time)
            y.append(iter_counter)

            lines[0].set_data(x, y)
            axes[0].relim()
            axes[0].autoscale_view()

            # Update time-step size vs time
            x = list(lines[1].get_xdata())
            y = list(lines[1].get_ydata())
            x.append(time)
            y.append(time_step_size)

            lines[1].set_data(x, y)
            axes[1].relim()
            axes[1].autoscale_view()

            if self._show_live_plot_figures():
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                snapshot_dir = self._live_plot_store["solver_fig"]["snapshot_dir"]
                self._save_live_plot_snapshot(fig, snapshot_dir, snapshot_index)
        """ End updating the figure containing axes for the properties of the Newton solver """

        """ Start updating the figure containing a pair of axes for the PH diagram """
        if (
            self.live_plot_config.enable_ph_diagram
            and self.physics.state_spec == self.physics.StateSpecification.PH
        ):
            fig = self._live_plot_store["ph_fig"]["fig"]
            axes = self._live_plot_store["ph_fig"]["axes"]
            lines = self._live_plot_store["ph_fig"]["lines"]

            # Update state of the desired block on the PH diagram
            assert (
                0
                <= self.live_plot_config.tracked_block_idx
                < self.reservoir.mesh.n_blocks
            ), "The specified tracked_block_idx is out of range!"
            block_idx = self.live_plot_config.tracked_block_idx
            X_np = np.asarray(self.physics.engine.X).reshape(-1, self.physics.n_vars)
            p_idx = self.physics.vars.index("pressure")
            h_idx = self.physics.vars.index("enthalpy")
            p = X_np[block_idx, p_idx]
            h = X_np[block_idx, h_idx]

            x = list(lines[0].get_xdata())
            y = list(lines[0].get_ydata())
            x.append(h)
            y.append(p)

            lines[0].set_data(x, y)

            axes.relim()
            axes.autoscale_view()

            # Update the figure title
            fig.suptitle(
                self.live_plot_config.title_template.format(
                    time=time, iter_counter=iter_counter
                )
            )

            if self._show_live_plot_figures():
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                snapshot_dir = self._live_plot_store["ph_fig"]["snapshot_dir"]
                self._save_live_plot_snapshot(fig, snapshot_dir, snapshot_index)
        """ Stop updating the figure containing a pair of axes for the PH diagram """

        """ Start updating the figure containing axes for profiles of wellbore and 1D reservoir properties """
        if self.live_plot_config.enable_well_res_profiles and self.has_dfm_well:
            fig = self._live_plot_store["well_fig"]["fig"]
            axes = self._live_plot_store["well_fig"]["axes"]
            lines = self._live_plot_store["well_fig"]["lines"]
            phase_names = self._live_plot_store["well_fig"]["phase_names"]
            mobile_phase_names = self._live_plot_store["well_fig"]["mobile_phase_names"]
            well_sat_lines = self._live_plot_store["well_fig"]["well_sat_lines"]
            res_sat_lines = self._live_plot_store["well_fig"]["res_sat_lines"]
            well_density_lines = self._live_plot_store["well_fig"]["well_density_lines"]
            res_density_lines = self._live_plot_store["well_fig"]["res_density_lines"]
            well_viscosity_lines = self._live_plot_store["well_fig"][
                "well_viscosity_lines"
            ]
            res_viscosity_lines = self._live_plot_store["well_fig"][
                "res_viscosity_lines"
            ]

            i_start_well = self.reservoir.wells[0].well_head_idx
            i_end_well = self.reservoir.wells[0].well_bottom_idx
            p_idx = self.physics.vars.index('pressure')
            # h_idx = self.physics.vars.index('enthalpy')
            X_np = np.asarray(self.physics.engine.X).reshape(-1, self.physics.n_vars)
            p_well = X_np[i_start_well : i_end_well + 1, p_idx]
            # h_well = X_np[i_start_well : i_end_well + 1, h_idx]

            # Get the property container to evaluate phase props
            pc = self.physics.property_containers[0]
            n_segments = self.wells['I1'].geometry.num_segments
            n_res_blocks = self.reservoir.mesh.n_res_blocks
            n_vars = self.physics.n_vars
            T_well = np.zeros(n_segments)
            sat_well = np.zeros((n_segments, len(phase_names)))
            dens_well = np.zeros((n_segments, len(phase_names)))
            mu_well = np.zeros((n_segments, len(mobile_phase_names)))
            for i in range(n_res_blocks, n_res_blocks + n_segments):
                state = np.asarray(self.physics.engine.X)[i * n_vars : (i + 1) * n_vars]
                pc.evaluate(state)
                if self.physics.thermal:
                    pc.evaluate_thermal(state)
                T_well[i - n_res_blocks] = pc.temperature - 273.15
                sat_well[i - n_res_blocks, :] = pc.sat[: len(phase_names)]
                dens_well[i - n_res_blocks, :] = pc.dens[: len(phase_names)]
                mu_well[i - n_res_blocks, :] = pc.mu[: len(mobile_phase_names)]

            # Plot till this reservoir cell index
            till_this_res_cell = self.live_plot_config.plot_till_this_res_cell
            assert till_this_res_cell <= n_res_blocks
            p_res = X_np[:i_start_well, p_idx][:till_this_res_cell]
            x_res = np.cumsum(
                self.reservoir.global_data['dx'].reshape(-1)[:till_this_res_cell]
            )

            # Preallocate phase props arrays
            T_res = np.zeros(till_this_res_cell)
            sat_res = np.zeros((till_this_res_cell, len(phase_names)))
            dens_res = np.zeros((till_this_res_cell, len(phase_names)))
            mu_res = np.zeros((till_this_res_cell, len(mobile_phase_names)))

            for i in range(till_this_res_cell):
                state = np.asarray(self.physics.engine.X)[i * n_vars : (i + 1) * n_vars]
                pc.evaluate(state)
                if self.physics.thermal:
                    pc.evaluate_thermal(state)
                T_res[i] = pc.temperature - 273.15
                sat_res[i, :] = pc.sat[: len(phase_names)]
                dens_res[i, :] = pc.dens[: len(phase_names)]
                mu_res[i, :] = pc.mu[: len(mobile_phase_names)]

            # Well props
            lines["well_pressure"].set_data(p_well, np.arange(n_segments))
            axes[0, 0].relim()
            axes[0, 0].autoscale_view()

            lines["well_temperature"].set_data(T_well, np.arange(n_segments))
            axes[0, 1].relim()
            axes[0, 1].autoscale_view()

            segment_idxs = np.arange(n_segments)
            for j, line in enumerate(well_sat_lines):
                line.set_data(sat_well[:, j], segment_idxs)
            axes[0, 2].relim()
            axes[0, 2].autoscale_view(scalex=False, scaley=True)
            axes[0, 2].set_xlim(0.0, 1.0)

            for j, line in enumerate(well_density_lines):
                line.set_data(dens_well[:, j], segment_idxs)
            axes[0, 3].relim()
            axes[0, 3].autoscale_view()

            for j, line in enumerate(well_viscosity_lines):
                line.set_data(mu_well[:, j], segment_idxs)
            axes[0, 4].relim()
            axes[0, 4].autoscale_view()

            # Reservoir props
            lines["reservoir_pressure"].set_data(x_res, p_res)
            axes[1, 0].relim()
            axes[1, 0].autoscale_view()

            lines["reservoir_temperature"].set_data(x_res, T_res)
            axes[1, 1].relim()
            axes[1, 1].autoscale_view()

            for j, line in enumerate(res_sat_lines):
                line.set_data(x_res, sat_res[:, j])
            axes[1, 2].relim()
            axes[1, 2].autoscale_view(scalex=True, scaley=False)
            axes[1, 2].set_ylim(0.0, 1.0)

            for j, line in enumerate(res_density_lines):
                line.set_data(x_res, dens_res[:, j])
            axes[1, 3].relim()
            axes[1, 3].autoscale_view()

            for j, line in enumerate(res_viscosity_lines):
                line.set_data(x_res, mu_res[:, j])
            axes[1, 4].relim()
            axes[1, 4].autoscale_view()

            # Update the figure title
            fig.suptitle(
                self.live_plot_config.title_template.format(
                    time=time, iter_counter=iter_counter
                )
            )

            if self._show_live_plot_figures():
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                snapshot_dir = self._live_plot_store["well_fig"]["snapshot_dir"]
                self._save_live_plot_snapshot(fig, snapshot_dir, snapshot_index)
        """ Stop updating the figure containing axes for profiles of wellbore and 1D reservoir properties """

        # plt.pause(0.5)

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
                if isinstance(self.data_ts.linear_type, linear_solver_types):
                    # solvers via Python interface
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
                else:
                    # compile-tyme C++ linear solvers
                    self.physics.engine.solve_linear_equation()
                self.timer.node["newton update"].start()
                self.physics.engine.apply_newton_update(dt)
                self.timer.node["newton update"].stop()

            """ Start live plotting for every Newton-Raphson iteration """
            if (
                self.live_plot_config.enable_solver_props
                or self.live_plot_config.enable_ph_diagram
                or self.live_plot_config.enable_well_res_profiles
            ) and self.live_plot_config.every_newton_iter:
                self.update_live_plots(t, i, dt)
            """ End live plotting for every Newton-Raphson iteration """

        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.time.append(t)
        self.n_newton_iters.append(self.physics.engine.n_newton_last_dt)
        self.time_step_size.append(dt)

        """ Start live plotting for every time step """
        if (
            self.live_plot_config.enable_solver_props
            or self.live_plot_config.enable_ph_diagram
            or self.live_plot_config.enable_well_res_profiles
        ) and not self.live_plot_config.every_newton_iter:
            self.update_live_plots(t, i, dt)
        """ End live plotting for every time step """

        self.timer.node["simulation"].stop()
        return converged
