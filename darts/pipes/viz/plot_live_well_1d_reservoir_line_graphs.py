import matplotlib.pyplot as plt
import numpy as np

from darts.models.darts_model import DartsModel


class DartsModelWithWellLivePlots(DartsModel):
    def __init__(self):
        """
        Initialize the class

        Use this class as the super class of your model to create live plots for profiles of well properties
        """
        super().__init__()

    def init_live_plots(self):
        plt.ion()

        """ Start initializing the figure containing axes for profiles of wellbore properties """
        fig, axes = plt.subplots(2, 8, figsize=(22, 7), constrained_layout=True)

        # Well props
        ax0 = axes[0, 0]
        ax1 = axes[0, 1]
        ax2 = axes[0, 2]
        ax3 = axes[0, 3]
        ax4 = axes[0, 4]
        ax5 = axes[0, 5]
        ax6 = axes[0, 6]
        ax7 = axes[0, 7]
        # Reservoir props
        ax8 = axes[1, 0]
        ax9 = axes[1, 1]
        ax10 = axes[1, 2]
        ax11 = axes[1, 3]
        ax12 = axes[1, 4]
        ax13 = axes[1, 5]
        ax14 = axes[1, 6]
        ax15 = axes[1, 7]

        # Axes for wellbore pressure
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

        ax0.set_xlabel("Pressure [bar]")
        ax0.set_ylabel("Segment index [-]")
        ax0.set_title("** Pressure **")
        ax0.invert_yaxis()

        # Axes for wellbore temperature
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

        ax1.set_xlabel(r"Temperature [$^\circ$C]")
        ax1.set_ylabel("Segment index [-]")
        ax1.set_title("** Temperature **")
        ax1.invert_yaxis()

        # Axes for wellbore gas volume fraction
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

        ax2.set_xlabel("Gas volume fraction [-]")
        ax2.set_ylabel("Segment index [-]")
        ax2.set_title("** Gas volume fraction **")
        ax2.invert_yaxis()

        # Axes for wellbore liquid volume fraction
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

        ax3.set_xlabel("Liquid volume fraction [-]")
        ax3.set_ylabel("Segment index [-]")
        ax3.set_title("** Liquid volume fraction **")
        ax3.invert_yaxis()

        # Axes for wellbore gas density
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

        ax4.set_xlabel(r"Gas density [kg/m$^3$]")
        ax4.set_ylabel("Segment index [-]")
        ax4.set_title("** Gas density **")
        ax4.invert_yaxis()

        # Axes for wellbore liquid density
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

        ax5.set_xlabel(r"Liquid density [kg/m$^3$]")
        ax5.set_ylabel("Segment index [-]")
        ax5.set_title("** Liquid density **")
        ax5.invert_yaxis()

        # Axes for wellbore gas viscosity
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

        ax6.set_xlabel("Gas viscosity [cP]")
        ax6.set_ylabel("Segment index [-]")
        ax6.set_title("** Gas viscosity **")
        ax6.invert_yaxis()

        # Axes for wellbore liquid viscosity
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

        ax7.set_xlabel("Liquid viscosity [cP]")
        ax7.set_ylabel("Segment index [-]")
        ax7.set_title("** Liquid viscosity **")
        ax7.invert_yaxis()

        # Axes for reservoir pressure
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

        ax8.set_xscale("log")
        ax8.set_xlabel("Reservoir radial distance [m]")
        ax8.set_ylabel("Pressure [bar]")

        # Axes for reservoir temperature
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

        ax9.set_xscale("log")
        ax9.set_xlabel("Reservoir radial distance [m]")
        ax9.set_ylabel("Temperature [$^\circ$C]")

        # Axes for reservoir gas volume fraction
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
        ax10.set_ylabel("Gas volume fraction [-]")

        # Axes for reservoir liquid volume fraction
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
        ax11.set_ylabel("Liquid volume fraction [-]")

        # Axes for reservoir gas density
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
        ax12.set_ylabel(r"Gas density [kg/m$^3$]")

        # Axes for reservoir liquid density
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
        ax13.set_ylabel(r"Liquid density [kg/m$^3$]")

        # Axes for reservoir gas viscosity
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
        ax14.set_ylabel("Gas viscosity [cP]")

        # Axes for reservoir liquid viscosity
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
        ax15.set_ylabel("Liquid viscosity [cP]")

        fig.show()

        lines = [
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
        ]

        self.live_fig_store["well_fig"] = {
            "fig": fig,
            "axes": axes,
            "lines": lines,
        }
        """ Stop initializing the figure containing axes for profiles of wellbore properties """

    def update_live_plots(self, time: float, iter_counter: int):
        # Initialize once (first call only)
        if not self.live_fig_store:
            self.init_live_plots()

        """ Start updating the figure containing axes for profiles of wellbore properties """
        fig = self.live_fig_store["well_fig"]["fig"]
        axes = self.live_fig_store["well_fig"]["axes"]
        lines = self.live_fig_store["well_fig"]["lines"]

        i_start_well = self.reservoir.wells[0].well_head_idx
        i_end_well = self.reservoir.wells[0].well_bottom_idx
        p_idx = self.physics.vars.index('pressure')
        h_idx = self.physics.vars.index('enthalpy')
        X_np = np.asarray(self.physics.engine.X).reshape(-1, self.physics.n_vars)
        p_well = X_np[i_start_well : i_end_well + 1, p_idx]
        _h_well = X_np[i_start_well : i_end_well + 1, h_idx]

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
        lines[0].set_data(p_well, np.arange(n_segments))
        axes[0, 0].relim()
        axes[0, 0].autoscale_view()

        lines[1].set_data(T_well, np.arange(n_segments))
        axes[0, 1].relim()
        axes[0, 1].autoscale_view()

        lines[2].set_data(sG_well, np.arange(n_segments))
        axes[0, 2].relim()
        axes[0, 2].autoscale_view()

        lines[3].set_data(1 - sG_well, np.arange(n_segments))
        axes[0, 3].relim()
        axes[0, 3].autoscale_view()

        lines[4].set_data(rhoG_well, np.arange(n_segments))
        axes[0, 4].relim()
        axes[0, 4].autoscale_view()

        lines[5].set_data(rhoL_well, np.arange(n_segments))
        axes[0, 5].relim()
        axes[0, 5].autoscale_view()

        lines[6].set_data(miuG_well, np.arange(n_segments))
        axes[0, 6].relim()
        axes[0, 6].autoscale_view()

        lines[7].set_data(miuL_well, np.arange(n_segments))
        axes[0, 7].relim()
        axes[0, 7].autoscale_view()

        # Reservoir props
        lines[8].set_data(x_res, p_res)
        axes[1, 0].relim()
        axes[1, 0].autoscale_view()

        lines[9].set_data(x_res, T_res)
        axes[1, 1].relim()
        axes[1, 1].autoscale_view()

        lines[10].set_data(x_res, sG_res)
        axes[1, 2].relim()
        axes[1, 2].autoscale_view()

        lines[11].set_data(x_res, 1 - sG_res)
        axes[1, 3].relim()
        axes[1, 3].autoscale_view()

        lines[12].set_data(x_res, rhoG_res)
        axes[1, 4].relim()
        axes[1, 4].autoscale_view()

        lines[13].set_data(x_res, rhoL_res)
        axes[1, 5].relim()
        axes[1, 5].autoscale_view()

        lines[14].set_data(x_res, miuG_res)
        axes[1, 6].relim()
        axes[1, 6].autoscale_view()

        lines[15].set_data(x_res, miuL_res)
        axes[1, 7].relim()
        axes[1, 7].autoscale_view()

        # Update the figure title
        fig.suptitle(
            self.live_plt_config.title_template.format(
                time=time, iter_counter=iter_counter
            )
        )

        # Refresh display
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        """ Stop updating the figure containing axes for profiles of wellbore properties """

        # plt.pause(0.5)
